#!/usr/bin/env python3
"""Stage the four Iowa7T portable HCP151 templates from explicit live inputs."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import warnings
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.io import savemat

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "scripts/core_workflow"
sys.path.insert(0, str(CORE))
sys.path.insert(0, str(ROOT / "scripts/visualize/lib"))
from lib.fc_math import corrcoef_rows, fisher_z_transform_fc
from lib.formal_cohorts import load_formal_subjects
from lib.kong2022_label_utils import (
    aggregate_kong2022_network_timeseries,
    canonical_families_7_order,
    load_kong2022_merge_mapping,
)
from cortex_group_contrasts import subject_label_fc

ROWS = (("2mm", "L"), ("2mm", "R"), ("4mm", "L"), ("4mm", "R"))
SELECTED = {
    "7net": ("Default", "VentralAttention", "Somatomotor"),
    "8net": ("Default", "VentralAttention", "Auditory", "Somatomotor"),
    "17net": ("DefaultA", "SalVenAttnA", "DefaultC", "SomMotB"),
}
DISPLAY = {
    "7net": ("DMN", "VAN", "SMN"),
    "8net": ("DMN", "VAN", "Aud", "SMN"),
    "17net": SELECTED["17net"],
}
COHORT = ROOT / "outputs_hcp/inputs/hcp/qc/vertex_qc_t1only/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_151.txt"
FIXED = ROOT / "outputs_hcp/templates/hcp151_routea_full_k3_fisherz"
BANK = ROOT / "outputs_hcp/work/group_atlas/kong2022_7net_full_k3_fisherz/wta-top/kong2022"
SURFACE = ROOT / "outputs_hcp/inputs/hcp/kong2022/shared_inputs/surface"
REFERENCE = ROOT / "outputs_hcp/inputs/hcp/kong2022/shared_inputs/routea"


@lru_cache(maxsize=1)
def atlas_module():
    name = "lss_templates_group_atlas"
    spec = importlib.util.spec_from_file_location(name, CORE / "05_02_run_group_atlas.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_source_manifest(path: Path, subjects: list[str]) -> dict:
    data = json.loads(path.read_text())
    recorded = [f"sub-{str(s).removeprefix('sub-')}" for s in data.get("subjects", [])]
    if (data.get("status") not in ("ok", "complete") or
            data.get("n_subjects") != len(subjects) or recorded != subjects):
        raise ValueError(f"HCP151 source cohort/status mismatch: {path}")
    return data


def selected_labels(rows: list[dict], n_vertices: int, names: tuple[str, ...]) -> np.ndarray:
    lookup = {network: i + 1 for i, network in enumerate(names)}
    labels = np.zeros(n_vertices, dtype=np.int32)
    for row in rows:
        labels[int(row["vertex_index"])] = lookup.get(str(row["top1_network"]), 0)
    return labels


def direct_labels(scores: np.ndarray, indices: np.ndarray, n_vertices: int,
                  network_names: list[str]) -> np.ndarray:
    rows = atlas_module().build_fisherz_vertex_assignment_rows(
        mean_profile=scores, support_indices=indices,
        network_names=network_names, network_column="canonical7",
    )
    return selected_labels(rows, n_vertices, SELECTED["7net"])


def subject_profile(ts: np.ndarray, roi: np.ndarray, valid: np.ndarray,
                    labels: np.ndarray, n_clusters: int) -> np.ndarray:
    """Fisher-z of each subject's cluster mean TS, retaining empty rows as NaN."""
    masks = [labels == i + 1 for i in range(n_clusters)]
    return subject_label_fc(ts, roi, masks, valid, "HCP151 LSS template export")


def finite_mean(values: np.ndarray, axis: int = 0) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        return np.nanmean(values, axis=axis).astype(np.float32)


def subject_paths(subject: str, smoothing: str, hemi: str) -> tuple[Path, ...]:
    sub = f"sub-{subject.removeprefix('sub-')}"
    surf = SURFACE / sub / "surface"
    ref = REFERENCE / sub / "reference/kong2022"
    return (surf / smoothing / f"{sub}_hemi-{hemi}_timeseries.npy",
            ref / "cortex_roi_parcel_timeseries.npy",
            ref / "cortex_roi_parcels.tsv",
            surf / "tsnr_gate" / f"{sub}_hemi-{hemi}_valid_mask.npy")


def subject_data(subject: str, smoothing: str, hemi: str,
                 registry: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ts_path, roi_path, reg_path, mask_path = subject_paths(subject, smoothing, hemi)
    with reg_path.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if (len(rows) != 400 or [int(r["parcel_id"]) for r in rows] != list(range(1, 401)) or
            any(row["parcel_name"] != target["roi_name"] or
                row["family17"] != target["family17"] for row, target in zip(rows, registry))):
        raise ValueError(f"Subject ROI order/identity differs from template registry: {reg_path}")
    ts = np.load(ts_path, allow_pickle=False).astype(np.float32, copy=False)
    roi = np.load(roi_path, allow_pickle=False).astype(np.float32, copy=False)
    valid = np.load(mask_path, allow_pickle=False).astype(bool)
    if ts.shape[0] != 512 or roi.shape[0] != 400 or valid.shape != (512,):
        raise ValueError(f"Expected consumed 512-vertex / 400-ROI inputs: {ts_path}, {roi_path}")
    if ts.ndim != 2 or roi.ndim != 2 or ts.shape[1] != roi.shape[1] or ts.shape[1] < 2:
        raise ValueError(f"Hippocampus/ROI frames differ: {ts_path}, {roi_path}")
    return ts, roi, valid


def write_profile_files(folder: Path, profiles: np.ndarray, names: tuple[str, ...],
                        display: tuple[str, ...], *, gradient: bool) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    payload = {"profiles": profiles.astype(np.float32),
               "row_keys": np.asarray([f"{s}_{h}" for s, h in ROWS]),
               "value_unit": np.asarray("fisher_z")}
    if gradient:
        payload["cluster_names"] = np.asarray(names)
    else:
        payload["network_names"] = np.asarray(names)
        payload["display_network_names"] = np.asarray(display)
    np.savez_compressed(folder / "template_profiles.npz", **payload)
    savemat(folder / "template_profiles.mat", payload, do_compression=True)


def render_overview(path: Path, profiles: np.ndarray, names: tuple[str, ...], title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    scale = float(np.nanmax(np.abs(profiles)))
    fig, axes = plt.subplots(4, len(names), figsize=(5.2 * len(names), 8.8),
                             squeeze=False, layout="constrained")
    for r, (smoothing, hemi) in enumerate(ROWS):
        for c, name in enumerate(names):
            img = axes[r, c].imshow(profiles[r, c][None, :], cmap="RdBu_r",
                                     vmin=-scale, vmax=scale, aspect="auto",
                                     extent=(0.5, 400.5, 0, 1), interpolation="nearest")
            axes[r, c].set_yticks([])
            axes[r, c].set_xticks([1, 100, 200, 300, 400])
            if r == 0:
                axes[r, c].set_title(name, weight="bold")
            if c == 0:
                axes[r, c].set_ylabel(f"{smoothing} {hemi}", weight="bold")
            if r == 3:
                axes[r, c].set_xlabel("Kong2022 ROI ID")
    fig.suptitle(f"HCP 7T (N=151) | {title}", weight="bold", fontsize=17)
    fig.colorbar(img, ax=axes, location="right", shrink=0.75, label="Mean Fisher-z")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def export(out_root: Path) -> None:
    subjects = load_formal_subjects(COHORT, "hcp151")
    template = read_source_manifest(FIXED / "template_manifest.json", subjects)
    combos = {(row["smoothing"], row["hemi"]): row for row in template["combinations"]}
    if len(template["combinations"]) != 4 or set(combos) != set(ROWS):
        raise ValueError(f"Expected four unique smoothing/hemisphere combinations: {FIXED}")
    sources = {}
    registry = None
    registry_bytes = None
    for smoothing, hemi in ROWS:
        run = BANK / smoothing / "cov_0p80" / f"hemi_{hemi}"
        manifest = read_source_manifest(run / "group_atlas_manifest.json", subjects)
        required = {"vertex_roi_profile": "vertex_roi_profile.npy",
                    "vertex_support_indices": "vertex_support_indices.npy",
                    "group_support_mask": "group_support_mask.npy",
                    "cluster_roi_registry": "cluster_roi_registry.tsv"}
        for key, filename in required.items():
            if Path(manifest[key]).resolve() != (run / filename).resolve():
                raise ValueError(f"Exact {key} declaration mismatch: {run / 'group_atlas_manifest.json'}")
        with (run / "cluster_roi_registry.tsv").open() as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        if len(rows) != 400 or [int(row["roi_id"]) for row in rows] != list(range(1, 401)):
            raise ValueError(f"Expected ROI IDs 1..400: {run}")
        if registry is None:
            registry = rows
            registry_bytes = (run / "cluster_roi_registry.tsv").read_bytes()
        elif rows != registry:
            raise ValueError(f"ROI registries differ across smoothing/hemisphere combinations: {run}")
        support = np.load(run / "group_support_mask.npy", allow_pickle=False).astype(bool)
        indices = np.load(run / "vertex_support_indices.npy", allow_pickle=False)
        fc = np.load(run / "vertex_roi_profile.npy", allow_pickle=False)
        if (support.shape != (512,) or not np.array_equal(indices, np.flatnonzero(support)) or
                fc.shape != (indices.size, 400)):
            raise ValueError(f"Consumed WTA FC/support dimensions disagree: {run}")
        sources[(smoothing, hemi)] = (fc, indices, support)
        for subject in subjects:
            for path in subject_paths(subject, smoothing, hemi):
                if not path.is_file():
                    raise FileNotFoundError(f"Required HCP151 input for {smoothing} hemi-{hemi}: {path}")
    mapping7 = load_kong2022_merge_mapping(taxonomy="canonical7")
    for row in registry:
        if row["canonical7"] != mapping7[row["family17"]]:
            raise ValueError("Current ROI registry differs from the maintained canonical7 mapping")

    # All exact source declarations are checked before creating the candidate output.
    out_root.mkdir(parents=True, exist_ok=False)
    results = {key: [] for key in SELECTED}
    counts = {key: {} for key in SELECTED}
    gradients = []
    for smoothing, hemi in ROWS:
        key = f"{smoothing}_{hemi}"
        print(f"[start] {key}: all 151 subjects, native threads=1", flush=True)
        fc, indices, support = sources[(smoothing, hemi)]
        labels = {}
        for variant, column in (("8net", "canonical8"), ("17net", "family17")):
            assigned = atlas_module().build_normalized_topn_count_vertex_rows(
                mean_profile=fc, support_indices=indices, roi_registry=registry,
                top_n_cap=20, network_column=column,
            )
            labels[variant] = selected_labels(assigned, 512, SELECTED[variant])
        profiles = {"8net": [], "17net": []}
        direct_fc = []
        for number, subject in enumerate(subjects, 1):
            ts, roi, valid = subject_data(subject, smoothing, hemi, registry)
            networks = aggregate_kong2022_network_timeseries(
                roi, registry, canonical_families_7_order(), network_column="canonical7",
            )
            network_fc = fisher_z_transform_fc(corrcoef_rows(ts, networks))
            network_fc[~valid] = np.nan
            direct_fc.append(network_fc)
            for variant in profiles:
                profiles[variant].append(subject_profile(
                    ts, roi, valid & support, labels[variant], len(SELECTED[variant]),
                ))
            if number % 30 == 0:
                print(f"[subjects] {key}: {number}/151", flush=True)
        mean_network_fc = finite_mean(np.stack(direct_fc))
        labels["7net"] = direct_labels(mean_network_fc[indices], indices, 512,
                                         canonical_families_7_order())
        direct_profiles = []
        for subject in subjects:
            ts, roi, valid = subject_data(subject, smoothing, hemi, registry)
            direct_profiles.append(subject_profile(ts, roi, valid & support, labels["7net"], 3))
        results["7net"].append(finite_mean(np.stack(direct_profiles)))
        for variant in profiles:
            results[variant].append(finite_mean(np.stack(profiles[variant])))
        for variant in labels:
            counts[variant][key] = {name: int(np.sum(labels[variant] == i + 1))
                                    for i, name in enumerate(SELECTED[variant])}
        prototype_path = FIXED / f"hcp151_{smoothing}_hemi-{hemi}_prototypes.npy"
        declared = Path(combos[(smoothing, hemi)]["outputs"]["prototypes"]["path"])
        if declared.resolve() != prototype_path.resolve():
            raise ValueError(f"Exact gradient prototype declaration mismatch: {declared}")
        prototype = np.load(prototype_path, allow_pickle=False)
        if prototype.shape != (3, 400):
            raise ValueError(f"Expected 3 x 400 gradient prototypes: {prototype_path}")
        gradients.append(prototype)
        print(f"[done] {key}: {counts['7net'][key]}", flush=True)

    packages = [("gradient_spectral", np.stack(gradients),
                 ("Anterior", "Body", "Posterior"), ("Anterior", "Body", "Posterior"))]
    packages.extend((f"wta_normalized/{v}", np.stack(results[v]), SELECTED[v], DISPLAY[v])
                    for v in ("7net", "8net", "17net"))
    generated = datetime.now(timezone.utc).isoformat()
    entries = []
    for package, values, names, display in packages:
        gradient = package == "gradient_spectral"
        target = out_root / package
        write_profile_files(target / "data", values, names, display, gradient=gradient)
        (target / "data/kong2022_400roi_registry.tsv").write_bytes(registry_bytes)
        variant = package.split('/')[-1]
        title = "Gradient-spectral AP prototypes" if gradient else (
            f"{variant} WTA | " +
            ("direct Fisher-z" if variant == "7net" else "normalized top20"))
        render_overview(target / "figures/overview.png", values, display, title)
        entry = {"id": package, "status": "ok", "n_subjects": 151,
                 "method": "gradient-spectral" if gradient else "wta-top",
                 "profile_shape": list(values.shape),
                 "row_order": [f"{s}_{h}" for s, h in ROWS],
                 "profiles": "data/template_profiles.npz", "mat_profiles": "data/template_profiles.mat",
                 "roi_registry": "data/kong2022_400roi_registry.tsv", "overview": "figures/overview.png",
                 "network_taxonomy": "canonical7" if gradient else
                    {"7net": "canonical7", "8net": "canonical8", "17net": "family17"}[package.split('/')[-1]],
                 "source_root": str(FIXED if gradient else BANK),
                 "source_manifests": [str(FIXED / "template_manifest.json")] if gradient else [
                     str(BANK / s / "cov_0p80" / f"hemi_{h}/group_atlas_manifest.json") for s, h in ROWS],
                 "generated_at_utc": generated}
        if gradient:
            entry.update(cluster_names=list(names), chosen_k_final=3,
                         profile_definition="finite mean group Fisher-z vertex-to-ROI FC in supported AP-ordered gradient clusters")
        else:
            variant = package.split('/')[-1]
            entry.update(network_names=list(names), display_network_names=list(display),
                         wta_assignment_variant="fisherz" if variant == "7net" else "normalized_top20_count",
                         wta_assignment_metric="top1_fisher_z" if variant == "7net" else "top20_count_normalized_by_network_roi_count",
                         selected_vertex_counts=counts[variant], surface_root=str(SURFACE),
                         reference_root=str(REFERENCE), subjects_manifest=str(COHORT),
                         profile_definition="equal-subject mean Fisher-z correlation of valid pseudo-cluster mean time series with each ROI time series",
                         assignment_definition="mean subject vertex-to-canonical7-network Fisher-z FC; network time series are equal-ROI means" if variant == "7net" else
                            "top20 positive group vertex-to-ROI FC counts divided by finite ROI opportunity in the declared taxonomy")
        entries.append(entry)
        if gradient or package == "wta_normalized/7net":
            (target / "provenance.json").write_text(json.dumps(entry, indent=2) + "\n")
    catalog = {"dataset": "HCP 7T", "atlas": "Kong2022 400 ROIs", "n_subjects": 151,
               "connectivity": "mean Fisher-z hippocampus-cortex connectivity",
               "smoothing": ["2mm", "4mm"], "hemisphere_rows": [f"{s}_{h}" for s, h in ROWS],
               "templates": entries}
    (out_root / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    (out_root / "README.md").write_text("""# HCP 7T templates

HCP151 Kong2022 400-ROI templates. Mean Fisher-z values; rows are
`2mm_L`, `2mm_R`, `4mm_L`, `4mm_R`; columns are ROI IDs 1..400.

- `gradient_spectral`: three AP-ordered formal fixed-template prototypes
  (Anterior, Body, Posterior).
- `wta_normalized/7net`: direct canonical7 Fisher-z WTA; rows Default (DMN),
  VentralAttention (VAN), Somatomotor (SMN).
- `wta_normalized/8net`: normalized-top20 canonical8 WTA; rows Default (DMN),
  VentralAttention (VAN), Auditory (Aud), Somatomotor (SMN).
- `wta_normalized/17net`: normalized-top20 family17 WTA; rows DefaultA,
  SalVenAttnA, DefaultC, SomMotB.

WTA profiles correlate each subject's valid pseudo-cluster mean hippocampal
time series with each ROI, apply Fisher-z, and average over all 151 subjects.
Canonical7 assignment correlates vertices with equally averaged ROI network
time series. Normalized-top20 assignment uses the retained 400-ROI group FC
bank and the declared taxonomy. Unobserved values remain NaN.

Each package contains `data/template_profiles.npz`, `data/template_profiles.mat`,
`data/kong2022_400roi_registry.tsv`, and `figures/overview.png`.
`catalog.json` declares current sources, exact variants, and network order.

From `/home/yojiao/hpchome/hipp`, choose a new directory under `.cache/`
and substitute its name for `NEW_OUTPUT_DIRECTORY`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /home/yojiao/.conda/envs/py314/bin/python scripts/iowa7t/export_hcp_templates.py --out-root .cache/NEW_OUTPUT_DIRECTORY
```

The stage directory must not already exist. Publishing updates this package;
existing Iowa7T subject matching and FC outputs require a separate rerun.
""")
    print(f"[complete] staged four packages at {out_root}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", required=True, type=Path,
                        help="New candidate directory; existing directories are refused")
    args = parser.parse_args()
    export(args.out_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

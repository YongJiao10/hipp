#!/usr/bin/env python3
from __future__ import annotations

"""
Build a post-hoc RouteA weighted_fisherz comparison branch.

This does not modify the maintained fisherz outputs. It copies selected RouteA
K=3 gradient-spectral group-atlas runs into an isolated output root, then
recomputes gradient cluster-level weighted_fisherz assignment labels.

The weighted target network profiles are computed by first reconstructing
network time series from Kong2022 ROI mean time series and ROI valid-grayordinate
counts, then recomputing hippocampus-to-network Fisher-z FC.
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from lib.gifti_labels import materialize_combo
from lib.io_utils import write_tsv, load_json, load_run_segment_metadata
from lib.group_atlas_lineage import validate_group_atlas_lineage
from lib.kong2022_label_utils import load_kong2022_merge_mapping
from lib.network_colors import load_network_colors
from lib.output_layout import (
    DEFAULT_KONG2022_ROUTEA_ROOT,
    DEFAULT_KONG2022_SURFACE_ROOT,
    DEFAULT_KONG2022_TEMPLATE_ROOT,
    DEFAULT_OVERVIEWS_DIR,
    DEFAULT_ROUTEA_GROUP_ATLAS_ROOT,
)
from lib.template_assets import canonicalize_template_manifest
from lib.surface_affinity import (
    GRADIENT_SPECTRAL_AFFINITY,
    GRADIENT_SPECTRAL_AFFINITY_DESCRIPTION,
    validate_gradient_spectral_affinity_provenance,
)
from lib.workflow_utils import write_json
from lib.weighted_fisherz import (
    assignment_status_from_weighted_network_profile,
    weighted_cluster_network_fisherz_profile_from_timeseries,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = DEFAULT_ROUTEA_GROUP_ATLAS_ROOT
DEFAULT_OUT_ROOT = REPO_ROOT / "outputs_hcp" / "work" / "group_atlas" / "pfm_weighted"
DEFAULT_ROUTEA_ROOT = REPO_ROOT / "outputs_hcp" / "inputs" / "kong2022" / "shared_inputs" / "pfm"
DEFAULT_REFERENCE_ROOT = DEFAULT_ROUTEA_ROOT if DEFAULT_ROUTEA_ROOT.exists() else DEFAULT_KONG2022_ROUTEA_ROOT
DEFAULT_ROUTEA_SURFACE_TIMESERIES_ROOT = DEFAULT_KONG2022_SURFACE_ROOT
DEFAULT_SURFACE_ROOT = DEFAULT_ROUTEA_SURFACE_TIMESERIES_ROOT
DEFAULT_PRESENT_DIR = DEFAULT_OVERVIEWS_DIR / "weighted_fisherz"
DEFAULT_STYLE = REPO_ROOT / "config" / "hipp_network_style.json"
DEFAULT_TEMPLATE_SURFACE_DIR = DEFAULT_KONG2022_TEMPLATE_ROOT / "group_average_surfaces"
DEFAULT_TEMPLATE_SPEC = REPO_ROOT / "outputs_hcp" / "inputs" / "hippunfold" / "dense_corobl_batch" / "sub-100610" / "hippunfold" / "sub-100610" / "surf" / "sub-100610_den-512_label-hipp_surfaces.spec"
METHODS = ("gradient-spectral",)
SMOOTHINGS = ("2mm", "4mm")
HEMIS = ("L", "R")
ATLAS = "kong2022"
NETWORK_COLUMN = "canonical7"
COVERAGE_TAG = "cov_0p80"
WEIGHTED_VARIANT = "weighted_fisherz"


def weighted_dataset_provenance(out_root: Path) -> tuple[str, Path]:
    resolved = Path(out_root).resolve()
    for dataset in ("outputs_hcp", "outputs_stanford", "outputs_iowa_jj_tms"):
        expected = (REPO_ROOT / dataset).resolve()
        if resolved == expected or expected in resolved.parents:
            return dataset, expected
    raise ValueError(f"Weighted output root is outside a canonical dataset root: {resolved}")


def apply_weighted_source_provenance(
    manifest: dict[str, object],
    *,
    dst_run: Path,
    source_manifest: Path,
) -> None:
    dataset, expected_root = weighted_dataset_provenance(dst_run)
    signature = manifest.get("resume_input_signature_payload")
    subjects_manifest = signature.get("subjects_manifest") if isinstance(signature, dict) else None
    if not isinstance(subjects_manifest, str) or not subjects_manifest.strip():
        raise ValueError("Weighted source manifest lacks subjects_manifest provenance")
    manifest["dataset"] = dataset
    manifest["expected_source_root"] = str(expected_root)
    manifest["weighted_fisherz_source_provenance"] = {
        "dataset": dataset,
        "expected_source_root": str(expected_root),
        "source_group_manifest": str(Path(source_manifest).resolve()),
        "subjects_manifest": str(Path(subjects_manifest).resolve()),
        "n_subjects": manifest.get("n_subjects"),
    }


def weighted_gradient_provenance(manifest: dict[str, object]) -> dict[str, object]:
    provenance = validate_gradient_spectral_affinity_provenance([manifest])
    return {
        "clustering_surface_affinity": provenance["surface_affinity"],
        "clustering_affinity_description": provenance["affinity_description"],
        "weighted_assignment_only": True,
    }


def read_tsv(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows: list[dict[str, object]] = [
            {str(k): v for k, v in row.items()}
            for row in csv.DictReader(f, delimiter="\t")
        ]
    mapping7: dict[str, str] | None = None
    for row in rows:
        if not str(row.get("canonical7", "")).strip():
            if mapping7 is None:
                mapping7 = load_kong2022_merge_mapping(taxonomy="canonical7")
            row["canonical7"] = mapping7[str(row["family17"]).strip()]
    return rows


def copy_run_tree(src_run: Path, dst_run: Path) -> None:
    if not src_run.exists():
        raise FileNotFoundError(f"Missing source run directory: {src_run}")
    dst_run.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_run, dst_run, dirs_exist_ok=True)


def reference_paths(reference_root: Path, subject: str, atlas: str | None = None) -> tuple[Path, Path]:
    atlas_key = str(atlas or ATLAS)
    base = reference_root / f"sub-{subject}" / "reference" / atlas_key
    return base / "cortex_roi_parcel_timeseries.npy", base / "cortex_roi_parcels.tsv"


def surface_timeseries_path(surface_root: Path, subject: str, smoothing: str, hemi: str) -> Path:
    return surface_root / f"sub-{subject}" / "surface" / smoothing / f"sub-{subject}_hemi-{hemi}_timeseries.npy"


def load_subject_weighted_cluster_profile(
    surface_root: Path,
    reference_root: Path,
    subject: str,
    smoothing: str,
    hemi: str,
    labels: np.ndarray,
    support: np.ndarray,
    network_names: list[str],
    *,
    network_column: str | None = None,
) -> np.ndarray:
    ts_path = surface_timeseries_path(surface_root, subject, smoothing, hemi)
    roi_path, rows_path = reference_paths(reference_root, subject)
    mask_path = surface_root / f"sub-{subject}" / "surface" / "tsnr_gate" / f"sub-{subject}_hemi-{hemi}_valid_mask.npy"
    summary_path = roi_path.parent / "reference_summary.json"
    if not ts_path.exists():
        raise FileNotFoundError(f"Missing hippocampal surface timeseries for strict weighted_fisherz: {ts_path}")
    if not mask_path.is_file():
        raise FileNotFoundError(f"Missing subject valid mask for strict weighted_fisherz: {mask_path}")
    ts = np.load(ts_path).astype(np.float32)
    valid_mask = np.load(mask_path).astype(bool)
    if ts.ndim != 2 or valid_mask.ndim != 1 or valid_mask.shape != (ts.shape[0],):
        raise ValueError(f"Subject valid mask vertex mismatch: {mask_path} shape={valid_mask.shape}; timeseries={ts_path} shape={ts.shape}")
    support_mask = np.asarray(support, dtype=bool).reshape(-1)
    if support_mask.shape != valid_mask.shape:
        raise ValueError(f"Group support/subject mask vertex mismatch: support={support_mask.shape}; {mask_path} shape={valid_mask.shape}")
    _run_ids, run_lengths = load_run_segment_metadata(summary_path)
    roi_ts = np.load(roi_path).astype(np.float32)
    rows = read_tsv(rows_path)
    if roi_ts.shape[0] != len(rows):
        raise RuntimeError(f"ROI rows mismatch for sub-{subject}: {roi_ts.shape} vs {len(rows)} rows")
    return weighted_cluster_network_fisherz_profile_from_timeseries(
        labels,
        support_mask & valid_mask,
        ts,
        roi_ts,
        rows,
        network_names,
        network_column=str(network_column or NETWORK_COLUMN),
        run_lengths=run_lengths,
    )


def normalize_template_paths(
    manifest: dict,
    *,
    template_surface_dir: Path = DEFAULT_TEMPLATE_SURFACE_DIR,
    template_spec: Path = DEFAULT_TEMPLATE_SPEC,
    run_dir: Path | None = None,
) -> dict:
    """Canonicalize producer manifest assets for the target dataset/variant.

    The historical ``template_surface_dir``/``template_spec`` arguments remain
    accepted for launcher compatibility, but copied manifests are never
    repaired from those caller paths.  The target run path and manifest variant
    determine the canonical assets.
    """
    del template_surface_dir, template_spec
    return canonicalize_template_manifest(
        manifest,
        hemi=str(manifest.get("hemi", "")),
        run_dir=run_dir,
    )


def normalize_copied_manifest_paths(
    run_dir: Path,
    manifest: dict,
    *,
    template_surface_dir: Path = DEFAULT_TEMPLATE_SURFACE_DIR,
    template_spec: Path = DEFAULT_TEMPLATE_SPEC,
) -> dict:
    variants = manifest.get("assignment_variants", {})
    if isinstance(variants, dict):
        for variant, spec in variants.items():
            if not isinstance(spec, dict):
                continue
            spec["cluster_assignment_status"] = str((run_dir / f"cluster_assignment_status_{variant}.json").resolve())
            spec["group_functional_atlas_labels"] = str((run_dir / f"group_functional_atlas_labels_{variant}.npy").resolve())
            spec["group_functional_atlas_label_map"] = str((run_dir / f"group_functional_atlas_label_map_{variant}.json").resolve())
            raw_label = spec.get("template_functional_atlas_label")
            if not isinstance(raw_label, str) or not raw_label.strip():
                raise ValueError(
                    f"Weighted branch requires the exact declared template_functional_atlas_label "
                    f"for {variant}: {run_dir}"
                )
            spec["template_functional_atlas_label"] = str(
                (run_dir / "template" / "workbench_assets" / Path(raw_label).name).resolve()
            )
            spec["template_functional_atlas_label_variant"] = spec["template_functional_atlas_label"]
    raw_cluster = manifest.get("template_cluster_label")
    if raw_cluster:
        manifest["template_cluster_label"] = str((run_dir / "template" / "workbench_assets" / Path(str(raw_cluster)).name).resolve())
    return normalize_template_paths(manifest, template_surface_dir=template_surface_dir, template_spec=template_spec, run_dir=run_dir)


def recompute_gradient_weighted_run(
    *,
    dst_run: Path,
    surface_root: Path,
    reference_root: Path,
    color_map: dict[str, tuple[int, int, int, int]],
    template_surface_dir: Path,
    template_spec: Path,
    source_manifest: Path,
) -> dict[str, object]:
    manifest_path = dst_run / "group_atlas_manifest.json"
    if ATLAS == "kong2022":
        validate_group_atlas_lineage(source_manifest)
    manifest = normalize_copied_manifest_paths(
        run_dir=dst_run,
        manifest=load_json(manifest_path),
        template_surface_dir=template_surface_dir,
        template_spec=template_spec,
    )
    clustering_provenance = weighted_gradient_provenance(manifest)
    if str(manifest.get("status", "ok")) != "ok":
        raise RuntimeError(f"Cannot recompute non-ok run: {manifest_path}")
    labels = np.load(dst_run / "group_labels_final.npy").astype(np.int32)
    support = np.load(dst_run / "group_support_mask.npy").astype(bool)
    feature_summary = load_json(dst_run / "group_feature_summary.json")
    network_names = [str(x) for x in feature_summary["network_names"]]
    subjects = [str(x).replace("sub-", "") for x in manifest["subjects"]]
    smoothing = str(manifest["smoothing"])
    hemi = str(manifest["hemi"])
    k_final = int(np.max(labels))

    profiles: list[np.ndarray] = []
    for subject in subjects:
        cluster_profile = load_subject_weighted_cluster_profile(
            surface_root,
            reference_root,
            subject,
            smoothing,
            hemi,
            labels,
            support,
            network_names,
            network_column=NETWORK_COLUMN,
        )
        if cluster_profile.shape != (k_final, len(network_names)):
            raise RuntimeError(f"Weighted cluster profile mismatch for sub-{subject}: {cluster_profile.shape}")
        profiles.append(cluster_profile)

    profile = np.nanmean(np.stack(profiles, axis=0), axis=0).astype(np.float32)
    old_profile = dst_run / "cluster_network_profile.npy"
    if not old_profile.is_file():
        raise FileNotFoundError(f"Weighted branch source lacks exact baseline profile: {old_profile}")
    baseline_profile = dst_run / "cluster_network_profile_roi_equal_fisherz_baseline.npy"
    shutil.copy2(old_profile, baseline_profile)
    np.save(dst_run / "cluster_network_profile.npy", profile)
    weighted_profile = dst_run / "cluster_network_profile_weighted_fisherz.npy"
    np.save(weighted_profile, profile)

    status_rows, dominant = assignment_status_from_weighted_network_profile(profile, network_names)
    status_payload = {
        "items": status_rows,
        "assignment_metric": "top1_weighted_network_fisher_z",
        "q_threshold": None,
        "top_n_cap": None,
        "network_profile_weighting": "network_timeseries_weighted_by_roi_n_grayordinates_used_then_fisherz_fc",
    }
    write_json(dst_run / "cluster_assignment_status_weighted_fisherz.json", status_payload)
    write_json(
        dst_run / "cluster_dominant_network_weighted_fisherz.json",
        {"dominant": dominant, "network_names": network_names, "assignment_variant": WEIGHTED_VARIANT},
    )

    variants = manifest.setdefault("assignment_variants", {})
    if isinstance(variants.get("fisherz"), dict):
        variants["fisherz"]["cluster_network_profile"] = str(baseline_profile.resolve())
    variants[WEIGHTED_VARIANT] = {
        "assignment_metric": "top1_weighted_network_fisher_z",
        "cluster_assignment_status": str((dst_run / "cluster_assignment_status_weighted_fisherz.json").resolve()),
        "group_functional_atlas_labels": str((dst_run / "group_functional_atlas_labels_weighted_fisherz.npy").resolve()),
        "group_functional_atlas_label_map": str((dst_run / "group_functional_atlas_label_map_weighted_fisherz.json").resolve()),
        "template_functional_atlas_label": str((dst_run / "template" / "workbench_assets" / f"weighted_fisherz_{smoothing}_hemi-{hemi}.label.gii").resolve()),
        "cluster_network_profile": str(weighted_profile.resolve()),
    }
    manifest["default_assignment_variant"] = WEIGHTED_VARIANT
    manifest["cluster_assignment_metric"] = "top1_weighted_network_fisher_z"
    manifest["cluster_assignment_status"] = str((dst_run / "cluster_assignment_status_weighted_fisherz.json").resolve())
    manifest["cluster_assignment_status_weighted_fisherz"] = str((dst_run / "cluster_assignment_status_weighted_fisherz.json").resolve())
    manifest["cluster_network_profile"] = str((dst_run / "cluster_network_profile.npy").resolve())
    manifest["weighted_fisherz_branch"] = {
        "status": "posthoc_recomputed",
        "source_assignment_variant_preserved": "fisherz",
        "network_profile_weighting": "network_timeseries_weighted_by_roi_n_grayordinates_used_then_fisherz_fc",
    }
    manifest.update(clustering_provenance)
    apply_weighted_source_provenance(
        manifest,
        dst_run=dst_run,
        source_manifest=source_manifest,
    )
    write_json(manifest_path, manifest)
    if not materialize_combo(manifest_path, color_map):
        raise RuntimeError(f"Failed to materialize weighted_fisherz labels: {manifest_path}")
    return {"run_dir": str(dst_run), "status": "ok", "method": "gradient-spectral"}


def run_dirs_for_method(root: Path, method: str, atlas: str | None = None) -> list[Path]:
    atlas_key = str(atlas or ATLAS)
    if method == "gradient-spectral":
        base = root / method / atlas_key
        out: list[Path] = []
        for smoothing in SMOOTHINGS:
            cov_dir = base / smoothing / COVERAGE_TAG
            if not cov_dir.exists():
                continue
            for comp_dir in sorted(p for p in cov_dir.iterdir() if p.is_dir() and p.name.startswith("g")):
                for hemi in HEMIS:
                    out.append(comp_dir / f"hemi_{hemi}")
        return out
    raise ValueError(f"Unsupported method: {method}")


def write_comparison_summary(source_root: Path, out_root: Path, present_dir: Path) -> None:
    rows: list[dict[str, object]] = []
    for method in METHODS:
        for new_run in run_dirs_for_method(out_root, method, atlas=ATLAS):
            rel = new_run.relative_to(out_root)
            old_run = source_root / rel
            if not old_run.exists() or not (new_run / "group_atlas_manifest.json").exists():
                continue
            new_manifest = load_json(new_run / "group_atlas_manifest.json")
            smoothing = str(new_manifest.get("smoothing", ""))
            hemi = str(new_manifest.get("hemi", ""))
            component = new_run.parent.name if method == "gradient-spectral" else ""
            if method == "gradient-spectral":
                old_labels = np.load(old_run / "group_functional_atlas_labels_fisherz.npy").astype(np.int32)
                new_labels = np.load(new_run / "group_functional_atlas_labels_weighted_fisherz.npy").astype(np.int32)
                old_profile = np.load(new_run / "cluster_network_profile_roi_equal_fisherz_baseline.npy").astype(np.float32)
                new_profile = np.load(new_run / "cluster_network_profile_weighted_fisherz.npy").astype(np.float32)
                changed = (old_labels != new_labels) & (old_labels > 0)
                n_old = int(np.sum(old_labels > 0))
                rows.append(
                    {
                        "method": method,
                        "component": component,
                        "smoothing": smoothing,
                        "hemi": hemi,
                        "n_vertices_compared": n_old,
                        "n_vertices_changed": int(np.sum(changed)),
                        "fraction_changed": float(np.sum(changed) / max(1, n_old)),
                        "mean_abs_profile_delta": float(np.nanmean(np.abs(new_profile - old_profile))),
                        "max_abs_profile_delta": float(np.nanmax(np.abs(new_profile - old_profile))),
                    }
                )
    present_dir.mkdir(parents=True, exist_ok=True)
    fields = ["method", "component", "smoothing", "hemi", "n_vertices_compared", "n_vertices_changed", "fraction_changed", "mean_abs_profile_delta", "max_abs_profile_delta"]
    write_tsv(present_dir / "weighted_fisherz_difference_summary.tsv", rows, fields)
    lines = [
        "# Weighted Fisherz Branch Difference Summary",
        "",
        f"Baseline: `{source_root}`",
        f"Weighted branch: `{out_root}`",
        "",
        "This branch preserves the original `fisherz` outputs and adds/recomputes `weighted_fisherz` comparison labels from vertex-count weighted network-timeseries Fisher-z FC profiles.",
        "",
        "| method | component | smoothing | hemi | vertices | changed | fraction changed | mean abs profile delta |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        delta = row["mean_abs_profile_delta"]
        delta_text = "" if delta == "" else f"{float(str(delta)):.6f}"
        lines.append(
            f"| {row['method']} | {row.get('component', '')} | {row['smoothing']} | {row['hemi']} | {row['n_vertices_compared']} | "
            f"{row['n_vertices_changed']} | {float(str(row['fraction_changed'])):.6f} | {delta_text} |"
        )
    lines.extend(
        [
            "",
            "Rendered overview products are listed in `group_overview_weighted_fisherz_manifest.json` after running `06_01_build_group_overview.py`.",
        ]
    )
    (present_dir / "weighted_fisherz_difference_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_branch(args: argparse.Namespace) -> dict[str, object]:
    global ATLAS, NETWORK_COLUMN
    ATLAS = str(args.atlas)
    NETWORK_COLUMN = str(
        args.network_column
        or "canonical7"
    )
    if ATLAS == "kong2022" and NETWORK_COLUMN == "canonical8":
        raise ValueError(
            "The canonical8 taxonomy is retired for active production; use canonical7 instead."
        )
    if ATLAS == "kong2022" and NETWORK_COLUMN != "canonical7":
        raise ValueError("kong2022 requires --network-column canonical7 (canonical8 is historical only)")
    source_root = Path(args.source_root).resolve()
    out_root = Path(args.out_root).resolve()
    surface_root = Path(args.surface_root).resolve()
    reference_root = Path(args.reference_root).resolve()
    present_dir = Path(args.present_dir).resolve()
    template_surface_dir = Path(args.template_surface_dir).resolve()
    template_spec = Path(args.template_spec).resolve()
    color_map = load_network_colors(Path(args.network_style).resolve())
    methods = tuple(str(method) for method in args.methods)
    results: list[dict[str, object]] = []
    for method in methods:
        for src_run in run_dirs_for_method(source_root, method, atlas=ATLAS):
            if not src_run.exists():
                continue
            source_manifest = src_run / "group_atlas_manifest.json"
            if ATLAS == "kong2022":
                validate_group_atlas_lineage(source_manifest)
            dst_run = out_root / src_run.relative_to(source_root)
            copy_run_tree(src_run, dst_run)
            results.append(
                recompute_gradient_weighted_run(
                    dst_run=dst_run,
                    surface_root=surface_root,
                    reference_root=reference_root,
                    color_map=color_map,
                    template_surface_dir=template_surface_dir,
                    template_spec=template_spec,
                    source_manifest=source_manifest,
                )
            )
    write_comparison_summary(source_root, out_root, present_dir)
    manifest = {
        "status": "ok",
        "branch": "weighted_fisherz",
        "atlas": ATLAS,
        "network_column": NETWORK_COLUMN,
        "source_root": str(source_root),
        "out_root": str(out_root),
        "surface_root": str(surface_root),
        "surface_root_note": "strict weighted_fisherz uses hippocampal surface timeseries plus ROI timeseries weighted by n_grayordinates_used",
        "reference_root": str(reference_root),
        "present_dir": str(present_dir),
        "template_surface_dir": str(template_surface_dir),
        "template_spec": str(template_spec),
        "methods": list(methods),
        "runs": results,
    }
    write_json(out_root / "weighted_fisherz_branch_manifest.json", manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build RouteA weighted_fisherz comparison branch from existing HCP group-atlas outputs.")
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--atlas", choices=("kong2022",), default="kong2022")
    parser.add_argument("--network-column", choices=("canonical7", "canonical8"), default="canonical7")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--surface-root", default=str(DEFAULT_SURFACE_ROOT), help="Subject shared-input root containing sub-*/surface/<smoothing> hippocampal timeseries for strict weighted_fisherz recomputation.")
    parser.add_argument("--reference-root", default=str(DEFAULT_REFERENCE_ROOT))
    parser.add_argument("--present-dir", default=str(DEFAULT_PRESENT_DIR))
    parser.add_argument("--network-style", default=str(DEFAULT_STYLE))
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
        help=(
            "Only the gradient-spectral weighted branch is supported."
        ),
    )
    parser.add_argument("--template-surface-dir", default=str(DEFAULT_TEMPLATE_SURFACE_DIR), help="Directory containing group-average hemi-L/R midthickness surfaces used in copied manifests.")
    parser.add_argument("--template-spec", default=str(DEFAULT_TEMPLATE_SPEC), help="Template subject HippUnfold spec path used in copied manifests.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    manifest = build_branch(args)
    runs = manifest.get("runs")
    run_count = len(runs) if isinstance(runs, list) else 0
    print(json.dumps({"status": manifest["status"], "out_root": manifest["out_root"], "runs": run_count}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

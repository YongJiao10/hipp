"""Template-matching dominant-atlas subject FC and Iowa-HC contrasts.

Separate preparation stage for the TM third column in
multidataset_hipp_cortex_gradient_wta (2mm/36p standard/ctxreg only).
Canonical 7-panel contrasts are untouched; TM uses its own schema,
panels, and manifest so existing hashes never change.

Masks are each cohort's own group dominant atlas (majority vote with
mean-template-correlation tie-break, same as
template_matching_dominant_subregions.compute_cohort_data).
FC math reuses the producer cluster-mean-timeseries rule
(subject_label_fc), never averaged vertex correlations.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import TypedDict

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CORE_WORKFLOW_DIR = ROOT / "scripts/core_workflow"
if str(CORE_WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_WORKFLOW_DIR))


def active_path(path):
    """Reject archive inputs, including symlinks, before reading contents."""
    resolved = Path(path).resolve()
    if any(part.lower().startswith(("_archive", "archive")) for part in resolved.parts):
        raise ValueError(
            f"Active lineage cannot consume an archive artifact: {path} -> {resolved}"
        )
    return resolved


def _read_subject_roi_tsv_by_path(path):
    """Read ROI rows via the canonical core-workflow reader."""
    from lib.cortex_template_profiles import _read_subject_roi_tsv

    return _read_subject_roi_tsv(Path(path))


FIGURE_DIR = ROOT / "scripts/visualize/multidataset_hipp_cortex_gradient_wta"
TM_FIGURE_DIR = ROOT / "scripts/visualize/template_matching_dominant_subregions"
SCHEMA = "tm_subject_roi_welch_hc_v1_uncorrected"
T_CMAP = "PiYG_r"

TM_PANELS: tuple[str, ...] = ("TM-Anterior", "TM-Middle", "TM-Posterior")
TM_DISPLAY_TITLES: tuple[tuple[str, str], ...] = (
    ("TM Anterior (VAN)", "#dc5cff"),
    ("TM Middle (SMN)", "#6396c2"),
    ("TM Posterior (DMN)", "#d85c6b"),
)
TM_LABEL_SEMANTICS: dict[int, str] = {
    1: "Anterior/VAN",
    2: "Middle/SMN",
    3: "Posterior/DMN",
}
TM_SUBREGION_DISPLAY: dict[int, str] = {
    1: "Anterior / VAN",
    2: "Middle / SMN",
    3: "Posterior / DMN",
}
TM_LEGACY_STATS_CSV = TM_FIGURE_DIR / "template_matching_dominant_subregions_stats.csv"
# Workbench label colors for the TM hippo atlas (template-matching contract:
# Anterior/VAN purple, Middle/SMN blue, Posterior/DMN coral).
TM_GIFTI_RGBA: dict[int, tuple[int, int, int, int]] = {
    1: (220, 92, 255, 255),
    2: (99, 150, 194, 255),
    3: (216, 92, 107, 255),
}

TM_COHORT_ROSTERS: dict[str, str] = {
    "iowa_hc": "jj_tms_hc_67_subjects.txt",
    "iowa_pat": "jj_tms_patient_26_subjects.txt",
    "iowa_pure_mdd": "jj_tms_pure_mdd_7_subjects.txt",
    "iowa_comorbid_mdd": "jj_tms_comorbid_mdd_19_subjects.txt",
}


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing TM cache source file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path):
    path = active_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing canonical TM contrast input: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def renderer_module():
    path = FIGURE_DIR / "render_multidataset_hipp_cortex_gradient_wta.py"
    spec = importlib.util.spec_from_file_location("hipp_cortex_renderer_tm", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load renderer module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _branch_matching_roots(data_root: Path = ROOT) -> dict[str, Path]:
    root = Path(data_root).resolve()
    base = root / "outputs_iowa_jj_tms" / "work" / "hcp151_fixed_template_matching"
    return {
        "standard": base / "subjects",
        "ctxreg": base / "pfm_ctxreg" / "subjects",
    }


def _load_roster(roster_file: str) -> list[str]:
    lines = (ROOT / "manifests" / roster_file).read_text(encoding="utf-8").splitlines()
    return [
        f"sub-{line.strip()}"
        for line in lines
        if line.strip() and not line.startswith("#")
    ]


def _declared_output_path(manifest_path: Path, manifest: dict, key: str) -> Path:
    record = manifest.get("outputs", {}).get(key)
    if not isinstance(record, dict):
        raise ValueError(f"Missing TM output receipt {key!r}: {manifest_path}")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise ValueError(f"TM output receipt {key!r} is not absolute: {manifest_path}")
    path = Path(raw_path)
    if not path.is_file():
        raise FileNotFoundError(f"Declared TM output {key!r} missing: {path}")
    expected_size = record.get("size_bytes")
    if expected_size is not None and path.stat().st_size != int(expected_size):
        raise ValueError(f"Declared TM output size mismatch {key!r}: {path}")
    expected_sha = record.get("sha256")
    if expected_sha and _sha256_file(path) != expected_sha:
        raise ValueError(f"Declared TM output checksum mismatch {key!r}: {path}")
    return path


class TmDominantInfo(TypedDict):
    """Group dominant atlas plus the exact upstream files behind it."""

    subjects: list[str]
    fingerprints: list[dict]
    combo_manifests: tuple[Path, ...]
    label_paths: tuple[Path, ...]
    score_paths: tuple[Path, ...]


def tm_dominant_labels(
    cohort_id: str,
    signal_branch: str,
    *,
    data_root: Path = ROOT,
) -> tuple[dict[str, np.ndarray], TmDominantInfo]:
    """Compute group dominant labels per hemi for one Iowa cohort/branch."""
    if cohort_id not in TM_COHORT_ROSTERS:
        raise ValueError(f"No TM dominant atlas for cohort {cohort_id!r}; HCP has none")
    if signal_branch not in ("standard", "ctxreg"):
        raise ValueError(f"Unsupported TM signal branch: {signal_branch!r}")
    subjects_root = _branch_matching_roots(data_root)[signal_branch]
    subjects = _load_roster(TM_COHORT_ROSTERS[cohort_id])
    dominant: dict[str, np.ndarray] = {}
    fingerprints: list[dict] = []
    combo_paths: list[Path] = []
    label_paths: list[Path] = []
    score_paths: list[Path] = []
    for hemi in ("L", "R"):
        sub_labels: list[np.ndarray] = []
        sub_corrs: list[np.ndarray] = []
        for subject in subjects:
            manifest_path = (
                subjects_root
                / subject
                / "matching"
                / "2mm"
                / f"hemi-{hemi}"
                / "combo_manifest.json"
            )
            if not manifest_path.is_file():
                raise FileNotFoundError(f"Missing TM manifest: {manifest_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("smoothing") != "2mm"
                or manifest.get("hemi") != hemi
                or manifest.get("status") != "ok"
            ):
                raise ValueError(f"TM manifest metadata mismatch: {manifest_path}")
            recorded_branch = manifest.get("signal_branch")
            if signal_branch == "ctxreg" and recorded_branch != "ctxreg":
                raise ValueError(f"TM CtxReg branch mismatch: {manifest_path}")
            if signal_branch == "standard" and recorded_branch not in (
                None,
                "standard",
            ):
                raise ValueError(f"TM Standard branch mismatch: {manifest_path}")
            raw_sem = manifest.get("label_semantics")
            semantics = (
                {int(k): str(v) for k, v in raw_sem.items()}
                if isinstance(raw_sem, dict)
                else None
            )
            if semantics != TM_LABEL_SEMANTICS:
                raise ValueError(f"Unexpected TM label semantics: {manifest_path}")
            f_lbl = _declared_output_path(manifest_path, manifest, "labels")
            f_cor = _declared_output_path(manifest_path, manifest, "scores")
            labels = np.asarray(np.load(f_lbl), dtype=np.int32)
            corrs = np.asarray(np.load(f_cor), dtype=np.float32)
            if labels.shape != (512,):
                raise ValueError(f"Expected 512 TM labels: {f_lbl} got {labels.shape}")
            if corrs.shape != (512, 3):
                raise ValueError(
                    f"Expected (512,3) TM scores: {f_cor} got {corrs.shape}"
                )
            sub_labels.append(labels)
            sub_corrs.append(corrs)
            fingerprints.append(fingerprint(manifest_path))
            fingerprints.append(fingerprint(f_lbl))
            fingerprints.append(fingerprint(f_cor))
            combo_paths.append(manifest_path)
            label_paths.append(f_lbl)
            score_paths.append(f_cor)
        arr_labels = np.asarray(sub_labels, dtype=np.int32)
        arr_corrs = np.asarray(sub_corrs, dtype=np.float32)
        counts = np.stack([np.sum(arr_labels == k, axis=0) for k in (1, 2, 3)], axis=-1)
        mean_corrs = np.mean(arr_corrs, axis=0)
        combined = counts.astype(np.float64) + 1e-4 * mean_corrs.astype(np.float64)
        dominant[hemi] = (np.argmax(combined, axis=-1) + 1).astype(np.int32)

    def _dedup(paths: list[Path]) -> tuple[Path, ...]:
        return tuple(dict.fromkeys(paths))

    return dominant, TmDominantInfo(
        subjects=[str(s) for s in subjects],
        fingerprints=fingerprints,
        combo_manifests=_dedup(combo_paths),
        label_paths=_dedup(label_paths),
        score_paths=_dedup(score_paths),
    )


def _signal_branch_for_grid(cohort, grid) -> str:
    branch = str(getattr(grid, "signal_branch", "standard"))
    if branch not in ("standard", "ctxreg"):
        raise ValueError(f"Unknown TM signal branch {branch!r}")
    recorded = cohort.get("signal_branch")
    if recorded is not None and str(recorded) != branch:
        raise ValueError(
            f"TM cohort branch disagrees with grid: {cohort.get('id')!r} "
            f"recorded={recorded!r} requested={branch!r}"
        )
    return branch


def _ids(manifest, key="subjects"):
    ids = [str(s).removeprefix("sub-") for s in (manifest.get(key) or [])]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"Missing or duplicate TM subject IDs: {key}")
    return ids


def extraction_code():
    import cortex_group_contrasts as _cc

    return _cc.extraction_code() + [fingerprint(Path(__file__))]


def extract_tm_hemisphere(
    renderer, cohort, grid, hemi, bundle, source_mode="canonical"
):
    """Extract (N,3,400) TM FC for one hemi using cohort's own dominant masks."""
    import cortex_group_contrasts as _cc

    if grid.smoothing != "2mm" or grid.variant != "36p":
        raise ValueError(
            f"TM contrasts require 2mm/36p, got {grid.smoothing}/{grid.variant}"
        )
    cid = cohort["id"]
    if cid not in TM_COHORT_ROSTERS:
        raise ValueError(f"No TM extraction for cohort {cid!r}")
    branch = _signal_branch_for_grid(cohort, grid)
    grad_dir = Path(cohort["grad_subpath"])
    manifest_path = grad_dir / f"hemi_{hemi}" / "group_atlas_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing gradient manifest for TM alignment: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recovery = (
        source_mode == "original-denoised" and cohort["dataset_key"] == "iowa_jj_tms"
    )
    subjects = _cc.validate_profile_subjects(manifest, cohort["n_subjects"], recovery)
    tm_roster = [s.removeprefix("sub-") for s in _load_roster(TM_COHORT_ROSTERS[cid])]
    if tm_roster != subjects:
        raise ValueError(
            f"TM roster differs from gradient subject order for {cid}; "
            "refusing potentially misaligned TM FC"
        )
    dominant, tm_info = tm_dominant_labels(cid, branch)
    tm_labels = dominant[hemi]
    if tm_labels.shape != (512,):
        raise ValueError(f"TM dominant labels must be (512,): {cid} {hemi}")
    masks_group = [(tm_labels == k) for k in (1, 2, 3)]
    sources: list[dict] = [fingerprint(manifest_path)]
    sources.extend(tm_info["fingerprints"])  # type: ignore[arg-type]
    surface, reference, masks = renderer.resolve_cohort_inputs_roots(
        cohort, grid.variant
    )
    registry = bundle["roi_registry"]
    columns = {int(row["roi_id"]): i for i, row in enumerate(registry)}
    stack = []
    for subject in subjects:
        paths = _cc.subject_input_paths(
            renderer, cohort, grid, subject, hemi, source_mode
        )
        sources.extend(_cc.fingerprint(p) for p in paths)
        ts, roi, valid = np.load(paths[0]), np.load(paths[1]), np.load(paths[3])
        if (
            source_mode == "original-denoised"
            and grid.variant == "36p"
            and cohort["dataset_key"] == "iowa_jj_tms"
        ):
            sources.append(_cc.validate_original_source(paths, subject))
            with paths[2].open() as handle:
                rows = [
                    {"parcel_id": row["roi_id"], "parcel_name": row["roi_name"]}
                    for row in csv.DictReader(handle, delimiter="\t")
                ]
        else:
            rows = _read_subject_roi_tsv_by_path(paths[2])
        if (
            ts.ndim != 2
            or roi.ndim != 2
            or ts.shape[1] != roi.shape[1]
            or ts.shape[0] != 512
            or valid.shape != (512,)
            or roi.shape[0] != len(rows)
            or len(rows) != 400
        ):
            raise ValueError(f"TM subject FC shape/400 ROI mismatch: {paths}")
        roi_ids = [int(row["parcel_id"]) for row in rows]
        if len(set(roi_ids)) != 400 or set(roi_ids) != set(columns):
            raise ValueError(f"TM ROI identity mismatch: {paths[2]}")
        for row in rows:
            reg = registry[columns[int(row["parcel_id"])]]
            if row["parcel_name"] != reg["roi_name"]:
                raise ValueError(f"TM ROI name mismatch: {paths[2]}")
        values = _cc.subject_label_fc(ts, roi, masks_group, valid, str(paths))
        ordered = np.full((3, 400), np.nan, dtype=np.float32)
        ordered[:, [columns[i] for i in roi_ids]] = values
        stack.append(ordered)
    return np.stack(stack), subjects, sources


def tm_subject_path(out_dir: Path, version: str, tag: str, cid: str) -> Path:
    return out_dir / f"{version}_{tag}_{cid}_tm_subjects.npz"


def tm_contrast_path(out_dir: Path, version: str, tag: str, cid: str) -> Path:
    return out_dir / f"{version}_{tag}_{cid}_tm_vs_iowa_hc.npz"


def prepare_tm_contrasts(
    grid_specs,
    version="canonical",
    rebuild_profiles=False,
    source_mode="canonical",
    manifest_path=None,
) -> Path:
    """Export TM subject FC and Welch vs Iowa HC for 2mm/36p branches only."""
    import cortex_group_contrasts as _cc

    renderer = renderer_module()
    bundle = renderer.load_kong2022_bundle(
        dlabel_path=renderer.resolve_cortex_assets()["dlabel"]
    )
    for grid in grid_specs:
        if (grid.smoothing, grid.variant) != ("2mm", "36p"):
            raise ValueError(f"TM supports only 2mm/36p, got {grid.tag}")
    out = FIGURE_DIR / "contrasts"
    out.mkdir(exist_ok=True)
    receipt = {
        "schema": SCHEMA,
        "version": version,
        "iowa36p_source": source_mode,
        "test": "two-sided Welch",
        "correction": "none; uncorrected p<0.05 independently per ROI",
        "threshold": "p<0.05",
        "direction": "group - Iowa HC",
        "bilateral": "within-subject arithmetic mean, both hemispheres required",
        "tm_masks": "cohort-own dominant atlas (majority vote + mean-correlation tie-break)",
        "grids": {},
        "t_vmax": 1.0,
        "t_display_vmax": renderer.T_DISPLAY_VMAX,
        "colormap": T_CMAP,
        "code": extraction_code(),
        "roi_registry": bundle["roi_registry"],
        "panels": list(TM_PANELS),
    }
    peak = 1.0
    for grid in grid_specs:
        cohorts = _cc.figure_cohorts(renderer, grid, version)
        profiles: dict[str, np.ndarray] = {}
        entries: dict[str, dict] = {}
        for cohort in cohorts:
            cid = cohort["id"]
            if cid == "hcp_7t":
                continue
            path = tm_subject_path(out, version, grid.tag, cid)
            sidecar = path.with_suffix(".manifest.json")
            manifest_file = (
                Path(cohort["grad_subpath"]) / "hemi_L/group_atlas_manifest.json"
            )
            if not manifest_file.is_file():
                raise FileNotFoundError(f"Missing cohort manifest: {manifest_file}")
            ids = _ids(json.loads(manifest_file.read_text()))
            if rebuild_profiles or not (path.is_file() and sidecar.is_file()):
                print(f"[TM FC export] {grid.tag} {cid}", flush=True)
                left, ids_l, lsrc = extract_tm_hemisphere(
                    renderer, cohort, grid, "L", bundle, source_mode
                )
                right, ids_r, rsrc = extract_tm_hemisphere(
                    renderer, cohort, grid, "R", bundle, source_mode
                )
                values = _cc.bilateral_profiles(left, ids_l, right, ids_r)
                np.savez_compressed(
                    path,
                    values=values,
                    left=left,
                    right=right,
                    subjects=np.array(ids_l),
                    panels=np.array(list(TM_PANELS)),
                )
                entry = {
                    "schema": SCHEMA,
                    "iowa36p_source": source_mode,
                    "profile": fingerprint(path),
                    "subjects": ids_l,
                    "panels": list(TM_PANELS),
                    "sources": lsrc + rsrc,
                    "code": receipt["code"],
                }
                sidecar.write_text(json.dumps(entry, indent=2) + "\n")
            else:
                print(f"[TM FC cache] {grid.tag} {cid}", flush=True)
                values, entry = _cc.read_subject_cache(
                    path, ids, list(TM_PANELS), receipt["code"]
                )
                if entry.get("schema") != SCHEMA:
                    raise ValueError(f"TM subject export schema mismatch: {path}")
                if entry.get("iowa36p_source") != source_mode:
                    raise ValueError(f"TM subject FC source mode differs: {path}")
            profiles[cid] = values
            entries[cid] = entry
        if "iowa_hc" not in profiles:
            raise ValueError("TM contrasts require Iowa HC reference")
        hc = profiles["iowa_hc"]
        for cid, values in profiles.items():
            if cid == "iowa_hc":
                entries[cid]["comparison"] = None
                continue
            result = _cc.compare_profiles(values, hc)
            sig_t = np.abs(result["t"][result["significant"]])
            if sig_t.size:
                peak = max(peak, float(sig_t.max()))
            path = tm_contrast_path(out, version, grid.tag, cid)
            np.savez_compressed(
                path,
                t=result["t"],
                p=result["p"],
                n_group=result["n_group"],
                n_reference=result["n_reference"],
                testable=result["testable"],
                significant=result["significant"],
            )
            entries[cid]["comparison"] = fingerprint(path)
            table = path.with_suffix(".tsv")
            with table.open("w") as handle:
                writer = csv.writer(handle, delimiter="\t")
                writer.writerow(
                    [
                        "panel",
                        "roi_id",
                        "roi_name",
                        "t",
                        "p",
                        "n_group",
                        "n_reference",
                        "testable",
                        "significant",
                    ]
                )
                for j, name in enumerate(entries[cid]["panels"]):
                    for i, roi in enumerate(bundle["roi_registry"]):
                        writer.writerow(
                            [name, roi["roi_id"], roi["roi_name"]]
                            + [
                                result[k][j, i]
                                for k in (
                                    "t",
                                    "p",
                                    "n_group",
                                    "n_reference",
                                    "testable",
                                    "significant",
                                )
                            ]
                        )
            print(
                f"[TM Welch/raw-p] {grid.tag} {cid}: "
                f"{result['significant'].sum(axis=1).tolist()} significant ROI",
                flush=True,
            )
        receipt["grids"][grid.tag] = entries
    receipt["t_vmax"] = float(math.ceil(peak))
    receipt["signal_branches"] = sorted({str(g.signal_branch) for g in grid_specs})
    if manifest_path is None:
        tags = sorted(str(g.tag) for g in grid_specs)
        path = out / f"{version}_{'__'.join(tags)}_tm.manifest.json"
    else:
        path = Path(manifest_path)
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        f"[TM contrasts ready] t peak +/-{receipt['t_vmax']}; "
        f"display +/-{receipt['t_display_vmax']}: {path}",
        flush=True,
    )
    return path


def load_tm_contrasts(path, grid, version="canonical"):

    receipt = json.loads(Path(path).read_text())
    if receipt.get("schema") != SCHEMA or receipt.get("version") != version:
        raise ValueError(f"TM contrast manifest contract mismatch: {path}")
    if receipt.get("panels") != list(TM_PANELS):
        raise ValueError(f"TM contrast panel order mismatch: {path}")
    grids = receipt.get("grids")
    if not isinstance(grids, dict) or grid.tag not in grids:
        raise ValueError(f"TM manifest lacks grid {grid.tag!r}: {path}")
    requested_branch = str(getattr(grid, "signal_branch", "standard"))
    recorded = receipt.get("signal_branches")
    if not isinstance(recorded, list) or requested_branch not in {
        str(v) for v in recorded
    }:
        raise ValueError(f"TM manifest branch mismatch for {grid.tag!r}: {path}")
    try:
        stat_vmax = float(receipt["t_vmax"])
        display_vmax = float(receipt["t_display_vmax"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"TM manifest must declare t ranges: {path}") from exc
    if (
        not np.isfinite(stat_vmax)
        or stat_vmax < 1
        or not np.isfinite(display_vmax)
        or display_vmax <= 0
        or receipt["code"] != extraction_code()
    ):
        raise ValueError(f"TM scale or code mismatch: {path}")
    entries = grids[grid.tag]
    result: dict[str, dict | None] = {"hcp_7t": None}
    for cid, entry in entries.items():
        if entry.get("panels") != list(TM_PANELS):
            raise ValueError(f"TM panel order mismatch: {cid}")
        source = entry.get("comparison")
        if cid == "iowa_hc":
            if source is not None:
                raise ValueError("iowa_hc must not have a TM comparison")
            result[cid] = None
            continue
        if source is None:
            raise ValueError(f"Missing TM comparison for {cid}")
        if fingerprint(source["path"]) != source:
            raise ValueError(f"Stale TM result: {source['path']}")
        with np.load(source["path"]) as data:
            keys = ("t", "p", "n_group", "n_reference", "testable", "significant")
            if set(data.files) != set(keys) or any(
                data[k].shape != (3, 400) for k in keys
            ):
                raise ValueError(f"TM array contract mismatch: {source['path']}")
            result[cid] = {k: data[k] for k in keys}
    # Validate all stored sources are fresh.
    checked: set[str] = set()
    all_entries = [e for group in grids.values() for e in group.values()]
    all_sources = (
        receipt["code"]
        + [e["profile"] for e in all_entries]
        + [e["comparison"] for e in all_entries if e.get("comparison") is not None]
    )
    for source in all_sources:
        if source["path"] not in checked:
            if fingerprint(source["path"]) != source:
                raise ValueError(f"Stale TM source: {source['path']}")
            checked.add(source["path"])
    for entry in all_entries:
        for src in entry.get("sources", []):
            sp = Path(src["path"])
            if sp.is_file() and str(sp) not in checked:
                if fingerprint(sp) != src:
                    raise ValueError(f"Stale TM source: {sp}")
                checked.add(str(sp))
    return receipt, result

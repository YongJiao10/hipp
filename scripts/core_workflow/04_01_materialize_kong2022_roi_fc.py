#!/usr/bin/env python3
from __future__ import annotations

"""
Process Step Description:
Materialize Kong2022 ROI-level functional connectivity (FC). Computes the functional connectivity matrix between hippocampal surface vertices and Kong2022 cortical network regions of interest (ROIs).

Main Flow Path:
- 6th step in the numbered pipeline sequence (ROI functional connectivity materialization for kong2022):
  01_01_run_hippunfold_t1w_local.py -> 01_02_check_hippunfold_outputs.py -> 02_01_prepare_group_inputs.py -> 02_02_build_qc_manifest.py -> 03_01/03_02 reference building -> 04_01_materialize_kong2022_roi_fc.py (this step) -> 05_01 gradient computation -> 05_02 group atlas parcellation -> 06_xx presentation rendering -> 07_01 group matrix overview.

Inputs:
- Command-line arguments:
  - --shared-root: Path to shared reference outputs (default: `outputs_core/inputs/hcp7t_hippocampus/kong2022/shared_inputs/routeb_151`).
  - --surface-root: Path to shared BOLD surfaces (default: `outputs_core/inputs/hcp7t_hippocampus/kong2022/shared_inputs/surface_151`).
  - --subjects-manifest: Path to QC-passed subject list manifest (default: `outputs_core/inputs/hcp7t_hippocampus/kong2022/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt`).
  - --subjects: Space-separated list of explicit subject IDs.
  - --smoothings: Smoothing levels to process: "2mm", "4mm".
  - --hemis: Hemispheres to process: "L", "R".
  - --jobs: Subject-level process concurrency cap (default: 1, hard-capped at 10).
- Physical inputs:
  - Resampled BOLD timeseries: `{surface_root}/sub-<subject>/surface/{smoothing}/sub-<subject>_hemi-{hemi}_timeseries.npy`.
  - BOLD active mask: `{surface_root}/sub-<subject>/surface/tsnr_gate/sub-<subject>_hemi-{hemi}_valid_mask.npy`.
  - Reference timeseries: `{shared_root}/sub-<subject>/reference/kong2022/cortex_roi_parcel_timeseries.npy`.
  - Reference parcels TSV: `{shared_root}/sub-<subject>/reference/kong2022/cortex_roi_parcels.tsv`.

Outputs:
- Command-line output / file outputs:
  - Complete vertex-to-ROI functional connectivity matrix: `{shared_root}/sub-<subject>/fc/kong2022/{smoothing}/hemi_{hemi}/hipp_vertex_to_network_fc.npy`.
  - Active-only vertex-to-ROI functional connectivity matrix: `{shared_root}/sub-<subject>/fc/kong2022/{smoothing}/hemi_{hemi}/hipp_vertex_to_network_fc_active.npy`.
  - Run manifest: `{shared_root}/sub-<subject>/fc/kong2022/{smoothing}/hemi_{hemi}/fc_summary.json`.
"""
import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from lib.output_layout import DEFAULT_KONG2022_MANIFEST_ROOT, DEFAULT_KONG2022_ROOT
from lib.workflow_utils import capped_jobs, utc_now_iso, write_json

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROUTE_ROOT = DEFAULT_KONG2022_ROOT
DEFAULT_SHARED_INPUTS_ROOT = DEFAULT_ROUTE_ROOT / "shared_inputs"
DEFAULT_SHARED_ROOT = DEFAULT_SHARED_INPUTS_ROOT / "routeb_151"
DEFAULT_SURFACE_ROOT = DEFAULT_SHARED_INPUTS_ROOT / "surface_151"
DEFAULT_SUBJECTS = DEFAULT_KONG2022_MANIFEST_ROOT / "hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt"
ATLAS = "kong2022"
SMOOTHINGS = ["2mm", "4mm"]
HEMIS = ["L", "R"]
RUN_IDS = [1, 2, 3, 4]


def load_subjects(path: Path) -> list[str]:
    subjects: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s:
            subjects.append(s.replace("sub-", ""))
    if not subjects:
        raise RuntimeError(f"No subjects found in {path}")
    return subjects


def read_roi_names(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise RuntimeError(f"Empty ROI TSV: {path}")
    return [str(row["parcel_name"]) for row in rows]


def load_run_segments(subj_root: Path) -> list[dict[str, int]]:
    segs: list[dict[str, int]] = []
    for rid in RUN_IDS:
        p = subj_root / "reference" / "common" / "runs" / f"run-{rid}" / "reference_summary.json"
        obj = json.loads(p.read_text(encoding="utf-8"))
        segs.append(
            {
                "run_id": int(rid),
                "run_start": int(obj["run_start"]),
                "run_stop": int(obj["run_stop"]),
                "run_length": int(obj["run_length"]),
            }
        )
    return segs


def corrcoef_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=np.float32)
    y = np.asarray(b, dtype=np.float32)
    x_mu = np.nanmean(x, axis=1, keepdims=True)
    y_mu = np.nanmean(y, axis=1, keepdims=True)
    x_sd = np.nanstd(x, axis=1, keepdims=True, ddof=1)
    y_sd = np.nanstd(y, axis=1, keepdims=True, ddof=1)
    x_z = (x - x_mu) / np.clip(x_sd, 1e-12, None)
    y_z = (y - y_mu) / np.clip(y_sd, 1e-12, None)
    x_z[~np.isfinite(x_z)] = 0.0
    y_z[~np.isfinite(y_z)] = 0.0
    corr = (x_z @ y_z.T) / float(max(1, x.shape[1] - 1))
    corr = np.clip(corr, -0.999999, 0.999999)
    corr[~np.isfinite(corr)] = 0.0
    return corr.astype(np.float32)


def compact_active_vertices(full: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    return np.asarray(full[valid_mask, :], dtype=np.float32)


def materialize_fc(shared_root: Path, surface_root: Path, subject: str, smoothing: str, hemi: str) -> None:
    subj_root = shared_root / f"sub-{subject}"
    surf_subj_root = surface_root / f"sub-{subject}"
    fc_dir = subj_root / "fc" / ATLAS / smoothing / f"hemi_{hemi}"
    fc_dir.mkdir(parents=True, exist_ok=True)

    ts = np.load(surf_subj_root / "surface" / smoothing / f"sub-{subject}_hemi-{hemi}_timeseries.npy").astype(np.float32)
    valid_mask = np.load(surf_subj_root / "surface" / "tsnr_gate" / f"sub-{subject}_hemi-{hemi}_valid_mask.npy").astype(bool)
    roi_ts = np.load(subj_root / "reference" / ATLAS / "cortex_roi_parcel_timeseries.npy").astype(np.float32)
    roi_names = read_roi_names(subj_root / "reference" / ATLAS / "cortex_roi_parcels.tsv")

    if ts.ndim != 2 or roi_ts.ndim != 2:
        raise ValueError(
            f"Expected 2D inputs for sub-{subject} {smoothing} hemi-{hemi}: ts={ts.shape} roi={roi_ts.shape}"
        )
    if ts.shape[1] != roi_ts.shape[1]:
        raise RuntimeError(
            f"Time mismatch for sub-{subject} {smoothing} hemi-{hemi}: ts={ts.shape} roi={roi_ts.shape}"
        )
    if roi_ts.shape[0] != len(roi_names):
        raise RuntimeError(
            f"ROI TSV mismatch for sub-{subject}: ts_rows={roi_ts.shape[0]} roi_names={len(roi_names)}"
        )

    pearson_valid = corrcoef_rows(ts[valid_mask, :], roi_ts)
    pearson_full = np.full((ts.shape[0], roi_ts.shape[0]), np.nan, dtype=np.float32)
    pearson_full[valid_mask, :] = pearson_valid
    np.save(fc_dir / "hipp_vertex_to_network_fc.npy", pearson_full.astype(np.float32))
    np.save(fc_dir / "hipp_vertex_to_network_fc_active.npy", compact_active_vertices(pearson_full, valid_mask))

    run_labels: list[str] = []
    run_lengths: list[int] = []
    run_segments = load_run_segments(subj_root)
    for seg in run_segments:
        rid = int(seg["run_id"])
        run_start = int(seg["run_start"])
        run_stop = int(seg["run_stop"])
        run_len = int(seg["run_length"])
        run_roi = roi_ts[:, run_start:run_stop].astype(np.float32, copy=False)
        if run_roi.shape[1] != run_len:
            raise RuntimeError(
                f"Run ROI slice mismatch for sub-{subject} run-{rid}: run_roi={run_roi.shape} expected_len={run_len}"
            )
        run_labels.append(f"run-{rid}")
        run_lengths.append(run_len)
        run_ts = ts[:, run_start:run_stop].astype(np.float32, copy=False)
        if run_ts.shape[1] != run_len:
            raise RuntimeError(
                f"Run slice mismatch for sub-{subject} {smoothing} hemi-{hemi} run-{rid}: "
                f"slice={run_ts.shape} expected_len={run_len}"
            )
        if run_roi.shape[0] != roi_ts.shape[0]:
            raise RuntimeError(
                f"Run ROI row mismatch for sub-{subject} run-{rid}: run_roi={run_roi.shape} full_roi={roi_ts.shape}"
            )
        run_dir = fc_dir / "runs" / f"run-{rid}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_pearson_valid = corrcoef_rows(run_ts[valid_mask, :], run_roi)
        np.save(run_dir / "hipp_vertex_to_network_fc_active.npy", run_pearson_valid.astype(np.float32))
    total_run_len = int(sum(run_lengths))
    if total_run_len != int(ts.shape[1]):
        raise RuntimeError(
            f"Run lengths do not sum to full time series for sub-{subject} {smoothing} hemi-{hemi}: "
            f"{total_run_len} vs {ts.shape[1]}"
        )

    write_json(
        fc_dir / "fc_summary.json",
        {
            "subject": subject,
            "atlas_slug": ATLAS,
            "smooth": smoothing,
            "hemi": hemi,
            "fc_shape": [int(pearson_full.shape[0]), int(pearson_full.shape[1])],
            "active_fc_shape": [int(pearson_valid.shape[0]), int(pearson_valid.shape[1])],
            "n_vertices_total": int(ts.shape[0]),
            "n_vertices_valid_high_tsnr": int(valid_mask.sum()),
            "n_vertices_invalid_initial": int(ts.shape[0] - valid_mask.sum()),
            "n_vertices_clustered": int(valid_mask.sum()),
            "n_timepoints": int(ts.shape[1]),
            "n_roi_targets_used": int(roi_ts.shape[0]),
            "run_labels": run_labels,
            "run_lengths": run_lengths,
            "fc_definition": "pearson_vertex_to_kong2022_roi_timeseries",
        },
    )
    write_json(
        fc_dir / "stage_manifest.json",
        {
            "stage": "shared_fc",
            "status": "done",
            "timestamp_utc": utc_now_iso(),
            "params": {
                "subject": subject,
                "atlas_slug": ATLAS,
                "smooth": smoothing,
                "hemi": hemi,
                "fc_definition": "pearson_vertex_to_kong2022_roi_timeseries",
            },
            "inputs": [
                str((subj_root / "reference" / ATLAS / "cortex_roi_parcel_timeseries.npy").resolve()),
                str((surf_subj_root / "surface" / smoothing / f"sub-{subject}_hemi-{hemi}_timeseries.npy").resolve()),
                str((surf_subj_root / "surface" / "tsnr_gate" / f"sub-{subject}_hemi-{hemi}_valid_mask.npy").resolve()),
            ],
            "outputs": [
                str((fc_dir / "hipp_vertex_to_network_fc.npy").resolve()),
                str((fc_dir / "hipp_vertex_to_network_fc_active.npy").resolve()),
                str((fc_dir / "fc_summary.json").resolve()),
            ],
        },
    )


def process_subject(
    subject: str,
    shared_root_str: str,
    surface_root_str: str,
    smoothings: list[str],
    hemis: list[str],
) -> dict[str, object]:
    shared_root = Path(shared_root_str)
    surface_root = Path(surface_root_str)
    n_fc = 0
    for smoothing in smoothings:
        for hemi in hemis:
            materialize_fc(shared_root, surface_root, subject, smoothing, hemi)
            n_fc += 1
    return {"subject": f"sub-{subject}", "fc_combos": int(n_fc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize Kong2022 hippocampal vertex-to-ROI FC caches.")
    parser.add_argument("--shared-root", default=str(DEFAULT_SHARED_ROOT))
    parser.add_argument("--surface-root", default=str(DEFAULT_SURFACE_ROOT))
    parser.add_argument("--subjects-manifest", default=str(DEFAULT_SUBJECTS))
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--smoothings", nargs="+", default=SMOOTHINGS, choices=SMOOTHINGS)
    parser.add_argument("--hemis", nargs="+", default=HEMIS, choices=HEMIS)
    parser.add_argument("--jobs", type=int, default=1, help="Subject-level parallel workers; hard-capped at 10.")
    args = parser.parse_args()

    shared_root = Path(args.shared_root).resolve()
    surface_root = Path(args.surface_root).resolve()
    subjects = [s.replace("sub-", "") for s in args.subjects] if args.subjects else load_subjects(
        Path(args.subjects_manifest).resolve()
    )
    jobs = capped_jobs(args.jobs, len(subjects))
    n_fc = 0
    if jobs == 1:
        for subject in subjects:
            obj = process_subject(subject, str(shared_root), str(surface_root), list(args.smoothings), list(args.hemis))
            n_fc += int(obj["fc_combos"])
            print(f"[done] {obj['subject']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = [
                ex.submit(process_subject, s, str(shared_root), str(surface_root), list(args.smoothings), list(args.hemis))
                for s in subjects
            ]
            for fut in as_completed(futs):
                obj = fut.result()
                n_fc += int(obj["fc_combos"])
                print(f"[done] {obj['subject']}", flush=True)
    print(json.dumps({"subjects": len(subjects), "fc_combos": int(n_fc), "jobs": int(jobs), "status": "ok"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

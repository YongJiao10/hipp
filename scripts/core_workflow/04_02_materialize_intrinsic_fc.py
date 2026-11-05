#!/usr/bin/env python3
"""
Process Step Description:
Materialize intrinsic functional connectivity. Computes the intrinsic functional connectivity matrix between vertices within the hippocampus itself.

Main Flow Path:
- 6th step in the numbered pipeline sequence (intrinsic functional connectivity materialization):
  01_01_run_hippunfold_t1w_local.py -> 01_02_check_hippunfold_outputs.py -> 02_01_prepare_group_inputs.py -> 02_02_build_qc_manifest.py -> 03_01/03_02 reference building -> 04_02_materialize_intrinsic_fc.py (this step) -> 05_01 gradient computation -> 05_02 group atlas parcellation -> 06_xx presentation rendering -> 07_01 group matrix overview.

Inputs:
- Command-line arguments:
  - --shared-root: Path to shared outputs root (default: `outputs_hcp/inputs/kong2022/shared_inputs/fixed`).
  - --surface-root: Path to shared BOLD surfaces (default: `outputs_hcp/inputs/kong2022/shared_inputs/surface`).
  - --subjects-manifest: Path to QC-passed subject list manifest (default: `outputs_hcp/inputs/kong2022/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt`).
  - --subjects: Space-separated list of explicit subject IDs.
  - --smoothings: Smoothing levels to process: "2mm", "4mm".
  - --hemis: Hemispheres to process: "L", "R".
  - --jobs: Subject-level process concurrency cap (default: 1).
- Physical inputs:
  - Resampled BOLD timeseries: `{surface_root}/sub-<subject>/surface/{smoothing}/sub-<subject>_hemi-{hemi}_timeseries.npy`.
  - BOLD active mask: `{surface_root}/sub-<subject>/surface/validity/sub-<subject>_hemi-{hemi}_valid_mask.npy`.

Outputs:
- Command-line output / file outputs:
  - Active-only vertex-to-vertex hippocampal functional connectivity matrix: `{shared_root}/sub-<subject>/fc/intrinsic/{smoothing}/hemi_{hemi}/hipp_vertex_to_vertex_fc_active.npy`.
  - Intrinsic FC step manifest: `{shared_root}/sub-<subject>/fc/intrinsic/{smoothing}/hemi_{hemi}/stage_manifest.json`.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TypedDict

import numpy as np
from lib.fc_math import corrcoef_rows
from lib.io_utils import load_run_segment_metadata, load_subject_ids
from lib.output_layout import DEFAULT_KONG2022_MANIFEST_ROOT, DEFAULT_KONG2022_ROOT
from lib.workflow_utils import capped_jobs, utc_now_iso, write_json


class SubjectResult(TypedDict):
    subject: str
    combos: int


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROUTE_ROOT = DEFAULT_KONG2022_ROOT
DEFAULT_SHARED_INPUTS_ROOT = DEFAULT_ROUTE_ROOT / "shared_inputs"
DEFAULT_SHARED_ROOT = DEFAULT_SHARED_INPUTS_ROOT / "fixed"
DEFAULT_SURFACE_ROOT = DEFAULT_SHARED_INPUTS_ROOT / "surface"
DEFAULT_SUBJECTS = DEFAULT_KONG2022_MANIFEST_ROOT / "hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt"
SMOOTHINGS = ["2mm", "4mm"]
HEMIS = ["L", "R"]



def materialize_subject(
    shared_root: Path,
    surface_root: Path,
    subject: str,
    smoothings: list[str],
    hemis: list[str],
    reference_root: Path | None = None,
    mask_root: Path | None = None,
) -> SubjectResult:
    reference_root = reference_root or shared_root
    mask_root = mask_root or surface_root
    run_metadata = reference_root / f"sub-{subject}" / "reference" / "kong2022" / "reference_summary.json"
    run_ids, run_lengths = load_run_segment_metadata(run_metadata)
    n_combo = 0
    for smoothing in smoothings:
        for hemi in hemis:
            subj = shared_root / f"sub-{subject}"
            surf_subj = surface_root / f"sub-{subject}"
            ts_path = surf_subj / "surface" / smoothing / f"sub-{subject}_hemi-{hemi}_timeseries.npy"
            mask_path = mask_root / f"sub-{subject}" / "surface" / "validity" / f"sub-{subject}_hemi-{hemi}_valid_mask.npy"
            ts = np.load(ts_path).astype(np.float32)
            valid = np.load(mask_path).astype(bool)
            active = ts[valid, :]
            fc = corrcoef_rows(active, active, run_lengths=run_lengths)
            np.fill_diagonal(fc, np.where(np.isfinite(np.diag(fc)), 1.0, np.nan))
            out_dir = subj / "fc" / "intrinsic" / smoothing / f"hemi_{hemi}"
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / "hipp_vertex_to_vertex_fc_active.npy", fc)
            offset = 0
            for rid, run_len in zip(run_ids, run_lengths, strict=True):
                run_ts = active[:, offset : offset + run_len]
                if run_ts.shape[1] != run_len:
                    raise RuntimeError(f"Run slice mismatch sub-{subject} {smoothing} hemi-{hemi} run-{rid}")
                run_dir = out_dir / "runs" / f"run-{rid}"
                run_dir.mkdir(parents=True, exist_ok=True)
                run_fc = corrcoef_rows(run_ts, run_ts)
                np.fill_diagonal(run_fc, np.where(np.isfinite(np.diag(run_fc)), 1.0, np.nan))
                np.save(run_dir / "hipp_vertex_to_vertex_fc_active.npy", run_fc)
                offset += run_len
            if offset != ts.shape[1]:
                raise RuntimeError(f"Run lengths do not sum to timeseries length for sub-{subject} {smoothing} hemi-{hemi}")
            write_json(
                out_dir / "stage_manifest.json",
                {
                    "stage": "t1only_intrinsic_vertex_to_vertex_fc",
                    "status": "done",
                    "timestamp_utc": utc_now_iso(),
                    "params": {"subject": subject, "smoothing": smoothing, "hemi": hemi},
                    "inputs": [str(ts_path.resolve()), str(mask_path.resolve()), str(run_metadata.resolve())],
                    "outputs": [str((out_dir / "hipp_vertex_to_vertex_fc_active.npy").resolve())],
                    "n_vertices_total": int(ts.shape[0]),
                    "n_vertices_active": int(active.shape[0]),
                    "n_timepoints": int(ts.shape[1]),
                    "run_ids": list(run_ids),
                    "run_lengths": list(run_lengths),
                },
            )
            n_combo += 1
    return {"subject": f"sub-{subject}", "combos": int(n_combo)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize T1only intrinsic hippocampal vertex-to-vertex FC caches.")
    parser.add_argument("--shared-root", default=str(DEFAULT_SHARED_ROOT))
    parser.add_argument("--surface-root", default=str(DEFAULT_SURFACE_ROOT))
    parser.add_argument("--reference-root", default=None, help="Read run metadata from this root; defaults to --shared-root.")
    parser.add_argument("--mask-root", default=None, help="Read hippocampal valid masks from this root; defaults to --surface-root.")
    parser.add_argument("--subjects-manifest", default=str(DEFAULT_SUBJECTS))
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--smoothings", nargs="+", default=SMOOTHINGS, choices=SMOOTHINGS)
    parser.add_argument("--hemis", nargs="+", default=HEMIS, choices=HEMIS)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()

    shared_root = Path(args.shared_root).resolve()
    surface_root = Path(args.surface_root).resolve()
    reference_root = Path(args.reference_root).resolve() if args.reference_root else shared_root
    mask_root = Path(args.mask_root).resolve() if args.mask_root else surface_root
    subjects = [s.replace("sub-", "") for s in args.subjects] if args.subjects else load_subject_ids(Path(args.subjects_manifest))
    jobs = capped_jobs(args.jobs, len(subjects))
    n_combo = 0
    if jobs == 1:
        for subject in subjects:
            obj = materialize_subject(shared_root, surface_root, subject, list(args.smoothings), list(args.hemis), reference_root=reference_root, mask_root=mask_root)
            n_combo += int(obj["combos"])
            print(f"[done] {obj['subject']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = [
                ex.submit(materialize_subject, shared_root, surface_root, s, list(args.smoothings), list(args.hemis), reference_root, mask_root)
                for s in subjects
            ]
            for fut in as_completed(futs):
                obj = fut.result()
                n_combo += int(obj["combos"])
                print(f"[done] {obj['subject']}", flush=True)
    print(json.dumps({"subjects": len(subjects), "intrinsic_combos": int(n_combo), "jobs": int(jobs), "status": "ok"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

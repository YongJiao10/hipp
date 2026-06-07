#!/usr/bin/env python3
from __future__ import annotations

"""
Process Step Description:
Build Kong2022 RouteA reference data. RouteA-specific step to generate reference template data for Kong2022 cortical networks.

Main Flow Path:
- 5th step in the numbered pipeline sequence (RouteA reference building for kong2022_routea):
  01_01_run_hippunfold_t1w_local.py -> 01_02_check_hippunfold_outputs.py -> 02_01_prepare_group_inputs.py -> 02_02_build_qc_manifest.py -> 03_02_build_kong2022_routea_reference.py (this step) -> 04_01/04_02 FC materialization -> 05_01 gradient computation -> 05_02 group atlas parcellation -> 06_xx presentation rendering -> 07_01 group matrix overview.

Inputs:
- Command-line arguments:
  - --shared-root: Path to output shared processed files (default: `outputs_hcp/inputs/hcp/kong2022/shared_inputs/routea_151`).
  - --route-a-root: RouteA specific reference outputs root (default: `outputs_hcp/inputs/hcp/kong2022/shared_inputs/reference_151`).
  - --data-root: Root directory containing raw structural BIDS folders (default: `data/hippunfold_input`).
  - --kong2022-dlabel: Path to standard Kong2022 dlabel.nii cortical atlas file.
  - --subjects-manifest: Path to QC-passed subject list manifest (default: `outputs_hcp/inputs/hcp/kong2022/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt`).
  - --subjects: Space-separated list of explicit subject IDs.
  - --jobs: Thread-pool concurrency cap (default: 4, hard-capped at 10).
  - --resume: Skip already completed subjects (default: True).
  - --overwrite: Overwrite existing files (default: False).
- Physical inputs:
  - Concat fMRI dtseries file: `{data_root}/sub-<subject>/func/sub-<subject>_task-rest_run-concat.dtseries.nii`.
  - Kong2022 standard cortical dlabel atlas.

Outputs:
- Command-line output / file outputs:
  - RouteA cortical network reference parcel timeseries: `{shared_root}/sub-<subject>/reference/kong2022/cortex_roi_parcel_timeseries.npy`.
  - RouteA cortical network parcels description table: `{shared_root}/sub-<subject>/reference/kong2022/cortex_roi_parcels.tsv`.
  - Reference generation run manifest: `{shared_root}/sub-<subject>/reference/kong2022/runs/run-<run_id>/reference_summary.json`.
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from lib.output_layout import DEFAULT_CORTEX_ROUTEA_ROOT, DEFAULT_KONG2022_MANIFEST_ROOT, DEFAULT_KONG2022_ROUTEA_ROOT
from lib.io_utils import load_subject_ids


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_EXE = sys.executable or "python"
EXTRACT_MODULE = "scripts.core_workflow.lib.extract_cortex_kong2022_route_a_roi_timeseries"
DEFAULT_SHARED_ROOT = DEFAULT_KONG2022_ROUTEA_ROOT
DEFAULT_ROUTE_A_ROOT = DEFAULT_CORTEX_ROUTEA_ROOT
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "hippunfold_input"
DEFAULT_MANIFEST = DEFAULT_KONG2022_MANIFEST_ROOT / "hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt"
MAX_CONCURRENCY = 10
SOURCE_TAG = "route_a_kong2022_areal_mshbm"
FORBIDDEN_ROUTEB_TOKEN = "kong2022_route_b"


def validate_routea_path(path: Path, arg_name: str) -> None:
    if FORBIDDEN_ROUTEB_TOKEN in path.parts:
        raise RuntimeError(
            f"RouteA compliance violation: {arg_name} points into a RouteB tree: {path}. "
            "Use route-agnostic/shared kong2022 manifests and RouteA-compatible roots."
        )





def subject_is_done(shared_root: Path, subject: str) -> bool:
    ref_dir = shared_root / f"sub-{subject}" / "reference" / "kong2022"
    summary = ref_dir / "reference_summary.json"
    stage = ref_dir / "stage_manifest.json"
    if not summary.exists() or not stage.exists():
        return False
    try:
        payload = json.loads(summary.read_text(encoding="utf-8"))
    except Exception:
        return False
    return str(payload.get("reference_label_source", "")) == SOURCE_TAG


def run_subject(
    *,
    subject: str,
    shared_root: Path,
    route_a_root: Path,
    data_root: Path,
    kong2022_dlabel: Path | None,
    overwrite: bool,
) -> tuple[str, bool, str]:
    cmd = [
        PYTHON_EXE,
        "-m",
        EXTRACT_MODULE,
        "--subject",
        str(subject),
        "--shared-root",
        str(shared_root),
        "--route-a-root",
        str(route_a_root),
        "--dtseries",
        str(data_root / f"sub-{subject}" / "func" / f"sub-{subject}_task-rest_run-concat.dtseries.nii"),
    ]
    if kong2022_dlabel is not None:
        cmd.extend(["--kong2022-dlabel", str(kong2022_dlabel)])
    if overwrite:
        cmd.append("--overwrite")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout)[-4000:]
        return subject, False, tail
    return subject, True, "ok"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch bridge Route A individualized labels into Route B-compatible reference/kong2022 arrays."
    )
    parser.add_argument("--shared-root", default=str(DEFAULT_SHARED_ROOT))
    parser.add_argument("--route-a-root", default=str(DEFAULT_ROUTE_A_ROOT))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--kong2022-dlabel", default=None)
    parser.add_argument("--subjects-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--subjects", nargs="*", default=None, help="Optional explicit subject IDs.")
    parser.add_argument("--jobs", type=int, default=4, help="Thread-pool concurrency (hard-capped at 10).")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()

    shared_root = Path(args.shared_root).resolve()
    route_a_root = Path(args.route_a_root).resolve()
    data_root = Path(args.data_root).resolve()
    kong2022_dlabel = Path(args.kong2022_dlabel).resolve() if args.kong2022_dlabel else None
    subjects_manifest = Path(args.subjects_manifest).resolve()
    validate_routea_path(shared_root, "--shared-root")
    validate_routea_path(subjects_manifest, "--subjects-manifest")
    subjects = (
        [str(s).replace("sub-", "") for s in args.subjects]
        if args.subjects
        else load_subject_ids(subjects_manifest)
    )

    if args.resume:
        todo = [s for s in subjects if not subject_is_done(shared_root, s)]
    else:
        todo = list(subjects)

    print(f"[info] total={len(subjects)} todo={len(todo)} resume={bool(args.resume)}", flush=True)
    if not todo:
        print("[info] all subjects already done", flush=True)
        return 0

    jobs = max(1, min(int(args.jobs), MAX_CONCURRENCY, len(todo)))
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futs = {
            pool.submit(
                run_subject,
                subject=s,
                shared_root=shared_root,
                route_a_root=route_a_root,
                data_root=data_root,
                kong2022_dlabel=kong2022_dlabel,
                overwrite=bool(args.overwrite),
            ): s
            for s in todo
        }
        done = 0
        for fut in as_completed(futs):
            done += 1
            subject, ok, msg = fut.result()
            if ok:
                print(f"[done {done}/{len(todo)}] sub-{subject}", flush=True)
            else:
                failed.append(subject)
                print(f"[FAIL {done}/{len(todo)}] sub-{subject}\n{msg}", flush=True)

    if failed:
        print(f"[error] failed_subjects={len(failed)} {failed}", flush=True)
        return 1
    print(f"[ok] completed={len(todo)} jobs={jobs}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

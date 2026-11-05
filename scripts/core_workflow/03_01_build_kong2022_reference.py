#!/usr/bin/env python3
"""
Process Step Description:
Build Kong2022 reference data. Generates Kong2022 parcellation reference files for cortical network mapping, providing a standard template for establishing hippocampal-cortical connectivity.

Main Flow Path:
- 5th step in the numbered pipeline sequence (shared reference building for kong2022):
  01_01_run_hippunfold_t1w_local.py -> 01_02_check_hippunfold_outputs.py -> 02_01_prepare_group_inputs.py -> 02_02_build_qc_manifest.py -> 03_01_build_kong2022_reference.py (this step, or 03_02 for RouteA specific reference) -> 04_01/04_02 FC materialization -> 05_01 gradient computation -> 05_02 group atlas parcellation -> 06_xx presentation rendering -> 07_01 group matrix overview.

Inputs:
- Command-line arguments:
  - --shared-root: Path to output shared processed files (default: `outputs_hcp/inputs/kong2022/shared_inputs/fixed`).
  - --data-root: Root directory containing raw structural BIDS folders (default: `data/hippunfold_input`).
  - --subjects-manifest: Path to QC-passed subject list manifest (default: `outputs_hcp/inputs/kong2022/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt`).
  - --subjects: Space-separated list of explicit subject IDs.
  - --jobs: Thread-pool concurrency cap (default: 4, hard-capped at 10).
  - --resume: Skip already completed subjects (default: True).
  - --kong2022-dlabel: Path to standard Kong2022 dlabel.nii cortical atlas file.
  - --cross-atlas-merge-json: Path to JSON mapping networks across atlases.
- Physical inputs:
  - A legacy single-file fMRI dtseries or an exact per-subject JSON run-artifact
    manifest. The manifest declares the ordered dtseries paths, run IDs, sizes,
    and SHA256 receipts; run concatenation remains in memory.
  - Kong2022 standard cortical dlabel atlas.

Outputs:
- Command-line output / file outputs:
  - Cortical network reference parcel timeseries: `{shared_root}/sub-<subject>/reference/kong2022/cortex_roi_parcel_timeseries.npy`.
  - Cortical network parcels description table: `{shared_root}/sub-<subject>/reference/kong2022/cortex_roi_parcels.tsv`.
  - Reference generation run manifest: `{shared_root}/sub-<subject>/reference/kong2022/runs/run-<run_id>/reference_summary.json`.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from lib.extract_cortex_kong2022_routeb_roi_timeseries import build_kong2022_reference
from lib.io_utils import load_subject_ids
from lib.task_quality import require_cortex_reference_policy
from lib.kong2022_reference import ReferenceRequest, resolve_batch_subject_dtseries
from lib.output_layout import (
    DEFAULT_KONG2022_MANIFEST_ROOT,
    DEFAULT_KONG2022_ROUTEB_ROOT,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHARED_ROOT = DEFAULT_KONG2022_ROUTEB_ROOT
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "hippunfold_input"
DEFAULT_DTSERIES_TEMPLATE = "{data_root}/{sub_subject}/func/{sub_subject}_task-rest_run-concat.dtseries.nii"
DEFAULT_MANIFEST = DEFAULT_KONG2022_MANIFEST_ROOT / "hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt"
MAX_CONCURRENCY = 10


def subject_is_done(shared_root: Path, subject: str) -> bool:
    ref_dir = shared_root / f"sub-{subject}" / "reference" / "kong2022"
    if not (ref_dir / "stage_manifest.json").is_file():
        return False
    try:
        require_cortex_reference_policy(ref_dir, dataset="hcp", task="rest")
    except ValueError:
        return False
    return True


def run_subject(
    *,
    subject: str,
    shared_root: Path,
    data_root: Path,
    dtseries_template: str,
    run_artifact_manifest_template: str | None,
    kong2022_dlabel: Path | None,
    cross_atlas_merge_json: Path | None,
    run_metadata_template: str | None = None,
) -> tuple[str, bool, str]:
    options: dict[str, object] = {"subject": subject, "shared_root": shared_root}
    try:
        dt_options, _expected = resolve_batch_subject_dtseries(
            subject=subject,
            data_root=data_root,
            dtseries_template=dtseries_template,
            run_artifact_manifest_template=run_artifact_manifest_template,
            run_metadata_template=run_metadata_template,
        )
    except (FileNotFoundError, ValueError, KeyError, TypeError) as exc:
        return subject, False, str(exc)
    options.update(dt_options)
    if kong2022_dlabel is not None:
        options["kong2022_dlabel"] = kong2022_dlabel
    if cross_atlas_merge_json is not None:
        options["cross_atlas_merge_json"] = cross_atlas_merge_json
    try:
        build_kong2022_reference(request=ReferenceRequest(**options))  # type: ignore[arg-type]
    except Exception as exc:
        return subject, False, str(exc)[-4000:]
    return subject, True, "ok"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch extract Kong2022 cortex ROI and canonical7 timeseries into the unsuffixed shared reference directory."
    )
    parser.add_argument("--shared-root", default=str(DEFAULT_SHARED_ROOT))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument(
        "--dtseries-template",
        default=DEFAULT_DTSERIES_TEMPLATE,
        help=(
            "Per-subject dtseries template. Variables: {data_root}, {subject}, "
            "{sub_subject}. Default preserves the legacy HCP layout."
        ),
    )
    parser.add_argument(
        "--run-artifact-manifest-template",
        default=None,
        help=(
            "Per-subject JSON receipt declaring exact ordered run dtseries files, "
            "run IDs, sizes, and SHA256 values. Variables: {data_root}, {subject}, {sub_subject}."
        ),
    )
    parser.add_argument(
        "--run-metadata-template",
        default=None,
        help="Per-subject JSON template containing explicit run IDs and lengths for single-CIFTI inputs.",
    )
    parser.add_argument("--subjects-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--subjects", nargs="*", default=None, help="Optional explicit subject IDs.")
    parser.add_argument("--jobs", type=int, default=4, help="Thread-pool concurrency (hard-capped at 10).")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--kong2022-dlabel", default=None)
    parser.add_argument("--cross-atlas-merge-json", default=None)
    args = parser.parse_args()

    shared_root = Path(args.shared_root).resolve()
    data_root = Path(args.data_root).resolve()
    subjects = (
        [str(s).replace("sub-", "") for s in args.subjects]
        if args.subjects
        else load_subject_ids(Path(args.subjects_manifest).resolve())
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
    dlabel = Path(args.kong2022_dlabel).resolve() if args.kong2022_dlabel else None
    merge_json = Path(args.cross_atlas_merge_json).resolve() if args.cross_atlas_merge_json else None
    dtseries_template = args.dtseries_template

    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futs = {
            pool.submit(
                run_subject,
                subject=s,
                shared_root=shared_root,
                data_root=data_root,
                dtseries_template=dtseries_template,
                run_artifact_manifest_template=args.run_artifact_manifest_template,
                run_metadata_template=args.run_metadata_template,
                kong2022_dlabel=dlabel,
                cross_atlas_merge_json=merge_json,
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

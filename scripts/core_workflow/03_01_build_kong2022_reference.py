#!/usr/bin/env python3
from __future__ import annotations

"""
Process Step Description:
Build Kong2022 reference data. Generates Kong2022 parcellation reference files for cortical network mapping, providing a standard template for establishing hippocampal-cortical connectivity.

Main Flow Path:
- 5th step in the numbered pipeline sequence (shared reference building for kong2022):
  01_01_run_hippunfold_t1w_local.py -> 01_02_check_hippunfold_outputs.py -> 02_01_prepare_group_inputs.py -> 02_02_build_qc_manifest.py -> 03_01_build_kong2022_reference.py (this step, or 03_02 for RouteA specific reference) -> 04_01/04_02 FC materialization -> 05_01 gradient computation -> 05_02 group atlas parcellation -> 06_xx presentation rendering -> 07_01 group matrix overview.

Inputs:
- Command-line arguments:
  - --shared-root: Path to output shared processed files (default: `outputs_hcp/inputs/hcp/kong2022/shared_inputs/routea`).
  - --data-root: Root directory containing raw structural BIDS folders (default: `data/hippunfold_input`).
  - --subjects-manifest: Path to QC-passed subject list manifest (default: `outputs_hcp/inputs/hcp/kong2022/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt`).
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
import argparse
import hashlib
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from lib.output_layout import DEFAULT_KONG2022_MANIFEST_ROOT, DEFAULT_KONG2022_ROUTEA_ROOT
from lib.io_utils import is_persistent_concat_path, load_run_artifact_manifest, load_run_segment_metadata, load_subject_ids
from lib.path_templates import format_subject_path_template


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_EXE = sys.executable or "python"
EXTRACT_MODULE = "scripts.core_workflow.lib.extract_cortex_kong2022_routeb_roi_timeseries"
DEFAULT_SHARED_ROOT = DEFAULT_KONG2022_ROUTEA_ROOT
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "hippunfold_input"
DEFAULT_DTSERIES_TEMPLATE = "{data_root}/{sub_subject}/func/{sub_subject}_task-rest_run-concat.dtseries.nii"
DEFAULT_MANIFEST = DEFAULT_KONG2022_MANIFEST_ROOT / "hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt"
MAX_CONCURRENCY = 10



def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def migrate_tree_with_conflict_guard(legacy_dir: Path, common_dir: Path) -> tuple[int, int]:
    moved = 0
    shared_same = 0
    if not legacy_dir.exists():
        return moved, shared_same
    common_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(p for p in legacy_dir.rglob("*") if p.is_file()):
        rel = src.relative_to(legacy_dir)
        dst = common_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if file_sha256(src) != file_sha256(dst):
                raise RuntimeError(
                    f"Legacy/common conflict with different content: legacy={src} common={dst}"
                )
            shared_same += 1
            src.unlink()
            continue
        os.replace(str(src), str(dst))
        moved += 1
    for d in sorted((p for p in legacy_dir.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        d.rmdir()
    legacy_dir.rmdir()
    return moved, shared_same


def migrate_legacy_reference_to_common(shared_root: Path, subject: str) -> dict[str, int]:
    subj_ref = shared_root / f"sub-{subject}" / "reference"
    legacy_root = subj_ref / "kong2022"
    common_root = subj_ref / "common"
    moved_a, same_a = migrate_tree_with_conflict_guard(legacy_root / "tsnr_gate", common_root / "tsnr_gate")
    moved_b, same_b = migrate_tree_with_conflict_guard(legacy_root / "runs", common_root / "runs")
    return {
        "moved_files": int(moved_a + moved_b),
        "deduped_same_files": int(same_a + same_b),
    }


def subject_is_done(shared_root: Path, subject: str) -> bool:
    ref_dir = shared_root / f"sub-{subject}" / "reference" / "kong2022"
    return (ref_dir / "reference_summary.json").exists() and (ref_dir / "stage_manifest.json").exists()


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
    cmd = [
        PYTHON_EXE,
        "-m",
        EXTRACT_MODULE,
        "--subject",
        str(subject),
        "--shared-root",
        str(shared_root),
    ]
    if run_artifact_manifest_template:
        manifest_path = Path(
            format_subject_path_template(
                run_artifact_manifest_template,
                data_root=data_root,
                subject=subject,
            )
        ).resolve()
        try:
            records = load_run_artifact_manifest(
                manifest_path,
                subject=subject,
                required_artifacts=("dtseries",),
            )
        except (FileNotFoundError, ValueError, KeyError, TypeError) as exc:
            return subject, False, str(exc)
        dtseries_paths = [record["artifacts"]["dtseries"] for record in records]
        run_ids = [str(record["run_id"]) for record in records]
        cmd.extend(
            [
                "--run-dtseries",
                *(str(path) for path in dtseries_paths),
                "--run-ids",
                *run_ids,
                "--run-artifact-manifest",
                str(manifest_path),
            ]
        )
    else:
        if run_metadata_template is None:
            return subject, False, (
                "Single CIFTI input requires --run-metadata-template"
            )
        dtseries = format_subject_path_template(
            dtseries_template,
            data_root=data_root,
            subject=subject,
        )
        if is_persistent_concat_path(dtseries):
            return subject, False, (
                "Iowa persistent concat input is not allowed; pass "
                "--run-artifact-manifest-template for canonical run-level files: "
                f"{dtseries}"
            )
        metadata_path = Path(
            format_subject_path_template(
                run_metadata_template,
                data_root=data_root,
                subject=subject,
            )
        ).resolve()
        try:
            load_run_segment_metadata(metadata_path)
        except (FileNotFoundError, ValueError, KeyError, TypeError) as exc:
            return subject, False, str(exc)
        cmd.extend(
            [
                "--dtseries",
                str(dtseries),
                "--run-metadata-template",
                str(metadata_path),
            ]
        )
    if kong2022_dlabel is not None:
        cmd.extend(["--kong2022-dlabel", str(kong2022_dlabel)])
    if cross_atlas_merge_json is not None:
        cmd.extend(["--cross-atlas-merge-json", str(cross_atlas_merge_json)])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout)[-4000:]
        return subject, False, tail
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
    migration_moved = 0
    migration_deduped = 0
    for s in subjects:
        migrated = migrate_legacy_reference_to_common(shared_root, s)
        migration_moved += int(migrated["moved_files"])
        migration_deduped += int(migrated["deduped_same_files"])
    if migration_moved or migration_deduped:
        print(
            f"[migrate] moved_files={migration_moved} deduped_same_files={migration_deduped} target=reference/common",
            flush=True,
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

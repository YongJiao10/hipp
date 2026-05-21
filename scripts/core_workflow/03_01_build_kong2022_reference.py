#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from lib.output_layout import DEFAULT_KONG2022_MANIFEST_ROOT, DEFAULT_KONG2022_ROUTEB_ROOT


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_EXE = sys.executable or "python"
EXTRACT_MODULE = "scripts.core_workflow.lib.extract_cortex_kong2022_roi_timeseries"
DEFAULT_SHARED_ROOT = DEFAULT_KONG2022_ROUTEB_ROOT
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "hippunfold_input"
DEFAULT_MANIFEST = DEFAULT_KONG2022_MANIFEST_ROOT / "hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt"
MAX_CONCURRENCY = 10


def load_subjects_from_manifest(path: Path) -> list[str]:
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        out.append(s.replace("sub-", ""))
    if not out:
        raise RuntimeError(f"No subjects found in manifest: {path}")
    return out


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
    kong2022_dlabel: Path | None,
    cross_atlas_merge_json: Path | None,
) -> tuple[str, bool, str]:
    cmd = [
        PYTHON_EXE,
        "-m",
        EXTRACT_MODULE,
        "--subject",
        str(subject),
        "--shared-root",
        str(shared_root),
        "--dtseries",
        str(data_root / f"sub-{subject}" / "func" / f"sub-{subject}_task-rest_run-concat.dtseries.nii"),
    ]
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
        description="Batch extract Kong2022 cortex ROI and canonical-8 timeseries into shared reference directories."
    )
    parser.add_argument("--shared-root", default=str(DEFAULT_SHARED_ROOT))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
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
        else load_subjects_from_manifest(Path(args.subjects_manifest).resolve())
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

    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futs = {
            pool.submit(
                run_subject,
                subject=s,
                shared_root=shared_root,
                data_root=data_root,
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

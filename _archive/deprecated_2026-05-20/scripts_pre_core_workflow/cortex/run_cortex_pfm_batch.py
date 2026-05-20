#!/usr/bin/env python3
"""Batch runner for cortex PFM + ROI component derivation across subjects.

Runs run_cortex_pfm_subject.py then derive_cortex_roi_components.py for each
subject. Max concurrency: 10 (CLAUDE.md hard cap).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_EXE = sys.executable or "python"
RUN_PFM = REPO_ROOT / "scripts" / "cortex" / "run_cortex_pfm_subject.py"
DERIVE_ROI = REPO_ROOT / "scripts" / "cortex" / "derive_cortex_roi_components.py"
DATA_ROOT = REPO_ROOT / "data" / "hippunfold_input"
DEFAULT_OUT_ROOT = REPO_ROOT / "outputs_migration" / "cortex_pfm"
DEFAULT_MANIFEST = REPO_ROOT / "manifests" / "hcp_7t_hippocampus_fmri_qc_pass_153.txt"
METHODS = ["Lynch2024", "Kong2019"]
MAX_CONCURRENCY = 10


def subject_is_done(subject: str, out_root: Path) -> bool:
    for method_slug in ["lynch2024", "kong2019"]:
        stats = out_root / f"sub-{subject}" / method_slug / "roi_components" / "roi_component_stats.json"
        if not stats.exists():
            return False
    return True


def run_subject(subject: str, out_root: Path) -> tuple[str, bool, str]:
    anat = DATA_ROOT / f"sub-{subject}" / "anat"
    func = DATA_ROOT / f"sub-{subject}" / "func"
    dtseries = func / f"sub-{subject}_task-rest_run-concat.dtseries.nii"

    pfm_cmd = [
        PYTHON_EXE, str(RUN_PFM),
        "--subject", subject,
        "--dtseries", str(dtseries),
        "--left-midthickness", str(anat / f"sub-{subject}_hemi-L_space-fsLR_den-32k_desc-MSMAll_midthickness.surf.gii"),
        "--right-midthickness", str(anat / f"sub-{subject}_hemi-R_space-fsLR_den-32k_desc-MSMAll_midthickness.surf.gii"),
        "--left-inflated", str(anat / f"sub-{subject}_hemi-L_space-fsLR_den-32k_desc-MSMAll_inflated.surf.gii"),
        "--right-inflated", str(anat / f"sub-{subject}_hemi-R_space-fsLR_den-32k_desc-MSMAll_inflated.surf.gii"),
        "--sulc-dscalar", str(anat / f"sub-{subject}_space-fsLR_den-32k_desc-MSMAll_sulc.dscalar.nii"),
        "--methods", *METHODS,
        "--out-root", str(out_root),
    ]
    proc = subprocess.run(pfm_cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return subject, False, f"run_cortex_pfm_subject failed:\n{proc.stderr[-2000:]}"

    for method in METHODS:
        derive_cmd = [
            PYTHON_EXE, str(DERIVE_ROI),
            "--subject", subject,
            "--method", method,
            "--out-root", str(out_root),
        ]
        proc = subprocess.run(derive_cmd, capture_output=True, text=True,
                              cwd=str(REPO_ROOT / "scripts" / "cortex"))
        if proc.returncode != 0:
            return subject, False, f"derive_cortex_roi_components failed ({method}):\n{proc.stderr[-2000:]}"

    return subject, True, "ok"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--concurrency", type=int, default=MAX_CONCURRENCY)
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()

    out_root = Path(args.out_root).resolve()
    subjects = Path(args.manifest).read_text().splitlines()
    subjects = [s.strip() for s in subjects if s.strip()]

    if args.resume:
        todo = [s for s in subjects if not subject_is_done(s, out_root)]
        print(f"[info] {len(subjects)} subjects total, {len(todo)} need cortex_pfm", flush=True)
    else:
        todo = subjects

    if not todo:
        print("[info] all subjects done", flush=True)
        return 0

    concurrency = min(args.concurrency, MAX_CONCURRENCY)
    failed: list[str] = []
    done = 0

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(run_subject, s, out_root): s for s in todo}
        for fut in as_completed(futures):
            subject, ok, msg = fut.result()
            done += 1
            if ok:
                print(f"[done {done}/{len(todo)}] {subject}", flush=True)
            else:
                print(f"[FAIL {done}/{len(todo)}] {subject}: {msg}", flush=True)
                failed.append(subject)

    if failed:
        print(f"\n[error] {len(failed)} subjects failed: {failed}", flush=True)
        return 1
    print(f"\n[info] all {len(todo)} subjects completed successfully", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

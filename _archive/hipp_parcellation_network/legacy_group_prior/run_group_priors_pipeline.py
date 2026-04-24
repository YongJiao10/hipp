#!/usr/bin/env python3
"""Full pipeline: wait for cortex_pfm → experimental subject batch → group priors.

Stage 1: waits for cortex_pfm batch PID to finish (or skips if already done)
Stage 2: runs experimental subject batch for all 153 QC subjects, parallel
         across subjects (max 4), sequential across combos per subject
Stage 3: runs run_group_prior_fastpfm.py

Usage:
    python scripts/run_group_priors_pipeline.py --cortex-pfm-pid <PID>
    python scripts/run_group_priors_pipeline.py  # skip wait, assume cortex_pfm done
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable or str(REPO_ROOT / ".venv" / "bin" / "python")
MANIFEST = REPO_ROOT / "manifests" / "hcp_7t_hippocampus_fmri_qc_pass_153.txt"
RUN_BATCH = REPO_ROOT / "scripts" / "hipp_parcellation_network" / "run_batch.py"
RUN_GROUP_PRIOR = REPO_ROOT / "scripts" / "hipp_parcellation_network" / "run_group_prior_fastpfm.py"
OUT_ROOT = REPO_ROOT / "outputs_migration" / "hipp_functional_parcellation_network"
PRESENT_DIR = REPO_ROOT / "present_network_migration"
GROUP_OUT_ROOT = REPO_ROOT / "outputs_migration" / "hipp_group_prior_fastpfm"
LOG_DIR = REPO_ROOT / "logs"

BRANCHES = ["network-spectral", "intrinsic-spectral"]
ATLASES = ["kong2019"]
MAX_CONCURRENCY = 4
WORKER_ENV_OVERRIDES = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    print(f"[{ts}] {msg}", flush=True)


def worker_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(WORKER_ENV_OVERRIDES)
    return env


def run(cmd: list[str], label: str) -> None:
    proc = subprocess.run(cmd, text=True, capture_output=True, env=worker_env())
    if proc.returncode != 0:
        raise RuntimeError(
            f"{label} failed (rc={proc.returncode}):\nSTDOUT:\n{proc.stdout[-3000:]}\nSTDERR:\n{proc.stderr[-3000:]}"
        )


def wait_for_pid(pid: int) -> None:
    log(f"waiting for cortex_pfm batch (PID {pid}) …")
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(15)
    log(f"cortex_pfm PID {pid} exited")


def subject_is_done(subject: str) -> bool:
    """True if all branch×atlas combos already have final_selection_summary.json."""
    for branch in BRANCHES:
        for atlas in ATLASES:
            p = OUT_ROOT / branch / atlas / f"sub-{subject}" / "final_selection_summary.json"
            if not p.exists():
                return False
    return True


def run_subject_all_combos(subject: str) -> tuple[str, bool, str]:
    """Run all branch×atlas combos for one subject sequentially."""
    for branch in BRANCHES:
        for atlas in ATLASES:
            summary = OUT_ROOT / branch / atlas / f"sub-{subject}" / "final_selection_summary.json"
            if summary.exists():
                continue
            cmd = [
                PYTHON,
                str(RUN_BATCH),
                "--k-selection-mode",
                "experimental",
                "--run-split-mode",
                "runwise",
                "--out-root",
                str(OUT_ROOT),
                "--present-dir",
                str(PRESENT_DIR),
                "--branches",
                branch,
                "--atlases",
                atlas,
                "--subjects",
                subject,
                "--resume-mode",
                "resume",
                "--retain-level",
                "render",
            ]
            proc = subprocess.run(cmd, text=True, capture_output=True, env=worker_env())
            if proc.returncode != 0:
                return subject, False, (
                    f"{branch}/{atlas} failed:\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}"
                )
    return subject, True, "ok"


def run_subject_batch(subjects: list[str]) -> list[str]:
    todo = [s for s in subjects if not subject_is_done(s)]
    log(f"subject batch: {len(subjects)} total, {len(todo)} need experimental run")
    if not todo:
        return []

    failed: list[str] = []
    done = 0
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as pool:
        futures = {pool.submit(run_subject_all_combos, s): s for s in todo}
        for fut in as_completed(futures):
            subject, ok, msg = fut.result()
            done += 1
            if ok:
                log(f"subject done [{done}/{len(todo)}] {subject}")
            else:
                log(f"SUBJECT FAIL [{done}/{len(todo)}] {subject}: {msg}")
                failed.append(subject)
    return failed


def run_group_priors(subjects: list[str]) -> None:
    log("running group priors …")
    cmd = [
        PYTHON,
        str(RUN_GROUP_PRIOR),
        "--branches",
        *BRANCHES,
        "--atlases",
        *ATLASES,
        "--subjects",
        *subjects,
        "--smoothings",
        "2mm",
        "4mm",
        "--group-k-rule",
        "mean-instability-1se",
        "--min-parcel-pass-rate",
        "0.67",
        "--out-root",
        str(OUT_ROOT),
        "--group-out-root",
        str(GROUP_OUT_ROOT),
        "--views",
        "ventral",
        "--layout",
        "1x2",
    ]
    run(cmd, "run_group_prior_fastpfm")
    log("group priors done")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cortex-pfm-pid", type=int, default=None)
    parser.add_argument("--skip-subject-batch", action="store_true")
    parser.add_argument("--skip-group-priors", action="store_true")
    args = parser.parse_args()

    LOG_DIR.mkdir(exist_ok=True)
    subjects = [s.strip() for s in MANIFEST.read_text().splitlines() if s.strip()]
    log(f"pipeline start — {len(subjects)} subjects, branches={BRANCHES}, atlases={ATLASES}")

    if args.cortex_pfm_pid:
        wait_for_pid(args.cortex_pfm_pid)

    if not args.skip_subject_batch:
        failed = run_subject_batch(subjects)
        if failed:
            log(f"ERROR: {len(failed)} subjects failed subject batch: {failed}")
            return 1
        log("subject batch complete")
    else:
        log("skipping subject batch (--skip-subject-batch)")

    if not args.skip_group_priors:
        good = [s for s in subjects if subject_is_done(s)]
        log(f"group priors using {len(good)}/{len(subjects)} subjects with all combos done")
        if len(good) < 2:
            log("ERROR: fewer than 2 subjects available for group priors, aborting")
            return 1
        try:
            run_group_priors(good)
        except RuntimeError as exc:
            log(f"ERROR in group priors: {exc}")
            return 1
    else:
        log("skipping group priors (--skip-group-priors)")

    log("pipeline finished successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())

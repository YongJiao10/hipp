#!/usr/bin/env python3
"""Continuation: wait for pipeline PID, retry failed subjects, then run group priors.

Usage:
    python scripts/run_group_priors_continuation.py --wait-pid <PID>
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
MANIFEST = REPO_ROOT / "manifests" / "hcp_7t_hippocampus_fmri_qc_pass_153.txt"
RUN_BATCH = REPO_ROOT / "scripts" / "hipp_parcellation_network" / "run_batch.py"
RUN_GROUP_PRIOR = REPO_ROOT / "scripts" / "hipp_parcellation_network" / "run_group_prior_fastpfm.py"
OUT_ROOT = REPO_ROOT / "outputs_migration" / "hipp_functional_parcellation_network"
PRESENT_DIR = REPO_ROOT / "present_network_migration"
GROUP_OUT_ROOT = REPO_ROOT / "outputs_migration" / "hipp_group_prior_fastpfm"
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


def wait_for_pid(pid: int) -> None:
    log(f"waiting for PID {pid} to finish …")
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(20)
    log(f"PID {pid} exited")


def subject_is_done(subject: str) -> bool:
    for branch in BRANCHES:
        for atlas in ATLASES:
            p = OUT_ROOT / branch / atlas / f"sub-{subject}" / "final_selection_summary.json"
            if not p.exists():
                return False
    return True


def run_subject_all_combos(subject: str) -> tuple[str, bool, str]:
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


def retry_failed(subjects: list[str]) -> list[str]:
    todo = [s for s in subjects if not subject_is_done(s)]
    log(f"retry pass: {len(todo)} subjects still incomplete")
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
                log(f"retry done [{done}/{len(todo)}] {subject}")
            else:
                log(f"retry FAIL [{done}/{len(todo)}] {subject}: {msg[:300]}")
                failed.append(subject)
    return failed


def run_group_priors(subjects: list[str]) -> None:
    log(f"running group priors with {len(subjects)} subjects …")
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
    proc = subprocess.run(cmd, text=True, capture_output=True, env=worker_env())
    if proc.returncode != 0:
        raise RuntimeError(
            f"group priors failed:\nSTDOUT:\n{proc.stdout[-3000:]}\nSTDERR:\n{proc.stderr[-3000:]}"
        )
    log("group priors done")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, required=True)
    args = parser.parse_args()

    subjects = [s.strip() for s in MANIFEST.read_text().splitlines() if s.strip()]

    wait_for_pid(args.wait_pid)

    still_failed = retry_failed(subjects)
    if still_failed:
        log(f"WARNING: {len(still_failed)} subjects still failed after retry: {still_failed}")

    good = [s for s in subjects if subject_is_done(s)]
    log(f"{len(good)}/{len(subjects)} subjects complete — proceeding to group priors")

    if len(good) < 10:
        log("ERROR: too few subjects for meaningful group priors, aborting")
        return 1

    try:
        run_group_priors(good)
    except RuntimeError as exc:
        log(f"ERROR: {exc}")
        return 1

    log("continuation pipeline finished successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())

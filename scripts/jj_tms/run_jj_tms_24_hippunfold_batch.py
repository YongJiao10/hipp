#!/usr/bin/env python3
"""Batch runner to generate standard space-corobl den-512 surfaces for remaining JJ_TMS subjects."""

from __future__ import annotations

import concurrent.futures
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_EXE = "/home/yojiao/.conda/envs/py314/bin/python"

MISSING_24 = [
    "1028", "1029", "1031", "1033", "1034", "1040", "1041",
    "1055", "1056", "1057", "1058", "1059", "1060", "1061",
    "1062", "1063", "1064", "1065", "1066", "1067", "1068",
    "3012", "3025", "3028"
]

LOG_DIR = REPO_ROOT / "outputs_iowa_jj_tms" / "logs" / "hippunfold_24_batch"
LOG_DIR.mkdir(parents=True, exist_ok=True)
BATCH_DIR = REPO_ROOT / "outputs_iowa_jj_tms" / "inputs" / "hippunfold" / "dense_corobl_batch"


def is_already_done(subject: str) -> bool:
    surf_dir = BATCH_DIR / f"sub-{subject}" / "hippunfold" / f"sub-{subject}" / "surf"
    l_mid = surf_dir / f"sub-{subject}_hemi-L_space-corobl_den-512_label-hipp_midthickness.surf.gii"
    r_mid = surf_dir / f"sub-{subject}_hemi-R_space-corobl_den-512_label-hipp_midthickness.surf.gii"
    return l_mid.exists() and r_mid.exists()


def run_subject(subject: str) -> tuple[str, bool, float]:
    if is_already_done(subject):
        return subject, True, 0.0

    # Auto-clean any stale locks
    locks_dir = BATCH_DIR / f"sub-{subject}" / "hippunfold" / ".snakemake" / "locks"
    if locks_dir.exists():
        for f in locks_dir.glob("*"):
            try:
                f.unlink()
            except Exception:
                pass

    start = time.time()
    log_file = LOG_DIR / f"sub-{subject}.log"
    cmd = [
        PYTHON_EXE,
        str(REPO_ROOT / "scripts" / "core_workflow" / "01_01_run_hippunfold_t1w_local.py"),
        "--dataset", "iowa_jj_tms",
        subject,
    ]
    env = os.environ.copy()
    env["HIPPUNFOLD_EXTRA_ARGS"] = "--force-output"

    with open(log_file, "w", encoding="utf-8") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=str(REPO_ROOT))

    elapsed = time.time() - start
    success = (proc.returncode == 0) and is_already_done(subject)
    return subject, success, elapsed


def main() -> int:
    todo = [sub for sub in MISSING_24 if not is_already_done(sub)]
    max_workers = min(6, len(todo)) if todo else 1  # Strictly capped at 6 (hard cap <= 10)
    print(f"Starting HippUnfold batch for {len(todo)} remaining subjects with concurrency={max_workers}...", flush=True)

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(run_subject, sub): sub for sub in todo}
        for future in concurrent.futures.as_completed(future_map):
            sub = future_map[future]
            try:
                subject, success, elapsed = future.result()
                completed += 1
                status = "SUCCESS" if success else "FAILED"
                print(f"[{completed}/{len(todo)}] sub-{subject}: {status} in {elapsed:.1f}s", flush=True)
            except Exception as exc:
                print(f"[{completed}/{len(todo)}] sub-{sub}: ERROR {exc}", flush=True)

    print("All batch jobs completed.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

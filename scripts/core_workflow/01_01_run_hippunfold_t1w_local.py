#!/usr/bin/env python3
"""Run one subject or a manifest through the locked local HippUnfold contract."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Mapping

from lib.hippunfold_run_contract import (
    DATASET_JOB_LIMITS,
    DATASET_PRESETS,
    REPO_ROOT,
    HippUnfoldRunContract,
    build_run_environment,
    detect_system_name,
    prepare_runtime_directories,
    prepare_runtime_python,
    prepare_subject_environment,
    resolve_run_contract,
    run_hippunfold,
    validate_local_assets,
)


PY314 = Path("/home/yojiao/.conda/envs/py314/bin/python")
OUTPUT_CHECKER = REPO_ROOT / "scripts/core_workflow/01_02_check_hippunfold_outputs.py"


def _is_completed(contract: HippUnfoldRunContract) -> bool:
    return (contract.output_dir / "core_workflow_cleanup_receipt.json").is_file()


def _run_subject(contract: HippUnfoldRunContract, base_env: Mapping[str, str]) -> None:
    subject = f"sub-{contract.subject}"
    env = prepare_subject_environment(contract, base_env)
    log_file = contract.preset.output_root / f"{subject}_hippunfold.log"
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"[start] {subject} at {now}", flush=True)
    with log_file.open("w", encoding="utf-8") as log:
        returncode = run_hippunfold(contract, env, stdout=log)
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, contract.hippunfold_argv)


def _check_subject(contract: HippUnfoldRunContract) -> None:
    subject = f"sub-{contract.subject}"
    log_file = contract.preset.output_root / f"{subject}_hippunfold.log"
    with log_file.open("a", encoding="utf-8") as log:
        subprocess.run(
            [
                str(PY314), str(OUTPUT_CHECKER),
                "--batch-root", str(contract.preset.output_root),
                "--subject", contract.subject, "--fail-on-missing",
            ],
            cwd=str(REPO_ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"[done ] {subject} at {now}", flush=True)


def _run_batch(contracts: list[HippUnfoldRunContract], jobs: int) -> None:
    dataset = contracts[0].dataset
    limit = DATASET_JOB_LIMITS.get(dataset, 10)
    if not 1 <= jobs <= limit:
        raise ValueError(f"HippUnfold batch for {dataset} allows between 1 and {limit} jobs.")
    pending = []
    for contract in contracts:
        if _is_completed(contract):
            print(f"[skip] sub-{contract.subject} (already completed)", flush=True)
        else:
            pending.append((contract, build_run_environment(contract)))
    if not pending:
        return
    if jobs > 1:
        detect_system_name()

    prepare_runtime_directories(pending[0][1])
    validate_local_assets(pending[0][1])
    prepare_runtime_python()
    next_index = 0
    failure: Exception | None = None
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        active = {}
        while active or (failure is None and next_index < len(pending)):
            while failure is None and next_index < len(pending) and len(active) < jobs:
                contract, env = pending[next_index]
                active[pool.submit(_run_subject, contract, env)] = contract
                next_index += 1
            finished, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in finished:
                contract = active.pop(future)
                try:
                    future.result()
                    _check_subject(contract)
                except Exception as exc:
                    if failure is None:
                        failure = exc
    if failure is not None:
        raise failure


def run_t1w_local(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the locked local HippUnfold T1w pipeline."
    )
    parser.add_argument("subject", nargs="?", help="Subject ID without the sub- prefix.")
    parser.add_argument("--manifest", type=Path, help="Subject list for batch execution.")
    parser.add_argument(
        "--jobs", type=int, default=1,
        help="Batch workers (PDC max 3; BANDA max 1).",
    )
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_PRESETS),
        required=True,
        help="Closed dataset preset; paths and all HippUnfold flags come from the preset.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved fixed contract without running HippUnfold.",
    )
    args = parser.parse_args(argv)
    if (args.subject is None) == (args.manifest is None):
        parser.error("Provide exactly one subject or --manifest.")
    if args.manifest is None and args.jobs != 1:
        parser.error("--jobs applies to --manifest mode only.")

    if args.manifest is not None:
        subjects = [
            line.strip()
            for line in args.manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not subjects:
            raise RuntimeError(f"No subjects found in HippUnfold manifest: {args.manifest}")
    else:
        subjects = [args.subject]
    contracts = [resolve_run_contract(args.dataset, subject) for subject in subjects]
    if args.dry_run:
        payload = [contract.as_json() for contract in contracts]
        print(json.dumps(
            payload if args.manifest is not None else payload[0],
            indent=2, sort_keys=True,
        ))
        return 0
    if args.manifest is not None:
        _run_batch(contracts, args.jobs)
        return 0

    contract = contracts[0]
    env = build_run_environment(contract)
    prepare_runtime_directories(env)
    validate_local_assets(env)
    env = prepare_subject_environment(contract, env)
    prepare_runtime_python()
    return int(run_hippunfold(contract, env))


if __name__ == "__main__":
    raise SystemExit(run_t1w_local(sys.argv[1:]))

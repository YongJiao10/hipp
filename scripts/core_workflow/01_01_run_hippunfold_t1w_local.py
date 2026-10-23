#!/usr/bin/env python3
"""Run one subject or a manifest through the locked local HippUnfold contract."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Mapping, TextIO

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
    start_hippunfold,
    validate_local_assets,
)


PY314 = Path("/home/yojiao/.conda/envs/py314/bin/python")
OUTPUT_CHECKER = REPO_ROOT / "scripts/core_workflow/01_02_check_hippunfold_outputs.py"


def _is_completed(contract: HippUnfoldRunContract) -> bool:
    report_path = contract.output_dir / "output_check.json"
    if not report_path.is_file():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Malformed HippUnfold output check: {report_path}: {exc}") from exc
    if not isinstance(report, list) or len(report) != 1:
        raise RuntimeError(f"Malformed HippUnfold output check: {report_path}")
    row = report[0]
    if row.get("run_contract") != contract.as_json():
        raise RuntimeError(f"HippUnfold output check has a mismatched run contract: {report_path}")
    if row.get("status") != "complete_all_outputs_present" or row.get("missing_count") != 0:
        return False
    manifest_path = contract.preset.output_root / "_batch_logs/output_integrity_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Malformed HippUnfold output contract: {manifest_path}: {exc}") from exc
    if not isinstance(manifest.get("expected_outputs"), list) or not manifest["expected_outputs"]:
        raise RuntimeError(f"Empty or malformed HippUnfold output contract: {manifest_path}")
    return all(
        (contract.output_dir / template.format(subject=contract.subject)).is_file()
        for template in manifest["expected_outputs"]
    )


def _start_subject(
    contract: HippUnfoldRunContract, base_env: Mapping[str, str]
) -> tuple[subprocess.Popen, TextIO]:
    subject = f"sub-{contract.subject}"
    env = prepare_subject_environment(contract, base_env)
    log_file = contract.preset.output_root / f"{subject}_hippunfold.log"
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"[start] {subject} at {now}", flush=True)
    log = log_file.open("w", encoding="utf-8")
    try:
        process = start_hippunfold(contract, env, stdout=log)
    except BaseException:
        log.close()
        raise
    return process, log


def _check_subject(contract: HippUnfoldRunContract) -> None:
    subject = f"sub-{contract.subject}"
    log_file = contract.preset.output_root / f"{subject}_hippunfold.log"
    with log_file.open("a", encoding="utf-8") as log:
        subprocess.run(
            [
                str(PY314), str(OUTPUT_CHECKER),
                "--batch-root", str(contract.preset.output_root),
                "--subject", contract.subject, "--fail-on-missing", "--no-cleanup",
                "--json-out", str(contract.output_dir / "output_check.json"),
            ],
            cwd=str(REPO_ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    report_path = contract.output_dir / "output_check.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if len(report) != 1 or report[0].get("subject") != contract.subject:
        raise RuntimeError(f"Unexpected subject in HippUnfold output check: {report_path}")
    report[0]["run_contract"] = contract.as_json()
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"[done ] {subject} at {now}", flush=True)


def _run_batch(contracts: list[HippUnfoldRunContract], jobs: int) -> None:
    dataset = contracts[0].dataset
    limit = DATASET_JOB_LIMITS.get(dataset, 2)
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
    active = {}
    while active or (failure is None and next_index < len(pending)):
        while failure is None and next_index < len(pending) and len(active) < jobs:
            contract, env = pending[next_index]
            try:
                process, log = _start_subject(contract, env)
            except Exception as exc:
                failure = exc
                break
            active[process] = (contract, log)
            next_index += 1
        finished = [process for process in active if process.poll() is not None]
        if not finished and active:
            time.sleep(0.2)
        for process in finished:
            contract, log = active.pop(process)
            log.close()
            try:
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, contract.hippunfold_argv)
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
        "--jobs", type=int, default=None,
        help="Manifest concurrency (default: HCP 8, others 1; HCP max 12, others max 2, BANDA max 1).",
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
    if args.manifest is None and args.jobs not in (None, 1):
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
        jobs = args.jobs
        if jobs is None:
            jobs = 8 if args.dataset == "hcp" else 1
        _run_batch(contracts, jobs)
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

#!/usr/bin/env python3
"""Run one subject through the locked local HippUnfold contract."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from lib.hippunfold_run_contract import (
    REPO_ROOT,
    build_run_environment,
    resolve_run_contract,
    validate_local_assets,
)


RUNTIME_PY = REPO_ROOT / ".runtime_py"


def _write_runtime_sitecustomize() -> None:
    RUNTIME_PY.mkdir(parents=True, exist_ok=True)
    (RUNTIME_PY / "sitecustomize.py").write_text(
        "import multiprocessing as mp\n"
        "try:\n"
        "    mp.set_start_method('fork')\n"
        "except RuntimeError:\n"
        "    pass\n"
        "try:\n"
        "    import torch.multiprocessing as tmp\n"
        "    tmp.set_sharing_strategy('file_system')\n"
        "except Exception:\n"
        "    pass\n",
        encoding="utf-8",
    )


def run_t1w_local(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the locked local HippUnfold T1w pipeline."
    )
    parser.add_argument("subject", help="Subject ID without the sub- prefix.")
    parser.add_argument(
        "--dataset",
        choices=("hcp", "stanford", "iowa_jj_tms"),
        required=True,
        help="Closed dataset preset; paths and all HippUnfold flags come from the preset.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved fixed contract without running HippUnfold.",
    )
    args = parser.parse_args(argv)

    contract = resolve_run_contract(args.dataset, args.subject)
    if args.dry_run:
        print(json.dumps(contract.as_json(), indent=2, sort_keys=True))
        return 0

    validate_local_assets()
    if not contract.bids_dir.is_dir():
        raise FileNotFoundError(f"Dataset input directory is missing: {contract.bids_dir}")
    contract.output_dir.mkdir(parents=True, exist_ok=True)
    _write_runtime_sitecustomize()

    env = build_run_environment(contract)
    for key in ("XDG_CACHE_HOME", "PIP_CACHE_DIR", "TMPDIR"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    env["PYTHONPATH"] = f"{RUNTIME_PY}:{env.get('PYTHONPATH', '')}"
    proc = subprocess.run(
        list(contract.hippunfold_argv),
        cwd=str(REPO_ROOT),
        env=env,
        check=False,
    )
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(run_t1w_local(sys.argv[1:]))

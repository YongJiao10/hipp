#!/usr/bin/env python3
"""Compatibility entry point for the legacy ESTT array launcher.

The old shared launcher invokes this path and passes the subject-level
arguments that were used by the pre-consolidation workflow.  Keep this file
as a thin delegator: the maintained implementation lives in
``scripts/ArealMSHBM_Kong2022/run_kong2022_areal_mshbm_subject_sge.sh`` under
its explicit ``DATASET=estt`` branch.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_LAUNCHER = (
    REPO_ROOT
    / "scripts"
    / "ArealMSHBM_Kong2022"
    / "run_kong2022_areal_mshbm_subject_sge.sh"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compatibility shim for the legacy ESTT Kong2022 launcher.")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--dtseries", required=True, type=Path)
    parser.add_argument("--left-inflated", required=True, type=Path)
    parser.add_argument("--right-inflated", required=True, type=Path)
    parser.add_argument("--left-midthickness", required=True, type=Path)
    parser.add_argument("--right-midthickness", required=True, type=Path)
    parser.add_argument("--sulc-dscalar", required=True, type=Path)
    parser.add_argument("--cbig-code-dir", required=True, type=Path)
    parser.add_argument("--group-prior", required=True, type=Path)
    parser.add_argument("--kong2022-dlabel", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--model", default="gMSHBM")
    parser.add_argument("--num-rois", default="400")
    parser.add_argument("--w", default="30")
    parser.add_argument("--c", default="30")
    parser.add_argument("--beta", default="50")
    parser.add_argument("--run-lengths", required=True, nargs=4, type=int)
    parser.add_argument("--subject-dir-layout", default="flat")
    parser.add_argument("--matlab-cmd", default="matlab")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not CANONICAL_LAUNCHER.is_file():
        raise SystemExit(f"Missing maintained launcher: {CANONICAL_LAUNCHER}")
    if args.model != "gMSHBM" or args.num_rois != "400" or args.w != "30" or args.c != "30" or args.beta != "50":
        raise SystemExit("Compatibility shim only supports the established gMSHBM 400/w30/c30/beta50 contract")
    if args.subject_dir_layout != "flat":
        raise SystemExit("Compatibility shim only supports --subject-dir-layout flat")
    if args.matlab_cmd != "matlab":
        raise SystemExit("Compatibility shim only supports --matlab-cmd matlab")
    if any(length < 1 for length in args.run_lengths):
        raise SystemExit("--run-lengths must contain four positive integers")

    subject_id = str(args.subject).removeprefix("sub-")
    subject_dir = args.dtseries.resolve().parent.parent
    data_root = subject_dir.parent
    env = os.environ.copy()
    env.update(
        {
            "DATASET": "estt",
            "DATA_ROOT": str(data_root),
            "OUT_ROOT": str(args.out_root.resolve()),
            "SUBJECT_IDS_CSV": subject_id,
            "SUBJECT_INDEX": "1",
            "REPO_ROOT": str(REPO_ROOT),
            "SCRIPT_DIR": str(CANONICAL_LAUNCHER.parent),
            "PYTHON_EXE": sys.executable,
            "CBIG_CODE_DIR": str(args.cbig_code_dir.resolve()),
            "GROUP_PRIOR": str(args.group_prior.resolve()),
            "KONG2022_DLABEL": str(args.kong2022_dlabel.resolve()),
            "WB_COMMAND": str(REPO_ROOT / "software" / "workbench_2.1.0" / "bin_linux64" / "wb_command"),
            "KEEP_TMP": "1",
            "TASK_DRY_RUN": "1" if args.dry_run else "0",
            "OVERWRITE": env.get("OVERWRITE", "0"),
        }
    )
    return subprocess.run([str(CANONICAL_LAUNCHER)], env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())

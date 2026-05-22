#!/usr/bin/env python3
from __future__ import annotations

"""
Process Step Description:
Run local HippUnfold T1w pipeline. This is the initial step of the workflow to perform hippocampal unfolding and surface mesh generation for a single subject.
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_T1W_SCRIPT = REPO_ROOT / "scripts" / "core_workflow" / "lib" / "hippunfold_shell" / "run_hippunfold_local_t1w_isolated_cache.sh"


def run_t1w_local(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run local HippUnfold T1w pipeline.")
    parser.add_argument("subject", help="Subject ID without sub- prefix.")
    parser.add_argument("input_dir", nargs="?", default="data/hippunfold_input", help="Input root containing sub-*/anat.")
    parser.add_argument("out_dir", help="Output directory for this subject.")
    args = parser.parse_args(argv)
    cmd = ["bash", str(RUN_T1W_SCRIPT), str(args.subject), str(args.input_dir), str(args.out_dir)]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(run_t1w_local(sys.argv[1:]))

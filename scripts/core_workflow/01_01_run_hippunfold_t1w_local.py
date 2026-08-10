#!/usr/bin/env python3
from __future__ import annotations

"""
Process Step Description:
Run local HippUnfold T1w pipeline. This is the initial step of the workflow to perform hippocampal unfolding and surface mesh generation for a single subject.

Main Flow Path:
- 1st step in the numbered pipeline sequence:
  01_01_run_hippunfold_t1w_local.py (this step) -> 01_02_check_hippunfold_outputs.py -> 02_01_prepare_group_inputs.py -> 02_02_build_qc_manifest.py -> 03_01/03_02 reference building -> 04_01/04_02/04_03 FC materialization -> 05_01 gradient computation -> 05_02 group atlas parcellation -> 06_xx presentation rendering -> 07_01 group matrix overview.

Inputs:
- Command-line arguments:
  - subject: Subject ID without the "sub-" prefix (required).
  - input_dir (optional, default: "data/hippunfold_input"): Root directory containing raw structural BIDS folders.
  - out_dir: Subject-specific output directory where HippUnfold results will be written (required).
  - --model-path (optional): Canonical local nnUNet model tar; the validated local-model overlay removes the remote rule and binds the canonical tar explicitly.
- Physical inputs:
  - Raw T1w anatomical MRI files under `{input_dir}/sub-<subject>/anat/`.

Outputs:
- Command-line output / file outputs:
  - Subject-level HippUnfold output directory: `{out_dir}/sub-<subject>/hippunfold/` containing reconstructed surface GIFTI files and volumetric segmentations.
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_T1W_SCRIPT = REPO_ROOT / "scripts" / "core_workflow" / "lib" / "hippunfold_shell" / "run_hippunfold_local_t1w_isolated_cache.sh"
DEFAULT_MODEL_PATH = Path("/home/yojiao/hpchome/hipp/runtime/hippunfold_asset_store_shared/hippunfold_assets/model/trained_model.3d_fullres.Task101_hcp1200_T1w.nnUNetTrainerV2.model_best.tar")


def run_t1w_local(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run local HippUnfold T1w pipeline.")
    parser.add_argument("subject", help="Subject ID without sub- prefix.")
    parser.add_argument("input_dir", nargs="?", default="data/hippunfold_input", help="Input root containing sub-*/anat.")
    parser.add_argument("out_dir", help="Output directory for this subject.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Canonical local HippUnfold nnUNet model tar; remote model retrieval is disabled.")
    args = parser.parse_args(argv)
    cmd = ["bash", str(RUN_T1W_SCRIPT), str(args.subject), str(args.input_dir), str(args.out_dir), str(args.model_path)]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(run_t1w_local(sys.argv[1:]))

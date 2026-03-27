#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage minimal hippunfold-compatible inputs")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args()

    subject = args.subject
    source_dir = Path(args.source_dir)
    input_dir = Path(args.input_dir)
    anat_dir = input_dir / f"sub-{subject}" / "anat"
    func_dir = input_dir / f"sub-{subject}" / "func"
    input_dir.mkdir(parents=True, exist_ok=True)

    dataset_description = {
        "Name": "HippoMaps HCP 7T analysis input",
        "BIDSVersion": "1.9.0",
        "DatasetType": "raw",
    }
    (input_dir / "dataset_description.json").write_text(
        json.dumps(dataset_description, indent=2), encoding="utf-8"
    )

    copy_file(
        source_dir / f"sub-{subject}_T1w_acpc_dc_restore.nii.gz",
        anat_dir / f"sub-{subject}_T1w.nii.gz",
    )
    copy_file(
        source_dir / f"sub-{subject}_T2w_acpc_dc_restore.nii.gz",
        anat_dir / f"sub-{subject}_T2w.nii.gz",
    )
    copy_file(
        source_dir / f"sub-{subject}_rfMRI_REST_7T_hp2000_clean_rclean_tclean.nii.gz",
        func_dir / f"sub-{subject}_task-rest_run-concat_bold.nii.gz",
    )
    copy_file(
        source_dir / f"sub-{subject}_rfMRI_REST_7T_brain_mask.nii.gz",
        func_dir / f"sub-{subject}_task-rest_run-concat_desc-brain_mask.nii.gz",
    )

    print(input_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

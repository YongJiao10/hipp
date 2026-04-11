#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re


def link_file(src: Path, dst: Path) -> None:
    """Create a hardlink dst → src, replacing an existing dst if present."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.link(src, dst)


RUNWISE_DTSERIES_RE = re.compile(
    r"sub-(?P<subject>\d+)_rfMRI_REST(?P<run_id>[1-4])_7T_(?P<phase>AP|PA)_Atlas_MSMAll_hp2000_clean_rclean_tclean\.dtseries\.nii$"
)
RUNWISE_BOLD_RE = re.compile(
    r"sub-(?P<subject>\d+)_rfMRI_REST(?P<run_id>[1-4])_7T_(?P<phase>AP|PA)_hp2000_clean_rclean_tclean\.nii\.gz$"
)


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

    link_file(
        source_dir / f"sub-{subject}_T1w_acpc_dc_restore.nii.gz",
        anat_dir / f"sub-{subject}_T1w.nii.gz",
    )
    link_file(
        source_dir / f"sub-{subject}_T2w_acpc_dc_restore.nii.gz",
        anat_dir / f"sub-{subject}_T2w.nii.gz",
    )
    link_file(
        source_dir / f"sub-{subject}_rfMRI_REST_7T_hp2000_clean_rclean_tclean.nii.gz",
        func_dir / f"sub-{subject}_task-rest_run-concat_bold.nii.gz",
    )
    link_file(
        source_dir / f"sub-{subject}_rfMRI_REST_7T_Atlas_MSMAll_hp2000_clean_rclean_tclean.dtseries.nii",
        func_dir / f"sub-{subject}_task-rest_run-concat.dtseries.nii",
    )
    link_file(
        source_dir / f"sub-{subject}_rfMRI_REST_7T_brain_mask.nii.gz",
        func_dir / f"sub-{subject}_task-rest_run-concat_desc-brain_mask.nii.gz",
    )

    for path in sorted(source_dir.iterdir()):
        dt_match = RUNWISE_DTSERIES_RE.fullmatch(path.name)
        if dt_match and dt_match.group("subject") == subject:
            link_file(
                path,
                func_dir / f"sub-{subject}_task-rest_run-{dt_match.group('run_id')}.dtseries.nii",
            )
            continue
        bold_match = RUNWISE_BOLD_RE.fullmatch(path.name)
        if bold_match and bold_match.group("subject") == subject:
            link_file(
                path,
                func_dir / f"sub-{subject}_task-rest_run-{bold_match.group('run_id')}_bold.nii.gz",
            )

    print(input_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

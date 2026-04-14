#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess


REMOTE_ROOT_DEFAULT = "/Volumes/Elements/HCP-YA-2025"
STRUCT_DIR = "Structural Preprocessed Recommended for 3T and 7T"
RSFMRI_ARCHIVE_DIR = "Resting State fMRI 7T Preprocessed Recommended archive"
DTSERIES_MEMBER = "MNINonLinear/Results/rfMRI_REST_7T/rfMRI_REST_7T_Atlas_MSMAll_hp2000_clean_rclean_tclean.dtseries.nii"
BOLD_MEMBER = "MNINonLinear/Results/rfMRI_REST_7T/rfMRI_REST_7T_hp2000_clean_rclean_tclean.nii.gz"
MASK_MEMBER = "MNINonLinear/Results/rfMRI_REST_7T/rfMRI_REST_7T_brain_mask.nii.gz"
RUN_SPECS = [
    {
        "run_id": "1",
        "hcp_name": "rfMRI_REST1_7T_PA",
        "dtseries_member": "MNINonLinear/Results/rfMRI_REST1_7T_PA/rfMRI_REST1_7T_PA_Atlas_MSMAll_hp2000_clean_rclean_tclean.dtseries.nii",
        "bold_member": "MNINonLinear/Results/rfMRI_REST1_7T_PA/rfMRI_REST1_7T_PA_hp2000_clean_rclean_tclean.nii.gz",
    },
    {
        "run_id": "2",
        "hcp_name": "rfMRI_REST2_7T_AP",
        "dtseries_member": "MNINonLinear/Results/rfMRI_REST2_7T_AP/rfMRI_REST2_7T_AP_Atlas_MSMAll_hp2000_clean_rclean_tclean.dtseries.nii",
        "bold_member": "MNINonLinear/Results/rfMRI_REST2_7T_AP/rfMRI_REST2_7T_AP_hp2000_clean_rclean_tclean.nii.gz",
    },
    {
        "run_id": "3",
        "hcp_name": "rfMRI_REST3_7T_PA",
        "dtseries_member": "MNINonLinear/Results/rfMRI_REST3_7T_PA/rfMRI_REST3_7T_PA_Atlas_MSMAll_hp2000_clean_rclean_tclean.dtseries.nii",
        "bold_member": "MNINonLinear/Results/rfMRI_REST3_7T_PA/rfMRI_REST3_7T_PA_hp2000_clean_rclean_tclean.nii.gz",
    },
    {
        "run_id": "4",
        "hcp_name": "rfMRI_REST4_7T_AP",
        "dtseries_member": "MNINonLinear/Results/rfMRI_REST4_7T_AP/rfMRI_REST4_7T_AP_Atlas_MSMAll_hp2000_clean_rclean_tclean.dtseries.nii",
        "bold_member": "MNINonLinear/Results/rfMRI_REST4_7T_AP/rfMRI_REST4_7T_AP_hp2000_clean_rclean_tclean.nii.gz",
    },
]


def ssh(host: str, remote_cmd: str, check: bool = True) -> str:
    proc = subprocess.run(["ssh", host, remote_cmd], check=check, text=True, capture_output=True)
    return proc.stdout


def stream_zip_member(host: str, zip_path: str, member: str, local_path: Path) -> None:
    remote_cmd = f"unzip -p {shlex.quote(zip_path)} {shlex.quote(member)}"
    with local_path.open("wb") as f:
        proc = subprocess.run(["ssh", host, remote_cmd], stdout=f, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="replace"))


def discover_subject(host: str, remote_root: str) -> str:
    archive_root = f"{remote_root}/{RSFMRI_ARCHIVE_DIR}"
    struct_root = f"{remote_root}/{STRUCT_DIR}"
    zips = ssh(
        host,
        f"find {shlex.quote(archive_root)} -mindepth 1 -maxdepth 1 -type f -name '*_Rest7TRecommended.zip' -exec basename {{}} \\; | sort",
    )
    subjects = [x.replace("_Rest7TRecommended.zip", "").strip() for x in zips.splitlines() if x.strip()]
    for subject in subjects:
        struct_zip = f"{struct_root}/{subject}_StructuralRecommended.zip"
        archive_zip = f"{archive_root}/{subject}_Rest7TRecommended.zip"
        check_cmd = (
            f"test -f {shlex.quote(struct_zip)}"
            f" && test -f {shlex.quote(archive_zip)}"
            f" && unzip -l {shlex.quote(archive_zip)} {shlex.quote(f'{subject}/{DTSERIES_MEMBER}')} >/dev/null 2>&1"
            f" && echo {shlex.quote(subject)}"
        )
        found = ssh(host, check_cmd, check=False).strip()
        if found:
            return found
    raise RuntimeError("No subject with both structural zip and archive dtseries found on remote drive")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch minimal HCP 7T single-subject data from remote drive into BIDS hippunfold input layout"
    )
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--remote-root", default=REMOTE_ROOT_DEFAULT)
    parser.add_argument("--subject", default="auto")
    parser.add_argument("--input-dir", required=True, help="BIDS hippunfold input root (e.g. data/hippunfold_input)")
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--include-runwise",
        action="store_true",
        help="Also fetch the four individual 7T resting-state runs.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    subject = args.subject
    if subject == "auto":
        subject = discover_subject(args.remote_host, args.remote_root)

    anat_dir = input_dir / f"sub-{subject}" / "anat"
    func_dir = input_dir / f"sub-{subject}" / "func"

    dataset_description = {
        "Name": "HippoMaps HCP 7T analysis input",
        "BIDSVersion": "1.9.0",
        "DatasetType": "raw",
    }
    input_dir.mkdir(parents=True, exist_ok=True)
    anat_dir.mkdir(parents=True, exist_ok=True)
    func_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "dataset_description.json").write_text(
        json.dumps(dataset_description, indent=2), encoding="utf-8"
    )

    struct_zip = f"{args.remote_root}/{STRUCT_DIR}/{subject}_StructuralRecommended.zip"
    archive_zip = f"{args.remote_root}/{RSFMRI_ARCHIVE_DIR}/{subject}_Rest7TRecommended.zip"

    files_to_fetch = [
        (struct_zip,  f"{subject}/T1w/T1w_acpc_dc_restore.nii.gz",   anat_dir / f"sub-{subject}_T1w.nii.gz",                                      "structural_t1w"),
        (struct_zip,  f"{subject}/T1w/T2w_acpc_dc_restore.nii.gz",   anat_dir / f"sub-{subject}_T2w.nii.gz",                                      "structural_t2w"),
        (archive_zip, f"{subject}/{DTSERIES_MEMBER}",                 func_dir / f"sub-{subject}_task-rest_run-concat.dtseries.nii",               "rsfmri_dtseries"),
        (archive_zip, f"{subject}/{BOLD_MEMBER}",                     func_dir / f"sub-{subject}_task-rest_run-concat_bold.nii.gz",                "rsfmri_volume"),
        (archive_zip, f"{subject}/{MASK_MEMBER}",                     func_dir / f"sub-{subject}_task-rest_run-concat_desc-brain_mask.nii.gz",     "brain_mask"),
    ]

    manifest_files = []
    for zip_path, member, local_path, kind in files_to_fetch:
        stream_zip_member(args.remote_host, zip_path, member, local_path)
        manifest_files.append({
            "kind": kind,
            "remote_source": f"{zip_path}::{member}",
            "local_path": str(local_path),
            "bytes": local_path.stat().st_size,
        })

    if args.include_runwise:
        for spec in RUN_SPECS:
            dt_member = f"{subject}/{spec['dtseries_member']}"
            bd_member = f"{subject}/{spec['bold_member']}"
            dt_local = func_dir / f"sub-{subject}_task-rest_run-{spec['run_id']}.dtseries.nii"
            bd_local = func_dir / f"sub-{subject}_task-rest_run-{spec['run_id']}_bold.nii.gz"
            stream_zip_member(args.remote_host, archive_zip, dt_member, dt_local)
            stream_zip_member(args.remote_host, archive_zip, bd_member, bd_local)
            manifest_files.extend([
                {
                    "kind": "rsfmri_dtseries_runwise",
                    "run_id": spec["run_id"],
                    "run_label": spec["hcp_name"],
                    "remote_source": f"{archive_zip}::{dt_member}",
                    "local_path": str(dt_local),
                    "bytes": dt_local.stat().st_size,
                },
                {
                    "kind": "rsfmri_volume_runwise",
                    "run_id": spec["run_id"],
                    "run_label": spec["hcp_name"],
                    "remote_source": f"{archive_zip}::{bd_member}",
                    "local_path": str(bd_local),
                    "bytes": bd_local.stat().st_size,
                },
            ])

    manifest = {
        "subject": subject,
        "remote_host": args.remote_host,
        "remote_root": args.remote_root,
        "files": manifest_files,
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

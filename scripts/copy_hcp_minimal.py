#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


REMOTE_ROOT_DEFAULT = "/Volumes/Elements/HCP-YA-2025"
STRUCT_DIR = "Structural Preprocessed Recommended for 3T and 7T"
RSFMRI_DIR = "Resting State fMRI 7T Preprocessed Recommended"
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


def run(cmd: list[str], check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        capture_output=capture,
    )


def ssh(host: str, remote_cmd: str, check: bool = True) -> str:
    proc = run(["ssh", host, remote_cmd], check=check, capture=True)
    return proc.stdout


def stream_remote_file(host: str, remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    remote_cmd = f"cat {shlex.quote(remote_path)}"
    with local_path.open("wb") as f:
        proc = subprocess.run(["ssh", host, remote_cmd], stdout=f, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="replace"))


def stream_zip_member(host: str, zip_path: str, member: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
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


def remote_stat(host: str, path: str) -> int:
    out = ssh(host, f"stat -f %z {shlex.quote(path)}").strip()
    return int(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy minimal HCP 7T single-subject data from remote drive")
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--remote-root", default=REMOTE_ROOT_DEFAULT)
    parser.add_argument("--subject", default="auto")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--include-runwise",
        action="store_true",
        help="Also copy the four individual 7T resting-state runs needed for run-aware instability testing.",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    subject = args.subject
    if subject == "auto":
        subject = discover_subject(args.remote_host, args.remote_root)

    struct_zip = f"{args.remote_root}/{STRUCT_DIR}/{subject}_StructuralRecommended.zip"
    archive_zip = f"{args.remote_root}/{RSFMRI_ARCHIVE_DIR}/{subject}_Rest7TRecommended.zip"
    t1_member = f"{subject}/T1w/T1w_acpc_dc_restore.nii.gz"
    t2_member = f"{subject}/T1w/T2w_acpc_dc_restore.nii.gz"
    dtseries_member = f"{subject}/{DTSERIES_MEMBER}"
    bold_member = f"{subject}/{BOLD_MEMBER}"
    mask_member = f"{subject}/{MASK_MEMBER}"

    t1_local = outdir / f"sub-{subject}_T1w_acpc_dc_restore.nii.gz"
    t2_local = outdir / f"sub-{subject}_T2w_acpc_dc_restore.nii.gz"
    dtseries_local = outdir / f"sub-{subject}_rfMRI_REST_7T_Atlas_MSMAll_hp2000_clean_rclean_tclean.dtseries.nii"
    bold_local = outdir / f"sub-{subject}_rfMRI_REST_7T_hp2000_clean_rclean_tclean.nii.gz"
    mask_local = outdir / f"sub-{subject}_rfMRI_REST_7T_brain_mask.nii.gz"

    stream_zip_member(args.remote_host, struct_zip, t1_member, t1_local)
    stream_zip_member(args.remote_host, struct_zip, t2_member, t2_local)
    stream_zip_member(args.remote_host, archive_zip, dtseries_member, dtseries_local)
    stream_zip_member(args.remote_host, archive_zip, bold_member, bold_local)
    stream_zip_member(args.remote_host, archive_zip, mask_member, mask_local)

    manifest = {
        "subject": subject,
        "remote_host": args.remote_host,
        "remote_root": args.remote_root,
        "files": [
            {
                "kind": "structural_t1w",
                "remote_source": f"{struct_zip}::{t1_member}",
                "local_path": str(t1_local),
                "bytes": t1_local.stat().st_size,
            },
            {
                "kind": "structural_t2w",
                "remote_source": f"{struct_zip}::{t2_member}",
                "local_path": str(t2_local),
                "bytes": t2_local.stat().st_size,
            },
            {
                "kind": "rsfmri_dtseries",
                "remote_source": f"{archive_zip}::{dtseries_member}",
                "local_path": str(dtseries_local),
                "bytes": dtseries_local.stat().st_size,
            },
            {
                "kind": "rsfmri_volume",
                "remote_source": f"{archive_zip}::{bold_member}",
                "local_path": str(bold_local),
                "bytes": bold_local.stat().st_size,
            },
            {
                "kind": "brain_mask",
                "remote_source": f"{archive_zip}::{mask_member}",
                "local_path": str(mask_local),
                "bytes": mask_local.stat().st_size,
            },
        ],
    }

    if args.include_runwise:
        for spec in RUN_SPECS:
            dtseries_member = f"{subject}/{spec['dtseries_member']}"
            bold_member = f"{subject}/{spec['bold_member']}"
            dtseries_local = outdir / f"sub-{subject}_{spec['hcp_name']}_Atlas_MSMAll_hp2000_clean_rclean_tclean.dtseries.nii"
            bold_local = outdir / f"sub-{subject}_{spec['hcp_name']}_hp2000_clean_rclean_tclean.nii.gz"
            stream_zip_member(args.remote_host, archive_zip, dtseries_member, dtseries_local)
            stream_zip_member(args.remote_host, archive_zip, bold_member, bold_local)
            manifest["files"].extend(
                [
                    {
                        "kind": "rsfmri_dtseries_runwise",
                        "run_id": spec["run_id"],
                        "run_label": spec["hcp_name"],
                        "remote_source": f"{archive_zip}::{dtseries_member}",
                        "local_path": str(dtseries_local),
                        "bytes": dtseries_local.stat().st_size,
                    },
                    {
                        "kind": "rsfmri_volume_runwise",
                        "run_id": spec["run_id"],
                        "run_label": spec["hcp_name"],
                        "remote_source": f"{archive_zip}::{bold_member}",
                        "local_path": str(bold_local),
                        "bytes": bold_local.stat().st_size,
                    },
                ]
            )

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

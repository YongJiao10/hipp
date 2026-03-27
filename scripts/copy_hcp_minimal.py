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
    rsfmri_root = f"{remote_root}/{RSFMRI_DIR}"
    struct_root = f"{remote_root}/{STRUCT_DIR}"
    dirs = ssh(host, f"find {shlex.quote(rsfmri_root)} -mindepth 1 -maxdepth 1 -type d -exec basename {{}} \\; | sort")
    subjects = [x.strip() for x in dirs.splitlines() if x.strip()]
    for subject in subjects:
        struct_zip = f"{struct_root}/{subject}_StructuralRecommended.zip"
        check_cmd = (
            f"test -f {shlex.quote(struct_zip)}"
            f" && test -f {shlex.quote(rsfmri_root + '/' + subject + '/rfMRI_REST_7T_hp2000_clean_rclean_tclean.nii.gz')}"
            f" && echo {shlex.quote(subject)}"
        )
        found = ssh(host, check_cmd, check=False).strip()
        if found:
            return found
    raise RuntimeError("No subject with both structural zip and 7T rsfMRI volume found on remote drive")


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
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    subject = args.subject
    if subject == "auto":
        subject = discover_subject(args.remote_host, args.remote_root)

    struct_zip = f"{args.remote_root}/{STRUCT_DIR}/{subject}_StructuralRecommended.zip"
    rsfmri_root = f"{args.remote_root}/{RSFMRI_DIR}/{subject}"
    rsfmri_volume = f"{rsfmri_root}/rfMRI_REST_7T_hp2000_clean_rclean_tclean.nii.gz"
    rsfmri_mask = f"{rsfmri_root}/rfMRI_REST_7T_brain_mask.nii.gz"
    t1_member = f"{subject}/T1w/T1w_acpc_dc_restore.nii.gz"
    t2_member = f"{subject}/T1w/T2w_acpc_dc_restore.nii.gz"

    t1_local = outdir / f"sub-{subject}_T1w_acpc_dc_restore.nii.gz"
    t2_local = outdir / f"sub-{subject}_T2w_acpc_dc_restore.nii.gz"
    bold_local = outdir / f"sub-{subject}_rfMRI_REST_7T_hp2000_clean_rclean_tclean.nii.gz"
    mask_local = outdir / f"sub-{subject}_rfMRI_REST_7T_brain_mask.nii.gz"

    stream_zip_member(args.remote_host, struct_zip, t1_member, t1_local)
    stream_zip_member(args.remote_host, struct_zip, t2_member, t2_local)
    stream_remote_file(args.remote_host, rsfmri_volume, bold_local)
    stream_remote_file(args.remote_host, rsfmri_mask, mask_local)

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
                "kind": "rsfmri_volume",
                "remote_source": rsfmri_volume,
                "local_path": str(bold_local),
                "bytes": remote_stat(args.remote_host, rsfmri_volume),
            },
            {
                "kind": "brain_mask",
                "remote_source": rsfmri_mask,
                "local_path": str(mask_local),
                "bytes": remote_stat(args.remote_host, rsfmri_mask),
            },
        ],
    }

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

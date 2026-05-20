#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess


REMOTE_ROOT_DEFAULT = "/Volumes/Elements/HCP-YA-2025"
STRUCT_DIR = "Structural Preprocessed Recommended for 3T and 7T"


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, text=True, capture_output=True)


def stream_zip_member(host: str, zip_path: str, member: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    remote_cmd = f"unzip -p {shlex.quote(zip_path)} {shlex.quote(member)}"
    with local_path.open("wb") as handle:
        proc = subprocess.run(["ssh", host, remote_cmd], stdout=handle, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="replace"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy HCP fsaverage_LR32k cortex assets needed by FASTANS cortex PFM."
    )
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--subject", required=True, help="Subject ID without sub- prefix, e.g. 100610")
    parser.add_argument("--remote-root", default=REMOTE_ROOT_DEFAULT)
    parser.add_argument("--outdir", required=True, help="Local anat output directory")
    parser.add_argument("--manifest", required=True, help="JSON manifest output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    subject = args.subject
    outdir = Path(args.outdir).resolve()
    manifest_path = Path(args.manifest).resolve()
    struct_zip = f"{args.remote_root}/{STRUCT_DIR}/{subject}_StructuralRecommended.zip"
    base_member = f"{subject}/MNINonLinear/fsaverage_LR32k"

    members = {
        "left_midthickness": (
            f"{base_member}/{subject}.L.midthickness_MSMAll.32k_fs_LR.surf.gii",
            f"sub-{subject}_hemi-L_space-fsLR_den-32k_desc-MSMAll_midthickness.surf.gii",
        ),
        "right_midthickness": (
            f"{base_member}/{subject}.R.midthickness_MSMAll.32k_fs_LR.surf.gii",
            f"sub-{subject}_hemi-R_space-fsLR_den-32k_desc-MSMAll_midthickness.surf.gii",
        ),
        "left_inflated": (
            f"{base_member}/{subject}.L.inflated_MSMAll.32k_fs_LR.surf.gii",
            f"sub-{subject}_hemi-L_space-fsLR_den-32k_desc-MSMAll_inflated.surf.gii",
        ),
        "right_inflated": (
            f"{base_member}/{subject}.R.inflated_MSMAll.32k_fs_LR.surf.gii",
            f"sub-{subject}_hemi-R_space-fsLR_den-32k_desc-MSMAll_inflated.surf.gii",
        ),
        "sulc_dscalar": (
            f"{base_member}/{subject}.sulc_MSMAll.32k_fs_LR.dscalar.nii",
            f"sub-{subject}_space-fsLR_den-32k_desc-MSMAll_sulc.dscalar.nii",
        ),
        "left_atlasroi": (
            f"{base_member}/{subject}.L.atlasroi.32k_fs_LR.shape.gii",
            f"sub-{subject}_hemi-L_space-fsLR_den-32k_atlasroi.shape.gii",
        ),
        "right_atlasroi": (
            f"{base_member}/{subject}.R.atlasroi.32k_fs_LR.shape.gii",
            f"sub-{subject}_hemi-R_space-fsLR_den-32k_atlasroi.shape.gii",
        ),
        "wb_spec": (
            f"{base_member}/{subject}.MSMAll.32k_fs_LR.wb.spec",
            f"sub-{subject}_space-fsLR_den-32k_desc-MSMAll.wb.spec",
        ),
    }

    manifest: dict[str, object] = {
        "subject": subject,
        "remote_host": args.remote_host,
        "remote_root": args.remote_root,
        "structural_zip": struct_zip,
        "files": [],
    }

    for kind, (member, filename) in members.items():
        local_path = outdir / filename
        stream_zip_member(args.remote_host, struct_zip, member, local_path)
        manifest["files"].append(
            {
                "kind": kind,
                "remote_source": f"{struct_zip}::{member}",
                "local_path": str(local_path),
                "bytes": local_path.stat().st_size,
            }
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

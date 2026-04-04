#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the formal structural + gradient views")
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--batch-root", required=True, help="dense_corobl_batch root containing sub-*/hippunfold and post_dense_corobl")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--scene", default="config/wb_locked_native_view.scene")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1024)
    args = parser.parse_args()

    python_exe = sys.executable or "/opt/miniconda3/envs/py314/bin/python"
    repo_root = Path(__file__).resolve().parents[2]
    scene = (repo_root / args.scene).resolve() if not Path(args.scene).is_absolute() else Path(args.scene).resolve()
    batch_root = Path(args.batch_root).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    renders = outdir / "renders"
    final = outdir / "final"
    renders.mkdir(parents=True, exist_ok=True)
    final.mkdir(parents=True, exist_ok=True)

    run(
        [
            python_exe,
            str(repo_root / "scripts" / "workbench" / "render_wb_scene_batch.py"),
            "--scene",
            str(scene),
            "--subjects",
            *args.subjects,
            "--outdir",
            str(renders / "structural"),
            "--scene-index",
            "1",
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--renderer",
            "OSMesa",
            "--name",
            "structural",
        ]
    )

    for subject in args.subjects:
        subdir = final / f"sub-{subject}"
        subdir.mkdir(parents=True, exist_ok=True)
        run(
            [
                python_exe,
                str(repo_root / "scripts" / "workbench" / "compose_wb_with_side_legend.py"),
                "--image",
                str(renders / "structural" / f"sub-{subject}" / f"sub-{subject}_wb_structural_native.png"),
                "--left-labels",
                str(batch_root / f"sub-{subject}" / "hippunfold" / f"sub-{subject}" / "surf" / f"sub-{subject}_hemi-L_space-corobl_label-hipp_atlas-multihist7_subfields.label.gii"),
                "--right-labels",
                str(batch_root / f"sub-{subject}" / "hippunfold" / f"sub-{subject}" / "surf" / f"sub-{subject}_hemi-R_space-corobl_label-hipp_atlas-multihist7_subfields.label.gii"),
                "--title",
                f"sub-{subject} Structural",
                "--out",
                str(subdir / f"sub-{subject}_wb_structural_biglegend.png"),
            ]
        )
        gradient_src = batch_root / f"sub-{subject}" / "post_dense_corobl" / f"sub-{subject}_hipp_fc_gradient1_native.png"
        if not gradient_src.exists():
            raise FileNotFoundError(f"Missing formal gradient image for subject {subject}: {gradient_src}")
        shutil.copyfile(gradient_src, subdir / f"sub-{subject}_gradient.png")

    print(final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

COMMON_DIR = Path(__file__).resolve().parent / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from hipp_density_assets import load_surface_density_from_pipeline_config


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def archive_dir(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        raise FileExistsError(f"Archive destination already exists: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def main() -> int:
    default_density = load_surface_density_from_pipeline_config(Path("config/hippo_pipeline.toml"))
    parser = argparse.ArgumentParser(description="Archive old functional outputs and regenerate formal structural-only renders")
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--input-dir", default="data/hippunfold_input")
    parser.add_argument("--batch-root", default="outputs/dense_corobl_batch")
    parser.add_argument("--archive-dir", default="outputs/dense_corobl_batch/_archived_volume_functional")
    parser.add_argument("--outdir", default="outputs/dense_corobl_batch/final_structural_only")
    parser.add_argument("--scene", default="config/wb_locked_native_view.scene")
    parser.add_argument("--density", default=default_density)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1024)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    py314 = Path("/opt/miniconda3/envs/py314/bin/python")
    python_exe = str(py314) if py314.exists() else (sys.executable or "python3")
    input_dir = Path(args.input_dir).resolve()
    batch_root = Path(args.batch_root).resolve()
    archive_root = Path(args.archive_dir).resolve()
    outdir = Path(args.outdir).resolve()
    renders_dir = outdir / "renders"
    final_dir = outdir / "final"

    for subject in args.subjects:
        dtseries = require_file(
            input_dir / f"sub-{subject}" / "func" / f"sub-{subject}_task-rest_run-concat.dtseries.nii",
            f"dtseries for subject {subject}",
        )
        if dtseries.stat().st_size <= 0:
            raise RuntimeError(f"Empty dtseries for subject {subject}: {dtseries}")
        surf_dir = batch_root / f"sub-{subject}" / "hippunfold" / f"sub-{subject}" / "surf"
        require_file(
            surf_dir / f"sub-{subject}_hemi-L_space-corobl_den-{args.density}_label-hipp_atlas-multihist7_subfields.label.gii",
            f"left structural label for subject {subject}",
        )
        require_file(
            surf_dir / f"sub-{subject}_hemi-R_space-corobl_den-{args.density}_label-hipp_atlas-multihist7_subfields.label.gii",
            f"right structural label for subject {subject}",
        )

    for subject in args.subjects:
        subject_archive = archive_root / f"sub-{subject}"
        archive_dir(batch_root / f"sub-{subject}" / "post_dense_corobl", subject_archive / "post_dense_corobl")
        archive_dir(batch_root / "final_wb_locked" / f"sub-{subject}", subject_archive / "final_wb_locked")

    renders_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    run(
        [
            python_exe,
            str(repo_root / "scripts" / "workbench" / "render_wb_scene_batch.py"),
            "--scene",
            str((repo_root / args.scene).resolve()),
            "--subjects",
            *args.subjects,
            "--outdir",
            str(renders_dir / "structural"),
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
        subject_final = final_dir / f"sub-{subject}"
        subject_final.mkdir(parents=True, exist_ok=True)
        run(
            [
                python_exe,
                str(repo_root / "scripts" / "workbench" / "compose_wb_with_side_legend.py"),
                "--image",
                str(renders_dir / "structural" / f"sub-{subject}" / f"sub-{subject}_wb_structural_native.png"),
                "--left-labels",
                str(
                    batch_root
                    / f"sub-{subject}"
                    / "hippunfold"
                    / f"sub-{subject}"
                    / "surf"
                    / f"sub-{subject}_hemi-L_space-corobl_den-{args.density}_label-hipp_atlas-multihist7_subfields.label.gii"
                ),
                "--right-labels",
                str(
                    batch_root
                    / f"sub-{subject}"
                    / "hippunfold"
                    / f"sub-{subject}"
                    / "surf"
                    / f"sub-{subject}_hemi-R_space-corobl_den-{args.density}_label-hipp_atlas-multihist7_subfields.label.gii"
                ),
                "--title",
                f"sub-{subject} Structural",
                "--out",
                str(subject_final / f"sub-{subject}_structural.png"),
            ]
        )

    print(final_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

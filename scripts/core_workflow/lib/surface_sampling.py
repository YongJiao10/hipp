#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np

from lib.hipp_density_assets import detect_space_strict, find_surface_asset_strict, subject_surf_dir
from lib.workbench import run_workbench


def sample_hipp_surface_timeseries(
    *,
    bold: Path,
    hippunfold_dir: Path,
    subject: str,
    density: str,
    outdir: Path,
    space: str = "auto",
    mapping_method: str = "enclosing",
    smooth_iters: int = 0,
) -> list[dict[str, object]]:
    outdir.mkdir(parents=True, exist_ok=True)
    surf_dir = subject_surf_dir(hippunfold_dir, subject)
    resolved_space = detect_space_strict(
        surf_dir=surf_dir,
        subject=subject,
        density=density,
        preferred=space,
    )

    summary: list[dict[str, object]] = []
    for hemi in ["L", "R"]:
        mid = find_surface_asset_strict(
            surf_dir=surf_dir,
            subject=subject,
            hemi=hemi,
            space=resolved_space,
            density=density,
            suffix="midthickness.surf.gii",
        )
        out_metric = outdir / f"sub-{subject}_hemi-{hemi}_space-{resolved_space}_den-{density}_label-hipp_bold.func.gii"
        tmp_metric = outdir / f"sub-{subject}_hemi-{hemi}_space-{resolved_space}_den-{density}_label-hipp_bold.tmp.{os.getpid()}.func.gii"
        cmd_args = [
            "-volume-to-surface-mapping",
            str(bold),
            str(mid),
            str(tmp_metric),
            f"-{mapping_method}",
        ]
        try:
            run_workbench(cmd_args)
            metric_img = nib.load(str(tmp_metric))
            metric = np.asarray(metric_img.agg_data(), dtype=np.float32)
            if metric.ndim == 1:
                metric = metric[:, None]
            n_vertices = int(np.asarray(nib.load(str(mid)).agg_data("pointset")).shape[0])
            if metric.shape[0] != n_vertices and metric.ndim == 2 and metric.shape[1] == n_vertices:
                metric = metric.T
            tmp_metric.replace(out_metric)
        finally:
            if tmp_metric.exists():
                tmp_metric.unlink()

        summary.append(
            {
                "hemi": hemi,
                "surface": str(mid),
                "metric_gii": str(out_metric),
                "n_vertices": int(metric.shape[0]),
                "n_timepoints": int(metric.shape[1]),
                "space": resolved_space,
                "mapping_method": mapping_method,
                "smooth_iters": int(smooth_iters),
            }
        )

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample 4D BOLD timeseries onto hippocampal surfaces with wb_command")
    parser.add_argument("--bold", required=True)
    parser.add_argument("--hippunfold-dir", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--density", required=True)
    parser.add_argument("--space", default="auto")
    parser.add_argument("--mapping-method", choices=["enclosing", "trilinear"], default="enclosing")
    parser.add_argument("--smooth-iters", type=int, default=0)
    parser.add_argument("--outdir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = sample_hipp_surface_timeseries(
        bold=Path(args.bold).resolve(),
        hippunfold_dir=Path(args.hippunfold_dir).resolve(),
        subject=str(args.subject),
        density=str(args.density),
        outdir=Path(args.outdir).resolve(),
        space=str(args.space),
        mapping_method=str(args.mapping_method),
        smooth_iters=int(args.smooth_iters),
    )
    outdir = Path(args.outdir).resolve()
    (outdir / "surface_sampling_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

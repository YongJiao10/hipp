#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np

from hipp_density_assets import detect_space_strict, find_surface_asset_strict, subject_surf_dir


def find_wb_command() -> list[str]:
    wrapper = Path(__file__).resolve().parents[1] / "wb_command"
    if not wrapper.exists():
        raise RuntimeError(f"Expected wrapper not found: {wrapper}")
    return [str(wrapper)]


def run_command(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def maybe_smooth(metric: np.ndarray, faces: np.ndarray, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return metric
    if metric.ndim == 1:
        metric = metric[:, None]
    metric = metric.astype(np.float32, copy=True)
    neighbors: list[np.ndarray] = []
    for vertex in range(metric.shape[0]):
        rows = np.where(faces == vertex)[0]
        neighbors.append(np.unique(faces[rows, :]))
    for _ in range(iterations):
        updated = metric.copy()
        for vertex, neigh in enumerate(neighbors):
            updated[vertex, :] = np.nanmean(metric[neigh, :], axis=0)
        metric = updated
    return metric


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample 4D BOLD timeseries onto hippocampal surfaces with wb_command")
    parser.add_argument("--bold", required=True)
    parser.add_argument("--hippunfold-dir", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--density", required=True)
    parser.add_argument("--space", default="auto")
    parser.add_argument("--mapping-method", choices=["enclosing", "trilinear"], default="enclosing")
    parser.add_argument("--smooth-iters", type=int, default=1)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    surf_dir = subject_surf_dir(Path(args.hippunfold_dir), args.subject)
    wb_prefix = find_wb_command()
    resolved_space = detect_space_strict(
        surf_dir=surf_dir,
        subject=args.subject,
        density=args.density,
        preferred=args.space,
    )

    summary: list[dict[str, object]] = []
    for hemi in ["L", "R"]:
        mid = find_surface_asset_strict(
            surf_dir=surf_dir,
            subject=args.subject,
            hemi=hemi,
            space=resolved_space,
            density=args.density,
            suffix="midthickness.surf.gii",
        )
        out_metric = outdir / f"sub-{args.subject}_hemi-{hemi}_space-{resolved_space}_den-{args.density}_label-hipp_bold.func.gii"
        cmd = wb_prefix + [
            "-volume-to-surface-mapping",
            args.bold,
            str(mid),
            str(out_metric),
            f"-{args.mapping_method}",
        ]
        run_command(cmd)

        metric_img = nib.load(str(out_metric))
        metric = np.asarray(metric_img.agg_data(), dtype=np.float32)
        if metric.ndim == 1:
            metric = metric[:, None]
        faces = nib.load(str(mid)).agg_data("triangle")
        n_vertices = int(np.asarray(nib.load(str(mid)).agg_data("pointset")).shape[0])
        if metric.shape[0] != n_vertices and metric.ndim == 2 and metric.shape[1] == n_vertices:
            metric = metric.T
        metric = maybe_smooth(metric, np.asarray(faces), args.smooth_iters)

        npy_path = outdir / f"sub-{args.subject}_hemi-{hemi}_space-{resolved_space}_den-{args.density}_label-hipp_bold.npy"
        np.save(npy_path, metric.astype(np.float32))
        summary.append(
            {
                "hemi": hemi,
                "surface": str(mid),
                "metric_gii": str(out_metric),
                "timeseries_npy": str(npy_path),
                "n_vertices": int(metric.shape[0]),
                "n_timepoints": int(metric.shape[1]),
                "space": resolved_space,
                "mapping_method": args.mapping_method,
                "smooth_iters": int(args.smooth_iters),
            }
        )

    (outdir / "surface_sampling_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

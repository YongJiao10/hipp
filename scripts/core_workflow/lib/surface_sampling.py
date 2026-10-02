#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

import nibabel as nib
import numpy as np

from lib.hipp_density_assets import detect_space_strict, find_surface_asset_strict, subject_surf_dir
from lib.workbench import run_workbench


def _load_metric_gifti(path: Path, *, n_vertices: int, expected_timepoints: int) -> np.ndarray:
    metric = np.asarray(nib.load(str(path)).agg_data(), dtype=np.float32)
    if metric.ndim == 1:
        metric = metric[:, None]
    if metric.ndim != 2:
        raise ValueError(f"Expected 2D metric GIFTI: {path} shape={metric.shape}")
    if metric.shape[0] != n_vertices and metric.shape[1] == n_vertices:
        metric = metric.T
    expected_shape = (int(n_vertices), int(expected_timepoints))
    if metric.shape != expected_shape:
        raise ValueError(f"Metric GIFTI shape mismatch: {path} shape={metric.shape} expected={expected_shape}")
    return metric.astype(np.float32, copy=False)


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


def persistent_sampled_metric_summary(sampled: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Remove paths to per-run metrics that only exist inside a temporary directory."""
    return [{key: value for key, value in record.items() if key != "metric_gii"} for record in sampled]


def sample_hipp_surface_run_timeseries(
    *,
    runs: Sequence[tuple[int, Path]],
    hippunfold_dir: Path,
    subject: str,
    density: str,
    outdir: Path,
    space: str = "auto",
    mapping_method: str = "enclosing",
    smooth_iters: int = 0,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Sample run-level volumes and concatenate only the arrays in memory.

    Workbench receives one source volume at a time.  Its small temporary
    per-run GIFTIs are removed when the call returns; no concatenated volume,
    CIFTI, or run-level output is persisted.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    normalized_runs = [(int(run_id), Path(path).resolve()) for run_id, path in runs]
    if not normalized_runs:
        raise ValueError("At least one run volume is required")
    run_ids = [run_id for run_id, _ in normalized_runs]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError(f"Duplicate run IDs in surface-sampling inputs: {run_ids}")
    if run_ids != sorted(run_ids):
        raise ValueError(f"Run volumes must be ordered by numeric run ID: {run_ids}")

    surf_dir = subject_surf_dir(hippunfold_dir, subject)
    resolved_space = detect_space_strict(
        surf_dir=surf_dir,
        subject=subject,
        density=density,
        preferred=space,
    )
    surface_vertices: dict[str, int] = {}
    for hemi in ("L", "R"):
        mid = find_surface_asset_strict(
            surf_dir=surf_dir,
            subject=subject,
            hemi=hemi,
            space=resolved_space,
            density=density,
            suffix="midthickness.surf.gii",
        )
        surface_vertices[hemi] = int(np.asarray(nib.load(str(mid)).agg_data("pointset")).shape[0])

    by_hemi: dict[str, list[np.ndarray]] = {"L": [], "R": []}
    run_lengths: list[int] = []
    run_records: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix=f"sub-{subject}_run_surface_", dir=str(outdir)) as tmp:
        tmp_root = Path(tmp)
        for run_id, bold in normalized_runs:
            if not bold.is_file():
                raise FileNotFoundError(f"Missing run volume for sub-{subject} run-{run_id}: {bold}")
            bold_img = nib.load(str(bold))
            if len(bold_img.shape) != 4:
                raise ValueError(f"Expected 4D run volume for sub-{subject} run-{run_id}: {bold_img.shape}")
            n_timepoints = int(bold_img.shape[3])
            run_tmp = tmp_root / f"run-{run_id}"
            run_tmp.mkdir(parents=True, exist_ok=True)
            sampled = sample_hipp_surface_timeseries(
                bold=bold,
                hippunfold_dir=hippunfold_dir,
                subject=subject,
                density=density,
                outdir=run_tmp,
                space=resolved_space,
                mapping_method=mapping_method,
                smooth_iters=smooth_iters,
            )
            for hemi in ("L", "R"):
                metric_path = run_tmp / (
                    f"sub-{subject}_hemi-{hemi}_space-{resolved_space}_den-{density}_"
                    "label-hipp_bold.func.gii"
                )
                by_hemi[hemi].append(
                    _load_metric_gifti(
                        metric_path,
                        n_vertices=surface_vertices[hemi],
                        expected_timepoints=n_timepoints,
                    )
                )
            run_lengths.append(n_timepoints)
            run_records.append(
                {
                    "run_id": run_id,
                    "run_label": f"run-{run_id}",
                    "bold": str(bold),
                    "n_timepoints": n_timepoints,
                    "sampled_metrics": persistent_sampled_metric_summary(sampled),
                }
            )

    concatenated = {
        hemi: np.concatenate(blocks, axis=1).astype(np.float32, copy=False)
        for hemi, blocks in by_hemi.items()
    }
    total_timepoints = int(sum(run_lengths))
    if any(int(data.shape[1]) != total_timepoints for data in concatenated.values()):
        raise RuntimeError("Run-level surface arrays do not share the same concatenated time axis")
    return concatenated, {
        "subject": str(subject).replace("sub-", ""),
        "input_mode": "in_memory_run_volume_concat",
        "space": resolved_space,
        "density": str(density),
        "mapping_method": mapping_method,
        "smooth_iters": int(smooth_iters),
        "run_ids": run_ids,
        "run_lengths": run_lengths,
        "n_timepoints": total_timepoints,
        "runs": run_records,
    }


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

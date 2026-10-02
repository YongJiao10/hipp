"""Runwise tSNR for baseline-preserving inputs; never infer a baseline of 10000.

The threshold is an explicit analysis parameter, not a calibrated quality label.
Rest: original mean / linearly detrended sample SD on retained frames.
Task: original mean / task-model residual sample SD, separately for each run.
A combined mask requires each included run to pass.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

REST_POLICY = "runwise_baseline_over_detrended_sd"
TASK_POLICY = "runwise_baseline_over_task_residual_sd"


def runwise_tsnr(runs: Sequence[np.ndarray], *, threshold: float,
                 designs: Sequence[np.ndarray] | None = None,
                 frames: Sequence[np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
    if not runs or not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("tSNR needs explicit runs and a finite positive threshold")
    if designs is not None and len(designs) != len(runs):
        raise ValueError("One task design is required per run")
    if frames is not None and len(frames) != len(runs):
        raise ValueError("One retained-frame index array is required per run")
    scores = []
    n_vertices = np.asarray(runs[0]).shape[0]
    for i, raw in enumerate(runs):
        y = np.asarray(raw, dtype=np.float64)
        if y.ndim != 2 or y.shape[0] != n_vertices:
            raise ValueError(f"Run {i}: expected matching vertex-by-time arrays, got {y.shape}")
        t = np.arange(y.shape[1], dtype=float)
        x = np.column_stack([np.ones(y.shape[1]), t]) if designs is None else np.asarray(designs[i], dtype=float)
        if x.ndim != 2 or x.shape[0] != y.shape[1] or not np.isfinite(x).all():
            raise ValueError(f"Run {i}: invalid or frame-mismatched design {x.shape}")
        if frames is not None:
            keep = np.asarray(frames[i])
            if (keep.ndim != 1 or keep.dtype.kind not in "iu" or keep.size < 3
                    or np.any(keep < 0) or np.any(keep >= y.shape[1]) or np.any(np.diff(keep) <= 0)):
                raise ValueError(f"Run {i}: retained indices must be sorted, unique and in range")
            y, x = y[:, keep], x[keep]
        if y.shape[1] <= np.linalg.matrix_rank(x):
            raise ValueError(f"Run {i}: no residual degrees of freedom")
        good = np.isfinite(y).all(axis=1) & (np.ptp(y, axis=1) > 0)
        score = np.full(n_vertices, np.nan)
        yy = y[good]
        mean = yy.mean(axis=1)
        residual = yy - (x @ (np.linalg.pinv(x, rcond=1e-6) @ yy.T)).T
        sd = residual.std(axis=1, ddof=1)
        # Reject constant/perfectly fitted rows (including numerical fit residue).
        ok = (mean > 0) & (sd > np.finfo(float).eps * np.maximum(np.max(np.abs(yy), axis=1), 1) * 100)
        local = np.full(mean.size, np.nan)
        local[ok] = mean[ok] / sd[ok]
        score[good] = local
        scores.append(score)
    values = np.asarray(scores, dtype=np.float32)
    return values, np.all(np.isfinite(values) & (values >= threshold), axis=0)


def task_tsnr(data: np.ndarray, run_lengths: Sequence[int], bold_paths: Sequence[str | Path],
              *, task: str, threshold: float | None = 25.0) -> tuple[np.ndarray, np.ndarray]:
    from .task_design import build_task_regressors, build_first_level_design, load_nuisance_regressors
    lengths = list(run_lengths)
    if len(lengths) != len(bold_paths) or not lengths or sum(lengths) != data.shape[1]:
        raise ValueError("Task QC requires exact BOLD paths and matching run boundaries")
    runs, designs = [], []
    start = 0
    for n, path in zip(lengths, bold_paths):
        directory = Path(path).parent
        _, ev = build_task_regressors(directory / "EVs", task=task, n_frames=n)
        designs.append(build_first_level_design(ev, load_nuisance_regressors(directory / "Movement_Regressors.txt", n_frames=n)))
        runs.append(data[:, start:start+n])
        start += n
    if threshold is None:
        from .task_quality import task_signal_validity
        valid = task_signal_validity(np.asarray(data, dtype=np.float64), lengths)
        scores = np.full((len(lengths), int(valid.size)), np.nan, dtype=np.float32)
        return scores, valid
    return runwise_tsnr(runs, designs=designs, threshold=float(threshold))


def rest_tsnr_input(path: Path, *, dataset: str, subject: str, hemi: str,
                    n_vertices: int, threshold: float | None = 25.0, n_frames: int | None = None,
                    vertex_ids: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Read explicit baseline samples, including retained original frame indices.

    NPZ is a numeric QC input, not a mask or a substitute for a missing baseline.
    Fields: dataset, subject, hemi, n_runs, run_0..., frames_0...; cortex also
    vertex_ids in the exact consumed CIFTI brain-model order.
    threshold=None skips tSNR (validity-only, still requires the baseline
    receipt for provenance); otherwise the per-run gate applies.
    """
    with np.load(path, allow_pickle=False) as source:
        for key, expected in (("dataset", dataset), ("subject", subject), ("hemi", hemi)):
            if str(source[key].item()) != expected:
                raise ValueError(f"{path}: {key} must be {expected}")
        count = int(source["n_runs"].item())
        runs = [source[f"run_{i}"] for i in range(count)]
        frames = [source[f"frames_{i}"] for i in range(count)]
        if n_frames is not None and sum(len(f) for f in frames) != n_frames:
            raise ValueError(f"{path}: retained QC frames do not match consumed analysis frames ({n_frames})")
        if not runs or any(a.shape[0] != n_vertices for a in runs):
            raise ValueError(f"{path}: baseline vertex count does not match analysis data")
        if vertex_ids is not None and not np.array_equal(source["vertex_ids"], vertex_ids):
            raise ValueError(f"{path}: baseline/analysis CIFTI vertex order differs")
    if threshold is None:
        scores = np.full((count, n_vertices), np.nan, dtype=np.float32)
        return scores, np.ones(n_vertices, dtype=bool)
    return runwise_tsnr(runs, frames=frames, threshold=float(threshold))

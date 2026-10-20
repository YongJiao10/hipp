from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def fisher_z_transform_fc(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    clipped = np.clip(arr, -0.999999, 0.999999)
    out = np.arctanh(clipped).astype(np.float32)
    out[~np.isfinite(arr)] = np.nan
    return out


def finite_pearson_1d(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    finite = np.isfinite(aa) & np.isfinite(bb)
    if int(np.sum(finite)) < 2:
        return float("nan")
    x = aa[finite] - float(np.mean(aa[finite]))
    y = bb[finite] - float(np.mean(bb[finite]))
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if not np.isfinite(denom) or denom <= 0.0:
        return float("nan")
    corr = float(np.dot(x, y) / denom)
    return float(np.clip(corr, -1.0, 1.0)) if np.isfinite(corr) else float("nan")


def center_run_timeseries(values: np.ndarray, run_lengths: Sequence[int]) -> np.ndarray:
    """Remove each declared run's mean without changing run variances or inputs.

    Incomplete rows remain undefined across the complete analysis time axis.
    Stored baseline-bearing series are never modified by this FC operation.
    """
    arr = np.asarray(values, dtype=np.float32)
    lengths = tuple(run_lengths)
    if (arr.ndim != 2 or not lengths
            or any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 2 for n in lengths)
            or sum(lengths) != arr.shape[1]):
        raise ValueError(f"Run lengths must partition the time series into runs of >=2 frames: {lengths}, shape={arr.shape}")
    out = np.full_like(arr, np.nan)
    complete = np.isfinite(arr).all(axis=1)
    start = 0
    for length in lengths:
        stop = start + length
        block = arr[complete, start:stop].astype(np.float64)
        out[complete, start:stop] = block - block.mean(axis=1, keepdims=True)
        start = stop
    return out


def corrcoef_rows(
    a: np.ndarray, b: np.ndarray, *, run_lengths: Sequence[int] | None = None,
) -> np.ndarray:
    """Pearson FC, optionally pooled after removing explicitly declared run means.

    Incomplete or constant rows have undefined correlations. Run centering
    preserves within-run variance weighting rather than normalizing each run.
    """
    x = np.asarray(a, dtype=np.float32)
    y = np.asarray(b, dtype=np.float32)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError(f"Expected 2D frame-matched arrays, got {x.shape} {y.shape}")
    if run_lengths is not None:
        x = center_run_timeseries(x, run_lengths)
        y = center_run_timeseries(y, run_lengths)
    corr = np.full((x.shape[0], y.shape[0]), np.nan, dtype=np.float32)
    if x.shape[1] < 2:
        return corr
    x_indices = np.flatnonzero(np.isfinite(x).all(axis=1))
    y_indices = np.flatnonzero(np.isfinite(y).all(axis=1))
    xx, yy = x[x_indices], y[y_indices]
    x_mu = np.nanmean(xx, axis=1, keepdims=True)
    y_mu = np.nanmean(yy, axis=1, keepdims=True)
    x_sd = np.nanstd(xx, axis=1, keepdims=True, ddof=1)
    y_sd = np.nanstd(yy, axis=1, keepdims=True, ddof=1)
    x_valid = np.isfinite(x_sd[:, 0]) & (x_sd[:, 0] > 0)
    y_valid = np.isfinite(y_sd[:, 0]) & (y_sd[:, 0] > 0)
    x_z = (xx[x_valid] - x_mu[x_valid]) / np.clip(x_sd[x_valid], 1e-12, None)
    y_z = (yy[y_valid] - y_mu[y_valid]) / np.clip(y_sd[y_valid], 1e-12, None)
    values = (x_z @ y_z.T) / float(x.shape[1] - 1)
    corr[np.ix_(x_indices[x_valid], y_indices[y_valid])] = np.clip(values, -0.999999, 0.999999)
    return corr


def compact_active_vertices(values: np.ndarray, support_mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    mask = np.asarray(support_mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError(f"support_mask must be 1D, got {mask.shape}")
    if arr.ndim < 1 or int(arr.shape[0]) != int(mask.size):
        raise ValueError(f"Leading dimension mismatch: values={arr.shape}, support_mask={mask.shape}")
    return arr[mask]


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    q = np.full_like(p, np.nan, dtype=np.float64)
    finite = np.isfinite(p)
    m = int(np.sum(finite))
    if m == 0:
        return q
    p_f = p[finite]
    order = np.argsort(p_f)
    sorted_p = p_f[order]
    ranks = np.arange(1, m + 1, dtype=np.float64)
    adjusted = sorted_p * float(m) / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    q_f = np.empty_like(sorted_p)
    q_f[order] = adjusted
    q[finite] = q_f
    return q


from __future__ import annotations

import numpy as np


def fisher_z_transform_fc(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    clipped = np.clip(arr, -0.999999, 0.999999)
    out = np.arctanh(clipped).astype(np.float32)
    out[~np.isfinite(arr)] = np.nan
    return out


def corrcoef_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=np.float32)
    y = np.asarray(b, dtype=np.float32)
    x_mu = np.nanmean(x, axis=1, keepdims=True)
    y_mu = np.nanmean(y, axis=1, keepdims=True)
    x_sd = np.nanstd(x, axis=1, keepdims=True, ddof=1)
    y_sd = np.nanstd(y, axis=1, keepdims=True, ddof=1)
    x_z = (x - x_mu) / np.clip(x_sd, 1e-12, None)
    y_z = (y - y_mu) / np.clip(y_sd, 1e-12, None)
    x_z[~np.isfinite(x_z)] = 0.0
    y_z[~np.isfinite(y_z)] = 0.0
    corr = (x_z @ y_z.T) / float(max(1, x.shape[1] - 1))
    corr = np.clip(corr, -0.999999, 0.999999)
    corr[~np.isfinite(corr)] = 0.0
    return corr.astype(np.float32)


def compact_active_vertices(values: np.ndarray, support_mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    mask = np.asarray(support_mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError(f"support_mask must be 1D, got {mask.shape}")
    if arr.ndim < 1 or int(arr.shape[0]) != int(mask.size):
        raise ValueError(f"Leading dimension mismatch: values={arr.shape}, support_mask={mask.shape}")
    return arr[mask]

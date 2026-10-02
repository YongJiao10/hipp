#!/usr/bin/env python3
"""First-level task GLM numeric helpers shared by stage 04_03/04_04.

Ordinary least squares per surface vertex: ``beta = pinv(X) @ y``.
Invalid (non-finite) vertices propagate as NaN; no imputation or fallback
is performed.
"""

from __future__ import annotations

import numpy as np


def fit_first_level_glm(
    timeseries: np.ndarray, design: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit ``Y (vertices x frames) = beta @ X`` and return betas/residuals/R².

    Returns ``(betas, residuals, r_squared)`` with betas shaped
    ``(n_vertices, n_regressors)`` and residuals shaped like ``timeseries``.
    """
    y = np.asarray(timeseries, dtype=np.float64)
    x = np.asarray(design, dtype=np.float64)
    if y.ndim != 2 or x.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got y={y.shape} x={x.shape}")
    if y.shape[1] != x.shape[0]:
        raise ValueError(f"Frame mismatch: y={y.shape} x={x.shape}")
    pinv = np.linalg.pinv(x, rcond=1e-6)
    betas = (pinv @ y.T).T
    fitted = (x @ betas.T).T
    residuals = y - fitted
    finite = np.isfinite(y)
    ss_res = np.sum(np.where(finite, (y - fitted) ** 2, 0.0), axis=1)
    means = np.sum(np.where(finite, y, 0.0), axis=1) / np.maximum(
        np.sum(finite, axis=1), 1
    )
    ss_tot = np.sum(np.where(finite, (y - means[:, None]) ** 2, 0.0), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = np.where(ss_tot > 0, 1.0 - ss_res / np.maximum(ss_tot, 1e-12), np.nan)
    invalid_rows = ~np.all(finite, axis=1)
    betas = betas.astype(np.float64)
    betas[invalid_rows, :] = np.nan
    residuals = residuals.astype(np.float64)
    residuals[invalid_rows, :] = np.nan
    r2 = np.asarray(r2, dtype=np.float64)
    r2[invalid_rows] = np.nan
    return betas, residuals, r2


def apply_contrast(betas: np.ndarray, contrast: np.ndarray) -> np.ndarray:
    """Apply a task-regressor contrast vector to per-vertex betas."""
    b = np.asarray(betas, dtype=np.float64)
    c = np.asarray(contrast, dtype=np.float64).reshape(-1)
    if b.shape[1] < c.size:
        raise ValueError(f"Beta columns {b.shape} smaller than contrast {c.shape}")
    return (b[:, : c.size] @ c).astype(np.float64)


def background_residuals(
    timeseries: np.ndarray, design: np.ndarray, n_task_regressors: int
) -> np.ndarray:
    """Remove task + nuisance fit, keeping residuals for background FC.

    The full design (task + motion + intercept + drift) is regressed out;
    ``n_task_regressors`` is validated but the residual keeps no task
    variance by construction.
    """
    if int(n_task_regressors) <= 0 or int(n_task_regressors) > int(design.shape[1]):
        raise ValueError(f"Invalid n_task_regressors={n_task_regressors}")
    _, residuals, _ = fit_first_level_glm(timeseries, design)
    return residuals.astype(np.float64)


def corrcoef_rows_nan(a: np.ndarray, b: np.ndarray, *, min_frames: int = 3) -> np.ndarray:
    """Pairwise-complete Pearson correlations with NaN propagation.

    Unlike ``fc_math.corrcoef_rows`` (which zeroes non-finite entries and can
    emit FC=0 for missing vertices), any row pair without enough jointly
    valid frames yields NaN, so missing signal never masquerades as zero
    connectivity. Row means use all finite entries of that row.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError(f"Expected 2D frame-matched arrays, got {x.shape} {y.shape}")
    fx = np.isfinite(x)
    fy = np.isfinite(y)
    with np.errstate(invalid="ignore"):
        x_mean = np.sum(np.where(fx, x, 0.0), axis=1, keepdims=True) / np.maximum(
            np.sum(fx, axis=1, keepdims=True), 1
        )
        y_mean = np.sum(np.where(fy, y, 0.0), axis=1, keepdims=True) / np.maximum(
            np.sum(fy, axis=1, keepdims=True), 1
        )
    xc = np.where(fx, x - x_mean, 0.0)
    yc = np.where(fy, y - y_mean, 0.0)
    n = fx.astype(np.float64) @ fy.astype(np.float64).T
    num = xc @ yc.T
    denom = np.sqrt(np.maximum((xc**2) @ fy.astype(np.float64).T, 0.0)) * np.sqrt(
        np.maximum(fx.astype(np.float64) @ (yc**2).T, 0.0)
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(
            (n >= int(min_frames)) & (denom > 0) & np.isfinite(denom),
            np.clip(num / np.maximum(denom, 1e-300), -1.0, 1.0),
            np.nan,
        )
    return out.astype(np.float64)

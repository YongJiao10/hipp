#!/usr/bin/env python3
"""Task-fMRI design-matrix helpers shared by numbered stage 04_03/04_04.

Parses FSL 3-column EV files (``onset duration weight`` in seconds),
convolves them with a canonical SPM double-gamma HRF sampled at the task
TR, and builds first-level design matrices plus maintained contrast
vectors.

No filesystem fallback is performed here: callers pass exact EV paths and
frame counts, and any mismatch fails closed.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


TR_HCP_3T = 0.72
TASK_HIGH_PASS_PERIOD_S = 128.0
N_GRAYORDINATES_91K = 91282

# Expected run lengths for the maintained HCP 3T task branches.
TASK_RUN_LENGTHS = {
    "emotion": 176,
    "wm": 405,
    "relational": 232,
}

# HCP run directory labels, ordered LR then RL, per maintained task branch.
TASK_RUN_LABELS = {
    "emotion": ("tfMRI_EMOTION_LR", "tfMRI_EMOTION_RL"),
    "wm": ("tfMRI_WM_LR", "tfMRI_WM_RL"),
    "relational": ("tfMRI_RELATIONAL_LR", "tfMRI_RELATIONAL_RL"),
}

# EV stems (without ``.txt``) per maintained task branch.
TASK_EV_STEMS = {
    "emotion": ("fear", "neut"),
    "wm": (
        "0bk_faces",
        "0bk_places",
        "0bk_tools",
        "0bk_body",
        "2bk_faces",
        "2bk_places",
        "2bk_tools",
        "2bk_body",
    ),
    "relational": ("rel", "match"),
}

# Maintained contrasts as {contrast_name: {regressor: weight}}.
TASK_CONTRASTS = {
    "emotion": {
        "faces_gt_shapes": {"fear": 1.0, "neut": -1.0},
    },
    "wm": {
        "twobk_gt_zerobk": {
            "0bk_faces": -0.25,
            "0bk_places": -0.25,
            "0bk_tools": -0.25,
            "0bk_body": -0.25,
            "2bk_faces": 0.25,
            "2bk_places": 0.25,
            "2bk_tools": 0.25,
            "2bk_body": 0.25,
        },
        "twobk_gt_zerobk_faces": {
            "0bk_faces": -1.0,
            "2bk_faces": 1.0,
        },
        "twobk_gt_zerobk_places": {
            "0bk_places": -1.0,
            "2bk_places": 1.0,
        },
        "twobk_gt_zerobk_tools": {
            "0bk_tools": -1.0,
            "2bk_tools": 1.0,
        },
        "twobk_gt_zerobk_body": {
            "0bk_body": -1.0,
            "2bk_body": 1.0,
        },
        "faces_gt_others": {
            "0bk_faces": 0.5,
            "0bk_places": -1.0 / 6.0,
            "0bk_tools": -1.0 / 6.0,
            "0bk_body": -1.0 / 6.0,
            "2bk_faces": 0.5,
            "2bk_places": -1.0 / 6.0,
            "2bk_tools": -1.0 / 6.0,
            "2bk_body": -1.0 / 6.0,
        },
        "places_gt_others": {
            "0bk_faces": -1.0 / 6.0,
            "0bk_places": 0.5,
            "0bk_tools": -1.0 / 6.0,
            "0bk_body": -1.0 / 6.0,
            "2bk_faces": -1.0 / 6.0,
            "2bk_places": 0.5,
            "2bk_tools": -1.0 / 6.0,
            "2bk_body": -1.0 / 6.0,
        },
        "tools_gt_others": {
            "0bk_faces": -1.0 / 6.0,
            "0bk_places": -1.0 / 6.0,
            "0bk_tools": 0.5,
            "0bk_body": -1.0 / 6.0,
            "2bk_faces": -1.0 / 6.0,
            "2bk_places": -1.0 / 6.0,
            "2bk_tools": 0.5,
            "2bk_body": -1.0 / 6.0,
        },
        "body_gt_others": {
            "0bk_faces": -1.0 / 6.0,
            "0bk_places": -1.0 / 6.0,
            "0bk_tools": -1.0 / 6.0,
            "0bk_body": 0.5,
            "2bk_faces": -1.0 / 6.0,
            "2bk_places": -1.0 / 6.0,
            "2bk_tools": -1.0 / 6.0,
            "2bk_body": 0.5,
        },
        "load_x_faces": {
            "0bk_faces": -0.5,
            "0bk_places": 1.0 / 6.0,
            "0bk_tools": 1.0 / 6.0,
            "0bk_body": 1.0 / 6.0,
            "2bk_faces": 0.5,
            "2bk_places": -1.0 / 6.0,
            "2bk_tools": -1.0 / 6.0,
            "2bk_body": -1.0 / 6.0,
        },
        "load_x_places": {
            "0bk_faces": 1.0 / 6.0,
            "0bk_places": -0.5,
            "0bk_tools": 1.0 / 6.0,
            "0bk_body": 1.0 / 6.0,
            "2bk_faces": -1.0 / 6.0,
            "2bk_places": 0.5,
            "2bk_tools": -1.0 / 6.0,
            "2bk_body": -1.0 / 6.0,
        },
        "load_x_tools": {
            "0bk_faces": 1.0 / 6.0,
            "0bk_places": 1.0 / 6.0,
            "0bk_tools": -0.5,
            "0bk_body": 1.0 / 6.0,
            "2bk_faces": -1.0 / 6.0,
            "2bk_places": -1.0 / 6.0,
            "2bk_tools": 0.5,
            "2bk_body": -1.0 / 6.0,
        },
        "load_x_body": {
            "0bk_faces": 1.0 / 6.0,
            "0bk_places": 1.0 / 6.0,
            "0bk_tools": 1.0 / 6.0,
            "0bk_body": -0.5,
            "2bk_faces": -1.0 / 6.0,
            "2bk_places": -1.0 / 6.0,
            "2bk_tools": -1.0 / 6.0,
            "2bk_body": 0.5,
        },
    },
    "relational": {
        "rel_gt_match": {"rel": 1.0, "match": -1.0},
    },
}


def normalize_task_branch(task: str) -> str:
    key = str(task).strip().lower().replace("-", "_")
    if key not in TASK_RUN_LENGTHS:
        raise ValueError(
            f"Unknown task branch: {task!r}; expected one of {sorted(TASK_RUN_LENGTHS)}"
        )
    return key


def parse_fsl_ev(path: Path) -> np.ndarray:
    """Parse one FSL 3-column EV file into ``(n_events, 3)`` seconds."""
    arr = np.loadtxt(str(path), dtype=np.float64, ndmin=2)
    if arr.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected FSL 3-column EV file: {path} shape={arr.shape}")
    if np.any(arr[:, 1] <= 0):
        raise ValueError(f"EV durations must be positive: {path}")
    return arr.astype(np.float64, copy=False)


def _gamma_pdf(x: np.ndarray, shape: float, scale: float) -> np.ndarray:
    """Gamma PDF matching SPM's spm_Gpdf (shape h>0, scale l>0)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    positive = (x > 0) & np.isfinite(x)
    xx = x[positive]
    with np.errstate(divide="ignore", invalid="ignore"):
        out[positive] = np.exp(
            (shape - 1.0) * np.log(xx)
            + shape * math.log(scale)
            - scale * xx
            - math.lgamma(shape)
        )
    out[~np.isfinite(out)] = 0.0
    return out


def spm_hrf(tr: float, length_s: float = 32.0) -> np.ndarray:
    """Canonical SPM double-gamma HRF sampled at ``tr`` seconds.

    Mirrors SPM12 ``spm_hrf.m`` with default parameters (response delay 6,
    undershoot delay 16, dispersions 1, ratio 6, onset 0, length 32): the
    mixture of two Gamma densities is evaluated at TR spacing and normalized
    to sum to 1, exactly as SPM does after microtime sampling. Peak lands at
    ~5 s, not ~5.8 s as with the previous ad-hoc formula.
    """
    if tr <= 0:
        raise ValueError(f"TR must be positive, got {tr}")
    if length_s <= 0:
        raise ValueError(f"HRF length must be positive, got {length_s}")
    # SPM evaluates the mixture in microtime bins and sum-normalizes, which
    # makes the continuous kernel independent of sampling: response ~ t^5 e^-t
    # (peak 5 s) minus undershoot ~ t^15 e^-t / 6. Evaluating Gamma densities
    # with scale=1 s at TR spacing reproduces exactly that kernel.
    t = np.arange(0.0, length_s + tr / 2.0, tr)
    response = _gamma_pdf(t, 6.0, 1.0)
    undershoot = _gamma_pdf(t, 16.0, 1.0)
    hrf = response - undershoot / 6.0
    total = float(np.sum(hrf))
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("SPM HRF computation produced non-finite values")
    return (hrf / total).astype(np.float64)


def convolve_ev_to_run(events: np.ndarray, n_frames: int, tr: float) -> np.ndarray:
    """Convolve onset/duration block events to one HRF regressor of ``n_frames``."""
    hrf = spm_hrf(tr)
    stim = np.zeros(n_frames, dtype=np.float64)
    for onset, duration, weight in events.tolist():
        start = int(round(float(onset) / tr))
        n = max(int(round(float(duration) / tr)), 1)
        stop = min(start + n, n_frames)
        if start < n_frames and stop > max(start, 0):
            stim[max(start, 0) : stop] += float(weight)
    conv = np.convolve(stim, hrf)[:n_frames]
    return conv.astype(np.float64)


def build_task_regressors(
    ev_dir: Path, *, task: str, n_frames: int, tr: float = TR_HCP_3T
) -> tuple[list[str], np.ndarray]:
    """Build HRF-convolved task regressors from exact EV files.

    Every maintained EV stem must exist; a missing stem fails closed so a
    task run can never silently fall back to a reduced design.
    """
    branch = normalize_task_branch(task)
    expected = TASK_RUN_LENGTHS[branch]
    if int(n_frames) != int(expected):
        raise ValueError(
            f"Task {branch} expects {expected} frames, got {n_frames}"
        )
    names: list[str] = []
    cols: list[np.ndarray] = []
    for stem in TASK_EV_STEMS[branch]:
        ev_path = Path(ev_dir) / f"{stem}.txt"
        if not ev_path.is_file():
            raise FileNotFoundError(f"Missing required EV file for task {branch}: {ev_path}")
        events = parse_fsl_ev(ev_path)
        names.append(stem)
        cols.append(convolve_ev_to_run(events, int(n_frames), float(tr)))
    design = np.column_stack(cols) if cols else np.zeros((int(n_frames), 0))
    return names, design.astype(np.float64)


def load_nuisance_regressors(path: Path, *, n_frames: int) -> np.ndarray:
    """Load the 12-column HCP ``Movement_Regressors.txt`` for one run."""
    arr = np.loadtxt(str(path), dtype=np.float64, ndmin=2)
    if arr.shape != (int(n_frames), 12):
        raise ValueError(
            f"Expected Movement_Regressors shape ({n_frames}, 12): {path} shape={arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Non-finite values in movement regressors: {path}")
    return arr.astype(np.float64)


def build_first_level_design(
    task_regressors: np.ndarray, nuisance: np.ndarray, *, tr: float = TR_HCP_3T
) -> np.ndarray:
    """Stack ``[task | nuisance | intercept | 128-s DCT drift]`` for one run.

    The nonconstant DCT-II columns match SPM12 ``spm_filter``/``spm_dctmtx``.
    Joint OLS with these nuisance columns is equivalent to applying the
    DCT high-pass projection to both data and model before OLS. The explicit
    intercept is retained; no additional linear trend is added.
    """
    n = int(task_regressors.shape[0])
    if int(nuisance.shape[0]) != n:
        raise ValueError(
            f"Frame mismatch: task={task_regressors.shape} nuisance={nuisance.shape}"
        )
    intercept = np.ones((n, 1), dtype=np.float64)
    # SPM uses fix(2 * n * TR / HParam + 1) DCT columns including the
    # constant, then excludes that constant from the high-pass basis.
    n_drift = int(np.fix(2.0 * n * tr / TASK_HIGH_PASS_PERIOD_S + 1.0)) - 1
    orders = np.arange(1, n_drift + 1, dtype=np.float64)
    samples = np.arange(n, dtype=np.float64)[:, None]
    drift = np.sqrt(2.0 / n) * np.cos(np.pi * (samples + 0.5) * orders / n)
    return np.concatenate([task_regressors, nuisance, intercept, drift], axis=1)


def contrast_vector(
    regressor_names: list[str], weights: dict[str, float]
) -> np.ndarray:
    """Build a contrast vector over task-regressor columns only."""
    vec = np.zeros(len(regressor_names), dtype=np.float64)
    unknown = [k for k in weights if k not in regressor_names]
    if unknown:
        raise ValueError(f"Contrast references unknown regressors: {unknown}")
    for key, weight in weights.items():
        vec[regressor_names.index(key)] = float(weight)
    if not np.any(vec != 0.0):
        raise ValueError("Contrast vector is all zeros")
    return vec

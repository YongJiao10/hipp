"""QC policy identifiers and basic validity prerequisites.

Basic validity helpers are not tSNR gates; production gates use lib.tsnr.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .tsnr import TASK_POLICY, REST_POLICY

TASK_VALIDITY_POLICY = TASK_POLICY


def parse_tsnr_threshold(value: str | float | None) -> float | None:
    """Arbitrary screening criterion: positive number gates, none disables.

    Accepts a positive float (any cutoff), or ``none``/``null`` (case
    insensitive) / ``None`` for validity-only with no tSNR screening.
    Anything else fails closed.
    """
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("none", "null", "off"):
        return None
    threshold = float(value)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError(f"tSNR threshold must be a positive number or none, got {value!r}")
    return threshold


def task_signal_validity(data: np.ndarray, run_lengths: Sequence[int]) -> np.ndarray:
    values = np.asarray(data, dtype=np.float64)
    lengths = list(run_lengths)
    if (values.ndim != 2 or not lengths
            or any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 2 for n in lengths)
            or sum(lengths) != values.shape[1]):
        raise ValueError(f"Task run lengths must partition the timeseries into runs of >=2 frames: {lengths}, shape={values.shape}")
    valid = np.ones(values.shape[0], dtype=bool)
    start = 0
    for length in lengths:
        block = values[:, start:start + length]
        finite = np.all(np.isfinite(block), axis=1)
        # Compare extrema only on finite rows, avoiding NaN/Inf arithmetic.
        variable = np.zeros(values.shape[0], dtype=bool)
        variable[finite] = np.max(block[finite], axis=1) > np.min(block[finite], axis=1)
        valid &= finite & variable
        start += length
    return valid


def require_task_policy(payload: dict, path: Path, task: str) -> None:
    if payload.get("validity_policy") != TASK_VALIDITY_POLICY or payload.get("task") != task:
        raise ValueError(f"Rebuild task inputs with task={task}, validity_policy={TASK_VALIDITY_POLICY}: {path}")


def require_task_surface_validity(surface_root: Path, subject: str, hemi: str, task: str) -> None:
    surface = Path(surface_root) / f"sub-{subject}" / "surface"
    stage = surface / "stage_manifest.json"
    params = json.loads(stage.read_text(encoding="utf-8")).get("params", {})
    require_task_policy(params, stage, task)
    gate = surface / "tsnr_gate" / f"sub-{subject}_hemi-{hemi}_tsnr_gate_summary.json"
    require_task_policy(json.loads(gate.read_text(encoding="utf-8")), gate, task)


REST_VALIDITY_POLICY = REST_POLICY
QC_DATASETS = ("hcp", "stanford", "iowa_jj_tms", "iowa7t")

def signal_validity_policy(dataset: str, task: str) -> str:
    if dataset not in QC_DATASETS:
        raise ValueError(f"Unsupported QC dataset: {dataset}; expected {QC_DATASETS}")
    if task != "rest":
        if dataset != "hcp":
            raise ValueError(f"Task QC is only implemented for HCP, got {dataset}/{task}")
        return TASK_VALIDITY_POLICY
    return "hcp_10000_over_std" if dataset == "hcp" else REST_VALIDITY_POLICY


def rest_signal_validity(data: np.ndarray) -> np.ndarray:
    """Finite, nonconstant complete residual series; no runwise rule or SD ceiling."""
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError(f"Expected residual series with >=2 frames, got {values.shape}")
    finite = np.isfinite(values).all(axis=1)
    valid = np.zeros(values.shape[0], dtype=bool)
    valid[finite] = np.max(values[finite], axis=1) > np.min(values[finite], axis=1)
    return valid

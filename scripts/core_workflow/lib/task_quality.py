"""QC policy identifiers and signal-validity prerequisites."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

TASK_VALIDITY_POLICY = "finite_nonconstant_each_run"
CORTEX_VALIDITY_POLICY = "finite_cortex_signal"
SERIES_VALIDITY_POLICY = "finite_nonconstant_series"


def _validate_runs(data: np.ndarray, run_lengths: Sequence[int]) -> tuple[np.ndarray, list[int]]:
    values = np.asarray(data)
    lengths = list(run_lengths)
    if (values.ndim != 2 or not lengths
            or any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 2 for n in lengths)
            or sum(lengths) != values.shape[1]):
        raise ValueError(f"Task run lengths must partition the timeseries into runs of >=2 frames: {lengths}, shape={values.shape}")
    return values, lengths


def cortex_signal_validity(data: np.ndarray, run_lengths: Sequence[int]) -> np.ndarray:
    values, _ = _validate_runs(data, run_lengths)
    return np.isfinite(values).all(axis=1)


def task_signal_validity(data: np.ndarray, run_lengths: Sequence[int]) -> np.ndarray:
    values, lengths = _validate_runs(data, run_lengths)
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
    gate = surface / "validity" / f"sub-{subject}_hemi-{hemi}_validity_summary.json"
    require_task_policy(json.loads(gate.read_text(encoding="utf-8")), gate, task)


REST_VALIDITY_POLICY = TASK_VALIDITY_POLICY
QC_DATASETS = ("hcp", "stanford", "iowa_jj_tms", "iowa7t", "banda", "pdc", "nih_rtms")

def signal_validity_policy(dataset: str, task: str) -> str:
    if dataset not in QC_DATASETS:
        raise ValueError(f"Unsupported QC dataset: {dataset}; expected {QC_DATASETS}")
    if task != "rest":
        if dataset != "hcp":
            raise ValueError(f"Task QC is only implemented for HCP, got {dataset}/{task}")
        return TASK_VALIDITY_POLICY
    return REST_VALIDITY_POLICY


def rest_signal_validity(data: np.ndarray) -> np.ndarray:
    """Finite, nonconstant complete residual series; no runwise rule or SD ceiling."""
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError(f"Expected residual series with >=2 frames, got {values.shape}")
    finite = np.isfinite(values).all(axis=1)
    valid = np.zeros(values.shape[0], dtype=bool)
    valid[finite] = np.max(values[finite], axis=1) > np.min(values[finite], axis=1)
    return valid


def require_cortex_reference_policy(
    ref_dir: Path, *, dataset: str, task: str | None = None,
) -> dict:
    """Require one declared cortex QC policy before consuming a reference."""
    from .task_design import normalize_task_branch

    def read(path: Path) -> dict:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("expected a JSON object")
            return payload
        except (OSError, ValueError) as exc:
            raise ValueError(f"Rebuild cortex reference: missing or malformed policy metadata at {path}: {exc}") from exc

    def require(payload: dict, path: Path, key: str, expected: object) -> None:
        if key not in payload or payload[key] != expected:
            raise ValueError(
                f"Rebuild cortex reference: {path}: {key}={payload.get(key)!r}, expected {expected!r}"
            )

    ref_dir = Path(ref_dir)
    summary_path = ref_dir / "reference_summary.json"
    summary = read(summary_path)
    declared_task = summary.get("task")
    if not isinstance(declared_task, str):
        raise ValueError(f"Rebuild cortex reference: missing task at {summary_path}")
    selected_task = task if task is not None else declared_task
    try:
        selected_task = normalize_task_branch(selected_task) if selected_task != "rest" else "rest"
        signal_validity_policy(dataset, selected_task)
        policy = CORTEX_VALIDITY_POLICY
    except ValueError as exc:
        raise ValueError(f"Rebuild cortex reference: invalid dataset/task policy at {summary_path}: {exc}") from exc
    for key, expected in (("dataset", dataset), ("task", selected_task), ("validity_policy", policy)):
        require(summary, summary_path, key, expected)
    stage_path = ref_dir / "stage_manifest.json"
    stage = read(stage_path)
    require(stage, stage_path, "status", "done")
    params = stage.get("params")
    if not isinstance(params, dict):
        raise ValueError(f"Rebuild cortex reference: missing params at {stage_path}")
    for key, expected in (("dataset", dataset), ("task", selected_task), ("validity_policy", policy)):
        require(params, stage_path, key, expected)
    for side in ("left", "right"):
        gate_path = ref_dir.parent / "common" / "validity" / f"cortex_{side}_validity_summary.json"
        gate = read(gate_path)
        for key, expected in (("dataset", dataset), ("task", selected_task), ("validity_policy", policy)):
            require(gate, gate_path, key, expected)
    return summary

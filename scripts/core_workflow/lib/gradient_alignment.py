from __future__ import annotations

from itertools import permutations
from typing import Any
import numpy as np
from .fc_math import finite_pearson_1d

def finite_pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    corr = finite_pearson_1d(a, b)
    return float(corr) if np.isfinite(corr) else 0.0


def best_abs_correlation_assignment(corr: np.ndarray) -> list[int]:
    mat = np.asarray(corr, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Expected square correlation matrix, got shape={mat.shape}")
    n_comp = int(mat.shape[0])
    best_perm: tuple[int, ...] | None = None
    best_score = -np.inf
    best_identity_hits = -1
    for perm in permutations(range(n_comp)):
        score = float(sum(abs(mat[target_idx, source_idx]) for target_idx, source_idx in enumerate(perm)))
        identity_hits = sum(1 for idx, source_idx in enumerate(perm) if idx == source_idx)
        if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and identity_hits > best_identity_hits):
            best_score = score
            best_identity_hits = int(identity_hits)
            best_perm = tuple(int(x) for x in perm)
    if best_perm is None:
        raise RuntimeError("Could not compute component assignment")
    return list(best_perm)


def align_gradient_components_to_reference(
    *,
    active: np.ndarray,
    full: np.ndarray,
    lambdas: np.ndarray,
    ev_fractions: np.ndarray,
    reference_full: np.ndarray,
    component_labels: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    active_arr = np.asarray(active, dtype=np.float32)
    full_arr = np.asarray(full, dtype=np.float32)
    ref_arr = np.asarray(reference_full, dtype=np.float32)
    lambda_arr = np.asarray(lambdas, dtype=np.float64).reshape(-1)
    ev_arr = np.asarray(ev_fractions, dtype=np.float64).reshape(-1)
    labels = [str(x) for x in component_labels]
    if active_arr.ndim != 2 or full_arr.ndim != 2 or ref_arr.ndim != 2:
        raise ValueError(
            f"Expected 2D active/full/reference arrays, got active={active_arr.shape} full={full_arr.shape} ref={ref_arr.shape}"
        )
    n_comp = len(labels)
    if n_comp == 0:
        raise ValueError("component_labels must not be empty")
    if active_arr.shape[1] < n_comp or full_arr.shape[1] < n_comp or ref_arr.shape[1] < n_comp:
        raise ValueError(
            f"Component count mismatch: active={active_arr.shape} full={full_arr.shape} ref={ref_arr.shape} labels={n_comp}"
        )
    if lambda_arr.size < n_comp or ev_arr.size < n_comp:
        raise ValueError(f"Lambda/EV vectors must contain at least {n_comp} values")

    corr = np.zeros((n_comp, n_comp), dtype=np.float64)
    for target_idx in range(n_comp):
        for source_idx in range(n_comp):
            corr[target_idx, source_idx] = finite_pearson_corr(ref_arr[:, target_idx], full_arr[:, source_idx])
    source_for_target = best_abs_correlation_assignment(corr)

    aligned_active = np.zeros((active_arr.shape[0], n_comp), dtype=np.float32)
    aligned_full = np.zeros((full_arr.shape[0], n_comp), dtype=np.float32)
    aligned_lambdas = np.zeros((n_comp,), dtype=np.float64)
    aligned_ev = np.zeros((n_comp,), dtype=np.float64)
    mapping: dict[str, str] = {}
    corr_by_label: dict[str, float] = {}
    flips: dict[str, bool] = {}
    for target_idx, source_idx in enumerate(source_for_target):
        label = labels[target_idx]
        source_label = labels[source_idx]
        raw_corr = float(corr[target_idx, source_idx])
        sign = -1.0 if raw_corr < 0.0 else 1.0
        aligned_active[:, target_idx] = (active_arr[:, source_idx] * sign).astype(np.float32)
        aligned_full[:, target_idx] = (full_arr[:, source_idx] * sign).astype(np.float32)
        aligned_lambdas[target_idx] = float(lambda_arr[source_idx])
        aligned_ev[target_idx] = float(ev_arr[source_idx])
        mapping[label] = source_label
        corr_by_label[label] = raw_corr
        flips[label] = bool(sign < 0.0)

    meta = {
        "matching_objective": "maximize_abs_pearson_correlation_to_anchor",
        "component_mapping": mapping,
        "correlation_before_flip": corr_by_label,
        "sign_flip_applied": flips,
        "raw_arrays_saved": False,
    }
    return aligned_active, aligned_full, aligned_lambdas, aligned_ev, meta

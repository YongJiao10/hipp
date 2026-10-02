from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from lib.fc_math import corrcoef_rows, fisher_z_transform_fc


def _network_column(network_column: str) -> str:
    col = str(network_column)
    if col not in {"canonical8", "canonical7", "family17"}:
        raise ValueError(f"Unsupported network column: {network_column!r}")
    return col


def _row_weight(row: dict[str, object]) -> float:
    raw = row.get("n_grayordinates_used", row.get("weight", 0))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 0.0
    return value if math.isfinite(value) and value > 0.0 else 0.0


def _validate_rows(values: np.ndarray, rows: Sequence[dict[str, object]]) -> None:
    if values.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={values.shape}")
    if values.shape[0] != len(rows):
        raise ValueError(f"Row count mismatch: values={values.shape[0]} rows={len(rows)}")


def weighted_network_timeseries_from_roi_timeseries(
    roi_timeseries: np.ndarray,
    roi_rows: Sequence[dict[str, object]],
    network_names: Sequence[str],
    *,
    network_column: str = "canonical7",
) -> np.ndarray:
    """Collapse ROI mean time series to network time series using ROI vertex counts.

    Each ROI time series is already a mean over its valid cortex grayordinates.
    Weighting by `n_grayordinates_used` reconstructs the same network mean that
    would be obtained by averaging all valid vertices in the network directly.
    """
    arr = np.asarray(roi_timeseries, dtype=np.float32)
    _validate_rows(arr, roi_rows)
    col = _network_column(network_column)
    out = np.full((len(network_names), arr.shape[1]), np.nan, dtype=np.float32)
    for net_idx, net_name in enumerate(network_names):
        idx: list[int] = []
        weights: list[float] = []
        for row_idx, row in enumerate(roi_rows):
            if str(row.get(col, "")) != str(net_name):
                continue
            weight = _row_weight(row)
            if weight <= 0.0:
                continue
            idx.append(row_idx)
            weights.append(weight)
        if not idx:
            continue
        values = arr[np.asarray(idx, dtype=np.int32), :].astype(np.float64)
        w = np.asarray(weights, dtype=np.float64)
        out[net_idx, :] = np.average(values, axis=0, weights=w).astype(np.float32)
    return out


def weighted_network_profile_from_roi_profile(
    roi_profile: np.ndarray,
    roi_rows: Sequence[dict[str, object]],
    network_names: Sequence[str],
    *,
    network_column: str = "canonical7",
) -> np.ndarray:
    """Collapse ROI Fisher-z profiles to networks using ROI vertex counts.

    This is useful for auditing old ROI-profile products, but new branch
    materialization should prefer recomputing FC against weighted network time
    series before Fisher-z transformation.
    """
    arr = np.asarray(roi_profile, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D ROI profile, got shape={arr.shape}")
    if arr.shape[1] != len(roi_rows):
        raise ValueError(f"Column count mismatch: profile={arr.shape[1]} rows={len(roi_rows)}")
    col = _network_column(network_column)
    out = np.full((arr.shape[0], len(network_names)), np.nan, dtype=np.float32)
    for net_idx, net_name in enumerate(network_names):
        cols: list[int] = []
        weights: list[float] = []
        for row_idx, row in enumerate(roi_rows):
            if str(row.get(col, "")) != str(net_name):
                continue
            weight = _row_weight(row)
            if weight <= 0.0:
                continue
            cols.append(row_idx)
            weights.append(weight)
        if not cols:
            continue
        values = arr[:, np.asarray(cols, dtype=np.int32)].astype(np.float64)
        w = np.asarray(weights, dtype=np.float64)
        finite = np.isfinite(values)
        weighted_values = np.where(finite, values * w[None, :], 0.0)
        weight_sums = np.sum(np.where(finite, w[None, :], 0.0), axis=1)
        ok = weight_sums > 0.0
        out[ok, net_idx] = (np.sum(weighted_values[ok, :], axis=1) / weight_sums[ok]).astype(np.float32)
    return out



def weighted_network_fisherz_profile_from_timeseries(
    hippocampal_timeseries: np.ndarray,
    roi_timeseries: np.ndarray,
    roi_rows: Sequence[dict[str, object]],
    network_names: Sequence[str],
    *,
    network_column: str = "canonical7",
) -> np.ndarray:
    """Compute Fisher-z FC against vertex-count weighted network time series.

    This is the strict weighted_fisherz definition: first reconstruct each
    cortical network time series from its ROI mean time series and ROI vertex
    counts, then correlate hippocampal rows against those network time series.
    It is intentionally not equivalent to averaging precomputed ROI FC values.
    """
    hipp = np.asarray(hippocampal_timeseries, dtype=np.float32)
    roi = np.asarray(roi_timeseries, dtype=np.float32)
    if hipp.ndim != 2:
        raise ValueError(f"Expected 2D hippocampal time series, got shape={hipp.shape}")
    _validate_rows(roi, roi_rows)
    if hipp.shape[1] != roi.shape[1]:
        raise ValueError(f"Timepoint mismatch: hippocampal={hipp.shape[1]} roi={roi.shape[1]}")
    network_ts = weighted_network_timeseries_from_roi_timeseries(
        roi,
        roi_rows,
        network_names,
        network_column=network_column,
    )
    profile = fisher_z_transform_fc(corrcoef_rows(hipp, network_ts))
    valid_hipp = np.any(np.isfinite(hipp), axis=1)
    valid_net = np.any(np.isfinite(network_ts), axis=1)
    profile[~valid_hipp, :] = np.nan
    profile[:, ~valid_net] = np.nan
    return profile.astype(np.float32)


def weighted_cluster_network_fisherz_profile_from_timeseries(
    group_labels_final: np.ndarray,
    support_mask: np.ndarray,
    hippocampal_timeseries: np.ndarray,
    roi_timeseries: np.ndarray,
    roi_rows: Sequence[dict[str, object]],
    network_names: Sequence[str],
    *,
    network_column: str = "canonical7",
) -> np.ndarray:
    """Compute cluster-level weighted_fisherz after averaging cluster time series."""
    labels = np.asarray(group_labels_final, dtype=np.int32).reshape(-1)
    support = np.asarray(support_mask, dtype=bool).reshape(-1)
    hipp = np.asarray(hippocampal_timeseries, dtype=np.float32)
    if hipp.ndim != 2:
        raise ValueError(f"Expected 2D hippocampal time series, got shape={hipp.shape}")
    if labels.size != hipp.shape[0] or support.size != hipp.shape[0]:
        raise ValueError(f"Vertex mismatch: labels={labels.size} support={support.size} timeseries={hipp.shape}")
    k_final = int(np.nanmax(labels)) if labels.size else 0
    cluster_ts = np.full((k_final, hipp.shape[1]), np.nan, dtype=np.float32)
    valid = support & np.any(np.isfinite(hipp), axis=1)
    for cid in range(1, k_final + 1):
        mask = (labels == cid) & valid
        if int(np.sum(mask)) > 0:
            cluster_ts[cid - 1, :] = np.nanmean(hipp[mask, :], axis=0).astype(np.float32)
    return weighted_network_fisherz_profile_from_timeseries(
        cluster_ts,
        roi_timeseries,
        roi_rows,
        network_names,
        network_column=network_column,
    )

def assignment_status_from_weighted_network_profile(
    profile: np.ndarray,
    network_names: Sequence[str],
) -> tuple[list[dict[str, object]], list[str | None]]:
    arr = np.asarray(profile, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D network profile, got shape={arr.shape}")
    if arr.shape[1] != len(network_names):
        raise ValueError(f"Network count mismatch: profile={arr.shape[1]} names={len(network_names)}")
    rows: list[dict[str, object]] = []
    dominant: list[str | None] = []
    for row_idx in range(arr.shape[0]):
        row = arr[row_idx, :]
        finite = np.flatnonzero(np.isfinite(row))
        if finite.size == 0:
            rows.append({"cluster_id": int(row_idx + 1), "status": "unassigned", "top1_network": "unassigned", "top2_network": "unassigned", "top1_fisher_z": 0.0, "top2_fisher_z": 0.0, "margin": 0.0})
            dominant.append(None)
            continue
        order = finite[np.argsort(row[finite])]
        top1_i = int(order[-1])
        top2_i = int(order[-2]) if order.size > 1 else int(order[-1])
        top1 = float(row[top1_i])
        top2 = float(row[top2_i])
        rows.append(
            {
                "cluster_id": int(row_idx + 1),
                "status": "dominant",
                "top1_network": str(network_names[top1_i]),
                "top1_fisher_z": top1,
                "top2_network": str(network_names[top2_i]),
                "top2_fisher_z": top2,
                "margin": float(top1 - top2),
                "low_margin_lt_0p05": int((top1 - top2) < 0.05),
            }
        )
        dominant.append(str(network_names[top1_i]))
    return rows, dominant


def assign_weighted_network_profile_rows(
    *,
    profile: np.ndarray,
    support_indices: np.ndarray,
    network_names: Sequence[str],
) -> list[dict[str, object]]:
    arr = np.asarray(profile, dtype=np.float32)
    idx = np.asarray(support_indices, dtype=np.int32).reshape(-1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D vertex profile, got shape={arr.shape}")
    if arr.shape[0] != idx.size:
        raise ValueError(f"Vertex/profile mismatch: profile={arr.shape[0]} support_indices={idx.size}")
    if arr.shape[1] != len(network_names):
        raise ValueError(f"Network count mismatch: profile={arr.shape[1]} names={len(network_names)}")
    rows: list[dict[str, object]] = []
    for local_vidx, global_vidx in enumerate(idx):
        row = arr[local_vidx, :]
        finite = np.flatnonzero(np.isfinite(row))
        if finite.size == 0:
            rows.append({"vertex_index": int(global_vidx), "status": "unassigned", "top1_network": "unassigned", "top2_network": "unassigned", "top1_fisher_z": 0.0, "top2_fisher_z": 0.0, "margin_score": 0.0})
            continue
        order = finite[np.argsort(row[finite])]
        top1_i = int(order[-1])
        top2_i = int(order[-2]) if order.size > 1 else int(order[-1])
        top1 = float(row[top1_i])
        top2 = float(row[top2_i])
        rows.append(
            {
                "vertex_index": int(global_vidx),
                "status": "dominant",
                "top1_network": str(network_names[top1_i]),
                "top2_network": str(network_names[top2_i]),
                "top1_fisher_z": top1,
                "top2_fisher_z": top2,
                "margin_score": float(top1 - top2),
            }
        )
    return rows

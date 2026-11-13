from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from lib.fc_math import center_run_timeseries, corrcoef_rows, fisher_z_transform_fc
from lib.io_utils import load_run_segment_metadata


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
    run_lengths: Sequence[int] | None = None,
) -> np.ndarray:
    """Compute Fisher-z FC against vertex-count weighted network time series.

    This is the network_ts_fisherz definition: first reconstruct each
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
    profile = fisher_z_transform_fc(corrcoef_rows(hipp, network_ts, run_lengths=run_lengths))
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
    run_lengths: Sequence[int] | None = None,
) -> np.ndarray:
    """Average supported, complete nonconstant rows, then compute weighted FC.

    Declared run means are removed before row eligibility and cluster averaging.
    Callers must intersect group support with the subject's valid-vertex mask.
    """
    labels = np.asarray(group_labels_final, dtype=np.int32).reshape(-1)
    support = np.asarray(support_mask, dtype=bool).reshape(-1)
    hipp = np.asarray(hippocampal_timeseries, dtype=np.float32)
    if hipp.ndim != 2:
        raise ValueError(f"Expected 2D hippocampal time series, got shape={hipp.shape}")
    if labels.size != hipp.shape[0] or support.size != hipp.shape[0]:
        raise ValueError(f"Vertex mismatch: labels={labels.size} support={support.size} timeseries={hipp.shape}")
    if run_lengths is not None:
        hipp = center_run_timeseries(hipp, run_lengths)
    k_final = int(np.nanmax(labels)) if labels.size else 0
    cluster_ts = np.full((k_final, hipp.shape[1]), np.nan, dtype=np.float32)
    valid = support & np.isfinite(hipp).all(axis=1)
    if hipp.shape[1] < 2:
        valid[:] = False
    else:
        valid[valid] = np.std(hipp[valid].astype(np.float64), axis=1, ddof=1) > 0
    for cid in range(1, k_final + 1):
        mask = (labels == cid) & valid
        if int(np.sum(mask)) > 0:
            cluster_ts[cid - 1, :] = np.mean(hipp[mask, :], axis=0).astype(np.float32)
    return weighted_network_fisherz_profile_from_timeseries(
        cluster_ts,
        roi_timeseries,
        roi_rows,
        network_names,
        network_column=network_column,
        run_lengths=run_lengths,
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


def subjectwise_cluster_network_fisherz_profile(
    *,
    group_labels_final: np.ndarray,
    support_mask: np.ndarray,
    subjects: Sequence[str],
    surface_root: Path,
    reference_root: Path,
    smoothing: str,
    hemi: str,
    valid_masks_by_subject: dict[str, np.ndarray],
    network_names: Sequence[str],
    reference_atlas: str = "kong2022",
    network_column: str = "canonical7",
    array_loader=np.load,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Compute subjectwise cluster-to-network Fisher-z against weighted network time series."""
    labels = np.asarray(group_labels_final, dtype=np.int32).reshape(-1)
    support = np.asarray(support_mask, dtype=bool).reshape(-1)
    k_final = int(np.nanmax(labels)) if labels.size else 0
    if k_final <= 0:
        raise ValueError("Kong2022 cluster network profile requires at least one positive cluster label")

    profiles: list[np.ndarray] = []
    used_subjects: list[str] = []
    for subject in subjects:
        ts_path = surface_root / f"sub-{subject}" / "surface" / smoothing / f"sub-{subject}_hemi-{hemi}_timeseries.npy"
        ref_dir = reference_root / f"sub-{subject}" / "reference" / reference_atlas
        roi_path = ref_dir / "cortex_roi_parcel_timeseries.npy"
        roi_tsv = ref_dir / "cortex_roi_parcels.tsv"
        summary_json = ref_dir / "reference_summary.json"

        missing = [str(p) for p in (ts_path, roi_path, roi_tsv) if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing Kong2022 cluster annotation input for sub-{subject}: "
                + ", ".join(missing)
            )
        if subject not in valid_masks_by_subject:
            raise ValueError(f"Missing valid mask for sub-{subject}")

        valid_mask = np.asarray(valid_masks_by_subject[subject], dtype=bool).reshape(-1)
        if valid_mask.shape != support.shape:
            raise ValueError(
                f"Subject valid mask vertex mismatch for sub-{subject}: mask={valid_mask.shape} support={support.shape}"
            )
        subj_support = support & valid_mask

        ts = array_loader(ts_path).astype(np.float32, copy=False)
        roi = array_loader(roi_path).astype(np.float32, copy=False)

        with roi_tsv.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))

        # Declared physical runs are required: without them run means are not removed.
        _run_ids, run_lengths = load_run_segment_metadata(summary_json)

        prof = weighted_cluster_network_fisherz_profile_from_timeseries(
            group_labels_final=labels,
            support_mask=subj_support,
            hippocampal_timeseries=ts,
            roi_timeseries=roi,
            roi_rows=rows,
            network_names=network_names,
            network_column=network_column,
            run_lengths=run_lengths,
        )
        profiles.append(prof)
        used_subjects.append(str(subject))

    stack = np.stack(profiles, axis=0).astype(np.float32)
    mean = np.nanmean(stack, axis=0).astype(np.float32)
    return stack, mean, used_subjects

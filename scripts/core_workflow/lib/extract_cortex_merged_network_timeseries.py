#!/usr/bin/env python3
from __future__ import annotations

import nibabel as nib
import numpy as np


def extract_structure_data(
    dt_axis: nib.cifti2.cifti2_axes.BrainModelAxis,
    dt_data_t: np.ndarray,
    structure_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    for name, slc, subaxis in dt_axis.iter_structures():
        if name == structure_name or name.endswith(structure_name):
            if getattr(subaxis, "vertex", None) is None:
                raise RuntimeError(f"Structure {structure_name} does not expose vertices")
            return np.asarray(subaxis.vertex, dtype=np.int32), dt_data_t[slc, :]
    raise RuntimeError(f"Could not find structure {structure_name} in dtseries")


def load_required_mask(path: str, expected_length: int, label: str) -> np.ndarray:
    mask = np.load(path).astype(bool)
    if mask.ndim != 1 or mask.shape[0] != expected_length:
        raise ValueError(f"{label} mask shape mismatch: expected ({expected_length},), got {mask.shape}")
    return mask


def aggregate_hemi_roi_timeseries(
    *,
    hemi: str,
    hemi_dt: np.ndarray,
    aligned_parcel_ids: np.ndarray,
    valid_mask: np.ndarray,
    roi_registry: list[dict[str, object]],
) -> tuple[list[np.ndarray], list[dict[str, object]], list[dict[str, object]], dict[str, int]]:
    """Keep ordered ROI identities; empty parcels have all-NaN time series."""
    n_total = int(hemi_dt.shape[0])
    n_used = int(np.sum(valid_mask))
    stats = {
        "n_grayordinates_total": n_total,
        "n_grayordinates_used": n_used,
        "n_grayordinates_masked": int(n_total - n_used),
    }

    # Build pid→total-count and pid→valid-vertex-indices once (O(n_vertices)),
    # replacing 400 per-ROI equality scans in the original loop.
    unique_all, counts_all = np.unique(aligned_parcel_ids, return_counts=True)
    pid_total: dict[int, int] = dict(zip(unique_all.tolist(), counts_all.tolist()))

    valid_idx = np.where(valid_mask)[0]
    valid_pids = aligned_parcel_ids[valid_idx]
    sort_order = np.argsort(valid_pids, kind="stable")
    sorted_idx = valid_idx[sort_order]
    sorted_pids = valid_pids[sort_order]
    unique_valid, group_sizes = np.unique(sorted_pids, return_counts=True)
    split_points = np.cumsum(group_sizes[:-1])
    pid_valid_arr: dict[int, np.ndarray] = {
        int(pid): grp
        for pid, grp in zip(unique_valid.tolist(), np.split(sorted_idx, split_points))
    }

    ts_rows: list[np.ndarray] = []
    kept_rows: list[dict[str, object]] = []
    empty_rows: list[dict[str, object]] = []

    for row in roi_registry:
        if str(row["hemi"]) != hemi:
            continue
        roi_id = int(row["roi_id"])
        n_g_total = pid_total.get(roi_id, 0)
        valid_idxs = pid_valid_arr.get(roi_id, np.empty(0, dtype=np.int32))
        n_g_used = int(len(valid_idxs))
        out_row = {
            "parcel_id": roi_id,
            "parcel_name": str(row["roi_name"]),
            "family17": str(row["family17"]),
            "canonical8": str(row["canonical8"]),
            "canonical7": str(row["canonical7"]),
            "hemisphere": hemi,
            "subregion": str(row["subregion"]),
            "rank": int(row["rank"]),
            "n_grayordinates_total": n_g_total,
            "n_grayordinates_used": n_g_used,
            "excluded_by_validity": bool(n_g_used <= 0),
        }
        if n_g_used <= 0:
            empty_rows.append(out_row)
            ts = np.full((hemi_dt.shape[1],), np.nan, dtype=np.float32)
        else:
            ts = np.nanmean(hemi_dt[valid_idxs, :], axis=0).astype(np.float32)
        ts_rows.append(ts)
        kept_rows.append(out_row)

    return ts_rows, kept_rows, empty_rows, stats

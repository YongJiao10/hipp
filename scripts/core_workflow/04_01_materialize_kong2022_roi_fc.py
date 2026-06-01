#!/usr/bin/env python3
from __future__ import annotations

"""
Process Step Description:
Materialize Kong2022 ROI-level functional connectivity (FC). Computes the functional connectivity matrix between hippocampal surface vertices and Kong2022 cortical network regions of interest (ROIs).

Main Flow Path:
- 6th step in the numbered pipeline sequence (ROI functional connectivity materialization for kong2022):
  01_01_run_hippunfold_t1w_local.py -> 01_02_check_hippunfold_outputs.py -> 02_01_prepare_group_inputs.py -> 02_02_build_qc_manifest.py -> 03_01/03_02 reference building -> 04_01_materialize_kong2022_roi_fc.py (this step) -> 05_01 gradient computation -> 05_02 group atlas parcellation -> 06_xx presentation rendering -> 07_01 group matrix overview.

Inputs:
- Command-line arguments:
  - --shared-root: Path to shared reference outputs (default: `outputs_core/inputs/hcp7t_hippocampus/kong2022/shared_inputs/routeb_151`).
  - --surface-root: Path to shared BOLD surfaces (default: `outputs_core/inputs/hcp7t_hippocampus/kong2022/shared_inputs/surface_151`).
  - --subjects-manifest: Path to QC-passed subject list manifest (default: `outputs_core/inputs/hcp7t_hippocampus/kong2022/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt`).
  - --subjects: Space-separated list of explicit subject IDs.
  - --smoothings: Smoothing levels to process: "2mm", "4mm".
  - --hemis: Hemispheres to process: "L", "R".
  - --jobs: Subject-level process concurrency cap (default: 1, hard-capped at 10).
- Physical inputs:
  - Resampled BOLD timeseries: `{surface_root}/sub-<subject>/surface/{smoothing}/sub-<subject>_hemi-{hemi}_timeseries.npy`.
  - BOLD active mask: `{surface_root}/sub-<subject>/surface/tsnr_gate/sub-<subject>_hemi-{hemi}_valid_mask.npy`.
  - Reference timeseries: `{shared_root}/sub-<subject>/reference/kong2022/cortex_roi_parcel_timeseries.npy`.
  - Reference parcels TSV: `{shared_root}/sub-<subject>/reference/kong2022/cortex_roi_parcels.tsv`.

Outputs:
- Command-line output / file outputs:
  - Complete vertex-to-ROI functional connectivity matrix: `{shared_root}/sub-<subject>/fc/kong2022/{smoothing}/hemi_{hemi}/hipp_vertex_to_network_fc.npy`.
  - Active-only vertex-to-ROI functional connectivity matrix: `{shared_root}/sub-<subject>/fc/kong2022/{smoothing}/hemi_{hemi}/hipp_vertex_to_network_fc_active.npy`.
  - Run manifest: `{shared_root}/sub-<subject>/fc/kong2022/{smoothing}/hemi_{hemi}/fc_summary.json`.
"""
import argparse
import csv
import json
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
from scipy.spatial import cKDTree

from lib.output_layout import (
    DEFAULT_CORTEX_ROUTEA_ROOT,
    DEFAULT_HIPPUNFOLD_BATCH_ROOT,
    DEFAULT_KONG2022_MANIFEST_ROOT,
    DEFAULT_KONG2022_ROOT,
)
from lib.workflow_utils import capped_jobs, utc_now_iso, write_json
from lib.io_utils import load_subject_ids
from lib.fc_math import corrcoef_rows, compact_active_vertices
from lib.extract_cortex_merged_network_timeseries import extract_structure_data
from lib.extract_cortex_kong2022_route_a_roi_timeseries import (
    align_surface_labels_to_dt_vertices,
    discover_route_a_outputs_dir,
    load_surface_label_vector,
)

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROUTE_ROOT = DEFAULT_KONG2022_ROOT
DEFAULT_SHARED_INPUTS_ROOT = DEFAULT_ROUTE_ROOT / "shared_inputs"
DEFAULT_SHARED_ROOT = DEFAULT_SHARED_INPUTS_ROOT / "routeb_151"
DEFAULT_SURFACE_ROOT = DEFAULT_SHARED_INPUTS_ROOT / "surface_151"
DEFAULT_SUBJECTS = DEFAULT_KONG2022_MANIFEST_ROOT / "hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt"
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "hippunfold_input"
DEFAULT_ROUTE_A_ROOT = DEFAULT_CORTEX_ROUTEA_ROOT
DEFAULT_LEFT_CORTEX_SURFACE_TEMPLATE = (
    "{data_root}/{sub_subject}/anat/{sub_subject}_hemi-L_space-fsLR_den-32k_desc-MSMAll_midthickness.surf.gii"
)
DEFAULT_RIGHT_CORTEX_SURFACE_TEMPLATE = (
    "{data_root}/{sub_subject}/anat/{sub_subject}_hemi-R_space-fsLR_den-32k_desc-MSMAll_midthickness.surf.gii"
)
ATLAS = "kong2022"
SMOOTHINGS = ["2mm", "4mm"]
HEMIS = ["L", "R"]
RUN_IDS = [1, 2, 3, 4]


@dataclass(frozen=True)
class CtxDropFCResult:
    fc: np.ndarray
    roi_cortex_counts: np.ndarray
    excluded_neighbor_counts: np.ndarray


def finite_corr_1d(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    finite = np.isfinite(aa) & np.isfinite(bb)
    if int(np.sum(finite)) < 2:
        return float("nan")
    x = aa[finite] - float(np.mean(aa[finite]))
    y = bb[finite] - float(np.mean(bb[finite]))
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 0.0 or not np.isfinite(denom):
        return float("nan")
    return float(np.clip(float(np.dot(x, y) / denom), -1.0, 1.0))


def corr_one_to_rows(y: np.ndarray, x_rows: np.ndarray) -> np.ndarray:
    yy = np.asarray(y, dtype=np.float64).reshape(1, -1)
    xx = np.asarray(x_rows, dtype=np.float64)
    finite = np.isfinite(xx) & np.isfinite(yy)
    n = np.sum(finite, axis=1).astype(np.float64)
    out = np.full((xx.shape[0],), np.nan, dtype=np.float32)
    ok = n >= 2
    if not np.any(ok):
        return out
    safe_n = np.where(ok, n, 1.0)
    x_mean = np.sum(np.where(finite, xx, 0.0), axis=1) / safe_n
    y_mean = np.sum(np.where(finite, yy, 0.0), axis=1) / safe_n
    xm = np.where(finite, xx - x_mean[:, None], 0.0)
    ym = np.where(finite, yy - y_mean[:, None], 0.0)
    denom = np.linalg.norm(xm, axis=1) * np.linalg.norm(ym, axis=1)
    valid = ok & np.isfinite(denom) & (denom > 0.0)
    out[valid] = np.clip(np.sum(xm[valid] * ym[valid], axis=1) / denom[valid], -1.0, 1.0).astype(np.float32)
    return out


def compute_ctxdrop_vertex_to_roi_fc(
    *,
    hipp_ts: np.ndarray,
    cortex_ts: np.ndarray,
    cortex_roi_ids: np.ndarray,
    roi_ids: list[int],
    cortex_valid_mask: np.ndarray,
    neighbor_indices: list[np.ndarray],
) -> CtxDropFCResult:
    hipp = np.asarray(hipp_ts, dtype=np.float32)
    cortex = np.asarray(cortex_ts, dtype=np.float32)
    parcel_ids = np.asarray(cortex_roi_ids, dtype=np.int32).reshape(-1)
    valid = np.asarray(cortex_valid_mask, dtype=bool).reshape(-1)
    if hipp.ndim != 2 or cortex.ndim != 2:
        raise ValueError(f"Expected 2D time series arrays, got hipp={hipp.shape} cortex={cortex.shape}")
    if hipp.shape[1] != cortex.shape[1]:
        raise ValueError(f"Timepoint mismatch: hipp={hipp.shape} cortex={cortex.shape}")
    if parcel_ids.size != int(cortex.shape[0]):
        raise ValueError(f"Cortex ROI id length mismatch: ids={parcel_ids.size} cortex_rows={cortex.shape[0]}")
    if valid.size != int(cortex.shape[0]):
        raise ValueError(f"Cortex valid mask length mismatch: mask={valid.size} cortex_rows={cortex.shape[0]}")
    if len(neighbor_indices) != int(hipp.shape[0]):
        raise ValueError(f"Expected one neighbor list per hippocampal vertex: {len(neighbor_indices)} vs {hipp.shape[0]}")

    roi_values = [int(x) for x in roi_ids]
    roi_col_by_id = {roi_id: idx for idx, roi_id in enumerate(roi_values)}
    roi_col_for_cortex = np.full((cortex.shape[0],), -1, dtype=np.int32)
    for roi_id, col in roi_col_by_id.items():
        roi_col_for_cortex[parcel_ids == int(roi_id)] = int(col)

    n_roi = len(roi_values)
    n_tp = int(cortex.shape[1])
    base_sums = np.zeros((n_roi, n_tp), dtype=np.float64)
    base_counts = np.zeros((n_roi,), dtype=np.int32)
    valid_roi_cols = roi_col_for_cortex[valid]
    valid_idx = np.flatnonzero(valid)
    keep = valid_roi_cols >= 0
    if np.any(keep):
        np.add.at(base_counts, valid_roi_cols[keep], 1)
        np.add.at(base_sums, valid_roi_cols[keep], cortex[valid_idx[keep], :].astype(np.float64))

    fc = np.full((hipp.shape[0], n_roi), np.nan, dtype=np.float32)
    roi_counts = np.zeros((hipp.shape[0], n_roi), dtype=np.int32)
    excluded_counts = np.zeros((hipp.shape[0],), dtype=np.int32)

    for vertex_idx, raw_neighbors in enumerate(neighbor_indices):
        neigh = np.asarray(raw_neighbors, dtype=np.int64).reshape(-1)
        if neigh.size:
            if int(np.min(neigh)) < 0 or int(np.max(neigh)) >= int(cortex.shape[0]):
                raise ValueError(f"Cortex neighbor index out of bounds for hippocampal vertex {vertex_idx}")
        neigh = neigh[valid[neigh]] if neigh.size else neigh
        neigh_cols = roi_col_for_cortex[neigh] if neigh.size else np.asarray([], dtype=np.int32)
        keep_neigh = neigh_cols >= 0
        neigh = neigh[keep_neigh]
        neigh_cols = neigh_cols[keep_neigh]
        excluded_counts[vertex_idx] = int(neigh.size)

        drop_sums = np.zeros((n_roi, n_tp), dtype=np.float64)
        drop_counts = np.zeros((n_roi,), dtype=np.int32)
        if neigh.size:
            np.add.at(drop_counts, neigh_cols, 1)
            np.add.at(drop_sums, neigh_cols, cortex[neigh, :].astype(np.float64))

        counts = (base_counts - drop_counts).astype(np.int32)
        roi_counts[vertex_idx, :] = counts
        has_data = counts > 0
        if not np.any(has_data):
            continue
        roi_ts = np.full((n_roi, n_tp), np.nan, dtype=np.float32)
        roi_ts[has_data, :] = ((base_sums[has_data, :] - drop_sums[has_data, :]) / counts[has_data, None]).astype(np.float32)
        fc[vertex_idx, :] = corr_one_to_rows(hipp[vertex_idx, :], roi_ts)

    return CtxDropFCResult(
        fc=fc.astype(np.float32),
        roi_cortex_counts=roi_counts.astype(np.int32),
        excluded_neighbor_counts=excluded_counts.astype(np.int32),
    )


def roi_ids_and_names_from_rows(roi_rows: list[dict[str, object]]) -> tuple[list[int], list[str]]:
    roi_ids: list[int] = []
    roi_names: list[str] = []
    for idx, row in enumerate(roi_rows):
        raw_id = row.get("parcel_id", idx + 1)
        roi_ids.append(int(raw_id))
        roi_names.append(str(row.get("parcel_name", row.get("roi_name", raw_id))))
    if not roi_ids:
        raise RuntimeError("No ROI rows available for ctxdrop FC materialization")
    return roi_ids, roi_names


def materialize_ctxdrop_fc_from_arrays(
    *,
    fc_dir: Path,
    subject: str,
    smoothing: str,
    hemi: str,
    hipp_ts: np.ndarray,
    valid_mask: np.ndarray,
    cortex_ts: np.ndarray,
    cortex_roi_ids: np.ndarray,
    roi_rows: list[dict[str, object]],
    neighbor_indices: list[np.ndarray],
    run_segments: list[dict[str, int]],
    distance_mm: float,
    local_neighbor_dir: Path,
    cortex_valid_mask: np.ndarray | None = None,
) -> None:
    ts = np.asarray(hipp_ts, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if ts.ndim != 2:
        raise ValueError(f"Expected 2D hippocampal time series, got {ts.shape}")
    if valid.size != int(ts.shape[0]):
        raise ValueError(f"Hippocampal valid mask length mismatch: mask={valid.size} rows={ts.shape[0]}")
    roi_ids, roi_names = roi_ids_and_names_from_rows(roi_rows)
    cortex_valid = (
        np.ones((np.asarray(cortex_ts).shape[0],), dtype=bool)
        if cortex_valid_mask is None
        else np.asarray(cortex_valid_mask, dtype=bool).reshape(-1)
    )
    result = compute_ctxdrop_vertex_to_roi_fc(
        hipp_ts=ts,
        cortex_ts=cortex_ts,
        cortex_roi_ids=cortex_roi_ids,
        roi_ids=roi_ids,
        cortex_valid_mask=cortex_valid,
        neighbor_indices=neighbor_indices,
    )
    pearson_full = result.fc.astype(np.float32, copy=True)
    pearson_full[~valid, :] = np.nan
    pearson_valid = pearson_full[valid, :]

    fc_dir.mkdir(parents=True, exist_ok=True)
    local_neighbor_dir.mkdir(parents=True, exist_ok=True)
    np.save(fc_dir / "hipp_vertex_to_network_fc.npy", pearson_full.astype(np.float32))
    np.save(fc_dir / "hipp_vertex_to_network_fc_active.npy", compact_active_vertices(pearson_full, valid))
    np.save(local_neighbor_dir / "neighbor_counts.npy", result.excluded_neighbor_counts.astype(np.int32))
    np.save(local_neighbor_dir / "ctxdrop_roi_cortex_counts.npy", result.roi_cortex_counts.astype(np.int32))

    run_labels: list[str] = []
    run_lengths: list[int] = []
    for seg in run_segments:
        rid = int(seg["run_id"])
        run_start = int(seg["run_start"])
        run_stop = int(seg["run_stop"])
        run_len = int(seg["run_length"])
        if run_stop - run_start != run_len:
            raise RuntimeError(f"Run segment mismatch for sub-{subject} run-{rid}: {seg}")
        run_result = compute_ctxdrop_vertex_to_roi_fc(
            hipp_ts=ts[:, run_start:run_stop],
            cortex_ts=np.asarray(cortex_ts, dtype=np.float32)[:, run_start:run_stop],
            cortex_roi_ids=cortex_roi_ids,
            roi_ids=roi_ids,
            cortex_valid_mask=cortex_valid,
            neighbor_indices=neighbor_indices,
        )
        run_full = run_result.fc.astype(np.float32, copy=True)
        run_full[~valid, :] = np.nan
        run_dir = fc_dir / "runs" / f"run-{rid}"
        run_dir.mkdir(parents=True, exist_ok=True)
        np.save(run_dir / "hipp_vertex_to_network_fc_active.npy", run_full[valid, :].astype(np.float32))
        run_labels.append(f"run-{rid}")
        run_lengths.append(run_len)

    write_json(
        fc_dir / "fc_summary.json",
        {
            "subject": subject,
            "atlas_slug": ATLAS,
            "smooth": smoothing,
            "hemi": hemi,
            "route_tag": "routea_ctxdrop",
            "ctxdrop_distance_mm": float(distance_mm),
            "ctxdrop_policy": "per_hippocampal_vertex_exclude_valid_cortex_vertices_within_radius_before_roi_mean",
            "fc_shape": [int(pearson_full.shape[0]), int(pearson_full.shape[1])],
            "active_fc_shape": [int(pearson_valid.shape[0]), int(pearson_valid.shape[1])],
            "n_vertices_total": int(ts.shape[0]),
            "n_vertices_valid_high_tsnr": int(valid.sum()),
            "n_vertices_invalid_initial": int(ts.shape[0] - valid.sum()),
            "n_vertices_clustered": int(valid.sum()),
            "n_timepoints": int(ts.shape[1]),
            "n_roi_targets_used": int(len(roi_names)),
            "run_labels": run_labels,
            "run_lengths": run_lengths,
            "fc_definition": "pearson_vertex_to_kong2022_roi_timeseries_ctxdrop_local20mm",
            "ctxdrop_neighbor_count_min": int(np.min(result.excluded_neighbor_counts)) if result.excluded_neighbor_counts.size else 0,
            "ctxdrop_neighbor_count_median": float(np.median(result.excluded_neighbor_counts)) if result.excluded_neighbor_counts.size else 0.0,
            "ctxdrop_neighbor_count_max": int(np.max(result.excluded_neighbor_counts)) if result.excluded_neighbor_counts.size else 0,
            "ctxdrop_roi_cortex_count_min": int(np.min(result.roi_cortex_counts)) if result.roi_cortex_counts.size else 0,
            "ctxdrop_roi_cortex_count_median": float(np.median(result.roi_cortex_counts)) if result.roi_cortex_counts.size else 0.0,
            "ctxdrop_roi_cortex_count_max": int(np.max(result.roi_cortex_counts)) if result.roi_cortex_counts.size else 0,
            "local_cortex_neighbor_diagnostics": str(local_neighbor_dir.resolve()),
        },
    )
    write_json(
        fc_dir / "stage_manifest.json",
        {
            "stage": "shared_fc_ctxdrop",
            "status": "done",
            "timestamp_utc": utc_now_iso(),
            "params": {
                "subject": subject,
                "atlas_slug": ATLAS,
                "smooth": smoothing,
                "hemi": hemi,
                "route_tag": "routea_ctxdrop",
                "distance_mm": float(distance_mm),
                "fc_definition": "pearson_vertex_to_kong2022_roi_timeseries_ctxdrop_local20mm",
            },
            "outputs": [
                str((fc_dir / "hipp_vertex_to_network_fc.npy").resolve()),
                str((fc_dir / "hipp_vertex_to_network_fc_active.npy").resolve()),
                str((local_neighbor_dir / "neighbor_counts.npy").resolve()),
                str((local_neighbor_dir / "ctxdrop_roi_cortex_counts.npy").resolve()),
                str((fc_dir / "fc_summary.json").resolve()),
            ],
        },
    )


def read_roi_names(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise RuntimeError(f"Empty ROI TSV: {path}")
    return [str(row["parcel_name"]) for row in rows]


def load_run_segments(subj_root: Path) -> list[dict[str, int]]:
    segs: list[dict[str, int]] = []
    for rid in RUN_IDS:
        p = subj_root / "reference" / "common" / "runs" / f"run-{rid}" / "reference_summary.json"
        obj = json.loads(p.read_text(encoding="utf-8"))
        segs.append(
            {
                "run_id": int(rid),
                "run_start": int(obj["run_start"]),
                "run_stop": int(obj["run_stop"]),
                "run_length": int(obj["run_length"]),
            }
        )
    return segs



def read_roi_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise RuntimeError(f"Empty ROI TSV: {path}")
    return rows


def format_subject_template(template: str, *, data_root: Path, subject: str, hemi: str) -> Path:
    subject = str(subject).replace("sub-", "")
    return Path(
        str(template).format(
            data_root=str(Path(data_root).resolve()),
            subject=subject,
            sub_subject=f"sub-{subject}",
            hemi=hemi,
            hemi_lower=hemi.lower(),
        )
    ).resolve()


def load_surface_coords(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    coords = None
    for darray in img.darrays:
        if int(darray.intent) == 1008:
            coords = np.asarray(darray.data, dtype=np.float32)
            break
    if coords is None:
        data = img.agg_data()
        coords = np.asarray(data[0] if isinstance(data, tuple) else data, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Expected surface POINTSET coordinates with shape (N, 3): {path} shape={coords.shape}")
    return coords.astype(np.float32, copy=False)


def hipp_surface_path(batch_root: Path, subject: str, hemi: str) -> Path:
    subject = str(subject).replace("sub-", "")
    return (
        Path(batch_root)
        / f"sub-{subject}"
        / "hippunfold"
        / f"sub-{subject}"
        / "surf"
        / f"sub-{subject}_hemi-{hemi}_space-corobl_den-512_label-hipp_midthickness.surf.gii"
    )


def dtseries_path(data_root: Path, subject: str) -> Path:
    subject = str(subject).replace("sub-", "")
    return Path(data_root) / f"sub-{subject}" / "func" / f"sub-{subject}_task-rest_run-concat.dtseries.nii"


def load_optional_cortex_valid_mask(reference_root: Path, subject: str, cortex_counts: dict[str, int]) -> np.ndarray:
    subject = str(subject).replace("sub-", "")
    gate = Path(reference_root) / f"sub-{subject}" / "reference" / "common" / "tsnr_gate"
    pieces: list[np.ndarray] = []
    for hemi_name, key in [("left", "CORTEX_LEFT"), ("right", "CORTEX_RIGHT")]:
        path = gate / f"cortex_{hemi_name}_valid_mask.npy"
        if path.exists():
            arr = np.load(path).astype(bool)
            if int(arr.size) != int(cortex_counts[key]):
                raise ValueError(f"Cortex valid mask size mismatch: {path} size={arr.size} expected={cortex_counts[key]}")
            pieces.append(arr)
        else:
            pieces.append(np.ones((int(cortex_counts[key]),), dtype=bool))
    return np.concatenate(pieces).astype(bool)


def load_ctxdrop_cortex_inputs(
    *,
    data_root: Path,
    subject: str,
    route_a_root: Path,
    route_a_label_dir: Path | None,
    left_cortex_surface_template: str,
    right_cortex_surface_template: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int], dict[str, Path]]:
    subject = str(subject).replace("sub-", "")
    dt_path = dtseries_path(data_root, subject)
    dt_img = nib.load(str(dt_path))
    dt_data = np.asarray(dt_img.dataobj, dtype=np.float32)
    if dt_data.ndim != 2:
        raise ValueError(f"Expected 2D dtseries data: {dt_path} shape={dt_data.shape}")
    dt_data_t = dt_data.T
    axis = dt_img.header.get_axis(1)
    left_vertices, left_dt = extract_structure_data(axis, dt_data_t, "CORTEX_LEFT")
    right_vertices, right_dt = extract_structure_data(axis, dt_data_t, "CORTEX_RIGHT")

    route_out_dir = discover_route_a_outputs_dir(route_a_root=route_a_root, subject=subject, explicit=route_a_label_dir)
    lh_full = load_surface_label_vector(route_out_dir / "lh_labels.txt")
    rh_full = load_surface_label_vector(route_out_dir / "rh_labels.txt")
    left_roi_ids = align_surface_labels_to_dt_vertices(lh_full, left_vertices, hemi="L")
    right_roi_ids = align_surface_labels_to_dt_vertices(rh_full, right_vertices, hemi="R")

    cortex_surface_paths = {
        "L": format_subject_template(left_cortex_surface_template, data_root=data_root, subject=subject, hemi="L"),
        "R": format_subject_template(right_cortex_surface_template, data_root=data_root, subject=subject, hemi="R"),
    }
    left_coords_full = load_surface_coords(cortex_surface_paths["L"])
    right_coords_full = load_surface_coords(cortex_surface_paths["R"])
    if int(np.max(left_vertices)) >= int(left_coords_full.shape[0]):
        raise ValueError(f"Left dtseries vertex index exceeds surface rows: {dt_path} vs {cortex_surface_paths['L']}")
    if int(np.max(right_vertices)) >= int(right_coords_full.shape[0]):
        raise ValueError(f"Right dtseries vertex index exceeds surface rows: {dt_path} vs {cortex_surface_paths['R']}")

    coords = np.vstack([left_coords_full[left_vertices], right_coords_full[right_vertices]]).astype(np.float32)
    ts = np.vstack([left_dt, right_dt]).astype(np.float32)
    roi_ids = np.concatenate([left_roi_ids, right_roi_ids]).astype(np.int32)
    counts = {"CORTEX_LEFT": int(left_dt.shape[0]), "CORTEX_RIGHT": int(right_dt.shape[0])}
    paths = {"dtseries": dt_path.resolve(), "left_surface": cortex_surface_paths["L"], "right_surface": cortex_surface_paths["R"], "route_a_labels": route_out_dir.resolve()}
    return coords, ts, roi_ids, counts, paths


def query_local_cortex_neighbors(
    *,
    hipp_coords: np.ndarray,
    cortex_coords: np.ndarray,
    cortex_valid_mask: np.ndarray,
    distance_mm: float,
) -> list[np.ndarray]:
    valid = np.asarray(cortex_valid_mask, dtype=bool).reshape(-1)
    if valid.size != int(np.asarray(cortex_coords).shape[0]):
        raise ValueError(f"Cortex valid mask size mismatch: mask={valid.size} coords={np.asarray(cortex_coords).shape[0]}")
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size == 0:
        return [np.asarray([], dtype=np.int64) for _ in range(int(np.asarray(hipp_coords).shape[0]))]
    tree = cKDTree(np.asarray(cortex_coords, dtype=np.float32)[valid_idx, :])
    local = tree.query_ball_point(np.asarray(hipp_coords, dtype=np.float32), r=float(distance_mm))
    return [valid_idx[np.asarray(row, dtype=np.int64)] for row in local]


def materialize_ctxdrop_fc(
    *,
    shared_root: Path,
    surface_root: Path,
    reference_root: Path,
    mask_root: Path,
    subject: str,
    smoothing: str,
    hemi: str,
    cortex_ts: np.ndarray,
    cortex_roi_ids: np.ndarray,
    cortex_valid_mask: np.ndarray,
    neighbor_indices: list[np.ndarray],
    distance_mm: float,
    local_neighbor_dir: Path,
) -> None:
    subject = str(subject).replace("sub-", "")
    subj_root = Path(shared_root) / f"sub-{subject}"
    ref_subj_root = Path(reference_root) / f"sub-{subject}"
    surf_subj_root = Path(surface_root) / f"sub-{subject}"
    mask_subj_root = Path(mask_root) / f"sub-{subject}"
    fc_dir = subj_root / "fc" / ATLAS / smoothing / f"hemi_{hemi}"

    ts = np.load(surf_subj_root / "surface" / smoothing / f"sub-{subject}_hemi-{hemi}_timeseries.npy").astype(np.float32)
    valid_mask = np.load(mask_subj_root / "surface" / "tsnr_gate" / f"sub-{subject}_hemi-{hemi}_valid_mask.npy").astype(bool)
    roi_rows = read_roi_rows(ref_subj_root / "reference" / ATLAS / "cortex_roi_parcels.tsv")
    run_segments = load_run_segments(ref_subj_root)
    materialize_ctxdrop_fc_from_arrays(
        fc_dir=fc_dir,
        subject=subject,
        smoothing=smoothing,
        hemi=hemi,
        hipp_ts=ts,
        valid_mask=valid_mask,
        cortex_ts=cortex_ts,
        cortex_roi_ids=cortex_roi_ids,
        roi_rows=roi_rows,
        neighbor_indices=neighbor_indices,
        run_segments=run_segments,
        distance_mm=float(distance_mm),
        local_neighbor_dir=local_neighbor_dir,
        cortex_valid_mask=cortex_valid_mask,
    )


def materialize_fc(
    shared_root: Path,
    surface_root: Path,
    subject: str,
    smoothing: str,
    hemi: str,
    reference_root: Path | None = None,
    mask_root: Path | None = None,
) -> None:
    subj_root = shared_root / f"sub-{subject}"
    ref_subj_root = (reference_root or shared_root) / f"sub-{subject}"
    surf_subj_root = surface_root / f"sub-{subject}"
    mask_subj_root = (mask_root or surface_root) / f"sub-{subject}"
    fc_dir = subj_root / "fc" / ATLAS / smoothing / f"hemi_{hemi}"
    fc_dir.mkdir(parents=True, exist_ok=True)

    ts = np.load(surf_subj_root / "surface" / smoothing / f"sub-{subject}_hemi-{hemi}_timeseries.npy").astype(np.float32)
    valid_mask = np.load(mask_subj_root / "surface" / "tsnr_gate" / f"sub-{subject}_hemi-{hemi}_valid_mask.npy").astype(bool)
    roi_ts = np.load(ref_subj_root / "reference" / ATLAS / "cortex_roi_parcel_timeseries.npy").astype(np.float32)
    roi_names = read_roi_names(ref_subj_root / "reference" / ATLAS / "cortex_roi_parcels.tsv")

    if ts.ndim != 2 or roi_ts.ndim != 2:
        raise ValueError(
            f"Expected 2D inputs for sub-{subject} {smoothing} hemi-{hemi}: ts={ts.shape} roi={roi_ts.shape}"
        )
    if ts.shape[1] != roi_ts.shape[1]:
        raise RuntimeError(
            f"Time mismatch for sub-{subject} {smoothing} hemi-{hemi}: ts={ts.shape} roi={roi_ts.shape}"
        )
    if roi_ts.shape[0] != len(roi_names):
        raise RuntimeError(
            f"ROI TSV mismatch for sub-{subject}: ts_rows={roi_ts.shape[0]} roi_names={len(roi_names)}"
        )

    pearson_valid = corrcoef_rows(ts[valid_mask, :], roi_ts)
    pearson_full = np.full((ts.shape[0], roi_ts.shape[0]), np.nan, dtype=np.float32)
    pearson_full[valid_mask, :] = pearson_valid
    np.save(fc_dir / "hipp_vertex_to_network_fc.npy", pearson_full.astype(np.float32))
    np.save(fc_dir / "hipp_vertex_to_network_fc_active.npy", compact_active_vertices(pearson_full, valid_mask))

    run_labels: list[str] = []
    run_lengths: list[int] = []
    run_segments = load_run_segments(ref_subj_root)
    for seg in run_segments:
        rid = int(seg["run_id"])
        run_start = int(seg["run_start"])
        run_stop = int(seg["run_stop"])
        run_len = int(seg["run_length"])
        run_roi = roi_ts[:, run_start:run_stop].astype(np.float32, copy=False)
        if run_roi.shape[1] != run_len:
            raise RuntimeError(
                f"Run ROI slice mismatch for sub-{subject} run-{rid}: run_roi={run_roi.shape} expected_len={run_len}"
            )
        run_labels.append(f"run-{rid}")
        run_lengths.append(run_len)
        run_ts = ts[:, run_start:run_stop].astype(np.float32, copy=False)
        if run_ts.shape[1] != run_len:
            raise RuntimeError(
                f"Run slice mismatch for sub-{subject} {smoothing} hemi-{hemi} run-{rid}: "
                f"slice={run_ts.shape} expected_len={run_len}"
            )
        if run_roi.shape[0] != roi_ts.shape[0]:
            raise RuntimeError(
                f"Run ROI row mismatch for sub-{subject} run-{rid}: run_roi={run_roi.shape} full_roi={roi_ts.shape}"
            )
        run_dir = fc_dir / "runs" / f"run-{rid}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_pearson_valid = corrcoef_rows(run_ts[valid_mask, :], run_roi)
        np.save(run_dir / "hipp_vertex_to_network_fc_active.npy", run_pearson_valid.astype(np.float32))
    total_run_len = int(sum(run_lengths))
    if total_run_len != int(ts.shape[1]):
        raise RuntimeError(
            f"Run lengths do not sum to full time series for sub-{subject} {smoothing} hemi-{hemi}: "
            f"{total_run_len} vs {ts.shape[1]}"
        )

    write_json(
        fc_dir / "fc_summary.json",
        {
            "subject": subject,
            "atlas_slug": ATLAS,
            "smooth": smoothing,
            "hemi": hemi,
            "fc_shape": [int(pearson_full.shape[0]), int(pearson_full.shape[1])],
            "active_fc_shape": [int(pearson_valid.shape[0]), int(pearson_valid.shape[1])],
            "n_vertices_total": int(ts.shape[0]),
            "n_vertices_valid_high_tsnr": int(valid_mask.sum()),
            "n_vertices_invalid_initial": int(ts.shape[0] - valid_mask.sum()),
            "n_vertices_clustered": int(valid_mask.sum()),
            "n_timepoints": int(ts.shape[1]),
            "n_roi_targets_used": int(roi_ts.shape[0]),
            "run_labels": run_labels,
            "run_lengths": run_lengths,
            "fc_definition": "pearson_vertex_to_kong2022_roi_timeseries",
        },
    )
    write_json(
        fc_dir / "stage_manifest.json",
        {
            "stage": "shared_fc",
            "status": "done",
            "timestamp_utc": utc_now_iso(),
            "params": {
                "subject": subject,
                "atlas_slug": ATLAS,
                "smooth": smoothing,
                "hemi": hemi,
                "fc_definition": "pearson_vertex_to_kong2022_roi_timeseries",
            },
            "inputs": [
                str((ref_subj_root / "reference" / ATLAS / "cortex_roi_parcel_timeseries.npy").resolve()),
                str((surf_subj_root / "surface" / smoothing / f"sub-{subject}_hemi-{hemi}_timeseries.npy").resolve()),
                str((mask_subj_root / "surface" / "tsnr_gate" / f"sub-{subject}_hemi-{hemi}_valid_mask.npy").resolve()),
            ],
            "outputs": [
                str((fc_dir / "hipp_vertex_to_network_fc.npy").resolve()),
                str((fc_dir / "hipp_vertex_to_network_fc_active.npy").resolve()),
                str((fc_dir / "fc_summary.json").resolve()),
            ],
        },
    )


def process_subject(
    subject: str,
    shared_root_str: str,
    surface_root_str: str,
    reference_root_str: str,
    mask_root_str: str,
    smoothings: list[str],
    hemis: list[str],
    ctxdrop: bool = False,
    data_root_str: str | None = None,
    hippunfold_batch_root_str: str | None = None,
    route_a_root_str: str | None = None,
    route_a_label_dir_str: str | None = None,
    left_cortex_surface_template: str = DEFAULT_LEFT_CORTEX_SURFACE_TEMPLATE,
    right_cortex_surface_template: str = DEFAULT_RIGHT_CORTEX_SURFACE_TEMPLATE,
    ctxdrop_distance_mm: float = 20.0,
) -> dict[str, object]:
    subject = str(subject).replace("sub-", "")
    shared_root = Path(shared_root_str)
    surface_root = Path(surface_root_str)
    reference_root = Path(reference_root_str)
    mask_root = Path(mask_root_str)
    n_fc = 0

    if ctxdrop:
        data_root = Path(data_root_str or DEFAULT_DATA_ROOT).resolve()
        hippunfold_batch_root = Path(hippunfold_batch_root_str or DEFAULT_HIPPUNFOLD_BATCH_ROOT).resolve()
        route_a_root = Path(route_a_root_str or DEFAULT_ROUTE_A_ROOT).resolve()
        route_a_label_dir = Path(route_a_label_dir_str).resolve() if route_a_label_dir_str else None
        cortex_coords, cortex_ts, cortex_roi_ids, cortex_counts, _cortex_paths = load_ctxdrop_cortex_inputs(
            data_root=data_root,
            subject=subject,
            route_a_root=route_a_root,
            route_a_label_dir=route_a_label_dir,
            left_cortex_surface_template=left_cortex_surface_template,
            right_cortex_surface_template=right_cortex_surface_template,
        )
        cortex_valid = load_optional_cortex_valid_mask(reference_root, subject, cortex_counts)
        neighbors_by_hemi: dict[str, list[np.ndarray]] = {}
        for hemi in hemis:
            hipp_coords = load_surface_coords(hipp_surface_path(hippunfold_batch_root, subject, hemi))
            neighbors_by_hemi[hemi] = query_local_cortex_neighbors(
                hipp_coords=hipp_coords,
                cortex_coords=cortex_coords,
                cortex_valid_mask=cortex_valid,
                distance_mm=float(ctxdrop_distance_mm),
            )
        for smoothing in smoothings:
            for hemi in hemis:
                materialize_ctxdrop_fc(
                    shared_root=shared_root,
                    surface_root=surface_root,
                    reference_root=reference_root,
                    mask_root=mask_root,
                    subject=subject,
                    smoothing=smoothing,
                    hemi=hemi,
                    cortex_ts=cortex_ts,
                    cortex_roi_ids=cortex_roi_ids,
                    cortex_valid_mask=cortex_valid,
                    neighbor_indices=neighbors_by_hemi[hemi],
                    distance_mm=float(ctxdrop_distance_mm),
                    local_neighbor_dir=shared_root / f"sub-{subject}" / "surface" / "local_cortex_neighbors" / f"hemi_{hemi}",
                )
                n_fc += 1
        return {"subject": f"sub-{subject}", "fc_combos": int(n_fc), "ctxdrop": True}

    for smoothing in smoothings:
        for hemi in hemis:
            materialize_fc(shared_root, surface_root, subject, smoothing, hemi, reference_root=reference_root, mask_root=mask_root)
            n_fc += 1
    return {"subject": f"sub-{subject}", "fc_combos": int(n_fc), "ctxdrop": False}

def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize Kong2022 hippocampal vertex-to-ROI FC caches.")
    parser.add_argument("--shared-root", default=str(DEFAULT_SHARED_ROOT))
    parser.add_argument("--surface-root", default=str(DEFAULT_SURFACE_ROOT))
    parser.add_argument("--reference-root", default=None, help="Read cortex reference/run metadata from this root; defaults to --shared-root.")
    parser.add_argument("--mask-root", default=None, help="Read hippocampal valid masks from this root; defaults to --surface-root.")
    parser.add_argument("--subjects-manifest", default=str(DEFAULT_SUBJECTS))
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--smoothings", nargs="+", default=SMOOTHINGS, choices=SMOOTHINGS)
    parser.add_argument("--hemis", nargs="+", default=HEMIS, choices=HEMIS)
    parser.add_argument("--jobs", type=int, default=1, help="Subject-level parallel workers; hard-capped at 10.")
    parser.add_argument("--ctxdrop", action="store_true", help="Materialize RouteA ctxdrop FC by excluding local cortex vertices before ROI averaging.")
    parser.add_argument("--ctxdrop-distance-mm", type=float, default=20.0)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--hippunfold-batch-root", default=str(DEFAULT_HIPPUNFOLD_BATCH_ROOT))
    parser.add_argument("--route-a-root", default=str(DEFAULT_ROUTE_A_ROOT))
    parser.add_argument("--route-a-label-dir", default=None)
    parser.add_argument("--left-cortex-surface-template", default=DEFAULT_LEFT_CORTEX_SURFACE_TEMPLATE)
    parser.add_argument("--right-cortex-surface-template", default=DEFAULT_RIGHT_CORTEX_SURFACE_TEMPLATE)
    args = parser.parse_args()

    shared_root = Path(args.shared_root).resolve()
    surface_root = Path(args.surface_root).resolve()
    reference_root = Path(args.reference_root).resolve() if args.reference_root else shared_root
    mask_root = Path(args.mask_root).resolve() if args.mask_root else surface_root
    subjects = [s.replace("sub-", "") for s in args.subjects] if args.subjects else load_subject_ids(
        Path(args.subjects_manifest).resolve()
    )
    jobs = capped_jobs(args.jobs, len(subjects))
    n_fc = 0
    if jobs == 1:
        for subject in subjects:
            obj = process_subject(
                subject,
                str(shared_root),
                str(surface_root),
                str(reference_root),
                str(mask_root),
                list(args.smoothings),
                list(args.hemis),
                ctxdrop=bool(args.ctxdrop),
                data_root_str=str(Path(args.data_root).resolve()),
                hippunfold_batch_root_str=str(Path(args.hippunfold_batch_root).resolve()),
                route_a_root_str=str(Path(args.route_a_root).resolve()),
                route_a_label_dir_str=(str(Path(args.route_a_label_dir).resolve()) if args.route_a_label_dir else None),
                left_cortex_surface_template=str(args.left_cortex_surface_template),
                right_cortex_surface_template=str(args.right_cortex_surface_template),
                ctxdrop_distance_mm=float(args.ctxdrop_distance_mm),
            )
            n_fc += int(obj["fc_combos"])
            print(f"[done] {obj['subject']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = [
                ex.submit(
                    process_subject,
                    s,
                    str(shared_root),
                    str(surface_root),
                    str(reference_root),
                    str(mask_root),
                    list(args.smoothings),
                    list(args.hemis),
                    bool(args.ctxdrop),
                    str(Path(args.data_root).resolve()),
                    str(Path(args.hippunfold_batch_root).resolve()),
                    str(Path(args.route_a_root).resolve()),
                    (str(Path(args.route_a_label_dir).resolve()) if args.route_a_label_dir else None),
                    str(args.left_cortex_surface_template),
                    str(args.right_cortex_surface_template),
                    float(args.ctxdrop_distance_mm),
                )
                for s in subjects
            ]
            for fut in as_completed(futs):
                obj = fut.result()
                n_fc += int(obj["fc_combos"])
                print(f"[done] {obj['subject']}", flush=True)
    print(json.dumps({"subjects": len(subjects), "fc_combos": int(n_fc), "jobs": int(jobs), "status": "ok", "ctxdrop": bool(args.ctxdrop)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

"""
Materialize RouteA hippocampal surface time series after local cortex regression.

This branch preserves RouteA cortex reference definitions in their upstream
root, but replaces the hippocampal time series consumed by downstream FC
materialization with residuals after regressing the mean signal from nearby
valid cortical vertices. Regression is run independently inside each run
segment, and this step writes only controlled time series and diagnostics.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree

from lib.extract_cortex_merged_network_timeseries import extract_structure_data
from lib.io_utils import load_subject_ids
from lib.output_layout import (
    DEFAULT_HIPPUNFOLD_BATCH_ROOT,
    DEFAULT_KONG2022_MANIFEST_ROOT,
    DEFAULT_KONG2022_ROUTEA_CTXREG_ROOT,
    DEFAULT_KONG2022_ROUTEA_ROOT,
    DEFAULT_KONG2022_SURFACE_ROOT,
)
from lib.path_templates import format_subject_path_template
from lib.workflow_utils import capped_jobs, utc_now_iso, write_json


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "hippunfold_input"
DEFAULT_DTSERIES_TEMPLATE = "{data_root}/{sub_subject}/func/{sub_subject}_task-rest_run-concat.dtseries.nii"
DEFAULT_SOURCE_SHARED_ROOT = DEFAULT_KONG2022_ROUTEA_ROOT
DEFAULT_OUT_SHARED_ROOT = DEFAULT_KONG2022_ROUTEA_CTXREG_ROOT
DEFAULT_SURFACE_ROOT = DEFAULT_KONG2022_SURFACE_ROOT
DEFAULT_SUBJECTS = DEFAULT_KONG2022_MANIFEST_ROOT / "hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt"
DEFAULT_LEFT_CORTEX_SURFACE_TEMPLATE = (
    "{data_root}/{sub_subject}/anat/{sub_subject}_hemi-L_space-fsLR_den-32k_desc-MSMAll_midthickness.surf.gii"
)
DEFAULT_RIGHT_CORTEX_SURFACE_TEMPLATE = (
    "{data_root}/{sub_subject}/anat/{sub_subject}_hemi-R_space-fsLR_den-32k_desc-MSMAll_midthickness.surf.gii"
)
SMOOTHINGS = ("2mm", "4mm")
HEMIS = ("L", "R")
RUN_IDS = (1, 2, 3, 4)
ROUTE_TAG = "routea_ctxreg"
SOURCE_ROUTE_TAG = "routea"


@dataclass(frozen=True)
class LocalRegressionResult:
    controlled_ts: np.ndarray
    neighbor_counts: np.ndarray
    skipped_no_neighbor: np.ndarray
    variance_before: np.ndarray
    variance_after: np.ndarray
    local_cortex_corr: np.ndarray


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
        if int(darray.intent) == 1008:  # NIFTI_INTENT_POINTSET
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


def load_cortex_from_dtseries(
    *,
    dtseries: Path,
    left_surface: Path,
    right_surface: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    img = nib.load(str(dtseries))
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D dtseries data: {dtseries} shape={data.shape}")
    data_t = data.T
    axis = img.header.get_axis(1)

    left_vertices, left_ts = extract_structure_data(axis, data_t, "CORTEX_LEFT")
    right_vertices, right_ts = extract_structure_data(axis, data_t, "CORTEX_RIGHT")
    left_coords_full = load_surface_coords(left_surface)
    right_coords_full = load_surface_coords(right_surface)
    if int(np.max(left_vertices)) >= int(left_coords_full.shape[0]):
        raise ValueError(f"Left dtseries vertex index exceeds surface rows: {dtseries} vs {left_surface}")
    if int(np.max(right_vertices)) >= int(right_coords_full.shape[0]):
        raise ValueError(f"Right dtseries vertex index exceeds surface rows: {dtseries} vs {right_surface}")

    coords = np.vstack([left_coords_full[left_vertices], right_coords_full[right_vertices]]).astype(np.float32)
    ts = np.vstack([left_ts, right_ts]).astype(np.float32)
    counts = {"CORTEX_LEFT": int(left_ts.shape[0]), "CORTEX_RIGHT": int(right_ts.shape[0])}
    return coords, ts, counts


def load_optional_cortex_valid_mask(source_shared_root: Path, subject: str, cortex_counts: dict[str, int]) -> np.ndarray:
    subject = str(subject).replace("sub-", "")
    gate = Path(source_shared_root) / f"sub-{subject}" / "reference" / "common" / "tsnr_gate"
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


def load_run_segments(shared_root: Path, subject: str) -> list[dict[str, int]]:
    subject = str(subject).replace("sub-", "")
    bases = [
        Path(shared_root) / f"sub-{subject}" / "reference" / "common" / "runs",
        Path(shared_root) / f"sub-{subject}" / "reference" / "kong2022" / "runs",
    ]
    for base in bases:
        segments: list[dict[str, int]] = []
        ok = True
        for rid in RUN_IDS:
            p = base / f"run-{rid}" / "reference_summary.json"
            if not p.exists():
                ok = False
                break
            obj = json.loads(p.read_text(encoding="utf-8"))
            segments.append(
                {
                    "run_id": int(obj.get("run_id", rid)),
                    "run_start": int(obj["run_start"]),
                    "run_stop": int(obj["run_stop"]),
                    "run_length": int(obj["run_length"]),
                }
            )
        if ok:
            return segments
    raise FileNotFoundError(f"Missing run segment reference summaries under {Path(shared_root) / f'sub-{subject}' / 'reference'}")


def validate_run_segments(run_segments: list[dict[str, int]], n_timepoints: int) -> list[dict[str, int]]:
    out: list[dict[str, int]] = []
    for seg in run_segments:
        start = int(seg["run_start"])
        stop = int(seg["run_stop"])
        length = int(seg["run_length"])
        if start < 0 or stop <= start or length != stop - start:
            raise ValueError(f"Invalid run segment: {seg}")
        if stop > int(n_timepoints):
            raise ValueError(f"Run segment exceeds time series length={n_timepoints}: {seg}")
        out.append({"run_id": int(seg.get("run_id", len(out) + 1)), "run_start": start, "run_stop": stop, "run_length": length})
    if sum(int(s["run_length"]) for s in out) != int(n_timepoints):
        raise ValueError(f"Run lengths do not sum to n_timepoints={n_timepoints}: {out}")
    return out


def validate_coordinate_space_compatibility(
    *,
    hipp_coords: np.ndarray,
    cortex_coords: np.ndarray,
    distance_mm: float,
    max_median_nearest_mm: float | None = None,
) -> dict[str, float]:
    hipp = np.asarray(hipp_coords, dtype=np.float64)
    cortex = np.asarray(cortex_coords, dtype=np.float64)
    if hipp.ndim != 2 or cortex.ndim != 2 or hipp.shape[1] != 3 or cortex.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) coordinate arrays, got hipp={hipp.shape} cortex={cortex.shape}")
    if hipp.shape[0] == 0 or cortex.shape[0] == 0:
        raise ValueError("coordinate-space validation requires non-empty hippocampal and cortical coordinates")
    if not (np.all(np.isfinite(hipp)) and np.all(np.isfinite(cortex))):
        raise ValueError("coordinate-space validation failed: non-finite coordinates")
    nearest, _idx = cKDTree(cortex).query(hipp, k=1)
    median_nearest = float(np.median(nearest))
    min_nearest = float(np.min(nearest))
    max_nearest = float(np.max(nearest))
    threshold = float(max_median_nearest_mm) if max_median_nearest_mm is not None else max(100.0, float(distance_mm) * 5.0)
    if median_nearest > threshold:
        raise ValueError(
            "coordinate-space validation failed: hippocampal and cortical surfaces look implausibly mismatched; "
            f"median_nearest_mm={median_nearest:.3f} threshold_mm={threshold:.3f}"
        )
    return {
        "nearest_min_mm": min_nearest,
        "nearest_median_mm": median_nearest,
        "nearest_max_mm": max_nearest,
        "median_threshold_mm": threshold,
    }


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


def _ols_residual(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    xx = np.asarray(x, dtype=np.float64).reshape(-1)
    finite = np.isfinite(yy) & np.isfinite(xx)
    if int(np.sum(finite)) < 2:
        return yy.astype(np.float32)
    design = np.column_stack([xx[finite], np.ones((int(np.sum(finite)),), dtype=np.float64)])
    beta, *_ = np.linalg.lstsq(design, yy[finite], rcond=None)
    out = yy.copy()
    out[finite] = yy[finite] - design @ beta
    return out.astype(np.float32)


def _finite_corr(a: np.ndarray, b: np.ndarray) -> float:
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


def regress_local_cortex_signal(
    *,
    hipp_ts: np.ndarray,
    cortex_ts: np.ndarray,
    neighbor_indices: list[np.ndarray],
    run_segments: list[dict[str, int]],
) -> LocalRegressionResult:
    hipp = np.asarray(hipp_ts, dtype=np.float32)
    cortex = np.asarray(cortex_ts, dtype=np.float32)
    if hipp.ndim != 2 or cortex.ndim != 2:
        raise ValueError(f"Expected 2D time series arrays, got hipp={hipp.shape} cortex={cortex.shape}")
    if hipp.shape[1] != cortex.shape[1]:
        raise ValueError(f"Timepoint mismatch: hipp={hipp.shape} cortex={cortex.shape}")
    if len(neighbor_indices) != int(hipp.shape[0]):
        raise ValueError(f"Expected one neighbor list per hippocampal vertex: {len(neighbor_indices)} vs {hipp.shape[0]}")
    segments = validate_run_segments(run_segments, int(hipp.shape[1]))

    controlled = hipp.copy()
    neighbor_counts = np.zeros((hipp.shape[0],), dtype=np.int32)
    skipped = np.zeros((hipp.shape[0],), dtype=bool)
    variance_before = np.nanvar(hipp.astype(np.float64), axis=1).astype(np.float32)
    local_corr = np.full((hipp.shape[0],), np.nan, dtype=np.float32)

    for vertex_idx, raw_neighbors in enumerate(neighbor_indices):
        neigh = np.asarray(raw_neighbors, dtype=np.int64).reshape(-1)
        neighbor_counts[vertex_idx] = int(neigh.size)
        if neigh.size == 0:
            skipped[vertex_idx] = True
            continue
        local_mean = np.nanmean(cortex[neigh, :], axis=0).astype(np.float32)
        local_corr[vertex_idx] = _finite_corr(hipp[vertex_idx, :], local_mean)
        for seg in segments:
            start = int(seg["run_start"])
            stop = int(seg["run_stop"])
            controlled[vertex_idx, start:stop] = _ols_residual(
                hipp[vertex_idx, start:stop],
                local_mean[start:stop],
            )

    variance_after = np.nanvar(controlled.astype(np.float64), axis=1).astype(np.float32)
    return LocalRegressionResult(
        controlled_ts=controlled.astype(np.float32),
        neighbor_counts=neighbor_counts,
        skipped_no_neighbor=skipped,
        variance_before=variance_before,
        variance_after=variance_after,
        local_cortex_corr=local_corr,
    )



def write_regression_outputs(
    *,
    out_shared_root: Path,
    source_surface_root: Path,
    source_shared_root: Path,
    subject: str,
    smoothing: str,
    hemi: str,
    result: LocalRegressionResult,
    source_ts_path: Path,
    source_mask_path: Path,
    distance_mm: float,
    coord_meta: dict[str, float],
    run_segments: list[dict[str, int]],
    cortex_dtseries: Path,
    cortex_surface_paths: dict[str, Path],
) -> None:
    subject = str(subject).replace("sub-", "")
    out_surface_dir = Path(out_shared_root) / f"sub-{subject}" / "surface" / smoothing
    out_surface_dir.mkdir(parents=True, exist_ok=True)
    out_ts = out_surface_dir / f"sub-{subject}_hemi-{hemi}_timeseries.npy"
    np.save(out_ts, result.controlled_ts.astype(np.float32))

    diag_dir = Path(out_shared_root) / f"sub-{subject}" / "surface" / "cortex_regression" / smoothing / f"hemi_{hemi}"
    local_neighbor_dir = Path(out_shared_root) / f"sub-{subject}" / "surface" / "local_cortex_neighbors" / f"hemi_{hemi}"
    diag_dir.mkdir(parents=True, exist_ok=True)
    local_neighbor_dir.mkdir(parents=True, exist_ok=True)
    np.save(local_neighbor_dir / "neighbor_counts.npy", result.neighbor_counts.astype(np.int32))
    np.save(local_neighbor_dir / "skipped_no_neighbor.npy", result.skipped_no_neighbor.astype(bool))
    np.save(diag_dir / "variance_before.npy", result.variance_before.astype(np.float32))
    np.save(diag_dir / "variance_after.npy", result.variance_after.astype(np.float32))
    np.save(diag_dir / "local_cortex_corr.npy", result.local_cortex_corr.astype(np.float32))

    summary = {
        "status": "ok",
        "route_tag": ROUTE_TAG,
        "source_route_tag": SOURCE_ROUTE_TAG,
        "subject": f"sub-{subject}",
        "smoothing": smoothing,
        "hemi": hemi,
        "distance_mm": float(distance_mm),
        "regression_model": "per_hippocampal_vertex_y ~ mean_valid_cortex_within_radius + intercept",
        "runwise_regression": True,
        "run_segments": run_segments,
        "n_vertices": int(result.controlled_ts.shape[0]),
        "n_timepoints": int(result.controlled_ts.shape[1]),
        "n_vertices_with_cortex_neighbors": int(np.sum(~result.skipped_no_neighbor)),
        "n_vertices_without_cortex_neighbors": int(np.sum(result.skipped_no_neighbor)),
        "neighbor_count_min": int(np.min(result.neighbor_counts)) if result.neighbor_counts.size else 0,
        "neighbor_count_median": float(np.median(result.neighbor_counts)) if result.neighbor_counts.size else 0.0,
        "neighbor_count_max": int(np.max(result.neighbor_counts)) if result.neighbor_counts.size else 0,
        "mean_variance_before": float(np.nanmean(result.variance_before)),
        "mean_variance_after": float(np.nanmean(result.variance_after)),
        "mean_local_cortex_corr": float(np.nanmean(result.local_cortex_corr)),
        "coordinate_space_gate": coord_meta,
        "inputs": {
            "source_surface_root": str(Path(source_surface_root).resolve()),
            "source_shared_root": str(Path(source_shared_root).resolve()),
            "hippocampal_timeseries": str(source_ts_path.resolve()),
            "hippocampal_valid_mask": str(source_mask_path.resolve()),
            "cortex_dtseries": str(cortex_dtseries.resolve()),
            "left_cortex_surface": str(cortex_surface_paths["L"].resolve()),
            "right_cortex_surface": str(cortex_surface_paths["R"].resolve()),
        },
        "outputs": {
            "controlled_timeseries": str(out_ts.resolve()),
            "neighbor_counts": str((local_neighbor_dir / "neighbor_counts.npy").resolve()),
            "skipped_no_neighbor": str((local_neighbor_dir / "skipped_no_neighbor.npy").resolve()),
            "variance_before": str((diag_dir / "variance_before.npy").resolve()),
            "variance_after": str((diag_dir / "variance_after.npy").resolve()),
            "local_cortex_corr": str((diag_dir / "local_cortex_corr.npy").resolve()),
        },
    }
    write_json(diag_dir / "cortex_regression_summary.json", summary, trailing_newline=True)
    write_json(
        diag_dir / "stage_manifest.json",
        {
            "stage": "routea_ctxreg_local_cortex_regression",
            "status": "done",
            "timestamp_utc": utc_now_iso(),
            "params": {
                "subject": subject,
                "smoothing": smoothing,
                "hemi": hemi,
                "distance_mm": float(distance_mm),
            },
            "inputs": [
                str(source_ts_path.resolve()),
                str(source_mask_path.resolve()),
                str(cortex_dtseries.resolve()),
                str(cortex_surface_paths["L"].resolve()),
                str(cortex_surface_paths["R"].resolve()),
            ],
            "outputs": [
                str(out_ts.resolve()),
                str((diag_dir / "cortex_regression_summary.json").resolve()),
                str((local_neighbor_dir / "neighbor_counts.npy").resolve()),
            ],
        },
        trailing_newline=True,
    )


def process_subject(
    *,
    subject: str,
    source_shared_root: str,
    source_surface_root: str,
    out_shared_root: str,
    data_root: str,
    hippunfold_batch_root: str,
    left_cortex_surface_template: str,
    right_cortex_surface_template: str,
    dtseries_template: str,
    smoothings: list[str],
    hemis: list[str],
    distance_mm: float,
) -> dict[str, object]:
    subject = str(subject).replace("sub-", "")
    source_shared = Path(source_shared_root)
    source_surface = Path(source_surface_root)
    out_shared = Path(out_shared_root)
    data = Path(data_root)
    (out_shared / f"sub-{subject}" / "surface").mkdir(parents=True, exist_ok=True)

    dt_path = format_subject_path_template(
        dtseries_template,
        data_root=data,
        subject=subject,
    )
    cortex_surface_paths = {
        "L": format_subject_template(left_cortex_surface_template, data_root=data, subject=subject, hemi="L"),
        "R": format_subject_template(right_cortex_surface_template, data_root=data, subject=subject, hemi="R"),
    }
    cortex_coords, cortex_ts, cortex_counts = load_cortex_from_dtseries(
        dtseries=dt_path,
        left_surface=cortex_surface_paths["L"],
        right_surface=cortex_surface_paths["R"],
    )
    cortex_valid = load_optional_cortex_valid_mask(source_shared, subject, cortex_counts)
    run_segments = load_run_segments(source_shared, subject)
    n_outputs = 0

    for hemi in hemis:
        hipp_coords = load_surface_coords(hipp_surface_path(Path(hippunfold_batch_root), subject, hemi))
        coord_meta = validate_coordinate_space_compatibility(
            hipp_coords=hipp_coords,
            cortex_coords=cortex_coords,
            distance_mm=float(distance_mm),
        )
        neighbors = query_local_cortex_neighbors(
            hipp_coords=hipp_coords,
            cortex_coords=cortex_coords,
            cortex_valid_mask=cortex_valid,
            distance_mm=float(distance_mm),
        )
        for smoothing in smoothings:
            ts_path = source_surface / f"sub-{subject}" / "surface" / smoothing / f"sub-{subject}_hemi-{hemi}_timeseries.npy"
            mask_path = source_surface / f"sub-{subject}" / "surface" / "tsnr_gate" / f"sub-{subject}_hemi-{hemi}_valid_mask.npy"
            hipp_ts = np.load(ts_path).astype(np.float32)
            valid_mask = np.load(mask_path).astype(bool)
            if int(hipp_ts.shape[0]) != int(hipp_coords.shape[0]):
                raise ValueError(f"Hippocampal surface/time-series row mismatch: {ts_path} rows={hipp_ts.shape[0]} coords={hipp_coords.shape[0]}")
            if int(valid_mask.size) != int(hipp_ts.shape[0]):
                raise ValueError(f"Hippocampal valid mask size mismatch: {mask_path} size={valid_mask.size} rows={hipp_ts.shape[0]}")
            result = regress_local_cortex_signal(
                hipp_ts=hipp_ts,
                cortex_ts=cortex_ts,
                neighbor_indices=neighbors,
                run_segments=run_segments,
            )
            write_regression_outputs(
                out_shared_root=out_shared,
                source_surface_root=source_surface,
                source_shared_root=source_shared,
                subject=subject,
                smoothing=smoothing,
                hemi=hemi,
                result=result,
                source_ts_path=ts_path,
                source_mask_path=mask_path,
                distance_mm=float(distance_mm),
                coord_meta=coord_meta,
                run_segments=validate_run_segments(run_segments, int(hipp_ts.shape[1])),
                cortex_dtseries=dt_path,
                cortex_surface_paths=cortex_surface_paths,
            )
            n_outputs += 1

    return {"subject": f"sub-{subject}", "controlled_timeseries": int(n_outputs)}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize RouteA local-cortex-regressed hippocampal time series.")
    parser.add_argument("--source-shared-root", default=str(DEFAULT_SOURCE_SHARED_ROOT))
    parser.add_argument("--source-surface-root", default=str(DEFAULT_SURFACE_ROOT))
    parser.add_argument("--out-shared-root", default=str(DEFAULT_OUT_SHARED_ROOT))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--hippunfold-batch-root", default=str(DEFAULT_HIPPUNFOLD_BATCH_ROOT))
    parser.add_argument("--subjects-manifest", default=str(DEFAULT_SUBJECTS))
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--smoothings", nargs="+", choices=SMOOTHINGS, default=list(SMOOTHINGS))
    parser.add_argument("--hemis", nargs="+", choices=HEMIS, default=list(HEMIS))
    parser.add_argument("--distance-mm", type=float, default=20.0)
    parser.add_argument("--left-cortex-surface-template", default=DEFAULT_LEFT_CORTEX_SURFACE_TEMPLATE)
    parser.add_argument("--right-cortex-surface-template", default=DEFAULT_RIGHT_CORTEX_SURFACE_TEMPLATE)
    parser.add_argument(
        "--dtseries-template",
        default=DEFAULT_DTSERIES_TEMPLATE,
        help=(
            "Per-subject dtseries template. Variables: {data_root}, {subject}, "
            "{sub_subject}. Default preserves the legacy HCP layout."
        ),
    )
    parser.add_argument("--jobs", type=int, default=1, help="Subject-level workers; hard-capped at 10.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    subjects = [s.replace("sub-", "") for s in args.subjects] if args.subjects else load_subject_ids(Path(args.subjects_manifest).resolve())
    jobs = capped_jobs(int(args.jobs), len(subjects))
    kwargs: dict[str, Any] = {
        "source_shared_root": str(Path(args.source_shared_root).resolve()),
        "source_surface_root": str(Path(args.source_surface_root).resolve()),
        "out_shared_root": str(Path(args.out_shared_root).resolve()),
        "data_root": str(Path(args.data_root).resolve()),
        "hippunfold_batch_root": str(Path(args.hippunfold_batch_root).resolve()),
        "left_cortex_surface_template": str(args.left_cortex_surface_template),
        "right_cortex_surface_template": str(args.right_cortex_surface_template),
        "dtseries_template": str(args.dtseries_template),
        "smoothings": [str(x) for x in args.smoothings],
        "hemis": [str(x) for x in args.hemis],
        "distance_mm": float(args.distance_mm),
    }
    results: list[dict[str, object]] = []
    if jobs == 1:
        for subject in subjects:
            obj = process_subject(subject=subject, **kwargs)
            results.append(obj)
            print(f"[done] {obj['subject']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = [ex.submit(process_subject, subject=subject, **kwargs) for subject in subjects]
            for fut in as_completed(futs):
                obj = fut.result()
                results.append(obj)
                print(f"[done] {obj['subject']}", flush=True)
    print(json.dumps({"status": "ok", "subjects": len(subjects), "jobs": int(jobs), "results": results}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

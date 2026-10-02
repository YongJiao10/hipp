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
import hashlib
import json
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree

from lib.extract_cortex_merged_network_timeseries import extract_structure_data
from lib.io_utils import load_run_segment_metadata, load_subject_ids
from lib.output_layout import (
    DEFAULT_HIPPUNFOLD_BATCH_ROOT,
    DEFAULT_KONG2022_MANIFEST_ROOT,
    DEFAULT_KONG400_CTXREG_SHARED_ROOT,
    DEFAULT_KONG2022_ROUTEA_ROOT,
    DEFAULT_KONG2022_SURFACE_ROOT,
)
from lib.path_templates import format_subject_path_template
from lib.workflow_utils import capped_jobs, utc_now_iso, write_json


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "hippunfold_input"
DEFAULT_DTSERIES_TEMPLATE = "{data_root}/{sub_subject}/func/{sub_subject}_task-rest_run-concat.dtseries.nii"
DEFAULT_SOURCE_SHARED_ROOT = DEFAULT_KONG2022_ROUTEA_ROOT
DEFAULT_OUT_SHARED_ROOT = DEFAULT_KONG400_CTXREG_SHARED_ROOT
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
HIPPOCAMPAL_VERTEX_COUNT = 512
ROUTE_TAG = "routea_ctxreg"
SOURCE_ROUTE_TAG = "routea"


@dataclass(frozen=True)
class LocalRegressionResult:
    controlled_ts: np.ndarray
    neighbor_counts: np.ndarray
    skipped_no_neighbor: np.ndarray
    hipp_valid_mask: np.ndarray
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


def format_subject_run_template(
    template: str,
    *,
    data_root: Path,
    subject: str,
    run_id: int,
) -> Path:
    """Format one explicitly selected acquisition-run input path."""
    subject = str(subject).replace("sub-", "")
    return Path(
        str(template).format(
            data_root=str(Path(data_root).resolve()),
            subject=subject,
            sub_subject=f"sub-{subject}",
            run=int(run_id),
            run_id=int(run_id),
            run_label=f"run-{int(run_id)}",
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


def load_cortex_from_dtseries_runs(
    dtseries_paths: list[Path],
    *,
    left_surface: Path,
    right_surface: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Concatenate an explicit ordered list of CIFTI runs in memory."""
    if not dtseries_paths:
        raise ValueError("Explicit dtseries run list is empty")
    loaded = [
        load_cortex_from_dtseries(
            dtseries=Path(path),
            left_surface=left_surface,
            right_surface=right_surface,
        )
        for path in dtseries_paths
    ]
    first_coords, first_ts, first_counts = loaded[0]
    pieces = [first_ts]
    for path, (coords, ts, counts) in zip(dtseries_paths[1:], loaded[1:], strict=True):
        if (
            counts != first_counts
            or coords.shape != first_coords.shape
            or not np.array_equal(coords, first_coords)
        ):
            raise ValueError(
                f"Cortical vertex correspondence differs across explicit runs: {path}"
            )
        pieces.append(ts)
    return (
        first_coords,
        np.concatenate(pieces, axis=1).astype(np.float32, copy=False),
        first_counts,
    )


def load_optional_cortex_valid_mask(
    source_shared_root: Path,
    subject: str,
    cortex_counts: dict[str, int],
    *,
    require_explicit: bool = True,
) -> np.ndarray:
    """Load explicit cortical validity masks; no implicit mask is default."""
    subject = str(subject).replace("sub-", "")
    gate = Path(source_shared_root) / f"sub-{subject}" / "reference" / "common" / "tsnr_gate"
    pieces: list[np.ndarray] = []
    missing: list[Path] = []
    for hemi_name, key in [("left", "CORTEX_LEFT"), ("right", "CORTEX_RIGHT")]:
        path = gate / f"cortex_{hemi_name}_valid_mask.npy"
        if path.is_file():
            arr = np.load(path).astype(bool)
            if int(arr.size) != int(cortex_counts[key]):
                raise ValueError(f"Cortex valid mask size mismatch: {path} size={arr.size} expected={cortex_counts[key]}")
            pieces.append(arr)
        else:
            missing.append(path)
            if not require_explicit:
                pieces.append(np.ones((int(cortex_counts[key]),), dtype=bool))
    if missing and require_explicit:
        raise FileNotFoundError(
            "Missing explicit cortical validity mask(s) required for cortex regression: "
            + ", ".join(str(path) for path in missing)
        )
    return np.concatenate(pieces).astype(bool)


def load_run_segments(
    shared_root: Path,
    subject: str,
    *,
    run_metadata: Path | None = None,
) -> list[dict[str, int]]:
    if run_metadata is not None:
        run_ids, run_lengths = load_run_segment_metadata(Path(run_metadata).resolve())
        segments: list[dict[str, int]] = []
        offset = 0
        for run_id, run_length in zip(run_ids, run_lengths, strict=True):
            stop = offset + int(run_length)
            segments.append(
                {
                    "run_id": int(run_id),
                    "run_start": int(offset),
                    "run_stop": int(stop),
                    "run_length": int(run_length),
                }
            )
            offset = stop
        return validate_run_segment_order(segments, source=Path(run_metadata).resolve())

    subject = str(subject).replace("sub-", "")
    run_root = Path(shared_root) / f"sub-{subject}" / "reference" / "common" / "runs"
    if not run_root.is_dir():
        raise FileNotFoundError(f"Missing canonical run-summary root: {run_root}")
    run_dirs: list[tuple[int, Path]] = []
    for child in run_root.iterdir():
        if not child.is_dir() or not child.name.startswith("run-"):
            continue
        raw_id = child.name[4:]
        if not raw_id.isdigit() or int(raw_id) <= 0:
            raise ValueError(f"Malformed run directory name under {run_root}: {child.name!r}")
        run_dirs.append((int(raw_id), child / "reference_summary.json"))
    run_dirs.sort(key=lambda item: item[0])
    if not run_dirs:
        raise FileNotFoundError(f"Missing run reference summaries under {run_root}")
    segments = []
    for run_id, summary_path in run_dirs:
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing run reference summary: {summary_path}")
        try:
            obj = json.loads(summary_path.read_text(encoding="utf-8"))
            segments.append(
                {
                    "run_id": int(obj["run_id"]),
                    "run_start": int(obj["run_start"]),
                    "run_stop": int(obj["run_stop"]),
                    "run_length": int(obj["run_length"]),
                }
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Malformed run reference summary: {summary_path}") from exc
    return validate_run_segment_order(segments, source=run_root)


def validate_run_segment_order(
    segments: list[dict[str, int]],
    *,
    source: Path,
) -> list[dict[str, int]]:
    if not segments:
        raise ValueError(f"Run metadata is empty: {source}")
    out: list[dict[str, int]] = []
    previous_id = 0
    previous_stop = 0
    for index, raw in enumerate(segments, start=1):
        try:
            run_id = int(raw["run_id"])
            start = int(raw["run_start"])
            stop = int(raw["run_stop"])
            length = int(raw["run_length"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed run segment at index {index}: {raw}; source={source}") from exc
        if run_id <= previous_id:
            raise ValueError(f"Run IDs must be strictly increasing: {source}")
        if index == 1 and start != 0:
            raise ValueError(f"Run segments must start at frame 0: {source}")
        if start != previous_stop:
            raise ValueError(f"Run segments must be contiguous: {source}")
        if start < 0 or stop <= start or length != stop - start:
            raise ValueError(f"Invalid run segment: {raw}; source={source}")
        out.append(
            {
                "run_id": run_id,
                "run_start": start,
                "run_stop": stop,
                "run_length": length,
            }
        )
        previous_id = run_id
        previous_stop = stop
    return out


def validate_run_segments(run_segments: list[dict[str, int]], n_timepoints: int) -> list[dict[str, int]]:
    out = validate_run_segment_order(run_segments, source=Path("<run_segments>"))
    for seg in out:
        start = int(seg["run_start"])
        stop = int(seg["run_stop"])
        if stop > int(n_timepoints):
            raise ValueError(f"Run segment exceeds time series length={n_timepoints}: {seg}")
    if out[-1]["run_stop"] != int(n_timepoints):
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
    hipp_valid_mask: np.ndarray | None = None,
) -> LocalRegressionResult:
    hipp = np.asarray(hipp_ts, dtype=np.float32)
    cortex = np.asarray(cortex_ts, dtype=np.float32)
    if hipp.ndim != 2 or cortex.ndim != 2:
        raise ValueError(f"Expected 2D time series arrays, got hipp={hipp.shape} cortex={cortex.shape}")
    if hipp.shape[1] != cortex.shape[1]:
        raise ValueError(f"Timepoint mismatch: hipp={hipp.shape} cortex={cortex.shape}")
    if len(neighbor_indices) != int(hipp.shape[0]):
        raise ValueError(f"Expected one neighbor list per hippocampal vertex: {len(neighbor_indices)} vs {hipp.shape[0]}")
    if hipp_valid_mask is None:
        valid_hipp = np.ones((hipp.shape[0],), dtype=bool)
    else:
        valid_hipp = np.asarray(hipp_valid_mask, dtype=bool).reshape(-1)
        if valid_hipp.size != int(hipp.shape[0]):
            raise ValueError(
                "Hippocampal valid mask size mismatch: "
                f"mask={valid_hipp.size} vertices={hipp.shape[0]}"
            )
    segments = validate_run_segments(run_segments, int(hipp.shape[1]))

    controlled = hipp.copy()
    neighbor_counts = np.zeros((hipp.shape[0],), dtype=np.int32)
    skipped = np.zeros((hipp.shape[0],), dtype=bool)
    variance_before = np.nanvar(hipp.astype(np.float64), axis=1).astype(np.float32)
    local_corr = np.full((hipp.shape[0],), np.nan, dtype=np.float32)

    for vertex_idx, raw_neighbors in enumerate(neighbor_indices):
        if not bool(valid_hipp[vertex_idx]):
            continue
        neigh = np.asarray(raw_neighbors, dtype=np.int64).reshape(-1)
        neighbor_counts[vertex_idx] = int(neigh.size)
        if neigh.size == 0:
            skipped[vertex_idx] = True
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            local_mean = np.nanmean(cortex[neigh, :], axis=0).astype(np.float32)
        if not np.isfinite(local_mean).any():
            raise ValueError(
                "Unusable regression data: cortical neighbors contain no finite signal "
                f"for hippocampal vertex {vertex_idx}"
            )
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
        hipp_valid_mask=valid_hipp.astype(bool),
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
    cortex_dtseries_paths: list[Path] | None = None,
    hippocampal_surface_path: Path | None = None,
    run_metadata_path: Path | None = None,
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
    np.save(diag_dir / "hippocampal_valid_mask.npy", result.hipp_valid_mask.astype(bool))

    consumed_cortex_dtseries = [
        Path(path).resolve() for path in (cortex_dtseries_paths or [cortex_dtseries])
    ]
    consumed_run_metadata = Path(run_metadata_path).resolve() if run_metadata_path is not None else None
    consumed_hipp_surface = (
        Path(hippocampal_surface_path).resolve()
        if hippocampal_surface_path is not None
        else None
    )
    consumed_cortex_masks = {
        "L": (
            Path(source_shared_root)
            / f"sub-{subject}"
            / "reference"
            / "common"
            / "tsnr_gate"
            / "cortex_left_valid_mask.npy"
        ).resolve(),
        "R": (
            Path(source_shared_root)
            / f"sub-{subject}"
            / "reference"
            / "common"
            / "tsnr_gate"
            / "cortex_right_valid_mask.npy"
        ).resolve(),
    }

    def fingerprint(path: Path) -> dict[str, object]:
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Cannot fingerprint missing source: {resolved}")
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return {
            "path": str(resolved),
            "sha256": digest.hexdigest(),
            "size_bytes": int(resolved.stat().st_size),
        }

    source_fingerprints: dict[str, object] = {
        "hippocampal_timeseries": fingerprint(source_ts_path),
        "hippocampal_valid_mask": fingerprint(source_mask_path),
        "cortex_dtseries": [fingerprint(path) for path in consumed_cortex_dtseries],
        "left_cortex_surface": fingerprint(cortex_surface_paths["L"]),
        "right_cortex_surface": fingerprint(cortex_surface_paths["R"]),
        "cortex_left_valid_mask": fingerprint(consumed_cortex_masks["L"]),
        "cortex_right_valid_mask": fingerprint(consumed_cortex_masks["R"]),
    }
    if consumed_hipp_surface is not None:
        source_fingerprints["hippocampal_surface"] = fingerprint(consumed_hipp_surface)
    if consumed_run_metadata is not None:
        source_fingerprints["run_metadata"] = fingerprint(consumed_run_metadata)

    valid_vertices = result.hipp_valid_mask.astype(bool)
    skipped_valid = result.skipped_no_neighbor & valid_vertices
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
        "n_runs": int(len(run_segments)),
        "run_ids": [int(seg["run_id"]) for seg in run_segments],
        "run_segments": run_segments,
        "n_vertices": int(result.controlled_ts.shape[0]),
        "n_timepoints": int(result.controlled_ts.shape[1]),
        "n_valid_hippocampal_vertices": int(np.sum(valid_vertices)),
        "n_invalid_hippocampal_vertices": int(np.sum(~valid_vertices)),
        "n_vertices_with_cortex_neighbors": int(np.sum(valid_vertices & ~result.skipped_no_neighbor)),
        "n_vertices_without_cortex_neighbors": int(np.sum(skipped_valid)),
        "neighbor_count_min": int(np.min(result.neighbor_counts)) if result.neighbor_counts.size else 0,
        "neighbor_count_median": float(np.median(result.neighbor_counts)) if result.neighbor_counts.size else 0.0,
        "neighbor_count_max": int(np.max(result.neighbor_counts)) if result.neighbor_counts.size else 0,
        "mean_variance_before": float(np.nanmean(result.variance_before)),
        "mean_variance_after": float(np.nanmean(result.variance_after)),
        "mean_local_cortex_corr": float(np.nanmean(result.local_cortex_corr)),
        "coordinate_space_gate": coord_meta,
        "source_fingerprints": source_fingerprints,
        "inputs": {
            "source_surface_root": str(Path(source_surface_root).resolve()),
            "source_shared_root": str(Path(source_shared_root).resolve()),
            "hippocampal_timeseries": str(source_ts_path.resolve()),
            "hippocampal_valid_mask": str(source_mask_path.resolve()),
            "cortex_dtseries": str(cortex_dtseries.resolve()),
            "cortex_dtseries_inputs": [str(path) for path in consumed_cortex_dtseries],
            "left_cortex_surface": str(cortex_surface_paths["L"].resolve()),
            "right_cortex_surface": str(cortex_surface_paths["R"].resolve()),
            "cortex_left_valid_mask": str(consumed_cortex_masks["L"]),
            "cortex_right_valid_mask": str(consumed_cortex_masks["R"]),
            "hippocampal_surface": str(consumed_hipp_surface) if consumed_hipp_surface is not None else None,
            "run_metadata": str(consumed_run_metadata) if consumed_run_metadata is not None else None,
        },
        "outputs": {
            "controlled_timeseries": str(out_ts.resolve()),
            "neighbor_counts": str((local_neighbor_dir / "neighbor_counts.npy").resolve()),
            "skipped_no_neighbor": str((local_neighbor_dir / "skipped_no_neighbor.npy").resolve()),
            "variance_before": str((diag_dir / "variance_before.npy").resolve()),
            "variance_after": str((diag_dir / "variance_after.npy").resolve()),
            "local_cortex_corr": str((diag_dir / "local_cortex_corr.npy").resolve()),
            "hippocampal_valid_mask": str((diag_dir / "hippocampal_valid_mask.npy").resolve()),
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
                "n_runs": int(len(run_segments)),
                "run_ids": [int(seg["run_id"]) for seg in run_segments],
                "run_segments": run_segments,
            },
            "inputs": [
                str(source_ts_path.resolve()),
                str(source_mask_path.resolve()),
                *[str(path) for path in consumed_cortex_dtseries],
                str(cortex_surface_paths["L"].resolve()),
                str(cortex_surface_paths["R"].resolve()),
                str(consumed_cortex_masks["L"]),
                str(consumed_cortex_masks["R"]),
                *([str(consumed_hipp_surface)] if consumed_hipp_surface is not None else []),
                *([str(consumed_run_metadata)] if consumed_run_metadata is not None else []),
            ],
            "outputs": [
                str(out_ts.resolve()),
                str((diag_dir / "cortex_regression_summary.json").resolve()),
                str((local_neighbor_dir / "neighbor_counts.npy").resolve()),
                str((diag_dir / "hippocampal_valid_mask.npy").resolve()),
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
    dtseries_run_template: str | None = None,
    smoothings: list[str],
    hemis: list[str],
    distance_mm: float,
    run_metadata_template: str | None = None,
) -> dict[str, object]:
    subject = str(subject).replace("sub-", "")
    source_shared = Path(source_shared_root)
    source_surface = Path(source_surface_root)
    out_shared = Path(out_shared_root)
    data = Path(data_root)
    (out_shared / f"sub-{subject}" / "surface").mkdir(parents=True, exist_ok=True)

    cortex_surface_paths = {
        "L": format_subject_template(left_cortex_surface_template, data_root=data, subject=subject, hemi="L"),
        "R": format_subject_template(right_cortex_surface_template, data_root=data, subject=subject, hemi="R"),
    }
    explicit_run_metadata = None
    if run_metadata_template is not None:
        explicit_run_metadata = format_subject_template(
            run_metadata_template,
            data_root=data,
            subject=subject,
            hemi="L",
        )
    run_segments = load_run_segments(
        source_shared,
        subject,
        run_metadata=explicit_run_metadata,
    )
    if dtseries_run_template is not None:
        if dtseries_template != DEFAULT_DTSERIES_TEMPLATE:
            raise ValueError(
                "Specify either --dtseries-template or --dtseries-run-template, not both"
            )
        if explicit_run_metadata is None:
            raise ValueError(
                "--dtseries-run-template requires --run-metadata-template so run order is explicit"
            )
        dtseries_paths = [
            format_subject_run_template(
                dtseries_run_template,
                data_root=data,
                subject=subject,
                run_id=int(segment["run_id"]),
            )
            for segment in run_segments
        ]
        cortex_coords, cortex_ts, cortex_counts = load_cortex_from_dtseries_runs(
            dtseries_paths,
            left_surface=cortex_surface_paths["L"],
            right_surface=cortex_surface_paths["R"],
        )
        dt_path = dtseries_paths[0]
    else:
        dt_path = format_subject_path_template(
            dtseries_template,
            data_root=data,
            subject=subject,
        )
        dtseries_paths = [dt_path]
        cortex_coords, cortex_ts, cortex_counts = load_cortex_from_dtseries(
            dtseries=dt_path,
            left_surface=cortex_surface_paths["L"],
            right_surface=cortex_surface_paths["R"],
        )
    cortex_valid = load_optional_cortex_valid_mask(source_shared, subject, cortex_counts)
    n_outputs = 0

    for hemi in hemis:
        hipp_surface = hipp_surface_path(Path(hippunfold_batch_root), subject, hemi)
        hipp_coords = load_surface_coords(hipp_surface)
        if int(hipp_coords.shape[0]) != HIPPOCAMPAL_VERTEX_COUNT:
            raise ValueError(
                "Hippocampal surface density mismatch: "
                f"{hipp_surface} has {hipp_coords.shape[0]} vertices; "
                f"expected den-{HIPPOCAMPAL_VERTEX_COUNT}"
            )
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
                hipp_valid_mask=valid_mask,
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
                cortex_dtseries_paths=dtseries_paths,
                hippocampal_surface_path=hipp_surface,
                run_metadata_path=explicit_run_metadata,
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
    parser.add_argument(
        "--dtseries-run-template",
        default=None,
        help=(
            "Explicit per-run CIFTI template. Variables: {data_root}, {subject}, "
            "{sub_subject}, {run}, {run_id}, and {run_label}; requires "
            "--run-metadata-template."
        ),
    )
    parser.add_argument(
        "--run-metadata-template",
        default=None,
        help=(
            "Explicit per-subject JSON run metadata template. Variables: "
            "{data_root}, {subject}, and {sub_subject}; missing or malformed "
            "metadata fails closed."
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
        "dtseries_run_template": str(args.dtseries_run_template) if args.dtseries_run_template else None,
        "run_metadata_template": str(args.run_metadata_template) if args.run_metadata_template else None,
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

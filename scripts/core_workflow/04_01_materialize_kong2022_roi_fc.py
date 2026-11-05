#!/usr/bin/env python3
"""
Process Step Description:
Materialize Kong2022 ROI-level functional connectivity (FC). Computes the functional connectivity matrix between hippocampal surface vertices and Kong2022 cortical network regions of interest (ROIs).

Main Flow Path:
- 6th step in the numbered pipeline sequence (ROI functional connectivity materialization for kong2022):
  01_01_run_hippunfold_t1w_local.py -> 01_02_check_hippunfold_outputs.py -> 02_01_prepare_group_inputs.py -> 02_02_build_qc_manifest.py -> 03_01/03_02 reference building -> 04_01_materialize_kong2022_roi_fc.py (this step) -> 05_01 gradient computation -> 05_02 group atlas parcellation -> 06_xx presentation rendering -> 07_01 group matrix overview.

Inputs:
- Command-line arguments:
  - --shared-root: Explicit FC output root; when omitted, normal runs use `outputs_hcp/inputs/kong2022/shared_inputs/pfm` and `--ctxdrop` uses the canonical `.../shared_inputs/pfm_ctxdrop` root.
  - --surface-root: Path to shared BOLD surfaces (default: `outputs_hcp/inputs/kong2022/shared_inputs/surface`).
  - --subjects-manifest: Path to QC-passed subject list manifest (default: `outputs_hcp/inputs/kong2022/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt`).
  - --subjects: Space-separated list of explicit subject IDs.
  - --smoothings: Smoothing levels to process: "2mm", "4mm".
  - --hemis: Hemispheres to process: "L", "R".
  - --jobs: Subject-level process concurrency cap (default: 1, hard-capped at 10).
- Physical inputs:
  - Resampled BOLD timeseries: `{surface_root}/sub-<subject>/surface/{smoothing}/sub-<subject>_hemi-{hemi}_timeseries.npy`.
  - BOLD active mask: `{surface_root}/sub-<subject>/surface/validity/sub-<subject>_hemi-{hemi}_valid_mask.npy`.
  - Reference timeseries: `{shared_root}/sub-<subject>/reference/kong2022/cortex_roi_parcel_timeseries.npy`.
  - Reference parcels TSV: `{shared_root}/sub-<subject>/reference/kong2022/cortex_roi_parcels.tsv`.

Outputs:
- Command-line output / file outputs:
  - Ordered 400-ROI FC bank under the historical filename `hipp_vertex_to_network_fc.npy`; normal Kong2022 runs also write `hipp_vertex_to_roi_fc.npy` and separate `hipp_vertex_to_canonical7_fc.npy` summaries.
  - Active-only caches use corresponding `_active.npy` filenames. The maintained ctxdrop methods consume the historical 400-ROI bank directly.
  - Run manifest: `{shared_root}/sub-<subject>/fc/kong2022/{smoothing}/hemi_{hemi}/fc_summary.json`.
"""
from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import nibabel as nib
import numpy as np
from lib.extract_cortex_kong2022_route_a_roi_timeseries import (
    align_surface_labels_to_dt_vertices,
    load_surface_label_vector,
    route_a_label_paths,
)
from lib.extract_cortex_merged_network_timeseries import extract_structure_data
from lib.fc_math import (
    center_run_timeseries,
    compact_active_vertices,
    corrcoef_rows,
)
from lib.io_utils import (
    load_optional_cortex_valid_mask,
    load_subject_ids,
)
from lib.io_utils import (
    segments_from_explicit_metadata as _segments_from_explicit_metadata,
)
from lib.io_utils import (
    validate_run_segment_order as _validate_run_segments,
)
from lib.kong2022_fc_contract import (
    authoritative_kong2022_roi_registry,  # noqa: F401
    kong2022_registry_digest,
    read_kong2022_roi_registry,
    validate_kong2022_roi_registry,
)
from lib.kong2022_label_utils import (
    aggregate_kong2022_network_timeseries,
    canonical_families_7_order,
    load_kong2022_merge_mapping,
)
from lib.lynch2024_reference import (
    lynch2024_7net_reference_contract,
    lynch2024_20net_order,
)
from lib.output_layout import (
    DEFAULT_CORTEX_ROUTEA_ROOT,
    DEFAULT_HIPPUNFOLD_BATCH_ROOT,
    DEFAULT_KONG2022_MANIFEST_ROOT,
    DEFAULT_KONG2022_ROOT,
)
from lib.output_layout import (
    DEFAULT_KONG400_CTXDROP_SHARED_ROOT as DEFAULT_HCP_KONG400_CTXDROP_SHARED_ROOT,
)
from lib.path_templates import format_subject_path_template
from lib.surface_coordinates import (
    MNI_SPACE,
    hipp_surface_path,
    load_surface_coords,
    native_cortex_coordinates,
    query_local_cortex_neighbors,
)
from lib.task_quality import QC_DATASETS, require_cortex_reference_policy
from lib.workflow_utils import capped_jobs, utc_now_iso, write_json

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROUTE_ROOT = DEFAULT_KONG2022_ROOT
DEFAULT_SHARED_INPUTS_ROOT = DEFAULT_ROUTE_ROOT / "shared_inputs"
DEFAULT_SHARED_ROOT = DEFAULT_SHARED_INPUTS_ROOT / "pfm"
DEFAULT_KONG400_CTXDROP_SHARED_ROOT = DEFAULT_HCP_KONG400_CTXDROP_SHARED_ROOT
DEFAULT_SURFACE_ROOT = DEFAULT_SHARED_INPUTS_ROOT / "surface"
DEFAULT_SUBJECTS = DEFAULT_KONG2022_MANIFEST_ROOT / "hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt"
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "hippunfold_input"
DEFAULT_DTSERIES_TEMPLATE = "{data_root}/{sub_subject}/func/{sub_subject}_task-rest_run-concat.dtseries.nii"
DEFAULT_ROUTE_A_ROOT = DEFAULT_CORTEX_ROUTEA_ROOT
DEFAULT_LEFT_CORTEX_SURFACE_TEMPLATE = (
    "{data_root}/{sub_subject}/anat/{sub_subject}_hemi-L_space-fsLR_den-32k_desc-MSMAll_midthickness.surf.gii"
)
DEFAULT_RIGHT_CORTEX_SURFACE_TEMPLATE = (
    "{data_root}/{sub_subject}/anat/{sub_subject}_hemi-R_space-fsLR_den-32k_desc-MSMAll_midthickness.surf.gii"
)
DEFAULT_ATLAS = "kong2022"
ATLAS = DEFAULT_ATLAS
SMOOTHINGS = ["2mm", "4mm"]
HEMIS = ["L", "R"]


def resolve_shared_root_for_mode(shared_root: str | Path | None, *, ctxdrop: bool) -> Path:
    if shared_root is not None:
        return Path(shared_root).resolve()
    return (DEFAULT_KONG400_CTXDROP_SHARED_ROOT if ctxdrop else DEFAULT_SHARED_ROOT).resolve()


def resolve_reference_root_for_mode(
    reference_root: str | Path | None,
    *,
    ctxdrop: bool,
    shared_root: Path | None = None,
) -> Path:
    if reference_root is not None:
        return Path(reference_root).resolve()
    if ctxdrop:
        return DEFAULT_SHARED_ROOT.resolve()
    return (shared_root if shared_root is not None else DEFAULT_SHARED_ROOT).resolve()


@dataclass(frozen=True)
class CtxDropFCResult:
    fc: np.ndarray
    roi_cortex_counts: np.ndarray
    excluded_neighbor_counts: np.ndarray


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
        if neigh.size and (int(np.min(neigh)) < 0 or int(np.max(neigh)) >= int(cortex.shape[0])):
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


def roi_ids_and_names_from_rows(roi_rows: list[dict[str, Any]]) -> tuple[list[int], list[str]]:
    roi_ids: list[int] = []
    roi_names: list[str] = []
    for idx, row in enumerate(roi_rows):
        raw_id = row.get("parcel_id", idx + 1)
        roi_ids.append(int(str(raw_id)))
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
    atlas_slug = DEFAULT_ATLAS
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
    run_segments = _validate_run_segments(run_segments, source=fc_dir)
    pooled_lengths = [int(seg["run_length"]) for seg in run_segments]
    pooled_ts = center_run_timeseries(ts, pooled_lengths)
    pooled_cortex = center_run_timeseries(cortex_ts, pooled_lengths)
    result = compute_ctxdrop_vertex_to_roi_fc(
        hipp_ts=pooled_ts,
        cortex_ts=pooled_cortex,
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
            "atlas_slug": atlas_slug,
            "smooth": smoothing,
            "hemi": hemi,
            "route_tag": "pfm_ctxdrop",
            "ctxdrop_distance_mm": float(distance_mm),
            "ctxdrop_policy": "per_hippocampal_vertex_exclude_valid_cortex_vertices_within_radius_before_roi_mean",
            "fc_shape": [int(pearson_full.shape[0]), int(pearson_full.shape[1])],
            "active_fc_shape": [int(pearson_valid.shape[0]), int(pearson_valid.shape[1])],
            "n_vertices_total": int(ts.shape[0]),
            "n_vertices_valid": int(valid.sum()),
            "n_vertices_invalid_initial": int(ts.shape[0] - valid.sum()),
            "n_vertices_clustered": int(valid.sum()),
            "n_timepoints": int(ts.shape[1]),
            "n_roi_targets_used": int(len(roi_names)),
            "run_labels": run_labels,
            "run_lengths": run_lengths,
            "fc_definition": "pearson_vertex_to_kong2022_roi_timeseries_ctxdrop_local20mm",
            "run_centering": "within_run_mean_removed_variance_preserved",
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
                "atlas_slug": atlas_slug,
                "smooth": smoothing,
                "hemi": hemi,
                "route_tag": "pfm_ctxdrop",
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


def reference_fc_target_files(atlas_slug: str) -> tuple[str, str, str]:
    """Return the direct reference targets for vertex-to-network FC."""
    if str(atlas_slug) == "kong2022":
        # The maintained Kong FC filename is a historical name for the
        # ordered 400-ROI bank. Canonical network summaries are emitted to a
        # separate explicit path below.
        return (
            "cortex_roi_parcels.tsv",
            "cortex_roi_parcel_timeseries.npy",
            "parcel_name",
        )
    if str(atlas_slug) == "lynch2024_20net":
        return (
            "cortex_lynch20_networks.tsv",
            "cortex_lynch20_network_timeseries.npy",
            "lynch20_network",
        )
    if str(atlas_slug) == "lynch2024_7net":
        return (
            "cortex_canonical7_networks.tsv",
            "cortex_canonical7_network_timeseries.npy",
            "canonical7_network",
        )
    return (
        "cortex_roi_parcels.tsv",
        "cortex_roi_parcel_timeseries.npy",
        "parcel_name",
    )


def roi_fc_filename_for_method(method: str) -> str:
    """Return the FC cache consumed by ROI-profile methods under Kong2022."""
    if str(method) in {"group-roi-sim-spectral", "wta-top"}:
        return "hipp_vertex_to_roi_fc.npy"
    return "hipp_vertex_to_network_fc.npy"


def roi_fc_active_filename_for_method(method: str) -> str:
    if str(method) in {"group-roi-sim-spectral", "wta-top"}:
        return "hipp_vertex_to_roi_fc_active.npy"
    return "hipp_vertex_to_network_fc_active.npy"


def read_reference_target_names(path: Path, name_column: str) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise RuntimeError(f"Empty reference TSV: {path}")
    if name_column not in rows[0]:
        raise RuntimeError(f"Reference TSV {path} is missing required column {name_column!r}")
    return [str(row[name_column]) for row in rows]


def validate_lynch_reference_contract(
    *,
    reference_dir: Path,
    atlas_slug: str,
    roi_names: list[str],
    roi_timeseries: np.ndarray,
) -> dict[str, object]:
    """Validate Lynch reference metadata before materializing FC."""
    manifest_path = Path(reference_dir) / "stage_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing Lynch reference provenance manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    merge_config = REPO_ROOT / "config" / "cross_atlas_network_merge.json"
    if atlas_slug == "lynch2024_20net":
        expected: dict[str, Any] = {
            "network_taxonomy": "lynch20",
            "network_count": 20,
            "network_registry_column": "lynch20_network",
            "network_name_column": "lynch20_network",
            "noise_excluded": True,
            "network_order": lynch2024_20net_order(merge_config),
        }
    elif atlas_slug == "lynch2024_7net":
        expected = dict(lynch2024_7net_reference_contract(merge_config))
    else:
        return {}
    for key, expected_value in expected.items():
        if manifest.get(key) != expected_value:
            raise ValueError(
                f"Lynch reference contract mismatch: {manifest_path} key={key!r} "
                f"consumed={manifest.get(key)!r} expected={expected_value!r}"
            )
    expected_order = [str(name) for name in expected["network_order"]]
    if roi_names != expected_order:
        raise ValueError(
            f"Lynch reference order mismatch: {reference_dir} consumed={roi_names} expected={expected_order}"
        )
    expected_count = int(expected["network_count"])
    if int(np.asarray(roi_timeseries).shape[0]) != expected_count:
        raise ValueError(
            f"Lynch reference network count mismatch: {reference_dir} "
            f"rows={np.asarray(roi_timeseries).shape[0]} expected={expected_count}"
        )
    return {str(key): value for key, value in expected.items()}


def derive_kong2022_canonical7_registry(
    validated_roi_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Derive canonical7 labels from a validated Kong2022 400-ROI registry.

    The legacy ROI TSVs intentionally contain the stable ``family17`` and
    ``canonical8`` fields but may omit ``canonical7``.  This explicit
    taxonomy-only transform preserves the validated 400-row order and never
    reads or writes a direct seven-network reference.
    """
    if len(validated_roi_rows) != 400:
        raise ValueError(
            "Kong2022 canonical7 derivation requires exactly 400 validated ROI rows: "
            f"rows={len(validated_roi_rows)}"
        )
    mapping = load_kong2022_merge_mapping(taxonomy="canonical7")
    derived: list[dict[str, object]] = []
    expected = set(canonical_families_7_order())
    for index, raw_row in enumerate(validated_roi_rows, start=1):
        row = dict(raw_row)
        family17 = str(row.get("family17", "")).strip()
        provided = str(row.get("canonical7", "")).strip()
        if family17:
            if family17 not in mapping:
                raise ValueError(
                    "Kong2022 canonical7 derivation encountered an unknown family17 "
                    f"at ROI row {index}: {family17!r}"
                )
            derived_label = mapping[family17]
            if provided and provided != derived_label:
                raise ValueError(
                    "Kong2022 canonical7 label disagrees with the maintained family17 mapping "
                    f"at ROI row {index}: provided={provided!r} derived={derived_label!r}"
                )
            row["canonical7"] = derived_label
        elif provided in expected:
            # Some synthetic/legacy registries already carry canonical7 but omit
            # family17; preserve that explicit taxonomy label while retaining the
            # 400-ROI feature space contract.
            row["canonical7"] = provided
        else:
            raise ValueError(
                "Kong2022 canonical7 derivation requires either a known family17 "
                f"or an explicit canonical7 label at ROI row {index}: "
                f"family17={family17!r} canonical7={provided!r}"
            )
        derived.append(row)
    observed = {str(row["canonical7"]) for row in derived}
    if observed != expected:
        raise ValueError(
            "Kong2022 canonical7 derivation does not cover the fixed seven-network order: "
            f"observed={sorted(observed)} expected={sorted(expected)}"
        )
    return derived


def prepare_reference_profiles(
    *,
    ref_subj_root: Path,
    atlas_slug: str,
    target_tsv_path: Path,
    target_timeseries_path: Path,
    roi_ts: np.ndarray,
    roi_names: list[str],
) -> tuple[
    np.ndarray | None,
    list[str] | None,
    Path | None,
    Path | None,
    np.ndarray | None,
    list[str] | None,
    list[dict[str, object]] | None,
    dict[str, object],
]:
    roi_profile_ts: np.ndarray | None = None
    roi_profile_names: list[str] | None = None
    roi_profile_tsv_path: Path | None = None
    roi_profile_timeseries_path: Path | None = None
    if atlas_slug == "kong2022":
        roi_profile_tsv_path = ref_subj_root / "reference" / atlas_slug / "cortex_roi_parcels.tsv"
        roi_profile_timeseries_path = ref_subj_root / "reference" / atlas_slug / "cortex_roi_parcel_timeseries.npy"
        roi_profile_ts = np.load(roi_profile_timeseries_path).astype(np.float32)
        roi_profile_names = read_reference_target_names(roi_profile_tsv_path, "parcel_name")

    kong_network_ts: np.ndarray | None = None
    kong_network_names: list[str] | None = None
    kong_registry_rows: list[dict[str, object]] | None = None
    if atlas_slug == "kong2022":
        kong_registry_rows = read_kong2022_roi_registry(target_tsv_path)
        validate_kong2022_roi_registry(kong_registry_rows, path=target_tsv_path)
        # The legacy source TSV is the authoritative ordered 400-ROI bank;
        # canonical7 is derived only after that identity/order validation.
        kong_registry_rows = derive_kong2022_canonical7_registry(kong_registry_rows)
        expected_order = [str(name) for name in canonical_families_7_order()]
        if int(roi_ts.shape[0]) != 400:
            raise ValueError(
                f"Kong2022 ROI reference must contain exactly 400 rows: {target_timeseries_path} "
                f"rows={roi_ts.shape[0]}"
            )
        registry_names = [str(row["roi_name"]) for row in kong_registry_rows]
        if roi_names != registry_names:
            raise ValueError(f"Kong2022 ROI names disagree with registry: {target_tsv_path}")
        kong_network_names = expected_order
        kong_network_ts = aggregate_kong2022_network_timeseries(
            roi_ts, kong_registry_rows, expected_order, network_column="canonical7"
        ).astype(np.float32)
        taxonomy_evidence = {
            "feature_taxonomy": "kong2022_roi400",
            "feature_count": 400,
            "roi_registry_column": "roi_name",
            "roi_registry_order": [str(row["roi_name"]) for row in kong_registry_rows],
            "roi_registry_digest": kong2022_registry_digest(kong_registry_rows),
            "network_taxonomy": "canonical7",
            "network_count": int(len(expected_order)),
            "network_registry_column": "canonical7",
            "network_order": expected_order,
            "reference_branch": "kong2022_roi400_upstream",
            "reference_upstream_reuse": False,
        }
    elif atlas_slug.startswith("lynch2024_"):
        taxonomy_evidence = validate_lynch_reference_contract(
            reference_dir=target_tsv_path.parent,
            atlas_slug=atlas_slug,
            roi_names=roi_names,
            roi_timeseries=roi_ts,
        )
    else:
        taxonomy_evidence = {}
    return (
        roi_profile_ts,
        roi_profile_names,
        roi_profile_tsv_path,
        roi_profile_timeseries_path,
        kong_network_ts,
        kong_network_names,
        kong_registry_rows,
        taxonomy_evidence,
    )


def load_run_segments(
    subj_root: Path,
    reference_atlas: str = "kong2022",
    *,
    run_metadata: Path | None = None,
) -> list[dict[str, int]]:
    """Load ordered run boundaries from one explicit reference contract."""
    if run_metadata is not None:
        return _segments_from_explicit_metadata(Path(run_metadata).resolve())
    current_summary = subj_root / "reference" / str(reference_atlas) / "reference_summary.json"
    return _segments_from_explicit_metadata(current_summary)


def read_roi_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = [dict(r) for r in csv.DictReader(f, delimiter="\t")]
    if not rows:
        raise RuntimeError(f"Empty ROI TSV: {path}")
    return rows


format_subject_template = format_subject_path_template


def load_ctxdrop_cortex_inputs(
    *,
    data_root: Path,
    subject: str,
    route_a_root: Path,
    left_cortex_surface_template: str,
    right_cortex_surface_template: str,
    dtseries_template: str = DEFAULT_DTSERIES_TEMPLATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int], dict[str, Path]]:
    subject = str(subject).replace("sub-", "")
    dt_path = format_subject_path_template(
        dtseries_template,
        data_root=data_root,
        subject=subject,
    )
    dt_img = cast(Any, nib.load(str(dt_path)))
    dt_data = np.asarray(dt_img.dataobj, dtype=np.float32)
    if dt_data.ndim != 2:
        raise ValueError(f"Expected 2D dtseries data: {dt_path} shape={dt_data.shape}")
    dt_data_t = dt_data.T
    axis = dt_img.header.get_axis(1)
    left_vertices, left_dt = extract_structure_data(axis, dt_data_t, "CORTEX_LEFT")
    right_vertices, right_dt = extract_structure_data(axis, dt_data_t, "CORTEX_RIGHT")

    lh_labels_path, rh_labels_path = route_a_label_paths(route_a_root=route_a_root, subject=subject)
    lh_full = load_surface_label_vector(lh_labels_path)
    rh_full = load_surface_label_vector(rh_labels_path)
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
    paths = {"dtseries": dt_path.resolve(), "left_surface": cortex_surface_paths["L"], "right_surface": cortex_surface_paths["R"], "route_a_labels": lh_labels_path.parent.resolve()}
    return coords, ts, roi_ids, counts, paths


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
    dataset: str,
) -> None:
    subject = str(subject).replace("sub-", "")
    subj_root = Path(shared_root) / f"sub-{subject}"
    ref_subj_root = Path(reference_root) / f"sub-{subject}"
    require_cortex_reference_policy(ref_subj_root / "reference" / ATLAS, dataset=dataset)
    surf_subj_root = Path(surface_root) / f"sub-{subject}"
    mask_subj_root = Path(mask_root) / f"sub-{subject}"
    atlas_slug = ATLAS
    fc_dir = subj_root / "fc" / atlas_slug / smoothing / f"hemi_{hemi}"

    ts = np.load(surf_subj_root / "surface" / smoothing / f"sub-{subject}_hemi-{hemi}_timeseries.npy").astype(np.float32)
    valid_mask = np.load(mask_subj_root / "surface" / "validity" / f"sub-{subject}_hemi-{hemi}_valid_mask.npy").astype(bool)
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
    atlas: str = DEFAULT_ATLAS,
    run_metadata: Path | None = None,
    *,
    dataset: str,
) -> None:
    subj_root = shared_root / f"sub-{subject}"
    ref_subj_root = (reference_root or shared_root) / f"sub-{subject}"
    surf_subj_root = surface_root / f"sub-{subject}"
    mask_subj_root = (mask_root or surface_root) / f"sub-{subject}"
    atlas_slug = str(atlas)
    fc_dir = subj_root / "fc" / atlas_slug / smoothing / f"hemi_{hemi}"
    if atlas_slug == "kong2022":
        require_cortex_reference_policy(ref_subj_root / "reference" / atlas_slug, dataset=dataset)

    ts = np.load(surf_subj_root / "surface" / smoothing / f"sub-{subject}_hemi-{hemi}_timeseries.npy").astype(np.float32)
    valid_mask = np.load(mask_subj_root / "surface" / "validity" / f"sub-{subject}_hemi-{hemi}_valid_mask.npy").astype(bool)
    target_tsv_name, target_timeseries_name, target_name_column = reference_fc_target_files(atlas_slug)
    target_timeseries_path = ref_subj_root / "reference" / atlas_slug / target_timeseries_name
    target_tsv_path = ref_subj_root / "reference" / atlas_slug / target_tsv_name
    roi_ts = np.load(target_timeseries_path).astype(np.float32)
    roi_names = read_reference_target_names(target_tsv_path, target_name_column)
    (
        roi_profile_ts,
        roi_profile_names,
        roi_profile_tsv_path,
        roi_profile_timeseries_path,
        kong_network_ts,
        kong_network_names,
        kong_registry_rows,
        taxonomy_evidence,
    ) = prepare_reference_profiles(
        ref_subj_root=ref_subj_root,
        atlas_slug=atlas_slug,
        target_tsv_path=target_tsv_path,
        target_timeseries_path=target_timeseries_path,
        roi_ts=roi_ts,
        roi_names=roi_names,
    )

    if roi_profile_ts is not None:
        if roi_profile_ts.ndim != 2:
            raise ValueError(f"Expected 2D Kong2022 ROI profile reference: {roi_profile_ts.shape}")
        if roi_profile_ts.shape[1] != ts.shape[1]:
            raise RuntimeError(
                f"Time mismatch for Kong2022 ROI profile sub-{subject} {smoothing} hemi-{hemi}: "
                f"hipp={ts.shape} roi={roi_profile_ts.shape}"
            )
        if roi_profile_ts.shape[0] != len(roi_profile_names or []):
            raise RuntimeError(
                f"Kong2022 ROI TSV mismatch for sub-{subject}: "
                f"ts_rows={roi_profile_ts.shape[0]} roi_names={len(roi_profile_names or [])}"
            )

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

    run_segments = load_run_segments(
        ref_subj_root,
        reference_atlas=atlas_slug,
        run_metadata=run_metadata,
    )
    pooled_lengths = [int(seg["run_length"]) for seg in run_segments]
    pearson_valid = corrcoef_rows(ts[valid_mask, :], roi_ts, run_lengths=pooled_lengths)
    fc_dir.mkdir(parents=True, exist_ok=True)
    pearson_full = np.full((ts.shape[0], roi_ts.shape[0]), np.nan, dtype=np.float32)
    pearson_full[valid_mask, :] = pearson_valid
    full_fc_path = fc_dir / "hipp_vertex_to_network_fc.npy"
    if atlas_slug == "lynch2024_20net" and pearson_full.shape[-1] != 20:
        raise ValueError(
            f"Lynch2024 formal FC must contain exactly 20 networks: {full_fc_path} "
            f"shape={pearson_full.shape}"
        )
    if atlas_slug == "lynch2024_7net" and pearson_full.shape[-1] != 7:
        raise ValueError(
            f"Lynch2024 7net FC must contain exactly 7 networks: {full_fc_path} "
            f"shape={pearson_full.shape}"
        )
    if atlas_slug == "kong2022" and pearson_full.shape[-1] != 400:
        raise ValueError(
            f"Kong2022 ROI FC must contain exactly 400 columns: {full_fc_path} "
            f"shape={pearson_full.shape}"
        )
    np.save(full_fc_path, pearson_full.astype(np.float32))
    np.save(fc_dir / "hipp_vertex_to_network_fc_active.npy", compact_active_vertices(pearson_full, valid_mask))
    if roi_profile_ts is not None:
        roi_pearson_valid = corrcoef_rows(ts[valid_mask, :], roi_profile_ts, run_lengths=pooled_lengths)
        roi_pearson_full = np.full((ts.shape[0], roi_profile_ts.shape[0]), np.nan, dtype=np.float32)
        roi_pearson_full[valid_mask, :] = roi_pearson_valid
        # Explicit ROI aliases make the feature family clear while retaining
        # the historical network_fc filename consumed by old launchers.
        np.save(fc_dir / "hipp_vertex_to_roi_fc.npy", roi_pearson_full.astype(np.float32))
        np.save(fc_dir / "hipp_vertex_to_roi_fc_active.npy", compact_active_vertices(roi_pearson_full, valid_mask))
    if atlas_slug == "kong2022":
        if kong_network_ts is None or kong_network_names is None:
            raise RuntimeError("Kong2022 canonical7 summary was not materialized from the 400-ROI reference")
        network_valid = corrcoef_rows(ts[valid_mask, :], kong_network_ts, run_lengths=pooled_lengths)
        network_full = np.full((ts.shape[0], len(kong_network_names)), np.nan, dtype=np.float32)
        network_full[valid_mask, :] = network_valid
        np.save(fc_dir / "hipp_vertex_to_canonical7_fc.npy", network_full.astype(np.float32))
        np.save(fc_dir / "hipp_vertex_to_canonical7_fc_active.npy", compact_active_vertices(network_full, valid_mask))

    run_labels: list[str] = []
    run_lengths: list[int] = []
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
        if atlas_slug == "kong2022":
            assert kong_network_ts is not None
            run_network = np.asarray(kong_network_ts[:, run_start:run_stop], dtype=np.float32)
            run_network_valid = corrcoef_rows(run_ts[valid_mask, :], run_network)
            np.save(run_dir / "hipp_vertex_to_canonical7_fc_active.npy", run_network_valid.astype(np.float32))
        if roi_profile_ts is not None:
            run_roi_profile = roi_profile_ts[:, run_start:run_stop].astype(np.float32, copy=False)
            run_roi_profile_valid = corrcoef_rows(run_ts[valid_mask, :], run_roi_profile)
            np.save(run_dir / "hipp_vertex_to_roi_fc_active.npy", run_roi_profile_valid.astype(np.float32))
    total_run_len = int(sum(run_lengths))
    if total_run_len != int(ts.shape[1]):
        raise RuntimeError(
            f"Run lengths do not sum to full time series for sub-{subject} {smoothing} hemi-{hemi}: "
            f"{total_run_len} vs {ts.shape[1]}"
        )

    if atlas_slug.startswith("lynch2024_"):
        taxonomy_evidence = {**taxonomy_evidence, "network_order": roi_names}
    write_json(
        fc_dir / "fc_summary.json",
        {
            "subject": subject,
            "atlas_slug": atlas_slug,
            "smooth": smoothing,
            "hemi": hemi,
            "fc_shape": [int(pearson_full.shape[0]), int(pearson_full.shape[1])],
            "active_fc_shape": [int(pearson_valid.shape[0]), int(pearson_valid.shape[1])],
            "n_vertices_total": int(ts.shape[0]),
            "n_vertices_valid": int(valid_mask.sum()),
            "n_vertices_invalid_initial": int(ts.shape[0] - valid_mask.sum()),
            "n_vertices_clustered": int(valid_mask.sum()),
            "n_timepoints": int(ts.shape[1]),
            "n_roi_targets_used": int(roi_ts.shape[0]),
            "n_network_targets_used": int(len(kong_network_names)) if kong_network_names is not None else int(roi_ts.shape[0]),
            "n_roi_profile_targets_used": int(roi_profile_ts.shape[0]) if roi_profile_ts is not None else None,
            "network_fc_path": str((fc_dir / "hipp_vertex_to_canonical7_fc.npy").resolve()) if atlas_slug == "kong2022" else str(full_fc_path.resolve()),
            "roi_fc_path": str((fc_dir / "hipp_vertex_to_roi_fc.npy").resolve()) if roi_profile_ts is not None else None,
            "run_labels": run_labels,
            "run_ids": [int(seg["run_id"]) for seg in run_segments],
            "run_lengths": run_lengths,
            "run_segments": run_segments,
            "run_centering": "within_run_mean_removed_variance_preserved",
            "run_metadata_path": str(Path(run_metadata).resolve()) if run_metadata is not None else None,
            "fc_definition": (
                "pearson_vertex_to_kong2022_roi_timeseries"
                if atlas_slug == "kong2022"
                else f"pearson_vertex_to_{atlas_slug}_roi_timeseries"
            ),
            **taxonomy_evidence,
        },
    )
    write_json(
        fc_dir / "stage_manifest.json",
        {
            "stage": "shared_fc",
            "status": "done",
            "timestamp_utc": utc_now_iso(),
            **taxonomy_evidence,
            "params": {
                "subject": subject,
                "atlas_slug": atlas_slug,
                "smooth": smoothing,
                "hemi": hemi,
                "run_ids": [int(seg["run_id"]) for seg in run_segments],
                "run_lengths": run_lengths,
                "run_segments": run_segments,
                "run_metadata_path": str(Path(run_metadata).resolve()) if run_metadata is not None else None,
                "fc_definition": (
                    f"pearson_vertex_to_{atlas_slug}_network_timeseries"
                    if atlas_slug == "kong2022"
                    else f"pearson_vertex_to_{atlas_slug}_roi_timeseries"
                ),
            },
            "inputs": [
                str(target_timeseries_path.resolve()),
                str(target_tsv_path.resolve()),
                *(
                    [str(roi_profile_timeseries_path.resolve()), str(roi_profile_tsv_path.resolve())]
                    if roi_profile_ts is not None and roi_profile_timeseries_path is not None and roi_profile_tsv_path is not None
                    else []
                ),
                *(
                    [str((target_tsv_path.parent / "stage_manifest.json").resolve())]
                    if atlas_slug.startswith("lynch2024_") or atlas_slug == "kong2022"
                    else []
                ),
                str((surf_subj_root / "surface" / smoothing / f"sub-{subject}_hemi-{hemi}_timeseries.npy").resolve()),
                str((mask_subj_root / "surface" / "validity" / f"sub-{subject}_hemi-{hemi}_valid_mask.npy").resolve()),
            ],
            "outputs": [
                str((fc_dir / "hipp_vertex_to_network_fc.npy").resolve()),
                str((fc_dir / "hipp_vertex_to_network_fc_active.npy").resolve()),
                *(
                    [
                        str((fc_dir / "hipp_vertex_to_canonical7_fc.npy").resolve()),
                        str((fc_dir / "hipp_vertex_to_canonical7_fc_active.npy").resolve()),
                    ]
                    if atlas_slug == "kong2022"
                    else []
                ),
                *(
                    [
                        str((fc_dir / "hipp_vertex_to_roi_fc.npy").resolve()),
                        str((fc_dir / "hipp_vertex_to_roi_fc_active.npy").resolve()),
                    ]
                    if roi_profile_ts is not None
                    else []
                ),
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
    left_cortex_surface_template: str = DEFAULT_LEFT_CORTEX_SURFACE_TEMPLATE,
    right_cortex_surface_template: str = DEFAULT_RIGHT_CORTEX_SURFACE_TEMPLATE,
    ctxdrop_distance_mm: float = 20.0,
    dtseries_template: str = DEFAULT_DTSERIES_TEMPLATE,
    atlas: str = DEFAULT_ATLAS,
    run_metadata_template: str | None = None,
    cortex_coordinate_space: str = MNI_SPACE,
    *,
    dataset: str,
) -> dict[str, object]:
    subject = str(subject).replace("sub-", "")
    shared_root = Path(shared_root_str)
    surface_root = Path(surface_root_str)
    reference_root = Path(reference_root_str)
    mask_root = Path(mask_root_str)
    n_fc = 0

    atlas_slug = str(atlas)
    if ctxdrop:
        if atlas_slug != DEFAULT_ATLAS:
            raise ValueError("--ctxdrop is only supported for --atlas kong2022")
        data_root = Path(data_root_str or DEFAULT_DATA_ROOT).resolve()
        hippunfold_batch_root = Path(hippunfold_batch_root_str or DEFAULT_HIPPUNFOLD_BATCH_ROOT).resolve()
        route_a_root = Path(route_a_root_str or DEFAULT_ROUTE_A_ROOT).resolve()
        cortex_coords, cortex_ts, cortex_roi_ids, cortex_counts, _cortex_paths = load_ctxdrop_cortex_inputs(
            data_root=data_root,
            subject=subject,
            route_a_root=route_a_root,
            left_cortex_surface_template=left_cortex_surface_template,
            right_cortex_surface_template=right_cortex_surface_template,
            dtseries_template=dtseries_template,
        )
        cortex_coords = native_cortex_coordinates(cortex_coords, subject=subject, space=cortex_coordinate_space)
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
                    dataset=dataset,
                )
                n_fc += 1
        return {"subject": f"sub-{subject}", "fc_combos": int(n_fc), "ctxdrop": True}

    for smoothing in smoothings:
        for hemi in hemis:
            run_metadata = (
                Path(
                    run_metadata_template.format(
                        subject=subject,
                        sub_subject=f"sub-{subject}",
                        shared_root=str(shared_root),
                        reference_root=str(reference_root),
                    )
                ).resolve()
                if run_metadata_template is not None
                else None
            )
            materialize_fc(
                shared_root,
                surface_root,
                subject,
                smoothing,
                hemi,
                reference_root=reference_root,
                mask_root=mask_root,
                atlas=atlas_slug,
                run_metadata=run_metadata,
                dataset=dataset,
            )
            n_fc += 1
    return {"subject": f"sub-{subject}", "fc_combos": int(n_fc), "ctxdrop": False}

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize Kong2022 hippocampal vertex-to-ROI FC caches.")
    parser.add_argument("--dataset", required=True, choices=QC_DATASETS)
    parser.add_argument("--shared-root", default=None, help="Output FC root; omitted ctxdrop runs use the dated HCP 400-ROI control root.")
    parser.add_argument("--surface-root", default=str(DEFAULT_SURFACE_ROOT))
    parser.add_argument("--reference-root", default=None, help="Read cortex reference/run metadata from this root; defaults to --shared-root.")
    parser.add_argument("--mask-root", default=None, help="Read hippocampal valid masks from this root; defaults to --surface-root.")
    parser.add_argument(
        "--run-metadata-template",
        default=None,
        help=(
            "Explicit per-subject JSON run metadata template. Variables: "
            "{subject}, {sub_subject}, {shared_root}, and {reference_root}; "
            "missing or malformed metadata fails closed."
        ),
    )
    parser.add_argument("--subjects-manifest", default=str(DEFAULT_SUBJECTS))
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--atlas", choices=("kong2022", "lynch2024_20net", "lynch2024_7net"), default=DEFAULT_ATLAS, help="Reference/FC atlas slug under sub-*/reference and sub-*/fc.")
    parser.add_argument("--smoothings", nargs="+", default=SMOOTHINGS, choices=SMOOTHINGS)
    parser.add_argument("--hemis", nargs="+", default=HEMIS, choices=HEMIS)
    parser.add_argument("--jobs", type=int, default=1, help="Subject-level parallel workers; hard-capped at 10.")
    parser.add_argument("--ctxdrop", action="store_true", help="Materialize RouteA ctxdrop FC by excluding local cortex vertices before ROI averaging.")
    parser.add_argument("--ctxdrop-distance-mm", type=float, default=20.0)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--hippunfold-batch-root", default=str(DEFAULT_HIPPUNFOLD_BATCH_ROOT))
    parser.add_argument("--route-a-root", default=str(DEFAULT_ROUTE_A_ROOT))
    parser.add_argument("--left-cortex-surface-template", default=DEFAULT_LEFT_CORTEX_SURFACE_TEMPLATE)
    parser.add_argument("--right-cortex-surface-template", default=DEFAULT_RIGHT_CORTEX_SURFACE_TEMPLATE)
    parser.add_argument('--cortex-coordinate-space', choices=['T1w', MNI_SPACE], default=MNI_SPACE,
                        help='Physical cortical-mesh space; default HCP meshes are MNI. Choose T1w for native fMRIPrep meshes.')
    parser.add_argument(
        "--dtseries-template",
        default=DEFAULT_DTSERIES_TEMPLATE,
        help=(
            "Per-subject dtseries template. Variables: {data_root}, {subject}, "
            "{sub_subject}. Default preserves the legacy HCP layout."
        ),
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    shared_root = resolve_shared_root_for_mode(args.shared_root, ctxdrop=bool(args.ctxdrop))
    surface_root = Path(args.surface_root).resolve()
    reference_root = resolve_reference_root_for_mode(args.reference_root, ctxdrop=bool(args.ctxdrop), shared_root=shared_root)
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
                left_cortex_surface_template=str(args.left_cortex_surface_template),
                right_cortex_surface_template=str(args.right_cortex_surface_template),
                ctxdrop_distance_mm=float(args.ctxdrop_distance_mm),
                dtseries_template=str(args.dtseries_template),
                atlas=str(args.atlas),
                run_metadata_template=str(args.run_metadata_template) if args.run_metadata_template else None,
                cortex_coordinate_space=args.cortex_coordinate_space,
                dataset=args.dataset,
            )
            n_fc += int(str(obj["fc_combos"]))
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
                    str(args.left_cortex_surface_template),
                    str(args.right_cortex_surface_template),
                    float(args.ctxdrop_distance_mm),
                    str(args.dtseries_template),
                    str(args.atlas),
                    str(args.run_metadata_template) if args.run_metadata_template else None,
                    args.cortex_coordinate_space,
                    dataset=args.dataset,
                )
                for s in subjects
            ]
            for fut in as_completed(futs):
                obj = fut.result()
                n_fc += int(str(obj["fc_combos"]))
                print(f"[done] {obj['subject']}", flush=True)
    print(json.dumps({"subjects": len(subjects), "fc_combos": int(n_fc), "jobs": int(jobs), "status": "ok", "ctxdrop": bool(args.ctxdrop)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

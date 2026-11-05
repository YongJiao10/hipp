"""Shared ROI and network aggregation for the two Kong2022 label sources."""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import nibabel as nib
import numpy as np

from .extract_cortex_merged_network_timeseries import aggregate_hemi_roi_timeseries
from .io_utils import (
    is_persistent_concat_path,
    load_run_artifact_manifest,
    load_run_segment_metadata,
    parse_run_segments,
    write_tsv,
)
from .kong2022_label_utils import KONG2022_DLABEL_PATH, KONG2022_MERGE_CONFIG
from .output_layout import DEFAULT_CORTEX_ROUTEA_ROOT
from .path_templates import format_subject_path_template
from .workflow_utils import file_meta_unresolved as file_meta
from .workflow_utils import utc_now_iso


@dataclass(frozen=True)
class ReferenceRequest:
    """Inputs shared by the reference CLI and in-process subject workers."""
    subject: str
    shared_root: Path
    dtseries: Path | None = None
    run_dtseries: Sequence[Path] | None = None
    run_ids: Sequence[int] | None = None
    run_artifact_manifest: Path | None = None
    run_metadata_template: str | None = None
    kong2022_dlabel: Path = KONG2022_DLABEL_PATH
    cross_atlas_merge_json: Path = KONG2022_MERGE_CONFIG
    route_a_root: Path = DEFAULT_CORTEX_ROUTEA_ROOT
    atlas_slug: str = "kong2022"
    run_lengths: Sequence[int] | None = None
    task: str = "rest"
    dataset: str = "hcp"
    outdir: Path | None = None
    overwrite: bool = False


@dataclass(frozen=True)
class ReferenceTimeseries:
    kept_rows: list[dict[str, Any]]
    empty_rows: list[dict[str, Any]]
    left_empty: list[dict[str, Any]]
    right_empty: list[dict[str, Any]]
    left_stats: dict[str, int]
    right_stats: dict[str, int]
    canonical7_ts: np.ndarray
    family17_ts: np.ndarray


def materialize_reference_timeseries(
    *, outdir: Path, left_dt: np.ndarray, right_dt: np.ndarray,
    left_parcel_ids: np.ndarray, right_parcel_ids: np.ndarray,
    left_valid: np.ndarray, right_valid: np.ndarray,
    roi_registry: list[dict[str, Any]], canonical7_order: list[str], family17_order: list[str],
) -> ReferenceTimeseries:
    """Write fixed-order parcel rows and unweighted means of usable parcels.

    Empty parcels retain their identity as NaN rows. The label source and QC
    policy are selected by the caller; this function does not choose either.
    """
    left_ts, left_rows, left_empty, left_stats = aggregate_hemi_roi_timeseries(
        hemi="L", hemi_dt=left_dt, aligned_parcel_ids=left_parcel_ids,
        valid_mask=left_valid, roi_registry=roi_registry,
    )
    right_ts, right_rows, right_empty, right_stats = aggregate_hemi_roi_timeseries(
        hemi="R", hemi_dt=right_dt, aligned_parcel_ids=right_parcel_ids,
        valid_mask=right_valid, roi_registry=roi_registry,
    )
    kept_rows = left_rows + right_rows
    empty_rows = left_empty + right_empty
    if len(kept_rows) == len(empty_rows):
        raise RuntimeError("No Kong2022 ROI timeseries remained after signal validity filtering.")
    parcel_ts = np.asarray(left_ts + right_ts, dtype=np.float32)
    np.save(outdir / "cortex_roi_parcel_timeseries.npy", parcel_ts)
    roi_fields = [
        "parcel_id", "parcel_name", "family17", "canonical8", "canonical7",
        "hemisphere", "subregion", "rank", "n_grayordinates_total",
        "n_grayordinates_used", "excluded_by_validity",
    ]
    write_tsv(outdir / "cortex_roi_parcels.tsv", kept_rows, roi_fields)
    write_tsv(outdir / "cortex_roi_parcels_empty_after_validity.tsv", empty_rows, roi_fields)

    canonical7_rows = []
    canonical7_ts_rows = []
    for canonical7 in canonical7_order:
        idx = [i for i, row in enumerate(kept_rows)
               if str(row["canonical7"]) == canonical7 and int(str(row["n_grayordinates_used"])) > 0]
        if not idx:
            raise RuntimeError(
                f"Canonical-7 network has zero parcels after validity filtering: {canonical7}. "
                "Kong2022 downstream requires fixed canonical-7 outputs."
            )
        rows = [kept_rows[i] for i in idx]
        canonical7_ts_rows.append(np.nanmean(parcel_ts[np.asarray(idx, dtype=np.int32), :], axis=0).astype(np.float32))
        canonical7_rows.append({
            "canonical7_network": canonical7, "n_parcels_merged": int(len(idx)),
            "original_parent_networks": ",".join(sorted({str(row["family17"]) for row in rows})),
            "n_grayordinates_used_total": int(sum(int(str(row["n_grayordinates_used"])) for row in rows)),
        })
    canonical7_ts = np.asarray(canonical7_ts_rows, dtype=np.float32)
    np.save(outdir / "cortex_canonical7_network_timeseries.npy", canonical7_ts)
    write_tsv(outdir / "cortex_canonical7_networks.tsv", canonical7_rows,
              ["canonical7_network", "n_parcels_merged", "original_parent_networks", "n_grayordinates_used_total"])

    family17_rows = []
    family17_ts_rows = []
    for family17 in family17_order:
        idx = [i for i, row in enumerate(kept_rows)
               if str(row["family17"]) == family17 and int(str(row["n_grayordinates_used"])) > 0]
        if not idx:
            raise RuntimeError(
                f"Family17 network has zero parcels after validity filtering: {family17}. "
                "Kong2022 17-net branch requires fixed original-family outputs."
            )
        rows = [kept_rows[i] for i in idx]
        family17_ts_rows.append(np.nanmean(parcel_ts[np.asarray(idx, dtype=np.int32), :], axis=0).astype(np.float32))
        family17_rows.append({
            "family17_network": family17, "n_parcels_merged": int(len(idx)),
            "canonical_parent_networks": ",".join(sorted({str(row["canonical8"]) for row in rows})),
            "n_grayordinates_used_total": int(sum(int(str(row["n_grayordinates_used"])) for row in rows)),
        })
    family17_ts = np.asarray(family17_ts_rows, dtype=np.float32)
    np.save(outdir / "cortex_family17_network_timeseries.npy", family17_ts)
    write_tsv(outdir / "cortex_family17_networks.tsv", family17_rows,
              ["family17_network", "n_parcels_merged", "canonical_parent_networks", "n_grayordinates_used_total"])
    return ReferenceTimeseries(kept_rows, empty_rows, left_empty, right_empty,
                              left_stats, right_stats, canonical7_ts, family17_ts)


@dataclass(frozen=True)
class ResolvedReferenceInputs:
    dtseries_inputs: list[Path]
    dtseries_input_mode: str
    run_ids: tuple[int, ...]
    declared_run_lengths: tuple[int, ...] | None
    run_artifact_manifest_path: Path | None
    run_artifact_receipts: list[dict[str, object]]
    run_metadata_path: Path | None
    records: list[dict[str, Any]]


def resolve_reference_cifti_inputs(
    request: ReferenceRequest,
    *,
    subject: str,
    shared_root: Path,
    required_artifacts: Sequence[str] = ("dtseries",),
) -> ResolvedReferenceInputs:
    if request.dtseries and request.run_dtseries:
        raise ValueError("--dtseries and --run-dtseries are mutually exclusive")
    if request.run_dtseries:
        if request.run_artifact_manifest is None:
            raise ValueError("Run-level CIFTI input requires --run-artifact-manifest")
        manifest_path = Path(request.run_artifact_manifest).resolve()
        records: list[dict[str, Any]] = load_run_artifact_manifest(
            manifest_path, subject=subject, required_artifacts=required_artifacts,
        )
        declared_paths = [Path(r["artifacts"]["dtseries"]).resolve() for r in records]
        dtseries_inputs = [Path(p).resolve() for p in request.run_dtseries]
        if dtseries_inputs != declared_paths:
            raise ValueError("Run dtseries arguments disagree with the selected artifact manifest")
        run_ids = tuple(int(r["run_id"]) for r in records)
        if request.run_ids is not None and tuple(request.run_ids) != run_ids:
            raise ValueError("Explicit --run-ids disagree with the selected artifact manifest")
        receipts = [dict(r["artifact_receipts"]["dtseries"]) for r in records]
        mode, metadata_path, declared_lengths = "in_memory_run_concat", None, None
    else:
        if request.run_artifact_manifest is not None or request.run_ids is not None:
            raise ValueError("--run-artifact-manifest/--run-ids require --run-dtseries")
        if request.dtseries is None:
            raise ValueError("--dtseries is required unless --run-dtseries is selected")
        if request.run_metadata_template is None:
            raise ValueError("Single CIFTI input requires --run-metadata-template")
        metadata_path = Path(
            format_subject_path_template(
                request.run_metadata_template, subject=subject, shared_root=shared_root,
            )
        ).resolve()
        run_ids, declared_lengths = load_run_segment_metadata(metadata_path)
        if request.run_lengths is not None and tuple(request.run_lengths) != declared_lengths:
            raise ValueError(f"Explicit --run-lengths disagree with {metadata_path}")
        dtseries = Path(request.dtseries).resolve()
        if not dtseries.is_file():
            raise FileNotFoundError(f"Missing explicit dtseries input: {dtseries}")
        dtseries_inputs, mode, manifest_path, receipts, records = [dtseries], "single_concat", None, [], []
    if any(is_persistent_concat_path(path) for path in dtseries_inputs):
        raise ValueError(
            "Iowa persistent concat input is not allowed; pass --run-dtseries over "
            "canonical run-level CIFTIs so concatenation stays in memory"
        )
    return ResolvedReferenceInputs(
        dtseries_inputs, mode, run_ids, declared_lengths, manifest_path, receipts, metadata_path, records,
    )


def validate_cifti_run_lengths(
    *,
    dtseries_input_mode: str,
    declared_run_lengths: tuple[int, ...] | None,
    source_run_lengths: list[int],
    n_timepoints: int,
) -> list[int]:
    if dtseries_input_mode == "single_concat" and declared_run_lengths is not None:
        if sum(declared_run_lengths) != n_timepoints:
            raise ValueError(
                f"Run metadata lengths disagree with CIFTI frames: metadata={declared_run_lengths}, "
                f"frames={n_timepoints}"
            )
        return list(declared_run_lengths)
    if declared_run_lengths is not None and tuple(source_run_lengths) != declared_run_lengths:
        raise ValueError(
            f"Run metadata lengths disagree with CIFTI headers: metadata={declared_run_lengths}, "
            f"headers={tuple(source_run_lengths)}"
        )
    return source_run_lengths


def write_reference_manifests(
    *,
    outdir: Path,
    common_ref_dir: Path,
    subject: str,
    atlas_slug: str,
    reference_scope: str,
    dataset: str,
    task: str,
    validity_policy: str,
    resolved: ResolvedReferenceInputs,
    source_run_lengths: list[int],
    n_timepoints: int,
    reference: ReferenceTimeseries,
    canonical7_order: list[str],
    family17_order: list[str],
    kong2022_dlabel: Path,
    cross_atlas_merge_json: Path,
    left_valid_mask: Path,
    right_valid_mask: Path,
    reference_label_source: str | None = None,
    route_a_extra_summary: dict[str, object] | None = None,
    extra_input_files: Sequence[Path] = (),
    include_dlabel_in_stage_inputs: bool = False,
) -> None:
    run_segments: list[dict[str, Any]]
    if resolved.dtseries_input_mode == "in_memory_run_concat":
        run_segments = parse_run_segments(source_run_lengths, n_timepoints, run_ids=resolved.run_ids)
        run_boundary_source = "manifest_declared_run_inputs"
    else:
        if resolved.declared_run_lengths is None:
            raise RuntimeError("Single CIFTI run metadata was not loaded")
        run_segments = parse_run_segments(resolved.declared_run_lengths, n_timepoints, run_ids=resolved.run_ids)
        run_boundary_source = "explicit_run_metadata_template"

    n_used = int(len(reference.kept_rows) - len(reference.empty_rows))
    roi_src = str((outdir / "cortex_roi_parcel_timeseries.npy").resolve())
    c7_src = str((outdir / "cortex_canonical7_network_timeseries.npy").resolve())
    f17_src = str((outdir / "cortex_family17_network_timeseries.npy").resolve())

    for seg in run_segments:
        rid = int(seg["run_id"])
        run_start, run_stop, run_len = int(seg["run_start"]), int(seg["run_stop"]), int(seg["run_length"])
        run_dir = common_ref_dir / "runs" / f"run-{rid}"
        run_dir.mkdir(parents=True, exist_ok=True)
        if run_stop - run_start != run_len:
            raise RuntimeError(
                f"Run segment mismatch for sub-{subject} run-{rid}: start={run_start} stop={run_stop} length={run_len}"
            )
        run_summary: dict[str, object] = {
            "subject": subject,
            "reference_scope": reference_scope,
            "atlas_slug": atlas_slug,
            **({"reference_label_source": reference_label_source} if reference_label_source else {
                "dataset": dataset, "task": task, "validity_policy": validity_policy,
                "network_taxonomy": "canonical7", "reference_branch": "kong2022_canonical7_default",
            }),
            "run_id": str(rid),
            "run_label": str(seg["run_label"]),
            "run_boundary_source": run_boundary_source,
            "run_start": run_start,
            "run_stop": run_stop,
            "run_length": run_len,
            "n_timepoints": run_len,
            "n_roi_parcels_used": n_used,
            "n_canonical7_networks_used": int(reference.canonical7_ts.shape[0]),
            "canonical7_network_timeseries_source": c7_src,
            "n_family17_networks_used": int(reference.family17_ts.shape[0]),
            "family17_network_timeseries_source": f17_src,
            "roi_timeseries_source": roi_src,
            "validity_policy": validity_policy,
            "dataset": dataset,
            "run_slice_policy": "read_time_slice_from_full_timeseries",
        }
        (run_dir / "reference_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
        run_params: dict[str, object] = {
            "subject": subject, "atlas_slug": atlas_slug, "reference_scope": reference_scope, "run_id": str(rid),
        }
        if reference_label_source:
            run_params["reference_label_source"] = reference_label_source
        else:
            run_params.update(dataset=dataset, task=task, validity_policy=validity_policy)
        (run_dir / "stage_manifest.json").write_text(
            json.dumps({
                "stage": "reference_run_slice", "status": "done", "timestamp_utc": utc_now_iso(),
                "params": run_params,
                "inputs": [roi_src, c7_src, f17_src],
                "outputs": [str((run_dir / "reference_summary.json").resolve())],
            }, indent=2),
            encoding="utf-8",
        )

    left_stats, right_stats = reference.left_stats, reference.right_stats
    summary: dict[str, object] = {
        "subject": subject,
        "atlas_slug": atlas_slug,
        "dataset": dataset,
        "task": task,
        "validity_policy": validity_policy,
        "network_taxonomy": "canonical7",
        "reference_branch": "kong2022_canonical7_default",
        **({"reference_upstream_reuse": False, "reference_label_source": reference_label_source}
           if reference_label_source else {}),
        "reference_scope": reference_scope,
        "dtseries": str(resolved.dtseries_inputs[0]) if resolved.dtseries_input_mode == "single_concat" else None,
        "dtseries_input_mode": resolved.dtseries_input_mode,
        "run_artifact_manifest": str(resolved.run_artifact_manifest_path) if resolved.run_artifact_manifest_path else None,
        "run_artifact_receipts": resolved.run_artifact_receipts,
        "run_metadata": str(resolved.run_metadata_path) if resolved.run_metadata_path else None,
        "dtseries_inputs": [str(path) for path in resolved.dtseries_inputs],
        "source_run_lengths": source_run_lengths,
        "run_ids": [int(seg["run_id"]) for seg in run_segments],
        "run_lengths": [int(seg["run_length"]) for seg in run_segments],
        "kong2022_dlabel": str(kong2022_dlabel),
        "cross_atlas_merge_json": str(cross_atlas_merge_json),
        **(route_a_extra_summary or {}),
        "n_parcels_total": 400,
        "n_parcels_used": n_used,
        "n_parcels_empty": int(len(reference.empty_rows)),
        "n_canonical7_networks_used": int(reference.canonical7_ts.shape[0]),
        "canonical7_networks_used": canonical7_order,
        "canonical7_network_timeseries_source": c7_src,
        "n_family17_networks_used": int(reference.family17_ts.shape[0]),
        "family17_networks_used": family17_order,
        "n_timepoints": int(n_timepoints),
        "left_valid_mask": str(left_valid_mask),
        "right_valid_mask": str(right_valid_mask),
        "run_boundaries_dir": str((common_ref_dir / "runs").resolve()),
        "run_boundary_source": run_boundary_source,
        "validity_stats": {
            "left": left_stats,
            "right": right_stats,
            "combined": {
                "n_grayordinates_total": int(left_stats["n_grayordinates_total"] + right_stats["n_grayordinates_total"]),
                "n_grayordinates_used": int(left_stats["n_grayordinates_used"] + right_stats["n_grayordinates_used"]),
                "n_grayordinates_masked": int(left_stats["n_grayordinates_masked"] + right_stats["n_grayordinates_masked"]),
            },
            "rois_empty_after_validity": {
                "left": int(len(reference.left_empty)),
                "right": int(len(reference.right_empty)),
                "combined": int(len(reference.empty_rows)),
            },
        },
    }
    (outdir / "reference_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    inputs = [
        *(file_meta(path) for path in resolved.dtseries_inputs),
        *([file_meta(resolved.run_artifact_manifest_path)] if resolved.run_artifact_manifest_path else []),
        *([file_meta(resolved.run_metadata_path)] if resolved.run_metadata_path else []),
        *([file_meta(kong2022_dlabel)] if include_dlabel_in_stage_inputs else []),
        file_meta(cross_atlas_merge_json),
        *(file_meta(path) for path in extra_input_files),
        file_meta(left_valid_mask),
        file_meta(right_valid_mask),
    ]
    stage_manifest: dict[str, object] = {
        "stage": "reference",
        "status": "done",
        "timestamp_utc": utc_now_iso(),
        **({"network_taxonomy": "canonical7", "reference_branch": "kong2022_canonical7_default", "reference_upstream_reuse": False}
           if reference_label_source else {}),
        "params": {
            "subject": subject,
            "atlas_slug": atlas_slug,
            "dataset": dataset,
            "task": task,
            "validity_policy": validity_policy,
            **({"reference_scope": reference_scope, "reference_label_source": reference_label_source}
               if reference_label_source else {
                   "network_taxonomy": "canonical7", "reference_branch": "kong2022_canonical7_default",
                   "reference_scope": reference_scope,
               }),
            "run_boundary_source": run_boundary_source,
            "run_lengths": [int(seg["run_length"]) for seg in run_segments],
            "dtseries_input_mode": resolved.dtseries_input_mode,
            "run_artifact_manifest": str(resolved.run_artifact_manifest_path) if resolved.run_artifact_manifest_path else None,
            "run_artifact_receipts": resolved.run_artifact_receipts,
            "run_metadata": str(resolved.run_metadata_path) if resolved.run_metadata_path else None,
            "source_run_lengths": source_run_lengths,
        },
        "inputs": inputs,
        "outputs": [
            str((outdir / "reference_summary.json").resolve()),
            str((outdir / "cortex_roi_parcels.tsv").resolve()),
            roi_src,
            str((outdir / "cortex_canonical7_networks.tsv").resolve()),
            c7_src,
            str((outdir / "cortex_family17_networks.tsv").resolve()),
            f17_src,
        ],
    }
    (outdir / "stage_manifest.json").write_text(json.dumps(stage_manifest, indent=2), encoding="utf-8")
    print(
        json.dumps({
            "subject": subject,
            "atlas_slug": atlas_slug,
            **({"reference_label_source": reference_label_source} if reference_label_source else {
                "dataset": dataset, "task": task, "validity_policy": validity_policy,
            }),
            "n_roi_used": n_used,
            "n_timepoints": int(n_timepoints),
            "outdir": str(outdir),
        }, indent=2)
    )


def resolve_batch_subject_dtseries(
    *,
    subject: str,
    data_root: Path,
    dtseries_template: str,
    run_artifact_manifest_template: str | None,
    run_metadata_template: str | None,
    direct_manifest: Path | None = None,
    record_expected_lengths: bool = False,
) -> tuple[dict[str, object], dict[str, object]]:
    manifest_target = direct_manifest
    if manifest_target is None and run_artifact_manifest_template:
        manifest_target = Path(
            format_subject_path_template(
                run_artifact_manifest_template, data_root=data_root, subject=subject,
            )
        ).resolve()
    if manifest_target is not None:
        records: list[dict[str, Any]] = load_run_artifact_manifest(
            manifest_target, subject=subject, required_artifacts=("dtseries",),
        )
        dtseries_paths = [Path(r["artifacts"]["dtseries"]).resolve() for r in records]
        run_ids = [int(r["run_id"]) for r in records]
        expected: dict[str, object] = {
            "dtseries_input_mode": "in_memory_run_concat",
            "dtseries_inputs": [str(p) for p in dtseries_paths],
            "run_artifact_manifest": str(manifest_target.resolve()),
            "run_metadata": None,
            "run_ids": run_ids,
        }
        if record_expected_lengths:
            expected["run_lengths"] = [int(cast(Any, nib.load(str(p))).shape[0]) for p in dtseries_paths]
        return (
            {"run_dtseries": dtseries_paths, "run_ids": run_ids, "run_artifact_manifest": manifest_target},
            expected,
        )
    if run_metadata_template is None:
        raise ValueError("Single CIFTI input requires --run-metadata-template")
    dtseries = format_subject_path_template(dtseries_template, data_root=data_root, subject=subject)
    if is_persistent_concat_path(dtseries):
        raise ValueError(
            "Iowa persistent concat input is not allowed; pass "
            f"--run-artifact-manifest-template for canonical run-level files: {dtseries}"
        )
    metadata_path = Path(
        format_subject_path_template(run_metadata_template, data_root=data_root, subject=subject)
    ).resolve()
    selected_ids, selected_lengths = load_run_segment_metadata(metadata_path)
    expected = {
        "dtseries_input_mode": "single_concat",
        "dtseries_inputs": [str(Path(dtseries).resolve())],
        "run_artifact_manifest": None,
        "run_metadata": str(metadata_path),
        "run_ids": list(selected_ids),
        "run_lengths": list(selected_lengths),
    }
    return {"dtseries": Path(dtseries), "run_metadata_template": str(metadata_path)}, expected

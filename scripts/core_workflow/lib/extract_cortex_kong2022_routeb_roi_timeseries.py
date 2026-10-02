#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    from .extract_cortex_merged_network_timeseries import (
        aggregate_hemi_roi_timeseries,
        extract_structure_data,
        load_required_mask,
    )
except ImportError:  # pragma: no cover - supports direct script execution during local debugging.
    from extract_cortex_merged_network_timeseries import (
        aggregate_hemi_roi_timeseries,
        extract_structure_data,
        load_required_mask,
    )
from .io_utils import (
    compute_cortex_tsnr_gate,
    is_persistent_concat_path,
    load_cifti_dtseries_inputs,
    parse_run_segments,
    load_run_artifact_manifest,
    load_run_segment_metadata,
    write_tsv,
)
from .output_layout import DEFAULT_KONG2022_ROUTEB_ROOT
from .workflow_utils import file_meta_unresolved as file_meta, utc_now_iso, write_json
from .kong2022_label_utils import (
    KONG2022_DLABEL_PATH,
    KONG2022_MERGE_CONFIG,
    canonical_families_7_order,
    canonical_families_17_order,
    load_kong2022_bundle,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "hippunfold_input"
DEFAULT_SHARED_ROOT = DEFAULT_KONG2022_ROUTEB_ROOT
DEFAULT_RUN_IDS = [1, 2, 3, 4]
ATLAS_SLUG = "kong2022"
COMMON_REFERENCE_SLUG = "common"
TSNR_THRESHOLD = 25.0
HCP_MODE = 10000.0


def ensure_vertices_aligned(
    *,
    dt_vertices: np.ndarray,
    dlabel_vertices: np.ndarray,
    dlabel_parcel_ids: np.ndarray,
    hemi: str,
) -> np.ndarray:
    dt_v = np.asarray(dt_vertices, dtype=np.int32).reshape(-1)
    dl_v = np.asarray(dlabel_vertices, dtype=np.int32).reshape(-1)
    dl_pid = np.asarray(dlabel_parcel_ids, dtype=np.int32).reshape(-1)
    if dl_v.shape[0] != dl_pid.shape[0]:
        raise ValueError(f"dlabel vertex/parcel length mismatch hemi-{hemi}: {dl_v.shape} vs {dl_pid.shape}")
    if dt_v.shape[0] == dl_v.shape[0] and np.array_equal(dt_v, dl_v):
        return dl_pid.astype(np.int32, copy=False)
    mapping = {int(v): int(pid) for v, pid in zip(dl_v, dl_pid, strict=True)}
    aligned = np.asarray([mapping.get(int(v), -1) for v in dt_v], dtype=np.int32)
    if np.any(aligned < 0):
        missing = int(np.sum(aligned < 0))
        raise RuntimeError(
            f"Failed to align dlabel parcel ids to dtseries vertices hemi-{hemi}: missing={missing}"
        )
    return aligned



def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract cortex Kong2022 ROI timeseries from public dlabel and derive the default canonical-7 network timeseries."
    )
    parser.add_argument("--subject", required=True, help="Subject ID without sub- prefix.")
    parser.add_argument("--dtseries", default=None)
    parser.add_argument(
        "--run-dtseries",
        nargs="+",
        default=None,
        help="Run-level CIFTI dtseries files; concatenate in memory without writing a concat file.",
    )
    parser.add_argument("--run-ids", nargs="+", type=int, default=None)
    parser.add_argument("--run-artifact-manifest", default=None)
    parser.add_argument("--run-metadata-template", default=None)
    parser.add_argument("--kong2022-dlabel", default=str(KONG2022_DLABEL_PATH))
    parser.add_argument("--cross-atlas-merge-json", default=str(KONG2022_MERGE_CONFIG))
    parser.add_argument("--shared-root", default=str(DEFAULT_SHARED_ROOT))
    parser.add_argument("--left-tsnr-mask", default=None)
    parser.add_argument("--right-tsnr-mask", default=None)
    parser.add_argument("--run-lengths", nargs="*", type=int, default=None)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    subject = str(args.subject).replace("sub-", "")
    shared_root = Path(args.shared_root).resolve()
    run_ids: tuple[int, ...]
    run_artifact_manifest_path: Path | None = None
    run_artifact_receipts: list[dict[str, object]] = []
    run_metadata_path: Path | None = None
    declared_run_lengths: tuple[int, ...] | None = None
    if args.dtseries and args.run_dtseries:
        parser.error("--dtseries and --run-dtseries are mutually exclusive")
    if args.run_dtseries:
        if args.run_artifact_manifest is None:
            raise ValueError(
                "Run-level CIFTI input requires --run-artifact-manifest"
            )
        run_artifact_manifest_path = Path(args.run_artifact_manifest).resolve()
        records = load_run_artifact_manifest(
            run_artifact_manifest_path,
            subject=subject,
            required_artifacts=("dtseries",),
        )
        declared_paths = [
            Path(record["artifacts"]["dtseries"]).resolve() for record in records
        ]
        dtseries_inputs = [Path(path).resolve() for path in args.run_dtseries]
        if dtseries_inputs != declared_paths:
            raise ValueError(
                "Run dtseries arguments disagree with the selected artifact manifest"
            )
        run_ids = tuple(int(record["run_id"]) for record in records)
        if args.run_ids is not None and tuple(args.run_ids) != run_ids:
            raise ValueError(
                "Explicit --run-ids disagree with the selected artifact manifest"
            )
        run_artifact_receipts = [
            dict(record["artifact_receipts"]["dtseries"]) for record in records
        ]
        dtseries_input_mode = "in_memory_run_concat"
    else:
        if args.run_artifact_manifest is not None or args.run_ids is not None:
            parser.error("--run-artifact-manifest/--run-ids require --run-dtseries")
        if args.dtseries is None:
            parser.error("--dtseries is required unless --run-dtseries is selected")
        if args.run_metadata_template is None:
            raise ValueError(
                "Single CIFTI input requires --run-metadata-template"
            )
        run_metadata_path = Path(
            format_subject_path_template(
                args.run_metadata_template,
                subject=subject,
                shared_root=shared_root,
            )
        ).resolve()
        run_ids, declared_run_lengths = load_run_segment_metadata(run_metadata_path)
        if args.run_lengths is not None and tuple(args.run_lengths) != declared_run_lengths:
            raise ValueError(
                f"Explicit --run-lengths disagree with {run_metadata_path}"
            )
        dtseries = Path(args.dtseries).resolve()
        if not dtseries.is_file():
            raise FileNotFoundError(f"Missing explicit dtseries input: {dtseries}")
        dtseries_inputs = [dtseries]
        dtseries_input_mode = "single_concat"
    if any(is_persistent_concat_path(path) for path in dtseries_inputs):
        raise ValueError(
            "Iowa persistent concat input is not allowed; pass --run-dtseries over "
            "canonical run-level CIFTIs so concatenation stays in memory"
        )
    outdir = (
        Path(args.outdir).resolve()
        if args.outdir
        else (shared_root / f"sub-{subject}" / "reference" / ATLAS_SLUG).resolve()
    )
    common_ref_dir = (shared_root / f"sub-{subject}" / "reference" / COMMON_REFERENCE_SLUG).resolve()

    bundle = load_kong2022_bundle(
        dlabel_path=Path(args.kong2022_dlabel).resolve(),
        config_path=Path(args.cross_atlas_merge_json).resolve(),
    )
    roi_registry = list(bundle["roi_registry"])
    canonical7_order = list(bundle["canonical7_order"])
    family17_order = canonical_families_17_order()
    dlabel_vertices = bundle["vertex_arrays"]

    dt_data, dt_axis, source_run_lengths = load_cifti_dtseries_inputs(dtseries_inputs)
    dt_data_t = dt_data.T
    n_timepoints = int(dt_data.shape[0])
    if declared_run_lengths is not None and tuple(source_run_lengths) != declared_run_lengths:
        raise ValueError(
            f"Run metadata lengths disagree with CIFTI headers: metadata={declared_run_lengths}, "
            f"headers={tuple(source_run_lengths)}"
        )
    outdir.mkdir(parents=True, exist_ok=True)
    common_ref_dir.mkdir(parents=True, exist_ok=True)
    left_vertices, left_dt = extract_structure_data(dt_axis, dt_data_t, "CORTEX_LEFT")
    right_vertices, right_dt = extract_structure_data(dt_axis, dt_data_t, "CORTEX_RIGHT")
    tsnr_dir = common_ref_dir / "tsnr_gate"
    left_tsnr_mask = (
        Path(args.left_tsnr_mask).resolve()
        if args.left_tsnr_mask
        else compute_cortex_tsnr_gate(data=left_dt, outdir=tsnr_dir, hemi="left", threshold=TSNR_THRESHOLD).resolve()
    )
    right_tsnr_mask = (
        Path(args.right_tsnr_mask).resolve()
        if args.right_tsnr_mask
        else compute_cortex_tsnr_gate(data=right_dt, outdir=tsnr_dir, hemi="right", threshold=TSNR_THRESHOLD).resolve()
    )
    left_valid = load_required_mask(str(left_tsnr_mask), int(left_vertices.shape[0]), "left cortex tSNR")
    right_valid = load_required_mask(str(right_tsnr_mask), int(right_vertices.shape[0]), "right cortex tSNR")

    left_parcel_ids = ensure_vertices_aligned(
        dt_vertices=left_vertices,
        dlabel_vertices=dlabel_vertices["L"][0],
        dlabel_parcel_ids=dlabel_vertices["L"][1],
        hemi="L",
    )
    right_parcel_ids = ensure_vertices_aligned(
        dt_vertices=right_vertices,
        dlabel_vertices=dlabel_vertices["R"][0],
        dlabel_parcel_ids=dlabel_vertices["R"][1],
        hemi="R",
    )

    left_ts, left_rows, left_empty, left_stats = aggregate_hemi_roi_timeseries(
        hemi="L",
        hemi_dt=left_dt,
        aligned_parcel_ids=left_parcel_ids,
        valid_mask=left_valid,
        roi_registry=roi_registry,
    )
    right_ts, right_rows, right_empty, right_stats = aggregate_hemi_roi_timeseries(
        hemi="R",
        hemi_dt=right_dt,
        aligned_parcel_ids=right_parcel_ids,
        valid_mask=right_valid,
        roi_registry=roi_registry,
    )

    kept_rows = left_rows + right_rows
    empty_rows = left_empty + right_empty
    if not kept_rows:
        raise RuntimeError("No Kong2022 ROI timeseries remained after tSNR gate.")

    parcel_ts = np.asarray(left_ts + right_ts, dtype=np.float32)
    np.save(outdir / "cortex_roi_parcel_timeseries.npy", parcel_ts)

    roi_tsv_fields = [
        "parcel_id",
        "parcel_name",
        "family17",
        "canonical8",
        "canonical7",
        "hemisphere",
        "subregion",
        "rank",
        "n_grayordinates_total",
        "n_grayordinates_used",
        "excluded_by_tsnr_gate",
    ]
    write_tsv(outdir / "cortex_roi_parcels.tsv", kept_rows, roi_tsv_fields)
    write_tsv(outdir / "cortex_roi_parcels_empty_after_tsnr.tsv", empty_rows, roi_tsv_fields)

    canonical7_rows: list[dict[str, object]] = []
    canonical7_ts_rows: list[np.ndarray] = []
    for canonical7 in canonical7_order:
        idx = [i for i, row in enumerate(kept_rows) if str(row["canonical7"]) == canonical7]
        if not idx:
            raise RuntimeError(
                f"Canonical-7 network has zero parcels after tSNR gate: {canonical7}. "
                "Kong2022 downstream requires fixed canonical-7 outputs."
            )
        row_subset = [kept_rows[i] for i in idx]
        ts = np.nanmean(parcel_ts[np.asarray(idx, dtype=np.int32), :], axis=0).astype(np.float32)
        canonical7_ts_rows.append(ts)
        canonical7_rows.append(
            {
                "canonical7_network": canonical7,
                "n_parcels_merged": int(len(idx)),
                "original_parent_networks": ",".join(sorted({str(r["family17"]) for r in row_subset})),
                "n_grayordinates_used_total": int(sum(int(r["n_grayordinates_used"]) for r in row_subset)),
            }
        )
    canonical7_ts = np.asarray(canonical7_ts_rows, dtype=np.float32)
    np.save(outdir / "cortex_canonical7_network_timeseries.npy", canonical7_ts)
    write_tsv(
        outdir / "cortex_canonical7_networks.tsv",
        canonical7_rows,
        ["canonical7_network", "n_parcels_merged", "original_parent_networks", "n_grayordinates_used_total"],
    )

    family17_rows: list[dict[str, object]] = []
    family17_ts_rows: list[np.ndarray] = []
    for family17 in family17_order:
        idx = [i for i, row in enumerate(kept_rows) if str(row["family17"]) == family17]
        if not idx:
            raise RuntimeError(
                f"Family17 network has zero parcels after tSNR gate: {family17}. "
                "Kong2022 17-net branch requires fixed original-family outputs."
            )
        row_subset = [kept_rows[i] for i in idx]
        ts = np.nanmean(parcel_ts[np.asarray(idx, dtype=np.int32), :], axis=0).astype(np.float32)
        family17_ts_rows.append(ts)
        family17_rows.append(
            {
                "family17_network": family17,
                "n_parcels_merged": int(len(idx)),
                "canonical_parent_networks": ",".join(sorted({str(r["canonical8"]) for r in row_subset})),
                "n_grayordinates_used_total": int(sum(int(r["n_grayordinates_used"]) for r in row_subset)),
            }
        )
    family17_ts = np.asarray(family17_ts_rows, dtype=np.float32)
    np.save(outdir / "cortex_family17_network_timeseries.npy", family17_ts)
    write_tsv(
        outdir / "cortex_family17_networks.tsv",
        family17_rows,
        ["family17_network", "n_parcels_merged", "canonical_parent_networks", "n_grayordinates_used_total"],
    )

    if dtseries_input_mode == "in_memory_run_concat":
        run_segments = parse_run_segments(
            source_run_lengths,
            n_timepoints,
            run_ids=run_ids,
        )
        run_boundary_source = "manifest_declared_run_inputs"
    else:
        if declared_run_lengths is None:
            raise RuntimeError("Single CIFTI run metadata was not loaded")
        run_segments = parse_run_segments(
            declared_run_lengths,
            n_timepoints,
            run_ids=run_ids,
        )
        run_boundary_source = "explicit_run_metadata_template"
    for seg in run_segments:
        rid = int(seg["run_id"])
        run_start = int(seg["run_start"])
        run_stop = int(seg["run_stop"])
        run_dir = common_ref_dir / "runs" / f"run-{rid}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_len = int(seg["run_length"])
        if run_stop - run_start != run_len:
            raise RuntimeError(
                f"Run segment mismatch for sub-{subject} run-{rid}: "
                f"start={run_start} stop={run_stop} length={run_len}"
            )

        run_summary = {
            "subject": subject,
            "reference_scope": COMMON_REFERENCE_SLUG,
            "atlas_slug": ATLAS_SLUG,
            "network_taxonomy": "canonical7",
            "reference_branch": "kong2022_canonical7_default",
            "run_id": str(rid),
            "run_label": str(seg["run_label"]),
            "run_boundary_source": run_boundary_source,
            "run_start": run_start,
            "run_stop": run_stop,
            "run_length": run_len,
            "n_timepoints": run_len,
            "n_roi_parcels_used": int(parcel_ts.shape[0]),
            "n_canonical7_networks_used": int(canonical7_ts.shape[0]),
            "canonical7_network_timeseries_source": str((outdir / "cortex_canonical7_network_timeseries.npy").resolve()),
            "n_family17_networks_used": int(family17_ts.shape[0]),
            "family17_network_timeseries_source": str((outdir / "cortex_family17_network_timeseries.npy").resolve()),
            "roi_timeseries_source": str((outdir / "cortex_roi_parcel_timeseries.npy").resolve()),
            "tsnr_threshold": float(TSNR_THRESHOLD),
            "run_slice_policy": "read_time_slice_from_full_timeseries",
        }
        (run_dir / "reference_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
        (run_dir / "stage_manifest.json").write_text(
            json.dumps(
                {
                    "stage": "reference_run_slice",
                    "status": "done",
                    "timestamp_utc": utc_now_iso(),
                    "params": {
                        "subject": subject,
                        "atlas_slug": ATLAS_SLUG,
                        "reference_scope": COMMON_REFERENCE_SLUG,
                        "run_id": str(rid),
                    },
                    "inputs": [
                        str((outdir / "cortex_roi_parcel_timeseries.npy").resolve()),
                        str((outdir / "cortex_canonical7_network_timeseries.npy").resolve()),
                        str((outdir / "cortex_family17_network_timeseries.npy").resolve()),
                    ],
                    "outputs": [
                        str((run_dir / "reference_summary.json").resolve()),
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    summary = {
        "subject": subject,
        "atlas_slug": ATLAS_SLUG,
        "network_taxonomy": "canonical7",
        "reference_branch": "kong2022_canonical7_default",
        "reference_scope": COMMON_REFERENCE_SLUG,
        "dtseries": str(dtseries_inputs[0]) if dtseries_input_mode == "single_concat" else None,
        "dtseries_input_mode": dtseries_input_mode,
        "run_artifact_manifest": str(run_artifact_manifest_path) if run_artifact_manifest_path is not None else None,
        "run_artifact_receipts": run_artifact_receipts,
        "run_metadata": str(run_metadata_path) if run_metadata_path is not None else None,
        "dtseries_inputs": [str(path) for path in dtseries_inputs],
        "source_run_lengths": source_run_lengths,
        "run_ids": [int(seg["run_id"]) for seg in run_segments],
        "run_lengths": [int(seg["run_length"]) for seg in run_segments],
        "kong2022_dlabel": str(Path(args.kong2022_dlabel).resolve()),
        "cross_atlas_merge_json": str(Path(args.cross_atlas_merge_json).resolve()),
        "n_parcels_total": 400,
        "n_parcels_used_after_tsnr": int(parcel_ts.shape[0]),
        "n_parcels_empty_after_tsnr": int(len(empty_rows)),
        "n_canonical7_networks_used": int(canonical7_ts.shape[0]),
        "canonical7_networks_used": canonical7_order,
        "canonical7_network_timeseries_source": str((outdir / "cortex_canonical7_network_timeseries.npy").resolve()),
        "n_family17_networks_used": int(family17_ts.shape[0]),
        "family17_networks_used": family17_order,
        "n_timepoints": int(n_timepoints),
        "tsnr_threshold": float(TSNR_THRESHOLD),
        "left_tsnr_mask": str(left_tsnr_mask),
        "right_tsnr_mask": str(right_tsnr_mask),
        "run_boundaries_dir": str((common_ref_dir / "runs").resolve()),
        "run_boundary_source": run_boundary_source,
        "tsnr_gate_stats": {
            "left": left_stats,
            "right": right_stats,
            "combined": {
                "n_grayordinates_total": int(left_stats["n_grayordinates_total"] + right_stats["n_grayordinates_total"]),
                "n_grayordinates_used": int(left_stats["n_grayordinates_used"] + right_stats["n_grayordinates_used"]),
                "n_grayordinates_masked": int(left_stats["n_grayordinates_masked"] + right_stats["n_grayordinates_masked"]),
            },
            "rois_empty_after_tsnr_gate": {
                "left": int(len(left_empty)),
                "right": int(len(right_empty)),
                "combined": int(len(empty_rows)),
            },
        },
    }
    (outdir / "reference_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    inputs = [
        *(file_meta(path) for path in dtseries_inputs),
        *([file_meta(run_artifact_manifest_path)] if run_artifact_manifest_path is not None else []),
        *([file_meta(run_metadata_path)] if run_metadata_path is not None else []),
        file_meta(Path(args.kong2022_dlabel).resolve()),
        file_meta(Path(args.cross_atlas_merge_json).resolve()),
        file_meta(left_tsnr_mask),
        file_meta(right_tsnr_mask),
    ]
    stage_manifest = {
        "stage": "reference",
        "status": "done",
        "timestamp_utc": utc_now_iso(),
        "params": {
            "subject": subject,
            "atlas_slug": ATLAS_SLUG,
            "network_taxonomy": "canonical7",
            "reference_branch": "kong2022_canonical7_default",
            "reference_scope": COMMON_REFERENCE_SLUG,
            "tsnr_threshold": float(TSNR_THRESHOLD),
            "run_boundary_source": run_boundary_source,
            "run_lengths": [int(seg["run_length"]) for seg in run_segments],
            "dtseries_input_mode": dtseries_input_mode,
            "run_artifact_manifest": str(run_artifact_manifest_path) if run_artifact_manifest_path is not None else None,
            "run_artifact_receipts": run_artifact_receipts,
            "run_metadata": str(run_metadata_path) if run_metadata_path is not None else None,
            "source_run_lengths": source_run_lengths,
        },
        "inputs": inputs,
        "outputs": [
            str((outdir / "reference_summary.json").resolve()),
            str((outdir / "cortex_roi_parcels.tsv").resolve()),
            str((outdir / "cortex_roi_parcel_timeseries.npy").resolve()),
            str((outdir / "cortex_canonical7_networks.tsv").resolve()),
            str((outdir / "cortex_canonical7_network_timeseries.npy").resolve()),
            str((outdir / "cortex_family17_networks.tsv").resolve()),
            str((outdir / "cortex_family17_network_timeseries.npy").resolve()),
        ],
    }
    (outdir / "stage_manifest.json").write_text(json.dumps(stage_manifest, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "subject": subject,
                "atlas_slug": ATLAS_SLUG,
                "n_roi_used": int(parcel_ts.shape[0]),
                "n_timepoints": int(n_timepoints),
                "outdir": str(outdir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np

try:
    from .extract_cortex_merged_network_timeseries import extract_structure_data, load_required_mask
except ImportError:  # pragma: no cover - supports direct script execution during local debugging.
    from extract_cortex_merged_network_timeseries import extract_structure_data, load_required_mask
from .output_layout import DEFAULT_CORTEX_ROUTEA_ROOT, DEFAULT_KONG2022_ROUTEA_ROOT
from .workflow_utils import file_meta_unresolved as file_meta, utc_now_iso, write_json
from .kong2022_label_utils import (
    KONG2022_DLABEL_PATH,
    KONG2022_MERGE_CONFIG,
    canonical_families_8_order,
    load_kong2022_bundle,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "hippunfold_input"
DEFAULT_SHARED_ROOT = DEFAULT_KONG2022_ROUTEA_ROOT
DEFAULT_ROUTE_A_ROOT = DEFAULT_CORTEX_ROUTEA_ROOT
DEFAULT_RUN_IDS = [1, 2, 3, 4]
ATLAS_SLUG = "kong2022"
COMMON_REFERENCE_SLUG = "common"
TSNR_THRESHOLD = 25.0
HCP_MODE = 10000.0
ROUTE_A_SOURCE_TAG = "route_a_kong2022_areal_mshbm"


def write_tsv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def compute_cortex_tsnr_gate(
    *,
    data: np.ndarray,
    outdir: Path,
    hemi: str,
    threshold: float,
) -> Path:
    sd = np.nanstd(np.asarray(data, dtype=np.float64), axis=1, ddof=1)
    tsnr = np.where(sd > 0, HCP_MODE / sd, np.nan).astype(np.float32)
    valid = np.isfinite(tsnr) & (tsnr >= float(threshold))
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / f"cortex_{hemi.lower()}_tsnr.npy", tsnr)
    mask_path = outdir / f"cortex_{hemi.lower()}_valid_mask.npy"
    np.save(mask_path, valid.astype(bool))
    write_json(
        outdir / f"cortex_{hemi.lower()}_tsnr_gate_summary.json",
        {
            "hemi": hemi,
            "threshold": float(threshold),
            "tsnr_definition": "10000/std",
            "n_grayordinates_total": int(valid.size),
            "n_grayordinates_valid": int(valid.sum()),
            "n_grayordinates_invalid": int(valid.size - valid.sum()),
            "invalid_fraction": float(1.0 - (valid.sum() / max(1, valid.size))),
        },
    )
    return mask_path


def parse_run_segments(run_lengths: list[int] | None, expected_total_tp: int) -> list[dict[str, object]]:
    if run_lengths is None:
        if int(expected_total_tp) % len(DEFAULT_RUN_IDS) != 0:
            raise RuntimeError(
                f"Cannot infer equal run lengths: n_timepoints={expected_total_tp} n_runs={len(DEFAULT_RUN_IDS)}"
            )
        inferred = int(expected_total_tp) // len(DEFAULT_RUN_IDS)
        run_lengths = [inferred for _ in DEFAULT_RUN_IDS]
    if len(run_lengths) != len(DEFAULT_RUN_IDS):
        raise ValueError(f"Expected {len(DEFAULT_RUN_IDS)} run lengths, got {run_lengths}")
    segments: list[dict[str, object]] = []
    offset = 0
    for rid, run_len in zip(DEFAULT_RUN_IDS, run_lengths, strict=True):
        run_len = int(run_len)
        if run_len <= 0:
            raise ValueError(f"Invalid run length for run-{rid}: {run_len}")
        run_start = int(offset)
        run_stop = int(offset + run_len)
        segments.append(
            {
                "run_id": int(rid),
                "run_label": f"run-{rid}",
                "run_start": int(run_start),
                "run_stop": int(run_stop),
                "run_length": int(run_len),
            }
        )
        offset = run_stop
    if offset != int(expected_total_tp):
        raise RuntimeError(
            f"Run lengths do not sum to concat length: sum={offset} concat={expected_total_tp}"
        )
    return segments


def discover_route_a_outputs_dir(route_a_root: Path, subject: str, explicit: Path | None) -> Path:
    if explicit is not None:
        out_dir = explicit.resolve()
    else:
        pattern = f"sub-{subject}/kong2022_areal_mshbm*/project_dir/route_a_outputs"
        candidates = [p for p in route_a_root.glob(pattern) if p.is_dir()]
        if not candidates:
            raise FileNotFoundError(
                f"No Route A route_a_outputs found for sub-{subject} under {route_a_root}"
            )
        candidates.sort(key=lambda p: p.stat().st_mtime_ns, reverse=True)
        out_dir = candidates[0].resolve()
    lh = out_dir / "lh_labels.txt"
    rh = out_dir / "rh_labels.txt"
    if not lh.exists() or not rh.exists():
        raise FileNotFoundError(f"Missing lh/rh label files in {out_dir}")
    return out_dir


def load_surface_label_vector(path: Path, expected_len: int = 32492) -> np.ndarray:
    arr = np.loadtxt(str(path), dtype=np.int32)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if int(arr.shape[0]) != int(expected_len):
        raise ValueError(f"Expected {expected_len} labels in {path}, got {arr.shape[0]}")
    return arr.astype(np.int32, copy=False)


def align_surface_labels_to_dt_vertices(full_surface_labels: np.ndarray, dt_vertices: np.ndarray, hemi: str) -> np.ndarray:
    v = np.asarray(dt_vertices, dtype=np.int32).reshape(-1)
    if v.size == 0:
        raise RuntimeError(f"No dtseries vertices for hemi-{hemi}")
    v_max = int(np.max(v))
    if v_max >= int(full_surface_labels.shape[0]):
        raise ValueError(
            f"Vertex index out of bounds for hemi-{hemi}: max_vertex={v_max} label_len={full_surface_labels.shape[0]}"
        )
    return full_surface_labels[v].astype(np.int32, copy=False)


def aggregate_hemi_roi_timeseries(
    *,
    hemi: str,
    hemi_dt: np.ndarray,
    aligned_parcel_ids: np.ndarray,
    valid_mask: np.ndarray,
    roi_registry: list[dict[str, object]],
) -> tuple[list[np.ndarray], list[dict[str, object]], list[dict[str, object]], dict[str, int]]:
    n_total = int(hemi_dt.shape[0])
    n_used = int(np.sum(valid_mask))
    stats = {
        "n_grayordinates_total": n_total,
        "n_grayordinates_used": n_used,
        "n_grayordinates_masked": int(n_total - n_used),
    }

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
        int(pid): grp for pid, grp in zip(unique_valid.tolist(), np.split(sorted_idx, split_points))
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
            "hemisphere": hemi,
            "subregion": str(row["subregion"]),
            "rank": int(row["rank"]),
            "n_grayordinates_total": n_g_total,
            "n_grayordinates_used": n_g_used,
            "excluded_by_tsnr_gate": bool(n_g_used <= 0),
        }
        if n_g_used <= 0:
            empty_rows.append(out_row)
            continue
        ts = np.nanmean(hemi_dt[valid_idxs, :], axis=0).astype(np.float32)
        ts_rows.append(ts)
        kept_rows.append(out_row)

    return ts_rows, kept_rows, empty_rows, stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bridge Route A individualized labels to Route B-compatible reference/kong2022 arrays."
    )
    parser.add_argument("--subject", required=True, help="Subject ID without sub- prefix.")
    parser.add_argument("--dtseries", default=None)
    parser.add_argument("--kong2022-dlabel", default=str(KONG2022_DLABEL_PATH))
    parser.add_argument("--cross-atlas-merge-json", default=str(KONG2022_MERGE_CONFIG))
    parser.add_argument("--shared-root", default=str(DEFAULT_SHARED_ROOT))
    parser.add_argument("--route-a-root", default=str(DEFAULT_ROUTE_A_ROOT))
    parser.add_argument("--route-a-label-dir", default=None)
    parser.add_argument("--left-tsnr-mask", default=None)
    parser.add_argument("--right-tsnr-mask", default=None)
    parser.add_argument("--run-lengths", nargs="*", type=int, default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()

    subject = str(args.subject).replace("sub-", "")
    shared_root = Path(args.shared_root).resolve()
    route_a_root = Path(args.route_a_root).resolve()
    route_a_label_dir = Path(args.route_a_label_dir).resolve() if args.route_a_label_dir else None
    dtseries = (
        Path(args.dtseries).resolve()
        if args.dtseries
        else (DEFAULT_DATA_ROOT / f"sub-{subject}" / "func" / f"sub-{subject}_task-rest_run-concat.dtseries.nii").resolve()
    )
    outdir = (
        Path(args.outdir).resolve()
        if args.outdir
        else (shared_root / f"sub-{subject}" / "reference" / ATLAS_SLUG).resolve()
    )
    outdir.mkdir(parents=True, exist_ok=True)
    common_ref_dir = (shared_root / f"sub-{subject}" / "reference" / COMMON_REFERENCE_SLUG).resolve()
    common_ref_dir.mkdir(parents=True, exist_ok=True)

    existing_summary = outdir / "reference_summary.json"
    if existing_summary.exists() and not args.overwrite:
        existing = json.loads(existing_summary.read_text(encoding="utf-8"))
        if str(existing.get("reference_label_source", "")) != ROUTE_A_SOURCE_TAG:
            raise RuntimeError(
                f"Refusing to overwrite non-Route-A reference at {existing_summary}. "
                "Use a different --shared-root or pass --overwrite."
            )

    route_out_dir = discover_route_a_outputs_dir(route_a_root=route_a_root, subject=subject, explicit=route_a_label_dir)
    lh_labels_path = route_out_dir / "lh_labels.txt"
    rh_labels_path = route_out_dir / "rh_labels.txt"
    lh_full = load_surface_label_vector(lh_labels_path)
    rh_full = load_surface_label_vector(rh_labels_path)

    bundle = load_kong2022_bundle(
        dlabel_path=Path(args.kong2022_dlabel).resolve(),
        config_path=Path(args.cross_atlas_merge_json).resolve(),
    )
    roi_registry = list(bundle["roi_registry"])
    canonical8_order = list(bundle["canonical8_order"])

    dt_img = nib.load(str(dtseries))
    dt_data = np.asarray(dt_img.dataobj, dtype=np.float32)
    if dt_data.ndim != 2:
        raise ValueError(f"Expected 2D dtseries data, got shape={dt_data.shape}")
    dt_data_t = dt_data.T
    n_timepoints = int(dt_data.shape[0])
    dt_axis = dt_img.header.get_axis(1)

    left_vertices, left_dt = extract_structure_data(dt_axis, dt_data_t, "CORTEX_LEFT")
    right_vertices, right_dt = extract_structure_data(dt_axis, dt_data_t, "CORTEX_RIGHT")
    left_parcel_ids = align_surface_labels_to_dt_vertices(lh_full, left_vertices, hemi="L")
    right_parcel_ids = align_surface_labels_to_dt_vertices(rh_full, right_vertices, hemi="R")

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
        "hemisphere",
        "subregion",
        "rank",
        "n_grayordinates_total",
        "n_grayordinates_used",
        "excluded_by_tsnr_gate",
    ]
    write_tsv(outdir / "cortex_roi_parcels.tsv", kept_rows, roi_tsv_fields)
    write_tsv(outdir / "cortex_roi_parcels_empty_after_tsnr.tsv", empty_rows, roi_tsv_fields)

    canonical_rows: list[dict[str, object]] = []
    canonical_ts_rows: list[np.ndarray] = []
    for canonical in canonical8_order:
        idx = [i for i, row in enumerate(kept_rows) if str(row["canonical8"]) == canonical]
        if not idx:
            raise RuntimeError(
                f"Canonical network has zero parcels after tSNR gate: {canonical}. "
                "Kong2022 downstream requires fixed canonical-8 outputs."
            )
        row_subset = [kept_rows[i] for i in idx]
        ts = np.nanmean(parcel_ts[np.asarray(idx, dtype=np.int32), :], axis=0).astype(np.float32)
        canonical_ts_rows.append(ts)
        canonical_rows.append(
            {
                "canonical_network": canonical,
                "n_parcels_merged": int(len(idx)),
                "original_parent_networks": ",".join(sorted({str(r["family17"]) for r in row_subset})),
                "n_grayordinates_used_total": int(sum(int(r["n_grayordinates_used"]) for r in row_subset)),
            }
        )
    canonical_ts = np.asarray(canonical_ts_rows, dtype=np.float32)
    np.save(outdir / "cortex_canonical_network_timeseries.npy", canonical_ts)
    write_tsv(
        outdir / "cortex_canonical_networks.tsv",
        canonical_rows,
        ["canonical_network", "n_parcels_merged", "original_parent_networks", "n_grayordinates_used_total"],
    )

    run_segments = parse_run_segments(args.run_lengths, n_timepoints)
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
            "reference_label_source": ROUTE_A_SOURCE_TAG,
            "run_id": str(rid),
            "run_label": str(seg["run_label"]),
            "run_boundary_source": "reference_common_explicit_run_lengths_or_equal_4way_default",
            "run_start": run_start,
            "run_stop": run_stop,
            "run_length": run_len,
            "n_timepoints": run_len,
            "n_roi_parcels_used": int(parcel_ts.shape[0]),
            "n_canonical_networks_used": int(canonical_ts.shape[0]),
            "canonical_network_timeseries_source": str((outdir / "cortex_canonical_network_timeseries.npy").resolve()),
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
                        "reference_label_source": ROUTE_A_SOURCE_TAG,
                        "run_id": str(rid),
                    },
                    "inputs": [
                        str((outdir / "cortex_roi_parcel_timeseries.npy").resolve()),
                        str((outdir / "cortex_canonical_network_timeseries.npy").resolve()),
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
        "reference_scope": COMMON_REFERENCE_SLUG,
        "reference_label_source": ROUTE_A_SOURCE_TAG,
        "dtseries": str(dtseries),
        "cross_atlas_merge_json": str(Path(args.cross_atlas_merge_json).resolve()),
        "route_a_root": str(route_a_root),
        "route_a_label_dir": str(route_out_dir),
        "route_a_lh_labels": str(lh_labels_path),
        "route_a_rh_labels": str(rh_labels_path),
        "n_parcels_total": 400,
        "n_parcels_used_after_tsnr": int(parcel_ts.shape[0]),
        "n_parcels_empty_after_tsnr": int(len(empty_rows)),
        "n_canonical_networks_used": int(canonical_ts.shape[0]),
        "canonical_networks_used": canonical8_order,
        "n_timepoints": int(n_timepoints),
        "tsnr_threshold": float(TSNR_THRESHOLD),
        "left_tsnr_mask": str(left_tsnr_mask),
        "right_tsnr_mask": str(right_tsnr_mask),
        "run_boundaries_dir": str((common_ref_dir / "runs").resolve()),
        "run_boundary_source": "reference_common_explicit_run_lengths_or_equal_4way_default",
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
        file_meta(dtseries),
        file_meta(Path(args.cross_atlas_merge_json).resolve()),
        file_meta(lh_labels_path),
        file_meta(rh_labels_path),
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
            "reference_scope": COMMON_REFERENCE_SLUG,
            "reference_label_source": ROUTE_A_SOURCE_TAG,
            "tsnr_threshold": float(TSNR_THRESHOLD),
            "run_boundary_source": "reference_common_explicit_run_lengths_or_equal_4way_default",
            "run_lengths": [int(seg["run_length"]) for seg in run_segments],
        },
        "inputs": inputs,
        "outputs": [
            str((outdir / "reference_summary.json").resolve()),
            str((outdir / "cortex_roi_parcels.tsv").resolve()),
            str((outdir / "cortex_roi_parcel_timeseries.npy").resolve()),
            str((outdir / "cortex_canonical_networks.tsv").resolve()),
            str((outdir / "cortex_canonical_network_timeseries.npy").resolve()),
        ],
    }
    (outdir / "stage_manifest.json").write_text(json.dumps(stage_manifest, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "subject": subject,
                "atlas_slug": ATLAS_SLUG,
                "reference_label_source": ROUTE_A_SOURCE_TAG,
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

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np


ROI_NAME_PATTERN = re.compile(r"(?P<network>.+)_(?P<hemi>[LR])_(?P<rank>\d+)$")
REPO_ROOT = Path(__file__).resolve().parents[2]
CROSS_ATLAS_NETWORK_MERGE_JSON = REPO_ROOT / "config" / "cross_atlas_network_merge.json"


def load_label_gifti(path: Path) -> tuple[np.ndarray, dict[int, str]]:
    img = nib.load(str(path))
    labels = np.asarray(img.darrays[0].data, dtype=np.int32)
    label_map: dict[int, str] = {}
    for label in img.labeltable.labels:
        key = int(label.key)
        if key == 0:
            continue
        label_map[key] = getattr(label, "label", None) or str(key)
    return labels, label_map


def parse_roi_name(name: str) -> tuple[str, str, int]:
    match = ROI_NAME_PATTERN.fullmatch(name)
    if not match:
        raise ValueError(f"Could not parse ROI component name: {name}")
    return match.group("network"), match.group("hemi"), int(match.group("rank"))


def load_roi_summary(path: Path) -> dict[str, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "raw_component_count": int(data["raw_component_count"]),
        "kept_roi_count": int(data["kept_roi_count"]),
    }


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


def load_optional_mask(path: str | None, expected_length: int, label: str) -> np.ndarray | None:
    if not path:
        return None
    mask = np.load(path).astype(bool)
    if mask.ndim != 1 or mask.shape[0] != expected_length:
        raise ValueError(
            f"{label} mask shape mismatch: expected ({expected_length},), got {mask.shape}"
        )
    return mask


def align_surface_labels(
    full_surface_labels: np.ndarray,
    full_surface_map: dict[int, str],
    dt_vertices: np.ndarray,
) -> tuple[np.ndarray, dict[int, str]]:
    if dt_vertices.max(initial=-1) >= full_surface_labels.shape[0]:
        raise ValueError(
            f"Vertex index out of bounds: max dtseries vertex={dt_vertices.max()}, "
            f"label length={full_surface_labels.shape[0]}"
        )
    aligned = full_surface_labels[dt_vertices]
    used_keys = sorted(int(x) for x in np.unique(aligned) if int(x) > 0)
    used_map = {key: full_surface_map[key] for key in used_keys}
    return aligned, used_map


def mean_timeseries_by_label(
    dt_data: np.ndarray,
    aligned_labels: np.ndarray,
    label_map: dict[int, str],
    hemisphere: str,
    valid_mask: np.ndarray | None = None,
) -> tuple[list[np.ndarray], list[dict[str, object]], list[dict[str, object]]]:
    ts_rows: list[np.ndarray] = []
    meta_rows: list[dict[str, object]] = []
    empty_rows: list[dict[str, object]] = []
    for key in sorted(label_map):
        mask = aligned_labels == key
        roi_name = label_map[key]
        parent_network, hemi_name, rank = parse_roi_name(roi_name)
        if hemi_name != hemisphere:
            raise ValueError(f"ROI name hemisphere mismatch for {roi_name}: expected {hemisphere}")
        grayordinate_mask = mask if valid_mask is None else (mask & valid_mask)
        n_grayordinates_total = int(mask.sum())
        n_grayordinates_used = int(grayordinate_mask.sum())
        if n_grayordinates_used <= 0:
            empty_rows.append(
                {
                    "parcel_id": int(key),
                    "parcel_name": roi_name,
                    "parent_network": parent_network,
                    "hemisphere": hemisphere,
                    "component_rank": int(rank),
                    "n_grayordinates_total": n_grayordinates_total,
                    "n_grayordinates_used": 0,
                    "excluded_noise": bool(parent_network == "Noise"),
                    "excluded_by_tsnr_gate": True,
                }
            )
            continue
        ts = np.nanmean(dt_data[grayordinate_mask, :], axis=0)
        ts_rows.append(ts.astype(np.float32, copy=False))
        meta_rows.append(
            {
                "parcel_id": int(key),
                "parcel_name": roi_name,
                "parent_network": parent_network,
                "hemisphere": hemisphere,
                "component_rank": int(rank),
                "n_grayordinates_total": n_grayordinates_total,
                "n_grayordinates_used": n_grayordinates_used,
                "excluded_noise": bool(parent_network == "Noise"),
                "excluded_by_tsnr_gate": False,
            }
        )
    return ts_rows, meta_rows, empty_rows


def write_tsv(rows: list[dict[str, object]], path: Path, fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def load_cross_atlas_network_merge(path: Path) -> tuple[list[str], set[str], dict[str, dict[str, str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    canonical_order = [str(item) for item in payload["canonical_network_order"]]
    exclude_labels = {str(item) for item in payload.get("exclude_labels", [])}
    atlas_mapping = {
        str(atlas_slug): {str(key): str(value) for key, value in atlas_spec["mapping"].items()}
        for atlas_slug, atlas_spec in payload["atlases"].items()
    }
    return canonical_order, exclude_labels, atlas_mapping


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract individualized cortex ROI component timeseries from a subject dtseries"
    )
    parser.add_argument("--subject", required=True)
    parser.add_argument("--dtseries", required=True)
    parser.add_argument("--left-labels", required=True)
    parser.add_argument("--right-labels", required=True)
    parser.add_argument("--roi-summary", required=True, help="roi_component_stats.json")
    parser.add_argument("--atlas-slug", required=True)
    parser.add_argument("--left-tsnr-mask", default=None)
    parser.add_argument("--right-tsnr-mask", default=None)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    roi_summary = load_roi_summary(Path(args.roi_summary))
    dt_img = nib.load(args.dtseries)
    dt_data = np.asarray(dt_img.dataobj, dtype=np.float32)
    if dt_data.ndim != 2:
        raise ValueError(f"Expected 2D dtseries data, got shape {dt_data.shape}")
    dt_data_t = dt_data.T
    dt_axis = dt_img.header.get_axis(1)

    left_all_labels, left_label_map = load_label_gifti(Path(args.left_labels))
    right_all_labels, right_label_map = load_label_gifti(Path(args.right_labels))

    left_vertices, left_dt = extract_structure_data(dt_axis, dt_data_t, "CORTEX_LEFT")
    right_vertices, right_dt = extract_structure_data(dt_axis, dt_data_t, "CORTEX_RIGHT")

    left_valid_mask = load_optional_mask(args.left_tsnr_mask, int(left_vertices.shape[0]), "left cortex tSNR")
    right_valid_mask = load_optional_mask(args.right_tsnr_mask, int(right_vertices.shape[0]), "right cortex tSNR")

    left_aligned, left_used_map = align_surface_labels(left_all_labels, left_label_map, left_vertices)
    right_aligned, right_used_map = align_surface_labels(right_all_labels, right_label_map, right_vertices)

    left_ts, left_rows, left_empty_rows = mean_timeseries_by_label(
        left_dt, left_aligned, left_used_map, "L", valid_mask=left_valid_mask
    )
    right_ts, right_rows, right_empty_rows = mean_timeseries_by_label(
        right_dt, right_aligned, right_used_map, "R", valid_mask=right_valid_mask
    )

    all_rows = left_rows + right_rows
    if not all_rows:
        raise RuntimeError("No cortex ROI parcel timeseries could be extracted")

    non_noise_rows = [row for row in all_rows if not row["excluded_noise"]]
    if not non_noise_rows:
        raise RuntimeError("All ROI components were excluded as Noise")

    ts_by_name = {
        row["parcel_name"]: ts
        for row, ts in zip(all_rows, left_ts + right_ts, strict=True)
    }
    ordered_non_noise_rows = sorted(
        non_noise_rows,
        key=lambda row: (str(row["hemisphere"]), str(row["parent_network"]), int(row["component_rank"])),
    )
    parcel_ts = np.asarray([ts_by_name[row["parcel_name"]] for row in ordered_non_noise_rows], dtype=np.float32)
    np.save(outdir / "cortex_roi_parcel_timeseries.npy", parcel_ts)

    parent_networks = sorted({str(row["parent_network"]) for row in ordered_non_noise_rows})
    parent_network_ts = []
    parent_network_rows = []
    for network in parent_networks:
        inds = [idx for idx, row in enumerate(ordered_non_noise_rows) if row["parent_network"] == network]
        ts = np.nanmean(parcel_ts[inds, :], axis=0)
        parent_network_ts.append(ts.astype(np.float32, copy=False))
        parent_network_rows.append(
            {
                "parent_network": network,
                "n_parcels": int(len(inds)),
            }
        )
    parent_network_ts_arr = np.asarray(parent_network_ts, dtype=np.float32)
    np.save(outdir / "cortex_parent_network_timeseries.npy", parent_network_ts_arr)

    canonical_order, exclude_labels, atlas_mapping = load_cross_atlas_network_merge(CROSS_ATLAS_NETWORK_MERGE_JSON)
    if args.atlas_slug not in atlas_mapping:
        raise KeyError(f"Atlas slug not found in cross-atlas network merge config: {args.atlas_slug}")
    mapping = atlas_mapping[args.atlas_slug]

    missing = sorted(set(parent_networks) - set(mapping))
    if missing:
        raise KeyError(f"Missing canonical merge mapping for atlas {args.atlas_slug}: {missing}")

    canonical_network_rows = []
    canonical_network_ts = []
    for canonical in canonical_order:
        matched_rows = [row for row in ordered_non_noise_rows if mapping[str(row["parent_network"])] == canonical]
        if canonical in exclude_labels or not matched_rows:
            continue
        inds = [idx for idx, row in enumerate(ordered_non_noise_rows) if mapping[str(row["parent_network"])] == canonical]
        ts = np.nanmean(parcel_ts[inds, :], axis=0)
        original_labels = sorted({str(row["parent_network"]) for row in matched_rows})
        canonical_network_ts.append(ts.astype(np.float32, copy=False))
        canonical_network_rows.append(
            {
                "canonical_network": canonical,
                "n_parcels_merged": int(len(inds)),
                "original_parent_networks": ",".join(original_labels),
            }
        )
    canonical_network_ts_arr = np.asarray(canonical_network_ts, dtype=np.float32)
    np.save(outdir / "cortex_canonical_network_timeseries.npy", canonical_network_ts_arr)

    write_tsv(
        ordered_non_noise_rows,
        outdir / "cortex_roi_parcels.tsv",
        [
            "parcel_id",
            "parcel_name",
            "parent_network",
            "hemisphere",
            "component_rank",
            "n_grayordinates_total",
            "n_grayordinates_used",
            "excluded_noise",
            "excluded_by_tsnr_gate",
        ],
    )
    write_tsv(
        left_empty_rows + right_empty_rows,
        outdir / "cortex_roi_parcels_empty_after_tsnr.tsv",
        [
            "parcel_id",
            "parcel_name",
            "parent_network",
            "hemisphere",
            "component_rank",
            "n_grayordinates_total",
            "n_grayordinates_used",
            "excluded_noise",
            "excluded_by_tsnr_gate",
        ],
    )
    write_tsv(parent_network_rows, outdir / "cortex_parent_networks.tsv", ["parent_network", "n_parcels"])
    write_tsv(
        canonical_network_rows,
        outdir / "cortex_canonical_networks.tsv",
        ["canonical_network", "n_parcels_merged", "original_parent_networks"],
    )

    summary = {
        "subject": args.subject,
        "atlas_slug": args.atlas_slug,
        "dtseries": str(Path(args.dtseries).resolve()),
        "left_labels": str(Path(args.left_labels).resolve()),
        "right_labels": str(Path(args.right_labels).resolve()),
        "raw_component_count": int(roi_summary["raw_component_count"]),
        "kept_roi_count": int(roi_summary["kept_roi_count"]),
        "n_parcels_used_after_noise_exclusion": int(parcel_ts.shape[0]),
        "n_parent_networks_used": int(parent_network_ts_arr.shape[0]),
        "n_canonical_networks_used": int(canonical_network_ts_arr.shape[0]),
        "canonical_networks_used": [row["canonical_network"] for row in canonical_network_rows],
        "n_timepoints": int(parcel_ts.shape[1]),
        "tsnr_threshold": 25.0 if (left_valid_mask is not None or right_valid_mask is not None) else None,
        "left_tsnr_mask": str(Path(args.left_tsnr_mask).resolve()) if args.left_tsnr_mask else None,
        "right_tsnr_mask": str(Path(args.right_tsnr_mask).resolve()) if args.right_tsnr_mask else None,
        "tsnr_gate_stats": {
            "left": {
                "n_grayordinates_total": int(left_dt.shape[0]),
                "n_grayordinates_used": int(left_valid_mask.sum()) if left_valid_mask is not None else int(left_dt.shape[0]),
                "n_grayordinates_masked": int((~left_valid_mask).sum()) if left_valid_mask is not None else 0,
            },
            "right": {
                "n_grayordinates_total": int(right_dt.shape[0]),
                "n_grayordinates_used": int(right_valid_mask.sum()) if right_valid_mask is not None else int(right_dt.shape[0]),
                "n_grayordinates_masked": int((~right_valid_mask).sum()) if right_valid_mask is not None else 0,
            },
            "combined": {
                "n_grayordinates_total": int(left_dt.shape[0] + right_dt.shape[0]),
                "n_grayordinates_used": int(
                    (left_valid_mask.sum() if left_valid_mask is not None else left_dt.shape[0])
                    + (right_valid_mask.sum() if right_valid_mask is not None else right_dt.shape[0])
                ),
                "n_grayordinates_masked": int(
                    ((~left_valid_mask).sum() if left_valid_mask is not None else 0)
                    + ((~right_valid_mask).sum() if right_valid_mask is not None else 0)
                ),
            },
            "rois_empty_after_tsnr_gate": {
                "left": int(len(left_empty_rows)),
                "right": int(len(right_empty_rows)),
                "combined": int(len(left_empty_rows) + len(right_empty_rows)),
            },
        },
    }
    (outdir / "reference_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

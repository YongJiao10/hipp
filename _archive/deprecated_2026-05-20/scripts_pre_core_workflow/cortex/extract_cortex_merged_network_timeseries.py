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
DEFAULT_CROSS_ATLAS_MERGE_JSON = REPO_ROOT / "config" / "cross_atlas_network_merge.json"


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


def parse_parent_network(label_name: str) -> str:
    m = ROI_NAME_PATTERN.fullmatch(label_name)
    if m:
        return str(m.group("network"))
    return str(label_name)


def assert_raw_label_input(path: Path, label_map: dict[int, str], hemi_tag: str) -> None:
    name = path.name
    if ".components." in name:
        raise ValueError(
            f"{hemi_tag} labels must be raw FASTANS network labels, not ROI components: {path}"
        )
    if ".boundary." in name:
        raise ValueError(f"{hemi_tag} labels must be network labels, not boundary labels: {path}")
    raw_like = sum(1 for v in label_map.values() if ROI_NAME_PATTERN.fullmatch(str(v)) is None)
    comp_like = sum(1 for v in label_map.values() if ROI_NAME_PATTERN.fullmatch(str(v)) is not None)
    if comp_like > 0 and raw_like == 0:
        raise ValueError(
            f"{hemi_tag} labels appear to be component labels (e.g., Network_L_01). "
            f"Use raw FASTANS labels such as PFM_Kong2019priors.{hemi_tag}.label.gii: {path}"
        )


def load_merge_config(path: Path, atlas_slug: str) -> tuple[list[str], set[str], dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    canonical_order = [str(item) for item in payload["canonical_network_order"]]
    exclude_labels = {str(item) for item in payload.get("exclude_labels", [])}
    atlases = payload.get("atlases", {})
    if atlas_slug not in atlases:
        raise KeyError(f"Atlas slug not found in merge config: {atlas_slug}")
    mapping = {str(k): str(v) for k, v in atlases[atlas_slug].get("mapping", {}).items()}
    return canonical_order, exclude_labels, mapping


def accumulate_by_canonical(
    *,
    hemi_data: np.ndarray,
    aligned_labels: np.ndarray,
    label_map: dict[int, str],
    valid_mask: np.ndarray,
    mapping: dict[str, str],
    exclude_labels: set[str],
    sums: dict[str, np.ndarray],
    counts_t: dict[str, np.ndarray],
    counts_vertices_used: dict[str, int],
    counts_vertices_total: dict[str, int],
    original_parents: dict[str, set[str]],
) -> None:
    key_to_canonical: dict[int, str] = {}
    key_to_parent: dict[int, str] = {}
    for key, name in label_map.items():
        parent = parse_parent_network(name)
        canonical = mapping.get(parent, parent)
        if canonical in exclude_labels:
            continue
        key_to_canonical[int(key)] = str(canonical)
        key_to_parent[int(key)] = str(parent)

    if not key_to_canonical:
        return

    for key, canonical in key_to_canonical.items():
        total_mask = aligned_labels == int(key)
        used_mask = total_mask & valid_mask
        n_total = int(np.sum(total_mask))
        n_used = int(np.sum(used_mask))
        if n_total <= 0:
            continue
        counts_vertices_total[canonical] = int(counts_vertices_total.get(canonical, 0) + n_total)
        original_parents.setdefault(canonical, set()).add(key_to_parent[key])
        if n_used <= 0:
            continue
        block = np.asarray(hemi_data[used_mask, :], dtype=np.float64)
        if canonical not in sums:
            sums[canonical] = np.zeros((block.shape[1],), dtype=np.float64)
            counts_t[canonical] = np.zeros((block.shape[1],), dtype=np.float64)
        finite = np.isfinite(block)
        sums[canonical] += np.where(finite, block, 0.0).sum(axis=0)
        counts_t[canonical] += finite.sum(axis=0)
        counts_vertices_used[canonical] = int(counts_vertices_used.get(canonical, 0) + n_used)


def write_tsv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract canonical merged cortex network timeseries directly from vertex labels and dtseries."
    )
    parser.add_argument("--subject", required=True)
    parser.add_argument("--atlas-slug", required=True)
    parser.add_argument("--dtseries", required=True)
    parser.add_argument(
        "--left-labels",
        required=True,
        help="Raw FASTANS left label.gii (for example PFM_Kong2019priors.L.label.gii); do not pass components.*.label.gii",
    )
    parser.add_argument(
        "--right-labels",
        required=True,
        help="Raw FASTANS right label.gii (for example PFM_Kong2019priors.R.label.gii); do not pass components.*.label.gii",
    )
    parser.add_argument("--left-tsnr-mask", required=True)
    parser.add_argument("--right-tsnr-mask", required=True)
    parser.add_argument(
        "--cross-atlas-merge-json",
        required=True,
        help=f"Path to cross-atlas merge config JSON (for example {DEFAULT_CROSS_ATLAS_MERGE_JSON}).",
    )
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    merge_json = Path(args.cross_atlas_merge_json).resolve()

    canonical_order, exclude_labels, mapping = load_merge_config(merge_json, str(args.atlas_slug))

    dt_img = nib.load(str(Path(args.dtseries).resolve()))
    dt_data = np.asarray(dt_img.dataobj, dtype=np.float32)
    if dt_data.ndim != 2:
        raise ValueError(f"Expected 2D dtseries, got shape={dt_data.shape}")
    dt_data_t = dt_data.T
    dt_axis = dt_img.header.get_axis(1)

    left_label_path = Path(args.left_labels).resolve()
    right_label_path = Path(args.right_labels).resolve()
    left_labels_all, left_label_map_all = load_label_gifti(left_label_path)
    right_labels_all, right_label_map_all = load_label_gifti(right_label_path)
    assert_raw_label_input(left_label_path, left_label_map_all, "L")
    assert_raw_label_input(right_label_path, right_label_map_all, "R")

    left_vertices, left_data = extract_structure_data(dt_axis, dt_data_t, "CORTEX_LEFT")
    right_vertices, right_data = extract_structure_data(dt_axis, dt_data_t, "CORTEX_RIGHT")

    left_valid = load_required_mask(args.left_tsnr_mask, int(left_vertices.shape[0]), "left cortex tSNR")
    right_valid = load_required_mask(args.right_tsnr_mask, int(right_vertices.shape[0]), "right cortex tSNR")

    left_aligned, left_label_map = align_surface_labels(left_labels_all, left_label_map_all, left_vertices)
    right_aligned, right_label_map = align_surface_labels(right_labels_all, right_label_map_all, right_vertices)

    sums: dict[str, np.ndarray] = {}
    counts_t: dict[str, np.ndarray] = {}
    counts_vertices_used: dict[str, int] = {}
    counts_vertices_total: dict[str, int] = {}
    original_parents: dict[str, set[str]] = {}

    accumulate_by_canonical(
        hemi_data=left_data,
        aligned_labels=left_aligned,
        label_map=left_label_map,
        valid_mask=left_valid,
        mapping=mapping,
        exclude_labels=exclude_labels,
        sums=sums,
        counts_t=counts_t,
        counts_vertices_used=counts_vertices_used,
        counts_vertices_total=counts_vertices_total,
        original_parents=original_parents,
    )
    accumulate_by_canonical(
        hemi_data=right_data,
        aligned_labels=right_aligned,
        label_map=right_label_map,
        valid_mask=right_valid,
        mapping=mapping,
        exclude_labels=exclude_labels,
        sums=sums,
        counts_t=counts_t,
        counts_vertices_used=counts_vertices_used,
        counts_vertices_total=counts_vertices_total,
        original_parents=original_parents,
    )

    atlas_targets = set(mapping.values()) - exclude_labels
    canonical_names = [name for name in canonical_order if name not in exclude_labels and name in atlas_targets]
    missing = [name for name in canonical_names if name not in sums]
    if missing:
        detail = []
        for name in missing:
            detail.append(
                f"{name}(total={counts_vertices_total.get(name, 0)}, used={counts_vertices_used.get(name, 0)})"
            )
        raise RuntimeError(
            "Missing canonical network timeseries; extraction must produce a fixed network set. "
            f"Missing: {', '.join(detail)}"
        )

    ts_rows: list[np.ndarray] = []
    table_rows: list[dict[str, object]] = []
    for name in canonical_names:
        den = np.clip(counts_t[name], 1.0, None)
        ts = (sums[name] / den).astype(np.float32)
        ts_rows.append(ts)
        table_rows.append(
            {
                "canonical_network": str(name),
                "n_grayordinates_total": int(counts_vertices_total.get(name, 0)),
                "n_grayordinates_used": int(counts_vertices_used.get(name, 0)),
                "original_parent_networks": ",".join(sorted(original_parents.get(name, set()))),
            }
        )

    ts_out = np.asarray(ts_rows, dtype=np.float32)
    np.save(outdir / "cortex_canonical_network_timeseries.npy", ts_out)
    write_tsv(
        outdir / "cortex_canonical_networks.tsv",
        table_rows,
        ["canonical_network", "n_grayordinates_total", "n_grayordinates_used", "original_parent_networks"],
    )

    summary = {
        "subject": str(args.subject),
        "atlas_slug": str(args.atlas_slug),
        "extraction_mode": "direct_vertex_level_merged_network_timeseries",
        "dtseries": str(Path(args.dtseries).resolve()),
        "left_labels": str(left_label_path),
        "right_labels": str(right_label_path),
        "cross_atlas_merge_json": str(merge_json),
        "n_canonical_networks_used": int(ts_out.shape[0]),
        "canonical_networks_used": canonical_names,
        "n_timepoints": int(ts_out.shape[1]),
        "left_tsnr_mask": str(Path(args.left_tsnr_mask).resolve()),
        "right_tsnr_mask": str(Path(args.right_tsnr_mask).resolve()),
        "tsnr_gate_stats": {
            "left": {
                "n_grayordinates_total": int(left_data.shape[0]),
                "n_grayordinates_used": int(left_valid.sum()),
                "n_grayordinates_masked": int((~left_valid).sum()),
            },
            "right": {
                "n_grayordinates_total": int(right_data.shape[0]),
                "n_grayordinates_used": int(right_valid.sum()),
                "n_grayordinates_masked": int((~right_valid).sum()),
            },
        },
    }
    (outdir / "reference_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np

from run_cortex_pfm_subject import METHODS, REPO_ROOT, WB_COMMAND


HEMIS = {
    "L": "LEFT",
    "R": "RIGHT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive network-component ROIs from existing cortex PFM label maps."
    )
    parser.add_argument("--subject", required=True, help="Subject ID without sub- prefix")
    parser.add_argument("--method", required=True, help="PFM method name or slug")
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data" / "hippunfold_input"))
    parser.add_argument("--out-root", default=str(REPO_ROOT / "outputs" / "cortex_pfm"))
    parser.add_argument("--roi-min-area-mm2", type=float, default=25.0)
    return parser.parse_args()


def canonical_method(method: str) -> str:
    if method in METHODS:
        return method
    lowered = method.lower()
    for key, info in METHODS.items():
        if lowered == info["slug"].lower():
            return key
    raise ValueError(f"Unsupported method: {method}")


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")


def metric_map_names(metric_path: Path) -> list[str]:
    img = nib.load(str(metric_path))
    names: list[str] = []
    for idx, darray in enumerate(img.darrays):
        names.append(darray.meta.get("Name") or f"map_{idx + 1}")
    return names


def write_metric(data: np.ndarray, out_path: Path, map_name: str) -> None:
    darray = nib.gifti.GiftiDataArray(
        data=np.asarray(data, dtype=np.float32),
        intent=nib.nifti1.intent_codes["NIFTI_INTENT_SHAPE"],
        meta=nib.gifti.GiftiMetaData.from_dict({"Name": map_name}),
    )
    img = nib.gifti.GiftiImage(darrays=[darray])
    nib.save(img, str(out_path))


def color_for_name(name: str) -> tuple[int, int, int, int]:
    digest = hashlib.sha1(name.encode("utf-8")).digest()
    base = np.array([digest[0], digest[1], digest[2]], dtype=float)
    base = 70.0 + (base / 255.0) * 165.0
    max_idx = int(np.argmax(base))
    base[max_idx] = 255.0
    rgb = tuple(int(round(value)) for value in base)
    return rgb[0], rgb[1], rgb[2], 255


def write_label_list(rows: list[dict[str, object]], out_path: Path) -> None:
    lines: list[str] = []
    for row in rows:
        lines.append(str(row["roi_name"]))
        rgba = row["rgba"]
        lines.append(f"{row['roi_key']} {rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_simple_label_list(
    out_path: Path,
    label_name: str,
    key: int,
    rgba: tuple[int, int, int, int],
) -> None:
    out_path.write_text(
        f"{label_name}\n{key} {rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}\n",
        encoding="utf-8",
    )


def ensure_vertex_areas(surface: Path, out_metric: Path) -> None:
    if out_metric.exists():
        return
    out_metric.parent.mkdir(parents=True, exist_ok=True)
    run([WB_COMMAND, "-surface-vertex-areas", str(surface), str(out_metric)])


def derive_hemisphere(
    subject: str,
    method: str,
    hemisphere: str,
    surface: Path,
    label_path: Path,
    out_dir: Path,
    roi_min_area_mm2: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    vertex_area_metric = out_dir / f"sub-{subject}_hemi-{hemisphere}_vertex_areas.func.gii"
    ensure_vertex_areas(surface, vertex_area_metric)
    vertex_areas = np.asarray(nib.load(str(vertex_area_metric)).darrays[0].data, dtype=float)

    all_rois_metric = out_dir / f"PFM_{method}priors.hemi-{hemisphere}.network_rois.func.gii"
    run([WB_COMMAND, "-gifti-all-labels-to-rois", str(label_path), "1", str(all_rois_metric)])
    network_names = metric_map_names(all_rois_metric)

    roi_ids = np.zeros(vertex_areas.shape[0], dtype=np.int32)
    label_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    networks_summary: dict[str, dict[str, object]] = {}
    next_roi_key = 1

    for network_name in network_names:
        safe_name = "".join(ch if ch.isalnum() else "_" for ch in network_name)
        clusters_metric = out_dir / f"clusters_{safe_name}.func.gii"
        run(
            [
                WB_COMMAND,
                "-metric-find-clusters",
                str(surface),
                str(all_rois_metric),
                "0.5",
                "0.0",
                str(clusters_metric),
                "-corrected-areas",
                str(vertex_area_metric),
                "-column",
                network_name,
            ]
        )
        cluster_values = np.asarray(nib.load(str(clusters_metric)).darrays[0].data, dtype=np.int32)
        cluster_ids = sorted(int(value) for value in np.unique(cluster_values) if int(value) > 0)
        cluster_records: list[dict[str, object]] = []
        for cluster_id in cluster_ids:
            mask = cluster_values == cluster_id
            area_mm2 = float(vertex_areas[mask].sum())
            cluster_records.append(
                {
                    "cluster_id": cluster_id,
                    "area_mm2": area_mm2,
                    "kept": area_mm2 >= roi_min_area_mm2,
                    "vertex_count": int(mask.sum()),
                }
            )

        cluster_records.sort(key=lambda row: (-row["area_mm2"], int(row["cluster_id"])))
        kept_rank = 0
        kept_count = 0
        for record in cluster_records:
            kept = bool(record["kept"])
            roi_name = ""
            roi_key = 0
            if kept:
                kept_rank += 1
                kept_count += 1
                roi_name = f"{network_name}_{hemisphere}_{kept_rank:02d}"
                roi_key = next_roi_key
                next_roi_key += 1
                rgba = color_for_name(roi_name)
                label_rows.append({"roi_key": roi_key, "roi_name": roi_name, "rgba": rgba})
                roi_ids[cluster_values == int(record["cluster_id"])] = roi_key

            component_rows.append(
                {
                    "subject": subject,
                    "method": method,
                    "hemisphere": hemisphere,
                    "network": network_name,
                    "cluster_id": int(record["cluster_id"]),
                    "component_rank_in_network": kept_rank if kept else "",
                    "area_mm2": round(float(record["area_mm2"]), 6),
                    "vertex_count": int(record["vertex_count"]),
                    "kept": kept,
                    "roi_key": roi_key,
                    "roi_name": roi_name,
                }
            )

        networks_summary[network_name] = {
            "raw_component_count": len(cluster_records),
            "kept_roi_count": kept_count,
            "components": [
                {
                    "cluster_id": int(record["cluster_id"]),
                    "area_mm2": round(float(record["area_mm2"]), 6),
                    "vertex_count": int(record["vertex_count"]),
                    "kept": bool(record["kept"]),
                }
                for record in cluster_records
            ],
        }

    roi_metric = out_dir / f"PFM_{method}priors.components.{hemisphere}.func.gii"
    roi_label_list = out_dir / f"PFM_{method}priors.components.{hemisphere}.txt"
    roi_label = out_dir / f"PFM_{method}priors.components.{hemisphere}.label.gii"
    write_metric(roi_ids, roi_metric, f"PFM_{method}priors.components.{hemisphere}")
    write_label_list(label_rows, roi_label_list)
    run(
        [
            WB_COMMAND,
            "-metric-label-import",
            str(roi_metric),
            str(roi_label_list),
            str(roi_label),
            "-discard-others",
            "-drop-unused-labels",
        ]
    )

    boundary_border = out_dir / f"PFM_{method}priors.components.{hemisphere}.border"
    boundary_metric_raw = out_dir / f"PFM_{method}priors.components.{hemisphere}.boundary.raw.func.gii"
    boundary_metric = out_dir / f"PFM_{method}priors.components.{hemisphere}.boundary.func.gii"
    boundary_label_list = out_dir / f"PFM_{method}priors.components.{hemisphere}.boundary.txt"
    boundary_label = out_dir / f"PFM_{method}priors.components.{hemisphere}.boundary.label.gii"
    run([WB_COMMAND, "-label-to-border", str(surface), str(roi_label), str(boundary_border)])
    run([WB_COMMAND, "-border-to-vertices", str(surface), str(boundary_border), str(boundary_metric_raw)])
    boundary_img = nib.load(str(boundary_metric_raw))
    boundary_data = np.vstack([np.asarray(darray.data, dtype=np.float32) for darray in boundary_img.darrays])
    merged_boundary = (boundary_data.max(axis=0) > 0).astype(np.float32)
    write_metric(merged_boundary, boundary_metric, f"PFM_{method}priors.components.{hemisphere}.boundary")
    write_simple_label_list(boundary_label_list, "ROI_Boundary", 1, (255, 0, 255, 255))
    run(
        [
            WB_COMMAND,
            "-metric-label-import",
            str(boundary_metric),
            str(boundary_label_list),
            str(boundary_label),
            "-discard-others",
            "-drop-unused-labels",
        ]
    )

    hemi_summary = {
        "hemisphere": hemisphere,
        "surface": str(surface),
        "input_label": str(label_path),
        "vertex_area_metric": str(vertex_area_metric),
        "network_roi_metric": str(all_rois_metric),
        "roi_metric": str(roi_metric),
        "roi_label_list": str(roi_label_list),
        "roi_label": str(roi_label),
        "boundary_border": str(boundary_border),
        "boundary_metric": str(boundary_metric),
        "boundary_label_list": str(boundary_label_list),
        "boundary_label": str(boundary_label),
        "raw_component_count": sum(item["raw_component_count"] for item in networks_summary.values()),
        "kept_roi_count": sum(item["kept_roi_count"] for item in networks_summary.values()),
        "networks": networks_summary,
    }
    return hemi_summary, component_rows


def write_component_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject",
        "method",
        "hemisphere",
        "network",
        "cluster_id",
        "component_rank_in_network",
        "area_mm2",
        "vertex_count",
        "kept",
        "roi_key",
        "roi_name",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def derive_roi_components(
    subject: str,
    method: str,
    data_root: Path,
    out_root: Path,
    roi_min_area_mm2: float,
) -> dict[str, object]:
    method = canonical_method(method)
    slug = METHODS[method]["slug"]
    subject_root = out_root / f"sub-{subject}"
    method_dir = subject_root / slug
    roi_dir = method_dir / "roi_components"
    roi_dir.mkdir(parents=True, exist_ok=True)

    hemi_results: dict[str, dict[str, object]] = {}
    all_rows: list[dict[str, object]] = []
    for hemisphere in ("L", "R"):
        hemi_result, rows = derive_hemisphere(
            subject=subject,
            method=method,
            hemisphere=hemisphere,
            surface=data_root
            / f"sub-{subject}"
            / "anat"
            / f"sub-{subject}_hemi-{hemisphere}_space-fsLR_den-32k_desc-MSMAll_midthickness.surf.gii",
            label_path=method_dir / f"PFM_{method}priors.{hemisphere}.label.gii",
            out_dir=roi_dir / f"hemi_{hemisphere}",
            roi_min_area_mm2=roi_min_area_mm2,
        )
        hemi_results[hemisphere] = hemi_result
        all_rows.extend(rows)

    stats = {
        "subject": subject,
        "method": method,
        "method_slug": slug,
        "roi_min_area_mm2": roi_min_area_mm2,
        "raw_component_count": sum(result["raw_component_count"] for result in hemi_results.values()),
        "kept_roi_count": sum(result["kept_roi_count"] for result in hemi_results.values()),
        "hemispheres": hemi_results,
        "component_csv": str(roi_dir / "roi_component_stats.csv"),
    }
    stats_path = roi_dir / "roi_component_stats.json"
    csv_path = roi_dir / "roi_component_stats.csv"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    write_component_csv(all_rows, csv_path)
    return stats


def main() -> int:
    args = parse_args()
    stats = derive_roi_components(
        subject=args.subject,
        method=args.method,
        data_root=Path(args.data_root).resolve(),
        out_root=Path(args.out_root).resolve(),
        roi_min_area_mm2=args.roi_min_area_mm2,
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Map Stanford volume-space TMS targets to individualized Kong2022 surface labels."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import scipy.io as sio
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_WORKFLOW_DIR = REPO_ROOT / "scripts" / "core_workflow"
if str(CORE_WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_WORKFLOW_DIR))

from lib.kong2022_label_utils import (
    KONG2022_DLABEL_PATH,
    load_kong2022_bundle,
)

DEFAULT_MAT_PATH = (
    REPO_ROOT
    / "outputs_stanford/inputs/stanford/tmsfmri_fmriprep/targetInfo_11sites_wMNI_20211230.mat"
)
DEFAULT_LSS_ROOT = Path(
    "/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep"
)
DEFAULT_ROSTER_FILE = DEFAULT_LSS_ROOT / "sub_list.txt"
DEFAULT_OUTPUT_CSV = (
    DEFAULT_LSS_ROOT / "stanford_tms_site_kong2022_individual_network_summary.csv"
)
DEFAULT_OUTPUT_JSON = (
    DEFAULT_LSS_ROOT / "stanford_tms_site_kong2022_individual_network_summary.json"
)
REPO_OUTPUT_DIR = REPO_ROOT / "outputs_stanford/present/tms_site_networks"

FSLR_32K_VERTICES_PER_HEMISPHERE = 32492


def build_dense_group_parcel_labels(
    vertex_arrays: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    vertex_count: int = FSLR_32K_VERTICES_PER_HEMISPHERE,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for hemi in ("L", "R"):
        if hemi not in vertex_arrays:
            raise KeyError(f"Group Kong2022 dlabel missing hemisphere {hemi}")
        raw_indices, raw_parcels = vertex_arrays[hemi]
        indices = np.asarray(raw_indices, dtype=np.int64).reshape(-1)
        parcels = np.asarray(raw_parcels, dtype=np.int32).reshape(-1)
        dense = np.zeros(vertex_count, dtype=np.int32)
        dense[indices] = parcels
        result[hemi] = dense
    return result


def find_subject_t1(lss_root: Path, subject_id: str) -> Path | None:
    candidates = [
        lss_root / f"DataOrg_PFM_22P/sub-{subject_id}/anat/sub-{subject_id}_T1w.nii.gz",
        lss_root / f"DataOrg_PFM_36P/sub-{subject_id}/anat/sub-{subject_id}_T1w.nii.gz",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def find_subject_surface(lss_root: Path, subject_id: str, hemi: str) -> Path | None:
    candidates = [
        lss_root
        / f"DataOrg_PFM_22P/sub-{subject_id}/anat/sub-{subject_id}_hemi-{hemi}_space-fsLR_den-32k_midthickness.surf.gii",
        lss_root
        / f"DataOrg_PFM_36P/sub-{subject_id}/anat/sub-{subject_id}_hemi-{hemi}_space-fsLR_den-32k_midthickness.surf.gii",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def find_subject_labels(lss_root: Path, subject_id: str, hemi: str) -> Path | None:
    filename = f"{hemi.lower()}h_labels.txt"
    candidates = [
        lss_root / f"ArealMSHBM_Kong2022_22P/sub-{subject_id}/{filename}",
        lss_root / f"ArealMSHBM_Kong2022/sub-{subject_id}/{filename}",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def voxel_to_ras(voxel: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Convert voxel coordinates to RAS+ mm using affine (handling LAS axcodes)."""
    world = affine @ np.append(voxel, 1.0)
    # Stanford T1w images have ('L', 'A', 'S') orientation; in RAS+, X is inverted
    return np.array([-world[0], world[1], world[2]], dtype=np.float64)


def map_stanford_tms_sites(
    mat_path: Path,
    lss_root: Path,
    roster_file: Path | None = None,
    neighborhood_radius_mm: float = 5.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bundle = load_kong2022_bundle()
    roi_by_id = {int(r["roi_id"]): r for r in bundle["roi_registry"]}
    dense_group_labels = build_dense_group_parcel_labels(bundle["vertex_arrays"])

    mat = sio.loadmat(str(mat_path))
    target_info = mat["targetInfo"][0]

    # Cache loaded surfaces, trees, affine, and labels to avoid redundant I/O
    cached_t1_affine: dict[str, np.ndarray] = {}
    cached_surf_coords: dict[tuple[str, str], np.ndarray] = {}
    cached_surf_trees: dict[tuple[str, str], cKDTree] = {}
    cached_labels: dict[tuple[str, str], np.ndarray] = {}
    cached_surf_paths: dict[tuple[str, str], str] = {}
    cached_label_paths: dict[tuple[str, str], str] = {}

    rows: list[dict[str, Any]] = []

    # Map all available targets
    for site_idx in range(len(target_info)):
        site_entry = target_info[site_idx]
        site_name = str(site_entry["name"].squeeze())
        hemi = "L" if site_name.startswith("L-") else "R"

        sublist = [str(s) for s in site_entry["sublist"].squeeze()]
        target_locs = site_entry["targetLoc"].squeeze()
        target_locs_mni = site_entry["targetLocMNI"].squeeze()
        target_brain_env_mni = site_entry["targetBrainEnvLocMNI"].squeeze()
        loc_brain_env = site_entry["loc_brainEnv2target"].squeeze()
        loc_scalp = site_entry["loc_scalp2target"].squeeze()

        for sub_pos, sub_id in enumerate(sublist):
            row: dict[str, Any] = {
                "subject": f"sub-{sub_id}",
                "site": site_name,
                "hemi": hemi,
                "mapping_status": "unmapped",
                "mapping_error": None,
            }

            target_vox = target_locs[sub_pos]
            mni_xyz = target_locs_mni[sub_pos]
            brain_env_mni_xyz = target_brain_env_mni[sub_pos]
            brain_env_vox = loc_brain_env[sub_pos]
            scalp_vox = loc_scalp[sub_pos]

            row["target_loc_voxel_x"] = float(target_vox[0])
            row["target_loc_voxel_y"] = float(target_vox[1])
            row["target_loc_voxel_z"] = float(target_vox[2])
            row["target_loc_mni_x"] = float(mni_xyz[0])
            row["target_loc_mni_y"] = float(mni_xyz[1])
            row["target_loc_mni_z"] = float(mni_xyz[2])
            row["target_brain_env_mni_x"] = float(brain_env_mni_xyz[0])
            row["target_brain_env_mni_y"] = float(brain_env_mni_xyz[1])
            row["target_brain_env_mni_z"] = float(brain_env_mni_xyz[2])
            row["brain_env_voxel_x"] = float(brain_env_vox[0])
            row["brain_env_voxel_y"] = float(brain_env_vox[1])
            row["brain_env_voxel_z"] = float(brain_env_vox[2])
            row["scalp_voxel_x"] = float(scalp_vox[0])
            row["scalp_voxel_y"] = float(scalp_vox[1])
            row["scalp_voxel_z"] = float(scalp_vox[2])

            # Load T1 affine
            if sub_id not in cached_t1_affine:
                t1_p = find_subject_t1(lss_root, sub_id)
                if t1_p is None:
                    row["mapping_error"] = f"Missing T1w volume for sub-{sub_id}"
                    rows.append(row)
                    continue
                cached_t1_affine[sub_id] = nib.load(t1_p).affine
            affine = cached_t1_affine[sub_id]

            # Compute RAS coordinates
            target_ras = voxel_to_ras(target_vox, affine)
            brain_env_ras = voxel_to_ras(brain_env_vox, affine)
            scalp_ras = voxel_to_ras(scalp_vox, affine)

            row["target_ras_x"] = float(target_ras[0])
            row["target_ras_y"] = float(target_ras[1])
            row["target_ras_z"] = float(target_ras[2])
            row["brain_env_ras_x"] = float(brain_env_ras[0])
            row["brain_env_ras_y"] = float(brain_env_ras[1])
            row["brain_env_ras_z"] = float(brain_env_ras[2])
            row["scalp_ras_x"] = float(scalp_ras[0])
            row["scalp_ras_y"] = float(scalp_ras[1])
            row["scalp_ras_z"] = float(scalp_ras[2])

            # Load surface
            surf_key = (sub_id, hemi)
            if surf_key not in cached_surf_coords:
                surf_p = find_subject_surface(lss_root, sub_id, hemi)
                if surf_p is None:
                    row["mapping_error"] = f"Missing {hemi} surface for sub-{sub_id}"
                    rows.append(row)
                    continue
                surf_gii = nib.load(surf_p)
                coords = surf_gii.darrays[0].data
                cached_surf_coords[surf_key] = coords
                cached_surf_trees[surf_key] = cKDTree(coords)
                cached_surf_paths[surf_key] = str(surf_p)
            surf_coords = cached_surf_coords[surf_key]
            tree = cached_surf_trees[surf_key]
            row["surface_path"] = cached_surf_paths[surf_key]

            # Load labels
            label_key = (sub_id, hemi)
            if label_key not in cached_labels:
                lbl_p = find_subject_labels(lss_root, sub_id, hemi)
                if lbl_p is None:
                    row["mapping_error"] = f"Missing {hemi} Kong2022 labels for sub-{sub_id}"
                    rows.append(row)
                    continue
                cached_labels[label_key] = np.loadtxt(lbl_p, dtype=np.int32)
                cached_label_paths[label_key] = str(lbl_p)
            labels = cached_labels[label_key]
            row["label_path"] = cached_label_paths[label_key]

            # 1. Primary Target Projection
            target_dist, target_vidx = tree.query(target_ras)
            target_vidx = int(target_vidx)
            row["target_to_surface_distance_mm"] = float(target_dist)
            row["target_vertex_index"] = target_vidx
            row["target_vertex_x"] = float(surf_coords[target_vidx, 0])
            row["target_vertex_y"] = float(surf_coords[target_vidx, 1])
            row["target_vertex_z"] = float(surf_coords[target_vidx, 2])

            target_pid = int(labels[target_vidx])
            row["target_parcel_id"] = target_pid
            roi = roi_by_id.get(target_pid)
            row["target_parcel_name"] = roi["roi_name"] if roi else "Unknown"
            row["target_family17"] = roi["family17"] if roi else "Unknown"
            row["target_canonical7"] = roi["canonical7"] if roi else "Unknown"
            row["target_canonical8"] = roi["canonical8"] if roi else "Unknown"

            # Group atlas comparison for primary target
            grp_pid = int(dense_group_labels[hemi][target_vidx])
            grp_roi = roi_by_id.get(grp_pid)
            row["target_group_parcel_id"] = grp_pid
            row["target_group_parcel_name"] = grp_roi["roi_name"] if grp_roi else "Unknown"
            row["target_group_family17"] = grp_roi["family17"] if grp_roi else "Unknown"
            row["target_group_canonical7"] = grp_roi["canonical7"] if grp_roi else "Unknown"
            row["target_group_canonical8"] = grp_roi["canonical8"] if grp_roi else "Unknown"
            row["target_individual_vs_group_same_parcel"] = bool(target_pid == grp_pid)
            row["target_individual_vs_group_same_family17"] = bool(
                row["target_family17"] == row["target_group_family17"]
            )
            row["target_individual_vs_group_same_canonical7"] = bool(
                row["target_canonical7"] == row["target_group_canonical7"]
            )
            row["target_individual_vs_group_same_canonical8"] = bool(
                row["target_canonical8"] == row["target_group_canonical8"]
            )

            # 2. Brain Envelope Projection
            env_dist, env_vidx = tree.query(brain_env_ras)
            env_vidx = int(env_vidx)
            row["brain_env_to_surface_distance_mm"] = float(env_dist)
            row["brain_env_vertex_index"] = env_vidx
            row["brain_env_vertex_x"] = float(surf_coords[env_vidx, 0])
            row["brain_env_vertex_y"] = float(surf_coords[env_vidx, 1])
            row["brain_env_vertex_z"] = float(surf_coords[env_vidx, 2])

            env_pid = int(labels[env_vidx])
            row["brain_env_parcel_id"] = env_pid
            env_roi = roi_by_id.get(env_pid)
            row["brain_env_parcel_name"] = env_roi["roi_name"] if env_roi else "Unknown"
            row["brain_env_family17"] = env_roi["family17"] if env_roi else "Unknown"
            row["brain_env_canonical7"] = env_roi["canonical7"] if env_roi else "Unknown"
            row["brain_env_canonical8"] = env_roi["canonical8"] if env_roi else "Unknown"

            env_grp_pid = int(dense_group_labels[hemi][env_vidx])
            env_grp_roi = roi_by_id.get(env_grp_pid)
            row["brain_env_group_family17"] = (
                env_grp_roi["family17"] if env_grp_roi else "Unknown"
            )
            row["brain_env_group_canonical7"] = (
                env_grp_roi["canonical7"] if env_grp_roi else "Unknown"
            )
            row["brain_env_group_canonical8"] = (
                env_grp_roi["canonical8"] if env_grp_roi else "Unknown"
            )
            row["brain_env_individual_vs_group_same_family17"] = bool(
                row["brain_env_family17"] == row["brain_env_group_family17"]
            )
            row["brain_env_individual_vs_group_same_canonical7"] = bool(
                row["brain_env_canonical7"] == row["brain_env_group_canonical7"]
            )

            # 3. Neighborhood Consistency around target
            nbr_indices = tree.query_ball_point(target_ras, r=neighborhood_radius_mm)
            row["neighborhood_radius_mm"] = neighborhood_radius_mm
            row["neighborhood_vertex_count"] = len(nbr_indices)
            if nbr_indices:
                nbr_pids = labels[nbr_indices]
                nbr_fams = [
                    roi_by_id[p]["family17"]
                    for p in nbr_pids
                    if p in roi_by_id
                ]
                if nbr_fams:
                    fam_counts = Counter(nbr_fams)
                    dom_fam, dom_cnt = fam_counts.most_common(1)[0]
                    row["neighborhood_primary_family17_fraction"] = float(
                        fam_counts.get(row["target_family17"], 0) / len(nbr_fams)
                    )
                    row["neighborhood_dominant_family17"] = dom_fam
                    row["neighborhood_dominant_family17_fraction"] = float(
                        dom_cnt / len(nbr_fams)
                    )
                else:
                    row["neighborhood_primary_family17_fraction"] = 0.0
                    row["neighborhood_dominant_family17"] = "Unknown"
                    row["neighborhood_dominant_family17_fraction"] = 0.0
            else:
                row["neighborhood_primary_family17_fraction"] = 0.0
                row["neighborhood_dominant_family17"] = "Unknown"
                row["neighborhood_dominant_family17_fraction"] = 0.0

            row["qc_distance_gt_10mm"] = bool(target_dist > 10.0)
            row["qc_neighborhood_primary_fraction_lt_0p5"] = bool(
                row["neighborhood_primary_family17_fraction"] < 0.5
            )
            row["qc_neighborhood_dominant_disagrees"] = bool(
                row["neighborhood_dominant_family17"] != row["target_family17"]
            )

            row["mapping_status"] = "mapped"
            rows.append(row)

    # Sort rows by subject, site
    rows.sort(key=lambda r: (r["subject"], r["site"]))

    mapped_rows = [r for r in rows if r["mapping_status"] == "mapped"]
    summary: dict[str, Any] = {
        "n_total_rows": len(rows),
        "n_mapped_rows": len(mapped_rows),
        "n_unmapped_rows": len(rows) - len(mapped_rows),
        "n_unique_subjects": len(set(r["subject"] for r in mapped_rows)),
        "n_sites": len(set(r["site"] for r in mapped_rows)),
        "sites": sorted(list(set(r["site"] for r in mapped_rows))),
        "family17_distribution": dict(Counter(r["target_family17"] for r in mapped_rows)),
        "canonical7_distribution": dict(Counter(r["target_canonical7"] for r in mapped_rows)),
        "canonical8_distribution": dict(Counter(r["target_canonical8"] for r in mapped_rows)),
        "brain_env_family17_distribution": dict(
            Counter(r["brain_env_family17"] for r in mapped_rows)
        ),
        "brain_env_canonical7_distribution": dict(
            Counter(r["brain_env_canonical7"] for r in mapped_rows)
        ),
        "brain_env_canonical8_distribution": dict(
            Counter(r["brain_env_canonical8"] for r in mapped_rows)
        ),
        "target_individual_vs_group_agreement": {
            "family17_match_count": sum(
                1 for r in mapped_rows if r["target_individual_vs_group_same_family17"]
            ),
            "family17_match_rate": (
                sum(1 for r in mapped_rows if r["target_individual_vs_group_same_family17"])
                / len(mapped_rows)
                if mapped_rows
                else 0.0
            ),
            "canonical7_match_count": sum(
                1 for r in mapped_rows if r["target_individual_vs_group_same_canonical7"]
            ),
            "canonical7_match_rate": (
                sum(1 for r in mapped_rows if r["target_individual_vs_group_same_canonical7"])
                / len(mapped_rows)
                if mapped_rows
                else 0.0
            ),
            "canonical8_match_count": sum(
                1 for r in mapped_rows if r["target_individual_vs_group_same_canonical8"]
            ),
            "canonical8_match_rate": (
                sum(1 for r in mapped_rows if r["target_individual_vs_group_same_canonical8"])
                / len(mapped_rows)
                if mapped_rows
                else 0.0
            ),
        },
        "distances_mm": {
            "target_median": float(
                np.median([r["target_to_surface_distance_mm"] for r in mapped_rows])
            )
            if mapped_rows
            else 0.0,
            "target_max": float(
                np.max([r["target_to_surface_distance_mm"] for r in mapped_rows])
            )
            if mapped_rows
            else 0.0,
            "brain_env_median": float(
                np.median([r["brain_env_to_surface_distance_mm"] for r in mapped_rows])
            )
            if mapped_rows
            else 0.0,
        },
    }

    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map Stanford TMS sites to individual Kong2022 network labels."
    )
    parser.add_argument(
        "--mat-path",
        type=Path,
        default=DEFAULT_MAT_PATH,
        help=f"Path to targetInfo mat file (default: {DEFAULT_MAT_PATH})",
    )
    parser.add_argument(
        "--lss-root",
        type=Path,
        default=DEFAULT_LSS_ROOT,
        help=f"Path to rest_fmriprep root (default: {DEFAULT_LSS_ROOT})",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_CSV})",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT_JSON})",
    )
    parser.add_argument(
        "--radius-mm",
        type=float,
        default=5.0,
        help="Neighborhood radius in mm (default: 5.0)",
    )
    args = parser.parse_args()

    print(f"Mapping Stanford TMS sites from {args.mat_path}...")
    rows, summary = map_stanford_tms_sites(
        mat_path=args.mat_path,
        lss_root=args.lss_root,
        roster_file=DEFAULT_ROSTER_FILE,
        neighborhood_radius_mm=args.radius_mm,
    )

    # Save to LSS destination
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV to {args.out_csv} ({len(rows)} rows)")

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved JSON summary to {args.out_json}")

    # Also save a repository-local copy in outputs_stanford/present/tms_site_networks/
    REPO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    repo_csv = REPO_OUTPUT_DIR / args.out_csv.name
    repo_json = REPO_OUTPUT_DIR / args.out_json.name
    with open(repo_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(repo_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved repo-local copy to {repo_csv} and {repo_json}")


if __name__ == "__main__":
    main()

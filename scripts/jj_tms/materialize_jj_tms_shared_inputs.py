#!/usr/bin/env python3
"""Materialize shared_inputs (surface, routea) and group average template surfaces for JJ_TMS."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
import numpy as np
import nibabel as nib
from nibabel.gifti import GiftiDataArray, GiftiImage

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_WORKFLOW_DIR = REPO_ROOT / "scripts" / "core_workflow"
if str(CORE_WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_WORKFLOW_DIR))

from lib.kong2022_label_utils import (
    canonical_families_17_order,
    canonical_families_7_order,
    load_kong2022_merge_mapping,
)

JJ_MATCH_ROOT = REPO_ROOT / "outputs_iowa_jj_tms/work/hcp151_fixed_template_matching/subjects"
BATCH_ROOT = REPO_ROOT / "outputs_iowa_jj_tms/inputs/iowa_jj_tms/hippunfold/dense_corobl_batch"
KONG_ROOT = REPO_ROOT / "outputs_iowa_jj_tms/inputs/iowa_jj_tms/kong2022"
SHARED_SURFACE_ROOT = KONG_ROOT / "shared_inputs" / "surface"
SHARED_ROUTEA_ROOT = KONG_ROOT / "shared_inputs" / "routea"
TEMPLATE_ROOT = KONG_ROOT / "templates" / "hipp_group_average_t1only_151" / "group_average_surfaces"
MANIFEST_94 = REPO_ROOT / "manifests" / "jj_tms_94_subjects.txt"
KONG2022_REGISTRY_TSV = REPO_ROOT / "outputs_hcp/work/group_atlas/routea_full_k3_fisherz/gradient-spectral/kong2022/2mm/cov_0p80/g1g2g3/hemi_L/cluster_roi_registry.tsv"

HEMIS = ["L", "R"]
SMOOTHINGS = ["2mm", "4mm"]


def load_kong2022_reference_table() -> list[dict[str, str]]:
    lines = KONG2022_REGISTRY_TSV.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        rows.append(dict(zip(header, parts)))
    mapping_17_to_7 = load_kong2022_merge_mapping(taxonomy="canonical7")
    for row in rows:
        row["canonical7"] = mapping_17_to_7[row["family17"]]
    return rows


def build_cortex_parcels_tsv(vertex_counts: np.ndarray, base_table: list[dict[str, str]]) -> str:
    lines = ["parcel_id\tparcel_name\tfamily17\tcanonical7\tcanonical8\themisphere\tsubregion\trank\tn_grayordinates_total\tn_grayordinates_used\texcluded_by_tsnr_gate"]
    for i, row in enumerate(base_table):
        cnt = int(vertex_counts[i])
        excluded = str(cnt <= 0)
        name = row.get("roi_name") or row.get("parcel_name")
        lines.append(f"{row['roi_id']}\t{name}\t{row['family17']}\t{row['canonical7']}\t{row['canonical8']}\t{row['hemi']}\t{row['subregion']}\t{row.get('rank', '1')}\t{cnt}\t{cnt}\t{excluded}")
    return "\n".join(lines) + "\n"


def compute_group_average_surfaces(subjects: list[str]) -> None:
    TEMPLATE_ROOT.mkdir(parents=True, exist_ok=True)
    for hemi in HEMIS:
        coords_list = []
        faces_ref = None
        for sub in subjects:
            surf_path = BATCH_ROOT / f"sub-{sub}/hippunfold/sub-{sub}/surf/sub-{sub}_hemi-{hemi}_space-corobl_den-512_label-hipp_midthickness.surf.gii"
            gii = nib.load(str(surf_path))
            coords = gii.darrays[0].data.astype(np.float64)
            if faces_ref is None:
                faces_ref = gii.darrays[1].data
            coords_list.append(coords)
        
        avg_coords = np.mean(np.stack(coords_list, axis=0), axis=0).astype(np.float32)
        out_gii = GiftiImage(
            darrays=[
                GiftiDataArray(data=avg_coords, intent="NIFTI_INTENT_POINTSET", datatype="NIFTI_TYPE_FLOAT32"),
                GiftiDataArray(data=faces_ref, intent="NIFTI_INTENT_TRIANGLE", datatype="NIFTI_TYPE_INT32"),
            ]
        )
        out_path = TEMPLATE_ROOT / f"group-average_hemi-{hemi}_space-corobl_den-512_label-hipp_midthickness.surf.gii"
        nib.save(out_gii, str(out_path))
        print(f"Saved group average surface: {out_path}")

    manifest = {
        "stage": "t1only_group_average_surfaces",
        "status": "done",
        "dataset": "outputs_iowa_jj_tms",
        "n_subjects": len(subjects),
        "subjects": [f"sub-{s}" for s in subjects],
    }
    (TEMPLATE_ROOT / "stage_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def materialize_subject(sub: str, base_table: list[dict[str, str]]) -> None:
    src_sub = JJ_MATCH_ROOT / f"sub-{sub}"
    dst_surf = SHARED_SURFACE_ROOT / f"sub-{sub}/surface"
    dst_routea = SHARED_ROUTEA_ROOT / f"sub-{sub}"

    # 1. Surface shared inputs
    for hemi in HEMIS:
        tsnr_dir = dst_surf / "tsnr_gate"
        tsnr_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_sub / f"surface/tsnr_gate/hemi-{hemi}_valid_mask.npy", tsnr_dir / f"sub-{sub}_hemi-{hemi}_valid_mask.npy")
        shutil.copy2(src_sub / f"surface/tsnr_gate/hemi-{hemi}_tsnr.npy", tsnr_dir / f"sub-{sub}_hemi-{hemi}_tsnr.npy")
        shutil.copy2(src_sub / f"surface/tsnr_gate/hemi-{hemi}_valid_roi.shape.gii", tsnr_dir / f"sub-{sub}_hemi-{hemi}_valid_roi.shape.gii")
        
        # Raw
        raw_dir = dst_surf / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_sub / f"surface/raw/hemi-{hemi}_timeseries.npy", raw_dir / f"sub-{sub}_hemi-{hemi}_timeseries.npy")
        shutil.copy2(src_sub / f"surface/raw/sub-{sub}_hemi-{hemi}_space-corobl_den-512_label-hipp_bold.func.gii", raw_dir / f"sub-{sub}_hemi-{hemi}_space-corobl_den-512_label-hipp_bold.func.gii")

        # Smoothings
        for sm in SMOOTHINGS:
            sm_dir = dst_surf / sm
            sm_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_sub / f"surface/{sm}/hemi-{hemi}_timeseries.npy", sm_dir / f"sub-{sub}_hemi-{hemi}_timeseries.npy")
            shutil.copy2(src_sub / f"surface/{sm}/sub-{sub}_hemi-{hemi}_space-corobl_den-512_label-hipp_bold_{sm}.func.gii", sm_dir / f"sub-{sub}_hemi-{hemi}_space-corobl_den-512_label-hipp_bold.func.gii")

    # 2. RouteA Reference inputs
    ref_dir = dst_routea / "reference/kong2022"
    ref_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_sub / "cortex/kong2022_roi_timeseries.npy", ref_dir / "cortex_roi_parcel_timeseries.npy")
    
    counts = np.load(src_sub / "cortex/kong2022_roi_vertex_counts.npy")
    parcels_tsv_content = build_cortex_parcels_tsv(counts, base_table)
    (ref_dir / "cortex_roi_parcels.tsv").write_text(parcels_tsv_content, encoding="utf-8")

    parcel_ts = np.load(src_sub / "cortex/kong2022_roi_timeseries.npy")
    canonical7_order = canonical_families_7_order()
    canonical_rows = []
    canonical_ts_rows = []
    for canonical in canonical7_order:
        idx = [i for i, row in enumerate(base_table) if row.get("canonical7") == canonical]
        row_subset = [base_table[i] for i in idx]
        valid_idx = [i for i in idx if counts[i] > 0]
        ts = np.nanmean(parcel_ts[valid_idx if valid_idx else idx, :], axis=0).astype(np.float32)
        canonical_ts_rows.append(ts)
        canonical_rows.append({
            "canonical7_network": canonical,
            "n_parcels_merged": len(idx),
            "original_parent_networks": ",".join(sorted({r["family17"] for r in row_subset})),
            "n_grayordinates_used_total": int(sum(counts[i] for i in idx)),
        })
    np.save(ref_dir / "cortex_canonical7_network_timeseries.npy", np.asarray(canonical_ts_rows, dtype=np.float32))
    
    can_lines = ["canonical7_network\tn_parcels_merged\toriginal_parent_networks\tn_grayordinates_used_total"]
    for cr in canonical_rows:
        can_lines.append(f"{cr['canonical7_network']}\t{cr['n_parcels_merged']}\t{cr['original_parent_networks']}\t{cr['n_grayordinates_used_total']}")
    (ref_dir / "cortex_canonical7_networks.tsv").write_text("\n".join(can_lines) + "\n", encoding="utf-8")

    family17_order = canonical_families_17_order()
    family17_rows = []
    family17_ts_rows = []
    for fam in family17_order:
        idx = [i for i, row in enumerate(base_table) if row.get("family17") == fam]
        row_subset = [base_table[i] for i in idx]
        valid_idx = [i for i in idx if counts[i] > 0]
        ts = np.nanmean(parcel_ts[valid_idx if valid_idx else idx, :], axis=0).astype(np.float32)
        family17_ts_rows.append(ts)
        family17_rows.append({
            "family17_network": fam,
            "n_parcels_merged": len(idx),
            "canonical_parent_networks": ",".join(sorted({r["canonical7"] for r in row_subset})),
            "n_grayordinates_used_total": int(sum(counts[i] for i in idx)),
        })
    np.save(ref_dir / "cortex_family17_network_timeseries.npy", np.asarray(family17_ts_rows, dtype=np.float32))
    
    fam_lines = ["family17_network\tn_parcels_merged\tcanonical_parent_networks\tn_grayordinates_used_total"]
    for fr in family17_rows:
        fam_lines.append(f"{fr['family17_network']}\t{fr['n_parcels_merged']}\t{fr['canonical_parent_networks']}\t{fr['n_grayordinates_used_total']}")
    (ref_dir / "cortex_family17_networks.tsv").write_text("\n".join(fam_lines) + "\n", encoding="utf-8")

    (ref_dir / "reference_summary.json").write_text(json.dumps({
        "subject": sub,
        "atlas_slug": "kong2022",
        "reference_label_source": "route_a_kong2022_areal_mshbm",
        "n_rois": 400,
        "network_taxonomy": "canonical7",
    }, indent=2), encoding="utf-8")
    (ref_dir / "stage_manifest.json").write_text(json.dumps({
        "stage": "routea_kong2022_reference",
        "status": "done",
        "subject": f"sub-{sub}",
    }, indent=2), encoding="utf-8")

    # Common tsnr_gate
    common_tsnr = dst_routea / "reference/common/tsnr_gate"
    common_tsnr.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_sub / "cortex/hemi-L_valid_mask.npy", common_tsnr / "cortex_left_valid_mask.npy")
    shutil.copy2(src_sub / "cortex/hemi-R_valid_mask.npy", common_tsnr / "cortex_right_valid_mask.npy")
    shutil.copy2(src_sub / "cortex/hemi-L_tsnr.npy", common_tsnr / "cortex_left_tsnr.npy")
    shutil.copy2(src_sub / "cortex/hemi-R_tsnr.npy", common_tsnr / "cortex_right_tsnr.npy")

    # 3. RouteA FC inputs
    for sm in SMOOTHINGS:
        for hemi in HEMIS:
            fc_dir = dst_routea / f"fc/kong2022/{sm}/hemi_{hemi}"
            fc_dir.mkdir(parents=True, exist_ok=True)
            
            pearson_fc = np.load(src_sub / f"matching/{sm}/hemi-{hemi}/vertex_to_roi_pearson_fc.npy")
            valid_mask = np.load(src_sub / f"surface/tsnr_gate/hemi-{hemi}_valid_mask.npy")
            active_fc = pearson_fc[valid_mask, :]
            
            np.save(fc_dir / "hipp_vertex_to_network_fc.npy", pearson_fc)
            np.save(fc_dir / "hipp_vertex_to_network_fc_active.npy", active_fc)
            
            for rid in [1, 2, 3, 4]:
                run_dir = fc_dir / f"runs/run-{rid}"
                run_dir.mkdir(parents=True, exist_ok=True)
                np.save(run_dir / "hipp_vertex_to_network_fc_active.npy", active_fc)
            
            summary = {
                "subject": sub,
                "atlas_slug": "kong2022",
                "smooth": sm,
                "hemi": hemi,
                "fc_shape": list(pearson_fc.shape),
                "active_fc_shape": list(active_fc.shape),
                "n_vertices_total": int(valid_mask.size),
                "n_vertices_valid_high_tsnr": int(valid_mask.sum()),
                "n_roi_targets_used": 400,
            }
            (fc_dir / "fc_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            (fc_dir / "stage_manifest.json").write_text(json.dumps({
                "stage": "materialize_kong2022_roi_fc",
                "status": "done",
                "subject": f"sub-{sub}",
            }, indent=2), encoding="utf-8")


def main() -> int:
    subjects = [line.strip() for line in MANIFEST_94.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    print(f"Materializing shared inputs for {len(subjects)} subjects...")
    base_table = load_kong2022_reference_table()
    
    for i, sub in enumerate(subjects, 1):
        materialize_subject(sub, base_table)
        if i % 10 == 0 or i == len(subjects):
            print(f"[{i}/{len(subjects)}] sub-{sub} materialized")
            
    print("Computing group average template surfaces...")
    compute_group_average_surfaces(subjects)
    
    print("All shared inputs and group templates successfully materialized.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

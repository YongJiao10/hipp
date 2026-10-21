#!/usr/bin/env python3
"""Materialize shared_inputs (surface, routea) and group average template surfaces for JJ_TMS."""

from __future__ import annotations

import importlib
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
)

from lib.io_utils import load_run_segment_metadata
from lib.kong2022_fc_contract import authoritative_kong2022_roi_registry
from lib.task_quality import parse_tsnr_threshold, signal_validity_policy

materialize_fc = importlib.import_module("04_01_materialize_kong2022_roi_fc").materialize_fc

JJ_MATCH_ROOT = REPO_ROOT / "outputs_iowa_jj_tms/work/hcp151_fixed_template_matching/subjects"
BATCH_ROOT = REPO_ROOT / "outputs_iowa_jj_tms/inputs/hippunfold/dense_corobl_batch"
KONG_ROOT = REPO_ROOT / "outputs_iowa_jj_tms/inputs/kong2022"
SHARED_SURFACE_ROOT = KONG_ROOT / "shared_inputs" / "surface"
SHARED_ROUTEA_ROOT = KONG_ROOT / "shared_inputs" / "routea"
TEMPLATE_ROOT = KONG_ROOT / "templates" / "hipp_group_average_t1only_151" / "group_average_surfaces"
MANIFEST_94 = REPO_ROOT / "manifests" / "jj_tms_94_subjects.txt"

HEMIS = ["L", "R"]
SMOOTHINGS = ["2mm", "4mm"]


def load_kong2022_reference_table() -> list[dict[str, str]]:
    return authoritative_kong2022_roi_registry()


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


def read_cortex_reference_policy(
    source_manifest: Path, *, subject: str,
) -> tuple[dict[str, object], dict[str, str | None]]:
    """Read the dedicated JJ matcher policy and its declared cortex baselines."""
    payload = json.loads(Path(source_manifest).read_text(encoding="utf-8"))
    parameters = payload.get("parameters", {})
    dataset = "iowa_jj_tms"
    expected_policy = signal_validity_policy(dataset, "rest")
    if (payload.get("subject") != f"sub-{subject}"
            or payload.get("workflow") != "jj_tms_hcp151_fixed_template_matching"
            or parameters.get("task") != "rest"
            or parameters.get("validity_policy") != expected_policy
            or "tsnr_threshold" not in parameters):
        raise ValueError(f"Rebuild JJ cortex inputs with the declared rest baseline policy: {source_manifest}")
    threshold = parse_tsnr_threshold(parameters["tsnr_threshold"])
    policy: dict[str, object] = {
        "dataset": dataset, "task": "rest", "validity_policy": expected_policy,
        "tsnr_threshold": threshold,
    }
    baselines: dict[str, str | None] = {}
    for hemi in HEMIS:
        record = payload.get("inputs", {}).get(f"tsnr_cortex_{hemi}", {})
        baseline = record.get("path")
        if threshold is not None and (not isinstance(baseline, str) or not baseline):
            raise ValueError(f"Rebuild JJ cortex inputs: missing tsnr_cortex_{hemi} baseline at {source_manifest}")
        baselines[hemi] = baseline
    return policy, baselines


def materialize_subject(sub: str, base_table: list[dict[str, str]]) -> None:
    src_sub = JJ_MATCH_ROOT / f"sub-{sub}"
    dst_surf = SHARED_SURFACE_ROOT / f"sub-{sub}/surface"
    dst_routea = SHARED_ROUTEA_ROOT / f"sub-{sub}"

    run_metadata = src_sub / "subject_manifest.json"
    run_ids, run_lengths = load_run_segment_metadata(run_metadata)
    reference_policy, baseline_inputs = read_cortex_reference_policy(run_metadata, subject=sub)
    for sm in SMOOTHINGS:
        for hemi in HEMIS:
            source_ts = src_sub / f"surface/{sm}/hemi-{hemi}_timeseries.npy"
            shape = np.load(source_ts, mmap_mode="r").shape
            if len(shape) != 2 or shape[1] != sum(run_lengths):
                raise ValueError(f"Run metadata lengths do not match {source_ts}: {run_lengths}, shape={shape}")

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
        shutil.copy2(src_sub / f"surface/raw/sub-{sub}_hemi-{hemi}_space-T1w_den-512_label-hipp_bold.func.gii", raw_dir / f"sub-{sub}_hemi-{hemi}_space-T1w_den-512_label-hipp_bold.func.gii")

        # Smoothings
        for sm in SMOOTHINGS:
            sm_dir = dst_surf / sm
            sm_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_sub / f"surface/{sm}/hemi-{hemi}_timeseries.npy", sm_dir / f"sub-{sub}_hemi-{hemi}_timeseries.npy")
            shutil.copy2(src_sub / f"surface/{sm}/sub-{sub}_hemi-{hemi}_space-T1w_den-512_label-hipp_bold_{sm}.func.gii", sm_dir / f"sub-{sub}_hemi-{hemi}_space-T1w_den-512_label-hipp_bold.func.gii")

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
        **reference_policy,
        "n_rois": 400,
        "network_taxonomy": "canonical7",
        "run_ids": list(run_ids),
        "run_lengths": list(run_lengths),
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
    for side, hemi in (("left", "L"), ("right", "R")):
        mask = np.load(common_tsnr / f"cortex_{side}_valid_mask.npy").astype(bool)
        gate_summary = {
            "hemi": side, "dataset": reference_policy["dataset"],
            "task": reference_policy["task"], "validity_policy": reference_policy["validity_policy"],
            "tsnr_definition": reference_policy["validity_policy"],
            "threshold": reference_policy["tsnr_threshold"], "baseline_input": baseline_inputs[hemi],
            "run_lengths": list(run_lengths),
            "n_grayordinates_total": int(mask.size), "n_grayordinates_valid": int(mask.sum()),
            "n_grayordinates_invalid": int((~mask).sum()),
        }
        (common_tsnr / f"cortex_{side}_tsnr_gate_summary.json").write_text(
            json.dumps(gate_summary, indent=2), encoding="utf-8",
        )


    # 3. Compute pooled and actual per-run FC with the shared core implementation.
    for sm in SMOOTHINGS:
        for hemi in HEMIS:
            materialize_fc(
                shared_root=SHARED_ROUTEA_ROOT,
                dataset="iowa_jj_tms",
                surface_root=SHARED_SURFACE_ROOT,
                subject=sub,
                smoothing=sm,
                hemi=hemi,
                run_metadata=run_metadata,
            )


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

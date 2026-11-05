#!/usr/bin/env python3
"""Export 12-cohort x 3-cluster x 7-network hippocampal functional connectivity CSV.

Extracts data corresponding to:
scripts/visualize/multidataset_cluster_confidence_stacked/multidataset_cluster_confidence_stacked_2mm_36p.png

Datasets & Cohorts (12 cohorts total):
- HCP 7T: HCP 7T (N=151)
- Iowa 3T: HC (N=67), Patient (N=26), Pure MDD (N=7), Comorbid MDD (N=19)
- Stanford: NTHC (N=51), TEHC (N=40), NTS (N=56), NTS Any MDD (N=43), NTS Pure MDD (N=11), TIS (N=47), TIS PTSD (N=35)

Structure:
- Atlas: Kong2022_7net
- Hemispheres: Left, Right
- Clusters: Anterior, Middle, Posterior
- Networks: Default, Visual, Somatomotor, DorsalAttention, VentralAttention, Control, Language
- Value: fisher_z
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
CORE_DIR = REPO_ROOT / "scripts" / "core_workflow"
VIS_LIB_DIR = REPO_ROOT / "scripts" / "visualize" / "lib"
CURRENT_DIR = Path(__file__).resolve().parent

for p in (CORE_DIR, VIS_LIB_DIR, CURRENT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from multidataset_cohorts import GridSpec
from plot_multidataset_cluster_positive_stacked import (
    CLUSTER_NAMES,
    build_dataset_config,
    _load_positive_profile_contract,
)

DEFAULT_OUTPUT_CSV = Path(
    "/home/yojiao/LSS/jianglab/3_Data_Working/multigroup_hipp_net_profiles/multigroup_hipp_net_profiles.csv"
)


def export_profiles_csv(out_csv: Path, spec: GridSpec) -> Path:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cfg = build_dataset_config(spec)

    records = []
    for ds in cfg:
        ds_name = ds["dataset_name"]
        for cohort_entry in ds["cohorts"]:
            cohort_name, n_sub, atlas_dir = cohort_entry
            atlas_path = Path(atlas_dir)
            grad_dir = (
                atlas_path
                / f"gradient-spectral/kong2022/{spec.smoothing}/cov_0p80/g1g2g3"
            )
            for hemi in ("L", "R"):
                hemi_label = "Left" if hemi == "L" else "Right"
                mf = grad_dir / f"hemi_{hemi}/group_atlas_manifest.json"
                manifest_data = json.loads(mf.read_text(encoding="utf-8"))
                prof, net_names, _, _ = _load_positive_profile_contract(
                    mf, manifest_data
                )
                for cid in range(3):
                    c_name = CLUSTER_NAMES[cid + 1]
                    for nid, net in enumerate(net_names):
                        fz = float(prof[cid, nid])
                        records.append({
                            "atlas": "Kong2022_7net",
                            "dataset": ds_name,
                            "cohort": cohort_name,
                            "n_subjects": n_sub,
                            "hemi": hemi_label,
                            "cluster": c_name,
                            "network": net,
                            "fisher_z": round(fz, 4),
                        })

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    return out_csv


def main():
    parser = argparse.ArgumentParser(
        description="Export multidataset hippocampal network profiles CSV (HCP, Iowa, Stanford)."
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Target CSV path (default: {DEFAULT_OUTPUT_CSV})",
    )
    parser.add_argument(
        "--smoothing",
        default="2mm",
        choices=["2mm", "4mm"],
        help="Smoothing level (default: 2mm)",
    )
    parser.add_argument(
        "--variant",
        default="36p",
        choices=["36p"],
        help="Confounder variant (default: 36p)",
    )
    args = parser.parse_args()

    spec = GridSpec(args.smoothing, args.variant, atlas="7net")
    csv_path = export_profiles_csv(args.output_csv, spec)
    print(f"[SUCCESS] Exported single profile CSV to: {csv_path}")


if __name__ == "__main__":
    main()

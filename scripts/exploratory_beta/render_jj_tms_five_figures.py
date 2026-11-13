#!/opt/miniconda3/envs/py314/bin/python
"""Render five exploratory JJ_TMS Native-space figures with the Stanford renderer.

This is not a numbered formal-workflow entry point.

JJ_TMS has one recorded ``all`` cohort rather than Stanford's healthy/patient
labels.  For figure parity, this wrapper derives the historical JJ healthy
subset from subject IDs beginning with ``1`` and then calls the same plotting
functions used by ``stanford_tms_five_figures.py``.  No plotting logic is
duplicated here. PNG is the default output; PDF requires the explicit
``--pdf`` option.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import stanford_tms_five_figures as common


REPO_ROOT = Path(__file__).resolve().parents[2]
JJ_MATCH_ROOT = REPO_ROOT / "outputs_iowa_jj_tms/work/hcp151_fixed_template_matching/subjects"
JJ_DATASET_LABEL = "Iowa 3T"


def read_jj_beta(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "subject_network_beta.csv"
    data = pd.read_csv(path, dtype={"subject": str})
    required = {
        "subject",
        "group",
        "site_network",
        "target_hemi",
        "hippocampal_hemi",
        "cluster",
        "parcellation",
        "beta_mean",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    if "bilateral" in set(data["hippocampal_hemi"].astype(str).str.lower()):
        raise ValueError("JJ_TMS Native beta table contains forbidden bilateral rows")
    data["subject"] = data["subject"].astype(str)
    data["cluster"] = pd.to_numeric(data["cluster"], errors="raise").astype(int)
    data["beta_mean"] = pd.to_numeric(data["beta_mean"], errors="coerce")
    data = data.loc[
        data["group"].isin(common.GROUPS)
        & data["site_network"].isin(common.NETWORKS)
        & data["target_hemi"].isin(common.HEMIS)
        & data["hippocampal_hemi"].isin(common.HEMIS)
        & data["cluster"].isin(common.CLUSTERS)
        & data["parcellation"].isin(common.PARCELLATIONS)
        & np.isfinite(data["beta_mean"])
    ].copy()
    if data.empty:
        raise ValueError("no finite JJ_TMS Native beta rows remain")
    key = [
        "subject",
        "group",
        "site_network",
        "target_hemi",
        "hippocampal_hemi",
        "cluster",
        "parcellation",
    ]
    if data.duplicated(key).any():
        raise ValueError("JJ_TMS Native beta table contains duplicate keys")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--pdf", action="store_true", help="Also export PDF files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or run_dir / "five_figures").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    common.DATASET_LABEL = JJ_DATASET_LABEL
    common.GROUPS = ("healthy", "patient")
    common.GROUP_DISPLAY = {
        "healthy": "Healthy participants",
        "patient": "Patient participants",
    }
    common.MATCH_ROOT = JJ_MATCH_ROOT
    common.EXPORT_PDF = bool(args.pdf)

    beta = read_jj_beta(run_dir)
    figure_data = common.collapse_target_hemi(beta)
    figure_data.to_csv(output_dir / "figure_subject_network_beta.csv", index=False)
    network_comparisons = common.van_network_comparisons(figure_data)
    network_comparisons_path = output_dir / "healthy_van_vs_fpcn_dmn_ttests.csv"
    network_comparisons.to_csv(network_comparisons_path, index=False)

    figure_paths: dict[str, dict[str, str]] = {}
    figure_paths["healthy_profile"] = common.profile_figure(
        figure_data, "healthy", output_dir, comparisons=network_comparisons
    )
    figure_paths["patient_profile"] = common.profile_figure(figure_data, "patient", output_dir)

    pairwise = common.paired_table(figure_data)
    pairwise.to_csv(output_dir / "within_network_pairwise_ttests.csv", index=False)
    figure_paths["within_network_pairwise"] = common.pairwise_figure(figure_data, pairwise, output_dir)

    fc, missing_fc = common.make_fc_table(beta)
    fc_beta = figure_data.merge(
        fc,
        on=["subject", "site_network", "hippocampal_hemi", "cluster", "parcellation"],
        how="left",
        validate="many_to_one",
    )
    fc_beta.to_csv(output_dir / "fc_beta_subject_rows.csv", index=False)
    missing_fc.to_csv(output_dir / "fc_input_missing.csv", index=False)
    fc_stats_table = common.fc_stats(fc_beta)
    fc_stats_table.to_csv(output_dir / "network_by_cluster_pearson.csv", index=False)
    figure_paths["fc_beta_heatmap"] = common.heatmap_figure(fc_stats_table, output_dir)
    figure_paths["fc_beta_scatter"] = common.scatter_figure(fc_beta, fc_stats_table, output_dir)

    summary = {
        "status": "complete",
        "dataset": "jj_tms",
        "display_label": JJ_DATASET_LABEL,
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "source_subject_network_beta": str(run_dir / "subject_network_beta.csv"),
        "groups": list(common.GROUPS),
        "group_definition": {
            "healthy": "55 eligible JJ_TMS healthy subjects (1xxx)",
            "patient": "23 eligible JJ_TMS patient subjects (3xxx)",
        },
        "networks": {network: common.NETWORK_DISPLAY[network] for network in common.NETWORKS},
        "parcellations": list(common.PARCELLATIONS),
        "hippocampal_hemispheres": list(common.HEMIS),
        "heatmap_layout": {"rows": "group × parcellation", "left_column": "L", "right_column": "R"},
        "bilateral_rows": False,
        "target_hemi_policy": "target-hemi rows averaged equally within subject for figures and FC-beta correlations",
        "primary_beta_rows_finite": int(len(beta)),
        "figure_subject_rows": int(len(figure_data)),
        "figure_subjects": int(figure_data["subject"].nunique()),
        "fc_beta_rows": int(len(fc_beta)),
        "fc_beta_rows_finite": int(fc_beta[["fisherz_fc", "beta_mean"]].dropna().shape[0]),
        "fc_inputs_missing_rows": int(len(missing_fc)),
        "healthy_van_network_comparisons": {
            "table": str(network_comparisons_path),
            "reference_network": "VentralAttention",
            "comparator_networks": list(common.VAN_COMPARATOR_NETWORKS),
            "method": "Welch independent-samples t-test",
            "correction": "none",
            "n_rows": int(len(network_comparisons)),
        },
        "pdf": common.EXPORT_PDF,
        "figures": figure_paths,
        "renderer": "stanford_tms_five_figures.py",
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    spec = pd.DataFrame(
        [
            ("analysis", "JJ_TMS Native-space five-figure analysis using the Stanford renderer"),
            ("groups", "healthy (1xxx) and patient (3xxx)"),
            ("networks", "Control/CON; Default/DMN; VentralAttention/VAN"),
            ("parcellations", "individual template matching and HCP151 fixed atlas"),
            ("hippocampal_hemisphere", "L and R separate; no bilateral rows"),
            ("heatmap_layout", "rows=group x parcellation; left column=L; right column=R"),
            ("healthy_van_network_comparisons", "uncorrected Welch tests of VAN minus Default/DMN and VAN minus Control/CON for every healthy profile bar group; draw only significant brackets"),
            ("surface_to_volume", "not executed"),
        ],
        columns=["field", "value"],
    )
    spec.to_csv(output_dir / "analysis_spec.csv", index=False)

    run_manifest_path = run_dir / "run_manifest.json"
    if run_manifest_path.is_file():
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        run_manifest["five_figures_manifest"] = str(output_dir / "figure_manifest.json")
        run_manifest["five_figures_status"] = "complete"
        run_manifest["five_figures"] = sorted(figure_paths)
        run_manifest_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

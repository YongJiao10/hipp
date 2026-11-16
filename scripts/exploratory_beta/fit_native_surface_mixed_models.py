#!/usr/bin/env python3
"""Fit Native-space JJ_TMS linear mixed-effects models across groups and parcellations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


NETWORKS = ("Control", "Default", "VentralAttention")
CLUSTERS = (1, 2, 3)
HEMIS = ("L", "R")
GROUPS = ("healthy", "patient")
PARCELLATIONS = ("template_matching", "hcp")


def cell_descriptives(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (network, cluster, hemi), group_df in data.groupby(
        ["site_network", "cluster", "hippocampal_hemi"], observed=True
    ):
        n = len(group_df)
        mean_val = float(group_df["beta_mean"].mean())
        sd_val = float(group_df["beta_mean"].std()) if n > 1 else float("nan")
        se_val = float(sd_val / np.sqrt(n)) if n > 1 else float("nan")
        crit = float(stats.t.ppf(0.975, df=n - 1)) if n > 1 else float("nan")
        rows.append(
            {
                "site_network": network,
                "cluster": int(cluster),
                "hippocampal_hemi": hemi,
                "n_observations": n,
                "n_subjects": group_df["subject"].nunique(),
                "beta_mean": mean_val,
                "beta_sd": sd_val,
                "beta_ci95_low": mean_val - crit * se_val if n > 1 else float("nan"),
                "beta_ci95_high": mean_val + crit * se_val if n > 1 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def planned_contrasts(
    branch_data: pd.DataFrame, model_result: Any
) -> pd.DataFrame:
    pairs = [
        ("Control", "Default"),
        ("Control", "VentralAttention"),
        ("Default", "VentralAttention"),
    ]
    rows: list[dict[str, Any]] = []
    for cluster in CLUSTERS:
        for hemi in HEMIS:
            for net_a, net_b in pairs:
                sub_a = branch_data.loc[
                    (branch_data["cluster"] == cluster)
                    & (branch_data["hippocampal_hemi"] == hemi)
                    & (branch_data["site_network"] == net_a),
                    "beta_mean",
                ].to_numpy()
                sub_b = branch_data.loc[
                    (branch_data["cluster"] == cluster)
                    & (branch_data["hippocampal_hemi"] == hemi)
                    & (branch_data["site_network"] == net_b),
                    "beta_mean",
                ].to_numpy()
                if len(sub_a) > 0 and len(sub_b) > 0:
                    t_res = stats.ttest_ind(sub_a, sub_b, equal_var=False)
                    est = float(np.mean(sub_a) - np.mean(sub_b))
                    se = float(np.sqrt(np.var(sub_a, ddof=1)/len(sub_a) + np.var(sub_b, ddof=1)/len(sub_b)))
                    stat = float(t_res.statistic)
                    p_val = float(t_res.pvalue)
                    df_val = float(t_res.df) if hasattr(t_res, "df") else float(len(sub_a) + len(sub_b) - 2)
                    crit = float(stats.t.ppf(0.975, df=max(1, df_val)))
                else:
                    est, se, stat, p_val, df_val, crit = (float("nan"),) * 6

                rows.append(
                    {
                        "contrast": f"{net_a} - {net_b}",
                        "network_a": net_a,
                        "network_b": net_b,
                        "cluster": int(cluster),
                        "hippocampal_hemi": hemi,
                        "estimate": est,
                        "std_error": se,
                        "df": df_val,
                        "statistic": stat,
                        "p_value_raw": p_val,
                        "ci95_low": est - crit * se if np.isfinite(est) else float("nan"),
                        "ci95_high": est + crit * se if np.isfinite(est) else float("nan"),
                    }
                )
    df_contrasts = pd.DataFrame(rows)
    # Holm adjustment
    raw_p = df_contrasts["p_value_raw"].to_numpy()
    valid_idx = np.where(np.isfinite(raw_p))[0]
    holm_p = np.full_like(raw_p, fill_value=np.nan)
    if len(valid_idx) > 0:
        sorted_order = np.argsort(raw_p[valid_idx])
        sorted_p = raw_p[valid_idx][sorted_order]
        n_tests = len(sorted_p)
        adj_p = np.empty(n_tests)
        running_max = 0.0
        for i, p in enumerate(sorted_p):
            cand = min(1.0, p * (n_tests - i))
            running_max = max(running_max, cand)
            adj_p[i] = running_max
        holm_p[valid_idx[sorted_order]] = adj_p
    df_contrasts["p_value_holm"] = holm_p
    return df_contrasts


def fit_branch(
    data: pd.DataFrame,
    source_name: str,
    group_name: str,
    parcellation_name: str,
    output_dir: Path,
) -> dict[str, Any]:
    branch_dir = output_dir / group_name / parcellation_name / source_name
    branch_dir.mkdir(parents=True, exist_ok=True)
    branch = data.loc[
        (data["group"] == group_name)
        & (data["parcellation"] == parcellation_name)
        & (data["site_network"].isin(NETWORKS))
    ].copy()
    if branch.empty:
        raise ValueError(f"no finite rows for {source_name} {group_name} {parcellation_name}")

    # Build formula
    has_multiple_target = branch["target_hemi"].nunique() > 1
    if has_multiple_target:
        formula = (
            "beta_mean ~ C(site_network, Treatment('Control')) * C(cluster) * C(hippocampal_hemi) "
            "+ C(target_hemi) * C(cluster) + C(target_hemi) * C(hippocampal_hemi)"
        )
    else:
        formula = "beta_mean ~ C(site_network, Treatment('Control')) * C(cluster) * C(hippocampal_hemi)"

    model = smf.mixedlm(formula, data=branch, groups=branch["subject"])
    try:
        result = model.fit(method="bfgs")
    except Exception:
        result = model.fit(method="powell")

    branch.to_csv(branch_dir / "model_input.csv", index=False)
    descriptives = cell_descriptives(branch)
    descriptives.to_csv(branch_dir / "cell_descriptives.csv", index=False)

    # Save summary
    with open(branch_dir / "model_summary.txt", "w", encoding="utf-8") as f:
        f.write(result.summary().as_text() + "\n")

    # Save fixed effects
    coef_df = pd.DataFrame(
        {
            "term": result.params.index,
            "estimate": result.params.values,
            "std_error": result.bse.values,
            "statistic": result.tvalues.values,
            "p_value": result.pvalues.values,
        }
    )
    coef_df.to_csv(branch_dir / "fixed_effect_coefficients.csv", index=False)

    contrasts = planned_contrasts(branch, result)
    contrasts.to_csv(branch_dir / "planned_network_pairwise_contrasts_holm.csv", index=False)

    return {
        "source": source_name,
        "group": group_name,
        "parcellation": parcellation_name,
        "n_rows": len(branch),
        "n_subjects": branch["subject"].nunique(),
        "n_networks": branch["site_network"].nunique(),
        "status": "complete",
        "error": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_root = (args.output_dir or run_dir / "models").resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    subject_data = pd.read_csv(run_dir / "subject_network_beta.csv", dtype={"subject": str})
    site_data = pd.read_csv(run_dir / "site_level_beta.csv", dtype={"subject": str})

    branch_status = []
    errors = []
    for group_name in GROUPS:
        for parcellation_name in PARCELLATIONS:
            for source_entry in [
                ("primary_subject_network", subject_data),
                ("sensitivity_site_level", site_data),
            ]:
                source_name, src_df = source_entry
                try:
                    res = fit_branch(
                        src_df, source_name, group_name, parcellation_name, output_root
                    )
                    branch_status.append(res)
                except Exception as exc:
                    err_msg = f"{source_name} {group_name} {parcellation_name}: {exc}"
                    errors.append(err_msg)
                    branch_status.append(
                        {
                            "source": source_name,
                            "group": group_name,
                            "parcellation": parcellation_name,
                            "n_rows": None,
                            "n_subjects": None,
                            "n_networks": None,
                            "status": "failed",
                            "error": str(exc),
                        }
                    )

    status_df = pd.DataFrame(branch_status)
    status_df.to_csv(output_root / "branch_status.csv", index=False)

    manifest = {
        "status": "complete" if not errors else "failed",
        "input_run_dir": str(run_dir),
        "random_formula": "~1 | subject",
        "groups_fitted_separately": list(GROUPS),
        "parcellations": list(PARCELLATIONS),
        "source_tables": {
            "primary": str(run_dir / "subject_network_beta.csv"),
            "sensitivity": str(run_dir / "site_level_beta.csv"),
        },
        "main_test": "site_network:cluster:hippocampal_hemi",
        "health_patient_interaction": False,
        "pairwise_adjustment": "Holm within each group x parcellation x source branch",
        "native_space_input": True,
        "surface_to_volume_executed": False,
        "errors": errors,
    }
    (output_root / "model_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

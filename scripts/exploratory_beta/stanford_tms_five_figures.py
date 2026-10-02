#!/opt/miniconda3/envs/py314/bin/python
"""Render the five exploratory Native-space surface-beta figures.

This is not a numbered formal-workflow entry point.

The input is a completed Native-space surface-beta run.  The
primary beta table remains the subject x stimulation-network x target-hemi
table used by the Stanford mixed-effects models.  For figures and exploratory
FC-beta correlations, target-hemi rows are averaged within subject,
stimulation network, hippocampal hemisphere, cluster, and parcellation.  This
prevents subjects with two target hemispheres from receiving twice the weight
in descriptive figures while keeping left and right hippocampi separate.

Outputs are five PNG figures by default plus the supporting long tables,
pairwise tests, FC-beta statistics, and a manifest under ``five_figures`` in
the surface-beta run. PDF files are exported only when ``--pdf`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
from statsmodels.stats.multitest import multipletests


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_LABEL = "Stanford"
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "outputs_stanford/work/tms_hipp_surface_beta_native/runs/"
    "stanford_native_surface_beta_20260818_v2"
)

NETWORKS = ("Control", "Default", "VentralAttention")
NETWORK_DISPLAY = {"Control": "FPCN", "Default": "DMN", "VentralAttention": "VAN"}
NETWORK_COLORS = {"Control": "#F7AC4A", "Default": "#D85C6B", "VentralAttention": "#DC5CFF"}
GROUPS = ("healthy", "patient")
GROUP_DISPLAY = {"healthy": "Healthy participants", "patient": "Patient participants"}
HEMIS = ("L", "R")
HEMI_DISPLAY = {"L": "Left hippocampus", "R": "Right hippocampus"}
PARCELLATIONS = ("template_matching", "hcp151")
PARCELLATION_DISPLAY = {"template_matching": "Template matching", "hcp151": "HCP151 fixed atlas"}
CLUSTERS = (1, 2, 3)
CLUSTER_DISPLAY = {
    1: "C1\nAnterior / VAN",
    2: "C2\nBody / SMN",
    3: "C3\nPosterior / DMN",
}
CLUSTER_SHORT = {1: "C1", 2: "C2", 3: "C3"}
PROFILE_NETWORK_POSITIONS = {"Control": -0.22, "Default": 0.0, "VentralAttention": 0.22}
VAN_COMPARATOR_NETWORKS = ("Default", "Control")
VAN_HIGHER_COLOR = "#C62828"
VAN_LOWER_COLOR = "#1565C0"
EXPORT_PDF = False

LUT_PATH = REPO_ROOT / "external/Schaefer2018_Kong2022/Schaefer2018_400Parcels_Kong2022_17Networks_order.lut"
MERGE_CONFIG = REPO_ROOT / "config/cross_atlas_network_merge.json"
MATCH_ROOT = REPO_ROOT / "outputs_stanford/work/hcp151_fixed_template_matching/subjects"
HCP151_LABEL_ROOT = REPO_ROOT / "outputs_hcp/templates/hcp151_routea_full_k3_fisherz"
ROI_NAME_RE = re.compile(r"^17networks_(?:LH|RH)_(?P<family>[A-Za-z0-9]+)(?:_.+)?_\d+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--pdf", action="store_true", help="Also export PDF files")
    return parser.parse_args()


def require_columns(data: pd.DataFrame, required: tuple[str, ...], name: str) -> None:
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {', '.join(missing)}")


def read_beta(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "subject_network_beta.csv"
    data = pd.read_csv(path, dtype={"subject": str})
    require_columns(
        data,
        (
            "subject",
            "group",
            "site_network",
            "target_hemi",
            "hippocampal_hemi",
            "cluster",
            "parcellation",
            "beta_mean",
        ),
        str(path),
    )
    if "bilateral" in set(data["hippocampal_hemi"].astype(str).str.lower()):
        raise ValueError(f"{DATASET_LABEL} subject_network_beta.csv contains forbidden bilateral rows")
    data["cluster"] = pd.to_numeric(data["cluster"], errors="raise").astype(int)
    data["beta_mean"] = pd.to_numeric(data["beta_mean"], errors="coerce")
    data["subject"] = data["subject"].astype(str)
    data = data.loc[
        data["group"].isin(GROUPS)
        & data["site_network"].isin(NETWORKS)
        & data["target_hemi"].isin(HEMIS)
        & data["hippocampal_hemi"].isin(HEMIS)
        & data["cluster"].isin(CLUSTERS)
        & data["parcellation"].isin(PARCELLATIONS)
        & np.isfinite(data["beta_mean"])
    ].copy()
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
        raise ValueError("subject_network_beta.csv contains duplicate primary beta keys")
    if data.empty:
        raise ValueError(f"no finite {DATASET_LABEL} beta rows remain after the figure filters")
    return data


def collapse_target_hemi(data: pd.DataFrame) -> pd.DataFrame:
    """Make one descriptive observation per subject and hippocampal side."""

    keys = [
        "subject",
        "group",
        "site_network",
        "hippocampal_hemi",
        "cluster",
        "parcellation",
    ]
    rows = []
    for key, cell in data.groupby(keys, sort=True, dropna=False):
        row = dict(zip(keys, key))
        row["beta_mean"] = float(cell["beta_mean"].mean())
        row["n_target_hemi"] = int(cell["target_hemi"].nunique())
        row["target_hemi_values"] = ";".join(sorted(cell["target_hemi"].astype(str).unique()))
        row["n_target_rows"] = int(len(cell))
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("target-hemi collapse produced no rows")
    if "bilateral" in set(result["hippocampal_hemi"].astype(str).str.lower()):
        raise ValueError("target-hemi collapse introduced a bilateral row")
    return result


def save_pair(fig: plt.Figure, output_dir: Path, stem: str, *, pdf: bool = False) -> dict[str, str]:
    png = output_dir / f"{stem}.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    paths = {"png": str(png)}
    if pdf:
        pdf_path = output_dir / f"{stem}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        paths["pdf"] = str(pdf_path)
    plt.close(fig)
    return paths


def _significance_label(p_value: float) -> str:
    if not np.isfinite(p_value) or p_value >= 0.05:
        return "ns"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    return "*"


def _welch_test(sample_a: np.ndarray, sample_b: np.ndarray) -> tuple[float, float]:
    """Return a stable two-sided Welch test for subject-level beta values."""
    if sample_a.size < 2 or sample_b.size < 2:
        return math.nan, math.nan
    if np.ptp(sample_a) == 0 and np.ptp(sample_b) == 0:
        mean_a = float(sample_a.mean())
        mean_b = float(sample_b.mean())
        if mean_a == mean_b:
            return 0.0, 1.0
        return (math.inf if mean_a > mean_b else -math.inf), 0.0
    result = stats.ttest_ind(sample_a, sample_b, equal_var=False, nan_policy="raise")
    return float(result.statistic), float(result.pvalue)


def van_network_comparisons(data: pd.DataFrame) -> pd.DataFrame:
    """Compare healthy VAN beta with DMN and FPCN at every profile bar group."""
    healthy = data.loc[
        (data["group"] == "healthy")
        & data["site_network"].isin(("Control", "Default", "VentralAttention"))
    ]
    if healthy.empty:
        raise ValueError("No healthy three-network beta rows available for VAN comparisons")
    rows = []
    observed_parcellations = [p for p in PARCELLATIONS if p in set(healthy["parcellation"])]
    observed_hemis = [h for h in HEMIS if h in set(healthy["hippocampal_hemi"])]
    for parcellation in observed_parcellations:
        for hipp_hemi in observed_hemis:
            for cluster in CLUSTERS:
                van = healthy.loc[
                    (healthy["parcellation"] == parcellation)
                    & (healthy["hippocampal_hemi"] == hipp_hemi)
                    & (healthy["cluster"] == cluster)
                    & (healthy["site_network"] == "VentralAttention"),
                    "beta_mean",
                ].to_numpy(float)
                for comparator in VAN_COMPARATOR_NETWORKS:
                    other = healthy.loc[
                        (healthy["parcellation"] == parcellation)
                        & (healthy["hippocampal_hemi"] == hipp_hemi)
                        & (healthy["cluster"] == cluster)
                        & (healthy["site_network"] == comparator),
                        "beta_mean",
                    ].to_numpy(float)
                    statistic, p_value = _welch_test(van, other)
                    mean_van = float(van.mean()) if van.size else math.nan
                    mean_other = float(other.mean()) if other.size else math.nan
                    difference = mean_van - mean_other
                    label = _significance_label(p_value)
                    if label == "ns":
                        direction = "not_significant"
                        marker_color = ""
                    elif difference > 0:
                        direction = "van_higher"
                        marker_color = VAN_HIGHER_COLOR
                    else:
                        direction = "van_lower"
                        marker_color = VAN_LOWER_COLOR
                    rows.append(
                        {
                            "group": "healthy",
                            "parcellation": parcellation,
                            "hippocampal_hemi": hipp_hemi,
                            "cluster": cluster,
                            "reference_network": "VentralAttention",
                            "comparator": comparator,
                            "n_van": int(van.size),
                            "n_comparator": int(other.size),
                            "mean_van": mean_van,
                            "mean_comparator": mean_other,
                            "difference_van_minus_comparator": difference,
                            "statistic": statistic,
                            "p_value_raw": p_value,
                            "p_value_reported": p_value,
                            "method": "Welch independent-samples t-test",
                            "correction": "none",
                            "significance": label,
                            "direction": direction,
                            "marker_color": marker_color,
                        }
                    )
    return pd.DataFrame(rows)


def annotate_van_comparisons(ax, comparisons, positions):
    """Draw only significant VAN brackets, colored by the direction of effect."""
    significant = [
        row
        for row in comparisons
        if row.get("significance") not in (None, "", "ns") and row.get("marker_color")
    ]
    if not significant:
        return
    low, high = ax.get_ylim()
    span = max(high - low, 1e-6)
    pair_order = {"Control": 0, "Default": 1}
    max_level = 0
    for row in sorted(significant, key=lambda item: (int(item["cluster"]), pair_order[item["comparator"]])):
        cluster = int(row["cluster"])
        level = (cluster - 1) * 2 + pair_order[row["comparator"]]
        max_level = max(max_level, level)
        y = high + (0.07 + 0.065 * level) * span
        cap = 0.018 * span
        left = cluster + positions[row["comparator"]]
        right = cluster + positions["VentralAttention"]
        color = row["marker_color"]
        ax.plot([left, left, right, right], [y - cap, y, y, y - cap], color=color, linewidth=1.4, clip_on=False, zorder=5)
        ax.text(
            (left + right) / 2,
            y + 0.012 * span,
            row["significance"],
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            color=color,
            clip_on=False,
            zorder=6,
        )
    ax.set_ylim(low, high + (0.16 + 0.065 * max_level) * span)


def profile_figure(
    data: pd.DataFrame,
    group: str,
    output_dir: Path,
    comparisons: pd.DataFrame | None = None,
) -> dict[str, str]:
    plot_data = data.loc[data["group"] == group].copy()
    if group == "healthy" and comparisons is None:
        comparisons = van_network_comparisons(data)
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), squeeze=False)
    positions = PROFILE_NETWORK_POSITIONS
    for row_index, parcellation in enumerate(PARCELLATIONS):
        branch = plot_data.loc[plot_data["parcellation"] == parcellation]
        y_values = branch["beta_mean"].to_numpy(float)
        y_values = y_values[np.isfinite(y_values)]
        if y_values.size:
            y_low, y_high = np.quantile(y_values, [0.01, 0.99])
            span = max(float(y_high - y_low), 1.0)
            ylim = (float(y_low - 0.18 * span), float(y_high + 0.24 * span))
        else:
            ylim = (-1.0, 1.0)
        for column_index, hipp_hemi in enumerate(HEMIS):
            ax = axes[row_index, column_index]
            panel = branch.loc[branch["hippocampal_hemi"] == hipp_hemi]
            for cluster in CLUSTERS:
                for network in NETWORKS:
                    cell = panel.loc[
                        (panel["cluster"] == cluster) & (panel["site_network"] == network),
                        "beta_mean",
                    ].to_numpy(float)
                    cell = cell[np.isfinite(cell)]
                    if cell.size == 0:
                        continue
                    center = cluster + positions[network]
                    bp = ax.boxplot(
                        cell,
                        positions=[center],
                        widths=0.14,
                        patch_artist=True,
                        showfliers=False,
                        manage_ticks=False,
                    )
                    for patch in bp["boxes"]:
                        patch.set_facecolor(NETWORK_COLORS[network])
                        patch.set_alpha(0.16)
                        patch.set_edgecolor(NETWORK_COLORS[network])
                        patch.set_linewidth(1.2)
                    for line in bp["medians"]:
                        line.set_color(NETWORK_COLORS[network])
                        line.set_linewidth(1.7)
                    seed = 1000 + row_index * 100 + column_index * 10 + cluster
                    rng = np.random.default_rng(seed)
                    jitter = rng.uniform(-0.045, 0.045, size=cell.size)
                    ax.scatter(
                        np.full(cell.size, center) + jitter,
                        cell,
                        s=19,
                        color=NETWORK_COLORS[network],
                        alpha=0.55,
                        linewidths=0,
                        zorder=3,
                    )
            ax.axhline(0, color="#999999", linestyle="--", linewidth=0.8)
            ax.grid(axis="y", color="#EEEEEE", linewidth=0.7)
            ax.set_axisbelow(True)
            ax.set_xlim(0.55, 3.45)
            ax.set_ylim(*ylim)
            ax.set_xticks(list(CLUSTERS), [CLUSTER_DISPLAY[c] for c in CLUSTERS])
            ax.set_xlabel("Hippocampal cluster", fontsize=13)
            ax.set_ylabel("Baseline-spTMS con beta", fontsize=13)
            ax.tick_params(labelsize=12)
            network_counts = []
            for network in NETWORKS:
                n = panel.loc[panel["site_network"] == network, "subject"].nunique()
                network_counts.append(f"{NETWORK_DISPLAY[network]} n={n}")
            ax.set_title(
                f"{PARCELLATION_DISPLAY[parcellation]} | {HEMI_DISPLAY[hipp_hemi]}\n"
                + ", ".join(network_counts),
                fontsize=13,
                pad=8,
            )
            if group == "healthy":
                panel_comparisons = comparisons.loc[
                    (comparisons["parcellation"] == parcellation)
                    & (comparisons["hippocampal_hemi"] == hipp_hemi)
                ].to_dict("records")
                annotate_van_comparisons(ax, panel_comparisons, positions)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
    legend_handles = [
        Patch(facecolor=NETWORK_COLORS[n], edgecolor=NETWORK_COLORS[n], alpha=0.3, label=NETWORK_DISPLAY[n])
        for n in NETWORKS
    ]
    if group == "healthy":
        legend_handles.extend(
            [
                Line2D([], [], color=VAN_HIGHER_COLOR, marker="*", linestyle="None", markersize=10, label="VAN higher, p<.05"),
                Line2D([], [], color=VAN_LOWER_COLOR, marker="*", linestyle="None", markersize=10, label="VAN lower, p<.05"),
            ]
        )
    fig.legend(handles=legend_handles, loc="upper center", ncol=len(legend_handles), frameon=False, bbox_to_anchor=(0.5, 0.995), fontsize=13)
    fig.suptitle(
        f"{DATASET_LABEL} {GROUP_DISPLAY[group]}: three-network hippocampal beta profiles\n"
        "Target-hemisphere rows averaged within subject; hippocampal L/R remain separate",
        fontsize=17,
        fontweight="bold",
        y=1.045,
    )
    fig.text(0.5, 0.005, "Boxes/points are subject-level descriptive values; no bilateral hippocampal value is used.", ha="center", fontsize=11, color="#444444")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    return save_pair(fig, output_dir, f"three_network_{group}_profile_boxscatter", pdf=EXPORT_PDF)


PAIRWISE_DEFINITIONS = (
    ("Default_C3_minus_C1", "Default", 3, 1),
    ("Default_C3_minus_C2", "Default", 3, 2),
    ("VentralAttention_C1_minus_C2", "VentralAttention", 1, 2),
    ("VentralAttention_C1_minus_C3", "VentralAttention", 1, 3),
)


def paired_table(data: pd.DataFrame) -> pd.DataFrame:
    wide = data.pivot_table(
        index=["subject", "group", "site_network", "hippocampal_hemi", "parcellation"],
        columns="cluster",
        values="beta_mean",
        aggfunc="mean",
    ).reset_index()
    for cluster in CLUSTERS:
        if cluster not in wide.columns:
            wide[cluster] = np.nan
    rows = []
    for group in GROUPS:
        for parcellation in PARCELLATIONS:
            for hipp_hemi in HEMIS:
                branch = wide.loc[
                    (wide["group"] == group)
                    & (wide["parcellation"] == parcellation)
                    & (wide["hippocampal_hemi"] == hipp_hemi)
                ]
                branch_results = []
                for name, network, numerator, denominator in PAIRWISE_DEFINITIONS:
                    cell = branch.loc[branch["site_network"] == network, [numerator, denominator]].dropna()
                    difference = (cell[numerator] - cell[denominator]).to_numpy(float)
                    if difference.size >= 2:
                        test = stats.ttest_1samp(difference, 0.0)
                        mean = float(np.mean(difference))
                        sd = float(np.std(difference, ddof=1))
                        ci = stats.t.interval(0.95, difference.size - 1, loc=mean, scale=stats.sem(difference))
                        t_value = float(test.statistic)
                        p_raw = float(test.pvalue)
                        df_value = int(difference.size - 1)
                        ci_low, ci_high = float(ci[0]), float(ci[1])
                    else:
                        mean = sd = t_value = p_raw = ci_low = ci_high = math.nan
                        df_value = max(int(difference.size - 1), 0)
                    branch_results.append(
                        {
                            "group": group,
                            "parcellation": parcellation,
                            "hippocampal_hemi": hipp_hemi,
                            "test": name,
                            "site_network": network,
                            "numerator_cluster": numerator,
                            "denominator_cluster": denominator,
                            "n": int(difference.size),
                            "mean_difference": mean,
                            "standard_deviation": sd,
                            "ci95_low": ci_low,
                            "ci95_high": ci_high,
                            "t_value": t_value,
                            "df": df_value,
                            "p_raw_two_sided": p_raw,
                        }
                    )
                p_values = [row["p_raw_two_sided"] for row in branch_results]
                finite_indices = [i for i, p in enumerate(p_values) if np.isfinite(p)]
                adjusted = np.full(len(p_values), np.nan)
                if finite_indices:
                    adjusted[finite_indices] = multipletests(
                        [p_values[i] for i in finite_indices], method="holm"
                    )[1]
                for index, row in enumerate(branch_results):
                    row["p_holm_within_group_parcellation_hippocampal_hemi"] = float(adjusted[index]) if np.isfinite(adjusted[index]) else math.nan
                    rows.append(row)
    return pd.DataFrame(rows)


def pairwise_figure(data: pd.DataFrame, tests: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    wide = data.pivot_table(
        index=["subject", "group", "site_network", "hippocampal_hemi", "parcellation"],
        columns="cluster",
        values="beta_mean",
        aggfunc="mean",
    ).reset_index()
    fig, axes = plt.subplots(8, 4, figsize=(20, 31), squeeze=False)
    row_index = 0
    for group in GROUPS:
        for parcellation in PARCELLATIONS:
            for hipp_hemi in HEMIS:
                branch = wide.loc[
                    (wide["group"] == group)
                    & (wide["parcellation"] == parcellation)
                    & (wide["hippocampal_hemi"] == hipp_hemi)
                ]
                branch_values = branch[list(CLUSTERS)].to_numpy(float).ravel()
                branch_values = branch_values[np.isfinite(branch_values)]
                if branch_values.size:
                    low, high = np.quantile(branch_values, [0.01, 0.99])
                    span = max(float(high - low), 1.0)
                    ylim = (float(low - 0.15 * span), float(high + 0.22 * span))
                else:
                    ylim = (-1.0, 1.0)
                for col_index, (name, network, numerator, denominator) in enumerate(PAIRWISE_DEFINITIONS):
                    ax = axes[row_index, col_index]
                    cell = branch.loc[branch["site_network"] == network, [numerator, denominator]].dropna()
                    x = cell[numerator].to_numpy(float)
                    y = cell[denominator].to_numpy(float)
                    color = NETWORK_COLORS[network]
                    bp = ax.boxplot([x, y], positions=[1, 2], widths=0.26, patch_artist=True, showfliers=False)
                    for patch, alpha in zip(bp["boxes"], (0.18, 0.05)):
                        patch.set_facecolor(color)
                        patch.set_edgecolor(color)
                        patch.set_alpha(alpha)
                    rng = np.random.default_rng(9000 + row_index * 10 + col_index)
                    jitter = rng.uniform(-0.035, 0.035, size=len(x))
                    if len(x):
                        ax.plot(np.vstack([np.ones(len(x)) + jitter, np.ones(len(x)) * 2 + jitter]), np.vstack([x, y]), color="#888888", alpha=0.25, linewidth=0.6)
                        ax.scatter(np.ones(len(x)) + jitter, x, color=color, alpha=0.55, s=13)
                        ax.scatter(np.ones(len(y)) * 2 + jitter, y, facecolors="none", edgecolors=color, alpha=0.75, s=15)
                    result = tests.loc[
                        (tests["group"] == group)
                        & (tests["parcellation"] == parcellation)
                        & (tests["hippocampal_hemi"] == hipp_hemi)
                        & (tests["test"] == name)
                    ].iloc[0]
                    ax.axhline(0, color="#999999", linestyle="--", linewidth=0.7)
                    ax.grid(axis="y", color="#EEEEEE", linewidth=0.6)
                    ax.set_axisbelow(True)
                    ax.set_ylim(*ylim)
                    ax.set_xticks([1, 2], [f"C{numerator}", f"C{denominator}"])
                    ax.tick_params(labelsize=10.5)
                    ax.set_title(f"{NETWORK_DISPLAY[network]}: C{numerator} − C{denominator}\nmean Δ={result['mean_difference']:.2f}\nraw p={format_p(result['p_raw_two_sided'])}, Holm p={format_p(result['p_holm_within_group_parcellation_hippocampal_hemi'])}, n={int(result['n'])}", fontsize=10.5)
                    if col_index == 0:
                        ax.set_ylabel(f"{GROUP_DISPLAY[group]}\n{PARCELLATION_DISPLAY[parcellation]}\n{HEMI_DISPLAY[hipp_hemi]}\nBeta", fontsize=10.5)
                    else:
                        ax.set_ylabel("")
                    for spine in ("top", "right"):
                        ax.spines[spine].set_visible(False)
                row_index += 1
    fig.suptitle(
        f"{DATASET_LABEL} within-network cluster pairwise tests\n"
        "Subject-level target-hemisphere average; four pre-defined contrasts, Holm corrected within group × parcellation × hippocampal side",
        fontsize=17,
        fontweight="bold",
        y=0.998,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985), h_pad=1.8, w_pad=0.8)
    return save_pair(fig, output_dir, "surface_template_matching_hcp151_within_network_pairwise_8x4", pdf=EXPORT_PDF)


def load_network_columns() -> dict[str, np.ndarray]:
    payload = json.loads(MERGE_CONFIG.read_text(encoding="utf-8"))
    family_to_network = payload["atlases"]["kong2022"]["mapping"]
    columns: dict[str, list[int]] = {network: [] for network in NETWORKS}
    with LUT_PATH.open(encoding="utf-8") as stream:
        for line in stream:
            fields = line.split()
            if not fields:
                continue
            roi_id = int(fields[0])
            match = ROI_NAME_RE.fullmatch(fields[-1])
            if match is None:
                raise ValueError(f"invalid Kong2022 LUT name: {fields[-1]}")
            network = family_to_network[match.group("family")]
            if network in columns:
                columns[network].append(roi_id - 1)
    result = {network: np.asarray(values, dtype=np.int32) for network, values in columns.items()}
    if any(values.size == 0 for values in result.values()):
        raise ValueError(f"missing ROI columns for one of the top networks: {result}")
    return result


def label_path(subject: str, hipp_hemi: str, parcellation: str) -> Path:
    if parcellation == "template_matching":
        return MATCH_ROOT / f"sub-{subject}/matching/2mm/hemi-{hipp_hemi}/labels.npy"
    return HCP151_LABEL_ROOT / f"hcp151_2mm_hemi-{hipp_hemi}_labels.npy"


def fc_path(subject: str, hipp_hemi: str) -> Path:
    return MATCH_ROOT / f"sub-{subject}/matching/2mm/hemi-{hipp_hemi}/vertex_to_roi_fisherz_fc.npy"


def make_fc_table(beta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    network_columns = load_network_columns()
    subjects = sorted(beta["subject"].unique())
    rows = []
    missing = []
    label_cache: dict[tuple[str, str, str], np.ndarray] = {}
    fc_cache: dict[tuple[str, str], np.ndarray] = {}
    for subject in subjects:
        for hipp_hemi in HEMIS:
            fc_file = fc_path(subject, hipp_hemi)
            if not fc_file.is_file():
                missing.append({"subject": subject, "hippocampal_hemi": hipp_hemi, "reason": "missing_vertex_to_roi_fisherz_fc"})
                continue
            fc_values = np.asarray(np.load(fc_file), dtype=float)
            if fc_values.shape != (512, 400):
                raise ValueError(f"unexpected FC shape {fc_values.shape}: {fc_file}")
            fc_cache[(subject, hipp_hemi)] = fc_values
            for parcellation in PARCELLATIONS:
                labels_file = label_path(subject, hipp_hemi, parcellation)
                if not labels_file.is_file():
                    missing.append({"subject": subject, "hippocampal_hemi": hipp_hemi, "parcellation": parcellation, "reason": "missing_hippocampal_labels"})
                    continue
                labels = np.asarray(np.load(labels_file), dtype=int).reshape(-1)
                if labels.shape != (512,):
                    raise ValueError(f"unexpected hippocampal label shape {labels.shape}: {labels_file}")
                label_cache[(subject, hipp_hemi, parcellation)] = labels
                for network in NETWORKS:
                    network_matrix = fc_values[:, network_columns[network]]
                    network_fc = np.full((network_matrix.shape[0],), np.nan, dtype=float)
                    has_fc = np.isfinite(network_matrix).any(axis=1)
                    if np.any(has_fc):
                        network_fc[has_fc] = np.nanmean(network_matrix[has_fc], axis=1)
                    for cluster in CLUSTERS:
                        keep = labels == cluster
                        values = network_fc[keep]
                        values = values[np.isfinite(values)]
                        rows.append(
                            {
                                "subject": subject,
                                "site_network": network,
                                "hippocampal_hemi": hipp_hemi,
                                "cluster": cluster,
                                "parcellation": parcellation,
                                "fisherz_fc": float(np.mean(values)) if values.size else math.nan,
                                "hippocampal_vertex_count": int(values.size),
                                "network_roi_count": int(network_columns[network].size),
                                "fc_definition": "mean vertex-to-Kong2022-ROI Fisher-z FC across stimulation-network ROIs",
                            }
                        )
    fc = pd.DataFrame(rows)
    if fc.empty:
        raise ValueError(f"no {DATASET_LABEL} FC rows could be built from local matching outputs")
    return fc, pd.DataFrame(missing)


def fisher_ci(r_value: float, n: int) -> tuple[float, float]:
    if n <= 3 or not np.isfinite(r_value) or abs(r_value) >= 1:
        return math.nan, math.nan
    z = np.arctanh(np.clip(r_value, -0.999999, 0.999999))
    half = stats.norm.ppf(0.975) / math.sqrt(n - 3)
    return float(np.tanh(z - half)), float(np.tanh(z + half))


def format_p(value: object) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(value):
        return "NA"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def fc_stats(fc_beta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in GROUPS:
        for parcellation in PARCELLATIONS:
            for hipp_hemi in HEMIS:
                for network in NETWORKS:
                    for cluster in CLUSTERS:
                        cell = fc_beta.loc[
                            (fc_beta["group"] == group)
                            & (fc_beta["parcellation"] == parcellation)
                            & (fc_beta["hippocampal_hemi"] == hipp_hemi)
                            & (fc_beta["site_network"] == network)
                            & (fc_beta["cluster"] == cluster),
                        ].dropna(subset=["fisherz_fc", "beta_mean"])
                        n = len(cell)
                        if n >= 3 and cell["fisherz_fc"].nunique() > 1 and cell["beta_mean"].nunique() > 1:
                            result = stats.pearsonr(cell["fisherz_fc"], cell["beta_mean"])
                            r_value, p_value = float(result.statistic), float(result.pvalue)
                            ci_low, ci_high = fisher_ci(r_value, n)
                        else:
                            r_value = p_value = ci_low = ci_high = math.nan
                        rows.append(
                            {
                                "group": group,
                                "parcellation": parcellation,
                                "hippocampal_hemi": hipp_hemi,
                                "network": network,
                                "network_display": NETWORK_DISPLAY[network],
                                "cluster": cluster,
                                "cluster_display": CLUSTER_DISPLAY[cluster].replace("\n", " / "),
                                "n": n,
                                "n_subjects": int(cell["subject"].nunique()),
                                "pearson_r": r_value,
                                "ci95_low": ci_low,
                                "ci95_high": ci_high,
                                "pearson_p_raw": p_value,
                            }
                        )
    return pd.DataFrame(rows)


def heatmap_figure(fc_stats_table: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    fig, axes = plt.subplots(4, 2, figsize=(14.5, 16.5), squeeze=False)
    norm = Normalize(vmin=-1, vmax=1)
    cmap = plt.get_cmap("RdBu_r")
    row_index = 0
    for group in GROUPS:
        for parcellation in PARCELLATIONS:
            for col_index, hipp_hemi in enumerate(HEMIS):
                ax = axes[row_index, col_index]
                branch = fc_stats_table.loc[
                    (fc_stats_table["group"] == group)
                    & (fc_stats_table["hippocampal_hemi"] == hipp_hemi)
                    & (fc_stats_table["parcellation"] == parcellation)
                ]
                for y_index, network in enumerate(NETWORKS):
                    for x_index, cluster in enumerate(CLUSTERS):
                        cell = branch.loc[(branch["network"] == network) & (branch["cluster"] == cluster)].iloc[0]
                        r_value = float(cell["pearson_r"])
                        color = "#F2F2F2" if not np.isfinite(r_value) else cmap(norm(r_value))
                        edge = "#111111" if np.isfinite(cell["pearson_p_raw"]) and cell["pearson_p_raw"] < 0.05 else "white"
                        linewidth = 2.4 if edge == "#111111" else 1.0
                        ax.add_patch(plt.Rectangle((x_index - 0.48, y_index - 0.48), 0.96, 0.96, facecolor=color, edgecolor=edge, linewidth=linewidth))
                        r_text = "r=NA" if not np.isfinite(r_value) else f"r={r_value:.2f}"
                        ax.text(x_index, y_index - 0.09, r_text, ha="center", va="center", fontsize=17, fontweight="bold")
                        ax.text(x_index, y_index + 0.21, f"p={format_p(cell['pearson_p_raw'])}, n={int(cell['n'])}", ha="center", va="center", fontsize=13.5)
                ax.set_xlim(-0.5, 2.5)
                ax.set_ylim(2.5, -0.5)
                ax.set_xticks(range(3), ["C1\nAnterior", "C2\nBody", "C3\nPosterior"])
                ax.set_yticks(range(3), [NETWORK_DISPLAY[n] for n in NETWORKS])
                ax.set_title(f"{GROUP_DISPLAY[group]} | {PARCELLATION_DISPLAY[parcellation]}", fontsize=15.5, pad=9, fontweight="bold")
                ax.tick_params(length=0, labelsize=14.5)
                for spine in ax.spines.values():
                    spine.set_visible(False)
            row_index += 1
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.subplots_adjust(left=0.125, right=0.885, top=0.84, bottom=0.08, hspace=0.45, wspace=0.22)
    for col_index, hipp_hemi in enumerate(HEMIS):
        bbox = axes[0, col_index].get_position()
        fig.text(
            bbox.x0 + bbox.width / 2,
            0.895,
            HEMI_DISPLAY[hipp_hemi].upper(),
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
        )
    cbar_axis = fig.add_axes([0.915, 0.16, 0.020, 0.60])
    cbar = fig.colorbar(sm, cax=cbar_axis)
    cbar.set_label("Pearson r", rotation=270, labelpad=18, fontsize=16.5)
    cbar.ax.tick_params(labelsize=13.5)
    fig.suptitle(
        f"{DATASET_LABEL} stimulation-network FC–beta correlations\n"
        "Kong2022 ROI-FC exploratory association; black border = raw p < 0.05",
        fontsize=20,
        fontweight="bold",
        y=0.985,
    )
    fig.text(0.5, 0.020, "FC and beta are one value per subject × network × hippocampal side × cluster; target-side rows were averaged before correlation.", ha="center", fontsize=11.5, color="#444444")
    return save_pair(fig, output_dir, "fc_beta_network_by_cluster_pearson_heatmap_4x2", pdf=EXPORT_PDF)


def scatter_figure(fc_beta: pd.DataFrame, fc_stats_table: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    parcellation = "hcp151"
    fig, axes = plt.subplots(4, 9, figsize=(28.0, 13.5), squeeze=False)
    for row_index, (group, hipp_hemi) in enumerate((g, h) for g in GROUPS for h in HEMIS):
        for network_index, network in enumerate(NETWORKS):
            for cluster_index, cluster in enumerate(CLUSTERS):
                col_index = network_index * 3 + cluster_index
                ax = axes[row_index, col_index]
                cell = fc_beta.loc[
                    (fc_beta["group"] == group)
                    & (fc_beta["hippocampal_hemi"] == hipp_hemi)
                    & (fc_beta["parcellation"] == parcellation)
                    & (fc_beta["site_network"] == network)
                    & (fc_beta["cluster"] == cluster)
                ].dropna(subset=["fisherz_fc", "beta_mean"])
                color = NETWORK_COLORS[network]
                if len(cell):
                    ax.scatter(cell["fisherz_fc"], cell["beta_mean"], s=26, color=color, alpha=0.55, edgecolors=color, linewidths=0.4)
                if len(cell) >= 2 and cell["fisherz_fc"].nunique() > 1:
                    slope, intercept = np.polyfit(cell["fisherz_fc"], cell["beta_mean"], 1)
                    x_min, x_max = ax.get_xlim()
                    x_line = np.linspace(x_min, x_max, 40)
                    ax.plot(x_line, intercept + slope * x_line, color="#333333", linestyle="--", linewidth=1.2)
                cell_stats = fc_stats_table.loc[
                    (fc_stats_table["group"] == group)
                    & (fc_stats_table["hippocampal_hemi"] == hipp_hemi)
                    & (fc_stats_table["parcellation"] == parcellation)
                    & (fc_stats_table["network"] == network)
                    & (fc_stats_table["cluster"] == cluster)
                ].iloc[0]
                ax.axhline(0, color="#AAAAAA", linestyle=":", linewidth=0.7)
                ax.set_title(
                    f"{NETWORK_DISPLAY[network]} / C{cluster}\nr={cell_stats['pearson_r']:.2f}  p={format_p(cell_stats['pearson_p_raw'])}  n={int(cell_stats['n'])}"
                    if np.isfinite(cell_stats["pearson_r"])
                    else f"{NETWORK_DISPLAY[network]} / C{cluster}\nr=NA  n={int(cell_stats['n'])}",
                    fontsize=9.5,
                    pad=4,
                )
                ax.grid(color="#F0F0F0", linewidth=0.6)
                ax.set_axisbelow(True)
                if row_index == 3:
                    ax.set_xlabel("FC", fontsize=10.5)
                if col_index == 0:
                    ax.set_ylabel("Beta", fontsize=10.5)
                ax.tick_params(labelsize=8.5, length=3)
                for spine in ("top", "right"):
                    ax.spines[spine].set_visible(False)
        axes[row_index, 0].text(
            -0.62,
            0.5,
            f"{GROUP_DISPLAY[group]}\n{HEMI_DISPLAY[hipp_hemi]}",
            transform=axes[row_index, 0].transAxes,
            ha="right",
            va="center",
            fontsize=11.5,
            fontweight="bold",
            rotation=90,
        )
    for network_index, network in enumerate(NETWORKS):
        axes[0, network_index * 3 + 1].annotate(
            NETWORK_DISPLAY[network],
            xy=(0.5, 1.32),
            xycoords="axes fraction",
            ha="center",
            va="center",
            fontsize=13.5,
            fontweight="bold",
            color=NETWORK_COLORS[network],
        )
    fig.suptitle(
        f"{DATASET_LABEL} HCP151 FC–beta associations by stimulation network and hippocampal cluster\n"
        "Rows preserve healthy/patient and left/right hippocampal analyses",
        fontsize=17,
        fontweight="bold",
        y=1.01,
    )
    fig.text(0.5, 0.002, "FC = mean vertex-to-Kong2022-ROI Fisher-z FC across each stimulation network; target-side rows averaged within subject.", ha="center", fontsize=10.0, color="#444444")
    fig.tight_layout(rect=(0.045, 0.025, 0.995, 0.955), w_pad=0.8, h_pad=1.4)
    return save_pair(fig, output_dir, "fc_beta_hcp151_network_by_cluster_scatter_4x9", pdf=EXPORT_PDF)


def main() -> int:
    args = parse_args()
    global EXPORT_PDF
    EXPORT_PDF = bool(args.pdf)
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or run_dir / "five_figures").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    beta = read_beta(run_dir)
    figure_data = collapse_target_hemi(beta)
    figure_data.to_csv(output_dir / "figure_subject_network_beta.csv", index=False)
    network_comparisons = van_network_comparisons(figure_data)
    network_comparisons_path = output_dir / "healthy_van_vs_fpcn_dmn_ttests.csv"
    network_comparisons.to_csv(network_comparisons_path, index=False)

    figure_paths: dict[str, dict[str, str]] = {}
    figure_paths["healthy_profile"] = profile_figure(
        figure_data, "healthy", output_dir, comparisons=network_comparisons
    )
    figure_paths["patient_profile"] = profile_figure(figure_data, "patient", output_dir)

    pairwise = paired_table(figure_data)
    pairwise.to_csv(output_dir / "within_network_pairwise_ttests.csv", index=False)
    figure_paths["within_network_pairwise"] = pairwise_figure(figure_data, pairwise, output_dir)

    fc, missing_fc = make_fc_table(beta)
    fc_beta = figure_data.merge(
        fc,
        on=["subject", "site_network", "hippocampal_hemi", "cluster", "parcellation"],
        how="left",
        validate="many_to_one",
    )
    fc_beta.to_csv(output_dir / "fc_beta_subject_rows.csv", index=False)
    missing_fc.to_csv(output_dir / "fc_input_missing.csv", index=False)
    fc_stats_table = fc_stats(fc_beta)
    fc_stats_table.to_csv(output_dir / "network_by_cluster_pearson.csv", index=False)
    figure_paths["fc_beta_heatmap"] = heatmap_figure(fc_stats_table, output_dir)
    figure_paths["fc_beta_scatter"] = scatter_figure(fc_beta, fc_stats_table, output_dir)

    summary = {
        "status": "complete",
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "source_subject_network_beta": str(run_dir / "subject_network_beta.csv"),
        "groups": list(GROUPS),
        "group_definition": {"healthy": ["NTHC", "TEHC"], "patient": ["NTS", "TIS"]},
        "networks": {network: NETWORK_DISPLAY[network] for network in NETWORKS},
        "parcellations": list(PARCELLATIONS),
        "hippocampal_hemispheres": list(HEMIS),
        "heatmap_layout": {"rows": "group × parcellation", "left_column": "L", "right_column": "R"},
        "bilateral_rows": False,
        "target_hemi_policy": "primary mixed-effects models retain target-hemi rows; figures and FC-beta correlations average available target-hemi rows equally within subject",
        "primary_beta_rows_finite": int(len(beta)),
        "figure_subject_rows": int(len(figure_data)),
        "figure_subjects": int(figure_data["subject"].nunique()),
        "fc_beta_rows": int(len(fc_beta)),
        "fc_beta_rows_finite": int(fc_beta[["fisherz_fc", "beta_mean"]].dropna().shape[0]),
        "fc_inputs_missing_rows": int(len(missing_fc)),
        "healthy_van_network_comparisons": {
            "table": str(network_comparisons_path),
            "reference_network": "VentralAttention",
            "comparator_networks": list(VAN_COMPARATOR_NETWORKS),
            "method": "Welch independent-samples t-test",
            "correction": "none",
            "n_rows": int(len(network_comparisons)),
        },
        "pdf": EXPORT_PDF,
        "figures": figure_paths,
    }
    (output_dir / "figure_manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    spec = pd.DataFrame(
        [
            ("analysis", f"{DATASET_LABEL} surface-beta five-figure analysis"),
            ("groups", "healthy=NTHC+TEHC; patient=NTS+TIS"),
            ("networks", "Control/FPCN; Default/DMN; VentralAttention/VAN"),
            ("parcellations", "individual template matching and HCP151 fixed atlas"),
            ("hippocampal_hemisphere", "L and R separate in every figure and statistic; no bilateral rows"),
            ("heatmap_layout", "rows=group x parcellation; left column=L hippocampus; right column=R hippocampus"),
            ("figure_observation_unit", "one subject x stimulation network x hippocampal hemisphere x cluster x parcellation after equal target-hemi averaging"),
            ("healthy_van_network_comparisons", "uncorrected Welch tests of VAN minus Default/DMN and VAN minus Control/FPCN for every healthy profile bar group; draw only significant brackets"),
            ("pairwise_tests", "four pre-defined within-network cluster contrasts; Holm within group x parcellation x hippocampal hemisphere"),
            ("fc_definition", "mean vertex-to-Kong2022-ROI Fisher-z FC across ROI columns assigned to the stimulation network"),
            ("fc_multiplicity", "raw exploratory Pearson p values, matching the JJ_TMS exploratory FC figure convention"),
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

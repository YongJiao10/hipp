#!/opt/miniconda3/envs/py314/bin/python
"""Render compact exploratory L/R Native-space hippocampal beta figures.

This is not a numbered formal-workflow entry point.

The heatmap layout is fixed: the left column is the left hippocampus and the
right column is the right hippocampus.  Target-hemisphere rows are averaged
within subject before plotting so a subject is not weighted by the number of
target sides represented in the extraction table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch


NETWORKS = ("Control", "Default", "VentralAttention")
NETWORK_DISPLAY = {"Control": "CON", "Default": "DMN", "VentralAttention": "VAN"}
NETWORK_COLORS = {"Control": "#F7AC4A", "Default": "#D85C6B", "VentralAttention": "#DC5CFF"}
PARCELLATIONS = ("template_matching", "hcp")
PARCELLATION_DISPLAY = {"template_matching": "Template matching", "hcp": "HCP fixed atlas"}
HEMIS = ("L", "R")
HEMI_DISPLAY = {"L": "Left hippocampus", "R": "Right hippocampus"}
CLUSTERS = (1, 2, 3)
CLUSTER_DISPLAY = {1: "C1\nAnterior / VAN", 2: "C2\nMiddle / SMN", 3: "C3\nPosterior / DMN"}


def read_beta(run_dir: Path) -> pd.DataFrame:
    data = pd.read_csv(run_dir / "subject_network_beta.csv", dtype={"subject": str})
    required = {"subject", "site_network", "target_hemi", "hippocampal_hemi", "cluster", "parcellation", "beta_mean"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"missing beta columns: {missing}")
    if "bilateral" in set(data["hippocampal_hemi"].astype(str).str.lower()):
        raise ValueError("Native beta table contains forbidden bilateral rows")
    data["cluster"] = pd.to_numeric(data["cluster"], errors="raise").astype(int)
    data["beta_mean"] = pd.to_numeric(data["beta_mean"], errors="coerce")
    data = data.loc[
        data["site_network"].isin(NETWORKS)
        & data["target_hemi"].isin(HEMIS)
        & data["hippocampal_hemi"].isin(HEMIS)
        & data["cluster"].isin(CLUSTERS)
        & data["parcellation"].isin(PARCELLATIONS)
        & np.isfinite(data["beta_mean"])
    ].copy()
    if data.empty:
        raise ValueError("no finite Native beta rows remain")
    return data


def collapse_target_hemi(data: pd.DataFrame) -> pd.DataFrame:
    keys = ["subject", "site_network", "hippocampal_hemi", "cluster", "parcellation"]
    return data.groupby(keys, as_index=False, dropna=False)["beta_mean"].mean()


def save_pair(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def profile_figure(data: pd.DataFrame) -> plt.Figure:
    collapsed = collapse_target_hemi(data)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.0), squeeze=False, sharey=False)
    for row_index, parcellation in enumerate(PARCELLATIONS):
        for column_index, hemi in enumerate(HEMIS):
            ax = axes[row_index, column_index]
            panel = collapsed.loc[
                (collapsed["parcellation"] == parcellation)
                & (collapsed["hippocampal_hemi"] == hemi)
            ]
            all_values = panel["beta_mean"].to_numpy(float)
            y_low, y_high = np.quantile(all_values, [0.01, 0.99]) if all_values.size else (-1, 1)
            span = max(float(y_high - y_low), 1.0)
            centers = {"Control": 1 - 0.22, "Default": 1.0, "VentralAttention": 1 + 0.22}
            for cluster in CLUSTERS:
                for network in NETWORKS:
                    values = panel.loc[
                        (panel["cluster"] == cluster) & (panel["site_network"] == network), "beta_mean"
                    ].to_numpy(float)
                    if not values.size:
                        continue
                    center = cluster + (centers[network] - 1)
                    box = ax.boxplot(values, positions=[center], widths=0.14, patch_artist=True, showfliers=False, manage_ticks=False)
                    for patch in box["boxes"]:
                        patch.set_facecolor(NETWORK_COLORS[network])
                        patch.set_alpha(0.18)
                        patch.set_edgecolor(NETWORK_COLORS[network])
                        patch.set_linewidth(1.2)
                    for median in box["medians"]:
                        median.set_color(NETWORK_COLORS[network])
                        median.set_linewidth(1.7)
                    rng = np.random.default_rng(1000 + row_index * 100 + column_index * 10 + cluster)
                    jitter = rng.uniform(-0.045, 0.045, size=values.size)
                    ax.scatter(np.full(values.size, center) + jitter, values, s=18, color=NETWORK_COLORS[network], alpha=0.55, linewidths=0, zorder=3)
            ax.axhline(0, color="#999999", linestyle="--", linewidth=0.8)
            ax.grid(axis="y", color="#EEEEEE", linewidth=0.7)
            ax.set_axisbelow(True)
            ax.set_xlim(0.55, 3.45)
            ax.set_ylim(float(y_low - 0.18 * span), float(y_high + 0.24 * span))
            ax.set_xticks(list(CLUSTERS), [CLUSTER_DISPLAY[c] for c in CLUSTERS], fontsize=11)
            ax.tick_params(axis="y", labelsize=10)
            ax.set_xlabel("Hippocampal cluster", fontsize=12)
            ax.set_ylabel("Native con beta", fontsize=12)
            ax.set_title(f"{PARCELLATION_DISPLAY[parcellation]} | {HEMI_DISPLAY[hemi]}", fontsize=14, pad=8)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
    handles = [Patch(facecolor=NETWORK_COLORS[n], edgecolor=NETWORK_COLORS[n], alpha=0.35, label=NETWORK_DISPLAY[n]) for n in NETWORKS]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=12, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("JJ_TMS Native-space hippocampal surface beta profiles", fontsize=18, fontweight="bold", y=1.015)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.07, top=0.90, wspace=0.22, hspace=0.30)
    return fig


def heatmap_figure(data: pd.DataFrame) -> plt.Figure:
    collapsed = collapse_target_hemi(data)
    cell_means = collapsed.groupby(["parcellation", "hippocampal_hemi", "site_network", "cluster"], as_index=False)["beta_mean"].mean()
    values = cell_means["beta_mean"].to_numpy(float)
    limit = max(abs(float(np.nanmin(values))), abs(float(np.nanmax(values))), 1e-6)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.2), squeeze=False)
    image = None
    for row_index, parcellation in enumerate(PARCELLATIONS):
        for column_index, hemi in enumerate(HEMIS):
            ax = axes[row_index, column_index]
            matrix = np.full((len(NETWORKS), len(CLUSTERS)), np.nan, dtype=float)
            for network_index, network in enumerate(NETWORKS):
                for cluster_index, cluster in enumerate(CLUSTERS):
                    match = cell_means.loc[
                        (cell_means["parcellation"] == parcellation)
                        & (cell_means["hippocampal_hemi"] == hemi)
                        & (cell_means["site_network"] == network)
                        & (cell_means["cluster"] == cluster),
                        "beta_mean",
                    ]
                    if len(match):
                        matrix[network_index, cluster_index] = float(match.iloc[0])
            image = ax.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")
            for network_index, network in enumerate(NETWORKS):
                for cluster_index, cluster in enumerate(CLUSTERS):
                    value = matrix[network_index, cluster_index]
                    if np.isfinite(value):
                        ax.text(cluster_index, network_index, f"{value:.2f}", ha="center", va="center", fontsize=13, color="black")
            ax.set_xticks(range(3), ["C1", "C2", "C3"], fontsize=12)
            ax.set_yticks(range(3), [NETWORK_DISPLAY[n] for n in NETWORKS], fontsize=12)
            ax.set_xlabel("Hippocampal cluster", fontsize=13)
            ax.set_ylabel("TMS network", fontsize=13)
            ax.set_title(f"{PARCELLATION_DISPLAY[parcellation]} | {HEMI_DISPLAY[hemi]}", fontsize=15, pad=9)
            ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=2)
            ax.tick_params(which="minor", bottom=False, left=False)
    if image is not None:
        # Reserve an explicit right-hand strip so the colorbar cannot cover
        # the right-hippocampus column when the figure is saved with a tight
        # bounding box.
        colorbar_axis = fig.add_axes([0.865, 0.20, 0.022, 0.62])
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label("Mean Native con beta", fontsize=13)
        colorbar.ax.tick_params(labelsize=11)
    fig.suptitle("JJ_TMS Native-space network × cluster beta heatmap", fontsize=18, fontweight="bold", y=1.015)
    fig.subplots_adjust(left=0.10, right=0.82, bottom=0.08, top=0.90, wspace=0.30, hspace=0.35)
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or run_dir / "figures").resolve()
    data = read_beta(run_dir)
    save_pair(profile_figure(data), output_dir, "native_surface_network_cluster_profiles_2x2")
    save_pair(heatmap_figure(data), output_dir, "native_surface_network_cluster_heatmap_2x2")
    manifest = {
        "status": "complete",
        "input_run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "hippocampal_column_order": ["L", "R"],
        "parcellations": list(PARCELLATIONS),
        "target_hemi_collapsed_within_subject": True,
        "bilateral_rows_used": False,
        "pngs": [
            str(output_dir / "native_surface_network_cluster_profiles_2x2.png"),
            str(output_dir / "native_surface_network_cluster_heatmap_2x2.png"),
        ],
    }
    (output_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

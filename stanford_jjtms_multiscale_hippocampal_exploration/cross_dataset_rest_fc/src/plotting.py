"""Paper-ready static figures for the cross-dataset rest-FC report.

The plotting functions consume already reviewed result tables.  They do not
read input assets or refit models.  Every figure is exported as PNG with the
same deterministic caption/source metadata and a neutral,
blue/orange/grey palette that remains interpretable without colour alone.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


BLUE = "#2F6DB0"
BLUE_LIGHT = "#A9C5E6"
BLUE_DARK = "#1F4D7A"
ORANGE = "#C56A2D"
ORANGE_LIGHT = "#E9B48F"
ORANGE_DARK = "#8D481F"
CHARCOAL = "#252A30"
GREY = "#7D8790"
GREY_LIGHT = "#D8DDE1"
GREY_PALE = "#F5F6F7"
WHITE = "#FFFFFF"

NETWORKS_8 = (
    "Default",
    "Visual",
    "Somatomotor",
    "DorsalAttention",
    "VentralAttention",
    "Control",
    "Auditory",
    "Language",
)
NETWORKS_17 = (
    "DefaultA",
    "DefaultB",
    "DefaultC",
    "ContA",
    "ContB",
    "ContC",
    "SalVenAttnA",
    "SalVenAttnB",
    "DorsAttnA",
    "DorsAttnB",
    "SomMotA",
    "SomMotB",
    "VisualA",
    "VisualB",
    "VisualC",
    "Aud",
    "Language",
)
REGIONS = ("anterior", "middle", "posterior")
MODES = ("bilateral", "L", "R", "L-R")


def _as_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _palette(dataset: str) -> str:
    return BLUE if str(dataset).casefold() == "jjtms" else ORANGE




def _base_figure(*, width: float = 12, height: float = 7) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(width, height), facecolor=WHITE)
    ax.set_facecolor(WHITE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(GREY_LIGHT)
    ax.spines["bottom"].set_color(GREY_LIGHT)
    ax.tick_params(colors=CHARCOAL, labelsize=9)
    return fig, ax


def _caption(fig: plt.Figure, text: str, *, fontsize: int = 7) -> None:
    fig.text(
        0.01,
        0.008,
        text,
        ha="left",
        va="bottom",
        fontsize=fontsize,
        color=GREY,
        wrap=True,
    )


def _save(
    fig: plt.Figure,
    out_dir: Path,
    stem: str,
    caption: str,
    *,
    caption_fontsize: int = 7,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    _caption(fig, caption, fontsize=caption_fontsize)
    fig.savefig(
        png,
        dpi=220,
        facecolor=WHITE,
        bbox_inches="tight",
        metadata={"Title": stem, "Subject": caption, "Creator": "cross_dataset_rest_fc"},
    )
    plt.close(fig)
    return {"png": str(png), "caption": caption}


def _unavailable_panel(ax: plt.Axes, message: str, *, fontsize: int = 10) -> None:
    ax.set_axis_off()
    ax.text(
        0.5,
        0.5,
        f"Unavailable\n{message}",
        ha="center",
        va="center",
        color=GREY,
        fontsize=fontsize,
        transform=ax.transAxes,
    )


def _finite(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if frame.empty or column not in frame:
        return frame.iloc[0:0].copy()
    values = pd.to_numeric(frame[column], errors="coerce")
    return frame.loc[np.isfinite(values.to_numpy(dtype=float))].copy()


def plot_primary_effect_ci(
    primary: pd.DataFrame,
    out_dir: Path,
    *,
    run_id: str,
    top_k_per_dataset_estimand: int = 4,
) -> dict[str, str]:
    """Plot a deterministic full-table ranked subset as dots and 95% CIs."""

    frame = _finite(primary, "coefficient")
    frame = frame.loc[frame["fc_estimator"].eq("fisher_z")].copy()
    if frame.empty:
        fig, ax = _base_figure()
        _unavailable_panel(ax, "no finite primary localisation rows")
        fig.suptitle("Primary network-wise effects", color=CHARCOAL)
        return _save(
            fig,
            out_dir,
            "primary_effect_ci",
            f"run={run_id}; source=primary_localisation.csv.gz; ranking=none; Fisher-z FC",
        )

    frame["_abs"] = frame["coefficient"].abs()
    sort_cols = [
        "_abs",
        "dataset",
        "estimand",
        "region",
        "network",
        "method",
        "smoothing_mm",
        "hemisphere_mode",
        "resolution",
    ]
    frame = frame.sort_values(sort_cols, ascending=[False, True, True, True, True, True, True, True, True], kind="stable")
    selected = (
        frame.groupby(["dataset", "estimand"], sort=False, dropna=False)
        .head(int(top_k_per_dataset_estimand))
        .copy()
    )
    selected = selected.sort_values(
        ["dataset", "estimand", "_abs", "region", "network"],
        ascending=[True, True, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    selected["label"] = selected["region"].astype(str) + "/" + selected["network"].astype(str)
    selected["panel"] = selected["dataset"].astype(str) + " · " + selected["estimand"].astype(str)
    selected["y"] = np.arange(len(selected))[::-1]

    fig, ax = _base_figure(width=13, height=max(6.5, 0.28 * len(selected) + 2.8))
    for _, row in selected.iterrows():
        x = _as_float(row["coefficient"])
        low = _as_float(row.get("ci_low"))
        high = _as_float(row.get("ci_high"))
        if not np.isfinite(low) or not np.isfinite(high):
            low, high = x, x
        color = _palette(str(row["dataset"]))
        ax.plot([low, high], [row["y"], row["y"]], color=color, lw=1.4, alpha=0.85, solid_capstyle="round")
        ax.scatter(
            [x],
            [row["y"]],
            s=36,
            marker="o",
            facecolors=color,
            edgecolors=color,
            linewidths=1.0,
            zorder=3,
        )
    labels = [f"{row['panel']} | {row['label']}" for _, row in selected.iterrows()]
    ax.set_yticks(selected["y"].to_numpy(), labels, fontsize=8)
    ax.axvline(0.0, color=CHARCOAL, lw=0.9, zorder=1)
    ax.grid(axis="x", color=GREY_LIGHT, lw=0.65)
    ax.set_axisbelow(True)
    ax.set_xlabel("Group-contrast coefficient (Fisher-z FC)", color=CHARCOAL)
    ax.set_title("Primary network-wise effects and 95% confidence intervals", loc="left", color=CHARCOAL, pad=12)
    ax.text(
        0.0,
        1.01,
        "Full-table deterministic ranking: top 4 absolute coefficients per dataset × estimand; points show effects and 95% CIs",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=GREY,
        fontsize=8,
    )
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor=BLUE, label="JJTMS", markersize=6),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ORANGE, markeredgecolor=ORANGE, label="Stanford", markersize=6),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8)
    caption = (
        f"run={run_id}; source=primary_localisation.csv.gz; units=Fisher-z FC; "
        f"selection=full-table rank by |coefficient|, top {top_k_per_dataset_estimand} per dataset×estimand; "
        "multiplicity correction deferred until the final analysis-object set is frozen"
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    return _save(fig, out_dir, "primary_effect_ci", caption)


def _heatmap_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "blue_orange_signed",
        [(0.0, BLUE_DARK), (0.40, BLUE_LIGHT), (0.5, WHITE), (0.60, ORANGE_LIGHT), (1.0, ORANGE_DARK)],
    )


def plot_primary_localisation_heatmaps(
    primary: pd.DataFrame,
    out_dir: Path,
    *,
    run_id: str,
) -> dict[str, str]:
    """Render fixed/template reference heatmaps for both network resolutions."""

    panels = [
        ("jjtms", "patient-healthy", "hcp151_fixed", "bilateral", 8),
        ("jjtms", "patient-healthy", "hcp151_fixed", "L-R", 8),
        ("jjtms", "patient-healthy", "hcp151_fixed", "bilateral", 17),
        ("jjtms", "patient-healthy", "hcp151_fixed", "L-R", 17),
        ("jjtms", "patient-healthy", "hcp151_template_matching", "bilateral", 8),
        ("jjtms", "patient-healthy", "hcp151_template_matching", "L-R", 8),
        ("jjtms", "patient-healthy", "hcp151_template_matching", "bilateral", 17),
        ("jjtms", "patient-healthy", "hcp151_template_matching", "L-R", 17),
        ("stanford", "symptom_marginal", "hcp151_fixed", "bilateral", 8),
        ("stanford", "symptom_marginal", "hcp151_fixed", "L-R", 8),
        ("stanford", "symptom_marginal", "hcp151_fixed", "bilateral", 17),
        ("stanford", "symptom_marginal", "hcp151_fixed", "L-R", 17),
        ("stanford", "symptom_marginal", "hcp151_template_matching", "bilateral", 8),
        ("stanford", "symptom_marginal", "hcp151_template_matching", "L-R", 8),
        ("stanford", "symptom_marginal", "hcp151_template_matching", "bilateral", 17),
        ("stanford", "symptom_marginal", "hcp151_template_matching", "L-R", 17),
    ]
    values: list[float] = []
    for dataset, estimand, method, mode, resolution in panels:
        subset = primary.loc[
            primary["dataset"].eq(dataset)
            & primary["estimand"].eq(estimand)
            & primary["method"].eq(method)
            & primary["smoothing_mm"].eq(2)
            & primary["hemisphere_mode"].eq(mode)
            & primary["resolution"].eq(resolution)
        ]
        values.extend(pd.to_numeric(subset.get("coefficient", pd.Series(dtype=float)), errors="coerce").dropna().tolist())
    vmax = max([abs(value) for value in values if np.isfinite(value)] or [1.0])
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = _heatmap_cmap()
    fig, axes = plt.subplots(4, 4, figsize=(18, 15.2), facecolor=WHITE)
    axes = np.asarray(axes).reshape(4, 4)
    im = None
    for index, (dataset, estimand, method, mode, resolution) in enumerate(panels):
        ax = axes.flat[index]
        networks = NETWORKS_8 if resolution == 8 else NETWORKS_17
        subset = primary.loc[
            primary["dataset"].eq(dataset)
            & primary["estimand"].eq(estimand)
            & primary["method"].eq(method)
            & primary["smoothing_mm"].eq(2)
            & primary["hemisphere_mode"].eq(mode)
            & primary["resolution"].eq(resolution)
        ].copy()
        if subset.empty:
            _unavailable_panel(ax, "reference branch has no rows", fontsize=12)
            continue
        matrix = np.full((len(REGIONS), len(networks)), np.nan)
        pmatrix = np.full((len(REGIONS), len(networks)), np.nan)
        for row_index, region in enumerate(REGIONS):
            for col_index, network in enumerate(networks):
                row = subset.loc[(subset["region"] == region) & (subset["network"] == network)]
                if not row.empty:
                    matrix[row_index, col_index] = _as_float(row.iloc[0].get("coefficient"))
                    pmatrix[row_index, col_index] = _as_float(row.iloc[0].get("p_raw"))
        im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
        ax.set_yticks(range(len(REGIONS)), [r.title() for r in REGIONS], fontsize=9)
        if resolution == 8:
            ax.set_xticks(range(len(networks)), networks, rotation=52, ha="right", fontsize=8)
        else:
            ax.set_xticks(range(len(networks)), networks, rotation=68, ha="right", fontsize=7)
        ax.tick_params(length=0, pad=1)
        ax.set_title(f"{dataset.upper()} · {method.replace('_', ' ')}\n{resolution}net · {mode}", fontsize=10, color=CHARCOAL, pad=5)
        for row_index in range(len(REGIONS)):
            for col_index in range(len(networks)):
                p = pmatrix[row_index, col_index]
                if np.isfinite(p):
                    if p < 0.001:
                        marker = "***"
                    elif p < 0.01:
                        marker = "**"
                    elif p < 0.05:
                        marker = "*"
                    else:
                        marker = ""
                else:
                    marker = ""
                if marker:
                    star_fontsize = 8.5 if resolution == 8 else 6.5
                    ax.text(col_index, row_index, marker, ha="center", va="center", fontsize=star_fontsize, color=CHARCOAL, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.subplots_adjust(
        left=0.055, right=0.94, bottom=0.075, top=0.89, wspace=0.26, hspace=0.52
    )
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.02, aspect=35)
        cbar.set_label("Coefficient (Fisher-z FC)", color=CHARCOAL, fontsize=11)
        cbar.ax.tick_params(labelsize=10, colors=CHARCOAL)
    fig.suptitle("Primary localisation heatmaps: signed region × network effects", x=0.01, ha="left", color=CHARCOAL, fontsize=16)
    fig.text(
        0.01,
        0.965,
        "Reference smoothing is 2 mm; fixed/template branches shown at bilateral and L−R. Stars denote unadjusted raw p-values (* p < .05, ** p < .01, *** p < .001); multiplicity correction is deferred.",
        ha="left",
        va="top",
        fontsize=10,
        color=GREY,
    )
    caption = (
        f"run={run_id}; source=primary_localisation.csv.gz; reference branches=hcp151_fixed and "
        "hcp151_template_matching at 2mm, bilateral/L-R; 8net and 17net; "
        "JJTMS patient-healthy and Stanford symptom_marginal; cell stars denote raw p thresholds (* p < .05, ** p < .01, *** p < .001)"
    )
    return _save(fig, out_dir, "primary_localisation_heatmaps", caption, caption_fontsize=9)


def plot_stanford_four_group_profiles(
    descriptives: pd.DataFrame,
    out_dir: Path,
    *,
    run_id: str,
) -> dict[str, str]:
    """Plot Stanford four-cell means on a common Fisher-z colour scale."""
    descriptives = descriptives.copy(deep=True)
    if "resolution" not in descriptives and "network_resolution" in descriptives:
        descriptives["resolution"] = descriptives["network_resolution"]

    reference = descriptives.loc[
        descriptives["dataset"].eq("stanford")
        & descriptives["method"].eq("hcp151_fixed")
        & descriptives["smoothing_mm"].eq(2)
        & descriptives["hemisphere_mode"].eq("bilateral")
    ].copy()
    all_values = pd.to_numeric(reference.get("mean", pd.Series(dtype=float)), errors="coerce")
    finite = all_values[np.isfinite(all_values)]
    if finite.empty:
        fig, ax = _base_figure()
        _unavailable_panel(ax, "Stanford group descriptives unavailable on reference branch")
        fig.suptitle("Stanford four-group Fisher-z profiles", color=CHARCOAL)
        return _save(fig, out_dir, "stanford_four_group_profiles", f"run={run_id}; source=stanford_four_group_descriptives.csv.gz")
    vmax = max(abs(float(finite.min())), abs(float(finite.max())), 1e-6)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = _heatmap_cmap()
    groups = ("NTHC", "TEHC", "NTS", "TIS")
    fig, axes = plt.subplots(2, 1, figsize=(18, 8.6), gridspec_kw={"height_ratios": [1, 1.35]}, facecolor=WHITE)
    axes = np.atleast_1d(axes)
    im = None
    for ax, resolution in zip(axes, (8, 17)):
        networks = NETWORKS_8 if resolution == 8 else NETWORKS_17
        columns = [(region, network) for region in REGIONS for network in networks]
        matrix = np.full((len(groups), len(columns)), np.nan)
        for row_index, group in enumerate(groups):
            for col_index, (region, network) in enumerate(columns):
                row = reference.loc[
                    reference["group"].eq(group)
                    & reference["resolution"].eq(resolution)
                    & reference["region"].eq(region)
                    & reference["network"].eq(network)
                ]
                if not row.empty:
                    matrix[row_index, col_index] = _as_float(row.iloc[0].get("mean"))
        if not np.isfinite(matrix).any():
            _unavailable_panel(ax, f"{resolution}net reference branch unavailable")
            continue
        im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
        ax.set_yticks(range(len(groups)), groups, fontsize=8)
        labels = [f"{region[:3]}:{network}" for region in REGIONS for network in networks]
        ax.set_xticks(range(len(labels)), labels, rotation=72, ha="right", fontsize=5 if resolution == 17 else 6)
        ax.tick_params(length=0, pad=1)
        ax.set_title(f"{resolution}-network mean Fisher-z FC", loc="left", fontsize=9, color=CHARCOAL)
        if resolution == 8:
            ax.text(0.5, -0.42, "anterior", transform=ax.transAxes, ha="center", color=GREY, fontsize=8)
            ax.text(0.5, -0.56, "middle", transform=ax.transAxes, ha="center", color=GREY, fontsize=8)
            ax.text(0.5, -0.70, "posterior", transform=ax.transAxes, ha="center", color=GREY, fontsize=8)
        else:
            for boundary in (len(networks) - 0.5, 2 * len(networks) - 0.5):
                ax.axvline(boundary, color=WHITE, lw=1.2)
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.02, aspect=35)
        cbar.set_label("Mean Fisher-z FC (common scale)", color=CHARCOAL, fontsize=9)
        cbar.ax.tick_params(labelsize=8, colors=CHARCOAL)
    fig.suptitle("Stanford four-group resting-FC profiles", x=0.01, ha="left", color=CHARCOAL, fontsize=14)
    fig.text(
        0.01,
        0.955,
        "NTHC, TEHC, NTS, and TIS; hcp151_fixed, 2 mm, bilateral reference branch; values are absolute Fisher-z FC means.",
        ha="left",
        va="top",
        fontsize=8,
        color=GREY,
    )
    caption = (
        f"run={run_id}; source=stanford_four_group_descriptives.csv.gz; reference=hcp151_fixed, 2mm, bilateral; "
        "3 regions × 8/17 networks; common colour scale across resolutions; n, SD, and 95% CI remain in source table"
    )
    fig.subplots_adjust(
        left=0.065, right=0.95, bottom=0.22, top=0.86, hspace=0.64
    )
    return _save(fig, out_dir, "stanford_four_group_profiles", caption)


def plot_specification_consistency(
    primary: pd.DataFrame,
    out_dir: Path,
    *,
    run_id: str,
) -> dict[str, str]:
    """Show top absolute components across correlated fixed/template specs."""

    frame = _finite(primary, "coefficient")
    frame = frame.loc[
        frame["fc_estimator"].eq("fisher_z")
        & frame["hemisphere_mode"].eq("bilateral")
        & frame["smoothing_mm"].eq(2)
    ].copy()
    if frame.empty:
        fig, ax = _base_figure()
        _unavailable_panel(ax, "no finite specification rows")
        fig.suptitle("Specification consistency", color=CHARCOAL)
        return _save(fig, out_dir, "specification_consistency", f"run={run_id}; source=primary_localisation.csv.gz")
    frame["_abs"] = frame["coefficient"].abs()
    component_keys = ["dataset", "estimand", "resolution", "region", "network"]
    top = (
        frame.loc[frame["method"].eq("hcp151_fixed")]
        .sort_values(["_abs", *component_keys], ascending=[False, True, True, True, True, True], kind="stable")
        .drop_duplicates(component_keys)
        .head(12)
    )
    selected = frame.merge(top[component_keys], on=component_keys, how="inner")
    selected["component"] = selected["dataset"].str.upper() + " · " + selected["estimand"] + " · " + selected["region"] + "/" + selected["network"] + " · " + selected["resolution"].astype(str) + "net"
    selected["spec"] = selected["method"].str.replace("hcp151_", "", regex=False) + " " + selected["smoothing_mm"].astype(str) + "mm"
    spec_order = {"fixed 2mm": 0, "fixed 4mm": 1, "template_matching 2mm": 2, "template_matching 4mm": 3, "template 2mm": 2, "template 4mm": 3}
    selected["spec_index"] = selected["spec"].map(spec_order).fillna(9)
    selected = selected.sort_values(["component", "spec_index"], kind="stable")
    components = list(selected["component"].drop_duplicates())
    y_positions = {component: index for index, component in enumerate(reversed(components))}
    fig, ax = _base_figure(width=13, height=max(6.5, 0.4 * len(components) + 2.7))
    for component, group in selected.groupby("component", sort=False):
        color = _palette(str(group.iloc[0]["dataset"]))
        ordered = group.sort_values("spec_index")
        ax.plot(ordered["coefficient"], [y_positions[component]] * len(ordered), color=color, alpha=0.45, lw=1.2)
        for _, row in ordered.iterrows():
            ax.scatter(
                [row["coefficient"]],
                [y_positions[component]],
                s=32,
                marker="o",
                facecolors=color,
                edgecolors=color,
                linewidths=0.9,
                zorder=3,
            )
    ax.axvline(0.0, color=CHARCOAL, lw=0.9)
    ax.set_yticks(list(y_positions.values()), list(y_positions.keys()), fontsize=7)
    ax.set_xticks(range(4), ["fixed 2mm", "fixed 4mm", "template 2mm", "template 4mm"])
    ax.set_xlabel("Specification index (points show coefficient; x positions are offset only for readability)", color=CHARCOAL)
    ax.set_title("Specification consistency of top absolute network-wise components", loc="left", color=CHARCOAL, pad=12)
    ax.text(
        0.0,
        1.01,
        "Fixed/template and 2/4 mm are correlated specifications, not independent replications; zero line is the sign reference.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=GREY,
        fontsize=8,
    )
    # Use coefficient values on the x axis rather than specification index. A
    # compact right-side annotation preserves the specification order.
    ax.set_xticks([])
    for component, group in selected.groupby("component", sort=False):
        for _, row in group.sort_values("spec_index").iterrows():
            x = _as_float(row["coefficient"])
            ax.text(x, y_positions[component] + 0.14, f"{row['spec']}: {x:.2f}", fontsize=6, color=CHARCOAL, rotation=20, ha="center")
    ax.grid(axis="x", color=GREY_LIGHT, lw=0.65)
    ax.set_axisbelow(True)
    caption = (
        f"run={run_id}; source=primary_localisation.csv.gz; selection=top 12 absolute reference components at 2mm bilateral; "
        "comparison across hcp151_fixed/template_matching and 2/4mm; no independence claim"
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    return _save(fig, out_dir, "specification_consistency", caption)


def plot_auxiliary_profile_overview(
    auxiliary: pd.DataFrame,
    out_dir: Path,
    *,
    run_id: str,
    n_perm: int,
) -> dict[str, str]:
    """Plot exploratory profile permutation p-values without gating localisation."""

    frame = auxiliary.loc[
        auxiliary["method"].eq("hcp151_fixed")
        & auxiliary["smoothing_mm"].eq(2)
        & auxiliary["hemisphere_mode"].eq("bilateral")
        & auxiliary["network_resolution"].eq(8)
    ].copy()
    frame["p_perm"] = pd.to_numeric(frame.get("p_perm"), errors="coerce")
    frame = frame.loc[np.isfinite(frame["p_perm"].to_numpy(dtype=float))].copy()
    if frame.empty:
        fig, ax = _base_figure()
        _unavailable_panel(ax, "no complete exploratory profile tests")
        fig.suptitle("Auxiliary profile overview", color=CHARCOAL)
        return _save(fig, out_dir, "auxiliary_profile_overview", f"run={run_id}; source=auxiliary_profiles.csv.gz; non-gating")
    frame["label"] = frame["dataset"].str.upper() + " · " + frame["estimand"] + " · " + frame["profile_scope"] + " · " + frame["region"]
    frame = frame.sort_values(["p_perm", "dataset", "estimand", "profile_scope", "region"], kind="stable").reset_index(drop=True)
    frame["y"] = np.arange(len(frame))[::-1]
    fig, ax = _base_figure(width=13, height=max(6.5, 0.25 * len(frame) + 2.6))
    for _, row in frame.iterrows():
        color = _palette(str(row["dataset"]))
        ax.scatter([row["p_perm"]], [row["y"]], s=34, color=color, edgecolors=CHARCOAL, linewidths=0.5)
    ax.axvline(0.05, color=CHARCOAL, ls="--", lw=0.8)
    ax.axvline(0.10, color=GREY, ls=":", lw=0.9)
    ax.set_xlim(0, 1)
    ax.set_yticks(frame["y"], frame["label"], fontsize=7)
    ax.set_xlabel("Freedman–Lane permutation p-value", color=CHARCOAL)
    ax.set_title("Auxiliary whole-hippocampus and subregion profile overview", loc="left", color=CHARCOAL, pad=12)
    ax.text(
        0.0,
        1.01,
        f"Exploratory and non-gating; plus-one p-values from n_perm={n_perm}; dashed/ dotted guides are descriptive only.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=GREY,
        fontsize=8,
    )
    ax.grid(axis="x", color=GREY_LIGHT, lw=0.65)
    ax.set_axisbelow(True)
    caption = (
        f"run={run_id}; source=auxiliary_profiles.csv.gz; hcp151_fixed, 2mm, bilateral, 8net reference; "
        f"n_perm={n_perm}; auxiliary permutation profile p-values are non-confirmatory and never gate localisation"
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    return _save(fig, out_dir, "auxiliary_profile_overview", caption)


def plot_all_figures(
    primary: pd.DataFrame,
    descriptives: pd.DataFrame,
    auxiliary: pd.DataFrame,
    out_dir: Path,
    *,
    run_id: str,
    n_perm: int,
) -> dict[str, dict[str, str]]:
    """Render the five core figures in a stable order."""

    consistency = plot_specification_consistency(primary, out_dir, run_id=run_id)
    return {
        "primary_effect_ci": plot_primary_effect_ci(primary, out_dir, run_id=run_id),
        "primary_localisation_heatmaps": plot_primary_localisation_heatmaps(primary, out_dir, run_id=run_id),
        "stanford_four_group_profiles": plot_stanford_four_group_profiles(descriptives, out_dir, run_id=run_id),
        "specification_consistency": consistency,
        "auxiliary_profile_overview": plot_auxiliary_profile_overview(auxiliary, out_dir, run_id=run_id, n_perm=n_perm),
    }


__all__ = [
    "plot_all_figures",
    "plot_auxiliary_profile_overview",
    "plot_primary_effect_ci",
    "plot_primary_localisation_heatmaps",
    "plot_specification_consistency",
    "plot_stanford_four_group_profiles",
]

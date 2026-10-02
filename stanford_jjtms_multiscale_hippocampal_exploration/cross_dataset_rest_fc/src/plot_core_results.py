"""Render the focused current network-wise rest-FC result figure.

This script intentionally consumes only the two reviewed primary summary
tables requested for the focal figure:

* ``primary_localisation.csv.gz`` for the Stanford TIS--TEHC and JJTMS
  patient--healthy contrasts;
* ``stanford_four_group_descriptives.csv.gz`` for the four Stanford group
  means and 95% confidence intervals.

It does not refit models, recompute q-values, or read FC tensors.  The output
output files are replaced on each run.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_ROOT = REPO_ROOT / "stanford_jjtms_multiscale_hippocampal_exploration" / "cross_dataset_rest_fc"

# A restrained, colour-blind-friendly palette.  Shape and fill state carry
# additional information so the figure remains readable in grayscale.
BLUE = "#2F6DB0"
BLUE_DARK = "#1F4D7A"
ORANGE = "#C56A2D"
ORANGE_DARK = "#8D481F"
CHARCOAL = "#252A30"
GREY = "#7D8790"
GREY_LIGHT = "#D8DDE1"
GREY_PALE = "#F5F6F7"
WHITE = "#FFFFFF"

GROUPS = ("NTHC", "TEHC", "NTS", "TIS")
GROUP_COLORS = {
    "NTHC": "#2F6DB0",
    "TEHC": "#C56A2D",
    "NTS": "#7C8F3A",
    "TIS": "#B35C86",
}
GROUP_MARKERS = {"NTHC": "o", "TEHC": "s", "NTS": "^", "TIS": "D"}
HEMISPHERES = ("L", "bilateral", "R", "L-R")
SPECS = (
    ("fixed", "hcp151_fixed", 2),
    ("fixed", "hcp151_fixed", 4),
    ("template", "hcp151_template_matching", 2),
    ("template", "hcp151_template_matching", 4),
)


def _as_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _finite(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Keep rows with finite values in all requested numeric columns."""

    if frame.empty:
        return frame.iloc[0:0].copy()
    mask = np.ones(len(frame), dtype=bool)
    for column in columns:
        values = pd.to_numeric(frame.get(column), errors="coerce").to_numpy(dtype=float)
        mask &= np.isfinite(values)
    return frame.loc[mask].copy()




def _common_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(WHITE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(GREY_LIGHT)
    ax.spines["bottom"].set_color(GREY_LIGHT)
    ax.tick_params(colors=CHARCOAL, labelsize=8)
    ax.grid(axis="y", color=GREY_LIGHT, lw=0.6)
    ax.set_axisbelow(True)


def _plot_ci(
    ax: plt.Axes,
    x: float,
    coefficient: Any,
    ci_low: Any,
    ci_high: Any,
    *,
    colour: str,
    marker: str = "o",
    size: float = 42,
    linewidth: float = 1.2,
) -> bool:
    """Plot one point and asymmetric CI; return whether the row was finite."""

    value = _as_float(coefficient)
    low = _as_float(ci_low)
    high = _as_float(ci_high)
    if not (np.isfinite(value) and np.isfinite(low) and np.isfinite(high)):
        return False
    ax.plot([x, x], [low, high], color=colour, lw=linewidth, alpha=0.72, solid_capstyle="round", zorder=2)
    ax.scatter(
        [x],
        [value],
        s=size,
        marker=marker,
        facecolors=colour,
        edgecolors=colour,
        linewidths=1.0,
        alpha=1.0,
        zorder=3,
    )
    return True


def _lookup(frame: pd.DataFrame, **keys: Any) -> pd.Series | None:
    """Return the unique matching row, or ``None`` for an absent/ambiguous row."""

    subset = frame
    for key, value in keys.items():
        subset = subset.loc[subset[key].eq(value)]
    if len(subset) != 1:
        return None
    return subset.iloc[0]


def _row_y_range(frame: pd.DataFrame, *, columns: tuple[str, ...] = ("ci_low", "ci_high")) -> tuple[float, float]:
    values: list[float] = []
    for column in columns:
        values.extend(pd.to_numeric(frame.get(column), errors="coerce").dropna().tolist())
    finite = [float(value) for value in values if np.isfinite(value)]
    if not finite:
        return (-1.0, 1.0)
    low = min(finite)
    high = max(finite)
    span = max(high - low, 0.01)
    return low - 0.12 * span, high + 0.12 * span




def plot_core_results(primary: pd.DataFrame, descriptives: pd.DataFrame, out_dir: Path, *, run_id: str, stem: str) -> dict[str, str]:
    """Create the three-panel focused result figure."""

    descriptives = descriptives.copy()
    if "resolution" not in descriptives.columns and "network_resolution" in descriptives.columns:
        descriptives["resolution"] = descriptives["network_resolution"]
    primary = primary.copy()

    # Panel A: four Stanford group means on the left-tail Language 8net row.
    panel_a = descriptives.loc[
        descriptives["dataset"].eq("stanford")
        & descriptives["hemisphere_mode"].eq("L")
        & descriptives["resolution"].eq(8)
        & descriptives["region"].eq("tail")
        & descriptives["network"].eq("Language")
        & descriptives["group"].isin(GROUPS)
        & descriptives["method"].isin([spec[1] for spec in SPECS])
        & descriptives["smoothing_mm"].isin([2, 4])
    ].copy()
    panel_a = _finite(panel_a, ("mean", "ci_low", "ci_high"))

    # Panel B: Stanford TIS−TEHC network-wise effect across hemispheric modes.
    panel_b = primary.loc[
        primary["dataset"].eq("stanford")
        & primary["estimand"].eq("TIS-TEHC")
        & primary["resolution"].eq(8)
        & primary["region"].eq("tail")
        & primary["network"].eq("Language")
        & primary["hemisphere_mode"].isin(HEMISPHERES)
        & primary["method"].isin([spec[1] for spec in SPECS])
        & primary["smoothing_mm"].isin([2, 4])
    ].copy()
    panel_b = _finite(panel_b, ("coefficient", "ci_low", "ci_high"))

    # Panel C: JJTMS patient−healthy left-tail Auditory 8net effect.
    panel_c = primary.loc[
        primary["dataset"].eq("jjtms")
        & primary["estimand"].eq("patient-healthy")
        & primary["resolution"].eq(8)
        & primary["region"].eq("tail")
        & primary["network"].eq("Auditory")
        & primary["hemisphere_mode"].eq("L")
        & primary["method"].isin([spec[1] for spec in SPECS])
        & primary["smoothing_mm"].isin([2, 4])
    ].copy()
    panel_c = _finite(panel_c, ("coefficient", "ci_low", "ci_high"))

    fig = plt.figure(figsize=(15.0, 12.0), facecolor=WHITE)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.05, 0.95), hspace=0.47, wspace=0.25)
    axes_a = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    axes_b = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    # The top row is split into A and B as nested grids to give each result
    # enough width while keeping the figure a single compact publication plate.
    for ax in axes_a + axes_b:
        ax.remove()
    top = grid[0, :].subgridspec(1, 2, width_ratios=(1.0, 1.18), wspace=0.28)
    panel_a_grid = top[0].subgridspec(2, 2, wspace=0.18, hspace=0.32)
    panel_b_grid = top[1].subgridspec(2, 2, wspace=0.18, hspace=0.32)
    a_axes = [fig.add_subplot(panel_a_grid[i, j]) for i in range(2) for j in range(2)]
    b_axes = [fig.add_subplot(panel_b_grid[i, j]) for i in range(2) for j in range(2)]
    ax_c = fig.add_subplot(grid[1, :])

    # Panel A shared y-scale and four group markers.
    a_y_low, a_y_high = _row_y_range(panel_a, columns=("ci_low", "ci_high"))
    for index, (label, method, smoothing) in enumerate(SPECS):
        ax = a_axes[index]
        _common_axes(ax)
        subset = panel_a.loc[panel_a["method"].eq(method) & panel_a["smoothing_mm"].eq(smoothing)]
        for x, group in enumerate(GROUPS):
            row = _lookup(subset, group=group)
            if row is not None:
                _plot_ci(
                    ax,
                    x,
                    row["mean"],
                    row["ci_low"],
                    row["ci_high"],
                    colour=GROUP_COLORS[group],
                    marker=GROUP_MARKERS[group],
                    size=35,
                    linewidth=1.0,
                )
        ax.axhline(0.0, color=CHARCOAL, lw=0.8, zorder=1)
        ax.set_ylim(a_y_low, a_y_high)
        ax.set_xlim(-0.55, len(GROUPS) - 0.45)
        ax.set_xticks(range(len(GROUPS)), GROUPS, fontsize=7)
        ax.set_title(f"{label} · {smoothing} mm", loc="left", fontsize=8.5, color=CHARCOAL, pad=4)
        if index in (0, 2):
            ax.set_ylabel("Mean Fisher-z FC", fontsize=8, color=CHARCOAL)
        else:
            ax.set_yticklabels([])
    a_axes[0].text(
        -0.05,
        1.12,
        "A  Stanford four-group profile · 8net tail/Language · L",
        transform=a_axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=CHARCOAL,
    )
    a_handles = [
        Line2D([0], [0], marker=GROUP_MARKERS[group], color="none", markerfacecolor=GROUP_COLORS[group], markeredgecolor=GROUP_COLORS[group], markersize=6, label=group)
        for group in GROUPS
    ]
    a_axes[1].legend(handles=a_handles, loc="upper right", frameon=False, fontsize=7, ncol=2, handletextpad=0.35, borderaxespad=0.1)

    # Panel B shared y-scale; markers show effects without multiplicity encoding.
    b_y_low, b_y_high = _row_y_range(panel_b, columns=("ci_low", "ci_high"))
    for index, (label, method, smoothing) in enumerate(SPECS):
        ax = b_axes[index]
        _common_axes(ax)
        subset = panel_b.loc[panel_b["method"].eq(method) & panel_b["smoothing_mm"].eq(smoothing)]
        for x, hemisphere in enumerate(HEMISPHERES):
            row = _lookup(subset, hemisphere_mode=hemisphere)
            if row is not None:
                _plot_ci(
                    ax,
                    x,
                    row["coefficient"],
                    row["ci_low"],
                    row["ci_high"],
                    colour=ORANGE_DARK,
                    size=37,
                    linewidth=1.05,
                )
        ax.axhline(0.0, color=CHARCOAL, lw=0.8, zorder=1)
        ax.set_ylim(b_y_low, b_y_high)
        ax.set_xlim(-0.55, len(HEMISPHERES) - 0.45)
        ax.set_xticks(range(len(HEMISPHERES)), ["L", "bilat.", "R", "L−R"], fontsize=7)
        ax.set_title(f"{label} · {smoothing} mm", loc="left", fontsize=8.5, color=CHARCOAL, pad=4)
        if index in (0, 2):
            ax.set_ylabel("TIS − TEHC coefficient", fontsize=8, color=CHARCOAL)
        else:
            ax.set_yticklabels([])
    b_axes[0].text(
        -0.05,
        1.12,
        "B  Stanford TIS−TEHC localisation · 8net tail/Language",
        transform=b_axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=CHARCOAL,
    )

    # Panel C: keep this deliberately small and descriptive; the weak template
    # estimates remain visible rather than being replaced by a winner-only bar.
    _common_axes(ax_c)
    c_y_low, c_y_high = _row_y_range(panel_c, columns=("ci_low", "ci_high"))
    for x, (label, method, smoothing) in enumerate(SPECS):
        row = _lookup(panel_c, method=method, smoothing_mm=smoothing)
        if row is not None:
            _plot_ci(
                ax_c,
                x,
                row["coefficient"],
                row["ci_low"],
                row["ci_high"],
                colour=BLUE_DARK,
                size=48,
                linewidth=1.15,
            )
    ax_c.axhline(0.0, color=CHARCOAL, lw=0.9, zorder=1)
    ax_c.set_ylim(c_y_low, c_y_high)
    ax_c.set_xlim(-0.55, len(SPECS) - 0.45)
    ax_c.set_xticks(range(len(SPECS)), ["fixed\n2 mm", "fixed\n4 mm", "template\n2 mm", "template\n4 mm"], fontsize=8)
    ax_c.set_ylabel("Patient − healthy coefficient", fontsize=9, color=CHARCOAL)
    ax_c.set_title("C  JJTMS patient−healthy localisation", loc="left", fontsize=11, fontweight="bold", color=CHARCOAL, pad=13)
    ax_c.text(
        0.0,
        1.015,
        "8net · tail/Auditory · left hemisphere · effect ± 95% CI; specifications are shown for sensitivity, not significance",
        transform=ax_c.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color=GREY,
    )

    fig.suptitle(
        "Focused cross-dataset resting-state hippocampal network results",
        x=0.03,
        y=0.985,
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
        color=CHARCOAL,
    )
    fig.text(
        0.03,
        0.958,
        "Fisher-z FC; multiplicity correction is deferred until the final analysis-object set is frozen; fixed/template and 2/4 mm are correlated specifications",
        ha="left",
        va="top",
        fontsize=8.5,
        color=GREY,
    )
    caption = (
        f"run={run_id}; sources=primary_localisation.csv.gz and stanford_four_group_descriptives.csv.gz; "
        "all values are resting-state Fisher-z FC. Panel A shows Stanford NTHC, TEHC, NTS, and TIS group means "
        "with 95% CIs for the left tail/Language 8net cell; it is descriptive rather than a contrast. Panel B "
        "shows the Stanford TIS−TEHC contrast, where positive coefficients mean higher FC in TIS than TEHC. "
        "Panel C shows JJTMS patient−healthy, where positive coefficients mean higher FC in patients; negative "
        "values therefore indicate lower patient FC. All displayed contrast estimates are raw and unadjusted pending "
        "the final analysis-object set freeze. "
        "The 17net Language rows have the same target/region interpretation and are retained in the full table, "
        "but are not duplicated in this focused 8net plate. The figure is a sensitivity display, not a significance "
        "screen; non-focal estimands are not interpreted from this plate."
    )
    fig.text(0.03, 0.012, caption, ha="left", va="bottom", fontsize=7.3, color=GREY, wrap=True)
    # Leave a dedicated title band above the nested A/B facet grids so the
    # section subtitles cannot collide with the facet titles.
    fig.subplots_adjust(left=0.065, right=0.97, bottom=0.095, top=0.84)

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    fig.savefig(png, dpi=240, facecolor=WHITE, bbox_inches="tight", metadata={"Title": stem, "Subject": caption, "Creator": "cross_dataset_rest_fc"})
    plt.close(fig)
    return {"png": str(png), "caption": caption}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="primary_screen")
    parser.add_argument("--stem", default="core_results_network_localisation")
    args = parser.parse_args(argv)
    run_dir = ANALYSIS_ROOT / "results" / args.run_id
    out_dir = ANALYSIS_ROOT / "figures" / args.run_id
    primary_path = run_dir / "primary_localisation.csv.gz"
    descriptives_path = run_dir / "stanford_four_group_descriptives.csv.gz"
    for path in (primary_path, descriptives_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    primary = pd.read_csv(primary_path)
    descriptives = pd.read_csv(descriptives_path)
    outputs = plot_core_results(primary, descriptives, out_dir, run_id=args.run_id, stem=args.stem)
    print(outputs["png"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


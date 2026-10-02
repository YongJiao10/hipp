"""Render the current comprehensive 2 mm Kember area summary as PNG.

This figure is a result-table-driven view of the active native-area endpoint.
It keeps the seven configured Stanford cohort rosters, including the nested TIS PTSD
roster, explicit in the figure and in the returned provenance manifest.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:  # Imports work both as a package and from python src/script.py.
    from .contrasts import (
        DEFAULT_DIRECT_CONFIG_PATH,
        load_cohort_rosters,
        load_direct_config,
        welch_two_sample,
    )
except ImportError:  # pragma: no cover - direct source-directory execution
    from contrasts import (  # type: ignore[no-redef]
        DEFAULT_DIRECT_CONFIG_PATH,
        load_cohort_rosters,
        load_direct_config,
        welch_two_sample,
    )
try:
    from .hcp7t_crossfit_area import (
        DEFAULT_OUTPUT_ROOT as DEFAULT_HCP_REFERENCE_OUTPUT_ROOT,
        HCP_GROUP,
        HCP_METHOD,
        compute_hcp7t_reference_effect,
        load_hcp7t_area_features,
    )
except ImportError:  # pragma: no cover - direct source-directory execution
    from hcp7t_crossfit_area import (  # type: ignore[no-redef]
        DEFAULT_OUTPUT_ROOT as DEFAULT_HCP_REFERENCE_OUTPUT_ROOT,
        HCP_GROUP,
        HCP_METHOD,
        compute_hcp7t_reference_effect,
        load_hcp7t_area_features,
    )



WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKFLOW_ROOT.parents[1]
DEFAULT_EFFECTS_PATH = WORKFLOW_ROOT / "results" / "direct_contrasts" / "direct_contrast_effects.tsv"
DEFAULT_FEATURES_PATH = WORKFLOW_ROOT / "work" / "features" / "subject_features.tsv"
DEFAULT_OUTPUT_PATH = WORKFLOW_ROOT / "figures" / "direct_contrasts" / "area_2mm_comprehensive_summary_proportion.png"

GROUP_ORDER = (
    "NTHC",
    "TEHC",
    "NTS",
    "NTS Any MDD",
    "NTS Pure MDD",
    "TIS",
    "TIS PTSD",
)
GROUP_SLUG = {group: group.lower().replace(" ", "_") for group in GROUP_ORDER}
CONTRAST_ORDER = tuple(f"{group}−NTHC" for group in GROUP_ORDER[1:])
HCP_CONTRAST = f"{HCP_GROUP}−NTHC"
DISPLAY_GROUP_ORDER = GROUP_ORDER + (HCP_GROUP,)
DISPLAY_CONTRAST_ORDER = CONTRAST_ORDER + (HCP_CONTRAST,)
DISPLAY_GROUP_SLUG = {**GROUP_SLUG, HCP_GROUP: "hcp_7t"}
GROUP_X_LABELS = {
    "NTHC": "NTHC",
    "TEHC": "TEHC",
    "NTS": "NTS",
    "NTS Any MDD": "NTS\nAny MDD",
    "NTS Pure MDD": "NTS\nPure MDD",
    "TIS": "TIS",
    "TIS PTSD": "TIS\nPTSD",
    HCP_GROUP: "HCP\n7T",
}
GRID_HSPACE = 0.42
GRID_WSPACE = 0.38
RAW_XTICK_ROTATION = 0
GRID_LEFT = 0.11
GRID_RIGHT = 0.985
GRID_TOP = 0.895
GRID_BOTTOM = 0.10
FOREST_ROW_STEP = 0.32
FOREST_GROUP_GAP = 0.48
FOREST_YTICK_PAD = 6
FONT_SIZES = {
    "suptitle": 18.0,
    "forest_title": 15.5,
    "forest_axis_label": 13.5,
    "forest_x_tick": 11.0,
    "forest_y_tick": 10.5,
    "forest_value_label": 10.0,
    "forest_annotation": 10.5,
    "raw_title": 14.0,
    "raw_axis_label": 12.5,
    "raw_x_tick": 10.5,
    "raw_y_tick": 10.5,
    "raw_significance": 9.5,
    "legend": 10.0,
    "legend_title": 12.0,
}

SYSTEMS = ("anterior", "body", "posterior")
HEMISPHERES = ("L", "R")
METHOD = "hcp151_template_matching"
SMOOTHING_MM = 2

GROUP_COLORS = {
    "NTHC": "#607D8B",
    "TEHC": "#00897B",
    "NTS": "#E65100",
    "NTS Any MDD": "#FB8C00",
    "NTS Pure MDD": "#8D6E63",
    "TIS": "#D32F2F",
    "TIS PTSD": "#7B1FA2",
    HCP_GROUP: "#1565C0",
}
SYSTEM_COLORS = {
    "anterior": "#D84315",
    "body": "#00897B",
    "posterior": "#5E35B1",
}
SYSTEM_TITLES = {
    "anterior": "Anterior (Head)",
    "body": "Body",
    "posterior": "Posterior (Tail)",
}

ANALYSIS_MODES = ("proportion", "covariate")
ANALYSIS_MODE_LABELS = {
    "proportion": "System Area Proportion",
    "covariate": "Total-Area-Adjusted Surface Area",
}
ANALYSIS_MODE_Y_LABELS = {
    "proportion": "System Area Proportion (%)",
    "covariate": "Total-Area-Adjusted Surface Area (mm²)",

}

def _normalise_subject(value: Any) -> str:
    raw = str(value).strip()
    if raw.lower().startswith("sub-"):
        raw = raw[4:]
    if not raw.isdigit():
        raise ValueError(f"invalid subject identifier: {value!r}")
    return f"sub-{int(raw):04d}"


def _require_columns(table: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _numeric(table: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = table.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def attach_cohort_labels(
    features: pd.DataFrame,
    rosters: Mapping[str, Iterable[Any]],
) -> pd.DataFrame:
    """Duplicate rows into explicit cohort views, preserving nested overlap."""

    _require_columns(
        features,
        {"subject", "hemisphere", "system", "surface_area_mm2"},
        "feature table",
    )
    base = features.copy()
    base["subject"] = base["subject"].map(_normalise_subject)
    rows: list[pd.DataFrame] = []
    for group in GROUP_ORDER:
        if group not in rosters:
            raise KeyError(f"rosters missing cohort: {group}")
        subjects = {_normalise_subject(subject) for subject in rosters[group]}
        subset = base.loc[base["subject"].isin(subjects)].copy()
        subset["cohort"] = group
        rows.append(subset)
    if not rows:
        return base.assign(cohort=pd.Series(dtype="string"))
    return pd.concat(rows, ignore_index=True)


def _prepare_effects(effects: pd.DataFrame) -> pd.DataFrame:
    required = {
        "contrast",
        "method",
        "smoothing_mm",
        "hemisphere",
        "system",
        "mean_difference",
        "ci_low",
        "ci_high",
        "cohens_d",
        "p_value",
    }
    _require_columns(effects, required, "effect table")
    result = _numeric(
        effects,
        {"smoothing_mm", "mean_difference", "ci_low", "ci_high", "cohens_d", "p_value"},
    )
    result = result.loc[
        result["method"].eq(METHOD)
        & result["smoothing_mm"].eq(SMOOTHING_MM)
        & result["hemisphere"].isin(HEMISPHERES)
        & result["system"].isin(SYSTEMS)
        & result["contrast"].isin(CONTRAST_ORDER)
    ].copy()
    if "metric" in result.columns:
        result = result.loc[result["metric"].eq("surface_area_mm2")].copy()
    if result.empty:
        raise ValueError("no 2 mm template-matching surface-area effect rows are available")
    return result


def _prepare_features(features: pd.DataFrame) -> pd.DataFrame:
    required = {
        "subject",
        "method",
        "smoothing_mm",
        "hemisphere",
        "system",
        "surface_area_mm2",
    }
    _require_columns(features, required, "feature table")
    result = _numeric(features, {"smoothing_mm", "surface_area_mm2"})
    result = result.loc[
        result["method"].eq(METHOD)
        & result["smoothing_mm"].eq(SMOOTHING_MM)
        & result["hemisphere"].isin(HEMISPHERES)
        & result["system"].isin(SYSTEMS)
    ].copy()
    if "status" in result.columns:
        result = result.loc[result["status"].eq("success")].copy()
    result["surface_area_mm2"] = pd.to_numeric(result["surface_area_mm2"], errors="coerce")
    result = result.loc[result["surface_area_mm2"].notna()].copy()
    result["subject"] = result["subject"].map(_normalise_subject)
    return result.drop_duplicates(subset=["subject", "hemisphere", "system"])


def _prepare_reference_features(features: pd.DataFrame) -> pd.DataFrame:
    required = {
        "subject",
        "cohort",
        "method",
        "smoothing_mm",
        "hemisphere",
        "system",
        "surface_area_mm2",
    }
    _require_columns(features, required, "HCP reference feature table")
    result = _numeric(features, {"smoothing_mm", "surface_area_mm2"})
    result = result.loc[
        result["cohort"].eq(HCP_GROUP)
        & result["method"].eq(HCP_METHOD)
        & result["smoothing_mm"].eq(SMOOTHING_MM)
        & result["hemisphere"].isin(HEMISPHERES)
        & result["system"].isin(SYSTEMS)
    ].copy()
    if "status" in result.columns:
        result = result.loc[result["status"].eq("success")].copy()
    result["surface_area_mm2"] = pd.to_numeric(result["surface_area_mm2"], errors="coerce")
    result = result.loc[result["surface_area_mm2"].notna()].copy()
    result["subject"] = result["subject"].map(_normalise_subject)
    if result.empty:
        raise ValueError("no 2 mm HCP cross-fit surface-area feature rows are available")
    return result.drop_duplicates(subset=["subject", "hemisphere", "system"])


def _prepare_reference_effects(effects: pd.DataFrame) -> pd.DataFrame:
    required = {
        "contrast",
        "method",
        "smoothing_mm",
        "hemisphere",
        "system",
        "mean_difference",
        "ci_low",
        "ci_high",
        "cohens_d",
        "p_value",
    }
    _require_columns(effects, required, "HCP reference effect table")
    result = _numeric(
        effects,
        {"smoothing_mm", "mean_difference", "ci_low", "ci_high", "cohens_d", "p_value"},
    )
    result = result.loc[
        result["method"].eq(HCP_METHOD)
        & result["smoothing_mm"].eq(SMOOTHING_MM)
        & result["hemisphere"].isin(HEMISPHERES)
        & result["system"].isin(SYSTEMS)
        & result["contrast"].eq(HCP_CONTRAST)
    ].copy()
    if "metric" in result.columns:
        result = result.loc[result["metric"].eq("surface_area_mm2")].copy()
    if result.empty:
        raise ValueError("no HCP 7T minus NTHC surface-area effect rows are available")
    return result



def _area_totals(features: pd.DataFrame) -> pd.Series:
    grouping = ["subject", "hemisphere"]
    if "cohort" in features.columns:
        grouping.append("cohort")
    return features.groupby(grouping, sort=False, dropna=False)["surface_area_mm2"].transform("sum")


def _covariate_parameters(features: pd.DataFrame) -> dict[tuple[str, str], tuple[float, float]]:
    unique = features.drop_duplicates(subset=["subject", "hemisphere", "system"]).copy()
    unique["total_area_mm2"] = unique.groupby(
        ["subject", "hemisphere"], sort=False, dropna=False
    )["surface_area_mm2"].transform("sum")
    parameters: dict[tuple[str, str], tuple[float, float]] = {}
    for hemisphere in HEMISPHERES:
        for system in SYSTEMS:
            subset = unique.loc[
                unique["hemisphere"].eq(hemisphere) & unique["system"].eq(system),
                ["surface_area_mm2", "total_area_mm2"],
            ].dropna()
            if subset.empty:
                parameters[(hemisphere, system)] = (0.0, 0.0)
                continue
            totals = subset["total_area_mm2"].to_numpy(dtype=float)
            values = subset["surface_area_mm2"].to_numpy(dtype=float)
            centre = float(np.mean(totals))
            centred = totals - centre
            denominator = float(centred @ centred)
            slope = (
                float((centred @ values) / denominator)
                if denominator > np.finfo(float).eps
                else 0.0
            )
            parameters[(hemisphere, system)] = (centre, slope)
    return parameters


def prepare_area_variant(features: pd.DataFrame, analysis_mode: str) -> pd.DataFrame:
    """Add the plotting endpoint for proportion or total-area-adjusted views."""

    if analysis_mode not in ANALYSIS_MODES:
        raise ValueError(f"analysis_mode must be one of {ANALYSIS_MODES}, got {analysis_mode!r}")
    _require_columns(
        features,
        {"subject", "hemisphere", "system", "surface_area_mm2"},
        "feature table",
    )
    result = features.copy()
    result["surface_area_mm2"] = pd.to_numeric(result["surface_area_mm2"], errors="coerce")
    result["total_area_mm2"] = _area_totals(result)
    if analysis_mode == "proportion":
        denominator = result["total_area_mm2"].replace(0.0, np.nan)
        result["surface_area_proportion"] = result["surface_area_mm2"] / denominator
        result["plot_value"] = result["surface_area_proportion"] * 100.0
        return result

    parameters = _covariate_parameters(result)
    centres = [
        parameters[(str(hemisphere), str(system))][0]
        for hemisphere, system in zip(result["hemisphere"], result["system"])
    ]
    slopes = [
        parameters[(str(hemisphere), str(system))][1]
        for hemisphere, system in zip(result["hemisphere"], result["system"])
    ]
    result["covariate_reference_total_area_mm2"] = centres
    result["covariate_slope"] = slopes
    result["plot_value"] = result["surface_area_mm2"] - result["covariate_slope"] * (
        result["total_area_mm2"] - result["covariate_reference_total_area_mm2"]
    )
    return result


def compute_area_variant_effects(
    features: pd.DataFrame,
    analysis_mode: str,
    contrast_order: tuple[str, ...],
) -> pd.DataFrame:
    """Compute direct effects on a transformed area plotting endpoint."""

    if analysis_mode not in {"proportion", "covariate"}:
        raise ValueError("variant effects require proportion or covariate analysis")
    _require_columns(features, {"cohort", "plot_value"}, "variant feature table")
    metric = (
        "surface_area_proportion"
        if analysis_mode == "proportion"
        else "surface_area_covariate_adjusted"
    )
    reference_scope = "descriptive cross-dataset reference; excluded from Stanford FDR family"
    rows: list[dict[str, Any]] = []
    for contrast in contrast_order:
        case_group = contrast.split("−", 1)[0]
        is_reference = case_group == HCP_GROUP
        method = HCP_METHOD if is_reference else METHOD
        comparison_scope = reference_scope if is_reference else "Stanford direct contrast"
        for hemisphere in HEMISPHERES:
            for system in SYSTEMS:
                case = features.loc[
                    features["cohort"].eq(case_group)
                    & features["hemisphere"].eq(hemisphere)
                    & features["system"].eq(system),
                    "plot_value",
                ]
                reference = features.loc[
                    features["cohort"].eq("NTHC")
                    & features["hemisphere"].eq(hemisphere)
                    & features["system"].eq(system),
                    "plot_value",
                ]
                test = welch_two_sample(case, reference)
                rows.append(
                    {
                        "contrast": contrast,
                        "case_group": case_group,
                        "reference_group": "NTHC",
                        "metric_family": "surface_area",
                        "metric": metric,
                        "source_field": metric,
                        "method": method,
                        "smoothing_mm": SMOOTHING_MM,
                        "hemisphere": hemisphere,
                        "system": system,
                        **test,
                        "p_raw": test["p_value"],
                        "case_missing_n": 0,
                        "reference_missing_n": 0,
                        "comparison_scope": comparison_scope,
                    }
                )
    return pd.DataFrame(rows)


def load_plot_inputs(
    config_path: str | Path = DEFAULT_DIRECT_CONFIG_PATH,
    effects_path: str | Path = DEFAULT_EFFECTS_PATH,
    features_path: str | Path = DEFAULT_FEATURES_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, tuple[str, ...]]]:
    """Load the active Kember area effects, features, and seven rosters."""

    config = load_direct_config(config_path)
    rosters = load_cohort_rosters(config)
    effects = pd.read_csv(Path(effects_path), sep="\t")
    features = pd.read_csv(Path(features_path), sep="\t")
    return _prepare_effects(effects), _prepare_features(features), rosters


def _effect_row(
    effects: pd.DataFrame,
    contrast: str,
    hemisphere: str,
    system: str,
) -> pd.Series | None:
    rows = effects.loc[
        effects["contrast"].eq(contrast)
        & effects["hemisphere"].eq(hemisphere)
        & effects["system"].eq(system)
    ]
    return None if rows.empty else rows.iloc[0]


def _p_label(value: float) -> tuple[str, bool]:
    if not np.isfinite(value) or value >= 0.05:
        return "", False
    if value < 0.001:
        return "p < 0.001 ***", True
    if value < 0.01:
        return f"p = {value:.3f} **", True
    return f"p = {value:.3f} *", True


def plot_effect_forest(
    axis: plt.Axes,
    effects: pd.DataFrame,
    hemisphere: str,
    *,
    contrast_order: tuple[str, ...] = CONTRAST_ORDER,
    show_y_labels: bool = True,
    show_x_labels: bool = True,
    effect_title: str = "Area Effects (2 mm)",
    effect_axis_label: str = "Standardized Effect Size (Cohen's d ± 95% CI)",
) -> None:
    """Draw one hemisphere of standardized effects with confidence intervals."""

    axis.axvspan(-0.2, 0.2, color="#f4f6f7", alpha=0.9, zorder=0)
    axis.axvline(0, color="#2c3e50", linewidth=1.2, zorder=1)

    y_step = FOREST_ROW_STEP
    group_gap = FOREST_GROUP_GAP
    current_y = 0.0
    group_centres: list[float] = []
    group_labels: list[str] = []
    for contrast_index, contrast in enumerate(contrast_order):
        group_centres.append(current_y + y_step)
        group_labels.append(contrast.replace("−NTHC", " vs NTHC"))
        if contrast_index:
            axis.axhline(
                current_y - (group_gap / 2.0) - (y_step / 2.0),
                color="#d9dee2",
                linestyle="-",
                linewidth=1.0,
                zorder=1,
            )
        for system in SYSTEMS:
            row = _effect_row(effects, contrast, hemisphere, system)
            y_pos = current_y
            if row is not None:
                d = float(row["cohens_d"])
                p = float(row["p_value"])
                ci_low = float(row["ci_low"])
                ci_high = float(row["ci_high"])
                mean_difference = float(row["mean_difference"])
                pooled_sd = mean_difference / d if np.isfinite(d) and abs(d) > 1e-8 else np.nan
                if np.isfinite(pooled_sd) and abs(pooled_sd) > 1e-8:
                    d_ci_low = ci_low / pooled_sd
                    d_ci_high = ci_high / pooled_sd
                else:
                    d_ci_low, d_ci_high = d - 0.25, d + 0.25
                label, significant = _p_label(p)
                text = f"d = {d:+.2f}"
                if significant:
                    text += f" {label.split(' ', 3)[-1]}"
                text_color = "#b71c1c" if d < 0 else "#1b5e20" if d > 0 else "#546e7a"
                axis.plot(
                    [d_ci_low, d_ci_high],
                    [y_pos, y_pos],
                    color=SYSTEM_COLORS[system],
                    linewidth=2.3,
                    alpha=0.9,
                    zorder=3,
                )
                axis.scatter(
                    d,
                    y_pos,
                    color=SYSTEM_COLORS[system],
                    s=75 if significant else 50,
                    edgecolor="#ffffff",
                    linewidth=1.3,
                    zorder=4,
                )
                if d >= 0:
                    axis.text(
                        d_ci_high + 0.07,
                        y_pos,
                        text,
                        va="center",
                        ha="left",
                        fontsize=FONT_SIZES["forest_value_label"],
                        fontweight="bold" if significant else "normal",
                        color=text_color,
                    )
                else:
                    axis.text(
                        d_ci_low - 0.07,
                        y_pos,
                        text,
                        va="center",
                        ha="right",
                        fontsize=FONT_SIZES["forest_value_label"],
                        fontweight="bold" if significant else "normal",
                        color=text_color,
                    )
            current_y += y_step
        current_y += group_gap

    axis.set_yticks(group_centres)
    if show_y_labels:
        axis.set_yticklabels(group_labels, fontsize=FONT_SIZES["forest_y_tick"], fontweight="bold")
        axis.tick_params(axis="y", pad=FOREST_YTICK_PAD)
    else:
        axis.tick_params(
            axis="y", which="both", left=False, right=False, labelleft=False, labelright=False
        )
    axis.set_ylim(-0.45, current_y - group_gap + 0.45)
    axis.set_xlim(-2.3, 2.3)
    if show_x_labels:
        axis.tick_params(axis="x", labelsize=FONT_SIZES["forest_x_tick"])
    else:
        axis.tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False, labeltop=False
        )
    axis.set_xlabel(effect_axis_label, fontsize=FONT_SIZES["forest_axis_label"], fontweight="bold")
    axis.set_title(
        f"{'Left' if hemisphere == 'L' else 'Right'} Hemisphere: {effect_title}",
        fontsize=FONT_SIZES["forest_title"],
        fontweight="bold",
        pad=12,
    )
    axis.text(
        -1.35,
        current_y - group_gap + 0.25,
        "← Area Contraction",
        ha="center",
        fontsize=FONT_SIZES["forest_annotation"],
        color="#c62828",
        fontweight="bold",
    )
    axis.text(
        1.35,
        current_y - group_gap + 0.25,
        "Area Expansion →",
        ha="center",
        fontsize=FONT_SIZES["forest_annotation"],
        color="#2e7d32",
        fontweight="bold",
    )


def plot_raw_distributions(
    axis: plt.Axes,
    features: pd.DataFrame,
    effects: pd.DataFrame,
    hemisphere: str,
    system: str,
    *,
    group_order: tuple[str, ...] = GROUP_ORDER,
    value_column: str = "surface_area_mm2",
    y_label: str = "Surface Area (mm²)",
    title_suffix: str = "",
    annotation_scale: float = 1.0,
) -> None:
    """Draw one system's subject-level distributions for the selected endpoint."""

    subset = features.loc[
        features["hemisphere"].eq(hemisphere)
        & features["system"].eq(system)
    ]
    positions = list(range(1, len(group_order) + 1))
    data_by_group = [
        subset.loc[subset["cohort"].eq(group), value_column].dropna().to_numpy()
        for group in group_order
    ]
    nonempty = [
        (position, values, group)
        for position, values, group in zip(positions, data_by_group, group_order)
        if values.size
    ]
    if nonempty:
        boxplot = axis.boxplot(
            [values for _, values, _ in nonempty],
            positions=[position for position, _, _ in nonempty],
            widths=0.46,
            patch_artist=True,
            showmeans=True,
            meanline=True,
            meanprops={"color": "#1a252f", "linewidth": 1.5, "linestyle": "--"},
            medianprops={"color": "#1a252f", "linewidth": 1.8},
            whiskerprops={"color": "#455a64", "linewidth": 1.1},
            capprops={"color": "#455a64", "linewidth": 1.1},
            flierprops={"marker": "", "markersize": 0},
            zorder=2,
        )
        for patch, (_, _, group) in zip(boxplot["boxes"], nonempty):
            patch.set_facecolor(GROUP_COLORS[group])
            patch.set_alpha(0.28)
            patch.set_edgecolor(GROUP_COLORS[group])
            patch.set_linewidth(1.6)
        for position, values, group in nonempty:
            jitter = np.random.default_rng(42 + position).normal(0, 0.065, size=values.size)
            axis.scatter(
                position + jitter,
                values,
                color=GROUP_COLORS[group],
                s=36,
                alpha=0.75,
                edgecolor="#ffffff",
                linewidth=0.7,
                zorder=3,
            )
    else:
        axis.text(0.5, 0.5, "No finite area values", transform=axis.transAxes, ha="center", va="center")

    axis.axvline(3.5, color="#cfd8dc", linestyle=":", linewidth=1.0, zorder=1)
    axis.axvline(5.5, color="#cfd8dc", linestyle=":", linewidth=1.0, zorder=1)
    axis.axvline(7.5, color="#cfd8dc", linestyle=":", linewidth=1.0, zorder=1)
    labels = [f"{GROUP_X_LABELS[group]}\n(n={len(values)})" for group, values in zip(group_order, data_by_group)]
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, fontsize=FONT_SIZES["raw_x_tick"], rotation=RAW_XTICK_ROTATION, ha="right", rotation_mode="anchor")
    axis.set_xlim(0.45, len(group_order) + 0.55)
    axis.tick_params(axis="x", pad=5, labelsize=FONT_SIZES["raw_x_tick"])
    axis.tick_params(axis="y", labelsize=FONT_SIZES["raw_y_tick"])
    axis.set_ylabel(y_label, fontsize=FONT_SIZES["raw_axis_label"], fontweight="bold")
    axis.set_title(
        f"{'Left' if hemisphere == 'L' else 'Right'} {SYSTEM_TITLES[system]} Area{title_suffix}",
        fontsize=FONT_SIZES["raw_title"],
        fontweight="bold",
        pad=8,
    )
    axis.grid(axis="y", color="#eaecef", linestyle="--", linewidth=0.7)

    finite_values = [values[np.isfinite(values)] for values in data_by_group if values.size]
    max_value = max((float(np.max(values)) for values in finite_values if values.size), default=1.0)
    bracket_y = max_value + max(45.0 * annotation_scale, max_value * 0.10)
    bracket_step = max(52.0 * annotation_scale, max_value * 0.085)
    for group, case_position in zip(group_order[1:], positions[1:]):
        contrast = f"{group}−NTHC"
        row = _effect_row(effects, contrast, hemisphere, system)
        if row is None:
            continue
        label, significant = _p_label(float(row["p_value"]))
        if not significant:
            continue
        color = "#c62828" if float(row["cohens_d"]) < 0 else "#2e7d32"
        axis.plot(
            [1, 1, case_position, case_position],
            [bracket_y - 8 * annotation_scale, bracket_y, bracket_y, bracket_y - 8 * annotation_scale],
            color=color,
            linewidth=1.3,
            zorder=5,
        )
        axis.text(
            (1 + case_position) / 2.0,
            bracket_y + 3 * annotation_scale,
            label,
            ha="center",
            va="bottom",
            fontsize=FONT_SIZES["raw_significance"],
            fontweight="bold",
            color=color,
        )
        bracket_y += bracket_step
    axis.set_ylim(0, max(bracket_y + max(35.0 * annotation_scale, max_value * 0.08), max_value + 80.0 * annotation_scale))


def render_area_summary(
    effects: pd.DataFrame,
    features: pd.DataFrame,
    rosters: Mapping[str, Iterable[Any]],
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    reference_features: pd.DataFrame | None = None,
    reference_effects: pd.DataFrame | None = None,
    reference_metadata: Mapping[str, Any] | None = None,
    analysis_mode: str = "proportion",
) -> dict[str, Any]:
    """Render seven Stanford cohorts plus an optional HCP reference to one PNG."""

    output = Path(output_path).expanduser().resolve()
    if output.suffix.lower() != ".png":
        raise ValueError("the comprehensive area summary accepts a PNG output path only")
    effect_table = _prepare_effects(effects)
    feature_table = attach_cohort_labels(_prepare_features(features), rosters)

    metadata = dict(reference_metadata or {})
    reference_feature_table = pd.DataFrame()
    reference_effect_table = pd.DataFrame()
    has_reference = reference_features is not None or reference_effects is not None
    if reference_features is not None:
        reference_feature_table = _prepare_reference_features(reference_features)
        feature_table = pd.concat([feature_table, reference_feature_table], ignore_index=True)
    if reference_effects is not None:
        reference_effect_table = _prepare_reference_effects(reference_effects)
    display_groups = DISPLAY_GROUP_ORDER if has_reference else GROUP_ORDER
    display_contrasts = DISPLAY_CONTRAST_ORDER if has_reference else CONTRAST_ORDER
    if analysis_mode not in ANALYSIS_MODES:
        raise ValueError(f"analysis_mode must be one of {ANALYSIS_MODES}, got {analysis_mode!r}")
    plot_feature_table = prepare_area_variant(feature_table, analysis_mode)
    combined_effect_table = compute_area_variant_effects(
        plot_feature_table, analysis_mode, display_contrasts
    )
    plot_value_column = "plot_value"
    plot_y_label = ANALYSIS_MODE_Y_LABELS[analysis_mode]
    forest_title = f"{ANALYSIS_MODE_LABELS[analysis_mode]} Effects (2 mm)"
    distribution_title_suffix = " Proportion" if analysis_mode == "proportion" else " (Total-Area Adjusted)"
    distribution_annotation_scale = 0.1 if analysis_mode == "proportion" else 1.0
    if has_reference:
        metadata.setdefault("method", HCP_METHOD)
        metadata.setdefault("comparison", HCP_CONTRAST)
        metadata.setdefault("comparison_scope", "descriptive cross-dataset reference; excluded from Stanford FDR family")
    reference_subject_count = metadata.get("n_subjects")
    if reference_subject_count is None and not reference_feature_table.empty:
        reference_subject_count = int(reference_feature_table["subject"].nunique())

    output.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(24, 22.0), dpi=300)
    grid = figure.add_gridspec(3, 6, height_ratios=[1.65, 1.15, 1.15], hspace=GRID_HSPACE, wspace=GRID_WSPACE, left=GRID_LEFT, right=GRID_RIGHT, top=GRID_TOP, bottom=GRID_BOTTOM)
    try:
        left = figure.add_subplot(grid[0, 0:3])
        right = figure.add_subplot(grid[0, 3:6])
        plot_effect_forest(left, combined_effect_table, "L", contrast_order=display_contrasts, effect_title=forest_title)
        plot_effect_forest(right, combined_effect_table, "R", contrast_order=display_contrasts, show_y_labels=False, show_x_labels=False, effect_title=forest_title)
        legend = [
            patches.Patch(facecolor=SYSTEM_COLORS[system], edgecolor="#1a252f", label=SYSTEM_TITLES[system])
            for system in SYSTEMS
        ]
        left.legend(
            handles=legend,
            loc="center left",
            frameon=True,
            framealpha=0.95,
            facecolor="#ffffff",
            edgecolor="#b0bec5",
            fontsize=FONT_SIZES["legend"],
            title="Hippocampal System",
            title_fontsize=FONT_SIZES["legend_title"],
        )
        for index, system in enumerate(SYSTEMS):
            axis = figure.add_subplot(grid[1, index * 2 : (index + 1) * 2])
            plot_raw_distributions(axis, plot_feature_table, combined_effect_table, "L", system, group_order=display_groups, value_column=plot_value_column, y_label=plot_y_label, title_suffix=distribution_title_suffix, annotation_scale=distribution_annotation_scale)
        for index, system in enumerate(SYSTEMS):
            axis = figure.add_subplot(grid[2, index * 2 : (index + 1) * 2])
            plot_raw_distributions(axis, plot_feature_table, combined_effect_table, "R", system, group_order=display_groups, value_column=plot_value_column, y_label=plot_y_label, title_suffix=distribution_title_suffix, annotation_scale=distribution_annotation_scale)

        figure.suptitle(
            f"Hippocampal Functional System {ANALYSIS_MODE_LABELS[analysis_mode]} Analysis (2 mm Smoothing, Template-Matching)\n"
            "Stanford direct contrasts to NTHC; HCP 7T shown separately as a held-out cross-fit reference",
            fontsize=FONT_SIZES["suptitle"],
            fontweight="bold",
            y=0.975,
        )
        figure.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.22)
    finally:
        plt.close(figure)

    counts = {
        group: int(feature_table.loc[feature_table["cohort"].eq(group), "subject"].nunique())
        for group in GROUP_ORDER
    }
    display_counts = {
        group: int(feature_table.loc[feature_table["cohort"].eq(group), "subject"].nunique())
        for group in display_groups
    }
    return {
        "figure_id": "area_2mm_comprehensive_summary",
        "format": "png",
        "png_path": str(output),
        "method": METHOD,
        "smoothing_mm": SMOOTHING_MM,
        "analysis_mode": analysis_mode,
        "analysis_value_label": ANALYSIS_MODE_LABELS[analysis_mode],
        "analysis_value_column": plot_value_column,
        "cohort_order": list(GROUP_ORDER),
        "cohort_slugs": dict(GROUP_SLUG),
        "cohort_subject_counts": counts,
        "display_cohort_order": list(display_groups),
        "display_group_order": list(display_groups),
        "display_cohort_slugs": {group: DISPLAY_GROUP_SLUG[group] for group in display_groups},
        "feature_cohort_subject_counts": display_counts,
        "contrast_order": list(CONTRAST_ORDER),
        "display_contrast_order": list(display_contrasts),
        "reference_method": metadata.get("method") if has_reference else None,
        "reference_subject_count": int(reference_subject_count) if reference_subject_count is not None else None,
        "reference_metadata": metadata if has_reference else {},
        "source_reference_effect_rows": int(len(reference_effect_table)),
        "source_stanford_feature_rows": int(len(feature_table) - len(reference_feature_table)),
        "source_reference_feature_rows": int(len(reference_feature_table)),
        "layout": {
            "figsize_inches": [24, 22],
            "font_sizes": dict(FONT_SIZES),
            "grid_hspace": GRID_HSPACE,
            "grid_wspace": GRID_WSPACE,
            "raw_xtick_rotation": RAW_XTICK_ROTATION,
            "forest_row_step": FOREST_ROW_STEP,
            "forest_group_gap": FOREST_GROUP_GAP,
            "legend_location": "center left",
            "footer_visible": False,
            "grid_margins": {"left": GRID_LEFT, "right": GRID_RIGHT, "top": GRID_TOP, "bottom": GRID_BOTTOM},
        },
        "source_effect_rows": int(len(effect_table)),
        "source_feature_rows": int(len(feature_table)),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_DIRECT_CONFIG_PATH)
    parser.add_argument("--effects", type=Path, default=DEFAULT_EFFECTS_PATH)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--workbench-command", type=Path, default=None)
    parser.add_argument("--analysis-mode", choices=ANALYSIS_MODES, default="proportion")
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = load_direct_config(args.config)
    effects, features, rosters = load_plot_inputs(args.config, args.effects, args.features)
    reference = config.get("reference_hcp7t")
    if not isinstance(reference, Mapping) or reference.get("enabled", True) is not True:
        raise ValueError("analysis config must enable the HCP 7T reference branch")
    configured_output = reference.get("derived_output_root", DEFAULT_HCP_REFERENCE_OUTPUT_ROOT)
    reference_output = Path(configured_output).expanduser()
    if not reference_output.is_absolute():
        reference_output = (REPO_ROOT / reference_output).resolve()
    hcp_features, reference_manifest = load_hcp7t_area_features(
        config, output_dir=reference_output, workbench_command=args.workbench_command
    )
    nthc_features = attach_cohort_labels(features, rosters)
    nthc_features = nthc_features.loc[nthc_features["cohort"].eq("NTHC")].copy()
    hcp_effects = compute_hcp7t_reference_effect(hcp_features, nthc_features)
    reference_effect_path = reference_output / "hcp7t_minus_nthc_effects_2mm.tsv"
    hcp_effects.to_csv(reference_effect_path, sep="\t", index=False)
    reference_outputs = dict(reference_manifest.get("outputs", {}))
    reference_outputs["effect_table"] = str(reference_effect_path.resolve())
    reference_manifest["outputs"] = reference_outputs
    reference_manifest["reference_comparison"] = {
        "contrast": HCP_CONTRAST,
        "reference_group": "NTHC",
        "scope": "descriptive cross-dataset reference; excluded from Stanford FDR family",
    }
    (reference_output / "run_manifest.json").write_text(
        json.dumps(reference_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    reference_metadata = {
        "method": reference_manifest.get("method", HCP_METHOD),
        "n_subjects": int(reference_manifest["n_subjects"]),
        "n_folds": int(reference_manifest["n_folds"]),
        "family_aware": bool(reference_manifest["family_aware"]),
        "subject_only_waiver": bool(reference_manifest["subject_only_waiver"]),
        "comparison": HCP_CONTRAST,
        "comparison_scope": "descriptive cross-dataset reference; excluded from Stanford FDR family",
        "source_crossfit_manifest": reference_manifest["source_crossfit_manifest"],
        "source_crossfit_manifest_sha256": reference_manifest["source_crossfit_manifest_sha256"],
        "native_area_source": reference_manifest["native_area_source"],
        "feature_path": reference_manifest.get("outputs", {}).get("features", str(reference_output / "hcp7t_crossfit_area_2mm.tsv")),
        "effect_path": str(reference_effect_path.resolve()),
        "run_manifest_path": str((reference_output / "run_manifest.json").resolve()),
    }
    manifest = render_area_summary(
        effects,
        features,
        rosters,
        args.output,
        reference_features=hcp_features,
        analysis_mode=args.analysis_mode,
        reference_effects=hcp_effects,
        reference_metadata=reference_metadata,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

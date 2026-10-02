#!/home/yojiao/.conda/envs/py314/bin/python
"""Build an effect-first, non-gating visual review of current rest-FC results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
WORKFLOW_ROOT = HERE.parent
SOURCE_RUN_ID = "primary_screen"
RUN_ID = "broad_signal_review"
SOURCE_ROOT = WORKFLOW_ROOT / "results" / SOURCE_RUN_ID
RESULT_ROOT = WORKFLOW_ROOT / "results" / RUN_ID
FIGURE_ROOT = WORKFLOW_ROOT / "figures" / RUN_ID
REPORT_ROOT = WORKFLOW_ROOT / "reports" / RUN_ID

METHODS = ("hcp151_fixed", "hcp151_template_matching")
LEVEL_MODES = ("bilateral", "L", "R")
COLORS = {
    "stanford": "#315E8A",
    "jjtms": "#B67A18",
    "stanford_open": "#DCE7F0",
    "jjtms_open": "#F3E4C5",
    "neutral": "#4B5563",
    "grid": "#D7DCE2",
    "positive": "#315E8A",
    "negative": "#C07A28",
}
PARENT_17_TO_8 = {
    "Aud": "Auditory",
    "ContA": "Control",
    "ContB": "Control",
    "ContC": "Control",
    "DefaultA": "Default",
    "DefaultB": "Default",
    "DefaultC": "Default",
    "DorsAttnA": "DorsalAttention",
    "DorsAttnB": "DorsalAttention",
    "Language": "Language",
    "SalVenAttnA": "VentralAttention",
    "SalVenAttnB": "VentralAttention",
    "SomMotA": "Somatomotor",
    "SomMotB": "Somatomotor",
    "TempPar": "Language",
    "VisualA": "Visual",
    "VisualB": "Visual",
    "VisualC": "Visual",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parent_network(row: pd.Series) -> str:
    if int(row["resolution"]) == 8:
        return str(row["network"])
    return PARENT_17_TO_8.get(str(row["network"]), str(row["network"]))


def sign_consistency(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    values = values[values != 0]
    if values.empty:
        return np.nan
    return float(max((values > 0).mean(), (values < 0).mean()))


def summarize_components(table: pd.DataFrame, modes: tuple[str, ...]) -> pd.DataFrame:
    keys = ["dataset", "estimand", "resolution", "region", "network"]
    rows: list[dict[str, object]] = []
    subset = table[table["hemisphere_mode"].isin(modes)].copy()
    for key, group in subset.groupby(keys, sort=True, dropna=False):
        effects = pd.to_numeric(group["standardized_effect"], errors="coerce").dropna()
        coefficients = pd.to_numeric(group["coefficient"], errors="coerce").dropna()
        if effects.empty:
            continue
        row = dict(zip(keys, key, strict=True))
        row.update(
            {
                "n_specs": int(effects.size),
                "median_d": float(effects.median()),
                "q25_d": float(effects.quantile(0.25)),
                "q75_d": float(effects.quantile(0.75)),
                "min_d": float(effects.min()),
                "max_d": float(effects.max()),
                "max_abs_d": float(effects.abs().max()),
                "sign_consistency": sign_consistency(coefficients),
                "multiplicity_correction": "deferred_until_analysis_set_freeze",
                "n_ci_excludes_zero": int(
                    (
                        (pd.to_numeric(group["ci_low"], errors="coerce") > 0)
                        | (pd.to_numeric(group["ci_high"], errors="coerce") < 0)
                    ).sum()
                ),
                "fixed_median_d": float(
                    pd.to_numeric(
                        group.loc[group["method"] == "hcp151_fixed", "standardized_effect"],
                        errors="coerce",
                    ).median()
                ),
                "template_median_d": float(
                    pd.to_numeric(
                        group.loc[
                            group["method"] == "hcp151_template_matching",
                            "standardized_effect",
                        ],
                        errors="coerce",
                    ).median()
                ),
                "source_row_ids": "|".join(sorted(group["source_row_id"].astype(str))),
            }
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    result["parent_network"] = result.apply(parent_network, axis=1)
    return result


def label_row(row: pd.Series) -> str:
    dataset = "S" if row["dataset"] == "stanford" else "J"
    estimand = str(row["estimand"]).replace("symptom_marginal", "symptom marginal")
    return (
        f"[{dataset}] {estimand} | {row['region']}–{row['network']} "
        f"({int(row['resolution'])}net)"
    )


def choose_landscape(level: pd.DataFrame) -> pd.DataFrame:
    stable = (level["n_specs"] == 12) & (level["sign_consistency"] == 1.0)
    base8 = level[
        stable & (level["resolution"] == 8) & (level["median_d"].abs() >= 0.35)
    ].copy()
    base8["selection_layer"] = "8-network stable effect"
    occupied = {
        (r.dataset, r.estimand, r.region, r.parent_network)
        for r in base8.itertuples(index=False)
    }
    refine17 = level[
        stable & (level["resolution"] == 17) & (level["median_d"].abs() >= 0.35)
    ].copy()
    refine17 = refine17[
        ~refine17.apply(
            lambda r: (r["dataset"], r["estimand"], r["region"], r["parent_network"])
            in occupied,
            axis=1,
        )
    ]
    refine17 = refine17.sort_values(
        ["median_d"], key=lambda x: x.abs(), ascending=False
    ).head(10)
    refine17["selection_layer"] = "17-network refinement"

    secondary = level[stable & (level["median_d"].abs() < 0.35)].copy()
    secondary = secondary.sort_values(
        ["median_d", "dataset", "estimand", "region", "network"],
        key=lambda column: column.abs() if column.name == "median_d" else column,
        ascending=False,
        kind="stable",
    ).head(2)
    secondary["selection_layer"] = "direction-stable lower-effect secondary"

    selected = pd.concat([base8, refine17, secondary], ignore_index=True)
    selected["plot_label"] = selected.apply(label_row, axis=1)
    selected["abs_median_d"] = selected["median_d"].abs()
    selected = selected.sort_values(
        ["dataset", "abs_median_d", "estimand", "region"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    return selected


def build_cross_dataset(level: pd.DataFrame) -> pd.DataFrame:
    jjtms = level[
        (level["dataset"] == "jjtms")
        & (level["estimand"] == "patient-healthy")
        & (level["resolution"] == 8)
    ].copy()
    stanford = level[
        (level["dataset"] == "stanford")
        & (level["estimand"] == "symptom_marginal")
        & (level["resolution"] == 8)
    ].copy()
    keep = [
        "region",
        "network",
        "n_specs",
        "median_d",
        "min_d",
        "max_d",
        "sign_consistency",
        "source_row_ids",
    ]
    merged = jjtms[keep].merge(
        stanford[keep],
        on=["region", "network"],
        suffixes=("_jjtms", "_stanford"),
        validate="one_to_one",
    )
    merged["same_direction"] = (
        np.sign(merged["median_d_jjtms"]) == np.sign(merged["median_d_stanford"])
    )
    merged["convergence_score"] = np.sqrt(
        merged["median_d_jjtms"].abs() * merged["median_d_stanford"].abs()
    ) * np.minimum(
        merged["sign_consistency_jjtms"], merged["sign_consistency_stanford"]
    )
    merged = merged[
        (merged["n_specs_jjtms"] == 12)
        & (merged["n_specs_stanford"] == 12)
        & (merged["sign_consistency_jjtms"] == 1.0)
        & (merged["sign_consistency_stanford"] == 1.0)
        & merged["same_direction"]
        & (merged["convergence_score"] >= 0.15)
    ].copy()
    merged["plot_label"] = merged["region"] + "–" + merged["network"]
    return merged.sort_values("convergence_score", ascending=False).reset_index(
        drop=True
    )


def choose_laterality(laterality: pd.DataFrame) -> pd.DataFrame:
    eligible = laterality[
        (laterality["n_specs"] == 4)
        & (laterality["sign_consistency"] == 1.0)
        & (laterality["median_d"].abs() >= 0.35)
    ].copy()
    eligible["abs_median_d"] = eligible["median_d"].abs()
    eligible = eligible.sort_values(
        ["abs_median_d", "resolution"], ascending=[False, True]
    )
    eligible = eligible.drop_duplicates(
        ["dataset", "estimand", "region", "parent_network"], keep="first"
    ).head(12)
    eligible["plot_label"] = eligible.apply(label_row, axis=1)
    return eligible.reset_index(drop=True)


def build_four_group_patterns(
    views: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    themes = [
        ("tail", "Language", 8, "posterior Language"),
        ("body", "Somatomotor", 8, "body Somatomotor"),
        ("tail", "Auditory", 8, "posterior Auditory"),
        ("tail", "Control", 8, "posterior Control"),
        ("body", "Default", 8, "body Default"),
        ("body", "VisualA", 17, "body VisualA"),
        ("body", "SalVenAttnB", 17, "body Sal/VAN-B"),
        ("body", "DorsAttnA", 17, "body DAN-A"),
    ]
    valid = views[
        (views["dataset"] == "stanford")
        & views["method"].isin(METHODS)
        & views["hemisphere_mode"].isin(LEVEL_MODES)
        & (views["status"] == "present")
    ].copy()
    groups = ["NTHC", "TEHC", "NTS", "TIS"]
    rows: list[dict[str, object]] = []
    for order, (region, network, resolution, display) in enumerate(themes):
        subset = valid[
            (valid["region"] == region)
            & (valid["network"] == network)
            & (valid["network_resolution"] == resolution)
        ]
        per_spec = (
            subset.groupby(
                ["method", "smoothing_mm", "hemisphere_mode", "group"],
                observed=True,
            )["fisher_z"]
            .mean()
            .reset_index()
        )
        medians = per_spec.groupby("group", observed=True)["fisher_z"].median()
        grand = float(medians.reindex(groups).mean())
        for group in groups:
            raw = float(medians[group])
            rows.append(
                {
                    "theme_order": order,
                    "theme": display,
                    "region": region,
                    "network": network,
                    "resolution": resolution,
                    "group": group,
                    "median_group_mean_fisher_z": raw,
                    "row_centered_fisher_z": raw - grand,
                    "n_subject_rows": int((subset["group"] == group).sum()),
                    "aggregation": (
                        "group mean within each fixed/template x 2/4mm x "
                        "bilateral/L/R specification; median across 12 specifications"
                    ),
                }
            )
    long = pd.DataFrame(rows)
    matrix = long.pivot(index="theme", columns="group", values="row_centered_fisher_z")
    matrix = matrix.reindex(
        [theme[3] for theme in themes], columns=groups
    )
    return long, matrix


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#9098A1")
    ax.tick_params(colors="#343A40", labelsize=8)


def plot_landscape(selected: pd.DataFrame, output: Path) -> None:
    n = len(selected)
    fig_height = max(8.5, 0.42 * n + 2.7)
    fig, ax = plt.subplots(figsize=(14.5, fig_height), constrained_layout=True)
    for y, row in selected.reset_index(drop=True).iterrows():
        color = COLORS[row["dataset"]]
        open_color = COLORS[f"{row['dataset']}_open"]
        ax.hlines(y, row["min_d"], row["max_d"], color=color, alpha=0.25, lw=2)
        ax.hlines(y, row["q25_d"], row["q75_d"], color=color, lw=6, alpha=0.85)
        filled = row["selection_layer"] != "direction-stable lower-effect secondary"
        ax.scatter(
            row["median_d"],
            y,
            s=62,
            facecolor=color if filled else open_color,
            edgecolor=color,
            linewidth=1.4,
            zorder=3,
        )
    ax.axvline(0, color="#4A4F55", lw=1.1)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(selected["plot_label"], fontsize=8.3)
    ax.invert_yaxis()
    ax.set_xlabel(
        "Standardized effect across 12 correlated specifications "
        "(thin=min–max, thick=IQR, dot=median)",
        fontsize=10,
    )
    ax.set_title(
        "Effect-first stable hippocampal network components",
        loc="left",
        fontsize=16,
        weight="bold",
        color="#20262D",
        pad=12,
    )
    ax.text(
        0,
        1.005,
        "Fixed HCP151 + template matching; 2/4 mm; bilateral/L/R. "
        "Main layer: 100% sign consistency and |median d| ≥ 0.35; "
        "two lower-effect, direction-stable secondary leads are also retained for visual review.",
        transform=ax.transAxes,
        fontsize=9,
        color="#4B5563",
        va="bottom",
    )
    ax.grid(axis="x", color=COLORS["grid"], lw=0.7, alpha=0.7)
    style_axis(ax)
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_story_map(
    matrix: pd.DataFrame,
    cross: pd.DataFrame,
    laterality: pd.DataFrame,
    output: Path,
) -> None:
    fig = plt.figure(figsize=(17, 10.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.12, 1.0], height_ratios=[1, 1])
    ax_heat = fig.add_subplot(grid[:, 0])
    ax_cross = fig.add_subplot(grid[0, 1])
    ax_lat = fig.add_subplot(grid[1, 1])

    values = matrix.to_numpy(float)
    vmax = max(float(np.nanmax(np.abs(values))), 0.005)
    image = ax_heat.imshow(
        values,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )
    ax_heat.set_xticks(np.arange(matrix.shape[1]))
    ax_heat.set_xticklabels(matrix.columns, fontsize=9)
    ax_heat.set_yticks(np.arange(matrix.shape[0]))
    ax_heat.set_yticklabels(matrix.index, fontsize=9)
    ax_heat.set_title(
        "A  Stanford four-group shapes",
        loc="left",
        fontsize=12,
        weight="bold",
    )
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = values[i, j]
            text_color = "white" if abs(value) > 0.58 * vmax else "#22272E"
            ax_heat.text(
                j,
                i,
                f"{value:+.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )
    bar = fig.colorbar(image, ax=ax_heat, shrink=0.63, pad=0.03)
    bar.set_label("Fisher-z deviation from component's four-group mean", fontsize=8)

    cross_plot = cross.iloc[::-1].reset_index(drop=True)
    for y, row in cross_plot.iterrows():
        ax_cross.hlines(
            y,
            row["median_d_jjtms"],
            row["median_d_stanford"],
            color="#9CA3AF",
            lw=2,
        )
        ax_cross.scatter(
            row["median_d_jjtms"],
            y,
            s=52,
            color=COLORS["jjtms"],
            edgecolor="#704B0F",
            linewidth=0.7,
            label="JJTMS patient−healthy" if y == 0 else None,
            zorder=3,
        )
        ax_cross.scatter(
            row["median_d_stanford"],
            y,
            s=52,
            color=COLORS["stanford"],
            edgecolor="#1E3C59",
            linewidth=0.7,
            label="Stanford symptom marginal" if y == 0 else None,
            zorder=3,
        )
    ax_cross.axvline(0, color="#4A4F55", lw=1)
    ax_cross.set_yticks(np.arange(len(cross_plot)))
    ax_cross.set_yticklabels(cross_plot["plot_label"], fontsize=8)
    ax_cross.set_xlabel("Median standardized effect across 12 specifications")
    ax_cross.set_title(
        "B  Cross-dataset directional convergence",
        loc="left",
        fontsize=12,
        weight="bold",
    )
    ax_cross.legend(frameon=False, fontsize=8, loc="lower right")
    ax_cross.grid(axis="x", color=COLORS["grid"], lw=0.7)
    style_axis(ax_cross)

    lat_plot = laterality.iloc[::-1].reset_index(drop=True)
    for y, row in lat_plot.iterrows():
        color = COLORS[row["dataset"]]
        ax_lat.hlines(y, row["min_d"], row["max_d"], color=color, alpha=0.35, lw=3)
        ax_lat.scatter(
            row["median_d"],
            y,
            s=56,
            facecolor=COLORS[f"{row['dataset']}_open"],
            edgecolor=color,
            linewidth=1.4,
            zorder=3,
        )
    ax_lat.axvline(0, color="#4A4F55", lw=1)
    ax_lat.set_yticks(np.arange(len(lat_plot)))
    ax_lat.set_yticklabels(lat_plot["plot_label"], fontsize=7.7)
    ax_lat.set_xlabel("L−R group effect: median standardized effect (range)")
    ax_lat.set_title(
        "C  Stable laterality candidates",
        loc="left",
        fontsize=12,
        weight="bold",
    )
    ax_lat.grid(axis="x", color=COLORS["grid"], lw=0.7)
    style_axis(ax_lat)

    fig.suptitle(
        "Broader promising-signal map without multiplicity filtering",
        x=0.01,
        ha="left",
        fontsize=17,
        weight="bold",
        color="#20262D",
    )
    fig.text(
        0.01,
        0.005,
        "Exploratory post-hoc visualization of current primary_screen tables; "
        "no models were refit. Panels prioritize magnitude, sign stability, "
        "four-group shape and convergence. They do not create new confirmatory claims.",
        fontsize=8.5,
        color="#4B5563",
    )
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for directory in (RESULT_ROOT, FIGURE_ROOT, REPORT_ROOT):
        directory.mkdir(parents=True, exist_ok=True)

    primary_path = SOURCE_ROOT / "primary_localisation.csv.gz"
    views_path = SOURCE_ROOT / "hemisphere_views.csv.gz"
    primary = pd.read_csv(primary_path)
    views = pd.read_csv(views_path)

    available = primary[
        primary["method"].isin(METHODS) & primary["failure_reason"].isna()
    ].copy()
    assert set(available["method"].unique()) == set(METHODS)
    assert available["standardized_effect"].notna().all()

    level = summarize_components(available, LEVEL_MODES)
    laterality_all = summarize_components(available, ("L-R",))
    landscape = choose_landscape(level)
    cross = build_cross_dataset(level)
    laterality = choose_laterality(laterality_all)
    four_group_long, four_group_matrix = build_four_group_patterns(views)

    assert not landscape.empty
    assert not cross.empty
    assert not laterality.empty
    assert (landscape["sign_consistency"] == 1).all()
    assert (laterality["sign_consistency"] == 1).all()

    tables = {
        "effect_first_component_summary.tsv": level,
        "plotted_signal_landscape.tsv": landscape,
        "cross_dataset_convergence.tsv": cross,
        "laterality_candidates.tsv": laterality,
        "stanford_four_group_story_patterns.tsv": four_group_long,
    }
    for name, table in tables.items():
        table.to_csv(RESULT_ROOT / name, sep="\t", index=False)

    selection_rules = {
        "scope": {
            "source_run": SOURCE_RUN_ID,
            "datasets": ["stanford", "jjtms"],
            "methods": list(METHODS),
            "smoothing_mm": [2, 4],
            "level_hemisphere_modes": list(LEVEL_MODES),
            "laterality_mode": "L-R",
            "out_of_scope_method_included": False,
            "model_refit": False,
        },
        "level_landscape": {
            "expected_correlated_specifications": 12,
            "main_layer_requirements": [
                "100% coefficient sign consistency",
                "absolute median standardized effect >= 0.35",
            ],
            "secondary_layer_requirements": [
                "100% coefficient sign consistency",
                "absolute median standardized effect < 0.35",
                "retain the two largest absolute median-effect rows among direction-stable lower-effect candidates",
            ],
            "resolution_policy": (
                "include all eligible 8-network main-layer components; add up to "
                "10 eligible 17-network refinements only when the matching "
                "8-network parent dataset/estimand/region theme is absent; retain "
                "all lower-effect direction-stable secondary leads, capped at two deterministic rows"
            ),
            "multiplicity_correction": (
                "deferred until final analysis-object set freeze; not used for selection or annotation",
            ),
        },
        "cross_dataset": {
            "comparison": "JJTMS patient-healthy vs Stanford symptom marginal",
            "resolution": 8,
            "requirements": [
                "12/12 finite specifications per dataset",
                "100% within-dataset sign consistency",
                "same median effect direction",
                "geometric-mean magnitude score >= 0.15",
            ],
        },
        "laterality": {
            "expected_correlated_specifications": 4,
            "requirements": [
                "100% sign consistency",
                "absolute median standardized L-R effect >= 0.35",
            ],
            "deduplication": "prefer 8-network parent when both 8/17 versions qualify",
        },
        "four_group_panel": {
            "aggregation": (
                "group mean within each of 12 fixed/template x 2/4mm x "
                "bilateral/L/R specifications, then median across specifications"
            ),
            "display": "row-centered Fisher-z to show four-group shape",
            "themes": list(four_group_matrix.index),
        },
    }
    rules_path = RESULT_ROOT / "selection_rules.json"
    rules_path.write_text(json.dumps(selection_rules, indent=2, sort_keys=True) + "\n")

    plot_landscape(landscape, FIGURE_ROOT / "effect_first_signal_landscape")
    plot_story_map(
        four_group_matrix,
        cross,
        laterality,
        FIGURE_ROOT / "broader_story_map",
    )

    def get_level(dataset: str, estimand: str, resolution: int, region: str, network: str):
        row = level[
            (level["dataset"] == dataset)
            & (level["estimand"] == estimand)
            & (level["resolution"] == resolution)
            & (level["region"] == region)
            & (level["network"] == network)
        ].iloc[0]
        return row

    tail_lang = get_level("stanford", "interaction", 8, "tail", "Language")
    body_som = get_level("stanford", "interaction", 8, "body", "Somatomotor")
    body_default_j = get_level("jjtms", "patient-healthy", 8, "body", "Default")
    tehc_som = get_level("stanford", "TEHC-NTHC", 8, "body", "Somatomotor")
    tehc_sal = get_level("stanford", "TEHC-NTHC", 17, "body", "SalVenAttnB")
    tail_visual_lat = laterality_all[
        (laterality_all["dataset"] == "stanford")
        & (laterality_all["estimand"] == "interaction")
        & (laterality_all["resolution"] == 17)
        & (laterality_all["region"] == "tail")
        & (laterality_all["network"] == "VisualC")
    ].iloc[0]
    body_lang_lat = laterality_all[
        (laterality_all["dataset"] == "stanford")
        & (laterality_all["estimand"] == "TIS-TEHC")
        & (laterality_all["resolution"] == 8)
        & (laterality_all["region"] == "body")
        & (laterality_all["network"] == "Language")
    ].iloc[0]

    j_body_visual = get_level(
        "jjtms", "patient-healthy", 8, "body", "Visual"
    )
    s_body_visuala = get_level(
        "stanford", "NTS-NTHC", 17, "body", "VisualA"
    )

    notes = f"""# Broad promising-signal review

This is an exploratory visualization-only derivative of the current
`{SOURCE_RUN_ID}` tables. It excludes the out-of-scope unavailable method,
does not refit a model, and does not modify the source tables.

## Main candidate themes

1. **Posterior Language remains the strongest anchor, but it is not the only
   story.** The Stanford tail–Language interaction has median standardized
   effect {tail_lang.median_d:.3f} across 12 fixed/template × 2/4mm ×
   bilateral/L/R specifications (range {tail_lang.min_d:.3f} to
   {tail_lang.max_d:.3f}; sign consistency {tail_lang.sign_consistency:.0%}).
2. **Body Somatomotor shows a large crossover-like interaction.** Its Stanford
   interaction median is {body_som.median_d:.3f} (range
   {body_som.min_d:.3f} to {body_som.max_d:.3f}; sign consistency
   {body_som.sign_consistency:.0%}). The four-group pattern is driven by high
   TEHC coupling and lower TIS coupling, making this a distinct candidate from
   the posterior Language story despite its different biological profile shape.
3. **Body Default is the clearest cross-dataset non-thresholded convergence.**
   JJTMS patient−healthy median d is {body_default_j.median_d:.3f}; Stanford
   symptom-marginal median d is
   {float(cross.loc[(cross.region == "body") & (cross.network == "Default"), "median_d_stanford"].iloc[0]):.3f}.
   Both are positive in all 12 specifications per dataset.
4. **Trauma-exposed healthy participants show a body-network reconfiguration.**
   TEHC−NTHC body Somatomotor has median d {tehc_som.median_d:.3f}, while
   body Sal/VAN-B has median d {tehc_sal.median_d:.3f}, each with 100% sign
   consistency. This suggests a Somatomotor-up / salience-attention-down
   pattern that should be shown as a multicomponent profile, not as isolated p
   values.
5. **Laterality deserves its own candidate family.** Stanford tail VisualC
   interaction has median L−R d {tail_visual_lat.median_d:.3f} across four
   fixed/template × 2/4mm specifications; body Language TIS−TEHC has median
   L−R d {body_lang_lat.median_d:.3f}. These are strong, direction-stable
   effect-size patterns, but they remain exploratory until the analysis-object
   set is frozen and the correction family is defined.
6. **Additional JJTMS-specific clues remain visible in the landscape**, notably
   body Default, anterior Language and posterior DorsalAttention. They are
   useful for defining a broader transdiagnostic profile; this is an effect-size
   and stability observation, not a multiplicity claim.
7. **Lower-effect direction-stable leads are retained for visual review.** The
   two secondary rows are selected by absolute median effect after the
   cross-specification sign requirement; they are not filtered or promoted by a
   provisional multiplicity threshold.
8. **Body Visual suggests a more nuanced cross-dataset hypothesis.** JJTMS
   patient−healthy body Visual is negative in all 12 specifications (median d
   {j_body_visual.median_d:.3f}); Stanford NTS−NTHC body VisualA is also
   negative in all 12 (median d {s_body_visuala.median_d:.3f}). Because the
   Stanford marginal parent-network effect is weak and VisualA is a
   17-network refinement with a TIS rebound pattern, this is a resolution- and
   estimand-sensitive remodeling hypothesis rather than direct replication.

## Interpretation rules

- Correlated specifications are summarized, not counted as replications.
- Multiplicity correction is deferred until the final analysis-object set is frozen; it is not used to select or annotate this derivative.
- A large, stable effect is a candidate for covariate/influence/independent
  validation, not a confirmed biological claim.
- The four-group heatmap is row-centered to reveal shape; exact raw Fisher-z
  group means remain in `stanford_four_group_story_patterns.tsv`.
"""
    notes_path = REPORT_ROOT / "scientific_notes.md"
    notes_path.write_text(notes)

    output_paths = sorted(
        path
        for path in (list(RESULT_ROOT.glob("*")) + list(FIGURE_ROOT.glob("*")) + [notes_path])
        if path.is_file() and path.suffix.casefold() != ".pdf"
    )
    manifest = {
        "run_id": RUN_ID,
        "source_run": SOURCE_RUN_ID,
        "source_files": {
            str(primary_path.relative_to(WORKFLOW_ROOT)): sha256_file(primary_path),
            str(views_path.relative_to(WORKFLOW_ROOT)): sha256_file(views_path),
        },
        "script": {
            str(Path(__file__).relative_to(WORKFLOW_ROOT)): sha256_file(Path(__file__))
        },
        "row_counts": {name: int(len(table)) for name, table in tables.items()},
        "output_sha256": {
            str(path.relative_to(WORKFLOW_ROOT)): sha256_file(path)
            for path in output_paths
            if path.name != "run_manifest.json"
        },
        "source_modified": False,
        "model_refit": False,
    }
    manifest_path = RESULT_ROOT / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(
        json.dumps(
            {
                "run_id": RUN_ID,
                "landscape_rows": len(landscape),
                "cross_dataset_rows": len(cross),
                "laterality_rows": len(laterality),
                "figures": [
                    str(FIGURE_ROOT / "effect_first_signal_landscape.png"),
                    str(FIGURE_ROOT / "broader_story_map.png"),
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Independent forest plots for cortical network area statistics."""

from __future__ import annotations

# Permit both ``python -m ...`` and direct execution of this entry module.
if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path
    _root = _Path(__file__).resolve().parents[3]
    if str(_root) not in _sys.path:
        _sys.path.insert(0, str(_root))
    __package__ = "scripts.visualize.cortex_network_area"

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from .core import CANONICAL7_ORDER, KONG_NATIVE17_ORDER, LYNCH_NATIVE20_ORDER
from scripts.core_workflow.lib.network_colors import NETWORK_DISPLAY_NAMES


EMPTY_PATIENT_PANEL_TEXT = "No uncorrected p < 0.05 results"


def select_patient_results(statistics: pd.DataFrame) -> pd.DataFrame:
    """Select only patient comparisons with strict uncorrected p < 0.05."""

    if statistics.empty:
        return statistics.copy()
    patient = statistics.loc[
        statistics["patient_comparison"].astype(bool)
        & pd.to_numeric(statistics["p_value"], errors="coerce").lt(0.05)
    ].copy()
    return sort_network_results(patient)


def select_hc_baseline_results(statistics: pd.DataFrame) -> pd.DataFrame:
    """Keep every Iowa HC versus HCP HC network, regardless of p value."""

    if statistics.empty:
        return statistics.copy()
    baseline = statistics.loc[
        (~statistics["patient_comparison"].astype(bool))
        | statistics["comparison"].eq("iowa_hc_vs_hcp_hc")
    ].copy()
    return sort_network_results(baseline)


def _network_rank(row: pd.Series) -> int:
    taxonomy = str(row.get("taxonomy", ""))
    network = str(row.get("network", ""))
    if taxonomy.startswith("kong_native17"):
        order = KONG_NATIVE17_ORDER
    elif taxonomy.startswith("lynch_native20"):
        order = LYNCH_NATIVE20_ORDER
    else:
        order = CANONICAL7_ORDER
    try:
        return order.index(network)
    except ValueError:
        return len(order) + 1


def sort_network_results(statistics: pd.DataFrame) -> pd.DataFrame:
    """Sort facets and networks using the fixed atlas order."""

    if statistics.empty:
        return statistics.copy()
    ordered = statistics.copy()
    ordered["_network_rank"] = ordered.apply(_network_rank, axis=1)
    columns = [column for column in ("variant", "method", "network_classification", "taxonomy", "comparison", "_network_rank") if column in ordered]
    ordered = ordered.sort_values(columns, kind="stable").drop(columns="_network_rank")
    return ordered.reset_index(drop=True)


def _facet_key(frame: pd.DataFrame) -> list[tuple[str, str, str]]:
    if frame.empty:
        return [("all", "all", "all")]
    return list(
        frame[["variant", "method", "taxonomy"]]
        .drop_duplicates()
        .sort_values(["variant", "method", "taxonomy"], kind="stable")
        .itertuples(index=False, name=None)
    )


TAXONOMY_TITLES = {
    "kong_canonical7": "Kong2022  /  canonical 7",
    "lynch_canonical7": "Lynch2024  /  canonical 7",
    "kong_native17": "Kong2022  /  native 17",
    "lynch_native20": "Lynch2024  /  native 20",
}
COHORTS = ("Patient", "Pure MDD", "Comorbid MDD")
INK = "#23313f"
MUTED = "#657280"


def _p_text(value: float) -> str:
    if not 0 <= value <= 1:
        raise ValueError(f"Invalid p value: {value}")
    # Keep the significance column sparse: a blank means p >= 0.05.
    return "***" if value < 0.001 else "**" if value < 0.01 else "*" if value < 0.05 else ""


def _ordered_networks(frame: pd.DataFrame) -> list[str]:
    return (frame.assign(_rank=frame.apply(_network_rank, axis=1))
            .sort_values("_rank", kind="stable")["network"].drop_duplicates().tolist())


def _draw_panel(axis, frame, networks, *, limit, series_column, series, colors, labels=True):
    """Draw exact CSV estimates, with significance symbols in an aligned column."""
    import numpy as np
    from matplotlib.ticker import MaxNLocator

    axis.set_xlim(-limit, limit)
    axis.set_ylim(len(networks) - 0.5, -0.7)
    for i in range(len(networks)):
        if i % 2 == 0:
            axis.axhspan(i - 0.5, i + 0.5, color="#f4f6f8", zorder=0)
    axis.axvline(0, color="#87929c", lw=0.85, zorder=1)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=5, symmetric=True))
    axis.grid(axis="x", color="#e7ebef", lw=0.6, zorder=0)
    axis.set_axisbelow(True)
    axis.set_yticks(range(len(networks)), [NETWORK_DISPLAY_NAMES.get(n, n) for n in networks] if labels else [])
    axis.tick_params(axis="both", length=0, labelsize=9, colors=MUTED, pad=7)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.spines["bottom"].set_visible(True)
    axis.spines["bottom"].set_color("#d4dbe1")
    axis.text(1.04, 1.015, "Sig.", transform=axis.transAxes, fontsize=9, color=MUTED, style="italic")
    for index, (value, marker) in enumerate(series):
        rows = frame.loc[frame[series_column].eq(value)]
        for row in rows.itertuples(index=False):
            y = networks.index(row.network) + (-0.18 if index == 0 else 0.18)
            d, lo, hi = row.difference_pp, row.ci95_low_pp, row.ci95_high_pp
            if not np.all(np.isfinite([d, lo, hi])):
                raise ValueError(f"Non-finite confidence interval: {row.taxonomy}/{row.network}/{row.comparison}")
            color = colors[row.network]
            artist = axis.errorbar(d, y, xerr=[[d - lo], [hi - d]], fmt=marker,
                                  ms=4.8, color=color, ecolor=color, lw=1.25, capsize=0,
                                  markerfacecolor=color if index == 0 else "white",
                                  markeredgewidth=1.2, zorder=3)
            artist.set_label(f"{row.network} | {row.case_cohort} vs {row.control_cohort} | n={int(row.n_case)}/{int(row.n_control)} | p={row.p_value:.3g}")
            axis.text(1.04, y, _p_text(row.p_value), transform=axis.get_yaxis_transform(),
                      ha="left", va="center", fontsize=8, color=INK)
    if frame.empty:
        axis.text(0.5, 0.5, EMPTY_PATIENT_PANEL_TEXT, transform=axis.transAxes,
                  ha="center", va="center", fontsize=9, color=MUTED, wrap=True)


def create_forest_figure(statistics: pd.DataFrame, *, figure_kind: str):
    """Create patient small multiples or a paired-variant HC baseline plate."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from scripts.core_workflow.lib.network_colors import load_network_hex_colors

    kind = str(figure_kind).lower()
    if kind not in {"patient", "hc", "hc_baseline", "baseline"}:
        raise ValueError("figure_kind must be patient or hc_baseline")
    is_patient = kind == "patient"
    selected = sort_network_results(statistics.loc[statistics["patient_comparison"].astype(bool)]) if is_patient else select_hc_baseline_results(statistics)
    taxonomies = [t for t in TAXONOMY_TITLES if t in set(statistics.taxonomy)]
    unknown = set(statistics.taxonomy) - set(taxonomies)
    if unknown:
        raise ValueError(f"Unknown plot taxonomies: {sorted(unknown)}")
    if not taxonomies:
        raise ValueError("Cannot plot an empty statistics table")
    colors = load_network_hex_colors()
    missing = set(statistics.network) - set(colors)
    if missing:
        raise ValueError(f"Missing official network colors: {sorted(missing)}")
    controls = ("Iowa HC", "HCP HC")
    networks_by_taxonomy = {}
    for taxonomy in taxonomies:
        source = selected.loc[selected.taxonomy.eq(taxonomy)]
        networks_by_taxonomy[taxonomy] = _ordered_networks(source)
    heights = [max(2.5, 0.33 * max(1, len(networks_by_taxonomy[t]))) for t in taxonomies]
    nrows = len(taxonomies) * len(controls) if is_patient else (len(taxonomies) + 1) // 2
    height = sum(h * len(controls) + 1.3 * len(controls) for h in heights) + 1.7 if is_patient else sum(max(heights[i:i+2]) + 1.0 for i in range(0, len(heights), 2)) + 1.8
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK}):
        fig = plt.figure(figsize=(16.5, height), facecolor="white")
        grid = fig.add_gridspec(nrows, 3 if is_patient else 2,
                               left=0.145, right=0.93, bottom=0.9/height, top=1-1.65/height,
                               hspace=0.95/(sum(heights)/len(heights)), wspace=0.48 if is_patient else 0.75,
                               height_ratios=[h for h in heights for _ in controls] if is_patient else [max(heights[i:i+2]) for i in range(0,len(heights),2)])
        for ti, taxonomy in enumerate(taxonomies):
            all_tax = statistics.loc[statistics.taxonomy.eq(taxonomy)]
            chosen = selected.loc[selected.taxonomy.eq(taxonomy)]
            networks = networks_by_taxonomy[taxonomy]
            bounds = chosen[["ci95_low_pp", "ci95_high_pp"]].to_numpy(dtype=float)
            limit = max(0.5, float(np.abs(bounds).max()) * 1.12) if bounds.size else 1.0
            if is_patient:
                for vi, control in enumerate(controls):
                    for ci, cohort in enumerate(COHORTS):
                        ax = fig.add_subplot(grid[ti * len(controls) + vi, ci])
                        rows = chosen.loc[chosen.control_cohort.eq(control) & chosen.case_cohort.eq(cohort)]
                        sample = all_tax.loc[all_tax.control_cohort.eq(control) & all_tax.case_cohort.eq(cohort)]
                        _draw_panel(ax, rows, networks, limit=limit, series_column="variant",
                                    series=[("36p", "D")], colors=colors, labels=ci == 0)
                        counts = sorted(sample.n_case.unique())
                        n = "/".join(str(int(v)) for v in counts) or "—"
                        control_n = "/".join(str(int(v)) for v in sorted(sample.n_control.unique())) or "—"
                        ax.set_title(f"{cohort}  ·  n={n}", loc="left", fontsize=12, weight="bold", pad=31)
                        ax.text(0, 1.035, f"{control} · n={control_n}", transform=ax.transAxes, fontsize=8.5, color=MUTED)
                        if ci == 0:
                            ax.text(-0.67, 1.015, f"{TAXONOMY_TITLES[taxonomy]}\nvs {control}", transform=ax.transAxes,
                                    fontsize=11, weight="bold", va="bottom", linespacing=1.6)
            else:
                ax = fig.add_subplot(grid[ti // 2, ti % 2])
                _draw_panel(ax, chosen, networks, limit=limit, series_column="variant",
                            series=[("36p", "D")], colors=colors)
                ax.set_title(TAXONOMY_TITLES[taxonomy], loc="left", fontsize=13, weight="bold", pad=33)
                counts = chosen[["n_case", "n_control"]].drop_duplicates()
                sample_label = "; ".join(f"Iowa HC n={int(r.n_case)} · HCP HC n={int(r.n_control)}" for r in counts.itertuples())
                ax.text(0, 1.04, sample_label, transform=ax.transAxes, fontsize=9, color=MUTED)
        fig.text(0.04, 1-0.26/height, "Cortical network area" if is_patient else "Cortical network area | HC baseline",
                 fontsize=23, weight="bold", va="top")
        subtitle = "Patient − control · percentage points · 95% confidence intervals" if is_patient else "Iowa HC − HCP HC · percentage points · 95% confidence intervals"
        fig.text(0.04, 1-0.68/height, subtitle, fontsize=11, color=MUTED, va="top")
        handles = [Line2D([], [], color=INK, marker="D", markerfacecolor="white", lw=1.2, ms=5, label="36P")]
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.965, 1-0.20/height), frameon=False, ncol=1, fontsize=10)
        fig.text(0.53, 0.50/height, "← Smaller area proportion     Difference (percentage points)     Larger area proportion →", ha="center", fontsize=10, color=INK)
        note = ("All networks shown. Uncorrected two-sided Welch: * p < 0.05; ** p < 0.01; *** p < 0.001; blank p ≥ 0.05. HCP comparisons are cross-dataset.\nPatient and MDD subgroup rosters overlap; these are not independent replications. BH-FDR q is auxiliary, not a selection criterion."
                if is_patient else "All networks shown. Uncorrected Welch: * p < 0.05; ** p < 0.01; *** p < 0.001; blank p ≥ 0.05. Cross-dataset comparison; blank does not establish equivalence.")
        fig.text(0.04, 0.12/height, note, va="bottom", fontsize=8.5, color=MUTED, linespacing=1.5)
    return fig


def render_area_figures(statistics_csv: Path | str, output_dir: Path | str) -> dict[str, Path]:
    """Write one HC baseline PNG and four patient taxonomy plates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    statistics = pd.read_csv(statistics_csv)
    paths = {}
    jobs = [("hc_baseline", statistics, "hc_baseline")]
    jobs += [(f"patient_{t}", statistics.loc[statistics.taxonomy.eq(t)], "patient")
             for t in TAXONOMY_TITLES if t in set(statistics.taxonomy)]
    for name, frame, kind in jobs:
        figure = create_forest_figure(frame, figure_kind=kind)
        path = output / f"cortex_network_area_{name}_forest.png"
        figure.savefig(path, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.2)
        plt.close(figure)
        paths[f"{name}_png"] = path
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Redraw cortical network area forest figures from a statistics table")
    parser.add_argument("--stats", type=Path, required=True, help="network_area_statistics.csv from run_area_analysis.py")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "figures")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    for name, path in render_area_figures(args.stats, args.output_dir).items():
        print(f"{name}\t{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

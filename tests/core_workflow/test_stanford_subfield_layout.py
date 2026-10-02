#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "visualize"
    / "multidataset_subfield_composition_2mm_2col"
    / "analyze_and_render_stanford_subfield_differences.py"
)


spec = importlib.util.spec_from_file_location("stanford_subfield_layout", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Failed to load module: {SCRIPT}")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_group_difference_titles_have_a_dedicated_header_gap() -> None:
    fig = plt.figure(figsize=(44, 28), dpi=100)
    try:
        grid = gridspec.GridSpec(
            nrows=2,
            ncols=3,
            left=0.105,
            right=0.950,
            top=module.GROUP_DIFFERENCES_GRID_TOP,
            bottom=0.065,
            wspace=0.34,
            hspace=0.26,
            height_ratios=[1.25, 0.95],
            width_ratios=[0.31, 0.34, 0.35],
        )
        ax_heat = fig.add_subplot(grid[0, :])
        vinfo = module.ATLAS_VERSIONS["v3_individual_template_matching"]

        fig.suptitle(
            "HIPPOCAMPAL SUBFIELD COMPOSITION DIFFERENCES VS NTHC BASELINE\n"
            f"[{vinfo['title_label']}]",
            fontsize=28.0,
            fontweight="heavy",
            y=module.GROUP_DIFFERENCES_SUPTITLE_Y,
        )
        ax_heat.set_title(
            "A. Subfield Composition Percentage Differences (Δ%) Relative to "
            "NTHC Baseline across All 15 Subregion × Subfield Units\n"
            f"[{vinfo['subtitle']}]",
            fontsize=22.0,
            fontweight="heavy",
            pad=16,
            loc="left",
        )

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        super_title_bbox = fig._suptitle.get_window_extent(renderer)
        panel_title_bbox = ax_heat._left_title.get_window_extent(renderer)
        gap_pixels = super_title_bbox.y0 - panel_title_bbox.y1

        assert gap_pixels >= 24.0
    finally:
        plt.close(fig)


def test_relative_subfield_panel_uses_percent_labels_and_readable_text() -> None:
    assert module.format_relative_difference_label(-6.44, "***") == "-6.4% (***)"
    assert "mm²" not in module.format_relative_difference_label(9.42, "*")
    assert module.PANEL_C_YTICK_FONTSIZE >= 19.0
    assert module.PANEL_D_BODY_FONTSIZE >= 14.0


def test_default_generation_targets_existing_v2_export() -> None:
    assert module.DEFAULT_GENERATION_MODE == "target"
    assert module.generation_plan("target") == [
        ("target", "v2_cohort_specific_atlas")
    ]


def test_group_difference_layout_uses_tighter_margins_and_gaps() -> None:
    assert module.GROUP_DIFFERENCES_GRID_LEFT <= 0.07
    assert module.GROUP_DIFFERENCES_GRID_RIGHT >= 0.975
    assert module.GROUP_DIFFERENCES_GRID_WSPACE == 0.22

def test_target_loader_reads_canonical_v2_exports() -> None:
    stats_results, subject_rows = module.load_exported_render_inputs(module.TARGET_ATLAS_MODE)
    assert stats_results and subject_rows

def test_target_figure_uses_reader_facing_title_and_balanced_lower_row() -> None:
    assert module.GROUP_DIFFERENCES_MAIN_TITLE == (
        "HIPPOCAMPAL SUBFIELD COMPOSITION DIFFERENCES VS NTHC"
    )
    assert "VERSION" not in module.GROUP_DIFFERENCES_MAIN_TITLE
    assert module.GROUP_DIFFERENCES_GRID_WSPACE >= 0.20
    assert module.GROUP_DIFFERENCES_GRID_WIDTH_RATIOS[2] <= 0.30
    assert module.GROUP_DIFFERENCES_GRID_WIDTH_RATIOS[1] >= 0.40
    assert module.PANEL_A_TITLE_FONTSIZE >= 26.0
    assert module.PANEL_B_TITLE_FONTSIZE >= 26.0
    assert module.PANEL_C_TITLE_FONTSIZE >= 26.0
    assert module.PANEL_B_LEGEND_FONTSIZE >= 16.0
    assert module.PANEL_D_TITLE_FONTSIZE >= 24.0
    assert module.PANEL_D_HEADING_FONTSIZE >= 20.0
    assert module.PANEL_D_BODY_FONTSIZE >= 17.0

def test_large_negative_c_bar_label_is_placed_inside_bar() -> None:
    x, ha, color = module.relative_bar_label_placement(-7.61)
    assert x > -7.61
    assert ha == "right"
    assert color == "#0f172a"

    x_small, ha_small, color_small = module.relative_bar_label_placement(-0.65)
    assert x_small < -0.65
    assert ha_small == "right"
    assert color_small == "#0f172a"

def test_c_selection_adds_largest_nonsignificant_bilateral_candidates() -> None:
    stats_results, _ = module.load_exported_render_inputs(module.TARGET_ATLAS_MODE)
    items = module.select_relative_bar_items(stats_results)
    assert items
    assert len(items) == 5
    assert sum(item[3] == "n.s." for item in items) == 3
    body_items = [item for item in items if item[0] == "MDD" and item[1] == "Body Sub"]
    assert len(body_items) == 1
    assert abs(body_items[0][2] - ((-6.3564 - 8.8565) / 2.0)) < 0.001
    ant_items = [item for item in items if item[0] == "MDD" and item[1] == "Ant Sub"]
    assert len(ant_items) == 1
    assert ant_items[0][3] == "n.s."
    assert not any(item[0] == "HCP" for item in items)


def test_stanford_subfield_grid_exports_exist() -> None:
    output_dir = module.OUTPUT_DIR
    stats_dir = getattr(module, "STATS_DIR", output_dir / "stats")
    for grid in module.GRID_VARIANTS:
        fig_path = output_dir / f"stanford_subfield_group_differences_figure_{grid}.png"
        stats_path = stats_dir / f"stanford_subfield_group_differences_stats_{grid}.tsv"
        long_path = stats_dir / f"stanford_subfield_subject_level_long_{grid}.tsv"
        assert fig_path.is_file(), f"Missing figure for {grid}"
        assert stats_path.is_file(), f"Missing stats TSV for {grid}"
        assert long_path.is_file(), f"Missing subject long TSV for {grid}"


#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pytest


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


@pytest.fixture
def synthetic_exports(tmp_path, monkeypatch):
    """Export small bilateral contrasts instead of reading historical results."""
    stats_dir = tmp_path / "stats"
    stats_dir.mkdir()
    monkeypatch.setattr(module, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(module, "STATS_DIR", stats_dir)
    cases = [
        ("NTS Pure MDD", "Middle", "Sub", (-6.3564, -8.8565), 0.001),
        ("TIS PTSD", "Middle", "CA1", (4.5, 4.0), 0.02),
        ("NTS Pure MDD", "Post", "Sub", (-2.0, -1.0), 0.3),
        ("NTS Pure MDD", "Ant", "Sub", (-5.0, -4.0), 0.4),
        ("NTS", "Ant", "CA1", (3.0, 2.5), 0.2),
        ("NTS", "Post", "CA1", (0.3, 0.2), 0.4),
        ("TEHC", "Post", "CA1", (0.2, 0.1), 0.5),
        ("HCP 7T", "Middle", "Sub", (0.01, 0.01), 0.8),
    ]
    stats = [
        dict(case_cohort=cohort, hemisphere=hemi, cluster_name=cluster,
             subfield_name=subfield, diff_pct=difference, fdr_q_parametric=q,
             significant=q < 0.05)
        for cohort, cluster, subfield, differences, q in cases
        for hemi, difference in zip(("L", "R"), differences, strict=True)
    ]
    subjects = [dict(subject_id="fixture-01", hemisphere=hemi, cohort_name="NTHC",
                     cluster_name="Middle", subfield_name="Sub", proportion_pct=35.25)
                for hemi in ("L", "R")]
    module.export_analysis_tsvs(subjects, stats, "fixture", atlas_mode=module.TARGET_ATLAS_MODE)
    return stats, subjects


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

def test_target_loader_reads_canonical_v2_exports(synthetic_exports) -> None:
    stats_results, subject_rows = module.load_exported_render_inputs(module.TARGET_ATLAS_MODE)
    expected_stats, expected_subjects = synthetic_exports
    assert stats_results == expected_stats
    assert subject_rows == expected_subjects


def test_target_loader_rejects_missing_grid_exports_even_with_legacy_files(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "STATS_DIR", tmp_path)
    monkeypatch.setattr(module, "OUTPUT_DIR", tmp_path)
    for name in ("stanford_subfield_group_differences_stats.tsv", "stanford_subfield_subject_level_long.tsv"):
        (tmp_path / name).write_text("legacy\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="canonical grid export"):
        module.load_exported_render_inputs(module.TARGET_ATLAS_MODE)

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

def test_c_selection_adds_largest_nonsignificant_bilateral_candidates(synthetic_exports) -> None:
    stats_results, _ = module.load_exported_render_inputs(module.TARGET_ATLAS_MODE)
    items = module.select_relative_bar_items(stats_results)
    assert items
    assert len(items) == 5
    assert sum(item[3] == "n.s." for item in items) == 3
    body_items = [item for item in items if item[0] == "MDD" and item[1] == "Middle Sub"]
    assert len(body_items) == 1
    assert abs(body_items[0][2] - ((-6.3564 - 8.8565) / 2.0)) < 0.001
    ant_items = [item for item in items if item[0] == "MDD" and item[1] == "Ant Sub"]
    assert len(ant_items) == 1
    assert ant_items[0][3] == "n.s."
    assert not any(item[0] == "HCP" for item in items)


@pytest.mark.parametrize("grid", module.GRID_VARIANTS)
def test_stanford_subfield_grid_exports_roundtrip(synthetic_exports, grid) -> None:
    stats, subjects = synthetic_exports
    spec = module.GridSpec.from_tag(grid)
    module.export_analysis_tsvs(subjects, stats, "fixture", spec=spec, atlas_mode=module.TARGET_ATLAS_MODE)
    stats_path = module.STATS_DIR / f"stanford_subfield_group_differences_stats_{grid}.tsv"
    long_path = module.STATS_DIR / f"stanford_subfield_subject_level_long_{grid}.tsv"
    assert stats_path.is_file()
    assert long_path.is_file()
    assert module.load_exported_render_inputs(module.TARGET_ATLAS_MODE, spec=spec) == (stats, subjects)


"""Tests for the current comprehensive 2 mm area summary figure."""

from pathlib import Path
import pytest
import sys
import matplotlib.pyplot as plt

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import plot_area_2mm_comprehensive_summary as summary  # noqa: E402


def _rosters() -> dict[str, tuple[str, ...]]:
    result = {
        group: (f"sub-{1001 + index:04d}",)
        for index, group in enumerate(summary.GROUP_ORDER[:-1])
    }
    result["TIS PTSD"] = result["TIS"]
    return result


def _features() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rosters = _rosters()
    for group, subjects in rosters.items():
        for subject in subjects:
            for hemisphere in ("L", "R"):
                for system_index, system in enumerate(summary.SYSTEMS):
                    rows.append(
                        {
                            "subject": subject,
                            "method": "hcp151_template_matching",
                            "smoothing_mm": 2,
                            "hemisphere": hemisphere,
                            "system": system,
                            "status": "success",
                            "surface_area_mm2": 300.0 + system_index * 50.0,
                        }
                    )
    return pd.DataFrame(rows).drop_duplicates(
        subset=["subject", "hemisphere", "system"]
    )


def _effects() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "contrast": "TIS PTSD−NTHC",
                "method": "hcp151_template_matching",
                "smoothing_mm": 2,
                "hemisphere": hemisphere,
                "system": system,
                "mean_difference": -20.0,
                "ci_low": -40.0,
                "ci_high": 0.0,
                "cohens_d": -0.4,
                "p_value": 0.04,
            }
            for hemisphere in ("L", "R")
            for system in summary.SYSTEMS
        ]
    )


def test_area_summary_registry_includes_tis_ptsd_group_and_contrast() -> None:
    assert summary.GROUP_ORDER == (
        "NTHC",
        "TEHC",
        "NTS",
        "NTS Any MDD",
        "NTS Pure MDD",
        "TIS",
        "TIS PTSD",
    )
    assert summary.GROUP_SLUG["TIS PTSD"] == "tis_ptsd"
    assert "TIS PTSD−NTHC" in summary.CONTRAST_ORDER


def test_roster_attachment_keeps_tis_ptsd_as_explicit_nested_group() -> None:
    labeled = summary.attach_cohort_labels(_features(), _rosters())
    assert set(labeled["cohort"]) == set(summary.GROUP_ORDER)
    assert len(labeled.loc[labeled["cohort"].eq("TIS PTSD")]) == 6
    assert len(labeled.loc[labeled["cohort"].eq("TIS")]) == 6


def test_area_summary_writes_png_only_and_reports_tis_ptsd(tmp_path: Path) -> None:
    output = tmp_path / "area_2mm_comprehensive_summary_proportion.png"
    manifest = summary.render_area_summary(
        _effects(),
        _features(),
        _rosters(),
        output,
    )
    assert output.is_file()
    assert not output.with_suffix(".pdf").exists()
    assert manifest["png_path"] == str(output.resolve())
    assert manifest["cohort_order"][-1] == "TIS PTSD"
    assert "TIS PTSD−NTHC" in manifest["contrast_order"]

def test_forest_layout_hides_repeated_right_y_labels_and_compacts_rows() -> None:
    figure, axes = plt.subplots(1, 2, figsize=(8, 4))
    try:
        summary.plot_effect_forest(axes[0], _effects(), "L")
        summary.plot_effect_forest(axes[1], _effects(), "R", show_y_labels=False, show_x_labels=False)

        assert all(not label.get_visible() for label in axes[1].get_yticklabels())
        assert all(not label.get_visible() for label in axes[1].get_xticklabels())
    finally:
        plt.close(figure)



def test_forest_uses_group_separators_instead_of_x_grid() -> None:
    figure, axis = plt.subplots(figsize=(8, 4))
    try:
        summary.plot_effect_forest(axis, _effects(), "L")

        separators = [line for line in axis.lines if line.get_color() == "#d9dee2"]
        assert len(separators) == len(summary.CONTRAST_ORDER) - 1
        assert all(len(set(line.get_ydata())) == 1 for line in separators)
        assert not any(line.get_visible() for line in axis.get_xgridlines())
    finally:
        plt.close(figure)



def _hcp_features() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subject_index, subject in enumerate(("sub-100610", "sub-100611")):
        for hemisphere in ("L", "R"):
            for system_index, system in enumerate(summary.SYSTEMS):
                rows.append(
                    {
                        "subject": subject,
                        "cohort": "HCP 7T",
                        "method": "hcp151_crossfit_template_matching",
                        "smoothing_mm": 2,
                        "hemisphere": hemisphere,
                        "system": system,
                        "status": "success",
                        "surface_area_mm2": 600.0 + subject_index * 10.0 + system_index * 5.0,
                    }
                )
    return pd.DataFrame(rows)


def _hcp_effects() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "contrast": "HCP 7T−NTHC",
                "case_group": "HCP 7T",
                "reference_group": "NTHC",
                "method": "hcp151_crossfit_template_matching",
                "smoothing_mm": 2,
                "hemisphere": hemisphere,
                "system": system,
                "mean_difference": 25.0,
                "ci_low": 5.0,
                "ci_high": 45.0,
                "cohens_d": 0.5,
                "p_value": 0.04,
            }
            for hemisphere in ("L", "R")
            for system in summary.SYSTEMS
        ]
    )


def test_area_summary_renders_hcp_crossfit_reference_with_explicit_layout_contract(tmp_path: Path) -> None:
    output = tmp_path / "area_2mm_comprehensive_summary_proportion.png"
    manifest = summary.render_area_summary(
        _effects(),
        _features(),
        _rosters(),
        output,
        reference_features=_hcp_features(),
        reference_effects=_hcp_effects(),
        reference_metadata={
            "method": "hcp151_crossfit_template_matching",
            "n_subjects": 151,
            "n_folds": 5,
            "family_aware": True,
        },
    )

    assert manifest["display_group_order"][-1] == "HCP 7T"
    assert manifest["display_contrast_order"][-1] == "HCP 7T−NTHC"
    assert manifest["reference_method"] == "hcp151_crossfit_template_matching"
    assert manifest["reference_subject_count"] == 151
    assert manifest["feature_cohort_subject_counts"]["HCP 7T"] == 2
    assert manifest["source_reference_effect_rows"] == 6
    font_sizes = manifest["layout"]["font_sizes"]
    assert font_sizes["suptitle"] >= 18.0
    assert font_sizes["forest_title"] >= 15.0
    assert font_sizes["forest_axis_label"] >= 13.0
    assert font_sizes["forest_y_tick"] >= 10.0
    assert font_sizes["forest_x_tick"] >= 11.0
    assert font_sizes["forest_annotation"] >= 10.5
    assert font_sizes["forest_value_label"] >= 9.5
    assert font_sizes["raw_title"] >= 14.0
    assert font_sizes["raw_axis_label"] >= 12.0
    assert font_sizes["raw_x_tick"] >= 10.0
    assert font_sizes["raw_y_tick"] >= 10.5
    assert font_sizes["raw_significance"] >= 9.0
    assert font_sizes["legend"] >= 10.0
    assert font_sizes["legend_title"] >= 12.0
    assert "footer" not in font_sizes
    assert manifest["layout"]["footer_visible"] is False
    assert manifest["layout"]["grid_margins"]["bottom"] == 0.1
    assert manifest["layout"]["grid_hspace"] == 0.42
    assert manifest["layout"]["grid_wspace"] >= 0.35
    assert manifest["layout"]["raw_xtick_rotation"] == 0
    assert manifest["layout"]["grid_margins"]["left"] == 0.11
    assert manifest["layout"]["forest_row_step"] == 0.32
    assert manifest["layout"]["forest_group_gap"] == 0.48
    assert manifest["layout"]["legend_location"] == "center left"


def test_area_variant_proportion_uses_within_hemisphere_system_total() -> None:
    variant = summary.prepare_area_variant(_features(), "proportion")

    rows = variant.loc[
        variant["subject"].eq("sub-1001") & variant["hemisphere"].eq("L")
    ].set_index("system")

    assert rows["surface_area_proportion"].to_dict() == {
        "anterior": 300.0 / 1050.0,
        "body": 350.0 / 1050.0,
        "posterior": 400.0 / 1050.0,
    }
    assert rows["plot_value"].to_dict() == pytest.approx(
        {"anterior": 300.0 / 10.5, "body": 350.0 / 10.5, "posterior": 400.0 / 10.5}
    )


def test_area_summary_variant_modes_are_registered() -> None:
    assert summary.ANALYSIS_MODES == ("proportion", "covariate")
    assert summary.ANALYSIS_MODE_LABELS["proportion"] == "System Area Proportion"
    assert summary.ANALYSIS_MODE_LABELS["covariate"] == "Total-Area-Adjusted Surface Area"

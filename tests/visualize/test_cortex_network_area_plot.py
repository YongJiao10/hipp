from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from scripts.visualize.cortex_network_area.plot import (
    EMPTY_PATIENT_PANEL_TEXT,
    create_forest_figure,
    render_area_figures,
    select_hc_baseline_results,
    select_patient_results,
    sort_network_results,
)


def _statistics() -> pd.DataFrame:
    rows = [
        {
            "variant": "36p",
            "method": "kong2022",
            "network_classification": "canonical7",
            "taxonomy": "kong_canonical7",
            "network": "Visual",
            "comparison": "patient_vs_iowa_hc",
            "case_cohort": "Patient",
            "control_cohort": "Iowa HC",
            "patient_comparison": True,
            "n_case": 26,
            "n_control": 67,
            "difference_pp": 1.2,
            "ci95_low_pp": 0.2,
            "ci95_high_pp": 2.2,
            "p_value": 0.049,
            "q_value_bh": 0.91,
        },
        {
            "variant": "36p",
            "method": "kong2022",
            "network_classification": "canonical7",
            "taxonomy": "kong_canonical7",
            "network": "Default",
            "comparison": "pure_mdd_vs_hcp_hc",
            "case_cohort": "Pure MDD",
            "control_cohort": "HCP HC",
            "patient_comparison": True,
            "n_case": 7,
            "n_control": 151,
            "difference_pp": -0.4,
            "ci95_low_pp": -1.0,
            "ci95_high_pp": 0.2,
            "p_value": 0.05,
            "q_value_bh": 0.01,
        },
        {
            "variant": "36p",
            "method": "kong2022",
            "network_classification": "canonical7",
            "taxonomy": "kong_canonical7",
            "network": "Default",
            "comparison": "iowa_hc_vs_hcp_hc",
            "case_cohort": "Iowa HC",
            "control_cohort": "HCP HC",
            "patient_comparison": False,
            "n_case": 67,
            "n_control": 151,
            "difference_pp": 0.1,
            "ci95_low_pp": -0.3,
            "ci95_high_pp": 0.5,
            "p_value": 0.80,
            "q_value_bh": 0.90,
        },
        {
            "variant": "36p",
            "method": "kong2022",
            "network_classification": "canonical7",
            "taxonomy": "kong_canonical7",
            "network": "Visual",
            "comparison": "iowa_hc_vs_hcp_hc",
            "case_cohort": "Iowa HC",
            "control_cohort": "HCP HC",
            "patient_comparison": False,
            "n_case": 67,
            "n_control": 151,
            "difference_pp": -0.7,
            "ci95_low_pp": -1.1,
            "ci95_high_pp": -0.3,
            "p_value": 0.003,
            "q_value_bh": 0.02,
        },
        {
            "variant": "36p",
            "method": "lynch2024",
            "network_classification": "canonical7",
            "taxonomy": "lynch_canonical7",
            "network": "Control",
            "comparison": "comorbid_mdd_vs_iowa_hc",
            "case_cohort": "Comorbid MDD",
            "control_cohort": "Iowa HC",
            "patient_comparison": True,
            "n_case": 19,
            "n_control": 67,
            "difference_pp": 0.2,
            "ci95_low_pp": -0.1,
            "ci95_high_pp": 0.5,
            "p_value": 0.20,
            "q_value_bh": 0.20,
        },
        {
            "variant": "36p",
            "method": "lynch2024",
            "network_classification": "canonical7",
            "taxonomy": "lynch_canonical7",
            "network": "Control",
            "comparison": "iowa_hc_vs_hcp_hc",
            "case_cohort": "Iowa HC",
            "control_cohort": "HCP HC",
            "patient_comparison": False,
            "n_case": 67,
            "n_control": 151,
            "difference_pp": 0.0,
            "ci95_low_pp": -0.2,
            "ci95_high_pp": 0.2,
            "p_value": 1.0,
            "q_value_bh": 1.0,
        },
    ]
    return pd.DataFrame(rows)


def test_patient_filter_uses_strict_uncorrected_p_value() -> None:
    selected = select_patient_results(_statistics())

    assert selected["comparison"].tolist() == ["patient_vs_iowa_hc"]
    assert selected["network"].tolist() == ["Visual"]


def test_hc_baseline_keeps_all_networks_regardless_of_p_value() -> None:
    selected = select_hc_baseline_results(_statistics())

    assert len(selected) == 3
    assert set(selected["p_value"]) == {0.003, 0.8, 1.0}


def test_network_rows_follow_fixed_taxonomy_order() -> None:
    rows = _statistics().loc[
        _statistics()["comparison"].eq("iowa_hc_vs_hcp_hc")
        & _statistics()["method"].eq("kong2022")
    ]

    ordered = sort_network_results(rows)

    assert ordered["network"].tolist() == ["Default", "Visual"]


def test_patient_figure_marks_empty_facets_and_states_interpretation_notes() -> None:
    figure = create_forest_figure(_statistics(), figure_kind="patient")
    all_text = "\n".join(
        text.get_text()
        for axis in figure.axes
        for text in [*axis.texts, *axis.get_yticklabels()]
    )
    figure_text = "\n".join(text.get_text() for text in figure.texts)

    assert "NS" not in all_text
    assert "Visual" in all_text
    assert "*" in all_text
    labels = [c.get_label() for ax in figure.axes for c in ax.containers]
    assert "Visual | Patient vs Iowa HC | n=26/67 | p=0.049" in labels
    assert "Uncorrected" in figure_text
    assert "overlap" in figure_text.lower()


def test_independent_redraw_writes_png_for_both_figures(
    tmp_path: Path,
) -> None:
    statistics_path = tmp_path / "network_area_statistics.csv"
    _statistics().to_csv(statistics_path, index=False)

    outputs = render_area_figures(statistics_path, tmp_path / "figures")

    assert set(outputs) == {
        "hc_baseline_png",
        "patient_kong_canonical7_png",
        "patient_lynch_canonical7_png",
    }
    for path in outputs.values():
        assert path.exists()
        assert path.stat().st_size > 1_000


def test_significance_symbols_use_strict_thresholds():
    from scripts.visualize.cortex_network_area.plot import _p_text
    assert [_p_text(p) for p in (0, 0.0009, 0.001, 0.009, 0.01, 0.049, 0.05, 1)] == ['***', '***', '**', '**', '*', '*', '', '']


def test_patient_figure_retains_nonsignificant_estimates():
    import matplotlib.pyplot as plt
    frame = _statistics()
    figure = create_forest_figure(frame, figure_kind='patient')
    labels = [c.get_label() for ax in figure.axes for c in ax.containers]
    assert len(labels) == int(frame.patient_comparison.sum())
    assert any('p=0.05' in label for label in labels)
    assert any('p=0.2' in label for label in labels)
    plt.close(figure)

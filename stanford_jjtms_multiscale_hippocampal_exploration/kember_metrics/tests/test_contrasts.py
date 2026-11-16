"""Contracts for the Kember template-matching direct-to-NTHC workflow."""

from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from scipy import stats as scipy_stats


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


def _module():
    return importlib.import_module("contrasts")


def test_default_cohort_rosters_have_exact_names_counts_and_nested_overlap() -> None:
    module = _module()
    rosters = module.load_cohort_rosters()

    assert list(rosters) == [
        "NTHC",
        "TEHC",
        "NTS",
        "NTS Any MDD",
        "NTS Pure MDD",
        "TIS",
        "TIS PTSD",
    ]
    assert {name: len(subjects) for name, subjects in rosters.items()} == {
        "NTHC": 51,
        "TEHC": 40,
        "NTS": 56,
        "NTS Any MDD": 43,
        "NTS Pure MDD": 11,
        "TIS": 47,
        "TIS PTSD": 35,
    }
    assert set(rosters["NTS Any MDD"]) < set(rosters["NTS"])
    assert set(rosters["NTS Pure MDD"]) < set(rosters["NTS Any MDD"])
    assert set(rosters["TIS PTSD"]) < set(rosters["TIS"])


def test_endpoint_enumeration_is_exactly_432_combined_kember_units() -> None:
    module = _module()
    endpoints = module.enumerate_direct_endpoints()

    assert len(endpoints) == 432
    assert endpoints.groupby("metric_family").size().to_dict() == {
        "surface_area": 144,
        "functional_segregation": 144,
        "boundary_sharpness": 144,
    }
    assert set(endpoints["method"]) == {
        "hcp_template_matching",
        "hcp_fixed",
    }
    assert set(endpoints["contrast"]) == {
        "TEHC−NTHC",
        "NTS−NTHC",
        "NTS Any MDD−NTHC",
        "NTS Pure MDD−NTHC",
        "TIS−NTHC",
        "TIS PTSD−NTHC",
    }
    assert set(endpoints["smoothing_mm"]) == {2, 4}
    assert set(endpoints["hemisphere"]) == {"L", "R"}
    assert set(endpoints["metric"]) == {
        "surface_area_mm2",
        "functional_segregation_d",
        "boundary_sharpness_d",
    }
    assert endpoints[["method", "contrast", "metric_family", "metric", "smoothing_mm", "hemisphere", "system"]].duplicated().sum() == 0


def test_analysis_summary_reports_two_methods_and_432_units(tmp_path: Path) -> None:
    module = _module()
    effects = module.enumerate_direct_endpoints()
    effects["p_value"] = np.nan
    effects["cohens_d"] = np.nan
    coverage = pd.DataFrame({"missing_n": [0]})
    summary_path = tmp_path / "analysis_summary.md"

    module.write_analysis_summary(
        summary_path,
        run_id="review-contract",
        effects=effects,
        descriptives=pd.DataFrame(),
        coverage=coverage,
        overlap={},
        figure_sources=[],
    )

    summary = summary_path.read_text(encoding="utf-8")
    assert summary.startswith("# Stanford two-method Kember direct-to-NTHC")
    assert "6 contrasts × 2 methods × 2 smoothing × 2 hemispheres" in summary
    assert "3 metrics × 3 systems = **432**" in summary
    assert "= **216**" not in summary


def test_active_direct_contract_rejects_a_single_method() -> None:
    module = _module()
    with pytest.raises(ValueError, match="exactly"):
        module.enumerate_direct_endpoints(
            {"methods": ["hcp_fixed"], "smoothings_mm": [2], "hemispheres": ["L"]}
        )


def test_multiplicity_correction_requires_frozen_analysis_set() -> None:
    module = _module()
    effects = pd.DataFrame(
        {"metric_family": ["surface_area"], "p_value": [0.01]}
    )
    with pytest.raises(ValueError, match="deferred"):
        module.apply_bh_fdr(effects)
    corrected = module.apply_bh_fdr(effects, analysis_object_set_frozen=True)
    assert np.isfinite(corrected.loc[0, "q_bh"])
    assert corrected.loc[0, "correction_status"] == "applied_to_frozen_analysis_set"

def test_welch_effect_contains_exact_t_df_d_and_confidence_interval() -> None:
    module = _module()
    reference = np.array([1.0, 2.0, 4.0, 5.0])
    case = np.array([3.0, 6.0, 8.0])
    result = module.welch_two_sample(case, reference)

    expected = scipy_stats.ttest_ind(case, reference, equal_var=False)
    difference = np.mean(case) - np.mean(reference)
    se = math.sqrt(np.var(case, ddof=1) / len(case) + np.var(reference, ddof=1) / len(reference))
    df = (
        (np.var(case, ddof=1) / len(case) + np.var(reference, ddof=1) / len(reference)) ** 2
        / (
            (np.var(case, ddof=1) / len(case)) ** 2 / (len(case) - 1)
            + (np.var(reference, ddof=1) / len(reference)) ** 2 / (len(reference) - 1)
        )
    )
    pooled = math.sqrt(
        ((len(case) - 1) * np.var(case, ddof=1) + (len(reference) - 1) * np.var(reference, ddof=1))
        / (len(case) + len(reference) - 2)
    )
    critical = scipy_stats.t.ppf(0.975, df)

    assert result["case_n"] == 3
    assert result["reference_n"] == 4
    assert np.isclose(result["mean_difference"], difference)
    assert np.isclose(result["welch_t"], expected.statistic)
    assert np.isclose(result["welch_df"], df)
    assert np.isclose(result["p_value"], expected.pvalue)
    assert np.isclose(result["cohens_d"], difference / pooled)
    assert np.isclose(result["ci_low"], difference - critical * se)
    assert np.isclose(result["ci_high"], difference + critical * se)


def _synthetic_feature_table() -> tuple[pd.DataFrame, dict[str, tuple[str, ...]], dict]:
    rosters = {
        "NTHC": ("sub-0001", "sub-0002"),
        "TEHC": ("sub-0003", "sub-0004"),
        "NTS": tuple(),
        "NTS Any MDD": tuple(),
        "NTS Pure MDD": tuple(),
        "TIS": tuple(),
    }
    values = {"sub-0001": 10.0, "sub-0002": 12.0, "sub-0003": 20.0, "sub-0004": np.nan}
    rows = []
    for method in ("hcp_template_matching", "hcp_fixed"):
        for subject, area in values.items():
            group = "NTHC" if subject in rosters["NTHC"] else "TEHC"
            for system in ("anterior", "middle", "posterior"):
                rows.append(
                {
                    "subject": subject,
                    "four_group": group,
                    "method": method,
                    "smoothing_mm": 2,
                    "hemisphere": "L",
                    "system": system,
                    "status": "success",
                    "endpoint_missing_fields": "surface_area_mm2" if np.isnan(area) else "",
                    "surface_area_mm2": area,
                    "functional_segregation_d": {"sub-0001": 0.20, "sub-0002": 0.25, "sub-0003": 0.50, "sub-0004": 0.60}[subject],
                    "boundary_sharpness_d": {"sub-0001": 0.30, "sub-0002": 0.35, "sub-0003": 0.60, "sub-0004": 0.70}[subject],
                    }
                )
    config = {
        "reference_group": "NTHC",
        "cohort_order": list(rosters),
        "contrasts": [{"contrast": "TEHC−NTHC", "case_group": "TEHC", "reference_group": "NTHC"}],
        "methods": ["hcp_template_matching", "hcp_fixed"],
        "smoothings_mm": [2],
        "hemispheres": ["L"],
    }
    return pd.DataFrame(rows), rosters, config


def test_missing_values_only_reduce_the_corresponding_direct_endpoint() -> None:
    module = _module()
    table, rosters, config = _synthetic_feature_table()
    effects, descriptives, coverage = module.compute_direct_contrast_statistics(table, config=config, rosters=rosters)

    area = effects.query("method == \"hcp_template_matching\" and metric == \"surface_area_mm2\" and system == \"anterior\"").iloc[0]
    segregation = effects.query("method == \"hcp_template_matching\" and metric == \"functional_segregation_d\" and system == \"anterior\"").iloc[0]
    assert area["case_n"] == 1
    assert area["reference_n"] == 2
    assert np.isnan(area["p_value"])
    assert "case" in str(area["missing_reason"])
    assert segregation["case_n"] == 2
    assert segregation["reference_n"] == 2
    assert np.isfinite(segregation["p_value"])
    assert coverage["missing_n"].sum() > 0
    assert descriptives["mean"].notna().any()


def test_figure_provenance_uses_only_effect_row_ids_and_large_cells(tmp_path: Path) -> None:
    module = _module()
    table, rosters, config = _synthetic_feature_table()
    effects, _descriptives, _coverage = module.compute_direct_contrast_statistics(table, config=config, rosters=rosters)
    visualization = importlib.import_module("plot_contrasts")
    manifest = visualization.render_direct_contrast_figures(effects, tmp_path)

    effect_ids = set(effects["row_id"].astype(str))
    assert {item["figure_id"] for item in manifest["figure_sources"]} == {"kember_effects_2mm"}
    for item in manifest["figure_sources"]:
        assert Path(item["png_path"]).is_file()
        assert "pdf_path" not in item
        assert not Path(item["png_path"]).with_suffix(".pdf").exists()
        assert set(item["source_row_ids"]) <= effect_ids
        assert item["source_row_ids"]
        with Image.open(item["png_path"]) as image:
            assert image.width >= 7000
def test_default_direct_run_name_is_stable_and_unversioned() -> None:
    assert _module().DEFAULT_DIRECT_RUN_ID == "direct_contrasts"


def test_run_direct_contrasts_writes_one_combined_method_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    table, rosters, config = _synthetic_feature_table()
    monkeypatch.setattr(module, "load_cohort_rosters", lambda _config: rosters)

    result = module.run_direct_contrasts(table, config, tmp_path / "direct_contrasts")

    effects = pd.read_csv(result / "direct_contrast_effects.tsv", sep="\t")
    manifest = json.loads((result / "run_manifest.json").read_text(encoding="utf-8"))
    assert set(effects["method"]) == {
        "hcp_template_matching",
        "hcp_fixed",
    }
    assert manifest["methods"] == [
        "hcp_template_matching",
        "hcp_fixed",
    ]
    assert manifest["endpoint_counts_by_method"] == {
        "hcp_fixed": 9,
        "hcp_template_matching": 9,
    }
    assert effects["q_bh"].isna().all()
    assert effects["correction_status"].eq("deferred_until_analysis_set_freeze").all()

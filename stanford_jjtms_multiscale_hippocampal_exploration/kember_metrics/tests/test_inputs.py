"""Behavioral tests for the rest-fMRI artifact inventory."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from inputs import (  # noqa: E402
    build_factorial_roster,
    inspect_rest_artifacts,
    validate_rest_only_path,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "analysis_config.json"


def test_beta_and_native_mapping_paths_are_rejected(tmp_path: Path) -> None:
    """The scope guard must fail closed for explicitly prohibited roots."""

    for path in [
        tmp_path / "tms_hipp_surface_beta",
        tmp_path / "outputs_native_surface_comparison",
    ]:
        with pytest.raises(ValueError, match="rest-fMRI-only scope"):
            validate_rest_only_path(path)


def test_scope_guard_rejects_normalized_prohibited_variants(tmp_path: Path) -> None:
    """Separator/case variants of prohibited names fail closed."""

    variants = [
        "beta-map",
        "con image",
        "native stimulation mapping",
        "native_stimulation_mapping",
        "native_surface_comparison",
        "outputs-native-surface-comparison",
        "jj_tms",
        "JJ-TMS",
        "tms_contrast",
        "TMS contrast",
        "TMS-contrast",
        "tms_response",
        "TMS response",
        "TMS-response",
        "native-surface-comparison",
        "prefix_JJ_TMS_suffix",
        "TMSResponse",
        "TMSContrast",
    ]
    for variant in variants:
        with pytest.raises(ValueError, match="rest-fMRI-only scope"):
            validate_rest_only_path(tmp_path / variant)


def test_fixed_and_matching_presence_requires_surface_prerequisite(
    tmp_path: Path,
) -> None:
    """A missing consumed surface cannot be reported as unqualified present."""

    config = json.loads(CONFIG_PATH.read_text())
    config["paths"]["routea_surface_root"] = str(tmp_path / "surface")
    coverage, _ = inspect_rest_artifacts(config)

    rows = coverage[
        coverage["method"].isin(["hcp151_fixed", "hcp151_template_matching"])
    ]
    assert set(rows["status"]) == {"missing"}
    assert all("surface" in value for value in rows["missing_components"])


def test_real_rosters_have_exact_cells() -> None:
    """The frozen Stanford QC roster has the four specified factorial cells."""

    roster = build_factorial_roster()

    assert roster.groupby("four_group").size().to_dict() == {
        "NTHC": 51,
        "TEHC": 40,
        "NTS": 56,
        "TIS": 47,
    }
    assert roster["subject"].nunique() == 194
    assert set(roster[["trauma", "symptomatic"]].itertuples(index=False, name=None)) == {
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
    }


def test_real_inventory_reports_method_axes_and_schema() -> None:
    """Inventory output records actual coverage and concrete representative schemas."""

    config = json.loads(CONFIG_PATH.read_text())
    assert config["methods"] == [
        "hcp151_template_matching",
        "hcp151_fixed",
    ]
    coverage, schema = inspect_rest_artifacts(config)

    assert set(coverage["method"]) == {
        "hcp151_fixed",
        "hcp151_template_matching",
    }
    assert set(coverage["smoothing_mm"]) == {2, 4}
    assert set(coverage["hemisphere"]) == {"L", "R"}
    assert set(coverage["subject"]) == {"sub-1001"}
    assert {row["status"] for row in coverage.to_dict("records")} <= {
        "present",
        "input_available",
        "missing",
    }

    assert schema["fc_vertex_to_network"]["shape"] == [512, 400]
    assert schema["fc_vertex_to_network"]["dtype"] == "float32"
    assert schema["template_matching_labels"]["shape"] == [512]
    assert schema["template_matching_labels"]["dtype"] == "int32"
    assert schema["template_matching_manifest"]["keys"]
    assert schema["surface_timeseries"]["shape"] == [512, 234]


def test_unified_config_freezes_native_assets_and_stable_output_directories() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["native_assets"]["label_transfer_method"] == "ADAP_BARY_AREA"
    assert config["native_assets"]["native_metric_filename_pattern"].endswith(
        "_den-native_label-hipp_surfareacorobl.shape.gii"
    )
    assert config["outputs"] == {
        "features": "stanford_jjtms_multiscale_hippocampal_exploration/kember_metrics/work/features",
        "direct_contrasts": "stanford_jjtms_multiscale_hippocampal_exploration/kember_metrics/results/direct_contrasts",
        "figures": "stanford_jjtms_multiscale_hippocampal_exploration/kember_metrics/figures/direct_contrasts",
        "reports": "stanford_jjtms_multiscale_hippocampal_exploration/kember_metrics/reports/direct_contrasts",
    }

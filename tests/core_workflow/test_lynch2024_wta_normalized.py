#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = REPO_ROOT / "scripts" / "core_workflow"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))


def _load_numbered_module(script_name: str, alias: str):
    module_path = CORE_DIR / script_name
    spec = importlib.util.spec_from_file_location(alias, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


atlas_mod = _load_numbered_module("05_02_run_group_atlas.py", "core_05_02_lynch_wta")
overview_mod = _load_numbered_module("06_01_build_group_overview.py", "core_06_01_lynch_wta")


def test_normalized_topn_wta_prefers_fraction_over_raw_count() -> None:
    mean_profile = np.asarray(
        [
            [0.9, 0.8, 0.6, -0.1, 0.7, np.nan],
        ],
        dtype=np.float32,
    )
    roi_registry = [
        {"roi_id": 1, "roi_name": "default1", "canonical8": "Default", "family17": "DefaultA", "lynch20_network": "Default"},
        {"roi_id": 2, "roi_name": "default2", "canonical8": "Default", "family17": "DefaultA", "lynch20_network": "Default"},
        {"roi_id": 3, "roi_name": "default3", "canonical8": "Default", "family17": "DefaultA", "lynch20_network": "Default"},
        {"roi_id": 4, "roi_name": "default4", "canonical8": "Default", "family17": "DefaultA", "lynch20_network": "Default"},
        {"roi_id": 5, "roi_name": "visual1", "canonical8": "Visual", "family17": "VisualA", "lynch20_network": "Visual"},
        {"roi_id": 6, "roi_name": "visual2", "canonical8": "Visual", "family17": "VisualA", "lynch20_network": "Visual"},
    ]

    rows = atlas_mod.build_normalized_topn_count_vertex_rows(
        mean_profile=mean_profile,
        support_indices=np.asarray([42], dtype=np.int32),
        roi_registry=roi_registry,
        top_n_cap=5,
        network_column="canonical8",
    )

    assert rows[0]["top1_network"] == "Visual"
    assert rows[0]["top1_count"] == 1
    assert rows[0]["top1_available_rois"] == 1
    assert rows[0]["top1_normalized_count"] == 1.0
    assert rows[0]["top2_network"] == "Default"
    assert rows[0]["top2_count"] == 3
    assert rows[0]["top2_available_rois"] == 4
    assert rows[0]["top2_normalized_count"] == 0.75

    canonical8_rows = atlas_mod.build_normalized_topn_count_vertex_rows(
        mean_profile=mean_profile,
        support_indices=np.asarray([42], dtype=np.int32),
        roi_registry=roi_registry,
        top_n_cap=5,
        network_column="canonical8",
    )
    assert canonical8_rows[0]["top1_network"] == "Visual"
    assert canonical8_rows[0]["network_column"] == "canonical8"


def test_wta_method_routes_by_atlas_taxonomy() -> None:
    assert atlas_mod.method_allowed_for_atlas("wta-top", "kong2022") is True
    assert atlas_mod.method_allowed_for_atlas("wta-top", "kong2022", network_taxonomy="canonical8") is True
    assert atlas_mod.method_allowed_for_atlas("wta-top", "kong2022", network_taxonomy="family17") is True
    assert atlas_mod.method_allowed_for_atlas("wta-top", "kong2022", network_taxonomy="lynch20") is False
    assert atlas_mod.method_allowed_for_atlas("wta-top", "kong2022_7net") is False
    assert atlas_mod.method_allowed_for_atlas("wta-top", "lynch2024_7net", network_taxonomy="lynch7") is True
    assert atlas_mod.method_allowed_for_atlas("wta-top", "lynch2024_7net", network_taxonomy="canonical7") is False


def test_lynch_wta_uses_direct_network_fisherz_assignment() -> None:
    rows = atlas_mod.build_fisherz_vertex_assignment_rows(
        mean_profile=np.asarray(
            [
                [0.4, 0.8, 0.8, np.nan],
                [np.nan, np.nan, np.nan, np.nan],
            ],
            dtype=np.float32,
        ),
        support_indices=np.asarray([42, 43], dtype=np.int32),
        network_names=["Default", "Visual", "Control", "Language"],
        network_column="lynch7",
    )
    assert rows[0]["top1_network"] == "Visual"
    assert rows[0]["top2_network"] == "Control"
    assert rows[0]["margin"] == 0.0
    assert rows[0]["low_margin_lt_0p05"] == 1
    assert rows[0]["n_networks_available"] == 3
    assert rows[0]["assignment_metric"] == "top1_fisher_z"
    assert rows[1]["status"] == "unassigned"
    assert rows[1]["unassigned_reason"] == "no_finite_network_fisher_z"
    assert atlas_mod.network_feature_filename("wta-top", "lynch2024_7net") == "hipp_vertex_to_network_fc.npy"
    assert atlas_mod.feature_family_for_method("wta-top", "lynch2024_7net", (1, 2, 3)) == "vertex_to_lynch2024_7net_network_fc_fisherz"
    assert atlas_mod.fc_mode_for_method("wta-top", "lynch2024_7net") == "pearson_vertex_to_network"


def test_lynch_omitted_methods_include_direct_wta() -> None:
    expected = ("gradient-spectral", "wta-top")
    assert atlas_mod.default_methods_for_atlas("lynch2024_7net") == expected
    assert atlas_mod.default_methods_for_atlas("lynch2024_20net") == expected
    assert overview_mod.default_methods_for_atlas("lynch2024_7net") == expected
    args = atlas_mod.build_arg_parser().parse_args(["--atlases", "lynch2024_7net"])
    assert args.methods is None


def test_lynch_omitted_taxonomy_is_inferred_from_atlas() -> None:
    assert atlas_mod.default_network_taxonomy_for_atlases(("lynch2024_20net",)) == "lynch20"
    assert atlas_mod.default_network_taxonomy_for_atlases(("lynch2024_7net",)) == "lynch7"
    with pytest.raises(ValueError, match="Cannot infer one network taxonomy"):
        atlas_mod.default_network_taxonomy_for_atlases(("kong2022", "lynch2024_7net"))


def test_group_atlas_accepts_family17_wta_and_rejects_unknown_taxonomy() -> None:
    atlas_mod.validate_requested_methods_for_taxonomy(("gradient-spectral", "wta-top"), "family17", ("kong2022",))
    atlas_mod.validate_requested_methods_for_taxonomy(("gradient-spectral",), "family17")
    with pytest.raises(ValueError, match="canonical7/canonical8/family17"):
        atlas_mod.validate_requested_methods_for_taxonomy(("wta-top",), "kong17")


def test_group_atlas_rejects_wta_request_for_non_kong_atlas() -> None:
    with pytest.raises(ValueError, match="unsupported WTA atlases"):
        atlas_mod.validate_requested_methods_for_taxonomy(
            ("wta-top",),
            "canonical8",
            ("kong2019",),
        )
    atlas_mod.validate_requested_methods_for_taxonomy(
        ("wta-top",),
        "canonical8",
        ("kong2022",),
    )

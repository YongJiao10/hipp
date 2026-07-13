#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


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
    )

    assert rows[0]["top1_network"] == "Visual"
    assert rows[0]["top1_count"] == 1
    assert rows[0]["top1_available_rois"] == 1
    assert rows[0]["top1_normalized_count"] == 1.0
    assert rows[0]["top2_network"] == "Default"
    assert rows[0]["top2_count"] == 3
    assert rows[0]["top2_available_rois"] == 4
    assert rows[0]["top2_normalized_count"] == 0.75

    lynch_rows = atlas_mod.build_normalized_topn_count_vertex_rows(
        mean_profile=mean_profile,
        support_indices=np.asarray([42], dtype=np.int32),
        roi_registry=roi_registry,
        top_n_cap=5,
        network_column="lynch20",
    )
    assert lynch_rows[0]["top1_network"] == "Visual"
    assert lynch_rows[0]["network_column"] == "lynch20_network"


def test_normalized_wta_variant_is_default_when_roi_registry_is_available() -> None:
    specs = atlas_mod.assignment_variant_specs_for_roi_registry(
        fisherz_status_rows=[
            {"cluster_id": 1, "status": "dominant", "top1_network": "Default"},
        ],
        fisherz_dominant=["Default"],
        normalized_status_rows=[
            {"cluster_id": 1, "status": "dominant", "top1_network": "Visual"},
        ],
    )

    assert specs["default_variant"] == "normalized_top20_count"
    assert set(specs["variants"]) == {"fisherz", "normalized_top20_count"}
    assert specs["variants"]["normalized_top20_count"]["assignment_metric"] == "top20_count_normalized_by_network_roi_count"

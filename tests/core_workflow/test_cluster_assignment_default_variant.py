#!/usr/bin/env python3
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = REPO_ROOT / "scripts" / "core_workflow"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))


atlas = importlib.import_module("scripts.core_workflow.05_02_run_group_atlas")
template_profiles = importlib.import_module("scripts.core_workflow.lib.cortex_template_profiles")


def test_cluster_level_default_assignment_variant_is_fisherz() -> None:
    assert (
        atlas.select_cluster_level_default_assignment_variant(
            {"fisherz": {}, "normalized_top20_count": {}}
        )
        == "fisherz"
    )


def test_cluster_level_default_fails_closed_without_fisherz() -> None:
    with pytest.raises(ValueError, match="fisherz"):
        atlas.select_cluster_level_default_assignment_variant(
            {"normalized_top20_count": {}}
        )


def test_kong_roi_registry_wta_default_remains_normalized_top20_count() -> None:
    assert template_profiles.DEFAULT_WTA_ASSIGNMENT_VARIANT == "normalized_top20_count"

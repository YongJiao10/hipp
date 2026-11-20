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


def test_kong2022_cluster_level_default_assignment_variant_is_network_ts_fisherz() -> None:
    assert (
        atlas.select_cluster_level_default_assignment_variant(
            {"fisherz": {}, "normalized_top20_count": {}, "network_ts_fisherz": {}},
            atlas="kong2022",
        )
        == "network_ts_fisherz"
    )


def test_kong2022_cluster_level_default_fails_closed_without_network_ts_fisherz() -> None:
    with pytest.raises(ValueError, match="network_ts_fisherz"):
        atlas.select_cluster_level_default_assignment_variant(
            {"fisherz": {}, "normalized_top20_count": {}},
            atlas="kong2022",
        )


def test_lynch_cluster_level_default_remains_direct_network_fisherz() -> None:
    for lynch in ("lynch2024_20net", "lynch2024_7net"):
        assert (
            atlas.select_cluster_level_default_assignment_variant({"fisherz": {}}, atlas=lynch)
            == "fisherz"
        )


def test_resume_invalidates_manifest_with_previous_default(tmp_path: Path) -> None:
    import json

    manifest = tmp_path / "group_atlas_manifest.json"
    payload = {
        "status": "ok",
        "resume_input_signature_version": atlas.RESUME_SIGNATURE_VERSION,
        "resume_input_signature": "sig",
        "default_assignment_variant": "fisherz",
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    assert atlas.resume_manifest_matches(
        manifest, "sig", expected_default_assignment_variant="network_ts_fisherz"
    ) == (False, "default_assignment_variant_changed")
    payload["default_assignment_variant"] = "network_ts_fisherz"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    assert atlas.resume_manifest_matches(
        manifest, "sig", expected_default_assignment_variant="network_ts_fisherz"
    ) == (True, "signature_match")


def test_kong_roi_registry_wta_default_remains_normalized_top20_count() -> None:
    assert template_profiles.DEFAULT_WTA_ASSIGNMENT_VARIANT == "normalized_top20_count"

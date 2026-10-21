# pyright: reportMissingImports=false
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_WORKFLOW_DIR = REPO_ROOT / "scripts" / "core_workflow"
BRANCH_CONFIG_PATH = CORE_WORKFLOW_DIR / "lib" / "branch_config.py"
if str(CORE_WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_WORKFLOW_DIR))

from scripts.core_workflow.lib.route_tags import (  # noqa: E402
    validate_group_atlas_root_for_route,
)


def load_branch_config():
    assert BRANCH_CONFIG_PATH.is_file(), "signal-branch contract module is missing"
    spec = importlib.util.spec_from_file_location("signal_branch_config_under_test", BRANCH_CONFIG_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_signal_branch_mapping_rejects_crossed_route() -> None:
    config = load_branch_config()

    assert config.signal_branch_for_route("pfm") == "standard"
    assert config.signal_branch_for_route("pfm_ctxreg") == "ctxreg"
    assert config.validate_signal_branch_route("ctxreg", "pfm_ctxreg") == "ctxreg"
    with pytest.raises(ValueError, match="Signal branch/route mismatch"):
        config.validate_signal_branch_route("ctxreg", "pfm")


def test_group_atlas_root_validation_rejects_cross_taxonomy_route(tmp_path: Path) -> None:
    run_root = (
        tmp_path
        / "outputs_hcp"
        / "work"
        / "group_atlas"
        / "pfm_ctxreg"
    )

    with pytest.raises(ValueError, match="Provenance discrepancy"):
        validate_group_atlas_root_for_route(
            run_root,
            "lynch2024_7net",
        )

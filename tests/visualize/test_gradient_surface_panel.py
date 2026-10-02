from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = ROOT / "scripts" / "core_workflow"
VIS_LIB_DIR = ROOT / "scripts" / "visualize" / "lib"
for path in (CORE_DIR, VIS_LIB_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _load_panel():
    assert importlib.util.find_spec("gradient_surface_panel") is not None, (
        "formal gradient surface panel module is missing"
    )
    return importlib.import_module("gradient_surface_panel")


def test_bar_composition_is_bound_to_validated_surface_assignment() -> None:
    entries = [
        {"cluster_id": 1, "comp": {"Default": 25.0, "Somatomotor": 75.0}},
        {"cluster_id": 2, "comp": {"Default": 100.0}},
        {"cluster_id": 3, "comp": {"VentralAttention": 100.0}},
    ]
    validated = {
        "winners": {1: "Somatomotor", 2: "Default", 3: "VentralAttention"},
        "compositions": {
            1: {"Default": 25.0, "Somatomotor": 75.0},
            2: {"Default": 100.0},
            3: {"VentralAttention": 100.0},
        },
    }

    _load_panel().validate_bar_composition(
        entries, validated, context="synthetic-run"
    )

    assert [entry["dominant_network"] for entry in entries] == [
        "Somatomotor",
        "Default",
        "VentralAttention",
    ]


def test_bar_composition_rejects_stale_percentage_against_surface_assignment() -> None:
    entries = [
        {"cluster_id": 1, "comp": {"Default": 100.0}},
        {"cluster_id": 2, "comp": {"Default": 100.0}},
        {"cluster_id": 3, "comp": {"VentralAttention": 100.0}},
    ]
    validated = {
        "winners": {1: "Somatomotor", 2: "Default", 3: "VentralAttention"},
        "compositions": {
            1: {"Default": 25.0, "Somatomotor": 75.0},
            2: {"Default": 100.0},
            3: {"VentralAttention": 100.0},
        },
    }

    with pytest.raises(ValueError, match="composition"):
        _load_panel().validate_bar_composition(
            entries, validated, context="synthetic-run"
        )

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    REPO_ROOT
    / "scripts"
    / "core_workflow"
    / "06_11_compare_gradient_spectral_affinities.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("gradient_affinity_comparison", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_compare_label_arrays_reports_semantic_dice_and_change_fraction():
    comparison = load_module()
    old = np.asarray([0, 1, 1, 2, 2, 3, 3], dtype=np.int32)
    new = np.asarray([0, 1, 2, 2, 2, 3, 3], dtype=np.int32)

    got = comparison.compare_label_arrays(old, new, k=3)

    assert got["n_common_support"] == 6
    assert got["changed_vertex_count"] == 1
    assert got["changed_vertex_fraction"] == pytest.approx(1.0 / 6.0)
    assert got["dice_anterior"] == pytest.approx(2.0 / 3.0)
    assert got["dice_body"] == pytest.approx(0.8)
    assert got["dice_posterior"] == pytest.approx(1.0)
    assert got["hungarian_mapping_identity"] is True


def test_compare_label_arrays_rejects_shape_mismatch():
    comparison = load_module()
    with pytest.raises(ValueError, match="shape mismatch"):
        comparison.compare_label_arrays(np.ones(2), np.ones(3), k=3)

"""Contracts separating descriptive atlas geometry from Kember metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from atlas_compare import compute_atlas_region_area  # noqa: E402


def test_atlas_area_outputs_are_explicitly_namespaced() -> None:
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    faces = np.asarray([[0, 1, 2], [0, 1, 3]])
    labels = np.asarray([1, 1, 2, 3])

    result = compute_atlas_region_area(vertices, faces, labels)

    assert "atlas_region_area_mm2" in result
    assert "atlas_region_area_fraction" in result
    assert "area_mm2" not in result
    assert "area_fraction" not in result

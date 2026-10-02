"""Behavioral contracts for Kember-method system surface area."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from surface_area import (  # noqa: E402
    combine_hemisphere_means,
    compute_system_surface_area,
)


def test_surface_area_sums_consumed_native_vertex_area_within_each_system() -> None:
    """Kember area is the sum of the supplied native per-vertex mm² values."""

    result = compute_system_surface_area(
        np.array([0.2, 0.3, 0.4, 0.1], dtype=np.float64),
        np.array([1, 1, 2, 3], dtype=np.int32),
    )

    assert result["system_area_mm2"] == pytest.approx(
        {"anterior": 0.5, "body": 0.4, "posterior": 0.1}
    )
    assert result["system_area_fraction"] == pytest.approx(
        {"anterior": 0.5, "body": 0.4, "posterior": 0.1}
    )
    assert result["total_valid_surface_area_mm2"] == pytest.approx(1.0)
    assert result["total_labeled_area_mm2"] == pytest.approx(1.0)
    assert result["unassigned_surface_area_mm2"] == pytest.approx(0.0)
    assert result["vertex_area_source"] == "hippunfold_native_surfareacorobl"
    assert "native per-vertex surface-area values" in result["vertex_area_convention"]
    assert result["systems"]["anterior"]["surface_area_mm2"] == pytest.approx(0.5)
    assert result["surface_area_mm2"] == pytest.approx(
        {"anterior": 0.5, "body": 0.4, "posterior": 0.1}
    )


def test_surface_area_fraction_uses_all_valid_native_surface_area() -> None:
    """Fails if unlabeled valid or excluded native vertices distort the denominator."""

    result = compute_system_surface_area(
        np.array([0.2, 0.3, 0.4, 0.1, 0.05], dtype=np.float64),
        np.array([1, 1, 2, 0, 0], dtype=np.int32),
        valid_mask=np.array([True, True, True, True, False]),
    )

    assert result["total_surface_area_mm2"] == pytest.approx(1.05)
    assert result["total_valid_surface_area_mm2"] == pytest.approx(1.0)
    assert result["total_labeled_area_mm2"] == pytest.approx(0.9)
    assert result["unassigned_surface_area_mm2"] == pytest.approx(0.15)
    assert result["system_area_fraction"] == pytest.approx(
        {"anterior": 0.5, "body": 0.4, "posterior": 0.0}
    )


def test_surface_area_rejects_unknown_system_label() -> None:
    """Fails if a label-order mismatch is turned into a numerical result."""

    with pytest.raises(ValueError, match="unsupported system label"):
        compute_system_surface_area(
            np.ones(4, dtype=np.float64),
            np.array([1, 2, 3, 4], dtype=np.int32),
        )


def test_surface_area_rejects_non_metric_area_vector() -> None:
    """An invalid or differently-sized area vector cannot be substituted."""

    with pytest.raises(ValueError, match="per-vertex area"):
        compute_system_surface_area(
            np.array([0.5, -0.1, 0.6], dtype=np.float64),
            np.array([1, 2, 3], dtype=np.int32),
        )


def test_combined_area_is_the_arithmetic_mean_of_left_and_right_values() -> None:
    """Fails if the published left/right convention is changed to a sum."""

    result = combine_hemisphere_means(
        {"anterior": 2.0, "body": 4.0, "posterior": 6.0},
        {"anterior": 4.0, "body": 8.0, "posterior": 10.0},
    )

    assert result == pytest.approx({"anterior": 3.0, "body": 6.0, "posterior": 8.0})

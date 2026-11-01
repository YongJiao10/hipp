"""Kember-method system surface-area calculations.

The published method sums HippUnfold midthickness per-vertex surface-area
values within each anterior/middle/posterior system.  This module accepts those
native, mm²-valued metrics directly, then applies the frozen label contract
(1=anterior, 2=middle, 3=posterior).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


SYSTEM_LABELS = {1: "anterior", 2: "middle", 3: "posterior"}
SYSTEM_NAMES = tuple(SYSTEM_LABELS.values())


def compute_system_surface_area(
    vertex_area_mm2: np.ndarray,
    labels: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Sum supplied native HippUnfold per-vertex mm² values by system label.

    Fractions use the total native surface area within the valid native ROI as
    their denominator.  The full consumed metric area and the valid-ROI area
    are both returned, while any area not assigned to a labelled system is
    reported separately.
    """

    vertex_area = np.asarray(vertex_area_mm2, dtype=np.float64)
    if vertex_area.ndim != 1:
        raise ValueError("per-vertex area must be one-dimensional")
    if not np.isfinite(vertex_area).all() or np.any(vertex_area < 0):
        raise ValueError("per-vertex area must be finite and non-negative")

    raw_labels = np.asarray(labels)
    if raw_labels.ndim != 1 or raw_labels.size != vertex_area.size:
        raise ValueError("labels must be one-dimensional and match per-vertex area")
    if not np.issubdtype(raw_labels.dtype, np.integer):
        raise ValueError("system labels must be integer-coded")
    system_labels = raw_labels.astype(np.int64, copy=False)
    unknown = sorted(set(np.unique(system_labels)) - {0, *SYSTEM_LABELS})
    if unknown:
        raise ValueError(f"unsupported system label(s): {unknown}")

    if valid_mask is None:
        valid_surface = np.ones(system_labels.size, dtype=bool)
    else:
        valid_surface = np.asarray(valid_mask, dtype=bool)
        if valid_surface.ndim != 1 or valid_surface.size != system_labels.size:
            raise ValueError("valid_mask must match the surface vertex count")
        valid_surface = valid_surface.copy()
    retained = valid_surface & (system_labels != 0)

    total_surface_area = float(np.sum(vertex_area))
    total_valid_surface_area = float(np.sum(vertex_area[valid_surface]))
    total_labeled_area = float(np.sum(vertex_area[retained]))
    if total_valid_surface_area <= 0:
        raise ValueError("valid native surface has zero surface area")
    if total_labeled_area <= 0:
        raise ValueError("retained system labels have zero surface area")

    system_area = {
        name: float(np.sum(vertex_area[retained & (system_labels == label)]))
        for label, name in SYSTEM_LABELS.items()
    }
    system_fraction = {
        name: area / total_valid_surface_area for name, area in system_area.items()
    }
    vertex_area_source = "hippunfold_native_surfareacorobl"
    vertex_area_convention = (
        "HippUnfold native per-vertex surface-area values in mm^2 summed over "
        "vertices selected by the native system label"
    )
    systems = {
        name: {
            "surface_area_mm2": area,
            "surface_area_fraction": system_fraction[name],
            "total_surface_area_mm2": total_surface_area,
            "total_valid_surface_area_mm2": total_valid_surface_area,
            "total_labeled_area_mm2": total_labeled_area,
            "unassigned_surface_area_mm2": total_surface_area - total_labeled_area,
            "vertex_area_source": vertex_area_source,
            "vertex_area_convention": vertex_area_convention,
        }
        for name, area in system_area.items()
    }
    return {
        "systems": systems,
        "surface_area_mm2": system_area,
        "surface_area_fraction": system_fraction,
        "system_area_mm2": system_area,
        "system_area_fraction": system_fraction,
        "total_surface_area_mm2": total_surface_area,
        "total_valid_surface_area_mm2": total_valid_surface_area,
        "total_labeled_area_mm2": total_labeled_area,
        "unassigned_surface_area_mm2": total_surface_area - total_labeled_area,
        "valid_labeled_vertex_count": int(np.sum(retained)),
        "vertex_area_source": vertex_area_source,
        "vertex_area_convention": vertex_area_convention,
    }


def combine_hemisphere_means(
    left: Mapping[str, float], right: Mapping[str, float]
) -> dict[str, float]:
    """Return the published arithmetic mean of left and right system values."""

    if set(left) != set(SYSTEM_NAMES) or set(right) != set(SYSTEM_NAMES):
        raise ValueError("both hemisphere mappings must contain anterior, middle, and posterior")
    result = {name: (float(left[name]) + float(right[name])) / 2.0 for name in SYSTEM_NAMES}
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("hemisphere values must be finite")
    return result

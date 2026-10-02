"""Integration contract for native HippUnfold area in Kember features."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


def test_kember_feature_area_uses_supplied_native_metric_not_mesh_geometry() -> None:
    """Native ``surfareacorobl`` remains authoritative after label alignment."""

    feature_build = importlib.import_module("feature_build")
    labels = np.array([1, 1, 2, 3, 3, 3], dtype=np.int32)
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [2.0, 2.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [[0, 1, 2], [1, 3, 2], [1, 4, 3], [4, 5, 3]], dtype=np.int64
    )
    native_vertex_area_mm2 = np.array([11.0, 13.0, 2.0, 3.0, 5.0, 7.0])
    time_series = np.column_stack(
        [np.arange(6, dtype=float), np.arange(6, dtype=float) ** 2, np.ones(6)]
    )

    metrics = feature_build.compute_kember_metrics(
        labels=labels,
        time_series=time_series,
        native_vertex_area_mm2=native_vertex_area_mm2,
        native_valid_mask=np.ones(labels.size, dtype=bool),
        coordinates=coordinates,
        faces=faces,
        vertex_area_metric_path=Path("sub-1001_hemi-L_surfareacorobl.shape.gii"),
        boundary_radius_mm=10.0,
    )

    geometric_vertex_area = np.zeros(coordinates.shape[0], dtype=np.float64)
    triangle_area = 0.5 * np.linalg.norm(
        np.cross(
            coordinates[faces[:, 1]] - coordinates[faces[:, 0]],
            coordinates[faces[:, 2]] - coordinates[faces[:, 0]],
        ),
        axis=1,
    )
    for corner in range(3):
        np.add.at(geometric_vertex_area, faces[:, corner], triangle_area / 3.0)
    anterior = metrics["anterior"]
    assert anterior["surface_area_mm2"] == 24.0
    assert anterior["vertex_area_source"] == "hippunfold_native_surfareacorobl"
    assert anterior["surface_area_mm2"] != float(np.sum(geometric_vertex_area[:2]))

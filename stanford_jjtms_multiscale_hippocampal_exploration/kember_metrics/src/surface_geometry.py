"""Geometry helpers for consumed Stanford HippUnfold surfaces.

These helpers load consumed mesh geometry and construct adjacency for the
functional boundary endpoint. They deliberately do not reconstruct vertex
area: Kember area uses the native HippUnfold ``surfareacorobl`` metric.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import sparse


def load_midthickness(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load point coordinates and triangle faces from a consumed GIFTI file."""

    surface_path = Path(path)
    image = nib.load(str(surface_path))
    coordinates: np.ndarray | None = None
    faces: np.ndarray | None = None
    for data_array in image.darrays:
        values = np.asarray(data_array.data)
        if data_array.intent == "NIFTI_INTENT_POINTSET" and coordinates is None:
            coordinates = values
        elif data_array.intent == "NIFTI_INTENT_TRIANGLE" and faces is None:
            faces = values
    if coordinates is None and image.darrays:
        coordinates = np.asarray(image.darrays[0].data)
    if faces is None and len(image.darrays) > 1:
        faces = np.asarray(image.darrays[1].data)
    if coordinates is None or faces is None:
        raise ValueError(f"midthickness GIFTI lacks pointset/triangle arrays: {surface_path}")
    coordinates = np.asarray(coordinates, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    _validate_geometry(coordinates, faces)
    return coordinates, faces


def _validate_geometry(coordinates: np.ndarray, faces: np.ndarray) -> None:
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(f"coordinates must have shape (N, 3), got {coordinates.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape (M, 3), got {faces.shape}")
    if not np.isfinite(coordinates).all():
        raise ValueError("coordinates contain non-finite values")
    if faces.size and (faces.min() < 0 or faces.max() >= coordinates.shape[0]):
        raise ValueError("faces contain a vertex index outside coordinates")


def surface_adjacency(faces: np.ndarray, n_vertices: int) -> sparse.csr_matrix:
    """Build an undirected binary vertex adjacency matrix from triangle faces."""

    triangles = np.asarray(faces, dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(f"faces must have shape (M, 3), got {triangles.shape}")
    if n_vertices < 1:
        raise ValueError("n_vertices must be positive")
    if triangles.size and (triangles.min() < 0 or triangles.max() >= n_vertices):
        raise ValueError("faces contain an out-of-range vertex")
    edges = (
        np.concatenate(
            [
                triangles[:, [0, 1]],
                triangles[:, [1, 2]],
                triangles[:, [2, 0]],
            ],
            axis=0,
        )
        if triangles.size
        else np.empty((0, 2), dtype=np.int64)
    )
    if edges.size == 0:
        return sparse.csr_matrix((n_vertices, n_vertices), dtype=np.float64)
    all_edges = np.concatenate([edges, edges[:, ::-1]], axis=0)
    values = np.ones(all_edges.shape[0], dtype=np.float64)
    adjacency = sparse.coo_matrix(
        (values, (all_edges[:, 0], all_edges[:, 1])),
        shape=(n_vertices, n_vertices),
    ).tocsr()
    adjacency.data[:] = 1.0
    adjacency.setdiag(0.0)
    adjacency.eliminate_zeros()
    return adjacency

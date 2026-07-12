from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_WORKFLOW_DIR = REPO_ROOT / "scripts" / "core_workflow"
if str(CORE_WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_WORKFLOW_DIR))

from lib.surface_affinity import (  # noqa: E402
    DEFAULT_SURFACE_AFFINITY,
    affinity_graph_diagnostics,
    build_surface_affinity,
    surface_affinity_description,
)


def test_shifted_cosine_is_default_and_preserves_negative_cosine_edges() -> None:
    features = np.asarray([[1.0, 0.0], [-0.5, np.sqrt(3.0) / 2.0]], dtype=np.float32)
    adjacency = sparse.csr_matrix(np.asarray([[0.0, 1.0], [1.0, 0.0]]))

    affinity = build_surface_affinity(features, adjacency)

    assert DEFAULT_SURFACE_AFFINITY == "shifted-cosine"
    assert np.allclose(affinity.toarray(), [[0.0, 0.25], [0.25, 0.0]])
    assert surface_affinity_description(DEFAULT_SURFACE_AFFINITY) == (
        "((1+cosine_similarity)/2)_hadamard_surface_adjacency_diag0"
    )


def test_relu_cosine_reproduces_legacy_zero_clipping() -> None:
    features = np.asarray([[1.0, 0.0], [-0.5, np.sqrt(3.0) / 2.0]], dtype=np.float32)
    adjacency = sparse.csr_matrix(np.asarray([[0.0, 1.0], [1.0, 0.0]]))

    affinity = build_surface_affinity(features, adjacency, mode="relu-cosine")

    assert np.allclose(affinity.toarray(), np.zeros((2, 2)))
    assert surface_affinity_description("relu-cosine") == (
        "max(cosine_similarity,0)_hadamard_surface_adjacency_diag0"
    )


def test_surface_affinity_rejects_unknown_mode_and_shape_mismatch() -> None:
    features = np.eye(2, dtype=np.float32)
    adjacency = sparse.eye(3, format="csr")

    with pytest.raises(ValueError, match="adjacency/features mismatch"):
        build_surface_affinity(features, adjacency)
    with pytest.raises(ValueError, match="Unsupported surface affinity"):
        build_surface_affinity(features, sparse.eye(2, format="csr"), mode="rbf")


def test_affinity_graph_diagnostics_reports_components_and_cluster_sizes() -> None:
    graph = sparse.csr_matrix(
        np.asarray(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
    )

    got = affinity_graph_diagnostics(
        graph,
        labels=np.asarray([1, 1, 2], dtype=np.int32),
        k=2,
    )

    assert got == {
        "n_vertices": 3,
        "n_connected_components": 2,
        "connected_component_sizes": [2, 1],
        "cluster_vertex_counts": {"1": 2, "2": 1},
        "min_cluster_fraction": pytest.approx(1.0 / 3.0),
    }

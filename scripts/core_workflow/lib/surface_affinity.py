from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


GRADIENT_SPECTRAL_AFFINITY = "shifted-cosine"
GRADIENT_SPECTRAL_AFFINITY_DESCRIPTION = (
    "((1+cosine_similarity)/2)_hadamard_surface_adjacency_diag0"
)
POSITIVE_COSINE_AFFINITY = "relu-cosine"
POSITIVE_COSINE_AFFINITY_DESCRIPTION = (
    "max(cosine_similarity,0)_hadamard_surface_adjacency_diag0"
)


def _cosine_on_surface_edges(
    features: np.ndarray,
    adjacency: sparse.spmatrix,
) -> tuple[np.ndarray, sparse.csr_matrix]:
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"features must be 2D, got shape={x.shape}")
    adj = sparse.csr_matrix(adjacency, dtype=np.float32)
    if adj.shape != (x.shape[0], x.shape[0]):
        raise ValueError(
            "adjacency/features mismatch: "
            f"adjacency={adj.shape} features={x.shape}"
        )
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    normalized = x / np.clip(norms, 1e-12, None)
    cosine = np.clip(normalized @ normalized.T, -1.0, 1.0)
    return cosine, adj


def _fuse_with_surface(
    dense_affinity: np.ndarray,
    adjacency: sparse.csr_matrix,
) -> sparse.csr_matrix:
    fused = sparse.csr_matrix(dense_affinity).multiply(adjacency).tocsr()
    fused = fused.maximum(fused.T).tocsr()
    fused.setdiag(0.0)
    fused.eliminate_zeros()
    return fused.astype(np.float32)


def build_gradient_spectral_affinity(
    features: np.ndarray,
    adjacency: sparse.spmatrix,
) -> sparse.csr_matrix:
    cosine, adj = _cosine_on_surface_edges(features, adjacency)
    shifted = (cosine + np.float32(1.0)) / np.float32(2.0)
    return _fuse_with_surface(shifted, adj)


def build_positive_cosine_surface_affinity(
    features: np.ndarray,
    adjacency: sparse.spmatrix,
) -> sparse.csr_matrix:
    """Build the established affinity used by non-gradient spectral methods."""
    cosine, adj = _cosine_on_surface_edges(features, adjacency)
    positive = np.maximum(cosine, np.float32(0.0))
    return _fuse_with_surface(positive, adj)


def affinity_graph_diagnostics(
    affinity: sparse.spmatrix,
    *,
    labels: np.ndarray,
    k: int,
) -> dict[str, object]:
    graph = sparse.csr_matrix(affinity)
    values = np.asarray(labels, dtype=np.int32).reshape(-1)
    if graph.shape != (values.size, values.size):
        raise ValueError(
            "affinity/labels mismatch: "
            f"affinity={graph.shape} labels={values.shape}"
        )
    n_components, component_labels = connected_components(graph, directed=False)
    component_sizes = sorted(
        np.bincount(component_labels, minlength=int(n_components)).astype(int).tolist(),
        reverse=True,
    )
    cluster_counts = {
        str(label): int(np.sum(values == label))
        for label in range(1, int(k) + 1)
    }
    minimum = min(cluster_counts.values()) if cluster_counts else 0
    return {
        "n_vertices": int(values.size),
        "n_connected_components": int(n_components),
        "connected_component_sizes": component_sizes,
        "cluster_vertex_counts": cluster_counts,
        "min_cluster_fraction": (
            float(minimum / values.size) if values.size else 0.0
        ),
    }

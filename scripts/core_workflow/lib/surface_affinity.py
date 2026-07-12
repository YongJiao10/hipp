from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


DEFAULT_SURFACE_AFFINITY = "shifted-cosine"
SURFACE_AFFINITY_CHOICES = (DEFAULT_SURFACE_AFFINITY, "relu-cosine")

_DESCRIPTIONS = {
    "shifted-cosine": "((1+cosine_similarity)/2)_hadamard_surface_adjacency_diag0",
    "relu-cosine": "max(cosine_similarity,0)_hadamard_surface_adjacency_diag0",
}


def surface_affinity_description(mode: str) -> str:
    key = str(mode)
    if key not in SURFACE_AFFINITY_CHOICES:
        raise ValueError(
            f"Unsupported surface affinity: {key!r}; expected {SURFACE_AFFINITY_CHOICES}"
        )
    return _DESCRIPTIONS[key]


def build_surface_affinity(
    features: np.ndarray,
    adjacency: sparse.spmatrix,
    *,
    mode: str = DEFAULT_SURFACE_AFFINITY,
) -> sparse.csr_matrix:
    key = str(mode)
    surface_affinity_description(key)
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
    if key == "shifted-cosine":
        dense_affinity = (cosine + np.float32(1.0)) / np.float32(2.0)
    else:
        dense_affinity = np.maximum(cosine, np.float32(0.0))
    fused = sparse.csr_matrix(dense_affinity).multiply(adj).tocsr()
    fused = fused.maximum(fused.T).tocsr()
    fused.setdiag(0.0)
    fused.eliminate_zeros()
    return fused.astype(np.float32)


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

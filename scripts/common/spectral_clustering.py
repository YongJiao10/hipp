"""Spatially constrained spectral clustering for hippocampal functional parcellation.

Pipeline (Steps 1-5):
  1. Pearson FC: hipp_timeseries x cortical_networks -> feature matrix X (N, D)
  2. Functional affinity: cosine similarity of X, mapped to [0, 1]
  3. Spatial adjacency: binary symmetric mesh adjacency from hippocampal triangles
  4. Graph fusion: Hadamard product of functional affinity and spatial adjacency
  5. Spectral clustering on the fused graph via scikit-learn
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import sparse

_COMMON_DIR = Path(__file__).resolve().parent
if str(_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(_COMMON_DIR))

from compute_fc_gradients import corrcoef_rows


def _reorder_cluster_labels(raw_labels: np.ndarray) -> np.ndarray:
    """Reorder integer labels so that cluster 1 is the largest, 2 the next, etc."""
    unique = sorted(int(x) for x in np.unique(raw_labels))
    size_pairs = sorted(
        ((int(np.count_nonzero(raw_labels == key)), key) for key in unique),
        key=lambda item: (-item[0], item[1]),
    )
    mapping = {old_key: new_idx + 1 for new_idx, (_size, old_key) in enumerate(size_pairs)}
    return np.asarray([mapping[int(key)] for key in raw_labels], dtype=np.int32)


def _build_functional_affinity(X: np.ndarray) -> np.ndarray:
    """Step 2: Cosine similarity of feature rows, linearly mapped to [0, 1].

    Args:
        X: Feature matrix of shape (N, D).

    Returns:
        Dense affinity matrix W_FC of shape (N, N), values in [0, 1], diagonal=1.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_unit = X / np.clip(norms, 1e-12, None)
    sim = X_unit @ X_unit.T
    sim = np.clip(sim, -1.0, 1.0)
    W_fc = (sim + 1.0) * 0.5
    np.fill_diagonal(W_fc, 1.0)
    return W_fc.astype(np.float32)


def _fuse_graphs(W_fc: np.ndarray, W_spatial: sparse.csr_matrix) -> sparse.csr_matrix:
    """Step 4: Hadamard product of functional affinity and spatial adjacency.

    Retains only functional weights where vertices are spatial neighbors.

    Args:
        W_fc: Dense functional affinity matrix (N, N) in [0, 1].
        W_spatial: Sparse binary spatial adjacency (N, N).

    Returns:
        Sparse fused weight matrix (N, N).
    """
    W_fc_sparse = sparse.csr_matrix(W_fc)
    W_final = W_spatial.multiply(W_fc_sparse)
    W_final = sparse.csr_matrix(W_final)
    W_final.eliminate_zeros()
    return W_final


def _spectral_embed_and_cluster(
    W: sparse.csr_matrix,
    n_clusters: int,
    random_state: int,
) -> np.ndarray:
    """Step 5: Spectral clustering on the fused graph.

    Args:
        W: Sparse symmetric weight matrix (N, N).
        n_clusters: Number of clusters K.
        random_state: Random seed for KMeans.

    Returns:
        Integer cluster labels of shape (N,), 1-indexed and ordered by cluster size.
    """
    from sklearn.cluster import SpectralClustering

    n = W.shape[0]
    if n <= n_clusters:
        raise RuntimeError(
            f"Not enough vertices ({n}) to compute {n_clusters} spectral components"
        )

    clustering = SpectralClustering(
        n_clusters=n_clusters,
        eigen_solver="arpack",
        random_state=random_state,
        n_init=10,
        affinity="precomputed",
        assign_labels="kmeans",
    )
    raw_labels = clustering.fit_predict(W)
    return _reorder_cluster_labels(raw_labels)


def spectral_cluster_from_features(
    features: np.ndarray,
    spatial_adjacency: sparse.csr_matrix,
    n_clusters: int,
    random_state: int = 42,
) -> np.ndarray:
    """Steps 2-5: Spatially constrained spectral clustering from a pre-computed feature matrix.

    Args:
        features: Pre-computed feature matrix of shape (N, D), e.g. z-scored FC profiles.
        spatial_adjacency: Sparse binary mesh adjacency of shape (N, N).
        n_clusters: Number of clusters.
        random_state: Random seed for KMeans.

    Returns:
        Integer labels of shape (N,), 1-indexed ordered by cluster size.
    """
    W_fc = _build_functional_affinity(features)
    W_final = _fuse_graphs(W_fc, spatial_adjacency)
    return _spectral_embed_and_cluster(W_final, n_clusters, random_state)


def spatially_constrained_spectral_clustering(
    hipp_timeseries: np.ndarray,
    cortical_networks: np.ndarray,
    spatial_adjacency: sparse.csr_matrix,
    n_clusters: int,
    random_state: int = 42,
) -> np.ndarray:
    """End-to-end spatially constrained spectral clustering for hippocampal vertices.

    Args:
        hipp_timeseries: BOLD time series of valid hippocampal vertices, shape (N, T).
        cortical_networks: Mean time series of cortical functional networks, shape (D, T).
        spatial_adjacency: Sparse binary mesh adjacency of hippocampal vertices, shape (N, N).
        n_clusters: Number of clusters.
        random_state: Random seed for KMeans reproducibility (default 42).

    Returns:
        Cluster labels of shape (N,), integer, 1-indexed and ordered by cluster size.
    """
    # Step 1: Pearson correlation features (N, D)
    features = corrcoef_rows(hipp_timeseries, cortical_networks)
    return spectral_cluster_from_features(features, spatial_adjacency, n_clusters, random_state=random_state)

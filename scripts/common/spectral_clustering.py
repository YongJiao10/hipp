"""Spatially Constrained Spectral Clustering for hippocampal functional parcellation.

Pipeline (Steps 1-5):
  1. Pearson FC: hipp_timeseries x cortical_networks -> feature matrix X (N, 7)
  2. Functional affinity: cosine similarity of X, mapped to [0, 1]
  3. Spatial adjacency: KNN on 3D coords -> sparse binary symmetric matrix
  4. Graph fusion: Hadamard product of functional affinity and spatial adjacency
  5. Spectral clustering: symmetric normalized Laplacian eigsh + L2-normalize + KMeans
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

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


def _build_spatial_adjacency(coords: np.ndarray, k: int) -> sparse.csr_matrix:
    """Step 3: Binary symmetric KNN adjacency on 3D coordinates.

    Args:
        coords: Physical coordinates of shape (N, 3).
        k: Number of spatial nearest neighbors.

    Returns:
        Sparse binary symmetric CSR matrix of shape (N, N), diagonal=1.
    """
    n = coords.shape[0]
    k_eff = min(k, n - 1)
    nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm="ball_tree").fit(coords)
    _, indices = nbrs.kneighbors(coords)
    rows = np.repeat(np.arange(n), k_eff)
    cols = indices.reshape(-1)
    data = np.ones(len(rows), dtype=np.float32)
    W = sparse.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
    W = W.maximum(W.T)
    W.setdiag(1.0)
    W.eliminate_zeros()
    return W


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
    """Step 5: Symmetric normalized Laplacian spectral embedding + KMeans.

    Args:
        W: Sparse symmetric weight matrix (N, N).
        n_clusters: Number of clusters K.
        random_state: Random seed for KMeans.

    Returns:
        Integer cluster labels of shape (N,), 1-indexed and ordered by cluster size.
    """
    n = W.shape[0]
    degree = np.asarray(W.sum(axis=1)).ravel().astype(np.float64)
    inv_sqrt_deg = 1.0 / np.sqrt(np.clip(degree, 1e-12, None))
    D_inv_sqrt = sparse.diags(inv_sqrt_deg)

    # Symmetric normalized Laplacian: L_sym = I - D^{-1/2} W D^{-1/2}
    L_sym = sparse.eye(n, format="csr") - D_inv_sqrt @ W @ D_inv_sqrt

    # Request exactly n_clusters smallest eigenvectors.
    # For a graph with k well-separated components, there are k near-zero eigenvalues
    # (one indicator vector per component). Including all k — without skipping the
    # trivial constant — handles both connected and disconnected graphs correctly.
    # (For connected graphs the constant column is uniform and does not affect KMeans.)
    n_eigs = min(n_clusters, n - 1)
    if n_eigs < n_clusters:
        raise RuntimeError(
            f"Not enough vertices ({n}) to compute {n_clusters} spectral components"
        )
    eigvals, eigvecs = eigsh(L_sym.astype(np.float64), k=n_eigs, which="SM")
    order = np.argsort(eigvals)
    U = eigvecs[:, order].astype(np.float32)

    # L2-normalize each row
    row_norms = np.linalg.norm(U, axis=1, keepdims=True)
    U = U / np.clip(row_norms, 1e-12, None)

    raw_labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state).fit_predict(U)
    return _reorder_cluster_labels(raw_labels)


def spectral_cluster_from_features(
    features: np.ndarray,
    hipp_coords: np.ndarray,
    n_clusters: int,
    k_spatial: int = 10,
    random_state: int = 42,
) -> np.ndarray:
    """Steps 2-5: Spatially constrained spectral clustering from a pre-computed feature matrix.

    Drop-in alternative to cluster_embedding for use in evaluate_k_range.

    Args:
        features: Pre-computed feature matrix of shape (N, D), e.g. z-scored FC profiles.
        hipp_coords: 3D physical coordinates of shape (N, 3).
        n_clusters: Number of clusters.
        k_spatial: Number of KNN spatial neighbors.
        random_state: Random seed for KMeans.

    Returns:
        Integer labels of shape (N,), 1-indexed ordered by cluster size.
    """
    W_fc = _build_functional_affinity(features)
    W_spatial = _build_spatial_adjacency(hipp_coords, k_spatial)
    W_final = _fuse_graphs(W_fc, W_spatial)
    return _spectral_embed_and_cluster(W_final, n_clusters, random_state)


def spatially_constrained_spectral_clustering(
    hipp_timeseries: np.ndarray,
    cortical_networks: np.ndarray,
    hipp_coords: np.ndarray,
    n_clusters: int,
    k_spatial: int = 10,
    random_state: int = 42,
) -> np.ndarray:
    """End-to-end spatially constrained spectral clustering for hippocampal vertices.

    Args:
        hipp_timeseries: BOLD time series of valid hippocampal vertices, shape (N, T).
        cortical_networks: Mean time series of cortical functional networks, shape (7, T).
        hipp_coords: 3D physical coordinates of hippocampal vertices, shape (N, 3).
        n_clusters: Number of clusters.
        k_spatial: Number of KNN spatial neighbors for the spatial constraint (default 10).
        random_state: Random seed for KMeans reproducibility (default 42).

    Returns:
        Cluster labels of shape (N,), integer, 1-indexed and ordered by cluster size.
    """
    # Step 1: Pearson correlation features (N, 7)
    features = corrcoef_rows(hipp_timeseries, cortical_networks)
    return spectral_cluster_from_features(
        features, hipp_coords, n_clusters, k_spatial=k_spatial, random_state=random_state
    )

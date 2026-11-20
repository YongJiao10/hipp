"""Clustering of declared diffusion features; network annotation is independent."""
from __future__ import annotations

import warnings

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

from .surface_affinity import (
    GRADIENT_SPECTRAL_AFFINITY,
    GRADIENT_SPECTRAL_AFFINITY_DESCRIPTION,
    build_gradient_spectral_affinity,
)

MEAN_SIMILARITY_WARD_METHOD = "mean-similarity-ward"
SUBJECT_GRADIENT_WARD_METHOD = "subject-gradient-ward"
SUBJECT_GRADIENT_HBP_METHOD = "subject-gradient-hbp"
# Ward on diffusion gradients; the variants differ only in how the gradients
# are formed (group-mean FC, mean participant similarity, or the Procrustes
# mean of each participant's own gradients).
WARD_GRADIENT_METHODS = frozenset({"gradient-ward", MEAN_SIMILARITY_WARD_METHOD, SUBJECT_GRADIENT_WARD_METHOD})
GRADIENT_METHODS = frozenset({
    "gradient-spectral", "gradient-ward", "gradient-gmm", "gradient-spectral-rbf",
    MEAN_SIMILARITY_WARD_METHOD, SUBJECT_GRADIENT_WARD_METHOD, SUBJECT_GRADIENT_HBP_METHOD,
})
SPECTRAL_GRADIENT_METHODS = frozenset({"gradient-spectral", "gradient-spectral-rbf"})
RBF_AFFINITY_DESCRIPTION = "binary_mesh_edges_exp(-squared_gradient_distance/(2*sigma^2))"


def is_gradient_method(method: str) -> bool:
    return method in GRADIENT_METHODS


def is_spectral_gradient_method(method: str) -> bool:
    return method in SPECTRAL_GRADIENT_METHODS


def gradient_method_provenance(method: str, *, rbf_scale: float = 1.) -> dict:
    if method == "gradient-spectral":
        affinity, description = GRADIENT_SPECTRAL_AFFINITY, GRADIENT_SPECTRAL_AFFINITY_DESCRIPTION
    elif method == "gradient-spectral-rbf":
        return {"method": method, "surface_affinity": "mesh-rbf",
                "affinity_description": RBF_AFFINITY_DESCRIPTION, "rbf_scale": float(rbf_scale),
                "bandwidth_rule": "rbf_scale * median(positive unique-undirected-mesh-edge distance)"}
    elif method in WARD_GRADIENT_METHODS:
        affinity, description = "none", "ward_euclidean_binary_mesh_connectivity"
    elif method == "gradient-gmm":
        affinity, description = "none", "full_covariance_gaussian_mixture_no_spatial_constraint"
    elif method == SUBJECT_GRADIENT_HBP_METHOD:
        affinity, description = "none", "hbp_gaussian_mixture_on_aligned_subject_gradients_then_mesh_constrained_ward_on_group_prior"
    else:
        raise ValueError(f"Unknown gradient method: {method}")
    return {"method": method, "surface_affinity": affinity, "affinity_description": description}


def validate_gradient_method_provenance(method: str, manifests: list[dict]) -> dict:
    if not manifests:
        raise ValueError("Missing gradient method provenance")
    scales = {item.get("rbf_scale") for item in manifests}
    if method == "gradient-spectral-rbf" and (len(scales) != 1 or None in scales):
        raise ValueError("Inconsistent RBF bandwidth provenance")
    expected = gradient_method_provenance(method, rbf_scale=float(next(iter(scales))) if method == "gradient-spectral-rbf" else 1.)
    if any(any(item.get(key) != value for key, value in expected.items()) for item in manifests):
        raise ValueError(f"Inconsistent {method} method provenance")
    return expected


class GradientClusteringError(RuntimeError):
    def __init__(self, message: str, diagnostics: dict | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


def binary_mesh_adjacency(adjacency, n_vertices: int) -> sparse.csr_matrix:
    graph = sparse.csr_matrix(adjacency, dtype=np.float64, copy=True)
    if graph.shape != (n_vertices, n_vertices):
        raise ValueError(f"Adjacency/features mismatch: {graph.shape} vs {n_vertices}")
    if not np.all(np.isfinite(graph.data)) or np.any(graph.data < 0):
        raise ValueError("Mesh adjacency must have finite nonnegative entries")
    graph.eliminate_zeros()
    graph.data[:] = 1.
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0.)
    graph.eliminate_zeros()
    return graph


def _require_connected(graph, label: str, diagnostics=None):
    count = int(connected_components(graph, directed=False, return_labels=False))
    if count != 1:
        raise GradientClusteringError(f"{label} disconnected: {count} components", diagnostics)
    return count


def cluster_mesh_diagnostics(labels, adjacency, *, k: int) -> dict:
    labels = np.asarray(labels, dtype=np.int32)
    graph = binary_mesh_adjacency(adjacency, len(labels))
    counts, components, outside = {}, {}, {}
    for cluster in range(1, k + 1):
        idx = np.flatnonzero(labels == cluster)
        counts[str(cluster)] = int(idx.size)
        if not idx.size:
            raise GradientClusteringError(f"Empty returned cluster {cluster}")
        n, part = connected_components(graph[idx][:, idx], directed=False)
        components[str(cluster)] = int(n)
        outside[str(cluster)] = float(1 - np.max(np.bincount(part)) / idx.size)
    threshold = int(np.ceil(.03 * len(labels)))
    return {
        "cluster_vertex_counts": counts,
        "cluster_connected_components": components,
        "cluster_fraction_outside_largest_component": outside,
        "small_cluster_threshold": threshold,
        "small_clusters": [int(c) for c, n in counts.items() if n < threshold],
    }


def spatial_cluster_diagnostics(diagnostics: dict, model_labels, spatial_labels, adjacency, *, k: int) -> dict:
    model = np.asarray(model_labels, dtype=np.int32)
    spatial = np.asarray(spatial_labels, dtype=np.int32)
    result = dict(diagnostics)
    result["model_cluster_to_spatial_cluster"] = {
        str(int(c)): int(spatial[np.flatnonzero(model == c)[0]]) for c in np.unique(model)
    }
    result.update(cluster_mesh_diagnostics(spatial, adjacency, k=k))
    return result


def build_rbf_mesh_affinity(features, adjacency, *, rbf_scale: float = 1.) -> tuple[sparse.csr_matrix, dict]:
    x = np.asarray(features, dtype=np.float64)
    graph = binary_mesh_adjacency(adjacency, len(x))
    _require_connected(graph, "Supported mesh")
    if not np.isfinite(rbf_scale) or rbf_scale <= 0:
        raise ValueError("rbf_scale must be finite and positive")
    edges = sparse.triu(graph, k=1).tocoo()
    distance = np.linalg.norm(x[edges.row] - x[edges.col], axis=1)
    positive = distance[distance > 0]
    if not positive.size:
        raise GradientClusteringError("RBF bandwidth undefined: no positive mesh-edge distance")
    sigma = float(rbf_scale * np.median(positive))
    weights = np.exp(-.5 * np.square(distance / sigma))
    w = sparse.coo_matrix((weights, (edges.row, edges.col)), shape=graph.shape).tocsr()
    w = (w + w.T).tocsr()
    w.eliminate_zeros()
    degree = np.asarray(w.sum(axis=1)).reshape(-1)
    minimum, median = float(np.min(degree)), float(np.median(degree))
    effective = w.copy()
    if effective.nnz:
        effective.data[effective.data / np.max(effective.data) < 1e-5] = 0.
    effective.eliminate_zeros()
    meta = {
        "surface_affinity": "mesh-rbf", "affinity_description": RBF_AFFINITY_DESCRIPTION,
        "rbf_scale": float(rbf_scale), "sigma": sigma,
        "bandwidth_rule": "rbf_scale * median(positive unique-undirected-mesh-edge distance)",
        "unique_undirected_edges": int(distance.size), "affinity_dtype": "float64",
        "edge_weight_quantiles": dict(zip(
            ["min", "q01", "q05", "q25", "median", "q75", "q95", "q99", "max"],
            np.quantile(weights, [0, .01, .05, .25, .5, .75, .95, .99, 1]).tolist(),
        )),
        "weighted_degree_min": minimum, "weighted_degree_median": median,
        "weighted_degree_min_median_ratio": minimum / median if median else 0.,
        "positive_connected_components": int(connected_components(w, directed=False, return_labels=False)),
        "effective_connected_components": int(connected_components(effective, directed=False, return_labels=False)),
        "effective_relative_weight_cutoff": 1e-5,
    }
    _require_connected(w, "Positive RBF support", meta)
    _require_connected(effective, "RBF effective support", meta)
    return w, meta


def cluster_gradients(method: str, features, adjacency, *, k: int, seed: int, rbf_scale: float = 1.) -> tuple[np.ndarray, dict]:
    if method not in GRADIENT_METHODS:
        raise ValueError(f"Unknown gradient method: {method}")
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or not x.shape[1] or not np.all(np.isfinite(x)):
        raise ValueError("Gradient clustering requires finite vertices-by-components features")
    if k < 2 or k >= len(x):
        raise ValueError(f"Require 2 <= k < supported vertices, got {k} / {len(x)}")
    graph = binary_mesh_adjacency(adjacency, len(x))
    meta = {"method": method, "k": int(k)}
    if method in WARD_GRADIENT_METHODS:
        _require_connected(graph, "Supported mesh")
        model = AgglomerativeClustering(n_clusters=k, linkage="ward", metric="euclidean", connectivity=graph)
        labels = model.fit_predict(x) + 1
        meta.update(linkage="ward", metric="euclidean", spatial_connectivity="binary_mesh")
    elif method == "gradient-gmm":
        meta["seed"] = int(seed)
        model = GaussianMixture(n_components=k, covariance_type="full", n_init=10,
                                max_iter=1000, tol=1e-3, reg_covar=1e-6, random_state=seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(x)
        if not model.converged_ or any(issubclass(w.category, ConvergenceWarning) for w in caught):
            raise GradientClusteringError("GMM failed to converge")
        labels = model.predict(x) + 1
        meta.update(converged=True, n_iter=int(model.n_iter_), lower_bound=float(model.lower_bound_),
                    covariance_type="full", n_init=10, max_iter=1000, tol=1e-3, reg_covar=1e-6,
                    covariance_condition_numbers=[float(np.linalg.cond(c)) for c in model.covariances_],
                    regularizer_fraction_of_covariance_eigenvalues=[
                        (1e-6 / np.linalg.eigvalsh(c)).tolist() for c in model.covariances_],
                    spatial_connectivity=None)
    else:
        if method == "gradient-spectral-rbf":
            affinity, affinity_meta = build_rbf_mesh_affinity(x, graph, rbf_scale=rbf_scale)
        else:
            # Retain the established shifted-cosine route, including its existing mesh weights.
            affinity = build_gradient_spectral_affinity(features, adjacency)
            affinity_meta = {"surface_affinity": GRADIENT_SPECTRAL_AFFINITY,
                             "affinity_description": GRADIENT_SPECTRAL_AFFINITY_DESCRIPTION}
        meta["seed"] = int(seed)
        labels = SpectralClustering(n_clusters=k, affinity="precomputed", random_state=seed,
                                   assign_labels="kmeans", n_init=50, eigen_solver="arpack").fit_predict(affinity) + 1
        meta.update(affinity_meta)
        meta.update(assign_labels="kmeans", n_init=50, eigen_solver="arpack")
    labels = np.asarray(labels, dtype=np.int32)
    meta.update(cluster_mesh_diagnostics(labels, graph, k=k))
    if method in WARD_GRADIENT_METHODS and any(n != 1 for n in meta["cluster_connected_components"].values()):
        raise GradientClusteringError("Ward returned a disconnected cluster", meta)
    return labels, meta

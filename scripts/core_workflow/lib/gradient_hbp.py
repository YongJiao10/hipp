"""Hierarchical Bayesian parcellation of aligned individual gradients with a connected group map.

The data are each participant's own diffusion gradients after generalized Procrustes alignment
(``subject-gradient-ward`` features before averaging): participants x vertices x components, NaN where
a participant does not observe a vertex. HierarchBayesParcel fits a vertex-specific independent
arrangement (the group prior) with an isotropic Gaussian mixture emission whose centres are not
normalized. The group partition is mesh-constrained Ward on the per-vertex group prior
probabilities, so every group parcel is one connected piece; the posterior argmax gives the
individual maps. The independent arrangement has no smoothing parameter; a Potts coupling was
tested and rejected because a learned prior plus coupling becomes confident without data structure.
"""
from __future__ import annotations

import subprocess
import sys
import threading
import warnings
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components

from .gradient_clustering import cluster_gradients

N_INITS = 10
FIRST_ITER = 30
MAX_ITER = 200
TOLERANCE = 0.01
ALGORITHM_DESCRIPTION = (
    "hierarchbayesparcel_independent_vertex_prior_isotropic_gaussian_mixture_unnormalized_centres_"
    "10_inits_then_mesh_constrained_ward_on_group_prior"
)
# HierarchBayesParcel draws from the global torch generator; fits are serialized so a seed
# fully determines each fit when K candidates run as threads.
_FIT_LOCK = threading.Lock()


def hbp_revision(hbp_root: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(hbp_root), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _import(hbp_root: Path):
    root = Path(hbp_root).resolve(strict=True)
    if not (root / "HierarchBayesParcel" / "full_model.py").is_file():
        raise FileNotFoundError(f"Not a HierarchBayesParcel checkout: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import torch
    from HierarchBayesParcel.arrangements import ArrangeIndependent
    from HierarchBayesParcel.emissions import MixGaussian
    from HierarchBayesParcel.full_model import FullMultiModel
    return torch, ArrangeIndependent, MixGaussian, FullMultiModel


def _components(labels: np.ndarray, adjacency) -> list[int]:
    return [int(connected_components(adjacency[np.flatnonzero(labels == k)][:, np.flatnonzero(labels == k)],
                                     directed=False)[0]) for k in np.unique(labels)]


def fit_hbp_connected(stack: np.ndarray, adjacency, *, k: int, seed: int, hbp_root: Path) -> tuple[np.ndarray, dict]:
    """Return labels 1..K (connected group parcels) and fit diagnostics."""
    torch, ArrangeIndependent, MixGaussian, FullMultiModel = _import(hbp_root)
    data = np.asarray(stack, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected participants x vertices x components, got {data.shape}")
    n, p, dim = data.shape
    observed = np.isfinite(data).all(axis=2)
    if not observed.any(axis=0).all():
        raise ValueError("A supported vertex is unobserved in every participant")
    with _FIT_LOCK:
        torch.set_num_threads(1)
        torch.manual_seed(int(seed))
        model = FullMultiModel(ArrangeIndependent(int(k), p, spatial_specific=True),
                               [MixGaussian(K=int(k), N=dim, P=p, std_V=False)])
        model.initialize([np.ascontiguousarray(data.transpose(0, 2, 1))])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model, objective, _, posterior, _ = model.fit_em_ninits(
                n_inits=N_INITS, first_iter=FIRST_ITER, iter=MAX_ITER, tol=TOLERANCE, align="arrange", verbose=False)
        prior = model.arrange.marginal_prob().detach().cpu().numpy().T
        posterior = posterior.detach().cpu().numpy().transpose(0, 2, 1)
        sigma2 = float(model.emissions[0].sigma2)
        objective = objective.detach().cpu().numpy()
    if not (np.isfinite(prior).all() and np.isfinite(posterior).all() and np.isfinite(sigma2)):
        raise ValueError("HBP fit produced non-finite parameters")
    labels, ward_meta = cluster_gradients("gradient-ward", prior, adjacency, k=int(k), seed=int(seed))
    argmax = prior.argmax(axis=1) + 1
    individual = posterior.argmax(axis=2) + 1
    entropy = -(prior * np.log(np.maximum(prior, 1e-12))).sum(axis=1) / np.log(int(k))
    from sklearn.metrics import adjusted_rand_score
    diagnostics = {
        "hbp_algorithm": ALGORITHM_DESCRIPTION, "hbp_n_inits": N_INITS, "hbp_max_iter": MAX_ITER,
        "hbp_tolerance": TOLERANCE, "hbp_seed": int(seed), "hbp_sigma2": sigma2,
        "hbp_final_objective": float(objective[-1]), "hbp_objective_length": int(objective.size),
        "hbp_warnings": sorted({str(w.message) for w in caught})[:5],
        "hbp_prior_normalized_entropy_mean": float(entropy.mean()),
        "hbp_prior_argmax_sizes": np.bincount(argmax, minlength=int(k) + 1)[1:].tolist(),
        "hbp_prior_argmax_components": _components(argmax, adjacency),
        "hbp_ari_prior_argmax_vs_connected": float(adjusted_rand_score(argmax, labels)),
        "hbp_individual_vs_group_ari_mean": float(np.mean(
            [adjusted_rand_score(individual[i][observed[i]], labels[observed[i]]) for i in range(n)])),
        "hbp_individual_max_components_mean": float(np.mean(
            [max(_components(individual[i][observed[i]], adjacency[observed[i]][:, observed[i]])) for i in range(n)])),
        "connected_group_ward": {key: ward_meta[key] for key in ("linkage", "metric", "spatial_connectivity") if key in ward_meta},
    }
    return labels.astype(np.int32), diagnostics

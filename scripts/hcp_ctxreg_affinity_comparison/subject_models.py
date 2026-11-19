"""Subject-level spectral and directional parcellation for the HCP comparison."""
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import helmert

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core_workflow"))

from lib.multigraph_spectral import (  # noqa: E402,F401  Shared with the group-atlas methods.
    MAX_ITER, N_STARTS, SEED, Fit, FitError, hippocampal_affinity, mean_graph_coordinates,
    shen_discretize, spectral_coordinates, subject_graphs,
)


SCREEN_ITER = 30
EM_TOLERANCE = 0.01
HBP_REVISION = "d4031910ebd16e9013d756eef4d057ba7106d64c"


def unit_directions(fisher_z, observed, *, centered):
    """Keep unobserved vertices all NaN and unit-normalize valid fingerprints."""
    profiles = fisher_z[observed].astype(np.float64)
    if centered:
        if np.any(np.ptp(profiles, axis=1) == 0):
            row = np.flatnonzero(np.ptp(profiles, axis=1) == 0)[0]
            raise FitError(f"Constant Fisher-z fingerprint at subject/vertex {np.argwhere(observed)[row].tolist()}")
        profiles -= profiles.mean(axis=1, keepdims=True)
        profiles = profiles @ helmert(profiles.shape[1], full=False).T
    norms = np.linalg.norm(profiles, axis=1)
    bad = ~np.isfinite(norms) | (norms == 0)
    if np.any(bad):
        row = np.flatnonzero(bad)[0]
        raise FitError(f"Invalid directional norm at subject/vertex {np.argwhere(observed)[row].tolist()}")
    result = np.full((*observed.shape, profiles.shape[1]), np.nan, dtype=np.float32)
    result[observed] = profiles / norms[:, None]
    return result


def fit_directional(directions, observed, *, k, spatial, mean_only, save_individual=False):
    """Fit the pinned upstream independent arrangement with one directional emission."""
    import warnings

    import torch
    from HierarchBayesParcel.arrangements import ArrangeIndependent
    from HierarchBayesParcel.emissions import MixVMF
    from HierarchBayesParcel.full_model import FullMultiModel

    if mean_only:
        mean = np.nansum(directions, axis=0, dtype=np.float64)
        norm = np.linalg.norm(mean, axis=1)
        if np.any(norm == 0):
            raise FitError("A supported vertex has an undefined group-mean direction")
        directions = (mean / norm[:, None]).astype(np.float32)[None, ...]
        observed = np.ones(directions.shape[:2], dtype=bool)

    n_people, n_vertices, dim = directions.shape
    torch.manual_seed(SEED)
    arrangement = ArrangeIndependent(k, n_vertices, spatial_specific=spatial)
    emission = MixVMF(K=k, N=dim, P=n_vertices)
    model = FullMultiModel(arrangement, [emission])
    model.initialize([np.ascontiguousarray(directions.transpose(0, 2, 1))])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model, objective, _, posterior, screening = model.fit_em_ninits(
            n_inits=N_STARTS, first_iter=SCREEN_ITER, iter=MAX_ITER, tol=EM_TOLERANCE,
            align="arrange" if spatial else 0, verbose=False,
        )
    emission = model.emissions[0]
    parcel_directions = emission.V.detach().cpu().numpy()
    prior = model.arrange.marginal_prob().detach().cpu().numpy().T
    posterior = posterior.detach().cpu().numpy().transpose(0, 2, 1)
    objective = objective.cpu()
    screening = screening.cpu()
    kappa = float(emission.kappa)
    if not (np.isfinite(kappa) and np.isfinite(parcel_directions).all()
            and np.isfinite(objective.numpy()).all() and np.isfinite(prior).all()
            and np.isfinite(posterior).all() and np.allclose(posterior.sum(axis=2), 1.0)):
        raise FitError("Bayesian fit produced nonfinite parameters or unnormalized posteriors")
    if not np.allclose(posterior[~observed], np.broadcast_to(prior, posterior.shape)[~observed]):
        raise FitError("Unobserved vertices did not give the expected prior-only posterior")
    weighted = np.einsum(
        "svp,svk->pk", np.nan_to_num(directions), posterior, optimize=True,
    )
    masses = posterior[observed].sum(axis=0)
    if np.any(masses <= 0):
        raise FitError("At least one directional mixture component has zero posterior mass")
    resultant = np.sum(weighted * parcel_directions, axis=0) / masses
    group_probability = (
        prior if spatial else
        np.where(observed[..., None], posterior, 0).sum(axis=0) / observed.sum(axis=0)[:, None]
    )
    group = group_probability.argmax(axis=1) + 1
    if np.unique(group).size != k:
        raise FitError(f"Directional solution has {np.unique(group).size} hard parcels instead of {k}")

    warnings_text = sorted({str(item.message) for item in caught})
    screen_lengths = np.isfinite(screening.numpy()).sum(axis=1)
    screen_scores = screening.numpy()[np.arange(N_STARTS), np.maximum(screen_lengths - 1, 0)]
    selected_length = int(screen_lengths[np.nanargmax(screen_scores)])
    if warnings_text and objective.numel() > 1 and float(objective[-1] - objective[-2]) < -EM_TOLERANCE:
        termination = "upstream_objective_decrease"
    elif objective.numel() - selected_length + 1 == MAX_ITER - SCREEN_ITER:
        termination = "iteration_limit"
    else:
        termination = "upstream_tolerance"
    diagnostics = {
        "feature_dimension": dim, "n_input_subjects": n_people, "kappa": kappa,
        "mean_resultant_length": float(resultant.mean()), "training_objective": float(objective[-1]),
        "objective_trace_length": objective.numel(), "termination": termination, "warnings": warnings_text,
    }
    if spatial:
        valid = posterior[observed]
        entropy = -(valid * np.log(np.maximum(valid, np.finfo(valid.dtype).tiny))).sum(axis=1) / np.log(k)
        data_only = (directions[observed] @ parcel_directions).argmax(axis=1)
        diagnostics.update(
            entropy_quantiles=np.quantile(entropy, [0, .25, .5, .75, 1]).tolist(),
            posterior_above_099=float((valid.max(axis=1) > .99).mean()),
            location_prior_changes_winner=float((data_only != valid.argmax(axis=1)).mean()),
        )
    return Fit(
        group, diagnostics,
        np.where(observed, posterior.argmax(axis=2) + 1, 0) if save_individual else None,
        group_probability,
        np.where(observed[..., None], posterior, np.nan) if save_individual else None,
        parcel_directions,
    )

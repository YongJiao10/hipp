"""Subject-level spectral and directional parcellation for the HCP comparison."""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh, helmert
from scipy.sparse.csgraph import connected_components


SEED = 20260420
N_STARTS = 10
MAX_ITER = 200
SCREEN_ITER = 30
EM_TOLERANCE = 0.01
HBP_REVISION = "d4031910ebd16e9013d756eef4d057ba7106d64c"


class FitError(ValueError):
    """A model cannot produce the prespecified fit on this condition and split."""


@dataclass
class Fit:
    """Labels use 1-based parcels and 0 outside the observed mesh."""

    labels: np.ndarray
    diagnostics: dict
    individual: np.ndarray | None = None
    probability: np.ndarray | None = None
    posterior: np.ndarray | None = None
    directions: np.ndarray | None = None


def hippocampal_affinity(similarity):
    """Keep positive weights among the closest 10% of other vertices."""
    n = len(similarity)
    if n < 2 or not np.isfinite(similarity).all():
        raise FitError("Undefined Pearson edge or fewer than two observed vertices")
    weights = similarity.copy()
    np.fill_diagonal(weights, -np.inf)
    count = int(np.ceil(0.10 * (n - 1)))
    neighbors = np.argsort(-weights, axis=1, kind="stable")[:, :count]
    rows = np.arange(n)[:, None]
    directed = np.zeros_like(weights)
    directed[rows, neighbors] = np.maximum(weights[rows, neighbors], 0)
    return (directed + directed.T) * 0.5


def spectral_coordinates(affinity, vertices, k):
    if len(vertices) < k:
        raise FitError(f"Only {len(vertices)} graph vertices for K={k}")
    degree = affinity.sum(axis=1, dtype=np.float64)
    isolates = vertices[degree <= 0]
    n_components, labels = connected_components(affinity, directed=False)
    if n_components != 1 or isolates.size:
        raise FitError(
            f"Disconnected affinity on {len(vertices)} vertices, "
            f"component sizes={np.bincount(labels).tolist()}, zero-degree vertices={isolates.tolist()}"
        )
    scale = degree**-0.5
    normalized = affinity * scale[:, None] * scale[None, :]
    n = len(vertices)
    _, rows = eigh(normalized, subset_by_index=(n - k, n - 1), check_finite=False)
    rows = rows[:, ::-1]
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def subject_graphs(similarity, observed, support, subjects, k):
    """Build each subject's graph on the common training support."""
    affinities, coordinates = [], []
    for matrix, mask, subject in zip(similarity, observed, subjects, strict=True):
        vertices = np.flatnonzero(mask & support)
        try:
            weights = hippocampal_affinity(matrix[np.ix_(vertices, vertices)])
            coords = spectral_coordinates(weights, vertices, k)
        except FitError as exc:
            raise FitError(f"Subject {subject}: {exc}") from exc
        affinities.append((vertices, weights))
        coordinates.append((vertices, coords))
    return affinities, coordinates


def mean_graph_coordinates(graphs, support, k):
    total = np.zeros((support.size, support.size), dtype=np.float64)
    count = np.zeros_like(total, dtype=np.int32)
    for vertices, affinity in graphs:
        ids = np.ix_(vertices, vertices)
        total[ids] += affinity
        count[ids] += 1
    active = np.flatnonzero(support)
    ids = np.ix_(active, active)
    if not np.all(count[ids]):
        raise FitError("A supported vertex pair has no observed contributors")
    return active, spectral_coordinates(total[ids] / count[ids], active, k)


def shen_discretize(embeddings, support, *, k, seed=SEED, n_starts=N_STARTS, max_iter=MAX_ITER):
    """Choose a shared discrete partition and a rotation for each embedding."""
    # Pad missing rows with zero so they add neither to the Procrustes cross
    # product nor to the sum of aligned scores. The loss masks them separately.
    n_subjects = len(embeddings)
    coordinates = np.zeros((n_subjects, support.size, k), dtype=np.float64)
    observed = np.zeros((n_subjects, support.size), dtype=bool)
    for i, (vertices, rows) in enumerate(embeddings):
        coordinates[i, vertices] = rows
        observed[i, vertices] = True
    if not np.array_equal(observed.any(axis=0), support):
        raise FitError("The training support differs from the union of embedded vertices")
    identity = np.eye(k)
    rng = np.random.default_rng(seed)
    failed = 0
    best = None
    for _ in range(n_starts):
        # A single subject's spectral rotation seeds the shared labels.
        pick = rng.integers(n_subjects)
        rotation, _ = np.linalg.qr(rng.standard_normal((k, k)))
        labels = (coordinates[pick] @ rotation).argmax(axis=1)
        missing = support & ~observed[pick]
        labels[missing] = rng.integers(k, size=missing.sum())
        for iteration in range(1, max_iter + 1):
            target = identity[labels]
            left, _, right = np.linalg.svd(coordinates.transpose(0, 2, 1) @ target)
            aligned = coordinates @ (left @ right)
            update = aligned.sum(axis=0).argmax(axis=1)
            if np.array_equal(labels[support], update[support]):
                break
            labels = update
        else:
            failed += 1
            continue
        if np.unique(labels[support]).size != k:
            failed += 1
            continue
        loss = float(np.square(aligned[observed] - np.broadcast_to(target, aligned.shape)[observed]).sum())
        if best is None or loss < best.diagnostics["loss"]:
            best = Fit(
                np.where(support, labels + 1, 0),
                {"loss": loss, "iterations": iteration},
                np.where(observed, aligned.argmax(axis=2) + 1, 0),
            )
    if best is None:
        raise FitError(f"All {n_starts} spectral starts failed to reach a nonempty stable partition")
    best.diagnostics["unusable_starts"] = failed
    return best


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

"""Multigraph and mean-graph spectral parcellation of hippocampal FC fingerprints.

``multigraph-spectral`` is the groupwise K-way clustering of Shen et al. (2013):
every participant keeps an own functional graph and spectral embedding, and one
shared discrete partition is fitted together with one rotation per participant.
``mean-graph-spectral`` is its matched single-graph comparator: the participant
affinities are averaged edgewise first and the same discretization runs once.

Neither method uses the surface mesh. A requested K that the data do not
support shows as an empty or very small parcel in the lowest-loss solution;
the caller's K-selection gates decide whether that K is usable.
"""
from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components

MULTIGRAPH_METHOD = "multigraph-spectral"
MEAN_GRAPH_METHOD = "mean-graph-spectral"
# Shen discretization on each participant's diffusion-map coordinates instead of raw eigenvectors.
MULTIGRAPH_GRADIENT_METHOD = "multigraph-gradient-spectral"
MULTIGRAPH_METHODS = frozenset({MULTIGRAPH_METHOD, MEAN_GRAPH_METHOD, MULTIGRAPH_GRADIENT_METHOD})
NEIGHBOUR_FRACTION = 0.10
MAX_COORDINATES = 8
SEED = 20260420
N_STARTS = 10
MAX_ITER = 200
AFFINITY_DESCRIPTION = (
    "positive_pearson_of_fisherz_roi_fingerprints_top10pct_row_neighbours_mean_symmetrized_diag0_"
    "no_spatial_constraint"
)


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


def vertex_similarity(fisher_z: np.ndarray) -> np.ndarray:
    """Pearson correlation between vertex fingerprints over the cortical ROIs.

    A vertex is either wholly observed or wholly missing (all NaN). Missing
    vertices keep NaN rows and columns; observed vertices have 1 on the diagonal.
    """
    profiles = np.asarray(fisher_z, dtype=np.float64)
    if profiles.ndim != 2:
        raise ValueError(f"Expected a vertex x ROI matrix, got shape={profiles.shape}")
    observed = np.isfinite(profiles).any(axis=1)
    rows = profiles[observed]
    rows = rows[:, np.isfinite(rows).any(axis=0)]
    if rows.shape[0] < 2 or rows.shape[1] < 2:
        raise FitError(f"Too few observed vertices or ROIs for a similarity matrix: {rows.shape}")
    if not np.isfinite(rows).all():
        vertex = int(np.flatnonzero(observed)[np.flatnonzero(~np.isfinite(rows).all(axis=1))[0]])
        raise FitError(f"Partially observed FC fingerprint at vertex {vertex}")
    constant = np.ptp(rows, axis=1) == 0
    if np.any(constant):
        vertex = int(np.flatnonzero(observed)[np.flatnonzero(constant)[0]])
        raise FitError(f"Constant FC fingerprint at vertex {vertex}")
    rows = rows - rows.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(rows, axis=1)
    unit = rows / norm[:, None]
    out = np.full((profiles.shape[0], profiles.shape[0]), np.nan, dtype=np.float32)
    out[np.ix_(observed, observed)] = np.clip(unit @ unit.T, -1.0, 1.0)
    out[observed, observed] = 1.0
    return out


def observed_vertices(similarity: np.ndarray) -> np.ndarray:
    """Vertices a similarity matrix from ``vertex_similarity`` declares observed."""
    return np.isfinite(np.diagonal(similarity))


def hippocampal_affinity(similarity: np.ndarray) -> np.ndarray:
    """Keep positive weights among the closest 10% of other vertices."""
    n = len(similarity)
    if n < 2 or not np.isfinite(similarity).all():
        raise FitError("Undefined Pearson edge or fewer than two observed vertices")
    weights = np.array(similarity, dtype=np.float64)
    np.fill_diagonal(weights, -np.inf)
    count = int(np.ceil(NEIGHBOUR_FRACTION * (n - 1)))
    neighbors = np.argsort(-weights, axis=1, kind="stable")[:, :count]
    rows = np.arange(n)[:, None]
    directed = np.zeros_like(weights)
    directed[rows, neighbors] = np.maximum(weights[rows, neighbors], 0)
    return (directed + directed.T) * 0.5


def leading_eigenvectors(affinity: np.ndarray, vertices: np.ndarray, n_components: int) -> np.ndarray:
    """Leading eigenvectors of the degree-normalized affinity, largest eigenvalue first."""
    if len(vertices) < n_components:
        raise FitError(f"Only {len(vertices)} graph vertices for {n_components} spectral coordinates")
    degree = affinity.sum(axis=1, dtype=np.float64)
    isolates = np.asarray(vertices)[degree <= 0]
    n_graph_components, labels = connected_components(affinity, directed=False)
    if n_graph_components != 1 or isolates.size:
        raise FitError(
            f"Disconnected affinity on {len(vertices)} vertices, "
            f"component sizes={np.bincount(labels).tolist()}, zero-degree vertices={isolates.tolist()}"
        )
    scale = degree**-0.5
    normalized = affinity * scale[:, None] * scale[None, :]
    n = len(vertices)
    _, vectors = eigh(normalized, subset_by_index=(n - n_components, n - 1), check_finite=False)
    return vectors[:, ::-1]


def diffusion_coordinates(affinity: np.ndarray, vertices: np.ndarray, n_components: int) -> np.ndarray:
    """A constant column followed by G1..G(n-1) of the exact diffusion map (alpha=0.5, eigenvalue-scaled).

    The constant column replaces the trivial eigenvector, so the first K columns
    give K coordinates for a K-way discretization, as the leading eigenvectors do.
    """
    from .hippomaps_adapter import diffusion_map_embedding

    if len(vertices) <= n_components:
        raise FitError(f"Only {len(vertices)} graph vertices for {n_components} diffusion coordinates")
    try:
        gradients, _ = diffusion_map_embedding(affinity, n_components=n_components - 1)
    except ValueError as exc:
        raise FitError(f"Diffusion map failed: {exc}") from exc
    return np.column_stack([np.ones(len(vertices)), gradients])


def spectral_coordinates(affinity: np.ndarray, vertices: np.ndarray, k: int) -> np.ndarray:
    """Unit-length rows of the K leading eigenvectors."""
    rows = leading_eigenvectors(affinity, vertices, k)
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


def mean_affinity(graphs, support: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Edgewise mean over the participants who observed both endpoints."""
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
    return active, total[ids] / count[ids]


def mean_graph_coordinates(graphs, support, k):
    active, affinity = mean_affinity(graphs, support)
    return active, spectral_coordinates(affinity, active, k)


def shen_discretize(embeddings, support, *, k, seed=SEED, n_starts=N_STARTS, max_iter=MAX_ITER,
                    require_occupied=True):
    """Choose a shared discrete partition and a rotation for each embedding.

    With ``require_occupied`` only converged starts that occupy all K parcels
    compete. Without it the lowest-loss converged start is returned as fitted,
    so an unsupported K is visible as an empty parcel.
    """
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
        occupied = int(np.unique(labels[support]).size)
        if require_occupied and occupied != k:
            failed += 1
            continue
        loss = float(np.square(aligned[observed] - np.broadcast_to(target, aligned.shape)[observed]).sum())
        if best is None or loss < best.diagnostics["loss"]:
            best = Fit(
                np.where(support, labels + 1, 0),
                {"loss": loss, "iterations": iteration, "occupied_parcels": occupied},
                np.where(observed, aligned.argmax(axis=2) + 1, 0),
            )
    if best is None:
        raise FitError(f"All {n_starts} spectral starts failed to reach a "
                       f"{'nonempty ' if require_occupied else ''}stable partition")
    best.diagnostics["unusable_starts"] = failed
    return best


# ── K-independent coordinate stacks for the group-atlas K scan ─────────────

_CACHE_LOCK = threading.Lock()
_SUBJECT_CACHE: OrderedDict[tuple, tuple[np.ndarray, np.ndarray]] = OrderedDict()
_SUBJECT_CACHE_LIMIT = 4096
_MEAN_GRAPH_CACHE: OrderedDict[tuple, np.ndarray] = OrderedDict()
_MEAN_GRAPH_CACHE_LIMIT = 512


def _digest(array: np.ndarray) -> bytes:
    return hashlib.blake2b(np.ascontiguousarray(array).view(np.uint8), digest_size=16).digest()


def _subject_eigenvectors(similarity: np.ndarray, support: np.ndarray, n_components: int,
                          method: str = MULTIGRAPH_METHOD):
    """One participant's observed vertices on the support and their spectral or diffusion coordinates."""
    key = (method, _digest(similarity), support.tobytes(), int(n_components))
    with _CACHE_LOCK:
        hit = _SUBJECT_CACHE.get(key)
        if hit is not None:
            _SUBJECT_CACHE.move_to_end(key)
            return hit
    vertices = np.flatnonzero(observed_vertices(similarity) & support)
    affinity = hippocampal_affinity(similarity[np.ix_(vertices, vertices)])
    coordinates = diffusion_coordinates if method == MULTIGRAPH_GRADIENT_METHOD else leading_eigenvectors
    value = (vertices, coordinates(affinity, vertices, n_components))
    with _CACHE_LOCK:
        _SUBJECT_CACHE[key] = value
        while len(_SUBJECT_CACHE) > _SUBJECT_CACHE_LIMIT:
            _SUBJECT_CACHE.popitem(last=False)
    return value


def coordinate_stack(similarities, subjects, support: np.ndarray, *, method: str,
                     n_components: int = MAX_COORDINATES) -> np.ndarray:
    """Leading eigenvectors on the supported vertices, one layer per fitted graph.

    ``multigraph-spectral`` returns one layer per participant with NaN rows at
    that participant's unobserved vertices. ``mean-graph-spectral`` returns the
    single layer of the edgewise mean affinity. Shape: graphs x supported
    vertices x ``n_components``.
    """
    if method not in MULTIGRAPH_METHODS:
        raise ValueError(f"Not a multigraph method: {method}")
    support = np.asarray(support, dtype=bool)
    active = np.flatnonzero(support)
    position = np.full(support.size, -1, dtype=np.int64)
    position[active] = np.arange(active.size)
    n_components = min(int(n_components), active.size - 1)
    if method == MEAN_GRAPH_METHOD:
        key = (tuple(_digest(np.asarray(similarity)) for similarity in similarities), support.tobytes(), n_components)
        with _CACHE_LOCK:
            hit = _MEAN_GRAPH_CACHE.get(key)
        if hit is not None:
            return hit
        graphs = []
        for similarity, subject in zip(similarities, subjects, strict=True):
            vertices = np.flatnonzero(observed_vertices(similarity) & support)
            try:
                graphs.append((vertices, hippocampal_affinity(similarity[np.ix_(vertices, vertices)])))
            except FitError as exc:
                raise FitError(f"Subject {subject}: {exc}") from exc
        _, affinity = mean_affinity(graphs, support)
        value = leading_eigenvectors(affinity, active, n_components)[None].astype(np.float32)
        with _CACHE_LOCK:
            _MEAN_GRAPH_CACHE[key] = value
            while len(_MEAN_GRAPH_CACHE) > _MEAN_GRAPH_CACHE_LIMIT:
                _MEAN_GRAPH_CACHE.popitem(last=False)
        return value
    stack = np.full((len(subjects), active.size, n_components), np.nan, dtype=np.float32)
    for index, (similarity, subject) in enumerate(zip(similarities, subjects, strict=True)):
        try:
            vertices, vectors = _subject_eigenvectors(np.asarray(similarity), support, n_components, method)
        except FitError as exc:
            raise FitError(f"Subject {subject}: {exc}") from exc
        stack[index, position[vertices]] = vectors
    return stack


def _unit_rows(stack: np.ndarray, k: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if stack.ndim != 3 or stack.shape[2] < k:
        raise FitError(f"Coordinate stack {stack.shape} does not hold {k} spectral coordinates")
    embeddings = []
    for layer in np.asarray(stack, dtype=np.float64):
        vertices = np.flatnonzero(np.isfinite(layer[:, 0]))
        rows = layer[vertices, :k]
        norm = np.linalg.norm(rows, axis=1, keepdims=True)
        if np.any(norm == 0):
            raise FitError("A vertex has a zero spectral coordinate row")
        embeddings.append((vertices, rows / norm))
    return embeddings


def discretize_stack(stack: np.ndarray, k: int, *, seed: int = SEED) -> tuple[np.ndarray, dict]:
    """Lowest-loss shared partition of a coordinate stack; labels 1..K on the supported vertices."""
    embeddings = _unit_rows(stack, int(k))
    support = np.ones(stack.shape[1], dtype=bool)
    covered = np.zeros(stack.shape[1], dtype=bool)
    for vertices, _ in embeddings:
        covered[vertices] = True
    if not covered.all():
        raise FitError("A supported vertex is unobserved in every fitted graph")
    fit = shen_discretize(embeddings, support, k=int(k), seed=int(seed), require_occupied=False)
    return fit.labels.astype(np.int32), {**fit.diagnostics, "n_graphs": int(stack.shape[0]), "n_starts": N_STARTS}


def stack_coherence(stack: np.ndarray, labels: np.ndarray, k: int) -> tuple[float, float]:
    """Mean cosine between a vertex's K coordinates and its parcel centroid, averaged over graphs.

    Cosines are taken inside one graph's own embedding, so no rotation between
    participants is needed. An empty parcel scores 0 and a single vertex 1.
    """
    per_parcel = np.zeros(int(k), dtype=np.float64)
    for parcel in range(1, int(k) + 1):
        values = []
        for vertices, rows in _unit_rows(stack, int(k)):
            member = rows[labels[vertices] == parcel]
            if len(member) == 1:
                values.append(1.0)
            elif len(member) > 1:
                centre = member.mean(axis=0)
                norm = np.linalg.norm(centre)
                values.append(float(np.mean(member @ centre / norm)) if norm > 0 else 0.0)
        per_parcel[parcel - 1] = float(np.mean(values)) if values else 0.0
    return float(per_parcel.min()), float(per_parcel.mean())

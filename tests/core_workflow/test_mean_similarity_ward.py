"""Mean-similarity Ward: averaged participant vertex similarity, TopSim affinity, diffusion gradients, Ward."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from brainspace.gradient.kernels import compute_affinity
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/core_workflow"))

from lib import gradient_clustering as gc  # noqa: E402
from lib.group_atlas_lineage import gradient_computation_provenance  # noqa: E402
from lib.hippomaps_adapter import (  # noqa: E402
    build_mean_similarity_affinity,
    diffusion_map_embedding,
    fit_mean_similarity_gradients,
)
from lib.multigraph_spectral import vertex_similarity  # noqa: E402


def stage(name):
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), ROOT / "scripts/core_workflow" / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def chain_adjacency(n):
    rows = np.arange(n - 1)
    graph = sparse.coo_matrix((np.ones(n - 1), (rows, rows + 1)), shape=(n, n))
    return (graph + graph.T).tocsr()


def subject_fingerprints(n_subjects=6, n_vertices=100, n_rois=40, seed=3):
    """Two smooth patterns along a chain so the long axis is the leading gradient."""
    rng = np.random.default_rng(seed)
    axis = np.linspace(-1, 1, n_vertices)[:, None]
    patterns = rng.normal(size=(2, n_rois))
    out = []
    for _ in range(n_subjects):
        fc = axis * patterns[0] + (1 - axis**2) * patterns[1] + 0.3 * rng.normal(size=(n_vertices, n_rois))
        out.append(fc.astype(np.float32))
    return out


def test_affinity_matches_pairwise_mean_then_topsim():
    fps = subject_fingerprints()
    sims = np.stack([vertex_similarity(fc) for fc in fps])
    sims[0, :4, :] = np.nan  # participant 0 does not observe vertices 0..3
    sims[0, :, :4] = np.nan
    expected = np.nanmean(sims.astype(np.float64), axis=0)
    np.fill_diagonal(expected, 1)
    expected = compute_affinity(expected, kernel=None, sparsity=0.9, non_negative=True)
    expected = (expected + expected.T) / 2
    np.testing.assert_allclose(build_mean_similarity_affinity(sims, sparsity=0.9), expected, rtol=1e-6, atol=1e-7)


def test_unobserved_pair_stops_the_fit():
    sims = np.stack([vertex_similarity(fc) for fc in subject_fingerprints(n_subjects=2)])
    sims[:, 5, :] = np.nan
    sims[:, :, 5] = np.nan
    with pytest.raises(ValueError, match="No participant observes"):
        build_mean_similarity_affinity(sims, sparsity=0.9)


def test_gradients_follow_the_long_axis():
    sims = np.stack([vertex_similarity(fc) for fc in subject_fingerprints()])
    gradients, lambdas, ev = fit_mean_similarity_gradients(sims, n_components=3, sparsity=0.9)
    affinity = build_mean_similarity_affinity(sims, sparsity=0.9)
    direct, _ = diffusion_map_embedding(affinity, n_components=3)
    np.testing.assert_allclose(gradients, direct, atol=1e-5)
    assert abs(np.corrcoef(gradients[:, 0], np.arange(len(gradients)))[0, 1]) > 0.9
    assert lambdas[0] >= lambdas[1] >= lambdas[2] and np.isclose(ev.sum(), 1.0)


def test_mean_similarity_ward_clusters_exactly_like_gradient_ward():
    rng = np.random.default_rng(0)
    features = np.cumsum(rng.normal(size=(40, 2)), axis=0)
    adjacency = chain_adjacency(40)
    ward, _ = gc.cluster_gradients("gradient-ward", features, adjacency, k=3, seed=1)
    mean_sim, meta = gc.cluster_gradients(gc.MEAN_SIMILARITY_WARD_METHOD, features, adjacency, k=3, seed=1)
    np.testing.assert_array_equal(ward, mean_sim)
    assert meta["linkage"] == "ward"
    assert gc.gradient_method_provenance(gc.MEAN_SIMILARITY_WARD_METHOD)["surface_affinity"] == "none"


def test_group_atlas_features_and_provenance(monkeypatch, tmp_path):
    atlas = stage("05_02_run_group_atlas.py")
    # Provenance lookups only; the embedding itself does not use HippoMaps.
    monkeypatch.setattr(atlas, "ensure_hippomaps_on_path", lambda root=None: tmp_path)
    monkeypatch.setattr(atlas, "hippomaps_commit", lambda root=None: "stub")
    monkeypatch.setattr(atlas, "hippomaps_import_path", lambda root=None: str(tmp_path))
    fps = subject_fingerprints(n_subjects=5)
    bank = {str(i): vertex_similarity(fc) for i, fc in enumerate(fps)}
    support = np.ones(100, dtype=bool)
    options = {"feature_scaling": "none", "sparsify_stage": "affinity", "sparsity": 0.9,
               "clustering_feature_scaling": "none"}
    features, meta = atlas.build_group_gradient_features(
        method=atlas.MEAN_SIMILARITY_WARD_METHOD, feature_bank=bank, subjects_subset=list(bank),
        support_mask=support, gradient_components=(1, 2), gradient_options=options,
    )
    assert features.shape == (100, 2)
    assert meta["group_mean_vertex_similarity_shape"] == [100, 100]
    assert meta["gradient_preprocessing"]["kernel"].startswith("none_precomputed")
    with pytest.raises(ValueError, match="requires --profile topsim"):
        atlas.build_group_gradient_features(
            method=atlas.MEAN_SIMILARITY_WARD_METHOD, feature_bank=bank, subjects_subset=list(bank),
            support_mask=support, gradient_options={**options, "sparsify_stage": "features"},
        )
    assert atlas.uses_subject_similarity_bank(atlas.MEAN_SIMILARITY_WARD_METHOD)
    assert atlas.MEAN_SIMILARITY_WARD_METHOD in atlas.KONG2022_ONLY_METHODS
    fields = gradient_computation_provenance(
        {"gradient_source": atlas.MEAN_SIMILARITY_GRADIENT_SOURCE}, fixed_cluster_reuse=None)
    assert fields["gradient_recomputed"] is True
    assert fields["gradient_computation"] == atlas.MEAN_SIMILARITY_GRADIENT_SOURCE
    assert "gradient_source_manifest" not in fields


def test_generalized_procrustes_recovers_rotated_reflected_copies():
    from lib.hippomaps_adapter import generalized_procrustes_mean
    rng = np.random.default_rng(1)
    template = rng.normal(size=(60, 3))
    subjects = []
    for i in range(5):
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))  # includes reflections
        rows = np.sort(rng.choice(60, size=50, replace=False)) if i else np.arange(60)
        subjects.append((rows, template[rows] @ rotation.T + 0.01 * rng.normal(size=(len(rows), 3))))
    mean, meta = generalized_procrustes_mean(subjects, template)
    assert meta["procrustes_converged"]
    assert min(meta["template_correlation_by_component"]) > 0.99
    np.testing.assert_allclose(mean, template, atol=0.05)


def test_subject_gradient_ward_features(monkeypatch, tmp_path):
    atlas = stage("05_02_run_group_atlas.py")
    monkeypatch.setattr(atlas, "ensure_hippomaps_on_path", lambda root=None: tmp_path)
    monkeypatch.setattr(atlas, "hippomaps_commit", lambda root=None: "stub")
    monkeypatch.setattr(atlas, "hippomaps_import_path", lambda root=None: str(tmp_path))
    bank = {str(i): vertex_similarity(fc) for i, fc in enumerate(subject_fingerprints(n_subjects=5))}
    options = {"feature_scaling": "none", "sparsify_stage": "affinity", "sparsity": 0.9,
               "clustering_feature_scaling": "none"}
    features, meta = atlas.build_group_gradient_features(
        method=atlas.SUBJECT_GRADIENT_WARD_METHOD, feature_bank=bank, subjects_subset=list(bank),
        support_mask=np.ones(100, dtype=bool), gradient_components=(1, 2), gradient_options=options,
    )
    assert features.shape == (100, 2)
    assert meta["n_subject_gradient_fits"] == 5 and meta["template_method"] == atlas.MEAN_SIMILARITY_WARD_METHOD
    assert meta["subject_gradient_alignment"]["template_correlation_by_component"][0] > 0.9
    assert abs(np.corrcoef(features[:, 0], np.arange(100))[0, 1]) > 0.9
    assert atlas.SUBJECT_GRADIENT_WARD_METHOD in atlas.KONG2022_ONLY_METHODS
    assert atlas.uses_subject_similarity_bank(atlas.SUBJECT_GRADIENT_WARD_METHOD)


def test_multigraph_gradient_coordinates():
    from lib import multigraph_spectral as ms
    sims = [vertex_similarity(fc) for fc in subject_fingerprints(n_subjects=4)]
    support = np.ones(100, dtype=bool)
    shen = ms.coordinate_stack(sims, list("abcd"), support, method=ms.MULTIGRAPH_METHOD, n_components=4)
    diff = ms.coordinate_stack(sims, list("abcd"), support, method=ms.MULTIGRAPH_GRADIENT_METHOD, n_components=4)
    assert diff.shape == shen.shape == (4, 100, 4)
    np.testing.assert_allclose(diff[:, :, 0], 1.0)
    assert not np.allclose(np.abs(diff[:, :, 1]), np.abs(shen[:, :, 1]))
    labels, meta = ms.discretize_stack(diff, 3)
    assert set(np.unique(labels)) <= {1, 2, 3} and meta["n_graphs"] == 4


def test_similarity_gradient_halves_use_their_own_feature_builder(monkeypatch, tmp_path):
    """Regression: resampled halves must embed the similarity bank, not average it as FC rows."""
    atlas = stage("05_02_run_group_atlas.py")
    monkeypatch.setattr(atlas, "ensure_hippomaps_on_path", lambda root=None: tmp_path)
    monkeypatch.setattr(atlas, "hippomaps_commit", lambda root=None: "stub")
    monkeypatch.setattr(atlas, "hippomaps_import_path", lambda root=None: str(tmp_path))
    bank = {str(i): vertex_similarity(fc) for i, fc in enumerate(subject_fingerprints(n_subjects=6))}
    options = {"feature_scaling": "none", "sparsify_stage": "affinity", "sparsity": 0.9,
               "clustering_feature_scaling": "none"}
    support = np.ones(100, dtype=bool)
    calls = []
    original = atlas.build_group_gradient_features
    monkeypatch.setattr(atlas, "build_group_gradient_features",
                        lambda **kw: calls.append(kw["method"]) or original(**kw))
    for method in (atlas.MEAN_SIMILARITY_WARD_METHOD, atlas.SUBJECT_GRADIENT_WARD_METHOD):
        half = list(bank)[:3]
        first = atlas.group_feature_matrix(method=method, feature_bank=bank, subjects_subset=half,
                                           support_mask=support, gradient_components=(1, 2), gradient_options=options)
        again = atlas.group_feature_matrix(method=method, feature_bank=bank, subjects_subset=half,
                                           support_mask=support, gradient_components=(1, 2), gradient_options=options)
        direct, _ = original(method=method, feature_bank=bank, subjects_subset=half, support_mask=support,
                             gradient_components=(1, 2), gradient_options=options)
        np.testing.assert_array_equal(first, again)
        np.testing.assert_allclose(first, direct)
    assert calls.count(atlas.MEAN_SIMILARITY_WARD_METHOD) >= 1 and calls.count(atlas.SUBJECT_GRADIENT_WARD_METHOD) == 1

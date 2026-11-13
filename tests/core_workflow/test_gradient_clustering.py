from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/core_workflow"))
from lib import gradient_clustering as gc


def chain(n):
    return sparse.diags([np.ones(n - 1), np.ones(n - 1)], [-1, 1], shape=(n, n)).tocsr()


def features():
    x = np.arange(60, dtype=float)
    return np.column_stack([x / 20, np.sin(x / 10), np.cos(x / 10)])


@pytest.mark.parametrize("method", sorted(gc.GRADIENT_METHODS))
def test_dispatch_returns_reproducible_k_nonempty_clusters(method):
    x, adj = features(), chain(60)
    a, meta = gc.cluster_gradients(method, x, adj, k=3, seed=17)
    b, _ = gc.cluster_gradients(method, x, adj, k=3, seed=17)
    assert np.array_equal(a, b)
    assert set(a) == {1, 2, 3}
    assert meta["method"] == method
    assert sum(meta["cluster_vertex_counts"].values()) == 60
    if method == "gradient-ward":
        assert set(meta["cluster_connected_components"].values()) == {1}


def test_binary_adjacency_removes_duplicate_triangle_weights_and_diagonal():
    adj = chain(5).tolil()
    adj[0, 1], adj[1, 0], adj[0, 0] = 8, 2, 99
    result = gc.binary_mesh_adjacency(adj, 5)
    assert (result != chain(5)).nnz == 0
    assert result.diagonal().sum() == 0


@pytest.mark.parametrize("method", ["gradient-ward", "gradient-spectral-rbf"])
def test_disconnected_mesh_fails_before_sklearn_completion(method):
    adj = sparse.block_diag([chain(30), chain(30)])
    with pytest.raises(gc.GradientClusteringError, match="mesh.*disconnected"):
        gc.cluster_gradients(method, features(), adj, k=3, seed=1)


def test_rbf_uses_unique_edges_and_gradient_magnitude():
    x = np.array([[1., 0], [2., 0], [5., 0], [6., 0]])
    adj = chain(4)
    w, meta = gc.build_rbf_mesh_affinity(x, adj, rbf_scale=1.)
    duplicate, _ = gc.build_rbf_mesh_affinity(x, adj * 4, rbf_scale=1.)
    assert w.dtype == np.float64
    assert np.array_equal(w.toarray(), duplicate.toarray())
    assert meta["sigma"] == 1.
    assert w[0, 1] == pytest.approx(np.exp(-.5))
    assert w[1, 2] == pytest.approx(np.exp(-4.5))
    assert w[1, 2] < w[0, 1]
    assert w.diagonal().sum() == 0


def test_tiny_nonzero_bridge_is_investigated_failure_without_changing_sigma():
    x = np.array([[0.], [1.], [7.], [8.]])
    with pytest.raises(gc.GradientClusteringError, match="effective.*disconnected") as caught:
        gc.build_rbf_mesh_affinity(x, chain(4), rbf_scale=1.)
    meta = caught.value.diagnostics
    assert meta["sigma"] == 1.
    assert meta["positive_connected_components"] == 1
    assert meta["effective_connected_components"] == 2
    assert meta["edge_weight_quantiles"]["min"] > 0
    assert meta["effective_relative_weight_cutoff"] == 1e-5


def test_undefined_bandwidth_fails():
    with pytest.raises(gc.GradientClusteringError, match="positive.*distance"):
        gc.build_rbf_mesh_affinity(np.ones((5, 3)), chain(5))


def test_gmm_reports_discontiguous_clusters_without_spatial_repair():
    x = np.tile(np.array([[-10., 0.], [10., 0.]]), (15, 1))
    labels, meta = gc.cluster_gradients("gradient-gmm", x, chain(30), k=2, seed=3)
    assert all(v == 15 for v in meta["cluster_connected_components"].values())
    assert np.array_equal(labels[::2], np.repeat(labels[0], 15))
    assert meta["converged"] is True
    assert meta["reg_covar"] == 1e-6
    assert len(meta["covariance_condition_numbers"]) == 2


def test_gmm_nonconvergence_is_fatal(monkeypatch):
    class FakeGMM:
        def __init__(self, **kwargs):
            assert kwargs["n_init"] == 10 and kwargs["max_iter"] == 1000
        def fit(self, x):
            self.converged_ = False
            return self
    monkeypatch.setattr(gc, "GaussianMixture", FakeGMM)
    with pytest.raises(gc.GradientClusteringError, match="converg"):
        gc.cluster_gradients("gradient-gmm", features(), chain(60), k=3, seed=3)


def test_small_clusters_are_flagged_without_rebalancing():
    labels = np.array([1] * 99 + [2])
    meta = gc.cluster_mesh_diagnostics(labels, chain(100), k=2)
    assert meta["small_cluster_threshold"] == 3
    assert meta["small_clusters"] == [2]
    assert np.sum(labels == 2) == 1


def test_registry_distinguishes_gradient_and_spectral_methods():
    assert gc.is_gradient_method("gradient-ward")
    assert gc.is_gradient_method("gradient-gmm")
    assert not gc.is_spectral_gradient_method("gradient-gmm")
    assert gc.is_spectral_gradient_method("gradient-spectral-rbf")
    with pytest.raises(ValueError, match="Unknown gradient method"):
        gc.cluster_gradients("made-up", features(), chain(60), k=3, seed=3)


def test_group_atlas_dispatches_new_methods_and_preserves_metadata():
    import importlib.util
    path = Path(__file__).resolve().parents[2] / "scripts/core_workflow/05_02_run_group_atlas.py"
    spec = importlib.util.spec_from_file_location("clustering_atlas_test", path)
    core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = core
    spec.loader.exec_module(core)
    assert all(core.is_gradient_method(m) for m in gc.GRADIENT_METHODS)
    meta = {}
    active, full = core.cluster_from_precomputed_group_features(
        method="gradient-ward", group_features=features(), support_mask=np.ones(60, bool),
        adjacency_full=chain(60), k=3, seed=3, clustering_metadata=meta,
    )
    assert np.array_equal(full, active)
    assert meta["method"] == "gradient-ward"
    assert set(meta["cluster_connected_components"].values()) == {1}
    with pytest.raises(ValueError, match="does not construct"):
        core.build_method_surface_affinity("gradient-gmm", features(), chain(60))


def test_crossfit_method_selection_is_explicit_in_configuration():
    import importlib.util
    path = Path(__file__).resolve().parents[2] / "scripts/individual_workflow/01_01_run_hcp_crossfit_template_matching.py"
    spec = importlib.util.spec_from_file_location("clustering_crossfit_test", path)
    workflow = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = workflow
    spec.loader.exec_module(workflow)
    args = workflow.build_arg_parser().parse_args(["--method", "gradient-gmm", "--rbf-scale", "2"])
    assert args.method == "gradient-gmm"
    assert args.rbf_scale == 2.
    assert workflow.CrossfitConfig.__dataclass_fields__["method"].default == "gradient-spectral"


def test_ward_records_no_seed_and_seeded_methods_do():
    import numpy as np
    from scipy import sparse
    from lib.gradient_clustering import cluster_gradients

    rng = np.random.default_rng(5)
    features = np.vstack([rng.normal(loc, 0.1, size=(10, 2)) for loc in (0.0, 3.0)])
    chain = sparse.diags([np.ones(19), np.ones(19)], [1, -1]).tocsr()
    _, ward = cluster_gradients("gradient-ward", features, chain, k=2, seed=7)
    _, gmm = cluster_gradients("gradient-gmm", features, chain, k=2, seed=7)
    assert "seed" not in ward
    assert gmm["seed"] == 7

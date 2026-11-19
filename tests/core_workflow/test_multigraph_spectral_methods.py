"""Regressions for the participant-graph spectral methods and their K-estimated group-atlas path."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/core_workflow"))

from lib import multigraph_spectral as ms  # noqa: E402
from lib import output_layout  # noqa: E402
from lib.cluster_roi_profile_contract import validate_cluster_roi_profile_manifest  # noqa: E402
from lib.cortex_overview_cache import hemisphere_cluster_roi_profiles  # noqa: E402


def stage(name):
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), ROOT / "scripts/core_workflow" / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parcel_similarities(n_subjects=9, per_parcel=14, n_parcels=3, missing=((0, 0), (3, 20)), seed=5):
    """Participant similarity matrices from three shared parcels that blend at their borders.

    The blend keeps every participant graph connected; fully separate parcels would stop the fit.
    """
    rng = np.random.default_rng(seed)
    truth = np.repeat(np.arange(n_parcels), per_parcel)
    centres = (np.arange(n_parcels) + 0.5) * per_parcel
    weights = np.exp(-0.5 * ((np.arange(truth.size)[:, None] - centres[None]) / (per_parcel / 2.0)) ** 2)
    weights /= weights.sum(axis=1, keepdims=True)
    signals = rng.normal(size=(n_parcels, 60))
    matrices = []
    for subject in range(n_subjects):
        fc = weights @ signals + 0.3 * rng.normal(size=(truth.size, 60))
        for who, vertex in missing:
            if who == subject:
                fc[vertex] = np.nan
        matrices.append(ms.vertex_similarity(fc))
    return truth, matrices


def test_vertex_similarity_is_pearson_and_keeps_missing_vertices():
    fc = np.random.default_rng(1).normal(size=(12, 40))
    fc[4] = np.nan
    similarity = ms.vertex_similarity(fc)
    observed = ms.observed_vertices(similarity)
    assert observed.tolist() == [i != 4 for i in range(12)]
    assert np.isnan(similarity[4]).all() and np.isnan(similarity[:, 4]).all()
    np.testing.assert_allclose(similarity[np.ix_(observed, observed)], np.corrcoef(fc[observed]), atol=1e-6)
    partial = fc.copy()
    partial[2, 7] = np.nan
    with pytest.raises(ms.FitError, match="Partially observed FC fingerprint at vertex 2"):
        ms.vertex_similarity(partial)
    constant = fc.copy()
    constant[9] = 0.3
    with pytest.raises(ms.FitError, match="Constant FC fingerprint at vertex 9"):
        ms.vertex_similarity(constant)


@pytest.mark.parametrize("method", sorted(ms.MULTIGRAPH_METHODS))
def test_both_methods_recover_shared_parcels_from_a_k_independent_stack(method):
    truth, matrices = parcel_similarities()
    support = np.ones(truth.size, dtype=bool)
    subjects = [f"s{i}" for i in range(len(matrices))]
    stack = ms.coordinate_stack(matrices, subjects, support, method=method, n_components=5)
    assert stack.shape == ((len(matrices) if method == ms.MULTIGRAPH_METHOD else 1), truth.size, 5)
    if method == ms.MULTIGRAPH_METHOD:
        assert np.isnan(stack[0, 0]).all() and np.isnan(stack[3, 20]).all()
        assert np.isfinite(stack[1]).all()
    labels, diagnostics = ms.discretize_stack(stack, 3)
    assert adjusted_rand_score(truth, labels) == 1
    assert diagnostics["occupied_parcels"] == 3
    min_coherence, mean_coherence = ms.stack_coherence(stack, labels, 3)
    assert 0.8 < min_coherence <= mean_coherence <= 1
    # The same stack serves every K, and a cached rebuild is identical.
    assert set(np.unique(ms.discretize_stack(stack, 2)[0])) == {1, 2}
    np.testing.assert_array_equal(
        stack, ms.coordinate_stack(matrices, subjects, support, method=method, n_components=5),
    )


def test_unsupported_k_is_returned_with_its_empty_parcel():
    # Two separable groups in three coordinates: the third parcel cannot be occupied.
    rows = np.vstack([np.tile([1.0, 0.0, 0.0], (6, 1)), np.tile([0.0, 1.0, 0.0], (6, 1))])
    support = np.ones(len(rows), dtype=bool)
    embeddings = [(np.arange(len(rows)), rows)] * 3
    with pytest.raises(ms.FitError, match="nonempty stable partition"):
        ms.shen_discretize(embeddings, support, k=3)
    fit = ms.shen_discretize(embeddings, support, k=3, require_occupied=False)
    assert fit.diagnostics["occupied_parcels"] == 2
    assert np.unique(fit.labels).size == 2
    labels, diagnostics = ms.discretize_stack(np.repeat(rows[None], 3, axis=0), 3)
    assert diagnostics["occupied_parcels"] == 2 and labels.min() >= 1
    assert ms.stack_coherence(np.repeat(rows[None], 3, axis=0), labels, 3)[0] == 0.0


def test_group_atlas_registers_the_methods_and_clusters_their_stacks():
    atlas = stage("05_02_run_group_atlas.py")
    overview = stage("06_01_build_group_overview.py")
    truth, matrices = parcel_similarities()
    subjects = [f"s{i}" for i in range(len(matrices))]
    bank = dict(zip(subjects, matrices, strict=True))
    support = np.ones(truth.size, dtype=bool)
    adjacency = atlas.sparse.csr_matrix(np.eye(truth.size, k=1) + np.eye(truth.size, k=-1))
    for method in sorted(ms.MULTIGRAPH_METHODS):
        assert method in atlas.METHOD_CHOICES and method in overview.METHOD_CHOICES
        assert atlas.method_allowed_for_atlas(method, "kong2022")
        assert not atlas.method_allowed_for_atlas(method, "lynch2024_20net")
        assert atlas.method_uses_network_fc_input(method) and not atlas.is_gradient_method(method)
        assert atlas.method_uses_kong2022_topn_cluster_annotation(method, "kong2022")
        assert atlas.method_surface_affinity_metadata(method) == ("none", ms.AFFINITY_DESCRIPTION)
        assert overview._uses_cluster_annotation_contract(method) and not overview._is_gradient_method(method)
        _, full = atlas.cluster_for_subset(
            method=method, feature_bank=bank, subjects_subset=subjects, support_mask=support,
            adjacency_full=adjacency, k=3, seed=11,
        )
        assert adjusted_rand_score(truth, full) == 1
    assert atlas.METHODS_DEFAULT == ["intrinsic-spectral", "network-spectral", "wta-top"]
    with pytest.raises(ValueError, match="supports K <= 8"):
        atlas.cluster_from_precomputed_group_features(
            method=ms.MULTIGRAPH_METHOD, group_features=np.zeros((2, truth.size, 8), dtype=np.float32),
            support_mask=support, adjacency_full=adjacency, k=9, seed=1,
        )
    with pytest.raises(ValueError, match="differ from the participant validity mask"):
        atlas.subject_similarity_feature(np.ones((4, 5)) * np.arange(5), np.array([True, True, False, True]),
                                         source=Path("fc.npy"))


def test_null_permutation_moves_rows_and_columns_of_a_similarity_feature(monkeypatch):
    atlas = stage("05_02_run_group_atlas.py")
    _, matrices = parcel_similarities(missing=())
    subjects = [f"s{i}" for i in range(len(matrices))]
    seen = {}

    def record(**kwargs):
        seen.update(bank=kwargs["feature_bank"], masks=kwargs["valid_masks_by_subject"])
        return {"values": [], "n_requested": 1, "n_used": 0, "n_skipped_overlap": 0, "n_skipped_empty": 0,
                "n_skipped_failure": 0, "failure_examples": []}

    monkeypatch.setattr(atlas, "subject_split_ari", record)
    atlas.strong_null_ari(
        method=ms.MULTIGRAPH_METHOD, feature_bank=dict(zip(subjects, matrices, strict=True)),
        valid_masks_by_subject={s: np.ones(len(matrices[0]), dtype=bool) for s in subjects},
        subjects=subjects, tau=0.8, adjacency_full=None, k=2, seeds=[3],
    )
    permuted = seen["bank"]["s0"]
    np.testing.assert_allclose(permuted, permuted.T)
    np.testing.assert_allclose(np.diagonal(permuted), 1.0)
    assert not np.allclose(permuted, matrices[0])


def test_single_parcel_halves_are_not_scored_as_perfect_agreement(monkeypatch):
    atlas = stage("05_02_run_group_atlas.py")
    subjects = [f"s{i}" for i in range(6)]
    masks = {s: np.ones(10, dtype=bool) for s in subjects}
    monkeypatch.setattr(atlas, "cluster_for_subset", lambda **kwargs: (np.ones(10, dtype=int), np.ones(10, dtype=int)))
    result = atlas.subject_split_ari(
        method=ms.MULTIGRAPH_METHOD, feature_bank={}, valid_masks_by_subject=masks, subjects=subjects,
        tau=0.8, adjacency_full=None, k=2, seeds=[1, 2],
    )
    assert result["values"] == [] and result["n_skipped_empty"] == 2


def test_automatic_primary_k_is_an_explicit_exclusive_choice():
    atlas = stage("05_02_run_group_atlas.py")
    parser = atlas.build_arg_parser()
    ks = list(range(2, 9))

    def resolved(*argv):
        args = parser.parse_args(list(argv))
        atlas.resolve_primary_k_options(args, ks, argv=list(argv))
        return args

    assert resolved().default_k == 3
    # The automatic K must become the exported K under either K rule.
    assert resolved("--automatic-primary-k").default_k is None
    assert resolved("--automatic-primary-k", "--k-final-rule", "largest-admissible").default_k is None
    fallback = resolved("--default-k", "3", "--undersized-default-k-fallback", "2")
    assert (fallback.default_k, fallback.undersized_default_k_fallback) == (3, 2)
    for argv, message in (
        (("--automatic-primary-k", "--default-k", "3"), "cannot be combined"),
        (("--automatic-primary-k", "--force-final-k", "3"), "cannot be combined"),
        (("--automatic-primary-k", "--undersized-default-k-fallback", "2"), "requires --default-k"),
        (("--undersized-default-k-fallback", "9"), "within the requested K range"),
        (("--k-final-rule", "largest-admissible", "--k-selection-mode", "fixed-comparison"), "redundancy-aware"),
    ):
        with pytest.raises(ValueError, match=message):
            resolved(*argv)


def test_largest_admissible_rule_ignores_stability_and_keeps_it_as_audit():
    atlas = stage("05_02_run_group_atlas.py")
    assert atlas.build_arg_parser().parse_args([]).k_final_rule == "smallest-stable"
    # K=2 is the most reproducible candidate; K=3 failed a gate and is absent; K=5 is the largest that passed.
    pool = [{"k": k, "group_ari_mean": ari, "group_ari_se": 0.01} for k, ari in ((2, 0.90), (4, 0.60), (5, 0.55))]
    assert atlas.select_k_final(pool, "smallest-stable")[:2] == (2, "local_max_within_1se_after_redundancy_filter")
    k_final, path, within_1se, argmax_k = atlas.select_k_final(pool, "largest-admissible")
    assert (k_final, path) == (5, "largest_k_passing_gates")
    assert (within_1se, argmax_k) == ({2}, 2)
    with pytest.raises(ValueError, match="Unsupported K rule"):
        atlas.select_k_final(pool, "largest")


def test_undersized_default_k_falls_back_to_the_declared_k():
    atlas = stage("05_02_run_group_atlas.py")
    rows = [
        {"k": 2, "nontrivial_min_size_ok": 1, "min_cluster_size": 64, "V_min_effective": 16},
        {"k": 3, "nontrivial_min_size_ok": 0, "min_cluster_size": 1, "V_min_effective": 16},
        {"k": 4, "nontrivial_min_size_ok": 1, "min_cluster_size": 40, "V_min_effective": 16},
    ]
    default = {"chosen_k_final": 3, "automatic_k_final": 2, "primary_k_source": "default_k"}
    assert atlas.apply_undersized_default_k_fallback(default, rows, 2) == {
        "chosen_k_final": 2, "automatic_k_final": 2, "primary_k_source": "default_k_undersized_fallback",
        "undersized_default_k": 3,
    }
    # A default K that meets the size gate, and any non-default primary, are left as resolved.
    assert atlas.apply_undersized_default_k_fallback({**default, "chosen_k_final": 4}, rows, 2)["chosen_k_final"] == 4
    automatic = {"chosen_k_final": 3, "automatic_k_final": 3, "primary_k_source": "automatic_k"}
    assert atlas.apply_undersized_default_k_fallback(automatic, rows, 2) is automatic
    rows[0]["nontrivial_min_size_ok"] = 0
    with pytest.raises(RuntimeError, match="both have a parcel below the minimum size"):
        atlas.apply_undersized_default_k_fallback(default, rows, 2)
    with pytest.raises(RuntimeError, match="Fallback K=7 was not evaluated"):
        atlas.apply_undersized_default_k_fallback(default, rows, 7)


def test_side_checkout_reads_another_data_root_and_writes_only_comparisons(monkeypatch, tmp_path):
    data = tmp_path / "main"
    (data / "outputs_hcp/inputs").mkdir(parents=True)
    assert output_layout.resolve_data_root({}) == output_layout.REPO_ROOT
    assert output_layout.resolve_data_root({output_layout.DATA_CHECKOUT_ENV: str(data)}) == data.resolve()
    with pytest.raises(ValueError, match="existing absolute checkout directory"):
        output_layout.resolve_data_root({output_layout.DATA_CHECKOUT_ENV: "relative/checkout"})

    atlas = stage("05_02_run_group_atlas.py")
    monkeypatch.setattr(atlas, "DATA_ROOT", data.resolve())
    sources = [data / "outputs_hcp/inputs"]
    comparison = atlas.REPO_ROOT / "tmp/example/group_atlas/pfm"
    assert atlas.validate_dataset_output_contract(
        comparison, "outputs_hcp", source_roots=sources, comparison=True,
    ) == ("outputs_hcp", data.resolve() / "outputs_hcp")
    with pytest.raises(ValueError, match="may only write --comparison-output runs under its own tmp"):
        atlas.validate_dataset_output_contract(
            data / "outputs_hcp/work/group_atlas/pfm", "outputs_hcp", source_roots=sources,
        )
    with pytest.raises(ValueError, match="outside repository tmp"):
        atlas.validate_dataset_output_contract(
            data / "tmp/elsewhere", "outputs_hcp", source_roots=sources, comparison=True,
        )


def test_cluster_roi_profile_contract_accepts_a_declared_estimated_k(tmp_path):
    def run(k):
        folder = tmp_path / f"k{k}"
        folder.mkdir()
        profile = folder / "cluster_roi_profile.npy"
        np.save(profile, np.arange(k * 400, dtype=float).reshape(k, 400))
        manifest = {
            "status": "ok", "cluster_roi_profile": str(profile), "cluster_roi_profile_contract": "roi_ap",
            "cluster_label_ordering": "long_axis_head_to_tail",
            "cluster_order_head_to_tail": list(range(1, k + 1)), "chosen_k_final": k,
        }
        return folder / "group_atlas_manifest.json", manifest, profile

    path2, manifest2, profile2 = run(2)
    path4, manifest4, _ = run(4)
    with pytest.raises(ValueError, match=r"cluster_order_head_to_tail must be \[1, 2, 3\]"):
        validate_cluster_roi_profile_manifest(manifest2, profile_path=profile2, profile_shape=(2, 400))
    validate_cluster_roi_profile_manifest(manifest2, profile_path=profile2, profile_shape=(2, 400), n_clusters=2)
    left, right = hemisphere_cluster_roi_profiles([path2, path4], [manifest2, manifest4])
    assert left.shape == (2, 400) and right.shape == (4, 400)
    assert hemisphere_cluster_roi_profiles([path2], [{**manifest2, "cluster_roi_profile": None}]) is None

"""Numerical contracts for explicitly selected gradient preprocessing."""
# pyright: reportMissingImports=false
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from brainspace.gradient import GradientMaps
from brainspace.gradient.kernels import compute_affinity

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.core_workflow.lib import hippomaps_adapter as adapter  # noqa: E402


def test_group_atlas_requires_explicit_validity_mask_even_when_fc_exists(tmp_path):
    atlas = _stage("05_02_run_group_atlas.py")
    shared = tmp_path / "shared"
    fc = shared / "sub-01/fc/kong2022/2mm/hemi_L/hipp_vertex_to_network_fc.npy"
    fc.parent.mkdir(parents=True)
    np.save(fc, np.ones((3, 2)))
    surface = tmp_path / "surface"
    expected = surface / "sub-01/surface/validity/sub-01_hemi-L_valid_mask.npy"
    with pytest.raises(FileNotFoundError, match="sub-01_hemi-L_valid_mask"):
        atlas.subject_valid_mask(surface, shared, "01", "2mm", "L")
    expected.parent.mkdir(parents=True)
    np.save(expected, np.array([True, False, True]))
    assert atlas.subject_valid_mask(surface, shared, "01", "2mm", "L").tolist() == [True, False, True]


@pytest.mark.parametrize("scaling", ["none", "col_zscore"])
@pytest.mark.parametrize("stage", ["features", "affinity", "none"])
def test_scaling_and_sparsification_are_separate_explicit_choices(monkeypatch, scaling, stage):
    raw = np.random.default_rng(17).normal(size=(20, 30)).astype(np.float32)
    raw *= np.linspace(1, 9, 30)
    before = raw.copy()
    calls = []

    def capture(matrix, **kwargs):
        calls.append((matrix.copy(), kwargs))
        return compute_affinity(matrix, **kwargs)

    monkeypatch.setattr("brainspace.gradient.kernels.compute_affinity", capture)
    affinity = adapter.build_pearson_affinity(
        raw, feature_scaling=scaling, sparsify_stage=stage, sparsity=0.5,
    )
    expected = raw if scaling == "none" else adapter.feature_zscore_columns(raw)
    assert len(calls) == 1
    assert np.array_equal(calls[0][0], expected)
    assert calls[0][1] == {
        "kernel": "pearson", "sparsity": None if stage == "none" else 0.5,
        "pre_sparsify": stage != "affinity", "non_negative": True,
    }
    expected_affinity = compute_affinity(expected.copy(), **calls[0][1])
    assert np.allclose(affinity, (expected_affinity + expected_affinity.T) / 2)
    assert np.all(affinity >= 0)
    assert np.allclose(np.diag(affinity), 1)
    assert np.array_equal(raw, before)


def test_embedding_consumes_the_affinity_without_second_kernel_or_sparsification(monkeypatch):
    calls = []
    real_embedding = adapter.diffusion_map_embedding

    def embedding(affinity, **kwargs):
        calls.append((affinity.copy(), kwargs))
        return real_embedding(affinity, **kwargs)

    monkeypatch.setattr(adapter, "diffusion_map_embedding", embedding)
    raw = np.random.default_rng(4).normal(size=(20, 30))
    expected_affinity = adapter.build_pearson_affinity(
        raw, feature_scaling="none", sparsify_stage="affinity", sparsity=0.3,
    )
    gradients, lambdas, ev = adapter.fit_diffusion_gradients(
        raw, n_components=3, feature_scaling="none", sparsify_stage="affinity", sparsity=0.3,
    )
    assert len(calls) == 1
    assert calls[0][1] == {"n_components": 3}
    assert np.array_equal(calls[0][0], expected_affinity)
    assert gradients.shape == (20, 3)
    assert np.allclose(ev, lambdas / lambdas.sum())


def test_exact_embedding_solves_the_diffusion_operator_and_has_no_seed():
    raw = np.random.default_rng(3).normal(size=(64, 80)).astype(np.float32)
    affinity = adapter.build_pearson_affinity(raw, feature_scaling="col_zscore", sparsify_stage="features", sparsity=0.9)
    gradients, lambdas = adapter.diffusion_map_embedding(affinity, n_components=5)
    again, lambdas_again = adapter.diffusion_map_embedding(affinity, n_components=5)
    assert np.array_equal(gradients, again) and np.array_equal(lambdas, lambdas_again)

    weights = affinity.astype(np.float64)
    scale = weights.sum(axis=1) ** -0.5
    weights = weights * scale[:, None] * scale[None, :]
    operator = weights / weights.sum(axis=1, keepdims=True)
    eigenvalues = lambdas / (1 + lambdas)
    eigenvectors = gradients / lambdas
    assert np.allclose(operator @ eigenvectors, eigenvectors * eigenvalues, atol=1e-10)
    assert np.all(np.diff(lambdas) < 0)
    stationary = weights.sum(axis=1) / weights.sum()
    assert np.allclose(stationary @ eigenvectors**2, 1)
    assert np.allclose(eigenvectors.T @ (stationary[:, None] * eigenvectors), np.eye(5), atol=1e-10)
    assert np.all(gradients[np.abs(gradients).argmax(axis=0), np.arange(5)] > 0)

    # BrainSpace feeds the non-symmetric operator to a symmetric iterative
    # solver; it approximates the same components but depends on its seed.
    legacy = GradientMaps(n_components=5, kernel=None, random_state=11).fit(affinity, sparsity=None)
    for component in range(5):
        assert abs(np.corrcoef(gradients[:, component], legacy.gradients_[:, component])[0, 1]) > 0.9


def test_exact_embedding_rejects_invalid_affinities():
    with pytest.raises(ValueError, match="symmetric"):
        adapter.diffusion_map_embedding(np.array([[1., .2, .1], [.9, 1., .3], [.1, .3, 1.]]), n_components=1)
    with pytest.raises(ValueError, match="disconnected"):
        adapter.diffusion_map_embedding(np.kron(np.eye(2), np.ones((3, 3))), n_components=2)
    with pytest.raises(ValueError, match="n_components"):
        adapter.diffusion_map_embedding(np.ones((3, 3)), n_components=3)


def test_disconnected_diffusion_affinity_is_rejected():
    row = np.arange(4, dtype=float)
    raw = np.vstack([np.tile(row, (4, 1)), np.tile(-row, (4, 1))])
    with pytest.raises(ValueError, match="disconnected"):
        adapter.build_pearson_affinity(raw, feature_scaling="none", sparsify_stage="none")


def test_gradient_cache_requires_exact_declared_preprocessing(tmp_path):
    path = tmp_path / "hippomaps_gradient_manifest.json"
    requested = adapter.gradient_preprocessing_settings(
        400, feature_scaling="none", sparsify_stage="affinity", sparsity=0.9,
    )
    with pytest.raises(ValueError, match="gradient_preprocessing.*regenerate"):
        adapter.validate_gradient_preprocessing({}, settings=requested, manifest_path=path)
    recorded = {"gradient_preprocessing": dict(requested)}
    adapter.validate_gradient_preprocessing(recorded, settings=requested, manifest_path=path)
    recorded["gradient_preprocessing"]["feature_scaling"] = "col_zscore"
    with pytest.raises(ValueError, match="feature_scaling.*regenerate"):
        adapter.validate_gradient_preprocessing(recorded, settings=requested, manifest_path=path)


def test_clustering_unit_sd_is_fitted_on_training_vertices_only():
    training = np.array([[1., 10.], [3., 50.], [8., 30.]])
    scaled, transform = adapter.scale_gradient_features(training, scaling="unit_sd")
    assert np.allclose(scaled.mean(axis=0), 0)
    assert np.allclose(scaled.std(axis=0, ddof=0), 1)
    assert np.allclose(transform["component_means"], training.mean(axis=0))
    assert np.allclose(transform["component_sds"], training.std(axis=0, ddof=0))
    heldout = np.array([[100., -500.]])
    applied, recorded = adapter.scale_gradient_features(
        heldout, scaling="unit_sd", transform=transform,
    )
    assert np.allclose(applied, (heldout-training.mean(axis=0))/training.std(axis=0))
    assert recorded == transform
    native, _ = adapter.scale_gradient_features(training, scaling="none")
    assert np.array_equal(native, training)


@pytest.mark.parametrize("scaling", ["none", "unit_sd"])
def test_clustering_rejects_a_zero_variance_component(scaling):
    with pytest.raises(ValueError, match="zero-variance"):
        adapter.scale_gradient_features(np.array([[1., 3.], [2., 3.]]), scaling=scaling)


def _stage(filename):
    core = ROOT / "scripts/core_workflow"
    sys.path.insert(0, str(core))
    name = "explicit_preprocessing_" + filename.split(".")[0]
    spec = importlib.util.spec_from_file_location(name, core / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_producer_forwards_and_exports_requested_preprocessing(tmp_path, monkeypatch):
    producer = _stage("05_01_compute_hippomaps_gradients.py")
    surface_dir = tmp_path / "surfaces"
    surface_dir.mkdir()
    (surface_dir / "group-average_hemi-L_space-corobl_den-512_label-hipp_midthickness.surf.gii").touch()
    mask = np.ones(8, dtype=bool)
    gradient = np.random.default_rng(4).normal(size=(8, 3)).astype(np.float32)
    calls = []
    monkeypatch.setattr(producer, "read_surface_vertex_count", lambda _: 8)
    monkeypatch.setattr(producer, "subject_valid_mask", lambda *args: mask)
    monkeypatch.setattr(producer, "load_subject_fc", lambda *args, **kwargs: (np.zeros((8,400)), tmp_path / "fc.npy"))
    monkeypatch.setattr(producer, "load_kong2022_roi_fc_contract", lambda *args, **kwargs: {"roi_names": [str(i) for i in range(400)]})

    def fit(matrix, **kwargs):
        calls.append(kwargs)
        return gradient, np.ones(3), np.ones(3)/3

    monkeypatch.setattr(producer, "fit_diffusion_gradients", fit)
    monkeypatch.setattr(producer, "hippomaps_import_path", lambda _: "test")
    monkeypatch.setattr(producer, "hippomaps_commit", lambda _: "test")
    result = producer.compute_combo(
        route="pfm", signal_branch="standard", shared_root=tmp_path, reference_root=tmp_path,
        atlas="kong2022", surface_root=tmp_path, subjects=["01"], subjects_manifest=tmp_path / "subjects.txt",
        smoothing="2mm", hemi="L", coverage_threshold=.8, group_average_surfaces_dir=surface_dir,
        out_root=tmp_path / "gradients", hippomaps_root=tmp_path, n_components=3,
        selected_components=[1,2,3], force=False,
        feature_scaling="none", sparsify_stage="affinity", sparsity=.6,
    )
    assert calls == [{"n_components":3, "feature_scaling":"none", "sparsify_stage":"affinity", "sparsity":.6}]
    path = producer.write_aligned_combo_outputs(
        result=result, gradients_active=gradient, gradients_full=gradient,
        lambdas=result.lambdas, ev_fractions=result.ev_fractions,
        selected_components=[1,2,3], alignment_meta={"anchor_smoothing":"4mm","anchor_hemi":"L"},
    )
    manifest = json.loads(path.read_text())
    assert manifest["gradient_preprocessing"] == adapter.gradient_preprocessing_settings(
        400, feature_scaling="none", sparsify_stage="affinity", sparsity=.6,
    )
    assert "matching HippoMaps tutorial" not in manifest["feature_zscore_pre_gradientmaps"]


@pytest.mark.parametrize("recorded", [None, "col_zscore"])
def test_group_atlas_loader_rejects_missing_or_other_preprocessing(tmp_path, recorded):
    consumer = _stage("05_02_run_group_atlas.py")
    run = tmp_path / "pfm/2mm/hemi_L"
    run.mkdir(parents=True)
    array_path = run / "hippomaps_gradients_full.npy"
    np.save(array_path, np.random.default_rng(2).normal(size=(8,5)))
    manifest = {"status":"ok", "group_fc_shape":[8,400], "feature_count":400,
                "outputs":{"hippomaps_gradients_full":str(array_path)}}
    if recorded is not None:
        manifest["gradient_preprocessing"] = adapter.gradient_preprocessing_settings(400, feature_scaling=recorded)
    (run / "hippomaps_gradient_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="regenerate"):
        consumer.load_hippomaps_gradient_features(
            gradient_root=tmp_path, route="pfm", smoothing="2mm", hemi="L",
            support_mask=np.ones(8,dtype=bool), feature_scaling="none",
        )


def test_group_atlas_loader_rejects_gradients_from_the_seeded_solver(tmp_path):
    consumer = _stage("05_02_run_group_atlas.py")
    run = tmp_path / "pfm/2mm/hemi_L"
    run.mkdir(parents=True)
    array_path = run / "hippomaps_gradients_full.npy"
    np.save(array_path, np.random.default_rng(2).normal(size=(8,5)))
    preprocessing = adapter.gradient_preprocessing_settings(400)
    assert preprocessing["eigensolver"] == adapter.GRADIENT_EIGENSOLVER
    del preprocessing["eigensolver"]
    manifest = {"status":"ok", "group_fc_shape":[8,400], "feature_count":400,
                "gradient_preprocessing": preprocessing, "gradient_random_state": 20260420,
                "outputs":{"hippomaps_gradients_full":str(array_path)}}
    (run / "hippomaps_gradient_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="eigensolver.*regenerate"):
        consumer.load_hippomaps_gradient_features(
            gradient_root=tmp_path, route="pfm", smoothing="2mm", hemi="L",
            support_mask=np.ones(8,dtype=bool),
        )


def test_subset_fit_uses_declared_preprocessing_and_its_own_component_sd(tmp_path, monkeypatch):
    atlas = _stage("05_02_run_group_atlas.py")
    monkeypatch.setattr(atlas, "ensure_hippomaps_on_path", lambda path: tmp_path)
    monkeypatch.setattr(atlas, "hippomaps_import_path", lambda path: "test")
    monkeypatch.setattr(atlas, "hippomaps_commit", lambda path: "test")
    raw = np.random.default_rng(7).normal(size=(12,5)).astype(np.float32)
    calls = []

    def fit(matrix, **kwargs):
        calls.append(kwargs)
        return raw, np.ones(5), np.ones(5)/5

    options = {"feature_scaling":"none", "sparsify_stage":"affinity", "sparsity":.7,
               "clustering_feature_scaling":"unit_sd"}
    selected, meta = atlas.build_group_gradient_features(
        feature_bank={"a":np.ones((12,400))}, subjects_subset=["a"],
        support_mask=np.ones(12,dtype=bool), gradient_fit_fn=fit,
        gradient_options=options,
    )
    assert calls == [{"n_components":5, "feature_scaling":"none", "sparsify_stage":"affinity", "sparsity":.7}]
    assert np.allclose(selected.std(axis=0), 1)
    assert meta["clustering_feature_transform"]["n_vertices_fit"] == 12
    assert meta["gradient_preprocessing"]["feature_scaling"] == "none"


def test_subject_split_passes_options_to_both_independent_subset_fits(monkeypatch):
    from scipy import sparse
    atlas = _stage("05_02_run_group_atlas.py")
    seen = []

    def cluster(**kwargs):
        seen.append(kwargs["gradient_options"])
        labels = np.tile([1,2,3], 4)
        return labels, labels

    monkeypatch.setattr(atlas, "cluster_for_subset", cluster)
    subjects = ["a","b","c","d"]
    options = {"feature_scaling":"none", "sparsify_stage":"none"}
    result = atlas.subject_split_ari(
        method="gradient-spectral", feature_bank={},
        valid_masks_by_subject={s:np.ones(12,dtype=bool) for s in subjects},
        subjects=subjects, tau=.8, adjacency_full=sparse.eye(12,format="csr"),
        k=3, seeds=[9], gradient_options=options,
    )
    assert seen == [options, options]
    assert result["values"] == [1.0]


def test_subset_alignment_refits_4mm_left_on_the_same_training_subjects(tmp_path, monkeypatch):
    atlas = _stage("05_02_run_group_atlas.py")
    monkeypatch.setattr(atlas, "ensure_hippomaps_on_path", lambda p: Path(p))
    monkeypatch.setattr(atlas, "hippomaps_import_path", lambda p: str(p))
    monkeypatch.setattr(atlas, "hippomaps_commit", lambda p: "test")
    rng = np.random.default_rng(13)
    reference = rng.normal(size=(20, 5)).astype(np.float32)
    target = reference[:, [2, 0, 1, 4, 3]] * [-1, 1, -1, 1, 1]
    fc_path, mask_path = tmp_path / "anchor_fc.npy", tmp_path / "anchor_mask.npy"
    np.save(fc_path, np.full((20, 400), .5, np.float32))
    np.save(mask_path, np.ones(20, bool))
    calls = []
    def fit(data, **kwargs):
        calls.append((data.copy(), kwargs))
        return (reference if data[0, 0] > .4 else target), np.arange(5, 0, -1.), np.ones(5)/5
    options = {"feature_scaling": "none", "sparsify_stage": "none",
               "alignment_fc_paths": {"train": str(fc_path), "heldout": str(tmp_path / "never_read.npy")},
               "alignment_mask_paths": {"train": str(mask_path)}, "alignment_coverage_threshold": .8}
    actual, meta = atlas.build_group_gradient_features(
        feature_bank={"train": np.full((20, 400), .1)}, subjects_subset=["train"],
        support_mask=np.ones(20, bool), gradient_options=options, gradient_fit_fn=fit,
    )
    assert len(calls) == 2
    assert calls[0][1] == calls[1][1]
    assert np.allclose(actual, reference[:, :3])
    assert meta["gradient_alignment"]["training_subjects"] == ["train"]
    assert meta["gradient_alignment"]["anchor_smoothing"] == "4mm"


def test_null_vertex_permutations_are_independent_of_python_hash_salt(monkeypatch):
    atlas = _stage("05_02_run_group_atlas.py")
    captured = []
    monkeypatch.setattr(atlas, "subject_split_ari", lambda **kw: captured.append(kw["feature_bank"]) or {})
    kwargs = {
        "method": "gradient-ward",
        "feature_bank": {"a": np.arange(40).reshape(10, 4)},
        "valid_masks_by_subject": {"a": np.ones(10, bool)},
        "subjects": ["a"],
        "tau": 0.8,
        "adjacency_full": None,
        "k": 3,
        "seeds": [7],
    }
    monkeypatch.setattr(atlas, "hash", lambda _: 1, raising=False)
    atlas.strong_null_ari(**kwargs)
    monkeypatch.setattr(atlas, "hash", lambda _: 99, raising=False)
    atlas.strong_null_ari(**kwargs)
    assert np.array_equal(captured[0]["a"], captured[1]["a"])


def test_crossfit_configuration_declares_preprocessing_for_resume():
    path = ROOT / "scripts/individual_workflow/01_01_run_hcp_crossfit_template_matching.py"
    spec = importlib.util.spec_from_file_location("explicit_crossfit_settings", path)
    assert spec is not None and spec.loader is not None
    workflow = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = workflow
    spec.loader.exec_module(workflow)
    args = workflow.build_arg_parser().parse_args([
        "--feature-scaling", "none", "--sparsify-stage", "affinity", "--sparsity", "0.7",
        "--clustering-feature-scaling", "unit_sd",
    ])
    config = workflow.run_config_payload(args, ["01", "02"], {}, {"01":0,"02":1})
    assert config["gradient_preprocessing"] == adapter.gradient_preprocessing_settings(
        400, feature_scaling="none", sparsify_stage="affinity", sparsity=.7,
    )
    assert config["clustering_feature_scaling"] == "unit_sd"


@pytest.mark.parametrize("filename,function_name,extra", [
    ("01_03_run_and_render_hcp_crossfit_disjoint_folds.py", "fit_disjoint_cohort", {}),
    ("01_04_render_hcp_runs_comparison.py", "fit_split_half_template", {"split_id":"split-1"}),
])
def test_independent_comparison_fits_honor_the_same_options(tmp_path, monkeypatch, filename, function_name, extra):
    from types import SimpleNamespace  # noqa: I001
    from scipy import sparse
    path = ROOT / "scripts/individual_workflow" / filename
    spec = importlib.util.spec_from_file_location("explicit_"+function_name, path)
    assert spec is not None and spec.loader is not None
    driver = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = driver
    spec.loader.exec_module(driver)
    monkeypatch.setattr(driver, "DEFAULT_KONG2022_SURFACE_ROOT", tmp_path / "surface")
    monkeypatch.setattr(driver, "DEFAULT_KONG2022_ROUTEA_ROOT", tmp_path / "pfm")
    gradient = np.random.default_rng(9).normal(size=(8,5))
    source = np.random.default_rng(3).uniform(-.5,.5,size=(8,400))
    for subject in ("01", "02"):
        mask_path = tmp_path / f"surface/sub-{subject}/surface/validity/sub-{subject}_hemi-L_valid_mask.npy"
        mask_path.parent.mkdir(parents=True)
        np.save(mask_path, np.ones(8,dtype=bool))
        for run in (1,2):
            fc_path = tmp_path / f"pfm/sub-{subject}/fc/kong2022/2mm/hemi_L/runs/run-{run}/hipp_vertex_to_network_fc_active.npy"
            fc_path.parent.mkdir(parents=True)
            np.save(fc_path, source)
    seen = []

    def fit(matrix, **kwargs):
        seen.append(kwargs)
        return gradient, np.ones(5), np.ones(5)/5

    def cluster(**kwargs):
        assert np.allclose(kwargs["group_features"].std(axis=0),1)
        labels = np.array([1,1,1,2,2,2,3,3])
        return labels, labels

    cf = SimpleNamespace(
        _load_full_subject_fc=lambda *args:source,
        fit_diffusion_gradients=fit,
        reconstruct_full_surface=adapter.reconstruct_full_surface,
        build_cluster_prototypes=lambda *args,**kwargs:np.zeros((3,400)),
    )
    core = SimpleNamespace(
        load_surface=lambda path:(np.zeros((8,3)),np.array([[0,1,2]])),
        build_surface_adjacency=lambda *args:sparse.eye(8,format="csr"),
        cluster_from_precomputed_group_features=cluster,
        reorder_labels_by_long_axis=lambda labels, order:labels,
    )
    monkeypatch.setattr(driver,"long_axis_order",lambda path:np.arange(8))
    group = SimpleNamespace(group_mean_fisherz_fc=lambda **kwargs:source)
    options = {"feature_scaling":"none","sparsify_stage":"none","sparsity":None,
               "clustering_feature_scaling":"unit_sd"}
    result = getattr(driver,function_name)(
        subjects=["01","02"],smoothing="2mm",hemi="L",core_atlas_module=core,
        group_grad_module=group,cf_module=cf,surface_path=tmp_path/"surface.gii",
        gradient_options=options,**extra,
    )
    assert seen == [{"n_components":5,"feature_scaling":"none","sparsify_stage":"none","sparsity":None}]
    assert result["gradient_preprocessing"]["feature_scaling"] == "none"
    assert result["clustering_feature_transform"]["n_vertices_fit"] == 8


def test_partial_crossfit_preflight_declares_and_counts_its_alignment_anchor(tmp_path):
    path = ROOT / "scripts/individual_workflow/01_01_run_hcp_crossfit_template_matching.py"
    spec = importlib.util.spec_from_file_location("explicit_crossfit_anchor_audit", path)
    assert spec is not None and spec.loader is not None
    workflow = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = workflow
    spec.loader.exec_module(workflow)
    args = workflow.build_arg_parser().parse_args([
        "--shared-root", str(tmp_path/"pfm"), "--surface-root", str(tmp_path/"surface"),
        "--hippunfold-batch-root", str(tmp_path/"hippunfold"),
        "--template-surfaces-dir", str(tmp_path/"templates"),
        "--n-folds", "2", "--smoothings", "2mm", "--hemis", "L",
    ])
    surface = workflow._template_surface_path(args.template_surfaces_dir,"L")
    surface.parent.mkdir(parents=True)
    surface.touch()
    for subject in ("01","02"):
        for source in (
            workflow._subject_valid_mask_path(args.surface_root,subject,"L"),
            workflow._subject_surface_path(args.hippunfold_batch_root,subject,"L"),
            workflow._subject_fc_path(args.shared_root,subject,"2mm","L"),
        ):
            source.parent.mkdir(parents=True,exist_ok=True)
            source.touch()
    result = workflow.dry_run_payload(args,["01","02"],{"01":0,"02":1})
    expected = [str(workflow._subject_fc_path(args.shared_root,s,"4mm","L")) for s in ("01","02")]
    assert result["missing_inputs"] == expected
    assert result["template_fits"] == 4


def test_gradient_alignment_does_not_substitute_the_first_requested_combo():
    from types import SimpleNamespace
    producer = _stage("05_01_compute_hippomaps_gradients.py")
    with pytest.raises(ValueError, match="4mm.*L"):
        producer.select_alignment_anchor([SimpleNamespace(smoothing="2mm",hemi="R")])


def test_strong_null_ari_permutes_validity_masks_with_subject_fc(monkeypatch):
    atlas = _stage("05_02_run_group_atlas.py")
    captured = {}

    def fake_split(**kwargs):
        captured["feature_bank"] = kwargs["feature_bank"]
        captured["valid_masks_by_subject"] = kwargs["valid_masks_by_subject"]
        captured["gradient_options"] = kwargs["gradient_options"]
        return {"values": [0.0]}

    monkeypatch.setattr(atlas, "subject_split_ari", fake_split)
    fc_a = np.arange(20, dtype=np.float32).reshape(10, 2)
    mask_a = np.array([True, True, False, True, False, True, True, False, True, True])
    fc_a[~mask_a] = np.nan
    atlas.strong_null_ari(
        method="gradient-ward",
        feature_bank={"a": fc_a},
        valid_masks_by_subject={"a": mask_a},
        subjects=["a"],
        tau=1.0,
        adjacency_full=None,
        k=3,
        seeds=[20260420],
    )
    perm = captured["gradient_options"]["alignment_vertex_permutations"]["a"]
    assert np.array_equal(captured["valid_masks_by_subject"]["a"], mask_a[perm])
    assert np.array_equal(np.isfinite(captured["feature_bank"]["a"][:, 0]), captured["valid_masks_by_subject"]["a"])
    empty = atlas.strong_null_ari(
        method="gradient-ward",
        feature_bank={"a": fc_a},
        valid_masks_by_subject={"a": mask_a},
        subjects=["a"],
        tau=1.0,
        adjacency_full=None,
        k=3,
        seeds=[],
    )
    assert empty["values"] == [] and empty["n_requested"] == 0


def test_group_atlas_resume_validates_gradient_manifest_and_summary_settings(tmp_path):
    atlas = _stage("05_02_run_group_atlas.py")
    grad_dir = tmp_path / "gradients" / "pfm" / "2mm" / "hemi_L"
    grad_dir.mkdir(parents=True)
    grad_array = grad_dir / "hippomaps_gradients_full.npy"
    np.save(grad_array, np.ones((8, 5), dtype=np.float32))
    grad_manifest = grad_dir / "hippomaps_gradient_manifest.json"
    grad_manifest.write_text(json.dumps({
        "status": "ok",
        "group_fc_shape": [8, 400],
        "feature_count": 400,
        "gradient_preprocessing": adapter.gradient_preprocessing_settings(
            400, feature_scaling="col_zscore", sparsify_stage="features", sparsity=0.9,
        ),
        "outputs": {"hippomaps_gradients_full": str(grad_array)},
    }), encoding="utf-8")
    summary_path = tmp_path / "group_gradient_feature_summary.json"
    summary_path.write_text(json.dumps({
        "hippomaps_gradient_manifest": str(grad_manifest),
        "hippomaps_gradients_full": str(grad_array),
        "gradient_preprocessing": adapter.gradient_preprocessing_settings(
            400, feature_scaling="col_zscore", sparsify_stage="features", sparsity=0.9,
        ),
        "clustering_feature_transform": {"scaling": "none", "n_vertices_fit": 8},
    }), encoding="utf-8")
    options = {
        "feature_scaling": "col_zscore",
        "sparsify_stage": "features",
        "sparsity": 0.9,
        "clustering_feature_scaling": "none",
        "rbf_scale": 1.0,
    }
    recorded = atlas.gradient_resume_inputs(
        gradient_root=tmp_path / "gradients", cohort=None, variant="36p",
        gradient_options=options, route="pfm", smoothing="2mm", hemi="L",
    )
    sig = atlas.canonical_json_sha256(recorded)
    manifest_path = tmp_path / "group_atlas_manifest.json"
    manifest_path.write_text(json.dumps({
        "method": "gradient-spectral-rbf",
        "rbf_scale": 1.0,
        "feature_count": 400,
        "resume_input_signature_version": atlas.RESUME_SIGNATURE_VERSION,
        "resume_input_signature": sig,
        "resume_input_signature_payload": {"gradient_options": options},
        "group_gradient_feature_summary": str(summary_path),
        "gradient_source_manifest": str(grad_manifest),
    }), encoding="utf-8")
    ok, reason = atlas.resume_manifest_matches(manifest_path, sig, gradient_options=options)
    assert ok is True and reason == "signature_match"

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["rbf_scale"] = 1.000001
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="rbf_scale"):
        atlas.resume_manifest_matches(manifest_path, sig, gradient_options=options)
    payload["rbf_scale"] = 1.0
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    # Corrupting upstream gradient manifest preprocessing must raise an explicit regeneration error even on resume
    grad_manifest.write_text(json.dumps({
        "status": "ok",
        "group_fc_shape": [8, 400],
        "feature_count": 400,
        "outputs": {"hippomaps_gradients_full": str(grad_array)},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="gradient_preprocessing.*regenerate"):
        atlas.resume_manifest_matches(manifest_path, sig, gradient_options=options)


@pytest.mark.parametrize("case", ["missing_summary", "missing_source", "conflicting_source"])
def test_gradient_resume_rejects_missing_or_conflicting_provenance(tmp_path, case):
    atlas = _stage("05_02_run_group_atlas.py")
    options = {
        "feature_scaling": "none",
        "sparsify_stage": "affinity",
        "sparsity": 0.9,
        "clustering_feature_scaling": "none",
    }
    settings = adapter.gradient_preprocessing_settings(
        400, feature_scaling="none", sparsify_stage="affinity", sparsity=0.9,
    )
    gradient = {
        "group_fc_shape": [8, 400],
        "gradient_preprocessing": settings,
    }
    first_source = tmp_path / "first_gradient_manifest.json"
    second_source = tmp_path / "second_gradient_manifest.json"
    first_source.write_text(json.dumps(gradient), encoding="utf-8")
    second_source.write_text(json.dumps(gradient), encoding="utf-8")
    summary = {
        "gradient_preprocessing": settings,
        "clustering_feature_transform": {"scaling": "none"},
        "hippomaps_gradient_manifest": str(first_source),
    }
    if case == "missing_source":
        del summary["hippomaps_gradient_manifest"]
    summary_path = tmp_path / "group_gradient_feature_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    payload = {
        "status": "ok",
        "method": "gradient-ward",
        "resume_input_signature_version": atlas.RESUME_SIGNATURE_VERSION,
        "resume_input_signature": "matching",
        "resume_input_signature_payload": {"gradient_options": options},
        "group_gradient_feature_summary": str(summary_path),
        "gradient_source_manifest": str(first_source),
    }
    if case == "missing_summary":
        del payload["group_gradient_feature_summary"]
    elif case == "missing_source":
        del payload["gradient_source_manifest"]
    else:
        payload["gradient_source_manifest"] = str(second_source)
    manifest = tmp_path / "group_atlas_manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    field = "group_gradient_feature_summary" if case == "missing_summary" else "gradient_source_manifest"
    with pytest.raises(ValueError, match=field):
        atlas.resume_manifest_matches(manifest, "matching", gradient_options=options)


@pytest.mark.parametrize("status", ["skipped_no_valid_k", "skipped_no_k_survived_constraints"])
def test_gradient_resume_preserves_matching_skipped_results(tmp_path, status):
    atlas = _stage("05_02_run_group_atlas.py")
    options = {
        "feature_scaling": "none",
        "sparsify_stage": "affinity",
        "sparsity": 0.9,
        "clustering_feature_scaling": "none",
        "rbf_scale": 1.0,
    }
    settings = adapter.gradient_preprocessing_settings(
        400, feature_scaling="none", sparsify_stage="affinity", sparsity=0.9,
    )
    source = tmp_path / "hippomaps_gradient_manifest.json"
    source.write_text(json.dumps({
        "group_fc_shape": [8, 400],
        "gradient_preprocessing": settings,
    }), encoding="utf-8")
    summary = tmp_path / "group_gradient_feature_summary.json"
    if status == "skipped_no_k_survived_constraints":
        summary.write_text(json.dumps({
            "gradient_preprocessing": settings,
            "clustering_feature_transform": {"scaling": "none"},
            "hippomaps_gradient_manifest": str(source),
        }), encoding="utf-8")
    manifest = tmp_path / "group_atlas_manifest.json"
    manifest.write_text(json.dumps({
        "status": status,
        "method": "gradient-spectral-rbf",
        "resume_input_signature_version": atlas.RESUME_SIGNATURE_VERSION,
        "resume_input_signature": "matching",
        "resume_input_signature_payload": {"gradient_options": options},
        "group_gradient_feature_summary": str(summary),
    }), encoding="utf-8")

    assert atlas.resume_manifest_matches(manifest, "matching", gradient_options=options) == (True, "signature_match")
    assert atlas.resume_manifest_matches(manifest, "changed", gradient_options=options) == (False, "upstream_signature_changed")


def test_unprofiled_null_draws_new_permutations_for_every_repetition(monkeypatch):
    atlas = _stage("05_02_run_group_atlas.py")
    seen = []

    def record(**kwargs):
        seen.append(kwargs["gradient_options"]["alignment_vertex_permutations"])
        return {"values": [0.0], "n_requested": 1, "n_used": 1, "n_skipped_overlap": 0,
                "n_skipped_empty": 0, "n_skipped_failure": 0, "failure_examples": []}

    monkeypatch.setattr(atlas, "subject_split_ari", record)
    subjects = [f"{index:02d}" for index in range(30)]
    bank = {subject: np.arange(64, dtype=np.float32)[:, None] * np.ones((1, 4), np.float32) for subject in subjects}
    out = atlas.strong_null_ari(
        method="gradient-spectral", feature_bank=bank,
        valid_masks_by_subject={subject: np.ones(64, bool) for subject in subjects},
        subjects=subjects, tau=0.8, adjacency_full=None, k=3, seeds=[100, 117, 134],
    )
    assert out["n_requested"] == 3 and out["values"] == [0.0, 0.0, 0.0]
    drawn = [{tuple(perm) for perm in repetition.values()} for repetition in seen]
    assert all(len(repetition) == len(subjects) for repetition in drawn)
    assert not (drawn[0] & drawn[1]) and not (drawn[1] & drawn[2]) and not (drawn[0] & drawn[2])

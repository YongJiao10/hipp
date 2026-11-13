#!/usr/bin/env python3
# pyright: reportArgumentType=false, reportIndexIssue=false, reportAttributeAccessIssue=false, reportUndefinedVariable=false, reportCallIssue=false
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_module(relpath: str, alias: str):
    module_path = REPO_ROOT / relpath
    spec = importlib.util.spec_from_file_location(alias, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    if "lib/" in relpath:
        module.__package__ = "lib"
    core_dir = (REPO_ROOT / "scripts" / "core_workflow").resolve()
    if str(core_dir) not in sys.path:
        sys.path.insert(0, str(core_dir))
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


adapter = _load_module("scripts/core_workflow/lib/hippomaps_adapter.py", "core_hippomaps_adapter")
compute_mod = _load_module(
    "scripts/core_workflow/05_01_compute_hippomaps_gradients.py",
    "core_05_01_compute_hippomaps_gradients",
)
render_mod = _load_module(
    "scripts/core_workflow/06_04_render_hippomaps_gradient_products.py",
    "core_06_04_render_hippomaps_gradient_products",
)
atlas_mod = _load_module(
    "scripts/core_workflow/05_02_run_group_atlas.py",
    "core_05_02_run_group_atlas",
)
overview_mod = _load_module(
    "scripts/core_workflow/06_01_build_group_overview.py",
    "core_06_01_build_group_overview",
)
matrix_mod = _load_module(
    "scripts/core_workflow/07_01_build_group_matrix.py",
    "core_07_01_build_group_matrix",
)
cortex_mod = _load_module(
    "scripts/core_workflow/06_03_build_cluster_cortex_panels.py",
    "core_06_03_build_cluster_cortex_panels",
)
confidence_mod = _load_module(
    "scripts/core_workflow/06_06_render_confidence_maps.py",
    "core_06_06_render_confidence_maps",
)
route_tags = _load_module(
    "scripts/core_workflow/lib/route_tags.py",
    "core_route_tags",
)
output_layout = _load_module(
    "scripts/core_workflow/lib/output_layout.py",
    "core_output_layout",
)
smoke_mod = _load_module(
    "tests/core_workflow/smoke_prepare_run.py",
    "core_smoke_prepare_run",
)


@pytest.mark.parametrize("variant,branch", [("36p", "pfm")])
def test_cohort_gradient_producer_and_consumers_share_layout(tmp_path, monkeypatch, variant, branch):
    surface_dir = tmp_path / "surfaces"
    surface_dir.mkdir()
    (surface_dir / "group-average_hemi-L_space-corobl_den-512_label-hipp_midthickness.surf.gii").touch()
    support = np.ones(3, dtype=bool)
    gradients = np.arange(9, dtype=np.float32).reshape(3, 3)
    monkeypatch.setattr(compute_mod, "read_surface_vertex_count", lambda _path: 3)
    monkeypatch.setattr(compute_mod, "subject_valid_mask", lambda *_args: support)
    monkeypatch.setattr(compute_mod, "load_subject_fc", lambda *_args, **_kwargs: (np.zeros((3, 400)), tmp_path / "fc.npy"))
    monkeypatch.setattr(compute_mod, "load_kong2022_roi_fc_contract", lambda *_args, **_kwargs: {"roi_names": [str(i) for i in range(400)]})
    monkeypatch.setattr(compute_mod, "fit_diffusion_gradients", lambda *_args, **_kwargs: (gradients, np.ones(3), np.ones(3) / 3))

    result = compute_mod.compute_combo(
        route="pfm", signal_branch="standard", shared_root=tmp_path, reference_root=tmp_path,
        atlas="kong2022", surface_root=tmp_path, subjects=["01"], subjects_manifest=tmp_path / "subjects.txt",
        smoothing="2mm", hemi="L", coverage_threshold=0.8, group_average_surfaces_dir=surface_dir,
        out_root=tmp_path / "gradients", hippomaps_root=tmp_path, n_components=3,
        selected_components=[1, 2, 3], force=False,
        cohort="nthc", variant=variant,
    )
    run_dir = tmp_path / "gradients" / branch / "nthc/2mm/hemi_L"
    assert result.run_dir == run_dir
    run_dir.mkdir(parents=True)
    np.save(run_dir / "hippomaps_gradients_full.npy", result.gradients_full)
    (run_dir / "hippomaps_gradient_manifest.json").write_text(atlas_mod.json.dumps({"gradient_preprocessing": adapter.gradient_preprocessing_settings(400),
        "status": "ok", "route": "pfm", "signal_branch": "standard", "cohort": "nthc", "variant": variant,
        "group_fc_shape": [3, 400], "feature_count": 400,
        "outputs": {"hippomaps_gradients_full": str(run_dir / "hippomaps_gradients_full.npy")},
    }), encoding="utf-8")
    selected, _ = atlas_mod.load_hippomaps_gradient_features(
        gradient_root=tmp_path / "gradients", route="pfm", smoothing="2mm", hemi="L",
        support_mask=support, cohort="nthc", variant=variant,
    )
    assert np.array_equal(selected, gradients)
    displayed = render_mod.load_gradient_arrays_for_display(
        gradient_root=tmp_path / "gradients", route="pfm", smoothing="2mm", hemis=["L"],
        cohort="nthc", variant=variant,
    )
    assert np.array_equal(displayed["L"], gradients)


def test_gradient_resume_fingerprints_the_profile_bank_the_run_reads(tmp_path):
    recorded = atlas_mod.gradient_resume_inputs(
        gradient_root=tmp_path, cohort=None, variant="36p",
        route="pfm", smoothing="2mm", hemi="L", profile="topsim",
    )
    paths = [Path(item["path"]) for item in recorded["gradient_artifact_inputs"]]
    assert {path.parent for path in paths} == {tmp_path / "pfm" / "topsim" / "2mm" / "hemi_L"}


@pytest.mark.parametrize("cohort,variant,can_resume", [
    ("nthc", "36p", True), ("tehc", "36p", False),
])
def test_gradient_resume_rejects_a_different_cohort_or_variant(tmp_path, cohort, variant, can_resume):
    recorded = atlas_mod.gradient_resume_inputs(
        gradient_root=tmp_path, cohort="nthc", variant="36p",
    )
    requested = atlas_mod.gradient_resume_inputs(
        gradient_root=tmp_path, cohort=cohort, variant=variant,
    )
    manifest = tmp_path / "group_atlas_manifest.json"
    manifest.write_text(atlas_mod.json.dumps({
        "resume_input_signature_version": atlas_mod.RESUME_SIGNATURE_VERSION,
        "resume_input_signature": atlas_mod.canonical_json_sha256(recorded),
    }), encoding="utf-8")
    matches, _ = atlas_mod.resume_manifest_matches(manifest, atlas_mod.canonical_json_sha256(requested))
    assert matches is can_resume


def test_group_mean_fc_applies_fisher_z_before_averaging() -> None:
    support = np.asarray([True, False, True], dtype=bool)
    subj_a = np.asarray(
        [
            [0.0, 0.5],
            [0.9, 0.9],
            [np.nan, -0.5],
        ],
        dtype=np.float32,
    )
    subj_b = np.asarray(
        [
            [0.5, 0.0],
            [0.8, 0.8],
            [0.5, np.nan],
        ],
        dtype=np.float32,
    )
    got = compute_mod.group_mean_fisherz_fc(
        feature_bank={"a": subj_a, "b": subj_b},
        subjects=["a", "b"],
        support_mask=support,
    )
    expected = np.nanmean(
        np.stack(
            [
                np.arctanh(np.clip(subj_a[support], -0.999999, 0.999999)),
                np.arctanh(np.clip(subj_b[support], -0.999999, 0.999999)),
            ],
            axis=0,
        ),
        axis=0,
    )
    expected[~np.isfinite(expected)] = 0.0
    assert np.allclose(got, expected.astype(np.float32), atol=1e-6), (got, expected)


def test_control_routes_use_routea_reference_root_without_fallback_search() -> None:
    shared_parent = Path("/data/inputs/shared_inputs")
    ctxreg_root = shared_parent / "pfm_ctxreg"
    ctxdrop_root = shared_parent / "pfm_ctxdrop"
    assert compute_mod.resolve_gradient_shared_root(shared_parent, "pfm_ctxreg") == ctxreg_root.resolve()
    assert compute_mod.resolve_gradient_reference_root(shared_parent, ctxreg_root, "pfm_ctxreg") == (shared_parent / "pfm").resolve()
    assert compute_mod.resolve_gradient_reference_root(shared_parent, ctxdrop_root, "pfm_ctxdrop") == (shared_parent / "pfm").resolve()
    explicit = Path("/data/inputs/references/pfm")
    assert compute_mod.resolve_gradient_reference_root(shared_parent, ctxreg_root, "pfm_ctxreg", explicit) == explicit.resolve()


def test_gradient_signal_branch_is_explicit_only_for_routea_controls() -> None:
    assert compute_mod.resolve_signal_branch_for_route("pfm") == "standard"
    assert compute_mod.resolve_signal_branch_for_route("pfm_ctxreg") == "ctxreg"
    with pytest.raises(ValueError, match="Signal branch/route mismatch"):
        compute_mod.resolve_signal_branch_for_route("pfm", "ctxreg")
    with pytest.raises(ValueError, match="only PFM controls"):
        compute_mod.resolve_signal_branch_for_route("lynch2024_20net", "ctxreg")


def test_group_atlas_signal_branch_is_explicit_only_for_routea_controls() -> None:
    assert atlas_mod.resolve_signal_branch_for_route("pfm") == "standard"
    assert atlas_mod.resolve_signal_branch_for_route("pfm_ctxreg") == "ctxreg"
    with pytest.raises(ValueError, match="Signal branch/route mismatch"):
        atlas_mod.resolve_signal_branch_for_route("pfm", "ctxreg")
    with pytest.raises(ValueError, match="only PFM controls"):
        atlas_mod.resolve_signal_branch_for_route("lynch2024_7net", "ctxreg")


def test_lynch20_fc_rejects_permuted_consumed_network_order(tmp_path: Path) -> None:
    order = compute_mod.lynch2024_20net_order(compute_mod.DEFAULT_LYNCH2024_MERGE_CONFIG)
    fc_dir = tmp_path / "sub-1001" / "fc" / "lynch2024_20net" / "2mm" / "hemi_L"
    fc_dir.mkdir(parents=True)
    np.save(fc_dir / "hipp_vertex_to_network_fc.npy", np.ones((3, 20), dtype=np.float32))
    (fc_dir / "stage_manifest.json").write_text(
        compute_mod.json.dumps(
            {
                "network_taxonomy": "lynch20",
                "network_count": 20,
                "noise_excluded": True,
                "network_order": [order[1], order[0], *order[2:]],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="network order mismatch"):
        compute_mod.load_subject_fc(
            tmp_path,
            "1001",
            "lynch2024_20net",
            "2mm",
            "L",
            np.ones((3,), dtype=bool),
        )


def test_gradientmaps_called_with_official_pearson_kernel_and_full_lambda_ev() -> None:
    calls: list[dict[str, object]] = []
    fake_gradients = np.arange(1, 13, dtype=np.float64).reshape(4, 3)
    raw = np.asarray(
        [
            [1.0, 10.0, 3.0, 5.0],
            [2.0, 20.0, 3.0, 5.0],
            [3.0, 30.0, 3.0, 5.0],
            [4.0, 40.0, 3.0, 5.0],
        ],
        dtype=np.float32,
    )

    def fake_affinity(matrix, **kwargs):
        calls.append({"affinity_matrix": np.asarray(matrix).copy(), "affinity_kwargs": kwargs})
        return np.ones((matrix.shape[0], matrix.shape[0]))

    def fake_embedding(affinity, **kwargs):
        calls.append({"embedding_matrix": np.asarray(affinity).copy(), "embedding_kwargs": kwargs})
        return fake_gradients, np.asarray([5.0, 3.0, 2.0])

    with patch.object(adapter, "diffusion_map_embedding", fake_embedding), patch(
        "brainspace.gradient.kernels.compute_affinity", fake_affinity
    ):
        gradients, lambdas, ev = adapter.fit_diffusion_gradients(
            raw,
            n_components=3,
        )
    assert calls[0]["affinity_kwargs"] == {"kernel":"pearson", "sparsity":0.0, "pre_sparsify":True, "non_negative":True}
    assert np.allclose(calls[0]["affinity_matrix"], adapter.feature_zscore_columns(raw))
    assert calls[1]["embedding_kwargs"] == {"n_components": 3}
    assert np.array_equal(calls[1]["embedding_matrix"], np.ones((4,4)))
    assert gradients.shape == (4, 3), gradients.shape
    assert np.allclose(lambdas, [5.0, 3.0, 2.0])
    assert np.allclose(ev, [0.5, 0.3, 0.2])


def test_gradientmaps_low_dimensional_sparsity_uses_full_graph() -> None:
    assert adapter.brainspace_sparsity_for_feature_count(8) == 0.0
    assert adapter.brainspace_sparsity_for_feature_count(20) == 0.0
    assert adapter.brainspace_sparsity_for_feature_count(400) == 0.9


def test_gradient_fit_has_no_seed_argument() -> None:
    with pytest.raises(TypeError, match="random_state"):
        adapter.fit_diffusion_gradients(np.eye(6, dtype=np.float32), n_components=2, random_state=17)


def test_group_atlas_gradient_spectral_uses_hippomaps_adapter_and_records_provenance() -> None:
    feature_bank = {
        "a": np.asarray(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
                [1.0, 1.1, 1.2],
                [1.3, 1.4, 1.5],
                [1.6, 1.7, 1.8],
                [1.9, 2.0, 2.1],
            ],
            dtype=np.float32,
        ),
        "b": np.asarray(
            [
                [0.2, 0.1, 0.0],
                [0.6, 0.5, 0.4],
                [1.0, 0.9, 0.8],
                [1.4, 1.3, 1.2],
                [1.8, 1.7, 1.6],
                [2.2, 2.1, 2.0],
                [2.6, 2.5, 2.4],
            ],
            dtype=np.float32,
        ),
    }
    support_mask = np.asarray([True, True, False, True, True, True, True], dtype=bool)
    calls: dict[str, np.ndarray | int] = {}

    def fake_fit(group_fc: np.ndarray, *, n_components: int, **kwargs):
        calls["group_fc"] = np.asarray(group_fc, dtype=np.float32).copy()
        calls["n_components"] = int(n_components)
        gradients = np.asarray(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0, 9.0, 10.0],
                [11.0, 12.0, 13.0, 14.0, 15.0],
                [16.0, 17.0, 18.0, 19.0, 20.0],
                [21.0, 22.0, 23.0, 24.0, 25.0],
                [26.0, 27.0, 28.0, 29.0, 30.0],
            ],
            dtype=np.float32,
        )
        lambdas = np.asarray([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float64)
        ev = lambdas / np.sum(lambdas)
        return gradients, lambdas, ev

    expected_group_fc = np.nanmean(
        np.stack([feature_bank["a"][support_mask], feature_bank["b"][support_mask]], axis=0),
        axis=0,
    ).astype(np.float32)

    with (
        patch.object(atlas_mod, "ensure_hippomaps_on_path", return_value=Path("/mock/hippomaps")),
        patch.object(atlas_mod, "hippomaps_import_path", return_value="/mock/hippomaps/hippomaps"),
        patch.object(atlas_mod, "hippomaps_commit", return_value="deadbeef"),
    ):
        selected, meta = atlas_mod.build_group_gradient_features(
            feature_bank=feature_bank,
            subjects_subset=["a", "b"],
            support_mask=support_mask,
            gradient_components=(1, 3),
            hippomaps_root=Path("/mock/hippomaps"),
            gradient_fit_fn=fake_fit,
        )

    assert int(calls["n_components"]) == atlas_mod.GRADIENT_N_COMPONENTS_COMPUTED
    assert np.allclose(calls["group_fc"], expected_group_fc, atol=1e-6)
    assert selected.shape == (6, 2), selected.shape
    assert np.allclose(selected[:, 0], [1.0, 6.0, 11.0, 16.0, 21.0, 26.0], atol=1e-6)
    assert np.allclose(selected[:, 1], [3.0, 8.0, 13.0, 18.0, 23.0, 28.0], atol=1e-6)
    assert "compute_affinity" in meta["gradient_embedding_call"]
    assert "diffusion_map_embedding" in meta["gradient_embedding_call"]
    assert meta["gradient_preprocessing"]["feature_scaling"] == "col_zscore"
    assert np.isclose(meta["brainspace_sparsity"], 0.0), meta
    assert meta["gradient_preprocessing"]["eigensolver"] == adapter.GRADIENT_EIGENSOLVER
    assert "gradient_random_state" not in meta
    assert meta["feature_zscore_pre_gradientmaps"] == "col_zscore"
    assert meta["gradient_component_labels"] == ["G1", "G3"]
    assert meta["selected_components"] == ["G1", "G3"]
    assert meta["hippomaps_repo"] == "/mock/hippomaps"
    assert meta["hippomaps_import_path"] == "/mock/hippomaps/hippomaps"
    assert meta["hippomaps_commit"] == "deadbeef"
    assert np.allclose(meta["gradient_eigenvalues"], [5.0, 4.0, 3.0, 2.0, 1.0], atol=1e-6)
    assert np.allclose(meta["ev_fractions"], [1.0 / 3.0, 4.0 / 15.0, 1.0 / 5.0, 2.0 / 15.0, 1.0 / 15.0], atol=1e-6)



def test_gradient_split_diagnostic_failures_are_counted_when_allowed() -> None:
    subjects = ["a", "b", "c", "d"]
    support = np.ones((5,), dtype=bool)
    feature_bank = {s: np.zeros((5, 2), dtype=np.float32) for s in subjects}
    valid_masks = {s: support.copy() for s in subjects}
    adjacency = atlas_mod.sparse.csr_matrix(np.ones((5, 5), dtype=np.float32))

    with patch.object(
        atlas_mod,
        "cluster_for_subset",
        side_effect=RuntimeError("GradientMaps lambdas contain non-finite values"),
    ):
        out = atlas_mod.subject_split_ari(
            method="gradient-spectral",
            feature_bank=feature_bank,
            valid_masks_by_subject=valid_masks,
            subjects=subjects,
            tau=0.8,
            adjacency_full=adjacency,
            k=2,
            seeds=[11, 22],
            allow_diagnostic_failures=True,
        )

    assert out["values"] == []
    assert out["n_requested"] == 2
    assert out["n_used"] == 0
    assert out["n_skipped_failure"] == 2
    assert "GradientMaps lambdas" in out["failure_examples"][0]

    with patch.object(
        atlas_mod,
        "cluster_for_subset",
        side_effect=RuntimeError("GradientMaps lambdas contain non-finite values"),
    ):
        try:
            atlas_mod.subject_split_ari(
                method="gradient-spectral",
                feature_bank=feature_bank,
                valid_masks_by_subject=valid_masks,
                subjects=subjects,
                tau=0.8,
                adjacency_full=adjacency,
                k=2,
                seeds=[11],
                allow_diagnostic_failures=False,
            )
        except RuntimeError as exc:
            assert "GradientMaps lambdas" in str(exc)
        else:
            raise AssertionError("strict diagnostic mode should propagate gradient failures")


def test_evaluate_k_candidate_records_no_valid_gradient_diagnostic_status() -> None:
    combo = atlas_mod.Combo(
        method="gradient-spectral",
        atlas="lynch2024_20net",
        smoothing="2mm",
        hemi="L",
        coverage_threshold=0.8,
    )
    support = np.ones((5,), dtype=bool)
    group_features = np.asarray(
        [
            [1.0, 0.0],
            [0.8, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
            [0.2, 0.7],
        ],
        dtype=np.float32,
    )
    labels_active = np.asarray([1, 1, 2, 2, 2], dtype=np.int32)
    labels_full = labels_active.copy()
    adjacency = atlas_mod.sparse.csr_matrix(np.ones((5, 5), dtype=np.float32))
    no_split = {
        "values": [],
        "n_requested": 1,
        "n_used": 0,
        "n_skipped_overlap": 0,
        "n_skipped_empty": 0,
        "n_skipped_failure": 1,
        "failure_examples": ["GradientMaps lambdas contain non-finite values"],
    }
    no_run = {"values": [], "n_pairs": 0, "n_used": 0, "n_skipped_failure": 4, "failure_examples": []}

    with (
        patch.object(atlas_mod, "cluster_from_precomputed_group_features", return_value=(labels_active, labels_full)),
        patch.object(atlas_mod, "subject_split_ari", return_value=dict(no_split)),
        patch.object(atlas_mod, "strong_null_ari", return_value={**dict(no_split), "permutation_scope": "subject_specific"}),
        patch.object(atlas_mod, "run_aware_ari", return_value=no_run),
    ):
        k, labels, row = atlas_mod.evaluate_k_candidate(
            combo=combo,
            method="gradient-spectral",
            k=2,
            seed=101,
            n_support=5,
            v_min=1,
            group_features_active=group_features,
            support_mask=support,
            adjacency_full=adjacency,
            adjacency_support=adjacency,
            feature_bank={},
            valid_masks={},
            subjects=["a", "b", "c", "d"],
            tau=0.8,
            split_seeds=[11],
            null_seeds=[22],
            null_resamples=1,
            run_feature_banks={},
            seed_base=20260420,
            max_components_threshold=3,
            min_profile_coherence_threshold=0.0,
            k_selection_mode="redundancy-aware",
            roi_fc_bank_for_redundancy=None,
            allow_diagnostic_failures=True,
        )

    assert k == 2
    assert labels.tolist() == labels_full.tolist()
    assert np.isnan(row["group_ari_mean"])
    assert np.isnan(row["group_ari_se"])
    assert row["group_ari_diagnostic_status"] == "no_valid_subject_split_resamples"
    assert row["subject_split_skipped_failure"] == 1
    assert row["run_aware_ari_status"] == "no_valid_run_pairs"
    assert row["run_aware_skipped_failure"] == 4
    assert row["B_null_skipped_failure_strong"] == 1


def test_run_aware_ari_pairs_only_successful_runs_after_skipped_failures() -> None:
    support = np.ones((4,), dtype=bool)
    adjacency = atlas_mod.sparse.csr_matrix(np.ones((4, 4), dtype=np.float32))
    full_by_rid = {
        1: np.asarray([1, 1, 2, 2], dtype=np.int32),
        3: np.asarray([1, 2, 2, 1], dtype=np.int32),
    }

    def fake_cluster_for_subset(**kwargs):
        seed = int(kwargs["seed"])
        rid = (seed - 20260420 - 2) // 1000
        if rid == 2:
            raise RuntimeError("GradientMaps lambdas contain non-finite values")
        full = full_by_rid[int(rid)]
        return full[support], full

    with patch.object(atlas_mod, "cluster_for_subset", side_effect=fake_cluster_for_subset):
        out = atlas_mod.run_aware_ari(
            method="gradient-spectral",
            run_feature_banks={1: {}, 2: {}, 3: {}},
            subjects=["a", "b", "c", "d"],
            support_mask=support,
            adjacency_full=adjacency,
            k=2,
            seed_base=20260420,
            allow_diagnostic_failures=True,
        )

    assert out["n_skipped_failure"] == 1
    assert out["n_pairs"] == 1
    assert out["n_used"] == 1
    assert len(out["values"]) == 1

def test_group_atlas_public_methods_drop_partial_corr_branches() -> None:
    assert "group-network-partial-spectral" not in atlas_mod.METHOD_CHOICES
    assert "group-intrinsic-partial-spectral" not in atlas_mod.METHOD_CHOICES
    assert not atlas_mod.is_network_method("group-network-partial-spectral")


def test_group_atlas_parser_defaults_to_primary_k3_without_forcing_selection() -> None:
    parser = atlas_mod.build_arg_parser()
    args = parser.parse_args([])
    assert int(args.k_min) == 2
    assert int(args.k_max) == 8
    assert args.force_final_k is None
    assert int(args.default_k) == 3
    assert not hasattr(args, "surface_affinity")


def test_group_atlas_parser_rejects_retired_surface_affinity_option() -> None:
    with pytest.raises(SystemExit):
        atlas_mod.build_arg_parser().parse_args(
            ["--surface-affinity", "relu-cosine"]
        )


def test_group_atlas_gradient_spectral_loads_hippomaps_gradient_artifacts() -> None:
    with TemporaryDirectory() as td:
        root = Path(td)
        run_dir = root / "routea" / "2mm" / "hemi_L"
        run_dir.mkdir(parents=True)
        gradients_full = np.asarray(
            [
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0, 9.0, 10.0],
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [11.0, 12.0, 13.0, 14.0, 15.0],
            ],
            dtype=np.float32,
        )
        np.save(run_dir / "hippomaps_gradients_full.npy", gradients_full)
        manifest = {"gradient_preprocessing": adapter.gradient_preprocessing_settings(400),
            "status": "ok",
            "route": "routea",
            "group_fc_shape": [5, 400],
            "feature_count": 400,
            "smoothing": "2mm",
            "hemi": "L",
            "n_components_computed": 5,
            "gradient_embedding_call": "brainspace.gradient.GradientMaps(n_components=N, kernel='pearson').fit(column_zscored_group_mean_fisherz_fc)",
            "gradient_alignment": {
                "enabled": True,
                "anchor_route": "routea",
                "anchor_smoothing": "4mm",
                "anchor_hemi": "L",
                "raw_arrays_saved": False,
            },
            "outputs": {
                "hippomaps_gradients_full": str((run_dir / "hippomaps_gradients_full.npy").resolve()),
            },
        }
        (run_dir / "hippomaps_gradient_manifest.json").write_text(
            atlas_mod.json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        support_mask = np.asarray([False, True, True, False, True], dtype=bool)

        selected, meta = atlas_mod.load_hippomaps_gradient_features(
            gradient_root=root,
            route="routea",
            smoothing="2mm",
            hemi="L",
            support_mask=support_mask,
            gradient_components=(1, 3),
        )

    assert selected.shape == (3, 2), selected.shape
    assert np.allclose(selected[:, 0], [1.0, 6.0, 11.0], atol=1e-6)
    assert np.allclose(selected[:, 1], [3.0, 8.0, 13.0], atol=1e-6)
    assert meta["gradient_source"] == "hippomaps_gradient_artifacts"
    assert meta["gradient_component_labels"] == ["G1", "G3"]
    assert meta["gradient_alignment"]["enabled"] is True
    assert meta["gradient_alignment"]["raw_arrays_saved"] is False
    assert meta["hippomaps_gradient_manifest"].endswith("hippomaps_gradient_manifest.json")


def test_group_atlas_gradient_loader_rejects_signal_branch_mismatch() -> None:
    with TemporaryDirectory() as td:
        root = Path(td)
        run_dir = root / "routea_ctxreg" / "2mm" / "hemi_L"
        run_dir.mkdir(parents=True)
        gradients_full = np.ones((2, 3), dtype=np.float32)
        np.save(run_dir / "hippomaps_gradients_full.npy", gradients_full)
        manifest = {"gradient_preprocessing": adapter.gradient_preprocessing_settings(400),
            "status": "ok",
            "route": "routea_ctxreg",
            "signal_branch": "standard",
            "group_fc_shape": [2, 400],
            "feature_count": 400,
            "outputs": {
                "hippomaps_gradients_full": str((run_dir / "hippomaps_gradients_full.npy").resolve()),
            },
        }
        (run_dir / "hippomaps_gradient_manifest.json").write_text(
            atlas_mod.json.dumps(manifest), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="signal branch mismatch"):
            atlas_mod.load_hippomaps_gradient_features(
                gradient_root=root,
                route="routea_ctxreg",
                smoothing="2mm",
                hemi="L",
                support_mask=np.asarray([True, True], dtype=bool),
                gradient_components=(1, 2),
                expected_signal_branch="ctxreg",
            )




def test_group_atlas_gradient_loader_rejects_stale_manifest_path() -> None:
    with TemporaryDirectory() as td:
        root = Path(td)
        run_dir = root / "routea" / "2mm" / "hemi_L"
        run_dir.mkdir(parents=True)
        np.save(run_dir / "hippomaps_gradients_full.npy", np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32))
        stale_path = root / "old_location" / "hippomaps_gradients_full.npy"
        manifest = {"gradient_preprocessing": adapter.gradient_preprocessing_settings(400),
            "status": "ok",
            "group_fc_shape": [1, 400],
            "feature_count": 400,
            "n_components_computed": 3,
            "outputs": {"hippomaps_gradients_full": str(stale_path.resolve())},
        }
        (run_dir / "hippomaps_gradient_manifest.json").write_text(atlas_mod.json.dumps(manifest), encoding="utf-8")
        with pytest.raises(FileNotFoundError, match="points to a missing"):
            atlas_mod.load_hippomaps_gradient_features(
                gradient_root=root,
                route="routea",
                smoothing="2mm",
                hemi="L",
                support_mask=np.asarray([True], dtype=bool),
                gradient_components=(1, 3),
            )

def test_canonical_public_surfaces_drop_partial_corr_branches() -> None:
    for mod in (atlas_mod, overview_mod, cortex_mod):
        assert "group-network-partial-spectral" not in mod.METHOD_CHOICES
        assert "group-intrinsic-partial-spectral" not in mod.METHOD_CHOICES


def test_canonical_present_defaults_use_outputs_hcp_layout() -> None:
    expected_root = REPO_ROOT / "outputs_hcp" / "present"
    assert overview_mod.DEFAULT_PRESENT_DIR == expected_root / "overviews"
    assert matrix_mod.DEFAULT_PRESENT_DIR == expected_root / "overviews"
    assert cortex_mod.DEFAULT_PRESENT_DIR == expected_root / "cortex_panels"
    assert render_mod.DEFAULT_PRESENT_ROOT == expected_root / "gradients"
    for path in (
        overview_mod.DEFAULT_PRESENT_DIR,
        matrix_mod.DEFAULT_PRESENT_DIR,
        cortex_mod.DEFAULT_PRESENT_DIR,
        render_mod.DEFAULT_PRESENT_ROOT,
    ):
        text = str(path)
        assert "present_group" not in text
        assert "present_fixedk" not in text
        assert "t1only" not in text
        assert "hcp7t_hippocampus" not in text
        assert "/runs/" not in text
        assert "/manual/" not in text


def test_run_layout_keeps_smoke_outputs_in_dedicated_run_tree() -> None:
    layout = output_layout.run_layout("smoke5_test")
    assert layout["run_root"] == REPO_ROOT / "outputs_hcp" / "runs" / "smoke5_test"
    assert layout["overviews"] == layout["present"] / "overviews"
    assert layout["gradient_products"] == layout["present"] / "gradients"
    assert layout["cortex_panels"] == layout["present"] / "cortex_panels"


def test_smoke_command_plan_uses_isolated_fixedk_full_layout() -> None:
    base = REPO_ROOT / "outputs_hcp" / "runs" / "smoke5_test"
    present = base / "present"
    layout = {
        "run_root": base,
        "inputs": base / "inputs",
        "logs": base / "logs",
        "manifests": base / "manifests",
        "hippunfold": base / "hippunfold",
        "features": base / "features",
        "group_atlas": base / "group_atlas",
        "gradients": base / "gradients",
        "present": present,
        "overviews": present / "overviews",
        "cortex_panels": present / "cortex_panels",
        "gradient_products": present / "gradients",
    }
    records = smoke_mod.build_command_plan(
        subjects=["100610", "102311", "104416", "105923", "108323"],
        layout=layout,
        python_exe="/home/yojiao/.conda/envs/py314/bin/python",
        input_dir=REPO_ROOT / "data" / "hippunfold_input",
        bold_root=REPO_ROOT / "bold",
    )
    joined = "\n".join(" ".join(row["command"]) for row in records)
    assert "present_group" not in joined
    assert "present_fixedk" not in joined
    assert "10perc" not in joined
    assert "--default-k 3" in joined
    assert "--force-final-k 3" not in joined
    assert "--k-min 2 --k-max 8" in joined
    assert str(layout["overviews"]) in joined
    assert "group-network-partial-spectral" not in joined
    assert "group-intrinsic-partial-spectral" not in joined

def test_compute_defaults_to_five_components_like_hippomaps_paper() -> None:
    with patch.object(sys, "argv", ["05_01_compute_hippomaps_gradients.py"]):
        args = compute_mod.parse_args()
    assert int(args.n_components) == 5


def test_full_surface_reconstruction_keeps_nan_outside_support() -> None:
    active = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    support = np.asarray([False, True, False, True], dtype=bool)
    full = adapter.reconstruct_full_surface(active, support, vertex_count=4)
    assert full.shape == (4, 2), full.shape
    assert math.isnan(float(full[0, 0])), full
    assert math.isnan(float(full[2, 1])), full
    assert np.allclose(full[support], active)


def test_gradient_alignment_reorders_components_and_flips_sign_to_anchor() -> None:
    anchor_full = np.asarray(
        [
            [-1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, -1.0],
        ],
        dtype=np.float32,
    )
    candidate_full = np.column_stack([-anchor_full[:, 1], anchor_full[:, 0]]).astype(np.float32)
    aligned_active, aligned_full, aligned_lambdas, aligned_ev, meta = compute_mod.align_gradient_components_to_reference(
        active=candidate_full.copy(),
        full=candidate_full.copy(),
        lambdas=np.asarray([20.0, 10.0], dtype=np.float64),
        ev_fractions=np.asarray([2.0 / 3.0, 1.0 / 3.0], dtype=np.float64),
        reference_full=anchor_full,
        component_labels=["G1", "G2"],
    )

    assert np.allclose(aligned_active, anchor_full, atol=1e-6), aligned_active
    assert np.allclose(aligned_full, anchor_full, atol=1e-6), aligned_full
    assert np.allclose(aligned_lambdas, [10.0, 20.0], atol=1e-6), aligned_lambdas
    assert np.allclose(aligned_ev, [1.0 / 3.0, 2.0 / 3.0], atol=1e-6), aligned_ev
    assert meta["component_mapping"] == {"G1": "G2", "G2": "G1"}, meta
    assert meta["sign_flip_applied"] == {"G1": False, "G2": True}, meta
    assert meta["raw_arrays_saved"] is False, meta


def test_gradient_alignment_anchor_combo_is_identity_without_raw_outputs() -> None:
    anchor_full = np.asarray(
        [
            [1.0, -2.0, 0.0],
            [2.0, -1.0, 1.0],
            [3.0, 0.0, 2.0],
            [4.0, 1.0, 3.0],
        ],
        dtype=np.float32,
    )
    aligned_active, aligned_full, aligned_lambdas, aligned_ev, meta = compute_mod.align_gradient_components_to_reference(
        active=anchor_full.copy(),
        full=anchor_full.copy(),
        lambdas=np.asarray([5.0, 4.0, 3.0], dtype=np.float64),
        ev_fractions=np.asarray([5.0 / 12.0, 4.0 / 12.0, 3.0 / 12.0], dtype=np.float64),
        reference_full=anchor_full,
        component_labels=["G1", "G2", "G3"],
    )

    assert np.allclose(aligned_active, anchor_full, atol=1e-6), aligned_active
    assert np.allclose(aligned_full, anchor_full, atol=1e-6), aligned_full
    assert np.allclose(aligned_lambdas, [5.0, 4.0, 3.0], atol=1e-6), aligned_lambdas
    assert np.allclose(aligned_ev, [5.0 / 12.0, 4.0 / 12.0, 3.0 / 12.0], atol=1e-6), aligned_ev
    assert meta["component_mapping"] == {"G1": "G1", "G2": "G2", "G3": "G3"}, meta
    assert meta["sign_flip_applied"] == {"G1": False, "G2": False, "G3": False}, meta
    assert meta["raw_arrays_saved"] is False, meta


def test_render_ev_plot_sets_integer_ticks(tmp_path: Path) -> None:
    manifests = {
        "L": {"ev_fractions": [0.4, 0.2, 0.15, 0.1, 0.05]},
        "R": {"ev_fractions": [0.45, 0.2, 0.15, 0.08, 0.07]},
    }
    out_png = tmp_path / "test_ev.png"
    res = render_mod.render_ev_plot(manifests, out_png)
    assert res.exists()
    assert res.stat().st_size > 0

    out_custom = tmp_path / "test_ev_custom.png"
    res_custom = render_mod.render_ev_plot(manifests, out_custom, title="Custom Title")
    assert res_custom.exists()


def test_render_gradient_figure_calls_official_hippomaps_plotter_options(tmp_path) -> None:
    calls: list[dict[str, object]] = []

    class FakeFigure:
        def savefig(self, path, **kwargs):
            calls.append({"savefig": Path(path).name, "kwargs": kwargs})
            Image.new("RGB", (24, 24), (255, 255, 255)).save(path)

    def fake_plotter(cdata, **kwargs):
        calls.append({"shape": tuple(cdata.shape), "kwargs": kwargs})
        return FakeFigure()

    out = tmp_path / "hippomaps_renderer_test.png"
    render_mod.render_gradient_figure(
        left=np.ones((4, 3), dtype=np.float32),
        right=np.zeros((4, 3), dtype=np.float32),
        output_png=out,
        den="512",
        plotter=fake_plotter,
    )
    assert calls[0]["shape"] == (4, 2, 3), calls
    assert calls[0]["kwargs"]["cmap"] == "plasma", calls
    assert calls[0]["kwargs"]["share"] == "row", calls
    assert calls[0]["kwargs"]["color_bar"] == "right", calls
    assert calls[0]["kwargs"]["unfoldAPrescale"] is False, calls
    assert calls[0]["kwargs"]["tighten_cwindow"] is True, calls
    assert calls[0]["kwargs"]["labels"] == ["hipp"], calls
    assert calls[0]["kwargs"]["den"] == "512", calls
    assert calls[1]["savefig"].endswith(".png"), calls
    assert Path(calls[1]["savefig"]).name != "", calls



def test_cross_route_display_alignment_flips_negative_components_without_mutating_source() -> None:
    source = {
        "L": np.asarray(
            [
                [1.0, -2.0, 7.0],
                [2.0, -3.0, 8.0],
                [3.0, -4.0, 9.0],
                [4.0, -5.0, 10.0],
            ],
            dtype=np.float32,
        ),
    }
    reference = {
        "L": np.asarray(
            [
                [10.0, 2.0, 7.0],
                [20.0, 3.0, 8.0],
                [30.0, 4.0, 9.0],
                [40.0, 5.0, 10.0],
            ],
            dtype=np.float32,
        ),
    }
    original = source["L"].copy()

    display, meta = render_mod.align_display_gradients_to_reference(
        route="routea_ctxreg",
        smoothing="4mm",
        gradients_by_hemi=source,
        reference_by_hemi=reference,
        selected_components=[1, 2, 3],
        reference_route="routea",
    )

    assert np.allclose(source["L"], original), "display alignment must not mutate saved analysis arrays"
    assert np.allclose(display["L"][:, 0], original[:, 0])
    assert np.allclose(display["L"][:, 1], -original[:, 1])
    assert np.allclose(display["L"][:, 2], original[:, 2])
    g2 = next(item for item in meta["decisions"] if item["hemi"] == "L" and item["component"] == "G2")
    assert g2["display_sign"] == -1
    assert g2["flipped"] is True
    assert g2["correlation_to_reference"] < 0.0


def test_shared_display_color_ranges_use_component_bounds_across_routes() -> None:
    routea = {
        "L": np.asarray([[0.0, -1.0], [2.0, 3.0]], dtype=np.float32),
        "R": np.asarray([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32),
    }
    ctxreg = {
        "L": np.asarray([[-4.0, -5.0], [5.0, 7.0]], dtype=np.float32),
    }

    ranges = render_mod.compute_shared_display_color_ranges(
        [routea, ctxreg],
        selected_components=[1, 2],
        trim_fraction=0.0,
    )

    assert ranges == {1: (-4.0, 5.0), 2: (-5.0, 7.0)}
    assert render_mod.format_color_ranges_for_components(ranges, [1, 2]) == [[(-4.0, 5.0)], [(-5.0, 7.0)]]


def test_render_gradient_figure_passes_explicit_shared_color_ranges(tmp_path) -> None:
    calls: list[dict[str, object]] = []

    class FakeFigure:
        def savefig(self, path, **kwargs):
            calls.append({"savefig": Path(path).name, "kwargs": kwargs})
            Image.new("RGB", (24, 24), (255, 255, 255)).save(path)

    def fake_plotter(cdata, **kwargs):
        calls.append({"shape": tuple(cdata.shape), "kwargs": kwargs})
        return FakeFigure()

    out = tmp_path / "hippomaps_renderer_test_ranges.png"
    render_mod.render_gradient_figure(
        left=np.ones((4, 2), dtype=np.float32),
        right=np.zeros((4, 2), dtype=np.float32),
        output_png=out,
        den="512",
        plotter=fake_plotter,
        color_ranges=[[(-2.0, 3.0)], [(-4.0, 5.0)]],
        tighten_cwindow=False,
    )

    assert calls[0]["kwargs"]["color_range"] == [[(-2.0, 3.0)], [(-4.0, 5.0)]], calls
    assert calls[0]["kwargs"]["tighten_cwindow"] is False, calls

def test_render_gradient_figure_can_render_left_hemi_only(tmp_path) -> None:
    calls: list[dict[str, object]] = []

    class FakeFigure:
        def savefig(self, path, **kwargs):
            calls.append({"savefig": Path(path).name, "kwargs": kwargs})
            Image.new("RGB", (24, 24), (255, 255, 255)).save(path)

    def fake_plotter(cdata, **kwargs):
        calls.append({"shape": tuple(cdata.shape), "kwargs": kwargs})
        return FakeFigure()

    out = tmp_path / "hippomaps_renderer_test_left.png"
    render_mod.render_gradient_figure(
        left=np.ones((4, 3), dtype=np.float32),
        output_png=out,
        den="512",
        hemis=["L"],
        plotter=fake_plotter,
    )

    assert calls[0]["shape"] == (4, 1, 3), calls
    assert calls[0]["kwargs"]["hemis"] == ["L"], calls
    assert calls[1]["savefig"].endswith(".png"), calls
    assert Path(calls[1]["savefig"]).name != "", calls


def test_annotate_gradient_image_adds_left_margin_and_ev_labels() -> None:
    with TemporaryDirectory() as td:
        img_path = Path(td) / "base.png"
        Image.new("RGB", (100, 60), (255, 255, 255)).save(img_path)
        render_mod.annotate_gradient_image(
            img_path,
            selected_components=[1, 2, 3],
            ev_percentages=[41.2, 8.4, 4.0],
        )
        out = Image.open(img_path).convert("RGB")
        assert out.size == (330, 120), out.size
        # Check that the pasted base image is placed on the right (at margin_w=230)
        assert out.getpixel((240, 40)) == (255, 255, 255), out.getpixel((240, 40))
        # Search the left margin for any non-white text pixels
        found_non_white = False
        for x in range(230):
            for y_px in range(out.size[1]):
                if out.getpixel((x, y_px)) != (255, 255, 255):
                    found_non_white = True
                    break
            if found_non_white:
                break
        assert found_non_white, "expected text pixels in the left label margin"


def test_ensure_bilateral_gradient_annotations_repairs_late_raw_screenshot() -> None:
    with TemporaryDirectory() as td:
        root = Path(td)
        left = root / "left.png"
        right = root / "right.png"
        final = root / "bilateral.png"
        Image.new("RGB", (510, 90), (255, 255, 255)).save(left)
        Image.new("RGB", (510, 90), (255, 255, 255)).save(right)
        Image.new("RGB", (1020, 90), (255, 255, 255)).save(final)
        render_mod.ensure_bilateral_gradient_annotations(
            side_paths={"L": left, "R": right},
            output_png=final,
            selected_components=[1, 2, 3],
            ev_by_hemi={"L": [10.0, 5.0, 2.5], "R": [9.0, 4.0, 2.0]},
            settle_seconds=0.0,
        )
        assert Image.open(left).size == (740, 150)
        assert Image.open(right).size == (740, 150)
        assert Image.open(final).size == (1480, 150)


def test_route_tag_auto_detection_and_validation() -> None:
    # Test 05_02 infer_route_tag function
    assert atlas_mod.infer_route_tag(Path("/path/to/pfm"), "auto") == "pfm"
    assert atlas_mod.infer_route_tag(Path("/path/to/pfm_ctxreg"), "auto") == "pfm_ctxreg"
    assert atlas_mod.infer_route_tag(Path("/path/to/fixed"), "auto") == "fixed"
    assert atlas_mod.infer_route_tag(Path("/path/to/any"), "pfm") == "pfm"

    # Test that mismatched out-root under group_atlas raises ValueError
    try:
        route_tags.validate_group_atlas_root_for_route(
            Path("outputs_hcp/work/group_atlas/fixed"),
            "pfm",
        )
    except ValueError as exc:
        assert "Provenance discrepancy" in str(exc)
    else:
        raise AssertionError("pfm must not validate against fixed group atlas root")

    # Verify main arg parser in 06_03 defaults to auto for route-tag
    with patch("sys.argv", ["06_03_build_cluster_cortex_panels.py"]):
        args_06_03 = cortex_mod.parse_args()
        assert args_06_03.route_tag == "auto"


def test_route_tag_exact_group_root_validation_rejects_prefix_collisions() -> None:
    route_tags.validate_group_atlas_root_for_route(
        Path("outputs_hcp/work/group_atlas/pfm"),
        "pfm",
    )
    route_tags.validate_group_atlas_root_for_route(
        Path("outputs_hcp/work/group_atlas/pfm_17net"),
        "pfm_17net",
    )

    try:
        route_tags.validate_group_atlas_root_for_route(
            Path("outputs_hcp/work/group_atlas/pfm_17net"),
            "pfm",
        )
    except ValueError as exc:
        assert "Provenance discrepancy" in str(exc)
    else:
        raise AssertionError("pfm must not validate against pfm_17net group atlas root")


def test_family17_auto_route_requires_explicit_routea_17net() -> None:
    try:
        route_tags.resolve_route_tag_for_group_atlas_run(
            shared_root=Path("outputs_hcp/inputs/kong2022/shared_inputs/pfm"),
            explicit_route="auto",
            network_taxonomy="family17",
        )
    except ValueError as exc:
        assert "pfm_17net" in str(exc)
    else:
        raise AssertionError("family17 auto route must refuse to silently use pfm")

    assert (
        route_tags.resolve_route_tag_for_group_atlas_run(
            shared_root=Path("outputs_hcp/inputs/kong2022/shared_inputs/pfm"),
            explicit_route="pfm_17net",
            network_taxonomy="family17",
        )
        == "pfm_17net"
    )


def test_route_auto_helpers_cover_renderers_and_confidence_choices() -> None:
    assert (
        route_tags.infer_route_tag_from_group_root(Path("outputs_hcp/work/group_atlas/pfm_ctxdrop"))
        == "pfm_ctxdrop"
    )
    assert (
        route_tags.infer_route_tag_from_group_root(Path("outputs_hcp/work/group_atlas/pfm_17net"))
        == "pfm_17net"
    )

    confidence_args = confidence_mod.build_arg_parser().parse_args([])
    assert confidence_args.route_tag == "auto"
    assert "pfm_17net" in confidence_mod.build_arg_parser()._option_string_actions["--route-tag"].choices


def main() -> int:
    test_group_mean_fc_applies_fisher_z_before_averaging()
    test_gradientmaps_called_with_official_pearson_kernel_and_full_lambda_ev()
    test_gradient_fit_has_no_seed_argument()
    test_group_atlas_gradient_spectral_uses_hippomaps_adapter_and_records_provenance()
    test_gradient_split_diagnostic_failures_are_counted_when_allowed()
    test_evaluate_k_candidate_records_no_valid_gradient_diagnostic_status()
    test_run_aware_ari_pairs_only_successful_runs_after_skipped_failures()
    test_group_atlas_public_methods_drop_partial_corr_branches()
    test_group_atlas_parser_defaults_to_fixed_k3()
    test_group_atlas_gradient_spectral_loads_hippomaps_gradient_artifacts()
    test_group_atlas_gradient_loader_prefers_local_artifact_after_relocation()
    test_canonical_public_surfaces_drop_partial_corr_branches()
    test_canonical_present_defaults_use_outputs_hcp_layout()
    test_run_layout_keeps_smoke_outputs_in_dedicated_run_tree()
    test_smoke_command_plan_uses_isolated_fixedk_full_layout()
    test_compute_defaults_to_five_components_like_hippomaps_paper()
    test_full_surface_reconstruction_keeps_nan_outside_support()
    test_gradient_alignment_reorders_components_and_flips_sign_to_anchor()
    test_gradient_alignment_anchor_combo_is_identity_without_raw_outputs()
    test_render_gradient_figure_calls_official_hippomaps_plotter_options()
    test_cross_route_display_alignment_flips_negative_components_without_mutating_source()
    test_shared_display_color_ranges_use_component_bounds_across_routes()
    test_render_gradient_figure_passes_explicit_shared_color_ranges()
    test_render_gradient_figure_can_render_left_hemi_only()
    test_annotate_gradient_image_adds_left_margin_and_ev_labels()
    test_ensure_bilateral_gradient_annotations_repairs_late_raw_screenshot()
    test_route_tag_auto_detection_and_validation()
    test_route_tag_exact_group_root_validation_rejects_prefix_collisions()
    test_family17_auto_route_requires_explicit_routea_17net()
    test_route_auto_helpers_cover_renderers_and_confidence_choices()
    print("ok: hippomaps gradient route checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

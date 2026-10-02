#!/usr/bin/env python3
from __future__ import annotations

import sys
import importlib.util
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def _load_numbered_module(script_name: str, alias: str):
    module_path = REPO_ROOT / "scripts" / "core_workflow" / script_name
    spec = importlib.util.spec_from_file_location(alias, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    core_dir = (REPO_ROOT / "scripts" / "core_workflow").resolve()
    if str(core_dir) not in sys.path:
        sys.path.insert(0, str(core_dir))
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


atlas_mod = _load_numbered_module("05_02_run_group_atlas.py", "core_05_02_run_group_atlas")
build_group_gradient_features = atlas_mod.build_group_gradient_features
brainspace_diffusion_gradients = atlas_mod.brainspace_diffusion_gradients
group_feature_matrix = atlas_mod.group_feature_matrix
method_uses_network_fc_input = atlas_mod.method_uses_network_fc_input
method_uses_subject_fc_preprocessing = atlas_mod.method_uses_subject_fc_preprocessing
parse_gradient_components = atlas_mod.parse_gradient_components
resolve_subject_fc_mode = atlas_mod.resolve_subject_fc_mode
preprocess_intrinsic_fc_map = atlas_mod.preprocess_intrinsic_fc_map
preprocess_network_fc_map = atlas_mod.preprocess_network_fc_map
preprocess_subject_fc_matrix = atlas_mod.preprocess_subject_fc_matrix
roi_similarity_from_network_fc = atlas_mod.roi_similarity_from_network_fc
runwise_feature_bank = atlas_mod.runwise_feature_bank
zscore_cols = atlas_mod.zscore_cols


def test_network_top_fraction_binarize_preserves_nan() -> None:
    x = np.array(
        [
            [0.1, 0.9, -0.2, np.nan, 0.8],
            [0.3, 0.4, 0.5, 0.6, -0.1],
        ],
        dtype=np.float32,
    )
    out = preprocess_network_fc_map(x, top_fraction=0.10, binarize=True)
    expected = np.array(
        [
            [0.0, 1.0, 0.0, np.nan, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    assert np.allclose(np.nan_to_num(out), np.nan_to_num(expected), atol=1e-6), out
    assert bool(np.isnan(out[0, 3])), out


def test_network_full_mode_is_passthrough() -> None:
    x = np.array(
        [
            [0.1, 0.9, -0.2, np.nan, 0.8],
            [0.3, 0.4, 0.5, 0.6, -0.1],
        ],
        dtype=np.float32,
    )
    out = preprocess_subject_fc_matrix(
        "network-spectral",
        x,
        subject_fc_mode=resolve_subject_fc_mode("full", None),
    )
    assert np.allclose(np.nan_to_num(out), np.nan_to_num(x), atol=1e-6), out
    assert bool(np.isnan(out[0, 3])), out


def test_intrinsic_top_fraction_binarize_keeps_symmetry_and_diag0() -> None:
    x = np.array(
        [
            [0.0, 0.1, 0.4, np.nan],
            [0.1, 0.0, 0.9, np.nan],
            [0.4, 0.9, 0.0, np.nan],
            [np.nan, np.nan, np.nan, 0.0],
        ],
        dtype=np.float32,
    )
    out = preprocess_intrinsic_fc_map(x, top_fraction=0.34, binarize=True)
    expected = np.array(
        [
            [0.0, 0.0, 1.0, np.nan],
            [0.0, 0.0, 1.0, np.nan],
            [1.0, 1.0, 0.0, np.nan],
            [np.nan, np.nan, np.nan, 0.0],
        ],
        dtype=np.float32,
    )
    assert np.allclose(np.nan_to_num(out), np.nan_to_num(expected), atol=1e-6), out
    assert np.allclose(out, out.T, equal_nan=True), out
    assert np.allclose(np.diag(out), np.zeros((4,), dtype=np.float32)), out


def test_intrinsic_full_mode_is_passthrough() -> None:
    x = np.array(
        [
            [0.0, 0.1, 0.4, np.nan],
            [0.1, 0.0, 0.9, np.nan],
            [0.4, 0.9, 0.0, np.nan],
            [np.nan, np.nan, np.nan, 0.0],
        ],
        dtype=np.float32,
    )
    out = preprocess_subject_fc_matrix(
        "intrinsic-spectral",
        x,
        subject_fc_mode=resolve_subject_fc_mode("full", None),
    )
    assert np.allclose(np.nan_to_num(out), np.nan_to_num(x), atol=1e-6), out
    assert np.allclose(out, out.T, equal_nan=True), out


def test_method_gate_covers_all_supported_spectral_preprocessing_methods() -> None:
    assert method_uses_subject_fc_preprocessing("group-roi-sim-spectral")
    assert method_uses_subject_fc_preprocessing("network-spectral")
    assert method_uses_subject_fc_preprocessing("intrinsic-spectral")
    assert not method_uses_subject_fc_preprocessing("wta-top")
    assert not method_uses_subject_fc_preprocessing("group-network-partial-spectral")
    assert not method_uses_subject_fc_preprocessing("group-intrinsic-partial-spectral")

    x = np.array([[0.2, 0.1], [0.4, 0.3]], dtype=np.float32)
    out = preprocess_subject_fc_matrix(
        "wta-top",
        x,
        subject_fc_mode=resolve_subject_fc_mode("top-percent", 10),
    )
    assert np.array_equal(out, x), out


def test_subject_fc_mode_validation_and_tags() -> None:
    full = resolve_subject_fc_mode("full", None)
    assert full.mode == "full"
    assert full.mode_tag == "full"
    assert full.top_percent is None

    sparse = resolve_subject_fc_mode("top-percent", 10)
    assert sparse.mode == "top-percent"
    assert sparse.mode_tag == "10perc"
    assert sparse.top_percent == 10

    try:
        resolve_subject_fc_mode("full", 10)
        raise AssertionError("expected full mode + top-percent to fail")
    except ValueError:
        pass
    try:
        resolve_subject_fc_mode("top-percent", None)
        raise AssertionError("expected top-percent mode without percent to fail")
    except ValueError:
        pass
    for bad in [0, -1, 101]:
        try:
            resolve_subject_fc_mode("top-percent", bad)
            raise AssertionError(f"expected bad percent to fail: {bad}")
        except ValueError:
            pass


def test_roi_similarity_from_network_fc_is_symmetric_and_finite_with_diag0() -> None:
    x = np.array(
        [
            [1.0, 2.0, np.nan, 4.0],
            [2.0, 1.0, 0.0, 3.0],
            [4.0, 0.0, 1.0, np.nan],
        ],
        dtype=np.float32,
    )
    out = roi_similarity_from_network_fc(x)
    assert out.shape == (3, 3), out.shape
    assert np.all(np.isfinite(out)), out
    assert np.allclose(out, out.T, atol=1e-6), out
    assert np.allclose(np.diag(out), np.zeros((3,), dtype=np.float32), atol=1e-6), out


def test_group_feature_matrix_matches_preprocessed_subject_average() -> None:
    raw_a = np.array(
        [
            [0.1, 0.9, 0.2],
            [0.3, -0.4, 0.8],
        ],
        dtype=np.float32,
    )
    raw_b = np.array(
        [
            [0.7, 0.6, -0.2],
            [0.5, 0.4, 0.1],
        ],
        dtype=np.float32,
    )
    mode = resolve_subject_fc_mode("top-percent", 50)
    pre_a = preprocess_subject_fc_matrix("network-spectral", raw_a, subject_fc_mode=mode)
    pre_b = preprocess_subject_fc_matrix("network-spectral", raw_b, subject_fc_mode=mode)
    got = group_feature_matrix(
        method="network-spectral",
        feature_bank={"a": pre_a, "b": pre_b},
        subjects_subset=["a", "b"],
        support_mask=np.array([True, True], dtype=bool),
    )
    expected = zscore_cols(np.nanmean(np.stack([pre_a, pre_b], axis=0), axis=0))
    assert np.allclose(got, expected, atol=1e-6), (got, expected)


def test_group_gradient_features_average_subject_roi_fc_then_select_components() -> None:
    rng = np.random.RandomState(2026)
    subj_a = rng.randn(8, 12).astype(np.float32)
    subj_b = subj_a + np.float32(0.05)
    support = np.array([True, True, True, True, True, True, True, True], dtype=bool)

    def fake_brainspace(group_fc: np.ndarray, **kwargs):
        n = int(group_fc.shape[0])
        grads = np.tile(np.asarray([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32), (n, 1))
        lambdas = np.asarray([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float64)
        ev = lambdas / np.sum(lambdas)
        return grads, lambdas, ev

    old = atlas_mod.brainspace_diffusion_gradients
    atlas_mod.brainspace_diffusion_gradients = fake_brainspace
    try:
        features, meta = build_group_gradient_features(
            feature_bank={"a": subj_a, "b": subj_b},
            subjects_subset=["a", "b"],
            support_mask=support,
            gradient_components=parse_gradient_components("1,2"),
        )
    finally:
        atlas_mod.brainspace_diffusion_gradients = old

    assert features.shape == (8, 2), features.shape
    assert np.all(np.isfinite(features)), features
    assert meta["gradient_components"] == [1, 2], meta
    assert meta["group_vertex_to_roi_fc_shape"] == [8, 12], meta
    assert meta["gradient_embedding_engine"] == "brainspace_gradientmaps_via_hippomaps_adapter", meta
    assert meta["kernel"] == "pearson", meta
    assert np.isclose(meta["brainspace_sparsity"], 0.0), meta
    assert int(meta["n_components_computed"]) == 5, meta
    assert meta["components_displayed"] == [1, 2, 3], meta
    assert len(meta["gradient_eigenvalues"]) == 5, meta
    assert len(meta["ev_fractions"]) == 5, meta


def test_brainspace_adapter_uses_expected_parameters() -> None:
    calls: dict[str, object] = {}

    def fake_fit(group_fc: np.ndarray, *, n_components: int):
        calls["fit"] = {
            "shape": tuple(np.asarray(group_fc).shape),
            "n_components": int(n_components),
        }
        grads = np.arange(30, dtype=np.float32).reshape(6, 5)
        lambdas = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float64)
        ev = lambdas / np.sum(lambdas)
        return grads, lambdas, ev

    old_ensure = atlas_mod.ensure_hippomaps_on_path
    atlas_mod.ensure_hippomaps_on_path = lambda root: Path(root)
    try:
        x = np.arange(24, dtype=np.float32).reshape(6, 4)
        grads, lambdas, ev = brainspace_diffusion_gradients(x, gradientmaps_fit_fn=fake_fit)
    finally:
        atlas_mod.ensure_hippomaps_on_path = old_ensure

    assert calls["fit"] == {"shape": (6, 4), "n_components": 5}, calls
    assert grads.shape == (6, 5), grads.shape
    assert lambdas.shape == (5,), lambdas.shape
    assert ev.shape == (5,), ev.shape


def test_group_feature_matrix_gradient_spectral_uses_selected_gradient_columns() -> None:
    rng = np.random.RandomState(2027)
    subj_a = rng.randn(8, 12).astype(np.float32)
    subj_b = subj_a + np.float32(0.05)
    support = np.array([True, True, True, True, True, True, True, True], dtype=bool)

    def fake_brainspace(group_fc: np.ndarray, **kwargs):
        n = int(group_fc.shape[0])
        grads = np.tile(np.asarray([[2.0, 1.0, -1.0, 0.5, 0.25]], dtype=np.float32), (n, 1))
        lambdas = np.asarray([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float64)
        ev = lambdas / np.sum(lambdas)
        return grads, lambdas, ev

    old = atlas_mod.brainspace_diffusion_gradients
    atlas_mod.brainspace_diffusion_gradients = fake_brainspace
    try:
        got = group_feature_matrix(
            method="gradient-spectral",
            feature_bank={"a": subj_a, "b": subj_b},
            subjects_subset=["a", "b"],
            support_mask=support,
            gradient_components=(1, 2, 3),
        )
    finally:
        atlas_mod.brainspace_diffusion_gradients = old

    assert got.shape == (8, 3), got.shape
    assert np.all(np.isfinite(got)), got


def test_group_feature_matrix_roi_similarity_method_uses_square_branch() -> None:
    a = np.array(
        [
            [0.0, 0.4, 0.1],
            [0.4, 0.0, 0.7],
            [0.1, 0.7, 0.0],
        ],
        dtype=np.float32,
    )
    b = np.array(
        [
            [0.0, 0.2, 0.5],
            [0.2, 0.0, 0.3],
            [0.5, 0.3, 0.0],
        ],
        dtype=np.float32,
    )
    support = np.array([True, True, False], dtype=bool)
    got = group_feature_matrix(
        method="group-roi-sim-spectral",
        feature_bank={"a": a, "b": b},
        subjects_subset=["a", "b"],
        support_mask=support,
    )
    expected = np.nanmean(np.stack([a[np.ix_(support, support)], b[np.ix_(support, support)]], axis=0), axis=0)
    np.fill_diagonal(expected, 0.0)
    assert np.allclose(got, expected, atol=1e-6), (got, expected)


def test_network_input_gate_includes_roi_similarity_method() -> None:
    assert method_uses_network_fc_input("group-roi-sim-spectral") is True
    assert method_uses_network_fc_input("network-spectral") is True
    assert method_uses_network_fc_input("gradient-spectral") is True
    assert method_uses_network_fc_input("wta-top") is True
    assert method_uses_network_fc_input("intrinsic-spectral") is False


def test_runwise_feature_bank_roi_similarity_uses_network_inputs() -> None:
    from lib.kong2022_label_utils import load_kong2022_bundle
    with TemporaryDirectory() as td:
        root = Path(td)
        subject = "100206"
        atlas = "kong2022"
        smoothing = "2mm"
        hemi = "L"
        n_vertices = 3
        n_roi = 400
        reference = root / f"sub-{subject}" / "reference" / atlas
        reference.mkdir(parents=True)
        rows = load_kong2022_bundle()["roi_registry"]
        (reference / "cortex_roi_parcels.tsv").write_text(
            "parcel_id\tparcel_name\n" + "".join(f"{r['roi_id']}\t{r['roi_name']}\n" for r in rows))
        np.save(reference / "cortex_roi_parcel_timeseries.npy", np.ones((400, 8), dtype=np.float32))
        valid_masks = {subject: np.array([True, True, True], dtype=bool)}
        for rid in [1, 2, 3, 4]:
            run_dir = (
                root
                / f"sub-{subject}"
                / "fc"
                / atlas
                / smoothing
                / f"hemi_{hemi}"
                / "runs"
                / f"run-{rid}"
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            arr = np.random.default_rng(rid).uniform(-0.5, 0.5, (n_vertices, n_roi)).astype(np.float32)
            np.save(run_dir / "hipp_vertex_to_roi_fc_active.npy", arr)

        out = runwise_feature_bank(
            method="group-roi-sim-spectral",
            shared_root=root,
            subjects=[subject],
            atlas=atlas,
            smoothing=smoothing,
            hemi=hemi,
            valid_masks=valid_masks,
            subject_fc_mode=resolve_subject_fc_mode("full", None),
            reference_root=root,
        )
        for rid in [1, 2, 3, 4]:
            got = out[rid][subject]
            assert got.shape == (n_vertices, n_vertices), got.shape
            assert np.all(np.isfinite(got)), got
            assert np.allclose(got, got.T, atol=1e-6), got
            assert np.allclose(np.diag(got), np.zeros((n_vertices,), dtype=np.float32), atol=1e-6), got


def main() -> int:
    test_network_top_fraction_binarize_preserves_nan()
    test_network_full_mode_is_passthrough()
    test_intrinsic_top_fraction_binarize_keeps_symmetry_and_diag0()
    test_intrinsic_full_mode_is_passthrough()
    test_method_gate_covers_all_supported_spectral_preprocessing_methods()
    test_subject_fc_mode_validation_and_tags()
    test_roi_similarity_from_network_fc_is_symmetric_and_finite_with_diag0()
    test_group_feature_matrix_matches_preprocessed_subject_average()
    test_group_gradient_features_average_subject_roi_fc_then_select_components()
    test_brainspace_adapter_uses_expected_parameters()
    test_group_feature_matrix_gradient_spectral_uses_selected_gradient_columns()
    test_group_feature_matrix_roi_similarity_method_uses_square_branch()
    test_network_input_gate_includes_roi_similarity_method()
    test_runwise_feature_bank_roi_similarity_uses_network_inputs()
    print("ok: subject FC preprocessing checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

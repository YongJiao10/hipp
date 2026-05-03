#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory

from run_merged_canonical_group_atlas import (
    group_feature_matrix,
    method_uses_network_fc_input,
    method_uses_subject_fc_preprocessing,
    preprocess_intrinsic_fc_map,
    preprocess_network_fc_map,
    preprocess_subject_fc_matrix,
    roi_similarity_from_network_fc,
    runwise_feature_bank,
    zscore_cols,
)


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


def test_network_top_fraction_without_binarize() -> None:
    x = np.array(
        [
            [0.1, 0.9, -0.2, np.nan, 0.8],
            [0.3, 0.4, 0.5, 0.6, -0.1],
        ],
        dtype=np.float32,
    )
    out = preprocess_network_fc_map(x, top_fraction=0.34, binarize=False)
    expected = np.array(
        [
            [0.0, 0.9, 0.0, np.nan, 0.8],
            [0.0, 0.0, 0.5, 0.6, 0.0],
        ],
        dtype=np.float32,
    )
    assert np.allclose(np.nan_to_num(out), np.nan_to_num(expected), atol=1e-6), out
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


def test_intrinsic_top_fraction_without_binarize() -> None:
    x = np.array(
        [
            [0.0, 0.1, 0.4, np.nan],
            [0.1, 0.0, 0.9, np.nan],
            [0.4, 0.9, 0.0, np.nan],
            [np.nan, np.nan, np.nan, 0.0],
        ],
        dtype=np.float32,
    )
    out = preprocess_intrinsic_fc_map(x, top_fraction=0.34, binarize=False)
    expected = np.array(
        [
            [0.0, 0.0, 0.4, np.nan],
            [0.0, 0.0, 0.9, np.nan],
            [0.4, 0.9, 0.0, np.nan],
            [np.nan, np.nan, np.nan, 0.0],
        ],
        dtype=np.float32,
    )
    assert np.allclose(np.nan_to_num(out), np.nan_to_num(expected), atol=1e-6), out
    assert np.allclose(out, out.T, equal_nan=True), out


def test_method_gate_covers_all_supported_spectral_preprocessing_methods() -> None:
    assert method_uses_subject_fc_preprocessing("group-roi-sim-spectral")
    assert method_uses_subject_fc_preprocessing("group-network-spectral")
    assert method_uses_subject_fc_preprocessing("group-intrinsic-spectral")
    assert not method_uses_subject_fc_preprocessing("group-network-top")
    assert not method_uses_subject_fc_preprocessing("group-network-all")
    assert not method_uses_subject_fc_preprocessing("group-network-partial-spectral")
    assert not method_uses_subject_fc_preprocessing("group-intrinsic-partial-spectral")

    x = np.array([[0.2, 0.1], [0.4, 0.3]], dtype=np.float32)
    out = preprocess_subject_fc_matrix("group-network-top", x, top_fraction=0.10, binarize=True)
    assert np.array_equal(out, x), out


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
    pre_a = preprocess_subject_fc_matrix("group-network-spectral", raw_a, top_fraction=0.50, binarize=True)
    pre_b = preprocess_subject_fc_matrix("group-network-spectral", raw_b, top_fraction=0.50, binarize=True)
    got = group_feature_matrix(
        method="group-network-spectral",
        feature_bank={"a": pre_a, "b": pre_b},
        subjects_subset=["a", "b"],
        support_mask=np.array([True, True], dtype=bool),
    )
    expected = zscore_cols(np.nanmean(np.stack([pre_a, pre_b], axis=0), axis=0))
    assert np.allclose(got, expected, atol=1e-6), (got, expected)


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
    assert method_uses_network_fc_input("group-network-spectral") is True
    assert method_uses_network_fc_input("group-network-top") is True
    assert method_uses_network_fc_input("group-intrinsic-spectral") is False


def test_runwise_feature_bank_roi_similarity_uses_network_inputs() -> None:
    with TemporaryDirectory() as td:
        root = Path(td)
        subject = "100206"
        atlas = "kong2022"
        smoothing = "2mm"
        hemi = "L"
        n_vertices = 3
        n_roi = 4
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
            arr = np.arange(1, n_vertices * n_roi + 1, dtype=np.float32).reshape(n_vertices, n_roi) + np.float32(rid)
            np.save(run_dir / "hipp_vertex_to_network_fc_active.npy", arr)

        out = runwise_feature_bank(
            method="group-roi-sim-spectral",
            shared_root=root,
            subjects=[subject],
            atlas=atlas,
            smoothing=smoothing,
            hemi=hemi,
            valid_masks=valid_masks,
            subject_fc_top_fraction=1.0,
            subject_fc_binarize=False,
        )
        for rid in [1, 2, 3, 4]:
            got = out[rid][subject]
            assert got.shape == (n_vertices, n_vertices), got.shape
            assert np.all(np.isfinite(got)), got
            assert np.allclose(got, got.T, atol=1e-6), got
            assert np.allclose(np.diag(got), np.zeros((n_vertices,), dtype=np.float32), atol=1e-6), got


def main() -> int:
    test_network_top_fraction_binarize_preserves_nan()
    test_network_top_fraction_without_binarize()
    test_intrinsic_top_fraction_binarize_keeps_symmetry_and_diag0()
    test_intrinsic_top_fraction_without_binarize()
    test_method_gate_covers_all_supported_spectral_preprocessing_methods()
    test_roi_similarity_from_network_fc_is_symmetric_and_finite_with_diag0()
    test_group_feature_matrix_matches_preprocessed_subject_average()
    test_group_feature_matrix_roi_similarity_method_uses_square_branch()
    test_network_input_gate_includes_roi_similarity_method()
    test_runwise_feature_bank_roi_similarity_uses_network_inputs()
    print("ok: subject FC preprocessing checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

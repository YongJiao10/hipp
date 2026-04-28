#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from run_merged_canonical_group_atlas import (
    group_feature_matrix,
    method_uses_subject_fc_preprocessing,
    preprocess_intrinsic_fc_map,
    preprocess_network_fc_map,
    preprocess_subject_fc_matrix,
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


def test_method_gate_only_targets_two_spectral_methods() -> None:
    assert method_uses_subject_fc_preprocessing("group-network-spectral")
    assert method_uses_subject_fc_preprocessing("group-intrinsic-spectral")
    assert not method_uses_subject_fc_preprocessing("group-network-top")
    assert not method_uses_subject_fc_preprocessing("group-network-all")
    assert not method_uses_subject_fc_preprocessing("group-network-partial-spectral")
    assert not method_uses_subject_fc_preprocessing("group-intrinsic-partial-spectral")

    x = np.array([[0.2, 0.1], [0.4, 0.3]], dtype=np.float32)
    out = preprocess_subject_fc_matrix("group-network-top", x, top_fraction=0.10, binarize=True)
    assert np.array_equal(out, x), out


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


def main() -> int:
    test_network_top_fraction_binarize_preserves_nan()
    test_network_top_fraction_without_binarize()
    test_intrinsic_top_fraction_binarize_keeps_symmetry_and_diag0()
    test_intrinsic_top_fraction_without_binarize()
    test_method_gate_only_targets_two_spectral_methods()
    test_group_feature_matrix_matches_preprocessed_subject_average()
    print("ok: subject FC preprocessing checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

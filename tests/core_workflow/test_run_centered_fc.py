from __future__ import annotations


import numpy as np
import pytest


from lib import fc_math


def test_run_offsets_cannot_reverse_within_run_correlation() -> None:
    x = np.array([[-1, 1, -1, 1, 99, 101, 99, 101]], dtype=np.float32)
    y = np.array([[1, -1, 1, -1, 101, 99, 101, 99]], dtype=np.float32)
    original_x, original_y = x.copy(), y.copy()
    assert fc_math.corrcoef_rows(x, y).item() > 0.99
    actual = fc_math.corrcoef_rows(x, y, run_lengths=[4, 4])
    assert actual.item() < -0.99
    np.testing.assert_array_equal(x, original_x)
    np.testing.assert_array_equal(y, original_y)


def test_run_centering_preserves_within_run_variance_weighting() -> None:
    x = np.array([[-2, -1, 1, 2, -20, -10, 10, 20]], dtype=np.float32)
    y = np.array([[-2, -1, 1, 2, 20, 10, -10, -20]], dtype=np.float32)
    actual = fc_math.corrcoef_rows(x, y, run_lengths=[4, 4])
    np.testing.assert_allclose(actual, [[-99 / 101]], atol=2e-6)


def test_run_step_only_and_incomplete_rows_remain_undefined() -> None:
    rows = np.array([
        [100, 100, 100, 100, 200, 200, 200, 200],
        [1, 2, np.nan, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 10, 11, 12, 13],
    ], dtype=np.float32)
    centered = fc_math.center_run_timeseries(rows, [4, 4])
    np.testing.assert_array_equal(centered[0], 0)
    assert np.isnan(centered[1]).all()
    np.testing.assert_allclose(centered[2, :4].mean(), 0, atol=1e-7)
    np.testing.assert_allclose(centered[2, 4:].mean(), 0, atol=1e-7)
    actual = fc_math.corrcoef_rows(rows, rows, run_lengths=[4, 4])
    assert np.isnan(actual[:2]).all()
    assert actual[2, 2] > 0.99


@pytest.mark.parametrize("lengths", [[3, 3], [1, 7], []])
def test_fc_requires_an_explicit_frame_matched_run_partition(lengths: list[int]) -> None:
    rows = np.arange(16, dtype=np.float32).reshape(2, 8)
    with pytest.raises(ValueError, match="run|Run"):
        fc_math.corrcoef_rows(rows, rows, run_lengths=lengths)


def test_single_run_centering_matches_standard_pearson() -> None:
    rng = np.random.default_rng(403)
    x, y = rng.normal(size=(3, 19)), rng.normal(size=(4, 19))
    expected = fc_math.corrcoef_rows(x, y)
    actual = fc_math.corrcoef_rows(x, y, run_lengths=[19])
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)

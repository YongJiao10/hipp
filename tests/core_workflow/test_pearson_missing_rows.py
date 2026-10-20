"""Undefined Pearson pairs stay missing; finite pairs match an independent oracle."""

import numpy as np

from lib.fc_math import corrcoef_rows


def test_undefined_correlations_are_nan_including_partial_nonfinite_rows():
    good = np.array([-1, 1, -1, 1], dtype=np.float32)
    rows = np.array([good, -good, np.zeros(4), np.full(4, np.nan),
                     [np.nan, 1, -1, 1], [np.inf, 1, -1, 1]], dtype=np.float32)
    actual = corrcoef_rows(rows, rows)
    np.testing.assert_allclose(actual[:2, :2], [[1, -1], [-1, 1]], atol=2e-6)
    assert np.isnan(actual[2:]).all()
    assert np.isnan(actual[:, 2:]).all()


def test_finite_correlations_match_numpy_without_a_new_small_sd_cutoff():
    rng = np.random.default_rng(1701)
    x = rng.normal(size=(4, 19)).astype(np.float32)
    y = rng.normal(size=(3, 19)).astype(np.float32)
    expected = np.corrcoef(x, y)[:4, 4:]
    np.testing.assert_allclose(corrcoef_rows(x, y), expected, rtol=2e-6, atol=2e-7)
    tiny = np.array([[-1, 1, -1, 1]], dtype=np.float32)*1e-13
    # Preserve the inherited normalization floor without treating it as a gate.
    assert np.isfinite(corrcoef_rows(tiny, tiny)).all()

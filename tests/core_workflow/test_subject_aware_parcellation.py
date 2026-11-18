"""Numerical regressions for the individual-data HCP comparison."""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/hcp_ctxreg_affinity_comparison"))

from subject_models import (  # noqa: E402
    FitError, fit_directional, hippocampal_affinity, shen_discretize, spectral_coordinates, unit_directions,
)


def test_joint_discretization_recovers_rotated_parcels_with_missing_vertices():
    rng = np.random.default_rng(7)
    expected = np.repeat(np.arange(3), 12)
    embedding = np.eye(3)[expected]
    support = np.ones(expected.size, dtype=bool)
    bank = []
    for _ in range(7):
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        observed = np.flatnonzero(rng.random(expected.size) > 0.14)
        bank.append((observed, embedding[observed] @ rotation))
    fit = shen_discretize(bank, support, k=3)
    assert adjusted_rand_score(expected, fit.labels) == 1
    assert fit.diagnostics["loss"] < 1e-20
    for (observed, _), labels in zip(bank, fit.individual, strict=True):
        np.testing.assert_array_equal(labels[observed], fit.labels[observed])
        assert np.count_nonzero(labels) == len(observed)

    single = shen_discretize([(np.flatnonzero(support), embedding)], support, k=3)
    assert adjusted_rand_score(expected, single.labels) == 1


def test_disconnected_affinity_stops_at_the_observed_vertex_ids():
    sim = np.array([[1, .8, -.1], [.8, 1, -.4], [-.1, -.4, 1]])
    with pytest.raises(FitError, match=r"component sizes=\[2, 1\], zero-degree vertices=\[77\]"):
        spectral_coordinates(hippocampal_affinity(sim), np.array([2, 3, 77]), k=3)


def test_centered_direction_dot_product_is_pearson():
    fc = np.random.default_rng(11).normal(size=(5, 20, 400))
    observed = np.ones(fc.shape[:2], dtype=bool)
    observed[0, 0] = False
    out = unit_directions(fc, observed, centered=True)
    assert out.shape == (5, 20, 399)
    assert np.isnan(out[0, 0]).all()
    x = out[observed]
    np.testing.assert_allclose(x @ x.T, np.corrcoef(fc[observed]), atol=2e-6)
    fc[0, 1] = 0.25
    with pytest.raises(FitError, match="Constant Fisher-z fingerprint"):
        unit_directions(fc, observed, centered=True)


def test_upstream_directional_fits_preserve_subject_and_parcel_axes():
    pytest.importorskip("HierarchBayesParcel")
    import torch

    torch.set_num_threads(1)
    rng = np.random.default_rng(41)
    expected = np.repeat(np.arange(3), 8)
    parcel_signals = rng.normal(size=(3, 15))
    fc = np.tile(parcel_signals[expected], (9, 1, 1)) + .1 * rng.normal(size=(9, 24, 15))
    observed = np.ones(fc.shape[:2], dtype=bool)
    observed[0, 0] = False
    x = unit_directions(fc, observed, centered=True)
    for spatial, average in ((False, True), (False, False), (True, False)):
        fit = fit_directional(x, observed, k=3, spatial=spatial, mean_only=average, save_individual=spatial)
        assert adjusted_rand_score(expected, fit.labels) == 1
        assert fit.probability.shape == (24, 3)
        np.testing.assert_allclose(fit.probability.sum(axis=1), 1)
        if spatial:
            assert fit.posterior.shape == (9, 24, 3)
            assert np.isnan(fit.posterior[0, 0]).all()
            assert fit.individual[0, 0] == 0

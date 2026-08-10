from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_WORKFLOW_DIR = REPO_ROOT / "scripts" / "core_workflow"
if str(CORE_WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_WORKFLOW_DIR))

from lib.template_matching import (  # noqa: E402
    build_cluster_prototypes,
    match_vertices_to_prototypes,
)


def test_prototypes_use_only_supported_cluster_vertices_and_ignore_nonfinite_values():
    group_fc = np.asarray(
        [
            [1.0, np.inf, 5.0],
            [3.0, 4.0, np.nan],
            [99.0, 99.0, 99.0],
            [-1.0, 2.0, 6.0],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([1, 1, 1, 2], dtype=np.int64)
    support = np.asarray([True, True, False, True])

    got = build_cluster_prototypes(group_fc, labels, support, k=2)

    assert got.dtype == np.float32
    assert got.shape == (2, 3)
    assert np.allclose(got, [[2.0, 4.0, 5.0], [-1.0, 2.0, 6.0]])


def test_prototypes_reject_a_cluster_without_supported_vertices():
    with pytest.raises(ValueError, match="cluster 2"):
        build_cluster_prototypes(
            np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.asarray([1, 1], dtype=np.int32),
            np.asarray([True, True]),
            k=2,
        )


def test_matching_uses_jointly_finite_features_and_preserves_invalid_vertices():
    fisherz_fc = np.asarray(
        [
            [1.0, np.nan, -1.0, 0.0],
            [-1.0, np.nan, 1.0, np.inf],
            [5.0, 5.0, 5.0, 5.0],
            [1.0, 0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )
    prototypes = np.asarray(
        [
            [1.0, 9.0, -1.0, 0.0],
            [-1.0, 9.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    labels, scores, margin = match_vertices_to_prototypes(
        fisherz_fc,
        np.asarray([True, True, True, False]),
        prototypes,
    )

    assert labels.dtype == np.int32
    assert scores.dtype == np.float32
    assert margin.dtype == np.float32
    assert labels.tolist() == [1, 2, 0, 0]
    assert np.allclose(scores[:2], [[1.0, -1.0], [-1.0, 1.0]])
    assert np.allclose(margin[:2], [2.0, 2.0])
    assert np.isnan(scores[2:]).all()
    assert np.isnan(margin[2:]).all()

#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from run_merged_canonical_group_atlas import (
    PROFILE_GAIN_MIN,
    PROFILE_REDUNDANCY_MAX_SIM,
    profile_gain_against_next,
    profile_redundancy_metrics,
)


def _underfit_rejected(profile_k: np.ndarray, profile_k1: np.ndarray, next_is_actionable: bool) -> bool:
    if not next_is_actionable:
        return False
    gain = profile_gain_against_next(profile_k, profile_k1)
    return bool(float(gain) >= float(PROFILE_GAIN_MIN))


def test_redundancy_detects_duplicate_profiles() -> None:
    profile = np.array(
        [
            [1.0, 0.2, -0.1, 0.0],
            [0.98, 0.18, -0.12, 0.01],
            [-0.3, 0.4, 0.5, -0.2],
        ],
        dtype=np.float32,
    )
    max_sim, redundant_pairs = profile_redundancy_metrics(profile)
    assert max_sim > float(PROFILE_REDUNDANCY_MAX_SIM), (max_sim, redundant_pairs)
    assert redundant_pairs >= 1, (max_sim, redundant_pairs)


def test_underfit_rejected_when_next_adds_distinct_profile() -> None:
    k2 = np.array(
        [
            [1.0, 0.0, 0.0, -0.2],
            [-0.2, 1.0, 0.0, 0.1],
        ],
        dtype=np.float32,
    )
    k3 = np.array(
        [
            [1.0, 0.0, 0.0, -0.2],
            [-0.2, 1.0, 0.0, 0.1],
            [0.0, -0.1, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    assert _underfit_rejected(k2, k3, next_is_actionable=True) is True


def test_underfit_not_rejected_when_next_not_actionable_or_redundant() -> None:
    k2 = np.array(
        [
            [1.0, 0.0, 0.0, -0.2],
            [-0.2, 1.0, 0.0, 0.1],
        ],
        dtype=np.float32,
    )
    k3_redundant = np.array(
        [
            [1.0, 0.0, 0.0, -0.2],
            [-0.2, 1.0, 0.0, 0.1],
            [0.99, 0.01, -0.01, -0.2],
        ],
        dtype=np.float32,
    )
    assert _underfit_rejected(k2, k3_redundant, next_is_actionable=True) is False
    assert _underfit_rejected(k2, k3_redundant, next_is_actionable=False) is False


def main() -> int:
    test_redundancy_detects_duplicate_profiles()
    test_underfit_rejected_when_next_adds_distinct_profile()
    test_underfit_not_rejected_when_next_not_actionable_or_redundant()
    print("ok: redundancy-aware synthetic checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

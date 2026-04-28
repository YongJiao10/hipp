#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from run_merged_canonical_group_atlas import (
    method_uses_kong2022_topn_cluster_annotation,
    network_mean_profile_from_kong2022_roi_profile,
)


def test_method_gate_covers_both_spectral_methods_only() -> None:
    assert method_uses_kong2022_topn_cluster_annotation("group-intrinsic-spectral", "kong2022") is True
    assert method_uses_kong2022_topn_cluster_annotation("group-network-spectral", "kong2022") is True
    assert method_uses_kong2022_topn_cluster_annotation("group-network-top", "kong2022") is False
    assert method_uses_kong2022_topn_cluster_annotation("group-network-spectral", "kong2019") is False


def test_network_mean_profile_from_kong2022_roi_profile_groups_by_canonical8() -> None:
    roi_profile = np.array(
        [
            [1.0, 3.0, 10.0, 14.0],
            [2.0, 6.0, 20.0, 22.0],
        ],
        dtype=np.float32,
    )
    roi_registry = [
        {"roi_id": 1, "roi_name": "a", "hemi": "L", "family17": "f", "subregion": "s", "rank": 1, "canonical8": "DMN"},
        {"roi_id": 2, "roi_name": "b", "hemi": "L", "family17": "f", "subregion": "s", "rank": 2, "canonical8": "DMN"},
        {"roi_id": 3, "roi_name": "c", "hemi": "R", "family17": "f", "subregion": "s", "rank": 3, "canonical8": "VIS"},
        {"roi_id": 4, "roi_name": "d", "hemi": "R", "family17": "f", "subregion": "s", "rank": 4, "canonical8": "VIS"},
    ]
    out = network_mean_profile_from_kong2022_roi_profile(roi_profile, roi_registry, ["DMN", "VIS"])
    expected = np.array(
        [
            [2.0, 12.0],
            [4.0, 21.0],
        ],
        dtype=np.float32,
    )
    assert np.allclose(out, expected, atol=1e-6), (out, expected)


def main() -> int:
    test_method_gate_covers_both_spectral_methods_only()
    test_network_mean_profile_from_kong2022_roi_profile_groups_by_canonical8()
    print("ok: kong2022 shared cluster annotation checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

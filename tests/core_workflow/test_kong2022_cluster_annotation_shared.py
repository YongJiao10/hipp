#!/usr/bin/env python3
from __future__ import annotations

import sys
import importlib.util
import numpy as np
from pathlib import Path

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
method_uses_kong2022_topn_cluster_annotation = atlas_mod.method_uses_kong2022_topn_cluster_annotation
network_mean_profile_from_kong2022_roi_profile = atlas_mod.network_mean_profile_from_kong2022_roi_profile
assignment_status_from_kong2022_roi_profile = atlas_mod.assignment_status_from_kong2022_roi_profile
cluster_assignment_variants_from_kong2022_roi_profile = atlas_mod.cluster_assignment_variants_from_kong2022_roi_profile


def test_method_gate_covers_both_spectral_methods_only() -> None:
    assert method_uses_kong2022_topn_cluster_annotation("group-roi-sim-spectral", "kong2022") is True
    assert method_uses_kong2022_topn_cluster_annotation("intrinsic-spectral", "kong2022") is True
    assert method_uses_kong2022_topn_cluster_annotation("network-spectral", "kong2022") is True
    assert method_uses_kong2022_topn_cluster_annotation("wta-top", "kong2022") is False
    assert method_uses_kong2022_topn_cluster_annotation("group-roi-sim-spectral", "kong2019") is False
    assert method_uses_kong2022_topn_cluster_annotation("network-spectral", "kong2019") is False


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


def test_kong2022_cluster_assignment_uses_mean_fisherz_not_topn_roi_counts() -> None:
    roi_profile = np.array(
        [
            [1.0, 1.0, 0.9, 0.9, 0.9],
        ],
        dtype=np.float32,
    )
    roi_registry = [
        {"roi_id": 1, "roi_name": "dmn1", "hemi": "L", "family17": "f", "subregion": "s", "rank": 1, "canonical8": "Default"},
        {"roi_id": 2, "roi_name": "dmn2", "hemi": "L", "family17": "f", "subregion": "s", "rank": 2, "canonical8": "Default"},
        {"roi_id": 3, "roi_name": "sm1", "hemi": "L", "family17": "f", "subregion": "s", "rank": 3, "canonical8": "Somatomotor"},
        {"roi_id": 4, "roi_name": "sm2", "hemi": "L", "family17": "f", "subregion": "s", "rank": 4, "canonical8": "Somatomotor"},
        {"roi_id": 5, "roi_name": "sm3", "hemi": "L", "family17": "f", "subregion": "s", "rank": 5, "canonical8": "Somatomotor"},
    ]
    profile, status_rows, dominant = assignment_status_from_kong2022_roi_profile(
        roi_profile,
        roi_registry,
        ["Default", "Somatomotor"],
    )
    assert np.allclose(profile, np.array([[1.0, 0.9]], dtype=np.float32), atol=1e-6)
    assert dominant == ["Default"]
    assert status_rows[0]["top1_network"] == "Default"
    assert status_rows[0]["top2_network"] == "Somatomotor"
    assert status_rows[0]["top1_fisher_z"] == 1.0
    assert "top1_count" not in status_rows[0]


def test_kong2022_cluster_assignment_variants_keep_fisherz_and_count_results() -> None:
    roi_profile = np.array(
        [
            [1.0, 1.0, 0.9, 0.9, 0.9],
        ],
        dtype=np.float32,
    )
    roi_profile_3d = np.stack([roi_profile, roi_profile], axis=0).astype(np.float32)
    roi_registry = [
        {"roi_id": 1, "roi_name": "dmn1", "hemi": "L", "family17": "f", "subregion": "s", "rank": 1, "canonical8": "Default"},
        {"roi_id": 2, "roi_name": "dmn2", "hemi": "L", "family17": "f", "subregion": "s", "rank": 2, "canonical8": "Default"},
        {"roi_id": 3, "roi_name": "sm1", "hemi": "L", "family17": "f", "subregion": "s", "rank": 3, "canonical8": "Somatomotor"},
        {"roi_id": 4, "roi_name": "sm2", "hemi": "L", "family17": "f", "subregion": "s", "rank": 4, "canonical8": "Somatomotor"},
        {"roi_id": 5, "roi_name": "sm3", "hemi": "L", "family17": "f", "subregion": "s", "rank": 5, "canonical8": "Somatomotor"},
    ]
    variants = cluster_assignment_variants_from_kong2022_roi_profile(
        roi_profile=roi_profile,
        roi_profile_3d=roi_profile_3d,
        roi_registry=roi_registry,
        network_names=["Default", "Somatomotor"],
        q_thresh=1.0,
        top_n_cap=20,
    )
    assert set(variants) == {"fisherz", "count"}
    assert variants["fisherz"]["assignment_metric"] == "top1_fisher_z"
    assert variants["count"]["assignment_metric"] == "topn_count_positive_roi"
    assert variants["fisherz"]["status_rows"][0]["top1_network"] == "Default"
    assert variants["count"]["status_rows"][0]["top1_network"] == "Somatomotor"


def main() -> int:
    test_method_gate_covers_both_spectral_methods_only()
    test_network_mean_profile_from_kong2022_roi_profile_groups_by_canonical8()
    test_kong2022_cluster_assignment_uses_mean_fisherz_not_topn_roi_counts()
    test_kong2022_cluster_assignment_variants_keep_fisherz_and_count_results()
    print("ok: kong2022 shared cluster annotation checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

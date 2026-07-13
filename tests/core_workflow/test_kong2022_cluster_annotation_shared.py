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
weighted_branch_mod = _load_numbered_module("05_05_build_weighted_fisherz_branch.py", "core_05_05_build_weighted_fisherz_branch")
overview_mod = _load_numbered_module("06_01_build_group_overview.py", "core_06_01_build_group_overview")
method_uses_kong2022_topn_cluster_annotation = atlas_mod.method_uses_kong2022_topn_cluster_annotation
network_mean_profile_from_kong2022_roi_profile = atlas_mod.network_mean_profile_from_kong2022_roi_profile
assignment_status_from_kong2022_roi_profile = atlas_mod.assignment_status_from_kong2022_roi_profile
cluster_assignment_variants_from_kong2022_roi_profile = atlas_mod.cluster_assignment_variants_from_kong2022_roi_profile
resolve_kong2022_network_registry_column = atlas_mod.resolve_kong2022_network_registry_column
build_global_top_connections = atlas_mod.build_global_top_connections
build_normalized_topn_count_vertex_rows = atlas_mod.build_normalized_topn_count_vertex_rows
functional_atlas_from_vertices = atlas_mod.functional_atlas_from_vertices
from lib.weighted_fisherz import (  # noqa: E402
    assign_weighted_network_profile_rows,
    weighted_cluster_network_fisherz_profile_from_timeseries,
    weighted_network_fisherz_profile_from_timeseries,
    weighted_network_profile_from_roi_profile,
    weighted_network_timeseries_from_roi_timeseries,
)
from lib.fc_math import corrcoef_rows, fisher_z_transform_fc  # noqa: E402



def test_overview_output_name_override_shortens_normalized_wta_path(tmp_path: Path) -> None:
    out = overview_mod.overview_png_path(
        present_dir=tmp_path / "wta_normalized",
        atlas="kong2022",
        method="wta-top",
        route_tag="routea_17net",
        mode_tag="family17",
        assignment_variant=None,
        gradient_comp_tag="",
        output_name="family17.png",
    )

    assert out == tmp_path / "wta_normalized" / "family17.png"


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



def test_network_mean_profile_from_kong2022_roi_profile_can_group_by_family17() -> None:
    roi_profile = np.array(
        [
            [1.0, 3.0, 10.0, 14.0],
            [2.0, 6.0, 20.0, 22.0],
        ],
        dtype=np.float32,
    )
    roi_registry = [
        {"roi_id": 1, "roi_name": "a", "hemi": "L", "family17": "DefaultA", "subregion": "s", "rank": 1, "canonical8": "Default"},
        {"roi_id": 2, "roi_name": "b", "hemi": "L", "family17": "DefaultA", "subregion": "s", "rank": 2, "canonical8": "Default"},
        {"roi_id": 3, "roi_name": "c", "hemi": "R", "family17": "DefaultB", "subregion": "s", "rank": 3, "canonical8": "Default"},
        {"roi_id": 4, "roi_name": "d", "hemi": "R", "family17": "VisualA", "subregion": "s", "rank": 4, "canonical8": "Visual"},
    ]
    out = network_mean_profile_from_kong2022_roi_profile(
        roi_profile,
        roi_registry,
        ["DefaultA", "DefaultB", "VisualA"],
        network_column="family17",
    )
    expected = np.array(
        [
            [2.0, 10.0, 14.0],
            [4.0, 20.0, 22.0],
        ],
        dtype=np.float32,
    )
    assert np.allclose(out, expected, atol=1e-6), (out, expected)


def test_network_taxonomy_resolves_registry_column() -> None:
    assert resolve_kong2022_network_registry_column("canonical8") == "canonical8"
    assert resolve_kong2022_network_registry_column("family17") == "family17"
    try:
        resolve_kong2022_network_registry_column("bad")
        raise AssertionError("expected invalid taxonomy to fail")
    except ValueError as exc:
        assert "network taxonomy" in str(exc)


def test_global_top_connections_include_family17_metadata() -> None:
    profile_3d = np.array([[[0.8, -0.7]], [[0.9, -0.8]]], dtype=np.float32)
    roi_registry = [
        {"roi_id": 1, "roi_name": "a", "hemi": "L", "family17": "DefaultA", "subregion": "s", "rank": 1, "canonical8": "Default"},
        {"roi_id": 2, "roi_name": "b", "hemi": "L", "family17": "VisualA", "subregion": "s", "rank": 2, "canonical8": "Visual"},
    ]
    out = build_global_top_connections(
        profile_3d=profile_3d,
        support_indices=np.array([10], dtype=np.int32),
        roi_registry=roi_registry,
        q_thresh=1.0,
        top_n_cap=5,
        network_column="family17",
    )
    assert out["top_positive"][0]["family17"] == "DefaultA"
    assert out["top_negative"][0]["family17"] == "VisualA"


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




def test_weighted_network_profile_uses_roi_vertex_counts_not_roi_equal_mean() -> None:
    roi_profile = np.array([[1.0, 9.0, 10.0]], dtype=np.float32)
    roi_registry = [
        {"canonical8": "Default", "n_grayordinates_used": 1},
        {"canonical8": "Default", "n_grayordinates_used": 3},
        {"canonical8": "Visual", "n_grayordinates_used": 2},
    ]

    weighted = weighted_network_profile_from_roi_profile(
        roi_profile,
        roi_registry,
        ["Default", "Visual"],
    )

    expected = np.array([[(1.0 * 1.0 + 9.0 * 3.0) / 4.0, 10.0]], dtype=np.float32)
    assert np.allclose(weighted, expected, atol=1e-6), (weighted, expected)
    assert not np.allclose(weighted[0, 0], np.float32(5.0), atol=1e-6)


def test_weighted_network_timeseries_matches_direct_vertex_average() -> None:
    roi_ts = np.array(
        [
            [1.0, 3.0, 5.0],
            [9.0, 9.0, 9.0],
            [2.0, 4.0, 6.0],
        ],
        dtype=np.float32,
    )
    roi_rows = [
        {"canonical8": "Default", "n_grayordinates_used": 1},
        {"canonical8": "Default", "n_grayordinates_used": 3},
        {"canonical8": "Visual", "n_grayordinates_used": 2},
    ]

    network_ts = weighted_network_timeseries_from_roi_timeseries(
        roi_ts,
        roi_rows,
        ["Default", "Visual"],
    )

    expected_default = (roi_ts[0] * 1.0 + roi_ts[1] * 3.0) / 4.0
    expected = np.vstack([expected_default, roi_ts[2]]).astype(np.float32)
    assert np.allclose(network_ts, expected, atol=1e-6), (network_ts, expected)




def test_weighted_fisherz_profile_uses_weighted_network_timeseries_before_fc() -> None:
    hipp_ts = np.array(
        [
            [-1.0, 0.0, 1.0, 2.0, 3.0],
            [3.0, 1.0, 0.0, -1.0, -2.0],
        ],
        dtype=np.float32,
    )
    roi_ts = np.array(
        [
            [-1.0, 0.0, 1.0, 2.0, 3.0],
            [2.0, -1.0, 0.5, 1.0, -0.5],
            [3.0, 1.0, 0.0, -1.0, -2.0],
        ],
        dtype=np.float32,
    )
    roi_rows = [
        {"canonical8": "Default", "n_grayordinates_used": 3},
        {"canonical8": "Default", "n_grayordinates_used": 1},
        {"canonical8": "Visual", "n_grayordinates_used": 2},
    ]
    network_names = ["Default", "Visual"]

    got = weighted_network_fisherz_profile_from_timeseries(
        hipp_ts,
        roi_ts,
        roi_rows,
        network_names,
    )

    network_ts = weighted_network_timeseries_from_roi_timeseries(roi_ts, roi_rows, network_names)
    expected = fisher_z_transform_fc(corrcoef_rows(hipp_ts, network_ts))
    roi_fc_then_weight = weighted_network_profile_from_roi_profile(
        fisher_z_transform_fc(corrcoef_rows(hipp_ts, roi_ts)),
        roi_rows,
        network_names,
    )
    assert np.allclose(got, expected, atol=1e-6), (got, expected)
    assert not np.allclose(got[:, 0], roi_fc_then_weight[:, 0], atol=1e-4)


def test_weighted_cluster_profile_averages_cluster_timeseries_before_fc() -> None:
    hipp_ts = np.array(
        [
            [-1.0, 0.0, 1.0, 2.0, 3.0],
            [0.0, 1.0, 1.0, 1.0, 2.0],
            [3.0, 1.0, 0.0, -1.0, -2.0],
        ],
        dtype=np.float32,
    )
    labels = np.array([1, 1, 2], dtype=np.int32)
    support = np.array([True, True, True], dtype=bool)
    roi_ts = np.array(
        [
            [-1.0, 0.0, 1.0, 2.0, 3.0],
            [2.0, -1.0, 0.5, 1.0, -0.5],
        ],
        dtype=np.float32,
    )
    roi_rows = [
        {"canonical8": "Default", "n_grayordinates_used": 3},
        {"canonical8": "Default", "n_grayordinates_used": 1},
    ]

    got = weighted_cluster_network_fisherz_profile_from_timeseries(
        labels,
        support,
        hipp_ts,
        roi_ts,
        roi_rows,
        ["Default"],
    )

    network_ts = weighted_network_timeseries_from_roi_timeseries(roi_ts, roi_rows, ["Default"])
    cluster_ts = np.vstack([hipp_ts[:2].mean(axis=0), hipp_ts[2]])
    expected = fisher_z_transform_fc(corrcoef_rows(cluster_ts, network_ts))
    vertex_fc_mean = np.nanmean(fisher_z_transform_fc(corrcoef_rows(hipp_ts[:2], network_ts)), axis=0)
    assert np.allclose(got, expected, atol=1e-6), (got, expected)
    assert not np.allclose(got[0], vertex_fc_mean, atol=1e-4)


def test_weighted_wta_rows_use_top_network_fisherz_scores() -> None:
    profile = np.array(
        [
            [0.2, 0.9],
            [1.1, 0.4],
        ],
        dtype=np.float32,
    )

    rows = assign_weighted_network_profile_rows(
        profile=profile,
        support_indices=np.array([5, 9], dtype=np.int32),
        network_names=["Default", "Visual"],
    )

    assert rows[0]["vertex_index"] == 5
    assert rows[0]["top1_network"] == "Visual"
    assert rows[0]["top1_fisher_z"] == np.float32(0.9)
    assert rows[0]["top2_network"] == "Default"
    assert "top1_count" not in rows[0]
    assert rows[1]["top1_network"] == "Default"


def test_weighted_branch_parser_accepts_dataset_template_overrides(tmp_path: Path) -> None:
    parser = weighted_branch_mod.build_arg_parser()

    args = parser.parse_args(
        [
            "--source-root",
            str(tmp_path / "source"),
            "--out-root",
            str(tmp_path / "out"),
            "--surface-root",
            str(tmp_path / "surface"),
            "--reference-root",
            str(tmp_path / "routea"),
            "--template-surface-dir",
            str(tmp_path / "templates" / "group_average_surfaces"),
            "--template-spec",
            str(tmp_path / "hippunfold" / "sub-1001_den-512_label-hipp_surfaces.spec"),
        ]
    )

    assert args.template_surface_dir == str(tmp_path / "templates" / "group_average_surfaces")
    assert args.template_spec == str(tmp_path / "hippunfold" / "sub-1001_den-512_label-hipp_surfaces.spec")


def test_weighted_branch_methods_default_is_backward_compatible_and_selectable() -> None:
    parser = weighted_branch_mod.build_arg_parser()

    defaults = parser.parse_args([])
    gradient_only = parser.parse_args(["--methods", "gradient-spectral"])

    assert tuple(defaults.methods) == ("gradient-spectral", "wta-top")
    assert tuple(gradient_only.methods) == ("gradient-spectral",)


def test_weighted_gradient_branch_preserves_shifted_cosine_clustering_provenance() -> None:
    provenance = weighted_branch_mod.weighted_gradient_provenance(
        {
            "surface_affinity": "shifted-cosine",
            "affinity_description": weighted_branch_mod.GRADIENT_SPECTRAL_AFFINITY_DESCRIPTION,
        }
    )

    assert provenance == {
        "clustering_surface_affinity": "shifted-cosine",
        "clustering_affinity_description": weighted_branch_mod.GRADIENT_SPECTRAL_AFFINITY_DESCRIPTION,
        "weighted_assignment_only": True,
    }


def test_weighted_gradient_branch_rejects_nonformal_clustering_provenance() -> None:
    for manifest in (
        {
            "surface_affinity": "relu-cosine",
            "affinity_description": weighted_branch_mod.GRADIENT_SPECTRAL_AFFINITY_DESCRIPTION,
        },
        {"affinity_description": weighted_branch_mod.GRADIENT_SPECTRAL_AFFINITY_DESCRIPTION},
    ):
        with np.testing.assert_raises_regex(
            ValueError,
            "^Inconsistent gradient-spectral affinity provenance$",
        ):
            weighted_branch_mod.weighted_gradient_provenance(manifest)


def test_weighted_gradient_provenance_delegates_to_shared_validator(monkeypatch) -> None:
    calls: list[list[dict[str, object]]] = []

    def shared_validator(manifests: list[dict[str, object]]) -> dict[str, str]:
        calls.append(manifests)
        return {
            "surface_affinity": "shifted-cosine",
            "affinity_description": weighted_branch_mod.GRADIENT_SPECTRAL_AFFINITY_DESCRIPTION,
        }

    monkeypatch.setattr(
        weighted_branch_mod,
        "validate_gradient_spectral_affinity_provenance",
        shared_validator,
    )
    manifest = {"source": "fixture"}

    weighted_branch_mod.weighted_gradient_provenance(manifest)

    assert calls == [[manifest]]


def test_weighted_branch_normalizes_template_paths_from_explicit_dataset_roots(tmp_path: Path) -> None:
    surface_dir = tmp_path / "templates" / "group_average_surfaces"
    surface_dir.mkdir(parents=True)
    left_surface = surface_dir / "group-average_hemi-L_space-corobl_den-512_label-hipp_midthickness.surf.gii"
    left_surface.write_text("surface\n", encoding="utf-8")
    spec = tmp_path / "hippunfold" / "sub-1001_den-512_label-hipp_surfaces.spec"
    spec.parent.mkdir(parents=True)
    spec.write_text("spec\n", encoding="utf-8")

    manifest = weighted_branch_mod.normalize_template_paths(
        {
            "hemi": "L",
            "template_subject": "sub-999999",
            "density_evidence": {"surface_path": "/stale/hcp/surface.surf.gii"},
        },
        template_surface_dir=surface_dir,
        template_spec=spec,
    )

    assert manifest["density_evidence"]["surface_path"] == str(left_surface.resolve())
    assert manifest["template_left_or_right_surface"] == str(left_surface.resolve())
    assert manifest["template_subject"] == "sub-1001"
    assert manifest["template_spec_path"] == str(spec.resolve())


def test_normalized_topn_count_wta_corrects_network_roi_count_bias() -> None:
    roi_registry = []
    for idx in range(2):
        roi_registry.append({"roi_id": idx + 1, "roi_name": f"default{idx}", "canonical8": "Default", "family17": "DefaultA"})
    for idx in range(10):
        roi_registry.append({"roi_id": idx + 3, "roi_name": f"visual{idx}", "canonical8": "Visual", "family17": "VisualA"})
    mean_profile = np.array(
        [
            [0.90, 0.89, 0.88, 0.87, 0.86, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1],
        ],
        dtype=np.float32,
    )

    rows = build_normalized_topn_count_vertex_rows(
        mean_profile=mean_profile,
        support_indices=np.array([42], dtype=np.int32),
        roi_registry=roi_registry,
        top_n_cap=20,
        network_column="canonical8",
    )

    row = rows[0]
    assert row["top1_network"] == "Default"
    assert row["top1_count"] == 2
    assert row["top1_available_rois"] == 2
    assert row["top1_normalized_count"] == 1.0
    assert row["top2_network"] == "Visual"
    assert row["top2_count"] == 3
    assert row["top2_available_rois"] == 10
    assert row["top2_normalized_count"] == 0.3


def test_count_profile_is_reconstructed_from_selected_roi_taxonomy() -> None:
    payload = {
        "items": [
            {
                "cluster_id": 2,
                "top_positive_selected": [
                    {"canonical8": "Default"},
                    {"canonical8": "Default"},
                    {"canonical8": "Visual"},
                ],
            },
            {
                "cluster_id": 1,
                "top_positive_selected": [
                    {"canonical8": "Visual"},
                ],
            },
        ]
    }

    got = overview_mod.count_profile_from_status(
        payload,
        network_names=["Visual", "Default"],
        cluster_ids=[1, 2],
    )

    assert np.array_equal(got, [[1.0, 0.0], [1.0, 2.0]])

def main() -> int:
    test_method_gate_covers_both_spectral_methods_only()
    test_network_mean_profile_from_kong2022_roi_profile_groups_by_canonical8()
    test_kong2022_cluster_assignment_uses_mean_fisherz_not_topn_roi_counts()
    test_kong2022_cluster_assignment_variants_keep_fisherz_and_count_results()
    print("ok: kong2022 shared cluster annotation checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

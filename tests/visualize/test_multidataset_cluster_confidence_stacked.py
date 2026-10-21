"""Focused contracts for the positive cluster-confidence stacked renderer."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RENDERER_PATH = (
    REPO_ROOT
    / "scripts"
    / "visualize"
    / "multidataset_cluster_confidence_stacked"
    / "plot_multidataset_cluster_positive_stacked.py"
)
SPEC = importlib.util.spec_from_file_location(
    "multidataset_cluster_confidence_stacked_renderer", RENDERER_PATH
)
assert SPEC is not None and SPEC.loader is not None
RENDERER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RENDERER)


def test_comparison_branch_resolver_keeps_standard_and_ctxreg_paths_disjoint():
    spec = RENDERER.GridSpec("2mm", "36p", atlas="7net")

    standard = RENDERER.resolve_comparison_visual_columns(spec, "standard")
    ctxreg = RENDERER.resolve_comparison_visual_columns(spec, "ctxreg")

    assert all(entry["signal_branch"] == "standard" for column in standard for entry in column)
    assert all(entry["signal_branch"] == "ctxreg" for column in ctxreg for entry in column)
    assert all("pfm_ctxreg" not in str(entry["grad_subpath"]) for column in standard for entry in column)
    assert all("pfm_ctxreg" in str(entry["grad_subpath"]) for column in ctxreg for entry in column)
    assert RENDERER.comparison_output_tag(spec, "standard") == "hcp_jjtms_7net_2mm_36p_standard"
    assert RENDERER.comparison_output_tag(spec, "ctxreg") == "hcp_jjtms_7net_2mm_36p_ctxreg"


def test_ctxreg_branch_rejects_noncanonical_grid():
    with pytest.raises(ValueError, match="ctxreg.*2mm/36p"):
        RENDERER.resolve_comparison_visual_columns(
            RENDERER.GridSpec("4mm", "36p", atlas="7net"), "ctxreg"
        )


def test_cluster_profile_loader_does_not_fallback_to_status_json(tmp_path):
    hemi_dir = tmp_path / "gradient-spectral" / "kong2022" / "2mm" / "cov_0p80" / "g1g2g3" / "hemi_L"
    hemi_dir.mkdir(parents=True)
    (hemi_dir / "group_atlas_manifest.json").write_text(
        json.dumps({"cluster_assignment_status_fisherz": "cluster_assignment_status_fisherz.json"}),
        encoding="utf-8",
    )
    (hemi_dir / "cluster_assignment_status_fisherz.json").write_text(
        json.dumps(
            {
                "items": [
                    {
                        "cluster_id": 1,
                        "top1_network": "Default",
                        "top1_fisher_z": 0.1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="canonical manifest.*cluster_network_profile"):
        RENDERER.load_cohort_cluster_profiles(tmp_path, "L")


def test_cluster_profile_loader_rejects_cross_branch_manifest(tmp_path):
    hemi_dir = tmp_path / "gradient-spectral" / "kong2022" / "2mm" / "cov_0p80" / "g1g2g3" / "hemi_L"
    hemi_dir.mkdir(parents=True)
    (hemi_dir / "group_atlas_manifest.json").write_text(
        json.dumps({"signal_branch": "standard"}), encoding="utf-8"
    )
    np.save(hemi_dir / "cluster_network_profile.npy", np.zeros((3, 7), dtype=float))
    (hemi_dir / "group_feature_summary.json").write_text(
        json.dumps(
            {
                "network_names": [
                    "Default",
                    "Visual",
                    "Somatomotor",
                    "DorsalAttention",
                    "VentralAttention",
                    "Control",
                    "Language",
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="signal_branch='standard'.*ctxreg"):
        RENDERER.load_cohort_cluster_profiles(tmp_path, "L", signal_branch="ctxreg")


def test_cluster_profile_loader_uses_manifest_declared_profile_and_summary(tmp_path):
    hemi_dir = tmp_path / "gradient-spectral" / "kong2022" / "2mm" / "cov_0p80" / "g1g2g3" / "hemi_L"
    hemi_dir.mkdir(parents=True)
    profile_path = hemi_dir / "cluster_network_profile.npy"
    summary_path = hemi_dir / "group_feature_summary.json"
    np.save(profile_path, np.array([[0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0] * 7, [0.0] * 7]))
    summary_path.write_text(
        json.dumps(
            {
                "network_names": [
                    "Default",
                    "Visual",
                    "Somatomotor",
                    "DorsalAttention",
                    "VentralAttention",
                    "Control",
                    "Language",
                ]
            }
        ),
        encoding="utf-8",
    )
    (hemi_dir / "group_atlas_manifest.json").write_text(
        json.dumps(
            {
                "cluster_network_profile": str(profile_path),
                "group_feature_summary": str(summary_path),
            }
        ),
        encoding="utf-8",
    )

    profiles = RENDERER.load_cohort_cluster_profiles(tmp_path, "L")

    assert profiles[0]["top1_network"] == "Default"
    assert profiles[0]["top1_z"] == pytest.approx(0.3)

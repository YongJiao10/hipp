"""Test suite for multi-dataset 2x2 grid visualization architecture and contract."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LIB_DIR = REPO_ROOT / "scripts" / "visualize" / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from multidataset_cohorts import (
    EXCLUDED_VISUALIZATIONS,
    GRID_VARIANTS,
    PARTICIPATING_VISUALIZATIONS,
    GridSpec,
    check_grid_readiness,
    get_multidataset_cohorts,
    resolve_cohort_paths,
)


def test_grid_variants_contract():
    assert GRID_VARIANTS == ("2mm_36p", "2mm_22p", "4mm_36p", "4mm_22p")


def test_excluded_visualizations_contract():
    expected_exclusions = {
        "multidataset_gradient_subregion_top20_fixed_hcp7t_2col",
        "iowa7t_cortex_master_2row",
        "kong2022_standard_lateral",
        "multidataset_cluster_confidence_negative_stacked",
        "multidataset_subfield_composition_2mm_2col/archive_v1_v3",
    }
    assert expected_exclusions == set(EXCLUDED_VISUALIZATIONS)


def test_participating_visualizations_contract():
    assert len(PARTICIPATING_VISUALIZATIONS) == 6
    for vis in PARTICIPATING_VISUALIZATIONS:
        assert (REPO_ROOT / "scripts" / "visualize" / vis).is_dir()


def test_grid_spec_validation():
    g = GridSpec("2mm", "36p")
    assert g.tag == "2mm_36p"

    g2 = GridSpec.from_tag("4mm_22p")
    assert g2.smoothing == "4mm"
    assert g.is_7net is True
    assert g.file_tag == "2mm_36p"
    assert GridSpec().atlas == "7net"
    assert g2.variant == "22p"

    with pytest.raises(ValueError):
        GridSpec("3mm", "36p")

    with pytest.raises(ValueError):
        GridSpec("2mm", "32p")


def test_cohorts_resolution_all_cells():
    for tag in GRID_VARIANTS:
        spec = GridSpec.from_tag(tag)
        col1, col2 = get_multidataset_cohorts(spec)
        assert len(col1) == 6
        assert len(col2) == 6
        all_cohorts = col1 + col2
        assert len(all_cohorts) == 12

        # Check HCPYoungAdult FIX baseline invariant
        hcp = [c for c in all_cohorts if c["id"] == "hcp_7t"][0]
        assert "kong2022_7net_full_k3_fisherz" in str(hcp["group_root"])
        assert f"/{spec.smoothing}/" in str(hcp["grad_subpath"])

        # Check Iowa variant routing
        iowa_hc = [c for c in all_cohorts if c["id"] == "iowa_hc"][0]
        if spec.variant == "22p":
            assert "kong2022_7net_hc_22p_full_k3_fisherz" in str(iowa_hc["group_root"])
        else:
            assert "kong2022_7net_hc_full_k3_fisherz" in str(iowa_hc["group_root"])
        assert f"/{spec.smoothing}/" in str(iowa_hc["grad_subpath"])

        # Check Iowa newly added cohorts routing
        pure_mdd = [c for c in all_cohorts if c["id"] == "iowa_pure_mdd"][0]
        comorbid_mdd = [c for c in all_cohorts if c["id"] == "iowa_comorbid_mdd"][0]
        assert "kong2022_7net_pure_mdd" in str(pure_mdd["group_root"])
        assert "kong2022_7net_comorbid_mdd" in str(comorbid_mdd["group_root"])

        # Check Stanford variant routing
        stan_nthc = [c for c in all_cohorts if c["id"] == "stan_nthc"][0]
        if spec.variant == "22p":
            assert "routea_7net_nthc_22p_full_k3_fisherz" in str(stan_nthc["group_root"])
        else:
            assert "routea_7net_nthc_full_k3_fisherz" in str(stan_nthc["group_root"])
        assert f"/{spec.smoothing}/" in str(stan_nthc["grad_subpath"])


def test_canonical7_patient_resolver_uses_materialized_7net_root():
    for tag in GRID_VARIANTS:
        spec = GridSpec.from_tag(tag, atlas="7net")
        paths = resolve_cohort_paths("iowa_pat", spec)
        variant_suffix = "_22p" if spec.variant == "22p" else ""
        expected_root = (
            REPO_ROOT
            / "outputs_iowa_jj_tms"
            / "work"
            / "group_atlas"
            / f"routea_patient{variant_suffix}_full_k3_fisherz"
        )
        assert paths["group_root"] == expected_root
        assert paths["grad_subpath"] == (
            expected_root
            / "gradient-spectral"
            / "kong2022"
            / spec.smoothing
            / "cov_0p80"
            / "g1g2g3"
        )


def test_explicit_canonical8_resolver_isolated_under_archive_namespace():
    spec = GridSpec("2mm", "36p", atlas="canonical8")
    paths = resolve_cohort_paths("hcp_7t", spec)

    assert paths["group_root"] == (
        REPO_ROOT
        / "outputs_hcp"
        / "work"
        / "archive"
        / "8net_20260909"
        / "group_atlas"
        / "routea_full_k3_fisherz"
    )
    assert paths["grad_subpath"] == (
        paths["group_root"]
        / "gradient-spectral"
        / "kong2022"
        / "2mm"
        / "cov_0p80"
        / "g1g2g3"
    )
    assert paths["wta_subpath"].is_relative_to(
        REPO_ROOT / "outputs_hcp" / "work" / "archive" / "8net_20260909"
    )


def test_no_active_7net_compatibility_symlinks_remain():
    symlink_paths = (
        REPO_ROOT
        / "outputs_iowa_jj_tms"
        / "work"
        / "group_atlas"
        / "routea_patient_full_k3_fisherz"
        / "gradient-spectral"
        / "kong2022",
        REPO_ROOT
        / "outputs_iowa_jj_tms"
        / "work"
        / "group_atlas"
        / "routea_patient_22p_full_k3_fisherz"
        / "gradient-spectral"
        / "kong2022",
    )
    assert all(not path.is_symlink() for path in symlink_paths)


def test_preflight_readiness_audit():
    # Production baseline (36p) is 100% materialized and ready for all 12 cohorts (both 2mm and 4mm)
    for tag in ("2mm_36p", "4mm_36p"):
        spec = GridSpec.from_tag(tag)
        report = check_grid_readiness(spec)
        assert report["total_cohorts"] == 12
        assert report["ready_count"] == 12
        assert report["all_ready"] is True

    # All canonical7 22p products are materialized and taxonomy-validated.
    for tag in ("2mm_22p", "4mm_22p"):
        spec = GridSpec.from_tag(tag)
        report = check_grid_readiness(spec)
        assert report["total_cohorts"] == 12
        assert report["ready_count"] == 12
        assert report["all_ready"] is True
        assert report["cohorts"]["iowa_hc"]["ready"] is True
        assert report["cohorts"]["iowa_pat"]["ready"] is True
        assert report["cohorts"]["hcp_7t"]["ready"] is True


def test_cli_help_all_participating_scripts():
    py_bin = "/home/yojiao/.conda/envs/py314/bin/python"
    suite_runner = REPO_ROOT / "scripts" / "visualize" / "run_multidataset_suite.py"
    res = subprocess.run([py_bin, str(suite_runner), "--help"], capture_output=True, text=True, check=False)
    assert res.returncode == 0
    assert "--grid" not in res.stdout
    assert "--atlas" in res.stdout
    assert "--figures" in res.stdout
    assert "--smoothing" not in res.stdout
    assert "--variant" not in res.stdout


def test_suite_refresh_defaults_to_all_four_grids_and_rejects_single_grid():
    py_bin = "/home/yojiao/.conda/envs/py314/bin/python"
    suite_runner = REPO_ROOT / "scripts" / "visualize" / "run_multidataset_suite.py"

    default_refresh = subprocess.run(
        [
            py_bin,
            str(suite_runner),
            "--figures",
            "cluster_confidence_stacked",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert default_refresh.returncode == 0
    assert default_refresh.stdout.count("Launching cluster_confidence_stacked") == 4

    # Any attempt to pass --grid (single or all) must fail closed as unrecognized argument
    single_grid_refresh = subprocess.run(
        [
            py_bin,
            str(suite_runner),
            "--figures",
            "cluster_confidence_stacked",
            "--grid",
            "2mm_36p",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert single_grid_refresh.returncode != 0
    assert "unrecognized arguments: --grid" in single_grid_refresh.stderr

    grid_all_refresh = subprocess.run(
        [
            py_bin,
            str(suite_runner),
            "--figures",
            "cluster_confidence_stacked",
            "--grid",
            "all",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert grid_all_refresh.returncode != 0
    assert "unrecognized arguments: --grid" in grid_all_refresh.stderr

    # Readiness check with --grid must also fail closed
    single_grid_check = subprocess.run(
        [
            py_bin,
            str(suite_runner),
            "--check-readiness",
            "--grid",
            "2mm_36p",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert single_grid_check.returncode != 0
    assert "unrecognized arguments: --grid" in single_grid_check.stderr

    # Removed single-grid flags (--smoothing, --variant) must fail closed as unrecognized
    smoothing_attempt = subprocess.run(
        [py_bin, str(suite_runner), "--smoothing", "2mm", "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert smoothing_attempt.returncode != 0
    assert "unrecognized arguments: --smoothing" in smoothing_attempt.stderr

    variant_attempt = subprocess.run(
        [py_bin, str(suite_runner), "--variant", "36p", "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert variant_attempt.returncode != 0
    assert "unrecognized arguments: --variant" in variant_attempt.stderr


def test_7net_grid_spec_and_cohorts_resolution():
    for tag in GRID_VARIANTS:
        spec = GridSpec.from_tag(tag, atlas="7net")
        assert spec.is_7net is True
        assert spec.file_tag == tag
        col1, col2 = get_multidataset_cohorts(spec)
        assert len(col1) == 6
        assert len(col2) == 6
        all_cohorts = col1 + col2
        assert len(all_cohorts) == 12

        # Verify 7net paths
        for c in all_cohorts:
            if c["dataset_key"] == "hcp":
                assert "kong2022_7net_" in str(c["group_root"])
            elif c["id"] == "iowa_pat":
                assert "routea_patient" in str(c["group_root"])
            elif c["dataset_key"] == "iowa_jj_tms":
                assert "kong2022_7net_" in str(c["group_root"])
            else:
                assert "routea_7net_" in str(c["group_root"])
            assert "/kong2022/" in str(c["grad_subpath"])
            assert "/kong2022/" in str(c["wta_subpath"])

        # Verify all 12 cohorts are ready for 7net across all grid cells
        report = check_grid_readiness(spec)
        assert report["total_cohorts"] == 12
        assert report["ready_count"] == 12
        assert report["all_ready"] is True

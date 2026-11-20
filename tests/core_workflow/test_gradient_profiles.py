from __future__ import annotations

import pytest

from lib.gradient_profiles import (
    GRADIENT_PROFILES,
    GRADIENT_COMBINATIONS,
    GradientProfile,
    get_gradient_profile,
    get_gradient_combination_role,
    is_gradient_combination,
    primary_combination,
    baseline_combination,
    comparison_display_combinations,
    validate_gradient_combination,
    # Backward compatibility aliases
    FORMAL_PROFILES,
    FormalProfile,
    get_formal_profile,
    validate_formal_combination,
)


def test_gradient_profiles_definitions():
    assert "topfc" in GRADIENT_PROFILES
    assert "topsim" in GRADIENT_PROFILES

    topfc = get_gradient_profile("topfc")
    assert topfc.name == "topfc"
    assert topfc.earlier_label == "B"
    assert topfc.feature_scaling == "none"
    assert topfc.sparsify_stage == "features"
    assert topfc.sparsity == 0.90
    assert topfc.role == "historical_baseline"

    topsim = get_gradient_profile("topsim")
    assert topsim.name == "topsim"
    assert topsim.earlier_label == "C"
    assert topsim.feature_scaling == "none"
    assert topsim.sparsify_stage == "affinity"
    assert topsim.sparsity == 0.90
    assert topsim.role == "primary"


def test_backward_compatibility_aliases():
    assert FORMAL_PROFILES is GRADIENT_PROFILES
    assert FormalProfile is GradientProfile
    assert get_formal_profile("topsim") == get_gradient_profile("topsim")
    assert validate_formal_combination("topsim", "gradient-ward") == "primary"


def test_retired_routes_not_in_gradient_profiles():
    for retired in ("route_a", "route_d", "A", "D"):
        with pytest.raises((KeyError, ValueError)):
            get_gradient_profile(retired)


def test_gradient_combinations():
    assert is_gradient_combination("topfc", "gradient-spectral")
    assert is_gradient_combination("topsim", "gradient-ward")
    assert is_gradient_combination("topsim", "gradient-gmm")

    assert not is_gradient_combination("topfc", "gradient-ward")
    assert not is_gradient_combination("topfc", "gradient-gmm")
    assert not is_gradient_combination("topsim", "gradient-spectral")

    assert get_gradient_combination_role("topfc", "gradient-spectral") == "historical_baseline"
    assert get_gradient_combination_role("topsim", "gradient-ward") == "primary"
    assert get_gradient_combination_role("topsim", "gradient-gmm") == "comparison_display"


def test_primary_and_baseline_selection():
    prof, method = primary_combination()
    assert prof == "topsim"
    assert method == "gradient-ward"

    b_prof, b_method = baseline_combination()
    assert b_prof == "topfc"
    assert b_method == "gradient-spectral"

    disp = comparison_display_combinations()
    assert disp == [("topsim", "gradient-gmm"), ("topsim", "mean-similarity-ward"),
                    ("topsim", "subject-gradient-ward"), ("topsim", "subject-gradient-hbp")]


def test_validate_gradient_combination():
    validate_gradient_combination("topfc", "gradient-spectral")
    validate_gradient_combination("topsim", "gradient-ward")
    validate_gradient_combination("topsim", "gradient-gmm")

    with pytest.raises(ValueError):
        validate_gradient_combination("route_a", "gradient-spectral")

    with pytest.raises(ValueError):
        validate_gradient_combination("topsim", "gradient-spectral-rbf")


def test_output_layout_with_gradient_profile():
    from lib.output_layout import (
        resolve_cohort_gradient_root,
        resolve_cohort_group_atlas_root,
        resolve_cohort_overview_root,
    )
    grad_topfc = resolve_cohort_gradient_root(route="pfm", profile="topfc")
    assert grad_topfc.parts[-2:] == ("pfm", "topfc")

    grad_topsim = resolve_cohort_gradient_root(route="pfm", profile="topsim")
    assert grad_topsim.parts[-2:] == ("pfm", "topsim")

    atlas_topfc = resolve_cohort_group_atlas_root(route="pfm", profile="topfc")
    assert atlas_topfc.parts[-2:] == ("pfm", "topfc")

    atlas_topsim = resolve_cohort_group_atlas_root(route="pfm", profile="topsim")
    assert atlas_topsim.parts[-2:] == ("pfm", "topsim")

    overview_topsim = resolve_cohort_overview_root(route="pfm", profile="topsim")
    assert overview_topsim.parts[-2:] == ("pfm", "topsim")

    # When profile is None, unchanged backward compatibility
    grad_none = resolve_cohort_gradient_root(route="pfm")
    assert grad_none.name == "pfm"

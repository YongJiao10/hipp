"""Hippocampal Gradient Parcellation Profiles and Roles.

Per GRADIENT_PARCELLATION_FORMAL_WORKFLOW_PLAN.md:
- TopFC: baseline. Features without column scaling, row-sparsified to top 10% (sparsity=0.90),
  shifted-cosine spectral clustering.
- TopSim: primary. Features without column scaling, Pearson affinity row-sparsified to top 10%
  (sparsity=0.90), connected Ward clustering.
- TopSim GMM: comparison display only. Same TopSim features/affinity, GMM clustering.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ProfileRole = Literal["primary", "historical_baseline", "comparison_display"]
GRADIENT_RANDOM_STATE = 20260420
# Route-C comparison split/null fits use the outer repetition seed for the
# first half and the immediately following seed for the second half.
RESAMPLE_HALF_SEED_OFFSETS = (0, 1)


@dataclass(frozen=True)
class GradientProfile:
    name: str
    earlier_label: str
    feature_scaling: str
    sparsify_stage: str
    sparsity: float
    role: ProfileRole
    description: str


TOPFC_PROFILE = GradientProfile(
    name="topfc",
    earlier_label="B",
    feature_scaling="none",
    sparsify_stage="features",
    sparsity=0.90,
    role="historical_baseline",
    description="Existing vertex-to-network FC features without column scaling, row-sparsified to top 10%",
)

TOPSIM_PROFILE = GradientProfile(
    name="topsim",
    earlier_label="C",
    feature_scaling="none",
    sparsify_stage="affinity",
    sparsity=0.90,
    role="primary",
    description="Unscaled Pearson affinity row-sparsified to top 10%",
)

GRADIENT_PROFILES: dict[str, GradientProfile] = {
    "topfc": TOPFC_PROFILE,
    "topsim": TOPSIM_PROFILE,
}
GRADIENT_PROFILE_CHOICES = tuple(sorted(GRADIENT_PROFILES.keys()))


def gradient_profile_config(name: str) -> dict[str, object]:
    prof = get_gradient_profile(name)
    return {
        "feature_scaling": prof.feature_scaling,
        "sparsify_stage": prof.sparsify_stage,
        "sparsity": prof.sparsity,
    }


def gradient_profile_role(name: str) -> str:
    return get_gradient_profile(name).role


def gradient_profile_earlier_label(name: str) -> str:
    return get_gradient_profile(name).earlier_label


GRADIENT_COMBINATIONS: dict[tuple[str, str], ProfileRole] = {
    ("topfc", "gradient-spectral"): "historical_baseline",
    ("topsim", "gradient-ward"): "primary",
    ("topsim", "gradient-gmm"): "comparison_display",
}


def get_gradient_profile(name: str) -> GradientProfile:
    key = str(name).strip().lower()
    if key not in GRADIENT_PROFILES:
        raise ValueError(
            f"Unknown gradient profile: {name!r}; valid profiles are: {sorted(GRADIENT_PROFILES.keys())}"
        )
    return GRADIENT_PROFILES[key]


def is_gradient_combination(profile: str, method: str) -> bool:
    prof_key = str(profile).strip().lower()
    meth_key = str(method).strip().lower()
    return (prof_key, meth_key) in GRADIENT_COMBINATIONS


def get_gradient_combination_role(profile: str, method: str) -> ProfileRole:
    prof_key = str(profile).strip().lower()
    meth_key = str(method).strip().lower()
    pair = (prof_key, meth_key)
    if pair not in GRADIENT_COMBINATIONS:
        raise ValueError(
            f"Unsupported profile/method combination: profile={profile!r}, method={method!r}; "
            f"valid combinations are: {list(GRADIENT_COMBINATIONS.keys())}"
        )
    return GRADIENT_COMBINATIONS[pair]


def validate_gradient_combination(profile: str, method: str) -> ProfileRole:
    return get_gradient_combination_role(profile, method)


def primary_combination() -> tuple[str, str]:
    return ("topsim", "gradient-ward")


def baseline_combination() -> tuple[str, str]:
    return ("topfc", "gradient-spectral")


def comparison_display_combinations() -> list[tuple[str, str]]:
    return [
        (prof, meth)
        for (prof, meth), role in GRADIENT_COMBINATIONS.items()
        if role == "comparison_display"
    ]


# Backward compatibility aliases
FormalProfile = GradientProfile
FORMAL_GRADIENT_RANDOM_STATE = GRADIENT_RANDOM_STATE
FORMAL_RESAMPLE_HALF_SEED_OFFSETS = RESAMPLE_HALF_SEED_OFFSETS
FORMAL_PROFILES = GRADIENT_PROFILES
FORMAL_PROFILE_CHOICES = GRADIENT_PROFILE_CHOICES
FORMAL_COMBINATIONS = GRADIENT_COMBINATIONS
formal_profile_config = gradient_profile_config
formal_profile_role = gradient_profile_role
formal_profile_earlier_label = gradient_profile_earlier_label
get_formal_profile = get_gradient_profile
is_formal_combination = is_gradient_combination
get_formal_combination_role = get_gradient_combination_role
validate_formal_combination = validate_gradient_combination

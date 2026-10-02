"""Centralized multi-dataset cohort metadata and 2x2 grid path resolution.

Supported grid dimensions:
  - smoothing: '2mm' | '4mm'
  - variant: '36p' | '22p'
  (Four combinations: 2mm_36p, 2mm_22p, 4mm_36p, 4mm_22p)

Excluded from the 2x2 grid matrix:
  - multidataset_gradient_subregion_top20_fixed_hcp7t_2col
  - iowa7t_cortex_master_2row
  - kong2022_standard_lateral
  - multidataset_cluster_confidence_negative_stacked
  - multidataset_subfield_composition_2mm_2col/archive_v1_v3
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
CORE_LIB_DIR = REPO_ROOT / "scripts" / "core_workflow"
if str(CORE_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_LIB_DIR))

from lib.group_atlas_lineage import validate_group_atlas_lineage

GRID_VARIANTS: tuple[str, ...] = ("2mm_36p", "2mm_22p", "4mm_36p", "4mm_22p")
SIGNAL_BRANCHES: tuple[str, ...] = ("standard", "ctxreg")

EXCLUDED_VISUALIZATIONS: tuple[str, ...] = (
    "multidataset_gradient_subregion_top20_fixed_hcp7t_2col",
    "iowa7t_cortex_master_2row",
    "kong2022_standard_lateral",
    "multidataset_cluster_confidence_negative_stacked",
    "multidataset_subfield_composition_2mm_2col/archive_v1_v3",
)

PARTICIPATING_VISUALIZATIONS: tuple[str, ...] = (
    "multidataset_cluster_confidence_stacked",
    "multidataset_gradient_subregion_top20_stacked_2col",
    "multidataset_hipp_cortex_gradient_wta",
    "multidataset_subfield_composition_2mm_2col",
    "multidataset_workbench_hippocampus_overlay",
    "multidataset_wta_normalized_longaxis_stacked_2col",
)

# Canonical method-comparison branch. Kong2022 WTA uses the ROI-registry
# normalized_top20_count assignment; Lynch2024 WTA uses direct network
# Fisher-z assignments from the vertex-to-network FC profile.
COMPARISON_VERSION = "hcp_jjtms_7net"
COMPARISON_METHODS: tuple[str, ...] = ("kong2022_7net", "lynch2024_7net")


@dataclass(frozen=True)
class GridSpec:
    """Specification for a legacy grid cell or an explicit signal branch."""

    smoothing: str = "2mm"
    variant: str = "36p"
    atlas: str = "7net"
    signal_branch: str = "standard"

    def __post_init__(self) -> None:
        if self.smoothing not in ("2mm", "4mm"):
            raise ValueError(
                f"smoothing must be '2mm' or '4mm', got {self.smoothing!r}"
            )
        if self.variant not in ("36p", "22p"):
            raise ValueError(f"variant must be '36p' or '22p', got {self.variant!r}")
        if self.atlas not in ("kong2022", "7net", "canonical7", "canonical8"):
            raise ValueError(
                f"atlas must be 'kong2022' or a canonical taxonomy alias, got {self.atlas!r}"
            )
        branch = self.signal_branch
        if branch not in SIGNAL_BRANCHES:
            raise ValueError(
                f"signal_branch must be one of {SIGNAL_BRANCHES}, "
                f"got {self.signal_branch!r}"
            )
        if branch == "ctxreg" and (self.smoothing, self.variant) != ("2mm", "36p"):
            raise ValueError(
                "ctxreg signal branch requires the canonical 2mm/36p grid; "
                f"got {self.smoothing}/{self.variant}"
            )
        if branch == "ctxreg" and not self.is_7net:
            raise ValueError(
                "ctxreg signal branch requires the canonical 7-network atlas; "
                f"got atlas={self.atlas!r}"
            )

    @property
    def is_7net(self) -> bool:
        return self.atlas in ("kong2022", "7net", "canonical7")

    @property
    def uses_legacy_7net_namespace(self) -> bool:
        """Semantic 7net aliases always resolve to the unsuffixed default namespace."""
        return False

    @property
    def tag(self) -> str:
        base = f"{self.smoothing}_{self.variant}"
        branch = self.signal_branch
        return base if branch == "standard" else f"{base}_{branch}"

    @property
    def file_tag(self) -> str:
        return f"8net_{self.tag}" if self.atlas == "canonical8" else self.tag

    @classmethod
    def from_tag(cls, tag: str, atlas: str | None = None) -> GridSpec:
        parsed_atlas = atlas or "7net"
        clean_tag = tag
        if clean_tag.startswith("7net_"):
            clean_tag = clean_tag[5:]
            parsed_atlas = "7net"
        elif clean_tag.endswith("_7net"):
            clean_tag = clean_tag[:-5]
            parsed_atlas = "7net"
        signal_branch = "standard"
        if clean_tag.endswith("_standard"):
            clean_tag = clean_tag.removesuffix("_standard")
            signal_branch = "standard"
        elif clean_tag.endswith("_ctxreg"):
            clean_tag = clean_tag.removesuffix("_ctxreg")
            signal_branch = "ctxreg"
        if clean_tag not in GRID_VARIANTS:
            raise ValueError(
                f"Invalid grid tag {tag!r}. Must be one of "
                f"{GRID_VARIANTS} or 2mm_36p_ctxreg"
            )
        parts = clean_tag.split("_", 1)
        return cls(
            smoothing=parts[0],
            variant=parts[1],
            atlas=parsed_atlas,
            signal_branch=signal_branch,
        )


# Base metadata for the 10 cohorts in canonical order
COHORT_SPECS_COL1: list[dict[str, Any]] = [
    {
        "id": "hcp_7t",
        "dataset_key": "hcp",
        "dataset_name": "HCP 7T",
        "cohort_name": "HCP 7T",
        "cohort_desc": "Healthy Young Adults",
        "n_subjects": 151,
        "bg_color": "#ffffff",
        "badge_bg": "#0b3c5d",
        "badge_fg": "#ffffff",
        "col": 1,
    },
    {
        "id": "iowa_hc",
        "dataset_key": "iowa_jj_tms",
        "dataset_name": "Iowa 3T",
        "cohort_name": "HC",
        "cohort_desc": "Healthy Controls",
        "n_subjects": 67,
        "bg_color": "#f7faff",
        "badge_bg": "#1d5b79",
        "badge_fg": "#ffffff",
        "col": 1,
    },
    {
        "id": "iowa_pat",
        "dataset_key": "iowa_jj_tms",
        "dataset_name": "Iowa 3T",
        "cohort_name": "Patient",
        "cohort_desc": "Depression Patients",
        "n_subjects": 26,
        "bg_color": "#f7faff",
        "badge_bg": "#1d5b79",
        "badge_fg": "#ffffff",
        "col": 1,
    },
    {
        "id": "iowa_pure_mdd",
        "dataset_key": "iowa_jj_tms",
        "dataset_name": "Iowa 3T",
        "cohort_name": "Pure MDD",
        "cohort_desc": "Pure MDD Subgroup",
        "n_subjects": 7,
        "bg_color": "#f7faff",
        "badge_bg": "#1d5b79",
        "badge_fg": "#ffffff",
        "col": 1,
    },
    {
        "id": "iowa_comorbid_mdd",
        "dataset_key": "iowa_jj_tms",
        "dataset_name": "Iowa 3T",
        "cohort_name": "Comorbid MDD",
        "cohort_desc": "Comorbid MDD Subgroup",
        "n_subjects": 19,
        "bg_color": "#f7faff",
        "badge_bg": "#1d5b79",
        "badge_fg": "#ffffff",
        "col": 1,
    },
    {
        "id": "stan_nthc",
        "dataset_key": "stanford",
        "dataset_name": "Stanford",
        "cohort_name": "NTHC",
        "cohort_desc": "Non-Trauma Controls",
        "n_subjects": 51,
        "bg_color": "#fffdf8",
        "badge_bg": "#8c4600",
        "badge_fg": "#ffffff",
        "col": 1,
    },
]

COHORT_SPECS_COL2: list[dict[str, Any]] = [
    {
        "id": "stan_tehc",
        "dataset_key": "stanford",
        "dataset_name": "Stanford",
        "cohort_name": "TEHC",
        "cohort_desc": "Trauma-Exposed Controls",
        "n_subjects": 40,
        "bg_color": "#fffdf8",
        "badge_bg": "#8c4600",
        "badge_fg": "#ffffff",
        "col": 2,
    },
    {
        "id": "stan_nts",
        "dataset_key": "stanford",
        "dataset_name": "Stanford",
        "cohort_name": "NTS",
        "cohort_desc": "Non-Trauma Symptomatic",
        "n_subjects": 56,
        "bg_color": "#fffdf8",
        "badge_bg": "#8c4600",
        "badge_fg": "#ffffff",
        "col": 2,
    },
    {
        "id": "stan_nts_mdd",
        "dataset_key": "stanford",
        "dataset_name": "Stanford",
        "cohort_name": "NTS Any MDD",
        "cohort_desc": "Non-Trauma MDD (All)",
        "n_subjects": 43,
        "bg_color": "#fffdf8",
        "badge_bg": "#8c4600",
        "badge_fg": "#ffffff",
        "col": 2,
    },
    {
        "id": "stan_nts_pure",
        "dataset_key": "stanford",
        "dataset_name": "Stanford",
        "cohort_name": "NTS Pure MDD",
        "cohort_desc": "Non-Trauma Pure MDD",
        "n_subjects": 11,
        "bg_color": "#fffdf8",
        "badge_bg": "#8c4600",
        "badge_fg": "#ffffff",
        "col": 2,
    },
    {
        "id": "stan_tis",
        "dataset_key": "stanford",
        "dataset_name": "Stanford",
        "cohort_name": "TIS",
        "cohort_desc": "Trauma-Exposed Symptomatic",
        "n_subjects": 47,
        "bg_color": "#fffdf8",
        "badge_bg": "#8c4600",
        "badge_fg": "#ffffff",
        "col": 2,
    },
    {
        "id": "stan_tis_ptsd",
        "dataset_key": "stanford",
        "dataset_name": "Stanford",
        "cohort_name": "TIS PTSD",
        "cohort_desc": "Trauma-Informed PTSD",
        "n_subjects": 35,
        "bg_color": "#fffdf8",
        "badge_bg": "#8c4600",
        "badge_fg": "#ffffff",
        "col": 2,
    },
]


_CTXREG_IOWA_STEMS: dict[str, str] = {
    "iowa_hc": "hc",
    "iowa_pat": "patient",
    "iowa_pure_mdd": "pure_mdd",
    "iowa_comorbid_mdd": "comorbid_mdd",
}
_CTXREG_STANFORD_STEMS: dict[str, str] = {
    "stan_nthc": "nthc",
    "stan_tehc": "tehc",
    "stan_nts": "nts",
    "stan_nts_mdd": "nts_any_mdd",
    "stan_nts_pure": "nts_pure_mdd",
    "stan_tis": "tis",
    "stan_tis_ptsd": "tis_ptsd",
}


def _resolve_ctxreg_cohort_paths(cohort_id: str, spec: GridSpec) -> dict[str, Path]:
    """Resolve the explicit ctxreg branch without consulting standard paths."""
    if cohort_id == "hcp_7t":
        dataset_root = REPO_ROOT / "outputs_hcp"
        group_root = dataset_root / "work" / "group_atlas" / "routea_ctxreg"
        present_dir = dataset_root / "present" / "overviews" / "routea_ctxreg"
        shared_root = (
            dataset_root
            / "inputs"
            / "hcp"
            / "kong2022"
            / "shared_inputs"
            / "routea_ctxreg"
        )
        standard_surface_root = (
            dataset_root / "inputs" / "hcp" / "kong2022" / "shared_inputs" / "surface"
        )
        reference_root = (
            dataset_root / "inputs" / "hcp" / "kong2022" / "shared_inputs" / "routea"
        )
        hippunfold_root = (
            dataset_root / "inputs" / "hcp" / "hippunfold" / "dense_corobl_batch"
        )
        stem = None
    elif cohort_id in _CTXREG_IOWA_STEMS:
        dataset_root = REPO_ROOT / "outputs_iowa_jj_tms"
        stem = _CTXREG_IOWA_STEMS[cohort_id]
        group_root = dataset_root / "work" / "group_atlas" / "routea_ctxreg" / stem
        present_dir = dataset_root / "present" / "overviews" / "routea_ctxreg" / stem
        shared_root = (
            dataset_root
            / "inputs"
            / "iowa_jj_tms"
            / "kong2022"
            / "shared_inputs"
            / "routea_ctxreg"
        )
        standard_surface_root = (
            dataset_root
            / "inputs"
            / "iowa_jj_tms"
            / "kong2022"
            / "shared_inputs"
            / "surface"
        )
        reference_root = (
            dataset_root
            / "inputs"
            / "iowa_jj_tms"
            / "kong2022"
            / "shared_inputs"
            / "routea"
        )
        hippunfold_root = (
            dataset_root
            / "inputs"
            / "iowa_jj_tms"
            / "hippunfold"
            / "dense_corobl_batch"
        )
    elif cohort_id in _CTXREG_STANFORD_STEMS:
        dataset_root = REPO_ROOT / "outputs_stanford"
        stem = _CTXREG_STANFORD_STEMS[cohort_id]
        group_root = dataset_root / "work" / "group_atlas" / "routea_ctxreg" / stem
        present_dir = dataset_root / "present" / "overviews" / "routea_ctxreg" / stem
        shared_root = (
            dataset_root
            / "inputs"
            / "stanford"
            / "kong2022"
            / "shared_inputs"
            / "routea_ctxreg"
            / stem
        )
        standard_surface_root = (
            dataset_root
            / "inputs"
            / "stanford"
            / "kong2022"
            / "shared_inputs"
            / "surface"
        )
        reference_root = (
            dataset_root
            / "inputs"
            / "stanford"
            / "kong2022"
            / "shared_inputs"
            / "routea"
            / stem
        )
        hippunfold_root = (
            dataset_root / "inputs" / "stanford" / "hippunfold" / "dense_corobl_batch"
        )
    else:
        raise ValueError(f"Unknown cohort id: {cohort_id!r}")

    grad_subpath = (
        group_root
        / "gradient-spectral"
        / "kong2022"
        / spec.smoothing
        / "cov_0p80"
        / "g1g2g3"
    )
    wta_subpath = group_root / "wta-top" / "kong2022" / spec.smoothing / "cov_0p80"
    overview_prefix = "kong2022"
    overview_grad_name = (
        f"{overview_prefix}_gradient-spectral_group_overview_g1g2g3_"
        "routea_ctxreg_fisherz.png"
    )
    overview_grad_normalized_name = (
        f"{overview_prefix}_gradient-spectral_group_overview_g1g2g3_"
        "routea_ctxreg_normalized_top20_count.png"
    )
    overview_wta_name = (
        f"{overview_prefix}_wta-top_group_overview_routea_ctxreg_"
        "normalized_top20_count.png"
    )
    return {
        "group_root": group_root,
        "group_root_grad": group_root,
        "group_root_wta": group_root,
        "grad_subpath": grad_subpath,
        "wta_subpath": wta_subpath,
        "present_dir": present_dir,
        "overview_grad": present_dir / overview_grad_name,
        "overview_grad_normalized_top20_count": (
            present_dir / overview_grad_normalized_name
        ),
        "overview_wta": present_dir / overview_wta_name,
        "shared_root": shared_root,
        "reference_root": reference_root,
        "surface_root": shared_root,
        "mask_root": standard_surface_root,
        "hippunfold_root": hippunfold_root,
    }


def resolve_cohort_paths(cohort_id: str, spec: GridSpec) -> dict[str, Path]:
    """Resolve group atlas, overview, and input paths for a cohort given a GridSpec."""
    sm = spec.smoothing
    var = spec.variant

    if spec.signal_branch == "ctxreg":
        return _resolve_ctxreg_cohort_paths(cohort_id, spec)

    if spec.is_7net:
        if cohort_id == "hcp_7t":
            grp_base = (
                REPO_ROOT / "outputs_hcp/work/group_atlas/kong2022_7net_full_k3_fisherz"
            )
            pres_base = REPO_ROOT / "outputs_hcp/present/overviews/kong2022_7net"
            return {
                "group_root": grp_base,
                "group_root_grad": grp_base,
                "group_root_wta": grp_base,
                "grad_subpath": grp_base
                / f"gradient-spectral/kong2022/{sm}/cov_0p80/g1g2g3",
                "wta_subpath": grp_base / f"wta-top/kong2022/{sm}/cov_0p80",
                "present_dir": pres_base,
                "overview_grad": pres_base
                / "kong2022_7net_gradient-spectral_group_overview_g1g2g3_kong2022_7net_fisherz.png",
                "overview_grad_normalized_top20_count": pres_base
                / "kong2022_7net_gradient-spectral_group_overview_g1g2g3_kong2022_7net_normalized_top20_count.png",
                "overview_wta": pres_base
                / "kong2022_7net_wta-top_group_overview_kong2022_7net_normalized_top20_count.png",
                "hippunfold_root": REPO_ROOT
                / "outputs_hcp/inputs/hcp/hippunfold/dense_corobl_batch",
            }

        iowa_map = {
            "iowa_hc": "hc",
            "iowa_pat": "patient",
            "iowa_pure_mdd": "pure_mdd",
            "iowa_comorbid_mdd": "comorbid_mdd",
        }
        if cohort_id in iowa_map:
            stem = iowa_map[cohort_id]
            iowa_var_suffix = "_22p" if var == "22p" else ""
            # Patient was rebuilt in the retained routea_patient roots; the
            # other Iowa seven-network cohorts keep their kong2022_7net roots.
            branch = (
                f"routea_patient{iowa_var_suffix}_full_k3_fisherz"
                if cohort_id == "iowa_pat"
                else f"kong2022_7net_{stem}{iowa_var_suffix}_full_k3_fisherz"
            )
            grp_base = REPO_ROOT / f"outputs_iowa_jj_tms/work/group_atlas/{branch}"
            pres_variant = "22p" if var == "22p" else "36p"
            pres_base = (
                REPO_ROOT
                / f"outputs_iowa_jj_tms/present/overviews/kong2022_7net/{pres_variant}/{stem}"
            )

            return {
                "group_root": grp_base,
                "group_root_grad": grp_base,
                "group_root_wta": grp_base,
                "grad_subpath": grp_base
                / f"gradient-spectral/kong2022/{sm}/cov_0p80/g1g2g3",
                "wta_subpath": grp_base / f"wta-top/kong2022/{sm}/cov_0p80",
                "present_dir": pres_base,
                "overview_grad": pres_base
                / "kong2022_7net_gradient-spectral_group_overview_g1g2g3_kong2022_7net_fisherz.png",
                "overview_grad_normalized_top20_count": pres_base
                / "kong2022_7net_gradient-spectral_group_overview_g1g2g3_kong2022_7net_normalized_top20_count.png",
                "overview_wta": pres_base
                / "kong2022_7net_wta-top_group_overview_kong2022_7net_normalized_top20_count.png",
                "hippunfold_root": REPO_ROOT
                / "outputs_iowa_jj_tms/inputs/iowa_jj_tms/hippunfold/dense_corobl_batch",
            }

        stan_map = {
            "stan_nthc": "nthc",
            "stan_tehc": "tehc",
            "stan_nts": "nts",
            "stan_nts_mdd": "nts_any_mdd",
            "stan_nts_pure": "nts_pure_mdd",
            "stan_tis": "tis",
            "stan_tis_ptsd": "tis_ptsd",
        }
        if cohort_id in stan_map:
            stem = stan_map[cohort_id]
            stan_var_suffix = "_22p" if var == "22p" else ""
            branch = f"routea_7net_{stem}{stan_var_suffix}_full_k3_fisherz"
            grp_base = REPO_ROOT / f"outputs_stanford/work/group_atlas/{branch}"
            pres_dir_name = "routea_7net_22p" if var == "22p" else "routea_7net"
            pres_base = (
                REPO_ROOT / f"outputs_stanford/present/overviews/{pres_dir_name}/{stem}"
            )

            return {
                "group_root": grp_base,
                "group_root_grad": grp_base,
                "group_root_wta": grp_base,
                "grad_subpath": grp_base
                / f"gradient-spectral/kong2022/{sm}/cov_0p80/g1g2g3",
                "wta_subpath": grp_base / f"wta-top/kong2022/{sm}/cov_0p80",
                "present_dir": pres_base,
                "overview_grad": pres_base
                / "kong2022_gradient-spectral_group_overview_g1g2g3_routea_7net_fisherz.png",
                "overview_grad_normalized_top20_count": pres_base
                / "kong2022_gradient-spectral_group_overview_g1g2g3_routea_7net_normalized_top20_count.png",
                "overview_wta": pres_base
                / "kong2022_wta-top_group_overview_routea_7net_normalized_top20_count.png",
                "hippunfold_root": REPO_ROOT
                / "outputs_stanford/inputs/stanford/hippunfold/dense_corobl_batch",
            }

        raise ValueError(f"Unknown cohort id: {cohort_id!r}")

    if cohort_id == "hcp_7t":
        archive_root = REPO_ROOT / "outputs_hcp/work/archive/8net_20260909/group_atlas"
        grp_base = archive_root / "routea_full_k3_fisherz"
        wta_base = archive_root / "routea_wta_normalized_top20_count_full_k3"
        return {
            "group_root": grp_base,
            "group_root_grad": grp_base,
            "group_root_wta": wta_base,
            "grad_subpath": grp_base
            / f"gradient-spectral/kong2022/{sm}/cov_0p80/g1g2g3",
            "wta_subpath": wta_base / f"wta-top/kong2022/{sm}/cov_0p80",
            "present_dir": REPO_ROOT / "outputs_hcp/present/overviews",
            "overview_grad": REPO_ROOT
            / "outputs_hcp/present/overviews/gradient_spectral/g1g2g3_bilateral.png",
            "overview_grad_normalized_top20_count": REPO_ROOT
            / "outputs_hcp/present/overviews/kong2022_gradient-spectral_group_overview_g1g2g3_routea_normalized_top20_count.png",
            "overview_wta": REPO_ROOT
            / "outputs_hcp/present/overviews/wta_normalized/top20_count.png",
            "hippunfold_root": REPO_ROOT
            / "outputs_hcp/inputs/hcp/hippunfold/dense_corobl_batch",
        }

    # Iowa 3T cohorts: HC, Patient, Pure MDD, Comorbid MDD
    iowa_map = {
        "iowa_hc": "hc",
        "iowa_pat": "patient",
        "iowa_pure_mdd": "pure_mdd",
        "iowa_comorbid_mdd": "comorbid_mdd",
    }
    if cohort_id in iowa_map:
        stem = iowa_map[cohort_id]
        iowa_var_suffix = "_22p" if var == "22p" else ""
        branch = f"routea_{stem}{iowa_var_suffix}_full_k3_fisherz"
        grp_base = (
            REPO_ROOT
            / f"outputs_iowa_jj_tms/work/archive/8net_20260909/group_atlas/{branch}"
        )
        pres_base = (
            REPO_ROOT
            / f"outputs_iowa_jj_tms/present/overviews/routea_{stem}{iowa_var_suffix}"
        )

        return {
            "group_root": grp_base,
            "group_root_grad": grp_base,
            "group_root_wta": grp_base,
            "grad_subpath": grp_base
            / f"gradient-spectral/kong2022/{sm}/cov_0p80/g1g2g3",
            "wta_subpath": grp_base / f"wta-top/kong2022/{sm}/cov_0p80",
            "present_dir": pres_base,
            "overview_grad": pres_base
            / "kong2022_gradient-spectral_group_overview_g1g2g3_routea_fisherz.png",
            "overview_grad_normalized_top20_count": pres_base
            / "kong2022_gradient-spectral_group_overview_g1g2g3_routea_normalized_top20_count.png",
            "overview_wta": pres_base
            / "kong2022_wta-top_group_overview_routea_normalized_top20_count.png",
            "hippunfold_root": REPO_ROOT
            / "outputs_iowa_jj_tms/inputs/iowa_jj_tms/hippunfold/dense_corobl_batch",
        }

    # Stanford cohorts
    stan_map = {
        "stan_nthc": "nthc",
        "stan_tehc": "tehc",
        "stan_nts": "nts",
        "stan_nts_mdd": "nts_any_mdd",
        "stan_nts_pure": "nts_pure_mdd",
        "stan_tis": "tis",
        "stan_tis_ptsd": "tis_ptsd",
    }
    if cohort_id in stan_map:
        stem = stan_map[cohort_id]
        stan_var_suffix = "_22p" if var == "22p" else ""
        branch_22p = f"routea_{stem}_22p_full_k3_fisherz"
        branch_36p = f"routea_{stem}_full_k3_fisherz"

        # Resolve primary group root
        target_branch = branch_22p if var == "22p" else branch_36p
        grp_base = (
            REPO_ROOT
            / f"outputs_stanford/work/archive/8net_20260909/group_atlas/{target_branch}"
        )

        pres_base = (
            REPO_ROOT
            / f"outputs_stanford/present/overviews/routea_{stem}{stan_var_suffix}"
        )

        overview_grad = (
            pres_base
            / "kong2022_gradient-spectral_group_overview_g1g2g3_routea_fisherz.png"
        )
        overview_wta = (
            pres_base
            / "kong2022_wta-top_group_overview_routea_normalized_top20_count.png"
        )

        return {
            "group_root": grp_base,
            "group_root_grad": grp_base,
            "group_root_wta": grp_base,
            "grad_subpath": grp_base
            / f"gradient-spectral/kong2022/{sm}/cov_0p80/g1g2g3",
            "wta_subpath": grp_base / f"wta-top/kong2022/{sm}/cov_0p80",
            "present_dir": pres_base,
            "overview_grad": overview_grad,
            "overview_grad_normalized_top20_count": pres_base
            / "kong2022_gradient-spectral_group_overview_g1g2g3_routea_normalized_top20_count.png",
            "overview_wta": overview_wta,
            "hippunfold_root": REPO_ROOT
            / "outputs_stanford/inputs/stanford/hippunfold/dense_corobl_batch",
        }

    raise ValueError(f"Unknown cohort id: {cohort_id!r}")


def get_multidataset_cohorts(
    spec: GridSpec | str = "2mm_36p",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return configured (col1_cohorts, col2_cohorts) for a given GridSpec."""
    if isinstance(spec, str):
        spec = GridSpec.from_tag(spec)

    col1: list[dict[str, Any]] = []
    for cspec in COHORT_SPECS_COL1:
        entry = dict(cspec)
        entry.update(resolve_cohort_paths(cspec["id"], spec))
        entry["grid_spec"] = spec
        entry["legacy_7net_namespace"] = spec.uses_legacy_7net_namespace
        if spec.signal_branch == "ctxreg":
            entry["signal_branch"] = "ctxreg"
        if spec.variant == "22p":
            if entry["id"] == "stan_nthc":
                entry["n_subjects"] = 50
            elif entry["id"] == "stan_tehc":
                entry["n_subjects"] = 39
        col1.append(entry)

    col2: list[dict[str, Any]] = []
    for cspec in COHORT_SPECS_COL2:
        entry = dict(cspec)
        entry.update(resolve_cohort_paths(cspec["id"], spec))
        entry["grid_spec"] = spec
        entry["legacy_7net_namespace"] = spec.uses_legacy_7net_namespace
        if spec.signal_branch == "ctxreg":
            entry["signal_branch"] = "ctxreg"
        if spec.variant == "22p":
            if entry["id"] == "stan_nthc":
                entry["n_subjects"] = 50
            elif entry["id"] == "stan_tehc":
                entry["n_subjects"] = 39
        col2.append(entry)

    return col1, col2


def get_all_cohorts(spec: GridSpec | str = "2mm_36p") -> list[dict[str, Any]]:
    """Return all 10 cohorts in flat list."""
    col1, col2 = get_multidataset_cohorts(spec)
    return col1 + col2


def check_grid_readiness(spec: GridSpec | str) -> dict[str, Any]:
    """Audit existence of required upstream atlas files for all cohorts under a GridSpec."""
    if isinstance(spec, str):
        spec = GridSpec.from_tag(spec)

    cohorts = get_all_cohorts(spec)
    report: dict[str, Any] = {
        "grid_spec": spec.tag,
        "smoothing": spec.smoothing,
        "variant": spec.variant,
        "total_cohorts": len(cohorts),
        "ready_count": 0,
        "cohorts": {},
    }

    ready_count = 0
    expected_network_order = [
        "Default",
        "Visual",
        "Somatomotor",
        "DorsalAttention",
        "VentralAttention",
        "Control",
        "Language",
    ]
    for c in cohorts:
        cid = c["id"]
        grad_sub = c["grad_subpath"]
        wta_sub = c["wta_subpath"]

        missing: list[str] = []
        for hemi in ("hemi_L", "hemi_R"):
            h_grad = grad_sub / hemi
            manifest_path = h_grad / "group_atlas_manifest.json"
            if not manifest_path.is_file():
                missing.append(f"grad_{hemi}_manifest")
            else:
                try:
                    validate_group_atlas_lineage(manifest_path)
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, ValueError, KeyError) as exc:
                    missing.append(f"grad_{hemi}_manifest_invalid:{exc}")
                else:
                    if manifest.get("network_taxonomy") != "canonical7":
                        missing.append(f"grad_{hemi}_canonical7_taxonomy")
                    if manifest.get("network_order") != expected_network_order:
                        missing.append(f"grad_{hemi}_canonical7_network_order")
            if not (h_grad / "group_labels_final.npy").is_file():
                missing.append(f"grad_{hemi}_labels")

        is_ready = len(missing) == 0
        if is_ready:
            ready_count += 1

        report["cohorts"][cid] = {
            "name": c["cohort_name"],
            "dataset": c["dataset_name"],
            "ready": is_ready,
            "missing": missing,
            "group_root": str(c["group_root"]),
            "grad_subpath": str(grad_sub),
        }

    report["ready_count"] = ready_count
    report["all_ready"] = ready_count == len(cohorts)
    return report


# ---------------------------------------------------------------------------
# Additive HCP/JJ_TMS method-comparison branch
# ---------------------------------------------------------------------------

COMPARISON_COHORT_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": "hcp_7t",
        "dataset_key": "hcp",
        "dataset_name": "HCP 7T",
        "cohort_name": "HCP 7T",
        "cohort_desc": "Healthy Young Adults",
        "n_subjects": 151,
        "bg_color": "#f4f8fc",
        "badge_bg": "#0b3c5d",
        "badge_fg": "#ffffff",
    },
    {
        "id": "iowa_hc",
        "dataset_key": "iowa_jj_tms",
        "dataset_name": "JJ_TMS 3T",
        "cohort_name": "HC",
        "cohort_desc": "Healthy Controls",
        "n_subjects": 67,
        "bg_color": "#f7faff",
        "badge_bg": "#1d5b79",
        "badge_fg": "#ffffff",
    },
    {
        "id": "iowa_pat",
        "dataset_key": "iowa_jj_tms",
        "dataset_name": "JJ_TMS 3T",
        "cohort_name": "Patient",
        "cohort_desc": "Depression Patients",
        "n_subjects": 26,
        "bg_color": "#f7faff",
        "badge_bg": "#1d5b79",
        "badge_fg": "#ffffff",
    },
    {
        "id": "iowa_pure_mdd",
        "dataset_key": "iowa_jj_tms",
        "dataset_name": "JJ_TMS 3T",
        "cohort_name": "Pure MDD",
        "cohort_desc": "Pure MDD Subgroup",
        "n_subjects": 7,
        "bg_color": "#f7faff",
        "badge_bg": "#1d5b79",
        "badge_fg": "#ffffff",
    },
    {
        "id": "iowa_comorbid_mdd",
        "dataset_key": "iowa_jj_tms",
        "dataset_name": "JJ_TMS 3T",
        "cohort_name": "Comorbid MDD",
        "cohort_desc": "Comorbid MDD Subgroup",
        "n_subjects": 19,
        "bg_color": "#f7faff",
        "badge_bg": "#1d5b79",
        "badge_fg": "#ffffff",
    },
)

_COMPARISON_METHOD_LABELS: dict[str, str] = {
    "kong2022_7net": "Kong2022 7-Network",
    "lynch2024_7net": "Lynch2024 7-Network",
}

COMPARISON_DATASET_STYLES: dict[str, tuple[str, str]] = {
    "hcp": ("HCP 7T", "#fafafa"),
    "iowa_jj_tms": ("Iowa 3T", "#f2f7fc"),
}


def comparison_dataset_style(entry: dict[str, Any]) -> tuple[str, str]:
    """Return the dataset label and background used by all comparison renderers."""
    try:
        return COMPARISON_DATASET_STYLES[str(entry["dataset_key"])]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported comparison dataset: {entry.get('dataset_key')!r}"
        ) from exc


def format_comparison_column_title(method_name: str | None, detail: str) -> str:
    """Put a method/atlas name above a comparison column while preserving legacy text."""
    if not method_name:
        return detail
    if not detail:
        return method_name
    return f"{method_name}\n{detail}"


_COMPARISON_IOWA_STEMS: dict[str, str] = {
    "iowa_hc": "hc",
    "iowa_pat": "patient",
    "iowa_pure_mdd": "pure_mdd",
    "iowa_comorbid_mdd": "comorbid_mdd",
}


def _coerce_comparison_spec(spec: GridSpec | str) -> GridSpec:
    if isinstance(spec, str):
        return GridSpec.from_tag(spec)
    if not isinstance(spec, GridSpec):
        raise TypeError(f"spec must be GridSpec or grid tag, got {type(spec)!r}")
    return spec


def get_comparison_output_tag(spec: GridSpec | str) -> str:
    """Return an output tag that cannot collide with any legacy atlas output."""
    resolved = _coerce_comparison_spec(spec)
    return f"{COMPARISON_VERSION}_{resolved.tag}"


def _resolve_comparison_ctxreg_paths(
    cohort_id: str, method: str, spec: GridSpec
) -> dict[str, Any]:
    """Resolve method-specific comparison products in the ctxreg branch."""
    if method not in COMPARISON_METHODS:
        raise ValueError(
            f"Unknown comparison method {method!r}; expected one of "
            f"{COMPARISON_METHODS}"
        )
    if cohort_id not in {row["id"] for row in COMPARISON_COHORT_SPECS}:
        raise ValueError(f"Unknown comparison cohort id: {cohort_id!r}")

    base = _resolve_ctxreg_cohort_paths(cohort_id, spec)
    comparison_dataset_keys = {
        str(row["id"]): str(row["dataset_key"]) for row in COMPARISON_COHORT_SPECS
    }
    dataset_key = comparison_dataset_keys.get(cohort_id)
    if dataset_key is None:
        raise ValueError(f"Unknown comparison cohort id: {cohort_id!r}")
    dataset_root = REPO_ROOT / (
        "outputs_hcp" if dataset_key == "hcp" else "outputs_iowa_jj_tms"
    )
    if method == "kong2022_7net":
        atlas_folder = "kong2022"
        reference_root = base["reference_root"]
        overview_grad_name = (
            "kong2022_gradient-spectral_group_overview_g1g2g3_routea_ctxreg_fisherz.png"
        )
        overview_grad_normalized_name = (
            "kong2022_gradient-spectral_group_overview_g1g2g3_"
            "routea_ctxreg_normalized_top20_count.png"
        )
        overview_wta_name = (
            "kong2022_wta-top_group_overview_routea_ctxreg_normalized_top20_count.png"
        )
    else:
        atlas_folder = "lynch2024_7net"
        if dataset_key == "hcp":
            reference_root = (
                dataset_root
                / "inputs"
                / "hcp"
                / "lynch2024_fast_pfm_t1only"
                / "shared_inputs"
                / "pfm_hcp151"
            )
        else:
            reference_root = (
                dataset_root
                / "inputs"
                / "iowa_jj_tms"
                / "lynch2024_fast_pfm_t1only"
                / "shared_inputs"
                / "pfm_nsi043_78"
            )
        # HCP and Iowa Lynch overviews carry different gradient-component tags
        # upstream: HCP embeds g1g2g3, Iowa does not. Match the manifest-declared
        # artifact exactly; never substitute across datasets.
        overview_grad_name = (
            "lynch2024_7net_gradient-spectral_group_overview_g1g2g3_"
            "routea_ctxreg_fisherz.png"
            if dataset_key == "hcp"
            else "lynch2024_7net_gradient-spectral_group_overview_"
            "routea_ctxreg_fisherz.png"
        )
        overview_grad_normalized_name = None
        overview_wta_name = (
            "lynch2024_7net_wta-top_group_overview_routea_ctxreg_fisherz.png"
        )

    grad_subpath = (
        base["group_root"]
        / "gradient-spectral"
        / atlas_folder
        / spec.smoothing
        / "cov_0p80"
        / "g1g2g3"
    )
    wta_subpath = (
        base["group_root"] / "wta-top" / atlas_folder / spec.smoothing / "cov_0p80"
    )
    result = dict(
        base,
        method=method,
        method_name=_COMPARISON_METHOD_LABELS[method],
        atlas="7net",
        atlas_folder=atlas_folder,
        wta_atlas_folder=atlas_folder,
        assignment_variant=(
            "normalized_top20_count" if method == "kong2022_7net" else "fisherz"
        ),
        subregion_metric=(
            "normalized_top20_count" if method == "kong2022_7net" else "fisherz_profile"
        ),
        variant=spec.variant,
        source_variant="36p" if dataset_key == "hcp" else spec.variant,
        grad_subpath=grad_subpath,
        wta_subpath=wta_subpath,
        reference_root=reference_root,
        signal_branch="ctxreg",
        overview_grad=base["present_dir"] / overview_grad_name,
        overview_wta=base["present_dir"] / overview_wta_name,
    )
    if overview_grad_normalized_name is not None:
        result["overview_grad_normalized_top20_count"] = (
            base["present_dir"] / overview_grad_normalized_name
        )
    return result


def resolve_comparison_paths(
    cohort_id: str, method: str, spec: GridSpec | str
) -> dict[str, Any]:
    """Resolve the canonical Kong2022/Lynch2024 7-network comparison products.

    The comparison renderer has two fixed major columns: Kong2022 first and
    Lynch2024 second. Each method keeps its own group-atlas and presentation
    namespace; no method may inherit the other method's paths.
    """
    resolved = _coerce_comparison_spec(spec)
    if method not in COMPARISON_METHODS:
        raise ValueError(
            f"Unknown comparison method {method!r}; expected one of {COMPARISON_METHODS}"
        )
    if resolved.signal_branch == "ctxreg":
        return _resolve_comparison_ctxreg_paths(cohort_id, method, resolved)

    comparison_cohorts_by_id = {str(row["id"]): row for row in COMPARISON_COHORT_SPECS}
    cohort = comparison_cohorts_by_id.get(cohort_id)
    if cohort is None:
        raise ValueError(f"Unknown comparison cohort id: {cohort_id!r}")

    sm = resolved.smoothing
    var = resolved.variant
    var_suffix = (
        "_22p" if var == "22p" and cohort["dataset_key"] == "iowa_jj_tms" else ""
    )
    dataset_root = REPO_ROOT / (
        "outputs_hcp" if cohort["dataset_key"] == "hcp" else "outputs_iowa_jj_tms"
    )

    if cohort["dataset_key"] == "hcp":
        if method == "kong2022_7net":
            group_root = dataset_root / "work/group_atlas/kong2022_7net_full_k3_fisherz"
            present_dir = dataset_root / "present/overviews/kong2022_7net"
            shared_root = dataset_root / "inputs/hcp/kong2022/shared_inputs"
            # The producer keeps cortex FC/reference exports in the explicit
            # routea branch; the parent shared_inputs directory is not an
            # input root and must never be used for subject resolution.
            reference_root = shared_root / "routea"
            overview_grad_name = "kong2022_7net_gradient-spectral_group_overview_g1g2g3_kong2022_7net_fisherz.png"
            overview_grad_normalized_name = "kong2022_7net_gradient-spectral_group_overview_g1g2g3_kong2022_7net_normalized_top20_count.png"
            overview_wta_name = "kong2022_7net_wta-top_group_overview_kong2022_7net_normalized_top20_count.png"
        else:
            group_root = (
                dataset_root / "work/group_atlas/lynch2024_7net_hcp151_full_k3_fisherz"
            )
            present_dir = dataset_root / "present/overviews/lynch2024_7net"
            shared_root = (
                dataset_root
                / "inputs/hcp/lynch2024_fast_pfm_t1only/shared_inputs/pfm_hcp151"
            )
            overview_grad_name = "lynch2024_7net_gradient-spectral_group_overview_g1g2g3_lynch2024_7net_fisherz.png"
            overview_wta_name = (
                "lynch2024_7net_wta-top_group_overview_lynch2024_7net_fisherz.png"
            )
            reference_root = shared_root
        surface_root = dataset_root / "inputs/hcp/kong2022/shared_inputs/surface"
        hippunfold_root = dataset_root / "inputs/hcp/hippunfold/dense_corobl_batch"
    else:
        stem = _COMPARISON_IOWA_STEMS[cohort_id]
        present_variant = "22p" if var == "22p" else "36p"
        surface_root = (
            dataset_root
            / f"inputs/iowa_jj_tms/kong2022/shared_inputs{var_suffix}/surface"
        )
        if method == "kong2022_7net":
            # Patient uses the explicitly rebuilt routea_patient root; the
            # other Iowa cohorts retain their kong2022_7net group roots.
            branch = (
                f"routea_patient{var_suffix}_full_k3_fisherz"
                if cohort_id == "iowa_pat"
                else f"kong2022_7net_{stem}{var_suffix}_full_k3_fisherz"
            )
            group_root = dataset_root / f"work/group_atlas/{branch}"
            present_dir = (
                dataset_root
                / f"present/overviews/kong2022_7net/{present_variant}/{stem}"
            )
            shared_root = (
                dataset_root / f"inputs/iowa_jj_tms/kong2022/shared_inputs{var_suffix}"
            )
            reference_root = shared_root / "routea"
            overview_grad_name = "kong2022_7net_gradient-spectral_group_overview_g1g2g3_kong2022_7net_fisherz.png"
            overview_grad_normalized_name = "kong2022_7net_gradient-spectral_group_overview_g1g2g3_kong2022_7net_normalized_top20_count.png"
            overview_wta_name = "kong2022_7net_wta-top_group_overview_kong2022_7net_normalized_top20_count.png"
        else:
            branch = f"lynch2024_7net_iowa3t_{stem}{var_suffix}_full_k3_fisherz_wta"
            group_root = dataset_root / f"work/group_atlas/{branch}"
            present_dir = (
                dataset_root
                / f"present/overviews/lynch2024_7net/{present_variant}/{stem}"
            )
            shared_root = (
                dataset_root
                / f"inputs/iowa_jj_tms/lynch2024_fast_pfm_t1only/shared_inputs{var_suffix}/pfm_nsi043_78"
            )
            reference_root = shared_root
            overview_grad_name = "lynch2024_7net_gradient-spectral_group_overview_g1g2g3_lynch2024_7net_fisherz.png"
            overview_wta_name = (
                "lynch2024_7net_wta-top_group_overview_lynch2024_7net_fisherz.png"
            )
        hippunfold_root = (
            dataset_root / "inputs/iowa_jj_tms/hippunfold/dense_corobl_batch"
        )

    # ``kong2022_7net`` is the comparison-branch namespace, while the
    # producer-published gradient artifacts remain under the canonical
    # ``gradient-spectral/kong2022`` folder.
    gradient_atlas_folder = "kong2022" if method == "kong2022_7net" else method
    wta_atlas_folder = "kong2022" if method == "kong2022_7net" else method
    grad_subpath = (
        group_root / f"gradient-spectral/{gradient_atlas_folder}/{sm}/cov_0p80/g1g2g3"
    )
    wta_subpath = group_root / f"wta-top/{wta_atlas_folder}/{sm}/cov_0p80"
    return {
        "method": method,
        "method_name": _COMPARISON_METHOD_LABELS[method],
        "atlas": "7net",
        "atlas_folder": gradient_atlas_folder,
        "wta_atlas_folder": wta_atlas_folder,
        "assignment_variant": (
            "normalized_top20_count" if method == "kong2022_7net" else "fisherz"
        ),
        "subregion_metric": (
            "normalized_top20_count" if method == "kong2022_7net" else "fisherz_profile"
        ),
        "variant": var,
        "source_variant": "36p" if cohort["dataset_key"] == "hcp" else var,
        "group_root": group_root,
        "group_root_grad": group_root,
        "group_root_wta": group_root,
        "grad_subpath": grad_subpath,
        "wta_subpath": wta_subpath,
        "present_dir": present_dir,
        "overview_grad": present_dir / overview_grad_name,
        **(
            {
                "overview_grad_normalized_top20_count": present_dir
                / overview_grad_normalized_name
            }
            if method == "kong2022_7net"
            else {}
        ),
        "overview_wta": present_dir / overview_wta_name,
        "shared_root": shared_root,
        "reference_root": reference_root,
        "surface_root": surface_root,
        "mask_root": surface_root,
        "hippunfold_root": hippunfold_root,
    }


def get_comparison_cohorts(
    spec: GridSpec | str = "2mm_36p",
) -> dict[str, list[dict[str, Any]]]:
    """Return five HCP/JJ_TMS cohorts for each canonical comparison method."""
    resolved = _coerce_comparison_spec(spec)
    out: dict[str, list[dict[str, Any]]] = {}
    for method_col, method in enumerate(COMPARISON_METHODS, start=1):
        entries: list[dict[str, Any]] = []
        for base in COMPARISON_COHORT_SPECS:
            entry = dict(base)
            entry.pop("n_subjects_by_method", None)
            n_by_method = base.get("n_subjects_by_method", {})
            entry["n_subjects"] = int(n_by_method.get(method, base["n_subjects"]))
            entry["col"] = method_col
            entry["method"] = method
            entry.update(resolve_comparison_paths(base["id"], method, resolved))
            entry["signal_branch"] = resolved.signal_branch
            entry["grid_spec"] = resolved
            entry["legacy_7net_namespace"] = True
            entries.append(entry)
        out[method] = entries
    return out


def get_comparison_visual_columns(
    spec: GridSpec | str = "2mm_36p",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return the fixed Kong2022 and Lynch2024 visual columns.

    The renderer-facing dataset badge is the original dataset grouping
    (HCP 7T or Iowa 3T).
    """
    methods = get_comparison_cohorts(spec)
    columns: list[list[dict[str, Any]]] = []
    for method in COMPARISON_METHODS:
        styled_entries: list[dict[str, Any]] = []
        for entry in methods[method]:
            dataset_name, bg_color = comparison_dataset_style(entry)
            styled_entries.append(
                dict(
                    entry,
                    dataset_name=dataset_name,
                    bg_color=bg_color,
                    comparison_dataset_name=dataset_name,
                    comparison_bg_color=bg_color,
                )
            )
        columns.append(styled_entries)
    return columns[0], columns[1]


def get_comparison_all_cohorts(
    spec: GridSpec | str = "2mm_36p",
) -> list[dict[str, Any]]:
    """Return all ten canonical method/cohort entries."""
    methods = get_comparison_cohorts(spec)
    return [entry for method in COMPARISON_METHODS for entry in methods[method]]


def has_explicit_legacy_hcp_ctxreg_provenance(
    path: Path,
    manifest: dict[str, Any],
    *,
    method: str | None,
) -> bool:
    """Recognize the pre-``signal_branch`` HCP Kong CtxReg contract explicitly."""
    if method != "kong2022_7net" or manifest.get("dataset") != "outputs_hcp":
        return False
    if "routea_ctxreg" not in path.parts or manifest.get("atlas") != "kong2022":
        return False

    def _is_ctxreg_path(value: object) -> bool:
        return (
            isinstance(value, str) and "routea_ctxreg" in Path(value).expanduser().parts
        )

    for key in ("shared_root", "profile_surface_root"):
        if not _is_ctxreg_path(manifest.get(key)):
            return False
    reference_root = manifest.get("reference_root")
    if (
        not isinstance(reference_root, str)
        or "routea_ctxreg" in Path(reference_root).expanduser().parts
    ):
        return False
    consumed = manifest.get("actual_consumed_input_paths")
    if not isinstance(consumed, dict) or not consumed:
        return False
    if not all(_is_ctxreg_path(value) for value in consumed.values()):
        return False
    for key in ("signal_root", "profile_source_root"):
        value = manifest.get(key)
        if value is not None and not _is_ctxreg_path(value):
            return False
    signal_control_route = manifest.get("signal_control_route")
    return signal_control_route in (None, "routea_ctxreg")


_EPHEMERAL_PATH_COMPONENT_RE = re.compile(
    r"^(?:\.?tmp(?:[_\-.].*)?|temporary(?:[_\-.].*)?)$",
    re.IGNORECASE,
)


def _is_ephemeral_provenance_path(value: object) -> bool:
    """Identify temporary path components that cannot be release inputs."""
    if not isinstance(value, str) or not value.strip():
        return False
    path = Path(value).expanduser()
    return any(_EPHEMERAL_PATH_COMPONENT_RE.match(part) for part in path.parts)


def _comparison_overview_input_errors(
    entry: dict[str, Any], *, method: str
) -> list[str]:
    """Audit the exact overview assets consumed by a comparison renderer."""
    keys = ["overview_grad", "overview_wta"]
    if method == "kong2022_7net":
        keys.insert(1, "overview_grad_normalized_top20_count")
    errors: list[str] = []
    for key in keys:
        value = entry.get(key)
        if value is None or not str(value).strip():
            errors.append(f"{key}:missing")
            continue
        path = Path(value).expanduser()
        if _is_ephemeral_provenance_path(str(path)):
            errors.append(f"{key}_temporary:{path}")
        elif not path.is_file():
            errors.append(f"{key}:{path}")
    return errors


_COMPARISON_IOWA_ROSTER_MANIFESTS: dict[str, str] = {
    "iowa_hc": "manifests/jj_tms_hc_67_subjects.txt",
    "iowa_pat": "manifests/jj_tms_patient_26_subjects.txt",
    "iowa_pure_mdd": "manifests/jj_tms_pure_mdd_7_subjects.txt",
    "iowa_comorbid_mdd": "manifests/jj_tms_comorbid_mdd_19_subjects.txt",
}
_PERSISTENT_CONCAT_PATH_COMPONENT_RE = re.compile(r"^xcpd_surf_concat$", re.IGNORECASE)


def _is_persistent_concat_provenance_path(value: object) -> bool:
    """Identify the prohibited pre-concatenated Iowa PFM input tree."""
    if not isinstance(value, str) or not value.strip():
        return False
    path = Path(value).expanduser()
    return any(_PERSISTENT_CONCAT_PATH_COMPONENT_RE.match(part) for part in path.parts)


def _comparison_lynch_input_errors(cohort_id: str) -> list[str]:
    """Audit roster-scoped Iowa Lynch PFM provenance before comparison rendering."""
    roster_rel = _COMPARISON_IOWA_ROSTER_MANIFESTS.get(cohort_id)
    if roster_rel is None:
        return []
    roster_path = REPO_ROOT / roster_rel
    if not roster_path.is_file():
        return [f"missing_iowa_roster_manifest:{roster_path}"]
    subjects = [
        line.strip().removeprefix("sub-")
        for line in roster_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    pfm_root = REPO_ROOT / "outputs_iowa_jj_tms/work/cortex_pfm/lynch2024_iowa3t"
    required_inputs = (
        "dtseries",
        "left_surface",
        "right_surface",
        "priors",
        "labels",
        "cortex_template",
    )
    required_outputs = ("component_timeseries", "component_tsv")
    errors: list[str] = []
    missing_subjects: dict[str, list[str]] = {}
    invalid_subjects: list[str] = []
    invalid_statuses: dict[str, list[str]] = {}
    temporary_paths: dict[str, list[tuple[str, str]]] = {}
    persistent_concat_paths: dict[str, list[tuple[str, str]]] = {}
    invalid_dtseries_modes: dict[str, list[str]] = {}
    absent_paths: dict[str, list[tuple[str, str]]] = {}

    def record_path(
        subject: str,
        key: str,
        value: object,
        *,
        persistent_concat: bool = False,
    ) -> None:
        if not isinstance(value, str) or not value.strip():
            missing_subjects.setdefault(key, []).append(subject)
        elif _is_ephemeral_provenance_path(value):
            temporary_paths.setdefault(key, []).append((subject, value))
        elif persistent_concat and _is_persistent_concat_provenance_path(value):
            persistent_concat_paths.setdefault(key, []).append((subject, value))
        elif not Path(value).expanduser().is_file():
            absent_paths.setdefault(key, []).append((subject, value))

    for subject in subjects:
        manifest_path = pfm_root / f"sub-{subject}/lynch2024/fastans_pfm_manifest.json"
        if not manifest_path.is_file():
            missing_subjects.setdefault("manifest", []).append(subject)
            continue
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            invalid_subjects.append(subject)
            continue

        status = payload.get("status")
        if status != "done":
            invalid_statuses.setdefault(str(status), []).append(subject)
        inputs = payload.get("inputs")
        if not isinstance(inputs, dict):
            missing_subjects.setdefault("inputs", []).append(subject)
            continue

        declared_runs = inputs.get("dtseries_inputs")
        if isinstance(declared_runs, list):
            if not declared_runs:
                missing_subjects.setdefault("dtseries", []).append(subject)
            mode = inputs.get("dtseries_input_mode")
            if mode != "in_memory_run_concat":
                invalid_dtseries_modes.setdefault(str(mode), []).append(subject)
            for value in declared_runs:
                record_path(subject, "dtseries", value, persistent_concat=True)
        else:
            record_path(
                subject, "dtseries", inputs.get("dtseries"), persistent_concat=True
            )

        for key in required_inputs:
            if key != "dtseries":
                record_path(subject, key, inputs.get(key))

        outputs = payload.get("outputs")
        if not isinstance(outputs, dict):
            missing_subjects.setdefault("outputs", []).append(subject)
        else:
            for key in required_outputs:
                record_path(subject, f"output_{key}", outputs.get(key))

    for key, subjects_for_key in sorted(missing_subjects.items()):
        errors.append(f"pfm_missing_{key}_subjects:{','.join(subjects_for_key)}")
    if invalid_subjects:
        errors.append(f"pfm_invalid_manifest_subjects:{','.join(invalid_subjects)}")
    for status, subjects_for_status in sorted(invalid_statuses.items()):
        errors.append(f"pfm_status_{status}_subjects:{','.join(subjects_for_status)}")
    for mode, subjects_for_mode in sorted(invalid_dtseries_modes.items()):
        errors.append(
            f"pfm_dtseries_input_mode_{mode}_subjects:{','.join(subjects_for_mode)}"
        )
    for key, rows in sorted(temporary_paths.items()):
        subjects_for_key = ",".join(subject for subject, _ in rows)
        examples = "; ".join(f"sub-{subject}={value}" for subject, value in rows[:2])
        errors.append(
            f"pfm_temporary_{key}_subjects:{subjects_for_key} (examples: {examples})"
        )
    for key, rows in sorted(persistent_concat_paths.items()):
        subjects_for_key = ",".join(subject for subject, _ in rows)
        examples = "; ".join(f"sub-{subject}={value}" for subject, value in rows[:2])
        errors.append(
            f"pfm_persistent_concat_{key}_subjects:{subjects_for_key} "
            f"(examples: {examples})"
        )
    for key, rows in sorted(absent_paths.items()):
        subjects_for_key = ",".join(subject for subject, _ in rows)
        examples = "; ".join(f"sub-{subject}={value}" for subject, value in rows[:2])
        errors.append(
            f"pfm_missing_file_{key}_subjects:{subjects_for_key} (examples: {examples})"
        )
    return errors


def _comparison_manifest_ready(
    path: Path,
    *,
    require_wta: bool = False,
    method: str | None = None,
    signal_branch: str | None = None,
) -> list[str]:
    missing: list[str] = []
    for hemi in ("hemi_L", "hemi_R"):
        hemi_dir = path / hemi
        manifest_path = hemi_dir / "group_atlas_manifest.json"
        if not manifest_path.is_file():
            missing.append(f"{hemi}_manifest")
        else:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                # Renderer-identical rule: pre-signal_branch HCP Kong CtxReg
                # products are accepted only under the explicit named legacy
                # provenance contract, never by loose fallback.
                if (
                    signal_branch == "ctxreg"
                    and manifest.get("signal_branch") != "ctxreg"
                    and not (
                        manifest.get("signal_branch") is None
                        and has_explicit_legacy_hcp_ctxreg_provenance(
                            path, manifest, method=method
                        )
                    )
                ):
                    raise ValueError(
                        "ctxreg comparison requires manifest signal_branch='ctxreg', "
                        f"got {manifest.get('signal_branch')!r}"
                    )
                if signal_branch == "standard" and manifest.get(
                    "signal_branch"
                ) not in (None, "standard"):
                    raise ValueError(
                        "standard comparison cannot consume a non-standard manifest: "
                        f"signal_branch={manifest.get('signal_branch')!r}"
                    )
                if require_wta:
                    if method == "kong2022_7net":
                        expected = {
                            "default_assignment_variant": "normalized_top20_count",
                            "confidence_metric": "normalized_top20_count",
                            "assignment_metric": "top20_count_normalized_by_network_roi_count",
                        }
                    elif method == "lynch2024_7net":
                        expected = {
                            "default_assignment_variant": "fisherz",
                            "confidence_metric": "fisherz_margin",
                            "assignment_metric": "top1_fisher_z",
                        }
                    else:
                        raise ValueError(
                            f"unsupported WTA comparison method: {method!r}"
                        )
                    mismatches = {
                        key: (manifest.get(key), value)
                        for key, value in expected.items()
                        if manifest.get(key) != value
                    }
                    if mismatches:
                        raise ValueError(
                            f"manifest assignment contract mismatch: {mismatches}"
                        )
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                missing.append(
                    f"{hemi}_{'canonical_assignment' if require_wta else 'manifest_contract'}:{exc}"
                )
            if require_wta:
                if not (hemi_dir / "group_functional_atlas_labels.npy").is_file():
                    missing.append(f"{hemi}_functional_labels")
                if method == "lynch2024_7net":
                    if not (hemi_dir / "vertex_network_profile.npy").is_file():
                        missing.append(f"{hemi}_vertex_network_profile")
                    if not (hemi_dir / "vertex_assignment_status.json").is_file():
                        missing.append(f"{hemi}_vertex_assignment_status")
            elif method == "kong2022_7net":
                try:
                    validate_group_atlas_lineage(manifest_path)
                except (OSError, ValueError, KeyError) as exc:
                    missing.append(f"{hemi}_lineage:{exc}")
                if not (
                    hemi_dir / "group_functional_atlas_labels_fisherz.npy"
                ).is_file():
                    missing.append(f"{hemi}_fisherz_labels")
        if not require_wta and not (hemi_dir / "group_labels_final.npy").is_file():
            missing.append(f"{hemi}_cluster_labels")
    return missing


def check_comparison_readiness(spec: GridSpec | str) -> dict[str, Any]:
    """Audit both canonical HCP/JJ_TMS comparison method columns."""
    resolved = _coerce_comparison_spec(spec)
    methods = get_comparison_cohorts(resolved)
    report: dict[str, Any] = {
        "version": COMPARISON_VERSION,
        "grid_spec": resolved.tag,
        "smoothing": resolved.smoothing,
        "variant": resolved.variant,
        "total_cohorts": len(COMPARISON_COHORT_SPECS),
        "total_entries": len(COMPARISON_COHORT_SPECS) * len(COMPARISON_METHODS),
        "ready_count": 0,
        "all_ready": False,
        "methods": {},
        "cohorts": {},
    }
    for method in COMPARISON_METHODS:
        method_rows: dict[str, Any] = {}
        for entry in methods[method]:
            grad_path = Path(entry["grad_subpath"])
            grad_missing = _comparison_manifest_ready(
                grad_path,
                method=method,
                signal_branch=resolved.signal_branch,
            )
            if method == "kong2022_7net":
                for hemi in ("hemi_L", "hemi_R"):
                    hemi_dir = grad_path / hemi
                    if not (
                        hemi_dir
                        / "cluster_assignment_status_normalized_top20_count.json"
                    ).is_file():
                        grad_missing.append(f"{hemi}_normalized_top20_status")
            elif method == "lynch2024_7net":
                for hemi in ("hemi_L", "hemi_R"):
                    hemi_dir = grad_path / hemi
                    if not (hemi_dir / "cluster_network_profile.npy").is_file():
                        grad_missing.append(f"{hemi}_fisherz_profile")
                    if not (hemi_dir / "group_feature_summary.json").is_file():
                        grad_missing.append(f"{hemi}_fisherz_profile_summary")
                grad_missing.extend(_comparison_lynch_input_errors(entry["id"]))
            wta_missing = _comparison_manifest_ready(
                Path(entry["wta_subpath"]),
                require_wta=True,
                method=method,
                signal_branch=resolved.signal_branch,
            )
            overview_missing = _comparison_overview_input_errors(entry, method=method)
            missing = (
                [f"grad_{item}" for item in grad_missing]
                + [f"wta_{item}" for item in wta_missing]
                + [f"input_{item}" for item in overview_missing]
            )
            ready = not missing
            if ready:
                report["ready_count"] += 1
            row = {
                "name": entry["cohort_name"],
                "dataset": entry["dataset_name"],
                "method": method,
                "ready": ready,
                "missing": missing,
                "group_root": str(entry["group_root"]),
                "grad_subpath": str(entry["grad_subpath"]),
                "wta_subpath": str(entry["wta_subpath"]),
            }
            method_rows[entry["id"]] = row
            report["cohorts"][f"{method}:{entry['id']}"] = row
        report["methods"][method] = method_rows
    report["all_ready"] = report["ready_count"] == report["total_entries"]
    return report

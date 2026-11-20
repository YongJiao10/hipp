#!/usr/bin/env python3
# pyright: reportArgumentType=false, reportGeneralTypeIssues=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportCallIssue=false, reportOperatorIssue=false, reportReturnType=false, reportOptionalSubscript=false, reportPrivateImportUsage=false, reportOptionalIterable=false, reportOptionalMemberAccess=false, reportPossiblyUnboundVariable=false
from __future__ import annotations

"""
Process Step Description:
Run group-level atlas clustering and network mapping. Performs clustering analysis on group-averaged functional connectivity or gradients to generate hippocampal functional parcellation atlases and maps them to known cortical networks.

Main Flow Path:
- 8th step in the numbered pipeline sequence (the core clustering and mapping execution step):
  01_01_run_hippunfold_t1w_local.py -> 01_02_check_hippunfold_outputs.py -> 02_01_prepare_group_inputs.py -> 02_02_build_qc_manifest.py -> 03_01/03_02 reference building -> 04_01/04_02 FC materialization -> 05_01 gradient computation -> 05_02_run_group_atlas.py (this step) -> 06_xx presentation rendering -> 07_01 group matrix overview.

Inputs:
- Command-line arguments:
  - --shared-root: Path to shared processed files (default: `outputs_hcp/inputs/kong2022/shared_inputs/pfm`).
  - --surface-root: Path to BOLD surfaces root directory (default: `outputs_hcp/inputs/kong2022/shared_inputs/surface`).
  - --profile-surface-root: Optional root for hippocampal time series used only by cluster-network profile annotation; defaults to `--surface-root`.
  - --profile-source-mode: Profile source contract (`canonical` or the explicit `original-denoised` recovery source).
  - --profile-source-root: Required with `--profile-source-mode original-denoised`; root containing the approved fixed-template subject outputs.
  - --profile-source-registry: Required with `--profile-source-mode original-denoised`; the verified ordered 400-ROI registry TSV.
  - --out-root: Directory where group-atlas intermediate files will be saved (default: `outputs_hcp/work/group_atlas/pfm`).
  - --gradient-root: Directory containing computed continuous gradients (default: `outputs_hcp/work/hippomaps_gradients`).
  - --hippomaps-root: Root path to external HippoMaps codebase (default: `external/hippomaps`).
  - --route-tag: Expected workflow branch tag, e.g., "pfm", "pfm_ctxreg", "pfm_ctxreg_10mm", or "fixed".
  - --subjects-manifest: Path to QC-passed subject list manifest.
  - --hippunfold-batch-root: Batch output root containing sub-*/hippunfold outputs.
  - --kong2022-dlabel: Path to standard Kong2022 dlabel.nii cortical atlas file.
  - --methods: Parcellation methods to evaluate, including "gradient-spectral", "gradient-ward", "gradient-gmm", and "gradient-spectral-rbf".
  - --atlases: Cortical atlases to map (default: "kong2022"); Lynch2024 atlases support gradient-spectral and direct-network wta-top.
  - --smoothings: Smoothing levels to process: "2mm", "4mm".
  - --hemis: Hemispheres to process: "L", "R".
  - --coverage-thresholds: Minimum coverage fraction values to evaluate (default: [0.80]).
  - --k-min: Minimum number of clusters to evaluate (default: 2).
  - --k-max: Maximum number of clusters to evaluate (default: 8).
  - --default-k: Primary K value for rendered labels (default: 3); candidate K evaluation and automatic selection remain recorded diagnostics.
  - --force-final-k: Explicit primary-K override only.
  - --instability-resamples: Number of split-half resamples for cluster instability (default: 100).
  - --null-resamples: Number of random spin tests for cortical mapping null distribution (default: 100).
  - --seed: Random seed for repeatability (default: 20260420).
  - --v-min-fraction: Minimum cluster vertex size fraction (default: 0.03).
  - --max-components-threshold: Max components for gradient selection.
  - --min-profile-coherence: Minimum profile coherence score threshold.
  - --subject-fc-mode: FC preprocessing sparsification mode: "full", "top-percent".
  - --subject-fc-top-percent: Sparsification top percentage integer.
  - --gradient-components: Selected gradient components for gradient clustering (default: "1,2,3").
  - --feature-scaling and --sparsify-stage: Explicit upstream FC scaling and Pearson-affinity sparsification settings.
  - --clustering-feature-scaling: Independent clustering feature scaling ("none" or "unit_sd").
  - --rbf-scale: Multiplier for the median positive mesh-edge gradient distance.
  - --assignment-variant: Select Fisher-z, normalized Top-20, or explicit canonical7/family17 OVR annotations; "both" means Fisher-z plus normalized Top-20.
  - --resume: Resume from existing runs.
  - --k-selection-mode: "redundancy-aware" or fixed-K "fixed-comparison".
  - --redundancy-early-stop: Stop evaluation of larger K values once first redundant K is found (default: False).
  - --overview-threshold: Selected coverage threshold for overview outputs (default: 0.80).
  - --jobs: Parallel worker pool size (default: 1).
  - --group-average-surfaces-dir: Directory containing group average template surfaces.
- Physical inputs:
  - Subject functional connectivity caches: e.g. `{shared_root}/sub-<subject>/fc/.../hipp_vertex_to_network_fc_active.npy`.
  - Group average template surfaces: e.g. `{group_average_surfaces_dir}/group-average_hemi-{L|R}_space-corobl_den-512_label-hipp_midthickness.surf.gii`.
  - Pre-computed continuous gradients under `{gradient_root}/` for gradient clustering methods.

Outputs:
- Command-line output / file outputs:
  - Detailed group-atlas results directories under `{out_root}/{route_tag}/{method}/smooth-{smoothing}_hemi-{hemi}_cov-{threshold}/`:
    - `group_atlas_manifest.json`: Step-level execution trace and results.
    - Suffixed default/variant files such as `group_functional_atlas_labels_fisherz.npy`.
    - Suffixed network mapping diagnostics such as `cluster_assignment_status_fisherz.json`.
    - `group_atlas_manifest.json` records `default_assignment_variant` and points top-level default fields to the selected suffixed variant. Cluster-level defaults are `fisherz`; Kong2022 ROI-registry `wta-top` remains the separate `normalized_top20_count` assignment.
"""
import argparse
import csv
import sys
import hashlib
import importlib.metadata
import json
import math
import os
import warnings
from collections.abc import Iterable, Sequence  # noqa: E402
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from lib.native_threads import NATIVE_THREAD_ENV_VARS  # pins native threads; must precede numpy

import nibabel as nib
import numpy as np
from nibabel.gifti import GiftiImage
from scipy import sparse
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score

REPO_ROOT = Path(__file__).resolve().parents[2]
from lib.hippomaps_adapter import (
    DEFAULT_HIPPOMAPS_ROOT,
    brainspace_sparsity_for_feature_count,
    ensure_hippomaps_on_path,
    MEAN_SIMILARITY_EMBEDDING_CALL,
    SUBJECT_GRADIENT_EMBEDDING_CALL,
    fit_diffusion_gradients,
    fit_mean_similarity_gradients,
    generalized_procrustes_mean,
    gradient_preprocessing_settings,
    validate_gradient_preprocessing,
    scale_gradient_features,
    hippomaps_commit,
    hippomaps_import_path,
)
from lib.surface_utils import infer_spec_path, long_axis_order
from lib.surface_affinity import (
    GRADIENT_SPECTRAL_AFFINITY,
    GRADIENT_SPECTRAL_AFFINITY_DESCRIPTION,
    POSITIVE_COSINE_AFFINITY,
    POSITIVE_COSINE_AFFINITY_DESCRIPTION,
    affinity_graph_diagnostics,
    build_gradient_spectral_affinity,
    build_positive_cosine_surface_affinity,
)
from lib.kong2022_fc_contract import (
    KONG2022_ROI_COUNT,
    read_kong2022_roi_registry,
    validate_kong2022_roi_fc,
    validate_kong2022_roi_registry,
)
from lib.kong2022_label_utils import (
    aggregate_kong2022_network_timeseries,
    canonical_families_7_order,
    load_kong2022_bundle,
    load_kong2022_merge_mapping,
)
from lib.lynch2024_reference import lynch2024_7net_reference_contract, lynch2024_20net_order
from lib.output_layout import (
    DATA_ROOT,
    DEFAULT_GROUP_ATLAS_ROOT,
    DEFAULT_HIPPOMAPS_GRADIENT_ROOT,
    DEFAULT_HIPPUNFOLD_BATCH_ROOT as CORE_HIPPUNFOLD_BATCH_ROOT,
    DEFAULT_KONG2022_MANIFEST_ROOT,
    DEFAULT_KONG2022_ROOT,
    DEFAULT_KONG2022_ROUTEA_ROOT,
    DEFAULT_KONG2022_SURFACE_ROOT,
    DEFAULT_KONG2022_TEMPLATE_ROOT,
    DEFAULT_GRADIENT_COMPONENTS,
    DEFAULT_ROUTEA_GROUP_ATLAS_ROOT,
    gradient_comp_tag,
    parse_gradient_components,
    resolve_cohort_gradient_root,
    resolve_cohort_group_atlas_root,
)
from lib.gradient_profiles import (
    RESAMPLE_HALF_SEED_OFFSETS,
    GRADIENT_PROFILE_CHOICES,
    gradient_profile_config,
    gradient_profile_role,
    gradient_profile_earlier_label,
    get_gradient_profile,
    validate_gradient_combination,
    # Backward compatibility aliases
    FORMAL_RESAMPLE_HALF_SEED_OFFSETS,
    FORMAL_PROFILE_CHOICES,
    formal_profile_config,
    formal_profile_role,
    formal_profile_earlier_label,
    get_formal_profile,
    validate_formal_combination,
)
from lib.network_ts_fisherz import (
    assignment_status_from_weighted_network_profile,
    subjectwise_cluster_network_fisherz_profile,
)
from lib.route_tags import (
    FAMILY17_ROUTE_TAGS,
    ROUTE_TAG_CHOICES,
    resolve_group_atlas_out_root,
    infer_route_tag_from_shared_root,
    resolve_route_tag_for_group_atlas_run,
    validate_group_atlas_root_for_route,
)
from lib.fc_math import corrcoef_rows, fisher_z_transform_fc, bh_fdr, center_run_timeseries
from lib.gradient_clustering import (
    GRADIENT_METHODS, MEAN_SIMILARITY_WARD_METHOD, RBF_AFFINITY_DESCRIPTION, SUBJECT_GRADIENT_WARD_METHOD,
    WARD_GRADIENT_METHODS,
    cluster_gradients,
    binary_mesh_adjacency, build_rbf_mesh_affinity,
    spatial_cluster_diagnostics,
    is_gradient_method as registered_gradient_method,
)
from lib.gifti_labels import make_cluster_label_gifti, make_named_label_gifti
from lib.multigraph_spectral import (
    AFFINITY_DESCRIPTION as MULTIGRAPH_AFFINITY_DESCRIPTION,
    MAX_COORDINATES as MULTIGRAPH_MAX_COORDINATES,
    MEAN_GRAPH_METHOD,
    MULTIGRAPH_GRADIENT_METHOD,
    MULTIGRAPH_METHOD,
    MULTIGRAPH_METHODS,
    coordinate_stack,
    discretize_stack,
    observed_vertices,
    stack_coherence,
    vertex_similarity,
)
from lib.io_utils import load_subject_ids, load_json, write_tsv, load_run_segment_metadata
from lib.group_atlas_lineage import validate_group_atlas_lineage, gradient_computation_provenance
from lib.network_colors import load_cluster_palette, load_network_colors
from lib.gradient_alignment import align_gradient_components_to_reference
from lib.network_ovr import (
    ALL_OVR_VARIANTS,
    NETWORK_TS_OVR_VARIANT,
    OVR_ASSIGNMENT_METRIC,
    OVR_VARIANTS,
    export_network_ovr,
    fit_network_ovr,
    fit_network_ovr_from_network_profiles,
    network_ts_ovr_variant,
)
from lib.cluster_roi_profile_contract import (
    GRADIENT_PROFILE_CONTRACT_TAG,
    atlas_produces_cluster_roi_profile,
)
from lib.branch_config import (
    CTXREG_ROUTES,
    SIGNAL_BRANCH_CHOICES,
    SIGNAL_BRANCH_TO_ROUTE,
    is_ctxreg_signal_branch,
    normalize_signal_branch,
    resolve_signal_branch_for_route,
)
from lib.stale_products import remind_stale_figures
from lib.workflow_utils import code_provenance, option_was_given, utc_now_iso, write_json
from lib.template_assets import canonical_dataset_key, canonicalize_template_manifest, dataset_key_from_path


DEFAULT_T1ONLY_ROOT = DEFAULT_KONG2022_ROOT
DEFAULT_SHARED_ROOT = DEFAULT_KONG2022_ROUTEA_ROOT
DEFAULT_SURFACE_ROOT = DEFAULT_KONG2022_SURFACE_ROOT
DEFAULT_OUT_ROOT = DEFAULT_ROUTEA_GROUP_ATLAS_ROOT
DEFAULT_GRADIENT_ROOT = DEFAULT_HIPPOMAPS_GRADIENT_ROOT
DEFAULT_SUBJECTS = DEFAULT_KONG2022_MANIFEST_ROOT / "hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt"
DEFAULT_HIPPUNFOLD_BATCH_ROOT = CORE_HIPPUNFOLD_BATCH_ROOT
DEFAULT_NETWORK_STYLE = REPO_ROOT / "config" / "hipp_network_style.json"
DEFAULT_GROUP_AVG_SURFACES_DIR = DEFAULT_KONG2022_TEMPLATE_ROOT / "group_average_surfaces"
KONG2022_DLABEL_OVERRIDE: Path | None = None
HIPPOMAPS_ROOT_OVERRIDE: Path = DEFAULT_HIPPOMAPS_ROOT

METHOD_CHOICES = [
    "group-roi-sim-spectral",
    "network-spectral",
    "gradient-spectral",
    "gradient-ward",
    "gradient-gmm",
    "gradient-spectral-rbf",
    "intrinsic-spectral",
    MEAN_SIMILARITY_WARD_METHOD,
    SUBJECT_GRADIENT_WARD_METHOD,
    MULTIGRAPH_METHOD,
    MULTIGRAPH_GRADIENT_METHOD,
    MEAN_GRAPH_METHOD,
    "wta-top",
]
METHODS_DEFAULT = [
    "intrinsic-spectral",
    "network-spectral",
    "wta-top",
]
# NOTE: kong2019/kong2019_8net were retired with 04_03_materialize_kong2019_fc.py.
# Active production defaults to kong2022; do not re-add retired atlases here.
ATLASES = ["kong2022", "lynch2024_20net", "lynch2024_7net"]
DEFAULT_ATLASES = ["kong2022"]
SMOOTHINGS = ["2mm", "4mm"]
HEMIS = ["L", "R"]
K_RANGE = list(range(2, 9))
SIMILARITY_GRADIENT_METHODS = frozenset({MEAN_SIMILARITY_WARD_METHOD, SUBJECT_GRADIENT_WARD_METHOD})
KONG2022_ONLY_METHODS = {"group-roi-sim-spectral"} | SIMILARITY_GRADIENT_METHODS | MULTIGRAPH_METHODS
LYNCH2024_METHODS = (set(GRADIENT_METHODS) - SIMILARITY_GRADIENT_METHODS) | {"wta-top"}
SUBJECT_GRADIENT_SOURCE = "computed_in_group_atlas_from_procrustes_mean_of_subject_gradients"
MEAN_SIMILARITY_GRADIENT_SOURCE = "computed_in_group_atlas_from_mean_subject_vertex_similarity"
CLUSTERLESS_METHODS = {"wta-top"}
PROFILE_REDUNDANCY_MAX_SIM = 0.80
SUBJECT_FC_PREPROCESS_METHODS = {"group-roi-sim-spectral", "network-spectral", "intrinsic-spectral"}
SUBJECT_FC_VALUE_RULE = "largest_raw_values"
SUBJECT_FC_MODE_CHOICES = ["full", "top-percent"]
GRADIENT_N_COMPONENTS_COMPUTED = 5
GRADIENT_BRAINSPACE_KERNEL = "pearson"
NETWORK_TAXONOMIES = ("canonical7", "canonical8", "family17", "lynch20", "lynch7")
WTA_NETWORK_TAXONOMIES = frozenset({"canonical7", "canonical8", "family17", "lynch20", "lynch7"})
DEFAULT_CLUSTER_LEVEL_ASSIGNMENT_VARIANT = "fisherz"
MAX_K_EVAL_JOBS = 10
RESUME_SIGNATURE_VERSION = 1
MAX_DIAGNOSTIC_FAILURE_EXAMPLES = 5

def select_cluster_level_default_assignment_variant(
    assignment_variant_specs: dict[str, dict[str, object]],
) -> str:
    """Select the explicit default for cluster-level network assignments.

    ``normalized_top20_count`` remains a supported Kong2022 comparison
    variant and the dedicated Kong ROI-registry WTA path, but it must not
    silently become the default for cluster-based atlases.
    """
    if DEFAULT_CLUSTER_LEVEL_ASSIGNMENT_VARIANT not in assignment_variant_specs:
        raise ValueError(
            "Cluster-level assignment variants must include the required "
            f"default {DEFAULT_CLUSTER_LEVEL_ASSIGNMENT_VARIANT!r}; "
            f"available={sorted(assignment_variant_specs)}"
        )
    return DEFAULT_CLUSTER_LEVEL_ASSIGNMENT_VARIANT


@dataclass(frozen=True)
class Combo:
    method: str
    atlas: str
    smoothing: str
    hemi: str
    coverage_threshold: float


@dataclass(frozen=True)
class SubjectFCMode:
    mode: str
    mode_tag: str
    top_percent: int | None
    top_fraction: float
    binarize: bool


@dataclass(frozen=True)
class GroupAtlasConfig:
    args: argparse.Namespace
    shared_root: Path
    surface_root: Path
    profile_surface_root: Path
    out_root: Path
    reference_root: Path
    gradient_root: Path
    route_tag: str
    gradient_route_tag: str
    signal_branch: str | None
    network_taxonomy: str
    hippunfold_batch_root: Path
    subjects_manifest_path: Path
    subjects: list[str]
    subject_fc_mode: SubjectFCMode
    gradient_components: tuple[int, ...]
    coverage_thresholds: list[float]
    k_values: list[int]
    split_seeds: list[int]
    null_seeds: list[int]


@dataclass(frozen=True)
class PreparedComboInputs:
    n_vertices: int
    valid_masks: dict[str, np.ndarray]
    coverage_count: np.ndarray
    coverage_fraction: np.ndarray
    feature_bank: dict[str, np.ndarray]
    actual_input_paths: dict[str, str]
    run_feature_banks: dict[str, object]
    roi_fc_bank_for_redundancy: dict[str, np.ndarray] | None
    subject_fc_preprocessing_for_resume: dict[str, object]
    resume_input_category_hashes: dict[str, str]


def canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_state(path: Path) -> dict[str, object]:
    p = Path(path).resolve()
    out: dict[str, object] = {"path": str(p), "exists": bool(p.exists())}
    if bool(out["exists"]):
        st = p.stat()
        out["size_bytes"] = int(st.st_size)
        out["mtime_ns"] = int(st.st_mtime_ns)
    return out


def file_state_list(paths: list[Path]) -> list[dict[str, object]]:
    uniq: dict[str, Path] = {}
    for p in paths:
        rp = Path(p).resolve()
        uniq[str(rp)] = rp
    return [file_state(uniq[k]) for k in sorted(uniq.keys())]


def gradient_resume_inputs(
    *, gradient_root: Path, cohort: str | None, variant: str,
    gradient_options: dict[str, object] | None = None,
    route: str | None = None,
    smoothing: str | None = None,
    hemi: str | None = None,
    profile: str | None = None,
) -> dict[str, object]:
    """Record the explicit gradient bank in the existing resume contract."""
    out: dict[str, object] = {
        "gradient_root": str(gradient_root),
        "gradient_cohort": cohort,
        "gradient_variant": variant,
        "gradient_options": dict(gradient_options or {}),
        "gradient_anchor_inputs": file_state_list([
            Path(path) for field in ("alignment_fc_paths", "alignment_mask_paths")
            for path in (gradient_options or {}).get(field, {}).values()
        ] + [
            (
                Path(path).parent / "runs" / f"run-{rid}" / Path(path).name
                if (Path(path).parent / "runs" / f"run-{rid}" / Path(path).name).exists()
                else Path(path).parent / "runs" / f"run-{rid}" / (Path(path).stem + "_active.npy")
            )
             for path in (gradient_options or {}).get("alignment_fc_paths", {}).values()
             for rid in (gradient_options or {}).get("alignment_run_ids", [])]),
    }
    if route is not None and smoothing is not None and hemi is not None:
        grad_run_dir = (
            resolve_cohort_gradient_root(
                route=route, cohort=cohort, variant=variant, base_root=Path(gradient_root).resolve(),
                profile=profile,
            )
            / smoothing
            / f"hemi_{hemi}"
        )
        out["gradient_artifact_inputs"] = file_state_list([
            grad_run_dir / "hippomaps_gradient_manifest.json",
            grad_run_dir / "hippomaps_gradients_full.npy",
        ])
    return out


def _validate_gradient_resume_contract(
    manifest_path: Path,
    payload: dict[str, object],
    *,
    gradient_options: dict[str, object] | None = None,
    expected_gradient_meta: dict[str, object] | None = None,
) -> None:
    sig_payload = payload.get("resume_input_signature_payload")
    recorded_opts = (
        sig_payload.get("gradient_options", {})
        if isinstance(sig_payload, dict) and isinstance(sig_payload.get("gradient_options"), dict)
        else {}
    )
    opts = dict(recorded_opts) if gradient_options is None else dict(gradient_options)
    summary_raw = payload.get("group_gradient_feature_summary")
    source_manifest_raw = payload.get("gradient_source_manifest")
    method = str(payload.get("method", ""))
    if (
        expected_gradient_meta is None
        and gradient_options is None
        and not summary_raw
        and not source_manifest_raw
        and not is_gradient_method(method)
    ):
        return
    feature_count = int(payload.get("feature_count", 400) or 400)
    expected_preprocessing = (
        dict(expected_gradient_meta["gradient_preprocessing"])
        if isinstance(expected_gradient_meta, dict) and isinstance(expected_gradient_meta.get("gradient_preprocessing"), dict)
        else gradient_preprocessing_settings(
            feature_count,
            feature_scaling=str(opts.get("feature_scaling", "col_zscore")),
            sparsify_stage=str(opts.get("sparsify_stage", "features")),
            sparsity=opts.get("sparsity"),
        )
    )
    expected_scaling = (
        str(expected_gradient_meta["clustering_feature_transform"]["scaling"])
        if isinstance(expected_gradient_meta, dict)
        and isinstance(expected_gradient_meta.get("clustering_feature_transform"), dict)
        and "scaling" in expected_gradient_meta["clustering_feature_transform"]
        else str(opts.get("clustering_feature_scaling", "none"))
    )
    expected_anchor = (
        str(expected_gradient_meta["gradient_alignment"]["anchor_smoothing"])
        if isinstance(expected_gradient_meta, dict)
        and isinstance(expected_gradient_meta.get("gradient_alignment"), dict)
        and "anchor_smoothing" in expected_gradient_meta["gradient_alignment"]
        else str(opts.get("alignment_anchor_smoothing", "4mm"))
    )
    recorded_anchor_opt = recorded_opts.get("alignment_anchor_smoothing")
    if recorded_anchor_opt is not None and str(recorded_anchor_opt) != expected_anchor:
        raise ValueError(
            f"alignment_anchor_smoothing mismatch in {manifest_path} options: "
            f"requested={expected_anchor}, recorded={recorded_anchor_opt}; explicitly regenerate"
        )
    if not summary_raw:
        raise ValueError(
            f"Missing group_gradient_feature_summary declaration in {manifest_path}; explicitly regenerate"
        )
    summary_path = Path(str(summary_raw)).resolve()
    if not summary_path.exists():
        raise ValueError(
            f"Missing group_gradient_feature_summary in {manifest_path}: {summary_path}; explicitly regenerate"
        )
    summary_obj = json.loads(summary_path.read_text(encoding="utf-8"))
    validate_gradient_preprocessing(
        summary_obj, settings=expected_preprocessing, manifest_path=summary_path,
    )
    transform = summary_obj.get("clustering_feature_transform")
    recorded_scaling = transform.get("scaling") if isinstance(transform, dict) else None
    if recorded_scaling != expected_scaling:
        raise ValueError(
            f"clustering_feature_scaling mismatch in {summary_path}: "
            f"requested={expected_scaling}, recorded={recorded_scaling}; explicitly regenerate"
        )
    summary_source = summary_obj.get("hippomaps_gradient_manifest")
    summary_align = summary_obj.get("gradient_alignment")
    if isinstance(summary_align, dict):
        summary_anchor = summary_align.get("anchor_smoothing")
        summary_hemi = summary_align.get("anchor_hemi")
        if summary_hemi not in {None, "L"}:
            raise ValueError(
                f"Invalid anchor_hemi in {summary_path}: recorded={summary_hemi!r}; expected 'L'"
            )
    else:
        summary_anchor = summary_obj.get("alignment_anchor_smoothing")

    if expected_anchor == "2mm":
        if summary_anchor != "2mm":
            raise ValueError(
                f"alignment_anchor_smoothing mismatch in {summary_path}: "
                f"requested={expected_anchor}, recorded={summary_anchor}; explicitly regenerate"
            )
        if isinstance(summary_align, dict) and summary_align.get("enabled") is False:
            raise ValueError(
                f"gradient_alignment not enabled in {summary_path}; explicitly regenerate"
            )
    elif summary_anchor is not None and str(summary_anchor) != expected_anchor:
        raise ValueError(
            f"alignment_anchor_smoothing mismatch in {summary_path}: "
            f"requested={expected_anchor}, recorded={summary_anchor}; explicitly regenerate"
        )
    if not summary_source:
        raise ValueError(
            f"Missing gradient_source_manifest declaration in {summary_path}; explicitly regenerate"
        )
    if source_manifest_raw and Path(str(source_manifest_raw)).resolve() != Path(str(summary_source)).resolve():
        raise ValueError(
            f"gradient_source_manifest mismatch in {manifest_path}: "
            f"manifest={source_manifest_raw!r}, summary={summary_source!r}; explicitly regenerate"
        )
    source_manifest_raw = summary_source
    if source_manifest_raw:
        grad_manifest_path = Path(str(source_manifest_raw)).resolve()
        if not grad_manifest_path.exists():
            raise FileNotFoundError(
                f"Missing upstream gradient manifest for resume in {manifest_path}: "
                f"{grad_manifest_path}; explicitly regenerate"
            )
        grad_manifest = json.loads(grad_manifest_path.read_text(encoding="utf-8"))
        grad_shape = grad_manifest.get("group_fc_shape")
        if isinstance(grad_shape, list) and len(grad_shape) == 2:
            expected_preprocessing = gradient_preprocessing_settings(
                int(grad_shape[1]),
                feature_scaling=str(opts.get("feature_scaling", expected_preprocessing["feature_scaling"])),
                sparsify_stage=str(opts.get("sparsify_stage", expected_preprocessing["sparsify_stage"])),
                sparsity=opts.get("sparsity", expected_preprocessing["effective_sparsity"]),
            )
        validate_gradient_preprocessing(
            grad_manifest, settings=expected_preprocessing, manifest_path=grad_manifest_path,
        )
        grad_align = grad_manifest.get("gradient_alignment")
        if isinstance(grad_align, dict):
            recorded_grad_anchor = grad_align.get("anchor_smoothing")
            recorded_grad_hemi = grad_align.get("anchor_hemi")
            if recorded_grad_hemi not in {None, "L"}:
                raise ValueError(
                    f"Invalid anchor_hemi in {grad_manifest_path}: recorded={recorded_grad_hemi!r}; expected 'L'"
                )
        else:
            recorded_grad_anchor = None

        if expected_anchor == "2mm":
            if recorded_grad_anchor != "2mm":
                raise ValueError(
                    f"alignment_anchor_smoothing mismatch in {grad_manifest_path}: "
                    f"requested={expected_anchor}, recorded={recorded_grad_anchor}; explicitly regenerate"
                )
            if isinstance(grad_align, dict) and grad_align.get("enabled") is False:
                raise ValueError(
                    f"gradient_alignment not enabled in {grad_manifest_path}; explicitly regenerate"
                )
        elif recorded_grad_anchor is not None and str(recorded_grad_anchor) != expected_anchor:
            raise ValueError(
                f"alignment_anchor_smoothing mismatch in {grad_manifest_path}: "
                f"requested={expected_anchor}, recorded={recorded_grad_anchor}; explicitly regenerate"
            )
    if method == "gradient-spectral-rbf":
        expected_rbf = float(opts.get("rbf_scale", 1.0))
        recorded_rbf = payload.get("rbf_scale")
        if recorded_rbf is None or float(recorded_rbf) != expected_rbf:
            raise ValueError(
                f"rbf_scale mismatch in {manifest_path}: "
                f"requested={expected_rbf}, recorded={recorded_rbf}; explicitly regenerate"
            )


def resume_manifest_matches(
    manifest_path: Path,
    expected_signature: str,
    *,
    gradient_options: dict[str, object] | None = None,
    expected_gradient_meta: dict[str, object] | None = None,
) -> tuple[bool, str]:
    if not manifest_path.exists():
        return False, "missing_manifest"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        return False, f"invalid_manifest:{exc.__class__.__name__}"
    if payload.get("status") not in {"skipped_no_valid_k", "skipped_no_k_survived_constraints"}:
        _validate_gradient_resume_contract(
            manifest_path,
            payload,
            gradient_options=gradient_options,
            expected_gradient_meta=expected_gradient_meta,
        )
    if int(payload.get("resume_input_signature_version", -1)) != int(RESUME_SIGNATURE_VERSION):
        return False, "missing_or_legacy_signature"
    got = payload.get("resume_input_signature")
    if not isinstance(got, str):
        return False, "missing_signature"
    if got != expected_signature:
        return False, "upstream_signature_changed"
    return True, "signature_match"


def zscore_cols(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    mu = np.nanmean(arr, axis=0, keepdims=True)
    sd = np.nanstd(arr, axis=0, keepdims=True)
    out = (arr - mu) / np.clip(sd, 1e-12, None)
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32)


def centered_cosine_similarity(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(a, dtype=np.float32)
    y = x if b is None else np.asarray(b, dtype=np.float32)
    x = x - np.nanmean(x, axis=1, keepdims=True)
    y = y - np.nanmean(y, axis=1, keepdims=True)
    x[~np.isfinite(x)] = 0.0
    y[~np.isfinite(y)] = 0.0
    x = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
    y = y / np.clip(np.linalg.norm(y, axis=1, keepdims=True), 1e-12, None)
    sim = x @ y.T
    sim = np.clip(sim, -1.0, 1.0)
    sim[~np.isfinite(sim)] = 0.0
    return sim.astype(np.float32)


def profile_redundancy_metrics(
    profile: np.ndarray, *, threshold: float = PROFILE_REDUNDANCY_MAX_SIM,
) -> tuple[float, int]:
    p = np.asarray(profile, dtype=np.float32)
    if p.ndim != 2 or int(p.shape[0]) <= 1:
        return 0.0, 0
    sim = centered_cosine_similarity(p)
    upper = np.triu_indices(int(p.shape[0]), k=1)
    vals = sim[upper]
    if vals.size == 0:
        return 0.0, 0
    max_sim = float(np.max(vals))
    redundant_pairs = int(np.sum(vals > float(threshold)))
    return max_sim, redundant_pairs


def suppress_known_runtime_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message="Graph is not fully connected, spectral embedding may not work as expected",
        category=UserWarning,
    )


def reorder_labels_by_size(labels_zero: np.ndarray) -> np.ndarray:
    labels_zero = labels_zero.astype(np.int32, copy=False)
    keys = sorted(int(k) for k in np.unique(labels_zero))
    pairs = sorted(((int(np.sum(labels_zero == k)), k) for k in keys), key=lambda x: (-x[0], x[1]))
    mapping = {old: idx + 1 for idx, (_sz, old) in enumerate(pairs)}
    return np.asarray([mapping[int(k)] for k in labels_zero], dtype=np.int32)


def reorder_labels_by_long_axis(labels_full: np.ndarray, axis_order: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels_full, dtype=np.int32)
    order = np.asarray(axis_order, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError(f"Expected 1D labels, got shape={labels.shape}")
    if order.ndim != 1 or int(order.size) != int(labels.size):
        raise ValueError(f"Long-axis order size mismatch: order={order.shape}, labels={labels.shape}")

    x = labels[order]
    rows: list[tuple[float, int]] = []
    for cid in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
        idx = np.flatnonzero(x == int(cid))
        pos = float(np.mean(idx)) if idx.size else float("inf")
        rows.append((pos, int(cid)))
    rows.sort(key=lambda t: (t[0], t[1]))
    mapping = {old: new_idx + 1 for new_idx, (_pos, old) in enumerate(rows)}
    out = np.zeros_like(labels, dtype=np.int32)
    for old, new in mapping.items():
        out[labels == int(old)] = int(new)
    return out


def cluster_ids_by_long_axis(labels_full: np.ndarray, axis_order: np.ndarray) -> list[int]:
    labels = np.asarray(labels_full, dtype=np.int32)
    order = np.asarray(axis_order, dtype=np.int64)
    x = labels[order]
    rows: list[tuple[float, int]] = []
    for cid in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
        idx = np.flatnonzero(x == int(cid))
        pos = float(np.mean(idx)) if idx.size else float("inf")
        rows.append((pos, int(cid)))
    rows.sort(key=lambda t: (t[0], t[1]))
    return [int(cid) for _pos, cid in rows]


def spectral_cluster_from_affinity(aff: np.ndarray, k: int, seed: int) -> np.ndarray:
    model = SpectralClustering(
        n_clusters=int(k),
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=int(seed),
        n_init=50,
        eigen_solver="arpack",
    )
    labels0 = model.fit_predict(aff)
    return reorder_labels_by_size(labels0)


def reference_path(reference_root: Path, subject: str, atlas: str, filename: str) -> Path:
    return Path(reference_root) / f"sub-{subject}" / "reference" / atlas / str(filename)


def parse_network_names(tsv: Path) -> list[str]:
    with tsv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        out = []
        for row in reader:
            if "canonical_network" in row:
                out.append(str(row["canonical_network"]))
            elif "canonical7_network" in row:
                out.append(str(row["canonical7_network"]))
            elif "family17_network" in row:
                out.append(str(row["family17_network"]))
            elif "lynch20_network" in row:
                out.append(str(row["lynch20_network"]))
            else:
                out.append(str(next(iter(row.values()))))
    if not out:
        raise RuntimeError(f"No network names found: {tsv}")
    return out


def resolve_kong2022_network_registry_column(network_taxonomy: str) -> str:
    taxonomy = str(network_taxonomy)
    if taxonomy == "canonical8":
        return "canonical8"
    if taxonomy in {"canonical7", "lynch7"}:
        return "canonical7"
    if taxonomy == "family17":
        return "family17"
    if taxonomy == "lynch20":
        return "lynch20_network"
    raise ValueError(f"Unsupported network taxonomy: {network_taxonomy!r}")


def network_reference_filenames(network_taxonomy: str) -> tuple[str, str]:
    taxonomy = str(network_taxonomy)
    if taxonomy == "canonical8":
        return "cortex_canonical_networks.tsv", "cortex_canonical_network_timeseries.npy"
    if taxonomy in {"canonical7", "lynch7"}:
        return "cortex_canonical7_networks.tsv", "cortex_canonical7_network_timeseries.npy"
    if taxonomy == "family17":
        return "cortex_family17_networks.tsv", "cortex_family17_network_timeseries.npy"
    if taxonomy == "lynch20":
        return "cortex_lynch20_networks.tsv", "cortex_lynch20_network_timeseries.npy"
    raise ValueError(f"Unsupported network taxonomy: {network_taxonomy!r}")


def load_kong2022_network_reference(
    reference_root: Path,
    subject: str,
    atlas: str,
    network_taxonomy: str,
) -> tuple[list[str], np.ndarray]:
    """Load the declared reference for a network-profile annotation.

    Kong2022 canonical7 is always a deterministic taxonomy transform of the
    ordered 400-ROI reference.  Direct seven-network files are prohibited for
    this atlas because their existence must never select a different feature
    bank by filesystem state.
    """
    order_name, timeseries_name = network_reference_filenames(network_taxonomy)
    order_path = reference_path(reference_root, subject, atlas, order_name)
    timeseries_path = reference_path(reference_root, subject, atlas, timeseries_name)
    if str(atlas) == "kong2022" and str(network_taxonomy) == "canonical7":
        roi_registry_path = reference_path(reference_root, subject, atlas, "cortex_roi_parcels.tsv")
        roi_timeseries_path = reference_path(reference_root, subject, atlas, "cortex_roi_parcel_timeseries.npy")
        if not roi_registry_path.exists() or not roi_timeseries_path.exists():
            raise FileNotFoundError(
                f"Missing explicit Kong2022 canonical7 taxonomy inputs for sub-{subject}: "
                f"registry={roi_registry_path} roi_timeseries={roi_timeseries_path}"
            )
        rows = read_kong2022_roi_registry(roi_registry_path)
        validate_kong2022_roi_registry(rows, path=roi_registry_path)
        rows = canonical7_registry_from_rows(rows)
        roi_timeseries = np.asarray(np.load(roi_timeseries_path), dtype=np.float32)
        if roi_timeseries.ndim != 2 or roi_timeseries.shape[0] != KONG2022_ROI_COUNT:
            raise ValueError(
                f"Kong2022 canonical7 derivation requires a (400,time) ROI time series: "
                f"{roi_timeseries_path} shape={roi_timeseries.shape}"
            )
        names = [str(name) for name in canonical_families_7_order()]
        network_timeseries = aggregate_kong2022_network_timeseries(
            roi_timeseries,
            rows,
            names,
            network_column="canonical7",
        )
        if network_timeseries.shape[0] != len(names):
            raise RuntimeError(
                f"Derived Kong2022 canonical7 shape mismatch for sub-{subject}: "
                f"names={len(names)} timeseries={network_timeseries.shape}"
            )
        return names, network_timeseries

    direct_exists = order_path.exists() or timeseries_path.exists()
    if direct_exists:
        if not order_path.exists() or not timeseries_path.exists():
            raise FileNotFoundError(
                f"Incomplete {network_taxonomy} network reference for sub-{subject}: "
                f"order={order_path} timeseries={timeseries_path}"
            )
        names = parse_network_names(order_path)
        timeseries = np.asarray(np.load(timeseries_path), dtype=np.float32)
        if timeseries.ndim != 2 or timeseries.shape[0] != len(names):
            raise RuntimeError(
                f"Network reference shape/order mismatch for sub-{subject}: "
                f"taxonomy={network_taxonomy} names={len(names)} timeseries={timeseries.shape}"
            )
        if str(atlas) == "kong2022" and str(network_taxonomy) == "canonical7":
            expected_names = [str(name) for name in canonical_families_7_order()]
            if names != expected_names or timeseries.shape[0] != len(expected_names):
                raise RuntimeError(
                    f"Canonical7 network reference contract mismatch for sub-{subject}: "
                    f"names={names} expected={expected_names} shape={timeseries.shape}"
                )
        if str(atlas) == "lynch2024_7net" and str(network_taxonomy) == "lynch7":
            manifest_path = reference_path(reference_root, subject, atlas, "stage_manifest.json")
            if not manifest_path.exists():
                raise FileNotFoundError(f"Missing Lynch7 reference provenance manifest: {manifest_path}")
            manifest = load_json(manifest_path)
            expected = lynch2024_7net_reference_contract(REPO_ROOT / "config" / "cross_atlas_network_merge.json")
            for key, expected_value in expected.items():
                if manifest.get(key) != expected_value:
                    raise ValueError(
                        f"Lynch7 reference contract mismatch: {manifest_path} key={key!r} "
                        f"consumed={manifest.get(key)!r} expected={expected_value!r}"
                    )
        return names, timeseries

    raise FileNotFoundError(
        f"Missing {network_taxonomy} network reference for sub-{subject}: "
        f"order={order_path} timeseries={timeseries_path}"
    )

def gradient_route_for_route_tag(route_tag: str) -> str:
    route = str(route_tag)
    if route == "pfm_17net":
        return "pfm"
    if route == "fixed_17net":
        return "fixed"
    if route == "lynch2024_7net":
        return "lynch2024_20net"
    return route


def standard_reference_root_for_signal_branch(
    shared_root: Path,
    signal_branch: str | None,
) -> Path:
    """Map a CtxReg shared-input root to its explicit PFM reference root."""
    root = Path(shared_root).resolve()
    if not is_ctxreg_signal_branch(signal_branch):
        return root
    route = SIGNAL_BRANCH_TO_ROUTE[str(signal_branch)]
    matches = [idx for idx, part in enumerate(root.parts) if part == route]
    if len(matches) != 1:
        raise ValueError(
            "CtxReg group-atlas runs require a shared root containing exactly one "
            f"{route} path component or an explicit --reference-root: {root}"
        )
    parts = list(root.parts)
    parts[matches[0]] = "pfm"
    return Path(*parts)


def validate_profile_surface_root_for_signal_branch(
    profile_surface_root: Path,
    *,
    signal_branch: str | None,
    shared_root: Path,
) -> Path:
    """Cluster annotation must use the same hippocampal signal branch as the FC.

    CtxReg annotation reads the residualized series under its own shared root;
    every other route reads uncontrolled series and must not borrow CtxReg
    residuals, otherwise clusters and their network profiles mix signal branches.
    """
    profile = Path(profile_surface_root).resolve()
    if is_ctxreg_signal_branch(signal_branch):
        expected = Path(shared_root).resolve()
        if profile != expected:
            raise ValueError(
                "CtxReg cluster annotation must read residualized hippocampal time series "
                f"from its shared root: expected profile_surface_root={expected}, got {profile}"
            )
        return profile
    if any(route in profile.parts for route in CTXREG_ROUTES):
        raise ValueError(
            "Signal-branch mismatch: cluster annotation for a non-CtxReg route must not read "
            f"CtxReg residualized hippocampal time series: profile_surface_root={profile}"
        )
    return profile


def is_network_method(method: str) -> bool:
    return method == "network-spectral"


def is_gradient_method(method: str) -> bool:
    return registered_gradient_method(method)


def is_roi_similarity_method(method: str) -> bool:
    return method == "group-roi-sim-spectral"


def is_multigraph_method(method: str) -> bool:
    """Participant-graph spectral methods; their subject feature is a vertex x vertex similarity."""
    return method in MULTIGRAPH_METHODS


def uses_subject_similarity_bank(method: str) -> bool:
    """Methods whose per-subject feature is the vertex x vertex Pearson similarity of FC fingerprints."""
    return is_multigraph_method(method) or method in SIMILARITY_GRADIENT_METHODS


def method_uses_network_fc_input(method: str) -> bool:
    return is_gradient_method(method) or is_network_method(method) or method in ({"group-roi-sim-spectral", "gradient-spectral", "wta-top"} | GRADIENT_METHODS | MULTIGRAPH_METHODS)


def is_lynch2024_atlas(atlas: str) -> bool:
    return str(atlas).startswith("lynch2024_")


def default_methods_for_atlas(atlas: str) -> tuple[str, ...]:
    """Return the methods generated when the caller omits --methods."""
    if is_lynch2024_atlas(atlas):
        return ("gradient-spectral", "wta-top")
    if str(atlas) != "kong2022":
        return tuple(method for method in METHODS_DEFAULT if method != "wta-top")
    return tuple(METHODS_DEFAULT)


def default_network_taxonomy_for_atlases(atlases: Iterable[str]) -> str:
    """Infer the single taxonomy required by an omitted --network-taxonomy."""
    taxonomies = {
        "lynch7" if str(atlas) == "lynch2024_7net"
        else "lynch20" if str(atlas) == "lynch2024_20net"
        else "canonical7"
        for atlas in atlases
    }
    if len(taxonomies) != 1:
        raise ValueError(
            "Cannot infer one network taxonomy for the requested atlases; "
            f"pass --network-taxonomy explicitly (inferred candidates={sorted(taxonomies)})"
        )
    return next(iter(taxonomies))


def method_allowed_for_atlas(
    method: str,
    atlas: str,
    *,
    network_taxonomy: str = "canonical7",
) -> bool:
    method_key = str(method)
    atlas_key = str(atlas)
    if is_lynch2024_atlas(atlas_key):
        if method_key == "wta-top":
            expected_taxonomy = "lynch7" if atlas_key == "lynch2024_7net" else "lynch20"
            return str(network_taxonomy) == expected_taxonomy
        return method_key in LYNCH2024_METHODS
    if method_key == "wta-top":
        # Direct WTA consumes the Kong2022 400-ROI profile.  Canonical7 and
        # family17 views are derived from that same invariant profile with the
        # same normalized Top-20 count definition over the selected registry
        # column.
        return atlas_key == "kong2022" and str(network_taxonomy) in {"canonical7", "canonical8", "family17"}
    if is_gradient_method(method_key):
        return atlas_key == "kong2022"
    if method_key in KONG2022_ONLY_METHODS:
        return atlas_key in {"kong2022"}
    return True


def validate_requested_methods_for_taxonomy(
    methods: Iterable[str],
    network_taxonomy: str,
    atlases: Iterable[str] | None = None,
) -> None:
    requested_methods = {str(method) for method in methods}
    if "wta-top" in requested_methods and str(network_taxonomy) not in WTA_NETWORK_TAXONOMIES:
        raise ValueError(
            "wta-top requires canonical7/canonical8/family17 for Kong2022 "
            "normalized_top20_count or lynch20/lynch7 for direct Lynch "
            "Fisher-z assignment; "
            f"got {network_taxonomy!r}."
        )
    if "wta-top" in requested_methods and atlases is not None:
        unsupported_atlases = sorted(
            {
                str(atlas)
                for atlas in atlases
                if not method_allowed_for_atlas(
                    "wta-top",
                    str(atlas),
                    network_taxonomy=str(network_taxonomy),
                )
            }
        )
        if unsupported_atlases:
            raise ValueError(
                "wta-top requires Kong2022 normalized_top20_count or the matching Lynch "
                "direct-network Fisher-z path; "
                f"unsupported WTA atlases for taxonomy={network_taxonomy!r}: {unsupported_atlases}"
            )


def redundancy_profile_applicable(atlas: str) -> bool:
    # Kong2022 and both Lynch2024 route registries use the same full-sample
    # cluster-to-network FC redundancy diagnostic.
    return True


def resolve_primary_k(
    *,
    force_final_k: int | None,
    default_k: int | None,
    automatic_k_final: int | None,
) -> tuple[int | None, str]:
    """Choose rendered/exported K without overwriting automatic selection evidence."""
    if force_final_k is not None:
        return int(force_final_k), "force_final_k"
    if default_k is not None:
        return int(default_k), "default_k"
    if automatic_k_final is not None:
        return int(automatic_k_final), "automatic_k"
    return None, "none"


def comparison_k_values(args: argparse.Namespace) -> list[int]:
    if args.k_selection_mode != "fixed-comparison":
        return list(range(int(args.k_min), int(args.k_max) + 1))
    if args.force_final_k is not None:
        k = int(args.force_final_k)
    elif args.k_min == args.k_max:
        k = int(args.k_min)
    else:
        raise ValueError("fixed-comparison requires a single requested K: --k-min=--k-max or --force-final-k")
    if k < 2:
        raise ValueError("fixed-comparison requires K >= 2")
    return [k]


def comparison_primary_k(k: int) -> dict:
    return {"chosen_k_final": int(k), "automatic_k_final": None, "primary_k_source": "fixed_comparison"}


def primary_k_row(rows: list[dict], k: int) -> dict:
    """Return diagnostics for the partition actually exported and displayed."""
    return next(row for row in rows if int(row["k"]) == int(k))


def effective_k_values(k_values: list[int], *, n_support: int, v_min: int, mode: str) -> tuple[int, list[int]]:
    maximum = (n_support - 1 if mode == "fixed-comparison" else
               min(max(k_values), n_support // max(1, v_min)))
    values = [k for k in k_values if k <= maximum and k < n_support]
    if mode == "fixed-comparison" and values != k_values:
        raise ValueError(f"Fixed-comparison requested K={k_values} requires more supported vertices; n_supported={n_support}")
    return maximum, values


def validate_dataset_output_contract(out_root: Path, dataset: str, *, source_roots: list[Path],
                                     comparison: bool = False, repo_root: Path = REPO_ROOT) -> tuple[str, Path]:
    out, repo = Path(out_root).resolve(), Path(repo_root).resolve()
    data_root = DATA_ROOT if repo == REPO_ROOT.resolve() else repo
    roots = {name: data_root / name for name in ("outputs_hcp", "outputs_stanford", "outputs_iowa_jj_tms")}
    if data_root != repo and not comparison:
        raise ValueError(
            f"This checkout reads dataset trees from {data_root}; it may only write "
            "--comparison-output runs under its own tmp directory"
        )
    if comparison:
        if dataset == "auto":
            raise ValueError("A comparison requires an explicit --dataset")
        if not out.is_relative_to(repo / "tmp"):
            raise ValueError(f"Comparison output is outside repository tmp: {out}")
    elif dataset == "auto":
        matches = [key for key, root in roots.items() if out.is_relative_to(root)]
        if len(matches) != 1:
            raise ValueError("Cannot infer --dataset from --out-root; pass an explicit dataset")
        dataset = matches[0]
    if dataset not in roots:
        raise ValueError(f"Unsupported dataset: {dataset}")
    expected = roots[dataset]
    if comparison:
        for source in source_roots:
            if not Path(source).resolve().is_relative_to(expected):
                raise ValueError(f"Comparison source {source} disagrees with declared dataset {dataset}: {expected}")
    elif not out.is_relative_to(expected):
        raise ValueError(f"--out-root {out} is outside the declared dataset root {expected}")
    return dataset, expected


def primary_k_provenance(
    *,
    force_final_k: int | None,
    default_k: int | None,
    automatic_k_final: int | None,
) -> dict[str, int | str | None]:
    chosen_k_final, primary_k_source = resolve_primary_k(
        force_final_k=force_final_k,
        default_k=default_k,
        automatic_k_final=automatic_k_final,
    )
    return {
        "chosen_k_final": chosen_k_final,
        "automatic_k_final": automatic_k_final,
        "primary_k_source": primary_k_source,
    }


def resolve_selected_primary_k(
    *,
    k_final: int,
    k_selection_mode: str,
    forced_fixed_k: bool,
    candidate_pool: list[dict],
    survivors: list[dict],
    force_final_k: int | None,
    default_k: int | None,
) -> dict[str, int | str | None]:
    """Record the rendered K without requiring redundancy diagnostics in fixed mode."""
    if k_selection_mode == "fixed-comparison":
        return comparison_primary_k(k_final)

    automatic_k_final = None if (forced_fixed_k or not candidate_pool) else int(survivors[0]["k"])
    primary_k = primary_k_provenance(
        force_final_k=force_final_k,
        default_k=default_k,
        automatic_k_final=automatic_k_final,
    )
    if not forced_fixed_k and not candidate_pool and default_k is not None:
        return {
            "chosen_k_final": int(default_k),
            "primary_k_source": "default_k",
            "automatic_k_final": None,
        }
    return primary_k


def resolve_primary_k_options(
    args: argparse.Namespace, k_values: list[int], argv: list[str] | None = None,
) -> None:
    """Validate the primary-K options; an automatic primary K clears the display default."""
    fixed = args.k_selection_mode == "fixed-comparison"
    if args.k_final_rule != K_FINAL_RULES[0] and fixed:
        raise ValueError("--k-final-rule applies to --k-selection-mode redundancy-aware")
    if args.automatic_primary_k:
        if option_was_given("--default-k", argv) or args.force_final_k is not None:
            raise ValueError("--automatic-primary-k cannot be combined with --default-k or --force-final-k")
        if fixed:
            raise ValueError("--automatic-primary-k requires --k-selection-mode redundancy-aware")
        args.default_k = None
    if args.undersized_default_k_fallback is not None:
        if args.default_k is None or args.force_final_k is not None or fixed:
            raise ValueError(
                "--undersized-default-k-fallback requires --default-k in redundancy-aware mode without --force-final-k"
            )
        if int(args.undersized_default_k_fallback) not in k_values:
            raise ValueError(
                "--undersized-default-k-fallback must be within the requested K range, "
                f"got {args.undersized_default_k_fallback}"
            )


def apply_undersized_default_k_fallback(
    primary_k: dict[str, int | str | None], rows: list[dict], fallback_k: int,
) -> dict[str, int | str | None]:
    """Export ``fallback_k`` when the default-K partition has a parcel below the minimum size.

    Only a ``default_k`` primary is replaced. The fallback partition must meet
    the size gate itself; the replaced K is recorded as ``undersized_default_k``.
    """
    if primary_k["primary_k_source"] != "default_k":
        return primary_k
    by_k = {int(row["k"]): row for row in rows}
    default_k = int(primary_k["chosen_k_final"])
    if default_k not in by_k:
        raise RuntimeError(f"Default K={default_k} was not evaluated; evaluated={sorted(by_k)}")
    if int(by_k[default_k]["nontrivial_min_size_ok"]) == 1:
        return primary_k
    if int(fallback_k) not in by_k:
        raise RuntimeError(f"Fallback K={fallback_k} was not evaluated; evaluated={sorted(by_k)}")
    if int(by_k[int(fallback_k)]["nontrivial_min_size_ok"]) != 1:
        raise RuntimeError(
            f"Default K={default_k} and fallback K={fallback_k} both have a parcel below the minimum size "
            f"({by_k[default_k]['min_cluster_size']} and {by_k[int(fallback_k)]['min_cluster_size']} vertices, "
            f"minimum {by_k[default_k]['V_min_effective']})"
        )
    return {
        **primary_k, "chosen_k_final": int(fallback_k),
        "primary_k_source": "default_k_undersized_fallback", "undersized_default_k": default_k,
    }


def is_clusterless_method(method: str) -> bool:
    return method in CLUSTERLESS_METHODS


def method_uses_subject_fc_preprocessing(method: str) -> bool:
    return method in SUBJECT_FC_PREPROCESS_METHODS


def feature_family_for_method(method: str, atlas: str, gradient_components: tuple[int, ...]) -> str:
    if method == "group-roi-sim-spectral":
        return "vertex_to_kong2022_400roi_fc_to_vertex_similarity_centered_cosine"
    if method == MULTIGRAPH_METHOD:
        return "subject_vertex_similarity_graphs_of_kong2022_400roi_fc_spectral_coordinates"
    if method == MEAN_GRAPH_METHOD:
        return "mean_subject_vertex_similarity_graph_of_kong2022_400roi_fc_spectral_coordinates"
    if method == MULTIGRAPH_GRADIENT_METHOD:
        return "subject_vertex_similarity_graphs_of_kong2022_400roi_fc_diffusion_coordinates"
    if method == MEAN_SIMILARITY_WARD_METHOD:
        comp_tag = "".join(f"g{int(c)}" for c in gradient_components)
        return f"mean_subject_vertex_similarity_of_kong2022_400roi_fc_diffusion_gradients_{comp_tag}"
    if method == SUBJECT_GRADIENT_WARD_METHOD:
        comp_tag = "".join(f"g{int(c)}" for c in gradient_components)
        return f"procrustes_mean_of_subject_similarity_diffusion_gradients_kong2022_400roi_fc_{comp_tag}"
    if is_gradient_method(method):
        comp_tag = "".join(f"g{int(c)}" for c in gradient_components)
        if is_lynch2024_atlas(atlas) or str(atlas) == "kong2022":
            if str(atlas) == "kong2022":
                return f"group_vertex_to_kong2022_400roi_fc_diffusion_gradients_{comp_tag}"
            return f"group_vertex_to_{atlas}_network_fc_diffusion_gradients_{comp_tag}"
        return f"group_vertex_to_{atlas}_roi_fc_diffusion_gradients_{comp_tag}"
    if method == "wta-top":
        if not method_allowed_for_atlas(
            method,
            atlas,
            network_taxonomy=("lynch7" if str(atlas) == "lynch2024_7net" else "lynch20")
            if is_lynch2024_atlas(atlas)
            else "canonical7",
        ):
            raise ValueError(
                "wta-top requires Kong2022 normalized_top20_count or direct Lynch "
                f"Fisher-z; atlas={atlas!r} has no compatible WTA path"
            )
        if is_lynch2024_atlas(atlas):
            return f"vertex_to_{atlas}_network_fc_fisherz"
        return f"vertex_to_{atlas}_roi_fc_fisherz_normalized_top20_count"
    if method == "network-spectral" and atlas == "kong2022":
        return "vertex_to_kong2022_network_fc_fisherz_colzscore"
    if method == "network-spectral":
        return "vertex_to_canonical_network_fc_fisherz_colzscore"
    if method == "intrinsic-spectral":
        return "vertex_to_vertex_intrinsic_fc_fisherz_diag0"
    return "vertex_to_vertex_intrinsic_fc_fisherz_diag0_plus_partial_profile"


def fc_mode_for_method(method: str, atlas: str | None = None) -> str:
    if method == "group-roi-sim-spectral":
        return "pearson_vertex_to_roi_then_centered_cosine_vertex_to_vertex"
    if is_multigraph_method(method):
        return "pearson_vertex_to_roi_fisherz_then_pearson_vertex_to_vertex_per_subject"
    if method == MEAN_SIMILARITY_WARD_METHOD:
        return "pearson_vertex_to_roi_fisherz_then_pearson_vertex_to_vertex_per_subject_then_subject_mean_then_diffusion_gradients"
    if method == SUBJECT_GRADIENT_WARD_METHOD:
        return "pearson_vertex_to_roi_fisherz_then_pearson_vertex_to_vertex_per_subject_then_subject_diffusion_gradients_procrustes_mean"
    if is_gradient_method(method):
        if atlas is not None and (is_lynch2024_atlas(atlas) or str(atlas) == "kong2022"):
            if str(atlas) == "kong2022":
                return "pearson_vertex_to_kong2022_400roi_group_average_then_diffusion_gradients"
            return "pearson_vertex_to_network_group_average_then_diffusion_gradients"
        return "pearson_vertex_to_roi_group_average_then_diffusion_gradients"
    if method == "wta-top":
        if atlas is not None and not method_allowed_for_atlas(
            method,
            atlas,
            network_taxonomy=("lynch7" if str(atlas) == "lynch2024_7net" else "lynch20")
            if is_lynch2024_atlas(atlas)
            else "canonical7",
        ):
            raise ValueError(
                "wta-top requires Kong2022 normalized_top20_count or direct Lynch "
                f"Fisher-z; atlas={atlas!r} has no compatible WTA path"
            )
        if atlas is not None and is_lynch2024_atlas(atlas):
            return "pearson_vertex_to_network"
        return "pearson_vertex_to_roi"
    if method == "network-spectral":
        return "pearson"
    if method == "intrinsic-spectral":
        return "intrinsic_vertex_to_vertex_pearson"
    return "intrinsic_vertex_to_vertex_partial_profile"


def resolve_subject_fc_mode(subject_fc_mode: str, subject_fc_top_percent: int | None) -> SubjectFCMode:
    mode = str(subject_fc_mode)
    if mode not in SUBJECT_FC_MODE_CHOICES:
        raise ValueError(f"--subject-fc-mode must be one of {SUBJECT_FC_MODE_CHOICES}, got {subject_fc_mode}")
    if mode == "full":
        if subject_fc_top_percent is not None:
            raise ValueError("--subject-fc-top-percent is not allowed when --subject-fc-mode=full")
        return SubjectFCMode(mode="full", mode_tag="full", top_percent=None, top_fraction=1.0, binarize=False)
    if subject_fc_top_percent is None:
        raise ValueError("--subject-fc-top-percent is required when --subject-fc-mode=top-percent")
    pct = int(subject_fc_top_percent)
    if pct <= 0 or pct > 100:
        raise ValueError(f"--subject-fc-top-percent must be within (0, 100], got {pct}")
    return SubjectFCMode(
        mode="top-percent",
        mode_tag=f"{pct}perc",
        top_percent=pct,
        top_fraction=float(pct) / 100.0,
        binarize=True,
    )


def subject_fc_preprocessing_manifest_block(method: str, subject_fc_mode: SubjectFCMode) -> dict[str, object]:
    enabled = bool(method_uses_subject_fc_preprocessing(method))
    if not enabled:
        return {
            "enabled": False,
            "mode": None,
            "mode_tag": None,
            "top_percent": None,
            "applied_methods": sorted(SUBJECT_FC_PREPROCESS_METHODS),
            "value_rule": SUBJECT_FC_VALUE_RULE,
        }
    return {
        "enabled": True,
        "mode": str(subject_fc_mode.mode),
        "mode_tag": str(subject_fc_mode.mode_tag),
        "top_percent": (
            None if subject_fc_mode.top_percent is None else int(subject_fc_mode.top_percent)
        ),
        "applied_methods": sorted(SUBJECT_FC_PREPROCESS_METHODS),
        "value_rule": SUBJECT_FC_VALUE_RULE,
    }


def _n_keep_from_fraction(n_total: int, top_fraction: float) -> int:
    if int(n_total) <= 0:
        return 0
    frac = float(np.clip(float(top_fraction), 0.0, 1.0))
    if frac <= 0.0:
        return 0
    if frac >= 1.0:
        return int(n_total)
    return int(max(1, math.ceil(frac * float(n_total))))


def _select_top_mask_1d(values: np.ndarray, top_fraction: float) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = np.isfinite(x)
    keep = np.zeros((x.size,), dtype=bool)
    n_finite = int(np.sum(finite))
    if n_finite <= 0:
        return keep
    n_keep = _n_keep_from_fraction(n_finite, top_fraction)
    if n_keep <= 0:
        return keep
    if n_keep >= n_finite:
        keep[finite] = True
        return keep
    finite_idx = np.flatnonzero(finite)
    finite_vals = x[finite_idx]
    top_local = np.argpartition(finite_vals, n_finite - n_keep)[n_finite - n_keep:]
    keep[finite_idx[top_local]] = True
    return keep


def preprocess_network_fc_map(fc: np.ndarray, top_fraction: float, binarize: bool) -> np.ndarray:
    arr = np.asarray(fc, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D FC matrix for network preprocessing, got shape={arr.shape}")
    out = arr.copy()
    flat = arr.reshape(-1)
    keep_flat = _select_top_mask_1d(flat, top_fraction)
    finite_flat = np.isfinite(flat)
    out_flat = out.reshape(-1)
    out_flat[finite_flat & ~keep_flat] = 0.0
    if bool(binarize):
        out_flat[keep_flat] = 1.0
    out_flat[~finite_flat] = np.nan
    return out.astype(np.float32)


def preprocess_intrinsic_fc_map(fc: np.ndarray, top_fraction: float, binarize: bool) -> np.ndarray:
    arr = np.asarray(fc, dtype=np.float32)
    if arr.ndim != 2 or int(arr.shape[0]) != int(arr.shape[1]):
        raise ValueError(f"Expected square FC matrix for intrinsic preprocessing, got shape={arr.shape}")
    out = arr.copy()
    tri = np.triu_indices(int(arr.shape[0]), k=1)
    tri_vals = arr[tri]
    keep_tri = _select_top_mask_1d(tri_vals, top_fraction)
    finite_tri = np.isfinite(tri_vals)
    zero_tri = finite_tri & ~keep_tri
    if np.any(zero_tri):
        zi = tri[0][zero_tri]
        zj = tri[1][zero_tri]
        out[zi, zj] = 0.0
        out[zj, zi] = 0.0
    if np.any(keep_tri):
        ki = tri[0][keep_tri]
        kj = tri[1][keep_tri]
        if bool(binarize):
            kv = np.ones((ki.size,), dtype=np.float32)
        else:
            kv = arr[ki, kj].astype(np.float32, copy=False)
        out[ki, kj] = kv
        out[kj, ki] = kv
    out[~np.isfinite(arr)] = np.nan
    np.fill_diagonal(out, 0.0)
    return out.astype(np.float32)


def preprocess_subject_fc_matrix(method: str, fc: np.ndarray, subject_fc_mode: SubjectFCMode) -> np.ndarray:
    arr = np.asarray(fc, dtype=np.float32)
    if not method_uses_subject_fc_preprocessing(method):
        return arr
    if str(subject_fc_mode.mode) == "full":
        return arr
    if str(subject_fc_mode.mode) != "top-percent":
        raise ValueError(f"Unsupported subject FC mode: {subject_fc_mode.mode}")
    if method in {"group-roi-sim-spectral", "network-spectral"}:
        return preprocess_network_fc_map(arr, subject_fc_mode.top_fraction, bool(subject_fc_mode.binarize))
    if method == "intrinsic-spectral":
        return preprocess_intrinsic_fc_map(arr, subject_fc_mode.top_fraction, bool(subject_fc_mode.binarize))
    return arr


def roi_similarity_from_network_fc(fc: np.ndarray) -> np.ndarray:
    arr = np.asarray(fc, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D vertex-to-ROI FC matrix, got shape={arr.shape}")
    sim = centered_cosine_similarity(arr)
    sim = 0.5 * (sim + sim.T)
    sim = np.clip(sim, -1.0, 1.0)
    sim[~np.isfinite(sim)] = 0.0
    np.fill_diagonal(sim, 0.0)
    return sim.astype(np.float32)


def subject_similarity_feature(fc_fisherz: np.ndarray, valid_mask: np.ndarray, *, source: Path) -> np.ndarray:
    """Participant vertex-by-vertex Pearson similarity; unobserved vertices stay NaN."""
    similarity = vertex_similarity(fc_fisherz)
    if not np.array_equal(observed_vertices(similarity), np.asarray(valid_mask, dtype=bool)):
        raise ValueError(f"Observed FC fingerprints differ from the participant validity mask: {source}")
    return similarity


def brainspace_version() -> str:
    try:
        return str(importlib.metadata.version("brainspace"))
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def brainspace_diffusion_gradients(
    features: np.ndarray,
    *,
    n_components: int = GRADIENT_N_COMPONENTS_COMPUTED,
    hippomaps_root: Path = DEFAULT_HIPPOMAPS_ROOT,
    gradient_fit_fn: object | None = None,
    gradient_options: dict[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    group_fc = np.asarray(features, dtype=np.float32)
    if group_fc.ndim != 2:
        raise ValueError(f"Expected 2D group FC feature matrix, got shape={group_fc.shape}")
    n_comp = int(n_components)
    if n_comp <= 0:
        raise ValueError(f"n_components must be positive, got {n_components}")
    if int(group_fc.shape[0]) <= n_comp:
        raise RuntimeError(
            f"Not enough supported vertices to compute G1..G{n_comp}: n_vertices={group_fc.shape[0]}"
        )
    ensure_hippomaps_on_path(hippomaps_root)
    fit_fn = fit_diffusion_gradients if gradient_fit_fn is None else gradient_fit_fn
    options = dict(gradient_options or {})
    gradients, lambdas, ev_fractions = fit_fn(
        group_fc, n_components=n_comp,
        feature_scaling=options.get("feature_scaling", "col_zscore"),
        sparsify_stage=options.get("sparsify_stage", "features"),
        sparsity=options.get("sparsity"),
    )
    gradients = np.asarray(gradients, dtype=np.float32)
    lambdas = np.asarray(lambdas, dtype=np.float64).reshape(-1)
    ev_fractions = np.asarray(ev_fractions, dtype=np.float64).reshape(-1)
    if gradients.ndim != 2:
        raise RuntimeError(f"Expected 2D GradientMaps gradients, got shape={gradients.shape}")
    if gradients.shape[1] < n_comp:
        raise RuntimeError(
            f"GradientMaps returned {gradients.shape[1]} gradients, expected at least {n_comp}"
        )
    if lambdas.size < n_comp:
        raise RuntimeError(f"GradientMaps returned {lambdas.size} lambdas, expected at least {n_comp}")
    if ev_fractions.size < n_comp:
        raise RuntimeError(f"GradientMaps returned {ev_fractions.size} EV fractions, expected at least {n_comp}")
    gradients = gradients[:, :n_comp]
    gradients[~np.isfinite(gradients)] = 0.0
    lambdas = lambdas[:n_comp]
    ev_fractions = ev_fractions[:n_comp]
    if not np.all(np.isfinite(lambdas)):
        raise RuntimeError(f"GradientMaps lambdas contain non-finite values: {lambdas.tolist()}")
    if not np.all(np.isfinite(ev_fractions)):
        raise RuntimeError(f"GradientMaps EV fractions contain non-finite values: {ev_fractions.tolist()}")
    return gradients.astype(np.float32), lambdas.astype(np.float64), ev_fractions.astype(np.float64)


def mean_similarity_options(options: dict[str, object]) -> float:
    """The mean-similarity affinity is TopSim-style: unscaled and row-sparsified after aggregation."""
    if options.get("sparsify_stage") != "affinity" or options.get("feature_scaling") != "none" or options.get("sparsity") is None:
        raise ValueError(
            f"{MEAN_SIMILARITY_WARD_METHOD} requires --profile topsim (unscaled fingerprints, affinity-stage "
            f"sparsification with an explicit sparsity); got feature_scaling={options.get('feature_scaling')!r} "
            f"sparsify_stage={options.get('sparsify_stage')!r} sparsity={options.get('sparsity')!r}"
        )
    return float(options["sparsity"])


def subject_gradient_features(
    *,
    feature_bank: dict[str, np.ndarray],
    subjects_subset: list[str],
    support_mask: np.ndarray,
    gradient_components: tuple[int, ...],
    hippomaps_root: Path,
    gradient_options: dict[str, object],
) -> tuple[np.ndarray, dict[str, object]]:
    """Procrustes mean of each participant's own diffusion gradients, aligned to the mean-similarity template."""
    options = dict(gradient_options or {})
    sparsity = mean_similarity_options(options)
    idx = np.flatnonzero(support_mask)
    template, template_meta = build_group_gradient_features(
        method=MEAN_SIMILARITY_WARD_METHOD, feature_bank=feature_bank, subjects_subset=subjects_subset,
        support_mask=support_mask, hippomaps_root=hippomaps_root, all_components=True,
        gradient_options={**options, "clustering_feature_scaling": "none"},
    )
    subject_gradients = []
    for s in subjects_subset:
        block = np.asarray(feature_bank[s], dtype=np.float32)[np.ix_(idx, idx)]
        rows = np.flatnonzero(np.isfinite(np.diag(block)))
        try:
            gradients, _, _ = fit_mean_similarity_gradients(
                block[np.ix_(rows, rows)][None], n_components=GRADIENT_N_COMPONENTS_COMPUTED, sparsity=sparsity,
            )
        except (ValueError, RuntimeError) as exc:
            raise RuntimeError(f"Subject sub-{s} gradients failed: {exc}") from exc
        subject_gradients.append((rows, gradients))
    mean, procrustes_meta = generalized_procrustes_mean(subject_gradients, template)
    comps = parse_gradient_components(",".join(str(c) for c in gradient_components))
    selected = mean[:, [int(c) - 1 for c in comps]]
    selected, clustering_transform = scale_gradient_features(
        selected, scaling=options.get("clustering_feature_scaling", "none"),
    )
    meta = dict(template_meta)
    meta.update(
        gradient_components=[int(c) for c in comps],
        gradient_component_labels=[f"G{int(c)}" for c in comps],
        selected_components=[f"G{int(c)}" for c in comps],
        gradient_embedding_call=SUBJECT_GRADIENT_EMBEDDING_CALL,
        clustering_feature_transform=clustering_transform,
        template_method=MEAN_SIMILARITY_WARD_METHOD,
        subject_gradient_alignment=procrustes_meta,
        n_subject_gradient_fits=len(subject_gradients),
    )
    meta.pop("gradient_eigenvalues", None)
    meta.pop("ev_fractions", None)
    return selected.astype(np.float32), meta


def build_group_gradient_features(
    *,
    method: str = "gradient-ward",
    feature_bank: dict[str, np.ndarray],
    subjects_subset: list[str],
    support_mask: np.ndarray,
    gradient_components: tuple[int, ...] = DEFAULT_GRADIENT_COMPONENTS,
    hippomaps_root: Path = DEFAULT_HIPPOMAPS_ROOT,
    gradient_fit_fn: object | None = None,
    gradient_options: dict[str, object] | None = None,
    all_components: bool = False,
) -> tuple[np.ndarray, dict[str, object]]:
    """Group gradient features; ``all_components`` returns every computed, aligned gradient unselected."""
    idx = np.flatnonzero(support_mask)
    if idx.size == 0:
        raise RuntimeError("Support mask has zero vertices")
    if method == SUBJECT_GRADIENT_WARD_METHOD:
        return subject_gradient_features(
            feature_bank=feature_bank, subjects_subset=subjects_subset, support_mask=support_mask,
            gradient_components=gradient_components, hippomaps_root=hippomaps_root,
            gradient_options=dict(gradient_options or {}),
        )
    comps = (tuple(range(1, GRADIENT_N_COMPONENTS_COMPUTED + 1)) if all_components
             else parse_gradient_components(",".join(str(c) for c in gradient_components)))
    mean_similarity = method == MEAN_SIMILARITY_WARD_METHOD
    options = dict(gradient_options or {})
    if mean_similarity:
        similarity_sparsity = mean_similarity_options(options)
        stack = np.stack(
            [np.asarray(feature_bank[s], dtype=np.float32)[np.ix_(idx, idx)] for s in subjects_subset], axis=0,
        )
        group_shape = [int(v) for v in stack.shape[1:]]
        gradients, lambdas, ev_fractions = fit_mean_similarity_gradients(
            stack, n_components=GRADIENT_N_COMPONENTS_COMPUTED, sparsity=similarity_sparsity,
        )
        del stack
    else:
        mats = []
        for s in subjects_subset:
            m = np.asarray(feature_bank[s], dtype=np.float32)[idx, :]
            mats.append(m)
        stack = np.stack(mats, axis=0).astype(np.float32)
        group_fc = np.nanmean(stack, axis=0)
        group_fc[~np.isfinite(group_fc)] = 0.0
        group_shape = [int(v) for v in group_fc.shape]
        gradients, lambdas, ev_fractions = brainspace_diffusion_gradients(
            group_fc,
            n_components=GRADIENT_N_COMPONENTS_COMPUTED,
            hippomaps_root=hippomaps_root,
            gradient_fit_fn=gradient_fit_fn,
            gradient_options=gradient_options,
        )
    alignment_meta = {"enabled": False, "role": "native_reference_fit"}
    if "alignment_fc_paths" in options:
        anchor_smoothing = str(options.get("alignment_anchor_smoothing", "4mm"))
        anchor_fc, anchor_masks = [], []
        for subject in subjects_subset:
            fc_path = Path(options["alignment_fc_paths"][subject])
            if "alignment_run_id" in options:
                run_dir = fc_path.parent / "runs" / f"run-{options['alignment_run_id']}"
                run_fc_path = run_dir / fc_path.name
                if not run_fc_path.exists():
                    alt = run_dir / (fc_path.stem + "_active.npy")
                    if alt.exists():
                        run_fc_path = alt
                fc_path = run_fc_path
            mask = np.asarray(np.load(options["alignment_mask_paths"][subject]), dtype=bool)
            raw_fc = np.asarray(np.load(fc_path), dtype=np.float32)
            n_rois = int(raw_fc.shape[1]) if mean_similarity else int(group_fc.shape[1])
            if raw_fc.shape[0] == int(np.sum(mask)) and raw_fc.shape[1] == n_rois:
                full_raw = np.full((mask.size, n_rois), np.nan, dtype=np.float32)
                full_raw[mask, :] = raw_fc
                raw_fc = full_raw
            if raw_fc.shape != (mask.size, n_rois):
                raise ValueError(f"{anchor_smoothing}-left anchor FC/mask mismatch: {fc_path}: {raw_fc.shape} vs {mask.shape}")
            permutation = options.get("alignment_vertex_permutations", {}).get(subject)
            if permutation is not None:
                raw_fc, mask = raw_fc[permutation], mask[permutation]
            raw_fc = raw_fc.copy()
            raw_fc[~mask] = np.nan
            anchor_fc.append(fisher_z_transform_fc(raw_fc))
            anchor_masks.append(mask)
        anchor_support = np.mean(np.stack(anchor_masks), axis=0) >= float(options["alignment_coverage_threshold"])
        if anchor_support.size != support_mask.size:
            raise ValueError(f"{anchor_smoothing}-left anchor and target vertex counts differ")
        if mean_similarity:
            # The anchor is this method's own 2 mm-left fit on the same subjects.
            anchor_stack = np.stack(
                [vertex_similarity(fc)[np.ix_(anchor_support, anchor_support)] for fc in anchor_fc], axis=0,
            )
            anchor_gradients, _, _ = fit_mean_similarity_gradients(
                anchor_stack, n_components=GRADIENT_N_COMPONENTS_COMPUTED, sparsity=similarity_sparsity,
            )
            del anchor_stack
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                anchor_mean = np.nanmean(np.stack(anchor_fc)[:, anchor_support], axis=0)
            anchor_mean[~np.isfinite(anchor_mean)] = 0.
            anchor_gradients, _, _ = brainspace_diffusion_gradients(
                anchor_mean, n_components=GRADIENT_N_COMPONENTS_COMPUTED,
                hippomaps_root=hippomaps_root, gradient_fit_fn=gradient_fit_fn,
                gradient_options=gradient_options,
            )
        target_full = np.full((support_mask.size, gradients.shape[1]), np.nan, dtype=np.float32)
        reference_full = target_full.copy()
        target_full[support_mask] = gradients
        reference_full[anchor_support] = anchor_gradients
        gradients, _, lambdas, ev_fractions, alignment_meta = align_gradient_components_to_reference(
            active=gradients, full=target_full, lambdas=lambdas, ev_fractions=ev_fractions,
            reference_full=reference_full, component_labels=[f"G{i+1}" for i in range(gradients.shape[1])],
        )
        alignment_meta.update(enabled=True, anchor_smoothing=anchor_smoothing, anchor_hemi="L",
                              training_subjects=list(subjects_subset))
    selected = gradients[:, [int(c) - 1 for c in comps]]
    selected[~np.isfinite(selected)] = 0.0
    options = dict(gradient_options or {})
    selected, clustering_transform = scale_gradient_features(
        selected, scaling=options.get("clustering_feature_scaling", "none"),
    )
    preprocessing = gradient_preprocessing_settings(
        400 if mean_similarity else group_shape[1], feature_scaling=options.get("feature_scaling", "col_zscore"),
        sparsify_stage=options.get("sparsify_stage", "features"), sparsity=options.get("sparsity"),
    )
    if mean_similarity:
        preprocessing["kernel"] = "none_precomputed_subject_mean_pearson_vertex_similarity"
    selected_labels = [f"G{int(c)}" for c in comps]
    hippomaps_root_resolved = ensure_hippomaps_on_path(hippomaps_root)
    meta = {
        "gradient_components": [int(c) for c in comps],
        "gradient_component_labels": selected_labels,
        "selected_components": selected_labels,
        "gradient_embedding_engine": "exact_diffusion_map_via_hippomaps_adapter",
        "brainspace_version": brainspace_version(),
        "gradient_embedding_call": (
            MEAN_SIMILARITY_EMBEDDING_CALL if mean_similarity else
            "compute_affinity(explicitly_scaled_group_mean_fisherz_fc, kernel='pearson', pre_sparsify=declared_stage); mean_symmetrize; diffusion_map_embedding(affinity, alpha=0.5, diffusion_time=0)"
        ),
        "feature_zscore_pre_gradientmaps": preprocessing["feature_scaling"],
        "gradient_preprocessing": preprocessing,
        "clustering_feature_transform": clustering_transform,
        "kernel": preprocessing["kernel"] if mean_similarity else GRADIENT_BRAINSPACE_KERNEL,
        "brainspace_sparsity": preprocessing["effective_sparsity"],
        "brainspace_sparsity_policy": "explicit setting; unspecified sparsity retains 0.0 for <=50 features and 0.9 otherwise",
        "n_components_computed": int(GRADIENT_N_COMPONENTS_COMPUTED),
        "components_displayed": [int(c) for c in DEFAULT_GRADIENT_COMPONENTS],
        ("group_mean_vertex_similarity_shape" if mean_similarity else "group_vertex_to_roi_fc_shape"): group_shape,
        "gradient_eigenvalues": [float(v) for v in lambdas],
        "gradient_alignment": alignment_meta,
        "ev_fractions": [float(v) for v in ev_fractions],
        "hippomaps_repo": str(Path(hippomaps_root_resolved).resolve()),
        "hippomaps_import_path": hippomaps_import_path(hippomaps_root_resolved),
        "hippomaps_commit": hippomaps_commit(hippomaps_root_resolved),
    }
    return selected.astype(np.float32), meta


def compute_similarity_gradient_features(
    *,
    method: str,
    feature_bank: dict[str, np.ndarray],
    subjects: list[str],
    support_mask: np.ndarray,
    gradient_components: tuple[int, ...],
    gradient_options: dict[str, object],
) -> tuple[np.ndarray, dict[str, object]]:
    """Full-cohort similarity-based gradients; these methods have no 05_01 producer."""
    features, meta = build_group_gradient_features(
        method=method,
        feature_bank=feature_bank,
        subjects_subset=subjects,
        support_mask=support_mask,
        gradient_components=gradient_components,
        hippomaps_root=HIPPOMAPS_ROOT_OVERRIDE,
        gradient_options=gradient_options,
    )
    if not meta["gradient_alignment"].get("enabled"):
        raise ValueError(f"{method} requires the 2 mm-left alignment anchor")
    source = SUBJECT_GRADIENT_SOURCE if method == SUBJECT_GRADIENT_WARD_METHOD else MEAN_SIMILARITY_GRADIENT_SOURCE
    meta.update(gradient_source=source, n_subjects=len(subjects))
    return features, meta


def load_hippomaps_gradient_features(
    *,
    gradient_root: Path,
    route: str,
    smoothing: str,
    hemi: str,
    support_mask: np.ndarray,
    gradient_components: tuple[int, ...] = DEFAULT_GRADIENT_COMPONENTS,
    expected_signal_branch: str | None = None,
    cohort: str | None = None,
    variant: str = "36p",
    feature_scaling: str = "col_zscore",
    sparsify_stage: str = "features",
    sparsity: float | None = None,
    clustering_feature_scaling: str = "none",
    alignment_anchor_smoothing: str = "4mm",
    profile: str | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    comps = parse_gradient_components(",".join(str(c) for c in gradient_components))
    run_dir = resolve_cohort_gradient_root(
        route=route, cohort=cohort, variant=variant, base_root=Path(gradient_root).resolve(), profile=profile,
    ) / smoothing / f"hemi_{hemi}"
    manifest_path = run_dir / "hippomaps_gradient_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing HippoMaps gradient manifest for gradient-spectral clustering: {manifest_path}. "
            "Run 05_01_compute_hippomaps_gradients.py before 05_02_run_group_atlas.py."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shape = manifest.get("group_fc_shape")
    if not isinstance(shape, list) or len(shape) != 2:
        raise ValueError(f"Gradient manifest lacks group_fc_shape: {manifest_path}")
    preprocessing = gradient_preprocessing_settings(
        int(shape[1]), feature_scaling=feature_scaling,
        sparsify_stage=sparsify_stage, sparsity=sparsity,
    )
    validate_gradient_preprocessing(manifest, settings=preprocessing, manifest_path=manifest_path)
    if cohort is not None or variant != "36p":
        for key, expected in (("cohort", cohort), ("variant", variant)):
            if manifest.get(key) != expected:
                raise ValueError(
                    f"Gradient {key} mismatch: expected={expected!r} "
                    f"manifest={manifest.get(key)!r} path={manifest_path}"
                )
    if str(manifest.get("status", "ok")) != "ok":
        raise RuntimeError(f"HippoMaps gradient manifest is not ok: {manifest_path} status={manifest.get('status')}")
    if expected_signal_branch is not None:
        expected_branch = normalize_signal_branch(expected_signal_branch)
        recorded_branch = manifest.get("signal_branch")
        if is_ctxreg_signal_branch(expected_branch) and recorded_branch != expected_branch:
            raise ValueError(
                "Gradient signal branch mismatch: "
                f"expected={expected_branch!r} manifest={recorded_branch!r} path={manifest_path}"
            )
        if expected_branch == "standard" and recorded_branch not in {None, "standard"}:
            raise ValueError(
                "Gradient signal branch mismatch: "
                f"expected='standard' manifest={recorded_branch!r} path={manifest_path}"
            )
        manifest_route = manifest.get("route")
        if manifest_route is None:
            raise ValueError(
                f"Gradient manifest lacks route provenance for expected signal branch "
                f"{expected_branch!r}: {manifest_path}"
            )
        if str(manifest_route) != str(route):
            raise ValueError(
                "Gradient atlas route mismatch: "
                f"expected={route!r} manifest={manifest_route!r} path={manifest_path}"
            )
    if str(route) in {"pfm", *CTXREG_ROUTES, "fixed"}:
        group_fc_shape = manifest.get("group_fc_shape")
        feature_count = manifest.get("feature_count")
        if not (isinstance(group_fc_shape, list) and len(group_fc_shape) == 2 and int(group_fc_shape[1]) == 400):
            raise ValueError(
                f"Kong gradient manifest must prove a 400-ROI group FC: {manifest_path} "
                f"group_fc_shape={group_fc_shape!r}"
            )
        if feature_count is not None and int(feature_count) != 400:
            raise ValueError(
                f"Kong gradient manifest feature_count is not 400: {manifest_path} feature_count={feature_count!r}"
            )
    local_gradients_path = run_dir / "hippomaps_gradients_full.npy"
    outputs = manifest.get("outputs", {})
    gradients_path_raw = outputs.get("hippomaps_gradients_full") if isinstance(outputs, dict) else None
    if not gradients_path_raw:
        raise ValueError(
            f"Gradient manifest lacks explicit outputs.hippomaps_gradients_full: {manifest_path}"
        )
    manifest_gradients_path = Path(str(gradients_path_raw)).resolve()
    if not manifest_gradients_path.exists():
        raise FileNotFoundError(
            f"Gradient manifest points to a missing full gradient array: {manifest_path} "
            f"path={manifest_gradients_path}; repair provenance explicitly before reuse"
        )
    if local_gradients_path.exists() and local_gradients_path.resolve() != manifest_gradients_path:
        raise ValueError(
            f"Gradient manifest/output path mismatch: {manifest_path} "
            f"local={local_gradients_path.resolve()} manifest={manifest_gradients_path}"
        )
    gradients_path = manifest_gradients_path
    gradients_full = np.load(gradients_path).astype(np.float32)
    support = np.asarray(support_mask, dtype=bool)
    if gradients_full.ndim != 2:
        raise ValueError(f"Expected 2D HippoMaps gradient array: {gradients_path} shape={gradients_full.shape}")
    if gradients_full.shape[0] != support.size:
        raise ValueError(
            f"Gradient/support vertex count mismatch: gradients={gradients_full.shape[0]} support={support.size} "
            f"path={gradients_path}"
        )
    max_comp = max(comps)
    if gradients_full.shape[1] < max_comp:
        raise ValueError(
            f"HippoMaps gradient array has {gradients_full.shape[1]} components; requested G{max_comp}: {gradients_path}"
        )
    selected = gradients_full[np.flatnonzero(support)][:, [int(c) - 1 for c in comps]].astype(np.float32)
    selected[~np.isfinite(selected)] = 0.0
    selected, clustering_transform = scale_gradient_features(selected, scaling=clustering_feature_scaling)
    labels = [f"G{int(c)}" for c in comps]
    meta = {
        "gradient_source": "hippomaps_gradient_artifacts",
        "gradient_preprocessing": preprocessing,
        "clustering_feature_transform": clustering_transform,
        "cohort": manifest.get("cohort"),
        "variant": manifest.get("variant"),
        "hippomaps_gradient_manifest": str(manifest_path.resolve()),
        "hippomaps_gradients_full": str(gradients_path.resolve()),
        "route": route,
        "signal_branch": manifest.get("signal_branch"),
        "smoothing": smoothing,
        "hemi": hemi,
        "gradient_components": [int(c) for c in comps],
        "gradient_component_labels": labels,
        "selected_components": labels,
        "n_components_computed": int(manifest.get("n_components_computed", gradients_full.shape[1])),
        "gradient_embedding_call": manifest.get("gradient_embedding_call"),
        "gradient_shape_full": [int(x) for x in gradients_full.shape],
        "gradient_array_semantics": manifest.get("gradient_array_semantics", "legacy_unaligned_or_unspecified"),
        "feature_count": manifest.get("feature_count"),
        "feature_family": manifest.get("feature_family"),
    }
    if "gradient_alignment" in manifest:
        grad_align = manifest["gradient_alignment"]
        if not isinstance(grad_align, dict):
            raise ValueError(
                f"Malformed gradient_alignment in HippoMaps gradient manifest: {manifest_path}; "
                f"expected dict, got {type(grad_align).__name__}"
            )
        meta["gradient_alignment"] = grad_align
        recorded_anchor = grad_align.get("anchor_smoothing")
        recorded_hemi = grad_align.get("anchor_hemi")
        if recorded_hemi not in {None, "L"}:
            raise ValueError(
                f"HippoMaps gradient alignment anchor_hemi mismatch in {manifest_path}: "
                f"recorded={recorded_hemi!r}; expected 'L'"
            )
        if alignment_anchor_smoothing == "2mm":
            if recorded_anchor != "2mm":
                raise ValueError(
                    f"HippoMaps gradient alignment anchor mismatch: expected={alignment_anchor_smoothing!r} "
                    f"recorded={recorded_anchor!r} path={manifest_path}"
                )
            if grad_align.get("enabled") is not True:
                raise ValueError(
                    f"HippoMaps gradient alignment is disabled in {manifest_path}; expected enabled=True"
                )
        elif recorded_anchor is not None and str(recorded_anchor) != alignment_anchor_smoothing:
            raise ValueError(
                f"HippoMaps gradient alignment anchor mismatch: expected={alignment_anchor_smoothing!r} "
                f"recorded={recorded_anchor!r} path={manifest_path}"
            )
    elif alignment_anchor_smoothing == "2mm":
        raise ValueError(
            f"HippoMaps gradient manifest lacks gradient_alignment declaration: {manifest_path}; "
            f"expected anchor={alignment_anchor_smoothing!r}"
        )
    for key in (
        "ev_fractions",
        "lambdas",
        "brainspace_sparsity",
        "brainspace_sparsity_policy",
        "hippomaps_repo",
        "hippomaps_import_path",
        "hippomaps_commit",
    ):
        if key in manifest:
            meta[key] = manifest[key]
    if "lambdas" in meta and "gradient_eigenvalues" not in meta:
        meta["gradient_eigenvalues"] = [float(v) for v in np.asarray(meta["lambdas"], dtype=np.float64).reshape(-1)]
    return selected, meta


def network_feature_filename(method: str, atlas: str | None = None) -> str:
    if method == "wta-top" and atlas not in {"kong2022", "lynch2024_20net", "lynch2024_7net"}:
        raise ValueError(
            "wta-top network input is restricted to Kong2022 normalized_top20_count "
            "or direct Lynch network Fisher-z paths; "
            f"atlas={atlas!r} is not supported"
        )
    if str(atlas) == "kong2022":
        if method == "network-spectral":
            return "hipp_vertex_to_canonical7_fc.npy"
        if method in ({"group-roi-sim-spectral", "gradient-spectral", "wta-top"} | GRADIENT_METHODS | MULTIGRAPH_METHODS):
            # Historical network_fc is now the strict 400-ROI feature bank.
            return "hipp_vertex_to_network_fc.npy"
    if method in ({"group-roi-sim-spectral", "network-spectral", "gradient-spectral", "wta-top"} | GRADIENT_METHODS):
        return "hipp_vertex_to_network_fc.npy"
    raise ValueError(f"Not a network feature method: {method}")


def runwise_network_feature_filename(method: str, atlas: str | None = None) -> str:
    if method == "group-roi-sim-spectral":
        return "hipp_vertex_to_roi_fc_active.npy"
    if method == "network-spectral" and str(atlas) == "kong2022":
        return "hipp_vertex_to_canonical7_fc_active.npy"
    if method in ({"network-spectral", "gradient-spectral"} | GRADIENT_METHODS | MULTIGRAPH_METHODS):
        return "hipp_vertex_to_network_fc_active.npy"
    raise ValueError(f"Not a network feature method: {method}")


def one_sample_one_sided_vector(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D values, got shape={arr.shape}")
    n = np.sum(np.isfinite(arr), axis=0).astype(np.int32)
    mean = np.nanmean(arr, axis=0).astype(np.float64)
    sd = np.nanstd(arr, axis=0, ddof=1).astype(np.float64)
    t_stat = np.full_like(mean, np.nan, dtype=np.float64)
    p_pos = np.full_like(mean, np.nan, dtype=np.float64)
    p_neg = np.full_like(mean, np.nan, dtype=np.float64)

    valid = n >= 2
    stable = valid & np.isfinite(sd) & (sd > 1e-12)
    if np.any(stable):
        nn = n[stable].astype(np.float64)
        t_stat[stable] = mean[stable] / (sd[stable] / np.sqrt(nn))
        df = np.clip(n[stable] - 1, 1, None)
        p_pos[stable] = stats.t.sf(t_stat[stable], df=df)
        p_neg[stable] = stats.t.cdf(t_stat[stable], df=df)

    degenerate = valid & ~stable
    if np.any(degenerate):
        mdeg = mean[degenerate]
        t_fill = np.zeros_like(mdeg)
        ppos_fill = np.full_like(mdeg, 0.5)
        pneg_fill = np.full_like(mdeg, 0.5)
        pos = mdeg > 0.0
        neg = mdeg < 0.0
        t_fill[pos] = np.inf
        t_fill[neg] = -np.inf
        ppos_fill[pos] = 0.0
        ppos_fill[neg] = 1.0
        pneg_fill[pos] = 1.0
        pneg_fill[neg] = 0.0
        t_stat[degenerate] = t_fill
        p_pos[degenerate] = ppos_fill
        p_neg[degenerate] = pneg_fill

    return mean, t_stat, p_pos, p_neg, n.astype(np.float64)


def split_kong2022_columns_by_network(
    roi_registry: list[dict[str, object]],
    network_column: str = "canonical7",
) -> tuple[dict[str, list[int]], list[str]]:
    column, network_values = _validated_kong2022_network_values(roi_registry, network_column)
    cols_by_network: dict[str, list[int]] = {}
    for col, net in enumerate(network_values):
        cols_by_network.setdefault(net, []).append(int(col))
    network_names = sorted(cols_by_network.keys())
    return cols_by_network, network_names


def build_normalized_topn_count_vertex_rows(
    *,
    mean_profile: np.ndarray,
    support_indices: np.ndarray,
    roi_registry: list[dict[str, object]],
    top_n_cap: int,
    network_column: str = "canonical7",
) -> list[dict[str, object]]:
    """Assign vertices by top-positive-ROI counts normalized by network ROI opportunity.

    The selected set remains the strongest top-N positive ROI FC values. Only the
    final network vote is normalized, using finite available ROI columns for the
    same vertex as the denominator.
    """
    arr = np.asarray(mean_profile, dtype=np.float32)
    idx = np.asarray(support_indices, dtype=np.int32).reshape(-1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mean_profile, got shape={arr.shape}")
    if arr.shape[0] != idx.size:
        raise ValueError(f"Profile/support mismatch: profile={arr.shape[0]} support_indices={idx.size}")
    if arr.shape[1] != len(roi_registry):
        raise ValueError(f"Profile/ROI registry mismatch: profile_cols={arr.shape[1]} registry={len(roi_registry)}")
    column, roi_networks = _validated_kong2022_network_values(roi_registry, network_column)
    rows: list[dict[str, object]] = []
    for local_vidx, global_vidx in enumerate(idx):
        values = arr[local_vidx, :]
        finite_idx = np.flatnonzero(np.isfinite(values))
        available_by_network: dict[str, int] = {}
        for col in finite_idx:
            net = roi_networks[int(col)]
            available_by_network[net] = int(available_by_network.get(net, 0) + 1)
        pos_idx = np.flatnonzero(np.isfinite(values) & (values > 0.0))
        if pos_idx.size == 0:
            rows.append(
                {
                    "vertex_index": int(global_vidx),
                    "status": "unassigned",
                    "unassigned_reason": "no_positive_roi",
                    "top1_network": "unassigned",
                    "top2_network": "unassigned",
                    "top1_count": 0,
                    "top2_count": 0,
                    "margin_count": 0,
                    "top1_available_rois": 0,
                    "top2_available_rois": 0,
                    "top1_normalized_count": 0.0,
                    "top2_normalized_count": 0.0,
                    "margin_normalized_count": 0.0,
                    "normalized_count_by_network": {},
                    "n_top": 0,
                    "network_column": str(column),
                    "assignment_metric": "top20_count_normalized_by_network_roi_count",
                }
            )
            continue
        ordered = pos_idx[np.argsort(-values[pos_idx])]
        top_idx = ordered[: int(min(int(top_n_cap), int(ordered.size)))]
        count_by_network: dict[str, int] = {}
        score_sum_by_network: dict[str, float] = {}
        for col in top_idx:
            net = roi_networks[int(col)]
            count_by_network[net] = int(count_by_network.get(net, 0) + 1)
            score_sum_by_network[net] = float(score_sum_by_network.get(net, 0.0) + float(values[int(col)]))
        ranked: list[tuple[float, float, int, str]] = []
        for net, count in count_by_network.items():
            available = int(available_by_network.get(net, 0))
            normalized = float(count) / float(available) if available > 0 else 0.0
            ranked.append((normalized, float(score_sum_by_network.get(net, 0.0)), int(count), str(net)))
        ranked.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
        normalized_count_by_network = {str(item[3]): float(item[0]) for item in ranked}
        top1_norm, top1_score, top1_count, top1_network = ranked[0]
        if len(ranked) > 1:
            top2_norm, top2_score, top2_count, top2_network = ranked[1]
        else:
            top2_norm, top2_score, top2_count, top2_network = 0.0, 0.0, 0, "none"
        top1_avail = int(available_by_network.get(str(top1_network), 0)) if str(top1_network) != "none" else 0
        top2_avail = int(available_by_network.get(str(top2_network), 0)) if str(top2_network) != "none" else 0
        rows.append(
            {
                "vertex_index": int(global_vidx),
                "status": "dominant",
                "top1_network": str(top1_network),
                "top2_network": str(top2_network),
                "top1_count": int(top1_count),
                "top2_count": int(top2_count),
                "margin_count": int(top1_count - top2_count),
                "top1_available_rois": int(top1_avail),
                "top2_available_rois": int(top2_avail),
                "top1_normalized_count": float(top1_norm),
                "top2_normalized_count": float(top2_norm),
                "margin_normalized_count": float(top1_norm - top2_norm),
                "normalized_count_by_network": normalized_count_by_network,
                "top1_score_sum": float(top1_score),
                "top2_score_sum": float(top2_score),
                "n_top": int(top_idx.size),
                "top1_fraction": float(top1_count / top_idx.size) if top_idx.size else 0.0,
                "top2_fraction": float(top2_count / top_idx.size) if top_idx.size else 0.0,
                "network_column": str(column),
                "assignment_metric": "top20_count_normalized_by_network_roi_count",
            }
        )
    return rows


def verify_canonical_network_order_consistency(
    reference_root: Path,
    subjects: list[str],
    atlas: str,
    network_taxonomy: str = "canonical7",
) -> list[str]:
    ref_subject = subjects[0]
    ref_names, _ref_timeseries = load_kong2022_network_reference(
        reference_root, ref_subject, atlas, network_taxonomy
    )
    mismatched: list[str] = []
    for subject in subjects[1:]:
        names, _timeseries = load_kong2022_network_reference(
            reference_root, subject, atlas, network_taxonomy
        )
        if names != ref_names:
            mismatched.append(f"sub-{subject}")
    if mismatched:
        raise RuntimeError(
            "Network ordering mismatch across subjects "
            f"for atlas={atlas} taxonomy={network_taxonomy}; template=sub-{ref_subject}; mismatched={mismatched[:10]}"
        )
    return ref_names


def load_surface(path: Path) -> tuple[np.ndarray, np.ndarray]:
    img = nib.load(str(path))
    coords = np.asarray(img.agg_data("pointset"), dtype=np.float32)
    faces = np.asarray(img.agg_data("triangle"), dtype=np.int32)
    return coords, faces


def build_surface_adjacency(faces: np.ndarray, n_vertices: int) -> sparse.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        rows.extend([a, b, b, c, c, a])
        cols.extend([b, a, c, b, a, c])
    data = np.ones(len(rows), dtype=np.float32)
    g = sparse.csr_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices), dtype=np.float32)
    g = g.maximum(g.T)
    g.setdiag(0.0)
    g.eliminate_zeros()
    return g


def connected_components_per_label(labels: np.ndarray, adjacency: sparse.csr_matrix) -> dict[int, int]:
    out: dict[int, int] = {}
    for k in sorted(int(x) for x in np.unique(labels) if int(x) > 0):
        m = labels == k
        if int(np.sum(m)) == 0:
            out[k] = 0
            continue
        sub = adjacency[m][:, m]
        n_comp, _ = sparse.csgraph.connected_components(sub, directed=False, return_labels=True)
        out[k] = int(n_comp)
    return out


def cluster_profile_coherence(features: np.ndarray, labels: np.ndarray, k: int) -> tuple[float, float]:
    # Mean cosine similarity between vertex profiles and cluster centroid.
    x = np.asarray(features, dtype=np.float32)
    out: list[float] = []
    for c in range(1, int(k) + 1):
        m = labels == c
        n = int(np.sum(m))
        if n == 0:
            out.append(0.0)
            continue
        if n == 1:
            out.append(1.0)
            continue
        xc = x[m, :]
        ctr = np.mean(xc, axis=0, keepdims=True)
        ctr_norm = np.linalg.norm(ctr, axis=1, keepdims=True)
        xc_norm = np.linalg.norm(xc, axis=1, keepdims=True)
        sims = (xc @ ctr.T) / np.clip(xc_norm * ctr_norm, 1e-12, None)
        sims = np.clip(sims.ravel(), -1.0, 1.0)
        out.append(float(np.mean(sims)))
    if not out:
        return 0.0, 0.0
    return float(min(out)), float(np.mean(out))


def build_hungarian_mapping(ref_labels: np.ndarray, tgt_labels: np.ndarray, k: int) -> dict[int, int]:
    contingency = np.zeros((k, k), dtype=np.int64)
    for a, b in zip(ref_labels, tgt_labels, strict=False):
        if 1 <= int(a) <= k and 1 <= int(b) <= k:
            contingency[int(a) - 1, int(b) - 1] += 1
    if int(contingency.sum()) == 0:
        return {i: i for i in range(1, k + 1)}
    cost = int(contingency.max()) - contingency
    ridx, cidx = linear_sum_assignment(cost)
    mapping: dict[int, int] = {int(c + 1): int(r + 1) for r, c in zip(ridx, cidx, strict=True)}
    for src in range(1, k + 1):
        mapping.setdefault(src, src)
    return mapping


def remap_labels(labels: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = np.zeros_like(labels, dtype=np.int32)
    for src, dst in mapping.items():
        out[labels == int(src)] = int(dst)
    return out


def local_maxima_flags(values: list[float]) -> list[bool]:
    if not values:
        return []
    finite_values = [float(x) for x in values if np.isfinite(float(x))]
    if not finite_values:
        return [False for _ in values]
    all_equal = len(set(finite_values)) == 1 and len(finite_values) == len(values)
    flags: list[bool] = []
    for i, raw_v in enumerate(values):
        v = float(raw_v)
        if not np.isfinite(v):
            flags.append(False)
            continue
        left = float(values[i - 1]) if i > 0 and np.isfinite(float(values[i - 1])) else -math.inf
        right = float(values[i + 1]) if i < len(values) - 1 and np.isfinite(float(values[i + 1])) else -math.inf
        is_local = all_equal or (v >= left and v >= right and (v > left or v > right or len(values) == 1))
        flags.append(bool(is_local))
    return flags


K_FINAL_RULES = ("smallest-stable", "largest-admissible")


def select_k_final(candidate_pool: list[dict], rule: str) -> tuple[int, str, set[int], int]:
    """Automatic K from the gate-passing candidates under the requested rule.

    Both rules return the split-half audit values of the default rule, so the
    summary table reports stability whether or not it decided K.
    """
    if rule not in K_FINAL_RULES:
        raise ValueError(f"Unsupported K rule {rule!r}; expected one of {K_FINAL_RULES}")
    k_final, decision_path, within_1se_k_set, argmax_k = select_k_final_after_redundancy_filter(candidate_pool)
    if rule == "largest-admissible":
        k_final = int(max(int(row["k"]) for row in candidate_pool))
        decision_path = "largest_k_passing_gates"
    return k_final, decision_path, within_1se_k_set, argmax_k


def select_k_final_after_redundancy_filter(candidate_pool: list[dict]) -> tuple[int, str, set[int], int]:
    """Select final K after removing redundant candidates."""
    ordered = sorted(candidate_pool, key=lambda r: int(r["k"]))
    means = [float(r["group_ari_mean"]) for r in ordered]
    ses = [float(r["group_ari_se"]) for r in ordered]
    local_flags = local_maxima_flags(means)
    finite_indices = [i for i, val in enumerate(means) if np.isfinite(float(val))]
    if not finite_indices:
        return (
            int(min(int(row["k"]) for row in ordered)),
            "no_valid_ari_smallest_after_redundancy_filter",
            set(),
            int(min(int(row["k"]) for row in ordered)),
        )
    best_idx = int(finite_indices[int(np.argmax(np.asarray([means[i] for i in finite_indices], dtype=np.float64)))])
    cutoff = float(means[best_idx] - ses[best_idx])
    within_1se = [
        row
        for row in ordered
        if np.isfinite(float(row["group_ari_mean"])) and float(row["group_ari_mean"]) >= cutoff - 1e-12
    ]
    within_1se_k_set = {int(row["k"]) for row in within_1se}
    local_max_in_1se = [
        row
        for row, is_local in zip(ordered, local_flags, strict=True)
        if is_local and int(row["k"]) in within_1se_k_set
    ]
    if local_max_in_1se:
        pool, path = local_max_in_1se, "local_max_within_1se_after_redundancy_filter"
    elif within_1se:
        pool, path = within_1se, "within_1se_smallest_after_redundancy_filter"
    else:
        pool, path = ordered, "smallest_among_nonredundant_fallback"
    return (
        int(min(int(row["k"]) for row in pool)),
        path,
        within_1se_k_set,
        int(ordered[best_idx]["k"]),
    )


def resolve_fixed_cluster_source_run(
    *,
    source_root: Path,
    method: str,
    atlas: str,
    smoothing: str,
    hemi: str,
    coverage_tag: str,
    gradient_components: tuple[int, ...],
) -> Path:
    """Resolve one explicitly selected fixed-cluster source run.

    The caller selects either a run directory, an atlas directory
    (``.../gradient-spectral/kong2022``), or a group-atlas root.  No directory
    enumeration or existence-order selection is allowed.
    """
    source = Path(source_root).resolve()
    comp_tag = gradient_comp_tag(gradient_components)
    if (source / "group_atlas_manifest.json").exists():
        run_dir = source
    elif source.name in {str(atlas), str(method)}:
        run_dir = source / smoothing / str(coverage_tag) / comp_tag / f"hemi_{hemi}"
    else:
        run_dir = source / str(method) / str(atlas) / smoothing / str(coverage_tag) / comp_tag / f"hemi_{hemi}"
    manifest_path = run_dir / "group_atlas_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "Explicit fixed-cluster source has no group_atlas_manifest.json: "
            f"{manifest_path}; pass the exact run, atlas, or group-atlas root"
        )
    manifest = load_json(manifest_path)
    expected = {
        "status": "ok",
        "method": str(method),
        "atlas": str(atlas),
        "smoothing": str(smoothing),
        "hemi": str(hemi),
    }
    mismatches = [f"{key}={manifest.get(key)!r} expected={value!r}" for key, value in expected.items() if manifest.get(key) != value]
    if mismatches:
        raise ValueError(f"Fixed-cluster source manifest mismatch: {manifest_path}; " + "; ".join(mismatches))
    if not (run_dir / "group_labels_final.npy").exists():
        raise FileNotFoundError(f"Fixed-cluster source is missing group_labels_final.npy: {run_dir}")
    return run_dir.resolve()


def _load_kong_gradient_identity(
    gradient_summary: dict[str, object],
    *,
    role: str,
) -> dict[str, object]:
    """Load the immutable 400-ROI gradient/FC identity for reuse validation."""
    manifest_raw = gradient_summary.get("hippomaps_gradient_manifest")
    selected_raw = gradient_summary.get("hippomaps_gradients_full")
    if not manifest_raw or not selected_raw:
        raise ValueError(
            f"Kong {role} gradient summary must declare hippomaps_gradient_manifest "
            f"and hippomaps_gradients_full"
        )
    manifest_path = Path(str(manifest_raw)).resolve()
    selected_path = Path(str(selected_raw)).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Kong {role} gradient manifest is missing: {manifest_path}")
    if not selected_path.exists():
        raise FileNotFoundError(f"Kong {role} selected gradient array is missing: {selected_path}")
    manifest = load_json(manifest_path)
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError(f"Kong {role} gradient manifest lacks outputs: {manifest_path}")
    manifest_gradient_raw = outputs.get("hippomaps_gradients_full")
    fc_raw = outputs.get("group_mean_fisherz_fc_active")
    if not manifest_gradient_raw or not fc_raw:
        raise ValueError(
            f"Kong {role} gradient manifest must declare selected gradient and 400-ROI FC outputs: "
            f"{manifest_path}"
        )
    manifest_gradient_path = Path(str(manifest_gradient_raw)).resolve()
    fc_path = Path(str(fc_raw)).resolve()
    if selected_path != manifest_gradient_path:
        raise ValueError(
            f"Kong {role} selected gradient path disagrees with its manifest: "
            f"summary={selected_path} manifest={manifest_gradient_path}"
        )
    if not manifest_gradient_path.exists():
        raise FileNotFoundError(f"Kong {role} manifest-selected gradient array is missing: {manifest_gradient_path}")
    if not fc_path.exists():
        raise FileNotFoundError(f"Kong {role} 400-ROI FC output is missing: {fc_path}")
    required_fields = (
        "route", "smoothing", "hemi", "feature_taxonomy", "feature_family",
        "feature_count", "group_fc_shape", "gradient_shape_full", "gradientmaps_kernel",
    )
    missing = [key for key in required_fields if key not in manifest]
    if missing:
        raise ValueError(f"Kong {role} gradient identity is incomplete: {manifest_path} missing={missing}")
    if str(manifest.get("feature_taxonomy")) != "kong2022_roi400":
        raise ValueError(
            f"Kong {role} gradient does not declare the 400-ROI taxonomy: "
            f"{manifest_path} feature_taxonomy={manifest.get('feature_taxonomy')!r}"
        )
    if int(manifest.get("feature_count", -1)) != KONG2022_ROI_COUNT:
        raise ValueError(
            f"Kong {role} gradient is not 400-ROI: {manifest_path} "
            f"feature_count={manifest.get('feature_count')!r}"
        )
    group_fc_shape = manifest.get("group_fc_shape")
    if not isinstance(group_fc_shape, list) or len(group_fc_shape) != 2 or int(group_fc_shape[1]) != KONG2022_ROI_COUNT:
        raise ValueError(
            f"Kong {role} gradient does not prove a 400-ROI FC branch: "
            f"{manifest_path} group_fc_shape={group_fc_shape!r}"
        )
    preprocessing = gradient_summary.get("gradient_preprocessing")
    required_preprocessing = set(gradient_preprocessing_settings(KONG2022_ROI_COUNT))
    if not isinstance(preprocessing, dict) or required_preprocessing - preprocessing.keys():
        raise ValueError(
            f"Kong {role} gradient summary lacks complete gradient_preprocessing for "
            f"{manifest_path}; explicitly regenerate"
        )
    validate_gradient_preprocessing(manifest, settings=preprocessing, manifest_path=manifest_path)
    gradient = np.asarray(np.load(selected_path), dtype=np.float32)
    fc = np.asarray(np.load(fc_path, mmap_mode="r"), dtype=np.float32)
    expected_gradient_shape = [int(x) for x in manifest["gradient_shape_full"]]
    if gradient.ndim != 2 or [int(x) for x in gradient.shape] != expected_gradient_shape:
        raise ValueError(
            f"Kong {role} selected gradient shape mismatch: {selected_path} "
            f"shape={gradient.shape} manifest={expected_gradient_shape}"
        )
    if fc.ndim != 2 or int(fc.shape[1]) != KONG2022_ROI_COUNT:
        raise ValueError(f"Kong {role} FC output is not (vertices,400): {fc_path} shape={fc.shape}")
    gradient_components = gradient_summary.get("gradient_components")
    selected_components = gradient_summary.get("selected_components")
    if not isinstance(gradient_components, list) or not isinstance(selected_components, list):
        raise ValueError(
            f"Kong {role} gradient summary must declare gradient_components and selected_components"
        )
    identity_fields = {
        key: manifest[key]
        for key in required_fields
    }
    identity_fields["gradient_components"] = gradient_components
    identity_fields["selected_components"] = selected_components
    identity_fields["gradient_preprocessing"] = preprocessing
    identity_fields["clustering_feature_transform"] = gradient_summary.get("clustering_feature_transform")
    align = manifest.get("gradient_alignment")
    anchor_smoothing = align.get("anchor_smoothing") if isinstance(align, dict) else None
    identity_fields["anchor_smoothing"] = anchor_smoothing
    identity_fields["gradient_manifest"] = str(manifest_path)
    identity_fields["selected_gradient_sha256"] = _sha256_file(selected_path)
    identity_fields["fc_sha256"] = _sha256_file(fc_path)
    identity_fields["fc_path"] = str(fc_path)
    identity_fields["gradient_shape_actual"] = [int(x) for x in gradient.shape]
    return identity_fields


def _extract_clustering_scaling(summary: dict[str, object] | None) -> str | None:
    if not isinstance(summary, dict):
        return None
    transform = summary.get("clustering_feature_transform")
    if isinstance(transform, dict) and "scaling" in transform:
        return str(transform["scaling"])
    if "clustering_feature_scaling" in summary and summary["clustering_feature_scaling"] is not None:
        return str(summary["clustering_feature_scaling"])
    return None


def load_fixed_cluster_reuse(
    *,
    source_run: Path,
    target_method: str,
    target_atlas: str,
    smoothing: str,
    hemi: str,
    coverage_threshold: float,
    support_mask: np.ndarray,
    n_vertices: int,
    k_values: list[int],
    force_final_k: int | None,
    default_k: int | None,
    k_selection_fields: list[str],
    target_subjects: list[str] | None = None,
    target_surface_path: Path | None = None,
    target_gradient_feature_summary: dict[str, object] | None = None,
    target_clustering_feature_scaling: str | None = None,
    target_rbf_scale: float | None = None,
    target_gradient_features_active: np.ndarray | None = None,
) -> tuple[int, np.ndarray, dict[str, object], dict[str, object]]:
    """Load fixed labels and source K evidence for taxonomy-only reuse."""
    source_run = Path(source_run).resolve()
    source_manifest_path = source_run / "group_atlas_manifest.json"
    if target_atlas == "kong2022" and is_gradient_method(target_method):
        validate_group_atlas_lineage(source_manifest_path)
    source_manifest = load_json(source_manifest_path)
    if source_manifest.get("status") != "ok":
        raise ValueError(f"Fixed-cluster source is not successful: {source_manifest_path}")
    if source_manifest.get("method") != str(target_method) or source_manifest.get("atlas") != str(target_atlas):
        raise ValueError(
            f"Fixed-cluster source method/atlas mismatch: {source_manifest_path} "
            f"method={source_manifest.get('method')!r} atlas={source_manifest.get('atlas')!r} "
            f"expected={target_method!r}/{target_atlas!r}"
        )
    if str(target_method) == "gradient-spectral-rbf" or target_rbf_scale is not None:
        expected_rbf_scale = 1.0 if target_rbf_scale is None else float(target_rbf_scale)
        source_rbf_scale = source_manifest.get("rbf_scale")
        if str(target_method) == "gradient-spectral-rbf" and (
            source_rbf_scale is None or float(source_rbf_scale) != expected_rbf_scale
        ):
            raise ValueError(
                f"Fixed-cluster source rbf_scale mismatch: {source_manifest_path} "
                f"source={source_rbf_scale!r} expected={expected_rbf_scale!r}; explicitly regenerate"
            )
    if target_subjects is not None:
        source_subjects = [str(x) for x in source_manifest.get("subjects", [])]
        expected_subjects = [f"sub-{str(x).removeprefix('sub-')}" for x in target_subjects]
        if source_subjects != expected_subjects:
            raise ValueError(
                f"Fixed-cluster source cohort mismatch: {source_manifest_path} "
                f"source={source_subjects} target={expected_subjects}"
            )
    if target_surface_path is not None:
        density = source_manifest.get("density_evidence", {})
        source_surface = density.get("surface_path")
        if source_surface is None or Path(str(source_surface)).resolve() != Path(target_surface_path).resolve():
            raise ValueError(
                f"Fixed-cluster source surface mismatch: {source_manifest_path} "
                f"source={source_surface!r} target={str(Path(target_surface_path).resolve())!r}"
            )
    source_gradient_summary = source_manifest.get("group_gradient_feature_summary")
    if is_gradient_method(target_method):
        source_summary_obj = (
            load_json(Path(str(source_gradient_summary)))
            if source_gradient_summary is not None and Path(str(source_gradient_summary)).exists()
            else None
        )
        source_scaling = _extract_clustering_scaling(source_summary_obj)
        target_scaling = (
            str(target_clustering_feature_scaling)
            if target_clustering_feature_scaling is not None
            else _extract_clustering_scaling(target_gradient_feature_summary)
        )
        if (source_scaling is not None or target_scaling is not None) and source_scaling != target_scaling:
            raise ValueError(
                f"Fixed-cluster source clustering_feature_scaling mismatch: {source_manifest_path} "
                f"source={source_scaling!r} target={target_scaling!r}; explicitly regenerate"
            )
        if (
            isinstance(source_summary_obj, dict)
            and isinstance(target_gradient_feature_summary, dict)
            and (
                "clustering_feature_transform" in source_summary_obj
                or "clustering_feature_transform" in target_gradient_feature_summary
            )
            and source_summary_obj.get("clustering_feature_transform")
            != target_gradient_feature_summary.get("clustering_feature_transform")
        ):
            raise ValueError(
                f"Fixed-cluster source clustering_feature_transform mismatch: {source_manifest_path} "
                f"source={source_summary_obj.get('clustering_feature_transform')!r} "
                f"target={target_gradient_feature_summary.get('clustering_feature_transform')!r}; "
                "explicitly regenerate"
            )
        if target_gradient_features_active is not None and (source_run / "group_gradient_features_active.npy").exists():
            source_active = np.asarray(np.load(source_run / "group_gradient_features_active.npy"), dtype=np.float32)
            target_active = np.asarray(target_gradient_features_active, dtype=np.float32)
            if source_active.shape != target_active.shape or not np.allclose(
                source_active, target_active, equal_nan=True, atol=1e-6, rtol=1e-6
            ):
                raise ValueError(
                    "Fixed-cluster source active gradient features mismatch: "
                    f"{source_run / 'group_gradient_features_active.npy'}; explicitly regenerate"
                )
        source_align = source_summary_obj.get("gradient_alignment") if isinstance(source_summary_obj, dict) else None
        target_align = target_gradient_feature_summary.get("gradient_alignment") if isinstance(target_gradient_feature_summary, dict) else None
        source_anchor = (
            source_align.get("anchor_smoothing")
            if isinstance(source_align, dict)
            else (source_summary_obj.get("alignment_anchor_smoothing") if isinstance(source_summary_obj, dict) else None)
        )
        target_anchor = (
            target_align.get("anchor_smoothing")
            if isinstance(target_align, dict)
            else (target_gradient_feature_summary.get("alignment_anchor_smoothing") if isinstance(target_gradient_feature_summary, dict) else None)
        )
        if target_anchor == "2mm":
            if source_anchor != "2mm":
                raise ValueError(
                    f"Fixed-cluster source alignment anchor mismatch: {source_manifest_path} "
                    f"source={source_anchor!r} target={target_anchor!r}; explicitly regenerate"
                )
        elif source_anchor is not None and target_anchor is not None and source_anchor != target_anchor:
            raise ValueError(
                f"Fixed-cluster source alignment anchor mismatch: {source_manifest_path} "
                f"source={source_anchor!r} target={target_anchor!r}; explicitly regenerate"
            )
    if is_gradient_method(target_method) and target_atlas == "kong2022":
        if target_gradient_feature_summary is None:
            raise ValueError(
                "Kong fixed-cluster reuse requires the target gradient feature summary "
                "to validate source-target gradient identity"
            )
        if source_gradient_summary is None or not Path(str(source_gradient_summary)).exists():
            raise FileNotFoundError(
                f"Kong fixed-cluster source lacks gradient feature summary: {source_gradient_summary!r}"
            )
        source_gradient_identity = _load_kong_gradient_identity(
            load_json(Path(str(source_gradient_summary))), role="source"
        )
        target_gradient_identity = _load_kong_gradient_identity(
            target_gradient_feature_summary, role="target"
        )
        identity_fields = (
            "route", "smoothing", "hemi", "feature_taxonomy", "feature_family",
            "feature_count", "group_fc_shape", "gradient_shape_full", "gradient_components",
            "selected_components", "gradientmaps_kernel", "anchor_smoothing",
            "gradient_preprocessing", "clustering_feature_transform",
        )
        mismatches = [
            f"{key}: source={source_gradient_identity[key]!r} target={target_gradient_identity[key]!r}"
            for key in identity_fields
            if source_gradient_identity[key] != target_gradient_identity[key]
        ]
        if mismatches:
            raise ValueError(
                "Kong fixed-cluster gradient identity mismatch; taxonomy-only reuse requires the same "
                "400-ROI FC branch and gradient provenance: " + "; ".join(mismatches)
            )
        if source_gradient_identity["selected_gradient_sha256"] != target_gradient_identity["selected_gradient_sha256"]:
            raise ValueError(
                "Kong fixed-cluster selected gradient array mismatch; taxonomy-only reuse cannot copy "
                "labels across different gradient arrays"
            )
        if source_gradient_identity["fc_sha256"] != target_gradient_identity["fc_sha256"]:
            raise ValueError(
                "Kong fixed-cluster 400-ROI FC branch mismatch; taxonomy-only reuse cannot copy labels "
                "across different FC inputs"
            )
    labels = np.asarray(np.load(source_run / "group_labels_final.npy"), dtype=np.int32).reshape(-1)
    if labels.size != int(n_vertices):
        raise ValueError(
            f"Fixed-cluster source vertex count mismatch: {source_run} labels={labels.size} expected={n_vertices}"
        )
    source_support_path = source_run / "group_support_mask.npy"
    if not source_support_path.exists():
        raise FileNotFoundError(f"Fixed-cluster source is missing support mask: {source_support_path}")
    source_support = np.asarray(np.load(source_support_path), dtype=bool).reshape(-1)
    target_support = np.asarray(support_mask, dtype=bool).reshape(-1)
    if source_support.shape != target_support.shape or not np.array_equal(source_support, target_support):
        raise ValueError(
            f"Fixed-cluster source support mask differs from target: {source_run} "
            f"source_shape={source_support.shape} target_shape={target_support.shape} "
            f"same={np.array_equal(source_support, target_support)}"
        )
    k_final = int(source_manifest.get("chosen_k_final", int(np.max(labels))))
    if k_final not in [int(k) for k in k_values]:
        raise ValueError(
            f"Fixed-cluster source K is outside requested target K range: source={k_final} requested={k_values}"
        )
    if force_final_k is not None and int(force_final_k) != k_final:
        raise ValueError(f"--force-final-k={force_final_k} disagrees with fixed-cluster source K={k_final}")
    if default_k is not None and int(default_k) != k_final:
        raise ValueError(f"--default-k={default_k} disagrees with fixed-cluster source K={k_final}")
    if int(np.min(labels[target_support])) < 1 or int(np.max(labels[target_support])) > k_final:
        raise ValueError(f"Fixed-cluster source labels are invalid for K={k_final}: {source_run}")

    summary_path = source_run / "k_selection_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Fixed-cluster source is missing K diagnostics: {summary_path}")
    summary = load_json(summary_path)
    rows = [dict(row) for row in summary.get("rows", []) if int(row.get("k", -1)) == k_final]
    if len(rows) != 1:
        raise ValueError(
            f"Fixed-cluster source must preserve exactly one K diagnostic row: {summary_path} "
            f"k={k_final} rows={len(rows)}"
        )
    row = rows[0]
    # Preserve every diagnostic value from the source. Only add target identity
    # and reuse provenance; no fabricated zeros or success flags are allowed.
    row.update(
        {
            "method": str(target_method),
            "atlas": str(target_atlas),
            "smoothing": str(smoothing),
            "hemi": str(hemi),
            "coverage_threshold": float(coverage_threshold),
            "k": int(k_final),
            "n_support_vertices": int(np.sum(target_support)),
            "k_final_decision_path": "reused_fixed_cluster_source",
            "reused_fixed_cluster_source": str(source_run),
        }
    )
    return k_final, labels, row, {
        "source_run": str(source_run),
        "source_manifest": str(source_manifest_path),
        "source_atlas": source_manifest.get("atlas"),
        "source_network_taxonomy": source_manifest.get("network_taxonomy"),
        "source_chosen_k_final": int(k_final),
        "gradient_recomputed": False,
        "clustering_recomputed": False,
        "k_selection_recomputed": False,
    }


def family17_reused_k_diagnostics(source_run: Path, k_final: int) -> tuple[list[dict], dict]:
    """Preserve the canonical7 source's full K diagnostics for family17."""
    path = Path(source_run) / "k_selection_summary.json"
    summary = load_json(path)
    if int(summary["chosen_k_final"]) != int(k_final):
        raise ValueError(f"Reused K diagnostics disagree with source labels: {path}")
    rows = [dict(row) for row in summary["rows"]]
    if sum(int(row["selected_final"]) for row in rows) != 1:
        raise ValueError(f"Reused K diagnostics must contain one selected K: {path}")
    primary = {key: summary[key] for key in ("chosen_k_final", "automatic_k_final", "primary_k_source")}
    return rows, primary


def load_cluster_colors() -> list[tuple[int, int, int, int]]:
    return load_cluster_palette(DEFAULT_NETWORK_STYLE)


def load_canonical_network_color_map() -> dict[str, tuple[int, int, int, int]]:
    return load_network_colors(DEFAULT_NETWORK_STYLE)


def infer_surface_path(subject: str, hemi: str, batch_root: Path | None = None) -> Path:
    root = Path(batch_root).resolve() if batch_root is not None else DEFAULT_HIPPUNFOLD_BATCH_ROOT.resolve()
    p = (
        root
        / f"sub-{subject}"
        / "hippunfold"
        / f"sub-{subject}"
        / "surf"
        / f"sub-{subject}_hemi-{hemi}_space-corobl_den-512_label-hipp_midthickness.surf.gii"
    )
    if not p.exists():
        raise FileNotFoundError(f"Missing midthickness surface: {p}")
    return p


def subject_valid_mask(surface_root: Path, shared_root: Path, subject: str, smoothing: str, hemi: str) -> np.ndarray:
    p = surface_root / f"sub-{subject}" / "surface" / "validity" / f"sub-{subject}_hemi-{hemi}_valid_mask.npy"
    if not p.exists():
        raise FileNotFoundError(f"Missing hippocampal validity mask: {p}; prepare the subject's validity inputs first")
    return np.load(p).astype(bool)


def subject_network_feature(
    shared_root: Path,
    subject: str,
    atlas: str,
    smoothing: str,
    hemi: str,
    method: str,
    reference_root: Path | None = None,
) -> np.ndarray:
    p = shared_root / f"sub-{subject}" / "fc" / atlas / smoothing / f"hemi_{hemi}" / network_feature_filename(method, atlas)
    if not p.exists():
        raise FileNotFoundError(f"Missing network FC cache: {p}")
    arr = np.load(p).astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D network feature matrix: {p} shape={arr.shape}")
    if str(atlas) == "kong2022" and method in ({"group-roi-sim-spectral", "gradient-spectral", "wta-top"} | GRADIENT_METHODS | MULTIGRAPH_METHODS):
        reference_dir = None
        if reference_root is not None:
            reference_dir = Path(reference_root) / f"sub-{subject}" / "reference" / "kong2022"
        validate_kong2022_roi_fc(p, reference_dir=reference_dir)
        if int(arr.shape[1]) != 400:
            raise ValueError(f"Kong2022 {method} requires 400-ROI FC: {p} shape={arr.shape}")
    if str(atlas) == "kong2022" and method == "network-spectral" and int(arr.shape[1]) != 7:
        raise ValueError(f"Kong2022 network-spectral requires canonical7 summary FC: {p} shape={arr.shape}")
    arr = fisher_z_transform_fc(arr)
    return arr


def intrinsic_feature_candidates(
    shared_root: Path,
    subject: str,
    smoothing: str,
    hemi: str,
    run_id: int | None = None,
) -> list[Path]:
    file_rel = Path("hipp_vertex_to_vertex_fc_active.npy")
    run_rel = Path("runs") / f"run-{run_id}" / file_rel if run_id is not None else file_rel
    return [shared_root / f"sub-{subject}" / "fc" / "intrinsic" / smoothing / f"hemi_{hemi}" / run_rel]


def resolve_intrinsic_feature_path(
    shared_root: Path,
    subject: str,
    smoothing: str,
    hemi: str,
    run_id: int | None = None,
) -> Path:
    candidates = intrinsic_feature_candidates(shared_root, subject, smoothing, hemi, run_id)
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Missing intrinsic FC cache. Tried:\n" + "\n".join(str(p) for p in candidates)
    )


def subject_intrinsic_feature(shared_root: Path, subject: str, atlas: str, smoothing: str, hemi: str, valid_mask: np.ndarray) -> np.ndarray:
    p = resolve_intrinsic_feature_path(shared_root, subject, smoothing, hemi)
    arr = np.load(p).astype(np.float32)
    n_v = int(valid_mask.size)
    n_valid = int(np.sum(valid_mask))
    if arr.shape == (n_v, n_v):
        full = arr
    elif arr.shape == (n_valid, n_valid):
        full = np.full((n_v, n_v), np.nan, dtype=np.float32)
        idx = np.flatnonzero(valid_mask)
        full[np.ix_(idx, idx)] = arr
    else:
        raise ValueError(
            f"Unexpected intrinsic FC shape for sub-{subject} {atlas} {smoothing} hemi-{hemi}: {arr.shape}; "
            f"expected ({n_v},{n_v}) or ({n_valid},{n_valid})"
        )
    full = fisher_z_transform_fc(full)
    np.fill_diagonal(full, 0.0)
    return full


def verify_intrinsic_atlas_invariance(shared_root: Path, subjects: list[str], smoothing: str, hemi: str, atlases: list[str]) -> None:
    for subject in subjects:
        p_intrinsic = shared_root / f"sub-{subject}" / "fc" / "intrinsic" / smoothing / f"hemi_{hemi}" / "hipp_vertex_to_vertex_fc_active.npy"
        ref = np.load(p_intrinsic) if p_intrinsic.exists() else None
        ref_path = p_intrinsic if p_intrinsic.exists() else None
        for atlas in atlases:
            p_cmp = shared_root / f"sub-{subject}" / "fc" / atlas / smoothing / f"hemi_{hemi}" / "hipp_vertex_to_vertex_fc_active.npy"
            if not p_cmp.exists():
                continue
            cmp = np.load(p_cmp)
            if ref is None:
                ref = cmp
                ref_path = p_cmp
                continue
            if ref.shape != cmp.shape:
                raise RuntimeError(
                    f"Intrinsic cache atlas-invariance violation for sub-{subject} {smoothing} hemi-{hemi}: "
                    f"{ref_path} vs {p_cmp}"
                )
            if not np.allclose(ref, cmp, rtol=1e-6, atol=1e-7, equal_nan=True):
                max_abs = float(np.nanmax(np.abs(ref - cmp)))
                raise RuntimeError(
                    f"Intrinsic cache atlas-invariance violation for sub-{subject} {smoothing} hemi-{hemi}: "
                    f"{ref_path} vs {p_cmp}; max_abs_diff={max_abs:.3e}"
                )


def group_feature_matrix(
    *,
    method: str,
    feature_bank: dict[str, np.ndarray],
    subjects_subset: list[str],
    support_mask: np.ndarray,
    gradient_components: tuple[int, ...] = DEFAULT_GRADIENT_COMPONENTS,
    gradient_options: dict[str, object] | None = None,
) -> np.ndarray:
    idx = np.flatnonzero(support_mask)
    if idx.size == 0:
        raise RuntimeError("Support mask has zero vertices")

    if is_network_method(method):
        mats = []
        for s in subjects_subset:
            m = feature_bank[s][idx, :]
            mats.append(m)
        stack = np.stack(mats, axis=0).astype(np.float32)
        group = np.nanmean(stack, axis=0)
        group[~np.isfinite(group)] = 0.0
        return zscore_cols(group)

    if is_multigraph_method(method):
        return coordinate_stack(
            [feature_bank[s] for s in subjects_subset], subjects_subset, support_mask,
            method=method, n_components=MULTIGRAPH_MAX_COORDINATES,
        )

    if is_gradient_method(method):
        features, _meta = build_group_gradient_features(
            method=method,
            feature_bank=feature_bank,
            subjects_subset=subjects_subset,
            support_mask=support_mask,
            gradient_components=gradient_components,
            hippomaps_root=HIPPOMAPS_ROOT_OVERRIDE,
            gradient_options=gradient_options,
        )
        return features

    if method in {"group-roi-sim-spectral", "intrinsic-spectral"}:
        mats = []
        for s in subjects_subset:
            m = feature_bank[s][np.ix_(idx, idx)]
            mats.append(m)
        stack = np.stack(mats, axis=0).astype(np.float32)
        group = np.nanmean(stack, axis=0)
        group[~np.isfinite(group)] = 0.0
        np.fill_diagonal(group, 0.0)
        return group.astype(np.float32)

    raise ValueError(f"Unknown method: {method}")


def cluster_for_subset(
    *,
    method: str,
    feature_bank: dict[str, np.ndarray],
    subjects_subset: list[str],
    support_mask: np.ndarray,
    adjacency_full: sparse.csr_matrix,
    k: int,
    seed: int,
    gradient_components: tuple[int, ...] = DEFAULT_GRADIENT_COMPONENTS,
    gradient_options: dict[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    features = group_feature_matrix(
        method=method,
        feature_bank=feature_bank,
        subjects_subset=subjects_subset,
        support_mask=support_mask,
        gradient_components=gradient_components,
        gradient_options=gradient_options,
    )
    return cluster_from_precomputed_group_features(
        group_features=features,
        support_mask=support_mask,
        adjacency_full=adjacency_full,
        k=int(k),
        seed=int(seed),
        method=method,
        rbf_scale=float((gradient_options or {}).get("rbf_scale", 1.0)),
    )


DEFAULT_HALF_FIT_SEED_OFFSETS = (1001, 2001)


def half_fit_seeds(
    outer_seed: int,
    offsets: tuple[int, int] = DEFAULT_HALF_FIT_SEED_OFFSETS,
) -> tuple[int, int]:
    """Derive the two clustering seeds (GMM and spectral initialisation) for one repetition."""
    return int(outer_seed + offsets[0]), int(outer_seed + offsets[1])


@dataclass(frozen=True)
class RepetitionOutcome:
    repetition_index: int
    resample_type: str  # "split" or "null"
    seed: int
    k: int
    ari: float
    status: str  # "ok" or "failed"
    reason: str
    n_overlap: int = 0


@dataclass
class FittedHalf:
    half_index: int
    seed: int
    features: np.ndarray | None = None
    support: np.ndarray | None = None
    reason: str | None = None
    full: np.ndarray | None = None


def split_seed_for_index(index: int, base_seed: int = 20260420) -> int:
    return int(base_seed + 1 + 13 * index)


def null_seed_for_index(index: int, base_seed: int = 20260420) -> int:
    return int(base_seed + 5000 + 17 * index)


def compute_resample_statistics(records: Sequence[RepetitionOutcome], requested: int = 20) -> dict[str, object]:
    valid_vals = [r.ari for r in records if r.status == "ok" and math.isfinite(r.ari)]
    failed_count = sum(1 for r in records if r.status != "ok" or not math.isfinite(r.ari))
    if valid_vals:
        arr = np.asarray(valid_vals, dtype=np.float64)
        mean = float(np.mean(arr))
        se = float(np.std(arr, ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
    else:
        mean = math.nan
        se = math.nan
    return {
        "group_ari_mean": mean,
        "group_ari_se": se,
        "subject_split_requested": int(requested),
        "subject_split_used": int(len(valid_vals)),
        "subject_split_failed": int(failed_count),
    }


def compute_null_statistics(records: Sequence[RepetitionOutcome], requested: int = 20) -> dict[str, object]:
    valid_vals = [r.ari for r in records if r.status == "ok" and math.isfinite(r.ari)]
    failed_count = sum(1 for r in records if r.status != "ok" or not math.isfinite(r.ari))
    if valid_vals:
        arr = np.asarray(valid_vals, dtype=np.float64)
        mean = float(np.mean(arr))
        pct5 = float(np.percentile(arr, 5.0))
        pct95 = float(np.percentile(arr, 95.0))
    else:
        mean = math.nan
        pct5 = math.nan
        pct95 = math.nan
    return {
        "ARI_null_mean": mean,
        "ARI_null_5pct": pct5,
        "ARI_null_95pct": pct95,
        "B_null_requested": int(requested),
        "B_null_used_strong": int(len(valid_vals)),
        "null_failed": int(failed_count),
    }


def compute_half_group_fc(
    fc_dict: dict[str, np.ndarray],
    half_subjects: Sequence[str],
    support_mask: np.ndarray,
    *,
    valid_masks: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Average the already Fisher-z-transformed subject feature bank once."""
    active_idx = np.flatnonzero(np.asarray(support_mask, dtype=bool))
    if active_idx.size == 0:
        raise ValueError("Empty support mask for half group FC")
    mats = []
    for s in half_subjects:
        sub_fc = np.asarray(fc_dict[s][active_idx, :], dtype=np.float32)
        if valid_masks is not None and s in valid_masks:
            sub_valid = np.asarray(valid_masks[s], dtype=bool)[active_idx]
            if not np.all(sub_valid):
                sub_fc = sub_fc.copy()
                sub_fc[~sub_valid, :] = np.nan
        mats.append(sub_fc)
    group_fc = np.nanmean(np.stack(mats, axis=0).astype(np.float32), axis=0)
    group_fc[~np.isfinite(group_fc)] = 0.0
    return group_fc.astype(np.float32)


def fit_and_align_half_gradient(
    group_fc_active: np.ndarray,
    support_mask: np.ndarray,
    vertex_count: int,
    *,
    feature_scaling: str,
    sparsify_stage: str,
    sparsity: float,
    reference_full: np.ndarray | None = None,
    n_components: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit diffusion gradients for a half and optionally align to reference_full."""
    active_g, lambdas, ev = brainspace_diffusion_gradients(
        group_fc_active,
        n_components=n_components,
        hippomaps_root=HIPPOMAPS_ROOT_OVERRIDE,
        gradient_options={
            "feature_scaling": feature_scaling,
            "sparsify_stage": sparsify_stage,
            "sparsity": sparsity,
        },
    )
    full_g = np.full((int(vertex_count), int(n_components)), np.nan, dtype=np.float32)
    full_g[np.asarray(support_mask, dtype=bool)] = active_g
    if reference_full is not None:
        component_labels = [f"G{i+1}" for i in range(n_components)]
        active_g, full_g, _, _, _ = align_gradient_components_to_reference(
            active=active_g,
            full=full_g,
            lambdas=lambdas,
            ev_fractions=ev,
            reference_full=reference_full,
            component_labels=component_labels,
        )
    return active_g, full_g


def evaluate_half_pair_for_k(
    half_a: FittedHalf,
    half_b: FittedHalf,
    adjacency_full: sparse.csr_matrix,
    method: str,
    k: int,
    repetition_index: int,
    resample_type: str,
    seed: int,
) -> RepetitionOutcome:
    if half_a.reason:
        return RepetitionOutcome(
            repetition_index, resample_type, seed, k, math.nan, "failed", f"Half A: {half_a.reason}"
        )
    if half_b.reason:
        return RepetitionOutcome(
            repetition_index, resample_type, seed, k, math.nan, "failed", f"Half B: {half_b.reason}"
        )
    assert half_a.support is not None and half_b.support is not None
    assert half_a.features is not None and half_b.features is not None

    overlap = half_a.support & half_b.support
    if int(overlap.sum()) <= int(k):
        return RepetitionOutcome(
            repetition_index,
            resample_type,
            seed,
            k,
            math.nan,
            "failed",
            f"Overlap vertices ({int(overlap.sum())}) <= k ({k})",
            n_overlap=int(overlap.sum()),
        )

    try:
        _, full_a = cluster_from_precomputed_group_features(
            method=method,
            group_features=half_a.features,
            support_mask=half_a.support,
            adjacency_full=adjacency_full,
            k=int(k),
            seed=int(half_a.seed),
        )
        _, full_b = cluster_from_precomputed_group_features(
            method=method,
            group_features=half_b.features,
            support_mask=half_b.support,
            adjacency_full=adjacency_full,
            k=int(k),
            seed=int(half_b.seed),
        )
        ref = full_a[overlap]
        tgt = full_b[overlap]
        ari = float(adjusted_rand_score(ref, tgt))
        return RepetitionOutcome(
            repetition_index, resample_type, seed, k, ari, "ok", "", n_overlap=int(overlap.sum())
        )
    except Exception as exc:
        return RepetitionOutcome(
            repetition_index, resample_type, seed, k, math.nan, "failed", str(exc), n_overlap=int(overlap.sum())
        )


def _fit_single_half(
    fc_stack_or_mean: np.ndarray,
    support_mask: np.ndarray,
    vertex_count: int,
    half_index: int,
    seed: int,
    feature_scaling: str,
    sparsify_stage: str,
    sparsity: float,
    reference_full: np.ndarray | None,
    n_components: int = 5,
    selected_components: tuple[int, ...] = (1, 2, 3),
) -> FittedHalf:
    try:
        active_g, full_g = fit_and_align_half_gradient(
            fc_stack_or_mean,
            support_mask,
            vertex_count=vertex_count,
            feature_scaling=feature_scaling,
            sparsify_stage=sparsify_stage,
            sparsity=sparsity,
            reference_full=reference_full,
            n_components=n_components,
        )
        col_indices = [c - 1 for c in selected_components]
        features = active_g[:, col_indices].astype(np.float32)
        return FittedHalf(half_index=half_index, seed=seed, features=features, support=support_mask, full=full_g)
    except Exception as exc:
        return FittedHalf(half_index=half_index, seed=seed, reason=str(exc))


def _resolve_half_reference(
    reference_full: np.ndarray | None,
    reference_half: FittedHalf | None,
) -> tuple[np.ndarray | None, str | None]:
    if reference_half is not None:
        if reference_half.reason:
            return None, f"Left reference failed: {reference_half.reason}"
        return reference_half.full, None
    return reference_full, None


def _prepare_one_split_repetition(
    repetition_index: int,
    subjects: Sequence[str],
    feature_bank: dict[str, np.ndarray],
    valid_masks: dict[str, np.ndarray],
    tau: float,
    base_seed: int,
    feature_scaling: str,
    sparsify_stage: str,
    sparsity: float,
    reference_full: np.ndarray | None,
    n_components: int = 5,
    selected_components: tuple[int, ...] = (1, 2, 3),
    half_fit_seed_offsets: tuple[int, int] = DEFAULT_HALF_FIT_SEED_OFFSETS,
    reference_pair: tuple[FittedHalf, FittedHalf] | None = None,
) -> tuple[int, FittedHalf, FittedHalf]:
    seed = split_seed_for_index(repetition_index, base_seed)
    rng = np.random.RandomState(seed)
    perm = list(subjects)
    rng.shuffle(perm)
    half = len(perm) // 2
    sub_a = perm[:half]
    sub_b = perm[half:]

    cov_a = np.mean([valid_masks[s] for s in sub_a], axis=0)
    cov_b = np.mean([valid_masks[s] for s in sub_b], axis=0)
    mask_a = cov_a >= float(tau)
    mask_b = cov_b >= float(tau)

    seed_a, seed_b = half_fit_seeds(seed, half_fit_seed_offsets)
    ref_full_a, ref_err_a = _resolve_half_reference(reference_full, reference_pair[0] if reference_pair is not None else None)
    ref_full_b, ref_err_b = _resolve_half_reference(reference_full, reference_pair[1] if reference_pair is not None else None)

    if ref_err_a is not None:
        half_a = FittedHalf(half_index=0, seed=seed_a, reason=ref_err_a)
    elif int(mask_a.sum()) < 2:
        half_a = FittedHalf(half_index=0, seed=seed_a, reason=f"Mask A vertices ({int(mask_a.sum())}) < 2")
    else:
        fc_a = compute_half_group_fc(feature_bank, sub_a, mask_a, valid_masks=valid_masks)
        half_a = _fit_single_half(
            fc_a, mask_a, vertex_count=mask_a.size, half_index=0, seed=seed_a,
            feature_scaling=feature_scaling, sparsify_stage=sparsify_stage, sparsity=sparsity,
            reference_full=ref_full_a, n_components=n_components, selected_components=selected_components,
        )

    if ref_err_b is not None:
        half_b = FittedHalf(half_index=1, seed=seed_b, reason=ref_err_b)
    elif int(mask_b.sum()) < 2:
        half_b = FittedHalf(half_index=1, seed=seed_b, reason=f"Mask B vertices ({int(mask_b.sum())}) < 2")
    else:
        fc_b = compute_half_group_fc(feature_bank, sub_b, mask_b, valid_masks=valid_masks)
        half_b = _fit_single_half(
            fc_b, mask_b, vertex_count=mask_b.size, half_index=1, seed=seed_b,
            feature_scaling=feature_scaling, sparsify_stage=sparsify_stage, sparsity=sparsity,
            reference_full=ref_full_b, n_components=n_components, selected_components=selected_components,
        )

    return repetition_index, half_a, half_b


def _prepare_one_null_repetition(
    repetition_index: int,
    subjects: Sequence[str],
    feature_bank: dict[str, np.ndarray],
    valid_masks: dict[str, np.ndarray],
    tau: float,
    base_seed: int,
    feature_scaling: str,
    sparsify_stage: str,
    sparsity: float,
    reference_full: np.ndarray | None,
    n_components: int = 5,
    selected_components: tuple[int, ...] = (1, 2, 3),
    half_fit_seed_offsets: tuple[int, int] = DEFAULT_HALF_FIT_SEED_OFFSETS,
    reference_pair: tuple[FittedHalf, FittedHalf] | None = None,
) -> tuple[int, FittedHalf, FittedHalf]:
    seed = null_seed_for_index(repetition_index, base_seed)
    half = len(subjects) // 2
    null_fc_bank: dict[str, np.ndarray] = {}
    null_masks: dict[str, np.ndarray] = {}
    # One generator per repetition. Seeding each subject with `seed + index`
    # reuses permutations across repetitions whose seeds differ by less than
    # the subject count, so the null repetitions would not be independent.
    rng = np.random.RandomState(seed)
    for s in subjects:
        n_verts = feature_bank[s].shape[0]
        perm = rng.permutation(n_verts)
        null_fc_bank[s] = feature_bank[s][perm, :].astype(np.float32)
        null_masks[s] = valid_masks[s][perm]

    sub_a = list(subjects[:half])
    sub_b = list(subjects[half:])

    cov_a = np.mean([null_masks[s] for s in sub_a], axis=0)
    cov_b = np.mean([null_masks[s] for s in sub_b], axis=0)
    mask_a = cov_a >= float(tau)
    mask_b = cov_b >= float(tau)

    seed_a, seed_b = half_fit_seeds(seed, half_fit_seed_offsets)
    ref_full_a, ref_err_a = _resolve_half_reference(reference_full, reference_pair[0] if reference_pair is not None else None)
    ref_full_b, ref_err_b = _resolve_half_reference(reference_full, reference_pair[1] if reference_pair is not None else None)

    if ref_err_a is not None:
        half_a = FittedHalf(half_index=0, seed=seed_a, reason=ref_err_a)
    elif int(mask_a.sum()) < 2:
        half_a = FittedHalf(half_index=0, seed=seed_a, reason=f"Null mask A vertices ({int(mask_a.sum())}) < 2")
    else:
        fc_a = compute_half_group_fc(null_fc_bank, sub_a, mask_a, valid_masks=null_masks)
        half_a = _fit_single_half(
            fc_a, mask_a, vertex_count=mask_a.size, half_index=0, seed=seed_a,
            feature_scaling=feature_scaling, sparsify_stage=sparsify_stage, sparsity=sparsity,
            reference_full=ref_full_a, n_components=n_components, selected_components=selected_components,
        )

    if ref_err_b is not None:
        half_b = FittedHalf(half_index=1, seed=seed_b, reason=ref_err_b)
    elif int(mask_b.sum()) < 2:
        half_b = FittedHalf(half_index=1, seed=seed_b, reason=f"Null mask B vertices ({int(mask_b.sum())}) < 2")
    else:
        fc_b = compute_half_group_fc(null_fc_bank, sub_b, mask_b, valid_masks=null_masks)
        half_b = _fit_single_half(
            fc_b, mask_b, vertex_count=mask_b.size, half_index=1, seed=seed_b,
            feature_scaling=feature_scaling, sparsify_stage=sparsify_stage, sparsity=sparsity,
            reference_full=ref_full_b, n_components=n_components, selected_components=selected_components,
        )

    return repetition_index, half_a, half_b


def prepare_all_split_halves(
    subjects: Sequence[str],
    feature_bank: dict[str, np.ndarray],
    valid_masks: dict[str, np.ndarray],
    tau: float,
    *,
    feature_scaling: str,
    sparsify_stage: str,
    sparsity: float,
    reference_full: np.ndarray | None,
    n_repetitions: int = 20,
    base_seed: int = 20260420,
    max_workers: int = 8,
    selected_components: tuple[int, ...] = (1, 2, 3),
    half_fit_seed_offsets: tuple[int, int] = DEFAULT_HALF_FIT_SEED_OFFSETS,
    reference_halves: Sequence[tuple[FittedHalf, FittedHalf]] | None = None,
) -> list[tuple[FittedHalf, FittedHalf]]:
    """Compute split-half gradient fits across all repetitions."""
    results: list[tuple[int, FittedHalf, FittedHalf]] = []
    for i in range(n_repetitions):
        results.append(
            _prepare_one_split_repetition(
                i, subjects, feature_bank, valid_masks, tau, base_seed,
                feature_scaling, sparsify_stage, sparsity, reference_full,
                selected_components=selected_components,
                half_fit_seed_offsets=half_fit_seed_offsets,
                reference_pair=reference_halves[i] if reference_halves is not None else None,
            )
        )
    results.sort(key=lambda r: r[0])
    return [(r[1], r[2]) for r in results]


def prepare_all_null_halves(
    subjects: Sequence[str],
    feature_bank: dict[str, np.ndarray],
    valid_masks: dict[str, np.ndarray],
    tau: float,
    *,
    feature_scaling: str,
    sparsify_stage: str,
    sparsity: float,
    reference_full: np.ndarray | None,
    n_repetitions: int = 20,
    base_seed: int = 20260420,
    max_workers: int = 8,
    selected_components: tuple[int, ...] = (1, 2, 3),
    half_fit_seed_offsets: tuple[int, int] = DEFAULT_HALF_FIT_SEED_OFFSETS,
    reference_halves: Sequence[tuple[FittedHalf, FittedHalf]] | None = None,
) -> list[tuple[FittedHalf, FittedHalf]]:
    """Compute null-half gradient fits across all repetitions."""
    results: list[tuple[int, FittedHalf, FittedHalf]] = []
    for i in range(n_repetitions):
        results.append(
            _prepare_one_null_repetition(
                i, subjects, feature_bank, valid_masks, tau, base_seed,
                feature_scaling, sparsify_stage, sparsity, reference_full,
                selected_components=selected_components,
                half_fit_seed_offsets=half_fit_seed_offsets,
                reference_pair=reference_halves[i] if reference_halves is not None else None,
            )
        )
    results.sort(key=lambda r: r[0])
    return [(r[1], r[2]) for r in results]


def subject_split_ari(
    *,
    method: str,
    feature_bank: dict[str, np.ndarray],
    valid_masks_by_subject: dict[str, np.ndarray],
    subjects: list[str],
    tau: float,
    adjacency_full: sparse.csr_matrix,
    k: int,
    seeds: list[int],
    gradient_components: tuple[int, ...] = DEFAULT_GRADIENT_COMPONENTS,
    allow_diagnostic_failures: bool = False,
    gradient_options: dict[str, object] | None = None,
) -> dict[str, object]:
    vals: list[float] = []
    n_skipped_overlap = 0
    n_skipped_empty = 0
    n_skipped_failure = 0
    failure_examples: list[str] = []
    n = len(subjects)
    if n < 4:
        raise RuntimeError("Need at least four subjects for subject-split instability")
    half = n // 2
    for seed in seeds:
        rng = np.random.RandomState(int(seed))
        perm = np.asarray(subjects, dtype=object).copy()
        rng.shuffle(perm)
        sub_a = [str(x) for x in perm[:half]]
        sub_b = [str(x) for x in perm[half:]]

        cov_a = np.mean(np.stack([valid_masks_by_subject[s] for s in sub_a], axis=0), axis=0)
        cov_b = np.mean(np.stack([valid_masks_by_subject[s] for s in sub_b], axis=0), axis=0)
        mask_a = cov_a >= float(tau)
        mask_b = cov_b >= float(tau)
        overlap = mask_a & mask_b
        if int(np.sum(overlap)) <= int(k):
            n_skipped_overlap += 1
            continue

        try:
            _, full_a = cluster_for_subset(
                method=method,
                feature_bank=feature_bank,
                subjects_subset=sub_a,
                support_mask=mask_a,
                adjacency_full=adjacency_full,
                k=k,
                seed=seed + 1001,
                gradient_components=gradient_components,
                gradient_options=gradient_options,
            )
            _, full_b = cluster_for_subset(
                method=method,
                feature_bank=feature_bank,
                subjects_subset=sub_b,
                support_mask=mask_b,
                adjacency_full=adjacency_full,
                k=k,
                seed=seed + 2001,
                gradient_components=gradient_components,
                gradient_options=gradient_options,
            )
        except RuntimeError as exc:
            if not allow_diagnostic_failures:
                raise
            n_skipped_failure += 1
            if len(failure_examples) < MAX_DIAGNOSTIC_FAILURE_EXAMPLES:
                failure_examples.append(str(exc))
            continue
        ref = full_a[overlap]
        tgt = full_b[overlap]
        if ref.size == 0 or np.unique(ref).size < 2 or np.unique(tgt).size < 2:
            # A partition that puts every shared vertex in one parcel has no
            # boundary to reproduce; scikit-learn would score two of them as 1.
            n_skipped_empty += 1
            continue
        mapping = build_hungarian_mapping(ref, tgt, int(k))
        tgt_r = remap_labels(tgt, mapping)
        ari = float(adjusted_rand_score(ref, tgt_r))
        vals.append(float(ari))
    return {
        "values": vals,
        "n_requested": int(len(seeds)),
        "n_used": int(len(vals)),
        "n_skipped_overlap": int(n_skipped_overlap),
        "n_skipped_empty": int(n_skipped_empty),
        "n_skipped_failure": int(n_skipped_failure),
        "failure_examples": failure_examples,
    }


def strong_null_ari(
    *,
    method: str,
    feature_bank: dict[str, np.ndarray],
    valid_masks_by_subject: dict[str, np.ndarray],
    subjects: list[str],
    tau: float,
    adjacency_full: sparse.csr_matrix,
    k: int,
    seeds: list[int],
    gradient_components: tuple[int, ...] = DEFAULT_GRADIENT_COMPONENTS,
    allow_diagnostic_failures: bool = False,
    gradient_options: dict[str, object] | None = None,
) -> dict[str, object]:
    if not seeds:
        return {
            "values": [],
            "n_requested": 0,
            "n_used": 0,
            "n_skipped_overlap": 0,
            "n_skipped_empty": 0,
            "n_skipped_failure": 0,
            "failure_examples": [],
            "permutation_scope": "subject_specific",
        }
    masks = {s: np.asarray(valid_masks_by_subject[s], dtype=bool).reshape(-1) for s in subjects}
    for s in subjects:
        if masks[s].size != int(feature_bank[s].shape[0]):
            raise ValueError(
                f"Subject sub-{s} null FC/mask vertex mismatch: fc={feature_bank[s].shape[0]} mask={masks[s].size}"
            )
    out: dict[str, object] = {
        "values": [], "n_requested": 0, "n_used": 0, "n_skipped_overlap": 0,
        "n_skipped_empty": 0, "n_skipped_failure": 0, "failure_examples": [],
    }
    for seed in seeds:
        # One generator per repetition draws every subject's permutation, so
        # repetitions share no permutation.
        rng = np.random.RandomState(int(seed))
        null_bank: dict[str, np.ndarray] = {}
        null_masks: dict[str, np.ndarray] = {}
        permutations_by_subject = {}
        for s in subjects:
            arr = feature_bank[s]
            perm = rng.permutation(arr.shape[0])
            permutations_by_subject[s] = perm.tolist()
            if method_uses_network_fc_input(method) and not uses_subject_similarity_bank(method):
                null_bank[s] = arr[perm, :].astype(np.float32)
            else:
                null_bank[s] = arr[np.ix_(perm, perm)].astype(np.float32)
            null_masks[s] = masks[s][perm]
        null_options = dict(gradient_options or {})
        null_options["alignment_vertex_permutations"] = permutations_by_subject
        repetition = subject_split_ari(
            method=method,
            feature_bank=null_bank,
            valid_masks_by_subject=null_masks,
            subjects=subjects,
            tau=tau,
            adjacency_full=adjacency_full,
            k=k,
            seeds=[seed],
            gradient_components=gradient_components,
            allow_diagnostic_failures=allow_diagnostic_failures,
            gradient_options=null_options,
        )
        for key, value in repetition.items():
            out[key] = out[key] + value
    out["failure_examples"] = out["failure_examples"][:MAX_DIAGNOSTIC_FAILURE_EXAMPLES]
    out["permutation_scope"] = "subject_specific"
    return out


def weak_null_ari(k: int, n_vertices: int, seeds: list[int]) -> dict[str, object]:
    vals: list[float] = []
    for seed in seeds:
        rng = np.random.RandomState(int(seed))
        a = rng.randint(1, int(k) + 1, size=(int(n_vertices),), dtype=np.int32)
        b = rng.randint(1, int(k) + 1, size=(int(n_vertices),), dtype=np.int32)
        ari = float(adjusted_rand_score(a, b))
        vals.append(float(ari))
    return {
        "values": vals,
        "n_requested": int(len(seeds)),
        "n_used": int(len(vals)),
    }


def resolve_k_eval_jobs(requested_jobs: int, n_k: int) -> int:
    if int(n_k) <= 0:
        return 1
    return max(1, min(int(requested_jobs), int(n_k), int(MAX_K_EVAL_JOBS)))


def redundancy_early_stop_triggered(row: dict[str, object], enabled: bool) -> bool:
    return bool(enabled) and int(row.get("nonredundant_ok", 1)) == 0


def method_surface_affinity_metadata(method: str) -> tuple[str, str]:
    if method == "gradient-spectral":
        return GRADIENT_SPECTRAL_AFFINITY, GRADIENT_SPECTRAL_AFFINITY_DESCRIPTION
    if method == "gradient-spectral-rbf":
        return "mesh-rbf", RBF_AFFINITY_DESCRIPTION
    if method in WARD_GRADIENT_METHODS:
        return "none", "ward_euclidean_binary_mesh_connectivity"
    if method == "gradient-gmm":
        return "none", "full_covariance_gaussian_mixture_no_spatial_constraint"
    if is_multigraph_method(method):
        return "none", MULTIGRAPH_AFFINITY_DESCRIPTION
    return POSITIVE_COSINE_AFFINITY, POSITIVE_COSINE_AFFINITY_DESCRIPTION


def build_method_surface_affinity(
    method: str,
    features: np.ndarray,
    adjacency: sparse.spmatrix,
    *,
    rbf_scale: float = 1.0,
) -> sparse.csr_matrix:
    if method == "gradient-spectral":
        return build_gradient_spectral_affinity(features, adjacency)
    if method == "gradient-spectral-rbf":
        return build_rbf_mesh_affinity(features, adjacency, rbf_scale=rbf_scale)[0]
    if method in (WARD_GRADIENT_METHODS | {"gradient-gmm"}) or is_multigraph_method(method):
        raise ValueError(f"{method} does not construct a surface affinity")
    return build_positive_cosine_surface_affinity(features, adjacency)


def cluster_from_precomputed_group_features(
    *,
    method: str,
    group_features: np.ndarray,
    support_mask: np.ndarray,
    adjacency_full: sparse.csr_matrix,
    k: int,
    seed: int,
    rbf_scale: float = 1.0,
    clustering_metadata: dict[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.flatnonzero(support_mask)
    if idx.size == 0:
        raise RuntimeError("Support mask has zero vertices")
    adj = adjacency_full[idx][:, idx]
    if is_multigraph_method(method):
        if int(k) > MULTIGRAPH_MAX_COORDINATES:
            raise ValueError(f"{method} supports K <= {MULTIGRAPH_MAX_COORDINATES}, got {k}")
        labels, metadata = discretize_stack(group_features, int(k), seed=int(seed))
        if clustering_metadata is not None:
            clustering_metadata.update(metadata)
    elif is_gradient_method(method):
        labels, metadata = cluster_gradients(
            method, group_features, adj, k=int(k), seed=int(seed), rbf_scale=rbf_scale,
        )
        if clustering_metadata is not None:
            clustering_metadata.update(metadata)
    else:
        fused = build_method_surface_affinity(method, group_features, adj)
        labels = spectral_cluster_from_affinity(fused.toarray(), int(k), int(seed))
    full = np.zeros((support_mask.size,), dtype=np.int32)
    full[idx] = labels
    return labels.astype(np.int32), full


def evaluate_k_candidate(
    *,
    combo: Combo,
    method: str,
    k: int,
    seed: int,
    n_support: int,
    v_min: int,
    group_features_active: np.ndarray,
    support_mask: np.ndarray,
    adjacency_full: sparse.csr_matrix,
    adjacency_support: sparse.csr_matrix,
    feature_bank: dict[str, np.ndarray],
    valid_masks: dict[str, np.ndarray],
    subjects: list[str],
    tau: float,
    split_seeds: list[int],
    null_seeds: list[int],
    null_resamples: int,
    run_feature_banks: dict[int, dict[str, np.ndarray]],
    seed_base: int,
    max_components_threshold: int,
    min_profile_coherence_threshold: float,
    k_selection_mode: str,
    roi_fc_bank_for_redundancy: dict[str, np.ndarray] | None,
    gradient_components: tuple[int, ...] = DEFAULT_GRADIENT_COMPONENTS,
    allow_diagnostic_failures: bool = False,
    gradient_options: dict[str, object] | None = None,
    precomputed_split_halves: list[tuple[FittedHalf, FittedHalf]] | None = None,
    precomputed_null_halves: list[tuple[FittedHalf, FittedHalf]] | None = None,
    redundancy_threshold: float = PROFILE_REDUNDANCY_MAX_SIM,
) -> tuple[int, np.ndarray, dict[str, object]]:
    clustering_metadata: dict[str, object] = {}
    labels_active, labels_full = cluster_from_precomputed_group_features(
        method=method,
        group_features=group_features_active,
        support_mask=support_mask,
        adjacency_full=adjacency_full,
        k=int(k),
        seed=int(seed),
        rbf_scale=float((gradient_options or {}).get("rbf_scale", 1.0)),
        clustering_metadata=clustering_metadata,
    )
    support_indices = np.flatnonzero(support_mask)
    supported_mesh = adjacency_full[support_indices][:, support_indices]
    affinity_graph = (
        binary_mesh_adjacency(supported_mesh, len(support_indices))
        if method in (WARD_GRADIENT_METHODS | {"gradient-gmm"}) or is_multigraph_method(method) else
        build_method_surface_affinity(
            method, group_features_active, supported_mesh,
            rbf_scale=float((gradient_options or {}).get("rbf_scale", 1.0)),
        )
    )
    graph_diagnostics = affinity_graph_diagnostics(
        affinity_graph,
        labels=labels_active,
        k=int(k),
    )

    sizes = [int(np.sum(labels_active == c)) for c in range(1, int(k) + 1)]
    min_size = int(min(sizes)) if sizes else 0
    comp = connected_components_per_label(labels_active, adjacency_support)
    max_components = int(max(comp.values())) if comp else 0
    if is_multigraph_method(method):
        min_coh, mean_coh = stack_coherence(group_features_active, labels_active, int(k))
    else:
        min_coh, mean_coh = cluster_profile_coherence(group_features_active, labels_active, int(k))

    if precomputed_split_halves is not None:
        split_records = [
            evaluate_half_pair_for_k(
                half_a, half_b, adjacency_full, method, int(k), i, "split",
                split_seeds[i] if i < len(split_seeds) else split_seed_for_index(i, int(seed_base))
            )
            for i, (half_a, half_b) in enumerate(precomputed_split_halves)
        ]
        split_stats = compute_resample_statistics(split_records, requested=len(precomputed_split_halves))
        ari_vals = [r.ari for r in split_records if r.status == "ok" and math.isfinite(r.ari)]
        ari_mean = split_stats["group_ari_mean"]
        ari_se = split_stats["group_ari_se"]
        ari_status = "ok" if split_stats["subject_split_failed"] == 0 else "partial_resample_failures_skipped"
        n_split_req = int(split_stats["subject_split_requested"])
        n_split_used = int(split_stats["subject_split_used"])
        n_split_overlap = sum(1 for r in split_records if "Overlap vertices" in r.reason)
        n_split_empty = 0
        n_split_fail = int(split_stats["subject_split_failed"])
    else:
        split_records = None
        ari_obj = subject_split_ari(
            method=method,
            feature_bank=feature_bank,
            valid_masks_by_subject=valid_masks,
            subjects=subjects,
            tau=float(tau),
            adjacency_full=adjacency_full,
            k=int(k),
            seeds=split_seeds,
            gradient_components=gradient_components,
            allow_diagnostic_failures=allow_diagnostic_failures,
            gradient_options=gradient_options,
        )
        ari_vals = list(ari_obj["values"])
        if not ari_vals:
            if not allow_diagnostic_failures:
                raise RuntimeError(f"No valid ARI resamples for combo={combo} k={k}")
            ari_mean = math.nan
            ari_se = math.nan
            ari_status = "no_valid_subject_split_resamples"
        else:
            ari_arr = np.asarray(ari_vals, dtype=np.float64)
            ari_mean = float(np.mean(ari_arr))
            ari_se = float(np.std(ari_arr, ddof=1) / np.sqrt(ari_arr.size)) if ari_arr.size > 1 else 0.0
            ari_status = "ok" if int(ari_obj["n_skipped_failure"]) == 0 else "partial_resample_failures_skipped"
        n_split_req = int(ari_obj["n_requested"])
        n_split_used = int(ari_obj["n_used"])
        n_split_overlap = int(ari_obj["n_skipped_overlap"])
        n_split_empty = int(ari_obj["n_skipped_empty"])
        n_split_fail = int(ari_obj["n_skipped_failure"])

    if precomputed_null_halves is not None:
        null_records = [
            evaluate_half_pair_for_k(
                null_a, null_b, adjacency_full, method, int(k), i, "null",
                null_seeds[i] if i < len(null_seeds) else null_seed_for_index(i, int(seed_base))
            )
            for i, (null_a, null_b) in enumerate(precomputed_null_halves)
        ]
        null_stats = compute_null_statistics(null_records, requested=len(precomputed_null_halves))
        strong = [r.ari for r in null_records if r.status == "ok" and math.isfinite(r.ari)]
        strong_arr = np.asarray(strong, dtype=np.float64) if strong else np.array([0.0], dtype=np.float64)
        null_mean = float(null_stats["ARI_null_mean"])
        null_pct5 = float(null_stats["ARI_null_5pct"])
        null_pct95 = float(null_stats["ARI_null_95pct"])
        n_null_req = int(null_stats["B_null_requested"])
        n_null_used = int(null_stats["B_null_used_strong"])
        n_null_fail = int(null_stats["null_failed"])
        weak = []
    else:
        null_records = None
        weak_obj = weak_null_ari(int(k), int(n_support), null_seeds)
        weak = list(weak_obj["values"])
        strong_obj = strong_null_ari(
            method=method,
            feature_bank=feature_bank,
            valid_masks_by_subject=valid_masks,
            subjects=subjects,
            tau=float(tau),
            adjacency_full=adjacency_full,
            k=int(k),
            seeds=null_seeds,
            gradient_components=gradient_components,
            allow_diagnostic_failures=allow_diagnostic_failures,
            gradient_options=gradient_options,
        )
        strong = list(strong_obj["values"])
        strong_arr = np.asarray(strong, dtype=np.float64) if strong else np.array([0.0], dtype=np.float64)
        null_mean = float(np.mean(strong_arr))
        null_pct5 = float(np.percentile(strong_arr, 5.0))
        null_pct95 = float(np.percentile(strong_arr, 95.0))
        n_null_req = int(null_resamples)
        n_null_used = int(strong_obj["n_used"])
        n_null_fail = int(strong_obj["n_skipped_failure"])

    run_obj = run_aware_ari(
        method=method,
        run_feature_banks=run_feature_banks,
        subjects=subjects,
        support_mask=support_mask,
        adjacency_full=adjacency_full,
        k=int(k),
        seed_base=int(seed_base),
        gradient_components=gradient_components,
        allow_diagnostic_failures=allow_diagnostic_failures,
        gradient_options=gradient_options,
    )
    run_vals = list(run_obj["values"])
    if run_vals:
        run_arr = np.asarray(run_vals, dtype=np.float64)
        run_ari_mean = float(np.nanmean(run_arr))
        run_ari_se = (
            float(np.nanstd(run_arr, ddof=1) / np.sqrt(np.sum(np.isfinite(run_arr))))
            if np.sum(np.isfinite(run_arr)) > 1
            else 0.0
        )
        run_ari_status = "ok" if int(run_obj["n_skipped_failure"]) == 0 else "partial_run_failures_skipped"
    else:
        run_ari_mean = 0.0
        run_ari_se = 0.0
        run_ari_status = "no_valid_run_pairs"

    max_profile_similarity = float("nan")
    redundant_pair_count = -1
    redundancy_profile_status = "not_requested"
    if roi_fc_bank_for_redundancy is not None:
        redundancy_profile = _cluster_vertex_fc_profile_for_redundancy_diagnostic(
            labels_full=labels_full,
            support_mask=support_mask,
            subjects=subjects,
            roi_fc_bank=roi_fc_bank_for_redundancy,
            valid_masks_by_subject=valid_masks,
        )
        max_profile_similarity, redundant_pair_count = profile_redundancy_metrics(
            redundancy_profile, threshold=redundancy_threshold,
        )
        redundancy_profile_status = "ok"

    row = {
        "method": method,
        "atlas": combo.atlas,
        "smoothing": combo.smoothing,
        "hemi": combo.hemi,
        "coverage_threshold": float(combo.coverage_threshold),
        "k": int(k),
        "n_subjects": int(len(subjects)),
        "n_support_vertices": int(n_support),
        "V_min_effective": int(v_min),
        "group_ari_mean": ari_mean,
        "group_ari_se": ari_se,
        "group_ari_diagnostic_status": ari_status,
        "subject_split_requested": n_split_req,
        "subject_split_used": n_split_used,
        "subject_split_skipped_overlap": n_split_overlap,
        "subject_split_skipped_empty": n_split_empty,
        "subject_split_skipped_failure": n_split_fail,
        "run_aware_ari_mean": run_ari_mean,
        "run_aware_ari_se": run_ari_se,
        "run_aware_ari_status": run_ari_status,
        "run_aware_n_pairs": int(run_obj["n_pairs"]),
        "run_aware_n_used": int(run_obj["n_used"]),
        "run_aware_skipped_failure": int(run_obj["n_skipped_failure"]),
        "ARI_null_mean": null_mean,
        "ARI_null_5pct": null_pct5,
        "ARI_null_95pct": null_pct95,
        "ARI_weak_null_mean": float(np.mean(np.asarray(weak, dtype=np.float64))) if weak else math.nan,
        "B_null_requested": n_null_req,
        "B_null_used_strong": n_null_used,
        "B_null_used_weak": int(weak_obj["n_used"]) if "weak_obj" in locals() else 0,
        "B_null_skipped_failure_strong": n_null_fail,
        "min_cluster_size": int(min_size),
        "max_components_per_cluster": int(max_components),
        "affinity_graph_diagnostics": graph_diagnostics,
        "gradient_clustering": clustering_metadata,
        "min_profile_coherence": float(min_coh),
        "mean_profile_coherence": float(mean_coh),
        "nontrivial_min_size_ok": int(min_size >= v_min),
        "nontrivial_connectedness_ok": int(max_components <= int(max_components_threshold)),
        "nontrivial_coherence_ok": int(min_coh >= float(min_profile_coherence_threshold)),
        "max_profile_similarity": float(max_profile_similarity),
        "redundant_pair_count": int(redundant_pair_count),
        "redundancy_profile_status": redundancy_profile_status,
        "redundancy_threshold": float(redundancy_threshold),
        "nonredundant_ok": int(
            np.isfinite(max_profile_similarity)
            and max_profile_similarity <= float(redundancy_threshold)
            and int(redundant_pair_count) == 0
        ),
        "reject_nontrivial_fail": 0,
        "reject_redundant": 0,
        "reject_reason_primary": "",
        "selected_final": 0,
    }
    if split_records is not None:
        row["split_repetitions"] = [r.__dict__ for r in split_records]
    if null_records is not None:
        row["null_repetitions"] = [r.__dict__ for r in null_records]
    return int(k), labels_full.astype(np.int32), row


def runwise_feature_bank(
    *,
    method: str,
    shared_root: Path,
    subjects: list[str],
    atlas: str,
    smoothing: str,
    hemi: str,
    valid_masks: dict[str, np.ndarray],
    subject_fc_mode: SubjectFCMode,
    reference_root: Path | None = None,
) -> dict[int, dict[str, np.ndarray]]:
    run_ids = common_fc_run_ids(shared_root, subjects, atlas, smoothing, hemi)
    out: dict[int, dict[str, np.ndarray]] = {rid: {} for rid in run_ids}
    for rid in run_ids:
        for s in subjects:
            if method_uses_network_fc_input(method):
                p = (
                    shared_root
                    / f"sub-{s}"
                    / "fc"
                    / atlas
                    / smoothing
                    / f"hemi_{hemi}"
                    / "runs"
                    / f"run-{rid}"
                    / runwise_network_feature_filename(method, atlas)
                )
                if not p.exists():
                    raise FileNotFoundError(f"Missing runwise network feature: {p}")
                if str(atlas) == "kong2022" and method in ({"group-roi-sim-spectral", "gradient-spectral", "wta-top"} | GRADIENT_METHODS | MULTIGRAPH_METHODS):
                    reference_dir = None
                    if reference_root is not None:
                        reference_dir = Path(reference_root) / f"sub-{s}" / "reference" / "kong2022"
                    validate_kong2022_roi_fc(p, reference_dir=reference_dir)
                active = fisher_z_transform_fc(np.load(p).astype(np.float32))
                full = np.full((valid_masks[s].size, active.shape[1]), np.nan, dtype=np.float32)
                idx = np.flatnonzero(valid_masks[s])
                if active.shape[0] == idx.size:
                    full[idx, :] = active
                elif active.shape[0] == valid_masks[s].size:
                    full = active
                else:
                    raise ValueError(f"Unexpected runwise network feature shape for {p}: {active.shape}")
                run_feature = preprocess_subject_fc_matrix(
                    method=method,
                    fc=full,
                    subject_fc_mode=subject_fc_mode,
                )
                if uses_subject_similarity_bank(method):
                    out[rid][s] = subject_similarity_feature(run_feature, valid_masks[s], source=p)
                else:
                    out[rid][s] = (
                        roi_similarity_from_network_fc(run_feature)
                        if is_roi_similarity_method(method)
                        else run_feature
                    )
            else:
                p = resolve_intrinsic_feature_path(shared_root, s, smoothing, hemi, rid)
                arr = np.load(p).astype(np.float32)
                arr = fisher_z_transform_fc(arr)
                np.fill_diagonal(arr, 0.0)
                idx = np.flatnonzero(valid_masks[s])
                full = np.full((valid_masks[s].size, valid_masks[s].size), np.nan, dtype=np.float32)
                if arr.shape == (idx.size, idx.size):
                    full[np.ix_(idx, idx)] = arr
                elif arr.shape == (valid_masks[s].size, valid_masks[s].size):
                    full = arr
                else:
                    raise ValueError(f"Unexpected runwise intrinsic feature shape for {p}: {arr.shape}")
                out[rid][s] = preprocess_subject_fc_matrix(
                    method=method,
                    fc=full,
                    subject_fc_mode=subject_fc_mode,
                )
    return out


def active_fc_run_ids(shared_root: Path, subject: str, atlas: str, smoothing: str, hemi: str) -> list[int]:
    summary_path = (
        shared_root
        / f"sub-{subject}"
        / "fc"
        / atlas
        / smoothing
        / f"hemi_{hemi}"
        / "fc_summary.json"
    )
    if not summary_path.exists():
        return [1, 2, 3, 4]
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    labels = payload.get("run_labels")
    if not isinstance(labels, list) or not labels:
        return [1, 2, 3, 4]
    run_ids: list[int] = []
    for label in labels:
        raw = str(label)
        if not raw.startswith("run-"):
            raise ValueError(f"Invalid run label in FC summary {summary_path}: {label!r}")
        run_ids.append(int(raw.removeprefix("run-")))
    if len(set(run_ids)) != len(run_ids):
        raise ValueError(f"Duplicate run labels in FC summary {summary_path}: {labels!r}")
    return sorted(run_ids)


def common_fc_run_ids(shared_root: Path, subjects: list[str], atlas: str, smoothing: str, hemi: str) -> list[int]:
    if not subjects:
        raise ValueError("At least one subject is required for runwise features")
    runs_by_subject = [set(active_fc_run_ids(shared_root, s, atlas, smoothing, hemi)) for s in subjects]
    common = sorted(set.intersection(*runs_by_subject))
    if not common:
        raise RuntimeError(
            f"No common FC runs across subjects for atlas={atlas} smoothing={smoothing} hemi={hemi}"
        )
    return common


def run_aware_ari(
    *,
    method: str,
    run_feature_banks: dict[int, dict[str, np.ndarray]],
    subjects: list[str],
    support_mask: np.ndarray,
    adjacency_full: sparse.csr_matrix,
    k: int,
    seed_base: int,
    gradient_components: tuple[int, ...] = DEFAULT_GRADIENT_COMPONENTS,
    allow_diagnostic_failures: bool = False,
    gradient_options: dict[str, object] | None = None,
) -> dict[str, object]:
    runs = sorted(run_feature_banks.keys())
    labels_by_run: dict[int, np.ndarray] = {}
    n_skipped_failure = 0
    failure_examples: list[str] = []
    for rid in runs:
        try:
            _, full = cluster_for_subset(
                method=method,
                feature_bank=run_feature_banks[rid],
                subjects_subset=subjects,
                support_mask=support_mask,
                adjacency_full=adjacency_full,
                k=int(k),
                seed=int(seed_base + rid * 1000 + k),
                gradient_components=gradient_components,
                gradient_options={**(gradient_options or {}), "alignment_run_id": rid},
            )
        except RuntimeError as exc:
            if not allow_diagnostic_failures:
                raise
            n_skipped_failure += 1
            if len(failure_examples) < MAX_DIAGNOSTIC_FAILURE_EXAMPLES:
                failure_examples.append(str(exc))
            continue
        labels_by_run[rid] = full[support_mask]
    vals: list[float] = []
    n_pairs = 0
    successful_runs = sorted(labels_by_run.keys())
    for i in range(len(successful_runs)):
        for j in range(i + 1, len(successful_runs)):
            n_pairs += 1
            a = labels_by_run[successful_runs[i]]
            b = labels_by_run[successful_runs[j]]
            mapping = build_hungarian_mapping(a, b, int(k))
            b_r = remap_labels(b, mapping)
            ari = float(adjusted_rand_score(a, b_r))
            vals.append(float(ari))
    return {
        "values": vals,
        "n_pairs": int(n_pairs),
        "n_used": int(len(vals)),
        "n_skipped_failure": int(n_skipped_failure),
        "failure_examples": failure_examples,
    }


def build_subject_surface_consensus(
    subjects: list[str],
    hemis: list[str],
    batch_root: Path,
) -> tuple[dict[str, Path], dict[str, dict[str, object]], dict[str, np.ndarray]]:
    """Build subject-consensus surface metadata from all subjects in the manifest.

    Consensus policy is strict: any topology mismatch OR long-axis order mismatch
    causes a hard failure and lists mismatched subjects.
    """
    consensus_surfaces: dict[str, Path] = {}
    topology_audit: dict[str, dict[str, object]] = {}
    long_axis_order_by_hemi: dict[str, np.ndarray] = {}
    for hemi in hemis:
        ref_subject = subjects[0]
        ref_surface = infer_surface_path(ref_subject, hemi, batch_root)
        ref_coords, ref_faces = load_surface(ref_surface)
        ref_order = long_axis_order(ref_surface)
        topology_mismatched: list[str] = []
        axis_mismatched: list[str] = []
        for subject in subjects[1:]:
            surf_path = infer_surface_path(subject, hemi, batch_root)
            coords, faces = load_surface(surf_path)
            topo_ok = (
                int(coords.shape[0]) == int(ref_coords.shape[0])
                and int(faces.shape[0]) == int(ref_faces.shape[0])
                and faces.shape == ref_faces.shape
                and np.array_equal(faces, ref_faces)
            )
            if not topo_ok:
                topology_mismatched.append(f"sub-{subject}")
                continue
            this_order = long_axis_order(surf_path)
            if this_order.shape != ref_order.shape or not np.array_equal(this_order, ref_order):
                axis_mismatched.append(f"sub-{subject}")
        if topology_mismatched:
            raise RuntimeError(
                "Subject-consensus surface topology mismatch "
                f"for hemi-{hemi}; reference=sub-{ref_subject}; mismatched={topology_mismatched[:10]}"
            )
        if axis_mismatched:
            raise RuntimeError(
                "Subject-consensus long-axis order mismatch "
                f"for hemi-{hemi}; reference=sub-{ref_subject}; mismatched={axis_mismatched[:10]}"
            )
        consensus_surfaces[hemi] = ref_surface
        long_axis_order_by_hemi[hemi] = ref_order
        topology_audit[hemi] = {
            "template_source_mode": "subject_consensus",
            "reference_subject_for_comparison": f"sub-{ref_subject}",
            "n_vertices": int(ref_coords.shape[0]),
            "n_faces": int(ref_faces.shape[0]),
            "topology_consistent_subjects": int(len(subjects)),
            "long_axis_order_consistent_subjects": int(len(subjects)),
        }
    return consensus_surfaces, topology_audit, long_axis_order_by_hemi


def verify_subject_topology_against_template(
    subjects: list[str],
    hemis: list[str],
    template_surfaces: dict[str, Path],
    batch_root: Path,
) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for hemi in hemis:
        t_coords, t_faces = load_surface(template_surfaces[hemi])
        n_vertices = int(t_coords.shape[0])
        n_faces = int(t_faces.shape[0])
        mismatched: list[str] = []
        for subject in subjects:
            p = infer_surface_path(subject, hemi, batch_root)
            coords, faces = load_surface(p)
            topo_ok = (
                int(coords.shape[0]) == n_vertices
                and int(faces.shape[0]) == n_faces
                and faces.shape == t_faces.shape
                and np.array_equal(faces, t_faces)
            )
            if not topo_ok:
                mismatched.append(f"sub-{subject}")
        if mismatched:
            raise RuntimeError(
                "Template topology mismatch against manifest subjects "
                f"for hemi-{hemi}; template={template_surfaces[hemi]}; mismatched={mismatched[:10]}"
            )
        out[hemi] = {
            "template_source_mode": "group_average_template",
            "n_vertices": n_vertices,
            "n_faces": n_faces,
            "topology_consistent_subjects": int(len(subjects)),
        }
    return out


def one_sided_network_partial_corr_rows(rows: np.ndarray, network_ts: np.ndarray) -> np.ndarray:
    """Residualize rows against other networks, then correlate with each raw target network."""
    x = np.asarray(rows, dtype=np.float32)
    y = np.asarray(network_ts, dtype=np.float32)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"Expected 2D rows and network_ts, got {x.shape} and {y.shape}")
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"Time dimension mismatch for partial FC: {x.shape} vs {y.shape}")
    out = np.zeros((x.shape[0], y.shape[0]), dtype=np.float32)
    x_center = x - np.nanmean(x, axis=1, keepdims=True)
    x_center[~np.isfinite(x_center)] = 0.0
    y_center = y - np.nanmean(y, axis=1, keepdims=True)
    y_center[~np.isfinite(y_center)] = 0.0
    for j in range(y.shape[0]):
        if y.shape[0] <= 1:
            resid = x_center
        else:
            keep = [i for i in range(y.shape[0]) if i != j]
            controls = y_center[np.asarray(keep, dtype=np.int32), :]
            gram = controls @ controls.T
            beta = (x_center @ controls.T) @ np.linalg.pinv(gram, rcond=1e-6)
            resid = x_center - beta @ controls
        out[:, j] = corrcoef_rows(resid, y[[j], :])[:, 0]
    return out.astype(np.float32)


def cluster_network_profile(
    *,
    group_labels_final: np.ndarray,
    support_mask: np.ndarray,
    subjects: list[str],
    shared_root: Path,
    surface_root: Path,
    reference_root: Path,
    atlas: str,
    smoothing: str,
    hemi: str,
    valid_masks_by_subject: dict[str, np.ndarray],
    profile_fc_mode: str,
    network_taxonomy: str = "canonical7",
) -> np.ndarray:
    labels = np.asarray(group_labels_final, dtype=np.int32).reshape(-1)
    support = np.asarray(support_mask, dtype=bool).reshape(-1)
    if labels.shape != support.shape:
        raise ValueError(
            "Cluster-network profile labels/support shape mismatch: "
            f"labels={labels.shape} support={support.shape}"
        )
    k_final = int(np.max(labels))
    if k_final <= 0:
        raise ValueError("Cluster-network profile requires at least one positive cluster label")
    profiles: list[np.ndarray] = []
    for subject in subjects:
        ts_path = surface_root / f"sub-{subject}" / "surface" / smoothing / f"sub-{subject}_hemi-{hemi}_timeseries.npy"
        if not ts_path.is_file():
            raise FileNotFoundError(
                f"Missing subject surface time series for sub-{subject}: {ts_path}"
            )
        if subject not in valid_masks_by_subject:
            raise ValueError(
                f"Missing subject valid mask for sub-{subject}: "
                "valid_masks_by_subject has no entry"
            )
        valid_mask = np.asarray(valid_masks_by_subject[subject], dtype=bool)
        if valid_mask.ndim != 1 or valid_mask.shape != labels.shape:
            raise ValueError(
                f"Subject valid mask shape mismatch for sub-{subject}: "
                f"mask={valid_mask.shape} expected={labels.shape}; surface={ts_path}"
            )
        if str(atlas) == "kong2022" and str(network_taxonomy) in {"canonical7", "canonical8"}:
            _network_names, net = load_kong2022_network_reference(
                reference_root, subject, atlas, network_taxonomy
            )
        else:
            _order_tsv, network_timeseries_name = network_reference_filenames(network_taxonomy)
            network_order_path = reference_path(reference_root, subject, atlas, _order_tsv)
            network_timeseries_path = reference_path(
                reference_root, subject, atlas, network_timeseries_name
            )
            missing_reference = [
                str(path)
                for path in (network_order_path, network_timeseries_path)
                if not path.is_file()
            ]
            if missing_reference:
                raise FileNotFoundError(
                    f"Missing subject network reference for sub-{subject}: "
                    + ", ".join(missing_reference)
                )
            net = np.asarray(np.load(network_timeseries_path), dtype=np.float32)
        ts = np.load(ts_path).astype(np.float32)
        if ts.ndim != 2:
            raise ValueError(
                f"Subject surface time series must be 2D for sub-{subject}: "
                f"path={ts_path} shape={ts.shape}"
            )
        if net.ndim != 2:
            raise ValueError(
                f"Subject network reference must be 2D for sub-{subject}: "
                f"path={network_timeseries_path if 'network_timeseries_path' in locals() else 'Kong reference'} "
                f"shape={net.shape}"
            )
        if ts.shape[0] != labels.size:
            raise ValueError(
                f"Subject surface vertex count mismatch for sub-{subject}: "
                f"path={ts_path} shape={ts.shape} labels={labels.shape}"
            )
        if ts.shape[1] != net.shape[1]:
            raise ValueError(
                f"Subject profile frame mismatch for sub-{subject}: "
                f"surface={ts_path} shape={ts.shape}; network_reference="
                f"{network_timeseries_path if 'network_timeseries_path' in locals() else 'Kong reference'} "
                f"shape={net.shape}"
            )
        _, run_lengths = load_run_segment_metadata(
            reference_path(reference_root, subject, atlas, "reference_summary.json")
        )
        ts = center_run_timeseries(ts, run_lengths)
        net = center_run_timeseries(net, run_lengths)
        valid = support & valid_mask & np.isfinite(ts).all(axis=1) & (np.ptp(ts, axis=1) > 0)
        cluster_ts = np.full((k_final, ts.shape[1]), np.nan, dtype=np.float32)
        for c in range(1, k_final + 1):
            m = (labels == c) & valid
            if int(np.sum(m)) == 0:
                continue
            with np.errstate(invalid="ignore"):
                cluster_ts[c - 1, :] = np.nanmean(ts[m, :], axis=0).astype(np.float32)
        if profile_fc_mode == "pearson":
            corr = corrcoef_rows(cluster_ts, net)
        elif profile_fc_mode == "partial_networks_only_one_sided":
            corr = one_sided_network_partial_corr_rows(cluster_ts, net)
        else:
            raise ValueError(f"Unknown profile_fc_mode: {profile_fc_mode}")
        corr = np.clip(corr, -0.999999, 0.999999)
        fisher = np.arctanh(corr).astype(np.float32)
        fisher[~np.isfinite(fisher)] = np.nan
        profiles.append(fisher)
    if not profiles:
        raise RuntimeError("No subject-level profiles available for cluster-network annotation")
    out = np.nanmean(np.stack(profiles, axis=0), axis=0).astype(np.float32)
    return out


def load_kong2022_roi_registry() -> list[dict[str, object]]:
    bundle = load_kong2022_bundle(dlabel_path=KONG2022_DLABEL_OVERRIDE)
    rows: list[dict[str, object]] = []
    for row in bundle["roi_registry"]:
        rows.append(
            {
                "roi_id": int(row["roi_id"]),
                "roi_name": str(row["roi_name"]),
                "hemi": str(row["hemi"]),
                "family17": str(row["family17"]),
                "subregion": str(row["subregion"]),
                "rank": int(row["rank"]),
                "canonical8": str(row["canonical8"]),
                "canonical7": str(row.get("canonical7", "")),
            }
        )
    if len(rows) != 400:
        raise RuntimeError(f"Expected 400 Kong2022 registry rows, got {len(rows)}")
    return rows


def read_kong2022_roi_tsv(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        raw_rows = list(csv.DictReader(f, delimiter="\t"))
    mapping7: dict[str, str] | None = None
    out: list[dict[str, object]] = []
    for row in raw_rows:
        family17 = str(row["family17"]).strip()
        canonical7 = str(row.get("canonical7", "")).strip()
        if not canonical7:
            if mapping7 is None:
                mapping7 = load_kong2022_merge_mapping(taxonomy="canonical7")
            canonical7 = mapping7[family17]
        out.append(
            {
                "parcel_id": int(row["parcel_id"]),
                "parcel_name": str(row["parcel_name"]),
                "family17": family17,
                "canonical8": str(row["canonical8"]),
                "canonical7": canonical7,
                "hemisphere": str(row["hemisphere"]),
                "subregion": str(row.get("subregion", "")),
                "rank": int(row.get("rank", "0")),
                "n_grayordinates_used": int(row.get("n_grayordinates_used", "0")),
                "n_grayordinates_total": int(row.get("n_grayordinates_total", "0")),
            }
        )
    return out


def canonical7_registry_from_rows(
    rows: list[dict[str, object]],
    *,
    require_all_networks: bool = True,
) -> list[dict[str, object]]:
    """Derive canonical7 metadata from an existing Kong2022 ROI registry.

    This is a metadata transform only: ROI IDs/order and canonical8 remain intact;
    Auditory is mapped into Somatomotor through the maintained merge configuration.
    """
    mapping7 = load_kong2022_merge_mapping(taxonomy="canonical7")
    mapping8 = load_kong2022_merge_mapping(taxonomy="canonical8")
    out: list[dict[str, object]] = []
    seen_ids: set[int] = set()
    for index, raw in enumerate(rows, start=1):
        row = dict(raw)
        roi_id = int(row.get("roi_id", row.get("parcel_id", index)))
        if roi_id in seen_ids:
            raise ValueError(f"Duplicate Kong2022 ROI id in registry: {roi_id}")
        seen_ids.add(roi_id)
        family17 = str(row.get("family17", "")).strip()
        if family17 not in mapping7:
            raise ValueError(f"Unknown Kong2022 family17 in registry: {family17!r}")
        canonical8 = str(row.get("canonical8", "")).strip() or mapping8[family17]
        out.append(
            {
                "roi_id": roi_id,
                "roi_name": str(row.get("roi_name", row.get("parcel_name", roi_id))),
                "hemi": str(row.get("hemi", row.get("hemisphere", ""))),
                "family17": family17,
                "subregion": str(row.get("subregion", "")),
                "rank": int(row.get("rank", index) or index),
                "canonical8": canonical8,
                "canonical7": mapping7[family17],
            }
        )
    if not out:
        raise ValueError("Kong2022 ROI registry is empty")
    if require_all_networks:
        missing = sorted(set(canonical_families_7_order()) - {str(row["canonical7"]) for row in out})
        if missing:
            raise ValueError(f"Kong2022 canonical7 registry is missing networks: {missing}")
    return out


def read_roi_registry_for_atlas(reference_root: Path, subject: str, atlas: str) -> list[dict[str, object]]:
    if is_lynch2024_atlas(atlas):
        raise ValueError(
            f"Atlas {atlas!r} has network references, not an ROI registry; "
            "the normalized_top20_count WTA path is not supported for this atlas."
        )

    path = reference_path(reference_root, subject, atlas, "cortex_roi_parcels.tsv")
    if atlas == "kong2022":
        rows = read_kong2022_roi_registry(path)
        validate_kong2022_roi_registry(rows, path=path)
        return canonical7_registry_from_rows(rows)

    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise RuntimeError(f"Empty ROI/component TSV for atlas={atlas}: {path}")
    out: list[dict[str, object]] = []
    for idx, row in enumerate(rows):
        roi_id = int(row.get("parcel_id", row.get("roi_id", row.get("component_id", idx + 1))))
        roi_name = str(row.get("parcel_name", row.get("roi_name", row.get("component_name", roi_id))))
        canonical8 = str(row.get("canonical8", row.get("canonical_network", "")))
        canonical7 = str(row.get("canonical7", ""))
        family17 = str(row.get("family17", ""))
        lynch20 = str(row.get("lynch20_network", row.get("network", "")))
        out.append(
            {
                "roi_id": roi_id,
                "roi_name": roi_name,
                "hemi": str(row.get("hemi", row.get("hemisphere", ""))),
                "family17": family17,
                "lynch20_network": lynch20,
                "canonical8": canonical8,
                "canonical7": canonical7,
                "subregion": str(row.get("subregion", "")),
                "rank": int(row.get("rank", row.get("component_order", idx + 1)) or idx + 1),
                "n_grayordinates_used": row.get("n_grayordinates_used", ""),
            }
        )
    return out


def _cluster_vertex_fc_profile_for_redundancy_diagnostic(
    *,
    labels_full: np.ndarray,
    support_mask: np.ndarray,
    subjects: list[str],
    roi_fc_bank: dict[str, np.ndarray],
    valid_masks_by_subject: dict[str, np.ndarray],
) -> np.ndarray:
    # Internal K-selection diagnostic only. This is not a network-profile
    # annotation path and must not write cluster_network_profile.npy.
    labels = np.asarray(labels_full, dtype=np.int32)
    if labels.ndim != 1:
        raise ValueError(f"Expected 1D labels, got shape={labels.shape}")
    k = int(np.max(labels))
    if k <= 0:
        raise ValueError("No positive clusters in labels_full")
    support = np.asarray(support_mask, dtype=bool)
    profiles: list[np.ndarray] = []
    for subject in subjects:
        fc = np.asarray(roi_fc_bank[subject], dtype=np.float32)
        if fc.ndim != 2:
            raise ValueError(f"Expected 2D ROI-FC matrix for sub-{subject}, got shape={fc.shape}")
        if fc.shape[0] != labels.size:
            raise ValueError(
                f"ROI-FC vertex mismatch for sub-{subject}: fc={fc.shape[0]} labels={labels.size}"
            )
        valid = support & np.asarray(valid_masks_by_subject[subject], dtype=bool)
        cluster_profile = np.full((k, fc.shape[1]), np.nan, dtype=np.float32)
        for c in range(1, k + 1):
            m = valid & (labels == c)
            if int(np.sum(m)) == 0:
                continue
            with np.errstate(invalid="ignore"):
                cluster_profile[c - 1, :] = np.nanmean(fc[m, :], axis=0).astype(np.float32)
        profiles.append(cluster_profile)
    if not profiles:
        raise RuntimeError("No redundancy profiles available from roi_fc_bank")
    out = np.nanmean(np.stack(profiles, axis=0), axis=0).astype(np.float32)
    out[~np.isfinite(out)] = 0.0
    return out


def cluster_roi_profile(
    *,
    group_labels_final: np.ndarray,
    support_mask: np.ndarray,
    subjects: list[str],
    shared_root: Path,
    surface_root: Path,
    reference_root: Path,
    smoothing: str,
    hemi: str,
    valid_masks_by_subject: dict[str, np.ndarray],
    reference_atlas: str = "kong2022",
) -> tuple[np.ndarray, list[dict[str, object]]]:
    k_final = int(np.max(group_labels_final))
    full_registry = load_kong2022_roi_registry()
    reg_by_id = {int(r["roi_id"]): r for r in full_registry}
    col_by_id = {int(r["roi_id"]): i for i, r in enumerate(full_registry)}

    profiles: list[np.ndarray] = []
    for subject in subjects:
        ts_path = surface_root / f"sub-{subject}" / "surface" / smoothing / f"sub-{subject}_hemi-{hemi}_timeseries.npy"
        roi_path = reference_path(reference_root, subject, reference_atlas, "cortex_roi_parcel_timeseries.npy")
        roi_tsv = reference_path(reference_root, subject, reference_atlas, "cortex_roi_parcels.tsv")
        if not (ts_path.exists() and roi_path.exists() and roi_tsv.exists()):
            continue
        ts = np.load(ts_path).astype(np.float32)
        roi = np.load(roi_path).astype(np.float32)
        rows = read_kong2022_roi_tsv(roi_tsv)
        if ts.ndim != 2 or roi.ndim != 2:
            continue
        if ts.shape[1] != roi.shape[1]:
            continue
        if roi.shape[0] != len(rows):
            raise RuntimeError(
                f"ROI TSV rows mismatch for sub-{subject}: roi_shape={roi.shape} rows={len(rows)}"
            )

        _, run_lengths = load_run_segment_metadata(
            reference_path(reference_root, subject, reference_atlas, "reference_summary.json")
        )
        ts = center_run_timeseries(ts, run_lengths)
        roi = center_run_timeseries(roi, run_lengths)
        valid = support_mask & valid_masks_by_subject[subject] & np.isfinite(ts).all(axis=1) & (np.ptp(ts, axis=1) > 0)
        cluster_ts = np.full((k_final, ts.shape[1]), np.nan, dtype=np.float32)
        for c in range(1, k_final + 1):
            m = (group_labels_final == c) & valid
            if int(np.sum(m)) == 0:
                continue
            with np.errstate(invalid="ignore"):
                cluster_ts[c - 1, :] = np.nanmean(ts[m, :], axis=0).astype(np.float32)

        corr = corrcoef_rows(cluster_ts, roi)
        corr = np.clip(corr, -0.999999, 0.999999)
        fisher = np.arctanh(corr).astype(np.float32)
        fisher[~np.isfinite(fisher)] = np.nan

        scatter = np.full((k_final, 400), np.nan, dtype=np.float32)
        cols: list[int] = []
        for row in rows:
            roi_id = int(row["parcel_id"])
            reg = reg_by_id.get(roi_id)
            if reg is None:
                raise KeyError(f"Unexpected ROI id in sub-{subject}: {roi_id}")
            if str(row["parcel_name"]) != str(reg["roi_name"]):
                raise RuntimeError(
                    f"ROI name mismatch in sub-{subject} for id={roi_id}: "
                    f"{row['parcel_name']} vs {reg['roi_name']}"
                )
            if str(row["family17"]) != str(reg["family17"]) or str(row["canonical8"]) != str(reg["canonical8"]):
                raise RuntimeError(
                    f"ROI metadata mismatch in sub-{subject} for id={roi_id}: "
                    f"family17/canonical8 inconsistent with registry"
                )
            cols.append(int(col_by_id[roi_id]))
        scatter[:, np.asarray(cols, dtype=np.int32)] = fisher
        profiles.append(scatter)

    if not profiles:
        raise RuntimeError("No subject-level ROI profiles available for kong2022 cluster annotation")
    out = np.nanmean(np.stack(profiles, axis=0), axis=0).astype(np.float32)
    return out, full_registry


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cluster_roi_profile_subjectwise_from_explicit_original_source(
    *,
    group_labels_final: np.ndarray,
    support_mask: np.ndarray,
    subjects: list[str],
    source_root: Path | None,
    registry_path: Path | None,
    smoothing: str,
    hemi: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]], list[str]]:
    """Build a Kong400 profile from the explicitly approved fixed-template source.

    This recovery path is deliberately separate from the canonical shared-input
    path.  Every subject must have a hash-verified original-denoised
    hippocampal series, 400-ROI cortex series, and valid mask; missing or
    mismatched data fail closed instead of reducing the subject roster.
    """
    if source_root is None or registry_path is None:
        raise ValueError(
            "original-denoised Kong2022 profile recovery requires both "
            "--profile-source-root and --profile-source-registry"
        )
    source_root = Path(source_root).resolve()
    registry_path = Path(registry_path).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Missing original-denoised profile source root: {source_root}")
    if not registry_path.is_file():
        raise FileNotFoundError(f"Missing original-denoised 400-ROI registry: {registry_path}")

    registry_rows = read_kong2022_roi_registry(registry_path)
    validate_kong2022_roi_registry(registry_rows, path=registry_path)
    # Older verified source TSVs carry canonical8 only.  Derive canonical7
    # from the maintained family17 merge map while preserving the exact 400
    # ROI IDs/order; this is a taxonomy transform, not a feature reduction.
    full_registry = canonical7_registry_from_rows(registry_rows)
    if [int(row["roi_id"]) for row in full_registry] != list(range(1, KONG2022_ROI_COUNT + 1)):
        raise ValueError(f"Original-denoised registry is not in ordered 1..400 form: {registry_path}")

    labels = np.asarray(group_labels_final, dtype=np.int32).reshape(-1)
    support = np.asarray(support_mask, dtype=bool).reshape(-1)
    if labels.shape != support.shape or labels.ndim != 1:
        raise ValueError(
            "Original-denoised cluster profile requires one-dimensional labels and support mask "
            f"with identical shapes, labels={labels.shape} support={support.shape}"
        )
    k_final = int(np.max(labels))
    if k_final <= 0 or not np.all(np.isin(labels[support], np.arange(1, k_final + 1))):
        raise ValueError(f"Invalid fixed cluster labels for original-denoised profile: K={k_final}")

    profiles: list[np.ndarray] = []
    used_subjects: list[str] = []
    expected_subjects = [str(subject).removeprefix("sub-") for subject in subjects]
    registry_sha256 = _sha256_file(registry_path)
    for subject in expected_subjects:
        subject_root = source_root / f"sub-{subject}"
        manifest_path = subject_root / "subject_manifest.json"
        ts_path = subject_root / "surface" / str(smoothing) / f"hemi-{hemi}_timeseries.npy"
        roi_path = subject_root / "cortex" / "kong2022_roi_timeseries.npy"
        valid_path = subject_root / "surface" / "validity" / f"hemi-{hemi}_valid_mask.npy"
        required = (manifest_path, ts_path, roi_path, valid_path)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Original-denoised profile source is incomplete for "
                f"sub-{subject} {smoothing} hemi-{hemi}: " + ", ".join(missing)
            )
        manifest = load_json(manifest_path)
        if manifest.get("status") != "ok" or str(manifest.get("subject", "")).removeprefix("sub-") != subject:
            raise ValueError(f"Original-denoised subject manifest identity/status mismatch: {manifest_path}")
        recorded_registry = manifest.get("roi_registry_fingerprint")
        if not isinstance(recorded_registry, dict) or recorded_registry.get("sha256") != registry_sha256:
            raise ValueError(
                "Original-denoised ROI registry hash mismatch: "
                f"{manifest_path} expected={registry_sha256} recorded={recorded_registry!r}"
            )
        parameters = manifest.get("parameters") or {}
        roi_order = parameters.get("roi_order")
        if roi_order != list(range(1, KONG2022_ROI_COUNT + 1)):
            raise ValueError(f"Original-denoised ROI order evidence is not 1..400: {manifest_path}")
        outputs = manifest.get("outputs") or {}
        recorded_outputs = {
            str(Path(item["path"]).resolve()): str(item["sha256"])
            for item in outputs.values()
            if isinstance(item, dict) and item.get("path") and item.get("sha256")
        }
        for path in (ts_path, roi_path, valid_path):
            actual_path = str(path.resolve())
            actual_sha = _sha256_file(path)
            if recorded_outputs.get(actual_path) != actual_sha:
                raise ValueError(
                    "Original-denoised subject output hash mismatch: "
                    f"{path} recorded={recorded_outputs.get(actual_path)!r} actual={actual_sha}"
                )

        ts = np.asarray(np.load(ts_path), dtype=np.float32)
        roi = np.asarray(np.load(roi_path), dtype=np.float32)
        valid = np.asarray(np.load(valid_path), dtype=bool).reshape(-1)
        if ts.ndim != 2 or roi.ndim != 2:
            raise ValueError(f"Original-denoised profile arrays must be 2D: {ts_path} {ts.shape}; {roi_path} {roi.shape}")
        if ts.shape[0] != labels.size or valid.shape != labels.shape:
            raise ValueError(
                "Original-denoised hippocampal vertex contract mismatch: "
                f"sub-{subject} ts={ts.shape} labels={labels.shape} valid={valid.shape}"
            )
        if roi.shape[0] != KONG2022_ROI_COUNT or ts.shape[1] != roi.shape[1] or ts.shape[1] < 2:
            raise ValueError(
                "Original-denoised Kong400 frame/ROI contract mismatch: "
                f"sub-{subject} hippocampus={ts.shape} roi={roi.shape}"
            )
        recorded_frames = (manifest.get("header_evidence") or {}).get("concat_timepoints")
        if recorded_frames is not None and int(recorded_frames) != int(ts.shape[1]):
            raise ValueError(
                f"Original-denoised frame count mismatch: {manifest_path} "
                f"recorded={recorded_frames} actual={ts.shape[1]}"
            )
        roi_observed = np.isfinite(roi).all(axis=1)
        roi_empty = np.isnan(roi).all(axis=1)
        if np.any(~roi_observed & ~roi_empty):
            raise ValueError(f"Partially nonfinite original-denoised ROI time series: {roi_path}")
        _, run_lengths = load_run_segment_metadata(manifest_path)
        ts = center_run_timeseries(ts, run_lengths)
        roi = center_run_timeseries(roi, run_lengths)
        cluster_ts = np.full((k_final, ts.shape[1]), np.nan, dtype=np.float32)
        for cluster_id in range(1, k_final + 1):
            selected = (labels == cluster_id) & support & valid
            if not np.any(selected):
                raise ValueError(
                    "Original-denoised profile has an empty fixed cluster after the valid mask: "
                    f"sub-{subject} cluster={cluster_id} hemi={hemi} smoothing={smoothing}"
                )
            if not np.isfinite(ts[selected]).all():
                raise ValueError(
                    "Nonfinite original-denoised hippocampal series inside fixed cluster support: "
                    f"sub-{subject} cluster={cluster_id}"
                )
            cluster_ts[cluster_id - 1] = np.mean(ts[selected], axis=0, dtype=np.float32)
        values = fisher_z_transform_fc(corrcoef_rows(cluster_ts, roi))
        usable_cluster = np.isfinite(cluster_ts).all(axis=1) & (np.nanstd(cluster_ts, axis=1) > 0)
        usable_roi = roi_observed & (np.nanstd(roi, axis=1) > 0)
        values[~usable_cluster] = np.nan
        values[:, ~usable_roi] = np.nan
        profiles.append(values.astype(np.float32))
        used_subjects.append(subject)

    if used_subjects != expected_subjects:
        raise RuntimeError(
            "Original-denoised profile recovery changed the fixed subject order: "
            f"used={used_subjects} expected={expected_subjects}"
        )
    stack = np.stack(profiles, axis=0).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out_mean = np.nanmean(stack, axis=0).astype(np.float32)
    if not np.isfinite(out_mean).any():
        raise RuntimeError("Original-denoised Kong400 cluster profile is entirely nonfinite")
    return stack, out_mean, full_registry, used_subjects


def cluster_roi_profile_subjectwise(
    *,
    group_labels_final: np.ndarray,
    support_mask: np.ndarray,
    subjects: list[str],
    shared_root: Path,
    surface_root: Path,
    reference_root: Path,
    smoothing: str,
    hemi: str,
    valid_masks_by_subject: dict[str, np.ndarray],
    reference_atlas: str = "kong2022",
    profile_fc_mode: str = "cluster_mean_timeseries",
    profile_source_mode: str = "canonical",
    profile_source_root: Path | None = None,
    profile_source_registry: Path | None = None,
    array_loader=np.load,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]], list[str]]:
    if str(profile_source_mode) == "original-denoised":
        return cluster_roi_profile_subjectwise_from_explicit_original_source(
            group_labels_final=group_labels_final,
            support_mask=support_mask,
            subjects=subjects,
            source_root=profile_source_root,
            registry_path=profile_source_registry,
            smoothing=smoothing,
            hemi=hemi,
        )
    if str(profile_source_mode) != "canonical":
        raise ValueError(
            "Unknown Kong2022 cluster profile source mode: "
            f"{profile_source_mode!r}; expected 'canonical' or 'original-denoised'"
        )
    if str(profile_fc_mode) != "cluster_mean_timeseries":
        raise ValueError(
            "Unknown Kong2022 cluster profile FC mode: "
            f"{profile_fc_mode!r}; expected 'cluster_mean_timeseries'"
        )
    labels = np.asarray(group_labels_final, dtype=np.int32).reshape(-1)
    support = np.asarray(support_mask, dtype=bool).reshape(-1)
    if labels.shape != support.shape:
        raise ValueError(
            "Kong2022 cluster ROI profile labels/support shape mismatch: "
            f"labels={labels.shape} support={support.shape}"
        )
    k_final = int(np.max(labels))
    if k_final <= 0:
        raise ValueError("Kong2022 cluster ROI profile requires at least one positive cluster label")
    full_registry = load_kong2022_roi_registry()
    reg_by_id = {int(r["roi_id"]): r for r in full_registry}
    col_by_id = {int(r["roi_id"]): i for i, r in enumerate(full_registry)}

    profiles: list[np.ndarray] = []
    used_subjects: list[str] = []
    for subject in subjects:
        ts_path = surface_root / f"sub-{subject}" / "surface" / smoothing / f"sub-{subject}_hemi-{hemi}_timeseries.npy"
        roi_path = reference_path(reference_root, subject, reference_atlas, "cortex_roi_parcel_timeseries.npy")
        roi_tsv = reference_path(reference_root, subject, reference_atlas, "cortex_roi_parcels.tsv")
        missing = [str(path) for path in (ts_path, roi_path, roi_tsv) if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing Kong2022 cluster annotation input for sub-{subject}: "
                + ", ".join(missing)
            )
        if subject not in valid_masks_by_subject:
            raise ValueError(
                f"Missing subject valid mask for sub-{subject}: "
                "valid_masks_by_subject has no entry"
            )
        valid_mask = np.asarray(valid_masks_by_subject[subject], dtype=bool)
        if valid_mask.ndim != 1 or valid_mask.shape != labels.shape:
            raise ValueError(
                f"Subject valid mask shape mismatch for sub-{subject}: "
                f"mask={valid_mask.shape} expected={labels.shape}; surface={ts_path}"
            )
        ts = array_loader(ts_path).astype(np.float32, copy=False)
        roi = array_loader(roi_path).astype(np.float32, copy=False)
        rows = read_kong2022_roi_tsv(roi_tsv)
        if ts.ndim != 2:
            raise ValueError(
                f"Subject surface time series must be 2D for sub-{subject}: "
                f"path={ts_path} shape={ts.shape}"
            )
        if roi.ndim != 2:
            raise ValueError(
                f"Subject Kong2022 ROI reference must be 2D for sub-{subject}: "
                f"path={roi_path} shape={roi.shape}"
            )
        if ts.shape[0] != labels.size:
            raise ValueError(
                f"Subject surface vertex count mismatch for sub-{subject}: "
                f"path={ts_path} shape={ts.shape} labels={labels.shape}"
            )
        if roi.shape[0] != KONG2022_ROI_COUNT or roi.shape[0] != len(rows):
            raise ValueError(
                f"Subject Kong2022 ROI count mismatch for sub-{subject}: "
                f"path={roi_path} shape={roi.shape} registry_rows={len(rows)} "
                f"expected={KONG2022_ROI_COUNT}"
            )
        if ts.shape[1] != roi.shape[1]:
            raise ValueError(
                f"Subject profile frame mismatch for sub-{subject}: "
                f"surface={ts_path} shape={ts.shape}; roi_reference={roi_path} shape={roi.shape}"
            )
        if roi.shape[0] != len(rows):
            raise RuntimeError(
                f"ROI TSV rows mismatch for sub-{subject}: roi_shape={roi.shape} rows={len(rows)}"
            )

        _, run_lengths = load_run_segment_metadata(
            reference_path(reference_root, subject, reference_atlas, "reference_summary.json")
        )
        ts = center_run_timeseries(ts, run_lengths)
        roi = center_run_timeseries(roi, run_lengths)
        valid = support & valid_mask & np.isfinite(ts).all(axis=1) & (np.ptp(ts, axis=1) > 0)
        cluster_ts = np.full((k_final, ts.shape[1]), np.nan, dtype=np.float32)
        for c in range(1, k_final + 1):
            m = (labels == c) & valid
            if int(np.sum(m)) == 0:
                continue
            with np.errstate(invalid="ignore"):
                cluster_ts[c - 1, :] = np.nanmean(ts[m, :], axis=0).astype(np.float32)

        corr = corrcoef_rows(cluster_ts, roi)
        corr = np.clip(corr, -0.999999, 0.999999)
        fisher = np.arctanh(corr).astype(np.float32)
        fisher[~np.isfinite(fisher)] = np.nan

        scatter = np.full((k_final, 400), np.nan, dtype=np.float32)
        cols: list[int] = []
        for row in rows:
            roi_id = int(row["parcel_id"])
            reg = reg_by_id.get(roi_id)
            if reg is None:
                raise KeyError(f"Unexpected ROI id in sub-{subject}: {roi_id}")
            if str(row["parcel_name"]) != str(reg["roi_name"]):
                raise RuntimeError(
                    f"ROI name mismatch in sub-{subject} for id={roi_id}: "
                    f"{row['parcel_name']} vs {reg['roi_name']}"
                )
            if str(row["family17"]) != str(reg["family17"]) or str(row["canonical8"]) != str(reg["canonical8"]):
                raise RuntimeError(
                    f"ROI metadata mismatch in sub-{subject} for id={roi_id}: "
                    f"family17/canonical8 inconsistent with registry"
                )
            cols.append(int(col_by_id[roi_id]))
        scatter[:, np.asarray(cols, dtype=np.int32)] = fisher
        profiles.append(scatter)
        used_subjects.append(subject)

    if not profiles:
        raise RuntimeError("No subject-level ROI profiles available for kong2022 cluster annotation")
    stack = np.stack(profiles, axis=0).astype(np.float32)
    out_mean = np.nanmean(stack, axis=0).astype(np.float32)
    return stack, out_mean, full_registry, used_subjects


def cluster_roi_top_k(
    profile_400: np.ndarray,
    roi_registry: list[dict[str, object]],
    k_top: int = 20,
) -> dict[str, object]:
    out: dict[str, object] = {}
    for cidx in range(profile_400.shape[0]):
        row = profile_400[cidx, :]
        valid = np.flatnonzero(np.isfinite(row))
        if valid.size == 0:
            out[str(cidx + 1)] = {"top_positive": [], "top_negative": []}
            continue
        row_valid = row[valid]
        pos_order = valid[np.argsort(-row_valid)]
        neg_order = valid[np.argsort(row_valid)]
        n_take = int(min(k_top, valid.size))

        def build_items(indices: np.ndarray) -> list[dict[str, object]]:
            items: list[dict[str, object]] = []
            for col in indices[:n_take]:
                reg = roi_registry[int(col)]
                items.append(
                    {
                        "roi_idx": int(reg["roi_id"]),
                        "roi_name": str(reg["roi_name"]),
                        "family17": str(reg["family17"]),
                        "canonical8": str(reg["canonical8"]),
                        "canonical7": str(reg.get("canonical7", "")),
                        "fisher_z": float(row[int(col)]),
                    }
                )
            return items

        out[str(cidx + 1)] = {
            "top_positive": build_items(pos_order),
            "top_negative": build_items(neg_order),
        }
    return {"cluster_id_to_top20": out, "roi_registry": roi_registry}


def assignment_status_from_cluster_roi_topn(
    *,
    profile_3d: np.ndarray,
    roi_registry: list[dict[str, object]],
    q_thresh: float,
    top_n_cap: int,
    network_column: str = "canonical7",
) -> tuple[list[dict[str, object]], list[str | None], np.ndarray]:
    profile_3d = np.asarray(profile_3d, dtype=np.float32)
    if profile_3d.ndim != 3:
        raise ValueError(f"Expected 3D subjectwise profile, got shape={profile_3d.shape}")
    if profile_3d.shape[2] != len(roi_registry):
        raise ValueError(
            f"Profile/ROI registry mismatch: profile_cols={profile_3d.shape[2]} registry={len(roi_registry)}"
        )
    k_final = int(profile_3d.shape[1])
    mean_profile = np.nanmean(profile_3d, axis=0).astype(np.float32)
    normalized_rows = build_normalized_topn_count_vertex_rows(
        mean_profile=mean_profile,
        support_indices=np.arange(k_final, dtype=np.int32),
        roi_registry=roi_registry,
        top_n_cap=int(top_n_cap),
        network_column=network_column,
    )

    rows: list[dict[str, object]] = []
    dominant: list[str | None] = []
    for cidx in range(k_final):
        values = profile_3d[:, cidx, :]
        mean_z, _t_stat, p_pos, p_neg, n_vec = one_sample_one_sided_vector(values)
        q_pos = bh_fdr(p_pos)
        q_neg = bh_fdr(p_neg)
        sig_pos = np.flatnonzero(
            np.isfinite(q_pos) & (q_pos < float(q_thresh)) & np.isfinite(mean_z) & (mean_z > 0.0)
        )
        sig_neg = np.flatnonzero(
            np.isfinite(q_neg) & (q_neg < float(q_thresh)) & np.isfinite(mean_z) & (mean_z < 0.0)
        )
        sig_order = (
            sig_pos[np.argsort(-mean_z[sig_pos])]
            if sig_pos.size > 0
            else np.asarray([], dtype=np.int32)
        )
        sig_take = sig_order[: int(min(int(top_n_cap), int(sig_order.size)))]
        top_items: list[dict[str, object]] = []
        for col in sig_take:
            reg = roi_registry[int(col)]
            top_items.append(
                {
                    "roi_idx": int(reg["roi_id"]),
                    "roi_name": str(reg["roi_name"]),
                    "family17": str(reg["family17"]),
                    "canonical8": str(reg["canonical8"]),
                    "fisher_z": float(mean_z[int(col)]),
                    "q_positive": float(q_pos[int(col)]),
                }
            )

        row_out = dict(normalized_rows[cidx])
        row_out.pop("vertex_index", None)
        row_out.update(
            {
                "cluster_id": int(cidx + 1),
                "n_significant_positive": int(sig_pos.size),
                "n_significant_negative": int(sig_neg.size),
                "n_significant_any": int(sig_pos.size + sig_neg.size),
                "top_positive_selected": top_items,
                "n_subjects_non_nan_median": (
                    int(np.nanmedian(n_vec)) if np.isfinite(np.nanmedian(n_vec)) else 0
                ),
            }
        )
        rows.append(row_out)
        dominant.append(
            None
            if str(row_out.get("status", "")) == "unassigned"
            else str(row_out.get("top1_network", "unassigned"))
        )
    return rows, dominant, mean_profile


def method_uses_kong2022_topn_cluster_annotation(method: str, atlas: str) -> bool:
    # Gradient Fisher-z and normalized_top20_count must annotate the same
    # spatial clusters from the same 400-ROI profile; only assignment differs.
    return str(atlas) == "kong2022" and str(method) in ({
        "group-roi-sim-spectral",
        "gradient-spectral",
        "intrinsic-spectral",
    } | GRADIENT_METHODS | MULTIGRAPH_METHODS)


def assignment_status_from_profile(profile: np.ndarray, network_names: list[str]) -> tuple[list[dict[str, object]], list[str | None]]:
    rows: list[dict[str, object]] = []
    dominant: list[str | None] = []
    for cidx in range(profile.shape[0]):
        row = profile[cidx, :]
        finite = np.flatnonzero(np.isfinite(row))
        if finite.size == 0:
            rows.append({
                "cluster_id": int(cidx + 1), "status": "unassigned",
                "top1_network": None, "top1_fisher_z": float("nan"),
                "top2_network": None, "top2_fisher_z": float("nan"),
                "margin": float("nan"), "low_margin_lt_0p05": 0,
            })
            dominant.append(None)
            continue
        order = finite[np.argsort(row[finite])]
        top1_i = int(order[-1])
        top2_i = int(order[-2]) if order.size > 1 else None
        top1 = float(row[top1_i])
        top2 = float(row[top2_i]) if top2_i is not None else float("nan")
        margin = float(top1 - top2)
        rows.append(
            {
                "cluster_id": int(cidx + 1),
                "status": "dominant",
                "top1_network": str(network_names[top1_i]),
                "top1_fisher_z": top1,
                "top2_network": str(network_names[top2_i]) if top2_i is not None else None,
                "top2_fisher_z": top2,
                "margin": margin,
                "low_margin_lt_0p05": int(margin < 0.05),
            }
        )
        dominant.append(str(network_names[top1_i]))
    return rows, dominant


def build_fisherz_vertex_assignment_rows(
    *,
    mean_profile: np.ndarray,
    support_indices: np.ndarray,
    network_names: list[str],
    network_column: str = "network",
) -> list[dict[str, object]]:
    """Assign each supported vertex to its strongest finite network Fisher-z value."""
    arr = np.asarray(mean_profile, dtype=np.float32)
    idx = np.asarray(support_indices, dtype=np.int32).reshape(-1)
    names = [str(name) for name in network_names]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mean_profile, got shape={arr.shape}")
    if arr.shape[0] != idx.size:
        raise ValueError(f"Profile/support mismatch: profile={arr.shape[0]} support_indices={idx.size}")
    if arr.shape[1] != len(names):
        raise ValueError(f"Profile/network mismatch: profile_cols={arr.shape[1]} network_names={len(names)}")
    if not names:
        raise ValueError("Network names must not be empty")

    rows: list[dict[str, object]] = []
    for local_vidx, global_vidx in enumerate(idx):
        values = arr[local_vidx, :]
        finite_idx = np.flatnonzero(np.isfinite(values))
        if finite_idx.size == 0:
            rows.append(
                {
                    "vertex_index": int(global_vidx),
                    "status": "unassigned",
                    "unassigned_reason": "no_finite_network_fisher_z",
                    "top1_network": "unassigned",
                    "top1_fisher_z": None,
                    "top2_network": "unassigned",
                    "top2_fisher_z": None,
                    "margin": None,
                    "low_margin_lt_0p05": None,
                    "n_networks_available": 0,
                    "network_column": str(network_column),
                    "assignment_metric": "top1_fisher_z",
                }
            )
            continue
        ordered = finite_idx[np.argsort(-values[finite_idx], kind="stable")]
        top1_i = int(ordered[0])
        top2_i = int(ordered[1]) if ordered.size > 1 else top1_i
        top1 = float(values[top1_i])
        top2 = float(values[top2_i])
        margin = float(top1 - top2)
        rows.append(
            {
                "vertex_index": int(global_vidx),
                "status": "dominant",
                "top1_network": names[top1_i],
                "top1_fisher_z": top1,
                "top2_network": names[top2_i],
                "top2_fisher_z": top2,
                "margin": margin,
                "low_margin_lt_0p05": int(margin < 0.05),
                "n_networks_available": int(finite_idx.size),
                "network_column": str(network_column),
                "assignment_metric": "top1_fisher_z",
            }
        )
    return rows


def network_mean_profile_from_kong2022_roi_profile(
    roi_profile: np.ndarray,
    roi_registry: list[dict[str, object]],
    network_names: list[str],
    network_column: str = "canonical7",
) -> np.ndarray:
    cols_by_network, _network_names_from_registry = split_kong2022_columns_by_network(roi_registry, network_column=network_column)
    out = np.full((int(roi_profile.shape[0]), int(len(network_names))), np.nan, dtype=np.float32)
    for net_idx, net_name in enumerate(network_names):
        cols = cols_by_network.get(str(net_name), [])
        if not cols:
            continue
        out[:, net_idx] = np.nanmean(roi_profile[:, np.asarray(cols, dtype=np.int32)], axis=1).astype(np.float32)
    return out


def assignment_status_from_kong2022_roi_profile(
    roi_profile: np.ndarray,
    roi_registry: list[dict[str, object]],
    network_names: list[str],
    network_column: str = "canonical7",
) -> tuple[np.ndarray, list[dict[str, object]], list[str | None]]:
    profile = network_mean_profile_from_kong2022_roi_profile(roi_profile, roi_registry, network_names, network_column=network_column)
    status_rows, dominant = assignment_status_from_profile(profile, network_names)
    return profile, status_rows, dominant


def cluster_assignment_variants_from_kong2022_roi_profile(
    *,
    roi_profile: np.ndarray,
    roi_profile_3d: np.ndarray,
    roi_registry: list[dict[str, object]],
    network_names: list[str],
    q_thresh: float,
    top_n_cap: int,
    network_column: str = "canonical7",
) -> dict[str, dict[str, object]]:
    profile, fisherz_rows, fisherz_dominant = assignment_status_from_kong2022_roi_profile(
        roi_profile,
        roi_registry,
        network_names,
        network_column=network_column,
    )
    normalized_rows, normalized_dominant, _profile400_mean = assignment_status_from_cluster_roi_topn(
        profile_3d=roi_profile_3d,
        roi_registry=roi_registry,
        q_thresh=float(q_thresh),
        top_n_cap=int(top_n_cap),
        network_column=network_column,
    )
    return {
        "fisherz": {
            "profile": profile,
            "status_rows": fisherz_rows,
            "dominant": fisherz_dominant,
            "assignment_metric": "top1_fisher_z",
            "q_threshold": None,
            "top_n_cap": None,
        },
        "normalized_top20_count": {
            "profile": profile,
            "status_rows": normalized_rows,
            "dominant": normalized_dominant,
            "assignment_metric": "top20_count_normalized_by_network_roi_count",
            "q_threshold": float(q_thresh),
            "top_n_cap": int(top_n_cap),
        },
    }


def requested_ovr_variants(variants: list[str], *, atlas: str, method: str) -> list[str]:
    unknown = set(variants) - {"both", "fisherz", "normalized_top20_count", "network_ts_fisherz", *ALL_OVR_VARIANTS}
    if unknown:
        raise ValueError(f"Unknown assignment variants: {sorted(unknown)}")
    selected = [variant for variant in ALL_OVR_VARIANTS if variant in variants]
    if selected and (atlas != "kong2022" or is_clusterless_method(method)):
        raise ValueError("OVR requires the declared Kong2022 400-ROI subjectwise cluster profile contract")
    return selected


def build_ovr_assignment_specs(*, roi_profile_3d: np.ndarray, subjects: list[str],
                              cluster_ids: list[int], roi_registry: list[dict],
                              variants: list[str], out_dir: Path, observed_profiles=None) -> dict[str, dict]:
    specs = {}
    for variant in variants:
        if variant not in OVR_VARIANTS:
            continue
        taxonomy = variant.removeprefix("ovr_logistic_")
        model = fit_network_ovr(
            roi_profile_3d, subjects, cluster_ids, roi_registry, taxonomy=taxonomy,
            observed_profiles=observed_profiles,
        )
        paths = export_network_ovr(model, out_dir)
        specs[variant] = {
            "status_rows": model["assignment_rows"],
            "dominant": [row["top1_network"] for row in model["assignment_rows"]],
            "assignment_metric": OVR_ASSIGNMENT_METRIC, "q_threshold": None, "top_n_cap": None,
            "network_names": model["network_names"], "network_taxonomy": taxonomy,
            "network_profile": str(paths["beta"].resolve()),
            "profile_units": "raw_beta_per_Fisher_z_unit",
            "ovr_model_metadata": str(paths["metadata"].resolve()),
        }
    return specs


def annotate_kong2022_clusters_topn(
    *,
    group_labels_final: np.ndarray,
    support_mask: np.ndarray,
    subjects: list[str],
    shared_root: Path,
    surface_root: Path,
    reference_root: Path,
    smoothing: str,
    hemi: str,
    valid_masks_by_subject: dict[str, np.ndarray],
    network_names: list[str],
    q_thresh: float,
    top_n_cap: int,
    network_column: str = "canonical7",
    reference_atlas: str = "kong2022",
    profile_fc_mode: str = "cluster_mean_timeseries",
    profile_source_mode: str = "canonical",
    profile_source_root: Path | None = None,
    profile_source_registry: Path | None = None,
) -> tuple[
    np.ndarray,
    list[dict[str, object]],
    list[str | None],
    np.ndarray,
    list[dict[str, object]],
    list[str],
    dict[str, object],
    list[dict[str, object]],
    np.ndarray,
]:
    roi_profile_3d, roi_profile, roi_registry, roi_profile_subjects_used = cluster_roi_profile_subjectwise(
        group_labels_final=group_labels_final,
        support_mask=support_mask,
        subjects=subjects,
        shared_root=shared_root,
        surface_root=surface_root,
        reference_root=reference_root,
        smoothing=smoothing,
        hemi=hemi,
        valid_masks_by_subject=valid_masks_by_subject,
        reference_atlas=reference_atlas,
        profile_fc_mode=profile_fc_mode,
        profile_source_mode=profile_source_mode,
        profile_source_root=profile_source_root,
        profile_source_registry=profile_source_registry,
    )
    assignment_variants = cluster_assignment_variants_from_kong2022_roi_profile(
        roi_profile=roi_profile,
        roi_profile_3d=roi_profile_3d,
        roi_registry=roi_registry,
        network_names=network_names,
        q_thresh=float(q_thresh),
        top_n_cap=int(top_n_cap),
        network_column=network_column,
    )
    fisherz_variant = assignment_variants["fisherz"]
    normalized_variant = assignment_variants["normalized_top20_count"]
    profile = np.asarray(fisherz_variant["profile"], dtype=np.float32)
    status_rows = [dict(x) for x in fisherz_variant["status_rows"]]
    dominant = list(fisherz_variant["dominant"])
    roi_normalized_status_rows = [dict(x) for x in normalized_variant["status_rows"]]
    roi_top20 = cluster_roi_top_k(roi_profile, roi_registry, k_top=int(top_n_cap))
    return profile, status_rows, dominant, roi_profile, roi_registry, roi_profile_subjects_used, roi_top20, roi_normalized_status_rows, roi_profile_3d


def build_global_top_network_connections(
    *,
    mean_profile: np.ndarray,
    support_indices: np.ndarray,
    network_names: list[str],
) -> dict[str, object]:
    """Summarize direct Lynch vertex-to-network Fisher-z assignments."""
    rows = build_fisherz_vertex_assignment_rows(
        mean_profile=mean_profile,
        support_indices=support_indices,
        network_names=network_names,
    )
    positive: list[dict[str, object]] = []
    negative: list[dict[str, object]] = []
    for row in rows:
        score = row.get("top1_fisher_z")
        if not isinstance(score, (int, float)) or not np.isfinite(float(score)):
            continue
        item = {
            "vertex_index": int(row["vertex_index"]),
            "network": str(row["top1_network"]),
            "fisher_z": float(score),
            "margin": float(row["margin"]),
        }
        if float(score) > 0.0:
            positive.append(item)
        elif float(score) < 0.0:
            negative.append(item)
    positive.sort(key=lambda item: float(item["fisher_z"]), reverse=True)
    negative.sort(key=lambda item: float(item["fisher_z"]))
    return {
        "assignment_metric": "top1_fisher_z",
        "network_names": [str(name) for name in network_names],
        "n_support_vertices": int(len(rows)),
        "n_positive_assignments": int(len(positive)),
        "n_negative_assignments": int(len(negative)),
        "top_positive": positive,
        "top_negative": negative,
    }


def build_global_top_connections(
    *,
    profile_3d: np.ndarray,
    support_indices: np.ndarray,
    roi_registry: list[dict[str, object]],
    q_thresh: float,
    top_n_cap: int,
    network_column: str = "canonical7",
) -> dict[str, object]:
    network_registry_column = resolve_kong2022_network_registry_column(network_column)
    pos_rows: list[dict[str, object]] = []
    neg_rows: list[dict[str, object]] = []
    for local_vidx, global_vidx in enumerate(support_indices):
        values = profile_3d[:, local_vidx, :]
        mean_z, _t_stat, p_pos, p_neg, _n = one_sample_one_sided_vector(values)
        q_pos = bh_fdr(p_pos)
        q_neg = bh_fdr(p_neg)
        sig_pos = np.flatnonzero(np.isfinite(q_pos) & (q_pos < float(q_thresh)) & np.isfinite(mean_z) & (mean_z > 0.0))
        sig_neg = np.flatnonzero(np.isfinite(q_neg) & (q_neg < float(q_thresh)) & np.isfinite(mean_z) & (mean_z < 0.0))
        for col in sig_pos:
            reg = roi_registry[int(col)]
            item = {
                    "vertex_index": int(global_vidx),
                    "roi_idx": int(reg["roi_id"]),
                    "roi_name": str(reg["roi_name"]),
                    "family17": str(reg["family17"]),
                    "sign": "positive",
                    "fisher_z": float(mean_z[int(col)]),
                    "q_value": float(q_pos[int(col)]),
                }
            item[network_registry_column] = str(reg[network_registry_column])
            pos_rows.append(item)
        for col in sig_neg:
            reg = roi_registry[int(col)]
            item = {
                    "vertex_index": int(global_vidx),
                    "roi_idx": int(reg["roi_id"]),
                    "roi_name": str(reg["roi_name"]),
                    "family17": str(reg["family17"]),
                    "sign": "negative",
                    "fisher_z": float(mean_z[int(col)]),
                    "q_value": float(q_neg[int(col)]),
                }
            item[network_registry_column] = str(reg[network_registry_column])
            neg_rows.append(item)
    pos_rows.sort(key=lambda r: float(r["fisher_z"]), reverse=True)
    neg_rows.sort(key=lambda r: float(r["fisher_z"]))
    pos_take = pos_rows[: int(min(int(top_n_cap), len(pos_rows)))]
    neg_take = neg_rows[: int(min(int(top_n_cap), len(neg_rows)))]
    return {
        "top_n_cap": int(top_n_cap),
        "q_threshold": float(q_thresh),
        "network_column": resolve_kong2022_network_registry_column(network_column),
        "n_candidates_positive": int(len(pos_rows)),
        "n_candidates_negative": int(len(neg_rows)),
        "top_positive": pos_take,
        "top_negative": neg_take,
    }


def _functional_atlas_label_items(
    network_names: list[str],
    color_map: dict[str, tuple[int, int, int, int]],
    key_unassigned: int,
) -> list[tuple[int, str, tuple[int, int, int, int]]]:
    label_items: list[tuple[int, str, tuple[int, int, int, int]]] = [(0, "Null", (158, 158, 158, 255))]
    for i, n in enumerate(network_names, start=1):
        name = str(n)
        if name not in color_map:
            raise KeyError(f"Missing network color for {name!r}")
        label_items.append((int(i), name, color_map[name]))
    label_items.append((int(key_unassigned), "unassigned", (20, 20, 20, 255)))
    return label_items


def functional_atlas_from_clusters(
    *,
    group_labels_final: np.ndarray,
    status_rows: list[dict[str, object]],
    network_names: list[str],
    color_map: dict[str, tuple[int, int, int, int]],
) -> tuple[np.ndarray, dict[str, object], list[tuple[int, str, tuple[int, int, int, int]]]]:
    net_to_key = {str(n): int(i + 1) for i, n in enumerate(network_names)}
    key_unassigned = int(len(network_names) + 1)
    cluster_to_key: dict[int, int] = {}

    for row in status_rows:
        cid = int(row["cluster_id"])
        status = str(row["status"])
        top1 = str(row["top1_network"])
        if status == "dominant" or status == "mixed":
            if top1 not in net_to_key:
                raise KeyError(f"Unknown network in assignment status: {top1}")
            cluster_to_key[cid] = int(net_to_key[top1])
        elif status == "unassigned":
            cluster_to_key[cid] = key_unassigned
        else:
            raise ValueError(f"Unknown status in assignment status: {status}")

    out = np.zeros_like(group_labels_final, dtype=np.int32)
    for cid, key in cluster_to_key.items():
        out[group_labels_final == int(cid)] = int(key)

    label_items = _functional_atlas_label_items(network_names, color_map, key_unassigned)

    meta = {
        "network_names": [str(x) for x in network_names],
        "network_label_keys": {str(n): int(i + 1) for i, n in enumerate(network_names)},
        "unassigned_label_key": int(key_unassigned),
        "cluster_to_functional_label_key": {str(k): int(v) for k, v in sorted(cluster_to_key.items())},
    }
    return out, meta, label_items


def functional_atlas_from_vertices(
    *,
    n_vertices: int,
    status_rows: list[dict[str, object]],
    network_names: list[str],
    color_map: dict[str, tuple[int, int, int, int]],
) -> tuple[np.ndarray, dict[str, object], list[tuple[int, str, tuple[int, int, int, int]]]]:
    net_to_key = {str(n): int(i + 1) for i, n in enumerate(network_names)}
    key_unassigned = int(len(network_names) + 1)
    out = np.zeros((int(n_vertices),), dtype=np.int32)
    for row in status_rows:
        vidx = int(row["vertex_index"])
        status = str(row.get("status", "unassigned"))
        top1 = str(row.get("top1_network", "unassigned"))
        if status == "dominant":
            if top1 not in net_to_key:
                raise ValueError(
                    f"dominant WTA assignment has no canonical network label: {top1!r}"
                )
            out[vidx] = int(net_to_key[top1])
        elif status == "unassigned":
            out[vidx] = int(key_unassigned)
        else:
            raise ValueError(f"Unknown vertex WTA assignment status: {status!r}")

    label_items = _functional_atlas_label_items(network_names, color_map, key_unassigned)

    meta = {
        "assignment_mode": "vertex_level",
        "network_names": [str(x) for x in network_names],
        "network_label_keys": {str(n): int(i + 1) for i, n in enumerate(network_names)},
        "unassigned_label_key": int(key_unassigned),
    }
    return out, meta, label_items


def prepare_combo_inputs(
    *,
    method: str,
    atlas: str,
    smoothing: str,
    hemi: str,
    shared_root: Path,
    surface_root: Path,
    profile_surface_root: Path,
    reference_root: Path,
    surfaces: dict[str, Path],
    subjects: list[str],
    subject_fc_mode: SubjectFCMode,
    k_selection_mode: str,
    network_taxonomy: str = "canonical7",
) -> PreparedComboInputs:
    valid_masks = {s: subject_valid_mask(surface_root, shared_root, s, smoothing, hemi) for s in subjects}
    n_vertices = int(next(iter(valid_masks.values())).size)
    coverage_count = np.sum(np.stack([valid_masks[s] for s in subjects], axis=0), axis=0).astype(np.int32)
    coverage_fraction = np.mean(np.stack([valid_masks[s] for s in subjects], axis=0), axis=0).astype(np.float32)

    feature_bank: dict[str, np.ndarray] = {}
    actual_input_paths: dict[str, str] = {}
    for s in subjects:
        if method_uses_network_fc_input(method):
            p = shared_root / f"sub-{s}" / "fc" / atlas / smoothing / f"hemi_{hemi}" / network_feature_filename(method, atlas)
            actual_input_paths[s] = str(p)
            raw_feature = subject_network_feature(
                shared_root, s, atlas, smoothing, hemi, method, reference_root=reference_root
            )
            pre_feature = preprocess_subject_fc_matrix(
                method=method,
                fc=raw_feature,
                subject_fc_mode=subject_fc_mode,
            )
            if uses_subject_similarity_bank(method):
                feature_bank[s] = subject_similarity_feature(pre_feature, valid_masks[s], source=p)
            else:
                feature_bank[s] = roi_similarity_from_network_fc(pre_feature) if is_roi_similarity_method(method) else pre_feature
        else:
            p = resolve_intrinsic_feature_path(shared_root, s, smoothing, hemi)
            actual_input_paths[s] = str(p)
            raw_feature = subject_intrinsic_feature(shared_root, s, atlas, smoothing, hemi, valid_masks[s])
            feature_bank[s] = preprocess_subject_fc_matrix(
                method=method,
                fc=raw_feature,
                subject_fc_mode=subject_fc_mode,
            )

    run_feature_banks = (
        runwise_feature_bank(
            method=method,
            shared_root=shared_root,
            subjects=subjects,
            atlas=atlas,
            smoothing=smoothing,
            hemi=hemi,
            valid_masks=valid_masks,
            subject_fc_mode=subject_fc_mode,
            reference_root=reference_root,
        )
        if not is_clusterless_method(method)
        else {}
    )
    roi_fc_bank_for_redundancy: dict[str, np.ndarray] | None = None
    if not is_clusterless_method(method):
        roi_fc_bank_for_redundancy = {}
        for s in subjects:
            if atlas == "kong2022" and method in ({"network-spectral", "gradient-spectral"} | GRADIENT_METHODS) and not uses_subject_similarity_bank(method):
                roi_fc_bank_for_redundancy[s] = feature_bank[s]
            else:
                redundancy_atlas = (
                    atlas
                    if is_lynch2024_atlas(atlas)
                    else "kong2022"
                )
                roi_raw = subject_network_feature(
                    shared_root,
                    s,
                    redundancy_atlas,
                    smoothing,
                    hemi,
                    "wta-top" if redundancy_atlas == "kong2022" else "network-spectral",
                    reference_root=reference_root,
                )
                if method_uses_subject_fc_preprocessing(method):
                    roi_fc_bank_for_redundancy[s] = preprocess_subject_fc_matrix(
                        method="network-spectral",
                        fc=roi_raw,
                        subject_fc_mode=subject_fc_mode,
                    )
                else:
                    roi_fc_bank_for_redundancy[s] = roi_raw

    subject_fc_preprocessing_for_resume = subject_fc_preprocessing_manifest_block(method, subject_fc_mode)
    valid_mask_paths = [
        surface_root / f"sub-{s}" / "surface" / "validity" / f"sub-{s}_hemi-{hemi}_valid_mask.npy"
        for s in subjects
    ]
    runwise_feature_paths: list[Path] = []
    if not is_clusterless_method(method):
        active_run_ids = common_fc_run_ids(shared_root, subjects, atlas, smoothing, hemi)
        for rid in active_run_ids:
            for s in subjects:
                if method_uses_network_fc_input(method):
                    p = (
                        shared_root
                        / f"sub-{s}"
                        / "fc"
                        / atlas
                        / smoothing
                        / f"hemi_{hemi}"
                        / "runs"
                        / f"run-{rid}"
                        / runwise_network_feature_filename(method, atlas)
                    )
                else:
                    p = resolve_intrinsic_feature_path(shared_root, s, smoothing, hemi, rid)
                runwise_feature_paths.append(p)
    reference_network_order_paths = [
        reference_path(reference_root, s, atlas, network_reference_filenames(network_taxonomy)[0])
        for s in subjects
    ]
    reference_network_ts_paths = [
        reference_path(reference_root, s, atlas, network_reference_filenames(network_taxonomy)[1])
        for s in subjects
    ]
    surface_timeseries_paths = [
        profile_surface_root / f"sub-{s}" / "surface" / smoothing / f"sub-{s}_hemi-{hemi}_timeseries.npy"
        for s in subjects
    ]
    kong2022_roi_reference_paths: list[Path] = []
    if atlas in {"kong2022"}:
        for s in subjects:
            kong2022_roi_reference_paths.extend(
                [
                    reference_path(reference_root, s, atlas, "cortex_roi_parcel_timeseries.npy"),
                    reference_path(reference_root, s, atlas, "cortex_roi_parcels.tsv"),
                ]
            )
    resume_file_state_payload = {
        "primary_feature_inputs": file_state_list([Path(p) for p in actual_input_paths.values()]),
        "valid_masks": file_state_list(valid_mask_paths),
        "runwise_feature_inputs": file_state_list(runwise_feature_paths),
        "reference_network_order_tsv": file_state_list(reference_network_order_paths),
        "reference_network_timeseries": file_state_list(reference_network_ts_paths),
        "surface_timeseries": file_state_list(surface_timeseries_paths),
        "kong2022_roi_reference_inputs": file_state_list(kong2022_roi_reference_paths),
        "template_surface": file_state_list([surfaces[hemi]]),
    }
    resume_input_category_hashes = {
        k: canonical_json_sha256(v) for k, v in resume_file_state_payload.items()
    }
    return PreparedComboInputs(
        n_vertices=n_vertices,
        valid_masks=valid_masks,
        coverage_count=coverage_count,
        coverage_fraction=coverage_fraction,
        feature_bank=feature_bank,
        actual_input_paths=actual_input_paths,
        run_feature_banks=run_feature_banks,
        roi_fc_bank_for_redundancy=roi_fc_bank_for_redundancy,
        subject_fc_preprocessing_for_resume=subject_fc_preprocessing_for_resume,
        resume_input_category_hashes=resume_input_category_hashes,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run merged canonical group hippocampal atlas (v3 plan).")
    parser.add_argument("--shared-root", default=str(DEFAULT_SHARED_ROOT))
    parser.add_argument("--surface-root", default=str(DEFAULT_SURFACE_ROOT))
    parser.add_argument(
        "--profile-surface-root",
        default=None,
        help=(
            "Root containing sub-*/surface/<smoothing>/sub-*_hemi-*_timeseries.npy "
            "for cluster-network profile annotation. Defaults to the CtxReg shared "
            "root for ctxreg routes and to --surface-root otherwise. The root must "
            "match the route's signal branch; non-CtxReg routes cannot read "
            "CtxReg residualized time series."
        ),
    )
    parser.add_argument(
        "--profile-source-mode",
        choices=["canonical", "original-denoised"],
        default="canonical",
        help=(
            "Cluster ROI profile source. The original-denoised mode is an explicit "
            "Kong2022 recovery path and requires --profile-source-root plus "
            "--profile-source-registry; it never falls back implicitly."
        ),
    )
    parser.add_argument(
        "--profile-source-root",
        default=None,
        help="Approved fixed-template subject root for --profile-source-mode original-denoised.",
    )
    parser.add_argument(
        "--profile-source-registry",
        default=None,
        help="Verified ordered 400-ROI TSV for --profile-source-mode original-denoised.",
    )
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--comparison-output", action="store_true", help="Explicit dataset comparison written under this checkout's tmp directory.")
    parser.add_argument(
        "--dataset", choices=["auto", "outputs_hcp", "outputs_stanford", "outputs_iowa_jj_tms"], default="auto",
        help="Dataset provenance recorded in generated manifests; auto infers it from --out-root.",
    )
    parser.add_argument("--reference-root", default=None, help="Read cortex reference metadata/time series from this root; defaults to --shared-root.")
    parser.add_argument("--gradient-root", default=str(DEFAULT_GRADIENT_ROOT))
    parser.add_argument("--cohort", default=None, help="Cohort directory below the selected gradient route; omit for HCP.")
    parser.add_argument("--variant", choices=("36p",), default="36p")
    parser.add_argument("--hippomaps-root", default=str(DEFAULT_HIPPOMAPS_ROOT))
    parser.add_argument("--route-tag", choices=["auto", *ROUTE_TAG_CHOICES], default="auto")
    parser.add_argument(
        "--signal-branch",
        choices=SIGNAL_BRANCH_CHOICES,
        default=None,
        help="Explicit signal branch for pfm (standard), pfm_ctxreg (ctxreg) or pfm_ctxreg_10mm (ctxreg_10mm).",
    )
    class _StoreExplicitNetworkTaxonomy(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, values)
            setattr(namespace, "_network_taxonomy_explicit", True)

    parser.add_argument(
        "--network-taxonomy",
        choices=NETWORK_TAXONOMIES,
        default="canonical7",
        action=_StoreExplicitNetworkTaxonomy,
    )
    parser.add_argument(
        "--reuse-fixed-cluster-root",
        default=None,
        help=(
            "Reuse an existing fixed-K cluster result for a taxonomy-only branch. "
            "The source run is located by its manifest; gradients, clustering, and K selection are not rerun."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow replacement of non-empty canonical7 target runs.")
    parser.add_argument("--subjects-manifest", default=str(DEFAULT_SUBJECTS))
    parser.add_argument("--hippunfold-batch-root", default=str(DEFAULT_HIPPUNFOLD_BATCH_ROOT))
    parser.add_argument("--kong2022-dlabel", default=None)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        choices=METHOD_CHOICES,
        help="Methods to run. If omitted, Lynch atlases use gradient-spectral and direct-network wta-top; Kong2022 uses METHODS_DEFAULT including normalized_top20_count WTA.",
    )
    parser.add_argument("--atlases", nargs="+", default=DEFAULT_ATLASES, choices=ATLASES)
    parser.add_argument("--smoothings", nargs="+", default=SMOOTHINGS)
    parser.add_argument("--hemis", nargs="+", default=HEMIS)
    parser.add_argument(
        "--alignment-anchor-smoothing",
        choices=SMOOTHINGS,
        default="4mm",
        help="Smoothing level of the left-hemisphere reference anchor for gradient component and sign alignment.",
    )
    parser.add_argument("--coverage-thresholds", nargs="+", type=float, default=[0.80])
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument(
        "--force-final-k",
        type=int,
        default=None,
        help="Explicitly override the primary output K after candidate K evaluation.",
    )
    parser.add_argument(
        "--default-k",
        type=int,
        default=3,
        help="Primary K for rendered and exported labels; automatic K selection remains diagnostic evidence.",
    )
    parser.add_argument(
        "--undersized-default-k-fallback",
        type=int,
        default=None,
        help="Export this K when the --default-k partition has a parcel below the minimum parcel size. "
             "The fallback partition must meet the size gate; the manifest records the replaced K.",
    )
    parser.add_argument(
        "--automatic-primary-k",
        action="store_true",
        help="Export the automatically selected K as the primary partition. A combo in which no K "
             "passes the selection gates is recorded as skipped. Cannot be combined with --default-k "
             "or --force-final-k.",
    )
    parser.add_argument("--instability-resamples", type=int, default=100)
    parser.add_argument("--null-resamples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260420)
    parser.add_argument("--assignment-variant", nargs="+", default=["both"],
                        choices=["both", "fisherz", "normalized_top20_count", "network_ts_fisherz", *ALL_OVR_VARIANTS],
                        help="Fisher-z and Top-20 retain their existing exports; explicitly select OVR taxonomies to add regression annotations.")
    parser.add_argument(
        "--profile",
        choices=GRADIENT_PROFILE_CHOICES,
        default=None,
        help="HCP PFM gradient profile name ('topfc' or 'topsim'). Automatically configures feature scaling, sparsify stage, sparsity, clustering scaling, and default roots.",
    )
    parser.add_argument(
        "--redundancy-threshold",
        type=float,
        default=None,
        help="Cluster redundancy maximum profile similarity threshold (default: 0.90 for active profile, 0.80 otherwise).",
    )
    parser.add_argument(
        "--resampling-workers",
        type=int,
        default=8,
        help="Number of worker processes for split/null resampling gradient fitting (default: 8, bounded by IDAS cap 16).",
    )
    parser.add_argument("--v-min-fraction", type=float, default=0.03)
    parser.add_argument("--max-components-threshold", type=int, default=3)
    parser.add_argument("--min-profile-coherence", type=float, default=0.0)
    parser.add_argument(
        "--subject-fc-mode",
        default="full",
        choices=SUBJECT_FC_MODE_CHOICES,
        help="Per-subject FC preprocessing mode for spectral methods: full (no sparsification) or top-percent (Top integer %% + retained entries binarized to 1).",
    )
    parser.add_argument(
        "--subject-fc-top-percent",
        type=int,
        default=None,
        help="Required only when --subject-fc-mode=top-percent. Integer percent in (0,100].",
    )
    parser.add_argument(
        "--gradient-components",
        default="1,2,3",
        help="Comma-separated gradient components for gradient-spectral clustering. "
             "Supported examples: 1, 2, 3, 1,2, 1,2,3. Default: 1,2,3. "
             "05_01 writes aligned G1-G5 via BrainSpace/HippoMaps; this option selects clustering columns.",
    )
    parser.add_argument("--feature-scaling", choices=("none", "col_zscore"), default="col_zscore")
    parser.add_argument("--sparsify-stage", choices=("features", "affinity", "none"), default="features")
    parser.add_argument("--sparsity", type=float, default=None)
    parser.add_argument("--rbf-scale", type=float, default=1.0)
    parser.add_argument("--clustering-feature-scaling", choices=("none", "unit_sd"), default="none")
    parser.add_argument("--resume", action="store_true", help="Skip combos that already have group_atlas_manifest.json.")
    parser.add_argument("--k-selection-mode", default="redundancy-aware", choices=["redundancy-aware", "fixed-comparison"])
    parser.add_argument(
        "--k-final-rule", default=K_FINAL_RULES[0], choices=K_FINAL_RULES,
        help="Automatic K among the candidates that pass the size, connectedness, coherence and redundancy gates: "
             "the smallest split-half ARI local maximum within one standard error of the best (default), or the "
             "largest passing K with split-half ARI kept as a diagnostic only.",
    )
    parser.add_argument(
        "--redundancy-early-stop",
        action="store_true",
        help="Heuristic speed mode for redundancy-aware runs: evaluate K in ascending order and stop once the first redundant K is observed; larger K are skipped.",
    )
    parser.add_argument("--overview-threshold", type=float, default=0.80)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Maximum parallel K evaluations per (method, atlas, smoothing, hemi, coverage) combo; capped at 10.",
    )
    parser.add_argument(
        "--group-average-surfaces-dir",
        default=str(DEFAULT_GROUP_AVG_SURFACES_DIR),
        help="Directory containing group-average template midthickness surfaces named "
             "group-average_hemi-{L|R}_space-corobl_den-512_label-hipp_midthickness.surf.gii.",
    )
    return parser


def infer_route_tag(
    shared_root: Path,
    explicit: str = "auto",
    network_taxonomy: str | None = None,
) -> str:
    if network_taxonomy is None:
        if str(explicit).strip().lower() == "auto":
            return infer_route_tag_from_shared_root(Path(shared_root))
        return str(explicit).strip().lower()
    return resolve_route_tag_for_group_atlas_run(
        shared_root=Path(shared_root),
        explicit_route=str(explicit),
        network_taxonomy=str(network_taxonomy),
    )


def _resolve_methods_and_taxonomy(args: argparse.Namespace) -> str:
    """Apply --methods/--network-taxonomy defaults and enforce the retired-taxonomy guard."""
    route = str(args.route_tag)
    if route == "auto":
        route = infer_route_tag_from_shared_root(Path(args.shared_root))
    if args.methods is None:
        args.methods = list(
            dict.fromkeys(
                method
                for atlas in args.atlases
                for method in default_methods_for_atlas(atlas)
            )
        )
    network_taxonomy = str(args.network_taxonomy)
    if not getattr(args, "_network_taxonomy_explicit", False):
        network_taxonomy = default_network_taxonomy_for_atlases(args.atlases)
        args.network_taxonomy = network_taxonomy
    if network_taxonomy == "canonical8":
        raise ValueError(
            "The canonical8 taxonomy is retired for active production; use canonical7 instead."
        )
    validate_requested_methods_for_taxonomy(args.methods, network_taxonomy, args.atlases)
    for atlas in args.atlases:
        for method in args.methods:
            requested_ovr_variants(args.assignment_variant, atlas=atlas, method=method)
    return network_taxonomy


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.profile is not None:
        p_cfg = gradient_profile_config(args.profile)
        args.feature_scaling = str(p_cfg["feature_scaling"])
        args.sparsify_stage = str(p_cfg["sparsify_stage"])
        args.sparsity = float(p_cfg["sparsity"])
        args.clustering_feature_scaling = "none"
        if not option_was_given("--alignment-anchor-smoothing"):
            args.alignment_anchor_smoothing = "2mm"

    gradient_options = {
        "feature_scaling": args.feature_scaling,
        "sparsify_stage": args.sparsify_stage,
        "sparsity": args.sparsity,
        "clustering_feature_scaling": args.clustering_feature_scaling,
        "rbf_scale": args.rbf_scale,
        "alignment_anchor_smoothing": args.alignment_anchor_smoothing,
    }
    suppress_known_runtime_warnings()
    print(
        "[thread-cap] "
        + " ".join(f"{name}={os.environ.get(name, "")}" for name in NATIVE_THREAD_ENV_VARS),
        flush=True,
    )

    global KONG2022_DLABEL_OVERRIDE, HIPPOMAPS_ROOT_OVERRIDE
    KONG2022_DLABEL_OVERRIDE = Path(args.kong2022_dlabel).resolve() if args.kong2022_dlabel else None
    HIPPOMAPS_ROOT_OVERRIDE = Path(args.hippomaps_root).resolve()

    shared_root = Path(args.shared_root).resolve()
    surface_root = Path(args.surface_root).resolve()
    profile_source_mode = str(args.profile_source_mode)
    profile_source_root = Path(args.profile_source_root).resolve() if args.profile_source_root else None
    profile_source_registry = Path(args.profile_source_registry).resolve() if args.profile_source_registry else None
    if profile_source_mode == "original-denoised":
        if "kong2022" not in {str(atlas) for atlas in args.atlases}:
            raise ValueError("--profile-source-mode original-denoised is supported only for --atlases kong2022")
        if profile_source_root is None or profile_source_registry is None:
            raise ValueError(
                "--profile-source-mode original-denoised requires both "
                "--profile-source-root and --profile-source-registry"
            )
    elif profile_source_root is not None or profile_source_registry is not None:
        raise ValueError(
            "--profile-source-root/--profile-source-registry require "
            "--profile-source-mode original-denoised"
        )
    network_taxonomy = _resolve_methods_and_taxonomy(args)
    route_tag = resolve_route_tag_for_group_atlas_run(
        shared_root=shared_root,
        explicit_route=str(args.route_tag),
        network_taxonomy=network_taxonomy,
    )
    signal_branch = resolve_signal_branch_for_route(route_tag, args.signal_branch)
    if args.profile is not None:
        for m in args.methods:
            validate_gradient_combination(args.profile, m)

    out_root, out_root_adjusted = resolve_group_atlas_out_root(
        args.out_root,
        route_tag,
        default_routea_out_root_selected=args.out_root == str(DEFAULT_ROUTEA_GROUP_ATLAS_ROOT),
    )
    if args.profile is not None and (args.out_root == str(DEFAULT_OUT_ROOT) or args.out_root == str(DEFAULT_ROUTEA_GROUP_ATLAS_ROOT)):
        out_root = resolve_cohort_group_atlas_root(dataset="hcp", route=route_tag, variant=args.variant, profile=args.profile)
    if out_root_adjusted:
        print(f"[info] Automatically adjusted --out-root to matching route directory: {out_root}", flush=True)
    validate_group_atlas_root_for_route(out_root, route_tag)
    dataset, expected_source_root = validate_dataset_output_contract(
        out_root, str(args.dataset), comparison=args.comparison_output,
        source_roots=[shared_root, surface_root, Path(args.reference_root).resolve() if args.reference_root else shared_root],
    )

    if args.reuse_fixed_cluster_root is not None:
        if route_tag == "lynch2024_7net":
            allowed_taxonomies = {"lynch7"}
        elif route_tag in {"pfm", *CTXREG_ROUTES}:
            # canonical8 is retired upstream (see _resolve_methods_and_taxonomy),
            # so it can never reach this reuse-fixed validation.
            allowed_taxonomies = {"canonical7"}
        elif route_tag in FAMILY17_ROUTE_TAGS:
            allowed_taxonomies = {"family17"}
        else:
            raise ValueError(
                "--reuse-fixed-cluster-root is supported only for explicit Kong PFM/control, "
                "Kong family17, or Lynch canonical7 routes"
            )
        if network_taxonomy not in allowed_taxonomies:
            raise ValueError(
                f"--reuse-fixed-cluster-root taxonomy mismatch: route={route_tag} taxonomy={network_taxonomy} "
                f"allowed={sorted(allowed_taxonomies)}"
            )
        if not any(is_gradient_method(m) for m in args.methods):
            raise ValueError("--reuse-fixed-cluster-root requires a gradient method in --methods")

    out_root.mkdir(parents=True, exist_ok=True)
    reference_root = (
        Path(args.reference_root).resolve()
        if args.reference_root
        else standard_reference_root_for_signal_branch(shared_root, signal_branch)
    )
    profile_surface_root = (
        Path(args.profile_surface_root).resolve()
        if args.profile_surface_root
        else shared_root if is_ctxreg_signal_branch(signal_branch) else surface_root
    )
    profile_surface_root = validate_profile_surface_root_for_signal_branch(
        profile_surface_root,
        signal_branch=signal_branch,
        shared_root=shared_root,
    )
    signal_provenance = {
        "signal_branch": signal_branch,
        "signal_control_route": route_tag if signal_branch is not None else None,
        "signal_root": str(shared_root.resolve()) if signal_branch is not None else None,
    }
    gradient_root = Path(args.gradient_root).resolve()
    gradient_route_tag = gradient_route_for_route_tag(route_tag)
    hippunfold_batch_root = Path(args.hippunfold_batch_root).resolve()

    redundancy_threshold = (
        float(args.redundancy_threshold)
        if args.redundancy_threshold is not None
        else (0.90 if args.profile is not None else float(PROFILE_REDUNDANCY_MAX_SIM))
    )

    subjects_manifest_path = Path(args.subjects_manifest).resolve()
    subjects = load_subject_ids(subjects_manifest_path)
    subject_fc_mode = resolve_subject_fc_mode(args.subject_fc_mode, args.subject_fc_top_percent)
    gradient_components = parse_gradient_components(args.gradient_components)
    coverage_thresholds = [float(x) for x in args.coverage_thresholds]
    k_values = comparison_k_values(args)
    if not k_values:
        raise ValueError(f"Empty K range: --k-min={args.k_min} --k-max={args.k_max}")
    if args.force_final_k is not None and int(args.force_final_k) not in k_values:
        raise ValueError(
            f"--force-final-k must be within requested K range [{min(k_values)}, {max(k_values)}], "
            f"got {args.force_final_k}"
        )
    resolve_primary_k_options(args, k_values)
    if args.default_k is not None and int(args.default_k) not in k_values:
        raise ValueError(
            f"--default-k must be within requested K range [{min(k_values)}, {max(k_values)}], "
            f"got {args.default_k}"
        )
    if args.profile is not None:
        split_seeds = [split_seed_for_index(i, int(args.seed)) for i in range(20)]
        null_seeds = [null_seed_for_index(i, int(args.seed)) for i in range(20)]
    else:
        rng = np.random.RandomState(int(args.seed))
        split_seeds = [int(x) for x in rng.randint(1, 2_000_000_000, size=int(args.instability_resamples))]
        null_seeds = [int(x) for x in rng.randint(1, 2_000_000_000, size=int(args.null_resamples))]
    config = GroupAtlasConfig(
        args=args,
        shared_root=shared_root,
        surface_root=surface_root,
        profile_surface_root=profile_surface_root,
        out_root=out_root,
        reference_root=reference_root,
        gradient_root=gradient_root,
        route_tag=route_tag,
        gradient_route_tag=gradient_route_tag,
        signal_branch=signal_branch,
        network_taxonomy=network_taxonomy,
        hippunfold_batch_root=hippunfold_batch_root,
        subjects_manifest_path=subjects_manifest_path,
        subjects=subjects,
        subject_fc_mode=subject_fc_mode,
        gradient_components=gradient_components,
        coverage_thresholds=coverage_thresholds,
        k_values=k_values,
        split_seeds=split_seeds,
        null_seeds=null_seeds,
    )

    group_avg_dir = Path(args.group_average_surfaces_dir).resolve()
    surfaces = {
        hemi: group_avg_dir / f"group-average_hemi-{hemi}_space-corobl_den-512_label-hipp_midthickness.surf.gii"
        for hemi in args.hemis
    }
    missing_template = [str(p) for p in surfaces.values() if not p.exists()]
    if missing_template:
        raise FileNotFoundError(
            "Missing required group-average template surfaces. "
            "Build template first (prepare_t1only_group_inputs.py --stage group-average) and point "
            "--group-average-surfaces-dir to the shared template directory.\n"
            + "\n".join(missing_template)
        )
    template_label = "group-average"
    template_source_mode = "group_average_template"
    template_source_note = (
        "Template-first mode: clustering topology/order is derived from shared group-average template surfaces; "
        "manifest subjects are validated for topology compatibility."
    )
    template_render_subject = f"sub-{subjects[0]}"
    template_spec_path = infer_spec_path(
        subject=subjects[0],
        surface_path=infer_surface_path(subjects[0], args.hemis[0], hippunfold_batch_root),
    )
    topology_audit = verify_subject_topology_against_template(
        subjects,
        list(args.hemis),
        surfaces,
        hippunfold_batch_root,
    )
    long_axis_order_by_hemi: dict[str, np.ndarray] = {
        hemi: long_axis_order(surfaces[hemi]) for hemi in args.hemis
    }
    surface_meta: dict[str, dict[str, object]] = {}
    adjacency_by_hemi: dict[str, sparse.csr_matrix] = {}
    for hemi in config.args.hemis:
        coords, faces = load_surface(surfaces[hemi])
        adjacency_by_hemi[hemi] = build_surface_adjacency(faces, int(coords.shape[0]))
        surface_meta[hemi] = {
            "surface_path": str(surfaces[hemi]),
            "n_vertices": int(coords.shape[0]),
            "n_faces": int(faces.shape[0]),
        }

    if "intrinsic-spectral" in args.methods:
        for smoothing in config.args.smoothings:
            for hemi in config.args.hemis:
                print(f"[check] intrinsic atlas-invariance smoothing={smoothing} hemi={hemi}", flush=True)
                verify_intrinsic_atlas_invariance(shared_root, subjects, smoothing, hemi, list(args.atlases))

    colors = load_cluster_colors()
    canonical_color_map = load_canonical_network_color_map()
    canonical_network_order_by_atlas: dict[str, list[str]] = {}

    for atlas in config.args.atlases:
        print(f"[check] canonical-network ordering atlas={atlas}", flush=True)
        canonical_network_order_by_atlas[atlas] = verify_canonical_network_order_consistency(
            reference_root, subjects, atlas, network_taxonomy=network_taxonomy
        )

    precomputed_resample_cache: dict[
        tuple[object, ...],
        tuple[list[tuple[FittedHalf, FittedHalf]], list[tuple[FittedHalf, FittedHalf]]],
    ] = {}
    anchor_inputs_cache: dict[tuple[str, str], tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {}

    for method in config.args.methods:
        for atlas in config.args.atlases:
            if not method_allowed_for_atlas(method, atlas, network_taxonomy=network_taxonomy):
                print(
                    f"[warn] skipping unsupported combo atlas={atlas} method={method}",
                    flush=True,
                )
                continue
            network_names = canonical_network_order_by_atlas[atlas]
            for smoothing in config.args.smoothings:
                for hemi in config.args.hemis:
                    prepared = prepare_combo_inputs(
                        method=method,
                        atlas=atlas,
                        smoothing=smoothing,
                        hemi=hemi,
                        shared_root=shared_root,
                        surface_root=surface_root,
                        profile_surface_root=config.profile_surface_root,
                        reference_root=reference_root,
                        surfaces=surfaces,
                        subjects=subjects,
                        subject_fc_mode=subject_fc_mode,
                        k_selection_mode=str(args.k_selection_mode),
                        network_taxonomy=network_taxonomy,
                    )
                    n_vertices = int(prepared.n_vertices)
                    valid_masks = prepared.valid_masks
                    coverage_count = prepared.coverage_count
                    coverage_fraction = prepared.coverage_fraction
                    feature_bank = prepared.feature_bank
                    actual_input_paths = prepared.actual_input_paths
                    run_feature_banks = prepared.run_feature_banks
                    roi_fc_bank_for_redundancy = prepared.roi_fc_bank_for_redundancy
                    subject_fc_preprocessing_for_resume = prepared.subject_fc_preprocessing_for_resume
                    resume_input_category_hashes = prepared.resume_input_category_hashes

                    for tau in config.coverage_thresholds:
                        combo = Combo(method=method, atlas=atlas, smoothing=smoothing, hemi=hemi, coverage_threshold=float(tau))
                        print(
                            f"[combo] method={method} atlas={atlas} smoothing={smoothing} hemi={hemi} tau={tau:.2f}",
                            flush=True,
                        )
                        if is_gradient_method(method):
                            anchor_smoothing = str(args.alignment_anchor_smoothing)
                            gradient_options = {**gradient_options,
                                "alignment_anchor_smoothing": anchor_smoothing,
                                "alignment_fc_paths": {s: str(shared_root / f"sub-{s}" / "fc" / atlas / anchor_smoothing / "hemi_L" / network_feature_filename(method, atlas)) for s in subjects},
                                "alignment_mask_paths": {s: str(surface_root / f"sub-{s}" / "surface" / "validity" / f"sub-{s}_hemi-L_valid_mask.npy") for s in subjects},
                                "alignment_coverage_threshold": float(tau),
                                "alignment_run_ids": sorted(run_feature_banks),
                            }
                        tau_tag = str(f"{tau:.2f}").replace(".", "p")
                        if is_gradient_method(method):
                            comp_tag = gradient_comp_tag(gradient_components)
                            run_dir = out_root / method / atlas / smoothing / f"cov_{tau_tag}" / comp_tag / f"hemi_{hemi}"
                        else:
                            run_dir = out_root / method / atlas / smoothing / f"cov_{tau_tag}" / f"hemi_{hemi}"
                        run_dir.mkdir(parents=True, exist_ok=True)
                        manifest_path = run_dir / "group_atlas_manifest.json"
                        support = coverage_fraction >= float(tau)
                        subject_fc_preprocessing = subject_fc_preprocessing_for_resume
                        n_support = int(np.sum(support))
                        v_min = int(math.ceil(float(args.v_min_fraction) * max(1, n_support)))
                        k_max_eff, k_eff = effective_k_values(
                            k_values, n_support=n_support, v_min=v_min, mode=args.k_selection_mode,
                        )
                        group_gradient_meta: dict[str, object] | None = None
                        group_features_active: np.ndarray | None = None
                        if method in SIMILARITY_GRADIENT_METHODS and k_eff:
                            group_features_active, group_gradient_meta = compute_similarity_gradient_features(
                                method=method, feature_bank=feature_bank, subjects=subjects, support_mask=support,
                                gradient_components=gradient_components, gradient_options=gradient_options,
                            )
                        elif is_gradient_method(method) and k_eff:
                            group_features_active, group_gradient_meta = load_hippomaps_gradient_features(
                                gradient_root=gradient_root,
                                route=gradient_route_tag,
                                smoothing=smoothing,
                                hemi=hemi,
                                support_mask=support,
                                gradient_components=gradient_components,
                                expected_signal_branch=config.signal_branch,
                                cohort=args.cohort,
                                variant=args.variant,
                                feature_scaling=args.feature_scaling,
                                sparsify_stage=args.sparsify_stage,
                                sparsity=args.sparsity,
                                clustering_feature_scaling=args.clustering_feature_scaling,
                                alignment_anchor_smoothing=args.alignment_anchor_smoothing,
                                profile=args.profile,
                            )
                        resume_signature_payload = {
                            "resume_input_signature_version": int(RESUME_SIGNATURE_VERSION),
                            "combo": {
                                "method": method,
                                "atlas": atlas,
                                "smoothing": smoothing,
                                "hemi": hemi,
                                "coverage_threshold": float(tau),
                            },
                            "subjects_manifest": str(subjects_manifest_path),
                            "subjects": [f"sub-{s}" for s in subjects],
                            "dataset": dataset,
                            "expected_source_root": str(expected_source_root),
                            "k_selection_mode": str(args.k_selection_mode),
                            "k_final_rule": str(args.k_final_rule),
                            "force_final_k": int(args.force_final_k) if args.force_final_k is not None else None,
                            "default_k": int(args.default_k) if args.default_k is not None else None,
                            "candidate_k_values": [int(k) for k in k_values],
                            "seed": int(args.seed),
                            "instability_resamples": int(len(split_seeds)),
                            "null_resamples": int(len(null_seeds)),
                            "null_permutation_scheme": "one_generator_per_repetition",
                            "resample_half_seed_offsets": (
                                list(RESAMPLE_HALF_SEED_OFFSETS)
                                if args.profile is not None
                                else list(DEFAULT_HALF_FIT_SEED_OFFSETS)
                            ),
                            "resample_alignment_reference": "half_specific_anchor_hemi_L",
                            "diagnostic_failure_policy": (
                                "skip_invalid_gradient_resamples_for_fixed_k"
                                if bool(is_gradient_method(method) and args.force_final_k is not None and args.k_selection_mode != "fixed-comparison")
                                else "strict"
                            ),
                            "v_min_fraction": float(args.v_min_fraction),
                            "max_components_threshold": int(args.max_components_threshold),
                            "min_profile_coherence": float(args.min_profile_coherence),
                            "profile_source_mode": profile_source_mode,
                            "profile_source_root": str(profile_source_root) if profile_source_root is not None else None,
                            "profile_source_registry": str(profile_source_registry) if profile_source_registry is not None else None,
                            "profile_fc_mode": "cluster_mean_timeseries",
                            "subject_fc_preprocessing": subject_fc_preprocessing_for_resume,
                            "assignment_variants_requested": list(args.assignment_variant),
                            "gradient_components": [int(c) for c in gradient_components],
                            "surface_affinity": method_surface_affinity_metadata(method)[0] if is_gradient_method(method) else None,
                            **(
                                gradient_resume_inputs(
                                    gradient_root=gradient_root, cohort=args.cohort, variant=args.variant,
                                    gradient_options=gradient_options,
                                    route=gradient_route_tag, smoothing=smoothing, hemi=hemi,
                                    profile=args.profile,
                                )
                                if is_gradient_method(method)
                                else {"gradient_root": None}
                            ),
                            "route_tag": route_tag if is_gradient_method(method) else None,
                            **signal_provenance,
                            "reuse_fixed_cluster_root": (
                                str(Path(args.reuse_fixed_cluster_root).resolve())
                                if args.reuse_fixed_cluster_root is not None and is_gradient_method(method)
                                else None
                            ),
                            "redundancy_early_stop": bool(args.redundancy_early_stop),
                            "input_category_hashes": resume_input_category_hashes,
                        }
                        resume_input_signature = canonical_json_sha256(resume_signature_payload)
                        if args.resume and manifest_path.exists():
                            skip_ok, resume_reason = resume_manifest_matches(
                                manifest_path,
                                resume_input_signature,
                                gradient_options=gradient_options if is_gradient_method(method) else None,
                                expected_gradient_meta=group_gradient_meta,
                            )
                            if skip_ok:
                                print(
                                    f"  [skip] existing manifest (resume-signature matched): {manifest_path}",
                                    flush=True,
                                )
                                continue
                            print(
                                f"  [resume-invalidate] {manifest_path} reason={resume_reason}; recomputing",
                                flush=True,
                            )

                        k_selection_fields = [
                            "method",
                            "atlas",
                            "smoothing",
                            "hemi",
                            "coverage_threshold",
                            "k",
                            "n_subjects",
                            "n_support_vertices",
                            "V_min_effective",
                            "group_ari_mean",
                            "group_ari_se",
                            "group_ari_diagnostic_status",
                            "subject_split_requested",
                            "subject_split_used",
                            "subject_split_skipped_overlap",
                            "subject_split_skipped_empty",
                            "subject_split_skipped_failure",
                            "run_aware_ari_mean",
                            "run_aware_ari_se",
                            "run_aware_ari_status",
                            "run_aware_n_pairs",
                            "run_aware_n_used",
                            "run_aware_skipped_failure",
                            "ARI_null_mean",
                            "ARI_null_5pct",
                            "ARI_null_95pct",
                            "ARI_weak_null_mean",
                            "B_null_requested",
                            "B_null_used_strong",
                            "B_null_used_weak",
                            "B_null_skipped_failure_strong",
                            "min_cluster_size",
                            "max_components_per_cluster",
                            "min_profile_coherence",
                            "mean_profile_coherence",
                            "nontrivial_min_size_ok",
                            "nontrivial_connectedness_ok",
                            "nontrivial_coherence_ok",
                            "local_max",
                            "within_1se_best",
                            "selected_nontrivial",
                            "max_profile_similarity",
                            "redundant_pair_count",
                            "redundancy_profile_status",
                            "redundancy_threshold",
                            "nonredundant_ok",
                            "reject_nontrivial_fail",
                            "reject_redundant",
                            "reject_reason_primary",
                            "selected_final",
                            "within_1se_survivor_pool",
                            "survivor_pool_argmax_k",
                            "k_final_decision_path",
                        ]

                        np.save(run_dir / "group_support_mask.npy", support.astype(np.uint8))
                        np.save(run_dir / "subject_coverage_count.npy", coverage_count.astype(np.int32))
                        np.save(run_dir / "subject_coverage_fraction.npy", coverage_fraction.astype(np.float32))

                        if is_clusterless_method(method):
                            support_indices = np.flatnonzero(support).astype(np.int32)
                            vertex_profile_3d = np.stack(
                                [feature_bank[s][support_indices, :] for s in subjects], axis=0
                            ).astype(np.float32)
                            vertex_profile_mean = np.nanmean(vertex_profile_3d, axis=0).astype(np.float32)
                            vertex_profile_mean[~np.isfinite(vertex_profile_mean)] = np.nan
                            if method != "wta-top":
                                raise ValueError(f"Unknown clusterless method: {method}")
                            if not method_allowed_for_atlas(method, atlas, network_taxonomy=network_taxonomy):
                                raise ValueError(
                                    "wta-top requires the atlas-matched assignment path: "
                                    f"atlas={atlas!r} network_taxonomy={network_taxonomy!r}"
                                )
                            if is_lynch2024_atlas(atlas):
                                vertex_status_rows = build_fisherz_vertex_assignment_rows(
                                    mean_profile=vertex_profile_mean,
                                    support_indices=support_indices,
                                    network_names=network_names,
                                    network_column=network_taxonomy,
                                )
                                confidence_metric = "fisherz_margin"
                                assignment_metric = "top1_fisher_z"
                                default_assignment_variant = "fisherz"
                                q_threshold = None
                                top_n_cap = None
                                profile_filename = "vertex_network_profile.npy"
                                global_top = build_global_top_network_connections(
                                    mean_profile=vertex_profile_mean,
                                    support_indices=support_indices,
                                    network_names=network_names,
                                )
                            else:
                                roi_registry = read_roi_registry_for_atlas(reference_root, subjects[0], atlas)
                                vertex_status_rows = build_normalized_topn_count_vertex_rows(
                                    mean_profile=vertex_profile_mean,
                                    support_indices=support_indices,
                                    roi_registry=roi_registry,
                                    top_n_cap=20,
                                    network_column=network_taxonomy,
                                )
                                confidence_metric = "normalized_top20_count"
                                assignment_metric = "top20_count_normalized_by_network_roi_count"
                                default_assignment_variant = "normalized_top20_count"
                                q_threshold = 0.05
                                top_n_cap = 20
                                profile_filename = "vertex_roi_profile.npy"
                                global_top = build_global_top_connections(
                                    profile_3d=vertex_profile_3d,
                                    support_indices=support_indices,
                                    roi_registry=roi_registry,
                                    q_thresh=0.05,
                                    top_n_cap=20,
                                    network_column=network_taxonomy,
                                )
                            functional_labels, functional_meta, functional_label_items = functional_atlas_from_vertices(
                                n_vertices=n_vertices,
                                status_rows=vertex_status_rows,
                                network_names=network_names,
                                color_map=canonical_color_map,
                            )
                            np.save(run_dir / "group_functional_atlas_labels.npy", functional_labels.astype(np.int32))
                            write_json(run_dir / "group_functional_atlas_label_map.json", functional_meta)
                            np.save(run_dir / profile_filename, vertex_profile_mean.astype(np.float32))
                            np.save(run_dir / "vertex_support_indices.npy", support_indices.astype(np.int32))
                            write_json(
                                run_dir / "vertex_assignment_status.json",
                                {"items": vertex_status_rows, "assignment_metric": assignment_metric},
                            )
                            write_json(run_dir / "global_top_connections.json", global_top)
                            write_json(run_dir / "k_selection_summary.json", {"rows": [], "k_final": None, "mode": "clusterless"})
                            write_tsv(run_dir / "k_selection_summary.tsv", [], k_selection_fields)

                            func_lbl = make_named_label_gifti(functional_labels.astype(np.int32), functional_label_items)
                            func_lbl_path = (
                                run_dir
                                / "template"
                                / "workbench_assets"
                                / f"{template_label}_hemi-{hemi}_space-corobl_den-512_label-group_functional_atlas_{method}_{atlas}_{smoothing}_cov-{tau_tag}.label.gii"
                            )
                            func_lbl_path.parent.mkdir(parents=True, exist_ok=True)
                            variant_func_lbl_path = func_lbl_path.with_name(
                                f"{func_lbl_path.stem}_{default_assignment_variant}{func_lbl_path.suffix}"
                            )
                            nib.save(func_lbl, str(func_lbl_path))
                            nib.save(func_lbl, str(variant_func_lbl_path))

                            manifest = {
                                "status": "ok",
                                **code_provenance(),
                                "dataset": dataset,
                                "expected_source_root": str(expected_source_root),
                                **signal_provenance,
                                "method": method,
                                "atlas": atlas,
                                "smoothing": smoothing,
                                "hemi": hemi,
                                "assignment_mode": "vertex_level",
                                "default_assignment_variant": default_assignment_variant,
                                "assignment_metric": assignment_metric,
                                "cluster_assignment_metric": assignment_metric,
                                "confidence_metric": confidence_metric,
                                "coverage_threshold": float(tau),
                                "chosen_k_final": None,
                                "q_threshold": q_threshold,
                                "top_n_cap": top_n_cap,
                                "actual_consumed_input_paths": actual_input_paths,
                                "shared_root": str(shared_root.resolve()),
                                "surface_root": str(surface_root.resolve()),
                                "profile_surface_root": str(config.profile_surface_root.resolve()),
                                "reference_root": str(reference_root.resolve()),
                                "resume_input_signature_version": int(RESUME_SIGNATURE_VERSION),
                                "resume_input_signature": resume_input_signature,
                                "resume_input_signature_payload": resume_signature_payload,
                                "feature_family": feature_family_for_method(method, atlas, gradient_components),
                                "fc_mode": fc_mode_for_method(method, atlas),
                                "subject_fc_preprocessing": subject_fc_preprocessing,
                                "n_subjects": int(len(subjects)),
                                "subjects": [f"sub-{s}" for s in subjects],
                                "n_support_vertices": int(n_support),
                                "n_excluded_vertices": int(n_vertices - n_support),
                                "group_support_mask": str((run_dir / "group_support_mask.npy").resolve()),
                                "subject_coverage_count": str((run_dir / "subject_coverage_count.npy").resolve()),
                                "subject_coverage_fraction": str((run_dir / "subject_coverage_fraction.npy").resolve()),
                                "group_functional_atlas_labels": str((run_dir / "group_functional_atlas_labels.npy").resolve()),
                                "group_functional_atlas_label_map": str((run_dir / "group_functional_atlas_label_map.json").resolve()),
                                "vertex_roi_profile": (
                                    str((run_dir / "vertex_roi_profile.npy").resolve())
                                    if not is_lynch2024_atlas(atlas)
                                    else None
                                ),
                                "vertex_network_profile": (
                                    str((run_dir / "vertex_network_profile.npy").resolve())
                                    if is_lynch2024_atlas(atlas)
                                    else None
                                ),
                                "vertex_support_indices": str((run_dir / "vertex_support_indices.npy").resolve()),
                                "vertex_assignment_status": str((run_dir / "vertex_assignment_status.json").resolve()),
                                "global_top_connections": str((run_dir / "global_top_connections.json").resolve()),
                                "cluster_network_profile": None,
                                "cluster_dominant_network": None,
                                "cluster_assignment_status": None,
                                "cluster_roi_profile": None,
                                "cluster_roi_top20": None,
                                "cluster_roi_registry": (
                                    str((run_dir / "cluster_roi_registry.tsv").resolve())
                                    if not is_lynch2024_atlas(atlas)
                                    else None
                                ),
                                "network_taxonomy": network_taxonomy,
                                "network_registry_column": (
                                    resolve_kong2022_network_registry_column(network_taxonomy)
                                    if not is_lynch2024_atlas(atlas)
                                    else None
                                ),
                                "network_column": network_taxonomy,
                                "network_order": network_names,
                                "canonical_network_order_consistency_checked": network_taxonomy in {"canonical7", "lynch7"},
                                "network_order_consistency_checked": True,
                                "canonical_network_order_consensus_subject_count": int(len(subjects)) if network_taxonomy in {"canonical7", "lynch7"} else None,
                                "gradient_recomputed": False,
                                "clustering_recomputed": False,
                                "k_selection_recomputed": False,
                                "fixed_cluster_reuse": None,
                                "k_selection_summary_tsv": str((run_dir / "k_selection_summary.tsv").resolve()),
                                "k_selection_summary_json": str((run_dir / "k_selection_summary.json").resolve()),
                                "template_subject": template_render_subject,
                                "template_source_mode": template_source_mode,
                                "template_source_note": template_source_note,
                                "template_left_or_right_surface": str(surfaces[hemi]),
                                "template_spec_path": str(template_spec_path),
                                "template_cluster_label": None,
                                "template_functional_atlas_label": str(variant_func_lbl_path.resolve()),
                            }
                            if not is_lynch2024_atlas(atlas):
                                write_tsv(
                                    run_dir / "cluster_roi_registry.tsv",
                                    roi_registry,
                                    [
                                        "roi_id",
                                        "roi_name",
                                        "hemi",
                                        "family17",
                                        "lynch20_network",
                                        "subregion",
                                        "rank",
                                        resolve_kong2022_network_registry_column(network_taxonomy),
                                    ],
                                )
                            write_json(
                                run_dir / "group_feature_summary.json",
                                {
                                    "network_names": network_names,
                                    "N_ref": int(len(network_names)),
                                    "network_taxonomy": network_taxonomy,
                                    "network_column": network_taxonomy,
                                },
                            )
                            write_json(manifest_path, manifest)
                            continue

                        if not k_eff:
                            skip_reason = "no_valid_k_for_support"
                            skip_detail = (
                                f"No valid K for combo={combo} n_support={n_support} "
                                f"v_min={v_min} k_max_eff={k_max_eff}"
                            )
                            write_tsv(run_dir / "k_selection_summary.tsv", [], k_selection_fields)
                            write_json(
                                run_dir / "k_selection_summary.json",
                                {
                                    "rows": [],
                                    "k_final": None,
                                    "skip_reason": skip_reason,
                                    "skip_detail": skip_detail,
                                },
                            )
                            manifest = {
                                "status": "skipped_no_valid_k",
                                **signal_provenance,
                                "skip_reason": skip_reason,
                                "skip_detail": skip_detail,
                                "method": method,
                                "atlas": atlas,
                                "smoothing": smoothing,
                                "hemi": hemi,
                                "k_selection_mode": str(args.k_selection_mode),
                                "k_final_rule": str(args.k_final_rule),
                                "force_final_k": int(args.force_final_k) if args.force_final_k is not None else None,
                            "default_k": int(args.default_k) if args.default_k is not None else None,
                                "coverage_threshold": float(tau),
                                "V_min_fraction": float(args.v_min_fraction),
                                "V_min_effective": int(v_min),
                                "gradient_components": [int(c) for c in gradient_components] if is_gradient_method(method) else None,
                                "candidate_k_range_requested": [int(min(k_values)), int(max(k_values))],
                                "candidate_k_range_effective": [],
                                "chosen_k_final": None,
                                "subject_split_seeds": split_seeds,
                                "null_seeds": null_seeds,
                                "n_subjects": int(len(subjects)),
                                "subjects": [f"sub-{s}" for s in subjects],
                                "n_support_vertices": int(n_support),
                                "n_excluded_vertices": int(n_vertices - n_support),
                                "actual_consumed_input_paths": actual_input_paths,
                                "resume_input_signature_version": int(RESUME_SIGNATURE_VERSION),
                                "resume_input_signature": resume_input_signature,
                                "resume_input_signature_payload": resume_signature_payload,
                                "subject_fc_preprocessing": subject_fc_preprocessing,
                                "group_gradient_features_active": None,
                                "group_gradient_feature_summary": None,
                                "group_feature_summary": str((run_dir / "group_feature_summary.json").resolve()),
                                "nontriviality_thresholds": {
                                    "V_min_fraction": float(args.v_min_fraction),
                                    "max_components_threshold": int(args.max_components_threshold),
                                    "min_profile_coherence": float(args.min_profile_coherence),
                                },
                                "redundancy_aware_thresholds": {
                                    "max_profile_similarity_lte": float(PROFILE_REDUNDANCY_MAX_SIM),
                                    "redundant_pair_count_eq": 0,
                                },
                                "density_evidence": {
                                    "surface_path": str(surfaces[hemi]),
                                    "n_vertices": int(surface_meta[hemi]["n_vertices"]),
                                    "n_faces": int(surface_meta[hemi]["n_faces"]),
                                },
                                "surface_topology_audit": topology_audit[hemi],
                                "group_support_mask": str((run_dir / "group_support_mask.npy").resolve()),
                                "subject_coverage_count": str((run_dir / "subject_coverage_count.npy").resolve()),
                                "subject_coverage_fraction": str((run_dir / "subject_coverage_fraction.npy").resolve()),
                                "k_selection_summary_tsv": str((run_dir / "k_selection_summary.tsv").resolve()),
                                "k_selection_summary_json": str((run_dir / "k_selection_summary.json").resolve()),
                            }
                            write_json(manifest_path, manifest)
                            print(f"  [skip] {skip_detail}", flush=True)
                            continue

                        rows: list[dict[str, object]] = []
                        labels_by_k: dict[int, np.ndarray] = {}
                        fixed_cluster_reuse: dict[str, object] | None = None
                        allow_gradient_diagnostic_failures = bool(is_gradient_method(method) and args.force_final_k is not None and args.k_selection_mode != "fixed-comparison")
                        support_idx = np.flatnonzero(support)
                        adjacency_support = adjacency_by_hemi[hemi][support_idx][:, support_idx]
                        if is_gradient_method(method):
                            if (group_features_active is None or group_gradient_meta is None) and method in SIMILARITY_GRADIENT_METHODS:
                                group_features_active, group_gradient_meta = compute_similarity_gradient_features(
                                    method=method, feature_bank=feature_bank, subjects=subjects, support_mask=support,
                                    gradient_components=gradient_components, gradient_options=gradient_options,
                                )
                            elif group_features_active is None or group_gradient_meta is None:
                                group_features_active, group_gradient_meta = load_hippomaps_gradient_features(
                                    gradient_root=gradient_root,
                                    route=gradient_route_tag,
                                    smoothing=smoothing,
                                    hemi=hemi,
                                    support_mask=support,
                                    gradient_components=gradient_components,
                                    expected_signal_branch=config.signal_branch,
                                    cohort=args.cohort,
                                    variant=args.variant,
                                    feature_scaling=args.feature_scaling,
                                    sparsify_stage=args.sparsify_stage,
                                    sparsity=args.sparsity,
                                    clustering_feature_scaling=args.clustering_feature_scaling,
                                    alignment_anchor_smoothing=args.alignment_anchor_smoothing,
                                )
                        else:
                            group_features_active = group_feature_matrix(
                                method=method,
                                feature_bank=feature_bank,
                                subjects_subset=subjects,
                                support_mask=support,
                                gradient_components=gradient_components,
                                gradient_options=gradient_options,
                            )
                        if args.reuse_fixed_cluster_root is not None and not is_clusterless_method(method):
                            source_run = resolve_fixed_cluster_source_run(
                                source_root=Path(args.reuse_fixed_cluster_root),
                                method=method,
                                atlas=atlas,
                                smoothing=smoothing,
                                hemi=hemi,
                                coverage_tag=f"cov_{tau_tag}",
                                gradient_components=gradient_components,
                            )
                            reused_k, reused_labels, reused_row, fixed_cluster_reuse = load_fixed_cluster_reuse(
                                source_run=source_run,
                                target_method=method,
                                target_atlas=atlas,
                                smoothing=smoothing,
                                hemi=hemi,
                                coverage_threshold=float(tau),
                                support_mask=support,
                                n_vertices=n_vertices,
                                k_values=k_values,
                                force_final_k=args.force_final_k,
                                default_k=args.default_k,
                                k_selection_fields=k_selection_fields,
                                target_subjects=subjects,
                                target_surface_path=surfaces[hemi],
                                target_gradient_feature_summary=group_gradient_meta,
                                target_clustering_feature_scaling=args.clustering_feature_scaling,
                                target_rbf_scale=args.rbf_scale,
                                target_gradient_features_active=group_features_active,
                            )
                            if network_taxonomy == "family17" and fixed_cluster_reuse["source_network_taxonomy"] != "canonical7":
                                raise ValueError("Family17 cluster reuse requires a canonical7 source manifest")
                            labels_by_k[int(reused_k)] = reused_labels.astype(np.int32)
                            np.save(run_dir / f"group_labels_k{reused_k}.npy", reused_labels.astype(np.int32))
                            rows.append(reused_row)
                            print(
                                f"  [reuse-fixed-cluster] source={source_run} k={reused_k}; "
                                "gradient/clustering/K-selection computation skipped",
                                flush=True,
                            )
                        if is_gradient_method(method):
                            np.save(run_dir / "group_gradient_features_active.npy", group_features_active.astype(np.float32))
                            write_json(run_dir / "group_gradient_feature_summary.json", group_gradient_meta)

                        precomputed_split_halves = None
                        precomputed_null_halves = None
                        if is_gradient_method(method) and fixed_cluster_reuse is None and (args.profile is not None or len(split_seeds) <= 20):
                            half_fit_seed_offsets = (
                                RESAMPLE_HALF_SEED_OFFSETS
                                if args.profile is not None
                                else DEFAULT_HALF_FIT_SEED_OFFSETS
                            )
                            anchor_smoothing = str(args.alignment_anchor_smoothing)
                            n_resamples = len(split_seeds)
                            base_resample_key = (
                                atlas,
                                str(args.feature_scaling),
                                str(args.sparsify_stage),
                                float(args.sparsity) if args.sparsity is not None else None,
                                anchor_smoothing,
                                float(tau),
                                int(args.seed),
                                int(n_resamples),
                                int(len(null_seeds)),
                                tuple(gradient_components),
                                tuple(half_fit_seed_offsets),
                            )
                            anchor_cache_key = (*base_resample_key, anchor_smoothing, "L")
                            if anchor_cache_key not in precomputed_resample_cache:
                                if smoothing == anchor_smoothing and hemi == "L":
                                    anchor_fb = feature_bank
                                    anchor_vm = valid_masks
                                else:
                                    anchor_input_key = (atlas, anchor_smoothing)
                                    if anchor_input_key not in anchor_inputs_cache:
                                        anchor_vm_loaded = {
                                            s: subject_valid_mask(surface_root, shared_root, s, anchor_smoothing, "L")
                                            for s in subjects
                                        }
                                        anchor_fb_loaded = {}
                                        for s in subjects:
                                            raw_anchor = subject_network_feature(
                                                shared_root,
                                                s,
                                                atlas,
                                                anchor_smoothing,
                                                "L",
                                                method,
                                                reference_root=reference_root,
                                            )
                                            anchor_fb_loaded[s] = preprocess_subject_fc_matrix(
                                                method=method,
                                                fc=raw_anchor,
                                                subject_fc_mode=subject_fc_mode,
                                            )
                                        anchor_inputs_cache[anchor_input_key] = (anchor_fb_loaded, anchor_vm_loaded)
                                    anchor_fb, anchor_vm = anchor_inputs_cache[anchor_input_key]
                                print(
                                    f"  [resample-precompute] Precomputing {n_resamples} anchor ({anchor_smoothing} hemi_L) split & null halves...",
                                    flush=True,
                                )
                                anchor_splits = prepare_all_split_halves(
                                    subjects=subjects,
                                    feature_bank=anchor_fb,
                                    valid_masks=anchor_vm,
                                    tau=float(tau),
                                    feature_scaling=args.feature_scaling,
                                    sparsify_stage=args.sparsify_stage,
                                    sparsity=args.sparsity,
                                    reference_full=None,
                                    n_repetitions=n_resamples,
                                    base_seed=int(args.seed),
                                    max_workers=int(args.resampling_workers),
                                    selected_components=gradient_components,
                                    half_fit_seed_offsets=half_fit_seed_offsets,
                                    reference_halves=None,
                                )
                                anchor_nulls = prepare_all_null_halves(
                                    subjects=subjects,
                                    feature_bank=anchor_fb,
                                    valid_masks=anchor_vm,
                                    tau=float(tau),
                                    feature_scaling=args.feature_scaling,
                                    sparsify_stage=args.sparsify_stage,
                                    sparsity=args.sparsity,
                                    reference_full=None,
                                    n_repetitions=len(null_seeds),
                                    base_seed=int(args.seed),
                                    max_workers=int(args.resampling_workers),
                                    selected_components=gradient_components,
                                    half_fit_seed_offsets=half_fit_seed_offsets,
                                    reference_halves=None,
                                )
                                precomputed_resample_cache[anchor_cache_key] = (anchor_splits, anchor_nulls)

                            combo_cache_key = (*base_resample_key, smoothing, hemi)
                            if combo_cache_key not in precomputed_resample_cache:
                                anchor_splits, anchor_nulls = precomputed_resample_cache[anchor_cache_key]
                                print(
                                    f"  [resample-precompute] Precomputing {n_resamples} split & null halves for {smoothing} hemi_{hemi}...",
                                    flush=True,
                                )
                                combo_splits = prepare_all_split_halves(
                                    subjects=subjects,
                                    feature_bank=feature_bank,
                                    valid_masks=valid_masks,
                                    tau=float(tau),
                                    feature_scaling=args.feature_scaling,
                                    sparsify_stage=args.sparsify_stage,
                                    sparsity=args.sparsity,
                                    reference_full=None,
                                    n_repetitions=n_resamples,
                                    base_seed=int(args.seed),
                                    max_workers=int(args.resampling_workers),
                                    selected_components=gradient_components,
                                    half_fit_seed_offsets=half_fit_seed_offsets,
                                    reference_halves=anchor_splits,
                                )
                                combo_nulls = prepare_all_null_halves(
                                    subjects=subjects,
                                    feature_bank=feature_bank,
                                    valid_masks=valid_masks,
                                    tau=float(tau),
                                    feature_scaling=args.feature_scaling,
                                    sparsify_stage=args.sparsify_stage,
                                    sparsity=args.sparsity,
                                    reference_full=None,
                                    n_repetitions=len(null_seeds),
                                    base_seed=int(args.seed),
                                    max_workers=int(args.resampling_workers),
                                    selected_components=gradient_components,
                                    half_fit_seed_offsets=half_fit_seed_offsets,
                                    reference_halves=anchor_nulls,
                                )
                                precomputed_resample_cache[combo_cache_key] = (combo_splits, combo_nulls)
                                print("  [resample-precompute] Precomputation complete.", flush=True)
                            else:
                                print(
                                    f"  [resample-precompute] Reusing cached split & null halves for {smoothing} hemi_{hemi}.",
                                    flush=True,
                                )
                            precomputed_split_halves, precomputed_null_halves = precomputed_resample_cache[combo_cache_key]

                        redundancy_early_stop_enabled = bool(
                            args.redundancy_early_stop
                            and fixed_cluster_reuse is None
                            and args.k_selection_mode != "fixed-comparison"
                        )
                        if fixed_cluster_reuse is not None:
                            pass
                        elif redundancy_early_stop_enabled:
                            print(
                                f"  [k-seq-heuristic] n_k={len(k_eff)} stop_on_first_redundant=1",
                                flush=True,
                            )
                            for k in k_eff:
                                print(f"  [k-start] {k}", flush=True)
                                k_result, labels_full, row = evaluate_k_candidate(
                                    combo=combo,
                                    method=method,
                                    k=int(k),
                                    seed=int(args.seed + 11 * k),
                                    n_support=int(n_support),
                                    v_min=int(v_min),
                                    group_features_active=group_features_active,
                                    support_mask=support,
                                    adjacency_full=adjacency_by_hemi[hemi],
                                    adjacency_support=adjacency_support,
                                    feature_bank=feature_bank,
                                    valid_masks=valid_masks,
                                    subjects=subjects,
                                    tau=float(tau),
                                    split_seeds=split_seeds,
                                    null_seeds=null_seeds,
                                    null_resamples=int(args.null_resamples),
                                    run_feature_banks=run_feature_banks,
                                    seed_base=int(args.seed),
                                    max_components_threshold=int(args.max_components_threshold),
                                    min_profile_coherence_threshold=float(args.min_profile_coherence),
                                    k_selection_mode=str(args.k_selection_mode),
                                    roi_fc_bank_for_redundancy=roi_fc_bank_for_redundancy,
                                    gradient_components=gradient_components,
                                    allow_diagnostic_failures=allow_gradient_diagnostic_failures,
                                    gradient_options=gradient_options,
                                    precomputed_split_halves=precomputed_split_halves,
                                    precomputed_null_halves=precomputed_null_halves,
                                    redundancy_threshold=redundancy_threshold,
                                )
                                print(f"  [k-done] {k_result}", flush=True)
                                labels_by_k[int(k_result)] = labels_full
                                np.save(run_dir / f"group_labels_k{k_result}.npy", labels_full.astype(np.int32))
                                rows.append(row)
                                if redundancy_early_stop_triggered(row, redundancy_early_stop_enabled):
                                    print(
                                        f"  [k-early-stop] first_redundant_k={k_result}; skipping larger K",
                                        flush=True,
                                    )
                                    break
                        else:
                            k_eval_jobs = 1 if precomputed_split_halves is not None else resolve_k_eval_jobs(int(args.jobs), len(k_eff))
                            print(
                                f"  [k-parallel] n_k={len(k_eff)} jobs={k_eval_jobs}",
                                flush=True,
                            )
                            with ThreadPoolExecutor(max_workers=k_eval_jobs) as executor:
                                future_to_k = {
                                    executor.submit(
                                        evaluate_k_candidate,
                                        combo=combo,
                                        method=method,
                                        k=int(k),
                                        seed=int(args.seed + 11 * k),
                                        n_support=int(n_support),
                                        v_min=int(v_min),
                                        group_features_active=group_features_active,
                                        support_mask=support,
                                        adjacency_full=adjacency_by_hemi[hemi],
                                        adjacency_support=adjacency_support,
                                        feature_bank=feature_bank,
                                        valid_masks=valid_masks,
                                        subjects=subjects,
                                        tau=float(tau),
                                        split_seeds=split_seeds,
                                        null_seeds=null_seeds,
                                        null_resamples=int(args.null_resamples),
                                        run_feature_banks=run_feature_banks,
                                        seed_base=int(args.seed),
                                        max_components_threshold=int(args.max_components_threshold),
                                        min_profile_coherence_threshold=float(args.min_profile_coherence),
                                        k_selection_mode=str(args.k_selection_mode),
                                        roi_fc_bank_for_redundancy=roi_fc_bank_for_redundancy,
                                        gradient_components=gradient_components,
                                        allow_diagnostic_failures=allow_gradient_diagnostic_failures,
                                        gradient_options=gradient_options,
                                        precomputed_split_halves=precomputed_split_halves,
                                        precomputed_null_halves=precomputed_null_halves,
                                        redundancy_threshold=redundancy_threshold,
                                    ): int(k)
                                    for k in k_eff
                                }
                                for future in as_completed(future_to_k):
                                    k_done = int(future_to_k[future])
                                    print(f"  [k-done] {k_done}", flush=True)
                                    k_result, labels_full, row = future.result()
                                    labels_by_k[int(k_result)] = labels_full
                                    np.save(run_dir / f"group_labels_k{k_result}.npy", labels_full.astype(np.int32))
                                    rows.append(row)

                        rows = sorted(rows, key=lambda r: int(r["k"]))
                        means = [float(r["group_ari_mean"]) for r in rows]
                        ses = [float(r["group_ari_se"]) for r in rows]
                        locals_ = local_maxima_flags(means)
                        finite_indices = [i for i, val in enumerate(means) if np.isfinite(float(val))]
                        if finite_indices:
                            best_idx = int(finite_indices[int(np.argmax(np.asarray([means[i] for i in finite_indices], dtype=np.float64)))])
                            cutoff = float(means[best_idx] - ses[best_idx])
                        else:
                            cutoff = math.nan
                        for i, row in enumerate(rows):
                            row["local_max"] = int(locals_[i])
                            row_mean = float(row["group_ari_mean"])
                            row["within_1se_best"] = int(np.isfinite(row_mean) and np.isfinite(cutoff) and row_mean >= cutoff - 1e-12)
                            row["selected_nontrivial"] = int(
                                int(row["nontrivial_min_size_ok"]) == 1
                                and int(row["nontrivial_connectedness_ok"]) == 1
                                and int(row["nontrivial_coherence_ok"]) == 1
                            )
                            # Survivor-pool audit columns; overwritten after k_final is chosen.
                            row["within_1se_survivor_pool"] = 0
                            row["survivor_pool_argmax_k"] = -1
                            row["k_final_decision_path"] = "none"
                        forced_fixed_k_target = (
                            int(args.force_final_k)
                            if args.force_final_k is not None
                            else (
                                int(fixed_cluster_reuse["source_chosen_k_final"])
                                if fixed_cluster_reuse is not None
                                else (int(k_values[0]) if len(k_values) == 1 else None)
                            )
                        )
                        if forced_fixed_k_target is not None and not any(
                            int(row["k"]) == int(forced_fixed_k_target) for row in rows
                        ):
                            raise RuntimeError(
                                "Forced final K was not evaluated for this combo: "
                                f"k={forced_fixed_k_target}, evaluated={[int(r['k']) for r in rows]}, combo={combo}"
                            )
                        forced_fixed_k = forced_fixed_k_target is not None
                        candidate_pool: list[dict] = []
                        if forced_fixed_k:
                            k_forced = int(forced_fixed_k_target)
                            for row in rows:
                                is_target = int(row["k"]) == k_forced
                                row["selected_final"] = int(is_target)
                                row["within_1se_survivor_pool"] = int(is_target)
                                row["survivor_pool_argmax_k"] = int(k_forced)
                                row["k_final_decision_path"] = "forced_fixed_k"
                                row["reject_nontrivial_fail"] = 0
                                row["reject_redundant"] = 0
                                row["reject_reason_primary"] = ""
                            survivors = [row for row in rows if int(row["selected_final"]) == 1]
                        else:
                            for row in rows:
                                redundancy_status = str(row.get("redundancy_profile_status", "ok"))
                                row["nonredundant_ok"] = int(
                                    (
                                        np.isfinite(float(row["max_profile_similarity"]))
                                        and float(row["max_profile_similarity"]) <= float(redundancy_threshold)
                                        and int(row["redundant_pair_count"]) == 0
                                    )
                                )
                                row["reject_nontrivial_fail"] = int(int(row["selected_nontrivial"]) != 1)
                                row["reject_redundant"] = int(int(row["nonredundant_ok"]) != 1)
                                row["selected_final"] = 0
                                if int(row["reject_nontrivial_fail"]) == 1:
                                    row["reject_reason_primary"] = "nontrivial_fail"
                                elif int(row["reject_redundant"]) == 1:
                                    row["reject_reason_primary"] = "redundant"
                                else:
                                    row["reject_reason_primary"] = "outside_selection_rule"
                            candidate_pool = [
                                row
                                for row in rows
                                if int(row["selected_nontrivial"]) == 1 and int(row["nonredundant_ok"]) == 1
                            ]
                            if not candidate_pool:
                                survivors = []
                            else:
                                k_final, decision_path, within_1se_pool, survivor_argmax_k = (
                                    select_k_final(candidate_pool, str(args.k_final_rule))
                                )
                                for row in rows:
                                    row["survivor_pool_argmax_k"] = int(survivor_argmax_k)
                                    row["within_1se_survivor_pool"] = int(int(row["k"]) in within_1se_pool)
                                    row["k_final_decision_path"] = decision_path
                                    if int(row["k"]) == int(k_final):
                                        row["selected_final"] = 1
                                        row["reject_reason_primary"] = ""
                                survivors = [r for r in rows if int(r["selected_final"]) == 1]
                        if not survivors:
                            if args.default_k is not None and int(args.default_k) in labels_by_k:
                                k_final = int(args.default_k)
                                for row in rows:
                                    if int(row["k"]) == k_final:
                                        row["selected_final"] = 1
                                        row["reject_reason_primary"] = "default_k_displayed"
                                    else:
                                        row["selected_final"] = 0
                                decision_path = "default_k_displayed_no_survivors"
                                automatic_k_final = None
                                primary_k = {
                                    "chosen_k_final": k_final,
                                    "primary_k_source": "default_k",
                                    "automatic_k_final": None,
                                    "force_final_k": None,
                                    "default_k": k_final,
                                }
                            else:
                                skip_reason = "no_k_survived_selection_constraints"
                                skip_detail = f"No K survived selection constraints for combo={combo}"
                                write_tsv(run_dir / "k_selection_summary.tsv", rows, k_selection_fields)
                                write_json(
                                    run_dir / "k_selection_summary.json",
                                    {
                                        "rows": rows,
                                        "k_final": None,
                                        "skip_reason": skip_reason,
                                        "skip_detail": skip_detail,
                                    },
                                )
                                manifest = {
                                    "status": "skipped_no_k_survived_constraints",
                                    **signal_provenance,
                                    "skip_reason": skip_reason,
                                    "skip_detail": skip_detail,
                                    "method": method,
                                    "atlas": atlas,
                                    "smoothing": smoothing,
                                    "hemi": hemi,
                                    "k_selection_mode": str(args.k_selection_mode),
                                    "k_final_rule": str(args.k_final_rule),
                                    "force_final_k": int(args.force_final_k) if args.force_final_k is not None else None,
                                "default_k": int(args.default_k) if args.default_k is not None else None,
                                    "coverage_threshold": float(tau),
                                    "V_min_fraction": float(args.v_min_fraction),
                                    "V_min_effective": int(v_min),
                                    "feature_family": feature_family_for_method(method, atlas, gradient_components),
                                    "fc_mode": fc_mode_for_method(method, atlas),
                                    "gradient_components": [int(c) for c in gradient_components] if is_gradient_method(method) else None,
                                    "candidate_k_range_requested": [int(min(k_values)), int(max(k_values))],
                                    "candidate_k_range_effective": [int(min(k_eff)), int(max(k_eff))],
                                    "candidate_k_values_evaluated": [int(r["k"]) for r in rows],
                                    "redundancy_early_stop_enabled": bool(redundancy_early_stop_enabled),
                                    "chosen_k_final": None,
                                    "subject_split_seeds": split_seeds,
                                    "null_seeds": null_seeds,
                                    "n_subjects": int(len(subjects)),
                                    "subjects": [f"sub-{s}" for s in subjects],
                                    "n_support_vertices": int(n_support),
                                    "n_excluded_vertices": int(n_vertices - n_support),
                                    "actual_consumed_input_paths": actual_input_paths,
                                    "resume_input_signature_version": int(RESUME_SIGNATURE_VERSION),
                                    "resume_input_signature": resume_input_signature,
                                    "resume_input_signature_payload": resume_signature_payload,
                                    "subject_fc_preprocessing": subject_fc_preprocessing,
                                    "group_gradient_features_active": (
                                        str((run_dir / "group_gradient_features_active.npy").resolve())
                                        if is_gradient_method(method)
                                        else None
                                    ),
                                    "group_gradient_feature_summary": (
                                        str((run_dir / "group_gradient_feature_summary.json").resolve())
                                        if is_gradient_method(method)
                                        else None
                                    ),
                                    "nontriviality_thresholds": {
                                        "V_min_fraction": float(args.v_min_fraction),
                                        "max_components_threshold": int(args.max_components_threshold),
                                        "min_profile_coherence": float(args.min_profile_coherence),
                                    },
                                    "redundancy_aware_thresholds": {
                                        "max_profile_similarity_lte": float(redundancy_threshold),
                                        "redundant_pair_count_eq": 0,
                                    },
                                    "density_evidence": {
                                        "surface_path": str(surfaces[hemi]),
                                        "n_vertices": int(surface_meta[hemi]["n_vertices"]),
                                        "n_faces": int(surface_meta[hemi]["n_faces"]),
                                    },
                                    "surface_topology_audit": topology_audit[hemi],
                                    "group_support_mask": str((run_dir / "group_support_mask.npy").resolve()),
                                    "subject_coverage_count": str((run_dir / "subject_coverage_count.npy").resolve()),
                                    "subject_coverage_fraction": str((run_dir / "subject_coverage_fraction.npy").resolve()),
                                    "k_selection_summary_tsv": str((run_dir / "k_selection_summary.tsv").resolve()),
                                    "k_selection_summary_json": str((run_dir / "k_selection_summary.json").resolve()),
                                }
                                write_json(manifest_path, manifest)
                                print(f"  [skip] {skip_detail}", flush=True)
                                continue
                        selected_rows = [row for row in rows if int(row["selected_final"]) == 1]
                        if len(selected_rows) != 1:
                            raise RuntimeError(
                                f"Expected exactly one selected_final row for combo={combo}, got {len(selected_rows)}"
                            )
                        k_final = int(selected_rows[0]["k"])
                        decision_path = str(selected_rows[0]["k_final_decision_path"])
                        primary_k = resolve_selected_primary_k(
                            k_final=k_final,
                            k_selection_mode=str(args.k_selection_mode),
                            forced_fixed_k=forced_fixed_k,
                            candidate_pool=candidate_pool,
                            survivors=survivors,
                            force_final_k=args.force_final_k,
                            default_k=args.default_k,
                        )
                        if args.undersized_default_k_fallback is not None:
                            primary_k = apply_undersized_default_k_fallback(
                                primary_k, rows, int(args.undersized_default_k_fallback),
                            )
                        if primary_k["chosen_k_final"] is None:
                            raise RuntimeError(f"No primary K available for combo={combo}")
                        k_final = int(primary_k["chosen_k_final"])
                        if k_final not in labels_by_k:
                            raise RuntimeError(
                                f"Primary K={k_final} was not evaluated for combo={combo}; "
                                f"evaluated={sorted(labels_by_k)}"
                            )
                        if network_taxonomy == "family17" and fixed_cluster_reuse is not None:
                            rows, primary_k = family17_reused_k_diagnostics(
                                Path(fixed_cluster_reuse["source_run"]), k_final,
                            )
                        primary_row = primary_k_row(rows, k_final)
                        labels_final = reorder_labels_by_long_axis(
                            labels_by_k[k_final],
                            long_axis_order_by_hemi[hemi],
                        )
                        if is_gradient_method(method) and fixed_cluster_reuse is None:
                            for row in rows:
                                if int(row["k"]) == k_final:
                                    row["gradient_clustering"] = spatial_cluster_diagnostics(
                                        row["gradient_clustering"], labels_by_k[k_final][support],
                                        labels_final[support], adjacency_support, k=k_final,
                                    )
                                    row["affinity_graph_diagnostics"]["cluster_vertex_counts"] = row["gradient_clustering"]["cluster_vertex_counts"]
                        fixed_cluster_order = cluster_ids_by_long_axis(
                            labels_final,
                            long_axis_order_by_hemi[hemi],
                        )

                        profile: np.ndarray
                        status_rows: list[dict[str, object]]
                        dominant: list[str | None]
                        roi_profile: np.ndarray | None = None
                        roi_profile_3d: np.ndarray | None = None
                        roi_profile_subjects_used: list[str] | None = None
                        roi_registry: list[dict[str, object]] | None = None
                        roi_top20: dict[str, object] | None = None
                        roi_normalized_status_rows: list[dict[str, object]] | None = None
                        ovr_variants = requested_ovr_variants(args.assignment_variant, atlas=atlas, method=method)
                        if args.profile is not None and not is_clusterless_method(method) and atlas == "kong2022":
                            if "ovr_logistic_canonical7" not in ovr_variants:
                                ovr_variants = list(ovr_variants) + ["ovr_logistic_canonical7"]
                        if method_uses_kong2022_topn_cluster_annotation(method, atlas):
                            (
                                profile,
                                status_rows,
                                dominant,
                                roi_profile,
                                roi_registry,
                                roi_profile_subjects_used,
                                roi_top20,
                                roi_normalized_status_rows,
                                roi_profile_3d,
                            ) = annotate_kong2022_clusters_topn(
                                group_labels_final=labels_final,
                                support_mask=support,
                                subjects=subjects,
                                shared_root=shared_root,
                                surface_root=config.profile_surface_root,
                                reference_root=reference_root,
                                smoothing=smoothing,
                                hemi=hemi,
                                valid_masks_by_subject=valid_masks,
                                network_names=network_names,
                                q_thresh=0.05,
                                top_n_cap=20,
                                network_column=resolve_kong2022_network_registry_column(network_taxonomy),
                                reference_atlas=atlas,
                                profile_fc_mode="cluster_mean_timeseries",
                                profile_source_mode=profile_source_mode,
                                profile_source_root=profile_source_root,
                                profile_source_registry=profile_source_registry,
                            )
                        else:
                            if is_lynch2024_atlas(atlas):
                                profile = cluster_network_profile(
                                    group_labels_final=labels_final,
                                    support_mask=support,
                                    subjects=subjects,
                                    shared_root=shared_root,
                                    surface_root=config.profile_surface_root,
                                    reference_root=reference_root,
                                    atlas=atlas,
                                    smoothing=smoothing,
                                    hemi=hemi,
                                    valid_masks_by_subject=valid_masks,
                                    profile_fc_mode="pearson",
                                    network_taxonomy=network_taxonomy,
                                )
                            else:
                                profile = cluster_network_profile(
                                    group_labels_final=labels_final,
                                    support_mask=support,
                                    subjects=subjects,
                                    shared_root=shared_root,
                                    surface_root=surface_root,
                                    reference_root=reference_root,
                                    atlas=atlas,
                                    smoothing=smoothing,
                                    hemi=hemi,
                                    valid_masks_by_subject=valid_masks,
                                    profile_fc_mode="pearson",
                                    network_taxonomy=network_taxonomy,
                                )
                            status_rows, dominant = assignment_status_from_profile(profile, network_names)
                        if atlas_produces_cluster_roi_profile(atlas) and roi_profile is None:
                            roi_profile, roi_registry = cluster_roi_profile(
                                group_labels_final=labels_final,
                                support_mask=support,
                                subjects=subjects,
                                shared_root=shared_root,
                                surface_root=config.profile_surface_root,
                                reference_root=reference_root,
                                smoothing=smoothing,
                                hemi=hemi,
                                valid_masks_by_subject=valid_masks,
                                reference_atlas=atlas,
                            )
                        if atlas_produces_cluster_roi_profile(atlas) and roi_top20 is None:
                            assert roi_profile is not None and roi_registry is not None
                            roi_top20 = cluster_roi_top_k(roi_profile, roi_registry, k_top=20)

                        np.save(run_dir / "group_labels_final.npy", labels_final.astype(np.int32))
                        np.save(run_dir / "cluster_network_profile.npy", profile.astype(np.float32))

                        assignment_variant_specs: dict[str, dict[str, object]] = {
                            "fisherz": {
                                "status_rows": status_rows,
                                "dominant": dominant,
                                "assignment_metric": "top1_fisher_z",
                                "network_names": network_names, "network_taxonomy": network_taxonomy,
                                "profile_units": "Fisher_z",
                                "q_threshold": None,
                                "top_n_cap": None,
                            }
                        }
                        if roi_normalized_status_rows is not None:
                            assignment_variant_specs["normalized_top20_count"] = {
                                "status_rows": roi_normalized_status_rows,
                                "dominant": [str(row.get("top1_network", "unassigned")) for row in roi_normalized_status_rows],
                                "assignment_metric": "top20_count_normalized_by_network_roi_count",
                                "network_names": network_names, "network_taxonomy": network_taxonomy,
                                "profile_units": "normalized_positive_ROI_count",
                                "q_threshold": 0.05,
                                "top_n_cap": 20,
                            }
                        if ovr_variants:
                            if roi_profile_3d is None or roi_registry is None or roi_profile_subjects_used != subjects:
                                raise ValueError("OVR requires all requested subjects in the Kong2022 400-ROI subjectwise cluster profile")
                            assignment_variant_specs.update(build_ovr_assignment_specs(
                                roi_profile_3d=roi_profile_3d, subjects=roi_profile_subjects_used,
                                cluster_ids=list(range(1, k_final + 1)), roi_registry=roi_registry,
                                variants=ovr_variants, out_dir=run_dir,
                            ))
                        if atlas == "kong2022" and not is_clusterless_method(method):
                            if args.profile is not None or "network_ts_fisherz" in args.assignment_variant or any(v.endswith("_network_ts") for v in args.assignment_variant):
                                net_ts_3d, net_ts_mean, net_ts_subs = subjectwise_cluster_network_fisherz_profile(
                                    group_labels_final=labels_final,
                                    support_mask=support,
                                    subjects=subjects,
                                    surface_root=config.profile_surface_root,
                                    reference_root=reference_root,
                                    smoothing=smoothing,
                                    hemi=hemi,
                                    valid_masks_by_subject=valid_masks,
                                    network_names=network_names,
                                    network_column=resolve_kong2022_network_registry_column(network_taxonomy),
                                )
                                np.save(run_dir / "cluster_network_ts_profile.npy", net_ts_mean.astype(np.float32))
                                net_ts_status_rows, net_ts_dominant = assignment_status_from_weighted_network_profile(
                                    net_ts_mean,
                                    network_names,
                                )
                                assignment_variant_specs["network_ts_fisherz"] = {
                                    "status_rows": net_ts_status_rows,
                                    "dominant": net_ts_dominant,
                                    "assignment_metric": "top1_network_ts_fisher_z",
                                    "network_names": network_names,
                                    "network_taxonomy": network_taxonomy,
                                    "profile_units": "Fisher_z",
                                    "q_threshold": None,
                                    "top_n_cap": None,
                                }
                                net_ts_ovr_res = fit_network_ovr_from_network_profiles(
                                    network_profiles=net_ts_3d,
                                    subject_ids=net_ts_subs,
                                    cluster_ids=list(range(1, k_final + 1)),
                                    network_names=list(network_names),
                                    taxonomy=network_taxonomy,
                                )
                                net_ts_variant = network_ts_ovr_variant(network_taxonomy)
                                net_ts_ovr_paths = export_network_ovr(
                                    net_ts_ovr_res,
                                    run_dir,
                                    prefix=net_ts_variant,
                                )
                                assignment_variant_specs[net_ts_variant] = {
                                    "status_rows": net_ts_ovr_res["assignment_rows"],
                                    "dominant": [row["top1_network"] for row in net_ts_ovr_res["assignment_rows"]],
                                    "assignment_metric": OVR_ASSIGNMENT_METRIC,
                                    "q_threshold": None,
                                    "top_n_cap": None,
                                    "network_names": net_ts_ovr_res["network_names"],
                                    "network_taxonomy": network_taxonomy,
                                    "network_profile": str(net_ts_ovr_paths["beta"].resolve()),
                                    "profile_units": "raw_beta_per_Fisher_z_unit",
                                    "ovr_model_metadata": str(net_ts_ovr_paths["metadata"].resolve()),
                                }
                        default_assignment_variant = select_cluster_level_default_assignment_variant(
                            assignment_variant_specs
                        )
                        default_variant_spec = assignment_variant_specs[default_assignment_variant]

                        for variant_name, variant_spec in assignment_variant_specs.items():
                            payload = {
                                "items": [dict(x) for x in variant_spec["status_rows"]],
                                "assignment_metric": str(variant_spec["assignment_metric"]),
                                "network_names": variant_spec["network_names"],
                                "network_taxonomy": variant_spec["network_taxonomy"],
                                "profile_units": variant_spec["profile_units"],
                                "q_threshold": variant_spec["q_threshold"],
                                "top_n_cap": variant_spec["top_n_cap"],
                            }
                            write_json(run_dir / f"cluster_assignment_status_{variant_name}.json", payload)
                            if variant_name == "normalized_top20_count":
                                write_json(run_dir / "cluster_roi_topn_assignment_status.json", payload)
                        write_json(
                            run_dir / "cluster_dominant_network.json",
                            {
                                "dominant": default_variant_spec["dominant"],
                                "network_names": network_names,
                                "assignment_variant": default_assignment_variant,
                            },
                        )
                        if atlas_produces_cluster_roi_profile(atlas):
                            assert roi_profile is not None and roi_registry is not None and roi_top20 is not None
                            np.save(run_dir / "cluster_roi_profile.npy", roi_profile.astype(np.float32))
                            write_json(run_dir / "cluster_roi_top20.json", roi_top20)
                            write_tsv(
                                run_dir / "cluster_roi_registry.tsv",
                                roi_registry,
                                ["roi_id", "roi_name", "hemi", "family17", "subregion", "rank", "canonical8", "canonical7"],
                            )

                        functional_variant_payloads: dict[str, dict[str, object]] = {}
                        functional_labels = np.zeros_like(labels_final, dtype=np.int32)
                        functional_label_items: list[tuple[int, str, tuple[int, int, int, int]]] = []
                        for variant_name, variant_spec in assignment_variant_specs.items():
                            labels_variant, meta_variant, items_variant = functional_atlas_from_clusters(
                                group_labels_final=labels_final,
                                status_rows=[dict(x) for x in variant_spec["status_rows"]],
                                network_names=variant_spec["network_names"],
                                color_map=canonical_color_map,
                            )
                            labels_variant = labels_variant.astype(np.int32)
                            np.save(run_dir / f"group_functional_atlas_labels_{variant_name}.npy", labels_variant)
                            write_json(run_dir / f"group_functional_atlas_label_map_{variant_name}.json", meta_variant)
                            functional_variant_payloads[variant_name] = {
                                "assignment_metric": str(variant_spec["assignment_metric"]),
                                "network_names": variant_spec["network_names"],
                                "network_taxonomy": variant_spec["network_taxonomy"],
                                "profile_units": variant_spec["profile_units"],
                                "cluster_assignment_status": str((run_dir / f"cluster_assignment_status_{variant_name}.json").resolve()),
                                "group_functional_atlas_labels": str((run_dir / f"group_functional_atlas_labels_{variant_name}.npy").resolve()),
                                "group_functional_atlas_label_map": str((run_dir / f"group_functional_atlas_label_map_{variant_name}.json").resolve()),
                            }
                            if variant_name in ALL_OVR_VARIANTS:
                                functional_variant_payloads[variant_name].update({
                                    field: variant_spec[field] for field in
                                    ("network_profile", "ovr_model_metadata", "profile_units")
                                })
                            if variant_name == default_assignment_variant:
                                functional_labels = labels_variant
                                functional_label_items = items_variant

                        write_tsv(
                            run_dir / "k_selection_summary.tsv",
                            rows,
                            k_selection_fields,
                        )
                        write_json(
                            run_dir / "k_selection_summary.json",
                            {"rows": rows, "k_final": int(k_final), **primary_k},
                        )

                        # Template label for downstream overview rendering.
                        lbl = make_cluster_label_gifti(labels_final.astype(np.int32), int(k_final), colors)
                        lbl_path = (
                            run_dir
                            / "template"
                            / "workbench_assets"
                            / f"{template_label}_hemi-{hemi}_space-corobl_den-512_label-group_cluster_{method}_{atlas}_{smoothing}_cov-{tau_tag}.label.gii"
                        )
                        lbl_path.parent.mkdir(parents=True, exist_ok=True)
                        nib.save(lbl, str(lbl_path))
                        func_lbl_path = (
                            run_dir
                            / "template"
                            / "workbench_assets"
                            / f"{template_label}_hemi-{hemi}_space-corobl_den-512_label-group_functional_atlas_{method}_{atlas}_{smoothing}_cov-{tau_tag}.label.gii"
                        )
                        func_lbl_path.parent.mkdir(parents=True, exist_ok=True)
                        for variant_name, variant_spec in assignment_variant_specs.items():
                            labels_path = run_dir / f"group_functional_atlas_labels_{variant_name}.npy"
                            map_path = run_dir / f"group_functional_atlas_label_map_{variant_name}.json"
                            labels_variant = np.load(labels_path).astype(np.int32)
                            _labels_tmp, _meta_tmp, items_variant = functional_atlas_from_clusters(
                                group_labels_final=labels_final,
                                status_rows=[dict(x) for x in variant_spec["status_rows"]],
                                network_names=variant_spec["network_names"],
                                color_map=canonical_color_map,
                            )
                            func_lbl_variant = make_named_label_gifti(labels_variant, items_variant)
                            variant_lbl_path = func_lbl_path.with_name(f"{func_lbl_path.stem}_{variant_name}{func_lbl_path.suffix}")
                            nib.save(func_lbl_variant, str(variant_lbl_path))
                            suffixed_lbl_path = variant_lbl_path
                            functional_variant_payloads[variant_name].update(
                                {
                                    "group_functional_atlas_labels": str(labels_path.resolve()),
                                    "group_functional_atlas_label_map": str(map_path.resolve()),
                                    "template_functional_atlas_label": str(variant_lbl_path.resolve()),
                                    "template_functional_atlas_label_variant": str(suffixed_lbl_path.resolve()),
                                }
                            )

                        default_variant_spec = functional_variant_payloads[default_assignment_variant]
                        manifest = {
                            "status": "ok",
                            **code_provenance(),
                            "dataset": dataset,
                            "expected_source_root": str(expected_source_root),
                            **signal_provenance,
                            "profile": get_gradient_profile(args.profile).name if args.profile is not None else None,
                            "profile_role": gradient_profile_role(args.profile) if args.profile is not None else None,
                            "profile_earlier_label": gradient_profile_earlier_label(args.profile) if args.profile is not None else None,
                            "formal_profile": get_gradient_profile(args.profile).name if args.profile is not None else None,
                            "formal_profile_role": gradient_profile_role(args.profile) if args.profile is not None else None,
                            "formal_profile_earlier_label": gradient_profile_earlier_label(args.profile) if args.profile is not None else None,
                            "redundancy_threshold": float(redundancy_threshold),
                            "method": method,
                            "atlas": atlas,
                            "smoothing": smoothing,
                            "hemi": hemi,
                            "assignment_mode": "cluster_level",
                            "cluster_assignment_metric": str(default_variant_spec["assignment_metric"]),
                            "default_assignment_variant": default_assignment_variant,
                            "assignment_variants": functional_variant_payloads,
                            "cluster_roi_topn_assignment_metric": (
                                str(assignment_variant_specs["normalized_top20_count"]["assignment_metric"])
                                if "normalized_top20_count" in assignment_variant_specs
                                else None
                            ),
                            "k_selection_mode": str(args.k_selection_mode),
                            "k_final_rule": str(args.k_final_rule),
                            "force_final_k": int(args.force_final_k) if args.force_final_k is not None else None,
                            "default_k": int(args.default_k) if args.default_k is not None else None,
                            "coverage_threshold": float(tau),
                            "V_min_fraction": float(args.v_min_fraction),
                            "V_min_effective": int(v_min),
                            "feature_family": feature_family_for_method(method, atlas, gradient_components),
                            "fc_mode": fc_mode_for_method(method, atlas),
                            "gradient_components": [int(c) for c in gradient_components] if is_gradient_method(method) else None,
                            "actual_consumed_input_paths": actual_input_paths,
                            "shared_root": str(shared_root.resolve()),
                            "surface_root": str(surface_root.resolve()),
                            "profile_surface_root": str(config.profile_surface_root.resolve()),
                            "profile_fc_mode": "cluster_mean_timeseries",
                            "profile_source_mode": profile_source_mode,
                            "profile_source_root": str(profile_source_root) if profile_source_root is not None else None,
                            "profile_source_registry": str(profile_source_registry) if profile_source_registry is not None else None,
                            "reference_root": str(reference_root.resolve()),
                            "resume_input_signature_version": int(RESUME_SIGNATURE_VERSION),
                            "resume_input_signature": resume_input_signature,
                            "resume_input_signature_payload": resume_signature_payload,
                            "subject_fc_preprocessing": subject_fc_preprocessing,
                            "group_gradient_features_active": (
                                str((run_dir / "group_gradient_features_active.npy").resolve())
                                if is_gradient_method(method)
                                else None
                            ),
                            "group_gradient_feature_summary": (
                                str((run_dir / "group_gradient_feature_summary.json").resolve())
                                if is_gradient_method(method)
                                else None
                            ),
                            "group_feature_summary": str((run_dir / "group_feature_summary.json").resolve()),
                            "clustering_algorithm": (
                                "gradient-ward" if method in WARD_GRADIENT_METHODS else
                                method if method == "gradient-gmm" else
                                "shared_partition_with_orthogonal_rotation_per_graph_lowest_loss_of_10_starts"
                                if is_multigraph_method(method) else
                                "spectral_clustering_precomputed_affinity_"
                                f"{method_surface_affinity_metadata(method)[0].replace('-', '_')}_diag0_"
                                "kmeans_n_init_50"
                            ),
                            "surface_affinity": method_surface_affinity_metadata(method)[0],
                            "rbf_scale": float(args.rbf_scale) if method == "gradient-spectral-rbf" else None,
                            "bandwidth_rule": "rbf_scale * median(positive unique-undirected-mesh-edge distance)" if method == "gradient-spectral-rbf" else None,
                            "gradient_clustering": primary_row.get("gradient_clustering") if is_gradient_method(method) else None,
                            "affinity_description": method_surface_affinity_metadata(method)[1],
                            "affinity_graph_diagnostics": primary_row["affinity_graph_diagnostics"],
                            "candidate_k_range_requested": [int(min(k_values)), int(max(k_values))],
                            "candidate_k_range_effective": [int(min(k_eff)), int(max(k_eff))],
                            "candidate_k_values_evaluated": [int(r["k"]) for r in rows],
                            "redundancy_early_stop_enabled": bool(redundancy_early_stop_enabled),
                            "chosen_k_final": int(k_final),
                            "automatic_k_final": primary_k["automatic_k_final"],
                            "primary_k_source": str(primary_k["primary_k_source"]),
                            "undersized_default_k": primary_k.get("undersized_default_k"),
                            **gradient_computation_provenance(
                                group_gradient_meta, fixed_cluster_reuse=fixed_cluster_reuse),
                            "fixed_cluster_reuse": fixed_cluster_reuse,
                            "subject_split_seeds": split_seeds,
                            "null_seeds": null_seeds,
                            "strong_null_network_permutation_scope": "subject_specific",
                            "run_aware_diagnostic": "independent_runwise_group_representations",
                            "nontriviality_thresholds": {
                                "V_min_fraction": float(args.v_min_fraction),
                                "max_components_threshold": int(args.max_components_threshold),
                                "min_profile_coherence": float(args.min_profile_coherence),
                            },
                            "redundancy_aware_thresholds": {
                                "max_profile_similarity_lte": float(redundancy_threshold),
                                "redundant_pair_count_eq": 0,
                            },
                            "density_evidence": {
                                "surface_path": str(surfaces[hemi]),
                                "n_vertices": int(surface_meta[hemi]["n_vertices"]),
                                "n_faces": int(surface_meta[hemi]["n_faces"]),
                            },
                            "surface_topology_audit": topology_audit[hemi],
                            "network_taxonomy": network_taxonomy,
                            "network_registry_column": resolve_kong2022_network_registry_column(network_taxonomy),
                            "network_order": [str(x) for x in network_names],
                            "canonical_network_order_consistency_checked": network_taxonomy in {"canonical7", "lynch7"},
                            "network_order_consistency_checked": True,
                            "network_order_consensus_subject_count": int(len(subjects)),
                            "canonical_network_order_consensus_subject_count": int(len(subjects)) if network_taxonomy in {"canonical7", "lynch7"} else None,
                            "n_subjects": int(len(subjects)),
                            "subjects": [f"sub-{s}" for s in subjects],
                            "n_support_vertices": int(n_support),
                            "n_excluded_vertices": int(n_vertices - n_support),
                            "group_support_mask": str((run_dir / "group_support_mask.npy").resolve()),
                            "subject_coverage_count": str((run_dir / "subject_coverage_count.npy").resolve()),
                            "subject_coverage_fraction": str((run_dir / "subject_coverage_fraction.npy").resolve()),
                            "group_labels_final": str((run_dir / "group_labels_final.npy").resolve()),
                            "group_functional_atlas_labels": str(default_variant_spec["group_functional_atlas_labels"]),
                            "cluster_network_profile": str((run_dir / "cluster_network_profile.npy").resolve()),
                            "cluster_dominant_network": str((run_dir / "cluster_dominant_network.json").resolve()),
                            "cluster_assignment_status": str(default_variant_spec["cluster_assignment_status"]),
                            "cluster_assignment_status_fisherz": str(functional_variant_payloads["fisherz"]["cluster_assignment_status"]),
                            "cluster_assignment_status_normalized_top20_count": (
                                str(functional_variant_payloads["normalized_top20_count"]["cluster_assignment_status"])
                                if "normalized_top20_count" in functional_variant_payloads
                                else None
                            ),
                            "cluster_assignment_status_network_ts_fisherz": (
                                str(functional_variant_payloads["network_ts_fisherz"]["cluster_assignment_status"])
                                if "network_ts_fisherz" in functional_variant_payloads
                                else None
                            ),
                            "cluster_assignment_status_ovr_logistic_canonical7_network_ts": (
                                str(functional_variant_payloads[NETWORK_TS_OVR_VARIANT]["cluster_assignment_status"])
                                if NETWORK_TS_OVR_VARIANT in functional_variant_payloads
                                else None
                            ),
                            "cluster_assignment_status_ovr_logistic_family17_network_ts": (
                                str(functional_variant_payloads["ovr_logistic_family17_network_ts"]["cluster_assignment_status"])
                                if "ovr_logistic_family17_network_ts" in functional_variant_payloads
                                else None
                            ),
                            "cluster_network_ts_profile": (
                                str((run_dir / "cluster_network_ts_profile.npy").resolve())
                                if (run_dir / "cluster_network_ts_profile.npy").is_file()
                                else None
                            ),
                            "cluster_roi_topn_assignment_status": (
                                str((run_dir / "cluster_roi_topn_assignment_status.json").resolve())
                                if "normalized_top20_count" in assignment_variant_specs
                                else None
                            ),
                            "group_functional_atlas_label_map": str(default_variant_spec["group_functional_atlas_label_map"]),
                            "cluster_roi_profile": (
                                str((run_dir / "cluster_roi_profile.npy").resolve()) if atlas_produces_cluster_roi_profile(atlas) else None
                            ),
                            "cluster_roi_profile_contract": (
                                GRADIENT_PROFILE_CONTRACT_TAG
                                if atlas_produces_cluster_roi_profile(atlas)
                                else None
                            ),
                            "cluster_roi_top20": (
                                str((run_dir / "cluster_roi_top20.json").resolve()) if atlas_produces_cluster_roi_profile(atlas) else None
                            ),
                            "cluster_roi_registry": (
                                str((run_dir / "cluster_roi_registry.tsv").resolve()) if atlas_produces_cluster_roi_profile(atlas) else None
                            ),
                            "cluster_roi_profile_subjects_used": (
                                [f"sub-{s}" for s in roi_profile_subjects_used]
                                if roi_profile_subjects_used is not None
                                else None
                            ),
                            "template_subject": template_render_subject,
                            "template_source_mode": template_source_mode,
                            "template_source_note": template_source_note,
                            "cluster_label_ordering": "long_axis_head_to_tail",
                            "cluster_order_head_to_tail": fixed_cluster_order,
                            "template_left_or_right_surface": str(surfaces[hemi]),
                            "template_spec_path": str(template_spec_path),
                            "template_cluster_label": str(lbl_path.resolve()),
                            "template_functional_atlas_label": str(default_variant_spec["template_functional_atlas_label"]),
                        }
                        feature_summary: dict[str, object] = {
                            "network_names": network_names,
                            "N_ref": int(len(network_names)),
                            "network_taxonomy": network_taxonomy,
                            "network_registry_column": resolve_kong2022_network_registry_column(network_taxonomy),
                        }
                        if group_gradient_meta is not None:
                            feature_summary["gradient"] = group_gradient_meta
                        write_json(run_dir / "group_feature_summary.json", feature_summary)
                        write_json(manifest_path, manifest)

    remind_stale_figures(out_root)
    print(json.dumps({"out_root": str(out_root), "status": "ok"}, indent=2))
    return 0


def _validated_kong2022_network_values(
    roi_registry: list[dict[str, object]],
    network_column: str,
) -> tuple[str, list[str]]:
    column = resolve_kong2022_network_registry_column(network_column)
    if column == "canonical7":
        allowed = set(canonical_families_7_order())
    elif column == "canonical8":
        allowed = set(load_kong2022_merge_mapping(taxonomy="canonical8").values())
    elif column == "family17":
        allowed = set(load_kong2022_merge_mapping(taxonomy="canonical7").keys())
    else:
        raise ValueError(f"Unsupported Kong2022 ROI network column: {column!r}")
    values = [str(row.get(column, "")).strip() for row in roi_registry]
    invalid = sorted({value for value in values if value not in allowed})
    if invalid:
        raise ValueError(
            f"Kong2022 {column} registry contains empty or unknown network labels: {invalid}"
        )
    return column, values


if __name__ == "__main__":
    raise SystemExit(main())

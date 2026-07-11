#!/usr/bin/env python3
from __future__ import annotations

"""
Process Step Description:
Compute HippoMaps gradients. Calculates continuous functional gradients of the hippocampus based on functional connectivity data to reflect macroscale functional transition patterns.

Main Flow Path:
- 7th step in the numbered pipeline sequence (gradient computation step for gradient-spectral atlas):
  01_01_run_hippunfold_t1w_local.py -> 01_02_check_hippunfold_outputs.py -> 02_01_prepare_group_inputs.py -> 02_02_build_qc_manifest.py -> 03_01/03_02 reference building -> 04_01/04_02/04_03 FC materialization -> 05_01_compute_hippomaps_gradients.py (this step) -> 05_02 group atlas parcellation -> 06_xx presentation rendering -> 07_01 group matrix overview.

Inputs:
- Command-line arguments:
  - --routes: Workflow branches to process (choices include routea, routea_ctxreg, routea_ctxdrop, routeb, lynch2024_21net, lynch2024_8net; default: routea and routeb).
  - --shared-inputs-root: Directory containing route-specific shared reference inputs (default: `outputs_hcp/inputs/hcp/kong2022/shared_inputs`).
  - --surface-root: Path to BOLD surfaces root directory (default: `outputs_hcp/inputs/hcp/kong2022/shared_inputs/surface`).
  - --subjects-manifest: Path to QC-passed subject list manifest (default: `outputs_hcp/inputs/hcp/kong2022/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt`).
  - --smoothings: Smoothing levels to process: "2mm", "4mm".
  - --hemis: Hemispheres to process: "L", "R".
  - --coverage-threshold: Minimum coverage fraction to include vertices (default: 0.80).
  - --group-average-surfaces-dir: Path to template midthickness surfaces (default: `outputs_hcp/inputs/hcp/kong2022/shared_inputs/template`).
  - --out-root: Directory where computed gradients will be saved (default: `outputs_hcp/work/hippomaps_gradients`).
  - --hippomaps-root: Root path to external HippoMaps codebase (default: `external/hippomaps`).
  - --n-components: Number of gradient components to fit in BrainSpace (default: 5).
  - --selected-components: Comma-separated components to save in output GIFTI (default: "1,2,3").
  - --max-subjects: Max subjects to process (useful for smoke tests).
  - --force: Force recompute even if outputs exist.
- Physical inputs:
  - Subject active vertex-to-ROI/component FC matrices: `{shared_inputs_root}/{route_dir}/sub-<subject>/fc/{atlas}/{smoothing}/hemi_{hemi}/hipp_vertex_to_network_fc_active.npy`.
  - Subject active masks: `{surface_root}/sub-<subject>/surface/tsnr_gate/sub-<subject>_hemi-{hemi}_valid_mask.npy`.
  - Group template midthickness surfaces: `{group_average_surfaces_dir}/group-average_hemi-{hemi}_space-corobl_den-512_label-hipp_midthickness.surf.gii`.

Outputs:
- Command-line output / file outputs:
  - Gradient GIFTI file: `{out_root}/{route}/smooth-{smoothing}_hemi-{hemi}/group_gradients.func.gii` (containing selected gradient maps).
  - Gradient eigenvalues fractions JSON: `{out_root}/{route}/smooth-{smoothing}_hemi-{hemi}/group_gradients_eigenvalues.json`.
  - Group mean Fisher-z FC matrix: `{out_root}/{route}/smooth-{smoothing}_hemi-{hemi}/group_mean_fisherz_fc_active.npy`.
  - Gradient stage manifest: `{out_root}/{route}/smooth-{smoothing}_hemi-{hemi}/group_gradients_manifest.json`.
"""
import argparse
import json
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from lib.hippomaps_adapter import (
    DEFAULT_HIPPOMAPS_ROOT,
    ensure_hippomaps_on_path,
    DEFAULT_GRADIENT_RANDOM_STATE,
    brainspace_sparsity_for_feature_count,
    fit_gradientmaps_pearson,
    hippomaps_commit,
    hippomaps_import_path,
    reconstruct_full_surface,
)
from lib.traceability import StageSpec, run_stage
from lib.output_layout import (
    DEFAULT_HIPPOMAPS_GRADIENT_ROOT,
    DEFAULT_KONG2022_MANIFEST_ROOT,
    DEFAULT_KONG2022_SHARED_INPUTS_ROOT,
    DEFAULT_KONG2022_SURFACE_ROOT,
    DEFAULT_KONG2022_TEMPLATE_ROOT,
)
from lib.workflow_utils import write_json
from lib.io_utils import load_subject_ids
from lib.fc_math import fisher_z_transform_fc
from lib.surface_utils import read_surface_vertex_count


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_T1ONLY_ROOT = DEFAULT_KONG2022_SHARED_INPUTS_ROOT.parent
DEFAULT_SHARED_INPUTS = DEFAULT_KONG2022_SHARED_INPUTS_ROOT
DEFAULT_SURFACE_ROOT = DEFAULT_KONG2022_SURFACE_ROOT
DEFAULT_SUBJECTS = DEFAULT_KONG2022_MANIFEST_ROOT / "hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt"
DEFAULT_GROUP_AVG_SURFACES_DIR = DEFAULT_KONG2022_TEMPLATE_ROOT / "group_average_surfaces"
DEFAULT_OUT_ROOT = DEFAULT_HIPPOMAPS_GRADIENT_ROOT
ROUTE_TO_SHARED_DIR = {
    "routea": "routea",
    "routea_ctxreg": "routea_ctxreg",
    "routea_ctxdrop": "routea_ctxdrop",
    "routeb": "routeb",
    "lynch2024_21net": "pfm_nsi043_78",
    "lynch2024_8net": "pfm_nsi043_78",
}
ROUTE_TO_ATLAS = {
    "routea": "kong2022",
    "routea_ctxreg": "kong2022",
    "routea_ctxdrop": "kong2022",
    "routeb": "kong2022",
    "lynch2024_21net": "lynch2024_21net",
    "lynch2024_8net": "lynch2024_8net",
}
SMOOTHINGS = ("2mm", "4mm")
HEMIS = ("L", "R")


@dataclass
class GradientComboResult:
    route: str
    atlas: str
    smoothing: str
    hemi: str
    subjects: list[str]
    subjects_manifest: Path
    coverage_threshold: float
    surface_path: Path
    vertex_count: int
    support_mask: np.ndarray
    coverage_count: np.ndarray
    coverage_fraction: np.ndarray
    fc_paths: dict[str, str]
    group_fc_active: np.ndarray
    group_fc_full: np.ndarray
    gradients_active: np.ndarray
    gradients_full: np.ndarray
    lambdas: np.ndarray
    ev_fractions: np.ndarray
    run_dir: Path
    hippomaps_root: Path
    n_components: int
    gradient_random_state: int | None


def finite_pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float64).reshape(-1)
    bv = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(av) & np.isfinite(bv)
    if int(np.sum(mask)) < 2:
        return 0.0
    x = av[mask] - float(np.mean(av[mask]))
    y = bv[mask] - float(np.mean(bv[mask]))
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if not np.isfinite(denom) or denom <= 0.0:
        return 0.0
    corr = float(np.dot(x, y) / denom)
    if not np.isfinite(corr):
        return 0.0
    return float(np.clip(corr, -1.0, 1.0))


def best_abs_correlation_assignment(corr: np.ndarray) -> list[int]:
    mat = np.asarray(corr, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Expected square correlation matrix, got shape={mat.shape}")
    n_comp = int(mat.shape[0])
    best_perm: tuple[int, ...] | None = None
    best_score = -np.inf
    best_identity_hits = -1
    for perm in permutations(range(n_comp)):
        score = float(sum(abs(mat[target_idx, source_idx]) for target_idx, source_idx in enumerate(perm)))
        identity_hits = sum(1 for idx, source_idx in enumerate(perm) if idx == source_idx)
        if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and identity_hits > best_identity_hits):
            best_score = score
            best_identity_hits = int(identity_hits)
            best_perm = tuple(int(x) for x in perm)
    if best_perm is None:
        raise RuntimeError("Could not compute component assignment")
    return list(best_perm)


def align_gradient_components_to_reference(
    *,
    active: np.ndarray,
    full: np.ndarray,
    lambdas: np.ndarray,
    ev_fractions: np.ndarray,
    reference_full: np.ndarray,
    component_labels: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    active_arr = np.asarray(active, dtype=np.float32)
    full_arr = np.asarray(full, dtype=np.float32)
    ref_arr = np.asarray(reference_full, dtype=np.float32)
    lambda_arr = np.asarray(lambdas, dtype=np.float64).reshape(-1)
    ev_arr = np.asarray(ev_fractions, dtype=np.float64).reshape(-1)
    labels = [str(x) for x in component_labels]
    if active_arr.ndim != 2 or full_arr.ndim != 2 or ref_arr.ndim != 2:
        raise ValueError(
            f"Expected 2D active/full/reference arrays, got active={active_arr.shape} full={full_arr.shape} ref={ref_arr.shape}"
        )
    n_comp = len(labels)
    if n_comp == 0:
        raise ValueError("component_labels must not be empty")
    if active_arr.shape[1] < n_comp or full_arr.shape[1] < n_comp or ref_arr.shape[1] < n_comp:
        raise ValueError(
            f"Component count mismatch: active={active_arr.shape} full={full_arr.shape} ref={ref_arr.shape} labels={n_comp}"
        )
    if lambda_arr.size < n_comp or ev_arr.size < n_comp:
        raise ValueError(f"Lambda/EV vectors must contain at least {n_comp} values")

    corr = np.zeros((n_comp, n_comp), dtype=np.float64)
    for target_idx in range(n_comp):
        for source_idx in range(n_comp):
            corr[target_idx, source_idx] = finite_pearson_corr(ref_arr[:, target_idx], full_arr[:, source_idx])
    source_for_target = best_abs_correlation_assignment(corr)

    aligned_active = np.zeros((active_arr.shape[0], n_comp), dtype=np.float32)
    aligned_full = np.zeros((full_arr.shape[0], n_comp), dtype=np.float32)
    aligned_lambdas = np.zeros((n_comp,), dtype=np.float64)
    aligned_ev = np.zeros((n_comp,), dtype=np.float64)
    mapping: dict[str, str] = {}
    corr_by_label: dict[str, float] = {}
    flips: dict[str, bool] = {}
    for target_idx, source_idx in enumerate(source_for_target):
        label = labels[target_idx]
        source_label = labels[source_idx]
        raw_corr = float(corr[target_idx, source_idx])
        sign = -1.0 if raw_corr < 0.0 else 1.0
        aligned_active[:, target_idx] = (active_arr[:, source_idx] * sign).astype(np.float32)
        aligned_full[:, target_idx] = (full_arr[:, source_idx] * sign).astype(np.float32)
        aligned_lambdas[target_idx] = float(lambda_arr[source_idx])
        aligned_ev[target_idx] = float(ev_arr[source_idx])
        mapping[label] = source_label
        corr_by_label[label] = raw_corr
        flips[label] = bool(sign < 0.0)

    meta = {
        "matching_objective": "maximize_abs_pearson_correlation_to_anchor",
        "component_mapping": mapping,
        "correlation_before_flip": corr_by_label,
        "sign_flip_applied": flips,
        "raw_arrays_saved": False,
    }
    return aligned_active, aligned_full, aligned_lambdas, aligned_ev, meta


def group_mean_fisherz_fc(
    *,
    feature_bank: dict[str, np.ndarray],
    subjects: list[str],
    support_mask: np.ndarray,
) -> np.ndarray:
    idx = np.flatnonzero(np.asarray(support_mask, dtype=bool))
    if idx.size == 0:
        raise RuntimeError("Support mask has zero vertices")
    mats = []
    for subject in subjects:
        mats.append(fisher_z_transform_fc(feature_bank[subject][idx, :]))
    group_fc = np.nanmean(np.stack(mats, axis=0).astype(np.float32), axis=0)
    group_fc[~np.isfinite(group_fc)] = 0.0
    return group_fc.astype(np.float32)



def subject_valid_mask(surface_root: Path, subject: str, hemi: str) -> np.ndarray:
    path = surface_root / f"sub-{subject}" / "surface" / "tsnr_gate" / f"sub-{subject}_hemi-{hemi}_valid_mask.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing valid mask: {path}")
    return np.load(path).astype(bool)


def atlas_for_gradient_route(route: str) -> str:
    route_key = str(route).strip().lower()
    if route_key not in ROUTE_TO_ATLAS:
        raise ValueError(f"Unsupported gradient route: {route!r}")
    return ROUTE_TO_ATLAS[route_key]


def resolve_gradient_shared_root(shared_inputs_root: Path, route: str) -> Path:
    route_key = str(route).strip().lower()
    if route_key not in ROUTE_TO_SHARED_DIR:
        raise ValueError(f"Unsupported gradient route: {route!r}")
    return Path(shared_inputs_root).resolve() / ROUTE_TO_SHARED_DIR[route_key]


def subject_fc_path(shared_root: Path, subject: str, atlas: str, smoothing: str, hemi: str) -> Path:
    return Path(shared_root) / f"sub-{subject}" / "fc" / str(atlas) / smoothing / f"hemi_{hemi}" / "hipp_vertex_to_network_fc.npy"


def load_subject_fc(shared_root: Path, subject: str, atlas: str, smoothing: str, hemi: str, valid_mask: np.ndarray) -> tuple[np.ndarray, Path]:
    fc_path = subject_fc_path(shared_root, subject, atlas, smoothing, hemi)
    if not fc_path.exists():
        raise FileNotFoundError(f"Missing vertex-to-network FC cache: {fc_path}")
    arr = np.load(fc_path).astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D FC cache: {fc_path} shape={arr.shape}")
    if arr.shape[0] == valid_mask.size:
        return arr, fc_path
    if arr.shape[0] == int(np.sum(valid_mask)):
        full = np.full((valid_mask.size, arr.shape[1]), np.nan, dtype=np.float32)
        full[np.flatnonzero(valid_mask), :] = arr
        return full, fc_path
    raise ValueError(
        f"Unexpected FC cache rows for {fc_path}: rows={arr.shape[0]} "
        f"vertex_count={valid_mask.size} valid_count={int(np.sum(valid_mask))}"
    )


def parse_components(raw: str) -> list[int]:
    comps = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not comps:
        raise ValueError("--selected-components must not be empty")
    if any(c < 1 for c in comps):
        raise ValueError(f"Components must be one-based positive integers, got {raw!r}")
    if len(set(comps)) != len(comps):
        raise ValueError(f"Duplicate components are not allowed: {raw!r}")
    return comps


def compute_combo(
    *,
    route: str,
    shared_root: Path,
    atlas: str,
    surface_root: Path,
    subjects: list[str],
    subjects_manifest: Path,
    smoothing: str,
    hemi: str,
    coverage_threshold: float,
    group_average_surfaces_dir: Path,
    out_root: Path,
    hippomaps_root: Path,
    n_components: int,
    selected_components: list[int],
    gradient_random_state: int | None,
    force: bool,
) -> GradientComboResult:
    surface_path = (
        group_average_surfaces_dir
        / f"group-average_hemi-{hemi}_space-corobl_den-512_label-hipp_midthickness.surf.gii"
    )
    if not surface_path.exists():
        raise FileNotFoundError(f"Missing group-average surface: {surface_path}")
    vertex_count = read_surface_vertex_count(surface_path)
    valid_masks = {s: subject_valid_mask(surface_root, s, hemi) for s in subjects}
    for subject, mask in valid_masks.items():
        if mask.size != vertex_count:
            raise ValueError(f"Valid mask vertex count mismatch for sub-{subject}: {mask.size} vs {vertex_count}")
    coverage_count = np.sum(np.stack([valid_masks[s] for s in subjects], axis=0), axis=0).astype(np.int32)
    coverage_fraction = (coverage_count.astype(np.float32) / float(len(subjects))).astype(np.float32)
    support_mask = coverage_fraction >= float(coverage_threshold)

    feature_bank: dict[str, np.ndarray] = {}
    fc_paths: dict[str, str] = {}
    for subject in subjects:
        fc, fc_path = load_subject_fc(shared_root, subject, atlas, smoothing, hemi, valid_masks[subject])
        feature_bank[subject] = fc
        fc_paths[f"sub-{subject}"] = str(fc_path.resolve())

    group_fc_active = group_mean_fisherz_fc(
        feature_bank=feature_bank,
        subjects=subjects,
        support_mask=support_mask,
    )
    gradients_active, lambdas, ev_fractions = fit_gradientmaps_pearson(
        group_fc_active,
        n_components=int(n_components),
        random_state=gradient_random_state,
    )
    gradients_full = reconstruct_full_surface(gradients_active, support_mask, vertex_count)
    group_fc_full = reconstruct_full_surface(group_fc_active, support_mask, vertex_count)

    run_dir = out_root / route / smoothing / f"hemi_{hemi}"
    manifest_path = run_dir / "hippomaps_gradient_manifest.json"
    if manifest_path.exists() and not force:
        raise FileExistsError(f"Output manifest exists; pass --force to overwrite: {manifest_path}")

    return GradientComboResult(
        route=route,
        atlas=atlas,
        smoothing=smoothing,
        hemi=hemi,
        subjects=list(subjects),
        subjects_manifest=subjects_manifest,
        coverage_threshold=float(coverage_threshold),
        surface_path=surface_path.resolve(),
        vertex_count=int(vertex_count),
        support_mask=support_mask.astype(bool),
        coverage_count=coverage_count.astype(np.int32),
        coverage_fraction=coverage_fraction.astype(np.float32),
        fc_paths=fc_paths,
        group_fc_active=group_fc_active.astype(np.float32),
        group_fc_full=group_fc_full.astype(np.float32),
        gradients_active=gradients_active.astype(np.float32),
        gradients_full=gradients_full.astype(np.float32),
        lambdas=lambdas.astype(np.float64),
        ev_fractions=ev_fractions.astype(np.float64),
        run_dir=run_dir,
        hippomaps_root=hippomaps_root,
        n_components=int(n_components),
        gradient_random_state=gradient_random_state,
    )


def select_alignment_anchor(results: list[GradientComboResult]) -> tuple[GradientComboResult, str]:
    if not results:
        raise ValueError("No gradient combinations were computed for alignment")
    for result in results:
        if result.smoothing == "4mm" and result.hemi == "L":
            return result, "preferred_4mm_hemi_L"
    return results[0], "fallback_first_requested_combo"


def write_aligned_combo_outputs(
    *,
    result: GradientComboResult,
    gradients_active: np.ndarray,
    gradients_full: np.ndarray,
    lambdas: np.ndarray,
    ev_fractions: np.ndarray,
    selected_components: list[int],
    alignment_meta: dict[str, Any],
) -> Path:
    run_dir = result.run_dir
    manifest_path = run_dir / "hippomaps_gradient_manifest.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    np.save(run_dir / "group_support_mask.npy", result.support_mask.astype(np.uint8))
    np.save(run_dir / "subject_coverage_count.npy", result.coverage_count.astype(np.int32))
    np.save(run_dir / "subject_coverage_fraction.npy", result.coverage_fraction.astype(np.float32))
    np.save(run_dir / "group_mean_fisherz_fc_active.npy", result.group_fc_active.astype(np.float32))
    np.save(run_dir / "group_mean_fisherz_fc_full.npy", result.group_fc_full.astype(np.float32))
    np.save(run_dir / "hippomaps_gradients_active.npy", gradients_active.astype(np.float32))
    np.save(run_dir / "hippomaps_gradients_full.npy", gradients_full.astype(np.float32))
    np.save(run_dir / "hippomaps_lambdas.npy", lambdas.astype(np.float64))
    np.save(run_dir / "hippomaps_ev_fractions.npy", ev_fractions.astype(np.float64))
    for comp in selected_components:
        comp_idx = int(comp) - 1
        if comp_idx >= gradients_full.shape[1]:
            raise ValueError(f"Selected component G{comp} exceeds computed gradient columns={gradients_full.shape[1]}")
        np.save(run_dir / f"hippomaps_gradient_G{comp}_full.npy", gradients_full[:, comp_idx].astype(np.float32))

    selected_labels = [f"G{c}" for c in selected_components]
    manifest = {
        "status": "ok",
        "route": result.route,
        "atlas": result.atlas,
        "smoothing": result.smoothing,
        "hemi": result.hemi,
        "method": "hippomaps-gradient",
        "product_role": "continuous_upstream_gradient",
        "gradient_array_semantics": "component_and_sign_aligned_to_route_anchor",
        "clustering_status": "consumed_by_05_02_gradient_spectral_when_gradient_root_points_here",
        "subjects_manifest": str(result.subjects_manifest.resolve()),
        "n_subjects": int(len(result.subjects)),
        "subjects": [f"sub-{s}" for s in result.subjects],
        "coverage_threshold": float(result.coverage_threshold),
        "n_vertices": int(result.vertex_count),
        "n_support_vertices": int(np.sum(result.support_mask)),
        "n_excluded_vertices": int(result.vertex_count - np.sum(result.support_mask)),
        "surface_path": str(result.surface_path.resolve()),
        "source_fc_paths": result.fc_paths,
        "fisher_z_preprocessing": "arctanh clipped to +/-0.999999 before subject averaging",
        "feature_zscore_pre_gradientmaps": "column-wise z-score across vertices after group Fisher-z averaging, matching HippoMaps tutorial feature scaling semantics",
        "group_fc_shape": [int(x) for x in result.group_fc_active.shape],
        "gradient_shape_active": [int(x) for x in gradients_active.shape],
        "gradient_shape_full": [int(x) for x in gradients_full.shape],
        "gradientmaps_kernel": "pearson",
        "gradientmaps_call": "brainspace.gradient.GradientMaps(n_components=N, kernel='pearson', random_state=gradient_random_state).fit(column_zscored_group_mean_fisherz_fc, sparsity=brainspace_sparsity)",
        "brainspace_sparsity": float(brainspace_sparsity_for_feature_count(result.group_fc_active.shape[1])),
        "brainspace_sparsity_policy": "0.0 for <=50 features; otherwise 0.9. Low-dimensional Lynch component banks use the full graph, while high-dimensional Kong400 retains BrainSpace default sparsity",
        "gradient_random_state": None if result.gradient_random_state is None else int(result.gradient_random_state),
        "n_components_computed": int(result.n_components),
        "selected_components": selected_labels,
        "lambdas": [float(x) for x in lambdas],
        "ev_fractions": [float(x) for x in ev_fractions],
        "gradient_alignment": alignment_meta,
        "hippomaps_repo": str(Path(result.hippomaps_root).resolve()),
        "hippomaps_import_path": hippomaps_import_path(result.hippomaps_root),
        "hippomaps_commit": hippomaps_commit(result.hippomaps_root),
        "outputs": {
            "group_support_mask": str((run_dir / "group_support_mask.npy").resolve()),
            "group_mean_fisherz_fc_active": str((run_dir / "group_mean_fisherz_fc_active.npy").resolve()),
            "group_mean_fisherz_fc_full": str((run_dir / "group_mean_fisherz_fc_full.npy").resolve()),
            "hippomaps_gradients_active": str((run_dir / "hippomaps_gradients_active.npy").resolve()),
            "hippomaps_gradients_full": str((run_dir / "hippomaps_gradients_full.npy").resolve()),
            "hippomaps_lambdas": str((run_dir / "hippomaps_lambdas.npy").resolve()),
            "hippomaps_ev_fractions": str((run_dir / "hippomaps_ev_fractions.npy").resolve()),
        },
    }
    write_json(manifest_path, manifest, sort_keys=True, trailing_newline=True)
    print(json.dumps({"status": "ok", "manifest": str(manifest_path), "n_subjects": len(result.subjects)}, indent=2), flush=True)
    return manifest_path


def align_and_write_route_results(*, results: list[GradientComboResult], selected_components: list[int]) -> list[Path]:
    anchor, anchor_selection = select_alignment_anchor(results)
    component_labels = [f"G{i + 1}" for i in range(int(anchor.gradients_full.shape[1]))]
    written: list[Path] = []
    for result in results:
        aligned_active, aligned_full, aligned_lambdas, aligned_ev, combo_alignment = align_gradient_components_to_reference(
            active=result.gradients_active,
            full=result.gradients_full,
            lambdas=result.lambdas,
            ev_fractions=result.ev_fractions,
            reference_full=anchor.gradients_full,
            component_labels=component_labels,
        )
        alignment_meta = {
            "enabled": True,
            "anchor_route": anchor.route,
            "anchor_smoothing": anchor.smoothing,
            "anchor_hemi": anchor.hemi,
            "anchor_selection": anchor_selection,
            **combo_alignment,
        }
        written.append(
            write_aligned_combo_outputs(
                result=result,
                gradients_active=aligned_active,
                gradients_full=aligned_full,
                lambdas=aligned_lambdas,
                ev_fractions=aligned_ev,
                selected_components=selected_components,
                alignment_meta=alignment_meta,
            )
        )
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute HippoMaps-style continuous gradients from Kong2022 FC caches.")
    parser.add_argument("--routes", nargs="+", choices=sorted(ROUTE_TO_SHARED_DIR), default=["routea", "routeb"])
    parser.add_argument("--shared-inputs-root", default=str(DEFAULT_SHARED_INPUTS))
    parser.add_argument("--surface-root", default=str(DEFAULT_SURFACE_ROOT))
    parser.add_argument("--subjects-manifest", default=str(DEFAULT_SUBJECTS))
    parser.add_argument("--smoothings", nargs="+", choices=SMOOTHINGS, default=list(SMOOTHINGS))
    parser.add_argument("--hemis", nargs="+", choices=HEMIS, default=list(HEMIS))
    parser.add_argument("--coverage-threshold", type=float, default=0.80)
    parser.add_argument("--group-average-surfaces-dir", default=str(DEFAULT_GROUP_AVG_SURFACES_DIR))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--hippomaps-root", default=str(DEFAULT_HIPPOMAPS_ROOT))
    parser.add_argument("--n-components", type=int, default=5)
    parser.add_argument(
        "--gradient-random-state",
        type=int,
        default=DEFAULT_GRADIENT_RANDOM_STATE,
        help="Random seed passed to BrainSpace GradientMaps for reproducible diffusion initialization.",
    )
    parser.add_argument("--selected-components", default="1,2,3")
    parser.add_argument("--max-subjects", type=int, default=None, help="Optional smoke-test cap after loading the manifest.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hippomaps_root = ensure_hippomaps_on_path(Path(args.hippomaps_root))
    subjects_manifest = Path(args.subjects_manifest).resolve()
    subjects = load_subject_ids(subjects_manifest)
    if args.max_subjects is not None:
        subjects = subjects[: int(args.max_subjects)]
    selected_components = parse_components(args.selected_components)
    for comp in selected_components:
        if comp > int(args.n_components):
            raise ValueError(f"Selected G{comp} exceeds --n-components={args.n_components}")
    for route in args.routes:
        shared_root = resolve_gradient_shared_root(Path(args.shared_inputs_root), route)
        atlas = atlas_for_gradient_route(route)
        route_results: list[GradientComboResult] = []
        for smoothing in args.smoothings:
            for hemi in args.hemis:
                route_results.append(
                    compute_combo(
                        route=route,
                        shared_root=shared_root,
                        atlas=atlas,
                        surface_root=Path(args.surface_root).resolve(),
                        subjects=subjects,
                        subjects_manifest=subjects_manifest,
                        smoothing=smoothing,
                        hemi=hemi,
                        coverage_threshold=float(args.coverage_threshold),
                        group_average_surfaces_dir=Path(args.group_average_surfaces_dir).resolve(),
                        out_root=Path(args.out_root).resolve(),
                        hippomaps_root=hippomaps_root,
                        n_components=int(args.n_components),
                        selected_components=selected_components,
                        gradient_random_state=int(args.gradient_random_state),
                        force=bool(args.force),
                    )
                )
        align_and_write_route_results(results=route_results, selected_components=selected_components)
    return 0


if __name__ == "__main__":
    raise SystemExit(
        run_stage(
            StageSpec(
                step_id="05_01_compute_hippomaps_gradients",
                purpose="Compute HippoMaps-style continuous gradients from group mean Fisher-z FC.",
                stage_main=main,
                upstream_defaults=("04_01_materialize_kong2022_roi_fc",),
            )
        )
    )

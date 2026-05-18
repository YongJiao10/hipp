#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from lib.hippomaps_adapter import (
    DEFAULT_HIPPOMAPS_ROOT,
    ensure_hippomaps_on_path,
    fit_gradientmaps_pearson,
    hippomaps_commit,
    hippomaps_import_path,
    reconstruct_full_surface,
)
from lib.traceability import StageSpec, run_stage


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_T1ONLY_ROOT = REPO_ROOT / "outputs_experiments" / "hippunfold_recommended_t1w_20260423" / "kong2022"
DEFAULT_SHARED_INPUTS = DEFAULT_T1ONLY_ROOT / "shared_inputs"
DEFAULT_SURFACE_ROOT = DEFAULT_SHARED_INPUTS / "surface_151"
DEFAULT_SUBJECTS = DEFAULT_T1ONLY_ROOT / "manifests" / "hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt"
DEFAULT_GROUP_AVG_SURFACES_DIR = (
    DEFAULT_T1ONLY_ROOT
    / "templates"
    / "hipp_group_average_t1only_151"
    / "group_average_surfaces"
)
DEFAULT_OUT_ROOT = DEFAULT_T1ONLY_ROOT / "hippomaps_gradients"
ROUTE_TO_SHARED_DIR = {"routea": "routea_151", "routeb": "routeb_151"}
SMOOTHINGS = ("2mm", "4mm")
HEMIS = ("L", "R")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_subjects(path: Path) -> list[str]:
    subjects = [line.strip().replace("sub-", "") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not subjects:
        raise RuntimeError(f"No subjects in manifest: {path}")
    return subjects


def fisher_z_transform_fc(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    clipped = np.clip(arr, -0.999999, 0.999999)
    out = np.arctanh(clipped).astype(np.float32)
    out[~np.isfinite(arr)] = np.nan
    return out


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


def read_surface_vertex_count(surface_path: Path) -> int:
    img = nib.load(str(surface_path))
    for da in img.darrays:
        if da.intent == "NIFTI_INTENT_POINTSET":
            return int(np.asarray(da.data).shape[0])
    return int(np.asarray(img.darrays[0].data).shape[0])


def subject_valid_mask(surface_root: Path, subject: str, hemi: str) -> np.ndarray:
    path = surface_root / f"sub-{subject}" / "surface" / "tsnr_gate" / f"sub-{subject}_hemi-{hemi}_valid_mask.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing valid mask: {path}")
    return np.load(path).astype(bool)


def load_subject_fc(shared_root: Path, subject: str, smoothing: str, hemi: str, valid_mask: np.ndarray) -> tuple[np.ndarray, Path]:
    fc_path = shared_root / f"sub-{subject}" / "fc" / "kong2022" / smoothing / f"hemi_{hemi}" / "hipp_vertex_to_network_fc.npy"
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
    force: bool,
) -> Path:
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
        fc, fc_path = load_subject_fc(shared_root, subject, smoothing, hemi, valid_masks[subject])
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
    )
    gradients_full = reconstruct_full_surface(gradients_active, support_mask, vertex_count)
    group_fc_full = reconstruct_full_surface(group_fc_active, support_mask, vertex_count)

    run_dir = out_root / route / smoothing / f"hemi_{hemi}"
    manifest_path = run_dir / "hippomaps_gradient_manifest.json"
    if manifest_path.exists() and not force:
        raise FileExistsError(f"Output manifest exists; pass --force to overwrite: {manifest_path}")
    run_dir.mkdir(parents=True, exist_ok=True)

    np.save(run_dir / "group_support_mask.npy", support_mask.astype(np.uint8))
    np.save(run_dir / "subject_coverage_count.npy", coverage_count.astype(np.int32))
    np.save(run_dir / "subject_coverage_fraction.npy", coverage_fraction.astype(np.float32))
    np.save(run_dir / "group_mean_fisherz_fc_active.npy", group_fc_active.astype(np.float32))
    np.save(run_dir / "group_mean_fisherz_fc_full.npy", group_fc_full.astype(np.float32))
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
        "route": route,
        "atlas": "kong2022",
        "smoothing": smoothing,
        "hemi": hemi,
        "method": "hippomaps-gradient",
        "product_role": "continuous_upstream_gradient",
        "clustering_status": "not_changed_downstream_gradient_spectral_remains_separate",
        "subjects_manifest": str(subjects_manifest.resolve()),
        "n_subjects": int(len(subjects)),
        "subjects": [f"sub-{s}" for s in subjects],
        "coverage_threshold": float(coverage_threshold),
        "n_vertices": int(vertex_count),
        "n_support_vertices": int(np.sum(support_mask)),
        "n_excluded_vertices": int(vertex_count - np.sum(support_mask)),
        "surface_path": str(surface_path.resolve()),
        "source_fc_paths": fc_paths,
        "fisher_z_preprocessing": "arctanh clipped to +/-0.999999 before subject averaging",
        "feature_zscore_pre_gradientmaps": "column-wise z-score across vertices after group Fisher-z averaging, matching HippoMaps tutorial feature scaling semantics",
        "group_fc_shape": [int(x) for x in group_fc_active.shape],
        "gradient_shape_active": [int(x) for x in gradients_active.shape],
        "gradient_shape_full": [int(x) for x in gradients_full.shape],
        "gradientmaps_kernel": "pearson",
        "gradientmaps_call": "brainspace.gradient.GradientMaps(n_components=N, kernel='pearson').fit(column_zscored_group_mean_fisherz_fc) with BrainSpace default sparsity=0.9",
        "n_components_computed": int(n_components),
        "selected_components": selected_labels,
        "lambdas": [float(x) for x in lambdas],
        "ev_fractions": [float(x) for x in ev_fractions],
        "hippomaps_repo": str(Path(hippomaps_root).resolve()),
        "hippomaps_import_path": hippomaps_import_path(hippomaps_root),
        "hippomaps_commit": hippomaps_commit(hippomaps_root),
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
    save_json(manifest_path, manifest)
    print(json.dumps({"status": "ok", "manifest": str(manifest_path), "n_subjects": len(subjects)}, indent=2), flush=True)
    return manifest_path


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
    parser.add_argument("--selected-components", default="1,2,3")
    parser.add_argument("--max-subjects", type=int, default=None, help="Optional smoke-test cap after loading the manifest.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hippomaps_root = ensure_hippomaps_on_path(Path(args.hippomaps_root))
    subjects_manifest = Path(args.subjects_manifest).resolve()
    subjects = load_subjects(subjects_manifest)
    if args.max_subjects is not None:
        subjects = subjects[: int(args.max_subjects)]
    selected_components = parse_components(args.selected_components)
    for comp in selected_components:
        if comp > int(args.n_components):
            raise ValueError(f"Selected G{comp} exceeds --n-components={args.n_components}")
    for route in args.routes:
        shared_root = Path(args.shared_inputs_root).resolve() / ROUTE_TO_SHARED_DIR[route]
        for smoothing in args.smoothings:
            for hemi in args.hemis:
                compute_combo(
                    route=route,
                    shared_root=shared_root,
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
                    force=bool(args.force),
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(
        run_stage(
            StageSpec(
                step_id="05_03_compute_hippomaps_gradients",
                purpose="Compute HippoMaps-style continuous gradients from group mean Fisher-z FC.",
                stage_main=main,
                upstream_defaults=("04_01_materialize_kong2022_roi_fc",),
            )
        )
    )

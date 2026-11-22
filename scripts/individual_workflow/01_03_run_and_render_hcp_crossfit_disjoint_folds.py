#!/usr/bin/env python3
"""Compute and render 5 completely disjoint, non-overlapping fold templates for HCP."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from nibabel.gifti import GiftiDataArray, GiftiImage, GiftiLabel, GiftiLabelTable
from PIL import Image, ImageDraw
from sklearn.metrics import adjusted_rand_score

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_WORKFLOW_DIR = REPO_ROOT / "scripts" / "core_workflow"
INDIVIDUAL_WORKFLOW_DIR = REPO_ROOT / "scripts" / "individual_workflow"
if str(CORE_WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_WORKFLOW_DIR))
if str(INDIVIDUAL_WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(INDIVIDUAL_WORKFLOW_DIR))

from lib.output_layout import (
    DEFAULT_KONG2022_ROUTEA_ROOT,
    DEFAULT_KONG2022_SURFACE_ROOT,
    DEFAULT_KONG2022_TEMPLATE_ROOT,
    resolve_crossfit_template_matching_overview_root,
    resolve_crossfit_template_matching_root,
)
from lib.surface_utils import render_hippocampal_grid, template_anterior_coordinate
from lib.workbench_grid import build_grid_canvas, load_font

DEFAULT_SMOOTHINGS = ("2mm", "4mm")
DEFAULT_HEMIS = ("L", "R")
DEFAULT_N_FOLDS = 5
from lib.gradient_clustering import GRADIENT_METHODS
from lib.hippomaps_adapter import (
    gradient_preprocessing_settings, scale_gradient_features, validate_gradient_preprocessing,
)

DEFAULT_K = 3
DEFAULT_SCENE = REPO_ROOT / "config" / "wb_locked_native_view_lateral_medial.scene"
DEFAULT_FIGURE_NAME = "hcp_routea_crossfit_k3_disjoint_folds_workbench_comparison.png"
DEFAULT_MANIFEST_NAME = "hcp_routea_crossfit_k3_disjoint_folds_workbench_comparison_manifest.json"
DEFAULT_SUMMARY_TSV_NAME = "disjoint_folds_robustness_summary.tsv"

DEFAULT_SURF_L = (
    DEFAULT_KONG2022_TEMPLATE_ROOT
    / "group_average_surfaces"
    / "group-average_hemi-L_space-corobl_den-512_label-hipp_midthickness.surf.gii"
)
DEFAULT_SURF_R = (
    DEFAULT_KONG2022_TEMPLATE_ROOT
    / "group_average_surfaces"
    / "group-average_hemi-R_space-corobl_den-512_label-hipp_midthickness.surf.gii"
)
DEFAULT_SPEC = (
    REPO_ROOT
    / "outputs_hcp"
    / "inputs"
    / "hippunfold"
    / "dense_corobl_batch"
    / "sub-100610"
    / "hippunfold"
    / "sub-100610"
    / "surf"
    / "sub-100610_den-512_label-hipp_surfaces.spec"
)

DISPLAY_LABEL_SPECS = {
    0: ("Null", (158, 158, 158, 255)),
    1: ("Anterior", (220, 92, 255, 255)),
    2: ("Middle", (99, 150, 194, 255)),
    3: ("Posterior", (216, 92, 107, 255)),
}


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def write_display_label_gifti(
    labels: np.ndarray,
    output_path: Path,
) -> Path:
    table = GiftiLabelTable()
    for key, (name, rgba) in DISPLAY_LABEL_SPECS.items():
        label = GiftiLabel(
            key=int(key),
            red=float(rgba[0]) / 255.0,
            green=float(rgba[1]) / 255.0,
            blue=float(rgba[2]) / 255.0,
            alpha=float(rgba[3]) / 255.0,
        )
        label.label = str(name)
        table.labels.append(label)

    out_img = GiftiImage(
        darrays=[GiftiDataArray(data=labels.astype(np.int32), intent="NIFTI_INTENT_LABEL")],
        labeltable=table,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(output_path))
    return output_path


def fit_disjoint_cohort(
    *,
    subjects: list[str],
    smoothing: str,
    hemi: str,
    core_atlas_module: Any,
    group_grad_module: Any,
    cf_module: Any,
    surface_path: Path,
    gradient_options: dict[str, Any] | None = None,
    alignment_reference: np.ndarray | None = None,
) -> dict[str, Any]:
    coords, faces = core_atlas_module.load_surface(surface_path)
    vertex_count = int(coords.shape[0])
    valid_masks = {}
    feature_bank = {}
    for s in subjects:
        m_path = DEFAULT_KONG2022_SURFACE_ROOT / f"sub-{s}" / "surface" / "validity" / f"sub-{s}_hemi-{hemi}_valid_mask.npy"
        fc_path = DEFAULT_KONG2022_ROUTEA_ROOT / f"sub-{s}" / "fc" / "kong2022" / smoothing / f"hemi_{hemi}" / "hipp_vertex_to_network_fc.npy"
        mask = np.load(m_path).astype(bool).reshape(-1)
        valid_masks[s] = mask
        feature_bank[s] = cf_module._load_full_subject_fc(fc_path, mask)

    cov_count = np.sum(np.stack([valid_masks[s] for s in subjects], axis=0), axis=0)
    cov_frac = cov_count.astype(np.float32) / float(len(subjects))
    support = cov_frac >= 0.80

    group_fc_act = group_grad_module.group_mean_fisherz_fc(
        feature_bank=feature_bank,
        subjects=subjects,
        support_mask=support,
    )
    options = dict(gradient_options or {})
    grad_act, lambdas, evs = cf_module.fit_diffusion_gradients(
        group_fc_act, n_components=5,
        feature_scaling=options.get("feature_scaling", "col_zscore"),
        sparsify_stage=options.get("sparsify_stage", "features"), sparsity=options.get("sparsity"),
    )
    gradients_full = cf_module.reconstruct_full_surface(grad_act, support, vertex_count)
    if alignment_reference is not None:
        grad_act, gradients_full, lambdas, evs, _ = group_grad_module.align_gradient_components_to_reference(
            active=grad_act, full=gradients_full, lambdas=lambdas, ev_fractions=evs,
            reference_full=alignment_reference,
            component_labels=[f"G{i+1}" for i in range(grad_act.shape[1])],
        )
    cluster_features, transform = scale_gradient_features(
        grad_act[:, :DEFAULT_K], scaling=options.get("clustering_feature_scaling", "none"),
    )
    adj = core_atlas_module.build_surface_adjacency(faces, vertex_count)
    _, labels_full = core_atlas_module.cluster_from_precomputed_group_features(
        method=options.get("method", "gradient-spectral"),
        group_features=np.asarray(cluster_features, dtype=np.float32),
        support_mask=support,
        adjacency_full=adj,
        k=DEFAULT_K,
        seed=int(options.get("cluster_seed", 0)),
        rbf_scale=float(options.get("rbf_scale", 1.0)),
    )
    labels_full = core_atlas_module.reorder_labels_head_to_tail(
        labels_full.astype(np.int32),
        template_anterior_coordinate(surface_path, hemi),
    ).astype(np.int32)

    group_fc_full = cf_module.reconstruct_full_surface(group_fc_act, support, vertex_count)
    prototypes = cf_module.build_cluster_prototypes(group_fc_full, labels_full, support, k=DEFAULT_K)

    return {
        "labels": labels_full,
        "support": support,
        "prototypes": prototypes,
        "group_fc_full": group_fc_full,
        "gradients_full": gradients_full,
        "gradient_preprocessing": gradient_preprocessing_settings(
            group_fc_act.shape[1], feature_scaling=options.get("feature_scaling", "col_zscore"),
            sparsify_stage=options.get("sparsify_stage", "features"), sparsity=options.get("sparsity"),
        ),
        "clustering_feature_transform": transform,
    }


def compute_pairwise_metrics(
    results_by_fold: dict[str, dict[str, Any]],
    fold_names: list[str],
) -> dict[str, Any]:
    n = len(fold_names)
    dice_matrix = np.ones((n, n), dtype=np.float64)
    ari_matrix = np.ones((n, n), dtype=np.float64)
    proto_matrix = np.ones((n, n), dtype=np.float64)

    all_dice = []
    all_ari = []
    all_proto = []

    for i in range(n):
        for j in range(i + 1, n):
            f1, f2 = fold_names[i], fold_names[j]
            r1, r2 = results_by_fold[f1], results_by_fold[f2]
            overlap = r1["support"] & r2["support"]

            ari = float(adjusted_rand_score(r1["labels"][overlap], r2["labels"][overlap]))
            ari_matrix[i, j] = ari_matrix[j, i] = ari
            all_ari.append(ari)

            dices = []
            for k in range(1, DEFAULT_K + 1):
                m1 = r1["labels"][overlap] == k
                m2 = r2["labels"][overlap] == k
                denom = np.sum(m1) + np.sum(m2)
                d = float(2.0 * np.sum(m1 & m2) / denom) if denom > 0 else 0.0
                dices.append(d)
            macro_d = float(np.mean(dices))
            dice_matrix[i, j] = dice_matrix[j, i] = macro_d
            all_dice.append(macro_d)

            corrs = []
            for k in range(DEFAULT_K):
                p1 = r1["prototypes"][k]
                p2 = r2["prototypes"][k]
                valid = np.isfinite(p1) & np.isfinite(p2)
                if np.sum(valid) >= 3:
                    c = float(np.corrcoef(p1[valid], p2[valid])[0, 1])
                else:
                    c = 0.0
                corrs.append(c)
            mean_c = float(np.mean(corrs))
            proto_matrix[i, j] = proto_matrix[j, i] = mean_c
            all_proto.append(mean_c)

    labels_stack = np.stack([results_by_fold[f]["labels"] for f in fold_names], axis=0)
    supports_stack = np.stack([results_by_fold[f]["support"] for f in fold_names], axis=0)

    n_vertices = labels_stack.shape[1]
    agreement = np.zeros(n_vertices, dtype=np.float64)
    for v in range(n_vertices):
        v_sup = supports_stack[:, v]
        if not np.any(v_sup):
            agreement[v] = np.nan
            continue
        v_labels = labels_stack[v_sup, v]
        v_nonzero = v_labels[v_labels > 0]
        if v_nonzero.size == 0:
            agreement[v] = np.nan
            continue
        mode_count = Counter(v_nonzero.tolist()).most_common(1)[0][1]
        agreement[v] = float(mode_count) / float(v_nonzero.size)

    valid_ag = agreement[np.isfinite(agreement)]
    mean_agreement = float(np.mean(valid_ag)) if valid_ag.size > 0 else 0.0

    return {
        "mean_dice": float(np.mean(all_dice)),
        "min_dice": float(np.min(all_dice)),
        "mean_ari": float(np.mean(all_ari)),
        "min_ari": float(np.min(all_ari)),
        "mean_prototype_corr": float(np.mean(all_proto)),
        "min_prototype_corr": float(np.min(all_proto)),
        "mean_vertex_agreement": mean_agreement,
        "dice_matrix": dice_matrix.tolist(),
        "ari_matrix": ari_matrix.tolist(),
        "prototype_matrix": proto_matrix.tolist(),
    }


def compose_disjoint_comparison_figure(
    *,
    grid_images: dict[tuple[str, str], Path],
    fold_names: list[str],
    smoothings: list[str],
    output_path: Path,
    fold_sizes: dict[str, int],
    metrics_summary: dict[str, dict[str, float]],
) -> None:
    loaded = {k: Image.open(v).convert("RGB") for k, v in grid_images.items()}
    cell_w = max(im.width for im in loaded.values())
    cell_h = max(im.height for im in loaded.values())

    n_cols = len(fold_names)
    n_rows = len(smoothings)

    outer_pad_x = 60
    outer_pad_y = 50
    col_gap = 40
    row_gap = 60
    header_h = 175
    col_header_h = 95
    smoothing_banner_h = 70
    footer_h = 220

    content_w = n_cols * cell_w + (n_cols - 1) * col_gap
    canvas_w = outer_pad_x * 2 + content_w
    
    total_rows_h = sum(smoothing_banner_h + col_header_h + cell_h for _ in range(n_rows))
    canvas_h = outer_pad_y * 2 + header_h + total_rows_h + (n_rows - 1) * row_gap + footer_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), (12, 14, 18))
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(46)
    subtitle_font = load_font(26)
    section_font = load_font(32)
    fold_title_font = load_font(30)
    fold_sub_font = load_font(22)
    legend_font = load_font(26)
    stats_font = load_font(22)

    # 1. Header Banner
    main_title = "HCP 7T RouteA Five Disjoint Independent Cohorts (K=3, 0% Sample Overlap)"
    sizes = sorted(fold_sizes.values())
    size_text = f"{sizes[0]}" if sizes[0] == sizes[-1] else f"{sizes[0]}-{sizes[-1]}"
    sub_title = f"Direct 3D Hippocampal Surface Parcellation Comparison Across 5 Non-Overlapping Independent Subgroups ({size_text} Subjects Each)"
    
    bbox_t = draw.textbbox((0, 0), main_title, font=title_font)
    draw.text(
        ((canvas_w - (bbox_t[2] - bbox_t[0])) // 2, outer_pad_y + 15),
        main_title,
        fill=(255, 255, 255),
        font=title_font,
    )
    bbox_st = draw.textbbox((0, 0), sub_title, font=subtitle_font)
    draw.text(
        ((canvas_w - (bbox_st[2] - bbox_st[0])) // 2, outer_pad_y + 80),
        sub_title,
        fill=(190, 205, 225),
        font=subtitle_font,
    )

    cur_y = outer_pad_y + header_h

    # 2. Render each smoothing row
    for row_idx, smoothing in enumerate(smoothings):
        sec_text = f"─── {smoothing.upper()} SURFACE SMOOTHING (5 Completely Disjoint Cohorts, 0% Overlap) ───"
        bbox_sec = draw.textbbox((0, 0), sec_text, font=section_font)
        draw.text(
            ((canvas_w - (bbox_sec[2] - bbox_sec[0])) // 2, cur_y + 10),
            sec_text,
            fill=(255, 215, 0),
            font=section_font,
        )
        cur_y += smoothing_banner_h

        for col_idx, fold_name in enumerate(fold_names):
            col_x = outer_pad_x + col_idx * (cell_w + col_gap)
            fold_num = int(fold_name.split("-")[-1])
            n_sub = fold_sizes[fold_name]

            f_title = f"Cohort {fold_num}"
            f_sub = f"Independent n = {n_sub} (0% overlap)"

            bbox_f = draw.textbbox((0, 0), f_title, font=fold_title_font)
            draw.text(
                (col_x + (cell_w - (bbox_f[2] - bbox_f[0])) // 2, cur_y + 8),
                f_title,
                fill=(240, 240, 240),
                font=fold_title_font,
            )
            bbox_fs = draw.textbbox((0, 0), f_sub, font=fold_sub_font)
            draw.text(
                (col_x + (cell_w - (bbox_fs[2] - bbox_fs[0])) // 2, cur_y + 50),
                f_sub,
                fill=(160, 180, 205),
                font=fold_sub_font,
            )

        cur_y += col_header_h

        for col_idx, fold_name in enumerate(fold_names):
            col_x = outer_pad_x + col_idx * (cell_w + col_gap)
            cell_im = loaded[(fold_name, smoothing)]
            offset_x = col_x + (cell_w - cell_im.width) // 2
            offset_y = cur_y + (cell_h - cell_im.height) // 2
            canvas.paste(cell_im, (offset_x, offset_y))

        cur_y += cell_h
        if row_idx + 1 < n_rows:
            cur_y += row_gap

    # 3. Bottom Footer
    cur_y += 30
    draw.line([(outer_pad_x, cur_y), (canvas_w - outer_pad_x, cur_y)], fill=(60, 70, 85), width=2)
    cur_y += 25

    legend_items = [(DISPLAY_LABEL_SPECS[c][0], DISPLAY_LABEL_SPECS[c][1][:3]) for c in (1, 2, 3)]
    swatch_size = 28
    item_spacing = 60

    total_legend_w = 0
    measured_items = []
    for text, color in legend_items:
        bbox = draw.textbbox((0, 0), text, font=legend_font)
        w = swatch_size + 14 + (bbox[2] - bbox[0])
        measured_items.append((text, color, w))
        total_legend_w += w
    total_legend_w += (len(legend_items) - 1) * item_spacing

    leg_start_x = (canvas_w - total_legend_w) // 2
    cur_leg_x = leg_start_x
    for text, color, w in measured_items:
        draw.rectangle(
            [(cur_leg_x, cur_y + 2), (cur_leg_x + swatch_size, cur_y + 2 + swatch_size)],
            fill=color,
            outline=(255, 255, 255),
            width=1,
        )
        draw.text((cur_leg_x + swatch_size + 14, cur_y), text, fill=(240, 240, 240), font=legend_font)
        cur_leg_x += w + item_spacing

    cur_y += 50
    m2 = metrics_summary.get("2mm", {})
    m4 = metrics_summary.get("4mm", {})
    line1 = (
        f"2 mm smoothing (0% overlap):  Pairwise Macro-Dice = {m2.get('mean_dice', 0):.3f} (min {m2.get('min_dice', 0):.3f})  |  "
        f"ARI = {m2.get('mean_ari', 0):.3f} (min {m2.get('min_ari', 0):.3f})  |  "
        f"Prototype r = {m2.get('mean_prototype_corr', 0):.3f}  |  Long-Axis Agreement = {m2.get('mean_vertex_agreement', 0)*100:.1f}%"
    )
    line2 = (
        f"4 mm smoothing (0% overlap):  Pairwise Macro-Dice = {m4.get('mean_dice', 0):.3f} (min {m4.get('min_dice', 0):.3f})  |  "
        f"ARI = {m4.get('mean_ari', 0):.3f} (min {m4.get('min_ari', 0):.3f})  |  "
        f"Prototype r = {m4.get('mean_prototype_corr', 0):.3f}  |  Long-Axis Agreement = {m4.get('mean_vertex_agreement', 0)*100:.1f}%"
    )
    bbox_l1 = draw.textbbox((0, 0), line1, font=stats_font)
    draw.text(((canvas_w - (bbox_l1[2] - bbox_l1[0])) // 2, cur_y), line1, fill=(150, 220, 180), font=stats_font)
    cur_y += 34
    bbox_l2 = draw.textbbox((0, 0), line2, font=stats_font)
    draw.text(((canvas_w - (bbox_l2[2] - bbox_l2[0])) // 2, cur_y), line2, fill=(150, 220, 180), font=stats_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", dpi=(300, 300))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit and render 5 completely disjoint fold templates (0% overlap)."
    )
    parser.add_argument(
        "--crossfit-root",
        type=Path,
        default=resolve_crossfit_template_matching_root(),
        help="Root directory of crossfit computation outputs.",
    )
    parser.add_argument(
        "--overview-root",
        type=Path,
        default=resolve_crossfit_template_matching_overview_root(),
        help="Target directory for overview outputs.",
    )
    parser.add_argument(
        "--scene",
        type=Path,
        default=DEFAULT_SCENE,
        help="Path to locked Workbench scene file.",
    )
    parser.add_argument("--feature-scaling", choices=("none", "col_zscore"), default="col_zscore")
    parser.add_argument("--sparsify-stage", choices=("features", "affinity", "none"), default="features")
    parser.add_argument("--sparsity", type=float, default=None)
    parser.add_argument("--method", choices=sorted(GRADIENT_METHODS), default="gradient-spectral")
    parser.add_argument("--rbf-scale", type=float, default=1.0)
    parser.add_argument("--cluster-seed", type=int, default=0)
    parser.add_argument("--clustering-feature-scaling", choices=("none", "unit_sd"), default="none")
    args = parser.parse_args()
    gradient_options = {
        "feature_scaling": args.feature_scaling, "sparsify_stage": args.sparsify_stage,
        "sparsity": args.sparsity,
        "clustering_feature_scaling": args.clustering_feature_scaling,
        "method": args.method, "rbf_scale": args.rbf_scale, "cluster_seed": args.cluster_seed,
    }

    crossfit_root = args.crossfit_root.resolve()
    overview_root = args.overview_root.resolve()
    scene_path = args.scene.resolve()

    disjoint_work_dir = crossfit_root / "disjoint_single_folds"
    disjoint_work_dir.mkdir(parents=True, exist_ok=True)

    output_png = overview_root / DEFAULT_FIGURE_NAME
    manifest_path = overview_root / DEFAULT_MANIFEST_NAME
    summary_tsv_path = overview_root / DEFAULT_SUMMARY_TSV_NAME
    cached_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    preprocessing = gradient_preprocessing_settings(
        400, feature_scaling=args.feature_scaling, sparsify_stage=args.sparsify_stage,
        sparsity=args.sparsity,
    )

    cf = load_module("crossfit", INDIVIDUAL_WORKFLOW_DIR / "01_01_run_hcp_crossfit_template_matching.py")
    core = cf.load_group_atlas_module()
    group_grad = cf.load_group_gradient_module()

    p_assign = crossfit_root / "folds" / "fold_assignments.tsv"
    if not p_assign.exists():
        raise FileNotFoundError(f"Missing fold assignments: {p_assign}")
    rows = list(csv.DictReader(open(p_assign), delimiter="	"))
    folds: dict[str, list[str]] = {}
    for r in rows:
        folds.setdefault(r["fold_id"], []).append(r["subject"].replace("sub-", ""))

    fold_names = sorted(folds.keys())
    fold_sizes = {f: len(folds[f]) for f in fold_names}
    print("[info] Disjoint cohort sizes (0% sample overlap):")
    for f in fold_names:
        print(f"  {f}: n = {fold_sizes[f]}")

    smoothings = list(DEFAULT_SMOOTHINGS)
    hemis = list(DEFAULT_HEMIS)

    print("[info] Fitting K=3 gradient-spectral templates for all 5 disjoint cohorts...")
    t0 = time.time()
    fitted_results: dict[tuple[str, str, str], dict[str, Any]] = {}

    for smoothing in ("4mm", "2mm"):
        for hemi in hemis:
            surf_path = DEFAULT_SURF_L if hemi == "L" else DEFAULT_SURF_R
            for fold_name in fold_names:
                f_dir = disjoint_work_dir / fold_name / smoothing / f"hemi_{hemi}"
                lbl_path = f_dir / "group_labels_k3.npy"
                sup_path = f_dir / "group_support_mask.npy"
                prt_path = f_dir / "template_profiles.npy"
                gii_path = f_dir / "group_labels_k3.label.gii"
                gradient_path = f_dir / "hippomaps_gradients_full.npy"
                metadata_key = f"{fold_name}|{smoothing}|{hemi}"

                if lbl_path.exists() and sup_path.exists() and prt_path.exists():
                    validate_gradient_preprocessing(
                        cached_manifest, settings=preprocessing, manifest_path=manifest_path,
                    )
                    if (cached_manifest.get("clustering_feature_scaling") != args.clustering_feature_scaling
                            or cached_manifest.get("method") != args.method
                            or cached_manifest.get("rbf_scale") != args.rbf_scale
                            or cached_manifest.get("cluster_seed") != args.cluster_seed):
                        raise ValueError(f"Comparison fit settings differ in {manifest_path}; explicitly regenerate")
                    if not gradient_path.is_file():
                        raise FileNotFoundError(f"Missing aligned comparison gradients: {gradient_path}; explicitly regenerate")
                    recorded_fit = cached_manifest.get("fit_metadata", {}).get(metadata_key)
                    if not isinstance(recorded_fit, dict):
                        raise ValueError(f"Missing comparison fit metadata in {manifest_path}: {metadata_key}; explicitly regenerate")

                    res = {
                        "labels": np.load(lbl_path),
                        "support": np.load(sup_path),
                        "prototypes": np.load(prt_path),
                        "gradients_full": np.load(gradient_path),
                        **recorded_fit,
                    }
                    if not gii_path.exists():
                        write_display_label_gifti(res["labels"], gii_path)
                    print(f"  [cached] Loaded {fold_name} | {smoothing} | hemi-{hemi}")
                else:
                    subs = folds[fold_name]
                    res = fit_disjoint_cohort(
                        subjects=subs,
                        smoothing=smoothing,
                        hemi=hemi,
                        core_atlas_module=core,
                        group_grad_module=group_grad,
                        cf_module=cf,
                        surface_path=surf_path,
                        gradient_options=gradient_options,
                        alignment_reference=(
                            None if smoothing == "4mm" and hemi == "L" else
                            fitted_results[(fold_name, "4mm", "L")]["gradients_full"]
                        ),
                    )
                    f_dir.mkdir(parents=True, exist_ok=True)
                    np.save(lbl_path, res["labels"])
                    np.save(sup_path, res["support"])
                    np.save(prt_path, res["prototypes"])
                    np.save(gradient_path, res["gradients_full"])
                    write_display_label_gifti(res["labels"], gii_path)
                    print(f"  [done] Fitted {fold_name} | {smoothing} | hemi-{hemi}")
                fitted_results[(fold_name, smoothing, hemi)] = res

    metrics_summary = {}
    summary_tsv_rows = []

    for smoothing in smoothings:
        combined_metrics = {}
        for hemi in hemis:
            fold_res_hemi = {f: fitted_results[(f, smoothing, hemi)] for f in fold_names}
            m = compute_pairwise_metrics(fold_res_hemi, fold_names)
            combined_metrics[hemi] = m
            summary_tsv_rows.append({
                "smoothing": smoothing,
                "hemi": hemi,
                "n_folds": DEFAULT_N_FOLDS,
                "sample_overlap": "0% (completely disjoint)",
                "mean_pairwise_macro_dice": m["mean_dice"],
                "min_pairwise_macro_dice": m["min_dice"],
                "mean_pairwise_ari": m["mean_ari"],
                "min_pairwise_ari": m["min_ari"],
                "mean_pairwise_prototype_correlation": m["mean_prototype_corr"],
                "min_pairwise_prototype_correlation": m["min_prototype_corr"],
                "mean_vertex_agreement": m["mean_vertex_agreement"],
            })
        metrics_summary[smoothing] = {
            "mean_dice": (combined_metrics["L"]["mean_dice"] + combined_metrics["R"]["mean_dice"]) / 2.0,
            "min_dice": min(combined_metrics["L"]["min_dice"], combined_metrics["R"]["min_dice"]),
            "mean_ari": (combined_metrics["L"]["mean_ari"] + combined_metrics["R"]["mean_ari"]) / 2.0,
            "min_ari": min(combined_metrics["L"]["min_ari"], combined_metrics["R"]["min_ari"]),
            "mean_prototype_corr": (combined_metrics["L"]["mean_prototype_corr"] + combined_metrics["R"]["mean_prototype_corr"]) / 2.0,
            "mean_vertex_agreement": (combined_metrics["L"]["mean_vertex_agreement"] + combined_metrics["R"]["mean_vertex_agreement"]) / 2.0,
        }

    with summary_tsv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_tsv_rows[0].keys()), delimiter="	")
        writer.writeheader()
        writer.writerows(summary_tsv_rows)
    print(f"[success] Saved quantitative metrics TSV: {summary_tsv_path}")

    print("[info] Rendering 3D Workbench hippocampal scenes for all 5 disjoint cohorts...")
    with tempfile.TemporaryDirectory(prefix="disjoint_render_") as tmp:
        scratch_dir = Path(tmp)
        grid_results = {}
        for fold_name in fold_names:
            for smoothing in smoothings:
                fold_dir = scratch_dir / fold_name / smoothing
                fold_dir.mkdir(parents=True, exist_ok=True)
                lbl_l = disjoint_work_dir / fold_name / smoothing / "hemi_L" / "group_labels_k3.label.gii"
                lbl_r = disjoint_work_dir / fold_name / smoothing / "hemi_R" / "group_labels_k3.label.gii"

                render_res = render_hippocampal_grid(
                    subject="100610",
                    outdir=fold_dir,
                    name=f"{fold_name}_{smoothing}",
                    title=f"{fold_name} | {smoothing}",
                    left_labels=lbl_l,
                    right_labels=lbl_r,
                    left_surface=DEFAULT_SURF_L,
                    right_surface=DEFAULT_SURF_R,
                )
                native_png = Path(render_res["surface_png"])
                native_im = Image.open(native_png).convert("RGBA")
                grid_im = build_grid_canvas(native_im, None, layout="1x2")
                grid_path = fold_dir / f"{fold_name}_{smoothing}_grid.png"
                grid_im.save(grid_path)
                grid_results[(fold_name, smoothing)] = grid_path
                print(f"  [done] Rendered 3D scene {fold_name} | {smoothing}")

        print("[info] Composing 5 disjoint cohorts comparative canvas...")
        compose_disjoint_comparison_figure(
            grid_images=grid_results,
            fold_names=fold_names,
            smoothings=smoothings,
            output_path=output_png,
            fold_sizes=fold_sizes,
            metrics_summary=metrics_summary,
        )

    elapsed = time.time() - t0
    output_sha256 = _file_sha256(output_png)
    print(f"[success] Saved disjoint comparison figure: {output_png} ({elapsed:.1f}s, SHA-256: {output_sha256[:12]}...)")

    manifest_payload = {
        "status": "ok",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "workflow": "hcp_routea_crossfit_k3_disjoint_folds_workbench_comparison",
        "figure_png": str(output_png),
        "figure_sha256": output_sha256,
        "n_folds": DEFAULT_N_FOLDS,
        "sample_overlap": "0%",
        "fold_sizes": fold_sizes,
        "smoothings": smoothings,
        "gradient_preprocessing": preprocessing,
        "clustering_feature_scaling": args.clustering_feature_scaling,
        "method": args.method, "rbf_scale": args.rbf_scale, "cluster_seed": args.cluster_seed,
        "gradient_alignment_anchor": {"smoothing":"4mm", "hemi":"L", "population":"each_independent_fit"},
        "fit_metadata": {
            "|".join(key): {
                "gradient_preprocessing": result["gradient_preprocessing"],
                "clustering_feature_transform": result["clustering_feature_transform"],
            } for key,result in fitted_results.items()
        },
        "metrics_summary": metrics_summary,
        "elapsed_seconds": round(elapsed, 2),
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    print(f"[success] Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Render Iowa 3T Hippocampal Template Matching Dominant Subregions & Dominance Heatmaps.

Outputs 4 standard grid figures:
- template_matching_dominant_subregions_2mm_36p.png
- template_matching_dominant_subregions_4mm_36p.png

Each figure displays all 4 Iowa cohorts in stacked rows:
1. Healthy Controls (HC, N=67)
2. Depression Patients (Patient, N=26, sub-3003 excluded)
3. Pure MDD Subgroup (Pure MDD, N=7)
4. Comorbid MDD Subgroup (Comorbid MDD, N=19)

Each cohort row contains:
- Left metadata card
- Dominant Subregion Atlas (Anterior/VAN, Middle/SMN, Posterior/DMN)
- Dominance Heatmap (% Agreement among subjects) with 'magma' colormap [33.3%, 80%]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Repository paths
REPO_ROOT = Path(__file__).resolve().parents[3]
CORE_DIR = REPO_ROOT / "scripts" / "core_workflow"
VIS_LIB_DIR = REPO_ROOT / "scripts" / "visualize" / "lib"
for p in (CORE_DIR, VIS_LIB_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lib.surface_utils import render_hippocampal_grid, make_network_label_gifti

from lib.workbench_scene_batch import (
    save_shape_gii, build_scene_root, capture_scene_root as _capture_scene_root,
)

# Canonical Paths
DEFAULT_36P_ROOT = REPO_ROOT / "outputs_iowa_jj_tms" / "work" / "hcp_fixed_template_matching" / "subjects"
DEFAULT_SCENE = REPO_ROOT / "config" / "wb_locked_native_view_lateral_medial.scene"
DEFAULT_SPEC = (
    REPO_ROOT
    / "outputs_iowa_jj_tms"
    / "inputs"
    / "hippunfold"
    / "dense_corobl_batch"
    / "sub-1001"
    / "hippunfold"
    / "sub-1001"
    / "surf"
    / "sub-1001_den-512_label-hipp_surfaces.spec"
)
DEFAULT_SURF_L = (
    REPO_ROOT
    / "outputs_iowa_jj_tms"
    / "inputs"
    / "kong2022"
    / "templates"
    / "hipp_group_average"
    / "group_average_surfaces"
    / "group-average_hemi-L_space-corobl_den-512_label-hipp_midthickness.surf.gii"
)
DEFAULT_SURF_R = (
    REPO_ROOT
    / "outputs_iowa_jj_tms"
    / "inputs"
    / "kong2022"
    / "templates"
    / "hipp_group_average"
    / "group_average_surfaces"
    / "group-average_hemi-R_space-corobl_den-512_label-hipp_midthickness.surf.gii"
)

# Colors and Semantics
SUBREGION_COLORS = {
    "Anterior/VAN": (220, 92, 255, 255),    # #dc5cff
    "Middle/SMN": (99, 150, 194, 255),      # #6396c2
    "Posterior/DMN": (216, 92, 107, 255),   # #d85c6b
}
SUBREGION_HEX = {
    1: "#dc5cff",
    2: "#6396c2",
    3: "#d85c6b",
}
SUBREGION_NAMES = {
    1: "Anterior / VAN",
    2: "Middle / SMN",
    3: "Posterior / DMN",
}
NETWORK_NAMES = ["Anterior/VAN", "Middle/SMN", "Posterior/DMN"]


def load_manifest_roster(fname: str) -> list[str]:
    lines = (REPO_ROOT / "manifests" / fname).read_text(encoding="utf-8").splitlines()
    return [f"sub-{l.strip()}" for l in lines if l.strip() and not l.startswith("#")]


COHORTS_DEF = [
    {
        "key": "iowa_hc",
        "name": "Healthy Controls",
        "badge": "HC",
        "badge_color": "#0b3c5d",
        "bg_color": "#f8fafc",
        "desc": "Baseline Healthy Volunteers",
        "roster_file": "jj_tms_hc_subjects.txt",
        "subjects": load_manifest_roster("jj_tms_hc_subjects.txt"),
    },
    {
        "key": "iowa_pat",
        "name": "Depression Patients",
        "badge": "Patient",
        "badge_color": "#831843",
        "bg_color": "#fff1f2",
        "desc": "Clinical MDD (sub-3003 excluded)",
        "roster_file": "jj_tms_patient_subjects.txt",
        "subjects": load_manifest_roster("jj_tms_patient_subjects.txt"),
    },
    {
        "key": "iowa_pure_mdd",
        "name": "Pure MDD Subgroup",
        "badge": "Pure MDD",
        "badge_color": "#854d0e",
        "bg_color": "#fefce8",
        "desc": "MDD without Clinical Anxiety",
        "roster_file": "jj_tms_pure_mdd_subjects.txt",
        "subjects": load_manifest_roster("jj_tms_pure_mdd_subjects.txt"),
    },
    {
        "key": "iowa_comorbid_mdd",
        "name": "Comorbid MDD Subgroup",
        "badge": "Comorbid MDD",
        "badge_color": "#14532d",
        "bg_color": "#f0fdf4",
        "desc": "MDD with MINI-confirmed GAD",
        "roster_file": "jj_tms_comorbid_mdd_subjects.txt",
        "subjects": load_manifest_roster("jj_tms_comorbid_mdd_subjects.txt"),
    },
]


def build_tight_4view_composite(native_im: Image.Image) -> Image.Image:
    """Build a tight 4-surface composite ordered: [Left Front, Right Front, Left Back, Right Back]."""
    arr = np.array(native_im)

    def trim_b(img: Image.Image) -> Image.Image:
        a = np.array(img)
        corners = np.array([a[0, 0, :3], a[0, -1, :3], a[-1, 0, :3], a[-1, -1, :3]])
        is_white_bg = float(np.mean(corners)) > 127.0
        if is_white_bg:
            fg = np.any(a[..., :3] < 245, axis=2)
        else:
            fg = (a[..., :3].sum(axis=2) > 15)
        if not np.any(fg):
            return img
        rmin, rmax = np.where(np.any(fg, axis=1))[0][[0, -1]]
        cmin, cmax = np.where(np.any(fg, axis=0))[0][[0, -1]]
        return img.crop((cmin, rmin, cmax + 1, rmax + 1))

    # Top: Front (Lateral), Bottom: Back (Medial)
    v_l_front = trim_b(Image.fromarray(arr[: arr.shape[0] // 2, : arr.shape[1] // 2]))
    v_r_front = trim_b(Image.fromarray(arr[: arr.shape[0] // 2, arr.shape[1] // 2 :]))
    v_l_back  = trim_b(Image.fromarray(arr[arr.shape[0] // 2 :, : arr.shape[1] // 2]))
    v_r_back  = trim_b(Image.fromarray(arr[arr.shape[0] // 2 :, arr.shape[1] // 2 :]))

    target_h = max(v_l_front.height, v_r_front.height, v_l_back.height, v_r_back.height)

    # 4 views ordered: [Left Front, Right Front, Left Back, Right Back]
    views = [v_l_front, v_r_front, v_l_back, v_r_back]
    scaled_views = []
    for v in views:
        sc = target_h / float(v.height)
        sw = max(1, int(round(v.width * sc)))
        scaled_views.append(v.resize((sw, target_h), Image.Resampling.LANCZOS))

    gap_inner = 3
    gap_pair = 10
    pad = 1

    total_w = pad * 2 + scaled_views[0].width + gap_inner + scaled_views[1].width + gap_pair + scaled_views[2].width + gap_inner + scaled_views[3].width
    total_h = pad * 2 + target_h

    canvas = Image.new("RGBA", (total_w, total_h), (255, 255, 255, 255))
    x = pad
    canvas.paste(scaled_views[0], (x, pad), scaled_views[0] if scaled_views[0].mode == "RGBA" else None)
    x += scaled_views[0].width + gap_inner
    canvas.paste(scaled_views[1], (x, pad), scaled_views[1] if scaled_views[1].mode == "RGBA" else None)
    x += scaled_views[1].width + gap_pair
    canvas.paste(scaled_views[2], (x, pad), scaled_views[2] if scaled_views[2].mode == "RGBA" else None)
    x += scaled_views[2].width + gap_inner
    canvas.paste(scaled_views[3], (x, pad), scaled_views[3] if scaled_views[3].mode == "RGBA" else None)
    return canvas


@dataclass
class CohortRowResult:
    cohort_key: str
    cohort_name: str
    cohort_badge: str
    badge_color: str
    bg_color: str
    desc: str
    n_subjects: int
    dominant_atlas_png: Path
    dominance_heatmap_png: Path
    dominant_counts: dict[int, int]
    dominant_percentages: dict[int, float]
    mean_dominance_pct: float
    median_dominance_pct: float
    min_dominance_pct: float
    max_dominance_pct: float


def compute_cohort_data(
    smoothing: str,
    subjects: list[str],
    subjects_root: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Compute dominant label and dominance percentage for left and right hemispheres."""
    dominant_labels = {}
    dominance_pcts = {}

    all_hemi_labels = []
    all_hemi_pcts = []

    for hemi in ("L", "R"):
        sub_labels = []
        sub_corrs = []
        for s in subjects:
            f_lbl = subjects_root / s / "matching" / smoothing / f"hemi-{hemi}" / "labels.npy"
            f_cor = subjects_root / s / "matching" / smoothing / f"hemi-{hemi}" / "template_correlations.npy"
            if not f_lbl.is_file() or not f_cor.is_file():
                raise FileNotFoundError(f"Missing matching result for {s} {smoothing} hemi-{hemi}: {f_lbl}")
            sub_labels.append(np.load(f_lbl))
            sub_corrs.append(np.load(f_cor))

        arr_labels = np.asarray(sub_labels, dtype=np.int32)       # (N, 512)
        arr_corrs = np.asarray(sub_corrs, dtype=np.float32)       # (N, 512, 3)

        c1 = np.sum(arr_labels == 1, axis=0)
        c2 = np.sum(arr_labels == 2, axis=0)
        c3 = np.sum(arr_labels == 3, axis=0)
        counts = np.stack([c1, c2, c3], axis=-1)                  # (512, 3)

        mean_corrs = np.mean(arr_corrs, axis=0)                   # (512, 3)
        # Deterministic tie-breaking using mean template correlation
        combined_score = counts.astype(np.float64) + 1e-4 * mean_corrs.astype(np.float64)

        winner = np.argmax(combined_score, axis=-1) + 1           # 1, 2, 3
        max_counts = np.max(counts, axis=-1)
        pct = (max_counts / float(len(subjects))) * 100.0         # % agreement

        dominant_labels[hemi] = winner.astype(np.int32)
        dominance_pcts[hemi] = pct.astype(np.float32)

        all_hemi_labels.append(winner)
        all_hemi_pcts.append(pct)

    combined_labels = np.concatenate(all_hemi_labels)
    combined_pcts = np.concatenate(all_hemi_pcts)

    total_vertices = len(combined_labels)
    dominant_counts = {k: int(np.sum(combined_labels == k)) for k in (1, 2, 3)}
    dominant_percentages = {k: (dominant_counts[k] / total_vertices) * 100.0 for k in (1, 2, 3)}

    stats_summary = {
        "dominant_counts": dominant_counts,
        "dominant_percentages": dominant_percentages,
        "mean_dominance_pct": float(np.mean(combined_pcts)),
        "median_dominance_pct": float(np.median(combined_pcts)),
        "min_dominance_pct": float(np.min(combined_pcts)),
        "max_dominance_pct": float(np.max(combined_pcts)),
    }

    return dominant_labels, dominance_pcts, stats_summary


def apply_heatmap_metric_palette(
    metric_path: Path,
    palette_name: str = "magma",
    vmin: float = 33.33,
    vmax: float = 80.0,
) -> None:
    """Apply Workbench metric palette with explicit user min/max range."""
    cmd = [
        "wb_command",
        "-metric-palette",
        str(metric_path),
        "MODE_USER_SCALE",
        "-palette-name",
        str(palette_name),
        "-disp-pos",
        "true",
        "-disp-neg",
        "false",
        "-disp-zero",
        "true",
        "-interpolate",
        "true",
        "-normalization",
        "NORMALIZATION_ALL_MAP_DATA",
        "-pos-user",
        str(vmin),
        str(vmax),
    ]
    subprocess.run(cmd, check=True)


def render_cohort_panels(
    cohort_key: str,
    smoothing: str,
    variant: str,
    dominant_labels: dict[str, np.ndarray],
    dominance_pcts: dict[str, np.ndarray],
    panels_dir: Path,
    scratch_dir: Path,
    surf_l: Path = DEFAULT_SURF_L,
    surf_r: Path = DEFAULT_SURF_R,
    spec_path: Path = DEFAULT_SPEC,
    scene_path: Path = DEFAULT_SCENE,
    vmin: float = 33.33,
    vmax: float = 80.0,
    palette_name: str = "magma",
    force_rerender_heatmaps: bool = False,
) -> tuple[Path, Path]:
    """Render dominant atlas panel and dominance heatmap panel using Workbench."""
    panels_dir.mkdir(parents=True, exist_ok=True)
    atlas_cached = panels_dir / f"wb_{cohort_key}_{smoothing}_{variant}_dominant_atlas_tight4view.png"
    heatmap_cached = panels_dir / f"wb_{cohort_key}_{smoothing}_{variant}_dominance_heatmap_tight4view.png"

    # 1. Render Dominant Subregion Atlas
    if atlas_cached.is_file():
        atlas_comp_path = atlas_cached
    else:
        lbl_l_path = scratch_dir / f"lbl_{cohort_key}_{smoothing}_{variant}_hemi-L.label.gii"
        lbl_r_path = scratch_dir / f"lbl_{cohort_key}_{smoothing}_{variant}_hemi-R.label.gii"
        nib.save(
            make_network_label_gifti(
                labels=dominant_labels["L"],
                network_names=NETWORK_NAMES,
                network_colors=SUBREGION_COLORS,
            ),
            str(lbl_l_path),
        )
        nib.save(
            make_network_label_gifti(
                labels=dominant_labels["R"],
                network_names=NETWORK_NAMES,
                network_colors=SUBREGION_COLORS,
            ),
            str(lbl_r_path),
        )

        render_res = render_hippocampal_grid(
            subject="1001",
            outdir=scratch_dir / f"atlas_{cohort_key}_{smoothing}_{variant}",
            name=f"dominant_atlas_{cohort_key}_{smoothing}_{variant}",
            title=f"Dominant Atlas | {cohort_key} {smoothing} {variant}",
            left_labels=lbl_l_path,
            right_labels=lbl_r_path,
            left_surface=surf_l,
            right_surface=surf_r,
        )
        native_png = Path(render_res["surface_png"])
        atlas_comp = build_tight_4view_composite(Image.open(native_png).convert("RGBA"))
        atlas_comp.save(atlas_cached)
        atlas_comp_path = atlas_cached

    # 2. Render Dominance Heatmap
    if heatmap_cached.is_file() and not force_rerender_heatmaps:
        heatmap_comp_path = heatmap_cached
    else:
        shp_l_path = scratch_dir / f"shape_{cohort_key}_{smoothing}_{variant}_hemi-L.shape.gii"
        shp_r_path = scratch_dir / f"shape_{cohort_key}_{smoothing}_{variant}_hemi-R.shape.gii"
        save_shape_gii(dominance_pcts["L"], shp_l_path)
        save_shape_gii(dominance_pcts["R"], shp_r_path)

        apply_heatmap_metric_palette(shp_l_path, palette_name, vmin=vmin, vmax=vmax)
        apply_heatmap_metric_palette(shp_r_path, palette_name, vmin=vmin, vmax=vmax)

        metric_scene_root = build_scene_root(
            scene_path=scene_path,
            template_subject="1001",
            left_surface=surf_l,
            right_surface=surf_r,
            spec_path=spec_path,
            left_shape=shp_l_path,
            right_shape=shp_r_path,
            vmin=vmin,
            vmax=vmax,
            palette_name=palette_name,
        )

        metric_native_png = scratch_dir / f"metric_{cohort_key}_{smoothing}_{variant}_native.png"
        _capture_scene_root(
            root=metric_scene_root,
            out_png=metric_native_png,
            width=1600,
            height=1024,
            renderer="OSMesa",
        )
        heatmap_comp = build_tight_4view_composite(Image.open(metric_native_png).convert("RGBA"))
        heatmap_comp.save(heatmap_cached)
        heatmap_comp_path = heatmap_cached

    return atlas_comp_path, heatmap_comp_path


def create_colorbar_image(width: int = 240, height: int = 16, vmin: float = 33.33, vmax: float = 80.0) -> Image.Image:
    """Create horizontal colorbar image with magma colormap."""
    fig, ax = plt.subplots(figsize=(width / 100.0, height / 100.0), dpi=100)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cb = matplotlib.colorbar.ColorbarBase(ax, cmap=cm.magma, norm=norm, orientation="horizontal")
    ax.axis("off")
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return Image.fromarray(rgba)


def compose_grid_figure(
    smoothing: str,
    variant: str,
    cohort_results: list[CohortRowResult],
    output_png: Path,
) -> None:
    """Compose single grid figure presenting all 4 Iowa cohorts in stacked rows."""
    total_w = 1940
    row_h = 575
    margin_x = 35
    margin_top = 185
    margin_bottom = 35
    gap_y = 20

    total_h = margin_top + len(cohort_results) * row_h + (len(cohort_results) - 1) * gap_y + margin_bottom

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Fonts
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 30)
        subtitle_font = ImageFont.truetype("DejaVuSans.ttf", 18)
        panel_title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 17)
        view_font = ImageFont.truetype("DejaVuSans.ttf", 13)
        stat_header_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
        stat_bold_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
        legend_font = ImageFont.truetype("DejaVuSans.ttf", 15)
        legend_bold = ImageFont.truetype("DejaVuSans-Bold.ttf", 15)
        badge_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
        n_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except OSError:
        title_font = subtitle_font = panel_title_font = view_font = stat_header_font = stat_bold_font = legend_font = legend_bold = badge_font = n_font = ImageFont.load_default()

    # 1. Main Figure Title & Subtitle
    title_text = "Iowa 3T Hippocampal Functional Atlas: Dominant Subregions & Agreement Heatmaps"
    draw.text((margin_x, 22), title_text, fill=(20, 30, 45), font=title_font)

    variant_desc = "36-Parameter (36P) Confound Regression"
    sub_text = (
        f"{smoothing.upper()} Surface Smoothing  |  {variant.upper()} Confound Regression ({variant_desc})  "
        f"—  Iowa 3T Cohorts (N=93 Total Subjects)"
    )
    draw.text((margin_x, 66), sub_text, fill=(71, 85, 105), font=subtitle_font)

    # 2. Global Legend Header Box
    legend_box_y = 105
    legend_box_h = 56
    box_w = total_w - margin_x * 2
    draw.rounded_rectangle(
        [(margin_x, legend_box_y), (margin_x + box_w, legend_box_y + legend_box_h)],
        radius=8,
        fill="#f8fafc",
        outline="#e2e8f0",
        width=1,
    )

    cur_x = margin_x + 18
    draw.text((cur_x, legend_box_y + 18), "Subregions:", fill=(30, 41, 59), font=legend_bold)
    cur_x += int(draw.textlength("Subregions:", font=legend_bold)) + 16

    # Subregion Swatches: Anterior / Middle / Posterior
    legend_subregions = [
        ("Anterior / VAN", "#dc5cff"),
        ("Middle / SMN", "#6396c2"),
        ("Posterior / DMN", "#d85c6b"),
    ]
    for name, hex_color in legend_subregions:
        draw.rounded_rectangle([(cur_x, legend_box_y + 17), (cur_x + 20, legend_box_y + 37)], radius=3, fill=hex_color, outline="#475569", width=1)
        cur_x += 26
        draw.text((cur_x, legend_box_y + 18), name, fill=(30, 41, 59), font=legend_font)
        cur_x += int(draw.textlength(name, font=legend_font)) + 20

    # Separator
    draw.line([(cur_x, legend_box_y + 10), (cur_x, legend_box_y + legend_box_h - 10)], fill="#cbd5e1", width=1)
    cur_x += 20

    # Dominance Agreement Colorbar (NO OVERLAP)
    cbar_title = "Dominance Agreement:"
    draw.text((cur_x, legend_box_y + 18), cbar_title, fill=(30, 41, 59), font=legend_bold)
    cur_x += int(draw.textlength(cbar_title, font=legend_bold)) + 20

    cbar_w, cbar_h = 200, 16
    cbar_img = create_colorbar_image(width=cbar_w, height=cbar_h, vmin=33.33, vmax=80.0)
    canvas.paste(cbar_img, (cur_x, legend_box_y + 13))

    # Ticks under colorbar (precisely placed)
    tick_y = legend_box_y + 33
    t1 = "33.3%"
    t2 = "56.7%"
    t3 = "≥80%"
    draw.text((cur_x - 6, tick_y), t1, fill=(71, 85, 105), font=view_font)
    t2_w = draw.textlength(t2, font=view_font)
    draw.text((cur_x + cbar_w // 2 - int(t2_w // 2), tick_y), t2, fill=(71, 85, 105), font=view_font)
    t3_w = draw.textlength(t3, font=view_font)
    draw.text((cur_x + cbar_w - int(t3_w) + 4, tick_y), t3, fill=(71, 85, 105), font=view_font)
    cur_x += cbar_w + 30

    # Separator
    draw.line([(cur_x, legend_box_y + 10), (cur_x, legend_box_y + legend_box_h - 10)], fill="#cbd5e1", width=1)
    cur_x += 20

    # Anatomical orientation hint
    draw.text(
        (cur_x, legend_box_y + 18),
        "Views (each panel): Left Lateral  |  Right Lateral  |  Left Medial  |  Right Medial",
        fill=(100, 116, 139),
        font=legend_font,
    )

    # 3. Render Stacked Cohort Rows
    meta_w = 210
    panel_gap = 25
    available_w = total_w - 2 * margin_x - meta_w - 2 * panel_gap
    target_pw = available_w // 2

    for r_idx, cr in enumerate(cohort_results):
        row_y = margin_top + r_idx * (row_h + gap_y)

        # Card container background
        draw.rounded_rectangle(
            [(margin_x, row_y), (total_w - margin_x, row_y + row_h)],
            radius=10,
            fill="#ffffff",
            outline="#cbd5e1",
            width=2,
        )

        # Left Metadata Card Column
        draw.rounded_rectangle(
            [(margin_x + 2, row_y + 2), (margin_x + meta_w, row_y + row_h - 2)],
            radius=9,
            fill=cr.bg_color,
            outline="#e2e8f0",
            width=1,
        )

        # Left card content: clean, centered, NO redundant descriptions
        meta_cx = margin_x + meta_w // 2
        badge_text = cr.cohort_badge
        n_text = f"N = {cr.n_subjects}"
        badge_w = int(draw.textlength(badge_text, font=badge_font)) + 36
        badge_h = 44
        badge_y = row_y + row_h // 2 - 32

        draw.rounded_rectangle(
            [(meta_cx - badge_w // 2, badge_y), (meta_cx + badge_w // 2, badge_y + badge_h)],
            radius=8,
            fill=cr.badge_color,
        )
        draw.text(
            (meta_cx - int(draw.textlength(badge_text, font=badge_font) // 2), badge_y + 10),
            badge_text,
            fill="#ffffff",
            font=badge_font,
        )
        draw.text(
            (meta_cx - int(draw.textlength(n_text, font=n_font) // 2), badge_y + 56),
            n_text,
            fill="#1e293b",
            font=n_font,
        )

        # Vertical divider between meta card and panels
        draw.line([(margin_x + meta_w, row_y + 10), (margin_x + meta_w, row_y + row_h - 10)], fill="#e2e8f0", width=1)

        # Panel 1: Dominant Subregion Atlas
        p1_x = margin_x + meta_w + panel_gap
        draw.text((p1_x + 4, row_y + 14), "Dominant Subregion Atlas", fill=(30, 41, 59), font=panel_title_font)

        im_atlas = Image.open(cr.dominant_atlas_png).convert("RGBA")
        scale_ratio = target_pw / float(im_atlas.width)
        target_ph = int(round(im_atlas.height * scale_ratio))
        im_atlas_scaled = im_atlas.resize((target_pw, target_ph), Image.Resampling.LANCZOS)

        surf_y = row_y + 42
        canvas.paste(im_atlas_scaled, (p1_x, surf_y), im_atlas_scaled)

        # Views hint under Atlas
        view_labels = ["LH Lateral", "RH Lateral", "LH Medial", "RH Medial"]
        v_step = target_pw / 4.0
        views_y = surf_y + target_ph + 4
        for v_idx, v_label in enumerate(view_labels):
            v_x = int(p1_x + v_idx * v_step + v_step * 0.22)
            draw.text((v_x, views_y), v_label, fill=(148, 163, 184), font=view_font)

        # Stats box for Atlas (Enlarged)
        stats_box_y = views_y + 20
        stats_box_h = 60
        draw.rounded_rectangle(
            [(p1_x, stats_box_y), (p1_x + target_pw, stats_box_y + stats_box_h)],
            radius=8,
            fill="#f8fafc",
            outline="#cbd5e1",
            width=1,
        )
        cnt1, pct1 = cr.dominant_counts[1], cr.dominant_percentages[1]
        cnt2, pct2 = cr.dominant_counts[2], cr.dominant_percentages[2]
        cnt3, pct3 = cr.dominant_counts[3], cr.dominant_percentages[3]

        s1_text = (
            f"Anterior/VAN: {pct1:.1f}% ({cnt1} vtx)   |   "
            f"Middle/SMN: {pct2:.1f}% ({cnt2} vtx)   |   "
            f"Posterior/DMN: {pct3:.1f}% ({cnt3} vtx)"
        )
        draw.text((p1_x + 16, stats_box_y + 10), "Subregion Composition across 1024 Bilateral Vertices:", fill="#475569", font=stat_header_font)
        draw.text((p1_x + 16, stats_box_y + 33), s1_text, fill="#0f172a", font=stat_bold_font)

        # Panel 2: Percentage Dominance Heatmap
        p2_x = p1_x + target_pw + panel_gap
        draw.text((p2_x + 4, row_y + 14), "Percentage Dominance Heatmap", fill=(30, 41, 59), font=panel_title_font)

        im_heatmap = Image.open(cr.dominance_heatmap_png).convert("RGBA")
        im_heatmap_scaled = im_heatmap.resize((target_pw, target_ph), Image.Resampling.LANCZOS)
        canvas.paste(im_heatmap_scaled, (p2_x, surf_y), im_heatmap_scaled)

        # Views hint under Heatmap
        for v_idx, v_label in enumerate(view_labels):
            v_x = int(p2_x + v_idx * v_step + v_step * 0.22)
            draw.text((v_x, views_y), v_label, fill=(148, 163, 184), font=view_font)

        # Stats box for Heatmap (Enlarged)
        draw.rounded_rectangle(
            [(p2_x, stats_box_y), (p2_x + target_pw, stats_box_y + stats_box_h)],
            radius=8,
            fill="#f8fafc",
            outline="#cbd5e1",
            width=1,
        )
        s2_title = f"Subject Dominance Agreement Statistics ({cr.cohort_badge}, N={cr.n_subjects}):"
        s2_text = (
            f"Mean: {cr.mean_dominance_pct:.1f}%   |   "
            f"Median: {cr.median_dominance_pct:.1f}%   |   "
            f"Range: [{cr.min_dominance_pct:.1f}%, {cr.max_dominance_pct:.1f}%]"
        )
        draw.text((p2_x + 16, stats_box_y + 10), s2_title, fill="#475569", font=stat_header_font)
        draw.text((p2_x + 16, stats_box_y + 33), s2_text, fill="#0f172a", font=stat_bold_font)

    # Save output
    output_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_png, "PNG", dpi=(300, 300))
    print(f"Successfully saved grid figure: {output_png} ({total_w}x{total_h})")


def write_comprehensive_csv(all_results: dict[str, list[CohortRowResult]], csv_path: Path) -> None:
    """Write comprehensive metrics across both 36p grid combinations and 4 cohorts."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for grid_tag, cohort_list in all_results.items():
        smoothing, variant = grid_tag.split("_")
        for cr in cohort_list:
            rows.append({
                "grid": grid_tag,
                "smoothing": smoothing,
                "variant": variant,
                "cohort_key": cr.cohort_key,
                "cohort_name": cr.cohort_name,
                "n_subjects": cr.n_subjects,
                "van_ant_vtx_count": cr.dominant_counts[1],
                "van_ant_vtx_pct": f"{cr.dominant_percentages[1]:.2f}",
                "smn_mid_vtx_count": cr.dominant_counts[2],
                "smn_mid_vtx_pct": f"{cr.dominant_percentages[2]:.2f}",
                "dmn_post_vtx_count": cr.dominant_counts[3],
                "dmn_post_vtx_pct": f"{cr.dominant_percentages[3]:.2f}",
                "mean_dominance_agreement_pct": f"{cr.mean_dominance_pct:.2f}",
                "median_dominance_agreement_pct": f"{cr.median_dominance_pct:.2f}",
                "min_dominance_agreement_pct": f"{cr.min_dominance_pct:.2f}",
                "max_dominance_agreement_pct": f"{cr.max_dominance_pct:.2f}",
            })

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved comprehensive stats CSV: {csv_path}")


def write_manifest_json(
    all_results: dict[str, list[CohortRowResult]],
    rendered_figures: dict[str, Path],
    manifest_path: Path,
) -> None:
    """Write run manifest and provenance metadata."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    def file_sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "script": str(Path(__file__).resolve()),
        "cohorts": [
            {
                "key": c["key"],
                "name": c["name"],
                "badge": c["badge"],
                "n_subjects": len(c["subjects"]),
                "roster_file": c["roster_file"],
            }
            for c in COHORTS_DEF
        ],
        "grid_combinations": ["2mm_36p", "4mm_36p"],
        "master_figures": {
            grid: {
                "path": str(fig_path.resolve()),
                "sha256": file_sha256(fig_path),
                "size_bytes": fig_path.stat().st_size,
            }
            for grid, fig_path in rendered_figures.items()
        },
        "condition_metrics": {
            grid: [
                {
                    "cohort_key": cr.cohort_key,
                    "cohort_name": cr.cohort_name,
                    "n_subjects": cr.n_subjects,
                    "dominant_counts": cr.dominant_counts,
                    "dominant_percentages": cr.dominant_percentages,
                    "mean_dominance_pct": cr.mean_dominance_pct,
                    "median_dominance_pct": cr.median_dominance_pct,
                    "min_dominance_pct": cr.min_dominance_pct,
                    "max_dominance_pct": cr.max_dominance_pct,
                    "atlas_png": str(cr.dominant_atlas_png),
                    "heatmap_png": str(cr.dominance_heatmap_png),
                }
                for cr in cohort_list
            ]
            for grid, cohort_list in all_results.items()
        },
    }

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved run manifest: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render Iowa 3T Hippocampal Template Matching Dominant Subregions & Dominance Heatmaps (36p Grids)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "scripts" / "visualize" / "template_matching_dominant_subregions",
        help="Directory to save output figures and stats.",
    )
    parser.add_argument(
        "--grids",
        nargs="+",
        default=["2mm_36p", "4mm_36p"],
        choices=["2mm_36p", "4mm_36p", "all"],
        help="Grid combinations to render.",
    )
    parser.add_argument(
        "--clean-cache",
        action="store_true",
        help="Force re-rendering of all intermediate panel composites.",
    )
    parser.add_argument(
        "--force-heatmaps",
        action="store_true",
        help="Force re-rendering of dominance heatmap panels with updated contrast settings.",
    )
    args = parser.parse_args()

    selected_grids = (
        ["2mm_36p", "4mm_36p"]
        if "all" in args.grids
        else args.grids
    )

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    panels_dir = out_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    force_heatmaps = args.clean_cache or args.force_heatmaps

    grid_roots = {
        "36p": DEFAULT_36P_ROOT,
    }

    all_results: dict[str, list[CohortRowResult]] = {}
    rendered_figures: dict[str, Path] = {}

    with tempfile.TemporaryDirectory(prefix="hipp_dominant_subregions_") as tmp:
        scratch_dir = Path(tmp)

        for grid_tag in selected_grids:
            smoothing, variant = grid_tag.split("_")
            sub_root = grid_roots[variant]
            print(f"\n==================================================")
            print(f"Processing Grid Combination: {grid_tag} ({smoothing}, {variant})")
            print(f"==================================================")

            cohort_results = []
            for c in COHORTS_DEF:
                ckey = c["key"]
                print(f"  Rendering cohort {ckey} ({c['name']}, N={len(c['subjects'])})...")

                dom_labels, dom_pcts, stats = compute_cohort_data(
                    smoothing=smoothing,
                    subjects=c["subjects"],
                    subjects_root=sub_root,
                )

                atlas_png, heatmap_png = render_cohort_panels(
                    cohort_key=ckey,
                    smoothing=smoothing,
                    variant=variant,
                    dominant_labels=dom_labels,
                    dominance_pcts=dom_pcts,
                    panels_dir=panels_dir,
                    scratch_dir=scratch_dir,
                    force_rerender_heatmaps=force_heatmaps,
                )

                cohort_results.append(
                    CohortRowResult(
                        cohort_key=ckey,
                        cohort_name=c["name"],
                        cohort_badge=c["badge"],
                        badge_color=c["badge_color"],
                        bg_color=c["bg_color"],
                        desc=c["desc"],
                        n_subjects=len(c["subjects"]),
                        dominant_atlas_png=atlas_png,
                        dominance_heatmap_png=heatmap_png,
                        dominant_counts=stats["dominant_counts"],
                        dominant_percentages=stats["dominant_percentages"],
                        mean_dominance_pct=stats["mean_dominance_pct"],
                        median_dominance_pct=stats["median_dominance_pct"],
                        min_dominance_pct=stats["min_dominance_pct"],
                        max_dominance_pct=stats["max_dominance_pct"],
                    )
                )

            all_results[grid_tag] = cohort_results

            # Compose Master Grid Figure
            grid_png = out_dir / f"template_matching_dominant_subregions_{grid_tag}.png"
            print(f"  Composing figure for {grid_tag} -> {grid_png.name}...")
            compose_grid_figure(
                smoothing=smoothing,
                variant=variant,
                cohort_results=cohort_results,
                output_png=grid_png,
            )
            rendered_figures[grid_tag] = grid_png

        # Write Summary CSV
        csv_path = out_dir / "template_matching_dominant_subregions_stats.csv"
        write_comprehensive_csv(all_results=all_results, csv_path=csv_path)

        # Write Run Manifest
        manifest_path = out_dir / "template_matching_dominant_subregions_manifest.json"
        write_manifest_json(
            all_results=all_results,
            rendered_figures=rendered_figures,
            manifest_path=manifest_path,
        )

    print("\nSelected 36p grid figures and statistics completed successfully!")


if __name__ == "__main__":
    main()

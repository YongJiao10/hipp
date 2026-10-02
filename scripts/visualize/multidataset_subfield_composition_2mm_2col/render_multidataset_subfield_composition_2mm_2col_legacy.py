#!/usr/bin/env python3
"""
render_multidataset_subfield_composition_2mm_2col.py

2-Column Multi-Dataset Hippocampal Subfield Composition Panorama (10 Cohorts, N=528).
Generates publication-quality figures and statistical summaries across THREE distinct functional atlas versions:
  1. Version 1 (v1_fixed_hcp151_atlas):
     All cohorts evaluated on the canonical HCP 151 Group Atlas functional boundaries.
  2. Version 2 (v2_cohort_specific_atlas):
     Each cohort evaluated on its own intrinsic data-driven Route A Group Atlas.
  3. Version 3 (v3_individual_template_matching):
     Each subject evaluated on individual-level functional boundaries (HCP 7T 5-Fold Cross-Fit & Stanford/Iowa HCP151 Fixed Template Matching).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import shutil
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import nibabel as nib
import numpy as np
from PIL import Image

REPO_ROOT = Path("/home/yojiao/hpchome/hipp")
OUTPUT_DIR = REPO_ROOT / "scripts/visualize/multidataset_subfield_composition_2mm_2col"
STATS_DIR = OUTPUT_DIR / "stats"
STATS_DIR.mkdir(parents=True, exist_ok=True)
PANELS_DIR = OUTPUT_DIR / "panels"
PANELS_DIR.mkdir(parents=True, exist_ok=True)

CORE_DIR = REPO_ROOT / "scripts/core_workflow"
VIS_LIB_DIR = REPO_ROOT / "scripts/visualize/lib"
for p in (CORE_DIR, VIS_LIB_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lib.surface_utils import render_locked_grid
from surface_panel_cache import panel_inputs, cache_matches, publish_cache
from multidataset_cohorts import (
    COMPARISON_METHODS,
    COMPARISON_VERSION,
    GridSpec,
    GRID_VARIANTS,
    format_comparison_column_title,
    get_comparison_visual_columns,
    get_multidataset_cohorts,
)


# MultiHist7 Subfield Definitions on midthickness surface
SUBFIELD_ORDER = (1, 2, 3, 4, 5)
SUBFIELD_NAMES = {
    1: "Sub",
    2: "CA1",
    3: "CA2",
    4: "CA3",
    5: "CA4",
}
SUBFIELD_FULL_NAMES = {
    1: "Subiculum",
    2: "CA1",
    3: "CA2",
    4: "CA3",
    5: "CA4",
}
SUBFIELD_COLORS = {
    1: "#292BFC",  # Sub: Royal Blue
    2: "#66D1D8",  # CA1: Sky / Cyan
    3: "#00B33C",  # CA2: Green
    4: "#FFB600",  # CA3: Amber / Gold
    5: "#FF2B17",  # CA4: Red
}
CLUSTER_NAMES = {
    1: "Ant",
    2: "Body",
    3: "Post",
}
CLUSTER_FULL_NAMES = {
    1: "Anterior (Head)",
    2: "Middle (Body)",
    3: "Posterior (Tail)",
}

PANORAMA_VERSIONS = {
    "v1_fixed_hcp151_atlas": {
        "title_label": "VERSION 1: UNIFIED FIXED HCP151 GROUP ATLAS",
        "file_tag": "v1_fixed_hcp151_atlas",
    },
    "v2_cohort_specific_atlas": {
        "title_label": "VERSION 2: COHORT-SPECIFIC INTRINSIC GROUP ATLAS",
        "file_tag": "v2_cohort_specific_atlas",
    },
    "v3_individual_template_matching": {
        "title_label": "VERSION 3: INDIVIDUAL-LEVEL TEMPLATE MATCHING ATLAS",
        "file_tag": "v3_individual_template_matching",
    },
}


def load_label_gifti(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    return np.asarray(img.darrays[0].data).astype(np.int32, copy=False).reshape(-1)


def vertex_area_weights(surface_path: Path) -> np.ndarray:
    img = nib.load(str(surface_path))
    coords = np.asarray(img.agg_data("pointset"), dtype=np.float64)
    faces = np.asarray(img.agg_data("triangle"), dtype=np.int64)
    tri = coords[faces]
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    weights = np.zeros(coords.shape[0], dtype=np.float64)
    for corner in range(3):
        np.add.at(weights, faces[:, corner], areas / 3.0)
    return weights


_LABEL_CACHE: dict[Path, np.ndarray] = {}
_WEIGHT_CACHE: dict[Path, np.ndarray] = {}


def get_cached_label(p: Path) -> np.ndarray:
    if p not in _LABEL_CACHE:
        _LABEL_CACHE[p] = load_label_gifti(p)
    return _LABEL_CACHE[p]


def get_cached_weight(p: Path) -> np.ndarray:
    if p not in _WEIGHT_CACHE:
        _WEIGHT_CACHE[p] = vertex_area_weights(p)
    return _WEIGHT_CACHE[p]


def compute_cohort_subfield_composition(
    cinfo: dict,
    atlas_mode: str,
    smoothing: str = "2mm",
    is_7net: bool = False,
) -> dict[str, list[dict]]:
    """Computes area-weighted MultiHist7 subfield compositions under specified atlas mode."""
    group_root = Path(cinfo["group_root"])
    hippunfold_root = Path(cinfo["hippunfold_root"])
    results = {}
    legacy_7net_namespace = bool(cinfo.get("legacy_7net_namespace", False))
    atlas_folder = "kong2022_7net" if legacy_7net_namespace else "kong2022"
    hcp_grp_root = REPO_ROOT / "outputs_hcp/work/group_atlas/kong2022_7net_full_k3_fisherz"

    for hemi in ["L", "R"]:
        grad_sub = Path(cinfo.get("grad_subpath", group_root / f"gradient-spectral/{atlas_folder}/{smoothing}/cov_0p80/g1g2g3"))
        hdir = grad_sub / f"hemi_{hemi}"
        manifest_path = hdir / "group_atlas_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_subjects = manifest.get("cluster_roi_profile_subjects_used")
        if not raw_subjects:
            raw_subjects = list(manifest.get("actual_consumed_input_paths", {}).keys())
        subjects = [str(s).removeprefix("sub-") for s in raw_subjects]

        if atlas_mode == "v1_fixed_hcp151_atlas":
            hdir_hcp = hcp_grp_root / f"gradient-spectral/{atlas_folder}/{smoothing}/cov_0p80/g1g2g3/hemi_{hemi}"
            labels = np.asarray(np.load(hdir_hcp / "group_labels_final.npy"), dtype=np.int32).reshape(-1)
            support = np.asarray(np.load(hdir_hcp / "group_support_mask.npy"), dtype=bool).reshape(-1)
        elif atlas_mode == "v2_cohort_specific_atlas":
            labels = np.asarray(np.load(hdir / "group_labels_final.npy"), dtype=np.int32).reshape(-1)
            support = np.asarray(np.load(hdir / "group_support_mask.npy"), dtype=bool).reshape(-1)
        else:
            labels = None
            support = None

        # Subject-level collector: {cluster_id: {subfield_id: [proportions]}}
        clust_props = defaultdict(lambda: defaultdict(list))
        subject_count = 0

        for s in subjects:
            lp = (
                hippunfold_root
                / f"sub-{s}"
                / "hippunfold"
                / f"sub-{s}"
                / "metric"
                / f"sub-{s}_hemi-{hemi}_den-512_label-hipp_atlas-multihist7_subfields.label.gii"
            )
            sp = (
                hippunfold_root
                / f"sub-{s}"
                / "hippunfold"
                / f"sub-{s}"
                / "surf"
                / f"sub-{s}_hemi-{hemi}_space-corobl_den-512_label-hipp_midthickness.surf.gii"
            )
            if not (lp.exists() and sp.exists()):
                continue

            if atlas_mode == "v3_individual_template_matching":
                if cinfo["id"] == "hcp_7t":
                    indiv_label_path = REPO_ROOT / f"outputs_hcp/work/crossfit_template_matching/subjects/sub-{s}/{smoothing}/hemi_{hemi}/individual_labels_k3.npy"
                elif "iowa" in cinfo["id"]:
                    indiv_label_path = REPO_ROOT / f"outputs_iowa_jj_tms/work/hcp151_fixed_template_matching/subjects/sub-{s}/matching/{smoothing}/hemi-{hemi}/labels.npy"
                else:
                    indiv_label_path = REPO_ROOT / f"outputs_stanford/work/hcp151_fixed_template_matching/subjects/sub-{s}/matching/{smoothing}/hemi-{hemi}/labels.npy"
                if not indiv_label_path.exists():
                    continue
                sub_labels = np.asarray(np.load(str(indiv_label_path)), dtype=np.int32).reshape(-1)
                sf = np.asarray(get_cached_label(lp), dtype=np.int32).reshape(-1)
                w = np.asarray(get_cached_weight(sp), dtype=np.float64).reshape(-1)
                valid_anatomic = (sub_labels > 0) & np.isin(sf, np.asarray(SUBFIELD_ORDER, dtype=np.int32))
            else:
                sub_labels = labels
                sf = np.asarray(get_cached_label(lp), dtype=np.int32).reshape(-1)
                w = np.asarray(get_cached_weight(sp), dtype=np.float64).reshape(-1)
                valid_anatomic = support & (sub_labels > 0) & np.isin(sf, np.asarray(SUBFIELD_ORDER, dtype=np.int32))

            if not np.any(valid_anatomic):
                continue

            subject_count += 1
            for cid in [1, 2, 3]:
                c_mask = valid_anatomic & (sub_labels == cid)
                total_w = np.sum(w[c_mask])
                if total_w > 0:
                    for sid in SUBFIELD_ORDER:
                        sid_mask = c_mask & (sf == sid)
                        clust_props[cid][sid].append(np.sum(w[sid_mask]) / total_w)
                else:
                    for sid in SUBFIELD_ORDER:
                        clust_props[cid][sid].append(0.0)

        # Aggregate across subjects for this hemisphere
        hemi_entries = []
        for cid in [1, 2, 3]:
            comp_mean = {}
            comp_sem = {}
            for sid in SUBFIELD_ORDER:
                arr = np.asarray(clust_props[cid][sid], dtype=np.float64)
                sname = SUBFIELD_NAMES[sid]
                if len(arr) > 0:
                    comp_mean[sname] = float(np.mean(arr) * 100.0)
                    comp_sem[sname] = float((np.std(arr, ddof=1) / math.sqrt(len(arr))) * 100.0) if len(arr) > 1 else 0.0
                else:
                    comp_mean[sname] = 0.0
                    comp_sem[sname] = 0.0

            hemi_entries.append({
                "cluster_id": cid,
                "subregion_name": CLUSTER_NAMES[cid],
                "subregion_full_name": CLUSTER_FULL_NAMES[cid],
                "composition": comp_mean,
                "sem": comp_sem,
                "n_subjects": subject_count,
            })
        results[hemi] = hemi_entries

    return results


from lib.workbench_grid import build_grid_canvas


def render_cohort_surface_img(
    cinfo: dict,
    render_tmp_dir: Path,
    smoothing: str = "2mm",
    variant: str = "36p",
    panels_dir: Path | None = None,
    is_7net: bool = False,
) -> Image.Image:
    """Render authentic 4-view Workbench hippocampus surface for the cohort."""
    cname_clean = cinfo["cohort_name"].replace(" ", "_")
    ds_clean = cinfo["dataset_name"].replace(" ", "_")
    method_clean = str(cinfo.get("method", "")).replace(" ", "_")
    cache_prefix = f"{method_clean}_" if method_clean else ""
    legacy_7net_namespace = bool(cinfo.get("legacy_7net_namespace", False))
    atlas_tag = "7net_" if legacy_7net_namespace else ""
    atlas_folder = "kong2022_7net" if legacy_7net_namespace else "kong2022"

    if panels_dir is not None:
        cached_png = panels_dir / f"wb_subfield_{cache_prefix}{ds_clean}_{cname_clean}_{atlas_tag}{smoothing}_{variant}.png"
    else:
        cached_png = None

    group_root = Path(cinfo["group_root"])
    grad_sub = Path(cinfo.get("grad_subpath", group_root / f"gradient-spectral/{atlas_folder}/{smoothing}/cov_0p80/g1g2g3"))
    ml_path = grad_sub / "hemi_L/group_atlas_manifest.json"
    mr_path = grad_sub / "hemi_R/group_atlas_manifest.json"

    ml = json.loads(ml_path.read_text(encoding="utf-8"))
    mr = json.loads(mr_path.read_text(encoding="utf-8"))

    surf_l = Path(ml["template_left_or_right_surface"])
    surf_r = Path(mr["template_left_or_right_surface"])
    # This panel is a gradient-spectral surface, not a WTA surface. The
    # comparison method metadata carries Kong's WTA normalized-top20 choice,
    # so it must not be reused for the gradient atlas labels.
    gradient_assignment_variant = (
        "fisherz"
        if cinfo.get("method") in COMPARISON_METHODS
        else cinfo.get("assignment_variant", "normalized_top20_count")
    )
    lbl_l = select_manifest_label_path(ml, assignment_variant=gradient_assignment_variant)
    lbl_r = select_manifest_label_path(mr, assignment_variant=gradient_assignment_variant)
    subject = str(ml.get("template_subject", "")).replace("sub-", "").strip()
    spec_path = Path(ml["template_spec_path"])

    inputs = panel_inputs(
        [ml_path, mr_path, surf_l, surf_r, lbl_l, lbl_r, spec_path,
         REPO_ROOT / 'config/wb_locked_native_view_lateral_medial.scene',
         Path(__file__), CORE_DIR / 'lib/surface_utils.py'],
        smoothing=smoothing, variant=variant, method=method_clean,
        assignment_variant=gradient_assignment_variant,
    )
    if cache_matches(cached_png, inputs):
        return Image.open(cached_png).convert('RGB')

    render_out = render_tmp_dir / f"wb_subfield_{cache_prefix}{ds_clean}_{cname_clean}_{atlas_tag}{smoothing}_{variant}"
    render = render_locked_grid(
        subject=subject,
        scene=REPO_ROOT / "config/wb_locked_native_view_lateral_medial.scene",
        outdir=render_out,
        name=f"k3_subfields_{cache_prefix}{cname_clean}_{smoothing}_{variant}",
        title=f"{cinfo['dataset_name']} {cinfo['cohort_name']}",
        left_labels=lbl_l,
        right_labels=lbl_r,
        left_surface=surf_l,
        right_surface=surf_r,
        spec_path=spec_path,
    )
    native_png = Path(render["native_scene_png"])
    im_native = Image.open(native_png).convert("RGBA")
    grid = build_grid_canvas(ventral=im_native, dorsal=None, layout="1x2")
    grid_rgb = grid.convert("RGB")
    if cached_png is not None:
        panels_dir.mkdir(parents=True, exist_ok=True)
        grid_rgb.save(cached_png)
        publish_cache(cached_png, inputs)
    return grid_rgb


def select_manifest_label_path(manifest: dict, *, assignment_variant: str = 'normalized_top20_count') -> Path:
    """Select the explicitly declared method-specific label; never substitute."""
    variants = manifest.get("assignment_variants", {})
    preferred = variants.get(assignment_variant)
    if preferred is None:
        raise KeyError(f"Manifest lacks the canonical {assignment_variant} label variant")
    label_path = preferred.get("template_functional_atlas_label_variant") or preferred.get(
        "template_functional_atlas_label"
    )
    if label_path is None:
        raise KeyError("Selected manifest label variant has no template label path")
    return Path(label_path)


def render_subfield_bar_axis(ax, subregion_entries: list[dict], show_ylabel: bool = False, is_bottom_row: bool = False):
    """Render clean stacked bars with maximal font sizes and zero text collision."""
    subregions = ["Ant", "Body", "Post"]
    x_positions = np.array([0, 1, 2])
    bar_width = 0.56

    bottom_vals = np.zeros(len(subregions), dtype=np.float64)

    for s_idx in SUBFIELD_ORDER:
        sname = SUBFIELD_NAMES[s_idx]
        scolor = SUBFIELD_COLORS[s_idx]
        vals = np.array([entry["composition"].get(sname, 0.0) for entry in subregion_entries], dtype=np.float64)

        ax.bar(
            x_positions,
            vals,
            width=bar_width,
            bottom=bottom_vals,
            color=scolor,
            edgecolor="#ffffff",
            linewidth=1.2,
            zorder=3,
        )

        for i, val in enumerate(vals):
            if val >= 5.5:
                y_center = bottom_vals[i] + val / 2.0
                text_color = "#ffffff" if s_idx in (1, 5) else "#0f172a"
                fs = 14.5 if val >= 12.0 else 12.0
                ax.text(
                    x_positions[i],
                    y_center,
                    f"{val:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=fs,
                    fontweight="heavy",
                    color=text_color,
                    zorder=4,
                )

        bottom_vals += vals

    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    if show_ylabel:
        ax.set_yticklabels(["0", "25", "50", "75", "100%"], fontsize=14.5, fontweight="bold", color="#1e293b")
        ax.set_ylabel("Subfield Area %", fontsize=15.0, fontweight="heavy", color="#0b2447")
    else:
        ax.set_yticklabels([])

    ax.set_xticks(x_positions)
    if is_bottom_row:
        ax.set_xticklabels(["Anterior\n(Head)", "Middle\n(Body)", "Posterior\n(Tail)"], fontsize=15.0, fontweight="heavy", color="#0f172a")
    else:
        ax.set_xticklabels(["Ant", "Body", "Post"], fontsize=14.5, fontweight="heavy", color="#1e293b")

    ax.grid(axis="y", linestyle="--", alpha=0.55, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(1.2)


def render_subfield_composition_figure(
    all_cohort_data: list[dict],
    atlas_mode: str,
    out_path: Path,
    spec: GridSpec = GridSpec("2mm", "36p"),
    is_comparison: bool | None = None,
    show_version_label: bool | None = None,
    show_subtitle: bool | None = None,
):
    """Render the full 2-column cross-dataset subfield composition figure."""
    if is_comparison is None:
        is_comparison = any("method" in c.get("cohort", {}) for c in all_cohort_data)
    if show_version_label is None:
        show_version_label = not is_comparison
    if show_subtitle is None:
        show_subtitle = not is_comparison

    data_col1 = [c for c in all_cohort_data if c["cohort"]["col"] == 1]
    data_col2 = [c for c in all_cohort_data if c["cohort"]["col"] == 2]
    n_rows = max(len(data_col1), len(data_col2))

    fig_h = 27.0 * (n_rows / 5.0) if n_rows > 5 else 27.0
    fig = plt.figure(figsize=(38, fig_h), dpi=220)

    top_margin = 0.935
    bottom_margin = 0.075

    vinfo = PANORAMA_VERSIONS[atlas_mode]

    # Title & Subtitle
    if show_version_label:
        title_text = (
            f"CROSS-DATASET HIPPOCAMPAL SUBFIELD COMPOSITION ({spec.smoothing.upper()} SMOOTH, K=3 GRADIENT CLUSTERS)\n"
            f"[{vinfo['title_label']}]"
        )
        title_y = 0.982
        title_va = "top"
    else:
        title_text = f"CROSS-DATASET HIPPOCAMPAL SUBFIELD COMPOSITION ({spec.smoothing.upper()} SMOOTH, K=3 GRADIENT CLUSTERS)"
        title_y = 0.972
        title_va = "center"

    fig.suptitle(
        title_text,
        fontsize=25.0,
        fontweight="heavy",
        color="#0a192f",
        y=title_y,
        va=title_va,
    )

    if show_subtitle:
        total_subjects = sum(c["cohort"]["n_subjects"] for c in all_cohort_data)
        subtitle = f"MultiHist7 Anatomical Subfields Overlapping Functional Clusters (Anterior, Body, Posterior) | N = {total_subjects} across {len(all_cohort_data)} cohorts"
        fig.text(
            0.50,
            0.956,
            subtitle,
            ha="center",
            va="center",
            fontsize=16.5,
            fontweight="medium",
            color="#475569",
        )

    gs_main = fig.add_gridspec(
        nrows=1,
        ncols=2,
        left=0.015,
        right=0.985,
        top=top_margin,
        bottom=bottom_margin,
        wspace=0.045,
    )

    width_ratios = [0.13, 0.19, 0.34, 0.34]

    gs_left = gs_main[0, 0].subgridspec(
        nrows=n_rows,
        ncols=4,
        width_ratios=width_ratios,
        hspace=0.16,
        wspace=0.055,
    )

    gs_right = gs_main[0, 1].subgridspec(
        nrows=n_rows,
        ncols=4,
        width_ratios=width_ratios,
        hspace=0.16,
        wspace=0.055,
    )

    data_col1 = [c for c in all_cohort_data if c["cohort"]["col"] == 1]
    data_col2 = [c for c in all_cohort_data if c["cohort"]["col"] == 2]

    top_axes_col1 = []
    top_axes_col2 = []

    def render_row(gs_row, cdata, is_top_row, is_bottom_row, top_axes_collector=None):
        spec = cdata["cohort"]
        ax_meta = fig.add_subplot(gs_row[0])
        ax_meta.axis("off")
        ax_meta.add_patch(
            patches.FancyBboxPatch(
                (0.02, 0.02),
                0.96,
                0.96,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                fc=spec["bg_color"],
                ec="#cbd2d9",
                lw=1.5,
                transform=ax_meta.transAxes,
                zorder=0,
            )
        )

        # Dataset Badge
        ax_meta.text(
            0.50,
            0.82,
            spec["dataset_name"],
            fontsize=22.0,
            fontweight="heavy",
            ha="center",
            va="center",
            color=spec["badge_fg"],
            transform=ax_meta.transAxes,
            bbox=dict(
                boxstyle="round,pad=0.35,rounding_size=0.30",
                fc=spec["badge_bg"],
                ec="none",
            ),
        )

        # Cohort Name
        cohort_fs = 26.0 if len(spec["cohort_name"]) <= 8 else 21.0
        ax_meta.text(
            0.50,
            0.54,
            spec["cohort_name"],
            fontsize=cohort_fs,
            fontweight="heavy",
            ha="center",
            va="center",
            color="#0b2447",
            transform=ax_meta.transAxes,
        )

        # Sample Size (N)
        ax_meta.text(
            0.50,
            0.30,
            f"(N = {cdata['profiles']['L'][0]['n_subjects']})",
            fontsize=20.0,
            fontweight="bold",
            ha="center",
            va="center",
            color="#334e68",
            transform=ax_meta.transAxes,
        )

        # Cohort Description
        desc_fs = 16.0 if len(spec["cohort_desc"]) <= 18 else 14.0
        ax_meta.text(
            0.50,
            0.11,
            spec["cohort_desc"],
            fontsize=desc_fs,
            style="italic",
            fontweight="medium",
            ha="center",
            va="center",
            color="#627d98",
            transform=ax_meta.transAxes,
        )

        # Surface Image
        ax_surf = fig.add_subplot(gs_row[1])
        ax_surf.imshow(cdata["surface_img"], aspect="equal")
        ax_surf.axis("off")

        # Left Hippocampus Subfields
        ax_bar_L = fig.add_subplot(gs_row[2])
        render_subfield_bar_axis(
            ax=ax_bar_L,
            subregion_entries=cdata["profiles"]["L"],
            show_ylabel=True,
            is_bottom_row=is_bottom_row,
        )

        # Right Hippocampus Subfields
        ax_bar_R = fig.add_subplot(gs_row[3])
        render_subfield_bar_axis(
            ax=ax_bar_R,
            subregion_entries=cdata["profiles"]["R"],
            show_ylabel=False,
            is_bottom_row=is_bottom_row,
        )

        if top_axes_collector is not None:
            top_axes_collector.extend([ax_meta, ax_surf, ax_bar_L, ax_bar_R])

    # Left Column (5 Cohorts)
    for idx, cdata in enumerate(data_col1):
        render_row(
            gs_row=[gs_left[idx, c] for c in range(4)],
            cdata=cdata,
            is_top_row=(idx == 0),
            is_bottom_row=(idx == len(data_col1) - 1),
            top_axes_collector=top_axes_col1 if idx == 0 else None,
        )

    # Right Column (5 Cohorts)
    for idx, cdata in enumerate(data_col2):
        render_row(
            gs_row=[gs_right[idx, c] for c in range(4)],
            cdata=cdata,
            is_top_row=(idx == 0),
            is_bottom_row=(idx == len(data_col2) - 1),
            top_axes_collector=top_axes_col2 if idx == 0 else None,
        )

    # Column Headers
    y_col_header = top_margin + 0.006
    method_names = [
        data_col1[0]["cohort"].get("method_name") if data_col1 else None,
        data_col2[0]["cohort"].get("method_name") if data_col2 else None,
    ]
    for ax_list, method_name in zip([top_axes_col1, top_axes_col2], method_names):
        p0 = (ax_list[0].get_position().x0 + ax_list[0].get_position().x1) / 2.0
        p1 = (ax_list[1].get_position().x0 + ax_list[1].get_position().x1) / 2.0
        p2 = (ax_list[2].get_position().x0 + ax_list[2].get_position().x1) / 2.0
        p3 = (ax_list[3].get_position().x0 + ax_list[3].get_position().x1) / 2.0

        atlas_header = format_comparison_column_title(method_name, "GROUP ATLAS SURFACE")

        fig.text(p0, y_col_header, "COHORT PROFILE", ha="center", va="bottom", fontsize=17.0, fontweight="heavy", color="#102a43")
        fig.text(p1, y_col_header, atlas_header, ha="center", va="bottom", fontsize=17.0, fontweight="heavy", color="#102a43")
        fig.text(p2, y_col_header, "LEFT HIPPOCAMPUS (HEMI-L) SUBFIELDS", ha="center", va="bottom", fontsize=18.0, fontweight="heavy", color="#0f4c81")
        fig.text(p3, y_col_header, "RIGHT HIPPOCAMPUS (HEMI-R) SUBFIELDS", ha="center", va="bottom", fontsize=18.0, fontweight="heavy", color="#0f4c81")

    # Global Footer Legend
    legend_ax = fig.add_axes([0.05, 0.006, 0.90, 0.036])
    legend_ax.axis("off")

    legend_patches = [
        patches.Patch(facecolor=SUBFIELD_COLORS[1], edgecolor="#333333", label="Sub (Subiculum)"),
        patches.Patch(facecolor=SUBFIELD_COLORS[2], edgecolor="#333333", label="CA1"),
        patches.Patch(facecolor=SUBFIELD_COLORS[3], edgecolor="#333333", label="CA2"),
        patches.Patch(facecolor=SUBFIELD_COLORS[4], edgecolor="#333333", label="CA3"),
        patches.Patch(facecolor=SUBFIELD_COLORS[5], edgecolor="#333333", label="CA4"),
    ]

    legend_ax.legend(
        handles=legend_patches,
        loc="center",
        ncol=5,
        frameon=True,
        facecolor="#fcfdfe",
        edgecolor="#94a3b8",
        fontsize=19.0,
        handlelength=1.6,
        handleheight=1.0,
        handletextpad=0.6,
        columnspacing=3.2,
        borderpad=0.45,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, facecolor="#ffffff", edgecolor="none")
    plt.close(fig)
    print(f"Saved figure to {out_path}")


def export_stats_tsv(all_cohort_data: list[dict], out_path: Path, atlas_mode: str):
    """Export comprehensive statistics TSV across all cohorts."""
    lines = [
        "atlas_mode\tcohort_id\tcohort_name\tcohort_desc\tn_subjects\themisphere\tcluster_id\tcluster_name\tsubfield_index\tsubfield_name\tsubfield_full_name\tmean_percent\tsem_percent"
    ]
    for cdata in all_cohort_data:
        cinfo = cdata["cohort"]
        n_sub = cdata["profiles"]["L"][0]["n_subjects"]
        for hemi in ["L", "R"]:
            for sentry in cdata["profiles"][hemi]:
                cid = sentry["cluster_id"]
                cname = sentry["subregion_name"]
                for s_idx in SUBFIELD_ORDER:
                    s_name = SUBFIELD_NAMES[s_idx]
                    s_full = SUBFIELD_FULL_NAMES[s_idx]
                    mean_p = sentry["composition"].get(s_name, 0.0)
                    sem_p = sentry["sem"].get(s_name, 0.0)
                    lines.append(
                        f"{atlas_mode}\t{cinfo['id']}\t{cinfo['cohort_name']}\t{cinfo['cohort_desc']}\t{n_sub}\t{hemi}\t{cid}\t{cname}\t{s_idx}\t{s_name}\t{s_full}\t{mean_p:.4f}\t{sem_p:.4f}"
                    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved stats TSV table to {out_path}")


def process_cohort(
    cinfo: dict,
    atlas_mode: str,
    render_tmp_dir: Path,
    smoothing: str = "2mm",
    variant: str = "36p",
    panels_dir: Path | None = None,
    is_7net: bool = False,
) -> dict:
    prof = compute_cohort_subfield_composition(
        cinfo=cinfo,
        atlas_mode=atlas_mode,
        smoothing=smoothing,
        is_7net=is_7net,
    )
    surf_img = render_cohort_surface_img(
        cinfo,
        render_tmp_dir,
        smoothing=smoothing,
        variant=variant,
        panels_dir=panels_dir,
        is_7net=is_7net,
    )
    return {
        "cohort": cinfo,
        "profiles": prof,
        "surface_img": surf_img,
    }


def _render_grid():
    parser = argparse.ArgumentParser(description="Render 2-column multidataset subfield composition across atlas modes.")
    parser.add_argument("--version", choices=["legacy", COMPARISON_VERSION], default="legacy", help="Data branch; legacy is unchanged, or use the additive HCP/JJ_TMS comparison.")
    parser.add_argument("--mode", choices=["all", "v1", "v2", "v3"], default="v2", help="Atlas mode to run (default: v2)")
    parser.add_argument("--smoothing", choices=["2mm", "4mm"], default="2mm", help="Smoothing resolution (default: 2mm)")
    parser.add_argument("--variant", choices=["36p", "22p"], default="36p", help="Preprocessing variant (default: 36p)")
    parser.add_argument(
        "--atlas",
        choices=["kong2022", "7net"],
        default="7net",
        help="Atlas / network taxonomy version. Default: 7net (canonical7, unsuffixed); kong2022 is an equivalent path alias.",
    )
    parser.add_argument("--grid", choices=list(GRID_VARIANTS), default=None, help="Grid variant shortcut")
    parser.add_argument("--out-png", default=None, help="Output PNG path override")
    parser.add_argument("--stats-tsv", default=None, help="Stats TSV path override")
    parser.add_argument(
        "--show-version-label",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show or hide version label in suptitle (defaults to False for comparison, True for legacy)",
    )
    parser.add_argument(
        "--show-subtitle",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show or hide subtitle (defaults to False for comparison, True for legacy)",
    )
    args = parser.parse_args()

    if args.version == COMPARISON_VERSION and args.mode != "v2":
        parser.error(f"--mode {args.mode!r} is only supported by --version legacy; {COMPARISON_VERSION} has one method-comparison panorama")

    spec = GridSpec.from_tag(args.grid, atlas=args.atlas) if args.grid else GridSpec(args.smoothing, args.variant, atlas=args.atlas)

    if args.version == COMPARISON_VERSION:
        spec = GridSpec(spec.smoothing, spec.variant, atlas="7net")
        col1, col2 = get_comparison_visual_columns(spec)
    else:
        col1, col2 = get_multidataset_cohorts(spec)
    cohorts = col1 + col2

    modes_to_run = ["v1_fixed_hcp151_atlas", "v2_cohort_specific_atlas", "v3_individual_template_matching"] if args.mode == "all" else [
        {"v1": "v1_fixed_hcp151_atlas", "v2": "v2_cohort_specific_atlas", "v3": "v3_individual_template_matching"}[args.mode]
    ]

    t_start = time.time()
    print(f"Pre-loading Gifti files in parallel ({spec.tag})...")
    preload_items = []
    atlas_folder = "kong2022_7net" if spec.uses_legacy_7net_namespace else "kong2022"
    for cinfo in cohorts:
        grad_sub = Path(cinfo.get("grad_subpath", cinfo["group_root"] / f"gradient-spectral/{atlas_folder}/{spec.smoothing}/cov_0p80/g1g2g3"))
        hdir = grad_sub / "hemi_L"
        if not (hdir / "group_atlas_manifest.json").exists():
            raise FileNotFoundError(hdir / "group_atlas_manifest.json")
        man = json.loads((hdir / "group_atlas_manifest.json").read_text(encoding="utf-8"))
        raw_subjects = man.get("cluster_roi_profile_subjects_used") or list(man.get("actual_consumed_input_paths", {}).keys())
        for raw_s in raw_subjects:
            s = str(raw_s).removeprefix("sub-")
            for hemi in ["L", "R"]:
                lp = cinfo["hippunfold_root"] / f"sub-{s}/hippunfold/sub-{s}/metric/sub-{s}_hemi-{hemi}_den-512_label-hipp_atlas-multihist7_subfields.label.gii"
                sp = cinfo["hippunfold_root"] / f"sub-{s}/hippunfold/sub-{s}/surf/sub-{s}_hemi-{hemi}_space-corobl_den-512_label-hipp_midthickness.surf.gii"
                preload_items.append((lp, sp))

    def _worker(item):
        lp, sp = item
        if lp.exists():
            get_cached_label(lp)
        if sp.exists():
            get_cached_weight(sp)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(_worker, preload_items))
    print(f"Preloaded {len(preload_items)} items in {time.time()-t0:.2f}s")

    render_tmp_dir = OUTPUT_DIR / f"temp_wb_renders_{spec.file_tag}"
    render_tmp_dir.mkdir(parents=True, exist_ok=True)

    for mode in modes_to_run:
        t_mode = time.time()
        file_tag = PANORAMA_VERSIONS[mode]["file_tag"]
        banner_tag = (
            f"HCP/JJ_TMS 7-Network Comparison ({spec.tag})"
            if args.version == COMPARISON_VERSION
            else f"{PANORAMA_VERSIONS[mode]['title_label']} ({spec.file_tag})"
        )
        print(f"\n=======================================================")
        print(f"=== Rendering Panorama: {banner_tag} ===")
        print(f"=======================================================")

        # Workbench launches a shell wrapper plus its native process per worker.
        # Two render workers keep the suite + workers + children below IDAS 10.
        with ThreadPoolExecutor(max_workers=2) as ex:
            def _process(cinfo):
                return process_cohort(
                    cinfo,
                    mode,
                    render_tmp_dir,
                    smoothing=spec.smoothing,
                    variant=spec.variant,
                    panels_dir=PANELS_DIR,
                    is_7net=spec.is_7net,
                )

            futures = [ex.submit(_process, cinfo) for cinfo in cohorts]
            all_cohort_data = [f.result() for f in futures]

        if args.stats_tsv:
            tsv_path = Path(args.stats_tsv).resolve()
        else:
            if mode == "v2_cohort_specific_atlas":
                branch_tag = f"{COMPARISON_VERSION}_{spec.tag}" if args.version == COMPARISON_VERSION else spec.file_tag
                tsv_path = STATS_DIR / f"multidataset_subfield_composition_{branch_tag}_stats.tsv"
            else:
                tsv_path = STATS_DIR / f"multidataset_subfield_composition_{spec.file_tag}_stats_{file_tag}.tsv"
        export_stats_tsv(all_cohort_data, tsv_path, mode)

        if spec.file_tag == "2mm_36p" and mode == "v2_cohort_specific_atlas":
            shutil.copy2(tsv_path, STATS_DIR / "multidataset_subfield_composition_stats.tsv")

        if args.out_png:
            out_png = Path(args.out_png).resolve()
        else:
            if mode == "v2_cohort_specific_atlas":
                branch_tag = f"{COMPARISON_VERSION}_{spec.tag}" if args.version == COMPARISON_VERSION else spec.file_tag
                out_png = OUTPUT_DIR / f"multidataset_subfield_composition_2mm_2col_{branch_tag}.png"
            else:
                out_png = OUTPUT_DIR / f"multidataset_subfield_composition_{spec.file_tag}_2col_{file_tag}.png"

        print(f"Rendering 2-column figure to {out_png}...")
        render_subfield_composition_figure(
            all_cohort_data,
            mode,
            out_png,
            spec=spec,
            is_comparison=(args.version == COMPARISON_VERSION),
            show_version_label=args.show_version_label,
            show_subtitle=args.show_subtitle,
        )

        if spec.file_tag == "2mm_36p" and mode == "v2_cohort_specific_atlas":
            shutil.copy2(out_png, OUTPUT_DIR / "multidataset_subfield_composition_2mm_2col.png")

        print(f"=== Completed {mode} in {time.time()-t_mode:.2f}s ===")

    print(f"\nAll panorama versions completed successfully in {time.time()-t_start:.2f}s!")


if __name__ == "__main__":
    raise SystemExit(
        "Standalone rendering has been removed. Use "
        "scripts/visualize/run_multidataset_suite.py "
        "with --figures to select the figure type; all four grids are processed in 4-way parallel by default."
    )

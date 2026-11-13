#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
GLOBAL_NETWORK_STYLE_JSON = REPO_ROOT / "config" / "hipp_network_style.json"
DEFAULT_SCENE = REPO_ROOT / "config" / "wb_locked_native_view_lateral_medial.scene"

import sys
CORE_DIR = REPO_ROOT / "scripts" / "core_workflow"
VIS_LIB_DIR = REPO_ROOT / "scripts" / "visualize" / "lib"
for p in (CORE_DIR, VIS_LIB_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lib.surface_utils import render_hippocampal_grid
from surface_panel_cache import panel_inputs, cache_matches, publish_cache
from multidataset_cohorts import (
    COMPARISON_VERSION,
    GridSpec,
    GRID_VARIANTS,
    format_comparison_column_title,
    get_comparison_visual_columns,
    get_comparison_output_tag,
    get_multidataset_cohorts,
)



def load_network_hex_colors(path: Path = GLOBAL_NETWORK_STYLE_JSON) -> dict[str, str]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for row in payload.values():
        if isinstance(row, dict) and "name" in row and "rgba" in row:
            r, g, b, _a = row["rgba"]
            out[row["name"]] = f"#{int(r):02x}{int(g):02x}{int(b):02x}"
    return out


COLOR_MAP = load_network_hex_colors()
DEFAULT_NET_COLORS = {
    "VentralAttention": "#d95f02",
    "Somatomotor": "#1f78b4",
    "Default": "#e31a1c",
    "Auditory": "#18b7c9",
    "DorsalAttention": "#33a02c",
    "Control": "#ff7f00",
    "Visual": "#6a3d9a",
    "Language": "#b15928",
    "CinguloOpercular": "#fb9a99",
}

for k, v in DEFAULT_NET_COLORS.items():
    if k not in COLOR_MAP:
        COLOR_MAP[k] = v

ABBREV_MAP = {
    "Default": "DMN",
    "Visual": "VN",
    "Somatomotor": "SMN",
    "DorsalAttention": "DAN",
    "VentralAttention": "VAN",
    "Control": "CON",
    "Auditory": "AUD",
    "Language": "LAN",
    "Limbic": "LIM",
    "CinguloOpercular": "CO",
    "none": "None",
    "unassigned": "None",
}


def to_abbrev(net_name: str) -> str:
    return ABBREV_MAP.get(net_name, net_name)


CLUSTER_NAMES = {
    1: "Anterior",
    2: "Middle",
    3: "Posterior",
}

CLUSTER_SHORT_NAMES = {
    1: "Ant",
    2: "Mid",
    3: "Post",
}

COHORTS_COL1 = [
    {
        "dataset_name": "HCP 7T",
        "cohort_name": "HCP 7T",
        "cohort_desc": "Healthy Young Adults",
        "n_subjects": 151,
        "group_root": REPO_ROOT / "outputs_hcp/work/group_atlas/pfm",
        "bg_color": "#ffffff",
        "badge_bg": "#0b3c5d",
        "badge_fg": "#ffffff",
    },
    {
        "dataset_name": "Iowa 3T",
        "cohort_name": "HC",
        "cohort_desc": "Healthy Controls",
        "n_subjects": 67,
        "group_root": REPO_ROOT / "outputs_iowa_jj_tms/work/group_atlas/pfm/hc",
        "bg_color": "#f7faff",
        "badge_bg": "#1d5b79",
        "badge_fg": "#ffffff",
    },
    {
        "dataset_name": "Iowa 3T",
        "cohort_name": "Patient",
        "cohort_desc": "Depression Patients",
        "n_subjects": 26,
        "group_root": REPO_ROOT / "outputs_iowa_jj_tms/work/group_atlas/pfm/patient",
        "bg_color": "#f7faff",
        "badge_bg": "#1d5b79",
        "badge_fg": "#ffffff",
    },
    {
        "dataset_name": "Stanford",
        "cohort_name": "NTHC",
        "cohort_desc": "Non-Trauma Controls",
        "n_subjects": 51,
        "group_root": REPO_ROOT / "outputs_stanford/work/group_atlas/pfm/nthc",
        "bg_color": "#fffdf8",
        "badge_bg": "#8c4600",
        "badge_fg": "#ffffff",
    },
    {
        "dataset_name": "Stanford",
        "cohort_name": "TEHC",
        "cohort_desc": "Trauma-Exposed Controls",
        "n_subjects": 40,
        "group_root": REPO_ROOT / "outputs_stanford/work/group_atlas/pfm/tehc",
        "bg_color": "#fffdf8",
        "badge_bg": "#8c4600",
        "badge_fg": "#ffffff",
    },
]

COHORTS_COL2 = [
    {
        "dataset_name": "Stanford",
        "cohort_name": "NTS",
        "cohort_desc": "Non-Trauma Symptomatic",
        "n_subjects": 56,
        "group_root": REPO_ROOT / "outputs_stanford/work/group_atlas/pfm/nts",
        "bg_color": "#fffdf8",
        "badge_bg": "#8c4600",
        "badge_fg": "#ffffff",
    },
    {
        "dataset_name": "Stanford",
        "cohort_name": "NTS Any MDD",
        "cohort_desc": "Non-Trauma MDD (All)",
        "n_subjects": 43,
        "group_root": REPO_ROOT / "outputs_stanford/work/group_atlas/pfm/nts_any_mdd",
        "bg_color": "#fffdf8",
        "badge_bg": "#8c4600",
        "badge_fg": "#ffffff",
    },
    {
        "dataset_name": "Stanford",
        "cohort_name": "NTS Pure MDD",
        "cohort_desc": "Non-Trauma Pure MDD",
        "n_subjects": 11,
        "group_root": REPO_ROOT / "outputs_stanford/work/group_atlas/pfm/nts_pure_mdd",
        "bg_color": "#fffdf8",
        "badge_bg": "#8c4600",
        "badge_fg": "#ffffff",
    },
    {
        "dataset_name": "Stanford",
        "cohort_name": "TIS",
        "cohort_desc": "Trauma Symptomatic",
        "n_subjects": 47,
        "group_root": REPO_ROOT / "outputs_stanford/work/group_atlas/pfm/tis",
        "bg_color": "#fffdf8",
        "badge_bg": "#8c4600",
        "badge_fg": "#ffffff",
    },
    {
        "dataset_name": "Stanford",
        "cohort_name": "TIS PTSD",
        "cohort_desc": "Trauma-Informed PTSD",
        "n_subjects": 35,
        "group_root": REPO_ROOT / "outputs_stanford/work/group_atlas/pfm/tis_ptsd",
        "bg_color": "#fffdf8",
        "badge_bg": "#8c4600",
        "badge_fg": "#ffffff",
    },
]

COHORTS_SPEC = COHORTS_COL1 + COHORTS_COL2


def load_cohort_data_and_render_wb(
    cohort_entry: dict,
    temp_dir: Path,
    scene_path: Path,
    smoothing: str = "2mm",
    variant: str = "36p",
    panels_dir: Path | None = None,
    is_7net: bool = False,
) -> dict:
    legacy_7net_namespace = bool(cohort_entry.get("legacy_7net_namespace", False))
    group_root = Path(cohort_entry["group_root"])
    atlas_tag = "7net_" if legacy_7net_namespace else ""
    atlas_folder = "kong2022_7net" if legacy_7net_namespace else "kong2022"
    grad_sub = Path(cohort_entry.get("grad_subpath", group_root / f"gradient-spectral/{atlas_folder}/{smoothing}/cov_0p80/g1g2g3"))
    ml_path = grad_sub / "hemi_L/group_atlas_manifest.json"
    mr_path = grad_sub / "hemi_R/group_atlas_manifest.json"

    if not ml_path.exists() or not mr_path.exists():
        raise FileNotFoundError(f"Missing manifests under {group_root}")

    ml = json.loads(ml_path.read_text(encoding="utf-8"))
    mr = json.loads(mr_path.read_text(encoding="utf-8"))

    # Surfaces & Labels
    surf_l = Path(ml["template_left_or_right_surface"])
    surf_r = Path(mr["template_left_or_right_surface"])
    lbl_l_gii = Path(ml["assignment_variants"]["fisherz"]["template_functional_atlas_label_variant"])
    lbl_r_gii = Path(mr["assignment_variants"]["fisherz"]["template_functional_atlas_label_variant"])

    subject = str(ml.get("template_subject", "")).replace("sub-", "").strip()
    spec_path = Path(ml["template_spec_path"])

    cname_clean = cohort_entry["cohort_name"].replace(" ", "_")
    ds_clean = cohort_entry["dataset_name"].replace(" ", "_")
    method_clean = str(cohort_entry.get("method", "")).replace(" ", "_")
    cache_prefix = f"{method_clean}_" if method_clean else ""
    ns = cohort_entry["n_subjects"]

    inputs = panel_inputs(
        [ml_path, mr_path, surf_l, surf_r, lbl_l_gii, lbl_r_gii,
         spec_path, scene_path, Path(__file__), CORE_DIR / 'lib/surface_utils.py'],
        smoothing=smoothing, variant=variant, method=method_clean, subjects=ns,
    )

    if panels_dir is not None:
        cached_png = panels_dir / f"wb_{cache_prefix}{ds_clean}_{cname_clean}_{atlas_tag}{smoothing}_{variant}_N{ns}_biglegend.png"
        if cache_matches(cached_png, inputs):
            wb_image = Image.open(cached_png).convert("RGBA")
        else:
            wb_image = None
    else:
        cached_png = None
        wb_image = None

    if wb_image is None:
        render_out = temp_dir / f"wb_{cache_prefix}{ds_clean}_{cname_clean}_{atlas_tag}{smoothing}_{variant}"
        render = render_hippocampal_grid(
            subject=subject,
            outdir=render_out,
            name=f"k3_{cache_prefix}{atlas_tag}{smoothing}_{variant}_{cname_clean}",
            title=f"{cohort_entry['dataset_name']} {cohort_entry['cohort_name']} (N={cohort_entry['n_subjects']})",
            left_labels=lbl_l_gii,
            right_labels=lbl_r_gii,
            left_surface=surf_l,
            right_surface=surf_r,
        )
        wb_biglegend_png = Path(render["biglegend_png"])
        wb_image = Image.open(wb_biglegend_png).convert("RGBA")
        if cached_png is not None:
            panels_dir.mkdir(parents=True, exist_ok=True)
            wb_image.save(cached_png)
            publish_cache(cached_png, inputs)

    # Vertex stats & status for L and R
    hemi_stats = {}
    for hemi, m, gii_path in [("L", ml, lbl_l_gii), ("R", mr, lbl_r_gii)]:
        hdir = grad_sub / f"hemi_{hemi}"
        lbl_npy_file = hdir / "group_labels_final.npy"
        if not lbl_npy_file.exists():
            lbl_npy_file = hdir / "group_labels.npy"
        lbl_arr = np.load(lbl_npy_file).flatten()
        total_vertices = len(lbl_arr)

        unique, counts = np.unique(lbl_arr, return_counts=True)
        counts_dict = dict(zip(unique, counts))

        status_file = hdir / "cluster_assignment_status_fisherz.json"
        status_data = json.loads(status_file.read_text(encoding="utf-8"))
        items_by_cid = {int(it["cluster_id"]): it for it in status_data.get("items", [])}

        clusters_info = []
        for cid in [1, 2, 3]:
            cnt = int(counts_dict.get(cid, 0))
            pct = (cnt / float(total_vertices)) * 100.0 if total_vertices > 0 else 0.0
            it = items_by_cid.get(cid, {})
            top1_net = str(it.get("top1_network", "unassigned"))
            top1_z = float(it.get("top1_fisher_z", it.get("top1_value", 0.0)))
            top2_net = str(it.get("top2_network", "unassigned"))
            top2_z = float(it.get("top2_fisher_z", it.get("top2_value", 0.0)))
            margin = float(it.get("margin", top1_z - top2_z))

            clusters_info.append({
                "cluster_id": cid,
                "cluster_name": CLUSTER_NAMES.get(cid, f"Cluster {cid}"),
                "cluster_short": CLUSTER_SHORT_NAMES.get(cid, f"C{cid}"),
                "vertex_count": cnt,
                "total_vertices": total_vertices,
                "percentage": pct,
                "top1_network": top1_net,
                "top1_z": top1_z,
                "top2_network": top2_net,
                "top2_z": top2_z,
                "margin": margin,
                "color": COLOR_MAP.get(top1_net, "#888888"),
            })

        hemi_stats[hemi] = {
            "total_vertices": total_vertices,
            "clusters": clusters_info,
        }

    return {
        "dataset_name": cohort_entry["dataset_name"],
        "cohort_name": cohort_entry["cohort_name"],
        "cohort_desc": cohort_entry["cohort_desc"],
        "n_subjects": cohort_entry["n_subjects"],
        "badge_bg": cohort_entry["badge_bg"],
        "badge_fg": cohort_entry["badge_fg"],
        "bg_color": cohort_entry["bg_color"],
        "wb_image": wb_image,
        "hemi_stats": hemi_stats,
    }


def render_cohort_row(fig, gs_row, d: dict, used_networks: set):
    """Render a single cohort row (Metadata Card + Workbench Surface + Subregion Stats Card)."""
    # 1. Left Metadata Axis
    ax_meta = fig.add_subplot(gs_row[0])
    ax_meta.axis("off")
    bg_c = d["bg_color"]
    ax_meta.add_patch(
        patches.FancyBboxPatch(
            (0.02, 0.02),
            0.96,
            0.96,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            fc=bg_c,
            ec="#cbd2d9",
            lw=1.3,
            transform=ax_meta.transAxes,
            zorder=0,
        )
    )

    # Dataset Pill
    ax_meta.text(
        0.50,
        0.82,
        f"{d['dataset_name']}",
        fontsize=21.0,
        fontweight="heavy",
        ha="center",
        va="center",
        color=d["badge_fg"],
        transform=ax_meta.transAxes,
        bbox=dict(
            boxstyle="round,pad=0.38,rounding_size=0.28",
            fc=d["badge_bg"],
            ec="none",
        ),
    )

    # Cohort Name & Sample Size
    cohort_fs = 26.0 if len(d["cohort_name"]) <= 8 else 21.0
    ax_meta.text(
        0.50,
        0.54,
        f"{d['cohort_name']}",
        fontsize=cohort_fs,
        fontweight="heavy",
        ha="center",
        va="center",
        color="#0b2447",
        transform=ax_meta.transAxes,
    )
    ax_meta.text(
        0.50,
        0.31,
        f"(N = {d['n_subjects']})",
        fontsize=19.0,
        fontweight="bold",
        ha="center",
        va="center",
        color="#334e68",
        transform=ax_meta.transAxes,
    )
    desc_fs = 14.5 if len(d["cohort_desc"]) <= 20 else 13.0
    ax_meta.text(
        0.50,
        0.12,
        f"{d['cohort_desc']}",
        fontsize=desc_fs,
        style="italic",
        fontweight="medium",
        ha="center",
        va="center",
        color="#627d98",
        transform=ax_meta.transAxes,
    )

    # 2. Middle Workbench Surface Axis - Strict Equal Aspect Ratio
    ax_wb = fig.add_subplot(gs_row[1])
    ax_wb.axis("off")
    ax_wb.imshow(d["wb_image"], aspect="equal")

    # 3. Right Statistics Card Axis
    ax_stats = fig.add_subplot(gs_row[2])
    ax_stats.axis("off")
    ax_stats.set_xlim(0, 1)
    ax_stats.set_ylim(0, 1)

    # Background card
    ax_stats.add_patch(
        patches.FancyBboxPatch(
            (0.008, 0.02),
            0.984,
            0.96,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            fc="#fcfdfe",
            ec="#cdd7e0",
            lw=1.2,
            transform=ax_stats.transAxes,
            zorder=0,
        )
    )

    # Headers for Left and Right Hippocampus
    ax_stats.text(
        0.025,
        0.90,
        "Left Hippocampus (hemi-L | 512 v)",
        fontsize=19.5,
        fontweight="heavy",
        color="#0f4c81",
        va="center",
    )
    ax_stats.text(
        0.525,
        0.90,
        "Right Hippocampus (hemi-R | 512 v)",
        fontsize=19.5,
        fontweight="heavy",
        color="#b91c1c",
        va="center",
    )

    # Horizontal Proportion Stacked Bars for L & R
    bar_y = 0.70
    bar_h = 0.155

    # Left Hemisphere Stacked Proportion Bar
    l_clusters = d["hemi_stats"]["L"]["clusters"]
    cum_pct_l = 0.0
    for c in l_clusters:
        pct = c["percentage"]
        net = c["top1_network"]
        used_networks.add(net)
        c_color = c["color"]
        seg_x = 0.025 + (cum_pct_l / 100.0) * 0.460
        seg_w = (pct / 100.0) * 0.460
        ax_stats.add_patch(
            patches.Rectangle(
                (seg_x, bar_y),
                seg_w,
                bar_h,
                facecolor=c_color,
                edgecolor="#111111",
                lw=1.1,
                zorder=2,
            )
        )
        # Label on segment (Ant 41%, Mid 33%, Post 26%)
        ax_stats.text(
            seg_x + seg_w / 2.0,
            bar_y + bar_h / 2.0,
            f"{c['cluster_short']} {pct:.1f}%",
            ha="center",
            va="center",
            fontsize=15.5,
            fontweight="heavy",
            color="#ffffff",
            path_effects=[pe.withStroke(linewidth=2.8, foreground="#111111")],
            zorder=4,
        )
        cum_pct_l += pct

    # Right Hemisphere Stacked Proportion Bar
    r_clusters = d["hemi_stats"]["R"]["clusters"]
    cum_pct_r = 0.0
    for c in r_clusters:
        pct = c["percentage"]
        net = c["top1_network"]
        used_networks.add(net)
        c_color = c["color"]
        seg_x = 0.525 + (cum_pct_r / 100.0) * 0.450
        seg_w = (pct / 100.0) * 0.450
        ax_stats.add_patch(
            patches.Rectangle(
                (seg_x, bar_y),
                seg_w,
                bar_h,
                facecolor=c_color,
                edgecolor="#111111",
                lw=1.1,
                zorder=2,
            )
        )
        # Label on segment
        ax_stats.text(
            seg_x + seg_w / 2.0,
            bar_y + bar_h / 2.0,
            f"{c['cluster_short']} {pct:.1f}%",
            ha="center",
            va="center",
            fontsize=15.5,
            fontweight="heavy",
            color="#ffffff",
            path_effects=[pe.withStroke(linewidth=2.8, foreground="#111111")],
            zorder=4,
        )
        cum_pct_r += pct

    # Detailed breakdown text lines (3 rows: C1, C2, C3)
    row_y_centers = [0.52, 0.33, 0.14]
    for c_idx in range(3):
        cl = l_clusters[c_idx]
        cr = r_clusters[c_idx]
        ry = row_y_centers[c_idx]

        # Left side line
        lbl_l_bold = f"{cl['cluster_short']}: {cl['vertex_count']}/512 ({cl['percentage']:.1f}%)"
        lbl_l_net = f"→ {cl['top1_network']} (z={cl['top1_z']:.3f})"
        ax_stats.text(
            0.025,
            ry + 0.040,
            lbl_l_bold,
            fontsize=17.5,
            fontweight="heavy",
            color="#0b2447",
            va="center",
        )
        ax_stats.text(
            0.025,
            ry - 0.038,
            lbl_l_net,
            fontsize=16.5,
            fontweight="heavy",
            color=cl["color"],
            va="center",
        )

        # Right side line
        lbl_r_bold = f"{cr['cluster_short']}: {cr['vertex_count']}/512 ({cr['percentage']:.1f}%)"
        lbl_r_net = f"→ {cr['top1_network']} (z={cr['top1_z']:.3f})"
        ax_stats.text(
            0.525,
            ry + 0.040,
            lbl_r_bold,
            fontsize=17.5,
            fontweight="heavy",
            color="#0b2447",
            va="center",
        )
        ax_stats.text(
            0.525,
            ry - 0.038,
            lbl_r_net,
            fontsize=16.5,
            fontweight="heavy",
            color=cr["color"],
            va="center",
        )


def render_all_cohorts_overlay_figure(
    out_png: Path,
    tsv_out: Path | None = None,
    spec: GridSpec = GridSpec("2mm", "36p"),
    cohort_columns: tuple[list[dict], list[dict]] | None = None,
):
    scene_path = DEFAULT_SCENE
    if not scene_path.exists():
        raise FileNotFoundError(f"Missing scene file: {scene_path}")

    if cohort_columns is None:
        col1_specs, col2_specs = get_multidataset_cohorts(spec)
    else:
        col1_specs, col2_specs = cohort_columns
    data_col1 = []
    data_col2 = []
    panels_dir = out_png.parent / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="wb_overlay_all_") as tmp_dir:
        tmp_p = Path(tmp_dir)
        print(f"[RENDER] Processing Column 1 ({len(col1_specs)} cohorts, {spec.file_tag})...")
        for c_spec in col1_specs:
            print(f"  Col 1: {c_spec['dataset_name']} {c_spec['cohort_name']} (N={c_spec['n_subjects']})...")
            d = load_cohort_data_and_render_wb(
                c_spec,
                tmp_p,
                scene_path,
                smoothing=spec.smoothing,
                variant=spec.variant,
                panels_dir=panels_dir,
                is_7net=spec.is_7net,
            )
            data_col1.append(d)

        print(f"[RENDER] Processing Column 2 ({len(col2_specs)} cohorts, {spec.file_tag})...")
        for c_spec in col2_specs:
            print(f"  Col 2: {c_spec['dataset_name']} {c_spec['cohort_name']} (N={c_spec['n_subjects']})...")
            d = load_cohort_data_and_render_wb(
                c_spec,
                tmp_p,
                scene_path,
                smoothing=spec.smoothing,
                variant=spec.variant,
                panels_dir=panels_dir,
                is_7net=spec.is_7net,
            )
            data_col2.append(d)

    all_data = data_col1 + data_col2

    # Save TSV summary table
    if tsv_out is not None:
        tsv_rows = [
            "dataset\tcohort\tn_subjects\themisphere\tcluster_id\tsubregion\tvertex_count\ttotal_vertices\tpercentage\ttop1_network\ttop1_fisher_z\ttop2_network\ttop2_fisher_z\tmargin"
        ]
        for d in all_data:
            for hemi in ["L", "R"]:
                for c in d["hemi_stats"][hemi]["clusters"]:
                    tsv_rows.append(
                        f"{d['dataset_name']}\t{d['cohort_name']}\t{d['n_subjects']}\t{hemi}\t{c['cluster_id']}\t{c['cluster_short']}\t{c['vertex_count']}\t{c['total_vertices']}\t{c['percentage']:.2f}\t{c['top1_network']}\t{c['top1_z']:.4f}\t{c['top2_network']}\t{c['top2_z']:.4f}\t{c['margin']:.4f}"
                    )
        tsv_out.parent.mkdir(parents=True, exist_ok=True)
        tsv_out.write_text("\n".join(tsv_rows) + "\n", encoding="utf-8")
        print(f"[SUCCESS] Saved vertex stats TSV to: {tsv_out}")

    # Master Figure Geometry: 2 Columns x N Rows
    n_rows = max(len(col1_specs), len(col2_specs))
    fig_w = 48.0
    fig_h = 28.0 * (n_rows / 5.0) if n_rows > 5 else 28.0

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=220)
    fig.patch.set_facecolor("#ffffff")

    total_subjects = sum(c["n_subjects"] for c in col1_specs + col2_specs)
    # Suptitle
    fig.suptitle(
        f"Cross-Dataset Hippocampal {spec.smoothing} Functional Atlas (K=3 RouteA) & Subregion Vertex Composition{' [7-Network]' if spec.is_7net else ''}",
        fontsize=42.0,
        fontweight="heavy",
        x=0.50,
        y=0.988,
        ha="center",
        color="#0b2447",
    )

    # Subtitle
    fig.text(
        0.50,
        0.963,
        f"Original Workbench 3D Parcellation Overlays & Exact Subregion Vertex Counts (den-512) across {len(col1_specs + col2_specs)} Cohorts (N={total_subjects})",
        fontsize=24.0,
        fontweight="medium",
        ha="center",
        color="#334e68",
    )

    top_margin = 0.932
    bottom_margin = 0.075

    # Master GridSpec: 1 row, 2 major columns
    gs_main = fig.add_gridspec(
        nrows=1,
        ncols=2,
        left=0.012,
        right=0.988,
        top=top_margin,
        bottom=bottom_margin,
        wspace=0.035,
    )

    width_ratios = [0.14, 0.28, 0.58]

    gs_left = gs_main[0, 0].subgridspec(
        nrows=n_rows,
        ncols=3,
        width_ratios=width_ratios,
        hspace=0.15,
        wspace=0.020,
    )

    gs_right = gs_main[0, 1].subgridspec(
        nrows=n_rows,
        ncols=3,
        width_ratios=width_ratios,
        hspace=0.15,
        wspace=0.020,
    )

    # Column Super-Headers
    y_col_header = top_margin + 0.007
    col_w = (0.988 - 0.012 - 0.035) / 2.0
    col1_x0 = 0.012
    col2_x0 = col1_x0 + col_w + 0.035

    r0, r1, r2 = width_ratios
    rtot = sum(width_ratios)

    p0 = col1_x0 + (r0 / (2 * rtot)) * col_w
    p1 = col1_x0 + ((r0 + r1 / 2.0) / rtot) * col_w
    p2 = col1_x0 + ((r0 + r1 + r2 / 2.0) / rtot) * col_w

    col1_method = col1_specs[0].get("method_name") if col1_specs else None
    col2_method = col2_specs[0].get("method_name") if col2_specs else None
    fig.text(p0, y_col_header, "Cohort Profile", ha="center", va="bottom", fontsize=23.0, fontweight="heavy", color="#102a43")
    fig.text(p1, y_col_header, format_comparison_column_title(col1_method, "3D Workbench Parcellation"), ha="center", va="bottom", fontsize=23.0, fontweight="heavy", color="#102a43")
    fig.text(p2, y_col_header, "Subregion Vertex Composition (hemi-L & hemi-R)", ha="center", va="bottom", fontsize=23.0, fontweight="heavy", color="#0f4c81")

    q0 = col2_x0 + (r0 / (2 * rtot)) * col_w
    q1 = col2_x0 + ((r0 + r1 / 2.0) / rtot) * col_w
    q2 = col2_x0 + ((r0 + r1 + r2 / 2.0) / rtot) * col_w

    fig.text(q0, y_col_header, "Cohort Profile", ha="center", va="bottom", fontsize=23.0, fontweight="heavy", color="#102a43")
    fig.text(q1, y_col_header, format_comparison_column_title(col2_method, "3D Workbench Parcellation"), ha="center", va="bottom", fontsize=23.0, fontweight="heavy", color="#102a43")
    fig.text(q2, y_col_header, "Subregion Vertex Composition (hemi-L & hemi-R)", ha="center", va="bottom", fontsize=23.0, fontweight="heavy", color="#0f4c81")

    used_networks = set()

    # Render Left Column (5 Cohorts)
    for idx, d in enumerate(data_col1):
        render_cohort_row(
            fig=fig,
            gs_row=[gs_left[idx, c] for c in range(3)],
            d=d,
            used_networks=used_networks,
        )

    # Render Right Column (5 Cohorts)
    for idx, d in enumerate(data_col2):
        render_cohort_row(
            fig=fig,
            gs_row=[gs_right[idx, c] for c in range(3)],
            d=d,
            used_networks=used_networks,
        )

    # Bottom Global Summary & Network Legend: Clean separate zones to prevent any text overlap
    # 1. Left summary statistics card
    ax_footer_stats = fig.add_axes([0.012, 0.010, 0.580, 0.052])
    ax_footer_stats.axis("off")
    ax_footer_stats.add_patch(
        patches.FancyBboxPatch(
            (0.00, 0.02),
            1.00,
            0.96,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            fc="#f0f4f8",
            ec="#9fb3c8",
            lw=1.4,
            transform=ax_footer_stats.transAxes,
        )
    )

    # Grand statistics across all cohorts
    l_c1 = [d["hemi_stats"]["L"]["clusters"][0]["vertex_count"] for d in all_data]
    l_c2 = [d["hemi_stats"]["L"]["clusters"][1]["vertex_count"] for d in all_data]
    l_c3 = [d["hemi_stats"]["L"]["clusters"][2]["vertex_count"] for d in all_data]
    r_c1 = [d["hemi_stats"]["R"]["clusters"][0]["vertex_count"] for d in all_data]
    r_c2 = [d["hemi_stats"]["R"]["clusters"][1]["vertex_count"] for d in all_data]
    r_c3 = [d["hemi_stats"]["R"]["clusters"][2]["vertex_count"] for d in all_data]

    n_tot = len(all_data)
    summary_text_l = (
        f"Mean Left ({n_tot} Cohorts):   Ant: {np.mean(l_c1):.1f}±{np.std(l_c1):.1f} ({np.mean(l_c1)/5.12:.1f}%)  |  "
        f"Mid: {np.mean(l_c2):.1f}±{np.std(l_c2):.1f} ({np.mean(l_c2)/5.12:.1f}%)  |  "
        f"Post: {np.mean(l_c3):.1f}±{np.std(l_c3):.1f} ({np.mean(l_c3)/5.12:.1f}%)"
    )
    summary_text_r = (
        f"Mean Right ({n_tot} Cohorts):  Ant: {np.mean(r_c1):.1f}±{np.std(r_c1):.1f} ({np.mean(r_c1)/5.12:.1f}%)  |  "
        f"Mid: {np.mean(r_c2):.1f}±{np.std(r_c2):.1f} ({np.mean(r_c2)/5.12:.1f}%)  |  "
        f"Post: {np.mean(r_c3):.1f}±{np.std(r_c3):.1f} ({np.mean(r_c3)/5.12:.1f}%)"
    )

    ax_footer_stats.text(
        0.02,
        0.70,
        summary_text_l,
        fontsize=16.5,
        fontweight="bold",
        color="#102a43",
        va="center",
    )
    ax_footer_stats.text(
        0.02,
        0.30,
        summary_text_r,
        fontsize=16.5,
        fontweight="bold",
        color="#102a43",
        va="center",
    )

    # 2. Right legend card
    ax_footer_leg = fig.add_axes([0.600, 0.010, 0.388, 0.052])
    ax_footer_leg.axis("off")
    ax_footer_leg.add_patch(
        patches.FancyBboxPatch(
            (0.00, 0.02),
            1.00,
            0.96,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            fc="#ffffff",
            ec="#cbd5e1",
            lw=1.4,
            transform=ax_footer_leg.transAxes,
        )
    )

    legend_patches = []
    canon_order = ["VentralAttention", "Somatomotor", "Default", "Auditory", "DorsalAttention", "Control", "Visual", "Language", "CinguloOpercular"]
    sorted_nets = [net for net in canon_order if net in used_networks] + [net for net in sorted(used_networks) if net not in canon_order]
    for net in sorted_nets:
        abbr = to_abbrev(net)
        legend_patches.append(
            patches.Patch(facecolor=COLOR_MAP.get(net, "#888888"), edgecolor="#111111", lw=1.0, label=f"{abbr} ({net})")
        )

    ax_footer_leg.legend(
        handles=legend_patches,
        loc="center",
        ncol=min(5, len(legend_patches)),
        fontsize=15.0,
        frameon=False,
        handlelength=1.5,
        handleheight=1.1,
        columnspacing=1.5,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Saved clean 2-column Workbench overlay figure to: {out_png}")


def _render_grid():
    parser = argparse.ArgumentParser(description="Render 2-column 10-cohort Workbench hippocampus overlay master figure.")
    parser.add_argument("--version", choices=["legacy", COMPARISON_VERSION], default="legacy", help="Data branch; legacy is unchanged, or use the additive HCP/JJ_TMS comparison.")
    default_dir = REPO_ROOT / "scripts" / "visualize" / "multidataset_workbench_hippocampus_overlay"
    default_png = default_dir / "multidataset_workbench_hippocampus_overlay.png"
    default_tsv = default_dir / "stats" / "multidataset_workbench_subregion_vertex_stats.tsv"

    parser.add_argument("--out-png", default=None, help=f"Output PNG path (default: {default_png})")
    parser.add_argument("--out-tsv", default=None, help=f"Output TSV path (default: {default_tsv})")
    parser.add_argument("--smoothing", choices=["2mm", "4mm"], default="2mm", help="Smoothing resolution (default: 2mm)")
    parser.add_argument("--variant", choices=["36p"], default="36p", help="Preprocessing variant (default: 36p)")
    parser.add_argument(
        "--atlas",
        choices=["kong2022", "7net"],
        default="7net",
        help="Atlas / network taxonomy version. Default: 7net (canonical7, unsuffixed); kong2022 is an equivalent path alias.",
    )
    parser.add_argument("--grid", choices=list(GRID_VARIANTS), default=None, help="Grid variant shortcut")
    args = parser.parse_args()

    spec = GridSpec.from_tag(args.grid, atlas=args.atlas) if args.grid else GridSpec(args.smoothing, args.variant, atlas=args.atlas)

    if args.out_png:
        out_png = Path(args.out_png).resolve()
    else:
        output_tag = get_comparison_output_tag(spec) if args.version == COMPARISON_VERSION else spec.file_tag
        out_png = default_dir / f"multidataset_workbench_hippocampus_overlay_{output_tag}.png"

    if args.out_tsv:
        out_tsv = Path(args.out_tsv).resolve()
    else:
        stats_dir = default_dir / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        output_tag = get_comparison_output_tag(spec) if args.version == COMPARISON_VERSION else spec.file_tag
        out_tsv = stats_dir / f"multidataset_workbench_hippocampus_overlay_{output_tag}_vertex_stats.tsv"

    if args.version == COMPARISON_VERSION:
        render_spec = GridSpec(spec.smoothing, spec.variant, atlas="7net")
        comparison = get_comparison_visual_columns(render_spec)
        render_all_cohorts_overlay_figure(
            out_png,
            out_tsv,
            spec=render_spec,
            cohort_columns=comparison,
        )
    else:
        render_all_cohorts_overlay_figure(out_png, out_tsv, spec=spec)


if __name__ == "__main__":
    raise SystemExit(
        "Standalone rendering has been removed. Use "
        "scripts/visualize/run_multidataset_suite.py "
        "with --figures to select the figure type; both 36p grids are processed by default."
    )

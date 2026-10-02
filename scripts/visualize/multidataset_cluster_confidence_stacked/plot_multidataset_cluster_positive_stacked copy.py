#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
GLOBAL_NETWORK_STYLE_JSON = REPO_ROOT / "config" / "hipp_network_style.json"
DEFAULT_SCENE = REPO_ROOT / "config" / "wb_locked_native_view_lateral_medial.scene"

import sys
CORE_DIR = REPO_ROOT / "scripts" / "core_workflow"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from lib.surface_utils import render_locked_grid


# Load canonical network colors
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

DATASET_CONFIG = [
    {
        "dataset_name": "HCP 7T",
        "cohorts": [
            ("HCP 7T", 151, REPO_ROOT / "outputs_hcp/work/group_atlas/routea_full_k3_fisherz"),
        ],
        "bg_color": "#fafafa",
    },
    {
        "dataset_name": "Iowa 3T",
        "cohorts": [
            ("HC", 67, REPO_ROOT / "outputs_iowa_jj_tms/work/group_atlas/routea_hc_full_k3_fisherz"),
            ("Patient", 26, REPO_ROOT / "outputs_iowa_jj_tms/work/group_atlas/routea_patient_full_k3_fisherz"),
        ],
        "bg_color": "#f2f7fc",
    },
    {
        "dataset_name": "Stanford",
        "cohorts": [
            ("NTHC", 51, REPO_ROOT / "outputs_stanford/work/group_atlas/routea_nthc_full_k3_fisherz"),
            ("TEHC", 40, REPO_ROOT / "outputs_stanford/work/group_atlas/routea_tehc_full_k3_fisherz"),
            ("NTS",  56, REPO_ROOT / "outputs_stanford/work/group_atlas/routea_nts_full_k3_fisherz"),
            ("NTS Any MDD", 43, REPO_ROOT / "outputs_stanford/work/group_atlas/routea_nts_any_mdd_full_k3_fisherz"),
            ("NTS Pure MDD", 11, REPO_ROOT / "outputs_stanford/work/group_atlas/routea_nts_pure_mdd_full_k3_fisherz"),
            ("TIS",  47, REPO_ROOT / "outputs_stanford/work/group_atlas/routea_tis_full_k3_fisherz"),
            ("TIS PTSD", 35, REPO_ROOT / "outputs_stanford/work/group_atlas/routea_tis_ptsd_full_k3_fisherz"),
        ],
        "bg_color": "#fbf9f4",
    },
]


def build_tight_4view_composite(native_im: Image.Image) -> Image.Image:
    """Build a tight 4-surface composite ordered: [Left Front, Right Front, Left Back, Right Back]."""
    arr = np.array(native_im)

    def trim_b(img: Image.Image) -> Image.Image:
        a = np.array(img)
        fg = (a[..., :3].sum(axis=2) > 15)
        if not np.any(fg):
            return img
        rmin, rmax = np.where(np.any(fg, axis=1))[0][[0, -1]]
        cmin, cmax = np.where(np.any(fg, axis=0))[0][[0, -1]]
        return img.crop((cmin, rmin, cmax + 1, rmax + 1))

    # Top: Front (Lateral), Bottom: Back (Medial)
    v_l_front = trim_b(Image.fromarray(arr[0:512, 0:800]))
    v_r_front = trim_b(Image.fromarray(arr[0:512, 800:1600]))
    v_l_back  = trim_b(Image.fromarray(arr[512:1024, 0:800]))
    v_r_back  = trim_b(Image.fromarray(arr[512:1024, 800:1600]))

    target_h = max(v_l_front.height, v_r_front.height, v_l_back.height, v_r_back.height)

    # 4 views ordered: [Left Front, Right Front, Left Back, Right Back]
    views = [v_l_front, v_r_front, v_l_back, v_r_back]
    scaled_views = []
    for v in views:
        sc = target_h / float(v.height)
        sw = max(1, int(round(v.width * sc)))
        scaled_views.append(v.resize((sw, target_h), Image.Resampling.LANCZOS))

    gap_inner = 3    # tight gap within Front pair (L-Front / R-Front) and Back pair (L-Back / R-Back)
    gap_pair = 10    # separation between Front pair and Back pair
    pad = 1

    total_w = pad * 2 + scaled_views[0].width + gap_inner + scaled_views[1].width + gap_pair + scaled_views[2].width + gap_inner + scaled_views[3].width
    total_h = pad * 2 + target_h

    canvas = Image.new("RGBA", (total_w, total_h), (12, 16, 24, 255))
    x = pad
    canvas.paste(scaled_views[0], (x, pad), scaled_views[0] if scaled_views[0].mode == "RGBA" else None)
    x += scaled_views[0].width + gap_inner
    canvas.paste(scaled_views[1], (x, pad), scaled_views[1] if scaled_views[1].mode == "RGBA" else None)
    x += scaled_views[1].width + gap_pair
    canvas.paste(scaled_views[2], (x, pad), scaled_views[2] if scaled_views[2].mode == "RGBA" else None)
    x += scaled_views[2].width + gap_inner
    canvas.paste(scaled_views[3], (x, pad), scaled_views[3] if scaled_views[3].mode == "RGBA" else None)

    return canvas


def load_cohort_wb_overlay(atlas_dir: Path, temp_dir: Path, scene_path: Path) -> Image.Image:
    ml_path = atlas_dir / "gradient-spectral/kong2022/2mm/cov_0p80/g1g2g3/hemi_L/group_atlas_manifest.json"
    mr_path = atlas_dir / "gradient-spectral/kong2022/2mm/cov_0p80/g1g2g3/hemi_R/group_atlas_manifest.json"
    ml = json.loads(ml_path.read_text(encoding="utf-8"))
    mr = json.loads(mr_path.read_text(encoding="utf-8"))
    surf_l = Path(ml["template_left_or_right_surface"])
    surf_r = Path(mr["template_left_or_right_surface"])
    lbl_l_gii = Path(ml["assignment_variants"]["fisherz"]["template_functional_atlas_label_variant"])
    lbl_r_gii = Path(mr["assignment_variants"]["fisherz"]["template_functional_atlas_label_variant"])
    subject = str(ml.get("template_subject", "")).replace("sub-", "").strip()
    spec_path = Path(ml["template_spec_path"])

    cname_clean = atlas_dir.name.replace(" ", "_")
    render_out = temp_dir / f"wb_{cname_clean}"
    render = render_locked_grid(
        subject=subject,
        scene=scene_path,
        outdir=render_out,
        name=f"k3_2mm_{cname_clean}",
        title=cname_clean,
        left_labels=lbl_l_gii,
        right_labels=lbl_r_gii,
        left_surface=surf_l,
        right_surface=surf_r,
        spec_path=spec_path,
    )
    native_png = Path(render["native_scene_png"])
    native_im = Image.open(native_png).convert("RGBA")
    return build_tight_4view_composite(native_im)


def load_cohort_cluster_profiles(atlas_dir: Path, hemi: str) -> list[dict]:
    hpath = atlas_dir / f"gradient-spectral/kong2022/2mm/cov_0p80/g1g2g3/hemi_{hemi}"
    prof_path = hpath / "cluster_network_profile.npy"
    summary_path = hpath / "group_feature_summary.json"
    
    if prof_path.exists() and summary_path.exists():
        prof = np.load(prof_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        network_names = summary.get("network_names", [])
        
        out = []
        for cid in range(prof.shape[0]):
            row = prof[cid]
            pos_nets = []
            for i, net in enumerate(network_names):
                val = float(row[i])
                if val > 1e-5:
                    pos_nets.append((net, val))
            # Sort descending: top 1 is highest positive FC
            pos_nets.sort(key=lambda x: x[1], reverse=True)
            
            top1_net = pos_nets[0][0] if len(pos_nets) > 0 else "unassigned"
            top1_z = pos_nets[0][1] if len(pos_nets) > 0 else 0.0
            top2_net = pos_nets[1][0] if len(pos_nets) > 1 else "unassigned"
            top2_z = pos_nets[1][1] if len(pos_nets) > 1 else 0.0
            margin = top1_z - top2_z
            total_pos_z = sum(v for _, v in pos_nets)
            
            out.append({
                "cluster_id": cid + 1,
                "cluster_name": CLUSTER_NAMES.get(cid + 1, f"Cluster {cid+1}"),
                "pos_networks": pos_nets,
                "top1_network": top1_net,
                "top1_z": top1_z,
                "top2_network": top2_net,
                "top2_z": top2_z,
                "margin": margin,
                "total_pos_z": total_pos_z,
            })
        return out
    
    # Fallback to status json if profile npy is missing
    mf_path = hpath / "group_atlas_manifest.json"
    if not mf_path.exists():
        return []
    mf_data = json.loads(mf_path.read_text(encoding="utf-8"))
    status_rel = mf_data.get("cluster_assignment_status_fisherz", "cluster_assignment_status_fisherz.json")
    status_path = status_rel if Path(status_rel).is_absolute() else mf_path.parent / status_rel
    if not Path(status_path).exists():
        status_path = mf_path.parent / "cluster_assignment_status_fisherz.json"
    if not Path(status_path).exists():
        return []
    s_data = json.loads(Path(status_path).read_text(encoding="utf-8"))
    items = s_data.get("items", [])
    out = []
    for it in items:
        cid = int(it.get("cluster_id", 1))
        top1_net = str(it.get("top1_network", "unassigned"))
        top1_z = float(it.get("top1_fisher_z", it.get("top1_value", 0.0)))
        top2_net = str(it.get("top2_network", "unassigned"))
        top2_z = float(it.get("top2_fisher_z", it.get("top2_value", 0.0)))
        pos_nets = [(top1_net, top1_z)]
        if top2_z > 0 and top2_net not in {"unassigned", "none", ""}:
            pos_nets.append((top2_net, top2_z))
        out.append({
            "cluster_id": cid,
            "cluster_name": CLUSTER_NAMES.get(cid, f"Cluster {cid}"),
            "pos_networks": pos_nets,
            "top1_network": top1_net,
            "top1_z": top1_z,
            "top2_network": top2_net,
            "top2_z": top2_z,
            "margin": top1_z - top2_z,
            "total_pos_z": sum(v for _, v in pos_nets),
        })
    return sorted(out, key=lambda x: x["cluster_id"])


def build_plot_data(temp_dir: Path, scene_path: Path):
    plot_rows = []
    for ds_info in DATASET_CONFIG:
        ds_name = ds_info["dataset_name"]
        bg_col = ds_info["bg_color"]
        for cohort_name, n_sub, atlas_dir in ds_info["cohorts"]:
            left_status = load_cohort_cluster_profiles(atlas_dir, "L")
            right_status = load_cohort_cluster_profiles(atlas_dir, "R")
            wb_img = load_cohort_wb_overlay(atlas_dir, temp_dir, scene_path)
            plot_rows.append({
                "dataset_name": ds_name,
                "cohort_name": cohort_name,
                "n_subjects": n_sub,
                "cohort_label": f"{cohort_name}\n(N={n_sub})",
                "bg_color": bg_col,
                "left_clusters": left_status,
                "right_clusters": right_status,
                "wb_overlay": wb_img,
            })
    return plot_rows


def render_multidataset_cluster_positive_stacked(out_path: Path):
    scene_path = DEFAULT_SCENE
    with tempfile.TemporaryDirectory(prefix="wb_pos_stacked_") as tmp_dir:
        plot_rows = build_plot_data(Path(tmp_dir), scene_path)
        clusters_per_cohort = 3

        # Compact Layout Spacing: tightly pack the 7 cohorts to eliminate empty vertical gaps
        cluster_gap = 1.15
        cohort_gap = 1.05
        dataset_gap = 1.65
        
        current_y = 0.0
        bar_entries = []
        cohort_boundaries = []
        dataset_bands = {}

        for c_idx, crow in enumerate(plot_rows):
            start_y = current_y
            c_y_positions = []
            for cl_idx in range(clusters_per_cohort):
                y_pos = current_y
                c_y_positions.append(y_pos)
                
                l_data = crow["left_clusters"][cl_idx] if cl_idx < len(crow["left_clusters"]) else None
                r_data = crow["right_clusters"][cl_idx] if cl_idx < len(crow["right_clusters"]) else None
                cl_name = CLUSTER_NAMES.get(cl_idx + 1, f"C{cl_idx+1}")
                
                bar_entries.append({
                    "y": y_pos,
                    "cohort_idx": c_idx,
                    "cluster_idx": cl_idx,
                    "dataset_name": crow["dataset_name"],
                    "cohort_name": crow["cohort_name"],
                    "cohort_label": crow["cohort_label"],
                    "cluster_name": cl_name,
                    "bg_color": crow["bg_color"],
                    "left": l_data,
                    "right": r_data,
                })
                current_y += cluster_gap
            
            cohort_center = np.mean(c_y_positions)
            cohort_boundaries.append({
                "cohort_idx": c_idx,
                "center_y": cohort_center,
                "top_y": start_y - 0.50,
                "bottom_y": current_y - cluster_gap + 0.50,
                "cohort_label": crow["cohort_label"],
                "dataset_name": crow["dataset_name"],
                "bg_color": crow["bg_color"],
                "wb_overlay": crow["wb_overlay"],
            })
            
            ds = crow["dataset_name"]
            if ds not in dataset_bands:
                dataset_bands[ds] = {"min_y": start_y - 0.55, "max_y": current_y - cluster_gap + 0.55, "bg": crow["bg_color"]}
            else:
                dataset_bands[ds]["max_y"] = current_y - cluster_gap + 0.55
                
            is_last_in_dataset = (c_idx == len(plot_rows) - 1) or (plot_rows[c_idx+1]["dataset_name"] != ds)
            if is_last_in_dataset:
                current_y += dataset_gap
            else:
                current_y += cohort_gap

        # Figure layout with spacious left margin for dataset labels
        fig = plt.figure(figsize=(24.0, 22.0), dpi=300)
        
        grid_left = 0.175
        grid_right = 0.985
        # 3 columns: Left Hemi, Snug Middle Column (width_ratio=0.355 matching 4-surface tight aspect ratio, ZERO left/right blank space), Right Hemi
        gs = fig.add_gridspec(
            1, 3,
            width_ratios=[1.0, 0.355, 1.0],
            wspace=0.018,
            left=grid_left,
            right=grid_right,
            top=0.93,
            bottom=0.075
        )
        
        ax_left = fig.add_subplot(gs[0, 0])
        ax_mid = fig.add_subplot(gs[0, 1], sharey=ax_left)
        ax_right = fig.add_subplot(gs[0, 2], sharey=ax_left)
        
        used_networks = set()
        # Max limit set to 0.32 so bars completely fill the left and right width!
        max_xlim = 0.32
        
        # Background shading for datasets
        for ds_name, band in dataset_bands.items():
            for ax in (ax_left, ax_mid, ax_right):
                ax.axhspan(band["min_y"], band["max_y"], color=band["bg"], zorder=0, alpha=0.95)
                ax.axhline(band["max_y"], color="#cccccc", lw=1.1, ls="--", zorder=1)
                ax.axhline(band["min_y"], color="#cccccc", lw=1.1, ls="--", zorder=1)

        # Plot stacked bars (top network closest to midline x=0, network label directly on each bar)
        for entry in bar_entries:
            y = entry["y"]
            
            # --- LEFT HEMISPHERE (hemi-L) ---
            l_data = entry["left"]
            if l_data and l_data["pos_networks"]:
                cum_x = 0.0
                for rank_idx, (net_name, z_val) in enumerate(l_data["pos_networks"]):
                    used_networks.add(net_name)
                    c = COLOR_MAP.get(net_name, "#888888")
                    ax_left.barh(
                        y,
                        z_val,
                        left=cum_x,
                        height=0.88,
                        color=c,
                        alpha=0.92,
                        edgecolor="#222222",
                        lw=0.9,
                        zorder=3,
                    )
                    
                    # Network label on the bar segment
                    abbr = to_abbrev(net_name)
                    x_center = cum_x + z_val / 2.0
                    
                    if z_val >= 0.020:
                        font_size = 11.0
                    elif z_val >= 0.012:
                        font_size = 9.0
                    else:
                        font_size = None
                    
                    if font_size is not None:
                        ax_left.text(
                            x_center,
                            y,
                            abbr,
                            va="center",
                            ha="center",
                            fontsize=font_size,
                            fontweight="heavy",
                            color="#ffffff",
                            path_effects=[pe.withStroke(linewidth=2.0, foreground="#111111")],
                            zorder=5,
                        )
                    
                    cum_x += z_val

            # --- RIGHT HEMISPHERE (hemi-R) ---
            r_data = entry["right"]
            if r_data and r_data["pos_networks"]:
                cum_x = 0.0
                for rank_idx, (net_name, z_val) in enumerate(r_data["pos_networks"]):
                    used_networks.add(net_name)
                    c = COLOR_MAP.get(net_name, "#888888")
                    ax_right.barh(
                        y,
                        z_val,
                        left=cum_x,
                        height=0.88,
                        color=c,
                        alpha=0.92,
                        edgecolor="#222222",
                        lw=0.9,
                        zorder=3,
                    )
                    
                    # Network label on the bar segment
                    abbr = to_abbrev(net_name)
                    x_center = cum_x + z_val / 2.0
                    
                    if z_val >= 0.020:
                        font_size = 11.0
                    elif z_val >= 0.012:
                        font_size = 9.0
                    else:
                        font_size = None
                    
                    if font_size is not None:
                        ax_right.text(
                            x_center,
                            y,
                            abbr,
                            va="center",
                            ha="center",
                            fontsize=font_size,
                            fontweight="heavy",
                            color="#ffffff",
                            path_effects=[pe.withStroke(linewidth=2.0, foreground="#111111")],
                            zorder=5,
                        )
                    
                    cum_x += z_val

        # Invert and configure subplots
        y_min_bound = dataset_bands["HCP 7T"]["min_y"] - 0.35
        y_max_bound = dataset_bands["Stanford"]["max_y"] + 0.35
        ax_left.set_ylim(y_max_bound, y_min_bound)
        
        # Ticks every 0.05 up to 0.30
        x_ticks = [0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.00]
        
        # Left Subplot Settings
        ax_left.set_xlim(max_xlim, 0.0)
        ax_left.set_xticks(x_ticks)
        ax_left.set_title("Left Hippocampus (hemi-L)", fontsize=17.5, fontweight="heavy", pad=14, color="#0b2447")
        ax_left.set_xlabel("Cumulative Positive FC Strength (Fisher-z)", fontsize=13.5, fontweight="heavy", labelpad=10)
        ax_left.grid(True, axis="x", color="#e2e2e2", lw=0.7, ls=":", zorder=1)
        ax_left.tick_params(axis="x", labelsize=11.0)
        ax_left.tick_params(axis="y", left=False, labelleft=False)
        ax_left.spines[["top", "left"]].set_visible(False)
        ax_left.spines["right"].set_color("#222222")
        ax_left.spines["right"].set_linewidth(1.5)

        # Right Subplot Settings
        ax_right.set_xlim(0.0, max_xlim)
        ax_right.set_xticks(list(reversed(x_ticks)))
        ax_right.set_title("Right Hippocampus (hemi-R)", fontsize=17.5, fontweight="heavy", pad=14, color="#0b2447")
        ax_right.set_xlabel("Cumulative Positive FC Strength (Fisher-z)", fontsize=13.5, fontweight="heavy", labelpad=10)
        ax_right.grid(True, axis="x", color="#e2e2e2", lw=0.7, ls=":", zorder=1)
        ax_right.tick_params(axis="x", labelsize=11.0)
        ax_right.tick_params(axis="y", left=False, labelleft=False)
        ax_right.spines[["top", "right"]].set_visible(False)
        ax_right.spines["left"].set_color("#222222")
        ax_right.spines["left"].set_linewidth(1.5)

        # Center Subplot Settings: Display each cohort's 4-view tightly packed 3D Workbench overlay [Left-Front, Right-Front, Left-Back, Right-Back]
        ax_mid.axis("off")
        ax_mid.set_xlim(0.0, 1.0)
        ax_mid.set_title("2mm Atlas (Front: L/R | Back: L/R)", fontsize=15.0, fontweight="heavy", pad=14, color="#0b2447")
        
        for cb in cohort_boundaries:
            surf_img = cb.get("wb_overlay")
            if surf_img is not None:
                span_h = cb["bottom_y"] - cb["top_y"]
                ax_ins = ax_mid.inset_axes(
                    [0.00, cb["top_y"], 1.00, span_h],
                    transform=ax_mid.transData,
                )
                ax_ins.imshow(surf_img, aspect="equal")
                ax_ins.axis("off")

        # Add Cohort Labels on the Left of Left Subplot (spacious, clear distance from dataset label)
        for cb in cohort_boundaries:
            cy = cb["center_y"]
            clbl = cb["cohort_label"]
            norm_y = 1.0 - (cy - y_min_bound) / (y_max_bound - y_min_bound)
            
            ax_left.text(
                -0.035,
                norm_y,
                clbl,
                va="center",
                ha="right",
                fontsize=13.5,
                fontweight="heavy",
                color="#0b2447",
                transform=ax_left.transAxes,
            )
            ax_left.axhline(cb["bottom_y"] + cohort_gap/2.0, color="#dcdcdc", lw=0.8, ls=":", zorder=1)
            ax_right.axhline(cb["bottom_y"] + cohort_gap/2.0, color="#dcdcdc", lw=0.8, ls=":", zorder=1)

        # Add Dataset Bracket Badges on Far Left (original spacious position)
        for ds_name, band in dataset_bands.items():
            ds_y_center = (band["min_y"] + band["max_y"]) / 2.0
            norm_y_ds = 1.0 - (ds_y_center - y_min_bound) / (y_max_bound - y_min_bound)
            ax_left.text(
                -0.165,
                norm_y_ds,
                ds_name,
                va="center",
                ha="center",
                fontsize=15.0,
                fontweight="heavy",
                color="#0b3c5d",
                rotation=90,
                transform=ax_left.transAxes,
                bbox=dict(boxstyle="round,pad=0.45", fc="#ffffff", ec="#0b3c5d", lw=1.6, alpha=0.98),
            )

        # Suptitle
        visual_center_x = (grid_left + grid_right) / 2.0
        fig.suptitle(
            "Cross-Dataset Hippocampal Positive Functional Network Profiles (Stacked)",
            fontsize=22.0,
            fontweight="heavy",
            x=visual_center_x,
            y=0.982,
            ha="center",
            color="#0b2447",
        )

        # Legend at bottom with all used positive networks
        legend_patches = []
        canon_order = ["VentralAttention", "Somatomotor", "Default", "Auditory", "DorsalAttention", "Control", "Visual", "Language", "CinguloOpercular"]
        sorted_nets = [net for net in canon_order if net in used_networks] + [net for net in sorted(used_networks) if net not in canon_order]
        
        for net in sorted_nets:
            c = COLOR_MAP.get(net, "#888888")
            abbr = to_abbrev(net)
            legend_patches.append(
                Patch(facecolor=c, edgecolor="#111111", lw=0.9, label=f"{abbr} ({net})")
            )

        fig.legend(
            handles=legend_patches,
            loc="lower center",
            bbox_to_anchor=(visual_center_x, 0.012),
            ncol=len(legend_patches),
            fontsize=12.5,
            frameon=True,
            facecolor="#ffffff",
            edgecolor="#cccccc",
            title="Cortical Functional Networks (Kong2022 2mm)",
            title_fontsize=13.0,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[SUCCESS] Saved compact tight stacked cluster positive FC figure to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot compact multi-dataset cluster positive FC stacked profiles with 2mm 3D overlays.")
    default_png = REPO_ROOT / "scripts" / "visualize" / "multidataset_cluster_confidence_stacked" / "multidataset_cluster_confidence_stacked.png"
    parser.add_argument(
        "--out-png",
        default=str(default_png),
        help=f"Output PNG path (defaults to {default_png})",
    )
    args = parser.parse_args()

    render_multidataset_cluster_positive_stacked(Path(args.out_png).resolve())


if __name__ == "__main__":
    main()

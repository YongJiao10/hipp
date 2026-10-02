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
VIS_LIB_DIR = REPO_ROOT / "scripts" / "visualize" / "lib"
for p in (CORE_DIR, VIS_LIB_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lib.surface_utils import render_locked_grid
from lib.gradient_assignment_contract import validate_gradient_assignment
from multidataset_cohorts import GridSpec, get_multidataset_cohorts


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
    "Control": "#fdbf6f",
    "Language": "#b2df8a",
    "Visual": "#984ea3",
    "Limbic": "#ffff33",
    "CinguloOpercular": "#a6cee3",
}
NET_COLORS = {**DEFAULT_NET_COLORS, **COLOR_MAP}

ABBREV_MAP = {
    "Visual": "VIS",
    "Somatomotor": "SMN",
    "DorsalAttention": "DAN",
    "VentralAttention": "VAN",
    "Language": "LAN",
    "Control": "CON",
    "Default": "DMN",
    "Auditory": "AUD",
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


def build_dataset_config(spec: GridSpec = GridSpec("2mm", "36p")) -> list[dict]:
    col1, col2 = get_multidataset_cohorts(spec)
    all_cohorts = col1 + col2
    hcp_cohorts = [(c["cohort_name"], c["n_subjects"], c["group_root"]) for c in all_cohorts if c["dataset_key"] == "hcp"]
    iowa_cohorts = [(c["cohort_name"], c["n_subjects"], c["group_root"]) for c in all_cohorts if c["dataset_key"] == "iowa_jj_tms"]
    stan_cohorts = [(c["cohort_name"], c["n_subjects"], c["group_root"]) for c in all_cohorts if c["dataset_key"] == "stanford"]
    return [
        {"dataset_name": "HCP 7T", "cohorts": hcp_cohorts, "bg_color": "#fafafa"},
        {"dataset_name": "Iowa 3T", "cohorts": iowa_cohorts, "bg_color": "#f2f7fc"},
        {"dataset_name": "Stanford", "cohorts": stan_cohorts, "bg_color": "#fbf9f4"},
    ]


DATASET_CONFIG = build_dataset_config(GridSpec("2mm", "36p"))


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


SCRIPT_DIR = Path(__file__).resolve().parent
PANELS_DIR = SCRIPT_DIR / "panels"
PANELS_DIR.mkdir(parents=True, exist_ok=True)


def load_cohort_wb_overlay(
    atlas_dir: Path,
    temp_dir: Path,
    scene_path: Path,
    panels_dir: Path | None = None,
) -> Image.Image:
    cname_clean = atlas_dir.name.replace(" ", "_")
    if panels_dir is not None:
        cached_png = panels_dir / f"wb_{cname_clean}_2mm_tight4view.png"
        if cached_png.exists() and cached_png.stat().st_size > 5000:
            with Image.open(cached_png) as test_im:
                corners = [test_im.getpixel((0, 0))[:3], test_im.getpixel((test_im.width - 1, 0))[:3]]
                if all(sum(c) > 700 for c in corners):
                    return test_im.copy()
    else:
        cached_png = None

    ml_path = atlas_dir / "gradient-spectral/kong2022/2mm/cov_0p80/g1g2g3/hemi_L/group_atlas_manifest.json"
    mr_path = atlas_dir / "gradient-spectral/kong2022/2mm/cov_0p80/g1g2g3/hemi_R/group_atlas_manifest.json"
    ml = json.loads(ml_path.read_text(encoding="utf-8"))
    mr = json.loads(mr_path.read_text(encoding="utf-8"))
    left_assignment = validate_gradient_assignment(ml_path.parent, ml, "fisherz")
    right_assignment = validate_gradient_assignment(mr_path.parent, mr, "fisherz")
    surf_l = Path(ml["template_left_or_right_surface"])
    surf_r = Path(mr["template_left_or_right_surface"])
    lbl_l_gii = Path(left_assignment["label_path"])
    lbl_r_gii = Path(right_assignment["label_path"])
    subject = str(ml.get("template_subject", "")).replace("sub-", "").strip()
    spec_path = Path(ml["template_spec_path"])

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
    comp = build_tight_4view_composite(native_im)
    if cached_png is not None:
        panels_dir.mkdir(parents=True, exist_ok=True)
        comp.save(cached_png)
    return comp


def load_cohort_cluster_neg_profiles(atlas_dir: Path, hemi: str) -> list[dict]:
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
            neg_nets = []
            for i, net in enumerate(network_names):
                val = float(row[i])
                if val < -1e-5:
                    neg_nets.append((net, val, abs(val)))
            # Sort descending by magnitude (most negative FC first, closest to midline x=0)
            neg_nets.sort(key=lambda x: x[2], reverse=True)

            top1_neg_net = neg_nets[0][0] if len(neg_nets) > 0 else "none"
            top1_neg_z = neg_nets[0][1] if len(neg_nets) > 0 else 0.0
            total_neg_mag = sum(m for _, _, m in neg_nets)

            out.append({
                "cluster_id": cid + 1,
                "cluster_name": CLUSTER_NAMES.get(cid + 1, f"Cluster {cid+1}"),
                "neg_networks": neg_nets,
                "top1_neg_network": top1_neg_net,
                "top1_neg_z": top1_neg_z,
                "total_neg_mag": total_neg_mag,
            })
        return out

    return []


def build_neg_plot_data(temp_dir: Path, scene_path: Path, panels_dir: Path | None = None) -> list[dict]:
    plot_rows = []
    for ds_info in DATASET_CONFIG:
        ds_name = ds_info["dataset_name"]
        bg_col = ds_info["bg_color"]
        for cohort_name, n_sub, atlas_dir in ds_info["cohorts"]:
            left_status = load_cohort_cluster_neg_profiles(atlas_dir, "L")
            right_status = load_cohort_cluster_neg_profiles(atlas_dir, "R")
            wb_img = load_cohort_wb_overlay(atlas_dir, temp_dir, scene_path, panels_dir=panels_dir)
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


def layout_column_data(cohort_list: list[dict], custom_gaps: list[float] | None = None):
    """Compute vertical coordinates, bar entries, cohort boundaries, and dataset bands for 1 column."""
    clusters_per_cohort = 3
    cluster_gap = 1.15
    default_cohort_gap = 1.05
    default_dataset_gap = 1.65

    current_y = 0.0
    bar_entries = []
    cohort_boundaries = []
    dataset_bands = {}

    for c_idx, crow in enumerate(cohort_list):
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

        cohort_center = float(np.mean(c_y_positions))
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

        # Only add gap if NOT the last cohort
        if c_idx < len(cohort_list) - 1:
            if custom_gaps is not None and c_idx < len(custom_gaps):
                current_y += custom_gaps[c_idx]
            else:
                is_last_in_dataset = (cohort_list[c_idx+1]["dataset_name"] != ds)
                if is_last_in_dataset:
                    current_y += default_dataset_gap
                else:
                    current_y += default_cohort_gap

    return bar_entries, cohort_boundaries, dataset_bands, current_y


def render_column_axes(
    fig: plt.Figure,
    gs_col,
    bar_entries: list[dict],
    cohort_boundaries: list[dict],
    dataset_bands: dict[str, dict],
    y_min_bound: float,
    y_max_bound: float,
    used_networks: set[str],
    badge_x: float = -0.250,
    cohort_gap: float = 1.05,
):
    ax_left = fig.add_subplot(gs_col[0])
    ax_mid = fig.add_subplot(gs_col[1], sharey=ax_left)
    ax_right = fig.add_subplot(gs_col[2], sharey=ax_left)

    max_xlim = 0.32

    # Background shading for datasets
    for ds_name, band in dataset_bands.items():
        for ax in (ax_left, ax_mid, ax_right):
            ax.axhspan(band["min_y"], band["max_y"], color=band["bg"], zorder=0, alpha=0.95)
            ax.axhline(band["max_y"], color="#cccccc", lw=1.1, ls="--", zorder=1)
            ax.axhline(band["min_y"], color="#cccccc", lw=1.1, ls="--", zorder=1)

    # Plot stacked bars for negative connections (ordered by absolute magnitude)
    for entry in bar_entries:
        y = entry["y"]

        # --- LEFT HEMISPHERE (hemi-L) ---
        l_data = entry["left"]
        if l_data and l_data["neg_networks"]:
            cum_x = 0.0
            for rank_idx, (net_name, raw_z, mag_z) in enumerate(l_data["neg_networks"]):
                used_networks.add(net_name)
                c = COLOR_MAP.get(net_name, "#888888")
                ax_left.barh(
                    y,
                    mag_z,
                    left=cum_x,
                    height=0.88,
                    color=c,
                    alpha=0.92,
                    edgecolor="#222222",
                    lw=0.9,
                    zorder=3,
                )

                abbr = to_abbrev(net_name)
                x_center = cum_x + mag_z / 2.0

                if mag_z >= 0.020:
                    font_size = 11.0
                elif mag_z >= 0.012:
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

                cum_x += mag_z

        # --- RIGHT HEMISPHERE (hemi-R) ---
        r_data = entry["right"]
        if r_data and r_data["neg_networks"]:
            cum_x = 0.0
            for rank_idx, (net_name, raw_z, mag_z) in enumerate(r_data["neg_networks"]):
                used_networks.add(net_name)
                c = COLOR_MAP.get(net_name, "#888888")
                ax_right.barh(
                    y,
                    mag_z,
                    left=cum_x,
                    height=0.88,
                    color=c,
                    alpha=0.92,
                    edgecolor="#222222",
                    lw=0.9,
                    zorder=3,
                )

                abbr = to_abbrev(net_name)
                x_center = cum_x + mag_z / 2.0

                if mag_z >= 0.020:
                    font_size = 11.0
                elif mag_z >= 0.012:
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

                cum_x += mag_z

    # Invert and configure subplots
    ax_left.set_ylim(y_max_bound, y_min_bound)

    # Ticks every 0.05 up to 0.30
    x_ticks = [0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.00]

    # Left Subplot Settings
    ax_left.set_xlim(max_xlim, 0.0)
    ax_left.set_xticks(x_ticks)
    ax_left.set_title("Left Hippocampus (hemi-L)", fontsize=17.5, fontweight="heavy", pad=14, color="#0b2447")
    ax_left.set_xlabel("Cumulative Negative FC Strength (|Fisher-z|)", fontsize=13.5, fontweight="heavy", labelpad=10)
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
    ax_right.set_xlabel("Cumulative Negative FC Strength (|Fisher-z|)", fontsize=13.5, fontweight="heavy", labelpad=10)
    ax_right.grid(True, axis="x", color="#e2e2e2", lw=0.7, ls=":", zorder=1)
    ax_right.tick_params(axis="x", labelsize=11.0)
    ax_right.tick_params(axis="y", left=False, labelleft=False)
    ax_right.spines[["top", "right"]].set_visible(False)
    ax_right.spines["left"].set_color("#222222")
    ax_right.spines["left"].set_linewidth(1.5)

    # Center Subplot Settings: Display each cohort's 4-view 3D Workbench overlay with natural aspect="equal"
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

    # Cohort Labels on the Left of Left Subplot (enlarged font size)
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
            fontsize=16.0,
            fontweight="heavy",
            color="#0b2447",
            transform=ax_left.transAxes,
        )
        ax_left.axhline(cb["bottom_y"] + cohort_gap / 2.0, color="#dcdcdc", lw=0.8, ls=":", zorder=1)
        ax_right.axhline(cb["bottom_y"] + cohort_gap / 2.0, color="#dcdcdc", lw=0.8, ls=":", zorder=1)

    # Dataset Bracket Badges on the Left
    for ds_name, band in dataset_bands.items():
        ds_y_center = (band["min_y"] + band["max_y"]) / 2.0
        norm_y_ds = 1.0 - (ds_y_center - y_min_bound) / (y_max_bound - y_min_bound)
        ax_left.text(
            badge_x,
            norm_y_ds,
            ds_name,
            va="center",
            ha="center",
            fontsize=16.5,
            fontweight="heavy",
            color="#0b3c5d",
            rotation=90,
            transform=ax_left.transAxes,
            bbox=dict(boxstyle="round,pad=0.45", fc="#ffffff", ec="#0b3c5d", lw=1.6, alpha=0.98),
        )


def render_multidataset_cluster_negative_stacked(out_path: Path):
    scene_path = DEFAULT_SCENE
    with tempfile.TemporaryDirectory(prefix="wb_neg_stacked_") as tmp_dir:
        plot_rows = build_neg_plot_data(Path(tmp_dir), scene_path, panels_dir=PANELS_DIR)

        # 10 Cohorts split into 5 rows x 2 columns
        col1_rows = plot_rows[:5]
        col2_rows = plot_rows[5:]

        # Col 1 has 2 dataset transitions (gap=1.65) and 2 cohort transitions (gap=1.05) -> total span = 22.00
        col1_gaps = [1.65, 1.05, 1.65, 1.05]
        bar_entries_1, cohort_boundaries_1, dataset_bands_1, total_y_1 = layout_column_data(col1_rows, custom_gaps=col1_gaps)

        # Col 2 has 4 cohort transitions uniformly spaced at 1.35 to perfectly match Col 1's vertical span (22.00)
        col2_uniform_gaps = [1.35, 1.35, 1.35, 1.35]
        bar_entries_2, cohort_boundaries_2, dataset_bands_2, total_y_2 = layout_column_data(col2_rows, custom_gaps=col2_uniform_gaps)

        # Both columns now have exact matching top (-0.50) and bottom (22.00)
        # Symmetrical breathing room on top and bottom (0.60 units) to guarantee clean, tight margins without any clipping
        y_min_bound = -0.50 - 0.60  # -1.10
        y_max_bound = 22.00 + 0.60  # 22.60

        # Compact landscape layout (38.0 x 13.0 inches)
        fig = plt.figure(figsize=(38.0, 13.0), dpi=300)

        # width_ratios=[1.0, 0.430, 1.0] and wspace=0.005 ensures the 4-view 3D surface sits snug against the left and right y-axes with NO empty gaps
        gs_col1 = fig.add_gridspec(
            1, 3,
            left=0.065,
            right=0.487,
            top=0.905,
            bottom=0.115,
            width_ratios=[1.0, 0.430, 1.0],
            wspace=0.005,
        )

        gs_col2 = fig.add_gridspec(
            1, 3,
            left=0.566,
            right=0.988,
            top=0.905,
            bottom=0.115,
            width_ratios=[1.0, 0.430, 1.0],
            wspace=0.005,
        )

        used_networks = set()

        # Col 1: closer badge_x = -0.250 for shorter labels (brings dataset badges closer to group labels)
        render_column_axes(
            fig=fig,
            gs_col=gs_col1,
            bar_entries=bar_entries_1,
            cohort_boundaries=cohort_boundaries_1,
            dataset_bands=dataset_bands_1,
            y_min_bound=y_min_bound,
            y_max_bound=y_max_bound,
            used_networks=used_networks,
            badge_x=-0.250,
        )

        # Col 2: badge_x = -0.350 comfortable distance for longer labels like NTS Pure MDD
        render_column_axes(
            fig=fig,
            gs_col=gs_col2,
            bar_entries=bar_entries_2,
            cohort_boundaries=cohort_boundaries_2,
            dataset_bands=dataset_bands_2,
            y_min_bound=y_min_bound,
            y_max_bound=y_max_bound,
            used_networks=used_networks,
            badge_x=-0.350,
        )

        # Suptitle
        fig.suptitle(
            "Cross-Dataset Hippocampal Negative Functional Network Profiles (Stacked)",
            fontsize=23.0,
            fontweight="heavy",
            x=0.50,
            y=0.975,
            ha="center",
            color="#0b2447",
        )

        # Global legend at bottom with all used negative networks
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
            bbox_to_anchor=(0.50, 0.015),
            ncol=len(legend_patches),
            fontsize=13.0,
            frameon=True,
            facecolor="#ffffff",
            edgecolor="#cccccc",
            title="Anticorrelated Cortical Networks (Kong2022 2mm)",
            title_fontsize=13.5,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[SUCCESS] Saved compact tight stacked cluster negative FC figure (5 rows x 2 cols) to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot compact multi-dataset cluster negative FC stacked profiles with 2mm 3D overlays in 5 rows x 2 columns.")
    default_png = REPO_ROOT / "scripts" / "visualize" / "multidataset_cluster_confidence_negative_stacked" / "multidataset_cluster_confidence_negative_stacked.png"
    parser.add_argument(
        "--out-png",
        default=str(default_png),
        help=f"Output PNG path (defaults to {default_png})",
    )
    args = parser.parse_args()

    render_multidataset_cluster_negative_stacked(Path(args.out_png).resolve())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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

from lib.gradient_assignment_contract import validate_gradient_assignment
from lib.surface_utils import render_locked_grid
from multidataset_cohorts import (
    COMPARISON_METHODS,
    COMPARISON_VERSION,
    GRID_VARIANTS,
    GridSpec,
    comparison_dataset_style,
    format_comparison_column_title,
    get_comparison_output_tag,
    get_comparison_visual_columns,
    get_multidataset_cohorts,
)
from surface_panel_cache import cache_matches, panel_inputs, publish_cache


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

FIGURE_TITLE = "Cross-Dataset Hippocampal Positive Functional Network Profiles"


def format_column_atlas_title(method_name: str, smoothing: str) -> str:
    return format_comparison_column_title(
        method_name, f"{smoothing} Atlas (Front: L/R | Back: L/R)"
    )


def build_dataset_config(
    spec: GridSpec = GridSpec("2mm", "36p"),
    cohort_columns: tuple[list[dict], list[dict]] | None = None,
) -> list[dict]:
    if cohort_columns is not None:
        return [
            {
                "dataset_name": entries[0]["method_name"],
                "cohorts": [
                    (
                        entry["cohort_name"],
                        entry["n_subjects"],
                        Path(entry["group_root"]),
                        entry,
                    )
                    for entry in entries
                ],
                "bg_color": entries[0]["bg_color"],
            }
            for entries in cohort_columns
        ]

    col1, col2 = get_multidataset_cohorts(spec)
    all_cohorts = col1 + col2
    hcp_cohorts = [
        (c["cohort_name"], c["n_subjects"], c["group_root"])
        for c in all_cohorts
        if c["dataset_key"] == "hcp"
    ]
    iowa_cohorts = [
        (c["cohort_name"], c["n_subjects"], c["group_root"])
        for c in all_cohorts
        if c["dataset_key"] == "iowa_jj_tms"
    ]
    stan_cohorts = [
        (c["cohort_name"], c["n_subjects"], c["group_root"])
        for c in all_cohorts
        if c["dataset_key"] == "stanford"
    ]
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
            fg = a[..., :3].sum(axis=2) > 15
        if not np.any(fg):
            return img
        rmin, rmax = np.where(np.any(fg, axis=1))[0][[0, -1]]
        cmin, cmax = np.where(np.any(fg, axis=0))[0][[0, -1]]
        return img.crop((cmin, rmin, cmax + 1, rmax + 1))

    # Top: Front (Lateral), Bottom: Back (Medial)
    v_l_front = trim_b(Image.fromarray(arr[0:512, 0:800]))
    v_r_front = trim_b(Image.fromarray(arr[0:512, 800:1600]))
    v_l_back = trim_b(Image.fromarray(arr[512:1024, 0:800]))
    v_r_back = trim_b(Image.fromarray(arr[512:1024, 800:1600]))

    target_h = max(v_l_front.height, v_r_front.height, v_l_back.height, v_r_back.height)

    # 4 views ordered: [Left Front, Right Front, Left Back, Right Back]
    views = [v_l_front, v_r_front, v_l_back, v_r_back]
    scaled_views = []
    for v in views:
        sc = target_h / float(v.height)
        sw = max(1, int(round(v.width * sc)))
        scaled_views.append(v.resize((sw, target_h), Image.Resampling.LANCZOS))

    gap_inner = 3  # tight gap within Front pair (L-Front / R-Front) and Back pair (L-Back / R-Back)
    gap_pair = 10  # separation between Front pair and Back pair
    pad = 1

    total_w = (
        pad * 2
        + scaled_views[0].width
        + gap_inner
        + scaled_views[1].width
        + gap_pair
        + scaled_views[2].width
        + gap_inner
        + scaled_views[3].width
    )
    total_h = pad * 2 + target_h

    canvas = Image.new("RGBA", (total_w, total_h), (255, 255, 255, 255))
    x = pad
    canvas.paste(
        scaled_views[0],
        (x, pad),
        scaled_views[0] if scaled_views[0].mode == "RGBA" else None,
    )
    x += scaled_views[0].width + gap_inner
    canvas.paste(
        scaled_views[1],
        (x, pad),
        scaled_views[1] if scaled_views[1].mode == "RGBA" else None,
    )
    x += scaled_views[1].width + gap_pair
    canvas.paste(
        scaled_views[2],
        (x, pad),
        scaled_views[2] if scaled_views[2].mode == "RGBA" else None,
    )
    x += scaled_views[2].width + gap_inner
    canvas.paste(
        scaled_views[3],
        (x, pad),
        scaled_views[3] if scaled_views[3].mode == "RGBA" else None,
    )

    return canvas


def load_cohort_wb_overlay(
    atlas_dir: Path,
    temp_dir: Path,
    scene_path: Path,
    smoothing: str = "2mm",
    variant: str = "36p",
    signal_branch: str | None = None,
    atlas: str = "7net",
    panels_dir: Path | None = None,
    grad_subpath: Path | None = None,
    cache_tag: str | None = None,
    method: str | None = None,
) -> Image.Image:
    _validate_signal_branch_value(signal_branch)
    cname_clean = atlas_dir.name.replace(" ", "_")
    cache_prefix = f"{cache_tag}_" if cache_tag else ""
    atlas_folder = "kong2022"
    branch_suffix = f"_{signal_branch}" if signal_branch is not None else ""
    file_tag = f"{smoothing}_{variant}{branch_suffix}"
    if panels_dir is not None:
        cached_png = (
            panels_dir / f"wb_{cache_prefix}{cname_clean}_{file_tag}_tight4view.png"
        )
    else:
        cached_png = None

    grad_dir = (
        Path(grad_subpath)
        if grad_subpath is not None
        else atlas_dir / f"gradient-spectral/{atlas_folder}/{smoothing}/cov_0p80/g1g2g3"
    )
    ml_path = grad_dir / "hemi_L/group_atlas_manifest.json"
    mr_path = grad_dir / "hemi_R/group_atlas_manifest.json"
    ml = json.loads(ml_path.read_text(encoding="utf-8"))
    mr = json.loads(mr_path.read_text(encoding="utf-8"))
    _validate_manifest_signal_branch(
        ml_path, ml, signal_branch, method=method, provenance_path=grad_dir
    )
    _validate_manifest_signal_branch(
        mr_path, mr, signal_branch, method=method, provenance_path=grad_dir
    )
    left_assignment = validate_gradient_assignment(ml_path.parent, ml, "fisherz")
    right_assignment = validate_gradient_assignment(mr_path.parent, mr, "fisherz")
    surf_l = _declared_file_value(ml_path, ml, "template_left_or_right_surface")
    surf_r = _declared_file_value(mr_path, mr, "template_left_or_right_surface")
    lbl_l_gii = left_assignment["label_path"]
    lbl_r_gii = right_assignment["label_path"]
    subject = _manifest_subject(ml_path, ml)
    right_subject = _manifest_subject(mr_path, mr)
    if subject != right_subject:
        raise ValueError(
            f"{grad_dir} has mismatched template_subject values: "
            f"hemi_L={subject!r}, hemi_R={right_subject!r}"
        )
    spec_path = _declared_file_value(ml_path, ml, "template_spec_path")
    right_spec_path = _declared_file_value(mr_path, mr, "template_spec_path")

    selection = dict(
        smoothing=smoothing, variant=variant, atlas=atlas, cache_tag=cache_tag
    )
    if signal_branch is not None:
        selection.update(signal_branch=signal_branch, subjects=[subject])
    inputs = panel_inputs(
        [
            ml_path,
            mr_path,
            surf_l,
            surf_r,
            lbl_l_gii,
            lbl_r_gii,
            *left_assignment["paths"],
            *right_assignment["paths"],
            spec_path,
            right_spec_path,
            scene_path,
            Path(__file__),
            CORE_DIR / "lib/surface_utils.py",
        ],
        **selection,
    )
    if cache_matches(cached_png, inputs):
        return Image.open(cached_png)

    render_out = temp_dir / f"wb_{cache_prefix}{cname_clean}_{file_tag}"
    render = render_locked_grid(
        subject=subject,
        scene=scene_path,
        outdir=render_out,
        name=f"k3_{cache_prefix}{file_tag}_{cname_clean}",
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
        publish_cache(cached_png, inputs)
    return comp


def load_cohort_cluster_profiles(
    atlas_dir: Path,
    hemi: str,
    smoothing: str = "2mm",
    atlas: str = "7net",
    grad_subpath: Path | None = None,
    signal_branch: str | None = None,
    method: str | None = None,
) -> list[dict]:
    atlas_folder = "kong2022"
    grad_dir = (
        Path(grad_subpath)
        if grad_subpath is not None
        else atlas_dir / f"gradient-spectral/{atlas_folder}/{smoothing}/cov_0p80/g1g2g3"
    )
    hpath = grad_dir / f"hemi_{hemi}"
    mf_path = hpath / "group_atlas_manifest.json"
    if not mf_path.is_file():
        raise FileNotFoundError(
            f"Missing canonical manifest for positive cluster profile: {mf_path}"
        )
    mf_data = json.loads(mf_path.read_text(encoding="utf-8"))
    _validate_manifest_signal_branch(
        mf_path,
        mf_data,
        signal_branch,
        method=method,
        provenance_path=grad_dir,
    )
    prof, network_names, _prof_path, _summary_path = _load_positive_profile_contract(
        mf_path, mf_data
    )
    out = []
    for cid, row in enumerate(prof, start=1):
        pos_nets = [
            (net, float(row[i]))
            for i, net in enumerate(network_names)
            if float(row[i]) > 1e-5
        ]
        pos_nets.sort(key=lambda x: x[1], reverse=True)
        top1_net = pos_nets[0][0] if pos_nets else "unassigned"
        top1_z = pos_nets[0][1] if pos_nets else 0.0
        top2_net = pos_nets[1][0] if len(pos_nets) > 1 else "unassigned"
        top2_z = pos_nets[1][1] if len(pos_nets) > 1 else 0.0
        out.append(
            {
                "cluster_id": cid,
                "cluster_name": CLUSTER_NAMES.get(cid, f"Cluster {cid}"),
                "pos_networks": pos_nets,
                "top1_network": top1_net,
                "top1_z": top1_z,
                "top2_network": top2_net,
                "top2_z": top2_z,
                "margin": top1_z - top2_z,
                "total_pos_z": sum(v for _, v in pos_nets),
            }
        )
    return out
    # Historical status-only loading is retained below as inert provenance text.
    """

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
    """


def build_plot_data(
    temp_dir: Path,
    scene_path: Path,
    spec: GridSpec = GridSpec("2mm", "36p"),
    cohort_columns: tuple[list[dict], list[dict]] | None = None,
    signal_branch: str | None = None,
) -> list[dict]:
    _validate_signal_branch_value(signal_branch)
    plot_rows = []
    dataset_config = build_dataset_config(spec, cohort_columns=cohort_columns)
    panels_dir = Path(__file__).resolve().parent / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    for ds_info in dataset_config:
        ds_name = ds_info["dataset_name"]
        bg_col = ds_info["bg_color"]
        for cohort_row in ds_info["cohorts"]:
            row_dataset_name = ds_name
            row_bg_col = bg_col
            column_title = None
            if len(cohort_row) == 4:
                cohort_name, n_sub, atlas_dir, comparison_entry = cohort_row
                grad_subpath = Path(comparison_entry["grad_subpath"])
                entry_branch = comparison_entry.get("signal_branch")
                if (
                    entry_branch is not None
                    and signal_branch is not None
                    and entry_branch != signal_branch
                ):
                    raise ValueError(
                        f"comparison entry signal_branch={entry_branch!r} conflicts with "
                        f"requested {signal_branch!r}: {comparison_entry['id']}"
                    )
                active_branch = (
                    signal_branch if signal_branch is not None else entry_branch
                )
                method = comparison_entry.get("method")
                row_dataset_name, row_bg_col = comparison_dataset_style(
                    comparison_entry
                )
                column_title = comparison_entry["method_name"]
                # Keep comparison caches separate from historical 7-net panels;
                # the direct branch can refresh group labels without changing
                # the locked Workbench renderer or its layout.
                cache_tag = f"{comparison_entry['method']}_direct"
            else:
                cohort_name, n_sub, atlas_dir = cohort_row
                comparison_entry = None
                grad_subpath = None
                cache_tag = None
                active_branch = signal_branch
                method = None
            left_status = load_cohort_cluster_profiles(
                atlas_dir,
                "L",
                smoothing=spec.smoothing,
                atlas=spec.atlas,
                grad_subpath=grad_subpath,
                signal_branch=active_branch,
                method=method,
            )
            right_status = load_cohort_cluster_profiles(
                atlas_dir,
                "R",
                smoothing=spec.smoothing,
                atlas=spec.atlas,
                grad_subpath=grad_subpath,
                signal_branch=active_branch,
                method=method,
            )
            wb_img = load_cohort_wb_overlay(
                atlas_dir,
                temp_dir,
                scene_path,
                smoothing=spec.smoothing,
                variant=spec.variant,
                signal_branch=active_branch,
                atlas=spec.atlas,
                panels_dir=panels_dir,
                grad_subpath=grad_subpath,
                cache_tag=cache_tag,
                method=method,
            )
            plot_rows.append(
                {
                    "dataset_name": row_dataset_name,
                    "cohort_name": cohort_name,
                    "n_subjects": n_sub,
                    "cohort_label": f"{cohort_name}\n(N={n_sub})",
                    "bg_color": row_bg_col,
                    "column_title": column_title,
                    "left_clusters": left_status,
                    "right_clusters": right_status,
                    "wb_overlay": wb_img,
                }
            )
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

            l_data = (
                crow["left_clusters"][cl_idx]
                if cl_idx < len(crow["left_clusters"])
                else None
            )
            r_data = (
                crow["right_clusters"][cl_idx]
                if cl_idx < len(crow["right_clusters"])
                else None
            )
            cl_name = CLUSTER_NAMES.get(cl_idx + 1, f"C{cl_idx + 1}")

            bar_entries.append(
                {
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
                }
            )
            current_y += cluster_gap

        cohort_center = float(np.mean(c_y_positions))
        cohort_boundaries.append(
            {
                "cohort_idx": c_idx,
                "center_y": cohort_center,
                "top_y": start_y - 0.50,
                "bottom_y": current_y - cluster_gap + 0.50,
                "cohort_label": crow["cohort_label"],
                "dataset_name": crow["dataset_name"],
                "bg_color": crow["bg_color"],
                "wb_overlay": crow["wb_overlay"],
            }
        )

        ds = crow["dataset_name"]
        if ds not in dataset_bands:
            dataset_bands[ds] = {
                "min_y": start_y - 0.55,
                "max_y": current_y - cluster_gap + 0.55,
                "bg": crow["bg_color"],
            }
        else:
            dataset_bands[ds]["max_y"] = current_y - cluster_gap + 0.55

        # Only add gap if NOT the last cohort
        if c_idx < len(cohort_list) - 1:
            if custom_gaps is not None and c_idx < len(custom_gaps):
                current_y += custom_gaps[c_idx]
            else:
                is_last_in_dataset = cohort_list[c_idx + 1]["dataset_name"] != ds
                if is_last_in_dataset:
                    current_y += default_dataset_gap
                else:
                    current_y += default_cohort_gap

    return bar_entries, cohort_boundaries, dataset_bands, current_y


def adaptive_positive_axis_scale(
    bar_entries: list[dict],
    *,
    padding_fraction: float = 0.08,
    target_intervals: int = 6,
) -> tuple[float, list[float]]:
    """Return a shared positive-FC x-range and descending tick positions.

    The left and right panels retain a common scale, but the scale is derived
    from the largest stacked positive profile in the current column.  Rounded
    tick steps keep labels readable while the small fractional padding avoids
    both clipping and the large empty margins caused by the old fixed ranges.
    """
    max_stack = 0.0
    for entry in bar_entries:
        for side in ("left", "right"):
            profile = entry.get(side) or {}
            stack_total = 0.0
            for _network, raw_value in profile.get("pos_networks", []):
                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value) and value > 0.0:
                    stack_total += value
            max_stack = max(max_stack, stack_total)

    if max_stack <= 0.0:
        max_xlim = 0.10
        step = 0.02
    else:
        padded_max = max_stack * (1.0 + padding_fraction)
        raw_step = padded_max / float(target_intervals)
        magnitude = 10.0 ** math.floor(math.log10(raw_step))
        normalized_step = raw_step / magnitude
        if normalized_step <= 1.0:
            nice_factor = 1.0
        elif normalized_step <= 2.0:
            nice_factor = 2.0
        elif normalized_step <= 2.5:
            nice_factor = 2.5
        elif normalized_step <= 5.0:
            nice_factor = 5.0
        else:
            nice_factor = 10.0
        step = nice_factor * magnitude
        max_xlim = math.ceil((padded_max / step) - 1e-12) * step
        if max_xlim > max_stack * 1.25:
            step *= 0.5
            max_xlim = math.ceil((padded_max / step) - 1e-12) * step

    decimal_places = max(2, int(math.ceil(-math.log10(step))) + 2) if step < 1.0 else 2
    max_xlim = round(max_xlim, decimal_places)
    n_intervals = int(round(max_xlim / step))
    x_ticks = [
        round(max_xlim - index * step, decimal_places)
        for index in range(n_intervals + 1)
    ]
    return max_xlim, x_ticks


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
    smoothing: str = "2mm",
    variant: str = "36p",
    column_title: str | None = None,
):
    ax_left = fig.add_subplot(gs_col[0])
    ax_mid = fig.add_subplot(gs_col[1], sharey=ax_left)
    ax_right = fig.add_subplot(gs_col[2], sharey=ax_left)

    max_xlim, x_ticks = adaptive_positive_axis_scale(bar_entries)

    # Background shading for datasets
    for ds_name, band in dataset_bands.items():
        for ax in (ax_left, ax_mid, ax_right):
            ax.axhspan(
                band["min_y"], band["max_y"], color=band["bg"], zorder=0, alpha=0.95
            )
            ax.axhline(band["max_y"], color="#cccccc", lw=1.1, ls="--", zorder=1)
            ax.axhline(band["min_y"], color="#cccccc", lw=1.1, ls="--", zorder=1)

    # Plot stacked bars
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

                abbr = to_abbrev(net_name)
                x_center = cum_x + z_val / 2.0

                if z_val >= 0.020:
                    font_size = 11.0
                    ax_left.text(
                        x_center,
                        y,
                        abbr,
                        va="center",
                        ha="center",
                        fontsize=font_size,
                        fontweight="bold",
                        color="#ffffff",
                        path_effects=[
                            pe.withStroke(linewidth=2.0, foreground="#111111")
                        ],
                        zorder=5,
                        clip_on=True,
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

                abbr = to_abbrev(net_name)
                x_center = cum_x + z_val / 2.0

                if z_val >= 0.020:
                    font_size = 11.0
                    ax_right.text(
                        x_center,
                        y,
                        abbr,
                        va="center",
                        ha="center",
                        fontsize=font_size,
                        fontweight="bold",
                        color="#ffffff",
                        path_effects=[
                            pe.withStroke(linewidth=2.0, foreground="#111111")
                        ],
                        zorder=5,
                        clip_on=True,
                    )

                cum_x += z_val

    # Invert and configure subplots
    ax_left.set_ylim(y_max_bound, y_min_bound)

    # Left Subplot Settings
    ax_left.set_xlim(max_xlim, 0.0)
    ax_left.set_xticks(x_ticks)
    ax_left.set_title(
        "Left Hippocampus (hemi-L)",
        fontsize=17.5,
        fontweight="heavy",
        pad=14,
        color="#0b2447",
    )
    ax_left.set_xlabel(
        "Cumulative Positive FC Strength (Fisher-z)",
        fontsize=13.5,
        fontweight="heavy",
        labelpad=10,
    )
    ax_left.grid(True, axis="x", color="#e2e2e2", lw=0.7, ls=":", zorder=1)
    ax_left.tick_params(axis="x", labelsize=11.0)
    ax_left.tick_params(axis="y", left=False, labelleft=False)
    ax_left.spines[["top", "left"]].set_visible(False)
    ax_left.spines["right"].set_color("#222222")
    ax_left.spines["right"].set_linewidth(1.5)

    # Right Subplot Settings
    ax_right.set_xlim(0.0, max_xlim)
    ax_right.set_xticks(list(reversed(x_ticks)))
    ax_right.set_title(
        "Right Hippocampus (hemi-R)",
        fontsize=17.5,
        fontweight="heavy",
        pad=14,
        color="#0b2447",
    )
    ax_right.set_xlabel(
        "Cumulative Positive FC Strength (Fisher-z)",
        fontsize=13.5,
        fontweight="heavy",
        labelpad=10,
    )
    ax_right.grid(True, axis="x", color="#e2e2e2", lw=0.7, ls=":", zorder=1)
    ax_right.tick_params(axis="x", labelsize=11.0)
    ax_right.tick_params(axis="y", left=False, labelleft=False)
    ax_right.spines[["top", "right"]].set_visible(False)
    ax_right.spines["left"].set_color("#222222")
    ax_right.spines["left"].set_linewidth(1.5)

    # Center Subplot Settings: Display each cohort's 4-view 3D Workbench overlay with natural aspect="equal"
    ax_mid.axis("off")
    ax_mid.set_xlim(0.0, 1.0)
    center_title = (
        format_column_atlas_title(column_title, smoothing)
        if column_title
        else f"{smoothing} Atlas (Front: L/R | Back: L/R)"
    )
    ax_mid.set_title(
        center_title, fontsize=15.0, fontweight="heavy", pad=14, color="#0b2447"
    )

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
        ax_left.axhline(
            cb["bottom_y"] + cohort_gap / 2.0, color="#dcdcdc", lw=0.8, ls=":", zorder=1
        )
        ax_right.axhline(
            cb["bottom_y"] + cohort_gap / 2.0, color="#dcdcdc", lw=0.8, ls=":", zorder=1
        )

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
            bbox=dict(
                boxstyle="round,pad=0.45",
                fc="#ffffff",
                ec="#0b3c5d",
                lw=1.6,
                alpha=0.98,
            ),
        )


def render_multidataset_cluster_positive_stacked(
    out_path: Path,
    spec: GridSpec = GridSpec("2mm", "36p"),
    cohort_columns: tuple[list[dict], list[dict]] | None = None,
    signal_branch: str | None = None,
):
    _validate_signal_branch_value(signal_branch)
    scene_path = DEFAULT_SCENE
    with tempfile.TemporaryDirectory(prefix="wb_pos_stacked_") as tmp_dir:
        plot_rows = build_plot_data(
            Path(tmp_dir),
            scene_path,
            spec=spec,
            cohort_columns=cohort_columns,
            signal_branch=signal_branch,
        )

        # Dynamic cohort split into Col 1 and Col 2
        n_half = len(plot_rows) // 2
        col1_rows = plot_rows[:n_half]
        col2_rows = plot_rows[n_half:]

        # Compute synchronized row transitions so rows in Col 1 and Col 2 align horizontally
        n_trans = min(len(col1_rows), len(col2_rows)) - 1
        default_cohort_gap = 1.05
        default_dataset_gap = 1.65
        custom_gaps = []
        for i in range(n_trans):
            g1 = (
                default_dataset_gap
                if (
                    i + 1 < len(col1_rows)
                    and col1_rows[i + 1]["dataset_name"] != col1_rows[i]["dataset_name"]
                )
                else default_cohort_gap
            )
            g2 = (
                default_dataset_gap
                if (
                    i + 1 < len(col2_rows)
                    and col2_rows[i + 1]["dataset_name"] != col2_rows[i]["dataset_name"]
                )
                else default_cohort_gap
            )
            custom_gaps.append(max(g1, g2))

        bar_entries_1, cohort_boundaries_1, dataset_bands_1, total_y_1 = (
            layout_column_data(col1_rows, custom_gaps=custom_gaps)
        )
        bar_entries_2, cohort_boundaries_2, dataset_bands_2, total_y_2 = (
            layout_column_data(col2_rows, custom_gaps=custom_gaps)
        )

        max_span = max(total_y_1, total_y_2)
        y_min_bound = -0.50 - 0.60
        y_max_bound = max_span + 0.60

        fig_h = 15.6 if len(col1_rows) >= 6 else 13.0
        fig = plt.figure(figsize=(38.0, fig_h), dpi=300)

        # width_ratios=[1.0, 0.430, 1.0] and wspace=0.005 ensures the 4-view 3D surface sits snug against the left and right y-axes with NO empty gaps
        gs_col1 = fig.add_gridspec(
            1,
            3,
            left=0.065,
            right=0.487,
            top=0.905,
            bottom=0.115,
            width_ratios=[1.0, 0.430, 1.0],
            wspace=0.005,
        )

        gs_col2 = fig.add_gridspec(
            1,
            3,
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
            smoothing=spec.smoothing,
            variant=spec.variant,
            column_title=col1_rows[0].get("column_title") if col1_rows else None,
        )

        # Col 2: badge_x = -0.320 comfortable distance for longer labels like NTS Pure MDD
        render_column_axes(
            fig=fig,
            gs_col=gs_col2,
            bar_entries=bar_entries_2,
            cohort_boundaries=cohort_boundaries_2,
            dataset_bands=dataset_bands_2,
            y_min_bound=y_min_bound,
            y_max_bound=y_max_bound,
            used_networks=used_networks,
            badge_x=-0.320,
            smoothing=spec.smoothing,
            variant=spec.variant,
            column_title=col2_rows[0].get("column_title") if col2_rows else None,
        )

        # Suptitle
        fig.suptitle(
            FIGURE_TITLE,
            fontsize=23.0,
            fontweight="heavy",
            x=0.50,
            y=0.975,
            ha="center",
            color="#0b2447",
        )

        # Global legend at bottom with all used positive networks
        legend_patches = []
        canon_order = [
            "VentralAttention",
            "Somatomotor",
            "Default",
            "Auditory",
            "DorsalAttention",
            "Control",
            "Visual",
            "Language",
            "CinguloOpercular",
        ]
        sorted_nets = [net for net in canon_order if net in used_networks] + [
            net for net in sorted(used_networks) if net not in canon_order
        ]

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
            title=f"Cortical Functional Networks ({'Kong2022 7-Net' if spec.is_7net else 'Kong2022'} {spec.smoothing})",
            title_fontsize=13.5,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(
            f"[SUCCESS] Saved compact tight stacked cluster positive FC figure (5 rows x 2 cols, {spec.file_tag}) to: {out_path}"
        )


def _render_grid():
    parser = argparse.ArgumentParser(
        description="Plot compact multi-dataset cluster positive FC stacked profiles with 3D overlays in 5 rows x 2 columns."
    )
    parser.add_argument(
        "--version",
        choices=["legacy", COMPARISON_VERSION],
        default="legacy",
        help="Data branch; legacy is unchanged, or use the additive HCP/JJ_TMS comparison.",
    )
    default_png = (
        REPO_ROOT
        / "scripts"
        / "visualize"
        / "multidataset_cluster_confidence_stacked"
        / "multidataset_cluster_confidence_stacked.png"
    )
    parser.add_argument(
        "--out-png",
        default=None,
        help=f"Output PNG path (defaults to {default_png} or grid-specific name)",
    )
    parser.add_argument(
        "--smoothing",
        choices=["2mm", "4mm"],
        default="2mm",
        help="Hippocampus smoothing resolution (default: 2mm)",
    )
    parser.add_argument(
        "--variant",
        choices=["36p", "22p"],
        default="36p",
        help="Denoising pipeline variant (default: 36p)",
    )
    parser.add_argument(
        "--signal-branch",
        choices=list(SIGNAL_BRANCHES),
        default="standard",
        help="Signal branch for hippocampal products (standard or ctxreg). "
        "ctxreg is only valid with the hcp_jjtms_7net comparison version.",
    )
    parser.add_argument(
        "--atlas",
        choices=["kong2022", "7net"],
        default="7net",
        help="Atlas / network taxonomy version. Default: 7net (canonical7, unsuffixed); kong2022 is an equivalent path alias.",
    )
    parser.add_argument(
        "--grid",
        choices=list(GRID_VARIANTS),
        default=None,
        help="Grid specification shortcut (e.g. 2mm_36p, 4mm_22p)",
    )
    args = parser.parse_args()

    if args.version != COMPARISON_VERSION and args.signal_branch != "standard":
        parser.error(
            "--signal-branch ctxreg is only valid with the hcp_jjtms_7net comparison version"
        )

    spec = (
        GridSpec.from_tag(args.grid, atlas=args.atlas)
        if args.grid
        else GridSpec(
            args.smoothing,
            args.variant,
            signal_branch=args.signal_branch,
            atlas=args.atlas,
        )
    )

    if args.out_png:
        out_png = Path(args.out_png).resolve()
    elif args.version == COMPARISON_VERSION and spec.signal_branch != "standard":
        out_png = (
            default_png.parent
            / f"multidataset_cluster_confidence_stacked_{comparison_output_tag(spec, spec.signal_branch)}.png"
        )
    else:
        output_tag = (
            get_comparison_output_tag(spec)
            if args.version == COMPARISON_VERSION
            else spec.file_tag
        )
        out_png = (
            default_png.parent
            / f"multidataset_cluster_confidence_stacked_{output_tag}.png"
        )

    if args.version == COMPARISON_VERSION:
        render_spec = GridSpec(
            spec.smoothing, spec.variant, signal_branch=spec.signal_branch, atlas="7net"
        )
        comparison = get_comparison_visual_columns(render_spec)
        render_multidataset_cluster_positive_stacked(
            out_png,
            spec=render_spec,
            cohort_columns=comparison,
            signal_branch=render_spec.signal_branch,
        )
    else:
        render_multidataset_cluster_positive_stacked(out_png, spec=spec)


if __name__ == "__main__":
    raise SystemExit(
        "Standalone rendering has been removed. Use "
        "scripts/visualize/run_multidataset_suite.py "
        "with --figures to select the figure type; all four grids are processed in 4-way parallel by default."
    )
SIGNAL_BRANCHES = ("standard", "ctxreg")
CANONICAL_NETWORK_ORDER = (
    "Default",
    "Visual",
    "Somatomotor",
    "DorsalAttention",
    "VentralAttention",
    "Control",
    "Language",
)
_CTXREG_IOWA_STEMS = {
    "iowa_hc": "hc",
    "iowa_pat": "patient",
    "iowa_pure_mdd": "pure_mdd",
    "iowa_comorbid_mdd": "comorbid_mdd",
}


def _coerce_comparison_spec(spec: GridSpec | str) -> GridSpec:
    if isinstance(spec, GridSpec):
        return spec
    if isinstance(spec, str):
        return GridSpec.from_tag(spec)
    raise TypeError(f"spec must be GridSpec or grid tag, got {type(spec)!r}")


def _validate_signal_branch_value(signal_branch: str | None) -> None:
    if signal_branch is not None and signal_branch not in SIGNAL_BRANCHES:
        raise ValueError(
            f"signal_branch must be one of {SIGNAL_BRANCHES}, got {signal_branch!r}"
        )


def _validate_comparison_request(spec: GridSpec, signal_branch: str) -> None:
    _validate_signal_branch_value(signal_branch)
    if not spec.is_7net:
        raise ValueError(
            f"{signal_branch} comparison branch requires the canonical 7-network atlas; "
            f"got atlas={spec.atlas!r}"
        )
    if signal_branch == "ctxreg" and (spec.smoothing, spec.variant) != ("2mm", "36p"):
        raise ValueError(
            "ctxreg comparison branch requires the canonical 2mm/36p grid; "
            f"got {spec.smoothing}/{spec.variant}"
        )


def comparison_output_tag(spec: GridSpec | str, signal_branch: str) -> str:
    """Return a collision-free output tag for one explicit signal branch."""
    resolved = _coerce_comparison_spec(spec)
    _validate_comparison_request(resolved, signal_branch)
    return f"{COMPARISON_VERSION}_{resolved.tag}_{signal_branch}"


def _ctxreg_comparison_paths(
    cohort_id: str,
    method: str,
    spec: GridSpec,
) -> dict[str, Path]:
    """Resolve formal ctxreg paths without consulting Standard paths."""
    if method not in COMPARISON_METHODS:
        raise ValueError(
            f"Unknown comparison method {method!r}; expected {COMPARISON_METHODS}"
        )

    if cohort_id == "hcp_7t":
        dataset_root = REPO_ROOT / "outputs_hcp"
        group_root = dataset_root / "work/group_atlas/routea_ctxreg"
        present_dir = dataset_root / "present/overviews/routea_ctxreg"
        signal_root = dataset_root / "inputs/kong2022/shared_inputs/routea_ctxreg"
        standard_surface_root = (
            dataset_root / "inputs/kong2022/shared_inputs/surface"
        )
        if method == "kong2022_7net":
            reference_root = dataset_root / "inputs/kong2022/shared_inputs/routea"
        else:
            reference_root = (
                dataset_root
                / "inputs/lynch2024_fast_pfm_t1only/shared_inputs/pfm_hcp151"
            )
        hippunfold_root = dataset_root / "inputs/hippunfold/dense_corobl_batch"
    elif cohort_id in _CTXREG_IOWA_STEMS:
        dataset_root = REPO_ROOT / "outputs_iowa_jj_tms"
        stem = _CTXREG_IOWA_STEMS[cohort_id]
        group_root = dataset_root / "work/group_atlas/routea_ctxreg" / stem
        present_dir = dataset_root / "present/overviews/routea_ctxreg" / stem
        signal_root = (
            dataset_root / "inputs/kong2022/shared_inputs/routea_ctxreg"
        )
        standard_surface_root = (
            dataset_root / "inputs/kong2022/shared_inputs/surface"
        )
        if method == "kong2022_7net":
            reference_root = (
                dataset_root / "inputs/kong2022/shared_inputs/routea"
            )
        else:
            reference_root = (
                dataset_root
                / "inputs/lynch2024_fast_pfm_t1only/shared_inputs/pfm_nsi043_78"
            )
        hippunfold_root = (
            dataset_root / "inputs/hippunfold/dense_corobl_batch"
        )
    else:
        raise ValueError(f"Unknown HCP/JJ_TMS comparison cohort id: {cohort_id!r}")

    atlas_folder = "kong2022" if method == "kong2022_7net" else method
    grad_subpath = (
        group_root
        / f"gradient-spectral/{atlas_folder}/{spec.smoothing}/cov_0p80/g1g2g3"
    )
    wta_subpath = group_root / f"wta-top/{atlas_folder}/{spec.smoothing}/cov_0p80"
    if method == "kong2022_7net":
        overview_grad_name = (
            "kong2022_gradient-spectral_group_overview_g1g2g3_routea_ctxreg_fisherz.png"
        )
        overview_grad_normalized_name = "kong2022_gradient-spectral_group_overview_g1g2g3_routea_ctxreg_normalized_top20_count.png"
        overview_wta_name = (
            "kong2022_wta-top_group_overview_routea_ctxreg_normalized_top20_count.png"
        )
    else:
        # Upstream Lynch overviews differ by dataset: HCP embeds the g1g2g3
        # component tag while Iowa does not. Resolve the manifest-declared name.
        overview_grad_name = (
            "lynch2024_7net_gradient-spectral_group_overview_g1g2g3_routea_ctxreg_fisherz.png"
            if cohort_id == "hcp_7t"
            else "lynch2024_7net_gradient-spectral_group_overview_routea_ctxreg_fisherz.png"
        )
        overview_grad_normalized_name = None
        overview_wta_name = (
            "lynch2024_7net_wta-top_group_overview_routea_ctxreg_fisherz.png"
        )

    return {
        "group_root": group_root,
        "group_root_grad": group_root,
        "group_root_wta": group_root,
        "grad_subpath": grad_subpath,
        "wta_subpath": wta_subpath,
        "present_dir": present_dir,
        "overview_grad": present_dir / overview_grad_name,
        "overview_wta": present_dir / overview_wta_name,
        "shared_root": signal_root,
        "reference_root": reference_root,
        "surface_root": signal_root,
        "mask_root": standard_surface_root,
        "hippunfold_root": hippunfold_root,
        "signal_branch": "ctxreg",
        **(
            {
                "overview_grad_normalized_top20_count": present_dir
                / overview_grad_normalized_name
            }
            if overview_grad_normalized_name is not None
            else {}
        ),
    }


def resolve_comparison_visual_columns(
    spec: GridSpec | str,
    signal_branch: str,
) -> tuple[list[dict], list[dict]]:
    """Resolve one explicit Standard or CtxReg visual column pair."""
    resolved = _coerce_comparison_spec(spec)
    _validate_comparison_request(resolved, signal_branch)
    standard_columns = get_comparison_visual_columns(resolved)
    if signal_branch == "standard":
        return tuple(
            [dict(entry, signal_branch="standard") for entry in column]
            for column in standard_columns
        )

    ctxreg_columns = []
    for column in standard_columns:
        branch_entries = []
        for entry in column:
            branch_paths = _ctxreg_comparison_paths(
                entry["id"], entry["method"], resolved
            )
            branch_entries.append(dict(entry, **branch_paths))
        ctxreg_columns.append(branch_entries)
    return ctxreg_columns[0], ctxreg_columns[1]


def _has_explicit_legacy_hcp_ctxreg_provenance(
    path: Path,
    manifest: dict,
    *,
    method: str | None,
) -> bool:
    """Accept only the named pre-signal_branch HCP Kong ctxreg contract."""
    if method != "kong2022_7net" or manifest.get("dataset") != "outputs_hcp":
        return False
    if "routea_ctxreg" not in path.parts or manifest.get("atlas") != "kong2022":
        return False

    def is_ctxreg_path(value: object) -> bool:
        return (
            isinstance(value, str) and "routea_ctxreg" in Path(value).expanduser().parts
        )

    for key in ("shared_root", "profile_surface_root"):
        if not is_ctxreg_path(manifest.get(key)):
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
    if not all(is_ctxreg_path(value) for value in consumed.values()):
        return False
    for key in ("signal_root", "profile_source_root"):
        value = manifest.get(key)
        if value is not None and not is_ctxreg_path(value):
            return False
    signal_control_route = manifest.get("signal_control_route")
    return signal_control_route in (None, "routea_ctxreg")


def _validate_manifest_signal_branch(
    manifest_path: Path,
    manifest: dict,
    signal_branch: str | None,
    *,
    method: str | None,
    provenance_path: Path,
) -> None:
    _validate_signal_branch_value(signal_branch)
    if signal_branch is None:
        return
    recorded = manifest.get("signal_branch")
    if signal_branch == "ctxreg" and recorded != "ctxreg":
        if not (
            recorded is None
            and _has_explicit_legacy_hcp_ctxreg_provenance(
                provenance_path, manifest, method=method
            )
        ):
            raise ValueError(
                f"{manifest_path} records signal_branch={recorded!r}; "
                f"expected {signal_branch!r}"
            )
    elif signal_branch == "standard" and recorded not in (None, "standard"):
        raise ValueError(
            f"{manifest_path} records signal_branch={recorded!r}; "
            f"expected {signal_branch!r}"
        )


def _declared_file_value(
    manifest_path: Path,
    owner: dict,
    key: str,
    *,
    within_manifest_dir: bool = False,
) -> Path:
    value = owner.get(key)
    if not isinstance(value, str) or not value.strip():
        raise FileNotFoundError(
            f"{manifest_path} canonical manifest lacks a declared {key} artifact"
        )
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{manifest_path} declares non-absolute {key}: {path}")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{manifest_path} declares missing {key}: {path}"
        ) from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"{manifest_path} declares non-file {key}: {path}")
    if within_manifest_dir and manifest_path.parent.resolve() not in resolved.parents:
        raise ValueError(
            f"{manifest_path} declares {key} outside its hemisphere directory: {path}"
        )
    return resolved


def _validate_assignment_manifest(manifest_path: Path, manifest: dict) -> dict:
    variants = manifest.get("assignment_variants")
    if not isinstance(variants, dict) or not isinstance(variants.get("fisherz"), dict):
        raise ValueError(
            f"{manifest_path} lacks the exact assignment_variants.fisherz contract"
        )
    assignment = variants["fisherz"]
    fields = (
        "cluster_assignment_status",
        "group_functional_atlas_labels",
        "group_functional_atlas_label_map",
        "template_functional_atlas_label",
    )
    paths = tuple(
        _declared_file_value(
            manifest_path,
            assignment,
            field,
            within_manifest_dir=True,
        )
        for field in fields
    )
    return {"label_path": paths[-1], "paths": paths}


def _load_positive_profile_contract(
    manifest_path: Path,
    manifest: dict,
) -> tuple[np.ndarray, list[str], Path, Path]:
    profile_path = _declared_file_value(
        manifest_path, manifest, "cluster_network_profile", within_manifest_dir=True
    )
    # Explicit documented decision (2026-09-16): no comparison manifest
    # (kong2022 or lynch, any cohort) declares a 'group_feature_summary' key,
    # while the exact file exists in every hemisphere directory with valid
    # content (verified across all 10 comparison entries). Prefer the manifest
    # declaration; when absent, accept ONLY the exact file inside the
    # manifest's own hemisphere directory and keep every content check below.
    # Anything else fails closed. This is not a cross-file substitution.
    if isinstance(manifest.get("group_feature_summary"), str) and manifest.get(
        "group_feature_summary"
    ):
        summary_path = _declared_file_value(
            manifest_path,
            manifest,
            "group_feature_summary",
            within_manifest_dir=True,
        )
    else:
        candidate = manifest_path.parent / "group_feature_summary.json"
        if not candidate.is_file():
            raise FileNotFoundError(
                f"{manifest_path} lacks a declared group_feature_summary artifact "
                f"and the exact hemisphere file is missing: {candidate}"
            )
        summary_path = candidate
    prof = np.asarray(np.load(profile_path, allow_pickle=False), dtype=np.float64)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    network_names = summary.get("network_names") if isinstance(summary, dict) else None
    if not isinstance(network_names, list):
        raise ValueError(f"{summary_path} lacks the exact network_names list")
    network_names = [str(name) for name in network_names]
    if prof.shape != (3, 7) or tuple(network_names) != CANONICAL_NETWORK_ORDER:
        raise ValueError(
            f"Malformed positive cluster profile contract under {manifest_path.parent}: "
            f"profile shape={prof.shape}, network_names={network_names!r}; "
            "expected K=3 and canonical7 order"
        )
    if not np.isfinite(prof).all():
        raise ValueError(
            f"Non-finite values in canonical positive cluster profile: {profile_path}"
        )
    return prof, network_names, profile_path, summary_path


def _manifest_subject(manifest_path: Path, manifest: dict) -> str:
    subject = str(manifest.get("template_subject", "")).replace("sub-", "").strip()
    if not subject:
        raise ValueError(f"{manifest_path} has no usable template_subject")
    return subject


def _validate_comparison_manifest_inputs(
    grad_dir: Path,
    *,
    method: str,
    signal_branch: str,
) -> None:
    subjects = {}
    for hemi in ("L", "R"):
        manifest_path = grad_dir / f"hemi_{hemi}/group_atlas_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing comparison {signal_branch} manifest for {method}: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_manifest_signal_branch(
            manifest_path,
            manifest,
            signal_branch,
            method=method,
            provenance_path=grad_dir,
        )
        _validate_assignment_manifest(manifest_path, manifest)
        _declared_file_value(manifest_path, manifest, "template_left_or_right_surface")
        _declared_file_value(manifest_path, manifest, "template_spec_path")
        _load_positive_profile_contract(manifest_path, manifest)
        subjects[hemi] = _manifest_subject(manifest_path, manifest)
    if subjects["L"] != subjects["R"]:
        raise ValueError(
            f"{grad_dir} has mismatched template_subject values: "
            f"hemi_L={subjects['L']!r}, hemi_R={subjects['R']!r}"
        )
    return None
    """
    mf_path = hpath / "group_atlas_manifest.json"
    if not mf_path.is_file():
        raise FileNotFoundError(
            f"Missing canonical manifest for positive cluster profile: {mf_path}"
        )
    mf_data = json.loads(mf_path.read_text(encoding="utf-8"))
    _validate_manifest_signal_branch(
        mf_path,
        mf_data,
        signal_branch,
        method=method,
        provenance_path=grad_dir,
    )
    prof, network_names, _prof_path, _summary_path = _load_positive_profile_contract(
        mf_path, mf_data
    )
    out = []
    for cid, row in enumerate(prof, start=1):
        pos_nets = [
            (net, float(row[i]))
            for i, net in enumerate(network_names)
            if float(row[i]) > 1e-5
        ]
        pos_nets.sort(key=lambda x: x[1], reverse=True)
        top1_net = pos_nets[0][0] if pos_nets else "unassigned"
        top1_z = pos_nets[0][1] if pos_nets else 0.0
        top2_net = pos_nets[1][0] if len(pos_nets) > 1 else "unassigned"
        top2_z = pos_nets[1][1] if len(pos_nets) > 1 else 0.0
        out.append({
            "cluster_id": cid,
            "cluster_name": CLUSTER_NAMES.get(cid, f"Cluster {cid}"),
            "pos_networks": pos_nets,
            "top1_network": top1_net,
            "top1_z": top1_z,
            "top2_network": top2_net,
    """

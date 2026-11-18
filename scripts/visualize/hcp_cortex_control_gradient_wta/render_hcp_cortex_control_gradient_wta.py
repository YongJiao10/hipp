#!/usr/bin/env python3
"""HCP 7T hippocampal organization: Standard vs. CtxReg 10 mm vs. CtxReg 20 mm, 2mm and 4mm smoothing.

One figure with two smoothing blocks (2mm on top, 4mm below), each with three rows
(Standard `pfm`, CtxReg 10 mm `pfm_ctxreg_10mm`, CtxReg 20 mm `pfm_ctxreg`) and four
modality columns. Gradient panels read the formal HCP Kong2022 gradient run selected by
`multidataset_cohorts.HCP_KONG_GRADIENT_RUN` (TopSim profile, primary Ward partition).

  1. Gradient clusters, positive Fisher-z FC (adapted from multidataset_cluster_confidence_stacked):
     [Left Hipp bar] | [2x2 surface] | [Right Hipp bar]; row-specific Fisher-z scale;
     C1 anterior at top, C3 posterior at bottom.
  2. Gradient clusters, OvR logistic model on network time-series Fisher-z features
     (`ovr_logistic_canonical7_network_ts`):
     same layout as column 1 with positive raw coefficients; the cluster label is the largest
     positive raw coefficient.
  3. Gradient clusters, normalized top-20 ROI count (adapted from
     multidataset_gradient_subregion_top20_stacked_2col): [Branch card] | [2x2 surface] | [Left] | [Right].
  4. Winner-take-all labels in 20 long-axis bins (adapted from
     multidataset_wta_normalized_longaxis_stacked_2col): bin 1 anterior at top.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
CORE_DIR = REPO_ROOT / "scripts" / "core_workflow"
VIS_LIB_DIR = REPO_ROOT / "scripts" / "visualize" / "lib"
for p in (CORE_DIR, VIS_LIB_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lib.surface_utils import long_axis_order, render_hippocampal_grid
from multidataset_cohorts import HCP_COHORT_SIZE, HCP_KONG_GRADIENT_RUN

# -----------------------------------------------------------------------------
# Canonical Styling & Palette (Exact Match to References)
# -----------------------------------------------------------------------------
NETWORK_COLORS: dict[str, str] = {
    "Default": "#de616c",
    "Somatomotor": "#5297c8",
    "Auditory": "#18b7c9",
    "VentralAttention": "#da42f5",
    "DorsalAttention": "#33a02c",
    "Control": "#f39c12",
    "Language": "#8b4513",
    "Visual": "#6a3d9a",
    "Null": "#888888",
    "None": "#888888",
    "non-network/null": "#888888",
}

ABBREV_MAP: dict[str, str] = {
    "Default": "DMN",
    "Visual": "VN",
    "Somatomotor": "SMN",
    "DorsalAttention": "DAN",
    "VentralAttention": "VAN",
    "Control": "CON",
    "Auditory": "AUD",
    "Language": "LAN",
    "Null": "Null",
    "None": "Null",
    "non-network/null": "Null",
}

STACK_ORDER: list[str] = [
    "Default",
    "Somatomotor",
    "VentralAttention",
    "DorsalAttention",
    "Control",
    "Language",
    "Visual",
    "non-network/null",
]

SUBREGION_NAMES: dict[int, str] = {
    1: "Anterior",
    2: "Middle",
    3: "Posterior",
}

CLUSTER_NAMES: dict[int, str] = {
    1: "C1: Anterior",
    2: "C2: Middle",
    3: "C3: Posterior",
}

# -----------------------------------------------------------------------------
# HCP Cortex-Control Branch Definitions (Contract-Compliant Formal Roots)
# -----------------------------------------------------------------------------
BRANCHES: list[dict[str, Any]] = [
    {
        "id": "standard",
        "cohort_name": "Standard",
        "dataset_name": "HCP 7T",
        "n_subjects": HCP_COHORT_SIZE,
        "badge_bg": "#0b3c5d",
        "badge_fg": "#ffffff",
        "bg_color": "#ffffff",
        "group_root": REPO_ROOT / "outputs_hcp" / "work" / "group_atlas" / "pfm",
    },
    {
        "id": "ctxreg_10mm",
        "cohort_name": "CtxReg 10 mm",
        "dataset_name": "HCP 7T",
        "n_subjects": HCP_COHORT_SIZE,
        "badge_bg": "#b45309",
        "badge_fg": "#ffffff",
        "bg_color": "#fffbf5",
        "group_root": REPO_ROOT / "outputs_hcp" / "work" / "group_atlas" / "pfm_ctxreg_10mm",
    },
    {
        "id": "ctxreg_20mm",
        "cohort_name": "CtxReg 20 mm",
        "dataset_name": "HCP 7T",
        "n_subjects": HCP_COHORT_SIZE,
        "badge_bg": "#7c2d12",
        "badge_fg": "#ffffff",
        "bg_color": "#fff7ef",
        "group_root": REPO_ROOT / "outputs_hcp" / "work" / "group_atlas" / "pfm_ctxreg",
    },
]
SMOOTHINGS = ("2mm", "4mm")
DEFAULT_DIR = REPO_ROOT / "scripts" / "visualize" / "hcp_cortex_control_gradient_wta"
DEFAULT_OUT_PNG = DEFAULT_DIR / "hcp_cortex_control_gradient_wta.png"
DEFAULT_STATS_TSV = DEFAULT_DIR / "stats" / "hcp_cortex_control_gradient_wta_stats.tsv"


REFERENCE_FIG_HEIGHT_IN = 18.0
# Column widths (relative): Fisher-z | OvR | Top-20 | WTA. FIG_WIDTH_IN keeps each
# column at the physical width of the earlier three-column 49 in layout.
COLUMN_WIDTH_RATIOS = (0.92, 0.92, 1.05, 1.05)
FIG_WIDTH_IN = 49.0 * sum(COLUMN_WIDTH_RATIOS) / (0.92 + 1.05 + 1.05)


# -----------------------------------------------------------------------------
# Data Loaders (Verified Contract Compliant)
# -----------------------------------------------------------------------------
def load_branch_fisherz(branch: dict[str, Any], smoothing: str) -> dict[str, list[dict[str, Any]]]:
    """Load cluster network profile and assignment status for Fisher-z."""
    group_root = Path(branch["group_root"])
    out: dict[str, list[dict[str, Any]]] = {}

    for hemi in ("L", "R"):
        hdir = group_root / f"{HCP_KONG_GRADIENT_RUN}/kong2022/{smoothing}/cov_0p80/g1g2g3/hemi_{hemi}"
        npy_path = hdir / "cluster_network_profile.npy"
        status_path = hdir / "cluster_assignment_status_fisherz.json"
        summary_path = hdir / "group_feature_summary.json"

        if not npy_path.is_file() or not status_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"Missing Fisher-z group atlas files in {hdir}")

        profile = np.load(npy_path)
        status = json.loads(status_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        network_names = summary.get("network_names", [])

        status_items = {it["cluster_id"]: it for it in status.get("items", [])}
        clusters = []

        for cid in range(1, 4):
            c_idx = cid - 1
            row = profile[c_idx]
            pos_nets = []
            for net_name, val in zip(network_names, row):
                if val > 0.0:
                    pos_nets.append((net_name, float(val)))
            pos_nets.sort(key=lambda x: (x[1], -STACK_ORDER.index(x[0]) if x[0] in STACK_ORDER else -99), reverse=True)

            it = status_items.get(cid, {})
            clusters.append({
                "cluster_id": cid,
                "cluster_name": CLUSTER_NAMES.get(cid, f"C{cid}"),
                "pos_networks": pos_nets,
                "top1_network": it.get("top1_network", "unassigned"),
                "top2_network": it.get("top2_network", "none"),
                "margin": float(it.get("margin", 0.0)),
            })
        out[hemi] = clusters
    return out


OVR_VARIANT = "ovr_logistic_canonical7_network_ts"


def load_branch_ovr(branch: dict[str, Any], smoothing: str) -> dict[str, list[dict[str, Any]]]:
    """Load the network-time-series OvR logistic model (canonical7) coefficients for each gradient cluster."""
    group_root = Path(branch["group_root"])
    out: dict[str, list[dict[str, Any]]] = {}

    for hemi in ("L", "R"):
        hdir = group_root / f"{HCP_KONG_GRADIENT_RUN}/kong2022/{smoothing}/cov_0p80/g1g2g3/hemi_{hemi}"
        status_path = hdir / f"cluster_assignment_status_{OVR_VARIANT}.json"
        if not status_path.is_file():
            raise FileNotFoundError(f"Missing {OVR_VARIANT} assignment status in {hdir}")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("assignment_metric") != "largest_positive_raw_beta" or status.get("network_taxonomy") != "canonical7":
            raise ValueError(f"Unexpected OvR assignment contract in {status_path}")
        items = {int(it["cluster_id"]): it for it in status["items"]}
        if sorted(items) != [1, 2, 3]:
            raise ValueError(f"Expected OvR clusters 1-3 in {status_path}, found {sorted(items)}")

        clusters = []
        for cid in range(1, 4):
            it = items[cid]
            pos_nets = [(str(k), float(v)) for k, v in it["beta_by_network"].items() if float(v) > 0.0]
            pos_nets.sort(key=lambda x: (x[1], -STACK_ORDER.index(x[0]) if x[0] in STACK_ORDER else -99), reverse=True)
            clusters.append({
                "cluster_id": cid,
                "cluster_name": CLUSTER_NAMES.get(cid, f"C{cid}"),
                "pos_networks": pos_nets,
                "top1_network": str(it["top1_network"]),
                "top1_beta": float(it["top1_beta"]),
            })
        out[hemi] = clusters
    return out


def load_branch_top20_count(branch: dict[str, Any], smoothing: str) -> dict[str, list[dict[str, Any]]]:
    """Load normalized Top-20 connected cortical network percentage for Anterior, Middle, Posterior."""
    group_root = Path(branch["group_root"])
    out: dict[str, list[dict[str, Any]]] = {}

    for hemi in ("L", "R"):
        hdir = group_root / f"{HCP_KONG_GRADIENT_RUN}/kong2022/{smoothing}/cov_0p80/g1g2g3/hemi_{hemi}"
        status_path = hdir / "cluster_assignment_status_normalized_top20_count.json"

        if not status_path.is_file():
            raise FileNotFoundError(f"Missing normalized Top-20 status in {hdir}")

        st_data = json.loads(status_path.read_text(encoding="utf-8"))
        items = st_data.get("items", [])
        hemi_subregions = []

        for it in items:
            cid = int(it.get("cluster_id", 0))
            norm_counts = it.get("normalized_count_by_network", {})
            values = {str(k): float(v) for k, v in norm_counts.items() if float(v) > 0.0}
            total = sum(values.values())
            comp = {k: (v / total * 100.0) if total > 0 else 0.0 for k, v in values.items()}

            hemi_subregions.append({
                "cluster_id": cid,
                "subregion_name": SUBREGION_NAMES.get(cid, f"Subregion {cid}"),
                "comp": comp,
                "top1_network": str(it.get("top1_network", "unassigned")),
                "top1_count": int(it.get("top1_count", 0)),
                "top2_network": str(it.get("top2_network", "none")),
                "top2_count": int(it.get("top2_count", 0)),
                "margin_count": int(it.get("margin_count", 0)),
            })
        hemi_subregions.sort(key=lambda x: x["cluster_id"])
        out[hemi] = hemi_subregions
    return out


def load_branch_wta(branch: dict[str, Any], smoothing: str, n_bins: int = 20) -> dict[str, list[dict[str, float]]]:
    """Load long-axis 20 bins network percentage data for hemi-L and hemi-R."""
    group_root = Path(branch["group_root"])
    out: dict[str, list[dict[str, float]]] = {}

    for hemi in ("L", "R"):
        hdir = group_root / f"wta-top/kong2022/{smoothing}/cov_0p80/hemi_{hemi}"
        mf_path = hdir / "group_atlas_manifest.json"
        lbl_path = hdir / "group_functional_atlas_labels.npy"
        lmap_path = hdir / "group_functional_atlas_label_map.json"

        if not mf_path.is_file() or not lbl_path.is_file() or not lmap_path.is_file():
            raise FileNotFoundError(f"Missing WTA group atlas data in {hdir}")

        mf = json.loads(mf_path.read_text(encoding="utf-8"))
        surf_path = Path(mf["template_left_or_right_surface"])
        order = long_axis_order(surf_path)

        labels = np.load(lbl_path).flatten()
        lmap = json.loads(lmap_path.read_text(encoding="utf-8"))
        network_names = lmap["network_names"]
        label_keys = {k: int(v) for k, v in lmap["network_label_keys"].items()}
        known_keys = set(label_keys.values())

        bins = np.array_split(order, int(n_bins))
        hemi_bins = []
        for b_verts in bins:
            b_lbls = labels[b_verts] if len(b_verts) > 0 else np.array([])
            denom = float(len(b_lbls))
            comp: dict[str, float] = {}
            for net in network_names:
                k = label_keys[net]
                cnt = float(np.sum(b_lbls == k)) if denom > 0 else 0.0
                comp[net] = (100.0 * cnt / denom) if denom > 0 else 0.0

            null_cnt = float(np.sum(~np.isin(b_lbls, list(known_keys)))) if denom > 0 else 0.0
            comp["non-network/null"] = (100.0 * null_cnt / denom) if denom > 0 else 0.0
            hemi_bins.append(comp)

        out[hemi] = hemi_bins
    return out


# -----------------------------------------------------------------------------
# Adaptive Positive-FC Axis Scale (Exact from plot_multidataset_cluster_positive_stacked.py)
# -----------------------------------------------------------------------------
def adaptive_positive_axis_scale(
    clusters_l: list[dict[str, Any]],
    clusters_r: list[dict[str, Any]],
    target_intervals: int = 5,
) -> tuple[float, list[float]]:
    """Return clean positive-FC x-range and descending tick positions.

    Uses standard nice step factors {1.0, 2.0, 5.0, 10.0} to ensure tick marks
    always fall on clean rounded intervals (e.g. 0.02, 0.05) and never on
    fractional oddities (such as 0.0125 or 0.0875).
    """
    max_stack = 0.0
    for side_clusters in (clusters_l, clusters_r):
        for cinfo in side_clusters:
            tot = sum(val for _, val in cinfo.get("pos_networks", []))
            max_stack = max(max_stack, tot)

    if max_stack <= 0.0:
        return 0.10, [0.10, 0.08, 0.06, 0.04, 0.02, 0.0]

    raw_step = max_stack / float(target_intervals)
    magnitude = 10.0 ** math.floor(math.log10(raw_step))
    norm = raw_step / magnitude

    if norm <= 1.2:
        factor = 1.0
    elif norm <= 2.5:
        factor = 2.0
    elif norm <= 6.0:
        factor = 5.0
    else:
        factor = 10.0

    step = factor * magnitude
    max_xlim = math.ceil((max_stack - 1e-12) / step) * step
    n_intervals = int(round(max_xlim / step))

    step_str = f"{step:.6f}".rstrip("0")
    decimal_places = len(step_str.split(".")[1]) if "." in step_str else 0

    x_ticks = [round(max_xlim - index * step, decimal_places) for index in range(n_intervals + 1)]
    x_ticks = [0.0 if abs(t) < 1e-9 else t for t in x_ticks]
    return round(max_xlim, decimal_places), x_ticks


# -----------------------------------------------------------------------------
# 2x2 Locked View Composite Builder (Ensures Hipp Surface Shares Bar Height)
# -----------------------------------------------------------------------------
def trim_bounding_box(img: Image.Image) -> Image.Image:
    """Trim uniform background margin from render."""
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


def build_2x2_composite(native_im: Image.Image) -> Image.Image:
    """Build a tight 2x2 composite: Front (L/R) on top, Back (L/R) on bottom.
    
    This matches the authentic layout in reference figures 2 and 3, and ensures
    the hippocampal 3D surface occupies the exact same vertical height as the stacked bars.
    """
    arr = np.array(native_im)
    v_l_front = trim_bounding_box(Image.fromarray(arr[: arr.shape[0] // 2, : arr.shape[1] // 2]))
    v_r_front = trim_bounding_box(Image.fromarray(arr[: arr.shape[0] // 2, arr.shape[1] // 2 :]))
    v_l_back  = trim_bounding_box(Image.fromarray(arr[arr.shape[0] // 2 :, : arr.shape[1] // 2]))
    v_r_back  = trim_bounding_box(Image.fromarray(arr[arr.shape[0] // 2 :, arr.shape[1] // 2 :]))

    target_h = max(v_l_front.height, v_r_front.height, v_l_back.height, v_r_back.height)
    scaled = []
    for v in [v_l_front, v_r_front, v_l_back, v_r_back]:
        sc = target_h / float(v.height)
        sw = max(1, int(round(v.width * sc)))
        scaled.append(v.resize((sw, target_h), Image.Resampling.LANCZOS))

    gap_x = 6
    gap_y = 10
    pad = 2

    row1_w = pad * 2 + scaled[0].width + gap_x + scaled[1].width
    row2_w = pad * 2 + scaled[2].width + gap_x + scaled[3].width
    total_w = max(row1_w, row2_w)
    total_h = pad * 2 + target_h * 2 + gap_y

    canvas = Image.new("RGBA", (total_w, total_h), (255, 255, 255, 255))
    # Paste row 1 centered
    r1_x = (total_w - (scaled[0].width + gap_x + scaled[1].width)) // 2
    canvas.paste(scaled[0], (r1_x, pad), scaled[0] if scaled[0].mode == "RGBA" else None)
    canvas.paste(scaled[1], (r1_x + scaled[0].width + gap_x, pad), scaled[1] if scaled[1].mode == "RGBA" else None)

    # Paste row 2 centered
    r2_x = (total_w - (scaled[2].width + gap_x + scaled[3].width)) // 2
    y2 = pad + target_h + gap_y
    canvas.paste(scaled[2], (r2_x, y2), scaled[2] if scaled[2].mode == "RGBA" else None)
    canvas.paste(scaled[3], (r2_x + scaled[2].width + gap_x, y2), scaled[3] if scaled[3].mode == "RGBA" else None)

    return canvas


def render_branch_surface_2x2(
    branch: dict[str, Any],
    modality: str,
    smoothing: str,
    temp_dir: Path,
    panels_dir: Path,
    force: bool = False,
) -> Image.Image:
    """Render or load cached 2x2 hippocampal surface overlay with SHA-256 provenance."""
    branch_id = branch["id"]
    group_root = Path(branch["group_root"])
    scene_path = REPO_ROOT / "config" / "wb_locked_native_view_lateral_medial.scene"
    cached_png = panels_dir / f"wb_{branch_id}_{modality}_{smoothing}_grid2x2.png"
    sources_json = panels_dir / f"wb_{branch_id}_{modality}_{smoothing}_grid2x2.sources.json"

    if modality == "wta":
        ml_path = group_root / f"wta-top/kong2022/{smoothing}/cov_0p80/hemi_L/group_atlas_manifest.json"
        mr_path = group_root / f"wta-top/kong2022/{smoothing}/cov_0p80/hemi_R/group_atlas_manifest.json"
        ml = json.loads(ml_path.read_text(encoding="utf-8"))
        mr = json.loads(mr_path.read_text(encoding="utf-8"))
        lbl_l = Path(ml["template_functional_atlas_label"])
        lbl_r = Path(mr["template_functional_atlas_label"])
    elif modality == "fisherz":
        ml_path = group_root / f"{HCP_KONG_GRADIENT_RUN}/kong2022/{smoothing}/cov_0p80/g1g2g3/hemi_L/group_atlas_manifest.json"
        mr_path = group_root / f"{HCP_KONG_GRADIENT_RUN}/kong2022/{smoothing}/cov_0p80/g1g2g3/hemi_R/group_atlas_manifest.json"
        ml = json.loads(ml_path.read_text(encoding="utf-8"))
        mr = json.loads(mr_path.read_text(encoding="utf-8"))
        lbl_l = Path(ml["assignment_variants"]["fisherz"]["template_functional_atlas_label_variant"])
        lbl_r = Path(mr["assignment_variants"]["fisherz"]["template_functional_atlas_label_variant"])
    elif modality == "ovr_network_ts":
        ml_path = group_root / f"{HCP_KONG_GRADIENT_RUN}/kong2022/{smoothing}/cov_0p80/g1g2g3/hemi_L/group_atlas_manifest.json"
        mr_path = group_root / f"{HCP_KONG_GRADIENT_RUN}/kong2022/{smoothing}/cov_0p80/g1g2g3/hemi_R/group_atlas_manifest.json"
        ml = json.loads(ml_path.read_text(encoding="utf-8"))
        mr = json.loads(mr_path.read_text(encoding="utf-8"))
        lbl_l = Path(ml["assignment_variants"][OVR_VARIANT]["template_functional_atlas_label_variant"])
        lbl_r = Path(mr["assignment_variants"][OVR_VARIANT]["template_functional_atlas_label_variant"])
    elif modality == "top20_count":
        ml_path = group_root / f"{HCP_KONG_GRADIENT_RUN}/kong2022/{smoothing}/cov_0p80/g1g2g3/hemi_L/group_atlas_manifest.json"
        mr_path = group_root / f"{HCP_KONG_GRADIENT_RUN}/kong2022/{smoothing}/cov_0p80/g1g2g3/hemi_R/group_atlas_manifest.json"
        ml = json.loads(ml_path.read_text(encoding="utf-8"))
        mr = json.loads(mr_path.read_text(encoding="utf-8"))
        lbl_l = Path(ml["assignment_variants"]["normalized_top20_count"]["template_functional_atlas_label_variant"])
        lbl_r = Path(mr["assignment_variants"]["normalized_top20_count"]["template_functional_atlas_label_variant"])
    else:
        raise ValueError(f"Unknown modality: {modality}")

    surf_l = Path(ml["template_left_or_right_surface"])
    surf_r = Path(mr["template_left_or_right_surface"])
    spec_path = Path(ml["template_spec_path"])
    subject = str(ml.get("template_subject", "100610")).replace("sub-", "").strip()

    lbl_l_sha = hashlib.sha256(lbl_l.read_bytes()).hexdigest()
    lbl_r_sha = hashlib.sha256(lbl_r.read_bytes()).hexdigest()

    # Check cache validity against output SHA and upstream label SHAs
    if not force and cached_png.is_file() and sources_json.is_file():
        try:
            s_data = json.loads(sources_json.read_text(encoding="utf-8"))
            curr_sha = hashlib.sha256(cached_png.read_bytes()).hexdigest()
            if (
                s_data.get("renderer") == "brainspace_hippocampal_grid"
                and s_data.get("output_sha256") == curr_sha
                and s_data.get("left_label_sha256") == lbl_l_sha
                and s_data.get("right_label_sha256") == lbl_r_sha
            ):
                return Image.open(cached_png)
        except Exception:
            pass

    # Render the selected hippocampal surfaces with BrainSpace.
    render_out = temp_dir / f"render_{branch_id}_{modality}_{smoothing}"
    res = render_hippocampal_grid(
        subject=subject,
        outdir=render_out,
        name=f"comb_{branch_id}_{modality}_{smoothing}",
        title=f"{branch['cohort_name']} {modality}",
        left_labels=lbl_l,
        right_labels=lbl_r,
        left_surface=surf_l,
        right_surface=surf_r,
    )
    native_im = Image.open(res["surface_png"]).convert("RGBA")
    comp = build_2x2_composite(native_im)

    # Save to panels directory with provenance
    panels_dir.mkdir(parents=True, exist_ok=True)
    comp.save(cached_png)
    out_sha = hashlib.sha256(cached_png.read_bytes()).hexdigest()
    provenance = {
        "branch_id": branch_id,
        "modality": modality,
        "smoothing": smoothing,
        "renderer": "brainspace_hippocampal_grid",
        "left_manifest": str(ml_path),
        "right_manifest": str(mr_path),
        "left_label": str(lbl_l),
        "right_label": str(lbl_r),
        "left_label_sha256": lbl_l_sha,
        "right_label_sha256": lbl_r_sha,
        "left_surface": str(surf_l),
        "right_surface": str(surf_r),
        "output_png": str(cached_png),
        "output_sha256": out_sha,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    sources_json.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return comp


# -----------------------------------------------------------------------------
# Subfigure Rendering Functions: Exact Reference Layouts, Ratios & Height Matching
# -----------------------------------------------------------------------------

# --- 1. Subfigure 1: Gradient Fisher-z (plot_multidataset_cluster_positive_stacked.py layout) ---
def render_subfig_fisherz_row(
    fig: plt.Figure,
    gs_sub,
    row_idx: int,
    bd: dict[str, Any],
    smoothing: str,
    is_bottom: bool,
    *,
    data_key: str = "fisherz",
    surf_key: str = "surf_fz",
    scale_key: str = "z",
    x_label: str = "Positive FC (Fisher-z)",
    show_badge: bool = True,
):
    """Render one branch row of positive stacked cluster profiles (Fisher-z FC or OvR coefficients)."""
    binfo = bd["info"]
    fz_l = bd[data_key]["L"]
    fz_r = bd[data_key]["R"]
    surf_img = bd[surf_key]
    max_xlim, x_ticks = bd[f"max_{scale_key}"], bd[f"{scale_key}_ticks"]

    ax_left = fig.add_subplot(gs_sub[row_idx, 0])
    ax_mid = fig.add_subplot(gs_sub[row_idx, 1])
    ax_right = fig.add_subplot(gs_sub[row_idx, 2])

    # Background shading for cohort
    for ax in (ax_left, ax_mid, ax_right):
        ax.set_facecolor(binfo["bg_color"])

    y_pos = np.arange(1, len(fz_l) + 1)
    bar_height = 0.65

    # Left & Right stacked horizontal bars (Anterior at top, Posterior at bottom)
    for side, ax, clist in (("left", ax_left, fz_l), ("right", ax_right, fz_r)):
        for idx, cinfo in enumerate(clist):
            y = y_pos[idx]
            cum_x = 0.0
            for net_name, z_val in cinfo["pos_networks"]:
                color = NETWORK_COLORS.get(net_name, "#888888")
                ax.barh(
                    y,
                    z_val,
                    left=cum_x,
                    height=bar_height,
                    color=color,
                    alpha=0.92,
                    edgecolor="#222222",
                    lw=0.9,
                    zorder=3,
                )
                if z_val >= 0.015 * (max_xlim / 0.20):
                    abbrev = ABBREV_MAP.get(net_name, net_name[:3])
                    cx = cum_x + z_val / 2.0
                    txt = ax.text(
                        cx,
                        y,
                        abbrev,
                        ha="center",
                        va="center",
                        fontsize=13.5,
                        fontweight="bold",
                        color="#ffffff",
                        zorder=5,
                    )
                    txt.set_path_effects([pe.withStroke(linewidth=2.2, foreground="#111111")])
                cum_x += z_val

        # Invert Y so C1 (Anterior) is at the top, C3 (Posterior) at bottom
        ax.set_ylim(len(clist) + 0.55, 0.45)
        ax.set_yticks(y_pos)

    # Left Subplot Settings (Inverted x-axis from max_xlim to 0.0)
    ax_left.set_xlim(max_xlim, 0.0)
    ax_left.set_xticks(x_ticks)
    # Row labels (C1/C2/C3) appear once, in the first column with the branch badge.
    ax_left.set_yticklabels(["C1 (Ant)", "C2 (Mid)", "C3 (Post)"] if show_badge else [], fontsize=14.0, fontweight="bold", color="#1e293b")
    ax_left.tick_params(axis="y", left=False, pad=8)
    ax_left.tick_params(axis="x", labelsize=13.0, colors="#334155")
    ax_left.grid(True, axis="x", color="#e2e2e2", lw=0.7, ls=":", zorder=1)
    ax_left.spines[["top", "left"]].set_visible(False)
    ax_left.spines["right"].set_color("#222222")
    ax_left.spines["right"].set_linewidth(1.5)

    # Right Subplot Settings (Normal x-axis from 0.0 to max_xlim)
    ax_right.set_xlim(0.0, max_xlim)
    ax_right.set_xticks(list(reversed(x_ticks)))
    ax_right.set_yticklabels([])
    ax_right.tick_params(axis="y", left=False)
    ax_right.tick_params(axis="x", labelsize=13.0, colors="#334155")
    ax_right.grid(True, axis="x", color="#e2e2e2", lw=0.7, ls=":", zorder=1)
    ax_right.spines[["top", "right"]].set_visible(False)
    ax_right.spines["left"].set_color("#222222")
    ax_right.spines["left"].set_linewidth(1.5)

    if row_idx == 0:
        ax_left.set_title("Left Hipp. (hemi-L)", fontsize=16.5, fontweight="heavy", color="#0b2447", pad=8)
        ax_right.set_title("Right Hipp. (hemi-R)", fontsize=16.5, fontweight="heavy", color="#0b2447", pad=8)
        ax_mid.set_title("Cluster labels (2x2 view)", fontsize=15.5, fontweight="heavy", color="#0b2447", pad=8)

    if is_bottom:
        ax_left.set_xlabel(x_label, fontsize=15.0, fontweight="heavy", color="#0b2447", labelpad=8)
        ax_right.set_xlabel(x_label, fontsize=15.0, fontweight="heavy", color="#0b2447", labelpad=8)

    # Center Subplot Settings: 2x2 3D surface overlay expanding to fill full row height
    ax_mid.axis("off")
    ax_mid.imshow(surf_img, aspect="equal")

    if not show_badge:
        return

    # Cohort Label Badge on the Left of Left Subplot
    badge_txt = f"{binfo['cohort_name']}\n{smoothing}"
    ax_left.text(
        -0.40,
        0.50,
        badge_txt,
        va="center",
        ha="center",
        fontsize=13.0,
        fontweight="heavy",
        color=binfo["badge_fg"],
        transform=ax_left.transAxes,
        bbox=dict(
            boxstyle="round,pad=0.42,rounding_size=0.25",
            fc=binfo["badge_bg"],
            ec="none",
        ),
    )


# --- 2. Subfigure 2: Gradient Top-20 Count (render_multidataset_gradient_subregion_top20_stacked_2col.py) ---
def render_subfig_top20_row(
    fig: plt.Figure,
    gs_sub,
    row_idx: int,
    bd: dict[str, Any],
    smoothing: str,
    is_bottom: bool,
):
    """Render a single branch row for Subfig 2 with exact height matching."""
    binfo = bd["info"]
    top20_l = bd["top20"]["L"]
    top20_r = bd["top20"]["R"]
    surf_img = bd["surf_top20"]

    ax_meta = fig.add_subplot(gs_sub[row_idx, 0])
    ax_surf = fig.add_subplot(gs_sub[row_idx, 1])
    ax_left = fig.add_subplot(gs_sub[row_idx, 2])
    ax_right = fig.add_subplot(gs_sub[row_idx, 3])

    # 1. Metadata Card
    ax_meta.axis("off")
    ax_meta.add_patch(
        patches.FancyBboxPatch(
            (0.02, 0.02),
            0.96,
            0.96,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            fc=binfo["bg_color"],
            ec="#cbd2d9",
            lw=1.5,
            transform=ax_meta.transAxes,
            zorder=0,
        )
    )
    ax_meta.text(
        0.50,
        0.82,
        binfo["dataset_name"],
        fontsize=16.0,
        fontweight="heavy",
        ha="center",
        va="center",
        color=binfo["badge_fg"],
        transform=ax_meta.transAxes,
        bbox=dict(boxstyle="round,pad=0.38,rounding_size=0.30", fc=binfo["badge_bg"], ec="none"),
    )
    ax_meta.text(
        0.50,
        0.54,
        binfo["cohort_name"],
        fontsize=16.0,
        fontweight="heavy",
        ha="center",
        va="center",
        color="#0b2447",
        transform=ax_meta.transAxes,
    )
    ax_meta.text(
        0.50,
        0.30,
        f"N = {binfo['n_subjects']}",
        fontsize=16.0,
        fontweight="bold",
        ha="center",
        va="center",
        color="#334e68",
        transform=ax_meta.transAxes,
    )
    ax_meta.text(
        0.50,
        0.12,
        f"{smoothing} smoothing",
        fontsize=14.0,
        fontweight="bold",
        ha="center",
        va="center",
        color="#64748b",
        transform=ax_meta.transAxes,
    )

    # 2. 2x2 3D Surface expanding to fill full row height
    ax_surf.axis("off")
    ax_surf.imshow(surf_img, aspect="equal")

    # 3. Left and Right Subregion Bars
    y_pos = np.arange(1, len(top20_l) + 1)
    bar_height = 0.62

    for side, ax, slist in (("left", ax_left, top20_l), ("right", ax_right, top20_r)):
        for s_idx, sinfo in enumerate(slist):
            y = y_pos[s_idx]
            comp = sinfo["comp"]
            active = [(net, pct) for net, pct in comp.items() if pct > 0.001]
            active.sort(key=lambda x: (x[1], -STACK_ORDER.index(x[0]) if x[0] in STACK_ORDER else -99), reverse=True)
            cum_pct = 0.0
            for net, val in active:
                color = NETWORK_COLORS.get(net, "#888888")
                abbrev = ABBREV_MAP.get(net, net)
                ax.barh(y, val, left=cum_pct, height=bar_height, color=color, edgecolor="#ffffff", linewidth=1.1, zorder=3)
                if val >= 7.0:
                    cx = cum_pct + val / 2.0
                    label_txt = f"{abbrev}\n{val:.0f}%" if val >= 14.0 else abbrev
                    fs = 13.5 if val >= 14.0 else 11.5
                    txt = ax.text(cx, y, label_txt, ha="center", va="center", fontsize=fs, fontweight="heavy", color="#ffffff", zorder=5)
                    txt.set_path_effects([pe.Stroke(linewidth=2.4, foreground="#111111"), pe.Normal()])
                cum_pct += val

        ax.set_xlim(0, 100)
        # Inverted so Ant (1) at top, Post (3) at bottom
        ax.set_ylim(len(slist) + 0.55, 0.45)
        ax.set_yticks([1, 2, 3])
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.grid(True, axis="x", color="#d8d8d8", linestyle="--", linewidth=0.8, alpha=0.85, zorder=1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines["left"].set_color("#333333")
        ax.spines["left"].set_linewidth(1.4)
        ax.spines["bottom"].set_color("#333333")
        ax.spines["bottom"].set_linewidth(1.4)
        ax.tick_params(direction="out", length=4, width=1.2, pad=6)

        if is_bottom:
            first = "0%" if side == "left" else ""
            ax.set_xticklabels([first, "25%", "50%", "75%", "100%"], fontsize=13.5, fontweight="bold", color="#111111")
            ax.set_xlabel("Normalized Top-20 Representation (%)", fontsize=15.0, fontweight="bold", color="#0b2447", labelpad=8)
        else:
            ax.set_xticklabels([])

    ax_left.set_yticklabels(["Ant", "Middle", "Post"], fontsize=14.5, fontweight="bold", color="#111111", rotation=90, va="center", ha="center")
    ax_right.set_yticklabels([])

    if row_idx == 0:
        ax_meta.set_title("Branch", fontsize=16.5, fontweight="heavy", color="#102a43", pad=8)
        ax_surf.set_title("Labels (2x2 view)", fontsize=16.5, fontweight="heavy", color="#102a43", pad=8)
        ax_left.set_title("Left Hipp.", fontsize=16.5, fontweight="heavy", color="#0f4c81", pad=8)
        ax_right.set_title("Right Hipp.", fontsize=16.5, fontweight="heavy", color="#0f4c81", pad=8)


# --- 3. Subfigure 3: WTA Long-Axis (render_multidataset_wta_normalized_longaxis_stacked_2col.py) ---
def render_subfig_wta_row(
    fig: plt.Figure,
    gs_sub,
    row_idx: int,
    bd: dict[str, Any],
    smoothing: str,
    is_bottom: bool,
):
    """Render a single branch row for Subfig 3 with exact height matching."""
    binfo = bd["info"]
    wta_l = bd["wta"]["L"]
    wta_r = bd["wta"]["R"]
    surf_img = bd["surf_wta"]

    ax_meta = fig.add_subplot(gs_sub[row_idx, 0])
    ax_surf = fig.add_subplot(gs_sub[row_idx, 1])
    ax_spacer = fig.add_subplot(gs_sub[row_idx, 2])
    ax_left = fig.add_subplot(gs_sub[row_idx, 3])
    ax_right = fig.add_subplot(gs_sub[row_idx, 4])

    ax_spacer.axis("off")

    # 1. Metadata Card
    ax_meta.set_facecolor(binfo["bg_color"])
    for spine in ax_meta.spines.values():
        spine.set_edgecolor("#cbd5e1")
        spine.set_linewidth(1.2)
    ax_meta.set_xticks([])
    ax_meta.set_yticks([])

    badge_patch = patches.FancyBboxPatch(
        (0.06, 0.72),
        0.88,
        0.22,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        facecolor=binfo["badge_bg"],
        edgecolor="none",
        transform=ax_meta.transAxes,
        zorder=2,
    )
    ax_meta.add_patch(badge_patch)
    ax_meta.text(0.50, 0.83, binfo["dataset_name"], fontsize=16.0, fontweight="bold", ha="center", va="center", color=binfo["badge_fg"], transform=ax_meta.transAxes, zorder=3)
    ax_meta.text(0.50, 0.50, binfo["cohort_name"], fontsize=16.0, fontweight="bold", ha="center", va="center", color="#0f172a", transform=ax_meta.transAxes)
    ax_meta.text(0.50, 0.30, f"N = {binfo['n_subjects']}", fontsize=16.0, fontweight="bold", ha="center", va="center", color="#334e68", transform=ax_meta.transAxes)
    ax_meta.text(0.50, 0.12, f"{smoothing} smoothing", fontsize=14.0, fontweight="bold", ha="center", va="center", color="#64748b", transform=ax_meta.transAxes)

    # 2. 2x2 3D Surface expanding to fill full row height
    ax_surf.axis("off")
    ax_surf.imshow(surf_img, aspect="equal")

    # 3. Left and Right WTA 20-Bin Bars
    n_bins = len(wta_l)
    y_pos = np.arange(1, n_bins + 1)
    bar_height = 0.82

    for side, ax, blist in (("left", ax_left, wta_l), ("right", ax_right, wta_r)):
        for b_idx, comp in enumerate(blist):
            y = y_pos[b_idx]
            active = sorted([(net, pct) for net, pct in comp.items() if pct > 0.001], key=lambda item: (-item[1], STACK_ORDER.index(item[0]) if item[0] in STACK_ORDER else 99))
            cum_pct = 0.0
            for net_name, pct in active:
                color = NETWORK_COLORS.get(net_name, "#888888")
                ax.barh(y, pct, left=cum_pct, height=bar_height, color=color, edgecolor="#222222", linewidth=0.5, zorder=2)
                if pct >= 11.0:
                    cx = cum_pct + pct / 2.0
                    abbrev = ABBREV_MAP.get(net_name, net_name[:3].upper())
                    txt = ax.text(cx, y, abbrev, ha="center", va="center", fontsize=10.0, fontweight="bold", color="#ffffff", zorder=4)
                    txt.set_path_effects([pe.Stroke(linewidth=1.8, foreground="#111111"), pe.Normal()])
                cum_pct += pct

        ax.set_xlim(0, 100)
        # Inverted so Bin 1 (Ant) is at top, Bin 20 (Post) at bottom
        ax.set_ylim(n_bins + 0.55, 0.45)
        ax.set_yticks([1, 5, 10, 15, 20])
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.tick_params(axis="both", which="both", length=2.5, width=0.8, direction="out", color="#334e68")
        for spine in ax.spines.values():
            spine.set_edgecolor("#9fb3c8")
            spine.set_linewidth(0.8)
        ax.set_facecolor("#fafbfc")

        if is_bottom:
            ax.set_xlabel("Vertices in Bin (%)", fontsize=15.0, fontweight="bold", color="#102a43", labelpad=8)
            first = "0%" if side == "left" else ""
            ax.set_xticklabels([first, "25%", "50%", "75%", "100%"], fontsize=13.5, fontweight="bold", color="#243b53")
        else:
            ax.set_xticklabels([])

    ax_left.set_ylabel("Long-Axis Bin", fontsize=14.0, fontweight="bold", color="#102a43", labelpad=6)
    ax_left.set_yticklabels(["1 (Ant)", "5", "10", "15", "20 (Post)"], fontsize=12.5, fontweight="bold", color="#243b53")
    ax_right.set_yticklabels([])

    if row_idx == 0:
        ax_meta.set_title("Branch", fontsize=16.5, fontweight="heavy", color="#102a43", pad=8)
        ax_surf.set_title("Labels (2x2 view)", fontsize=16.5, fontweight="heavy", color="#102a43", pad=8)
        ax_left.set_title("Left Hipp.", fontsize=16.5, fontweight="heavy", color="#0f4c81", pad=8)
        ax_right.set_title("Right Hipp.", fontsize=16.5, fontweight="heavy", color="#0f4c81", pad=8)


# -----------------------------------------------------------------------------
# Master Figure Assembly
# -----------------------------------------------------------------------------
def render_hcp_cortex_control_gradient_wta(
    out_path: Path,
    tsv_path: Path | None = None,
    force: bool = False,
):
    """Render Standard, CtxReg 10 mm and CtxReg 20 mm rows for 2mm and 4mm hippocampal smoothing."""
    print("\n[RENDER] Building HCP local cortex regression comparison figure...")

    panels_dir = DEFAULT_DIR / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data & render 2x2 3D surfaces: one row per (smoothing, branch)
    branch_data: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        for smoothing in SMOOTHINGS:
            for b in BRANCHES:
                print(f"  -> Loading & rendering assets for {b['cohort_name']} ({smoothing})...")
                fz_data = load_branch_fisherz(b, smoothing)
                ovr_data = load_branch_ovr(b, smoothing)
                top20_data = load_branch_top20_count(b, smoothing)
                wta_data = load_branch_wta(b, smoothing)

                surf_fz = render_branch_surface_2x2(b, "fisherz", smoothing, temp_dir, panels_dir, force=force)
                surf_ovr = render_branch_surface_2x2(b, "ovr_network_ts", smoothing, temp_dir, panels_dir, force=force)
                surf_top20 = render_branch_surface_2x2(b, "top20_count", smoothing, temp_dir, panels_dir, force=force)
                surf_wta = render_branch_surface_2x2(b, "wta", smoothing, temp_dir, panels_dir, force=force)

                # Compute row-specific dynamic range for Fisher-z
                max_z, z_ticks = adaptive_positive_axis_scale(fz_data["L"], fz_data["R"])
                max_beta, beta_ticks = adaptive_positive_axis_scale(ovr_data["L"], ovr_data["R"])

                branch_data.append({
                    "info": b,
                    "smoothing": smoothing,
                    "fisherz": fz_data,
                    "ovr": ovr_data,
                    "top20": top20_data,
                    "wta": wta_data,
                    "surf_fz": surf_fz,
                    "surf_ovr": surf_ovr,
                    "surf_top20": surf_top20,
                    "surf_wta": surf_wta,
                    "max_z": max_z,
                    "z_ticks": z_ticks,
                    "max_beta": max_beta,
                    "beta_ticks": beta_ticks,
                })

    # Save TSV statistics if requested
    if tsv_path is not None:
        tsv_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "branch\tsmoothing\tmodality\themisphere\tcluster_or_bin_id\tregion_name\tnetwork_name\tvalue_or_pct\ttop1_network\tmargin"
        ]
        for bd in branch_data:
            b_id = bd["info"]["id"]
            smoothing = bd["smoothing"]
            # Fisher-z
            for hemi in ("L", "R"):
                for c in bd["fisherz"][hemi]:
                    cid = c["cluster_id"]
                    cname = c["cluster_name"]
                    top1 = c["top1_network"]
                    m = c["margin"]
                    for net, val in c["pos_networks"]:
                        lines.append(f"{b_id}\t{smoothing}\tgradient_fisherz\t{hemi}\t{cid}\t{cname}\t{net}\t{val:.6f}\t{top1}\t{m:.6f}")
            # OvR (positive raw coefficients; margin column holds the winning coefficient)
            for hemi in ("L", "R"):
                for c in bd["ovr"][hemi]:
                    for net, val in c["pos_networks"]:
                        lines.append(f"{b_id}\t{smoothing}\tgradient_ovr_network_ts\t{hemi}\t{c['cluster_id']}\t{c['cluster_name']}\t{net}\t{val:.6f}\t{c['top1_network']}\t{c['top1_beta']:.6f}")
            # Top20
            for hemi in ("L", "R"):
                for s in bd["top20"][hemi]:
                    cid = s["cluster_id"]
                    sname = s["subregion_name"]
                    top1 = s["top1_network"]
                    m = s["margin_count"]
                    for net, pct in s["comp"].items():
                        lines.append(f"{b_id}\t{smoothing}\tgradient_top20\t{hemi}\t{cid}\t{sname}\t{net}\t{pct:.4f}\t{top1}\t{m}")
            # WTA
            for hemi in ("L", "R"):
                for b_idx, comp in enumerate(bd["wta"][hemi], start=1):
                    for net, pct in comp.items():
                        lines.append(f"{b_id}\t{smoothing}\twta_longaxis\t{hemi}\t{b_idx}\tBin_{b_idx}\t{net}\t{pct:.4f}\t-\t-")
        tsv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  [TSV] Saved statistics to: {tsv_path}")

    # Master canvas: fixed header (2.25 in) and legend (1.35 in) zones plus ~4.8 in per
    # branch row, so the 2x2 hippocampal surface keeps the same height as the
    # stacked bars whatever the number of branch rows. Layout fractions below are
    # written for the three-row reference height (18 in) and rescaled here.
    n_rows = len(branch_data)
    fig_height = 2.25 + 4.8 * n_rows + 1.35
    scale = REFERENCE_FIG_HEIGHT_IN / fig_height

    def from_top(y: float) -> float:
        return 1.0 - (1.0 - y) * scale

    def from_bottom(y: float) -> float:
        return y * scale

    fig = plt.figure(figsize=(FIG_WIDTH_IN, fig_height), facecolor="#ffffff", dpi=220)
    xw = 49.0 / FIG_WIDTH_IN  # converts the three-column (49 in) horizontal fractions

    # -------------------------------------------------------------------------
    # 1. Non-Overlapping Header Zone
    # -------------------------------------------------------------------------
    fig.suptitle(
        "HCP 7T Hippocampal Organization with Local Cortex Regression (CtxReg 10 mm vs. 20 mm)",
        fontsize=38.0,
        fontweight="heavy",
        x=0.50,
        y=from_top(0.985),
        ha="center",
        va="top",
        color="#0b2447",
    )
    fig.text(
        0.50,
        from_top(0.940),
        f"Gradient clusters (TopSim profile, Ward, K = 3) labelled by Fisher-z FC, OvR logistic model and normalized top-20 count; winner-take-all labels; Kong2022 PFM canonical7 networks  |  rest, ICA-FIX, N = {HCP_COHORT_SIZE} young adults  |  top: 2mm, bottom: 4mm hippocampal smoothing",
        ha="center",
        va="top",
        fontsize=20.0,
        fontweight="medium",
        color="#334e68",
    )

    # -------------------------------------------------------------------------
    # 2. Main Master GridSpec: 3 Columns side-by-side
    # -------------------------------------------------------------------------
    top_margin = from_top(0.875)
    bottom_margin = from_bottom(0.075)

    gs_main = fig.add_gridspec(
        nrows=1,
        ncols=len(COLUMN_WIDTH_RATIOS),
        left=0.072 * xw,
        right=1.0 - 0.012 * xw,
        top=top_margin,
        bottom=bottom_margin,
        width_ratios=list(COLUMN_WIDTH_RATIOS),
        wspace=0.035,
    )

    # Subfigure section banners centred over each column
    y_subfig_title = from_top(0.916)
    banners = (
        ("GRADIENT CLUSTERS: POSITIVE FISHER-Z FC (ROW-SPECIFIC SCALE)", "#0b3c5d"),
        ("GRADIENT CLUSTERS: OvR LOGISTIC, NETWORK TIME-SERIES FC (ROW-SPECIFIC SCALE)", "#3b3b98"),
        ("GRADIENT CLUSTERS: NORMALIZED TOP-20 ROI COUNT (%)", "#1d5b79"),
        ("WINNER-TAKE-ALL LABELS: 20 LONG-AXIS BINS (%)", "#8c4600"),
    )
    for col, (text, color) in enumerate(banners):
        pos = gs_main[0, col].get_position(fig)
        fig.text(0.5 * (pos.x0 + pos.x1), y_subfig_title, text, fontsize=17.0, fontweight="heavy", color=color, ha="center", va="top")

    # Subgridspec for each subfigure: one row per (smoothing, branch)
    gs_sub1 = gs_main[0, 0].subgridspec(
        nrows=n_rows,
        ncols=3,
        width_ratios=[0.41, 0.18, 0.41],
        wspace=0.012,
        hspace=0.14,
    )

    gs_sub_ovr = gs_main[0, 1].subgridspec(
        nrows=n_rows,
        ncols=3,
        width_ratios=[0.41, 0.18, 0.41],
        wspace=0.012,
        hspace=0.14,
    )

    gs_sub2 = gs_main[0, 2].subgridspec(
        nrows=n_rows,
        ncols=4,
        width_ratios=[0.12, 0.18, 0.35, 0.35],
        wspace=0.024,
        hspace=0.14,
    )

    gs_sub3 = gs_main[0, 3].subgridspec(
        nrows=n_rows,
        ncols=5,
        width_ratios=[0.11, 0.15, 0.085, 0.3275, 0.3275],
        wspace=0.015,
        hspace=0.12,
    )

    # Render each row across all 3 subfigures
    print("  [RENDER] Drawing Subfigure 1: Gradient Fisher-z...")
    for row_idx, bd in enumerate(branch_data):
        is_bottom = (row_idx == len(branch_data) - 1)
        render_subfig_fisherz_row(fig, gs_sub1, row_idx, bd, bd["smoothing"], is_bottom)

    print("  [RENDER] Drawing Subfigure 2: Gradient OvR...")
    for row_idx, bd in enumerate(branch_data):
        is_bottom = (row_idx == len(branch_data) - 1)
        render_subfig_fisherz_row(
            fig, gs_sub_ovr, row_idx, bd, bd["smoothing"], is_bottom,
            data_key="ovr", surf_key="surf_ovr", scale_key="beta",
            x_label="Positive OvR coefficient (raw β)", show_badge=False,
        )

    print("  [RENDER] Drawing Subfigure 3: Gradient Top-20 Count...")
    for row_idx, bd in enumerate(branch_data):
        is_bottom = (row_idx == len(branch_data) - 1)
        render_subfig_top20_row(fig, gs_sub2, row_idx, bd, bd["smoothing"], is_bottom)

    print("  [RENDER] Drawing Subfigure 4: WTA Long-Axis...")
    for row_idx, bd in enumerate(branch_data):
        is_bottom = (row_idx == len(branch_data) - 1)
        render_subfig_wta_row(fig, gs_sub3, row_idx, bd, bd["smoothing"], is_bottom)

    # Separator between the 2mm and 4mm smoothing blocks
    rows_per_block = len(BRANCHES)
    for boundary in range(rows_per_block, n_rows, rows_per_block):
        y_sep = 0.5 * (gs_sub2[boundary - 1, 2].get_position(fig).y0 + gs_sub2[boundary, 2].get_position(fig).y1)
        fig.add_artist(plt.Line2D([0.012 * xw, 1.0 - 0.012 * xw], [y_sep, y_sep], transform=fig.transFigure, color="#334e68", lw=2.0, ls="--"))

    # -------------------------------------------------------------------------
    # 3. Global Bottom Legend: Canonical 7 Networks
    # -------------------------------------------------------------------------
    legend_nets = [
        ("Default", "Default (DMN)"),
        ("Somatomotor", "Somatomotor (SMN)"),
        ("VentralAttention", "Ventral Attention (VAN)"),
        ("DorsalAttention", "Dorsal Attention (DAN)"),
        ("Control", "Control (CON)"),
        ("Language", "Language (LAN)"),
        ("Visual", "Visual (VN)"),
    ]
    y_leg = from_bottom(0.024)
    dx_leg = 0.110 * xw
    x_leg_start = 0.5 - 0.5 * (dx_leg * (len(legend_nets) - 1) + 0.019 * xw + 0.075 * xw)
    for idx, (net_key, net_label) in enumerate(legend_nets):
        col = NETWORK_COLORS.get(net_key, "#888888")
        x_pos = x_leg_start + idx * dx_leg
        rect = patches.Rectangle(
            (x_pos, y_leg - 0.007 * scale),
            0.015 * xw,
            0.014 * scale,
            facecolor=col,
            edgecolor="#222222",
            linewidth=1.0,
            transform=fig.transFigure,
            figure=fig,
            zorder=10,
        )
        fig.patches.append(rect)
        fig.text(
            x_pos + 0.019 * xw,
            y_leg,
            net_label,
            ha="left",
            va="center",
            fontsize=16.0,
            fontweight="bold",
            color="#0f172a",
        )

    # Save master high-resolution figure
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [SAVING] Writing master figure to: {out_path}...")
    fig.savefig(out_path, dpi=220, facecolor="#ffffff", edgecolor="none")
    plt.close(fig)
    print(f"  [DONE] Successfully saved: {out_path}")


# -----------------------------------------------------------------------------
# Entrypoint for suite and CLI
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Render the HCP Standard vs. CtxReg 10 mm vs. CtxReg 20 mm comparison (2mm and 4mm).")
    parser.add_argument("--out-png", type=Path, default=DEFAULT_OUT_PNG, help="Output PNG path")
    parser.add_argument("--stats-tsv", type=Path, default=DEFAULT_STATS_TSV, help="Output statistics TSV path")
    parser.add_argument("--force", action="store_true", help="Force re-rendering of cached surface panels")
    args = parser.parse_args()
    render_hcp_cortex_control_gradient_wta(out_path=args.out_png, tsv_path=args.stats_tsv, force=args.force)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Cross-Dataset Hippocampal 2mm Gradient Atlas & Subregion Top-20 Network Composition (2-Column Layout).

Supports rendering both:
  1. Raw Top-20 Count (Raw percentage of Top-20 ROIs)
  2. Normalized Top-20 Count (Top-20 count normalized by total network ROI capacity)
with substantially enlarged typography across the entire layout.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
CORE_DIR = REPO_ROOT / "scripts" / "core_workflow"
VIS_LIB_DIR = REPO_ROOT / "scripts" / "visualize" / "lib"
for p in (CORE_DIR, VIS_LIB_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gradient_surface_panel import render_gradient_surface_panel
from lib.gradient_assignment_contract import validate_gradient_assignment
from multidataset_cohorts import (
    COMPARISON_METHODS,
    COMPARISON_VERSION,
    GRID_VARIANTS,
    SIGNAL_BRANCHES,
    GridSpec,
    format_comparison_column_title,
    get_comparison_output_tag,
    get_comparison_visual_columns,
    get_multidataset_cohorts,
)
from surface_panel_cache import fingerprint

# Canonical network styling and color definitions
NETWORK_COLORS = {
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

ABBREV_MAP = {
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

# Kong2022 400-ROI canonical8 total ROI capacity (for opportunity normalization)
KONG2022_CANONICAL8_TOTAL_ROIS = {
    "Default": 77,
    "Control": 73,
    "Visual": 61,
    "VentralAttention": 52,
    "DorsalAttention": 51,
    "Somatomotor": 49,
    "Auditory": 19,
    "Language": 18,
}

# Kong2022 400-ROI canonical7 total ROI capacity (Auditory merged into Somatomotor: 49 + 19 = 68)
KONG2022_CANONICAL7_TOTAL_ROIS = {
    "Default": 77,
    "Control": 73,
    "Visual": 61,
    "VentralAttention": 52,
    "DorsalAttention": 51,
    "Somatomotor": 68,
    "Language": 18,
}

# Preferred stacking order
STACK_ORDER = [
    "Default",
    "Somatomotor",
    "Auditory",  # retained only for explicit canonical8 inputs
    "VentralAttention",
    "DorsalAttention",
    "Control",
    "Language",
    "Visual",
    "non-network/null",
]

SUBREGION_NAMES = {
    1: "Anterior",
    2: "Body",
    3: "Posterior",
}
NORMALIZED_TOP20_ASSIGNMENT_METRIC = "top20_count_normalized_by_network_roi_count"
SUBREGION_METRIC_NORMALIZED_TOP20 = "normalized_top20_count"
SUBREGION_METRIC_FISHERZ_PROFILE = "fisherz_profile"


def resolve_subregion_metric_mode(
    cohort_entry: dict,
    *,
    requested_normalized: bool,
) -> str:
    """Resolve the declared subregion metric without cross-method fallback."""
    method = cohort_entry.get("method")
    if method in ("kong2022_7net", "lynch2024_7net"):
        expected = (
            SUBREGION_METRIC_NORMALIZED_TOP20
            if method == "kong2022_7net"
            else SUBREGION_METRIC_FISHERZ_PROFILE
        )
        declared = cohort_entry.get("subregion_metric")
        if declared != expected:
            raise ValueError(
                f"Comparison method {method!r} must declare subregion_metric={expected!r}; "
                f"got {declared!r}"
            )
        return declared
    return (
        SUBREGION_METRIC_NORMALIZED_TOP20 if requested_normalized else "raw_top20_count"
    )


def subregion_metric_label(metric_mode: str) -> str:
    """Return the human-readable metric label used in figure metadata."""
    labels = {
        SUBREGION_METRIC_NORMALIZED_TOP20: "Normalized Top-20 Count",
        SUBREGION_METRIC_FISHERZ_PROFILE: "Positive Fisher-z Profile",
        "raw_top20_count": "Raw Top-20 Count",
    }
    try:
        return labels[metric_mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported subregion metric mode: {metric_mode!r}") from exc


def subregion_bar_column_label(hemi: str, metric_mode: str) -> str:
    """Return the concise left/right hippocampus label above each bar panel."""
    return "Left Hipp." if hemi == "L" else "Right Hipp."


def gradient_atlas_column_label(metric_mode: str) -> str:
    """Return a metric-only header for the atlas surface column."""
    return subregion_metric_label(metric_mode)


# Cohort specifications for 2-column layout
COHORTS_COL1 = [
    {
        "dataset_name": "HCP 7T",
        "cohort_name": "HCP 7T",
        "cohort_desc": "Healthy Young Adults",
        "n_subjects": 151,
        "group_root": REPO_ROOT
        / "outputs_hcp/work/group_atlas/pfm",
        "present_dir": REPO_ROOT / "outputs_hcp/present/overviews/pfm",
        "bg_color": "#ffffff",
        "badge_bg": "#0b3c5d",
        "badge_fg": "#ffffff",
    },
    {
        "dataset_name": "Iowa 3T",
        "cohort_name": "HC",
        "cohort_desc": "Healthy Controls",
        "n_subjects": 67,
        "group_root": REPO_ROOT
        / "outputs_iowa_jj_tms/work/group_atlas/pfm/hc",
        "present_dir": REPO_ROOT
        / "outputs_iowa_jj_tms/present/overviews/pfm/hc",
        "bg_color": "#f7faff",
        "badge_bg": "#1d5b79",
        "badge_fg": "#ffffff",
    },
    {
        "dataset_name": "Iowa 3T",
        "cohort_name": "Patient",
        "cohort_desc": "Depression Patients",
        "n_subjects": 27,
        "group_root": REPO_ROOT
        / "outputs_iowa_jj_tms/work/group_atlas/pfm/patient",
        "present_dir": REPO_ROOT
        / "outputs_iowa_jj_tms/present/overviews/pfm/patient",
        "bg_color": "#f7faff",
        "badge_bg": "#1d5b79",
        "badge_fg": "#ffffff",
    },
    {
        "dataset_name": "Stanford",
        "cohort_name": "NTHC",
        "cohort_desc": "Non-Trauma Controls",
        "n_subjects": 51,
        "group_root": REPO_ROOT
        / "outputs_stanford/work/group_atlas/pfm/nthc",
        "present_dir": REPO_ROOT
        / "outputs_stanford/present/overviews/pfm/nthc",
        "bg_color": "#fffdf8",
        "badge_bg": "#8c4600",
        "badge_fg": "#ffffff",
    },
    {
        "dataset_name": "Stanford",
        "cohort_name": "TEHC",
        "cohort_desc": "Trauma-Exposed Controls",
        "n_subjects": 40,
        "group_root": REPO_ROOT
        / "outputs_stanford/work/group_atlas/pfm/tehc",
        "present_dir": REPO_ROOT
        / "outputs_stanford/present/overviews/pfm/tehc",
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
        "group_root": REPO_ROOT
        / "outputs_stanford/work/group_atlas/pfm/nts",
        "present_dir": REPO_ROOT / "outputs_stanford/present/overviews/pfm/nts",
        "bg_color": "#fffdf8",
        "badge_bg": "#8c4600",
        "badge_fg": "#ffffff",
    },
    {
        "dataset_name": "Stanford",
        "cohort_name": "NTS Any MDD",
        "cohort_desc": "Non-Trauma MDD (All)",
        "n_subjects": 43,
        "group_root": REPO_ROOT
        / "outputs_stanford/work/group_atlas/pfm/nts_any_mdd",
        "present_dir": REPO_ROOT
        / "outputs_stanford/present/overviews/pfm/nts_any_mdd",
        "bg_color": "#fffdf8",
        "badge_bg": "#8c4600",
        "badge_fg": "#ffffff",
    },
    {
        "dataset_name": "Stanford",
        "cohort_name": "NTS Pure MDD",
        "cohort_desc": "Non-Trauma Pure MDD",
        "n_subjects": 11,
        "group_root": REPO_ROOT
        / "outputs_stanford/work/group_atlas/pfm/nts_pure_mdd",
        "present_dir": REPO_ROOT
        / "outputs_stanford/present/overviews/pfm/nts_pure_mdd",
        "bg_color": "#fffdf8",
        "badge_bg": "#8c4600",
        "badge_fg": "#ffffff",
    },
    {
        "dataset_name": "Stanford",
        "cohort_name": "TIS",
        "cohort_desc": "Trauma-Informed Symptomatic",
        "n_subjects": 47,
        "group_root": REPO_ROOT
        / "outputs_stanford/work/group_atlas/pfm/tis",
        "present_dir": REPO_ROOT / "outputs_stanford/present/overviews/pfm/tis",
        "bg_color": "#fffdf8",
        "badge_bg": "#8c4600",
        "badge_fg": "#ffffff",
    },
    {
        "dataset_name": "Stanford",
        "cohort_name": "TIS PTSD",
        "cohort_desc": "Trauma-Informed PTSD",
        "n_subjects": 35,
        "group_root": REPO_ROOT
        / "outputs_stanford/work/group_atlas/pfm/tis_ptsd",
        "present_dir": REPO_ROOT
        / "outputs_stanford/present/overviews/pfm/tis_ptsd",
        "bg_color": "#fffdf8",
        "badge_bg": "#8c4600",
        "badge_fg": "#ffffff",
    },
]


def unify_surface_background(
    im: Image.Image, *, dark_threshold: int = 20
) -> Image.Image:
    """Convert border-connected legacy dark background pixels to canonical white."""
    arr = np.array(im)
    rgb = arr[..., :3]
    border_rgb = np.concatenate([rgb[0, :], rgb[-1, :], rgb[:, 0], rgb[:, -1]], axis=0)
    if np.mean(border_rgb) >= 128:
        return im

    dark_mask = np.all(rgb < dark_threshold, axis=-1)
    labeled, _ = ndi.label(dark_mask)
    border_labels = (
        set(np.unique(labeled[0, :]))
        | set(np.unique(labeled[-1, :]))
        | set(np.unique(labeled[:, 0]))
        | set(np.unique(labeled[:, -1]))
    )
    border_labels.discard(0)
    if not border_labels:
        return im

    background_mask = np.isin(labeled, list(border_labels))
    normalized = arr.copy()
    normalized[background_mask, :3] = 255
    if normalized.shape[-1] == 4:
        normalized[background_mask, 3] = 255
    return Image.fromarray(normalized)


def crop_gradient_surface_from_overview(
    overview_path: Path, *, smoothing: str
) -> Image.Image:
    """Select an exact overview row and crop its 860x1644 surface panel."""
    if smoothing not in ("2mm", "4mm"):
        raise ValueError(f"Unsupported smoothing {smoothing!r}: {overview_path}")
    im = Image.open(overview_path).convert("RGB")
    arr = np.array(im)

    if im.size == (4620, 1830):
        if smoothing != "2mm":
            raise ValueError(
                f"Requested 4mm but overview contains a single 2mm row: {overview_path}"
            )
        min_y, max_y = 180, 1824
        min_x = 2591
        max_x = min_x + 860
        return unify_surface_background(im.crop((min_x, min_y, max_x, max_y)))

    if im.size != (4620, 3660):
        raise ValueError(
            "Expected 4620x3660 or 4620x1830 gradient overview, "
            f"got {im.size}: {overview_path}"
        )

    # First use the legacy dark-panel geometry when it is present.
    row_mask = np.all(arr[:, 2500:] < 15, axis=-1).sum(axis=1) > 600
    edges = np.flatnonzero(np.diff(np.r_[False, row_mask, False]))
    bands = [(start, end) for start, end in edges.reshape(-1, 2) if end - start > 600]
    if len(bands) == 2 and all(end - start == 1646 for start, end in bands):
        start, end = bands[("2mm", "4mm").index(smoothing)]
        min_y, max_y = int(start + 1), int(end - 1)
        quad = arr[min_y:max_y, 2500:]
        black_mask = np.all(quad < 15, axis=-1)
        valid_cols = np.where(np.sum(black_mask, axis=0) > 600)[0]
        if len(valid_cols) == 0:
            raise ValueError(f"Missing {smoothing} surface panel: {overview_path}")
        min_x = 2500 + int(valid_cols[0]) + 1
        max_x = min_x + 860
        if max_x > im.width:
            raise ValueError(f"Incomplete {smoothing} surface panel: {overview_path}")
        return unify_surface_background(im.crop((min_x, min_y, max_x, max_y)))

    # White-background overviews retain the producer's fixed row geometry.
    canonical_bands = [(179, 1825), (2001, 3647)]
    has_content = [
        bool(np.any(arr[start:end, 2500:] < 240)) for start, end in canonical_bands
    ]
    if not all(has_content):
        present = [
            band for band, present in zip(canonical_bands, has_content) if present
        ]
        raise ValueError(
            "Expected two smoothing rows of height 1646, got "
            f"{present}: {overview_path}"
        )
    start, end = canonical_bands[("2mm", "4mm").index(smoothing)]
    min_x = 2591
    max_x = min_x + 860
    if max_x > im.width:
        raise ValueError(f"Incomplete {smoothing} surface panel: {overview_path}")
    return unify_surface_background(im.crop((min_x, start + 1, max_x, end - 1)))


def render_subregion_bar_axis(
    ax,
    subregion_entries: list[dict],
    hemi: str,
    normalized: bool = True,
    metric_mode: str | None = None,
    show_ylabel: bool = True,
    is_top_row: bool = False,
    is_bottom_row: bool = False,
):
    """Render a 3-bar subregion composition axis (Ant, Body, Post) with clean rotated labels."""
    n_subregions = len(subregion_entries)
    y_positions = np.arange(1, n_subregions + 1)
    bar_height = 0.62

    for s_idx, s_entry in enumerate(subregion_entries):
        y = y_positions[s_idx]
        comp = s_entry["comp"]
        current_left = 0.0

        # Sort networks by percentage value descending
        active_items = [
            (net, comp.get(net, 0.0)) for net in comp if comp.get(net, 0.0) > 1e-4
        ]
        sorted_items = sorted(
            active_items,
            key=lambda x: (
                x[1],
                -STACK_ORDER.index(x[0]) if x[0] in STACK_ORDER else -99,
            ),
            reverse=True,
        )

        for net, val in sorted_items:
            color = NETWORK_COLORS.get(net, "#888888")
            abbrev = ABBREV_MAP.get(net, net)

            rect = ax.barh(
                y,
                val,
                left=current_left,
                height=bar_height,
                color=color,
                edgecolor="#ffffff",
                linewidth=1.2,
                zorder=3,
            )

            # Label on segments: large font, bold with outline
            if val >= 7.0:
                cx = current_left + val / 2.0
                if val >= 14.0:
                    label_text = f"{abbrev}\n{val:.0f}%"
                    fs = 14.5
                else:
                    label_text = abbrev
                    fs = 13.0
                txt = ax.text(
                    cx,
                    y,
                    label_text,
                    ha="center",
                    va="center",
                    fontsize=fs,
                    fontweight="heavy",
                    color="#ffffff",
                    zorder=4,
                )
                txt.set_path_effects(
                    [
                        pe.Stroke(linewidth=2.8, foreground="#111111"),
                        pe.Normal(),
                    ]
                )

            current_left += val

    ax.set_xlim(0, 100)
    ax.set_ylim(n_subregions + 0.55, 0.45)  # Inverted so Anterior is at top

    # Y-axis ticks and labels: rotated 90 degrees, large font (17pt), centered
    ax.set_yticks([1, 2, 3])
    if show_ylabel:
        ax.set_yticklabels(
            ["Ant", "Body", "Post"],
            fontsize=17.0,
            fontweight="bold",
            color="#111111",
            rotation=90,
            va="center",
            ha="center",
        )
    else:
        ax.set_yticklabels([])

    # X-axis ticks: large font (16pt)
    ax.set_xticks([0, 25, 50, 75, 100])
    if is_bottom_row:
        ax.set_xticklabels(
            ["0%", "25%", "50%", "75%", "100%"],
            fontsize=16.0,
            fontweight="bold",
            color="#111111",
        )
        if metric_mode == SUBREGION_METRIC_FISHERZ_PROFILE:
            axis_title = "Positive Fisher-z Profile (%)"
        elif metric_mode == SUBREGION_METRIC_NORMALIZED_TOP20 or normalized:
            axis_title = "Normalized Top-20 Representation (%)"
        else:
            axis_title = "Raw Top-20 Composition (%)"
        ax.set_xlabel(
            axis_title, fontsize=18.0, fontweight="bold", color="#0b2447", labelpad=8
        )
    else:
        ax.set_xticklabels([])

    ax.grid(
        True,
        axis="x",
        color="#d8d8d8",
        linestyle="--",
        linewidth=0.8,
        alpha=0.85,
        zorder=1,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["left"].set_linewidth(1.4)
    ax.spines["bottom"].set_color("#333333")
    ax.spines["bottom"].set_linewidth(1.4)
    ax.tick_params(direction="out", length=5, width=1.4, pad=10)


def render_summary_card(ax, normalized: bool = True, is_7net: bool = False):
    """Render an informative summary overview card in the 5th cell of Column 2 with well-padded borders."""
    ax.axis("off")
    # Outer card box: top edge at y=0.80, leaving top 20% clear for Row 3 X-axis labels
    ax.add_patch(
        patches.FancyBboxPatch(
            (0.01, 0.02),
            0.98,
            0.78,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            fc="#f8fafc",
            ec="#94a3b8",
            lw=1.8,
            transform=ax.transAxes,
            zorder=0,
        )
    )

    # Header Banner
    ax.add_patch(
        patches.FancyBboxPatch(
            (0.02, 0.64),
            0.96,
            0.14,
            boxstyle="round,pad=0.01,rounding_size=0.03",
            fc="#1e293b",
            ec="none",
            transform=ax.transAxes,
            zorder=1,
        )
    )
    title_suffix = "(Normalized Count)" if normalized else "(Raw Count)"
    net_str = "7-Net" if is_7net else "8-Net"
    ax.text(
        0.50,
        0.71,
        f"Gradient-Spectral Parcellation & Subregion Top-20 Summary {title_suffix} ({net_str}, N = 528 across 12 Cohorts)",
        fontsize=17.5,
        fontweight="heavy",
        ha="center",
        va="center",
        color="#ffffff",
        transform=ax.transAxes,
        zorder=2,
    )

    left_x = 0.04
    ax.text(
        left_x,
        0.52,
        "• Cohort Structure: Left Column = Controls & References (6 cohorts, N=392)  |  Right Column = Clinical & Subgroups (6 cohorts, N=136)",
        fontsize=14.5,
        fontweight="bold",
        color="#0f172a",
        transform=ax.transAxes,
    )

    if normalized:
        if is_7net:
            norm_text = "• Normalization Method: Top-20 counts divided by total network ROI capacity (DMN=77, CON=73, VN=61, VAN=52, DAN=51, SMN=68 [Auditory merged], LAN=18)."
        else:
            norm_text = "• Normalization Method: Top-20 counts divided by total network ROI capacity (DMN=77, CON=73, VN=61, VAN=52, DAN=51, SMN=49, AUD=19, LAN=18)."
    else:
        norm_text = "• Raw Count Method: Direct count of ROIs among top-20 strongest connected cortical targets (each ROI = 5.0% of total)."

    ax.text(
        left_x,
        0.41,
        norm_text,
        fontsize=14.0,
        fontweight="medium",
        color="#334155",
        transform=ax.transAxes,
    )
    ax.text(
        left_x,
        0.29,
        "• Anterior Subregion: Marked enrichment of Ventral Attention (VAN, up to ~90% in 7T) and distinct Language (LAN) / DMN targets.",
        fontsize=14.0,
        fontweight="medium",
        color="#334155",
        transform=ax.transAxes,
    )
    ax.text(
        left_x,
        0.18,
        "• Body & Posterior: Body exhibits Somatomotor (SMN)"
        + ("" if is_7net else ", Auditory (AUD),")
        + " and Language co-enrichment; Posterior is DMN-dominant.",
        fontsize=14.0,
        fontweight="medium",
        color="#334155",
        transform=ax.transAxes,
    )
    ax.text(
        left_x,
        0.07,
        "• Clinical Phenotype Differentiation: NTS Pure MDD vs Any MDD/TIS displays differential dorsal attention & subregion profiles.",
        fontsize=13.5,
        fontweight="semibold",
        color="#475569",
        transform=ax.transAxes,
    )


def render_cohort_row(
    fig,
    gs_row,
    cdata,
    normalized: bool,
    is_top_row: bool,
    is_bottom_row: bool,
    metric_mode: str | None = None,
):
    """Render a single cohort row (Card + Surface + L-Bar + R-Bar) with enlarged typography."""
    spec = cdata["spec"]
    subregion_data = cdata["subregion_data"]
    surf_img = cdata["surface_img"]

    # 1. Left Metadata Card Axis
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
        f"(N = {spec['n_subjects']})",
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

    # 2. Middle Surface Atlas Axis
    ax_surf = fig.add_subplot(gs_row[1])
    ax_surf.axis("off")
    ax_surf.imshow(surf_img, aspect="equal")

    # 3. Left Hippocampus Stacked Bar Axis
    ax_bar_L = fig.add_subplot(gs_row[2])
    render_subregion_bar_axis(
        ax_bar_L,
        subregion_entries=subregion_data["L"],
        hemi="L",
        normalized=normalized,
        metric_mode=metric_mode,
        show_ylabel=True,
        is_top_row=is_top_row,
        is_bottom_row=is_bottom_row,
    )

    # 4. Right Hippocampus Stacked Bar Axis
    ax_bar_R = fig.add_subplot(gs_row[3])
    render_subregion_bar_axis(
        ax_bar_R,
        subregion_entries=subregion_data["R"],
        hemi="R",
        normalized=normalized,
        metric_mode=metric_mode,
        show_ylabel=False,
        is_top_row=is_top_row,
        is_bottom_row=is_bottom_row,
    )


def _has_explicit_legacy_hcp_ctxreg_provenance(path, manifest, *, method):
    """Accept only the named pre-signal_branch HCP Kong ctxreg contract."""
    if method != "kong2022_7net" or manifest.get("dataset") != "outputs_hcp":
        return False
    if "routea_ctxreg" not in Path(path).parts or manifest.get("atlas") != "kong2022":
        return False

    def is_ctxreg_path(value):
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
    return manifest.get("signal_control_route") in (None, "routea_ctxreg")


def load_cohort_subregion_top20_data(
    cohort_entry: dict,
    normalized: bool = True,
    smoothing: str = "2mm",
    is_7net: bool = False,
) -> dict[str, list[dict]]:
    """Load top-20 connected cortical network percentage for each subregion."""
    group_root = Path(cohort_entry["group_root"])
    legacy_7net_namespace = bool(cohort_entry.get("legacy_7net_namespace", False))
    atlas_folder = "kong2022_7net" if legacy_7net_namespace else "kong2022"
    grad_dir = Path(
        cohort_entry.get(
            "grad_subpath",
            group_root
            / f"gradient-spectral/{atlas_folder}/{smoothing}/cov_0p80/g1g2g3",
        )
    )

    subregion_data_by_hemi = {}
    for hemi in ["L", "R"]:
        hdir = grad_dir / f"hemi_{hemi}"
        top20_file = hdir / "cluster_roi_top20.json"

        method = cohort_entry.get("method")
        if method in COMPARISON_METHODS:
            expected_variant = (
                "normalized_top20_count" if method == "kong2022_7net" else "fisherz"
            )
            expected_normalized = expected_variant == "normalized_top20_count"
            if normalized != expected_normalized:
                raise ValueError(
                    f"Comparison method {method!r} requires "
                    f"{expected_variant!r} subregion data"
                )
            branch = cohort_entry.get("signal_branch")
            if branch not in SIGNAL_BRANCHES:
                raise ValueError(
                    "Comparison cohort must declare signal_branch as one of "
                    f"{SIGNAL_BRANCHES}: {cohort_entry.get('id', '<unknown>')}"
                )
            if branch == "ctxreg":
                manifest_path = hdir / "group_atlas_manifest.json"
                if not manifest_path.is_file():
                    raise FileNotFoundError(
                        f"Missing exact comparison gradient manifest: {manifest_path}"
                    )
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                recorded = manifest.get("signal_branch")
                if recorded != "ctxreg" and not (
                    recorded is None
                    and _has_explicit_legacy_hcp_ctxreg_provenance(
                        hdir, manifest, method=cohort_entry.get("method")
                    )
                ):
                    raise ValueError(
                        f"{manifest_path} records signal_branch="
                        f"{manifest.get('signal_branch')!r}; expected 'ctxreg'"
                    )
                validated = validate_gradient_assignment(
                    hdir, manifest, expected_variant
                )
                hemi_subregions = []
                for cid in (1, 2, 3):
                    comp = dict(validated["compositions"].get(cid, {}))
                    if not comp:
                        raise ValueError(
                            f"Comparison assignment has no positive composition for "
                            f"cluster {cid}: {hdir}"
                        )
                    hemi_subregions.append(
                        {
                            "cluster_id": cid,
                            "subregion_name": SUBREGION_NAMES.get(
                                cid, f"Subregion {cid}"
                            ),
                            "comp": comp,
                            "raw_counts": Counter(),
                        }
                    )
                subregion_data_by_hemi[hemi] = hemi_subregions
                continue

            if method == "lynch2024_7net":
                profile_path = hdir / "cluster_network_profile.npy"
                summary_path = hdir / "group_feature_summary.json"
                for required_path in (profile_path, summary_path):
                    if not required_path.is_file():
                        raise FileNotFoundError(
                            f"Missing exact standard Lynch comparison artifact: "
                            f"{required_path}"
                        )
                profile = np.asarray(np.load(profile_path), dtype=np.float64)
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                network_names = summary.get("network_names")
                if (
                    profile.ndim != 2
                    or profile.shape[0] != 3
                    or not isinstance(network_names, list)
                    or len(network_names) != profile.shape[1]
                ):
                    raise ValueError(
                        "Standard Lynch comparison profile/summary shape mismatch: "
                        f"profile={profile_path} summary={summary_path}"
                    )
                hemi_subregions = []
                for row_id, row in enumerate(profile, start=1):
                    positive = {
                        str(name): float(value)
                        for name, value in zip(network_names, row)
                        if float(value) > 0.0
                    }
                    total_positive = sum(positive.values())
                    if total_positive <= 0.0:
                        raise ValueError(
                            f"Standard Lynch comparison cluster {row_id} has no "
                            f"positive Fisher-z profile: {profile_path}"
                        )
                    comp = {
                        name: value / total_positive * 100.0
                        for name, value in positive.items()
                    }
                    hemi_subregions.append(
                        {
                            "cluster_id": row_id,
                            "subregion_name": SUBREGION_NAMES.get(
                                row_id, f"Subregion {row_id}"
                            ),
                            "comp": comp,
                            "raw_counts": Counter(),
                        }
                    )
                subregion_data_by_hemi[hemi] = hemi_subregions
                continue

            # Standard Kong2022 keeps its established normalized-top20 status
            # artifact below; it is an exact metric-specific input, not a
            # fallback from a missing comparison manifest.

        if normalized:
            normalized_status = (
                hdir / "cluster_assignment_status_normalized_top20_count.json"
            )
            if not normalized_status.is_file():
                raise FileNotFoundError(
                    "Normalized-top20 assignment status is required; refusing to read any "
                    f"fallback artifact for hemi-{hemi}: {normalized_status}"
                )
            status = json.loads(normalized_status.read_text(encoding="utf-8"))
            if not isinstance(status, dict):
                raise ValueError(
                    f"Normalized-top20 assignment status must be a JSON object: {normalized_status}"
                )
            assignment_metric = status.get("assignment_metric")
            if assignment_metric != NORMALIZED_TOP20_ASSIGNMENT_METRIC:
                raise ValueError(
                    "Normalized-top20 assignment status has the wrong assignment_metric "
                    f"{assignment_metric!r}; expected {NORMALIZED_TOP20_ASSIGNMENT_METRIC!r}: "
                    f"{normalized_status}"
                )
            items = status.get("items")
            if not isinstance(items, list):
                raise ValueError(
                    f"Normalized-top20 assignment status has no valid items list: {normalized_status}"
                )

            hemi_subregions = []
            seen_cluster_ids = set()
            for item in items:
                if not isinstance(item, dict):
                    raise ValueError(
                        f"Normalized-top20 assignment status contains a non-object item: {normalized_status}"
                    )
                cid = int(item.get("cluster_id", 0))
                if cid not in (1, 2, 3):
                    continue
                if cid in seen_cluster_ids:
                    raise ValueError(
                        f"Duplicate cluster_id {cid} in normalized-top20 assignment status: {normalized_status}"
                    )
                seen_cluster_ids.add(cid)
                normalized_counts = item.get("normalized_count_by_network")
                if not isinstance(normalized_counts, dict):
                    raise ValueError(
                        f"Cluster {cid} has no normalized_count_by_network mapping: {normalized_status}"
                    )
                values = {
                    str(name): float(value)
                    for name, value in normalized_counts.items()
                    if float(value) > 0.0
                }
                total = sum(values.values())
                if total <= 0.0:
                    raise ValueError(
                        f"Cluster {cid} has no positive normalized-top20 counts: {normalized_status}"
                    )
                comp = {name: value / total * 100.0 for name, value in values.items()}
                hemi_subregions.append(
                    {
                        "cluster_id": cid,
                        "subregion_name": SUBREGION_NAMES.get(cid, f"Subregion {cid}"),
                        "comp": comp,
                        "raw_counts": Counter(),
                    }
                )
            if seen_cluster_ids != {1, 2, 3}:
                raise ValueError(
                    "Normalized-top20 assignment status must contain exactly cluster IDs 1, 2, and 3 "
                    f"(got {sorted(seen_cluster_ids)}): {normalized_status}"
                )
            subregion_data_by_hemi[hemi] = sorted(
                hemi_subregions, key=lambda item: item["cluster_id"]
            )
            continue

        top20_json = json.loads(top20_file.read_text(encoding="utf-8"))
        c_map = top20_json.get("cluster_id_to_top20", {})

        hemi_subregions = []
        for cid in ["1", "2", "3"]:
            c_info = c_map.get(cid, {})
            top_rois = c_info.get("top_positive", [])

            # Count networks among top 20
            raw_counts = Counter()
            for r in top_rois:
                if is_7net:
                    net = (
                        r.get("canonical7")
                        or r.get("canonical8")
                        or r.get("network_canonical8")
                        or r.get("network_name")
                        or "Unknown"
                    )
                else:
                    net = (
                        r.get("canonical8")
                        or r.get("network_canonical8")
                        or r.get("network_name")
                        or "Unknown"
                    )
                raw_counts[net] += 1

            # Compute percentage: either normalized by network total size, or raw count percentage
            total_rois_map = (
                KONG2022_CANONICAL7_TOTAL_ROIS
                if is_7net
                else KONG2022_CANONICAL8_TOTAL_ROIS
            )
            if normalized:
                # Weighted count: count / total_rois_in_network
                weighted_counts = {
                    net: cnt / total_rois_map.get(net, 50)
                    for net, cnt in raw_counts.items()
                }
                total_weighted = sum(weighted_counts.values())
                if total_weighted > 0:
                    comp = {
                        net: (w / total_weighted) * 100.0
                        for net, w in weighted_counts.items()
                    }
                else:
                    comp = {}
            else:
                total_raw = sum(raw_counts.values())
                if total_raw > 0:
                    comp = {
                        net: (cnt / total_raw) * 100.0
                        for net, cnt in raw_counts.items()
                    }
                else:
                    comp = {}

            hemi_subregions.append(
                {
                    "cluster_id": int(cid),
                    "subregion_name": SUBREGION_NAMES.get(int(cid), f"Subregion {cid}"),
                    "comp": comp,
                    "raw_counts": raw_counts,
                }
            )
        subregion_data_by_hemi[hemi] = hemi_subregions

    return subregion_data_by_hemi


def _render_cohort_surface_image(
    cohort_entry: dict,
    subregion_data: dict[str, list[dict]],
    *,
    metric_mode: str,
    smoothing: str,
    panel_root: Path,
) -> Image.Image:
    """Render a comparison surface from the exact assignment used by its bars."""
    method = cohort_entry.get("method")
    if method in COMPARISON_METHODS:
        assignment_variant = (
            "normalized_top20_count"
            if metric_mode == SUBREGION_METRIC_NORMALIZED_TOP20
            else "fisherz"
        )
        branch = str(cohort_entry["signal_branch"])
        if branch not in SIGNAL_BRANCHES:
            raise ValueError(
                f"Unsupported comparison signal branch {branch!r}: "
                f"{cohort_entry.get('id', '<unknown>')}"
            )
        panel_dir = Path(panel_root) / str(method) / str(cohort_entry["id"])
        panel_cohort = dict(cohort_entry)
        panel_cohort["signal_branch"] = branch
        return render_gradient_surface_panel(
            panel_cohort,
            subregion_data,
            variant=assignment_variant,
            smoothing=smoothing,
            outdir=panel_dir,
        )

    overview_path = resolve_gradient_overview_path(
        cohort_entry,
        normalized=metric_mode == SUBREGION_METRIC_NORMALIZED_TOP20,
    )
    if not overview_path.is_file():
        raise FileNotFoundError(
            f"Overview image not found for cohort "
            f"{cohort_entry['dataset_name']} {cohort_entry['cohort_name']}: "
            f"{overview_path}"
        )
    return crop_gradient_surface_from_overview(
        overview_path,
        smoothing=smoothing,
    )


def resolve_gradient_overview_path(cohort_entry: dict, *, normalized: bool) -> Path:
    """Resolve the overview image without falling back across assignment metrics."""
    declared_metric = cohort_entry.get("subregion_metric")
    if declared_metric == SUBREGION_METRIC_NORMALIZED_TOP20 and not normalized:
        raise ValueError("Kong2022 comparison overview requires normalized-top20 mode")
    if declared_metric == SUBREGION_METRIC_FISHERZ_PROFILE and normalized:
        raise ValueError("Lynch2024 comparison overview requires Fisher-z profile mode")
    if normalized:
        overview_value = cohort_entry.get("overview_grad_normalized_top20_count")
        cohort_label = (
            f"{cohort_entry.get('dataset_name', '<unknown dataset>')} "
            f"{cohort_entry.get('cohort_name', '<unknown cohort>')}"
        )
        if overview_value is None:
            raise FileNotFoundError(
                "Normalized-top20 gradient overview is required for "
                f"{cohort_label}; missing path contract key "
                "'overview_grad_normalized_top20_count'"
            )
        overview_path = Path(overview_value)
        if not overview_path.name.endswith("_normalized_top20_count.png"):
            raise ValueError(
                "Normalized-top20 gradient overview contract resolved to a non-normalized artifact "
                f"for {cohort_label}: {overview_path}"
            )
        return overview_path

    overview_value = cohort_entry.get("overview_grad")
    if overview_value is None:
        raise FileNotFoundError(
            "Gradient overview path contract is missing; refusing to infer an overview artifact"
        )
    overview_path = Path(overview_value)
    if cohort_entry.get(
        "subregion_metric"
    ) == SUBREGION_METRIC_FISHERZ_PROFILE and not overview_path.name.endswith(
        "_fisherz.png"
    ):
        raise ValueError(
            "Fisher-z profile overview contract resolved to a non-Fisher-z artifact: "
            f"{overview_path}"
        )
    return overview_path


def render_multidataset_gradient_subregion_top20_stacked_2col(
    out_path: Path,
    tsv_out: Path | None = None,
    normalized: bool = True,
    spec: GridSpec = GridSpec("2mm", "36p"),
    cohort_columns: tuple[list[dict], list[dict]] | None = None,
):
    """Main function to build and save the 2-column gradient subregion top-20 figure with enlarged fonts."""
    is_7net = spec.is_7net
    if cohort_columns is None:
        col1_specs, col2_specs = get_multidataset_cohorts(spec)
    else:
        col1_specs, col2_specs = cohort_columns
    all_specs = col1_specs + col2_specs
    all_metric_modes = [
        resolve_subregion_metric_mode(c_spec, requested_normalized=normalized)
        for c_spec in all_specs
    ]
    mode_str = (
        subregion_metric_label(all_metric_modes[0])
        if len(set(all_metric_modes)) == 1
        else "Method-specific metrics"
    )
    print(
        f"Loading data and pre-rendering assets for {len(all_specs)} cohorts ({mode_str}, {spec.tag})..."
    )
    panel_root = out_path.parent / f"{out_path.stem}_panels"
    panel_records = []

    data_col1 = []
    for c_spec in col1_specs:
        metric_mode = resolve_subregion_metric_mode(
            c_spec, requested_normalized=normalized
        )
        metric_is_normalized = metric_mode == SUBREGION_METRIC_NORMALIZED_TOP20
        subregion_data = load_cohort_subregion_top20_data(
            c_spec,
            normalized=metric_is_normalized,
            smoothing=spec.smoothing,
            is_7net=spec.is_7net,
        )
        surf_img = _render_cohort_surface_image(
            c_spec,
            subregion_data,
            metric_mode=metric_mode,
            smoothing=spec.smoothing,
            panel_root=panel_root,
        )
        if c_spec.get("method") in COMPARISON_METHODS:
            panel_dir = panel_root / str(c_spec["method"]) / str(c_spec["id"])
            panel_records.append(
                {
                    "cohort": str(c_spec["id"]),
                    "method": str(c_spec["method"]),
                    "surface_png": str((panel_dir / "surface.png").resolve()),
                    "surface_sources": str(
                        (panel_dir / "surface.sources.json").resolve()
                    ),
                    "assignment_audit": str(
                        (panel_dir / "assignment_audit.json").resolve()
                    ),
                }
            )
        else:
            overview_path = resolve_gradient_overview_path(
                c_spec,
                normalized=metric_is_normalized,
            )
            panel_records.append(
                {
                    "cohort": str(c_spec["id"]),
                    "method": "",
                    "overview_png": str(overview_path.resolve()),
                }
            )
        data_col1.append(
            {
                "spec": c_spec,
                "subregion_data": subregion_data,
                "surface_img": surf_img,
                "metric_mode": metric_mode,
            }
        )

    data_col2 = []
    for c_spec in col2_specs:
        metric_mode = resolve_subregion_metric_mode(
            c_spec, requested_normalized=normalized
        )
        metric_is_normalized = metric_mode == SUBREGION_METRIC_NORMALIZED_TOP20
        subregion_data = load_cohort_subregion_top20_data(
            c_spec,
            normalized=metric_is_normalized,
            smoothing=spec.smoothing,
            is_7net=spec.is_7net,
        )
        surf_img = _render_cohort_surface_image(
            c_spec,
            subregion_data,
            metric_mode=metric_mode,
            smoothing=spec.smoothing,
            panel_root=panel_root,
        )
        if c_spec.get("method") in COMPARISON_METHODS:
            panel_dir = panel_root / str(c_spec["method"]) / str(c_spec["id"])
            panel_records.append(
                {
                    "cohort": str(c_spec["id"]),
                    "method": str(c_spec["method"]),
                    "surface_png": str((panel_dir / "surface.png").resolve()),
                    "surface_sources": str(
                        (panel_dir / "surface.sources.json").resolve()
                    ),
                    "assignment_audit": str(
                        (panel_dir / "assignment_audit.json").resolve()
                    ),
                }
            )
        else:
            overview_path = resolve_gradient_overview_path(
                c_spec,
                normalized=metric_is_normalized,
            )
            panel_records.append(
                {
                    "cohort": str(c_spec["id"]),
                    "method": "",
                    "overview_png": str(overview_path.resolve()),
                }
            )
        data_col2.append(
            {
                "spec": c_spec,
                "subregion_data": subregion_data,
                "surface_img": surf_img,
                "metric_mode": metric_mode,
            }
        )

    # Save TSV data
    if tsv_out is not None:
        tsv_rows = [
            "dataset\tcohort\tn_subjects\themisphere\tcluster_id\tsubregion_name\tnetwork_name\tpercentage\traw_roi_count_in_top20\ttotal_network_rois\tnormalization_mode"
        ]
        for cdata in data_col1 + data_col2:
            cspec = cdata["spec"]
            metric_mode = cdata["metric_mode"]
            for hemi in ["L", "R"]:
                for s_entry in cdata["subregion_data"][hemi]:
                    cid = s_entry["cluster_id"]
                    sname = s_entry["subregion_name"]
                    comp = s_entry["comp"]
                    raw_cnts = s_entry["raw_counts"]
                    for net, pct in comp.items():
                        if pct > 0:
                            if metric_mode == SUBREGION_METRIC_FISHERZ_PROFILE:
                                cnt = ""
                                tot = ""
                            else:
                                cnt = raw_cnts.get(net, 0)
                                capacity_map = (
                                    KONG2022_CANONICAL7_TOTAL_ROIS
                                    if is_7net
                                    else KONG2022_CANONICAL8_TOTAL_ROIS
                                )
                                if (
                                    metric_mode == SUBREGION_METRIC_NORMALIZED_TOP20
                                    and net not in capacity_map
                                ):
                                    raise ValueError(
                                        "Normalized-top20 TSV export encountered a network absent from "
                                        f"the registered ROI-capacity map: {net!r}"
                                    )
                                tot = capacity_map.get(net, "")
                            tsv_rows.append(
                                f"{cspec['dataset_name']}\t{cspec['cohort_name']}\t{cspec['n_subjects']}\t{hemi}\t{cid}\t{sname}\t{net}\t{pct:.2f}\t{cnt}\t{tot}\t{metric_mode}"
                            )
        tsv_out.parent.mkdir(parents=True, exist_ok=True)
        tsv_out.write_text("\n".join(tsv_rows) + "\n", encoding="utf-8")
        print(f"Saved stats TSV table to {tsv_out}")

    n_rows = max(len(col1_specs), len(col2_specs))
    fig_w = 48.0
    fig_h = 28.0 * (n_rows / 5.0) if n_rows > 5 else 28.0

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=220)
    fig.patch.set_facecolor("#ffffff")

    total_subjects = sum(c["n_subjects"] for c in all_specs)
    method_specific = len(set(all_metric_modes)) > 1
    if method_specific:
        main_title = (
            f"Cross-Dataset Hippocampal {spec.smoothing} Gradient Atlas & "
            "Method-Specific Subregion Network Composition"
        )
        sub_title = (
            f"Gradient-Spectral Parcellation ($K=3$) & Canonical {'7' if is_7net else '8'}-Network Composition "
            "Kong2022: Normalized Top-20 Count | Lynch2024: Positive Fisher-z Profile "
            f"across {len(all_specs)} Cohorts (N={total_subjects})"
        )
    else:
        main_title = f"Cross-Dataset Hippocampal {spec.smoothing} Gradient Atlas & Subregion {'Normalized ' if normalized else ''}Top-20 Network Composition"
        sub_title = (
            f"Gradient-Spectral Parcellation ($K=3$) & Canonical {'7' if is_7net else '8'}-Network Composition "
            f"{'Normalized by Network ROI Capacity' if normalized else '(Raw Top-20 Hits)'} across {len(all_specs)} Cohorts (N={total_subjects})"
        )

    fig.suptitle(
        main_title,
        fontsize=38.0,
        fontweight="heavy",
        x=0.50,
        y=0.986,
        ha="center",
        color="#0b2447",
    )
    fig.text(
        0.50,
        0.961,
        sub_title,
        fontsize=22.0,
        fontweight="medium",
        ha="center",
        color="#334e68",
    )

    top_margin = 0.930
    bottom_margin = 0.088

    # Top GridSpec: 1 row, 2 major columns
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

    # Column Headers (Enlarged to 20pt)
    y_col_header = top_margin + 0.007
    col1_x0 = 0.015
    col1_w = (0.985 - 0.015 - 0.045) / 2.0
    r0, r1, r2, r3 = width_ratios
    rtot = sum(width_ratios)

    p0 = col1_x0 + (r0 / (2 * rtot)) * col1_w
    p1 = col1_x0 + ((r0 + r1 / 2.0) / rtot) * col1_w
    p2 = col1_x0 + ((r0 + r1 + r2 / 2.0) / rtot) * col1_w
    p3 = col1_x0 + ((r0 + r1 + r2 + r3 / 2.0) / rtot) * col1_w

    col1_metric_mode = data_col1[0]["metric_mode"] if data_col1 else all_metric_modes[0]
    col2_metric_mode = (
        data_col2[0]["metric_mode"] if data_col2 else all_metric_modes[-1]
    )
    col1_atlas_title = format_comparison_column_title(
        col1_specs[0].get("method_name") if col1_specs else None,
        gradient_atlas_column_label(col1_metric_mode),
    )
    col2_atlas_title = format_comparison_column_title(
        col2_specs[0].get("method_name") if col2_specs else None,
        gradient_atlas_column_label(col2_metric_mode),
    )
    fig.text(
        p0,
        y_col_header,
        "Cohort Profile",
        ha="center",
        va="bottom",
        fontsize=20.0,
        fontweight="heavy",
        color="#102a43",
    )
    fig.text(
        p1,
        y_col_header,
        col1_atlas_title,
        ha="center",
        va="bottom",
        fontsize=20.0,
        fontweight="heavy",
        color="#102a43",
    )
    fig.text(
        p2,
        y_col_header,
        subregion_bar_column_label("L", col1_metric_mode),
        ha="center",
        va="bottom",
        fontsize=20.0,
        fontweight="heavy",
        color="#0f4c81",
    )
    fig.text(
        p3,
        y_col_header,
        subregion_bar_column_label("R", col1_metric_mode),
        ha="center",
        va="bottom",
        fontsize=20.0,
        fontweight="heavy",
        color="#0f4c81",
    )

    col2_x0 = col1_x0 + col1_w + 0.045
    q0 = col2_x0 + (r0 / (2 * rtot)) * col1_w
    q1 = col2_x0 + ((r0 + r1 / 2.0) / rtot) * col1_w
    q2 = col2_x0 + ((r0 + r1 + r2 / 2.0) / rtot) * col1_w
    q3 = col2_x0 + ((r0 + r1 + r2 + r3 / 2.0) / rtot) * col1_w

    fig.text(
        q0,
        y_col_header,
        "Cohort Profile",
        ha="center",
        va="bottom",
        fontsize=20.0,
        fontweight="heavy",
        color="#102a43",
    )
    fig.text(
        q1,
        y_col_header,
        col2_atlas_title,
        ha="center",
        va="bottom",
        fontsize=20.0,
        fontweight="heavy",
        color="#102a43",
    )
    fig.text(
        q2,
        y_col_header,
        subregion_bar_column_label("L", col2_metric_mode),
        ha="center",
        va="bottom",
        fontsize=20.0,
        fontweight="heavy",
        color="#0f4c81",
    )
    fig.text(
        q3,
        y_col_header,
        subregion_bar_column_label("R", col2_metric_mode),
        ha="center",
        va="bottom",
        fontsize=20.0,
        fontweight="heavy",
        color="#0f4c81",
    )

    # Render Left Column
    for idx, cdata in enumerate(data_col1):
        render_cohort_row(
            fig=fig,
            gs_row=[gs_left[idx, c] for c in range(4)],
            cdata=cdata,
            normalized=cdata["metric_mode"] == SUBREGION_METRIC_NORMALIZED_TOP20,
            is_top_row=(idx == 0),
            is_bottom_row=(idx == len(data_col1) - 1),
            metric_mode=cdata["metric_mode"],
        )

    # Render Right Column
    for idx, cdata in enumerate(data_col2):
        render_cohort_row(
            fig=fig,
            gs_row=[gs_right[idx, c] for c in range(4)],
            cdata=cdata,
            normalized=cdata["metric_mode"] == SUBREGION_METRIC_NORMALIZED_TOP20,
            is_top_row=(idx == 0),
            is_bottom_row=(idx == len(data_col2) - 1),
            metric_mode=cdata["metric_mode"],
        )

    # Global Footer Legend
    legend_ax = fig.add_axes([0.015, 0.010, 0.97, 0.042])
    legend_ax.axis("off")

    if is_7net:
        legend_patches = [
            patches.Patch(
                facecolor=NETWORK_COLORS["Default"],
                edgecolor="#333333",
                label="DMN (Default)",
            ),
            patches.Patch(
                facecolor=NETWORK_COLORS["Somatomotor"],
                edgecolor="#333333",
                label="SMN (Somatomotor)",
            ),
            patches.Patch(
                facecolor=NETWORK_COLORS["VentralAttention"],
                edgecolor="#333333",
                label="VAN (VentralAttention)",
            ),
            patches.Patch(
                facecolor=NETWORK_COLORS["DorsalAttention"],
                edgecolor="#333333",
                label="DAN (DorsalAttention)",
            ),
            patches.Patch(
                facecolor=NETWORK_COLORS["Control"],
                edgecolor="#333333",
                label="CON (Control)",
            ),
            patches.Patch(
                facecolor=NETWORK_COLORS["Language"],
                edgecolor="#333333",
                label="LAN (Language)",
            ),
            patches.Patch(
                facecolor=NETWORK_COLORS["Visual"],
                edgecolor="#333333",
                label="VN (Visual)",
            ),
        ]
        n_legend_cols = 7
    else:
        legend_patches = [
            patches.Patch(
                facecolor=NETWORK_COLORS["Default"],
                edgecolor="#333333",
                label="DMN (Default)",
            ),
            patches.Patch(
                facecolor=NETWORK_COLORS["Somatomotor"],
                edgecolor="#333333",
                label="SMN (Somatomotor)",
            ),
            patches.Patch(
                facecolor=NETWORK_COLORS["Auditory"],
                edgecolor="#333333",
                label="AUD (Auditory)",
            ),
            patches.Patch(
                facecolor=NETWORK_COLORS["VentralAttention"],
                edgecolor="#333333",
                label="VAN (VentralAttention)",
            ),
            patches.Patch(
                facecolor=NETWORK_COLORS["DorsalAttention"],
                edgecolor="#333333",
                label="DAN (DorsalAttention)",
            ),
            patches.Patch(
                facecolor=NETWORK_COLORS["Control"],
                edgecolor="#333333",
                label="CON (Control)",
            ),
            patches.Patch(
                facecolor=NETWORK_COLORS["Language"],
                edgecolor="#333333",
                label="LAN (Language)",
            ),
            patches.Patch(
                facecolor=NETWORK_COLORS["Visual"],
                edgecolor="#333333",
                label="VN (Visual)",
            ),
        ]
        n_legend_cols = 8

    leg = legend_ax.legend(
        handles=legend_patches,
        loc="center",
        ncol=n_legend_cols,
        frameon=True,
        facecolor="#fcfdfe",
        edgecolor="#94a3b8",
        fontsize=20.0,
        handlelength=1.8,
        handleheight=1.2,
        columnspacing=1.8,
        borderpad=0.5,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving high-resolution figure to {out_path}...")
    plt.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    source_sidecar = out_path.with_suffix(out_path.suffix + ".sources.json")
    source_fingerprints = []
    for record in panel_records:
        for key in ("surface_sources", "assignment_audit", "overview_png"):
            raw_path = record.get(key)
            if raw_path is None:
                continue
            source_path = Path(str(raw_path))
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"Composite source lock requires exact source: {source_path}"
                )
            source_fingerprints.append(fingerprint(source_path))
        surface_path = record.get("surface_png")
        if surface_path is not None:
            source_path = Path(str(surface_path))
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"Composite source lock requires exact panel image: {source_path}"
                )
            source_fingerprints.append(fingerprint(source_path))
    if tsv_out is not None:
        tsv_path = Path(tsv_out)
        if not tsv_path.is_file():
            raise FileNotFoundError(
                f"Composite source lock requires exact statistics TSV: {tsv_path}"
            )
        source_fingerprints.append(fingerprint(tsv_path))
    source_sidecar.write_text(
        json.dumps(
            {
                "schema": "multidataset_composite_source_lock.v1",
                "output_png": str(out_path.resolve()),
                "output_png_sha256": fingerprint(out_path)["sha256"],
                "grid": spec.tag,
                "signal_branch": spec.signal_branch,
                "version": (
                    COMPARISON_VERSION
                    if any(c.get("method") in COMPARISON_METHODS for c in all_specs)
                    else "legacy"
                ),
                "panels": panel_records,
                "source_files": source_fingerprints,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("Done!")


def _render_grid():
    parser = argparse.ArgumentParser(
        description="Render 2-column multi-dataset hippocampal gradient subregion Top-20 stacked bar chart."
    )
    parser.add_argument(
        "--version",
        choices=["legacy", COMPARISON_VERSION],
        default="legacy",
        help="Data branch; legacy is unchanged, or use the additive HCP/JJ_TMS comparison.",
    )
    default_dir = (
        REPO_ROOT
        / "scripts/visualize/multidataset_gradient_subregion_top20_stacked_2col"
    )
    default_png = (
        default_dir / "multidataset_gradient_subregion_top20_normalized_count_2col.png"
    )
    default_tsv = (
        default_dir
        / "stats"
        / "multidataset_gradient_subregion_top20_normalized_stats.tsv"
    )

    parser.add_argument("--out-png", default=None, help="Path to output PNG.")
    parser.add_argument("--stats-tsv", default=None, help="Path to output TSV.")
    parser.add_argument(
        "--smoothing",
        choices=["2mm", "4mm"],
        default="2mm",
        help="Smoothing resolution (default: 2mm)",
    )
    parser.add_argument(
        "--variant",
        choices=["36p", "22p"],
        default="36p",
        help="Preprocessing variant (default: 36p)",
    )
    parser.add_argument(
        "--signal-branch",
        choices=list(SIGNAL_BRANCHES),
        default="standard",
        help="Explicit signal branch for the comparison renderer.",
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
        help="Grid variant shortcut",
    )
    parser.add_argument(
        "--raw-count",
        action="store_true",
        help="Plot raw count rather than normalized count.",
    )
    args = parser.parse_args()

    if args.version == COMPARISON_VERSION and args.raw_count:
        parser.error(
            f"--raw-count is only supported by --version legacy; {COMPARISON_VERSION} uses method-specific metrics"
        )
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
    if args.grid and args.signal_branch != "standard":
        spec = GridSpec(
            spec.smoothing,
            spec.variant,
            signal_branch=args.signal_branch,
            atlas=spec.atlas,
        )

    if args.out_png:
        out_png = Path(args.out_png).resolve()
    else:
        output_tag = (
            get_comparison_output_tag(spec)
            if args.version == COMPARISON_VERSION
            else spec.file_tag
        )
        out_png = (
            default_dir
            / f"multidataset_gradient_subregion_top20_stacked_2col_{output_tag}.png"
        )

    if args.stats_tsv:
        stats_tsv = Path(args.stats_tsv).resolve()
    else:
        stats_dir = default_dir / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        stats_tsv = (
            stats_dir
            / f"multidataset_gradient_subregion_top20_stacked_2col_{spec.file_tag}_stats.tsv"
        )

    normalized = not args.raw_count
    if args.version == COMPARISON_VERSION:
        render_spec = GridSpec(
            spec.smoothing,
            spec.variant,
            signal_branch=spec.signal_branch,
            atlas="7net",
        )
        comparison = get_comparison_visual_columns(render_spec)
        comparison_tsv = (
            Path(args.stats_tsv).resolve()
            if args.stats_tsv
            else default_dir
            / "stats"
            / f"{COMPARISON_VERSION}_{render_spec.tag}_stats.tsv"
        )
        render_multidataset_gradient_subregion_top20_stacked_2col(
            out_png,
            comparison_tsv,
            normalized=normalized,
            spec=render_spec,
            cohort_columns=comparison,
        )
    else:
        render_multidataset_gradient_subregion_top20_stacked_2col(
            out_png, stats_tsv, normalized=normalized, spec=spec
        )


if __name__ == "__main__":
    raise SystemExit(
        "Standalone rendering has been removed. Use "
        "scripts/visualize/run_multidataset_suite.py "
        "with --figures to select the figure type; all four grids are processed in 4-way parallel by default."
    )

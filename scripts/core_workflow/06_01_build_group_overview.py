#!/usr/bin/env python3
from __future__ import annotations

"""
Process Step Description:
Build group-level results overview. Summarizes clustering and functional network mapping results to generate comprehensive overview charts or files for visualization.

Main Flow Path:
- 9th step in the numbered pipeline sequence (first stage of presentation output generation):
  01_01_run_hippunfold_t1w_local.py -> 01_02_check_hippunfold_outputs.py -> 02_01_prepare_group_inputs.py -> 02_02_build_qc_manifest.py -> 03_01/03_02 reference building -> 04_01/04_02/04_03 FC materialization -> 05_01 gradient computation -> 05_02 group atlas parcellation -> 06_01_build_group_overview.py (this step) -> 07_01 group matrix overview.

Inputs:
- Command-line arguments:
  - --group-root: Directory where group-atlas intermediate files are saved (default: `outputs_core/work/group_atlas/routea_full_k3_fisherz`).
  - --present-dir: Directory where final presentation figures will be saved (default: `outputs_core/present/overviews`).
  - --methods: Parcellation methods to process: "intrinsic-spectral", "network-spectral", "gradient-spectral", "wta-top".
  - --atlases: Cortical atlases to process (default: ["kong2022"]).
  - --coverage-threshold: Selected coverage threshold for overview outputs (default: 0.80).
  - --scene: Path to standard wb scene template file.
  - --route-tag: Custom workflow branch tag to override automatic detection.
  - --mode-tag: Custom sparsification mode tag to override automatic detection.
  - --assignment-variant: Network mapping variant: "fisherz", "count" (default: "fisherz").
  - --manifest-name: Name of generated presentation manifest JSON (default: "group_overview_manifest.json").
  - --skip-readme: Skip writing the README.md summary in presentation directory.
  - --skip-manifest: Skip writing the overview manifest JSON file.
  - --allow-missing-combos: Allow running even when some methods/hemispheres are missing.
- Physical inputs:
  - Group parcellation label arrays: `{group_root}/{method}/smooth-{smoothing}_hemi-{hemi}_cov-{threshold}/group_functional_atlas_labels.npy`.
  - Group mapping status files: `{group_root}/{method}/smooth-{smoothing}_hemi-{hemi}_cov-{threshold}/cluster_assignment_status.json`.

Outputs:
- Command-line output / file outputs:
  - Overview panel PNG figures for each combination: `{present_dir}/{atlas}_{method}_overview_{route_tag}_{smoothing}_cov{threshold_tag}.png`.
  - Group overview manifest JSON file: `{present_dir}/{manifest_name}`.
  - Summary README file: `{present_dir}/README.md`.
"""
import argparse
import csv
import json
import re
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from nibabel.gifti import GiftiDataArray, GiftiImage, GiftiLabel, GiftiLabelTable

from lib.output_layout import DEFAULT_OVERVIEWS_DIR, DEFAULT_ROUTEA_GROUP_ATLAS_ROOT
from lib.workflow_utils import write_json
from lib.surface_utils import long_axis_order, render_locked_grid, network_color_map
from lib.io_utils import load_json
from lib.gifti_labels import resolve_cluster_label_path, make_named_label_gifti, materialize_combo


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GROUP_ROOT = DEFAULT_ROUTEA_GROUP_ATLAS_ROOT
DEFAULT_PRESENT_DIR = DEFAULT_OVERVIEWS_DIR
DEFAULT_SCENE = REPO_ROOT / "config" / "wb_locked_native_view_lateral_medial.scene"
METHOD_CHOICES = [
    "network-spectral",
    "gradient-spectral",
    "intrinsic-spectral",
    "wta-top",
]
ROUTE_TAG_CHOICES = ["routea", "routea_ctxreg", "routea_ctxdrop", "routeb"]
METHODS_DEFAULT = [
    "intrinsic-spectral",
    "network-spectral",
    "wta-top",
]
ATLASES = ["kong2019", "kong2019_8net", "kong2022"]
SMOOTHINGS = ["2mm", "4mm"]
FS_TITLE = 11
FS_LABEL = 10
FS_TICK = 9
FS_SMALL = 8.5
FS_LEGEND_SMALL = FS_SMALL - 2.0
FS_CELL = 6
FS_MISSING = 13
FS_BIG_TITLE = 13
FS_SUPTITLE = 18
CLUSTERLESS_METHODS = {"wta-top"}
ASSIGNMENT_VARIANTS = ("fisherz", "count")
DEFAULT_CROSS_ATLAS_MERGE = REPO_ROOT / "config" / "cross_atlas_network_merge.json"
DEFAULT_NETWORK_STYLE = REPO_ROOT / "config" / "hipp_network_style.json"
WTA_OVERVIEW_PROFILE_BINS = 20
WTA_OVERVIEW_SIDE_PANEL_FONT_DELTA_PT = 4
WTA_OVERVIEW_SIDE_PANEL_TITLE_FS = FS_TITLE + WTA_OVERVIEW_SIDE_PANEL_FONT_DELTA_PT
WTA_OVERVIEW_SIDE_PANEL_LABEL_FS = FS_LABEL + WTA_OVERVIEW_SIDE_PANEL_FONT_DELTA_PT
WTA_OVERVIEW_SIDE_PANEL_TICK_FS = FS_TICK + WTA_OVERVIEW_SIDE_PANEL_FONT_DELTA_PT
WTA_OVERVIEW_SIDE_PANEL_LEGEND_FS = FS_LEGEND_SMALL + WTA_OVERVIEW_SIDE_PANEL_FONT_DELTA_PT
WTA_NON_NETWORK_NAME = "__non_network_or_null__"
GRADIENT_METHOD = "gradient-spectral"


def _is_gradient_method(method: str) -> bool:
    return method == GRADIENT_METHOD


def _discover_gradient_comp_tags(
    group_root: Path,
    method: str,
    atlas: str,
    smoothing: str,
    coverage_tag: str,
) -> list[str]:
    """Discover gradient component subdirectories under the coverage directory."""
    if not _is_gradient_method(method):
        return [""]
    base = group_root / method / atlas / smoothing / coverage_tag
    if not base.exists():
        return [""]
    tags = sorted(d.name for d in base.iterdir() if d.is_dir() and d.name.startswith("g"))
    return tags if tags else [""]


def _build_run_dir(
    group_root: Path,
    method: str,
    atlas: str,
    smoothing: str,
    coverage_tag: str,
    hemi: str,
    *,
    gradient_comp_tag: str = "",
) -> Path:
    """Build the run directory path, including gradient component subdir for gradient-spectral."""
    if _is_gradient_method(method) and gradient_comp_tag:
        return group_root / method / atlas / smoothing / coverage_tag / gradient_comp_tag / f"hemi_{hemi}"
    return group_root / method / atlas / smoothing / coverage_tag / f"hemi_{hemi}"


def load_network_bar_colors() -> dict[str, str]:
    out: dict[str, str] = {}
    if not DEFAULT_CROSS_ATLAS_MERGE.exists():
        return out
    payload = json.loads(DEFAULT_CROSS_ATLAS_MERGE.read_text(encoding="utf-8"))
    for name, rgba in payload.get("shared_colors_rgba", {}).items():
        if not isinstance(rgba, list) or len(rgba) != 4:
            continue
        r, g, b = int(rgba[0]), int(rgba[1]), int(rgba[2])
        out[str(name)] = f"#{r:02x}{g:02x}{b:02x}"
    return out


NETWORK_BAR_COLORS = load_network_bar_colors()




def resolve_assignment_variant_paths(
    run_dir: Path,
    manifest: dict,
    variant: str,
) -> tuple[Path, Path]:
    if variant not in ASSIGNMENT_VARIANTS:
        raise ValueError(f"Unknown assignment variant: {variant}")
    variants = manifest.get("assignment_variants", {})
    if isinstance(variants, dict) and variant in variants and isinstance(variants[variant], dict):
        row = variants[variant]
        status_raw = row.get("cluster_assignment_status")
        label_raw = row.get("template_functional_atlas_label") or row.get("template_functional_atlas_label_variant")
        if status_raw and label_raw:
            return Path(str(status_raw)).resolve(), Path(str(label_raw)).resolve()
    if variant == "fisherz":
        status = run_dir / "cluster_assignment_status_fisherz.json"
        if not status.exists():
            status = run_dir / "cluster_assignment_status.json"
        label_raw = manifest.get("template_functional_atlas_label")
        label = Path(str(label_raw)).resolve() if label_raw else run_dir / "missing_fisherz_functional.label.gii"
        return status, label
    status = run_dir / "cluster_assignment_status_count.json"
    if not status.exists():
        status = run_dir / "cluster_roi_topn_assignment_status.json"
    label = run_dir / "template" / "workbench_assets"
    candidates = sorted(label.glob("*label-group_functional_atlas_*_count.label.gii"))
    if candidates:
        return status, candidates[0].resolve()
    return status, run_dir / "missing_count_functional.label.gii"


MATERIALIZE_COLOR_MAP = network_color_map()


def _is_path_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _load_label_vector(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    return np.asarray(img.darrays[0].data).astype(np.int32, copy=False)


def ensure_functional_label_consistency(manifest_path: Path, manifest: dict) -> dict:
    run_dir = manifest_path.parent
    expected_npy = run_dir / "group_functional_atlas_labels.npy"
    raw_label = manifest.get("template_functional_atlas_label", None)
    label_path = Path(str(raw_label)).resolve() if raw_label is not None else None

    need_refresh = False
    reasons: list[str] = []
    if label_path is None:
        need_refresh = True
        reasons.append("missing_template_functional_atlas_label")
    elif not label_path.exists():
        need_refresh = True
        reasons.append("missing_functional_label_gii")
    elif not _is_path_within(label_path, run_dir):
        need_refresh = True
        reasons.append("functional_label_path_outside_run_dir")
    if not expected_npy.exists():
        need_refresh = True
        reasons.append("missing_group_functional_atlas_labels_npy")

    if not need_refresh and label_path is not None:
        try:
            arr_npy = np.load(expected_npy).astype(np.int32, copy=False)
            arr_gii = _load_label_vector(label_path)
            if arr_npy.shape != arr_gii.shape or not np.array_equal(arr_npy, arr_gii):
                need_refresh = True
                reasons.append("npy_gii_mismatch")
        except Exception as exc:  # pragma: no cover
            need_refresh = True
            reasons.append(f"read_error:{type(exc).__name__}")

    if not need_refresh:
        return manifest

    updated = materialize_combo(manifest_path, MATERIALIZE_COLOR_MAP)
    if not updated:
        raise RuntimeError(
            f"Functional atlas label inconsistency detected but materialization was skipped: {manifest_path} reasons={','.join(reasons)}"
        )
    refreshed = load_json(manifest_path)
    refreshed_label_raw = refreshed.get("template_functional_atlas_label", None)
    if refreshed_label_raw is None:
        raise RuntimeError(f"materialize_combo did not write template_functional_atlas_label: {manifest_path}")
    refreshed_label = Path(str(refreshed_label_raw)).resolve()
    if not refreshed_label.exists():
        raise RuntimeError(f"materialize_combo wrote missing label file: {refreshed_label}")
    if not _is_path_within(refreshed_label, run_dir):
        raise RuntimeError(f"materialize_combo wrote label outside run dir: {refreshed_label} (run_dir={run_dir})")
    if not expected_npy.exists():
        raise RuntimeError(f"materialize_combo did not write expected npy: {expected_npy}")
    arr_npy = np.load(expected_npy).astype(np.int32, copy=False)
    arr_gii = _load_label_vector(refreshed_label)
    if arr_npy.shape != arr_gii.shape or not np.array_equal(arr_npy, arr_gii):
        raise RuntimeError(f"materialize_combo still produced mismatched npy/gii labels: {manifest_path}")
    return refreshed


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def resolve_template_render_spec(manifest: dict) -> tuple[str | None, Path | None]:
    raw_spec = manifest.get("template_spec_path")
    if raw_spec is None:
        return None, None
    spec_text = str(raw_spec).strip()
    if spec_text == "" or spec_text.lower() in {"none", "null"}:
        return None, None
    spec_path = Path(spec_text)
    if not spec_path.exists():
        return None, None
    subject = str(manifest.get("template_subject", "")).replace("sub-", "").strip()
    if subject == "" or subject.lower() == "subject-consensus":
        return None, None
    return subject, spec_path


def style_imshow(ax, data: np.ndarray, title: str, *, vmin: float | None = None, vmax: float | None = None, cmap: str = "magma"):
    arr = np.asarray(data, dtype=np.float32)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("#f2f2f2")
    im = ax.imshow(arr, aspect="auto", interpolation="nearest", cmap=cm, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=FS_TITLE)
    return im


def add_cbar(fig, ax, im, label: str, ticks: list[float] | None = None, ticklabels: list[str] | None = None) -> None:
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="3%", pad=0.06)
    cb = fig.colorbar(im, cax=cax, ticks=ticks)
    if ticklabels is not None:
        cb.set_ticklabels(ticklabels)
    cb.ax.tick_params(labelsize=FS_TICK)
    cb.set_label(label, fontsize=FS_LABEL)


def concat_lr(a: np.ndarray, la: list[str], b: np.ndarray, lb: list[str]) -> tuple[np.ndarray, list[int], list[str]]:
    sep = np.full((1, a.shape[1]), np.nan, dtype=np.float32)
    mat = np.vstack([a, sep, b]).astype(np.float32)
    yt = list(range(a.shape[0])) + list(range(a.shape[0] + 1, a.shape[0] + 1 + b.shape[0]))
    yl = la + lb
    return mat, yt, yl


def ordered_cluster_ids_by_position(labels: np.ndarray, order: np.ndarray, k: int) -> list[int]:
    x = labels[order]
    rows: list[tuple[float, int]] = []
    for cid in range(1, int(k) + 1):
        idx = np.flatnonzero(x == cid)
        pos = float(np.mean(idx)) if idx.size else float("inf")
        rows.append((pos, int(cid)))
    rows.sort(key=lambda t: (t[0], t[1]))
    return [cid for _, cid in rows]


def resolve_cluster_order_from_manifest(manifest: dict, labels: np.ndarray, order: np.ndarray, k: int) -> list[int]:
    fixed = manifest.get("cluster_order_head_to_tail")
    if isinstance(fixed, list):
        vals = [int(x) for x in fixed]
        expect = set(range(1, int(k) + 1))
        if set(vals) == expect and len(vals) == int(k):
            return vals
    return ordered_cluster_ids_by_position(labels, order, k)


def labels_to_indicator_by_ids(labels: np.ndarray, order: np.ndarray, cluster_ids: list[int]) -> np.ndarray:
    out = np.zeros((len(cluster_ids), int(order.size)), dtype=np.float32)
    x = labels[order]
    for i, cid in enumerate(cluster_ids):
        out[i, :] = (x == int(cid)).astype(np.float32)
    return out


def reorder_profile(profile: np.ndarray, cluster_ids: list[int]) -> np.ndarray:
    idx = [int(cid) - 1 for cid in cluster_ids]
    return profile[np.asarray(idx, dtype=np.int32), :].astype(np.float32)


def annotate_profile_values(
    ax,
    data: np.ndarray,
    vmax: float,
    annotation_mask: np.ndarray | None = None,
) -> None:
    if vmax <= 0:
        return
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            if annotation_mask is not None and not bool(annotation_mask[r, c]):
                continue
            val = float(data[r, c])
            if not np.isfinite(val):
                continue
            color = "white" if abs(val) >= (0.55 * vmax) else "#222222"
            ax.text(c, r, f"{abs(val):.2f}", ha="center", va="center", fontsize=FS_CELL, color=color)


def highlight_top1_profile_cells(ax, data: np.ndarray) -> None:
    for r in range(data.shape[0]):
        row = data[r, :]
        if not np.isfinite(row).any():
            continue
        c = int(np.nanargmax(row))
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1.0, 1.0, fill=False, edgecolor="#111111", linewidth=1.1))


def _k_failure_lines(rows_tsv: list[dict[str, str]]) -> list[str]:
    rows = sorted(rows_tsv, key=lambda r: int(r["k"]))
    lines: list[str] = []
    for row in rows:
        if row.get("selected_final", "0") == "1":
            continue
        reasons: list[str] = []
        if row.get("reject_redundant", "0") == "1":
            reasons.append("redundant")
        if row.get("reject_nontrivial_fail", "0") == "1":
            reasons.append("nontrivial_fail")
        if reasons:
            lines.append(f"  K={int(row['k'])}: {','.join(reasons)}")
    if not lines and rows:
        lines.append("  No explicit rejection reasons found")
    return lines


def confidence_rows(status_payload: dict, hemi: str, cluster_ids: list[int]) -> list[dict[str, object]]:
    by_cluster = {int(row["cluster_id"]): row for row in status_payload.get("items", [])}
    top_n_cap_payload = status_payload.get("top_n_cap", None)
    out: list[dict[str, object]] = []
    for display_id, old_id in enumerate(cluster_ids, start=1):
        if int(old_id) not in by_cluster:
            continue
        row = by_cluster[int(old_id)]
        status = str(row.get("status", "dominant"))
        if "top1_count" in row and "top2_count" in row:
            top1_val = float(row.get("top1_count", 0.0))
            top2_val = float(row.get("top2_count", 0.0))
            margin = float(row.get("margin_count", top1_val - top2_val))
            top1_network = str(row.get("top1_network", "unassigned"))
            top2_network = str(row.get("top2_network", "none"))
            n_top = int(row.get("n_top", 0))
            # Backward-compatible display fix for older outputs where single-network clusters
            # duplicated 1st/2nd with equal counts.
            if top2_network == top1_network and (top2_val <= 0.0 or top2_val >= top1_val):
                top2_network = "none"
                top2_val = 0.0
                margin = float(top1_val)
            if top2_network in {"", "unassigned"} and top2_val <= 0.0:
                top2_network = "none"
            low_margin = bool((top2_network != "none") and (margin < 1.0))
            metric = "count"
        else:
            top1_val = float(row.get("top1_fisher_z", 0.0))
            top2_val = float(row.get("top2_fisher_z", 0.0))
            margin = float(row.get("margin", top1_val - top2_val))
            low_margin = bool(int(row.get("low_margin_lt_0p05", 1 if margin < 0.05 else 0)))
            top1_network = str(row.get("top1_network", "unassigned"))
            top2_network = str(row.get("top2_network", "unassigned"))
            n_top = None
            metric = "fisher"
        out.append(
            {
                "label": f"{hemi} C{int(display_id)}",
                "status": status,
                "top1_network": top1_network,
                "top2_network": top2_network,
                "top1_value": top1_val,
                "top2_value": top2_val,
                "margin": margin,
                "n_top": n_top,
                "top_n_cap": int(top_n_cap_payload) if top_n_cap_payload is not None else None,
                "metric": metric,
                "ambiguous": bool(status != "dominant" or low_margin),
            }
        )
    return out


def plot_confidence_panel(ax, rows: list[dict[str, object]], title: str) -> None:
    if not rows:
        ax.axis("off")
        ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=FS_MISSING)
        return
    metric = str(rows[0].get("metric", "fisher"))
    y = np.arange(len(rows), dtype=float)
    top1 = np.array([float(r["top1_value"]) for r in rows], dtype=float)
    top2 = np.array([float(r["top2_value"]) for r in rows], dtype=float)
    colors = []
    for r in rows:
        net = str(r.get("top1_network", "unassigned"))
        if net in {"", "none", "unassigned"}:
            colors.append("#9a9a9a")
        else:
            colors.append(str(NETWORK_BAR_COLORS.get(net, "#2c7fb8")))
    labels = []
    for r in rows:
        top1_name = str(r.get("top1_network", "unassigned"))
        top2_name = str(r.get("top2_network", ""))
        if top2_name in {"", "none", "unassigned"}:
            labels.append(f"{r['label']} {top1_name}")
        else:
            labels.append(f"{r['label']} {top1_name} | {top2_name}")
    ax.barh(y, top1, height=0.62, color=colors, alpha=0.88, label="1st")
    ax.barh(y, top2, height=0.32, color="none", edgecolor="#111111", linewidth=0.9, label="2nd")
    for yi, row in zip(y, rows, strict=False):
        ax.text(
            max(float(row["top1_value"]), float(row["top2_value"])) + (0.25 if metric == "count" else 0.005),
            yi,
            (f"Δn={int(round(float(row['margin'])))}" if metric == "count" else f"Δz={float(row['margin']):.3f}"),
            va="center",
            ha="left",
            fontsize=FS_TICK,
            color="#c97c00" if bool(row["ambiguous"]) else "#333333",
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=FS_TICK)
    ax.invert_yaxis()
    xmax = max(2.0, float(np.max(top1)) * 1.22) if metric == "count" else max(0.12, float(np.max(top1)) * 1.22)
    ax.set_xlim(0.0, xmax)
    if metric == "count":
        cap_vals = [int(r["top_n_cap"]) for r in rows if r.get("top_n_cap") is not None]
        cap_unique = sorted(set(cap_vals))
        if len(cap_unique) == 1 and cap_unique[0] > 0:
            xlabel = f"Top{cap_unique[0]} ROI count"
        else:
            xlabel = "TopN ROI count"
    else:
        xlabel = "Fisher-z"
    ax.set_xlabel(xlabel, fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    ax.axvline(1.0 if metric == "count" else 0.05, color="#999999", lw=0.8, ls=":")
    ax.grid(axis="x", color="#dddddd", lw=0.5, alpha=0.7)
    ax.legend(fontsize=FS_TICK, loc="lower right")


def plot_vertex_assignment_counts(
    ax,
    rows: list[dict[str, object]],
    title: str,
    *,
    title_fs: float = FS_TITLE,
    label_fs: float = FS_LABEL,
    tick_fs: float = FS_TICK,
) -> None:
    if not rows:
        ax.axis("off")
        ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=FS_MISSING)
        return
    counts: dict[str, int] = {}
    for row in rows:
        net = str(row.get("top1_network", "unassigned")) if str(row.get("status", "unassigned")) == "dominant" else "unassigned"
        counts[net] = int(counts.get(net, 0) + 1)
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    names = [k for k, _ in ordered][:10]
    vals = [v for _, v in ordered][:10]
    y = np.arange(len(names))
    ax.barh(y, vals, color="#2c7fb8", alpha=0.88)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=tick_fs)
    ax.invert_yaxis()
    ax.set_xlabel("Vertices", fontsize=label_fs)
    ax.set_title(title, fontsize=title_fs)
    ax.tick_params(labelsize=tick_fs)
    ax.grid(axis="x", color="#dddddd", lw=0.5, alpha=0.7)


def plot_vertex_confidence(ax, rows: list[dict[str, object]], title: str) -> None:
    if not rows:
        ax.axis("off")
        ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=FS_MISSING)
        return
    if "margin_count" in rows[0]:
        margins = np.array([float(r.get("margin_count", 0.0)) for r in rows], dtype=float)
        ax.hist(margins, bins=min(12, max(4, int(np.nanmax(margins) + 1))), color="#2c7fb8", alpha=0.85, edgecolor="#222222")
        ax.set_xlabel("TopN count margin (1st-2nd)", fontsize=FS_LABEL)
    else:
        margins = np.array([float(r.get("margin_score", 0.0)) for r in rows], dtype=float)
        ax.hist(margins, bins=12, color="#2c7fb8", alpha=0.85, edgecolor="#222222")
        ax.set_xlabel("Network mean-score margin (1st-2nd)", fontsize=FS_LABEL)
    ax.set_ylabel("Vertices", fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(axis="y", color="#dddddd", lw=0.5, alpha=0.7)


def _metric_array(rows: list[dict[str, str]], key: str) -> np.ndarray:
    vals: list[float] = []
    for row in rows:
        raw = row.get(key, "")
        try:
            vals.append(float(raw))
        except (TypeError, ValueError):
            vals.append(float("nan"))
    return np.asarray(vals, dtype=float)


def plot_k_gate_panel(
    ax,
    *,
    rows_l: list[dict[str, str]],
    rows_r: list[dict[str, str]],
    k_l: int,
    k_r: int,
    smoothing: str,
) -> None:
    rows_l = sorted(rows_l, key=lambda r: int(r["k"]))
    rows_r = sorted(rows_r, key=lambda r: int(r["k"]))
    ks_l = np.asarray([int(r["k"]) for r in rows_l], dtype=int)
    ks_r = np.asarray([int(r["k"]) for r in rows_r], dtype=int)
    max_sim_l = _metric_array(rows_l, "max_profile_similarity")
    max_sim_r = _metric_array(rows_r, "max_profile_similarity")

    has_metric = bool(np.isfinite(max_sim_l).any() or np.isfinite(max_sim_r).any())
    if not has_metric:
        ax.axis("off")
        ax.text(0.5, 0.5, f"{smoothing} metrics unavailable", ha="center", va="center", fontsize=FS_MISSING)
        return

    ax.plot(ks_l, max_sim_l, color="#1f77b4", lw=1.3, marker="o")
    ax.plot(ks_r, max_sim_r, color="#d62728", lw=1.3, marker="o")
    ax.axhline(0.80, color="#b00020", lw=1.0, ls=":", alpha=0.9)
    ax.axvline(int(k_l), color="#1f77b4", ls="--", lw=1.0)
    ax.axvline(int(k_r), color="#d62728", ls="--", lw=1.0)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("K", fontsize=FS_LABEL)
    ax.set_ylabel("max_profile_similarity", fontsize=FS_LABEL)
    ax.set_title(f"{smoothing} | redundancy gate", fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(axis="y", color="#dddddd", lw=0.5, alpha=0.7)
    ax.legend(
        handles=[
            Line2D([0], [0], color="#1f77b4", lw=1.6, marker="o", label="Left hemisphere"),
            Line2D([0], [0], color="#d62728", lw=1.6, marker="o", label="Right hemisphere"),
            Line2D([0], [0], color="#b00020", lw=1.2, ls=":", label="Redundancy threshold"),
            Line2D([0], [0], color="#555555", lw=1.2, ls="--", label="Selected K*"),
        ],
        loc="center",
        fontsize=FS_LEGEND_SMALL,
        framealpha=0.9,
        edgecolor="#dddddd",
        borderpad=0.3,
        labelspacing=0.28,
        handlelength=2.1,
    )


def plot_global_top_connections(ax, payload: dict[str, object], title: str) -> None:
    pos = [dict(x) for x in payload.get("top_positive", [])]
    neg = [dict(x) for x in payload.get("top_negative", [])]
    ax.axis("off")
    lines = [title, ""]
    lines.append(f"Top+ edges: {len(pos)}")
    lines.append(f"Top- edges: {len(neg)}")
    lines.append("")
    lines.append("Top+ preview:")
    for row in pos[:5]:
        lines.append(f"  v{int(row['vertex_index'])} -> {row['roi_name']} ({row['canonical8']}) z={float(row['fisher_z']):+.2f}")
    lines.append("")
    lines.append("Top- preview:")
    for row in neg[:5]:
        lines.append(f"  v{int(row['vertex_index'])} -> {row['roi_name']} ({row['canonical8']}) z={float(row['fisher_z']):+.2f}")
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=FS_SMALL, family="monospace")



def compute_wta_overview_profile_rows(
    *,
    labels_by_hemi: dict[str, np.ndarray],
    order_by_hemi: dict[str, np.ndarray],
    network_names: list[str],
    network_label_keys: dict[str, int],
    n_bins: int = WTA_OVERVIEW_PROFILE_BINS,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    network_keys = {int(network_label_keys[name]) for name in network_names}
    for hemi in sorted(labels_by_hemi):
        labels = np.asarray(labels_by_hemi[hemi]).reshape(-1)
        order = np.asarray(order_by_hemi[hemi], dtype=np.int64).reshape(-1)
        if labels.size != order.size:
            raise ValueError(f"Label/order size mismatch for hemi-{hemi}: labels={labels.size} order={order.size}")
        bins = [np.asarray(x, dtype=np.int64) for x in np.array_split(order, int(n_bins))]
        for bin_idx, vertices in enumerate(bins, start=1):
            bin_labels = labels[vertices] if vertices.size else np.asarray([], dtype=labels.dtype)
            denom = int(bin_labels.size)
            for network_name in network_names:
                key = int(network_label_keys[network_name])
                count = int(np.sum(bin_labels == key)) if denom else 0
                pct = (100.0 * float(count) / float(denom)) if denom else 0.0
                rows.append(
                    {
                        "hemi": str(hemi),
                        "bin_index": int(bin_idx),
                        "network_name": str(network_name),
                        "percent_of_bin_vertices": float(pct),
                    }
                )
            non_network_count = int(np.sum(~np.isin(bin_labels, list(network_keys)))) if denom else 0
            if non_network_count > 0:
                rows.append(
                    {
                        "hemi": str(hemi),
                        "bin_index": int(bin_idx),
                        "network_name": WTA_NON_NETWORK_NAME,
                        "percent_of_bin_vertices": 100.0 * float(non_network_count) / float(denom),
                    }
                )
    return rows


def plot_wta_overview_profile_axis(
    ax,
    *,
    rows: list[dict[str, object]],
    network_names: list[str],
    hemi: str,
    title: str,
    show_ylabel: bool,
) -> None:
    hemi_rows = [r for r in rows if str(r["hemi"]) == str(hemi)]
    if not hemi_rows:
        ax.axis("off")
        ax.text(0.5, 0.5, "profile missing", ha="center", va="center", fontsize=FS_MISSING)
        return
    max_bin = max(int(r["bin_index"]) for r in hemi_rows)
    plotted_names = list(network_names)
    if any(str(r["network_name"]) == WTA_NON_NETWORK_NAME for r in hemi_rows):
        plotted_names.append(WTA_NON_NETWORK_NAME)
    for network_name in plotted_names:
        pts = sorted(
            [r for r in hemi_rows if str(r["network_name"]) == network_name],
            key=lambda r: int(r["bin_index"]),
        )
        if not pts:
            continue
        xs = [float(r["percent_of_bin_vertices"]) for r in pts]
        ys = [int(r["bin_index"]) for r in pts]
        if network_name == WTA_NON_NETWORK_NAME:
            ax.plot(xs, ys, color="#777777", lw=0.9, ls="--", marker="o", markersize=1.9, label="non-network/null")
        else:
            ax.plot(
                xs,
                ys,
                color=NETWORK_BAR_COLORS.get(network_name),
                lw=1.15,
                marker="o",
                markersize=2.1,
                label=network_name,
            )
    ax.set_title(title, fontsize=WTA_OVERVIEW_SIDE_PANEL_TITLE_FS)
    ax.set_xlabel("Vertices in bin (%)", fontsize=WTA_OVERVIEW_SIDE_PANEL_LABEL_FS)
    ax.set_ylabel("Long-axis bin" if show_ylabel else "", fontsize=WTA_OVERVIEW_SIDE_PANEL_LABEL_FS)
    ax.set_ylim(max_bin + 0.5, 0.5)
    ax.set_xlim(left=0.0)
    ax.grid(True, axis="x", color="#dddddd", linewidth=0.5, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=WTA_OVERVIEW_SIDE_PANEL_TICK_FS)
    if show_ylabel:
        step = 5 if max_bin >= 20 else max(1, max_bin // 4)
        ticks = list(range(1, max_bin + 1, step))
        if ticks[-1] != max_bin:
            ticks.append(max_bin)
        ax.set_yticks(ticks)
    else:
        ax.set_yticklabels([])
    if not show_ylabel:
        ax.legend(fontsize=WTA_OVERVIEW_SIDE_PANEL_LEGEND_FS, frameon=False, loc="lower right", handlelength=1.4, labelspacing=0.22)

def render_combo(
    *,
    group_root: Path,
    present_dir: Path,
    method: str,
    atlas: str,
    coverage_tag: str,
    scene: Path,
    route_tag: str | None,
    mode_tag: str | None,
    assignment_variant: str = "fisherz",
    gradient_comp_tag: str = "",
) -> dict[str, object]:
    effective_mode_tag = (None if method in CLUSTERLESS_METHODS else mode_tag)
    effective_assignment_variant = None if method in CLUSTERLESS_METHODS else str(assignment_variant)
    if method in CLUSTERLESS_METHODS:
        fig = plt.figure(figsize=(19.0, 12.8), dpi=180, constrained_layout=True)
        gs = fig.add_gridspec(nrows=2, ncols=3, width_ratios=[0.62, 1.18, 1.42], hspace=0.08, wspace=0.06)
    else:
        fig = plt.figure(figsize=(15.4, 12.2), dpi=180, constrained_layout=True)
        gs = fig.add_gridspec(nrows=2, ncols=2, width_ratios=[1.40, 1.30], hspace=0.06, wspace=0.05)
    rows_meta: list[dict[str, object]] = []

    for ridx, smoothing in enumerate(SMOOTHINGS):
        left_manifest = _build_run_dir(group_root, method, atlas, smoothing, coverage_tag, "L", gradient_comp_tag=gradient_comp_tag) / "group_atlas_manifest.json"
        right_manifest = _build_run_dir(group_root, method, atlas, smoothing, coverage_tag, "R", gradient_comp_tag=gradient_comp_tag) / "group_atlas_manifest.json"
        if not (left_manifest.exists() and right_manifest.exists()):
            sub = gs[ridx, 0].subgridspec(2, 2, wspace=0.08, hspace=0.035)
            for rr in range(2):
                for cc in range(2):
                    ax = fig.add_subplot(sub[rr, cc])
                    ax.axis("off")
                    ax.text(0.5, 0.5, f"{smoothing} missing", ha="center", va="center", fontsize=FS_MISSING)
            ax_big = fig.add_subplot(gs[ridx, 1])
            ax_big.axis("off")
            ax_big.text(0.5, 0.5, f"{smoothing} missing", ha="center", va="center", fontsize=FS_MISSING)
            continue

        m_l = load_json(left_manifest)
        m_r = load_json(right_manifest)
        d_l = left_manifest.parent
        d_r = right_manifest.parent
        if method not in CLUSTERLESS_METHODS and effective_assignment_variant == "fisherz":
            m_l = ensure_functional_label_consistency(left_manifest, m_l)
            m_r = ensure_functional_label_consistency(right_manifest, m_r)
        if method in CLUSTERLESS_METHODS:
            status_l = load_json(d_l / "vertex_assignment_status.json") if (d_l / "vertex_assignment_status.json").exists() else {"items": []}
            status_r = load_json(d_r / "vertex_assignment_status.json") if (d_r / "vertex_assignment_status.json").exists() else {"items": []}
            rows_l = [dict(x) for x in status_l.get("items", [])]
            rows_r = [dict(x) for x in status_r.get("items", [])]
            rows_all = rows_l + rows_r
            ax_left = fig.add_subplot(gs[ridx, 0])
            plot_vertex_assignment_counts(
                ax_left,
                rows_all,
                f"{smoothing} | vertex assignment counts",
                title_fs=WTA_OVERVIEW_SIDE_PANEL_TITLE_FS,
                label_fs=WTA_OVERVIEW_SIDE_PANEL_LABEL_FS,
                tick_fs=WTA_OVERVIEW_SIDE_PANEL_TICK_FS,
            )

            surf_l = Path(str(m_l["template_left_or_right_surface"]))
            surf_r = Path(str(m_r["template_left_or_right_surface"]))
            profile_sub = gs[ridx, 1].subgridspec(1, 2, wspace=0.18)
            profile_axes = [fig.add_subplot(profile_sub[0, 0]), fig.add_subplot(profile_sub[0, 1])]
            try:
                label_map = load_json(d_l / "group_functional_atlas_label_map.json")
                network_names = [str(x) for x in label_map["network_names"]]
                network_label_keys = {str(k): int(v) for k, v in label_map["network_label_keys"].items()}
                labels_by_hemi = {
                    "L": np.load(d_l / "group_functional_atlas_labels.npy").astype(np.int32),
                    "R": np.load(d_r / "group_functional_atlas_labels.npy").astype(np.int32),
                }
                order_by_hemi = {"L": long_axis_order(surf_l), "R": long_axis_order(surf_r)}
                profile_rows = compute_wta_overview_profile_rows(
                    labels_by_hemi=labels_by_hemi,
                    order_by_hemi=order_by_hemi,
                    network_names=network_names,
                    network_label_keys=network_label_keys,
                )
                for ax_prof, hemi in zip(profile_axes, ["L", "R"]):
                    plot_wta_overview_profile_axis(
                        ax_prof,
                        rows=profile_rows,
                        network_names=network_names,
                        hemi=hemi,
                        title=f"{smoothing} | hemi-{hemi} long-axis profile",
                        show_ylabel=(hemi == "L"),
                    )
            except Exception as exc:
                for ax_prof in profile_axes:
                    ax_prof.axis("off")
                    ax_prof.text(0.5, 0.5, f"Profile unavailable\n{exc}", ha="center", va="center", fontsize=FS_SMALL)

            ax_big = fig.add_subplot(gs[ridx, 2])
            left_label = Path(str(m_l.get("template_functional_atlas_label", m_l.get("template_cluster_label"))))
            right_label = Path(str(m_r.get("template_functional_atlas_label", m_r.get("template_cluster_label"))))
            template_subject, spec_path = resolve_template_render_spec(m_l)
            if template_subject is None or spec_path is None:
                ax_big.text(
                    0.5,
                    0.5,
                    "Surface render skipped\n(template_spec_path unavailable in subject-consensus mode)",
                    ha="center",
                    va="center",
                    fontsize=FS_MISSING,
                )
            else:
                with tempfile.TemporaryDirectory(prefix="present_group_surface_render_") as tmp:
                    render = render_locked_grid(
                        subject=template_subject,
                        scene=scene,
                        outdir=Path(tmp),
                        name=f"merged_group_vertex_wta_{method}_{atlas}_{smoothing}",
                        title=f"Group functional atlas {method} {atlas} {smoothing} {effective_assignment_variant}",
                        left_labels=left_label,
                        right_labels=right_label,
                        left_surface=surf_l,
                        right_surface=surf_r,
                        spec_path=spec_path,
                    )
                    big = Path(str(render["biglegend_png"]))
                    ax_big.imshow(plt.imread(big))
            ax_big.set_title(f"{smoothing} | functional atlas surface", fontsize=FS_BIG_TITLE)
            ax_big.axis("off")
            rows_meta.append(
                {
                    "smoothing": smoothing,
                    "k_final_L": None,
                    "k_final_R": None,
                    "n_subjects": int(m_l["n_subjects"]),
                    "profile_bins": WTA_OVERVIEW_PROFILE_BINS,
                }
            )
            continue

        k_l_raw = m_l.get("chosen_k_final")
        k_r_raw = m_r.get("chosen_k_final")
        if k_l_raw is None or k_r_raw is None:
            sub = gs[ridx, 0].subgridspec(2, 2, wspace=0.08, hspace=0.035)
            ax0 = fig.add_subplot(sub[0, 0])
            rows_l_tsv = read_tsv(d_l / "k_selection_summary.tsv")
            rows_r_tsv = read_tsv(d_r / "k_selection_summary.tsv")
            for hemi_lbl, tsv_rows, color in [("L", rows_l_tsv, "#1f77b4"), ("R", rows_r_tsv, "#d62728")]:
                tsv_rows = sorted(tsv_rows, key=lambda r: int(r["k"]))
                ks = np.array([int(r["k"]) for r in tsv_rows], dtype=int)
                mean = np.array([float(r["group_ari_mean"]) for r in tsv_rows], dtype=float)
                se = np.array([float(r["group_ari_se"]) for r in tsv_rows], dtype=float)
                ax0.plot(ks, mean, marker="o", lw=1.3, color=color, label=hemi_lbl)
                ax0.fill_between(ks, mean - se, mean + se, color=color, alpha=0.18)
                null_ref = np.array([float(r["ARI_null_5pct"]) for r in tsv_rows], dtype=float)
                ax0.plot(ks, null_ref, ls=":", lw=1.0, color=color, alpha=0.85)
            if k_l_raw is not None:
                ax0.axvline(int(k_l_raw), color="#1f77b4", ls="--", lw=1.0)
            if k_r_raw is not None:
                ax0.axvline(int(k_r_raw), color="#d62728", ls="--", lw=1.0)
            ax0.set_title(f"{smoothing} | K selection (ARI)", fontsize=FS_TITLE)
            ax0.set_xlabel("K", fontsize=FS_LABEL)
            ax0.set_ylabel("Subject-split ARI", fontsize=FS_LABEL)
            ax0.tick_params(labelsize=FS_TICK)
            ax0.legend(
                handles=[
                    Line2D([0], [0], color="#1f77b4", lw=1.6, marker="o", label="Left hemisphere"),
                    Line2D([0], [0], color="#d62728", lw=1.6, marker="o", label="Right hemisphere"),
                    Line2D([0], [0], color="#555555", lw=1.2, ls=":", label="Null baseline"),
                ],
                loc="center",
                fontsize=FS_LEGEND_SMALL,
                framealpha=0.9,
                edgecolor="#dddddd",
                borderpad=0.3,
                labelspacing=0.28,
                handlelength=2.1,
            )
            diag_lines: list[str] = []
            for hemi_lbl, k_raw, d_hemi, tsv_rows in [
                ("L", k_l_raw, d_l, rows_l_tsv),
                ("R", k_r_raw, d_r, rows_r_tsv),
            ]:
                if k_raw is None:
                    diag_lines.append(f"hemi-{hemi_lbl}: NO K SELECTED")
                    diag_lines.extend(_k_failure_lines(read_tsv(d_hemi / "k_selection_summary.tsv")))
                else:
                    diag_lines.append(f"hemi-{hemi_lbl}: K={k_raw} selected (ok)")
            diag_text = "\n".join(diag_lines)
            for rr, cc in [(0, 1), (1, 0), (1, 1)]:
                ax_diag = fig.add_subplot(sub[rr, cc])
                ax_diag.axis("off")
                ax_diag.text(0.04, 0.96, diag_text, transform=ax_diag.transAxes,
                             ha="left", va="top", fontsize=FS_SMALL, family="monospace")
            ax_big = fig.add_subplot(gs[ridx, 1])
            ax_big.axis("off")
            ax_big.set_title(f"{smoothing} | functional atlas surface", fontsize=FS_BIG_TITLE)
            none_hemis = "/".join(h for h, k in [("L", k_l_raw), ("R", k_r_raw)] if k is None)
            ax_big.text(0.5, 0.5, f"No K selected\n(hemi-{none_hemis})",
                        ha="center", va="center", fontsize=FS_MISSING, color="#cc0000")
            rows_meta.append({"smoothing": smoothing, "k_final_L": k_l_raw,
                               "k_final_R": k_r_raw, "n_subjects": int(m_l["n_subjects"])})
            continue

        k_l = int(k_l_raw)
        k_r = int(k_r_raw)
        prof_l = np.load(d_l / "cluster_network_profile.npy").astype(np.float32)
        prof_r = np.load(d_r / "cluster_network_profile.npy").astype(np.float32)
        lbl_l = np.load(d_l / "group_labels_final.npy").astype(np.int32)
        lbl_r = np.load(d_r / "group_labels_final.npy").astype(np.int32)
        assert effective_assignment_variant is not None
        status_l_path, left_label_variant = resolve_assignment_variant_paths(d_l, m_l, effective_assignment_variant)
        status_r_path, right_label_variant = resolve_assignment_variant_paths(d_r, m_r, effective_assignment_variant)
        if not status_l_path.exists() or not status_r_path.exists():
            raise FileNotFoundError(
                f"Missing assignment variant status for {method} {atlas} {smoothing} variant={effective_assignment_variant}: "
                f"left={status_l_path} right={status_r_path}"
            )
        if not left_label_variant.exists() or not right_label_variant.exists():
            raise FileNotFoundError(
                f"Missing assignment variant functional labels for {method} {atlas} {smoothing} variant={effective_assignment_variant}: "
                f"left={left_label_variant} right={right_label_variant}"
            )
        status_l = load_json(status_l_path)
        status_r = load_json(status_r_path)
        surf_l = Path(str(m_l["template_left_or_right_surface"]))
        surf_r = Path(str(m_r["template_left_or_right_surface"]))
        order_l = long_axis_order(surf_l)
        order_r = long_axis_order(surf_r)
        cids_l = resolve_cluster_order_from_manifest(m_l, lbl_l, order_l, k_l)
        cids_r = resolve_cluster_order_from_manifest(m_r, lbl_r, order_r, k_r)
        prof_l_o = reorder_profile(prof_l, cids_l)
        prof_r_o = reorder_profile(prof_r, cids_r)
        row_lbl_l = [f"L-C{i}" for i in range(1, len(cids_l) + 1)]
        row_lbl_r = [f"R-C{i}" for i in range(1, len(cids_r) + 1)]

        sub = gs[ridx, 0].subgridspec(2, 2, wspace=0.08, hspace=0.035)
        ax0 = fig.add_subplot(sub[0, 0])
        rows_l = read_tsv(d_l / "k_selection_summary.tsv")
        rows_r = read_tsv(d_r / "k_selection_summary.tsv")
        for hemi, rows, color in [("L", rows_l, "#1f77b4"), ("R", rows_r, "#d62728")]:
            rows = sorted(rows, key=lambda r: int(r["k"]))
            ks = np.array([int(r["k"]) for r in rows], dtype=int)
            mean = np.array([float(r["group_ari_mean"]) for r in rows], dtype=float)
            se = np.array([float(r["group_ari_se"]) for r in rows], dtype=float)
            ax0.plot(ks, mean, marker="o", lw=1.3, color=color, label=hemi)
            ax0.fill_between(ks, mean - se, mean + se, color=color, alpha=0.18)
            null_ref = np.array([float(r["ARI_null_5pct"]) for r in rows], dtype=float)
            ax0.plot(ks, null_ref, ls=":", lw=1.0, color=color, alpha=0.85)
        ax0.axvline(k_l, color="#1f77b4", ls="--", lw=1.0)
        ax0.axvline(k_r, color="#d62728", ls="--", lw=1.0)
        ax0.legend(
            handles=[
                Line2D([0], [0], color="#1f77b4", lw=1.6, marker="o", label="Left hemisphere"),
                Line2D([0], [0], color="#d62728", lw=1.6, marker="o", label="Right hemisphere"),
                Line2D([0], [0], color="#555555", lw=1.2, ls=":", label="Null baseline"),
                Line2D([0], [0], color="#555555", lw=1.2, ls="--", label="Selected K*"),
            ],
            loc="center",
            fontsize=FS_LEGEND_SMALL,
            framealpha=0.9,
            edgecolor="#dddddd",
            borderpad=0.3,
            labelspacing=0.28,
            handlelength=2.1,
        )
        ax0.set_title(f"{smoothing} | K selection (ARI)", fontsize=FS_TITLE)
        ax0.set_xlabel("K", fontsize=FS_LABEL)
        ax0.set_ylabel("Subject-split ARI", fontsize=FS_LABEL)
        ax0.tick_params(labelsize=FS_TICK)

        ax1 = fig.add_subplot(sub[0, 1])
        nets = [str(x) for x in load_json(d_l / "group_feature_summary.json")["network_names"]]
        profile_title = f"{smoothing} | cluster_network_profile (signed)"
        prof_lr, yt_prof, yl_prof = concat_lr(prof_l_o, row_lbl_l, prof_r_o, row_lbl_r)
        vmax = float(np.nanmax(np.abs(prof_lr))) if np.isfinite(prof_lr).any() else 1.0
        im1 = style_imshow(ax1, prof_lr, profile_title, vmin=-vmax, vmax=vmax, cmap="coolwarm")
        ax1.set_yticks(yt_prof)
        ax1.set_yticklabels(yl_prof, fontsize=FS_LABEL)
        ax1.set_xticks(np.arange(len(nets)))
        ax1.set_xticklabels(nets, fontsize=FS_TICK, rotation=35, ha="right")
        annotate_profile_values(ax1, prof_lr, vmax)
        highlight_top1_profile_cells(ax1, prof_lr)
        add_cbar(fig, ax1, im1, "Fisher-z")

        ax2 = fig.add_subplot(sub[1, 0])
        plot_k_gate_panel(
            ax2,
            rows_l=rows_l,
            rows_r=rows_r,
            k_l=k_l,
            k_r=k_r,
            smoothing=smoothing,
        )

        ax3 = fig.add_subplot(sub[1, 1])
        conf_rows = confidence_rows(status_l, "L", cids_l) + confidence_rows(status_r, "R", cids_r)
        plot_confidence_panel(ax3, conf_rows, f"{smoothing} | cluster confidence / ambiguity")

        ax_big = fig.add_subplot(gs[ridx, 1])
        left_label = left_label_variant
        right_label = right_label_variant
        template_subject, spec_path = resolve_template_render_spec(m_l)
        if template_subject is None or spec_path is None:
            ax_big.text(
                0.5,
                0.5,
                "Surface render skipped\n(template_spec_path unavailable in subject-consensus mode)",
                ha="center",
                va="center",
                fontsize=FS_MISSING,
            )
        else:
            with tempfile.TemporaryDirectory(prefix="present_group_surface_render_") as tmp:
                render = render_locked_grid(
                    subject=template_subject,
                    scene=scene,
                    outdir=Path(tmp),
                    name=f"merged_group_cluster_wta_{method}_{atlas}_{smoothing}",
                    title=f"Group functional atlas {method} {atlas} {smoothing} {effective_assignment_variant}",
                    left_labels=left_label,
                    right_labels=right_label,
                    left_surface=surf_l,
                    right_surface=surf_r,
                    spec_path=spec_path,
                )
                big = Path(str(render["biglegend_png"]))
                ax_big.imshow(plt.imread(big))
        ax_big.set_title(f"{smoothing} | functional atlas surface", fontsize=FS_BIG_TITLE)
        ax_big.axis("off")

        rows_meta.append(
            {
                "smoothing": smoothing,
                "k_final_L": k_l,
                "k_final_R": k_r,
                "n_subjects": int(m_l["n_subjects"]),
            }
        )

    name_bits = [atlas, method, "group_overview"]
    if gradient_comp_tag:
        name_bits.append(gradient_comp_tag)
    if route_tag:
        name_bits.append(route_tag)
    if effective_mode_tag:
        name_bits.append(effective_mode_tag)
    if effective_assignment_variant:
        name_bits.append(effective_assignment_variant)
    out_png = present_dir / f"{'_'.join(name_bits)}.png"
    fig.suptitle(
        f"Group Overview | {atlas} | {method}"
        + (f" | {route_tag}" if route_tag else "")
        + (f" | {effective_mode_tag}" if effective_mode_tag else "")
        + (f" | {effective_assignment_variant}" if effective_assignment_variant else ""),
        fontsize=FS_SUPTITLE,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    return {"overview_png": str(out_png), "rows": rows_meta}


def write_readme(present_dir: Path, items: list[dict[str, object]]) -> None:
    lines = [
        "# HCP 7T hippocampus group overviews",
        "",
        "| atlas | method | route | mode | assignment | overview | smoothings | key metrics |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for item in sorted(items, key=lambda x: (str(x["atlas"]), str(x["method"]))):
        png = Path(str(item["overview_png"])).name
        sm = ", ".join(str(s) for s in item.get("smoothings", []))
        metric = str(item.get("metric_summary", ""))
        route = str(item.get("route_tag", ""))
        mode = str(item.get("mode_tag", ""))
        assignment = str(item.get("assignment_variant", ""))
        lines.append(f"| {item['atlas']} | {item['method']} | {route} | {mode} | {assignment} | [{png}]({png}) | {sm} | {metric} |")
    (present_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_combo_ready(
    *,
    group_root: Path,
    method: str,
    atlas: str,
    coverage_tag: str,
    smoothings: list[str] | None = None,
    hemis: list[str] | None = None,
) -> None:
    sm = smoothings or ["2mm", "4mm"]
    hs = hemis or ["L", "R"]
    problems: list[str] = []
    for smoothing in sm:
        comp_tags = _discover_gradient_comp_tags(group_root, method, atlas, smoothing, coverage_tag)
        for comp_tag in comp_tags:
            for hemi in hs:
                run_dir = _build_run_dir(group_root, method, atlas, smoothing, coverage_tag, hemi, gradient_comp_tag=comp_tag)
                manifest_path = run_dir / "group_atlas_manifest.json"
                if not manifest_path.exists():
                    problems.append(f"missing_manifest: {manifest_path}")
                    continue
                m = load_json(manifest_path)
                if str(m.get("status", "")) != "ok":
                    problems.append(f"manifest_not_ok: {manifest_path} status={m.get('status')}")
                    continue
                # For cluster-based methods, group labels must exist.
                if method not in CLUSTERLESS_METHODS:
                    labels = run_dir / "group_labels_final.npy"
                    if not labels.exists():
                        problems.append(f"missing_group_labels: {labels}")
    if problems:
        raise RuntimeError(
            "Cluster outputs are not ready. Do not skip clustering before present rendering.\n"
            + "\n".join(problems)
        )


def main() -> int:
    p = argparse.ArgumentParser(description="Build canonical HCP 7T hippocampus group overviews from group atlas outputs.")
    p.add_argument("--group-root", default=str(DEFAULT_GROUP_ROOT))
    p.add_argument("--present-dir", default=str(DEFAULT_PRESENT_DIR))
    p.add_argument("--methods", nargs="+", default=METHODS_DEFAULT, choices=METHOD_CHOICES)
    p.add_argument("--atlases", nargs="+", default=ATLASES)
    p.add_argument("--coverage-threshold", type=float, default=0.80)
    p.add_argument("--scene", default=str(DEFAULT_SCENE))
    p.add_argument("--route-tag", choices=ROUTE_TAG_CHOICES, default=None)
    p.add_argument("--mode-tag", default=None)
    p.add_argument("--assignment-variant", choices=ASSIGNMENT_VARIANTS, default="fisherz")
    p.add_argument(
        "--manifest-name",
        default="group_overview_manifest.json",
        help="Output manifest filename under --present-dir.",
    )
    p.add_argument(
        "--skip-readme",
        action="store_true",
        help="Skip writing README.md (useful when an upstream driver writes a consolidated README).",
    )
    p.add_argument(
        "--skip-manifest",
        action="store_true",
        help="Skip writing manifest JSON (useful when an upstream driver writes a consolidated manifest).",
    )
    p.add_argument(
        "--allow-missing-combos",
        action="store_true",
        help="Allow rendering placeholders when group-atlas outputs are missing. "
        "Default is strict: require complete clustering outputs before rendering.",
    )
    args = p.parse_args()
    if args.mode_tag is not None:
        mode_tag = str(args.mode_tag)
        if mode_tag != "full" and re.fullmatch(r"(?:[1-9][0-9]?|100)perc", mode_tag) is None:
            raise ValueError(
                f"--mode-tag must be 'full' or '<int>perc' where int is in [1,100], got {args.mode_tag}"
            )

    group_root = Path(args.group_root).resolve()
    present_dir = Path(args.present_dir).resolve()
    present_dir.mkdir(parents=True, exist_ok=True)
    tau_tag = f"cov_{args.coverage_threshold:.2f}".replace(".", "p")

    items: list[dict[str, object]] = []
    for atlas in args.atlases:
        for method in args.methods:
            if not bool(args.allow_missing_combos):
                ensure_combo_ready(
                    group_root=group_root,
                    method=method,
                    atlas=atlas,
                    coverage_tag=tau_tag,
                )
            comp_tags = _discover_gradient_comp_tags(
                group_root, method, atlas, SMOOTHINGS[0], tau_tag,
            )
            for comp_tag in comp_tags:
                meta = render_combo(
                    group_root=group_root,
                    present_dir=present_dir,
                    method=method,
                    atlas=atlas,
                    coverage_tag=tau_tag,
                    scene=Path(args.scene).resolve(),
                    route_tag=args.route_tag,
                    mode_tag=args.mode_tag,
                    assignment_variant=str(args.assignment_variant),
                    gradient_comp_tag=comp_tag,
                )
                row_meta = meta["rows"]
                metric = (
                    "; ".join(
                        [
                            f"{r['smoothing']}: K(L/R)={('-' if r['k_final_L'] is None else r['k_final_L'])}/"
                            f"{('-' if r['k_final_R'] is None else r['k_final_R'])} N={r['n_subjects']}"
                            for r in row_meta
                        ]
                    )
                    if row_meta
                    else "missing"
                )
                items.append(
                    {
                        "atlas": atlas,
                        "method": method,
                        "gradient_comp_tag": comp_tag or None,
                        "overview_png": meta["overview_png"],
                        "smoothings": [r["smoothing"] for r in row_meta],
                        "metric_summary": metric,
                        "route_tag": args.route_tag,
                        "mode_tag": (None if method in CLUSTERLESS_METHODS else args.mode_tag),
                        "assignment_variant": (None if method in CLUSTERLESS_METHODS else str(args.assignment_variant)),
                        "source_group_root": str(group_root),
                    }
                )

    manifest = {"present_dir": str(present_dir), "count": len(items), "items": items}
    if not bool(args.skip_manifest):
        (present_dir / args.manifest_name).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not bool(args.skip_readme):
        write_readme(present_dir, items)
    print(json.dumps({"present_dir": str(present_dir), "count": len(items)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

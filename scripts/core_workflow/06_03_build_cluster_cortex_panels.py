#!/usr/bin/env python3
"""
Process Step Description:
Build cortical projection panels for clusters. Generates visualization panel data projecting hippocampal clustered networks onto the cerebral cortex surface to show hippocampal-cortical connectivity patterns.

Main Flow Path:
- 9th step in the numbered pipeline sequence (first stage of presentation output generation, run in parallel with 06_01, 06_02, 06_04, 06_05, 06_06, 06_07):
  01_01_run_hippunfold_t1w_local.py -> 01_02_check_hippunfold_outputs.py -> 02_01_prepare_group_inputs.py -> 02_02_build_qc_manifest.py -> 03_01/03_02 reference building -> 04_01/04_02 FC materialization -> 05_01 gradient computation -> 05_02 group atlas parcellation -> 06_03_build_cluster_cortex_panels.py (this step) -> 07_01 group matrix overview.

Inputs:
- Command-line arguments:
  - --group-root: Root directory containing group atlas results (default: `outputs_hcp/work/group_atlas`).
  - --shared-root: Root directory containing shared atlas reference files (default: `outputs_hcp/work/kong2022_reference_routea`).
  - --present-dir: Output directory for final presentation figures (default: `outputs_hcp/present/cortex_panels`).
  - --runtime-root: Root for intermediate files and index layout (default: `<present-dir>/runtime`).
  - --atlas: Cortical atlas choice (default: "kong2022", direct canonical7).
  - --methods: Cluster-based methods to process (default: ["intrinsic-spectral", "network-spectral", "gradient-spectral"]).
  - --smoothings: Smoothing kernel values to process (default: ["2mm", "4mm"]).
  - --hemis: Hemispheres to process (default: ["L", "R"]).
  - --coverage-threshold: Coverage threshold to process (default: 0.80).
  - --fdr-q: FDR q-threshold for one-sample t-test (default: 0.05).
  - --max-rois-per-sign: Maximum cortical ROIs to display per positive/negative sign (default: 20).
  - --cortex-template-subject: Subject ID used as cortical template (default: "100610").
  - --data-root: Source of exact `sub-<cortex-template-subject>/anat/` rendering inputs (default: `data/hippunfold_input`).
  - --scene: Cortical workbench manual scene template (default: `config/manual_wb_scenes/cortex_manual.scene`).
- Physical inputs:
  - Shared network colors: `config/hipp_network_style.json`
  - Group-level clusters: `group_labels_final.npy` under `{group_root}/{method}/smooth-{smoothing}_hemi-{hemi}_cov-{threshold}/`.
  - Subject-level cluster-to-cortex functional connectivity matrices.
  - Bilateral fsLR den-32k MSMAll inflated surfaces and sulc dscalar under `{data_root}/sub-<cortex-template-subject>/anat/`.

Outputs:
- File outputs:
  - Bilateral sulc metrics split from that subject's dscalar: `{runtime_root}/cortex_assets/`.
  - Standalone cortical projection panel figures: `{runtime_root}/combos/{combo_id}/cluster_{cluster_id:02d}_panel.png`
  - ROI statistical outputs: `{runtime_root}/combos/{combo_id}/cluster_{cluster_id:02d}_stats.tsv` and `.json`
  - Combination-level manifest: `{runtime_root}/combos/{combo_id}/cortex_panel_combo_manifest.json`
  - Merged overview panel figures: `{present_dir}/cortex_{atlas}_{method}_overview.png` or `{present_dir}/cortex_{atlas}_{method}_{smoothing}_overview.png`
  - Manifests:
    - Runtime manifest: `{runtime_root}/cluster_roi_cortex_runtime_manifest.json`
    - Viewer manifest: `{runtime_root}/cluster_roi_cortex_manifest.json`
    - Presentation manifest: `{present_dir}/cluster_roi_cortex_overview_manifest.json`
  - Index README: `{runtime_root}/README.md` and `{present_dir}/README.md`
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib  # type: ignore[import-untyped]
import nibabel as nib
import numpy as np
from lib.gradient_clustering import is_gradient_method
from lib.cortex_template_profiles import (
    compute_subject_cluster_roi_profiles,
    parse_roi_registry,
    resolve_cortex_assets,
    subject_reference_path,
)
from lib.fc_math import bh_fdr
from lib.gifti_labels import make_named_label_gifti
from lib.io_utils import load_json, write_tsv
from lib.kong2022_label_utils import load_kong2022_bundle
from lib.network_colors import load_network_colors
from lib.output_layout import (
    DEFAULT_CORTEX_PANELS_DIR,
    DEFAULT_KONG2022_ROUTEA_ROOT,
    DEFAULT_ROUTEA_GROUP_ATLAS_ROOT,
)
from lib.render_cortex_pfm_scene import render_cortex_pfm_scene
from lib.route_tags import (
    ROUTE_TAG_CHOICES,
    infer_route_tag_from_group_root,
    validate_group_atlas_root_for_route,
)
from lib.workflow_utils import write_json
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore[import-untyped]  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_GROUP_ROOT = DEFAULT_ROUTEA_GROUP_ATLAS_ROOT
DEFAULT_SHARED_ROOT = DEFAULT_KONG2022_ROUTEA_ROOT
DEFAULT_PRESENT_DIR = DEFAULT_CORTEX_PANELS_DIR
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "hippunfold_input"
DEFAULT_SCENE = REPO_ROOT / "config" / "manual_wb_scenes" / "cortex_manual.scene"
DEFAULT_HIPP_SCENE = REPO_ROOT / "config" / "wb_locked_native_view_lateral_medial.scene"

METHOD_CHOICES = [
    "network-spectral",
    "gradient-spectral",
    "gradient-ward",
    "gradient-gmm",
    "gradient-spectral-rbf",
    "intrinsic-spectral",
]
METHODS_DEFAULT = ["intrinsic-spectral", "network-spectral", "gradient-spectral"]
SMOOTHINGS = ["2mm", "4mm"]
HEMIS = ["L", "R"]
VIEW_SPECS = [
    {"name": "lateral", "axis": None, "deg": 0.0},
    {"name": "medial", "axis": "y", "deg": -90.0},
    {"name": "dorsal", "axis": "x", "deg": 90.0},
    {"name": "ventral", "axis": "x", "deg": -90.0},
]


@dataclass(frozen=True)
class Combo:
    atlas: str
    method: str
    smoothing: str
    hemi: str
    gradient_comp_tag: str = ""


GRADIENT_METHOD = "gradient-spectral"


def _is_gradient_method(method: str) -> bool:
    return is_gradient_method(method)


def _discover_gradient_comp_tags(
    group_root: Path,
    method: str,
    atlas: str,
    smoothing: str,
    tau_tag: str,
) -> list[str]:
    """Discover gradient component subdirectories under the coverage directory."""
    if not _is_gradient_method(method):
        return [""]
    base = group_root / method / atlas / smoothing / tau_tag
    if not base.exists():
        return [""]
    tags = sorted(d.name for d in base.iterdir() if d.is_dir() and d.name.startswith("g"))
    return tags if tags else [""]


def _build_run_dir(
    group_root: Path,
    combo: Combo,
    tau_tag: str,
) -> Path:
    """Build the run directory path, including gradient component subdir for gradient-spectral."""
    if _is_gradient_method(combo.method) and combo.gradient_comp_tag:
        return group_root / combo.method / combo.atlas / combo.smoothing / tau_tag / combo.gradient_comp_tag / f"hemi_{combo.hemi}"
    return group_root / combo.method / combo.atlas / combo.smoothing / tau_tag / f"hemi_{combo.hemi}"


@dataclass(frozen=True)
class ROIStat:
    roi_id: int
    roi_name: str
    hemi: str
    family17: str
    canonical7: str
    canonical8: str
    mean_z: float
    t_stat: float
    p_positive: float
    p_negative: float
    q_positive: float
    q_negative: float
    n_subjects: int
    is_sig_positive: int
    is_sig_negative: int
    is_selected: int
    selected_sign: str


@dataclass(frozen=True)
class PanelComboInputs:
    combo: Combo
    combo_id: str
    run_dir: Path
    manifest: dict[str, Any]
    subjects: list[str]
    labels_path: Path
    support_path: Path
    registry_path: Path



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build standalone cluster-to-cortex top-ROI figures (independent from overview) "
            "for Kong2022 group-atlas combos."
        )
    )
    parser.add_argument("--group-root", default=str(DEFAULT_GROUP_ROOT))
    parser.add_argument("--shared-root", default=str(DEFAULT_SHARED_ROOT))
    parser.add_argument("--reference-root", default=None, help="Read cortex reference files from this root; defaults to --shared-root.")
    parser.add_argument("--mask-root", default=None, help="Read hippocampal valid masks from this root; defaults to --shared-root.")
    parser.add_argument("--present-dir", default=str(DEFAULT_PRESENT_DIR))
    parser.add_argument(
        "--runtime-root",
        default=None,
        help="Root for all outputs (cluster figures, manifests, quicklook index). Defaults to <present-dir>/runtime.",
    )
    parser.add_argument("--atlas", default="kong2022", choices=["kong2022"])
    parser.add_argument("--route-tag", default="auto", choices=["auto", *ROUTE_TAG_CHOICES])
    parser.add_argument("--methods", nargs="+", default=METHODS_DEFAULT, choices=METHOD_CHOICES)
    parser.add_argument("--smoothings", nargs="+", default=SMOOTHINGS)
    parser.add_argument("--hemis", nargs="+", default=HEMIS)
    parser.add_argument("--coverage-threshold", type=float, default=0.80)
    parser.add_argument("--fdr-q", type=float, default=0.05)
    parser.add_argument("--max-rois-per-sign", type=int, default=20)
    parser.add_argument("--cortex-template-subject", default="100610")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT),
                        help="Root of exact sub-<cortex-template-subject>/anat cortex rendering inputs.")
    parser.add_argument("--scene", default=str(DEFAULT_SCENE))
    return parser.parse_args()


def load_shared_network_colors() -> dict[str, tuple[int, int, int, int]]:
    return load_network_colors()


def neg_color_rgba(rgba: tuple[int, int, int, int], blend: float = 0.72) -> tuple[int, int, int, int]:
    """Lighten network color toward white to signal negative FC."""
    r, g, b, a = rgba
    rr = int(round(r + (255 - r) * blend))
    gg = int(round(g + (255 - g) * blend))
    bb = int(round(b + (255 - b) * blend))
    return rr, gg, bb, a


def display_color_for_sign(base: tuple[int, int, int, int], sign: str) -> tuple[int, int, int, int]:
    if sign == "positive":
        return base
    if sign == "negative":
        return neg_color_rgba(base)
    return base



def one_sample_one_sided_stats(x: np.ndarray) -> tuple[float, float, float, float, int]:
    vals = np.asarray(x, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    n = int(vals.size)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0
    mean = float(np.mean(vals))
    if n < 2:
        return mean, float("nan"), float("nan"), float("nan"), n
    sd = float(np.std(vals, ddof=1))
    if not np.isfinite(sd) or sd <= 1e-12:
        if mean > 0.0:
            return mean, float("inf"), 0.0, 1.0, n
        if mean < 0.0:
            return mean, float("-inf"), 1.0, 0.0, n
        return mean, 0.0, 0.5, 0.5, n
    t_stat = mean / (sd / math.sqrt(float(n)))
    p_pos = float(stats.t.sf(t_stat, df=n - 1))
    p_neg = float(stats.t.cdf(t_stat, df=n - 1))
    return mean, float(t_stat), p_pos, p_neg, n


def analyze_cluster(
    *,
    cluster_values: np.ndarray,
    roi_registry: list[dict[str, Any]],
    fdr_q: float,
    max_rois_per_sign: int,
) -> tuple[list[ROIStat], list[dict[str, Any]], list[dict[str, Any]]]:
    n_roi = int(cluster_values.shape[1])
    mean_z = np.full((n_roi,), np.nan, dtype=np.float64)
    t_stat = np.full((n_roi,), np.nan, dtype=np.float64)
    p_pos = np.full((n_roi,), np.nan, dtype=np.float64)
    p_neg = np.full((n_roi,), np.nan, dtype=np.float64)
    n_subj = np.zeros((n_roi,), dtype=np.int32)

    for ridx in range(n_roi):
        mz, ts_val, pp, pn, n = one_sample_one_sided_stats(cluster_values[:, ridx])
        mean_z[ridx] = mz
        t_stat[ridx] = ts_val
        p_pos[ridx] = pp
        p_neg[ridx] = pn
        n_subj[ridx] = int(n)

    q_pos = bh_fdr(p_pos)
    q_neg = bh_fdr(p_neg)
    sig_pos = np.isfinite(q_pos) & (q_pos < float(fdr_q)) & np.isfinite(mean_z) & (mean_z > 0.0)
    sig_neg = np.isfinite(q_neg) & (q_neg < float(fdr_q)) & np.isfinite(mean_z) & (mean_z < 0.0)

    pos_idx = np.flatnonzero(sig_pos)
    neg_idx = np.flatnonzero(sig_neg)
    pos_order = pos_idx[np.argsort(-mean_z[pos_idx])] if pos_idx.size else np.array([], dtype=np.int32)
    neg_order = neg_idx[np.argsort(mean_z[neg_idx])] if neg_idx.size else np.array([], dtype=np.int32)
    pos_take = pos_order[: int(max_rois_per_sign)]
    neg_take = neg_order[: int(max_rois_per_sign)]

    selected_sign = np.full((n_roi,), "", dtype=object)
    selected_sign[pos_take] = "positive"
    selected_sign[neg_take] = "negative"

    rows: list[ROIStat] = []
    selected_pos: list[dict[str, Any]] = []
    selected_neg: list[dict[str, Any]] = []
    for ridx, reg in enumerate(roi_registry):
        stat_row = ROIStat(
            roi_id=int(reg["roi_id"]),
            roi_name=str(reg["roi_name"]),
            hemi=str(reg["hemi"]),
            family17=str(reg["family17"]),
            canonical7=str(reg["canonical7"]),
            canonical8=str(reg["canonical8"]),
            mean_z=float(mean_z[ridx]) if np.isfinite(mean_z[ridx]) else float("nan"),
            t_stat=float(t_stat[ridx]) if np.isfinite(t_stat[ridx]) else float("nan"),
            p_positive=float(p_pos[ridx]) if np.isfinite(p_pos[ridx]) else float("nan"),
            p_negative=float(p_neg[ridx]) if np.isfinite(p_neg[ridx]) else float("nan"),
            q_positive=float(q_pos[ridx]) if np.isfinite(q_pos[ridx]) else float("nan"),
            q_negative=float(q_neg[ridx]) if np.isfinite(q_neg[ridx]) else float("nan"),
            n_subjects=int(n_subj[ridx]),
            is_sig_positive=int(bool(sig_pos[ridx])),
            is_sig_negative=int(bool(sig_neg[ridx])),
            is_selected=int(selected_sign[ridx] != ""),
            selected_sign=str(selected_sign[ridx]),
        )
        rows.append(stat_row)
        if selected_sign[ridx] == "positive":
            selected_pos.append(
                {
                    "roi_id": stat_row.roi_id,
                    "roi_name": stat_row.roi_name,
                    "canonical7": stat_row.canonical7,
                    "canonical8": stat_row.canonical8,
                    "family17": stat_row.family17,
                    "mean_z": stat_row.mean_z,
                    "q": stat_row.q_positive,
                    "sign": "positive",
                }
            )
        elif selected_sign[ridx] == "negative":
            selected_neg.append(
                {
                    "roi_id": stat_row.roi_id,
                    "roi_name": stat_row.roi_name,
                    "canonical7": stat_row.canonical7,
                    "canonical8": stat_row.canonical8,
                    "family17": stat_row.family17,
                    "mean_z": stat_row.mean_z,
                    "q": stat_row.q_negative,
                    "sign": "negative",
                }
            )

    return rows, selected_pos, selected_neg


def build_cluster_label_files(
    *,
    selected_items: list[dict[str, Any]],
    cluster_dir: Path,
    vertex_arrays: dict[str, tuple[np.ndarray, np.ndarray]],
    network_colors: dict[str, tuple[int, int, int, int]],
) -> tuple[Path, Path, list[dict[str, Any]]]:
    by_roi_id = {int(item["roi_id"]): item for item in selected_items}
    label_items: list[tuple[int, str, tuple[int, int, int, int]]] = [(0, "Null", (158, 158, 158, 255))]
    legend_items: list[dict[str, Any]] = []
    for item in selected_items:
        roi_id = int(item["roi_id"])
        canonical = str(item["canonical7"])
        if canonical not in network_colors:
            raise KeyError(f"Missing network color for {canonical!r}")
        base = network_colors[canonical]
        sign = str(item["sign"])
        color = display_color_for_sign(base, sign)
        sign_tag = "+" if sign == "positive" else "-"
        label_name = f"{sign_tag} {item['roi_name']}"
        label_items.append((roi_id, label_name, color))
        legend_items.append({"name": f"{sign_tag} {item['roi_name']} ({canonical})", "rgba": color})

    cluster_dir.mkdir(parents=True, exist_ok=True)
    wb_dir = cluster_dir / "workbench_assets"
    wb_dir.mkdir(parents=True, exist_ok=True)

    label_paths: dict[str, Path] = {}
    for hemi in ["L", "R"]:
        vertex_idx, parcel_ids = vertex_arrays[hemi]
        n_vertices = int(np.max(vertex_idx)) + 1
        values = np.zeros((n_vertices,), dtype=np.int32)
        if by_roi_id:
            selected_ids = np.asarray(sorted(by_roi_id), dtype=np.int32)
            keep = np.isin(parcel_ids.astype(np.int32), selected_ids)
            values[vertex_idx[keep].astype(np.int32)] = parcel_ids[keep].astype(np.int32)
        img = make_named_label_gifti(values, label_items)
        suffix = "L" if hemi == "L" else "R"
        out_path = wb_dir / f"cluster_selected_rois.{suffix}.label.gii"
        nib.save(img, str(out_path))
        label_paths[hemi] = out_path

    return label_paths["L"], label_paths["R"], legend_items


def render_cortex_single_image(
    *,
    cluster_dir: Path,
    left_label: Path,
    right_label: Path,
    cortex_assets: dict[str, Path],
) -> Path:
    views_dir = cluster_dir / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    view = VIEW_SPECS[0]
    out_scene = views_dir / f"cluster_rois_{view['name']}.scene"
    out_png_single = views_dir / f"cluster_rois_{view['name']}.png"
    if not out_png_single.exists():
        render_cortex_pfm_scene(
            source_scene=cortex_assets["scene"],
            left_surface=cortex_assets["left_inflated"], right_surface=cortex_assets["right_inflated"],
            left_label=left_label, right_label=right_label,
            left_sulc=cortex_assets["left_sulc"], right_sulc=cortex_assets["right_sulc"],
            sulc_dscalar=cortex_assets["sulc_dscalar"], dlabel=cortex_assets["dlabel"],
            out_scene=out_scene, out_png=out_png_single, width=1000, height=700,
            rotation_axis=view["axis"], rotation_deg=float(view["deg"]),
        )
    return out_png_single



def render_cluster_figure(
    *,
    cluster_dir: Path,
    title: str,
    left_label: Path,
    right_label: Path,
    selected_items: list[dict[str, Any]],
    network_colors: dict[str, tuple[int, int, int, int]],
    n_selected_positive: int,
    n_selected_negative: int,
    cortex_assets: dict[str, Path],
) -> Path:
    out_png_single = render_cortex_single_image(
        cluster_dir=cluster_dir, left_label=left_label,
        right_label=right_label, cortex_assets=cortex_assets,
    )

    out_png = cluster_dir / "cluster_cortex_toproi_4view.png"
    img = plt.imread(out_png_single)
    fig = plt.figure(figsize=(16.4, 9.0), dpi=180, constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[4.9, 2.3], wspace=0.02)
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(img)
    ax_img.axis("off")
    ax_img.set_title(title, fontsize=18, loc="left", pad=6.0)

    ax_leg = fig.add_subplot(gs[0, 1])
    ax_leg.axis("off")
    total_selected = int(n_selected_positive) + int(n_selected_negative)
    ax_leg.text(
        0.0,
        0.98,
        f"Selected ROI: total={total_selected}  (+{int(n_selected_positive)} / -{int(n_selected_negative)})",
        ha="left",
        va="top",
        fontsize=22,
        fontweight="bold",
    )
    ax_leg.text(0.0, 0.90, "Sign coding:", ha="left", va="top", fontsize=22, fontweight="bold")
    ax_leg.text(0.0, 0.84, "+ : network color (saturated)", ha="left", va="top", fontsize=21)
    ax_leg.text(0.0, 0.78, "- : pale tint of network color", ha="left", va="top", fontsize=21)

    net_counts: dict[str, dict[str, int]] = {}
    for item in selected_items:
        net = str(item.get("canonical7", "Unknown"))
        sign = str(item.get("sign", ""))
        row = net_counts.setdefault(net, {"positive": 0, "negative": 0})
        if sign == "positive":
            row["positive"] += 1
        elif sign == "negative":
            row["negative"] += 1

    if not net_counts:
        ax_leg.text(0.0, 0.68, "No significant ROI selected.", ha="left", va="top", fontsize=22)
    else:
        ax_leg.text(0.0, 0.68, "Network legend:", ha="left", va="top", fontsize=22, fontweight="bold")
        y = 0.63
        step = 0.064
        ordered_nets = sorted(net_counts.keys())
        for net in ordered_nets[:12]:
            if net not in network_colors:
                raise KeyError(f"Missing network color for {net!r}")
            base = network_colors[net]
            neg = neg_color_rgba(base)
            c_pos = tuple(x / 255.0 for x in base[:3]) + (1.0,)
            c_neg = tuple(x / 255.0 for x in neg[:3]) + (1.0,)
            ax_leg.add_patch(plt.Rectangle(
                (0.0, y - 0.028), 0.052, 0.042,
                transform=ax_leg.transAxes, facecolor=c_pos, edgecolor="none"))
            ax_leg.add_patch(plt.Rectangle(
                (0.060, y - 0.028), 0.052, 0.042,
                transform=ax_leg.transAxes, facecolor=c_neg, edgecolor="none"))
            c = net_counts[net]
            label = f"{net} (+{int(c['positive'])}/-{int(c['negative'])})"
            ax_leg.text(0.124, y, label, ha="left", va="center", fontsize=21, transform=ax_leg.transAxes)
            y -= step
            if y < 0.02:
                break

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_png


def roi_stats_to_rows(stats_rows: list[ROIStat]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in stats_rows:
        out.append(
            {
                "roi_id": row.roi_id,
                "roi_name": row.roi_name,
                "hemi": row.hemi,
                "family17": row.family17,
                "canonical7": row.canonical7,
                "canonical8": row.canonical8,
                "mean_z": row.mean_z,
                "t_stat": row.t_stat,
                "p_positive": row.p_positive,
                "p_negative": row.p_negative,
                "q_positive": row.q_positive,
                "q_negative": row.q_negative,
                "n_subjects": row.n_subjects,
                "is_sig_positive": row.is_sig_positive,
                "is_sig_negative": row.is_sig_negative,
                "is_selected": row.is_selected,
                "selected_sign": row.selected_sign,
            }
        )
    return out


def coverage_tag(tau: float) -> str:
    return f"cov_{tau:.2f}".replace(".", "p")


def combo_tag(combo: Combo) -> str:
    base = f"{combo.atlas}__{combo.method}__{combo.smoothing}__hemi-{combo.hemi}"
    if combo.gradient_comp_tag:
        base = f"{base}__{combo.gradient_comp_tag}"
    return base


def write_viewer_readme(
    *,
    viewer_root: Path,
    present_dir: Path,
    runtime_root: Path,
    items: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    grouped_overviews: list[dict[str, Any]],
) -> None:
    lines = [
        "# cluster_roi_cortex quicklook",
        "",
        "This directory is a shallow viewer root.",
        f"- runtime_root: `{runtime_root}`",
        f"- present_root: `{present_dir}`",
        f"- rendered combos: {len(items)}",
        f"- skipped combos: {len(skipped)}",
        "",
        "Quick entry points:",
        "- `present_root/cortex_*_overview.png` significance-filtered cluster-to-cortex collages per atlas/method/smoothing",
        "- `cluster_roi_cortex_manifest.json` index linking viewer and runtime manifests",
        "",
    ]
    if grouped_overviews:
        lines.append("Top-level merged-hemisphere overviews:")
        for row in grouped_overviews:
            lines.append(f"- `{Path(str(row['overview_png'])).name}`")
        lines.append("")
    (viewer_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def compose_combo_overview(
    *,
    cluster_previews: list[dict[str, Any]],
    out_png: Path,
    title: str,
    stats_note: str | None = None,
) -> None:
    if not cluster_previews:
        fig, ax = plt.subplots(1, 1, figsize=(8.0, 4.5), dpi=180)
        ax.axis("off")
        ax.text(0.5, 0.5, "No cluster preview", ha="center", va="center", fontsize=12)
        ax.set_title(title, fontsize=12)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return

    n_panels = int(len(cluster_previews))
    if n_panels <= 2:
        cols = n_panels
    elif n_panels <= 4:
        cols = 2
    elif n_panels <= 9:
        cols = 3
    else:
        cols = 4
    rows = int(math.ceil(n_panels / float(cols)))

    # Preserve source aspect ratio to avoid vertical squeezing.
    ref_img = plt.imread(Path(str(cluster_previews[0]["panel_png"])))
    ref_aspect = float(ref_img.shape[1]) / float(ref_img.shape[0])  # w / h
    cell_w = 8.5
    cell_h = cell_w / max(ref_aspect, 1e-6)
    suptitle_h = 1.20
    fig_w = cols * cell_w
    fig_h = rows * cell_h + suptitle_h
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=180, constrained_layout=False)
    gs = fig.add_gridspec(rows, cols, left=0.01, right=0.99, bottom=0.02, top=0.82, wspace=0.01, hspace=0.03)

    for idx, item in enumerate(cluster_previews):
        if idx >= rows * cols:
            break
        rr = idx // cols
        cc = idx % cols
        ax = fig.add_subplot(gs[rr, cc])
        png = Path(str(item["panel_png"]))
        img = plt.imread(png)
        ax.imshow(img, interpolation="nearest")
        ax.set_aspect("equal")
        label = str(item.get("panel_label", f"C{int(item['cluster_id']):02d}"))
        n_pos = int(item.get("n_selected_positive", 0))
        n_neg = int(item.get("n_selected_negative", 0))
        n_tot = int(n_pos + n_neg)
        ax.set_title(f"{label} | n={n_tot} (+{n_pos}/-{n_neg})", fontsize=15)
        ax.axis("off")

    if not stats_note:
        fig.suptitle(title, fontsize=20)
    else:
        fig.suptitle(title, fontsize=20, y=0.965)
        fig.text(
            0.5,
            0.925,
            f"({stats_note})",
            ha="center",
            va="top",
            fontsize=18,
        )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def collect_panel_combo_inputs(
    *,
    combo: Combo,
    group_root: Path,
    tau_tag: str,
    skipped: list[dict[str, Any]],
) -> PanelComboInputs | None:
    run_dir = _build_run_dir(group_root, combo, tau_tag)
    manifest_path = run_dir / "group_atlas_manifest.json"
    labels_path = run_dir / "group_labels_final.npy"
    support_path = run_dir / "group_support_mask.npy"
    registry_path = run_dir / "cluster_roi_registry.tsv"
    combo_id = combo_tag(combo)

    if not manifest_path.exists():
        skipped.append(
            {
                "atlas": combo.atlas,
                "method": combo.method,
                "smoothing": combo.smoothing,
                "hemi": combo.hemi,
                "reason": "missing_group_manifest",
                "run_dir": str(run_dir),
            }
        )
        return None

    manifest = load_json(manifest_path)
    group_status = str(manifest.get("status", "ok"))
    if group_status != "ok":
        skipped.append(
            {
                "atlas": combo.atlas,
                "method": combo.method,
                "smoothing": combo.smoothing,
                "hemi": combo.hemi,
                "reason": "group_combo_not_ok",
                "group_status": group_status,
                "group_skip_reason": manifest.get("skip_reason"),
                "run_dir": str(run_dir),
            }
        )
        return None

    required_ok = labels_path.exists() and support_path.exists() and registry_path.exists()
    if not required_ok:
        skipped.append(
            {
                "atlas": combo.atlas,
                "method": combo.method,
                "smoothing": combo.smoothing,
                "hemi": combo.hemi,
                "reason": "missing_required_group_atlas_artifacts",
                "run_dir": str(run_dir),
            }
        )
        return None

    subjects = [str(s).replace("sub-", "") for s in manifest.get("subjects", [])]
    if not subjects:
        skipped.append(
            {
                "atlas": combo.atlas,
                "method": combo.method,
                "smoothing": combo.smoothing,
                "hemi": combo.hemi,
                "reason": "empty_subjects_in_manifest",
                "run_dir": str(run_dir),
            }
        )
        return None

    return PanelComboInputs(
        combo=combo,
        combo_id=combo_id,
        run_dir=run_dir,
        manifest=manifest,
        subjects=subjects,
        labels_path=labels_path,
        support_path=support_path,
        registry_path=registry_path,
    )


def write_cortex_panel_manifests(
    *,
    combo: Combo,
    combo_id: str,
    combo_runtime_out: Path,
    viewer_combo_manifest_dir: Path,
    combo_manifest: dict[str, Any],
    cluster_previews: list[dict[str, Any]],
    n_subjects_used: int,
) -> dict[str, Any]:
    combo_manifest_path = combo_runtime_out / "cluster_roi_cortex_combo_manifest.json"
    write_json(combo_manifest_path, combo_manifest)
    combo_view_manifest_path = viewer_combo_manifest_dir / f"{combo_id}.json"
    write_json(
        combo_view_manifest_path,
        {
            "atlas": combo.atlas,
            "method": combo.method,
            "smoothing": combo.smoothing,
            "hemi": combo.hemi,
            "combo_id": combo_id,
            "runtime_combo_manifest": str(combo_manifest_path),
            "clusters": cluster_previews,
        },
    )
    return {
        "atlas": combo.atlas,
        "method": combo.method,
        "smoothing": combo.smoothing,
        "hemi": combo.hemi,
        "combo_manifest_runtime": str(combo_manifest_path),
        "combo_manifest_viewer": str(combo_view_manifest_path),
        "cluster_count": int(combo_manifest.get("cluster_count", 0)),
        "n_subjects_used": int(n_subjects_used),
    }



def render_clustered_combo_panels(
    *,
    panel_inputs: PanelComboInputs,
    args: argparse.Namespace,
    shared_root: Path,
    reference_root: Path,
    mask_root: Path,
    runtime_root: Path,
    tau_tag: str,
    cortex_assets: dict[str, Path],
    vertex_arrays: dict[str, tuple[np.ndarray, np.ndarray]],
    network_colors: dict[str, tuple[int, int, int, int]],
    viewer_combo_manifest_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], tuple[str, str, str]]:
    combo = panel_inputs.combo
    labels = np.load(panel_inputs.labels_path).astype(np.int32)
    support = np.load(panel_inputs.support_path).astype(bool)
    if labels.ndim != 1 or support.ndim != 1 or int(labels.size) != int(support.size):
        raise RuntimeError(f"Invalid labels/support shape in {panel_inputs.run_dir}")

    roi_registry = parse_roi_registry(panel_inputs.registry_path)
    profile_3d, used_subjects = compute_subject_cluster_roi_profiles(
        subjects=panel_inputs.subjects,
        shared_root=shared_root,
        reference_root=reference_root,
        mask_root=mask_root,
        smoothing=combo.smoothing,
        hemi=combo.hemi,
        group_labels=labels,
        support_mask=support,
        roi_registry=roi_registry,
    )

    k_final = int(np.max(labels))
    combo_runtime_out = runtime_root / combo.atlas / combo.method / combo.smoothing / tau_tag / f"hemi_{combo.hemi}"
    combo_runtime_out.mkdir(parents=True, exist_ok=True)

    cluster_items: list[dict[str, Any]] = []
    cluster_previews: list[dict[str, Any]] = []
    for cluster_id in range(1, k_final + 1):
        cluster_dir = combo_runtime_out / f"cluster_C{cluster_id:02d}"
        cluster_vals = profile_3d[:, cluster_id - 1, :]
        stats_rows, selected_pos, selected_neg = analyze_cluster(
            cluster_values=cluster_vals,
            roi_registry=roi_registry,
            fdr_q=float(args.fdr_q),
            max_rois_per_sign=int(args.max_rois_per_sign),
        )
        selected_items = selected_pos + selected_neg
        left_label, right_label, _legend_items = build_cluster_label_files(
            selected_items=selected_items,
            cluster_dir=cluster_dir,
            vertex_arrays=vertex_arrays,
            network_colors=network_colors,
        )

        title = f"cluster C{cluster_id:02d} | {combo.atlas} | {combo.method} | {combo.smoothing} | hemi-{combo.hemi}"
        figure_png = render_cluster_figure(
            cluster_dir=cluster_dir,
            title=title,
            left_label=left_label,
            right_label=right_label,
            selected_items=selected_items,
            network_colors=network_colors,
            n_selected_positive=int(len(selected_pos)),
            n_selected_negative=int(len(selected_neg)),
            cortex_assets=cortex_assets,
        )

        rows_dict = roi_stats_to_rows(stats_rows)
        stats_tsv = cluster_dir / "cluster_roi_stats.tsv"
        stats_json = cluster_dir / "cluster_roi_stats.json"
        write_tsv(
            stats_tsv,
            rows_dict,
            [
                "roi_id",
                "roi_name",
                "hemi",
                "family17",
                "canonical7",
                "canonical8",
                "mean_z",
                "t_stat",
                "p_positive",
                "p_negative",
                "q_positive",
                "q_negative",
                "n_subjects",
                "is_sig_positive",
                "is_sig_negative",
                "is_selected",
                "selected_sign",
            ],
        )
        write_json(
            stats_json,
            {
                "cluster_id": cluster_id,
                "selected_positive": selected_pos,
                "selected_negative": selected_neg,
                "n_selected_total": int(len(selected_items)),
                "rows": rows_dict,
            },
        )

        cluster_previews.append(
            {
                "atlas": combo.atlas,
                "method": combo.method,
                "smoothing": combo.smoothing,
                "hemi": combo.hemi,
                "cluster_id": int(cluster_id),
                "runtime_png": str(figure_png),
                "n_selected_positive": int(len(selected_pos)),
                "n_selected_negative": int(len(selected_neg)),
                "n_subjects_used": int(len(used_subjects)),
            }
        )

        cluster_items.append(
            {
                "cluster_id": cluster_id,
                "figure_png": str(figure_png),
                "stats_tsv": str(stats_tsv),
                "stats_json": str(stats_json),
                "n_selected_positive": int(len(selected_pos)),
                "n_selected_negative": int(len(selected_neg)),
            }
        )

    combo_manifest = {
        "atlas": combo.atlas,
        "method": combo.method,
        "smoothing": combo.smoothing,
        "hemi": combo.hemi,
        "coverage_threshold": float(args.coverage_threshold),
        "fdr_q": float(args.fdr_q),
        "selection_policy": "sig_only_topN_per_sign",
        "max_rois_per_sign": int(args.max_rois_per_sign),
        "n_subjects_used": int(len(used_subjects)),
        "subjects_used": [f"sub-{s}" for s in used_subjects],
        "cluster_count": int(k_final),
        "clusters": cluster_items,
    }
    item = write_cortex_panel_manifests(
        combo=combo,
        combo_id=panel_inputs.combo_id,
        combo_runtime_out=combo_runtime_out,
        viewer_combo_manifest_dir=viewer_combo_manifest_dir,
        combo_manifest=combo_manifest,
        cluster_previews=cluster_previews,
        n_subjects_used=len(used_subjects),
    )
    return item, cluster_previews, (combo.atlas, combo.method, combo.smoothing)


def main() -> int:
    args = parse_args()

    group_root = Path(args.group_root).resolve()
    if args.route_tag == "auto":
        args.route_tag = infer_route_tag_from_group_root(group_root)
    else:
        validate_group_atlas_root_for_route(group_root, str(args.route_tag))
    shared_root = Path(args.shared_root).resolve()
    reference_root = Path(args.reference_root).resolve() if args.reference_root else shared_root
    mask_root = Path(args.mask_root).resolve() if args.mask_root else shared_root
    present_dir = Path(args.present_dir).resolve()
    present_dir.mkdir(parents=True, exist_ok=True)
    runtime_root = Path(args.runtime_root).resolve() if args.runtime_root else ((present_dir / "runtime") if str(args.route_tag) == "pfm" else (present_dir / str(args.route_tag) / "runtime"))
    viewer_root = runtime_root
    runtime_root.mkdir(parents=True, exist_ok=True)
    viewer_combo_manifest_dir = runtime_root / "combo_manifests"
    viewer_combo_manifest_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root).resolve()
    scene = Path(args.scene).resolve()
    cortex_assets = resolve_cortex_assets(
        subject=str(args.cortex_template_subject), data_root=data_root, scene=scene,
        sulc_dir=runtime_root / "cortex_assets",
    )

    tau_tag = coverage_tag(float(args.coverage_threshold))
    network_colors = load_shared_network_colors()
    bundle = load_kong2022_bundle()
    vertex_arrays = {
        "L": (bundle["vertex_arrays"]["L"][0].astype(np.int32), bundle["vertex_arrays"]["L"][1].astype(np.int32)),
        "R": (bundle["vertex_arrays"]["R"][0].astype(np.int32), bundle["vertex_arrays"]["R"][1].astype(np.int32)),
    }

    combos: list[Combo] = []
    for method in args.methods:
        for smoothing in args.smoothings:
            comp_tags = _discover_gradient_comp_tags(group_root, method, str(args.atlas), smoothing, tau_tag)
            for comp_tag in comp_tags:
                for hemi in args.hemis:
                    combos.append(Combo(atlas=str(args.atlas), method=str(method), smoothing=str(smoothing), hemi=str(hemi), gradient_comp_tag=comp_tag))

    items: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    grouped_previews: dict[tuple[str, str, str], list[dict[str, Any]]] = {}

    for combo in combos:
        panel_inputs = collect_panel_combo_inputs(
            combo=combo,
            group_root=group_root,
            tau_tag=tau_tag,
            skipped=skipped,
        )
        if panel_inputs is None:
            continue

        rendered = render_clustered_combo_panels(
            panel_inputs=panel_inputs,
            args=args,
            shared_root=shared_root,
            reference_root=reference_root,
            mask_root=mask_root,
            runtime_root=runtime_root,
            tau_tag=tau_tag,
            cortex_assets=cortex_assets,
            vertex_arrays=vertex_arrays,
            network_colors=network_colors,
            viewer_combo_manifest_dir=viewer_combo_manifest_dir,
        )
        item, cluster_previews, group_key = rendered
        items.append(item)
        grouped_previews.setdefault(group_key, []).extend(cluster_previews)

    grouped_overviews: list[dict[str, Any]] = []
    smoothing_rank = {"2mm": 0, "4mm": 1, "2mm+4mm": 2}
    for (atlas, method, smoothing), rows in sorted(grouped_previews.items(), key=lambda x: x[0]):
        merged_rows = sorted(
            rows,
            key=lambda r: (
                int(smoothing_rank.get(str(r.get("smoothing", "")), 99)),
                str(r.get("hemi", "")),
                int(r.get("cluster_id", 0)),
            ),
        )
        previews_for_plot: list[dict[str, Any]] = []
        for r in merged_rows:
            previews_for_plot.append(
                {
                    "cluster_id": int(r["cluster_id"]),
                    "panel_png": str(r["runtime_png"]),
                    "panel_label": str(r.get("panel_label", f"{str(r['hemi'])}-C{int(r['cluster_id']):02d}")),
                    "n_selected_positive": int(r.get("n_selected_positive", 0)),
                    "n_selected_negative": int(r.get("n_selected_negative", 0)),
                }
            )
        route_suffix = f"_{str(args.route_tag)}"
        if str(smoothing) == "2mm+4mm":
            overview_png = present_dir / f"cortex_{atlas}_{method}{route_suffix}_overview.png"
            title_txt = f"{atlas} | {method} | {args.route_tag} | 2mm+4mm merged | L+R merged"
        else:
            overview_png = present_dir / f"cortex_{atlas}_{method}{route_suffix}_{smoothing}_overview.png"
            title_txt = f"{atlas} | {method} | {args.route_tag} | {smoothing} | L+R merged"
        n_subjects_used_vals = sorted(
            {
                int(r.get("n_subjects_used", 0))
                for r in merged_rows
                if int(r.get("n_subjects_used", 0)) > 0
            }
        )
        n_used = int(n_subjects_used_vals[0]) if len(n_subjects_used_vals) == 1 else (
            int(max(n_subjects_used_vals)) if n_subjects_used_vals else 0
        )
        stats_note = (
            f"one-sided one-sample t-test + BH-FDR(q<{float(args.fdr_q):.2f}); "
            f"sig-only; max {int(args.max_rois_per_sign)}/sign; N={n_used}"
        )
        compose_combo_overview(
            cluster_previews=previews_for_plot,
            out_png=overview_png,
            title=title_txt,
            stats_note=stats_note,
        )
        grouped_overviews.append(
            {
                "atlas": atlas,
                "method": method,
                "smoothing": smoothing,
                "overview_png": str(overview_png),
                "n_panels": int(len(previews_for_plot)),
            }
        )

    runtime_manifest = {
        "stage": "cluster_roi_cortex_independent",
        "route_tag": str(args.route_tag),
        "atlas": str(args.atlas),
        "coverage_threshold": float(args.coverage_threshold),
        "fdr_q": float(args.fdr_q),
        "max_rois_per_sign": int(args.max_rois_per_sign),
        "runtime_root": str(runtime_root),
        "viewer_root": str(viewer_root),
        "count": int(len(items)),
        "items": items,
        "skipped": skipped,
        "grouped_overviews": grouped_overviews,
    }
    runtime_manifest_path = runtime_root / "cluster_roi_cortex_runtime_manifest.json"
    write_json(runtime_manifest_path, runtime_manifest)

    viewer_manifest = dict(runtime_manifest)
    viewer_manifest["runtime_manifest"] = str(runtime_manifest_path)
    viewer_manifest["viewer_layout"] = {
        "present_root_overview_pattern": str(present_dir / "cortex_*_overview.png"),
        "combo_manifest_dir": str(viewer_combo_manifest_dir),
    }
    viewer_manifest_path = viewer_root / "cluster_roi_cortex_manifest.json"
    write_json(viewer_manifest_path, viewer_manifest)
    present_overview_manifest_path = present_dir / "cluster_roi_cortex_overview_manifest.json"
    write_json(
        present_overview_manifest_path,
        {
            "stage": "cluster_roi_cortex_overview",
            "viewer_manifest": str(viewer_manifest_path),
            "runtime_manifest": str(runtime_manifest_path),
            "grouped_overviews": grouped_overviews,
            "count": int(len(grouped_overviews)),
        },
    )
    write_viewer_readme(
        viewer_root=viewer_root,
        present_dir=present_dir,
        runtime_root=runtime_root,
        items=items,
        skipped=skipped,
        grouped_overviews=grouped_overviews,
    )
    print(
        json.dumps(
            {
                "runtime_root": str(runtime_root),
                "viewer_root": str(viewer_root),
                "present_root": str(present_dir),
                "count": len(items),
                "skipped": len(skipped),
                "grouped_overviews": len(grouped_overviews),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

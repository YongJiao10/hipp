#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib.patches import Rectangle

from run_subject import EVAL_K, REPO_ROOT, is_soft_branch, render_locked_grid_png, save_combined_label_assets


NETWORK_SHORT = {
    "Auditory": "AUD",
    "Control": "FPCN",
    "CinguloOpercular/Action-mode": "CO/Act",
    "Default": "DMN",
    "Default_Anterolateral": "Def-AL",
    "Default_Dorsolateral": "Def-DL",
    "Default_Parietal": "Def-Par",
    "Default_Retrosplenial": "Def-Rsp",
    "DorsalAttention": "DAN",
    "Frontoparietal": "FPN",
    "Language": "LN",
    "Limbic": "LIM",
    "MedialParietal": "MedPar",
    "Premotor/DorsalAttentionII": "Prem/DAN2",
    "Salience": "Sal",
    "SomatoCognitiveAction": "SCA",
    "Somatomotor_Face": "SM-Face",
    "Somatomotor_Foot": "SM-Foot",
    "Somatomotor_Hand": "SM-Hand",
    "Somatomotor": "SMN",
    "TemporalParietal": "TPN",
    "VentralAttention": "VAN",
    "Visual": "VN",
    "Visual_Dorsal/VentralStream": "Vis-DV",
    "Visual_Lateral": "Vis-Lat",
    "Visual_V1": "V1",
    "Visual_V5": "V5",
}

SMOOTH_LABEL = {
    "2mm": "2mm",
    "4mm": "4mm",
}

HEMIS = ["L", "R"]
SMOOTHS = ["2mm", "4mm"]
PLOT_SPEC = {
    "canvas_w": 5200,
    "margin": 80,
    "gutter": 28,
    "section_gap": 42,
    "title_h": 64,
    "panel_title_h": 66,
    "title_font": 40,
    "panel_title_font": 32,
    "metric_font": 15,
    "tick_font": 11,
}
RENDER_SHORTLIST = [2, 3, 4, 5, 6]
SMOOTH_COLORS = {"2mm": "#f58518", "4mm": "#54a24b"}
SOFT_DIAG_COLORS = {
    ("2mm", "mean_probabilities"): "#4c78a8",
    ("2mm", "mean_regularized_probabilities"): "#f58518",
    ("4mm", "mean_probabilities"): "#72b7b2",
    ("4mm", "mean_regularized_probabilities"): "#e45756",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_summary_stage_manifest(root: Path, params: dict[str, object], inputs: list[Path], outputs: list[Path]) -> None:
    payload = {
        "stage": "summary",
        "status": "done",
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "params": params,
        "inputs": [
            {
                "path": str(path.resolve()),
                "size": int(path.stat().st_size),
                "mtime_ns": int(path.stat().st_mtime_ns),
            }
            for path in sorted(inputs, key=lambda p: str(p.resolve()))
        ],
        "outputs": [str(path.resolve()) for path in outputs],
    }
    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "stage_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def copy_key_images(root: Path, final_selection: dict[str, object]) -> list[Path]:
    copied: list[Path] = []
    mapping = {
        root / f"functional_{smooth}.png": Path(final_selection["per_smooth"][smooth]["final_png"])
        for smooth in SMOOTHS
    }
    mapping[root / "structural_locked.png"] = Path(final_selection["structural_png"])
    for dst, src in mapping.items():
        if src.exists():
            shutil.copyfile(src, dst)
            copied.append(dst)
    return copied


def trim_black(img: Image.Image, threshold: int = 8) -> Image.Image:
    arr = np.asarray(img)
    mask = np.any(arr > threshold, axis=2)
    if not np.any(mask):
        return img
    ys, xs = np.where(mask)
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return img.crop(box)


def foreground_mask(image: Image.Image, threshold: int = 15) -> np.ndarray:
    arr = np.asarray(image.convert("RGB"))
    return arr.sum(axis=2) > threshold


def find_runs(flags: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(flags.tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(flags)))
    return runs


def split_native_hemi_panels(path: Path) -> dict[str, Image.Image]:
    img = Image.open(path).convert("RGB")
    mask = foreground_mask(img)
    col_runs = find_runs(mask.any(axis=0))
    if len(col_runs) < 2:
        raise RuntimeError(f"Could not detect bilateral foreground runs in native render: {path}")
    selected_runs = sorted(col_runs, key=lambda item: item[1] - item[0], reverse=True)[:2]
    selected_runs = sorted(selected_runs, key=lambda item: item[0])
    panels: dict[str, Image.Image] = {}
    for hemi, (x0, x1) in zip(HEMIS, selected_runs, strict=True):
        crop = img.crop((x0, 0, x1, img.height))
        panels[hemi] = trim_black(crop)
    return panels


def load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for candidate in [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/SFNS.ttf",
    ]:
        path = Path(candidate)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def fit_image(path: Path, width: int, max_height: int | None = None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    scale = width / img.width
    height = max(1, int(round(img.height * scale)))
    if max_height is not None and height > max_height:
        scale = max_height / img.height
        width = max(1, int(round(img.width * scale)))
        height = max_height
    return img.resize((width, height), Image.Resampling.LANCZOS)


def fit_image_obj(img: Image.Image, width: int, max_height: int | None = None) -> Image.Image:
    scale = width / img.width
    height = max(1, int(round(img.height * scale)))
    if max_height is not None and height > max_height:
        scale = max_height / img.height
        width = max(1, int(round(img.width * scale)))
        height = max_height
    return img.resize((width, height), Image.Resampling.LANCZOS)


def paste_center(canvas: Image.Image, img: Image.Image, x: int, y: int, cell_w: int, cell_h: int) -> None:
    offset_x = x + max(0, (cell_w - img.width) // 2)
    offset_y = y + max(0, (cell_h - img.height) // 2)
    canvas.paste(img, (offset_x, offset_y))


def draw_centered_titles(
    draw: ImageDraw.ImageDraw,
    titles: list[str],
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    y: int,
    margin: int,
    cell_w: int,
    gutter: int,
) -> None:
    for idx, title in enumerate(titles):
        x = margin + idx * (cell_w + gutter)
        box = draw.textbbox((0, 0), title, font=font)
        text_w = box[2] - box[0]
        text_x = x + max(0, (cell_w - text_w) // 2)
        draw.text((text_x, y), title, fill="black", font=font)


METRIC_SPECS = [
    ("instability_mean", "Instability (1-ARI)", "lower better"),
    ("instability_se", "Instability SE", "lower better"),
    ("null_corrected_score", "ARI", "higher better"),
    ("homogeneity", "Homogeneity", "higher better"),
    ("silhouette", "Silhouette", "higher better"),
    ("min_cluster_size_fraction", "Min Cluster Fraction", "higher better"),
    ("connected_component_count", "Connected Components", "lower better"),
    ("within_1se_best", "Within 1-SE", "1 is candidate"),
]


def plot_curves(axs: list[plt.Axes], final_selection: dict[str, object], hemi: str) -> None:
    suffix = "final"
    for ax, (metric_key, metric_label, direction_label) in zip(axs, METRIC_SPECS, strict=True):
        for smooth_name in SMOOTHS:
            hemi_node = final_selection["per_smooth"][smooth_name]["hemis"][hemi]
            rows = hemi_node["k_metrics"]
            ks = [row["k"] for row in rows]
            ys = [np.nan if row[metric_key] is None else row[metric_key] for row in rows]
            best_k = int(hemi_node["k_final"])
            ax.plot(
                ks,
                ys,
                marker="o",
                linewidth=2.0,
                color=SMOOTH_COLORS[smooth_name],
                label=f"{SMOOTH_LABEL[smooth_name]} (best K={best_k})",
            )
            kf = int(hemi_node["k_final"])
            yf = next(row[metric_key] for row in rows if int(row["k"]) == kf)
            if yf is not None and np.isfinite(float(yf)):
                yf_val = float(yf)
                ax.scatter([kf], [yf_val], color=SMOOTH_COLORS[smooth_name], s=72, zorder=4, edgecolors="black", linewidths=0.9)
                ax.annotate(
                    f"{kf}",
                    (kf, yf_val),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                    fontsize=PLOT_SPEC["tick_font"],
                    color=SMOOTH_COLORS[smooth_name],
                    fontweight="bold",
                )
        ax.set_title(f"{hemi} {metric_label} ({direction_label})", fontsize=PLOT_SPEC["metric_font"])
        ax.set_xlabel(f"K ({suffix})", fontsize=PLOT_SPEC["tick_font"])
        ax.set_xticks(EVAL_K)
        ax.tick_params(axis="both", labelsize=PLOT_SPEC["tick_font"])
        ax.grid(alpha=0.25, linewidth=0.5)
    axs[0].legend(frameon=False, fontsize=PLOT_SPEC["tick_font"], loc="best")


def plot_soft_diagnostics(ax: plt.Axes, final_selection: dict[str, object], hemi: str) -> None:
    color_specs = [
        ("2mm", "mean_probabilities", "2mm mean"),
        ("2mm", "mean_regularized_probabilities", "2mm reg"),
        ("4mm", "mean_probabilities", "4mm mean"),
        ("4mm", "mean_regularized_probabilities", "4mm reg"),
    ]
    networks = list(final_selection["per_smooth"]["2mm"]["hemis"][hemi]["soft_outputs"]["networks"])
    score = np.zeros(len(networks), dtype=np.float32)
    for smooth_name in SMOOTHS:
        soft = final_selection["per_smooth"][smooth_name]["hemis"][hemi]["soft_outputs"]
        score = np.maximum(score, np.asarray(soft["mean_probabilities"], dtype=np.float32))
        score = np.maximum(score, np.asarray(soft["mean_regularized_probabilities"], dtype=np.float32))
    keep = np.argsort(score)[::-1][: min(6, score.size)]
    x = np.arange(len(keep))
    width = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float32) * width
    for offset, (smooth_name, key, label) in zip(offsets, color_specs, strict=True):
        soft = final_selection["per_smooth"][smooth_name]["hemis"][hemi]["soft_outputs"]
        values = np.asarray(soft[key], dtype=np.float32)
        ax.bar(x + offset, values[keep], width=width, label=label, color=SOFT_DIAG_COLORS[(smooth_name, key)])
    ax.set_xticks(x)
    ax.set_xticklabels(
        [NETWORK_SHORT.get(networks[idx], networks[idx]) for idx in keep],
        rotation=18,
        ha="right",
        fontsize=PLOT_SPEC["tick_font"],
    )
    ax.set_ylim(0, max(0.16, float(np.max(score[keep]) * 1.22)))
    labels_text = ", ".join(
        f"{SMOOTH_LABEL[smooth]} K={final_selection['per_smooth'][smooth]['hemis'][hemi]['k_final']}"
        for smooth in SMOOTHS
    )
    ax.set_title(f"{hemi} Soft Diagnostics\n{labels_text}", fontsize=PLOT_SPEC["metric_font"])
    ax.grid(alpha=0.25, linewidth=0.5, axis="y")
    ax.legend(frameon=False, fontsize=PLOT_SPEC["tick_font"], loc="best")


def create_curve_figure(root: Path, final_selection: dict[str, object]) -> Path:
    atlas = str(final_selection.get("atlas_display_name", final_selection["atlas_slug"]))
    branch_slug = str(final_selection["branch_slug"])
    fig, axs = plt.subplots(2, 8, figsize=(24, 8.8), constrained_layout=True)
    plot_curves(list(axs[:, :4].ravel()), final_selection, "L")
    plot_curves(list(axs[:, 4:].ravel()), final_selection, "R")
    fig.suptitle(f"sub-{final_selection['subject']} {atlas} {branch_slug} K-selection", fontsize=18)
    out_path = root / "k_selection_curves.png"
    fig.savefig(out_path, dpi=220, facecolor="white")
    plt.close(fig)
    return out_path


def heatmap_text_color(value: float, vmax: float, cmap_name: str = "magma") -> str:
    cmap = plt.get_cmap(cmap_name)
    norm_value = 0.0 if vmax <= 0 else float(np.clip(value / vmax, 0.0, 1.0))
    r, g, b, _ = cmap(norm_value)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if luminance >= 0.55 else "white"


def is_soft_branch(branch_slug: str) -> bool:
    return False


def is_wta_branch(branch_slug: str) -> bool:
    return False


def shorten_cluster_name(cluster_name: str) -> str:
    if "_" not in cluster_name:
        return cluster_name
    prefix, network = cluster_name.split("_", 1)
    return f"{prefix}_{NETWORK_SHORT.get(network, network)}"


def create_probability_figure(root: Path, final_selection: dict[str, object]) -> Path:
    branch_slug = str(final_selection["branch_slug"])
    fig = plt.figure(figsize=(22.5, 7.6))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.035], wspace=0.08)
    heatmaps = outer[0, 0].subgridspec(1, 2, wspace=0.55)
    axs = [fig.add_subplot(heatmaps[0, idx]) for idx in range(2)]
    cax = fig.add_subplot(outer[0, 1])
    last_image = None
    summaries: dict[str, dict[str, object]] = {}

    for ax, hemi in zip(axs, HEMIS, strict=True):
        if is_soft_branch(branch_slug):
            networks = list(final_selection["per_smooth"]["2mm"]["hemis"][hemi]["soft_outputs"]["networks"])
            block_rows = []
            row_labels = []
            for smooth_name in SMOOTHS:
                soft = final_selection["per_smooth"][smooth_name]["hemis"][hemi]["soft_outputs"]
                hemi_node = final_selection["per_smooth"][smooth_name]["hemis"][hemi]
                block_rows.extend(
                    [
                        np.asarray(soft["mean_probabilities"], dtype=np.float32),
                        np.asarray(soft["mean_regularized_probabilities"], dtype=np.float32),
                        np.asarray(soft["argmax_occupancy"], dtype=np.float32),
                    ]
                )
                row_labels.extend(
                    [
                        f"{SMOOTH_LABEL[smooth_name]} mean",
                        f"{SMOOTH_LABEL[smooth_name]} regularized",
                        f"{SMOOTH_LABEL[smooth_name]} argmax",
                    ]
                )
                block_rows.extend(np.asarray(hemi_node["probability_rows"], dtype=np.float32))
                row_labels.extend(
                    [
                        f"{SMOOTH_LABEL[smooth_name]} {shorten_cluster_name(str(row['cluster_name']))}"
                        for row in hemi_node["cluster_annotations"]
                    ]
                )
            probs = np.asarray(block_rows, dtype=np.float32)
            keep = np.arange(len(networks), dtype=np.int32)
        elif is_wta_branch(branch_slug):
            networks = list(final_selection["per_smooth"]["2mm"]["hemis"][hemi]["soft_outputs"]["networks"])
            block_rows = []
            row_labels = []
            for smooth_name in SMOOTHS:
                soft = final_selection["per_smooth"][smooth_name]["hemis"][hemi]["soft_outputs"]
                block_rows.extend(
                    [
                        np.asarray(soft["mean_grouped_fc"], dtype=np.float32),
                        np.asarray(soft["network_occupancy"], dtype=np.float32),
                    ]
                )
                row_labels.extend(
                    [
                        f"{SMOOTH_LABEL[smooth_name]} mean FC",
                        f"{SMOOTH_LABEL[smooth_name]} occupancy",
                    ]
                )
            probs = np.asarray(block_rows, dtype=np.float32)
            keep = np.arange(len(networks), dtype=np.int32)
        else:
            networks = list(final_selection["per_smooth"]["2mm"]["hemis"][hemi]["profile_networks"])
            block_rows = []
            row_labels = []
            for smooth_name in SMOOTHS:
                hemi_node = final_selection["per_smooth"][smooth_name]["hemis"][hemi]
                block_rows.extend(np.asarray(hemi_node["probability_rows"], dtype=np.float32))
                row_labels.extend(
                    [
                        f"{SMOOTH_LABEL[smooth_name]} {shorten_cluster_name(str(row['cluster_name']))}"
                        for row in hemi_node["cluster_annotations"]
                    ]
                )
            probs = np.asarray(block_rows, dtype=np.float32)
            keep = np.arange(len(networks), dtype=np.int32)
        probs_plot = probs[:, keep]
        networks_plot = [networks[idx] for idx in keep]
        vmax = max(0.12, float(np.nanmax(probs_plot)))
        image = ax.imshow(probs_plot, aspect="auto", cmap="magma", vmin=0.0, vmax=vmax)
        last_image = image
        title = f"{hemi} {'soft summaries' if is_soft_branch(branch_slug) else ('WTA confidence + occupancy' if is_wta_branch(branch_slug) else 'cluster profiles')}"
        ax.set_title(title, fontsize=PLOT_SPEC["metric_font"])
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=PLOT_SPEC["tick_font"])
        ax.tick_params(axis="y", pad=8)
        ax.set_xticks(range(len(networks_plot)))
        ax.set_xticklabels(
            [NETWORK_SHORT.get(net, net) for net in networks_plot],
            rotation=0,
            ha="center",
            fontsize=PLOT_SPEC["tick_font"],
        )
        for label in ax.get_yticklabels():
            label.set_horizontalalignment("right")
        for i in range(probs_plot.shape[0]):
            dominant_idx = int(np.argmax(probs[i, :]))
            for j in range(probs_plot.shape[1]):
                value = float(probs_plot[i, j])
                network_idx = int(keep[j])
                is_selected = network_idx == dominant_idx
                text_color = heatmap_text_color(value, vmax, "magma")
                if is_selected:
                    ax.add_patch(
                        Rectangle(
                            (j - 0.5, i - 0.5),
                            1.0,
                            1.0,
                            fill=False,
                            edgecolor=text_color,
                            linewidth=1.5,
                        )
                    )
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=PLOT_SPEC["tick_font"],
                    fontweight="bold" if is_selected else "normal",
                    color=text_color,
                )
        summaries[hemi] = {
            "networks": networks,
            "probabilities": probs.tolist(),
            "mode": "soft+subregions" if is_soft_branch(branch_slug) else ("wta" if is_wta_branch(branch_slug) else "cluster"),
            "display_networks": networks_plot,
            "display_probabilities": probs_plot.tolist(),
            "row_labels": row_labels,
        }

    if last_image is not None:
        cbar = fig.colorbar(last_image, cax=cax)
        cbar.set_label("Probability / Score", fontsize=12)
        cbar.ax.tick_params(labelsize=10)
    fig.subplots_adjust(left=0.18, right=0.94, bottom=0.18, top=0.86)
    out_path = root / "network_probability_heatmaps.png"
    fig.savefig(out_path, dpi=220, facecolor="white")
    plt.close(fig)
    (root / "overview_probability_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return out_path


def load_cluster_name_map(annotation_path: Path) -> dict[int, str]:
    payload = load_json(annotation_path)
    rows = payload["clusters"]
    return {int(row["cluster_id"]): str(row["cluster_name"]) for row in rows}


def render_cache_is_valid(output_png: Path, input_paths: list[Path]) -> bool:
    if not output_png.exists():
        return False
    try:
        out_mtime = output_png.stat().st_mtime
    except OSError:
        return False
    for path in input_paths:
        if not path.exists():
            return False
        try:
            if path.stat().st_mtime > out_mtime:
                return False
        except OSError:
            return False
    return True


def render_shortlist_panels(
    root: Path,
    final_selection: dict[str, object],
    *,
    reuse_existing: bool,
) -> dict[str, list[tuple[str, Path]]]:
    branch_slug = str(final_selection["branch_slug"])
    render_cfg = final_selection["render_config"]
    scene = Path(render_cfg["scene"])
    layout = str(render_cfg["layout"])
    views = [str(token) for token in render_cfg["views"]]
    subject = str(final_selection["subject"])
    shortlist_panels: dict[str, list[tuple[str, Path]]] = {}
    for smooth_name in SMOOTHS:
        smooth_panels: list[tuple[str, Path]] = []
        smooth_root = root / "_overview_shortlist" / smooth_name
        left_surface = Path(final_selection["per_smooth"][smooth_name]["final_assets"]["left_surface"])
        right_surface = Path(final_selection["per_smooth"][smooth_name]["final_assets"]["right_surface"])
        left_k_final = int(final_selection["per_smooth"][smooth_name]["hemis"]["L"]["k_final"])
        right_k_final = int(final_selection["per_smooth"][smooth_name]["hemis"]["R"]["k_final"])
        for k in RENDER_SHORTLIST:
            render_png = (
                smooth_root
                / f"k_{k}"
                / "renders"
                / f"sub-{subject}_wb_{branch_slug.replace('-', '_')}_{smooth_name}_k{k}_biglegend.png"
            )
            left_label_path = root / "clustering" / smooth_name / "hemi_L" / f"k_{k}" / "cluster_labels.npy"
            right_label_path = root / "clustering" / smooth_name / "hemi_R" / f"k_{k}" / "cluster_labels.npy"
            left_annot_path = root / "clustering" / smooth_name / "hemi_L" / f"k_{k}" / "cluster_annotation.json"
            right_annot_path = root / "clustering" / smooth_name / "hemi_R" / f"k_{k}" / "cluster_annotation.json"
            cache_inputs = [
                left_label_path,
                right_label_path,
                left_annot_path,
                right_annot_path,
                left_surface,
                right_surface,
            ]
            if not (reuse_existing and render_cache_is_valid(render_png, cache_inputs)):
                if not left_label_path.exists() or not right_label_path.exists():
                    continue
                left_labels = np.load(left_label_path).astype(np.int32)
                right_labels = np.load(right_label_path).astype(np.int32)
                left_name_map = load_cluster_name_map(left_annot_path)
                right_name_map = load_cluster_name_map(right_annot_path)
                assets = save_combined_label_assets(
                    subject=subject,
                    left_labels=left_labels,
                    right_labels=right_labels,
                    output_dir=smooth_root / f"k_{k}" / "assets",
                    left_surface=left_surface,
                    right_surface=right_surface,
                    left_key_to_name=left_name_map,
                    right_key_to_name=right_name_map,
                    stem=f"hipp_network_cluster_{branch_slug.replace('-', '_')}_{smooth_name}_k{k}",
                )
                render = render_locked_grid_png(
                    subject=subject,
                    scene=scene,
                    views=views,
                    layout=layout,
                    outdir=smooth_root / f"k_{k}" / "renders",
                    name=f"{branch_slug.replace('-', '_')}_{smooth_name}_k{k}",
                    title=f"sub-{subject} {branch_slug} {smooth_name} K={k} | Lbest={left_k_final} Rbest={right_k_final}",
                    left_labels=Path(assets["left_label"]),
                    right_labels=Path(assets["right_label"]),
                    legend_group="network",
                )
                render_png = Path(render["biglegend_png"])
            if not render_png.exists():
                continue
            title = f"{SMOOTH_LABEL[smooth_name]} K={k}"
            if k == left_k_final or k == right_k_final:
                title = f"{title} (best)"
            smooth_panels.append((title, render_png))
        final_title = f"{SMOOTH_LABEL[smooth_name]} final (L={left_k_final}, R={right_k_final})"
        smooth_panels.append((final_title, Path(final_selection["per_smooth"][smooth_name]["final_png"])))
        shortlist_panels[smooth_name] = smooth_panels
    return shortlist_panels


def create_overview(root: Path, final_selection: dict[str, object], *, reuse_existing_shortlist: bool) -> Path:
    out_path = root / "hipp_functional_parcellation_network_overview.png"
    subject = str(final_selection["subject"])
    branch_slug = str(final_selection["branch_slug"])
    atlas_display = str(final_selection.get("atlas_display_name", final_selection.get("atlas_slug", "")))

    canvas_w = int(PLOT_SPEC["canvas_w"])
    margin = int(PLOT_SPEC["margin"])
    gutter = int(PLOT_SPEC["gutter"])
    section_gap = int(PLOT_SPEC["section_gap"])
    title_h = int(PLOT_SPEC["title_h"])
    panel_title_h = int(PLOT_SPEC["panel_title_h"])

    row1_path = root / "k_selection_curves.png"
    row1 = fit_image(row1_path, canvas_w - 2 * margin) if row1_path.exists() else None
    row2 = fit_image(root / "network_probability_heatmaps.png", canvas_w - 2 * margin)
    shortlist = render_shortlist_panels(root, final_selection, reuse_existing=reuse_existing_shortlist)
    render_rows: list[tuple[list[str], list[Image.Image], int]] = []
    max_cols = max(len(shortlist[s]) for s in SMOOTHS)
    cell_w = (canvas_w - 2 * margin - gutter * (max_cols - 1)) // max_cols
    for smooth_name in SMOOTHS:
        titles = [title for title, _path in shortlist[smooth_name]]
        imgs = [fit_image(path, cell_w, max_height=920) for _title, path in shortlist[smooth_name]]
        render_rows.append((titles, imgs, cell_w))

    render_block_h = 0
    for _titles, imgs, _cell_w in render_rows:
        render_block_h += panel_title_h + max(img.height for img in imgs)
    render_block_h += section_gap * max(0, len(render_rows) - 1)
    canvas_h = (
        margin
        + title_h
        + (row1.height + section_gap if row1 else 0)
        + row2.height
        + section_gap
        + render_block_h
        + margin
    )
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(int(PLOT_SPEC["title_font"]))
    panel_title_font = load_font(int(PLOT_SPEC["panel_title_font"]))
    title = f"sub-{subject} | {atlas_display} | {branch_slug} overview"
    title_box = draw.textbbox((0, 0), title, font=title_font)
    title_w = title_box[2] - title_box[0]
    draw.text(((canvas_w - title_w) // 2, margin // 2), title, fill="black", font=title_font)

    y = margin + title_h
    if row1:
        canvas.paste(row1, (margin, y))
        y += row1.height + section_gap
    canvas.paste(row2, (margin, y))
    y += row2.height + section_gap
    for titles, imgs, row_cell_w in render_rows:
        draw_centered_titles(draw, titles, panel_title_font, y, margin, row_cell_w, gutter)
        y += panel_title_h
        row_h = max(img.height for img in imgs)
        for idx, img in enumerate(imgs):
            x = margin + idx * (row_cell_w + gutter)
            paste_center(canvas, img, x, y, row_cell_w, row_h)
        y += row_h + section_gap
    canvas.save(out_path)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize network-first hippocampal functional parcellation outputs")
    parser.add_argument("--root", required=True, help="Branch output root, e.g. outputs/hipp_functional_parcellation_network/network-gradient/lynch2024/sub-100610")
    parser.add_argument(
        "--rebuild-shortlist",
        action="store_true",
        help="Force regenerate shortlist K-panel renders instead of reusing existing _overview_shortlist images.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    final_selection = load_json(root / "final_selection_summary.json")
    copied = copy_key_images(root, final_selection)
    branch_slug = final_selection.get("branch_slug", "")
    curves = None
    if not is_wta_branch(str(branch_slug)):
        curves = create_curve_figure(root, final_selection)
    probs = create_probability_figure(root, final_selection)
    overview = create_overview(root, final_selection, reuse_existing_shortlist=not args.rebuild_shortlist)
    manifest = {
        "root": str(root),
        "branch_slug": str(final_selection["branch_slug"]),
        "k_selection_curves": str(curves) if curves else None,
        "network_probability_heatmaps": str(probs),
        "overview": str(overview),
        "copied_images": [str(path) for path in copied],
    }
    (root / "summary_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_summary_stage_manifest(
        root=root,
        params={
            "branch_slug": str(final_selection["branch_slug"]),
            "subject": str(final_selection["subject"]),
            "atlas_slug": str(final_selection["atlas_slug"]),
            "rebuild_shortlist": bool(args.rebuild_shortlist),
        },
        inputs=[root / "final_selection_summary.json"],
        outputs=([curves] if curves else []) + [probs, overview, root / "summary_manifest.json"],
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

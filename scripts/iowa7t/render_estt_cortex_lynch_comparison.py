#!/usr/bin/env python3
"""
Render 2-Row Master comparison figure of:
Row 1: Kong2022 Group Standard vs Individual (Areal MSHBM) [17-Network / 400 Parcels]
Row 2: Lynch2024 Group Standard Prior vs Individual (FASTANS PFM) [20-Network]
on native fsLR-32k surface with large, high-legibility fonts, 2-column network legends,
literature-standardized short network names (e.g. SCAN, CinguloOpercular), and clean
non-overlapping vertex similarity / Dice overlap metric cards.
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import tempfile
from importlib import import_module
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_WORKFLOW_DIR = REPO_ROOT / "scripts" / "core_workflow"
if str(CORE_WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_WORKFLOW_DIR))

from lib.network_colors import load_network_colors
from lib.workbench import resolve_wb_command, run_workbench

DEFAULT_SCENE = REPO_ROOT / "config" / "manual_wb_scenes" / "cortex_manual.scene"
DEFAULT_KONG_STD_DLABEL = (
    REPO_ROOT
    / "external"
    / "Schaefer2018_Kong2022"
    / "Schaefer2018_400Parcels_Kong2022_17Networks_order.dlabel.nii"
)
RENDER_SCRIPT = REPO_ROOT / "scripts" / "core_workflow" / "lib" / "render_cortex_pfm_scene.py"
DEFAULT_ESTT_BASE = Path("/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_esTT/DataOrg_PFM_36P")
DEFAULT_FASTANS_RESOURCES = REPO_ROOT / "external" / "FASTANS" / "resources"
LYNCH_LABEL_LIST = DEFAULT_FASTANS_RESOURCES / "PFM" / "priors" / "Lynch2024" / "Lynch2024_LabelList.txt"
LYNCH_PRIORS_PICKLE = DEFAULT_FASTANS_RESOURCES / "PFM" / "priors" / "Lynch2024" / "Lynch2024_priors.pickle"
DEFAULT_LYNCH_STD_DLABEL = DEFAULT_FASTANS_RESOURCES / "PFM" / "priors" / "Lynch2024" / "Lynch2024_Standard_Prior.dlabel.nii"

# Standardized short/literature display names for Lynch 2024 networks
LYNCH_SHORT_NAMES: dict[str, str] = {
    "Default_Parietal": "Def_Parietal",
    "Default_Anterolateral": "Def_AntLat",
    "Default_Dorsolateral": "Def_DorsLat",
    "Default_Retrosplenial": "Def_Retro",
    "Visual_Lateral": "Vis_Lateral",
    "Visual_Dorsal/VentralStream": "Vis_DV_Stream",
    "Visual_V5": "Vis_V5",
    "Visual_V1": "Vis_V1",
    "Frontoparietal": "Frontoparietal",
    "DorsalAttention": "DorsalAttn",
    "Premotor/DorsalAttentionII": "PM/DorsAttnII",
    "Language": "Language",
    "Salience": "Salience",
    "CinguloOpercular/Action-mode": "CinguloOpercular",
    "MedialParietal": "MedialParietal",
    "Somatomotor_Hand": "SomMot_Hand",
    "Somatomotor_Face": "SomMot_Face",
    "Somatomotor_Foot": "SomMot_Foot",
    "Auditory": "Auditory",
    "SomatoCognitiveAction": "SCAN",
    "Noise": "Noise",
}


def load_font(size: int) -> ImageFont.ImageFont:
    for font_name in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
        "Arial.ttf",
        "Helvetica.ttc",
    ]:
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def load_lynch_label_colors(label_list_path: Path = LYNCH_LABEL_LIST) -> list[tuple[str, tuple[int, int, int, int]]]:
    lines = [line.strip() for line in label_list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    items: list[tuple[str, tuple[int, int, int, int]]] = []
    for i in range(0, len(lines), 2):
        if i + 1 >= len(lines):
            break
        raw_name = lines[i]
        short_name = LYNCH_SHORT_NAMES.get(raw_name, raw_name)
        parts = [int(p) for p in lines[i + 1].split()]
        rgba = (parts[1], parts[2], parts[3], parts[4])
        items.append((short_name, rgba))
    return items


def ensure_lynch_standard_prior_dlabel(
    *,
    resources_root: Path = DEFAULT_FASTANS_RESOURCES,
    priors_pickle: Path = LYNCH_PRIORS_PICKLE,
    label_list: Path = LYNCH_LABEL_LIST,
    target_dlabel: Path = DEFAULT_LYNCH_STD_DLABEL,
) -> Path:
    if target_dlabel.exists():
        return target_dlabel

    print(f"Generating Lynch2024 Standard Prior dlabel from {priors_pickle}...")
    cortex_template = resources_root / "templates" / "CORTEX.32k_fs_LR.dscalar.nii"
    tpl_img = nib.load(str(cortex_template))
    tpl_mask = tpl_img.get_fdata()[0]

    with open(priors_pickle, "rb") as f:
        priors = pickle.load(f)

    # Argmax over 20 network priors
    std_labels = (np.argmax(priors, axis=0) + 1).astype(np.float64)
    std_labels[tpl_mask == 0] = 0

    target_dlabel.parent.mkdir(parents=True, exist_ok=True)
    temp_txt = target_dlabel.with_suffix(".txt")
    temp_dscalar = target_dlabel.with_suffix(".dscalar.nii")

    np.savetxt(str(temp_txt), std_labels.reshape(-1), fmt="%.10g")
    run_workbench([
        "-cifti-convert",
        "-from-text",
        str(temp_txt),
        str(cortex_template),
        str(temp_dscalar),
    ])
    run_workbench([
        "-cifti-label-import",
        str(temp_dscalar),
        str(label_list),
        str(target_dlabel),
    ])
    return target_dlabel


def compute_kong_similarity(
    std_dlabel_path: Path,
    ind_dlabel_path: Path,
) -> dict[str, object]:
    std_img = nib.load(str(std_dlabel_path))
    ind_img = nib.load(str(ind_dlabel_path))
    std_data = std_img.get_fdata()[0].astype(int)
    ind_data = ind_img.get_fdata()[0].astype(int)
    std_table = std_img.header.get_axis(0).label[0]

    valid_mask = (std_data > 0) & (ind_data > 0)
    parcel_agree = float(np.sum((std_data == ind_data) & valid_mask) / np.sum(valid_mask) * 100)

    def extract_net17(name: str) -> str:
        parts = name.split("_")
        return parts[2] if len(parts) >= 3 else "Unknown"

    val2net17 = {val: extract_net17(name) for val, (name, _) in std_table.items() if val > 0}
    all_17nets = sorted(list(set(val2net17.values())))
    net2id = {name: i + 1 for i, name in enumerate(all_17nets)}

    std_17 = np.zeros_like(std_data)
    ind_17 = np.zeros_like(ind_data)
    for val, net_name in val2net17.items():
        std_17[std_data == val] = net2id[net_name]
        ind_17[ind_data == val] = net2id[net_name]

    net17_agree = float(np.sum((std_17 == ind_17) & valid_mask) / np.sum(valid_mask) * 100)

    per_net_dice: dict[str, float] = {}
    for name, nid in net2id.items():
        s = (std_17 == nid) & valid_mask
        i = (ind_17 == nid) & valid_mask
        inter = np.sum(s & i)
        tot = np.sum(s) + np.sum(i)
        per_net_dice[name] = float(2.0 * inter / tot) if tot > 0 else 0.0

    return {
        "n_valid_vertices": int(np.sum(valid_mask)),
        "parcel_agreement_percent": parcel_agree,
        "net17_agreement_percent": net17_agree,
        "mean_dice": float(np.mean(list(per_net_dice.values()))),
        "per_net_dice": per_net_dice,
    }


def compute_lynch_similarity(
    std_dlabel_path: Path,
    pfm_dlabel_path: Path,
    label_list_path: Path = LYNCH_LABEL_LIST,
) -> dict[str, object]:
    std_img = nib.load(str(std_dlabel_path))
    pfm_img = nib.load(str(pfm_dlabel_path))
    std_data = std_img.get_fdata()[0].astype(int)
    pfm_data = pfm_img.get_fdata()[0].astype(int)

    valid_mask = (std_data > 0) & (pfm_data > 0)
    n_valid = int(np.sum(valid_mask))
    agree = float(np.sum((std_data == pfm_data) & valid_mask) / n_valid * 100)

    lines = [line.strip() for line in label_list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    raw_names = [lines[i] for i in range(0, len(lines), 2) if lines[i] != "Noise"]

    per_net_dice: dict[str, float] = {}
    for i, raw_name in enumerate(raw_names, start=1):
        short_name = LYNCH_SHORT_NAMES.get(raw_name, raw_name)
        s_m = (std_data == i) & valid_mask
        p_m = (pfm_data == i) & valid_mask
        inter = np.sum(s_m & p_m)
        tot = np.sum(s_m) + np.sum(p_m)
        per_net_dice[short_name] = float(2.0 * inter / tot) if tot > 0 else 0.0

    return {
        "n_valid_vertices": n_valid,
        "net20_agreement_percent": agree,
        "mean_dice": float(np.mean(list(per_net_dice.values()))),
        "per_net_dice": per_net_dice,
    }


def render_4view_scene(
    *,
    source_scene: Path,
    lh_surf: Path,
    rh_surf: Path,
    lh_label: Path,
    rh_label: Path,
    lh_sulc: Path,
    rh_sulc: Path,
    sulc_dscalar: Path,
    dlabel: Path,
    out_png: Path,
    width: int = 1200,
    height: int = 900,
    renderer: str = "OSMesa",
) -> None:
    out_scene = out_png.with_suffix(".scene")
    cmd = [
        sys.executable,
        str(RENDER_SCRIPT),
        "--source-scene",
        str(source_scene),
        "--left-surface",
        str(lh_surf),
        "--right-surface",
        str(rh_surf),
        "--left-label",
        str(lh_label),
        "--right-label",
        str(rh_label),
        "--left-sulc",
        str(lh_sulc),
        "--right-sulc",
        str(rh_sulc),
        "--sulc-dscalar",
        str(sulc_dscalar),
        "--dlabel",
        str(dlabel),
        "--out-scene",
        str(out_scene),
        "--out-png",
        str(out_png),
        "--scene-index",
        "1",
        "--renderer",
        renderer,
        "--width",
        str(width),
        "--height",
        str(height),
    ]
    subprocess.run(cmd, check=True)


def draw_2col_kong_legend(
    draw: ImageDraw.ImageDraw,
    x_start: int,
    y_start: int,
    panel_w: int,
    panel_h: int,
    network_colors: dict[str, tuple[int, int, int, int]],
) -> None:
    title_font = load_font(28)
    subtitle_font = load_font(18)
    item_font = load_font(20)

    draw.text((x_start, y_start), "Kong2022 Network Legend", fill=(255, 255, 255), font=title_font)
    draw.text((x_start, y_start + 34), "17 Functional Networks (Schaefer 400)", fill=(200, 200, 200), font=subtitle_font)

    families_17 = (
        "DefaultA",
        "DefaultB",
        "DefaultC",
        "ContA",
        "ContB",
        "ContC",
        "SalVenAttnA",
        "SalVenAttnB",
        "DorsAttnA",
        "DorsAttnB",
        "SomMotA",
        "SomMotB",
        "VisualA",
        "VisualB",
        "VisualC",
        "Aud",
        "Language",
    )

    top_offset = y_start + 70
    col_w = panel_w // 2
    n_per_col = 9
    row_h = (panel_h - 75) / n_per_col

    for idx, name in enumerate(families_17):
        col_idx = 0 if idx < n_per_col else 1
        row_idx = idx if col_idx == 0 else idx - n_per_col

        x_col = x_start + col_idx * col_w
        y_item = top_offset + row_idx * row_h
        rgba = network_colors.get(name, (158, 158, 158, 255))

        swatch_w = 24
        swatch_h = 24
        x_swatch = x_col
        y_swatch = y_item + (row_h - swatch_h) / 2

        draw.rectangle(
            [x_swatch, y_swatch, x_swatch + swatch_w, y_swatch + swatch_h],
            fill=rgba,
            outline=(255, 255, 255),
            width=2,
        )

        bbox = draw.textbbox((0, 0), name, font=item_font)
        text_h = bbox[3] - bbox[1]
        y_text = y_item + (row_h - text_h) / 2 - bbox[1]

        draw.text(
            (x_swatch + swatch_w + 10, y_text),
            name,
            fill=(255, 255, 255),
            font=item_font,
        )


def draw_2col_lynch_legend(
    draw: ImageDraw.ImageDraw,
    x_start: int,
    y_start: int,
    panel_w: int,
    panel_h: int,
    lynch_items: list[tuple[str, tuple[int, int, int, int]]],
) -> None:
    title_font = load_font(28)
    subtitle_font = load_font(18)
    item_font = load_font(19)

    draw.text((x_start, y_start), "Lynch2024 Network Legend", fill=(255, 255, 255), font=title_font)
    draw.text((x_start, y_start + 34), "20 Functional Networks (FASTANS PFM)", fill=(200, 200, 200), font=subtitle_font)

    display_items = [item for item in lynch_items if item[0] != "Noise"]

    top_offset = y_start + 70
    col_w = panel_w // 2
    n_per_col = 10
    row_h = (panel_h - 75) / n_per_col

    for idx, (name, rgba) in enumerate(display_items):
        col_idx = 0 if idx < n_per_col else 1
        row_idx = idx if col_idx == 0 else idx - n_per_col

        x_col = x_start + col_idx * col_w
        y_item = top_offset + row_idx * row_h

        swatch_w = 22
        swatch_h = 22
        x_swatch = x_col
        y_swatch = y_item + (row_h - swatch_h) / 2

        draw.rectangle(
            [x_swatch, y_swatch, x_swatch + swatch_w, y_swatch + swatch_h],
            fill=rgba,
            outline=(255, 255, 255),
            width=2,
        )

        bbox = draw.textbbox((0, 0), name, font=item_font)
        text_h = bbox[3] - bbox[1]
        y_text = y_item + (row_h - text_h) / 2 - bbox[1]

        draw.text(
            (x_swatch + swatch_w + 10, y_text),
            name,
            fill=(255, 255, 255),
            font=item_font,
        )


def draw_similarity_card(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    line1: tuple[str, str, tuple[int, int, int]],
    line2: tuple[str, str, tuple[int, int, int]],
    dice_dict: dict[str, float],
    line3: tuple[str, str, tuple[int, int, int]] | None = None,
) -> None:
    draw.rounded_rectangle(
        [x, y, x + w, y + h],
        radius=10,
        fill=(20, 24, 34),
        outline=(75, 95, 130),
        width=2,
    )

    header_font = load_font(23)
    title_font = load_font(18)
    stat_font = load_font(18)
    val_font = load_font(20)
    dice_net_font = load_font(16)
    dice_val_font = load_font(18)

    draw.text((x + 20, y + 14), title, fill=(255, 220, 100), font=header_font)

    y_cur = y + 48
    x_val = x + 340  # Ample space to prevent any overlap

    draw.text((x + 24, y_cur), line1[0], fill=(230, 230, 230), font=stat_font)
    draw.text((x_val, y_cur - 2), line1[1], fill=line1[2], font=val_font)
    y_cur += 28

    draw.text((x + 24, y_cur), line2[0], fill=(230, 230, 230), font=stat_font)
    draw.text((x_val, y_cur - 2), line2[1], fill=line2[2], font=val_font)
    y_cur += 28

    if line3:
        draw.text((x + 24, y_cur), line3[0], fill=(230, 230, 230), font=stat_font)
        draw.text((x_val, y_cur - 2), line3[1], fill=line3[2], font=val_font)
        y_cur += 32
    else:
        y_cur += 8

    draw.text((x + 20, y_cur), "Top Network Dice Breakdown:", fill=(210, 210, 210), font=title_font)
    y_cur += 26

    sorted_dice = sorted(dice_dict.items(), key=lambda it: it[1], reverse=True)
    col_w = w // 2 - 10

    for r in range(min(4, len(sorted_dice) // 2)):
        k1, d1 = sorted_dice[r]
        k2, d2 = sorted_dice[r + 4]

        # Col 1: name at x+24, value at x+185
        x_c1 = x + 24
        draw.text((x_c1, y_cur + r * 22), f"{k1}:", fill=(190, 190, 190), font=dice_net_font)
        draw.text((x_c1 + 160, y_cur + r * 22), f"{d1:.3f}", fill=(255, 255, 255), font=dice_val_font)

        # Col 2: name at x+300, value at x+465
        x_c2 = x + col_w + 10
        draw.text((x_c2, y_cur + r * 22), f"{k2}:", fill=(190, 190, 190), font=dice_net_font)
        draw.text((x_c2 + 160, y_cur + r * 22), f"{d2:.3f}", fill=(255, 255, 255), font=dice_val_font)


def compose_two_row_master_figure(
    *,
    subject: str,
    kong_std_png: Path,
    kong_ind_png: Path,
    lynch_std_png: Path,
    lynch_ind_png: Path,
    out_png_path: Path,
    network_colors: dict[str, tuple[int, int, int, int]],
    lynch_items: list[tuple[str, tuple[int, int, int, int]]],
    kong_metrics: dict[str, object],
    lynch_metrics: dict[str, object],
) -> None:
    k_std_img = Image.open(kong_std_png).convert("RGB")
    k_ind_img = Image.open(kong_ind_png).convert("RGB")
    l_std_img = Image.open(lynch_std_png).convert("RGB")
    l_ind_img = Image.open(lynch_ind_png).convert("RGB")

    tile_w, tile_h = k_std_img.size  # 1200 x 900
    margin = 40
    header_h = 125
    row_gap = 50
    section_h = 45
    legend_w = 580

    canvas_w = margin + tile_w + margin + tile_w + margin + legend_w + margin
    canvas_h = header_h + (section_h + tile_h + row_gap) * 2 + margin

    canvas = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    main_title_font = load_font(38)
    sub_title_font = load_font(22)
    row_title_font = load_font(26)

    agree_k17 = float(kong_metrics["net17_agreement_percent"])
    dice_k17 = float(kong_metrics["mean_dice"])
    agree_l20 = float(lynch_metrics["net20_agreement_percent"])
    dice_l20 = float(lynch_metrics["mean_dice"])

    draw.text(
        (margin, 20),
        f"sub-{subject} Cortical Functional Parcellation Dual Comparison: Standard vs Individual",
        fill=(255, 255, 255),
        font=main_title_font,
    )
    draw.text(
        (margin, 72),
        f"Dataset: esTT 36P | Native fsLR-32k Surface | Kong 17-Net Agreement: {agree_k17:.1f}% (Mean Dice: {dice_k17:.3f}) | Lynch 20-Net Agreement: {agree_l20:.1f}% (Mean Dice: {dice_l20:.3f})",
        fill=(190, 190, 190),
        font=sub_title_font,
    )

    # ─────────────────────────────────────────────────────────────
    # Row 1: Kong2022
    # ─────────────────────────────────────────────────────────────
    y_r1_header = header_h
    y_r1_content = y_r1_header + section_h

    x_std = margin
    x_ind = margin + tile_w + margin
    x_legend = x_ind + tile_w + margin

    draw.text((x_std, y_r1_header + 4), "Row 1: Kong2022 Group Standard Parcellation (17-Net)", fill=(255, 220, 100), font=row_title_font)
    draw.text((x_ind, y_r1_header + 4), f"Kong2022 Individual MSHBM (Agreement: {agree_k17:.1f}%)", fill=(100, 220, 255), font=row_title_font)

    canvas.paste(k_std_img, (x_std, y_r1_content))
    canvas.paste(k_ind_img, (x_ind, y_r1_content))

    legend_h = 490
    draw_2col_kong_legend(draw, x_legend, y_r1_content + 4, legend_w, legend_h, network_colors)
    draw_similarity_card(
        draw,
        x=x_legend,
        y=y_r1_content + legend_h + 15,
        w=legend_w,
        h=tile_h - legend_h - 15,
        title="Kong2022 Similarity Metrics",
        line1=("• 17-Net Agreement:", f"{agree_k17:.2f}%", (100, 220, 255)),
        line2=("• Mean 17-Net Dice:", f"{dice_k17:.3f}", (130, 255, 140)),
        line3=("• 400-Parcel Agreement:", f"{float(kong_metrics['parcel_agreement_percent']):.2f}%", (255, 200, 100)),
        dice_dict=kong_metrics["per_net_dice"],
    )

    # ─────────────────────────────────────────────────────────────
    # Row 2: Lynch2024
    # ─────────────────────────────────────────────────────────────
    y_r2_header = y_r1_content + tile_h + row_gap
    y_r2_content = y_r2_header + section_h

    draw.text((x_std, y_r2_header + 4), "Row 2: Lynch2024 Group Standard Prior (20-Net)", fill=(255, 220, 100), font=row_title_font)
    draw.text((x_ind, y_r2_header + 4), f"Lynch2024 Individual PFM (Agreement: {agree_l20:.1f}%)", fill=(130, 255, 140), font=row_title_font)

    canvas.paste(l_std_img, (x_std, y_r2_content))
    canvas.paste(l_ind_img, (x_ind, y_r2_content))

    draw_2col_lynch_legend(draw, x_legend, y_r2_content + 4, legend_w, legend_h, lynch_items)
    draw_similarity_card(
        draw,
        x=x_legend,
        y=y_r2_content + legend_h + 15,
        w=legend_w,
        h=tile_h - legend_h - 15,
        title="Lynch2024 Similarity Metrics",
        line1=("• 20-Net Agreement:", f"{agree_l20:.2f}%", (130, 255, 140)),
        line2=("• Mean 20-Net Dice:", f"{dice_l20:.3f}", (100, 220, 255)),
        dice_dict=lynch_metrics["per_net_dice"],
    )

    out_png_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png_path)
    print(f"Successfully generated 2-Row Master comparison figure: {out_png_path}")


def ensure_lynch_pfm_computed(
    *,
    subject: str,
    estt_base: Path,
    resources_root: Path,
) -> Path:
    sid = str(subject).replace("sub-", "")
    lynch_dir = estt_base / "hipp_outputs" / "cortex_pfm" / f"sub-{sid}" / "lynch2024"
    final_dlabel = lynch_dir / "PFM_Lynch2024priors.dlabel.nii"

    if final_dlabel.exists():
        return final_dlabel

    print(f"Lynch PFM dlabel not found for sub-{sid}. Running FASTANS Lynch2024 PFM computation...")
    step_03_04 = import_module("03_04_run_lynch2024_fastans_pfm")
    wb_command = resolve_wb_command()

    step_03_04.run_fastans_lynch2024_subject(
        subject=sid,
        data_root=estt_base,
        output_root=estt_base / "hipp_outputs" / "cortex_pfm",
        resources_root=resources_root,
        dtseries_template="{data_root}/{sub_subject}/func/{sub_subject}_task-rest_run-concat.dtseries.nii",
        left_surface_template="{data_root}/{sub_subject}/anat/{sub_subject}_hemi-L_space-fsLR_den-32k_midthickness.surf.gii",
        right_surface_template="{data_root}/{sub_subject}/anat/{sub_subject}_hemi-R_space-fsLR_den-32k_midthickness.surf.gii",
        wb_command=wb_command,
    )

    if not final_dlabel.exists():
        raise FileNotFoundError(f"FASTANS run finished but {final_dlabel} was not produced.")
    return final_dlabel


def run_pipeline(
    *,
    subject: str,
    out_png: Path,
    estt_base: Path = DEFAULT_ESTT_BASE,
    kong_std_dlabel: Path = DEFAULT_KONG_STD_DLABEL,
    fastans_resources: Path = DEFAULT_FASTANS_RESOURCES,
    renderer: str = "OSMesa",
) -> None:
    subject = str(subject).replace("sub-", "")
    anat_dir = estt_base / f"sub-{subject}" / "anat"
    lh_surf = anat_dir / f"sub-{subject}_hemi-L_space-fsLR_den-32k_inflated.surf.gii"
    rh_surf = anat_dir / f"sub-{subject}_hemi-R_space-fsLR_den-32k_inflated.surf.gii"
    sulc_dscalar = anat_dir / f"sub-{subject}_space-fsLR_den-91k_sulc.dscalar.nii"

    # Kong individual
    kong_ind_dir = estt_base / "hipp_outputs" / "cortex_kong2022_areal_mshbm" / f"sub-{subject}"
    kong_ind_dlabel = kong_ind_dir / f"sub-{subject}_atlas-kong2022_areal_mshbm.dlabel.nii"
    kong_ind_lh = kong_ind_dir / "labels" / f"sub-{subject}_atlas-kong2022_areal_mshbm.L.label.gii"
    kong_ind_rh = kong_ind_dir / "labels" / f"sub-{subject}_atlas-kong2022_areal_mshbm.R.label.gii"

    # Lynch labels
    lynch_std_dlabel = ensure_lynch_standard_prior_dlabel(resources_root=fastans_resources)
    lynch_pfm_dlabel = ensure_lynch_pfm_computed(subject=subject, estt_base=estt_base, resources_root=fastans_resources)

    for p in [lh_surf, rh_surf, sulc_dscalar, DEFAULT_SCENE]:
        if not p.exists():
            raise FileNotFoundError(f"Required path does not exist: {p}")

    network_colors = load_network_colors()
    lynch_items = load_lynch_label_colors(fastans_resources / "PFM" / "priors" / "Lynch2024" / "Lynch2024_LabelList.txt")

    # Compute metrics
    kong_metrics = compute_kong_similarity(std_dlabel_path=kong_std_dlabel, ind_dlabel_path=kong_ind_dlabel)
    lynch_metrics = compute_lynch_similarity(std_dlabel_path=lynch_std_dlabel, pfm_dlabel_path=lynch_pfm_dlabel)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        lh_sulc = tmp / "lh_sulc.func.gii"
        rh_sulc = tmp / "rh_sulc.func.gii"
        run_workbench([
            "-cifti-separate",
            str(sulc_dscalar),
            "COLUMN",
            "-metric",
            "CORTEX_LEFT",
            str(lh_sulc),
            "-metric",
            "CORTEX_RIGHT",
            str(rh_sulc),
        ])

        # Separate Lynch Standard Prior
        l_std_lh = tmp / "lynch_std.L.label.gii"
        l_std_rh = tmp / "lynch_std.R.label.gii"
        run_workbench([
            "-cifti-separate",
            str(lynch_std_dlabel),
            "COLUMN",
            "-label",
            "CORTEX_LEFT",
            str(l_std_lh),
            "-label",
            "CORTEX_RIGHT",
            str(l_std_rh),
        ])
        lynch_std_png = tmp / f"sub-{subject}_lynch_std.png"
        render_4view_scene(
            source_scene=DEFAULT_SCENE,
            lh_surf=lh_surf,
            rh_surf=rh_surf,
            lh_label=l_std_lh,
            rh_label=l_std_rh,
            lh_sulc=lh_sulc,
            rh_sulc=rh_sulc,
            sulc_dscalar=sulc_dscalar,
            dlabel=lynch_std_dlabel,
            out_png=lynch_std_png,
            renderer=renderer,
        )

        # Separate Lynch PFM
        l_pfm_lh = tmp / "lynch_pfm.L.label.gii"
        l_pfm_rh = tmp / "lynch_pfm.R.label.gii"
        run_workbench([
            "-cifti-separate",
            str(lynch_pfm_dlabel),
            "COLUMN",
            "-label",
            "CORTEX_LEFT",
            str(l_pfm_lh),
            "-label",
            "CORTEX_RIGHT",
            str(l_pfm_rh),
        ])
        lynch_pfm_png = tmp / f"sub-{subject}_lynch_pfm.png"
        render_4view_scene(
            source_scene=DEFAULT_SCENE,
            lh_surf=lh_surf,
            rh_surf=rh_surf,
            lh_label=l_pfm_lh,
            rh_label=l_pfm_rh,
            lh_sulc=lh_sulc,
            rh_sulc=rh_sulc,
            sulc_dscalar=sulc_dscalar,
            dlabel=lynch_pfm_dlabel,
            out_png=lynch_pfm_png,
            renderer=renderer,
        )

        # Separate Kong Standard
        k_std_lh = tmp / "kong_std.L.label.gii"
        k_std_rh = tmp / "kong_std.R.label.gii"
        run_workbench([
            "-cifti-separate",
            str(kong_std_dlabel),
            "COLUMN",
            "-label",
            "CORTEX_LEFT",
            str(k_std_lh),
            "-label",
            "CORTEX_RIGHT",
            str(k_std_rh),
        ])
        kong_std_png = tmp / f"sub-{subject}_kong_std.png"
        render_4view_scene(
            source_scene=DEFAULT_SCENE,
            lh_surf=lh_surf,
            rh_surf=rh_surf,
            lh_label=k_std_lh,
            rh_label=k_std_rh,
            lh_sulc=lh_sulc,
            rh_sulc=rh_sulc,
            sulc_dscalar=sulc_dscalar,
            dlabel=kong_std_dlabel,
            out_png=kong_std_png,
            renderer=renderer,
        )

        # Render Kong Individual
        kong_ind_png = tmp / f"sub-{subject}_kong_ind.png"
        render_4view_scene(
            source_scene=DEFAULT_SCENE,
            lh_surf=lh_surf,
            rh_surf=rh_surf,
            lh_label=kong_ind_lh,
            rh_label=kong_ind_rh,
            lh_sulc=lh_sulc,
            rh_sulc=rh_sulc,
            sulc_dscalar=sulc_dscalar,
            dlabel=kong_ind_dlabel,
            out_png=kong_ind_png,
            renderer=renderer,
        )

        # Compose 2-Row Master figure
        compose_two_row_master_figure(
            subject=subject,
            kong_std_png=kong_std_png,
            kong_ind_png=kong_ind_png,
            lynch_std_png=lynch_std_png,
            lynch_ind_png=lynch_pfm_png,
            out_png_path=out_png,
            network_colors=network_colors,
            lynch_items=lynch_items,
            kong_metrics=kong_metrics,
            lynch_metrics=lynch_metrics,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render 2-Row Master comparison figure of Kong2022 and Lynch2024 Standard vs Individual on native space."
    )
    parser.add_argument("--subject", default="369", help="Subject ID (e.g. 369)")
    parser.add_argument(
        "--out-png",
        default=None,
        help="Output PNG path. Default: REPO_ROOT/sub-<id>_cortex_master_2row.png",
    )
    parser.add_argument("--renderer", default="OSMesa", help="Workbench renderer (OSMesa)")
    args = parser.parse_args()

    out_png = args.out_png
    if out_png is None:
        out_png = str(REPO_ROOT / f"sub-{args.subject}_cortex_master_2row.png")

    run_pipeline(
        subject=args.subject,
        out_png=Path(out_png).resolve(),
        renderer=args.renderer,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

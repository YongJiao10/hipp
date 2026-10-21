#!/usr/bin/env python3
"""
Render comprehensive comparison figures of cortical parcellations:
- Kong2022 Native 17-Network (Schaefer 400)
- Kong2022 Merged Canonical 7-Network
- Lynch2024 Native 20-Network (FASTANS PFM Standard Prior)
- Lynch2024 Merged Canonical 7-Network

on standard fsLR-32k surface in 4-view layout (LH/RH lateral, LH/RH medial)
with dedicated high-legibility network legends, constituent network annotations,
and a 2x2 Master Comparison Figure.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[3]
CORE_WORKFLOW_DIR = REPO_ROOT / "scripts" / "core_workflow"
if str(CORE_WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_WORKFLOW_DIR))

from lib.gifti_labels import make_named_label_gifti
from lib.kong2022_label_utils import (
    build_roi_registry,
    load_kong2022_dlabel,
    load_kong2022_merge_mapping,
    split_vertex_label_arrays,
)
from lib.network_colors import load_network_colors
from lib.workbench import resolve_wb_command, run_workbench

DEFAULT_SCENE = REPO_ROOT / "config" / "manual_wb_scenes" / "cortex_manual.scene"
RENDER_SCRIPT = REPO_ROOT / "scripts" / "core_workflow" / "lib" / "render_cortex_pfm_scene.py"
DEFAULT_KONG_DLABEL = (
    REPO_ROOT
    / "external"
    / "Schaefer2018_Kong2022"
    / "Schaefer2018_400Parcels_Kong2022_17Networks_order.dlabel.nii"
)
DEFAULT_FASTANS_RESOURCES = REPO_ROOT / "external" / "FASTANS" / "resources"
DEFAULT_LYNCH_DLABEL = (
    DEFAULT_FASTANS_RESOURCES
    / "PFM"
    / "priors"
    / "Lynch2024"
    / "Lynch2024_Standard_Prior.dlabel.nii"
)
DEFAULT_LYNCH_LABEL_LIST = (
    DEFAULT_FASTANS_RESOURCES
    / "PFM"
    / "priors"
    / "Lynch2024"
    / "Lynch2024_LabelList.txt"
)
DEFAULT_ANAT_DIR = REPO_ROOT / "data" / "hippunfold_input" / "sub-100610" / "anat"
DEFAULT_SULC_ASSETS_DIR = (
    REPO_ROOT
    / "outputs_hcp"
    / "inputs"
    / "kong2022"
    / "cortex_references"
    / "routea_areal_mshbm"
    / "sub-100610"
    / "assets"
)

CANONICAL_7_ORDER: tuple[str, ...] = (
    "Default",
    "Visual",
    "Somatomotor",
    "DorsalAttention",
    "VentralAttention",
    "Control",
    "Language",
)

KONG_CAN7_CONSTITUENTS: dict[str, str] = {
    "Default": "DefaultA, DefaultB, DefaultC",
    "Visual": "VisualA, VisualB, VisualC",
    "Somatomotor": "SomMotA, SomMotB, Aud",
    "DorsalAttention": "DorsAttnA, DorsAttnB",
    "VentralAttention": "SalVenAttnA, SalVenAttnB",
    "Control": "ContA, ContB, ContC",
    "Language": "Language",
}

LYNCH_CAN7_CONSTITUENTS: dict[str, str] = {
    "Default": "Def_Par, AntLat, DorsLat, Retro, MedPar",
    "Visual": "Vis_Lat, DV_Stream, Vis_V5, Vis_V1",
    "Somatomotor": "SomMot_Hand, Face, Foot, Aud, SCAN",
    "DorsalAttention": "DorsalAttn, PM/DorsAttnII",
    "VentralAttention": "Salience",
    "Control": "Frontoparietal, CinguloOpercular",
    "Language": "Language",
}

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


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf"]
        if bold
        else ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf", "Arial.ttf"]
    )
    for font_name in candidates:
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def load_lynch_label_colors(
    label_list_path: Path = DEFAULT_LYNCH_LABEL_LIST,
) -> list[tuple[str, tuple[int, int, int, int]]]:
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


def draw_legend_panel(
    *,
    title: str,
    subtitle: str,
    items: list[tuple[str, tuple[int, int, int, int]]],
    width: int = 360,
    height: int = 900,
    sub_notes: dict[str, str] | None = None,
) -> Image.Image:
    panel = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(panel)

    title_font = load_font(22, bold=True)
    subtitle_font = load_font(14)
    item_font = load_font(15, bold=True)
    subnote_font = load_font(11)

    margin_x = 18
    y_start = 22
    draw.text((margin_x, y_start), title, fill=(15, 23, 42), font=title_font)
    draw.text((margin_x, y_start + 28), subtitle, fill=(100, 116, 139), font=subtitle_font)

    content_y = y_start + 60
    avail_h = height - content_y - 20
    n_items = len(items)

    row_h = avail_h / max(n_items, 1)
    swatch_size = min(22, max(14, int(row_h * 0.55)))

    for i, (name, rgba) in enumerate(items):
        y_item = content_y + i * row_h
        y_swatch = y_item + 3

        # Swatch with subtle border
        draw.rectangle(
            [margin_x, y_swatch, margin_x + swatch_size, y_swatch + swatch_size],
            fill=rgba[:3],
            outline=(203, 213, 225),
            width=1,
        )

        x_text = margin_x + swatch_size + 12
        draw.text((x_text, y_item), name, fill=(15, 23, 42), font=item_font)

        if sub_notes and name in sub_notes:
            draw.text(
                (x_text, y_item + 20),
                f"({sub_notes[name]})",
                fill=(100, 116, 139),
                font=subnote_font,
            )

    return panel


def compose_single_figure(
    *,
    brain_img: Image.Image,
    legend_img: Image.Image,
) -> Image.Image:
    w = brain_img.width + legend_img.width
    h = max(brain_img.height, legend_img.height)
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    canvas.paste(brain_img, (0, 0))
    canvas.paste(legend_img, (brain_img.width, 0))
    return canvas


def run_pipeline(
    *,
    out_dir: Path,
    renderer: str = "OSMesa",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Surface and sulc assets
    lh_surf = DEFAULT_ANAT_DIR / "sub-100610_hemi-L_space-fsLR_den-32k_desc-MSMAll_inflated.surf.gii"
    rh_surf = DEFAULT_ANAT_DIR / "sub-100610_hemi-R_space-fsLR_den-32k_desc-MSMAll_inflated.surf.gii"
    sulc_dscalar = DEFAULT_ANAT_DIR / "sub-100610_space-fsLR_den-32k_desc-MSMAll_sulc.dscalar.nii"
    lh_sulc = DEFAULT_SULC_ASSETS_DIR / "sub-100610_hemi-L_desc-MSMAll_sulc.func.gii"
    rh_sulc = DEFAULT_SULC_ASSETS_DIR / "sub-100610_hemi-R_desc-MSMAll_sulc.func.gii"

    for p in [lh_surf, rh_surf, sulc_dscalar, lh_sulc, rh_sulc, DEFAULT_SCENE, DEFAULT_KONG_DLABEL, DEFAULT_LYNCH_DLABEL]:
        if not p.exists():
            raise FileNotFoundError(f"Required path does not exist: {p}")

    net_colors = load_network_colors()

    # ─────────────────────────────────────────────────────────────
    # Prepare Kong 17 and Canonical 7 labels
    # ─────────────────────────────────────────────────────────────
    print("Preparing Kong2022 label arrays...")
    k_img, k_label_map = load_kong2022_dlabel(DEFAULT_KONG_DLABEL)
    mapping_17_to_8 = load_kong2022_merge_mapping(taxonomy="canonical8")
    mapping_17_to_7 = load_kong2022_merge_mapping(taxonomy="canonical7")
    k_roi_reg = build_roi_registry(k_label_map, mapping_17_to_8, mapping_17_to_7)

    k_split = split_vertex_label_arrays(k_img)
    lh_k17 = np.zeros(32492, dtype=np.int32)
    rh_k17 = np.zeros(32492, dtype=np.int32)
    lh_k17[k_split["L"][0]] = k_split["L"][1]
    rh_k17[k_split["R"][0]] = k_split["R"][1]

    # Label items for Kong 17
    lh_k17_items = [(0, "Null", (158, 158, 158, 255))]
    rh_k17_items = [(0, "Null", (158, 158, 158, 255))]
    for row in k_roi_reg:
        roi_id = int(row["roi_id"])
        roi_name = str(row["roi_name"])
        family17 = str(row["family17"])
        rgba = net_colors.get(family17, (158, 158, 158, 255))
        if row["hemi"] == "L":
            lh_k17_items.append((roi_id, roi_name, rgba))
        else:
            rh_k17_items.append((roi_id, roi_name, rgba))

    # Map Kong to Canonical 7
    can7_to_id = {name: i + 1 for i, name in enumerate(CANONICAL_7_ORDER)}
    parcel_to_can7 = {int(row["roi_id"]): can7_to_id[str(row["canonical7"])] for row in k_roi_reg}

    lh_k7 = np.zeros(32492, dtype=np.int32)
    rh_k7 = np.zeros(32492, dtype=np.int32)
    for pid, c7id in parcel_to_can7.items():
        lh_k7[lh_k17 == pid] = c7id
        rh_k7[rh_k17 == pid] = c7id

    k7_items = [(0, "Null", (158, 158, 158, 255))]
    for i, name in enumerate(CANONICAL_7_ORDER, start=1):
        rgba = net_colors.get(name, (158, 158, 158, 255))
        k7_items.append((i, name, rgba))

    # ─────────────────────────────────────────────────────────────
    # Prepare Lynch 20 and Canonical 7 labels
    # ─────────────────────────────────────────────────────────────
    print("Preparing Lynch2024 label arrays...")
    l_img = nib.load(str(DEFAULT_LYNCH_DLABEL))
    l_data = l_img.get_fdata()[0].astype(int)
    l_geom = l_img.header.get_axis(1)
    l_split: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, slc, subaxis in l_geom.iter_structures():
        if "CORTEX_LEFT" in name:
            l_split["L"] = (np.asarray(subaxis.vertex, dtype=np.int32), l_data[slc])
        elif "CORTEX_RIGHT" in name:
            l_split["R"] = (np.asarray(subaxis.vertex, dtype=np.int32), l_data[slc])

    lh_l20 = np.zeros(32492, dtype=np.int32)
    rh_l20 = np.zeros(32492, dtype=np.int32)
    lh_l20[l_split["L"][0]] = l_split["L"][1]
    rh_l20[l_split["R"][0]] = l_split["R"][1]

    l20_raw_items = load_lynch_label_colors(DEFAULT_LYNCH_LABEL_LIST)
    l20_items = [(0, "Null", (158, 158, 158, 255))]
    for i, (s_name, rgba) in enumerate(l20_raw_items[:20], start=1):
        l20_items.append((i, s_name, rgba))

    # Map Lynch to Canonical 7
    lynch_to_can7 = {
        1: 1, 2: 1, 3: 1, 4: 1, 15: 1,
        5: 2, 6: 2, 7: 2, 8: 2,
        16: 3, 17: 3, 18: 3, 19: 3, 20: 3,
        10: 4, 11: 4,
        13: 5,
        9: 6, 14: 6,
        12: 7,
    }
    lh_l7 = np.zeros(32492, dtype=np.int32)
    rh_l7 = np.zeros(32492, dtype=np.int32)
    for l_id, c7_id in lynch_to_can7.items():
        lh_l7[lh_l20 == l_id] = c7_id
        rh_l7[rh_l20 == l_id] = c7_id

    # ─────────────────────────────────────────────────────────────
    # Render all 4 conditions
    # ─────────────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Helper to create label giftis and render scene
        def render_condition(
            lh_arr: np.ndarray,
            rh_arr: np.ndarray,
            lh_items_list: list,
            rh_items_list: list,
            cond_prefix: str,
        ) -> Image.Image:
            lh_gii = make_named_label_gifti(lh_arr, lh_items_list)
            rh_gii = make_named_label_gifti(rh_arr, rh_items_list)
            lh_gii_path = tmp / f"{cond_prefix}.L.label.gii"
            rh_gii_path = tmp / f"{cond_prefix}.R.label.gii"
            nib.save(lh_gii, str(lh_gii_path))
            nib.save(rh_gii, str(rh_gii_path))

            dlabel_path = tmp / f"{cond_prefix}.dlabel.nii"
            run_workbench([
                "-cifti-create-label",
                str(dlabel_path),
                "-left-label", str(lh_gii_path),
                "-right-label", str(rh_gii_path),
            ])

            out_png = tmp / f"{cond_prefix}_render.png"
            render_4view_scene(
                source_scene=DEFAULT_SCENE,
                lh_surf=lh_surf,
                rh_surf=rh_surf,
                lh_label=lh_gii_path,
                rh_label=rh_gii_path,
                lh_sulc=lh_sulc,
                rh_sulc=rh_sulc,
                sulc_dscalar=sulc_dscalar,
                dlabel=dlabel_path,
                out_png=out_png,
                renderer=renderer,
            )
            return Image.open(out_png).convert("RGB")

        print("Rendering Condition 1: Kong2022 Native 17-Network...")
        k17_brain = render_condition(lh_k17, rh_k17, lh_k17_items, rh_k17_items, "kong17")

        print("Rendering Condition 2: Kong2022 Canonical 7-Network...")
        k7_brain = render_condition(lh_k7, rh_k7, k7_items, k7_items, "kong7")

        print("Rendering Condition 3: Lynch2024 Native 20-Network...")
        l20_brain = render_condition(lh_l20, rh_l20, l20_items, l20_items, "lynch20")

        print("Rendering Condition 4: Lynch2024 Canonical 7-Network...")
        l7_brain = render_condition(lh_l7, rh_l7, k7_items, k7_items, "lynch7")

        # ─────────────────────────────────────────────────────────
        # Build Legends
        # ─────────────────────────────────────────────────────────
        legend_w = 360
        legend_h = 900

        # Kong 17 Legend
        from lib.kong2022_label_utils import FAMILIES_17_ORDER
        k17_legend_items = [(fam, net_colors[fam]) for fam in FAMILIES_17_ORDER]
        k17_legend = draw_legend_panel(
            title="Kong2022 Network Legend",
            subtitle="17 Functional Networks (Schaefer 400)",
            items=k17_legend_items,
            width=legend_w,
            height=legend_h,
        )

        # Kong Canonical 7 Legend
        k7_legend_items = [(name, net_colors[name]) for name in CANONICAL_7_ORDER]
        k7_legend = draw_legend_panel(
            title="Kong2022 Canonical 7-Net",
            subtitle="Merged Functional Networks",
            items=k7_legend_items,
            width=legend_w,
            height=legend_h,
            sub_notes=KONG_CAN7_CONSTITUENTS,
        )

        # Lynch 20 Legend
        l20_legend_items = l20_raw_items[:20]
        l20_legend = draw_legend_panel(
            title="Lynch2024 Network Legend",
            subtitle="20 Functional Networks (FASTANS PFM)",
            items=l20_legend_items,
            width=legend_w,
            height=legend_h,
        )

        # Lynch Canonical 7 Legend
        l7_legend_items = [(name, net_colors[name]) for name in CANONICAL_7_ORDER]
        l7_legend = draw_legend_panel(
            title="Lynch2024 Canonical 7-Net",
            subtitle="Merged Functional Networks",
            items=l7_legend_items,
            width=legend_w,
            height=legend_h,
            sub_notes=LYNCH_CAN7_CONSTITUENTS,
        )

        # ─────────────────────────────────────────────────────────
        # Save Standalone Figures
        # ─────────────────────────────────────────────────────────
        k17_legend_standalone = draw_legend_panel(
            title="Kong2022 Network Legend",
            subtitle="17 Functional Networks (Schaefer 400)",
            items=k17_legend_items,
            width=250,
            height=legend_h,
        )
        p_k17 = compose_single_figure(brain_img=k17_brain, legend_img=k17_legend_standalone)
        p_k7 = compose_single_figure(brain_img=k7_brain, legend_img=k7_legend)
        p_l20 = compose_single_figure(brain_img=l20_brain, legend_img=l20_legend)
        p_l7 = compose_single_figure(brain_img=l7_brain, legend_img=l7_legend)

        p_k17.save(out_dir / "kong2022_standard_lateral.png")
        p_k7.save(out_dir / "kong2022_standard_canonical7.png")
        p_l20.save(out_dir / "lynch2024_standard_20net.png")
        p_l7.save(out_dir / "lynch2024_standard_canonical7.png")
        print(f"Saved standalone panels to {out_dir}")

        # ─────────────────────────────────────────────────────────
        # Compose 2x2 Master Comparison Figure
        # ─────────────────────────────────────────────────────────
        print("Composing 2x2 Master Comparison Figure...")
        panel_w = 1560  # 1200 brain + 360 legend
        panel_h = 900
        margin = 40
        gap = 50
        header_h = 135
        subhead_h = 42

        canvas_w = margin * 2 + panel_w * 2 + gap
        canvas_h = header_h + (subhead_h + panel_h) * 2 + gap + margin

        canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        title_font = load_font(36, bold=True)
        subtitle_font = load_font(20)
        col_title_font = load_font(20, bold=True)

        # Main Header
        draw.text(
            (margin, 24),
            "Cortical Functional Parcellation: Kong2022 vs Lynch2024",
            fill=(15, 23, 42),
            font=title_font,
        )
        draw.text(
            (margin, 76),
            "Native Functional Networks (Kong 17-Net / Lynch 20-Net) & Merged Canonical 7-Networks | fsLR-32k Standard Surface",
            fill=(71, 85, 105),
            font=subtitle_font,
        )

        # Layout coordinates
        x_col1 = margin
        x_col2 = margin + panel_w + gap

        # ── Row 1: Kong2022 ──
        y_r1_head = header_h
        draw.text(
            (x_col1, y_r1_head + 4),
            "Kong2022 Native 17-Network (Schaefer 400)",
            fill=(146, 64, 14),
            font=col_title_font,
        )
        draw.text(
            (x_col2, y_r1_head + 4),
            "Kong2022 Merged Canonical 7-Network",
            fill=(14, 116, 144),
            font=col_title_font,
        )

        y_r1_content = y_r1_head + subhead_h
        canvas.paste(p_k17, (x_col1, y_r1_content))
        canvas.paste(p_k7, (x_col2, y_r1_content))

        # ── Row 2: Lynch2024 ──
        y_r2_head = y_r1_content + panel_h + gap
        draw.text(
            (x_col1, y_r2_head + 4),
            "Lynch2024 Native 20-Network (FASTANS PFM Standard Prior)",
            fill=(146, 64, 14),
            font=col_title_font,
        )
        draw.text(
            (x_col2, y_r2_head + 4),
            "Lynch2024 Merged Canonical 7-Network",
            fill=(14, 116, 144),
            font=col_title_font,
        )

        y_r2_content = y_r2_head + subhead_h
        canvas.paste(p_l20, (x_col1, y_r2_content))
        canvas.paste(p_l7, (x_col2, y_r2_content))

        master_out = out_dir / "kong_lynch_native_canonical7_comparison.png"
        canvas.save(master_out)
        print(f"Successfully generated Master Comparison Figure: {master_out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render standard Kong2022 vs Lynch2024 native and canonical 7-network cortical figures."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "scripts" / "visualize" / "kong2022_standard_lateral",
        help="Output directory for generated figures",
    )
    parser.add_argument(
        "--renderer",
        default="OSMesa",
        help="Workbench scene renderer (default: OSMesa)",
    )
    args = parser.parse_args()

    run_pipeline(out_dir=args.out_dir, renderer=args.renderer)


if __name__ == "__main__":
    main()

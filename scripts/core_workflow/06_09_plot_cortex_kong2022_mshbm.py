#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Set up path to core_workflow library
CORE_WORKFLOW_DIR = Path(__file__).resolve().parent
if str(CORE_WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_WORKFLOW_DIR))

from lib.workbench import run_workbench
from lib.render_cortex_pfm_scene import render_cortex_pfm_scene
from lib.gifti_labels import make_named_label_gifti
from lib.network_colors import load_network_colors
from lib.kong2022_label_utils import load_kong2022_bundle
from lib.path_templates import format_subject_path_template
from lib.workflow_utils import file_meta_unresolved as file_meta, utc_now_iso, write_json
from lib.traceability import StageSpec, run_stage

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENE = REPO_ROOT / "config" / "manual_wb_scenes" / "cortex_manual.scene"
DEFAULT_ROUTE_A_ROOT = REPO_ROOT / "outputs_hcp" / "inputs" / "hcp" / "kong2022" / "cortex_references" / "routea_areal_mshbm"
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "hippunfold_input"
DEFAULT_LEFT_SURFACE_TEMPLATE = "{data_root}/{sub_subject}/anat/{sub_subject}_hemi-L_space-fsLR_den-32k_desc-MSMAll_inflated.surf.gii"
DEFAULT_RIGHT_SURFACE_TEMPLATE = "{data_root}/{sub_subject}/anat/{sub_subject}_hemi-R_space-fsLR_den-32k_desc-MSMAll_inflated.surf.gii"
DEFAULT_SULC_TEMPLATE = "{data_root}/{sub_subject}/anat/{sub_subject}_space-fsLR_den-32k_desc-MSMAll_sulc.dscalar.nii"

VIEW_SPECS = [
    {"name": "lateral", "axis": None, "deg": 0.0},
    {"name": "medial", "axis": "y", "deg": -90.0},
    {"name": "dorsal", "axis": "x", "deg": 90.0},
    {"name": "ventral", "axis": "x", "deg": -90.0},
]


def load_font(size: int) -> ImageFont.ImageFont:
    for font_name in ["DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc"]:
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def add_legend_to_view(
    image_path: Path,
    subject: str,
    view_name: str,
    network_colors: dict,
) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    tile_w, tile_h = img.size

    panel_w = 250
    margin = 15
    title_h = 55

    # Uniform black background for subplots
    out = Image.new("RGB", (tile_w + panel_w, tile_h), (0, 0, 0))
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)

    x_start = tile_w + margin

    # Draw Title on black background (White text)
    title_font = load_font(26)
    subtitle_font = load_font(16)

    draw.text(
        (x_start, 10),
        "Network Legend",
        fill=(255, 255, 255),
        font=title_font,
    )
    draw.text(
        (x_start, 38),
        f"Kong2022 17-Net ({view_name.capitalize()})",
        fill=(200, 200, 200),
        font=subtitle_font,
    )

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

    avail_h = tile_h - title_h - margin * 2
    n_items = len(families_17)
    row_h = avail_h / n_items

    for i, name in enumerate(families_17):
        y_item = title_h + margin + i * row_h
        rgba = network_colors.get(name, (158, 158, 158, 255))

        swatch_w = min(28, max(18, int(row_h * 0.6)))
        swatch_h = swatch_w
        x_swatch = x_start
        y_swatch = y_item + (row_h - swatch_h) / 2

        # Draw color swatch with a thin white border to stand out on black background
        draw.rectangle(
            [x_swatch, y_swatch, x_swatch + swatch_w, y_swatch + swatch_h],
            fill=rgba,
            outline=(255, 255, 255),
            width=1,
        )

        font_size = max(16, min(27, int(row_h * 0.65)))
        item_font = load_font(font_size)
        bbox = draw.textbbox((0, 0), name, font=item_font)
        text_h = bbox[3] - bbox[1]
        y_text = y_item + (row_h - text_h) / 2 - bbox[1]

        # White text for items
        draw.text(
            (x_swatch + swatch_w + 14, y_text),
            name,
            fill=(255, 255, 255),
            font=item_font,
        )

    return out


def compose_overview(
    *,
    subject: str,
    view_pngs: list[tuple[str, Path]],
    out_png: Path,
) -> None:
    network_colors = load_network_colors()

    # Process each view to add its side legend
    processed_images = []
    for label, path in view_pngs:
        img_with_legend = add_legend_to_view(path, subject, label, network_colors)
        img_with_legend.save(path)
        processed_images.append((label, img_with_legend))

    tile_w_legend = processed_images[0][1].width
    tile_h_legend = processed_images[0][1].height

    margin = 28
    title_h = 86
    label_h = 36

    canvas_w = tile_w_legend * 2 + margin * 3
    canvas_h = title_h + (tile_h_legend + label_h) * 2 + margin * 3
    
    # Uniform black background for final canvas
    canvas = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(28)
    subtitle_font = load_font(18)
    label_font = load_font(18)

    # White text for titles/labels
    draw.text(
        (margin, 20),
        f"sub-{subject} Kong2022 MSHBM Cortex Overview",
        fill=(255, 255, 255),
        font=title_font,
    )
    draw.text(
        (margin, 56),
        "Views: lateral, medial, dorsal, ventral",
        fill=(200, 200, 200),
        font=subtitle_font,
    )

    for idx, (label, image) in enumerate(processed_images):
        row = idx // 2
        col = idx % 2
        x = margin + col * (tile_w_legend + margin)
        y = title_h + margin + row * (tile_h_legend + label_h)
        canvas.paste(image, (x, y))
        draw.text((x + 8, y + tile_h_legend + 8), label.capitalize(), fill=(255, 255, 255), font=label_font)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)


def generate_gifti_labels(
    subject: str,
    lh_txt: Path,
    rh_txt: Path,
    out_dir: Path,
    bundle: dict,
    network_colors: dict,
) -> tuple[Path, Path]:
    lh_arr = np.loadtxt(str(lh_txt), dtype=np.int32)
    rh_arr = np.loadtxt(str(rh_txt), dtype=np.int32)

    # Check shape
    if lh_arr.shape[0] != 32492 or rh_arr.shape[0] != 32492:
        raise ValueError(f"Expected surface labels of length 32492, got L={lh_arr.shape[0]}, R={rh_arr.shape[0]}")

    roi_registry = bundle["roi_registry"]

    # Build label tables
    lh_items = [(0, "Null", (158, 158, 158, 255))]
    rh_items = [(0, "Null", (158, 158, 158, 255))]

    for row in roi_registry:
        roi_id = int(row["roi_id"])
        roi_name = str(row["roi_name"])
        family17 = str(row["family17"])
        rgba = network_colors.get(family17, (158, 158, 158, 255))

        if row["hemi"] == "L":
            lh_items.append((roi_id, roi_name, rgba))
        else:
            rh_items.append((roi_id, roi_name, rgba))

    # Save GIFTI label images
    assets_dir = out_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    lh_gii_path = assets_dir / f"sub-{subject}_atlas-kong2022_areal_mshbm.L.label.gii"
    rh_gii_path = assets_dir / f"sub-{subject}_atlas-kong2022_areal_mshbm.R.label.gii"

    lh_gii = make_named_label_gifti(lh_arr, lh_items)
    rh_gii = make_named_label_gifti(rh_arr, rh_items)

    nib.save(lh_gii, str(lh_gii_path))
    nib.save(rh_gii, str(rh_gii_path))

    return lh_gii_path, rh_gii_path


def create_cifti_dlabel(
    subject: str,
    left_label_gii: Path,
    right_label_gii: Path,
    out_dir: Path,
) -> Path:
    assets_dir = out_dir / "assets"
    dlabel_path = assets_dir / f"sub-{subject}_atlas-kong2022_areal_mshbm.dlabel.nii"
    
    run_workbench(
        [
            "-cifti-create-label",
            str(dlabel_path),
            "-left-label",
            str(left_label_gii),
            "-right-label",
            str(right_label_gii),
        ]
    )
    return dlabel_path


def separate_sulc(
    subject: str,
    sulc_dscalar: Path,
    out_dir: Path,
) -> tuple[Path, Path]:
    assets_dir = out_dir / "assets"
    left_sulc_path = assets_dir / f"sub-{subject}_hemi-L_desc-MSMAll_sulc.func.gii"
    right_sulc_path = assets_dir / f"sub-{subject}_hemi-R_desc-MSMAll_sulc.func.gii"
    
    if not left_sulc_path.exists() or not right_sulc_path.exists():
        run_workbench(
            [
                "-cifti-separate",
                str(sulc_dscalar),
                "COLUMN",
                "-metric",
                "CORTEX_LEFT",
                str(left_sulc_path),
                "-metric",
                "CORTEX_RIGHT",
                str(right_sulc_path),
            ]
        )
    return left_sulc_path, right_sulc_path


def run_render_stage() -> int:
    parser = argparse.ArgumentParser(description="Render subject-level Kong2022 MSHBM cortical brain maps.")
    parser.add_argument("--subject", required=True, help="Subject ID (without sub- prefix).")
    parser.add_argument("--routea-root", default=str(DEFAULT_ROUTE_A_ROOT))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--left-surface-template", default=DEFAULT_LEFT_SURFACE_TEMPLATE)
    parser.add_argument("--right-surface-template", default=DEFAULT_RIGHT_SURFACE_TEMPLATE)
    parser.add_argument("--sulc-template", default=DEFAULT_SULC_TEMPLATE)
    parser.add_argument("--source-scene", default=str(DEFAULT_SCENE))
    parser.add_argument("--out-dir", default=None, help="Output directory; defaults to routea-root/sub-subject")
    parser.add_argument("--scene-index", type=int, default=1)
    parser.add_argument("--renderer", default="OSMesa")
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args()

    subject = str(args.subject).replace("sub-", "")
    routea_root = Path(args.routea_root).resolve()
    data_root = Path(args.data_root).resolve()
    source_scene = Path(args.source_scene).resolve()
    
    out_dir = Path(args.out_dir).resolve() if args.out_dir else routea_root / f"sub-{subject}"
    out_dir.mkdir(parents=True, exist_ok=True)

    lh_txt = routea_root / f"sub-{subject}" / "lh_labels.txt"
    rh_txt = routea_root / f"sub-{subject}" / "rh_labels.txt"

    if not lh_txt.exists() or not rh_txt.exists():
        raise FileNotFoundError(f"Missing subject flat label files: {lh_txt} or {rh_txt}")

    left_surface = format_subject_path_template(args.left_surface_template, subject=subject, data_root=data_root)
    right_surface = format_subject_path_template(args.right_surface_template, subject=subject, data_root=data_root)
    sulc_dscalar = format_subject_path_template(args.sulc_template, subject=subject, data_root=data_root)

    for p in [left_surface, right_surface, sulc_dscalar, source_scene]:
        if not p.exists():
            raise FileNotFoundError(p)

    # 1. Load bundles and styles
    bundle = load_kong2022_bundle()
    network_colors = load_network_colors()

    # 2. Generate Gifti labels and Cifti dlabels
    left_label_gii, right_label_gii = generate_gifti_labels(
        subject=subject,
        lh_txt=lh_txt,
        rh_txt=rh_txt,
        out_dir=out_dir,
        bundle=bundle,
        network_colors=network_colors,
    )
    
    dlabel_path = create_cifti_dlabel(
        subject=subject,
        left_label_gii=left_label_gii,
        right_label_gii=right_label_gii,
        out_dir=out_dir,
    )

    # 3. Separate sulc
    left_sulc_gii, right_sulc_gii = separate_sulc(
        subject=subject,
        sulc_dscalar=sulc_dscalar,
        out_dir=out_dir,
    )

    # 4. Render individual views
    figures_dir = out_dir / "figures"
    scenes_dir = out_dir / "scenes"
    figures_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir.mkdir(parents=True, exist_ok=True)

    view_pngs: list[tuple[str, Path]] = []
    for spec in VIEW_SPECS:
        out_scene = scenes_dir / f"sub-{subject}_atlas-kong2022_areal_mshbm_{spec['name']}.scene"
        out_png = figures_dir / f"sub-{subject}_atlas-kong2022_areal_mshbm_{spec['name']}.png"
        
        render_cortex_pfm_scene(
            source_scene=source_scene, left_surface=left_surface, right_surface=right_surface,
            left_label=left_label_gii, right_label=right_label_gii,
            left_sulc=left_sulc_gii, right_sulc=right_sulc_gii,
            sulc_dscalar=sulc_dscalar, dlabel=dlabel_path, out_scene=out_scene, out_png=out_png,
            scene_index=args.scene_index, renderer=args.renderer, width=args.width, height=args.height,
            rotation_axis=spec["axis"], rotation_deg=float(spec["deg"]),
        )
        view_pngs.append((spec["name"], out_png))

    # 5. Compose overview image
    overview_png = figures_dir / f"sub-{subject}_kong2022_areal_mshbm_overview.png"
    compose_overview(subject=subject, view_pngs=view_pngs, out_png=overview_png)
    
    print(json.dumps({
        "status": "ok",
        "subject": f"sub-{subject}",
        "overview_png": str(overview_png),
        "dlabel": str(dlabel_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(
        run_stage(
            StageSpec(
                step_id="06_09_plot_cortex_kong2022_mshbm",
                purpose="Render subject-level Kong2022 MSHBM cortical brain maps.",
                stage_main=run_render_stage,
                upstream_defaults=("03_02_build_kong2022_routea_reference",),
            )
        )
    )

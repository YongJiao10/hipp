#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/System/Library/Fonts/SFNS.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def load_style_json(path: Path) -> dict[int, tuple[str, np.ndarray]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    style: dict[int, tuple[str, np.ndarray]] = {}
    for key_str, spec in data.items():
        key = int(key_str)
        rgba = np.array(spec["rgba"], dtype=np.float32)
        style[key] = (str(spec["name"]), rgba)
    return style


def load_label_gifti(path: Path) -> tuple[np.ndarray, dict[int, tuple[str, np.ndarray]]]:
    img = nib.load(str(path))
    labels = np.asarray(img.darrays[0].data).astype(np.int32)
    table: dict[int, tuple[str, np.ndarray]] = {}
    for lab in img.labeltable.labels:
        key = int(lab.key)
        rgba = np.array([lab.red, lab.green, lab.blue, lab.alpha], dtype=np.float32) * 255.0
        name = getattr(lab, "label", None) or str(key)
        table[key] = (name, rgba)
    return labels, table


def load_labels(path: Path) -> tuple[np.ndarray, dict[int, tuple[str, np.ndarray]]]:
    if path.suffix == ".npy":
        return np.load(path).astype(np.int32), {}
    return load_label_gifti(path)


def build_style(
    left_table: dict[int, tuple[str, np.ndarray]],
    right_table: dict[int, tuple[str, np.ndarray]],
    style_json: Path | None,
) -> dict[int, tuple[str, np.ndarray]]:
    if style_json is not None:
        style = load_style_json(style_json)
    else:
        style = dict(left_table)
        for key, value in right_table.items():
            style.setdefault(key, value)
    return style


def foreground_mask(image: Image.Image, threshold: int = 15) -> np.ndarray:
    arr = np.asarray(image.convert("RGBA"))
    return arr[..., :3].sum(axis=2) > threshold


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


def trim_black(img: Image.Image, threshold: int = 8) -> Image.Image:
    arr = np.asarray(img.convert("RGB"))
    mask = np.any(arr > threshold, axis=2)
    if not np.any(mask):
        return img
    ys, xs = np.where(mask)
    return img.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))


def split_native_hemi_panels(image: Image.Image) -> dict[str, Image.Image]:
    mask = foreground_mask(image)
    col_runs = find_runs(mask.any(axis=0))
    if len(col_runs) < 2:
        raise RuntimeError("Could not detect bilateral foreground runs in native render")
    selected_runs = sorted(col_runs, key=lambda item: item[1] - item[0], reverse=True)[:2]
    selected_runs = sorted(selected_runs, key=lambda item: item[0])
    panels: dict[str, Image.Image] = {}
    for hemi, (x0, x1) in zip(["L", "R"], selected_runs, strict=True):
        panels[hemi] = trim_black(image.crop((x0, 0, x1, image.height)))
    return panels


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


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    lines = [words[0]]
    for word in words[1:]:
        candidate = f"{lines[-1]} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width:
            lines[-1] = candidate
        else:
            lines.append(word)
    return lines


def build_grid_canvas(ventral: Image.Image, dorsal: Image.Image | None, layout: str) -> Image.Image:
    ventral_panels = split_native_hemi_panels(ventral.convert("RGBA"))
    dorsal_panels = split_native_hemi_panels(dorsal.convert("RGBA")) if dorsal is not None else None

    row_gap = 48
    col_gap = 64
    outer_pad = 34
    header_h = 86

    if layout == "1x2":
        cell_w = max(ventral_panels["L"].width, ventral_panels["R"].width)
        row_h = max(ventral_panels["L"].height, ventral_panels["R"].height)
        left_img = fit_image_obj(ventral_panels["L"], cell_w, max_height=row_h)
        right_img = fit_image_obj(ventral_panels["R"], cell_w, max_height=row_h)

        canvas_w = outer_pad * 2 + cell_w * 2 + col_gap
        canvas_h = outer_pad * 2 + header_h + row_h
        canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 255))
        y = outer_pad + header_h
        paste_center(canvas, left_img, outer_pad, y, cell_w, row_h)
        paste_center(canvas, right_img, outer_pad + cell_w + col_gap, y, cell_w, row_h)
    else:
        if dorsal_panels is None:
            raise ValueError("2x2 layout requires dorsal image")
        cell_w = max(
            ventral_panels["L"].width,
            ventral_panels["R"].width,
            dorsal_panels["L"].width,
            dorsal_panels["R"].width,
        )
        top_h = max(ventral_panels["L"].height, ventral_panels["R"].height)
        bottom_h = max(dorsal_panels["L"].height, dorsal_panels["R"].height)

        panels = {
            ("top", "L"): fit_image_obj(ventral_panels["L"], cell_w, max_height=top_h),
            ("top", "R"): fit_image_obj(ventral_panels["R"], cell_w, max_height=top_h),
            ("bottom", "L"): fit_image_obj(dorsal_panels["L"], cell_w, max_height=bottom_h),
            ("bottom", "R"): fit_image_obj(dorsal_panels["R"], cell_w, max_height=bottom_h),
        }
        canvas_w = outer_pad * 2 + cell_w * 2 + col_gap
        canvas_h = outer_pad * 2 + header_h + top_h + row_gap + bottom_h
        canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 255))
        top_y = outer_pad + header_h
        bottom_y = top_y + top_h + row_gap
        paste_center(canvas, panels[("top", "L")], outer_pad, top_y, cell_w, top_h)
        paste_center(canvas, panels[("top", "R")], outer_pad + cell_w + col_gap, top_y, cell_w, top_h)
        paste_center(canvas, panels[("bottom", "L")], outer_pad, bottom_y, cell_w, bottom_h)
        paste_center(canvas, panels[("bottom", "R")], outer_pad + cell_w + col_gap, bottom_y, cell_w, bottom_h)

    draw = ImageDraw.Draw(canvas)
    lr_font = load_font(60)
    left_cx = outer_pad + cell_w // 2
    right_cx = outer_pad + cell_w + col_gap + cell_w // 2
    for text, cx in [("L", left_cx), ("R", right_cx)]:
        box = draw.textbbox((0, 0), text, font=lr_font)
        tw = box[2] - box[0]
        th = box[3] - box[1]
        draw.text((cx - tw // 2, outer_pad + (header_h - th) // 2), text, fill=(255, 255, 255, 255), font=lr_font)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description="Compose ventral/dorsal Workbench renders as a 2x2 grid with right-side legend")
    parser.add_argument("--ventral-image", required=True)
    parser.add_argument("--dorsal-image", default=None)
    parser.add_argument("--layout", choices=["1x2", "2x2"], default="2x2")
    parser.add_argument("--left-labels", required=True)
    parser.add_argument("--right-labels", required=True)
    parser.add_argument("--style-json", default=None)
    parser.add_argument("--legend-group", choices=["label", "network"], default="label")
    parser.add_argument("--title", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ventral = Image.open(args.ventral_image).convert("RGBA")
    dorsal = Image.open(args.dorsal_image).convert("RGBA") if args.dorsal_image else None
    grid = build_grid_canvas(ventral, dorsal, args.layout)
    img_w, img_h = grid.size

    left_labels, left_table = load_labels(Path(args.left_labels))
    right_labels, right_table = load_labels(Path(args.right_labels))
    style = build_style(left_table, right_table, Path(args.style_json) if args.style_json else None)

    combined = np.concatenate([left_labels[left_labels > 0], right_labels[right_labels > 0]])
    total = max(1, combined.size)
    present = sorted(int(k) for k in np.unique(combined) if int(k) > 0)

    group_props: dict[str, float] = {}
    group_colors: dict[str, tuple[int, int, int, int]] = {}
    for key in present:
        if key not in style:
            raise KeyError(f"Missing style entry for key={key}")
        label_name, rgba = style[key]
        group_name = label_name.rsplit("_", 1)[-1] if args.legend_group == "network" else label_name
        prop = float((combined == key).sum()) / total
        group_props[group_name] = group_props.get(group_name, 0.0) + prop
        color = tuple(int(round(v)) for v in rgba[:4])
        if group_name in group_colors and group_colors[group_name] != color:
            raise ValueError(f"Inconsistent colors within legend group '{group_name}'")
        group_colors[group_name] = color

    title_font = load_font(30)
    item_font = load_font(30)
    side_pad = 24
    gap_after_swatch = 16
    swatch_w = 30
    swatch_h = 30
    row_h = 42

    title_text = args.title.strip()
    group_names = sorted(group_props, key=lambda name: (-group_props[name], name))
    item_texts = [f"{name}  {group_props[name] * 100:.1f}%" for name in group_names]

    measure_image = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    measure_draw = ImageDraw.Draw(measure_image)
    text_widths = []
    for text in item_texts:
        bbox = measure_draw.textbbox((0, 0), text, font=item_font)
        text_widths.append(bbox[2] - bbox[0])
    max_text_w = max(text_widths, default=0)
    panel_w = side_pad * 2 + swatch_w + gap_after_swatch + max_text_w + 8
    title_lines = wrap_text(measure_draw, title_text, title_font, max(panel_w - 2 * side_pad, 1))

    canvas = Image.new("RGBA", (img_w + panel_w, img_h), (0, 0, 0, 255))
    canvas.alpha_composite(grid, (0, 0))
    draw = ImageDraw.Draw(canvas)

    x0 = img_w + side_pad
    y = 28
    if title_lines:
        for line in title_lines:
            draw.text((x0, y), line, fill=(255, 255, 255, 255), font=title_font)
            y += 40
        y += 10

    for group_name, item_text in zip(group_names, item_texts):
        color = group_colors[group_name]
        draw.rectangle((x0, y + 6, x0 + swatch_w, y + 6 + swatch_h), fill=color)
        draw.text((x0 + swatch_w + gap_after_swatch, y), item_text, fill=(255, 255, 255, 255), font=item_font)
        y += row_h

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

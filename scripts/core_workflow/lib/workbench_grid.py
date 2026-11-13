#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/System/Library/Fonts/SFNS.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except Exception:
                continue
    try:
        mpl_font = Path(font_manager.findfont("DejaVu Sans", fallback_to_default=True))
        if mpl_font.exists():
            return ImageFont.truetype(str(mpl_font), size=size)
    except Exception:
        pass
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
        return load_style_json(style_json)
    style = dict(left_table)
    for key, value in right_table.items():
        style.setdefault(key, value)
    return style


def foreground_mask(image: Image.Image, threshold: int = 15) -> np.ndarray:
    arr = np.asarray(image.convert("RGBA"))
    corners = np.array([arr[0, 0, :3], arr[0, -1, :3], arr[-1, 0, :3], arr[-1, -1, :3]])
    is_white_bg = float(np.mean(corners)) > 127.0
    if is_white_bg:
        return np.any(arr[..., :3] < (255 - threshold), axis=2)
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


def trim_background(img: Image.Image, threshold: int = 8) -> Image.Image:
    arr = np.asarray(img.convert("RGB"))
    corners = np.array([arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]])
    is_white_bg = float(np.mean(corners)) > 127.0
    if is_white_bg:
        mask = np.any(arr < (255 - threshold), axis=2)
    else:
        mask = np.any(arr > threshold, axis=2)
    if not np.any(mask):
        return img
    ys, xs = np.where(mask)
    return img.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))


def trim_black(img: Image.Image, threshold: int = 8) -> Image.Image:
    return trim_background(img, threshold=threshold)


def split_native_hemi_panels(image: Image.Image) -> dict[str, Image.Image]:
    mask = foreground_mask(image)
    col_runs = find_runs(mask.any(axis=0))
    if len(col_runs) < 2:
        raise RuntimeError("Could not detect bilateral foreground runs in native render")
    selected_runs = sorted(col_runs, key=lambda item: item[1] - item[0], reverse=True)[:2]
    selected_runs = sorted(selected_runs, key=lambda item: item[0])
    panels: dict[str, Image.Image] = {}
    for hemi, (x0, x1) in zip(["L", "R"], selected_runs, strict=True):
        panels[hemi] = trim_background(image.crop((x0, 0, x1, image.height)))
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


def build_grid_canvas(
    ventral: Image.Image,
    dorsal: Image.Image | None,
    layout: str,
    bg_color: tuple[int, int, int, int] = (255, 255, 255, 255),
) -> Image.Image:
    ventral_panels = split_native_hemi_panels(ventral.convert("RGBA"))
    dorsal_panels = split_native_hemi_panels(dorsal.convert("RGBA")) if dorsal is not None else None

    row_gap = 48
    col_gap = 64
    outer_pad = 40
    header_h = 140

    if layout == "1x2":
        cell_w = max(ventral_panels["L"].width, ventral_panels["R"].width)
        row_h = max(ventral_panels["L"].height, ventral_panels["R"].height)
        left_img = fit_image_obj(ventral_panels["L"], cell_w, max_height=row_h)
        right_img = fit_image_obj(ventral_panels["R"], cell_w, max_height=row_h)

        canvas_w = outer_pad * 2 + cell_w * 2 + col_gap
        canvas_h = outer_pad * 2 + header_h + row_h
        canvas = Image.new("RGBA", (canvas_w, canvas_h), bg_color)
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
        canvas = Image.new("RGBA", (canvas_w, canvas_h), bg_color)
        top_y = outer_pad + header_h
        bottom_y = top_y + top_h + row_gap
        paste_center(canvas, panels[("top", "L")], outer_pad, top_y, cell_w, top_h)
        paste_center(canvas, panels[("top", "R")], outer_pad + cell_w + col_gap, top_y, cell_w, top_h)
        paste_center(canvas, panels[("bottom", "L")], outer_pad, bottom_y, cell_w, bottom_h)
        paste_center(canvas, panels[("bottom", "R")], outer_pad + cell_w + col_gap, bottom_y, cell_w, bottom_h)

    draw = ImageDraw.Draw(canvas)
    lr_font = load_font(52)
    left_cx = outer_pad + cell_w // 2
    right_cx = outer_pad + cell_w + col_gap + cell_w // 2
    is_dark = (bg_color[0] + bg_color[1] + bg_color[2]) / 3 < 128
    text_color = (255, 255, 255, 255) if is_dark else (0, 0, 0, 255)
    for text, cx in [("L", left_cx), ("R", right_cx)]:
        box = draw.textbbox((0, 0), text, font=lr_font)
        tw = box[2] - box[0]
        th = box[3] - box[1]
        draw.text((cx - tw // 2, outer_pad + (header_h - th) // 2), text, fill=text_color, font=lr_font)
    return canvas


def compose_wb_grid_with_legend(
    *,
    ventral_image: Path,
    left_labels: Path,
    right_labels: Path,
    out: Path,
    dorsal_image: Path | None = None,
    layout: str = "2x2",
    style_json: Path | None = None,
    legend_group: str = "label",
    title: str = "",
    bg_color: tuple[int, int, int, int] = (255, 255, 255, 255),
    min_panel_w: int = 520,
    legend_display_names: dict[str, str] | None = None,
) -> Path:
    ventral = Image.open(ventral_image).convert("RGBA")
    dorsal = Image.open(dorsal_image).convert("RGBA") if dorsal_image else None
    grid = build_grid_canvas(ventral, dorsal, layout, bg_color=bg_color)
    img_w, img_h = grid.size

    left_vals, left_table = load_labels(left_labels)
    right_vals, right_table = load_labels(right_labels)
    style = build_style(left_table, right_table, style_json)

    def get_group_props(labels: np.ndarray) -> dict[str, float]:
        group_props: dict[str, float] = {}
        total = max(1, labels.size)
        present = sorted(int(k) for k in np.unique(labels))
        for key in present:
            if key not in style:
                raise KeyError(f"Missing style entry for key={key}")
            label_name, _rgba = style[key]
            group_name = label_name.rsplit("_", 1)[-1] if legend_group == "network" else label_name
            prop = float((labels == key).sum()) / total
            group_props[group_name] = group_props.get(group_name, 0.0) + prop
        return group_props

    left_group_props = get_group_props(left_vals)
    right_group_props = get_group_props(right_vals)

    group_colors: dict[str, tuple[int, int, int, int]] = {}
    for _key, (label_name, rgba) in style.items():
        group_name = label_name.rsplit("_", 1)[-1] if legend_group == "network" else label_name
        group_colors[group_name] = tuple(int(round(v)) for v in rgba[:4])

    title_font = load_font(36)
    item_font = load_font(32)
    side_pad = 28
    gap_after_swatch = 14
    swatch_w = 32
    swatch_h = 32
    row_h = 44
    title_text = title.strip()

    measure_image = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    measure_draw = ImageDraw.Draw(measure_image)
    display_names = legend_display_names or {}
    left_item_texts = [f"{display_names.get(n, n)}  {left_group_props[n] * 100:.1f}%" for n in left_group_props]
    right_item_texts = [f"{display_names.get(n, n)}  {right_group_props[n] * 100:.1f}%" for n in right_group_props]
    all_item_texts = left_item_texts + right_item_texts

    max_text_w = 0
    for text in all_item_texts:
        bbox = measure_draw.textbbox((0, 0), text, font=item_font)
        max_text_w = max(max_text_w, bbox[2] - bbox[0])
    panel_w = max(min_panel_w, side_pad * 2 + swatch_w + gap_after_swatch + max_text_w + 8)

    left_title = f"{title_text} (Left)" if title_text else "Left"
    left_title_lines = wrap_text(measure_draw, left_title, title_font, max(panel_w - 2 * side_pad, 1))
    right_title = f"{title_text} (Right)" if title_text else "Right"
    right_title_lines = wrap_text(measure_draw, right_title, title_font, max(panel_w - 2 * side_pad, 1))

    top_pad = 20
    bottom_pad = 20
    inter_legend_gap = 48
    title_line_h = 42

    def legend_block_height(title_lines: list[str], item_count: int) -> int:
        height = 0
        if title_lines:
            height += len(title_lines) * title_line_h + 12
        height += item_count * row_h
        return height

    left_block_h = legend_block_height(left_title_lines, len(left_group_props))
    right_block_h = legend_block_height(right_title_lines, len(right_group_props))
    legend_total_h = top_pad + left_block_h + inter_legend_gap + right_block_h + bottom_pad

    canvas_h = max(img_h, legend_total_h)
    canvas = Image.new("RGBA", (img_w + panel_w, canvas_h), bg_color)
    grid_y = max(0, (canvas_h - img_h) // 2)
    canvas.alpha_composite(grid, (0, grid_y))
    draw = ImageDraw.Draw(canvas)

    is_dark = (bg_color[0] + bg_color[1] + bg_color[2]) / 3 < 128
    text_color = (255, 255, 255, 255) if is_dark else (0, 0, 0, 255)
    swatch_outline = None if is_dark else (210, 210, 210, 255)

    def draw_legend(group_props_map: dict[str, float], x0: int, y_start: int, title_lines: list[str]) -> int:
        y = y_start
        if title_lines:
            for line in title_lines:
                draw.text((x0, y), line, fill=text_color, font=title_font)
                y += title_line_h
            y += 12

        group_names = sorted(group_props_map, key=lambda name: (-group_props_map[name], name))
        item_texts = [f"{display_names.get(name, name)}  {group_props_map[name] * 100:.1f}%" for name in group_names]
        for group_name, item_text in zip(group_names, item_texts):
            color = group_colors[group_name]
            draw.rectangle((x0, y + 5, x0 + swatch_w, y + 5 + swatch_h), fill=color, outline=swatch_outline)
            draw.text((x0 + swatch_w + gap_after_swatch, y), item_text, fill=text_color, font=item_font)
            y += row_h
        return y

    x0 = img_w + side_pad
    y = max(top_pad, (canvas_h - (left_block_h + inter_legend_gap + right_block_h)) // 2)
    y = draw_legend(left_group_props, x0, y, left_title_lines)
    y += inter_legend_gap
    _ = draw_legend(right_group_props, x0, y, right_title_lines)

    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, dpi=(300, 300))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose Workbench renders as a grid with right-side legend")
    parser.add_argument("--ventral-image", required=True)
    parser.add_argument("--dorsal-image", default=None)
    parser.add_argument("--layout", choices=["1x2", "2x2"], default="2x2")
    parser.add_argument("--left-labels", required=True)
    parser.add_argument("--right-labels", required=True)
    parser.add_argument("--style-json", default=None)
    parser.add_argument("--legend-group", choices=["label", "network"], default="label")
    parser.add_argument("--title", default="")
    parser.add_argument("--bg-color", choices=["white", "black"], default="white")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bg_tuple = (255, 255, 255, 255) if args.bg_color == "white" else (0, 0, 0, 255)
    out = compose_wb_grid_with_legend(
        ventral_image=Path(args.ventral_image).resolve(),
        dorsal_image=Path(args.dorsal_image).resolve() if args.dorsal_image else None,
        layout=str(args.layout),
        left_labels=Path(args.left_labels).resolve(),
        right_labels=Path(args.right_labels).resolve(),
        style_json=Path(args.style_json).resolve() if args.style_json else None,
        legend_group=str(args.legend_group),
        title=str(args.title),
        bg_color=bg_tuple,
        out=Path(args.out).resolve(),
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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


def tighten_wb_render(image: Image.Image, outer_pad: int = 36, center_gap: int = 110) -> Image.Image:
    mask = foreground_mask(image)
    cols = mask.any(axis=0)
    runs = find_runs(cols)
    if len(runs) < 2:
        return image

    left_run = runs[0]
    right_run = runs[-1]
    left_crop = image.crop((left_run[0], 0, left_run[1], image.height))
    right_crop = image.crop((right_run[0], 0, right_run[1], image.height))

    compact_w = outer_pad + left_crop.width + center_gap + right_crop.width + outer_pad
    compact = Image.new("RGBA", (compact_w, image.height), (0, 0, 0, 255))
    compact.alpha_composite(left_crop, (outer_pad, 0))
    compact.alpha_composite(right_crop, (outer_pad + left_crop.width + center_gap, 0))
    return compact


def main() -> int:
    parser = argparse.ArgumentParser(description="Compose a Workbench image with a right-side legend panel")
    parser.add_argument("--image", required=True)
    parser.add_argument("--left-labels", required=True)
    parser.add_argument("--right-labels", required=True)
    parser.add_argument("--style-json", default=None)
    parser.add_argument("--legend-group", choices=["label", "network"], default="label")
    parser.add_argument("--title", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    image = tighten_wb_render(Image.open(args.image).convert("RGBA"))
    img_w, img_h = image.size
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
        if args.legend_group == "network":
            group_name = label_name.rsplit("_", 1)[-1]
        else:
            group_name = label_name
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
    canvas.alpha_composite(image, (0, 0))
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

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Compose a Workbench image with a right-side legend panel")
    parser.add_argument("--image", required=True)
    parser.add_argument("--left-labels", required=True)
    parser.add_argument("--right-labels", required=True)
    parser.add_argument("--style-json", default=None)
    parser.add_argument("--title", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGBA")
    img_w, img_h = image.size
    left_labels, left_table = load_labels(Path(args.left_labels))
    right_labels, right_table = load_labels(Path(args.right_labels))
    style = build_style(left_table, right_table, Path(args.style_json) if args.style_json else None)

    combined = np.concatenate([left_labels[left_labels > 0], right_labels[right_labels > 0]])
    total = max(1, combined.size)
    present = sorted(int(k) for k in np.unique(combined) if int(k) > 0)
    props = {k: float((combined == k).sum()) / total for k in present}

    panel_w = 620
    canvas = Image.new("RGBA", (img_w + panel_w, img_h), (0, 0, 0, 255))
    canvas.alpha_composite(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(34)
    item_font = load_font(30)

    x0 = img_w + 28
    y = 28
    if args.title:
        draw.text((x0, y), args.title, fill=(255, 255, 255, 255), font=title_font)
        y += 54

    for key in present:
        name, rgba = style[key]
        color = tuple(int(round(v)) for v in rgba[:4])
        draw.rectangle((x0, y + 6, x0 + 30, y + 36), fill=color)
        draw.text((x0 + 46, y), f"{name}  {props[key] * 100:.1f}%", fill=(255, 255, 255, 255), font=item_font)
        y += 42

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

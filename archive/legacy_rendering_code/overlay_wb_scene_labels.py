#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw


def load_style_json(path: Path) -> dict[int, tuple[str, np.ndarray]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    style: dict[int, tuple[str, np.ndarray]] = {}
    for key_str, spec in data.items():
        key = int(key_str)
        rgba = np.array(spec["rgba"], dtype=np.float32) / 255.0
        style[key] = (str(spec["name"]), rgba)
    return style


def load_label_gifti(path: Path) -> tuple[np.ndarray, dict[int, tuple[str, np.ndarray]]]:
    img = nib.load(str(path))
    labels = np.asarray(img.darrays[0].data).astype(np.int32)
    table: dict[int, tuple[str, np.ndarray]] = {}
    for lab in img.labeltable.labels:
        key = int(lab.key)
        rgba = np.array([lab.red, lab.green, lab.blue, lab.alpha], dtype=np.float32)
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


def find_rotation_matrix(scene_file: Path) -> np.ndarray:
    root = ET.parse(scene_file).getroot()
    for obj in root.iter("Object"):
        if obj.attrib.get("Class") == "ViewingTransformations" and obj.attrib.get("Name") == "m_viewingTransformation":
            for child in obj:
                if child.attrib.get("Name") == "m_rotationMatrix":
                    values = [float(elem.text.strip()) for elem in child.findall("Element")]
                    if len(values) != 16:
                        raise RuntimeError("Unexpected viewing transform matrix size")
                    return np.array(values, dtype=np.float32).reshape(4, 4)[:3, :3]
    raise RuntimeError("Could not find Workbench viewing rotation matrix in scene file")


def project_points(coords: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    centered = coords - np.median(coords, axis=0, keepdims=True)
    rotated = centered @ rotation.T
    projected = np.column_stack([rotated[:, 0], -rotated[:, 1]])
    return projected


def label_centroids(points_2d: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for key in sorted(int(x) for x in np.unique(labels) if x > 0):
        pts = points_2d[labels == key]
        if pts.size == 0:
            continue
        out[key] = np.median(pts, axis=0)
    return out


def find_foreground_bboxes(image: np.ndarray, threshold: int = 6) -> list[tuple[int, int, int, int]]:
    mask = image[:, :, :3].max(axis=2) > threshold
    cols = np.where(mask.any(axis=0))[0]
    if cols.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = int(cols[0])
    prev = int(cols[0])
    for col in cols[1:]:
        col = int(col)
        if col == prev + 1:
            prev = col
            continue
        runs.append((start, prev + 1))
        start = col
        prev = col
    runs.append((start, prev + 1))
    bboxes: list[tuple[int, int, int, int]] = []
    for x0, x1 in runs:
        submask = mask[:, x0:x1]
        ys, xs = np.where(submask)
        if ys.size == 0:
            continue
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        bboxes.append((x0, y0, x1, y1))
    return sorted(bboxes, key=lambda box: box[0])


def map_points_to_bbox(points_2d: np.ndarray, bbox: tuple[int, int, int, int], pad_frac: float = 0.04) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    mins = points_2d.min(axis=0)
    maxs = points_2d.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-6)
    inner_w = max(1.0, (x1 - x0) * (1.0 - 2.0 * pad_frac))
    inner_h = max(1.0, (y1 - y0) * (1.0 - 2.0 * pad_frac))
    scale = min(inner_w / spans[0], inner_h / spans[1])
    scaled = (points_2d - mins) * scale
    scaled_w = spans[0] * scale
    scaled_h = spans[1] * scale
    offset_x = x0 + (x1 - x0 - scaled_w) / 2.0
    offset_y = y0 + (y1 - y0 - scaled_h) / 2.0
    mapped = np.column_stack([scaled[:, 0] + offset_x, scaled[:, 1] + offset_y])
    return mapped


def draw_label_boxes(
    draw: ImageDraw.ImageDraw,
    points: dict[int, np.ndarray],
    style: dict[int, tuple[str, np.ndarray]],
) -> None:
    for key in sorted(points):
        if key <= 0:
            continue
        name = style.get(key, (str(key), np.array([1, 1, 1, 1], dtype=np.float32)))[0]
        x, y = float(points[key][0]), float(points[key][1])
        left, top, right, bottom = draw.textbbox((x, y), name, anchor="mm")
        pad_x = 4
        pad_y = 2
        draw.rectangle(
            (left - pad_x, top - pad_y, right + pad_x, bottom + pad_y),
            fill=(255, 255, 255, 215),
        )
        draw.text((x, y), name, fill=(0, 0, 0), anchor="mm")


def main() -> int:
    parser = argparse.ArgumentParser(description="Overlay label names onto a Workbench scene capture")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--left-surface", required=True)
    parser.add_argument("--right-surface", required=True)
    parser.add_argument("--left-labels", required=True)
    parser.add_argument("--right-labels", required=True)
    parser.add_argument("--style-json", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rotation = find_rotation_matrix(Path(args.scene))
    image = Image.open(args.image).convert("RGBA")
    image_np = np.asarray(image)
    bboxes = find_foreground_bboxes(image_np)
    if len(bboxes) < 2:
        raise RuntimeError("Could not find left/right foreground objects in Workbench render")
    left_bbox, right_bbox = bboxes[:2]

    left_surf = nib.load(args.left_surface)
    right_surf = nib.load(args.right_surface)
    left_coords = np.asarray(left_surf.agg_data("pointset"), dtype=np.float32)
    right_coords = np.asarray(right_surf.agg_data("pointset"), dtype=np.float32)
    left_labels, left_table = load_labels(Path(args.left_labels))
    right_labels, right_table = load_labels(Path(args.right_labels))
    style = build_style(left_table, right_table, Path(args.style_json) if args.style_json else None)

    left_proj = project_points(left_coords, rotation)
    right_proj = project_points(right_coords, rotation)
    left_centroids = label_centroids(left_proj, left_labels)
    right_centroids = label_centroids(right_proj, right_labels)

    left_xy = {
        key: mapped
        for key, mapped in zip(
            sorted(left_centroids),
            map_points_to_bbox(np.vstack([left_centroids[k] for k in sorted(left_centroids)]), left_bbox),
        )
    }
    right_xy = {
        key: mapped
        for key, mapped in zip(
            sorted(right_centroids),
            map_points_to_bbox(np.vstack([right_centroids[k] for k in sorted(right_centroids)]), right_bbox),
        )
    }

    out = image.copy()
    draw = ImageDraw.Draw(out)
    draw_label_boxes(draw, left_xy, style)
    draw_label_boxes(draw, right_xy, style)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

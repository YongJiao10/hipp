#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nibabel as nib
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image


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
    background_rgba: np.ndarray,
) -> dict[int, tuple[str, np.ndarray]]:
    if style_json is not None:
        style = load_style_json(style_json)
    else:
        style = dict(left_table)
        for key, value in right_table.items():
            style.setdefault(key, value)
    style.setdefault(0, ("Background", background_rgba))
    return style


def majority_face_labels(vertex_labels: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri_labels = vertex_labels[faces]
    face_labels = np.empty(faces.shape[0], dtype=np.int32)
    for idx, labs in enumerate(tri_labels):
        values, counts = np.unique(labs, return_counts=True)
        face_labels[idx] = int(values[np.argmax(counts)])
    return face_labels


def proportions(labels: np.ndarray) -> dict[int, float]:
    positive = labels[labels > 0]
    total = max(1, positive.size)
    return {int(k): float((positive == k).sum()) / total for k in np.unique(positive)}


def prepare_hemi_vertices(coords: np.ndarray, mirror_x: bool, shared_scale: float) -> np.ndarray:
    centered = coords - np.median(coords, axis=0, keepdims=True)
    vertices = centered / max(shared_scale, 1e-6)
    if mirror_x:
        vertices[:, 0] *= -1.0
    return vertices


def add_surface(
    ax,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_labels: np.ndarray,
    style: dict[int, tuple[str, np.ndarray]],
) -> None:
    polys = vertices[faces]
    facecolors = np.zeros((faces.shape[0], 4), dtype=np.float32)
    for idx, label in enumerate(face_labels):
        facecolors[idx] = style.get(int(label), ("Unknown", np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)))[1]
    coll = Poly3DCollection(
        polys,
        facecolors=facecolors,
        edgecolors="none",
        linewidths=0.0,
        antialiaseds=False,
    )
    ax.add_collection3d(coll)


def apply_camera(ax, elev: float, azim: float, roll: float, vertical_axis: str, zoom: float) -> None:
    ax.view_init(elev=elev, azim=azim, roll=roll, vertical_axis=vertical_axis)
    ax.set_proj_type("ortho")
    try:
        ax.dist = zoom
    except Exception:
        pass


def set_hemi_limits(ax, vertices: np.ndarray, pad: float) -> None:
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    centers = (mins + maxs) / 2.0
    half_range = float(np.max(maxs - mins) / 2.0)
    if half_range <= 0:
        half_range = 1.0
    half_range *= pad
    ax.set_xlim(centers[0] - half_range, centers[0] + half_range)
    ax.set_ylim(centers[1] - half_range, centers[1] + half_range)
    ax.set_zlim(centers[2] - half_range, centers[2] + half_range)
    ax.set_box_aspect((1, 1, 1))


def render_hemi_rgba(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_labels: np.ndarray,
    style: dict[int, tuple[str, np.ndarray]],
    background_rgb: np.ndarray,
    elev: float,
    azim: float,
    roll: float,
    vertical_axis: str,
    zoom: float,
    pad: float,
    fig_width: float,
    fig_height: float,
    dpi: int,
) -> np.ndarray:
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    fig.patch.set_facecolor(background_rgb)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection="3d")
    ax.set_facecolor(background_rgb)
    ax.set_axis_off()
    add_surface(ax, vertices, faces, face_labels, style)
    set_hemi_limits(ax, vertices, pad=pad)
    apply_camera(ax, elev=elev, azim=azim, roll=roll, vertical_axis=vertical_axis, zoom=zoom)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return rgba


def crop_to_foreground(rgba: np.ndarray, background_rgb: np.ndarray, threshold: int = 6) -> np.ndarray:
    bg = np.round(background_rgb * 255.0).astype(np.uint8)
    delta = np.abs(rgba[:, :, :3].astype(np.int16) - bg[None, None, :].astype(np.int16))
    mask = (delta.max(axis=2) > threshold) & (rgba[:, :, 3] > 0)
    if not np.any(mask):
        return rgba
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return rgba[y0:y1, x0:x1, :]


def find_foreground_bboxes_in_reference(ref_rgba: np.ndarray, threshold: int = 6) -> list[tuple[int, int, int, int]]:
    mask = (ref_rgba[:, :, :3].max(axis=2) > threshold) & (ref_rgba[:, :, 3] > 0)
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
        bboxes.append((x0, y0, x0 + int(xs.max()) + 1, y1))
    return sorted(bboxes, key=lambda box: box[0])


def resize_to_fit(rgba: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    src_h, src_w = rgba.shape[:2]
    if src_h <= 0 or src_w <= 0 or target_w <= 0 or target_h <= 0:
        return rgba
    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    img = Image.fromarray(rgba, mode="RGBA").resize((new_w, new_h), Image.Resampling.LANCZOS)
    return np.asarray(img)


def alpha_composite(dst: np.ndarray, src: np.ndarray, x0: int, y0: int) -> np.ndarray:
    out = dst.copy()
    h, w = src.shape[:2]
    x1 = min(out.shape[1], x0 + w)
    y1 = min(out.shape[0], y0 + h)
    if x1 <= x0 or y1 <= y0:
        return out
    src_crop = src[: y1 - y0, : x1 - x0, :].astype(np.float32) / 255.0
    dst_crop = out[y0:y1, x0:x1, :].astype(np.float32) / 255.0
    src_a = src_crop[:, :, 3:4]
    dst_a = dst_crop[:, :, 3:4]
    out_a = src_a + dst_a * (1.0 - src_a)
    out_rgb = np.where(
        out_a > 1e-6,
        (src_crop[:, :, :3] * src_a + dst_crop[:, :, :3] * dst_a * (1.0 - src_a)) / out_a,
        0.0,
    )
    composed = np.concatenate([out_rgb, out_a], axis=2)
    out[y0:y1, x0:x1, :] = np.clip(np.round(composed * 255.0), 0, 255).astype(np.uint8)
    return out


def compose_from_reference_layout(
    left_rgba: np.ndarray,
    right_rgba: np.ndarray,
    background_rgb: np.ndarray,
    ref_image: Path,
) -> np.ndarray:
    ref_rgba = np.asarray(Image.open(ref_image).convert("RGBA"))
    bboxes = find_foreground_bboxes_in_reference(ref_rgba)
    if len(bboxes) < 2:
        raise ValueError(f"Could not find two foreground components in reference image: {ref_image}")
    left_box, right_box = bboxes[:2]
    canvas = np.zeros_like(ref_rgba)
    canvas[:, :, :3] = np.round(background_rgb * 255.0).astype(np.uint8)
    canvas[:, :, 3] = 255
    for hemi_rgba, (x0, y0, x1, y1) in [(left_rgba, left_box), (right_rgba, right_box)]:
        target = resize_to_fit(hemi_rgba, target_w=x1 - x0, target_h=y1 - y0)
        ox = x0 + max(0, ((x1 - x0) - target.shape[1]) // 2)
        oy = y0 + max(0, ((y1 - y0) - target.shape[0]) // 2)
        canvas = alpha_composite(canvas, target, ox, oy)
    return canvas


def compose_from_layout_template(
    left_rgba: np.ndarray,
    right_rgba: np.ndarray,
    background_rgb: np.ndarray,
    template_json: Path,
) -> np.ndarray:
    template = json.loads(template_json.read_text(encoding="utf-8"))
    canvas_w = int(template["canvas_width"])
    canvas_h = int(template["canvas_height"])
    left_box = tuple(int(v) for v in template["left_box"])
    right_box = tuple(int(v) for v in template["right_box"])
    canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
    canvas[:, :, :3] = np.round(background_rgb * 255.0).astype(np.uint8)
    canvas[:, :, 3] = 255
    for hemi_rgba, (x0, y0, x1, y1) in [(left_rgba, left_box), (right_rgba, right_box)]:
        target = resize_to_fit(hemi_rgba, target_w=x1 - x0, target_h=y1 - y0)
        ox = x0 + max(0, ((x1 - x0) - target.shape[1]) // 2)
        oy = y0 + max(0, ((y1 - y0) - target.shape[0]) // 2)
        canvas = alpha_composite(canvas, target, ox, oy)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description="Render native/folded hippocampal labels with a true 3D camera")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--left-surface", required=True)
    parser.add_argument("--right-surface", required=True)
    parser.add_argument("--left-labels", required=True)
    parser.add_argument("--right-labels", required=True)
    parser.add_argument("--style-json", default=None)
    parser.add_argument("--title", default="")
    parser.add_argument("--legend-title", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--elev", type=float, default=15.0)
    parser.add_argument("--azim", type=float, default=10.0)
    parser.add_argument("--roll", type=float, default=0.0)
    parser.add_argument("--vertical-axis", default="z", choices=["x", "y", "z"])
    parser.add_argument("--zoom", type=float, default=7.5)
    parser.add_argument("--pad", type=float, default=1.06)
    parser.add_argument("--fig-width", type=float, default=10.5)
    parser.add_argument("--fig-height", type=float, default=6.8)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--layout-ref-image", default=None)
    parser.add_argument("--layout-template-json", default=None)
    parser.add_argument("--background", default="black", choices=["black", "white"])
    parser.add_argument("--show-legend", action="store_true")
    args = parser.parse_args()

    bg_rgb = np.array([0.0, 0.0, 0.0]) if args.background == "black" else np.array([1.0, 1.0, 1.0])
    bg_rgba = np.array([*bg_rgb, 1.0], dtype=np.float32)

    left_surf = nib.load(args.left_surface)
    right_surf = nib.load(args.right_surface)
    left_vertices_raw = np.asarray(left_surf.agg_data("pointset"), dtype=np.float32)
    right_vertices_raw = np.asarray(right_surf.agg_data("pointset"), dtype=np.float32)
    left_faces = np.asarray(left_surf.agg_data("triangle"), dtype=np.int32)
    right_faces = np.asarray(right_surf.agg_data("triangle"), dtype=np.int32)

    left_labels, left_table = load_labels(Path(args.left_labels))
    right_labels, right_table = load_labels(Path(args.right_labels))
    style = build_style(left_table, right_table, Path(args.style_json) if args.style_json else None, bg_rgba)

    left_centered = left_vertices_raw - np.median(left_vertices_raw, axis=0, keepdims=True)
    right_centered = right_vertices_raw - np.median(right_vertices_raw, axis=0, keepdims=True)
    shared_scale = float(
        max(
            np.percentile(np.linalg.norm(left_centered, axis=1), 99),
            np.percentile(np.linalg.norm(right_centered, axis=1), 99),
            1e-6,
        )
    )
    left_vertices = prepare_hemi_vertices(left_vertices_raw, mirror_x=True, shared_scale=shared_scale)
    right_vertices = prepare_hemi_vertices(right_vertices_raw, mirror_x=False, shared_scale=shared_scale)

    left_face_labels = majority_face_labels(left_labels, left_faces)
    right_face_labels = majority_face_labels(right_labels, right_faces)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    left_rgba = crop_to_foreground(
        render_hemi_rgba(
            left_vertices,
            left_faces,
            left_face_labels,
            style,
            bg_rgb,
            elev=args.elev,
            azim=args.azim,
            roll=args.roll,
            vertical_axis=args.vertical_axis,
            zoom=args.zoom,
            pad=args.pad,
            fig_width=args.fig_width / 2.0,
            fig_height=args.fig_height,
            dpi=args.dpi,
        ),
        bg_rgb,
    )
    right_rgba = crop_to_foreground(
        render_hemi_rgba(
            right_vertices,
            right_faces,
            right_face_labels,
            style,
            bg_rgb,
            elev=args.elev,
            azim=args.azim,
            roll=args.roll,
            vertical_axis=args.vertical_axis,
            zoom=args.zoom,
            pad=args.pad,
            fig_width=args.fig_width / 2.0,
            fig_height=args.fig_height,
            dpi=args.dpi,
        ),
        bg_rgb,
    )

    if args.layout_template_json:
        composed = compose_from_layout_template(
            left_rgba,
            right_rgba,
            background_rgb=bg_rgb,
            template_json=Path(args.layout_template_json),
        )
        Image.fromarray(composed, mode="RGBA").save(out_path)
    elif args.layout_ref_image:
        composed = compose_from_reference_layout(
            left_rgba,
            right_rgba,
            background_rgb=bg_rgb,
            ref_image=Path(args.layout_ref_image),
        )
        Image.fromarray(composed, mode="RGBA").save(out_path)
    else:
        fig = plt.figure(figsize=(args.fig_width, args.fig_height), constrained_layout=True)
        fig.patch.set_facecolor(bg_rgb)
        if args.title:
            fig.suptitle(args.title, fontsize=14, fontweight="bold", color="white" if args.background == "black" else "black")
        ax = fig.add_subplot(111)
        ax.imshow(np.concatenate([left_rgba, right_rgba], axis=1))
        ax.axis("off")
        if args.show_legend:
            combined = np.concatenate([left_labels[left_labels > 0], right_labels[right_labels > 0]])
            combined_props = proportions(combined)
            handles = []
            for key in sorted(k for k in style if k > 0 and combined_props.get(k, 0.0) > 0):
                name, rgba = style[key]
                handles.append(mpatches.Patch(color=rgba, label=f"{name}  {combined_props[key] * 100:.1f}%"))
            fig.legend(
                handles=handles,
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                frameon=False,
                title=args.legend_title or args.subject,
                labelcolor="white" if args.background == "black" else "black",
            )
        fig.savefig(out_path, dpi=300, facecolor=bg_rgb, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

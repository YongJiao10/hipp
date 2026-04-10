#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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


def load_scalar(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)
    img = nib.load(str(path))
    return np.asarray(img.darrays[0].data, dtype=np.float32)


def prepare_hemi_vertices(coords: np.ndarray, mirror_x: bool, shared_scale: float) -> np.ndarray:
    centered = coords - np.median(coords, axis=0, keepdims=True)
    vertices = centered / max(shared_scale, 1e-6)
    if mirror_x:
        vertices[:, 0] *= -1.0
    return vertices


def face_values(vertex_values: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return vertex_values[faces].mean(axis=1)


def add_surface(
    ax,
    vertices: np.ndarray,
    faces: np.ndarray,
    values: np.ndarray,
    norm: colors.Normalize,
    cmap,
) -> None:
    polys = vertices[faces]
    vals = face_values(values, faces)
    facecolors = cmap(norm(vals))
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
    values: np.ndarray,
    norm: colors.Normalize,
    cmap,
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
    add_surface(ax, vertices, faces, values, norm, cmap)
    set_hemi_limits(ax, vertices, pad=pad)
    apply_camera(ax, elev=elev, azim=azim, roll=roll, vertical_axis=vertical_axis, zoom=zoom)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return rgba


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
    scale = min(target_w / max(1, src_w), target_h / max(1, src_h))
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


def compose_from_layout_template(
    left_rgba: np.ndarray,
    right_rgba: np.ndarray,
    background_rgb: np.ndarray,
    template_json: Path,
) -> np.ndarray:
    import json

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


def append_colorbar(
    rgba: np.ndarray,
    norm: colors.Normalize,
    cmap,
    label: str,
    background_rgb: np.ndarray,
) -> np.ndarray:
    panel_w = 150
    panel = Image.new("RGBA", (panel_w, rgba.shape[0]), tuple(np.round(background_rgb * 255.0).astype(np.uint8)) + (255,))
    draw = ImageDraw.Draw(panel)
    title_font = load_font(28)
    tick_font = load_font(22)

    bar_w = 34
    bar_h = int(rgba.shape[0] * 0.48)
    bar_x = 36
    bar_y = int((rgba.shape[0] - bar_h) / 2)

    values = np.linspace(norm.vmax, norm.vmin, bar_h, dtype=np.float32)
    bar_rgba = np.round(cmap(norm(values)) * 255.0).astype(np.uint8).reshape(bar_h, 1, 4)
    bar_rgba = np.repeat(bar_rgba, bar_w, axis=1)
    panel.alpha_composite(Image.fromarray(bar_rgba, mode="RGBA"), (bar_x, bar_y))
    draw.rectangle((bar_x, bar_y, bar_x + bar_w, bar_y + bar_h), outline=(255, 255, 255, 255), width=1)

    draw.text((22, bar_y - 52), label, fill=(255, 255, 255, 255), font=title_font)
    ticks = [(norm.vmax, bar_y), (0.0, bar_y + bar_h // 2), (norm.vmin, bar_y + bar_h)]
    for value, y in ticks:
        draw.line((bar_x + bar_w + 6, y, bar_x + bar_w + 16, y), fill=(255, 255, 255, 255), width=1)
        draw.text((bar_x + bar_w + 22, y - 12), f"{value:.2f}", fill=(255, 255, 255, 255), font=tick_font)

    canvas = np.zeros((rgba.shape[0], rgba.shape[1] + panel_w, 4), dtype=np.uint8)
    canvas[:, :, :3] = np.round(background_rgb * 255.0).astype(np.uint8)
    canvas[:, :, 3] = 255
    canvas[:, : rgba.shape[1], :] = rgba
    panel_rgba = np.asarray(panel, dtype=np.uint8)
    canvas[:, rgba.shape[1] :, :] = panel_rgba
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description="Render continuous hippocampal surface scalars as a native/folded 3D PNG")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--left-surface", required=True)
    parser.add_argument("--right-surface", required=True)
    parser.add_argument("--left-scalars", required=True)
    parser.add_argument("--right-scalars", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cmap", default="coolwarm")
    parser.add_argument("--colorbar-label", default="Gradient 1")
    parser.add_argument("--robust-percentile", type=float, default=98.0)
    parser.add_argument("--vertical-axis", default="z")
    parser.add_argument("--elev", type=float, default=20.0)
    parser.add_argument("--azim", type=float, default=-90.0)
    parser.add_argument("--roll", type=float, default=0.0)
    parser.add_argument("--zoom", type=float, default=7.5)
    parser.add_argument("--pad", type=float, default=1.08)
    parser.add_argument("--dpi", type=int, default=260)
    parser.add_argument("--fig-width", type=float, default=7.6)
    parser.add_argument("--fig-height", type=float, default=7.6)
    parser.add_argument("--background", default="#000000")
    parser.add_argument("--layout-template-json", default="config/native_surface_layout_template.json")
    args = parser.parse_args()

    background_rgb = np.array(matplotlib.colors.to_rgb(args.background), dtype=np.float32)
    cmap = cm.get_cmap(args.cmap)

    left_surf = nib.load(args.left_surface)
    right_surf = nib.load(args.right_surface)
    left_coords = left_surf.agg_data("pointset").astype(np.float32)
    right_coords = right_surf.agg_data("pointset").astype(np.float32)
    left_faces = left_surf.agg_data("triangle")
    right_faces = right_surf.agg_data("triangle")
    left_vals = load_scalar(Path(args.left_scalars))
    right_vals = load_scalar(Path(args.right_scalars))

    shared_scale = float(
        max(
            np.ptp(left_coords, axis=0).max(initial=1.0),
            np.ptp(right_coords, axis=0).max(initial=1.0),
        )
    )
    left_vertices = prepare_hemi_vertices(left_coords, mirror_x=True, shared_scale=shared_scale)
    right_vertices = prepare_hemi_vertices(right_coords, mirror_x=False, shared_scale=shared_scale)

    combined = np.concatenate([left_vals[np.isfinite(left_vals)], right_vals[np.isfinite(right_vals)]])
    vmax = float(np.percentile(np.abs(combined), args.robust_percentile))
    vmax = max(vmax, 1e-6)
    norm = colors.Normalize(vmin=-vmax, vmax=vmax)

    left_rgba = render_hemi_rgba(
        left_vertices,
        left_faces,
        left_vals,
        norm,
        cmap,
        background_rgb,
        args.elev,
        args.azim,
        args.roll,
        args.vertical_axis,
        args.zoom,
        args.pad,
        args.fig_width,
        args.fig_height,
        args.dpi,
    )
    right_rgba = render_hemi_rgba(
        right_vertices,
        right_faces,
        right_vals,
        norm,
        cmap,
        background_rgb,
        args.elev,
        args.azim,
        args.roll,
        args.vertical_axis,
        args.zoom,
        args.pad,
        args.fig_width,
        args.fig_height,
        args.dpi,
    )

    composed = compose_from_layout_template(
        left_rgba,
        right_rgba,
        background_rgb,
        Path(args.layout_template_json),
    )
    with_bar = append_colorbar(composed, norm, cmap, args.colorbar_label, background_rgb)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(with_bar, mode="RGBA").save(out_path)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap


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
    style.setdefault(0, ("Background", np.array([0.95, 0.95, 0.95, 1.0], dtype=np.float32)))
    return style


def make_colormap(style: dict[int, tuple[str, np.ndarray]]) -> ListedColormap:
    max_key = max(style)
    colors = np.zeros((max_key + 1, 4), dtype=np.float32)
    for key, (_, rgba) in style.items():
        colors[key] = rgba
    return ListedColormap(colors)


def project_native_surface(coords: np.ndarray, hemi: str) -> np.ndarray:
    centered = coords - np.median(coords, axis=0, keepdims=True)
    # Fixed view parameters selected by a 1000-trial search against the
    # requested reference layout. Keeping this deterministic enforces the
    # same angle across all subjects.
    x = centered[:, 0]
    y = centered[:, 1]
    z = centered[:, 2]
    horizontal = 1.0 * x + 0.685362486825064 * y + 0.15068431890584774 * z
    vertical = 0.08032837071328769 * x + 0.29228430412537704 * y + 1.0 * z

    # User-requested global rotation adjustment: rot90
    theta = np.deg2rad(33.76566781731727 + 90.0)
    c = np.cos(theta)
    s = np.sin(theta)
    h_rot = c * horizontal - s * vertical
    v_rot = s * horizontal + c * vertical
    horizontal = h_rot
    vertical = 1.6217994796607889 * v_rot

    coords_2d = np.column_stack([horizontal, vertical])

    # Mirror hemispheres so the medial side faces inward, matching the prior layout.
    if hemi == "L":
        coords_2d[:, 0] *= -1.0

    return coords_2d


def label_centroids(coords_2d: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for key in sorted(int(x) for x in np.unique(labels) if x > 0):
        pts = coords_2d[labels == key]
        if pts.size == 0:
            continue
        out[key] = np.median(pts, axis=0)
    return out


def proportions(labels: np.ndarray) -> dict[int, float]:
    positive = labels[labels > 0]
    total = max(1, positive.size)
    return {int(k): float((positive == k).sum()) / total for k in np.unique(positive)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Render native/folded hippocampal surface labels as a PNG")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--left-surface", required=True)
    parser.add_argument("--right-surface", required=True)
    parser.add_argument("--left-labels", required=True, help=".label.gii or .npy")
    parser.add_argument("--right-labels", required=True, help=".label.gii or .npy")
    parser.add_argument("--style-json", default=None)
    parser.add_argument("--title", required=True)
    parser.add_argument("--legend-title", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    left_surf = nib.load(args.left_surface)
    right_surf = nib.load(args.right_surface)
    left_coords = project_native_surface(left_surf.agg_data("pointset"), "L")
    right_coords = project_native_surface(right_surf.agg_data("pointset"), "R")
    left_faces = left_surf.agg_data("triangle")
    right_faces = right_surf.agg_data("triangle")

    left_labels, left_table = load_labels(Path(args.left_labels))
    right_labels, right_table = load_labels(Path(args.right_labels))
    style = build_style(left_table, right_table, Path(args.style_json) if args.style_json else None)
    cmap = make_colormap(style)

    fig, axes = plt.subplots(1, 2, figsize=(14, 8.8), constrained_layout=True)
    panels = [
        ("Left Hippocampus", axes[0], left_coords, left_faces, left_labels),
        ("Right Hippocampus", axes[1], right_coords, right_faces, right_labels),
    ]

    for title, ax, coords, faces, labels in panels:
        ax.tripcolor(coords[:, 0], coords[:, 1], faces, labels, cmap=cmap, shading="flat", vmin=0, vmax=max(style))
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_aspect("equal")
        ax.axis("off")
        for key, xy in label_centroids(coords, labels).items():
            name = style.get(key, (str(key), np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)))[0]
            ax.text(
                float(xy[0]),
                float(xy[1]),
                name,
                fontsize=7,
                ha="center",
                va="center",
                color="black",
                bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
            )

    combined = np.concatenate([left_labels[left_labels > 0], right_labels[right_labels > 0]])
    combined_props = proportions(combined)
    legend_handles = []
    for key in sorted(k for k in style if k > 0 and combined_props.get(k, 0.0) > 0):
        name, rgba = style[key]
        legend_handles.append(mpatches.Patch(color=rgba, label=f"{name}  {combined_props[key] * 100:.1f}%"))
    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        title=args.legend_title,
    )
    fig.suptitle(args.title, fontsize=14, fontweight="bold")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

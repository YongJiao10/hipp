#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nibabel as nib
import numpy as np
import scipy.stats
from matplotlib.colors import ListedColormap


def load_style(path: Path) -> dict[int, dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): v for k, v in data.items()}


def make_colormap(style: dict[int, dict[str, object]]) -> ListedColormap:
    max_key = max(style)
    colors = np.zeros((max_key + 1, 4), dtype=np.float32)
    colors[0] = np.array([0.95, 0.95, 0.95, 1.0], dtype=np.float32)
    for key, spec in style.items():
        rgba = np.array(spec["rgba"], dtype=np.float32) / 255.0
        colors[key] = rgba
    return ListedColormap(colors)


def label_centroids(coords: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for key in sorted(int(x) for x in np.unique(labels) if x > 0):
        pts = coords[labels == key]
        if pts.size == 0:
            continue
        out[key] = np.median(pts[:, :2], axis=0)
    return out


def proportions(labels: np.ndarray) -> dict[int, float]:
    positive = labels[labels > 0]
    total = max(1, positive.size)
    return {int(k): float((positive == k).sum()) / total for k in np.unique(positive)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Render unfolded hippocampal WTA labels as a publication-style PNG")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--left-surface", required=True)
    parser.add_argument("--right-surface", required=True)
    parser.add_argument("--left-labels", required=True, help=".npy or .label.gii")
    parser.add_argument("--right-labels", required=True, help=".npy or .label.gii")
    parser.add_argument("--style-json", default="config/hipp_network_style.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    style = load_style(Path(args.style_json))
    cmap = make_colormap(style)

    def load_labels(path_str: str) -> np.ndarray:
        path = Path(path_str)
        if path.suffix == ".npy":
            return np.load(path).astype(np.int32)
        return np.asarray(nib.load(str(path)).darrays[0].data).astype(np.int32)

    left_surf = nib.load(args.left_surface)
    right_surf = nib.load(args.right_surface)
    left_coords = left_surf.agg_data("pointset")
    right_coords = right_surf.agg_data("pointset")
    left_faces = left_surf.agg_data("triangle")
    right_faces = right_surf.agg_data("triangle")
    left_labels = load_labels(args.left_labels)
    right_labels = load_labels(args.right_labels)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    panels = [
        ("Left Hippocampus", axes[0], left_coords, left_faces, left_labels),
        ("Right Hippocampus", axes[1], right_coords, right_faces, right_labels),
    ]

    for title, ax, coords, faces, labels in panels:
        try:
            face_labels = scipy.stats.mode(labels[faces], axis=1, keepdims=False)[0]
        except TypeError:
            face_labels = scipy.stats.mode(labels[faces], axis=1)[0].squeeze()
            
        ax.tripcolor(coords[:, 0], coords[:, 1], faces, face_labels, cmap=cmap, shading="flat", vmin=0, vmax=max(style))
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")
        for key, xy in label_centroids(coords, labels).items():
            ax.text(
                float(xy[0]),
                float(xy[1]),
                str(style[key]["name"]),
                fontsize=8,
                ha="center",
                va="center",
                color="black",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
            )

    combined = np.concatenate([left_labels[left_labels > 0], right_labels[right_labels > 0]])
    combined_props = proportions(combined)
    legend_handles = []
    for key in sorted(style):
        frac = combined_props.get(key, 0.0)
        rgba = np.array(style[key]["rgba"], dtype=np.float32) / 255.0
        legend_handles.append(
            mpatches.Patch(
                color=rgba,
                label=f"{style[key]['name']}  {frac * 100:.1f}%",
            )
        )
    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        title=f"sub-{args.subject} Networks",
    )
    fig.suptitle(f"sub-{args.subject} Hippocampal WTA Networks (Unfolded)", fontsize=14, fontweight="bold")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap


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


def build_style(left_table: dict[int, tuple[str, np.ndarray]], right_table: dict[int, tuple[str, np.ndarray]]) -> dict[int, tuple[str, np.ndarray]]:
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
    parser = argparse.ArgumentParser(description="Render unfolded hippocampal structural subfields as a publication-style PNG")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--left-surface", required=True)
    parser.add_argument("--right-surface", required=True)
    parser.add_argument("--left-labels", required=True, help=".label.gii")
    parser.add_argument("--right-labels", required=True, help=".label.gii")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    left_surf = nib.load(args.left_surface)
    right_surf = nib.load(args.right_surface)
    left_coords = left_surf.agg_data("pointset")
    right_coords = right_surf.agg_data("pointset")
    left_faces = left_surf.agg_data("triangle")
    right_faces = right_surf.agg_data("triangle")

    left_labels, left_table = load_label_gifti(Path(args.left_labels))
    right_labels, right_table = load_label_gifti(Path(args.right_labels))
    style = build_style(left_table, right_table)
    cmap = make_colormap(style)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    panels = [
        ("Left Hippocampus", axes[0], left_coords, left_faces, left_labels),
        ("Right Hippocampus", axes[1], right_coords, right_faces, right_labels),
    ]

    for title, ax, coords, faces, labels in panels:
        ax.tripcolor(coords[:, 0], coords[:, 1], faces, labels, cmap=cmap, shading="flat", vmin=0, vmax=max(style))
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")
        for key, xy in label_centroids(coords, labels).items():
            name = style[key][0]
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
    for key in sorted(k for k in style if k > 0):
        name, rgba = style[key]
        frac = combined_props.get(key, 0.0)
        legend_handles.append(mpatches.Patch(color=rgba, label=f"{name}  {frac * 100:.1f}%"))
    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        title=f"sub-{args.subject} Structural Subfields",
    )
    fig.suptitle(f"sub-{args.subject} Hippocampal Structural Subfields (Unfolded)", fontsize=14, fontweight="bold")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

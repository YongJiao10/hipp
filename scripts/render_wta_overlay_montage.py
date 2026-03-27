#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nibabel as nib
import numpy as np


def load_style(path: Path) -> dict[int, dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): v for k, v in data.items()}


def load_labels(path: Path) -> np.ndarray:
    return np.load(path).astype(np.int32)


def world_to_voxel(coords_mm: np.ndarray, affine: np.ndarray) -> np.ndarray:
    inv_affine = np.linalg.inv(affine)
    homo = np.c_[coords_mm, np.ones(coords_mm.shape[0], dtype=np.float32)]
    vox = (inv_affine @ homo.T).T[:, :3]
    return vox


def robust_normalize(volume: np.ndarray) -> np.ndarray:
    nonzero = volume[np.isfinite(volume) & (volume > 0)]
    if nonzero.size == 0:
        return np.zeros_like(volume, dtype=np.float32)
    lo, hi = np.percentile(nonzero, [2.0, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    scaled = (volume.astype(np.float32) - lo) / (hi - lo)
    return np.clip(scaled, 0.0, 1.0)


def select_slices(values: np.ndarray, n_slices: int) -> list[int]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return [0] * n_slices
    lo = int(np.floor(np.quantile(finite, 0.05)))
    hi = int(np.ceil(np.quantile(finite, 0.95)))
    if hi <= lo:
        center = int(round(np.median(finite)))
        return [center] * n_slices
    raw = np.linspace(lo, hi, n_slices)
    unique = []
    for value in raw:
        ivalue = int(round(value))
        if not unique or ivalue != unique[-1]:
            unique.append(ivalue)
    while len(unique) < n_slices:
        unique.append(unique[-1])
    return unique[:n_slices]


def select_bilateral_sagittal_slices(left_x: np.ndarray, right_x: np.ndarray, n_slices: int) -> list[int]:
    n_left = max(1, n_slices // 2)
    n_right = max(1, n_slices - n_left)
    slices = select_slices(left_x, n_left) + select_slices(right_x, n_right)
    return sorted(int(x) for x in slices)


def bounding_box(points_2d: np.ndarray, margin: int, max_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    mins = np.floor(points_2d.min(axis=0)).astype(int) - margin
    maxs = np.ceil(points_2d.max(axis=0)).astype(int) + margin
    x0 = int(np.clip(mins[0], 0, max_shape[0] - 1))
    x1 = int(np.clip(maxs[0], x0 + 1, max_shape[0]))
    y0 = int(np.clip(mins[1], 0, max_shape[1] - 1))
    y1 = int(np.clip(maxs[1], y0 + 1, max_shape[1]))
    return x0, x1, y0, y1


def combined_proportions(left_labels: np.ndarray, right_labels: np.ndarray, style: dict[int, dict[str, object]]) -> list[tuple[int, str, int, float]]:
    combined = np.concatenate([left_labels[left_labels > 0], right_labels[right_labels > 0]])
    total = max(1, int(combined.size))
    rows = []
    for key in sorted(style):
        count = int(np.count_nonzero(combined == key))
        rows.append((key, str(style[key]["name"]), count, count / total))
    return rows


def plot_plane(
    ax: plt.Axes,
    volume: np.ndarray,
    coords_vox: np.ndarray,
    labels: np.ndarray,
    style: dict[int, dict[str, object]],
    axis: int,
    slice_index: int,
    slab_thickness: float,
    crop: tuple[int, int, int, int],
    title: str,
) -> None:
    if axis == 0:
        base = volume[slice_index, :, :].T
        plane_coords = coords_vox[:, [1, 2]]
        distance = np.abs(coords_vox[:, 0] - slice_index)
    elif axis == 1:
        base = volume[:, slice_index, :].T
        plane_coords = coords_vox[:, [0, 2]]
        distance = np.abs(coords_vox[:, 1] - slice_index)
    else:
        base = volume[:, :, slice_index].T
        plane_coords = coords_vox[:, [0, 1]]
        distance = np.abs(coords_vox[:, 2] - slice_index)

    x0, x1, y0, y1 = crop
    ax.imshow(base[y0:y1, x0:x1], cmap="gray", origin="lower", interpolation="nearest", vmin=0.0, vmax=1.0)

    slab_mask = (distance <= slab_thickness) & (labels > 0)
    for key in sorted(style):
        mask = slab_mask & (labels == key)
        if not np.any(mask):
            continue
        pts = plane_coords[mask]
        ax.scatter(
            pts[:, 0] - x0,
            pts[:, 1] - y0,
            s=18,
            marker="s",
            c=[np.array(style[key]["rgba"], dtype=np.float32) / 255.0],
            linewidths=0.0,
            alpha=0.92,
        )

    ax.set_title(title, fontsize=10, pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render native slab-style hippocampal WTA overlay montage")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--t2w", required=True)
    parser.add_argument("--left-surface", required=True)
    parser.add_argument("--right-surface", required=True)
    parser.add_argument("--left-labels", required=True)
    parser.add_argument("--right-labels", required=True)
    parser.add_argument("--style-json", default="config/hipp_network_style.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-slices", type=int, default=6)
    parser.add_argument("--slab-thickness", type=float, default=1.25)
    parser.add_argument("--margin", type=int, default=10)
    args = parser.parse_args()

    style = load_style(Path(args.style_json))
    left_labels = load_labels(Path(args.left_labels))
    right_labels = load_labels(Path(args.right_labels))

    t2w_nii = nib.load(args.t2w)
    t2w = robust_normalize(np.asarray(t2w_nii.get_fdata(), dtype=np.float32))

    left_coords = nib.load(args.left_surface).agg_data("pointset")
    right_coords = nib.load(args.right_surface).agg_data("pointset")
    left_vox = world_to_voxel(left_coords, t2w_nii.affine)
    right_vox = world_to_voxel(right_coords, t2w_nii.affine)
    coords_vox = np.vstack([left_vox, right_vox])
    labels = np.concatenate([left_labels, right_labels])

    valid = labels > 0
    valid_coords = coords_vox[valid]

    sagittal_slices = select_bilateral_sagittal_slices(left_vox[:, 0], right_vox[:, 0], args.n_slices)
    coronal_slices = select_slices(valid_coords[:, 1], args.n_slices)
    axial_slices = select_slices(valid_coords[:, 2], args.n_slices)

    sagittal_crop = bounding_box(valid_coords[:, [1, 2]], args.margin, (t2w.shape[1], t2w.shape[2]))
    coronal_crop = bounding_box(valid_coords[:, [0, 2]], args.margin, (t2w.shape[0], t2w.shape[2]))
    axial_crop = bounding_box(valid_coords[:, [0, 1]], args.margin, (t2w.shape[0], t2w.shape[1]))

    fig = plt.figure(figsize=(2.55 * args.n_slices, 10.6), constrained_layout=True)
    gs = fig.add_gridspec(4, args.n_slices, height_ratios=[1.0, 1.0, 1.0, 0.23])

    row_defs = [
        ("Sagittal", 0, sagittal_slices, sagittal_crop, "S"),
        ("Coronal", 1, coronal_slices, coronal_crop, "C"),
        ("Axial", 2, axial_slices, axial_crop, "A"),
    ]

    for row_index, (row_name, axis, slices, crop, prefix) in enumerate(row_defs):
        for col_index, slice_index in enumerate(slices):
            ax = fig.add_subplot(gs[row_index, col_index])
            plot_plane(
                ax=ax,
                volume=t2w,
                coords_vox=coords_vox,
                labels=labels,
                style=style,
                axis=axis,
                slice_index=slice_index,
                slab_thickness=args.slab_thickness,
                crop=crop,
                title=f"{prefix}{slice_index}",
            )
            if col_index == 0:
                ax.set_ylabel(row_name, fontsize=12)

    ax_leg = fig.add_subplot(gs[3, :])
    ax_leg.axis("off")
    handles = []
    for key, name, _count, frac in combined_proportions(left_labels, right_labels, style):
        rgba = np.array(style[key]["rgba"], dtype=np.float32) / 255.0
        handles.append(mpatches.Patch(color=rgba, label=f"{name}  {frac * 100:.1f}%"))
    ax_leg.legend(
        handles=handles,
        loc="center",
        ncol=min(4, len(handles)),
        frameon=False,
        fontsize=10,
    )

    fig.suptitle(f"Hippocampal WTA local slab views: {args.subject}", fontsize=18, y=1.01)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

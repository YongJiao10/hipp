#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.gifti import GiftiDataArray, GiftiImage, GiftiLabel, GiftiLabelTable

from lib.workbench_grid import compose_wb_grid_with_legend
from lib.workbench_scene_batch import render_scene_batch
from lib.network_colors import GLOBAL_NETWORK_STYLE_JSON, load_network_colors

REPO_ROOT = Path(__file__).resolve().parents[3]
NETWORK_STYLE_JSON = GLOBAL_NETWORK_STYLE_JSON
DEFAULT_SCENE = REPO_ROOT / "config" / "wb_locked_native_view_lateral_medial.scene"


def infer_density_from_surface(path: Path) -> str:
    m = re.search(r"_den-([A-Za-z0-9]+)_label-", path.name)
    if not m:
        raise ValueError(f"Unable to infer density from: {path}")
    return m.group(1)


def infer_spec_path(subject: str, surface_path: Path) -> Path:
    density = infer_density_from_surface(surface_path)
    spec_path = surface_path.parent / f"sub-{subject}_den-{density}_label-hipp_surfaces.spec"
    if not spec_path.exists():
        raise FileNotFoundError(f"Missing spec file: {spec_path}")
    return spec_path


def long_axis_order(surface_path: Path) -> np.ndarray:
    img = nib.load(str(surface_path))
    coords = None
    for da in img.darrays:
        if da.intent == "NIFTI_INTENT_POINTSET":
            coords = np.asarray(da.data, dtype=np.float64)
            break
    if coords is None:
        coords = np.asarray(img.darrays[0].data, dtype=np.float64)
    centered = coords - np.mean(coords, axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    score = centered @ vt[0]
    # PCA sign is arbitrary; orient PC1 so larger score points to anterior/head
    # using surface Y coordinate in space-corobl (anterior-positive in this project).
    y = coords[:, 1]
    valid = np.isfinite(score) & np.isfinite(y)
    if np.sum(valid) >= 3:
        corr = np.corrcoef(score[valid], y[valid])[0, 1]
        if np.isfinite(corr) and corr < 0:
            score = -score
    # Head -> tail ordering: anterior(high score) first.
    return np.argsort(-score, kind="stable").astype(np.int32)


def equal_vertex_long_axis_bins(order: np.ndarray, n_bins: int) -> list[np.ndarray]:
    """Split head-to-tail vertex order into consecutive, nearly equal bins.

    Earlier bins receive the extra vertices when the vertex count is not
    divisible by ``n_bins``. Each item contains vertex indices, not ranks.
    """
    return [np.asarray(vertices, dtype=np.int32) for vertices in np.array_split(order, n_bins)]


def vertex_area_weights(surface_path: Path) -> np.ndarray:
    """Per-vertex tissue area from a midthickness mesh (triangle area / 3)."""
    img = nib.load(str(surface_path))
    coords = np.asarray(img.agg_data("pointset"), dtype=np.float64)
    faces = np.asarray(img.agg_data("triangle"), dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Expected Nx3 coordinates in {surface_path}, got {coords.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected triangle faces in {surface_path}, got {faces.shape}")
    tri = coords[faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )
    weights = np.zeros(coords.shape[0], dtype=np.float64)
    for corner in range(3):
        np.add.at(weights, faces[:, corner], areas / 3.0)
    if not np.all(np.isfinite(weights)) or float(np.sum(weights)) <= 0.0:
        raise ValueError(f"Invalid vertex area weights computed from {surface_path}")
    return weights


def make_network_label_gifti(
    *,
    labels: np.ndarray,
    network_names: list[str],
    network_colors: dict[str, tuple[int, int, int, int]],
) -> GiftiImage:
    table = GiftiLabelTable()
    null_label = GiftiLabel(key=0, red=0.62, green=0.62, blue=0.62, alpha=1.0)
    null_label.label = "Null"
    table.labels.append(null_label)

    for idx, name in enumerate(network_names, start=1):
        if name not in network_colors:
            raise KeyError(f"Missing network color for {name!r}")
        rgba = network_colors[name]
        g = GiftiLabel(
            key=int(idx),
            red=float(rgba[0]) / 255.0,
            green=float(rgba[1]) / 255.0,
            blue=float(rgba[2]) / 255.0,
            alpha=float(rgba[3]) / 255.0,
        )
        g.label = str(name)
        table.labels.append(g)

    arr = GiftiDataArray(data=labels.astype(np.int32), intent="NIFTI_INTENT_LABEL", datatype="NIFTI_TYPE_INT32")
    return GiftiImage(darrays=[arr], labeltable=table)


def render_locked_grid(
    *,
    subject: str,
    scene: Path,
    outdir: Path,
    name: str,
    title: str,
    left_labels: Path,
    right_labels: Path,
    left_surface: Path,
    right_surface: Path,
    spec_path: Path,
) -> dict[str, str]:
    outdir.mkdir(parents=True, exist_ok=True)
    native_root = outdir / "native"
    native_root.mkdir(parents=True, exist_ok=True)
    render_scene_batch(
        scene=scene,
        subjects=[subject],
        outdir=native_root / "batch",
        scene_index=1,
        width=1600,
        height=1024,
        renderer="OSMesa",
        name=f"{name}_native",
        no_template_scene=True,
        left_label_template=str(left_labels),
        right_label_template=str(right_labels),
        left_surface_template=str(left_surface),
        right_surface_template=str(right_surface),
        spec_template=str(spec_path),
    )
    native_png = native_root / "batch" / f"sub-{subject}" / f"sub-{subject}_wb_{name}_native_native.png"
    final_png = outdir / f"sub-{subject}_wb_{name}_biglegend.png"
    compose_wb_grid_with_legend(
        ventral_image=native_png,
        layout="1x2",
        left_labels=left_labels,
        right_labels=right_labels,
        legend_group="label",
        title=title,
        out=final_png,
    )
    return {"native_scene_png": str(native_png), "biglegend_png": str(final_png)}


def read_surface_vertex_count(surface_path: Path) -> int:
    img = nib.load(str(surface_path))
    for da in img.darrays:
        if da.intent == "NIFTI_INTENT_POINTSET":
            return int(np.asarray(da.data).shape[0])
    return int(np.asarray(img.darrays[0].data).shape[0])


def network_color_map(
    style_path: Path = NETWORK_STYLE_JSON,
) -> dict[str, tuple[int, int, int, int]]:
    return load_network_colors(style_path)



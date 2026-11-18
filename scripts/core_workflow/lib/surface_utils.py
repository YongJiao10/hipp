#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import tempfile
from importlib.metadata import version
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.gifti import GiftiDataArray, GiftiImage, GiftiLabel, GiftiLabelTable

from lib.workbench_grid import compose_wb_grid_with_legend
from lib.network_colors import GLOBAL_NETWORK_STYLE_JSON, NETWORK_DISPLAY_NAMES, load_network_colors

REPO_ROOT = Path(__file__).resolve().parents[3]
NETWORK_STYLE_JSON = GLOBAL_NETWORK_STYLE_JSON


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


def render_hippocampal_grid(
    *,
    subject: str,
    outdir: Path,
    name: str,
    title: str,
    left_labels: Path,
    right_labels: Path,
    left_surface: Path,
    right_surface: Path,
    legend_display_names: dict[str, str] | None = None,
    surface_cache_root: Path | None = None,
    unavailable: tuple[str, ...] = (),
) -> dict[str, str]:
    """Render the selected subject surfaces and label GIFTIs with BrainSpace."""
    os.environ.setdefault("LP_NUM_THREADS", "1")
    os.environ.setdefault("VTK_SMP_MAX_THREADS", "1")
    from PIL import Image

    outdir.mkdir(parents=True, exist_ok=True)
    inputs = tuple(Path(p).resolve(strict=True) for p in (left_surface, right_surface, left_labels, right_labels))
    native_png = outdir / f"sub-{subject}_{name}_surface.png"
    reuse_surface = False
    if surface_cache_root is not None:
        from vtkmodules.vtkCommonCore import vtkVersion

        runtime = (
            f"brainspace-{version('brainspace')}_vtk-{vtkVersion.GetVTKVersion()}"
            f"_nibabel-{version('nibabel')}_matplotlib-{version('matplotlib')}"
            f"_numpy-{version('numpy')}"
        )
        cache_dir = Path(surface_cache_root) / runtime
        for part, path in zip(("left_surface", "right_surface", "left_labels", "right_labels"), inputs, strict=True):
            cache_dir = cache_dir / part / path.relative_to(path.anchor)
        cache_dir.mkdir(parents=True, exist_ok=True)
        native_png = cache_dir / "surface.png"
        if native_png.is_file():
            newest_source = max(
                max(stat.st_mtime_ns, stat.st_ctime_ns)
                for stat in (path.stat() for path in (*inputs, Path(__file__).resolve()))
            )
            if native_png.stat().st_mtime_ns >= newest_source:
                try:
                    with Image.open(native_png) as image:
                        reuse_surface = image.format == "PNG" and image.size == (3200, 2048)
                        image.verify()
                except (OSError, ValueError, SyntaxError):
                    reuse_surface = False
    if not reuse_surface:
        _render_hippocampal_label_surface(inputs, native_png)
    final_png = outdir / f"sub-{subject}_wb_{name}_biglegend.png"
    compose_wb_grid_with_legend(
        ventral_image=native_png,
        layout="1x2",
        left_labels=left_labels,
        right_labels=right_labels,
        legend_group="label",
        legend_display_names=legend_display_names or NETWORK_DISPLAY_NAMES,
        title=title,
        out=final_png,
        scale=2,
        adaptive_legend=True,
        unavailable=unavailable,
    )
    return {"surface_png": str(native_png), "biglegend_png": str(final_png)}


def _render_hippocampal_label_surface(inputs: tuple[Path, ...], native_png: Path) -> None:
    from brainspace.mesh.mesh_io import read_surface
    from brainspace.plotting import plot_surf
    from matplotlib.colors import ListedColormap

    left_surface, right_surface, left_labels, right_labels = inputs
    surfaces = {}
    tables = {}
    for hemi, surface_path, label_path in (
        ("L", left_surface, left_labels), ("R", right_surface, right_labels)
    ):
        surface = read_surface(str(surface_path))
        label_image = nib.load(str(label_path))
        values = np.asarray(label_image.darrays[0].data)
        if values.ndim != 1 or values.size != surface.n_points:
            raise ValueError(f"Label/surface vertex mismatch: {label_path} and {surface_path}")
        table = {int(label.key): tuple(float(c) for c in label.rgba)
                 for label in label_image.labeltable.labels}
        missing = set(np.unique(values).astype(int)) - set(table)
        if missing:
            raise ValueError(f"Label table in {label_path} is missing keys {sorted(missing)}")
        surface.append_array(values.astype(np.float32), name="network", at="point")
        surfaces[hemi], tables[hemi] = surface, table

    min_key = min(min(table) for table in tables.values())
    max_key = max(max(table) for table in tables.values())
    cmaps = {
        hemi: ListedColormap([tables[hemi].get(key, (0, 0, 0, 0))
                              for key in range(min_key, max_key + 1)])
        for hemi in ("L", "R")
    }
    plotter = plot_surf(
        surfaces, [["L", "R"], ["L", "R"]], array_name="network",
        view=[["dorsal", "dorsal"], ["ventral", "ventral"]],
        cmap=[[cmaps["L"], cmaps["R"]], [cmaps["L"], cmaps["R"]]],
        color_range=(min_key - 0.5, max_key + 0.5), zoom=1.1,
        size=(1600, 1024), background=(1, 1, 1), offscreen=True,
        mapper__interpolateScalarsBeforeMapping=False, return_plotter=True,
    )
    try:
        with tempfile.NamedTemporaryFile(dir=native_png.parent, suffix=".png", delete=False) as temporary:
            temporary_path = Path(temporary.name)
        plotter.screenshot(str(temporary_path), transparent_bg=False, scale=(2, 2))
        os.replace(temporary_path, native_png)
    finally:
        plotter.close()


def render_hippocampal_metric(
    *, left_surface: Path, right_surface: Path,
    left_values: np.ndarray, right_values: np.ndarray,
    output_path: Path, cmap: str, color_range: tuple[float, float],
) -> Path:
    """Render continuous vertex values on the same four-view hippocampal layout."""
    os.environ.setdefault("LP_NUM_THREADS", "1")
    os.environ.setdefault("VTK_SMP_MAX_THREADS", "1")
    from brainspace.mesh.mesh_io import read_surface
    from brainspace.plotting import plot_surf

    surfaces = {}
    for hemi, path, values in (("L", left_surface, left_values), ("R", right_surface, right_values)):
        surface = read_surface(str(path))
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 1 or array.size != surface.n_points:
            raise ValueError(f"Metric/surface vertex mismatch: {path}")
        surface.append_array(array, name="metric", at="point")
        surfaces[hemi] = surface
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotter = plot_surf(
        surfaces, [["L", "R"], ["L", "R"]], array_name="metric",
        view=[["dorsal", "dorsal"], ["ventral", "ventral"]], cmap=cmap,
        color_range=color_range, zoom=1.1, size=(1600, 1024),
        background=(1, 1, 1), offscreen=True, return_plotter=True,
    )
    try:
        plotter.screenshot(str(output_path), transparent_bg=False, scale=(2, 2))
    finally:
        plotter.close()
    return output_path


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



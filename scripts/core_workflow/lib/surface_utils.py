#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
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


def unfolded_anterior_coordinate(unfold_surface: Path, folded_surface: Path) -> np.ndarray:
    """Per-vertex anterior-posterior coordinate of the HippUnfold unfolded sheet (larger = nearer the head).

    The folded surface's first principal axis compresses the curled head: patches that are clearly
    anterior on the unfolded sheet can share a folded position. Cluster numbering therefore reads
    the unfolded y axis, which must be planar, match the folded mesh vertex for vertex and point
    anterior on the folded surface.
    """
    unfolded = np.asarray(nib.load(str(unfold_surface)).agg_data("pointset"), dtype=np.float64)
    folded = np.asarray(nib.load(str(folded_surface)).agg_data("pointset"), dtype=np.float64)
    if unfolded.shape != folded.shape:
        raise ValueError(
            f"Unfolded/folded vertex mismatch: {unfold_surface} {unfolded.shape} vs {folded_surface} {folded.shape}"
        )
    if np.ptp(unfolded[:, 2]) > 1e-3:
        raise ValueError(f"Unfolded surface is not planar: {unfold_surface}")
    corr = float(np.corrcoef(unfolded[:, 1], folded[:, 1])[0, 1])
    if not corr > 0.5:
        raise ValueError(
            f"Unfolded y axis does not track the folded anterior axis (r={corr:.2f}): {unfold_surface} vs {folded_surface}"
        )
    return unfolded[:, 1].copy()


def template_anterior_coordinate(spec_path: Path, folded_surface: Path, hemi: str) -> np.ndarray:
    """Unfolded anterior coordinate of a template mesh, from the unfold surface its spec declares."""
    assets = resolve_unfolded_assets(Path(spec_path), density=infer_density_from_surface(Path(folded_surface)))
    return unfolded_anterior_coordinate(assets.unfold_surface[str(hemi).upper()], Path(folded_surface))


def cluster_ids_head_to_tail(labels: np.ndarray, anterior_coordinate: np.ndarray) -> list[int]:
    """Positive cluster ids ordered head to tail by the mean unfolded anterior coordinate of their vertices."""
    labels = np.asarray(labels, dtype=np.int64)
    coordinate = np.asarray(anterior_coordinate, dtype=np.float64)
    if labels.ndim != 1 or coordinate.shape != labels.shape:
        raise ValueError(f"Labels/coordinate shape mismatch: labels={labels.shape}, coordinate={coordinate.shape}")
    ids = sorted(int(v) for v in np.unique(labels) if int(v) > 0)
    return sorted(ids, key=lambda cid: (-float(np.mean(coordinate[labels == cid])), cid))


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
    unfolded: UnfoldedHippocampusAssets | None = None,
) -> dict[str, str]:
    """Render the selected subject surfaces and label GIFTIs with BrainSpace.

    When ``unfolded`` is given (caller opt-in), also write ``foldunfold_png`` with the
    unfolded maps and subfield boundaries beside the folded views.
    """
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
    )
    result = {"surface_png": str(native_png), "biglegend_png": str(final_png)}
    if unfolded is not None:
        result["foldunfold_png"] = str(compose_folded_unfolded_panel(
            native_png=native_png,
            biglegend_png=final_png,
            left_labels=inputs[2],
            right_labels=inputs[3],
            left_surface=inputs[0],
            right_surface=inputs[1],
            unfolded=unfolded,
            out=outdir / f"sub-{subject}_{name}_foldunfold.png",
        ))
    return result


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


UNFOLDED_SUBFIELD_SHORT_NAMES = {"Subiculum": "Sub"}
# Unfolded map height as a fraction of the folded panel (about one folded view).
UNFOLDED_HEIGHT_FRACTION = 0.42
# Legend column width (gridspec ratio units) and legend scale in the fold/unfold panel.
FOLDUNFOLD_LEGEND_COLUMN_RATIO = 0.42
FOLDUNFOLD_LEGEND_INCH_PER_PX = 0.0019


@dataclass(frozen=True)
class UnfoldedHippocampusAssets:
    """HippUnfold unfolded-space surface and subfield labels declared by a template spec."""

    spec_path: Path
    density: str
    subfield_atlas: str
    unfold_surface: dict[str, Path]
    subfield_labels: dict[str, Path]

    def provenance(self) -> dict[str, object]:
        return {
            "spec_path": str(self.spec_path),
            "density": self.density,
            "subfield_atlas": self.subfield_atlas,
            "unfold_surface": {h: str(p) for h, p in self.unfold_surface.items()},
            "subfield_labels": {h: str(p) for h, p in self.subfield_labels.items()},
        }


def resolve_unfolded_assets(spec_path: Path, *, density: str) -> UnfoldedHippocampusAssets:
    """Select the unfolded midthickness surface and subfield label GIFTI listed in a HippUnfold spec.

    Exactly one unfold-space hippocampal midthickness SURFACE and one subfield LABEL file
    must be declared per hemisphere at ``density``; anything else is an error.
    """
    import xml.etree.ElementTree as ET

    spec_path = Path(spec_path).resolve(strict=True)
    structures = {"L": "HippocampusLeft", "R": "HippocampusRight"}
    surface_re = {h: re.compile(rf"_hemi-{h}_space-unfold_den-{re.escape(density)}_label-hipp_midthickness\.surf\.gii$") for h in structures}
    label_re = {h: re.compile(rf"_hemi-{h}_den-{re.escape(density)}_label-hipp_atlas-([A-Za-z0-9]+)_subfields\.label\.gii$") for h in structures}
    entries = [
        (str(node.get("Structure")), str(node.get("DataFileType")), (node.text or "").strip())
        for node in ET.parse(spec_path).getroot().iter("DataFile")
    ]
    surfaces: dict[str, Path] = {}
    labels: dict[str, Path] = {}
    atlases: dict[str, str] = {}
    for hemi, structure in structures.items():
        surf_hits = [name for st, kind, name in entries if st == structure and kind == "SURFACE" and surface_re[hemi].search(name)]
        label_hits = [name for st, kind, name in entries if st == structure and kind == "LABEL" and label_re[hemi].search(name)]
        for what, hits in (("unfold-space hipp midthickness SURFACE", surf_hits), ("hipp subfield LABEL", label_hits)):
            if len(hits) != 1:
                raise ValueError(
                    f"Template spec must declare exactly one hemi-{hemi} den-{density} {what}; "
                    f"found {len(hits)} {hits} in {spec_path}"
                )
        surfaces[hemi] = (spec_path.parent / surf_hits[0]).resolve(strict=True)
        labels[hemi] = (spec_path.parent / label_hits[0]).resolve(strict=True)
        atlases[hemi] = label_re[hemi].search(label_hits[0]).group(1)
    if atlases["L"] != atlases["R"]:
        raise ValueError(f"Left/right subfield atlases disagree in {spec_path}: {atlases}")
    return UnfoldedHippocampusAssets(
        spec_path=spec_path,
        density=density,
        subfield_atlas=atlases["L"],
        unfold_surface=surfaces,
        subfield_labels=labels,
    )


def unfolded_grid(unfold_surface: Path, folded_surface: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Map vertices to (row, col) cells of the regular unfolded grid.

    Rows run anterior -> posterior and columns follow increasing unfolded x. The
    unfolded mesh must be planar and fully populate a regular grid, and its y axis
    must point anterior on the folded surface it is displayed with.
    """
    xyz = np.asarray(nib.load(str(unfold_surface)).agg_data("pointset"), dtype=np.float64)
    folded = np.asarray(nib.load(str(folded_surface)).agg_data("pointset"), dtype=np.float64)
    if xyz.shape != folded.shape:
        raise ValueError(f"Unfolded/folded vertex mismatch: {unfold_surface} {xyz.shape} vs {folded_surface} {folded.shape}")
    if np.ptp(xyz[:, 2]) > 1e-3:
        raise ValueError(f"Unfolded surface is not planar: {unfold_surface}")
    xs, col = np.unique(np.round(xyz[:, 0], 4), return_inverse=True)
    ys, row_up = np.unique(np.round(xyz[:, 1], 4), return_inverse=True)
    shape = (ys.size, xs.size)
    if ys.size * xs.size != xyz.shape[0] or np.unique(row_up * xs.size + col).size != xyz.shape[0]:
        raise ValueError(f"Unfolded surface is not a fully populated regular grid {shape}: {unfold_surface}")
    corr = float(np.corrcoef(xyz[:, 1], folded[:, 1])[0, 1])
    if not corr > 0.5:
        raise ValueError(
            f"Unfolded y axis does not track the folded anterior axis (r={corr:.2f}): {unfold_surface} vs {folded_surface}"
        )
    row = (ys.size - 1) - row_up  # anterior (largest y) first
    return row.astype(int), col.astype(int), shape


def _draw_unfolded_map(ax, *, hemi: str, labels_path: Path, folded_surface: Path, assets: UnfoldedHippocampusAssets) -> None:
    import matplotlib.patheffects as pe
    from matplotlib import transforms
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap

    row, col, shape = unfolded_grid(assets.unfold_surface[hemi], folded_surface)
    func_img = nib.load(str(labels_path))
    func = np.asarray(func_img.darrays[0].data).astype(int)
    sub_img = nib.load(str(assets.subfield_labels[hemi]))
    sub = np.asarray(sub_img.darrays[0].data).astype(int)
    for name, values in (("functional labels", func), ("subfield labels", sub)):
        if values.shape != row.shape:
            raise ValueError(f"Unfolded {name} vertex mismatch for hemi-{hemi}: {values.shape} vs {row.shape}")
    func_table = {int(lab.key): tuple(float(c) for c in lab.rgba) for lab in func_img.labeltable.labels}
    sub_table = {int(lab.key): (tuple(float(c) for c in lab.rgba)[:3], str(lab.label)) for lab in sub_img.labeltable.labels}
    for name, values, table in (("functional", func, func_table), ("subfield", sub, sub_table)):
        missing = set(np.unique(values).tolist()) - set(table)
        if missing:
            raise ValueError(f"Unfolded {name} label table for hemi-{hemi} is missing keys {sorted(missing)}")

    keys = sorted(func_table)
    func_grid = np.empty(shape, dtype=int)
    func_grid[row, col] = np.searchsorted(keys, func)
    sub_grid = np.empty(shape, dtype=int)
    sub_grid[row, col] = sub
    n_rows, n_cols = shape
    ax.imshow(
        func_grid, cmap=ListedColormap([func_table[k] for k in keys]), vmin=-0.5, vmax=len(keys) - 0.5,
        interpolation="nearest", extent=(0, n_cols, n_rows, 0),
    )

    # One line per subfield boundary, coloured by the subfield with the larger atlas key
    # (Sub -> CA4 order in the HippUnfold atlases), i.e. the subfield the line opens.
    segments: list[list[tuple[float, float]]] = []
    colours: list[tuple[float, float, float]] = []
    for r in range(n_rows):
        for c in range(n_cols):
            here = sub_grid[r, c]
            if c + 1 < n_cols and sub_grid[r, c + 1] != here:
                segments.append([(c + 1, r), (c + 1, r + 1)])
                colours.append(sub_table[max(here, sub_grid[r, c + 1])][0])
            if r + 1 < n_rows and sub_grid[r + 1, c] != here:
                segments.append([(c, r + 1), (c + 1, r + 1)])
                colours.append(sub_table[max(here, sub_grid[r + 1, c])][0])
    # Opaque halo first, then the coloured lines, so segment joints do not darken.
    ax.add_collection(LineCollection(segments, colors="#5a5a5a", linewidths=2.4, capstyle="projecting"))
    ax.add_collection(LineCollection(segments, colors=colours, linewidths=1.4, capstyle="projecting"))

    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#999999")
        spine.set_linewidth(0.6)

    # Annotation offsets are in points so text size and spacing do not scale with the map.
    def above(points: float):
        return transforms.offset_copy(ax.transData, fig=ax.figure, y=points, units="points")

    present = sorted(int(k) for k in np.unique(sub))
    centres = {k: float(np.mean(np.nonzero(sub_grid == k)[1])) + 0.5 for k in present}
    ordered = sorted(present, key=lambda k: centres[k])
    names = {k: UNFOLDED_SUBFIELD_SHORT_NAMES.get(sub_table[k][1], sub_table[k][1]) for k in present}
    # One row of labels, evenly spaced; the row may extend past the map edges so that
    # names keep their size when the map is narrow.
    ax.apply_aspect()
    axes_width_pt = ax.get_position().width * ax.figure.get_figwidth() * 72.0
    label_fs = 7
    widest_pt = max(len(name) for name in names.values()) * 0.68 * label_fs
    spread = max(1.0, len(ordered) * widest_pt * 1.2 / axes_width_pt) * n_cols
    slots = n_cols / 2.0 + (np.arange(len(ordered)) + 0.5 - len(ordered) / 2.0) * spread / len(ordered)
    for key, slot in zip(ordered, slots):
        colour = sub_table[key][0]
        label = ax.annotate(
            names[key], xy=(centres[key], 0.0), xycoords="data",
            xytext=(slot, 0.0), textcoords=above(10.5), ha="center", va="bottom",
            fontsize=label_fs, fontweight="bold", color=colour, annotation_clip=False,
            arrowprops={"arrowstyle": "-", "color": colour, "lw": 0.8, "shrinkA": 1.0, "shrinkB": 0.0},
        )
        label.arrow_patch.set_path_effects(
            [pe.Stroke(linewidth=1.6, foreground=(0.2, 0.2, 0.2, 0.4)), pe.Normal()]
        )

    side = transforms.offset_copy(ax.transData, fig=ax.figure, x=-3.0 if hemi == "L" else 3.0, units="points")
    x_text, ha = (0.0, "right") if hemi == "L" else (n_cols, "left")
    ax.text(x_text, 0.0, "Ant", transform=side, ha=ha, va="top", fontsize=8, color="#333333")
    ax.text(x_text, n_rows, "Post", transform=side, ha=ha, va="bottom", fontsize=8, color="#333333")
    ax.set_title(f"{hemi} unfolded", fontsize=9, pad=22)


def _white_margin(image, pad: int = 6):
    from PIL import Image

    rgb = Image.new("RGB", image.size, (255, 255, 255))
    rgba = image.convert("RGBA")
    rgb.paste(rgba, mask=rgba.split()[-1])
    arr = np.asarray(rgb).astype(int)
    ys, xs = np.nonzero((arr < 245).any(axis=2))
    if ys.size == 0:
        raise ValueError("Cannot crop an empty panel image")
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    out = Image.new("RGB", (box[2] - box[0] + 2 * pad, box[3] - box[1] + 2 * pad), (255, 255, 255))
    out.paste(rgb.crop(box), (pad, pad))
    return out


def compose_folded_unfolded_panel(
    *,
    native_png: Path,
    biglegend_png: Path,
    left_labels: Path,
    right_labels: Path,
    left_surface: Path,
    right_surface: Path,
    unfolded: UnfoldedHippocampusAssets,
    out: Path,
) -> Path:
    """Lay out [L unfolded | L folded | R folded | R unfolded | legend] as one PNG."""
    import matplotlib.pyplot as plt
    from PIL import Image

    from lib.workbench_grid import build_grid_canvas, split_native_hemi_panels

    native = Image.open(native_png).convert("RGBA")
    folded = {h: _white_margin(p) for h, p in split_native_hemi_panels(native).items()}
    big = Image.open(biglegend_png).convert("RGBA")
    grid_w = build_grid_canvas(native, None, "1x2", scale=2).width
    if big.width <= grid_w:
        raise ValueError(f"Legend region not found in {biglegend_png}")
    legend = _white_margin(big.crop((grid_w, 0, big.width, big.height)))

    fold_w = max(p.width / p.height for p in folded.values())
    unf_w = UNFOLDED_HEIGHT_FRACTION * 0.5  # grid aspect is 1:2 (PD x AP)
    # Fixed legend column and legend scale: every panel gets the same geometry whatever
    # its legend holds, so stacked overview rows line up and legend text keeps one size.
    ratios = [unf_w, fold_w, fold_w, unf_w, 0.07, FOLDUNFOLD_LEGEND_COLUMN_RATIO]  # 0.07: gap
    fig = plt.figure(figsize=(1.25 + 4.6 * sum(ratios), 4.6), dpi=200)
    gs = fig.add_gridspec(1, 6, width_ratios=ratios, wspace=0.10, left=0.045, right=0.99, top=0.90, bottom=0.02)
    try:
        for idx, hemi, labels, surface in ((0, "L", left_labels, left_surface), (3, "R", right_labels, right_surface)):
            _draw_unfolded_map(fig.add_subplot(gs[0, idx]), hemi=hemi, labels_path=labels,
                               folded_surface=surface, assets=unfolded)
        for idx, hemi in ((1, "L"), (2, "R")):
            ax = fig.add_subplot(gs[0, idx])
            ax.imshow(folded[hemi])
            ax.set_title(hemi, fontsize=15)
            ax.axis("off")
        column = gs[0, 5].get_position(fig)
        fig_w, fig_h = fig.get_size_inches()
        inch_per_px = min(FOLDUNFOLD_LEGEND_INCH_PER_PX,
                          column.width * fig_w / legend.width, column.height * fig_h / legend.height)
        leg_w, leg_h = legend.width * inch_per_px / fig_w, legend.height * inch_per_px / fig_h
        ax_leg = fig.add_axes((column.x0, column.y0 + (column.height - leg_h) / 2.0, leg_w, leg_h))
        ax_leg.imshow(legend, aspect="auto")
        ax_leg.axis("off")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, facecolor="white")
    finally:
        plt.close(fig)
    return out


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



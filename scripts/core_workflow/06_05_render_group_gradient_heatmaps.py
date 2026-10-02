#!/usr/bin/env python3
from __future__ import annotations

"""
DEPRECATED: Locked-scene composite heatmap output is retired.

Process Step Description:
Render group-level gradient heatmaps. Generates heatmap visualizations of functional gradients based on group-level data to intuitively display gradient patterns in a composite 2x3 grid image layout.

Main Flow Path:
- 9th step in the numbered pipeline sequence (first stage of presentation output generation, run in parallel with 06_01, 06_02, 06_03, 06_04, 06_06, 06_07):
  01_01_run_hippunfold_t1w_local.py -> 01_02_check_hippunfold_outputs.py -> 02_01_prepare_group_inputs.py -> 02_02_build_qc_manifest.py -> 03_01/03_02 reference building -> 04_01/04_02 FC materialization -> 05_01 gradient computation -> 05_02 group atlas parcellation -> 06_05_render_group_gradient_heatmaps.py (this step) -> 07_01 group matrix overview.

Inputs:
- Command-line arguments:
  - --group-root: Root directory containing group atlas results (default: `outputs_hcp/work/group_atlas`).
  - --scene: Connectome Workbench template scene file (default: `config/wb_locked_native_view_lateral_medial.scene`).
  - --output-png: Output path for the composite 2x3 heatmap figure (default: `outputs_hcp/present/gradients/kong2022_gradient-spectral_group_gradients_heatmap_routea_2mm4mm_g1g2g3.png`).
  - --output-manifest: Output path for the manifest JSON (default: `outputs_hcp/present/gradients/kong2022_gradient-spectral_group_gradients_heatmap_routea_2mm4mm_g1g2g3_manifest.json`).
  - --runtime-root: Root for intermediate files and temporary scene directories (default: `outputs_hcp/present/gradients/group_gradient_heatmaps/runtime/routea_full`).
  - --method: Parcellation method tag (default: "gradient-spectral").
  - --atlas: Cortical atlas name (default: "kong2022").
  - --smoothings: Smoothing values to process (default: ["2mm", "4mm"]).
  - --hemis: Hemispheres to process (default: ["L", "R"]).
  - --components: Comma-separated list of gradient components to plot (default: "G1,G2,G3").
  - --coverage-tag: Coverage threshold directory name tag (default: "cov_0p80").
  - --renderer: Workbench window renderer choice (default: "OSMesa").
  - --width: Rendered window width (default: 1600).
  - --height: Rendered window height (default: 1024).
  - --palette: Color palette for gradients (default: "magma").
  - --dry-run: Enable dry run to skip actual Workbench command invocation.
- Physical inputs:
  - Group atlas run subdirectories: `{group_root}/{method}/smooth-{smoothing}_hemi-{hemi}_cov-{threshold}/`
  - Group manifest: `group_atlas_manifest.json` under each run directory
  - Gradient data files (`*.npy` or `*.gii`) referenced in the manifest files.

Outputs:
- File outputs:
  - A composite 2x3 heatmap grid image showing gradient columns and smoothing rows: `{output_png}`
  - Manifest file capturing run configuration and output details: `{output_manifest}`
  - Intermediate single-hemisphere/single-component PNG renderings under `{runtime_root}/`.
"""
import argparse
import json
import sys
import math
import re
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from nibabel.gifti import GiftiDataArray, GiftiImage

from lib.output_layout import DEFAULT_GRADIENTS_DIR, DEFAULT_ROUTEA_GROUP_ATLAS_ROOT
from lib.workflow_utils import write_json
from lib.workbench import run_workbench
from lib.io_utils import load_json
from lib.surface_utils import read_surface_vertex_count
from lib.workbench_grid import load_font, trim_black
from lib.workbench_scene_batch import (
    detect_template_subject,
    replace_exact_ref,
    replace_subject_text,
    validate_scene_paths,
    absolutize_scene_paths,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GROUP_ROOT = DEFAULT_ROUTEA_GROUP_ATLAS_ROOT
DEFAULT_SCENE = REPO_ROOT / "config" / "wb_locked_native_view_lateral_medial.scene"
DEFAULT_OUT_PNG = DEFAULT_GRADIENTS_DIR / "kong2022_gradient-spectral_group_gradients_heatmap_routea_2mm4mm_g1g2g3.png"
DEFAULT_OUT_MANIFEST = DEFAULT_GRADIENTS_DIR / "kong2022_gradient-spectral_group_gradients_heatmap_routea_2mm4mm_g1g2g3_manifest.json"
DEFAULT_RUNTIME_ROOT = DEFAULT_GRADIENTS_DIR / "group_gradient_heatmaps" / "runtime" / "routea_full"

METHOD = "gradient-spectral"
ATLAS = "kong2022"
SMOOTHINGS = ["2mm", "4mm"]
HEMIS = ["L", "R"]
COMPONENTS = ["G1", "G2", "G3"]
COVERAGE_TAG = "cov_0p80"
ARCHIVE_TOKENS = ("/archive/", "/_archive/")


@dataclass(frozen=True)
class ComboData:
    smoothing: str
    hemi: str
    run_dir: Path
    manifest_path: Path
    group_manifest: dict[str, Any]
    active_path: Path
    support_mask_path: Path
    template_surface_path: Path
    template_spec_path: Path
    template_subject: str
    active: np.ndarray
    support_mask: np.ndarray
    vertex_count: int
    support_count: int
    full_components: np.ndarray  # [vertex, component]
    gradient_summary_path: Path
    component_ev_percentages: dict[str, float]


@dataclass(frozen=True)
class RenderMeta:
    panel_png: Path
    left_shape: Path
    right_shape: Path
    selected_colorbar_metric: str | None
    colorbar_crop_box: tuple[int, int, int, int] | None


def expected_output_name() -> str:
    return "kong2022_gradient-spectral_group_gradients_heatmap_routea_2mm4mm_g1g2g3.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render 2x3 locked-scene group gradient heatmaps for RouteA full outputs. "
            "Columns: G1/G2/G3. Rows: 2mm/4mm."
        )
    )
    parser.add_argument("--group-root", default=str(DEFAULT_GROUP_ROOT))
    parser.add_argument("--scene", default=str(DEFAULT_SCENE))
    parser.add_argument("--output-png", default=str(DEFAULT_OUT_PNG))
    parser.add_argument("--output-manifest", default=str(DEFAULT_OUT_MANIFEST))
    parser.add_argument("--runtime-root", default=str(DEFAULT_RUNTIME_ROOT))
    parser.add_argument("--method", default=METHOD)
    parser.add_argument("--atlas", default=ATLAS)
    parser.add_argument("--smoothings", nargs="+", default=SMOOTHINGS)
    parser.add_argument("--hemis", nargs="+", default=HEMIS)
    parser.add_argument("--components", default="G1,G2,G3")
    parser.add_argument("--coverage-tag", default=COVERAGE_TAG)
    parser.add_argument("--renderer", default="OSMesa")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--palette", default="magma")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_workbench_palette_name(requested_palette: str) -> str:
    req = str(requested_palette).strip()
    if not req:
        raise ValueError("Palette must not be empty")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", req):
        raise ValueError(f"Invalid Workbench palette token: {req!r}")
    return req


def validate_manifest_path(manifest_path: Path, group_root: Path) -> None:
    manifest_resolved = manifest_path.resolve()
    group_root_resolved = group_root.resolve()
    text = str(manifest_resolved)
    if any(tok in text for tok in ARCHIVE_TOKENS):
        raise RuntimeError(f"Archive path is forbidden for this renderer: {manifest_resolved}")
    try:
        manifest_resolved.relative_to(group_root_resolved)
    except ValueError as exc:
        raise RuntimeError(
            f"Manifest path must stay within group root.\nmanifest={manifest_resolved}\ngroup_root={group_root_resolved}"
        ) from exc


def component_index_from_name(name: str) -> int:
    m = re.fullmatch(r"[Gg](\d+)", name.strip())
    if not m:
        raise ValueError(f"Invalid component token: {name}")
    idx = int(m.group(1))
    if idx < 1:
        raise ValueError(f"Component index must be >=1: {name}")
    return idx - 1


def reconstruct_full_components(active: np.ndarray, support_mask: np.ndarray, vertex_count: int) -> tuple[np.ndarray, int]:
    if active.ndim != 2:
        raise ValueError(f"group_gradient_features_active.npy must be 2D, got shape={active.shape}")
    mask = np.asarray(support_mask).astype(bool).reshape(-1)
    if mask.shape[0] != vertex_count:
        raise ValueError(f"Support mask length mismatch: mask={mask.shape[0]} vertex_count={vertex_count}")
    support_count = int(np.sum(mask))
    if active.shape[0] != support_count:
        raise ValueError(
            f"Active/support mismatch: active rows={active.shape[0]} support_count={support_count} vertex_count={vertex_count}"
        )
    full = np.full((vertex_count, active.shape[1]), np.nan, dtype=np.float32)
    full[mask, :] = active.astype(np.float32)
    return full, support_count


def compute_component_scales(full_by_key: dict[tuple[str, str], np.ndarray], component_names: list[str]) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for comp_name in component_names:
        comp_idx = component_index_from_name(comp_name)
        vals: list[np.ndarray] = []
        for arr in full_by_key.values():
            if comp_idx >= arr.shape[1]:
                raise ValueError(f"Component {comp_name} out of bounds for array with {arr.shape[1]} columns")
            vec = arr[:, comp_idx]
            finite = vec[np.isfinite(vec)]
            if finite.size > 0:
                vals.append(finite)
        if not vals:
            raise ValueError(f"No finite values found for component {comp_name}")
        merged = np.concatenate(vals, axis=0)
        max_abs = float(np.max(np.abs(merged)))
        if not np.isfinite(max_abs) or max_abs <= 0.0:
            max_abs = 1e-6
        out[comp_name] = (-max_abs, max_abs)
    return out


def component_ev_percentages(summary: dict[str, Any], component_names: list[str]) -> dict[str, float]:
    gradient_summary = summary.get("gradient", summary)
    raw_eigvals = gradient_summary.get("gradient_eigenvalues")
    if raw_eigvals is None:
        raise ValueError("Gradient summary is missing gradient_eigenvalues")
    eigvals = np.asarray(raw_eigvals, dtype=np.float64).reshape(-1)
    if eigvals.size < 1:
        raise ValueError(f"Expected at least one eigenvalue, got {eigvals.size}")
    denom = float(np.sum(eigvals))
    if not np.isfinite(denom) or denom <= 0.0:
        raise ValueError(f"Eigenvalue sum must be positive, got {denom}")

    out: dict[str, float] = {}
    for comp_name in component_names:
        comp_idx = component_index_from_name(comp_name)
        if comp_idx >= eigvals.size:
            raise ValueError(f"Component {comp_name} out of bounds for {eigvals.size} eigenvalues")
        out[comp_name] = float(eigvals[comp_idx] / denom * 100.0)
    return out


def compute_panel_ev_percentages(
    component_ev_by_combo: dict[tuple[str, str], dict[str, float]],
    smoothings: list[str],
    hemis: list[str],
    components: list[str],
) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for smoothing in smoothings:
        for comp in components:
            vals = [
                float(component_ev_by_combo[(smoothing, hemi)][comp])
                for hemi in hemis
                if (smoothing, hemi) in component_ev_by_combo and comp in component_ev_by_combo[(smoothing, hemi)]
            ]
            if not vals:
                raise ValueError(f"No EV percentages available for {smoothing} {comp}")
            out[(smoothing, comp)] = float(np.mean(vals))
    return out


def format_cell_label(component: str, ev_percent: float) -> str:
    return f"{component} EV={float(ev_percent):.1f}%"


def save_shape_gii(values: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    da = GiftiDataArray(
        data=np.asarray(values, dtype=np.float32),
        intent="NIFTI_INTENT_NONE",
        datatype="NIFTI_TYPE_FLOAT32",
    )
    nib.save(GiftiImage(darrays=[da]), str(out_path))


def apply_metric_palette(metric_path: Path, palette_name: str, vmin: float, vmax: float) -> None:
    neg_min = float(min(vmin, 0.0))
    pos_max = float(max(vmax, 0.0))
    if abs(neg_min) < 1e-12:
        neg_min = -1e-6
    if abs(pos_max) < 1e-12:
        pos_max = 1e-6
    run_workbench(
        [
            "-metric-palette",
            str(metric_path),
            "MODE_USER_SCALE",
            "-palette-name",
            str(palette_name),
            "-disp-pos",
            "true",
            "-disp-neg",
            "true",
            "-disp-zero",
            "true",
            "-interpolate",
            "true",
            "-normalization",
            "NORMALIZATION_ALL_MAP_DATA",
            "-pos-user",
            "0",
            str(pos_max),
            "-neg-user",
            str(neg_min),
            "0",
        ]
    )


def verify_workbench_palette_supported(palette_name: str, runtime_root: Path) -> None:
    probe_metric = runtime_root / "palette_probe" / f"probe_{palette_name}.shape.gii"
    save_shape_gii(np.asarray([-1.0, 0.0, 1.0], dtype=np.float32), probe_metric)
    try:
        apply_metric_palette(
            metric_path=probe_metric,
            palette_name=palette_name,
            vmin=-1.0,
            vmax=1.0,
        )
    except RuntimeError as exc:
        raise ValueError(
            f"Unsupported Workbench palette for wb_command -metric-palette: {palette_name!r}"
        ) from exc





def find_first_ref(root: ET.Element, pattern: str) -> str:
    rgx = re.compile(pattern)
    for elem in root.iter():
        text = (elem.text or "").strip()
        if text and rgx.search(text):
            return text
    raise RuntimeError(f"Could not find scene reference for pattern: {pattern}")


def find_corobl_surface_refs(root: ET.Element) -> dict[str, str]:
    refs: dict[str, str] = {}
    for elem in root.iter():
        text = (elem.text or "").strip()
        if "space-corobl" not in text or "midthickness.surf.gii" not in text:
            continue
        if "hemi-L" in text and "L" not in refs:
            refs["L"] = text
        if "hemi-R" in text and "R" not in refs:
            refs["R"] = text
    if "L" not in refs or "R" not in refs:
        raise RuntimeError("Could not find current corobl midthickness surface refs in scene")
    return refs


def find_gyrification_refs(root: ET.Element) -> dict[str, str]:
    refs: dict[str, str] = {}
    for elem in root.iter():
        text = (elem.text or "").strip()
        if "space-corobl" not in text or "_gyrification.shape.gii" not in text:
            continue
        if "hemi-L" in text and "L" not in refs:
            refs["L"] = text
        if "hemi-R" in text and "R" not in refs:
            refs["R"] = text
    if "L" not in refs or "R" not in refs:
        raise RuntimeError("Could not find current gyrification shape refs in scene")
    return refs


def disable_label_overlays(root: ET.Element) -> None:
    parent_map: dict[int, ET.Element] = {}
    for elem in root.iter():
        for child in elem:
            parent_map[id(child)] = elem
    for obj in root.iter("Object"):
        if obj.attrib.get("Name") != "m_enabled":
            continue
        parent = parent_map.get(id(obj))
        if parent is None:
            continue
        for sib in list(parent):
            txt = (sib.text or "").strip()
            if "group_gradient" in txt and txt.endswith(".label.gii"):
                continue
            if ".label.gii" in txt or ".dlabel.nii" in txt:
                obj.text = "false"
                break


def inject_fixed_palette(root: ET.Element, vmin: float, vmax: float, palette_name: str) -> None:
    for obj in root.iter("Object"):
        nm = obj.attrib.get("Name", "")
        if nm == "m_paletteNormalizationMode":
            obj.text = "NORMALIZATION_SPECIFIED_VALUES"
        elif nm == "m_selectedPaletteName":
            obj.text = palette_name
        elif nm == "m_userScalePercentageMinimum":
            obj.text = str(vmin)
        elif nm == "m_userScalePercentageMaximum":
            obj.text = str(vmax)


def _iter_colorbar_annotations(root: ET.Element) -> list[ET.Element]:
    out: list[ET.Element] = []
    for obj in root.iter("Object"):
        if obj.attrib.get("Type") != "class":
            continue
        if obj.attrib.get("Class") != "Annotation":
            continue
        if obj.attrib.get("Name") != "m_colorBar":
            continue
        out.append(obj)
    return out


def _iter_overlay_nodes(root: ET.Element) -> list[ET.Element]:
    out: list[ET.Element] = []
    for obj in root.iter("Object"):
        if obj.attrib.get("Type") != "class":
            continue
        if obj.attrib.get("Class") != "Overlay":
            continue
        out.append(obj)
    return out


def _child_text(node: ET.Element, field_name: str) -> str | None:
    for child in node:
        if child.attrib.get("Name") == field_name:
            return (child.text or "").strip()
    return None


def _child_object(node: ET.Element, field_name: str, class_name: str | None = None) -> ET.Element | None:
    for child in node:
        if child.attrib.get("Name") != field_name:
            continue
        if class_name is not None and child.attrib.get("Class") != class_name:
            continue
        return child
    return None


def _set_annotation_field(annotation: ET.Element, field_name: str, value: str) -> bool:
    for child in annotation:
        if child.attrib.get("Name") == field_name:
            child.text = value
            return True
    return False


def _set_colorbar_visible(annotation: ET.Element, visible: bool) -> None:
    _set_annotation_field(annotation, "m_displayedFlag", "true" if visible else "false")
    _set_annotation_field(annotation, "m_showTickMarksSelected", "true" if visible else "false")
    if visible:
        _set_annotation_field(annotation, "m_width", "25")
        _set_annotation_field(annotation, "m_height", "7")
        _set_annotation_field(annotation, "m_fontPercentViewportHeight", "3.33")
        _set_annotation_field(annotation, "m_positionMode", "AUTOMATIC")


def _set_single_visible_colorbar(root: ET.Element, chosen: ET.Element | None) -> int:
    all_colorbars = _iter_colorbar_annotations(root)
    for cbar in all_colorbars:
        _set_colorbar_visible(cbar, cbar is chosen)
    return len(all_colorbars)


def _canonical_scene_path(text: str | None) -> str:
    s = str(text or "").strip()
    if not s:
        return ""
    try:
        return str(Path(s).resolve())
    except Exception:
        return s


def _target_colorbar_candidates(
    root: ET.Element,
    target_metric_paths: set[str],
) -> list[tuple[str, ET.Element]]:
    out: list[tuple[str, ET.Element]] = []
    for overlay in _iter_overlay_nodes(root):
        selected = _canonical_scene_path(_child_text(overlay, "selectedMapFileNameWithPath"))
        if selected not in target_metric_paths:
            continue
        cbar = _child_object(overlay, "m_colorBar", class_name="Annotation")
        if cbar is None:
            continue
        out.append((selected, cbar))
    return out



def _capture_scene_root(
    *,
    root: ET.Element,
    out_png: Path,
    width: int,
    height: int,
    renderer: str,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".scene",
        prefix="group_gradient_heatmap_",
        dir=str(out_png.parent),
        delete=False,
    ) as tmp:
        temp_scene = Path(tmp.name)
    try:
        ET.ElementTree(root).write(temp_scene, encoding="UTF-8", xml_declaration=True)
        run_workbench(
            [
                "-scene-capture-image",
                str(temp_scene),
                "1",
                str(out_png),
                "-size-width-height",
                str(int(width)),
                str(int(height)),
                "-renderer",
                str(renderer),
            ]
        )
    finally:
        temp_scene.unlink(missing_ok=True)


def _diff_bbox(
    baseline_png: Path,
    probe_png: Path,
    *,
    pixel_delta: int = 6,
    min_changed_pixels: int = 300,
) -> tuple[int, int, int, int] | None:
    a = np.asarray(Image.open(baseline_png).convert("RGB"), dtype=np.int16)
    b = np.asarray(Image.open(probe_png).convert("RGB"), dtype=np.int16)
    if a.shape != b.shape:
        return None
    changed = np.any(np.abs(a - b) >= int(pixel_delta), axis=2)
    if int(np.count_nonzero(changed)) < int(min_changed_pixels):
        return None
    ys, xs = np.where(changed)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _select_visible_colorbar_overlay(
    *,
    root: ET.Element,
    target_metric_paths: set[str],
    scratch_dir: Path,
    width: int,
    height: int,
    renderer: str,
) -> tuple[str, ET.Element, tuple[int, int, int, int]]:
    candidates = _target_colorbar_candidates(root, target_metric_paths)
    if not candidates:
        raise RuntimeError("No target metric overlays with colorbar annotations found in scene")

    _set_single_visible_colorbar(root, None)
    baseline_png = scratch_dir / "colorbar_probe_baseline.png"
    _capture_scene_root(root=root, out_png=baseline_png, width=width, height=height, renderer=renderer)

    for idx, (selected_metric, cbar) in enumerate(candidates):
        _set_single_visible_colorbar(root, cbar)
        probe_png = scratch_dir / f"colorbar_probe_{idx:02d}.png"
        _capture_scene_root(root=root, out_png=probe_png, width=width, height=height, renderer=renderer)
        diff_box = _diff_bbox(baseline_png=baseline_png, probe_png=probe_png)
        if diff_box is not None:
            return selected_metric, cbar, diff_box

    raise RuntimeError(
        "Could not find any visible Workbench colorbar from target overlays after render validation"
    )


def infer_dense_corobl_root_from_spec(spec_path: Path) -> Path:
    parts = spec_path.resolve().parts
    if "dense_corobl_batch" not in parts:
        raise RuntimeError(f"Could not infer dense_corobl_batch root from spec path: {spec_path}")
    idx = parts.index("dense_corobl_batch")
    return Path(*parts[: idx + 1])


def build_scene_root(
    *,
    scene_path: Path,
    template_subject: str,
    left_surface: Path,
    right_surface: Path,
    spec_path: Path,
    left_shape: Path,
    right_shape: Path,
    vmin: float,
    vmax: float,
    palette_name: str,
) -> ET.Element:
    dense_corobl_root = infer_dense_corobl_root_from_spec(spec_path)
    root = ET.parse(scene_path).getroot()
    absolutize_scene_paths(root, scene_path, dense_corobl_root=dense_corobl_root)
    source_subject = detect_template_subject(root)
    replace_subject_text(root, source_subject, template_subject)

    surface_refs = find_corobl_surface_refs(root)
    replace_exact_ref(root, surface_refs["L"], str(left_surface.resolve()))
    replace_exact_ref(root, surface_refs["R"], str(right_surface.resolve()))

    current_spec = find_first_ref(root, r"space-corobl_surfaces\.spec$")
    replace_exact_ref(root, current_spec, str(spec_path.resolve()))

    gyr_refs = find_gyrification_refs(root)
    replace_exact_ref(root, gyr_refs["L"], str(left_shape.resolve()))
    replace_exact_ref(root, gyr_refs["R"], str(right_shape.resolve()))

    disable_label_overlays(root)
    inject_fixed_palette(root, vmin=vmin, vmax=vmax, palette_name=palette_name)
    return root


def render_panel(
    *,
    scene_path: Path,
    template_subject: str,
    left_surface: Path,
    right_surface: Path,
    spec_path: Path,
    left_shape: Path,
    right_shape: Path,
    vmin: float,
    vmax: float,
    palette_name: str,
    out_png: Path,
    width: int,
    height: int,
    renderer: str,
    enable_colorbar: bool,
    selected_colorbar_metric_hint: str | None = None,
) -> RenderMeta:
    root = build_scene_root(
        scene_path=scene_path,
        template_subject=template_subject,
        left_surface=left_surface,
        right_surface=right_surface,
        spec_path=spec_path,
        left_shape=left_shape,
        right_shape=right_shape,
        vmin=vmin,
        vmax=vmax,
        palette_name=palette_name,
    )

    target_metric_paths = {
        _canonical_scene_path(str(left_shape.resolve())),
        _canonical_scene_path(str(right_shape.resolve())),
    }
    scratch_dir = out_png.parent / "_colorbar_probe"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    selected_metric: str | None = None
    crop_box: tuple[int, int, int, int] | None = None
    if enable_colorbar:
        if _set_single_visible_colorbar(root, None) <= 0:
            raise RuntimeError("No Workbench colorbar annotations found in scene")

        if selected_colorbar_metric_hint:
            for metric_path, cbar in _target_colorbar_candidates(root, target_metric_paths):
                if metric_path == selected_colorbar_metric_hint:
                    selected_metric = metric_path
                    _set_single_visible_colorbar(root, cbar)
                    baseline_png = scratch_dir / "colorbar_probe_baseline_hint.png"
                    probe_png = scratch_dir / "colorbar_probe_hint.png"
                    _set_single_visible_colorbar(root, None)
                    _capture_scene_root(root=root, out_png=baseline_png, width=width, height=height, renderer=renderer)
                    _set_single_visible_colorbar(root, cbar)
                    _capture_scene_root(root=root, out_png=probe_png, width=width, height=height, renderer=renderer)
                    crop_box = _diff_bbox(baseline_png, probe_png)
                    if crop_box is None:
                        selected_metric = None
                        _set_single_visible_colorbar(root, None)
                    break

        if selected_metric is None:
            selected_metric, chosen_cbar, crop_box = _select_visible_colorbar_overlay(
                root=root,
                target_metric_paths=target_metric_paths,
                scratch_dir=scratch_dir,
                width=width,
                height=height,
                renderer=renderer,
            )
            _set_single_visible_colorbar(root, chosen_cbar)
    else:
        _set_single_visible_colorbar(root, None)

    _capture_scene_root(root=root, out_png=out_png, width=width, height=height, renderer=renderer)
    return RenderMeta(
        panel_png=out_png.resolve(),
        left_shape=left_shape.resolve(),
        right_shape=right_shape.resolve(),
        selected_colorbar_metric=selected_metric,
        colorbar_crop_box=crop_box,
    )



def crop_to_foreground(img: Image.Image, *, tolerance: int = 12, padding: int = 4) -> Image.Image:
    arr = np.asarray(img.convert("RGB"), dtype=np.int16)
    corners = np.asarray(
        [
            arr[0, 0],
            arr[0, -1],
            arr[-1, 0],
            arr[-1, -1],
        ],
        dtype=np.int16,
    )
    background = np.median(corners, axis=0)
    diff = np.max(np.abs(arr - background[None, None, :]), axis=2)
    mask = diff > int(tolerance)
    if not np.any(mask):
        return img
    ys, xs = np.where(mask)
    l = max(0, int(xs.min()) - int(padding))
    t = max(0, int(ys.min()) - int(padding))
    r = min(img.width, int(xs.max()) + 1 + int(padding))
    b = min(img.height, int(ys.max()) + 1 + int(padding))
    return img.crop((l, t, r, b))



def _find_horizontal_colorbar_box(src: Image.Image) -> tuple[int, int, int, int] | None:
    arr = np.asarray(src.convert("RGB"), dtype=np.int16)
    chroma = arr.max(axis=2) - arr.min(axis=2)
    nonwhite = np.any(arr < 245, axis=2)
    candidate = (chroma > 35) & nonwhite

    h, w = candidate.shape
    min_width = max(50, int(round(w * 0.10)))
    min_height = 6
    best: tuple[int, int, int, int, int] | None = None

    for y in range(h):
        xs = np.where(candidate[y])[0]
        if xs.size == 0:
            continue
        runs: list[tuple[int, int]] = []
        start = int(xs[0])
        prev = int(xs[0])
        for x in xs[1:]:
            x_i = int(x)
            if x_i == prev + 1:
                prev = x_i
                continue
            runs.append((start, prev + 1))
            start = x_i
            prev = x_i
        runs.append((start, prev + 1))

        for l, r in runs:
            if r - l < min_width:
                continue
            t = y
            b = y + 1
            while t > 0 and np.mean(candidate[t - 1, l:r]) > 0.35:
                t -= 1
            while b < h and np.mean(candidate[b, l:r]) > 0.35:
                b += 1
            if b - t < min_height:
                continue
            score = (r - l) * (b - t)
            if best is None or score > best[4]:
                best = (l, t, r, b, score)

    if best is None:
        return None

    l, t, r, b, _ = best
    pad_x = max(4, int(round((r - l) * 0.04)))
    label_pad_y = max(18, int(round((b - t) * 2.2)))
    return (
        max(0, l - pad_x),
        max(0, t - 4),
        min(w, r + pad_x),
        min(h, b + label_pad_y),
    )


def _has_horizontal_colorbar(img: Image.Image) -> bool:
    arr = np.asarray(img.convert("RGB"), dtype=np.int16)
    chroma = arr.max(axis=2) - arr.min(axis=2)
    colorful = chroma > 35
    min_run = max(40, int(round(arr.shape[1] * 0.30)))
    for row in colorful:
        xs = np.where(row)[0]
        if xs.size < min_run:
            continue
        run = 1
        prev = int(xs[0])
        for x in xs[1:]:
            x_i = int(x)
            if x_i == prev + 1:
                run += 1
                if run >= min_run:
                    return True
            else:
                run = 1
            prev = x_i
    return False


def _refine_colorbar_box_from_crop(crop: Image.Image, outer_box: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    inner = _find_horizontal_colorbar_box(crop)
    if inner is None:
        return None
    outer_l, outer_t, outer_r, outer_b = outer_box
    il, it, ir, ib = inner
    bar_w = max(1, ir - il)
    l = outer_l + max(0, il - int(round(bar_w * 0.42)))
    t = outer_t
    r = outer_l + min(crop.width, ir + int(round(bar_w * 0.06)))
    b = outer_t + min(crop.height, ib)
    return (
        max(outer_l, l),
        max(outer_t, t),
        min(outer_r, r),
        min(outer_b, b),
    )


def crop_workbench_colorbar(
    *,
    source_png: Path,
    crop_box: tuple[int, int, int, int] | None,
    fallback_ratio: tuple[float, float, float, float] = (0.58, 0.74, 0.985, 0.98),
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    src = Image.open(source_png).convert("RGBA")
    if crop_box is not None:
        l, t, r, b = crop_box
        l = max(0, min(src.width - 1, int(l)))
        t = max(0, min(src.height - 1, int(t)))
        r = max(l + 1, min(src.width, int(r)))
        b = max(t + 1, min(src.height, int(b)))
        cropped = src.crop((l, t, r, b))
        if _has_horizontal_colorbar(cropped):
            refined = _refine_colorbar_box_from_crop(cropped, (l, t, r, b))
            if refined is not None:
                rl, rt, rr, rb = refined
                return src.crop((rl, rt, rr, rb)), refined
            return cropped, (l, t, r, b)

    detected = _find_horizontal_colorbar_box(src)
    if detected is not None:
        l, t, r, b = detected
        return src.crop((l, t, r, b)), (l, t, r, b)

    l = int(round(src.width * fallback_ratio[0]))
    t = int(round(src.height * fallback_ratio[1]))
    r = int(round(src.width * fallback_ratio[2]))
    b = int(round(src.height * fallback_ratio[3]))
    l = max(0, min(src.width - 1, l))
    t = max(0, min(src.height - 1, t))
    r = max(l + 1, min(src.width, r))
    b = max(t + 1, min(src.height, b))
    return src.crop((l, t, r, b)), (l, t, r, b)


def fit_image(img: Image.Image, *, max_width: int, max_height: int) -> Image.Image:
    scale = min(float(max_width) / float(img.width), float(max_height) / float(img.height), 1.0)
    if scale >= 1.0:
        return img
    new_size = (
        max(1, int(round(img.width * scale))),
        max(1, int(round(img.height * scale))),
    )
    return img.resize(new_size, Image.Resampling.LANCZOS)


def compose_figure(
    panel_paths: dict[tuple[str, str], Path],
    colorbar_paths: dict[str, Path],
    colorbar_crop_boxes: dict[str, tuple[int, int, int, int] | None],
    panel_ev_percentages: dict[tuple[str, str], float],
    smoothings: list[str],
    components: list[str],
    output_png: Path,
) -> tuple[int, int]:
    compact_panels: dict[tuple[str, str], Image.Image] = {}
    for key, path in panel_paths.items():
        raw = Image.open(path).convert("RGBA")
        cropped = crop_to_foreground(raw, tolerance=12, padding=8)
        compact_panels[key] = fit_image(cropped, max_width=520, max_height=470)

    first = compact_panels[(smoothings[0], components[0])]
    cell_w, cell_h = first.size
    for key in panel_paths:
        im = compact_panels[key]
        cell_w = max(cell_w, im.size[0])
        cell_h = max(cell_h, im.size[1])

    colorbar_imgs: dict[str, Image.Image] = {}
    for comp in components:
        cropped, _ = crop_workbench_colorbar(
            source_png=colorbar_paths[comp],
            crop_box=colorbar_crop_boxes.get(comp),
        )
        colorbar_imgs[comp] = cropped

    col_gap = 20
    row_gap = 8
    left_margin = 104
    right_margin = 24
    top_margin = 16
    bottom_margin = 74
    cell_label_h = 28

    out_w = left_margin + len(components) * cell_w + (len(components) - 1) * col_gap + right_margin
    out_h = top_margin + len(smoothings) * (cell_label_h + cell_h) + (len(smoothings) - 1) * row_gap + bottom_margin
    canvas = Image.new("RGBA", (out_w, out_h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    cell_label_font = load_font(22)
    row_font = load_font(28)

    x0 = left_margin
    y0 = top_margin
    for r, smooth in enumerate(smoothings):
        for c, comp in enumerate(components):
            px = x0 + c * (cell_w + col_gap)
            py = y0 + r * (cell_label_h + cell_h + row_gap)
            panel = compact_panels[(smooth, comp)]
            off_x = (cell_w - panel.width) // 2
            off_y = (cell_h - panel.height) // 2
            label = format_cell_label(comp, panel_ev_percentages[(smooth, comp)])
            bbox = draw.textbbox((0, 0), label, font=cell_label_font)
            tx = px + (cell_w - (bbox[2] - bbox[0])) // 2
            ty = py + (cell_label_h - (bbox[3] - bbox[1])) // 2
            draw.text((tx, ty), label, font=cell_label_font, fill=(0, 0, 0, 255))
            canvas.paste(panel, (px + off_x, py + cell_label_h + off_y))
        row_text = smooth
        rb = draw.textbbox((0, 0), row_text, font=row_font)
        ry = y0 + r * (cell_label_h + cell_h + row_gap) + cell_label_h + (cell_h - (rb[3] - rb[1])) // 2
        draw.text((28, ry), row_text, font=row_font, fill=(0, 0, 0, 255))

    bar_y = y0 + len(smoothings) * (cell_label_h + cell_h) + (len(smoothings) - 1) * row_gap + 12
    for c, comp in enumerate(components):
        px = x0 + c * (cell_w + col_gap)
        src = colorbar_imgs[comp]
        target_w = int(cell_w * 0.62)
        if src.width <= 0 or src.height <= 0:
            continue
        target_h = min(52, max(16, int(round(target_w * (src.height / float(src.width))))))
        resized = src.resize((target_w, target_h), Image.Resampling.BICUBIC)
        bx = px + (cell_w - target_w) // 2
        canvas.paste(resized, (bx, bar_y))

    output_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_png)
    return out_w, out_h


def collect_combo_data(
    *,
    group_root: Path,
    smoothings: list[str],
    hemis: list[str],
    method: str,
    atlas: str,
    coverage_tag: str,
) -> dict[tuple[str, str], ComboData]:
    out: dict[tuple[str, str], ComboData] = {}
    for smoothing in smoothings:
        for hemi in hemis:
            run_dir = group_root / method / atlas / smoothing / coverage_tag / f"hemi_{hemi}"
            manifest_path = run_dir / "group_atlas_manifest.json"
            validate_manifest_path(manifest_path, group_root=group_root)
            if not manifest_path.exists():
                raise FileNotFoundError(f"Missing manifest: {manifest_path}")
            group_manifest = load_json(manifest_path)
            if str(group_manifest.get("status", "")) != "ok":
                raise RuntimeError(f"Manifest status is not ok: {manifest_path}")
            active_path = Path(str(group_manifest.get("group_gradient_features_active", run_dir / "group_gradient_features_active.npy")))
            support_path = Path(str(group_manifest.get("group_support_mask", run_dir / "group_support_mask.npy")))
            template_surface = Path(str(group_manifest.get("template_left_or_right_surface", "")))
            template_spec = Path(str(group_manifest.get("template_spec_path", "")))
            template_subject = str(group_manifest.get("template_subject", "")).replace("sub-", "").strip()
            if not template_subject:
                raise RuntimeError(f"Missing template_subject in {manifest_path}")
            gradient_summary_raw = group_manifest.get("group_gradient_feature_summary")
            gradient_summary_path = (
                Path(str(gradient_summary_raw)) if gradient_summary_raw else run_dir / "group_gradient_feature_summary.json"
            )
            for p in (active_path, support_path, template_surface, template_spec):
                if not p.exists():
                    raise FileNotFoundError(f"Missing required input from manifest: {p}")
            if not gradient_summary_path.exists():
                raise FileNotFoundError(f"Missing gradient summary input: {gradient_summary_path}")
            active = np.load(active_path).astype(np.float32)
            support_mask = np.load(support_path)
            gradient_summary = load_json(gradient_summary_path)
            vertex_count = read_surface_vertex_count(template_surface)
            full, support_count = reconstruct_full_components(active, support_mask, vertex_count)
            out[(smoothing, hemi)] = ComboData(
                smoothing=smoothing,
                hemi=hemi,
                run_dir=run_dir.resolve(),
                manifest_path=manifest_path.resolve(),
                group_manifest=group_manifest,
                active_path=active_path.resolve(),
                support_mask_path=support_path.resolve(),
                template_surface_path=template_surface.resolve(),
                template_spec_path=template_spec.resolve(),
                template_subject=template_subject,
                active=active,
                support_mask=np.asarray(support_mask).astype(bool),
                vertex_count=int(vertex_count),
                support_count=int(support_count),
                full_components=full,
                gradient_summary_path=gradient_summary_path.resolve(),
                component_ev_percentages=component_ev_percentages(
                    gradient_summary,
                    [f"G{i + 1}" for i in range(active.shape[1])],
                ),
            )
    return out


def main() -> int:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        parse_args()
        return 0
    print(json.dumps({
        "status": "deprecated",
        "step": "06_05_render_group_gradient_heatmaps",
        "message": "This legacy locked-scene heatmap output is retired and must not be refreshed.",
        "replacement": "Use scripts/core_workflow/06_04_render_hippomaps_gradient_products.py outputs under outputs_hcp/present/gradients/routea and outputs_hcp/present/gradients/routeb.",
    }, indent=2), flush=True)
    return 2

    args = parse_args()
    group_root = Path(args.group_root).resolve()
    scene_path = Path(args.scene).resolve()
    output_png = Path(args.output_png).resolve()
    output_manifest = Path(args.output_manifest).resolve()
    runtime_root = Path(args.runtime_root).resolve()
    smoothings = [str(s) for s in args.smoothings]
    hemis = [str(h) for h in args.hemis]
    component_names = [c.strip() for c in str(args.components).split(",") if c.strip()]
    effective_palette = resolve_workbench_palette_name(str(args.palette))

    if not scene_path.exists():
        raise FileNotFoundError(f"Missing scene file: {scene_path}")
    if any(tok in str(group_root) for tok in ARCHIVE_TOKENS):
        raise RuntimeError(f"Archive path is forbidden for group root: {group_root}")

    verify_workbench_palette_supported(effective_palette, runtime_root=runtime_root)

    combos = collect_combo_data(
        group_root=group_root,
        smoothings=smoothings,
        hemis=hemis,
        method=str(args.method),
        atlas=str(args.atlas),
        coverage_tag=str(args.coverage_tag),
    )
    full_for_scale = {(s, h): combos[(s, h)].full_components for s in smoothings for h in hemis}
    scales = compute_component_scales(full_for_scale, component_names)
    component_ev_by_combo = {(s, h): combos[(s, h)].component_ev_percentages for s in smoothings for h in hemis}
    panel_ev_percentages = compute_panel_ev_percentages(
        component_ev_by_combo=component_ev_by_combo,
        smoothings=smoothings,
        hemis=hemis,
        components=component_names,
    )

    runtime_scalars = runtime_root / "scalars"
    runtime_renders = runtime_root / "renders"
    shape_paths: dict[tuple[str, str, str], Path] = {}
    for smoothing in smoothings:
        for hemi in hemis:
            combo = combos[(smoothing, hemi)]
            for comp in component_names:
                comp_idx = component_index_from_name(comp)
                vals = combo.full_components[:, comp_idx]
                shape_name = (
                    f"group-average_hemi-{hemi}_space-corobl_den-512_label-"
                    f"group_gradient_{str(args.method)}_{str(args.atlas)}_{smoothing}_{comp}.shape.gii"
                )
                out_shape = runtime_scalars / smoothing / f"hemi_{hemi}" / shape_name
                save_shape_gii(vals, out_shape)
                comp_vmin, comp_vmax = scales[comp]
                apply_metric_palette(
                    metric_path=out_shape,
                    palette_name=str(effective_palette),
                    vmin=float(comp_vmin),
                    vmax=float(comp_vmax),
                )
                shape_paths[(smoothing, hemi, comp)] = out_shape.resolve()

    panel_paths: dict[tuple[str, str], Path] = {}
    panel_meta: dict[str, dict[str, Any]] = {}
    colorbar_sources: dict[str, dict[str, Any]] = {}
    if not args.dry_run:
        selected_colorbar_metric_hint: str | None = None
        ref_smoothing_for_colorbar = smoothings[0]
        for smoothing in smoothings:
            left_combo = combos[(smoothing, "L")]
            right_combo = combos[(smoothing, "R")]
            if left_combo.template_subject != right_combo.template_subject:
                raise RuntimeError(
                    f"Template subject mismatch for smoothing {smoothing}: "
                    f"L={left_combo.template_subject} R={right_combo.template_subject}"
                )
            if left_combo.template_spec_path != right_combo.template_spec_path:
                raise RuntimeError(
                    f"Template spec mismatch for smoothing {smoothing}: "
                    f"L={left_combo.template_spec_path} R={right_combo.template_spec_path}"
                )
            for comp in component_names:
                vmin, vmax = scales[comp]
                panel_png = runtime_renders / smoothing / f"{comp}_wb_locked.png"
                panel_render = render_panel(
                    scene_path=scene_path,
                    template_subject=left_combo.template_subject,
                    left_surface=left_combo.template_surface_path,
                    right_surface=right_combo.template_surface_path,
                    spec_path=left_combo.template_spec_path,
                    left_shape=shape_paths[(smoothing, "L", comp)],
                    right_shape=shape_paths[(smoothing, "R", comp)],
                    vmin=float(vmin),
                    vmax=float(vmax),
                    palette_name=str(effective_palette),
                    out_png=panel_png,
                    width=int(args.width),
                    height=int(args.height),
                    renderer=str(args.renderer),
                    enable_colorbar=False,
                )
                panel_paths[(smoothing, comp)] = panel_png.resolve()
                panel_meta[f"{smoothing}_{comp}"] = {
                    "panel_png": str(panel_render.panel_png),
                    "left_shape": str(panel_render.left_shape),
                    "right_shape": str(panel_render.right_shape),
                    "vmin": float(vmin),
                    "vmax": float(vmax),
                    "ev_percent": float(panel_ev_percentages[(smoothing, comp)]),
                }

                if smoothing == ref_smoothing_for_colorbar:
                    colorbar_png = runtime_renders / "colorbars" / f"{comp}_wb_colorbar_source.png"
                    colorbar_render = render_panel(
                        scene_path=scene_path,
                        template_subject=left_combo.template_subject,
                        left_surface=left_combo.template_surface_path,
                        right_surface=right_combo.template_surface_path,
                        spec_path=left_combo.template_spec_path,
                        left_shape=shape_paths[(smoothing, "L", comp)],
                        right_shape=shape_paths[(smoothing, "R", comp)],
                        vmin=float(vmin),
                        vmax=float(vmax),
                        palette_name=str(effective_palette),
                        out_png=colorbar_png,
                        width=int(args.width),
                        height=int(args.height),
                        renderer=str(args.renderer),
                        enable_colorbar=True,
                        selected_colorbar_metric_hint=selected_colorbar_metric_hint,
                    )
                    if selected_colorbar_metric_hint is None:
                        selected_colorbar_metric_hint = colorbar_render.selected_colorbar_metric
                    colorbar_sources[comp] = {
                        "source_png": str(colorbar_render.panel_png),
                        "selected_colorbar_metric": colorbar_render.selected_colorbar_metric,
                        "crop_box": list(colorbar_render.colorbar_crop_box) if colorbar_render.colorbar_crop_box else None,
                        "vmin": float(vmin),
                        "vmax": float(vmax),
                    }

    final_dims = None
    if not args.dry_run:
        if set(colorbar_sources) != set(component_names):
            raise RuntimeError(
                f"Missing colorbar source renders. expected={component_names} got={sorted(colorbar_sources)}"
            )
        final_dims = compose_figure(
            panel_paths=panel_paths,
            colorbar_paths={c: Path(str(colorbar_sources[c]["source_png"])) for c in component_names},
            colorbar_crop_boxes={
                c: tuple(colorbar_sources[c]["crop_box"]) if colorbar_sources[c]["crop_box"] else None
                for c in component_names
            },
            panel_ev_percentages=panel_ev_percentages,
            smoothings=smoothings,
            components=component_names,
            output_png=output_png,
        )

    manifest: dict[str, Any] = {
        "status": "dry-run-ok" if args.dry_run else "ok",
        "dry_run": bool(args.dry_run),
        "group_root": str(group_root),
        "scene": str(scene_path),
        "method": str(args.method),
        "atlas": str(args.atlas),
        "smoothings": smoothings,
        "hemis": hemis,
        "components": component_names,
        "coverage_tag": str(args.coverage_tag),
        "palette_engine": "workbench",
        "palette": str(effective_palette),
        "colorbar_mode": "workbench-rendered-cropped",
        "runtime_root": str(runtime_root),
        "output_png": str(output_png),
        "output_manifest": str(output_manifest),
        "scales": {k: {"vmin": float(v[0]), "vmax": float(v[1])} for k, v in scales.items()},
        "combos": {
            f"{smoothing}_{hemi}": {
                "run_dir": str(c.run_dir),
                "manifest_path": str(c.manifest_path),
                "group_gradient_features_active": str(c.active_path),
                "group_gradient_feature_summary": str(c.gradient_summary_path),
                "group_support_mask": str(c.support_mask_path),
                "template_subject": f"sub-{c.template_subject}",
                "template_surface_path": str(c.template_surface_path),
                "template_spec_path": str(c.template_spec_path),
                "vertex_count": int(c.vertex_count),
                "support_count": int(c.support_count),
                "active_rows": int(c.active.shape[0]),
                "active_cols": int(c.active.shape[1]),
                "component_ev_percentages": {k: float(v) for k, v in c.component_ev_percentages.items()},
            }
            for (smoothing, hemi), c in combos.items()
        },
        "panel_ev_percentages": {
            f"{smoothing}_{comp}": float(ev)
            for (smoothing, comp), ev in panel_ev_percentages.items()
        },
        "runtime_shapes": {
            f"{s}_{h}_{comp}": str(path)
            for (s, h, comp), path in shape_paths.items()
        },
        "panels": panel_meta,
        "colorbar_sources": colorbar_sources,
    }
    if final_dims is not None:
        manifest["output_dimensions"] = {"width": int(final_dims[0]), "height": int(final_dims[1])}
    write_json(output_manifest, manifest)

    if args.dry_run:
        print(f"[dry-run] validated inputs, wrote manifest: {output_manifest}")
    else:
        print(f"[ok] wrote figure: {output_png}")
        print(f"[ok] wrote manifest: {output_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

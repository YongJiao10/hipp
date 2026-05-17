#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
from matplotlib import colormaps, colors
from nibabel.gifti import GiftiDataArray, GiftiImage

from lib.workbench import run_workbench


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GROUP_ROOT = (
    REPO_ROOT
    / "outputs_experiments"
    / "hippunfold_recommended_t1w_20260423"
    / "kong2022"
    / "hipp_group_merged_canonical_atlas_routea_full"
)
DEFAULT_SCENE = REPO_ROOT / "config" / "wb_locked_native_view_lateral_medial.scene"
DEFAULT_OUT_PNG = (
    REPO_ROOT
    / "present_group_t1only"
    / "kong2022_gradient-spectral_group_gradients_heatmap_routea_2mm4mm_g1g2g3.png"
)
DEFAULT_OUT_MANIFEST = (
    REPO_ROOT
    / "present_group_t1only"
    / "kong2022_gradient-spectral_group_gradients_heatmap_routea_2mm4mm_g1g2g3_manifest.json"
)
DEFAULT_RUNTIME_ROOT = REPO_ROOT / "present_group_t1only" / "group_gradient_heatmaps" / "runtime" / "routea_full"

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
    parser.add_argument("--palette", default="plasma")
    parser.add_argument("--colorbar-cmap", default="plasma")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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


def read_surface_vertex_count(surface_path: Path) -> int:
    img = nib.load(str(surface_path))
    for da in img.darrays:
        if da.intent == "NIFTI_INTENT_POINTSET":
            return int(np.asarray(da.data).shape[0])
    return int(np.asarray(img.darrays[0].data).shape[0])


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
    if eigvals.size < 2:
        raise ValueError(f"Expected at least trivial + one non-trivial eigenvalue, got {eigvals.size}")
    nontrivial = eigvals[1:]
    denom = float(np.sum(nontrivial))
    if not np.isfinite(denom) or denom <= 0.0:
        raise ValueError(f"Non-trivial eigenvalue sum must be positive, got {denom}")

    out: dict[str, float] = {}
    for comp_name in component_names:
        comp_idx = component_index_from_name(comp_name)
        if comp_idx >= nontrivial.size:
            raise ValueError(
                f"Component {comp_name} out of bounds for {nontrivial.size} non-trivial eigenvalues"
            )
        out[comp_name] = float(nontrivial[comp_idx] / denom * 100.0)
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


def detect_template_subject(root: ET.Element) -> str:
    pat = re.compile(r"sub-(\d+)")
    for elem in root.iter():
        if elem.text:
            m = pat.search(elem.text)
            if m:
                return m.group(1)
    raise RuntimeError("Could not detect template subject in scene file")


def replace_subject_text(root: ET.Element, template_subject: str, target_subject: str) -> None:
    old = f"sub-{template_subject}"
    new = f"sub-{target_subject}"
    for elem in root.iter():
        if elem.text and old in elem.text:
            elem.text = elem.text.replace(old, new)


def absolutize_scene_paths(root: ET.Element, scene_path: Path, dense_corobl_root: Path | None = None) -> None:
    base = scene_path.parent
    for obj in root.iter("Object"):
        if obj.attrib.get("Type") != "pathName" or not obj.text:
            continue
        text = obj.text.strip()
        if not text:
            continue
        candidate = Path(text)
        resolved = (base / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        if dense_corobl_root is not None:
            m = re.search(r"dense_corobl_batch(/.*)", str(resolved))
            if m:
                resolved = (dense_corobl_root / m.group(1).lstrip("/")).resolve()
        obj.text = str(resolved)


def replace_exact_ref(root: ET.Element, current_ref: str, target_ref: str) -> None:
    current_name = Path(current_ref).name
    target_name = Path(target_ref).name
    for elem in root.iter():
        name_attr = elem.attrib.get("Name")
        if name_attr and current_name in name_attr:
            elem.attrib["Name"] = name_attr.replace(current_name, target_name)
        if not elem.text:
            continue
        if current_ref in elem.text:
            elem.text = elem.text.replace(current_ref, target_ref)
        if current_name in elem.text:
            elem.text = elem.text.replace(current_name, target_name)


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


def validate_scene_paths(root: ET.Element) -> None:
    missing: list[str] = []
    for obj in root.iter("Object"):
        if obj.attrib.get("Type") != "pathName" or not obj.text:
            continue
        p = Path(obj.text.strip())
        if p.exists():
            continue
        if p.suffix in {".palette", ".scene", ".png", ".jpg", ".jpeg"}:
            continue
        missing.append(str(p))
    if missing:
        preview = "\n".join(missing[:10])
        raise RuntimeError(f"Scene has missing path references after substitution:\n{preview}")


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
) -> None:
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


def infer_dense_corobl_root_from_spec(spec_path: Path) -> Path:
    parts = spec_path.resolve().parts
    if "dense_corobl_batch" not in parts:
        raise RuntimeError(f"Could not infer dense_corobl_batch root from spec path: {spec_path}")
    idx = parts.index("dense_corobl_batch")
    return Path(*parts[: idx + 1])


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
    ):
        p = Path(candidate)
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def foreground_mask(image: Image.Image, threshold: int = 15) -> np.ndarray:
    arr = np.asarray(image.convert("RGBA"))
    return arr[..., :3].sum(axis=2) > threshold


def find_runs(flags: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(flags.tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(flags)))
    return runs


def trim_black(img: Image.Image, threshold: int = 8) -> Image.Image:
    arr = np.asarray(img.convert("RGB"))
    mask = np.any(arr > threshold, axis=2)
    if not np.any(mask):
        return img
    ys, xs = np.where(mask)
    return img.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))


def split_native_hemi_panels(image: Image.Image) -> dict[str, Image.Image]:
    mask = foreground_mask(image)
    col_runs = find_runs(mask.any(axis=0))
    if len(col_runs) < 2:
        raise RuntimeError("Could not detect bilateral foreground runs in native render")
    selected_runs = sorted(col_runs, key=lambda item: item[1] - item[0], reverse=True)[:2]
    selected_runs = sorted(selected_runs, key=lambda item: item[0])
    panels: dict[str, Image.Image] = {}
    for hemi, (x0, x1) in zip(["L", "R"], selected_runs, strict=True):
        panels[hemi] = trim_black(image.crop((x0, 0, x1, image.height)))
    return panels


def compact_locked_panel(image: Image.Image) -> Image.Image:
    panels = split_native_hemi_panels(image.convert("RGBA"))
    left = panels["L"]
    right = panels["R"]
    target_h = max(left.height, right.height)
    if left.height != target_h:
        scale = target_h / float(left.height)
        left = left.resize((max(1, int(round(left.width * scale))), target_h), Image.Resampling.LANCZOS)
    if right.height != target_h:
        scale = target_h / float(right.height)
        right = right.resize((max(1, int(round(right.width * scale))), target_h), Image.Resampling.LANCZOS)
    gap = 0
    canvas = Image.new("RGBA", (left.width + gap + right.width, target_h), (255, 255, 255, 255))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + gap, 0))
    return canvas


def hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
    h = hex_color.strip().lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def wb_palette_stops(palette_name: str) -> list[tuple[float, tuple[float, float, float]]]:
    if palette_name == "ROY-BIG-BL":
        # Workbench ROY-BIG-BL landmarks (value in [-1,1], converted to [0,1]).
        raw = [
            (-1.000, "#00ffff"),
            (-0.990, "#00ffff"),
            (-0.875, "#00ff00"),
            (-0.750, "#00c800"),
            (-0.625, "#4b7d00"),
            (-0.500, "#7d00a0"),
            (-0.375, "#4b007d"),
            (-0.250, "#0000aa"),
            (-0.125, "#000050"),
            (0.000, "#000000"),
            (0.125, "#3c0000"),
            (0.250, "#640000"),
            (0.375, "#960000"),
            (0.500, "#c80000"),
            (0.625, "#ff0000"),
            (0.750, "#ff7800"),
            (0.875, "#ffc800"),
            (0.975, "#fff400"),
            (1.000, "#ffff00"),
        ]
        return [((v + 1.0) * 0.5, hex_to_rgb01(c)) for v, c in raw]
    if palette_name == "RedWhiteBlue":
        # Simple non-rainbow diverging map (negative blue, zero white, positive red).
        raw = [(-1.0, "#0000ff"), (0.0, "#ffffff"), (1.0, "#ff0000")]
        return [((v + 1.0) * 0.5, hex_to_rgb01(c)) for v, c in raw]
    return []


def interpolate_palette(stops: list[tuple[float, tuple[float, float, float]]], xs: np.ndarray) -> np.ndarray:
    loc = np.asarray([s[0] for s in stops], dtype=np.float32)
    cols = np.asarray([s[1] for s in stops], dtype=np.float32)
    r = np.interp(xs, loc, cols[:, 0])
    g = np.interp(xs, loc, cols[:, 1])
    b = np.interp(xs, loc, cols[:, 2])
    out = np.stack([r, g, b], axis=1)
    return np.clip(out, 0.0, 1.0)


def make_colorbar_strip(width: int, height: int, cmap_name: str, palette_name: str) -> Image.Image:
    xs = np.linspace(0.0, 1.0, width, dtype=np.float32)
    stops = wb_palette_stops(palette_name)
    if stops:
        rgb = interpolate_palette(stops, xs)
        rgba = np.concatenate([rgb, np.ones((rgb.shape[0], 1), dtype=np.float32)], axis=1)
    else:
        cmap_name_mpl = {"spectral": "Spectral", "cool-warm": "coolwarm"}.get(cmap_name, cmap_name)
        cmap = colormaps.get_cmap(cmap_name_mpl)
        rgba = cmap(xs)
    row = (rgba * 255.0).astype(np.uint8)[None, :, :]
    arr = np.repeat(row, height, axis=0)
    return Image.fromarray(arr, mode="RGBA")


def compose_figure(
    panel_paths: dict[tuple[str, str], Path],
    scales: dict[str, tuple[float, float]],
    panel_ev_percentages: dict[tuple[str, str], float],
    smoothings: list[str],
    components: list[str],
    output_png: Path,
    colorbar_cmap: str,
    palette_name: str,
) -> tuple[int, int]:
    compact_panels: dict[tuple[str, str], Image.Image] = {}
    for key, path in panel_paths.items():
        raw = Image.open(path).convert("RGBA")
        compact_panels[key] = compact_locked_panel(raw)

    first = compact_panels[(smoothings[0], components[0])]
    cell_w, cell_h = first.size
    for key, path in panel_paths.items():
        im = compact_panels[key]
        cell_w = max(cell_w, im.size[0])
        cell_h = max(cell_h, im.size[1])

    col_gap = 20
    row_gap = 14
    left_margin = 112
    right_margin = 24
    top_margin = 28
    bottom_margin = 112
    bar_h = 22
    cell_label_h = 38

    out_w = left_margin + 3 * cell_w + 2 * col_gap + right_margin
    out_h = top_margin + 2 * (cell_label_h + cell_h) + row_gap + bottom_margin
    canvas = Image.new("RGBA", (out_w, out_h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    cell_label_font = load_font(28)
    row_font = load_font(30)
    tick_font = load_font(22)

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

    bar_y = y0 + 2 * (cell_label_h + cell_h) + row_gap + 34
    for c, comp in enumerate(components):
        px = x0 + c * (cell_w + col_gap)
        vmin, vmax = scales[comp]
        bar_w = int(cell_w * 0.74)
        bx = px + (cell_w - bar_w) // 2
        strip = make_colorbar_strip(bar_w, bar_h, colorbar_cmap, palette_name)
        canvas.paste(strip, (bx, bar_y))
        draw.rectangle([bx, bar_y, bx + bar_w, bar_y + bar_h], outline=(0, 0, 0, 255), width=1)
        left_txt = f"{vmin:.3f}"
        zero_txt = "0"
        right_txt = f"{vmax:.3f}"
        lb = draw.textbbox((0, 0), left_txt, font=tick_font)
        zb = draw.textbbox((0, 0), zero_txt, font=tick_font)
        rb = draw.textbbox((0, 0), right_txt, font=tick_font)
        tick_y = bar_y + bar_h + 8
        draw.text((bx - (lb[2] - lb[0]) // 2, tick_y), left_txt, font=tick_font, fill=(0, 0, 0, 255))
        draw.text((bx + bar_w // 2 - (zb[2] - zb[0]) // 2, tick_y), zero_txt, font=tick_font, fill=(0, 0, 0, 255))
        draw.text((bx + bar_w - (rb[2] - rb[0]) // 2, tick_y), right_txt, font=tick_font, fill=(0, 0, 0, 255))

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
    args = parse_args()
    group_root = Path(args.group_root).resolve()
    scene_path = Path(args.scene).resolve()
    output_png = Path(args.output_png).resolve()
    output_manifest = Path(args.output_manifest).resolve()
    runtime_root = Path(args.runtime_root).resolve()
    smoothings = [str(s) for s in args.smoothings]
    hemis = [str(h) for h in args.hemis]
    component_names = [c.strip() for c in str(args.components).split(",") if c.strip()]

    if not scene_path.exists():
        raise FileNotFoundError(f"Missing scene file: {scene_path}")
    if any(tok in str(group_root) for tok in ARCHIVE_TOKENS):
        raise RuntimeError(f"Archive path is forbidden for group root: {group_root}")

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
                    palette_name=str(args.palette),
                    vmin=float(comp_vmin),
                    vmax=float(comp_vmax),
                )
                shape_paths[(smoothing, hemi, comp)] = out_shape.resolve()

    panel_paths: dict[tuple[str, str], Path] = {}
    panel_meta: dict[str, dict[str, Any]] = {}
    if not args.dry_run:
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
                render_panel(
                    scene_path=scene_path,
                    template_subject=left_combo.template_subject,
                    left_surface=left_combo.template_surface_path,
                    right_surface=right_combo.template_surface_path,
                    spec_path=left_combo.template_spec_path,
                    left_shape=shape_paths[(smoothing, "L", comp)],
                    right_shape=shape_paths[(smoothing, "R", comp)],
                    vmin=float(vmin),
                    vmax=float(vmax),
                    palette_name=str(args.palette),
                    out_png=panel_png,
                    width=int(args.width),
                    height=int(args.height),
                    renderer=str(args.renderer),
                )
                panel_paths[(smoothing, comp)] = panel_png.resolve()
                panel_meta[f"{smoothing}_{comp}"] = {
                    "panel_png": str(panel_png.resolve()),
                    "left_shape": str(shape_paths[(smoothing, "L", comp)]),
                    "right_shape": str(shape_paths[(smoothing, "R", comp)]),
                    "vmin": float(vmin),
                    "vmax": float(vmax),
                    "ev_percent": float(panel_ev_percentages[(smoothing, comp)]),
                }

    final_dims = None
    if not args.dry_run:
        final_dims = compose_figure(
            panel_paths=panel_paths,
            scales=scales,
            panel_ev_percentages=panel_ev_percentages,
            smoothings=smoothings,
            components=component_names,
            output_png=output_png,
            colorbar_cmap=str(args.colorbar_cmap),
            palette_name=str(args.palette),
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
        "palette": str(args.palette),
        "colorbar_cmap": str(args.colorbar_cmap),
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
    }
    if final_dims is not None:
        manifest["output_dimensions"] = {"width": int(final_dims[0]), "height": int(final_dims[1])}
    save_json(output_manifest, manifest)

    if args.dry_run:
        print(f"[dry-run] validated inputs, wrote manifest: {output_manifest}")
    else:
        print(f"[ok] wrote figure: {output_png}")
        print(f"[ok] wrote manifest: {output_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

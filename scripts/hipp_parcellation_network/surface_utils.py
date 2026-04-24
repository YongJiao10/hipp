#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.gifti import GiftiDataArray, GiftiImage, GiftiLabel, GiftiLabelTable


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_EXE = sys.executable or "python3"
RENDER_SCENE_BATCH = REPO_ROOT / "scripts" / "workbench" / "render_wb_scene_batch.py"
COMPOSE_WB_GRID = REPO_ROOT / "scripts" / "workbench" / "compose_wb_grid_with_legend.py"
NETWORK_STYLE_JSON = REPO_ROOT / "config" / "hipp_network_style.json"
DEFAULT_SCENE = REPO_ROOT / "config" / "wb_locked_native_view_lateral_medial.scene"


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\\nSTDOUT:\\n{proc.stdout}\\nSTDERR:\\n{proc.stderr}"
        )


def load_network_colors(path: Path) -> dict[str, tuple[int, int, int, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, tuple[int, int, int, int]] = {}
    for _, row in payload.items():
        out[str(row["name"])] = tuple(int(v) for v in row["rgba"])
    return out


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
        rgba = network_colors.get(name, (128, 128, 128, 255))
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
    cmd_render = [
        PYTHON_EXE,
        str(RENDER_SCENE_BATCH),
        "--scene",
        str(scene),
        "--subjects",
        subject,
        "--outdir",
        str(native_root / "batch"),
        "--scene-index",
        "1",
        "--width",
        "1600",
        "--height",
        "1024",
        "--renderer",
        "OSMesa",
        "--name",
        f"{name}_native",
        "--no-template-scene",
        "--left-label-template",
        str(left_labels),
        "--right-label-template",
        str(right_labels),
        "--left-surface-template",
        str(left_surface),
        "--right-surface-template",
        str(right_surface),
        "--spec-template",
        str(spec_path),
    ]
    run(cmd_render)
    native_png = native_root / "batch" / f"sub-{subject}" / f"sub-{subject}_wb_{name}_native_native.png"
    final_png = outdir / f"sub-{subject}_wb_{name}_biglegend.png"
    cmd_compose = [
        PYTHON_EXE,
        str(COMPOSE_WB_GRID),
        "--ventral-image",
        str(native_png),
        "--layout",
        "1x2",
        "--left-labels",
        str(left_labels),
        "--right-labels",
        str(right_labels),
        "--legend-group",
        "label",
        "--title",
        title,
        "--out",
        str(final_png),
    ]
    run(cmd_compose)
    return {"native_scene_png": str(native_png), "biglegend_png": str(final_png)}



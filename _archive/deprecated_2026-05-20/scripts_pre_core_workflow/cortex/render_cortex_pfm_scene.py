#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


WB_COMMAND = str((Path(__file__).resolve().parents[1] / "wb_command").resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a cortex PFM label from a saved Workbench scene template."
    )
    parser.add_argument("--source-scene", required=True, help="Existing Workbench scene template")
    parser.add_argument("--left-surface", required=True, help="Left inflated surface")
    parser.add_argument("--right-surface", required=True, help="Right inflated surface")
    parser.add_argument("--left-label", required=True, help="Left hemisphere label.gii")
    parser.add_argument("--right-label", required=True, help="Right hemisphere label.gii")
    parser.add_argument("--left-sulc", required=True, help="Left hemisphere sulc metric/func gii")
    parser.add_argument("--right-sulc", required=True, help="Right hemisphere sulc metric/func gii")
    parser.add_argument("--sulc-dscalar", required=True, help="Whole-cortex sulc dscalar")
    parser.add_argument("--dlabel", required=True, help="Whole-cortex dlabel")
    parser.add_argument("--out-scene", required=True)
    parser.add_argument("--out-png", required=True)
    parser.add_argument("--scene-index", type=int, default=1)
    parser.add_argument("--width", type=int, default=2000)
    parser.add_argument("--height", type=int, default=1400)
    parser.add_argument("--renderer", default="OSMesa")
    parser.add_argument("--rotation-axis", choices=["x", "y", "z"], default=None)
    parser.add_argument("--rotation-deg", type=float, default=0.0)
    return parser.parse_args()


def axis_rotation(axis: str, degrees: float) -> np.ndarray:
    theta = math.radians(degrees)
    c = math.cos(theta)
    s = math.sin(theta)
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)
    if axis == "z":
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    raise ValueError(f"Unsupported axis: {axis}")


def find_rotation_matrix_element(root: ET.Element) -> ET.Element:
    for obj in root.iter("Object"):
        if obj.attrib.get("Class") == "ViewingTransformations" and obj.attrib.get("Name") == "m_viewingTransformation":
            for child in obj:
                if child.attrib.get("Name") == "m_rotationMatrix":
                    return child
    raise RuntimeError("Could not find viewing transformation matrix")


def read_rotation_matrix(elem: ET.Element) -> np.ndarray:
    values = [float(child.text.strip()) for child in elem.findall("Element")]
    if len(values) != 16:
        raise RuntimeError(f"Expected 16 rotation matrix elements, found {len(values)}")
    return np.array(values, dtype=float).reshape(4, 4)


def write_rotation_matrix(elem: ET.Element, mat: np.ndarray) -> None:
    for child, value in zip(elem.findall("Element"), mat.reshape(-1)):
        child.text = f"{value:.9g}"


def classify_scene_file(name_or_path: str) -> str | None:
    name = Path(name_or_path.strip()).name
    if not name:
        return None
    if name.endswith("_inflated.surf.gii"):
        return "left_surface" if "hemi-L" in name else "right_surface" if "hemi-R" in name else None
    if name.endswith(".label.gii"):
        return "left_label" if name.endswith(".L.label.gii") else "right_label" if name.endswith(".R.label.gii") else None
    if name.endswith("_sulc.func.gii"):
        return "left_sulc" if "hemi-L" in name else "right_sulc" if "hemi-R" in name else None
    if name.endswith("_sulc.dscalar.nii"):
        return "sulc_dscalar"
    if name.endswith(".dlabel.nii"):
        return "dlabel"
    return None


def rewrite_scene_file_refs(root: ET.Element, replacements: dict[str, Path]) -> None:
    basename_lookup = {key: path.name for key, path in replacements.items()}

    for elem in root.iter():
        text = elem.text.strip() if elem.text else ""
        file_kind = classify_scene_file(text) if text else None
        if file_kind:
            target = replacements[file_kind]
            if elem.attrib.get("Type") == "pathName":
                elem.text = str(target)
            elif elem.attrib.get("Type") == "string":
                elem.text = target.name

        name_attr = elem.attrib.get("Name")
        file_kind = classify_scene_file(name_attr) if name_attr else None
        if file_kind:
            elem.attrib["Name"] = basename_lookup[file_kind]


def validate_required_paths(args: argparse.Namespace) -> None:
    for path_str in [
        args.source_scene,
        args.left_surface,
        args.right_surface,
        args.left_label,
        args.right_label,
        args.left_sulc,
        args.right_sulc,
        args.sulc_dscalar,
        args.dlabel,
    ]:
        path = Path(path_str).resolve()
        if not path.exists():
            raise FileNotFoundError(path)


def capture_scene(scene_file: Path, scene_index: int, out_png: Path, width: int, height: int, renderer: str) -> None:
    cmd = [
        WB_COMMAND,
        "-scene-capture-image",
        str(scene_file),
        str(scene_index),
        str(out_png),
        "-size-width-height",
        str(width),
        str(height),
        "-renderer",
        renderer,
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"wb_command failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")


def main() -> int:
    args = parse_args()
    validate_required_paths(args)

    source_scene = Path(args.source_scene).resolve()
    out_scene = Path(args.out_scene).resolve()
    out_png = Path(args.out_png).resolve()
    out_scene.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(source_scene)
    root = tree.getroot()
    rewrite_scene_file_refs(
        root,
        {
            "left_surface": Path(args.left_surface).resolve(),
            "right_surface": Path(args.right_surface).resolve(),
            "left_label": Path(args.left_label).resolve(),
            "right_label": Path(args.right_label).resolve(),
            "left_sulc": Path(args.left_sulc).resolve(),
            "right_sulc": Path(args.right_sulc).resolve(),
            "sulc_dscalar": Path(args.sulc_dscalar).resolve(),
            "dlabel": Path(args.dlabel).resolve(),
        },
    )

    if args.rotation_axis:
        rot_elem = find_rotation_matrix_element(root)
        base_matrix = read_rotation_matrix(rot_elem)
        updated = np.array(base_matrix, copy=True)
        updated[:3, :3] = axis_rotation(args.rotation_axis, args.rotation_deg) @ base_matrix[:3, :3]
        write_rotation_matrix(rot_elem, updated)

    ET.ElementTree(root).write(out_scene, encoding="UTF-8", xml_declaration=True)
    capture_scene(out_scene, args.scene_index, out_png, args.width, args.height, args.renderer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

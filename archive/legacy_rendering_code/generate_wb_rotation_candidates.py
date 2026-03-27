#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import math
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


WB_COMMAND = 'arch -x86_64 "/Applications/wb_view.app/Contents/usr/bin/wb_command"'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Workbench rotation candidate sheets by editing a scene file's viewing matrix."
    )
    parser.add_argument("--scene", required=True, help="Source .scene file saved from wb_view")
    parser.add_argument("--outdir", required=True, help="Output directory for frames and montages")
    parser.add_argument("--step-deg", type=int, default=10)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--scene-index", type=int, default=1)
    parser.add_argument("--renderer", default="OSMesa")
    parser.add_argument("--subject-label", default="")
    return parser.parse_args()


def axis_rotation(axis: str, degrees: float) -> np.ndarray:
    theta = math.radians(degrees)
    c = math.cos(theta)
    s = math.sin(theta)
    if axis == "x":
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, c, -s],
                [0.0, s, c],
            ],
            dtype=float,
        )
    if axis == "y":
        return np.array(
            [
                [c, 0.0, s],
                [0.0, 1.0, 0.0],
                [-s, 0.0, c],
            ],
            dtype=float,
        )
    if axis == "z":
        return np.array(
            [
                [c, -s, 0.0],
                [s, c, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
    raise ValueError(f"Unsupported axis: {axis}")


def find_rotation_matrix_element(root: ET.Element) -> ET.Element:
    for obj in root.iter("Object"):
        if obj.attrib.get("Class") == "ViewingTransformations" and obj.attrib.get("Name") == "m_viewingTransformation":
            for child in obj:
                if child.attrib.get("Name") == "m_rotationMatrix":
                    return child
    raise RuntimeError("Could not find m_viewingTransformation/m_rotationMatrix in scene file")


def read_rotation_matrix(elem: ET.Element) -> np.ndarray:
    values = [float(child.text.strip()) for child in elem.findall("Element")]
    if len(values) != 16:
        raise RuntimeError(f"Expected 16 matrix elements, found {len(values)}")
    mat = np.array(values, dtype=float).reshape(4, 4)
    return mat


def write_rotation_matrix(elem: ET.Element, mat: np.ndarray) -> None:
    flat = mat.reshape(-1)
    for child, value in zip(elem.findall("Element"), flat):
        child.text = f"{value:.9g}"


def absolutize_scene_paths(root: ET.Element, scene_path: Path) -> None:
    base = scene_path.parent
    for obj in root.iter("Object"):
        if obj.attrib.get("Type") == "pathName" and obj.text:
            text = obj.text.strip()
            if not text:
                continue
            candidate = Path(text)
            if not candidate.is_absolute():
                obj.text = str((base / candidate).resolve())


def capture_scene(scene_file: Path, scene_index: int, out_png: Path, width: int, height: int, renderer: str) -> None:
    cmd = (
        f'{WB_COMMAND} -scene-capture-image "{scene_file}" {scene_index} "{out_png}" '
        f"-size-width-height {width} {height} -renderer {renderer}"
    )
    proc = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"wb_command failed for {scene_file}:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")


def make_montage(
    axis: str,
    frame_paths: list[Path],
    degrees: list[int],
    out_path: Path,
    subject_label: str,
    cols: int = 6,
) -> None:
    images = [Image.open(path).convert("RGB") for path in frame_paths]
    tile_w, tile_h = images[0].size
    rows = math.ceil(len(images) / cols)
    title_h = 52
    label_h = 28
    canvas = Image.new("RGB", (cols * tile_w, title_h + rows * (tile_h + label_h)), color=(18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    title = f"{subject_label + ' ' if subject_label else ''}Workbench Rotation Sweep - axis {axis.upper()} (0..350 by 10 deg)"
    draw.text((16, 14), title, fill=(255, 255, 255))
    for idx, (img, deg) in enumerate(zip(images, degrees)):
        row = idx // cols
        col = idx % cols
        x = col * tile_w
        y = title_h + row * (tile_h + label_h)
        canvas.paste(img, (x, y))
        draw.text((x + 10, y + tile_h + 5), f"{deg:03d}°", fill=(255, 255, 255))
    canvas.save(out_path)


def main() -> int:
    args = parse_args()
    scene_path = Path(args.scene).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    frames_root = outdir / "frames"
    frames_root.mkdir(parents=True, exist_ok=True)

    base_tree = ET.parse(scene_path)
    base_root = base_tree.getroot()
    absolutize_scene_paths(base_root, scene_path)
    matrix_elem = find_rotation_matrix_element(base_root)
    base_matrix = read_rotation_matrix(matrix_elem)
    base_rot = base_matrix[:3, :3]

    angles = list(range(0, 360, args.step_deg))
    for axis in ["x", "y", "z"]:
        axis_dir = frames_root / axis
        axis_dir.mkdir(parents=True, exist_ok=True)
        frame_paths: list[Path] = []
        for deg in angles:
            root = copy.deepcopy(base_root)
            rot_elem = find_rotation_matrix_element(root)
            mat = np.eye(4, dtype=float)
            mat[:3, :3] = axis_rotation(axis, deg) @ base_rot
            write_rotation_matrix(rot_elem, mat)
            scene_variant = outdir / f"scene_axis_{axis}_{deg:03d}.scene"
            ET.ElementTree(root).write(scene_variant, encoding="UTF-8", xml_declaration=True)
            png_path = axis_dir / f"{deg:03d}.png"
            capture_scene(scene_variant, args.scene_index, png_path, args.width, args.height, args.renderer)
            frame_paths.append(png_path)
        make_montage(
            axis=axis,
            frame_paths=frame_paths,
            degrees=angles,
            out_path=outdir / f"montage_axis_{axis}.png",
            subject_label=args.subject_label,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

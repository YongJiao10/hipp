#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


WB_COMMAND = 'arch -x86_64 "/Applications/wb_view.app/Contents/usr/bin/wb_command"'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render multiple subjects from a saved Workbench scene template."
    )
    parser.add_argument("--scene", required=True, help="Source .scene file saved from wb_view")
    parser.add_argument("--subjects", nargs="+", required=True, help="Subject IDs without sub- prefix")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--scene-index", type=int, default=1)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--renderer", default="OSMesa")
    parser.add_argument("--left-label-template", default=None, help="Optional absolute path template, e.g. /path/sub-{subject}_...label.gii")
    parser.add_argument("--right-label-template", default=None, help="Optional absolute path template, e.g. /path/sub-{subject}_...label.gii")
    parser.add_argument("--name", default="structural", help="Output stem label, e.g. structural or wta")
    parser.add_argument(
        "--no-template-scene",
        action="store_true",
        help="Do not persist per-subject *_template.scene files; render from temporary scene files only.",
    )
    return parser.parse_args()


def detect_template_subject(root: ET.Element) -> str:
    pattern = re.compile(r"sub-(\d+)")
    for elem in root.iter():
        if elem.text:
            match = pattern.search(elem.text)
            if match:
                return match.group(1)
    raise RuntimeError("Could not detect template subject in scene file")


def absolutize_scene_paths(root: ET.Element, scene_path: Path) -> None:
    base = scene_path.parent
    for obj in root.iter("Object"):
        if obj.attrib.get("Type") == "pathName" and obj.text:
            text = obj.text.strip()
            if not text:
                continue
            candidate = Path(text)
            if candidate.is_absolute():
                obj.text = str(candidate)
                continue

            resolved = (base / candidate).resolve()
            if resolved.exists():
                obj.text = str(resolved)
                continue

            parts = candidate.parts
            if len(parts) >= 3 and parts[1:3] == ("Documents", "HippoMaps"):
                obj.text = str((Path.home() / Path(*parts[1:])).resolve())
                continue

            obj.text = str(resolved)


def replace_subject_text(root: ET.Element, template_subject: str, target_subject: str) -> None:
    template_label = f"sub-{template_subject}"
    target_label = f"sub-{target_subject}"
    for elem in root.iter():
        if elem.text and template_label in elem.text:
            elem.text = elem.text.replace(template_label, target_label)


def find_current_structural_label_refs(root: ET.Element) -> dict[str, str]:
    refs: dict[str, str] = {}
    for elem in root.iter():
        text = (elem.text or "").strip()
        if not text or "atlas-multihist7_subfields.label.gii" not in text or "space-corobl" not in text:
            continue
        if "hemi-L" in text and "L" not in refs:
            refs["L"] = text
        elif "hemi-R" in text and "R" not in refs:
            refs["R"] = text
    if "L" not in refs or "R" not in refs:
        raise RuntimeError("Could not find current structural label references in scene file")
    return refs


def replace_label_refs(root: ET.Element, current_ref: str, target_ref: str) -> None:
    current_name = Path(current_ref).name
    target_name = Path(target_ref).name
    for elem in root.iter():
        if not elem.text:
            continue
        if current_ref in elem.text:
            elem.text = elem.text.replace(current_ref, target_ref)
        if current_name in elem.text:
            elem.text = elem.text.replace(current_name, target_name)


def validate_scene_paths(root: ET.Element) -> None:
    missing: list[str] = []
    for obj in root.iter("Object"):
        if obj.attrib.get("Type") == "pathName" and obj.text:
            path = Path(obj.text.strip())
            if not path.exists():
                # Workbench scene files can retain optional local palette references
                # or preview artifacts that are not needed for batch surface rendering.
                if path.suffix in {".palette", ".scene", ".png", ".jpg", ".jpeg"}:
                    continue
                missing.append(str(path))
    if missing:
        preview = "\n".join(missing[:10])
        raise RuntimeError(f"Scene references missing files after subject replacement:\n{preview}")


def capture_scene(scene_file: Path, scene_index: int, out_png: Path, width: int, height: int, renderer: str) -> None:
    cmd = (
        f'{WB_COMMAND} -scene-capture-image "{scene_file}" {scene_index} "{out_png}" '
        f"-size-width-height {width} {height} -renderer {renderer}"
    )
    proc = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"wb_command failed for {scene_file}:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")


def main() -> int:
    args = parse_args()
    scene_path = Path(args.scene).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    base_tree = ET.parse(scene_path)
    base_root = base_tree.getroot()
    absolutize_scene_paths(base_root, scene_path)
    template_subject = detect_template_subject(base_root)

    for subject in args.subjects:
        root = ET.fromstring(ET.tostring(base_root, encoding="unicode"))
        replace_subject_text(root, template_subject, subject)
        if args.left_label_template or args.right_label_template:
            refs = find_current_structural_label_refs(root)
            if args.left_label_template:
                left_target = str(Path(args.left_label_template.format(subject=subject)).resolve())
                replace_label_refs(root, refs["L"], left_target)
            if args.right_label_template:
                right_target = str(Path(args.right_label_template.format(subject=subject)).resolve())
                replace_label_refs(root, refs["R"], right_target)
        validate_scene_paths(root)
        subject_dir = outdir / f"sub-{subject}"
        subject_dir.mkdir(parents=True, exist_ok=True)
        scene_out = subject_dir / f"sub-{subject}_template.scene"
        png_out = subject_dir / f"sub-{subject}_wb_{args.name}_native.png"
        if args.no_template_scene:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".scene",
                prefix=f"sub-{subject}_",
                dir=str(subject_dir),
                delete=False,
            ) as tmp:
                temp_scene = Path(tmp.name)
            try:
                ET.ElementTree(root).write(temp_scene, encoding="UTF-8", xml_declaration=True)
                capture_scene(temp_scene, args.scene_index, png_out, args.width, args.height, args.renderer)
            finally:
                temp_scene.unlink(missing_ok=True)
        else:
            ET.ElementTree(root).write(scene_out, encoding="UTF-8", xml_declaration=True)
            capture_scene(scene_out, args.scene_index, png_out, args.width, args.height, args.renderer)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

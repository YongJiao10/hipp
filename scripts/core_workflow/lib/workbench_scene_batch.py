#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from lib.output_layout import DEFAULT_HIPPUNFOLD_BATCH_ROOT
from lib.workbench import capture_scene


REPO_ROOT = Path(__file__).resolve().parents[3]
_OLD_BATCH = Path.home() / "Documents" / "HippoMaps" / "outputs" / "dense_corobl_batch"
_NEW_BATCH = DEFAULT_HIPPUNFOLD_BATCH_ROOT


def detect_template_subject(root: ET.Element) -> str:
    pattern = re.compile(r"sub-(\d+)")
    for elem in root.iter():
        if elem.text:
            match = pattern.search(elem.text)
            if match:
                return match.group(1)
    raise RuntimeError("Could not detect template subject in scene file")


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
        else:
            try:
                rel = resolved.relative_to(_OLD_BATCH)
                resolved = (_NEW_BATCH / rel).resolve()
            except ValueError:
                pass
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


def find_current_corobl_surface_refs(root: ET.Element) -> dict[str, str]:
    refs: dict[str, str] = {}
    for elem in root.iter():
        text = (elem.text or "").strip()
        if not text or "space-corobl" not in text or "midthickness.surf.gii" not in text:
            continue
        if "hemi-L" in text and "L" not in refs:
            refs["L"] = text
        elif "hemi-R" in text and "R" not in refs:
            refs["R"] = text
    if "L" not in refs or "R" not in refs:
        raise RuntimeError("Could not find current corobl midthickness surface references in scene file")
    return refs


def find_current_spec_ref(root: ET.Element) -> str:
    for elem in root.iter():
        text = (elem.text or "").strip()
        if "space-corobl_surfaces.spec" in text:
            return text
    raise RuntimeError("Could not find current corobl spec reference in scene file")


def replace_label_refs(root: ET.Element, current_ref: str, target_ref: str) -> None:
    current_name = Path(current_ref).name
    target_name = Path(target_ref).name
    for elem in root.iter():
        if elem.get("Name") and current_name in elem.get("Name", ""):
            elem.set("Name", elem.get("Name").replace(current_name, target_name))
        if not elem.text:
            continue
        if current_ref in elem.text:
            elem.text = elem.text.replace(current_ref, target_ref)
        if current_name in elem.text:
            elem.text = elem.text.replace(current_name, target_name)


def replace_exact_ref(root: ET.Element, current_ref: str, target_ref: str) -> None:
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
                if path.suffix in {".palette", ".scene", ".png", ".jpg", ".jpeg"}:
                    continue
                missing.append(str(path))
    if missing:
        preview = "\n".join(missing[:10])
        raise RuntimeError(f"Scene references missing files after subject replacement:\n{preview}")


def render_scene_batch(
    *,
    scene: Path,
    subjects: list[str],
    outdir: Path,
    scene_index: int = 1,
    width: int = 3200,
    height: int = 2048,
    renderer: str = "OSMesa",
    left_label_template: str | None = None,
    right_label_template: str | None = None,
    left_surface_template: str | None = None,
    right_surface_template: str | None = None,
    spec_template: str | None = None,
    name: str = "structural",
    no_template_scene: bool = False,
) -> list[Path]:
    scene_path = scene.resolve()
    outdir = outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    base_tree = ET.parse(scene_path)
    base_root = base_tree.getroot()
    absolutize_scene_paths(base_root, scene_path)
    template_subject = detect_template_subject(base_root)
    png_paths: list[Path] = []

    for subject in subjects:
        root = ET.fromstring(ET.tostring(base_root, encoding="unicode"))
        replace_subject_text(root, template_subject, subject)
        if left_label_template or right_label_template:
            refs = find_current_structural_label_refs(root)
            if left_label_template:
                left_target = str(Path(left_label_template.format(subject=subject)).resolve())
                replace_label_refs(root, refs["L"], left_target)
            if right_label_template:
                right_target = str(Path(right_label_template.format(subject=subject)).resolve())
                replace_label_refs(root, refs["R"], right_target)
        if left_surface_template or right_surface_template or spec_template:
            surface_refs = find_current_corobl_surface_refs(root)
            if left_surface_template:
                left_surface_target = str(Path(left_surface_template.format(subject=subject)).resolve())
                replace_exact_ref(root, surface_refs["L"], left_surface_target)
            if right_surface_template:
                right_surface_target = str(Path(right_surface_template.format(subject=subject)).resolve())
                replace_exact_ref(root, surface_refs["R"], right_surface_target)
            if spec_template:
                spec_target = str(Path(spec_template.format(subject=subject)).resolve())
                replace_exact_ref(root, find_current_spec_ref(root), spec_target)
        if not no_template_scene:
            validate_scene_paths(root)

        subject_dir = outdir / f"sub-{subject}"
        subject_dir.mkdir(parents=True, exist_ok=True)
        scene_out = subject_dir / f"sub-{subject}_template.scene"
        png_out = subject_dir / f"sub-{subject}_wb_{name}_native.png"
        if no_template_scene:
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
                capture_scene(temp_scene, scene_index, png_out, width, height, renderer)
            finally:
                temp_scene.unlink(missing_ok=True)
        else:
            ET.ElementTree(root).write(scene_out, encoding="UTF-8", xml_declaration=True)
            capture_scene(scene_out, scene_index, png_out, width, height, renderer)
        png_paths.append(png_out)
    return png_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render multiple subjects from a saved Workbench scene template.")
    parser.add_argument("--scene", required=True, help="Source .scene file saved from wb_view")
    parser.add_argument("--subjects", nargs="+", required=True, help="Subject IDs without sub- prefix")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--scene-index", type=int, default=1)
    parser.add_argument("--width", type=int, default=3200)
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--renderer", default="OSMesa")
    parser.add_argument("--left-label-template", default=None)
    parser.add_argument("--right-label-template", default=None)
    parser.add_argument("--left-surface-template", default=None)
    parser.add_argument("--right-surface-template", default=None)
    parser.add_argument("--spec-template", default=None)
    parser.add_argument("--name", default="structural")
    parser.add_argument("--no-template-scene", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    render_scene_batch(
        scene=Path(args.scene),
        subjects=[str(s) for s in args.subjects],
        outdir=Path(args.outdir),
        scene_index=int(args.scene_index),
        width=int(args.width),
        height=int(args.height),
        renderer=str(args.renderer),
        left_label_template=args.left_label_template,
        right_label_template=args.right_label_template,
        left_surface_template=args.left_surface_template,
        right_surface_template=args.right_surface_template,
        spec_template=args.spec_template,
        name=str(args.name),
        no_template_scene=bool(args.no_template_scene),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

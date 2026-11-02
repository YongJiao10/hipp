from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest


CORE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "core_workflow"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from lib import workbench_scene_batch as module


def test_scene_uses_current_structural_atlas_references() -> None:
    root = ET.fromstring(
        "<Scene><Object>sub-1001_hemi-L_space-corobl_atlas-multihist7sgi_subfields.label.gii</Object>"
        "<Object>sub-1001_hemi-R_space-corobl_atlas-multihist7sgi_subfields.label.gii</Object></Scene>"
    )
    refs = module.find_current_structural_label_refs(root)
    assert refs["L"].endswith("atlas-multihist7sgi_subfields.label.gii")
    assert refs["R"].endswith("atlas-multihist7sgi_subfields.label.gii")


def test_no_template_scene_can_retain_subject_specific_scene(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_scene = tmp_path / "template.scene"
    source_scene.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<Scene>
  <Object Type="label" Name="subject"><Text>sub-1001</Text></Object>
</Scene>
""",
        encoding="utf-8",
    )
    output_dir = tmp_path / "rendered"
    captured: list[Path] = []

    def fake_capture_scene(
        scene: Path,
        scene_index: int,
        png_out: Path,
        width: int,
        height: int,
        renderer: str,
    ) -> None:
        assert scene.exists()
        captured.append(scene)
        png_out.write_bytes(b"fake png")

    monkeypatch.setattr(module, "capture_scene", fake_capture_scene)

    png_paths = module.render_scene_batch(
        scene=source_scene,
        subjects=["2002"],
        outdir=output_dir,
        name="gradient_assignment",
        no_template_scene=True,
        retain_scene=True,
    )

    retained_scene = output_dir / "sub-2002" / "sub-2002_template.scene"
    assert png_paths == [
        output_dir / "sub-2002" / "sub-2002_wb_gradient_assignment_native.png"
    ]
    assert captured == [retained_scene]
    assert retained_scene.exists()
    assert "sub-2002" in retained_scene.read_text(encoding="utf-8")

"""Retired renderer behavior and the scene helpers retained by live consumers."""
from __future__ import annotations

import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import nibabel as nib
import numpy as np

from lib import workbench_scene_batch as scenes


def test_retired_heatmap_entrypoint_reports_replacement_without_rendering(monkeypatch, capsys):
    path = Path(__file__).resolve().parents[2] / "scripts/core_workflow/06_05_render_group_gradient_heatmaps.py"
    spec = importlib.util.spec_from_file_location("retired_heatmaps", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", [str(path)])
    assert module.main() == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "deprecated"
    assert "06_04_render_hippomaps_gradient_products.py" in payload["replacement"]


def test_shape_export_preserves_vertex_order_and_missing_values(tmp_path):
    path = tmp_path / "values.shape.gii"
    scenes.save_shape_gii(np.asarray([2., np.nan, -3.]), path)
    data = nib.load(path).darrays[0].data
    assert data.dtype == np.float32
    np.testing.assert_allclose(data, [2., np.nan, -3.], equal_nan=True)


def test_folded_scene_replaces_bilateral_assets_and_disables_label_overlay(tmp_path):
    source = tmp_path / "template.scene"
    paths = [f"sub-1001_hemi-{h}_space-corobl_den-512_label-hipp_{kind}"
             for h in ("L", "R") for kind in ("midthickness.surf.gii", "gyrification.shape.gii")]
    paths.append("sub-1001_space-corobl_surfaces.spec")
    root = ET.Element("Scene")
    for path in paths:
        ET.SubElement(root, "Object", Type="pathName").text = path
    overlay = ET.SubElement(root, "Object", Class="Overlay")
    ET.SubElement(overlay, "Object", Name="m_enabled").text = "true"
    ET.SubElement(overlay, "Object", Name="selectedMapFileNameWithPath").text = "anatomy.label.gii"
    ET.SubElement(root, "Object", Name="m_selectedPaletteName").text = "old"
    ET.ElementTree(root).write(source)
    assets = {key: tmp_path / name for key, name in (
        ("left_surface", "left.surf.gii"), ("right_surface", "right.surf.gii"),
        ("left_shape", "left.shape.gii"), ("right_shape", "right.shape.gii"),
        ("spec_path", "dense_corobl_batch/sub-2002/surf/sub-2002_surfaces.spec"),
    )}
    result = scenes.build_scene_root(scene_path=source, template_subject="2002",
        **assets, vmin=-2., vmax=2., palette_name="magma")
    refs = [obj.text for obj in result.findall("Object") if obj.get("Type") == "pathName"]
    assert refs == [str(assets[key]) for key in (
        "left_surface", "left_shape", "right_surface", "right_shape", "spec_path")]
    assert result.find(".//Object[@Name='m_enabled']").text == "false"
    assert result.find(".//Object[@Name='m_selectedPaletteName']").text == "magma"


def test_scene_capture_creates_project_temporary_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(scenes, 'REPO_ROOT', tmp_path / 'checkout')
    def capture(scene, index, output, width, height, renderer):
        assert scene.parent == tmp_path / 'checkout/tmp'
        assert ET.parse(scene).getroot().tag == 'Scene'
    monkeypatch.setattr(scenes, 'capture_scene', capture)
    scenes.capture_scene_root(root=ET.Element('Scene'), out_png=tmp_path / 'output/scene.png',
                              width=100, height=80, renderer='OSMesa')

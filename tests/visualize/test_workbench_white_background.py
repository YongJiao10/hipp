"""Validate Workbench white background rendering and adaptive grid compositing."""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_LIB = REPO_ROOT / "scripts/core_workflow/lib"
if str(CORE_LIB) not in sys.path:
    sys.path.insert(0, str(CORE_LIB))

from workbench_grid import (
    build_grid_canvas,
    foreground_mask,
    trim_background,
    trim_black,
)
from render_cortex_pfm_scene import ensure_scene_white_background


def test_scene_templates_default_to_white_background():
    """Verify both scene templates default to white background (255) and black foreground (0)."""
    scene_paths = [
        REPO_ROOT / "config/wb_locked_native_view_lateral_medial.scene",
        REPO_ROOT / "config/manual_wb_scenes/cortex_manual.scene",
    ]
    for sp in scene_paths:
        assert sp.is_file(), f"Missing scene file: {sp}"
        tree = ET.parse(sp)
        root = tree.getroot()
        found_helpers = 0
        for obj in root.iter("Object"):
            if obj.attrib.get("Class") == "BackgroundAndForegroundColorsSceneHelper":
                found_helpers += 1
                for arr in obj.findall("ObjectArray"):
                    name = arr.attrib.get("Name", "")
                    elements = [int(e.text.strip()) for e in arr.findall("Element")]
                    if "Background" in name:
                        assert elements == [255, 255, 255], f"{sp}: {name} is not [255, 255, 255]: {elements}"
                    elif "Foreground" in name:
                        assert elements == [0, 0, 0], f"{sp}: {name} is not [0, 0, 0]: {elements}"
        assert found_helpers > 0, f"No BackgroundAndForegroundColorsSceneHelper found in {sp}"


def test_ensure_scene_white_background_enforcement():
    """Verify programmatic enforcement of white background on any scene XML."""
    dummy_xml = """<SceneFile Version="3">
        <Scene>
            <Object Class="BackgroundAndForegroundColorsSceneHelper" Name="backgroundAndForegroundColors">
                <ObjectArray Name="m_colorForegroundWindow"><Element Index="0">255</Element><Element Index="1">255</Element><Element Index="2">255</Element></ObjectArray>
                <ObjectArray Name="m_colorBackgroundWindow"><Element Index="0">0</Element><Element Index="1">0</Element><Element Index="2">0</Element></ObjectArray>
            </Object>
        </Scene>
    </SceneFile>"""
    root = ET.fromstring(dummy_xml)
    ensure_scene_white_background(root)
    for arr in root.iter("ObjectArray"):
        name = arr.attrib.get("Name", "")
        elements = [int(e.text.strip()) for e in arr.findall("Element")]
        if "Background" in name:
            assert elements == [255, 255, 255]
        elif "Foreground" in name:
            assert elements == [0, 0, 0]


def test_adaptive_trim_and_mask_on_white_and_black():
    """Verify foreground_mask and trim_background adapt correctly to both background colors."""
    # 1. White background
    arr_w = np.full((100, 100, 4), 255, dtype=np.uint8)
    arr_w[30:70, 20:80, :3] = [50, 100, 150]
    im_w = Image.fromarray(arr_w)

    mask_w = foreground_mask(im_w)
    assert mask_w[0, 0] is np.bool_(False)
    assert mask_w[50, 50] is np.bool_(True)

    trimmed_w = trim_background(im_w)
    assert trimmed_w.size == (60, 40)

    # 2. Black background
    arr_b = np.zeros((100, 100, 4), dtype=np.uint8)
    arr_b[..., 3] = 255
    arr_b[30:70, 20:80, :3] = [50, 100, 150]
    im_b = Image.fromarray(arr_b)

    mask_b = foreground_mask(im_b)
    assert mask_b[0, 0] is np.bool_(False)
    assert mask_b[50, 50] is np.bool_(True)

    trimmed_b = trim_black(im_b)
    assert trimmed_b.size == (60, 40)


def test_build_grid_canvas_white_defaults():
    """Verify build_grid_canvas defaults to white canvas and dark text."""
    arr = np.full((200, 400, 4), 255, dtype=np.uint8)
    # create two simulated foreground hemispheres
    arr[50:150, 50:150, :3] = [200, 50, 50]
    arr[50:150, 250:350, :3] = [50, 50, 200]
    im = Image.fromarray(arr)

    canvas = build_grid_canvas(ventral=im, dorsal=None, layout="1x2")
    assert canvas.mode == "RGBA"
    # Corner pixel should be pure white
    assert canvas.getpixel((0, 0)) == (255, 255, 255, 255)

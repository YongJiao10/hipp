"""Test suite for Kong2022 vs Lynch2024 native and canonical 7-network comparison figures."""
from __future__ import annotations

from pathlib import Path
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
VIS_DIR = REPO_ROOT / "scripts" / "visualize" / "kong2022_standard_lateral"


def test_comparison_script_exists() -> None:
    script_path = VIS_DIR / "render_kong_lynch_native_canonical7.py"
    assert script_path.exists(), f"Missing render script: {script_path}"


def test_master_comparison_figure_exists_and_valid() -> None:
    fig_path = VIS_DIR / "kong_lynch_native_canonical7_comparison.png"
    assert fig_path.exists(), f"Missing master comparison figure: {fig_path}"
    with Image.open(fig_path) as im:
        assert im.size == (3250, 2109), f"Unexpected master figure dimensions: {im.size}"


def test_standalone_panels_exist_and_valid() -> None:
    panels = {
        "kong2022_standard_lateral.png": (1450, 900),
        "kong2022_standard_canonical7.png": (1560, 900),
        "lynch2024_standard_20net.png": (1560, 900),
        "lynch2024_standard_canonical7.png": (1560, 900),
    }
    for filename, expected_size in panels.items():
        panel_path = VIS_DIR / filename
        assert panel_path.exists(), f"Missing panel: {panel_path}"
        with Image.open(panel_path) as im:
            assert im.size == expected_size, f"Panel {filename} has size {im.size}, expected {expected_size}"


def test_native_to_canonical7_taxonomy_contract() -> None:
    from scripts.visualize.kong2022_standard_lateral.render_kong_lynch_native_canonical7 import (
        CANONICAL_7_ORDER,
        KONG_CAN7_CONSTITUENTS,
        LYNCH_CAN7_CONSTITUENTS,
    )

    assert len(CANONICAL_7_ORDER) == 7
    assert set(KONG_CAN7_CONSTITUENTS.keys()) == set(CANONICAL_7_ORDER)
    assert set(LYNCH_CAN7_CONSTITUENTS.keys()) == set(CANONICAL_7_ORDER)

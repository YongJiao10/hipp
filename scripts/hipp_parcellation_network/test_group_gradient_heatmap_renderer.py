#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_numbered_module(script_name: str, alias: str):
    module_path = REPO_ROOT / "scripts" / "core_workflow" / script_name
    spec = importlib.util.spec_from_file_location(alias, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    core_dir = (REPO_ROOT / "scripts" / "core_workflow").resolve()
    if str(core_dir) not in sys.path:
        sys.path.insert(0, str(core_dir))
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


mod = _load_numbered_module("06_05_render_group_gradient_heatmaps.py", "core_06_05_group_gradient_heatmaps")


def test_reconstruct_full_components_fills_nans_for_non_support_vertices() -> None:
    active = np.asarray([[1.0, -1.0], [2.0, -2.0]], dtype=np.float32)
    support = np.asarray([1, 0, 1], dtype=np.int32)
    full, support_count = mod.reconstruct_full_components(active=active, support_mask=support, vertex_count=3)
    assert int(support_count) == 2, support_count
    assert full.shape == (3, 2), full.shape
    assert math.isnan(float(full[1, 0])), full
    assert math.isnan(float(full[1, 1])), full
    assert float(full[0, 0]) == 1.0 and float(full[2, 0]) == 2.0, full


def test_compute_component_scales_is_symmetric_per_component() -> None:
    arr_a = np.asarray([[1.0, -2.0], [np.nan, 3.0]], dtype=np.float32)
    arr_b = np.asarray([[-4.0, 0.5], [2.5, np.nan]], dtype=np.float32)
    scales = mod.compute_component_scales(
        full_by_key={("2mm", "L"): arr_a, ("4mm", "R"): arr_b},
        component_names=["G1", "G2"],
    )
    assert scales["G1"][0] == -4.0 and scales["G1"][1] == 4.0, scales
    assert scales["G2"][0] == -3.0 and scales["G2"][1] == 3.0, scales


def test_component_ev_percentages_ignore_trivial_eigenvalue() -> None:
    got = mod.component_ev_percentages(
        {"gradient_eigenvalues": [1.0, 0.5, 0.25, 0.25]},
        ["G1", "G2", "G3"],
    )
    assert got == {"G1": 50.0, "G2": 25.0, "G3": 25.0}, got


def test_panel_ev_percentages_average_hemispheres_by_cell() -> None:
    got = mod.compute_panel_ev_percentages(
        {
            ("2mm", "L"): {"G1": 40.0, "G2": 20.0},
            ("2mm", "R"): {"G1": 30.0, "G2": 10.0},
        },
        smoothings=["2mm"],
        hemis=["L", "R"],
        components=["G1", "G2"],
    )
    assert got == {("2mm", "G1"): 35.0, ("2mm", "G2"): 15.0}, got


def test_cell_label_includes_ev_percent() -> None:
    assert mod.format_cell_label("G3", 12.34) == "G3 EV=12.3%"


def test_default_palette_uses_plasma() -> None:
    old_argv = sys.argv[:]
    try:
        sys.argv = ["render"]
        args = mod.parse_args()
    finally:
        sys.argv = old_argv
    assert args.palette == "plasma", args.palette
    assert args.colorbar_cmap == "plasma", args.colorbar_cmap


def test_validate_manifest_path_requires_group_root_and_rejects_archive() -> None:
    with TemporaryDirectory() as td:
        root = Path(td) / "root"
        root.mkdir(parents=True, exist_ok=True)
        inside = root / "gradient-spectral" / "kong2022" / "2mm" / "cov_0p80" / "hemi_L" / "group_atlas_manifest.json"
        inside.parent.mkdir(parents=True, exist_ok=True)
        inside.touch()
        mod.validate_manifest_path(inside, root)

        outside = Path(td) / "outside" / "group_atlas_manifest.json"
        outside.parent.mkdir(parents=True, exist_ok=True)
        outside.touch()
        try:
            mod.validate_manifest_path(outside, root)
            raise AssertionError("expected validate_manifest_path failure for outside path")
        except RuntimeError as exc:
            assert "within group root" in str(exc), exc

        archived = root / "x" / "archive" / "group_atlas_manifest.json"
        archived.parent.mkdir(parents=True, exist_ok=True)
        archived.touch()
        try:
            mod.validate_manifest_path(archived, root)
            raise AssertionError("expected validate_manifest_path failure for archive path")
        except RuntimeError as exc:
            assert "Archive path is forbidden" in str(exc), exc


def test_expected_output_name_contract() -> None:
    got = mod.expected_output_name()
    want = "kong2022_gradient-spectral_group_gradients_heatmap_routea_2mm4mm_g1g2g3.png"
    assert got == want, got


def main() -> int:
    test_reconstruct_full_components_fills_nans_for_non_support_vertices()
    test_compute_component_scales_is_symmetric_per_component()
    test_component_ev_percentages_ignore_trivial_eigenvalue()
    test_panel_ev_percentages_average_hemispheres_by_cell()
    test_cell_label_includes_ev_percent()
    test_default_palette_uses_plasma()
    test_validate_manifest_path_requires_group_root_and_rejects_archive()
    test_expected_output_name_contract()
    print("ok: group gradient heatmap renderer tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

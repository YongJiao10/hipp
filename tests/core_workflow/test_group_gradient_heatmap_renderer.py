#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image


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


def test_component_ev_percentages_use_full_exported_eigenvalue_sum() -> None:
    got = mod.component_ev_percentages(
        {"gradient_eigenvalues": [0.50, 0.25, 0.15, 0.07, 0.03]},
        ["G1", "G2", "G3"],
    )
    assert got == {"G1": 50.0, "G2": 25.0, "G3": 15.0}, got
    assert float(sum(got.values())) == 90.0, got


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


def test_default_palette_is_workbench_magma_and_no_colorbar_cmap_arg() -> None:
    old_argv = sys.argv[:]
    try:
        sys.argv = ["render"]
        args = mod.parse_args()
    finally:
        sys.argv = old_argv
    assert args.palette == "magma", args.palette
    assert not hasattr(args, "colorbar_cmap"), "renderer should not expose --colorbar-cmap"


def test_resolve_workbench_palette_name_strict_token_validation() -> None:
    assert mod.resolve_workbench_palette_name("magma") == "magma"
    assert mod.resolve_workbench_palette_name("RedWhiteBlue") == "RedWhiteBlue"
    try:
        mod.resolve_workbench_palette_name("")
        raise AssertionError("expected empty palette to fail")
    except ValueError as exc:
        assert "must not be empty" in str(exc)
    try:
        mod.resolve_workbench_palette_name("bad palette")
        raise AssertionError("expected invalid token to fail")
    except ValueError as exc:
        assert "Invalid Workbench palette token" in str(exc)


def test_verify_workbench_palette_supported_raises_without_fallback() -> None:
    old_apply = mod.apply_metric_palette

    def _fail(*args, **kwargs):
        raise RuntimeError("wb_command palette fail")

    mod.apply_metric_palette = _fail
    try:
        with TemporaryDirectory() as td:
            try:
                mod.verify_workbench_palette_supported("plasma", Path(td))
                raise AssertionError("expected unsupported palette failure")
            except ValueError as exc:
                assert "Unsupported Workbench palette" in str(exc), exc
    finally:
        mod.apply_metric_palette = old_apply


def test_target_colorbar_selector_enables_single_visible_metric_colorbar() -> None:
    root = mod.ET.fromstring(
        """
        <Root>
          <Object Type="class" Class="Overlay">
            <Object Type="pathName" Name="selectedMapFileNameWithPath">/tmp/left.shape.gii</Object>
            <Object Type="class" Class="Annotation" Name="m_colorBar">
              <Object Type="boolean" Name="m_displayedFlag">false</Object>
              <Object Type="boolean" Name="m_showTickMarksSelected">false</Object>
              <Object Type="string" Name="m_width">0</Object>
              <Object Type="string" Name="m_height">0</Object>
              <Object Type="string" Name="m_fontPercentViewportHeight">0</Object>
              <Object Type="string" Name="m_positionMode">MANUAL</Object>
            </Object>
          </Object>
          <Object Type="class" Class="Overlay">
            <Object Type="pathName" Name="selectedMapFileNameWithPath">/tmp/right.shape.gii</Object>
            <Object Type="class" Class="Annotation" Name="m_colorBar">
              <Object Type="boolean" Name="m_displayedFlag">false</Object>
              <Object Type="boolean" Name="m_showTickMarksSelected">false</Object>
              <Object Type="string" Name="m_width">0</Object>
              <Object Type="string" Name="m_height">0</Object>
              <Object Type="string" Name="m_fontPercentViewportHeight">0</Object>
              <Object Type="string" Name="m_positionMode">MANUAL</Object>
            </Object>
          </Object>
          <Object Type="class" Class="Overlay">
            <Object Type="pathName" Name="selectedMapFileNameWithPath">/tmp/other.shape.gii</Object>
            <Object Type="class" Class="Annotation" Name="m_colorBar">
              <Object Type="boolean" Name="m_displayedFlag">true</Object>
              <Object Type="boolean" Name="m_showTickMarksSelected">true</Object>
              <Object Type="string" Name="m_width">25</Object>
              <Object Type="string" Name="m_height">7</Object>
              <Object Type="string" Name="m_fontPercentViewportHeight">3.33</Object>
              <Object Type="string" Name="m_positionMode">AUTOMATIC</Object>
            </Object>
          </Object>
        </Root>
        """
    )

    old_capture = mod._capture_scene_root
    old_diff = mod._diff_bbox

    def _fake_capture(**kwargs):
        out = kwargs["out_png"]
        Image.new("RGB", (20, 20), (0, 0, 0)).save(out)

    def _fake_diff(baseline_png, probe_png, **kwargs):
        if str(probe_png).endswith("_00.png"):
            return (1, 2, 11, 12)
        return None

    mod._capture_scene_root = _fake_capture
    mod._diff_bbox = _fake_diff
    try:
        with TemporaryDirectory() as td:
            metric, chosen, diff_box = mod._select_visible_colorbar_overlay(
                root=root,
                target_metric_paths={"/tmp/left.shape.gii", "/tmp/right.shape.gii"},
                scratch_dir=Path(td),
                width=200,
                height=100,
                renderer="OSMesa",
            )
            assert metric == "/tmp/left.shape.gii", metric
            assert diff_box == (1, 2, 11, 12), diff_box
            mod._set_single_visible_colorbar(root, chosen)
            colorbars = mod._iter_colorbar_annotations(root)
            visible = [mod._child_text(cbar, "m_displayedFlag") for cbar in colorbars]
            assert visible.count("true") == 1, visible
    finally:
        mod._capture_scene_root = old_capture
        mod._diff_bbox = old_diff


def test_compose_figure_uses_external_workbench_colorbar_sources() -> None:
    with TemporaryDirectory() as td:
        root = Path(td)
        panel_paths: dict[tuple[str, str], Path] = {}
        panel_ev: dict[tuple[str, str], float] = {}
        for smooth in ("2mm", "4mm"):
            for comp in ("G1", "G2", "G3"):
                path = root / f"{smooth}_{comp}.png"
                img = Image.new("RGBA", (320, 220), (255, 255, 255, 255))
                arr = np.asarray(img).copy()
                arr[70:150, 95:225, 0:3] = 0
                Image.fromarray(arr, mode="RGBA").save(path)
                panel_paths[(smooth, comp)] = path
                panel_ev[(smooth, comp)] = 10.0

        colorbar_paths: dict[str, Path] = {}
        colorbar_boxes: dict[str, tuple[int, int, int, int] | None] = {}
        for comp, rgb in (("G1", (255, 0, 0)), ("G2", (0, 255, 0)), ("G3", (0, 0, 255))):
            src = root / f"{comp}_source.png"
            img = Image.new("RGBA", (500, 220), (255, 255, 255, 255))
            arr = np.asarray(img).copy()
            # Simulate a bad diff crop in the lower-left plus the real Workbench
            # horizontal colorbar elsewhere in the source render.
            arr[170:210, 15:120, 0:3] = 20
            arr[120:142, 250:455, 0:3] = np.asarray(rgb, dtype=np.uint8)
            arr[150:164, 250:455, 0:3] = 0
            Image.fromarray(arr, mode="RGBA").save(src)
            colorbar_paths[comp] = src
            colorbar_boxes[comp] = (15, 170, 120, 210)

        out_png = root / "out.png"
        width, height = mod.compose_figure(
            panel_paths=panel_paths,
            colorbar_paths=colorbar_paths,
            colorbar_crop_boxes=colorbar_boxes,
            panel_ev_percentages=panel_ev,
            smoothings=["2mm", "4mm"],
            components=["G1", "G2", "G3"],
            output_png=out_png,
        )

        assert (width, height) == Image.open(out_png).size
        assert width <= 700, width
        assert height <= 360, height

        arr = np.asarray(Image.open(out_png).convert("RGB"))
        lower = arr[int(height * 0.74) :, :, :]
        nonwhite = np.any(lower < 245, axis=2)
        assert int(nonwhite.sum()) > 100, "expected visible bottom colorbars from external sources"
        colorful = (lower.max(axis=2).astype(int) - lower.min(axis=2).astype(int)) > 80
        assert int(colorful.sum()) > 1000, "expected real colorbar color, not only tick labels"


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
    test_component_ev_percentages_use_full_exported_eigenvalue_sum()
    test_panel_ev_percentages_average_hemispheres_by_cell()
    test_cell_label_includes_ev_percent()
    test_default_palette_is_workbench_magma_and_no_colorbar_cmap_arg()
    test_resolve_workbench_palette_name_strict_token_validation()
    test_verify_workbench_palette_supported_raises_without_fallback()
    test_target_colorbar_selector_enables_single_visible_metric_colorbar()
    test_compose_figure_uses_external_workbench_colorbar_sources()
    test_validate_manifest_path_requires_group_root_and_rejects_archive()
    test_expected_output_name_contract()
    print("ok: group gradient heatmap renderer tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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


def _load_module(relpath: str, alias: str):
    module_path = REPO_ROOT / relpath
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


adapter = _load_module("scripts/core_workflow/lib/hippomaps_adapter.py", "core_hippomaps_adapter")
compute_mod = _load_module(
    "scripts/core_workflow/05_03_compute_hippomaps_gradients.py",
    "core_05_03_compute_hippomaps_gradients",
)
render_mod = _load_module(
    "scripts/core_workflow/06_05_render_hippomaps_gradient_products.py",
    "core_06_05_render_hippomaps_gradient_products",
)


def test_group_mean_fc_applies_fisher_z_before_averaging() -> None:
    support = np.asarray([True, False, True], dtype=bool)
    subj_a = np.asarray(
        [
            [0.0, 0.5],
            [0.9, 0.9],
            [np.nan, -0.5],
        ],
        dtype=np.float32,
    )
    subj_b = np.asarray(
        [
            [0.5, 0.0],
            [0.8, 0.8],
            [0.5, np.nan],
        ],
        dtype=np.float32,
    )
    got = compute_mod.group_mean_fisherz_fc(
        feature_bank={"a": subj_a, "b": subj_b},
        subjects=["a", "b"],
        support_mask=support,
    )
    expected = np.nanmean(
        np.stack(
            [
                np.arctanh(np.clip(subj_a[support], -0.999999, 0.999999)),
                np.arctanh(np.clip(subj_b[support], -0.999999, 0.999999)),
            ],
            axis=0,
        ),
        axis=0,
    )
    expected[~np.isfinite(expected)] = 0.0
    assert np.allclose(got, expected.astype(np.float32), atol=1e-6), (got, expected)


def test_gradientmaps_called_with_official_pearson_kernel_and_full_lambda_ev() -> None:
    calls: list[dict[str, object]] = []

    class FakeGradientMaps:
        def __init__(self, **kwargs):
            calls.append({"init": kwargs})
            self.gradients_ = np.asarray(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0],
                ],
                dtype=np.float32,
            )
            self.lambdas_ = np.asarray([5.0, 3.0, 2.0], dtype=np.float32)

        def fit(self, matrix):
            calls.append({"fit_shape": tuple(matrix.shape), "fit_matrix": np.asarray(matrix).copy()})
            return self

    raw = np.asarray(
        [
            [1.0, 10.0, 3.0, 5.0],
            [2.0, 20.0, 3.0, 5.0],
            [3.0, 30.0, 3.0, 5.0],
            [4.0, 40.0, 3.0, 5.0],
        ],
        dtype=np.float32,
    )
    gradients, lambdas, ev = adapter.fit_gradientmaps_pearson(
        raw,
        n_components=3,
        gradient_maps_cls=FakeGradientMaps,
    )
    assert calls[0]["init"] == {"n_components": 3, "kernel": "pearson"}, calls
    assert calls[1]["fit_shape"] == (4, 4), calls
    expected = adapter.feature_zscore_columns(raw)
    assert np.allclose(calls[1]["fit_matrix"], expected, atol=1e-6), calls[1]["fit_matrix"]
    assert np.allclose(calls[1]["fit_matrix"][:, 2:], 0.0), calls[1]["fit_matrix"]
    assert gradients.shape == (4, 3), gradients.shape
    assert np.allclose(lambdas, [5.0, 3.0, 2.0])
    assert np.allclose(ev, [0.5, 0.3, 0.2])


def test_full_surface_reconstruction_keeps_nan_outside_support() -> None:
    active = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    support = np.asarray([False, True, False, True], dtype=bool)
    full = adapter.reconstruct_full_surface(active, support, vertex_count=4)
    assert full.shape == (4, 2), full.shape
    assert math.isnan(float(full[0, 0])), full
    assert math.isnan(float(full[2, 1])), full
    assert np.allclose(full[support], active)


def test_endpoint_fc_contrast_uses_top_minus_bottom_gradient_quartiles() -> None:
    gradient_full = np.asarray([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    group_fc_full = np.asarray(
        [
            [np.nan, np.nan],
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ],
        dtype=np.float32,
    )
    support = np.asarray([False, True, True, True, True], dtype=bool)
    got = render_mod.endpoint_fc_contrast(
        gradient_full=gradient_full,
        group_fc_full=group_fc_full,
        support_mask=support,
        endpoint_fraction=0.25,
    )
    assert np.allclose(got, [3.0, 30.0]), got


def test_render_gradient_figure_calls_official_hippomaps_plotter_options() -> None:
    calls: list[dict[str, object]] = []

    class FakeFigure:
        def savefig(self, path, **kwargs):
            calls.append({"savefig": Path(path).name, "kwargs": kwargs})

    def fake_plotter(cdata, **kwargs):
        calls.append({"shape": tuple(cdata.shape), "kwargs": kwargs})
        return FakeFigure()

    out = REPO_ROOT / "present_group_t1only" / "hippomaps_renderer_test.png"
    render_mod.render_gradient_figure(
        left=np.ones((4, 3), dtype=np.float32),
        right=np.zeros((4, 3), dtype=np.float32),
        output_png=out,
        den="512",
        plotter=fake_plotter,
    )
    assert calls[0]["shape"] == (4, 2, 3), calls
    assert calls[0]["kwargs"]["cmap"] == "plasma", calls
    assert calls[0]["kwargs"]["share"] == "row", calls
    assert calls[0]["kwargs"]["color_bar"] == "right", calls
    assert calls[0]["kwargs"]["unfoldAPrescale"] is False, calls
    assert calls[0]["kwargs"]["tighten_cwindow"] is True, calls
    assert calls[0]["kwargs"]["labels"] == ["hipp"], calls
    assert calls[0]["kwargs"]["den"] == "512", calls
    assert calls[1]["savefig"] == "hippomaps_renderer_test.png", calls


def test_render_gradient_figure_can_render_left_hemi_only() -> None:
    calls: list[dict[str, object]] = []

    class FakeFigure:
        def savefig(self, path, **kwargs):
            calls.append({"savefig": Path(path).name, "kwargs": kwargs})

    def fake_plotter(cdata, **kwargs):
        calls.append({"shape": tuple(cdata.shape), "kwargs": kwargs})
        return FakeFigure()

    out = REPO_ROOT / "present_group_t1only" / "hippomaps_renderer_test_left.png"
    render_mod.render_gradient_figure(
        left=np.ones((4, 3), dtype=np.float32),
        output_png=out,
        den="512",
        hemis=["L"],
        plotter=fake_plotter,
    )

    assert calls[0]["shape"] == (4, 1, 3), calls
    assert calls[0]["kwargs"]["hemis"] == ["L"], calls
    assert calls[1]["savefig"] == "hippomaps_renderer_test_left.png", calls


def test_annotate_gradient_image_adds_left_margin_and_ev_labels() -> None:
    with TemporaryDirectory() as td:
        img_path = Path(td) / "base.png"
        Image.new("RGB", (100, 60), (255, 255, 255)).save(img_path)
        render_mod.annotate_gradient_image(
            img_path,
            selected_components=[1, 2, 3],
            ev_percentages=[41.2, 8.4, 4.0],
        )
        out = Image.open(img_path).convert("RGB")
        assert out.size == (330, 60), out.size
        assert out.getpixel((10, 10)) == (255, 255, 255), out.getpixel((10, 10))
        assert out.getpixel((20, 20)) != (255, 255, 255), "expected text pixels in the left label margin"


def main() -> int:
    test_group_mean_fc_applies_fisher_z_before_averaging()
    test_gradientmaps_called_with_official_pearson_kernel_and_full_lambda_ev()
    test_full_surface_reconstruction_keeps_nan_outside_support()
    test_endpoint_fc_contrast_uses_top_minus_bottom_gradient_quartiles()
    test_render_gradient_figure_calls_official_hippomaps_plotter_options()
    test_render_gradient_figure_can_render_left_hemi_only()
    test_annotate_gradient_image_adds_left_margin_and_ev_labels()
    print("ok: hippomaps gradient route checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

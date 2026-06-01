#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

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
    if "lib/" in relpath:
        module.__package__ = "lib"
    core_dir = (REPO_ROOT / "scripts" / "core_workflow").resolve()
    if str(core_dir) not in sys.path:
        sys.path.insert(0, str(core_dir))
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


adapter = _load_module("scripts/core_workflow/lib/hippomaps_adapter.py", "core_hippomaps_adapter")
compute_mod = _load_module(
    "scripts/core_workflow/05_01_compute_hippomaps_gradients.py",
    "core_05_01_compute_hippomaps_gradients",
)
render_mod = _load_module(
    "scripts/core_workflow/06_05_render_hippomaps_gradient_products.py",
    "core_06_05_render_hippomaps_gradient_products",
)
atlas_mod = _load_module(
    "scripts/core_workflow/05_02_run_group_atlas.py",
    "core_05_02_run_group_atlas",
)
overview_mod = _load_module(
    "scripts/core_workflow/06_01_build_group_overview.py",
    "core_06_01_build_group_overview",
)
matrix_mod = _load_module(
    "scripts/core_workflow/07_01_build_group_matrix.py",
    "core_07_01_build_group_matrix",
)
cortex_mod = _load_module(
    "scripts/core_workflow/06_04_build_cluster_cortex_panels.py",
    "core_06_04_build_cluster_cortex_panels",
)
output_layout = _load_module(
    "scripts/core_workflow/lib/output_layout.py",
    "core_output_layout",
)
smoke_mod = _load_module(
    "tests/core_workflow/smoke_prepare_run.py",
    "core_smoke_prepare_run",
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
    with patch("brainspace.gradient.GradientMaps", FakeGradientMaps):
        gradients, lambdas, ev = adapter.fit_gradientmaps_pearson(
            raw,
            n_components=3,
        )
    assert calls[0]["init"] == {"n_components": 3, "kernel": "pearson", "random_state": 0}, calls
    assert calls[1]["fit_shape"] == (4, 4), calls
    expected = adapter.feature_zscore_columns(raw)
    assert np.allclose(calls[1]["fit_matrix"], expected, atol=1e-6), calls[1]["fit_matrix"]
    assert np.allclose(calls[1]["fit_matrix"][:, 2:], 0.0), calls[1]["fit_matrix"]
    assert gradients.shape == (4, 3), gradients.shape
    assert np.allclose(lambdas, [5.0, 3.0, 2.0])
    assert np.allclose(ev, [0.5, 0.3, 0.2])


def test_gradientmaps_accepts_explicit_random_state() -> None:
    calls: list[dict[str, object]] = []

    class FakeGradientMaps:
        def __init__(self, **kwargs):
            calls.append({"init": kwargs})
            self.gradients_ = np.ones((4, 2), dtype=np.float32)
            self.lambdas_ = np.asarray([3.0, 1.0], dtype=np.float32)

        def fit(self, matrix):
            return self

    raw = np.arange(16, dtype=np.float32).reshape(4, 4)
    with patch("brainspace.gradient.GradientMaps", FakeGradientMaps):
        adapter.fit_gradientmaps_pearson(
            raw,
            n_components=2,
            random_state=17,
        )

    assert calls[0]["init"] == {"n_components": 2, "kernel": "pearson", "random_state": 17}, calls


def test_group_atlas_gradient_spectral_uses_hippomaps_adapter_and_records_provenance() -> None:
    feature_bank = {
        "a": np.asarray(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
                [1.0, 1.1, 1.2],
                [1.3, 1.4, 1.5],
                [1.6, 1.7, 1.8],
                [1.9, 2.0, 2.1],
            ],
            dtype=np.float32,
        ),
        "b": np.asarray(
            [
                [0.2, 0.1, 0.0],
                [0.6, 0.5, 0.4],
                [1.0, 0.9, 0.8],
                [1.4, 1.3, 1.2],
                [1.8, 1.7, 1.6],
                [2.2, 2.1, 2.0],
                [2.6, 2.5, 2.4],
            ],
            dtype=np.float32,
        ),
    }
    support_mask = np.asarray([True, True, False, True, True, True, True], dtype=bool)
    calls: dict[str, np.ndarray | int] = {}

    def fake_fit(group_fc: np.ndarray, *, n_components: int):
        calls["group_fc"] = np.asarray(group_fc, dtype=np.float32).copy()
        calls["n_components"] = int(n_components)
        gradients = np.asarray(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0, 9.0, 10.0],
                [11.0, 12.0, 13.0, 14.0, 15.0],
                [16.0, 17.0, 18.0, 19.0, 20.0],
                [21.0, 22.0, 23.0, 24.0, 25.0],
                [26.0, 27.0, 28.0, 29.0, 30.0],
            ],
            dtype=np.float32,
        )
        lambdas = np.asarray([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float64)
        ev = lambdas / np.sum(lambdas)
        return gradients, lambdas, ev

    expected_group_fc = np.nanmean(
        np.stack([feature_bank["a"][support_mask], feature_bank["b"][support_mask]], axis=0),
        axis=0,
    ).astype(np.float32)

    with (
        patch.object(atlas_mod, "ensure_hippomaps_on_path", return_value=Path("/mock/hippomaps")),
        patch.object(atlas_mod, "hippomaps_import_path", return_value="/mock/hippomaps/hippomaps"),
        patch.object(atlas_mod, "hippomaps_commit", return_value="deadbeef"),
    ):
        selected, meta = atlas_mod.build_group_gradient_features(
            feature_bank=feature_bank,
            subjects_subset=["a", "b"],
            support_mask=support_mask,
            gradient_components=(1, 3),
            hippomaps_root=Path("/mock/hippomaps"),
            gradientmaps_fit_fn=fake_fit,
        )

    assert int(calls["n_components"]) == atlas_mod.GRADIENT_N_COMPONENTS_COMPUTED
    assert np.allclose(calls["group_fc"], expected_group_fc, atol=1e-6)
    assert selected.shape == (6, 2), selected.shape
    assert np.allclose(selected[:, 0], [1.0, 6.0, 11.0, 16.0, 21.0, 26.0], atol=1e-6)
    assert np.allclose(selected[:, 1], [3.0, 8.0, 13.0, 18.0, 23.0, 28.0], atol=1e-6)
    assert meta["gradientmaps_call"] == (
        "brainspace.gradient.GradientMaps(n_components=N, kernel='pearson', random_state=gradient_random_state).fit(column_zscored_group_mean_fisherz_fc)"
    )
    assert meta["gradient_random_state"] == 0
    assert meta["feature_zscore_pre_gradientmaps"] == (
        "column-wise z-score across vertices after group Fisher-z averaging"
    )
    assert meta["gradient_component_labels"] == ["G1", "G3"]
    assert meta["selected_components"] == ["G1", "G3"]
    assert meta["hippomaps_repo"] == "/mock/hippomaps"
    assert meta["hippomaps_import_path"] == "/mock/hippomaps/hippomaps"
    assert meta["hippomaps_commit"] == "deadbeef"
    assert np.allclose(meta["gradient_eigenvalues"], [5.0, 4.0, 3.0, 2.0, 1.0], atol=1e-6)
    assert np.allclose(meta["ev_fractions"], [1.0 / 3.0, 4.0 / 15.0, 1.0 / 5.0, 2.0 / 15.0, 1.0 / 15.0], atol=1e-6)


def test_group_atlas_public_methods_drop_partial_corr_branches() -> None:
    assert "group-network-partial-spectral" not in atlas_mod.METHOD_CHOICES
    assert "group-intrinsic-partial-spectral" not in atlas_mod.METHOD_CHOICES
    assert not atlas_mod.is_network_method("group-network-partial-spectral")


def test_group_atlas_parser_defaults_to_fixed_k3() -> None:
    parser = atlas_mod.build_arg_parser()
    args = parser.parse_args([])
    assert int(args.k_min) == 2
    assert int(args.k_max) == 10
    assert int(args.force_final_k) == 3


def test_group_atlas_gradient_spectral_loads_hippomaps_gradient_artifacts() -> None:
    with TemporaryDirectory() as td:
        root = Path(td)
        run_dir = root / "routea" / "2mm" / "hemi_L"
        run_dir.mkdir(parents=True)
        gradients_full = np.asarray(
            [
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0, 9.0, 10.0],
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [11.0, 12.0, 13.0, 14.0, 15.0],
            ],
            dtype=np.float32,
        )
        np.save(run_dir / "hippomaps_gradients_full.npy", gradients_full)
        manifest = {
            "status": "ok",
            "route": "routea",
            "smoothing": "2mm",
            "hemi": "L",
            "n_components_computed": 5,
            "gradientmaps_call": "brainspace.gradient.GradientMaps(n_components=N, kernel='pearson').fit(column_zscored_group_mean_fisherz_fc)",
            "gradient_alignment": {
                "enabled": True,
                "anchor_route": "routea",
                "anchor_smoothing": "4mm",
                "anchor_hemi": "L",
                "raw_arrays_saved": False,
            },
            "outputs": {
                "hippomaps_gradients_full": str((run_dir / "hippomaps_gradients_full.npy").resolve()),
            },
        }
        (run_dir / "hippomaps_gradient_manifest.json").write_text(
            atlas_mod.json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        support_mask = np.asarray([False, True, True, False, True], dtype=bool)

        selected, meta = atlas_mod.load_hippomaps_gradient_features(
            gradient_root=root,
            route="routea",
            smoothing="2mm",
            hemi="L",
            support_mask=support_mask,
            gradient_components=(1, 3),
        )

    assert selected.shape == (3, 2), selected.shape
    assert np.allclose(selected[:, 0], [1.0, 6.0, 11.0], atol=1e-6)
    assert np.allclose(selected[:, 1], [3.0, 8.0, 13.0], atol=1e-6)
    assert meta["gradient_source"] == "hippomaps_gradient_artifacts"
    assert meta["gradient_component_labels"] == ["G1", "G3"]
    assert meta["gradient_alignment"]["enabled"] is True
    assert meta["gradient_alignment"]["raw_arrays_saved"] is False
    assert meta["hippomaps_gradient_manifest"].endswith("hippomaps_gradient_manifest.json")




def test_group_atlas_gradient_loader_prefers_local_artifact_after_relocation() -> None:
    with TemporaryDirectory() as td:
        root = Path(td)
        run_dir = root / "routea" / "2mm" / "hemi_L"
        run_dir.mkdir(parents=True)
        np.save(run_dir / "hippomaps_gradients_full.npy", np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32))
        stale_path = root / "old_location" / "hippomaps_gradients_full.npy"
        manifest = {
            "status": "ok",
            "n_components_computed": 3,
            "outputs": {"hippomaps_gradients_full": str(stale_path.resolve())},
        }
        (run_dir / "hippomaps_gradient_manifest.json").write_text(atlas_mod.json.dumps(manifest), encoding="utf-8")
        selected, meta = atlas_mod.load_hippomaps_gradient_features(
            gradient_root=root,
            route="routea",
            smoothing="2mm",
            hemi="L",
            support_mask=np.asarray([True], dtype=bool),
            gradient_components=(1, 3),
        )
    assert np.allclose(selected, [[1.0, 3.0]])
    assert meta["hippomaps_gradients_full"].endswith("routea/2mm/hemi_L/hippomaps_gradients_full.npy")

def test_canonical_public_surfaces_drop_partial_corr_branches() -> None:
    for mod in (atlas_mod, overview_mod, cortex_mod):
        assert "group-network-partial-spectral" not in mod.METHOD_CHOICES
        assert "group-intrinsic-partial-spectral" not in mod.METHOD_CHOICES


def test_canonical_present_defaults_use_outputs_core_layout() -> None:
    expected_root = REPO_ROOT / "outputs_core" / "present"
    assert overview_mod.DEFAULT_PRESENT_DIR == expected_root / "overviews"
    assert matrix_mod.DEFAULT_PRESENT_DIR == expected_root / "overviews"
    assert cortex_mod.DEFAULT_PRESENT_DIR == expected_root / "cortex_panels"
    assert render_mod.DEFAULT_PRESENT_ROOT == expected_root / "gradients"
    for path in (
        overview_mod.DEFAULT_PRESENT_DIR,
        matrix_mod.DEFAULT_PRESENT_DIR,
        cortex_mod.DEFAULT_PRESENT_DIR,
        render_mod.DEFAULT_PRESENT_ROOT,
    ):
        text = str(path)
        assert "present_group" not in text
        assert "present_fixedk" not in text
        assert "t1only" not in text
        assert "hcp7t_hippocampus" not in text
        assert "/runs/" not in text
        assert "/manual/" not in text


def test_run_layout_keeps_smoke_outputs_in_dedicated_run_tree() -> None:
    layout = output_layout.run_layout("smoke5_test")
    assert layout["run_root"] == REPO_ROOT / "outputs_core" / "smoke_runs" / "smoke5_test"
    assert layout["overviews"] == layout["present"] / "overviews"
    assert layout["gradient_products"] == layout["present"] / "gradients"
    assert layout["cortex_panels"] == layout["present"] / "cortex_panels"


def test_smoke_command_plan_uses_isolated_fixedk_full_layout() -> None:
    base = REPO_ROOT / "outputs_core" / "smoke_runs" / "smoke5_test"
    present = base / "present"
    layout = {
        "run_root": base,
        "inputs": base / "inputs",
        "logs": base / "logs",
        "manifests": base / "manifests",
        "hippunfold": base / "hippunfold",
        "features": base / "features",
        "group_atlas": base / "group_atlas",
        "gradients": base / "gradients",
        "present": present,
        "overviews": present / "overviews",
        "cortex_panels": present / "cortex_panels",
        "gradient_products": present / "gradients",
    }
    records = smoke_mod.build_command_plan(
        subjects=["100610", "102311", "104416", "105923", "108323"],
        layout=layout,
        python_exe="/home/yojiao/.conda/envs/py314/bin/python",
        input_dir=REPO_ROOT / "data" / "hippunfold_input",
        bold_root=REPO_ROOT / "bold",
    )
    joined = "\n".join(" ".join(row["command"]) for row in records)
    assert "present_group" not in joined
    assert "present_fixedk" not in joined
    assert "10perc" not in joined
    assert "--force-final-k 3" in joined
    assert "--k-min 2 --k-max 10" in joined
    assert str(layout["overviews"]) in joined
    assert "group-network-partial-spectral" not in joined
    assert "group-intrinsic-partial-spectral" not in joined

def test_compute_defaults_to_five_components_like_hippomaps_paper() -> None:
    with patch.object(sys, "argv", ["05_01_compute_hippomaps_gradients.py"]):
        args = compute_mod.parse_args()
    assert int(args.n_components) == 5


def test_full_surface_reconstruction_keeps_nan_outside_support() -> None:
    active = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    support = np.asarray([False, True, False, True], dtype=bool)
    full = adapter.reconstruct_full_surface(active, support, vertex_count=4)
    assert full.shape == (4, 2), full.shape
    assert math.isnan(float(full[0, 0])), full
    assert math.isnan(float(full[2, 1])), full
    assert np.allclose(full[support], active)


def test_gradient_alignment_reorders_components_and_flips_sign_to_anchor() -> None:
    anchor_full = np.asarray(
        [
            [-1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, -1.0],
        ],
        dtype=np.float32,
    )
    candidate_full = np.column_stack([-anchor_full[:, 1], anchor_full[:, 0]]).astype(np.float32)
    aligned_active, aligned_full, aligned_lambdas, aligned_ev, meta = compute_mod.align_gradient_components_to_reference(
        active=candidate_full.copy(),
        full=candidate_full.copy(),
        lambdas=np.asarray([20.0, 10.0], dtype=np.float64),
        ev_fractions=np.asarray([2.0 / 3.0, 1.0 / 3.0], dtype=np.float64),
        reference_full=anchor_full,
        component_labels=["G1", "G2"],
    )

    assert np.allclose(aligned_active, anchor_full, atol=1e-6), aligned_active
    assert np.allclose(aligned_full, anchor_full, atol=1e-6), aligned_full
    assert np.allclose(aligned_lambdas, [10.0, 20.0], atol=1e-6), aligned_lambdas
    assert np.allclose(aligned_ev, [1.0 / 3.0, 2.0 / 3.0], atol=1e-6), aligned_ev
    assert meta["component_mapping"] == {"G1": "G2", "G2": "G1"}, meta
    assert meta["sign_flip_applied"] == {"G1": False, "G2": True}, meta
    assert meta["raw_arrays_saved"] is False, meta


def test_gradient_alignment_anchor_combo_is_identity_without_raw_outputs() -> None:
    anchor_full = np.asarray(
        [
            [1.0, -2.0, 0.0],
            [2.0, -1.0, 1.0],
            [3.0, 0.0, 2.0],
            [4.0, 1.0, 3.0],
        ],
        dtype=np.float32,
    )
    aligned_active, aligned_full, aligned_lambdas, aligned_ev, meta = compute_mod.align_gradient_components_to_reference(
        active=anchor_full.copy(),
        full=anchor_full.copy(),
        lambdas=np.asarray([5.0, 4.0, 3.0], dtype=np.float64),
        ev_fractions=np.asarray([5.0 / 12.0, 4.0 / 12.0, 3.0 / 12.0], dtype=np.float64),
        reference_full=anchor_full,
        component_labels=["G1", "G2", "G3"],
    )

    assert np.allclose(aligned_active, anchor_full, atol=1e-6), aligned_active
    assert np.allclose(aligned_full, anchor_full, atol=1e-6), aligned_full
    assert np.allclose(aligned_lambdas, [5.0, 4.0, 3.0], atol=1e-6), aligned_lambdas
    assert np.allclose(aligned_ev, [5.0 / 12.0, 4.0 / 12.0, 3.0 / 12.0], atol=1e-6), aligned_ev
    assert meta["component_mapping"] == {"G1": "G1", "G2": "G2", "G3": "G3"}, meta
    assert meta["sign_flip_applied"] == {"G1": False, "G2": False, "G3": False}, meta
    assert meta["raw_arrays_saved"] is False, meta


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
            Image.new("RGB", (24, 24), (255, 255, 255)).save(path)

    def fake_plotter(cdata, **kwargs):
        calls.append({"shape": tuple(cdata.shape), "kwargs": kwargs})
        return FakeFigure()

    out = REPO_ROOT / "outputs_core" / "reruns" / "test_hippomaps_renderer" / "hippomaps_renderer_test.png"
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
    assert calls[1]["savefig"].endswith(".png"), calls
    assert Path(calls[1]["savefig"]).name != "", calls


def test_render_gradient_figure_can_render_left_hemi_only() -> None:
    calls: list[dict[str, object]] = []

    class FakeFigure:
        def savefig(self, path, **kwargs):
            calls.append({"savefig": Path(path).name, "kwargs": kwargs})
            Image.new("RGB", (24, 24), (255, 255, 255)).save(path)

    def fake_plotter(cdata, **kwargs):
        calls.append({"shape": tuple(cdata.shape), "kwargs": kwargs})
        return FakeFigure()

    out = REPO_ROOT / "outputs_core" / "reruns" / "test_hippomaps_renderer" / "hippomaps_renderer_test_left.png"
    render_mod.render_gradient_figure(
        left=np.ones((4, 3), dtype=np.float32),
        output_png=out,
        den="512",
        hemis=["L"],
        plotter=fake_plotter,
    )

    assert calls[0]["shape"] == (4, 1, 3), calls
    assert calls[0]["kwargs"]["hemis"] == ["L"], calls
    assert calls[1]["savefig"].endswith(".png"), calls
    assert Path(calls[1]["savefig"]).name != "", calls


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
        assert out.size == (330, 120), out.size
        # Check that the pasted base image is placed on the right (at margin_w=230)
        assert out.getpixel((240, 40)) == (255, 255, 255), out.getpixel((240, 40))
        # Search the left margin for any non-white text pixels
        found_non_white = False
        for x in range(230):
            for y_px in range(out.size[1]):
                if out.getpixel((x, y_px)) != (255, 255, 255):
                    found_non_white = True
                    break
            if found_non_white:
                break
        assert found_non_white, "expected text pixels in the left label margin"


def test_ensure_bilateral_gradient_annotations_repairs_late_raw_screenshot() -> None:
    with TemporaryDirectory() as td:
        root = Path(td)
        left = root / "left.png"
        right = root / "right.png"
        final = root / "bilateral.png"
        Image.new("RGB", (510, 90), (255, 255, 255)).save(left)
        Image.new("RGB", (510, 90), (255, 255, 255)).save(right)
        Image.new("RGB", (1020, 90), (255, 255, 255)).save(final)
        render_mod.ensure_bilateral_gradient_annotations(
            side_paths={"L": left, "R": right},
            output_png=final,
            selected_components=[1, 2, 3],
            ev_by_hemi={"L": [10.0, 5.0, 2.5], "R": [9.0, 4.0, 2.0]},
            settle_seconds=0.0,
        )
        assert Image.open(left).size == (740, 150)
        assert Image.open(right).size == (740, 150)
        assert Image.open(final).size == (1480, 150)


def main() -> int:
    test_group_mean_fc_applies_fisher_z_before_averaging()
    test_gradientmaps_called_with_official_pearson_kernel_and_full_lambda_ev()
    test_gradientmaps_accepts_explicit_random_state()
    test_group_atlas_gradient_spectral_uses_hippomaps_adapter_and_records_provenance()
    test_group_atlas_public_methods_drop_partial_corr_branches()
    test_group_atlas_parser_defaults_to_fixed_k3()
    test_group_atlas_gradient_spectral_loads_hippomaps_gradient_artifacts()
    test_group_atlas_gradient_loader_prefers_local_artifact_after_relocation()
    test_canonical_public_surfaces_drop_partial_corr_branches()
    test_canonical_present_defaults_use_outputs_core_layout()
    test_run_layout_keeps_smoke_outputs_in_dedicated_run_tree()
    test_smoke_command_plan_uses_isolated_fixedk_full_layout()
    test_compute_defaults_to_five_components_like_hippomaps_paper()
    test_full_surface_reconstruction_keeps_nan_outside_support()
    test_gradient_alignment_reorders_components_and_flips_sign_to_anchor()
    test_gradient_alignment_anchor_combo_is_identity_without_raw_outputs()
    test_endpoint_fc_contrast_uses_top_minus_bottom_gradient_quartiles()
    test_render_gradient_figure_calls_official_hippomaps_plotter_options()
    test_render_gradient_figure_can_render_left_hemi_only()
    test_annotate_gradient_image_adds_left_margin_and_ev_labels()
    test_ensure_bilateral_gradient_annotations_repairs_late_raw_screenshot()
    print("ok: hippomaps gradient route checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

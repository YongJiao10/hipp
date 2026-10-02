from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest  # pyright: ignore[reportMissingImports]
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RENDERER_PATH = ROOT / (
    "scripts/visualize/multidataset_hipp_cortex_gradient_wta/"
    "render_multidataset_hipp_cortex_gradient_wta.py"
)


@pytest.fixture(scope="module")
def renderer():
    module_spec = importlib.util.spec_from_file_location(
        "test_hipp_cortex_gradient_wta_renderer", RENDERER_PATH
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


def _cohort_spec(tmp_path: Path, *, signal_branch: str = "standard") -> dict:
    return {
        "id": "iowa_hc",
        "dataset_key": "iowa_jj_tms",
        "dataset_name": "Iowa 3T",
        "cohort_name": "HC",
        "n_subjects": 67,
        "group_root_grad": tmp_path / "group",
        "group_root_wta": tmp_path / "group",
        "grid_spec": SimpleNamespace(is_7net=True, signal_branch=signal_branch),
        "legacy_7net_namespace": True,
        "signal_branch": signal_branch,
        "grad_subpath": tmp_path / "missing-gradient",
        "wta_subpath": tmp_path / "missing-wta",
    }


def test_tm_does_not_add_a_footer_subregion_legend(renderer):
    """TM must use the existing network footer, not add another legend."""
    assert not hasattr(renderer, "TM_FOOTER_BADGES")


def test_gradient_panel_does_not_reuse_stale_png_when_manifest_is_missing(
    renderer, tmp_path
):
    stale_dir = tmp_path / "panels/iowa_hc_36p_7net_hippo_grad_biglegend_2mm"
    stale_dir.mkdir(parents=True)
    Image.new("RGB", (10, 10), "white").save(
        stale_dir / "sub-100610_wb_iowa_hc_36p_7net_grad_2mm_biglegend.png"
    )

    with pytest.raises(FileNotFoundError, match="Missing gradient cohort"):
        renderer.render_hippo_grad_biglegend(
            cid="iowa_hc",
            spec=_cohort_spec(tmp_path),
            panels_dir=tmp_path / "panels",
            smoothing="2mm",
            variant="36p",
            is_7net=True,
        )


def test_wta_panel_does_not_reuse_stale_png_when_manifest_is_missing(
    renderer, tmp_path
):
    stale_dir = tmp_path / "panels/iowa_hc_36p_7net_hippo_wta_biglegend_2mm"
    stale_dir.mkdir(parents=True)
    Image.new("RGB", (10, 10), "white").save(
        stale_dir / "sub-100610_wb_iowa_hc_36p_7net_wta_2mm_biglegend.png"
    )

    with pytest.raises(FileNotFoundError, match="Missing WTA cohort"):
        renderer.render_hippo_wta_biglegend(
            cid="iowa_hc",
            spec=_cohort_spec(tmp_path),
            panels_dir=tmp_path / "panels",
            smoothing="2mm",
            variant="36p",
            is_7net=True,
        )


def test_missing_gradient_manifest_does_not_use_legacy_contrast_cache(
    renderer, tmp_path, monkeypatch
):
    renderer_root = tmp_path / "renderer"
    (renderer_root / "contrasts").mkdir(parents=True)
    np.savez_compressed(
        renderer_root / "contrasts/legacy_2mm_36p_iowa_hc_subjects.npz",
        values=np.zeros((1, 7, 400), dtype=np.float32),
    )
    monkeypatch.setattr(renderer, "__file__", str(renderer_root / "render.py"))
    monkeypatch.setattr(
        renderer,
        "render_hippo_grad_biglegend",
        lambda **_: Image.new("RGB", (10, 10), "white"),
    )
    monkeypatch.setattr(
        renderer,
        "render_hippo_wta_biglegend",
        lambda **_: Image.new("RGB", (10, 10), "white"),
    )
    monkeypatch.setattr(
        renderer,
        "render_cortex_tile",
        lambda *, panel_tag, **_: tmp_path / f"{panel_tag}.png",
    )
    fake_contrasts = ModuleType("cortex_group_contrasts")
    fake_contrasts.__dict__["canonical_wta_mean"] = lambda *_: (
        np.zeros((4, 400), dtype=np.float64),
        {},
    )
    monkeypatch.setitem(sys.modules, "cortex_group_contrasts", fake_contrasts)

    with pytest.raises(FileNotFoundError, match="Missing gradient atlas manifest"):
        renderer.load_cohort_data(
            _cohort_spec(tmp_path),
            panels_dir=tmp_path / "panels",
            cortex_assets={},
            bundle={"roi_registry": [], "vertex_arrays": {}},
            smoothing="2mm",
            variant="36p",
            is_7net=True,
        )


def test_ctxreg_input_roots_require_explicit_registry(renderer, tmp_path):
    spec = _cohort_spec(tmp_path, signal_branch="ctxreg")
    spec.pop("surface_root", None)
    spec.pop("reference_root", None)
    spec.pop("mask_root", None)

    with pytest.raises(ValueError, match="refusing to fall back"):
        renderer.resolve_cohort_inputs_roots(spec, "36p")


def test_panel_cache_inputs_include_signal_branch(renderer, tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("source\n", encoding="utf-8")
    common = {
        "id": "iowa_hc",
        "dataset_name": "Iowa 3T",
        "cohort_name": "HC",
        "n_subjects": 67,
        "signal_branch": "standard",
    }
    standard = renderer._hippo_workbench_cache_inputs(
        artifact="test",
        panel_tag="standard",
        cid="iowa_hc",
        spec=common,
        variant="36p",
        smoothing="2mm",
        is_7net=True,
        source_files={"source": source},
    )
    ctxreg = renderer._hippo_workbench_cache_inputs(
        artifact="test",
        panel_tag="ctxreg",
        cid="iowa_hc",
        spec={**common, "signal_branch": "ctxreg"},
        variant="36p",
        smoothing="2mm",
        is_7net=True,
        source_files={"source": source},
    )

    assert standard["signal_branch"] == "standard"
    assert ctxreg["signal_branch"] == "ctxreg"
    assert standard != ctxreg

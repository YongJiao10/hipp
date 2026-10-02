from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TOP20 = _load(
    "signal_branch_top20_renderer",
    "scripts/visualize/multidataset_gradient_subregion_top20_stacked_2col/"
    "render_multidataset_gradient_subregion_top20_stacked_2col.py",
)
WTA = _load(
    "signal_branch_wta_renderer",
    "scripts/visualize/multidataset_wta_normalized_longaxis_stacked_2col/"
    "render_multidataset_wta_normalized_longaxis_stacked_2col.py",
)
OVERLAY = _load(
    "signal_branch_overlay_renderer",
    "scripts/visualize/multidataset_workbench_hippocampus_overlay/"
    "render_multidataset_workbench_hippocampus_overlay.py",
)


def test_top20_comparison_surface_is_rendered_from_the_declared_assignment(
    tmp_path, monkeypatch
):
    calls = []

    def fake_panel(cohort, subregions, *, variant, smoothing, outdir):
        calls.append((cohort, subregions, variant, smoothing, Path(outdir)))
        return Image.new("RGB", (8, 8), "white")

    monkeypatch.setattr(TOP20, "render_gradient_surface_panel", fake_panel)
    cohort = {
        "id": "hcp_7t",
        "method": "kong2022_7net",
        "signal_branch": "ctxreg",
        "grad_subpath": tmp_path / "gradient",
    }
    subregions = {"L": [], "R": []}

    image = TOP20._render_cohort_surface_image(
        cohort,
        subregions,
        metric_mode=TOP20.SUBREGION_METRIC_NORMALIZED_TOP20,
        smoothing="2mm",
        panel_root=tmp_path / "panels",
    )

    assert image.size == (8, 8)
    assert len(calls) == 1
    assert calls[0][2] == "normalized_top20_count"
    assert calls[0][4] == (
        tmp_path / "panels" / "kong2022_7net" / "hcp_7t"
    )


def test_wta_surface_cache_namespace_contains_signal_branch(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("source\n", encoding="utf-8")
    common = {
        "cid": "iowa_hc",
        "method": "kong2022_7net",
        "dataset_name": "Iowa 3T",
        "cohort_name": "HC",
        "n_subjects": 67,
        "variant": "36p",
        "smoothing": "2mm",
        "is_7net": True,
        "source_files": {"source": source},
    }

    standard = WTA._wta_surface_cache_inputs(
        **common, signal_branch="standard"
    )
    ctxreg = WTA._wta_surface_cache_inputs(
        **common, signal_branch="ctxreg"
    )

    assert standard["signal_branch"] == "standard"
    assert ctxreg["signal_branch"] == "ctxreg"
    assert standard != ctxreg


def test_overlay_resolves_only_the_requested_ctxreg_assignment(tmp_path):
    run_dir = tmp_path / "hemi_L"
    run_dir.mkdir()
    paths = {
        "cluster_assignment_status": run_dir / "status.json",
        "group_functional_atlas_labels": run_dir / "labels.npy",
        "group_functional_atlas_label_map": run_dir / "label_map.json",
        "template_functional_atlas_label": run_dir / "canonical.label.gii",
    }
    paths["cluster_assignment_status"].write_text("{}", encoding="utf-8")
    np.save(paths["group_functional_atlas_labels"], np.array([1, 2, 3]))
    paths["group_functional_atlas_label_map"].write_text("{}", encoding="utf-8")
    paths["template_functional_atlas_label"].write_bytes(b"fixture")
    np.save(run_dir / "group_labels_final.npy", np.array([1, 2, 3]))
    manifest = {
        "signal_branch": "ctxreg",
        "assignment_variants": {
            "fisherz": {
                key: str(path)
                for key, path in paths.items()
            }
        },
    }

    resolved = OVERLAY._resolve_fisherz_assignment(
        run_dir, manifest, expected_signal_branch="ctxreg"
    )

    assert resolved["label_path"] == paths["template_functional_atlas_label"]

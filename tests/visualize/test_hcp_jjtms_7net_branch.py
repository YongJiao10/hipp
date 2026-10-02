"""Contract tests for the additive HCP/JJ_TMS method-comparison view."""
from __future__ import annotations

import subprocess
import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
LIB_DIR = REPO_ROOT / "scripts" / "visualize" / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from multidataset_cohorts import (  # noqa: E402
    COMPARISON_METHODS,
    COMPARISON_VERSION,
    GRID_VARIANTS,
    GridSpec,
    check_comparison_readiness,
    comparison_dataset_style,
    format_comparison_column_title,
    get_comparison_cohorts,
    get_comparison_output_tag,
    get_comparison_visual_columns,
    resolve_comparison_paths,
)

_TOP20_RENDERER_PATH = (
    REPO_ROOT
    / "scripts"
    / "visualize"
    / "multidataset_gradient_subregion_top20_stacked_2col"
    / "render_multidataset_gradient_subregion_top20_stacked_2col.py"
)
_TOP20_SPEC = importlib.util.spec_from_file_location("comparison_top20_renderer", _TOP20_RENDERER_PATH)
assert _TOP20_SPEC is not None and _TOP20_SPEC.loader is not None
_TOP20_RENDERER = importlib.util.module_from_spec(_TOP20_SPEC)
_TOP20_SPEC.loader.exec_module(_TOP20_RENDERER)
resolve_gradient_overview_path = _TOP20_RENDERER.resolve_gradient_overview_path
load_cohort_subregion_top20_data = _TOP20_RENDERER.load_cohort_subregion_top20_data
resolve_subregion_metric_mode = _TOP20_RENDERER.resolve_subregion_metric_mode
resolve_subregion_metric_mode = _TOP20_RENDERER.resolve_subregion_metric_mode

_CLUSTER_RENDERER_PATH = (
    REPO_ROOT
    / "scripts"
    / "visualize"
    / "multidataset_cluster_confidence_stacked"
    / "plot_multidataset_cluster_positive_stacked.py"
)
_CLUSTER_SPEC = importlib.util.spec_from_file_location("comparison_cluster_renderer", _CLUSTER_RENDERER_PATH)
assert _CLUSTER_SPEC is not None and _CLUSTER_SPEC.loader is not None
_CLUSTER_RENDERER = importlib.util.module_from_spec(_CLUSTER_SPEC)
_CLUSTER_SPEC.loader.exec_module(_CLUSTER_RENDERER)
adaptive_positive_axis_scale = getattr(
    _CLUSTER_RENDERER,
    "adaptive_positive_axis_scale",
    lambda _bar_entries: (0.0, []),
)
comparison_dataset_style = getattr(
    _CLUSTER_RENDERER,
    "comparison_dataset_style",
    lambda _entry: ("", ""),
)
format_column_atlas_title = getattr(
    _CLUSTER_RENDERER,
    "format_column_atlas_title",
    lambda _method_name, _smoothing: "",
)
figure_title = getattr(
    _CLUSTER_RENDERER,
    "FIGURE_TITLE",
    "Cross-Dataset Hippocampal Positive Functional Network Profiles (Stacked)",
)

_HIPPO_CORTEX_RENDERER_PATH = (
    REPO_ROOT
    / "scripts"
    / "visualize"
    / "multidataset_hipp_cortex_gradient_wta"
    / "render_multidataset_hipp_cortex_gradient_wta.py"
)
_HIPPO_CORTEX_SPEC = importlib.util.spec_from_file_location(
    "comparison_hippo_cortex_renderer", _HIPPO_CORTEX_RENDERER_PATH
)
assert _HIPPO_CORTEX_SPEC is not None and _HIPPO_CORTEX_SPEC.loader is not None
_HIPPO_CORTEX_RENDERER = importlib.util.module_from_spec(_HIPPO_CORTEX_SPEC)
_HIPPO_CORTEX_SPEC.loader.exec_module(_HIPPO_CORTEX_RENDERER)
load_gradient_roi_profile = _HIPPO_CORTEX_RENDERER._load_gradient_roi_profile
build_wta_profile = _HIPPO_CORTEX_RENDERER._build_wta_profile
validate_wta_profile_cache_manifest = _HIPPO_CORTEX_RENDERER.validate_wta_profile_cache_manifest

_WTA_RENDERER_PATH = (
    REPO_ROOT
    / "scripts"
    / "visualize"
    / "multidataset_wta_normalized_longaxis_stacked_2col"
    / "render_multidataset_wta_normalized_longaxis_stacked_2col.py"
)
_WTA_SPEC = importlib.util.spec_from_file_location("comparison_wta_surface_renderer", _WTA_RENDERER_PATH)
assert _WTA_SPEC is not None and _WTA_SPEC.loader is not None
_WTA_RENDERER = importlib.util.module_from_spec(_WTA_SPEC)
_WTA_SPEC.loader.exec_module(_WTA_RENDERER)

_SUBFIELD_RENDERER_PATH = (
    REPO_ROOT
    / "scripts"
    / "visualize"
    / "multidataset_subfield_composition_2mm_2col"
    / "render_multidataset_subfield_composition_2mm_2col.py"
)
_SUBFIELD_SPEC = importlib.util.spec_from_file_location(
    "comparison_subfield_renderer", _SUBFIELD_RENDERER_PATH
)
assert _SUBFIELD_SPEC is not None and _SUBFIELD_SPEC.loader is not None
_SUBFIELD_RENDERER = importlib.util.module_from_spec(_SUBFIELD_SPEC)
_SUBFIELD_SPEC.loader.exec_module(_SUBFIELD_RENDERER)
select_manifest_label_path = getattr(_SUBFIELD_RENDERER, "select_manifest_label_path", None)


EXPECTED_COHORTS = (
    "hcp_7t",
    "iowa_hc",
    "iowa_pat",
    "iowa_pure_mdd",
    "iowa_comorbid_mdd",
)


def test_subfield_surface_render_uses_gradient_fisherz_for_kong_comparison(tmp_path, monkeypatch):
    """The subfield panorama's gradient surface must not use the Kong WTA variant."""
    grad_root = tmp_path / "gradient-spectral" / "kong2022" / "2mm" / "cov_0p80" / "g1g2g3"
    for hemi in ("L", "R"):
        hemi_dir = grad_root / f"hemi_{hemi}"
        hemi_dir.mkdir(parents=True)
        manifest = {
            "template_left_or_right_surface": str(tmp_path / f"{hemi}.surf.gii"),
            "template_subject": "sub-100610",
            "template_spec_path": str(tmp_path / "template.spec"),
            "assignment_variants": {
                "fisherz": {
                    "template_functional_atlas_label_variant": str(
                        tmp_path / f"{hemi}_gradient_fisherz.label.gii"
                    )
                },
                "normalized_top20_count": {
                    "template_functional_atlas_label_variant": str(
                        tmp_path / f"{hemi}_gradient_normalized_top20_count.label.gii"
                    )
                },
            },
        }
        (hemi_dir / "group_atlas_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    panels_dir = tmp_path / "panels"
    panels_dir.mkdir()
    cached_png = panels_dir / "wb_subfield_kong2022_7net_HCP_7T_HCP_7T_7net_2mm_36p.png"
    Image.new("RGB", (1, 1), color="white").save(cached_png)
    captured = {}

    def capture_panel_inputs(paths, **kwargs):
        captured["paths"] = [Path(p) for p in paths]
        return {}

    monkeypatch.setattr(_SUBFIELD_RENDERER, "panel_inputs", capture_panel_inputs)
    monkeypatch.setattr(_SUBFIELD_RENDERER, "cache_matches", lambda _path, _inputs: True)
    monkeypatch.setattr(
        _SUBFIELD_RENDERER,
        "validate_gradient_assignment",
        lambda run_dir, _manifest, variant: {
            "label_path": tmp_path / f"validated_{Path(run_dir).name[-1]}_{variant}.label.gii"
        },
        raising=False,
    )

    _SUBFIELD_RENDERER.render_cohort_surface_img(
        {
            "dataset_name": "HCP 7T",
            "cohort_name": "HCP 7T",
            "method": "kong2022_7net",
            "legacy_7net_namespace": True,
            "signal_branch": "standard",
            "group_root": tmp_path,
            "grad_subpath": grad_root,
            # This is the method-level WTA choice and must not control gradient panels.
            "assignment_variant": "normalized_top20_count",
        },
        tmp_path / "render_tmp",
        smoothing="2mm",
        variant="36p",
        panels_dir=panels_dir,
    )

    label_paths = {path.name for path in captured["paths"] if path.name.endswith(".label.gii")}
    assert label_paths == {
        "validated_L_fisherz.label.gii",
        "validated_R_fisherz.label.gii",
    }


def test_comparison_version_restores_both_canonical_7net_method_columns():
    assert COMPARISON_VERSION == "hcp_jjtms_7net"
    assert COMPARISON_METHODS == ("kong2022_7net", "lynch2024_7net")

    for tag in GRID_VARIANTS:
        methods = get_comparison_cohorts(GridSpec.from_tag(tag))
        assert tuple(methods) == COMPARISON_METHODS
        for method in COMPARISON_METHODS:
            cohorts = methods[method]
            assert tuple(c["id"] for c in cohorts) == EXPECTED_COHORTS
            assert {c["dataset_key"] for c in cohorts} <= {"hcp", "iowa_jj_tms"}
            assert not any("stanford" in str(c).lower() for c in cohorts)
            assert all(c["method"] == method for c in cohorts)


@pytest.mark.parametrize("tag", GRID_VARIANTS)
@pytest.mark.parametrize("method", COMPARISON_METHODS)
def test_comparison_paths_are_method_and_grid_specific(tag: str, method: str):
    spec = GridSpec.from_tag(tag)
    for cohort_id in EXPECTED_COHORTS:
        paths = resolve_comparison_paths(cohort_id, method, spec)
        assert paths["method"] == method
        assert paths["variant"] == spec.variant
        assert f"/{spec.smoothing}/" in str(paths["grad_subpath"])
        assert "outputs_stanford" not in str(paths).lower()

        assert "routea_7net" not in str(paths)
        if method == "kong2022_7net":
            if cohort_id == "hcp_7t":
                assert paths["group_root"].name == "kong2022_7net_full_k3_fisherz"
            assert paths["atlas_folder"] == "kong2022"
            assert "/kong2022/" in str(paths["grad_subpath"])
            assert "/kong2022/" in str(paths["wta_subpath"])
            assert paths["assignment_variant"] == "normalized_top20_count"
            assert paths["overview_grad_normalized_top20_count"].name.endswith(
                "_normalized_top20_count.png"
            )
            assert paths["overview_wta"].name.endswith("normalized_top20_count.png")
        else:
            assert f"/{method}/" in str(paths["grad_subpath"])
            assert f"/{method}/" in str(paths["wta_subpath"])
            assert paths["assignment_variant"] == "fisherz"
            assert "overview_grad_normalized_top20_count" not in paths
            assert paths["overview_wta"].name.endswith("_fisherz.png")

        if cohort_id != "hcp_7t":
            expected_present_namespace = "kong2022_7net" if method == "kong2022_7net" else "lynch2024_7net"
            assert paths["present_dir"].parent.parent.name == expected_present_namespace
            assert paths["present_dir"].parent.name == ("22p" if spec.variant == "22p" else "36p")
            assert "iowa3t" not in str(paths["present_dir"])

        if cohort_id == "iowa_pat" and method == "kong2022_7net":
            variant_suffix = "_22p" if spec.variant == "22p" else ""
            expected_root = f"routea_patient{variant_suffix}_full_k3_fisherz"
            assert paths["group_root"].name == expected_root


@pytest.mark.parametrize("tag", GRID_VARIANTS)
def test_standard_kong_reference_inputs_use_exact_routea_roots(tag: str):
    """Standard FC extraction must use the producer's explicit routea tree."""
    spec = GridSpec.from_tag(tag)
    for cohort_id in EXPECTED_COHORTS:
        paths = resolve_comparison_paths(cohort_id, "kong2022_7net", spec)
        if cohort_id == "hcp_7t":
            expected_reference = (
                REPO_ROOT / "outputs_hcp/inputs/hcp/kong2022/shared_inputs/routea"
            )
            expected_surface = (
                REPO_ROOT / "outputs_hcp/inputs/hcp/kong2022/shared_inputs/surface"
            )
        else:
            suffix = "_22p" if spec.variant == "22p" else ""
            expected_reference = REPO_ROOT / (
                "outputs_iowa_jj_tms/inputs/iowa_jj_tms/kong2022/"
                f"shared_inputs{suffix}/routea"
            )
            expected_surface = REPO_ROOT / (
                "outputs_iowa_jj_tms/inputs/iowa_jj_tms/kong2022/"
                f"shared_inputs{suffix}/surface"
            )
        assert Path(paths["reference_root"]).resolve() == expected_reference.resolve()
        assert Path(paths["surface_root"]).resolve() == expected_surface.resolve()


def test_comparison_columns_declare_method_specific_subregion_metrics():
    methods = get_comparison_cohorts(GridSpec("2mm", "36p"))

    kong = methods["kong2022_7net"][0]
    lynch = methods["lynch2024_7net"][0]

    assert kong["subregion_metric"] == "normalized_top20_count"
    assert lynch["subregion_metric"] == "fisherz_profile"
    assert kong["overview_grad_normalized_top20_count"].name.endswith(
        "_normalized_top20_count.png"
    )
    assert lynch["overview_grad"].name.endswith("_fisherz.png")


def test_comparison_renderer_selects_lynch_fisherz_profile_without_normalized_top20():
    methods = get_comparison_cohorts(GridSpec("2mm", "36p"))

    assert resolve_subregion_metric_mode(methods["kong2022_7net"][0], requested_normalized=True) == (
        "normalized_top20_count"
    )
    assert resolve_subregion_metric_mode(methods["lynch2024_7net"][0], requested_normalized=True) == (
        "fisherz_profile"
    )


def test_lynch_profile_overview_rejects_a_non_fisherz_artifact(tmp_path):
    lynch = get_comparison_cohorts(GridSpec("2mm", "36p"))["lynch2024_7net"][0]
    wrong = dict(lynch)
    wrong["overview_grad"] = tmp_path / "atlas_normalized_top20_count.png"

    with pytest.raises(ValueError, match="non-Fisher-z artifact"):
        resolve_gradient_overview_path(wrong, normalized=False)


def test_comparison_overview_resolver_rejects_cross_method_metric_requests():
    methods = get_comparison_cohorts(GridSpec("2mm", "36p"))
    kong = methods["kong2022_7net"][0]
    lynch = methods["lynch2024_7net"][0]

    with pytest.raises(ValueError, match="Kong2022 comparison overview"):
        resolve_gradient_overview_path(kong, normalized=False)
    with pytest.raises(ValueError, match="Lynch2024 comparison overview"):
        resolve_gradient_overview_path(lynch, normalized=True)


def test_comparison_output_tag_is_disjoint_from_legacy_and_atlas_tags():
    assert get_comparison_output_tag(GridSpec("2mm", "36p")) == "hcp_jjtms_7net_2mm_36p"
    assert get_comparison_output_tag(GridSpec("4mm", "22p")) == "hcp_jjtms_7net_4mm_22p"


def test_wta_loader_rejects_count_only_label_cache(tmp_path, monkeypatch):
    renderer = _WTA_RENDERER
    wta_dir = tmp_path / "wta"
    for hemi in ("L", "R"):
        hemi_dir = wta_dir / f"hemi_{hemi}"
        hemi_dir.mkdir(parents=True)
        surface_path = tmp_path / f"{hemi}.surf.gii"
        label_path = tmp_path / f"{hemi}.label.gii"
        surface_path.write_bytes(b"surface")
        label_path.write_bytes(b"label")
        manifest = {
            "template_left_or_right_surface": str(surface_path),
            "template_functional_atlas_label": str(label_path),
        }
        (hemi_dir / "group_atlas_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        (hemi_dir / "group_functional_atlas_label_map.json").write_text(
            json.dumps({"network_names": ["Default"], "network_label_keys": {"Default": 1}}),
            encoding="utf-8",
        )
        np.save(hemi_dir / "group_functional_atlas_labels_count.npy", np.asarray([1], dtype=np.int32))

    monkeypatch.setattr(renderer, "long_axis_order", lambda _path: np.asarray([0]))
    monkeypatch.setattr(renderer, "load_labels", lambda _path: (np.asarray([1]), {}))
    cohort_entry = {
        "dataset_name": "HCP 7T",
        "cohort_name": "HCP 7T",
        "signal_branch": "standard",
        "group_root": str(tmp_path),
        "wta_subpath": str(wta_dir),
    }
    with pytest.raises(FileNotFoundError, match="group_functional_atlas_labels.npy"):
        renderer.load_cohort_long_axis_bins(cohort_entry, n_bins=1, smoothing="2mm", is_7net=True)

def test_wta_surface_cache_rerenders_when_source_label_changes(tmp_path, monkeypatch):
    renderer = _WTA_RENDERER
    wta_dir = tmp_path / "wta-top" / "kong2022" / "2mm" / "cov_0p80"
    hemi_l = wta_dir / "hemi_L"
    hemi_r = wta_dir / "hemi_R"
    hemi_l.mkdir(parents=True)
    hemi_r.mkdir(parents=True)

    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    source_files = {
        "left_surface": source_dir / "left.surf.gii",
        "right_surface": source_dir / "right.surf.gii",
        "left_label": source_dir / "left.label.gii",
        "right_label": source_dir / "right.label.gii",
        "spec": source_dir / "spec.scene",
    }
    for key, path in source_files.items():
        path.write_bytes(key.encode("utf-8"))

    for hemi_dir, surface_key, label_key in (
        (hemi_l, "left_surface", "left_label"),
        (hemi_r, "right_surface", "right_label"),
    ):
        manifest = {
            "template_left_or_right_surface": str(source_files[surface_key]),
            "template_functional_atlas_label": str(source_files[label_key]),
            "template_spec_path": str(source_files["spec"]),
        }
        (hemi_dir / "group_atlas_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

    calls = []
    def fake_render_locked_grid(**kwargs):
        calls.append(kwargs)
        png_path = kwargs["outdir"] / f"sub-100610_wb_{kwargs['name']}_biglegend.png"
        png_path.parent.mkdir(parents=True, exist_ok=True)
        renderer.Image.new("RGBA", (8, 8), "white").save(png_path)
        return {"biglegend_png": str(png_path)}

    monkeypatch.setattr(renderer, "render_locked_grid", fake_render_locked_grid)
    cohort_entry = {
        "dataset_name": "HCP 7T",
        "cohort_name": "HCP 7T",
        "n_subjects": 151,
        "method": "kong2022_7net",
        "signal_branch": "standard",
        "group_root": str(tmp_path),
        "wta_subpath": str(wta_dir),
    }
    panels_dir = tmp_path / "panels"
    result = renderer.render_cohort_wta_surface_img(
        cohort_entry, panels_dir, smoothing="2mm", variant="36p", is_7net=True
    )
    assert result.size == (8, 8)
    assert len(calls) == 1

    png_path = next(panels_dir.rglob("*_wta_biglegend.png"))
    assert png_path.with_suffix(".cache.json").exists()

    source_files["left_label"].write_bytes(b"left-label-v2")
    result = renderer.render_cohort_wta_surface_img(
        cohort_entry, panels_dir, smoothing="2mm", variant="36p", is_7net=True
    )
    assert result.size == (8, 8)
    assert len(calls) == 2

def test_workbench_cache_requires_matching_manifest_and_output_fingerprints(tmp_path):
    renderer = _HIPPO_CORTEX_RENDERER
    outputs = {
        "surface": tmp_path / "surface.png",
        "bars": tmp_path / "bars.png",
        "composite": tmp_path / "composite.png",
    }
    for path in outputs.values():
        renderer.Image.new("RGBA", (4, 4), "white").save(path)

    manifest_path = tmp_path / "panel.cache.json"
    expected_inputs = {"artifact": "cortex_tile", "input_fingerprint": "source-a"}
    dimensions = {key: (4, 4) for key in outputs}

    assert not renderer._is_valid_workbench_cache(
        manifest_path,
        expected_inputs=expected_inputs,
        output_paths=outputs,
        expected_dimensions=dimensions,
    )

    manifest_path.write_text(
        json.dumps(renderer._workbench_panel_cache_manifest(expected_inputs, outputs)),
        encoding="utf-8",
    )
    assert renderer._is_valid_workbench_cache(
        manifest_path,
        expected_inputs=expected_inputs,
        output_paths=outputs,
        expected_dimensions=dimensions,
    )

    stale_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stale_manifest["inputs"]["input_fingerprint"] = "source-b"
    manifest_path.write_text(json.dumps(stale_manifest), encoding="utf-8")
    assert not renderer._is_valid_workbench_cache(
        manifest_path,
        expected_inputs=expected_inputs,
        output_paths=outputs,
        expected_dimensions=dimensions,
    )


def test_render_cortex_tile_skips_workbench_on_valid_cache(tmp_path, monkeypatch):
    renderer = _HIPPO_CORTEX_RENDERER
    panels_dir = tmp_path / "panels"
    panel_dir = panels_dir / "smoke"
    panel_dir.mkdir(parents=True)
    output_paths = {
        "surface": panel_dir / "smoke_vmax15_pseudocluster.png",
        "bars": panel_dir / "smoke_top10_bars_v3.png",
        "composite": panel_dir / "smoke_vmax15_pseudocluster_composite_v3.png",
    }
    for key, path in output_paths.items():
        size = (1000, 1408) if key == "composite" else (1000, 700)
        renderer.Image.new("RGBA", size, "white").save(path)

    assets = {}
    for key in ("scene", "left_inflated", "right_inflated", "left_sulc", "right_sulc", "sulc_dscalar", "dlabel"):
        path = tmp_path / f"{key}.asset"
        path.write_bytes(key.encode("utf-8"))
        assets[key] = path
    values = np.asarray([0.1], dtype=float)
    registry = [{
        "roi_id": 1,
        "roi_name": "R1",
        "canonical7": "Default",
        "hemi": "L",
        "family17": "Default",
        "subregion": "Anterior",
        "rank": 1,
    }]
    vertices = {
        "L": (np.asarray([0], dtype=np.int32), np.asarray([1], dtype=np.int32)),
        "R": (np.asarray([0], dtype=np.int32), np.asarray([1], dtype=np.int32)),
    }
    inputs = renderer._cortex_tile_cache_inputs(
        panel_tag="smoke",
        values=values,
        vmax=0.15,
        roi_registry=registry,
        vertex_arrays=vertices,
        cortex_assets=assets,
    )
    manifest_path = output_paths["composite"].with_suffix(".cache.json")
    manifest_path.write_text(
        json.dumps(renderer._workbench_panel_cache_manifest(inputs, output_paths)),
        encoding="utf-8",
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Workbench subprocess ran despite a valid cache")

    monkeypatch.setattr(renderer.subprocess, "run", fail_if_called)
    result = renderer.render_cortex_tile(
        panel_tag="smoke",
        values=values,
        vmax=0.15,
        roi_registry=registry,
        vertex_arrays=vertices,
        cortex_assets=assets,
        panels_dir=panels_dir,
    )
    assert result == output_paths["composite"]


def test_legacy_iowa_hc_overview_refs_use_canonical7_layout():
    renderer_paths = (
        _TOP20_RENDERER_PATH,
        REPO_ROOT
        / "scripts"
        / "visualize"
        / "multidataset_hipp_cortex_gradient_wta"
        / "render_multidataset_hipp_cortex_gradient_wta.py",
    )
    for renderer_path in renderer_paths:
        source = renderer_path.read_text(encoding="utf-8")
        assert "outputs_iowa_jj_tms/present/overviews/routea_hc" not in source
        assert "outputs_iowa_jj_tms/present/overviews/kong2022_7net/36p/hc" in source


def test_hcp_comparison_uses_the_full_qc_cohort():
    spec = GridSpec("2mm", "36p")
    hcp = next(row for row in get_comparison_cohorts(spec)["kong2022_7net"] if row["id"] == "hcp_7t")

    assert hcp["n_subjects"] == 151
    assert hcp["shared_root"].name == "shared_inputs"
    assert hcp["group_root"].name == "kong2022_7net_full_k3_fisherz"


def test_comparison_readiness_reports_two_canonical_method_columns_and_ten_entries():
    report = check_comparison_readiness(GridSpec("2mm", "36p"))
    assert report["version"] == COMPARISON_VERSION
    assert report["total_cohorts"] == 5
    assert report["total_entries"] == 10
    assert tuple(report["methods"]) == COMPARISON_METHODS
    assert not any("stanford" in str(row).lower() for row in report["cohorts"].values())


def test_kong_top20_prefers_capacity_normalized_status_values(tmp_path):
    grad_dir = tmp_path / "gradient-spectral" / "kong2022_7net" / "2mm" / "cov_0p80" / "g1g2g3"
    payload = {
        "assignment_metric": "top20_count_normalized_by_network_roi_count",
        "items": [
            {
                "cluster_id": 1,
                "normalized_count_by_network": {"Default": 0.10, "Visual": 0.30},
            },
            {"cluster_id": 2, "normalized_count_by_network": {"Default": 1.0}},
            {"cluster_id": 3, "normalized_count_by_network": {"Visual": 1.0}},
        ]
    }
    for hemi in ("L", "R"):
        hdir = grad_dir / f"hemi_{hemi}"
        hdir.mkdir(parents=True)
        np.save(hdir / "cluster_network_profile.npy", np.asarray([[2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32))
        (hdir / "group_feature_summary.json").write_text(json.dumps({"network_names": ["Default", "Visual", "Somatomotor", "DorsalAttention", "VentralAttention", "Control", "Language"]}), encoding="utf-8")
        (hdir / "cluster_assignment_status_normalized_top20_count.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    result = load_cohort_subregion_top20_data(
        {
            "group_root": tmp_path,
            "grad_subpath": grad_dir,
            "method": "kong2022_7net",
            "signal_branch": "standard",
        },
        normalized=True,
        smoothing="2mm",
        is_7net=True,
    )
    values = result["L"][0]["comp"]
    assert values["Default"] == pytest.approx(25.0)
    assert values["Visual"] == pytest.approx(75.0)


def test_kong_normalized_top20_rejects_direct_network_profile_when_status_is_missing(tmp_path):
    grad_dir = tmp_path / "gradient-spectral" / "kong2022_7net" / "2mm" / "cov_0p80" / "g1g2g3"
    for hemi in ("L", "R"):
        hdir = grad_dir / f"hemi_{hemi}"
        hdir.mkdir(parents=True)
        npy = np.asarray(
            [[2.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
             [1.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        np.save(hdir / "cluster_network_profile.npy", npy)
        (hdir / "group_feature_summary.json").write_text(
            json.dumps({"network_names": ["Default", "Visual", "Somatomotor", "DorsalAttention", "VentralAttention", "Control", "Language"]}),
            encoding="utf-8",
        )

    with pytest.raises(
        FileNotFoundError,
        match="cluster_assignment_status_normalized_top20_count.json",
    ):
        load_cohort_subregion_top20_data(
            {
                "group_root": tmp_path,
                "grad_subpath": grad_dir,
                "method": "kong2022_7net",
                "signal_branch": "standard",
            },
            normalized=True,
            smoothing="2mm",
            is_7net=True,
        )



def test_normalized_overview_resolver_never_uses_fisherz_or_implicit_fallback(tmp_path):
    normalized_path = tmp_path / "atlas_normalized_top20_count.png"
    cohort = {
        "dataset_name": "HCP 7T",
        "cohort_name": "HCP 7T",
        "overview_grad": tmp_path / "atlas_fisherz.png",
        "overview_grad_normalized_top20_count": normalized_path,
    }

    assert resolve_gradient_overview_path(cohort, normalized=True) == normalized_path

    missing_key = dict(cohort)
    missing_key.pop("overview_grad_normalized_top20_count")
    with pytest.raises(FileNotFoundError, match="overview_grad_normalized_top20_count"):
        resolve_gradient_overview_path(missing_key, normalized=True)

    wrong_variant = dict(cohort)
    wrong_variant["overview_grad_normalized_top20_count"] = cohort["overview_grad"]
    with pytest.raises(ValueError, match="non-normalized artifact"):
        resolve_gradient_overview_path(wrong_variant, normalized=True)

def test_cluster_positive_axis_scale_follows_stacked_data_with_small_margin():
    bar_entries = [
        {
            "left": {"pos_networks": [("Default", 0.10), ("Visual", 0.08)]},
            "right": {"pos_networks": [("Somatomotor", 0.17)]},
        },
        {
            "left": {"pos_networks": [("Control", 0.03)]},
            "right": {"pos_networks": [("Language", 0.04)]},
        },
    ]

    max_xlim, x_ticks = adaptive_positive_axis_scale(bar_entries)

    assert max_xlim == pytest.approx(0.20)
    assert x_ticks == pytest.approx([0.20, 0.15, 0.10, 0.05, 0.0])


def test_cluster_positive_axis_scale_does_not_keep_large_legacy_range_for_small_data():
    bar_entries = [
        {
            "left": {"pos_networks": [("Default", 0.012), ("Visual", 0.010)]},
            "right": {"pos_networks": [("Somatomotor", 0.018)]},
        }
    ]

    max_xlim, x_ticks = adaptive_positive_axis_scale(bar_entries)

    assert max_xlim == pytest.approx(0.025)
    assert max_xlim > 0.022
    assert max_xlim < 0.10
    assert x_ticks == pytest.approx([0.025, 0.020, 0.015, 0.010, 0.005, 0.0])


def test_cluster_positive_axis_scale_limits_rounding_whitespace():
    bar_entries = [
        {
            "left": {"pos_networks": [("Default", 0.21), ("Visual", 0.17)]},
            "right": {"pos_networks": [("Somatomotor", 0.38)]},
        }
    ]

    max_xlim, x_ticks = adaptive_positive_axis_scale(bar_entries)

    assert max_xlim == pytest.approx(0.45)
    assert x_ticks == pytest.approx([0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.0])


def test_comparison_cluster_view_uses_dataset_bands_and_top_atlas_headers():
    assert comparison_dataset_style({"dataset_key": "hcp"}) == ("HCP 7T", "#fafafa")
    assert comparison_dataset_style({"dataset_key": "iowa_jj_tms"}) == ("Iowa 3T", "#f2f7fc")
    assert format_column_atlas_title("Kong2022 7-Network", "2mm") == (
        "Kong2022 7-Network\n2mm Atlas (Front: L/R | Back: L/R)"
    )
    assert "(Stacked)" not in figure_title


def test_each_internal_renderer_preserves_the_additive_version_flag():
    renderer_paths = (
        "scripts/visualize/multidataset_cluster_confidence_stacked/plot_multidataset_cluster_positive_stacked.py",
        "scripts/visualize/multidataset_gradient_subregion_top20_stacked_2col/render_multidataset_gradient_subregion_top20_stacked_2col.py",
        "scripts/visualize/multidataset_subfield_composition_2mm_2col/render_multidataset_subfield_composition_2mm_2col.py",
        "scripts/visualize/multidataset_workbench_hippocampus_overlay/render_multidataset_workbench_hippocampus_overlay.py",
        "scripts/visualize/multidataset_wta_normalized_longaxis_stacked_2col/render_multidataset_wta_normalized_longaxis_stacked_2col.py",
    )
    py_bin = "/home/yojiao/.conda/envs/py314/bin/python"
    for relative_path in renderer_paths:
        result = subprocess.run(
            [py_bin, "-c",
             "import runpy, sys; path = sys.argv.pop(1); sys.argv[0] = path; "
             "runpy.run_path(path, run_name='suite_renderer')['_render_grid']()",
             str(REPO_ROOT / relative_path), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "--version" in result.stdout
        assert COMPARISON_VERSION in result.stdout


def test_suite_runner_dry_run_exposes_additive_version_and_20_jobs():
    runner = REPO_ROOT / "scripts" / "visualize" / "run_multidataset_suite.py"
    py_bin = "/home/yojiao/.conda/envs/py314/bin/python"
    result = subprocess.run(
        [py_bin, str(runner), "--version", COMPARISON_VERSION, "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout.count("--version hcp_jjtms_7net") == 20
    assert "stanford" not in result.stdout.lower()


def test_comparison_branch_declares_legacy_visual_renderer_contract():
    """The additive branch must stay inside the six maintained renderers."""
    expected = {
        "cluster_confidence_stacked": (
            "scripts/visualize/multidataset_cluster_confidence_stacked/"
            "plot_multidataset_cluster_positive_stacked.py",
            "render_multidataset_cluster_positive_stacked",
        ),
        "gradient_subregion_top20_stacked_2col": (
            "scripts/visualize/multidataset_gradient_subregion_top20_stacked_2col/"
            "render_multidataset_gradient_subregion_top20_stacked_2col.py",
            "render_multidataset_gradient_subregion_top20_stacked_2col",
        ),
        "subfield_composition_2col": (
            "scripts/visualize/multidataset_subfield_composition_2mm_2col/"
            "render_multidataset_subfield_composition_2mm_2col.py",
            "render_subfield_composition_figure",
        ),
        "workbench_hippocampus_overlay": (
            "scripts/visualize/multidataset_workbench_hippocampus_overlay/"
            "render_multidataset_workbench_hippocampus_overlay.py",
            "render_all_cohorts_overlay_figure",
        ),
        "wta_normalized_longaxis_stacked_2col": (
            "scripts/visualize/multidataset_wta_normalized_longaxis_stacked_2col/"
            "render_multidataset_wta_normalized_longaxis_stacked_2col.py",
            "render_multidataset_wta_normalized_longaxis_stacked_2col",
        ),
    }
    for _kind, (relative_path, renderer_name) in expected.items():
        source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert "get_comparison_visual_columns" in source
        assert renderer_name in source
        assert "hcp_jjtms_7net_comparison" not in source


def test_workbench_comparison_contract_keeps_locked_3d_rendering():
    """Workbench content changes must retain the original 3D surface renderer."""
    source = (
        REPO_ROOT
        / "scripts/visualize/multidataset_workbench_hippocampus_overlay/"
        "render_multidataset_workbench_hippocampus_overlay.py"
    ).read_text(encoding="utf-8")
    assert "render_locked_grid" in source
    assert "np.vstack([left, right])" not in source
    assert "load_cohort_data_and_render_wb" in source


def test_patient_uses_verified_subject_count_and_has_no_blank_fallback():
    methods = get_comparison_cohorts(GridSpec("2mm", "36p"))
    patient = next(row for row in methods["kong2022_7net"] if row["id"] == "iowa_pat")
    assert patient["n_subjects"] == 26

    renderer_paths = (
        REPO_ROOT / "scripts/visualize/multidataset_cluster_confidence_stacked/plot_multidataset_cluster_positive_stacked.py",
        REPO_ROOT / "scripts/visualize/multidataset_gradient_subregion_top20_stacked_2col/render_multidataset_gradient_subregion_top20_stacked_2col.py",
        REPO_ROOT / "scripts/visualize/multidataset_hipp_cortex_gradient_wta/render_multidataset_hipp_cortex_gradient_wta.py",
        REPO_ROOT / "scripts/visualize/multidataset_subfield_composition_2mm_2col/render_multidataset_subfield_composition_2mm_2col.py",
        REPO_ROOT / "scripts/visualize/multidataset_workbench_hippocampus_overlay/render_multidataset_workbench_hippocampus_overlay.py",
        REPO_ROOT / "scripts/visualize/multidataset_wta_normalized_longaxis_stacked_2col/render_multidataset_wta_normalized_longaxis_stacked_2col.py",
    )
    for path in renderer_paths:
        source = path.read_text(encoding="utf-8")
        assert "unavailable_cohort_data" not in source
        assert "process_unavailable_comparison_cohort" not in source


def test_patient_sample_count_is_aligned_across_comparison_methods():
    methods = get_comparison_cohorts(GridSpec("2mm", "36p"))
    patient_counts = {
        method: next(row for row in cohorts if row["id"] == "iowa_pat")["n_subjects"]
        for method, cohorts in methods.items()
    }
    assert patient_counts == {"kong2022_7net": 26, "lynch2024_7net": 26}


def test_comparison_visual_rows_keep_dataset_bands_and_correct_counts():
    col1, col2 = get_comparison_visual_columns(GridSpec("2mm", "36p"))

    assert [row["dataset_name"] for row in col1] == ["HCP 7T", "Iowa 3T", "Iowa 3T", "Iowa 3T", "Iowa 3T"]
    assert [row["dataset_name"] for row in col2] == ["HCP 7T", "Iowa 3T", "Iowa 3T", "Iowa 3T", "Iowa 3T"]
    assert [row["n_subjects"] for row in col1] == [151, 67, 26, 7, 19]
    assert [row["method_name"] for row in col1] == ["Kong2022 7-Network"] * 5
    assert [row["n_subjects"] for row in col2] == [151, 67, 26, 7, 19]
    assert [row["method_name"] for row in col2] == ["Lynch2024 7-Network"] * 5
    assert {row["bg_color"] for row in col1} == {"#fafafa", "#f2f7fc"}


def test_shared_comparison_display_helpers_define_the_six_renderer_contract():
    assert comparison_dataset_style({"dataset_key": "hcp"}) == ("HCP 7T", "#fafafa")
    assert comparison_dataset_style({"dataset_key": "iowa_jj_tms"}) == ("Iowa 3T", "#f2f7fc")
    assert format_comparison_column_title("Kong2022 7-Network", "2mm Atlas") == (
        "Kong2022 7-Network\n2mm Atlas"
    )
    assert format_comparison_column_title(None, "2mm Atlas") == "2mm Atlas"


def test_gradient_loader_does_not_expose_network_profile_expansion_as_roi_data():
    assert not hasattr(_HIPPO_CORTEX_RENDERER, "expand_network_profile_to_roi_profile")


def _write_gradient_profile_case(
    tmp_path: Path,
    *,
    profile: np.ndarray | None = None,
    manifest_updates: dict[str, object] | None = None,
    include_network_profile: bool = False,
) -> Path:
    sys.path.insert(0, str(REPO_ROOT / "tests/core_workflow"))
    from lineage_fixtures import make_run
    grad_sub = tmp_path / "gradient-spectral" / "kong2022_7net" / "2mm" / "cov_0p80" / "g1g2g3"
    hemi_dir = grad_sub / "hemi_L"
    manifest_path = make_run(hemi_dir, omit_profile=True)
    profile_path = hemi_dir / "cluster_roi_profile.npy"
    if profile is not None:
        np.save(profile_path, profile)
    manifest = {**json.loads(manifest_path.read_text()),
        "status": "ok",
        "atlas": "kong2022",
        "cluster_roi_profile_contract": "roi_ap",
        "cluster_roi_profile": str(profile_path.resolve()),
        "cluster_label_ordering": "long_axis_head_to_tail",
        "cluster_order_head_to_tail": [1, 2, 3],
    }
    if manifest_updates:
        manifest.update(manifest_updates)
    (hemi_dir / "group_atlas_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if include_network_profile:
        np.save(hemi_dir / "cluster_network_profile.npy", np.ones((3, 7), dtype=np.float32))
        (hemi_dir / "group_feature_summary.json").write_text(
            json.dumps({"network_names": ["Default", "Visual", "Somatomotor", "DorsalAttention", "VentralAttention", "Control", "Language"]}),
            encoding="utf-8",
        )
    return grad_sub


def test_gradient_loader_accepts_only_a_registered_long_axis_roi_profile(tmp_path: Path):
    expected = np.arange(1200, dtype=np.float32).reshape(3, 400)
    grad_sub = _write_gradient_profile_case(tmp_path, profile=expected)

    got = load_gradient_roi_profile(grad_sub, "L", bundle={})

    assert got.shape == (3, 400)
    np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize(
    ("profile", "manifest_updates", "match"),
    [
        (
            np.zeros((3, 400), dtype=np.float32),
            {"cluster_roi_profile": "/tmp/not-the-profile.npy"},
            "manifest cluster_roi_profile",
        ),
        (
            np.zeros((2, 400), dtype=np.float32),
            {},
            "shape.*3.*400",
        ),
        (
            np.zeros((3, 400), dtype=np.float32),
            {"cluster_label_ordering": "raw_cluster_id"},
            "long_axis_head_to_tail",
        ),
        (
            np.zeros((3, 400), dtype=np.float32),
            {"cluster_order_head_to_tail": [1, 3, 2]},
            "cluster_order_head_to_tail",
        ),
    ],
)
def test_gradient_loader_rejects_mismatched_profile_contract(
    tmp_path: Path,
    profile: np.ndarray,
    manifest_updates: dict[str, object],
    match: str,
):
    grad_sub = _write_gradient_profile_case(
        tmp_path,
        profile=profile,
        manifest_updates=manifest_updates,
    )

    with pytest.raises((ValueError, FileNotFoundError), match=match):
        load_gradient_roi_profile(grad_sub, "L", bundle={})


def test_gradient_loader_rejects_network_profile_as_an_roi_profile(tmp_path: Path):
    grad_sub = _write_gradient_profile_case(tmp_path, include_network_profile=True)

    with pytest.raises(FileNotFoundError, match="cluster_roi_profile"):
        load_gradient_roi_profile(grad_sub, "L", bundle={})


def _write_wta_profile_cache_case(
    tmp_path: Path,
    *,
    profile: np.ndarray | None = None,
    manifest_updates: dict[str, object] | None = None,
) -> tuple[Path, dict[str, object]]:
    profile_path = tmp_path / "wta_pseudocluster_bilat.npy"
    manifest_path = tmp_path / "wta_pseudocluster_bilat.manifest.json"
    if profile is not None:
        np.save(profile_path, profile)
    manifest: dict[str, object] = {
        "status": "ok",
        "wta_profile_contract": "wta_roi",
        "wta_profile": str(profile_path.resolve()),
        "cohort_id": "iowa_pat",
        "variant": "36p",
        "smoothing": "2mm",
        "atlas": "kong2022",
        "method": "wta-top",
        "profile_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
        "network_order": ["Default", "VentralAttention", "Somatomotor", "Language"],
        "source_runs": {
            hemi: {
                key: f"{hemi}-{key}"
                for key in (
                    "manifest_path",
                    "manifest_sha256",
                    "label_path",
                    "label_sha256",
                    "support_path",
                    "support_sha256",
                    "label_map_path",
                    "label_map_sha256",
                )
            }
            for hemi in ("L", "R")
        },
    }
    if manifest_updates:
        manifest.update(manifest_updates)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return profile_path, manifest


def test_wta_profile_contract_registers_four_rows_in_one_canonical_order(tmp_path: Path):
    profile = np.arange(1600, dtype=np.float32).reshape(4, 400)
    profile_path, manifest = _write_wta_profile_cache_case(tmp_path, profile=profile)

    validate_wta_profile_cache_manifest(
        manifest,
        profile_path=profile_path,
        profile_shape=tuple(profile.shape),
        expected_cohort_id="iowa_pat",
        expected_variant="36p",
        expected_smoothing="2mm",
    )


@pytest.mark.parametrize(
    ("profile", "manifest_updates", "match"),
    [
        (np.zeros((3, 400), dtype=np.float32), {}, "shape.*4.*400"),
        (
            np.zeros((4, 400), dtype=np.float32),
            {"network_order": ["VentralAttention", "Default", "Somatomotor", "Language"]},
            "network_order",
        ),
        (
            np.zeros((4, 400), dtype=np.float32),
            {"variant": "22p"},
            "variant",
        ),
        (
            np.zeros((4, 400), dtype=np.float32),
            {"wta_profile": "/tmp/not-the-profile.npy"},
            "wta_profile",
        ),
    ],
)
def test_wta_profile_contract_rejects_mismatched_rows_or_source_identity(
    tmp_path: Path,
    profile: np.ndarray,
    manifest_updates: dict[str, object],
    match: str,
):
    profile_path, manifest = _write_wta_profile_cache_case(
        tmp_path,
        profile=profile,
        manifest_updates=manifest_updates,
    )

    with pytest.raises(ValueError, match=match):
        validate_wta_profile_cache_manifest(
            manifest,
            profile_path=profile_path,
            profile_shape=tuple(profile.shape),
            expected_cohort_id="iowa_pat",
            expected_variant="36p",
            expected_smoothing="2mm",
        )


def test_wta_profile_contract_rejects_tampered_profile(tmp_path: Path):
    profile = np.arange(1600, dtype=np.float32).reshape(4, 400)
    profile_path, manifest = _write_wta_profile_cache_case(tmp_path, profile=profile)
    np.save(profile_path, profile + 1.0)

    with pytest.raises(ValueError, match="profile_sha256"):
        validate_wta_profile_cache_manifest(
            manifest,
            profile_path=profile_path,
            profile_shape=tuple(profile.shape),
            expected_cohort_id="iowa_pat",
            expected_variant="36p",
            expected_smoothing="2mm",
        )


def test_wta_builder_preserves_an_empty_middle_network_row(tmp_path: Path, monkeypatch):
    renderer = _HIPPO_CORTEX_RENDERER
    grad_sub = tmp_path / "gradient-spectral" / "kong2022_7net" / "2mm" / "cov_0p80" / "g1g2g3"
    grad_hemi = grad_sub / "hemi_L"
    grad_hemi.mkdir(parents=True)
    (grad_hemi / "group_atlas_manifest.json").write_text(
        json.dumps({"subjects": ["sub-s1"]}), encoding="utf-8"
    )
    panels_dir = tmp_path / "panels"
    panels_dir.mkdir()

    captured_labels: list[np.ndarray] = []
    network_label_keys = {
        "Default": 10,
        "VentralAttention": 20,
        "Somatomotor": 30,
        "Language": 40,
    }

    def fake_source_run(
        _wta_dir: Path,
        hemi: str,
        *,
        smoothing: str,
        is_7net: bool,
        signal_branch: str = "standard",
    ):
        assert smoothing == "2mm"
        assert is_7net
        assert signal_branch == "standard"
        snapshot = {
            key: f"{hemi}-{key}"
            for key in (
                "manifest_path",
                "manifest_sha256",
                "label_path",
                "label_sha256",
                "support_path",
                "support_sha256",
                "label_map_path",
                "label_map_sha256",
            )
        }
        return {
            "manifest": {"subjects": ["s1"]},
            "labels": np.asarray([10, 30], dtype=np.int32),
            "support": np.asarray([True, True]),
            "network_label_keys": network_label_keys,
            "snapshot": snapshot,
        }

    def fake_compute_subject_profiles(**kwargs):
        captured_labels.append(np.asarray(kwargs["group_labels"]).copy())
        profile = np.full((1, 4, 400), np.nan, dtype=np.float32)
        profile[:, 0, :] = 1.0
        # Row 2 (VentralAttention) is intentionally empty; rows 3 and 4
        # remain populated so a compacting implementation would be visible.
        profile[:, 2, :] = 3.0
        profile[:, 3, :] = 4.0
        return profile, ["s1"]

    monkeypatch.setattr(renderer, "_wta_source_run", fake_source_run)
    monkeypatch.setattr(renderer, "compute_subject_label_roi_profiles", fake_compute_subject_profiles)
    monkeypatch.setattr(
        renderer,
        "resolve_cohort_inputs_roots",
        lambda _cohort_spec, _variant: (tmp_path, tmp_path, tmp_path),
    )

    profile, network_order = build_wta_profile(
        cohort_spec={
            "id": "iowa_pat",
            "method": "kong2022_7net",
            "dataset_key": "iowa_jj_tms",
            "signal_branch": "standard",
        },
        panels_dir=panels_dir,
        grad_sub=grad_sub,
        wta_dir=tmp_path / "wta",
        bundle={"roi_registry": []},
        smoothing="2mm",
        variant="36p",
        is_7net=True,
    )

    assert tuple(network_order) == ("Default", "VentralAttention", "Somatomotor", "Language")
    assert captured_labels[0].tolist() == [1, 3]
    assert captured_labels[1].tolist() == [1, 3]
    assert profile.shape == (4, 400)
    np.testing.assert_allclose(profile[0], 1.0)
    np.testing.assert_allclose(profile[1], 0.0)
    np.testing.assert_allclose(profile[2], 3.0)
    np.testing.assert_allclose(profile[3], 4.0)

def test_manifest_label_selection_rejects_fisherz_when_normalized_variant_is_absent():
    assert select_manifest_label_path is not None
    manifest = {
        "assignment_variants": {
            "fisherz": {"template_functional_atlas_label_variant": "/tmp/fisherz.label.gii"}
        }
    }
    with pytest.raises(KeyError, match="normalized_top20_count"):
        select_manifest_label_path(manifest)
    assert select_manifest_label_path(manifest, assignment_variant='fisherz') == Path('/tmp/fisherz.label.gii')


def test_comparison_subfield_composition_header_contract():
    import inspect

    sig = inspect.signature(_SUBFIELD_RENDERER.render_subfield_composition_figure)
    assert "is_comparison" in sig.parameters
    assert "show_version_label" in sig.parameters
    assert "show_subtitle" in sig.parameters

    source = _SUBFIELD_RENDERER_PATH.read_text(encoding="utf-8")
    assert "--show-version-label" in source
    assert "--show-subtitle" in source


def test_comparison_cortex_panel_namespacing_and_differentiation():
    renderer_path = (
        REPO_ROOT
        / "scripts"
        / "visualize"
        / "multidataset_hipp_cortex_gradient_wta"
        / "render_multidataset_hipp_cortex_gradient_wta.py"
    )
    source = renderer_path.read_text(encoding="utf-8")
    assert 'method_prefix = f"{cohort_spec[\'method\']}_" if cohort_spec.get("method") else ""' in source
    assert 'panel_tag = f"{method_prefix}{cid}_{variant}_{signal_branch}_{atlas_tag}grad_{subreg.lower()}_{smoothing}"' in source
    assert 'panel_tag = f"{method_prefix}{cid}_{variant}_{signal_branch}_{atlas_tag}wta_{net.lower()}_{smoothing}"' in source
    assert "wta_pseudocluster_bilat_{smoothing}_{signal_branch}.npy" in source
    assert "_build_wta_profile(" in source
    assert "if prof_cache.exists()" not in source
    assert "expand_network_profile_to_roi_profile" not in source
    assert "legacy_tag" not in source

    # Verify that rendered panels on disk for 2mm_36p are distinct
    panels_dir = renderer_path.parent / "panels"
    k_tile = panels_dir / "kong2022_7net_hcp_7t_36p_7net_grad_anterior_2mm" / "kong2022_7net_hcp_7t_36p_7net_grad_anterior_2mm_vmax15_pseudocluster.png"
    l_tile = panels_dir / "lynch2024_7net_hcp_7t_36p_7net_grad_anterior_2mm" / "lynch2024_7net_hcp_7t_36p_7net_grad_anterior_2mm_vmax15_pseudocluster.png"
    if k_tile.exists() and l_tile.exists():
        from PIL import Image
        arr_k = np.array(Image.open(k_tile))
        arr_l = np.array(Image.open(l_tile))
        diff = np.abs(arr_k.astype(float) - arr_l.astype(float))
        assert np.max(diff) > 50.0, "Kong and Lynch anterior gradient tiles should be significantly different"


def test_all_renderer_variants_reject_silent_profile_fallbacks():
    renderer_dir = REPO_ROOT / "scripts" / "visualize" / "multidataset_hipp_cortex_gradient_wta"
    forbidden = (
        "expand_network_profile_to_roi_profile",
        "legacy_tag",
        "cluster_network_profile",
        "group_labels_k3",
        "group_labels_final",
        "group_functional_atlas_labels_count",
        "np.ones_like(lbl",
        "if prof_cache.exists()",
    )
    renderer_paths = sorted(renderer_dir.glob("render_multidataset_hipp_cortex_gradient_wta*.py"))
    assert renderer_paths
    for renderer_path in renderer_paths:
        source = renderer_path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in source, f"{renderer_path.name} retains forbidden fallback token {token!r}"

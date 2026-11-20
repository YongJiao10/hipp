#!/usr/bin/env python3
from __future__ import annotations

import sys
import importlib.util
import inspect
import subprocess
from pathlib import Path

import pytest

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


cortex_mod = _load_numbered_module("06_03_build_cluster_cortex_panels.py", "core_06_03_build_cluster_cortex_panels")
present_mod = _load_numbered_module("06_01_build_group_overview.py", "core_06_01_build_group_overview")
atlas_mod = _load_numbered_module("05_02_run_group_atlas.py", "core_05_02_run_group_atlas")
matrix_mod = _load_numbered_module("07_01_build_group_matrix.py", "core_07_01_build_group_matrix")

CORTEX_CLUSTERLESS_METHODS = getattr(cortex_mod, "CLUSTERLESS_METHODS", set())
CORTEX_DEFAULTS = cortex_mod.METHODS_DEFAULT
CORTEX_METHOD_CHOICES = cortex_mod.METHOD_CHOICES
PRESENT_CLUSTERLESS_METHODS = present_mod.CLUSTERLESS_METHODS
PRESENT_DEFAULTS = present_mod.METHODS_DEFAULT
PRESENT_METHOD_CHOICES = present_mod.METHOD_CHOICES
CLUSTERLESS_METHODS = atlas_mod.CLUSTERLESS_METHODS
METHODS_DEFAULT = atlas_mod.METHODS_DEFAULT
METHOD_CHOICES = atlas_mod.METHOD_CHOICES
canonical_json_sha256 = atlas_mod.canonical_json_sha256
method_uses_network_fc_input = atlas_mod.method_uses_network_fc_input
parse_gradient_components = atlas_mod.parse_gradient_components
resolve_subject_fc_mode = atlas_mod.resolve_subject_fc_mode
subject_fc_preprocessing_manifest_block = atlas_mod.subject_fc_preprocessing_manifest_block
build_render_jobs = matrix_mod.build_render_jobs
expected_overview_path = matrix_mod.expected_overview_path

OLD_METHOD_NAMES = {"group-network-top", "group-network-spectral", "group-intrinsic-spectral"}
NEW_METHOD_NAMES = {"wta-top", "network-spectral", "intrinsic-spectral"}
GRADIENT_METHOD = "gradient-spectral"
DEFAULT_METHODS = ["intrinsic-spectral", "network-spectral", "wta-top"]


def test_signed_profile_heatmap_annotations_preserve_negative_values() -> None:
    import numpy as np

    values = np.array([[-0.10, 0.20]])
    fig, ax = present_mod.plt.subplots()
    try:
        ax.imshow(values, vmin=-0.20, vmax=0.20)
        present_mod.annotate_profile_values(ax, values, vmax=0.20)
        assert [text.get_text() for text in ax.texts] == ["-0.10", "0.20"]
    finally:
        present_mod.plt.close(fig)


def test_main_pipeline_method_contract() -> None:
    assert NEW_METHOD_NAMES.issubset(set(METHOD_CHOICES))
    assert GRADIENT_METHOD in set(METHOD_CHOICES)
    assert OLD_METHOD_NAMES.isdisjoint(set(METHOD_CHOICES))
    assert METHODS_DEFAULT == DEFAULT_METHODS
    assert CLUSTERLESS_METHODS == {"wta-top"}
    assert method_uses_network_fc_input("wta-top") is True
    assert method_uses_network_fc_input(GRADIENT_METHOD) is True


def test_downstream_builders_method_contract() -> None:
    assert NEW_METHOD_NAMES.issubset(set(PRESENT_METHOD_CHOICES))
    assert GRADIENT_METHOD in set(PRESENT_METHOD_CHOICES)
    assert OLD_METHOD_NAMES.isdisjoint(set(PRESENT_METHOD_CHOICES))
    assert "wta-top" not in set(CORTEX_METHOD_CHOICES)
    assert GRADIENT_METHOD in set(CORTEX_METHOD_CHOICES)
    assert OLD_METHOD_NAMES.isdisjoint(set(CORTEX_METHOD_CHOICES))
    assert PRESENT_DEFAULTS == DEFAULT_METHODS
    assert set(CORTEX_DEFAULTS) == {"intrinsic-spectral", "network-spectral", "gradient-spectral"}
    assert CORTEX_CLUSTERLESS_METHODS == set()
    assert PRESENT_CLUSTERLESS_METHODS == {"wta-top"}


def test_downstream_renderers_do_not_accept_legacy_group_prefixed_intermediates(tmp_path: Path) -> None:
    for mod in (present_mod, cortex_mod):
        assert not hasattr(mod, "LEGACY_METHOD_DIRS")
        assert not hasattr(mod, "resolve_method_root")


def test_overview_matrix_uses_core_work_fixed_k3_fisherz_root_for_all_routea_methods() -> None:
    jobs = build_render_jobs(REPO_ROOT / "outputs_hcp" / "inputs" / "kong2022")
    by_method = {job.method: job for job in jobs if job.route_tag == "pfm"}
    assert set(by_method) == {"intrinsic-spectral", "network-spectral", "gradient-spectral", "wta-top"}
    for method in ("intrinsic-spectral", "network-spectral", "gradient-spectral", "wta-top"):
        root_text = str(by_method[method].group_root)
        assert root_text.endswith("outputs_hcp/work/group_atlas/pfm")
        assert "fixedk_work" not in root_text
        assert "hipp_group_merged_canonical_atlas_routea_full" not in root_text


def test_routeb_group_atlas_root_is_core_work_fixed_k3_fisherz() -> None:
    jobs = build_render_jobs(
        REPO_ROOT / "outputs_hcp" / "work" / "group_atlas",
        routes=("fixed",),
    )
    assert {job.route_tag for job in jobs} == {"fixed"}
    for job in jobs:
        root_text = str(job.group_root)
        assert root_text.endswith("outputs_hcp/work/group_atlas/fixed")
        assert "outputs_hcp/inputs" not in root_text
        assert "hipp_group_merged_canonical_atlas_routeb_full" not in root_text
        assert "routeb_auto" not in root_text


def test_overview_matrix_uses_formal_hcp_control_roots() -> None:
    jobs = build_render_jobs(
        REPO_ROOT / "outputs_hcp" / "work" / "group_atlas",
        routes=("pfm_ctxreg",),
        methods=("gradient-spectral", "wta-top"),
    )
    expected = {
        "pfm_ctxreg": "pfm_ctxreg",
    }
    assert jobs
    for job in jobs:
        assert expected[job.route_tag] in str(job.group_root)
        assert "_full_k3_fisherz" not in str(job.group_root)


def test_default_control_out_root_uses_formal_kong400_roots() -> None:
    from scripts.core_workflow.lib.route_tags import canonical_group_atlas_root_for_route

    assert str(canonical_group_atlas_root_for_route("pfm_ctxreg")).endswith(
        "pfm_ctxreg"
    )


def test_non_hcp_control_roots_keep_their_dataset_generic_contract() -> None:
    from scripts.core_workflow.lib.route_tags import canonical_group_atlas_root_for_route

    assert str(canonical_group_atlas_root_for_route("pfm_ctxreg", dataset="stanford")).endswith(
        "pfm_ctxreg"
    )


def test_group_atlas_default_out_root_matches_routea_fixed_k3_contract() -> None:
    root_text = str(atlas_mod.DEFAULT_OUT_ROOT)
    assert root_text.endswith("outputs_hcp/work/group_atlas/pfm")
    assert "routea_auto" not in root_text


def test_overview_matrix_custom_group_atlas_root_uses_canonical_branch_children() -> None:
    custom_root = Path("/tmp/custom_group_atlas")
    jobs = build_render_jobs(custom_root, routes=("pfm", "fixed"), methods=("wta-top",))
    by_route = {job.route_tag: job for job in jobs}
    assert by_route["pfm"].group_root == (custom_root / "pfm").resolve()
    assert by_route["fixed"].group_root == (custom_root / "fixed").resolve()


def test_overview_matrix_creates_family17_wta_jobs() -> None:
    jobs = build_render_jobs(
        REPO_ROOT / "outputs_hcp" / "work" / "group_atlas",
        routes=("pfm_17net", "fixed_17net"),
        methods=("wta-top",),
    )
    assert {(job.route_tag, job.smoothing) for job in jobs} == {
        (route, smoothing) for route in ("pfm_17net", "fixed_17net") for smoothing in ("2mm", "4mm")
    }
    assert {job.assignment_variant for job in jobs} == {None}


def test_overview_matrix_defaults_to_routea_only() -> None:
    jobs = build_render_jobs(REPO_ROOT / "outputs_hcp" / "inputs" / "kong2022")
    assert {job.route_tag for job in jobs} == {"pfm"}
    assert len(jobs) == 14


def test_overview_matrix_can_filter_methods_for_gradient_and_wta_only() -> None:
    jobs = build_render_jobs(
        REPO_ROOT / "outputs_hcp" / "inputs" / "kong2022",
        methods=("gradient-spectral", "wta-top"),
    )
    assert {job.route_tag for job in jobs} == {"pfm"}
    assert {job.method for job in jobs} == {"gradient-spectral", "wta-top"}
    assert len(jobs) == 6


def test_overview_matrix_defaults_to_network_ts_fisherz_and_normalized_top20() -> None:
    jobs = build_render_jobs(REPO_ROOT / "outputs_hcp" / "inputs" / "kong2022")
    by_method = {}
    for job in jobs:
        by_method.setdefault(job.method, []).append(job.assignment_variant)
    for method in ("intrinsic-spectral", "network-spectral", "gradient-spectral"):
        assert sorted(by_method[method]) == [
            "network_ts_fisherz", "network_ts_fisherz", "normalized_top20_count", "normalized_top20_count"
        ]
    assert "fisherz" in set(matrix_mod.ASSIGNMENT_VARIANTS)
    assert by_method["wta-top"] == [None, None]
    assert "count" not in set(present_mod.ASSIGNMENT_VARIANT_CHOICES)
    assert "count" not in set(matrix_mod.ASSIGNMENT_VARIANTS)
    out = expected_overview_path(
        Path("/tmp/present"),
        method="network-spectral",
        route_tag="pfm",
        mode_tag=None,
        assignment_variant="normalized_top20_count", smoothing="2mm",
    )
    assert out.name == "kong2022_network-spectral_group_overview_pfm_normalized_top20_count.png"
    assert out.parent == Path("/tmp/present/pfm/network-spectral/2mm")


    grad_out = expected_overview_path(
        Path("/tmp/present"),
        method="gradient-spectral",
        route_tag="pfm_ctxreg",
        mode_tag=None,
        assignment_variant="fisherz", smoothing="4mm",
    )
    assert grad_out.name == "roi-fisherz_ovr_overview.png"
    assert grad_out.parent == Path("/tmp/present/pfm_ctxreg/gradient-spectral/4mm")


def test_group_overview_direct_renderer_keeps_gradient_variants_but_wta_is_canonical() -> None:
    assert present_mod.resolve_render_assignment_variants("gradient-spectral", "both") == ("network_ts_fisherz", "normalized_top20_count")
    assert inspect.signature(present_mod.render_combo).parameters["assignment_variant"].default is None
    assert present_mod.default_render_assignment_variant("gradient-spectral", "kong2022") == "network_ts_fisherz"
    assert present_mod.default_render_assignment_variant("gradient-spectral", "lynch2024_7net") == "fisherz"
    assert present_mod.default_render_assignment_variant("wta-top", "kong2022") == "normalized_top20_count"
    assert present_mod.default_render_assignment_variant("wta-top", "lynch2024_7net") == "fisherz"
    assert present_mod.resolve_render_assignment_variants("network-spectral", "normalized_top20_count") == ("normalized_top20_count",)
    assert present_mod.resolve_render_assignment_variants("wta-top", "both") == ("normalized_top20_count",)
    assert present_mod.effective_overview_mode_tag("wta-top", "normalized_top20_count", "legacy") is None
    with pytest.raises(ValueError, match="normalized_top20_count"):
        present_mod.resolve_render_assignment_variants("wta-top", "network_ts_fisherz")




def test_group_overview_wta_requires_route_taxonomy() -> None:
    paths = [Path("/tmp/hemi_L/group_atlas_manifest.json"), Path("/tmp/hemi_R/group_atlas_manifest.json")]
    family17 = [{"network_taxonomy": "family17"}, {"network_taxonomy": "family17"}]
    canonical7 = [{"network_taxonomy": "canonical7"}, {"network_taxonomy": "canonical7"}]
    present_mod.validate_kong_wta_taxonomy(paths, family17, "pfm_17net")
    present_mod.validate_kong_wta_taxonomy(paths, family17, "fixed_17net")
    present_mod.validate_kong_wta_taxonomy(paths, canonical7, "pfm")
    with pytest.raises(ValueError, match="expected 'family17'"):
        present_mod.validate_kong_wta_taxonomy(paths, canonical7, "pfm_17net")
    with pytest.raises(ValueError, match="expected 'canonical7'"):
        present_mod.validate_kong_wta_taxonomy(paths, family17, "pfm")


def test_group_overview_direct_renderer_defaults_both_to_network_ts_fisherz() -> None:
    assert present_mod.resolve_render_assignment_variants("gradient-spectral", "network_ts_fisherz") == ("network_ts_fisherz",)
    assert present_mod.resolve_render_assignment_variants("network-spectral", "network_ts_fisherz") == ("network_ts_fisherz",)
    assert "weighted_fisherz" not in present_mod.ASSIGNMENT_VARIANTS
    assert present_mod.resolve_render_assignment_variants("gradient-spectral", "both") == ("network_ts_fisherz", "normalized_top20_count")


def test_group_overview_keeps_explicit_mode_tag_for_full_root(tmp_path: Path) -> None:
    group_root = tmp_path / "comparison_full_k3"
    assert present_mod.resolve_overview_mode_tag(group_root, "comparison") == "comparison"
    assert present_mod.resolve_overview_mode_tag(group_root, None) is None

def test_group_overview_assignment_variant_paths_reject_stale_manifest_paths_without_fallback(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    asset_dir = run_dir / "template" / "workbench_assets"
    asset_dir.mkdir(parents=True)
    status = run_dir / "cluster_assignment_status_normalized_top20_count.json"
    label = asset_dir / "group-average_hemi-L_label_normalized_top20_count.gii"
    status.write_text("{}\n", encoding="utf-8")
    label.write_text("label\n", encoding="utf-8")
    manifest = {
        "assignment_variants": {
            "normalized_top20_count": {
                "cluster_assignment_status": str(tmp_path / "stale" / "cluster_assignment_status_normalized_top20_count.json"),
                "template_functional_atlas_label": str(tmp_path / "stale" / "template" / "workbench_assets" / label.name),
            }
        }
    }

    with pytest.raises(ValueError, match="points outside run directory"):
        present_mod.resolve_assignment_variant_paths(run_dir, manifest, "normalized_top20_count")


def test_wta_profile_is_embedded_in_overview_with_fixed_20_bins() -> None:
    assert present_mod.WTA_OVERVIEW_PROFILE_BINS == 20
    assert present_mod.WTA_OVERVIEW_SIDE_PANEL_FONT_DELTA_PT == 4
    assert present_mod.WTA_OVERVIEW_SIDE_PANEL_TITLE_FS == present_mod.FS_TITLE + 4
    assert present_mod.WTA_OVERVIEW_SIDE_PANEL_LABEL_FS == present_mod.FS_LABEL + 4
    assert present_mod.WTA_OVERVIEW_SIDE_PANEL_TICK_FS == present_mod.FS_TICK + 4
    assert not hasattr(matrix_mod, "WTA_PROFILE_SCRIPT")
    assert not hasattr(matrix_mod, "WTA_PROFILE_BINS")
    assert not hasattr(matrix_mod, "wta_profile_command")


def test_cortex_scene_renderer_help_works_when_executed_by_path() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "core_workflow" / "lib" / "render_cortex_pfm_scene.py"),
            "--help",
        ],
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--source-scene" in proc.stdout


def test_gradient_component_parser_defaults_and_valid_combos() -> None:
    assert parse_gradient_components(None) == (1, 2, 3)
    assert parse_gradient_components("1") == (1,)
    assert parse_gradient_components("2") == (2,)
    assert parse_gradient_components("3") == (3,)
    assert parse_gradient_components("1,2") == (1, 2)
    assert parse_gradient_components("1,2,3") == (1, 2, 3)


def test_gradient_component_parser_rejects_bad_values() -> None:
    for bad in ["", "0", "4", "1,1", "3,1", "g1", "1,,2"]:
        try:
            parse_gradient_components(bad)
            raise AssertionError(f"expected invalid gradient component spec to fail: {bad}")
        except ValueError:
            pass


def test_gradient_overview_readiness_checks_each_component_hemi(tmp_path: Path) -> None:
    base = tmp_path / GRADIENT_METHOD / "kong2022" / "2mm" / "cov_0p80" / "g1"
    (base / "hemi_L").mkdir(parents=True, exist_ok=True)
    run_dir = base / "hemi_R"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "group_atlas_manifest.json").write_text('{"status": "ok"}\n', encoding="utf-8")
    (run_dir / "group_labels_final.npy").write_bytes(b"placeholder")

    try:
        present_mod.ensure_combo_ready(
            group_root=tmp_path,
            method=GRADIENT_METHOD,
            atlas="kong2022",
            coverage_tag="cov_0p80",
            smoothings=["2mm"],
            hemis=["L", "R"],
            gradient_comp_tag="g1",
        )
        raise AssertionError("expected missing hemi_L manifest to fail readiness")
    except RuntimeError as exc:
        assert "g1/hemi_L/group_atlas_manifest.json" in str(exc)


def test_wta_subject_fc_preprocessing_manifest_is_mode_neutral() -> None:
    full_mode = resolve_subject_fc_mode("full", None)
    sparse_mode = resolve_subject_fc_mode("top-percent", 10)
    wta_full = subject_fc_preprocessing_manifest_block("wta-top", full_mode)
    wta_sparse = subject_fc_preprocessing_manifest_block("wta-top", sparse_mode)
    assert canonical_json_sha256(wta_full) == canonical_json_sha256(wta_sparse)

    network_full = subject_fc_preprocessing_manifest_block("network-spectral", full_mode)
    network_sparse = subject_fc_preprocessing_manifest_block("network-spectral", sparse_mode)
    assert canonical_json_sha256(network_full) != canonical_json_sha256(network_sparse)


def test_gradient_subject_fc_preprocessing_manifest_is_mode_neutral() -> None:
    full_mode = resolve_subject_fc_mode("full", None)
    sparse_mode = resolve_subject_fc_mode("top-percent", 10)
    gradient_full = subject_fc_preprocessing_manifest_block(GRADIENT_METHOD, full_mode)
    gradient_sparse = subject_fc_preprocessing_manifest_block(GRADIENT_METHOD, sparse_mode)
    assert canonical_json_sha256(gradient_full) == canonical_json_sha256(gradient_sparse)


def test_gradient_matrix_jobs_are_route_only_and_core_rooted_for_routea() -> None:
    jobs = build_render_jobs(REPO_ROOT / "outputs_hcp" / "inputs" / "kong2022")
    gradient_jobs = [job for job in jobs if job.method == GRADIENT_METHOD]
    assert gradient_jobs
    assert {job.route_tag for job in gradient_jobs}.issubset({"pfm", "fixed"})
    assert "pfm" in {job.route_tag for job in gradient_jobs}
    assert all(job.mode_tag is None for job in gradient_jobs)
    pfm_jobs = [job for job in gradient_jobs if job.route_tag == "pfm"]
    assert pfm_jobs
    assert all(str(job.group_root).endswith("outputs_hcp/work/group_atlas/pfm") for job in pfm_jobs)
    assert all("10perc" not in str(job.group_root) for job in gradient_jobs)


def main() -> int:
    test_main_pipeline_method_contract()
    test_downstream_builders_method_contract()
    test_downstream_renderers_do_not_accept_legacy_group_prefixed_intermediates(Path("/tmp/core_workflow_method_contract"))
    test_overview_matrix_uses_core_work_fixed_k3_fisherz_root_for_all_routea_methods()
    test_overview_matrix_defaults_to_routea_only()
    test_overview_matrix_keeps_normalized_top20_and_fisherz_assignment_variants()
    test_wta_profile_is_embedded_in_overview_with_fixed_20_bins()
    test_cortex_scene_renderer_help_works_when_executed_by_path()
    test_gradient_component_parser_defaults_and_valid_combos()
    test_gradient_component_parser_rejects_bad_values()
    test_gradient_overview_readiness_checks_each_component_hemi(Path("/tmp/core_workflow_gradient_overview_ready"))
    test_wta_subject_fc_preprocessing_manifest_is_mode_neutral()
    test_gradient_subject_fc_preprocessing_manifest_is_mode_neutral()
    test_gradient_matrix_jobs_are_route_only_and_core_rooted_for_routea()
    print("ok: method naming contract checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

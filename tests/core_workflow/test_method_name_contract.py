#!/usr/bin/env python3
from __future__ import annotations

import sys
import importlib.util
import subprocess
from pathlib import Path

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


cortex_mod = _load_numbered_module("06_04_build_cluster_cortex_panels.py", "core_06_04_build_cluster_cortex_panels")
present_mod = _load_numbered_module("06_01_build_group_overview.py", "core_06_01_build_group_overview")
atlas_mod = _load_numbered_module("05_02_run_group_atlas.py", "core_05_02_run_group_atlas")
matrix_mod = _load_numbered_module("07_01_build_group_matrix.py", "core_07_01_build_group_matrix")

CORTEX_CLUSTERLESS_METHODS = cortex_mod.CLUSTERLESS_METHODS
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
wta_profile_command = matrix_mod.wta_profile_command

OLD_METHOD_NAMES = {"group-network-top", "group-network-spectral", "group-intrinsic-spectral"}
NEW_METHOD_NAMES = {"wta-top", "network-spectral", "intrinsic-spectral"}
GRADIENT_METHOD = "gradient-spectral"
DEFAULT_METHODS = ["intrinsic-spectral", "network-spectral", "wta-top"]


def test_main_pipeline_method_contract() -> None:
    assert NEW_METHOD_NAMES.issubset(set(METHOD_CHOICES))
    assert GRADIENT_METHOD in set(METHOD_CHOICES)
    assert OLD_METHOD_NAMES.isdisjoint(set(METHOD_CHOICES))
    assert METHODS_DEFAULT == DEFAULT_METHODS
    assert CLUSTERLESS_METHODS == {"wta-top"}
    assert method_uses_network_fc_input("wta-top") is True
    assert method_uses_network_fc_input(GRADIENT_METHOD) is True


def test_downstream_builders_method_contract() -> None:
    for choices in (CORTEX_METHOD_CHOICES, PRESENT_METHOD_CHOICES):
        assert NEW_METHOD_NAMES.issubset(set(choices))
        assert GRADIENT_METHOD in set(choices)
        assert OLD_METHOD_NAMES.isdisjoint(set(choices))
    for defaults in (CORTEX_DEFAULTS, PRESENT_DEFAULTS):
        assert defaults == DEFAULT_METHODS
    for clusterless in (CORTEX_CLUSTERLESS_METHODS, PRESENT_CLUSTERLESS_METHODS):
        assert clusterless == {"wta-top"}


def test_downstream_renderers_do_not_accept_legacy_group_prefixed_intermediates(tmp_path: Path) -> None:
    for mod in (present_mod, cortex_mod):
        assert not hasattr(mod, "LEGACY_METHOD_DIRS")
        assert not hasattr(mod, "resolve_method_root")


def test_overview_matrix_uses_core_work_fixed_k3_fisherz_root_for_all_routea_methods() -> None:
    jobs = build_render_jobs(REPO_ROOT / "outputs_core" / "inputs" / "hcp7t_hippocampus" / "kong2022")
    by_method = {job.method: job for job in jobs if job.route_tag == "routea"}
    assert set(by_method) == {"intrinsic-spectral", "network-spectral", "gradient-spectral", "wta-top"}
    for method in ("intrinsic-spectral", "network-spectral", "gradient-spectral", "wta-top"):
        root_text = str(by_method[method].group_root)
        assert "outputs_core/work/group_atlas/routea_full_k3_fisherz" in root_text
        assert "fixedk_work" not in root_text
        assert "hipp_group_merged_canonical_atlas_routea_full" not in root_text


def test_overview_matrix_defaults_to_routea_only() -> None:
    jobs = build_render_jobs(REPO_ROOT / "outputs_core" / "inputs" / "hcp7t_hippocampus" / "kong2022")
    assert {job.route_tag for job in jobs} == {"routea"}
    assert len(jobs) == 7


def test_overview_matrix_keeps_count_and_fisherz_assignment_variants() -> None:
    jobs = build_render_jobs(REPO_ROOT / "outputs_core" / "inputs" / "hcp7t_hippocampus" / "kong2022")
    by_method = {}
    for job in jobs:
        by_method.setdefault(job.method, []).append(job.assignment_variant)
    for method in ("intrinsic-spectral", "network-spectral", "gradient-spectral"):
        assert sorted(by_method[method]) == ["count", "fisherz"]
    assert by_method["wta-top"] == [None]
    out = expected_overview_path(
        Path("/tmp/present"),
        method="network-spectral",
        route_tag="routea",
        mode_tag=None,
        assignment_variant="count",
    )
    assert out.name == "kong2022_network-spectral_group_overview_routea_count.png"


def test_overview_matrix_generates_wta_long_axis_profiles_from_core_workflow() -> None:
    cmd = wta_profile_command(
        python_exe="/home/yojiao/.conda/envs/py314/bin/python",
        group_root=REPO_ROOT / "outputs_core" / "work" / "group_atlas" / "routea_full_k3_fisherz",
        present_dir=REPO_ROOT / "outputs_core" / "present" / "overviews",
        route_tag="routea",
        coverage_threshold=0.80,
    )
    joined = " ".join(cmd)
    assert "06_02_build_wta_long_axis_profiles.py" in joined
    assert "--bins 20 25" in joined
    assert ("present_group_" + "t1only") not in joined
    assert ("outputs_" + "experiments") not in joined


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
    jobs = build_render_jobs(REPO_ROOT / "outputs_core" / "inputs" / "hcp7t_hippocampus" / "kong2022")
    gradient_jobs = [job for job in jobs if job.method == GRADIENT_METHOD]
    assert gradient_jobs
    assert set(job.route_tag for job in gradient_jobs).issubset({"routea", "routeb"})
    assert "routea" in {job.route_tag for job in gradient_jobs}
    assert all(job.mode_tag is None for job in gradient_jobs)
    routea_jobs = [job for job in gradient_jobs if job.route_tag == "routea"]
    assert routea_jobs
    assert all("outputs_core/work/group_atlas/routea_full_k3_fisherz" in str(job.group_root) for job in routea_jobs)
    assert all("10perc" not in str(job.group_root) for job in gradient_jobs)


def main() -> int:
    test_main_pipeline_method_contract()
    test_downstream_builders_method_contract()
    test_downstream_renderers_do_not_accept_legacy_group_prefixed_intermediates(Path("/tmp/core_workflow_method_contract"))
    test_overview_matrix_uses_core_work_fixed_k3_fisherz_root_for_all_routea_methods()
    test_overview_matrix_defaults_to_routea_only()
    test_overview_matrix_keeps_count_and_fisherz_assignment_variants()
    test_overview_matrix_generates_wta_long_axis_profiles_from_core_workflow()
    test_cortex_scene_renderer_help_works_when_executed_by_path()
    test_gradient_component_parser_defaults_and_valid_combos()
    test_gradient_component_parser_rejects_bad_values()
    test_wta_subject_fc_preprocessing_manifest_is_mode_neutral()
    test_gradient_subject_fc_preprocessing_manifest_is_mode_neutral()
    test_gradient_matrix_jobs_are_route_only_and_core_rooted_for_routea()
    print("ok: method naming contract checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

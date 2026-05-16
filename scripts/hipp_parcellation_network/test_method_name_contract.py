#!/usr/bin/env python3
from __future__ import annotations

import sys
import importlib.util
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
atlas_mod = _load_numbered_module("05_01_run_group_atlas.py", "core_05_01_run_group_atlas")

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


def main() -> int:
    test_main_pipeline_method_contract()
    test_downstream_builders_method_contract()
    test_gradient_component_parser_defaults_and_valid_combos()
    test_gradient_component_parser_rejects_bad_values()
    test_wta_subject_fc_preprocessing_manifest_is_mode_neutral()
    print("ok: method naming contract checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

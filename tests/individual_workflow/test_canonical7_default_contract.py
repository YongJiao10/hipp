from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_individual_network_helper_supports_canonical7_order():
    workflow = load_module(
        REPO_ROOT / "scripts/individual_workflow/lib/routea_individual_gradients_wta.py",
        "routea_individual_gradients_wta_canonical7_contract",
    )
    names, keys = workflow.load_network_names_and_keys(
        [{"canonical7": name} for name in workflow.canonical_families_7_order()],
        "canonical7",
    )

    assert names == workflow.canonical_families_7_order()
    assert keys["Language"] == 7


def test_individual_overview_builder_defaults_to_canonical7():
    builder = load_module(
        REPO_ROOT / "scripts/individual_workflow/02_build_individual_overviews.py",
        "build_individual_overviews_canonical7_contract",
    )

    assert builder.build_arg_parser().parse_args([]).network_column == "canonical7"

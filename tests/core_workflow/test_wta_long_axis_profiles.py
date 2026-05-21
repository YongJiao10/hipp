#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


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


wta_mod = _load_module(
    "scripts/core_workflow/06_02_build_wta_long_axis_profiles.py",
    "core_06_02_build_wta_long_axis_profiles",
)


def test_wta_long_axis_profile_rows_preserve_bin_percent_semantics() -> None:
    labels_by_hemi = {
        "L": np.asarray([1, 2, 2, 9, 1, 0], dtype=np.int32),
        "R": np.asarray([2, 2, 1, 1, 9, 0], dtype=np.int32),
    }
    order_by_hemi = {
        "L": np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64),
        "R": np.asarray([5, 4, 3, 2, 1, 0], dtype=np.int64),
    }
    rows, bin_sizes = wta_mod.compute_profile_rows(
        labels_by_hemi=labels_by_hemi,
        order_by_hemi=order_by_hemi,
        network_names=["Default", "Visual"],
        network_label_keys={"Default": 1, "Visual": 2},
        smoothing="2mm",
        n_bins=3,
    )

    assert bin_sizes == {"L": [2, 2, 2], "R": [2, 2, 2]}
    lookup = {(r["hemi"], int(r["bin_index"]), r["network_name"]): r for r in rows}
    assert lookup[("L", 1, "Default")]["network_vertex_count"] == 1
    assert lookup[("L", 1, "Default")]["percent_of_bin_vertices"] == "50.000000"
    assert lookup[("L", 2, "__non_network_or_null__")]["network_vertex_count"] == 1
    assert lookup[("R", 1, "__non_network_or_null__")]["percent_of_bin_vertices"] == "100.000000"


def test_wta_long_axis_output_paths_match_outputs_core_contract() -> None:
    paths = wta_mod.profile_output_paths(
        present_dir=REPO_ROOT / "outputs_core" / "present" / "overviews",
        atlas="kong2022",
        method="wta-top",
        route_tag="routea",
        smoothing="4mm",
        n_bins=25,
    )
    assert paths["png"].name == "kong2022_wta-top_long_axis_network_profile_routea_4mm_bins25.png"
    assert paths["tsv"].name == "kong2022_wta-top_long_axis_network_profile_routea_4mm_bins25.tsv"


def main() -> int:
    test_wta_long_axis_profile_rows_preserve_bin_percent_semantics()
    test_wta_long_axis_output_paths_match_outputs_core_contract()
    print("ok: wta long-axis profile checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

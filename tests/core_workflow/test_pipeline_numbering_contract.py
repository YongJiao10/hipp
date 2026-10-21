#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE = REPO_ROOT / "scripts" / "core_workflow"

EXPECTED_NUMBERED_SCRIPTS = [
    "01_01_run_hippunfold_t1w_local.py",
    "01_02_check_hippunfold_outputs.py",
    "02_01_prepare_group_inputs.py",
    "02_02_build_qc_manifest.py",
    "02_03_build_lynch2024_nsi_manifest.py",
    "03_01_build_kong2022_reference.py",
    "03_02_build_kong2022_routea_reference.py",
    "03_03_build_lynch2024_reference.py",
    "03_04_run_lynch2024_fastans_pfm.py",
    "04_00_materialize_cortex_regressed_surface_timeseries.py",
    "04_01_materialize_kong2022_roi_fc.py",
    "04_02_materialize_intrinsic_fc.py",
    "04_03_materialize_task_activation.py",
    "04_04_materialize_background_fc.py",
    "05_01_compute_hippomaps_gradients.py",
    "05_02_run_group_atlas.py",
    "05_03_run_spectral_fast_rerun.py",
    "05_04_compute_confidence_maps.py",
    "05_05_build_weighted_fisherz_branch.py",
    "05_06_build_task_activation_group.py",
    "06_01_build_group_overview.py",
    "06_02_materialize_group_labels.py",
    "06_03_build_cluster_cortex_panels.py",
    "06_04_render_hippomaps_gradient_products.py",
    "06_05_render_group_gradient_heatmaps.py",
    "06_06_render_confidence_maps.py",
    "06_07_render_ctxdrop_neighbor_counts.py",
    "06_08_render_hcp_subfield_composition.py",
    "06_09_plot_cortex_kong2022_mshbm.py",
    "06_10_render_cluster_roi_strength_panels.py",
    "06_11_render_task_activation_products.py",
    "07_01_build_group_matrix.py",
]

def test_numbered_core_workflow_scripts_are_unique_and_ordered() -> None:
    scripts = sorted(p.name for p in CORE.glob("[0-9][0-9]_*_*.py"))
    assert scripts == EXPECTED_NUMBERED_SCRIPTS
    prefixes = ["_".join(name.split("_")[:2]) for name in scripts]
    assert len(prefixes) == len(set(prefixes)), prefixes
    assert not any(name.startswith("00_") for name in scripts)


def test_registry_keeps_smoke_out_of_numbered_pipeline() -> None:
    registry = json.loads((CORE / "step_registry.json").read_text(encoding="utf-8"))
    step_ids = [row["step_id"] for row in registry["steps"]]
    assert step_ids == [name.removesuffix(".py") for name in EXPECTED_NUMBERED_SCRIPTS]
    assert "smoke_prepare_run" not in step_ids
    assert not (CORE / "smoke_prepare_run.py").exists()
    utilities = {row["step_id"]: row for row in registry.get("utilities", [])}
    assert utilities["smoke_prepare_run"]["entry_script"] == "tests/core_workflow/smoke_prepare_run.py"
    assert utilities["smoke_prepare_run"]["implementation_script"] == "tests/core_workflow/smoke_prepare_run.py"


def test_kong2022_canonical7_is_the_unsuffixed_routea_branch() -> None:
    registry = json.loads((CORE / "step_registry.json").read_text(encoding="utf-8"))
    step_ids = {row["step_id"] for row in registry["steps"]}
    utility_ids = {row["step_id"] for row in registry.get("utilities", [])}

    assert "run_kong2022_7net_branch" not in step_ids
    assert "run_kong2022_7net_branch" not in utility_ids
    assert not (CORE / "run_kong2022_7net.py").exists()
    assert not (CORE / "kong2022_7net").exists()

    steps = {row["step_id"]: row for row in registry["steps"]}
    for step_id in (
        "03_02_build_kong2022_routea_reference",
        "05_02_run_group_atlas",
        "05_05_build_weighted_fisherz_branch",
        "06_01_build_group_overview",
    ):
        assert "kong2022_pfm_7net" not in steps[step_id]["branches"]
        assert "kong2022_pfm" in steps[step_id]["branches"]


def main() -> int:
    test_numbered_core_workflow_scripts_are_unique_and_ordered()
    test_registry_keeps_smoke_out_of_numbered_pipeline()
    print("ok: pipeline numbering contract checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

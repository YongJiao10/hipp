from __future__ import annotations

import sys
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from branch_plan import (  # noqa: E402
    build_branch_plan,
    can_start,
    run_priority_batches,
)


CONFIG = {
    "unified_branch_policy": {
        "method_order": [
            "hcp151_fixed",
            "hcp151_template_matching",
            "group_atlas_gradient_spectral",
        ],
        "subject_fc_estimands": {
            "jjtms": ["patient-healthy"],
            "stanford": ["symptom_marginal"],
        },
        "subject_fc_priority": [
            {"tier": 0, "network_resolution": 8, "smoothing_mm": 2},
            {"tier": 1, "network_resolution": 8, "smoothing_mm": 4},
            {"tier": 1, "network_resolution": 17, "smoothing_mm": 2},
            {"tier": 2, "network_resolution": 17, "smoothing_mm": 4},
        ],
        "group_atlas_priority": [
            {"tier": 0, "smoothing_mm": 2},
            {"tier": 2, "smoothing_mm": 4},
        ],
        "barrier": "strict",
    },
    "datasets": {"jjtms": {}, "stanford": {}},
    "smoothings_mm": [2, 4],
    "hemispheres": ["L", "R"],
    "group_atlas": {
        "method": "group_atlas_gradient_spectral",
        "reference_group": "HC",
        "groups": [
            "HC",
            "NTHC",
            "TEHC",
            "NTS",
            "NTS Any MDD",
            "NTS Pure MDD",
            "TIS",
            "TIS PTSD",
        ],
        "assignment_variants": ["plain", "weighted"],
        "smoothings_mm": [2, 4],
        "hemispheres": ["L", "R"],
    },
}


def test_unified_plan_has_three_methods_and_excludes_individual_gradient() -> None:
    plan = build_branch_plan(CONFIG)
    methods = {branch.method for branch in plan}
    assert methods == {
        "hcp151_fixed",
        "hcp151_template_matching",
        "group_atlas_gradient_spectral",
    }
    assert all("individual_gradient" not in branch.method for branch in plan)


def test_group_atlas_expands_each_hemisphere_as_a_branch_cell() -> None:
    plan = build_branch_plan(CONFIG)
    atlas = [branch for branch in plan if branch.analysis_family == "group_atlas_geometry"]

    assert len(atlas) == 8 * 2 * 2 * 2
    assert {branch.hemisphere_modes for branch in atlas} == {("L",), ("R",)}
    assert len({branch.branch_id for branch in atlas}) == len(atlas)


def test_priority_order_puts_all_8net_2mm_before_lower_tiers() -> None:
    plan = build_branch_plan(CONFIG)
    tiers = [branch.priority_tier for branch in plan]
    assert tiers == sorted(tiers)
    assert all(
        branch.priority_tier == 0
        for branch in plan
        if branch.analysis_family == "subject_fc"
        and branch.network_resolution == 8
        and branch.smoothing_mm == 2
    )
    assert all(
        branch.priority_tier > 0
        for branch in plan
        if branch.analysis_family == "subject_fc"
        and (branch.network_resolution == 17 or branch.smoothing_mm == 4)
    )


def test_lower_tier_is_blocked_until_every_higher_tier_is_complete() -> None:
    plan = build_branch_plan(CONFIG)
    low = next(branch for branch in plan if branch.priority_tier == 1)
    statuses = {branch.branch_id: "complete" for branch in plan if branch.priority_tier == 0}
    assert can_start(low.branch_id, statuses, plan)
    statuses[next(branch.branch_id for branch in plan if branch.priority_tier == 0)] = "running"
    assert not can_start(low.branch_id, statuses, plan)


def test_priority_batches_stop_before_lower_tier_after_high_tier_failure() -> None:
    calls: list[str] = []
    plan = build_branch_plan(CONFIG)

    def execute(branch):
        calls.append(branch.branch_id)
        return "failed" if branch.priority_tier == 0 else "complete"

    with pytest.raises(RuntimeError, match="higher-priority"):
        run_priority_batches(plan, execute)
    p0_ids = {branch.branch_id for branch in plan if branch.priority_tier == 0}
    assert set(calls).issubset(p0_ids)

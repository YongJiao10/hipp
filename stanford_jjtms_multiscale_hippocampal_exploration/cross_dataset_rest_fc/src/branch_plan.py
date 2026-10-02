"""Unified, deterministic execution plan for all current analysis branches.

The plan is intentionally an orchestration layer.  Subject-level FC branches
and group-atlas geometry branches remain separate analysis families in their
metadata, but share one ordered plan and one strict priority barrier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping


@dataclass(frozen=True)
class Branch:
    """One independently materialisable analysis cell."""

    analysis_family: str
    method: str
    dataset: str
    network_resolution: int | None
    smoothing_mm: int
    hemisphere_modes: tuple[str, ...]
    group: str | None
    assignment_variant: str | None
    priority_tier: int

    @property
    def branch_id(self) -> str:
        """Return a stable identifier suitable for status ledgers and logs."""

        fields = (
            self.analysis_family,
            self.method,
            self.dataset,
            str(self.network_resolution or "group"),
            f"{self.smoothing_mm}mm",
            ",".join(self.hemisphere_modes),
            self.group or "all",
            self.assignment_variant or "na",
        )
        return "|".join(fields)


def _priority_lookup(
    rows: Iterable[Mapping[str, Any]],
    *,
    keys: tuple[str, ...],
) -> dict[tuple[Any, ...], int]:
    lookup: dict[tuple[Any, ...], int] = {}
    for row in rows:
        key = tuple(row[key_name] for key_name in keys)
        tier = int(row["tier"])
        if tier < 0:
            raise ValueError(f"priority tier must be non-negative: {row!r}")
        if key in lookup and lookup[key] != tier:
            raise ValueError(f"conflicting priority rows for {key!r}")
        lookup[key] = tier
    return lookup


def _subject_priority(
    policy: Mapping[str, Any],
    network_resolution: int,
    smoothing_mm: int,
) -> int:
    lookup = _priority_lookup(
        policy.get("subject_fc_priority", ()),
        keys=("network_resolution", "smoothing_mm"),
    )
    key = (network_resolution, smoothing_mm)
    if key not in lookup:
        raise ValueError(f"missing subject FC priority for {key!r}")
    return lookup[key]


def _group_atlas_priority(policy: Mapping[str, Any], smoothing_mm: int) -> int:
    lookup = _priority_lookup(
        policy.get("group_atlas_priority", ()),
        keys=("smoothing_mm",),
    )
    key = (smoothing_mm,)
    if key not in lookup:
        raise ValueError(f"missing group-atlas priority for {smoothing_mm!r}mm")
    return lookup[key]


def _sort_key(branch: Branch, method_order: Mapping[str, int]) -> tuple[Any, ...]:
    return (
        branch.priority_tier,
        method_order.get(branch.method, len(method_order)),
        branch.analysis_family,
        branch.dataset,
        branch.group or "",
        branch.assignment_variant or "",
        branch.network_resolution or 0,
        branch.smoothing_mm,
        branch.hemisphere_modes,
    )


def build_branch_plan(config: Mapping[str, Any]) -> tuple[Branch, ...]:
    """Expand the current config into one deterministically ordered plan.

    The function accepts the JSON-shaped configuration directly so it can be
    tested without touching data.  Individual-gradient methods are not
    registered here and therefore cannot re-enter the current workflow by
    accident.
    """

    policy = config.get("unified_branch_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("unified_branch_policy is required")

    method_order_values = tuple(str(value) for value in policy.get("method_order", ()))
    method_order = {method: index for index, method in enumerate(method_order_values)}
    if len(method_order) != len(method_order_values):
        raise ValueError("method_order contains duplicates")

    smoothings = tuple(int(value) for value in config.get("smoothings_mm", ()))
    hemispheres = tuple(str(value) for value in config.get("hemispheres", ()))
    if not smoothings or not hemispheres:
        raise ValueError("smoothings_mm and hemispheres must be non-empty")

    branches: list[Branch] = []
    estimands = policy.get("subject_fc_estimands", {})
    datasets = config.get("datasets", {})
    subject_methods = tuple(
        method for method in method_order_values if method != "group_atlas_gradient_spectral"
    )
    for dataset in datasets:
        dataset_estimands = estimands.get(dataset, ()) if isinstance(estimands, Mapping) else ()
        if not dataset_estimands:
            continue
        for method in subject_methods:
            for network_resolution in (8, 17):
                for smoothing_mm in smoothings:
                    branches.append(
                        Branch(
                            analysis_family="subject_fc",
                            method=method,
                            dataset=str(dataset),
                            network_resolution=network_resolution,
                            smoothing_mm=smoothing_mm,
                            hemisphere_modes=("bilateral", "L", "R", "L-R"),
                            group=None,
                            assignment_variant=None,
                            priority_tier=_subject_priority(
                                policy, network_resolution, smoothing_mm
                            ),
                        )
                    )

    atlas = config.get("group_atlas")
    if isinstance(atlas, Mapping):
        atlas_method = str(atlas.get("method", "group_atlas_gradient_spectral"))
        groups = tuple(str(group) for group in atlas.get("groups", ()))
        variants = tuple(str(variant) for variant in atlas.get("assignment_variants", ()))
        atlas_smoothings = tuple(
            int(value) for value in atlas.get("smoothings_mm", smoothings)
        )
        atlas_hemispheres = tuple(
            str(value) for value in atlas.get("hemispheres", hemispheres)
        )
        if atlas_method not in method_order:
            raise ValueError(f"group atlas method is absent from method_order: {atlas_method}")
        if str(atlas.get("reference_group", "")) not in groups:
            raise ValueError("group_atlas reference_group must be included in groups")
        for group in groups:
            for variant in variants:
                for smoothing_mm in atlas_smoothings:
                    for hemisphere in atlas_hemispheres:
                        branches.append(
                            Branch(
                                analysis_family="group_atlas_geometry",
                                method=atlas_method,
                                dataset="stanford",
                                network_resolution=None,
                                smoothing_mm=smoothing_mm,
                                hemisphere_modes=(hemisphere,),
                                group=group,
                                assignment_variant=variant,
                                priority_tier=_group_atlas_priority(policy, smoothing_mm),
                            )
                        )

    if not branches:
        raise ValueError("unified branch plan is empty")
    return tuple(sorted(branches, key=lambda branch: _sort_key(branch, method_order)))


def can_start(
    branch_id: str,
    statuses: Mapping[str, str],
    plan: Iterable[Branch],
) -> bool:
    """Whether a branch may start under the strict priority barrier."""

    branches = tuple(plan)
    by_id = {branch.branch_id: branch for branch in branches}
    if branch_id not in by_id:
        raise KeyError(f"unknown branch id: {branch_id}")
    branch = by_id[branch_id]
    current = statuses.get(branch_id, "pending")
    if current != "pending":
        return False
    return all(
        statuses.get(other.branch_id, "pending") == "complete"
        for other in branches
        if other.priority_tier < branch.priority_tier
    )


def run_priority_batches(
    plan: Iterable[Branch],
    execute_branch: Callable[[Branch], str | None],
    *,
    initial_statuses: Mapping[str, str] | None = None,
) -> tuple[tuple[dict[str, Any], ...], dict[str, str]]:
    """Execute complete priority tiers with a strict success barrier.

    ``execute_branch`` must return ``complete`` for success; ``failed`` (or any
    other value) prevents all lower-priority tiers from starting.  A branch
    already marked complete in ``initial_statuses`` is skipped, which permits
    safe resumption without weakening the barrier.
    """

    branches = tuple(plan)
    statuses = dict(initial_statuses or {})
    records: list[dict[str, Any]] = []
    tier_values = sorted({branch.priority_tier for branch in branches})
    for tier in tier_values:
        tier_branches = tuple(branch for branch in branches if branch.priority_tier == tier)
        if any(
            statuses.get(branch.branch_id, "pending") != "complete"
            for prior in branches
            if prior.priority_tier < tier
            for branch in (prior,)
        ):
            raise RuntimeError(
                f"higher-priority branches are not complete; tier {tier} is blocked"
            )
        for branch in tier_branches:
            branch_id = branch.branch_id
            if statuses.get(branch_id, "pending") == "complete":
                continue
            statuses[branch_id] = "running"
            try:
                result = execute_branch(branch)
            except Exception as exc:
                statuses[branch_id] = "failed"
                records.append(
                    {
                        "branch_id": branch_id,
                        "analysis_family": branch.analysis_family,
                        "priority_tier": branch.priority_tier,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                raise RuntimeError(
                    f"higher-priority branch {branch_id} raised an error; "
                    "lower-priority work is blocked"
                ) from exc
            status = "complete" if result == "complete" else "failed"
            statuses[branch_id] = status
            records.append(
                {
                    "branch_id": branch_id,
                    "analysis_family": branch.analysis_family,
                    "priority_tier": branch.priority_tier,
                    "status": status,
                }
            )
            if status != "complete":
                raise RuntimeError(
                    f"higher-priority branch {branch_id} failed; lower-priority work is blocked"
                )
    return tuple(records), statuses


__all__ = ["Branch", "build_branch_plan", "can_start", "run_priority_batches"]

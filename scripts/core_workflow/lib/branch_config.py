from __future__ import annotations

DEFAULT_BRANCH = "kong2022_pfm"
BRANCH_CHOICES = (
    "kong2022_pfm",
    "kong2022_pfm_17net",
    "kong2022_pfm_ctxreg",
    "kong2022_pfm_ctxreg_10mm",
    "kong2022_fixed",
    "kong2022_fixed_17net",
    "lynch2024_20net",
    "lynch2024_7net",
)


# Signal branches are orthogonal to the cortical atlas/taxonomy.  ``ctxreg``
# (route ``pfm_ctxreg``) regresses local cortex within 20 mm; ``ctxreg_10mm``
# (route ``pfm_ctxreg_10mm``) within 10 mm. The multi-dataset visualization
# release maintains only the standard and 20 mm ctxreg choices.
SIGNAL_BRANCH_CHOICES = ("standard", "ctxreg", "ctxreg_10mm")
SIGNAL_BRANCH_TO_ROUTE = {"standard": "pfm", "ctxreg": "pfm_ctxreg", "ctxreg_10mm": "pfm_ctxreg_10mm"}
ROUTE_TO_SIGNAL_BRANCH = {route: branch for branch, route in SIGNAL_BRANCH_TO_ROUTE.items()}
CTXREG_SIGNAL_BRANCHES = frozenset({"ctxreg", "ctxreg_10mm"})
CTXREG_ROUTES = frozenset(SIGNAL_BRANCH_TO_ROUTE[b] for b in CTXREG_SIGNAL_BRANCHES)
# Local-cortex radius (native T1w mm) that each ctxreg route is defined by.
CTXREG_ROUTE_DISTANCE_MM = {"pfm_ctxreg": 20.0, "pfm_ctxreg_10mm": 10.0}


def is_ctxreg_signal_branch(signal_branch: str | None) -> bool:
    return signal_branch in CTXREG_SIGNAL_BRANCHES


def normalize_signal_branch(signal_branch: str) -> str:
    value = str(signal_branch).strip().lower()
    if value not in SIGNAL_BRANCH_CHOICES:
        raise ValueError(
            f"Unsupported signal branch: {signal_branch!r}; expected one of {SIGNAL_BRANCH_CHOICES}"
        )
    return value


def signal_branch_for_route(route_tag: str) -> str:
    route = str(route_tag).strip().lower()
    if route not in ROUTE_TO_SIGNAL_BRANCH:
        raise ValueError(
            f"Route {route_tag!r} has no maintained standard/ctxreg signal branch; "
            "other routes require their own explicit workflow."
        )
    return ROUTE_TO_SIGNAL_BRANCH[route]


def route_for_signal_branch(signal_branch: str) -> str:
    return SIGNAL_BRANCH_TO_ROUTE[normalize_signal_branch(signal_branch)]


def validate_signal_branch_route(signal_branch: str, route_tag: str) -> str:
    branch = normalize_signal_branch(signal_branch)
    route = str(route_tag).strip().lower()
    expected = SIGNAL_BRANCH_TO_ROUTE[branch]
    if route != expected:
        raise ValueError(
            f"Signal branch/route mismatch: signal_branch={branch!r} requires "
            f"route_tag={expected!r}, got {route!r}"
        )
    return branch


def resolve_signal_branch_for_route(
    route: str, explicit_signal_branch: str | None = None,
) -> str | None:
    """Resolve standard/ctxreg only for the maintained PFM controls."""
    route_key = str(route).strip().lower()
    if route_key in ROUTE_TO_SIGNAL_BRANCH:
        if explicit_signal_branch is None:
            return signal_branch_for_route(route_key)
        return validate_signal_branch_route(
            normalize_signal_branch(explicit_signal_branch), route_key
        )
    if explicit_signal_branch is not None:
        raise ValueError(
            "--signal-branch applies only PFM controls; "
            f"route={route!r} has no maintained {SIGNAL_BRANCH_CHOICES} signal branch"
        )
    return None

from __future__ import annotations

DEFAULT_BRANCH = "kong2022_pfm"
BRANCH_CHOICES = (
    "kong2022_pfm",
    "kong2022_pfm_17net",
    "kong2022_pfm_ctxreg",
    "kong2022_pfm_ctxdrop",
    "kong2022_fixed",
    "kong2022_fixed_17net",
    "lynch2024_20net",
    "lynch2024_7net",
)


# Signal branches are orthogonal to the cortical atlas/taxonomy.  The
# visualization release maintains only these two 2mm/36P choices; core control
# routes retain the separate ctxdrop route for its own explicit workflow.
SIGNAL_BRANCH_CHOICES = ("standard", "ctxreg")
SIGNAL_BRANCH_TO_ROUTE = {"standard": "pfm", "ctxreg": "pfm_ctxreg"}
ROUTE_TO_SIGNAL_BRANCH = {"pfm": "standard", "pfm_ctxreg": "ctxreg"}


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
            "ctxdrop and other control routes require their own explicit workflow."
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
    if route_key in {"pfm", "pfm_ctxreg"}:
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

from __future__ import annotations

DEFAULT_BRANCH = "kong2022_routea"
BRANCH_CHOICES = ("kong2022_routea", "kong2022_routea_17net", "kong2022_routea_ctxreg", "kong2022_routea_ctxdrop", "kong2022_routeb", "lynch2024_20net", "lynch2024_7net")


# Signal branches are orthogonal to the cortical atlas/taxonomy.  The
# visualization release maintains only these two 2mm/36P choices; core control
# routes retain the separate ctxdrop route for its own explicit workflow.
SIGNAL_BRANCH_CHOICES = ("standard", "ctxreg")
SIGNAL_BRANCH_TO_ROUTE = {"standard": "routea", "ctxreg": "routea_ctxreg"}
ROUTE_TO_SIGNAL_BRANCH = {"routea": "standard", "routea_ctxreg": "ctxreg"}


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

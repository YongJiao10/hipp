from __future__ import annotations

from pathlib import Path

from .output_layout import (
    DEFAULT_DATASET,
    resolve_routea_group_atlas_root,
    resolve_routea_17net_group_atlas_root,
    resolve_routea_ctxreg_group_atlas_root,
    resolve_routea_ctxdrop_group_atlas_root,
    resolve_routeb_group_atlas_root,
    resolve_lynch2024_20net_group_atlas_root,
    resolve_lynch2024_8net_group_atlas_root,
)

ROUTE_TAG_CHOICES = (
    "routea",
    "routea_17net",
    "routea_ctxreg",
    "routea_ctxdrop",
    "routeb",
    "lynch2024_20net",
    "lynch2024_8net",
)
NETWORK_TAXONOMY_CHOICES = ("canonical8", "family17", "lynch20", "lynch8")

_ROUTE_DETECTION_ORDER = (
    "lynch2024_20net",
    "lynch2024_8net",
    "routea_ctxreg",
    "routea_ctxdrop",
    "routea_17net",
    "routeb",
    "routea",
)

ROUTE_TO_GROUP_ATLAS_ROOT = {
    "routea": resolve_routea_group_atlas_root,
    "routea_17net": resolve_routea_17net_group_atlas_root,
    "routea_ctxreg": resolve_routea_ctxreg_group_atlas_root,
    "routea_ctxdrop": resolve_routea_ctxdrop_group_atlas_root,
    "routeb": resolve_routeb_group_atlas_root,
    "lynch2024_20net": resolve_lynch2024_20net_group_atlas_root,
    "lynch2024_8net": resolve_lynch2024_8net_group_atlas_root,
}


def _normalize_route_tag(route_tag: str) -> str:
    route = str(route_tag).strip().lower()
    if route not in ROUTE_TAG_CHOICES:
        raise ValueError(f"Unsupported route tag: {route_tag!r}; expected one of {ROUTE_TAG_CHOICES}")
    return route


def _route_from_name(name: str) -> str | None:
    value = str(name).strip().lower()
    if value == "lynch2024_fast_pfm_t1only":
        return "lynch2024_20net"
    for route in _ROUTE_DETECTION_ORDER:
        if value == route or value.startswith(f"{route}_"):
            return route
    return None


def infer_route_tag_from_path(path: Path, *, arg_name: str) -> str:
    candidate = Path(path)
    for part in reversed(candidate.parts):
        route = _route_from_name(part)
        if route is not None:
            return route
    raise ValueError(f"Could not infer --route-tag from {arg_name}={candidate}; pass an explicit --route-tag.")


def infer_route_tag_from_shared_root(shared_root: Path) -> str:
    return infer_route_tag_from_path(Path(shared_root), arg_name="--shared-root")


def infer_route_tag_from_group_root(group_root: Path) -> str:
    return infer_route_tag_from_path(Path(group_root), arg_name="--group-root")


def canonical_group_atlas_root_for_route(route_tag: str, dataset: str = DEFAULT_DATASET) -> Path:
    return ROUTE_TO_GROUP_ATLAS_ROOT[_normalize_route_tag(route_tag)](dataset)


def validate_group_atlas_root_for_route(group_root: Path, route_tag: str) -> None:
    root = Path(group_root).resolve()
    route = _normalize_route_tag(route_tag)
    if root.parent.name != "group_atlas":
        return
    actual = infer_route_tag_from_group_root(root)
    if actual != route:
        raise ValueError(
            f"Provenance discrepancy: requested --route-tag is {route!r}, "
            f"but --group-root/--out-root directory name {root.name!r} resolves to {actual!r}."
        )


def resolve_route_tag_for_group_atlas_run(*, shared_root: Path, explicit_route: str, network_taxonomy: str) -> str:
    explicit = str(explicit_route).strip().lower()
    taxonomy = str(network_taxonomy).strip().lower()
    if explicit == "auto":
        if taxonomy == "family17":
            raise ValueError(
                "--network-taxonomy family17 reuses the RouteA input root, so --route-tag cannot be inferred. "
                "Pass --route-tag routea_17net explicitly."
            )
        route = infer_route_tag_from_shared_root(shared_root)
    else:
        route = _normalize_route_tag(explicit)

    if route == "routea_17net" and taxonomy != "family17":
        raise ValueError("--route-tag routea_17net requires --network-taxonomy family17.")
    if taxonomy == "family17" and route != "routea_17net":
        raise ValueError("--network-taxonomy family17 requires --route-tag routea_17net.")
    if route == "lynch2024_20net" and taxonomy != "lynch20":
        raise ValueError("--route-tag lynch2024_20net requires --network-taxonomy lynch20.")
    if route == "lynch2024_8net" and taxonomy != "lynch8":
        raise ValueError("--route-tag lynch2024_8net requires --network-taxonomy lynch8.")
    if taxonomy == "lynch20" and route != "lynch2024_20net":
        raise ValueError("--network-taxonomy lynch20 requires --route-tag lynch2024_20net.")
    if taxonomy == "lynch8" and route != "lynch2024_8net":
        raise ValueError("--network-taxonomy lynch8 requires --route-tag lynch2024_8net.")
    return route


def resolve_group_atlas_out_root(out_root: Path | str, route_tag: str, *, default_routeb_out_root_selected: bool, dataset: str = DEFAULT_DATASET) -> tuple[Path, bool]:
    route = _normalize_route_tag(route_tag)
    if default_routeb_out_root_selected and route != "routeb":
        return canonical_group_atlas_root_for_route(route, dataset=dataset).resolve(), True
    return Path(out_root).resolve(), False

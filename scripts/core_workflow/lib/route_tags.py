from __future__ import annotations

from pathlib import Path

from .output_layout import (
    DEFAULT_DATASET,
    resolve_fixed_17net_group_atlas_root,
    resolve_fixed_group_atlas_root,
    resolve_lynch2024_20net_group_atlas_root,
    resolve_lynch2024_7net_group_atlas_root,
    resolve_pfm_17net_group_atlas_root,
    resolve_pfm_ctxdrop_group_atlas_root,
    resolve_pfm_ctxreg_group_atlas_root,
    resolve_pfm_group_atlas_root,
)

ROUTE_TAG_CHOICES = (
    "pfm",
    "pfm_17net",
    "pfm_ctxreg",
    "pfm_ctxdrop",
    "fixed",
    "fixed_17net",
    "lynch2024_20net",
    "lynch2024_7net",
)
NETWORK_TAXONOMY_CHOICES = ("canonical7", "family17", "lynch20", "lynch7")
_CANONICAL7_ROUTE_TAGS = frozenset({"pfm", "pfm_ctxreg", "pfm_ctxdrop", "fixed"})
_FAMILY17_ROUTE_TAGS = frozenset({"pfm_17net", "fixed_17net"})


_ROUTE_DETECTION_ORDER = (
    "lynch2024_20net",
    "lynch2024_7net",
    "pfm_ctxreg",
    "pfm_ctxdrop",
    "pfm_17net",
    "fixed_17net",
    "fixed",
    "pfm",
)

ROUTE_TO_GROUP_ATLAS_ROOT = {
    "pfm": resolve_pfm_group_atlas_root,
    "pfm_17net": resolve_pfm_17net_group_atlas_root,
    "pfm_ctxreg": resolve_pfm_ctxreg_group_atlas_root,
    "pfm_ctxdrop": resolve_pfm_ctxdrop_group_atlas_root,
    "fixed": resolve_fixed_group_atlas_root,
    "fixed_17net": resolve_fixed_17net_group_atlas_root,
    "lynch2024_20net": resolve_lynch2024_20net_group_atlas_root,
    "lynch2024_7net": resolve_lynch2024_7net_group_atlas_root,
}


def _normalize_route_tag(route_tag: str) -> str:
    route = str(route_tag).strip().lower()
    if route not in ROUTE_TAG_CHOICES:
        raise ValueError(f"Unsupported route tag: {route_tag!r}; expected one of {ROUTE_TAG_CHOICES}")
    return route


def _route_from_name(name: str) -> str | None:
    value = str(name).strip().lower()
    if value == "lynch2024":
        return "lynch2024_20net"
    if value in {"pfm_7net", "routea_7net"} or value.startswith(("pfm_7net_", "routea_7net_")):
        raise ValueError("The suffixed *_7net namespace is not an active route; use unsuffixed pfm.")
    for route in _ROUTE_DETECTION_ORDER:
        if value == route or value.startswith(f"{route}_"):
            return route
    return None


def infer_route_tag_from_path(path: Path, *, arg_name: str) -> str:
    candidate = Path(path)
    lowered_parts = [str(part).strip().lower() for part in candidate.parts]
    if "lynch2024" in lowered_parts:
        for part in reversed(lowered_parts):
            if part == "lynch2024_7net" or part.startswith("lynch2024_7net_"):
                return "lynch2024_7net"
            if part == "lynch2024_20net" or part.startswith("lynch2024_20net_"):
                return "lynch2024_20net"
        return "lynch2024_20net"
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
    if "group_atlas" not in root.parts:
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
                "--network-taxonomy family17 reuses the PFM/Fixed input root, so --route-tag cannot be inferred. "
                "Pass --route-tag pfm_17net or fixed_17net explicitly."
            )
        route = infer_route_tag_from_shared_root(shared_root)
    else:
        route = _normalize_route_tag(explicit)

    if route in _FAMILY17_ROUTE_TAGS and taxonomy != "family17":
        raise ValueError(f"--route-tag {route} requires --network-taxonomy family17.")
    if taxonomy == "family17" and route not in _FAMILY17_ROUTE_TAGS:
        raise ValueError("--network-taxonomy family17 requires --route-tag pfm_17net or fixed_17net.")
    if route in _CANONICAL7_ROUTE_TAGS and taxonomy != "canonical7":
        raise ValueError(f"--route-tag {route} requires --network-taxonomy canonical7.")
    if taxonomy == "canonical7" and route not in _CANONICAL7_ROUTE_TAGS:
        raise ValueError("--network-taxonomy canonical7 requires a Kong2022 canonical7 route (pfm, pfm_ctxreg, pfm_ctxdrop, or fixed).")
    if route == "lynch2024_20net" and taxonomy != "lynch20":
        raise ValueError("--route-tag lynch2024_20net requires --network-taxonomy lynch20.")
    if route == "lynch2024_7net" and taxonomy != "lynch7":
        raise ValueError("--route-tag lynch2024_7net requires --network-taxonomy lynch7.")
    if taxonomy == "lynch20" and route != "lynch2024_20net":
        raise ValueError("--network-taxonomy lynch20 requires --route-tag lynch2024_20net.")
    if taxonomy == "lynch7" and route != "lynch2024_7net":
        raise ValueError("--network-taxonomy lynch7 requires --route-tag lynch2024_7net.")
    return route


def resolve_group_atlas_out_root(out_root: Path | str, route_tag: str, *, default_routea_out_root_selected: bool, dataset: str = DEFAULT_DATASET) -> tuple[Path, bool]:
    route = _normalize_route_tag(route_tag)
    if default_routea_out_root_selected and route != "pfm":
        return canonical_group_atlas_root_for_route(route, dataset=dataset).resolve(), True
    return Path(out_root).resolve(), False

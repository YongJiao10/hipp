"""Shared contracts for Kong2022 hippocampal cortex projection profiles.

The profile is the only valid input for gradient Anterior/Middle/Posterior
cortical projections.  Network-level profiles and raw, pre-relabel cluster
rows are intentionally outside this contract.

Normalized WTA projections have a separate four-row contract.  Their rows are
named pseudo-clusters, so a valid cache must register the exact network order
and the exact figure variant that produced it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping


KONG2022_ROI_PROFILE_ATLASES = frozenset({"kong2022"})
KONG2022_CLUSTER_PROFILE_SUBREGIONS = ("Anterior", "Middle", "Posterior")
KONG2022_CLUSTER_ROI_PROFILE_SHAPE = (3, 400)
KONG2022_CLUSTER_LABEL_ORDERING = "long_axis_head_to_tail"
KONG2022_CLUSTER_ORDER_HEAD_TO_TAIL = (1, 2, 3)
GRADIENT_PROFILE_CONTRACT_TAG = "roi_ap"

KONG2022_CANONICAL7_NETWORK_ORDER = (
    "Default",
    "Visual",
    "Somatomotor",
    "DorsalAttention",
    "VentralAttention",
    "Control",
    "Language",
)
KONG2022_WTA_TOP4_NETWORK_ORDER = (
    "Default",
    "VentralAttention",
    "Somatomotor",
    "Language",
)
KONG2022_WTA_PROFILE_SHAPE = (4, 400)
# Stable canonical contract name.  Corrected profiles overwrite the canonical
# cache and sidecar in place; the contract is not versioned in the path.
WTA_PROFILE_CONTRACT_TAG = "wta_roi"


def atlas_produces_cluster_roi_profile(atlas: str) -> bool:
    """Return whether an atlas branch must publish the canonical ROI profile."""
    return str(atlas) in KONG2022_ROI_PROFILE_ATLASES


def validate_cluster_roi_profile_manifest(
    manifest: Mapping[str, Any],
    *,
    profile_path: Path,
    profile_shape: tuple[int, ...],
) -> None:
    """Fail closed unless a profile is registered with the canonical ordering."""
    if manifest.get("status") != "ok":
        raise ValueError(
            f"gradient ROI profile manifest status must be 'ok', got {manifest.get('status')!r}"
        )

    declared_profile = manifest.get("cluster_roi_profile")
    if not declared_profile:
        raise ValueError(
            "manifest cluster_roi_profile is required; no alternate profile source is valid"
        )

    if manifest.get("cluster_roi_profile_contract") != GRADIENT_PROFILE_CONTRACT_TAG:
        raise ValueError(
            "cluster_roi_profile_contract must be "
            f"{GRADIENT_PROFILE_CONTRACT_TAG!r}"
        )

    declared_path = Path(str(declared_profile)).resolve()
    expected_path = profile_path.resolve()
    if declared_path != expected_path:
        raise ValueError(
            "manifest cluster_roi_profile does not match the canonical profile path: "
            f"declared={declared_path} expected={expected_path}"
        )

    if manifest.get("cluster_label_ordering") != KONG2022_CLUSTER_LABEL_ORDERING:
        raise ValueError(
            "cluster_label_ordering must be "
            f"{KONG2022_CLUSTER_LABEL_ORDERING!r} for gradient ROI profiles"
        )

    if manifest.get("cluster_order_head_to_tail") != list(KONG2022_CLUSTER_ORDER_HEAD_TO_TAIL):
        raise ValueError(
            "cluster_order_head_to_tail must be "
            f"{list(KONG2022_CLUSTER_ORDER_HEAD_TO_TAIL)!r}"
        )

    if tuple(profile_shape) != KONG2022_CLUSTER_ROI_PROFILE_SHAPE:
        raise ValueError(
            "canonical cluster_roi_profile shape must be "
            f"{KONG2022_CLUSTER_ROI_PROFILE_SHAPE}, got {tuple(profile_shape)}"
        )


def validate_wta_profile_cache_manifest(
    manifest: Mapping[str, Any],
    *,
    profile_path: Path,
    profile_shape: tuple[int, ...],
    expected_cohort_id: str,
    expected_variant: str,
    expected_smoothing: str,
    expected_atlas: str = "kong2022",
    expected_network_order: tuple[str, ...] = KONG2022_WTA_TOP4_NETWORK_ORDER,
    expected_signal_branch: str | None = None,
) -> None:
    """Fail closed unless a bilateral WTA cache is registered unambiguously."""
    if manifest.get("status") != "ok":
        raise ValueError(
            f"WTA profile cache manifest status must be 'ok', got {manifest.get('status')!r}"
        )

    declared_profile = manifest.get("wta_profile")
    if not declared_profile:
        raise ValueError(
            "manifest wta_profile is required; no unregistered WTA cache is valid"
        )
    if Path(str(declared_profile)).resolve() != profile_path.resolve():
        raise ValueError(
            "manifest wta_profile does not match the canonical profile path: "
            f"declared={Path(str(declared_profile)).resolve()} expected={profile_path.resolve()}"
        )

    expected_scalars = {
        "wta_profile_contract": WTA_PROFILE_CONTRACT_TAG,
        "cohort_id": str(expected_cohort_id),
        "variant": str(expected_variant),
        "smoothing": str(expected_smoothing),
        "atlas": str(expected_atlas),
        "method": "wta-top",
    }
    for key, expected in expected_scalars.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"WTA profile contract field {key!r} must be {expected!r}, "
                f"got {manifest.get(key)!r}"
            )
    if expected_signal_branch is not None:
        recorded_branch = manifest.get("signal_branch")
        if expected_signal_branch in ("ctxreg", "ctxreg_10mm") and recorded_branch != expected_signal_branch:
            raise ValueError(
                f"WTA profile signal_branch must be {expected_signal_branch!r}, got {recorded_branch!r}"
            )
        if expected_signal_branch == "standard" and recorded_branch not in (None, "standard"):
            raise ValueError(
                f"WTA profile signal_branch must be 'standard', got {recorded_branch!r}"
            )

    if manifest.get("network_order") != list(expected_network_order):
        raise ValueError(
            "WTA profile network_order must be "
            f"{list(expected_network_order)!r}"
        )

    if tuple(profile_shape) != KONG2022_WTA_PROFILE_SHAPE:
        raise ValueError(
            "canonical WTA profile shape must be "
            f"{KONG2022_WTA_PROFILE_SHAPE}, got {tuple(profile_shape)}"
        )

    declared_profile_sha256 = manifest.get("profile_sha256")
    if not isinstance(declared_profile_sha256, str) or len(declared_profile_sha256) != 64:
        raise ValueError(
            "WTA profile_sha256 is required; the cache content must be registered"
        )
    digest = hashlib.sha256()
    with profile_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if declared_profile_sha256 != digest.hexdigest():
        raise ValueError(
            "WTA profile_sha256 does not match the profile file; rebuild the cache"
        )

    source_runs = manifest.get("source_runs")
    if not isinstance(source_runs, Mapping) or set(source_runs) != {"L", "R"}:
        raise ValueError(
            "WTA profile source_runs must register exactly the L and R source runs"
        )
    required_source_fields = {
        "manifest_path",
        "manifest_sha256",
        "label_path",
        "label_sha256",
        "support_path",
        "support_sha256",
        "label_map_path",
        "label_map_sha256",
    }
    for hemi in ("L", "R"):
        source = source_runs[hemi]
        if not isinstance(source, Mapping) or not required_source_fields <= set(source):
            raise ValueError(
                f"WTA profile source_runs[{hemi!r}] lacks required provenance fields"
            )

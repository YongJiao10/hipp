"""Shared visualization library for multi-dataset figures."""
from __future__ import annotations

from .multidataset_cohorts import (
    EXCLUDED_VISUALIZATIONS,
    GRID_VARIANTS,
    SIGNAL_BRANCHES,
    GridSpec,
    check_grid_readiness,
    get_multidataset_cohorts,
    resolve_cohort_paths,
)
from .visual_style import (
    ABBREV_MAP,
    COLOR_MAP,
    DEFAULT_NET_COLORS,
    DEFAULT_SCENE,
    GLOBAL_NETWORK_STYLE_JSON,
    SUBREGION_NAMES,
    load_network_hex_colors,
)

__all__ = [
    "GridSpec",
    "GRID_VARIANTS",
    "SIGNAL_BRANCHES",
    "EXCLUDED_VISUALIZATIONS",
    "resolve_cohort_paths",
    "get_multidataset_cohorts",
    "check_grid_readiness",
    "COLOR_MAP",
    "DEFAULT_NET_COLORS",
    "ABBREV_MAP",
    "SUBREGION_NAMES",
    "GLOBAL_NETWORK_STYLE_JSON",
    "DEFAULT_SCENE",
    "load_network_hex_colors",
]

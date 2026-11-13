#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402
"""Render the canonical Kong2022 seven-network hippocampus/cortex figure.

The maintained five-row suite contains HCP 7T (151), Iowa HC (67), Iowa
Patient (26), Pure MDD (7), and Comorbid MDD (19). Each cortical panel carries
its mean FC/bar tile and, when a validated contrast receipt is supplied, a
significant-only group-minus-Iowa-HC t map. The latter uses the explicit
uncorrected two-sided p<0.05 rule recorded by the contrast preparation stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from nibabel.gifti.gifti import GiftiImage
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[3]
CORE_DIR = REPO_ROOT / "scripts/core_workflow"
VIS_LIB_DIR = REPO_ROOT / "scripts/visualize/lib"
for p in (CORE_DIR, VIS_LIB_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lib.cluster_roi_profile_contract import (
    KONG2022_CANONICAL7_NETWORK_ORDER,
    KONG2022_CLUSTER_PROFILE_SUBREGIONS,
    KONG2022_WTA_TOP4_NETWORK_ORDER,
    WTA_PROFILE_CONTRACT_TAG,
    validate_cluster_roi_profile_manifest,
    validate_wta_profile_cache_manifest,
)
from lib.cortex_fc_tiles import (
    COLORBAR_TICKS,
    COLORBAR_VMAX,
    DEFAULT_CMAP,
    NETWORK_BAR_COLORS,
    render_cortex_surface,
    render_cortex_top10_bars,
    join_surface_and_bars,
    strength_colors,
)
from lib.cortex_template_profiles import compute_subject_label_roi_profiles
from lib.gradient_assignment_contract import validate_gradient_assignment
from lib.group_atlas_lineage import validate_group_atlas_lineage
from lib.kong2022_label_utils import KONG2022_DLABEL_PATH, load_kong2022_bundle
from lib.output_layout import (
    DEFAULT_CORTEX_ASSETS_ROOT,
    resolve_kong2022_routea_root,
    resolve_kong2022_surface_root,
)
from lib.surface_utils import make_network_label_gifti, render_locked_grid
from multidataset_cohorts import (  # pyright: ignore[reportMissingImports]
    GRID_VARIANTS,
    SIGNAL_BRANCHES,
    GridSpec,
    has_explicit_legacy_hcp_ctxreg_provenance,
)

DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "hippunfold_input"
DEFAULT_SCENE = REPO_ROOT / "config" / "manual_wb_scenes" / "cortex_manual.scene"

# 10 Target Cohorts Specification (Split into 2 columns: 5 rows each)
COHORTS_COL1_SPEC = [
    {
        "id": "hcp7t",
        "dataset_key": "hcp",
        "dataset_name": "HCP 7T",
        "cohort_name": "HCP 7T",
        "cohort_desc": "Healthy Young Adults",
        "n_subjects": 151,
        "group_root_grad": REPO_ROOT
        / "outputs_hcp/work/group_atlas/pfm",
        "group_root_wta": REPO_ROOT
        / "outputs_hcp/work/group_atlas/pfm",
        "overview_grad": REPO_ROOT
        / "outputs_hcp/present/overviews/pfm/kong2022_gradient-spectral_group_overview_g1g2g3_pfm_fisherz.png",
        "overview_wta": REPO_ROOT
        / "outputs_hcp/present/overviews/pfm/kong2022_wta-top_group_overview_pfm_normalized_top20_count.png",
        "bg_color": "#ffffff",
        "badge_bg": "#0b3c5d",
        "badge_fg": "#ffffff",
        "col": 1,
    },
    {
        "id": "iowa_hc",
        "dataset_key": "iowa_jj_tms",
        "dataset_name": "Iowa 3T",
        "cohort_name": "HC",
        "cohort_desc": "Healthy Controls",
        "n_subjects": 67,
        "group_root_grad": REPO_ROOT
        / "outputs_iowa_jj_tms/work/group_atlas/pfm/hc",
        "group_root_wta": REPO_ROOT
        / "outputs_iowa_jj_tms/work/group_atlas/pfm/hc",
        "overview_grad": REPO_ROOT
        / "outputs_iowa_jj_tms/present/overviews/pfm/hc/kong2022_gradient-spectral_group_overview_g1g2g3_pfm_fisherz.png",
        "overview_wta": REPO_ROOT
        / "outputs_iowa_jj_tms/present/overviews/pfm/hc/kong2022_wta-top_group_overview_pfm_normalized_top20_count.png",
        "bg_color": "#f7faff",
        "badge_bg": "#1d5b79",
        "badge_fg": "#ffffff",
        "col": 1,
    },
    {
        "id": "iowa_pat",
        "dataset_key": "iowa_jj_tms",
        "dataset_name": "Iowa 3T",
        "cohort_name": "Patient",
        "cohort_desc": "Depression Patients",
        "n_subjects": 26,
        "group_root_grad": REPO_ROOT
        / "outputs_iowa_jj_tms/work/group_atlas/pfm/patient",
        "group_root_wta": REPO_ROOT
        / "outputs_iowa_jj_tms/work/group_atlas/pfm/patient",
        "overview_grad": REPO_ROOT
        / "outputs_iowa_jj_tms/present/overviews/pfm/patient/kong2022_gradient-spectral_group_overview_g1g2g3_pfm_fisherz.png",
        "overview_wta": REPO_ROOT
        / "outputs_iowa_jj_tms/present/overviews/pfm/patient/kong2022_wta-top_group_overview_pfm_normalized_top20_count.png",
        "bg_color": "#f7faff",
        "badge_bg": "#1d5b79",
        "badge_fg": "#ffffff",
        "col": 1,
    },
    {
        "id": "stan_nthc",
        "dataset_key": "stanford",
        "dataset_name": "Stanford",
        "cohort_name": "NTHC",
        "cohort_desc": "Non-Trauma Controls",
        "n_subjects": 51,
        "group_root_grad": REPO_ROOT
        / "outputs_stanford/work/group_atlas/pfm/nthc",
        "group_root_wta": REPO_ROOT
        / "outputs_stanford/work/group_atlas/pfm/nthc",
        "overview_grad": REPO_ROOT
        / "outputs_stanford/present/overviews/pfm/nthc/kong2022_gradient-spectral_group_overview_g1g2g3_pfm_fisherz.png",
        "overview_wta": REPO_ROOT
        / "outputs_stanford/present/overviews/pfm/nthc/kong2022_wta-top_group_overview_pfm_normalized_top20_count.png",
        "bg_color": "#fffdf8",
        "badge_bg": "#8c4600",
        "badge_fg": "#ffffff",
        "col": 1,
    },
    {
        "id": "stan_tehc",
        "dataset_key": "stanford",
        "dataset_name": "Stanford",
        "cohort_name": "TEHC",
        "cohort_desc": "Trauma-Exposed Controls",
        "n_subjects": 40,
        "group_root_grad": REPO_ROOT
        / "outputs_stanford/work/group_atlas/pfm/tehc",
        "group_root_wta": REPO_ROOT
        / "outputs_stanford/work/group_atlas/pfm/tehc",
        "overview_grad": REPO_ROOT
        / "outputs_stanford/present/overviews/pfm/tehc/kong2022_gradient-spectral_group_overview_g1g2g3_pfm_fisherz.png",
        "overview_wta": REPO_ROOT
        / "outputs_stanford/present/overviews/pfm/tehc/kong2022_wta-top_group_overview_pfm_normalized_top20_count.png",
        "bg_color": "#fffdf8",
        "badge_bg": "#8c4600",
        "badge_fg": "#ffffff",
        "col": 1,
    },
]

COHORTS_COL2_SPEC = [
    {
        "id": "stan_nts",
        "dataset_key": "stanford",
        "dataset_name": "Stanford",
        "cohort_name": "NTS",
        "cohort_desc": "Non-Trauma Symptomatic",
        "n_subjects": 56,
        "group_root_grad": REPO_ROOT
        / "outputs_stanford/work/group_atlas/pfm/nts",
        "group_root_wta": REPO_ROOT
        / "outputs_stanford/work/group_atlas/pfm/nts",
        "overview_grad": REPO_ROOT
        / "outputs_stanford/present/overviews/pfm/nts/kong2022_gradient-spectral_group_overview_g1g2g3_pfm_fisherz.png",
        "overview_wta": REPO_ROOT
        / "outputs_stanford/present/overviews/pfm/nts/kong2022_wta-top_group_overview_pfm_normalized_top20_count.png",
        "bg_color": "#fffdf8",
        "badge_bg": "#8c4600",
        "badge_fg": "#ffffff",
        "col": 2,
    },
    {
        "id": "stan_nts_any_mdd",
        "dataset_key": "stanford",
        "dataset_name": "Stanford",
        "cohort_name": "NTS Any MDD",
        "cohort_desc": "Non-Trauma MDD",
        "n_subjects": 43,
        "group_root_grad": REPO_ROOT
        / "outputs_stanford/work/group_atlas/pfm/nts_any_mdd",
        "group_root_wta": REPO_ROOT
        / "outputs_stanford/work/group_atlas/pfm/nts_any_mdd",
        "overview_grad": REPO_ROOT
        / "outputs_stanford/present/overviews/pfm/nts_any_mdd/kong2022_gradient-spectral_group_overview_g1g2g3_pfm_fisherz.png",
        "overview_wta": REPO_ROOT
        / "outputs_stanford/present/overviews/pfm/nts_any_mdd/kong2022_wta-top_group_overview_pfm_normalized_top20_count.png",
        "bg_color": "#fffdf8",
        "badge_bg": "#8c4600",
        "badge_fg": "#ffffff",
        "col": 2,
    },
    {
        "id": "stan_nts_pure_mdd",
        "dataset_key": "stanford",
        "dataset_name": "Stanford",
        "cohort_name": "NTS Pure MDD",
        "cohort_desc": "Non-Trauma Pure MDD",
        "n_subjects": 11,
        "group_root_grad": REPO_ROOT
        / "outputs_stanford/work/group_atlas/pfm/nts_pure_mdd",
        "group_root_wta": REPO_ROOT
        / "outputs_stanford/work/group_atlas/pfm/nts_pure_mdd",
        "overview_grad": REPO_ROOT
        / "outputs_stanford/present/overviews/pfm/nts_pure_mdd/kong2022_gradient-spectral_group_overview_g1g2g3_pfm_fisherz.png",
        "overview_wta": REPO_ROOT
        / "outputs_stanford/present/overviews/pfm/nts_pure_mdd/kong2022_wta-top_group_overview_pfm_normalized_top20_count.png",
        "bg_color": "#fffdf8",
        "badge_bg": "#8c4600",
        "badge_fg": "#ffffff",
        "col": 2,
    },
    {
        "id": "stan_tis",
        "dataset_key": "stanford",
        "dataset_name": "Stanford",
        "cohort_name": "TIS",
        "cohort_desc": "Trauma Symptomatic",
        "n_subjects": 47,
        "group_root_grad": REPO_ROOT
        / "outputs_stanford/work/group_atlas/pfm/tis",
        "group_root_wta": REPO_ROOT
        / "outputs_stanford/work/group_atlas/pfm/tis",
        "overview_grad": REPO_ROOT
        / "outputs_stanford/present/overviews/pfm/tis/kong2022_gradient-spectral_group_overview_g1g2g3_pfm_fisherz.png",
        "overview_wta": REPO_ROOT
        / "outputs_stanford/present/overviews/pfm/tis/kong2022_wta-top_group_overview_pfm_normalized_top20_count.png",
        "bg_color": "#fffdf8",
        "badge_bg": "#8c4600",
        "badge_fg": "#ffffff",
        "col": 2,
    },
    {
        "id": "stan_tis_ptsd",
        "dataset_key": "stanford",
        "dataset_name": "Stanford",
        "cohort_name": "TIS PTSD",
        "cohort_desc": "Trauma PTSD",
        "n_subjects": 35,
        "group_root_grad": REPO_ROOT
        / "outputs_stanford/work/group_atlas/pfm/tis_ptsd",
        "group_root_wta": REPO_ROOT
        / "outputs_stanford/work/group_atlas/pfm/tis_ptsd",
        "overview_grad": REPO_ROOT
        / "outputs_stanford/present/overviews/pfm/tis_ptsd/kong2022_gradient-spectral_group_overview_g1g2g3_pfm_fisherz.png",
        "overview_wta": REPO_ROOT
        / "outputs_stanford/present/overviews/pfm/tis_ptsd/kong2022_wta-top_group_overview_pfm_normalized_top20_count.png",
        "bg_color": "#fffdf8",
        "badge_bg": "#8c4600",
        "badge_fg": "#ffffff",
        "col": 2,
    },
]

COHORTS_SPEC = COHORTS_COL1_SPEC + COHORTS_COL2_SPEC

GRADIENT_SUBREGIONS = list(KONG2022_CLUSTER_PROFILE_SUBREGIONS)
WTA_TOP_NETWORKS = ("Default", "VentralAttention", "Auditory", "Somatomotor")
WTA_TOP_NETWORKS_7NET = KONG2022_WTA_TOP4_NETWORK_ORDER
WTA_DISPLAY_NAMES = {
    "Default": "DMN",
    "VentralAttention": "VAN",
    "Auditory": "Aud",
    "Somatomotor": "SMN",
}
WTA_DISPLAY_ORDER_7NET = ("VentralAttention", "Somatomotor", "Language", "Default")
WTA_DISPLAY_ORDER_8NET = ("VentralAttention", "Somatomotor", "Auditory", "Default")

TM_SUBREGIONS = ("TM-Anterior", "TM-Middle", "TM-Posterior")
TM_DISPLAY_TITLES = (
    ("TM Anterior (VAN)", "#dc5cff"),
    ("TM Middle (SMN)", "#6396c2"),
    ("TM Posterior (DMN)", "#d85c6b"),
)


def _wta_network_order(is_7net: bool) -> tuple[str, ...]:
    return WTA_TOP_NETWORKS_7NET if is_7net else WTA_TOP_NETWORKS


def _wta_display_order(is_7net: bool) -> tuple[str, ...]:
    return WTA_DISPLAY_ORDER_7NET if is_7net else WTA_DISPLAY_ORDER_8NET


# Linear display range for the lower t maps. Statistical peak is retained
# separately in the contrast receipt so clipped high HCP values remain auditable.
T_DISPLAY_VMAX = 4.0


def resolve_cortex_assets() -> dict[str, Path]:
    subject = "100610"
    anat = DEFAULT_DATA_ROOT / f"sub-{subject}" / "anat"
    assets_root = DEFAULT_CORTEX_ASSETS_ROOT / f"sub-{subject}" / "assets"
    public_root = REPO_ROOT / "external"
    fastans_tpl = public_root / "FASTANS" / "resources" / "templates"
    scene = DEFAULT_SCENE
    dlabel = KONG2022_DLABEL_PATH

    def _first(paths: list[Path]) -> Path | None:
        for p in paths:
            if p.exists():
                return p.resolve()
        return None

    def _required(path: Path | None, name: str) -> Path:
        if path is None:
            raise FileNotFoundError(f"Missing required cortex asset: {name}")
        return path

    left_inflated = _first(
        [
            anat
            / f"sub-{subject}_hemi-L_space-fsLR_den-32k_desc-MSMAll_inflated.surf.gii",
            DEFAULT_CORTEX_ASSETS_ROOT
            / f"sub-{subject}_hemi-L_space-fsLR_den-32k_desc-MSMAll_inflated.surf.gii",
            public_root
            / f"sub-{subject}_hemi-L_space-fsLR_den-32k_desc-MSMAll_inflated.surf.gii",
        ]
    )
    right_inflated = _first(
        [
            anat
            / f"sub-{subject}_hemi-R_space-fsLR_den-32k_desc-MSMAll_inflated.surf.gii",
            DEFAULT_CORTEX_ASSETS_ROOT
            / f"sub-{subject}_hemi-R_space-fsLR_den-32k_desc-MSMAll_inflated.surf.gii",
            public_root
            / f"sub-{subject}_hemi-R_space-fsLR_den-32k_desc-MSMAll_inflated.surf.gii",
        ]
    )
    sulc_dscalar = _first(
        [
            anat / f"sub-{subject}_space-fsLR_den-32k_desc-MSMAll_sulc.dscalar.nii",
            DEFAULT_CORTEX_ASSETS_ROOT
            / f"sub-{subject}_space-fsLR_den-32k_desc-MSMAll_sulc.dscalar.nii",
            public_root
            / f"sub-{subject}_space-fsLR_den-32k_desc-MSMAll_sulc.dscalar.nii",
        ]
    )
    left_sulc = _first(
        [
            assets_root
            / f"sub-{subject}_hemi-L_space-fsLR_den-32k_desc-MSMAll_sulc.func.gii",
            DEFAULT_CORTEX_ASSETS_ROOT
            / f"sub-{subject}_hemi-L_space-fsLR_den-32k_desc-MSMAll_sulc.func.gii",
            fastans_tpl / "CORTEX_LEFT.32k_fs_LR..func.gii",
        ]
    )
    right_sulc = _first(
        [
            assets_root
            / f"sub-{subject}_hemi-R_space-fsLR_den-32k_desc-MSMAll_sulc.func.gii",
            DEFAULT_CORTEX_ASSETS_ROOT
            / f"sub-{subject}_hemi-R_space-fsLR_den-32k_desc-MSMAll_sulc.func.gii",
            fastans_tpl / "CORTEX_RIGHT.32k_fs_LR.func.gii",
        ]
    )
    return {
        "left_inflated": _required(left_inflated, "left_inflated"),
        "right_inflated": _required(right_inflated, "right_inflated"),
        "left_sulc": _required(left_sulc, "left_sulc"),
        "right_sulc": _required(right_sulc, "right_sulc"),
        "sulc_dscalar": _required(sulc_dscalar, "sulc_dscalar"),
        "scene": scene.resolve(),
        "dlabel": dlabel.resolve(),
    }


def render_cortex_tile(
    *,
    panel_tag: str,
    values: np.ndarray,
    vmax: float,
    roi_registry: list[dict[str, Any]],
    vertex_arrays: dict[str, tuple[np.ndarray, np.ndarray]],
    cortex_assets: dict[str, Path],
    panels_dir: Path,
) -> Path:
    panel_dir = panels_dir / panel_tag
    panel_dir.mkdir(parents=True, exist_ok=True)

    vmax_tag = f"vmax{int(round(vmax * 100))}_pseudocluster"
    out_composite = panel_dir / f"{panel_tag}_{vmax_tag}_composite_v3.png"
    out_png = panel_dir / f"{panel_tag}_{vmax_tag}.png"
    bar_png = panel_dir / f"{panel_tag}_top10_bars_v3.png"
    cache_manifest = out_composite.with_suffix(".cache.json")
    cache_inputs = _cortex_tile_cache_inputs(
        panel_tag=panel_tag,
        values=values,
        vmax=vmax,
        roi_registry=roi_registry,
        vertex_arrays=vertex_arrays,
        cortex_assets=cortex_assets,
    )
    if _is_valid_workbench_cache(
        cache_manifest,
        expected_inputs=cache_inputs,
        output_paths={"surface": out_png, "bars": bar_png, "composite": out_composite},
        expected_dimensions={
            "surface": (1000, 700),
            "bars": (1000, 700),
            "composite": (1000, 1408),
        },
    ):
        return out_composite

    render_cortex_surface(
        panel_tag=panel_tag,
        colors=strength_colors(values, vmax=vmax),
        roi_registry=roi_registry,
        vertex_arrays=vertex_arrays,
        cortex_assets=cortex_assets,
        panel_dir=panel_dir,
        out_png=out_png,
    )

    # Render top 10 bar chart and create composite image
    bar_png = render_cortex_top10_bars(
        panel_tag=panel_tag,
        values=values,
        roi_registry=roi_registry,
        panels_dir=panels_dir,
    )
    join_surface_and_bars(out_png, bar_png, out_composite)
    cache_manifest.write_text(
        json.dumps(
            _workbench_panel_cache_manifest(
                cache_inputs,
                {"surface": out_png, "bars": bar_png, "composite": out_composite},
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return out_composite


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing cache source file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


WORKBENCH_PANEL_CACHE_SCHEMA = "workbench_panel_cache_v2"
_FILE_FINGERPRINT_CACHE: dict[tuple[str, int, int], dict[str, Any]] = {}


def _canonical_json_bytes(value: Any) -> bytes:
    def _default(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"Unsupported cache provenance value: {type(obj)!r}")

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=_default,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _array_fingerprint(value: np.ndarray) -> dict[str, Any]:
    arr = np.ascontiguousarray(np.asarray(value))
    return {
        "dtype": arr.dtype.str,
        "shape": [int(dim) for dim in arr.shape],
        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
    }


def _file_fingerprint(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    key = (str(resolved), int(stat.st_size), int(stat.st_mtime_ns))
    cached = _FILE_FINGERPRINT_CACHE.get(key)
    if cached is not None:
        return dict(cached)

    record = {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256_file(resolved),
    }
    _FILE_FINGERPRINT_CACHE[key] = record
    return dict(record)


def _workbench_panel_cache_manifest(
    expected_inputs: dict[str, Any],
    output_paths: dict[str, Path],
) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    for key, raw_path in output_paths.items():
        path = Path(raw_path).resolve()
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(
                f"Cannot publish cache manifest for missing output: {path}"
            )
        with Image.open(path) as image:
            image.verify()
            dimensions = [int(image.width), int(image.height)]
        outputs[key] = {
            "path": str(path),
            "size_bytes": int(path.stat().st_size),
            "sha256": _sha256_file(path),
            "dimensions": dimensions,
        }
    return {
        "cache_schema": WORKBENCH_PANEL_CACHE_SCHEMA,
        "inputs": expected_inputs,
        "outputs": outputs,
    }


def _is_valid_workbench_cache(
    manifest_path: Path,
    *,
    expected_inputs: dict[str, Any],
    output_paths: dict[str, Path],
    expected_dimensions: dict[str, tuple[int, int] | None] | None = None,
) -> bool:
    """Return true only for a complete cache with exact provenance and outputs."""
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if manifest.get("cache_schema") != WORKBENCH_PANEL_CACHE_SCHEMA:
            return False
        if manifest.get("inputs") != expected_inputs:
            return False

        output_records = manifest.get("outputs")
        if not isinstance(output_records, dict) or set(output_records) != set(
            output_paths
        ):
            return False
        for key, raw_path in output_paths.items():
            path = Path(raw_path).resolve()
            record = output_records.get(key)
            if not isinstance(record, dict):
                return False
            if record.get("path") != str(path):
                return False
            if not path.is_file() or path.stat().st_size <= 0:
                return False
            if int(record.get("size_bytes", -1)) != int(path.stat().st_size):
                return False
            if record.get("sha256") != _sha256_file(path):
                return False
            with Image.open(path) as image:
                image.verify()
                dimensions = (int(image.width), int(image.height))
            recorded_dimensions = tuple(int(x) for x in record.get("dimensions", []))
            if recorded_dimensions != dimensions:
                return False
            required_dimensions = (expected_dimensions or {}).get(key)
            if required_dimensions is not None and dimensions != tuple(
                required_dimensions
            ):
                return False
        return True
    except Exception:
        # A malformed or incomplete cache is a miss; canonical source inputs
        # are still required by the caller and are never replaced implicitly.
        return False


def _cortex_tile_cache_inputs(
    *,
    panel_tag: str,
    values: np.ndarray,
    vmax: float,
    roi_registry: list[dict[str, Any]],
    vertex_arrays: dict[str, tuple[np.ndarray, np.ndarray]],
    cortex_assets: dict[str, Path],
) -> dict[str, Any]:
    return {
        "artifact": "cortex_tile",
        "panel_tag": panel_tag,
        "vmax": float(vmax),
        "values": _array_fingerprint(values),
        "roi_registry_sha256": _sha256_json(roi_registry),
        "vertex_arrays": {
            hemi: {
                "vertex_index": _array_fingerprint(vertex_idx),
                "parcel_ids": _array_fingerprint(parcel_ids),
            }
            for hemi, (vertex_idx, parcel_ids) in sorted(vertex_arrays.items())
        },
        "cortex_assets": {
            key: _file_fingerprint(Path(path))
            for key, path in sorted(cortex_assets.items())
        },
        "render": {
            "renderer": "OSMesa",
            "width": 1000,
            "height": 700,
            "bar_schema": "top10_bars_v3",
            "composite_divider_px": 8,
        },
    }


def _hippo_workbench_cache_inputs(
    *,
    artifact: str,
    panel_tag: str,
    cid: str,
    spec: dict[str, Any],
    variant: str,
    smoothing: str,
    is_7net: bool,
    source_files: dict[str, Path | tuple[Path, ...]],
) -> dict[str, Any]:
    def fingerprint_source(value: Path | tuple[Path, ...]):
        if isinstance(value, tuple):
            return [_file_fingerprint(path) for path in value]
        return _file_fingerprint(value)

    return {
        "artifact": artifact,
        "panel_tag": panel_tag,
        "cohort_id": cid,
        "method": spec.get("method"),
        "dataset_name": spec["dataset_name"],
        "cohort_name": spec["cohort_name"],
        "n_subjects": int(spec["n_subjects"]),
        "variant": variant,
        "smoothing": smoothing,
        "is_7net": bool(is_7net),
        "signal_branch": str(
            spec.get(
                "signal_branch",
                getattr(spec.get("grid_spec"), "signal_branch", "standard"),
            )
        ),
        "source_files": {
            key: fingerprint_source(path) for key, path in sorted(source_files.items())
        },
        "render": {
            "subject": "100610",
            "scene": "wb_locked_native_view_lateral_medial",
        },
    }


def _wta_source_run(
    wta_dir: Path,
    hemi: str,
    *,
    smoothing: str,
    is_7net: bool,
    signal_branch: str = "standard",
) -> dict[str, Any]:
    """Load one exact WTA run and its label identity for profile construction."""
    hemi_dir = wta_dir / f"hemi_{hemi}"
    manifest_path = hemi_dir / "group_atlas_manifest.json"
    support_path = hemi_dir / "group_support_mask.npy"
    labels_path = hemi_dir / "group_functional_atlas_labels.npy"
    label_map_path = hemi_dir / "group_functional_atlas_label_map.json"
    required_paths = (manifest_path, support_path, labels_path, label_map_path)
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing exact WTA source artifact: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded_branch = manifest.get("signal_branch")
    if (
        signal_branch == "ctxreg"
        and recorded_branch != "ctxreg"
        and not (
            recorded_branch is None
            and has_explicit_legacy_hcp_ctxreg_provenance(
                manifest_path, manifest, method="kong2022_7net"
            )
        )
    ):
        raise ValueError(
            f"WTA source signal_branch mismatch: expected ctxreg, got {recorded_branch!r}: {manifest_path}"
        )
    if signal_branch == "standard" and recorded_branch not in (None, "standard"):
        raise ValueError(
            f"WTA source signal_branch mismatch: expected standard, got {recorded_branch!r}: {manifest_path}"
        )
    expected_atlas = "kong2022"
    if manifest.get("status") != "ok":
        raise ValueError(f"WTA source manifest is not ok: {manifest_path}")
    for key, expected in (
        ("method", "wta-top"),
        ("atlas", expected_atlas),
        ("smoothing", smoothing),
        ("assignment_mode", "vertex_level"),
        ("network_taxonomy", "canonical7" if is_7net else "canonical8"),
    ):
        if manifest.get(key) != expected:
            raise ValueError(
                f"WTA source manifest field {key!r} must be {expected!r}: "
                f"{manifest_path}"
            )

    expected_networks = (
        list(KONG2022_CANONICAL7_NETWORK_ORDER)
        if is_7net
        else list(manifest.get("network_order") or [])
    )
    if not expected_networks:
        raise ValueError(f"WTA source manifest has no network_order: {manifest_path}")
    if manifest.get("network_order") != expected_networks:
        raise ValueError(f"WTA source network_order is not canonical: {manifest_path}")
    if (
        manifest.get("default_assignment_variant") != "normalized_top20_count"
        or manifest.get("confidence_metric") != "normalized_top20_count"
        or manifest.get("assignment_metric")
        != "top20_count_normalized_by_network_roi_count"
    ):
        raise ValueError(
            f"WTA source manifest must use normalized_top20_count: {manifest_path}"
        )

    for manifest_key, expected_path in (
        ("group_support_mask", support_path),
        ("group_functional_atlas_labels", labels_path),
        ("group_functional_atlas_label_map", label_map_path),
    ):
        declared = manifest.get(manifest_key)
        if not declared or Path(str(declared)).resolve() != expected_path.resolve():
            raise ValueError(
                f"WTA source manifest {manifest_key!r} does not point to the exact "
                f"artifact under {hemi_dir}"
            )

    label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
    if label_map.get("network_names") != expected_networks:
        raise ValueError(
            f"WTA label-map network_names are not canonical: {label_map_path}"
        )
    network_label_keys = label_map.get("network_label_keys")
    if not isinstance(network_label_keys, dict):
        raise ValueError(f"WTA label-map lacks network_label_keys: {label_map_path}")
    required_networks = KONG2022_WTA_TOP4_NETWORK_ORDER if is_7net else WTA_TOP_NETWORKS
    required_ids: list[int] = []
    for network in required_networks:
        if network not in network_label_keys:
            raise KeyError(
                f"WTA label map is missing required network {network!r}: {label_map_path}"
            )
        required_ids.append(int(network_label_keys[network]))
    if len(set(required_ids)) != len(required_ids):
        raise ValueError(f"WTA top-network label IDs are not unique: {label_map_path}")

    labels = np.asarray(np.load(labels_path)).flatten().astype(np.int32)
    support = np.asarray(np.load(support_path)).flatten().astype(bool)
    if labels.size != support.size:
        raise ValueError(
            f"WTA labels/support length mismatch for {hemi_dir}: "
            f"labels={labels.size} support={support.size}"
        )

    snapshot = {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256_file(manifest_path),
        "label_path": str(labels_path.resolve()),
        "label_sha256": _sha256_file(labels_path),
        "support_path": str(support_path.resolve()),
        "support_sha256": _sha256_file(support_path),
        "label_map_path": str(label_map_path.resolve()),
        "label_map_sha256": _sha256_file(label_map_path),
    }
    return {
        "manifest": manifest,
        "labels": labels,
        "support": support,
        "network_label_keys": {str(k): int(v) for k, v in network_label_keys.items()},
        "snapshot": snapshot,
    }


def _load_wta_profile_cache(
    profile_path: Path,
    manifest_path: Path,
    *,
    expected_cohort_id: str,
    expected_variant: str,
    expected_smoothing: str,
    source_runs: dict[str, dict[str, Any]],
    expected_atlas: str,
    expected_network_order: tuple[str, ...],
    expected_signal_branch: str = "standard",
) -> np.ndarray:
    """Load a WTA cache only after validating its exact bilateral provenance."""
    if not profile_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "WTA profile cache and provenance manifest must both exist; "
            f"profile={profile_path} manifest={manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded_branch = manifest.get("signal_branch")
    if expected_signal_branch == "ctxreg" and recorded_branch != "ctxreg":
        raise ValueError(
            f"WTA profile cache signal_branch mismatch: expected ctxreg, got {recorded_branch!r}: {manifest_path}"
        )
    if expected_signal_branch == "standard" and recorded_branch not in (
        None,
        "standard",
    ):
        raise ValueError(
            f"WTA profile cache signal_branch mismatch: expected standard, got {recorded_branch!r}: {manifest_path}"
        )
    profile = np.asarray(np.load(profile_path), dtype=np.float64)
    validate_wta_profile_cache_manifest(
        manifest,
        profile_path=profile_path,
        profile_shape=tuple(int(dim) for dim in profile.shape),
        expected_cohort_id=expected_cohort_id,
        expected_variant=expected_variant,
        expected_smoothing=expected_smoothing,
        expected_atlas=expected_atlas,
        expected_network_order=expected_network_order,
        expected_signal_branch=expected_signal_branch,
    )
    registered_runs = manifest["source_runs"]
    for hemi in ("L", "R"):
        expected = source_runs[hemi]["snapshot"]
        registered = registered_runs[hemi]
        for key, value in expected.items():
            if registered.get(key) != value:
                raise ValueError(
                    f"WTA profile cache source fingerprint mismatch for hemi={hemi}, "
                    f"field={key!r}; rebuild the cache from the exact WTA run"
                )
    return profile


def _wta_profile_cache_manifest(
    *,
    profile_path: Path,
    cohort_id: str,
    variant: str,
    smoothing: str,
    source_runs: dict[str, dict[str, Any]],
    atlas: str,
    network_order: tuple[str, ...],
    signal_branch: str = "standard",
) -> dict[str, Any]:
    return {
        "status": "ok",
        "wta_profile_contract": WTA_PROFILE_CONTRACT_TAG,
        "wta_profile": str(profile_path.resolve()),
        "cohort_id": str(cohort_id),
        "variant": str(variant),
        "smoothing": str(smoothing),
        "atlas": str(atlas),
        "signal_branch": str(signal_branch),
        "method": "wta-top",
        "profile_sha256": _sha256_file(profile_path),
        "network_order": list(network_order),
        "source_runs": {
            hemi: dict(source_runs[hemi]["snapshot"]) for hemi in ("L", "R")
        },
    }


def _build_wta_profile(
    *,
    cohort_spec: dict[str, Any],
    panels_dir: Path,
    grad_sub: Path,
    wta_dir: Path,
    bundle: dict[str, Any],
    smoothing: str,
    variant: str,
    is_7net: bool,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build and overwrite one named, provenance-locked bilateral WTA profile.

    The profile rows are assigned by network name, never by the order of rows
    returned by an intermediate helper.  The canonical profile and sidecar are
    overwritten after exact source validation; no prior cache is reused.
    """
    cid = str(cohort_spec["id"])
    method_prefix = f"{cohort_spec['method']}_" if cohort_spec.get("method") else ""
    atlas_tag = "7net_" if cohort_spec.get("legacy_7net_namespace", False) else ""
    expected_atlas = "kong2022"
    network_order = tuple(_wta_network_order(is_7net))
    signal_branch = str(
        cohort_spec.get(
            "signal_branch",
            getattr(cohort_spec.get("grid_spec"), "signal_branch", "standard"),
        )
    )

    source_runs = {
        hemi: _wta_source_run(
            wta_dir,
            hemi,
            smoothing=smoothing,
            is_7net=is_7net,
            signal_branch=signal_branch,
        )
        for hemi in ("L", "R")
    }

    gradient_manifest_path = grad_sub / "hemi_L" / "group_atlas_manifest.json"
    if not gradient_manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing gradient cohort manifest needed for WTA subject alignment: "
            f"{gradient_manifest_path}"
        )
    gradient_manifest = json.loads(gradient_manifest_path.read_text(encoding="utf-8"))
    subjects = [
        str(s).removeprefix("sub-") for s in gradient_manifest.get("subjects", [])
    ]
    if not subjects:
        raise ValueError(
            f"Gradient cohort manifest has no subjects for WTA profile construction: "
            f"{gradient_manifest_path}"
        )
    for hemi, source in source_runs.items():
        source_subjects = [
            str(s).removeprefix("sub-") for s in source["manifest"].get("subjects", [])
        ]
        if source_subjects != subjects:
            raise ValueError(
                "Gradient and WTA source subject order differ; refusing to load or build a "
                f"potentially misaligned WTA profile for hemi={hemi}: "
                f"gradient={gradient_manifest_path} wta={source['snapshot']['manifest_path']}"
            )

    prof_cache = panels_dir / (
        f"{method_prefix}{cid}_{variant}_{atlas_tag}"
        f"wta_pseudocluster_bilat_{smoothing}_{signal_branch}.npy"
    )
    prof_cache_manifest = prof_cache.with_suffix(".manifest.json")

    shared_root, reference_root, mask_root = resolve_cohort_inputs_roots(
        cohort_spec, variant
    )
    prof_hemis: dict[str, np.ndarray] = {}
    for hemi in ("L", "R"):
        source = source_runs[hemi]
        pseudo_labels = np.zeros_like(source["labels"], dtype=np.int32)
        for row_id, network in enumerate(network_order, start=1):
            pseudo_labels[source["labels"] == source["network_label_keys"][network]] = (
                row_id
            )

        p3d, _used_subjects = compute_subject_label_roi_profiles(
            subjects=subjects,
            shared_root=shared_root,
            reference_root=reference_root,
            mask_root=mask_root,
            smoothing=smoothing,
            hemi=hemi,
            group_labels=pseudo_labels,
            support_mask=source["support"],
            roi_registry=bundle["roi_registry"],
        )
        # An explicitly named network may have no valid vertices in this
        # cohort.  Preserve its all-NaN row for the fixed network ID without
        # turning the expected empty-row condition into a misleading warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            raw_profile = np.nanmean(p3d, axis=0)
        if (
            raw_profile.ndim != 2
            or raw_profile.shape[1] != 400
            or raw_profile.shape[0] > len(network_order)
        ):
            raise ValueError(
                "WTA pseudo-cluster ROI profile has unexpected shape: "
                f"{raw_profile.shape}, expected <= {(len(network_order), 400)}"
            )

        # Empty trailing networks are represented by missing rows in the
        # helper output.  Keep the row IDs assigned above; never shift a later
        # named network into an earlier row because one network is empty.
        full_profile = np.full((len(network_order), 400), np.nan, dtype=np.float32)
        full_profile[: raw_profile.shape[0], :] = raw_profile.astype(np.float32)
        prof_hemis[hemi] = full_profile

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        profile = np.nanmean([prof_hemis["L"], prof_hemis["R"]], axis=0)
    profile = np.nan_to_num(profile, nan=0.0)
    if profile.shape != (4, 400):
        raise ValueError(
            f"WTA bilateral profile shape is not (4, 400): {profile.shape}"
        )

    np.save(prof_cache, profile.astype(np.float32))
    prof_cache_manifest.write_text(
        json.dumps(
            _wta_profile_cache_manifest(
                profile_path=prof_cache,
                cohort_id=cid,
                variant=variant,
                smoothing=smoothing,
                atlas=expected_atlas,
                network_order=network_order,
                source_runs=source_runs,
                signal_branch=signal_branch,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    profile = _load_wta_profile_cache(
        prof_cache,
        prof_cache_manifest,
        expected_cohort_id=cid,
        expected_variant=variant,
        expected_smoothing=smoothing,
        source_runs=source_runs,
        expected_atlas=expected_atlas,
        expected_network_order=network_order,
        expected_signal_branch=signal_branch,
    )
    return profile, network_order


def _tm_signal_branch(cohort_spec: dict[str, Any]) -> str:
    return str(
        cohort_spec.get(
            "signal_branch",
            getattr(cohort_spec.get("grid_spec"), "signal_branch", "standard"),
        )
    )


def render_hippo_tm_biglegend(
    *,
    cid: str,
    spec: dict[str, Any],
    panels_dir: Path,
    smoothing: str = "2mm",
    variant: str = "36p",
    is_7net: bool = False,
) -> Image.Image:
    """Render the TM dominant atlas in the locked biglegend style.

    Same scene, template surfaces, spec and title scheme as the
    Gradient/WTA hippo cells; only the labels (TM dominant atlas) differ.
    No fallback: any missing input fails closed.
    """
    import tm_group_contrasts as _tm

    if cid == "hcp_7t":
        raise ValueError("HCP 7T has no template-matching dominant atlas")
    signal_branch = _tm_signal_branch(spec)
    if signal_branch not in ("standard", "ctxreg"):
        raise ValueError(f"Unsupported TM signal branch: {signal_branch!r}")
    if (smoothing, variant) != ("2mm", "36p"):
        raise ValueError(f"TM hippo render requires 2mm/36p, got {smoothing}/{variant}")
    atlas_prefix = "7net_" if is_7net else ""
    method_prefix = f"{spec['method']}_" if spec.get("method") else ""
    outdir = (
        panels_dir
        / f"{method_prefix}{cid}_{variant}_{signal_branch}_{atlas_prefix}hippo_tm_biglegend_{smoothing}"
    )
    png_p = (
        outdir
        / f"sub-100610_wb_{cid}_{variant}_{signal_branch}_{atlas_prefix}tm_{smoothing}_biglegend.png"
    )
    scene = REPO_ROOT / "config/wb_locked_native_view_lateral_medial.scene"
    grad_sub = Path(
        spec.get(
            "grad_subpath",
            Path(spec["group_root_grad"])
            / f"gradient-spectral/kong2022/{smoothing}/cov_0p80/g1g2g3",
        )
    )
    ml_path = grad_sub / "hemi_L/group_atlas_manifest.json"
    mr_path = grad_sub / "hemi_R/group_atlas_manifest.json"
    if not (ml_path.is_file() and mr_path.is_file()):
        raise FileNotFoundError(
            "Missing gradient cohort manifests for TM hippo render: "
            f"left={ml_path} right={mr_path}"
        )
    ml = json.loads(ml_path.read_text(encoding="utf-8"))
    mr = json.loads(mr_path.read_text(encoding="utf-8"))
    surf_l = Path(ml["template_left_or_right_surface"])
    surf_r = Path(mr["template_left_or_right_surface"])
    spec_path = Path(ml["template_spec_path"])
    dominant, tm_info = _tm.tm_dominant_labels(cid, signal_branch)
    if len(tm_info["subjects"]) != int(spec["n_subjects"]):
        raise ValueError(
            f"TM roster size differs from cohort N={spec['n_subjects']}: {cid}"
        )
    display = _tm.TM_SUBREGION_DISPLAY
    colors = {display[k]: _tm.TM_GIFTI_RGBA[k] for k in (1, 2, 3)}
    names = [display[k] for k in (1, 2, 3)]
    outdir.mkdir(parents=True, exist_ok=True)
    lbl_l_path = outdir / f"tm_{cid}_2mm_{signal_branch}_hemi-L.label.gii"
    lbl_r_path = outdir / f"tm_{cid}_2mm_{signal_branch}_hemi-R.label.gii"
    nib.save(
        make_network_label_gifti(
            labels=dominant["L"],
            network_names=names,
            network_colors=colors,
        ),
        str(lbl_l_path),
    )
    nib.save(
        make_network_label_gifti(
            labels=dominant["R"],
            network_names=names,
            network_colors=colors,
        ),
        str(lbl_r_path),
    )
    cache_inputs = _hippo_workbench_cache_inputs(
        artifact="hippo_tm_biglegend",
        panel_tag=png_p.stem,
        cid=cid,
        spec=spec,
        variant=variant,
        smoothing=smoothing,
        is_7net=is_7net,
        source_files={
            "scene": scene,
            "left_manifest": ml_path,
            "right_manifest": mr_path,
            "left_surface": surf_l,
            "right_surface": surf_r,
            "left_tm_label": lbl_l_path,
            "right_tm_label": lbl_r_path,
            "template_spec": spec_path,
            "tm_combo_manifests": tuple(tm_info["combo_manifests"]),
            "tm_labels": tuple(tm_info["label_paths"]),
            "tm_scores": tuple(tm_info["score_paths"]),
        },
    )
    cache_manifest = png_p.with_suffix(".cache.json")
    if _is_valid_workbench_cache(
        cache_manifest,
        expected_inputs=cache_inputs,
        output_paths={"biglegend": png_p},
    ):
        with Image.open(png_p) as cached:
            return cached.copy()

    ds = spec["dataset_name"]
    cn = spec["cohort_name"]
    ns = spec["n_subjects"]
    t_str = f"{ds} {cn} (N={ns})"

    res = render_locked_grid(
        subject="100610",
        scene=scene,
        outdir=outdir,
        name=f"{method_prefix}{cid}_{variant}_{signal_branch}_{atlas_prefix}tm_{smoothing}",
        title=t_str,
        left_labels=lbl_l_path,
        right_labels=lbl_r_path,
        left_surface=surf_l,
        right_surface=surf_r,
        spec_path=spec_path,
    )
    rendered = Path(res["biglegend_png"])
    cache_manifest.write_text(
        json.dumps(
            _workbench_panel_cache_manifest(cache_inputs, {"biglegend": rendered}),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with Image.open(rendered) as image:
        return image.copy()


def tm_display_means_from_receipt(tm_receipt: dict[str, Any]) -> dict[str, np.ndarray]:
    """Group display means from the validated TM receipt's subject profiles.

    Reads the exact profile path recorded per cohort so the upper maps
    always reproduce the t-test inputs. Entirely-missing rows stay neutral
    zero only in this upper display, never in t tests.
    """
    import tm_group_contrasts as _tm

    if tm_receipt.get("panels") != list(_tm.TM_PANELS):
        raise ValueError("TM receipt panel order is not TM-Anterior/Middle/Posterior")
    means: dict[str, np.ndarray] = {}
    for grid_entries in tm_receipt.get("grids", {}).values():
        for cid, entry in grid_entries.items():
            if cid in means:
                continue
            prof_path = Path(entry["profile"]["path"])
            if not prof_path.is_file():
                raise FileNotFoundError(f"Missing TM subject profile: {prof_path}")
            with np.load(prof_path) as data:
                if data["panels"].tolist() != list(_tm.TM_PANELS):
                    raise ValueError(f"TM subject panel order mismatch: {prof_path}")
                values = np.asarray(data["values"], dtype=np.float64)
            if values.ndim != 3 or values.shape[1:] != (3, 400):
                raise ValueError(f"TM subject values must be (N,3,400): {prof_path}")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean = np.nanmean(values, axis=0)
            means[cid] = np.nan_to_num(mean, nan=0.0)
    return means


def load_tm_cohort_data(
    cohort_spec: dict[str, Any],
    tm_mean: np.ndarray | None,
    panels_dir: Path,
    cortex_assets: dict[str, Path],
    bundle: dict[str, Any],
    smoothing: str = "2mm",
    variant: str = "36p",
) -> dict[str, Any]:
    """Render the TM hippo atlas image and three TM cortex tiles for one cohort.

    HCP 7T has no template-matching atlas and returns blank placeholders.
    """
    cid = str(cohort_spec["id"])
    signal_branch = _tm_signal_branch(cohort_spec)
    method_prefix = f"{cohort_spec['method']}_" if cohort_spec.get("method") else ""
    atlas_tag = "7net_" if bool(cohort_spec.get("legacy_7net_namespace", False)) else ""
    if cid == "hcp_7t":
        return {
            "spec": cohort_spec,
            "surf_tm": None,
            "tm_cortex_pngs": {},
            "tm_network_order": tuple(TM_SUBREGIONS),
            "has_tm": False,
        }
    if tm_mean is None:
        raise ValueError(f"Missing TM display mean for cohort {cid!r}")
    arr = np.asarray(tm_mean, dtype=np.float64)
    if arr.shape != (3, 400):
        raise ValueError(f"TM display mean must be (3,400) for {cid}: {arr.shape}")
    print(f"Processing TM {cid} ({signal_branch})...")
    surf_tm = render_hippo_tm_biglegend(
        cid=cid,
        spec=cohort_spec,
        panels_dir=panels_dir,
        smoothing=smoothing,
        variant=variant,
        is_7net=bool(cohort_spec.get("legacy_7net_namespace", False)),
    )
    tm_cortex_pngs: dict[str, Path] = {}
    for idx, name in enumerate(TM_SUBREGIONS):
        panel_tag = (
            f"{method_prefix}{cid}_{variant}_{signal_branch}_{atlas_tag}"
            f"tm_{name.lower()}_{smoothing}"
        )
        png_p = render_cortex_tile(
            panel_tag=panel_tag,
            values=arr[idx, :],
            vmax=COLORBAR_VMAX,
            roi_registry=bundle["roi_registry"],
            vertex_arrays=bundle["vertex_arrays"],
            cortex_assets=cortex_assets,
            panels_dir=panels_dir,
        )
        tm_cortex_pngs[name] = png_p
    return {
        "spec": cohort_spec,
        "surf_tm": surf_tm,
        "tm_cortex_pngs": tm_cortex_pngs,
        "tm_network_order": tuple(TM_SUBREGIONS),
        "has_tm": True,
    }


def append_tm_contrast_maps(
    cdata, tm_result, tm_receipt, cortex_assets, bundle, panels_dir, grid
):
    """Stack TM t tiles under TM FC tiles, mirroring gradient/WTA contrasts."""
    from cortex_group_contrasts import T_CMAP

    if tm_result is None:
        cdata["has_tm_contrast"] = False
        return cdata
    if not cdata.get("has_tm", False):
        raise ValueError(
            f"TM contrast supplied for cohort without TM tiles: {cdata['spec']['id']!r}"
        )
    cdata["has_tm_contrast"] = True
    display_vmax = contrast_display_vmax(tm_receipt)
    cid = cdata["spec"]["id"]
    for idx, name in enumerate(TM_SUBREGIONS):
        tiles = cdata["tm_cortex_pngs"]
        fc_path = tiles[name]
        panel_dir = fc_path.parent
        tag = f"{cid}_{grid.tag}_tm_{name.lower()}_vs_iowa_hc"
        t_path = panel_dir / f"{tag}.png"
        composite = panel_dir / f"{tag}_composite.png"
        cache = composite.with_suffix(".cache.json")
        colors = contrast_network_colors(
            tm_result["significant"][idx], bundle["roi_registry"]
        )
        nsig = int(tm_result["significant"][idx].sum())
        nvalid = int(tm_result["testable"][idx].sum())
        caption = f"t vs Iowa HC | {nsig}/400 significant ROIs (uncorrected p < 0.05)"
        if nsig == 0:
            caption = "No significant ROIs (uncorrected p < 0.05)"
        if nvalid < 400:
            caption += f" | {400 - nvalid} not testable"
        inputs = {
            "schema": "fc_bar_t_network_fill_sign_contour_v4_tm",
            "fc": _file_fingerprint(fc_path),
            "contrast": {k: _array_fingerprint(v[idx]) for k, v in tm_result.items()},
            "t_stat_vmax": tm_receipt["t_vmax"],
            "t_display_vmax": display_vmax,
            "cmap": T_CMAP,
            "caption": caption,
            "registry": _sha256_json(bundle["roi_registry"]),
            "assets": {k: _file_fingerprint(v) for k, v in cortex_assets.items()},
            "code": _file_fingerprint(Path(__file__)),
            "vertex_arrays": {
                h: [_array_fingerprint(a) for a in arrays]
                for h, arrays in bundle["vertex_arrays"].items()
            },
        }
        outputs = {"composite": composite, "surface": t_path}
        dimensions: dict[str, tuple[int, int] | None] = {
            "composite": (1000, 2186),
            "surface": (1000, 700),
        }
        if not _is_valid_workbench_cache(
            cache,
            expected_inputs=inputs,
            output_paths=outputs,
            expected_dimensions=dimensions,
        ):
            render_cortex_surface(
                panel_tag=tag,
                colors=colors,
                roi_registry=bundle["roi_registry"],
                vertex_arrays=bundle["vertex_arrays"],
                cortex_assets=cortex_assets,
                panel_dir=panel_dir,
                out_png=t_path,
            )
            pos_mask_roi = (
                tm_result["significant"][idx]
                & np.isfinite(tm_result["t"][idx])
                & (tm_result["t"][idx] > 0)
            )
            neg_mask_roi = (
                tm_result["significant"][idx]
                & np.isfinite(tm_result["t"][idx])
                & (tm_result["t"][idx] < 0)
            )
            apply_significant_cluster_contours(
                t_path,
                pos_mask_roi=pos_mask_roi,
                neg_mask_roi=neg_mask_roi,
                roi_registry=bundle["roi_registry"],
                vertex_arrays=bundle["vertex_arrays"],
                cortex_assets=cortex_assets,
                panel_dir=panel_dir,
                panel_tag=tag,
                iterations=4,
            )
            stack_contrast_panel(fc_path, t_path, composite, caption)
            cache.write_text(
                json.dumps(_workbench_panel_cache_manifest(inputs, outputs), indent=2)
                + "\n"
            )
        tiles[name] = composite
    return cdata


def validate_hippo_panel_source(manifest, label_path, expected_n, taxonomy):
    """Validate actual label-table semantics before either cache reuse or rendering."""
    if manifest.get("network_taxonomy") != taxonomy:
        raise ValueError(f"Hippocampal panel taxonomy must be {taxonomy}: {label_path}")
    subjects = [str(x).removeprefix("sub-") for x in manifest.get("subjects", [])]
    if (
        manifest.get("n_subjects") != expected_n
        or len(subjects) != expected_n
        or len(set(subjects)) != expected_n
    ):
        raise ValueError(
            f"Hippocampal panel sample count differs from N={expected_n}: {label_path}"
        )
    label_image = nib.load(label_path)
    if not isinstance(label_image, GiftiImage):
        raise TypeError(f"Expected GIFTI label image: {label_path}")
    names = {label.label for label in label_image.labeltable.labels}
    allowed = set(KONG2022_CANONICAL7_NETWORK_ORDER) | {"Null", "unassigned"}
    if taxonomy == "canonical8":
        allowed.add("Auditory")
    unexpected = names - allowed
    if unexpected:
        raise ValueError(
            f"Unexpected {taxonomy} hippocampal labels {sorted(unexpected)}: {label_path}"
        )


def render_hippo_grad_biglegend(
    *,
    cid: str,
    spec: dict[str, Any],
    panels_dir: Path,
    smoothing: str = "2mm",
    variant: str = "36p",
    is_7net: bool = False,
) -> Image.Image:
    atlas_prefix = "7net_" if is_7net else ""
    signal_branch = str(
        spec.get(
            "signal_branch", getattr(spec.get("grid_spec"), "signal_branch", "standard")
        )
    )
    atlas_folder = "kong2022"
    method_prefix = f"{spec['method']}_" if spec.get("method") else ""
    outdir = (
        panels_dir
        / f"{method_prefix}{cid}_{variant}_{signal_branch}_{atlas_prefix}hippo_grad_biglegend_{smoothing}"
    )
    png_p = (
        outdir
        / f"sub-100610_wb_{cid}_{variant}_{signal_branch}_{atlas_prefix}grad_{smoothing}_biglegend.png"
    )
    scene = REPO_ROOT / "config/wb_locked_native_view_lateral_medial.scene"
    grad_sub = Path(
        spec.get(
            "grad_subpath",
            Path(spec["group_root_grad"])
            / f"gradient-spectral/{atlas_folder}/{smoothing}/cov_0p80/g1g2g3",
        )
    )
    ml_path = grad_sub / "hemi_L/group_atlas_manifest.json"
    mr_path = grad_sub / "hemi_R/group_atlas_manifest.json"
    if not (ml_path.is_file() and mr_path.is_file()):
        raise FileNotFoundError(
            f"Missing gradient cohort manifests for signal branch {spec.get('signal_branch', 'standard')!r}: "
            f"left={ml_path} right={mr_path}"
        )
    ml = json.loads(ml_path.read_text(encoding="utf-8"))
    mr = json.loads(mr_path.read_text(encoding="utf-8"))
    left_assignment = validate_gradient_assignment(ml_path.parent, ml, "fisherz")
    right_assignment = validate_gradient_assignment(mr_path.parent, mr, "fisherz")

    surf_l = Path(ml["template_left_or_right_surface"])
    surf_r = Path(mr["template_left_or_right_surface"])
    lbl_l = Path(left_assignment["label_path"])
    lbl_r = Path(right_assignment["label_path"])
    spec_path = Path(ml["template_spec_path"])

    taxonomy = (
        "canonical7" if spec.get("grid_spec", GridSpec()).is_7net else "canonical8"
    )
    validate_hippo_panel_source(ml, lbl_l, int(spec["n_subjects"]), taxonomy)
    validate_hippo_panel_source(mr, lbl_r, int(spec["n_subjects"]), taxonomy)

    cache_inputs = _hippo_workbench_cache_inputs(
        artifact="hippo_grad_biglegend",
        panel_tag=png_p.stem,
        cid=cid,
        spec=spec,
        variant=variant,
        smoothing=smoothing,
        is_7net=is_7net,
        source_files={
            "scene": scene,
            "left_manifest": ml_path,
            "right_manifest": mr_path,
            "left_surface": surf_l,
            "right_surface": surf_r,
            "left_label": lbl_l,
            "right_label": lbl_r,
            "left_assignment_assets": tuple(left_assignment["paths"]),
            "right_assignment_assets": tuple(right_assignment["paths"]),
            "template_spec": spec_path,
        },
    )
    cache_manifest = png_p.with_suffix(".cache.json")
    if _is_valid_workbench_cache(
        cache_manifest,
        expected_inputs=cache_inputs,
        output_paths={"biglegend": png_p},
    ):
        with Image.open(png_p) as cached:
            return cached.copy()

    ds = spec["dataset_name"]
    cn = spec["cohort_name"]
    ns = spec["n_subjects"]
    t_str = f"{ds} {cn} (N={ns})"

    res = render_locked_grid(
        subject="100610",
        scene=scene,
        outdir=outdir,
        name=f"{method_prefix}{cid}_{variant}_{signal_branch}_{atlas_prefix}grad_{smoothing}",
        title=t_str,
        left_labels=lbl_l,
        right_labels=lbl_r,
        left_surface=surf_l,
        right_surface=surf_r,
        spec_path=spec_path,
    )
    rendered = Path(res["biglegend_png"])
    cache_manifest.write_text(
        json.dumps(
            _workbench_panel_cache_manifest(cache_inputs, {"biglegend": rendered}),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with Image.open(rendered) as image:
        return image.copy()


def render_hippo_wta_biglegend(
    *,
    cid: str,
    spec: dict[str, Any],
    panels_dir: Path,
    smoothing: str = "2mm",
    variant: str = "36p",
    is_7net: bool = False,
) -> Image.Image:
    atlas_prefix = "7net_" if is_7net else ""
    signal_branch = str(
        spec.get(
            "signal_branch", getattr(spec.get("grid_spec"), "signal_branch", "standard")
        )
    )
    # WTA is always read from the canonical Kong2022 ROI-registry product.
    # ``is_7net`` controls display order and output naming only; it must not
    # select a second WTA atlas tree.
    atlas_folder = "kong2022"
    method_prefix = f"{spec['method']}_" if spec.get("method") else ""
    outdir = (
        panels_dir
        / f"{method_prefix}{cid}_{variant}_{signal_branch}_{atlas_prefix}hippo_wta_biglegend_{smoothing}"
    )
    png_p = (
        outdir
        / f"sub-100610_wb_{cid}_{variant}_{signal_branch}_{atlas_prefix}wta_{smoothing}_biglegend.png"
    )
    scene = REPO_ROOT / "config/wb_locked_native_view_lateral_medial.scene"
    wta_dir = Path(
        spec.get(
            "wta_subpath",
            Path(spec["group_root_wta"])
            / f"wta-top/{atlas_folder}/{smoothing}/cov_0p80",
        )
    )
    ml_path = wta_dir / "hemi_L/group_atlas_manifest.json"
    mr_path = wta_dir / "hemi_R/group_atlas_manifest.json"
    if not (ml_path.is_file() and mr_path.is_file()):
        raise FileNotFoundError(
            f"Missing WTA cohort manifests for signal branch {spec.get('signal_branch', 'standard')!r}: "
            f"left={ml_path} right={mr_path}"
        )
    ml = json.loads(ml_path.read_text(encoding="utf-8"))
    mr = json.loads(mr_path.read_text(encoding="utf-8"))

    surf_l = Path(ml["template_left_or_right_surface"])
    surf_r = Path(mr["template_left_or_right_surface"])
    lbl_l = Path(ml["template_functional_atlas_label"])
    lbl_r = Path(mr["template_functional_atlas_label"])
    spec_path = Path(ml["template_spec_path"])

    taxonomy = (
        "canonical7" if spec.get("grid_spec", GridSpec()).is_7net else "canonical8"
    )
    validate_hippo_panel_source(ml, lbl_l, int(spec["n_subjects"]), taxonomy)
    validate_hippo_panel_source(mr, lbl_r, int(spec["n_subjects"]), taxonomy)

    cache_inputs = _hippo_workbench_cache_inputs(
        artifact="hippo_wta_biglegend",
        panel_tag=png_p.stem,
        cid=cid,
        spec=spec,
        variant=variant,
        smoothing=smoothing,
        is_7net=is_7net,
        source_files={
            "scene": scene,
            "left_manifest": ml_path,
            "right_manifest": mr_path,
            "left_surface": surf_l,
            "right_surface": surf_r,
            "left_label": lbl_l,
            "right_label": lbl_r,
            "template_spec": spec_path,
        },
    )
    cache_manifest = png_p.with_suffix(".cache.json")
    if _is_valid_workbench_cache(
        cache_manifest,
        expected_inputs=cache_inputs,
        output_paths={"biglegend": png_p},
    ):
        with Image.open(png_p) as cached:
            return cached.copy()

    ds = spec["dataset_name"]
    cn = spec["cohort_name"]
    ns = spec["n_subjects"]
    t_str = f"{ds} {cn} (N={ns})"

    res = render_locked_grid(
        subject="100610",
        scene=scene,
        outdir=outdir,
        name=f"{method_prefix}{cid}_{variant}_{signal_branch}_{atlas_prefix}wta_{smoothing}",
        title=t_str,
        left_labels=lbl_l,
        right_labels=lbl_r,
        left_surface=surf_l,
        right_surface=surf_r,
        spec_path=spec_path,
    )
    rendered = Path(res["biglegend_png"])
    cache_manifest.write_text(
        json.dumps(
            _workbench_panel_cache_manifest(cache_inputs, {"biglegend": rendered}),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with Image.open(rendered) as image:
        return image.copy()


def resolve_cohort_inputs_roots(
    cohort_spec: dict[str, Any], variant: str
) -> tuple[Path, Path, Path]:
    """Resolve (surface_root, reference_root, mask_root) from the explicit registry contract."""
    if variant != "36p":
        raise ValueError(f"Unsupported variant: {variant}")
    declared = (
        cohort_spec.get("surface_root"),
        cohort_spec.get("reference_root"),
        cohort_spec.get("mask_root"),
    )
    signal_branch = str(
        cohort_spec.get(
            "signal_branch",
            getattr(cohort_spec.get("grid_spec"), "signal_branch", "standard"),
        )
    )
    if all(value is not None for value in declared):
        return tuple(Path(value) for value in declared)  # type: ignore[return-value]
    if signal_branch == "ctxreg":
        raise ValueError(
            "CtxReg cohort input registry is incomplete; refusing to fall back to standard surface/reference/mask roots: "
            f"cohort={cohort_spec.get('id')!r} declared={declared!r}"
        )
    ds_key = cohort_spec["dataset_key"]
    if ds_key == "hcp":
        surface_root = (
            REPO_ROOT / "outputs_hcp/inputs/kong2022/shared_inputs/surface"
        )
        reference_root = (
            REPO_ROOT / "outputs_hcp/inputs/kong2022/shared_inputs/pfm"
        )
        return surface_root, reference_root, surface_root
    elif ds_key == "iowa_jj_tms":
        var_suffix = ""
        surface_root = (
            REPO_ROOT
            / f"outputs_iowa_jj_tms/inputs/kong2022{var_suffix}/shared_inputs/surface"
        )
        reference_root = (
            REPO_ROOT
            / f"outputs_iowa_jj_tms/inputs/kong2022{var_suffix}/shared_inputs/pfm"
        )
        return surface_root, reference_root, surface_root
    elif ds_key == "stanford":
        surface_root = REPO_ROOT / "outputs_stanford/inputs/kong2022/shared_inputs/surface"
        reference_root = REPO_ROOT / "outputs_stanford/inputs/kong2022/shared_inputs/pfm"
        return surface_root, reference_root, surface_root

    surface_root = resolve_kong2022_surface_root(ds_key, variant)
    reference_root = resolve_kong2022_routea_root(ds_key, variant)
    return surface_root, reference_root, surface_root


def _load_gradient_roi_profile(
    grad_sub: Path,
    hemi: str,
    bundle: dict[str, Any],
    method: str | None = None,
    cohort_spec: dict[str, Any] | None = None,
    variant: str = "36p",
    smoothing: str = "2mm",
    panels_dir: Path | None = None,
    expected_signal_branch: str = "standard",
) -> np.ndarray:
    """Load only the registered, long-axis-ordered 400-ROI profile."""
    del bundle, method, cohort_spec, variant, smoothing, panels_dir

    hemi_dir = grad_sub / f"hemi_{hemi}"
    manifest_path = hemi_dir / "group_atlas_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing group_atlas_manifest.json for gradient ROI profile: {manifest_path}"
        )

    profile_path = hemi_dir / "cluster_roi_profile.npy"
    if not profile_path.is_file():
        raise FileNotFoundError(
            f"Missing canonical cluster_roi_profile.npy for {grad_sub} hemi={hemi}; "
            "the producer must regenerate it; no alternate profile source is valid"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded_branch = manifest.get("signal_branch")
    if (
        expected_signal_branch == "ctxreg"
        and recorded_branch != "ctxreg"
        and not (
            recorded_branch is None
            and has_explicit_legacy_hcp_ctxreg_provenance(
                manifest_path, manifest, method="kong2022_7net"
            )
        )
    ):
        raise ValueError(
            f"Gradient profile signal_branch mismatch: expected ctxreg, got {recorded_branch!r}: {manifest_path}"
        )
    if expected_signal_branch == "standard" and recorded_branch not in (
        None,
        "standard",
    ):
        raise ValueError(
            f"Gradient profile signal_branch mismatch: expected standard, got {recorded_branch!r}: {manifest_path}"
        )
    profile = np.asarray(np.load(profile_path), dtype=np.float64)
    validate_cluster_roi_profile_manifest(
        manifest,
        profile_path=profile_path,
        profile_shape=tuple(int(dim) for dim in profile.shape),
    )
    validate_group_atlas_lineage(manifest_path)
    return profile


def load_cohort_data(
    cohort_spec: dict[str, Any],
    panels_dir: Path,
    cortex_assets: dict[str, Path],
    bundle: dict[str, Any],
    smoothing: str = "2mm",
    variant: str = "36p",
    is_7net: bool = False,
) -> dict[str, Any]:
    cid = cohort_spec["id"]
    ds = cohort_spec["dataset_name"]
    cn = cohort_spec["cohort_name"]
    legacy_7net_namespace = bool(cohort_spec.get("legacy_7net_namespace", False))
    signal_branch = str(
        cohort_spec.get(
            "signal_branch",
            getattr(cohort_spec.get("grid_spec"), "signal_branch", "standard"),
        )
    )
    atlas_tag = "7net_" if legacy_7net_namespace else ""
    gradient_atlas_folder = cohort_spec.get(
        "atlas_folder", "kong2022_7net" if legacy_7net_namespace else "kong2022"
    )
    method_prefix = f"{cohort_spec['method']}_" if cohort_spec.get("method") else ""
    print(
        f"Processing {ds} {cn} ({method_prefix}{atlas_tag}{smoothing}_{variant}_{signal_branch})..."
    )

    # 1. Hippocampus 3D Surface Atlas & Legend (Workbench SSOT rendering)
    surf_grad = render_hippo_grad_biglegend(
        cid=cid,
        spec=cohort_spec,
        panels_dir=panels_dir,
        smoothing=smoothing,
        variant=variant,
        is_7net=legacy_7net_namespace,
    )
    surf_wta = render_hippo_wta_biglegend(
        cid=cid,
        spec=cohort_spec,
        panels_dir=panels_dir,
        smoothing=smoothing,
        variant=variant,
        is_7net=legacy_7net_namespace,
    )

    # 2. Gradient Cortex Projections (K=3)
    grad_sub = Path(
        cohort_spec.get(
            "grad_subpath",
            Path(cohort_spec["group_root_grad"])
            / f"gradient-spectral/{gradient_atlas_folder}/{smoothing}/cov_0p80/g1g2g3",
        )
    )
    if not (grad_sub / "hemi_L" / "group_atlas_manifest.json").is_file():
        raise FileNotFoundError(
            f"Missing gradient atlas manifest for signal branch {signal_branch!r}: "
            f"{grad_sub / 'hemi_L' / 'group_atlas_manifest.json'}"
        )
    else:
        prof_grad_L = _load_gradient_roi_profile(
            grad_sub,
            "L",
            bundle,
            cohort_spec.get("method"),
            cohort_spec=cohort_spec,
            variant=variant,
            smoothing=smoothing,
            panels_dir=panels_dir,
            expected_signal_branch=signal_branch,
        )
        prof_grad_R = _load_gradient_roi_profile(
            grad_sub,
            "R",
            bundle,
            cohort_spec.get("method"),
            cohort_spec=cohort_spec,
            variant=variant,
            smoothing=smoothing,
            panels_dir=panels_dir,
            expected_signal_branch=signal_branch,
        )
        prof_grad_bilat = (prof_grad_L + prof_grad_R) / 2.0

    grad_cortex_pngs = {}
    for c_idx, subreg in enumerate(GRADIENT_SUBREGIONS):
        panel_tag = f"{method_prefix}{cid}_{variant}_{signal_branch}_{atlas_tag}grad_{subreg.lower()}_{smoothing}"
        vals = prof_grad_bilat[c_idx, :]
        png_p = render_cortex_tile(
            panel_tag=panel_tag,
            values=vals,
            vmax=COLORBAR_VMAX,
            roi_registry=bundle["roi_registry"],
            vertex_arrays=bundle["vertex_arrays"],
            cortex_assets=cortex_assets,
            panels_dir=panels_dir,
        )
        grad_cortex_pngs[subreg] = png_p

    # 3. Normalized WTA Cortex Projections (Subject-Level Pseudo-Cluster FC)
    from types import SimpleNamespace

    from cortex_group_contrasts import (
        canonical_wta_mean,  # pyright: ignore[reportMissingImports]
    )

    prof_wta_all, _mean_sources = canonical_wta_mean(
        SimpleNamespace(**globals()),
        cohort_spec,
        GridSpec(
            smoothing,
            variant,
            signal_branch=signal_branch,
            atlas="7net" if is_7net else "canonical8",
        ),
    )
    wta_networks = _wta_network_order(is_7net)
    wta_cortex_pngs = {}

    for net_idx, net in enumerate(wta_networks):
        vals = prof_wta_all[net_idx, :]
        panel_tag = f"{method_prefix}{cid}_{variant}_{signal_branch}_{atlas_tag}wta_{net.lower()}_{smoothing}"
        png_p = render_cortex_tile(
            panel_tag=panel_tag,
            values=vals,
            vmax=COLORBAR_VMAX,
            roi_registry=bundle["roi_registry"],
            vertex_arrays=bundle["vertex_arrays"],
            cortex_assets=cortex_assets,
            panels_dir=panels_dir,
        )
        wta_cortex_pngs[net] = png_p

    return {
        "spec": cohort_spec,
        "surf_grad": surf_grad,
        "surf_wta": surf_wta,
        "grad_cortex_pngs": grad_cortex_pngs,
        "wta_cortex_pngs": wta_cortex_pngs,
        "wta_network_order": tuple(wta_networks),
    }


def contrast_network_colors(
    significant: np.ndarray, roi_registry: list[dict[str, Any]]
) -> np.ndarray:
    significant = np.asarray(significant, dtype=bool)
    colors = np.ones((len(significant), 4), dtype=float)
    for idx, is_sig in enumerate(significant):
        if is_sig:
            net_name = roi_registry[idx]["canonical7"]
            hex_c = NETWORK_BAR_COLORS.get(net_name, "#888888")
            colors[idx] = mcolors.to_rgba(hex_c)
    return colors


def contrast_colors(values, significant, vmax=4.0, roi_registry=None):
    if roi_registry is not None:
        return contrast_network_colors(significant, roi_registry)
    from cortex_group_contrasts import T_CMAP

    values = np.asarray(values, dtype=float)
    significant = np.asarray(significant, dtype=bool)
    if values.shape != significant.shape or not np.isfinite(vmax) or vmax <= 0:
        raise ValueError("Invalid t map mask or color range")
    colors = np.ones(values.shape + (4,), dtype=float)
    show = significant & np.isfinite(values)
    colors[show] = plt.get_cmap(T_CMAP)(Normalize(-vmax, vmax)(values[show]))
    return colors


def contrast_display_vmax(receipt):
    """Return the explicit linear t-map display range from a validated receipt."""
    try:
        stat_vmax = float(receipt["t_vmax"])
        display_vmax = float(receipt["t_display_vmax"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Contrast receipt must declare t_display_vmax separately from t_vmax"
        ) from exc
    if (
        not np.isfinite(stat_vmax)
        or stat_vmax <= 0
        or not np.isfinite(display_vmax)
        or display_vmax <= 0
    ):
        raise ValueError(
            "Contrast receipt t_vmax and t_display_vmax must be finite and positive"
        )
    return display_vmax


def stack_contrast_panel(fc_path, t_path, output, caption):
    with Image.open(fc_path) as source:
        if source.size != (1000, 1408):
            raise ValueError(f"FC composite must be 1000x1408: {fc_path}")
        composite = Image.new("RGB", (1000, 2186), "white")
        composite.paste(source.convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(composite)
    from matplotlib.font_manager import findfont

    font = ImageFont.truetype(findfont("DejaVu Sans"), 26)
    draw.text((500, 1451), caption, font=font, fill="#452765", anchor="mm")
    if t_path is not None:
        with Image.open(t_path) as source:
            if source.size != (1000, 700):
                raise ValueError(f"t cortex surface must be 1000x700: {t_path}")
            composite.paste(source.convert("RGB"), (0, 1486))
    composite.save(output)
    return output


_CANONICAL7_PIXEL_MAP: np.ndarray | None = None


def get_canonical7_pixel_map(
    panels_dir: Path,
    bundle: dict[str, Any],
    cortex_assets: dict[str, Path],
) -> np.ndarray:
    global _CANONICAL7_PIXEL_MAP
    if _CANONICAL7_PIXEL_MAP is not None:
        return _CANONICAL7_PIXEL_MAP

    npy_cache = panels_dir / "canonical7_pixel_map.npy"
    if npy_cache.exists():
        try:
            cached_map = np.asarray(np.load(npy_cache))
            _CANONICAL7_PIXEL_MAP = cached_map
            return cached_map
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"Invalid canonical7 pixel-map cache; refusing stale fallback: {npy_cache}"
            ) from exc

    ref_png = panels_dir / "canonical7_reference_surface.png"
    if not ref_png.exists() or ref_png.stat().st_size < 10000:
        roi_reg = bundle["roi_registry"]
        colors_7net = np.zeros((400, 4), dtype=float)
        for idx, row in enumerate(roi_reg):
            net_name = row["canonical7"]
            hex_c = NETWORK_BAR_COLORS.get(net_name, "#888888")
            colors_7net[idx] = mcolors.to_rgba(hex_c)
        render_cortex_surface(
            panel_tag="canonical7_reference",
            colors=colors_7net,
            roi_registry=roi_reg,
            vertex_arrays=bundle["vertex_arrays"],
            cortex_assets=cortex_assets,
            panel_dir=panels_dir,
            out_png=ref_png,
        )

    net_img = Image.open(ref_png).convert("RGB")
    arr = np.array(net_img, dtype=float)
    nets = list(KONG2022_CANONICAL7_NETWORK_ORDER)
    net_rgbs = np.array([mcolors.to_rgb(NETWORK_BAR_COLORS[n]) for n in nets]) * 255.0
    medial_rgb = np.array([166.0, 166.0, 166.0])
    all_rgbs = np.vstack([net_rgbs, medial_rgb])

    brain_mask = (arr[:, :, 0] > 20) | (arr[:, :, 1] > 20) | (arr[:, :, 2] > 20)
    arr_norm = arr / np.maximum(np.linalg.norm(arr, axis=2, keepdims=True), 1e-5)
    all_norm = all_rgbs / np.linalg.norm(all_rgbs, axis=1, keepdims=True)
    sims = np.dot(arr_norm, all_norm.T)
    best_net = np.argmax(sims, axis=2).astype(np.int8)
    best_net[~brain_mask] = -1

    np.save(npy_cache, best_net)
    _CANONICAL7_PIXEL_MAP = best_net
    return best_net


def apply_significant_cluster_contours(
    t_png: Path,
    canonical7_pixel_map: np.ndarray | None = None,
    *,
    pos_mask_roi: np.ndarray | None = None,
    neg_mask_roi: np.ndarray | None = None,
    roi_registry: list[dict[str, Any]] | None = None,
    vertex_arrays: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    cortex_assets: dict[str, Path] | None = None,
    panel_dir: Path | None = None,
    panel_tag: str = "",
    iterations: int = 4,
) -> None:
    """Overlay bold outlines around significant ROI clusters representing pos/neg t (Red: +, Blue: -)."""
    if pos_mask_roi is None or neg_mask_roi is None:
        return
    has_pos = bool(np.any(pos_mask_roi))
    has_neg = bool(np.any(neg_mask_roi))
    if not has_pos and not has_neg:
        return

    t_img = Image.open(t_png).convert("RGB")
    t_arr = np.array(t_img)
    brain_mask = (t_arr[:, :, 0] > 20) | (t_arr[:, :, 1] > 20) | (t_arr[:, :, 2] > 20)
    struct = ndi.generate_binary_structure(2, 2)
    brain_inner = ndi.binary_erosion(brain_mask, iterations=2)

    diff = np.abs(t_arr[:, :, 0].astype(float) - t_arr[:, :, 1].astype(float)) + np.abs(
        t_arr[:, :, 1].astype(float) - t_arr[:, :, 2].astype(float)
    )
    sig_pixels = brain_mask & (diff > 12)

    if not np.any(sig_pixels):
        return

    if has_pos and not has_neg:
        pos_pixels = sig_pixels
        neg_pixels = np.zeros_like(sig_pixels, dtype=bool)
    elif has_neg and not has_pos:
        neg_pixels = sig_pixels
        pos_pixels = np.zeros_like(sig_pixels, dtype=bool)
    else:
        if (
            panel_dir is None
            or roi_registry is None
            or vertex_arrays is None
            or cortex_assets is None
        ):
            pos_pixels = sig_pixels
            neg_pixels = np.zeros_like(sig_pixels, dtype=bool)
        else:
            sign_tag = f"{panel_tag}_sign"
            sign_png = panel_dir / f"{sign_tag}.png"
            sign_colors = np.ones((len(roi_registry), 4), dtype=float)
            for i in range(len(roi_registry)):
                if pos_mask_roi[i]:
                    sign_colors[i] = [1.0, 0.0, 0.0, 1.0]
                elif neg_mask_roi[i]:
                    sign_colors[i] = [0.0, 0.0, 1.0, 1.0]
            render_cortex_surface(
                panel_tag=sign_tag,
                colors=sign_colors,
                roi_registry=roi_registry,
                vertex_arrays=vertex_arrays,
                cortex_assets=cortex_assets,
                panel_dir=panel_dir,
                out_png=sign_png,
            )
            sign_arr = np.array(Image.open(sign_png).convert("RGB"), dtype=float)
            diff_rb = sign_arr[:, :, 0] - sign_arr[:, :, 2]
            pos_pixels = sig_pixels & (diff_rb > 30)
            neg_pixels = sig_pixels & (diff_rb < -30)

    RED_RGB = np.array([220, 20, 20], dtype=np.uint8)
    BLUE_RGB = np.array([20, 100, 240], dtype=np.uint8)

    overlay = np.zeros((700, 1000, 4), dtype=np.uint8)
    if np.any(pos_pixels):
        labeled_pos, num_pos = cast(
            tuple[np.ndarray, int], ndi.label(pos_pixels, structure=struct)
        )
        for p_id in range(1, num_pos + 1):
            p_mask = labeled_pos == p_id
            if np.sum(p_mask) < 5:
                continue
            c_border = (
                ndi.binary_dilation(p_mask, structure=struct, iterations=iterations)
                & ~p_mask
            ) & brain_inner
            overlay[c_border, :3] = RED_RGB
            overlay[c_border, 3] = 255

    if np.any(neg_pixels):
        labeled_neg, num_neg = cast(
            tuple[np.ndarray, int], ndi.label(neg_pixels, structure=struct)
        )
        for n_id in range(1, num_neg + 1):
            n_mask = labeled_neg == n_id
            if np.sum(n_mask) < 5:
                continue
            c_border = (
                ndi.binary_dilation(n_mask, structure=struct, iterations=iterations)
                & ~n_mask
            ) & brain_inner
            overlay[c_border, :3] = BLUE_RGB
            overlay[c_border, 3] = 255

    border_idx = overlay[:, :, 3] > 0
    if np.any(border_idx):
        t_arr[border_idx] = overlay[border_idx, :3]
        Image.fromarray(t_arr).save(t_png)


def append_contrast_maps(
    cdata, result, receipt, cortex_assets, bundle, panels_dir, grid
):
    from cortex_group_contrasts import T_CMAP

    if result is None:
        cdata["has_contrast"] = False
        return cdata
    assert result is not None
    contrast_result = cast(dict[str, np.ndarray], result)
    cdata["has_contrast"] = True
    display_vmax = contrast_display_vmax(receipt)
    names = list(GRADIENT_SUBREGIONS) + list(_wta_network_order(grid.is_7net))
    cid = cdata["spec"]["id"]
    for idx, name in enumerate(names):
        tiles = cdata["grad_cortex_pngs"] if idx < 3 else cdata["wta_cortex_pngs"]
        fc_path = tiles[name]
        panel_dir = fc_path.parent
        tag = f"{cid}_{grid.tag}_{name.lower()}_vs_iowa_hc"
        t_path = panel_dir / f"{tag}.png"
        composite = panel_dir / f"{tag}_composite.png"
        cache = composite.with_suffix(".cache.json")
        colors = contrast_network_colors(
            contrast_result["significant"][idx], bundle["roi_registry"]
        )
        if result is None:
            caption = "Reference group - no comparison"
        else:
            nsig = int(result["significant"][idx].sum())
            nvalid = int(result["testable"][idx].sum())
            caption = (
                f"t vs Iowa HC | {nsig}/400 significant ROIs (uncorrected p < 0.05)"
            )
            if nsig == 0:
                caption = "No significant ROIs (uncorrected p < 0.05)"
            if nvalid < 400:
                caption += f" | {400 - nvalid} not testable"
        inputs = {
            "schema": "fc_bar_t_network_fill_sign_contour_v4",
            "fc": _file_fingerprint(fc_path),
            "contrast": None
            if result is None
            else {k: _array_fingerprint(v[idx]) for k, v in contrast_result.items()},
            "t_stat_vmax": receipt["t_vmax"],
            "t_display_vmax": display_vmax,
            "cmap": T_CMAP,
            "caption": caption,
            "registry": _sha256_json(bundle["roi_registry"]),
            "assets": {k: _file_fingerprint(v) for k, v in cortex_assets.items()},
            "code": _file_fingerprint(Path(__file__)),
            "vertex_arrays": {
                h: [_array_fingerprint(a) for a in arrays]
                for h, arrays in bundle["vertex_arrays"].items()
            },
        }
        outputs = {"composite": composite}
        dimensions: dict[str, tuple[int, int] | None] = {"composite": (1000, 2186)}
        if result is not None:
            outputs["surface"] = t_path
            dimensions["surface"] = (1000, 700)
        if not _is_valid_workbench_cache(
            cache,
            expected_inputs=inputs,
            output_paths=outputs,
            expected_dimensions=dimensions,
        ):
            if result is not None:
                render_cortex_surface(
                    panel_tag=tag,
                    colors=colors,
                    roi_registry=bundle["roi_registry"],
                    vertex_arrays=bundle["vertex_arrays"],
                    cortex_assets=cortex_assets,
                    panel_dir=panel_dir,
                    out_png=t_path,
                )
                pos_mask_roi = (
                    contrast_result["significant"][idx]
                    & np.isfinite(contrast_result["t"][idx])
                    & (contrast_result["t"][idx] > 0)
                )
                neg_mask_roi = (
                    contrast_result["significant"][idx]
                    & np.isfinite(contrast_result["t"][idx])
                    & (contrast_result["t"][idx] < 0)
                )
                apply_significant_cluster_contours(
                    t_path,
                    pos_mask_roi=pos_mask_roi,
                    neg_mask_roi=neg_mask_roi,
                    roi_registry=bundle["roi_registry"],
                    vertex_arrays=bundle["vertex_arrays"],
                    cortex_assets=cortex_assets,
                    panel_dir=panel_dir,
                    panel_tag=tag,
                    iterations=4,
                )
            stack_contrast_panel(
                fc_path, None if result is None else t_path, composite, caption
            )
            cache.write_text(
                json.dumps(_workbench_panel_cache_manifest(inputs, outputs), indent=2)
                + "\n"
            )
        tiles[name] = composite
    return cdata


def render_multidataset_hipp_cortex_figure_2col(
    cohort_data: list[dict[str, Any]],
    output_png: Path,
    spec: GridSpec = GridSpec("2mm", "36p"),
    t_display_vmax: float | None = None,
    dpi: int = 220,
    layout_only: bool = False,
):
    # Split cohort data into Column 1 and Column 2
    data_col1 = [c for c in cohort_data if c["spec"].get("col", 1) == 1]
    data_col2 = [c for c in cohort_data if c["spec"].get("col", 1) == 2]
    if not data_col1 and cohort_data:
        data_col1 = cohort_data
    is_single_col = len(data_col2) == 0
    n_rows = (
        max(len(data_col1), len(data_col2)) if not is_single_col else len(data_col1)
    )

    row_heights_col1 = [
        2186 if c.get("has_contrast", False) else 1408 for c in data_col1
    ]
    row_heights_col2 = [
        2186 if c.get("has_contrast", False) else 1408 for c in data_col2
    ]
    has_any_contrast = any(c.get("has_contrast", False) for c in cohort_data)
    has_tm = any(bool(c.get("has_tm", False)) for c in cohort_data)
    ncols = 14 if has_tm else 10

    is_7net = spec.is_7net
    wta_display_networks = _wta_display_order(is_7net)
    expected_wta_networks = _wta_network_order(is_7net)
    if set(wta_display_networks) != set(expected_wta_networks):
        raise ValueError(
            "WTA display network keys do not match the registered profile rows: "
            f"display={wta_display_networks!r} profile={expected_wta_networks!r}"
        )
    total_subjects = sum(c["spec"]["n_subjects"] for c in cohort_data)

    if is_single_col:
        fig_w = 58.0
        if has_any_contrast:
            scale_factor = sum(row_heights_col1) / (len(data_col1) * 1408.0)
            fig_h = 44.0 * (n_rows / 5.0) * scale_factor
            top_margin = 0.900
            bottom_margin = 0.088
            y_col_header = top_margin + 0.012
            y_section_banner = top_margin + 0.038
            banner_fs = 23.0
            y_footer_title = 0.070
            y_footer_badges = 0.053
        else:
            fig_h = 44.0 * (n_rows / 5.0) if n_rows != 5 else 44.0
            top_margin = 0.900
            bottom_margin = 0.075
            y_col_header = top_margin + 0.012
            y_section_banner = top_margin + 0.038
            banner_fs = 23.0
            y_footer_title = 0.048
            y_footer_badges = 0.018
        cbar_bottom = 0.014
        cbar_height = 0.011
    else:
        fig_w = 96.0
        if has_any_contrast:
            scale_factor = sum(row_heights_col1) / (len(data_col1) * 1408.0)
            fig_h = 44.0 * (n_rows / 5.0) * scale_factor
            top_margin = 0.885
            bottom_margin = 0.082
            y_col_header = top_margin + 0.007
            y_section_banner = top_margin + 0.026
            banner_fs = 25.0
            y_footer_title = 0.065
            y_footer_badges = 0.048
        else:
            fig_h = 44.0 * (n_rows / 5.0) if n_rows > 5 else 44.0
            top_margin = 0.885
            bottom_margin = 0.070
            y_col_header = top_margin + 0.007
            y_section_banner = top_margin + 0.026
            banner_fs = 25.0
            y_footer_title = 0.046
            y_footer_badges = 0.018
        cbar_bottom = 0.012
        cbar_height = 0.011

    if has_tm:
        # TM adds only the four figure columns; the existing network footer
        # remains the single footer legend. Do not reserve or draw a second
        # TM subregion legend row.
        fig_w = fig_w * (1.42 if is_single_col else 1.45)

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("#ffffff")

    atlas_title_part = " (Kong2022 7-Network)" if is_7net else ""
    title_cohort_info = f" (N={total_subjects} across {len(cohort_data)} Cohorts)"
    fig.suptitle(
        f"Cross-Dataset Hippocampal {spec.smoothing} Parcellation & Cortical Functional Projections{atlas_title_part}{title_cohort_info}",
        fontsize=40.0,
        fontweight="bold",
        x=0.50,
        y=0.985,
        ha="center",
        color="#0b2447",
    )

    if is_single_col:
        gs_main = fig.add_gridspec(
            nrows=1,
            ncols=1,
            left=0.015,
            right=0.985,
            top=top_margin,
            bottom=bottom_margin,
        )
        width_ratios = [
            0.060,
            0.115,
            0.103,
            0.103,
            0.103,
            0.115,
            0.103,
            0.103,
            0.103,
            0.103,
        ] + ([0.115, 0.103, 0.103, 0.103] if has_tm else [])
        gs_left = gs_main[0, 0].subgridspec(
            nrows=n_rows,
            ncols=ncols,
            width_ratios=width_ratios,
            height_ratios=row_heights_col1 if has_any_contrast else None,
            hspace=0.07,
            wspace=0.014,
        )
        gs_right = None
    else:
        gs_main = fig.add_gridspec(
            nrows=1,
            ncols=2,
            left=0.010,
            right=0.990,
            top=top_margin,
            bottom=bottom_margin,
            wspace=0.024,
        )
        width_ratios = [
            0.052,
            0.102,
            0.098,
            0.098,
            0.098,
            0.102,
            0.098,
            0.098,
            0.098,
            0.098,
        ] + ([0.102, 0.098, 0.098, 0.098] if has_tm else [])
        gs_left = gs_main[0, 0].subgridspec(
            nrows=n_rows,
            ncols=ncols,
            width_ratios=width_ratios,
            height_ratios=row_heights_col1 if has_any_contrast else None,
            hspace=0.07,
            wspace=0.014,
        )
        gs_right = gs_main[0, 1].subgridspec(
            nrows=n_rows,
            ncols=ncols,
            width_ratios=width_ratios,
            height_ratios=row_heights_col2
            if has_any_contrast and len(row_heights_col2) == n_rows
            else None,
            hspace=0.07,
            wspace=0.014,
        )

    def render_column_block(gs_col_grid, col_data):
        col_axes = {}
        for r_idx in range(n_rows):
            for c_idx in range(ncols):
                col_axes[(r_idx, c_idx)] = fig.add_subplot(gs_col_grid[r_idx, c_idx])

        # Section 1 Banner: Gradient-Spectral (Cols 1-4)
        bbox_g_start = col_axes[(0, 1)].get_position(fig)
        bbox_g_end = col_axes[(0, 4)].get_position(fig)
        center_sec1 = (bbox_g_start.x0 + bbox_g_end.x1) / 2.0
        grad_banner_text = f"{spec.smoothing} Gradient-Spectral Parcellation (K=3) & Cortical Projections"
        fig.text(
            center_sec1,
            y_section_banner,
            grad_banner_text,
            ha="center",
            va="bottom",
            fontsize=banner_fs,
            fontweight="bold",
            color="#0f4c81",
            bbox={
                "boxstyle": "round,pad=0.38",
                "facecolor": "#e6f0fa",
                "edgecolor": "#b8d5f2",
                "linewidth": 1.8,
            },
        )

        # Section 2 Banner: Normalized WTA (Cols 5-9)
        bbox_w_start = col_axes[(0, 5)].get_position(fig)
        bbox_w_end = col_axes[(0, 9)].get_position(fig)
        center_sec2 = (bbox_w_start.x0 + bbox_w_end.x1) / 2.0
        wta_banner_text = (
            f"{spec.smoothing} Normalized WTA Parcellation & Top Cortical Projections"
        )
        fig.text(
            center_sec2,
            y_section_banner,
            wta_banner_text,
            ha="center",
            va="bottom",
            fontsize=banner_fs,
            fontweight="bold",
            color="#4c1d95",
            bbox={
                "boxstyle": "round,pad=0.38",
                "facecolor": "#f3e8ff",
                "edgecolor": "#d8b4fe",
                "linewidth": 1.8,
            },
        )

        if has_tm:
            # Section 3 Banner: Template-Matching Dominant Atlas (Cols 10-13)
            bbox_t_start = col_axes[(0, 10)].get_position(fig)
            bbox_t_end = col_axes[(0, 13)].get_position(fig)
            center_sec3 = (bbox_t_start.x0 + bbox_t_end.x1) / 2.0
            tm_banner_text = f"{spec.smoothing} Template-Matching Dominant Atlas & Cortical Projections"
            fig.text(
                center_sec3,
                y_section_banner,
                tm_banner_text,
                ha="center",
                va="bottom",
                fontsize=banner_fs,
                fontweight="bold",
                color="#7c2d12",
                bbox={
                    "boxstyle": "round,pad=0.38",
                    "facecolor": "#fef3ec",
                    "edgecolor": "#f5c99b",
                    "linewidth": 1.8,
                },
            )

        # Column Headers
        if is_7net:
            wta_col_titles = [
                ("VAN (VentralAttn)", "#da42f5"),
                ("SMN (Somatomotor)", "#5297c8"),
                ("LAN (Language)", "#8b4513"),
                ("DMN (Default)", "#de616c"),
            ]
        else:
            wta_col_titles = [
                ("VAN (VentralAttn)", "#da42f5"),
                ("SMN (Somatomotor)", "#5297c8"),
                ("Aud (Auditory)", "#18b7c9"),
                ("DMN (Default)", "#de616c"),
            ]

        col_titles = [
            ("Profile", "#102a43"),
            ("Hippo 3D Atlas", "#0f4c81"),
            ("Anterior", "#d95f02"),
            ("Middle (Middle)", "#1b9e77"),
            ("Posterior", "#7570b3"),
            ("Hippo 3D Atlas", "#4c1d95"),
        ] + wta_col_titles
        if has_tm:
            col_titles = (
                col_titles + [("TM Hippo Atlas", "#7c2d12")] + list(TM_DISPLAY_TITLES)
            )
        header_fs = 23.0 if is_single_col else 27.0
        for c_idx, (title, col) in enumerate(col_titles):
            ax_col = col_axes[(0, c_idx)]
            bbox = ax_col.get_position(fig)
            col_x = (bbox.x0 + bbox.x1) / 2.0
            fig.text(
                col_x,
                y_col_header,
                title,
                ha="center",
                va="bottom",
                fontsize=header_fs,
                fontweight="bold",
                color=col,
            )

        # Render Cohort Rows
        for row_idx, cdata in enumerate(col_data):
            cspec = cdata["spec"]
            if tuple(cdata.get("wta_network_order", ())) != tuple(
                expected_wta_networks
            ):
                raise ValueError(
                    f"WTA cohort row {row_idx} has a different profile network order: "
                    f"{cdata.get('wta_network_order')!r}"
                )
            if has_tm and tuple(cdata.get("tm_network_order", ())) != tuple(
                TM_SUBREGIONS
            ):
                raise ValueError(
                    f"TM cohort row {row_idx} has a different TM panel order: "
                    f"{cdata.get('tm_network_order')!r}"
                )

            # 1. Profile Card (Col 0)
            ax_meta = col_axes[(row_idx, 0)]
            ax_meta.set_facecolor(cspec["bg_color"])
            for spine in ax_meta.spines.values():
                spine.set_edgecolor("#cbd5e1")
                spine.set_linewidth(1.4)
            ax_meta.set_xticks([])
            ax_meta.set_yticks([])

            badge_patch = patches.FancyBboxPatch(
                (0.04, 0.73),
                0.92,
                0.22,
                boxstyle="round,pad=0.04,rounding_size=0.08",
                facecolor=cspec["badge_bg"],
                edgecolor="none",
                transform=ax_meta.transAxes,
                zorder=2,
            )
            ax_meta.add_patch(badge_patch)
            ax_meta.text(
                0.50,
                0.84,
                cspec["dataset_name"],
                transform=ax_meta.transAxes,
                ha="center",
                va="center",
                color=cspec["badge_fg"],
                fontsize=22.0,
                fontweight="bold",
                zorder=3,
            )
            cohort_fs = 28.0 if len(cspec["cohort_name"]) <= 8 else 22.0
            ax_meta.text(
                0.50,
                0.50,
                cspec["cohort_name"],
                transform=ax_meta.transAxes,
                ha="center",
                va="center",
                color="#0f172a",
                fontsize=cohort_fs,
                fontweight="bold",
            )
            ax_meta.text(
                0.50,
                0.27,
                f"(N = {cspec['n_subjects']})",
                transform=ax_meta.transAxes,
                ha="center",
                va="center",
                color="#334155",
                fontsize=21.0,
                fontweight="bold",
            )
            desc_fs = 15.0 if len(cspec["cohort_desc"]) <= 18 else 13.0
            ax_meta.text(
                0.50,
                0.08,
                cspec["cohort_desc"],
                transform=ax_meta.transAxes,
                ha="center",
                va="center",
                color="#64748b",
                fontsize=desc_fs,
                fontstyle="italic",
            )

            # 2. Gradient Hippocampus Surface & Legend (Col 1)
            ax_gsurf = col_axes[(row_idx, 1)]
            ax_gsurf.imshow(cdata["surf_grad"])
            ax_gsurf.axis("off")

            # 3. Gradient Cortex Maps (Cols 2-4: Anterior, Middle, Posterior)
            for c_idx, subreg in enumerate(GRADIENT_SUBREGIONS):
                ax_gcort = col_axes[(row_idx, 2 + c_idx)]
                img_cortex = plt.imread(cdata["grad_cortex_pngs"][subreg])
                ax_gcort.imshow(img_cortex)
                ax_gcort.axis("off")

            # 4. WTA Hippocampus Surface & Legend (Col 5)
            ax_wsurf = col_axes[(row_idx, 5)]
            ax_wsurf.imshow(cdata["surf_wta"])
            ax_wsurf.axis("off")

            # 5. WTA Cortex Maps (Cols 6-9: 4 top networks in display order: VAN, SMN, LAN, DMN)
            for w_idx, net in enumerate(wta_display_networks):
                ax_wcort = col_axes[(row_idx, 6 + w_idx)]
                img_cortex = plt.imread(cdata["wta_cortex_pngs"][net])
                ax_wcort.imshow(img_cortex)
                ax_wcort.axis("off")

            if has_tm:
                # 6. TM Dominant-Atlas Hippocampus (Col 10; blank for HCP 7T)
                ax_tsurf = col_axes[(row_idx, 10)]
                if cdata.get("surf_tm") is not None:
                    ax_tsurf.imshow(cdata["surf_tm"])
                ax_tsurf.axis("off")

                # 7. TM Cortex Maps (Cols 11-13: Anterior/VAN, Middle/SMN, Posterior/DMN)
                tm_tiles = cdata.get("tm_cortex_pngs", {})
                for t_idx, name in enumerate(TM_SUBREGIONS):
                    ax_tcort = col_axes[(row_idx, 11 + t_idx)]
                    if name in tm_tiles:
                        img_tm = plt.imread(tm_tiles[name])
                        ax_tcort.imshow(img_tm)
                    ax_tcort.axis("off")

        # Horizontal dividers between cohort rows (gray dashed line)
        for r_idx in range(len(col_data) - 1):
            ax_curr = col_axes[(r_idx, 0)]
            ax_next = col_axes[(r_idx + 1, 0)]
            ax_last_col = col_axes[(r_idx, ncols - 1)]
            y_curr_bot = ax_curr.get_position(fig).y0
            y_next_top = ax_next.get_position(fig).y1
            y_div = (y_curr_bot + y_next_top) / 2.0
            x_left = ax_curr.get_position(fig).x0 - 0.003
            x_right = ax_last_col.get_position(fig).x1 + 0.003
            div_line = Line2D(
                [x_left, x_right],
                [y_div, y_div],
                color="#94a3b8",
                linestyle=(0, (6, 5)),
                linewidth=2.2,
                alpha=0.85,
                transform=fig.transFigure,
                zorder=10,
            )
            fig.add_artist(div_line)

        return col_axes

    render_column_block(gs_left, data_col1)
    if not is_single_col and gs_right is not None and data_col2:
        render_column_block(gs_right, data_col2)

    # Global Footers & Colorbar
    if is_single_col:
        # 1-Column Footer Layout
        if is_7net:
            fig.text(
                0.300,
                y_footer_title,
                "Normalized WTA Functional Networks (Kong2022 7-Network):",
                ha="center",
                va="center",
                fontsize=27.0,
                fontweight="bold",
                color="#4c1d95",
            )
            w_patches = [
                ("DMN", "#de616c"),
                ("SMN", "#5297c8"),
                ("VAN", "#da42f5"),
                ("DAN", "#33a02c"),
                ("CON", "#f39c12"),
                ("LAN", "#8b4513"),
                ("VN", "#6a3d9a"),
            ]
            badge_offset_y = 0.011 if has_any_contrast else 0.013
            badge_h = 0.022 if has_any_contrast else 0.026
            for idx, (label, col) in enumerate(w_patches):
                x_p = 0.055 + idx * 0.075
                fig.patches.append(
                    patches.Rectangle(
                        (x_p - 0.009, y_footer_badges - badge_offset_y),
                        0.008,
                        badge_h,
                        facecolor=col,
                        edgecolor="#333333",
                        linewidth=1.4,
                        transform=fig.transFigure,
                        figure=fig,
                    )
                )
                fig.text(
                    x_p + 0.002,
                    y_footer_badges,
                    label,
                    ha="left",
                    va="center",
                    fontsize=24.0,
                    fontweight="bold",
                    color="#1e293b",
                )
        else:
            fig.text(
                0.300,
                y_footer_title,
                "Normalized WTA Functional Networks (Kong2022):",
                ha="center",
                va="center",
                fontsize=27.0,
                fontweight="bold",
                color="#4c1d95",
            )
            w_patches = [
                ("DMN", "#de616c"),
                ("VAN", "#da42f5"),
                ("Aud", "#18b7c9"),
                ("SMN", "#5297c8"),
                ("DAN", "#33a02c"),
                ("CON", "#f39c12"),
                ("LAN", "#8b4513"),
                ("VN", "#6a3d9a"),
            ]
            badge_offset_y = 0.011 if has_any_contrast else 0.013
            badge_h = 0.022 if has_any_contrast else 0.026
            for idx, (label, col) in enumerate(w_patches):
                x_p = 0.045 + idx * 0.075
                fig.patches.append(
                    patches.Rectangle(
                        (x_p - 0.009, y_footer_badges - badge_offset_y),
                        0.008,
                        badge_h,
                        facecolor=col,
                        edgecolor="#333333",
                        linewidth=1.4,
                        transform=fig.transFigure,
                        figure=fig,
                    )
                )
                fig.text(
                    x_p + 0.002,
                    y_footer_badges,
                    label,
                    ha="left",
                    va="center",
                    fontsize=24.0,
                    fontweight="bold",
                    color="#1e293b",
                )

        fig.text(
            0.820,
            y_footer_title,
            "Mean Fisher-z Hippo-Cortex FC",
            ha="center",
            va="center",
            fontsize=25.0,
            fontweight="bold",
            color="#102a43",
        )
        cbar_ax = fig.add_axes((0.690, cbar_bottom, 0.260, cbar_height))
    else:
        # 2-Column Footer Layout
        if is_7net:
            fig.text(
                0.370,
                y_footer_title,
                "Normalized WTA Functional Networks (Kong2022 7-Network):",
                ha="center",
                va="center",
                fontsize=31.0,
                fontweight="bold",
                color="#4c1d95",
            )
            w_patches = [
                ("DMN", "#de616c"),
                ("SMN", "#5297c8"),
                ("VAN", "#da42f5"),
                ("DAN", "#33a02c"),
                ("CON", "#f39c12"),
                ("LAN", "#8b4513"),
                ("VN", "#6a3d9a"),
            ]
            for idx, (label, col) in enumerate(w_patches):
                x_p = 0.080 + idx * 0.082
                fig.patches.append(
                    patches.Rectangle(
                        (x_p - 0.010, y_footer_badges - 0.013),
                        0.009,
                        0.026,
                        facecolor=col,
                        edgecolor="#333333",
                        linewidth=1.4,
                        transform=fig.transFigure,
                        figure=fig,
                    )
                )
                fig.text(
                    x_p + 0.002,
                    y_footer_badges,
                    label,
                    ha="left",
                    va="center",
                    fontsize=27.5,
                    fontweight="bold",
                    color="#1e293b",
                )
        else:
            fig.text(
                0.370,
                y_footer_title,
                "Normalized WTA Functional Networks (Kong2022):",
                ha="center",
                va="center",
                fontsize=31.0,
                fontweight="bold",
                color="#4c1d95",
            )
            w_patches = [
                ("DMN", "#de616c"),
                ("VAN", "#da42f5"),
                ("Aud", "#18b7c9"),
                ("SMN", "#5297c8"),
                ("DAN", "#33a02c"),
                ("CON", "#f39c12"),
                ("LAN", "#8b4513"),
                ("VN", "#6a3d9a"),
            ]
            for idx, (label, col) in enumerate(w_patches):
                x_p = 0.065 + idx * 0.082
                fig.patches.append(
                    patches.Rectangle(
                        (x_p - 0.010, y_footer_badges - 0.013),
                        0.009,
                        0.026,
                        facecolor=col,
                        edgecolor="#333333",
                        linewidth=1.4,
                        transform=fig.transFigure,
                        figure=fig,
                    )
                )
                fig.text(
                    x_p + 0.002,
                    y_footer_badges,
                    label,
                    ha="left",
                    va="center",
                    fontsize=27.5,
                    fontweight="bold",
                    color="#1e293b",
                )

        fig.text(
            0.855,
            y_footer_title,
            "Mean Fisher-z Hippo-Cortex FC",
            ha="center",
            va="center",
            fontsize=28.0,
            fontweight="bold",
            color="#102a43",
        )
        cbar_ax = fig.add_axes((0.745, cbar_bottom, 0.220, cbar_height))

    norm = Normalize(vmin=-COLORBAR_VMAX, vmax=COLORBAR_VMAX)
    cmap = plt.get_cmap(DEFAULT_CMAP)
    cbar = fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        orientation="horizontal",
    )
    cbar.set_ticks(list(COLORBAR_TICKS))
    cbar.ax.tick_params(labelsize=22.0 if is_single_col else 24.0)
    for t in cbar.ax.xaxis.get_ticklabels():
        t.set_fontweight("bold")

    if has_any_contrast:
        cbar_title_y = 0.032 if is_single_col else 0.030
        cbar_y = 0.014 if is_single_col else 0.013
        cbar_h = 0.011 if is_single_col else 0.010
        for text in fig.texts:
            if text.get_text() == "Mean Fisher-z Hippo-Cortex FC":
                text.set_text("Mean Fisher-z FC (upper maps)")
                text.set_position((0.265 if is_single_col else 0.370, cbar_title_y))
        cbar_ax.set_position(
            (
                0.065 if is_single_col else 0.170,
                cbar_y,
                0.400 if is_single_col else 0.400,
                cbar_h,
            )
        )

        center_x = 0.745 if is_single_col else 0.770
        fig.text(
            center_x,
            cbar_title_y,
            "Group Differences vs Iowa HC (lower maps):",
            ha="center",
            va="center",
            fontsize=25.0 if is_single_col else 28.0,
            fontweight="bold",
            color="#452765",
        )

        badge_w = 0.018 if is_single_col else 0.014
        badge_h = 0.018 if is_single_col else 0.015
        b_y = cbar_y - 0.003
        b_fs = 21.0 if is_single_col else 24.0

        pos_box_x = 0.565 if is_single_col else 0.600
        fig.patches.append(
            patches.FancyBboxPatch(
                (pos_box_x, b_y),
                badge_w,
                badge_h,
                boxstyle="round,pad=0.002,rounding_size=0.004",
                facecolor="#ffffff",
                edgecolor="#dc1414",
                linewidth=3.2,
                transform=fig.transFigure,
                figure=fig,
            )
        )
        fig.text(
            pos_box_x + badge_w / 2.0,
            b_y + badge_h / 2.0,
            "+",
            ha="center",
            va="center",
            fontsize=b_fs,
            fontweight="bold",
            color="#dc1414",
        )
        fig.text(
            pos_box_x + badge_w + 0.007,
            b_y + badge_h / 2.0,
            "Higher FC than HC (t > 0)",
            ha="left",
            va="center",
            fontsize=b_fs - 1.0,
            fontweight="bold",
            color="#b91c1c",
        )

        neg_box_x = 0.760 if is_single_col else 0.785
        fig.patches.append(
            patches.FancyBboxPatch(
                (neg_box_x, b_y),
                badge_w,
                badge_h,
                boxstyle="round,pad=0.002,rounding_size=0.004",
                facecolor="#ffffff",
                edgecolor="#1464f0",
                linewidth=3.2,
                transform=fig.transFigure,
                figure=fig,
            )
        )
        fig.text(
            neg_box_x + badge_w / 2.0,
            b_y + badge_h / 2.0,
            "−",
            ha="center",
            va="center",
            fontsize=b_fs,
            fontweight="bold",
            color="#1464f0",
        )
        fig.text(
            neg_box_x + badge_w + 0.007,
            b_y + badge_h / 2.0,
            "Lower FC than HC (t < 0)",
            ha="left",
            va="center",
            fontsize=b_fs - 1.0,
            fontweight="bold",
            color="#1d4ed8",
        )

        note_fs = 17.5 if is_single_col else 20.0
        fig.text(
            center_x,
            0.004,
            "Outlines: Red (+) / Blue (−) t direction (uncorrected p < 0.05) | Inner fill: Canonical 7 Networks (matching WTA legend on left)",
            ha="center",
            fontsize=note_fs,
            fontweight="bold",
            color="#452765",
        )

    if layout_only:
        suptitle = getattr(fig, "_suptitle", None)
        if suptitle is None:
            raise RuntimeError("Figure suptitle was not initialized")
        suptitle.set_text("LAYOUT PREVIEW - statistical results pending input repair")
    print(f"Saving high-resolution master figure to {output_png}...")
    fig.savefig(output_png, dpi=dpi, facecolor="#ffffff", edgecolor="none")
    plt.close(fig)
    print("Done!")


def _render_grid():
    parser = argparse.ArgumentParser(
        description="Render Gradient & WTA Hippocampus + Cortex comparison"
    )
    parser.add_argument(
        "--version",
        choices=["canonical"],
        default="canonical",
        help="Data branch (single canonical version).",
    )
    script_dir = Path(__file__).resolve().parent
    default_png = script_dir / "multidataset_hipp_cortex_gradient_wta.png"
    parser.add_argument(
        "--out", default=None, help="Output PNG path (synonym for --out-png)"
    )
    parser.add_argument(
        "--out-png", default=None, help=f"Output PNG path (default: {default_png})"
    )
    parser.add_argument(
        "--smoothing",
        choices=["2mm"],
        default="2mm",
        help="Smoothing resolution (default: 2mm)",
    )
    parser.add_argument(
        "--variant",
        choices=["36p"],
        default="36p",
        help="Preprocessing variant (default: 36p)",
    )
    parser.add_argument(
        "--signal-branch",
        choices=list(SIGNAL_BRANCHES),
        default="standard",
        help="Signal branch for hippocampal products (standard or ctxreg).",
    )
    parser.add_argument(
        "--atlas",
        choices=["kong2022", "7net"],
        default="7net",
        help="Atlas / network taxonomy version. Default: 7net (canonical7, unsuffixed); kong2022 is an equivalent path alias.",
    )
    parser.add_argument(
        "--grid",
        choices=list(GRID_VARIANTS),
        default=None,
        help="Grid variant shortcut",
    )
    parser.add_argument(
        "--contrast-manifest",
        required=True,
        help="Explicit all-grid subject-level Welch/raw-p preparation manifest",
    )
    parser.add_argument(
        "--tm-contrast-manifest",
        default=None,
        help=(
            "Explicit TM dominant-atlas subject-level Welch manifest. "
            "When omitted, the legacy Gradient+WTA layout is rendered."
        ),
    )
    args = parser.parse_args()

    spec = (
        GridSpec.from_tag(args.grid, atlas=args.atlas)
        if args.grid
        else GridSpec(
            args.smoothing,
            args.variant,
            signal_branch=args.signal_branch,
            atlas=args.atlas,
        )
    )

    out_arg = args.out_png or args.out
    if out_arg:
        output_png = Path(out_arg).resolve()
    else:
        output_tag = spec.file_tag
        output_png = (
            script_dir / f"multidataset_hipp_cortex_gradient_wta_{output_tag}.png"
        )

    script_dir.mkdir(parents=True, exist_ok=True)
    panels_dir = script_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    cortex_assets = resolve_cortex_assets()
    bundle = load_kong2022_bundle(dlabel_path=cortex_assets["dlabel"])

    # Pass this module's namespace without re-importing its plotting state.
    from types import SimpleNamespace

    from cortex_group_contrasts import (  # pyright: ignore[reportMissingImports]
        figure_cohorts,
        load_contrasts,
    )

    cohorts_spec = figure_cohorts(SimpleNamespace(**globals()), spec, args.version)
    receipt, contrasts = load_contrasts(args.contrast_manifest, spec, args.version)
    if set(contrasts) != {c["id"] for c in cohorts_spec}:
        raise ValueError("Contrast cohort roster differs from figure roster")
    if receipt["roi_registry"] != bundle["roi_registry"]:
        raise ValueError("Contrast ROI registry differs from rendered atlas")

    cohort_data = []
    tm_receipt = None
    tm_contrasts = None
    tm_means: dict[str, Any] = {}
    use_tm = args.tm_contrast_manifest is not None
    if use_tm:
        if (spec.smoothing, spec.variant) != ("2mm", "36p"):
            raise ValueError(f"TM third column requires 2mm/36p, got {spec.tag!r}")
        from tm_group_contrasts import (  # pyright: ignore[reportMissingImports]
            load_tm_contrasts as _load_tm_contrasts,
        )

        tm_receipt, tm_contrasts = _load_tm_contrasts(
            args.tm_contrast_manifest, spec, args.version
        )
        if set(tm_contrasts) != {c["id"] for c in cohorts_spec}:
            raise ValueError("TM contrast roster differs from figure roster")
        if tm_receipt["roi_registry"] != bundle["roi_registry"]:
            raise ValueError("TM ROI registry differs from rendered atlas")
        tm_means = tm_display_means_from_receipt(tm_receipt)
    for cspec in cohorts_spec:
        cdata = load_cohort_data(
            cspec,
            panels_dir=panels_dir,
            cortex_assets=cortex_assets,
            bundle=bundle,
            smoothing=spec.smoothing,
            variant=spec.variant,
            is_7net=spec.is_7net,
        )
        append_contrast_maps(
            cdata,
            contrasts[cspec["id"]],
            receipt,
            cortex_assets,
            bundle,
            panels_dir,
            spec,
        )
        if use_tm:
            if tm_receipt is None or tm_contrasts is None:
                raise ValueError(
                    "TM manifest requested but TM contrasts failed to load"
                )
            tm_mean = tm_means.get(cspec["id"])
            tm_cdata = load_tm_cohort_data(
                cspec,
                tm_mean,
                panels_dir,
                cortex_assets,
                bundle,
                smoothing=spec.smoothing,
                variant=spec.variant,
            )
            cdata["surf_tm"] = tm_cdata["surf_tm"]
            cdata["tm_cortex_pngs"] = tm_cdata["tm_cortex_pngs"]
            cdata["tm_network_order"] = tm_cdata["tm_network_order"]
            cdata["has_tm"] = tm_cdata["has_tm"]
            append_tm_contrast_maps(
                cdata,
                tm_contrasts[cspec["id"]],
                tm_receipt,
                cortex_assets,
                bundle,
                panels_dir,
                spec,
            )
        cohort_data.append(cdata)

    render_multidataset_hipp_cortex_figure_2col(
        cohort_data=cohort_data,
        output_png=output_png,
        spec=spec,
        t_display_vmax=contrast_display_vmax(receipt),
    )


if __name__ == "__main__":
    raise SystemExit(
        "Standalone rendering has been removed. Use "
        "scripts/visualize/run_multidataset_suite.py "
        "with --figures to select the figure type; both maintained 2mm/36P signal branches are processed in bounded parallel by default."
    )

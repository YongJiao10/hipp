#!/usr/bin/env python3
from __future__ import annotations

"""
Process Step Description:
Build group-level results overview. Summarizes clustering and functional network mapping results to generate comprehensive overview charts or files for visualization.

Main Flow Path:
- 9th step in the numbered pipeline sequence (first stage of presentation output generation):
  01_01_run_hippunfold_t1w_local.py -> 01_02_check_hippunfold_outputs.py -> 02_01_prepare_group_inputs.py -> 02_02_build_qc_manifest.py -> 03_01/03_02 reference building -> 04_01/04_02 FC materialization -> 05_01 gradient computation -> 05_02 group atlas parcellation -> 06_01_build_group_overview.py (this step) -> 07_01 group matrix overview.

Inputs:
- Command-line arguments:
  - --group-root: Directory where group-atlas intermediate files are saved (default: `outputs_hcp/work/group_atlas/routea`).
  - --present-dir: Directory where final presentation figures will be saved (default: `outputs_hcp/present/overviews`).
  - --methods: Parcellation methods to process: "intrinsic-spectral", "network-spectral", "gradient-spectral", "wta-top".
  - --atlases: Cortical atlases to process (default: Kong2022); Lynch2024 atlases support gradient-spectral and direct-network wta-top.
  - --coverage-threshold: Selected coverage threshold for overview outputs (default: 0.80).
  - --scene: Path to standard wb scene template file.
  - --route-tag: Custom workflow branch tag to override automatic detection.
  - --mode-tag: Optional explicit overview label override (default: none). No automatic sparse/full inference is performed; all maintained formal runs are full.
  - --assignment-variant: Network mapping variant: "fisherz", "normalized_top20_count", "weighted_fisherz", or "both" (default: "both").
  - --manifest-name: Name of generated presentation manifest JSON (default: "group_overview_manifest.json").
  - --skip-readme: Skip writing the README.md summary in presentation directory.
  - --skip-manifest: Skip writing the overview manifest JSON file.
- Physical inputs:
  - Group parcellation label arrays: manifest `assignment_variants[default_assignment_variant].group_functional_atlas_labels` for cluster-level methods; vertex-level methods use the manifest top-level label array.
  - Group mapping status files and dataset-scoped template assets from manifest paths; stale or foreign absolute mounts fail validation against canonical current assets.

Outputs:
- Command-line output / file outputs:
  - Overview panel PNG figures for each rendered method/assignment variant; gradient names include the component tag, and cluster-based names include the assignment suffix.
  - Group overview manifest JSON file: `{present_dir}/{manifest_name}`.
  - Summary README file: `{present_dir}/README.md`.
"""
import argparse
import csv
import json
import re
import tempfile
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from nibabel.gifti import GiftiDataArray, GiftiImage, GiftiLabel, GiftiLabelTable

from lib.output_layout import DEFAULT_OVERVIEWS_DIR, DEFAULT_ROUTEA_GROUP_ATLAS_ROOT
from lib.template_assets import validate_template_assets
from lib.route_tags import ROUTE_TAG_CHOICES, infer_route_tag_from_group_root, validate_group_atlas_root_for_route
from lib.workflow_utils import write_json
from lib.surface_utils import equal_vertex_long_axis_bins, long_axis_order, render_locked_grid, network_color_map
from lib.io_utils import load_json
from lib.group_atlas_lineage import validate_group_atlas_lineage
from lib.gifti_labels import (
    resolve_cluster_label_path,
    resolve_functional_atlas_array_paths,
    make_named_label_gifti,
    materialize_combo,
)
from lib.network_colors import load_network_hex_colors
from lib.gradient_assignment_contract import validate_gradient_assignment, sha256_file
from lib.surface_affinity import (
    GRADIENT_SPECTRAL_AFFINITY,
    GRADIENT_SPECTRAL_AFFINITY_DESCRIPTION,
    validate_gradient_spectral_affinity_provenance,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GROUP_ROOT = DEFAULT_ROUTEA_GROUP_ATLAS_ROOT
DEFAULT_PRESENT_DIR = DEFAULT_OVERVIEWS_DIR
DEFAULT_SCENE = REPO_ROOT / "config" / "wb_locked_native_view_lateral_medial.scene"
METHOD_CHOICES = [
    "network-spectral",
    "gradient-spectral",
    "intrinsic-spectral",
    "wta-top",
]
METHODS_DEFAULT = [
    "intrinsic-spectral",
    "network-spectral",
    "wta-top",
]
# kong2019/kong2019_8net retired with 04_03_materialize_kong2019_fc.py.
ATLASES = ["kong2022", "lynch2024_20net", "lynch2024_7net"]
DEFAULT_ATLASES = ["kong2022"]
SMOOTHINGS = ["2mm", "4mm"]
FS_TITLE = 11
FS_LABEL = 10
FS_TICK = 9
FS_SMALL = 8.5
FS_LEGEND_SMALL = FS_SMALL - 2.0
FS_CELL = 6
FS_MISSING = 13
FS_BIG_TITLE = 13
FS_SUPTITLE = 18
CLUSTERLESS_METHODS = {"wta-top"}
ASSIGNMENT_VARIANTS = ("fisherz", "normalized_top20_count", "weighted_fisherz")
DEFAULT_RENDER_ASSIGNMENT_VARIANTS = ("fisherz", "normalized_top20_count")
ASSIGNMENT_VARIANT_CHOICES = ASSIGNMENT_VARIANTS + ("both",)
WTA_OVERVIEW_PROFILE_BINS = 20
WTA_OVERVIEW_SIDE_PANEL_FONT_DELTA_PT = 4
WTA_OVERVIEW_SIDE_PANEL_TITLE_FS = FS_TITLE + WTA_OVERVIEW_SIDE_PANEL_FONT_DELTA_PT
WTA_OVERVIEW_SIDE_PANEL_LABEL_FS = FS_LABEL + WTA_OVERVIEW_SIDE_PANEL_FONT_DELTA_PT
WTA_OVERVIEW_SIDE_PANEL_TICK_FS = FS_TICK + WTA_OVERVIEW_SIDE_PANEL_FONT_DELTA_PT
WTA_OVERVIEW_SIDE_PANEL_LEGEND_FS = FS_LEGEND_SMALL + WTA_OVERVIEW_SIDE_PANEL_FONT_DELTA_PT
WTA_NON_NETWORK_NAME = "__non_network_or_null__"
GRADIENT_METHOD = "gradient-spectral"
CTXDROP_ORIGINAL_FC_ANNOTATION_SIGNAL = "original_timeseries"
CTXDROP_ORIGINAL_FC_ANNOTATION_NOTE = "Original-FC annotation of cortex-controlled clusters"
CTXDROP_NORMALIZED_COUNT_ANNOTATION_NOTE = "Counts derived from original-FC ranking"


def default_methods_for_atlas(atlas: str) -> tuple[str, ...]:
    """Return the methods rendered when the caller omits --methods."""
    if str(atlas) in {"lynch2024_20net", "lynch2024_7net"}:
        return ("gradient-spectral", "wta-top")
    if str(atlas) != "kong2022":
        return tuple(method for method in METHODS_DEFAULT if method != "wta-top")
    return tuple(METHODS_DEFAULT)


def gradient_affinity_provenance(manifests: list[dict[str, object]]) -> dict[str, str]:
    return validate_gradient_spectral_affinity_provenance(manifests)


def ctxdrop_original_fc_annotation_provenance(
    *,
    manifest_paths: list[Path],
    manifests: list[dict[str, object]],
) -> dict[str, object]:
    """Validate and describe the accepted ctxdrop cluster annotation source."""
    if len(manifest_paths) != 4 or len(manifests) != 4:
        raise ValueError(
            "ctxdrop original-FC annotation provenance requires exactly four "
            "2mm/4mm by L/R manifests"
        )

    known_routes = ("routea_ctxdrop", "routea_ctxreg", "routea")
    path_route_hints: list[str | None] = []
    for path in manifest_paths:
        matches = [route for route in known_routes if route in Path(path).parts]
        path_route_hints.append(matches[0] if len(matches) == 1 else None)
    if "routea_ctxdrop" not in path_route_hints:
        return {}
    if any(route != "routea_ctxdrop" for route in path_route_hints):
        raise ValueError(
            "ctxdrop original-FC annotation provenance mismatch: mixed route paths; "
            f"paths={[str(path) for path in manifest_paths]}"
        )

    route_tags: list[str] = []
    for path, manifest in zip(manifest_paths, manifests):
        payload = manifest.get("resume_input_signature_payload")
        if not isinstance(payload, dict) or not str(payload.get("route_tag", "")).strip():
            raise ValueError(
                "ctxdrop original-FC annotation provenance mismatch: missing "
                f"resume route_tag in {path}"
            )
        route_tag = str(payload["route_tag"]).strip()
        route_tags.append(route_tag)
    if set(route_tags) != {"routea_ctxdrop"}:
        raise ValueError(
            "ctxdrop original-FC annotation provenance mismatch: route tags="
            f"{route_tags}"
        )

    expected_grid = {("2mm", "L"), ("2mm", "R"), ("4mm", "L"), ("4mm", "R")}
    observed_grid = {(str(m.get("smoothing", "")), str(m.get("hemi", ""))) for m in manifests}
    if observed_grid != expected_grid:
        raise ValueError(
            "ctxdrop original-FC annotation provenance mismatch: expected complete "
            f"smoothing/hemi grid={sorted(expected_grid)}, observed={sorted(observed_grid)}"
        )

    profile_fc_modes = {str(m.get("profile_fc_mode", "")) for m in manifests}
    if profile_fc_modes != {"cluster_mean_timeseries"}:
        raise ValueError(
            "ctxdrop original-FC annotation provenance mismatch: profile_fc_mode="
            f"{sorted(profile_fc_modes)}"
        )

    profile_source_modes = {str(m.get("profile_source_mode", "")) for m in manifests}
    if profile_source_modes != {"canonical"}:
        raise ValueError(
            "ctxdrop original-FC annotation provenance mismatch: profile_source_mode="
            f"{sorted(profile_source_modes)}"
        )
    if any(m.get("profile_source_root") is not None for m in manifests):
        raise ValueError(
            "ctxdrop original-FC annotation provenance mismatch: explicit profile_source_root present"
        )
    if any(m.get("profile_source_registry") is not None for m in manifests):
        raise ValueError(
            "ctxdrop original-FC annotation provenance mismatch: explicit profile_source_registry present"
        )

    profile_surface_roots = {str(m.get("profile_surface_root", "")).strip() for m in manifests}
    if len(profile_surface_roots) != 1 or "" in profile_surface_roots:
        raise ValueError(
            "ctxdrop original-FC annotation provenance mismatch: profile_surface_root="
            f"{sorted(profile_surface_roots)}"
        )
    profile_surface_root = Path(next(iter(profile_surface_roots))).resolve()
    if (
        profile_surface_root.name != "surface"
        or profile_surface_root.parent.name != "shared_inputs"
        or any(route in profile_surface_root.parts for route in ("routea_ctxdrop", "routea_ctxreg"))
    ):
        raise ValueError(
            "ctxdrop original-FC annotation provenance mismatch: expected the canonical "
            f"original surface root, got {profile_surface_root}"
        )

    reference_roots = {str(m.get("reference_root", "")).strip() for m in manifests}
    if len(reference_roots) != 1 or "" in reference_roots:
        raise ValueError(
            "ctxdrop original-FC annotation provenance mismatch: reference_root="
            f"{sorted(reference_roots)}"
        )

    return {
        "cluster_annotation_signal": CTXDROP_ORIGINAL_FC_ANNOTATION_SIGNAL,
        "cluster_annotation_note": CTXDROP_ORIGINAL_FC_ANNOTATION_NOTE,
        "cluster_annotation_profile_fc_mode": "cluster_mean_timeseries",
        "cluster_annotation_profile_surface_root": str(profile_surface_root),
        "cluster_annotation_reference_root": next(iter(reference_roots)),
    }


def overview_source_provenance(
    *,
    method: str,
    manifest_paths: list[Path],
    manifests: list[dict[str, object]],
) -> dict[str, object]:
    provenance: dict[str, object] = {
        "source_group_manifests": [str(path) for path in manifest_paths],
    }
    if _is_gradient_method(method):
        provenance.update(gradient_affinity_provenance(manifests))
        provenance.update(
            ctxdrop_original_fc_annotation_provenance(
                manifest_paths=manifest_paths,
                manifests=manifests,
            )
        )
    return provenance


def overview_manifest_item(
    *, base: dict[str, object], meta: dict[str, object]
) -> dict[str, object]:
    item = dict(base)
    for key in (
        "rendered_title",
        "overview_scope",
        "subject_split_ari",
        "source_group_manifests",
        "overview_source_sidecar",
        "assignment_assets",
        "surface_affinity",
        "affinity_description",
        "cluster_annotation_signal",
        "cluster_annotation_note",
        "cluster_annotation_profile_fc_mode",
        "cluster_annotation_profile_surface_root",
        "cluster_annotation_reference_root",
        "assignment_annotation_note",
    ):
        if key in meta:
            item[key] = meta[key]
    return item


def infer_group_name_from_path(path: Path | str | None) -> str | None:
    if path is None:
        return None
    p_str = str(path).lower()
    if "routea_nthc" in p_str or "_nthc" in p_str or "/nthc" in p_str or "1xxx" in p_str:
        return "NTHC"
    if "routea_tehc" in p_str or "_tehc" in p_str or "/tehc" in p_str or "2xxx" in p_str:
        return "TEHC"
    if "routea_nts" in p_str or "_nts" in p_str or "/nts" in p_str or "3xxx" in p_str:
        return "NTS"
    if "routea_tis" in p_str or "_tis" in p_str or "/tis" in p_str or "4xxx" in p_str:
        return "TIS"
    if "nontrauma" in p_str or "non_trauma" in p_str:
        return "Non-Trauma"
    if "trauma" in p_str:
        return "Trauma"
    if "routea_hc" in p_str or "_hc_" in p_str or p_str.endswith("_hc") or "/hc/" in p_str or p_str.endswith("/hc"):
        return "HC"
    if "routea_symp" in p_str or "_symp_" in p_str or p_str.endswith("_symp") or "/symp/" in p_str or p_str.endswith("/symp"):
        return "Symp"
    if "patient" in p_str:
        return "Patient"
    return None


def overview_title(
    *,
    atlas: str,
    method: str,
    route_tag: str | None,
    mode_tag: str | None,
    assignment_variant: str | None,
    surface_affinity: object = None,
    overview_scope: str = "Group",
    group_name: str | None = None,
    n_subjects: int | None = None,
) -> str:
    group_tag_str = None
    if group_name and n_subjects is not None:
        if "N=" in str(group_name):
            group_tag_str = str(group_name)
        else:
            group_tag_str = f"{group_name} (N={n_subjects})"
    elif group_name:
        group_tag_str = str(group_name)
    elif n_subjects is not None:
        group_tag_str = f"N={n_subjects}"

    affinity_or_group = (
        group_tag_str
        if group_tag_str
        else (surface_affinity if _is_gradient_method(method) else None)
    )

    return (
        f"{str(overview_scope).strip() or 'Group'} Overview | {atlas} | {method}"
        + (f" | {route_tag}" if route_tag else "")
        + (f" | {mode_tag}" if mode_tag else "")
        + (f" | {assignment_variant}" if assignment_variant else "")
        + (f" | {affinity_or_group}" if affinity_or_group else "")
    )


def resolve_render_assignment_variants(
    method: str,
    assignment_variant: str,
    *,
    atlas: str | None = None,
) -> tuple[str | None, ...]:
    if method in CLUSTERLESS_METHODS:
        is_lynch = atlas is not None and str(atlas).startswith("lynch2024_")
        if atlas is not None and str(atlas) != "kong2022" and not is_lynch:
            raise ValueError(
                f"WTA is not supported for atlas={atlas!r}"
            )
        if is_lynch:
            if assignment_variant in {"both", "fisherz"}:
                return ("fisherz",)
            raise ValueError(
                "Lynch WTA exposes the direct-network Fisher-z assignment only; "
                "normalized_top20_count is a Kong2022 ROI-registry variant."
            )
        if assignment_variant == "both":
            return ("normalized_top20_count",)
        if assignment_variant != "normalized_top20_count":
            raise ValueError(
                "Group overview WTA supports only assignment_variant='normalized_top20_count'; "
                f"got {assignment_variant!r}"
            )
        return ("normalized_top20_count",)
    if str(atlas) in {"lynch2024_20net", "lynch2024_7net"} and method == "gradient-spectral":
        if assignment_variant in {"both", "fisherz"}:
            return ("fisherz",)
        raise ValueError(
            "Lynch gradient-spectral overviews expose the registered fisherz "
            "cluster assignment only; normalized_top20_count is a Kong2022 "
            "ROI-registry variant."
        )
    if assignment_variant == "both":
        return DEFAULT_RENDER_ASSIGNMENT_VARIANTS
    if assignment_variant in ASSIGNMENT_VARIANTS:
        return (assignment_variant,)
    raise ValueError(f"Unknown assignment variant: {assignment_variant!r}")


def default_render_assignment_variant(method: str, atlas: str) -> str:
    """Return the method-specific variant used when a direct call omits one."""
    if method in CLUSTERLESS_METHODS:
        if str(atlas) == "kong2022":
            return "normalized_top20_count"
        if str(atlas).startswith("lynch2024_"):
            return "fisherz"
        raise ValueError(f"WTA is not supported for atlas={atlas!r}")
    return "fisherz"


def effective_overview_mode_tag(method: str, mode_tag: str | None, assignment_variant: str | None) -> str | None:
    if method in CLUSTERLESS_METHODS:
        return None
    return mode_tag


def resolve_overview_mode_tag(group_root: Path, explicit_mode_tag: str | None) -> str | None:
    # All maintained formal runs are full computations; there is no sparse
    # overview mode in production. Only honor an explicit caller override
    # (e.g. side-by-side comparisons). Never infer "sparse" from the
    # directory name: formal roots such as work/group_atlas/routea_ctxdrop
    # omit the legacy _full_k3_fisherz suffix by contract yet remain full.
    if explicit_mode_tag is not None:
        text = str(explicit_mode_tag).strip()
        return text if text else None
    return None


def validate_method_route_compatibility(method: str, route_tag: str | None) -> None:
    if str(method) == "wta-top" and str(route_tag) == "routea_17net":
        raise ValueError(
            "WTA is not valid for the routea_17net/family17 gradient-only branch; "
            "use a canonical7 Kong2022 route."
        )


def _is_gradient_method(method: str) -> bool:
    return method == GRADIENT_METHOD


def _discover_gradient_comp_tags(
    group_root: Path,
    method: str,
    atlas: str,
    smoothing: str,
    coverage_tag: str,
) -> list[str]:
    """Discover gradient component subdirectories under the coverage directory."""
    if not _is_gradient_method(method):
        return [""]
    base = group_root / method / atlas / smoothing / coverage_tag
    if not base.exists():
        return [""]
    tags = sorted(d.name for d in base.iterdir() if d.is_dir() and d.name.startswith("g"))
    return tags if tags else [""]


def _build_run_dir(
    group_root: Path,
    method: str,
    atlas: str,
    smoothing: str,
    coverage_tag: str,
    hemi: str,
    *,
    gradient_comp_tag: str = "",
) -> Path:
    """Build the run directory path, including gradient component subdir for gradient-spectral."""
    if _is_gradient_method(method) and gradient_comp_tag:
        return group_root / method / atlas / smoothing / coverage_tag / gradient_comp_tag / f"hemi_{hemi}"
    return group_root / method / atlas / smoothing / coverage_tag / f"hemi_{hemi}"


def load_network_bar_colors() -> dict[str, str]:
    return load_network_hex_colors()


NETWORK_BAR_COLORS = load_network_bar_colors()




def resolve_existing_manifest_path(raw: object, *, repo_root: Path = REPO_ROOT) -> Path:
    path = Path(str(raw)).expanduser()
    base = Path(repo_root).resolve()
    path_abs = path if path.is_absolute() else base / path
    return path_abs.resolve()


def resolve_assignment_variant_paths(
    run_dir: Path,
    manifest: dict,
    variant: str,
) -> tuple[Path, Path]:
    """Resolve the exact status/GIFTI pair declared for ``variant``.

    A missing declaration is an error. This function deliberately does not
    inspect a generic status file, choose a first existing candidate, or use
    a legacy alias because those assets can carry another assignment semantic.
    """
    if variant not in ASSIGNMENT_VARIANTS:
        raise ValueError(f"Unknown assignment variant: {variant}")
    run_dir = Path(run_dir).resolve()
    variants = manifest.get("assignment_variants")
    row = variants.get(variant) if isinstance(variants, dict) else None
    if not isinstance(row, dict):
        raise ValueError(
            f"Missing exact assignment_variants.{variant} in manifest: "
            f"{run_dir / 'group_atlas_manifest.json'}"
        )
    required = ("cluster_assignment_status", "template_functional_atlas_label")
    resolved: dict[str, Path] = {}
    for field in required:
        raw = row.get(field)
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(
                f"Missing exact assignment_variants.{variant}.{field}: "
                f"{run_dir / 'group_atlas_manifest.json'}"
            )
        path = Path(raw).expanduser()
        if not path.is_absolute():
            raise ValueError(
                f"Exact assignment_variants.{variant}.{field} must be absolute: {path}"
            )
        path = path.resolve()
        if not _is_path_within(path, run_dir):
            raise ValueError(
                f"Exact assignment_variants.{variant}.{field} points outside run directory: {path}"
            )
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing exact assignment_variants.{variant}.{field}: {path}"
            )
        resolved[field] = path
    # A deprecated alias is metadata only; the exact declared label path above
    # is the sole render input.
    return resolved["cluster_assignment_status"], resolved["template_functional_atlas_label"]


MATERIALIZE_COLOR_MAP = network_color_map()


def _is_path_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _load_label_vector(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    return np.asarray(img.darrays[0].data).astype(np.int32, copy=False)


def load_default_functional_labels(run_dir: Path, manifest: dict) -> np.ndarray:
    labels_path, _map_path = resolve_functional_atlas_array_paths(manifest, run_dir)
    return np.load(labels_path).astype(np.int32, copy=False)


def load_default_functional_label_map(run_dir: Path, manifest: dict) -> dict:
    _labels_path, map_path = resolve_functional_atlas_array_paths(manifest, run_dir)
    return load_json(map_path)


def ensure_functional_label_consistency(manifest_path: Path, manifest: dict) -> dict:
    run_dir = manifest_path.parent
    raw_label = manifest.get("template_functional_atlas_label", None)
    label_path = Path(str(raw_label)).resolve() if raw_label is not None else None

    need_refresh = False
    reasons: list[str] = []
    try:
        expected_npy, _map_path = resolve_functional_atlas_array_paths(manifest, run_dir)
    except ValueError as exc:
        expected_npy = None
        need_refresh = True
        reasons.append(f"manifest_default_variant_error:{type(exc).__name__}")

    if label_path is None:
        need_refresh = True
        reasons.append("missing_template_functional_atlas_label")
    elif not label_path.exists():
        need_refresh = True
        reasons.append("missing_functional_label_gii")
    elif not _is_path_within(label_path, run_dir):
        need_refresh = True
        reasons.append("functional_label_path_outside_run_dir")
    if expected_npy is None or not expected_npy.exists():
        need_refresh = True
        reasons.append("missing_default_functional_atlas_labels_npy")

    if not need_refresh and label_path is not None and expected_npy is not None:
        try:
            arr_npy = np.load(expected_npy).astype(np.int32, copy=False)
            arr_gii = _load_label_vector(label_path)
            if arr_npy.shape != arr_gii.shape or not np.array_equal(arr_npy, arr_gii):
                need_refresh = True
                reasons.append("npy_gii_mismatch")
        except Exception as exc:  # pragma: no cover
            need_refresh = True
            reasons.append(f"read_error:{type(exc).__name__}")

    if not need_refresh:
        return manifest

    updated = materialize_combo(manifest_path, MATERIALIZE_COLOR_MAP)
    if not updated:
        raise RuntimeError(
            f"Functional atlas label inconsistency detected but materialization was skipped: {manifest_path} reasons={",".join(reasons)}"
        )
    refreshed = load_json(manifest_path)
    refreshed_label_raw = refreshed.get("template_functional_atlas_label", None)
    if refreshed_label_raw is None:
        raise RuntimeError(f"materialize_combo did not write template_functional_atlas_label: {manifest_path}")
    refreshed_label = Path(str(refreshed_label_raw)).resolve()
    if not refreshed_label.exists():
        raise RuntimeError(f"materialize_combo wrote missing label file: {refreshed_label}")
    if not _is_path_within(refreshed_label, run_dir):
        raise RuntimeError(f"materialize_combo wrote label outside run dir: {refreshed_label} (run_dir={run_dir})")
    refreshed_npy, _refreshed_map = resolve_functional_atlas_array_paths(refreshed, run_dir)
    if not refreshed_npy.exists():
        raise RuntimeError(f"materialize_combo did not write default npy: {refreshed_npy}")
    arr_npy = np.load(refreshed_npy).astype(np.int32, copy=False)
    arr_gii = _load_label_vector(refreshed_label)
    if arr_npy.shape != arr_gii.shape or not np.array_equal(arr_npy, arr_gii):
        raise RuntimeError(f"materialize_combo still produced mismatched npy/gii labels: {manifest_path}")
    return refreshed


def ensure_gradient_assignment_variants(manifest_path: Path, manifest: dict) -> dict:
    """Validate every declared gradient assignment before rendering.

    Group-overview rendering is read-only with respect to assignment assets.
    A malformed, stale, or semantically mismatched variant must be repaired by
    the explicit materialization stage before this renderer is called.
    """
    variants = manifest.get("assignment_variants")
    if not isinstance(variants, dict) or not variants:
        raise ValueError(
            f"Gradient overview requires explicit assignment_variants: {manifest_path}"
        )
    for variant in sorted(variants):
        validate_gradient_assignment(manifest_path.parent, manifest, str(variant))
    return manifest


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def resolve_template_pair_assets(
    left_manifest: dict,
    right_manifest: dict,
    *,
    left_run_dir: Path,
    right_run_dir: Path,
) -> tuple[Path, Path, str | None, Path | None]:
    surface_l, subject_l, spec_l = validate_template_assets(
        left_manifest,
        hemi="L",
        run_dir=left_run_dir,
    )
    surface_r, subject_r, spec_r = validate_template_assets(
        right_manifest,
        hemi="R",
        run_dir=right_run_dir,
    )
    if (subject_l, spec_l) != (subject_r, spec_r):
        raise RuntimeError(
            "Left/right template render assets disagree: "
            f"subject={subject_l!r}/{subject_r!r}, "
            f"spec={spec_l.name if spec_l else None}/{spec_r.name if spec_r else None}"
        )
    return surface_l, surface_r, subject_l, spec_l


def resolve_template_render_spec(
    manifest: dict,
    *,
    run_dir: Path | None = None,
) -> tuple[str | None, Path | None]:
    _, subject, spec = validate_template_assets(
        manifest,
        hemi=str(manifest.get("hemi", "L")),
        run_dir=run_dir,
    )
    return subject, spec


def format_panel_error(label: str, exc: Exception, *, width: int = 36, max_detail_lines: int = 2) -> str:
    detail = " ".join(str(exc).split())
    if not detail:
        detail = type(exc).__name__
    lines = textwrap.wrap(
        f"{type(exc).__name__}: {detail}",
        width=width,
        break_long_words=True,
        break_on_hyphens=False,
    )
    if len(lines) > max_detail_lines:
        lines = lines[: max_detail_lines - 1] + ["..."]
    return "\n".join([label, *lines])


def style_imshow(ax, data: np.ndarray, title: str, *, vmin: float | None = None, vmax: float | None = None, cmap: str = "magma"):
    arr = np.asarray(data, dtype=np.float32)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("#f2f2f2")
    im = ax.imshow(arr, aspect="auto", interpolation="nearest", cmap=cm, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=FS_TITLE)
    return im


def add_cbar(fig, ax, im, label: str, ticks: list[float] | None = None, ticklabels: list[str] | None = None) -> None:
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="3%", pad=0.06)
    cb = fig.colorbar(im, cax=cax, ticks=ticks)
    if ticklabels is not None:
        cb.set_ticklabels(ticklabels)
    cb.ax.tick_params(labelsize=FS_TICK)
    cb.set_label(label, fontsize=FS_LABEL)


def concat_lr(a: np.ndarray, la: list[str], b: np.ndarray, lb: list[str]) -> tuple[np.ndarray, list[int], list[str]]:
    sep = np.full((1, a.shape[1]), np.nan, dtype=np.float32)
    mat = np.vstack([a, sep, b]).astype(np.float32)
    yt = list(range(a.shape[0])) + list(range(a.shape[0] + 1, a.shape[0] + 1 + b.shape[0]))
    yl = la + lb
    return mat, yt, yl


def ordered_cluster_ids_by_position(labels: np.ndarray, order: np.ndarray, k: int) -> list[int]:
    x = labels[order]
    rows: list[tuple[float, int]] = []
    for cid in range(1, int(k) + 1):
        idx = np.flatnonzero(x == cid)
        pos = float(np.mean(idx)) if idx.size else float("inf")
        rows.append((pos, int(cid)))
    rows.sort(key=lambda t: (t[0], t[1]))
    return [cid for _, cid in rows]


def resolve_cluster_order_from_manifest(manifest: dict, labels: np.ndarray, order: np.ndarray, k: int) -> list[int]:
    fixed = manifest.get("cluster_order_head_to_tail")
    if isinstance(fixed, list):
        vals = [int(x) for x in fixed]
        expect = set(range(1, int(k) + 1))
        if set(vals) == expect and len(vals) == int(k):
            return vals
    return ordered_cluster_ids_by_position(labels, order, k)


def labels_to_indicator_by_ids(labels: np.ndarray, order: np.ndarray, cluster_ids: list[int]) -> np.ndarray:
    out = np.zeros((len(cluster_ids), int(order.size)), dtype=np.float32)
    x = labels[order]
    for i, cid in enumerate(cluster_ids):
        out[i, :] = (x == int(cid)).astype(np.float32)
    return out


def reorder_profile(profile: np.ndarray, cluster_ids: list[int]) -> np.ndarray:
    idx = [int(cid) - 1 for cid in cluster_ids]
    return profile[np.asarray(idx, dtype=np.int32), :].astype(np.float32)


def annotate_profile_values(
    ax,
    data: np.ndarray,
    vmax: float,
    annotation_mask: np.ndarray | None = None,
) -> None:
    if vmax <= 0:
        return
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            if annotation_mask is not None and not bool(annotation_mask[r, c]):
                continue
            val = float(data[r, c])
            if not np.isfinite(val):
                continue
            image = ax.images[-1]
            rgb = np.asarray(image.cmap(image.norm(val))[:3])
            linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
            luminance = float(linear @ np.array([0.2126, 0.7152, 0.0722]))
            color = "white" if luminance < 0.179 else "#222222"
            ax.text(c, r, f"{abs(val):.2f}", ha="center", va="center", fontsize=FS_CELL, color=color)


def highlight_top1_profile_cells(ax, data: np.ndarray) -> None:
    for r in range(data.shape[0]):
        row = data[r, :]
        if not np.isfinite(row).any():
            continue
        c = int(np.nanargmax(row))
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1.0, 1.0, fill=False, edgecolor="#111111", linewidth=1.1))


def _k_failure_lines(rows_tsv: list[dict[str, str]]) -> list[str]:
    rows = sorted(rows_tsv, key=lambda r: int(r["k"]))
    lines: list[str] = []
    for row in rows:
        if row.get("selected_final", "0") == "1":
            continue
        reasons: list[str] = []
        if row.get("reject_redundant", "0") == "1":
            reasons.append("redundant")
        if row.get("reject_nontrivial_fail", "0") == "1":
            reasons.append("nontrivial_fail")
        if reasons:
            lines.append(f"  K={int(row['k'])}: {','.join(reasons)}")
    if not lines and rows:
        lines.append("  No explicit rejection reasons found")
    return lines


def confidence_rows(status_payload: dict, hemi: str, cluster_ids: list[int]) -> list[dict[str, object]]:
    by_cluster = {int(row["cluster_id"]): row for row in status_payload.get("items", [])}
    top_n_cap_payload = status_payload.get("top_n_cap", None)
    out: list[dict[str, object]] = []
    for display_id, old_id in enumerate(cluster_ids, start=1):
        if int(old_id) not in by_cluster:
            continue
        row = by_cluster[int(old_id)]
        status = str(row.get("status", "dominant"))
        if "top1_normalized_count" in row and "top2_normalized_count" in row:
            top1_val = float(row.get("top1_normalized_count", 0.0))
            top2_val = float(row.get("top2_normalized_count", 0.0))
            margin = float(row.get("margin_normalized_count", top1_val - top2_val))
            top1_network = str(row.get("top1_network", "unassigned"))
            top2_network = str(row.get("top2_network", "none"))
            n_top = int(row.get("n_top", 0))
            if top2_network == top1_network and (top2_val <= 0.0 or top2_val >= top1_val):
                top2_network = "none"
                top2_val = 0.0
                margin = float(top1_val)
            if top2_network in {"", "unassigned"} and top2_val <= 0.0:
                top2_network = "none"
            low_margin = bool((top2_network != "none") and (margin < 0.05))
            metric = "normalized_count"
        elif "top1_count" in row or "top2_count" in row:
            raise RuntimeError(
                "Raw count assignment status is not supported by core_workflow; "
                "regenerate the group atlas with normalized_top20_count"
            )
        else:
            top1_val = float(row.get("top1_fisher_z", 0.0))
            top2_val = float(row.get("top2_fisher_z", 0.0))
            margin = float(row.get("margin", top1_val - top2_val))
            low_margin = bool(int(row.get("low_margin_lt_0p05", 1 if margin < 0.05 else 0)))
            top1_network = str(row.get("top1_network", "unassigned"))
            top2_network = str(row.get("top2_network", "unassigned"))
            n_top = None
            metric = "fisher"
        out.append(
            {
                "label": f"{hemi} C{int(display_id)}",
                "status": status,
                "top1_network": top1_network,
                "top2_network": top2_network,
                "top1_value": top1_val,
                "top2_value": top2_val,
                "margin": margin,
                "n_top": n_top,
                "top_n_cap": int(top_n_cap_payload) if top_n_cap_payload is not None else None,
                "metric": metric,
                "ambiguous": bool(status != "dominant" or low_margin),
            }
        )
    return out


def validate_fisher_status_matches_profile(
    *,
    profile_ordered: np.ndarray,
    status_payload: dict,
    network_names: list[str],
    cluster_ids: list[int],
    context: str,
) -> None:
    by_cluster = {int(row["cluster_id"]): row for row in status_payload.get("items", []) if "cluster_id" in row}
    for display_idx, old_id in enumerate(cluster_ids):
        row = by_cluster.get(int(old_id))
        if row is None or "top1_fisher_z" not in row:
            continue
        prof_row = np.asarray(profile_ordered[int(display_idx), :], dtype=np.float32)
        if not np.isfinite(prof_row).any():
            continue
        top_idx = int(np.nanargmax(prof_row))
        expected_network = str(network_names[top_idx])
        got_network = str(row.get("top1_network", ""))
        if got_network != expected_network:
            raise RuntimeError(
                f"Profile/status network-order mismatch in {context} cluster_id={int(old_id)}: "
                f"profile_top={expected_network} status_top={got_network}"
            )
        if "top1_fisher_z" in row:
            got_value = float(row.get("top1_fisher_z", 0.0))
            expected_value = float(prof_row[top_idx])
            if abs(got_value - expected_value) > 1e-4:
                raise RuntimeError(
                    f"Profile/status Fisher-z mismatch in {context} cluster_id={int(old_id)}: "
                    f"profile_top_z={expected_value:.6g} status_top_z={got_value:.6g}"
                )


def validate_normalized_count_status_matches_profile(
    *,
    profile_ordered: np.ndarray,
    status_payload: dict,
    network_names: list[str],
    cluster_ids: list[int],
    context: str,
) -> None:
    by_cluster = {int(row["cluster_id"]): row for row in status_payload.get("items", []) if "cluster_id" in row}
    for display_idx, old_id in enumerate(cluster_ids):
        row = by_cluster.get(int(old_id))
        if row is None or "top1_normalized_count" not in row:
            continue
        prof_row = np.asarray(profile_ordered[int(display_idx), :], dtype=np.float32)
        if not np.isfinite(prof_row).any():
            continue
        top1_network = str(row.get("top1_network", ""))
        if top1_network in {"", "none", "unassigned"}:
            continue
        if top1_network not in network_names:
            raise RuntimeError(
                f"Normalized count profile/status network mismatch in {context} cluster_id={int(old_id)}: "
                f"status_top={top1_network} not in profile network names"
            )
        top_idx = int(network_names.index(top1_network))
        expected_value = float(prof_row[top_idx])
        got_value = float(row.get("top1_normalized_count", 0.0))
        max_value = float(np.nanmax(prof_row)) if prof_row.size else 0.0
        if abs(got_value - expected_value) > 1e-4 or got_value < max_value - 1e-4:
            raise RuntimeError(
                f"Normalized count profile/status mismatch in {context} cluster_id={int(old_id)}: "
                f"profile_top={max_value:.6g} status_top={got_value:.6g}"
            )


def normalized_count_profile_from_status(
    status_payload: dict,
    *,
    network_names: list[str],
    cluster_ids: list[int],
) -> np.ndarray:
    by_cluster = {
        int(row["cluster_id"]): row
        for row in status_payload.get("items", [])
        if "cluster_id" in row
    }
    network_index = {str(name): index for index, name in enumerate(network_names)}
    profile = np.zeros((len(cluster_ids), len(network_names)), dtype=np.float32)
    for display_index, cluster_id in enumerate(cluster_ids):
        row = by_cluster.get(int(cluster_id), {})
        raw_values = row.get("normalized_count_by_network", {})
        if not isinstance(raw_values, dict):
            raise RuntimeError(
                "Normalized Top-20 status row is missing normalized_count_by_network: "
                f"cluster_id={cluster_id}"
            )
        for network, value in raw_values.items():
            name = str(network)
            if name not in network_index:
                raise RuntimeError(
                    "Normalized Top-20 status network is not represented in network_names: "
                    f"cluster_id={cluster_id} network={name}"
                )
            profile[display_index, network_index[name]] = float(value)
    return profile


def plot_confidence_panel(ax, rows: list[dict[str, object]], title: str) -> None:
    if not rows:
        ax.axis("off")
        ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=FS_MISSING)
        return
    metric = str(rows[0].get("metric", "fisher"))
    y = np.arange(len(rows), dtype=float)
    top1 = np.array([float(r["top1_value"]) for r in rows], dtype=float)
    top2 = np.array([float(r["top2_value"]) for r in rows], dtype=float)
    colors = []
    for r in rows:
        net = str(r.get("top1_network", "unassigned"))
        if net in {"", "none", "unassigned"}:
            colors.append("#9a9a9a")
        else:
            if net not in NETWORK_BAR_COLORS:
                raise KeyError(f"Missing network color for {net!r}")
            colors.append(str(NETWORK_BAR_COLORS[net]))
    labels = []
    for r in rows:
        top1_name = str(r.get("top1_network", "unassigned"))
        top2_name = str(r.get("top2_network", ""))
        if top2_name in {"", "none", "unassigned"}:
            labels.append(f"{r['label']} {top1_name}")
        else:
            labels.append(f"{r['label']} {top1_name} | {top2_name}")
    ax.barh(y, top1, height=0.62, color=colors, alpha=0.88, label="1st")
    ax.barh(y, top2, height=0.32, color="none", edgecolor="#111111", linewidth=0.9, label="2nd")
    is_normalized_count = metric == "normalized_count"
    if metric not in {"normalized_count", "fisher"}:
        raise RuntimeError(f"Unsupported assignment confidence metric: {metric!r}")
    for yi, row in zip(y, rows, strict=False):
        ax.text(
            max(float(row["top1_value"]), float(row["top2_value"])) + 0.005,
            yi,
            (
                f"Δn_norm={float(row['margin']):.3f}"
                if is_normalized_count
                else f"Δz={float(row['margin']):.3f}"
            ),
            va="center",
            ha="left",
            fontsize=FS_TICK,
            color="#c97c00" if bool(row["ambiguous"]) else "#333333",
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=FS_TICK)
    ax.invert_yaxis()
    xmax = max(0.12, float(np.max(top1)) * 1.22)
    ax.set_xlim(0.0, xmax)
    ax.set_xlabel("Normalized TopN ROI count" if is_normalized_count else "Fisher-z", fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    ax.axvline(0.05, color="#999999", lw=0.8, ls=":")
    ax.grid(axis="x", color="#dddddd", lw=0.5, alpha=0.7)
    ax.legend(fontsize=FS_TICK, loc="lower right")


def plot_vertex_assignment_counts(
    ax,
    rows: list[dict[str, object]],
    title: str,
    *,
    title_fs: float = FS_TITLE,
    label_fs: float = FS_LABEL,
    tick_fs: float = FS_TICK,
) -> None:
    if not rows:
        ax.axis("off")
        ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=FS_MISSING)
        return
    counts: dict[str, int] = {}
    for row in rows:
        net = str(row.get("top1_network", "unassigned")) if str(row.get("status", "unassigned")) == "dominant" else "unassigned"
        counts[net] = int(counts.get(net, 0) + 1)
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    names = [k for k, _ in ordered][:10]
    vals = [v for _, v in ordered][:10]
    y = np.arange(len(names))
    ax.barh(y, vals, color="#2c7fb8", alpha=0.88)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=tick_fs)
    ax.invert_yaxis()
    ax.set_xlabel("Vertices", fontsize=label_fs)
    ax.set_title(title, fontsize=title_fs)
    ax.tick_params(labelsize=tick_fs)
    ax.grid(axis="x", color="#dddddd", lw=0.5, alpha=0.7)


def plot_vertex_confidence(ax, rows: list[dict[str, object]], title: str) -> None:
    if not rows:
        ax.axis("off")
        ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=FS_MISSING)
        return
    if "margin_count" in rows[0]:
        margins = np.array([float(r.get("margin_count", 0.0)) for r in rows], dtype=float)
        ax.hist(margins, bins=min(12, max(4, int(np.nanmax(margins) + 1))), color="#2c7fb8", alpha=0.85, edgecolor="#222222")
        ax.set_xlabel("TopN count margin (1st-2nd)", fontsize=FS_LABEL)
    else:
        margins = np.array([float(r.get("margin_score", 0.0)) for r in rows], dtype=float)
        ax.hist(margins, bins=12, color="#2c7fb8", alpha=0.85, edgecolor="#222222")
        ax.set_xlabel("Network mean-score margin (1st-2nd)", fontsize=FS_LABEL)
    ax.set_ylabel("Vertices", fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(axis="y", color="#dddddd", lw=0.5, alpha=0.7)


def _metric_array(rows: list[dict[str, str]], key: str) -> np.ndarray:
    vals: list[float] = []
    for row in rows:
        raw = row.get(key, "")
        try:
            vals.append(float(raw))
        except (TypeError, ValueError):
            vals.append(float("nan"))
    return np.asarray(vals, dtype=float)


def plot_ari_summary_curve(ax, rows: list[dict[str, str]], *, hemi: str, color: str) -> bool:
    rows = sorted(rows, key=lambda r: int(r["k"]))
    ks = np.array([int(r["k"]) for r in rows], dtype=int)
    mean = _metric_array(rows, "group_ari_mean")
    se = _metric_array(rows, "group_ari_se")
    finite = np.isfinite(mean)
    if bool(finite.any()):
        ax.plot(ks, mean, marker="o", lw=1.3, color=color, label=hemi)
        ax.fill_between(ks, mean - se, mean + se, color=color, alpha=0.18)
    null_ref = _metric_array(rows, "ARI_null_5pct")
    if bool(np.isfinite(null_ref).any()):
        ax.plot(ks, null_ref, ls=":", lw=1.0, color=color, alpha=0.85)
    return bool(finite.any())


def annotate_missing_ari_curves(ax, missing_hemis: list[str]) -> None:
    if not missing_hemis:
        return
    ax.text(
        0.03,
        0.04,
        "Subject-split ARI unavailable: hemi-" + "/".join(missing_hemis),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=FS_SMALL,
        color="#7a2e00",
        bbox={"facecolor": "#fff7e6", "edgecolor": "#d08b00", "linewidth": 0.6, "alpha": 0.9},
    )


def plot_subject_split_ari_skipped(ax, *, smoothing: str) -> None:
    """Keep the ARI panel scaffold while omitting non-estimable curves.

    A single subject cannot support a subject-split ARI estimate.  The panel
    therefore retains the same limits, labels, and footprint as the group
    overview, but deliberately contains no curves or reference lines.
    """
    # Individual overviews use the fixed K=2..8 evaluation range.  Keeping
    # those limits preserves the original 2x2 layout instead of collapsing the
    # panel when the subject-split statistic is skipped.
    ax.set_xlim(1.5, 8.5)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(np.arange(2, 9, dtype=int))
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.set_xlabel("K", fontsize=FS_LABEL)
    ax.set_ylabel("Subject-split ARI", fontsize=FS_LABEL)
    ax.set_title(f"{smoothing} | K selection (ARI skipped)", fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    ax.text(
        0.03,
        0.04,
        "ARI skipped (single subject)",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=FS_SMALL,
        color="#666666",
    )


def plot_k_gate_panel(
    ax,
    *,
    rows_l: list[dict[str, str]],
    rows_r: list[dict[str, str]],
    k_l: int,
    k_r: int,
    smoothing: str,
) -> None:
    rows_l = sorted(rows_l, key=lambda r: int(r["k"]))
    rows_r = sorted(rows_r, key=lambda r: int(r["k"]))
    ks_l = np.asarray([int(r["k"]) for r in rows_l], dtype=int)
    ks_r = np.asarray([int(r["k"]) for r in rows_r], dtype=int)
    max_sim_l = _metric_array(rows_l, "max_profile_similarity")
    max_sim_r = _metric_array(rows_r, "max_profile_similarity")

    has_metric = bool(np.isfinite(max_sim_l).any() or np.isfinite(max_sim_r).any())
    if not has_metric:
        ax.axis("off")
        ax.text(0.5, 0.5, f"{smoothing} metrics unavailable", ha="center", va="center", fontsize=FS_MISSING)
        return

    ax.plot(ks_l, max_sim_l, color="#1f77b4", lw=1.3, marker="o")
    ax.plot(ks_r, max_sim_r, color="#d62728", lw=1.3, marker="o")
    ax.axhline(0.80, color="#b00020", lw=1.0, ls=":", alpha=0.9)
    ax.axvline(int(k_l), color="#1f77b4", ls="--", lw=1.0)
    ax.axvline(int(k_r), color="#d62728", ls="--", lw=1.0)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("K", fontsize=FS_LABEL)
    ax.set_ylabel("max_profile_similarity", fontsize=FS_LABEL)
    ax.set_title(f"{smoothing} | redundancy gate", fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(axis="y", color="#dddddd", lw=0.5, alpha=0.7)
    ax.legend(
        handles=[
            Line2D([0], [0], color="#1f77b4", lw=1.6, marker="o", label="Left hemisphere"),
            Line2D([0], [0], color="#d62728", lw=1.6, marker="o", label="Right hemisphere"),
            Line2D([0], [0], color="#b00020", lw=1.2, ls=":", label="Redundancy threshold"),
            Line2D([0], [0], color="#555555", lw=1.2, ls="--", label="Selected K*"),
        ],
        loc="center",
        fontsize=FS_LEGEND_SMALL,
        framealpha=0.9,
        edgecolor="#dddddd",
        borderpad=0.3,
        labelspacing=0.28,
        handlelength=2.1,
    )


def _require_canonical7_network(row: dict[str, object]) -> str:
    network = str(row.get("canonical7", "")).strip()
    if not network:
        raise ValueError(
            "Top-connection row is missing canonical7 network; "
            "canonical8 fallback is retired"
        )
    return network


def plot_global_top_connections(ax, payload: dict[str, object], title: str) -> None:
    pos = [dict(x) for x in payload.get("top_positive", [])]
    neg = [dict(x) for x in payload.get("top_negative", [])]
    ax.axis("off")
    lines = [title, ""]
    lines.append(f"Top+ edges: {len(pos)}")
    lines.append(f"Top- edges: {len(neg)}")
    lines.append("")
    lines.append("Top+ preview:")
    for row in pos[:5]:
        network_name = _require_canonical7_network(row)
        lines.append(f"  v{int(row['vertex_index'])} -> {row['roi_name']} ({network_name}) z={float(row['fisher_z']):+.2f}")
    lines.append("")
    lines.append("Top- preview:")
    for row in neg[:5]:
        network_name = _require_canonical7_network(row)
        lines.append(f"  v{int(row['vertex_index'])} -> {row['roi_name']} ({network_name}) z={float(row['fisher_z']):+.2f}")
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=FS_SMALL, family="monospace")



def compute_wta_overview_profile_rows(
    *,
    labels_by_hemi: dict[str, np.ndarray],
    order_by_hemi: dict[str, np.ndarray],
    network_names: list[str],
    network_label_keys: dict[str, int],
    n_bins: int = WTA_OVERVIEW_PROFILE_BINS,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    network_keys = {int(network_label_keys[name]) for name in network_names}
    for hemi in sorted(labels_by_hemi):
        labels = np.asarray(labels_by_hemi[hemi]).reshape(-1)
        order = np.asarray(order_by_hemi[hemi], dtype=np.int64).reshape(-1)
        if labels.size != order.size:
            raise ValueError(f"Label/order size mismatch for hemi-{hemi}: labels={labels.size} order={order.size}")
        bins = equal_vertex_long_axis_bins(order, n_bins)
        for bin_idx, vertices in enumerate(bins, start=1):
            bin_labels = labels[vertices] if vertices.size else np.asarray([], dtype=labels.dtype)
            denom = int(bin_labels.size)
            for network_name in network_names:
                key = int(network_label_keys[network_name])
                count = int(np.sum(bin_labels == key)) if denom else 0
                pct = (100.0 * float(count) / float(denom)) if denom else 0.0
                rows.append(
                    {
                        "hemi": str(hemi),
                        "bin_index": int(bin_idx),
                        "network_name": str(network_name),
                        "percent_of_bin_vertices": float(pct),
                    }
                )
            non_network_count = int(np.sum(~np.isin(bin_labels, list(network_keys)))) if denom else 0
            if non_network_count > 0:
                rows.append(
                    {
                        "hemi": str(hemi),
                        "bin_index": int(bin_idx),
                        "network_name": WTA_NON_NETWORK_NAME,
                        "percent_of_bin_vertices": 100.0 * float(non_network_count) / float(denom),
                    }
                )
    return rows


def plot_wta_overview_profile_axis(
    ax,
    *,
    rows: list[dict[str, object]],
    network_names: list[str],
    hemi: str,
    title: str,
    show_ylabel: bool,
) -> None:
    hemi_rows = [r for r in rows if str(r["hemi"]) == str(hemi)]
    if not hemi_rows:
        ax.axis("off")
        ax.text(0.5, 0.5, "profile missing", ha="center", va="center", fontsize=FS_MISSING)
        return
    max_bin = max(int(r["bin_index"]) for r in hemi_rows)
    plotted_names = list(network_names)
    if any(str(r["network_name"]) == WTA_NON_NETWORK_NAME for r in hemi_rows):
        plotted_names.append(WTA_NON_NETWORK_NAME)
    for network_name in plotted_names:
        pts = sorted(
            [r for r in hemi_rows if str(r["network_name"]) == network_name],
            key=lambda r: int(r["bin_index"]),
        )
        if not pts:
            continue
        xs = [float(r["percent_of_bin_vertices"]) for r in pts]
        ys = [int(r["bin_index"]) for r in pts]
        if network_name == WTA_NON_NETWORK_NAME:
            ax.plot(xs, ys, color="#777777", lw=0.9, ls="--", marker="o", markersize=1.9, label="non-network/null")
        else:
            ax.plot(
                xs,
                ys,
                color=NETWORK_BAR_COLORS[network_name],
                lw=1.15,
                marker="o",
                markersize=2.1,
                label=network_name,
            )
    ax.set_title(title, fontsize=WTA_OVERVIEW_SIDE_PANEL_TITLE_FS)
    ax.set_xlabel("Vertices in bin (%)", fontsize=WTA_OVERVIEW_SIDE_PANEL_LABEL_FS)
    ax.set_ylabel("Long-axis bin" if show_ylabel else "", fontsize=WTA_OVERVIEW_SIDE_PANEL_LABEL_FS)
    ax.set_ylim(max_bin + 0.5, 0.5)
    ax.set_xlim(left=0.0)
    ax.grid(True, axis="x", color="#dddddd", linewidth=0.5, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=WTA_OVERVIEW_SIDE_PANEL_TICK_FS)
    if show_ylabel:
        step = 5 if max_bin >= 20 else max(1, max_bin // 4)
        ticks = list(range(1, max_bin + 1, step))
        if ticks[-1] != max_bin:
            ticks.append(max_bin)
        ax.set_yticks(ticks)
    else:
        ax.set_yticklabels([])
    if not show_ylabel:
        ax.legend(fontsize=WTA_OVERVIEW_SIDE_PANEL_LEGEND_FS, frameon=False, loc="lower right", handlelength=1.4, labelspacing=0.22)


def overview_png_path(
    *,
    present_dir: Path,
    atlas: str,
    method: str,
    route_tag: str | None,
    mode_tag: str | None,
    assignment_variant: str | None,
    gradient_comp_tag: str = "",
    output_name: str | None = None,
) -> Path:
    if output_name is not None:
        name = str(output_name).strip()
        if not name:
            raise ValueError("--output-name must not be empty")
        out_name = Path(name)
        if out_name.name != name or out_name.is_absolute():
            raise ValueError("--output-name must be a filename, not a path")
        if out_name.suffix == "":
            out_name = out_name.with_suffix(".png")
        return Path(present_dir) / out_name

    name_bits = [atlas, method, "group_overview"]
    if gradient_comp_tag:
        name_bits.append(gradient_comp_tag)
    if route_tag:
        name_bits.append(route_tag)
    if mode_tag:
        name_bits.append(mode_tag)
    if assignment_variant:
        name_bits.append(assignment_variant)
    return Path(present_dir) / f"{'_'.join(name_bits)}.png"

def render_combo(
    *,
    group_root: Path,
    present_dir: Path,
    method: str,
    atlas: str,
    coverage_tag: str,
    scene: Path,
    route_tag: str | None,
    mode_tag: str | None,
    assignment_variant: str | None = None,
    gradient_comp_tag: str = "",
    output_name: str | None = None,
    overview_scope: str = "Group",
    skip_subject_split_ari: bool = False,
    group_name: str | None = None,
    ) -> dict[str, object]:
    if assignment_variant is None:
        assignment_variant = default_render_assignment_variant(method, atlas)
    if method in CLUSTERLESS_METHODS:
        is_lynch = str(atlas).startswith("lynch2024_")
        expected_variant = "fisherz" if is_lynch else "normalized_top20_count"
        if str(atlas) != "kong2022" and not is_lynch:
            raise ValueError(f"WTA is not supported for atlas={atlas!r}")
        if str(assignment_variant) != expected_variant:
            raise ValueError(
                f"Group overview WTA supports only assignment_variant={expected_variant!r}; "
                f"got {assignment_variant!r}"
            )
    effective_mode_tag = effective_overview_mode_tag(method, mode_tag, assignment_variant)
    effective_assignment_variant = (
        str(assignment_variant) if method in CLUSTERLESS_METHODS else str(assignment_variant)
    )
    manifest_paths: list[Path] = []
    for smoothing in SMOOTHINGS:
        manifest_paths.extend(
            [
                _build_run_dir(group_root, method, atlas, smoothing, coverage_tag, hemi, gradient_comp_tag=gradient_comp_tag)
                / "group_atlas_manifest.json"
                for hemi in ("L", "R")
            ]
        )
    source_manifests = [load_json(path) for path in manifest_paths]
    source_provenance = overview_source_provenance(
        method=method,
        manifest_paths=manifest_paths,
        manifests=source_manifests,
    )
    assignment_validation_results: dict[tuple[str, str, str], dict[str, object]] = {}
    if atlas == "kong2022" and method == "gradient-spectral":
        for path in manifest_paths:
            validate_group_atlas_lineage(path)
        # Validate all declared assignment variants before constructing any
        # surface/profile panel. The renderer never repairs or substitutes a
        # gradient assignment asset.
        for index, (path, manifest) in enumerate(zip(manifest_paths, source_manifests, strict=True)):
            ensure_gradient_assignment_variants(path, manifest)
            smoothing_name = SMOOTHINGS[index // 2]
            hemi_name = "L" if index % 2 == 0 else "R"
            for variant_name in sorted(manifest["assignment_variants"]):
                assignment_validation_results[(smoothing_name, hemi_name, str(variant_name))] = (
                    validate_gradient_assignment(path.parent, manifest, str(variant_name))
                )

    if method in CLUSTERLESS_METHODS:
        fig = plt.figure(figsize=(19.0, 12.8), dpi=180, constrained_layout=True)
        gs = fig.add_gridspec(nrows=2, ncols=3, width_ratios=[0.62, 1.18, 1.42], hspace=0.08, wspace=0.06)
    else:
        fig = plt.figure(figsize=(15.4, 12.2), dpi=180, constrained_layout=True)
        gs = fig.add_gridspec(nrows=2, ncols=2, width_ratios=[1.40, 1.30], hspace=0.06, wspace=0.05)
    rows_meta: list[dict[str, object]] = []

    for ridx, smoothing in enumerate(SMOOTHINGS):
        left_manifest, right_manifest = manifest_paths[2 * ridx : 2 * ridx + 2]
        m_l, m_r = source_manifests[2 * ridx : 2 * ridx + 2]
        d_l = left_manifest.parent
        d_r = right_manifest.parent
        if method not in CLUSTERLESS_METHODS and effective_assignment_variant == "fisherz":
            if not (atlas == "kong2022" and method == "gradient-spectral"):
                m_l = ensure_functional_label_consistency(left_manifest, m_l)
                m_r = ensure_functional_label_consistency(right_manifest, m_r)
        if method in CLUSTERLESS_METHODS:
            status_l_path = d_l / "vertex_assignment_status.json"
            status_r_path = d_r / "vertex_assignment_status.json"
            if not status_l_path.exists() or not status_r_path.exists():
                raise FileNotFoundError(
                    "Missing vertex WTA assignment status: "
                    f"left={status_l_path} right={status_r_path}"
                )
            status_l = load_json(status_l_path)
            status_r = load_json(status_r_path)
            rows_l = [dict(x) for x in status_l.get("items", [])]
            rows_r = [dict(x) for x in status_r.get("items", [])]
            rows_all = rows_l + rows_r
            ax_left = fig.add_subplot(gs[ridx, 0])
            plot_vertex_assignment_counts(
                ax_left,
                rows_all,
                f"{smoothing} | vertex assignment counts",
                title_fs=WTA_OVERVIEW_SIDE_PANEL_TITLE_FS,
                label_fs=WTA_OVERVIEW_SIDE_PANEL_LABEL_FS,
                tick_fs=WTA_OVERVIEW_SIDE_PANEL_TICK_FS,
            )

            surf_l, surf_r, template_subject, spec_path = resolve_template_pair_assets(
                m_l,
                m_r,
                left_run_dir=d_l,
                right_run_dir=d_r,
            )
            profile_sub = gs[ridx, 1].subgridspec(1, 2, wspace=0.18)
            profile_axes = [fig.add_subplot(profile_sub[0, 0]), fig.add_subplot(profile_sub[0, 1])]
            try:
                label_map = load_default_functional_label_map(d_l, m_l)
                network_names = [str(x) for x in label_map["network_names"]]
                network_label_keys = {str(k): int(v) for k, v in label_map["network_label_keys"].items()}
                labels_by_hemi = {
                    "L": load_default_functional_labels(d_l, m_l),
                    "R": load_default_functional_labels(d_r, m_r),
                }
                order_by_hemi = {"L": long_axis_order(surf_l), "R": long_axis_order(surf_r)}
                profile_rows = compute_wta_overview_profile_rows(
                    labels_by_hemi=labels_by_hemi,
                    order_by_hemi=order_by_hemi,
                    network_names=network_names,
                    network_label_keys=network_label_keys,
                )
                for ax_prof, hemi in zip(profile_axes, ["L", "R"]):
                    plot_wta_overview_profile_axis(
                        ax_prof,
                        rows=profile_rows,
                        network_names=network_names,
                        hemi=hemi,
                        title=f"{smoothing} | hemi-{hemi} long-axis profile",
                        show_ylabel=(hemi == "L"),
                    )
            except Exception as exc:
                for ax_prof in profile_axes:
                    ax_prof.axis("off")
                    ax_prof.text(
                        0.5,
                        0.5,
                        format_panel_error("Profile unavailable", exc),
                        ha="center",
                        va="center",
                        fontsize=FS_SMALL,
                        wrap=True,
                        clip_on=True,
                    )

            ax_big = fig.add_subplot(gs[ridx, 2])
            left_label = Path(str(m_l.get("template_functional_atlas_label", m_l.get("template_cluster_label"))))
            right_label = Path(str(m_r.get("template_functional_atlas_label", m_r.get("template_cluster_label"))))
            if template_subject is None or spec_path is None:
                ax_big.text(
                    0.5,
                    0.5,
                    "Surface render unavailable\n(valid template subject/spec not found)",
                    ha="center",
                    va="center",
                    fontsize=FS_MISSING,
                )
            else:
                with tempfile.TemporaryDirectory(prefix="present_group_surface_render_") as tmp:
                    render = render_locked_grid(
                        subject=template_subject,
                        scene=scene,
                        outdir=Path(tmp),
                        name=f"merged_group_vertex_wta_{method}_{atlas}_{smoothing}",
                        title=f"{str(overview_scope).strip() or 'Group'} functional atlas {method} {atlas} {smoothing} {effective_assignment_variant}",
                        left_labels=left_label,
                        right_labels=right_label,
                        left_surface=surf_l,
                        right_surface=surf_r,
                        spec_path=spec_path,
                    )
                    big = Path(str(render["biglegend_png"]))
                    ax_big.imshow(plt.imread(big))
            ax_big.set_title(f"{smoothing} | functional atlas surface", fontsize=FS_BIG_TITLE)
            ax_big.axis("off")
            rows_meta.append(
                {
                    "smoothing": smoothing,
                    "k_final_L": None,
                    "k_final_R": None,
                    "n_subjects": int(m_l["n_subjects"]),
                    "profile_bins": WTA_OVERVIEW_PROFILE_BINS,
                }
            )
            continue

        k_l_raw = m_l.get("chosen_k_final")
        k_r_raw = m_r.get("chosen_k_final")
        if k_l_raw is None or k_r_raw is None:
            sub = gs[ridx, 0].subgridspec(2, 2, wspace=0.08, hspace=0.035)
            ax0 = fig.add_subplot(sub[0, 0])
            rows_l_tsv = read_tsv(d_l / "k_selection_summary.tsv")
            rows_r_tsv = read_tsv(d_r / "k_selection_summary.tsv")
            missing_ari_hemis: list[str] = []
            for hemi_lbl, tsv_rows, color in [("L", rows_l_tsv, "#1f77b4"), ("R", rows_r_tsv, "#d62728")]:
                if not plot_ari_summary_curve(ax0, tsv_rows, hemi=hemi_lbl, color=color):
                    missing_ari_hemis.append(hemi_lbl)
            annotate_missing_ari_curves(ax0, missing_ari_hemis)
            if k_l_raw is not None:
                ax0.axvline(int(k_l_raw), color="#1f77b4", ls="--", lw=1.0)
            if k_r_raw is not None:
                ax0.axvline(int(k_r_raw), color="#d62728", ls="--", lw=1.0)
            ax0.set_title(f"{smoothing} | K diagnostics (ARI)", fontsize=FS_TITLE)
            ax0.set_xlabel("K", fontsize=FS_LABEL)
            ax0.set_ylabel("Subject-split ARI", fontsize=FS_LABEL)
            ax0.tick_params(labelsize=FS_TICK)
            if skip_subject_split_ari:
                ax0.clear()
                plot_subject_split_ari_skipped(ax0, smoothing=smoothing)
            ax0.legend(
                handles=[
                    Line2D([0], [0], color="#1f77b4", lw=1.6, marker="o", label="Left hemisphere"),
                    Line2D([0], [0], color="#d62728", lw=1.6, marker="o", label="Right hemisphere"),
                    Line2D([0], [0], color="#555555", lw=1.2, ls=":", label="Null baseline"),
                ],
                loc="center",
                fontsize=FS_LEGEND_SMALL,
                framealpha=0.9,
                edgecolor="#dddddd",
                borderpad=0.3,
                labelspacing=0.28,
                handlelength=2.1,
            )
            if skip_subject_split_ari:
                ax0.clear()
                plot_subject_split_ari_skipped(ax0, smoothing=smoothing)
            diag_lines: list[str] = []
            for hemi_lbl, k_raw, d_hemi, tsv_rows in [
                ("L", k_l_raw, d_l, rows_l_tsv),
                ("R", k_r_raw, d_r, rows_r_tsv),
            ]:
                if k_raw is None:
                    diag_lines.append(f"hemi-{hemi_lbl}: NO K SELECTED")
                    diag_lines.extend(_k_failure_lines(read_tsv(d_hemi / "k_selection_summary.tsv")))
                else:
                    diag_lines.append(f"hemi-{hemi_lbl}: K={k_raw} selected (ok)")
            diag_text = "\n".join(diag_lines)
            for rr, cc in [(0, 1), (1, 0), (1, 1)]:
                ax_diag = fig.add_subplot(sub[rr, cc])
                ax_diag.axis("off")
                ax_diag.text(0.04, 0.96, diag_text, transform=ax_diag.transAxes,
                             ha="left", va="top", fontsize=FS_SMALL, family="monospace")
            ax_big = fig.add_subplot(gs[ridx, 1])
            ax_big.axis("off")
            ax_big.set_title(f"{smoothing} | functional atlas surface", fontsize=FS_BIG_TITLE)
            none_hemis = "/".join(h for h, k in [("L", k_l_raw), ("R", k_r_raw)] if k is None)
            ax_big.text(0.5, 0.5, f"No K selected\n(hemi-{none_hemis})",
                        ha="center", va="center", fontsize=FS_MISSING, color="#cc0000")
            rows_meta.append({"smoothing": smoothing, "k_final_L": k_l_raw,
                               "k_final_R": k_r_raw, "n_subjects": int(m_l["n_subjects"])})
            continue

        k_l = int(k_l_raw)
        k_r = int(k_r_raw)
        prof_l = np.load(d_l / "cluster_network_profile.npy").astype(np.float32)
        prof_r = np.load(d_r / "cluster_network_profile.npy").astype(np.float32)
        lbl_l = np.load(d_l / "group_labels_final.npy").astype(np.int32)
        lbl_r = np.load(d_r / "group_labels_final.npy").astype(np.int32)
        assert effective_assignment_variant is not None
        status_l_path, left_label_variant = resolve_assignment_variant_paths(d_l, m_l, effective_assignment_variant)
        status_r_path, right_label_variant = resolve_assignment_variant_paths(d_r, m_r, effective_assignment_variant)
        if not status_l_path.exists() or not status_r_path.exists():
            raise FileNotFoundError(
                f"Missing assignment variant status for {method} {atlas} {smoothing} variant={effective_assignment_variant}: "
                f"left={status_l_path} right={status_r_path}"
            )
        if not left_label_variant.exists() or not right_label_variant.exists():
            raise FileNotFoundError(
                f"Missing assignment variant functional labels for {method} {atlas} {smoothing} variant={effective_assignment_variant}: "
                f"left={left_label_variant} right={right_label_variant}"
            )
        if method == "gradient-spectral" and atlas == "kong2022":
            validated_l = assignment_validation_results[(str(smoothing), "L", effective_assignment_variant)]
            validated_r = assignment_validation_results[(str(smoothing), "R", effective_assignment_variant)]
        elif method == "gradient-spectral":
            validated_l = validate_gradient_assignment(d_l, m_l, effective_assignment_variant)
            validated_r = validate_gradient_assignment(d_r, m_r, effective_assignment_variant)
            assignment_validation_results[(str(smoothing), "L", effective_assignment_variant)] = validated_l
            assignment_validation_results[(str(smoothing), "R", effective_assignment_variant)] = validated_r
        else:
            validated_l = {}
            validated_r = {}
        status_l = load_json(status_l_path)
        status_r = load_json(status_r_path)
        surf_l, surf_r, template_subject, spec_path = resolve_template_pair_assets(
            m_l,
            m_r,
            left_run_dir=d_l,
            right_run_dir=d_r,
        )
        order_l = long_axis_order(surf_l)
        order_r = long_axis_order(surf_r)
        cids_l = resolve_cluster_order_from_manifest(m_l, lbl_l, order_l, k_l)
        cids_r = resolve_cluster_order_from_manifest(m_r, lbl_r, order_r, k_r)
        prof_l_o = reorder_profile(prof_l, cids_l)
        prof_r_o = reorder_profile(prof_r, cids_r)
        row_lbl_l = [f"L-C{i}" for i in range(1, len(cids_l) + 1)]
        row_lbl_r = [f"R-C{i}" for i in range(1, len(cids_r) + 1)]

        sub = gs[ridx, 0].subgridspec(2, 2, wspace=0.08, hspace=0.035)
        ax0 = fig.add_subplot(sub[0, 0])
        rows_l = read_tsv(d_l / "k_selection_summary.tsv")
        rows_r = read_tsv(d_r / "k_selection_summary.tsv")
        if skip_subject_split_ari:
            plot_subject_split_ari_skipped(ax0, smoothing=smoothing)
        else:
            missing_ari_hemis: list[str] = []
            for hemi, rows, color in [("L", rows_l, "#1f77b4"), ("R", rows_r, "#d62728")]:
                if not plot_ari_summary_curve(ax0, rows, hemi=hemi, color=color):
                    missing_ari_hemis.append(hemi)
            annotate_missing_ari_curves(ax0, missing_ari_hemis)
        primary_sources = {str(m_l.get("primary_k_source", "automatic_k")), str(m_r.get("primary_k_source", "automatic_k"))}
        primary_label = "Default K" if primary_sources == {"default_k"} else "Primary K"
        ax0.axvline(k_l, color="#1f77b4", ls="--", lw=1.0)
        ax0.axvline(k_r, color="#d62728", ls="--", lw=1.0)
        automatic_k_l = m_l.get("automatic_k_final")
        automatic_k_r = m_r.get("automatic_k_final")
        automatic_marker = False
        for automatic_k, primary_k, color in [(automatic_k_l, k_l, "#1f77b4"), (automatic_k_r, k_r, "#d62728")]:
            if automatic_k is not None and int(automatic_k) != int(primary_k):
                ax0.axvline(int(automatic_k), color=color, ls=":", lw=1.0)
                automatic_marker = True
        legend_handles = [
            Line2D([0], [0], color="#1f77b4", lw=1.6, marker="o", label="Left hemisphere"),
            Line2D([0], [0], color="#d62728", lw=1.6, marker="o", label="Right hemisphere"),
            Line2D([0], [0], color="#555555", lw=1.2, ls=":", label="Null baseline"),
            Line2D([0], [0], color="#555555", lw=1.2, ls="--", label=primary_label),
        ]
        if automatic_marker:
            legend_handles.append(Line2D([0], [0], color="#555555", lw=1.2, ls=":", label="Automatic K"))
        ax0.legend(
            handles=legend_handles,
            loc="center",
            fontsize=FS_LEGEND_SMALL,
            framealpha=0.9,
            edgecolor="#dddddd",
            borderpad=0.3,
            labelspacing=0.28,
            handlelength=2.1,
        )
        if not skip_subject_split_ari:
            ax0.set_title(f"{smoothing} | K selection (ARI)", fontsize=FS_TITLE)
        ax0.set_xlabel("K", fontsize=FS_LABEL)
        ax0.set_ylabel("Subject-split ARI", fontsize=FS_LABEL)
        ax0.tick_params(labelsize=FS_TICK)
        if skip_subject_split_ari:
            ax0.clear()
            plot_subject_split_ari_skipped(ax0, smoothing=smoothing)

        ax1 = fig.add_subplot(sub[0, 1])
        nets = [str(x) for x in load_json(d_l / "group_feature_summary.json")["network_names"]]
        if effective_assignment_variant in {"fisherz", "weighted_fisherz"}:
            validate_fisher_status_matches_profile(
                profile_ordered=prof_l_o,
                status_payload=status_l,
                network_names=nets,
                cluster_ids=cids_l,
                context=f"{method} {atlas} {smoothing} hemi-L variant={effective_assignment_variant}",
            )
            validate_fisher_status_matches_profile(
                profile_ordered=prof_r_o,
                status_payload=status_r,
                network_names=nets,
                cluster_ids=cids_r,
                context=f"{method} {atlas} {smoothing} hemi-R variant={effective_assignment_variant}",
            )
        elif effective_assignment_variant == "normalized_top20_count":
            prof_l_o = normalized_count_profile_from_status(
                status_l,
                network_names=nets,
                cluster_ids=cids_l,
            )
            prof_r_o = normalized_count_profile_from_status(
                status_r,
                network_names=nets,
                cluster_ids=cids_r,
            )
            validate_normalized_count_status_matches_profile(
                profile_ordered=prof_l_o,
                status_payload=status_l,
                network_names=nets,
                cluster_ids=cids_l,
                context=f"{method} {atlas} {smoothing} hemi-L variant={effective_assignment_variant}",
            )
            validate_normalized_count_status_matches_profile(
                profile_ordered=prof_r_o,
                status_payload=status_r,
                network_names=nets,
                cluster_ids=cids_r,
                context=f"{method} {atlas} {smoothing} hemi-R variant={effective_assignment_variant}",
            )
        profile_is_normalized_count = effective_assignment_variant == "normalized_top20_count"
        profile_title = (
            f"{smoothing} | cluster_network_profile "
            f"({'normalized topN counts' if profile_is_normalized_count else 'signed'})"
        )
        prof_lr, yt_prof, yl_prof = concat_lr(prof_l_o, row_lbl_l, prof_r_o, row_lbl_r)
        if profile_is_normalized_count:
            vmax = float(np.nanmax(prof_lr)) if np.isfinite(prof_lr).any() else 1.0
            im1 = style_imshow(ax1, prof_lr, profile_title, vmin=0.0, vmax=max(1.0, vmax), cmap="magma")
        else:
            vmax = float(np.nanmax(np.abs(prof_lr))) if np.isfinite(prof_lr).any() else 1.0
            im1 = style_imshow(ax1, prof_lr, profile_title, vmin=-vmax, vmax=vmax, cmap="coolwarm")
        ax1.set_yticks(yt_prof)
        ax1.set_yticklabels(yl_prof, fontsize=FS_LABEL)
        ax1.set_xticks(np.arange(len(nets)))
        ax1.set_xticklabels(nets, fontsize=FS_TICK, rotation=35, ha="right")
        annotate_profile_values(ax1, prof_lr, vmax)
        highlight_top1_profile_cells(ax1, prof_lr)
        add_cbar(fig, ax1, im1, "Normalized TopN ROI count" if profile_is_normalized_count else "Fisher-z")

        ax2 = fig.add_subplot(sub[1, 0])
        plot_k_gate_panel(
            ax2,
            rows_l=rows_l,
            rows_r=rows_r,
            k_l=k_l,
            k_r=k_r,
            smoothing=smoothing,
        )

        ax3 = fig.add_subplot(sub[1, 1])
        conf_rows = confidence_rows(status_l, "L", cids_l) + confidence_rows(status_r, "R", cids_r)
        plot_confidence_panel(ax3, conf_rows, f"{smoothing} | cluster confidence / ambiguity")

        ax_big = fig.add_subplot(gs[ridx, 1])
        left_label = left_label_variant
        right_label = right_label_variant
        if template_subject is None or spec_path is None:
            ax_big.text(
                0.5,
                0.5,
                "Surface render unavailable\n(valid template subject/spec not found)",
                ha="center",
                va="center",
                fontsize=FS_MISSING,
            )
        else:
            with tempfile.TemporaryDirectory(prefix="present_group_surface_render_") as tmp:
                render = render_locked_grid(
                    subject=template_subject,
                    scene=scene,
                    outdir=Path(tmp),
                    name=f"merged_group_cluster_wta_{method}_{atlas}_{smoothing}",
                    title=f"{str(overview_scope).strip() or 'Group'} functional atlas {method} {atlas} {smoothing} {effective_assignment_variant}",
                    left_labels=left_label,
                    right_labels=right_label,
                    left_surface=surf_l,
                    right_surface=surf_r,
                    spec_path=spec_path,
                )
                big = Path(str(render["biglegend_png"]))
                ax_big.imshow(plt.imread(big))
        ax_big.set_title(f"{smoothing} | functional atlas surface", fontsize=FS_BIG_TITLE)
        ax_big.axis("off")

        rows_meta.append(
            {
                "smoothing": smoothing,
                "k_final_L": k_l,
                "k_final_R": k_r,
                "n_subjects": int(m_l["n_subjects"]),
            }
        )

    out_png = overview_png_path(
        present_dir=present_dir,
        atlas=atlas,
        method=method,
        route_tag=route_tag,
        mode_tag=effective_mode_tag,
        assignment_variant=effective_assignment_variant,
        gradient_comp_tag=gradient_comp_tag,
        output_name=output_name,
    )
    inferred_group_name = (
        group_name
        or infer_group_name_from_path(group_root)
        or infer_group_name_from_path(present_dir)
    )
    n_sub = rows_meta[0]["n_subjects"] if rows_meta and "n_subjects" in rows_meta[0] else None
    rendered_title = overview_title(
        atlas=atlas,
        method=method,
        route_tag=route_tag,
        mode_tag=effective_mode_tag,
        assignment_variant=effective_assignment_variant,
        surface_affinity=source_provenance.get("surface_affinity"),
        overview_scope=overview_scope,
        group_name=inferred_group_name,
        n_subjects=n_sub,
    )
    assignment_annotation_note = None
    annotation_note = source_provenance.get("cluster_annotation_note")
    if annotation_note is not None:
        title_lines = [rendered_title, str(annotation_note)]
        if effective_assignment_variant == "normalized_top20_count":
            assignment_annotation_note = CTXDROP_NORMALIZED_COUNT_ANNOTATION_NOTE
            title_lines.append(assignment_annotation_note)
        rendered_title = "\n".join(title_lines)
    fig.suptitle(rendered_title, fontsize=FS_SUPTITLE)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    assignment_assets = {
        f"{smoothing}/hemi-{hemi}/{variant}": result.get("asset_fingerprints", {})
        for (smoothing, hemi, variant), result in sorted(assignment_validation_results.items())
    }
    source_sidecar = out_png.with_suffix(out_png.suffix + ".sources.json")
    write_json(
        source_sidecar,
        {
            "schema": "group_overview_source_lock.v1",
            "overview_png": str(out_png.resolve()),
            "overview_png_sha256": sha256_file(out_png),
            "method": str(method),
            "atlas": str(atlas),
            "assignment_variant": str(effective_assignment_variant),
            "source_manifests": [str(path.resolve()) for path in manifest_paths],
            "assignment_assets": assignment_assets,
        },
    )
    return {
        "overview_png": str(out_png),
        "overview_source_sidecar": str(source_sidecar),
        "rows": rows_meta,
        "rendered_title": rendered_title,
        "overview_scope": str(overview_scope).strip() or "Group",
        "subject_split_ari": "skipped" if skip_subject_split_ari else "rendered",
        "assignment_annotation_note": assignment_annotation_note,
        "assignment_assets": assignment_assets,
        **source_provenance,
    }


def write_readme(present_dir: Path, items: list[dict[str, object]]) -> None:
    lines = [
        "# HCP 7T hippocampus group overviews",
        "",
        "| atlas | method | route | mode | assignment | overview | smoothings | annotation | key metrics |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for item in sorted(items, key=lambda x: (str(x["atlas"]), str(x["method"]))):
        png = Path(str(item["overview_png"])).name
        sm = ", ".join(str(s) for s in item.get("smoothings", []))
        metric = str(item.get("metric_summary", ""))
        route = str(item.get("route_tag", ""))
        mode = str(item.get("mode_tag", ""))
        assignment = str(item.get("assignment_variant", ""))
        annotation_parts = [
            str(item[key])
            for key in ("cluster_annotation_note", "assignment_annotation_note")
            if item.get(key)
        ]
        annotation = "<br>".join(annotation_parts)
        lines.append(f"| {item['atlas']} | {item['method']} | {route} | {mode} | {assignment} | [{png}]({png}) | {sm} | {annotation} | {metric} |")
    (present_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_vertex_wta_status_payload(
    *,
    status_payload: dict[str, object],
    manifest: dict[str, object],
    status_path: Path,
) -> None:
    raw_items = status_payload.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise RuntimeError(f"WTA status has no support-domain rows: {status_path}")
    raw_network_names = manifest.get("network_order", manifest.get("network_names"))
    if not isinstance(raw_network_names, list) or not raw_network_names:
        raise RuntimeError(f"WTA manifest has no declared network order: {status_path}")
    network_names = {str(name) for name in raw_network_names}
    invalid: list[str] = []
    assigned = 0
    for row_index, raw_row in enumerate(raw_items):
        if not isinstance(raw_row, dict):
            invalid.append(f"row={row_index} is not an object")
            continue
        status = str(raw_row.get("status", "unassigned"))
        if status == "dominant":
            top1 = str(raw_row.get("top1_network", "")).strip()
            if top1 not in network_names:
                invalid.append(f"row={row_index} dominant top1={top1!r}")
            else:
                assigned += 1
        elif status != "unassigned":
            invalid.append(f"row={row_index} unknown status={status!r}")
    if assigned == 0:
        raise RuntimeError(
            "WTA support domain is all unassigned or invalid; "
            f"status={status_path} invalid={invalid[:5]}"
        )
    if invalid:
        raise RuntimeError(f"WTA status contains invalid assignments: {status_path} invalid={invalid[:5]}")


def ensure_combo_ready(
    *,
    group_root: Path,
    method: str,
    atlas: str,
    coverage_tag: str,
    smoothings: list[str] | None = None,
    hemis: list[str] | None = None,
) -> None:
    sm = smoothings or ["2mm", "4mm"]
    hs = hemis or ["L", "R"]
    problems: list[str] = []
    for smoothing in sm:
        comp_tags = _discover_gradient_comp_tags(group_root, method, atlas, smoothing, coverage_tag)
        for comp_tag in comp_tags:
            for hemi in hs:
                run_dir = _build_run_dir(group_root, method, atlas, smoothing, coverage_tag, hemi, gradient_comp_tag=comp_tag)
                manifest_path = run_dir / "group_atlas_manifest.json"
                if not manifest_path.exists():
                    problems.append(f"missing_manifest: {manifest_path}")
                    continue
                m = load_json(manifest_path)
                if atlas == "kong2022" and method == "gradient-spectral":
                    try:
                        validate_group_atlas_lineage(manifest_path)
                    except ValueError as exc:
                        problems.append(f"invalid_group_atlas_lineage: {exc}")
                if str(m.get("status", "")) != "ok":
                    problems.append(f"manifest_not_ok: {manifest_path} status={m.get('status')}")
                    continue
                if method in CLUSTERLESS_METHODS:
                    status_path = run_dir / "vertex_assignment_status.json"
                    if not status_path.exists():
                        problems.append(f"missing_vertex_assignment_status: {status_path}")
                    else:
                        try:
                            validate_vertex_wta_status_payload(
                                status_payload=load_json(status_path), manifest=m, status_path=status_path
                            )
                        except RuntimeError as exc:
                            problems.append(f"invalid_vertex_assignment_status: {exc}")
                else:
                    # For cluster-based methods, group labels must exist.
                    labels = run_dir / "group_labels_final.npy"
                    if not labels.exists():
                        problems.append(f"missing_group_labels: {labels}")
    if problems:
        raise RuntimeError(
            "Cluster outputs are not ready. Do not skip clustering before present rendering.\n"
            + "\n".join(problems)
        )


def main() -> int:
    p = argparse.ArgumentParser(description="Build canonical HCP 7T hippocampus group overviews from group atlas outputs.")
    p.add_argument("--group-root", default=str(DEFAULT_GROUP_ROOT))
    p.add_argument("--present-dir", default=str(DEFAULT_PRESENT_DIR))
    p.add_argument(
        "--methods",
        nargs="+",
        default=None,
        choices=METHOD_CHOICES,
        help="Methods to render. If omitted, Lynch atlases render gradient-spectral and direct-network wta-top; Kong2022 uses its standard methods including normalized_top20_count WTA.",
    )
    p.add_argument("--atlases", nargs="+", default=DEFAULT_ATLASES)
    p.add_argument("--coverage-threshold", type=float, default=0.80)
    p.add_argument("--scene", default=str(DEFAULT_SCENE))
    p.add_argument("--route-tag", choices=ROUTE_TAG_CHOICES, default=None)
    p.add_argument("--mode-tag", default=None)
    p.add_argument(
        "--output-name",
        default=None,
        help="Optional PNG filename override under --present-dir. Use only for single-render calls.",
    )
    p.add_argument("--assignment-variant", choices=ASSIGNMENT_VARIANT_CHOICES, default="both")
    p.add_argument(
        "--group-name",
        default=None,
        help="Optional explicit group name label in overview title, e.g. NTHC.",
    )
    p.add_argument(
        "--manifest-name",
        default="group_overview_manifest.json",
        help="Output manifest filename under --present-dir.",
    )
    p.add_argument(
        "--skip-readme",
        action="store_true",
        help="Skip writing README.md (useful when an upstream driver writes a consolidated README).",
    )
    p.add_argument(
        "--skip-manifest",
        action="store_true",
        help="Skip writing manifest JSON (useful when an upstream driver writes a consolidated manifest).",
    )
    args = p.parse_args()
    group_root = Path(args.group_root).resolve()

    # Preserve explicit comparison mode tags; otherwise infer from the group-root name.
    args.mode_tag = resolve_overview_mode_tag(group_root, args.mode_tag)

    # Enforce strict route tag detection from group_root to prevent provenance decoupling.
    if args.route_tag is None:
        args.route_tag = infer_route_tag_from_group_root(group_root)
    else:
        validate_group_atlas_root_for_route(group_root, str(args.route_tag))

    present_dir = Path(args.present_dir).resolve()
    present_dir.mkdir(parents=True, exist_ok=True)
    tau_tag = f"cov_{args.coverage_threshold:.2f}".replace(".", "p")

    methods_explicit = args.methods is not None
    items: list[dict[str, object]] = []
    for atlas in args.atlases:
        methods = tuple(args.methods) if methods_explicit else default_methods_for_atlas(atlas)
        for method in methods:
            validate_method_route_compatibility(method, args.route_tag)
            ensure_combo_ready(
                group_root=group_root,
                method=method,
                atlas=atlas,
                coverage_tag=tau_tag,
            )
            comp_tags = _discover_gradient_comp_tags(
                group_root, method, atlas, SMOOTHINGS[0], tau_tag,
            )
            render_variants = resolve_render_assignment_variants(
                method,
                str(args.assignment_variant),
                atlas=atlas,
            )
            for comp_tag in comp_tags:
                for assignment_variant in render_variants:
                    meta = render_combo(
                        group_root=group_root,
                        present_dir=present_dir,
                        method=method,
                        atlas=atlas,
                        coverage_tag=tau_tag,
                        scene=Path(args.scene).resolve(),
                        route_tag=args.route_tag,
                        mode_tag=args.mode_tag,
                        assignment_variant=(assignment_variant or "fisherz"),
                        gradient_comp_tag=comp_tag,
                        output_name=args.output_name,
                        group_name=args.group_name,
                    )
                    row_meta = meta["rows"]
                    metric = (
                        "; ".join(
                            [
                                f"{r['smoothing']}: K(L/R)={('-' if r['k_final_L'] is None else r['k_final_L'])}/"
                                f"{('-' if r['k_final_R'] is None else r['k_final_R'])} N={r['n_subjects']}"
                                for r in row_meta
                            ]
                        )
                        if row_meta
                        else "missing"
                    )
                    item = overview_manifest_item(
                        base={
                            "atlas": atlas,
                            "method": method,
                            "gradient_comp_tag": comp_tag or None,
                            "overview_png": meta["overview_png"],
                            "smoothings": [r["smoothing"] for r in row_meta],
                            "metric_summary": metric,
                            "route_tag": args.route_tag,
                            "mode_tag": effective_overview_mode_tag(method, args.mode_tag, assignment_variant),
                            "assignment_variant": assignment_variant,
                            "source_group_root": str(group_root),
                        },
                        meta=meta,
                    )
                    items.append(item)

    manifest = {"present_dir": str(present_dir), "count": len(items), "items": items}
    if not bool(args.skip_manifest):
        (present_dir / args.manifest_name).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not bool(args.skip_readme):
        write_readme(present_dir, items)
    print(json.dumps({"present_dir": str(present_dir), "count": len(items)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Cross-Dataset Hippocampal 2mm Normalized WTA Atlas & Long-Axis Composition (2-Column Layout).

Renders a 2-column comparative panel across 10 cohorts:
  Left Column (5 Cohorts - Controls & Reference):
    - HCP 7T (Healthy Young Adults, N=151)
    - Iowa 3T HC (Healthy Controls, N=67)
    - Iowa 3T Patient (Depression Patients, N=26; sub-3003 excluded)
    - Stanford NTHC (Non-Trauma Controls, N=51)
    - Stanford TEHC (Trauma-Exposed Controls, N=40)
  Right Column (5 Cohorts - Symptomatic & MDD Stratification):
    - Stanford NTS (Non-Trauma Symptomatic, N=56)
    - Stanford NTS Any MDD (Non-Trauma MDD [All], N=43)
    - Stanford NTS Pure MDD (Non-Trauma Pure MDD, N=11)
    - Stanford TIS (Trauma Symptomatic, N=47)
    - Stanford TIS PTSD (Trauma-Informed PTSD, N=35)

Architecture Guarantee:
  - Single Source of Truth (SSOT): Hippocampus 3D surface atlas renders directly from canonical label GIITIs via Workbench, guaranteed 100% mathematically consistent with long-axis stacked bar data.
  - Runtime Consistency Gate: Asserts that label arrays in NPY and GIITI match perfectly before plotting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.patches as patches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
CORE_DIR = REPO_ROOT / "scripts" / "core_workflow"
VIS_LIB_DIR = REPO_ROOT / "scripts" / "visualize" / "lib"
for p in (CORE_DIR, VIS_LIB_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lib.surface_utils import long_axis_order, render_locked_grid
from lib.workbench_grid import load_labels
from multidataset_cohorts import (
    COMPARISON_VERSION,
    GRID_VARIANTS,
    SIGNAL_BRANCHES,
    GridSpec,
    format_comparison_column_title,
    get_comparison_output_tag,
    get_comparison_visual_columns,
    get_multidataset_cohorts,
    has_explicit_legacy_hcp_ctxreg_provenance,
)

# Canonical network styling and color definitions
NETWORK_COLORS = {
    "Default": "#de616c",
    "Somatomotor": "#5297c8",
    "Auditory": "#18b7c9",
    "VentralAttention": "#da42f5",
    "DorsalAttention": "#33a02c",
    "Control": "#f39c12",
    "Language": "#8b4513",
    "Visual": "#6a3d9a",
    "Null": "#888888",
    "None": "#888888",
    "non-network/null": "#888888",
}

ABBREV_MAP = {
    "Default": "DMN",
    "Visual": "VN",
    "Somatomotor": "SMN",
    "DorsalAttention": "DAN",
    "VentralAttention": "VAN",
    "Control": "CON",
    "Auditory": "AUD",
    "Language": "LAN",
    "Null": "Null",
    "None": "Null",
    "non-network/null": "Null",
}

# Preferred stacking order
STACK_ORDER = [
    "Default",
    "Somatomotor",
    "Auditory",
    "VentralAttention",
    "DorsalAttention",
    "Control",
    "Language",
    "Visual",
    "non-network/null",
]

# Cohort specifications for 2-column layout
COHORTS_COL1 = [
    {
        "dataset_name": "HCP 7T",
        "cohort_name": "HCP 7T",
        "cohort_desc": "Healthy Young Adults",
        "n_subjects": 151,
        "group_root": REPO_ROOT
        / "outputs_hcp/work/group_atlas/pfm",
        "bg_color": "#ffffff",
        "badge_bg": "#0b3c5d",
        "badge_fg": "#ffffff",
    },
    {
        "dataset_name": "Iowa 3T",
        "cohort_name": "HC",
        "cohort_desc": "Healthy Controls",
        "n_subjects": 67,
        "group_root": REPO_ROOT
        / "outputs_iowa_jj_tms/work/group_atlas/pfm/hc",
        "bg_color": "#f7faff",
        "badge_bg": "#1d5b79",
        "badge_fg": "#ffffff",
    },
    {
        "dataset_name": "Iowa 3T",
        "cohort_name": "Patient",
        "cohort_desc": "Depression Patients",
        "n_subjects": 26,
        "group_root": REPO_ROOT
        / "outputs_iowa_jj_tms/work/group_atlas/pfm/patient",
        "bg_color": "#f7faff",
        "badge_bg": "#1d5b79",
        "badge_fg": "#ffffff",
    },
    {
        "dataset_name": "Stanford",
        "cohort_name": "NTHC",
        "cohort_desc": "Non-Trauma Controls",
        "n_subjects": 51,
        "group_root": REPO_ROOT
        / "outputs_stanford/work/group_atlas/pfm/nthc",
        "bg_color": "#fffdf8",
        "badge_bg": "#8c4600",
        "badge_fg": "#ffffff",
    },
    {
        "dataset_name": "Stanford",
        "cohort_name": "TEHC",
        "cohort_desc": "Trauma-Exposed Controls",
        "n_subjects": 40,
        "group_root": REPO_ROOT
        / "outputs_stanford/work/group_atlas/pfm/tehc",
        "bg_color": "#fffdf8",
        "badge_bg": "#8c4600",
        "badge_fg": "#ffffff",
    },
]

COHORTS_COL2 = [
    {
        "dataset_name": "Stanford",
        "cohort_name": "NTS",
        "cohort_desc": "Non-Trauma Symptomatic",
        "n_subjects": 56,
        "group_root": REPO_ROOT
        / "outputs_stanford/work/group_atlas/pfm/nts",
        "bg_color": "#fffdf8",
        "badge_bg": "#8c4600",
        "badge_fg": "#ffffff",
    },
    {
        "dataset_name": "Stanford",
        "cohort_name": "NTS Any MDD",
        "cohort_desc": "Non-Trauma MDD (All)",
        "n_subjects": 43,
        "group_root": REPO_ROOT
        / "outputs_stanford/work/group_atlas/pfm/nts_any_mdd",
        "bg_color": "#fffdf8",
        "badge_bg": "#8c4600",
        "badge_fg": "#ffffff",
    },
    {
        "dataset_name": "Stanford",
        "cohort_name": "NTS Pure MDD",
        "cohort_desc": "Non-Trauma Pure MDD",
        "n_subjects": 11,
        "group_root": REPO_ROOT
        / "outputs_stanford/work/group_atlas/pfm/nts_pure_mdd",
        "bg_color": "#fffdf8",
        "badge_bg": "#8c4600",
        "badge_fg": "#ffffff",
    },
    {
        "dataset_name": "Stanford",
        "cohort_name": "TIS",
        "cohort_desc": "Trauma Symptomatic",
        "n_subjects": 47,
        "group_root": REPO_ROOT
        / "outputs_stanford/work/group_atlas/pfm/tis",
        "bg_color": "#fffdf8",
        "badge_bg": "#8c4600",
        "badge_fg": "#ffffff",
    },
    {
        "dataset_name": "Stanford",
        "cohort_name": "TIS PTSD",
        "cohort_desc": "Trauma-Informed PTSD",
        "n_subjects": 35,
        "group_root": REPO_ROOT
        / "outputs_stanford/work/group_atlas/pfm/tis_ptsd",
        "bg_color": "#fffdf8",
        "badge_bg": "#8c4600",
        "badge_fg": "#ffffff",
    },
]


def _entry_signal_branch(cohort_entry: dict) -> str:
    """Read the branch carried by a renderer-owned cohort entry."""
    raw_branch = cohort_entry.get("signal_branch")
    if raw_branch is None:
        raise ValueError(
            "WTA cohort entry must declare signal_branch explicitly: "
            f"{cohort_entry.get('dataset_name', '<unknown>')} "
            f"{cohort_entry.get('cohort_name', '<unknown>')}"
        )
    branch = str(raw_branch)
    if branch not in SIGNAL_BRANCHES:
        raise ValueError(
            f"Unsupported WTA signal_branch {branch!r} for "
            f"{cohort_entry.get('dataset_name', '<unknown>')} "
            f"{cohort_entry.get('cohort_name', '<unknown>')}"
        )
    return branch


def _resolve_wta_manifest_file(
    manifest: dict,
    field: str,
    *,
    manifest_path: Path,
    must_be_within: Path | None = None,
) -> Path:
    """Resolve one exact manifest-declared WTA asset."""
    raw_path = manifest.get(field)
    if not isinstance(raw_path, str) or not raw_path.strip():
        canonical_name = (
            "group_functional_atlas_labels.npy"
            if field == "group_functional_atlas_labels"
            else field
        )
        raise FileNotFoundError(
            f"Missing exact WTA manifest field {field!r} "
            f"(canonical filename contract: {canonical_name}): {manifest_path}"
        )
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        raise ValueError(
            f"WTA manifest field {field!r} must be an absolute path: {manifest_path}"
        )
    path = path.resolve()
    if must_be_within is not None:
        try:
            path.relative_to(Path(must_be_within).resolve())
        except ValueError as exc:
            raise ValueError(
                f"WTA manifest field {field!r} points outside its hemisphere "
                f"run directory: {path}"
            ) from exc
    if not path.is_file():
        raise FileNotFoundError(f"Missing exact WTA manifest field {field!r}: {path}")
    return path


def _validate_wta_manifest_signal_branch(
    manifest: dict,
    *,
    manifest_path: Path,
    expected_branch: str,
    method: str | None = None,
) -> None:
    """Reject a manifest whose recorded signal branch differs from the request."""
    recorded_branch = manifest.get("signal_branch")
    # Renderer-identical legacy rule (see comparison readiness audit):
    # pre-signal_branch HCP Kong CtxReg products pass only under the explicit
    # named legacy provenance contract, never by loose fallback.
    if (
        expected_branch == "ctxreg"
        and recorded_branch != "ctxreg"
        and not (
            recorded_branch is None
            and has_explicit_legacy_hcp_ctxreg_provenance(
                manifest_path.parent, manifest, method=method
            )
        )
    ):
        raise ValueError(
            f"{manifest_path} records signal_branch={recorded_branch!r}; "
            "expected 'ctxreg'"
        )
    if expected_branch != "ctxreg" and recorded_branch not in (None, "standard"):
        raise ValueError(
            f"{manifest_path} records signal_branch={recorded_branch!r}; "
            "expected standard"
        )


def load_cohort_long_axis_bins(
    cohort_entry: dict, n_bins: int = 20, smoothing: str = "2mm", is_7net: bool = False
) -> dict[str, list[dict[str, float]]]:
    """Load long-axis 20 bins network percentage data for Left and Right hippocampus."""
    group_root = Path(cohort_entry["group_root"])
    expected_branch = _entry_signal_branch(cohort_entry)
    # Taxonomy is metadata; the active Kong producer folder is always the
    # canonical ``kong2022`` branch. The old ``kong2022_7net`` folder is not
    # an input selection mechanism.
    atlas_folder = "kong2022"
    wta_dir = Path(
        cohort_entry.get(
            "wta_subpath", group_root / f"wta-top/{atlas_folder}/{smoothing}/cov_0p80"
        )
    )

    bin_data_by_hemi = {}
    for hemi in ["L", "R"]:
        hdir = wta_dir / f"hemi_{hemi}"
        manifest_path = hdir / "group_atlas_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing exact WTA manifest: {manifest_path}")
        mf = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(mf, dict):
            raise ValueError(f"WTA manifest must be a JSON object: {manifest_path}")
        _validate_wta_manifest_signal_branch(
            mf,
            manifest_path=manifest_path,
            expected_branch=expected_branch,
            method=cohort_entry.get("method"),
        )
        surf_path = _resolve_wta_manifest_file(
            mf, "template_left_or_right_surface", manifest_path=manifest_path
        )
        order = long_axis_order(surf_path)

        lbl_file = _resolve_wta_manifest_file(
            mf,
            "group_functional_atlas_labels",
            manifest_path=manifest_path,
            must_be_within=hdir,
        )
        labels = np.load(lbl_file).flatten()

        # Runtime Data Integrity Gate: verify NPY labels match the canonical label GIITI
        lbl_gii_path = _resolve_wta_manifest_file(
            mf, "template_functional_atlas_label", manifest_path=manifest_path
        )
        gii_vals, _ = load_labels(lbl_gii_path)
        if labels.shape != gii_vals.shape:
            raise ValueError(
                f"[DATA INTEGRITY GATE] Label shape mismatch in "
                f"{cohort_entry['dataset_name']} {cohort_entry['cohort_name']} "
                f"hemi_{hemi}: NPY {labels.shape} vs GIFTI {gii_vals.shape}"
            )
        mismatch_cnt = int(np.sum(labels != gii_vals))
        if mismatch_cnt > 0:
            raise ValueError(
                f"[DATA INTEGRITY GATE] Label array mismatch in "
                f"{cohort_entry['dataset_name']} {cohort_entry['cohort_name']} "
                f"hemi_{hemi}: {mismatch_cnt} vertices differ between NPY and GIFTI!"
            )
        if labels.shape != order.shape:
            raise ValueError(
                f"[DATA INTEGRITY GATE] Long-axis order length {order.shape} "
                f"disagrees with labels {labels.shape}: {manifest_path}"
            )

        lmap_file = _resolve_wta_manifest_file(
            mf,
            "group_functional_atlas_label_map",
            manifest_path=manifest_path,
            must_be_within=hdir,
        )
        lmap = json.loads(lmap_file.read_text(encoding="utf-8"))
        network_names = lmap["network_names"]
        label_keys = {k: int(v) for k, v in lmap["network_label_keys"].items()}
        known_keys = set(label_keys.values())

        bins = np.array_split(order, int(n_bins))
        hemi_bins = []
        for bin_idx, b_verts in enumerate(bins, start=1):
            b_lbls = labels[b_verts] if len(b_verts) > 0 else np.array([])
            denom = float(len(b_lbls))
            comp = {}
            for net in network_names:
                k = label_keys[net]
                cnt = float(np.sum(b_lbls == k)) if denom > 0 else 0.0
                comp[net] = (100.0 * cnt / denom) if denom > 0 else 0.0

            null_cnt = (
                float(np.sum(~np.isin(b_lbls, list(known_keys)))) if denom > 0 else 0.0
            )
            if null_cnt > 0:
                comp["non-network/null"] = 100.0 * null_cnt / denom
            else:
                comp["non-network/null"] = 0.0
            hemi_bins.append(comp)

        bin_data_by_hemi[hemi] = hemi_bins

    return bin_data_by_hemi


WTA_SURFACE_CACHE_SCHEMA = "wta_surface_cache_v1"
_WTA_FILE_FINGERPRINT_CACHE: dict[tuple[str, int, int], dict[str, Any]] = {}


def _wta_sha256_file(path: Path) -> str:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing WTA surface cache source file: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wta_file_fingerprint(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing WTA surface cache source file: {resolved}")
    stat = resolved.stat()
    key = (str(resolved), int(stat.st_size), int(stat.st_mtime_ns))
    cached = _WTA_FILE_FINGERPRINT_CACHE.get(key)
    if cached is not None:
        return dict(cached)
    record = {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _wta_sha256_file(resolved),
    }
    _WTA_FILE_FINGERPRINT_CACHE[key] = record
    return dict(record)


def _wta_surface_cache_inputs(
    *,
    cid: str,
    method: str | None,
    dataset_name: str,
    cohort_name: str,
    n_subjects: int,
    variant: str,
    smoothing: str,
    is_7net: bool,
    signal_branch: str,
    source_files: dict[str, Path],
) -> dict[str, Any]:
    return {
        "artifact": "wta_surface",
        "cohort_id": cid,
        "method": method,
        "dataset_name": dataset_name,
        "cohort_name": cohort_name,
        "n_subjects": int(n_subjects),
        "variant": variant,
        "smoothing": smoothing,
        "is_7net": bool(is_7net),
        "signal_branch": signal_branch,
        "source_files": {
            key: _wta_file_fingerprint(path)
            for key, path in sorted(source_files.items())
        },
        "render": {
            "subject": "100610",
            "scene": "wb_locked_native_view_lateral_medial",
        },
    }


def _wta_surface_cache_manifest(
    expected_inputs: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    path = Path(output_path).resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(
            f"Cannot publish WTA surface cache for missing output: {path}"
        )
    with Image.open(path) as image:
        image.verify()
        dimensions = [int(image.width), int(image.height)]
    return {
        "cache_schema": WTA_SURFACE_CACHE_SCHEMA,
        "inputs": expected_inputs,
        "outputs": {
            "biglegend": {
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "sha256": _wta_sha256_file(path),
                "dimensions": dimensions,
            }
        },
    }


def _is_valid_wta_surface_cache(
    manifest_path: Path,
    *,
    expected_inputs: dict[str, Any],
    output_path: Path,
) -> bool:
    """Return true only for a complete WTA surface cache with exact provenance."""
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if manifest.get("cache_schema") != WTA_SURFACE_CACHE_SCHEMA:
            return False
        if manifest.get("inputs") != expected_inputs:
            return False
        record = manifest.get("outputs", {}).get("biglegend")
        expected_path = Path(output_path).resolve()
        if not isinstance(record, dict) or record.get("path") != str(expected_path):
            return False
        if not expected_path.is_file():
            return False
        if record.get("size_bytes") != int(expected_path.stat().st_size):
            return False
        if record.get("sha256") != _wta_sha256_file(expected_path):
            return False
        with Image.open(expected_path) as image:
            image.verify()
            dimensions = [int(image.width), int(image.height)]
        return record.get("dimensions") == dimensions
    except Exception:
        # A malformed or incomplete cache is a miss; canonical inputs are
        # still required above and are never replaced implicitly.
        return False


def render_cohort_wta_surface_img(
    cohort_entry: dict,
    panels_dir: Path,
    smoothing: str = "2mm",
    variant: str = "36p",
    is_7net: bool = False,
) -> Image.Image:
    """Render authentic 3D hippocampus surface & legend directly from canonical label GIITIs via Workbench."""
    signal_branch = _entry_signal_branch(cohort_entry)
    cname_clean = cohort_entry["cohort_name"].replace(" ", "_").lower()
    dname_clean = cohort_entry["dataset_name"].replace(" ", "_").lower()
    atlas_tag = "7net_" if is_7net else ""
    # Taxonomy is metadata; the active Kong producer folder is always the
    # canonical ``kong2022`` branch. The old ``kong2022_7net`` folder is not
    # an input selection mechanism.
    atlas_folder = "kong2022"
    method_tag = f"{cohort_entry['method']}_" if cohort_entry.get("method") else ""
    cid = f"{method_tag}{dname_clean}_{cname_clean}_{atlas_tag}{smoothing}_{variant}_{signal_branch}"
    outdir = panels_dir / f"{cid}_wta_surface"
    png_p = outdir / f"sub-100610_wb_{cid}_wta_biglegend.png"

    group_root = Path(cohort_entry["group_root"])
    wta_dir = Path(
        cohort_entry.get(
            "wta_subpath", group_root / f"wta-top/{atlas_folder}/{smoothing}/cov_0p80"
        )
    )
    mf_l_path = wta_dir / "hemi_L/group_atlas_manifest.json"
    mf_r_path = wta_dir / "hemi_R/group_atlas_manifest.json"
    if not mf_l_path.is_file() or not mf_r_path.is_file():
        missing = [str(path) for path in (mf_l_path, mf_r_path) if not path.is_file()]
        raise FileNotFoundError(f"Missing exact WTA manifest(s): {', '.join(missing)}")
    mf_l = json.loads(mf_l_path.read_text(encoding="utf-8"))
    mf_r = json.loads(mf_r_path.read_text(encoding="utf-8"))
    if not isinstance(mf_l, dict) or not isinstance(mf_r, dict):
        raise ValueError(
            f"WTA manifests must be JSON objects: {mf_l_path}, {mf_r_path}"
        )
    _validate_wta_manifest_signal_branch(
        mf_l,
        manifest_path=mf_l_path,
        expected_branch=signal_branch,
        method=cohort_entry.get("method"),
    )
    _validate_wta_manifest_signal_branch(
        mf_r,
        manifest_path=mf_r_path,
        expected_branch=signal_branch,
        method=cohort_entry.get("method"),
    )

    surf_l = _resolve_wta_manifest_file(
        mf_l, "template_left_or_right_surface", manifest_path=mf_l_path
    )
    surf_r = _resolve_wta_manifest_file(
        mf_r, "template_left_or_right_surface", manifest_path=mf_r_path
    )
    lbl_l = _resolve_wta_manifest_file(
        mf_l, "template_functional_atlas_label", manifest_path=mf_l_path
    )
    lbl_r = _resolve_wta_manifest_file(
        mf_r, "template_functional_atlas_label", manifest_path=mf_r_path
    )
    spec_path = _resolve_wta_manifest_file(
        mf_l, "template_spec_path", manifest_path=mf_l_path
    )
    subject = "100610"

    ds = cohort_entry["dataset_name"]
    ns = cohort_entry["n_subjects"]
    cn = cohort_entry["cohort_name"]

    scene_path = REPO_ROOT / "config/wb_locked_native_view_lateral_medial.scene"
    cache_manifest_path = png_p.with_suffix(".cache.json")
    cache_inputs = _wta_surface_cache_inputs(
        cid=cid,
        method=cohort_entry.get("method"),
        dataset_name=ds,
        cohort_name=cn,
        n_subjects=ns,
        variant=variant,
        smoothing=smoothing,
        is_7net=is_7net,
        signal_branch=signal_branch,
        source_files={
            "scene": scene_path,
            "left_manifest": mf_l_path,
            "right_manifest": mf_r_path,
            "left_surface": surf_l,
            "right_surface": surf_r,
            "left_label": lbl_l,
            "right_label": lbl_r,
            "spec": spec_path,
        },
    )
    if _is_valid_wta_surface_cache(
        cache_manifest_path,
        expected_inputs=cache_inputs,
        output_path=png_p,
    ):
        with Image.open(png_p) as cached_image:
            return cached_image.copy()
    t_str = f"{ds} {cn} (N={ns})"

    res = render_locked_grid(
        subject=subject,
        scene=scene_path,
        outdir=outdir,
        name=f"{cid}_wta",
        title=t_str,
        left_labels=lbl_l,
        right_labels=lbl_r,
        left_surface=surf_l,
        right_surface=surf_r,
        spec_path=spec_path,
    )
    rendered_path = Path(res["biglegend_png"]).resolve()
    if rendered_path != png_p.resolve():
        raise RuntimeError(
            f"Workbench returned WTA surface path {rendered_path}, expected {png_p.resolve()}"
        )
    cache_manifest_path.write_text(
        json.dumps(
            _wta_surface_cache_manifest(cache_inputs, rendered_path),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with Image.open(rendered_path) as rendered_image:
        return rendered_image.copy()


def render_stacked_bar_axis(
    ax,
    bin_comps: list[dict[str, float]],
    hemi: str,
    show_ylabel: bool = True,
    is_top_row: bool = False,
    is_bottom_row: bool = False,
):
    """Render 20 horizontal stacked bars for a single hemisphere."""
    n_bins = len(bin_comps)
    y_positions = np.arange(1, n_bins + 1)
    bar_height = 0.82

    for bin_idx, comp in enumerate(bin_comps):
        y = y_positions[bin_idx]
        current_left = 0.0

        # Sort non-zero networks by percentage descending
        sorted_nets = sorted(
            [(net, pct) for net, pct in comp.items() if pct > 0.001],
            key=lambda item: (
                -item[1],
                STACK_ORDER.index(item[0]) if item[0] in STACK_ORDER else 99,
            ),
        )

        for net_name, pct in sorted_nets:
            color = NETWORK_COLORS.get(net_name, "#888888")
            ax.barh(
                y,
                pct,
                left=current_left,
                height=bar_height,
                color=color,
                edgecolor="#222222",
                linewidth=0.5,
                zorder=2,
            )

            # Label prominent segments (> 11% width)
            if pct >= 11.0:
                cx = current_left + pct / 2.0
                abbrev = ABBREV_MAP.get(net_name, net_name[:3].upper())
                txt = ax.text(
                    cx,
                    y,
                    abbrev,
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    fontweight="bold",
                    color="#ffffff",
                    zorder=4,
                )
                txt.set_path_effects(
                    [
                        pe.Stroke(linewidth=1.4, foreground="#111111"),
                        pe.Normal(),
                    ]
                )

            current_left += pct

    ax.set_xlim(0, 100)
    ax.set_ylim(n_bins + 0.55, 0.45)  # Invert Y so Bin 1 (Ant) is at the top

    if show_ylabel:
        ax.set_ylabel(
            "Long-Axis Bin",
            fontsize=8.0,
            fontweight="bold",
            color="#102a43",
            labelpad=2,
        )
        ax.set_yticks([1, 5, 10, 15, 20])
        ax.set_yticklabels(
            ["1 (Ant)", "5", "10", "15", "20 (Post)"],
            fontsize=8.0,
            fontweight="bold",
            color="#243b53",
        )
    else:
        ax.set_yticks([])
        ax.set_yticklabels([])

    if is_bottom_row:
        ax.set_xlabel(
            "Vertices in Bin (%)",
            fontsize=9.0,
            fontweight="bold",
            color="#102a43",
            labelpad=4,
        )
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(
            ["0%", "25%", "50%", "75%", "100%"],
            fontsize=8.5,
            fontweight="bold",
            color="#243b53",
        )
    else:
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels([])

    ax.tick_params(
        axis="both",
        which="both",
        length=2.5,
        width=0.8,
        direction="out",
        color="#334e68",
    )
    for spine in ax.spines.values():
        spine.set_edgecolor("#9fb3c8")
        spine.set_linewidth(0.8)
    ax.set_facecolor("#fafbfc")


def render_cohort_row(
    fig,
    gs_grid,
    row_idx: int,
    cohort_entry: dict,
    bin_data: dict[str, list[dict[str, float]]],
    surf_img: Image.Image,
    is_top_row: bool = False,
    is_bottom_row: bool = False,
):
    """Render a single 4-panel cohort row inside a Column Block."""
    # 1. Profile Metadata Card
    ax_meta = fig.add_subplot(gs_grid[row_idx, 0])
    ax_meta.set_facecolor(cohort_entry["bg_color"])
    for spine in ax_meta.spines.values():
        spine.set_edgecolor("#cbd5e1")
        spine.set_linewidth(1.2)
    ax_meta.set_xticks([])
    ax_meta.set_yticks([])

    badge_patch = patches.FancyBboxPatch(
        (0.06, 0.72),
        0.88,
        0.22,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        facecolor=cohort_entry["badge_bg"],
        edgecolor="none",
        transform=ax_meta.transAxes,
        zorder=2,
    )
    ax_meta.add_patch(badge_patch)

    ax_meta.text(
        0.50,
        0.83,
        cohort_entry["dataset_name"],
        transform=ax_meta.transAxes,
        ha="center",
        va="center",
        color=cohort_entry["badge_fg"],
        fontsize=16.0,
        fontweight="bold",
        zorder=3,
    )

    cohort_fs = 20.0 if len(cohort_entry["cohort_name"]) <= 8 else 16.0
    ax_meta.text(
        0.50,
        0.50,
        cohort_entry["cohort_name"],
        transform=ax_meta.transAxes,
        ha="center",
        va="center",
        color="#0f172a",
        fontsize=cohort_fs,
        fontweight="bold",
        zorder=3,
    )

    ax_meta.text(
        0.50,
        0.28,
        f"(N = {cohort_entry['n_subjects']})",
        transform=ax_meta.transAxes,
        ha="center",
        va="center",
        color="#334155",
        fontsize=14.5,
        fontweight="bold",
        zorder=3,
    )

    desc_fs = 11.5 if len(cohort_entry["cohort_desc"]) <= 18 else 10.0
    ax_meta.text(
        0.50,
        0.08,
        cohort_entry["cohort_desc"],
        transform=ax_meta.transAxes,
        ha="center",
        va="center",
        color="#64748b",
        fontsize=desc_fs,
        fontstyle="italic",
        zorder=3,
    )

    # 2. Surface Render Axis
    ax_surf = fig.add_subplot(gs_grid[row_idx, 1])
    ax_surf.imshow(surf_img)
    ax_surf.axis("off")

    # 3. Left Hippocampus Stacked Bar Axis
    ax_bar_L = fig.add_subplot(gs_grid[row_idx, 2])
    render_stacked_bar_axis(
        ax_bar_L,
        bin_comps=bin_data["L"],
        hemi="L",
        show_ylabel=True,
        is_top_row=is_top_row,
        is_bottom_row=is_bottom_row,
    )

    # 4. Right Hippocampus Stacked Bar Axis
    ax_bar_R = fig.add_subplot(gs_grid[row_idx, 3])
    render_stacked_bar_axis(
        ax_bar_R,
        bin_comps=bin_data["R"],
        hemi="R",
        show_ylabel=False,
        is_top_row=is_top_row,
        is_bottom_row=is_bottom_row,
    )


def render_multidataset_wta_normalized_longaxis_stacked_2col(
    out_path: Path,
    tsv_out: Path | None = None,
    spec: GridSpec = GridSpec("2mm", "36p"),
    cohort_columns: tuple[list[dict], list[dict]] | None = None,
):
    """Main function to build and save the 2-column multi-dataset overlay figure."""
    if cohort_columns is None:
        col1_specs, col2_specs = get_multidataset_cohorts(spec)
    else:
        col1_specs, col2_specs = cohort_columns
    all_specs = col1_specs + col2_specs
    render_branch = spec.signal_branch
    panels_dir = out_path.parent / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Loading data and pre-rendering authentic 3D surfaces for {len(all_specs)} cohorts ({spec.tag})..."
    )

    is_7net = spec.is_7net
    data_col1 = []
    for c_spec in col1_specs:
        render_entry = dict(c_spec)
        render_entry["signal_branch"] = render_branch
        print(
            f"  Col 1: {render_entry['dataset_name']} {render_entry['cohort_name']} ({spec.file_tag})..."
        )
        bin_data = load_cohort_long_axis_bins(
            render_entry, n_bins=20, smoothing=spec.smoothing, is_7net=is_7net
        )
        surf_img = render_cohort_wta_surface_img(
            render_entry,
            panels_dir=panels_dir,
            smoothing=spec.smoothing,
            variant=spec.variant,
            is_7net=is_7net,
        )
        data_col1.append(
            {
                "spec": render_entry,
                "bin_data": bin_data,
                "surface_img": surf_img,
            }
        )

    data_col2 = []
    for c_spec in col2_specs:
        render_entry = dict(c_spec)
        render_entry["signal_branch"] = render_branch
        print(
            f"  Col 2: {render_entry['dataset_name']} {render_entry['cohort_name']} ({spec.file_tag})..."
        )
        bin_data = load_cohort_long_axis_bins(
            render_entry, n_bins=20, smoothing=spec.smoothing, is_7net=is_7net
        )
        surf_img = render_cohort_wta_surface_img(
            render_entry,
            panels_dir=panels_dir,
            smoothing=spec.smoothing,
            variant=spec.variant,
            is_7net=is_7net,
        )
        data_col2.append(
            {
                "spec": render_entry,
                "bin_data": bin_data,
                "surface_img": surf_img,
            }
        )

    # Optionally save TSV data
    if tsv_out is not None:
        tsv_rows = [
            "dataset\tcohort\tn_subjects\themisphere\tbin_index\tnetwork_name\tpercentage"
        ]
        for cdata in data_col1 + data_col2:
            spec_info = cdata["spec"]
            for hemi in ["L", "R"]:
                for b_idx, b_comp in enumerate(cdata["bin_data"][hemi], start=1):
                    for net, pct in b_comp.items():
                        if pct > 0:
                            tsv_rows.append(
                                f"{spec_info['dataset_name']}\t{spec_info['cohort_name']}\t{spec_info['n_subjects']}\t{hemi}\t{b_idx}\t{net}\t{pct:.2f}"
                            )
        tsv_out.parent.mkdir(parents=True, exist_ok=True)
        tsv_out.write_text("\n".join(tsv_rows) + "\n", encoding="utf-8")
        print(f"Saved stats TSV table to {tsv_out}")

    # Figure Dimensions: 46 inches wide x 27 inches high (16:9 widescreen landscape layout)
    n_rows = max(len(col1_specs), len(col2_specs))
    fig_w = 46.0
    fig_h = 27.0 * (n_rows / 5.0) if n_rows > 5 else 27.0

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=220)
    fig.patch.set_facecolor("#ffffff")

    total_subjects = sum(c["n_subjects"] for c in all_specs)
    # Suptitle & Subtitle
    fig.suptitle(
        f"Cross-Dataset Hippocampal {spec.smoothing} Normalized WTA Atlas & Long-Axis Composition"
        f"{' [7-Network]' if is_7net else ''}",
        fontsize=32.0,
        fontweight="bold",
        x=0.50,
        y=0.985,
        ha="center",
        color="#0b2447",
    )
    fig.text(
        0.50,
        0.963,
        f"Kong2022 {'Canonical 7-Network' if is_7net else 'Canonical 8-Network'} Parcellation (Normalized Top-20 Hits) Across {len(all_specs)} Cohorts (N={total_subjects})",
        fontsize=18.0,
        fontweight="medium",
        ha="center",
        color="#334e68",
    )

    top_margin = 0.932
    bottom_margin = 0.065

    # Master GridSpec: 1 row, 2 columns
    gs_main = fig.add_gridspec(
        nrows=1,
        ncols=2,
        left=0.015,
        right=0.985,
        top=top_margin,
        bottom=bottom_margin,
        wspace=0.045,
    )

    width_ratios = [0.12, 0.20, 0.34, 0.34]

    gs_left = gs_main[0, 0].subgridspec(
        nrows=n_rows,
        ncols=4,
        width_ratios=width_ratios,
        hspace=0.14,
        wspace=0.045,
    )

    gs_right = gs_main[0, 1].subgridspec(
        nrows=n_rows,
        ncols=4,
        width_ratios=width_ratios,
        hspace=0.14,
        wspace=0.045,
    )

    # Column Headers (Enlarged)
    y_col_header = top_margin + 0.005
    col1_x0 = 0.015
    col1_w = (0.985 - 0.015 - 0.045) / 2.0
    r0, r1, r2, r3 = width_ratios
    rtot = sum(width_ratios)

    p0 = col1_x0 + (r0 / (2 * rtot)) * col1_w
    p1 = col1_x0 + ((r0 + r1 / 2.0) / rtot) * col1_w
    p2 = col1_x0 + ((r0 + r1 + r2 / 2.0) / rtot) * col1_w
    p3 = col1_x0 + ((r0 + r1 + r2 + r3 / 2.0) / rtot) * col1_w

    col1_method = col1_specs[0].get("method_name") if col1_specs else None
    col2_method = col2_specs[0].get("method_name") if col2_specs else None
    wta_header = f"{spec.smoothing} Normalized WTA"
    fig.text(
        p0,
        y_col_header,
        "Cohort Profile",
        ha="center",
        va="bottom",
        fontsize=17.0,
        fontweight="bold",
        color="#102a43",
    )
    fig.text(
        p1,
        y_col_header,
        format_comparison_column_title(col1_method, wta_header),
        ha="center",
        va="bottom",
        fontsize=17.0,
        fontweight="bold",
        color="#102a43",
    )
    fig.text(
        p2,
        y_col_header,
        "Left Hippocampus (hemi-L)",
        ha="center",
        va="bottom",
        fontsize=17.0,
        fontweight="bold",
        color="#0f4c81",
    )
    fig.text(
        p3,
        y_col_header,
        "Right Hippocampus (hemi-R)",
        ha="center",
        va="bottom",
        fontsize=17.0,
        fontweight="bold",
        color="#0f4c81",
    )

    col2_x0 = col1_x0 + col1_w + 0.045
    q0 = col2_x0 + (r0 / (2 * rtot)) * col1_w
    q1 = col2_x0 + ((r0 + r1 / 2.0) / rtot) * col1_w
    q2 = col2_x0 + ((r0 + r1 + r2 / 2.0) / rtot) * col1_w
    q3 = col2_x0 + ((r0 + r1 + r2 + r3 / 2.0) / rtot) * col1_w

    fig.text(
        q0,
        y_col_header,
        "Cohort Profile",
        ha="center",
        va="bottom",
        fontsize=17.0,
        fontweight="bold",
        color="#102a43",
    )
    fig.text(
        q1,
        y_col_header,
        format_comparison_column_title(col2_method, wta_header),
        ha="center",
        va="bottom",
        fontsize=17.0,
        fontweight="bold",
        color="#102a43",
    )
    fig.text(
        q2,
        y_col_header,
        "Left Hippocampus (hemi-L)",
        ha="center",
        va="bottom",
        fontsize=17.0,
        fontweight="bold",
        color="#0f4c81",
    )
    fig.text(
        q3,
        y_col_header,
        "Right Hippocampus (hemi-R)",
        ha="center",
        va="bottom",
        fontsize=17.0,
        fontweight="bold",
        color="#0f4c81",
    )

    # Render Left Column Rows (Cohorts 0-4)
    for r_idx, cdata in enumerate(data_col1):
        is_top = r_idx == 0
        is_bot = r_idx == n_rows - 1
        render_cohort_row(
            fig=fig,
            gs_grid=gs_left,
            row_idx=r_idx,
            cohort_entry=cdata["spec"],
            bin_data=cdata["bin_data"],
            surf_img=cdata["surface_img"],
            is_top_row=is_top,
            is_bottom_row=is_bot,
        )

    # Render Right Column Rows (Cohorts 5-9)
    for r_idx, cdata in enumerate(data_col2):
        is_top = r_idx == 0
        is_bot = r_idx == n_rows - 1
        render_cohort_row(
            fig=fig,
            gs_grid=gs_right,
            row_idx=r_idx,
            cohort_entry=cdata["spec"],
            bin_data=cdata["bin_data"],
            surf_img=cdata["surface_img"],
            is_top_row=is_top,
            is_bottom_row=is_bot,
        )

    # Global Bottom Legend (7 or 8 Canonical Networks)
    if is_7net:
        legend_nets = [
            ("Default", "DMN (Default)"),
            ("Somatomotor", "SMN (Somatomotor)"),
            ("VentralAttention", "VAN (VentralAttention)"),
            ("DorsalAttention", "DAN (DorsalAttention)"),
            ("Control", "CON (Control)"),
            ("Language", "LAN (Language)"),
            ("Visual", "VN (Visual)"),
        ]
    else:
        legend_nets = [
            ("Default", "DMN (Default)"),
            ("Somatomotor", "SMN (Somatomotor)"),
            ("Auditory", "AUD (Auditory)"),
            ("VentralAttention", "VAN (VentralAttention)"),
            ("DorsalAttention", "DAN (DorsalAttention)"),
            ("Control", "CON (Control)"),
            ("Language", "LAN (Language)"),
            ("Visual", "VN (Visual)"),
        ]

    y_leg = 0.022
    fig_w_leg = 0.88
    x_leg_start = 0.06
    dx_leg = fig_w_leg / len(legend_nets)

    for idx, (net_key, net_label) in enumerate(legend_nets):
        col = NETWORK_COLORS.get(net_key, "#888888")
        x_pos = x_leg_start + idx * dx_leg

        rect = patches.Rectangle(
            (x_pos, y_leg - 0.007),
            0.016,
            0.014,
            facecolor=col,
            edgecolor="#222222",
            linewidth=1.0,
            transform=fig.transFigure,
            figure=fig,
            zorder=10,
        )
        fig.patches.append(rect)

        fig.text(
            x_pos + 0.020,
            y_leg,
            net_label,
            ha="left",
            va="center",
            fontsize=13.0,
            fontweight="bold",
            color="#102a43",
        )

    print(f"Saving high-resolution master figure to {out_path}...")
    fig.savefig(out_path, dpi=220, facecolor="#ffffff", edgecolor="none")
    plt.close(fig)
    print("Master figure saved successfully.")


def _render_grid():
    parser = argparse.ArgumentParser(
        description="Render 2-column Multi-Dataset Normalized WTA & Long-Axis Composition Overlay."
    )
    parser.add_argument(
        "--version",
        choices=["legacy", COMPARISON_VERSION],
        default="legacy",
        help="Data branch; legacy is unchanged, or use the additive HCP/JJ_TMS comparison.",
    )
    default_dir = (
        REPO_ROOT
        / "scripts/visualize/multidataset_wta_normalized_longaxis_stacked_2col"
    )
    default_out = default_dir / "multidataset_wta_normalized_longaxis_stacked_2col.png"
    default_tsv = (
        default_dir / "stats" / "multidataset_wta_normalized_longaxis_bins_stats.tsv"
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG file path (synonym for --out-png).",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=None,
        help=f"Output PNG file path (default: {default_out}).",
    )
    parser.add_argument(
        "--tsv",
        type=Path,
        default=None,
        help="Output stats TSV file path (synonym for --stats-tsv).",
    )
    parser.add_argument(
        "--stats-tsv",
        type=Path,
        default=None,
        help=f"Output stats TSV file path (default: {default_tsv}).",
    )
    parser.add_argument(
        "--smoothing",
        choices=["2mm", "4mm"],
        default="2mm",
        help="Smoothing resolution (default: 2mm)",
    )
    parser.add_argument(
        "--variant",
        choices=["36p", "22p"],
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
        default="kong2022",
        help="Atlas / network taxonomy version. Default: kong2022. Use '7net' for the canonical-7 taxonomy.",
    )
    parser.add_argument(
        "--grid",
        choices=list(GRID_VARIANTS),
        default=None,
        help="Grid variant shortcut",
    )
    args = parser.parse_args()

    if args.grid and args.signal_branch != "standard":
        parser.error(
            "--signal-branch cannot be combined with --grid; encode only legacy grid cells in --grid"
        )
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
        out_png = Path(out_arg).resolve()
    else:
        output_tag = (
            get_comparison_output_tag(spec)
            if args.version == COMPARISON_VERSION
            else spec.file_tag
        )
        out_png = (
            default_dir
            / f"multidataset_wta_normalized_longaxis_stacked_2col_{output_tag}.png"
        )

    tsv_arg = args.stats_tsv or args.tsv
    if tsv_arg:
        out_tsv = Path(tsv_arg).resolve()
    else:
        stats_dir = default_dir / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        out_tsv = (
            stats_dir
            / f"multidataset_wta_normalized_longaxis_stacked_2col_{spec.file_tag}_bins_stats.tsv"
        )

    if args.version == COMPARISON_VERSION:
        render_spec = GridSpec(
            spec.smoothing,
            spec.variant,
            signal_branch=spec.signal_branch,
            atlas="7net",
        )
        comparison = get_comparison_visual_columns(render_spec)
        comparison_tsv = (
            Path(args.stats_tsv or args.tsv).resolve()
            if (args.stats_tsv or args.tsv)
            else default_dir
            / "stats"
            / f"{COMPARISON_VERSION}_{render_spec.tag}_stats.tsv"
        )
        render_multidataset_wta_normalized_longaxis_stacked_2col(
            out_path=out_png,
            tsv_out=comparison_tsv,
            spec=render_spec,
            cohort_columns=comparison,
        )
    else:
        render_multidataset_wta_normalized_longaxis_stacked_2col(
            out_path=out_png,
            tsv_out=out_tsv,
            spec=spec,
        )


if __name__ == "__main__":
    raise SystemExit(
        "Standalone rendering has been removed. Use "
        "scripts/visualize/run_multidataset_suite.py "
        "with --figures to select the figure type; all four grids are processed in 4-way parallel by default."
    )

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import cast

import matplotlib.patches as patches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.gifti.gifti import GiftiImage
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
GLOBAL_NETWORK_STYLE_JSON = REPO_ROOT / "config" / "hipp_network_style.json"
DEFAULT_SCENE = REPO_ROOT / "config" / "wb_locked_native_view_lateral_medial.scene"

import sys

CORE_DIR = REPO_ROOT / "scripts" / "core_workflow"
VIS_LIB_DIR = REPO_ROOT / "scripts" / "visualize" / "lib"
for p in (CORE_DIR, VIS_LIB_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lib.gradient_assignment_contract import validate_gradient_assignment
from lib.surface_utils import render_locked_grid
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
from surface_panel_cache import cache_matches, panel_inputs, publish_cache


def load_network_hex_colors(path: Path = GLOBAL_NETWORK_STYLE_JSON) -> dict[str, str]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for row in payload.values():
        if isinstance(row, dict) and "name" in row and "rgba" in row:
            r, g, b, _a = row["rgba"]
            out[row["name"]] = f"#{int(r):02x}{int(g):02x}{int(b):02x}"
    return out


COLOR_MAP = load_network_hex_colors()
DEFAULT_NET_COLORS = {
    "VentralAttention": "#d95f02",
    "Somatomotor": "#1f78b4",
    "Default": "#e31a1c",
    "Auditory": "#18b7c9",
    "DorsalAttention": "#33a02c",
    "Control": "#ff7f00",
    "Visual": "#6a3d9a",
    "Language": "#b15928",
    "CinguloOpercular": "#fb9a99",
}

for k, v in DEFAULT_NET_COLORS.items():
    if k not in COLOR_MAP:
        COLOR_MAP[k] = v

ABBREV_MAP = {
    "Default": "DMN",
    "Visual": "VN",
    "Somatomotor": "SMN",
    "DorsalAttention": "DAN",
    "VentralAttention": "VAN",
    "Control": "CON",
    "Auditory": "AUD",
    "Language": "LAN",
    "Limbic": "LIM",
    "CinguloOpercular": "CO",
    "none": "None",
    "unassigned": "None",
}


def to_abbrev(net_name: str) -> str:
    return ABBREV_MAP.get(net_name, net_name)


CLUSTER_NAMES = {
    1: "Anterior (Head)",
    2: "Middle (Body)",
    3: "Posterior (Tail)",
}

CLUSTER_SHORT_NAMES = {
    1: "Ant",
    2: "Mid",
    3: "Post",
}

# Template-matching (TM) third-column contract, mirrored from
# multidataset_hipp_cortex_gradient_wta (2mm/36p standard/ctxreg only).
# The TM column uses the exact WTA TM atlas PNG and label GIFTIs to derive
# its vertex composition; HCP 7T has no TM dominant atlas and renders blank.
TM_COLUMN_TITLES = (
    ("TM Hippo Atlas", "#7c2d12"),
    ("TM Vertex Composition (hemi-L & hemi-R)", "#0f4c81"),
)
TM_COMPOSITION_LABELS = {
    1: "Anterior / VAN",
    2: "Middle / SMN",
    3: "Posterior / DMN",
}
TM_COMPOSITION_SHORT = {1: "Ant", 2: "Mid", 3: "Post"}
TM_COMPOSITION_COLORS = {1: "#dc5cff", 2: "#6396c2", 3: "#d85c6b"}


def _build_tm_column_data(
    tm_specs: list[dict],
    tm_panels_dir: Path,
    variant: str,
    signal_branch: str,
) -> list[dict]:
    """Load the exact WTA TM atlas and derive its 512-vertex composition."""
    tm_data: list[dict] = []
    for cspec in tm_specs:
        cid = str(cspec.get("id"))
        if cid == "hcp_7t":
            tm_data.append(
                {
                    "spec": cspec,
                    "surf_tm": None,
                    "tm_hemi_stats": None,
                    "has_tm": False,
                }
            )
            continue

        prefix = f"kong2022_7net_{cid}_{variant}_{signal_branch}_7net"
        hippo_dir = tm_panels_dir / f"{prefix}_hippo_tm_biglegend_2mm"
        hippo_path = hippo_dir / f"sub-100610_wb_{prefix}_tm_2mm_biglegend.png"
        if not hippo_path.is_file():
            raise FileNotFoundError(
                f"Missing canonical WTA TM hippo panel for {cid} "
                f"({signal_branch}): {hippo_path}"
            )

        hemi_stats: dict[str, dict] = {}
        for hemi in ("L", "R"):
            label_path = (
                hippo_dir / f"tm_{cid}_2mm_{signal_branch}_hemi-{hemi}.label.gii"
            )
            if not label_path.is_file():
                raise FileNotFoundError(
                    f"Missing canonical WTA TM label for {cid} hemi-{hemi}: "
                    f"{label_path}"
                )
            tm_image = cast(GiftiImage, nib.load(str(label_path)))
            labels = np.asarray(tm_image.darrays[0].data).reshape(-1)
            if labels.shape != (512,) or set(np.unique(labels).tolist()) != {1, 2, 3}:
                raise ValueError(
                    f"TM label contract mismatch for {cid} hemi-{hemi}: "
                    f"expected 512 vertices with labels 1/2/3, got {labels.shape} "
                    f"and {np.unique(labels).tolist()}"
                )
            counts = np.bincount(labels.astype(np.int64), minlength=4)
            hemi_stats[hemi] = {
                "total_vertices": 512,
                "clusters": [
                    {
                        "cluster_id": label_id,
                        "cluster_short": TM_COMPOSITION_SHORT[label_id],
                        "label": TM_COMPOSITION_LABELS[label_id],
                        "vertex_count": int(counts[label_id]),
                        "percentage": float(counts[label_id]) / 512.0 * 100.0,
                        "color": TM_COMPOSITION_COLORS[label_id],
                    }
                    for label_id in (1, 2, 3)
                ],
            }

        with Image.open(hippo_path) as handle:
            surf_tm = handle.copy()
        tm_data.append(
            {
                "spec": cspec,
                "surf_tm": surf_tm,
                "tm_hemi_stats": hemi_stats,
                "has_tm": True,
            }
        )
    return tm_data


COHORTS_COL1 = [
    {
        "dataset_name": "HCP 7T",
        "cohort_name": "HCP 7T",
        "cohort_desc": "Healthy Young Adults",
        "n_subjects": 151,
        "group_root": REPO_ROOT
        / "outputs_hcp/work/group_atlas/kong2022_7net_full_k3_fisherz",
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
        / "outputs_iowa_jj_tms/work/group_atlas/kong2022_7net_hc_full_k3_fisherz",
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
        / "outputs_iowa_jj_tms/work/group_atlas/routea_patient_full_k3_fisherz",
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
        / "outputs_stanford/work/group_atlas/routea_7net_nthc_full_k3_fisherz",
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
        / "outputs_stanford/work/group_atlas/routea_7net_tehc_full_k3_fisherz",
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
        / "outputs_stanford/work/group_atlas/routea_7net_nts_full_k3_fisherz",
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
        / "outputs_stanford/work/group_atlas/routea_7net_nts_any_mdd_full_k3_fisherz",
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
        / "outputs_stanford/work/group_atlas/routea_7net_nts_pure_mdd_full_k3_fisherz",
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
        / "outputs_stanford/work/group_atlas/routea_7net_tis_full_k3_fisherz",
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
        / "outputs_stanford/work/group_atlas/routea_7net_tis_ptsd_full_k3_fisherz",
        "bg_color": "#fffdf8",
        "badge_bg": "#8c4600",
        "badge_fg": "#ffffff",
    },
]

COHORTS_SPEC = COHORTS_COL1 + COHORTS_COL2

_FISHERZ_ASSIGNMENT_FIELDS = (
    "cluster_assignment_status",
    "group_functional_atlas_labels",
    "group_functional_atlas_label_map",
    "template_functional_atlas_label",
)


def _resolve_declared_file(
    declaration: object,
    field: str,
    *,
    context: Path,
    must_be_within: Path | None = None,
) -> Path:
    if not isinstance(declaration, str) or not declaration.strip():
        raise ValueError(f"{context}: missing declared file field {field!r}")

    declared = Path(declaration)
    if not declared.is_absolute():
        raise ValueError(
            f"{context}: declared file field {field!r} must be an absolute path; "
            f"got {declaration!r}"
        )

    resolved = declared.resolve()
    if must_be_within is not None:
        root = must_be_within.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"{context}: declared file field {field!r} resolves outside "
                f"{root}: {resolved}"
            ) from exc

    if not resolved.is_file():
        raise FileNotFoundError(
            f"{context}: declared file field {field!r} is missing: {resolved}"
        )
    return resolved


def _resolve_fisherz_assignment(
    run_dir: Path,
    manifest: dict,
    *,
    expected_signal_branch: str | None = None,
    method: str | None = None,
) -> dict[str, object]:
    """Resolve the exact Fisher-z assignment assets declared by one manifest."""
    run_root = Path(run_dir).resolve()
    cluster_labels_path = run_root / "group_labels_final.npy"
    if not cluster_labels_path.is_file():
        raise FileNotFoundError(
            "Missing canonical group_labels_final.npy for Fisher-z assignment: "
            f"{cluster_labels_path}"
        )

    if not isinstance(manifest, dict):
        raise ValueError(f"{run_root}: group atlas manifest must be a JSON object")
    signal_branch = manifest.get("signal_branch")
    if (
        expected_signal_branch is not None
        and expected_signal_branch not in SIGNAL_BRANCHES
    ):
        raise ValueError(
            f"Unsupported expected signal_branch {expected_signal_branch!r} "
            f"for {run_root}"
        )
    if expected_signal_branch == "ctxreg":
        if signal_branch != "ctxreg":
            # Renderer-identical rule shared with the comparison readiness
            # audit: pre-signal_branch HCP Kong CtxReg products are accepted
            # only under the explicit named legacy provenance contract.
            if not (
                signal_branch is None
                and has_explicit_legacy_hcp_ctxreg_provenance(
                    run_root, manifest, method=method
                )
            ):
                raise ValueError(
                    f"{run_root}/group_atlas_manifest.json records "
                    f"signal_branch={signal_branch!r}; expected 'ctxreg'"
                )
    elif signal_branch not in (None, "standard"):
        raise ValueError(
            f"{run_root}: unsupported signal_branch {signal_branch!r}; "
            "select an explicit maintained branch before rendering"
        )
    variants = manifest.get("assignment_variants")
    if not isinstance(variants, dict):
        raise ValueError(f"{run_root}: missing assignment_variants object")
    fisherz = variants.get("fisherz")
    if not isinstance(fisherz, dict):
        raise ValueError(f"{run_root}: missing assignment_variants.fisherz object")

    assignment_metric = fisherz.get("assignment_metric")
    if assignment_metric is not None and assignment_metric != "top1_fisher_z":
        raise ValueError(
            f"{run_root}: Fisher-z assignment metric must be top1_fisher_z; "
            f"got {assignment_metric!r}"
        )

    declared = {
        field: _resolve_declared_file(
            fisherz.get(field),
            field,
            context=run_root / "group_atlas_manifest.json",
            must_be_within=run_root,
        )
        for field in _FISHERZ_ASSIGNMENT_FIELDS
    }
    return {
        **declared,
        "cluster_labels_path": cluster_labels_path,
        "label_path": declared["template_functional_atlas_label"],
        "status_path": declared["cluster_assignment_status"],
        "paths": [cluster_labels_path, *declared.values()],
    }


def _load_fisherz_status(
    status_path: Path, cluster_labels: np.ndarray
) -> dict[int, dict]:
    try:
        status_data = json.loads(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Malformed Fisher-z assignment status JSON: {status_path}"
        ) from exc
    if not isinstance(status_data, dict):
        raise ValueError(f"Fisher-z assignment status must be an object: {status_path}")
    if status_data.get("assignment_metric") != "top1_fisher_z":
        raise ValueError(
            f"Fisher-z assignment status has unexpected assignment_metric at "
            f"{status_path}: {status_data.get('assignment_metric')!r}"
        )

    items = status_data.get("items")
    if not isinstance(items, list) or len(items) != 3:
        raise ValueError(
            f"Fisher-z assignment status must contain exactly three items: {status_path}"
        )

    required_text = ("top1_network", "top2_network")
    required_numeric = ("top1_fisher_z", "top2_fisher_z", "margin")
    items_by_cid: dict[int, dict] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"Malformed Fisher-z assignment item in {status_path}")
        cluster_id = item.get("cluster_id")
        if isinstance(cluster_id, bool) or not isinstance(cluster_id, int):
            raise ValueError(
                f"Malformed cluster_id in Fisher-z assignment status: {status_path}"
            )
        if cluster_id in items_by_cid:
            raise ValueError(f"Duplicate cluster_id {cluster_id} in {status_path}")
        for field in required_text:
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise ValueError(
                    f"Missing {field} for cluster {cluster_id}: {status_path}"
                )
        for field in required_numeric:
            value = item.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
            ):
                raise ValueError(
                    f"Missing or non-finite {field} for cluster {cluster_id}: {status_path}"
                )
        items_by_cid[cluster_id] = item

    if set(items_by_cid) != {1, 2, 3}:
        raise ValueError(
            f"Fisher-z assignment status cluster IDs must be exactly 1, 2, 3: {status_path}"
        )

    labels = np.asarray(cluster_labels)
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(
            f"Cluster labels must be integer-valued: {status_path.parent / 'group_labels_final.npy'}"
        )
    positive_ids = {int(value) for value in np.unique(labels) if value > 0}
    if positive_ids != {1, 2, 3}:
        raise ValueError(
            "group_labels_final.npy must contain exactly positive cluster IDs 1, 2, 3: "
            f"{status_path.parent / 'group_labels_final.npy'}"
        )
    return items_by_cid


def load_cohort_data_and_render_wb(
    cohort_entry: dict,
    temp_dir: Path,
    scene_path: Path,
    smoothing: str = "2mm",
    variant: str = "36p",
    *,
    signal_branch: str,
    panels_dir: Path | None = None,
    is_7net: bool = False,
) -> dict:
    if signal_branch not in SIGNAL_BRANCHES:
        raise ValueError(
            f"Unsupported signal_branch {signal_branch!r}; "
            f"expected one of {SIGNAL_BRANCHES}"
        )
    legacy_7net_namespace = bool(cohort_entry.get("legacy_7net_namespace", False))
    group_root = Path(cohort_entry["group_root"])
    atlas_tag = "7net_" if legacy_7net_namespace else ""
    atlas_folder = "kong2022_7net" if legacy_7net_namespace else "kong2022"
    grad_sub = Path(
        cohort_entry.get(
            "grad_subpath",
            group_root
            / f"gradient-spectral/{atlas_folder}/{smoothing}/cov_0p80/g1g2g3",
        )
    )
    ml_path = grad_sub / "hemi_L/group_atlas_manifest.json"
    mr_path = grad_sub / "hemi_R/group_atlas_manifest.json"

    if not ml_path.is_file() or not mr_path.is_file():
        missing = [str(path) for path in (ml_path, mr_path) if not path.is_file()]
        raise FileNotFoundError(
            f"Missing exact hemisphere manifest(s): {', '.join(missing)}"
        )

    ml = json.loads(ml_path.read_text(encoding="utf-8"))
    mr = json.loads(mr_path.read_text(encoding="utf-8"))
    if not isinstance(ml, dict) or not isinstance(mr, dict):
        raise ValueError(
            f"Group atlas manifests must be JSON objects: {ml_path}, {mr_path}"
        )

    left_assignment = _resolve_fisherz_assignment(
        ml_path.parent,
        ml,
        expected_signal_branch=signal_branch,
        method=cohort_entry.get("method"),
    )
    right_assignment = _resolve_fisherz_assignment(
        mr_path.parent,
        mr,
        expected_signal_branch=signal_branch,
        method=cohort_entry.get("method"),
    )
    left_validated = validate_gradient_assignment(ml_path.parent, ml, "fisherz")
    right_validated = validate_gradient_assignment(mr_path.parent, mr, "fisherz")

    # Surfaces, exact assignment labels, and spec are all manifest-declared assets.
    surf_l = _resolve_declared_file(
        ml.get("template_left_or_right_surface"),
        "template_left_or_right_surface",
        context=ml_path,
    )
    surf_r = _resolve_declared_file(
        mr.get("template_left_or_right_surface"),
        "template_left_or_right_surface",
        context=mr_path,
    )
    lbl_l_gii = left_validated["label_path"]
    lbl_r_gii = right_validated["label_path"]

    subject_value = ml.get("template_subject")
    if not isinstance(subject_value, str) or not subject_value.strip():
        raise ValueError(f"{ml_path}: missing template_subject")
    subject = subject_value.replace("sub-", "").strip()
    spec_path = _resolve_declared_file(
        ml.get("template_spec_path"),
        "template_spec_path",
        context=ml_path,
    )

    cname_clean = cohort_entry["cohort_name"].replace(" ", "_")
    ds_clean = cohort_entry["dataset_name"].replace(" ", "_")
    method_clean = str(cohort_entry.get("method", "")).replace(" ", "_")
    cache_prefix = f"{method_clean}_" if method_clean else ""
    ns = cohort_entry["n_subjects"]

    inputs = panel_inputs(
        [
            ml_path,
            mr_path,
            surf_l,
            surf_r,
            lbl_l_gii,
            lbl_r_gii,
            *left_validated["paths"],
            *right_validated["paths"],
            CORE_DIR / "lib/gradient_assignment_contract.py",
            spec_path,
            scene_path,
            Path(__file__),
            CORE_DIR / "lib/surface_utils.py",
        ],
        smoothing=smoothing,
        variant=variant,
        signal_branch=signal_branch,
        method=method_clean,
        subjects=ns,
    )

    if panels_dir is not None:
        cached_png = (
            panels_dir
            / f"wb_{cache_prefix}{ds_clean}_{cname_clean}_{atlas_tag}{smoothing}_{variant}_{signal_branch}_N{ns}_biglegend.png"
        )
        if cache_matches(cached_png, inputs):
            wb_image = Image.open(cached_png).convert("RGBA")
        else:
            wb_image = None
    else:
        cached_png = None
        wb_image = None

    if wb_image is None:
        render_out = (
            temp_dir
            / f"wb_{cache_prefix}{ds_clean}_{cname_clean}_{atlas_tag}{smoothing}_{variant}_{signal_branch}"
        )
        render = render_locked_grid(
            subject=subject,
            scene=scene_path,
            outdir=render_out,
            name=f"k3_{cache_prefix}{atlas_tag}{smoothing}_{variant}_{cname_clean}",
            title=f"{cohort_entry['dataset_name']} {cohort_entry['cohort_name']} (N={cohort_entry['n_subjects']})",
            left_labels=lbl_l_gii,
            right_labels=lbl_r_gii,
            left_surface=surf_l,
            right_surface=surf_r,
            spec_path=spec_path,
        )
        wb_biglegend_png = Path(render["biglegend_png"])
        wb_image = Image.open(wb_biglegend_png).convert("RGBA")
        if cached_png is not None:
            panels_dir.mkdir(parents=True, exist_ok=True)
            wb_image.save(cached_png)
            publish_cache(cached_png, inputs)

    # Vertex stats & status for L and R
    hemi_stats = {}
    for hemi, assignment in [("L", left_assignment), ("R", right_assignment)]:
        lbl_npy_file = assignment["cluster_labels_path"]
        lbl_arr = np.load(lbl_npy_file, allow_pickle=False).flatten()
        total_vertices = len(lbl_arr)

        unique, counts = np.unique(lbl_arr, return_counts=True)
        counts_dict = dict(zip(unique, counts))

        items_by_cid = _load_fisherz_status(assignment["status_path"], lbl_arr)

        clusters_info = []
        for cid in [1, 2, 3]:
            cnt = int(counts_dict.get(cid, 0))
            pct = (cnt / float(total_vertices)) * 100.0 if total_vertices > 0 else 0.0
            it = items_by_cid[cid]
            top1_net = it["top1_network"]
            top1_z = float(it["top1_fisher_z"])
            top2_net = it["top2_network"]
            top2_z = float(it["top2_fisher_z"])
            margin = float(it["margin"])

            clusters_info.append(
                {
                    "cluster_id": cid,
                    "cluster_name": CLUSTER_NAMES.get(cid, f"Cluster {cid}"),
                    "cluster_short": CLUSTER_SHORT_NAMES.get(cid, f"C{cid}"),
                    "vertex_count": cnt,
                    "total_vertices": total_vertices,
                    "percentage": pct,
                    "top1_network": top1_net,
                    "top1_z": top1_z,
                    "top2_network": top2_net,
                    "top2_z": top2_z,
                    "margin": margin,
                    "color": COLOR_MAP.get(top1_net, "#888888"),
                }
            )

        hemi_stats[hemi] = {
            "total_vertices": total_vertices,
            "clusters": clusters_info,
        }

    return {
        "dataset_name": cohort_entry["dataset_name"],
        "cohort_name": cohort_entry["cohort_name"],
        "cohort_desc": cohort_entry["cohort_desc"],
        "n_subjects": cohort_entry["n_subjects"],
        "signal_branch": signal_branch,
        "badge_bg": cohort_entry["badge_bg"],
        "badge_fg": cohort_entry["badge_fg"],
        "bg_color": cohort_entry["bg_color"],
        "wb_image": wb_image,
        "hemi_stats": hemi_stats,
    }


def render_cohort_row(fig, gs_row, d: dict, used_networks: set):
    """Render a single cohort row (Metadata Card + Workbench Surface + Subregion Stats Card)."""
    # 1. Left Metadata Axis
    ax_meta = fig.add_subplot(gs_row[0])
    ax_meta.axis("off")
    bg_c = d["bg_color"]
    ax_meta.add_patch(
        patches.FancyBboxPatch(
            (0.02, 0.02),
            0.96,
            0.96,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            fc=bg_c,
            ec="#cbd2d9",
            lw=1.3,
            transform=ax_meta.transAxes,
            zorder=0,
        )
    )

    # Dataset Pill
    ax_meta.text(
        0.50,
        0.82,
        f"{d['dataset_name']}",
        fontsize=21.0,
        fontweight="heavy",
        ha="center",
        va="center",
        color=d["badge_fg"],
        transform=ax_meta.transAxes,
        bbox=dict(
            boxstyle="round,pad=0.38,rounding_size=0.28",
            fc=d["badge_bg"],
            ec="none",
        ),
    )

    # Cohort Name & Sample Size
    cohort_fs = 26.0 if len(d["cohort_name"]) <= 8 else 21.0
    ax_meta.text(
        0.50,
        0.54,
        f"{d['cohort_name']}",
        fontsize=cohort_fs,
        fontweight="heavy",
        ha="center",
        va="center",
        color="#0b2447",
        transform=ax_meta.transAxes,
    )
    ax_meta.text(
        0.50,
        0.31,
        f"(N = {d['n_subjects']})",
        fontsize=19.0,
        fontweight="bold",
        ha="center",
        va="center",
        color="#334e68",
        transform=ax_meta.transAxes,
    )
    desc_fs = 14.5 if len(d["cohort_desc"]) <= 20 else 13.0
    ax_meta.text(
        0.50,
        0.12,
        f"{d['cohort_desc']}",
        fontsize=desc_fs,
        style="italic",
        fontweight="medium",
        ha="center",
        va="center",
        color="#627d98",
        transform=ax_meta.transAxes,
    )

    # 2. Middle Workbench Surface Axis - Strict Equal Aspect Ratio
    ax_wb = fig.add_subplot(gs_row[1])
    ax_wb.axis("off")
    ax_wb.imshow(d["wb_image"], aspect="equal")

    # 3. Right Statistics Card Axis
    ax_stats = fig.add_subplot(gs_row[2])
    ax_stats.axis("off")
    ax_stats.set_xlim(0, 1)
    ax_stats.set_ylim(0, 1)

    # Background card
    ax_stats.add_patch(
        patches.FancyBboxPatch(
            (0.008, 0.02),
            0.984,
            0.96,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            fc="#fcfdfe",
            ec="#cdd7e0",
            lw=1.2,
            transform=ax_stats.transAxes,
            zorder=0,
        )
    )

    # Headers for Left and Right Hippocampus
    ax_stats.text(
        0.025,
        0.90,
        "Left Hippocampus (hemi-L | 512 v)",
        fontsize=19.5,
        fontweight="heavy",
        color="#0f4c81",
        va="center",
    )
    ax_stats.text(
        0.525,
        0.90,
        "Right Hippocampus (hemi-R | 512 v)",
        fontsize=19.5,
        fontweight="heavy",
        color="#b91c1c",
        va="center",
    )

    # Horizontal Proportion Stacked Bars for L & R
    bar_y = 0.70
    bar_h = 0.155

    # Left Hemisphere Stacked Proportion Bar
    l_clusters = d["hemi_stats"]["L"]["clusters"]
    cum_pct_l = 0.0
    for c in l_clusters:
        pct = c["percentage"]
        net = c["top1_network"]
        used_networks.add(net)
        c_color = c["color"]
        seg_x = 0.025 + (cum_pct_l / 100.0) * 0.460
        seg_w = (pct / 100.0) * 0.460
        ax_stats.add_patch(
            patches.Rectangle(
                (seg_x, bar_y),
                seg_w,
                bar_h,
                facecolor=c_color,
                edgecolor="#111111",
                lw=1.1,
                zorder=2,
            )
        )
        # Label on segment (Ant 41%, Mid 33%, Post 26%)
        ax_stats.text(
            seg_x + seg_w / 2.0,
            bar_y + bar_h / 2.0,
            f"{c['cluster_short']} {pct:.1f}%",
            ha="center",
            va="center",
            fontsize=15.5,
            fontweight="heavy",
            color="#ffffff",
            path_effects=[pe.withStroke(linewidth=2.8, foreground="#111111")],
            zorder=4,
        )
        cum_pct_l += pct

    # Right Hemisphere Stacked Proportion Bar
    r_clusters = d["hemi_stats"]["R"]["clusters"]
    cum_pct_r = 0.0
    for c in r_clusters:
        pct = c["percentage"]
        net = c["top1_network"]
        used_networks.add(net)
        c_color = c["color"]
        seg_x = 0.525 + (cum_pct_r / 100.0) * 0.450
        seg_w = (pct / 100.0) * 0.450
        ax_stats.add_patch(
            patches.Rectangle(
                (seg_x, bar_y),
                seg_w,
                bar_h,
                facecolor=c_color,
                edgecolor="#111111",
                lw=1.1,
                zorder=2,
            )
        )
        # Label on segment
        ax_stats.text(
            seg_x + seg_w / 2.0,
            bar_y + bar_h / 2.0,
            f"{c['cluster_short']} {pct:.1f}%",
            ha="center",
            va="center",
            fontsize=15.5,
            fontweight="heavy",
            color="#ffffff",
            path_effects=[pe.withStroke(linewidth=2.8, foreground="#111111")],
            zorder=4,
        )
        cum_pct_r += pct

    # Detailed breakdown text lines (3 rows: C1, C2, C3)
    row_y_centers = [0.52, 0.33, 0.14]
    for c_idx in range(3):
        cl = l_clusters[c_idx]
        cr = r_clusters[c_idx]
        ry = row_y_centers[c_idx]

        # Left side line
        lbl_l_bold = (
            f"{cl['cluster_short']}: {cl['vertex_count']}/512 ({cl['percentage']:.1f}%)"
        )
        lbl_l_net = f"→ {cl['top1_network']} (z={cl['top1_z']:.3f})"
        ax_stats.text(
            0.025,
            ry + 0.040,
            lbl_l_bold,
            fontsize=17.5,
            fontweight="heavy",
            color="#0b2447",
            va="center",
        )
        ax_stats.text(
            0.025,
            ry - 0.038,
            lbl_l_net,
            fontsize=16.5,
            fontweight="heavy",
            color=cl["color"],
            va="center",
        )

        # Right side line
        lbl_r_bold = (
            f"{cr['cluster_short']}: {cr['vertex_count']}/512 ({cr['percentage']:.1f}%)"
        )
        lbl_r_net = f"→ {cr['top1_network']} (z={cr['top1_z']:.3f})"
        ax_stats.text(
            0.525,
            ry + 0.040,
            lbl_r_bold,
            fontsize=17.5,
            fontweight="heavy",
            color="#0b2447",
            va="center",
        )
        ax_stats.text(
            0.525,
            ry - 0.038,
            lbl_r_net,
            fontsize=16.5,
            fontweight="heavy",
            color=cr["color"],
            va="center",
        )


def render_tm_vertex_composition_card(ax_stats, hemi_stats: dict | None) -> None:
    """Render TM label counts using the same composition-card language."""
    ax_stats.axis("off")
    ax_stats.set_xlim(0, 1)
    ax_stats.set_ylim(0, 1)
    ax_stats.add_patch(
        patches.FancyBboxPatch(
            (0.008, 0.02),
            0.984,
            0.96,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            fc="#fcfdfe",
            ec="#cdd7e0",
            lw=1.2,
            transform=ax_stats.transAxes,
            zorder=0,
        )
    )
    if hemi_stats is None:
        ax_stats.text(
            0.50,
            0.50,
            "No TM vertex composition",
            ha="center",
            va="center",
            fontsize=17.0,
            fontweight="bold",
            color="#94a3b8",
            transform=ax_stats.transAxes,
        )
        return

    ax_stats.text(
        0.025,
        0.90,
        "Left Hippocampus (hemi-L | 512 v)",
        fontsize=17.0,
        fontweight="heavy",
        color="#0f4c81",
        va="center",
    )
    ax_stats.text(
        0.525,
        0.90,
        "Right Hippocampus (hemi-R | 512 v)",
        fontsize=17.0,
        fontweight="heavy",
        color="#b91c1c",
        va="center",
    )

    bar_y = 0.70
    bar_h = 0.155
    l_clusters = hemi_stats["L"]["clusters"]
    r_clusters = hemi_stats["R"]["clusters"]
    for side_x, clusters, bar_width in (
        (0.025, l_clusters, 0.460),
        (0.525, r_clusters, 0.450),
    ):
        cumulative = 0.0
        for cluster in clusters:
            pct = cluster["percentage"]
            seg_x = side_x + (cumulative / 100.0) * bar_width
            seg_w = (pct / 100.0) * bar_width
            ax_stats.add_patch(
                patches.Rectangle(
                    (seg_x, bar_y),
                    seg_w,
                    bar_h,
                    facecolor=cluster["color"],
                    edgecolor="#111111",
                    lw=1.1,
                    zorder=2,
                )
            )
            ax_stats.text(
                seg_x + seg_w / 2.0,
                bar_y + bar_h / 2.0,
                f"{cluster['cluster_short']} {pct:.1f}%",
                ha="center",
                va="center",
                fontsize=14.0,
                fontweight="heavy",
                color="#ffffff",
                path_effects=[pe.withStroke(linewidth=2.8, foreground="#111111")],
                zorder=4,
            )
            cumulative += pct

    row_y_centers = [0.52, 0.33, 0.14]
    for cluster_index, (left, right) in enumerate(
        zip(l_clusters, r_clusters, strict=True)
    ):
        ry = row_y_centers[cluster_index]
        for x, cluster in ((0.025, left), (0.525, right)):
            ax_stats.text(
                x,
                ry + 0.040,
                f"{cluster['cluster_short']}: {cluster['vertex_count']}/512 ({cluster['percentage']:.1f}%)",
                fontsize=15.5,
                fontweight="heavy",
                color="#0b2447",
                va="center",
            )
            ax_stats.text(
                x,
                ry - 0.038,
                f"→ {cluster['label']}",
                fontsize=14.5,
                fontweight="heavy",
                color=cluster["color"],
                va="center",
            )


def render_all_cohorts_overlay_figure(
    out_png: Path,
    tsv_out: Path | None = None,
    spec: GridSpec = GridSpec("2mm", "36p"),
    cohort_columns: tuple[list[dict], list[dict]] | None = None,
    tm_contrast_manifest: Path | str | None = None,
):
    scene_path = DEFAULT_SCENE
    if not scene_path.is_file():
        raise FileNotFoundError(f"Missing scene file: {scene_path}")

    if cohort_columns is None:
        col1_specs, col2_specs = get_multidataset_cohorts(spec)
    else:
        col1_specs, col2_specs = cohort_columns
    data_col1 = []
    data_col2 = []
    panels_dir = out_png.parent / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    use_tm = tm_contrast_manifest is not None
    if use_tm and (spec.smoothing, spec.variant) != ("2mm", "36p"):
        raise ValueError(f"TM third column requires 2mm/36p, got {spec.tag!r}")
    tm_specs: list[dict] = []
    if use_tm:
        # TM column follows the Kong2022 method cohort roster; Lynch2024
        # has no template-matching dominant atlas and is never substituted.
        kong_specs = [
            c
            for c in (col1_specs + col2_specs)
            if str(c.get("method", "")) == "kong2022_7net"
        ]
        if not kong_specs:
            raise ValueError("TM column requires Kong2022 method cohorts")
        seen: set[str] = set()
        for c in kong_specs:
            cid = str(c.get("id"))
            if cid not in seen:
                seen.add(cid)
                tm_specs.append(c)

    with tempfile.TemporaryDirectory(prefix="wb_overlay_all_") as tmp_dir:
        tmp_p = Path(tmp_dir)
        render_branch = spec.signal_branch
        print(
            f"[RENDER] Processing Column 1 ({len(col1_specs)} cohorts, {spec.file_tag})..."
        )
        for c_spec in col1_specs:
            render_entry = dict(c_spec)
            render_entry["signal_branch"] = render_branch
            print(
                f"  Col 1: {c_spec['dataset_name']} {c_spec['cohort_name']} (N={c_spec['n_subjects']})..."
            )
            d = load_cohort_data_and_render_wb(
                render_entry,
                tmp_p,
                scene_path,
                smoothing=spec.smoothing,
                variant=spec.variant,
                signal_branch=render_branch,
                panels_dir=panels_dir,
                is_7net=spec.is_7net,
            )
            data_col1.append(d)

        print(
            f"[RENDER] Processing Column 2 ({len(col2_specs)} cohorts, {spec.file_tag})..."
        )
        for c_spec in col2_specs:
            render_entry = dict(c_spec)
            render_entry["signal_branch"] = render_branch
            print(
                f"  Col 2: {c_spec['dataset_name']} {c_spec['cohort_name']} (N={c_spec['n_subjects']})..."
            )
            d = load_cohort_data_and_render_wb(
                render_entry,
                tmp_p,
                scene_path,
                smoothing=spec.smoothing,
                variant=spec.variant,
                signal_branch=render_branch,
                panels_dir=panels_dir,
                is_7net=spec.is_7net,
            )
            data_col2.append(d)

    all_data = data_col1 + data_col2

    tm_data: list[dict] = []
    has_tm = False
    if use_tm:
        import tm_group_contrasts as _tm_mod

        assert tm_contrast_manifest is not None
        # The overlay TM column consumes the WTA TM atlas/label contract —
        # never TM statistical contrasts. It validates the manifest
        # schema/panels/grid/branch contract directly; exact atlas and label
        # paths then fail closed if missing.
        tm_receipt = json.loads(Path(tm_contrast_manifest).read_text())
        if tm_receipt.get("schema") != _tm_mod.SCHEMA:
            raise ValueError(f"TM manifest schema mismatch: {tm_contrast_manifest}")
        if tm_receipt.get("panels") != list(_tm_mod.TM_PANELS):
            raise ValueError(
                f"TM manifest panel order mismatch: {tm_contrast_manifest}"
            )
        tm_grids = tm_receipt.get("grids")
        if not isinstance(tm_grids, dict) or spec.tag not in tm_grids:
            raise ValueError(
                f"TM manifest lacks grid {spec.tag!r}: {tm_contrast_manifest}"
            )
        recorded_branches = tm_receipt.get("signal_branches")
        if not isinstance(recorded_branches, list) or spec.signal_branch not in {
            str(v) for v in recorded_branches
        }:
            raise ValueError(
                f"TM manifest branch mismatch for {spec.tag!r}: {tm_contrast_manifest}"
            )
        tm_panels_dir = (
            REPO_ROOT
            / "scripts"
            / "visualize"
            / "multidataset_hipp_cortex_gradient_wta"
            / "panels"
        )
        tm_data = _build_tm_column_data(
            tm_specs,
            tm_panels_dir,
            spec.variant,
            spec.signal_branch,
        )
        has_tm = any(bool(t.get("has_tm", False)) for t in tm_data)

    # Save TSV summary table
    if tsv_out is not None:
        tsv_rows = [
            "dataset\tcohort\tn_subjects\themisphere\tcluster_id\tsubregion\tvertex_count\ttotal_vertices\tpercentage\ttop1_network\ttop1_fisher_z\ttop2_network\ttop2_fisher_z\tmargin"
        ]
        for d in all_data:
            for hemi in ["L", "R"]:
                for c in d["hemi_stats"][hemi]["clusters"]:
                    tsv_rows.append(
                        f"{d['dataset_name']}\t{d['cohort_name']}\t{d['n_subjects']}\t{hemi}\t{c['cluster_id']}\t{c['cluster_short']}\t{c['vertex_count']}\t{c['total_vertices']}\t{c['percentage']:.2f}\t{c['top1_network']}\t{c['top1_z']:.4f}\t{c['top2_network']}\t{c['top2_z']:.4f}\t{c['margin']:.4f}"
                    )
        tsv_out.parent.mkdir(parents=True, exist_ok=True)
        tsv_out.write_text("\n".join(tsv_rows) + "\n", encoding="utf-8")
        print(f"[SUCCESS] Saved vertex stats TSV to: {tsv_out}")

    # Master Figure Geometry: 2 Columns x N Rows (+ TM column when requested)
    n_rows = max(len(col1_specs), len(col2_specs))
    if has_tm:
        n_rows = max(n_rows, len(tm_data))
    fig_w = 48.0
    fig_h = 28.0 * (n_rows / 5.0) if n_rows > 5 else 28.0
    if has_tm:
        fig_w = fig_w * 1.45

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=220)
    fig.patch.set_facecolor("#ffffff")

    total_subjects = sum(c["n_subjects"] for c in col1_specs + col2_specs)
    # Suptitle
    fig.suptitle(
        f"Cross-Dataset Hippocampal {spec.smoothing} "
        f"{render_branch.title()} Functional Atlas (K=3 RouteA) & "
        f"Subregion Vertex Composition"
        f"{' [7-Network]' if spec.is_7net else ''}",
        fontsize=42.0,
        fontweight="heavy",
        x=0.50,
        y=0.988,
        ha="center",
        color="#0b2447",
    )

    # Subtitle
    fig.text(
        0.50,
        0.963,
        f"Original Workbench 3D Parcellation Overlays & Exact Subregion Vertex Counts (den-512) across {len(col1_specs + col2_specs)} Cohorts (N={total_subjects})",
        fontsize=24.0,
        fontweight="medium",
        ha="center",
        color="#334e68",
    )

    top_margin = 0.932
    bottom_margin = 0.075

    # Master GridSpec: 1 row, 2 major columns (+ TM column when requested)
    n_major = 3 if has_tm else 2
    gs_main = fig.add_gridspec(
        nrows=1,
        ncols=n_major,
        left=0.012,
        right=0.988,
        top=top_margin,
        bottom=bottom_margin,
        wspace=0.035,
    )

    width_ratios = [0.14, 0.28, 0.58]

    gs_left = gs_main[0, 0].subgridspec(
        nrows=n_rows,
        ncols=3,
        width_ratios=width_ratios,
        hspace=0.15,
        wspace=0.020,
    )

    gs_right = gs_main[0, 1].subgridspec(
        nrows=n_rows,
        ncols=3,
        width_ratios=width_ratios,
        hspace=0.15,
        wspace=0.020,
    )

    gs_tm = None
    tm_width_ratios = [0.38, 0.62]
    if has_tm:
        gs_tm = gs_main[0, 2].subgridspec(
            nrows=n_rows,
            ncols=2,
            width_ratios=tm_width_ratios,
            hspace=0.15,
            wspace=0.020,
        )

    # Column Super-Headers
    y_col_header = top_margin + 0.007
    n_gaps = (n_major - 1) * 0.035
    col_w = (0.988 - 0.012 - n_gaps) / float(n_major)
    col1_x0 = 0.012
    col2_x0 = col1_x0 + col_w + 0.035
    tm_x0 = col2_x0 + col_w + 0.035 if has_tm else None

    r0, r1, r2 = width_ratios
    rtot = sum(width_ratios)

    p0 = col1_x0 + (r0 / (2 * rtot)) * col_w
    p1 = col1_x0 + ((r0 + r1 / 2.0) / rtot) * col_w
    p2 = col1_x0 + ((r0 + r1 + r2 / 2.0) / rtot) * col_w

    col1_method = col1_specs[0].get("method_name") if col1_specs else None
    col2_method = col2_specs[0].get("method_name") if col2_specs else None
    fig.text(
        p0,
        y_col_header,
        "Cohort Profile",
        ha="center",
        va="bottom",
        fontsize=23.0,
        fontweight="heavy",
        color="#102a43",
    )
    fig.text(
        p1,
        y_col_header,
        format_comparison_column_title(col1_method, "3D Workbench Parcellation"),
        ha="center",
        va="bottom",
        fontsize=23.0,
        fontweight="heavy",
        color="#102a43",
    )
    fig.text(
        p2,
        y_col_header,
        "Subregion Vertex Composition (hemi-L & hemi-R)",
        ha="center",
        va="bottom",
        fontsize=23.0,
        fontweight="heavy",
        color="#0f4c81",
    )

    q0 = col2_x0 + (r0 / (2 * rtot)) * col_w
    q1 = col2_x0 + ((r0 + r1 / 2.0) / rtot) * col_w
    q2 = col2_x0 + ((r0 + r1 + r2 / 2.0) / rtot) * col_w

    fig.text(
        q0,
        y_col_header,
        "Cohort Profile",
        ha="center",
        va="bottom",
        fontsize=23.0,
        fontweight="heavy",
        color="#102a43",
    )
    fig.text(
        q1,
        y_col_header,
        format_comparison_column_title(col2_method, "3D Workbench Parcellation"),
        ha="center",
        va="bottom",
        fontsize=23.0,
        fontweight="heavy",
        color="#102a43",
    )
    fig.text(
        q2,
        y_col_header,
        "Subregion Vertex Composition (hemi-L & hemi-R)",
        ha="center",
        va="bottom",
        fontsize=23.0,
        fontweight="heavy",
        color="#0f4c81",
    )

    if has_tm:
        assert tm_x0 is not None and gs_tm is not None
        t_tot = sum(tm_width_ratios)
        for t_idx, (t_title, t_color) in enumerate(TM_COLUMN_TITLES):
            tx = (
                tm_x0
                + (sum(tm_width_ratios[:t_idx]) + tm_width_ratios[t_idx] / 2.0)
                / t_tot
                * col_w
            )
            fig.text(
                tx,
                y_col_header,
                t_title,
                ha="center",
                va="bottom",
                fontsize=23.0,
                fontweight="heavy",
                color=t_color,
            )
        fig.text(
            tm_x0 + col_w / 2.0,
            y_col_header + 0.017,
            f"{spec.smoothing} Template-Matching Atlas & Vertex Composition",
            ha="center",
            va="bottom",
            fontsize=19.0,
            fontweight="bold",
            color="#7c2d12",
            bbox=dict(
                boxstyle="round,pad=0.38",
                facecolor="#fef3ec",
                edgecolor="#f5c99b",
                linewidth=1.8,
            ),
        )

    used_networks = set()

    # Render Left Column (5 Cohorts)
    for idx, d in enumerate(data_col1):
        render_cohort_row(
            fig=fig,
            gs_row=[gs_left[idx, c] for c in range(3)],
            d=d,
            used_networks=used_networks,
        )

    # Render Right Column (5 Cohorts)
    for idx, d in enumerate(data_col2):
        render_cohort_row(
            fig=fig,
            gs_row=[gs_right[idx, c] for c in range(3)],
            d=d,
            used_networks=used_networks,
        )

    # Render TM Column: exact TM atlas plus its 512-vertex composition.
    if has_tm:
        assert gs_tm is not None
        for idx, t in enumerate(tm_data):
            ax_tm_hippo = fig.add_subplot(gs_tm[idx, 0])
            ax_tm_hippo.axis("off")
            if t.get("surf_tm") is not None:
                ax_tm_hippo.imshow(t["surf_tm"], aspect="equal")
            else:
                ax_tm_hippo.text(
                    0.50,
                    0.50,
                    "No TM atlas",
                    ha="center",
                    va="center",
                    fontsize=18.0,
                    fontweight="bold",
                    color="#94a3b8",
                    transform=ax_tm_hippo.transAxes,
                )
            render_tm_vertex_composition_card(
                fig.add_subplot(gs_tm[idx, 1]), t.get("tm_hemi_stats")
            )

    # Bottom Global Summary & Network Legend: Clean separate zones to prevent any text overlap
    # 1. Left summary statistics card
    ax_footer_stats = fig.add_axes([0.012, 0.010, 0.580, 0.052])
    ax_footer_stats.axis("off")
    ax_footer_stats.add_patch(
        patches.FancyBboxPatch(
            (0.00, 0.02),
            1.00,
            0.96,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            fc="#f0f4f8",
            ec="#9fb3c8",
            lw=1.4,
            transform=ax_footer_stats.transAxes,
        )
    )

    # Grand statistics across all cohorts
    l_c1 = [d["hemi_stats"]["L"]["clusters"][0]["vertex_count"] for d in all_data]
    l_c2 = [d["hemi_stats"]["L"]["clusters"][1]["vertex_count"] for d in all_data]
    l_c3 = [d["hemi_stats"]["L"]["clusters"][2]["vertex_count"] for d in all_data]
    r_c1 = [d["hemi_stats"]["R"]["clusters"][0]["vertex_count"] for d in all_data]
    r_c2 = [d["hemi_stats"]["R"]["clusters"][1]["vertex_count"] for d in all_data]
    r_c3 = [d["hemi_stats"]["R"]["clusters"][2]["vertex_count"] for d in all_data]

    n_tot = len(all_data)
    summary_text_l = (
        f"Mean Left ({n_tot} Cohorts):   Ant: {np.mean(l_c1):.1f}±{np.std(l_c1):.1f} ({np.mean(l_c1) / 5.12:.1f}%)  |  "
        f"Mid: {np.mean(l_c2):.1f}±{np.std(l_c2):.1f} ({np.mean(l_c2) / 5.12:.1f}%)  |  "
        f"Post: {np.mean(l_c3):.1f}±{np.std(l_c3):.1f} ({np.mean(l_c3) / 5.12:.1f}%)"
    )
    summary_text_r = (
        f"Mean Right ({n_tot} Cohorts):  Ant: {np.mean(r_c1):.1f}±{np.std(r_c1):.1f} ({np.mean(r_c1) / 5.12:.1f}%)  |  "
        f"Mid: {np.mean(r_c2):.1f}±{np.std(r_c2):.1f} ({np.mean(r_c2) / 5.12:.1f}%)  |  "
        f"Post: {np.mean(r_c3):.1f}±{np.std(r_c3):.1f} ({np.mean(r_c3) / 5.12:.1f}%)"
    )

    ax_footer_stats.text(
        0.02,
        0.70,
        summary_text_l,
        fontsize=16.5,
        fontweight="bold",
        color="#102a43",
        va="center",
    )
    ax_footer_stats.text(
        0.02,
        0.30,
        summary_text_r,
        fontsize=16.5,
        fontweight="bold",
        color="#102a43",
        va="center",
    )

    # 2. Right legend card
    ax_footer_leg = fig.add_axes([0.600, 0.010, 0.388, 0.052])
    ax_footer_leg.axis("off")
    ax_footer_leg.add_patch(
        patches.FancyBboxPatch(
            (0.00, 0.02),
            1.00,
            0.96,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            fc="#ffffff",
            ec="#cbd5e1",
            lw=1.4,
            transform=ax_footer_leg.transAxes,
        )
    )

    legend_patches = []
    canon_order = [
        "VentralAttention",
        "Somatomotor",
        "Default",
        "Auditory",
        "DorsalAttention",
        "Control",
        "Visual",
        "Language",
        "CinguloOpercular",
    ]
    sorted_nets = [net for net in canon_order if net in used_networks] + [
        net for net in sorted(used_networks) if net not in canon_order
    ]
    for net in sorted_nets:
        abbr = to_abbrev(net)
        legend_patches.append(
            patches.Patch(
                facecolor=COLOR_MAP.get(net, "#888888"),
                edgecolor="#111111",
                lw=1.0,
                label=f"{abbr} ({net})",
            )
        )

    ax_footer_leg.legend(
        handles=legend_patches,
        loc="center",
        ncol=min(5, len(legend_patches)),
        fontsize=15.0,
        frameon=False,
        handlelength=1.5,
        handleheight=1.1,
        columnspacing=1.5,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    n_cols = "3-column (with TM)" if has_tm else "2-column"
    print(f"[SUCCESS] Saved clean {n_cols} Workbench overlay figure to: {out_png}")


def _render_grid():
    parser = argparse.ArgumentParser(
        description="Render Workbench hippocampus overlay master figure (2 method columns; + TM column with --tm-contrast-manifest on 2mm/36p)."
    )
    parser.add_argument(
        "--version",
        choices=["legacy", COMPARISON_VERSION],
        default="legacy",
        help="Data branch; legacy is unchanged, or use the additive HCP/JJ_TMS comparison.",
    )
    default_dir = (
        REPO_ROOT
        / "scripts"
        / "visualize"
        / "multidataset_workbench_hippocampus_overlay"
    )
    default_png = default_dir / "multidataset_workbench_hippocampus_overlay.png"
    default_tsv = (
        default_dir / "stats" / "multidataset_workbench_subregion_vertex_stats.tsv"
    )

    parser.add_argument(
        "--out-png", default=None, help=f"Output PNG path (default: {default_png})"
    )
    parser.add_argument(
        "--out-tsv", default=None, help=f"Output TSV path (default: {default_tsv})"
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
        "--tm-contrast-manifest",
        default=None,
        help=(
            "Explicit TM dominant-atlas subject-level Welch manifest. "
            "When set with the 2mm/36p comparison grid, a third "
            "template-matching column is rendered."
        ),
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

    if args.out_png:
        out_png = Path(args.out_png).resolve()
    else:
        output_tag = (
            get_comparison_output_tag(spec)
            if args.version == COMPARISON_VERSION
            else spec.file_tag
        )
        out_png = (
            default_dir / f"multidataset_workbench_hippocampus_overlay_{output_tag}.png"
        )

    if args.out_tsv:
        out_tsv = Path(args.out_tsv).resolve()
    else:
        stats_dir = default_dir / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        output_tag = (
            get_comparison_output_tag(spec)
            if args.version == COMPARISON_VERSION
            else spec.file_tag
        )
        out_tsv = (
            stats_dir
            / f"multidataset_workbench_hippocampus_overlay_{output_tag}_vertex_stats.tsv"
        )

    if args.version == COMPARISON_VERSION:
        render_spec = GridSpec(
            spec.smoothing,
            spec.variant,
            signal_branch=spec.signal_branch,
            atlas="7net",
        )
        comparison = get_comparison_visual_columns(render_spec)
        render_all_cohorts_overlay_figure(
            out_png,
            out_tsv,
            spec=render_spec,
            cohort_columns=comparison,
            tm_contrast_manifest=args.tm_contrast_manifest,
        )
    else:
        render_all_cohorts_overlay_figure(out_png, out_tsv, spec=spec)


if __name__ == "__main__":
    raise SystemExit(
        "Standalone rendering has been removed. Use "
        "scripts/visualize/run_multidataset_suite.py "
        "with --figures to select the figure type; all four grids are processed in 4-way parallel by default."
    )

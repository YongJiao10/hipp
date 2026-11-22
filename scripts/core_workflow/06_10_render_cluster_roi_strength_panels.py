#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import matplotlib  # type: ignore[import-untyped]
import nibabel as nib
import numpy as np
from lib.cortex_template_profiles import (
    DEFAULT_WTA_ASSIGNMENT_METRIC,
    build_wta_pseudocluster_labels,
    compute_subject_label_roi_profiles,
    load_wta_run,
    parse_roi_registry,
    resolve_cortex_assets,
    select_top_wta_networks,
)
from lib.gifti_labels import make_named_label_gifti
from lib.kong2022_label_utils import KONG2022_DLABEL_PATH, load_kong2022_bundle
from lib.output_layout import (
    DEFAULT_CORTEX_PANELS_DIR,
    DEFAULT_GROUP_ATLAS_ROOT,
    DEFAULT_KONG2022_ROUTEA_ROOT,
    DEFAULT_KONG2022_SHARED_INPUTS_ROOT,
    DEFAULT_KONG2022_SURFACE_ROOT,
    DEFAULT_ROUTEA_GROUP_ATLAS_ROOT,
)
from lib.render_cortex_pfm_scene import render_cortex_pfm_scene
from lib.route_tags import (
    ROUTE_TAG_CHOICES,
    infer_route_tag_from_group_root,
    validate_group_atlas_root_for_route,
)
from lib.surface_utils import cluster_ids_head_to_tail, template_anterior_coordinate
from lib.template_assets import validate_template_surface_path
from lib.workflow_utils import write_json
from matplotlib.colors import Normalize  # type: ignore[import-untyped]
from nibabel.gifti.gifti import GiftiImage

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore[import-untyped]  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
METHOD = "gradient-spectral"
DEFAULT_SMOOTHINGS = ("2mm", "4mm")
DEFAULT_HEMIS = ("L", "R")
DEFAULT_GRADIENT_COMPONENT = "g1g2g3"
DEFAULT_TAU_TAG = "cov_0p80"
DEFAULT_CLUSTER_NAMES = ("Anterior", "Middle", "Posterior")
DEFAULT_NAN_RGBA = (0.65, 0.65, 0.65, 1.0)
DEFAULT_CMAP = "RdBu_r"
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "hippunfold_input"
DEFAULT_SCENE = REPO_ROOT / "config" / "manual_wb_scenes" / "cortex_manual.scene"
DEFAULT_OUTPUT_NAME = "cortex_kong2022_gradient-spectral_2mm4mm_template_strength_overview.png"
DEFAULT_WTA_ROOT = DEFAULT_GROUP_ATLAS_ROOT / "pfm"
DEFAULT_WTA_SHARED_ROOT = DEFAULT_KONG2022_SURFACE_ROOT
DEFAULT_WTA_REFERENCE_ROOT = DEFAULT_KONG2022_ROUTEA_ROOT
if not (DEFAULT_WTA_SHARED_ROOT / "sub-100610").exists():
    DEFAULT_WTA_SHARED_ROOT = DEFAULT_KONG2022_SHARED_INPUTS_ROOT / "surface"
if not (DEFAULT_WTA_REFERENCE_ROOT / "sub-100610").exists():
    DEFAULT_WTA_REFERENCE_ROOT = DEFAULT_KONG2022_SHARED_INPUTS_ROOT / "pfm"
DEFAULT_WTA_OUTPUT_NAMES = {
    "canonical7": "cortex_kong2022_wta-normalized_7net_2mm4mm_template_strength_overview.png",
}
WTA_DISPLAY_NAMES = {
    "canonical7": {
        "Default": "DMN",
        "VentralAttention": "VAN",
        "Language": "LAN",
        "Somatomotor": "SMN",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render all 400 Kong2022 cortical ROI values from gradient-spectral or "
            "normalized WTA hippocampus-to-cortex templates."
        )
    )
    parser.add_argument("--group-root", default=str(DEFAULT_ROUTEA_GROUP_ATLAS_ROOT))
    parser.add_argument("--route-tag", default="auto", choices=["auto", *ROUTE_TAG_CHOICES])
    parser.add_argument("--atlas", default="kong2022", choices=["kong2022"])
    parser.add_argument("--method", default=METHOD, choices=[METHOD, "wta-top"])
    parser.add_argument("--smoothings", nargs="+", default=list(DEFAULT_SMOOTHINGS), choices=list(DEFAULT_SMOOTHINGS))
    parser.add_argument("--hemis", nargs="+", default=list(DEFAULT_HEMIS), choices=list(DEFAULT_HEMIS))
    parser.add_argument("--gradient-component", default=DEFAULT_GRADIENT_COMPONENT)
    parser.add_argument("--tau-tag", default=DEFAULT_TAU_TAG)
    parser.add_argument("--coverage-threshold", type=float, default=0.80)
    parser.add_argument("--present-dir", default=str(DEFAULT_CORTEX_PANELS_DIR))
    parser.add_argument("--runtime-root", default=None)
    parser.add_argument("--output-png", default=None)
    parser.add_argument("--output-manifest", default=None)
    parser.add_argument("--cortex-template-subject", default="100610")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT),
                        help="Root of exact sub-<cortex-template-subject>/anat cortex rendering inputs.")
    parser.add_argument("--scene", default=str(DEFAULT_SCENE))
    parser.add_argument("--dlabel", default=str(KONG2022_DLABEL_PATH))
    parser.add_argument("--wta-root", default=str(DEFAULT_WTA_ROOT))
    parser.add_argument(
        "--wta-taxonomy",
        default="canonical7",
        choices=["canonical7"],
        help="Canonical taxonomy view for the normalized_top20_count WTA assignment.",
    )
    parser.add_argument("--shared-root", default=str(DEFAULT_WTA_SHARED_ROOT))
    parser.add_argument("--reference-root", default=str(DEFAULT_WTA_REFERENCE_ROOT))
    parser.add_argument("--mask-root", default=str(DEFAULT_WTA_SHARED_ROOT))
    return parser.parse_args()


def validate_template_profile(profile: np.ndarray, path: Path) -> np.ndarray:
    arr = np.asarray(profile, dtype=np.float32)
    if arr.ndim != 2 or int(arr.shape[1]) != 400:
        raise ValueError(f"Template profile must have shape K x 400, got {arr.shape} from {path}")
    if int(arr.shape[0]) < 1:
        raise ValueError(f"Template profile has no clusters: {path}")
    return arr


def compute_global_symmetric_scale(profiles: Iterable[np.ndarray]) -> float:
    finite_values: list[np.ndarray] = []
    for profile in profiles:
        values = np.asarray(profile, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size:
            finite_values.append(values)
    if not finite_values:
        raise ValueError("No finite template values available for color scaling")
    vmax = float(np.max(np.abs(np.concatenate(finite_values))))
    return vmax if vmax > 0.0 else 1.0


def strength_colors(
    values: np.ndarray,
    *,
    vmax: float,
    cmap_name: str = DEFAULT_CMAP,
    nan_rgba: tuple[float, float, float, float] = DEFAULT_NAN_RGBA,
) -> np.ndarray:
    if not np.isfinite(vmax) or float(vmax) <= 0.0:
        raise ValueError(f"vmax must be positive and finite, got {vmax}")
    arr = np.asarray(values, dtype=np.float64)
    norm = Normalize(vmin=-float(vmax), vmax=float(vmax), clip=True)
    cmap = plt.get_cmap(cmap_name)
    colors = np.asarray(cmap(norm(np.nan_to_num(arr, nan=0.0))), dtype=np.float64)
    zero_mask = np.isfinite(arr) & (arr == 0.0)
    colors[zero_mask, :3] = 1.0
    colors[~np.isfinite(arr), :] = np.asarray(nan_rgba, dtype=np.float64)
    return colors


def _rgba255(rgba: Iterable[float]) -> tuple[int, int, int, int]:
    vals = [int(round(float(value) * 255.0)) for value in rgba]
    return tuple(max(0, min(255, value)) for value in vals)  # type: ignore[return-value]


def resolve_cluster_names(
    *,
    manifest: dict[str, Any],
    group_labels: np.ndarray,
    anterior_coordinate: np.ndarray,
) -> dict[int, str]:
    labels = np.asarray(group_labels, dtype=np.int32)
    order = [int(value) for value in manifest.get("cluster_order_head_to_tail", [])]
    active_ids = sorted(int(value) for value in np.unique(labels) if int(value) > 0)
    if not order:
        raise ValueError("Missing cluster_order_head_to_tail for anatomical cluster naming")
    if set(order) != set(active_ids) or len(order) != len(active_ids):
        raise ValueError(
            "cluster_order_head_to_tail does not match group labels: "
            f"order={order} active_ids={active_ids}"
        )
    verified = cluster_ids_head_to_tail(labels, anterior_coordinate)
    if order != verified:
        raise ValueError(
            "cluster_order_head_to_tail disagrees with the head-to-tail order on the unfolded "
            f"hippocampal sheet: declared={order} verified={verified}"
        )
    if len(order) == 3:
        names = DEFAULT_CLUSTER_NAMES
    else:
        names = tuple(f"C{cluster_id:02d}" for cluster_id in order)
    return {cluster_id: str(name) for cluster_id, name in zip(order, names)}


def ordered_cluster_names(names_by_id: dict[int, str]) -> tuple[str, ...]:
    return tuple(str(name) for name in names_by_id.values())


def panel_order(
    *,
    smoothings: tuple[str, ...],
    hemis: tuple[str, ...],
    cluster_names: tuple[str, ...],
) -> list[tuple[str, str, str]]:
    return [
        (smoothing, hemi, cluster_name)
        for smoothing in smoothings
        for hemi in hemis
        for cluster_name in cluster_names
    ]


def _run_dir(
    *,
    group_root: Path,
    method: str,
    atlas: str,
    smoothing: str,
    tau_tag: str,
    gradient_component: str,
    hemi: str,
) -> Path:
    path = group_root / method / atlas / smoothing / tau_tag
    if method == METHOD:
        path = path / gradient_component
    return path / f"hemi_{hemi}"


def load_run(
    *,
    group_root: Path,
    smoothing: str,
    hemi: str,
    tau_tag: str,
    gradient_component: str,
) -> dict[str, Any]:
    run_dir = _run_dir(
        group_root=group_root,
        method=METHOD,
        atlas="kong2022",
        smoothing=smoothing,
        tau_tag=tau_tag,
        gradient_component=gradient_component,
        hemi=hemi,
    )
    manifest_path = run_dir / "group_atlas_manifest.json"
    profile_path = run_dir / "cluster_roi_profile.npy"
    labels_path = run_dir / "group_labels_final.npy"
    registry_path = run_dir / "cluster_roi_registry.tsv"
    for path in (manifest_path, profile_path, labels_path, registry_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required group-atlas input: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("status", "ok")) != "ok":
        raise RuntimeError(f"Group-atlas manifest is not ok: {manifest_path}")
    profile = validate_template_profile(np.load(profile_path), profile_path)
    labels = np.load(labels_path).astype(np.int32)
    surface_raw = manifest.get("template_left_or_right_surface")
    if not surface_raw:
        raise ValueError(f"Missing template_left_or_right_surface in {manifest_path}")
    surface_path = validate_template_surface_path(
        Path(str(surface_raw)),
        dataset=manifest.get("dataset"),
        run_dir=run_dir,
        hemi=str(manifest.get("hemi", "")),
    )
    names_by_id = resolve_cluster_names(
        manifest=manifest,
        group_labels=labels,
        anterior_coordinate=template_anterior_coordinate(surface_path, str(manifest.get("hemi", ""))),
    )
    registry = parse_roi_registry(registry_path)
    return {
        "run_dir": run_dir.resolve(),
        "manifest_path": manifest_path.resolve(),
        "profile_path": profile_path.resolve(),
        "labels_path": labels_path.resolve(),
        "registry_path": registry_path.resolve(),
        "manifest": manifest,
        "profile": profile,
        "labels": labels,
        "surface_path": surface_path,
        "cluster_names_by_id": names_by_id,
        "registry": registry,
    }


def _strength_label_items(
    *,
    roi_registry: list[dict[str, Any]],
    values: np.ndarray,
    vmax: float,
) -> list[tuple[int, str, tuple[int, int, int, int]]]:
    colors = strength_colors(values, vmax=vmax)
    items: list[tuple[int, str, tuple[int, int, int, int]]] = [
        (0, "Null", _rgba255(DEFAULT_NAN_RGBA))
    ]
    for idx, row in enumerate(roi_registry):
        items.append(
            (
                int(row["roi_id"]),
                str(row["roi_name"]),
                _rgba255(colors[idx]),
            )
        )
    return items


def _label_table_signature(image: GiftiImage) -> tuple[tuple[Any, ...], ...]:
    labels = []
    for label in image.labeltable.labels:
        labels.append(
            (
                int(label.key),
                str(label.label),
                int(round(float(label.red) * 255.0)),
                int(round(float(label.green) * 255.0)),
                int(round(float(label.blue) * 255.0)),
                int(round(float(label.alpha) * 255.0)),
            )
        )
    return tuple(sorted(labels))


def _label_gifti_matches(path: Path, image: GiftiImage) -> bool:
    if not path.exists():
        return False
    try:
        existing = cast(GiftiImage, nib.load(str(path)))
        return (
            len(existing.darrays) == len(image.darrays)
            and all(np.array_equal(old.data, new.data) for old, new in zip(existing.darrays, image.darrays))
            and _label_table_signature(existing) == _label_table_signature(image)
        )
    except Exception:
        return False


def write_strength_label(
    *,
    output_path: Path,
    hemi: str,
    include_values: bool,
    values: np.ndarray,
    vmax: float,
    roi_registry: list[dict[str, Any]],
    vertex_arrays: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    label_items = _strength_label_items(roi_registry=roi_registry, values=values, vmax=vmax)
    vertex_idx, parcel_ids = vertex_arrays[hemi]
    n_vertices = int(np.max(vertex_idx)) + 1
    hemi_values = np.zeros((n_vertices,), dtype=np.int32)
    if include_values:
        known_ids = {int(row["roi_id"]) for row in roi_registry}
        unknown = sorted(set(int(value) for value in np.unique(parcel_ids) if int(value) > 0) - known_ids)
        if unknown:
            raise ValueError(f"Vertex ROI IDs missing from registry for hemi-{hemi}: {unknown[:10]}")
        hemi_values[vertex_idx.astype(np.int32)] = parcel_ids.astype(np.int32)
    image = make_named_label_gifti(hemi_values, label_items)
    if _label_gifti_matches(output_path, image):
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(output_path))


def render_panel(
    *,
    runtime_root: Path,
    panel_key: tuple[str, str, str],
    cluster_id: int,
    values: np.ndarray,
    vmax: float,
    roi_registry: list[dict[str, Any]],
    vertex_arrays: dict[str, tuple[np.ndarray, np.ndarray]],
    cortex_assets: dict[str, Path],
) -> Path:
    smoothing, hemi, cluster_name = panel_key
    panel_tag = f"{smoothing}_{hemi}_{cluster_name.lower()}"
    panel_dir = runtime_root / "panels" / panel_tag
    left_label = panel_dir / f"strength_{panel_tag}.L.label.gii"
    right_label = panel_dir / f"strength_{panel_tag}.R.label.gii"
    write_strength_label(
        output_path=left_label,
        hemi="L",
        include_values=True,
        values=values,
        vmax=vmax,
        roi_registry=roi_registry,
        vertex_arrays=vertex_arrays,
    )
    write_strength_label(
        output_path=right_label,
        hemi="R",
        include_values=True,
        values=values,
        vmax=vmax,
        roi_registry=roi_registry,
        vertex_arrays=vertex_arrays,
    )

    out_scene = panel_dir / f"strength_{panel_tag}.scene"
    out_png = panel_dir / f"strength_{panel_tag}.png"
    render_inputs = (left_label, right_label, out_scene)
    needs_render = not out_png.exists()
    if not needs_render:
        output_mtime = out_png.stat().st_mtime_ns
        needs_render = any(path.stat().st_mtime_ns > output_mtime for path in render_inputs if path.exists())
    if needs_render:
        render_cortex_pfm_scene(
            source_scene=cortex_assets["scene"],
            left_surface=cortex_assets["left_inflated"], right_surface=cortex_assets["right_inflated"],
            left_label=left_label, right_label=right_label,
            left_sulc=cortex_assets["left_sulc"], right_sulc=cortex_assets["right_sulc"],
            sulc_dscalar=cortex_assets["sulc_dscalar"], dlabel=cortex_assets["dlabel"],
            out_scene=out_scene, out_png=out_png, width=1000, height=700, renderer="OSMesa",
        )
    return out_png


def overview_title(mode: str = METHOD, taxonomy: str | None = None) -> str:
    if mode == "wta-top":
        if taxonomy not in {None, "canonical7"}:
            raise ValueError(
                "WTA supports only the canonical7 normalized_top20_count view; "
                f"got {taxonomy!r}"
            )
        return "HCP 7T | Kong2022 400 ROIs | WTA normalized | 7-network"
    return "HCP 7T | Kong2022 400 ROIs | gradient-spectral"


def overview_row_label(smoothing: str, hemi: str) -> str:
    return f"{smoothing} | {hemi}"


def compose_strength_overview(
    *,
    panel_pngs: dict[tuple[str, str, str], Path],
    output_png: Path,
    colorbar_vmax: float,
    cluster_names: tuple[str, ...],
    title: str | None = None,
    column_labels: tuple[str, ...] | None = None,
) -> None:
    row_specs = (("2mm", "L"), ("2mm", "R"), ("4mm", "L"), ("4mm", "R"))
    labels = tuple(column_labels) if column_labels is not None else cluster_names
    if len(labels) != len(cluster_names):
        raise ValueError("column_labels must match cluster_names")
    fig, axes = plt.subplots(
        nrows=len(row_specs),
        ncols=len(cluster_names),
        figsize=(21.0, 17.0),
        squeeze=False,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.16, right=0.90, bottom=0.04, top=0.84, wspace=0.02, hspace=0.08)

    for row_idx, (smoothing, hemi) in enumerate(row_specs):
        for col_idx, cluster_name in enumerate(cluster_names):
            key = (smoothing, hemi, cluster_name)
            if key not in panel_pngs:
                raise KeyError(f"Missing panel for {key}")
            image = plt.imread(panel_pngs[key])
            ax = axes[row_idx, col_idx]
            ax.imshow(image, interpolation="nearest")
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(labels[col_idx], fontsize=40, fontweight="bold", pad=14)
        axes[row_idx, 0].text(
            -0.08,
            0.5,
            overview_row_label(smoothing, hemi),
            transform=axes[row_idx, 0].transAxes,
            ha="right",
            va="center",
            clip_on=False,
            fontsize=34,
            fontweight="bold",
        )

    sm = plt.cm.ScalarMappable(
        norm=Normalize(vmin=-float(colorbar_vmax), vmax=float(colorbar_vmax)),
        cmap=plt.get_cmap(DEFAULT_CMAP),
    )
    sm.set_array(np.asarray([], dtype=np.float32))
    colorbar_axis = fig.add_axes([0.925, 0.14, 0.010, 0.72])
    colorbar = fig.colorbar(sm, cax=colorbar_axis)
    colorbar.set_label(
        "mean Fisher-z hippo-cortex connectivity",
        fontsize=38,
        labelpad=26,
    )
    colorbar.ax.tick_params(labelsize=42, length=10, width=2)

    fig.suptitle(
        title or overview_title(),
        fontsize=42,
        fontweight="bold",
        y=0.975,
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_wta_overview(args: argparse.Namespace) -> dict[str, Any]:
    taxonomy = str(args.wta_taxonomy)
    group_root = Path(args.wta_root).resolve()
    run_data: dict[tuple[str, str], dict[str, Any]] = {}
    for smoothing in DEFAULT_SMOOTHINGS:
        for hemi in DEFAULT_HEMIS:
            run_data[(smoothing, hemi)] = load_wta_run(
                group_root=group_root,
                smoothing=smoothing,
                hemi=hemi,
                tau_tag=str(args.tau_tag),
                taxonomy=taxonomy,
            )

    network_names, network_counts = select_top_wta_networks(run_data.values(), n_networks=4)
    display_names = tuple(WTA_DISPLAY_NAMES[taxonomy].get(name, name) for name in network_names)
    shared_root = Path(args.shared_root).resolve()
    reference_root = Path(args.reference_root).resolve() if args.reference_root else shared_root
    mask_root = Path(args.mask_root).resolve() if args.mask_root else shared_root
    all_profiles: list[np.ndarray] = []
    for key, loaded in run_data.items():
        labels = build_wta_pseudocluster_labels(loaded, network_names)
        profile_3d, used_subjects = compute_subject_label_roi_profiles(
            subjects=loaded["subjects"],
            shared_root=shared_root,
            reference_root=reference_root,
            mask_root=mask_root,
            smoothing=key[0],
            hemi=key[1],
            group_labels=labels,
            support_mask=loaded["support_mask"],
            roi_registry=loaded["registry"],
        )
        loaded["pseudo_labels"] = labels
        loaded["profile"] = validate_template_profile(
            np.nanmean(profile_3d, axis=0).astype(np.float32),
            loaded["run_dir"] / "wta_pseudocluster_roi_profile.npy",
        )
        loaded["profile_3d"] = profile_3d
        loaded["used_subjects"] = used_subjects
        loaded["network_counts"] = {
            network: int(sum(1 for value in loaded["network_by_vertex"].values() if value == network))
            for network in network_names
        }
        all_profiles.append(loaded["profile"])

    bundle = load_kong2022_bundle(dlabel_path=Path(args.dlabel).resolve())
    vertex_arrays = bundle["vertex_arrays"]
    expected_ids = {int(row["roi_id"]) for row in bundle["roi_registry"]}
    for key, loaded in run_data.items():
        registry_ids = {int(row["roi_id"]) for row in loaded["registry"]}
        if registry_ids != expected_ids:
            raise ValueError(f"ROI registry mismatch for {key}: expected Kong2022 400 IDs")

    vmax = compute_global_symmetric_scale(all_profiles)
    present_dir = Path(args.present_dir).resolve()
    runtime_base = (
        Path(args.runtime_root).resolve()
        if args.runtime_root
        else present_dir / "runtime" / "cluster_roi_strength" / "wta_normalized"
    )
    runtime_root = runtime_base / taxonomy
    cortex_assets = resolve_cortex_assets(
        subject=str(args.cortex_template_subject),
        data_root=Path(args.data_root).resolve(),
        scene=Path(args.scene).resolve(),
        dlabel=Path(args.dlabel).resolve(),
        sulc_dir=runtime_root / "cortex_assets",
    )
    output_png = (
        Path(args.output_png).resolve()
        if args.output_png
        else present_dir / DEFAULT_WTA_OUTPUT_NAMES[taxonomy]
    )
    output_manifest = (
        Path(args.output_manifest).resolve()
        if args.output_manifest
        else output_png.with_name(f"{output_png.stem}_manifest.json")
    )

    panel_pngs: dict[tuple[str, str, str], Path] = {}
    panel_manifest: list[dict[str, Any]] = []
    template_paths: dict[str, str] = {}
    for panel_key in panel_order(
        smoothings=DEFAULT_SMOOTHINGS,
        hemis=DEFAULT_HEMIS,
        cluster_names=network_names,
    ):
        smoothing, hemi, network_name = panel_key
        loaded = run_data[(smoothing, hemi)]
        profile_path = runtime_root / "templates" / f"{smoothing}_{hemi}_wta_pseudocluster_roi_profile.npy"
        if str(profile_path) not in template_paths.values():
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(profile_path, loaded["profile"])
            template_paths[f"{smoothing}_{hemi}"] = str(profile_path.resolve())
        cluster_id = network_names.index(network_name) + 1
        panel_png = render_panel(
            runtime_root=runtime_root,
            panel_key=panel_key,
            cluster_id=cluster_id,
            values=loaded["profile"][cluster_id - 1, :],
            vmax=vmax,
            roi_registry=loaded["registry"],
            vertex_arrays=vertex_arrays,
            cortex_assets=cortex_assets,
        )
        panel_pngs[panel_key] = panel_png
        panel_manifest.append(
            {
                "smoothing": smoothing,
                "hemi": hemi,
                "cluster_id": int(cluster_id),
                "cluster_name": network_name,
                "display_name": display_names[cluster_id - 1],
                "panel_png": str(panel_png.resolve()),
                "template_profile": str(profile_path.resolve()),
                "n_subjects_used": int(len(loaded["used_subjects"])),
            }
        )

    compose_strength_overview(
        panel_pngs=panel_pngs,
        output_png=output_png,
        colorbar_vmax=vmax,
        cluster_names=network_names,
        title=overview_title("wta-top", taxonomy),
        column_labels=display_names,
    )
    manifest = {
        "status": "ok",
        "stage": "wta_pseudocluster_roi_strength_overview",
        "atlas": str(args.atlas),
        "method": "wta-top",
        "wta_network_taxonomy": taxonomy,
        "wta_assignment_variant": "normalized_top20_count",
        "wta_assignment_metric": DEFAULT_WTA_ASSIGNMENT_METRIC,
        "selected_networks": list(network_names),
        "display_networks": list(display_names),
        "selected_network_counts": network_counts,
        "tau_tag": str(args.tau_tag),
        "coverage_threshold": float(args.coverage_threshold),
        "value_unit": "fisher_z",
        "colormap": DEFAULT_CMAP,
        "scale_mode": "global_symmetric",
        "vmin": -float(vmax),
        "vmax": float(vmax),
        "nan_color": list(DEFAULT_NAN_RGBA),
        "n_rois": 400,
        "cortex_display": "bilateral L/R cortex in every source-hippocampus panel",
        "row_order": [
            {"smoothing": smoothing, "hemi": hemi}
            for smoothing in DEFAULT_SMOOTHINGS
            for hemi in DEFAULT_HEMIS
        ],
        "column_order": list(network_names),
        "display_column_order": list(display_names),
        "panel_order": [
            {"smoothing": smoothing, "hemi": hemi, "cluster_name": network_name}
            for smoothing, hemi, network_name in panel_order(
                smoothings=DEFAULT_SMOOTHINGS, hemis=DEFAULT_HEMIS, cluster_names=network_names
            )
        ],
        "panel_paths": panel_manifest,
        "template_profile_paths": template_paths,
        "input_wta_run_paths": {
            f"{smoothing}_{hemi}": str(run_data[(smoothing, hemi)]["run_dir"])
            for smoothing, hemi in run_data
        },
        "shared_root": str(shared_root),
        "reference_root": str(reference_root),
        "mask_root": str(mask_root),
        "cortex_assets": {key: str(value) for key, value in cortex_assets.items()},
        "output_png": str(output_png),
        "runtime_root": str(runtime_root),
    }
    write_json(output_manifest, manifest)
    return {"output_png": str(output_png), "manifest": str(output_manifest), "taxonomy": taxonomy}


def main() -> int:
    args = parse_args()
    if args.method == "wta-top":
        result = render_wta_overview(args)
        print(json.dumps({"status": "ok", "outputs": [result]}, indent=2))
        return 0
    group_root = Path(args.group_root).resolve()
    route_tag = infer_route_tag_from_group_root(group_root) if args.route_tag == "auto" else str(args.route_tag)
    validate_group_atlas_root_for_route(group_root, route_tag)

    smoothings = tuple(str(value) for value in args.smoothings)
    hemis = tuple(str(value) for value in args.hemis)
    if smoothings != DEFAULT_SMOOTHINGS or hemis != DEFAULT_HEMIS:
        raise ValueError("The maintained full template overview requires smoothings=2mm 4mm and hemis=L R")

    run_data: dict[tuple[str, str], dict[str, Any]] = {}
    all_profiles: list[np.ndarray] = []
    for smoothing in smoothings:
        for hemi in hemis:
            loaded = load_run(
                group_root=group_root,
                smoothing=smoothing,
                hemi=hemi,
                tau_tag=str(args.tau_tag),
                gradient_component=str(args.gradient_component),
            )
            run_data[(smoothing, hemi)] = loaded
            all_profiles.append(loaded["profile"])

    first_names = run_data[("2mm", "L")]["cluster_names_by_id"]
    cluster_names = ordered_cluster_names(first_names)
    for key, loaded in run_data.items():
        names = loaded["cluster_names_by_id"]
        if set(names.values()) != set(cluster_names):
            raise ValueError(f"Cluster anatomical names disagree for {key}: {names} vs {cluster_names}")

    vmax = compute_global_symmetric_scale(all_profiles)
    bundle = load_kong2022_bundle(dlabel_path=Path(args.dlabel).resolve())
    vertex_arrays = bundle["vertex_arrays"]
    expected_ids = {int(row["roi_id"]) for row in bundle["roi_registry"]}

    for key, loaded in run_data.items():
        registry_ids = {int(row["roi_id"]) for row in loaded["registry"]}
        if registry_ids != expected_ids:
            raise ValueError(f"ROI registry mismatch for {key}: expected Kong2022 400 IDs")

    present_dir = Path(args.present_dir).resolve()
    runtime_root = (
        Path(args.runtime_root).resolve()
        if args.runtime_root
        else present_dir / "runtime" / "cluster_roi_strength" / str(route_tag)
    )
    cortex_assets = resolve_cortex_assets(
        subject=str(args.cortex_template_subject),
        data_root=Path(args.data_root).resolve(),
        scene=Path(args.scene).resolve(),
        dlabel=Path(args.dlabel).resolve(),
        sulc_dir=runtime_root / "cortex_assets",
    )
    output_png = (
        Path(args.output_png).resolve()
        if args.output_png
        else present_dir / DEFAULT_OUTPUT_NAME
    )
    output_manifest = (
        Path(args.output_manifest).resolve()
        if args.output_manifest
        else output_png.with_name(f"{output_png.stem}_manifest.json")
    )

    panel_pngs: dict[tuple[str, str, str], Path] = {}
    panel_manifest: list[dict[str, Any]] = []
    for panel_key in panel_order(
        smoothings=smoothings,
        hemis=hemis,
        cluster_names=cluster_names,
    ):
        smoothing, hemi, cluster_name = panel_key
        loaded = run_data[(smoothing, hemi)]
        cluster_id = next(cid for cid, name in loaded["cluster_names_by_id"].items() if name == cluster_name)
        values = loaded["profile"][cluster_id - 1, :]
        panel_png = render_panel(
            runtime_root=runtime_root,
            panel_key=panel_key,
            cluster_id=cluster_id,
            values=values,
            vmax=vmax,
            roi_registry=loaded["registry"],
            vertex_arrays=vertex_arrays,
            cortex_assets=cortex_assets,
        )
        panel_pngs[panel_key] = panel_png
        panel_manifest.append(
            {
                "smoothing": smoothing,
                "hemi": hemi,
                "cluster_id": int(cluster_id),
                "cluster_name": cluster_name,
                "panel_png": str(panel_png.resolve()),
                "template_profile": str(loaded["profile_path"]),
            }
        )

    compose_strength_overview(
        panel_pngs=panel_pngs,
        output_png=output_png,
        colorbar_vmax=vmax,
        cluster_names=cluster_names,
    )
    manifest = {
        "status": "ok",
        "stage": "cluster_roi_strength_overview",
        "route_tag": route_tag,
        "atlas": str(args.atlas),
        "method": METHOD,
        "gradient_component": str(args.gradient_component),
        "tau_tag": str(args.tau_tag),
        "coverage_threshold": float(args.coverage_threshold),
        "value_unit": "fisher_z",
        "colormap": DEFAULT_CMAP,
        "scale_mode": "global_symmetric",
        "vmin": -float(vmax),
        "vmax": float(vmax),
        "nan_color": list(DEFAULT_NAN_RGBA),
        "n_rois": 400,
        "cortex_display": "bilateral L/R cortex in every source-hippocampus panel",
        "cluster_names": list(cluster_names),
        "panel_order": [
            {"smoothing": smoothing, "hemi": hemi, "cluster_name": cluster_name}
            for smoothing, hemi, cluster_name in panel_order(
                smoothings=smoothings,
                hemis=hemis,
                cluster_names=cluster_names,
            )
        ],
        "panel_paths": panel_manifest,
        "input_template_paths": {
            f"{smoothing}_{hemi}": str(run_data[(smoothing, hemi)]["profile_path"])
            for smoothing, hemi in run_data
        },
        "cluster_order_by_run": {
            f"{smoothing}_{hemi}": {
                str(cluster_id): name
                for cluster_id, name in run_data[(smoothing, hemi)]["cluster_names_by_id"].items()
            }
            for smoothing, hemi in run_data
        },
        "cortex_assets": {key: str(value) for key, value in cortex_assets.items()},
        "output_png": str(output_png),
        "runtime_root": str(runtime_root),
    }
    write_json(output_manifest, manifest)
    print(json.dumps({"status": "ok", "output_png": str(output_png), "manifest": str(output_manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

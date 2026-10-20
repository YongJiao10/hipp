"""Shared subjectwise hippocampus-to-cortex template calculations."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .fc_math import center_run_timeseries, corrcoef_rows, fisher_z_transform_fc
from .io_utils import load_run_segment_metadata, read_tsv_rows
from .kong2022_label_utils import KONG2022_DLABEL_PATH, load_kong2022_merge_mapping
from .workbench import REPO_ROOT, run_workbench


WTA_NETWORK_COLUMNS = {
    "canonical8": "canonical8",
    "canonical7": "canonical7",
}
DEFAULT_WTA_ASSIGNMENT_VARIANT = "normalized_top20_count"
DEFAULT_WTA_ASSIGNMENT_METRIC = "top20_count_normalized_by_network_roi_count"


def parse_roi_registry(path: Path) -> list[dict[str, Any]]:
    rows = read_tsv_rows(path)
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "roi_id": int(row["roi_id"]),
                "roi_name": str(row["roi_name"]),
                "hemi": str(row["hemi"]),
                "family17": str(row.get("family17", "")),
                "subregion": str(row.get("subregion", "")),
                "rank": int(row.get("rank", "0")),
                "canonical8": str(row.get("canonical8", "")),
                "canonical7": str(row.get("canonical7", "")),
            }
        )
    if len(out) != 400:
        raise ValueError(f"Expected 400 ROI registry rows, got {len(out)} from {path}")
    ids = [int(row["roi_id"]) for row in out]
    if len(set(ids)) != 400:
        raise ValueError(f"ROI registry has duplicate IDs: {path}")
    return out


def subject_reference_path(
    reference_root: Path, subject: str, atlas: str = "kong2022", filename: str = ""
) -> Path:
    """Resolve the per-subject reference file for an atlas."""
    return Path(reference_root) / f"sub-{subject}" / "reference" / str(atlas) / str(filename)


def _subject_reference_path(reference_root: Path, subject: str, filename: str) -> Path:
    """Resolve the shared Kong2022 reference used by every network taxonomy."""
    return subject_reference_path(reference_root, subject, "kong2022", filename)


def _read_subject_roi_tsv(path: Path) -> list[dict[str, Any]]:
    rows = read_tsv_rows(path)
    mapping7: dict[str, str] | None = None
    out: list[dict[str, Any]] = []
    for row in rows:
        family17 = str(row["family17"])
        canonical7 = str(row.get("canonical7", "")).strip()
        if not canonical7:
            if mapping7 is None:
                mapping7 = load_kong2022_merge_mapping(taxonomy="canonical7")
            canonical7 = mapping7[family17]
        out.append(
            {
                "parcel_id": int(row["parcel_id"]),
                "parcel_name": str(row["parcel_name"]),
                "family17": family17,
                "canonical8": str(row["canonical8"]),
                "canonical7": canonical7,
            }
        )
    return out


def wta_run_dir(
    *,
    group_root: Path,
    smoothing: str,
    hemi: str,
    tau_tag: str,
) -> Path:
    return (
        Path(group_root)
        / "wta-top"
        / "kong2022"
        / str(smoothing)
        / str(tau_tag)
        / f"hemi_{hemi}"
    )


def load_wta_run(
    *,
    group_root: Path,
    smoothing: str,
    hemi: str,
    tau_tag: str,
    taxonomy: str,
) -> dict[str, Any]:
    if taxonomy not in WTA_NETWORK_COLUMNS:
        raise ValueError(f"Unsupported WTA taxonomy: {taxonomy}")
    run_dir = wta_run_dir(
        group_root=group_root,
        smoothing=smoothing,
        hemi=hemi,
        tau_tag=tau_tag,
    )
    manifest_path = run_dir / "group_atlas_manifest.json"
    status_path = run_dir / "vertex_assignment_status.json"
    support_path = run_dir / "group_support_mask.npy"
    labels_path = run_dir / "group_functional_atlas_labels.npy"
    for path in (manifest_path, status_path, support_path, labels_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing normalized WTA input: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("status")) != "ok":
        raise RuntimeError(f"WTA group-atlas manifest is not ok: {manifest_path}")
    assignment_variant = manifest.get("default_assignment_variant")
    if assignment_variant != DEFAULT_WTA_ASSIGNMENT_VARIANT:
        raise ValueError(
            f"Expected default WTA variant {DEFAULT_WTA_ASSIGNMENT_VARIANT!r}, "
            f"got {assignment_variant!r} in {manifest_path}"
        )
    expected_column = WTA_NETWORK_COLUMNS[taxonomy]
    manifest_column = manifest.get("network_column")
    if manifest_column != expected_column:
        raise ValueError(
            f"WTA taxonomy mismatch for {run_dir}: manifest network_column="
            f"{manifest_column!r}, expected {expected_column!r}"
        )

    status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    items = list(status_payload.get("items", []))
    if status_payload.get("assignment_metric") != DEFAULT_WTA_ASSIGNMENT_METRIC:
        raise ValueError(f"Unexpected WTA assignment metric in {status_path}")
    support_mask = np.load(support_path).astype(bool)
    functional_labels = np.load(labels_path).astype(np.int32)
    if support_mask.ndim != 1 or functional_labels.ndim != 1:
        raise ValueError(f"WTA support/label arrays must be 1D in {run_dir}")
    if support_mask.size != functional_labels.size:
        raise ValueError(f"WTA support/label size mismatch in {run_dir}")

    network_by_vertex: dict[int, str] = {}
    for item in items:
        vertex_index = int(item["vertex_index"])
        network = str(item.get("top1_network", "unassigned"))
        if vertex_index in network_by_vertex and network_by_vertex[vertex_index] != network:
            raise ValueError(f"Conflicting WTA labels for vertex {vertex_index} in {status_path}")
        network_by_vertex[vertex_index] = network
    support_indices = np.flatnonzero(support_mask).astype(np.int32)
    missing = [int(index) for index in support_indices if int(index) not in network_by_vertex]
    if missing:
        raise ValueError(f"WTA status is missing supported vertices: {missing[:10]}")

    registry_raw = manifest.get("cluster_roi_registry")
    if not registry_raw:
        raise ValueError(f"WTA manifest is missing cluster_roi_registry: {manifest_path}")
    registry_path = Path(str(registry_raw))
    if not registry_path.exists():
        raise FileNotFoundError(f"Missing WTA registry from manifest: {registry_path}")
    registry = parse_roi_registry(registry_path)
    subjects = [str(subject).removeprefix("sub-") for subject in manifest.get("subjects", [])]
    if not subjects:
        raise ValueError(f"No subjects in WTA manifest: {manifest_path}")
    return {
        "run_dir": run_dir.resolve(),
        "manifest_path": manifest_path.resolve(),
        "status_path": status_path.resolve(),
        "support_path": support_path.resolve(),
        "labels_path": labels_path.resolve(),
        "manifest": manifest,
        "status_items": items,
        "network_by_vertex": network_by_vertex,
        "support_mask": support_mask,
        "functional_labels": functional_labels,
        "subjects": subjects,
        "registry": registry,
        "registry_path": registry_path.resolve(),
    }


def select_top_wta_networks(
    runs: Iterable[dict[str, Any]],
    *,
    n_networks: int = 4,
) -> tuple[tuple[str, ...], dict[str, int]]:
    counts: Counter[str] = Counter()
    for run in runs:
        for item in run["status_items"]:
            network = str(item.get("top1_network", "unassigned"))
            if network not in {"", "none", "unassigned"}:
                counts[network] += 1
    if len(counts) < int(n_networks):
        raise ValueError(f"Only {len(counts)} WTA networks available; need {n_networks}")
    ordered = tuple(
        network
        for network, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[: int(n_networks)]
    )
    return ordered, {network: int(counts[network]) for network in ordered}


def build_wta_pseudocluster_labels(
    run: dict[str, Any],
    network_names: tuple[str, ...],
) -> np.ndarray:
    labels = np.zeros(run["support_mask"].size, dtype=np.int32)
    network_to_label = {network: index + 1 for index, network in enumerate(network_names)}
    for vertex_index, network in run["network_by_vertex"].items():
        label = network_to_label.get(str(network), 0)
        if label and bool(run["support_mask"][int(vertex_index)]):
            labels[int(vertex_index)] = int(label)
    return labels


def compute_subject_label_roi_profiles(
    *,
    subjects: list[str],
    shared_root: Path,
    reference_root: Path | None = None,
    mask_root: Path | None = None,
    smoothing: str,
    hemi: str,
    group_labels: np.ndarray,
    support_mask: np.ndarray,
    roi_registry: list[dict[str, Any]],
) -> tuple[np.ndarray, list[str]]:
    """Compute subjectwise label-to-400-ROI Fisher-z profiles."""
    reference_root = reference_root or shared_root
    mask_root = mask_root or shared_root
    labels = np.asarray(group_labels, dtype=np.int32)
    support = np.asarray(support_mask, dtype=bool)
    if labels.ndim != 1 or support.ndim != 1 or labels.size != support.size:
        raise ValueError(f"Invalid pseudo-cluster labels/support: {labels.shape}, {support.shape}")
    k_final = int(np.max(labels))
    if k_final < 1:
        raise ValueError("Pseudo-cluster labels contain no active network")
    col_by_roi_id = {int(row["roi_id"]): index for index, row in enumerate(roi_registry)}
    reg_by_id = {int(row["roi_id"]): row for row in roi_registry}
    profiles: list[np.ndarray] = []
    used_subjects: list[str] = []

    for subject in subjects:
        ts_path = shared_root / f"sub-{subject}" / "surface" / smoothing / f"sub-{subject}_hemi-{hemi}_timeseries.npy"
        roi_path = _subject_reference_path(reference_root, subject, "cortex_roi_parcel_timeseries.npy")
        roi_tsv_path = _subject_reference_path(reference_root, subject, "cortex_roi_parcels.tsv")
        valid_path = mask_root / f"sub-{subject}" / "surface" / "tsnr_gate" / f"sub-{subject}_hemi-{hemi}_valid_mask.npy"
        if not (ts_path.exists() and roi_path.exists() and roi_tsv_path.exists() and valid_path.exists()):
            continue

        ts = np.load(ts_path).astype(np.float32)
        roi = np.load(roi_path).astype(np.float32)
        roi_rows = _read_subject_roi_tsv(roi_tsv_path)
        valid_mask = np.load(valid_path).astype(bool)
        if ts.ndim != 2 or roi.ndim != 2 or ts.shape[1] != roi.shape[1]:
            continue
        if ts.shape[0] != labels.size or valid_mask.size != labels.size or roi.shape[0] != len(roi_rows):
            continue

        _run_ids, run_lengths = load_run_segment_metadata(roi_path.parent / "reference_summary.json")
        ts = center_run_timeseries(ts, run_lengths)
        valid_vertices = support & valid_mask & np.isfinite(ts).all(axis=1)
        valid_vertices[valid_vertices] = np.std(
            ts[valid_vertices].astype(np.float64), axis=1, ddof=1
        ) > 0
        pseudo_cluster_ts = np.full((k_final, ts.shape[1]), np.nan, dtype=np.float32)
        for cluster_id in range(1, k_final + 1):
            vertex_mask = (labels == cluster_id) & valid_vertices
            if int(np.sum(vertex_mask)):
                pseudo_cluster_ts[cluster_id - 1, :] = np.mean(ts[vertex_mask, :], axis=0).astype(np.float32)

        corr = corrcoef_rows(pseudo_cluster_ts, roi, run_lengths=run_lengths)
        fisher = fisher_z_transform_fc(corr)
        scatter = np.full((k_final, 400), np.nan, dtype=np.float32)
        for local_col, row in enumerate(roi_rows):
            roi_id = int(row["parcel_id"])
            registry_row = reg_by_id.get(roi_id)
            if registry_row is None:
                raise KeyError(f"Unexpected ROI id={roi_id} in sub-{subject}")
            if str(row["parcel_name"]) != str(registry_row["roi_name"]):
                raise ValueError(f"ROI name mismatch for sub-{subject}, roi_id={roi_id}")
            scatter[:, int(col_by_roi_id[roi_id])] = fisher[:, int(local_col)]
        profiles.append(scatter)
        used_subjects.append(subject)

    if not profiles:
        raise RuntimeError(
            f"No valid subject profiles for WTA pseudo-clusters, smoothing={smoothing}, hemi={hemi}"
        )
    return np.stack(profiles, axis=0).astype(np.float32), used_subjects


compute_subject_cluster_roi_profiles = compute_subject_label_roi_profiles


def resolve_cortex_assets(
    subject: str,
    data_root: Path,
    assets_root: Path,
    scene: Path,
    dlabel: Path = KONG2022_DLABEL_PATH,
    *,
    public_root: Path | None = None,
) -> dict[str, Path]:
    subject = str(subject).removeprefix("sub-")
    anat = data_root / f"sub-{subject}" / "anat"
    asset_subject = assets_root / f"sub-{subject}" / "assets"
    pub = (public_root or (REPO_ROOT / "external")).resolve()
    fastans_tpl = pub / "FASTANS" / "resources" / "templates"

    def first_existing(paths: Iterable[Path]) -> Path | None:
        for path in paths:
            if path.exists():
                return path.resolve()
        return None

    left_inflated = first_existing((
        anat / f"sub-{subject}_hemi-L_space-fsLR_den-32k_desc-MSMAll_inflated.surf.gii",
        assets_root / f"sub-{subject}_hemi-L_space-fsLR_den-32k_desc-MSMAll_inflated.surf.gii",
        pub / f"sub-{subject}_hemi-L_space-fsLR_den-32k_desc-MSMAll_inflated.surf.gii",
    ))
    right_inflated = first_existing((
        anat / f"sub-{subject}_hemi-R_space-fsLR_den-32k_desc-MSMAll_inflated.surf.gii",
        assets_root / f"sub-{subject}_hemi-R_space-fsLR_den-32k_desc-MSMAll_inflated.surf.gii",
        pub / f"sub-{subject}_hemi-R_space-fsLR_den-32k_desc-MSMAll_inflated.surf.gii",
    ))
    sulc_dscalar = first_existing((
        anat / f"sub-{subject}_space-fsLR_den-32k_desc-MSMAll_sulc.dscalar.nii",
        assets_root / f"sub-{subject}_space-fsLR_den-32k_desc-MSMAll_sulc.dscalar.nii",
        pub / f"sub-{subject}_space-fsLR_den-32k_desc-MSMAll_sulc.dscalar.nii",
    ))
    left_sulc = first_existing((
        asset_subject / f"sub-{subject}_hemi-L_space-fsLR_den-32k_desc-MSMAll_sulc.func.gii",
        assets_root / f"sub-{subject}_hemi-L_space-fsLR_den-32k_desc-MSMAll_sulc.func.gii",
        fastans_tpl / "CORTEX_LEFT.32k_fs_LR..func.gii",
    ))
    right_sulc = first_existing((
        asset_subject / f"sub-{subject}_hemi-R_space-fsLR_den-32k_desc-MSMAll_sulc.func.gii",
        assets_root / f"sub-{subject}_hemi-R_space-fsLR_den-32k_desc-MSMAll_sulc.func.gii",
        fastans_tpl / "CORTEX_RIGHT.32k_fs_LR.func.gii",
    ))
    if (left_sulc is None or right_sulc is None) and sulc_dscalar is not None:
        cache_dir = (REPO_ROOT / "tmp" / "cortex_template_cache" / f"sub-{subject}").resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        left_gen = cache_dir / f"sub-{subject}_hemi-L_space-fsLR_den-32k_desc-MSMAll_sulc.func.gii"
        right_gen = cache_dir / f"sub-{subject}_hemi-R_space-fsLR_den-32k_desc-MSMAll_sulc.func.gii"
        if not left_gen.exists() or not right_gen.exists():
            run_workbench([
                "-cifti-separate", str(sulc_dscalar), "COLUMN",
                "-metric", "CORTEX_LEFT", str(left_gen),
                "-metric", "CORTEX_RIGHT", str(right_gen),
            ])
        left_sulc = left_sulc or left_gen.resolve()
        right_sulc = right_sulc or right_gen.resolve()

    required: dict[str, Path | None] = {
        "left_inflated": left_inflated,
        "right_inflated": right_inflated,
        "left_sulc": left_sulc,
        "right_sulc": right_sulc,
        "sulc_dscalar": sulc_dscalar,
        "scene": scene.resolve() if scene.exists() else None,
        "dlabel": dlabel.resolve() if dlabel.exists() else None,
    }
    missing = [name for name, path in required.items() if path is None]
    if missing:
        raise FileNotFoundError(
            f"Missing cortex rendering assets for sub-{subject}: {', '.join(missing)}"
        )
    return {name: path for name, path in required.items() if path is not None}

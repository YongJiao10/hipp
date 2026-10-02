"""Fail-closed lineage checks for active Kong400 gradient-spectral products.

Historical references are descriptive only. Active parents and scientific
artifacts must be non-archive files; no missing-source fallback is supported.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

KONG2022_ROI_COUNT = 400


def gradient_computation_provenance(summary, *, fixed_cluster_reuse):
    """Describe this stage, without rewriting the upstream producer's history."""
    fields = {"gradient_recomputed": False,
              "clustering_recomputed": fixed_cluster_reuse is None,
              "k_selection_recomputed": fixed_cluster_reuse is None}
    if summary is not None and fixed_cluster_reuse is None:
        source = summary.get("hippomaps_gradient_manifest")
        if not source:
            raise ValueError("Consumed upstream gradients require a source manifest")
        fields.update(gradient_computation="consumed_upstream",
                      gradient_source_manifest=str(active_path(source)))
    return fields


def active_path(path: Path | str) -> Path:
    """Reject archive inputs, including symlinks, before reading their contents."""
    resolved = Path(path).resolve()
    if any(part.lower().startswith(("_archive", "archive")) for part in resolved.parts):
        raise ValueError(f"Active lineage cannot consume an archive artifact: {path} -> {resolved}")
    return resolved


def load_json(path: Path | str) -> dict[str, Any]:
    path = active_path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a manifest object: {path}")
    return value


def _array(path: Path) -> np.ndarray:
    return np.load(active_path(path), mmap_mode="r", allow_pickle=False)


def _historical_source(value: Any, path: Path) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"historical_source must be an object: {path}")
    kinds = {
        "canonical8_fixed_cluster_reuse": "canonical8",
        "canonical7_in_place_fixed_cluster_reuse": "canonical7",
    }
    if value.get("kind") not in kinds or value.get("source_network_taxonomy") != kinds[value["kind"]]:
        raise ValueError(f"Unsupported historical_source kind/taxonomy: {path}")
    if value.get("archive_role") != "historical_only_not_runtime_input":
        raise ValueError(f"historical_source must explicitly be historical_only_not_runtime_input: {path}")
    identities = (value.get("source_run_id"), value.get("historical_reference_path"))
    if not any(isinstance(v, str) and v.strip() for v in identities):
        raise ValueError(f"historical_source requires a source identity: {path}")
    # Deliberately do not open or test existence of historical reference paths.


def _local_artifacts(path: Path, manifest: dict[str, Any]) -> None:
    run = path.parent
    # These are required even if the source is historical.
    profile = _array(run / "cluster_roi_profile.npy")
    labels = _array(run / "group_labels_final.npy")
    raw_mask = _array(run / "group_support_mask.npy")
    k = manifest.get("chosen_k_final")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(f"Missing/invalid chosen_k_final: {path}")
    if (labels.ndim != 1 or labels.shape != raw_mask.shape
            or not np.issubdtype(labels.dtype, np.integer)
            or not np.isin(raw_mask, [0, 1]).all()):
        raise ValueError(f"Labels/support shape or type mismatch: {path}")
    mask = np.asarray(raw_mask, dtype=bool)
    if not mask.any() or set(np.unique(labels[mask])) != set(range(1, k + 1)):
        raise ValueError(f"Labels do not contain exactly the selected K clusters: {path}")
    if np.any(labels[~mask] != 0):
        raise ValueError(f"Labels outside the support mask must be zero: {path}")
    if profile.shape != (k, KONG2022_ROI_COUNT):
        raise ValueError(f"400-ROI cluster profile shape mismatch: {path} shape={profile.shape}")
    density = manifest.get("density_evidence")
    if not isinstance(density, dict) or density.get("n_vertices") != labels.size:
        raise ValueError(f"Missing/mismatched density vertex count: {path}")
    subjects = manifest.get("subjects")
    if (not isinstance(subjects, list) or not subjects
            or not all(isinstance(s, str) and s for s in subjects)
            or len(set(subjects)) != len(subjects)
            or manifest.get("n_subjects") != len(subjects)):
        raise ValueError(f"Missing/duplicate/mismatched cohort subjects: {path}")
    summary = load_json(run / "k_selection_summary.json")
    rows = summary.get("rows")
    if not isinstance(rows, list) or not any(row.get("k") == k for row in rows if isinstance(row, dict)):
        raise ValueError(f"K selection summary has no selected K={k} evidence: {path}")
    if "chosen_k_final" in summary and summary["chosen_k_final"] != k:
        raise ValueError(f"K selection summary disagrees with manifest: {path}")
    features = _array(run / "group_gradient_features_active.npy")
    source = load_json(run / "group_gradient_feature_summary.json")
    gradient = load_json(source["hippomaps_gradient_manifest"])
    outputs = gradient["outputs"]
    full = _array(Path(outputs["hippomaps_gradients_full"]))
    fc = _array(Path(outputs["group_mean_fisherz_fc_active"]))
    # Older producers record group_fc_shape, not the redundant feature_count.
    # Always inspect the declared FC itself; any additional claim must agree.
    if (("feature_count" in gradient and gradient["feature_count"] != 400)
            or fc.ndim != 2 or fc.shape[1] != 400
            or gradient.get("group_fc_shape") != list(fc.shape)):
        raise ValueError(f"Upstream gradient must prove actual 400-ROI FC: {path}")
    if full.ndim != 2 or full.shape[0] != labels.size:
        raise ValueError(f"Upstream gradient vertex count mismatch: {path}")
    components = source.get("gradient_components")
    if (not isinstance(components, list) or not components
            or not all(isinstance(c, int) and 1 <= c <= full.shape[1] for c in components)
            or len(set(components)) != len(components)):
        raise ValueError(f"Invalid selected gradient components: {path}")
    selected = full[mask][:, [c - 1 for c in components]]
    if features.shape != selected.shape or not np.allclose(features, selected, equal_nan=True, atol=1e-6, rtol=1e-6):
        raise ValueError(f"Selected gradients differ from declared upstream: {path}")
    if source.get("hippomaps_gradients_full") and active_path(source["hippomaps_gradients_full"]) != active_path(outputs["hippomaps_gradients_full"]):
        raise ValueError(f"Gradient summary/output paths disagree: {path}")


def validate_group_atlas_lineage(
    manifest_path: Path | str,
    *,
    require_complete_local_artifacts: bool = True,
    visited_paths: set[Path] | None = None,
) -> dict[str, Any]:
    """Validate explicit active-parent, recomputed, or historical-reuse mode."""
    path = active_path(manifest_path)
    manifest = load_json(path)
    if manifest.get("status") != "ok":
        raise ValueError(f"Manifest status is not ok: {path}")
    if (manifest.get("atlas"), manifest.get("method"), manifest.get("network_taxonomy")) != (
        "kong2022", "gradient-spectral", "canonical7"
    ):
        raise ValueError(f"Expected active canonical7 Kong gradient-spectral manifest: {path}")
    visited = set(visited_paths or ())
    if path in visited:
        raise ValueError(f"Cyclic manifest lineage detected: {path}")
    visited.add(path)
    reuse = manifest.get("fixed_cluster_reuse")
    if reuse is not None and not isinstance(reuse, dict):
        raise ValueError(f"Invalid fixed_cluster_reuse provenance: {path}")
    parent_refs = [manifest.get("source_group_atlas_manifest")]
    if reuse:
        parent_refs.append(reuse.get("source_manifest"))
        if not reuse.get("source_manifest"):
            raise ValueError(f"Fixed reuse lacks an explicit source_manifest: {path}")
    parents = {active_path(p) for p in parent_refs if p is not None}
    if path in parents:
        raise ValueError(f"Self-referencing source manifest: {path}")
    if len(parents) > 1:
        raise ValueError(f"Conflicting active parent declarations: {path}")
    history = manifest.get("historical_source")
    consumed = manifest.get("gradient_computation") == "consumed_upstream"
    if consumed:
        if (manifest.get("gradient_recomputed") is not False
                or manifest.get("clustering_recomputed") is not True
                or manifest.get("k_selection_recomputed") is not True
                or reuse or history or parents):
            raise ValueError(f"Consumed upstream gradients conflict with computation provenance: {path}")
        summary = load_json(path.parent / "group_gradient_feature_summary.json")
        declared = manifest.get("gradient_source_manifest")
        if not declared or active_path(declared) != active_path(summary["hippomaps_gradient_manifest"]):
            raise ValueError(f"Gradient source differs from actual feature summary: {path}")
        load_json(declared)
    recomputed = manifest.get("gradient_recomputed") is True and manifest.get("clustering_recomputed") is True
    if recomputed and (reuse or history or parents):
        raise ValueError(f"Conflict in manifest provenance: recomputed and fixed reuse: {path}")
    if parents:
        parent = parents.pop()
        if not parent.is_file():
            raise FileNotFoundError(f"Declared parent group atlas manifest does not exist: {parent} (from {path})")
        validate_group_atlas_lineage(parent, require_complete_local_artifacts=require_complete_local_artifacts,
                                     visited_paths=visited)
        parent_manifest = load_json(parent)
        for field in ("subjects", "n_subjects", "smoothing", "hemi", "coverage_threshold", "chosen_k_final"):
            if manifest.get(field) != parent_manifest.get(field):
                raise ValueError(f"Active parent {field} mismatch: {path} -> {parent}")
        if require_complete_local_artifacts:
            for name in ("group_labels_final.npy", "group_support_mask.npy", "group_gradient_features_active.npy"):
                a, b = _array(path.parent / name), _array(parent.parent / name)
                if a.shape != b.shape or not np.array_equal(a, b, equal_nan=True):
                    raise ValueError(f"Fixed parent artifact mismatch ({name}): {path} -> {parent}")
        mode = "active_parent_declared"
    elif history is not None:
        _historical_source(history, path)
        if manifest.get("gradient_recomputed") is not False or manifest.get("clustering_recomputed") is not False:
            raise ValueError(f"Historical reuse must explicitly record non-recomputed flags: {path}")
        mode = "historical_fixed_cluster_reuse"
    elif consumed:
        mode = "upstream_gradients_consumed"
    elif recomputed:
        mode = "independent_recomputed"
    else:
        raise ValueError(f"Missing explicit computation or historical provenance: {path}")
    if require_complete_local_artifacts:
        _local_artifacts(path, manifest)
    return {"status": "ok", "path": str(path), "mode": mode,
            "network_taxonomy": "canonical7",
            "source_manifest": str(parent) if mode == "active_parent_declared" else None}


def repair_self_referencing_manifest(
    manifest_path: Path | str, *, historical_source: dict[str, Any] | None = None,
) -> bool:
    """Explicit migration only; never infer history from a missing old path."""
    path = active_path(manifest_path)
    manifest = load_json(path)
    src = manifest.get("source_group_atlas_manifest")
    if src is None or active_path(src) != path:
        return False
    if historical_source is None:
        raise ValueError(f"Repair requires an explicit historical_source selected by the caller: {path}")
    _historical_source(historical_source, path)
    _local_artifacts(path, manifest)
    manifest["source_group_atlas_manifest"] = None
    manifest["source_group_run"] = None
    manifest["target_group_run"] = None
    manifest["historical_source"] = dict(historical_source)
    manifest["gradient_recomputed"] = False
    manifest["clustering_recomputed"] = False
    if isinstance(manifest.get("taxonomy_transform"), dict):
        manifest["taxonomy_transform"]["source_group_manifest"] = historical_source.get("historical_reference_path")
        manifest["taxonomy_transform"]["source_manifest_network_taxonomy"] = historical_source["source_network_taxonomy"]
    manifest["source_path_repair"] = {"status": "explicit_historical_fixed_cluster_reuse",
                                     "self_reference_removed": True}
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return True

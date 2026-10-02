"""Descriptive comparison of the completed Stanford group K=3 atlases.

This module is the group-atlas geometry family of the unified hippocampal
workflow.  It reads only the configured Stanford group maps, their manifests,
and the consumed template surfaces named by those manifests.  The resulting
comparisons are descriptive map/specification diagnostics, not inferential
tests of FC or of clinical groups.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.metrics import adjusted_rand_score


REFERENCE_GROUP = "HC"
FACTORIAL_GROUP_ORDER = ("NTHC", "TEHC", "NTS", "TIS")
NESTED_GROUP_ORDER = ("NTS Any MDD", "NTS Pure MDD", "TIS PTSD")
GROUP_ORDER = (
    REFERENCE_GROUP,
    "NTHC",
    "TEHC",
    "NTS",
    "NTS Any MDD",
    "NTS Pure MDD",
    "TIS",
    "TIS PTSD",
)
GROUP_SLUG = {group: group.lower().replace(" ", "_") for group in GROUP_ORDER}
VARIANT_ORDER = ("plain", "weighted")
SMOOTHING_ORDER = (2, 4)
HEMISPHERE_ORDER = ("L", "R")
HC_COMPARISON_ORDER = tuple(
    (group, REFERENCE_GROUP) for group in GROUP_ORDER if group != REFERENCE_GROUP
)
# Compatibility name for plotting and callers that used the former pair
# registry.  It now intentionally contains HC-only pairs.
PAIR_ORDER = HC_COMPARISON_ORDER
BOUNDARY_PAIRS = ((1, 2), (2, 3))


@dataclass(frozen=True)
class LoadedAtlas:
    """One validated group map plus its consumed geometry and provenance."""

    group: str
    variant: str
    smoothing_mm: int
    hemisphere: str
    directory: Path
    manifest_path: Path
    labels_path: Path
    labels: np.ndarray
    vertices: np.ndarray
    faces: np.ndarray
    manifest: Mapping[str, Any]
    density_evidence: Mapping[str, Any]

    @property
    def key(self) -> tuple[str, str, int, str]:
        return (self.group, self.variant, self.smoothing_mm, self.hemisphere)

    @property
    def consumed_geometry_path(self) -> str:
        return str(Path(self.manifest["template_left_or_right_surface"]).resolve())


def _forbidden_path_gate(path: Path) -> None:
    """Keep this analysis explicitly outside beta/CON/native/archive branches."""

    lowered = {part.lower() for part in path.resolve().parts}
    forbidden = {"archive", "beta", "con", "native"}
    hit = sorted(lowered & forbidden)
    if hit:
        raise ValueError(f"external atlas analysis scope rejects path component(s): {hit}")


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    expected_n_vertices: int | None = None,
) -> dict[str, Any]:
    """Validate the frozen K=3 manifest contract and return density evidence."""

    if manifest.get("status") != "ok":
        raise ValueError(f"{manifest_path}: manifest status is not ok")
    if int(manifest.get("chosen_k_final", -1)) != 3:
        raise ValueError(f"{manifest_path}: chosen_k_final must be 3")
    if manifest.get("cluster_label_ordering") != "long_axis_head_to_tail":
        raise ValueError(f"{manifest_path}: cluster_label_ordering is not head-to-tail")
    if list(manifest.get("cluster_order_head_to_tail", [])) != [1, 2, 3]:
        raise ValueError(f"{manifest_path}: cluster_order_head_to_tail must be [1, 2, 3]")

    consumed = manifest.get("template_left_or_right_surface")
    if not isinstance(consumed, str) or not consumed:
        raise ValueError(f"{manifest_path}: template_left_or_right_surface is missing")
    consumed_path = Path(consumed).expanduser().resolve()
    _forbidden_path_gate(consumed_path)
    if not consumed_path.is_file():
        raise FileNotFoundError(f"{manifest_path}: consumed surface is missing: {consumed_path}")

    density = manifest.get("density_evidence")
    if not isinstance(density, Mapping):
        raise ValueError(f"{manifest_path}: density_evidence is missing")
    n_vertices = int(density.get("n_vertices", -1))
    n_faces = int(density.get("n_faces", -1))
    if expected_n_vertices is not None and n_vertices != expected_n_vertices:
        raise ValueError(
            f"{manifest_path}: density evidence reports {n_vertices} vertices; "
            f"expected {expected_n_vertices}"
        )
    if n_vertices <= 0 or n_faces <= 0:
        raise ValueError(f"{manifest_path}: invalid density evidence: {density}")

    evidence_path = density.get("surface_path")
    if evidence_path is not None and Path(str(evidence_path)).expanduser().resolve() != consumed_path:
        raise ValueError(
            f"{manifest_path}: density_evidence surface_path does not match "
            "template_left_or_right_surface"
        )
    return {
        "consumed_geometry_path": str(consumed_path),
        "density_surface_path": str(Path(str(evidence_path)).expanduser().resolve())
        if evidence_path is not None
        else str(consumed_path),
        "n_vertices": n_vertices,
        "n_faces": n_faces,
        "template_source_mode": manifest.get("template_source_mode"),
    }


def _read_gifti_surface(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = nib.load(str(path))
    arrays = [np.asarray(darray.data) for darray in image.darrays]
    vertices = next((array for array in arrays if array.ndim == 2 and array.shape[1] == 3 and np.issubdtype(array.dtype, np.floating)), None)
    faces = next((array for array in arrays if array.ndim == 2 and array.shape[1] == 3 and np.issubdtype(array.dtype, np.integer)), None)
    if vertices is None or faces is None:
        raise ValueError(f"surface is not a coordinate/triangle GIFTI: {path}")
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64)


def load_validated_atlas(
    directory: Path,
    *,
    group: str | None = None,
    variant: str | None = None,
    smoothing_mm: int | None = None,
    hemisphere: str | None = None,
    expected_n_vertices: int | None = None,
) -> LoadedAtlas:
    """Load one group map only after validating its manifest and geometry."""

    directory = Path(directory).resolve()
    manifest_path = directory / "group_atlas_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"atlas directory is missing its manifest: {directory}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evidence = validate_manifest(
        manifest,
        manifest_path=manifest_path,
        expected_n_vertices=expected_n_vertices,
    )
    declared_labels = manifest.get("group_labels_final")
    if not isinstance(declared_labels, str) or not declared_labels:
        raise ValueError(f"{manifest_path}: group_labels_final is missing")
    labels_path = Path(declared_labels).expanduser().resolve()
    _forbidden_path_gate(labels_path)
    if not labels_path.is_file():
        raise FileNotFoundError(f"{manifest_path}: declared labels are missing: {labels_path}")
    labels = np.asarray(np.load(labels_path), dtype=np.int64)
    local_labels_path = (directory / "group_labels_final.npy").resolve()
    if local_labels_path.is_file() and local_labels_path != labels_path:
        local_labels = np.asarray(np.load(local_labels_path), dtype=np.int64)
        if not np.array_equal(local_labels, labels):
            raise ValueError(
                f"{local_labels_path}: local label copy disagrees with "
                f"manifest-declared source {labels_path}"
            )
    if labels.ndim != 1 or labels.size != evidence["n_vertices"]:
        raise ValueError(f"{labels_path}: label shape {labels.shape} disagrees with density evidence")
    if not np.array_equal(np.unique(labels), np.asarray([1, 2, 3])):
        raise ValueError(f"{labels_path}: expected exactly labels 1, 2, 3")

    vertices, faces = _read_gifti_surface(Path(evidence["consumed_geometry_path"]))
    if vertices.shape != (evidence["n_vertices"], 3):
        raise ValueError(f"{evidence['consumed_geometry_path']}: vertex shape mismatch")
    if faces.shape != (evidence["n_faces"], 3):
        raise ValueError(f"{evidence['consumed_geometry_path']}: face shape mismatch")
    if np.any(faces < 0) or np.any(faces >= vertices.shape[0]):
        raise ValueError(f"{evidence['consumed_geometry_path']}: face index out of range")
    return LoadedAtlas(
        group=group or str(manifest.get("group", "unknown")),
        variant=variant or "unknown",
        smoothing_mm=int(smoothing_mm or 0),
        hemisphere=hemisphere or str(manifest.get("hemi", "?")),
        directory=directory,
        manifest_path=manifest_path,
        labels_path=labels_path,
        labels=labels,
        vertices=vertices,
        faces=faces,
        manifest=manifest,
        density_evidence=evidence,
    )


def _face_edges(faces: np.ndarray) -> np.ndarray:
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    return np.unique(np.sort(edges, axis=1), axis=0)


def mesh_geodesic_matrix(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """All-pairs shortest paths using Euclidean mesh-edge lengths in mm."""

    edges = _face_edges(faces)
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    graph = csr_matrix(
        (np.concatenate([lengths, lengths]), (np.concatenate([edges[:, 0], edges[:, 1]]), np.concatenate([edges[:, 1], edges[:, 0]]))),
        shape=(vertices.shape[0], vertices.shape[0]),
    )
    return np.asarray(shortest_path(graph, directed=False, unweighted=False), dtype=float)


def compute_atlas_region_area(
    vertices: np.ndarray, faces: np.ndarray, labels: np.ndarray
) -> dict[str, Any]:
    """Assign one third of each triangle area to each of its vertices."""

    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    triangle_vectors = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    triangle_vectors_2 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    triangle_area = 0.5 * np.linalg.norm(np.cross(triangle_vectors, triangle_vectors_2), axis=1)
    vertex_area = np.zeros(vertices.shape[0], dtype=float)
    contribution = triangle_area / 3.0
    for corner in range(3):
        np.add.at(vertex_area, faces[:, corner], contribution)
    area = {cluster: float(vertex_area[labels == cluster].sum()) for cluster in (1, 2, 3)}
    total = float(triangle_area.sum())
    fraction = {cluster: area[cluster] / total if total else np.nan for cluster in (1, 2, 3)}
    return {
        "atlas_triangle_area_mm2": triangle_area,
        "atlas_vertex_area_mm2": vertex_area,
        "atlas_region_area_mm2": area,
        "atlas_region_area_fraction": fraction,
        "atlas_total_surface_area_mm2": total,
    }


def boundary_vertex_sets(labels: np.ndarray, faces: np.ndarray) -> dict[str, np.ndarray]:
    """Return the union of vertices on each ordered adjacent-cluster boundary."""

    labels = np.asarray(labels, dtype=np.int64)
    edges = _face_edges(np.asarray(faces, dtype=np.int64))
    edge_labels = labels[edges]
    result: dict[str, np.ndarray] = {}
    for first, second in BOUNDARY_PAIRS:
        mask = ((edge_labels[:, 0] == first) & (edge_labels[:, 1] == second)) | (
            (edge_labels[:, 0] == second) & (edge_labels[:, 1] == first)
        )
        result[f"{first}-{second}"] = np.unique(edges[mask])
    return result


def _dice(first: np.ndarray, second: np.ndarray) -> float:
    first_set = set(np.asarray(first, dtype=np.int64).tolist())
    second_set = set(np.asarray(second, dtype=np.int64).tolist())
    if not first_set and not second_set:
        return 1.0
    if not first_set or not second_set:
        return 0.0
    return float(2.0 * len(first_set & second_set) / (len(first_set) + len(second_set)))


def _symmetric_distance_summary(
    first: np.ndarray, second: np.ndarray, geodesic: np.ndarray
) -> dict[str, float]:
    if first.size == 0 or second.size == 0:
        return {"mean": np.nan, "p95": np.nan, "hausdorff": np.nan}
    first_to_second = np.min(geodesic[np.ix_(first, second)], axis=1)
    second_to_first = np.min(geodesic[np.ix_(second, first)], axis=1)
    both = np.concatenate([first_to_second, second_to_first])
    return {
        "mean": float(np.mean(both)),
        "p95": float(np.percentile(both, 95)),
        "hausdorff": float(max(np.max(first_to_second), np.max(second_to_first))),
    }


def compare_label_maps(
    first_labels: np.ndarray,
    second_labels: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    geodesic: np.ndarray | None = None,
    axis_coordinate: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compare two ordered K=3 maps and their 1-2/2-3 boundaries."""

    first_labels = np.asarray(first_labels, dtype=np.int64)
    second_labels = np.asarray(second_labels, dtype=np.int64)
    if first_labels.shape != second_labels.shape or first_labels.ndim != 1:
        raise ValueError("label maps must have the same one-dimensional shape")
    if geodesic is None:
        geodesic = mesh_geodesic_matrix(np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64))
    first_boundaries = boundary_vertex_sets(first_labels, faces)
    second_boundaries = boundary_vertex_sets(second_labels, faces)
    ordered_dice = {cluster: _dice(np.flatnonzero(first_labels == cluster), np.flatnonzero(second_labels == cluster)) for cluster in (1, 2, 3)}
    boundary_dice = {key: _dice(first_boundaries[key], second_boundaries[key]) for key in first_boundaries}
    boundary_distance = {
        key: _symmetric_distance_summary(first_boundaries[key], second_boundaries[key], geodesic)
        for key in first_boundaries
    }
    positions: dict[str, dict[str, float]] = {}
    if axis_coordinate is not None:
        for key in first_boundaries:
            positions[key] = {
                "first": float(np.mean(axis_coordinate[first_boundaries[key]])) if first_boundaries[key].size else np.nan,
                "second": float(np.mean(axis_coordinate[second_boundaries[key]])) if second_boundaries[key].size else np.nan,
            }
            positions[key]["delta_second_minus_first"] = positions[key]["second"] - positions[key]["first"]
    return {
        "raw_label_agreement": float(np.mean(first_labels == second_labels)),
        "ari": float(adjusted_rand_score(first_labels, second_labels)),
        "ordered_cluster_dice": ordered_dice,
        "boundary_dice": boundary_dice,
        "boundary_distance_mm": boundary_distance,
        "boundary_vertex_counts": {
            key: {"first": int(first_boundaries[key].size), "second": int(second_boundaries[key].size)}
            for key in first_boundaries
        },
        "boundary_positions": positions,
    }


def reference_axis_coordinate(
    reference_labels: np.ndarray, vertices: np.ndarray, faces: np.ndarray, *, geodesic: np.ndarray | None = None
) -> np.ndarray:
    """Create a transparent 0=head-to-1=tail coordinate from a reference map."""

    if geodesic is None:
        geodesic = mesh_geodesic_matrix(vertices, faces)
    head = np.flatnonzero(np.asarray(reference_labels) == 1)
    tail = np.flatnonzero(np.asarray(reference_labels) == 3)
    to_head = np.min(geodesic[:, head], axis=1)
    to_tail = np.min(geodesic[:, tail], axis=1)
    denominator = to_head + to_tail
    return np.divide(to_head, denominator, out=np.full_like(to_head, np.nan), where=denominator > 0)


def _atlas_directory(input_root: Path, group: str, variant: str, smoothing_mm: int, hemisphere: str) -> Path:
    if variant == "plain":
        dirname = f"routea_{GROUP_SLUG[group]}_full_k3_fisherz"
    elif variant == "weighted":
        dirname = f"routea_{GROUP_SLUG[group]}_weighted_fisherz_full_k3"
    else:
        raise ValueError(f"unknown atlas variant: {variant}")
    return input_root / dirname / "gradient-spectral" / "kong2022" / f"{smoothing_mm}mm" / "cov_0p80" / "g1g2g3" / f"hemi_{hemisphere}"


def _select_axis(
    requested: Sequence[Any] | None,
    allowed: Sequence[Any],
    *,
    name: str,
) -> tuple[Any, ...]:
    """Select an explicit subset while retaining the frozen registry order."""

    allowed_values = tuple(allowed)
    if requested is None:
        return allowed_values
    if isinstance(requested, (str, bytes)):
        raise TypeError(f"{name} must be a sequence")
    selected = tuple(requested)
    if not selected:
        raise ValueError(f"{name} must not be empty")
    if len(selected) != len(set(selected)):
        raise ValueError(f"{name} contains duplicates")
    unknown = sorted(set(selected) - set(allowed_values), key=str)
    if unknown:
        raise ValueError(f"{name} contains values outside the registry: {unknown}")
    return tuple(value for value in allowed_values if value in set(selected))


def discover_atlases(
    input_root: Path,
    *,
    expected_n_vertices: int = 512,
    groups: Sequence[str] | None = None,
    variants: Sequence[str] | None = None,
    smoothings_mm: Sequence[int] | None = None,
    hemispheres: Sequence[str] | None = None,
) -> list[LoadedAtlas]:
    """Read explicit frozen atlas cells; never perform recursive discovery."""

    input_root = Path(input_root).resolve()
    _forbidden_path_gate(input_root)
    selected_groups = _select_axis(groups, GROUP_ORDER, name="groups")
    selected_variants = _select_axis(variants, VARIANT_ORDER, name="variants")
    selected_smoothings = _select_axis(
        tuple(int(value) for value in smoothings_mm) if smoothings_mm is not None else None,
        SMOOTHING_ORDER,
        name="smoothings_mm",
    )
    selected_hemispheres = _select_axis(hemispheres, HEMISPHERE_ORDER, name="hemispheres")
    records: list[LoadedAtlas] = []
    for group in selected_groups:
        for variant in selected_variants:
            for smoothing_mm in selected_smoothings:
                for hemisphere in selected_hemispheres:
                    directory = _atlas_directory(input_root, group, variant, smoothing_mm, hemisphere)
                    records.append(
                        load_validated_atlas(
                            directory,
                            group=group,
                            variant=variant,
                            smoothing_mm=smoothing_mm,
                            hemisphere=hemisphere,
                            expected_n_vertices=expected_n_vertices,
                        )
                    )
    return records


def build_inventory(records: Iterable[LoadedAtlas]) -> pd.DataFrame:
    rows = []
    for record in records:
        rows.append(
            {
                "group": record.group,
                "variant": record.variant,
                "smoothing_mm": record.smoothing_mm,
                "hemisphere": record.hemisphere,
                "atlas_directory": str(record.directory),
                "manifest_path": str(record.manifest_path),
                "labels_path": str(record.labels_path),
                "consumed_geometry_path": record.density_evidence["consumed_geometry_path"],
                "density_surface_path": record.density_evidence["density_surface_path"],
                "n_vertices": record.density_evidence["n_vertices"],
                "n_faces": record.density_evidence["n_faces"],
                "manifest_status": record.manifest.get("status"),
                "chosen_k_final": record.manifest.get("chosen_k_final"),
                "cluster_label_ordering": record.manifest.get("cluster_label_ordering"),
                "cluster_order_head_to_tail": "1,2,3",
                "density_evidence_note": "consumed template_left_or_right_surface",
            }
        )
    return pd.DataFrame(rows).sort_values(["group", "variant", "smoothing_mm", "hemisphere"], key=lambda s: s.map({g: i for i, g in enumerate(GROUP_ORDER)}) if s.name == "group" else s.map({v: i for i, v in enumerate(VARIANT_ORDER)}) if s.name == "variant" else s).reset_index(drop=True)


def _geometry_cache(records: Iterable[LoadedAtlas]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for record in records:
        result.setdefault(record.consumed_geometry_path, (record.vertices, record.faces))
    return result


def _area_rows(records: Iterable[LoadedAtlas]) -> tuple[pd.DataFrame, dict[tuple[str, str, int, str], dict[str, Any]]]:
    rows = []
    summaries: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for record in records:
        summary = compute_atlas_region_area(record.vertices, record.faces, record.labels)
        geodesic = mesh_geodesic_matrix(record.vertices, record.faces)
        summaries[record.key] = {
            "record": record,
            "area": summary,
            "geodesic": geodesic,
        }
        for cluster in (1, 2, 3):
            rows.append(
                {
                    "group": record.group,
                    "variant": record.variant,
                    "smoothing_mm": record.smoothing_mm,
                    "hemisphere": record.hemisphere,
                    "cluster": cluster,
                    "atlas_region_area_mm2": summary["atlas_region_area_mm2"][cluster],
                    "atlas_region_area_fraction": summary["atlas_region_area_fraction"][cluster],
                    "atlas_total_surface_area_mm2": summary["atlas_total_surface_area_mm2"],
                    "n_vertices": record.density_evidence["n_vertices"],
                    "n_faces": record.density_evidence["n_faces"],
                    "consumed_geometry_path": record.consumed_geometry_path,
                    "density_evidence": json.dumps(record.density_evidence, sort_keys=True),
                    "descriptive_only": True,
                }
            )
    return pd.DataFrame(rows), summaries


def _comparison_row(
    first: LoadedAtlas,
    second: LoadedAtlas,
    comparison: Mapping[str, Any],
    *,
    comparison_kind: str,
    spec_axis: str = "",
) -> dict[str, Any]:
    geometry_match = first.consumed_geometry_path == second.consumed_geometry_path
    row: dict[str, Any] = {
        "comparison_kind": comparison_kind,
        "spec_axis": spec_axis,
        "group_a": first.group,
        "group_b": second.group,
        "variant_a": first.variant,
        "variant_b": second.variant,
        "smoothing_a_mm": first.smoothing_mm,
        "smoothing_b_mm": second.smoothing_mm,
        "hemisphere": first.hemisphere,
        "map_a": str(first.labels_path),
        "map_b": str(second.labels_path),
        "geometry_a": first.consumed_geometry_path,
        "geometry_b": second.consumed_geometry_path,
        "geometry_match": geometry_match,
        "physical_geometry_comparable": geometry_match,
        "comparison_basis": (
            "same_geometry_physical_and_vertex_labels"
            if geometry_match
            else "vertex_index_labels_only_physical_metrics_unavailable"
        ),
        "geodesic_basis": first.consumed_geometry_path if geometry_match else "",
        "raw_label_agreement": comparison["raw_label_agreement"],
        "ari": comparison["ari"],
        "descriptive_only": True,
        "uncertainty": "none",
        "p_value": np.nan,
        "q_value": np.nan,
    }
    for cluster in (1, 2, 3):
        row[f"ordered_cluster_dice_{cluster}"] = comparison["ordered_cluster_dice"][cluster]
    for boundary in ("1-2", "2-3"):
        row[f"boundary_{boundary}_dice"] = comparison["boundary_dice"][boundary]
        distance = comparison["boundary_distance_mm"][boundary]
        row[f"boundary_{boundary}_distance_mean_mm"] = distance["mean"]
        row[f"boundary_{boundary}_distance_p95_mm"] = distance["p95"]
        row[f"boundary_{boundary}_distance_hausdorff_mm"] = distance["hausdorff"]
        counts = comparison["boundary_vertex_counts"][boundary]
        row[f"boundary_{boundary}_vertices_a"] = counts["first"]
        row[f"boundary_{boundary}_vertices_b"] = counts["second"]
        if boundary in comparison["boundary_positions"]:
            position = comparison["boundary_positions"][boundary]
            row[f"boundary_{boundary}_position_a"] = position["first"]
            row[f"boundary_{boundary}_position_b"] = position["second"]
            row[f"boundary_{boundary}_position_delta_b_minus_a"] = position["delta_second_minus_first"]
        else:
            row[f"boundary_{boundary}_position_a"] = np.nan
            row[f"boundary_{boundary}_position_b"] = np.nan
            row[f"boundary_{boundary}_position_delta_b_minus_a"] = np.nan
    return row


def build_matched_group_pairs(
    records: Iterable[LoadedAtlas],
    summaries: Mapping[tuple[str, str, int, str], Mapping[str, Any]],
) -> pd.DataFrame:
    """Compare each non-HC atlas only against the matched HC atlas."""

    by_key = {record.key: record for record in records}
    rows = []
    for variant in VARIANT_ORDER:
        for smoothing_mm in SMOOTHING_ORDER:
            for hemisphere in HEMISPHERE_ORDER:
                reference = by_key[(REFERENCE_GROUP, variant, 2, hemisphere)]
                axis = reference_axis_coordinate(
                    reference.labels,
                    reference.vertices,
                    reference.faces,
                    geodesic=summaries[reference.key]["geodesic"],
                )
                for group_a, group_b in HC_COMPARISON_ORDER:
                    first = by_key[(group_a, variant, smoothing_mm, hemisphere)]
                    second = by_key[(group_b, variant, smoothing_mm, hemisphere)]
                    comparison = compare_label_maps(
                        first.labels,
                        second.labels,
                        first.vertices,
                        first.faces,
                        geodesic=summaries[first.key]["geodesic"],
                        axis_coordinate=axis,
                    )
                    rows.append(_comparison_row(first, second, comparison, comparison_kind="matched_group_pair"))
    return pd.DataFrame(rows)


def build_spec_stability(
    records: Iterable[LoadedAtlas],
    summaries: Mapping[tuple[str, str, int, str], Mapping[str, Any]],
) -> pd.DataFrame:
    """Use smoothing and assignment-variant changes as descriptive noise references."""

    by_key = {record.key: record for record in records}
    rows = []
    for group in GROUP_ORDER:
        for hemisphere in HEMISPHERE_ORDER:
            for variant in VARIANT_ORDER:
                reference = by_key[(REFERENCE_GROUP, variant, 2, hemisphere)]
                axis = reference_axis_coordinate(
                    reference.labels,
                    reference.vertices,
                    reference.faces,
                    geodesic=summaries[reference.key]["geodesic"],
                )
                first = by_key[(group, variant, 2, hemisphere)]
                second = by_key[(group, variant, 4, hemisphere)]
                comparison = compare_label_maps(
                    first.labels,
                    second.labels,
                    first.vertices,
                    first.faces,
                    geodesic=summaries[first.key]["geodesic"],
                    axis_coordinate=axis,
                )
                rows.append(
                    _comparison_row(
                        first,
                        second,
                        comparison,
                        comparison_kind="within_group_spec_stability",
                        spec_axis="smoothing",
                    )
                )
            for smoothing_mm in SMOOTHING_ORDER:
                first = by_key[(group, "plain", smoothing_mm, hemisphere)]
                second = by_key[(group, "weighted", smoothing_mm, hemisphere)]
                geometry_match = first.consumed_geometry_path == second.consumed_geometry_path
                if not np.array_equal(first.faces, second.faces):
                    raise ValueError(
                        "assignment-variant label comparison requires matching vertex topology"
                    )
                axis = None
                if geometry_match:
                    reference = by_key[(REFERENCE_GROUP, "plain", 2, hemisphere)]
                    axis = reference_axis_coordinate(
                        reference.labels,
                        reference.vertices,
                        reference.faces,
                        geodesic=summaries[reference.key]["geodesic"],
                    )
                comparison = compare_label_maps(
                    first.labels,
                    second.labels,
                    first.vertices,
                    first.faces,
                    geodesic=summaries[first.key]["geodesic"],
                    axis_coordinate=axis,
                )
                if not geometry_match:
                    comparison["boundary_positions"] = {}
                    comparison["boundary_distance_mm"] = {
                        boundary: {"mean": np.nan, "p95": np.nan, "hausdorff": np.nan}
                        for boundary in ("1-2", "2-3")
                    }
                rows.append(
                    _comparison_row(
                        first,
                        second,
                        comparison,
                        comparison_kind="within_group_spec_stability",
                        spec_axis="assignment_variant",
                    )
                )
    return pd.DataFrame(rows)


def _boundary_position_by_map(
    records: Iterable[LoadedAtlas], summaries: Mapping[tuple[str, str, int, str], Mapping[str, Any]]
) -> dict[tuple[str, str, int, str, str], float]:
    records = tuple(records)
    by_key = {record.key: record for record in records}
    result: dict[tuple[str, str, int, str, str], float] = {}
    for variant in VARIANT_ORDER:
        for hemisphere in HEMISPHERE_ORDER:
            reference = by_key[(REFERENCE_GROUP, variant, 2, hemisphere)]
            axis = reference_axis_coordinate(
                reference.labels,
                reference.vertices,
                reference.faces,
                geodesic=summaries[reference.key]["geodesic"],
            )
            for record in records:
                if record.variant != variant or record.hemisphere != hemisphere:
                    continue
                boundaries = boundary_vertex_sets(record.labels, record.faces)
                for boundary, vertices in boundaries.items():
                    result[(*record.key, boundary)] = (
                        float(np.mean(axis[vertices])) if vertices.size else np.nan
                    )
    return result


def _legacy_build_descriptive_contrasts(
    records: Iterable[LoadedAtlas],
    area_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    summaries: Mapping[tuple[str, str, int, str], Mapping[str, Any]],
) -> pd.DataFrame:
    """Retained inactive helper for historical compatibility; never called by active runs.

    The authoritative implementation below emits only direct non-HC-versus-HC
    descriptive contrasts.
    """

    direct = (
        ("NTS-NTHC", "symptom_nontrauma", "NTS", "NTHC"),
        ("TIS-TEHC", "symptom_trauma", "TIS", "TEHC"),
        ("TEHC-NTHC", "trauma_healthy", "TEHC", "NTHC"),
        ("TIS-NTS", "trauma_symptomatic", "TIS", "NTS"),
        ("NTS Any MDD-NTHC", "nested_any_mdd", "NTS Any MDD", "NTHC"),
        ("NTS Pure MDD-NTHC", "nested_pure_mdd", "NTS Pure MDD", "NTHC"),
        ("TIS PTSD-NTHC", "nested_ptsd", "TIS PTSD", "NTHC"),
    )
    positions = _boundary_position_by_map(records, summaries)
    rows = []
    cells = [(variant, smoothing_mm, hemisphere) for variant in VARIANT_ORDER for smoothing_mm in SMOOTHING_ORDER for hemisphere in HEMISPHERE_ORDER]

    def area_value(group: str, variant: str, smoothing_mm: int, hemisphere: str, cluster: int) -> float:
        value = area_df.loc[
            (area_df["group"] == group)
            & (area_df["variant"] == variant)
            & (area_df["smoothing_mm"] == smoothing_mm)
            & (area_df["hemisphere"] == hemisphere)
            & (area_df["cluster"] == cluster),
            "atlas_region_area_fraction",
        ]
        return float(value.iloc[0])

    def add_row(**kwargs: Any) -> None:
        rows.append({
            **kwargs,
            "descriptive_only": True,
            "uncertainty": "none",
            "p_value": np.nan,
            "q_value": np.nan,
        })

    for variant, smoothing_mm, hemisphere in cells:
        for contrast, family, first_group, second_group in direct:
            for cluster in (1, 2, 3):
                add_row(
                    contrast=contrast,
                    contrast_family=family,
                    variant=variant,
                    smoothing_mm=smoothing_mm,
                    hemisphere=hemisphere,
                    metric="atlas_region_area_fraction_difference",
                    cluster=cluster,
                    boundary="",
                    estimate=area_value(first_group, variant, smoothing_mm, hemisphere, cluster) - area_value(second_group, variant, smoothing_mm, hemisphere, cluster),
                    units="fraction",
                    interpretation="first group minus second group; group-map descriptive only",
                )
            first_pair = pair_df.loc[
                (pair_df["group_a"] == min(first_group, second_group, key=GROUP_ORDER.index))
                & (pair_df["group_b"] == max(first_group, second_group, key=GROUP_ORDER.index))
                & (pair_df["variant_a"] == variant)
                & (pair_df["smoothing_a_mm"] == smoothing_mm)
                & (pair_df["hemisphere"] == hemisphere)
            ]
            # Pair rows are stored in canonical GROUP_ORDER. Physical boundary
            # discrepancies are symmetric and therefore have no contrast sign.
            if first_pair.empty:
                raise AssertionError(f"missing pair row for {contrast}")
            pair_row = first_pair.iloc[0]
            for boundary in ("1-2", "2-3"):
                position_first = positions[(first_group, variant, smoothing_mm, hemisphere, boundary)]
                position_second = positions[(second_group, variant, smoothing_mm, hemisphere, boundary)]
                add_row(
                    contrast=contrast,
                    contrast_family=family,
                    variant=variant,
                    smoothing_mm=smoothing_mm,
                    hemisphere=hemisphere,
                    metric="boundary_position_difference",
                    cluster="",
                    boundary=boundary,
                    estimate=position_first - position_second,
                    units="normalized head-to-tail fraction",
                    interpretation="first group boundary position minus second group; reference axis is NTHC same-variant 2mm within hemisphere",
                )
                for metric_suffix, units in (("distance_mean_mm", "mm"), ("distance_p95_mm", "mm"), ("distance_hausdorff_mm", "mm")):
                    add_row(
                        contrast=contrast,
                        contrast_family=family,
                        variant=variant,
                        smoothing_mm=smoothing_mm,
                        hemisphere=hemisphere,
                        metric=f"boundary_{metric_suffix}",
                        cluster="",
                        boundary=boundary,
                        estimate=float(pair_row[f"boundary_{boundary}_{metric_suffix}"]),
                        units=units,
                        interpretation="symmetric boundary discrepancy between the two group maps; not a group effect estimate",
                    )

        # Difference-in-differences is the descriptive symptom-by-trauma map
        # contrast.  It is intentionally emitted without p/q values.
        for cluster in (1, 2, 3):
            interaction = (area_value("TIS", variant, smoothing_mm, hemisphere, cluster) - area_value("TEHC", variant, smoothing_mm, hemisphere, cluster)) - (area_value("NTS", variant, smoothing_mm, hemisphere, cluster) - area_value("NTHC", variant, smoothing_mm, hemisphere, cluster))
            add_row(
                contrast="(TIS-TEHC)-(NTS-NTHC)",
                contrast_family="symptom_x_trauma",
                variant=variant,
                smoothing_mm=smoothing_mm,
                hemisphere=hemisphere,
                metric="atlas_region_area_fraction_interaction",
                cluster=cluster,
                boundary="",
                estimate=interaction,
                units="fraction",
                interpretation="descriptive difference-in-differences of group map area fractions",
            )
        for boundary in ("1-2", "2-3"):
            interaction = (positions[("TIS", variant, smoothing_mm, hemisphere, boundary)] - positions[("TEHC", variant, smoothing_mm, hemisphere, boundary)]) - (positions[("NTS", variant, smoothing_mm, hemisphere, boundary)] - positions[("NTHC", variant, smoothing_mm, hemisphere, boundary)])
            add_row(
                contrast="(TIS-TEHC)-(NTS-NTHC)",
                contrast_family="symptom_x_trauma",
                variant=variant,
                smoothing_mm=smoothing_mm,
                hemisphere=hemisphere,
                metric="boundary_position_interaction",
                cluster="",
                boundary=boundary,
                estimate=interaction,
                units="normalized head-to-tail fraction",
                interpretation="descriptive difference-in-differences of boundary positions",
            )
            for metric_suffix, units in (("distance_mean_mm", "mm"), ("distance_p95_mm", "mm"), ("distance_hausdorff_mm", "mm")):
                def pair_distance(first_group: str, second_group: str) -> float:
                    group_a, group_b = sorted((first_group, second_group), key=GROUP_ORDER.index)
                    matched = pair_df.loc[
                        (pair_df["group_a"] == group_a)
                        & (pair_df["group_b"] == group_b)
                        & (pair_df["variant_a"] == variant)
                        & (pair_df["smoothing_a_mm"] == smoothing_mm)
                        & (pair_df["hemisphere"] == hemisphere)
                    ]
                    if matched.empty:
                        raise AssertionError(f"missing pair row for {first_group}-{second_group}")
                    return float(matched.iloc[0][f"boundary_{boundary}_{metric_suffix}"])

                add_row(
                    contrast="(TIS-TEHC)-(NTS-NTHC)",
                    contrast_family="symptom_x_trauma",
                    variant=variant,
                    smoothing_mm=smoothing_mm,
                    hemisphere=hemisphere,
                    metric=f"boundary_{metric_suffix}_interaction",
                    cluster="",
                    boundary=boundary,
                    estimate=pair_distance("TIS", "TEHC") - pair_distance("NTS", "NTHC"),
                    units=units,
                    interpretation="descriptive difference-in-differences of within-condition group-map boundary discrepancy",
                )
    return pd.DataFrame(rows)


def build_descriptive_contrasts(
    records: Iterable[LoadedAtlas],
    area_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    summaries: Mapping[tuple[str, str, int, str], Mapping[str, Any]],
) -> pd.DataFrame:
    """Emit only direct non-HC-versus-HC descriptive atlas contrasts."""

    positions = _boundary_position_by_map(records, summaries)
    rows: list[dict[str, Any]] = []
    cells = [
        (variant, smoothing_mm, hemisphere)
        for variant in VARIANT_ORDER
        for smoothing_mm in SMOOTHING_ORDER
        for hemisphere in HEMISPHERE_ORDER
    ]

    def area_value(
        group: str, variant: str, smoothing_mm: int, hemisphere: str, cluster: int
    ) -> float:
        value = area_df.loc[
            (area_df["group"] == group)
            & (area_df["variant"] == variant)
            & (area_df["smoothing_mm"] == smoothing_mm)
            & (area_df["hemisphere"] == hemisphere)
            & (area_df["cluster"] == cluster),
            "atlas_region_area_fraction",
        ]
        if value.empty:
            raise AssertionError(
                f"missing area row for {group}/{variant}/{smoothing_mm}/{hemisphere}"
            )
        return float(value.iloc[0])

    def add_row(**kwargs: Any) -> None:
        rows.append(
            {
                **kwargs,
                "descriptive_only": True,
                "uncertainty": "none",
                "p_value": np.nan,
                "q_value": np.nan,
            }
        )

    for variant, smoothing_mm, hemisphere in cells:
        for group, reference_group in HC_COMPARISON_ORDER:
            contrast = f"{group}-{reference_group}"
            pair = pair_df.loc[
                (pair_df["group_a"] == group)
                & (pair_df["group_b"] == reference_group)
                & (pair_df["variant_a"] == variant)
                & (pair_df["smoothing_a_mm"] == smoothing_mm)
                & (pair_df["hemisphere"] == hemisphere)
            ]
            if pair.empty:
                raise AssertionError(f"missing HC pair row for {contrast}")
            pair_row = pair.iloc[0]
            for cluster in (1, 2, 3):
                add_row(
                    contrast=contrast,
                    contrast_family="group_vs_hc",
                    variant=variant,
                    smoothing_mm=smoothing_mm,
                    hemisphere=hemisphere,
                    metric="atlas_region_area_fraction_difference",
                    cluster=cluster,
                    boundary="",
                    estimate=area_value(
                        group, variant, smoothing_mm, hemisphere, cluster
                    )
                    - area_value(
                        reference_group, variant, smoothing_mm, hemisphere, cluster
                    ),
                    units="fraction",
                    interpretation="non-HC group minus HC; group-map descriptive only",
                )
            for boundary in ("1-2", "2-3"):
                position_group = positions[
                    (group, variant, smoothing_mm, hemisphere, boundary)
                ]
                position_hc = positions[
                    (reference_group, variant, smoothing_mm, hemisphere, boundary)
                ]
                add_row(
                    contrast=contrast,
                    contrast_family="group_vs_hc",
                    variant=variant,
                    smoothing_mm=smoothing_mm,
                    hemisphere=hemisphere,
                    metric="boundary_position_difference",
                    cluster="",
                    boundary=boundary,
                    estimate=position_group - position_hc,
                    units="normalized head-to-tail fraction",
                    interpretation="non-HC group boundary position minus HC; reference axis is HC same-variant 2mm within hemisphere",
                )
                for metric_suffix, units in (
                    ("distance_mean_mm", "mm"),
                    ("distance_p95_mm", "mm"),
                    ("distance_hausdorff_mm", "mm"),
                ):
                    add_row(
                        contrast=contrast,
                        contrast_family="group_vs_hc",
                        variant=variant,
                        smoothing_mm=smoothing_mm,
                        hemisphere=hemisphere,
                        metric=f"boundary_{metric_suffix}",
                        cluster="",
                        boundary=boundary,
                        estimate=float(
                            pair_row[f"boundary_{boundary}_{metric_suffix}"]
                        ),
                        units=units,
                        interpretation="symmetric boundary discrepancy between the non-HC and HC maps; descriptive only",
                    )
    return pd.DataFrame(rows)


def _write_figure(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")


def plot_reference_area_profiles(area_df: pd.DataFrame, figure_dir: Path) -> None:
    import matplotlib.pyplot as plt

    data = area_df.loc[(area_df["variant"] == "plain") & (area_df["smoothing_mm"] == 2)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    colors = {
        "HC": "#222222",
        "NTHC": "#1f4e79",
        "TEHC": "#b07d00",
        "NTS": "#7a4a9e",
        "NTS Any MDD": "#9b7fc5",
        "NTS Pure MDD": "#a23b72",
        "TIS": "#b44c3c",
        "TIS PTSD": "#d95f02",
    }
    for axis, hemisphere in zip(axes, HEMISPHERE_ORDER):
        subset = data.loc[data["hemisphere"] == hemisphere]
        for group in GROUP_ORDER:
            values = subset.loc[subset["group"] == group].sort_values("cluster")["atlas_region_area_fraction"]
            axis.plot([1, 2, 3], values, marker="o", linewidth=1.8, label=group, color=colors[group])
        axis.set_title(f"Hemisphere {hemisphere}")
        axis.set_xlabel("Ordered cluster (head → tail)")
        axis.set_xticks([1, 2, 3])
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.7)
    axes[0].set_ylabel("Surface area fraction")
    axes[1].legend(frameon=False, loc="best")
    fig.suptitle("Reference 2mm plain K=3 atlas area profiles", fontsize=12)
    fig.tight_layout()
    _write_figure(fig, figure_dir / "reference_2mm_plain_area_profiles")
    plt.close(fig)


def plot_pairwise_ari(pair_df: pd.DataFrame, figure_dir: Path) -> None:
    import matplotlib.pyplot as plt

    data = pair_df.loc[(pair_df["variant_a"] == "plain") & (pair_df["smoothing_a_mm"] == 2)].copy()
    data["pair"] = data["group_a"] + "–" + data["group_b"]
    pivot = data.pivot(index="pair", columns="hemisphere", values="ari").reindex([a + "–" + b for a, b in PAIR_ORDER])
    fig, axis = plt.subplots(figsize=(13, 6))
    x = np.arange(len(pivot.index))
    width = 0.36
    axis.bar(x - width / 2, pivot["L"], width, label="L", color="#1f4e79")
    axis.bar(x + width / 2, pivot["R"], width, label="R", color="#b07d00")
    axis.set_xticks(x, pivot.index, rotation=45, ha="right")
    axis.set_ylabel("Adjusted Rand index")
    axis.set_ylim(0, 1.05)
    axis.set_title("HC-referenced ARI of reference 2mm plain group maps")
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.7)
    axis.legend(frameon=False, title="Hemisphere")
    fig.tight_layout()
    _write_figure(fig, figure_dir / "reference_2mm_plain_pairwise_ari")
    plt.close(fig)


def plot_boundary_spec_comparison(stability_df: pd.DataFrame, figure_dir: Path) -> None:
    import matplotlib.pyplot as plt

    rows = []
    for _, row in stability_df.iterrows():
        for boundary in ("1-2", "2-3"):
            rows.append({
                "spec_axis": row["spec_axis"],
                "hemisphere": row["hemisphere"],
                "boundary": boundary,
                "distance_mm": row[f"boundary_{boundary}_distance_mean_mm"],
            })
    data = pd.DataFrame(rows)
    summary = data.groupby(
        ["spec_axis", "hemisphere", "boundary"],
        as_index=False,
        dropna=False,
    )["distance_mm"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, hemisphere in zip(axes, HEMISPHERE_ORDER):
        subset = summary.loc[summary["hemisphere"] == hemisphere]
        for boundary in ("1-2", "2-3"):
            values = subset.loc[subset["boundary"] == boundary].set_index("spec_axis")
            smoothing_value = values.loc["smoothing", "distance_mm"]
            axis.plot([0], [smoothing_value], marker="o", linewidth=1.8, label=boundary)
        axis.text(
            1,
            0.08,
            "N/A\ndifferent geometry",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#555555",
        )
        axis.set_xlim(-0.2, 1.2)
        axis.set_xticks(
            [0, 1],
            ["2 vs 4 mm", "plain vs weighted\n(label-only)"],
        )
        axis.set_title(f"Hemisphere {hemisphere}")
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.7)
    axes[0].set_ylabel("Mean symmetric boundary distance (mm)")
    axes[1].legend(frameon=False, title="Boundary")
    fig.suptitle("Physical boundary discrepancy (same geometry only)", fontsize=12)
    fig.tight_layout()
    _write_figure(fig, figure_dir / "boundary_specification_comparison")
    plt.close(fig)


def run_analysis(
    *,
    input_root: Path,
    analysis_root: Path | None = None,
    run_id: str = "atlas_descriptive",
    results_dir: Path | None = None,
    figure_dir: Path | None = None,
    report_dir: Path | None = None,
    groups: Sequence[str] | None = None,
    variants: Sequence[str] | None = None,
    smoothings_mm: Sequence[int] | None = None,
    hemispheres: Sequence[str] | None = None,
) -> dict[str, Path]:
    """Run the current descriptive comparison and overwrite its artifacts."""

    if analysis_root is None and results_dir is None:
        raise ValueError("analysis_root or results_dir is required")
    base_root = Path(analysis_root).resolve() if analysis_root is not None else None
    result_dir = Path(results_dir).resolve() if results_dir is not None else base_root / "results" / run_id
    figure_dir = Path(figure_dir).resolve() if figure_dir is not None else (
        base_root / "figures" / run_id
    )
    report_dir = Path(report_dir).resolve() if report_dir is not None else (
        base_root / "reports" / run_id
    )
    discovery_kwargs = {
        "groups": groups,
        "variants": variants,
        "smoothings_mm": smoothings_mm,
        "hemispheres": hemispheres,
    }
    if all(value is None for value in discovery_kwargs.values()):
        records = discover_atlases(Path(input_root))
    else:
        records = discover_atlases(Path(input_root), **discovery_kwargs)
    inventory = build_inventory(records)
    area_df, summaries = _area_rows(records)
    pair_df = build_matched_group_pairs(records, summaries)
    stability_df = build_spec_stability(records, summaries)
    contrast_df = build_descriptive_contrasts(records, area_df, pair_df, summaries)

    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    output_tables = {
        "atlas_inventory": inventory,
        "region_area": area_df,
        "matched_group_pair_comparison": pair_df,
        "within_group_spec_stability": stability_df,
        "descriptive_contrasts": contrast_df,
    }
    paths: dict[str, Path] = {}
    for stem, frame in output_tables.items():
        path = result_dir / f"{stem}.csv"
        frame.to_csv(path, index=False)
        paths[stem] = path
    manifest = {
        "run_id": run_id,
        "analysis": "Unified Stanford group-atlas K=3 geometry comparison",
        "input_root": str(Path(input_root).resolve()),
        "n_atlas_cells": len(records),
        "groups": list(GROUP_ORDER),
        "parent_factorial_groups": list(FACTORIAL_GROUP_ORDER),
        "nested_clinical_groups": list(NESTED_GROUP_ORDER),
        "n_group_pairs": len(HC_COMPARISON_ORDER),
        "groupwise_contrast": "each_non_reference_group_vs_reference",
        "reference_group": REFERENCE_GROUP,
        "variants": list(VARIANT_ORDER),
        "smoothings_mm": list(SMOOTHING_ORDER),
        "hemispheres": list(HEMISPHERE_ORDER),
        "surface_method": "triangle area divided equally across three vertices",
        "area_field_namespace": {
            "region_area_mm2": "atlas_region_area_mm2",
            "region_area_fraction": "atlas_region_area_fraction",
            "total_surface_area_mm2": "atlas_total_surface_area_mm2",
        },
        "area_role": "descriptive group-atlas geometry only; never Kember subject-level area",
        "boundary_method": "mesh-edge weighted shortest-path symmetric mean/p95/Hausdorff",
        "reference_axis": "HC same-variant 2mm map separately within each hemisphere",
        "cross_geometry_policy": "vertex-index label metrics retained; physical distance and position unavailable",
        "inference": "descriptive only; no uncertainty, p values, or q values",
        "scope_guard": "rest-FC atlas labels only; no beta, CON, native, or archive inputs",
    }
    (result_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    paths["analysis_manifest"] = result_dir / "analysis_manifest.json"

    plot_reference_area_profiles(area_df, figure_dir)
    plot_pairwise_ari(pair_df, figure_dir)
    plot_boundary_spec_comparison(stability_df, figure_dir)
    paths["figure_dir"] = figure_dir

    summary = _technical_summary(manifest, inventory, area_df, pair_df, stability_df, contrast_df)
    (report_dir / "technical_summary.md").write_text(summary, encoding="utf-8")
    (report_dir / "README.md").write_text(
        "# Stanford group atlas descriptive comparison\n\n"
        "This is the current descriptive comparison of completed parent and nested Stanford K=3 group maps. "
        "It cannot be used as a direct group-level inferential atlas test and does not modify the rest-FC primary screen.\n\n"
        "See `technical_summary.md` and the sibling `results/` and `figures/` directories.\n",
        encoding="utf-8",
    )
    paths["report_dir"] = report_dir
    return paths


def _technical_summary(
    manifest: Mapping[str, Any],
    inventory: pd.DataFrame,
    area_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    contrast_df: pd.DataFrame,
) -> str:
    reference_pairs = pair_df.loc[(pair_df["variant_a"] == "plain") & (pair_df["smoothing_a_mm"] == 2)].copy()
    ari = reference_pairs["ari"]
    mean_ari = float(ari.mean()) if not ari.empty else float("nan")
    lowest = reference_pairs.sort_values("ari").iloc[0] if not reference_pairs.empty else None
    stability = stability_df.copy()
    stability["mean_boundary_distance_mm"] = stability[["boundary_1-2_distance_mean_mm", "boundary_2-3_distance_mean_mm"]].mean(axis=1)
    variant_stability = stability.loc[stability["spec_axis"] == "assignment_variant"]
    smoothing_stability = stability.loc[stability["spec_axis"] == "smoothing"]
    area_reference = contrast_df.loc[
        (contrast_df["metric"] == "atlas_region_area_fraction_difference")
        & (contrast_df["variant"] == "plain")
        & (contrast_df["smoothing_mm"] == 2)
    ].copy()
    largest_area = area_reference.iloc[area_reference["estimate"].abs().argmax()] if not area_reference.empty else None
    lowest_text = (
        f"{lowest['group_a']}–{lowest['group_b']} ({lowest['hemisphere']}, ARI={lowest['ari']:.3f})"
        if lowest is not None
        else "not available"
    )
    largest_area_text = (
        f"{largest_area['contrast']} ({largest_area['hemisphere']}, cluster {int(largest_area['cluster'])}, Δfraction={largest_area['estimate']:+.3f})"
        if largest_area is not None
        else "not available"
    )
    variant_ari_min = float(variant_stability["ari"].min()) if not variant_stability.empty else float("nan")
    smoothing_distance = float(smoothing_stability["mean_boundary_distance_mm"].mean()) if not smoothing_stability.empty else float("nan")
    return f"""# Technical summary: HC-referenced Stanford K=3 atlas comparison

## Scope and status

- Run: `{manifest['run_id']}`.
- Inputs: exactly {len(inventory)} cells = 8 groups (HC plus seven cohorts) × 2 assignment variants × 2 smoothings × 2 hemispheres.
- Every cell passed the manifest gate (`status=ok`, `chosen_k_final=3`, ordered labels `[1,2,3]`).
- Geometry: the path in each manifest's `template_left_or_right_surface` was consumed directly. Density evidence was recorded from the manifest; the completed atlas surfaces are 512 vertices and 930 faces.
- Cluster order is treated as head → tail. Atlas region area is computed by triangle area/3 vertex-area allocation; these fields are descriptive atlas geometry and are not Kember native area.
- Boundary distances are symmetric shortest-path distances on the consumed mesh weighted by edge length, summarized by mean, 95th percentile, and Hausdorff distance, only when both maps share the same consumed geometry.
- Reference axes use the HC 2 mm map from the same assignment variant and hemisphere.
- Plain-vs-weighted rows use different consumed surfaces. Their vertex-index label agreement/ARI/Dice are retained, but physical boundary distance and position are unavailable rather than reported as 0 mm.

## Interpretation boundary

These outputs are a descriptive screen of parent/nested group-map shape and specification stability. They are not direct group-level atlas inference: no uncertainty intervals, p values, q values, split-half reliability, ICC, or causal/clinical claims are computed here. The seven direct non-HC-versus-HC comparisons are descriptive group-map summaries only. They are executed by the unified primary run but are not mixed into subject-level FC inference.

## Output tables

- `atlas_inventory.csv`: validated 64-cell inventory and consumed geometry/density evidence.
- `region_area.csv`: namespaced `atlas_region_area_mm2`, `atlas_total_surface_area_mm2`, and `atlas_region_area_fraction` fields.
- `matched_group_pair_comparison.csv`: seven non-HC-versus-HC pairs within each variant/smoothing/hemisphere, including raw agreement, ARI, ordered cluster Dice, boundary Dice, and mesh distances.
- `within_group_spec_stability.csv`: 2 vs 4 mm physical comparisons and plain-vs-weighted vertex-index label comparisons; cross-geometry physical fields are NA.
- `descriptive_contrasts.csv`: seven direct non-HC-minus-HC summaries for area fraction, boundary position, and boundary distance, explicitly marked descriptive-only.

## Reference screen

The mean ARI across seven non-HC-versus-HC pairs and both hemispheres for the reference plain 2 mm maps is `{mean_ari:.3f}`. This is a map-similarity description, not evidence that the group maps are statistically equivalent or different.

## Promising descriptive observations

- Reference-map ARI ranged from `{ari.min():.3f}` to `{ari.max():.3f}`; the lowest reference similarity was `{lowest_text}`.
- The largest absolute reference area-fraction contrast was `{largest_area_text}`. These are map-shape summaries, not corrected group effects.
- Plain-vs-weighted labels were identical by vertex index in the current cells (minimum ARI `{variant_ari_min:.3f}`), but the consumed surfaces differ, so no physical millimetre boundary discrepancy is estimated for that comparison. The mean same-geometry 2-vs-4 mm boundary discrepancy was `{smoothing_distance:.3f}` mm.
"""

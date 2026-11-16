"""Reusable cortical network surface-area primitives.

The module deliberately keeps geometry, label alignment, input provenance, and
area accounting separate.  It is used by the analysis CLI but is also useful
for small synthetic checks and independent redraws.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import re
from pathlib import Path
import warnings
from typing import Mapping, Sequence

import nibabel as nib
import numpy as np


KONG_NATIVE17_ORDER: tuple[str, ...] = (
    "DefaultA", "DefaultB", "DefaultC", "ContA", "ContB", "ContC",
    "SalVenAttnA", "SalVenAttnB", "DorsAttnA", "DorsAttnB",
    "SomMotA", "SomMotB", "VisualA", "VisualB", "VisualC", "Aud",
    "Language",
)
CANONICAL7_ORDER: tuple[str, ...] = (
    "Default", "Visual", "Somatomotor", "DorsalAttention",
    "VentralAttention", "Control", "Language",
)
LYNCH_NATIVE20_ORDER: tuple[str, ...] = (
    "Default_Parietal", "Default_Anterolateral", "Default_Dorsolateral",
    "Default_Retrosplenial", "Visual_Lateral", "Visual_Dorsal/VentralStream",
    "Visual_V5", "Visual_V1", "Frontoparietal", "DorsalAttention",
    "Premotor/DorsalAttentionII", "Language", "Salience",
    "CinguloOpercular/Action-mode", "MedialParietal", "Somatomotor_Hand",
    "Somatomotor_Face", "Somatomotor_Foot", "Auditory", "SomatoCognitiveAction",
)
LYNCH_LABELS_WITH_NOISE: tuple[str, ...] = LYNCH_NATIVE20_ORDER + ("Noise",)


@dataclass(frozen=True)
class SurfaceMesh:
    coordinates: np.ndarray
    triangles: np.ndarray
    hemisphere: str
    source: Path


@dataclass(frozen=True)
class AlignedLabels:
    values: np.ndarray
    label_names: Mapping[int, str]
    valid_anatomical_mask: np.ndarray
    vertex_indices: np.ndarray
    hemisphere: str
    source: Path
    alignment_source: str


@dataclass(frozen=True)
class Taxonomy:
    key: str
    method: str
    classification: str
    network_order: tuple[str, ...]
    label_to_network: Mapping[str, str]
    source: str


@dataclass(frozen=True)
class AreaMetrics:
    network_area_mm2: Mapping[str, float]
    area_percent: Mapping[str, float]
    total_cortical_area_mm2: float
    noise_area_mm2: float
    unassigned_area_mm2: float
    vertex_area_mm2: np.ndarray | None = None
    valid_vertex_count: int = 0

    def assert_conserved(self, *, atol: float = 1e-7) -> None:
        accounted = (
            float(sum(self.network_area_mm2.values()))
            + float(self.noise_area_mm2)
            + float(self.unassigned_area_mm2)
        )
        if not np.isclose(accounted, self.total_cortical_area_mm2, atol=atol, rtol=1e-8):
            raise AssertionError(
                f"Area is not conserved: networks+noise+unassigned={accounted}, "
                f"total={self.total_cortical_area_mm2}"
            )
        if any(value < -atol for value in self.network_area_mm2.values()):
            raise AssertionError("Network area cannot be negative")
        if self.noise_area_mm2 < -atol or self.unassigned_area_mm2 < -atol:
            raise AssertionError("Noise and unassigned area cannot be negative")


@dataclass(frozen=True)
class CohortRoster:
    name: str
    subjects: tuple[str, ...]
    manifest: Path


@dataclass(frozen=True)
class SubjectRecord:
    subject: str
    cohort: str
    dataset: str
    variant: str
    atlas: str
    surface_paths: Mapping[str, Path]
    label_path: Path
    functional_manifest: Path


@dataclass(frozen=True)
class InputLayout:
    hcp_surface_template: str
    hcp_functional_manifest_template: str
    hcp_label_templates: Mapping[str, str]
    iowa_surface_templates: Mapping[str, str]
    iowa_functional_manifest_templates: Mapping[str, str]
    iowa_label_templates: Mapping[str, Mapping[str, str]]


def _as_path(value: str | Path) -> Path:
    return Path(value).expanduser()


def _format_template(template: str, *, subject: str, hemi: str) -> Path:
    return Path(template.format(
        subject=subject,
        sub_subject=subject,
        hemi=hemi,
        hemi_lower=hemi.lower(),
        lh="lh" if hemi == "L" else "rh",
    ))


def compute_vertex_areas(coordinates: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Apportion one third of each triangle area to each incident vertex."""

    points = np.asarray(coordinates, dtype=np.float64)
    faces = np.asarray(triangles)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"coordinates must have shape (n, 3), got {points.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"triangles must have shape (m, 3), got {faces.shape}")
    if not np.isfinite(points).all():
        raise ValueError("coordinates contain non-finite values")
    if not np.issubdtype(faces.dtype, np.integer):
        if not np.all(np.equal(faces, np.floor(faces))):
            raise ValueError("triangles must contain integer vertex indices")
        faces = faces.astype(np.int64)
    else:
        faces = faces.astype(np.int64, copy=False)
    if faces.size and (faces.min() < 0 or faces.max() >= len(points)):
        raise ValueError("triangles contain out-of-range vertex indices")
    if faces.size == 0:
        return np.zeros(len(points), dtype=np.float64)
    vectors_a = points[faces[:, 1]] - points[faces[:, 0]]
    vectors_b = points[faces[:, 2]] - points[faces[:, 0]]
    triangle_areas = 0.5 * np.linalg.norm(np.cross(vectors_a, vectors_b), axis=1)
    if not np.isfinite(triangle_areas).all() or np.any(triangle_areas < 0):
        raise ValueError("triangle areas are non-finite or negative")
    vertex_areas = np.zeros(len(points), dtype=np.float64)
    contribution = triangle_areas / 3.0
    for column in range(3):
        np.add.at(vertex_areas, faces[:, column], contribution)
    return vertex_areas


@lru_cache(maxsize=512)
def load_surface(path: Path, *, hemisphere: str | None = None) -> SurfaceMesh:
    """Load a GIFTI midthickness mesh and validate its geometry."""

    source = _as_path(path).resolve()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"pixdim\[1,2,3\] should be non-zero")
        image = nib.load(str(source))
    pointset = None
    triangles = None
    for array in image.darrays:
        data = np.asarray(array.data)
        intent = str(array.intent)
        if data.ndim == 2 and data.shape[1] == 3 and ("POINTSET" in intent or pointset is None):
            pointset = data
        if data.ndim == 2 and data.shape[1] == 3 and "TRIANGLE" in intent:
            triangles = data
    if pointset is None or triangles is None:
        candidates = [np.asarray(a.data) for a in image.darrays if np.asarray(a.data).ndim == 2]
        if pointset is None:
            pointset = next((a for a in candidates if a.shape[1] == 3 and np.issubdtype(a.dtype, np.floating)), None)
        if triangles is None:
            triangles = next((a for a in candidates if a.shape[1] == 3 and np.issubdtype(a.dtype, np.integer)), None)
    if pointset is None or triangles is None:
        raise ValueError(f"Surface GIFTI does not contain pointset and triangle arrays: {source}")
    if hemisphere is None:
        text = source.name.lower()
        hemisphere = "L" if "hemi-l" in text or "left" in text else "R" if "hemi-r" in text or "right" in text else "?"
    return SurfaceMesh(
        coordinates=np.asarray(pointset, dtype=np.float64),
        triangles=np.asarray(triangles, dtype=np.int32),
        hemisphere=str(hemisphere),
        source=source,
    )


def _label_axis_names(image: nib.Cifti2Image) -> dict[int, str]:
    axis = image.header.get_axis(0)
    labels = getattr(axis, "label", None)
    if labels is None:
        return {}
    table = labels[0] if len(labels) else {}
    result: dict[int, str] = {}
    for key, value in table.items():
        if isinstance(value, tuple):
            result[int(key)] = str(value[0])
        else:
            result[int(key)] = str(value)
    return result


def load_cifti_labels(path: Path, surface_vertex_counts: Mapping[str, int]) -> dict[str, AlignedLabels]:
    counts = tuple(sorted((str(key), int(value)) for key, value in surface_vertex_counts.items()))
    return _load_cifti_labels_cached(str(_as_path(path).resolve()), counts)


@lru_cache(maxsize=512)
def _load_cifti_labels_cached(path_str: str, counts: tuple[tuple[str, int], ...]) -> dict[str, AlignedLabels]:
    return _load_cifti_labels_impl(Path(path_str), dict(counts))


def _load_cifti_labels_impl(path: Path, surface_vertex_counts: Mapping[str, int]) -> dict[str, AlignedLabels]:
    """Load a dlabel using BrainModel vertex indices, never array position alone."""

    source = _as_path(path).resolve()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"pixdim\[1,2,3\] should be non-zero")
        image = nib.load(str(source))
        data = np.asarray(image.get_fdata(dtype=np.float32))
    if data.ndim != 2 or data.shape[0] < 1:
        raise ValueError(f"CIFTI label image must have at least one scalar map: {source}")
    data = np.rint(data[0]).astype(np.int32)
    geometry = image.header.get_axis(1)
    names = _label_axis_names(image)
    result: dict[str, AlignedLabels] = {}
    for structure, slc, subaxis in geometry.iter_structures():
        upper = str(structure).upper()
        if "CORTEX_LEFT" in upper:
            hemi = "L"
        elif "CORTEX_RIGHT" in upper:
            hemi = "R"
        else:
            continue
        if hemi not in surface_vertex_counts:
            raise ValueError(f"CIFTI BrainModel contains {hemi} but no matching surface count")
        expected = int(surface_vertex_counts[hemi])
        indices = np.asarray(getattr(subaxis, "vertex", []), dtype=np.int64)
        if indices.ndim != 1 or len(indices) != len(np.unique(indices)):
            raise ValueError(f"CIFTI BrainModel has invalid {hemi} vertex indices")
        declared = getattr(subaxis, "nvertices", None)
        if isinstance(declared, Mapping):
            declared_values = list(declared.values())
            declared = declared_values[0] if declared_values else None
        if declared is not None and int(declared) != expected:
            raise ValueError(
                f"CIFTI BrainModel {hemi} declares {int(declared)} vertices but surface has {expected}"
            )
        if len(indices) and (indices.min() < 0 or indices.max() >= expected):
            raise ValueError(f"CIFTI BrainModel {hemi} vertex indices exceed surface size {expected}")
        values = data[slc]
        if len(values) != len(indices):
            raise ValueError(f"CIFTI BrainModel {hemi} label count does not match vertex indices")
        full_values = np.zeros(expected, dtype=np.int32)
        full_values[indices] = values
        anatomical = np.zeros(expected, dtype=bool)
        anatomical[indices] = True
        result[hemi] = AlignedLabels(
            values=full_values,
            label_names=names,
            valid_anatomical_mask=anatomical,
            vertex_indices=indices.astype(np.int32),
            hemisphere=hemi,
            source=source,
            alignment_source="CIFTI BrainModel vertex indices",
        )
    if set(result) != {"L", "R"}:
        raise ValueError(f"CIFTI label image must contain both cortical hemispheres: {source}")
    return result


def _gifti_label_names(image: nib.gifti.GiftiImage) -> dict[int, str]:
    result: dict[int, str] = {}
    table = getattr(image, "labeltable", None)
    for label in getattr(table, "labels", []) if table is not None else []:
        key = int(getattr(label, "key", getattr(label, "key", 0)))
        name = getattr(label, "label", None)
        if name is None:
            name = getattr(label, "labelname", "")
        result[key] = str(name)
    return result


def load_gifti_labels(
    path: Path,
    *,
    hemisphere: str,
    surface_vertex_count: int,
    vertex_indices: np.ndarray | None = None,
    anatomical_vertex_indices: np.ndarray | None = None,
) -> AlignedLabels:
    """Load a GIFTI label array with explicit anatomical/index correspondence."""

    source = _as_path(path).resolve()
    image = nib.load(str(source))
    data_array = next((a for a in image.darrays if np.asarray(a.data).ndim == 1), None)
    if data_array is None:
        raise ValueError(f"GIFTI label image has no one-dimensional label array: {source}")
    values = np.rint(np.asarray(data_array.data)).astype(np.int32).reshape(-1)
    if vertex_indices is None:
        if len(values) != int(surface_vertex_count):
            raise ValueError("GIFTI labels require explicit vertex_indices when lengths differ")
        indices = np.arange(surface_vertex_count, dtype=np.int32)
        alignment_source = "GIFTI full vertex array"
    else:
        indices = np.asarray(vertex_indices, dtype=np.int64).reshape(-1)
        if len(indices) != len(values):
            raise ValueError("GIFTI vertex_indices length does not match labels")
        if len(indices) and (indices.min() < 0 or indices.max() >= surface_vertex_count):
            raise ValueError("GIFTI vertex_indices exceed declared surface size")
        if len(indices) != len(np.unique(indices)):
            raise ValueError("GIFTI vertex_indices must be unique")
        alignment_source = "GIFTI explicit vertex indices"
    full_values = np.zeros(int(surface_vertex_count), dtype=np.int32)
    full_values[indices] = values
    if anatomical_vertex_indices is None:
        anatomical = np.zeros(int(surface_vertex_count), dtype=bool)
        anatomical[indices] = True
    else:
        anatomical_indices = np.asarray(anatomical_vertex_indices, dtype=np.int64).reshape(-1)
        if len(anatomical_indices) and (anatomical_indices.min() < 0 or anatomical_indices.max() >= surface_vertex_count):
            raise ValueError("GIFTI anatomical_vertex_indices exceed surface size")
        anatomical = np.zeros(int(surface_vertex_count), dtype=bool)
        anatomical[anatomical_indices] = True
    return AlignedLabels(
        values=full_values,
        label_names=_gifti_label_names(image),
        valid_anatomical_mask=anatomical,
        vertex_indices=indices.astype(np.int32),
        hemisphere=str(hemisphere),
        source=source,
        alignment_source=alignment_source,
    )


def load_text_labels(
    path: Path,
    *,
    hemisphere: str,
    surface_vertex_count: int,
    label_names: Mapping[int, str] | None = None,
    anatomical_vertex_indices: np.ndarray | None = None,
) -> AlignedLabels:
    """Load the plain newline label arrays produced by ArealMSHBM."""

    source = _as_path(path).resolve()
    try:
        values = np.loadtxt(source, dtype=np.int64, ndmin=1).reshape(-1)
    except Exception as exc:
        raise ValueError(f"Could not parse text labels: {source}") from exc
    if len(values) != int(surface_vertex_count):
        raise ValueError(
            f"Text labels contain {len(values)} values but surface has {surface_vertex_count}; "
            "an explicit vertex mapping is required"
        )
    if np.any(values < 0) or not np.isfinite(values).all():
        raise ValueError(f"Text labels contain invalid values: {source}")
    if anatomical_vertex_indices is None:
        anatomical = np.ones(int(surface_vertex_count), dtype=bool)
    else:
        anatomical = np.zeros(int(surface_vertex_count), dtype=bool)
        indices = np.asarray(anatomical_vertex_indices, dtype=np.int64).reshape(-1)
        if len(indices) and (indices.min() < 0 or indices.max() >= surface_vertex_count):
            raise ValueError("Text anatomical_vertex_indices exceed surface size")
        anatomical[indices] = True
    return AlignedLabels(
        values=values.astype(np.int32),
        label_names=dict(label_names or {}),
        valid_anatomical_mask=anatomical,
        vertex_indices=np.arange(surface_vertex_count, dtype=np.int32),
        hemisphere=str(hemisphere),
        source=source,
        alignment_source="explicit hemisphere vertex numbering (text array)",
    )


@lru_cache(maxsize=16)
def _load_external_kong_names(repo_root: Path) -> dict[int, str]:
    path = Path(repo_root) / "external" / "Schaefer2018_Kong2022" / "Schaefer2018_400Parcels_Kong2022_17Networks_order.dlabel.nii"
    if not path.is_file():
        raise FileNotFoundError(f"Missing Kong label registry required for text labels: {path}")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"pixdim\[1,2,3\] should be non-zero")
        names = _label_axis_names(nib.load(str(path)))
    if len(names) < 400:
        raise ValueError(f"Kong label registry has only {len(names)} labels: {path}")
    return names


def _kong_family_mapping() -> dict[str, str]:
    return {
        "DefaultA": "Default", "DefaultB": "Default", "DefaultC": "Default",
        "ContA": "Control", "ContB": "Control", "ContC": "Control",
        "SalVenAttnA": "VentralAttention", "SalVenAttnB": "VentralAttention",
        "DorsAttnA": "DorsalAttention", "DorsAttnB": "DorsalAttention",
        "SomMotA": "Somatomotor", "SomMotB": "Somatomotor", "Aud": "Somatomotor",
        "VisualA": "Visual", "VisualB": "Visual", "VisualC": "Visual",
        "Language": "Language",
    }


def _lynch_mapping() -> dict[str, str]:
    return {
        "Default_Parietal": "Default", "Default_Anterolateral": "Default",
        "Default_Dorsolateral": "Default", "Default_Retrosplenial": "Default",
        "MedialParietal": "Default", "Visual_Lateral": "Visual",
        "Visual_Dorsal/VentralStream": "Visual", "Visual_V5": "Visual",
        "Visual_V1": "Visual", "Frontoparietal": "Control",
        "CinguloOpercular/Action-mode": "Control", "DorsalAttention": "DorsalAttention",
        "Premotor/DorsalAttentionII": "DorsalAttention", "Language": "Language",
        "Salience": "VentralAttention", "Somatomotor_Hand": "Somatomotor",
        "Somatomotor_Face": "Somatomotor", "Somatomotor_Foot": "Somatomotor",
        "Auditory": "Somatomotor", "SomatoCognitiveAction": "Somatomotor",
        "Noise": "Noise",
    }


def load_taxonomies(repo_root: Path) -> dict[str, Taxonomy]:
    """Return the four fixed native/canonical taxonomy definitions."""

    root = Path(repo_root).resolve()
    config_path = root / "config" / "cross_atlas_network_merge.json"
    source = str(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing canonical network mapping: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    source = f"{config_path} (cross_atlas_network_merge.json)"
    configured_kong = {str(k): str(v) for k, v in payload["atlases"]["kong2022"]["mapping"].items()}
    if set(configured_kong) != set(KONG_NATIVE17_ORDER):
        missing = sorted(set(KONG_NATIVE17_ORDER) - set(configured_kong))
        extra = sorted(set(configured_kong) - set(KONG_NATIVE17_ORDER))
        raise ValueError(f"Kong mapping keys do not match native17 order: missing={missing}, extra={extra}")
    # Native 17 is the identity over Kong family tokens; canonical7 uses the
    # same fixed order and the repository's Auditory -> Somatomotor rule.
    native_kong = {name: name for name in KONG_NATIVE17_ORDER}
    canonical_kong = {name: _kong_family_mapping()[name] for name in KONG_NATIVE17_ORDER}
    native_lynch = {name: name for name in LYNCH_NATIVE20_ORDER}
    canonical_lynch = _lynch_mapping()
    return {
        "kong_native17": Taxonomy("kong_native17", "Kong2022", "native17", KONG_NATIVE17_ORDER, native_kong, source),
        "kong_canonical7": Taxonomy("kong_canonical7", "Kong2022", "canonical7", CANONICAL7_ORDER, canonical_kong, source),
        "lynch_native20": Taxonomy("lynch_native20", "Lynch2024", "native20", LYNCH_NATIVE20_ORDER, native_lynch, source),
        "lynch_canonical7": Taxonomy("lynch_canonical7", "Lynch2024", "canonical7", CANONICAL7_ORDER, canonical_lynch, source),
    }


def _family_from_label(label_name: str) -> str | None:
    match = re.search(r"17networks_(?:LH|RH)_([A-Za-z0-9]+)(?:_|$)", str(label_name))
    if match:
        return match.group(1)
    return str(label_name) if str(label_name) in KONG_NATIVE17_ORDER else None


def _network_for_label(label_name: str, taxonomy: Taxonomy) -> str | None:
    name = str(label_name)
    if name.strip().lower() == "noise":
        return "Noise"
    if taxonomy.method == "Kong2022":
        family = _family_from_label(name)
        if family is None:
            family = name
        return taxonomy.label_to_network.get(family)
    if name in taxonomy.label_to_network:
        return taxonomy.label_to_network[name]
    # Some label lists use a short ``Def_Par`` spelling; map it back to the
    # formal Lynch label names without treating unknown labels as a network.
    short = {
        "Def_Par": "Default_Parietal", "AntLat": "Default_Anterolateral",
        "DorsLat": "Default_Dorsolateral", "Retro": "Default_Retrosplenial",
        "MedPar": "MedialParietal", "Vis_Lateral": "Visual_Lateral",
        "DV_Stream": "Visual_Dorsal/VentralStream", "Vis_V5": "Visual_V5",
        "Vis_V1": "Visual_V1", "PM/DorsAttnII": "Premotor/DorsalAttentionII",
        "CinguloOpercular": "CinguloOpercular/Action-mode", "SCAN": "SomatoCognitiveAction",
    }
    formal = short.get(name, name)
    return taxonomy.label_to_network.get(formal)


def calculate_area_metrics(
    *,
    meshes: Mapping[str, SurfaceMesh],
    labels: Mapping[str, AlignedLabels],
    taxonomy: Taxonomy,
    vertex_areas: Mapping[str, np.ndarray] | None = None,
) -> AreaMetrics:
    """Calculate network areas and percentages for one subject/taxonomy."""

    network_area = {name: 0.0 for name in taxonomy.network_order}
    total = 0.0
    noise = 0.0
    unassigned = 0.0
    all_vertex_areas: list[np.ndarray] = []
    valid_count = 0
    hemispheres = tuple(hemi for hemi in ("L", "R") if hemi in meshes and hemi in labels)
    if not hemispheres:
        raise ValueError("At least one matching mesh and label hemisphere is required")
    for hemi in hemispheres:
        mesh = meshes[hemi]
        aligned = labels[hemi]
        areas = (
            np.asarray(vertex_areas[hemi], dtype=np.float64)
            if vertex_areas is not None and hemi in vertex_areas
            else compute_vertex_areas(mesh.coordinates, mesh.triangles)
        )
        if len(aligned.values) != len(areas) or len(aligned.valid_anatomical_mask) != len(areas):
            raise ValueError(
                f"Label/geometry vertex mismatch for {hemi}: labels={len(aligned.values)}, mesh={len(areas)}"
            )
        if len(aligned.vertex_indices) and (
            aligned.vertex_indices.min() < 0 or aligned.vertex_indices.max() >= len(areas)
        ):
            raise ValueError(f"Label vertex indices exceed mesh for {hemi}")
        valid = np.asarray(aligned.valid_anatomical_mask, dtype=bool)
        valid_count += int(valid.sum())
        total += float(areas[valid].sum())
        all_vertex_areas.append(areas)
        for index in np.flatnonzero(valid):
            amount = float(areas[index])
            label_id = int(aligned.values[index])
            if label_id == 0:
                unassigned += amount
                continue
            network = _network_for_label(aligned.label_names.get(label_id, ""), taxonomy)
            if network == "Noise":
                noise += amount
            elif network in network_area:
                network_area[network] += amount
            else:
                unassigned += amount
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"Valid cortical area must be finite and positive, got {total}")
    percentages = {name: 100.0 * value / total for name, value in network_area.items()}
    metrics = AreaMetrics(
        network_area_mm2=network_area,
        area_percent=percentages,
        total_cortical_area_mm2=total,
        noise_area_mm2=noise,
        unassigned_area_mm2=unassigned,
        vertex_area_mm2=np.concatenate(all_vertex_areas),
        valid_vertex_count=valid_count,
    )
    metrics.assert_conserved(atol=max(1e-7, total * 1e-10))
    return metrics


def _formal_manifest_paths(root: Path) -> dict[str, Path]:
    return {
        "HCP HC": root / "outputs_hcp" / "inputs" / "kong2022" / "manifests" / "hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt",
        "Iowa HC": root / "manifests" / "jj_tms_hc_subjects.txt",
        "Patient": root / "manifests" / "jj_tms_patient_subjects.txt",
        "Pure MDD": root / "manifests" / "jj_tms_pure_mdd_subjects.txt",
        "Comorbid MDD": root / "manifests" / "jj_tms_comorbid_mdd_subjects.txt",
    }


def load_formal_rosters(repo_root: Path) -> dict[str, CohortRoster]:
    rosters: dict[str, CohortRoster] = {}
    for name, manifest in _formal_manifest_paths(Path(repo_root).resolve()).items():
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing formal cohort manifest: {manifest}")
        subjects = tuple(
            f"sub-{line.strip().removeprefix('sub-')}"
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        if not subjects or len(set(subjects)) != len(subjects):
            raise ValueError(f"Formal roster {manifest} must contain unique subjects, got {len(subjects)}")
        if name == "Patient" and "sub-3003" in subjects:
            raise ValueError("Patient formal roster must exclude sub-3003")
        rosters[name] = CohortRoster(name, subjects, manifest)
    return rosters


def default_input_layout(repo_root: Path) -> InputLayout:
    root = Path(repo_root).resolve()
    jiang = Path("/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS")
    if not jiang.exists():
        jiang = Path("/Shared/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS")
    hcp_surface = str(root / "data" / "hippunfold_input" / "{sub_subject}" / "anat" / "{sub_subject}_hemi-{hemi}_space-fsLR_den-32k_desc-MSMAll_midthickness.surf.gii")
    hcp_manifest = str(root / "outputs_hcp" / "work" / "cortex_pfm" / "{sub_subject}" / "lynch2024" / "fastans_pfm_manifest.json")
    hcp_labels = {
        "kong2022": str(root / "outputs_hcp" / "inputs" / "kong2022" / "cortex_references" / "routea_areal_mshbm" / "{sub_subject}" / "{hemi_lower}_labels.txt"),
        "lynch2024": str(root / "outputs_hcp" / "work" / "cortex_pfm" / "{sub_subject}" / "lynch2024" / "PFM_Lynch2024priors.dlabel.nii"),
    }
    iowa_surface = {
        "36p": str(jiang / "derivatives_surf" / "{sub_subject}" / "anat" / "{sub_subject}_hemi-{hemi}_space-fsLR_den-32k_midthickness.surf.gii"),
    }
    iowa_manifest = {
        "36p": str(jiang / "ArealMSHBM_Kong2022" / "_manifests" / "jj_tms_pinc_20260607_000010.txt"),
    }
    iowa_labels = {
        "36p": {
            "kong2022": str(jiang / "ArealMSHBM_Kong2022" / "{sub_subject}"),
            "lynch2024": str(root / "outputs_iowa_jj_tms" / "work" / "cortex_pfm" / "lynch2024_iowa3t" / "{sub_subject}" / "lynch2024" / "PFM_Lynch2024priors.dlabel.nii"),
        },
    }
    return InputLayout(hcp_surface, hcp_manifest, hcp_labels, iowa_surface, iowa_manifest, iowa_labels)


def resolve_subject_record(
    subject: str,
    cohort: str,
    variant: str,
    atlas: str,
    layout: InputLayout | None = None,
    *,
    repo_root: Path | None = None,
) -> SubjectRecord:
    """Resolve one formal subject without cross-branch fallback."""

    normalized = f"sub-{str(subject).removeprefix('sub-')}"
    variant_key = str(variant).lower()
    if variant_key != "36p":
        raise ValueError(f"Unsupported variant: {variant}")
    if layout is None:
        layout = default_input_layout(repo_root or Path(__file__).resolve().parents[3])
    if cohort == "HCP HC":
        surfaces = {hemi: _format_template(layout.hcp_surface_template, subject=normalized, hemi=hemi) for hemi in ("L", "R")}
        label_template = layout.hcp_label_templates[str(atlas).lower()]
        label_path = _format_template(label_template, subject=normalized, hemi="L")
        # A CIFTI dlabel is one path; a pair of HCP text labels uses the
        # directory/template convention and is handled by the loader.
        if label_path.suffix == ".txt":
            label_path = label_path.parent
        functional = _format_template(layout.hcp_functional_manifest_template, subject=normalized, hemi="L")
        return SubjectRecord(normalized, cohort, "hcp", variant_key, str(atlas).lower(), surfaces, label_path, functional)
    if cohort not in {"Iowa HC", "Patient", "Pure MDD", "Comorbid MDD"}:
        raise ValueError(f"Unknown cohort: {cohort}")
    surfaces = {hemi: _format_template(layout.iowa_surface_templates[variant_key], subject=normalized, hemi=hemi) for hemi in ("L", "R")}
    label_template = layout.iowa_label_templates[variant_key][str(atlas).lower()]
    label_path = _format_template(label_template, subject=normalized, hemi="L")
    if label_path.suffix == ".txt":
        label_path = label_path.parent
    functional = _format_template(layout.iowa_functional_manifest_templates[variant_key], subject=normalized, hemi="L")
    return SubjectRecord(normalized, cohort, "iowa", variant_key, str(atlas).lower(), surfaces, label_path, functional)


def _label_input_paths(record: SubjectRecord) -> dict[str, Path] | None:
    path = Path(record.label_path)
    if path.is_dir():
        return {"L": path / "lh_labels.txt", "R": path / "rh_labels.txt"}
    if path.name.endswith("lh_labels.txt") or path.name.endswith("lh.label.txt"):
        return {"L": path, "R": path.with_name(path.name.replace("lh_", "rh_", 1).replace("lh.", "rh.", 1))}
    if path.suffix == ".txt":
        return {"L": path, "R": path.with_name("rh_labels.txt")}
    return None


def _load_record_labels(record: SubjectRecord, meshes: Mapping[str, SurfaceMesh], repo_root: Path) -> dict[str, AlignedLabels]:
    label_paths = _label_input_paths(record)
    if label_paths is not None:
        names = _load_external_kong_names(repo_root) if record.atlas == "kong2022" else {}
        return {
            hemi: load_text_labels(
                path,
                hemisphere=hemi,
                surface_vertex_count=len(meshes[hemi].coordinates),
                label_names=names,
            )
            for hemi, path in label_paths.items()
        }
    counts = {hemi: len(mesh.coordinates) for hemi, mesh in meshes.items()}
    if str(record.label_path).endswith(".dlabel.nii"):
        return load_cifti_labels(record.label_path, counts)
    if Path(record.label_path).suffix == ".npy":
        array = np.asarray(np.load(record.label_path)).reshape(-1)
        expected = sum(counts.values())
        if len(array) != expected:
            raise ValueError(f"NPY labels contain {len(array)} values, expected {expected}")
        # NPY cortex vectors use the same explicit L/R split as CIFTI 32k.
        names = _load_external_kong_names(repo_root) if record.atlas == "kong2022" else {}
        offset = 0
        result = {}
        for hemi in ("L", "R"):
            n = counts[hemi]
            result[hemi] = AlignedLabels(
                array[offset:offset+n].astype(np.int32), names, np.ones(n, dtype=bool), np.arange(n, dtype=np.int32), hemi, Path(record.label_path), "explicit L/R NPY split",
            )
            offset += n
        return result
    raise ValueError(f"Unsupported label input: {record.label_path}")


@lru_cache(maxsize=512)
def _cached_vertex_areas(path_str: str, hemisphere: str) -> np.ndarray:
    mesh = load_surface(Path(path_str), hemisphere=hemisphere)
    return compute_vertex_areas(mesh.coordinates, mesh.triangles)


def calculate_subject_record(
    record: SubjectRecord,
    *,
    taxonomies: Mapping[str, Taxonomy] | None = None,
    repo_root: Path | None = None,
) -> dict[str, AreaMetrics]:
    root = Path(repo_root or Path(__file__).resolve().parents[3]).resolve()
    meshes = {hemi: load_surface(path, hemisphere=hemi) for hemi, path in record.surface_paths.items()}
    labels = _load_record_labels(record, meshes, root)
    all_taxonomies = dict(taxonomies or load_taxonomies(root))
    prefix = "kong_" if str(record.atlas).lower().startswith("kong") else "lynch_"
    selected = {key: value for key, value in all_taxonomies.items() if key.startswith(prefix)}
    if not selected:
        raise ValueError(f"No taxonomies available for atlas {record.atlas}")
    vertex_areas = {
        hemi: _cached_vertex_areas(str(mesh.source), hemi)
        for hemi, mesh in meshes.items()
    }
    return {
        key: calculate_area_metrics(
            meshes=meshes, labels=labels, taxonomy=taxonomy, vertex_areas=vertex_areas
        )
        for key, taxonomy in selected.items()
    }



def _validate_cifti_alignment_header(path: Path, surface_vertex_counts: Mapping[str, int]) -> None:
    """Validate CIFTI BrainModel geometry without reading the full label array."""

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"pixdim\[1,2,3\] should be non-zero")
        image = nib.load(str(Path(path).resolve()))
    geometry = image.header.get_axis(1)
    found: set[str] = set()
    for structure, _slc, subaxis in geometry.iter_structures():
        upper = str(structure).upper()
        hemi = "L" if "CORTEX_LEFT" in upper else "R" if "CORTEX_RIGHT" in upper else None
        if hemi is None:
            continue
        found.add(hemi)
        expected = int(surface_vertex_counts[hemi])
        declared = getattr(subaxis, "nvertices", None)
        if isinstance(declared, Mapping):
            declared = next(iter(declared.values()), None)
        if declared is not None and int(declared) != expected:
            raise ValueError(
                f"CIFTI BrainModel {hemi} declares {int(declared)} vertices but surface has {expected}"
            )
        indices = np.asarray(getattr(subaxis, "vertex", []), dtype=np.int64)
        if len(indices) != len(np.unique(indices)) or (len(indices) and (indices.min() < 0 or indices.max() >= expected)):
            raise ValueError(f"CIFTI BrainModel {hemi} vertex indices are incompatible with surface size {expected}")
    if found != {"L", "R"}:
        raise ValueError(f"CIFTI label image must contain both cortical hemispheres: {path}")


def audit_subject_records(records: Sequence[SubjectRecord]) -> list[dict[str, object]]:
    """Return one explicit audit row per input record, including failures."""

    rows: list[dict[str, object]] = []
    repo_root = Path(__file__).resolve().parents[3]
    for record in records:
        missing: list[str] = []
        for hemi in ("L", "R"):
            if not Path(record.surface_paths[hemi]).is_file():
                missing.append(f"surface_{hemi}")
        label_paths = _label_input_paths(record)
        if label_paths is None:
            if not Path(record.label_path).is_file():
                missing.append("label")
        else:
            if not all(path.is_file() for path in label_paths.values()):
                missing.append("label")
        if not Path(record.functional_manifest).is_file():
            missing.append("functional_manifest")
        row: dict[str, object] = {
            "subject": record.subject,
            "cohort": record.cohort,
            "dataset": record.dataset,
            "variant": record.variant,
            "atlas": record.atlas,
            "surface_L": str(record.surface_paths["L"]),
            "surface_R": str(record.surface_paths["R"]),
            "label": str(record.label_path),
            "functional_manifest": str(record.functional_manifest),
            "anatomical_mask_source": (
                "CIFTI BrainModel cortical vertices"
                if str(record.label_path).endswith(".dlabel.nii")
                else "explicit full fsLR hemisphere vertex numbering"
            ),
            "missing_inputs": missing,
            "status": "missing" if missing else "ready",
            "error": "",
        }
        if not missing:
            try:
                left = load_surface(record.surface_paths["L"], hemisphere="L")
                right = load_surface(record.surface_paths["R"], hemisphere="R")
                # Validate finite, non-negative vertex areas during readiness;
                # the same arrays are cached and reused by calculation.
                _cached_vertex_areas(str(left.source), "L")
                _cached_vertex_areas(str(right.source), "R")
                counts = {"L": len(left.coordinates), "R": len(right.coordinates)}
                if str(record.label_path).endswith(".dlabel.nii"):
                    _validate_cifti_alignment_header(record.label_path, counts)
                else:
                    labels = _load_record_labels(record, {"L": left, "R": right}, repo_root)
                    for hemi in ("L", "R"):
                        if len(labels[hemi].values) != len((left if hemi == "L" else right).coordinates):
                            raise ValueError(f"label/geometry vertex count mismatch for {hemi}")
            except Exception as exc:
                row["status"] = "incompatible"
                row["error"] = str(exc)
        rows.append(row)
    return rows

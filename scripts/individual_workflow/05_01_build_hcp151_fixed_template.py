#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_WORKFLOW_DIR = REPO_ROOT / "scripts" / "core_workflow"
if str(CORE_WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_WORKFLOW_DIR))

from lib.template_matching import build_cluster_prototypes
from lib.gifti_contracts import validate_gifti_pointset


TEMPLATE_ID = "hcp151_routea_full_k3_fisherz"
CTXREG_TEMPLATE_ID = "hcp151_routea_ctxreg"
TEMPLATE_ID_CHOICES = (TEMPLATE_ID, CTXREG_TEMPLATE_ID)
SMOOTHINGS = ("2mm", "4mm")
HEMIS = ("L", "R")
N_SUBJECTS = 151
N_VERTICES = 512
N_ROIS = 400
K = 3
DEFAULT_GROUP_ROOT = (
    REPO_ROOT
    / "outputs_hcp"
    / "work"
    / "group_atlas"
    / "kong2022_7net_full_k3_fisherz"
)
DEFAULT_GRADIENT_ROOT = (
    REPO_ROOT / "outputs_hcp" / "work" / "hippomaps_gradients" / "routea"
)
DEFAULT_OUT_ROOT = REPO_ROOT / "outputs_hcp" / "templates" / TEMPLATE_ID


def _validate_template_id(raw: object) -> str:
    if not isinstance(raw, str) or raw not in TEMPLATE_ID_CHOICES:
        raise ValueError(
            "template_id must be one of the supported identities: "
            + ", ".join(TEMPLATE_ID_CHOICES)
        )
    return raw


def _signal_branch_for_template(template_id: str) -> str:
    return "ctxreg" if template_id == CTXREG_TEMPLATE_ID else "standard"


def _normalized_subjects(raw: object, *, manifest_path: Path) -> tuple[str, ...]:
    if not isinstance(raw, list) or len(raw) != N_SUBJECTS:
        raise ValueError(f"Expected exactly 151 subjects in {manifest_path}")
    subjects = tuple(str(value).strip().removeprefix("sub-") for value in raw)
    if any(not subject or not subject.isdigit() for subject in subjects):
        raise ValueError(f"Invalid subject ID in {manifest_path}")
    if len(set(subjects)) != N_SUBJECTS:
        raise ValueError(f"Expected 151 unique subjects in {manifest_path}")
    return subjects


def _surface_vertex_count(path: Path) -> int:
    points = validate_gifti_pointset(path)
    return int(points.shape[0])


@dataclass(frozen=True)
class ComboSources:
    smoothing: str
    hemi: str
    group_run: Path
    gradient_run: Path
    group_manifest: Path
    gradient_manifest: Path
    labels: Path
    support: Path
    group_fc: Path
    roi_registry: Path
    surface: Path
    subjects: tuple[str, ...]


def _fingerprint(path: Path) -> dict[str, object]:
    resolved = Path(path).resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(resolved),
        "sha256": digest.hexdigest(),
        "size_bytes": int(resolved.stat().st_size),
    }


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _group_run(group_root: Path, smoothing: str, hemi: str) -> Path:
    return (
        Path(group_root)
        / "gradient-spectral"
        / "kong2022"
        / smoothing
        / "cov_0p80"
        / "g1g2g3"
        / f"hemi_{hemi}"
    )


def _validate_registry(path: Path) -> None:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        rows = list(reader)
    if "roi_id" not in list(reader.fieldnames or []):
        raise ValueError(f"ROI registry is missing roi_id: {path}")
    try:
        roi_ids = [int(row["roi_id"]) for row in rows]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"ROI registry has invalid roi_id values: {path}") from exc
    if roi_ids != list(range(1, N_ROIS + 1)):
        raise ValueError(
            f"ROI registry must contain fixed ordered ids 1..{N_ROIS}: {path}"
        )


def _require_manifest_value(
    manifest: Mapping[str, object],
    key: str,
    expected: object,
    path: Path,
) -> None:
    if manifest.get(key) != expected:
        raise ValueError(
            f"Expected {key}={expected!r} in {path}, got {manifest.get(key)!r}"
        )


def _validate_template_source_identity(
    *,
    group_manifest: Mapping[str, object],
    gradient_manifest: Mapping[str, object],
    group_manifest_path: Path,
    gradient_manifest_path: Path,
    template_id: str,
) -> None:
    if template_id != CTXREG_TEMPLATE_ID:
        return
    _require_manifest_value(
        gradient_manifest,
        "route",
        "routea_ctxreg",
        gradient_manifest_path,
    )
    group_signature = group_manifest.get("resume_input_signature_payload")
    if not isinstance(group_signature, Mapping):
        raise ValueError(
            "CtxReg template requires resume_input_signature_payload.route_tag="
            f"'routea_ctxreg' in {group_manifest_path}"
        )
    if group_signature.get("route_tag") != "routea_ctxreg":
        raise ValueError(
            "CtxReg template requires group source route_tag='routea_ctxreg' "
            f"in {group_manifest_path}, got {group_signature.get('route_tag')!r}"
        )
    for manifest, path in (
        (group_manifest, group_manifest_path),
        (gradient_manifest, gradient_manifest_path),
    ):
        declared_branch = manifest.get("signal_branch")
        if declared_branch is not None and declared_branch != "ctxreg":
            raise ValueError(
                f"CtxReg template source has incompatible signal_branch in {path}: "
                f"{declared_branch!r}"
            )


def _path_from_manifest(
    manifest: Mapping[str, object],
    key: str,
    expected: Path,
    manifest_path: Path,
) -> Path:
    raw = manifest.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"Missing {key} in {manifest_path}")
    recorded = Path(raw).resolve()
    if recorded != Path(expected).resolve():
        raise ValueError(
            f"Recorded {key} path mismatch in {manifest_path}: "
            f"recorded={recorded} expected={Path(expected).resolve()}"
        )
    return recorded


def _validate_source_combo(
    *,
    group_root: Path,
    gradient_root: Path,
    smoothing: str,
    hemi: str,
    template_id: str = TEMPLATE_ID,
) -> ComboSources:
    template_id = _validate_template_id(template_id)
    if smoothing not in SMOOTHINGS:
        raise ValueError(f"Unsupported smoothing: {smoothing}")
    if hemi not in HEMIS:
        raise ValueError(f"Unsupported hemi: {hemi}")
    group_run = _group_run(group_root, smoothing, hemi)
    gradient_run = Path(gradient_root) / smoothing / f"hemi_{hemi}"
    group_manifest_path = group_run / "group_atlas_manifest.json"
    gradient_manifest_path = gradient_run / "hippomaps_gradient_manifest.json"
    group_manifest = _read_json(group_manifest_path)
    gradient_manifest = _read_json(gradient_manifest_path)

    for manifest, path in (
        (group_manifest, group_manifest_path),
        (gradient_manifest, gradient_manifest_path),
    ):
        _require_manifest_value(manifest, "status", "ok", path)
        _require_manifest_value(manifest, "smoothing", smoothing, path)
        _require_manifest_value(manifest, "hemi", hemi, path)
        _require_manifest_value(manifest, "n_subjects", N_SUBJECTS, path)
        _normalized_subjects(manifest.get("subjects"), manifest_path=path)

    _validate_template_source_identity(
        group_manifest=group_manifest,
        gradient_manifest=gradient_manifest,
        group_manifest_path=group_manifest_path,
        gradient_manifest_path=gradient_manifest_path,
        template_id=template_id,
    )
    _require_manifest_value(group_manifest, "method", "gradient-spectral", group_manifest_path)
    _require_manifest_value(group_manifest, "atlas", "kong2022", group_manifest_path)
    _require_manifest_value(group_manifest, "chosen_k_final", K, group_manifest_path)
    _require_manifest_value(
        group_manifest,
        "cluster_label_ordering",
        "long_axis_head_to_tail",
        group_manifest_path,
    )
    group_subjects = _normalized_subjects(
        group_manifest["subjects"], manifest_path=group_manifest_path
    )
    gradient_subjects = _normalized_subjects(
        gradient_manifest["subjects"], manifest_path=gradient_manifest_path
    )
    if group_subjects != gradient_subjects:
        raise ValueError(
            f"Group/gradient subject mismatch for {smoothing} hemi-{hemi}"
        )
    fisher_definition = gradient_manifest.get("fisher_z_preprocessing")
    if not isinstance(fisher_definition, str) or "arctanh" not in fisher_definition:
        raise ValueError(f"Missing Fisher-z aggregation definition: {gradient_manifest_path}")

    labels_path = group_run / "group_labels_final.npy"
    support_path = group_run / "group_support_mask.npy"
    group_fc_path = gradient_run / "group_mean_fisherz_fc_full.npy"
    _path_from_manifest(
        group_manifest,
        "group_labels_final",
        labels_path,
        group_manifest_path,
    )
    _path_from_manifest(
        group_manifest,
        "group_support_mask",
        support_path,
        group_manifest_path,
    )
    labels = np.load(labels_path, allow_pickle=False)
    support_raw = np.load(support_path, allow_pickle=False)
    group_fc = np.load(group_fc_path, allow_pickle=False)
    if labels.shape != (N_VERTICES,) or labels.dtype != np.dtype(np.int32):
        raise ValueError(
            f"Expected labels shape=(512,) dtype=int32 in {labels_path}; "
            f"got shape={labels.shape} dtype={labels.dtype}"
        )
    if support_raw.shape != (N_VERTICES,) or support_raw.dtype not in (
        np.dtype(np.uint8),
        np.dtype(bool),
    ):
        raise ValueError(
            f"Expected support shape=(512,) dtype=uint8/bool in {support_path}; "
            f"got shape={support_raw.shape} dtype={support_raw.dtype}"
        )
    if group_fc.shape != (N_VERTICES, N_ROIS) or group_fc.dtype != np.dtype(np.float32):
        raise ValueError(
            f"Expected group Fisher-z FC shape=(512, 400) dtype=float32 in {group_fc_path}; "
            f"got shape={group_fc.shape} dtype={group_fc.dtype}"
        )
    support = support_raw.astype(bool, copy=False)
    supported_labels = set(int(value) for value in np.unique(labels[support]))
    if supported_labels != {1, 2, 3}:
        raise ValueError(
            f"Expected supported AP labels 1..3 (K=3) in {labels_path}; "
            f"got {sorted(supported_labels)}"
        )

    density = group_manifest.get("density_evidence")
    if not isinstance(density, Mapping) or int(density.get("n_vertices", -1)) != N_VERTICES:
        raise ValueError(f"Expected consumed den-512 density evidence in {group_manifest_path}")
    surface_raw = density.get("surface_path")
    if not isinstance(surface_raw, str) or not surface_raw.strip():
        raise ValueError(f"Missing consumed surface path in {group_manifest_path}")
    surface = Path(surface_raw).resolve()
    _path_from_manifest(
        group_manifest,
        "template_left_or_right_surface",
        surface,
        group_manifest_path,
    )
    if "_den-512_" not in surface.name or not surface.is_file():
        raise ValueError(f"Consumed surface is not an existing den-512 asset: {surface}")
    surface_vertices = _surface_vertex_count(surface)
    if surface_vertices != N_VERTICES:
        raise ValueError(
            f"Consumed surface point count is not 512: {surface} ({surface_vertices})"
        )

    registry_raw = group_manifest.get("cluster_roi_registry")
    if not isinstance(registry_raw, str) or not registry_raw.strip():
        raise ValueError(f"Missing cluster_roi_registry in {group_manifest_path}")
    registry = Path(registry_raw).resolve()
    _validate_registry(registry)
    return ComboSources(
        smoothing=smoothing,
        hemi=hemi,
        group_run=group_run.resolve(),
        gradient_run=gradient_run.resolve(),
        group_manifest=group_manifest_path.resolve(),
        gradient_manifest=gradient_manifest_path.resolve(),
        labels=labels_path.resolve(),
        support=support_path.resolve(),
        group_fc=group_fc_path.resolve(),
        roi_registry=registry,
        surface=surface,
        subjects=group_subjects,
    )


def preflight_sources(
    *,
    group_root: Path,
    gradient_root: Path,
    template_id: str = TEMPLATE_ID,
) -> list[ComboSources]:
    template_id = _validate_template_id(template_id)
    rows = [
        _validate_source_combo(
            group_root=Path(group_root),
            gradient_root=Path(gradient_root),
            smoothing=smoothing,
            hemi=hemi,
            template_id=template_id,
        )
        for smoothing in SMOOTHINGS
        for hemi in HEMIS
    ]
    expected_subjects = rows[0].subjects
    if any(row.subjects != expected_subjects for row in rows[1:]):
        raise ValueError("The four HCP source combinations do not use the same 151 subjects")
    registry_sha = _fingerprint(rows[0].roi_registry)["sha256"]
    for row in rows[1:]:
        if _fingerprint(row.roi_registry)["sha256"] != registry_sha:
            raise ValueError("The four HCP source combinations have different ROI registries")
    return rows


def _write_npy_exclusive(path: Path, array: np.ndarray) -> None:
    with Path(path).open("xb") as stream:
        np.save(stream, array, allow_pickle=False)


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _combo_prefix(out_root: Path, smoothing: str, hemi: str) -> Path:
    return Path(out_root) / f"hcp151_{smoothing}_hemi-{hemi}"


def build_template(
    *,
    group_root: Path,
    gradient_root: Path,
    out_root: Path,
    template_id: str = TEMPLATE_ID,
) -> dict[str, object]:
    template_id = _validate_template_id(template_id)
    rows = preflight_sources(
        group_root=group_root,
        gradient_root=gradient_root,
        template_id=template_id,
    )
    output_root = Path(out_root).resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    combinations: list[dict[str, object]] = []
    for row in rows:
        labels = np.load(row.labels, allow_pickle=False).astype(np.int32, copy=False)
        support = np.load(row.support, allow_pickle=False).astype(bool, copy=False)
        group_fc = np.load(row.group_fc, allow_pickle=False).astype(np.float32, copy=False)
        prototypes = build_cluster_prototypes(group_fc, labels, support, k=K)
        prefix = _combo_prefix(output_root, row.smoothing, row.hemi)
        labels_out = prefix.with_name(prefix.name + "_labels.npy")
        support_out = prefix.with_name(prefix.name + "_support.npy")
        prototypes_out = prefix.with_name(prefix.name + "_prototypes.npy")
        _write_npy_exclusive(labels_out, labels.astype(np.int32, copy=False))
        _write_npy_exclusive(support_out, support.astype(np.uint8, copy=False))
        _write_npy_exclusive(prototypes_out, prototypes.astype(np.float32, copy=False))
        source_paths = {
            "group_manifest": row.group_manifest,
            "gradient_manifest": row.gradient_manifest,
            "labels": row.labels,
            "support": row.support,
            "group_mean_fisherz_fc_full": row.group_fc,
            "roi_registry": row.roi_registry,
            "consumed_surface": row.surface,
        }
        output_paths = {
            "labels": labels_out,
            "support": support_out,
            "prototypes": prototypes_out,
        }
        combinations.append(
            {
                "smoothing": row.smoothing,
                "hemi": row.hemi,
                "source_files": {
                    key: _fingerprint(path) for key, path in source_paths.items()
                },
                "outputs": {
                    key: {
                        **_fingerprint(path),
                        "shape": list(np.load(path, mmap_mode="r").shape),
                        "dtype": str(np.load(path, mmap_mode="r").dtype),
                    }
                    for key, path in output_paths.items()
                },
                "density_evidence": {
                    "consumed_asset": str(row.surface),
                    "n_vertices": N_VERTICES,
                    "evidence_source": "group_atlas_manifest.density_evidence",
                },
            }
        )
    registry = _fingerprint(rows[0].roi_registry)
    manifest: dict[str, object] = {
        "status": "complete",
        "workflow": "build_hcp151_fixed_template",
        "template_id": template_id,
        "signal_branch": _signal_branch_for_template(template_id),
        "signal_control_route": (
            "routea_ctxreg" if template_id == CTXREG_TEMPLATE_ID else "routea"
        ),
        "n_subjects": N_SUBJECTS,
        "density": N_VERTICES,
        "k": K,
        "n_rois": N_ROIS,
        "fc_space": "subject-mean Fisher-z vertex-to-Kong2022-ROI FC",
        "prototype_definition": (
            "finite mean of group_mean_fisherz_fc_full rows within supported "
            "AP-ordered group_labels_final cluster"
        ),
        "cluster_label_semantics": {
            "1": "Anterior/VAN",
            "2": "Body/SMN",
            "3": "Posterior/DMN",
        },
        "source_group_root": str(Path(group_root).resolve()),
        "source_gradient_root": str(Path(gradient_root).resolve()),
        "roi_registry": registry,
        "subjects": list(rows[0].subjects),
        "combinations": combinations,
    }
    _write_json_exclusive(output_root / "template_manifest.json", manifest)
    return manifest


def _validate_fingerprint(recorded: object, *, context: str) -> None:
    if not isinstance(recorded, Mapping):
        raise ValueError(f"Missing fingerprint metadata for {context}")
    raw_path = recorded.get("path")
    if not isinstance(raw_path, str):
        raise ValueError(f"Missing fingerprint path for {context}")
    actual = _fingerprint(Path(raw_path))
    if dict(recorded) != actual:
        raise ValueError(
            f"{context} fingerprint mismatch: recorded={dict(recorded)!r} actual={actual!r}"
        )


def _source_roster_from_record(
    recorded: object,
    *,
    context: str,
) -> tuple[str, ...]:
    if not isinstance(recorded, Mapping):
        raise ValueError(f"Missing source manifest fingerprint for {context}")
    _validate_fingerprint(recorded, context=context)
    path = Path(str(recorded["path"]))
    payload = _read_json(path)
    return _normalized_subjects(payload.get("subjects"), manifest_path=path)


def _validate_template_roster(
    *,
    manifest: Mapping[str, object],
    combinations: Sequence[Mapping[str, object]],
    manifest_path: Path,
) -> tuple[str, ...]:
    source_rosters: list[tuple[str, ...]] = []
    for row in combinations:
        sources = row.get("source_files")
        if not isinstance(sources, Mapping):
            raise ValueError("Template combination lacks source files for roster recovery")
        source_rosters.extend(
            [
                _source_roster_from_record(
                    sources.get("group_manifest"),
                    context="template group source manifest",
                ),
                _source_roster_from_record(
                    sources.get("gradient_manifest"),
                    context="template gradient source manifest",
                ),
            ]
        )
    expected = source_rosters[0]
    if any(roster != expected for roster in source_rosters[1:]):
        raise ValueError("Template source manifests do not share one exact 151-subject roster")
    recorded = manifest.get("subjects")
    if recorded is None:
        return expected
    normalized = _normalized_subjects(recorded, manifest_path=manifest_path)
    if normalized != expected:
        raise ValueError("Template manifest roster differs from fingerprinted source roster")
    return normalized


def validate_template(
    *,
    out_root: Path,
    template_id: str | None = None,
) -> dict[str, object]:
    output_root = Path(out_root).resolve()
    manifest_path = output_root / "template_manifest.json"
    manifest = _read_json(manifest_path)
    expected_template_id = _validate_template_id(
        manifest.get("template_id") if template_id is None else template_id
    )
    for key, expected in (
        ("status", "complete"),
        ("template_id", expected_template_id),
        ("n_subjects", N_SUBJECTS),
        ("density", N_VERTICES),
        ("k", K),
        ("n_rois", N_ROIS),
    ):
        if manifest.get(key) != expected:
            raise ValueError(
                f"Template manifest {key} mismatch: {manifest.get(key)!r} != {expected!r}"
            )
    expected_signal_branch = _signal_branch_for_template(expected_template_id)
    if manifest.get("signal_branch") is not None and manifest.get("signal_branch") != expected_signal_branch:
        raise ValueError(
            f"Template signal_branch mismatch: {manifest.get('signal_branch')!r} "
            f"!= {expected_signal_branch!r}"
        )
    expected_route = (
        "routea_ctxreg" if expected_template_id == CTXREG_TEMPLATE_ID else "routea"
    )
    if manifest.get("signal_control_route") is not None and manifest.get(
        "signal_control_route"
    ) != expected_route:
        raise ValueError(
            f"Template signal_control_route mismatch: "
            f"{manifest.get('signal_control_route')!r} != {expected_route!r}"
        )
    combinations = manifest.get("combinations")
    if not isinstance(combinations, list) or len(combinations) != 4:
        raise ValueError("Template manifest must contain exactly four combinations")
    by_key: dict[tuple[str, str], Mapping[str, object]] = {}
    for raw in combinations:
        if not isinstance(raw, Mapping):
            raise ValueError("Invalid template combination record")
        key = (str(raw.get("smoothing")), str(raw.get("hemi")))
        if key in by_key:
            raise ValueError(f"Duplicate template combination: {key}")
        by_key[key] = raw
    expected_keys = {(smoothing, hemi) for smoothing in SMOOTHINGS for hemi in HEMIS}
    if set(by_key) != expected_keys:
        raise ValueError(f"Template combination mismatch: {sorted(by_key)}")
    _validate_template_roster(
        manifest=manifest,
        combinations=list(by_key.values()),
        manifest_path=manifest_path,
    )

    expected_contracts = {
        "labels": ((N_VERTICES,), np.dtype(np.int32)),
        "support": ((N_VERTICES,), np.dtype(np.uint8)),
        "prototypes": ((K, N_ROIS), np.dtype(np.float32)),
    }
    for smoothing, hemi in sorted(expected_keys):
        row = by_key[(smoothing, hemi)]
        outputs = row.get("outputs")
        sources = row.get("source_files")
        if not isinstance(outputs, Mapping) or not isinstance(sources, Mapping):
            raise ValueError(f"Missing source/output records for {smoothing} hemi-{hemi}")
        group_manifest_path = Path(str(sources["group_manifest"]["path"])).resolve()
        gradient_manifest_path = Path(str(sources["gradient_manifest"]["path"])).resolve()
        _validate_template_source_identity(
            group_manifest=_read_json(group_manifest_path),
            gradient_manifest=_read_json(gradient_manifest_path),
            group_manifest_path=group_manifest_path,
            gradient_manifest_path=gradient_manifest_path,
            template_id=expected_template_id,
        )
        prefix = _combo_prefix(output_root, smoothing, hemi)
        for name, (shape, dtype) in expected_contracts.items():
            recorded = outputs.get(name)
            if not isinstance(recorded, Mapping):
                raise ValueError(f"Missing output fingerprint for {name}")
            expected_path = prefix.with_name(prefix.name + f"_{name}.npy")
            if Path(str(recorded.get("path"))).resolve() != expected_path:
                raise ValueError(f"Unexpected output path for {smoothing} hemi-{hemi} {name}")
            array = np.load(expected_path, allow_pickle=False)
            if list(recorded.get("shape", [])) != list(shape):
                raise ValueError(
                    f"Missing or incorrect shape metadata for {expected_path}: "
                    f"{recorded.get('shape')!r} != {list(shape)!r}"
                )
            if recorded.get("dtype") != str(dtype):
                raise ValueError(
                    f"Missing or incorrect dtype metadata for {expected_path}: "
                    f"{recorded.get('dtype')!r} != {str(dtype)!r}"
                )
            if array.shape != shape:
                raise ValueError(
                    f"Unexpected shape for {expected_path}: {array.shape} != {shape}"
                )
            if array.dtype != dtype:
                raise ValueError(
                    f"Unexpected dtype for {expected_path}: {array.dtype} != {dtype}"
                )
            fingerprint_only = {
                key: recorded[key] for key in ("path", "sha256", "size_bytes")
            }
            _validate_fingerprint(
                fingerprint_only,
                context=f"{smoothing} hemi-{hemi} output {name}",
            )
        for name, recorded in sources.items():
            _validate_fingerprint(
                recorded,
                context=f"{smoothing} hemi-{hemi} source {name}",
            )
        surface_record = sources.get("consumed_surface")
        density = row.get("density_evidence")
        if not isinstance(surface_record, Mapping) or not isinstance(density, Mapping):
            raise ValueError(
                f"Missing consumed-surface density evidence for {smoothing} hemi-{hemi}"
            )
        surface_path = Path(str(surface_record.get("path"))).resolve()
        if (
            Path(str(density.get("consumed_asset"))).resolve() != surface_path
            or int(density.get("n_vertices", -1)) != N_VERTICES
            or density.get("evidence_source")
            != "group_atlas_manifest.density_evidence"
        ):
            raise ValueError(
                f"Consumed-surface density evidence mismatch for {smoothing} hemi-{hemi}"
            )
        surface_vertices = _surface_vertex_count(surface_path)
        if surface_vertices != N_VERTICES:
            raise ValueError(
                f"Consumed surface point count is not 512: {surface_path} "
                f"({surface_vertices})"
            )
    _validate_fingerprint(manifest.get("roi_registry"), context="ROI registry")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and validate the immutable HCP151 RouteA K=3 template."
    )
    parser.add_argument("--mode", choices=("preflight", "build", "validate"), required=True)
    parser.add_argument("--group-root", type=Path, default=DEFAULT_GROUP_ROOT)
    parser.add_argument("--gradient-root", type=Path, default=DEFAULT_GRADIENT_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--template-id", choices=TEMPLATE_ID_CHOICES, default=TEMPLATE_ID)
    return parser


def _preflight_payload(
    rows: Sequence[ComboSources],
    template_id: str = TEMPLATE_ID,
) -> dict[str, object]:
    template_id = _validate_template_id(template_id)
    return {
        "status": "ok",
        "mode": "preflight",
        "template_id": template_id,
        "signal_branch": _signal_branch_for_template(template_id),
        "n_subjects": N_SUBJECTS,
        "density": N_VERTICES,
        "k": K,
        "n_rois": N_ROIS,
        "combinations": [
            {
                "smoothing": row.smoothing,
                "hemi": row.hemi,
                "labels": str(row.labels),
                "support": str(row.support),
                "group_mean_fisherz_fc_full": str(row.group_fc),
                "roi_registry": str(row.roi_registry),
                "consumed_surface": str(row.surface),
            }
            for row in rows
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.mode == "preflight":
        payload = _preflight_payload(
            preflight_sources(
                group_root=args.group_root,
                gradient_root=args.gradient_root,
                template_id=args.template_id,
            ),
            template_id=args.template_id,
        )
    elif args.mode == "build":
        payload = build_template(
            group_root=args.group_root,
            gradient_root=args.gradient_root,
            out_root=args.out_root,
            template_id=args.template_id,
        )
    else:
        payload = validate_template(
            out_root=args.out_root,
            template_id=args.template_id,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

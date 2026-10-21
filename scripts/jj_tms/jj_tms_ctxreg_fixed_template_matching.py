#!/usr/bin/env python3
"""Strict Iowa JJ_TMS CtxReg matching against the HCP151 CtxReg template."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

NATIVE_THREAD_ENV_KEYS = (
    "CORE_WORKFLOW_WB_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
)
for _thread_env_key in NATIVE_THREAD_ENV_KEYS:
    os.environ[_thread_env_key] = "1"
del _thread_env_key

import nibabel as nib
import numpy as np
from nibabel.gifti import GiftiDataArray, GiftiImage

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_WORKFLOW_DIR = REPO_ROOT / "scripts" / "core_workflow"
if str(CORE_WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_WORKFLOW_DIR))

from lib.fc_math import fisher_z_transform_fc
from lib.gifti_contracts import (
    gifti_logical_shape_dtype,
    validate_gifti_logical_contract,
    validate_gifti_pointset,
)
from lib.gifti_labels import make_named_label_gifti
from lib.output_layout import resolve_outputs_root
from lib.template_matching import match_vertices_to_prototypes

HEMIS = ("L", "R")
SMOOTHING = "2mm"
N_VERTICES = 512
N_ROIS = 400
K = 3
MAX_JOBS = 10
DEFAULT_JOBS = 4
TEMPLATE_ID = "hcp151_routea_ctxreg"
ARTIFACT_CONTRACT_SCHEMA_VERSION = 2
DEFAULT_HCP_OUTPUTS_ROOT = resolve_outputs_root("hcp")
DEFAULT_IOWA_OUTPUTS_ROOT = resolve_outputs_root("iowa_jj_tms")
DEFAULT_TEMPLATE_ROOT = DEFAULT_HCP_OUTPUTS_ROOT / "templates" / TEMPLATE_ID
DEFAULT_SHARED_FC_ROOT = (
    DEFAULT_IOWA_OUTPUTS_ROOT
    / "inputs"
    / "kong2022"
    / "shared_inputs"
    / "routea_ctxreg"
)
DEFAULT_SURFACE_INPUT_ROOT = (
    DEFAULT_IOWA_OUTPUTS_ROOT
    / "inputs"
    / "kong2022"
    / "shared_inputs"
    / "surface"
)
DEFAULT_HIPPUNFOLD_ROOT = (
    DEFAULT_IOWA_OUTPUTS_ROOT
    / "inputs"
    / "hippunfold"
    / "dense_corobl_batch"
)
DEFAULT_OUT_ROOT = (
    DEFAULT_IOWA_OUTPUTS_ROOT
    / "work"
    / "hcp151_fixed_template_matching"
    / "routea_ctxreg"
)
DEFAULT_MANIFESTS = (
    REPO_ROOT / "manifests" / "jj_tms_hc_67_subjects.txt",
    REPO_ROOT / "manifests" / "jj_tms_patient_26_subjects.txt",
)
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
LABEL_ITEMS = [
    (0, "Null", (158, 158, 158, 255)),
    (1, "Anterior/VAN", (220, 92, 255, 255)),
    (2, "Body/SMN", (99, 150, 194, 255)),
    (3, "Posterior/DMN", (216, 92, 107, 255)),
]
LABEL_SEMANTICS = {
    "1": "Anterior/VAN",
    "2": "Body/SMN",
    "3": "Posterior/DMN",
}


def fingerprint(path: Path) -> dict[str, object]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing artifact for fingerprint: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(resolved),
        "sha256": digest.hexdigest(),
        "size_bytes": int(resolved.stat().st_size),
    }


def artifact_record(path: Path) -> dict[str, object]:
    record = fingerprint(path)
    if Path(path).suffix == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        record.update(shape=list(array.shape), dtype=str(array.dtype))
    elif Path(path).suffix == ".gii":
        shape, dtype = gifti_logical_shape_dtype(path)
        record.update(shape=list(shape), dtype=str(dtype))
    return record


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _validate_fingerprint(
    recorded: object,
    *,
    context: str,
    expected_path: Path | None = None,
) -> dict[str, object]:
    if not isinstance(recorded, Mapping) or not isinstance(recorded.get("path"), str):
        raise ValueError(f"Missing fingerprint metadata for {context}")
    recorded_path = Path(str(recorded["path"])).resolve()
    if expected_path is not None and recorded_path != Path(expected_path).resolve():
        raise ValueError(
            f"{context} path mismatch: {recorded_path} != {Path(expected_path).resolve()}"
        )
    actual = artifact_record(recorded_path) if (
        "shape" in recorded or "dtype" in recorded
    ) else fingerprint(recorded_path)
    for key, value in recorded.items():
        if actual.get(key) != value:
            raise ValueError(
                f"{context} fingerprint mismatch for {key}: "
                f"recorded={value!r} actual={actual.get(key)!r}"
            )
    return dict(actual)


def _validate_artifact_record(
    recorded: object,
    *,
    expected_path: Path,
    expected_shape: tuple[int, ...],
    expected_dtype: np.dtype | type,
    context: str,
) -> dict[str, object]:
    actual = _validate_fingerprint(
        recorded,
        context=context,
        expected_path=expected_path,
    )
    expected = np.dtype(expected_dtype)
    if actual.get("shape") != list(expected_shape):
        raise ValueError(
            f"{context} shape mismatch: {actual.get('shape')} != {list(expected_shape)}"
        )
    if actual.get("dtype") != str(expected):
        raise ValueError(
            f"{context} dtype mismatch: {actual.get('dtype')} != {str(expected)}"
        )
    return actual


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_npy_exclusive(path: Path, array: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        np.save(stream, np.asarray(array), allow_pickle=False)


def _write_label_gifti_exclusive(path: Path, labels: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite GIFTI: {target}")
    image = make_named_label_gifti(np.asarray(labels, dtype=np.int32), LABEL_ITEMS)
    nib.save(image, str(target))
    validate_gifti_logical_contract(
        target,
        expected_shape=(N_VERTICES,),
        expected_dtype=np.int32,
        context=str(target),
    )


def normalize_subject(raw: object) -> str:
    subject = str(raw).strip()
    if subject.startswith("sub-"):
        subject = subject[4:]
    if not subject or not subject.isdigit():
        raise ValueError(f"Invalid JJ_TMS subject ID: {raw!r}")
    return subject


def _read_subject_manifest(path: Path) -> list[str]:
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing subject manifest: {manifest_path}")
    subjects: list[str] = []
    for line_number, raw in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        try:
            subjects.append(normalize_subject(value))
        except ValueError as exc:
            raise ValueError(
                f"Invalid subject manifest line {line_number} in {manifest_path}: {exc}"
            ) from exc
    if len(subjects) != len(set(subjects)):
        raise ValueError(f"Duplicate subjects in manifest: {manifest_path}")
    return subjects


def load_subjects(
    manifest_paths: Sequence[Path] | None = None,
    explicit_subjects: Sequence[str] | None = None,
) -> list[str]:
    if explicit_subjects is not None and manifest_paths is not None:
        raise ValueError("Use either --subjects or --manifests, not both")
    if explicit_subjects is not None:
        subjects = [normalize_subject(value) for value in explicit_subjects]
        if len(subjects) != len(set(subjects)):
            raise ValueError("Duplicate --subjects values")
        if not subjects:
            raise ValueError("At least one subject is required")
        return subjects

    using_default_roster = manifest_paths is None
    selected_paths = list(DEFAULT_MANIFESTS if manifest_paths is None else manifest_paths)
    if not selected_paths:
        raise ValueError("At least one subject manifest is required")
    subjects: list[str] = []
    seen: set[str] = set()
    for path in selected_paths:
        current = _read_subject_manifest(Path(path))
        for subject in current:
            if subject in seen:
                raise ValueError(
                    f"Subject appears in more than one selected manifest: sub-{subject}"
                )
            seen.add(subject)
            subjects.append(subject)
    if using_default_roster and [len(_read_subject_manifest(path)) for path in selected_paths] != [67, 26]:
        raise ValueError(
            "Default JJ_TMS CtxReg roster must be HC-67 plus Patient-26"
        )
    if using_default_roster and len(subjects) != 93:
        raise ValueError(
            f"Default JJ_TMS CtxReg roster must contain 93 subjects, got {len(subjects)}"
        )
    return subjects


def _normalize_template_subjects(raw: object, *, path: Path) -> tuple[str, ...]:
    if not isinstance(raw, list) or len(raw) != 151:
        raise ValueError(f"CtxReg template source roster must contain 151 subjects: {path}")
    subjects = tuple(normalize_subject(value) for value in raw)
    if len(set(subjects)) != 151:
        raise ValueError(f"Duplicate subjects in CtxReg template roster: {path}")
    return subjects


def _read_roi_registry(path: Path) -> None:
    import csv

    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        rows = list(reader)
    if "roi_id" not in list(reader.fieldnames or []):
        raise ValueError(f"ROI registry lacks roi_id: {path}")
    try:
        roi_ids = [int(row["roi_id"]) for row in rows]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"ROI registry has invalid IDs: {path}") from exc
    if roi_ids != list(range(1, N_ROIS + 1)):
        raise ValueError(
            f"ROI registry order must be exactly 1..{N_ROIS}: {path}"
        )


def _validate_ctxreg_source_manifests(
    *,
    group_path: Path,
    gradient_path: Path,
    smoothing: str,
    hemi: str,
) -> tuple[str, ...]:
    group = _read_json(group_path)
    gradient = _read_json(gradient_path)
    for key, expected in (
        ("status", "ok"),
        ("method", "gradient-spectral"),
        ("atlas", "kong2022"),
        ("smoothing", smoothing),
        ("hemi", hemi),
        ("n_subjects", 151),
    ):
        if group.get(key) != expected:
            raise ValueError(
                f"CtxReg template group source {key} mismatch in {group_path}: "
                f"{group.get(key)!r} != {expected!r}"
            )
    for key, expected in (
        ("smoothing", smoothing),
        ("hemi", hemi),
        ("n_subjects", 151),
        ("route", "routea_ctxreg"),
    ):
        if gradient.get(key) != expected:
            raise ValueError(
                f"CtxReg template gradient source {key} mismatch in {gradient_path}: "
                f"{gradient.get(key)!r} != {expected!r}"
            )
    signature = group.get("resume_input_signature_payload")
    if not isinstance(signature, Mapping) or signature.get("route_tag") != "routea_ctxreg":
        raise ValueError(
            f"CtxReg template group source lacks routea_ctxreg provenance: {group_path}"
        )
    group_subjects = _normalize_template_subjects(
        group.get("subjects"), path=group_path
    )
    gradient_subjects = _normalize_template_subjects(
        gradient.get("subjects"), path=gradient_path
    )
    if group_subjects != gradient_subjects:
        raise ValueError(
            f"CtxReg template group/gradient roster mismatch: {group_path} {gradient_path}"
        )
    return group_subjects


def load_template(template_root: Path) -> dict[str, object]:
    root = Path(template_root).resolve()
    manifest_path = root / "template_manifest.json"
    manifest = _read_json(manifest_path)
    required = (
        ("status", "complete"),
        ("template_id", TEMPLATE_ID),
        ("density", N_VERTICES),
        ("k", K),
        ("n_rois", N_ROIS),
    )
    for key, expected in required:
        if manifest.get(key) != expected:
            raise ValueError(
                f"CtxReg template {key} mismatch in {manifest_path}: "
                f"{manifest.get(key)!r} != {expected!r}"
            )
    if manifest.get("signal_branch") not in (None, "ctxreg"):
        raise ValueError(
            f"CtxReg template signal_branch is incompatible: "
            f"{manifest.get('signal_branch')!r}"
        )
    if manifest.get("signal_control_route") not in (None, "routea_ctxreg"):
        raise ValueError(
            f"CtxReg template route is incompatible: "
            f"{manifest.get('signal_control_route')!r}"
        )
    template_subjects = _normalize_template_subjects(
        manifest.get("subjects"), path=manifest_path
    )

    registry_record = manifest.get("roi_registry")
    registry_path = Path(
        str(registry_record.get("path"))
    ).resolve() if isinstance(registry_record, Mapping) else None
    if registry_path is None:
        raise ValueError(f"CtxReg template lacks roi_registry: {manifest_path}")
    _validate_fingerprint(
        registry_record,
        context="CtxReg template ROI registry",
        expected_path=registry_path,
    )
    _read_roi_registry(registry_path)

    combinations = manifest.get("combinations")
    if not isinstance(combinations, list) or len(combinations) != 4:
        raise ValueError(
            f"CtxReg template must contain exactly four combinations: {manifest_path}"
        )
    expected_keys = {(smoothing, hemi) for smoothing in ("2mm", "4mm") for hemi in HEMIS}
    rows: dict[tuple[str, str], Mapping[str, object]] = {}
    for raw in combinations:
        if not isinstance(raw, Mapping):
            raise ValueError(f"Invalid CtxReg template combination: {manifest_path}")
        key = (str(raw.get("smoothing")), str(raw.get("hemi")))
        if key in rows:
            raise ValueError(f"Duplicate CtxReg template combination: {key}")
        rows[key] = raw
    if set(rows) != expected_keys:
        raise ValueError(
            f"CtxReg template combinations mismatch: {sorted(rows)}"
        )

    prototypes: dict[str, Path] = {}
    prototype_records: dict[str, dict[str, object]] = {}
    for smoothing, hemi in sorted(expected_keys):
        row = rows[(smoothing, hemi)]
        sources = row.get("source_files")
        outputs = row.get("outputs")
        if not isinstance(sources, Mapping) or not isinstance(outputs, Mapping):
            raise ValueError(f"Missing source/output records for {smoothing} hemi-{hemi}")
        expected_source_keys = {
            "group_manifest",
            "gradient_manifest",
            "labels",
            "support",
            "group_mean_fisherz_fc_full",
            "roi_registry",
            "consumed_surface",
        }
        if set(sources) != expected_source_keys:
            raise ValueError(
                f"CtxReg template source record mismatch for {smoothing} hemi-{hemi}"
            )
        for name, record in sources.items():
            _validate_fingerprint(
                record,
                context=f"CtxReg template source {smoothing} hemi-{hemi} {name}",
            )
        group_path = Path(str(sources["group_manifest"]["path"])).resolve()
        gradient_path = Path(str(sources["gradient_manifest"]["path"])).resolve()
        source_subjects = _validate_ctxreg_source_manifests(
            group_path=group_path,
            gradient_path=gradient_path,
            smoothing=smoothing,
            hemi=hemi,
        )
        if source_subjects != template_subjects:
            raise ValueError(
                f"CtxReg template roster differs from source roster for {smoothing} hemi-{hemi}"
            )
        if dict(sources["roi_registry"]) != dict(registry_record):
            raise ValueError(
                f"CtxReg template ROI registry drift for {smoothing} hemi-{hemi}"
            )

        surface_path = Path(str(sources["consumed_surface"]["path"])).resolve()
        density = row.get("density_evidence")
        if (
            not isinstance(density, Mapping)
            or Path(str(density.get("consumed_asset"))).resolve() != surface_path
            or int(density.get("n_vertices", -1)) != N_VERTICES
        ):
            raise ValueError(
                f"CtxReg template density evidence mismatch for {smoothing} hemi-{hemi}"
            )
        if "_space-corobl_den-512_" not in surface_path.name:
            raise ValueError(f"CtxReg template surface is not corobl den-512: {surface_path}")
        validate_gifti_pointset(surface_path, expected_vertices=N_VERTICES)

        expected_output_names = {"labels", "support", "prototypes"}
        if set(outputs) != expected_output_names:
            raise ValueError(
                f"CtxReg template output record mismatch for {smoothing} hemi-{hemi}"
            )
        prefix = root / f"hcp151_{smoothing}_hemi-{hemi}"
        output_contracts = {
            "labels": (prefix.with_name(prefix.name + "_labels.npy"), (N_VERTICES,), np.int32),
            "support": (prefix.with_name(prefix.name + "_support.npy"), (N_VERTICES,), np.uint8),
            "prototypes": (
                prefix.with_name(prefix.name + "_prototypes.npy"),
                (K, N_ROIS),
                np.float32,
            ),
        }
        for name, (expected_path, shape, dtype) in output_contracts.items():
            _validate_artifact_record(
                outputs[name],
                expected_path=expected_path,
                expected_shape=shape,
                expected_dtype=dtype,
                context=f"CtxReg template output {smoothing} hemi-{hemi} {name}",
            )
            array = np.load(expected_path, allow_pickle=False)
            if name == "labels":
                support = np.load(output_contracts["support"][0], allow_pickle=False).astype(bool)
                if set(int(value) for value in np.unique(array[support])) != {1, 2, 3}:
                    raise ValueError(
                        f"CtxReg template labels do not contain all AP clusters: {expected_path}"
                    )
            elif name == "support":
                if not np.all(np.isin(array, (0, 1))):
                    raise ValueError(f"Invalid CtxReg template support values: {expected_path}")
            else:
                if not np.all(np.isfinite(array)):
                    raise ValueError(f"Non-finite CtxReg template prototypes: {expected_path}")
            if smoothing == SMOOTHING:
                prototypes[hemi] = expected_path
                prototype_records[hemi] = dict(outputs["prototypes"])

    if set(prototypes) != set(HEMIS):
        raise ValueError("CtxReg template lacks both 2mm prototype hemispheres")
    return {
        "root": root,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "manifest_fingerprint": fingerprint(manifest_path),
        "prototypes": prototypes,
        "prototype_records": prototype_records,
        "roi_registry": dict(registry_record),
        "roi_registry_path": registry_path,
    }


def _fc_path(shared_fc_root: Path, subject: str, hemi: str) -> Path:
    return (
        Path(shared_fc_root)
        / f"sub-{normalize_subject(subject)}"
        / "fc"
        / "kong2022"
        / SMOOTHING
        / f"hemi_{hemi}"
        / "hipp_vertex_to_roi_fc.npy"
    ).resolve()


def _mask_path(surface_root: Path, subject: str, hemi: str) -> Path:
    sub_id = f"sub-{normalize_subject(subject)}"
    return (
        Path(surface_root)
        / sub_id
        / "surface"
        / "tsnr_gate"
        / f"{sub_id}_hemi-{hemi}_valid_mask.npy"
    ).resolve()


def _surface_path(hippunfold_root: Path, subject: str, hemi: str) -> Path:
    sub_id = f"sub-{normalize_subject(subject)}"
    return (
        Path(hippunfold_root)
        / sub_id
        / "hippunfold"
        / sub_id
        / "surf"
        / (
            f"{sub_id}_hemi-{hemi}_space-T1w_den-512_"
            "label-hipp_midthickness.surf.gii"
        )
    ).resolve()


def preflight_subject(
    subject: str,
    *,
    shared_fc_root: Path,
    surface_root: Path,
    hippunfold_root: Path,
) -> dict[str, object]:
    sub_id = f"sub-{normalize_subject(subject)}"
    evidence: dict[str, object] = {"subject": sub_id, "hemis": {}}
    for hemi in HEMIS:
        fc_path = _fc_path(shared_fc_root, subject, hemi)
        mask_path = _mask_path(surface_root, subject, hemi)
        surface_path = _surface_path(hippunfold_root, subject, hemi)
        for name, path in (
            ("exact 2mm CtxReg FC", fc_path),
            ("standard validity mask", mask_path),
            ("T1w den-512 surface", surface_path),
        ):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing {name} for {sub_id} hemi-{hemi}: {path}"
                )
        fc = np.load(fc_path, mmap_mode="r", allow_pickle=False)
        if fc.shape != (N_VERTICES, N_ROIS) or fc.dtype != np.dtype(np.float32):
            raise ValueError(
                f"CtxReg FC contract mismatch for {sub_id} hemi-{hemi}: "
                f"shape={fc.shape} dtype={fc.dtype}"
            )
        mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
        if mask.shape != (N_VERTICES,) or mask.dtype != np.dtype(bool):
            raise ValueError(
                f"Standard validity-mask contract mismatch for {sub_id} hemi-{hemi}: "
                f"shape={mask.shape} dtype={mask.dtype}"
            )
        if int(np.sum(mask)) < K:
            raise ValueError(
                f"Too few valid hippocampal vertices for {sub_id} hemi-{hemi}: "
                f"{int(np.sum(mask))}"
            )
        validate_gifti_pointset(
            surface_path,
            expected_vertices=N_VERTICES,
        )
        evidence["hemis"][hemi] = {
            "fc": artifact_record(fc_path),
            "mask": artifact_record(mask_path),
            "surface": artifact_record(surface_path),
            "valid_vertex_count": int(np.sum(mask)),
        }
    return evidence


def preflight_subjects(
    subjects: Sequence[str],
    *,
    shared_fc_root: Path,
    surface_root: Path,
    hippunfold_root: Path,
) -> list[dict[str, object]]:
    if not subjects:
        raise ValueError("At least one subject is required")
    return [
        preflight_subject(
            subject,
            shared_fc_root=shared_fc_root,
            surface_root=surface_root,
            hippunfold_root=hippunfold_root,
        )
        for subject in subjects
    ]


def _input_records(evidence: Mapping[str, object]) -> dict[str, object]:
    hemis = evidence.get("hemis")
    if not isinstance(hemis, Mapping):
        raise ValueError("Subject evidence lacks hemispheric inputs")
    records: dict[str, object] = {}
    for hemi in HEMIS:
        row = hemis.get(hemi)
        if not isinstance(row, Mapping):
            raise ValueError(f"Subject evidence lacks hemi-{hemi}")
        for kind in ("fc", "mask", "surface"):
            if kind not in row:
                raise ValueError(f"Subject evidence lacks hemi-{hemi} {kind}")
            records[f"hemi-{hemi}_{kind}"] = dict(row[kind])
    return records


def _summary(
    labels: np.ndarray,
    scores: np.ndarray,
    margin: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, object]:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    margin = np.asarray(margin)
    valid = np.asarray(valid_mask, dtype=bool)
    if labels.shape != (N_VERTICES,) or labels.dtype != np.dtype(np.int32):
        raise ValueError(f"Invalid template labels output: {labels.shape} {labels.dtype}")
    if scores.shape != (N_VERTICES, K) or scores.dtype != np.dtype(np.float32):
        raise ValueError(f"Invalid template score output: {scores.shape} {scores.dtype}")
    if margin.shape != (N_VERTICES,) or margin.dtype != np.dtype(np.float32):
        raise ValueError(f"Invalid template margin output: {margin.shape} {margin.dtype}")
    if valid.shape != (N_VERTICES,):
        raise ValueError(f"Invalid validity mask shape: {valid.shape}")
    if np.any(labels[~valid] != 0):
        raise ValueError("Invalid vertices must have null template labels")
    valid_labels = labels[valid]
    if np.any(valid_labels == 0) or set(int(value) for value in np.unique(valid_labels)) != {1, 2, 3}:
        raise ValueError("Every valid subject must contain AP labels 1, 2, and 3")
    valid_scores = scores[valid]
    valid_margin = margin[valid]
    if not np.all(np.isfinite(valid_scores)) or np.any(
        (valid_scores < -1.0) | (valid_scores > 1.0)
    ):
        raise ValueError("Every valid vertex must have finite correlations in [-1, 1]")
    if not np.all(np.isfinite(valid_margin)) or np.any(
        (valid_margin < 0.0) | (valid_margin > 2.0)
    ):
        raise ValueError("Every valid vertex must have a finite margin in [0, 2]")
    return {
        "valid_vertices": int(np.sum(valid)),
        "label_counts": {
            str(code): int(np.sum(valid_labels == code)) for code in range(1, K + 1)
        },
        "margin_summary": {
            "min": float(np.min(valid_margin)),
            "median": float(np.median(valid_margin)),
            "mean": float(np.mean(valid_margin)),
            "max": float(np.max(valid_margin)),
        },
    }


def _expected_output_paths(out_root: Path, subject: str) -> dict[str, Path]:
    sub_id = f"sub-{normalize_subject(subject)}"
    subject_dir = Path(out_root).resolve() / "subjects" / sub_id
    paths: dict[str, Path] = {}
    for hemi in HEMIS:
        combo_dir = subject_dir / "matching" / SMOOTHING / f"hemi-{hemi}"
        prefix = f"{SMOOTHING}_hemi-{hemi}"
        paths[f"{prefix}_pearson_fc"] = combo_dir / "vertex_to_roi_pearson_fc.npy"
        paths[f"{prefix}_fisherz_fc"] = combo_dir / "vertex_to_roi_fisherz_fc.npy"
        paths[f"{prefix}_labels"] = combo_dir / "labels.npy"
        paths[f"{prefix}_scores"] = combo_dir / "template_correlations.npy"
        paths[f"{prefix}_margin"] = combo_dir / "correlation_margin.npy"
        paths[f"{prefix}_label_gifti"] = combo_dir / (
            f"{sub_id}_hemi-{hemi}_space-corobl_den-512_"
            f"label-hcp151_fixed_template_k3_{SMOOTHING}.label.gii"
        )
        paths[f"{prefix}_combo_manifest"] = combo_dir / "combo_manifest.json"
    paths["subject_manifest"] = subject_dir / "subject_manifest.json"
    return {key: path.resolve() for key, path in paths.items()}


def _validate_subject_manifest(
    *,
    subject: str,
    template_info: Mapping[str, object],
    evidence: Mapping[str, object],
    out_root: Path,
) -> dict[str, object]:
    paths = _expected_output_paths(out_root, subject)
    manifest_path = paths["subject_manifest"]
    payload = _read_json(manifest_path)
    sub_id = f"sub-{normalize_subject(subject)}"
    for key, expected in (
        ("status", "ok"),
        ("workflow", "jj_tms_ctxreg_fixed_template_matching"),
        ("artifact_contract_schema_version", ARTIFACT_CONTRACT_SCHEMA_VERSION),
        ("subject", sub_id),
        ("signal_branch", "ctxreg"),
        ("template_id", TEMPLATE_ID),
        ("smoothing", SMOOTHING),
    ):
        if payload.get(key) != expected:
            raise ValueError(
                f"CtxReg subject manifest {key} mismatch for {manifest_path}: "
                f"{payload.get(key)!r} != {expected!r}"
            )
    if payload.get("template_manifest") != template_info["manifest_fingerprint"]:
        raise ValueError(f"CtxReg subject template manifest drift: {manifest_path}")
    if payload.get("roi_registry") != template_info["roi_registry"]:
        raise ValueError(f"CtxReg subject ROI registry drift: {manifest_path}")
    expected_inputs = _input_records(evidence)
    if payload.get("inputs") != expected_inputs:
        raise ValueError(f"CtxReg subject input fingerprint drift: {manifest_path}")
    density_by_hemi = {
        hemi: {
            "consumed_asset": evidence["hemis"][hemi]["surface"]["path"],
            "evidence_source": "consumed_surface_path_and_gifti_point_count",
            "n_vertices": N_VERTICES,
        }
        for hemi in HEMIS
    }
    if payload.get("density_evidence") != density_by_hemi:
        raise ValueError(f"CtxReg subject density evidence drift: {manifest_path}")
    combinations = payload.get("combinations")
    if not isinstance(combinations, list) or len(combinations) != 2:
        raise ValueError(f"CtxReg subject must contain exactly two combinations: {manifest_path}")
    by_hemi: dict[str, Mapping[str, object]] = {}
    for row in combinations:
        if not isinstance(row, Mapping) or row.get("hemi") not in HEMIS:
            raise ValueError(f"Invalid CtxReg subject combination: {manifest_path}")
        hemi = str(row["hemi"])
        if hemi in by_hemi:
            raise ValueError(f"Duplicate CtxReg subject combination: {hemi}")
        by_hemi[hemi] = row
    if set(by_hemi) != set(HEMIS):
        raise ValueError(f"CtxReg subject combination set mismatch: {manifest_path}")

    expected_files = {manifest_path.resolve()}
    for hemi in HEMIS:
        prefix = f"{SMOOTHING}_hemi-{hemi}"
        row = by_hemi[hemi]
        combo_path = paths[f"{prefix}_combo_manifest"]
        combo_record = row.get("manifest")
        _validate_fingerprint(
            combo_record,
            context=f"CtxReg combo manifest hemi-{hemi}",
            expected_path=combo_path,
        )
        combo = _read_json(combo_path)
        for key, expected in (
            ("status", "ok"),
            ("workflow", "jj_tms_ctxreg_fixed_template_matching"),
            ("artifact_contract_schema_version", ARTIFACT_CONTRACT_SCHEMA_VERSION),
            ("subject", sub_id),
            ("signal_branch", "ctxreg"),
            ("template_id", TEMPLATE_ID),
            ("smoothing", SMOOTHING),
            ("hemi", hemi),
        ):
            if combo.get(key) != expected:
                raise ValueError(f"CtxReg combo identity mismatch: {combo_path} {key}")
        if combo.get("inputs") != evidence["hemis"][hemi]:
            raise ValueError(f"CtxReg combo input drift: {combo_path}")
        if combo.get("template_prototype") != template_info["prototype_records"][hemi]:
            raise ValueError(f"CtxReg combo template drift: {combo_path}")
        if combo.get("roi_registry") != template_info["roi_registry"]:
            raise ValueError(f"CtxReg combo registry drift: {combo_path}")
        density = combo.get("density_evidence")
        expected_density = density_by_hemi[hemi]
        if density != expected_density or row.get("density_evidence") != expected_density:
            raise ValueError(f"CtxReg combo density evidence drift: {combo_path}")
        if combo.get("label_semantics") != LABEL_SEMANTICS:
            raise ValueError(f"CtxReg combo label semantics drift: {combo_path}")
        if row.get("label_semantics") != LABEL_SEMANTICS:
            raise ValueError(f"CtxReg subject label semantics drift: {combo_path}")
        outputs = combo.get("outputs")
        if not isinstance(outputs, Mapping):
            raise ValueError(f"CtxReg combo outputs are missing: {combo_path}")
        expected_output_keys = {
            "pearson_fc",
            "fisherz_fc",
            "labels",
            "scores",
            "margin",
            "label_gifti",
        }
        if set(outputs) != expected_output_keys or row.get("outputs") != outputs:
            raise ValueError(f"CtxReg combo output set drift: {combo_path}")
        output_contracts = {
            "pearson_fc": ((N_VERTICES, N_ROIS), np.float32),
            "fisherz_fc": ((N_VERTICES, N_ROIS), np.float32),
            "labels": ((N_VERTICES,), np.int32),
            "scores": ((N_VERTICES, K), np.float32),
            "margin": ((N_VERTICES,), np.float32),
            "label_gifti": ((N_VERTICES,), np.int32),
        }
        for name, (shape, dtype) in output_contracts.items():
            output_path = paths[f"{prefix}_{name}"]
            _validate_artifact_record(
                outputs[name],
                expected_path=output_path,
                expected_shape=shape,
                expected_dtype=dtype,
                context=f"CtxReg combo output {prefix}_{name}",
            )
            expected_files.add(output_path)
        labels = np.load(paths[f"{prefix}_labels"], allow_pickle=False)
        scores = np.load(paths[f"{prefix}_scores"], allow_pickle=False)
        margin = np.load(paths[f"{prefix}_margin"], allow_pickle=False)
        valid = np.load(
            Path(str(evidence["hemis"][hemi]["mask"]["path"])),
            allow_pickle=False,
        )
        summary = _summary(labels, scores, margin, valid)
        if combo.get("summary") != summary or row.get("summary") != summary:
            raise ValueError(f"CtxReg combo summary drift: {combo_path}")
        for metadata in (combo, row):
            if (
                metadata.get("valid_vertices") != summary["valid_vertices"]
                or metadata.get("label_counts") != summary["label_counts"]
                or metadata.get("margin_summary") != summary["margin_summary"]
            ):
                raise ValueError(f"CtxReg combo summary fields drift: {combo_path}")
        expected_files.add(combo_path)
    for path in manifest_path.parent.rglob("*"):
        if path.is_file() and path.resolve() not in expected_files:
            raise ValueError(f"Unexpected retained CtxReg subject artifact: {path}")
    return payload


def process_subject(
    subject: str,
    *,
    template_info: Mapping[str, object],
    shared_fc_root: Path,
    surface_root: Path,
    hippunfold_root: Path,
    out_root: Path,
    evidence: Mapping[str, object] | None = None,
    resume: bool = False,
) -> dict[str, object]:
    sub_id = f"sub-{normalize_subject(subject)}"
    output_root = Path(out_root).resolve()
    subject_dir = output_root / "subjects" / sub_id
    subject_manifest_path = subject_dir / "subject_manifest.json"

    if subject_manifest_path.exists():
        if not resume:
            raise FileExistsError(
                f"CtxReg subject output exists; use --resume for exact validation: "
                f"{subject_manifest_path}"
            )
        current_evidence = preflight_subject(
            subject,
            shared_fc_root=shared_fc_root,
            surface_root=surface_root,
            hippunfold_root=hippunfold_root,
        )
        return _validate_subject_manifest(
            subject=subject,
            template_info=template_info,
            evidence=current_evidence,
            out_root=output_root,
        )
    if subject_dir.exists() and any(subject_dir.iterdir()):
        raise FileExistsError(
            f"CtxReg subject output is partial/nonempty and cannot be overwritten: "
            f"{subject_dir}"
        )
    if evidence is None:
        evidence = preflight_subject(
            subject,
            shared_fc_root=shared_fc_root,
            surface_root=surface_root,
            hippunfold_root=hippunfold_root,
        )
    subject_dir.mkdir(parents=True, exist_ok=False)
    combinations: list[dict[str, object]] = []
    density_evidence: dict[str, object] = {}
    input_records = _input_records(evidence)

    for hemi in HEMIS:
        hemi_evidence = evidence["hemis"][hemi]
        fc_path = Path(str(hemi_evidence["fc"]["path"]))
        mask_path = Path(str(hemi_evidence["mask"]["path"]))
        surface_path = Path(str(hemi_evidence["surface"]["path"]))
        pearson = np.load(fc_path, allow_pickle=False).astype(np.float32, copy=True)
        valid_mask = np.load(mask_path, allow_pickle=False).astype(bool, copy=False)
        prototypes = np.load(
            template_info["prototypes"][hemi],
            allow_pickle=False,
        ).astype(np.float32, copy=False)
        pearson[~valid_mask, :] = np.nan
        fisherz = np.asarray(fisher_z_transform_fc(pearson), dtype=np.float32)
        labels, scores, margin = match_vertices_to_prototypes(
            fisherz,
            valid_mask,
            prototypes,
        )
        labels = np.asarray(labels, dtype=np.int32)
        scores = np.asarray(scores, dtype=np.float32)
        margin = np.asarray(margin, dtype=np.float32)
        summary = _summary(labels, scores, margin, valid_mask)
        missing_roi_count = int(np.sum(np.all(np.isnan(pearson), axis=0)))

        prefix = f"{SMOOTHING}_hemi-{hemi}"
        paths = _expected_output_paths(output_root, subject)
        output_paths = {
            "pearson_fc": paths[f"{prefix}_pearson_fc"],
            "fisherz_fc": paths[f"{prefix}_fisherz_fc"],
            "labels": paths[f"{prefix}_labels"],
            "scores": paths[f"{prefix}_scores"],
            "margin": paths[f"{prefix}_margin"],
            "label_gifti": paths[f"{prefix}_label_gifti"],
        }
        _write_npy_exclusive(output_paths["pearson_fc"], pearson)
        _write_npy_exclusive(output_paths["fisherz_fc"], fisherz)
        _write_npy_exclusive(output_paths["labels"], labels)
        _write_npy_exclusive(output_paths["scores"], scores)
        _write_npy_exclusive(output_paths["margin"], margin)
        _write_label_gifti_exclusive(output_paths["label_gifti"], labels)
        output_records = {
            name: artifact_record(path) for name, path in output_paths.items()
        }
        density = {
            "consumed_asset": str(surface_path),
            "evidence_source": "consumed_surface_path_and_gifti_point_count",
            "n_vertices": N_VERTICES,
        }
        density_evidence[hemi] = density
        combo_path = paths[f"{prefix}_combo_manifest"]
        combo_manifest = {
            "artifact_contract_schema_version": ARTIFACT_CONTRACT_SCHEMA_VERSION,
            "status": "ok",
            "workflow": "jj_tms_ctxreg_fixed_template_matching",
            "subject": sub_id,
            "signal_branch": "ctxreg",
            "template_id": TEMPLATE_ID,
            "smoothing": SMOOTHING,
            "hemi": hemi,
            "label_semantics": dict(LABEL_SEMANTICS),
            "inputs": dict(hemi_evidence),
            "template_prototype": dict(template_info["prototype_records"][hemi]),
            "roi_registry": dict(template_info["roi_registry"]),
            "density_evidence": density,
            "summary": summary,
            "valid_vertices": summary["valid_vertices"],
            "label_counts": summary["label_counts"],
            "margin_summary": summary["margin_summary"],
            "missing_roi_count": missing_roi_count,
            "outputs": output_records,
        }
        _write_json_exclusive(combo_path, combo_manifest)
        combinations.append(
            {
                "status": "ok",
                "smoothing": SMOOTHING,
                "hemi": hemi,
                "label_semantics": dict(LABEL_SEMANTICS),
                "manifest": fingerprint(combo_path),
                "density_evidence": density,
                "summary": summary,
                "valid_vertices": summary["valid_vertices"],
                "label_counts": summary["label_counts"],
                "margin_summary": summary["margin_summary"],
                "outputs": output_records,
            }
        )

    subject_manifest = {
        "artifact_contract_schema_version": ARTIFACT_CONTRACT_SCHEMA_VERSION,
        "status": "ok",
        "workflow": "jj_tms_ctxreg_fixed_template_matching",
        "subject": sub_id,
        "signal_branch": "ctxreg",
        "template_id": TEMPLATE_ID,
        "smoothing": SMOOTHING,
        "template_manifest": dict(template_info["manifest_fingerprint"]),
        "roi_registry": dict(template_info["roi_registry"]),
        "inputs": input_records,
        "density_evidence": density_evidence,
        "combinations": combinations,
    }
    _write_json_exclusive(subject_manifest_path, subject_manifest)
    return subject_manifest


def _validate_run_id(raw: str | None) -> str:
    if raw is None or not raw.strip() or RUN_ID_RE.fullmatch(raw.strip()) is None:
        raise ValueError(
            "--run-id must be a nonempty path-safe identifier containing only "
            "letters, numbers, '.', '_' or '-'"
        )
    return raw.strip()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run strict Iowa JJ_TMS CtxReg HCP151 fixed-template matching."
    )
    parser.add_argument("--mode", choices=("preflight", "run", "validate"), required=True)
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--manifests", nargs="+", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--template-root", type=Path, default=DEFAULT_TEMPLATE_ROOT)
    parser.add_argument("--shared-fc-root", type=Path, default=DEFAULT_SHARED_FC_ROOT)
    parser.add_argument("--surface-root", type=Path, default=DEFAULT_SURFACE_INPUT_ROOT)
    parser.add_argument("--hippunfold-root", type=Path, default=DEFAULT_HIPPUNFOLD_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not 1 <= int(args.jobs) <= MAX_JOBS:
        raise ValueError(f"--jobs must be in [1, {MAX_JOBS}]")
    subjects = load_subjects(
        manifest_paths=args.manifests,
        explicit_subjects=args.subjects,
    )
    template_info = load_template(args.template_root)
    evidence = preflight_subjects(
        subjects,
        shared_fc_root=args.shared_fc_root,
        surface_root=args.surface_root,
        hippunfold_root=args.hippunfold_root,
    )

    if args.mode == "preflight":
        print(
            json.dumps(
                {
                    "status": "ok",
                    "mode": "preflight",
                    "subject_count": len(subjects),
                    "subjects": [f"sub-{subject}" for subject in subjects],
                    "template_id": TEMPLATE_ID,
                    "signal_branch": "ctxreg",
                    "smoothing": SMOOTHING,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    output_root = Path(args.out_root).resolve()
    if args.mode == "validate":
        for subject, subject_evidence in zip(subjects, evidence, strict=True):
            _validate_subject_manifest(
                subject=subject,
                template_info=template_info,
                evidence=subject_evidence,
                out_root=output_root,
            )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "mode": "validate",
                    "subject_count": len(subjects),
                    "template_id": TEMPLATE_ID,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    run_id = _validate_run_id(args.run_id) if args.run_id else (
        "ctxreg_matching_"
        + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    )
    run_dir = output_root / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(
            f"CtxReg run directory already exists; choose a new run-id: {run_dir}"
        )
    evidence_by_subject = {
        subject: row for subject, row in zip(subjects, evidence, strict=True)
    }
    results: dict[str, dict[str, object]] = {}
    print(
        f"Running CtxReg fixed-template matching for {len(subjects)} subjects "
        f"(jobs={args.jobs}, run_id={run_id})..."
    )
    with ThreadPoolExecutor(max_workers=int(args.jobs)) as pool:
        futures = {
            pool.submit(
                process_subject,
                subject,
                template_info=template_info,
                shared_fc_root=args.shared_fc_root,
                surface_root=args.surface_root,
                hippunfold_root=args.hippunfold_root,
                out_root=output_root,
                evidence=evidence_by_subject[subject],
                resume=bool(args.resume),
            ): subject
            for subject in subjects
        }
        for future in as_completed(futures):
            subject = futures[future]
            results[subject] = future.result()
            print(f"  [DONE] sub-{subject}")

    run_dir.mkdir(parents=True, exist_ok=False)
    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest = {
        "artifact_contract_schema_version": ARTIFACT_CONTRACT_SCHEMA_VERSION,
        "status": "complete",
        "workflow": "jj_tms_ctxreg_fixed_template_matching",
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "signal_branch": "ctxreg",
        "template_id": TEMPLATE_ID,
        "smoothing": SMOOTHING,
        "subject_count": len(subjects),
        "subjects": [f"sub-{subject}" for subject in subjects],
        "template_manifest": dict(template_info["manifest_fingerprint"]),
        "roi_registry": dict(template_info["roi_registry"]),
        "path_roots": {
            "template_root": str(template_info["root"]),
            "shared_fc_root": str(Path(args.shared_fc_root).resolve()),
            "surface_root": str(Path(args.surface_root).resolve()),
            "hippunfold_root": str(Path(args.hippunfold_root).resolve()),
            "out_root": str(output_root),
        },
        "subject_manifests": [
            fingerprint(
                output_root
                / "subjects"
                / f"sub-{subject}"
                / "subject_manifest.json"
            )
            for subject in subjects
        ],
    }
    _write_json_exclusive(run_manifest_path, run_manifest)
    print(f"Run complete. Manifest: {run_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

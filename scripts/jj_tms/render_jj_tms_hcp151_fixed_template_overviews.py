#!/usr/bin/env python3
"""Render fixed-template JJ_TMS subject overviews with the shared Workbench layout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

NATIVE_THREAD_ENV_OVERRIDES = {
    "CORE_WORKFLOW_WB_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": "1",
}
for _thread_env_key, _thread_env_value in NATIVE_THREAD_ENV_OVERRIDES.items():
    os.environ[_thread_env_key] = _thread_env_value
del _thread_env_key, _thread_env_value

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_WORKFLOW_DIR = REPO_ROOT / "scripts" / "core_workflow"
if str(CORE_WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_WORKFLOW_DIR))

from lib.gifti_contracts import (  # noqa: E402
    gifti_logical_shape_dtype,
    validate_gifti_pointset,
    validate_gifti_vector,
)
from lib.network_colors import require_network_colors  # noqa: E402
from lib.surface_utils import infer_spec_path, render_locked_grid  # noqa: E402

PRODUCTION_ROOT = Path("/home/yojiao/hpchome/hipp")
DEFAULT_WB_COMMAND = (
    PRODUCTION_ROOT / "software" / "workbench" / "bin_linux64" / "wb_command"
)
if DEFAULT_WB_COMMAND.is_file():
    os.environ.setdefault("CORE_WORKFLOW_WB_COMMAND", str(DEFAULT_WB_COMMAND))
DEFAULT_MATCHING_ROOT = (
    PRODUCTION_ROOT / "outputs_iowa_jj_tms" / "work" / "hcp151_fixed_template_matching"
)
DEFAULT_OUTPUT_DIR = (
    PRODUCTION_ROOT
    / "outputs_iowa_jj_tms"
    / "present"
    / "overviews"
    / "hcp151_fixed_template_matching"
)
DEFAULT_SUBJECTS_MANIFEST = (
    REPO_ROOT / "manifests" / "jj_tms_hcp151_fixed_template_subjects.txt"
)
DEFAULT_SCENE = (
    PRODUCTION_ROOT / "config" / "wb_locked_native_view_lateral_medial.scene"
)
DEFAULT_RUN_ID = "jj_tms_hcp151_full_20260809"
DEFAULT_JOBS = 2
MAX_JOBS = 5
SMOOTHINGS = ("2mm", "4mm")
HEMIS = ("L", "R")
N_VERTICES = 512
EXPECTED_SUBJECT_COUNT = 94
WORKFLOW_NAME = "jj_tms_hcp151_fixed_template_matching_overviews"
STATUS_FIELDS = (
    "status",
    "subject",
    "output",
    "resumed",
    "elapsed_seconds",
    "error_type",
    "error",
)
LABEL_SEMANTICS = {
    "1": "Anterior/VAN",
    "2": "Body/SMN",
    "3": "Posterior/DMN",
}
NETWORK_DISPLAY = {
    "1": ("VentralAttention", "VAN"),
    "2": ("Somatomotor", "SMN"),
    "3": ("Default", "DMN"),
}
THREAD_ENV_OVERRIDES = dict(NATIVE_THREAD_ENV_OVERRIDES)

# These helpers are the canonical cross-fit presentation implementation.  The
# fixed-template adapter supplies a different, strictly validated source
# manifest while retaining the same label-table, locked-grid, and composition
# behavior.
_CROSSFIT_PATH = (
    REPO_ROOT
    / "scripts"
    / "individual_workflow"
    / "04_01_render_hcp_crossfit_individual_overviews.py"
)
_CROSSFIT_SPEC = importlib.util.spec_from_file_location(
    "_fixed_template_shared_overview_renderer", _CROSSFIT_PATH
)
if _CROSSFIT_SPEC is None or _CROSSFIT_SPEC.loader is None:
    raise ImportError(f"Unable to load shared overview renderer: {_CROSSFIT_PATH}")
_CROSSFIT = importlib.util.module_from_spec(_CROSSFIT_SPEC)
sys.modules[_CROSSFIT_SPEC.name] = _CROSSFIT
_CROSSFIT_SPEC.loader.exec_module(_CROSSFIT)
compose_stacked_rows = _CROSSFIT.compose_stacked_rows
write_network_display_label_gifti = _CROSSFIT.write_network_display_label_gifti

_MATCHER_PATH = (
    REPO_ROOT / "scripts" / "jj_tms" / "jj_tms_hcp151_fixed_template_matching.py"
)
_MATCHER_MODULE: Any | None = None


def _load_matcher_module() -> Any:
    global _MATCHER_MODULE
    if _MATCHER_MODULE is None:
        spec = importlib.util.spec_from_file_location(
            "_fixed_template_matcher_for_overview_validation",
            _MATCHER_PATH,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load matcher validator: {_MATCHER_PATH}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _MATCHER_MODULE = module
    return _MATCHER_MODULE


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)


def fingerprint(path: Path) -> dict[str, object]:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing fingerprint input: {resolved}")
    digest = hashlib.sha256()
    size = 0
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "size_bytes": int(size)}


def _validate_fingerprint(
    recorded: object,
    path: Path,
    *,
    context: str,
) -> dict[str, object]:
    actual = fingerprint(path)
    if not isinstance(recorded, Mapping) or dict(recorded) != actual:
        raise ValueError(
            f"{context} fingerprint mismatch: recorded={recorded!r} actual={actual!r}"
        )
    return actual


def normalize_subject(value: str) -> str:
    subject = str(value).strip().removeprefix("sub-")
    if not subject or not subject.isdigit():
        raise ValueError(f"Invalid numeric subject: {value!r}")
    return subject


def normalize_subjects(values: Sequence[str]) -> list[str]:
    normalized = [normalize_subject(value) for value in values]
    duplicates = sorted({value for value in normalized if normalized.count(value) > 1})
    if duplicates:
        raise ValueError(
            "Requested subjects contain duplicate IDs: "
            + ", ".join(f"sub-{value}" for value in duplicates)
        )
    return normalized


def load_subject_roster(path: Path) -> list[str]:
    values = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    subjects = normalize_subjects(values)
    if len(subjects) != EXPECTED_SUBJECT_COUNT:
        raise ValueError(
            f"Expected the fixed {EXPECTED_SUBJECT_COUNT}-subject roster, got {len(subjects)}"
        )
    return subjects


def _source_output_path(recorded: object, *, context: str) -> Path:
    if not isinstance(recorded, Mapping) or not isinstance(recorded.get("path"), str):
        raise ValueError(f"{context} path metadata is missing")
    path = Path(str(recorded["path"])).resolve()
    actual = fingerprint(path)
    expected = {
        key: recorded.get(key) for key in ("sha256", "size_bytes") if key in recorded
    }
    if expected != actual:
        raise ValueError(
            f"{context} fingerprint mismatch: recorded={expected!r} actual={actual!r}"
        )
    if path.suffix == ".gii" and ("shape" in recorded or "dtype" in recorded):
        shape, dtype = gifti_logical_shape_dtype(path)
        if recorded.get("shape") != list(shape) or recorded.get("dtype") != str(dtype):
            raise ValueError(f"{context} logical GIFTI metadata drift")
    return path


def _validate_labels(path: Path, *, context: str) -> np.ndarray:
    labels = validate_gifti_vector(
        path,
        expected_length=N_VERTICES,
        expected_dtype=np.int32,
        context=context,
    )
    labels = np.asarray(labels, dtype=np.int32)
    if np.any((labels < 0) | (labels > 3)):
        raise ValueError(f"{context} contains labels outside [0, 3]")
    if any(not np.any(labels == code) for code in (1, 2, 3)):
        raise ValueError(f"{context} must contain all three non-null labels")
    return labels


def _validate_surface(
    *,
    subject: str,
    hemi: str,
    subject_manifest: Mapping[str, object],
) -> dict[str, object]:
    inputs = subject_manifest.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError(f"Subject manifest has no inputs map: sub-{subject}")
    recorded = inputs.get(f"surface_hemi-{hemi}")
    surface = _source_output_path(
        recorded,
        context=f"sub-{subject} {hemi} subject surface",
    )
    expected_name = (
        f"sub-{subject}_hemi-{hemi}_space-T1w_den-512_"
        "label-hipp_midthickness.surf.gii"
    )
    if surface.name != expected_name:
        raise ValueError(f"Consumed surface identity mismatch: {surface}")
    points = validate_gifti_pointset(
        surface,
        expected_vertices=N_VERTICES,
        expected_dtype=np.float32,
    )
    spec = infer_spec_path(subject, surface).resolve()
    if not spec.is_file():
        raise FileNotFoundError(f"Missing subject surface spec: {spec}")
    return {
        "path": str(surface),
        "fingerprint": fingerprint(surface),
        "point_shape": [int(value) for value in points.shape],
        "point_dtype": str(points.dtype),
        "spec_path": str(spec),
        "spec_fingerprint": fingerprint(spec),
    }


def _combo_rows(
    *,
    subject: str,
    subject_dir: Path,
    subject_manifest: Mapping[str, object],
) -> dict[tuple[str, str], Mapping[str, object]]:
    combinations = subject_manifest.get("combinations")
    if not isinstance(combinations, list):
        raise ValueError(f"Subject manifest combinations are missing: sub-{subject}")
    rows: dict[tuple[str, str], Mapping[str, object]] = {}
    for row in combinations:
        if not isinstance(row, Mapping):
            raise ValueError(f"Subject combination row is invalid: sub-{subject}")
        key = (str(row.get("smoothing")), str(row.get("hemi")))
        if key in rows:
            raise ValueError(f"Duplicate subject combination: sub-{subject} {key}")
        rows[key] = row
    expected = {(smoothing, hemi) for smoothing in SMOOTHINGS for hemi in HEMIS}
    if set(rows) != expected:
        raise ValueError(f"Subject combination set mismatch: sub-{subject}")
    for smoothing, hemi in sorted(expected):
        manifest_raw = rows[(smoothing, hemi)].get("manifest")
        if not isinstance(manifest_raw, str):
            raise ValueError(
                f"Subject combo manifest path is missing: sub-{subject} {smoothing} {hemi}"
            )
        manifest_path = Path(manifest_raw).resolve()
        expected_path = (
            subject_dir
            / "matching"
            / smoothing
            / f"hemi-{hemi}"
            / "combo_manifest.json"
        ).resolve()
        if manifest_path != expected_path:
            raise ValueError(f"Subject combo manifest path mismatch: {manifest_path}")
        combo = _load_json(manifest_path)
        if (
            combo.get("status") != "ok"
            or combo.get("subject") != f"sub-{subject}"
            or combo.get("smoothing") != smoothing
            or combo.get("hemi") != hemi
            or combo.get("label_semantics") != LABEL_SEMANTICS
        ):
            raise ValueError(
                f"Subject combo manifest identity mismatch: {manifest_path}"
            )
        outputs = combo.get("outputs")
        if not isinstance(outputs, Mapping):
            raise ValueError(f"Subject combo outputs are missing: {manifest_path}")
        subject_outputs = subject_manifest.get("outputs")
        if not isinstance(subject_outputs, Mapping):
            raise ValueError(f"Subject manifest outputs are missing: sub-{subject}")
        prefix = f"{smoothing}_hemi-{hemi}"
        expected_combo_record = subject_outputs.get(f"{prefix}_combo_manifest")
        actual_combo_record = {
            "path": str(manifest_path),
            **fingerprint(manifest_path),
        }
        if expected_combo_record != actual_combo_record:
            raise ValueError(
                f"Subject combo provenance drift: sub-{subject} {smoothing} {hemi}"
            )
        for output_name, output_record in outputs.items():
            expected_record = subject_outputs.get(f"{prefix}_{output_name}")
            if expected_record != output_record:
                raise ValueError(
                    f"Subject output provenance drift: "
                    f"sub-{subject} {smoothing} {hemi} {output_name}"
                )
        label_path = _source_output_path(
            outputs.get("label_gifti"),
            context=f"sub-{subject} {smoothing} hemi-{hemi} label GIFTI",
        )
        expected_label = (
            subject_dir
            / "matching"
            / smoothing
            / f"hemi-{hemi}"
            / (
                f"sub-{subject}_hemi-{hemi}_space-corobl_den-512_"
                f"label-hcp151_fixed_template_k3_{smoothing}.label.gii"
            )
        ).resolve()
        if label_path != expected_label:
            raise ValueError(f"Subject label GIFTI path mismatch: {label_path}")
        _validate_labels(
            label_path,
            context=f"sub-{subject} {smoothing} hemi-{hemi} label GIFTI",
        )
        density = combo.get("density_evidence")
        if not isinstance(density, Mapping):
            raise ValueError(
                f"Subject combo density evidence is missing: {manifest_path}"
            )
        surface_raw = density.get("consumed_asset")
        if not isinstance(surface_raw, str):
            raise ValueError(
                f"Subject combo consumed surface is missing: {manifest_path}"
            )
        if str(Path(surface_raw).resolve()) != str(
            Path(
                str(subject_manifest["inputs"][f"surface_hemi-{hemi}"]["path"])
            ).resolve()
        ):
            raise ValueError(f"Subject combo surface dependency drift: {manifest_path}")
        if int(density.get("n_vertices", -1)) != N_VERTICES:
            raise ValueError(
                f"Subject combo density is not 512 vertices: {manifest_path}"
            )
    return rows


def load_subject_sources(
    *,
    matching_root: Path,
    subject: str,
    matcher_context: Mapping[str, object] | None = None,
) -> dict[str, object]:
    subject_id = normalize_subject(subject)
    subject_dir = Path(matching_root).resolve() / "subjects" / f"sub-{subject_id}"
    manifest_path = subject_dir / "subject_manifest.json"
    payload = _load_json(manifest_path)
    if payload.get("status") != "ok" or payload.get("subject") != f"sub-{subject_id}":
        raise ValueError(f"Subject mapping manifest is not complete: {manifest_path}")
    if matcher_context:
        _validate_subject_with_matcher(
            matching_root=Path(matching_root),
            subject=subject_id,
            matcher_context=matcher_context,
        )
    surfaces = {
        hemi: _validate_surface(
            subject=subject_id,
            hemi=hemi,
            subject_manifest=payload,
        )
        for hemi in HEMIS
    }
    combo_rows = _combo_rows(
        subject=subject_id,
        subject_dir=subject_dir,
        subject_manifest=payload,
    )
    source_rows: list[dict[str, object]] = []
    for smoothing in SMOOTHINGS:
        row: dict[str, object] = {"smoothing": smoothing}
        for hemi in HEMIS:
            combo_row = combo_rows[(smoothing, hemi)]
            combo_path = Path(str(combo_row["manifest"])).resolve()
            combo_payload = _load_json(combo_path)
            label_record = combo_payload["outputs"]["label_gifti"]
            label_path = Path(str(label_record["path"])).resolve()
            row[hemi] = {
                "combo_manifest": str(combo_path),
                "combo_manifest_fingerprint": fingerprint(combo_path),
                "label_gifti": str(label_path),
                "label_fingerprint": fingerprint(label_path),
                "surface": dict(surfaces[hemi]),
            }
        source_rows.append(row)
    return {
        "subject": f"sub-{subject_id}",
        "subject_manifest": str(manifest_path.resolve()),
        "subject_manifest_fingerprint": fingerprint(manifest_path),
        "source_rows": source_rows,
    }


def _validate_subject_with_matcher(
    *,
    matching_root: Path,
    subject: str,
    matcher_context: Mapping[str, object],
) -> None:
    matcher = _load_matcher_module()
    roots = matcher_context.get("path_roots")
    parameters = matcher_context.get("parameters")
    if not isinstance(roots, Mapping) or not isinstance(parameters, Mapping):
        raise ValueError("Matcher validation context is incomplete")
    if Path(str(roots["out_root"])).resolve() != Path(matching_root).resolve():
        raise ValueError("Matcher and overview output roots differ")
    concat_root = roots.get("concat_dtseries_root")
    concat_dtseries_root = (
        Path(str(concat_root))
        if concat_root is not None
        and str(concat_root).strip().lower() not in ("none", "null", "")
        else None
    )
    functional = matcher.resolve_subject_runs(
        subject=subject,
        volume_root=Path(str(roots["volume_root"])),
        dtseries_root=Path(str(roots["dtseries_root"])),
        concat_dtseries_root=concat_dtseries_root,
    )
    matcher.validate_functional_headers(functional)
    matcher.load_areal_labels(
        label_root=Path(str(roots["label_root"])),
        subject=subject,
    )
    matcher._validate_surfaces(
        hippunfold_root=Path(str(roots["hippunfold_root"])),
        subject=subject,
    )
    template = matcher._load_template_contract(Path(str(roots["template_root"])))
    subject_dir = Path(matching_root).resolve() / "subjects" / f"sub-{subject}"
    expected_inputs = matcher._expected_subject_inputs(
        functional=functional,
        label_root=Path(str(roots["label_root"])),
        hippunfold_root=Path(str(roots["hippunfold_root"])),
        template=template,
    )
    expected_outputs = matcher._expected_subject_output_paths(
        functional=functional,
        subject_dir=subject_dir,
    )
    matcher.validate_reusable_subject(
        subject_dir=subject_dir,
        expected_inputs=expected_inputs,
        expected_parameters=parameters,
        expected_outputs=expected_outputs,
        functional=functional,
    )


def _validate_full_validation_manifest(
    *,
    path: Path,
    matching_root: Path,
    subjects: Sequence[str],
) -> dict[str, object]:
    payload = _load_json(path)
    if payload.get("status") != "complete":
        raise ValueError(f"Validation source is not complete: {path}")
    if (
        len(subjects) != EXPECTED_SUBJECT_COUNT
        or len(set(subjects)) != EXPECTED_SUBJECT_COUNT
    ):
        raise ValueError(
            f"Validation source must cover the full {EXPECTED_SUBJECT_COUNT}-subject roster: {path}"
        )
    expected_subjects = [f"sub-{subject}" for subject in subjects]
    recorded_subjects = payload.get("subject_manifests")
    expected_manifests = [
        str(
            (
                Path(matching_root).resolve()
                / "subjects"
                / subject
                / "subject_manifest.json"
            ).resolve()
        )
        for subject in expected_subjects
    ]
    if not isinstance(recorded_subjects, list) or set(recorded_subjects) != set(
        expected_manifests
    ):
        raise ValueError(f"Validation subject closure mismatch: {path}")
    run_record = payload.get("run_manifest")
    if not isinstance(run_record, Mapping):
        raise ValueError(f"Validation run manifest record is missing: {path}")
    run_path = _source_output_path(run_record, context="validation run manifest")
    run_payload = _load_json(run_path)
    if (
        run_payload.get("status") != "complete"
        or run_payload.get("scope") != "full"
        or run_payload.get("subject_count") != EXPECTED_SUBJECT_COUNT
        or run_payload.get("complete_count") != EXPECTED_SUBJECT_COUNT
        or run_payload.get("failed_count") != 0
        or len(run_payload.get("subjects", [])) != EXPECTED_SUBJECT_COUNT
    ):
        raise ValueError(f"Validation run dependency is not complete: {run_path}")
    if [str(value) for value in run_payload.get("subjects", [])] != expected_subjects:
        raise ValueError(f"Validation run subject roster mismatch: {run_path}")
    path_roots = run_payload.get("path_roots")
    parameters = run_payload.get("parameters")
    if not isinstance(path_roots, Mapping) or not isinstance(parameters, Mapping):
        raise ValueError(f"Validation run roots/parameters are missing: {run_path}")
    if Path(str(path_roots.get("out_root"))).resolve() != Path(matching_root).resolve():
        raise ValueError(f"Validation run output root mismatch: {run_path}")
    table_path = _source_output_path(
        payload.get("validation_items"),
        context="validation table",
    )
    expected_fields = {
        "subject",
        "smoothing",
        "hemi",
        "status",
        "label_1_count",
        "label_2_count",
        "label_3_count",
        "margin_min",
        "margin_median",
        "margin_mean",
        "margin_max",
        "labels_sha256",
        "consumed_surface",
    }
    with table_path.open("r", encoding="utf-8", newline="") as stream:
        table_rows = list(csv.DictReader(stream, delimiter="\t"))
    expected_count = len(expected_subjects) * len(SMOOTHINGS) * len(HEMIS)
    if len(table_rows) != expected_count:
        raise ValueError(
            f"Validation table must contain {expected_count} rows: {table_path}"
        )
    if any(set(row) != expected_fields for row in table_rows):
        raise ValueError(f"Validation table schema drift: {table_path}")
    items = payload.get("items")
    if not isinstance(items, list) or len(items) != expected_count:
        raise ValueError(f"Validation item closure is incomplete: {path}")
    if any(not isinstance(row, Mapping) or row.get("status") != "ok" for row in items):
        raise ValueError(f"Validation contains non-ok item: {path}")
    expected_keys = {
        (subject, smoothing, hemi)
        for subject in expected_subjects
        for smoothing in SMOOTHINGS
        for hemi in HEMIS
    }
    item_keys = {
        (str(row.get("subject")), str(row.get("smoothing")), str(row.get("hemi")))
        for row in items
    }
    table_keys = {(row["subject"], row["smoothing"], row["hemi"]) for row in table_rows}
    if len(item_keys) != len(items) or len(table_keys) != len(table_rows):
        raise ValueError(f"Validation contains duplicate subject/combo rows: {path}")
    if item_keys != expected_keys or table_keys != expected_keys:
        raise ValueError(f"Validation item key closure drift: {path}")
    label_fingerprints: dict[tuple[str, str, str], str] = {}
    consumed_surfaces: dict[tuple[str, str, str], str] = {}
    for subject in expected_subjects:
        subject_dir = Path(matching_root).resolve() / "subjects" / subject
        subject_payload = _load_json(subject_dir / "subject_manifest.json")
        combos = _combo_rows(
            subject=normalize_subject(subject),
            subject_dir=subject_dir,
            subject_manifest=subject_payload,
        )
        for (smoothing, hemi), combo_row in combos.items():
            combo_payload = _load_json(Path(str(combo_row["manifest"])))
            combo_outputs = combo_payload.get("outputs")
            if not isinstance(combo_outputs, Mapping):
                raise ValueError(
                    f"Validation combo outputs are missing: {combo_row['manifest']}"
                )
            label_path = _source_output_path(
                combo_outputs.get("label_gifti"),
                context=f"Validation source label {subject} {smoothing} {hemi}",
            )
            labels_record = combo_outputs.get(
                "labels", combo_outputs.get("label_gifti")
            )
            labels_path = _source_output_path(
                labels_record,
                context=f"Validation source labels {subject} {smoothing} {hemi}",
            )
            key = (subject, smoothing, hemi)
            label_fingerprints[key] = fingerprint(labels_path)["sha256"]
            consumed_surfaces[key] = str(
                Path(
                    str(subject_payload["inputs"][f"surface_hemi-{hemi}"]["path"])
                ).resolve()
            )
    for row in table_rows:
        key = (row["subject"], row["smoothing"], row["hemi"])
        if row["labels_sha256"] != label_fingerprints[key]:
            raise ValueError(f"Validation label fingerprint drift: {key}")
        if str(Path(row["consumed_surface"]).resolve()) != consumed_surfaces[key]:
            raise ValueError(f"Validation consumed surface drift: {key}")
    matcher = _load_matcher_module()
    template_root = path_roots.get("template_root")
    if not isinstance(template_root, str):
        raise ValueError(f"Validation template root is missing: {run_path}")
    dependency_template = matcher._load_template_contract(Path(template_root))
    matcher._validate_validation_payload_dependencies(
        payload=payload,
        expected_run_manifest=run_record,
        template_fingerprint=payload["template_manifest"],
        roi_registry_fingerprint=payload["roi_registry"],
        out_root=Path(str(path_roots["out_root"])),
        template=dependency_template,
    )
    matcher_context: dict[str, object] = {
        "path_roots": dict(path_roots),
        "parameters": dict(parameters),
    }
    return {
        "mode": "validation_manifest",
        "path": str(Path(path).resolve()),
        "fingerprint": fingerprint(path),
        "matcher_context": matcher_context,
    }


def load_source_manifest(
    *,
    matching_root: Path,
    subjects: Sequence[str],
    validation_manifest: Path | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    subject_ids = normalize_subjects(subjects)
    matching = Path(matching_root).resolve()
    candidate = (
        Path(validation_manifest).resolve() if validation_manifest is not None else None
    )
    if candidate is not None:
        source_validation = _validate_full_validation_manifest(
            path=candidate,
            matching_root=matching,
            subjects=subject_ids,
        )
    else:
        source_validation = {
            "mode": "subject_manifests",
            "subjects": [
                str(
                    (
                        matching
                        / "subjects"
                        / f"sub-{subject}"
                        / "subject_manifest.json"
                    ).resolve()
                )
                for subject in subject_ids
            ],
            "fingerprints": [],
        }
    matcher_context = (
        source_validation.get("matcher_context")
        if isinstance(source_validation.get("matcher_context"), Mapping)
        else None
    )
    sources = [
        load_subject_sources(
            matching_root=matching,
            subject=subject,
            matcher_context=matcher_context,
        )
        for subject in subject_ids
    ]
    source_validation["fingerprints"] = [
        {
            "path": source["subject_manifest"],
            "fingerprint": source["subject_manifest_fingerprint"],
        }
        for source in sources
    ]
    return sources, source_validation


def _canonical_json_sha256(payload: Mapping[str, object]) -> str:
    serialized = json.dumps(
        dict(payload),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def build_visual_label_contract() -> dict[str, object]:
    colors = require_network_colors(["VentralAttention", "Somatomotor", "Default"])
    mapping = {
        code: {
            "canonical_network": canonical,
            "display_name": display,
            "rgba": list(colors[canonical]),
            "source_label_semantics": LABEL_SEMANTICS[code],
        }
        for code, (canonical, display) in NETWORK_DISPLAY.items()
    }
    contract: dict[str, object] = {
        "schema_version": 1,
        "label_semantics": dict(LABEL_SEMANTICS),
        "label_mapping": mapping,
    }
    contract["contract_fingerprint"] = _canonical_json_sha256(contract)
    return contract


def _validated_png_fingerprint(path: Path) -> dict[str, object]:
    output = Path(path)
    if not output.is_file() or output.stat().st_size <= 0:
        raise ValueError(f"Overview PNG is missing or empty: {output}")
    try:
        with Image.open(output) as image:
            if image.format != "PNG":
                raise ValueError(f"Expected PNG, got {image.format!r}: {output}")
            image.verify()
        with Image.open(output) as image:
            image.load()
            if image.width <= 0 or image.height <= 0:
                raise ValueError(f"Overview PNG has invalid dimensions: {output}")
    except (OSError, SyntaxError, ValueError) as exc:
        raise ValueError(f"Overview PNG is invalid: {output}") from exc
    return fingerprint(output)


def overview_png_path(output_dir: Path, subject: str) -> Path:
    return (
        Path(output_dir).resolve()
        / f"sub-{normalize_subject(subject)}_hcp151_fixed_template_matching_overview.png"
    )


def overview_item_record_path(output_dir: Path, subject: str) -> Path:
    return (
        Path(output_dir).resolve()
        / f"fixed_template_overview_item_sub-{normalize_subject(subject)}.json"
    )


def _publish_item_record(output_dir: Path, item: Mapping[str, object]) -> None:
    path = overview_item_record_path(output_dir, str(item["subject"]))
    if path.exists():
        return
    _write_json_exclusive(path, item)


def _item(
    *,
    source: Mapping[str, object],
    output: Path,
    scene: Path,
    scene_fingerprint: Mapping[str, object],
    visual_contract: Mapping[str, object],
    resumed: bool,
    started: float,
) -> dict[str, object]:
    return {
        "status": "ok",
        "subject": str(source["subject"]),
        "output": str(Path(output).resolve()),
        "output_fingerprint": _validated_png_fingerprint(output),
        "scene": str(Path(scene).resolve()),
        "scene_fingerprint": dict(scene_fingerprint),
        "visual_label_contract_fingerprint": str(
            visual_contract["contract_fingerprint"]
        ),
        "source_manifest": str(source["subject_manifest"]),
        "source_manifest_fingerprint": dict(source["subject_manifest_fingerprint"]),
        "source_rows": [dict(row) for row in source["source_rows"]],
        "resumed": bool(resumed),
        "elapsed_seconds": float(time.time() - started),
    }


def _validate_existing_item(
    *,
    source: Mapping[str, object],
    item: Mapping[str, object],
    output: Path,
    scene: Path,
    scene_fingerprint: Mapping[str, object],
    visual_contract: Mapping[str, object],
) -> dict[str, object]:
    if item.get("status") != "ok" or item.get("subject") != source["subject"]:
        raise ValueError(f"Existing overview item is not reusable: {source['subject']}")
    if item.get("output") != str(output.resolve()):
        raise ValueError(f"Existing overview output path mismatch: {source['subject']}")
    if item.get("scene") != str(scene.resolve()) or item.get(
        "scene_fingerprint"
    ) != dict(scene_fingerprint):
        raise ValueError(f"Existing overview scene drift: {source['subject']}")
    if (
        item.get("visual_label_contract_fingerprint")
        != visual_contract["contract_fingerprint"]
    ):
        raise ValueError(
            f"Existing overview visual contract drift: {source['subject']}"
        )
    if item.get("source_manifest") != source["subject_manifest"]:
        raise ValueError(f"Existing overview source path drift: {source['subject']}")
    if (
        item.get("source_manifest_fingerprint")
        != source["subject_manifest_fingerprint"]
    ):
        raise ValueError(
            f"Existing overview source fingerprint drift: {source['subject']}"
        )
    if item.get("source_rows") != source["source_rows"]:
        raise ValueError(f"Existing overview source rows drift: {source['subject']}")
    recorded = item.get("output_fingerprint")
    _validate_fingerprint(
        recorded, output, context=f"Existing overview PNG {source['subject']}"
    )
    return _item(
        source=source,
        output=output,
        scene=scene,
        scene_fingerprint=scene_fingerprint,
        visual_contract=visual_contract,
        resumed=True,
        started=time.time(),
    )


def render_subject_overview(
    *,
    source: Mapping[str, object],
    output_dir: Path,
    scene: Path,
    visual_contract: Mapping[str, object],
    resume: bool = False,
    previous_item: Mapping[str, object] | None = None,
) -> dict[str, object]:
    started = time.time()
    subject = normalize_subject(str(source["subject"]))
    scene_path = Path(scene).resolve()
    if not scene_path.is_file():
        raise FileNotFoundError(f"Missing locked Workbench scene: {scene_path}")
    scene_fingerprint = fingerprint(scene_path)
    output = overview_png_path(output_dir, subject)
    if output.exists():
        if not resume:
            raise FileExistsError(
                f"Refusing to overwrite fixed-template overview: {output}; use --resume"
            )
        if previous_item is not None:
            try:
                item = _validate_existing_item(
                    source=source,
                    item=previous_item,
                    output=output,
                    scene=scene_path,
                    scene_fingerprint=scene_fingerprint,
                    visual_contract=visual_contract,
                )
                _publish_item_record(output_dir, item)
                return item
            except ValueError:
                pass

    with tempfile.TemporaryDirectory(prefix="jj_tms_hcp151_overview_") as scratch:
        scratch_root = Path(scratch)
        rendered_rows: list[tuple[str, Path]] = []
        for row in source["source_rows"]:
            smoothing = str(row["smoothing"])
            display_root = scratch_root / "display_labels" / smoothing
            left = row["L"]
            right = row["R"]
            left_display = write_network_display_label_gifti(
                source_label_path=Path(str(left["label_gifti"])),
                output_path=display_root / "hemi-L.label.gii",
                visual_label_contract=visual_contract,
            )
            right_display = write_network_display_label_gifti(
                source_label_path=Path(str(right["label_gifti"])),
                output_path=display_root / "hemi-R.label.gii",
                visual_label_contract=visual_contract,
            )
            rendered = render_locked_grid(
                subject=subject,
                scene=scene_path,
                outdir=scratch_root / smoothing,
                name=f"hcp151_fixed_template_matching_{smoothing}",
                title=(f"HCP151 fixed-template matching | sub-{subject} | {smoothing}"),
                left_labels=left_display,
                right_labels=right_display,
                left_surface=Path(str(left["surface"]["path"])),
                right_surface=Path(str(right["surface"]["path"])),
                spec_path=Path(str(left["surface"]["spec_path"])),
            )
            image = Path(str(rendered["biglegend_png"]))
            _validated_png_fingerprint(image)
            rendered_rows.append((smoothing, image))
        compose_stacked_rows(
            subject=subject,
            rows=rendered_rows,
            output_path=output,
            title=(f"HCP151 fixed-template Fisher-z network matching | sub-{subject}"),
        )
    item = _item(
        source=source,
        output=output,
        scene=scene_path,
        scene_fingerprint=scene_fingerprint,
        visual_contract=visual_contract,
        resumed=False,
        started=started,
    )
    _publish_item_record(output_dir, item)
    return item


def _worker(payload: Mapping[str, object]) -> dict[str, object]:
    for key, value in THREAD_ENV_OVERRIDES.items():
        os.environ[key] = value
    source = payload["source"]
    if not isinstance(source, Mapping):
        raise ValueError("Overview worker source must be a mapping")
    try:
        return render_subject_overview(
            source=source,
            output_dir=Path(str(payload["output_dir"])),
            scene=Path(str(payload["scene"])),
            visual_contract=payload["visual_contract"],
            resume=bool(payload["resume"]),
            previous_item=payload.get("previous_item"),
        )
    except Exception as exc:
        return {
            "status": "failed",
            "subject": str(source.get("subject", "")),
            "output": "",
            "resumed": False,
            "elapsed_seconds": 0.0,
            "error_type": exc.__class__.__name__,
            "error": str(exc),
        }


def _run_workers(
    *,
    sources: Sequence[Mapping[str, object]],
    output_dir: Path,
    scene: Path,
    visual_contract: Mapping[str, object],
    previous_items: Mapping[str, Mapping[str, object]],
    resume: bool,
    jobs: int,
) -> list[dict[str, object]]:
    tasks = [
        {
            "source": dict(source),
            "output_dir": str(output_dir),
            "scene": str(scene),
            "visual_contract": dict(visual_contract),
            "resume": bool(resume),
            "previous_item": previous_items.get(
                normalize_subject(str(source["subject"]))
            ),
        }
        for source in sources
    ]
    if jobs == 1:
        return [_worker(task) for task in tasks]
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = [executor.submit(_worker, task) for task in tasks]
        for future in as_completed(futures):
            rows.append(future.result())
    return rows


def _archive_failed_provenance(
    *,
    manifest_path: Path,
    status_path: Path,
) -> None:
    existing = [path for path in (manifest_path, status_path) if path.exists()]
    if not existing:
        return
    attempts: list[int] = []
    for path in manifest_path.parent.iterdir():
        name = path.name
        marker = ".attempt-"
        if (
            marker not in name
            or not name.endswith(".failed.json")
            and not name.endswith(".failed.tsv")
        ):
            continue
        try:
            attempts.append(int(name.split(marker, 1)[1].split(".", 1)[0]))
        except ValueError:
            continue
    attempt = max(attempts, default=0) + 1
    if manifest_path.exists():
        manifest_path.rename(
            manifest_path.with_name(
                f"fixed_template_overview_manifest.attempt-{attempt:03d}.failed.json"
            )
        )
    if status_path.exists():
        status_path.rename(
            status_path.with_name(
                f"fixed_template_overview_status.attempt-{attempt:03d}.failed.tsv"
            )
        )


def _archive_interrupted_provenance(
    *,
    manifest_path: Path,
    status_path: Path,
    manifest_partial_path: Path,
    status_partial_path: Path,
) -> None:
    existing = [
        path
        for path in (
            manifest_path,
            status_path,
            manifest_partial_path,
            status_partial_path,
        )
        if path.exists()
    ]
    if not existing:
        return
    attempts: list[int] = []
    for path in manifest_path.parent.iterdir():
        match = re.fullmatch(
            r"fixed_template_overview_(?:manifest|status)\.attempt-"
            r"(?P<number>[0-9]{3,})\.interrupted\.(?:json|tsv)",
            path.name,
        )
        if match is not None:
            attempts.append(int(match.group("number")))
    attempt = max(attempts, default=0) + 1
    for path in existing:
        stem = (
            "fixed_template_overview_manifest"
            if path.suffix == ".json"
            else "fixed_template_overview_status"
        )
        history = manifest_path.parent / (
            f"{stem}.attempt-{attempt:03d}.interrupted{path.suffix}"
        )
        while history.exists():
            attempt += 1
            history = manifest_path.parent / (
                f"{stem}.attempt-{attempt:03d}.interrupted{path.suffix}"
            )
        path.rename(history)


def _validate_existing_status_tsv(
    *,
    status_path: Path,
    summary: Mapping[str, object],
    items: Sequence[Mapping[str, object]],
) -> None:
    recorded = summary.get("status_tsv_fingerprint")
    if recorded is not None and recorded != fingerprint(status_path):
        raise ValueError("Existing overview status TSV fingerprint drift")
    with status_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != len(items):
        raise ValueError("Existing overview status TSV row count drift")
    expected_subjects = [str(item.get("subject")) for item in items]
    if [row.get("subject") for row in rows] != expected_subjects:
        raise ValueError("Existing overview status TSV subject drift")
    if any(row.get("status") != "ok" for row in rows):
        raise ValueError("Existing overview status TSV contains non-ok rows")


def _source_validation_compatible(
    recorded: object,
    expected: Mapping[str, object],
) -> bool:
    if recorded == expected:
        return True
    if not isinstance(recorded, Mapping):
        return False
    recorded_without_context = dict(recorded)
    expected_without_context = dict(expected)
    recorded_without_context.pop("matcher_context", None)
    expected_without_context.pop("matcher_context", None)
    return recorded_without_context == expected_without_context


def _load_item_records(
    *,
    destination: Path,
    sources: Sequence[Mapping[str, object]],
    scene: Path,
    scene_fingerprint: Mapping[str, object],
    visual_contract: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    source_by_subject = {
        normalize_subject(str(source["subject"])): source for source in sources
    }
    records: dict[str, Mapping[str, object]] = {}
    for path in sorted(destination.glob("fixed_template_overview_item_sub-*.json")):
        match = re.fullmatch(r"fixed_template_overview_item_sub-(\d+)\.json", path.name)
        if match is None or match.group(1) not in source_by_subject:
            raise ValueError(f"Unexpected overview item record: {path}")
        subject = match.group(1)
        item = _load_json(path)
        _validate_existing_item(
            source=source_by_subject[subject],
            item=item,
            output=overview_png_path(destination, subject),
            scene=scene,
            scene_fingerprint=scene_fingerprint,
            visual_contract=visual_contract,
        )
        records[subject] = item
    return records


def _write_status_exclusive(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite overview status: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=STATUS_FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: str(row.get(field, "")).replace("\t", " ").replace("\n", " ")
                    for field in STATUS_FIELDS
                }
            )
    os.replace(temporary, destination)


def render_fixed_template_overviews(
    *,
    matching_root: Path = DEFAULT_MATCHING_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    scene: Path = DEFAULT_SCENE,
    subjects: Sequence[str] | None = None,
    subjects_manifest: Path = DEFAULT_SUBJECTS_MANIFEST,
    run_id: str | None = DEFAULT_RUN_ID,
    validation_manifest: Path | None = None,
    jobs: int = DEFAULT_JOBS,
    resume: bool = False,
    allow_subset: bool = False,
) -> dict[str, Path]:
    if not 1 <= int(jobs) <= MAX_JOBS:
        raise ValueError(f"jobs must be in [1, {MAX_JOBS}]")
    selected = (
        normalize_subjects(subjects)
        if subjects is not None
        else load_subject_roster(Path(subjects_manifest))
    )
    canonical_subjects = load_subject_roster(Path(subjects_manifest))
    full_scope = selected == canonical_subjects
    if not full_scope and not allow_subset:
        raise ValueError("Subset overview rendering requires explicit --allow-subset")
    scope = "full" if full_scope else "subset"
    matching = Path(matching_root).resolve()
    candidate_validation = (
        Path(validation_manifest).resolve()
        if validation_manifest is not None
        else matching / "runs" / str(run_id) / "validation_manifest.json"
        if run_id
        else None
    )
    if candidate_validation is not None and not candidate_validation.is_file():
        if validation_manifest is not None:
            raise FileNotFoundError(
                f"Missing validation manifest: {candidate_validation}"
            )
        candidate_validation = None
    sources, source_validation = load_source_manifest(
        matching_root=matching,
        subjects=selected,
        validation_manifest=candidate_validation,
    )
    scene_path = Path(scene).resolve()
    scene_fingerprint = fingerprint(scene_path)
    visual_contract = build_visual_label_contract()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "fixed_template_overview_manifest.json"
    status_path = destination / "fixed_template_overview_status.tsv"
    manifest_partial_path = (
        destination / "fixed_template_overview_manifest.partial.json"
    )
    status_partial_path = destination / "fixed_template_overview_status.partial.tsv"
    aggregate_paths = (
        manifest_path,
        status_path,
        manifest_partial_path,
        status_partial_path,
    )
    existing_pngs = list(
        destination.glob("*_hcp151_fixed_template_matching_overview.png")
    )
    if existing_pngs and not (manifest_path.exists() or manifest_partial_path.exists()):
        raise ValueError(
            "Existing fixed-template overview PNGs have no provenance manifest; "
            "refuse resume reuse"
        )
    if any(path.exists() for path in aggregate_paths) and not resume:
        raise FileExistsError(
            f"Overview provenance already exists under {destination}; use --resume"
        )
    item_records = _load_item_records(
        destination=destination,
        sources=sources,
        scene=scene_path,
        scene_fingerprint=scene_fingerprint,
        visual_contract=visual_contract,
    )

    previous_items: dict[str, Mapping[str, object]] = {}
    manifest_source = (
        manifest_path
        if manifest_path.exists()
        else manifest_partial_path
        if manifest_partial_path.exists()
        else None
    )
    prior_summary: Mapping[str, object] | None = None
    if manifest_source is not None:
        try:
            prior = _load_json(manifest_source)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if not resume:
                raise FileExistsError(
                    f"Overview provenance is malformed; use --resume: {manifest_source}"
                )
            _archive_interrupted_provenance(
                manifest_path=manifest_path,
                status_path=status_path,
                manifest_partial_path=manifest_partial_path,
                status_partial_path=status_partial_path,
            )
            manifest_source = None
            prior = None
        if prior is None:
            prior_summary = None
        else:
            summary = prior.get("summary")
            items = prior.get("items")
            if (
                not isinstance(summary, Mapping)
                or summary.get("workflow") != WORKFLOW_NAME
                or summary.get("subjects") != [f"sub-{subject}" for subject in selected]
                or summary.get("scope", "full" if full_scope else "subset") != scope
                or summary.get("scene") != str(scene_path)
                or summary.get("scene_fingerprint") != scene_fingerprint
                or not _source_validation_compatible(
                    summary.get("source_validation"),
                    source_validation,
                )
                or summary.get("visual_label_contract_fingerprint")
                != visual_contract["contract_fingerprint"]
                or not isinstance(items, list)
            ):
                if resume:
                    _archive_interrupted_provenance(
                        manifest_path=manifest_path,
                        status_path=status_path,
                        manifest_partial_path=manifest_partial_path,
                        status_partial_path=status_partial_path,
                    )
                    manifest_source = None
                    prior = None
                    prior_summary = None
                else:
                    raise ValueError(
                        "Existing fixed-template overview manifest contract drift"
                    )
            for item in items:
                if not isinstance(item, Mapping):
                    raise ValueError("Existing fixed-template overview item is invalid")
                subject = normalize_subject(str(item.get("subject", "")))
                if subject in previous_items:
                    raise ValueError(
                        f"Existing overview manifest has duplicate subject: {subject}"
                    )
                previous_items[subject] = dict(item)
            for subject, item in item_records.items():
                if subject in previous_items and previous_items[subject] != item:
                    raise ValueError(f"Overview item record drift: sub-{subject}")
                previous_items[subject] = item
            prior_summary = summary
            status_source = (
                status_path
                if status_path.exists()
                else status_partial_path
                if status_partial_path.exists()
                else None
            )
            if (
                summary.get("status") in {"ok", "subset"}
                and len(previous_items) == len(selected)
                and status_source is not None
            ):
                _validate_existing_status_tsv(
                    status_path=status_source,
                    summary=summary,
                    items=[
                        dict(previous_items[subject])
                        for subject in sorted(previous_items)
                    ],
                )
            if (
                manifest_source == manifest_path
                and status_source == status_path
                and summary.get("status") in {"ok", "subset"}
                and len(previous_items) == len(selected)
            ):
                for source in sources:
                    subject = str(source["subject"])
                    _validate_existing_item(
                        source=source,
                        item=previous_items[normalize_subject(subject)],
                        output=overview_png_path(destination, subject),
                        scene=scene_path,
                        scene_fingerprint=scene_fingerprint,
                        visual_contract=visual_contract,
                    )
                return {
                    "status_tsv": status_path,
                    "manifest": manifest_path,
                }

    if any(path.exists() for path in aggregate_paths):
        if (
            manifest_path.exists()
            and isinstance(prior_summary, Mapping)
            and prior_summary.get("status") == "failed"
        ):
            _archive_failed_provenance(
                manifest_path=manifest_path,
                status_path=status_path,
            )
        else:
            _archive_interrupted_provenance(
                manifest_path=manifest_path,
                status_path=status_path,
                manifest_partial_path=manifest_partial_path,
                status_partial_path=status_partial_path,
            )

    in_progress_summary = {
        "status": "in_progress",
        "scope": scope,
        "workflow": WORKFLOW_NAME,
        "matching_root": str(matching),
        "output_dir": str(destination),
        "subjects": [f"sub-{subject}" for subject in selected],
        "requested_subject_count": len(selected),
        "total": len(previous_items),
        "ok": len(previous_items),
        "failed": 0,
        "scene": str(scene_path),
        "scene_fingerprint": scene_fingerprint,
        "source_validation": source_validation,
        "visual_label_contract": visual_contract,
        "visual_label_contract_fingerprint": visual_contract["contract_fingerprint"],
        "smoothings": list(SMOOTHINGS),
        "hemis": list(HEMIS),
        "jobs": int(jobs),
        "updated_at": _now(),
    }
    _write_json_exclusive(
        manifest_partial_path,
        {
            "summary": in_progress_summary,
            "items": [previous_items[subject] for subject in sorted(previous_items)],
        },
    )
    effective_jobs = max(1, min(int(jobs), MAX_JOBS, len(sources) or 1))
    rows = sorted(
        _run_workers(
            sources=sources,
            output_dir=destination,
            scene=scene_path,
            visual_contract=visual_contract,
            previous_items=previous_items,
            resume=resume,
            jobs=effective_jobs,
        ),
        key=lambda row: str(row.get("subject", "")),
    )
    failed = [row for row in rows if row.get("status") != "ok"]
    summary = {
        "status": "failed" if failed else ("ok" if full_scope else "subset"),
        "scope": scope,
        "workflow": WORKFLOW_NAME,
        "matching_root": str(matching),
        "output_dir": str(destination),
        "subjects": [f"sub-{subject}" for subject in selected],
        "requested_subject_count": len(selected),
        "total": len(rows),
        "ok": len(rows) - len(failed),
        "failed": len(failed),
        "scene": str(scene_path),
        "scene_fingerprint": scene_fingerprint,
        "source_validation": source_validation,
        "visual_label_contract": visual_contract,
        "visual_label_contract_fingerprint": visual_contract["contract_fingerprint"],
        "smoothings": list(SMOOTHINGS),
        "hemis": list(HEMIS),
        "jobs": effective_jobs,
        "updated_at": _now(),
    }
    _write_status_exclusive(status_partial_path, rows)
    summary["status_tsv"] = str(status_path)
    summary["status_tsv_fingerprint"] = fingerprint(status_partial_path)
    _write_json_exclusive(
        manifest_partial_path,
        {
            "summary": summary,
            "items": rows,
        },
    )
    status_partial_path.rename(status_path)
    manifest_partial_path.rename(manifest_path)
    if failed:
        raise RuntimeError(
            f"Fixed-template overview rendering failed for {len(failed)} subjects"
        )
    return {
        "status_tsv": status_path,
        "manifest": manifest_path,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render one HCP-style 2mm/4mm PNG per JJ_TMS fixed-template subject."
    )
    parser.add_argument("--matching-root", type=Path, default=DEFAULT_MATCHING_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument(
        "--subjects-manifest", type=Path, default=DEFAULT_SUBJECTS_MANIFEST
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-subset", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    outputs = render_fixed_template_overviews(
        matching_root=args.matching_root,
        output_dir=args.output_dir,
        scene=args.scene,
        subjects=args.subjects,
        subjects_manifest=args.subjects_manifest,
        run_id=args.run_id,
        validation_manifest=args.validation_manifest,
        jobs=args.jobs,
        resume=args.resume,
        allow_subset=args.allow_subset,
    )
    print(f"status_tsv={outputs['status_tsv']}")
    print(f"manifest={outputs['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

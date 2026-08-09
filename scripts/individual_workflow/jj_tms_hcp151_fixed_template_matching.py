#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.gifti import GiftiDataArray, GiftiImage


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_WORKFLOW_DIR = REPO_ROOT / "scripts" / "core_workflow"
if str(CORE_WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_WORKFLOW_DIR))

from lib.extract_cortex_kong2022_route_a_roi_timeseries import (
    align_surface_labels_to_dt_vertices,
)
from lib.extract_cortex_merged_network_timeseries import extract_structure_data
from lib.fc_math import corrcoef_rows, fisher_z_transform_fc
from lib.gifti_labels import make_named_label_gifti
from lib.template_matching import match_vertices_to_prototypes
from lib.workbench import workbench_env


SMOOTHINGS = ("2mm", "4mm")
HEMIS = ("L", "R")
N_VERTICES = 512
N_SURFACE_VERTICES_32K = 32492
N_ROIS = 400
K = 3
TSNR_THRESHOLD = 25.0
HCP_MODE = 10000.0
MAX_JOBS = 10
DEFAULT_JOBS = 2
EXPECTED_SUBJECT_COUNT = 70
TWO_RUN_SUBJECT = "1016"
TEMPLATE_ID = "hcp151_routea_full_k3_fisherz"
DEFAULT_DATA_ROOT = Path(
    "/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS"
)
DEFAULT_VOLUME_ROOT = DEFAULT_DATA_ROOT / "xcpd" / "restfmri"
DEFAULT_DTSERIES_ROOT = DEFAULT_DATA_ROOT / "xcpd_surf"
DEFAULT_CONCAT_DTSERIES_ROOT = DEFAULT_DATA_ROOT / "xcpd_surf_concat"
DEFAULT_AREAL_LABEL_ROOT = DEFAULT_DATA_ROOT / "ArealMSHBM_Kong2022"
DEFAULT_HIPPUNFOLD_ROOT = (
    Path("/home/yojiao/hpchome/hipp")
    / "outputs_iowa_jj_tms"
    / "inputs"
    / "iowa_jj_tms"
    / "hippunfold"
    / "dense_corobl_batch"
)
DEFAULT_TEMPLATE_ROOT = (
    Path("/home/yojiao/hpchome/hipp")
    / "outputs_hcp"
    / "templates"
    / TEMPLATE_ID
)
DEFAULT_OUT_ROOT = (
    Path("/home/yojiao/hpchome/hipp")
    / "outputs_iowa_jj_tms"
    / "work"
    / "hcp151_fixed_template_matching"
)
DEFAULT_SUBJECTS_MANIFEST = (
    REPO_ROOT / "manifests" / "jj_tms_hcp151_fixed_template_subjects.txt"
)
DEFAULT_WB_COMMAND = (
    Path("/home/yojiao/hpchome/hipp")
    / "software"
    / "workbench"
    / "bin_linux64"
    / "wb_command"
)
RUN_RE = re.compile(r"_run-(?P<run>\d+)_")
LABEL_ITEMS = [
    (0, "Null", (158, 158, 158, 255)),
    (1, "Anterior/VAN", (220, 92, 255, 255)),
    (2, "Body/SMN", (99, 150, 194, 255)),
    (3, "Posterior/DMN", (216, 92, 107, 255)),
]


@dataclass(frozen=True)
class RunInput:
    run_id: int
    volume: Path
    dtseries: Path


@dataclass(frozen=True)
class SubjectInputs:
    subject: str
    runs: tuple[RunInput, ...]
    concat_dtseries: Path


@dataclass(frozen=True)
class TemplateContract:
    root: Path
    manifest_path: Path
    manifest_fingerprint: Mapping[str, object]
    roi_registry_path: Path
    roi_registry_fingerprint: Mapping[str, object]
    roi_registry: tuple[Mapping[str, object], ...]
    prototypes: Mapping[tuple[str, str], Path]


def normalize_subject(raw: object) -> str:
    subject = str(raw).strip().removeprefix("sub-")
    if not subject or not subject.isdigit():
        raise ValueError(f"Invalid JJ_TMS subject ID: {raw!r}")
    return subject


def read_subject_manifest(path: Path) -> list[str]:
    subjects: list[str] = []
    for line_number, raw in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            subjects.append(normalize_subject(line))
        except ValueError as exc:
            raise ValueError(f"Invalid subject manifest line {line_number}: {exc}") from exc
    if len(subjects) != len(set(subjects)):
        raise ValueError(f"Duplicate subjects in manifest: {path}")
    return subjects


def select_subjects(
    *,
    subjects_manifest: Path,
    explicit_subjects: Sequence[str] | None,
) -> tuple[list[str], list[str]]:
    canonical = read_subject_manifest(DEFAULT_SUBJECTS_MANIFEST)
    if len(canonical) != EXPECTED_SUBJECT_COUNT:
        raise ValueError(
            f"Canonical JJ_TMS roster must contain 70 subjects, got {len(canonical)}"
        )
    requested_manifest = read_subject_manifest(subjects_manifest)
    if requested_manifest != canonical:
        raise ValueError("Requested roster manifest differs from the fixed 70-subject roster")
    if not explicit_subjects:
        return list(canonical), list(canonical)
    normalized = [normalize_subject(subject) for subject in explicit_subjects]
    if len(normalized) != len(set(normalized)):
        raise ValueError("Duplicate --subjects values")
    unknown = sorted(set(normalized) - set(canonical), key=int)
    if unknown:
        raise ValueError(f"Subjects are outside the fixed JJ_TMS roster: {unknown}")
    selected = [subject for subject in canonical if subject in set(normalized)]
    if selected != normalized:
        raise ValueError("--subjects must follow the fixed roster order")
    return selected, list(canonical)


def _paths_by_run(paths: Sequence[Path], *, label: str) -> dict[int, Path]:
    by_run: dict[int, Path] = {}
    for path in paths:
        match = RUN_RE.search(path.name)
        if match is None:
            raise ValueError(f"Could not parse numeric run ID from {label}: {path}")
        run_id = int(match.group("run"))
        if run_id in by_run:
            raise ValueError(f"Duplicate run-{run_id:02d} {label} files")
        by_run[run_id] = path.resolve()
    return by_run


def resolve_subject_runs(
    *,
    subject: str,
    volume_root: Path,
    dtseries_root: Path,
    concat_dtseries_root: Path,
) -> SubjectInputs:
    subject_id = normalize_subject(subject)
    volume_dir = Path(volume_root) / f"sub-{subject_id}" / "func"
    dtseries_dir = Path(dtseries_root) / f"sub-{subject_id}" / "func"
    volumes = _paths_by_run(
        list(
            volume_dir.glob(
                f"sub-{subject_id}_task-rest_run-*_space-MNI152NLin6Asym_"
                "res-2_desc-denoised_bold.nii.gz"
            )
        ),
        label="volume",
    )
    dtseries = _paths_by_run(
        list(
            dtseries_dir.glob(
                f"sub-{subject_id}_task-rest_run-*_space-fsLR_den-91k_"
                "desc-denoised_bold.dtseries.nii"
            )
        ),
        label="dtseries",
    )
    expected_ids = [1, 2] if subject_id == TWO_RUN_SUBJECT else [1, 2, 3]
    if sorted(volumes) != expected_ids or sorted(dtseries) != expected_ids:
        raise ValueError(
            f"Unexpected task-rest run IDs for sub-{subject_id}: "
            f"volume={sorted(volumes)} dtseries={sorted(dtseries)} "
            f"expected={expected_ids}"
        )
    concat = (
        Path(concat_dtseries_root)
        / f"sub-{subject_id}"
        / "func"
        / (
            f"sub-{subject_id}_task-rest_run-concat_space-fsLR_den-91k_"
            "desc-denoised_bold.dtseries.nii"
        )
    ).resolve()
    if not concat.is_file():
        raise FileNotFoundError(f"Missing 91k concat dtseries: {concat}")
    return SubjectInputs(
        subject=subject_id,
        runs=tuple(
            RunInput(run_id=run_id, volume=volumes[run_id], dtseries=dtseries[run_id])
            for run_id in expected_ids
        ),
        concat_dtseries=concat,
    )


def validate_functional_headers(inputs: SubjectInputs) -> dict[str, object]:
    spatial_shape: tuple[int, ...] | None = None
    affine: np.ndarray | None = None
    grayordinates: int | None = None
    run_lengths: list[int] = []
    rows: list[dict[str, object]] = []
    for run in inputs.runs:
        volume = nib.load(str(run.volume))
        if len(volume.shape) != 4:
            raise ValueError(f"Expected 4D volume for run-{run.run_id:02d}: {volume.shape}")
        current_grid = tuple(int(value) for value in volume.shape[:3])
        current_affine = np.asarray(volume.affine, dtype=np.float64)
        if spatial_shape is None:
            spatial_shape = current_grid
            affine = current_affine
        elif current_grid != spatial_shape:
            raise ValueError(
                f"Volume grid mismatch for sub-{inputs.subject} run-{run.run_id:02d}: "
                f"{current_grid} != {spatial_shape}"
            )
        elif not np.allclose(current_affine, affine, rtol=0.0, atol=1e-6):
            raise ValueError(
                f"Volume affine mismatch for sub-{inputs.subject} run-{run.run_id:02d}"
            )
        dtseries = nib.load(str(run.dtseries))
        if len(dtseries.shape) != 2:
            raise ValueError(f"Expected 2D dtseries: {run.dtseries}")
        volume_tp = int(volume.shape[3])
        dtseries_tp = int(dtseries.shape[0])
        if volume_tp != dtseries_tp:
            raise ValueError(
                f"Run timepoint mismatch for sub-{inputs.subject} run-{run.run_id:02d}: "
                f"volume={volume_tp} dtseries={dtseries_tp}"
            )
        current_grayordinates = int(dtseries.shape[1])
        if grayordinates is None:
            grayordinates = current_grayordinates
        elif current_grayordinates != grayordinates:
            raise ValueError("Run dtseries grayordinate grid mismatch")
        run_lengths.append(volume_tp)
        rows.append(
            {
                "run_id": run.run_id,
                "volume_timepoints": volume_tp,
                "dtseries_timepoints": dtseries_tp,
                "grayordinates": current_grayordinates,
            }
        )
    concat = nib.load(str(inputs.concat_dtseries))
    if len(concat.shape) != 2:
        raise ValueError(f"Expected 2D concat dtseries: {inputs.concat_dtseries}")
    total = int(sum(run_lengths))
    if int(concat.shape[0]) != total:
        raise ValueError(
            f"Concat timepoint mismatch for sub-{inputs.subject}: "
            f"concat={concat.shape[0]} run_sum={total}"
        )
    if int(concat.shape[1]) != int(grayordinates or -1):
        raise ValueError("Concat dtseries grayordinate grid mismatch")
    return {
        "subject": inputs.subject,
        "run_lengths": run_lengths,
        "total_timepoints": total,
        "volume_grid": list(spatial_shape or ()),
        "volume_affine": np.asarray(affine).tolist(),
        "grayordinates": int(grayordinates or 0),
        "concat_timepoints": int(concat.shape[0]),
        "runs": rows,
    }


def load_areal_labels(*, label_root: Path, subject: str) -> dict[str, np.ndarray]:
    subject_id = normalize_subject(subject)
    subject_dir = Path(label_root) / f"sub-{subject_id}"
    result: dict[str, np.ndarray] = {}
    for hemi, filename in (("L", "lh_labels.txt"), ("R", "rh_labels.txt")):
        path = subject_dir / filename
        values = np.loadtxt(path, dtype=np.int32).reshape(-1)
        if values.shape != (N_SURFACE_VERTICES_32K,):
            raise ValueError(
                f"Expected 32492 ArealMSHBM labels in {path}, got {values.shape}"
            )
        allowed = (
            (values >= 0) & (values <= 200)
            if hemi == "L"
            else ((values == 0) | ((values >= 201) & (values <= 400)))
        )
        if not np.all(allowed):
            bad = sorted(set(int(value) for value in values[~allowed]))[:10]
            raise ValueError(f"Invalid global Kong2022 ROI IDs in {path}: {bad}")
        result[hemi] = values.astype(np.int32, copy=False)
    return result


def read_roi_registry(path: Path) -> list[dict[str, object]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        rows = [dict(row) for row in reader]
    if "roi_id" not in list(reader.fieldnames or []):
        raise ValueError(f"ROI registry lacks roi_id: {path}")
    try:
        ids = [int(row["roi_id"]) for row in rows]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"ROI registry contains invalid IDs: {path}") from exc
    if ids != list(range(1, N_ROIS + 1)):
        raise ValueError("ROI registry order must be exactly 1 through 400")
    for row, roi_id in zip(rows, ids, strict=True):
        row["roi_id"] = roi_id
    return rows


def extract_fixed_roi_timeseries(
    *,
    cortex_by_hemi: Mapping[str, np.ndarray],
    aligned_labels_by_hemi: Mapping[str, np.ndarray],
    roi_ids: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    requested = [int(value) for value in roi_ids]
    if requested != list(range(1, N_ROIS + 1)):
        raise ValueError("ROI extraction order must be exactly 1 through 400")
    n_timepoints: int | None = None
    for hemi in HEMIS:
        data = np.asarray(cortex_by_hemi[hemi], dtype=np.float32)
        labels = np.asarray(aligned_labels_by_hemi[hemi], dtype=np.int32).reshape(-1)
        if data.ndim != 2 or data.shape[0] != labels.size:
            raise ValueError(
                f"Cortex/label shape mismatch for hemi-{hemi}: {data.shape} vs {labels.shape}"
            )
        if n_timepoints is None:
            n_timepoints = int(data.shape[1])
        elif int(data.shape[1]) != n_timepoints:
            raise ValueError("Left/right cortex timepoint mismatch")
    output = np.full((N_ROIS, int(n_timepoints or 0)), np.nan, dtype=np.float32)
    counts = np.zeros(N_ROIS, dtype=np.int32)
    for roi_id in requested:
        hemi = "L" if roi_id <= 200 else "R"
        data = np.asarray(cortex_by_hemi[hemi], dtype=np.float32)
        labels = np.asarray(aligned_labels_by_hemi[hemi], dtype=np.int32).reshape(-1)
        indices = np.flatnonzero(labels == roi_id)
        counts[roi_id - 1] = int(indices.size)
        if indices.size:
            output[roi_id - 1] = np.nanmean(data[indices], axis=0).astype(np.float32)
    return output, counts


def compute_tsnr(
    timeseries: np.ndarray,
    *,
    threshold: float = TSNR_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(timeseries, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D time series, got {data.shape}")
    standard_deviation = np.nanstd(data, axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        tsnr = np.where(
            np.isfinite(standard_deviation) & (standard_deviation > 0.0),
            HCP_MODE / standard_deviation,
            np.nan,
        ).astype(np.float32)
    valid = np.isfinite(tsnr) & (tsnr >= float(threshold))
    return tsnr, valid.astype(bool)


def validate_combo_arrays(
    *,
    labels: np.ndarray,
    scores: np.ndarray,
    margin: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, object]:
    label_array = np.asarray(labels)
    score_array = np.asarray(scores)
    margin_array = np.asarray(margin)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if label_array.shape != (N_VERTICES,) or label_array.dtype != np.dtype(np.int32):
        raise ValueError(
            f"Expected labels shape=(512,) dtype=int32, got {label_array.shape} {label_array.dtype}"
        )
    if score_array.shape != (N_VERTICES, K) or score_array.dtype != np.dtype(np.float32):
        raise ValueError(
            f"Expected scores shape=(512, 3) dtype=float32, got {score_array.shape} {score_array.dtype}"
        )
    if margin_array.shape != (N_VERTICES,) or margin_array.dtype != np.dtype(np.float32):
        raise ValueError(
            f"Expected margin shape=(512,) dtype=float32, got {margin_array.shape} {margin_array.dtype}"
        )
    if valid.shape != (N_VERTICES,):
        raise ValueError(f"Expected valid mask shape=(512,), got {valid.shape}")
    if np.any(label_array[valid] == 0):
        raise ValueError("Every valid vertex must receive a nonzero label")
    if set(int(value) for value in np.unique(label_array[valid])) != {1, 2, 3}:
        raise ValueError("All three AP labels must appear among valid vertices")
    if np.any(label_array[~valid] != 0):
        raise ValueError("Invalid vertices must retain null label 0")
    finite_scores = score_array[valid][np.isfinite(score_array[valid])]
    if finite_scores.size == 0 or np.any((finite_scores < -1.0) | (finite_scores > 1.0)):
        raise ValueError("Template correlations must contain finite values in [-1, 1]")
    finite_margin = margin_array[valid][np.isfinite(margin_array[valid])]
    if finite_margin.size == 0 or np.any((finite_margin < 0.0) | (finite_margin > 2.0)):
        raise ValueError("Correlation margins must contain finite values in [0, 2]")
    counts = {
        str(code): int(np.sum(label_array[valid] == code)) for code in range(1, K + 1)
    }
    return {
        "valid_vertices": int(np.sum(valid)),
        "label_counts": counts,
        "margin_summary": {
            "min": float(np.min(finite_margin)),
            "median": float(np.median(finite_margin)),
            "mean": float(np.mean(finite_margin)),
            "max": float(np.max(finite_margin)),
        },
    }


def fingerprint(path: Path) -> dict[str, object]:
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


def artifact_record(path: Path) -> dict[str, object]:
    record = fingerprint(path)
    if Path(path).suffix == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        record.update(shape=list(array.shape), dtype=str(array.dtype))
    return record


def build_subject_manifest(
    *,
    subject: str,
    inputs: Mapping[str, Path],
    parameters: Mapping[str, object],
    outputs: Mapping[str, Path],
    combinations: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "status": "ok",
        "workflow": "jj_tms_hcp151_fixed_template_matching",
        "subject": f"sub-{normalize_subject(subject)}",
        "parameters": dict(parameters),
        "inputs": {key: fingerprint(path) for key, path in sorted(inputs.items())},
        "outputs": {key: artifact_record(path) for key, path in sorted(outputs.items())},
        "combinations": [dict(row) for row in combinations],
    }


def _validate_fingerprint(recorded: object, *, context: str) -> None:
    if not isinstance(recorded, Mapping) or not isinstance(recorded.get("path"), str):
        raise ValueError(f"Missing fingerprint metadata for {context}")
    actual = fingerprint(Path(str(recorded["path"])))
    expected = {key: recorded.get(key) for key in ("path", "sha256", "size_bytes")}
    if actual != expected:
        raise ValueError(
            f"{context} fingerprint mismatch: recorded={expected!r} actual={actual!r}"
        )


def validate_reusable_subject(
    *,
    subject_dir: Path,
    expected_inputs: Mapping[str, Path],
    expected_parameters: Mapping[str, object],
) -> dict[str, object]:
    manifest_path = Path(subject_dir) / "subject_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("status") != "ok":
        raise ValueError(f"Reusable subject manifest is not status=ok: {manifest_path}")
    if payload.get("parameters") != dict(expected_parameters):
        raise ValueError("Resume parameter mismatch")
    recorded_inputs = payload.get("inputs")
    if not isinstance(recorded_inputs, Mapping) or set(recorded_inputs) != set(expected_inputs):
        raise ValueError("Resume input set mismatch")
    for key, path in expected_inputs.items():
        recorded = recorded_inputs[key]
        if not isinstance(recorded, Mapping) or Path(str(recorded.get("path"))).resolve() != Path(path).resolve():
            raise ValueError(f"Resume input path mismatch for {key}")
        _validate_fingerprint(recorded, context=f"resume input {key}")
    recorded_outputs = payload.get("outputs")
    if not isinstance(recorded_outputs, Mapping) or not recorded_outputs:
        raise ValueError("Resume output metadata is missing")
    for key, raw in recorded_outputs.items():
        if not isinstance(raw, Mapping) or not isinstance(raw.get("path"), str):
            raise ValueError(f"Resume output metadata is invalid for {key}")
        output_path = Path(str(raw["path"]))
        if "shape" in raw or "dtype" in raw:
            array = np.load(output_path, mmap_mode="r", allow_pickle=False)
            if list(array.shape) != raw.get("shape"):
                raise ValueError(f"Resume output shape mismatch for {key}")
            if str(array.dtype) != raw.get("dtype"):
                raise ValueError(f"Resume output dtype mismatch for {key}")
        _validate_fingerprint(raw, context=f"resume output {key}")
    return payload


def prepare_empty_failed_subject_resume(
    *,
    subject_dir: Path,
    expected_inputs: Mapping[str, Path],
    expected_parameters: Mapping[str, object],
) -> dict[str, object]:
    directory = Path(subject_dir)
    partial = directory / "subject_manifest.partial.json"
    entries = list(directory.iterdir())
    if entries != [partial]:
        raise ValueError(
            "Failed subject resume requires the partial manifest to be the only "
            f"directory entry: {directory}"
        )
    original = partial.read_bytes()
    payload = json.loads(original)
    if not isinstance(payload, dict) or payload.get("status") != "failed":
        raise ValueError(f"Subject partial is not status=failed: {partial}")
    if payload.get("parameters") != dict(expected_parameters):
        raise ValueError("Failed subject resume parameter mismatch")
    recorded_inputs = payload.get("inputs")
    if not isinstance(recorded_inputs, Mapping) or set(recorded_inputs) != set(
        expected_inputs
    ):
        raise ValueError("Failed subject resume input set mismatch")
    for key, path in expected_inputs.items():
        recorded = recorded_inputs[key]
        if (
            not isinstance(recorded, Mapping)
            or Path(str(recorded.get("path"))).resolve() != Path(path).resolve()
        ):
            raise ValueError(f"Failed subject resume input path mismatch for {key}")
        _validate_fingerprint(recorded, context=f"failed resume input {key}")
    retained = payload.get("retained_outputs")
    if not isinstance(retained, Mapping) or retained:
        raise ValueError("Failed subject resume retained outputs are not empty")
    history = directory / "subject_manifest.attempt-001.failed.json"
    with history.open("xb") as stream:
        stream.write(original)
    record = fingerprint(history)
    record["prior_error"] = payload.get("error")
    return record


def _load_template_contract(template_root: Path) -> TemplateContract:
    root = Path(template_root).resolve()
    manifest_path = root / "template_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "status": "complete",
        "template_id": TEMPLATE_ID,
        "n_subjects": 151,
        "density": N_VERTICES,
        "k": K,
        "n_rois": N_ROIS,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"HCP template {key} mismatch: {manifest.get(key)!r} != {value!r}"
            )
    registry_record = manifest.get("roi_registry")
    _validate_fingerprint(registry_record, context="HCP ROI registry")
    registry_path = Path(str(registry_record["path"])).resolve()
    registry = tuple(read_roi_registry(registry_path))
    combinations = manifest.get("combinations")
    if not isinstance(combinations, list) or len(combinations) != 4:
        raise ValueError("HCP template must contain exactly four combinations")
    prototypes: dict[tuple[str, str], Path] = {}
    for raw in combinations:
        if not isinstance(raw, Mapping):
            raise ValueError("Invalid HCP template combination")
        smoothing = str(raw.get("smoothing"))
        hemi = str(raw.get("hemi"))
        if smoothing not in SMOOTHINGS or hemi not in HEMIS:
            raise ValueError(f"Invalid HCP template combination: {smoothing} hemi-{hemi}")
        sources = raw.get("source_files")
        outputs = raw.get("outputs")
        if not isinstance(sources, Mapping) or not isinstance(outputs, Mapping):
            raise ValueError("HCP template combination lacks provenance records")
        for name, record in sources.items():
            _validate_fingerprint(record, context=f"HCP template source {name}")
        for name, record in outputs.items():
            _validate_fingerprint(record, context=f"HCP template output {name}")
        prototype_record = outputs.get("prototypes")
        if not isinstance(prototype_record, Mapping):
            raise ValueError("HCP template prototype record is missing")
        prototype_path = Path(str(prototype_record["path"])).resolve()
        prototype = np.load(prototype_path, allow_pickle=False)
        if prototype.shape != (K, N_ROIS) or prototype.dtype != np.dtype(np.float32):
            raise ValueError(f"Invalid HCP prototype contract: {prototype_path}")
        prototypes[(smoothing, hemi)] = prototype_path
    expected_keys = {(smoothing, hemi) for smoothing in SMOOTHINGS for hemi in HEMIS}
    if set(prototypes) != expected_keys:
        raise ValueError("HCP template combination set is incomplete")
    return TemplateContract(
        root=root,
        manifest_path=manifest_path,
        manifest_fingerprint=fingerprint(manifest_path),
        roi_registry_path=registry_path,
        roi_registry_fingerprint=dict(registry_record),
        roi_registry=registry,
        prototypes=prototypes,
    )


def _surface_path(hippunfold_root: Path, subject: str, hemi: str) -> Path:
    subject_id = normalize_subject(subject)
    return (
        Path(hippunfold_root)
        / f"sub-{subject_id}"
        / "hippunfold"
        / f"sub-{subject_id}"
        / "surf"
        / (
            f"sub-{subject_id}_hemi-{hemi}_space-corobl_den-512_"
            "label-hipp_midthickness.surf.gii"
        )
    ).resolve()


def _surface_vertex_count(path: Path) -> int:
    image = nib.load(str(path))
    points = np.asarray(image.agg_data("pointset"))
    return int(points.shape[0])


def _validate_surfaces(
    *,
    hippunfold_root: Path,
    subject: str,
) -> dict[str, dict[str, object]]:
    evidence: dict[str, dict[str, object]] = {}
    for hemi in HEMIS:
        path = _surface_path(hippunfold_root, subject, hemi)
        if not path.is_file() or "_space-corobl_den-512_" not in path.name:
            raise FileNotFoundError(f"Missing consumed corobl den-512 surface: {path}")
        count = _surface_vertex_count(path)
        if count != N_VERTICES:
            raise ValueError(f"Consumed surface is not 512 vertices: {path} ({count})")
        evidence[hemi] = {
            "consumed_asset": str(path),
            "n_vertices": count,
            "evidence_source": "consumed_surface_path_and_gifti_point_count",
        }
    return evidence


def _expected_subject_inputs(
    *,
    functional: SubjectInputs,
    label_root: Path,
    hippunfold_root: Path,
    template: TemplateContract,
) -> dict[str, Path]:
    subject = functional.subject
    paths: dict[str, Path] = {
        "concat_dtseries": functional.concat_dtseries,
        "areal_lh_labels": Path(label_root) / f"sub-{subject}" / "lh_labels.txt",
        "areal_rh_labels": Path(label_root) / f"sub-{subject}" / "rh_labels.txt",
        "surface_hemi-L": _surface_path(hippunfold_root, subject, "L"),
        "surface_hemi-R": _surface_path(hippunfold_root, subject, "R"),
        "template_manifest": template.manifest_path,
        "roi_registry": template.roi_registry_path,
    }
    for run in functional.runs:
        paths[f"run-{run.run_id:02d}_volume"] = run.volume
        paths[f"run-{run.run_id:02d}_dtseries"] = run.dtseries
    for smoothing in SMOOTHINGS:
        for hemi in HEMIS:
            paths[f"template_{smoothing}_hemi-{hemi}_prototypes"] = template.prototypes[
                (smoothing, hemi)
            ]
    return {key: Path(path).resolve() for key, path in paths.items()}


def _parameters(*, wb_command: Path) -> dict[str, object]:
    return {
        "task": "rest",
        "mapping_method": "enclosing",
        "density": N_VERTICES,
        "surface_space": "corobl",
        "smoothings": list(SMOOTHINGS),
        "hemis": list(HEMIS),
        "tsnr_definition": "10000/std",
        "tsnr_threshold": TSNR_THRESHOLD,
        "fc": "vertex-to-ROI Pearson",
        "fc_transform": "arctanh clipped to +/-0.999999",
        "matching": "jointly-finite Pearson correlation to fixed HCP151 prototypes",
        "roi_order": list(range(1, N_ROIS + 1)),
        "wb_command": str(Path(wb_command).resolve()),
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _save_npy_exclusive(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)


def _write_func_gifti_exclusive(path: Path, data: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite GIFTI: {path}")
    array = np.asarray(data, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected vertices-by-time GIFTI data, got {array.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    image = GiftiImage(
        darrays=[
            GiftiDataArray(
                data=array[:, index].astype(np.float32, copy=False),
                intent="NIFTI_INTENT_TIME_SERIES",
                datatype="NIFTI_TYPE_FLOAT32",
            )
            for index in range(array.shape[1])
        ]
    )
    nib.save(image, str(path))


def _write_roi_gifti_exclusive(path: Path, valid_mask: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite ROI GIFTI: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        GiftiImage(
            darrays=[
                GiftiDataArray(
                    data=np.asarray(valid_mask, dtype=np.float32),
                    intent="NIFTI_INTENT_SHAPE",
                    datatype="NIFTI_TYPE_FLOAT32",
                )
            ]
        ),
        str(path),
    )


def load_func_gifti(path: Path, *, expected_vertices: int = N_VERTICES) -> np.ndarray:
    image = nib.load(str(path))
    arrays = [np.asarray(data_array.data, dtype=np.float32) for data_array in image.darrays]
    if not arrays:
        raise ValueError(f"GIFTI contains no data arrays: {path}")
    if len(arrays) == 1:
        data = arrays[0]
        if data.ndim == 1:
            data = data[:, None]
    elif all(array.ndim == 1 for array in arrays):
        data = np.column_stack(arrays)
    else:
        data = np.asarray(image.agg_data(), dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D GIFTI time series: {path} ({data.shape})")
    if data.shape[0] != expected_vertices and data.shape[1] == expected_vertices:
        data = data.T
    if data.shape[0] != expected_vertices:
        raise ValueError(
            f"GIFTI vertex mismatch: {path} ({data.shape[0]} != {expected_vertices})"
        )
    return np.asarray(data, dtype=np.float32)


def _run_workbench(wb_command: Path, args: Sequence[str]) -> None:
    executable = Path(wb_command).resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise FileNotFoundError(f"Workbench executable is unavailable: {executable}")
    process = subprocess.run(
        [str(executable), *[str(value) for value in args]],
        text=True,
        capture_output=True,
        env=workbench_env(),
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"wb_command failed ({process.returncode}): {process.stderr or process.stdout}"
        )


def _publish_partial(partial: Path, final: Path) -> None:
    if final.exists():
        raise FileExistsError(f"Refusing to overwrite output: {final}")
    partial.rename(final)


def _sample_runs(
    *,
    functional: SubjectInputs,
    surfaces: Mapping[str, Mapping[str, object]],
    subject_dir: Path,
    wb_command: Path,
    outputs: dict[str, Path],
) -> dict[str, np.ndarray]:
    by_hemi: dict[str, list[np.ndarray]] = {hemi: [] for hemi in HEMIS}
    for run in functional.runs:
        run_dir = subject_dir / "runs" / f"run-{run.run_id:02d}"
        for hemi in HEMIS:
            final = run_dir / (
                f"sub-{functional.subject}_task-rest_run-{run.run_id:02d}_hemi-{hemi}_"
                "space-corobl_den-512_label-hipp_bold.func.gii"
            )
            partial = final.with_name(f".{final.stem}.partial.func.gii")
            partial.parent.mkdir(parents=True, exist_ok=True)
            _run_workbench(
                wb_command,
                [
                    "-volume-to-surface-mapping",
                    run.volume,
                    surfaces[hemi]["consumed_asset"],
                    partial,
                    "-enclosing",
                ],
            )
            data = load_func_gifti(partial)
            expected_timepoints = int(nib.load(str(run.volume)).shape[3])
            if data.shape != (N_VERTICES, expected_timepoints):
                raise ValueError(
                    f"Sampled GIFTI shape mismatch for run-{run.run_id:02d} hemi-{hemi}: "
                    f"{data.shape} != {(N_VERTICES, expected_timepoints)}"
                )
            _publish_partial(partial, final)
            outputs[f"run-{run.run_id:02d}_hemi-{hemi}_gifti"] = final
            by_hemi[hemi].append(data)
    concatenated: dict[str, np.ndarray] = {}
    for hemi in HEMIS:
        data = np.concatenate(by_hemi[hemi], axis=1).astype(np.float32, copy=False)
        if data.shape[1] != int(nib.load(str(functional.concat_dtseries)).shape[0]):
            raise ValueError("Sampled surface concatenation timepoint mismatch")
        concatenated[hemi] = data
        npy = subject_dir / "surface" / "raw" / f"hemi-{hemi}_timeseries.npy"
        gifti = subject_dir / "surface" / "raw" / (
            f"sub-{functional.subject}_hemi-{hemi}_space-corobl_den-512_"
            "label-hipp_bold.func.gii"
        )
        _save_npy_exclusive(npy, data)
        _write_func_gifti_exclusive(gifti, data)
        outputs[f"raw_hemi-{hemi}_timeseries"] = npy
        outputs[f"raw_hemi-{hemi}_gifti"] = gifti
    return concatenated


def _smooth_surfaces(
    *,
    raw: Mapping[str, np.ndarray],
    surfaces: Mapping[str, Mapping[str, object]],
    subject: str,
    subject_dir: Path,
    wb_command: Path,
    outputs: dict[str, Path],
) -> tuple[dict[str, np.ndarray], dict[tuple[str, str], np.ndarray]]:
    valid_by_hemi: dict[str, np.ndarray] = {}
    smoothed: dict[tuple[str, str], np.ndarray] = {}
    for hemi in HEMIS:
        tsnr, valid = compute_tsnr(raw[hemi], threshold=TSNR_THRESHOLD)
        tsnr_path = subject_dir / "surface" / "tsnr_gate" / f"hemi-{hemi}_tsnr.npy"
        valid_path = subject_dir / "surface" / "tsnr_gate" / f"hemi-{hemi}_valid_mask.npy"
        roi_path = subject_dir / "surface" / "tsnr_gate" / f"hemi-{hemi}_valid_roi.shape.gii"
        _save_npy_exclusive(tsnr_path, tsnr)
        _save_npy_exclusive(valid_path, valid.astype(bool))
        _write_roi_gifti_exclusive(roi_path, valid)
        outputs[f"tsnr_hemi-{hemi}"] = tsnr_path
        outputs[f"valid_mask_hemi-{hemi}"] = valid_path
        outputs[f"valid_roi_hemi-{hemi}"] = roi_path
        valid_by_hemi[hemi] = valid
        raw_gifti = subject_dir / "surface" / "raw" / (
            f"sub-{subject}_hemi-{hemi}_space-corobl_den-512_label-hipp_bold.func.gii"
        )
        for smoothing in SMOOTHINGS:
            out_dir = subject_dir / "surface" / smoothing
            final = out_dir / (
                f"sub-{subject}_hemi-{hemi}_space-corobl_den-512_"
                f"label-hipp_bold_{smoothing}.func.gii"
            )
            partial = final.with_name(f".{final.stem}.partial.func.gii")
            partial.parent.mkdir(parents=True, exist_ok=True)
            _run_workbench(
                wb_command,
                [
                    "-metric-smoothing",
                    surfaces[hemi]["consumed_asset"],
                    raw_gifti,
                    float(smoothing.removesuffix("mm")),
                    partial,
                    "-fwhm",
                    "-roi",
                    roi_path,
                ],
            )
            data = load_func_gifti(partial)
            if data.shape != raw[hemi].shape:
                raise ValueError(
                    f"Smoothed GIFTI shape mismatch for {smoothing} hemi-{hemi}: "
                    f"{data.shape} != {raw[hemi].shape}"
                )
            _publish_partial(partial, final)
            npy = out_dir / f"hemi-{hemi}_timeseries.npy"
            _save_npy_exclusive(npy, data)
            outputs[f"{smoothing}_hemi-{hemi}_gifti"] = final
            outputs[f"{smoothing}_hemi-{hemi}_timeseries"] = npy
            smoothed[(smoothing, hemi)] = data
    return valid_by_hemi, smoothed


def _extract_cortex_roi_timeseries(
    *,
    concat_dtseries: Path,
    areal_labels: Mapping[str, np.ndarray],
    roi_registry: Sequence[Mapping[str, object]],
    subject_dir: Path,
    outputs: dict[str, Path],
) -> tuple[np.ndarray, np.ndarray]:
    image = nib.load(str(concat_dtseries))
    data = np.asarray(image.dataobj, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D concat dtseries data, got {data.shape}")
    axis = image.header.get_axis(1)
    data_t = data.T
    cortex: dict[str, np.ndarray] = {}
    aligned: dict[str, np.ndarray] = {}
    for hemi, structure in (("L", "CORTEX_LEFT"), ("R", "CORTEX_RIGHT")):
        vertices, hemi_data = extract_structure_data(axis, data_t, structure)
        parcel_ids = align_surface_labels_to_dt_vertices(
            areal_labels[hemi],
            vertices,
            hemi=hemi,
        )
        cortex_tsnr, cortex_valid = compute_tsnr(hemi_data, threshold=TSNR_THRESHOLD)
        parcel_ids = parcel_ids.copy()
        parcel_ids[~cortex_valid] = 0
        cortex[hemi] = np.asarray(hemi_data, dtype=np.float32)
        aligned[hemi] = parcel_ids
        tsnr_path = subject_dir / "cortex" / f"hemi-{hemi}_tsnr.npy"
        valid_path = subject_dir / "cortex" / f"hemi-{hemi}_valid_mask.npy"
        _save_npy_exclusive(tsnr_path, cortex_tsnr)
        _save_npy_exclusive(valid_path, cortex_valid.astype(bool))
        outputs[f"cortex_tsnr_hemi-{hemi}"] = tsnr_path
        outputs[f"cortex_valid_mask_hemi-{hemi}"] = valid_path
    roi_ids = [int(row["roi_id"]) for row in roi_registry]
    roi_timeseries, counts = extract_fixed_roi_timeseries(
        cortex_by_hemi=cortex,
        aligned_labels_by_hemi=aligned,
        roi_ids=roi_ids,
    )
    roi_path = subject_dir / "cortex" / "kong2022_roi_timeseries.npy"
    count_path = subject_dir / "cortex" / "kong2022_roi_vertex_counts.npy"
    _save_npy_exclusive(roi_path, roi_timeseries)
    _save_npy_exclusive(count_path, counts)
    outputs["cortex_roi_timeseries"] = roi_path
    outputs["cortex_roi_vertex_counts"] = count_path
    return roi_timeseries, counts


def _write_label_gifti_exclusive(path: Path, labels: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite label GIFTI: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(make_named_label_gifti(labels, LABEL_ITEMS), str(path))


def _match_combinations(
    *,
    subject: str,
    subject_dir: Path,
    smoothed: Mapping[tuple[str, str], np.ndarray],
    valid_by_hemi: Mapping[str, np.ndarray],
    roi_timeseries: np.ndarray,
    roi_counts: np.ndarray,
    template: TemplateContract,
    surfaces: Mapping[str, Mapping[str, object]],
    outputs: dict[str, Path],
) -> list[dict[str, object]]:
    combinations: list[dict[str, object]] = []
    missing_rois = roi_counts == 0
    for smoothing in SMOOTHINGS:
        for hemi in HEMIS:
            combo_dir = subject_dir / "matching" / smoothing / f"hemi-{hemi}"
            pearson = corrcoef_rows(smoothed[(smoothing, hemi)], roi_timeseries)
            pearson[:, missing_rois] = np.nan
            pearson[~valid_by_hemi[hemi], :] = np.nan
            fisherz = fisher_z_transform_fc(pearson)
            prototypes = np.load(template.prototypes[(smoothing, hemi)], allow_pickle=False)
            labels, scores, margin = match_vertices_to_prototypes(
                fisherz,
                valid_by_hemi[hemi],
                prototypes,
            )
            summary = validate_combo_arrays(
                labels=labels,
                scores=scores,
                margin=margin,
                valid_mask=valid_by_hemi[hemi],
            )
            paths = {
                "pearson_fc": combo_dir / "vertex_to_roi_pearson_fc.npy",
                "fisherz_fc": combo_dir / "vertex_to_roi_fisherz_fc.npy",
                "labels": combo_dir / "labels.npy",
                "scores": combo_dir / "template_correlations.npy",
                "margin": combo_dir / "correlation_margin.npy",
                "label_gifti": combo_dir / (
                    f"sub-{subject}_hemi-{hemi}_space-corobl_den-512_"
                    f"label-hcp151_fixed_template_k3_{smoothing}.label.gii"
                ),
            }
            _save_npy_exclusive(paths["pearson_fc"], pearson.astype(np.float32))
            _save_npy_exclusive(paths["fisherz_fc"], fisherz.astype(np.float32))
            _save_npy_exclusive(paths["labels"], labels.astype(np.int32))
            _save_npy_exclusive(paths["scores"], scores.astype(np.float32))
            _save_npy_exclusive(paths["margin"], margin.astype(np.float32))
            _write_label_gifti_exclusive(paths["label_gifti"], labels)
            for name, path in paths.items():
                outputs[f"{smoothing}_hemi-{hemi}_{name}"] = path
            combo_manifest_path = combo_dir / "combo_manifest.json"
            combo_manifest = {
                "status": "ok",
                "subject": f"sub-{subject}",
                "smoothing": smoothing,
                "hemi": hemi,
                "label_semantics": {
                    "1": "Anterior/VAN",
                    "2": "Body/SMN",
                    "3": "Posterior/DMN",
                },
                "template_prototypes": fingerprint(template.prototypes[(smoothing, hemi)]),
                "roi_registry": dict(template.roi_registry_fingerprint),
                "density_evidence": dict(surfaces[hemi]),
                "missing_roi_count": int(np.sum(missing_rois)),
                **summary,
                "outputs": {name: artifact_record(path) for name, path in paths.items()},
            }
            _write_json(combo_manifest_path, combo_manifest)
            outputs[f"{smoothing}_hemi-{hemi}_combo_manifest"] = combo_manifest_path
            combinations.append(
                {
                    "status": "ok",
                    "smoothing": smoothing,
                    "hemi": hemi,
                    "manifest": str(combo_manifest_path.resolve()),
                    **summary,
                    "density_evidence": dict(surfaces[hemi]),
                }
            )
    return combinations


def preflight_subject(
    *,
    subject: str,
    volume_root: Path,
    dtseries_root: Path,
    concat_dtseries_root: Path,
    label_root: Path,
    hippunfold_root: Path,
) -> tuple[SubjectInputs, dict[str, object], dict[str, dict[str, object]]]:
    functional = resolve_subject_runs(
        subject=subject,
        volume_root=volume_root,
        dtseries_root=dtseries_root,
        concat_dtseries_root=concat_dtseries_root,
    )
    header_evidence = validate_functional_headers(functional)
    load_areal_labels(label_root=label_root, subject=subject)
    surfaces = _validate_surfaces(hippunfold_root=hippunfold_root, subject=subject)
    return functional, header_evidence, surfaces


def _run_subject(
    *,
    subject: str,
    volume_root: Path,
    dtseries_root: Path,
    concat_dtseries_root: Path,
    label_root: Path,
    hippunfold_root: Path,
    out_root: Path,
    wb_command: Path,
    template: TemplateContract,
    resume: bool,
) -> dict[str, object]:
    functional, header_evidence, surfaces = preflight_subject(
        subject=subject,
        volume_root=volume_root,
        dtseries_root=dtseries_root,
        concat_dtseries_root=concat_dtseries_root,
        label_root=label_root,
        hippunfold_root=hippunfold_root,
    )
    subject_dir = Path(out_root) / "subjects" / f"sub-{subject}"
    expected_inputs = _expected_subject_inputs(
        functional=functional,
        label_root=label_root,
        hippunfold_root=hippunfold_root,
        template=template,
    )
    parameters = _parameters(wb_command=wb_command)
    final_manifest = subject_dir / "subject_manifest.json"
    if final_manifest.is_file():
        if not resume:
            raise FileExistsError(
                f"Subject output exists; use --resume for exact reuse: {final_manifest}"
            )
        reused = validate_reusable_subject(
            subject_dir=subject_dir,
            expected_inputs=expected_inputs,
            expected_parameters=parameters,
        )
        return {
            "subject": f"sub-{subject}",
            "status": "ok",
            "reused": True,
            "manifest": str(final_manifest.resolve()),
            "combinations": reused.get("combinations", []),
        }
    attempt_history: list[dict[str, object]] = []
    if subject_dir.exists() and any(subject_dir.iterdir()):
        if not resume:
            raise FileExistsError(
                "Partial/nonvalidated subject output is retained and requires "
                f"--resume: {subject_dir}"
            )
        attempt_history.append(
            prepare_empty_failed_subject_resume(
                subject_dir=subject_dir,
                expected_inputs=expected_inputs,
                expected_parameters=parameters,
            )
        )
    subject_dir.mkdir(parents=True, exist_ok=True)
    partial_manifest = subject_dir / "subject_manifest.partial.json"
    input_fingerprints = {
        key: fingerprint(path) for key, path in sorted(expected_inputs.items())
    }
    _write_json(
        partial_manifest,
        {
            "status": "in_progress",
            "subject": f"sub-{subject}",
            "attempt_history": attempt_history,
            "parameters": parameters,
            "inputs": input_fingerprints,
            "header_evidence": header_evidence,
            "density_evidence": surfaces,
        },
    )
    outputs: dict[str, Path] = {}
    try:
        areal_labels = load_areal_labels(label_root=label_root, subject=subject)
        raw = _sample_runs(
            functional=functional,
            surfaces=surfaces,
            subject_dir=subject_dir,
            wb_command=wb_command,
            outputs=outputs,
        )
        valid_by_hemi, smoothed = _smooth_surfaces(
            raw=raw,
            surfaces=surfaces,
            subject=subject,
            subject_dir=subject_dir,
            wb_command=wb_command,
            outputs=outputs,
        )
        roi_timeseries, roi_counts = _extract_cortex_roi_timeseries(
            concat_dtseries=functional.concat_dtseries,
            areal_labels=areal_labels,
            roi_registry=template.roi_registry,
            subject_dir=subject_dir,
            outputs=outputs,
        )
        combinations = _match_combinations(
            subject=subject,
            subject_dir=subject_dir,
            smoothed=smoothed,
            valid_by_hemi=valid_by_hemi,
            roi_timeseries=roi_timeseries,
            roi_counts=roi_counts,
            template=template,
            surfaces=surfaces,
            outputs=outputs,
        )
        manifest = build_subject_manifest(
            subject=subject,
            inputs=expected_inputs,
            parameters=parameters,
            outputs=outputs,
            combinations=combinations,
        )
        manifest.update(
            {
                "attempt_history": attempt_history,
                "header_evidence": header_evidence,
                "density_evidence": surfaces,
                "template_manifest_fingerprint": dict(template.manifest_fingerprint),
                "roi_registry_fingerprint": dict(template.roi_registry_fingerprint),
            }
        )
        _write_json(partial_manifest, manifest)
        _publish_partial(partial_manifest, final_manifest)
        return {
            "subject": f"sub-{subject}",
            "status": "ok",
            "reused": False,
            "manifest": str(final_manifest.resolve()),
            "combinations": combinations,
        }
    except Exception as exc:
        _write_json(
            partial_manifest,
            {
                "status": "failed",
                "subject": f"sub-{subject}",
                "attempt_history": attempt_history,
                "parameters": parameters,
                "inputs": input_fingerprints,
                "header_evidence": header_evidence,
                "density_evidence": surfaces,
                "error": f"{type(exc).__name__}: {exc}",
                "retained_outputs": {
                    key: str(path.resolve()) for key, path in outputs.items()
                },
            },
        )
        raise


def _run_dir(out_root: Path, run_id: str) -> Path:
    if (
        not run_id
        or len(run_id) > 128
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id) is None
    ):
        raise ValueError(f"Unsafe --run-id: {run_id!r}")
    return Path(out_root) / "runs" / run_id


def _preflight_batch(
    *,
    subjects: Sequence[str],
    volume_root: Path,
    dtseries_root: Path,
    concat_dtseries_root: Path,
    label_root: Path,
    hippunfold_root: Path,
    template: TemplateContract,
    wb_command: Path,
) -> dict[str, object]:
    if not Path(wb_command).is_file() or not os.access(Path(wb_command), os.X_OK):
        raise FileNotFoundError(f"Workbench executable is unavailable: {wb_command}")
    items: list[dict[str, object]] = []
    for subject in subjects:
        functional, headers, surfaces = preflight_subject(
            subject=subject,
            volume_root=volume_root,
            dtseries_root=dtseries_root,
            concat_dtseries_root=concat_dtseries_root,
            label_root=label_root,
            hippunfold_root=hippunfold_root,
        )
        items.append(
            {
                "subject": f"sub-{subject}",
                "status": "ok",
                "run_ids": [run.run_id for run in functional.runs],
                "header_evidence": headers,
                "density_evidence": surfaces,
            }
        )
    return {
        "status": "ok",
        "mode": "preflight",
        "subject_count": len(subjects),
        "subjects": [f"sub-{subject}" for subject in subjects],
        "template_manifest": dict(template.manifest_fingerprint),
        "roi_registry": dict(template.roi_registry_fingerprint),
        "items": items,
    }


def validate_run_resume_contract(
    recorded: Mapping[str, object],
    expected: Mapping[str, object],
) -> None:
    immutable_keys = (
        "workflow",
        "run_id",
        "scope",
        "subjects",
        "subject_count",
        "canonical_roster_count",
        "template_manifest",
        "roi_registry",
        "parameters",
        "path_roots",
    )
    for key in immutable_keys:
        if recorded.get(key) != expected.get(key):
            raise ValueError(f"Run resume contract mismatch for {key}")


def _run_batch(
    *,
    subjects: Sequence[str],
    canonical_roster: Sequence[str],
    run_id: str,
    jobs: int,
    resume: bool,
    volume_root: Path,
    dtseries_root: Path,
    concat_dtseries_root: Path,
    label_root: Path,
    hippunfold_root: Path,
    out_root: Path,
    template: TemplateContract,
    wb_command: Path,
) -> dict[str, object]:
    _preflight_batch(
        subjects=subjects,
        volume_root=volume_root,
        dtseries_root=dtseries_root,
        concat_dtseries_root=concat_dtseries_root,
        label_root=label_root,
        hippunfold_root=hippunfold_root,
        template=template,
        wb_command=wb_command,
    )
    run_dir = _run_dir(out_root, run_id)
    final_manifest = run_dir / "run_manifest.json"
    partial_manifest = run_dir / "run_manifest.partial.json"
    if run_dir.exists() and any(run_dir.iterdir()) and not resume:
        raise FileExistsError(f"Run directory is nonempty; use --resume: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    base = {
        "workflow": "jj_tms_hcp151_fixed_template_matching",
        "run_id": run_id,
        "scope": "full" if list(subjects) == list(canonical_roster) else "subset",
        "subjects": [f"sub-{subject}" for subject in subjects],
        "subject_count": len(subjects),
        "canonical_roster_count": len(canonical_roster),
        "jobs": int(jobs),
        "resume": bool(resume),
        "template_manifest": dict(template.manifest_fingerprint),
        "roi_registry": dict(template.roi_registry_fingerprint),
        "parameters": _parameters(wb_command=wb_command),
        "path_roots": {
            "volume_root": str(Path(volume_root).resolve()),
            "dtseries_root": str(Path(dtseries_root).resolve()),
            "concat_dtseries_root": str(Path(concat_dtseries_root).resolve()),
            "label_root": str(Path(label_root).resolve()),
            "hippunfold_root": str(Path(hippunfold_root).resolve()),
            "template_root": str(template.root),
            "out_root": str(Path(out_root).resolve()),
        },
    }
    if partial_manifest.exists():
        raise FileExistsError(
            f"Retained run partial manifest will not be overwritten: {partial_manifest}"
        )
    if final_manifest.exists():
        recorded_run = json.loads(final_manifest.read_text(encoding="utf-8"))
        if not isinstance(recorded_run, Mapping):
            raise ValueError(f"Invalid existing run manifest: {final_manifest}")
        validate_run_resume_contract(recorded_run, base)
    elif resume and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Resume run directory lacks a validated run manifest: {run_dir}"
        )
    _write_json(partial_manifest, {**base, "status": "in_progress", "items": []})
    items: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=min(int(jobs), len(subjects))) as pool:
        futures = {
            pool.submit(
                _run_subject,
                subject=subject,
                volume_root=volume_root,
                dtseries_root=dtseries_root,
                concat_dtseries_root=concat_dtseries_root,
                label_root=label_root,
                hippunfold_root=hippunfold_root,
                out_root=out_root,
                wb_command=wb_command,
                template=template,
                resume=resume,
            ): subject
            for subject in subjects
        }
        for future in as_completed(futures):
            subject = futures[future]
            try:
                items.append(future.result())
            except Exception as exc:
                items.append(
                    {
                        "subject": f"sub-{subject}",
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    order = {subject: index for index, subject in enumerate(subjects)}
    items.sort(key=lambda row: order[str(row["subject"]).removeprefix("sub-")])
    complete_count = sum(row.get("status") == "ok" for row in items)
    status = "complete" if complete_count == len(subjects) else "failed"
    if list(subjects) == list(canonical_roster) and len(subjects) != EXPECTED_SUBJECT_COUNT:
        status = "failed"
    payload = {
        **base,
        "status": status,
        "complete_count": complete_count,
        "failed_count": len(subjects) - complete_count,
        "items": items,
    }
    _write_json(partial_manifest, payload)
    if final_manifest.exists():
        os.replace(partial_manifest, final_manifest)
    else:
        _publish_partial(partial_manifest, final_manifest)
    return payload


def _validate_subject_for_run(
    *,
    subject: str,
    roots: Mapping[str, object],
    out_root: Path,
    template: TemplateContract,
    parameters: Mapping[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    functional = resolve_subject_runs(
        subject=subject,
        volume_root=Path(str(roots["volume_root"])),
        dtseries_root=Path(str(roots["dtseries_root"])),
        concat_dtseries_root=Path(str(roots["concat_dtseries_root"])),
    )
    validate_functional_headers(functional)
    expected_inputs = _expected_subject_inputs(
        functional=functional,
        label_root=Path(str(roots["label_root"])),
        hippunfold_root=Path(str(roots["hippunfold_root"])),
        template=template,
    )
    subject_dir = Path(out_root) / "subjects" / f"sub-{subject}"
    manifest = validate_reusable_subject(
        subject_dir=subject_dir,
        expected_inputs=expected_inputs,
        expected_parameters=parameters,
    )
    rows: list[dict[str, object]] = []
    combinations = manifest.get("combinations")
    if not isinstance(combinations, list) or len(combinations) != 4:
        raise ValueError(f"Expected four subject combinations: sub-{subject}")
    for combo in combinations:
        if not isinstance(combo, Mapping) or combo.get("status") != "ok":
            raise ValueError(f"Invalid combo manifest row: sub-{subject}")
        smoothing = str(combo.get("smoothing"))
        hemi = str(combo.get("hemi"))
        manifest_path = Path(str(combo.get("manifest")))
        combo_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        outputs = combo_payload["outputs"]
        labels = np.load(outputs["labels"]["path"], allow_pickle=False)
        scores = np.load(outputs["scores"]["path"], allow_pickle=False)
        margin = np.load(outputs["margin"]["path"], allow_pickle=False)
        valid = np.load(
            subject_dir / "surface" / "tsnr_gate" / f"hemi-{hemi}_valid_mask.npy",
            allow_pickle=False,
        ).astype(bool)
        summary = validate_combo_arrays(
            labels=labels,
            scores=scores,
            margin=margin,
            valid_mask=valid,
        )
        surface = _surface_path(Path(str(roots["hippunfold_root"])), subject, hemi)
        density = combo_payload.get("density_evidence")
        if not isinstance(density, Mapping) or Path(str(density.get("consumed_asset"))).resolve() != surface:
            raise ValueError(f"Consumed-surface density evidence drift: sub-{subject} {smoothing} {hemi}")
        rows.append(
            {
                "subject": f"sub-{subject}",
                "smoothing": smoothing,
                "hemi": hemi,
                "status": "ok",
                "label_counts": summary["label_counts"],
                "margin_summary": summary["margin_summary"],
                "density_evidence": dict(density),
                "labels_sha256": outputs["labels"]["sha256"],
            }
        )
    return manifest, rows


def _write_validation_tsv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite validation table: {path}")
    fields = (
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
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            counts = row["label_counts"]
            margin = row["margin_summary"]
            density = row["density_evidence"]
            writer.writerow(
                {
                    "subject": row["subject"],
                    "smoothing": row["smoothing"],
                    "hemi": row["hemi"],
                    "status": row["status"],
                    "label_1_count": counts["1"],
                    "label_2_count": counts["2"],
                    "label_3_count": counts["3"],
                    "margin_min": margin["min"],
                    "margin_median": margin["median"],
                    "margin_mean": margin["mean"],
                    "margin_max": margin["max"],
                    "labels_sha256": row["labels_sha256"],
                    "consumed_surface": density["consumed_asset"],
                }
            )


def _validate_run(
    *,
    run_id: str,
    out_root: Path,
    template: TemplateContract,
    selected_subjects: Sequence[str] | None,
    canonical_roster: Sequence[str],
) -> dict[str, object]:
    run_dir = _run_dir(out_root, run_id)
    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("status") != "complete":
        raise ValueError(f"Run is not complete: {run_manifest_path}")
    subjects = [str(value).removeprefix("sub-") for value in run_manifest.get("subjects", [])]
    if selected_subjects and list(selected_subjects) != subjects:
        raise ValueError("Validation --subjects differ from the run manifest")
    if run_manifest.get("scope") == "full":
        if subjects != list(canonical_roster) or len(subjects) != EXPECTED_SUBJECT_COUNT:
            raise ValueError("Full-run manifest does not contain the fixed 70-subject roster")
    if run_manifest.get("template_manifest") != dict(template.manifest_fingerprint):
        raise ValueError("Run template manifest fingerprint drift")
    if run_manifest.get("roi_registry") != dict(template.roi_registry_fingerprint):
        raise ValueError("Run ROI registry fingerprint drift")
    roots = run_manifest.get("path_roots")
    parameters = run_manifest.get("parameters")
    if not isinstance(roots, Mapping) or not isinstance(parameters, Mapping):
        raise ValueError("Run path/parameter metadata is missing")
    rows: list[dict[str, object]] = []
    subject_manifests: list[str] = []
    for subject in subjects:
        manifest, subject_rows = _validate_subject_for_run(
            subject=subject,
            roots=roots,
            out_root=out_root,
            template=template,
            parameters=parameters,
        )
        rows.extend(subject_rows)
        subject_manifests.append(
            str((Path(out_root) / "subjects" / f"sub-{subject}" / "subject_manifest.json").resolve())
        )
    if len(rows) != len(subjects) * 4:
        raise ValueError("Validation did not produce four combinations per subject")
    table_path = run_dir / "validation_items.tsv"
    manifest_path = run_dir / "validation_manifest.json"
    _write_validation_tsv(table_path, rows)
    payload = {
        "status": "complete",
        "workflow": "validate_jj_tms_hcp151_fixed_template_matching",
        "run_id": run_id,
        "scope": run_manifest.get("scope"),
        "subject_count": len(subjects),
        "complete_subject_count": len(subjects),
        "combination_count": len(rows),
        "template_manifest": dict(template.manifest_fingerprint),
        "roi_registry": dict(template.roi_registry_fingerprint),
        "run_manifest": fingerprint(run_manifest_path),
        "subject_manifests": subject_manifests,
        "validation_items": fingerprint(table_path),
        "items": rows,
    }
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite validation manifest: {manifest_path}")
    _write_json(manifest_path, payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply immutable HCP151 prototypes to fixed-roster JJ_TMS rest data."
    )
    parser.add_argument("--mode", choices=("preflight", "run", "validate"), required=True)
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--subjects-manifest", type=Path, default=DEFAULT_SUBJECTS_MANIFEST)
    parser.add_argument("--run-id")
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--volume-root", type=Path, default=DEFAULT_VOLUME_ROOT)
    parser.add_argument("--dtseries-root", type=Path, default=DEFAULT_DTSERIES_ROOT)
    parser.add_argument(
        "--concat-dtseries-root",
        type=Path,
        default=DEFAULT_CONCAT_DTSERIES_ROOT,
    )
    parser.add_argument("--areal-label-root", type=Path, default=DEFAULT_AREAL_LABEL_ROOT)
    parser.add_argument("--hippunfold-root", type=Path, default=DEFAULT_HIPPUNFOLD_ROOT)
    parser.add_argument("--template-root", type=Path, default=DEFAULT_TEMPLATE_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--wb-command", type=Path, default=DEFAULT_WB_COMMAND)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not 1 <= int(args.jobs) <= MAX_JOBS:
        raise ValueError(f"--jobs must be in [1, {MAX_JOBS}]")
    subjects, canonical = select_subjects(
        subjects_manifest=args.subjects_manifest,
        explicit_subjects=args.subjects,
    )
    template = _load_template_contract(args.template_root)
    if args.mode == "preflight":
        payload = _preflight_batch(
            subjects=subjects,
            volume_root=args.volume_root,
            dtseries_root=args.dtseries_root,
            concat_dtseries_root=args.concat_dtseries_root,
            label_root=args.areal_label_root,
            hippunfold_root=args.hippunfold_root,
            template=template,
            wb_command=args.wb_command,
        )
    elif args.mode == "run":
        if not args.run_id:
            raise ValueError("--run-id is required for --mode run")
        payload = _run_batch(
            subjects=subjects,
            canonical_roster=canonical,
            run_id=args.run_id,
            jobs=int(args.jobs),
            resume=bool(args.resume),
            volume_root=args.volume_root,
            dtseries_root=args.dtseries_root,
            concat_dtseries_root=args.concat_dtseries_root,
            label_root=args.areal_label_root,
            hippunfold_root=args.hippunfold_root,
            out_root=args.out_root,
            template=template,
            wb_command=args.wb_command,
        )
    else:
        if not args.run_id:
            raise ValueError("--run-id is required for --mode validate")
        payload = _validate_run(
            run_id=args.run_id,
            out_root=args.out_root,
            template=template,
            selected_subjects=(
                [normalize_subject(value) for value in args.subjects]
                if args.subjects
                else None
            ),
            canonical_roster=canonical,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("status") in {"ok", "complete"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

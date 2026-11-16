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
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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

from lib.extract_cortex_kong2022_route_a_roi_timeseries import (
    align_surface_labels_to_dt_vertices,
)
from lib.extract_cortex_merged_network_timeseries import extract_structure_data
from lib.fc_math import corrcoef_rows, fisher_z_transform_fc
from lib.gifti_labels import make_named_label_gifti
from lib.template_matching import match_vertices_to_prototypes
from lib.workbench import workbench_env
from lib.task_quality import cortex_signal_validity, rest_signal_validity, CORTEX_VALIDITY_POLICY, SERIES_VALIDITY_POLICY
from lib.gifti_contracts import (
    gifti_logical_shape_dtype,
    load_gifti_timeseries,
    validate_gifti_logical_contract,
    validate_gifti_pointset,
)
from lib.surface_coordinates import MNI_SPACE, dataset_registration, registered_sampling_surfaces
from lib.surface_sampling import sample_volume_to_surface


SMOOTHINGS = ("2mm", "4mm")
HEMIS = ("L", "R")
N_VERTICES = 512
N_SURFACE_VERTICES_32K = 32492
ARTIFACT_CONTRACT_SCHEMA_VERSION = 2
LEGACY_PUBLISHED_RUN_IDS = frozenset(
    {
        "jj_tms_hcp_smoke_20260809",
        "jj_tms_hcp_full_20260809",
    }
)
N_ROIS = 400
K = 3
MAX_JOBS = 10
DEFAULT_JOBS = 2
TWO_RUN_SUBJECT = "1016"
TEMPLATE_ID = "hcp_routea_full_k3_fisherz"
SMOKE_RUN_ID = "jj_tms_hcp_smoke_20260809"
SMOKE_SUBJECTS = ("1001", "1002", "1003")
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
    / "hcp_fixed_template_matching"
)
DEFAULT_SUBJECTS_MANIFEST = (
    REPO_ROOT / "manifests" / "jj_tms_hcp_fixed_template_subjects.txt"
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
    (2, "Middle/SMN", (99, 150, 194, 255)),
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
    concat_dtseries: Path | None


@dataclass(frozen=True)
class TemplateContract:
    root: Path
    manifest_path: Path
    manifest_fingerprint: Mapping[str, object]
    roi_registry_path: Path
    roi_registry_fingerprint: Mapping[str, object]
    roi_registry: tuple[Mapping[str, object], ...]
    prototypes: Mapping[tuple[str, str], Path]
    subjects: tuple[str, ...]


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
    if not canonical:
        raise ValueError("Canonical JJ_TMS roster contains no subjects")
    requested_manifest = read_subject_manifest(subjects_manifest)
    if requested_manifest != canonical:
        raise ValueError(f"Requested roster manifest differs from the fixed canonical roster")
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
    concat_dtseries_root: Path | None = None,
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
    detected_volumes = sorted(volumes)
    detected_dtseries = sorted(dtseries)
    if detected_volumes != detected_dtseries:
        raise ValueError(
            f"Mismatched volume and dtseries run IDs for sub-{subject_id}: "
            f"volume={detected_volumes} dtseries={detected_dtseries}"
        )
    if subject_id == TWO_RUN_SUBJECT or subject_id == "1016":
        expected_ids = [1, 2]
    elif subject_id == "1027":
        if detected_volumes == [2, 3]:
            expected_ids = [2, 3]
        elif detected_volumes == [1, 2, 3]:
            expected_ids = [1, 2, 3]
        else:
            expected_ids = [2, 3]
    else:
        expected_ids = [1, 2, 3]
    if detected_volumes != expected_ids:
        raise ValueError(
            f"Unexpected task-rest run IDs for sub-{subject_id}: "
            f"volume={detected_volumes} dtseries={detected_dtseries} "
            f"expected={expected_ids}"
        )
    concat: Path | None = None
    if concat_dtseries_root is not None and str(concat_dtseries_root).strip().lower() not in ("none", "null", ""):
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
    total = int(sum(run_lengths))
    if inputs.concat_dtseries is not None:
        concat = nib.load(str(inputs.concat_dtseries))
        if len(concat.shape) != 2:
            raise ValueError(f"Expected 2D concat dtseries: {inputs.concat_dtseries}")
        if int(concat.shape[0]) != total:
            raise ValueError(
                f"Concat timepoint mismatch for sub-{inputs.subject}: "
                f"concat={concat.shape[0]} run_sum={total}"
            )
        if int(concat.shape[1]) != int(grayordinates or -1):
            raise ValueError("Concat dtseries grayordinate grid mismatch")
        axis = concat.header.get_axis(1)
        concat_timepoints = int(concat.shape[0])
    else:
        sample_dtseries = nib.load(str(inputs.runs[0].dtseries))
        axis = sample_dtseries.header.get_axis(1)
        concat_timepoints = total

    cortex_vertex_counts: dict[str, int] = {}
    for name, _slice, subaxis in axis.iter_structures():
        if name.endswith("CORTEX_LEFT"):
            cortex_vertex_counts["L"] = int(np.asarray(subaxis.vertex).size)
        elif name.endswith("CORTEX_RIGHT"):
            cortex_vertex_counts["R"] = int(np.asarray(subaxis.vertex).size)
    if set(cortex_vertex_counts) != set(HEMIS):
        raise ValueError("Dtseries must contain left and right cortex structures")
    return {
        "subject": inputs.subject,
        "run_lengths": run_lengths,
        "total_timepoints": total,
        "volume_grid": list(spatial_shape or ()),
        "volume_affine": np.asarray(affine).tolist(),
        "grayordinates": int(grayordinates or 0),
        "concat_timepoints": concat_timepoints,
        "cortex_vertex_counts": cortex_vertex_counts,
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
    valid_scores = score_array[valid]
    if not np.all(np.isfinite(valid_scores)) or np.any(
        (valid_scores < -1.0) | (valid_scores > 1.0)
    ):
        raise ValueError(
            "Every valid vertex must have three finite template correlations in [-1, 1]"
        )
    valid_margin = margin_array[valid]
    if not np.all(np.isfinite(valid_margin)) or np.any(
        (valid_margin < 0.0) | (valid_margin > 2.0)
    ):
        raise ValueError(
            "Every valid vertex must have a finite correlation margin in [0, 2]"
        )
    counts = {
        str(code): int(np.sum(label_array[valid] == code)) for code in range(1, K + 1)
    }
    return {
        "valid_vertices": int(np.sum(valid)),
        "label_counts": counts,
        "margin_summary": {
            "min": float(np.min(valid_margin)),
            "median": float(np.median(valid_margin)),
            "mean": float(np.mean(valid_margin)),
            "max": float(np.max(valid_margin)),
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
    suffix = Path(path).suffix
    if suffix == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        record.update(shape=list(array.shape), dtype=str(array.dtype))
    elif suffix == ".gii":
        shape, dtype = gifti_logical_shape_dtype(path)
        record.update(shape=list(shape), dtype=str(dtype))
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
        "workflow": "jj_tms_hcp_fixed_template_matching",
        "artifact_contract_schema_version": ARTIFACT_CONTRACT_SCHEMA_VERSION,
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


def _cortex_counts_from_functional(functional: SubjectInputs) -> dict[str, int]:
    sample_path = functional.concat_dtseries if functional.concat_dtseries is not None else functional.runs[0].dtseries
    image = nib.load(str(sample_path))
    counts: dict[str, int] = {}
    for name, _slice, subaxis in image.header.get_axis(1).iter_structures():
        if name.endswith("CORTEX_LEFT"):
            counts["L"] = int(np.asarray(subaxis.vertex).size)
        elif name.endswith("CORTEX_RIGHT"):
            counts["R"] = int(np.asarray(subaxis.vertex).size)
    if set(counts) != set(HEMIS):
        raise ValueError(f"Missing cortex structures in {sample_path}")
    return counts


def canonical_output_contracts(
    functional: SubjectInputs | None,
) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    contracts: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
    fixed: dict[str, tuple[tuple[int, ...], np.dtype]] = {
        "valid_mask_hemi-L": ((N_VERTICES,), np.dtype(bool)),
        "valid_mask_hemi-R": ((N_VERTICES,), np.dtype(bool)),
        "cortex_roi_vertex_counts": ((N_ROIS,), np.dtype(np.int32)),
    }
    if functional is None:
        for key in ("labels", "margin"):
            fixed[key] = ((N_VERTICES,), np.dtype(np.int32 if key == "labels" else np.float32))
        fixed["scores"] = ((N_VERTICES, K), np.dtype(np.float32))
        for key, contract in fixed.items():
            contracts[key] = contract
        return contracts
    total_timepoints = int(
        sum(int(nib.load(str(run.volume)).shape[3]) for run in functional.runs)
    )
    cortex_counts = _cortex_counts_from_functional(functional)
    for run in functional.runs:
        for hemi in HEMIS:
            contracts[f"run-{run.run_id:02d}_hemi-{hemi}_gifti"] = (
                (N_VERTICES, int(nib.load(str(run.volume)).shape[3])),
                np.dtype(np.float32),
            )
    for hemi in HEMIS:
        contracts[f"raw_hemi-{hemi}_timeseries"] = (
            (N_VERTICES, total_timepoints),
            np.dtype(np.float32),
        )
        contracts[f"raw_hemi-{hemi}_gifti"] = (
            (N_VERTICES, total_timepoints),
            np.dtype(np.float32),
        )
        contracts[f"valid_mask_hemi-{hemi}"] = ((N_VERTICES,), np.dtype(bool))
        contracts[f"valid_roi_hemi-{hemi}"] = (
            (N_VERTICES,),
            np.dtype(np.float32),
        )
        contracts[f"cortex_valid_mask_hemi-{hemi}"] = (
            (cortex_counts[hemi],),
            np.dtype(bool),
        )
        for smoothing in SMOOTHINGS:
            prefix = f"{smoothing}_hemi-{hemi}"
            contracts[f"{prefix}_timeseries"] = (
                (N_VERTICES, total_timepoints),
                np.dtype(np.float32),
            )
            contracts[f"{prefix}_gifti"] = (
                (N_VERTICES, total_timepoints),
                np.dtype(np.float32),
            )
            contracts[f"{prefix}_pearson_fc"] = (
                (N_VERTICES, N_ROIS),
                np.dtype(np.float32),
            )
            contracts[f"{prefix}_fisherz_fc"] = (
                (N_VERTICES, N_ROIS),
                np.dtype(np.float32),
            )
            contracts[f"{prefix}_labels"] = ((N_VERTICES,), np.dtype(np.int32))
            contracts[f"{prefix}_label_gifti"] = (
                (N_VERTICES,),
                np.dtype(np.int32),
            )
            contracts[f"{prefix}_scores"] = ((N_VERTICES, K), np.dtype(np.float32))
            contracts[f"{prefix}_margin"] = ((N_VERTICES,), np.dtype(np.float32))
    contracts["cortex_roi_timeseries"] = (
        (N_ROIS, total_timepoints),
        np.dtype(np.float32),
    )
    contracts["cortex_roi_vertex_counts"] = ((N_ROIS,), np.dtype(np.int32))
    return contracts


def _validate_output_record(
    *,
    key: str,
    recorded: object,
    expected_path: Path,
    contract: tuple[tuple[int, ...], np.dtype] | None,
    legacy_metadata: bool = False,
) -> None:
    if not isinstance(recorded, Mapping) or not isinstance(recorded.get("path"), str):
        raise ValueError(f"Resume output metadata is invalid for {key}")
    output_path = Path(str(recorded["path"]))
    if output_path.resolve() != Path(expected_path).resolve():
        raise ValueError(f"Resume output path mismatch for {key}")
    if output_path.suffix in {".npy", ".gii"}:
        has_metadata = "shape" in recorded and "dtype" in recorded
        if not has_metadata and not legacy_metadata:
            raise ValueError(f"Resume output shape/dtype metadata is missing for {key}")
        actual = artifact_record(output_path)
        if has_metadata:
            if recorded.get("shape") != actual.get("shape"):
                raise ValueError(f"Resume output shape metadata drift for {key}")
            if recorded.get("dtype") != actual.get("dtype"):
                raise ValueError(f"Resume output dtype metadata drift for {key}")
        if contract is not None:
            shape, dtype = contract
            if actual.get("shape") != list(shape):
                raise ValueError(
                    f"Resume output canonical combination shape mismatch for {key}: "
                    f"{actual.get('shape')} != {list(shape)}"
                )
            if actual.get("dtype") != str(dtype):
                raise ValueError(
                    f"Resume output canonical combination dtype mismatch for {key}: "
                    f"{actual.get('dtype')} != {dtype}"
                )
    _validate_fingerprint(recorded, context=f"resume output {key}")


def validate_reusable_subject(
    *,
    subject_dir: Path,
    expected_inputs: Mapping[str, Path],
    expected_parameters: Mapping[str, object],
    expected_outputs: Mapping[str, Path],
    manifest_path: Path | None = None,
    functional: SubjectInputs | None = None,
) -> dict[str, object]:
    directory = Path(subject_dir)
    manifest_file = Path(manifest_path or (directory / "subject_manifest.json"))
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("status") != "ok":
        raise ValueError(f"Reusable subject manifest is not status=ok: {manifest_file}")
    if payload.get("subject") != directory.name:
        raise ValueError(f"Resume subject identity mismatch: {manifest_file}")
    if payload.get("parameters") != dict(expected_parameters):
        raise ValueError("Resume parameter mismatch")
    recorded_inputs = payload.get("inputs")
    if not isinstance(recorded_inputs, Mapping) or set(recorded_inputs) != set(expected_inputs):
        raise ValueError("Resume input set mismatch")
    for key, path in expected_inputs.items():
        recorded = recorded_inputs[key]
        if (
            not isinstance(recorded, Mapping)
            or Path(str(recorded.get("path"))).resolve() != Path(path).resolve()
        ):
            raise ValueError(f"Resume input path mismatch for {key}")
        _validate_fingerprint(recorded, context=f"resume input {key}")
    recorded_outputs = payload.get("outputs")
    if not isinstance(recorded_outputs, Mapping):
        raise ValueError("Resume output metadata is missing")
    if set(recorded_outputs) != set(expected_outputs):
        raise ValueError(
            "Resume output set mismatch: "
            f"recorded={sorted(recorded_outputs)} expected={sorted(expected_outputs)}"
        )
    contracts = canonical_output_contracts(functional)
    legacy_metadata = payload.get("artifact_contract_schema_version") != ARTIFACT_CONTRACT_SCHEMA_VERSION
    for key, expected_path in expected_outputs.items():
        _validate_output_record(
            key=key,
            recorded=recorded_outputs[key],
            expected_path=expected_path,
            contract=contracts.get(key),
            legacy_metadata=legacy_metadata,
        )
    expected_files = {
        Path(path).resolve()
        for path in expected_outputs.values()
        if Path(path).suffix in {".npy", ".gii", ".json"}
    }
    expected_files.add(manifest_file.resolve())
    for path in directory.rglob("*"):
        if path.is_file() and path.resolve() not in expected_files:
            if re.fullmatch(
                r"subject_manifest\.attempt-[0-9]{3,}\.(?:failed|interrupted)\.json",
                path.name,
            ):
                continue
            raise ValueError(f"Unexpected retained subject output: {path}")
    combinations = payload.get("combinations")
    if not isinstance(combinations, list) or len(combinations) != 4:
        raise ValueError("Resume combination set must contain exactly four rows")
    by_key: dict[tuple[str, str], Mapping[str, object]] = {}
    for row in combinations:
        if not isinstance(row, Mapping) or row.get("status") != "ok":
            raise ValueError("Resume combination row is not status=ok")
        key = (str(row.get("smoothing")), str(row.get("hemi")))
        if key in by_key:
            raise ValueError(f"Duplicate resume combination: {key}")
        by_key[key] = row
    expected_keys = {(smoothing, hemi) for smoothing in SMOOTHINGS for hemi in HEMIS}
    if set(by_key) != expected_keys:
        raise ValueError(f"Resume combination set mismatch: {sorted(by_key)}")
    for smoothing, hemi in sorted(expected_keys):
        prefix = f"{smoothing}_hemi-{hemi}"
        row = by_key[(smoothing, hemi)]
        combo_manifest_path = Path(expected_outputs[prefix + "_combo_manifest"]).resolve()
        if Path(str(row.get("manifest"))).resolve() != combo_manifest_path:
            raise ValueError(f"Resume combo manifest path mismatch: {prefix}")
        combo_payload = json.loads(combo_manifest_path.read_text(encoding="utf-8"))
        if (
            combo_payload.get("status") != "ok"
            or combo_payload.get("subject") != directory.name
            or combo_payload.get("smoothing") != smoothing
            or combo_payload.get("hemi") != hemi
        ):
            raise ValueError(f"Resume combo identity mismatch: {prefix}")
        combo_outputs = combo_payload.get("outputs")
        if not isinstance(combo_outputs, Mapping):
            raise ValueError(f"Resume combo outputs are missing: {prefix}")
        for name in ("labels", "scores", "margin"):
            subject_key = prefix + f"_{name}"
            _validate_output_record(
                key=subject_key,
                recorded=combo_outputs.get(name),
                expected_path=expected_outputs[subject_key],
                contract=contracts.get(subject_key),
                legacy_metadata=(
                    combo_payload.get("artifact_contract_schema_version")
                    != ARTIFACT_CONTRACT_SCHEMA_VERSION
                ),
            )
            if combo_outputs.get(name) != recorded_outputs[subject_key]:
                raise ValueError(f"Resume combo output fingerprint mismatch: {subject_key}")
        if "label_gifti" in combo_outputs:
            subject_key = prefix + "_label_gifti"
            if subject_key not in expected_outputs:
                raise ValueError(f"Resume combo label output is unexpected: {prefix}")
            _validate_output_record(
                key=subject_key,
                recorded=combo_outputs["label_gifti"],
                expected_path=expected_outputs[subject_key],
                contract=contracts.get(subject_key),
                legacy_metadata=(
                    combo_payload.get("artifact_contract_schema_version")
                    != ARTIFACT_CONTRACT_SCHEMA_VERSION
                ),
            )
            if combo_outputs["label_gifti"] != recorded_outputs[subject_key]:
                raise ValueError(f"Resume combo output fingerprint mismatch: {subject_key}")
        labels = np.load(expected_outputs[prefix + "_labels"], allow_pickle=False)
        scores = np.load(expected_outputs[prefix + "_scores"], allow_pickle=False)
        margin = np.load(expected_outputs[prefix + "_margin"], allow_pickle=False)
        valid = np.load(
            expected_outputs[f"valid_mask_hemi-{hemi}"], allow_pickle=False
        )
        summary = validate_combo_arrays(
            labels=labels, scores=scores, margin=margin, valid_mask=valid
        )
        for key in ("valid_vertices", "label_counts", "margin_summary"):
            if row.get(key) != summary[key] or combo_payload.get(key) != summary[key]:
                raise ValueError(f"Resume combo summary mismatch for {prefix} {key}")
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
    attempt_pattern = re.compile(
        r"subject_manifest\.attempt-(?P<number>[0-9]{3,})\."
        r"(?:failed|interrupted)\.json"
    )
    attempts: list[int] = []
    unexpected: list[Path] = []
    for entry in entries:
        if entry == partial:
            continue
        match = attempt_pattern.fullmatch(entry.name)
        if match is None or not entry.is_file():
            unexpected.append(entry)
        else:
            attempts.append(int(match.group("number")))
    if not partial.is_file() or unexpected:
        raise ValueError(
            "Failed subject resume requires only the active partial and versioned "
            f"failed/interrupted histories; unexpected={unexpected}"
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
    attempt_number = max(attempts, default=0) + 1
    history = directory / (
        f"subject_manifest.attempt-{attempt_number:03d}.failed.json"
    )
    with history.open("xb") as stream:
        stream.write(original)
    record = fingerprint(history)
    record["prior_error"] = payload.get("error")
    return record


def _normalized_template_roster(
    raw: object,
    *,
    context: str,
) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Template roster must be a non-empty list: {context}")
    subjects = tuple(normalize_subject(value) for value in raw)
    if len(set(subjects)) != len(subjects):
        raise ValueError(f"Template roster must contain unique subjects: {context}")
    return subjects


def _template_source_roster(recorded: object, *, context: str) -> tuple[str, ...]:
    if not isinstance(recorded, Mapping):
        raise ValueError(f"Missing template source manifest fingerprint: {context}")
    _validate_fingerprint(recorded, context=context)
    path = Path(str(recorded["path"]))
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _normalized_template_roster(payload.get("subjects"), context=str(path))


def _load_template_contract(template_root: Path) -> TemplateContract:
    root = Path(template_root).resolve()
    manifest_path = root / "template_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "status": "complete",
        "template_id": TEMPLATE_ID,
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
    source_rosters: list[tuple[str, ...]] = []
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
        source_rosters.extend(
            [
                _template_source_roster(
                    sources.get("group_manifest"),
                    context=f"HCP template {smoothing} hemi-{hemi} group",
                ),
                _template_source_roster(
                    sources.get("gradient_manifest"),
                    context=f"HCP template {smoothing} hemi-{hemi} gradient",
                ),
            ]
        )
        validate_consumed_surface_density(
            density=raw.get("density_evidence"),
            source_record=sources.get("consumed_surface"),
            context=f"HCP template {smoothing} hemi-{hemi}",
        )
        for name, record in outputs.items():
            _validate_fingerprint(record, context=f"HCP template output {name}")
            output_path = Path(str(record.get("path")))
            if output_path.suffix == ".npy":
                if "shape" not in record or "dtype" not in record:
                    raise ValueError(
                        f"HCP template output shape/dtype metadata is missing: {name}"
                    )
                actual = artifact_record(output_path)
                if record.get("shape") != actual.get("shape") or record.get("dtype") != actual.get("dtype"):
                    raise ValueError(f"HCP template output metadata drift: {name}")
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
    if not source_rosters:
        raise ValueError("HCP template source roster is missing")
    expected_roster = source_rosters[0]
    if any(roster != expected_roster for roster in source_rosters[1:]):
        raise ValueError("HCP template source rosters differ")
    recorded_roster = manifest.get("subjects")
    subjects = (
        expected_roster
        if recorded_roster is None
        else _normalized_template_roster(
            recorded_roster,
            context=str(manifest_path),
        )
    )
    if subjects != expected_roster:
        raise ValueError("HCP template roster differs from fingerprinted source manifests")
    if manifest.get("n_subjects") != len(subjects):
        raise ValueError(
            f"HCP template n_subjects mismatch: {manifest.get('n_subjects')!r} != {len(subjects)!r}"
        )
    return TemplateContract(
        root=root,
        manifest_path=manifest_path,
        manifest_fingerprint=fingerprint(manifest_path),
        roi_registry_path=registry_path,
        roi_registry_fingerprint=dict(registry_record),
        roi_registry=registry,
        prototypes=prototypes,
        subjects=subjects,
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
            f"sub-{subject_id}_hemi-{hemi}_space-T1w_den-512_"
            "label-hipp_midthickness.surf.gii"
        )
    ).resolve()


def _surface_vertex_count(path: Path) -> int:
    points = validate_gifti_pointset(path)
    return int(points.shape[0])

def validate_consumed_surface_density(
    *,
    density: object,
    source_record: object,
    context: str,
) -> dict[str, object]:
    if not isinstance(density, Mapping) or not isinstance(source_record, Mapping):
        raise ValueError(f"Missing consumed-surface density evidence: {context}")
    _validate_fingerprint(source_record, context=f"{context} consumed surface")
    surface = Path(str(source_record.get("path"))).resolve()
    if Path(str(density.get("consumed_asset"))).resolve() != surface:
        raise ValueError(f"Consumed-surface density path mismatch: {context}")
    recorded_vertices = int(density.get("n_vertices", -1))
    actual_vertices = _surface_vertex_count(surface)
    if recorded_vertices != N_VERTICES or actual_vertices != N_VERTICES:
        raise ValueError(
            f"Consumed surface is not 512 vertices: {context} "
            f"(recorded={recorded_vertices}, actual={actual_vertices})"
        )
    evidence_source = density.get("evidence_source")
    if evidence_source not in {
        "group_atlas_manifest.density_evidence",
        "consumed_surface_path_and_gifti_point_count",
    }:
        raise ValueError(f"Unexpected density evidence source: {context}")
    return dict(density)


def _validate_surfaces(
    *,
    hippunfold_root: Path,
    subject: str,
) -> dict[str, dict[str, object]]:
    evidence: dict[str, dict[str, object]] = {}
    for hemi in HEMIS:
        path = _surface_path(hippunfold_root, subject, hemi)
        if not path.is_file() or "_space-T1w_den-512_" not in path.name:
            raise FileNotFoundError(f"Missing consumed T1w den-512 surface: {path}")
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
        "areal_lh_labels": Path(label_root) / f"sub-{subject}" / "lh_labels.txt",
        "areal_rh_labels": Path(label_root) / f"sub-{subject}" / "rh_labels.txt",
        "surface_hemi-L": _surface_path(hippunfold_root, subject, "L"),
        "surface_hemi-R": _surface_path(hippunfold_root, subject, "R"),
        "template_manifest": template.manifest_path,
        "roi_registry": template.roi_registry_path,
    }
    if functional.concat_dtseries is not None:
        paths["concat_dtseries"] = functional.concat_dtseries
    for run in functional.runs:
        paths[f"run-{run.run_id:02d}_volume"] = run.volume
        paths[f"run-{run.run_id:02d}_dtseries"] = run.dtseries
    paths['native_to_mni_image_transform'] = dataset_registration('iowa_jj_tms', subject).image_transform
    for smoothing in SMOOTHINGS:
        for hemi in HEMIS:
            paths[f"template_{smoothing}_hemi-{hemi}_prototypes"] = template.prototypes[
                (smoothing, hemi)
            ]
    return {key: Path(path).resolve() for key, path in paths.items()}


def _expected_subject_output_paths(
    *,
    functional: SubjectInputs,
    subject_dir: Path,
) -> dict[str, Path]:
    subject = functional.subject
    directory = Path(subject_dir)
    paths: dict[str, Path] = {}
    for run in functional.runs:
        for hemi in HEMIS:
            paths[f"run-{run.run_id:02d}_hemi-{hemi}_gifti"] = (
                directory
                / "runs"
                / f"run-{run.run_id:02d}"
                / (
                    f"sub-{subject}_task-rest_run-{run.run_id:02d}_hemi-{hemi}_"
                    "space-T1w_den-512_label-hipp_bold.func.gii"
                )
            )
    for hemi in HEMIS:
        paths[f'sampling_surface_hemi-{hemi}'] = directory / 'surface/sampling_surfaces' / (
            f'sub-{subject}_hemi-{hemi}_space-{MNI_SPACE}_den-512_label-hipp_midthickness.surf.gii'
        )
        paths[f"raw_hemi-{hemi}_timeseries"] = (
            directory / "surface" / "raw" / f"hemi-{hemi}_timeseries.npy"
        )
        paths[f"raw_hemi-{hemi}_gifti"] = directory / "surface" / "raw" / (
            f"sub-{subject}_hemi-{hemi}_space-T1w_den-512_"
            "label-hipp_bold.func.gii"
        )
        paths[f"valid_mask_hemi-{hemi}"] = (
            directory / "surface" / "validity" / f"hemi-{hemi}_valid_mask.npy"
        )
        paths[f"valid_roi_hemi-{hemi}"] = directory / "surface" / "validity" / (
            f"hemi-{hemi}_valid_roi.shape.gii"
        )
        paths[f"cortex_valid_mask_hemi-{hemi}"] = (
            directory / "cortex" / f"hemi-{hemi}_valid_mask.npy"
        )
        for smoothing in SMOOTHINGS:
            paths[f"{smoothing}_hemi-{hemi}_gifti"] = (
                directory
                / "surface"
                / smoothing
                / (
                    f"sub-{subject}_hemi-{hemi}_space-T1w_den-512_"
                    f"label-hipp_bold_{smoothing}.func.gii"
                )
            )
            paths[f"{smoothing}_hemi-{hemi}_timeseries"] = (
                directory / "surface" / smoothing / f"hemi-{hemi}_timeseries.npy"
            )
            combo_dir = directory / "matching" / smoothing / f"hemi-{hemi}"
            paths[f"{smoothing}_hemi-{hemi}_pearson_fc"] = (
                combo_dir / "vertex_to_roi_pearson_fc.npy"
            )
            paths[f"{smoothing}_hemi-{hemi}_fisherz_fc"] = (
                combo_dir / "vertex_to_roi_fisherz_fc.npy"
            )
            paths[f"{smoothing}_hemi-{hemi}_labels"] = combo_dir / "labels.npy"
            paths[f"{smoothing}_hemi-{hemi}_scores"] = (
                combo_dir / "template_correlations.npy"
            )
            paths[f"{smoothing}_hemi-{hemi}_margin"] = (
                combo_dir / "correlation_margin.npy"
            )
            paths[f"{smoothing}_hemi-{hemi}_label_gifti"] = combo_dir / (
                f"sub-{subject}_hemi-{hemi}_space-corobl_den-512_"
                f"label-hcp_fixed_template_k3_{smoothing}.label.gii"
            )
            paths[f"{smoothing}_hemi-{hemi}_combo_manifest"] = (
                combo_dir / "combo_manifest.json"
            )
    paths["cortex_roi_timeseries"] = (
        directory / "cortex" / "kong2022_roi_timeseries.npy"
    )
    paths["cortex_roi_vertex_counts"] = (
        directory / "cortex" / "kong2022_roi_vertex_counts.npy"
    )
    return {key: path.resolve() for key, path in paths.items()}


def _parameters(*, wb_command: Path) -> dict[str, object]:
    return {
        "task": "rest",
        "mapping_method": "trilinear",
        "density": N_VERTICES,
        "surface_space": "T1w",
        "sampling_space": MNI_SPACE,
        "geometry_space": "T1w",
        "smoothings": list(SMOOTHINGS),
        "hemis": list(HEMIS),
        "validity_policy": SERIES_VALIDITY_POLICY,
        "cortex_validity_policy": CORTEX_VALIDITY_POLICY,
        "fc": "vertex-to-ROI Pearson",
        "fc_transform": "arctanh clipped to +/-0.999999",
        "matching": "jointly-finite Pearson correlation to fixed HCP prototypes",
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
    validate_gifti_logical_contract(
        path,
        expected_shape=tuple(int(value) for value in array.shape),
        expected_dtype=np.float32,
        context=str(path),
    )

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
    validate_gifti_logical_contract(
        path,
        expected_shape=(N_VERTICES,),
        expected_dtype=np.float32,
        context=str(path),
    )

def load_func_gifti(
    path: Path,
    *,
    expected_vertices: int = N_VERTICES,
    expected_timepoints: int | None = None,
) -> np.ndarray:
    return np.asarray(
        load_gifti_timeseries(
            path,
            expected_vertices=expected_vertices,
            expected_timepoints=expected_timepoints,
        ),
        dtype=np.float32,
    )

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
    mapped = registered_sampling_surfaces(
        {hemi: Path(str(surfaces[hemi]['consumed_asset'])) for hemi in HEMIS},
        registration=dataset_registration('iowa_jj_tms', functional.subject),
        outdir=subject_dir/'surface/sampling_surfaces', wb_command=wb_command,
    )
    for hemi, path in mapped.items():
        outputs[f'sampling_surface_hemi-{hemi}'] = path
    for run in functional.runs:
        run_dir = subject_dir / "runs" / f"run-{run.run_id:02d}"
        for hemi in HEMIS:
            final = run_dir / (
                f"sub-{functional.subject}_task-rest_run-{run.run_id:02d}_hemi-{hemi}_"
                "space-T1w_den-512_label-hipp_bold.func.gii"
            )
            partial = final.with_name(f".{final.stem}.partial.func.gii")
            partial.parent.mkdir(parents=True, exist_ok=True)
            data = sample_volume_to_surface(
                bold=run.volume, surface=mapped[hemi], out_metric=partial,
                mapping_method="trilinear", wb_command=wb_command,
            )
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
        expected_total_tp = (
            int(nib.load(str(functional.concat_dtseries)).shape[0])
            if functional.concat_dtseries is not None
            else sum(int(nib.load(str(r.volume)).shape[3]) for r in functional.runs)
        )
        if data.shape[1] != expected_total_tp:
            raise ValueError("Sampled surface concatenation timepoint mismatch")
        concatenated[hemi] = data
        npy = subject_dir / "surface" / "raw" / f"hemi-{hemi}_timeseries.npy"
        gifti = subject_dir / "surface" / "raw" / (
            f"sub-{functional.subject}_hemi-{hemi}_space-T1w_den-512_"
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
        valid = rest_signal_validity(raw[hemi])
        valid_path = subject_dir / "surface" / "validity" / f"hemi-{hemi}_valid_mask.npy"
        roi_path = subject_dir / "surface" / "validity" / f"hemi-{hemi}_valid_roi.shape.gii"
        _save_npy_exclusive(valid_path, valid.astype(bool))
        _write_roi_gifti_exclusive(roi_path, valid)
        outputs[f"valid_mask_hemi-{hemi}"] = valid_path
        outputs[f"valid_roi_hemi-{hemi}"] = roi_path
        valid_by_hemi[hemi] = valid
        raw_gifti = subject_dir / "surface" / "raw" / (
            f"sub-{subject}_hemi-{hemi}_space-T1w_den-512_label-hipp_bold.func.gii"
        )
        for smoothing in SMOOTHINGS:
            out_dir = subject_dir / "surface" / smoothing
            final = out_dir / (
                f"sub-{subject}_hemi-{hemi}_space-T1w_den-512_"
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
            data = load_func_gifti(
                partial, expected_timepoints=int(raw[hemi].shape[1])
            )
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
    concat_dtseries: Path | None = None,
    runs: Sequence[RunInput] | None = None,
    areal_labels: Mapping[str, np.ndarray],
    roi_registry: Sequence[Mapping[str, object]],
    subject_dir: Path,
    outputs: dict[str, Path],
    run_lengths: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    if concat_dtseries is not None:
        image = nib.load(str(concat_dtseries))
        data = np.asarray(image.dataobj, dtype=np.float32)
        axis = image.header.get_axis(1)
    elif runs:
        parts: list[np.ndarray] = []
        axis = None
        for run in runs:
            img = nib.load(str(run.dtseries))
            if axis is None:
                axis = img.header.get_axis(1)
            parts.append(np.asarray(img.dataobj, dtype=np.float32))
        data = np.concatenate(parts, axis=0)
    else:
        raise ValueError("Must provide either concat_dtseries or runs")
    if data.ndim != 2:
        raise ValueError(f"Expected 2D concat dtseries data, got {data.shape}")
    axis = axis or image.header.get_axis(1)
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
        cortex_valid = cortex_signal_validity(hemi_data, run_lengths)
        parcel_ids = parcel_ids.copy()
        parcel_ids[~cortex_valid] = 0
        cortex[hemi] = np.asarray(hemi_data, dtype=np.float32)
        aligned[hemi] = parcel_ids
        valid_path = subject_dir / "cortex" / f"hemi-{hemi}_valid_mask.npy"
        _save_npy_exclusive(valid_path, cortex_valid.astype(bool))
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
    validate_gifti_logical_contract(
        path,
        expected_shape=(N_VERTICES,),
        expected_dtype=np.int32,
        context=str(path),
    )

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
                    f"label-hcp_fixed_template_k3_{smoothing}.label.gii"
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
                "artifact_contract_schema_version": ARTIFACT_CONTRACT_SCHEMA_VERSION,
                "subject": f"sub-{subject}",
                "smoothing": smoothing,
                "hemi": hemi,
                "label_semantics": {
                    "1": "Anterior/VAN",
                    "2": "Middle/SMN",
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


def _validate_subject_partial_header(
    *,
    payload: Mapping[str, object],
    partial: Path,
    subject_dir: Path,
    expected_inputs: Mapping[str, Path],
    expected_parameters: Mapping[str, object],
) -> None:
    if payload.get("subject") != subject_dir.name:
        raise ValueError(f"Subject partial identity mismatch: {partial}")
    if payload.get("parameters") != dict(expected_parameters):
        raise ValueError("Subject partial parameter mismatch")
    recorded_inputs = payload.get("inputs")
    if not isinstance(recorded_inputs, Mapping) or set(recorded_inputs) != set(expected_inputs):
        raise ValueError("Subject partial input set mismatch")
    for key, path in expected_inputs.items():
        recorded = recorded_inputs[key]
        if (
            not isinstance(recorded, Mapping)
            or Path(str(recorded.get("path"))).resolve() != Path(path).resolve()
        ):
            raise ValueError(f"Subject partial input path mismatch for {key}")
        _validate_fingerprint(recorded, context=f"subject partial input {key}")


def _subject_attempt_history_number(directory: Path) -> int:
    pattern = re.compile(
        r"subject_manifest\.attempt-(?P<number>[0-9]{3,})\.(?:failed|interrupted)\.json"
    )
    numbers = [
        int(match.group("number"))
        for entry in directory.iterdir()
        if (match := pattern.fullmatch(entry.name)) is not None
    ]
    return max(numbers, default=0) + 1


def prepare_subject_resume_state(
    *,
    subject_dir: Path,
    expected_inputs: Mapping[str, Path],
    expected_parameters: Mapping[str, object],
    expected_outputs: Mapping[str, Path],
    resume: bool,
) -> dict[str, object] | None:
    directory = Path(subject_dir)
    partial = directory / "subject_manifest.partial.json"
    if not directory.exists():
        return None
    entries = list(directory.iterdir())
    if not entries:
        return None
    if not resume:
        raise FileExistsError(
            "Partial/nonvalidated subject output is retained and requires "
            f"--resume: {directory}"
        )
    attempt_pattern = re.compile(
        r"subject_manifest\.attempt-(?P<number>[0-9]{3,})\.(?:failed|interrupted)\.json"
    )
    unexpected = [
        entry
        for entry in entries
        if entry != partial
        and (not entry.is_file() or attempt_pattern.fullmatch(entry.name) is None)
    ]
    if unexpected or not partial.is_file():
        raise ValueError(
            "Subject resume requires one active partial and only versioned histories; "
            f"unexpected={unexpected}"
        )
    payload = json.loads(partial.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid subject partial: {partial}")
    status = payload.get("status")
    _validate_subject_partial_header(
        payload=payload,
        partial=partial,
        subject_dir=directory,
        expected_inputs=expected_inputs,
        expected_parameters=expected_parameters,
    )
    if status != "in_progress":
        raise ValueError(f"Only an empty in_progress subject partial can be restarted: {partial}")
    retained = payload.get("retained_outputs")
    if retained not in (None, {}):
        raise ValueError("in_progress subject partial retains outputs")
    if payload.get("outputs") not in (None, {}):
        raise ValueError("in_progress subject partial unexpectedly declares outputs")
    attempt_number = _subject_attempt_history_number(directory)
    history = directory / (
        f"subject_manifest.attempt-{attempt_number:03d}.interrupted.json"
    )
    with history.open("xb") as stream:
        stream.write(partial.read_bytes())
    record = fingerprint(history)
    record["prior_status"] = status
    return record


def promote_subject_partial(
    *,
    subject_dir: Path,
    expected_inputs: Mapping[str, Path],
    expected_parameters: Mapping[str, object],
    expected_outputs: Mapping[str, Path],
    functional: SubjectInputs | None = None,
) -> dict[str, object]:
    directory = Path(subject_dir)
    partial = directory / "subject_manifest.partial.json"
    final = directory / "subject_manifest.json"
    if final.exists():
        raise FileExistsError(f"Refusing to overwrite subject manifest: {final}")
    payload = validate_reusable_subject(
        subject_dir=directory,
        expected_inputs=expected_inputs,
        expected_parameters=expected_parameters,
        expected_outputs=expected_outputs,
        manifest_path=partial,
        functional=functional,
    )
    _publish_partial(partial, final)
    return payload


def preflight_subject(
    *,
    subject: str,
    volume_root: Path,
    dtseries_root: Path,
    concat_dtseries_root: Path | None = None,
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
    concat_dtseries_root: Path | None = None,
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
    expected_outputs = _expected_subject_output_paths(
        functional=functional, subject_dir=subject_dir
    )
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
            expected_outputs=expected_outputs,
            functional=functional,
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
        partial = subject_dir / "subject_manifest.partial.json"
        if partial.is_file():
            partial_payload = json.loads(partial.read_text(encoding="utf-8"))
            partial_status = (
                partial_payload.get("status")
                if isinstance(partial_payload, Mapping)
                else None
            )
            if partial_status == "ok":
                reused = promote_subject_partial(
                    subject_dir=subject_dir,
                    expected_inputs=expected_inputs,
                    expected_parameters=parameters,
                    expected_outputs=expected_outputs,
                    functional=functional,
                )
                return {
                    "subject": f"sub-{subject}",
                    "status": "ok",
                    "reused": True,
                    "manifest": str(final_manifest.resolve()),
                    "combinations": reused.get("combinations", []),
                }
            if partial_status == "in_progress":
                attempt_history.append(
                    prepare_subject_resume_state(
                        subject_dir=subject_dir,
                        expected_inputs=expected_inputs,
                        expected_parameters=parameters,
                        expected_outputs=expected_outputs,
                        resume=True,
                    )
                )
            elif partial_status == "failed":
                attempt_history.append(
                    prepare_empty_failed_subject_resume(
                        subject_dir=subject_dir,
                        expected_inputs=expected_inputs,
                        expected_parameters=parameters,
                    )
                )
            else:
                raise ValueError(f"Unsupported subject partial status: {partial}")
        else:
            raise ValueError(
                "Partial/nonvalidated subject output lacks active partial manifest: "
                f"{subject_dir}"
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
            run_lengths=header_evidence["run_lengths"],
            concat_dtseries=functional.concat_dtseries,
            runs=functional.runs,
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
    concat_dtseries_root: Path | None = None,
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
        "smoke_gate",
    )
    for key in immutable_keys:
        if key == "smoke_gate" and key not in recorded:
            continue
        if recorded.get(key) != expected.get(key):
            raise ValueError(f"Run resume contract mismatch for {key}")


def _subject_artifact_anchor(
    manifest_path: Path,
    *,
    combinations: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    path = Path(manifest_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = combinations if combinations is not None else payload.get("combinations")
    if not isinstance(rows, list):
        raise ValueError(f"Subject combination provenance is missing: {path}")
    combo_fingerprints: dict[str, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"Subject combination provenance is invalid: {path}")
        smoothing = str(row.get("smoothing"))
        hemi = str(row.get("hemi"))
        key = f"{smoothing}_hemi-{hemi}"
        combo_path = Path(str(row.get("manifest"))).resolve()
        if key in combo_fingerprints:
            raise ValueError(f"Duplicate subject combination provenance: {path} {key}")
        combo_fingerprints[key] = fingerprint(combo_path)
    expected_keys = {
        f"{smoothing}_hemi-{hemi}"
        for smoothing in SMOOTHINGS
        for hemi in HEMIS
    }
    if set(combo_fingerprints) != expected_keys:
        raise ValueError(f"Subject combination provenance is incomplete: {path}")
    return {
        "subject_manifest_fingerprint": fingerprint(path),
        "combination_fingerprints": combo_fingerprints,
    }


def _validate_complete_run_payload(
    recorded: Mapping[str, object],
    *,
    context: str,
) -> None:
    subjects = recorded.get("subjects")
    items = recorded.get("items")
    if (
        recorded.get("status") != "complete"
        or not isinstance(subjects, list)
        or not isinstance(items, list)
        or len(items) != len(subjects)
        or recorded.get("complete_count") != recorded.get("subject_count")
        or recorded.get("failed_count") != 0
    ):
        raise ValueError(f"{context} is not reusable complete output")
    schema_version = recorded.get("artifact_contract_schema_version")
    explicit_legacy = (
        schema_version is None
        and recorded.get("run_id") in LEGACY_PUBLISHED_RUN_IDS
    )
    if schema_version not in {None, ARTIFACT_CONTRACT_SCHEMA_VERSION}:
        raise ValueError(f"{context} artifact contract schema is unsupported")
    if schema_version is None and not explicit_legacy:
        roots = recorded.get("path_roots")
        if isinstance(roots, Mapping) and len(roots) >= 4:
            raise ValueError(
                f"{context} missing artifact contract schema for anchored output"
            )
    require_anchors = schema_version == ARTIFACT_CONTRACT_SCHEMA_VERSION
    expected_keys = {
        (smoothing, hemi) for smoothing in SMOOTHINGS for hemi in HEMIS
    }
    for item, expected_subject in zip(items, subjects, strict=True):
        if not isinstance(item, Mapping) or item.get("subject") != expected_subject:
            raise ValueError(f"{context} subject item contract mismatch")
        combinations = item.get("combinations")
        if (
            item.get("status") != "ok"
            or not isinstance(combinations, list)
            or len(combinations) != 4
        ):
            raise ValueError(f"{context} is not reusable complete: subject item is incomplete")
        if require_anchors and any(
            field not in item
            for field in ("subject_manifest_fingerprint", "combination_fingerprints")
        ):
            raise ValueError(f"{context} subject artifact anchors are missing")
        keys = {
            (str(row.get("smoothing")), str(row.get("hemi")))
            for row in combinations
            if isinstance(row, Mapping) and row.get("status") == "ok"
        }
        if keys != expected_keys:
            raise ValueError(f"{context} subject combination set mismatch")


def prepare_run_resume_state(
    *,
    run_dir: Path,
    expected: Mapping[str, object],
    resume: bool,
    complete_validator: Callable[[Mapping[str, object]], None] | None = None,
) -> dict[str, object] | None:
    directory = Path(run_dir)
    final = directory / "run_manifest.json"
    partial = directory / "run_manifest.partial.json"
    if not directory.exists():
        directory.mkdir(parents=True)
        return None
    entries = list(directory.iterdir())
    if not entries:
        return None
    if not resume:
        raise FileExistsError(f"Run directory is nonempty; use --resume: {directory}")
    if final.exists() and partial.exists():
        raise FileExistsError(f"Run has both final and partial manifests: {directory}")
    if final.exists():
        recorded = json.loads(final.read_text(encoding="utf-8"))
        if not isinstance(recorded, dict):
            raise ValueError(f"Invalid existing run manifest: {final}")
        validate_run_resume_contract(recorded, expected)
        _validate_complete_run_payload(recorded, context=f"Published run manifest {final}")
        if complete_validator is not None:
            complete_validator(recorded)
        return recorded
    attempt_pattern = re.compile(
        r"run_manifest\.attempt-(?P<number>[0-9]{3,})\."
        r"(?:failed|interrupted)\.json"
    )
    attempts: list[int] = []
    unexpected: list[Path] = []
    for entry in entries:
        if entry == partial:
            continue
        match = attempt_pattern.fullmatch(entry.name)
        if match is None or not entry.is_file():
            unexpected.append(entry)
        else:
            attempts.append(int(match.group("number")))
    if not partial.is_file() or unexpected:
        raise FileExistsError(
            f"Run resume has unexpected retained state: {unexpected or entries}"
        )
    original = partial.read_bytes()
    recorded = json.loads(original)
    if not isinstance(recorded, dict):
        raise ValueError(f"Run partial is invalid: {partial}")
    validate_run_resume_contract(recorded, expected)
    if recorded.get("status") == "complete":
        _validate_complete_run_payload(recorded, context=f"Complete run partial {partial}")
        if complete_validator is not None:
            complete_validator(recorded)
        _publish_partial(partial, final)
        return recorded
    if recorded.get("status") not in {"in_progress", "failed"}:
        raise ValueError(f"Run partial has invalid status: {partial}")
    attempt_number = max(attempts, default=0) + 1
    suffix = "failed" if recorded["status"] == "failed" else "interrupted"
    history = directory / (
        f"run_manifest.attempt-{attempt_number:03d}.{suffix}.json"
    )
    with history.open("xb") as stream:
        stream.write(original)
    return None


def _validate_complete_run_dependencies(
    *,
    recorded: Mapping[str, object],
    out_root: Path,
    template: TemplateContract,
    parameters: Mapping[str, object],
) -> None:
    _validate_complete_run_payload(recorded, context="Run manifest")
    roots = recorded.get("path_roots")
    subjects = recorded.get("subjects")
    items = recorded.get("items")
    if not isinstance(roots, Mapping) or not isinstance(subjects, list) or not isinstance(items, list):
        raise ValueError("Complete run dependency closure metadata is missing")
    require_anchors = (
        recorded.get("artifact_contract_schema_version")
        == ARTIFACT_CONTRACT_SCHEMA_VERSION
    )
    fresh_rows: list[dict[str, object]] = []
    for subject, item in zip(subjects, items, strict=True):
        normalized = normalize_subject(subject)
        manifest, rows = _validate_subject_for_run(
            subject=normalized,
            roots=roots,
            out_root=out_root,
            template=template,
            parameters=parameters,
        )
        expected_manifest_path = (
            Path(out_root) / "subjects" / f"sub-{normalized}" / "subject_manifest.json"
        ).resolve()
        if not isinstance(item, Mapping) or item.get("manifest") != str(expected_manifest_path):
            raise ValueError(f"Run subject manifest path mismatch: sub-{normalized}")
        current_anchor = _subject_artifact_anchor(expected_manifest_path)
        for field in ("subject_manifest_fingerprint", "combination_fingerprints"):
            if require_anchors and field not in item:
                raise ValueError(f"Run subject artifact anchor is missing: sub-{normalized}")
            if field in item and item.get(field) != current_anchor[field]:
                raise ValueError(f"Run subject artifact fingerprint drift: sub-{normalized}")
        item_rows = item.get("combinations")
        if not isinstance(item_rows, list) or {
            (str(row.get("smoothing")), str(row.get("hemi")))
            for row in item_rows
            if isinstance(row, Mapping)
        } != {
            (str(row["smoothing"]), str(row["hemi"])) for row in rows
        }:
            raise ValueError(f"Run combination provenance mismatch: sub-{normalized}")
        if manifest.get("status") != "ok":
            raise ValueError(f"Run subject manifest is not ok: sub-{normalized}")
        fresh_rows.extend(rows)
    return fresh_rows


def _run_batch(
    *,
    subjects: Sequence[str],
    canonical_roster: Sequence[str],
    run_id: str,
    jobs: int,
    resume: bool,
    volume_root: Path,
    dtseries_root: Path,
    concat_dtseries_root: Path | None = None,
    label_root: Path,
    hippunfold_root: Path,
    out_root: Path,
    template: TemplateContract,
    wb_command: Path,
    skip_smoke_gate: bool = False,
) -> dict[str, object]:
    smoke_gate: dict[str, object] | None = None
    if list(subjects) == list(canonical_roster) and not skip_smoke_gate:
        smoke_gate = validate_full_run_smoke_gate(
            out_root=out_root,
            template_fingerprint=template.manifest_fingerprint,
            roi_registry_fingerprint=template.roi_registry_fingerprint,
            template=template,
        )
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
    base = {
        "workflow": "jj_tms_hcp_fixed_template_matching",
        "artifact_contract_schema_version": ARTIFACT_CONTRACT_SCHEMA_VERSION,
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
            "concat_dtseries_root": (
                str(Path(concat_dtseries_root).resolve())
                if concat_dtseries_root is not None and str(concat_dtseries_root).strip().lower() not in ("none", "null", "")
                else None
            ),
            "label_root": str(Path(label_root).resolve()),
            "hippunfold_root": str(Path(hippunfold_root).resolve()),
            "template_root": str(template.root),
            "out_root": str(Path(out_root).resolve()),
        },
    }
    if smoke_gate is not None:
        base["smoke_gate"] = dict(smoke_gate)
    reusable_run = prepare_run_resume_state(
        run_dir=run_dir,
        expected=base,
        resume=resume,
        complete_validator=lambda recorded: _validate_complete_run_dependencies(
            recorded=recorded,
            out_root=out_root,
            template=template,
            parameters=base["parameters"],
        ),
    )
    if reusable_run is not None:
        return reusable_run
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
                result = future.result()
                if result.get("status") == "ok":
                    result = {
                        **result,
                        **_subject_artifact_anchor(
                            Path(str(result["manifest"])),
                            combinations=result.get("combinations"),
                        ),
                    }
                items.append(result)
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
    payload = {
        **base,
        "status": status,
        "complete_count": complete_count,
        "failed_count": len(subjects) - complete_count,
        "items": items,
    }
    _write_json(partial_manifest, payload)
    if status == "complete":
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
    concat_root = roots.get("concat_dtseries_root")
    concat_dtseries_root = (
        Path(str(concat_root))
        if concat_root is not None and str(concat_root).strip().lower() not in ("none", "null", "")
        else None
    )
    functional = resolve_subject_runs(
        subject=subject,
        volume_root=Path(str(roots["volume_root"])),
        dtseries_root=Path(str(roots["dtseries_root"])),
        concat_dtseries_root=concat_dtseries_root,
    )
    validate_functional_headers(functional)
    load_areal_labels(label_root=Path(str(roots["label_root"])), subject=subject)
    _validate_surfaces(
        hippunfold_root=Path(str(roots["hippunfold_root"])),
        subject=subject,
    )
    expected_inputs = _expected_subject_inputs(
        functional=functional,
        label_root=Path(str(roots["label_root"])),
        hippunfold_root=Path(str(roots["hippunfold_root"])),
        template=template,
    )
    subject_dir = Path(out_root) / "subjects" / f"sub-{subject}"
    expected_outputs = _expected_subject_output_paths(
        functional=functional, subject_dir=subject_dir
    )
    manifest = validate_reusable_subject(
        subject_dir=subject_dir,
        expected_inputs=expected_inputs,
        expected_parameters=parameters,
        expected_outputs=expected_outputs,
        functional=functional,
    )
    rows: list[dict[str, object]] = []
    combinations = manifest.get("combinations")
    if not isinstance(combinations, list) or len(combinations) != 4:
        raise ValueError(f"Expected four subject combinations: sub-{subject}")
    subject_manifest_fingerprint = fingerprint(
        subject_dir / "subject_manifest.json"
    )
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
            subject_dir / "surface" / "validity" / f"hemi-{hemi}_valid_mask.npy",
            allow_pickle=False,
        ).astype(bool)
        summary = validate_combo_arrays(
            labels=labels,
            scores=scores,
            margin=margin,
            valid_mask=valid,
        )
        density = validate_consumed_surface_density(
            density=combo_payload.get("density_evidence"),
            source_record=manifest["inputs"].get(f"surface_hemi-{hemi}"),
            context=f"sub-{subject} {smoothing} hemi-{hemi}",
        )
        rows.append(
            {
                "subject": f"sub-{subject}",
                "smoothing": smoothing,
                "hemi": hemi,
                "status": "ok",
                "label_counts": summary["label_counts"],
                "margin_summary": summary["margin_summary"],
                "density_evidence": density,
                "labels_sha256": outputs["labels"]["sha256"],
                "subject_manifest_fingerprint": subject_manifest_fingerprint,
                "combo_manifest_fingerprint": fingerprint(manifest_path),
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
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8", newline="") as stream:
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
    os.replace(temporary, path)


def _read_validation_table(path: Path) -> list[dict[str, str]]:
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
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if set(reader.fieldnames or ()) != expected_fields:
            raise ValueError(f"Validation table header contract mismatch: {path}")
        rows = [dict(row) for row in reader]
    if any(set(row) != expected_fields for row in rows):
        raise ValueError(f"Validation table row contract mismatch: {path}")
    return rows


def _validate_validation_payload_dependencies(
    *,
    payload: Mapping[str, object],
    expected_run_manifest: Mapping[str, object],
    template_fingerprint: Mapping[str, object],
    roi_registry_fingerprint: Mapping[str, object],
    out_root: Path,
    template: TemplateContract,
) -> None:
    if payload.get("status") != "complete":
        raise ValueError("Validation manifest is not complete")
    schema_version = payload.get("artifact_contract_schema_version")
    explicit_legacy = (
        schema_version is None
        and payload.get("run_id") in LEGACY_PUBLISHED_RUN_IDS
    )
    if schema_version not in {None, ARTIFACT_CONTRACT_SCHEMA_VERSION}:
        raise ValueError("Validation artifact contract schema is unsupported")
    if schema_version is None and not explicit_legacy:
        raise ValueError("Validation artifact contract schema is missing")
    if payload.get("run_manifest") != dict(expected_run_manifest):
        raise ValueError("Validation run-manifest fingerprint drift")
    if payload.get("template_manifest") != dict(template_fingerprint):
        raise ValueError("Validation template fingerprint drift")
    if payload.get("roi_registry") != dict(roi_registry_fingerprint):
        raise ValueError("Validation ROI registry fingerprint drift")
    _validate_fingerprint(payload.get("run_manifest"), context="validation run manifest")
    _validate_fingerprint(payload.get("template_manifest"), context="validation template")
    _validate_fingerprint(payload.get("roi_registry"), context="validation ROI registry")
    _validate_fingerprint(payload.get("validation_items"), context="validation table")
    table_record = payload["validation_items"]
    table_path = Path(str(table_record["path"])).resolve()
    rows = _read_validation_table(table_path)
    run_path = Path(str(payload["run_manifest"]["path"])).resolve()
    run_payload = json.loads(run_path.read_text(encoding="utf-8"))
    fresh_rows = _validate_complete_run_dependencies(
        recorded=run_payload,
        out_root=out_root,
        template=template,
        parameters=run_payload["parameters"],
    )
    subjects = run_payload.get("subjects")
    if not isinstance(subjects, list):
        raise ValueError("Validation run manifest subjects are missing")
    expected_subject_manifests = [
        str((Path(out_root) / "subjects" / f"sub-{normalize_subject(subject)}" / "subject_manifest.json").resolve())
        for subject in subjects
    ]
    if payload.get("subject_manifests") != expected_subject_manifests:
        raise ValueError("Validation subject-manifest dependency closure mismatch")
    items = payload.get("items")
    if not isinstance(items, list) or len(items) != len(subjects) * 4:
        raise ValueError("Validation item dependency closure is incomplete")
    expected_keys: set[tuple[str, str, str]] = set()
    for item in items:
        if not isinstance(item, Mapping) or item.get("status") != "ok":
            raise ValueError("Validation item is not complete")
        key = (
            str(item.get("subject")),
            str(item.get("smoothing")),
            str(item.get("hemi")),
        )
        if key in expected_keys:
            raise ValueError(f"Duplicate validation item: {key}")
        expected_keys.add(key)
    expected_key_set = {
        (str(subject), smoothing, hemi)
        for subject in subjects
        for smoothing in SMOOTHINGS
        for hemi in HEMIS
    }
    if expected_keys != expected_key_set:
        raise ValueError("Validation item subject/combination set mismatch")
    anchor_fields = {
        "subject_manifest_fingerprint",
        "combo_manifest_fingerprint",
    }
    fresh_by_key = {
        (str(item["subject"]), str(item["smoothing"]), str(item["hemi"])): item
        for item in fresh_rows
    }
    for recorded_item in items:
        key = (
            str(recorded_item["subject"]),
            str(recorded_item["smoothing"]),
            str(recorded_item["hemi"]),
        )
        fresh_item = fresh_by_key.get(key)
        if fresh_item is None:
            raise ValueError(f"Fresh validation item is missing: {key}")
        if anchor_fields.intersection(recorded_item):
            if recorded_item != fresh_item:
                raise ValueError(f"Fresh validation item drift: {key}")
        else:
            legacy_fresh = {
                name: value
                for name, value in fresh_item.items()
                if name not in anchor_fields
            }
            if recorded_item != legacy_fresh:
                raise ValueError(f"Legacy validation item drift: {key}")
    if len(rows) != len(items):
        raise ValueError("Validation table row count mismatch")
    table_by_key = {
        (row["subject"], row["smoothing"], row["hemi"]): row
        for row in rows
    }
    if set(table_by_key) != expected_keys:
        raise ValueError("Validation table subject/combination set mismatch")
    for item in items:
        key = (str(item["subject"]), str(item["smoothing"]), str(item["hemi"]))
        row = table_by_key[key]
        counts = item["label_counts"]
        margin = item["margin_summary"]
        if (
            row["status"] != "ok"
            or row["labels_sha256"] != str(item["labels_sha256"])
            or row["consumed_surface"] != str(item["density_evidence"]["consumed_asset"])
            or [row[f"label_{code}_count"] for code in range(1, K + 1)]
            != [str(counts[str(code)]) for code in range(1, K + 1)]
        ):
            raise ValueError(f"Validation table content drift: {key}")
        for name in ("min", "median", "mean", "max"):
            if float(row[f"margin_{name}"]) != float(margin[name]):
                raise ValueError(f"Validation table margin drift: {key} {name}")


def _validate_run(
    *,
    run_id: str,
    out_root: Path,
    template: TemplateContract,
    selected_subjects: Sequence[str] | None,
    canonical_roster: Sequence[str],
    resume: bool,
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
        if subjects != list(canonical_roster):
            raise ValueError("Full-run manifest does not contain the fixed canonical roster")
    if run_manifest.get("template_manifest") != dict(template.manifest_fingerprint):
        raise ValueError("Run template manifest fingerprint drift")
    if run_manifest.get("roi_registry") != dict(template.roi_registry_fingerprint):
        raise ValueError("Run ROI registry fingerprint drift")
    roots = run_manifest.get("path_roots")
    parameters = run_manifest.get("parameters")
    if not isinstance(roots, Mapping) or not isinstance(parameters, Mapping):
        raise ValueError("Run path/parameter metadata is missing")
    run_manifest_fingerprint = fingerprint(run_manifest_path)
    reusable_validation = prepare_validation_resume_state(
        run_dir=run_dir,
        expected_run_manifest=run_manifest_fingerprint,
        template_fingerprint=template.manifest_fingerprint,
        roi_registry_fingerprint=template.roi_registry_fingerprint,
        resume=resume,
        complete_validator=lambda payload: _validate_validation_payload_dependencies(
            payload=payload,
            expected_run_manifest=run_manifest_fingerprint,
            template_fingerprint=template.manifest_fingerprint,
            roi_registry_fingerprint=template.roi_registry_fingerprint,
            out_root=out_root,
            template=template,
        ),
    )
    if reusable_validation is not None:
        return reusable_validation
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
        subject_manifest_path = (
            Path(out_root) / "subjects" / f"sub-{subject}" / "subject_manifest.json"
        ).resolve()
        subject_manifests.append(str(subject_manifest_path))
    if len(rows) != len(subjects) * 4:
        raise ValueError("Validation did not produce four combinations per subject")
    payload_base = {
        "status": "complete",
        "workflow": "validate_jj_tms_hcp_fixed_template_matching",
        "artifact_contract_schema_version": ARTIFACT_CONTRACT_SCHEMA_VERSION,
        "run_id": run_id,
        "scope": run_manifest.get("scope"),
        "subject_count": len(subjects),
        "complete_subject_count": len(subjects),
        "combination_count": len(rows),
        "template_manifest": dict(template.manifest_fingerprint),
        "roi_registry": dict(template.roi_registry_fingerprint),
        "run_manifest": run_manifest_fingerprint,
        "subject_manifests": subject_manifests,
        "items": rows,
    }
    return publish_validation_artifacts(
        run_dir=run_dir, rows=rows, payload_base=payload_base
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply immutable HCP prototypes to fixed-roster JJ_TMS rest data."
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
        type=str,
        default=str(DEFAULT_CONCAT_DTSERIES_ROOT),
        help="Path to concatenated dtseries root, or 'none' for direct in-memory concatenation.",
    )
    parser.add_argument(
        "--skip-smoke-gate",
        action="store_true",
        help="Skip canonical smoke gate requirement before full run.",
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
    concat_root = (
        None
        if args.concat_dtseries_root is None
        or str(args.concat_dtseries_root).strip().lower() in ("none", "null", "")
        else Path(args.concat_dtseries_root)
    )
    template = _load_template_contract(args.template_root)
    if args.mode == "preflight":
        payload = _preflight_batch(
            subjects=subjects,
            volume_root=args.volume_root,
            dtseries_root=args.dtseries_root,
            concat_dtseries_root=concat_root,
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
            concat_dtseries_root=concat_root,
            label_root=args.areal_label_root,
            hippunfold_root=args.hippunfold_root,
            out_root=args.out_root,
            template=template,
            wb_command=args.wb_command,
            skip_smoke_gate=bool(args.skip_smoke_gate),
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
            resume=bool(args.resume),
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("status") in {"ok", "complete"} else 1


def validate_full_run_smoke_gate(
    *,
    out_root: Path,
    template_fingerprint: Mapping[str, object],
    roi_registry_fingerprint: Mapping[str, object],
    template: TemplateContract | None = None,
) -> dict[str, object]:
    run_dir = _run_dir(out_root, SMOKE_RUN_ID)
    manifest_path = run_dir / "validation_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Canonical smoke validation is required before a full run: {manifest_path}"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_summary = {
        "status": "complete",
        "run_id": SMOKE_RUN_ID,
        "scope": "subset",
        "subject_count": len(SMOKE_SUBJECTS),
        "complete_subject_count": len(SMOKE_SUBJECTS),
        "combination_count": len(SMOKE_SUBJECTS) * len(SMOOTHINGS) * len(HEMIS),
    }
    for key, expected_value in expected_summary.items():
        if payload.get(key) != expected_value:
            raise ValueError(
                f"Canonical smoke validation {key} mismatch: "
                f"{payload.get(key)!r} != {expected_value!r}"
            )
    if payload.get("template_manifest") != dict(template_fingerprint):
        raise ValueError("Canonical smoke template fingerprint drift")
    if payload.get("roi_registry") != dict(roi_registry_fingerprint):
        raise ValueError("Canonical smoke ROI registry fingerprint drift")
    _validate_fingerprint(payload.get("template_manifest"), context="smoke template")
    _validate_fingerprint(payload.get("roi_registry"), context="smoke ROI registry")
    _validate_fingerprint(payload.get("run_manifest"), context="smoke run manifest")
    _validate_fingerprint(payload.get("validation_items"), context="smoke validation table")
    run_record = payload["run_manifest"]
    run_payload = json.loads(Path(str(run_record["path"])).read_text(encoding="utf-8"))
    if (
        run_payload.get("status") != "complete"
        or run_payload.get("run_id") != SMOKE_RUN_ID
        or run_payload.get("scope") != "subset"
        or run_payload.get("subjects")
        != [f"sub-{subject}" for subject in SMOKE_SUBJECTS]
        or run_payload.get("subject_count") != len(SMOKE_SUBJECTS)
        or run_payload.get("complete_count") != len(SMOKE_SUBJECTS)
        or run_payload.get("failed_count") != 0
        or run_payload.get("template_manifest") != dict(template_fingerprint)
        or run_payload.get("roi_registry") != dict(roi_registry_fingerprint)
    ):
        raise ValueError("Canonical smoke run manifest contract mismatch")
    path_roots = run_payload.get("path_roots")
    parameters = run_payload.get("parameters")
    if not isinstance(path_roots, Mapping) or not isinstance(parameters, Mapping):
        raise ValueError("Canonical smoke dependency roots/parameters are missing")
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("Canonical smoke validation items are missing")
    actual_keys: set[tuple[str, str, str]] = set()
    for row in items:
        if not isinstance(row, Mapping) or row.get("status") != "ok":
            raise ValueError("Canonical smoke contains a non-ok combination")
        key = (
            str(row.get("subject")),
            str(row.get("smoothing")),
            str(row.get("hemi")),
        )
        if key in actual_keys:
            raise ValueError(f"Duplicate canonical smoke combination: {key}")
        actual_keys.add(key)
        counts = row.get("label_counts")
        if not isinstance(counts, Mapping) or any(
            int(counts.get(str(label), 0)) <= 0 for label in range(1, K + 1)
        ):
            raise ValueError(f"Canonical smoke label coverage failed: {key}")
    expected_keys = {
        (f"sub-{subject}", smoothing, hemi)
        for subject in SMOKE_SUBJECTS
        for smoothing in SMOOTHINGS
        for hemi in HEMIS
    }
    if actual_keys != expected_keys:
        raise ValueError("Canonical smoke subject/combination set mismatch")
    dependency_template = template
    if dependency_template is None:
        template_root = path_roots.get("template_root")
        if not isinstance(template_root, str):
            raise ValueError("Canonical smoke template dependency is missing")
        dependency_template = _load_template_contract(Path(template_root))
    _validate_validation_payload_dependencies(
        payload=payload,
        expected_run_manifest=run_record,
        template_fingerprint=template_fingerprint,
        roi_registry_fingerprint=roi_registry_fingerprint,
        out_root=out_root,
        template=dependency_template,
    )
    return fingerprint(manifest_path)


def _archive_validation_interrupted_paths(
    directory: Path,
    paths: Sequence[Path],
) -> None:
    existing = [Path(path) for path in paths if Path(path).exists()]
    if not existing:
        return
    pattern = re.compile(
        r"validation_(?:items|manifest)\.attempt-(?P<number>[0-9]{3,})"
        r"\.interrupted\.(?:tsv|json)"
    )
    attempts = [
        int(match.group("number"))
        for path in directory.iterdir()
        if (match := pattern.fullmatch(path.name)) is not None
    ]
    attempt_number = max(attempts, default=0) + 1
    for path in existing:
        suffix = ".tsv" if path.suffix == ".tsv" else ".json"
        stem = "validation_items" if suffix == ".tsv" else "validation_manifest"
        history = directory / (
            f"{stem}.attempt-{attempt_number:03d}.interrupted{suffix}"
        )
        while history.exists():
            attempt_number += 1
            history = directory / (
                f"{stem}.attempt-{attempt_number:03d}.interrupted{suffix}"
            )
        path.rename(history)


def prepare_validation_resume_state(
    *,
    run_dir: Path,
    expected_run_manifest: Mapping[str, object],
    template_fingerprint: Mapping[str, object],
    roi_registry_fingerprint: Mapping[str, object],
    resume: bool,
    complete_validator: Callable[[Mapping[str, object]], None] | None = None,
) -> dict[str, object] | None:
    directory = Path(run_dir)
    manifest = directory / "validation_manifest.json"
    table_partial = directory / "validation_items.partial.tsv"
    manifest_partial = directory / "validation_manifest.partial.json"
    table_final = directory / "validation_items.tsv"
    partial_payload: dict[str, object] | None = None
    active = [path for path in (table_partial, manifest_partial, table_final, manifest) if path.exists()]
    if active and not resume:
        raise FileExistsError(
            f"Retained validation artifacts require --resume: {active}"
        )
    if manifest_partial.exists() and not manifest.exists():
        try:
            raw_partial = json.loads(manifest_partial.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            _archive_validation_interrupted_paths(
                directory,
                [table_partial, manifest_partial],
            )
            return None
        if not isinstance(raw_partial, dict):
            _archive_validation_interrupted_paths(
                directory,
                [table_partial, manifest_partial],
            )
            return None
        partial_payload = raw_partial
    if manifest.exists():
        if not resume:
            raise FileExistsError(
                f"Validation output exists; use --resume for exact reuse: {manifest}"
            )
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("status") != "complete":
            raise ValueError(f"Validation manifest is not reusable complete output: {manifest}")
        if payload.get("run_manifest") != dict(expected_run_manifest):
            raise ValueError("Validation run-manifest fingerprint drift")
        if payload.get("template_manifest") != dict(template_fingerprint):
            raise ValueError("Validation template fingerprint drift")
        if payload.get("roi_registry") != dict(roi_registry_fingerprint):
            raise ValueError("Validation ROI registry fingerprint drift")
        _validate_fingerprint(payload.get("run_manifest"), context="validation run manifest")
        _validate_fingerprint(payload.get("template_manifest"), context="validation template")
        _validate_fingerprint(payload.get("roi_registry"), context="validation ROI registry")
        _validate_fingerprint(payload.get("validation_items"), context="validation table")
        if complete_validator is not None:
            complete_validator(payload)
        _archive_validation_interrupted_paths(
            directory,
            [table_partial, manifest_partial],
        )
        return payload
    active = [path for path in (table_partial, manifest_partial, table_final) if path.exists()]
    if not active:
        return None
    if not resume:
        raise FileExistsError(
            f"Interrupted validation artifacts are retained; use --resume: {active}"
        )
    if partial_payload is not None and table_final.exists():
        payload = partial_payload
        if not isinstance(payload, dict) or payload.get("status") != "complete":
            raise ValueError(f"Validation partial manifest is not complete: {manifest_partial}")
        if payload.get("run_manifest") != dict(expected_run_manifest):
            raise ValueError("Validation partial run-manifest fingerprint drift")
        if payload.get("template_manifest") != dict(template_fingerprint):
            raise ValueError("Validation partial template fingerprint drift")
        if payload.get("roi_registry") != dict(roi_registry_fingerprint):
            raise ValueError("Validation partial ROI registry fingerprint drift")
        if payload.get("validation_items") != fingerprint(table_final):
            raise ValueError("Validation partial table fingerprint drift")
        _validate_fingerprint(payload.get("run_manifest"), context="validation partial run manifest")
        _validate_fingerprint(payload.get("template_manifest"), context="validation partial template")
        _validate_fingerprint(payload.get("roi_registry"), context="validation partial ROI registry")
        _validate_fingerprint(payload.get("validation_items"), context="validation partial table")
        if complete_validator is not None:
            complete_validator(payload)
        _publish_partial(manifest_partial, manifest)
        return payload
    if table_partial.exists() and table_final.exists():
        raise FileExistsError("Validation has both partial and final tables")
    _archive_validation_interrupted_paths(
        directory,
        [table_partial, manifest_partial, table_final],
    )
    return None


def publish_validation_artifacts(
    *,
    run_dir: Path,
    rows: Sequence[Mapping[str, object]],
    payload_base: Mapping[str, object],
) -> dict[str, object]:
    directory = Path(run_dir)
    table_partial = directory / "validation_items.partial.tsv"
    table_final = directory / "validation_items.tsv"
    manifest_partial = directory / "validation_manifest.partial.json"
    manifest_final = directory / "validation_manifest.json"
    for path in (table_partial, table_final, manifest_partial, manifest_final):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite validation artifact: {path}")
    _write_validation_tsv(table_partial, rows)
    _publish_partial(table_partial, table_final)
    payload = {**dict(payload_base), "validation_items": fingerprint(table_final)}
    _write_json(manifest_partial, payload)
    _publish_partial(manifest_partial, manifest_final)
    return payload


if __name__ == "__main__":
    raise SystemExit(main())

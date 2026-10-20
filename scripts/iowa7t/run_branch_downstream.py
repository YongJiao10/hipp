#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.gifti import GiftiDataArray, GiftiImage


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_WORKFLOW_DIR = REPO_ROOT / "scripts" / "core_workflow"
if str(CORE_WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_WORKFLOW_DIR))

from lib.extract_cortex_merged_network_timeseries import extract_structure_data, load_required_mask
from lib.extract_cortex_kong2022_route_a_roi_timeseries import (
    ROUTE_A_SOURCE_TAG,
    TSNR_THRESHOLD,
    aggregate_hemi_roi_timeseries,
    align_surface_labels_to_dt_vertices,
    compute_cortex_tsnr_gate,
    load_surface_label_vector,
)
from lib.fc_math import compact_active_vertices, corrcoef_rows
from lib.kong2022_label_utils import (
    KONG2022_DLABEL_PATH,
    KONG2022_MERGE_CONFIG,
    canonical_families_7_order,
    canonical_families_17_order,
    load_kong2022_bundle,
)
from lib.surface_sampling import sample_hipp_surface_timeseries
from lib.surface_coordinates import dataset_registration, require_registered_sampling
from lib.tsnr import rest_tsnr_input, REST_POLICY
from lib.task_quality import rest_signal_validity
from lib.workbench import run_workbench
from lib.output_layout import (
    IOWA7T_VARIANTS,
    resolve_iowa7t_data_root,
    resolve_iowa7t_hipp_outputs_root,
    validate_iowa7t_variant,
)
from lib.io_utils import load_run_segment_metadata, parse_run_segments, write_tsv
from lib.path_templates import format_subject_path_template
from lib.workflow_utils import file_meta, file_meta_unresolved, git_commit, utc_now_iso, write_json


DEFAULT_VARIANT = "36p"
DEFAULT_DATA_ROOT = resolve_iowa7t_data_root(variant=DEFAULT_VARIANT, require_exists=False)
DEFAULT_OUTPUT_ROOT = resolve_iowa7t_hipp_outputs_root(variant=DEFAULT_VARIANT, require_exists=False)
DEFAULT_36P_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
SUPPORTED_BRANCHES = {"multiband", "multi_echo"}
HEMIS = ["L", "R"]
SMOOTHINGS = ["2mm", "4mm"]
ATLAS = "kong2022"
COMMON_REFERENCE_SLUG = "common"
MAX_CONCURRENCY = 10


@dataclass(frozen=True)
class BranchJob:
    subject: str
    branch: str
    variant: str
    data_root: Path
    output_root: Path
    subject_root: Path
    branch_root: Path
    hippunfold_batch_root: Path
    hippunfold_dir: Path
    bold_volume: Path
    dtseries: Path
    surface_root: Path
    cortex_root: Path
    cortex_common_root: Path
    cortex_kong2022_root: Path
    fc_root: Path
    areal_mshbm_label_dir: Path

    tsnr_input_template: str | None = None


def normalize_subject(subject: str) -> str:
    return str(subject).strip().replace("sub-", "")


def resolve_branch_job(
    subject: str,
    branch: str,
    *,
    variant: str = DEFAULT_VARIANT,
    data_root: Path | None = None,
    output_root: Path | None = None,
    hippunfold_batch_root: Path | None = None,
    default_36p_output_root: Path | None = None,
) -> BranchJob:
    subject = normalize_subject(subject)
    branch = str(branch).strip()
    if branch not in SUPPORTED_BRANCHES:
        raise ValueError(f"Unsupported Iowa7T branch {branch!r}; expected one of {sorted(SUPPORTED_BRANCHES)}")

    variant_key = validate_iowa7t_variant(variant)
    data_root = Path(data_root) if data_root is not None else resolve_iowa7t_data_root(variant=variant_key, require_exists=False)
    output_root = Path(output_root) if output_root is not None else resolve_iowa7t_hipp_outputs_root(variant=variant_key, require_exists=False)
    if default_36p_output_root is not None:
        # Retained for backward-compatible callers; NoXCPD now defaults to its own subject tree.
        Path(default_36p_output_root)
    sub = f"sub-{subject}"
    subject_root = output_root / "subjects" / sub
    branch_root = subject_root / branch
    func_dir = data_root / sub / "func"
    if branch == "multiband":
        desc = "NoXCPD" if variant_key == "noxcpd" else "denoised"
        bold_volume = func_dir / f"{sub}_task-rest_dir-concat_space-MNI152NLin6Asym_res-2_desc-{desc}_bold.nii.gz"
        dtseries = func_dir / f"{sub}_task-rest_dir-concat_space-fsLR_den-91k_desc-{desc}_bold.dtseries.nii"
    else:
        multi_echo_desc_by_variant = {
            "0p": "MEICA_XCPD0P_denoised",
            "22p": "MEICA_XCPD22P_denoised",
            "32p": "MEICA_XCPD32P_denoised",
            "36p": "MEICA_XCPD36P_denoised",
            "noxcpd": "MEICA_NoXCPD",
        }
        desc = multi_echo_desc_by_variant[variant_key]
        bold_volume = func_dir / f"{sub}_task-rest_run-concat_space-MNI152NLin6Asym_res-2_desc-{desc}_bold.nii.gz"
        dtseries = func_dir / f"{sub}_task-rest_run-concat_space-fsLR_den-91k_desc-{desc}_bold.dtseries.nii"

    if hippunfold_batch_root is not None:
        resolved_hippunfold_batch_root = Path(hippunfold_batch_root)
    else:
        resolved_hippunfold_batch_root = output_root / "subjects"

    cortex_root = branch_root / "cortex"
    hippunfold_dir = resolved_hippunfold_batch_root / sub / "hippunfold"
    if not hippunfold_dir.is_symlink() and not hippunfold_dir.exists() and variant_key != "36p":
        shared_36p = (
            resolve_iowa7t_hipp_outputs_root(variant="36p", require_exists=False)
            / "subjects"
            / sub
            / "hippunfold"
        )
        if shared_36p.exists():
            rel_target = Path("../../../../DataOrg_PFM_36P/hipp_outputs/subjects") / sub / "hippunfold"
            hippunfold_dir.parent.mkdir(parents=True, exist_ok=True)
            hippunfold_dir.symlink_to(rel_target)

    return BranchJob(
        subject=subject,
        branch=branch,
        variant=variant_key,
        data_root=data_root,
        output_root=output_root,
        subject_root=subject_root,
        branch_root=branch_root,
        hippunfold_batch_root=resolved_hippunfold_batch_root,
        hippunfold_dir=hippunfold_dir,
        bold_volume=bold_volume,
        dtseries=dtseries,
        surface_root=branch_root / "surface",
        cortex_root=cortex_root,
        cortex_common_root=cortex_root / COMMON_REFERENCE_SLUG,
        cortex_kong2022_root=cortex_root / ATLAS,
        fc_root=branch_root / "fc" / ATLAS,
        areal_mshbm_label_dir=cortex_root / "areal_mshbm",
    )


def hippunfold_surface_path(job: BranchJob, hemi: str) -> Path:
    return (
        job.hippunfold_dir
        / f"sub-{job.subject}"
        / "surf"
        / f"sub-{job.subject}_hemi-{hemi}_space-T1w_den-512_label-hipp_midthickness.surf.gii"
    )


def require_inputs(job: BranchJob) -> None:
    required = [
        job.bold_volume,
        job.dtseries,
        job.hippunfold_dir / f"sub-{job.subject}" / "surf" / f"sub-{job.subject}_den-512_surfaces.spec",
        job.areal_mshbm_label_dir / "lh_labels.txt",
        job.areal_mshbm_label_dir / "rh_labels.txt",
    ]
    required.extend(hippunfold_surface_path(job, hemi) for hemi in HEMIS)
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing required Iowa7T branch inputs:\n" + "\n".join(missing))


def load_func_gii(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    arr = np.asarray(img.agg_data(), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[0] != 512 and arr.shape[1] == 512:
        arr = arr.T
    return arr.astype(np.float32, copy=False)


def make_shape_gifti(data: np.ndarray) -> GiftiImage:
    return GiftiImage(
        darrays=[
            GiftiDataArray(
                data=np.asarray(data, dtype=np.float32),
                intent="NIFTI_INTENT_SHAPE",
                datatype="NIFTI_TYPE_FLOAT32",
            )
        ]
    )


def raw_outputs_done(job: BranchJob) -> bool:
    raw_dir = job.surface_root / "raw"
    summary = raw_dir / 'surface_sampling_summary.json'
    if summary.is_file():
        require_registered_sampling(json.loads(summary.read_text()), context=summary)
    return all(
        (raw_dir / f"sub-{job.subject}_hemi-{hemi}_space-T1w_den-512_label-hipp_bold.func.gii").exists()
        for hemi in HEMIS
    )


def tsnr_outputs_done(job: BranchJob) -> bool:
    gate_dir = job.surface_root / "tsnr_gate"
    return all((gate_dir / f"sub-{job.subject}_hemi-{hemi}_valid_mask.npy").exists() for hemi in HEMIS)


def smoothing_outputs_done(job: BranchJob) -> bool:
    stage = job.surface_root / 'stage_manifest.json'
    if not stage.is_file() or json.loads(stage.read_text()).get('params', {}).get('smoothing_space') != 'T1w':
        return False
    return all(
        (job.surface_root / smoothing / f"sub-{job.subject}_hemi-{hemi}_timeseries.npy").exists()
        for smoothing in SMOOTHINGS
        for hemi in HEMIS
    )


def qc_input(job, hemi: str, domain: str) -> Path:
    if job.tsnr_input_template is None:
        raise ValueError("Iowa7T requires --tsnr-input-template: explicit baseline samples per acquisition/run")
    path = Path(job.tsnr_input_template.format(subject=job.subject, hemi=hemi, domain=domain,
                variant=job.variant, branch=getattr(job, "branch", getattr(job, "kind", "")),
                run=getattr(job, "run_label", "")))
    if not path.is_file():
        raise FileNotFoundError(f"Missing Iowa7T baseline tSNR input: {path}")
    return path


def cortex_qc_done(directory: Path) -> bool:
    for hemi in ("left", "right"):
        path = directory / "tsnr_gate" / f"cortex_{hemi}_tsnr_gate_summary.json"
        if not path.is_file() or json.loads(path.read_text()).get("validity_policy") != REST_POLICY:
            return False
    return True


def qc_policy_done(job) -> bool:
    for hemi in HEMIS:
        path = job.surface_root / "tsnr_gate" / f"sub-{job.subject}_hemi-{hemi}_tsnr_gate_summary.json"
        if not path.is_file():
            return False
        summary = json.loads(path.read_text())
        if summary.get('validity_policy') != REST_POLICY or summary.get('sampling_space') != 'MNI152NLin6Asym':
            return False
    return True


def run_surface_stage(job: BranchJob, *, overwrite: bool = False) -> None:
    for hemi in HEMIS:
        qc_input(job, hemi, "hipp")
    require_inputs(job)
    t0 = time.monotonic()
    stage_inputs: list[dict[str, object]] = []
    stage_outputs: list[dict[str, object]] = []

    raw_dir = job.surface_root / "raw"
    if overwrite or not raw_outputs_done(job):
        raw_dir.mkdir(parents=True, exist_ok=True)
        summary = sample_hipp_surface_timeseries(
            bold=job.bold_volume,
            hippunfold_dir=job.hippunfold_dir,
            subject=job.subject,
            density="512",
            outdir=raw_dir,
            space="T1w",
            registration=dataset_registration('iowa7t', job.subject),
            mapping_method="trilinear",
            smooth_iters=0,
        )
        write_json(raw_dir / "surface_sampling_summary.json", summary)
        write_json(
            raw_dir / "stage_manifest.json",
            {
                "stage": "iowa7t_branch_raw_surface_sampling",
                "status": "done",
                "timestamp_utc": utc_now_iso(),
                "git_commit": git_commit(),
                "params": {"subject": job.subject, "branch": job.branch, "density": "512"},
                "inputs": [file_meta(job.bold_volume)],
                "outputs": [file_meta(p) for p in raw_dir.glob("*") if p.is_file()],
                "elapsed_seconds": float(time.monotonic() - t0),
            },
        )

    gate_dir = job.surface_root / "tsnr_gate"
    qc_changed = not qc_policy_done(job)
    if overwrite or not tsnr_outputs_done(job) or not qc_policy_done(job):
        gate_dir.mkdir(parents=True, exist_ok=True)
        for hemi in HEMIS:
            raw = raw_dir / f"sub-{job.subject}_hemi-{hemi}_space-T1w_den-512_label-hipp_bold.func.gii"
            surf = hippunfold_surface_path(job, hemi)
            data = load_func_gii(raw).astype(np.float64)
            scores, valid = rest_tsnr_input(qc_input(job, hemi, "hipp"), dataset="iowa7t", subject=job.subject, hemi=hemi, n_vertices=data.shape[0], n_frames=data.shape[1])
            valid &= rest_signal_validity(data)
            tsnr = np.min(scores, axis=0)
            invalid = ~valid
            np.save(gate_dir / f"sub-{job.subject}_hemi-{hemi}_tsnr.npy", tsnr)
            np.save(gate_dir / f"sub-{job.subject}_hemi-{hemi}_valid_mask.npy", valid.astype(bool))
            np.save(gate_dir / f"sub-{job.subject}_hemi-{hemi}_invalid_initial_mask.npy", invalid.astype(bool))
            nib.save(
                make_shape_gifti(valid.astype(np.float32)),
                str(gate_dir / f"sub-{job.subject}_hemi-{hemi}_valid_roi.shape.gii"),
            )
            write_json(
                gate_dir / f"sub-{job.subject}_hemi-{hemi}_tsnr_gate_summary.json",
                {
                    "subject": job.subject,
                    "branch": job.branch,
                    "hemi": hemi,
                    "threshold": 25.0,
                    "tsnr_definition": REST_POLICY,
                    "baseline_input": str(qc_input(job, hemi, "hipp")),
                    "validity_policy": REST_POLICY,
                    "sampling_space": "MNI152NLin6Asym",
                    "n_vertices_total": int(valid.size),
                    "n_vertices_valid": int(valid.sum()),
                    "n_vertices_invalid": int(invalid.sum()),
                    "invalid_fraction": float(invalid.sum() / max(1, valid.size)),
                },
            )
            stage_inputs.extend([file_meta(raw), file_meta(surf)])
            stage_outputs.extend(file_meta(p) for p in gate_dir.glob(f"sub-{job.subject}_hemi-{hemi}_*") if p.is_file())

    if overwrite or not smoothing_outputs_done(job) or qc_changed:
        for smoothing in SMOOTHINGS:
            fwhm = float(smoothing.replace("mm", ""))
            out_dir = job.surface_root / smoothing
            out_dir.mkdir(parents=True, exist_ok=True)
            for hemi in HEMIS:
                raw = raw_dir / f"sub-{job.subject}_hemi-{hemi}_space-T1w_den-512_label-hipp_bold.func.gii"
                roi = gate_dir / f"sub-{job.subject}_hemi-{hemi}_valid_roi.shape.gii"
                surf = hippunfold_surface_path(job, hemi)
                out_gii = out_dir / f"sub-{job.subject}_hemi-{hemi}_space-T1w_den-512_label-hipp_bold.func.gii"
                run_workbench(
                    [
                        "-metric-smoothing",
                        str(surf),
                        str(raw),
                        str(fwhm),
                        str(out_gii),
                        "-fwhm",
                        "-roi",
                        str(roi),
                    ],
                    env={},
                )
                arr = load_func_gii(out_gii)
                np.save(out_dir / f"sub-{job.subject}_hemi-{hemi}_timeseries.npy", arr.astype(np.float32))
                stage_inputs.extend([file_meta(raw), file_meta(roi), file_meta(surf)])
                stage_outputs.extend(
                    [file_meta(out_gii), file_meta(out_dir / f"sub-{job.subject}_hemi-{hemi}_timeseries.npy")]
                )

    write_json(
        job.surface_root / "stage_manifest.json",
        {
            "stage": "iowa7t_branch_surface",
            "status": "done",
            "timestamp_utc": utc_now_iso(),
            "git_commit": git_commit(),
            "params": {"subject": job.subject, "branch": job.branch, "smoothings": SMOOTHINGS,
                       "smoothing_space": "T1w"},
            "inputs": stage_inputs,
            "outputs": stage_outputs,
            "elapsed_seconds": float(time.monotonic() - t0),
        },
    )


def reference_outputs_done(job: BranchJob) -> bool:
    return all(
        path.exists()
        for path in [
            job.cortex_kong2022_root / "reference_summary.json",
            job.cortex_kong2022_root / "cortex_roi_parcels.tsv",
            job.cortex_kong2022_root / "cortex_roi_parcel_timeseries.npy",
            job.cortex_kong2022_root / "cortex_canonical7_networks.tsv",
            job.cortex_kong2022_root / "cortex_canonical7_network_timeseries.npy",
            job.cortex_kong2022_root / "cortex_family17_network_timeseries.npy",
            job.cortex_common_root / "runs" / "run-1" / "reference_summary.json",
        ]
    )


def fixed_400_roi_reference(
    *,
    roi_registry: list[dict[str, object]],
    left_ts: list[np.ndarray],
    left_rows: list[dict[str, object]],
    right_ts: list[np.ndarray],
    right_rows: list[dict[str, object]],
    n_timepoints: int,
    left_empty: list[dict[str, object]] | None = None,
    right_empty: list[dict[str, object]] | None = None,
) -> tuple[np.ndarray, list[dict[str, object]], list[int]]:
    row_by_id: dict[int, dict[str, object]] = {}
    ts_by_id: dict[int, np.ndarray] = {}
    for ts, row in zip(left_ts, left_rows, strict=True):
        roi_id = int(row["parcel_id"])
        row_by_id[roi_id] = dict(row)
        ts_by_id[roi_id] = np.asarray(ts, dtype=np.float32)
    for ts, row in zip(right_ts, right_rows, strict=True):
        roi_id = int(row["parcel_id"])
        row_by_id[roi_id] = dict(row)
        ts_by_id[roi_id] = np.asarray(ts, dtype=np.float32)

    for row in list(left_empty or []) + list(right_empty or []):
        roi_id = int(row["parcel_id"])
        row_by_id.setdefault(roi_id, dict(row))

    rows: list[dict[str, object]] = []
    ts_rows: list[np.ndarray] = []
    missing_roi_ids: list[int] = []
    for registry_row in sorted(roi_registry, key=lambda r: int(r["roi_id"])):
        roi_id = int(registry_row["roi_id"])
        source_row = dict(row_by_id.get(roi_id, {}))
        row = {
            "parcel_id": roi_id,
            "parcel_name": str(source_row.get("parcel_name", registry_row.get("roi_name", f"roi_{roi_id}"))),
            "family17": str(source_row.get("family17", registry_row.get("family17", ""))),
            "canonical7": str(source_row.get("canonical7", registry_row.get("canonical7", ""))),
            "canonical8": str(source_row.get("canonical8", registry_row.get("canonical8", ""))),
            "hemisphere": str(source_row.get("hemisphere", registry_row.get("hemi", ""))),
            "subregion": str(source_row.get("subregion", registry_row.get("subregion", ""))),
            "rank": int(source_row.get("rank", registry_row.get("rank", 0))),
            "n_grayordinates_total": int(source_row.get("n_grayordinates_total", 0)),
            "n_grayordinates_used": int(source_row.get("n_grayordinates_used", 0)),
            "excluded_by_tsnr_gate": bool(source_row.get("excluded_by_tsnr_gate", True)),
        }
        has_timeseries = roi_id in ts_by_id and int(row["n_grayordinates_used"]) > 0
        row["excluded_by_no_vertices"] = bool(int(row["n_grayordinates_total"]) <= 0)
        row["included_in_timeseries"] = bool(has_timeseries)
        if has_timeseries:
            ts = np.asarray(ts_by_id[roi_id], dtype=np.float32).reshape(-1)
            if int(ts.shape[0]) != int(n_timepoints):
                raise ValueError(f"ROI {roi_id} timepoint mismatch: {ts.shape[0]} vs {n_timepoints}")
            ts_rows.append(ts)
        else:
            missing_roi_ids.append(roi_id)
            ts_rows.append(np.full((int(n_timepoints),), np.nan, dtype=np.float32))
        rows.append(row)

    return np.asarray(ts_rows, dtype=np.float32), rows, missing_roi_ids


def read_roi_metadata(path: Path) -> tuple[list[str], list[int], list[int]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        names: list[str] = []
        roi_ids: list[int] = []
        missing_roi_ids: list[int] = []
        for row in reader:
            roi_id = int(row["parcel_id"])
            value = row.get("parcel_name") or row.get("roi_name")
            if not value:
                raise ValueError(f"Missing parcel_name/roi_name in {path}")
            names.append(str(value))
            roi_ids.append(roi_id)
            if str(row.get("included_in_timeseries", "True")).strip().lower() in {"false", "0", "no"}:
                missing_roi_ids.append(roi_id)
        return names, roi_ids, missing_roi_ids


def mark_missing_roi_fc_columns(fc: np.ndarray, *, missing_roi_ids: list[int], roi_ids: list[int]) -> np.ndarray:
    out = np.asarray(fc, dtype=np.float32).copy()
    col_by_id = {int(roi_id): idx for idx, roi_id in enumerate(roi_ids)}
    for roi_id in missing_roi_ids:
        idx = col_by_id.get(int(roi_id))
        if idx is not None:
            out[:, idx] = np.nan
    return out


def run_reference_stage(
    job: BranchJob,
    *,
    overwrite: bool = False,
    run_metadata_path: Path | None = None,
) -> None:
    require_inputs(job)
    if run_metadata_path is None:
        raise ValueError(
            "Iowa7T reference stage requires an explicit run metadata path"
        )
    run_metadata_path = Path(run_metadata_path).resolve()
    run_ids, run_lengths = load_run_segment_metadata(run_metadata_path)
    if not overwrite and reference_outputs_done(job) and cortex_qc_done(job.cortex_common_root):
        return

    outdir = job.cortex_kong2022_root
    common_ref_dir = job.cortex_common_root

    lh_labels_path = job.areal_mshbm_label_dir / "lh_labels.txt"
    rh_labels_path = job.areal_mshbm_label_dir / "rh_labels.txt"
    lh_full = load_surface_label_vector(lh_labels_path)
    rh_full = load_surface_label_vector(rh_labels_path)

    bundle = load_kong2022_bundle(
        dlabel_path=Path(KONG2022_DLABEL_PATH).resolve(),
        config_path=Path(KONG2022_MERGE_CONFIG).resolve(),
    )
    roi_registry = list(bundle["roi_registry"])
    canonical7_order = list(bundle["canonical7_order"])
    family17_order = canonical_families_17_order()

    dt_img = nib.load(str(job.dtseries))
    dt_data = np.asarray(dt_img.dataobj, dtype=np.float32)
    if dt_data.ndim != 2:
        raise ValueError(f"Expected 2D dtseries data, got shape={dt_data.shape}")
    dt_data_t = dt_data.T
    n_timepoints = int(dt_data.shape[0])
    run_segments = parse_run_segments(run_lengths, n_timepoints, run_ids=run_ids)
    outdir.mkdir(parents=True, exist_ok=True)
    common_ref_dir.mkdir(parents=True, exist_ok=True)
    dt_axis = dt_img.header.get_axis(1)

    left_vertices, left_dt = extract_structure_data(dt_axis, dt_data_t, "CORTEX_LEFT")
    right_vertices, right_dt = extract_structure_data(dt_axis, dt_data_t, "CORTEX_RIGHT")
    left_parcel_ids = align_surface_labels_to_dt_vertices(lh_full, left_vertices, hemi="L")
    right_parcel_ids = align_surface_labels_to_dt_vertices(rh_full, right_vertices, hemi="R")

    tsnr_dir = common_ref_dir / "tsnr_gate"
    left_tsnr_mask = compute_cortex_tsnr_gate(data=left_dt, outdir=tsnr_dir, hemi="left", threshold=TSNR_THRESHOLD, dataset="iowa7t", subject=job.subject, tsnr_input=qc_input(job, "L", "cortex"), vertex_ids=left_vertices)
    right_tsnr_mask = compute_cortex_tsnr_gate(data=right_dt, outdir=tsnr_dir, hemi="right", threshold=TSNR_THRESHOLD, dataset="iowa7t", subject=job.subject, tsnr_input=qc_input(job, "R", "cortex"), vertex_ids=right_vertices)
    left_valid = load_required_mask(str(left_tsnr_mask), int(left_vertices.shape[0]), "left cortex tSNR")
    right_valid = load_required_mask(str(right_tsnr_mask), int(right_vertices.shape[0]), "right cortex tSNR")

    left_ts, left_rows, left_empty, left_stats = aggregate_hemi_roi_timeseries(
        hemi="L",
        hemi_dt=left_dt,
        aligned_parcel_ids=left_parcel_ids,
        valid_mask=left_valid,
        roi_registry=roi_registry,
    )
    right_ts, right_rows, right_empty, right_stats = aggregate_hemi_roi_timeseries(
        hemi="R",
        hemi_dt=right_dt,
        aligned_parcel_ids=right_parcel_ids,
        valid_mask=right_valid,
        roi_registry=roi_registry,
    )

    parcel_ts, roi_rows, missing_roi_ids = fixed_400_roi_reference(
        roi_registry=roi_registry,
        left_ts=left_ts,
        left_rows=left_rows,
        right_ts=right_ts,
        right_rows=right_rows,
        left_empty=left_empty,
        right_empty=right_empty,
        n_timepoints=n_timepoints,
    )
    kept_rows = [row for row in roi_rows if bool(row["included_in_timeseries"])]
    empty_rows = [row for row in roi_rows if not bool(row["included_in_timeseries"])]
    if not kept_rows:
        raise RuntimeError(f"No Kong2022 ROI timeseries remained after tSNR gate for sub-{job.subject} {job.branch}.")

    np.save(outdir / "cortex_roi_parcel_timeseries.npy", parcel_ts)

    roi_tsv_fields = [
        "parcel_id",
        "parcel_name",
        "family17",
        "canonical7",
        "canonical8",
        "hemisphere",
        "subregion",
        "rank",
        "n_grayordinates_total",
        "n_grayordinates_used",
        "excluded_by_tsnr_gate",
        "excluded_by_no_vertices",
        "included_in_timeseries",
    ]
    write_tsv(outdir / "cortex_roi_parcels.tsv", roi_rows, roi_tsv_fields)
    write_tsv(outdir / "cortex_roi_parcels_empty_after_tsnr.tsv", empty_rows, roi_tsv_fields)

    canonical_rows: list[dict[str, object]] = []
    canonical_ts_rows: list[np.ndarray] = []
    for canonical in canonical7_order:
        idx = [i for i, row in enumerate(roi_rows) if str(row["canonical7"]) == canonical and bool(row["included_in_timeseries"])]
        if not idx:
            raise RuntimeError(f"Canonical network has zero parcels after tSNR gate: {canonical}")
        row_subset = [roi_rows[i] for i in idx]
        ts = np.nanmean(parcel_ts[np.asarray(idx, dtype=np.int32), :], axis=0).astype(np.float32)
        canonical_ts_rows.append(ts)
        canonical_rows.append(
            {
                "canonical7_network": canonical,
                "n_parcels_merged": int(len(idx)),
                "original_parent_networks": ",".join(sorted({str(r["family17"]) for r in row_subset})),
                "n_grayordinates_used_total": int(sum(int(r["n_grayordinates_used"]) for r in row_subset)),
            }
        )
    canonical_ts = np.asarray(canonical_ts_rows, dtype=np.float32)
    np.save(outdir / "cortex_canonical7_network_timeseries.npy", canonical_ts)
    write_tsv(
        outdir / "cortex_canonical7_networks.tsv",
        canonical_rows,
        ["canonical7_network", "n_parcels_merged", "original_parent_networks", "n_grayordinates_used_total"],
    )

    family17_rows: list[dict[str, object]] = []
    family17_ts_rows: list[np.ndarray] = []
    for family17 in family17_order:
        idx = [i for i, row in enumerate(roi_rows) if str(row["family17"]) == family17 and bool(row["included_in_timeseries"])]
        if not idx:
            raise RuntimeError(f"Family17 network has zero parcels after tSNR gate: {family17}")
        row_subset = [roi_rows[i] for i in idx]
        ts = np.nanmean(parcel_ts[np.asarray(idx, dtype=np.int32), :], axis=0).astype(np.float32)
        family17_ts_rows.append(ts)
        family17_rows.append(
            {
                "family17_network": family17,
                "n_parcels_merged": int(len(idx)),
                "canonical_parent_networks": ",".join(sorted({str(r["canonical7"]) for r in row_subset})),
                "n_grayordinates_used_total": int(sum(int(r["n_grayordinates_used"]) for r in row_subset)),
            }
        )
    family17_ts = np.asarray(family17_ts_rows, dtype=np.float32)
    np.save(outdir / "cortex_family17_network_timeseries.npy", family17_ts)
    write_tsv(
        outdir / "cortex_family17_networks.tsv",
        family17_rows,
        ["family17_network", "n_parcels_merged", "canonical_parent_networks", "n_grayordinates_used_total"],
    )

    for seg in run_segments:
        rid = int(seg["run_id"])
        run_dir = common_ref_dir / "runs" / f"run-{rid}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_summary = {
            "subject": job.subject,
            "branch": job.branch,
            "reference_scope": COMMON_REFERENCE_SLUG,
            "atlas_slug": ATLAS,
            "reference_label_source": ROUTE_A_SOURCE_TAG,
            "run_id": str(rid),
            "run_label": str(seg["run_label"]),
            "run_boundary_source": "explicit_run_metadata_template",
            "run_metadata": str(run_metadata_path),
            "run_start": int(seg["run_start"]),
            "run_stop": int(seg["run_stop"]),
            "run_length": int(seg["run_length"]),
            "n_timepoints": int(seg["run_length"]),
            "n_roi_parcels_total": int(parcel_ts.shape[0]),
            "n_roi_parcels_with_timeseries": int(len(kept_rows)),
            "n_roi_parcels_missing_timeseries": int(len(missing_roi_ids)),
            "n_canonical7_networks_used": int(canonical_ts.shape[0]),
            "network_taxonomy": "canonical7",
            "n_family17_networks_used": int(family17_ts.shape[0]),
            "roi_timeseries_source": str((outdir / "cortex_roi_parcel_timeseries.npy").resolve()),
            "tsnr_threshold": float(TSNR_THRESHOLD),
            "run_slice_policy": "read_time_slice_from_full_timeseries",
        }
        write_json(run_dir / "reference_summary.json", run_summary)
        write_json(
            run_dir / "stage_manifest.json",
            {
                "stage": "iowa7t_branch_reference_run_slice",
                "status": "done",
                "timestamp_utc": utc_now_iso(),
                "params": {
                    "subject": job.subject,
                    "branch": job.branch,
                    "atlas_slug": ATLAS,
                    "run_id": str(rid),
                    "run_metadata": str(run_metadata_path),
                },
                "inputs": [file_meta_unresolved((outdir / "cortex_roi_parcel_timeseries.npy").resolve()), file_meta_unresolved(run_metadata_path)],
                "outputs": [str((run_dir / "reference_summary.json").resolve())],
            },
        )

    summary = {
        "subject": job.subject,
        "branch": job.branch,
        "atlas_slug": ATLAS,
        "reference_scope": COMMON_REFERENCE_SLUG,
        "reference_label_source": ROUTE_A_SOURCE_TAG,
        "dtseries": str(job.dtseries),
        "cross_atlas_merge_json": str(Path(KONG2022_MERGE_CONFIG).resolve()),
        "route_a_label_dir": str(job.areal_mshbm_label_dir),
        "route_a_lh_labels": str(lh_labels_path),
        "route_a_rh_labels": str(rh_labels_path),
        "n_parcels_total": 400,
        "n_parcels_materialized": int(parcel_ts.shape[0]),
        "n_parcels_used_after_tsnr": int(len(kept_rows)),
        "n_parcels_empty_after_tsnr": int(len(empty_rows)),
        "missing_roi_ids": missing_roi_ids,
        "n_canonical7_networks_used": int(canonical_ts.shape[0]),
        "canonical7_networks_used": canonical7_order,
        "network_taxonomy": "canonical7",
        "n_family17_networks_used": int(family17_ts.shape[0]),
        "family17_networks_used": family17_order,
        "n_timepoints": int(n_timepoints),
        "tsnr_threshold": float(TSNR_THRESHOLD),
        "left_tsnr_mask": str(left_tsnr_mask),
        "right_tsnr_mask": str(right_tsnr_mask),
        "run_boundaries_dir": str((common_ref_dir / "runs").resolve()),
        "run_boundary_source": "explicit_run_metadata_template",
        "run_metadata": str(run_metadata_path),
        "tsnr_gate_stats": {
            "left": left_stats,
            "right": right_stats,
            "combined": {
                "n_grayordinates_total": int(left_stats["n_grayordinates_total"] + right_stats["n_grayordinates_total"]),
                "n_grayordinates_used": int(left_stats["n_grayordinates_used"] + right_stats["n_grayordinates_used"]),
                "n_grayordinates_masked": int(left_stats["n_grayordinates_masked"] + right_stats["n_grayordinates_masked"]),
            },
            "rois_empty_after_tsnr_gate": {
                "left": int(len(left_empty)),
                "right": int(len(right_empty)),
                "combined": int(len(empty_rows)),
            },
        },
    }
    write_json(outdir / "reference_summary.json", summary)
    write_json(
        outdir / "stage_manifest.json",
        {
            "stage": "iowa7t_branch_reference",
            "status": "done",
            "timestamp_utc": utc_now_iso(),
            "params": {
                "subject": job.subject,
                "branch": job.branch,
                "atlas_slug": ATLAS,
                "reference_label_source": ROUTE_A_SOURCE_TAG,
                "tsnr_threshold": float(TSNR_THRESHOLD),
                "run_lengths": [int(seg["run_length"]) for seg in run_segments],
                "run_ids": [int(seg["run_id"]) for seg in run_segments],
                "run_metadata": str(run_metadata_path),
                "n_parcels_materialized": int(parcel_ts.shape[0]),
                "n_parcels_with_timeseries": int(len(kept_rows)),
                "missing_roi_ids": missing_roi_ids,
            },
            "inputs": [
                file_meta_unresolved(job.dtseries),
                file_meta_unresolved(Path(KONG2022_MERGE_CONFIG).resolve()),
                file_meta_unresolved(lh_labels_path),
                file_meta_unresolved(rh_labels_path),
                file_meta_unresolved(left_tsnr_mask),
                file_meta_unresolved(right_tsnr_mask),
                file_meta_unresolved(run_metadata_path),
            ],
            "outputs": [
                str((outdir / "reference_summary.json").resolve()),
                str((outdir / "cortex_roi_parcels.tsv").resolve()),
                str((outdir / "cortex_roi_parcel_timeseries.npy").resolve()),
                str((outdir / "cortex_canonical7_networks.tsv").resolve()),
                str((outdir / "cortex_canonical7_network_timeseries.npy").resolve()),
                str((outdir / "cortex_family17_networks.tsv").resolve()),
                str((outdir / "cortex_family17_network_timeseries.npy").resolve()),
            ],
        },
    )


def read_roi_names(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        out = []
        for row in reader:
            value = row.get("parcel_name") or row.get("roi_name")
            if not value:
                raise ValueError(f"Missing parcel_name/roi_name in {path}")
            out.append(str(value))
        return out


def load_run_segments_direct(
    job: BranchJob,
    *,
    run_metadata_path: Path,
) -> list[dict[str, int]]:
    run_ids, run_lengths = load_run_segment_metadata(Path(run_metadata_path))
    runs_dir = job.cortex_common_root / "runs"
    segments: list[dict[str, int]] = []
    expected_start = 0
    for rid, expected_length in zip(run_ids, run_lengths):
        p = runs_dir / f"run-{rid}" / "reference_summary.json"
        if not p.is_file():
            raise FileNotFoundError(f"Missing run reference summary: {p}")
        summary = json.loads(p.read_text(encoding="utf-8"))
        actual_run_id = int(summary["run_id"])
        run_start = int(summary["run_start"])
        run_stop = int(summary["run_stop"])
        run_length = int(summary["run_length"])
        if actual_run_id != int(rid) or run_length != int(expected_length):
            raise ValueError(
                f"Run reference summary disagrees with metadata for run-{rid}: {p}"
            )
        if run_start != expected_start or run_stop != expected_start + int(expected_length):
            raise ValueError(
                f"Run reference boundaries are not contiguous for run-{rid}: {p}"
            )
        segments.append(
            {
                "run_id": int(rid),
                "run_start": run_start,
                "run_stop": run_stop,
                "run_length": run_length,
            }
        )
        expected_start = run_stop
    return segments


def fc_outputs_done(job: BranchJob) -> bool:
    return all(
        (job.fc_root / smoothing / f"hemi_{hemi}" / "fc_summary.json").exists()
        and json.loads((job.fc_root / smoothing / f"hemi_{hemi}" / "fc_summary.json").read_text()).get("validity_policy") == REST_POLICY
        and (job.fc_root / smoothing / f"hemi_{hemi}" / "hipp_vertex_to_network_fc.npy").exists()
        for smoothing in SMOOTHINGS
        for hemi in HEMIS
    )


def run_fc_stage(
    job: BranchJob,
    *,
    overwrite: bool = False,
    run_metadata_path: Path | None = None,
) -> None:
    if run_metadata_path is None:
        raise ValueError("Iowa7T FC stage requires an explicit run metadata path")
    run_metadata_path = Path(run_metadata_path).resolve()
    if not qc_policy_done(job) or not cortex_qc_done(job.cortex_common_root):
        raise ValueError("Rebuild Iowa7T surface and cortex with true baseline tSNR before FC")
    if not overwrite and fc_outputs_done(job):
        return
    if not reference_outputs_done(job):
        raise FileNotFoundError(f"Reference stage is incomplete for sub-{job.subject} {job.branch}: {job.cortex_root}")
    if not smoothing_outputs_done(job):
        raise FileNotFoundError(f"Surface smoothing stage is incomplete for sub-{job.subject} {job.branch}: {job.surface_root}")

    roi_ts = np.load(job.cortex_kong2022_root / "cortex_roi_parcel_timeseries.npy").astype(np.float32)
    roi_names, roi_ids, missing_roi_ids = read_roi_metadata(job.cortex_kong2022_root / "cortex_roi_parcels.tsv")
    if roi_ts.ndim != 2:
        raise ValueError(f"Expected 2D ROI timeseries, got {roi_ts.shape}")
    if roi_ts.shape[0] != len(roi_names):
        raise RuntimeError(f"ROI TSV mismatch for sub-{job.subject}: ts_rows={roi_ts.shape[0]} roi_names={len(roi_names)}")

    run_segments = load_run_segments_direct(job, run_metadata_path=run_metadata_path)
    for smoothing in SMOOTHINGS:
        for hemi in HEMIS:
            fc_dir = job.fc_root / smoothing / f"hemi_{hemi}"
            fc_dir.mkdir(parents=True, exist_ok=True)
            ts = np.load(job.surface_root / smoothing / f"sub-{job.subject}_hemi-{hemi}_timeseries.npy").astype(np.float32)
            valid_mask = np.load(job.surface_root / "tsnr_gate" / f"sub-{job.subject}_hemi-{hemi}_valid_mask.npy").astype(bool)
            if ts.ndim != 2:
                raise ValueError(f"Expected 2D hippocampal timeseries for sub-{job.subject} {smoothing} hemi-{hemi}: {ts.shape}")
            if ts.shape[1] != roi_ts.shape[1]:
                raise RuntimeError(
                    f"Time mismatch for sub-{job.subject} {job.branch} {smoothing} hemi-{hemi}: ts={ts.shape} roi={roi_ts.shape}"
                )
            pearson_valid = mark_missing_roi_fc_columns(
                corrcoef_rows(ts[valid_mask, :], roi_ts),
                missing_roi_ids=missing_roi_ids,
                roi_ids=roi_ids,
            )
            pearson_full = np.full((ts.shape[0], roi_ts.shape[0]), np.nan, dtype=np.float32)
            pearson_full[valid_mask, :] = pearson_valid
            np.save(fc_dir / "hipp_vertex_to_network_fc.npy", pearson_full.astype(np.float32))
            np.save(fc_dir / "hipp_vertex_to_network_fc_active.npy", compact_active_vertices(pearson_full, valid_mask))

            run_labels: list[str] = []
            run_lengths: list[int] = []
            for seg in run_segments:
                rid = int(seg["run_id"])
                run_start = int(seg["run_start"])
                run_stop = int(seg["run_stop"])
                run_len = int(seg["run_length"])
                run_labels.append(f"run-{rid}")
                run_lengths.append(run_len)
                run_ts = ts[:, run_start:run_stop].astype(np.float32, copy=False)
                run_roi = roi_ts[:, run_start:run_stop].astype(np.float32, copy=False)
                if run_ts.shape[1] != run_len or run_roi.shape[1] != run_len:
                    raise RuntimeError(
                        f"Run slice mismatch for sub-{job.subject} {job.branch} {smoothing} hemi-{hemi} run-{rid}"
                    )
                run_dir = fc_dir / "runs" / f"run-{rid}"
                run_dir.mkdir(parents=True, exist_ok=True)
                run_pearson_valid = mark_missing_roi_fc_columns(
                    corrcoef_rows(run_ts[valid_mask, :], run_roi),
                    missing_roi_ids=missing_roi_ids,
                    roi_ids=roi_ids,
                )
                np.save(run_dir / "hipp_vertex_to_network_fc_active.npy", run_pearson_valid.astype(np.float32))

            if int(sum(run_lengths)) != int(ts.shape[1]):
                raise RuntimeError(
                    f"Run lengths do not sum to full time series for sub-{job.subject} {job.branch}: {sum(run_lengths)} vs {ts.shape[1]}"
                )
            write_json(
                fc_dir / "fc_summary.json",
                {
                    "validity_policy": REST_POLICY,
                    "subject": job.subject,
                    "branch": job.branch,
                    "atlas_slug": ATLAS,
                    "smooth": smoothing,
                    "hemi": hemi,
                    "fc_shape": [int(pearson_full.shape[0]), int(pearson_full.shape[1])],
                    "active_fc_shape": [int(pearson_valid.shape[0]), int(pearson_valid.shape[1])],
                    "n_vertices_total": int(ts.shape[0]),
                    "n_vertices_valid_high_tsnr": int(valid_mask.sum()),
                    "n_vertices_invalid_initial": int(ts.shape[0] - valid_mask.sum()),
                    "n_timepoints": int(ts.shape[1]),
                    "n_roi_targets_total": int(roi_ts.shape[0]),
                    "n_roi_targets_with_timeseries": int(roi_ts.shape[0] - len(missing_roi_ids)),
                    "missing_roi_ids": missing_roi_ids,
                    "run_labels": run_labels,
                    "run_lengths": run_lengths,
                    "run_ids": [int(seg["run_id"]) for seg in run_segments],
                    "run_metadata": str(run_metadata_path),
                    "fc_definition": "pearson_vertex_to_kong2022_roi_timeseries",
                },
            )
            write_json(
                fc_dir / "stage_manifest.json",
                {
                    "stage": "iowa7t_branch_fc",
                    "status": "done",
                    "timestamp_utc": utc_now_iso(),
                    "params": {
                        "subject": job.subject,
                        "branch": job.branch,
                        "atlas_slug": ATLAS,
                        "smooth": smoothing,
                        "hemi": hemi,
                        "run_ids": [int(seg["run_id"]) for seg in run_segments],
                        "run_metadata": str(run_metadata_path),
                    },
                    "inputs": [
                        str((job.cortex_kong2022_root / "cortex_roi_parcel_timeseries.npy").resolve()),
                        str((job.surface_root / smoothing / f"sub-{job.subject}_hemi-{hemi}_timeseries.npy").resolve()),
                        str((job.surface_root / "tsnr_gate" / f"sub-{job.subject}_hemi-{hemi}_valid_mask.npy").resolve()),
                        file_meta_unresolved(run_metadata_path),
                    ],
                    "outputs": [
                        str((fc_dir / "hipp_vertex_to_network_fc.npy").resolve()),
                        str((fc_dir / "hipp_vertex_to_network_fc_active.npy").resolve()),
                        str((fc_dir / "fc_summary.json").resolve()),
                    ],
                },
            )


def run_job(
    job: BranchJob,
    *,
    stage: str,
    overwrite: bool,
    run_metadata_path: Path | None = None,
) -> dict[str, object]:
    require_inputs(job)
    if stage in {"all", "reference", "fc"}:
        if run_metadata_path is None:
            raise ValueError(
                f"Iowa7T {stage} stage requires an explicit run metadata path"
            )
        run_metadata_path = Path(run_metadata_path).resolve()
        load_run_segment_metadata(run_metadata_path)
    t0 = time.monotonic()
    if stage in {"all", "surface"}:
        run_surface_stage(job, overwrite=overwrite)
    if stage in {"all", "reference"}:
        run_reference_stage(job, overwrite=overwrite, run_metadata_path=run_metadata_path)
    if stage in {"all", "fc"}:
        run_fc_stage(job, overwrite=overwrite, run_metadata_path=run_metadata_path)
    return {
        "subject": job.subject,
        "branch": job.branch,
        "stage": stage,
        "branch_root": str(job.branch_root),
        "elapsed_seconds": round(float(time.monotonic() - t0), 3),
    }


def parse_job(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("Expected --job value as <subject>:<multiband|multi_echo>")
    subject, branch = value.split(":", 1)
    return normalize_subject(subject), branch.strip()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run direct fixed-layout Iowa7T branch downstream products.")
    parser.add_argument("--variant", choices=IOWA7T_VARIANTS, default=DEFAULT_VARIANT)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--hippunfold-batch-root", default=None)
    parser.add_argument(
        "--run-metadata-template",
        default=None,
        help=(
            "Per-subject JSON template declaring exact run_ids and run_lengths; "
            "required for all/reference/fc stages. Variables: {data_root}, {output_root}, {subject}, {sub_subject}."
        ),
    )
    parser.add_argument("--job", action="append", type=parse_job, required=True, help="Subject/branch pair, e.g. 06112026:multiband")
    parser.add_argument("--stage", choices=["all", "surface", "reference", "fc"], default="all")
    parser.add_argument("--jobs", type=int, default=1, help="Reserved for future parallel execution; capped at 10.")
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--tsnr-input-template", help="Baseline NPZ template with {subject}, {hemi}, {domain}, {variant}, {branch}, {run}")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.stage in {"all", "reference", "fc"} and args.run_metadata_template is None:
        parser.error("--run-metadata-template is required for --stage all/reference/fc")
    if args.jobs < 1:
        raise ValueError("--jobs must be >= 1")
    if args.jobs > MAX_CONCURRENCY:
        raise ValueError(f"--jobs={args.jobs} exceeds repository hard cap {MAX_CONCURRENCY}")
    if args.jobs != 1:
        print("[info] adapter currently executes jobs sequentially to keep Workbench process count bounded", flush=True)

    data_root = Path(args.data_root) if args.data_root is not None else None
    output_root = Path(args.output_root) if args.output_root is not None else None
    hippunfold_batch_root = Path(args.hippunfold_batch_root) if args.hippunfold_batch_root is not None else None
    results = []
    for subject, branch in args.job:
        job = resolve_branch_job(
            subject,
            branch,
            variant=str(args.variant),
            data_root=data_root,
            output_root=output_root,
            hippunfold_batch_root=hippunfold_batch_root,
        )
        job = replace(job, tsnr_input_template=args.tsnr_input_template)
        run_metadata_path = None
        if args.stage in {"all", "reference", "fc"}:
            run_metadata_path = format_subject_path_template(
                str(args.run_metadata_template),
                subject=job.subject,
                data_root=job.data_root,
                output_root=job.output_root,
            )
        print(f"[start] sub-{job.subject} {job.branch} stage={args.stage}", flush=True)
        result = run_job(job, stage=args.stage, overwrite=bool(args.overwrite), run_metadata_path=run_metadata_path)
        results.append(result)
        print(f"[done] sub-{job.subject} {job.branch} elapsed={result['elapsed_seconds']}s", flush=True)
    print(json.dumps({"jobs": results}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    raise SystemExit(main())

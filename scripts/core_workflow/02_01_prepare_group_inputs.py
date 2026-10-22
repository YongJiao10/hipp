#!/usr/bin/env python3
from __future__ import annotations

"""
Process Step Description:
Prepare group inputs. Preprocess and organize individual-level data to meet the requirements for downstream group-level analysis.

Main Flow Path:
- 3rd step in the numbered pipeline sequence:
  01_01_run_hippunfold_t1w_local.py -> 01_02_check_hippunfold_outputs.py -> 02_01_prepare_group_inputs.py (this step) -> 02_02_build_qc_manifest.py -> 03_01/03_02 reference building -> 04_01/04_02 FC materialization -> 05_01 gradient computation -> 05_02 group atlas parcellation -> 06_xx presentation rendering -> 07_01 group matrix overview.

Inputs:
- Command-line arguments:
  - --hippunfold-batch-root: Batch output root containing sub-*/hippunfold outputs (default: `outputs_hcp/inputs/hippunfold/dense_corobl_batch`).
  - --route-root: Route-specific processing directory (default: `outputs_hcp/inputs/kong2022`).
  - --shared-root: Path to output shared processed surface files (default: `outputs_hcp/inputs/kong2022/shared_inputs/surface`).
  - --template-root: Path to template root directory (default: `outputs_hcp/inputs/kong2022/templates/hipp_group_average`).
  - --bold-root: Root path containing raw fMRI volume timeseries (default: `/home/yojiao/LSS/jianglab/3_Data_Working/HCP_7T_Hippocampus/rsfmri_volume`).
  - --subjects-manifest: Path to subject manifest list (txt file, one ID per line).
  - --subjects: Space-separated list of explicit subject IDs.
  - --jobs: Worker threads for parallel execution (default: 2).
  - --resume/--no-resume: Skip already completed subjects or force recomputation (default: resume).
  - --stage: Steps to run: "all", "raw", "tsnr", "qc", "smooth", "group-average" (default: "all").
  - --dice-threshold: Dice overlap QC threshold (default: 0.70).
- Physical inputs:
  - HippUnfold subject outputs (`*_hemi-{L|R}_desc-unetf3d_dice.tsv`, midthickness gifti surfaces, etc.) under `{hippunfold_batch_root}/`.
  - Raw fMRI timeseries: an explicitly paired concatenated BOLD/CIFTI with physical-run metadata for HCP7T rest, or full denoised run-level volumes.
  - Exact run-level BOLD and dtseries inputs are supplied through the per-subject JSON passed to `--run-artifact-manifest-template`; the manifest declares run IDs, paths, sizes, and SHA256 receipts, and neither modality is concatenated on disk.

Outputs:
- Command-line output / file outputs:
  - Resampled hippocampal BOLD surface timeseries: `{shared_root}/sub-<subject>/surface/{smoothing}/sub-<subject>_hemi-{hemi}_timeseries.npy`.
  - Vertex-mask summaries: `{shared_root}/sub-<subject>/surface/tsnr_gate/sub-<subject>_hemi-{hemi}_tsnr_gate_summary.json` and `valid_mask.npy`; HCP rest uses functional-GM and signal validity without a numerical tSNR gate.
  - HCP functional-GM coverage: `{shared_root}/sub-<subject>/surface/raw/sub-<subject>_coverage_summary.json` and per-hemisphere `*_gm_weight.npy`.
  - Group average template surfaces under `{template_root}/`.
  - Run preparation manifest: `{route_root}/prepare_t1only_group_inputs_manifest.json`.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.gifti import GiftiDataArray, GiftiImage

from lib.surface_sampling import (
    FUNCTIONAL_GM_POLICY,
    sample_hippocampal_gm_support,
    sample_hipp_surface_run_timeseries,
    sample_hipp_surface_timeseries,
)
from lib.surface_coordinates import dataset_registration, require_registered_sampling
from lib.io_utils import (
    build_run_level_input_summary,
    load_run_artifact_manifest,
    load_run_segment_metadata,
    load_subject_ids,
)
from lib.output_layout import (
    DEFAULT_HIPPUNFOLD_BATCH_ROOT,
    DEFAULT_KONG2022_ROOT,
    DEFAULT_KONG2022_SURFACE_ROOT,
    DEFAULT_KONG2022_TEMPLATE_ROOT,
)
from lib.path_templates import format_subject_path_template
from lib.task_design import TASK_RUN_LENGTHS, normalize_task_branch
from lib.task_manifests import task_run_manifest_path
from lib.task_quality import TASK_VALIDITY_POLICY, task_signal_validity, require_task_policy, rest_signal_validity, signal_validity_policy, QC_DATASETS
from lib.tsnr import task_tsnr, rest_tsnr_input
from lib.workbench import run_workbench
from lib.workflow_utils import file_meta, git_commit, utc_now_iso, write_json


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BATCH_ROOT = DEFAULT_HIPPUNFOLD_BATCH_ROOT
DEFAULT_ROUTE_ROOT = DEFAULT_KONG2022_ROOT
DEFAULT_SHARED_ROOT = DEFAULT_KONG2022_SURFACE_ROOT
DEFAULT_TEMPLATE_ROOT = DEFAULT_KONG2022_TEMPLATE_ROOT
DEFAULT_BOLD_ROOT = Path("/home/yojiao/LSS/jianglab/3_Data_Working/HCP_7T_Hippocampus/rsfmri_volume")
DEFAULT_BOLD_TEMPLATE = "{bold_root}/{subject}/func_uncompressed.nii"
QC_SCRIPT = REPO_ROOT / "scripts" / "core_workflow" / "02_02_build_qc_manifest.py"
MAX_CONCURRENCY = 10
# Workbench stages spawn a Python worker plus a wb_command wrapper and the real wb_command process per subject.
# Keep subject-level Workbench concurrency at 2 so total process count stays below the repository hard cap.
MAX_WORKBENCH_SUBJECT_JOBS = 2
HEMIS = ["L", "R"]
SMOOTHINGS = ["2mm", "4mm"]


def resolve_max_concurrency() -> int:
    raw = os.environ.get("HIPP_MAX_CONCURRENCY", "").strip()
    if not raw:
        return MAX_CONCURRENCY
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid HIPP_MAX_CONCURRENCY={raw!r}; expected integer >= 1") from exc
    if value < 1:
        raise ValueError(f"Invalid HIPP_MAX_CONCURRENCY={raw!r}; expected integer >= 1")
    return value


def load_subjects(batch_root: Path, bold_root: Path, subjects_manifest: Path | None, explicit: list[str] | None) -> list[str]:
    if explicit:
        return sorted({s.replace("sub-", "") for s in explicit})
    if subjects_manifest is not None:
        return sorted(load_subject_ids(subjects_manifest))
    hipp = {p.name.replace("sub-", "") for p in batch_root.glob("sub-*") if p.is_dir()}
    bold = {p.name for p in bold_root.glob("*") if p.is_dir()}
    return sorted(hipp & bold)


def dice_pass_subjects(batch_root: Path, subjects: list[str], threshold: float) -> list[str]:
    out: list[str] = []
    for subject in subjects:
        vals = []
        for hemi in HEMIS:
            p = (
                batch_root
                / f"sub-{subject}"
                / "hippunfold"
                / f"sub-{subject}"
                / "qc"
                / f"sub-{subject}_hemi-{hemi}_desc-unetf3d_dice.tsv"
            )
            if not p.exists():
                vals = []
                break
            vals.append(float(p.read_text(encoding="utf-8").strip().splitlines()[0]))
        if len(vals) == 2 and all(v >= threshold for v in vals):
            out.append(subject)
    return out


def raw_done(shared_root: Path, subject: str, dataset: str = "hcp", *, task: str = "rest") -> bool:
    raw_dir = shared_root / f"sub-{subject}" / "surface" / "raw"
    summary = raw_dir / 'surface_sampling_summary.json'
    if summary.is_file():
        require_registered_sampling(json.loads(summary.read_text()), context=summary)
        if dataset == "hcp" and task == "rest":
            coverage = load_hcp_coverage(raw_dir, subject)
            sampling = json.loads(summary.read_text())
            if any(coverage[key] != sampling.get(key) for key in ("run_ids", "run_lengths")):
                raise ValueError(f"Coverage/run metadata mismatch at {summary}; rebuild raw sampling with --no-resume")
    return all(
        (raw_dir / f"sub-{subject}_hemi-{hemi}_space-T1w_den-512_label-hipp_bold.func.gii").exists()
        for hemi in HEMIS
    ) and (raw_dir / "surface_sampling_summary.json").exists()


def load_hcp_coverage(raw_dir: Path, subject: str) -> dict[str, object]:
    path = raw_dir / f"sub-{subject}_coverage_summary.json"
    if not path.is_file():
        raise ValueError(f"Missing functional-GM coverage at {path}; rebuild raw sampling with explicit CIFTI/run metadata and --no-resume")
    coverage = json.loads(path.read_text())
    if (str(coverage.get("subject", "")).removeprefix("sub-") != str(subject).removeprefix("sub-")
            or coverage.get("coverage_policy") != FUNCTIONAL_GM_POLICY
            or coverage.get("sampling_space") != "MNI152NLin6Asym"):
        raise ValueError(f"Invalid functional-GM coverage contract at {path}")
    return coverage


def require_hcp_validity_only(dataset: str, task: str, cutoff: float | None) -> None:
    if dataset == "hcp" and task == "rest" and cutoff is not None:
        raise ValueError("HCP rest hippocampal QC uses functional-GM coverage and finite, nonconstant signals; use --tsnr-threshold none")


def _recorded_files_current(records: list[dict[str, object]]) -> bool:
    return bool(records) and all(
        Path(record["path"]).is_file() and file_meta(Path(record["path"])) == record
        for record in records
    )


def tsnr_done(shared_root: Path, subject: str, dataset: str = "hcp", *,
              task: str = "rest", tsnr_threshold: str | None = None,
              tsnr_input_template: str | None = None) -> bool:
    from lib.task_quality import parse_tsnr_threshold
    cutoff = parse_tsnr_threshold(tsnr_threshold if tsnr_threshold is not None else ("none" if dataset == "hcp" and task == "rest" else "25"))
    require_hcp_validity_only(dataset, task, cutoff)
    policy = signal_validity_policy(dataset, task)
    gate_dir = shared_root / f"sub-{subject}" / "surface" / "tsnr_gate"
    for hemi in HEMIS:
        summary_path = gate_dir / f"sub-{subject}_hemi-{hemi}_tsnr_gate_summary.json"
        stage_path = gate_dir / f"hemi-{hemi}/stage_manifest.json"
        if not summary_path.is_file():
            return False
        summary = json.loads(summary_path.read_text())
        if summary.get('sampling_space') != 'MNI152NLin6Asym':
            raise ValueError(f'QC cache lacks registered sampling coordinates: {summary_path}; rebuild raw and QC with --no-resume')
        expected = {"threshold": cutoff, "dataset": dataset, "task": task,
                    "validity_policy": policy}
        if dataset == "hcp" and task == "rest":
            expected["coverage_policy"] = FUNCTIONAL_GM_POLICY
        if any(summary.get(key) != value for key, value in expected.items()):
            return False
        baseline = None
        if task == "rest" and dataset != "hcp":
            if tsnr_input_template is None:
                return False
            baseline = Path(tsnr_input_template.format(subject=subject, hemi=hemi)).resolve()
            if summary.get("baseline_input") is None or Path(summary["baseline_input"]).resolve() != baseline:
                return False
        if not stage_path.is_file():
            return False
        stage = json.loads(stage_path.read_text())
        recorded_inputs = stage.get("inputs", [])
        if baseline is not None and str(baseline) not in {record["path"] for record in recorded_inputs}:
            return False
        suffixes = ["tsnr.npy", "valid_mask.npy", "invalid_initial_mask.npy", "valid_roi.shape.gii"]
        if policy != "hcp_10000_over_std":
            suffixes.append("run_tsnr.npy")
        if not all((gate_dir / f"sub-{subject}_hemi-{hemi}_{suffix}").is_file() for suffix in suffixes):
            return False
        if not _recorded_files_current(recorded_inputs) or not _recorded_files_current(stage.get("outputs", [])):
            return False
    return True


def make_func_gifti(data: np.ndarray) -> GiftiImage:
    return GiftiImage(
        darrays=[
            GiftiDataArray(
                data=np.asarray(data, dtype=np.float32),
                intent="NIFTI_INTENT_TIME_SERIES",
                datatype="NIFTI_TYPE_FLOAT32",
            )
        ]
    )


def run_raw_sampling(
    subject: str,
    batch_root: str,
    shared_root: str,
    bold_root: str,
    bold_template: str,
    log_dir: str,
    run_artifact_manifest_template: str | None = None,
    task: str = "rest",
    task_manifest_root: str | None = None,
    dataset: str = 'hcp',
    registration_root: str | None = None,
    dtseries_template: str | None = None,
    run_metadata_template: str | None = None,
) -> dict[str, object]:
    batch = Path(batch_root)
    shared = Path(shared_root)
    out_dir = shared / f"sub-{subject}" / "surface" / "raw"
    log_path = Path(log_dir) / f"sub-{subject}_raw_sampling.log"
    t0 = time.monotonic()
    stage_command = "lib.surface_sampling.sample_hipp_surface_timeseries"
    stage_inputs: list[Path] = []
    task_branch = normalize_task_branch(task) if task != "rest" else "rest"
    try:
        manifest_source: Path | None = None
        if task_branch != "rest":
            if not task_manifest_root:
                raise ValueError(
                    f"--task {task_branch} requires --task-manifest-root with a "
                    "prebuilt per-subject run-artifact manifest (see lib.task_manifests)"
                )
            manifest_source = task_run_manifest_path(
                task_manifest_root, subject=subject, task=task_branch
            )
        if manifest_source is not None:
            records = load_run_artifact_manifest(
                manifest_source,
                subject=subject,
                required_artifacts=("bold", "dtseries"),
            )
            manifest_path = manifest_source
        elif run_artifact_manifest_template:
            manifest_path = format_subject_path_template(
                run_artifact_manifest_template,
                bold_root=bold_root,
                data_root=bold_root,
                subject=subject,
            )
            records = load_run_artifact_manifest(
                manifest_path,
                subject=subject,
                required_artifacts=("bold", "dtseries"),
            )
        else:
            manifest_path = None
            records = []
        if manifest_path is not None:
            selected_runs: list[tuple[int, Path]] = []
            selected_dtseries: list[Path] = []
            declared_run_ids: list[int] = []
            for record in records:
                run_id = int(record["run_id"])
                bold = Path(record["artifacts"]["bold"])
                dtseries = Path(record["artifacts"]["dtseries"])
                bold_img = nib.load(str(bold))
                if len(bold_img.shape) != 4:
                    raise ValueError(
                        f"Expected 4D run BOLD volume for sub-{subject} run-{run_id}: {bold}"
                    )
                volume_tp = int(bold_img.shape[3])
                dtseries_img = nib.load(str(dtseries))
                if len(dtseries_img.shape) != 2:
                    raise ValueError(
                        f"Expected 2D run dtseries for sub-{subject} run-{run_id}: {dtseries}"
                    )
                dtseries_tp = int(dtseries_img.shape[0])
                if volume_tp != dtseries_tp:
                    raise ValueError(
                        f"Run timepoint mismatch for sub-{subject} run-{run_id}: "
                        f"volume={volume_tp} dtseries={dtseries_tp}"
                    )
                if task_branch != "rest" and volume_tp != int(TASK_RUN_LENGTHS[task_branch]):
                    raise ValueError(
                        f"Task {task_branch} run-{run_id} expects "
                        f"{int(TASK_RUN_LENGTHS[task_branch])} frames, got {volume_tp}"
                    )
                selected_runs.append((run_id, bold))
                selected_dtseries.append(dtseries)
                declared_run_ids.append(run_id)
            out_dir.mkdir(parents=True, exist_ok=True)
            surface_data, summary = sample_hipp_surface_run_timeseries(
                runs=selected_runs,
                hippunfold_dir=batch / f"sub-{subject}" / "hippunfold",
                subject=subject,
                density="512",
                outdir=out_dir,
                space="T1w",
                registration=dataset_registration(dataset, subject, root=Path(registration_root) if registration_root else None),
                mapping_method="trilinear",
                smooth_iters=0,
            )
            dtseries_summary = build_run_level_input_summary(
                selected_dtseries,
                summary["run_lengths"],
                run_ids=declared_run_ids,
                input_mode="in_memory_run_volume_concat",
            )
            if dtseries_summary["run_ids"] != summary["run_ids"]:
                raise ValueError(
                    f"Surface/dtseries run ID mismatch for sub-{subject}: "
                    f"surface={summary['run_ids']} dtseries={dtseries_summary['run_ids']}"
                )
            summary.update(dtseries_summary)
            summary["task"] = task_branch
            summary["surface_sampling_source"] = "volume"
            summary["run_artifact_manifest"] = str(manifest_path)
            summary["run_artifact_receipts"] = [
                {
                    "run_id": int(record["run_id"]),
                    "bold": dict(record["artifact_receipts"]["bold"]),
                    "dtseries": dict(record["artifact_receipts"]["dtseries"]),
                }
                for record in records
            ]
            summary["bold_inputs"] = [str(path) for _, path in selected_runs]
            if dataset == "hcp" and task_branch == "rest":
                payload = json.loads(manifest_path.read_text())
                summary["run_labels"] = [row["run_label"] for row in payload["runs"]]
            stage_command = "lib.surface_sampling.sample_hipp_surface_run_timeseries"
            stage_inputs = [manifest_path] + [path for _, path in selected_runs] + selected_dtseries
        else:
            if dataset == "hcp" and task_branch == "rest" and (not dtseries_template or not run_metadata_template):
                raise ValueError("HCP rest volume sampling requires both --dtseries-template and --run-metadata-template")
            bold = format_subject_path_template(
                bold_template,
                bold_root=bold_root,
                subject=subject,
            )
            if not bold.is_file():
                raise FileNotFoundError(f"Missing declared BOLD volume for sub-{subject}: {bold}")
            if dataset == "hcp" and task_branch == "rest":
                dtseries = format_subject_path_template(dtseries_template, bold_root=bold_root, data_root=bold_root, subject=subject)
                metadata = format_subject_path_template(run_metadata_template, bold_root=bold_root, data_root=bold_root, subject=subject)
                run_ids, run_lengths = load_run_segment_metadata(metadata)
                dtseries_image = nib.load(dtseries)
                bold_image = nib.load(bold)
                if (len(bold_image.shape) != 4 or len(dtseries_image.shape) != 2
                        or int(bold_image.shape[3]) != sum(run_lengths)
                        or int(dtseries_image.shape[0]) != sum(run_lengths)):
                    raise ValueError(f"Declared physical run lengths {run_lengths} disagree with BOLD/CIFTI timepoints: {bold}, {dtseries}")
                metadata_payload = json.loads(metadata.read_text())
                if str(metadata_payload.get("subject", "")).removeprefix("sub-") != str(subject).removeprefix("sub-"):
                    raise ValueError(f"Physical run metadata subject mismatch at {metadata}")
                run_labels = [record["run_label"] for record in metadata_payload["runs"]]
                if len(run_labels) != len(run_ids) or len(set(run_labels)) != len(run_labels):
                    raise ValueError(f"Physical run metadata requires one distinct label per run at {metadata}")
            out_dir.mkdir(parents=True, exist_ok=True)
            surface_data = None
            summary = sample_hipp_surface_timeseries(
                bold=bold,
                hippunfold_dir=batch / f"sub-{subject}" / "hippunfold",
                subject=subject,
                density="512",
                outdir=out_dir,
                space="T1w",
                registration=dataset_registration(dataset, subject, root=Path(registration_root) if registration_root else None),
                mapping_method="trilinear",
                smooth_iters=0,
            )
            stage_inputs = [bold]
            if dataset == "hcp" and task_branch == "rest":
                sampled = summary
                summary = {
                    "subject": subject, "task": "rest", "space": "T1w",
                    "sampling_space": "MNI152NLin6Asym", "sampling_surfaces": sampled,
                    "surface_sampling_source": "volume", "bold_inputs": [str(bold)],
                    "dtseries_inputs": [str(dtseries)], "run_ids": list(run_ids),
                    "run_lengths": list(run_lengths), "n_timepoints": sum(run_lengths),
                    "run_metadata": str(metadata),
                    "run_labels": run_labels,
                }
                selected_dtseries = [dtseries]
                stage_inputs.extend([dtseries, metadata])
        if dataset == "hcp" and task_branch == "rest":
            sampled = summary["sampling_surfaces"]
            weights, source_summary = sample_hippocampal_gm_support(
                dtseries_inputs=selected_dtseries, run_ids=summary["run_ids"],
                run_lengths=summary["run_lengths"],
                sampling_surfaces={record["hemi"]: Path(record["surface"]) for record in sampled},
            )
            hemispheres = {}
            for hemi in HEMIS:
                weight_path = out_dir / f"sub-{subject}_hemi-{hemi}_gm_weight.npy"
                np.save(weight_path, weights[hemi])
                data = surface_data[hemi] if surface_data is not None else load_func_gii(out_dir / f"sub-{subject}_hemi-{hemi}_space-T1w_den-512_label-hipp_bold.func.gii")
                if weights[hemi].shape != (data.shape[0],):
                    raise ValueError(f"GM coverage/consumed surface shape mismatch for sub-{subject} hemi-{hemi}")
                valid = (weights[hemi] > 0) & rest_signal_validity(data)
                hemispheres[hemi] = {
                    **source_summary[hemi], "n_vertices_total": int(data.shape[0]),
                    "n_vertices_unsupported": int(np.count_nonzero(weights[hemi] == 0)),
                    "n_vertices_valid": int(valid.sum()), "gm_weight_path": str(weight_path),
                    "valid_mask_path": str(shared / f"sub-{subject}/surface/tsnr_gate/sub-{subject}_hemi-{hemi}_valid_mask.npy"),
                }
            coverage = {
                "subject": f"sub-{subject}", "coverage_policy": FUNCTIONAL_GM_POLICY,
                "sampling_space": summary["sampling_space"], "run_ids": summary["run_ids"],
                "run_lengths": summary["run_lengths"],
                "run_labels": summary["run_labels"],
                "density_evidence": {"density": "512", "n_vertices": {hemi: hemispheres[hemi]["n_vertices_total"] for hemi in HEMIS},
                                     "source_surfaces": {record["hemi"]: record["source_surface"] for record in sampled}},
                "hemispheres": hemispheres,
            }
            coverage_path = out_dir / f"sub-{subject}_coverage_summary.json"
            write_json(coverage_path, coverage)
            summary["coverage_policy"] = FUNCTIONAL_GM_POLICY
            summary["coverage_summary"] = str(coverage_path)
    except Exception as exc:
        log_path.write_text(str(exc), encoding="utf-8")
        return {"subject": subject, "ok": False, "stage": "raw_sampling", "error": str(exc)[-4000:]}
    if surface_data is not None:
        for hemi in HEMIS:
            data = surface_data[hemi]
            out_gii = out_dir / f"sub-{subject}_hemi-{hemi}_space-T1w_den-512_label-hipp_bold.func.gii"
            np.save(out_dir / f"sub-{subject}_hemi-{hemi}_timeseries.npy", data.astype(np.float32))
            nib.save(make_func_gifti(data), str(out_gii))
    log_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "surface_sampling_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_json(
        out_dir / "stage_manifest.json",
        {
            "stage": "t1only_raw_surface_sampling",
            "status": "done",
            "timestamp_utc": utc_now_iso(),
            "git_commit": git_commit(),
            "command": stage_command,
            "inputs": [file_meta(path) for path in stage_inputs],
            "outputs": [file_meta(p) for p in out_dir.glob("*") if p.is_file()],
            "elapsed_seconds": float(time.monotonic() - t0),
        },
    )
    return {"subject": subject, "ok": True, "stage": "raw_sampling", "summary": summary}


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


def surface_path(batch_root: Path, subject: str, hemi: str, *, space: str = 'T1w') -> Path:
    return (
        batch_root
        / f"sub-{subject}"
        / "hippunfold"
        / f"sub-{subject}"
        / "surf"
        / f"sub-{subject}_hemi-{hemi}_space-{space}_den-512_label-hipp_midthickness.surf.gii"
    )


def run_tsnr_gate(subject: str, batch_root: str, shared_root: str, task: str = "rest", dataset: str = "hcp", tsnr_input_template: str | None = None, tsnr_threshold: str | None = None) -> dict[str, object]:
    batch = Path(batch_root)
    shared = Path(shared_root)
    gate_dir = shared / f"sub-{subject}" / "surface" / "tsnr_gate"
    gate_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, object]] = []
    inputs: list[dict[str, object]] = []
    policy = signal_validity_policy(dataset, task)
    from lib.task_quality import parse_tsnr_threshold
    cutoff = parse_tsnr_threshold(tsnr_threshold if tsnr_threshold is not None else ("none" if dataset == "hcp" and task == "rest" else "25"))
    require_hcp_validity_only(dataset, task, cutoff)
    run_lengths = None
    coverage = None
    if dataset == "hcp" and task == "rest":
        raw_dir = shared / f"sub-{subject}/surface/raw"
        coverage = load_hcp_coverage(raw_dir, subject)
        sampling_path = raw_dir / "surface_sampling_summary.json"
        sampling = json.loads(sampling_path.read_text())
        if any(coverage[key] != sampling.get(key) for key in ("run_ids", "run_lengths")):
            raise ValueError(f"Coverage/run metadata mismatch at {sampling_path}")
        run_lengths = sampling["run_lengths"]
    if task != "rest":
        task = normalize_task_branch(task)
        sampling_path = shared / f"sub-{subject}" / "surface" / "raw" / "surface_sampling_summary.json"
        sampling = json.loads(sampling_path.read_text(encoding="utf-8"))
        if sampling.get("task") != task:
            raise ValueError(f"Task sampling mismatch for {task}: {sampling_path}")
        run_lengths = sampling["run_lengths"]
        if not run_lengths or any(n != TASK_RUN_LENGTHS[task] for n in run_lengths):
            raise ValueError(f"Invalid {task} run lengths at {sampling_path}: {run_lengths}")
    for hemi in HEMIS:
        raw = shared / f"sub-{subject}" / "surface" / "raw" / f"sub-{subject}_hemi-{hemi}_space-T1w_den-512_label-hipp_bold.func.gii"
        surf = surface_path(batch, subject, hemi)
        data = load_func_gii(raw).astype(np.float64)
        if coverage is not None and sum(run_lengths) != data.shape[1]:
            raise ValueError(f"Physical run lengths {run_lengths} do not cover consumed surface timepoints {data.shape[1]} at {raw}")
        hemi_inputs = [file_meta(raw), file_meta(surf)]
        if policy == "hcp_10000_over_std":
            # HCP rest has no interpretable retained baseline and no numeric
            # tSNR gate. Retain the existing diagnostic path as an empty score.
            tsnr = np.full(data.shape[0], np.nan, dtype=np.float32)
            weight_path = raw.parent / f"sub-{subject}_hemi-{hemi}_gm_weight.npy"
            declared = Path(coverage["hemispheres"][hemi]["gm_weight_path"])
            if declared.resolve() != weight_path.resolve():
                raise ValueError(f"Coverage GM weight declaration mismatch for sub-{subject} hemi-{hemi}: {declared}; expected {weight_path}")
            weight = np.load(weight_path)
            if weight.shape != (data.shape[0],) or not np.isfinite(weight).all() or np.any(weight < 0):
                raise ValueError(f"Invalid functional-GM weights at {weight_path}: shape={weight.shape}, vertices={data.shape[0]}")
            valid = weight > 0
            hemi_inputs.extend([file_meta(raw.parent / f"sub-{subject}_coverage_summary.json"), file_meta(weight_path), file_meta(sampling_path)])
            np.save(gate_dir / f"sub-{subject}_hemi-{hemi}_tsnr.npy", tsnr)
        elif task == "rest":
            if tsnr_input_template is None:
                raise ValueError(f"{dataset} tSNR requires --tsnr-input-template with baseline-preserving per-run samples and retained frames")
            qc_path = Path(tsnr_input_template.format(subject=subject, hemi=hemi))
            run_tsnr, valid = rest_tsnr_input(qc_path, dataset=dataset, subject=subject, hemi=hemi, n_vertices=data.shape[0], n_frames=data.shape[1], threshold=cutoff)
            hemi_inputs.append(file_meta(qc_path))
        else:
            run_tsnr, valid = task_tsnr(data, run_lengths, sampling["bold_inputs"], task=task, threshold=cutoff)
        if task == "rest":
            valid &= rest_signal_validity(data)
        if policy != "hcp_10000_over_std":
            tsnr = np.min(run_tsnr, axis=0)
            np.save(gate_dir / f"sub-{subject}_hemi-{hemi}_tsnr.npy", tsnr)
            np.save(gate_dir / f"sub-{subject}_hemi-{hemi}_run_tsnr.npy", run_tsnr)
        invalid = ~valid
        np.save(gate_dir / f"sub-{subject}_hemi-{hemi}_valid_mask.npy", valid.astype(bool))
        np.save(gate_dir / f"sub-{subject}_hemi-{hemi}_invalid_initial_mask.npy", invalid.astype(bool))
        nib.save(make_shape_gifti(valid.astype(np.float32)), str(gate_dir / f"sub-{subject}_hemi-{hemi}_valid_roi.shape.gii"))
        summary = {
            "subject": subject,
            "hemi": hemi,
            "threshold": None if cutoff is None else float(cutoff),
            "tsnr_definition": "10000/std" if policy == "hcp_10000_over_std" else policy,
            "task": task,
            "validity_policy": policy,
            "dataset": dataset,
            "sampling_space": "MNI152NLin6Asym",
            "tsnr_computed": cutoff is not None,
            "baseline_input": str(qc_path) if task == "rest" and dataset != "hcp" else None,
            "run_lengths": run_lengths,
            "run_aggregation": "validity_only_no_tsnr" if cutoff is None else ("all_runs_pass" if policy != "hcp_10000_over_std" else None),
            "n_vertices_total": int(valid.size),
            "n_vertices_valid": int(valid.sum()),
            "n_vertices_invalid": int(invalid.sum()),
            "invalid_fraction": float(invalid.sum() / max(1, valid.size)),
        }
        if coverage is not None:
            summary["coverage_policy"] = FUNCTIONAL_GM_POLICY
            summary["tsnr_definition"] = None
            summary["n_vertices_unsupported"] = int(np.count_nonzero(weight == 0))
        write_json(gate_dir / f"sub-{subject}_hemi-{hemi}_tsnr_gate_summary.json", summary)
        hemi_dir = gate_dir / f"hemi-{hemi}"
        write_json(
            hemi_dir / "stage_manifest.json",
            {
                "stage": "t1only_hipp_tsnr_gate",
                "status": "done",
                "timestamp_utc": utc_now_iso(),
                "git_commit": git_commit(),
                "params": {key: summary[key] for key in ("subject", "hemi", "threshold", "tsnr_definition", "task", "dataset", "validity_policy", "run_lengths")},
                "inputs": hemi_inputs,
                "outputs": [
                    file_meta(gate_dir / f"sub-{subject}_hemi-{hemi}_tsnr.npy"),
                    file_meta(gate_dir / f"sub-{subject}_hemi-{hemi}_valid_mask.npy"),
                    file_meta(gate_dir / f"sub-{subject}_hemi-{hemi}_invalid_initial_mask.npy"),
                    file_meta(gate_dir / f"sub-{subject}_hemi-{hemi}_valid_roi.shape.gii"),
                    file_meta(gate_dir / f"sub-{subject}_hemi-{hemi}_tsnr_gate_summary.json"),
                ],
            },
        )
        inputs.extend(hemi_inputs)
        suffixes = ["valid_mask.npy", "invalid_initial_mask.npy", "valid_roi.shape.gii", "tsnr_gate_summary.json"]
        suffixes.append("tsnr.npy")
        if policy != "hcp_10000_over_std":
            suffixes.append("run_tsnr.npy")
        outputs.extend(file_meta(gate_dir / f"sub-{subject}_hemi-{hemi}_{suffix}") for suffix in suffixes)
    return {"subject": subject, "ok": True, "stage": "tsnr_gate", "inputs": inputs, "outputs": outputs}


def smoothing_done(shared_root: Path, subject: str, dataset: str = "hcp", *, task: str = "rest") -> bool:
    surface = shared_root / f'sub-{subject}/surface'
    stage_path = surface / 'stage_manifest.json'
    if not stage_path.is_file():
        return False
    stage = json.loads(stage_path.read_text())
    params = stage.get('params', {})
    if params.get('smoothing_space') != 'T1w':
        raise ValueError(f'Smoothing cache lacks native T1w geometry: {stage_path}; rebuild smoothing with --no-resume')
    expected = {"dataset": dataset, "task": task, "validity_policy": signal_validity_policy(dataset, task)}
    if dataset == "hcp" and task == "rest" and params.get("smoothing_roi_policy") != FUNCTIONAL_GM_POLICY:
        return False
    if any(params.get(key) != value for key, value in expected.items()):
        return False
    thresholds = {}
    for hemi in HEMIS:
        summary_path = surface / f'tsnr_gate/sub-{subject}_hemi-{hemi}_tsnr_gate_summary.json'
        if not summary_path.is_file():
            return False
        gate = json.loads(summary_path.read_text())
        if dataset == "hcp" and task == "rest" and (gate.get("coverage_policy") != FUNCTIONAL_GM_POLICY or gate.get("threshold") is not None):
            return False
        if any(gate.get(key) != value for key, value in expected.items()):
            return False
        thresholds[hemi] = gate['threshold']
    if params.get('tsnr_thresholds') != thresholds:
        return False
    for smoothing in SMOOTHINGS:
        for hemi in HEMIS:
            if not (surface / smoothing / f"sub-{subject}_hemi-{hemi}_timeseries.npy").is_file():
                return False
    return (_recorded_files_current(stage.get('inputs', []))
            and _recorded_files_current(stage.get('outputs', [])))


def run_smoothing(subject: str, batch_root: str, shared_root: str, task: str = "rest", dataset: str = "hcp") -> dict[str, object]:
    policy = signal_validity_policy(dataset, task)
    batch = Path(batch_root)
    shared = Path(shared_root)
    inputs: list[dict[str, object]] = []
    outputs: list[dict[str, object]] = []
    thresholds: dict[str, float | None] = {}
    # Thread caps are enforced centrally by lib.workbench.run_workbench.
    _wb_env = {}
    for smoothing in SMOOTHINGS:
        fwhm = float(smoothing.replace("mm", ""))
        out_dir = shared / f"sub-{subject}" / "surface" / smoothing
        out_dir.mkdir(parents=True, exist_ok=True)
        for hemi in HEMIS:
            raw = shared / f"sub-{subject}" / "surface" / "raw" / f"sub-{subject}_hemi-{hemi}_space-T1w_den-512_label-hipp_bold.func.gii"
            roi = shared / f"sub-{subject}" / "surface" / "tsnr_gate" / f"sub-{subject}_hemi-{hemi}_valid_roi.shape.gii"
            gate_summary = roi.parent / f"sub-{subject}_hemi-{hemi}_tsnr_gate_summary.json"
            gate = json.loads(gate_summary.read_text(encoding="utf-8"))
            if gate.get("validity_policy") != policy or gate.get("task") != task or gate.get("dataset") != dataset:
                raise ValueError(f"Rebuild {dataset}/{task} gate with policy={policy}: {gate_summary}")
            if dataset == "hcp" and task == "rest" and (gate.get("coverage_policy") != FUNCTIONAL_GM_POLICY or gate.get("threshold") is not None):
                raise ValueError(f"Rebuild HCP rest mask with functional-GM support and --tsnr-threshold none: {gate_summary}")
            thresholds[hemi] = gate["threshold"]
            surf = surface_path(batch, subject, hemi)
            out_gii = out_dir / f"sub-{subject}_hemi-{hemi}_space-T1w_den-512_label-hipp_bold.func.gii"
            cmd = [
                "-metric-smoothing",
                str(surf),
                str(raw),
                str(fwhm),
                str(out_gii),
                "-fwhm",
                "-roi",
                str(roi),
            ]
            try:
                run_workbench(cmd, env=_wb_env)
            except Exception as exc:
                return {"subject": subject, "ok": False, "stage": "smoothing", "error": str(exc)[-4000:]}
            arr = load_func_gii(out_gii)
            np.save(out_dir / f"sub-{subject}_hemi-{hemi}_timeseries.npy", arr.astype(np.float32))
            inputs.extend([file_meta(raw), file_meta(roi), file_meta(surf)])
            outputs.extend([file_meta(out_gii), file_meta(out_dir / f"sub-{subject}_hemi-{hemi}_timeseries.npy")])
    write_json(
        shared / f"sub-{subject}" / "surface" / "stage_manifest.json",
        {
            "stage": "t1only_surface_smoothing",
            "status": "done",
            "timestamp_utc": utc_now_iso(),
            "git_commit": git_commit(),
            "params": {"subject": subject, "smoothings": SMOOTHINGS, "task": task,
                       "smoothing_space": "T1w",
                       "validity_policy": policy,
                       "dataset": dataset,
                       "tsnr_thresholds": thresholds,
                       "smoothing_roi_policy": FUNCTIONAL_GM_POLICY if dataset == "hcp" and task == "rest" else policy},
            "inputs": inputs,
            "outputs": outputs,
        },
    )
    return {"subject": subject, "ok": True, "stage": "smoothing"}


def load_surface(path: Path) -> tuple[np.ndarray, np.ndarray]:
    img = nib.load(str(path))
    coords = None
    faces = None
    for da in img.darrays:
        if da.intent == "NIFTI_INTENT_POINTSET":
            coords = np.asarray(da.data, dtype=np.float32)
        elif da.intent == "NIFTI_INTENT_TRIANGLE":
            faces = np.asarray(da.data, dtype=np.int32)
    if coords is None:
        coords = np.asarray(img.darrays[0].data, dtype=np.float32)
    if faces is None:
        faces = np.asarray(img.darrays[1].data, dtype=np.int32)
    return coords, faces


def save_surface(path: Path, coords: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = GiftiImage(
        darrays=[
            GiftiDataArray(data=np.asarray(coords, dtype=np.float32), intent="NIFTI_INTENT_POINTSET"),
            GiftiDataArray(data=np.asarray(faces, dtype=np.int32), intent="NIFTI_INTENT_TRIANGLE"),
        ]
    )
    nib.save(img, str(path))


def build_group_average_surfaces(batch_root: Path, template_root: Path, subjects: list[str]) -> None:
    out_dir = template_root / "group_average_surfaces"
    outputs: list[dict[str, object]] = []
    inputs: list[dict[str, object]] = []
    for hemi in HEMIS:
        coords_all: list[np.ndarray] = []
        ref_faces: np.ndarray | None = None
        for subject in subjects:
            p = surface_path(batch_root, subject, hemi, space='corobl')
            coords, faces = load_surface(p)
            if ref_faces is None:
                ref_faces = faces
            elif coords.shape != coords_all[0].shape or not np.array_equal(faces, ref_faces):
                raise RuntimeError(f"Surface topology mismatch for sub-{subject} hemi-{hemi}: {p}")
            coords_all.append(coords)
            inputs.append(file_meta(p))
        if ref_faces is None or not coords_all:
            raise RuntimeError(f"No surfaces available for hemi-{hemi}")
        avg = np.mean(np.stack(coords_all, axis=0), axis=0).astype(np.float32)
        out = out_dir / f"group-average_hemi-{hemi}_space-corobl_den-512_label-hipp_midthickness.surf.gii"
        save_surface(out, avg, ref_faces)
        outputs.append(file_meta(out))
    write_json(
        out_dir / "stage_manifest.json",
        {
            "stage": "t1only_group_average_surfaces",
            "status": "done",
            "timestamp_utc": utc_now_iso(),
            "git_commit": git_commit(),
            "n_subjects": int(len(subjects)),
            "subjects": [f"sub-{s}" for s in subjects],
            "inputs": inputs,
            "outputs": outputs,
        },
    )


def run_parallel(label: str, subjects: list[str], jobs: int, func, *args) -> None:
    todo = list(subjects)
    if not todo:
        print(f"[info] {label}: no subjects", flush=True)
        return
    failed: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=max(1, min(jobs, len(todo)))) as pool:
        futs = {pool.submit(func, s, *args): s for s in todo}
        done = 0
        for fut in as_completed(futs):
            done += 1
            result = fut.result()
            if result.get("ok"):
                print(f"[done {done}/{len(todo)}] {label} sub-{result['subject']}", flush=True)
            else:
                failed.append(result)
                print(f"[FAIL {done}/{len(todo)}] {label} sub-{result.get('subject')} {result.get('error', '')}", flush=True)
    if failed:
        raise RuntimeError(f"{label} failed for {len(failed)} subjects: {[x.get('subject') for x in failed]}")


def read_latest_manifest(route_root: Path) -> list[str]:
    p = route_root / "manifests" / "hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt"
    return load_subject_ids(p)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare T1only shared hippocampal inputs and group-average surfaces.")
    parser.add_argument("--hippunfold-batch-root", default=str(DEFAULT_BATCH_ROOT))
    parser.add_argument("--route-root", default=str(DEFAULT_ROUTE_ROOT))
    parser.add_argument("--shared-root", default=str(DEFAULT_SHARED_ROOT))
    parser.add_argument("--template-root", default=str(DEFAULT_TEMPLATE_ROOT))
    parser.add_argument("--bold-root", default=str(DEFAULT_BOLD_ROOT))
    parser.add_argument("--log-dir", help="Explicit preparation log directory; reruns use their runs/<run_id>/prepare directory.")
    parser.add_argument(
        "--bold-template",
        default=DEFAULT_BOLD_TEMPLATE,
        help=(
            "Per-subject BOLD volume template. Variables: {bold_root}, {subject}, "
            "{sub_subject}. Default preserves the legacy HCP layout."
        ),
    )
    parser.add_argument(
        "--run-artifact-manifest-template",
        default=None,
        help=(
            "Per-subject JSON receipt declaring the exact ordered BOLD/dtseries runs. "
            "Variables: {bold_root}, {data_root}, {subject}, {sub_subject}."
        ),
    )
    parser.add_argument("--dtseries-template", help="Exact per-subject HCP rest CIFTI template; use with --run-metadata-template for a concatenated BOLD/CIFTI pair.")
    parser.add_argument("--run-metadata-template", help="Exact per-subject JSON declaring physical run IDs, labels and frame counts for the concatenated HCP rest pair.")
    parser.add_argument("--motion-manifest", default=str(REPO_ROOT / "manifests/hcp7t_motion_transfer_20261019.tsv"), help="Exact HCP7T Power-FD motion table passed to the participant QC stage.")
    parser.add_argument(
        "--task",
        default="rest",
        help=(
            "Condition branch: rest uses the dataset's declared inputs; emotion|wm|relational "
            "requires --task-manifest-root with prebuilt per-subject manifests "
            "(see lib.task_manifests) and validates run frames against the task contract."
        ),
    )
    parser.add_argument(
        "--task-manifest-root",
        default=None,
        help="Root holding task-<task>/sub-<subject>.json run-artifact manifests.",
    )
    parser.add_argument("--dataset", choices=QC_DATASETS, default="hcp", help="Select dataset-specific tSNR inputs and definition.")
    parser.add_argument('--registration-root', help='Explicit Jianglab storage prefix for registration inputs; defaults to the active system prefix.')
    parser.add_argument(
        "--tsnr-threshold",
        default=None,
        help="HCP rest uses 'none' and requires positive functional-GM support plus finite, nonconstant signal. Other branches retain a default cutoff of 25.",
    )
    parser.add_argument("--tsnr-input-template", help="Rest baseline NPZ template containing {subject} and {hemi}; required outside HCP.")
    parser.add_argument("--subjects-manifest", default=None)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stage", choices=["all", "raw", "tsnr", "qc", "smooth", "group-average"], default="all")
    parser.add_argument("--dice-threshold", type=float, default=0.70)
    args = parser.parse_args()
    if (args.dataset != "hcp" or args.task != "rest") and args.stage in {"all", "qc"}:
        parser.error("Participant QC currently implements HCP7T rest only; the approved motion/Dice/GM-coverage policy is not yet implemented for this dataset/task. Run the other preparation stages explicitly.")
    if args.tsnr_threshold is None:
        args.tsnr_threshold = "none" if args.dataset == "hcp" and args.task == "rest" else "25"
    from lib.task_quality import parse_tsnr_threshold
    try:
        require_hcp_validity_only(args.dataset, args.task, parse_tsnr_threshold(args.tsnr_threshold))
    except ValueError as exc:
        parser.error(str(exc))
    if bool(args.dtseries_template) != bool(args.run_metadata_template):
        parser.error("--dtseries-template and --run-metadata-template must be supplied together")
    if args.run_artifact_manifest_template and args.dtseries_template:
        parser.error("Choose the exact run-artifact manifest or the concatenated BOLD/CIFTI metadata mode")
    if (args.dataset == "hcp" and args.task == "rest" and args.stage in {"all", "raw"}
            and not args.run_artifact_manifest_template and not args.dtseries_template):
        parser.error("HCP rest raw sampling requires --dtseries-template and --run-metadata-template, or --run-artifact-manifest-template")

    batch_root = Path(args.hippunfold_batch_root).resolve()
    route_root = Path(args.route_root).resolve()
    shared_root = Path(args.shared_root).resolve()
    template_root = Path(args.template_root).resolve()
    bold_root = Path(args.bold_root).resolve()
    if args.task != "rest" or args.dataset != "hcp":
        destinations = [
            ("--route-root", route_root, DEFAULT_ROUTE_ROOT),
            ("--shared-root", shared_root, DEFAULT_SHARED_ROOT),
        ]
        if args.stage in {"all", "group-average"}:
            destinations.append(("--template-root", template_root, DEFAULT_TEMPLATE_ROOT))
        conflicts = [
            f"{option}={path}"
            for option, path, rest_path in destinations
            if path == rest_path.resolve()
        ]
        if conflicts:
            parser.error(
                f"--task {args.task} --dataset {args.dataset} cannot write HCP rest "
                f"destinations: {', '.join(conflicts)}. Pass explicit separate roots."
            )
    log_dir = Path(args.log_dir).resolve() if args.log_dir else route_root / "logs" / "prepare_t1only_group_inputs"
    log_dir.mkdir(parents=True, exist_ok=True)
    shared_root.mkdir(parents=True, exist_ok=True)
    max_concurrency = resolve_max_concurrency()
    requested_jobs = int(args.jobs)
    jobs = max(1, min(requested_jobs, max_concurrency))
    workbench_jobs = max(1, min(jobs, MAX_WORKBENCH_SUBJECT_JOBS))
    if requested_jobs > max_concurrency:
        print(
            f"[warn] requested jobs={requested_jobs} exceeds max_concurrency={max_concurrency}; using jobs={jobs}",
            flush=True,
        )
    if jobs != workbench_jobs:
        print(
            f"[warn] Workbench stages use jobs={workbench_jobs} to keep worker+wb_command processes below the hard cap",
            flush=True,
        )

    subjects_manifest = Path(args.subjects_manifest).resolve() if args.subjects_manifest else None
    subjects = load_subjects(batch_root, bold_root, subjects_manifest, args.subjects)
    dice_subjects = dice_pass_subjects(batch_root, subjects, args.dice_threshold)
    print(f"[info] subjects={len(subjects)} dice_pass={len(dice_subjects)} jobs={jobs} workbench_jobs={workbench_jobs}", flush=True)

    if args.stage in {"all", "raw"}:
        raw_todo = [
            s
            for s in dice_subjects
            if not (
                args.resume
                and raw_done(shared_root, s, args.dataset, task=args.task)
            )
        ]
        print(f"[info] raw sampling todo={len(raw_todo)}", flush=True)
        run_parallel(
            "raw",
            raw_todo,
            workbench_jobs,
            run_raw_sampling,
            str(batch_root),
            str(shared_root),
            str(bold_root),
            args.bold_template,
            str(log_dir),
            args.run_artifact_manifest_template,
            args.task,
            args.task_manifest_root,
            args.dataset,
            args.registration_root,
            args.dtseries_template,
            args.run_metadata_template,
        )
    if args.stage in {"all", "tsnr"}:
        tsnr_todo = [s for s in dice_subjects if not (args.resume and args.task == "rest" and tsnr_done(shared_root, s, args.dataset, task=args.task, tsnr_threshold=args.tsnr_threshold, tsnr_input_template=args.tsnr_input_template))]
        print(f"[info] tsnr gate todo={len(tsnr_todo)}", flush=True)
        run_parallel("tsnr", tsnr_todo, jobs, run_tsnr_gate, str(batch_root), str(shared_root), args.task, args.dataset, args.tsnr_input_template, str(args.tsnr_threshold))
    manifest_path = route_root / "manifests" / "hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt"
    if args.stage in {"all", "qc"}:
        cmd = [
            sys.executable,
            str(QC_SCRIPT),
            "--hippunfold-batch-root", str(batch_root),
            "--shared-root", str(shared_root),
            "--out-dir", str(route_root / "qc"),
            "--manifest-dir", str(route_root / "manifests"),
            "--dice-threshold", str(args.dice_threshold),
            "--subjects", *subjects,
        ]
        if args.dataset == "hcp" and args.task == "rest":
            cmd += ["--motion-manifest", args.motion_manifest]
        proc = subprocess.run(cmd, text=True)
        if proc.returncode != 0:
            return proc.returncode
    pass_subjects = read_latest_manifest(route_root) if manifest_path.exists() else []
    if args.stage in {"all", "smooth"}:
        smooth_todo = [s for s in pass_subjects if not (args.resume and args.task == "rest" and smoothing_done(shared_root, s, args.dataset, task=args.task))]
        print(f"[info] smoothing pass_subjects={len(pass_subjects)} todo={len(smooth_todo)}", flush=True)
        run_parallel("smooth", smooth_todo, workbench_jobs, run_smoothing, str(batch_root), str(shared_root), args.task, args.dataset)
    if args.stage in {"all", "group-average"}:
        print(f"[info] group-average subjects={len(pass_subjects)}", flush=True)
        build_group_average_surfaces(batch_root, template_root, pass_subjects)

    write_json(
        route_root / "prepare_t1only_group_inputs_manifest.json",
        {
            "stage": "prepare_t1only_group_inputs",
            "status": "done",
            "timestamp_utc": utc_now_iso(),
            "git_commit": git_commit(),
            "hippunfold_batch_root": str(batch_root),
            "shared_root": str(shared_root),
            "template_root": str(template_root),
            "task": str(args.task),
            "tsnr_threshold": str(args.tsnr_threshold),
            "tsnr_input_template": args.tsnr_input_template,
            "task_manifest_root": args.task_manifest_root,
            "run_artifact_manifest_template": args.run_artifact_manifest_template,
            "dtseries_template": args.dtseries_template,
            "run_metadata_template": args.run_metadata_template,
            "n_subjects_total": int(len(subjects)),
            "n_subjects_dice_pass": int(len(dice_subjects)),
            "n_subjects_qc_pass": int(len(pass_subjects)),
            "jobs": int(jobs),
            "max_concurrency": int(max_concurrency),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

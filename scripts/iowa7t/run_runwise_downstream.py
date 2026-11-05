#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.gifti import GiftiDataArray, GiftiImage

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_WORKFLOW_DIR = REPO_ROOT / "scripts" / "core_workflow"
IOWA7T_DIR = REPO_ROOT / "scripts" / "iowa7t"
if str(CORE_WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_WORKFLOW_DIR))
if str(IOWA7T_DIR) not in sys.path:
    sys.path.insert(0, str(IOWA7T_DIR))

from lib.extract_cortex_merged_network_timeseries import extract_structure_data, load_required_mask
from lib.extract_cortex_kong2022_route_a_roi_timeseries import (
    ROUTE_A_SOURCE_TAG,
    aggregate_hemi_roi_timeseries,
    align_surface_labels_to_dt_vertices,
    compute_cortex_validity,
    load_surface_label_vector,
)
from lib.fc_math import compact_active_vertices, corrcoef_rows
from lib.io_utils import write_tsv
from lib.kong2022_label_utils import (
    KONG2022_DLABEL_PATH,
    KONG2022_MERGE_CONFIG,
    canonical_families_7_order,
    canonical_families_17_order,
    load_kong2022_bundle,
)
from lib.surface_sampling import sample_hipp_surface_timeseries
from lib.surface_coordinates import dataset_registration, require_registered_sampling
from lib.task_quality import CORTEX_VALIDITY_POLICY, SERIES_VALIDITY_POLICY, rest_signal_validity
from lib.workbench import run_workbench
from lib.workflow_utils import file_meta, file_meta_unresolved, git_commit, utc_now_iso, write_json

from run_branch_downstream import (
    fixed_400_roi_reference,
    load_func_gii,
    make_shape_gifti,
    mark_missing_roi_fc_columns,
    read_roi_metadata,
)

DEFAULT_SUBJECT = "06112026"
DEFAULT_RUN_TAG = "06112026_20260904"
DEFAULT_BASE_ROOT = Path("/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_Iowa7T")
SUPPORTED_VARIANTS = ["36p"]
RUN_SPECS = [
    {"run_label": "run-01", "acquisition": "ME01", "kind": "multi_echo", "dir": ""},
    {"run_label": "run-02", "acquisition": "ME02", "kind": "multi_echo", "dir": ""},
    {"run_label": "run-03", "acquisition": "MB03", "kind": "multiband", "dir": "PA"},
    {"run_label": "run-04", "acquisition": "MB04", "kind": "multiband", "dir": "AP"},
]
HEMIS = ["L", "R"]
SMOOTHINGS = ["2mm", "4mm"]
ATLAS = "kong2022"


@dataclass(frozen=True)
class RunwiseJob:
    subject: str
    variant: str
    run_label: str
    acquisition: str
    kind: str
    data_root: Path
    runwise_root: Path
    run_dir: Path
    bold_volume: Path
    dtseries: Path
    lh_labels: Path
    rh_labels: Path
    hippunfold_dir: Path
    surface_root: Path
    cortex_root: Path
    fc_root: Path



def resolve_runwise_job(
    subject: str,
    variant: str,
    spec: dict[str, str],
    *,
    base_root: Path | None = None,
    run_tag: str | None = None,
) -> RunwiseJob:
    subject = str(subject).strip().replace("sub-", "")
    base = base_root or DEFAULT_BASE_ROOT
    var_upper = variant.upper()
    data_root = base / f"DataOrg_PFM_{var_upper}"
    if not run_tag:
        matches = sorted(data_root.glob(f"kong2022_runwise_{subject}_*"))
        if matches:
            run_tag = matches[-1].name.replace("kong2022_runwise_", "")
        else:
            run_tag = DEFAULT_RUN_TAG if subject == DEFAULT_SUBJECT else f"{subject}_runwise"
    runwise_root = data_root / f"kong2022_runwise_{run_tag}"
    run_label = spec["run_label"]
    acq = spec["acquisition"]
    run_dir = runwise_root / run_label

    src_sub = data_root / f"sub-{subject}_{acq}"
    func_dir = src_sub / "func"

    if spec["kind"] == "multi_echo":
        run_num = "01" if acq == "ME01" else "02"
        bold_volume = func_dir / f"sub-{subject}_{acq}_task-rest_run-{run_num}_space-MNI152NLin6Asym_res-2_desc-denoised_bold.nii.gz"
        dtseries = func_dir / f"sub-{subject}_{acq}_task-rest_run-{run_num}_space-fsLR_den-91k_desc-MEICA_XCPD{var_upper}_denoised_bold.dtseries.nii"
    else:
        run_num = "03" if acq == "MB03" else "04"
        dir_tag = spec["dir"]
        bold_volume = func_dir / f"sub-{subject}_{acq}_task-rest_dir-{dir_tag}_run-{run_num}_space-MNI152NLin6Asym_res-2_desc-denoised_bold.nii.gz"
        dtseries = func_dir / f"sub-{subject}_{acq}_task-rest_dir-{dir_tag}_run-{run_num}_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii"

    hippunfold_dir = (
        base
        / "DataOrg_PFM_36P"
        / "hipp_outputs"
        / "subjects"
        / f"sub-{subject}"
        / "hippunfold"
    )

    return RunwiseJob(
        subject=subject,
        variant=variant,
        run_label=run_label,
        acquisition=acq,
        kind=spec["kind"],
        data_root=data_root,
        runwise_root=runwise_root,
        run_dir=run_dir,
        bold_volume=bold_volume,
        dtseries=dtseries,
        lh_labels=run_dir / "lh_labels.txt",
        rh_labels=run_dir / "rh_labels.txt",
        hippunfold_dir=hippunfold_dir,
        surface_root=run_dir / "surface",
        cortex_root=run_dir / "cortex",
        fc_root=run_dir / "fc",
    )


def require_inputs(job: RunwiseJob) -> None:
    required = [
        job.bold_volume,
        job.dtseries,
        job.lh_labels,
        job.rh_labels,
        job.hippunfold_dir / f"sub-{job.subject}" / "surf" / f"sub-{job.subject}_hemi-L_space-T1w_den-512_label-hipp_midthickness.surf.gii",
        job.hippunfold_dir / f"sub-{job.subject}" / "surf" / f"sub-{job.subject}_hemi-R_space-T1w_den-512_label-hipp_midthickness.surf.gii",
    ]
    missing = [str(p) for p in required if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required runwise inputs for {job.variant} {job.run_label}:\n" + "\n".join(missing))


def cortex_qc_done(directory: Path) -> bool:
    for hemi in ("left", "right"):
        path = directory / "validity" / f"cortex_{hemi}_validity_summary.json"
        if not path.is_file() or json.loads(path.read_text()).get("validity_policy") != CORTEX_VALIDITY_POLICY:
            return False
    return True


def qc_policy_done(job) -> bool:
    for hemi in HEMIS:
        path = job.surface_root / "validity" / f"sub-{job.subject}_hemi-{hemi}_validity_summary.json"
        if not path.is_file():
            return False
        summary = json.loads(path.read_text())
        if summary.get('validity_policy') != SERIES_VALIDITY_POLICY or summary.get('sampling_space') != 'MNI152NLin6Asym':
            return False
    return True


def run_surface_stage(job: RunwiseJob, *, overwrite: bool = False) -> None:
    t0 = time.monotonic()
    raw_dir = job.surface_root / "raw"
    gate_dir = job.surface_root / "validity"
    summary_path = raw_dir / 'surface_sampling_summary.json'
    if not overwrite and summary_path.is_file():
        require_registered_sampling(json.loads(summary_path.read_text()), context=summary_path)
    
    raw_done = all(
        (raw_dir / f"sub-{job.subject}_hemi-{hemi}_space-T1w_den-512_label-hipp_bold.func.gii").exists()
        for hemi in HEMIS
    )
    if overwrite or not raw_done:
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

    qc_changed = not qc_policy_done(job)
    gate_done = all((gate_dir / f"sub-{job.subject}_hemi-{hemi}_valid_mask.npy").exists() for hemi in HEMIS)
    if overwrite or not gate_done or not qc_policy_done(job):
        gate_dir.mkdir(parents=True, exist_ok=True)
        for hemi in HEMIS:
            raw = raw_dir / f"sub-{job.subject}_hemi-{hemi}_space-T1w_den-512_label-hipp_bold.func.gii"
            data = load_func_gii(raw).astype(np.float64)
            valid = rest_signal_validity(data)
            invalid = ~valid
            np.save(gate_dir / f"sub-{job.subject}_hemi-{hemi}_valid_mask.npy", valid.astype(bool))
            np.save(gate_dir / f"sub-{job.subject}_hemi-{hemi}_invalid_initial_mask.npy", invalid.astype(bool))
            nib.save(
                make_shape_gifti(valid.astype(np.float32)),
                str(gate_dir / f"sub-{job.subject}_hemi-{hemi}_valid_roi.shape.gii"),
            )
            write_json(
                gate_dir / f"sub-{job.subject}_hemi-{hemi}_validity_summary.json",
                {
                    "subject": job.subject,
                    "run_label": job.run_label,
                    "hemi": hemi,
                    "validity_policy": SERIES_VALIDITY_POLICY,
                    "sampling_space": "MNI152NLin6Asym",
                    "n_vertices_total": int(valid.size),
                    "n_vertices_valid": int(valid.sum()),
                    "n_vertices_invalid": int(invalid.sum()),
                    "invalid_fraction": float(invalid.sum() / max(1, valid.size)),
                },
            )

    smoothing_stage = job.surface_root / 'stage_manifest.json'
    native_smoothing = smoothing_stage.is_file() and json.loads(smoothing_stage.read_text()).get('params', {}).get('smoothing_space') == 'T1w'
    for smoothing in SMOOTHINGS:
        fwhm = float(smoothing.replace("mm", ""))
        out_dir = job.surface_root / smoothing
        out_dir.mkdir(parents=True, exist_ok=True)
        for hemi in HEMIS:
            ts_npy = out_dir / f"sub-{job.subject}_hemi-{hemi}_timeseries.npy"
            if not overwrite and ts_npy.exists() and not qc_changed and native_smoothing:
                continue
            raw = raw_dir / f"sub-{job.subject}_hemi-{hemi}_space-T1w_den-512_label-hipp_bold.func.gii"
            roi = gate_dir / f"sub-{job.subject}_hemi-{hemi}_valid_roi.shape.gii"
            surf = (
                job.hippunfold_dir
                / f"sub-{job.subject}"
                / "surf"
                / f"sub-{job.subject}_hemi-{hemi}_space-T1w_den-512_label-hipp_midthickness.surf.gii"
            )
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
            np.save(ts_npy, arr.astype(np.float32))

    write_json(
        job.surface_root / "stage_manifest.json",
        {
            "stage": "iowa7t_runwise_surface",
            "status": "done",
            "timestamp_utc": utc_now_iso(),
            "git_commit": git_commit(),
            "params": {"subject": job.subject, "run": job.run_label, "smoothings": SMOOTHINGS,
                       "smoothing_space": "T1w"},
            "elapsed_seconds": float(time.monotonic() - t0),
        },
    )


def run_cortex_stage(job: RunwiseJob, *, overwrite: bool = False) -> None:
    outdir = job.cortex_root
    if not overwrite:
        required_outputs = (
            outdir / "cortex_roi_parcel_timeseries.npy",
            outdir / "cortex_roi_parcels.tsv",
            outdir / "cortex_canonical7_networks.tsv",
            outdir / "cortex_canonical7_network_timeseries.npy",
            outdir / "cortex_family17_network_timeseries.npy",
        )
        if all(path.exists() for path in required_outputs) and cortex_qc_done(outdir):
            return
    outdir.mkdir(parents=True, exist_ok=True)

    lh_full = load_surface_label_vector(job.lh_labels)
    rh_full = load_surface_label_vector(job.rh_labels)

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
    dt_axis = dt_img.header.get_axis(1)

    left_vertices, left_dt = extract_structure_data(dt_axis, dt_data_t, "CORTEX_LEFT")
    right_vertices, right_dt = extract_structure_data(dt_axis, dt_data_t, "CORTEX_RIGHT")
    left_parcel_ids = align_surface_labels_to_dt_vertices(lh_full, left_vertices, hemi="L")
    right_parcel_ids = align_surface_labels_to_dt_vertices(rh_full, right_vertices, hemi="R")

    validity_dir = outdir / "validity"
    validity_dir.mkdir(parents=True, exist_ok=True)
    left_valid_mask = compute_cortex_validity(data=left_dt, outdir=validity_dir, hemi="left", dataset="iowa7t", task="rest", run_lengths=[n_timepoints])
    right_valid_mask = compute_cortex_validity(data=right_dt, outdir=validity_dir, hemi="right", dataset="iowa7t", task="rest", run_lengths=[n_timepoints])
    left_valid = load_required_mask(str(left_valid_mask), int(left_vertices.shape[0]), "left cortex validity")
    right_valid = load_required_mask(str(right_valid_mask), int(right_vertices.shape[0]), "right cortex validity")

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
        raise RuntimeError(f"No Kong2022 ROI timeseries remained after validity filtering for sub-{job.subject} {job.run_label}.")

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
        "excluded_by_validity",
        "excluded_by_no_vertices",
        "included_in_timeseries",
    ]
    write_tsv(outdir / "cortex_roi_parcels.tsv", roi_rows, roi_tsv_fields)
    write_tsv(outdir / "cortex_roi_parcels_empty_after_validity.tsv", empty_rows, roi_tsv_fields)

    canonical_rows: list[dict[str, object]] = []
    canonical_ts_rows: list[np.ndarray] = []
    for canonical in canonical7_order:
        idx = [i for i, row in enumerate(roi_rows) if str(row["canonical7"]) == canonical and bool(row["included_in_timeseries"])]
        if not idx:
            raise RuntimeError(f"Canonical network has zero parcels after validity filtering: {canonical}")
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
            raise RuntimeError(f"Family17 network has zero parcels after validity filtering: {family17}")
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

    write_json(
        outdir / "reference_summary.json",
        {
            "subject": job.subject,
            "run_label": job.run_label,
            "variant": job.variant,
            "dataset": "iowa7t",
            "task": "rest",
            "validity_policy": CORTEX_VALIDITY_POLICY,
            "network_taxonomy": "canonical7",
            "n_timepoints": n_timepoints,
            "n_parcels_total": 400,
            "n_parcels_used": int(len(kept_rows)),
            "n_parcels_missing": int(len(empty_rows)),
            "canonical7_networks_used": canonical7_order,
            "missing_roi_ids": missing_roi_ids,
        },
    )


def run_fc_stage(job: RunwiseJob, *, overwrite: bool = False) -> None:
    if not qc_policy_done(job) or not cortex_qc_done(job.cortex_root):
        raise ValueError("Rebuild Iowa7T runwise surface and cortex with signal validity before FC")
    roi_ts = np.load(job.cortex_root / "cortex_roi_parcel_timeseries.npy").astype(np.float32)
    roi_names, roi_ids, missing_roi_ids = read_roi_metadata(job.cortex_root / "cortex_roi_parcels.tsv")

    for smoothing in SMOOTHINGS:
        for hemi in HEMIS:
            fc_dir = job.fc_root / smoothing / f"hemi_{hemi}"
            fc_dir.mkdir(parents=True, exist_ok=True)
            out_fc = fc_dir / "hipp_vertex_to_network_fc.npy"
            summary_path = fc_dir / "fc_summary.json"
            if (not overwrite and out_fc.exists() and summary_path.is_file()
                    and json.loads(summary_path.read_text()).get("validity_policy") == SERIES_VALIDITY_POLICY):
                continue

            ts = np.load(job.surface_root / smoothing / f"sub-{job.subject}_hemi-{hemi}_timeseries.npy").astype(np.float32)
            valid_mask = np.load(job.surface_root / "validity" / f"sub-{job.subject}_hemi-{hemi}_valid_mask.npy").astype(bool)

            if ts.shape[1] != roi_ts.shape[1]:
                raise RuntimeError(f"Time mismatch: ts={ts.shape} roi={roi_ts.shape}")

            pearson_valid = mark_missing_roi_fc_columns(
                corrcoef_rows(ts[valid_mask, :], roi_ts),
                missing_roi_ids=missing_roi_ids,
                roi_ids=roi_ids,
            )
            pearson_full = np.full((ts.shape[0], roi_ts.shape[0]), np.nan, dtype=np.float32)
            pearson_full[valid_mask, :] = pearson_valid

            np.save(out_fc, pearson_full.astype(np.float32))
            np.save(fc_dir / "hipp_vertex_to_network_fc_active.npy", compact_active_vertices(pearson_full, valid_mask))
            # Direct compatibility alias for notebook consumption
            np.save(fc_dir / "hippo512_to_kong400_corr.npy", pearson_full.astype(np.float32))

            write_json(
                fc_dir / "fc_summary.json",
                {
                    "validity_policy": SERIES_VALIDITY_POLICY,
                    "subject": job.subject,
                    "variant": job.variant,
                    "run_label": job.run_label,
                    "smooth": smoothing,
                    "hemi": hemi,
                    "fc_shape": [int(pearson_full.shape[0]), int(pearson_full.shape[1])],
                    "n_timepoints": int(ts.shape[1]),
                    "n_vertices_valid": int(valid_mask.sum()),
                    "fc_definition": "pearson_vertex_to_kong2022_roi_timeseries",
                },
            )


def run_job(job: RunwiseJob, *, stage: str = "all", overwrite: bool = False) -> dict[str, object]:
    require_inputs(job)
    t0 = time.monotonic()
    if stage in {"all", "surface"}:
        run_surface_stage(job, overwrite=overwrite)
    if stage in {"all", "cortex"}:
        run_cortex_stage(job, overwrite=overwrite)
    if stage in {"all", "fc"}:
        run_fc_stage(job, overwrite=overwrite)
    return {
        "subject": job.subject,
        "variant": job.variant,
        "run_label": job.run_label,
        "elapsed_seconds": round(time.monotonic() - t0, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Iowa7T runwise downstream surface, cortex, and FC stages.")
    parser.add_argument("--subject", default=DEFAULT_SUBJECT)
    parser.add_argument("--variant", choices=["36p"], default="36p")
    parser.add_argument("--run", choices=["run-01", "run-02", "run-03", "run-04", "all"], default="all")
    parser.add_argument("--stage", choices=["all", "surface", "cortex", "fc"], default="all")
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--base-root", default=str(DEFAULT_BASE_ROOT))
    parser.add_argument("--run-tag", default=None, help="Run tag string (auto-detected from kong2022_runwise_<subject>_* if omitted).")
    args = parser.parse_args()

    variants = [args.variant]
    runs = RUN_SPECS if args.run == "all" else [s for s in RUN_SPECS if s["run_label"] == args.run]
    base_root = Path(args.base_root)
    subject = str(args.subject).strip().replace("sub-", "")

    for variant in variants:
        for spec in runs:
            job = resolve_runwise_job(subject, variant, spec, base_root=base_root, run_tag=args.run_tag)
            print(f"[start] {variant} {spec['run_label']} ({spec['acquisition']}) stage={args.stage} ...", flush=True)
            res = run_job(job, stage=args.stage, overwrite=args.overwrite)
            print(f"[done] {variant} {spec['run_label']} in {res['elapsed_seconds']}s", flush=True)

    print("\n[ALL COMPLETE] Runwise downstream processing finished successfully.")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    sys.exit(main())

#!/usr/bin/env python3
from __future__ import annotations

"""
Process Step Description:
Materialize HCP 3T task background (residual) functional connectivity.
Regresses the full first-level design (task + motion + intercept + 128-s DCT drift)
out of BOTH task-sampled hippocampal surface timeseries and task cortex
parcel timeseries per run, concatenates bilateral residuals across runs,
and correlates them with NaN propagation (missing vertices yield NaN FC,
never 0).

EXPLORATORY: outputs are not yet wired into the 05_01 gradient contract;
downstream task-gradient work must validate residual task coactivation
(e.g. flexible response bases) before formal use.

Main Flow Path:
- Stage-04 task branch (background connectivity, exploratory):
  02_01 (task surfaces) -> 03_02 (task cortex references) -> 04_04_materialize_background_fc.py (this step).

Inputs:
- Command-line arguments: --task, --surface-root, --shared-root, --reference-root, --task-manifest-root (required), --subjects/--subjects-manifest, --smoothings, --hemis, --jobs, --tr.
- Physical inputs:
  - Task surface timeseries + per-run signal-validity masks + per-run EVs/motion from the manifest-declared run directories.
  - Task cortex parcel timeseries: `{reference_root}/sub-<subject>/reference/kong2022/cortex_roi_parcel_timeseries.npy` with matching concatenated frames.

Outputs:
- Command-line output / file outputs:
  - Background vertex-to-ROI FC: `{shared_root}/sub-<subject>/task/{task}/{smoothing}/hemi_{hemi}/background_hipp_vertex_to_roi_fc.npy` plus `background_fc_summary.json`.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np

from lib.output_layout import DEFAULT_KONG2022_SHARED_INPUTS_ROOT
from lib.workflow_utils import capped_jobs, utc_now_iso, write_json
from lib.io_utils import load_subject_ids
from lib.task_design import (
    TASK_HIGH_PASS_PERIOD_S,
    TASK_RUN_LENGTHS,
    TR_HCP_3T,
    build_first_level_design,
    build_task_regressors,
    load_nuisance_regressors,
    normalize_task_branch,
)
from lib.task_glm import background_residuals, corrcoef_rows_nan, fit_first_level_glm
from lib.task_quality import require_task_surface_validity, require_task_policy
from lib.task_manifests import (
    task_run_manifest_path,
    task_runs_from_manifest,
    verify_cached_runs_against_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOOTHINGS = ["2mm"]
HEMIS = ["L", "R"]


def task_roots(task: str) -> tuple[Path, Path]:
    base = DEFAULT_KONG2022_SHARED_INPUTS_ROOT
    return base / f"task-{task}" / "surface", base / f"task-{task}"


def run_subject(
    *,
    subject: str,
    task: str,
    surface_root: Path,
    shared_root: Path,
    reference_root: Path,
    task_manifest_root: Path | None,
    smoothings: list[str],
    hemis: list[str],
    tr: float,
) -> tuple[str, bool, str]:
    try:
        if task_manifest_root is None:
            raise ValueError(
                "--task-manifest-root is required: run identity and order "
                "must come from the manifest, never from frame totals"
            )
        manifest_runs = task_runs_from_manifest(
            task_run_manifest_path(task_manifest_root, subject=subject, task=task),
            subject=subject,
        )
        expected = int(TASK_RUN_LENGTHS[task])
        runs: list[tuple[int, Path, int]] = []
        header_lengths: list[int] = []
        for record in manifest_runs:
            bold_img = nib.load(str(record["bold"]))
            if len(bold_img.shape) != 4:
                raise ValueError(f"Expected 4D run volume: {record['bold']}")
            length = int(bold_img.shape[3])
            if length != expected:
                raise ValueError(
                    f"Task {task} run-{record['run_id']} expects {expected} frames, got {length}"
                )
            runs.append((int(record["run_id"]), Path(record["run_dir"]), length))
            header_lengths.append(length)
        surface_receipt_path = (
            surface_root
            / f"sub-{subject}"
            / "surface"
            / "raw"
            / "surface_sampling_summary.json"
        )
        cortex_receipt_path = (
            reference_root
            / f"sub-{subject}"
            / "reference"
            / "kong2022"
            / "reference_summary.json"
        )
        for receipt_path in (surface_receipt_path, cortex_receipt_path):
            if not receipt_path.is_file():
                raise FileNotFoundError(
                    f"Missing sampling receipt: {receipt_path}"
                )
        surface_receipt = json.loads(surface_receipt_path.read_text(encoding="utf-8"))
        cortex_receipt = json.loads(cortex_receipt_path.read_text(encoding="utf-8"))
        require_task_policy(cortex_receipt, cortex_receipt_path, task)
        run_verification = {
            "surface": verify_cached_runs_against_manifest(
                subject=subject,
                task=task,
                cached_kind="surface sampling",
                cached_summary_path=surface_receipt_path,
                cached_run_ids=[int(v) for v in surface_receipt.get("run_ids", [])],
                cached_run_lengths=[int(v) for v in surface_receipt.get("run_lengths", [])],
                cached_source_paths=[str(v) for v in surface_receipt.get("bold_inputs", [])],
                manifest_runs=manifest_runs,
                manifest_source_key="bold",
                manifest_lengths=header_lengths,
            ),
            "cortex": verify_cached_runs_against_manifest(
                subject=subject,
                task=task,
                cached_kind="cortex reference",
                cached_summary_path=cortex_receipt_path,
                cached_run_ids=[int(v) for v in cortex_receipt.get("run_ids", [])],
                cached_run_lengths=[int(v) for v in cortex_receipt.get("run_lengths", [])],
                cached_source_paths=[str(v) for v in cortex_receipt.get("dtseries_inputs", [])],
                manifest_runs=manifest_runs,
                manifest_source_key="dtseries",
                manifest_lengths=header_lengths,
            ),
        }
        total = int(sum(length for _, _, length in runs))
        ref_ts_path = (
            reference_root
            / f"sub-{subject}"
            / "reference"
            / "kong2022"
            / "cortex_roi_parcel_timeseries.npy"
        )
        if not ref_ts_path.is_file():
            raise FileNotFoundError(f"Missing task cortex reference: {ref_ts_path}")
        cortex = np.load(str(ref_ts_path)).astype(np.float64)
        if cortex.ndim != 2 or cortex.shape[1] != total:
            raise ValueError(
                f"Cortex frame mismatch: {ref_ts_path} shape={cortex.shape} expected (*, {total})"
            )
        for smoothing in smoothings:
            for hemi in hemis:
                ts_path = (
                    surface_root
                    / f"sub-{subject}"
                    / "surface"
                    / smoothing
                    / f"sub-{subject}_hemi-{hemi}_timeseries.npy"
                )
                if not ts_path.is_file():
                    raise FileNotFoundError(f"Missing task surface timeseries: {ts_path}")
                mask_path = (
                    surface_root
                    / f"sub-{subject}"
                    / "surface"
                    / "tsnr_gate"
                    / f"sub-{subject}_hemi-{hemi}_valid_mask.npy"
                )
                if not mask_path.is_file():
                    raise FileNotFoundError(f"Missing tSNR valid mask: {mask_path}")
                require_task_surface_validity(surface_root, subject, hemi, task)
                valid = np.load(str(mask_path)).astype(bool).reshape(-1)
                ts = np.load(str(ts_path)).astype(np.float64)
                if valid.shape[0] != ts.shape[0]:
                    raise ValueError(f"Mask/vertex mismatch: {valid.shape} vs {ts.shape}")
                if ts.ndim != 2 or ts.shape[1] != total:
                    raise ValueError(
                        f"Frame mismatch: {ts_path} shape={ts.shape} expected (*, {total})"
                    )
                ts[~valid, :] = np.nan
                hipp_blocks: list[np.ndarray] = []
                ctx_blocks: list[np.ndarray] = []
                offset = 0
                n_task_regs: int | None = None
                for run_id, run_dir, length in runs:
                    ev_dir = run_dir / "EVs"
                    mov_path = run_dir / "Movement_Regressors.txt"
                    if not ev_dir.is_dir():
                        raise FileNotFoundError(f"Missing EVs dir: {ev_dir}")
                    if not mov_path.is_file():
                        raise FileNotFoundError(f"Missing movement regressors: {mov_path}")
                    names, task_regs = build_task_regressors(
                        ev_dir, task=task, n_frames=length, tr=tr
                    )
                    nuisance = load_nuisance_regressors(mov_path, n_frames=length)
                    design = build_first_level_design(task_regs, nuisance, tr=tr)
                    if n_task_regs is None:
                        n_task_regs = len(names)
                    hipp_blocks.append(
                        background_residuals(
                            ts[:, offset : offset + length], design, n_task_regs
                        )
                    )
                    cortex_block = cortex[:, offset : offset + length]
                    _, cortex_resid, _ = fit_first_level_glm(cortex_block, design)
                    ctx_blocks.append(cortex_resid)
                    offset += length
                hipp_bg = np.concatenate(hipp_blocks, axis=1)
                ctx_bg = np.concatenate(ctx_blocks, axis=1)
                fc = corrcoef_rows_nan(hipp_bg, ctx_bg)
                out_dir = shared_root / f"sub-{subject}" / "task" / task / smoothing / f"hemi_{hemi}"
                out_dir.mkdir(parents=True, exist_ok=True)
                np.save(
                    str(out_dir / "background_hipp_vertex_to_roi_fc.npy"),
                    np.asarray(fc, dtype=np.float32),
                )
                write_json(
                    out_dir / "background_fc_summary.json",
                    {
                        "subject": subject,
                        "task": task,
                        "smoothing": smoothing,
                        "hemi": hemi,
                        "analysis": "exploratory_bilateral_residuals_nan",
                        "tr": tr,
                        "drift_model": "dct",
                        "high_pass_period_s": TASK_HIGH_PASS_PERIOD_S,
                        "run_ids": [run_id for run_id, _, _ in runs],
                        "run_labels": [run_dir.name for _, run_dir, _ in runs],
                        "run_lengths": [length for _, _, length in runs],
                        "run_verification": run_verification,
                        "n_vertices_total": int(valid.shape[0]),
                        "n_vertices_valid": int(np.sum(valid)),
                        "n_nan_fc": int(np.sum(~np.isfinite(fc))),
                        "cortex_reference": str(ref_ts_path.resolve()),
                        "created_utc": utc_now_iso(),
                    },
                )
        return subject, True, "ok"
    except Exception as exc:
        return subject, False, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Task background FC (exploratory).")
    parser.add_argument("--task", default="emotion")
    parser.add_argument("--surface-root", default=None)
    parser.add_argument("--shared-root", default=None)
    parser.add_argument("--reference-root", default=None)
    parser.add_argument(
        "--task-manifest-root",
        default=None,
        help="Root holding task-<task>/sub-<subject>.json run-artifact manifests (required).",
    )
    parser.add_argument("--subjects-manifest", default=None)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--smoothings", nargs="*", default=SMOOTHINGS)
    parser.add_argument("--hemis", nargs="*", default=HEMIS)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--tr", type=float, default=TR_HCP_3T)
    args = parser.parse_args()

    task = normalize_task_branch(args.task)
    default_surface, default_shared = task_roots(task)
    surface_root = Path(args.surface_root) if args.surface_root else default_surface
    shared_root = Path(args.shared_root) if args.shared_root else default_shared
    reference_root = Path(args.reference_root) if args.reference_root else default_shared
    if args.subjects:
        subjects = sorted({s.replace("sub-", "") for s in args.subjects})
    elif args.subjects_manifest:
        subjects = sorted(load_subject_ids(Path(args.subjects_manifest)))
    else:
        subjects = sorted(
            p.name.replace("sub-", "")
            for p in surface_root.glob("sub-*")
            if p.is_dir()
        )
    if not subjects:
        raise SystemExit("No subjects found")
    jobs = capped_jobs(args.jobs, len(subjects))
    failed: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(
                run_subject,
                subject=s,
                task=task,
                surface_root=surface_root,
                shared_root=shared_root,
                reference_root=reference_root,
                task_manifest_root=Path(args.task_manifest_root) if args.task_manifest_root else None,
                smoothings=list(args.smoothings),
                hemis=list(args.hemis),
                tr=float(args.tr),
            ): s
            for s in subjects
        }
        for fut in as_completed(futures):
            subject, ok, msg = fut.result()
            print(json.dumps({"subject": subject, "ok": ok, "msg": msg}))
            if not ok:
                failed[subject] = msg
    if failed:
        raise SystemExit(f"{len(failed)} subjects failed: {sorted(failed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

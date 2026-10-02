#!/usr/bin/env python3
from __future__ import annotations

"""
Process Step Description:
Materialize HCP 3T task-fMRI first-level activation. Fits a per-vertex OLS
GLM on task-sampled hippocampal surface timeseries with HRF-convolved FSL
EV regressors plus 12-column movement nuisance, and writes run-averaged
beta and contrast maps (faces>shapes, 2bk>0bk, rel>match).

Main Flow Path:
- Stage-04 task branch (task activation):
  02_01 (task volumes sampled to den-512 surfaces) -> 04_03_materialize_task_activation.py (this step) -> 04_04 background FC -> 05_01 task gradients -> 05_02 group atlas -> 06_xx presentation.

Inputs:
- Command-line arguments:
  - --task: Task branch: emotion | wm | relational (default: emotion).
  - --surface-root: Task hippocampal surface root holding concatenated run timeseries (default: `outputs_hcp/inputs/hcp/kong2022/shared_inputs/task-emotion/surface`).
  - --shared-root: Task activation output root (default: `outputs_hcp/inputs/hcp/kong2022/shared_inputs/task-emotion`).
  - --task-manifest-root: Root holding per-subject run-artifact manifests (required; run identity, order, and EV/motion paths come from the manifest, never from frame totals).
  - --subjects-manifest / --subjects / --smoothings / --hemis / --jobs / --tr.
- Physical inputs:
  - Concatenated task surface timeseries: `{surface_root}/sub-<subject>/surface/{smoothing}/sub-<subject>_hemi-{hemi}_timeseries.npy`.
  - Task signal-validity mask: `{surface_root}/sub-<subject>/surface/tsnr_gate/sub-<subject>_hemi-{hemi}_valid_mask.npy`; invalid vertices are NaNed before the GLM so ROI-smoothing zeros never enter as responses.
  - Per-run `EVs/<stem>.txt` and `Movement_Regressors.txt` under the manifest-declared run directories.
- Run-order verification: the cached surface series is only trusted when the
  `surface/raw/surface_sampling_summary.json` receipt matches the manifest
  run ids, lengths, and bold source paths entry-by-entry (total-frame
  equality cannot catch equal-length LR/RL swaps).

Outputs:
- Command-line output / file outputs:
  - Run-averaged betas: `{shared_root}/sub-<subject>/task/{task}/{smoothing}/hemi_{hemi}/task_betas.npy`.
  - Contrast maps: `.../contrast_{name}.npy` plus per-run `.../contrast_{name}_run-{id}.npy` and `task_activation_summary.json` (with vertex coverage).
"""
import argparse
import json
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np

from lib.output_layout import DEFAULT_KONG2022_SHARED_INPUTS_ROOT
from lib.workflow_utils import capped_jobs, utc_now_iso, write_json
from lib.io_utils import load_subject_ids
from lib.task_design import (
    TASK_CONTRASTS,
    TASK_RUN_LENGTHS,
    TR_HCP_3T,
    build_first_level_design,
    build_task_regressors,
    contrast_vector,
    load_nuisance_regressors,
    normalize_task_branch,
)
from lib.task_glm import apply_contrast, fit_first_level_glm
from lib.task_quality import require_task_surface_validity
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
    task_manifest_root: Path | None,
    smoothings: list[str],
    hemis: list[str],
    tr: float,
) -> tuple[str, bool, str]:
    try:
        manifest_runs: list[dict[str, object]] | None = None
        if task_manifest_root is not None:
            manifest_runs = task_runs_from_manifest(
                task_run_manifest_path(task_manifest_root, subject=subject, task=task),
                subject=subject,
            )
        expected = int(TASK_RUN_LENGTHS[task])
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
                    raise FileNotFoundError(f"Missing Task signal-validity mask: {mask_path}")
                require_task_surface_validity(surface_root, subject, hemi, task)
                valid = np.load(str(mask_path)).astype(bool).reshape(-1)
                ts = np.load(str(ts_path)).astype(np.float64)
                if valid.shape[0] != ts.shape[0]:
                    raise ValueError(
                        f"Mask/vertex mismatch: mask={valid.shape} ts={ts.shape}"
                    )
                # Invalid (QC-masked) vertices carry exact zeros from ROI
                # smoothing; NaN them so no zero response enters the GLM.
                ts[~valid, :] = np.nan
                n_valid = int(np.sum(valid))
                if manifest_runs is not None:
                    runs: list[tuple[int, Path, int]] = []
                    header_lengths: list[int] = []
                    for record in manifest_runs:
                        bold_img = nib.load(str(record["bold"]))
                        if len(bold_img.shape) != 4:
                            raise ValueError(
                                f"Expected 4D run volume: {record['bold']}"
                            )
                        length = int(bold_img.shape[3])
                        if length != expected:
                            raise ValueError(
                                f"Task {task} run-{record['run_id']} expects "
                                f"{expected} frames, got {length}"
                            )
                        runs.append(
                            (int(record["run_id"]), Path(record["run_dir"]), length)
                        )
                        header_lengths.append(length)
                    sampling_summary_path = (
                        surface_root
                        / f"sub-{subject}"
                        / "surface"
                        / "raw"
                        / "surface_sampling_summary.json"
                    )
                    if not sampling_summary_path.is_file():
                        raise FileNotFoundError(
                            f"Missing surface sampling receipt: {sampling_summary_path}"
                        )
                    sampling_summary = json.loads(
                        sampling_summary_path.read_text(encoding="utf-8")
                    )
                    run_verification = verify_cached_runs_against_manifest(
                        subject=subject,
                        task=task,
                        cached_kind="surface sampling",
                        cached_summary_path=sampling_summary_path,
                        cached_run_ids=[int(v) for v in sampling_summary.get("run_ids", [])],
                        cached_run_lengths=[int(v) for v in sampling_summary.get("run_lengths", [])],
                        cached_source_paths=[str(v) for v in sampling_summary.get("bold_inputs", [])],
                        manifest_runs=manifest_runs,
                        manifest_source_key="bold",
                        manifest_lengths=header_lengths,
                    )
                else:
                    raise ValueError(
                        "--task-manifest-root is required: run identity and order "
                        "must come from the manifest, never from frame totals"
                    )
                total = int(sum(length for _, _, length in runs))
                if ts.ndim != 2 or ts.shape[1] != total:
                    raise ValueError(
                        f"Frame mismatch: {ts_path} shape={ts.shape} expected (*, {total})"
                    )
                betas_runs: list[np.ndarray] = []
                r2_runs: list[np.ndarray] = []
                regressor_names: list[str] | None = None
                run_contrasts: dict[str, dict[str, str]] = {}
                offset = 0
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
                    design = build_first_level_design(task_regs, nuisance)
                    block = ts[:, offset : offset + length]
                    betas, _, r2 = fit_first_level_glm(block, design)
                    betas_runs.append(betas[:, : len(names)])
                    r2_runs.append(r2)
                    if regressor_names is None:
                        regressor_names = names
                    elif regressor_names != names:
                        raise ValueError(
                            f"Regressor order changed between runs: run-{run_id}"
                        )
                    offset += length
                assert regressor_names is not None
                with warnings.catch_warnings():
                    # Fully invalid vertices stay NaN in every run; silence the
                    # expected empty-slice warning for those vertices.
                    warnings.filterwarnings("ignore", message="Mean of empty slice")
                    betas_mean = np.nanmean(np.stack(betas_runs, axis=0), axis=0)
                    r2_mean = np.nanmean(np.stack(r2_runs, axis=0), axis=0)
                out_dir = shared_root / f"sub-{subject}" / "task" / task / smoothing / f"hemi_{hemi}"
                out_dir.mkdir(parents=True, exist_ok=True)
                np.save(str(out_dir / "task_betas.npy"), betas_mean.astype(np.float32))
                np.save(str(out_dir / "task_r2_mean.npy"), r2_mean.astype(np.float32))
                contrasts: dict[str, str] = {}
                for cname, weights in TASK_CONTRASTS[task].items():
                    vec = contrast_vector(regressor_names, weights)
                    cmap = apply_contrast(betas_mean, vec)
                    cpath = out_dir / f"contrast_{cname}.npy"
                    np.save(str(cpath), cmap.astype(np.float32))
                    contrasts[cname] = str(cpath.resolve())
                    for run_index, (run_id, _, _) in enumerate(runs):
                        run_cmap = apply_contrast(betas_runs[run_index], vec)
                        run_cpath = out_dir / f"contrast_{cname}_run-{run_id}.npy"
                        np.save(str(run_cpath), run_cmap.astype(np.float32))
                        run_contrasts.setdefault(cname, {})[str(run_id)] = str(
                            run_cpath.resolve()
                        )
                write_json(
                    out_dir / "task_activation_summary.json",
                    {
                        "subject": subject,
                        "task": task,
                        "smoothing": smoothing,
                        "hemi": hemi,
                        "tr": tr,
                        "run_ids": [run_id for run_id, _, _ in runs],
                        "run_labels": [run_dir.name for _, run_dir, _ in runs],
                        "run_lengths": [length for _, _, length in runs],
                        "run_verification": run_verification,
                        "regressor_names": regressor_names,
                        "n_vertices_total": int(valid.shape[0]),
                        "n_vertices_valid": n_valid,
                        "coverage_fraction": float(n_valid / max(1, valid.shape[0])),
                        "contrasts": contrasts,
                        "run_contrasts": run_contrasts,
                        "created_utc": utc_now_iso(),
                    },
                )
        return subject, True, "ok"
    except Exception as exc:  # fail closed per subject, never partial-write contrasts
        return subject, False, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Task first-level activation.")
    parser.add_argument("--task", default="emotion")
    parser.add_argument("--surface-root", default=None)
    parser.add_argument("--shared-root", default=None)
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

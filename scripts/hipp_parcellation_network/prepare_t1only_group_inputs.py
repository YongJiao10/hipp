#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.gifti import GiftiDataArray, GiftiImage


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BATCH_ROOT = REPO_ROOT / "outputs_experiments" / "hippunfold_recommended_t1w_20260423" / "dense_corobl_batch"
DEFAULT_ROUTE_ROOT = REPO_ROOT / "outputs_experiments" / "hippunfold_recommended_t1w_20260423" / "kong2022_route_b"
DEFAULT_SHARED_ROOT = DEFAULT_ROUTE_ROOT / "hipp_group_inputs_shared"
DEFAULT_BOLD_ROOT = Path("/home/yojiao/LSS/jianglab/3_Data_Working/HCP_7T_Hippocampus/rsfmri_volume")
SAMPLE_SCRIPT = REPO_ROOT / "scripts" / "common" / "sample_hipp_surface_timeseries.py"
QC_SCRIPT = REPO_ROOT / "scripts" / "hipp_parcellation_network" / "build_t1only_qc_manifest.py"
WB_COMMAND = REPO_ROOT / "scripts" / "wb_command"
MAX_CONCURRENCY = 10
HEMIS = ["L", "R"]
SMOOTHINGS = ["2mm", "4mm"]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_commit() -> str | None:
    proc = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True)
    return proc.stdout.strip() if proc.returncode == 0 else None


def file_meta(path: Path) -> dict[str, object]:
    st = path.stat()
    return {"path": str(path.resolve()), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
        return sorted(
            line.strip().replace("sub-", "")
            for line in subjects_manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
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


def raw_done(shared_root: Path, subject: str) -> bool:
    raw_dir = shared_root / f"sub-{subject}" / "surface" / "raw"
    return all(
        (raw_dir / f"sub-{subject}_hemi-{hemi}_space-corobl_den-512_label-hipp_bold.func.gii").exists()
        for hemi in HEMIS
    ) and (raw_dir / "surface_sampling_summary.json").exists()


def run_raw_sampling(subject: str, batch_root: str, shared_root: str, bold_root: str, log_dir: str) -> dict[str, object]:
    batch = Path(batch_root)
    shared = Path(shared_root)
    bold = Path(bold_root) / subject / "func_uncompressed.nii"
    out_dir = shared / f"sub-{subject}" / "surface" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f"sub-{subject}_raw_sampling.log"
    cmd = [
        sys.executable,
        str(SAMPLE_SCRIPT),
        "--bold",
        str(bold),
        "--hippunfold-dir",
        str(batch / f"sub-{subject}" / "hippunfold"),
        "--subject",
        subject,
        "--density",
        "512",
        "--space",
        "corobl",
        "--mapping-method",
        "trilinear",
        "--smooth-iters",
        "0",
        "--outdir",
        str(out_dir),
    ]
    t0 = time.monotonic()
    proc = subprocess.run(cmd, text=True, capture_output=True, cwd=str(SAMPLE_SCRIPT.parent))
    log_path.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        return {"subject": subject, "ok": False, "stage": "raw_sampling", "error": proc.stderr[-4000:]}
    save_json(
        out_dir / "stage_manifest.json",
        {
            "stage": "t1only_raw_surface_sampling",
            "status": "done",
            "timestamp_utc": now_utc(),
            "git_commit": git_commit(),
            "command": cmd,
            "inputs": [file_meta(bold)],
            "outputs": [file_meta(p) for p in out_dir.glob("*") if p.is_file()],
            "elapsed_seconds": float(time.monotonic() - t0),
        },
    )
    return {"subject": subject, "ok": True, "stage": "raw_sampling"}


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


def surface_path(batch_root: Path, subject: str, hemi: str) -> Path:
    return (
        batch_root
        / f"sub-{subject}"
        / "hippunfold"
        / f"sub-{subject}"
        / "surf"
        / f"sub-{subject}_hemi-{hemi}_space-corobl_den-512_label-hipp_midthickness.surf.gii"
    )


def run_tsnr_gate(subject: str, batch_root: str, shared_root: str) -> dict[str, object]:
    batch = Path(batch_root)
    shared = Path(shared_root)
    gate_dir = shared / f"sub-{subject}" / "surface" / "tsnr_gate"
    gate_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, object]] = []
    inputs: list[dict[str, object]] = []
    for hemi in HEMIS:
        raw = shared / f"sub-{subject}" / "surface" / "raw" / f"sub-{subject}_hemi-{hemi}_space-corobl_den-512_label-hipp_bold.func.gii"
        surf = surface_path(batch, subject, hemi)
        data = load_func_gii(raw).astype(np.float64)
        sd = np.nanstd(data, axis=1, ddof=1)
        tsnr = np.where(sd > 0, 10000.0 / sd, np.nan).astype(np.float32)
        valid = np.isfinite(tsnr) & (tsnr >= 25.0)
        invalid = ~valid
        np.save(gate_dir / f"sub-{subject}_hemi-{hemi}_tsnr.npy", tsnr)
        np.save(gate_dir / f"sub-{subject}_hemi-{hemi}_valid_mask.npy", valid.astype(bool))
        np.save(gate_dir / f"sub-{subject}_hemi-{hemi}_invalid_initial_mask.npy", invalid.astype(bool))
        nib.save(make_shape_gifti(valid.astype(np.float32)), str(gate_dir / f"sub-{subject}_hemi-{hemi}_valid_roi.shape.gii"))
        summary = {
            "subject": subject,
            "hemi": hemi,
            "threshold": 25.0,
            "tsnr_definition": "10000/std",
            "n_vertices_total": int(valid.size),
            "n_vertices_valid": int(valid.sum()),
            "n_vertices_invalid": int(invalid.sum()),
            "invalid_fraction": float(invalid.sum() / max(1, valid.size)),
        }
        save_json(gate_dir / f"sub-{subject}_hemi-{hemi}_tsnr_gate_summary.json", summary)
        hemi_dir = gate_dir / f"hemi-{hemi}"
        save_json(
            hemi_dir / "stage_manifest.json",
            {
                "stage": "t1only_hipp_tsnr_gate",
                "status": "done",
                "timestamp_utc": now_utc(),
                "git_commit": git_commit(),
                "params": {"subject": subject, "hemi": hemi, "threshold": 25.0, "tsnr_definition": "10000/std"},
                "inputs": [file_meta(raw), file_meta(surf)],
                "outputs": [
                    file_meta(gate_dir / f"sub-{subject}_hemi-{hemi}_tsnr.npy"),
                    file_meta(gate_dir / f"sub-{subject}_hemi-{hemi}_valid_mask.npy"),
                    file_meta(gate_dir / f"sub-{subject}_hemi-{hemi}_invalid_initial_mask.npy"),
                    file_meta(gate_dir / f"sub-{subject}_hemi-{hemi}_valid_roi.shape.gii"),
                    file_meta(gate_dir / f"sub-{subject}_hemi-{hemi}_tsnr_gate_summary.json"),
                ],
            },
        )
        inputs.extend([file_meta(raw), file_meta(surf)])
        outputs.extend([file_meta(p) for p in gate_dir.glob(f"sub-{subject}_hemi-{hemi}_*") if p.is_file()])
    return {"subject": subject, "ok": True, "stage": "tsnr_gate", "inputs": inputs, "outputs": outputs}


def smoothing_done(shared_root: Path, subject: str) -> bool:
    for smoothing in SMOOTHINGS:
        for hemi in HEMIS:
            p = shared_root / f"sub-{subject}" / "surface" / smoothing / f"sub-{subject}_hemi-{hemi}_timeseries.npy"
            if not p.exists():
                return False
    return True


def run_smoothing(subject: str, batch_root: str, shared_root: str) -> dict[str, object]:
    batch = Path(batch_root)
    shared = Path(shared_root)
    inputs: list[dict[str, object]] = []
    outputs: list[dict[str, object]] = []
    for smoothing in SMOOTHINGS:
        fwhm = float(smoothing.replace("mm", ""))
        out_dir = shared / f"sub-{subject}" / "surface" / smoothing
        out_dir.mkdir(parents=True, exist_ok=True)
        for hemi in HEMIS:
            raw = shared / f"sub-{subject}" / "surface" / "raw" / f"sub-{subject}_hemi-{hemi}_space-corobl_den-512_label-hipp_bold.func.gii"
            roi = shared / f"sub-{subject}" / "surface" / "tsnr_gate" / f"sub-{subject}_hemi-{hemi}_valid_roi.shape.gii"
            surf = surface_path(batch, subject, hemi)
            out_gii = out_dir / f"sub-{subject}_hemi-{hemi}_space-corobl_den-512_label-hipp_bold.func.gii"
            cmd = [
                str(WB_COMMAND),
                "-metric-smoothing",
                str(surf),
                str(raw),
                str(fwhm),
                str(out_gii),
                "-fwhm",
                "-roi",
                str(roi),
            ]
            proc = subprocess.run(cmd, text=True, capture_output=True)
            if proc.returncode != 0:
                return {"subject": subject, "ok": False, "stage": "smoothing", "error": proc.stderr[-4000:]}
            arr = load_func_gii(out_gii)
            np.save(out_dir / f"sub-{subject}_hemi-{hemi}_timeseries.npy", arr.astype(np.float32))
            inputs.extend([file_meta(raw), file_meta(roi), file_meta(surf)])
            outputs.extend([file_meta(out_gii), file_meta(out_dir / f"sub-{subject}_hemi-{hemi}_timeseries.npy")])
    save_json(
        shared / f"sub-{subject}" / "surface" / "stage_manifest.json",
        {
            "stage": "t1only_surface_smoothing",
            "status": "done",
            "timestamp_utc": now_utc(),
            "git_commit": git_commit(),
            "params": {"subject": subject, "smoothings": SMOOTHINGS, "smoothing_roi_policy": "high_tsnr_vertices_only"},
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


def build_group_average_surfaces(batch_root: Path, route_root: Path, subjects: list[str]) -> None:
    out_dir = route_root / "hipp_group_merged_canonical_atlas" / "group_average_surfaces"
    outputs: list[dict[str, object]] = []
    inputs: list[dict[str, object]] = []
    for hemi in HEMIS:
        coords_all: list[np.ndarray] = []
        ref_faces: np.ndarray | None = None
        for subject in subjects:
            p = surface_path(batch_root, subject, hemi)
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
    save_json(
        out_dir / "stage_manifest.json",
        {
            "stage": "t1only_group_average_surfaces",
            "status": "done",
            "timestamp_utc": now_utc(),
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
    return [line.strip().replace("sub-", "") for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare T1only shared hippocampal inputs and group-average surfaces.")
    parser.add_argument("--hippunfold-batch-root", default=str(DEFAULT_BATCH_ROOT))
    parser.add_argument("--route-root", default=str(DEFAULT_ROUTE_ROOT))
    parser.add_argument("--shared-root", default=str(DEFAULT_SHARED_ROOT))
    parser.add_argument("--bold-root", default=str(DEFAULT_BOLD_ROOT))
    parser.add_argument("--subjects-manifest", default=None)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--stage", choices=["all", "raw", "tsnr", "qc", "smooth", "group-average"], default="all")
    parser.add_argument("--dice-threshold", type=float, default=0.70)
    args = parser.parse_args()

    batch_root = Path(args.hippunfold_batch_root).resolve()
    route_root = Path(args.route_root).resolve()
    shared_root = Path(args.shared_root).resolve()
    bold_root = Path(args.bold_root).resolve()
    log_dir = route_root / "logs" / "prepare_t1only_group_inputs"
    log_dir.mkdir(parents=True, exist_ok=True)
    shared_root.mkdir(parents=True, exist_ok=True)
    max_concurrency = resolve_max_concurrency()
    requested_jobs = int(args.jobs)
    jobs = max(1, min(requested_jobs, max_concurrency))
    if requested_jobs > max_concurrency:
        print(
            f"[warn] requested jobs={requested_jobs} exceeds max_concurrency={max_concurrency}; using jobs={jobs}",
            flush=True,
        )

    subjects_manifest = Path(args.subjects_manifest).resolve() if args.subjects_manifest else None
    subjects = load_subjects(batch_root, bold_root, subjects_manifest, args.subjects)
    dice_subjects = dice_pass_subjects(batch_root, subjects, args.dice_threshold)
    print(f"[info] subjects={len(subjects)} dice_pass={len(dice_subjects)} jobs={jobs}", flush=True)

    if args.stage in {"all", "raw"}:
        raw_todo = [s for s in dice_subjects if not (args.resume and raw_done(shared_root, s))]
        print(f"[info] raw sampling todo={len(raw_todo)}", flush=True)
        run_parallel("raw", raw_todo, jobs, run_raw_sampling, str(batch_root), str(shared_root), str(bold_root), str(log_dir))
    if args.stage in {"all", "tsnr"}:
        tsnr_todo = [s for s in dice_subjects]
        print(f"[info] tsnr gate todo={len(tsnr_todo)}", flush=True)
        run_parallel("tsnr", tsnr_todo, jobs, run_tsnr_gate, str(batch_root), str(shared_root))
    if args.stage in {"all", "qc"}:
        cmd = [
            sys.executable,
            str(QC_SCRIPT),
            "--hippunfold-batch-root",
            str(batch_root),
            "--shared-root",
            str(shared_root),
            "--out-dir",
            str(route_root / "qc"),
            "--manifest-dir",
            str(route_root / "manifests"),
        ]
        cmd.extend(["--subjects", *subjects])
        proc = subprocess.run(cmd, text=True)
        if proc.returncode != 0:
            return proc.returncode
    pass_subjects = read_latest_manifest(route_root) if (route_root / "manifests" / "hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt").exists() else []
    if args.stage in {"all", "smooth"}:
        smooth_todo = [s for s in pass_subjects if not (args.resume and smoothing_done(shared_root, s))]
        print(f"[info] smoothing pass_subjects={len(pass_subjects)} todo={len(smooth_todo)}", flush=True)
        run_parallel("smooth", smooth_todo, jobs, run_smoothing, str(batch_root), str(shared_root))
    if args.stage in {"all", "group-average"}:
        print(f"[info] group-average subjects={len(pass_subjects)}", flush=True)
        build_group_average_surfaces(batch_root, route_root, pass_subjects)

    save_json(
        route_root / "prepare_t1only_group_inputs_manifest.json",
        {
            "stage": "prepare_t1only_group_inputs",
            "status": "done",
            "timestamp_utc": now_utc(),
            "git_commit": git_commit(),
            "hippunfold_batch_root": str(batch_root),
            "shared_root": str(shared_root),
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

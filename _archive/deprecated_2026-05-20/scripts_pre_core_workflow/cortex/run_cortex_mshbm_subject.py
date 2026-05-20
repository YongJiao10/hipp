#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
WB_COMMAND = str((REPO_ROOT / "scripts" / "wb_command").resolve())
DEFAULT_OUT_ROOT = REPO_ROOT / "outputs_experiments" / "cortex_kong2022_areal_mshbm"
DEFAULT_CBIG_CODE_DIR = REPO_ROOT / "external" / "CBIG"
DEFAULT_KONG2022_DLABEL = (
    REPO_ROOT
    / "external"
    / "Schaefer2018_Kong2022"
    / "Schaefer2018_400Parcels_Kong2022_17Networks_order.dlabel.nii"
)
DEFAULT_SCENE = REPO_ROOT / "config" / "manual_wb_scenes" / "cortex_manual.scene"
DEFAULT_RUN_LENGTHS = [900, 900, 900, 900]
ATLAS_SLUG = "kong2022_areal_mshbm"
N_ROIS = 400


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_meta(path: Path) -> dict[str, object]:
    st = path.stat()
    return {"path": str(path), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed: "
            + " ".join(cmd)
            + "\nSTDOUT:\n"
            + (proc.stdout or "")
            + "\nSTDERR:\n"
            + (proc.stderr or "")
        )
    return proc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Kong2022_ArealMSHBM single-subject parcellation for fs_LR_32k and emit subject dlabel."
    )
    p.add_argument("--subject", required=True, help="Subject ID without sub- prefix")
    p.add_argument("--dtseries", required=True)
    p.add_argument("--left-inflated", required=True)
    p.add_argument("--right-inflated", required=True)
    p.add_argument("--left-midthickness", required=True)
    p.add_argument("--right-midthickness", required=True)
    p.add_argument("--sulc-dscalar", required=True)
    p.add_argument("--cbig-code-dir", default=str(DEFAULT_CBIG_CODE_DIR))
    p.add_argument("--group-prior", default=None)
    p.add_argument("--kong2022-dlabel", default=str(DEFAULT_KONG2022_DLABEL))
    p.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--model", default="gMSHBM", choices=["gMSHBM", "cMSHBM", "dMSHBM"])
    p.add_argument("--num-rois", type=int, default=400)
    p.add_argument("--w", type=int, default=30)
    p.add_argument("--c", type=int, default=30)
    p.add_argument("--beta", type=int, default=50)
    p.add_argument("--run-lengths", nargs="+", type=int, default=DEFAULT_RUN_LENGTHS)
    p.add_argument("--matlab-cmd", default="matlab")
    p.add_argument(
        "--subject-dir-layout",
        default="nested",
        choices=["nested", "flat"],
        help="nested: sub-<ID>/<atlas_slug>/... ; flat: sub-<ID>/...",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def find_matlab_executable(matlab_cmd: str) -> str | None:
    if "/" in matlab_cmd or Path(matlab_cmd).is_absolute():
        p = Path(matlab_cmd)
        return str(p.resolve()) if p.exists() else None
    resolved = shutil.which(matlab_cmd)
    return str(Path(resolved).resolve()) if resolved else None


def find_file_in_tree(root: Path, filename: str) -> Path | None:
    items = sorted(p for p in root.rglob(filename) if p.is_file())
    return items[0] if items else None


def find_mtimesx_candidates(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("mtimesx*"):
        if not p.is_file():
            continue
        if p.suffix.lower() == ".m" or p.suffix.lower().startswith(".mex") or p.name.lower() == "mtimesx":
            out.append(p)
    return sorted(set(out))


def resolve_default_group_prior(cbig_code_dir: Path, num_rois: int, model: str, beta: int) -> Path:
    model_dir = model
    if model.lower() == "gmshbm":
        beta_dir = f"beta{int(beta)}"
    else:
        beta_dir = ""
    root = (
        cbig_code_dir
        / "stable_projects"
        / "brain_parcellation"
        / "Kong2022_ArealMSHBM"
        / "lib"
        / "group_priors"
        / "HCP_fs_LR_32k_40sub"
        / str(int(num_rois))
        / model_dir
    )
    if beta_dir:
        root = root / beta_dir
    return root / "Params_Final.mat"


def preflight(
    *,
    matlab_cmd: str,
    cbig_code_dir: Path,
    group_prior: Path,
    kong2022_dlabel: Path,
    required_inputs: list[Path],
) -> dict[str, object]:
    missing: list[dict[str, str]] = []

    matlab_exe = find_matlab_executable(matlab_cmd)
    if matlab_exe is None:
        missing.append(
            {
                "item": "matlab",
                "reason": f"Executable not found for --matlab-cmd={matlab_cmd!r}",
                "hint": "Load MATLAB first (e.g., module load matlab/R2021a) or pass --matlab-cmd /abs/path/to/matlab.",
            }
        )

    if not cbig_code_dir.exists():
        missing.append(
            {
                "item": "CBIG_CODE_DIR",
                "reason": f"Path not found: {cbig_code_dir}",
                "hint": "Set --cbig-code-dir to a full CBIG checkout containing Kong2022_ArealMSHBM.",
            }
        )

    for p in required_inputs:
        if not p.exists():
            missing.append(
                {
                    "item": "input",
                    "reason": f"Missing required input: {p}",
                    "hint": "Provide existing dtseries/surface/sulc input files.",
                }
            )

    if not group_prior.exists():
        missing.append(
            {
                "item": "group_prior",
                "reason": f"Missing group prior file: {group_prior}",
                "hint": "Use Kong2022_ArealMSHBM group prior Params_Final.mat for your model/ROI settings.",
            }
        )

    if not kong2022_dlabel.exists():
        missing.append(
            {
                "item": "kong2022_dlabel",
                "reason": f"Missing dlabel file: {kong2022_dlabel}",
                "hint": "Provide Schaefer2018_400Parcels_Kong2022_17Networks_order.dlabel.nii",
            }
        )

    req = {
        "Kong2022_ArealMSHBM_root": cbig_code_dir
        / "stable_projects"
        / "brain_parcellation"
        / "Kong2022_ArealMSHBM",
        "CBIG_ArealMSHBM_parcellation_single_subject.m": cbig_code_dir
        / "stable_projects"
        / "brain_parcellation"
        / "Kong2022_ArealMSHBM"
        / "CBIG_ArealMSHBM_parcellation_single_subject.m",
        "CBIG_ArealMSHBM_generate_profiles.m": cbig_code_dir
        / "stable_projects"
        / "brain_parcellation"
        / "Kong2022_ArealMSHBM"
        / "step1_generate_profiles_and_ini_params"
        / "CBIG_ArealMSHBM_generate_profiles.m",
        "CBIG_ArealMSHBM_gMSHBM_generate_individual_parcellation.m": cbig_code_dir
        / "stable_projects"
        / "brain_parcellation"
        / "Kong2022_ArealMSHBM"
        / "step3_generate_ind_parcellations"
        / "CBIG_ArealMSHBM_gMSHBM_generate_individual_parcellation.m",
    }
    found_paths: dict[str, str] = {}
    for name, path in req.items():
        if path.exists():
            found_paths[name] = str(path.resolve())
        else:
            missing.append(
                {
                    "item": name,
                    "reason": f"Missing path: {path}",
                    "hint": "Your local CBIG checkout appears incomplete for Kong2022_ArealMSHBM.",
                }
            )

    corr_profile = find_file_in_tree(cbig_code_dir, "CBIG_ComputeCorrelationProfile.m")
    if corr_profile is None:
        missing.append(
            {
                "item": "CBIG_ComputeCorrelationProfile.m",
                "reason": "Not found under CBIG_CODE_DIR.",
                "hint": "Install full CBIG utilities/matlab/surf dependencies.",
            }
        )
    else:
        found_paths["CBIG_ComputeCorrelationProfile.m"] = str(corr_profile.resolve())

    mtimesx = find_mtimesx_candidates(cbig_code_dir)
    if not mtimesx:
        missing.append(
            {
                "item": "mtimesx",
                "reason": "No mtimesx.m / mtimesx.mex* found under CBIG_CODE_DIR.",
                "hint": "Build/install mtimesx in CBIG dependency tree.",
            }
        )

    return {
        "ok": len(missing) == 0,
        "missing": missing,
        "matlab_executable": matlab_exe,
        "found_paths": found_paths,
        "mtimesx_candidates": [str(p.resolve()) for p in mtimesx],
    }


def parse_run_segments(run_lengths: list[int], expected_total_tp: int) -> list[dict[str, int]]:
    if len(run_lengths) != 4:
        raise ValueError(f"Expected exactly 4 run lengths, got {run_lengths}")
    segments: list[dict[str, int]] = []
    offset = 0
    for idx, ln in enumerate(run_lengths, start=1):
        if int(ln) <= 0:
            raise ValueError(f"Invalid run length run-{idx}: {ln}")
        start = int(offset)
        stop = int(offset + int(ln))
        segments.append({"run_id": idx, "run_start": start, "run_stop": stop, "run_length": int(ln)})
        offset = stop
    if offset != int(expected_total_tp):
        raise RuntimeError(
            f"Run lengths do not match concat length: sum={offset} concat={expected_total_tp}. "
            "Pass explicit --run-lengths."
        )
    return segments


def split_concat_dtseries(dtseries: Path, run_segments: list[dict[str, int]], out_dir: Path, subject: str) -> list[Path]:
    img = nib.load(str(dtseries))
    if len(img.shape) != 2:
        raise ValueError(f"Expected dtseries shape (time, grayordinates), got {img.shape}")
    series_axis = img.header.get_axis(0)
    brain_axis = img.header.get_axis(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for seg in run_segments:
        data = np.asarray(img.dataobj[seg["run_start"] : seg["run_stop"], :], dtype=np.float32)
        run_img = nib.Cifti2Image(
            data,
            header=nib.Cifti2Header.from_axes((series_axis[seg["run_start"] : seg["run_stop"]], brain_axis)),
            nifti_header=img.nifti_header,
        )
        p = out_dir / f"sub-{subject}_task-rest_run-{seg['run_id']}.dtseries.nii"
        nib.save(run_img, str(p))
        out.append(p)
    return out


def matlab_quote(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def run_matlab_wrapper(
    *,
    matlab_cmd: str,
    matlab_executable: str,
    cbig_code_dir: Path,
    project_dir: Path,
    run_list_file: Path,
    censor_list_file: Path,
    group_prior: Path,
    model: str,
    num_rois: int,
    w: int,
    c: int,
    beta: int,
) -> str:
    wrapper_dir = (REPO_ROOT / "scripts" / "cortex" / "matlab").resolve()
    wrapper_file = wrapper_dir / "run_kong2022_mshbm_subject.m"
    if not wrapper_file.exists():
        raise FileNotFoundError(f"Missing MATLAB wrapper: {wrapper_file}")

    batch_expr = (
        f"addpath({matlab_quote(str(wrapper_dir))});"
        f"run_kong2022_mshbm_subject({matlab_quote(str(project_dir))},"
        f"{matlab_quote(str(cbig_code_dir))},{matlab_quote(str(run_list_file))},"
        f"{matlab_quote(str(group_prior))},{matlab_quote(model)},{matlab_quote(str(num_rois))},"
        f"{matlab_quote(str(w))},{matlab_quote(str(c))},{matlab_quote(str(beta))},"
        f"{matlab_quote(str(censor_list_file))});"
    )

    env = dict(os.environ)
    env["CBIG_CODE_DIR"] = str(cbig_code_dir)
    wb_dir = (REPO_ROOT / "software" / "workbench" / "bin_linux64").resolve()
    path_parts = [str(Path(matlab_executable).parent)]
    if wb_dir.exists():
        path_parts.append(str(wb_dir))
    path_parts.append(env.get("PATH", ""))
    env["PATH"] = os.pathsep.join([p for p in path_parts if p])
    proc = run([matlab_cmd, "-batch", batch_expr], env=env)
    return (proc.stdout or "") + "\n" + (proc.stderr or "")


def normalize_rgba(rgba_value: object) -> tuple[int, int, int, int]:
    arr = np.asarray(rgba_value, dtype=float).reshape(-1)
    if arr.shape[0] < 4:
        raise ValueError(f"Invalid RGBA value: {rgba_value!r}")
    arr = arr[:4]
    if float(np.nanmax(arr)) <= 1.0:
        arr = arr * 255.0
    arr = np.clip(np.round(arr), 0, 255).astype(np.int32)
    return int(arr[0]), int(arr[1]), int(arr[2]), int(arr[3])


def parse_label_axis_value(value: object) -> tuple[str, tuple[int, int, int, int]]:
    if isinstance(value, tuple) and len(value) >= 2:
        return str(value[0]), normalize_rgba(value[1])
    return str(value), (255, 255, 255, 255)


def load_kong2022_label_registry(path: Path) -> dict[int, dict[str, object]]:
    img = nib.load(str(path))
    axis = img.header.get_axis(0)
    out: dict[int, dict[str, object]] = {}
    for key, value in axis.label[0].items():
        idx = int(key)
        if idx == 0:
            continue
        name, rgba = parse_label_axis_value(value)
        out[idx] = {"name": name, "rgba": rgba}
    if len(out) != N_ROIS:
        raise ValueError(f"Expected {N_ROIS} labels in Kong2022 dlabel, got {len(out)}")
    return out


def write_label_list_from_registry(reg: dict[int, dict[str, object]], out_path: Path) -> None:
    lines: list[str] = []
    for idx in range(1, N_ROIS + 1):
        row = reg[idx]
        rgba = row["rgba"]
        lines.append(str(row["name"]))
        lines.append(f"{idx} {rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_surface_vertex_count(path: Path) -> int:
    img = nib.load(str(path))
    for darray in img.darrays:
        arr = np.asarray(darray.data)
        if arr.ndim == 2 and arr.shape[1] == 3:
            return int(arr.shape[0])
    raise RuntimeError(f"Could not find pointset in surface: {path}")


def write_metric(values: np.ndarray, out_path: Path, map_name: str) -> None:
    arr = nib.gifti.GiftiDataArray(
        data=np.asarray(values, dtype=np.float32),
        intent=nib.nifti1.intent_codes["NIFTI_INTENT_SHAPE"],
        meta=nib.gifti.GiftiMetaData.from_dict({"Name": map_name}),
    )
    nib.save(nib.gifti.GiftiImage(darrays=[arr]), str(out_path))


def load_label_vector(path: Path) -> np.ndarray:
    data = np.loadtxt(path, dtype=np.int32)
    data = np.asarray(data, dtype=np.int32).reshape(-1)
    if data.size == 0:
        raise RuntimeError(f"Empty label file: {path}")
    return data


def validate_label_ids(lh: np.ndarray, rh: np.ndarray) -> dict[str, object]:
    all_ids = np.concatenate([lh.reshape(-1), rh.reshape(-1)]).astype(np.int32)
    unique, counts = np.unique(all_ids, return_counts=True)
    invalid = sorted(int(v) for v in unique if int(v) < 0 or int(v) > N_ROIS)
    if invalid:
        raise RuntimeError(f"Found invalid labels outside 0..{N_ROIS}: {invalid[:20]}")

    positive = [int(v) for v in unique.tolist() if int(v) > 0]
    missing = [v for v in range(1, N_ROIS + 1) if v not in set(positive)]
    unique_list = unique.tolist()
    counts_list = counts.tolist()
    if len(unique_list) != len(counts_list):
        raise RuntimeError("Label histogram length mismatch between unique ids and counts")

    return {
        "combined_unique_count": int(len(unique)),
        "positive_unique_count": int(len(positive)),
        "missing_positive_labels": missing,
        "left_unique_count": int(np.unique(lh).size),
        "right_unique_count": int(np.unique(rh).size),
        "counts": {str(int(k)): int(v) for k, v in zip(unique_list, counts_list)},
    }


def create_subject_labels_and_dlabel(
    *,
    subject: str,
    atlas_dir: Path,
    left_midthickness: Path,
    right_midthickness: Path,
    registry: dict[int, dict[str, object]],
    lh_labels: np.ndarray,
    rh_labels: np.ndarray,
) -> dict[str, str]:
    labels_dir = atlas_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    if lh_labels.shape[0] != read_surface_vertex_count(left_midthickness):
        raise RuntimeError("lh_labels length does not match left midthickness vertex count")
    if rh_labels.shape[0] != read_surface_vertex_count(right_midthickness):
        raise RuntimeError("rh_labels length does not match right midthickness vertex count")

    label_table = labels_dir / "kong2022_label_table.txt"
    write_label_list_from_registry(registry, label_table)

    left_metric = labels_dir / f"sub-{subject}_atlas-{ATLAS_SLUG}.L.func.gii"
    right_metric = labels_dir / f"sub-{subject}_atlas-{ATLAS_SLUG}.R.func.gii"
    left_label = labels_dir / f"sub-{subject}_atlas-{ATLAS_SLUG}.L.label.gii"
    right_label = labels_dir / f"sub-{subject}_atlas-{ATLAS_SLUG}.R.label.gii"
    out_dlabel = atlas_dir / f"sub-{subject}_atlas-{ATLAS_SLUG}.dlabel.nii"

    write_metric(lh_labels, left_metric, f"sub-{subject}_hemi-L_{ATLAS_SLUG}")
    write_metric(rh_labels, right_metric, f"sub-{subject}_hemi-R_{ATLAS_SLUG}")

    run([WB_COMMAND, "-metric-label-import", str(left_metric), str(label_table), str(left_label), "-discard-others"])
    run([WB_COMMAND, "-metric-label-import", str(right_metric), str(label_table), str(right_label), "-discard-others"])
    run([WB_COMMAND, "-cifti-create-label", str(out_dlabel), "-left-label", str(left_label), "-right-label", str(right_label)])

    return {
        "left_metric": str(left_metric.resolve()),
        "right_metric": str(right_metric.resolve()),
        "left_label": str(left_label.resolve()),
        "right_label": str(right_label.resolve()),
        "dlabel": str(out_dlabel.resolve()),
        "label_table_txt": str(label_table.resolve()),
    }


def get_tool_version(cmd: list[str]) -> str | None:
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True)
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    lines = [x.strip() for x in (proc.stdout or "").splitlines() if x.strip()]
    if not lines:
        lines = [x.strip() for x in (proc.stderr or "").splitlines() if x.strip()]
    return lines[-1] if lines else None


def main() -> int:
    args = parse_args()

    subject = str(args.subject).replace("sub-", "")
    dtseries = Path(args.dtseries).resolve()
    left_inflated = Path(args.left_inflated).resolve()
    right_inflated = Path(args.right_inflated).resolve()
    left_midthickness = Path(args.left_midthickness).resolve()
    right_midthickness = Path(args.right_midthickness).resolve()
    sulc_dscalar = Path(args.sulc_dscalar).resolve()
    cbig_code_dir = Path(args.cbig_code_dir).resolve()
    kong2022_dlabel = Path(args.kong2022_dlabel).resolve()
    out_root = Path(args.out_root).resolve()

    group_prior = (
        Path(args.group_prior).resolve()
        if args.group_prior
        else resolve_default_group_prior(cbig_code_dir, int(args.num_rois), str(args.model), int(args.beta)).resolve()
    )

    required_inputs = [dtseries, left_inflated, right_inflated, left_midthickness, right_midthickness, sulc_dscalar]
    pre = preflight(
        matlab_cmd=str(args.matlab_cmd),
        cbig_code_dir=cbig_code_dir,
        group_prior=group_prior,
        kong2022_dlabel=kong2022_dlabel,
        required_inputs=required_inputs,
    )
    if not bool(pre["ok"]):
        print(
            json.dumps(
                {
                    "status": "preflight_failed",
                    "subject": subject,
                    "atlas_slug": ATLAS_SLUG,
                    "missing": pre["missing"],
                    "matlab_executable": pre["matlab_executable"],
                    "cbig_code_dir": str(cbig_code_dir),
                    "group_prior": str(group_prior),
                },
                indent=2,
            )
        )
        return 2

    run_lengths = [int(v) for v in args.run_lengths]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run_ok",
                    "subject": subject,
                    "atlas_slug": ATLAS_SLUG,
                    "model": str(args.model),
                    "num_rois": int(args.num_rois),
                    "w": int(args.w),
                    "c": int(args.c),
                    "beta": int(args.beta),
                    "run_lengths": run_lengths,
                    "group_prior": str(group_prior),
                    "out_root": str(out_root),
                },
                indent=2,
            )
        )
        return 0

    dt_img = nib.load(str(dtseries))
    if len(dt_img.shape) != 2:
        raise RuntimeError(f"Expected dtseries shape (time, grayordinates), got {dt_img.shape}")
    run_segments = parse_run_segments(run_lengths, int(dt_img.shape[0]))

    subject_root = out_root / f"sub-{subject}"
    atlas_dir = subject_root / ATLAS_SLUG if str(args.subject_dir_layout) == "nested" else subject_root
    atlas_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = atlas_dir / "runs"
    project_dir = atlas_dir / "project_dir"
    project_dir.mkdir(parents=True, exist_ok=True)

    run_paths = split_concat_dtseries(dtseries, run_segments, runs_dir, subject)
    run_list_file = project_dir / "input_run_dtseries.txt"
    run_list_file.write_text("\n".join(str(p.resolve()) for p in run_paths) + "\n", encoding="utf-8")
    # Keep censor inputs outside project_dir because Kong2022 code may reset
    # project_dir when overwrite is enabled.
    censor_dir = atlas_dir / "censor_inputs"
    censor_dir.mkdir(parents=True, exist_ok=True)
    censor_paths: list[Path] = []
    for seg in run_segments:
        censor_path = censor_dir / f"sub-{subject}_task-rest_run-{int(seg['run_id'])}_censor.txt"
        np.savetxt(censor_path, np.ones(int(seg["run_length"]), dtype=np.int32), fmt="%d")
        censor_paths.append(censor_path)
    censor_list_file = project_dir / "input_run_censor.txt"
    censor_list_file.write_text("\n".join(str(p.resolve()) for p in censor_paths) + "\n", encoding="utf-8")

    matlab_log = run_matlab_wrapper(
        matlab_cmd=str(args.matlab_cmd),
        matlab_executable=str(pre["matlab_executable"]),
        cbig_code_dir=cbig_code_dir,
        project_dir=project_dir,
        run_list_file=run_list_file,
        censor_list_file=censor_list_file,
        group_prior=group_prior,
        model=str(args.model),
        num_rois=int(args.num_rois),
        w=int(args.w),
        c=int(args.c),
        beta=int(args.beta),
    )
    (atlas_dir / "matlab_wrapper.log").write_text(matlab_log, encoding="utf-8")

    route_out_dir = project_dir / "route_a_outputs"
    lh_txt = route_out_dir / "lh_labels.txt"
    rh_txt = route_out_dir / "rh_labels.txt"
    if not lh_txt.exists() or not rh_txt.exists():
        raise RuntimeError(f"Wrapper output missing lh/rh labels: {lh_txt} {rh_txt}")

    lh_labels = load_label_vector(lh_txt)
    rh_labels = load_label_vector(rh_txt)
    label_stats = validate_label_ids(lh_labels, rh_labels)
    registry = load_kong2022_label_registry(kong2022_dlabel)

    outputs = create_subject_labels_and_dlabel(
        subject=subject,
        atlas_dir=atlas_dir,
        left_midthickness=left_midthickness,
        right_midthickness=right_midthickness,
        registry=registry,
        lh_labels=lh_labels,
        rh_labels=rh_labels,
    )

    summary = {
        "subject": subject,
        "atlas_slug": ATLAS_SLUG,
        "status": "done",
        "timestamp_utc": now_utc_iso(),
        "params": {
            "model": str(args.model),
            "num_rois": int(args.num_rois),
            "w": int(args.w),
            "c": int(args.c),
            "beta": int(args.beta),
            "num_sessions": int(len(run_paths)),
            "run_lengths": [int(seg["run_length"]) for seg in run_segments],
        },
        "inputs": {
            "dtseries": str(dtseries),
            "left_inflated": str(left_inflated),
            "right_inflated": str(right_inflated),
            "left_midthickness": str(left_midthickness),
            "right_midthickness": str(right_midthickness),
            "sulc_dscalar": str(sulc_dscalar),
            "group_prior": str(group_prior),
            "kong2022_dlabel": str(kong2022_dlabel),
        },
        "outputs": outputs,
        "label_stats": label_stats,
        "dependencies": {
            "matlab_cmd": str(args.matlab_cmd),
            "matlab_executable": str(pre["matlab_executable"]),
            "wb_command": WB_COMMAND,
            "cbig_code_dir": str(cbig_code_dir),
            "python": sys.executable,
            "numpy": np.__version__,
            "nibabel": nib.__version__,
            "wb_command_version": get_tool_version([WB_COMMAND, "-version"]),
            "matlab_version": get_tool_version([str(args.matlab_cmd), "-batch", "disp(version);"]),
        },
    }

    summary_path = atlas_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    stage_manifest = {
        "stage": "kong2022_areal_mshbm_individualize",
        "status": "done",
        "timestamp_utc": now_utc_iso(),
        "params": summary["params"],
        "inputs": [
            file_meta(dtseries),
            file_meta(left_inflated),
            file_meta(right_inflated),
            file_meta(left_midthickness),
            file_meta(right_midthickness),
            file_meta(sulc_dscalar),
            file_meta(group_prior),
            file_meta(kong2022_dlabel),
        ],
        "outputs": [
            str(Path(outputs["left_label"]).resolve()),
            str(Path(outputs["right_label"]).resolve()),
            str(Path(outputs["dlabel"]).resolve()),
            str(summary_path.resolve()),
            str((atlas_dir / "matlab_wrapper.log").resolve()),
        ]
        + [str(p.resolve()) for p in run_paths],
    }
    (atlas_dir / "stage_manifest.json").write_text(json.dumps(stage_manifest, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "subject": subject,
                "atlas_slug": ATLAS_SLUG,
                "out_dir": str(atlas_dir),
                "dlabel": outputs["dlabel"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

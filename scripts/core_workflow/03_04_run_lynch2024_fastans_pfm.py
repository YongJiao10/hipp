#!/usr/bin/env python3
from __future__ import annotations

"""Run official FASTANS-style Lynch2024 PFM and export hard-label references.

This maintained entrypoint mirrors the official FASTANS ``fast_pfm`` algorithm:
load Lynch priors, compute prior timecourses, z-score prior and vertex time
series, compute network weights, assign each cortex vertex by argmax label,
then run Workbench cluster cleanup and dilation before extracting hard-label
network time series.
"""

import argparse
import csv
import json
import pickle
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np

from lib.io_utils import load_subject_ids
from lib.path_templates import format_subject_path_template
from lib.workflow_utils import capped_jobs, utc_now_iso, write_json


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESOURCES_ROOT = REPO_ROOT / "external" / "FASTANS" / "resources"
DEFAULT_CORTEX_PFM_ROOT = REPO_ROOT / "outputs_hcp" / "work" / "cortex_pfm"
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "hippunfold_input"
DEFAULT_DTSERIES_TEMPLATE = "{data_root}/{sub_subject}/func/{sub_subject}_task-rest_run-concat.dtseries.nii"
DEFAULT_LEFT_SURFACE_TEMPLATE = "{data_root}/{sub_subject}/anat/{sub_subject}_hemi-L_space-fsLR_den-32k_desc-MSMAll_midthickness.surf.gii"
DEFAULT_RIGHT_SURFACE_TEMPLATE = "{data_root}/{sub_subject}/anat/{sub_subject}_hemi-R_space-fsLR_den-32k_desc-MSMAll_midthickness.surf.gii"
DEFAULT_WB_COMMAND_CANDIDATES = [
    REPO_ROOT / "software" / "workbench" / "bin_linux64" / "wb_command",
    Path("/home/yojiao/FASTANS/workbench/bin_linux64/wb_command"),
]


def resolve_wb_command(raw: str | None) -> Path:
    candidates = [Path(raw).expanduser()] if raw else DEFAULT_WB_COMMAND_CANDIDATES
    for candidate in candidates:
        p = candidate.resolve()
        if p.exists() and p.is_file():
            return p
    tried = ", ".join(str(Path(c).expanduser()) for c in candidates)
    raise FileNotFoundError(f"Could not resolve wb_command. Tried: {tried}")


def run_wb(wb_command: Path, args: list[str]) -> None:
    proc = subprocess.run([str(wb_command), *args], text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            "wb_command failed: " + " ".join([str(wb_command), *args]) + "\nSTDOUT:\n" + proc.stdout + "\nSTDERR:\n" + proc.stderr
        )


def load_cifti_values(cifti_path: Path, cortex_template_path: Path) -> np.ndarray:
    template = nib.load(str(cortex_template_path))
    template_values = template.get_fdata()[0]
    img = nib.load(str(cifti_path))
    brainmodel_axis = img.header.get_axis(1)
    values = img.get_fdata()[:, brainmodel_axis.surface_mask]
    if values.shape[0] == 1:
        out = values[0]
        if out.shape[0] == 59412:
            padded = np.zeros(64984, dtype=np.float64)
            padded[template_values == 1] = out
            out = padded
        return out
    if values.shape[1] == 59412:
        padded = np.zeros((values.shape[0], 64984), dtype=np.float64)
        padded[:, template_values == 1] = values
        values = padded
    return values


def write_cifti_dscalar_from_array(*, array: np.ndarray, template_path: Path, output_path: Path, wb_command: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path = output_path.with_suffix(".txt")
    np.savetxt(txt_path, np.asarray(array).reshape(-1), fmt="%.10g")
    run_wb(wb_command, ["-cifti-convert", "-from-text", str(txt_path), str(template_path), str(output_path)])
    return txt_path


def label_import(*, dscalar_path: Path, labels_path: Path, dlabel_path: Path, wb_command: Path) -> None:
    run_wb(wb_command, ["-cifti-label-import", str(dscalar_path), str(labels_path), str(dlabel_path)])


def parse_fastans_label_list(labels_path: Path, n_priors: int) -> list[dict[str, object]]:
    lines = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows: list[dict[str, object]] = []
    for i in range(0, len(lines), 2):
        if i + 1 >= len(lines):
            break
        name = lines[i]
        parts = lines[i + 1].split()
        if len(parts) < 5:
            raise RuntimeError(f"Invalid FASTANS label line in {labels_path}: {lines[i + 1]!r}")
        label_id = int(parts[0])
        if label_id <= n_priors:
            rows.append(
                {
                    "component_id": label_id,
                    "component_name": name,
                    "network": name,
                    "hemisphere": "LR",
                    "label_rgba": " ".join(parts[1:5]),
                }
            )
    if len(rows) != int(n_priors):
        raise RuntimeError(f"Expected {n_priors} label rows from {labels_path}, got {len(rows)}")
    return rows


def finite_zscore(arr: np.ndarray, axis: int) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64)
    mean = x.mean(axis=axis, keepdims=True)
    std = x.std(axis=axis, keepdims=True)
    return (x - mean) / std


def hard_label_timeseries(timeseries: np.ndarray, labels: np.ndarray, label_rows: list[dict[str, object]]) -> tuple[np.ndarray, list[dict[str, object]]]:
    ts = np.asarray(timeseries, dtype=np.float64)
    lab = np.asarray(labels, dtype=np.int32).reshape(-1)
    out = []
    rows = []
    for row in label_rows:
        label_id = int(row["component_id"])
        mask = lab == label_id
        n = int(np.sum(mask))
        if n <= 0:
            series = np.full((ts.shape[0],), np.nan, dtype=np.float32)
        else:
            series = np.mean(ts[:, mask], axis=1).astype(np.float32)
        out.append(series)
        rows.append({**row, "n_grayordinates_used": n})
    return np.asarray(out, dtype=np.float32), rows


def write_component_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["component_id", "component_name", "network", "hemisphere", "n_grayordinates_used", "label_rgba"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def run_fastans_lynch2024_subject(
    *,
    subject: str,
    data_root: Path,
    output_root: Path,
    resources_root: Path,
    dtseries_template: str,
    left_surface_template: str,
    right_surface_template: str,
    wb_command: Path,
) -> dict[str, object]:
    sid = str(subject).replace("sub-", "")
    out_dir = output_root / f"sub-{sid}" / "lynch2024"
    out_dir.mkdir(parents=True, exist_ok=True)
    cortex_template = resources_root / "templates" / "CORTEX.32k_fs_LR.dscalar.nii"
    priors_path = resources_root / "PFM" / "priors" / "Lynch2024" / "Lynch2024_priors.pickle"
    labels_path = resources_root / "PFM" / "priors" / "Lynch2024" / "Lynch2024_LabelList.txt"
    dtseries_path = format_subject_path_template(dtseries_template, subject=sid, data_root=data_root)
    left_surface = format_subject_path_template(left_surface_template, subject=sid, hemi="L", data_root=data_root)
    right_surface = format_subject_path_template(right_surface_template, subject=sid, hemi="R", data_root=data_root)
    for path in (cortex_template, priors_path, labels_path, dtseries_path, left_surface, right_surface):
        if not path.exists():
            raise FileNotFoundError(path)

    with priors_path.open("rb") as f:
        priors = np.asarray(pickle.load(f), dtype=np.float64)
    timeseries = np.asarray(load_cifti_values(dtseries_path, cortex_template), dtype=np.float64)
    if timeseries.ndim != 2 or timeseries.shape[1] != priors.shape[1]:
        raise RuntimeError(f"FASTANS timeseries/prior shape mismatch for sub-{sid}: timeseries={timeseries.shape} priors={priors.shape}")

    timeseries_standardized = finite_zscore(timeseries, axis=0)
    prior_timeseries = priors @ timeseries.T
    prior_timeseries_standardized = finite_zscore(prior_timeseries, axis=1)
    network_weights = prior_timeseries_standardized @ timeseries_standardized / (len(timeseries_standardized) - 1)
    raw_labels = (np.argmax(network_weights, axis=0) + 1).astype(np.float64)

    raw_dscalar = out_dir / "PFM_Lynch2024priors_raw_argmax.dscalar.nii"
    raw_txt = write_cifti_dscalar_from_array(array=raw_labels, template_path=cortex_template, output_path=raw_dscalar, wb_command=wb_command)
    raw_dlabel = out_dir / "PFM_Lynch2024priors_raw_argmax.dlabel.nii"
    label_import(dscalar_path=raw_dscalar, labels_path=labels_path, dlabel_path=raw_dlabel, wb_command=wb_command)

    pfm_data = load_cifti_values(raw_dlabel, cortex_template)
    cleaned = np.zeros((pfm_data.shape[0],), dtype=np.float64)
    for label_value in np.unique(pfm_data):
        if int(label_value) <= 0:
            continue
        network_arr = np.where(pfm_data == label_value, pfm_data, 0.0)
        network_dscalar = out_dir / f"PFM_Lynch2024priors_network{int(label_value)}.dscalar.nii"
        write_cifti_dscalar_from_array(array=network_arr, template_path=cortex_template, output_path=network_dscalar, wb_command=wb_command)
        run_wb(
            wb_command,
            [
                "-cifti-find-clusters",
                str(network_dscalar),
                "0",
                "50",
                "0",
                "50",
                "COLUMN",
                str(network_dscalar),
                "-left-surface",
                str(left_surface),
                "-right-surface",
                str(right_surface),
                "-merged-volume",
            ],
        )
        network_data = load_cifti_values(network_dscalar, cortex_template)
        network_data[network_data != 0] = label_value
        cleaned += network_data

    cleaned_dscalar = out_dir / "PFM_Lynch2024priors_cleaned.dscalar.nii"
    write_cifti_dscalar_from_array(array=cleaned, template_path=cortex_template, output_path=cleaned_dscalar, wb_command=wb_command)
    cleaned_dlabel = out_dir / "PFM_Lynch2024priors_cleaned.dlabel.nii"
    label_import(dscalar_path=cleaned_dscalar, labels_path=labels_path, dlabel_path=cleaned_dlabel, wb_command=wb_command)

    final_dlabel = out_dir / "PFM_Lynch2024priors.dlabel.nii"
    run_wb(
        wb_command,
        [
            "-cifti-dilate",
            str(cleaned_dlabel),
            "COLUMN",
            "50",
            "50",
            "-left-surface",
            str(left_surface),
            "-right-surface",
            str(right_surface),
            str(final_dlabel),
            "-nearest",
        ],
    )
    template_mask = load_cifti_values(cortex_template, cortex_template)
    final_values = load_cifti_values(final_dlabel, cortex_template)
    final_values[template_mask == 0] = 0
    final_dscalar = out_dir / "PFM_Lynch2024priors.dscalar.nii"
    final_txt = write_cifti_dscalar_from_array(array=final_values, template_path=cortex_template, output_path=final_dscalar, wb_command=wb_command)
    label_import(dscalar_path=final_dscalar, labels_path=labels_path, dlabel_path=final_dlabel, wb_command=wb_command)
    run_wb(
        wb_command,
        [
            "-cifti-label-to-border",
            str(final_dlabel),
            "-border",
            str(left_surface),
            str(out_dir / "PFM_Lynch2024priors.L.border"),
            "-border",
            str(right_surface),
            str(out_dir / "PFM_Lynch2024priors.R.border"),
        ],
    )

    final_labels = load_cifti_values(final_dlabel, cortex_template).astype(np.int32)
    label_rows = parse_fastans_label_list(labels_path, int(priors.shape[0]))
    component_ts, component_rows = hard_label_timeseries(timeseries, final_labels, label_rows)
    np.save(out_dir / "cortex_roi_component_timeseries.npy", component_ts.astype(np.float32))
    write_component_tsv(out_dir / "cortex_roi_components.tsv", component_rows)
    np.save(out_dir / "cortex_roi_labels_64984.npy", final_labels.astype(np.int16))

    label_counts = {str(int(k)): int(v) for k, v in zip(*np.unique(final_labels, return_counts=True), strict=False)}
    manifest = {
        "stage": "lynch2024_fastans_official_pfm",
        "status": "done",
        "timestamp_utc": utc_now_iso(),
        "subject": f"sub-{sid}",
        "algorithm_source": "external/FASTANS/code/FASTANS.py fast_pfm mirrored in maintained core workflow",
        "parcellation": "Lynch2024",
        "inputs": {
            "dtseries": str(dtseries_path),
            "left_surface": str(left_surface),
            "right_surface": str(right_surface),
            "priors": str(priors_path),
            "labels": str(labels_path),
            "cortex_template": str(cortex_template),
            "wb_command": str(wb_command),
        },
        "outputs": {
            "final_dlabel": str(final_dlabel),
            "final_dscalar": str(final_dscalar),
            "component_timeseries": str(out_dir / "cortex_roi_component_timeseries.npy"),
            "component_tsv": str(out_dir / "cortex_roi_components.tsv"),
            "labels_64984": str(out_dir / "cortex_roi_labels_64984.npy"),
            "raw_argmax_dlabel": str(raw_dlabel),
            "cleaned_dlabel": str(cleaned_dlabel),
            "raw_argmax_text": str(raw_txt),
            "final_text": str(final_txt),
        },
        "shape": {
            "priors": [int(x) for x in priors.shape],
            "timeseries": [int(x) for x in timeseries.shape],
            "component_timeseries": [int(x) for x in component_ts.shape],
        },
        "label_counts_64984": label_counts,
        "cleanup": {
            "cluster_threshold_command": "wb_command -cifti-find-clusters <network> 0 50 0 50 COLUMN <network> -left-surface <L> -right-surface <R> -merged-volume",
            "dilation_command": "wb_command -cifti-dilate <cleaned.dlabel.nii> COLUMN 50 50 -left-surface <L> -right-surface <R> <final.dlabel.nii> -nearest",
            "intermediate_files_retained": True,
        },
    }
    write_json(out_dir / "fastans_pfm_manifest.json", manifest, sort_keys=True, trailing_newline=True)
    return {"subject": f"sub-{sid}", "output": str(out_dir), "component_shape": [int(x) for x in component_ts.shape]}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run official FASTANS-style Lynch2024 PFM hard-label generation.")
    parser.add_argument("--subjects-manifest", required=True)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--cortex-pfm-root", default=str(DEFAULT_CORTEX_PFM_ROOT))
    parser.add_argument("--fastans-resources-root", default=str(DEFAULT_RESOURCES_ROOT))
    parser.add_argument("--dtseries-template", default=DEFAULT_DTSERIES_TEMPLATE)
    parser.add_argument("--left-surface-template", default=DEFAULT_LEFT_SURFACE_TEMPLATE)
    parser.add_argument("--right-surface-template", default=DEFAULT_RIGHT_SURFACE_TEMPLATE)
    parser.add_argument("--wb-command", default=None)
    parser.add_argument("--jobs", type=int, default=1, help="Subject-level concurrency; hard-capped at 10.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    subjects = [str(s).replace("sub-", "") for s in args.subjects] if args.subjects else load_subject_ids(Path(args.subjects_manifest).resolve())
    wb_command = resolve_wb_command(args.wb_command)
    jobs = capped_jobs(args.jobs, len(subjects))
    kwargs = {
        "data_root": Path(args.data_root).resolve(),
        "output_root": Path(args.cortex_pfm_root).resolve(),
        "resources_root": Path(args.fastans_resources_root).resolve(),
        "dtseries_template": str(args.dtseries_template),
        "left_surface_template": str(args.left_surface_template),
        "right_surface_template": str(args.right_surface_template),
        "wb_command": wb_command,
    }
    results = []
    if jobs == 1:
        for subject in subjects:
            result = run_fastans_lynch2024_subject(subject=subject, **kwargs)
            results.append(result)
            print(f"[done] {result['subject']}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(run_fastans_lynch2024_subject, subject=s, **kwargs) for s in subjects]
            for fut in as_completed(futures):
                result = fut.result()
                results.append(result)
                print(f"[done] {result['subject']}", flush=True)
    print(json.dumps({"status": "ok", "subjects": len(results), "jobs": jobs, "wb_command": str(wb_command)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

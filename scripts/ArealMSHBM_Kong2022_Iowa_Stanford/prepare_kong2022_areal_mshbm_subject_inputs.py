#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import nibabel as nib


DATASET_DTSERIES_GLOBS = {
    "jj_tms": ["*_task-rest_run-0*_space-fsLR_den-91k_bold.dtseries.nii"],
    "stanford": ["*_task-rest_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii"],
}


def normalize_dataset(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def natural_key(path: Path) -> tuple:
    parts = re.split(r"(\d+)", path.name)
    return tuple(int(part) if part.isdigit() else part for part in parts)


def dtseries_n_timepoints(path: Path) -> int:
    img = nib.load(str(path))
    if len(img.shape) != 2:
        raise ValueError(f"Expected 2D CIFTI dtseries, got shape={img.shape}: {path}")
    return int(img.shape[0])


def write_ones(path: Path, count: int) -> None:
    with path.open("w", encoding="utf-8") as f:
        for _ in range(count):
            f.write("1\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare task-rest dtseries inputs for one Kong2022 ArealMSHBM SGE task."
    )
    parser.add_argument(
        "--dataset",
        default="jj_tms",
        help="Dataset preset: jj_tms or stanford. Default: jj_tms.",
    )
    parser.add_argument("--subject", required=True, help="Subject ID with or without sub- prefix.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--subject-output-dir", required=True, type=Path)
    parser.add_argument(
        "--dtseries-glob",
        action="append",
        default=None,
        help="Override func/ dtseries glob. Repeat to allow multiple patterns.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset = normalize_dataset(args.dataset)
    if dataset not in DATASET_DTSERIES_GLOBS:
        known = ", ".join(sorted(DATASET_DTSERIES_GLOBS))
        raise SystemExit(f"Unsupported dataset: {args.dataset}. Expected one of: {known}")
    dtseries_globs = args.dtseries_glob or DATASET_DTSERIES_GLOBS[dataset]

    subject = args.subject if args.subject.startswith("sub-") else f"sub-{args.subject}"
    subject_dir = args.data_root / subject
    func_dir = subject_dir / "func"
    if not func_dir.is_dir():
        raise SystemExit(f"Missing func directory: {func_dir}")

    dtseries_by_path = {}
    for pattern in dtseries_globs:
        for path in func_dir.glob(pattern):
            dtseries_by_path[path] = path
    dtseries = sorted(dtseries_by_path, key=natural_key)
    if not dtseries:
        patterns = ", ".join(dtseries_globs)
        raise SystemExit(f"No task-rest fsLR den-91k dtseries found in {func_dir}; patterns={patterns}")

    inputs_dir = args.subject_output_dir / "inputs"
    censor_dir = inputs_dir / "censors"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    censor_dir.mkdir(parents=True, exist_ok=True)

    run_list_path = inputs_dir / "task_rest_dtseries.txt"
    censor_list_path = inputs_dir / "task_rest_censors.txt"
    rows = []
    censor_paths = []
    for idx, dt_path in enumerate(dtseries, start=1):
        n_tp = dtseries_n_timepoints(dt_path)
        censor_path = censor_dir / f"{subject}_task-rest_run-{idx:02d}_censor.txt"
        if args.overwrite or not censor_path.exists():
            write_ones(censor_path, n_tp)
        rows.append({"run_index": idx, "dtseries": str(dt_path), "timepoints": n_tp, "censor": str(censor_path)})
        censor_paths.append(censor_path)

    run_list_path.write_text("\n".join(str(p) for p in dtseries) + "\n", encoding="utf-8")
    censor_list_path.write_text("\n".join(str(p) for p in censor_paths) + "\n", encoding="utf-8")

    manifest = {
        "dataset": dataset,
        "subject": subject,
        "data_root": str(args.data_root),
        "dtseries_globs": dtseries_globs,
        "subject_dir": str(subject_dir),
        "run_count": len(rows),
        "runs": rows,
        "run_list": str(run_list_path),
        "censor_list": str(censor_list_path),
    }
    manifest_path = inputs_dir / "input_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/opt/miniconda3/envs/py314/bin/python
"""Copy only the native con files used by the current exploratory cohorts.

This is not a numbered formal-workflow entry point.

The mounted directories are slow to enumerate recursively.  This script reads
the repository's existing Standard-space cohort flow and constructs each
expected native filename directly, avoiding a scan of unrelated site folders
and excluded subjects.  Copies are staged beside the destination and then
atomically renamed into place; no source files are changed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]

STANFORD_SOURCE_ROOT = Path(
    "/Volumes/lss_jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/"
    "tmsfmri_fmriprep/group_con_results_NativeSpace/group_con_data_6P"
)
STANFORD_FLOW = (
    REPO_ROOT
    / "outputs_stanford/archive/20260818_standard_space_beta_replaced_by_native/results/"
    / "stanford_surface_beta_20260816_argon_final/cohort_flow.csv"
)
STANFORD_DEST_ROOT = (
    REPO_ROOT
    / "outputs_stanford/inputs/stanford/tmsfmri_fmriprep/"
    / "group_con_results_NativeSpace/group_con_data_6P"
)

JJ_SOURCE_ROOT = Path(
    "/Volumes/lss_jianglab/3_Data_Working/fmriprep_processed_JJ_TMS/"
    "spm_glm_output/summary_NativeSpace_Float32/baseline_spTMS"
)
JJ_FLOW = (
    REPO_ROOT
    / "outputs_iowa_jj_tms/archive/20260818_standard_space_beta_replaced_by_native/results/"
    / "sptms_surface_grid2_20260811/cohort_flow.csv"
)
JJ_DEST_ROOT = (
    REPO_ROOT
    / "outputs_iowa_jj_tms/inputs/iowa_jj_tms/spm_glm_output/"
    / "summary_NativeSpace_Float32/baseline_spTMS"
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def selected_rows(dataset: str) -> list[dict[str, str]]:
    flow = STANFORD_FLOW if dataset == "stanford" else JJ_FLOW
    rows = [row for row in read_rows(flow) if row.get("include", "").strip().lower() == "true"]
    if not rows:
        raise ValueError(f"no included rows in {flow}")
    return rows


def job_paths(dataset: str, row: dict[str, str]) -> tuple[Path, Path]:
    subject = row["subject"].strip()
    if dataset == "stanford":
        site_dir = row["site_dir"].strip()
        cohort = row["cohort"].strip()
        filename = f"NativeSpace_sub-{subject}_tms_{site_dir}_con_0001.nii.gz"
        relative = Path(site_dir) / cohort / filename
        return STANFORD_SOURCE_ROOT / relative, STANFORD_DEST_ROOT / relative
    filename = f"NativeSpace_sub-{subject}_baseline_spTMS_con_0001.nii.gz"
    return JJ_SOURCE_ROOT / filename, JJ_DEST_ROOT / filename


def copy_one(dataset: str, row: dict[str, str]) -> dict[str, Any]:
    source, destination = job_paths(dataset, row)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == source.stat().st_size:
        return {
            "dataset": dataset,
            "subject": row["subject"],
            "site": row.get("site", ""),
            "site_dir": row.get("site_dir", ""),
            "cohort": row.get("cohort", ""),
            "source": str(source),
            "destination": str(destination),
            "bytes": int(destination.stat().st_size),
            "status": "already_present_size_match",
        }
    temporary = destination.with_name(destination.name + ".partial_copy")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)
    return {
        "dataset": dataset,
        "subject": row["subject"],
        "site": row.get("site", ""),
        "site_dir": row.get("site_dir", ""),
        "cohort": row.get("cohort", ""),
        "source": str(source),
        "destination": str(destination),
        "bytes": int(destination.stat().st_size),
        "status": "copied",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("stanford", "jj_tms"), required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--manifest-root", type=Path, default=REPO_ROOT / "outputs_native_surface_comparison/native_con_copy")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 10:
        raise ValueError("workers must be between 1 and 10")

    rows = selected_rows(args.dataset)
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=min(args.workers, len(rows))) as executor:
        futures = {executor.submit(copy_one, args.dataset, row): row for row in rows}
        for index, future in enumerate(as_completed(futures), start=1):
            row = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001 - preserve per-file failure evidence
                errors.append(
                    {
                        "dataset": args.dataset,
                        "subject": row.get("subject", ""),
                        "site": row.get("site", ""),
                        "error": repr(exc),
                    }
                )
            if index % 10 == 0 or index == len(futures):
                print(f"[{index}/{len(futures)}] {args.dataset} native con copies", flush=True)

    args.manifest_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest_root / f"{args.dataset}_copy_manifest.csv"
    if results:
        fieldnames = list(results[0])
        with manifest_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sorted(results, key=lambda item: (item["subject"], item["site"])))
    summary = {
        "dataset": args.dataset,
        "requested": len(rows),
        "copied": len(results),
        "errors": len(errors),
        "source_root": str(STANFORD_SOURCE_ROOT if args.dataset == "stanford" else JJ_SOURCE_ROOT),
        "destination_root": str(STANFORD_DEST_ROOT if args.dataset == "stanford" else JJ_DEST_ROOT),
        "manifest": str(manifest_path),
        "error_rows": errors,
    }
    (args.manifest_root / f"{args.dataset}_copy_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if errors:
        raise RuntimeError(f"{len(errors)} native con copies failed; see {args.manifest_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

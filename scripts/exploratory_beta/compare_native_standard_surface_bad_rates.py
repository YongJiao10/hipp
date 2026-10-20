#!/opt/miniconda3/envs/py314/bin/python
"""Exploratorily map native con images and compare hippocampal bad-vertex rates.

This is not a numbered formal-workflow entry point.

The archived Standard-space runs are treated as immutable reference outputs.
This entry point maps copied native-space con images directly to the native
HippUnfold hippocampal ribbons and writes a paired bad-vertex comparison.

Native Float32 con products in these datasets encode background as exact zero,
whereas the Standard-space products encode background as NaN.  Consequently,
the run records both a literal finite-value ROI and a nonzero-support ROI.  The
nonzero-support result is the primary spatial-overlap comparison; the literal
finite result is retained to expose the zero-fill effect.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "1")

import nibabel as nib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
WB_COMMAND = Path("/Applications/wb_view.app/Contents/usr/bin/wb_command")
EXPECTED_VERTICES = 512
ROI_POLICIES = ("finite", "nonzero")

STANFORD_STANDARD_RUN = (
    REPO_ROOT
    / "outputs_stanford/archive/20260818_standard_space_beta_replaced_by_native/results/"
    / "stanford_surface_beta_20260816_argon_final"
)
STANFORD_NATIVE_CON_ROOT = (
    REPO_ROOT
    / "outputs_stanford/inputs/stanford/tmsfmri_fmriprep/"
    / "group_con_results_NativeSpace/group_con_data_6P"
)
STANFORD_NATIVE_HIPP_ROOT = (
    REPO_ROOT / "outputs_stanford/inputs/stanford/hippunfold/dense_corobl_batch"
)
STANFORD_NATIVE_SURFACE_SPACE = "T1w"

JJ_STANDARD_RUN = (
    REPO_ROOT
    / "outputs_iowa_jj_tms/archive/20260818_standard_space_beta_replaced_by_native/results/"
    / "sptms_surface_grid2_20260811"
)
JJ_NATIVE_CON_ROOT = (
    REPO_ROOT
    / "outputs_iowa_jj_tms/inputs/iowa_jj_tms/spm_glm_output/"
    / "summary_NativeSpace_Float32/baseline_spTMS"
)
JJ_NATIVE_HIPP_ROOT = (
    REPO_ROOT / "outputs_iowa_jj_tms/inputs/iowa_jj_tms/hippunfold/dense_corobl_batch"
)
JJ_NATIVE_SURFACE_SPACE = "T1w"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = list(rows[0])
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def normalize_subject(value: object) -> str:
    match = re.fullmatch(r"(?:sub-)?(\d+)", str(value).strip(), flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"invalid subject identifier: {value!r}")
    return match.group(1)


def wb_prefix() -> list[str]:
    import shutil
    candidate = Path("/home/yojiao/FASTANS/workbench/bin_linux64/wb_command")
    if candidate.is_file():
        return [str(candidate)]
    which = shutil.which("wb_command")
    if which:
        return [which]
    if WB_COMMAND.is_file():
        return ["arch", "-x86_64", str(WB_COMMAND)]
    raise FileNotFoundError(f"Workbench executable is missing")


def load_gifti_vector(path: Path) -> np.ndarray:
    values = np.asarray(nib.load(str(path)).agg_data(), dtype=np.float64).reshape(-1)
    if values.shape != (EXPECTED_VERTICES,):
        raise ValueError(f"unexpected 512-vertex GIFTI shape: {path} {values.shape}")
    return values


def native_surface_paths(dataset: str, subject: str, hemi: str) -> dict[str, Path]:
    subject_root = (
        STANFORD_NATIVE_HIPP_ROOT if dataset == "stanford" else JJ_NATIVE_HIPP_ROOT
    ) / f"sub-{subject}/hippunfold/sub-{subject}/surf"
    space = STANFORD_NATIVE_SURFACE_SPACE if dataset == "stanford" else JJ_NATIVE_SURFACE_SPACE
    paths = {
        kind: subject_root
        / f"sub-{subject}_hemi-{hemi}_space-{space}_den-512_label-hipp_{kind}.surf.gii"
        for kind in ("inner", "midthickness", "outer")
    }
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing native HippUnfold surface(s) for {dataset} sub-{subject} hemi-{hemi}: {missing}"
        )
    return paths


def native_con_path(dataset: str, row: dict[str, str]) -> Path:
    if dataset == "stanford":
        site_dir = str(row["site_dir"])
        cohort = str(row["cohort"])
        root = STANFORD_NATIVE_CON_ROOT / site_dir / cohort
        matches = sorted(
            path
            for path in root.glob(f"NativeSpace_sub-{normalize_subject(row['subject'])}*_con_0001.nii.gz")
            if not path.name.startswith("._")
        )
    else:
        subject = normalize_subject(row["subject"])
        matches = sorted(
            path
            for path in JJ_NATIVE_CON_ROOT.glob(
                f"NativeSpace_sub-{subject}_baseline_spTMS_con_0001.nii.gz"
            )
            if not path.name.startswith("._")
        )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one native con for {dataset} sub-{row['subject']}; found {len(matches)} under "
            f"{STANFORD_NATIVE_CON_ROOT if dataset == 'stanford' else JJ_NATIVE_CON_ROOT}"
        )
    return matches[0]


def standard_bad_counts(
    dataset: str,
    row: dict[str, str],
    *,
    stanford_qc_rows: list[dict[str, str]] | None = None,
) -> dict[str, int]:
    subject = normalize_subject(row["subject"])
    if dataset == "stanford":
        qc_rows = stanford_qc_rows
        if qc_rows is None:
            qc_rows = read_csv(STANFORD_STANDARD_RUN / "argon_mapping_qc.csv")
        key = (subject, str(row["site"]), str(row["cohort"]))
        matches = [
            qc
            for qc in qc_rows
            if (normalize_subject(qc["subject"]), qc["site"], qc["condition"]) == key
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one Standard QC row for Stanford {key}, found {len(matches)}")
        qc = matches[0]
        return {"L": int(qc["bad_vertex_count_L"]), "R": int(qc["bad_vertex_count_R"])}

    subject_dir = JJ_STANDARD_RUN / "subjects" / f"sub-{subject}"
    if (subject_dir / "hemi-L_bad_vertices.shape.gii").is_file():
        return {
            hemi: int(np.count_nonzero(load_gifti_vector(subject_dir / f"hemi-{hemi}_bad_vertices.shape.gii") >= 0.5))
            for hemi in ("L", "R")
        }
    return {"L": 0, "R": 0}


def build_jobs(dataset: str) -> list[dict[str, Any]]:
    if dataset == "stanford":
        flow = read_csv(STANFORD_STANDARD_RUN / "cohort_flow.csv")
        selected = [row for row in flow if as_bool(row.get("include"))]
        stanford_qc_rows = read_csv(STANFORD_STANDARD_RUN / "argon_mapping_qc.csv")
        jobs: list[dict[str, Any]] = []
        for row in selected:
            subject = normalize_subject(row["subject"])
            con_path = native_con_path(dataset, row)
            standard_bad = standard_bad_counts(dataset, row, stanford_qc_rows=stanford_qc_rows)
            jobs.append(
                {
                    "dataset": dataset,
                    "subject": subject,
                    "site": row.get("site", ""),
                    "site_dir": row.get("site_dir", ""),
                    "cohort": row.get("cohort", ""),
                    "group": row.get("group", ""),
                    "site_network": row.get("site_network", ""),
                    "target_hemi": row.get("target_hemi", ""),
                    "con_path": str(con_path.resolve()),
                    "standard_bad": standard_bad,
                }
            )
        return jobs

    # JJ_TMS branch: 78 eligible subjects
    manifest_78 = REPO_ROOT / "manifests/jj_tms_78_eligible_subjects.txt"
    if manifest_78.is_file():
        eligible_subs = [
            line.strip()
            for line in manifest_78.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
    else:
        eligible_subs = []

    site_csv = REPO_ROOT / "outputs_iowa_jj_tms/present/tms_site_networks/jj_tms_kong2022_tms_site_networks.csv"
    site_records = {
        normalize_subject(row["subject"]): row
        for row in read_csv(site_csv)
    }

    jobs = []
    for sub in eligible_subs:
        subject = normalize_subject(sub)
        rec = site_records[subject]
        con_path = native_con_path("jj_tms", {"subject": subject})
        standard_bad = standard_bad_counts("jj_tms", {"subject": subject})
        group = "healthy" if subject.startswith("1") else "patient"
        target_hemi = str(rec.get("hemi", "") or ("L" if "LDLPFC" in str(rec.get("visit2_target", "")) else "R"))
        site_network = str(rec.get("canonical8", ""))
        jobs.append(
            {
                "dataset": "jj_tms",
                "subject": subject,
                "site": f"sub-{subject}",
                "site_dir": "",
                "cohort": "",
                "group": group,
                "site_network": site_network,
                "target_hemi": target_hemi,
                "con_path": str(con_path.resolve()),
                "standard_bad": standard_bad,
            }
        )
    return jobs


def save_roi(con_path: Path, output_path: Path, policy: str) -> dict[str, Any]:
    image = nib.load(str(con_path))
    data = np.asarray(image.dataobj, dtype=np.float64)
    finite = np.isfinite(data)
    if policy == "finite":
        roi = finite
    elif policy == "nonzero":
        roi = finite & (data != 0)
    else:
        raise ValueError(f"unsupported ROI policy: {policy}")
    if not np.any(roi):
        raise ValueError(f"ROI has no voxels for {con_path} policy={policy}")
    header = image.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(roi.astype(np.uint8), image.affine, header), str(output_path))
    return {
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "finite_voxels": int(finite.sum()),
        "finite_fraction": float(finite.mean()),
        "zero_voxels": int(np.count_nonzero(data == 0)),
        "zero_fraction": float(np.mean(data == 0)),
        "roi_voxels": int(roi.sum()),
        "roi_fraction": float(roi.mean()),
        "nan_voxels": int(np.isnan(data).sum()),
        "posinf_voxels": int(np.isposinf(data).sum()),
        "neginf_voxels": int(np.isneginf(data).sum()),
        "affine": np.asarray(image.affine, dtype=np.float64).tolist(),
    }


def map_one_policy(
    job: dict[str, Any],
    *,
    run_dir: Path,
    policy: str,
    workbench: list[str],
) -> list[dict[str, Any]]:
    dataset = str(job["dataset"])
    subject = str(job["subject"])
    if dataset == "stanford":
        site_slug = re.sub(r"[^A-Za-z0-9]+", "_", str(job["site"])).strip("_")
        subject_dir = run_dir / "subjects" / f"sub-{subject}" / f"site-{site_slug}" / f"condition-{job['cohort']}"
    else:
        subject_dir = run_dir / "subjects" / f"sub-{subject}"
    subject_dir.mkdir(parents=True, exist_ok=True)

    con_path = Path(str(job["con_path"]))
    roi_path = subject_dir / f"{policy}_con_roi.nii.gz"
    con_summary = save_roi(con_path, roi_path, policy)
    rows: list[dict[str, Any]] = []
    for hemi in ("L", "R"):
        surfaces = native_surface_paths(dataset, subject, hemi)
        metric_path = subject_dir / f"hemi-{hemi}_con_beta_{policy}.func.gii"
        bad_path = subject_dir / f"hemi-{hemi}_bad_vertices_{policy}.shape.gii"
        command = [
            *workbench,
            "-volume-to-surface-mapping",
            str(con_path),
            str(surfaces["midthickness"]),
            str(metric_path),
            "-ribbon-constrained",
            str(surfaces["inner"]),
            str(surfaces["outer"]),
            "-volume-roi",
            str(roi_path),
            "-voxel-subdiv",
            "3",
            "-bad-vertices-out",
            str(bad_path),
        ]
        subprocess.run(command, check=True)
        metric = load_gifti_vector(metric_path)
        bad = load_gifti_vector(bad_path)
        bad_count = int(np.count_nonzero(bad >= 0.5))
        rows.append(
            {
                "dataset": dataset,
                "subject": subject,
                "site": job["site"],
                "cohort": job["cohort"],
                "group": job["group"],
                "site_network": job["site_network"],
                "target_hemi": job["target_hemi"],
                "hippocampal_hemi": hemi,
                "roi_policy": policy,
                "native_con_path": str(con_path),
                "native_surface_space": STANFORD_NATIVE_SURFACE_SPACE if dataset == "stanford" else JJ_NATIVE_SURFACE_SPACE,
                "native_bad_vertex_count": bad_count,
                "native_bad_vertex_rate": float(bad_count / EXPECTED_VERTICES),
                "native_valid_vertex_count": int(np.count_nonzero((bad < 0.5) & np.isfinite(metric))),
                "native_metric_finite_count": int(np.isfinite(metric).sum()),
                "standard_bad_vertex_count": int(job["standard_bad"][hemi]),
                "standard_bad_vertex_rate": float(job["standard_bad"][hemi] / EXPECTED_VERTICES),
                "bad_rate_delta_native_minus_standard": float(
                    bad_count / EXPECTED_VERTICES - job["standard_bad"][hemi] / EXPECTED_VERTICES
                ),
                "roi_voxels": con_summary["roi_voxels"],
                "roi_fraction": con_summary["roi_fraction"],
                "finite_voxels": con_summary["finite_voxels"],
                "zero_fraction": con_summary["zero_fraction"],
                "mapping_method": "ribbon_constrained_finite_roi",
                "voxel_subdiv": 3,
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["dataset"]), str(row["roi_policy"]), str(row["group"]), str(row["target_hemi"]))].append(row)
    output: list[dict[str, Any]] = []
    for (dataset, policy, group, target_hemi), items in sorted(groups.items()):
        standard = np.asarray([float(item["standard_bad_vertex_rate"]) for item in items])
        native = np.asarray([float(item["native_bad_vertex_rate"]) for item in items])
        delta = native - standard
        output.append(
            {
                "dataset": dataset,
                "roi_policy": policy,
                "group": group,
                "target_hemi": target_hemi,
                "n_surface_pairs": int(len(items)),
                "standard_mean_bad_rate": float(np.mean(standard)),
                "native_mean_bad_rate": float(np.mean(native)),
                "mean_delta_native_minus_standard": float(np.mean(delta)),
                "standard_median_bad_rate": float(np.median(standard)),
                "native_median_bad_rate": float(np.median(native)),
                "median_delta_native_minus_standard": float(np.median(delta)),
                "native_lower_count": int(np.count_nonzero(delta < 0)),
                "native_higher_count": int(np.count_nonzero(delta > 0)),
                "same_count": int(np.count_nonzero(delta == 0)),
                "standard_all_bad_count": int(np.count_nonzero(standard >= 1.0)),
                "native_all_bad_count": int(np.count_nonzero(native >= 1.0)),
            }
        )
    return output


def run_dataset(dataset: str, run_dir: Path, workers: int) -> None:
    if workers < 1 or workers > 8:
        raise ValueError("workers must be between 1 and 8")
    run_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(dataset)
    manifest = {
        "workflow": "compare_native_standard_surface_bad_rates",
        "dataset": dataset,
        "run_dir": str(run_dir),
        "standard_run": str(STANFORD_STANDARD_RUN if dataset == "stanford" else JJ_STANDARD_RUN),
        "native_con_root": str(STANFORD_NATIVE_CON_ROOT if dataset == "stanford" else JJ_NATIVE_CON_ROOT),
        "native_hippunfold_root": str(STANFORD_NATIVE_HIPP_ROOT if dataset == "stanford" else JJ_NATIVE_HIPP_ROOT),
        "native_surface_space": STANFORD_NATIVE_SURFACE_SPACE if dataset == "stanford" else JJ_NATIVE_SURFACE_SPACE,
        "n_jobs": len(jobs),
        "roi_policies": list(ROI_POLICIES),
        "primary_roi_policy": "nonzero",
        "native_zero_fill_note": "Native Float32 con products contain exact-zero background; finite ROI is retained only as a mechanical sensitivity analysis.",
        "mapping_method": "ribbon_constrained_finite_roi",
        "voxel_subdiv": 3,
        "only_volume_to_surface": True,
        "surface_to_volume": False,
        "status": "running",
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(
        run_dir / "native_jobs.csv",
        [
            {key: value for key, value in job.items() if key != "standard_bad"}
            for job in jobs
        ],
    )

    all_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    workbench = wb_prefix()
    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
        futures = {
            executor.submit(
                map_one_policy,
                job,
                run_dir=run_dir,
                policy=policy,
                workbench=workbench,
            ): (job, policy)
            for job in jobs
            for policy in ROI_POLICIES
        }
        for index, future in enumerate(as_completed(futures), start=1):
            job, policy = futures[future]
            try:
                all_rows.extend(future.result())
            except Exception as exc:  # noqa: BLE001 - retain per-job failure evidence
                errors.append(
                    {
                        "dataset": dataset,
                        "subject": str(job["subject"]),
                        "site": str(job["site"]),
                        "roi_policy": policy,
                        "error": repr(exc),
                    }
                )
            if index % 10 == 0 or index == len(futures):
                print(f"[{index}/{len(futures)}] {dataset} native mapping tasks", flush=True)

    if all_rows:
        write_csv(run_dir / "bad_rate_comparison.csv", sorted(all_rows, key=lambda row: (row["roi_policy"], row["subject"], row["site"], row["hippocampal_hemi"])))
        write_csv(run_dir / "bad_rate_summary.csv", summarize(all_rows))
    if errors:
        write_csv(run_dir / "mapping_errors.csv", errors)
    manifest.update(
        {
            "status": "failed" if errors else "complete",
            "n_rows": len(all_rows),
            "n_errors": len(errors),
            "outputs": {
                "bad_rate_comparison": str(run_dir / "bad_rate_comparison.csv"),
                "bad_rate_summary": str(run_dir / "bad_rate_summary.csv"),
            },
        }
    )
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if errors:
        raise RuntimeError(f"{dataset} native mapping had {len(errors)} failed tasks; see {run_dir / 'mapping_errors.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("stanford", "jj_tms"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out-root", type=Path, default=REPO_ROOT / "outputs_native_surface_comparison")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dataset(args.dataset, args.out_root / args.run_id / args.dataset, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

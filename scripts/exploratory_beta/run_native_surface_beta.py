#!/opt/miniconda3/envs/py314/bin/python
"""Exploratorily extract Native-space surface beta rows for Stanford or JJ_TMS.

This is not a numbered formal-workflow entry point.

The volume-to-surface conversion is completed upstream by
``compare_native_standard_surface_bad_rates.py``.  This entry point consumes
only its Native-space Workbench metrics, bad-vertex masks, and the local
den-512 HippUnfold geometry.  It never reads Standard-space metrics and never
calls surface-to-volume mapping.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAPPING_ROOTS = {
    "stanford": REPO_ROOT
    / "outputs_native_surface_comparison/stanford_native_standard_bad_compare_20260818/stanford",
    "jj_tms": REPO_ROOT
    / "outputs_native_surface_comparison/jj_tms_native_standard_bad_compare_20260818/jj_tms",
}
DEFAULT_OUT_ROOTS = {
    "stanford": REPO_ROOT / "outputs_stanford/work/tms_hipp_surface_beta_native",
    "jj_tms": REPO_ROOT / "outputs_iowa_jj_tms/work/sptms_hipp_cluster_surface_beta_native",
}
SURFACE_ROOTS = {
    "stanford": REPO_ROOT / "outputs_stanford/inputs/stanford/hippunfold/dense_corobl_batch",
    "jj_tms": REPO_ROOT / "outputs_iowa_jj_tms/inputs/iowa_jj_tms/hippunfold/dense_corobl_batch",
}
SURFACE_SPACES = {"stanford": "corobl", "jj_tms": "T1w"}
MATCH_ROOTS = {
    "stanford": REPO_ROOT / "outputs_stanford/work/hcp151_fixed_template_matching/subjects",
    "jj_tms": REPO_ROOT / "outputs_iowa_jj_tms/work/hcp151_fixed_template_matching/subjects",
}
HCP151_LABEL_ROOT = REPO_ROOT / "outputs_hcp/templates/hcp151_routea_full_k3_fisherz"
NETWORK_DISPLAY = {"Control": "FPCN", "Default": "DMN", "VentralAttention": "VAN"}
CLUSTER_NAMES = {1: "Anterior_VAN", 2: "Body_SMN", 3: "Posterior_DMN"}
HEMIS = ("L", "R")
PARCELLATIONS = ("template_matching", "hcp151")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def surface_path(dataset: str, subject: str, hemi: str, kind: str) -> Path:
    space = SURFACE_SPACES[dataset]
    root = SURFACE_ROOTS[dataset] / f"sub-{subject}/hippunfold/sub-{subject}/surf"
    return root / f"sub-{subject}_hemi-{hemi}_space-{space}_den-512_label-hipp_{kind}.surf.gii"


def load_surface_areas(path: Path) -> np.ndarray:
    image = nib.load(str(path))
    coordinates = np.asarray(image.agg_data("pointset"), dtype=np.float64)
    triangles = np.asarray(image.agg_data("triangle"), dtype=np.int64)
    if coordinates.shape != (512, 3) or triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(f"unexpected den-512 surface geometry: {path}")
    if np.any(triangles < 0) or np.any(triangles >= 512):
        raise ValueError(f"surface triangle index out of range: {path}")
    triangle_area = 0.5 * np.linalg.norm(
        np.cross(
            coordinates[triangles[:, 1]] - coordinates[triangles[:, 0]],
            coordinates[triangles[:, 2]] - coordinates[triangles[:, 0]],
        ),
        axis=1,
    )
    areas = np.zeros(512, dtype=np.float64)
    for corner in range(3):
        np.add.at(areas, triangles[:, corner], triangle_area / 3.0)
    if not np.all(np.isfinite(areas)) or np.any(areas <= 0):
        raise ValueError(f"invalid vertex areas: {path}")
    return areas


def load_labels(dataset: str, subject: str, hemi: str, parcellation: str) -> np.ndarray:
    if parcellation == "template_matching":
        path = MATCH_ROOTS[dataset] / f"sub-{subject}/matching/2mm/hemi-{hemi}/labels.npy"
    else:
        path = HCP151_LABEL_ROOT / f"hcp151_2mm_hemi-{hemi}_labels.npy"
    labels = np.asarray(np.load(path), dtype=np.int32).reshape(-1)
    if labels.shape != (512,) or not set(np.unique(labels)).issubset({0, 1, 2, 3}):
        raise ValueError(f"invalid K=3 hippocampal labels: {path}")
    if not np.all(np.bincount(labels, minlength=4)[1:] > 0):
        raise ValueError(f"one or more hippocampal clusters are empty: {path}")
    return labels


def mapping_dir(mapping_root: Path, dataset: str, row: dict[str, str]) -> Path:
    subject_dir = mapping_root / "subjects" / f"sub-{row['subject']}"
    if dataset == "stanford":
        return subject_dir / f"site-{row['site_dir']}" / f"condition-{row['cohort']}"
    return subject_dir


def metric_path(mapping_root: Path, dataset: str, row: dict[str, str], hemi: str, kind: str, policy: str) -> Path:
    directory = mapping_dir(mapping_root, dataset, row)
    return directory / f"hemi-{hemi}_{kind}_{policy}.{ 'func.gii' if kind == 'con_beta' else 'shape.gii'}"


def extract_site(
    dataset: str,
    row: dict[str, str],
    mapping_root: Path,
    output_subject_dir: Path,
    policy: str,
    area_cache: dict[tuple[str, str], np.ndarray],
    label_cache: dict[tuple[str, str, str], np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    subject = str(row["subject"])
    native_dir = mapping_dir(mapping_root, dataset, row)
    if not native_dir.is_dir():
        raise FileNotFoundError(native_dir)
    output_subject_dir.mkdir(parents=True, exist_ok=True)
    sampled: dict[str, np.ndarray] = {}
    bad_masks: dict[str, np.ndarray] = {}
    areas: dict[str, np.ndarray] = {}
    for hemi in HEMIS:
        metric = metric_path(mapping_root, dataset, row, hemi, "con_beta", policy)
        bad_path = metric_path(mapping_root, dataset, row, hemi, "bad_vertices", policy)
        sampled[hemi] = np.asarray(nib.load(str(metric)).agg_data(), dtype=np.float64).reshape(-1)
        bad = np.asarray(nib.load(str(bad_path)).agg_data(), dtype=np.float64).reshape(-1)
        if sampled[hemi].shape != (512,) or bad.shape != (512,):
            raise ValueError(f"unexpected Native metric shape under {native_dir}: {metric}, {bad_path}")
        bad_masks[hemi] = bad >= 0.5
        area_key = (subject, hemi)
        if area_key not in area_cache:
            areas[hemi] = load_surface_areas(surface_path(dataset, subject, hemi, "midthickness"))
            area_cache[area_key] = areas[hemi]
        else:
            areas[hemi] = area_cache[area_key]

    rows: list[dict[str, Any]] = []
    parcellation_exclusions: list[dict[str, str]] = []
    for parcellation in PARCELLATIONS:
        labels: dict[str, np.ndarray] = {}
        try:
            for hemi in HEMIS:
                key = (subject, hemi, parcellation)
                if key not in label_cache:
                    label_cache[key] = load_labels(dataset, subject, hemi, parcellation)
                labels[hemi] = label_cache[key]
        except FileNotFoundError as exc:
            # Stanford template matching intentionally uses the fixed 194-person
            # QC-passed roster.  The Native con/mapping cohort is larger (966
            # site-level jobs), so the five HippUnfold-only subjects without
            # template-matching labels are excluded from this branch while the
            # independent HCP151 fixed atlas remains estimable.  Missing HCP151
            # labels, or any JJ label, remains a hard extraction failure.
            if dataset == "stanford" and parcellation == "template_matching":
                parcellation_exclusions.append(
                    {
                        "subject": subject,
                        "site": str(row.get("site", "")),
                        "site_dir": str(row.get("site_dir", "")),
                        "cohort": str(row.get("cohort", "")),
                        "parcellation": parcellation,
                        "reason": f"missing individual template-matching label: {exc}",
                    }
                )
                continue
            raise
        for hemi in HEMIS:
            finite_metric = np.isfinite(sampled[hemi])
            for cluster in (1, 2, 3):
                label_mask = labels[hemi] == cluster
                valid_mask = label_mask & ~bad_masks[hemi] & finite_metric
                values = sampled[hemi][valid_mask]
                weights = areas[hemi][valid_mask]
                beta_mean = float(np.average(values, weights=weights)) if values.size else math.nan
                beta_unweighted = float(np.mean(values)) if values.size else math.nan
                beta_sd = (
                    float(np.sqrt(np.average((values - beta_mean) ** 2, weights=weights)))
                    if values.size
                    else math.nan
                )
                site_name = str(row.get("site", ""))
                if not site_name:
                    site_name = f"sub-{subject}"
                rows.append(
                    {
                        "subject": subject,
                        "group": str(row.get("group", "all") or "all"),
                        "cohort": str(row.get("cohort", "") or ""),
                        "site": str(row.get("site", "") or ""),
                        "site_dir": str(row.get("site_dir", "") or ""),
                        "site_network": str(row["site_network"]),
                        "site_network_display": NETWORK_DISPLAY.get(str(row["site_network"]), str(row["site_network"])),
                        "target_hemi": str(row["target_hemi"]),
                        "hippocampal_hemi": hemi,
                        "cluster": cluster,
                        "cluster_name": CLUSTER_NAMES[cluster],
                        "parcellation": parcellation,
                        "beta_mean": beta_mean,
                        "beta_mean_unweighted": beta_unweighted,
                        "beta_sd_weighted": beta_sd,
                        "beta_median": float(np.median(values)) if values.size else math.nan,
                        "vertex_count": int(values.size),
                        "label_vertex_count": int(np.count_nonzero(label_mask)),
                        "bad_vertex_count": int(np.count_nonzero(bad_masks[hemi])),
                        "valid_vertex_fraction": float(values.size / np.count_nonzero(label_mask)),
                        "sampled_surface_area_mm2": float(np.sum(weights)) if values.size else 0.0,
                        "label_surface_area_mm2": float(np.sum(areas[hemi][label_mask])),
                        "measurement_domain": "hippocampal_surface",
                        "surface_space": SURFACE_SPACES[dataset],
                        "surface_density": "den-512",
                        "mapping_method": "native_workbench_volume_to_surface_ribbon_constrained",
                        "roi_policy": policy,
                        "native_metric_path": str(metric_path(mapping_root, dataset, row, hemi, "con_beta", policy)),
                        "native_bad_vertices_path": str(metric_path(mapping_root, dataset, row, hemi, "bad_vertices", policy)),
                        "native_con_path": str(row["con_path"]),
                        "site_id_for_aggregation": site_name,
                        "gray_z_scored": False,
                        "resampled": False,
                    }
                )
    write_csv(output_subject_dir / "cluster_beta_rows.csv", rows)
    return rows, parcellation_exclusions


def aggregate_subject_network(site_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "subject", "group", "cohort", "site_network", "site_network_display",
        "target_hemi", "hippocampal_hemi", "cluster", "cluster_name", "parcellation",
    )
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in site_rows:
        grouped[tuple(str(row.get(key, "")) for key in keys)].append(row)
    aggregated: list[dict[str, Any]] = []
    for key, cells in sorted(grouped.items()):
        values = np.asarray([float(row["beta_mean"]) for row in cells if np.isfinite(float(row["beta_mean"]))], dtype=float)
        site_ids = sorted({str(row["site_id_for_aggregation"]) for row in cells})
        record = dict(zip(keys, key))
        record.update(
            {
                "beta_mean": float(np.mean(values)) if values.size else math.nan,
                "n_sites": len(site_ids),
                "n_sites_valid": int(values.size),
                "site_beta_sd": float(np.std(values, ddof=1)) if values.size >= 2 else math.nan,
                "site_ids": ";".join(site_ids),
                "aggregation": "equal_weight_site_mean_within_subject_network_target_hemi",
                "measurement_domain": "hippocampal_surface",
                "mapping_method": "native_workbench_volume_to_surface_ribbon_constrained",
                "roi_policy": str(cells[0].get("roi_policy", "nonzero")),
                "gray_z_scored": False,
                "resampled": False,
            }
        )
        aggregated.append(record)
    return aggregated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("stanford", "jj_tms"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mapping-root", type=Path)
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--roi-policy", choices=("finite", "nonzero"), default="nonzero")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = args.dataset
    mapping_root = (args.mapping_root or DEFAULT_MAPPING_ROOTS[dataset]).resolve()
    out_root = (args.out_root or DEFAULT_OUT_ROOTS[dataset]).resolve()
    run_dir = out_root / "runs" / args.run_id
    jobs_path = mapping_root / "native_jobs.csv"
    if not jobs_path.is_file():
        raise FileNotFoundError(jobs_path)
    jobs = read_csv(jobs_path)
    if not jobs:
        raise ValueError(f"empty native job table: {jobs_path}")
    required = {"subject", "site_network", "target_hemi", "con_path"}
    missing = sorted(required - set(jobs[0]))
    if missing:
        raise ValueError(f"native job table is missing columns: {missing}")
    keys = [(row["subject"], row.get("site_dir", ""), row.get("cohort", "")) for row in jobs]
    if len(keys) != len(set(keys)):
        raise ValueError("native job table contains duplicate subject/site/cohort keys")
    if dataset == "stanford" and len(jobs) != 966:
        raise ValueError(f"expected 966 Stanford top-three native jobs, found {len(jobs)}")
    if dataset == "jj_tms" and len(jobs) != 59:
        raise ValueError(f"expected 59 JJ_TMS native jobs, found {len(jobs)}")

    run_dir.mkdir(parents=True, exist_ok=True)
    cohort_rows: list[dict[str, Any]] = []
    site_rows: list[dict[str, Any]] = []
    area_cache: dict[tuple[str, str], np.ndarray] = {}
    label_cache: dict[tuple[str, str, str], np.ndarray] = {}
    errors: list[dict[str, str]] = []
    parcellation_exclusions: list[dict[str, str]] = []
    for index, row in enumerate(jobs, start=1):
        subject_dir = run_dir / "subjects" / f"sub-{row['subject']}"
        if dataset == "stanford":
            subject_dir = subject_dir / f"site-{row['site_dir']}" / f"condition-{row['cohort']}"
        print(f"[{index}/{len(jobs)}] {dataset} sub-{row['subject']}", flush=True)
        try:
            extracted, excluded = extract_site(
                dataset, row, mapping_root, subject_dir, args.roi_policy, area_cache, label_cache
            )
            site_rows.extend(extracted)
            parcellation_exclusions.extend(excluded)
            excluded_names = sorted({item["parcellation"] for item in excluded})
            cohort_rows.append(
                {
                    **row,
                    "extraction_status": "complete",
                    "extraction_error": "",
                    "parcellation_exclusions": ";".join(excluded_names),
                    "parcellations_extracted": ";".join(
                        name for name in PARCELLATIONS if name not in excluded_names
                    ),
                }
            )
        except Exception as exc:  # noqa: BLE001 - preserve row-level provenance
            errors.append({"subject": row["subject"], "site": row.get("site", ""), "error": repr(exc)})
            cohort_rows.append({**row, "extraction_status": "failed", "extraction_error": repr(exc)})

    write_csv(run_dir / "cohort_flow.csv", cohort_rows)
    if parcellation_exclusions:
        write_csv(run_dir / "parcellation_exclusions.csv", parcellation_exclusions)
    if errors:
        write_csv(run_dir / "extraction_errors.csv", errors)
        raise RuntimeError(f"{len(errors)} Native beta extraction rows failed; see {run_dir / 'extraction_errors.csv'}")
    subject_rows = aggregate_subject_network(site_rows)
    write_csv(run_dir / "site_level_beta.csv", site_rows)
    write_csv(run_dir / "subject_network_beta.csv", subject_rows)
    summary = {
        "status": "complete",
        "dataset": dataset,
        "run_id": args.run_id,
        "native_mapping_root": str(mapping_root),
        "native_jobs_csv": str(jobs_path),
        "native_surface_root": str(SURFACE_ROOTS[dataset]),
        "native_surface_space": SURFACE_SPACES[dataset],
        "roi_policy": args.roi_policy,
        "parcellations": list(PARCELLATIONS),
        "hippocampal_hemispheres": list(HEMIS),
        "bilateral_rows_emitted": False,
        "volume_to_surface_only": True,
        "surface_to_volume_executed": False,
        "gray_z_scored": False,
        "resampled": False,
        "native_jobs": len(jobs),
        "extraction_failed_jobs": len(errors),
        "parcellation_exclusion_rows": len(parcellation_exclusions),
        "parcellation_exclusion_subjects": sorted({row["subject"] for row in parcellation_exclusions}),
        "parcellation_exclusion_counts": {
            name: sum(row["parcellation"] == name for row in parcellation_exclusions)
            for name in PARCELLATIONS
        },
        "site_level_beta_rows": len(site_rows),
        "subject_network_beta_rows": len(subject_rows),
        "finite_site_beta_rows": int(sum(np.isfinite(float(row["beta_mean"])) for row in site_rows)),
        "finite_subject_network_beta_rows": int(sum(np.isfinite(float(row["beta_mean"])) for row in subject_rows)),
        "network_counts": {network: sum(row["site_network"] == network for row in jobs) for network in NETWORK_DISPLAY},
        "outputs": {
            "cohort_flow": str(run_dir / "cohort_flow.csv"),
            "parcellation_exclusions": str(run_dir / "parcellation_exclusions.csv")
            if parcellation_exclusions
            else None,
            "site_level_beta": str(run_dir / "site_level_beta.csv"),
            "subject_network_beta": str(run_dir / "subject_network_beta.csv"),
        },
    }
    (run_dir / "extraction_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "workflow": "native_surface_beta",
        "script": str(Path(__file__).resolve()),
        "status": "complete",
        "summary": str(run_dir / "extraction_summary.json"),
        "measurement_contract": {
            "input_space": "Native subject space",
            "mapping_source": "precomputed Native Workbench volume-to-surface metrics",
            "roi_policy": args.roi_policy,
            "hippocampal_hemispheres": list(HEMIS),
            "bilateral_rows": False,
            "surface_to_volume": False,
        },
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

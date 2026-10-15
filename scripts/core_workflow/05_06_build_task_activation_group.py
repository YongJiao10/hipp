#!/usr/bin/env python3
from __future__ import annotations

"""
Process Step Description:
Build group-level task-activation products. Stacks per-subject contrast maps
across the resolved roster, writes group mean / one-sample t / Cohen's d
maps, per-subject cluster tables in two subregion zones (hcp151_atlas group
labels and template_matching individualized labels), and a group summary
with roster provenance.

Main Flow Path:
- Stage-05 task branch (group activation):
  04_03 (per-subject contrasts) -> 05_06_build_task_activation_group.py (this step) -> 06_11 render -> brain-behavior tables.

Inputs:
- Command-line arguments:
  - --task: Task branch (required: emotion | wm | relational).
  - --shared-root: Task shared root holding per-subject contrasts (default: `outputs_hcp/inputs/hcp/kong2022/shared_inputs/task-<task>`).
  - --subjects / --subjects-manifest: Required complete analysis roster; missing subject products are errors.
  - --smoothings / --hemis / --contrasts: Combos to build; --contrasts defaults to all maintained contrasts.
  - --zones: Subregion branches: hcp151_atlas (rest group K=3), template_matching (crossfit individualized K=3). Default: both.
  - --group-atlas-root: Rest gradient-spectral root holding K=3 labels (default: routea_17net root, which reuses the RouteA gradient bank).
  - --group-average-surfaces-dir: Group-average midthickness surfaces for anatomical cluster naming.
  - --hippunfold-batch-root: Native HippUnfold input root for per-subject vertex areas.
  - --crossfit-subjects-root: Crossfit individualized-label subject root.
  - --out-root: Stable group output root (default: `outputs_hcp/work/task_activation/task-<task>/group/`).
- Physical inputs:
  - Per-subject contrasts: `{shared_root}/sub-*/task/{task}/{smoothing}/hemi_{hemi}/contrast_{name}.npy`.
  - Rest labels: `{group_atlas_root}/gradient-spectral/.../group_labels_final.npy` + support mask + manifest.
  - Individual labels: `{crossfit_subjects_root}/sub-*/{smoothing}/hemi_{hemi}/individual_labels_k3.npy`.

Outputs:
- Command-line output / file outputs:
  - Group maps: `{out_root}/{smoothing}/hemi_{hemi}/group_contrast_{name}_{mean,tstat,cohens_d}.npy` plus per-condition `group_beta_{stem}_{mean,tstat,cohens_d}.npy`.
  - `{out_root}/subject_cluster_table.tsv`, `{out_root}/group_cluster_summary.tsv` (with BH-FDR `group_p_fdr`), `{out_root}/group_build_manifest.json` (contrast column holds maintained contrasts and `beta_{stem}` items).
  - `{out_root}/group_bin_summary.tsv`: 2mm subject-level bin means, effective
    n, and two-sided one-sample t/p; BH-FDR across task contrasts x hemispheres
    x 20 bins. Beta rows provide colored profile means and have no FDR test.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from lib.io_utils import load_subject_ids, write_tsv
from lib.output_layout import (
    DEFAULT_HIPPUNFOLD_BATCH_ROOT,
    DEFAULT_KONG2022_TEMPLATE_ROOT,
)
from lib.task_design import TASK_CONTRASTS, normalize_task_branch
from lib.surface_utils import equal_vertex_long_axis_bins, long_axis_order
from lib.task_group import (
    N_LONG_AXIS_BINS,
    PARCELLATION_CHOICES,
    PARCELLATION_HCP151_ATLAS,
    PARCELLATION_TEMPLATE_MATCHING,
    anatomical_cluster_names,
    fdr_adjust,
    group_summary_stats,
    group_bin_stats,
    individual_labels_path,
    load_group_labels,
    load_subject_betas,
    load_subject_stack,
    native_vertex_areas,
    resolve_zone_subjects,
    subject_cluster_rows,
    two_tailed_p,
)
from lib.workflow_utils import capped_jobs, file_meta, utc_now_iso, write_json


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GROUP_ATLAS_ROOT = (
    REPO_ROOT
    / "outputs_hcp"
    / "work"
    / "group_atlas"
    / "routea_17net_full_k3_fisherz"
)
DEFAULT_GROUP_AVG_SURFACES_DIR = (
    DEFAULT_KONG2022_TEMPLATE_ROOT / "group_average_surfaces"
)
DEFAULT_CROSSFIT_SUBJECTS_ROOT = (
    REPO_ROOT / "outputs_hcp" / "work" / "crossfit_template_matching" / "subjects"
)
SMOOTHINGS = ["2mm"]
HEMIS = ["L", "R"]


def default_shared_root(task: str) -> Path:
    return (
        REPO_ROOT
        / "outputs_hcp"
        / "inputs"
        / "hcp"
        / "kong2022"
        / "shared_inputs"
        / f"task-{task}"
    )


def default_out_root(task: str) -> Path:
    return REPO_ROOT / "outputs_hcp" / "work" / "task_activation" / f"task-{task}" / "group"


def group_average_surface(surfaces_dir: Path, hemi: str) -> Path:
    path = (
        Path(surfaces_dir)
        / f"group-average_hemi-{hemi}_space-corobl_den-512_label-hipp_midthickness.surf.gii"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing group-average surface: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Group task-activation build.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--shared-root", default=None)
    cohort = parser.add_mutually_exclusive_group(required=True)
    cohort.add_argument("--subjects", nargs="+", default=None)
    cohort.add_argument("--subjects-manifest", default=None)
    parser.add_argument("--smoothings", nargs="*", default=SMOOTHINGS)
    parser.add_argument("--hemis", nargs="*", default=HEMIS)
    parser.add_argument("--contrasts", nargs="*", default=None)
    parser.add_argument("--zones", nargs="*", default=list(PARCELLATION_CHOICES))
    parser.add_argument("--group-atlas-root", default=str(DEFAULT_GROUP_ATLAS_ROOT))
    parser.add_argument("--group-average-surfaces-dir", default=str(DEFAULT_GROUP_AVG_SURFACES_DIR))
    parser.add_argument("--hippunfold-batch-root", default=str(DEFAULT_HIPPUNFOLD_BATCH_ROOT))
    parser.add_argument("--crossfit-subjects-root", default=str(DEFAULT_CROSSFIT_SUBJECTS_ROOT))
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--min-valid-area-fraction",
        type=float,
        default=0.10,
        help="Subject-zone entries join statistics only when their finite "
        "tissue area reaches this fraction of the zone's support area "
        "(preset coverage gate; gated-out entries are reported, not silent).",
    )
    args = parser.parse_args()

    task = normalize_task_branch(args.task)
    zones = list(args.zones)
    unknown_zones = [z for z in zones if z not in PARCELLATION_CHOICES]
    if not zones or unknown_zones:
        raise ValueError(f"Invalid --zones {args.zones}; expected subset of {PARCELLATION_CHOICES}")
    min_valid_area_fraction = float(args.min_valid_area_fraction)
    if not 0.0 < min_valid_area_fraction <= 1.0:
        raise ValueError(
            f"--min-valid-area-fraction must be in (0, 1], got {min_valid_area_fraction}"
        )
    shared_root = Path(args.shared_root) if args.shared_root else default_shared_root(task)
    out_root = Path(args.out_root) if args.out_root else default_out_root(task)
    group_atlas_root = Path(args.group_atlas_root)
    surfaces_dir = Path(args.group_average_surfaces_dir)
    batch_root = Path(args.hippunfold_batch_root)
    crossfit_root = Path(args.crossfit_subjects_root)
    if args.subjects:
        analysis_roster = sorted({s.replace("sub-", "") for s in args.subjects})
    else:
        analysis_roster = sorted(load_subject_ids(Path(args.subjects_manifest)))

    subject_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    input_metas: list[dict[str, object]] = []
    combos: list[dict[str, object]] = []
    areas_cache: dict[tuple[str, str], np.ndarray] = {}
    bin_rows: list[dict[str, object]] = []
    bins_by_hemi: dict[str, list[np.ndarray]] = {}

    def collect_bin_stats(stack: np.ndarray, smoothing: str, hemi: str, contrast: str) -> None:
        if smoothing != "2mm":
            return
        if hemi not in bins_by_hemi:
            bins_by_hemi[hemi] = equal_vertex_long_axis_bins(
                long_axis_order(group_average_surface(surfaces_dir, hemi)), N_LONG_AXIS_BINS
            )
        stats = group_bin_stats(stack, bins_by_hemi[hemi])
        for i in range(N_LONG_AXIS_BINS):
            bin_rows.append({
                "task": task, "smoothing": smoothing, "hemi": hemi,
                "contrast": contrast, "bin": i + 1,
                "n_subjects": int(stats["n"][i]), "group_mean": float(stats["mean"][i]),
                "group_t": float(stats["tstat"][i]), "group_p": float(stats["p"][i]),
                "group_p_fdr": float("nan"),
            })

    for smoothing in args.smoothings:
        for hemi in args.hemis:
            contrasts = (
                list(args.contrasts)
                if args.contrasts
                else list(TASK_CONTRASTS[task])
            )
            if not contrasts:
                raise ValueError(f"No contrasts selected for {smoothing}/hemi-{hemi}")
            labels, mask, atlas_manifest = load_group_labels(
                group_atlas_root, smoothing=smoothing, hemi=hemi, k=3
            )
            names_by_id = anatomical_cluster_names(
                manifest=atlas_manifest,
                group_labels=labels,
                surface_path=group_average_surface(surfaces_dir, hemi),
            )
            # One unified roster per combo: every subject must hold all
            # contrast files AND task betas, so profile lines and cluster
            # tables always describe the same subjects.
            unified_roster: list[str] = []
            excluded_incomplete: list[str] = []
            for subject in analysis_roster:
                task_dir = (
                    Path(shared_root)
                    / f"sub-{subject}"
                    / "task"
                    / task
                    / smoothing
                    / f"hemi_{hemi}"
                )
                missing = [
                    name
                    for name in ["task_betas.npy"]
                    + [f"contrast_{c}.npy" for c in contrasts]
                    if not (task_dir / name).is_file()
                ]
                if not missing:
                    unified_roster.append(subject)
                else:
                    excluded_incomplete.append(
                        f"sub-{subject} (missing: {', '.join(missing)})"
                    )
            if excluded_incomplete:
                raise ValueError(
                    f"Explicit roster is incomplete for {smoothing}/hemi-{hemi}: "
                    f"{excluded_incomplete}. Re-run 04_03 for these subjects "
                    "instead of silently changing the cohort."
                )
            if len(unified_roster) < 2:
                raise ValueError(
                    f"Need >=2 complete subjects for {smoothing}/hemi-{hemi}, "
                    f"got {len(unified_roster)} (excluded: {excluded_incomplete})"
                )
            beta_roster = list(unified_roster)
            beta_items: list[str] = []
            beta_combo_dir = out_root / smoothing / f"hemi_{hemi}"
            beta_combo_dir.mkdir(parents=True, exist_ok=True)
            beta_stacks, beta_items = load_subject_betas(
                shared_root,
                subjects=beta_roster,
                task=task,
                smoothing=smoothing,
                hemi=hemi,
            )
            for stem in beta_items:
                beta_stats = group_summary_stats(beta_stacks[stem])
                collect_bin_stats(beta_stacks[stem], smoothing, hemi, f"beta_{stem}")
                for key in ("mean", "tstat", "cohens_d"):
                    np.save(
                        str(beta_combo_dir / f"group_beta_{stem}_{key}.npy"),
                        np.asarray(beta_stats[key], dtype=np.float32),
                    )
            excluded_incomplete_combo = list(excluded_incomplete)
            for contrast in contrasts:
                if contrast not in TASK_CONTRASTS[task]:
                    raise ValueError(
                        f"Contrast {contrast!r} is not maintained for task {task!r}"
                    )
                roster = list(unified_roster)
                stack, used = load_subject_stack(
                    shared_root,
                    subjects=roster,
                    task=task,
                    smoothing=smoothing,
                    hemi=hemi,
                    contrast=contrast,
                )
                stats = group_summary_stats(stack)
                collect_bin_stats(stack, smoothing, hemi, contrast)
                combo_dir = out_root / smoothing / f"hemi_{hemi}"
                combo_dir.mkdir(parents=True, exist_ok=True)
                for key in ("mean", "tstat", "cohens_d"):
                    np.save(
                        str(combo_dir / f"group_contrast_{contrast}_{key}.npy"),
                        np.asarray(stats[key], dtype=np.float32),
                    )
                for subject in used:
                    key = (subject, hemi)
                    if key not in areas_cache:
                        areas_cache[key] = native_vertex_areas(
                            batch_root, subject=subject, hemi=hemi
                        )
                vertex_areas = {subject: areas_cache[(subject, hemi)] for subject in used}
                for zone in zones:
                    zone_roster, zone_notes = resolve_zone_subjects(
                        zone=zone,
                        used=used,
                        smoothing=smoothing,
                        hemi=hemi,
                        crossfit_root=crossfit_root,
                    )
                    if len(zone_roster) < 2:
                        raise ValueError(
                            f"Need >=2 subjects for zone {zone} {contrast} "
                            f"{smoothing}/hemi-{hemi}, got {len(zone_roster)}"
                        )
                    zone_rows: list[dict[str, object]] = []
                    for subject in zone_roster:
                        if zone == PARCELLATION_TEMPLATE_MATCHING:
                            subject_labels = np.asarray(
                                np.load(
                                    str(
                                        individual_labels_path(
                                            crossfit_root,
                                            subject=subject,
                                            smoothing=smoothing,
                                            hemi=hemi,
                                        )
                                    )
                                ),
                                dtype=np.int32,
                            ).reshape(-1)
                        else:
                            subject_labels = labels
                        index = used.index(subject)
                        single = subject_cluster_rows(
                            stack[index : index + 1],
                            [subject],
                            group_labels=subject_labels,
                            support_mask=mask,
                            names_by_id=names_by_id,
                            vertex_areas={subject: vertex_areas[subject]},
                            min_valid_area_fraction=min_valid_area_fraction,
                        )
                        for row in single:
                            row.update(
                                {
                                    "task": task,
                                    "contrast": contrast,
                                    "smoothing": smoothing,
                                    "hemi": hemi,
                                    "parcellation": zone,
                                    "parcellation_note": zone_notes[subject],
                                }
                            )
                        zone_rows.extend(single)
                    subject_rows.extend(zone_rows)
                    cluster_ids = sorted(names_by_id)
                    per_subject_means = np.array(
                        [
                            [
                                next(
                                    r["mean"]
                                    for r in zone_rows
                                    if r["subject"] == f"sub-{s}"
                                    and r["cluster_id"] == c
                                )
                                for c in cluster_ids
                            ]
                            for s in zone_roster
                        ],
                        dtype=np.float64,
                    )
                    n = np.sum(np.isfinite(per_subject_means), axis=0)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        gmean = np.nanmean(per_subject_means, axis=0)
                        gsd = np.nanstd(per_subject_means, axis=0, ddof=1)
                        gt = gmean / (gsd / np.sqrt(np.maximum(n, 1)))
                    for idx, cluster_id in enumerate(cluster_ids):
                        tval = float(gt[idx])
                        nval = int(n[idx])
                        group_rows.append(
                            {
                                "task": task,
                                "contrast": contrast,
                                "smoothing": smoothing,
                                "hemi": hemi,
                                "parcellation": zone,
                                "cluster_id": int(cluster_id),
                                "cluster_name": names_by_id[cluster_id],
                                "n_subjects": nval,
                                "group_mean": float(gmean[idx]),
                                "group_sd": float(gsd[idx]),
                                "group_t": tval,
                                "group_p": two_tailed_p(tval, nval),
                            }
                        )
                    combos.append(
                        {
                            "smoothing": smoothing,
                            "hemi": hemi,
                            "contrast": contrast,
                            "parcellation": zone,
                            "n_subjects": len(zone_roster),
                            "roster": [f"sub-{s}" for s in zone_roster],
                            "excluded": [
                                f"sub-{s}" for s in used if s not in zone_roster
                            ],
                            "excluded_incomplete": excluded_incomplete_combo,
                        }
                    )
                for subject in used:
                    input_metas.append(
                        {
                            "subject": f"sub-{subject}",
                            "smoothing": smoothing,
                            "hemi": hemi,
                            "contrast": contrast,
                            "file": file_meta(
                                shared_root
                                / f"sub-{subject}"
                                / "task"
                                / task
                                / smoothing
                                / f"hemi_{hemi}"
                                / f"contrast_{contrast}.npy"
                            ),
                        }
                    )
                print(
                    json.dumps(
                        {
                            "contrast": contrast,
                            "smoothing": smoothing,
                            "hemi": hemi,
                            "n": len(used),
                            "zones": zones,
                        }
                    ),
                    flush=True,
                )
            if beta_items:
                for subject in beta_roster:
                    key = (subject, hemi)
                    if key not in areas_cache:
                        areas_cache[key] = native_vertex_areas(
                            batch_root, subject=subject, hemi=hemi
                        )
                beta_areas = {s: areas_cache[(s, hemi)] for s in beta_roster}
                for zone in zones:
                    zone_beta_roster, zone_notes = resolve_zone_subjects(
                        zone=zone,
                        used=beta_roster,
                        smoothing=smoothing,
                        hemi=hemi,
                        crossfit_root=crossfit_root,
                    )
                    if len(zone_beta_roster) < 2:
                        raise ValueError(
                            f"Need >=2 subjects for zone {zone} betas "
                            f"{smoothing}/hemi-{hemi}, got {len(zone_beta_roster)}"
                        )
                    for stem in beta_items:
                        beta_contrast = f"beta_{stem}"
                        stem_rows: list[dict[str, object]] = []
                        for subject in zone_beta_roster:
                            if zone == PARCELLATION_TEMPLATE_MATCHING:
                                subject_labels = np.asarray(
                                    np.load(
                                        str(
                                            individual_labels_path(
                                                crossfit_root,
                                                subject=subject,
                                                smoothing=smoothing,
                                                hemi=hemi,
                                            )
                                        )
                                    ),
                                    dtype=np.int32,
                                ).reshape(-1)
                            else:
                                subject_labels = labels
                            index = beta_roster.index(subject)
                            single = subject_cluster_rows(
                                beta_stacks[stem][index : index + 1],
                                [subject],
                                group_labels=subject_labels,
                                support_mask=mask,
                                names_by_id=names_by_id,
                                vertex_areas={subject: beta_areas[subject]},
                                min_valid_area_fraction=min_valid_area_fraction,
                            )
                            for row in single:
                                row.update(
                                    {
                                        "task": task,
                                        "contrast": beta_contrast,
                                        "smoothing": smoothing,
                                        "hemi": hemi,
                                        "parcellation": zone,
                                        "parcellation_note": zone_notes[subject],
                                    }
                                )
                            stem_rows.extend(single)
                        subject_rows.extend(stem_rows)
                        cluster_ids = sorted(names_by_id)
                        per_subject_means = np.array(
                            [
                                [
                                    next(
                                        r["mean"]
                                        for r in stem_rows
                                        if r["subject"] == f"sub-{s}"
                                        and r["cluster_id"] == c
                                    )
                                    for c in cluster_ids
                                ]
                                for s in zone_beta_roster
                            ],
                            dtype=np.float64,
                        )
                        n = np.sum(np.isfinite(per_subject_means), axis=0)
                        with np.errstate(invalid="ignore", divide="ignore"):
                            gmean = np.nanmean(per_subject_means, axis=0)
                            gsd = np.nanstd(per_subject_means, axis=0, ddof=1)
                            gt = gmean / (gsd / np.sqrt(np.maximum(n, 1)))
                        for idx, cluster_id in enumerate(cluster_ids):
                            tval = float(gt[idx])
                            nval = int(n[idx])
                            group_rows.append(
                                {
                                    "task": task,
                                    "contrast": beta_contrast,
                                    "smoothing": smoothing,
                                    "hemi": hemi,
                                    "parcellation": zone,
                                    "cluster_id": int(cluster_id),
                                    "cluster_name": names_by_id[cluster_id],
                                    "n_subjects": nval,
                                    "group_mean": float(gmean[idx]),
                                    "group_sd": float(gsd[idx]),
                                    "group_t": tval,
                                    "group_p": two_tailed_p(tval, nval),
                                }
                            )
                        combos.append(
                            {
                                "smoothing": smoothing,
                                "hemi": hemi,
                                "contrast": beta_contrast,
                                "parcellation": zone,
                                "n_subjects": len(zone_beta_roster),
                                "roster": [f"sub-{s}" for s in zone_beta_roster],
                                "excluded": [
                                    f"sub-{s}" for s in beta_roster if s not in zone_beta_roster
                                ],
                            }
                        )
                for subject in beta_roster:
                    input_metas.append(
                        {
                            "subject": f"sub-{subject}",
                            "smoothing": smoothing,
                            "hemi": hemi,
                            "contrast": "task_betas",
                            "file": file_meta(
                                shared_root
                                / f"sub-{subject}"
                                / "task"
                                / task
                                / smoothing
                                / f"hemi_{hemi}"
                                / "task_betas.npy"
                            ),
                        }
                    )
                print(
                    json.dumps(
                        {
                            "betas": beta_items,
                            "smoothing": smoothing,
                            "hemi": hemi,
                            "n": len(beta_roster),
                            "zones": zones,
                        }
                    ),
                    flush=True,
                )
    subject_fields = [
        "subject",
        "task",
        "contrast",
        "smoothing",
        "hemi",
        "parcellation",
        "parcellation_note",
        "cluster_id",
        "cluster_name",
        "n_vertices",
        "n_finite",
        "area_mm2",
        "area_total_mm2",
        "valid_area_fraction",
        "excluded_low_coverage",
        "mean",
        "sd",
    ]
    write_tsv(out_root / "subject_cluster_table.tsv", subject_rows, subject_fields)
    exclusion_rows: list[dict[str, object]] = []
    for row in subject_rows:
        if bool(row.get("excluded_low_coverage")):
            exclusion_rows.append(
                {
                    "subject": str(row["subject"]),
                    "contrast": str(row["contrast"]),
                    "smoothing": str(row["smoothing"]),
                    "hemi": str(row["hemi"]),
                    "parcellation": str(row["parcellation"]),
                    "cluster_name": str(row["cluster_name"]),
                    "valid_area_fraction": float(row["valid_area_fraction"]),
                }
            )
    write_tsv(
        out_root / "coverage_exclusions.tsv",
        exclusion_rows,
        [
            "subject",
            "contrast",
            "smoothing",
            "hemi",
            "parcellation",
            "cluster_name",
            "valid_area_fraction",
        ],
    )
    group_fields = [
        "task",
        "contrast",
        "smoothing",
        "hemi",
        "parcellation",
        "cluster_id",
        "cluster_name",
        "n_subjects",
        "group_mean",
        "group_sd",
        "group_t",
        "group_p",
        "group_p_fdr",
    ]
    fdr_values = fdr_adjust([float(r["group_p"]) for r in group_rows])
    for row, adjusted in zip(group_rows, fdr_values):
        row["group_p_fdr"] = adjusted
    write_tsv(out_root / "group_cluster_summary.tsv", group_rows, group_fields)
    # One family across all 2mm task contrasts, hemispheres and bins.
    # Condition betas supply the colored profile lines and are not tested here.
    contrast_bin_rows = [r for r in bin_rows if r["contrast"] in TASK_CONTRASTS[task]]
    bin_adjusted = fdr_adjust([float(r["group_p"]) for r in contrast_bin_rows])
    for row, adjusted in zip(contrast_bin_rows, bin_adjusted):
        row["group_p_fdr"] = adjusted
    bin_path = out_root / "group_bin_summary.tsv"
    write_tsv(bin_path, bin_rows, [
        "task", "smoothing", "hemi", "contrast", "bin", "n_subjects",
        "group_mean", "group_t", "group_p", "group_p_fdr",
    ])
    all_subjects = sorted({r["subject"] for r in subject_rows})
    write_json(
        out_root / "group_build_manifest.json",
        {
            "stage": "task_activation_group",
            "status": "done",
            "timestamp_utc": utc_now_iso(),
            "task": task,
            "zones": zones,
            "n_subjects_total": len(all_subjects),
            "roster": all_subjects,
            "roster_source": "explicit",
            "min_valid_area_fraction": min_valid_area_fraction,
            "n_coverage_excluded_entries": len(exclusion_rows),
            "smoothings": list(args.smoothings),
            "hemis": list(args.hemis),
            "combos": combos,
            "group_atlas_root": str(group_atlas_root.resolve()),
            "group_average_surfaces_dir": str(surfaces_dir.resolve()),
            "hippunfold_batch_root": str(batch_root.resolve()),
            "crossfit_subjects_root": str(crossfit_root.resolve()),
            "bin_summary": str(bin_path.resolve()),
            "bin_fdr_family": "2mm_task_contrasts_x_hemispheres_x_bins",
            "n_bin_tests": len(contrast_bin_rows),
            "inputs": input_metas,
        },
    )
    jobs_note = capped_jobs(args.jobs, max(1, len(all_subjects)))
    print(
        json.dumps(
            {
                "task": task,
                "n_subjects": len(all_subjects),
                "out_root": str(out_root.resolve()),
                "jobs": jobs_note,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

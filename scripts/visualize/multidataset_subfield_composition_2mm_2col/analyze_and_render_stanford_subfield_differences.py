#!/usr/bin/env python3
"""
analyze_and_render_stanford_subfield_differences.py

Statistical Analysis & Publication Figure Generator:
Testing differences in MultiHist7 hippocampal subfield compositions across
Stanford clinical cohorts and HCP 7T benchmark relative to the Stanford NTHC baseline.

Supports THREE distinct functional atlas partitioning paradigms:
  1. Version 1 (v1_fixed_hcp151_atlas):
     Unified fixed HCP 151 Group Atlas functional boundaries applied across all cohorts.
  2. Version 2 (v2_cohort_specific_atlas):
     Cohort-specific intrinsic Group Atlas functional boundaries derived from each cohort's own gradients.
  3. Version 3 (v3_individual_template_matching):
     Subject-level functional boundaries (HCP 7T uses 5-Fold Cross-Fit Template Matching;
     Stanford cohorts use HCP151 Fixed Template Matching).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import stats

REPO_ROOT = Path("/home/yojiao/hpchome/hipp")
OUTPUT_DIR = REPO_ROOT / "scripts/visualize/multidataset_subfield_composition_2mm_2col"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATS_DIR = OUTPUT_DIR / "stats"
STATS_DIR.mkdir(parents=True, exist_ok=True)

CORE_DIR = REPO_ROOT / "scripts/core_workflow"
VIS_LIB_DIR = REPO_ROOT / "scripts/visualize/lib"
for p in (CORE_DIR, VIS_LIB_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lib.hippunfold_run_contract import HIPPUNFOLD_ATLAS
from multidataset_cohorts import GridSpec, GRID_VARIANTS

HIPPUNFOLD_ROOT = REPO_ROOT / "outputs_stanford/inputs/hippunfold/dense_corobl_batch"
HCP_GROUP_ROOT = REPO_ROOT / "outputs_hcp/work/group_atlas/pfm"

BASE_COHORTS = [
    {
        "id": "stanford_nthc",
        "stem": "nthc",
        "name": "NTHC",
        "full_name": "Non-Trauma Controls",
        "desc": "Healthy Controls (3T Baseline)",
        "is_baseline": True,
        "color": "#1d5b79",
    },
    {
        "id": "hcp_7t",
        "stem": "hcp_7t",
        "name": "HCP 7T",
        "full_name": "HCP 7T Benchmark Controls",
        "desc": "Normative Benchmark (7T)",
        "is_baseline": False,
        "color": "#0b3c5d",
    },
    {
        "id": "stanford_tehc",
        "stem": "tehc",
        "name": "TEHC",
        "full_name": "Trauma-Exposed Controls",
        "desc": "Resilient Trauma (3T)",
        "is_baseline": False,
        "color": "#2e7d32",
    },
    {
        "id": "stanford_nts",
        "stem": "nts",
        "name": "NTS",
        "full_name": "Non-Trauma Symptomatic",
        "desc": "All Non-Trauma Patients",
        "is_baseline": False,
        "color": "#d97706",
    },
    {
        "id": "stanford_nts_any_mdd",
        "stem": "nts_any_mdd",
        "name": "NTS Any MDD",
        "full_name": "Non-Trauma MDD (All)",
        "desc": "MDD with/without Comorbidities",
        "is_baseline": False,
        "color": "#dc2626",
    },
    {
        "id": "stanford_nts_pure_mdd",
        "stem": "nts_pure_mdd",
        "name": "NTS Pure MDD",
        "full_name": "Non-Trauma Pure MDD",
        "desc": "Pure MDD (No Comorbidities)",
        "is_baseline": False,
        "color": "#7c2d12",
    },
    {
        "id": "stanford_tis",
        "stem": "tis",
        "name": "TIS",
        "full_name": "Trauma-Informed Symptomatic",
        "desc": "Trauma + PTSD/MDD (3T)",
        "is_baseline": False,
        "color": "#7c3aed",
    },
    {
        "id": "stanford_tis_ptsd",
        "stem": "tis_ptsd",
        "name": "TIS PTSD",
        "full_name": "Trauma-Informed PTSD",
        "desc": "Trauma + PTSD (3T)",
        "is_baseline": False,
        "color": "#4c1d95",
    },
]


def get_diff_cohorts(spec: GridSpec | None = None) -> list[dict]:
    """Resolve group atlas and hippunfold paths for the 8 Stanford + HCP cohorts."""
    if spec is None:
        spec = GridSpec("2mm", "36p")
    cohorts = []
    for c in BASE_COHORTS:
        cinfo = dict(c)
        if cinfo["id"] == "hcp_7t":
            cinfo["group_root"] = REPO_ROOT / "outputs_hcp/work/group_atlas/pfm"
            cinfo["hippunfold_root"] = REPO_ROOT / "outputs_hcp/inputs/hippunfold/dense_corobl_batch"
            cinfo["n"] = 151
        else:
            stem = cinfo["stem"]
            var_sfx = ""
            cinfo["group_root"] = REPO_ROOT / f"outputs_stanford/work/group_atlas/pfm{var_sfx}/{stem}"
            cinfo["hippunfold_root"] = REPO_ROOT / "outputs_stanford/inputs/hippunfold/dense_corobl_batch"
            default_ns = {
                "nthc": 51,
                "tehc": 40,
                "nts": 56,
                "nts_any_mdd": 43,
                "nts_pure_mdd": 11,
                "tis": 47,
                "tis_ptsd": 35,
            }
            cinfo["n"] = default_ns.get(stem, 0)
        cohorts.append(cinfo)
    return cohorts


COHORTS = get_diff_cohorts(GridSpec("2mm", "36p"))

SUBFIELD_ORDER = (1, 2, 3, 4, 5)
SUBFIELD_NAMES = {1: "Sub", 2: "CA1", 3: "CA2", 4: "CA3", 5: "CA4"}
SUBFIELD_FULL_NAMES = {
    1: "Subiculum",
    2: "CA1",
    3: "CA2",
    4: "CA3",
    5: "CA4",
}
SUBFIELD_COLORS = {
    1: "#292BFC",  # Sub: Royal Blue
    2: "#66D1D8",  # CA1: Sky / Cyan
    3: "#00B33C",  # CA2: Green
    4: "#FFB600",  # CA3: Amber / Gold
    5: "#FF2B17",  # CA4: Red
}

SUBREGION_NAMES = {1: "Ant", 2: "Middle", 3: "Post"}
SUBREGION_FULL_NAMES = {1: "Anterior", 2: "Middle", 3: "Posterior"}

ATLAS_VERSIONS = {
    "v1_fixed_hcp151_atlas": {
        "title_label": "VERSION 1: UNIFIED FIXED HCP151 GROUP ATLAS",
        "subtitle": "Functional Subregions Defined Uniformly by Canonical HCP 151 Group Atlas | Subfields: Native MultiHist7",
        "file_tag": "v1_fixed_hcp151_atlas",
        "paradox_note": "• Unified Fixed HCP151 Group Atlas eliminates cross-group functional boundary variance.",
    },
    "v2_cohort_specific_atlas": {
        "title_label": "VERSION 2: COHORT-SPECIFIC INTRINSIC GROUP ATLAS",
        "subtitle": "Functional Subregions Defined Intrinsically by Each Cohort's Own Group Atlas | Subfields: Native MultiHist7",
        "file_tag": "v2_cohort_specific_atlas",
        "paradox_note": "• Cohort-Specific Group Atlas preserves intrinsic group-level gradient spectral boundaries.",
    },
    "v3_individual_template_matching": {
        "title_label": "VERSION 3: INDIVIDUAL-LEVEL TEMPLATE MATCHING ATLAS",
        "subtitle": "Functional Subregions: Subject-Level Matching (HCP 7T: 5-Fold Cross-Fit; Stanford: Fixed HCP151) | Subfields: Native MultiHist7",
        "file_tag": "v3_individual_template_matching",
        "paradox_note": "• Individual Template Matching accounts for subject-specific functional topography shifts.",
    },
}


_LABEL_CACHE: dict[str, np.ndarray] = {}
_WEIGHT_CACHE: dict[str, np.ndarray] = {}


def vertex_area_weights(surface_path: Path) -> np.ndarray:
    p_str = str(surface_path)
    if p_str in _WEIGHT_CACHE:
        return _WEIGHT_CACHE[p_str]
    img = nib.load(p_str)
    coords = np.asarray(img.agg_data("pointset"), dtype=np.float64)
    faces = np.asarray(img.agg_data("triangle"), dtype=np.int64)
    tri = coords[faces]
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    weights = np.zeros(coords.shape[0], dtype=np.float64)
    for corner in range(3):
        np.add.at(weights, faces[:, corner], areas / 3.0)
    _WEIGHT_CACHE[p_str] = weights
    return weights


def get_subfield_labels(label_path: Path) -> np.ndarray:
    p_str = str(label_path)
    if p_str in _LABEL_CACHE:
        return _LABEL_CACHE[p_str]
    arr = np.asarray(nib.load(p_str).darrays[0].data, dtype=np.int32).reshape(-1)
    _LABEL_CACHE[p_str] = arr
    return arr


def load_all_subject_data(atlas_mode: str, spec: GridSpec | None = None) -> tuple[list[dict], dict[str, dict]]:
    """Extract individual subject subfield area and proportion data for all cohorts in parallel under specified atlas mode."""
    if spec is None:
        spec = GridSpec("2mm", "36p")
    cohorts = get_diff_cohorts(spec)
    cohort_subject_data: dict[str, dict] = {
        c["name"]: {
            "L": defaultdict(lambda: defaultdict(list)),
            "R": defaultdict(lambda: defaultdict(list)),
            "L_area": defaultdict(lambda: defaultdict(list)),
            "R_area": defaultdict(lambda: defaultdict(list)),
        }
        for c in cohorts
    }

    tasks = []
    for cinfo in cohorts:
        cname = cinfo["name"]
        for hemi in ["L", "R"]:
            hdir = cinfo["group_root"] / f"gradient-spectral/kong2022/{spec.smoothing}/cov_0p80/g1g2g3/hemi_{hemi}"
            man = json.loads((hdir / "group_atlas_manifest.json").read_text(encoding="utf-8"))
            raw_subjects = man.get("cluster_roi_profile_subjects_used") or list(man.get("actual_consumed_input_paths", {}).keys())
            subjects = [str(s).removeprefix("sub-") for s in raw_subjects]

            hipp_root = cinfo.get("hippunfold_root", HIPPUNFOLD_ROOT)

            if atlas_mode == "v1_fixed_hcp151_atlas":
                hdir_hcp = HCP_GROUP_ROOT / f"gradient-spectral/kong2022/{spec.smoothing}/cov_0p80/g1g2g3/hemi_{hemi}"
                labels = np.asarray(np.load(hdir_hcp / "group_labels_final.npy"), dtype=np.int32).reshape(-1)
                support = np.asarray(np.load(hdir_hcp / "group_support_mask.npy"), dtype=bool).reshape(-1)
                for s in subjects:
                    lp = hipp_root / f"sub-{s}/hippunfold/sub-{s}/metric/sub-{s}_hemi-{hemi}_den-512_label-hipp_atlas-{HIPPUNFOLD_ATLAS}_subfields.label.gii"
                    sp = hipp_root / f"sub-{s}/hippunfold/sub-{s}/surf/sub-{s}_hemi-{hemi}_space-T1w_den-512_label-hipp_midthickness.surf.gii"
                    tasks.append((cinfo, cname, s, hemi, labels, support, lp, sp, None))

            elif atlas_mode == "v2_cohort_specific_atlas":
                labels = np.asarray(np.load(hdir / "group_labels_final.npy"), dtype=np.int32).reshape(-1)
                support = np.asarray(np.load(hdir / "group_support_mask.npy"), dtype=bool).reshape(-1)
                for s in subjects:
                    lp = hipp_root / f"sub-{s}/hippunfold/sub-{s}/metric/sub-{s}_hemi-{hemi}_den-512_label-hipp_atlas-{HIPPUNFOLD_ATLAS}_subfields.label.gii"
                    sp = hipp_root / f"sub-{s}/hippunfold/sub-{s}/surf/sub-{s}_hemi-{hemi}_space-T1w_den-512_label-hipp_midthickness.surf.gii"
                    tasks.append((cinfo, cname, s, hemi, labels, support, lp, sp, None))

            elif atlas_mode == "v3_individual_template_matching":
                for s in subjects:
                    lp = hipp_root / f"sub-{s}/hippunfold/sub-{s}/metric/sub-{s}_hemi-{hemi}_den-512_label-hipp_atlas-{HIPPUNFOLD_ATLAS}_subfields.label.gii"
                    sp = hipp_root / f"sub-{s}/hippunfold/sub-{s}/surf/sub-{s}_hemi-{hemi}_space-T1w_den-512_label-hipp_midthickness.surf.gii"
                    if cinfo["id"] == "hcp_7t":
                        indiv_label_path = REPO_ROOT / f"outputs_hcp/work/crossfit_template_matching/subjects/sub-{s}/{spec.smoothing}/hemi_{hemi}/individual_labels_k3.npy"
                    else:
                        indiv_label_path = REPO_ROOT / f"outputs_stanford/work/hcp151_fixed_template_matching/subjects/sub-{s}/matching/{spec.smoothing}/hemi-{hemi}/labels.npy"
                    tasks.append((cinfo, cname, s, hemi, None, None, lp, sp, indiv_label_path))

    def _process_subject(item):
        cinfo, cname, s, hemi, labels, support, lp, sp, indiv_label_path = item
        missing_subject_inputs = [str(path) for path in (lp, sp) if not path.is_file()]
        if missing_subject_inputs:
            raise FileNotFoundError(
                "Missing declared subject anatomical input(s) for "
                f"{cinfo['id']} sub-{s} hemi-{hemi}: "
                + ", ".join(missing_subject_inputs)
            )
        sf = get_subfield_labels(lp)
        w = vertex_area_weights(sp)

        if indiv_label_path is not None:
            if not indiv_label_path.is_file():
                raise FileNotFoundError(
                    "Missing declared individual atlas label input: "
                    f"{indiv_label_path}"
                )
            sub_labels = np.asarray(np.load(str(indiv_label_path)), dtype=np.int32).reshape(-1)
            valid = (sub_labels > 0) & np.isin(sf, np.asarray(SUBFIELD_ORDER, dtype=np.int32))
        else:
            sub_labels = labels
            valid = support & (labels > 0) & np.isin(sf, np.asarray(SUBFIELD_ORDER, dtype=np.int32))

        records = []
        for k in (1, 2, 3):
            k_mask = valid & (sub_labels == int(k))
            denom = float(np.sum(w[k_mask]))
            for s_idx in SUBFIELD_ORDER:
                sf_mask = k_mask & (sf == int(s_idx))
                sf_area = float(np.sum(w[sf_mask]))
                prop = sf_area / denom * 100.0 if denom > 0.0 else 0.0
                records.append((cname, hemi, k, s_idx, prop, sf_area, denom, cinfo, s))
        return records

    print(f"[{atlas_mode}] Extracting subject-level subfield data in parallel ({spec.tag})...")
    with ThreadPoolExecutor(max_workers=8) as ex:
        results_list = list(ex.map(_process_subject, tasks))

    subject_rows = []
    for recs in results_list:
        if not recs:
            continue
        for cname, hemi, k, s_idx, prop, sf_area, denom, cinfo, s in recs:
            cohort_subject_data[cname][hemi][k][s_idx].append(prop)
            cohort_subject_data[cname][f"{hemi}_area"][k][s_idx].append(sf_area)
            subject_rows.append({
                "cohort_id": cinfo["id"],
                "cohort_name": cname,
                "cohort_desc": cinfo["desc"],
                "subject_id": s,
                "hemisphere": hemi,
                "cluster_id": k,
                "cluster_name": SUBREGION_NAMES[k],
                "subfield_index": s_idx,
                "subfield_name": SUBFIELD_NAMES[s_idx],
                "subfield_full_name": SUBFIELD_FULL_NAMES[s_idx],
                "proportion_pct": prop,
                "absolute_area_mm2": sf_area,
                "cluster_valid_area_mm2": denom,
                "atlas_mode": atlas_mode,
                "grid_tag": spec.tag,
                "smoothing": spec.smoothing,
                "variant": spec.variant,
            })

    print(f"[{atlas_mode}] Extracted {len(subject_rows)} subject-subfield measurement records ({spec.tag}).")
    return subject_rows, cohort_subject_data


def compute_whole_hippocampus_summary(
    subject_rows: list[dict], cohorts: list[dict] | None = None
) -> dict[str, dict[str, Any]]:
    """Compute whole-hippocampus composition means and hemisphere-level SDs."""
    if cohorts is None:
        cohorts = COHORTS
    observation_data: dict[tuple[str, str, str], dict[str, Any]] = {}

    for row in subject_rows:
        observation_key = (
            str(row["cohort_name"]),
            str(row["subject_id"]),
            str(row["hemisphere"]),
        )
        bucket = observation_data.setdefault(
            observation_key,
            {
                "area_by_subfield": defaultdict(float),
                "denom_by_cluster": {},
            },
        )

        cluster_id = int(row["cluster_id"])
        bucket["denom_by_cluster"].setdefault(
            cluster_id,
            float(row["cluster_valid_area_mm2"]),
        )
        bucket["area_by_subfield"][str(row["subfield_name"])] += float(
            row["absolute_area_mm2"]
        )

    values_by_cohort: dict[str, dict[str, list[float]]] = {
        c["name"]: {SUBFIELD_NAMES[s_idx]: [] for s_idx in SUBFIELD_ORDER}
        for c in cohorts
    }

    for (cohort_name, _, _), bucket in observation_data.items():
        total_denom = float(sum(bucket["denom_by_cluster"].values()))
        if total_denom <= 0.0:
            continue
        for s_idx in SUBFIELD_ORDER:
            subfield_name = SUBFIELD_NAMES[s_idx]
            subfield_area = float(bucket["area_by_subfield"].get(subfield_name, 0.0))
            whole_pct = (subfield_area / total_denom) * 100.0
            values_by_cohort[cohort_name][subfield_name].append(whole_pct)

    summary: dict[str, dict[str, Any]] = {}
    for cohort_name, values_by_subfield in values_by_cohort.items():
        n_observations = len(values_by_subfield[SUBFIELD_NAMES[SUBFIELD_ORDER[0]]])
        summary[cohort_name] = {
            "n_observations": n_observations,
            "means": {},
            "sds": {},
        }
        for subfield_name, values in values_by_subfield.items():
            arr = np.asarray(values, dtype=np.float64)
            summary[cohort_name]["means"][subfield_name] = (
                float(np.mean(arr)) if arr.size else 0.0
            )
            summary[cohort_name]["sds"][subfield_name] = (
                float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
            )

    return summary


def compute_permutation_pvalue(
    v_case: np.ndarray, v_ctrl: np.ndarray, n_permutations: int = 5000
) -> float:
    """Compute Monte Carlo two-sided permutation p-value."""
    obs_diff = abs(float(np.mean(v_case) - np.mean(v_ctrl)))
    combined = np.concatenate([v_case, v_ctrl])
    n_case = len(v_case)
    
    rng = np.random.default_rng(seed=42)
    shuffles = np.tile(combined, (n_permutations, 1))
    for row in shuffles:
        rng.shuffle(row)
    
    case_means = np.mean(shuffles[:, :n_case], axis=1)
    ctrl_means = np.mean(shuffles[:, n_case:], axis=1)
    perm_diffs = np.abs(case_means - ctrl_means)
    
    count = int(np.sum(perm_diffs >= obs_diff))
    return float((count + 1) / (n_permutations + 1))


def apply_bh_fdr_by_case_cohort(results: list[dict]) -> None:
    """Apply Benjamini-Hochberg FDR independently within each cohort contrast."""
    result_indices_by_cohort: dict[str, list[int]] = defaultdict(list)
    for result_index, result in enumerate(results):
        result_indices_by_cohort[str(result["case_cohort"])].append(result_index)

    for result_indices in result_indices_by_cohort.values():
        p_values = np.asarray(
            [float(results[index]["p_parametric"]) for index in result_indices],
            dtype=np.float64,
        )
        ranked_local_indices = np.argsort(p_values, kind="mergesort")
        n_tests = len(p_values)
        q_values = np.zeros(n_tests, dtype=np.float64)
        min_q = 1.0
        for rank_index in range(n_tests - 1, -1, -1):
            local_index = int(ranked_local_indices[rank_index])
            q = p_values[local_index] * n_tests / (rank_index + 1)
            min_q = min(min_q, q)
            q_values[local_index] = min(min_q, 1.0)

        for local_index, result_index in enumerate(result_indices):
            results[result_index]["fdr_q_parametric"] = float(q_values[local_index])
            results[result_index]["fdr_n_tests"] = n_tests


def compute_statistical_contrasts(
    cohort_subject_data: dict[str, dict],
    atlas_mode: str,
    cohorts: list[dict] | None = None,
    spec: GridSpec | None = None,
) -> list[dict]:
    """Compute comprehensive statistical comparisons for each cohort vs NTHC baseline."""
    if cohorts is None:
        cohorts = COHORTS
    if spec is None:
        spec = GridSpec("2mm", "36p")
    results = []
    ctrl_data = cohort_subject_data["NTHC"]
    case_cohorts = [c for c in cohorts if not c["is_baseline"]]

    for cinfo in case_cohorts:
        cname = cinfo["name"]
        case_data = cohort_subject_data[cname]

        for hemi in ["L", "R"]:
            for k in (1, 2, 3):
                kname = SUBREGION_NAMES[k]
                for s_idx in SUBFIELD_ORDER:
                    sname = SUBFIELD_NAMES[s_idx]

                    v_ctrl = np.asarray(ctrl_data[hemi][k][s_idx], dtype=np.float64)
                    v_case = np.asarray(case_data[hemi][k][s_idx], dtype=np.float64)

                    m_ctrl = float(np.mean(v_ctrl))
                    s_ctrl = float(np.std(v_ctrl, ddof=1)) if len(v_ctrl) > 1 else 0.0
                    m_case = float(np.mean(v_case))
                    s_case = float(np.std(v_case, ddof=1)) if len(v_case) > 1 else 0.0

                    diff = m_case - m_ctrl

                    # Pooled standard deviation & Cohen's d
                    n1, n2 = len(v_ctrl), len(v_case)
                    s_pooled = math.sqrt(((n1 - 1) * s_ctrl**2 + (n2 - 1) * s_case**2) / (n1 + n2 - 2)) if (n1 + n2 > 2) else 1.0
                    cohen_d = diff / s_pooled if s_pooled > 0 else 0.0

                    # Welch's t-test
                    t_stat, p_param = stats.ttest_ind(v_case, v_ctrl, equal_var=False)

                    # Mann-Whitney U test
                    u_stat, p_nonparam = stats.mannwhitneyu(v_case, v_ctrl, alternative="two-sided")

                    # Permutation test
                    p_perm = compute_permutation_pvalue(v_case, v_ctrl, n_permutations=5000)

                    # Absolute Area Analysis
                    v_ctrl_area = np.asarray(ctrl_data[f"{hemi}_area"][k][s_idx], dtype=np.float64)
                    v_case_area = np.asarray(case_data[f"{hemi}_area"][k][s_idx], dtype=np.float64)
                    m_ctrl_area = float(np.mean(v_ctrl_area))
                    m_case_area = float(np.mean(v_case_area))
                    diff_area = m_case_area - m_ctrl_area
                    t_stat_area, p_param_area = stats.ttest_ind(v_case_area, v_ctrl_area, equal_var=False)

                    results.append({
                        "atlas_mode": atlas_mode,
                        "grid_tag": spec.tag,
                        "smoothing": spec.smoothing,
                        "variant": spec.variant,
                        "contrast": f"{cname} vs NTHC",
                        "case_cohort": cname,
                        "case_n": n2,
                        "ctrl_cohort": "NTHC",
                        "ctrl_n": n1,
                        "hemisphere": hemi,
                        "cluster_id": k,
                        "cluster_name": kname,
                        "subfield_index": s_idx,
                        "subfield_name": sname,
                        "subfield_full_name": SUBFIELD_FULL_NAMES[s_idx],
                        "ctrl_mean_pct": m_ctrl,
                        "ctrl_sd_pct": s_ctrl,
                        "case_mean_pct": m_case,
                        "case_sd_pct": s_case,
                        "diff_pct": diff,
                        "cohen_d": cohen_d,
                        "t_stat": float(t_stat),
                        "p_parametric": float(p_param),
                        "u_stat": float(u_stat),
                        "p_nonparametric": float(p_nonparam),
                        "p_permutation": float(p_perm),
                        "ctrl_mean_area_mm2": m_ctrl_area,
                        "case_mean_area_mm2": m_case_area,
                        "diff_area_mm2": diff_area,
                        "p_area": float(p_param_area),
                    })

    # Benjamini-Hochberg FDR correction within each case-cohort contrast.
    apply_bh_fdr_by_case_cohort(results)
    for i, r in enumerate(results):
        r["is_sig_uncorrected"] = bool(r["p_parametric"] < 0.05)
        r["is_sig_fdr"] = bool(r["fdr_q_parametric"] < 0.05)

    return results


def export_analysis_tsvs(
    subject_rows: list[dict],
    stats_results: list[dict],
    file_tag: str,
    spec: GridSpec | None = None,
    atlas_mode: str | None = None,
):
    """Export subject-level long TSV and contrast summary TSV."""
    if spec is None:
        spec = GridSpec("2mm", "36p")
    tag_suffix = spec.tag if (atlas_mode == "v2_cohort_specific_atlas" or atlas_mode == TARGET_ATLAS_MODE) else f"{spec.tag}_{file_tag}"

    long_tsv_path = STATS_DIR / f"stanford_subfield_subject_level_long_{tag_suffix}.tsv"
    with open(long_tsv_path, "w", encoding="utf-8") as f:
        headers = list(subject_rows[0].keys())
        f.write("\t".join(headers) + "\n")
        for r in subject_rows:
            f.write("\t".join(str(r[h]) if not isinstance(r[h], float) else f"{r[h]:.4f}" for h in headers) + "\n")
    print(f"Saved subject-level TSV to {long_tsv_path}")

    stats_tsv_path = STATS_DIR / f"stanford_subfield_group_differences_stats_{tag_suffix}.tsv"
    with open(stats_tsv_path, "w", encoding="utf-8") as f:
        headers = list(stats_results[0].keys())
        f.write("\t".join(headers) + "\n")
        for r in stats_results:
            vals = []
            for h in headers:
                val = r[h]
                if isinstance(val, float):
                    vals.append(f"{val:.4e}" if (abs(val) < 1e-3 and val != 0) else f"{val:.4f}")
                elif isinstance(val, bool):
                    vals.append("TRUE" if val else "FALSE")
                else:
                    vals.append(str(val))
            f.write("\t".join(vals) + "\n")
    print(f"Saved statistical contrast TSV to {stats_tsv_path}")

    if spec.tag == "2mm_36p" and (atlas_mode == "v2_cohort_specific_atlas" or atlas_mode == TARGET_ATLAS_MODE):
        shutil.copy2(long_tsv_path, STATS_DIR / "stanford_subfield_subject_level_long.tsv")
        shutil.copy2(stats_tsv_path, STATS_DIR / "stanford_subfield_group_differences_stats.tsv")


TARGET_GENERATION_MODE = "target"
TARGET_ATLAS_MODE = "v2_cohort_specific_atlas"
DEFAULT_GENERATION_MODE = TARGET_GENERATION_MODE
GROUP_DIFFERENCES_MAIN_TITLE = "HIPPOCAMPAL SUBFIELD COMPOSITION DIFFERENCES VS NTHC"

GROUP_DIFFERENCES_GRID_LEFT = 0.055
GROUP_DIFFERENCES_GRID_RIGHT = 0.975
GROUP_DIFFERENCES_SUPTITLE_Y = 0.975
GROUP_DIFFERENCES_GRID_TOP = 0.895
GROUP_DIFFERENCES_GRID_BOTTOM = 0.055
GROUP_DIFFERENCES_GRID_WSPACE = 0.22
GROUP_DIFFERENCES_GRID_HSPACE = 0.18
GROUP_DIFFERENCES_GRID_WIDTH_RATIOS = (0.31, 0.41, 0.28)
PANEL_A_TITLE_FONTSIZE = 26.0
PANEL_B_TITLE_FONTSIZE = 26.0
PANEL_C_TITLE_FONTSIZE = 26.0
PANEL_B_LEGEND_FONTSIZE = 17.0
PANEL_A_CELL_FONTSIZE = 17.0
PANEL_A_XTICK_FONTSIZE = 16.0
PANEL_A_YTICK_FONTSIZE = 15.0
PANEL_C_YTICK_FONTSIZE = 19.0
PANEL_C_VALUE_FONTSIZE = 17.0
PANEL_C_NEGATIVE_LABEL_INSIDE_THRESHOLD = -4.0
PANEL_C_NON_SIGNIFICANT_BAR_COUNT = 3
PANEL_D_TITLE_FONTSIZE = 24.0
PANEL_D_HEADING_FONTSIZE = 20.0
PANEL_D_BODY_FONTSIZE = 17.0


def _read_exported_tsv(path: Path) -> list[dict]:
    """Read an exported analysis table without rerunning the analysis."""
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = []
        for row in csv.DictReader(f, delimiter="\t"):
            parsed = {}
            for key, value in row.items():
                if value in {"TRUE", "FALSE"}:
                    parsed[key] = value == "TRUE"
                    continue
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows.append(parsed)
    return rows


def load_exported_render_inputs(atlas_mode: str, spec: GridSpec | None = None) -> tuple[list[dict], list[dict]]:
    """Load an existing atlas export for direct figure rendering."""
    if spec is None:
        spec = GridSpec("2mm", "36p")
    file_tag = ATLAS_VERSIONS[atlas_mode]["file_tag"]
    tag_suffix = spec.tag if atlas_mode == TARGET_ATLAS_MODE else f"{spec.tag}_{file_tag}"
    stats_path = STATS_DIR / f"stanford_subfield_group_differences_stats_{tag_suffix}.tsv"
    subject_path = STATS_DIR / f"stanford_subfield_subject_level_long_{tag_suffix}.tsv"
    canonical_missing = [
        path
        for path in (stats_path, subject_path)
        if not path.is_file()
    ]
    if canonical_missing:
        raise FileNotFoundError(
            "Missing canonical grid export(s) for "
            f"atlas_mode={atlas_mode}, grid={spec.tag}: {canonical_missing}"
        )
    if not (stats_path.exists() and subject_path.exists()):
        if atlas_mode == TARGET_ATLAS_MODE:
            stats_path = STATS_DIR / "stanford_subfield_group_differences_stats.tsv"
            subject_path = STATS_DIR / "stanford_subfield_subject_level_long.tsv"
        else:
            stats_path = STATS_DIR / f"stanford_subfield_group_differences_stats_{file_tag}.tsv"
            subject_path = STATS_DIR / f"stanford_subfield_subject_level_long_{file_tag}.tsv"
    if not (stats_path.exists() and subject_path.exists()):
        # Fallback to legacy OUTPUT_DIR if needed
        stats_path = OUTPUT_DIR / stats_path.name
        subject_path = OUTPUT_DIR / subject_path.name
    return _read_exported_tsv(stats_path), _read_exported_tsv(subject_path)


def generation_plan(requested_mode: str) -> list[tuple[str, str]]:
    """Resolve CLI mode into direct-render or explicit recomputation work."""
    if requested_mode == TARGET_GENERATION_MODE:
        return [(TARGET_GENERATION_MODE, TARGET_ATLAS_MODE)]

    mode_by_short_name = {
        "v1": "v1_fixed_hcp151_atlas",
        "v2": "v2_cohort_specific_atlas",
        "v3": "v3_individual_template_matching",
    }
    if requested_mode == "all":
        return [("analysis", mode) for mode in mode_by_short_name.values()]
    return [("analysis", mode_by_short_name[requested_mode])]

def format_relative_difference_label(value: float, sig_stars: str) -> str:
    """Format a subfield composition difference in percent notation."""
    return f"{value:+.1f}% ({sig_stars})"

def relative_bar_label_placement(value: float) -> tuple[float, str, str]:
    """Keep a large negative bar label inside its bar, away from y-tick text."""
    if value <= PANEL_C_NEGATIVE_LABEL_INSIDE_THRESHOLD:
        inside_offset = max(0.25, abs(value) * 0.04)
        return value + inside_offset, "right", "#0f172a"

    offset = max(0.4, abs(value) * 0.08)
    x = value + (offset if value >= 0 else -offset)
    alignment = "left" if value >= 0 else "right"
    return x, alignment, "#0f172a"


def select_relative_bar_items(stats_results: list[dict]) -> list[tuple[str, str, float, str, str]]:
    """Select bilateral candidates plus the largest non-significant effects."""
    target_units = [
        ("NTS Pure MDD", "MDD", "Middle", "Sub", "#7c2d12"),
        ("NTS Pure MDD", "MDD", "Post", "Sub", "#15803d"),
        ("NTS Pure MDD", "MDD", "Ant", "Sub", "#15803d"),
        ("TIS PTSD", "PTSD", "Middle", "CA1", "#4c1d95"),
        ("NTS", "NTS", "Ant", "CA1", "#d97706"),
        ("NTS", "NTS", "Post", "CA1", "#b91c1c"),
        ("TEHC", "TEHC", "Post", "CA1", "#2e7d32"),
        ("HCP 7T", "HCP", "Middle", "Sub", "#0b3c5d"),
    ]

    candidate_items = []
    for cohort_name, display_group, cluster_name, subfield_name, color in target_units:
        matches = [
            row for row in stats_results
            if row["case_cohort"] == cohort_name
            and row["cluster_name"] == cluster_name
            and row["subfield_name"] == subfield_name
        ]
        if len(matches) != 2 or {row["hemisphere"] for row in matches} != {"L", "R"}:
            continue
        q_values = [float(row["fdr_q_parametric"]) for row in matches]
        diff_pct = float(np.mean([float(row["diff_pct"]) for row in matches]))
        is_significant = all(q_value < 0.05 for q_value in q_values)
        q_value = max(q_values)
        sig_stars = "***" if q_value < 0.001 else ("**" if q_value < 0.01 else ("*" if is_significant else "n.s."))
        candidate_items.append(
            (
                display_group,
                f"{cluster_name} {subfield_name}",
                diff_pct,
                sig_stars,
                color,
                is_significant,
            )
        )
    significant_items = [item for item in candidate_items if item[5]]
    nonsignificant_items = sorted(
        (item for item in candidate_items if not item[5]),
        key=lambda item: abs(item[2]),
        reverse=True,
    )[:PANEL_C_NON_SIGNIFICANT_BAR_COUNT]
    selected_items = significant_items + nonsignificant_items
    return [item[:5] for item in selected_items]

def _relative_difference_values(
    stats_results: list[dict],
    case_cohort: str,
    cluster_name: str,
    subfield_name: str,
) -> list[float]:
    return [
        float(row["diff_pct"])
        for row in stats_results
        if row["case_cohort"] == case_cohort
        and row["cluster_name"] == cluster_name
        and row["subfield_name"] == subfield_name
    ]


def _format_relative_difference_range(
    stats_results: list[dict],
    case_cohort: str,
    cluster_name: str,
    subfield_name: str,
) -> str:
    values = _relative_difference_values(stats_results, case_cohort, cluster_name, subfield_name)
    if not values:
        return "n/a"
    return f"{min(values):+.1f}% to {max(values):+.1f}%"

def render_group_differences_figure(
    stats_results: list[dict],
    whole_summary: dict[str, dict[str, Any]],
    atlas_mode: str,
    out_png_path: Path,
    spec: GridSpec | None = None,
    cohorts: list[dict] | None = None,
):
    """Render publication-quality figure with MAXIMUM SIZED, CRYSTAL-CLEAR FONTS."""
    if spec is None:
        spec = GridSpec("2mm", "36p")
    if cohorts is None:
        cohorts = get_diff_cohorts(spec)

    fig = plt.figure(figsize=(42, 25.5), dpi=220)
    # Keep the header compact while leaving enough room for the atlas label.
    top_margin = GROUP_DIFFERENCES_GRID_TOP
    bottom_margin = GROUP_DIFFERENCES_GRID_BOTTOM

    vinfo = ATLAS_VERSIONS[atlas_mode]

    # Main Super Title
    fig.suptitle(
        f"{GROUP_DIFFERENCES_MAIN_TITLE} ({spec.smoothing.upper()}, {spec.variant.upper()})",
        fontsize=28.0,
        fontweight="heavy",
        color="#0a192f",
        y=GROUP_DIFFERENCES_SUPTITLE_Y,
    )

    gs = gridspec.GridSpec(
        nrows=2,
        ncols=3,
        left=GROUP_DIFFERENCES_GRID_LEFT,
        right=GROUP_DIFFERENCES_GRID_RIGHT,
        top=top_margin,
        bottom=bottom_margin,
        wspace=GROUP_DIFFERENCES_GRID_WSPACE,
        hspace=GROUP_DIFFERENCES_GRID_HSPACE,
        height_ratios=[1.18, 1.0],
        width_ratios=GROUP_DIFFERENCES_GRID_WIDTH_RATIOS,
    )

    # ----------------------------------------------------
    # PANEL A: Contrast Heatmap
    # ----------------------------------------------------
    ax_heat = fig.add_subplot(gs[0, :])
    ax_heat.set_title(
        "A. Subfield Composition Differences (Δ% vs NTHC)",
        fontsize=PANEL_A_TITLE_FONTSIZE,
        fontweight="heavy",
        color="#0b2447",
        pad=8,
        loc="left",
    )

    case_names = [c["name"] for c in cohorts if not c["is_baseline"]]
    row_labels = []
    for c in case_names:
        row_labels.append(f"{c} (L)")
        row_labels.append(f"{c} (R)")

    col_keys = []
    col_labels = []
    for k in (1, 2, 3):
        kname = SUBREGION_NAMES[k]
        for s_idx in SUBFIELD_ORDER:
            sname = SUBFIELD_NAMES[s_idx]
            col_keys.append((k, s_idx))
            col_labels.append(f"{kname}\n{sname}")

    mat_diff = np.zeros((len(row_labels), len(col_keys)), dtype=np.float64)
    mat_q = np.zeros((len(row_labels), len(col_keys)), dtype=np.float64)

    for r in stats_results:
        cname = r["case_cohort"]
        hemi = r["hemisphere"]
        k = r["cluster_id"]
        s_idx = r["subfield_index"]

        row_idx = case_names.index(cname) * 2 + (0 if hemi == "L" else 1)
        col_idx = col_keys.index((k, s_idx))

        mat_diff[row_idx, col_idx] = r["diff_pct"]
        mat_q[row_idx, col_idx] = r["fdr_q_parametric"]

    vmax = 10.5
    vmin = -10.5
    im = ax_heat.imshow(mat_diff, cmap="coolwarm", vmin=vmin, vmax=vmax, aspect="auto")

    # Add text annotations with MAXIMUM SIZE TEXT (18.0 pt Extra Bold)
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = mat_diff[i, j]
            q = mat_q[i, j]

            sig_mark = ""
            if q < 0.001:
                sig_mark = " ***"
            elif q < 0.01:
                sig_mark = " **"
            elif q < 0.05:
                sig_mark = " *"

            txt_color = "#ffffff" if abs(val) > 4.2 else "#0f172a"
            txt_str = f"{val:+.1f}%{sig_mark}"
            txt = ax_heat.text(
                j,
                i,
                txt_str,
                ha="center",
                va="center",
                fontsize=PANEL_A_CELL_FONTSIZE,
                fontweight="heavy",
                color=txt_color,
            )
            if q < 0.05 or abs(val) > 3.0:
                txt.set_path_effects([
                    pe.Stroke(linewidth=1.8, foreground="#111111" if txt_color == "#ffffff" else "#ffffff"),
                    pe.Normal(),
                ])

    ax_heat.set_xticks(range(len(col_labels)))
    ax_heat.set_xticklabels(col_labels, fontsize=PANEL_A_XTICK_FONTSIZE, fontweight="heavy", color="#0f172a")
    ax_heat.set_yticks(range(len(row_labels)))
    ax_heat.set_yticklabels(row_labels, fontsize=PANEL_A_YTICK_FONTSIZE, fontweight="heavy", color="#0f172a")

    # Vertical separators between Ant, Middle, Post
    ax_heat.axvline(4.5, color="#0f172a", linewidth=3.5, linestyle="-")
    ax_heat.axvline(9.5, color="#0f172a", linewidth=3.5, linestyle="-")

    # Horizontal separators between cohorts
    for h_sep in np.arange(1.5, len(row_labels) - 1, 2.0):
        ax_heat.axhline(h_sep, color="#475569", linewidth=2.0, linestyle="--")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax_heat, orientation="vertical", pad=0.012, shrink=0.92)
    cbar.set_label("Composition difference (Δ%)", fontsize=15.0, fontweight="heavy", color="#0b2447")
    cbar.ax.tick_params(labelsize=13.0)

    # ----------------------------------------------------
    # PANEL B: Whole-Hippocampus Subfield Differences
    # ----------------------------------------------------
    ax_whole = fig.add_subplot(gs[1, 0])
    ax_whole.set_title(
        "B. Whole-Hippocampus Subfield Proportions",
        fontsize=PANEL_B_TITLE_FONTSIZE,
        fontweight="heavy",
        color="#0b2447",
        pad=8,
        loc="left",
    )

    all_cohort_names = [c["name"] for c in cohorts]
    cohort_colors = [c["color"] for c in cohorts]
    subfields_names = ["Sub", "CA1", "CA2", "CA3", "CA4"]

    whole_means = {
        g_name: [whole_summary[g_name]["means"][sf_name] for sf_name in subfields_names]
        for g_name in all_cohort_names
    }
    whole_sds = {
        g_name: [whole_summary[g_name]["sds"][sf_name] for sf_name in subfields_names]
        for g_name in all_cohort_names
    }

    n_sf = len(subfields_names)
    n_grps = len(all_cohort_names)
    bar_width = 0.82 / float(n_grps)
    x_base = np.arange(n_sf)

    for g_idx, g_name in enumerate(all_cohort_names):
        offsets = x_base + (g_idx - (n_grps - 1) / 2.0) * bar_width
        ax_whole.bar(
            offsets,
            whole_means[g_name],
            yerr=whole_sds[g_name],
            width=bar_width * 0.90,
            color=cohort_colors[g_idx],
            label=g_name,
            error_kw=dict(capsize=3.5, capthick=1.5, elinewidth=1.2, ecolor="#1e293b"),
            edgecolor="#ffffff",
            linewidth=0.8,
            zorder=3,
        )

    ax_whole.set_xticks(x_base)
    ax_whole.set_xticklabels(subfields_names, fontsize=16.0, fontweight="heavy", color="#0f172a")
    ax_whole.set_ylabel("Whole-Hippocampus Proportion (%)", fontsize=16.0, fontweight="heavy", color="#0b2447")
    ax_whole.set_ylim(0, 75)
    ax_whole.tick_params(labelsize=14.0)
    ax_whole.grid(True, axis="y", linestyle="--", alpha=0.65, zorder=1)

    ax_whole.legend(
        loc="upper right",
        fontsize=PANEL_B_LEGEND_FONTSIZE,
        ncol=2,
        frameon=True,
        facecolor="#f8fafc",
        edgecolor="#64748b",
        framealpha=0.95,
        borderpad=0.35,
        handlelength=1.1,
        handletextpad=0.3,
    )

    # ----------------------------------------------------
    # PANEL C: Relative Subfield Composition Differences (Dynamically Computed)
    # ----------------------------------------------------
    ax_area = fig.add_subplot(gs[1, 1])
    ax_area.set_title(
        "C. Bilateral Mean Differences vs NTHC",
        fontsize=PANEL_C_TITLE_FONTSIZE,
        fontweight="heavy",
        color="#0b2447",
        pad=8,
        loc="left",
    )

    # Keep all bilateral significant candidates and the largest non-significant effects.
    notable_items = select_relative_bar_items(stats_results)

    y_pos = np.arange(len(notable_items))
    for idx, (grp, sf_label, d_pct, sig_stars, c_bar) in enumerate(notable_items):
        ax_area.barh(
            y=idx,
            width=d_pct,
            height=0.62,
            color=c_bar,
            edgecolor="#ffffff",
            linewidth=1.6,
            zorder=3,
        )

        lbl_x, ha_align, label_color = relative_bar_label_placement(d_pct)
        label_bbox = (
            {"facecolor": "#ffffff", "edgecolor": "none", "alpha": 0.92, "pad": 0.2}
            if d_pct <= PANEL_C_NEGATIVE_LABEL_INSIDE_THRESHOLD
            else None
        )
        val_text = format_relative_difference_label(d_pct, sig_stars)
        ax_area.text(
            lbl_x,
            idx,
            val_text,
            ha=ha_align,
            va="center",
            fontsize=PANEL_C_VALUE_FONTSIZE,
            fontweight="heavy",
            color=label_color,
            bbox=label_bbox,
            zorder=4,
        )

    ax_area.set_yticks(y_pos)
    ax_area.set_yticklabels(
        [f"{grp} {sf_label}" for grp, sf_label, _, _, _ in notable_items],
        fontsize=PANEL_C_YTICK_FONTSIZE,
        fontweight="heavy",
        color="#0f172a",
    )
    ax_area.set_xlabel("Bilateral Mean Difference (Δ%)", fontsize=16.0, fontweight="heavy", color="#0b2447")
    ax_area.axvline(0, color="#0f172a", linewidth=2.4, zorder=2)

    min_pct = min(d for _, _, d, _, _ in notable_items)
    max_pct = max(d for _, _, d, _, _ in notable_items)
    pad = max(1.0, max(abs(min_pct), abs(max_pct)) * 0.22)
    ax_area.set_xlim(min_pct - pad, max_pct + pad)
    ax_area.tick_params(axis="x", labelsize=14.0)
    ax_area.grid(True, axis="x", linestyle="--", alpha=0.65, zorder=1)
    ax_area.spines[["top", "right"]].set_visible(False)
    pure_mdd_body_sub = _format_relative_difference_range(stats_results, "NTS Pure MDD", "Middle", "Sub")
    pure_mdd_post_sub = _format_relative_difference_range(stats_results, "NTS Pure MDD", "Post", "Sub")

    # ----------------------------------------------------
    # PANEL D: Key Takeaways
    # ----------------------------------------------------
    ax_summary = fig.add_subplot(gs[1, 2])
    ax_summary.axis("off")

    ax_summary.add_patch(
        patches.FancyBboxPatch(
            (0.01, 0.01),
            0.98,
            0.98,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            fc="#f8fafc",
            ec="#475569",
            lw=2.5,
            transform=ax_summary.transAxes,
            zorder=0,
        )
    )

    ax_summary.add_patch(
        patches.FancyBboxPatch(
            (0.03, 0.83),
            0.94,
            0.13,
            boxstyle="round,pad=0.01,rounding_size=0.03",
            fc="#0f172a",
            ec="none",
            transform=ax_summary.transAxes,
            zorder=1,
        )
    )
    ax_summary.text(
        0.50,
        0.895,
        "D. KEY TAKEAWAYS",
        fontsize=PANEL_D_TITLE_FONTSIZE,
        fontweight="heavy",
        ha="center",
        va="center",
        color="#ffffff",
        transform=ax_summary.transAxes,
        zorder=2,
    )

    # Takeaway 1
    hcp_sub_diff = abs(whole_summary["HCP 7T"]["means"]["Sub"] - whole_summary["NTHC"]["means"]["Sub"])
    hcp_ca1_diff = abs(whole_summary["HCP 7T"]["means"]["CA1"] - whole_summary["NTHC"]["means"]["CA1"])
    max_hcp_diff = max(hcp_sub_diff, hcp_ca1_diff)

    ax_summary.text(
        0.05,
        0.74,
        "1. Whole-Hippocampus Stability",
        fontsize=PANEL_D_HEADING_FONTSIZE,
        fontweight="heavy",
        color="#b91c1c",
        transform=ax_summary.transAxes,
    )
    ax_summary.text(
        0.05,
        0.60,
        f"• HCP 7T vs NTHC: Sub/CA1 Δ ≤ {max_hcp_diff:.1f}%.\n• Whole-organ composition is stable.",
        fontsize=PANEL_D_BODY_FONTSIZE,
        fontweight="bold",
        color="#1e293b",
        transform=ax_summary.transAxes,
    )

    # Takeaway 2
    ax_summary.text(
        0.05,
        0.48,
        "2. Local Remodeling",
        fontsize=PANEL_D_HEADING_FONTSIZE,
        fontweight="heavy",
        color="#15803d",
        transform=ax_summary.transAxes,
    )
    ax_summary.text(
        0.05,
        0.33,
        f"• Pure MDD: Middle Sub {pure_mdd_body_sub}; Posterior Sub {pure_mdd_post_sub}.\n• Redistribution across longitudinal subregions.",
        fontsize=PANEL_D_BODY_FONTSIZE,
        fontweight="bold",
        color="#1e293b",
        transform=ax_summary.transAxes,
    )

    # Takeaway 3
    ax_summary.text(
        0.05,
        0.22,
        "3. Clinical Pattern",
        fontsize=PANEL_D_HEADING_FONTSIZE,
        fontweight="heavy",
        color="#0369a1",
        transform=ax_summary.transAxes,
    )
    ax_summary.text(
        0.05,
        0.08,
        "• TEHC remains close to NTHC.\n• Clinical effects are localized.",
        fontsize=PANEL_D_BODY_FONTSIZE,
        fontweight="bold",
        color="#1e293b",
        transform=ax_summary.transAxes,
    )

    out_png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png_path, dpi=220, facecolor="#ffffff", edgecolor="none")
    plt.close(fig)
    print(f"Saved publication-quality figure to {out_png_path}")


def main():
    parser = argparse.ArgumentParser(description="Render the target figure or recompute selected atlas analyses across grid combinations.")
    parser.add_argument(
        "--mode",
        choices=["target", "all", "v1", "v2", "v3"],
        default=DEFAULT_GENERATION_MODE,
        help="target (default) renders from existing exports; all/v1/v2/v3 recompute analyses",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputation of statistics even in target mode",
    )
    args = parser.parse_args()

    # Both 36p grids are processed by default; no single-grid bypass
    grid_specs = tuple(GridSpec.from_tag(t) for t in GRID_VARIANTS)

    def process_spec(spec: GridSpec) -> None:
        cohorts = get_diff_cohorts(spec)
        for generation_type, mode in generation_plan(args.mode):
            t0 = time.time()
            file_tag = ATLAS_VERSIONS[mode]["file_tag"]
            tag_suffix = spec.tag if mode == TARGET_ATLAS_MODE or mode == "v2_cohort_specific_atlas" else f"{spec.tag}_{file_tag}"
            fig_path = OUTPUT_DIR / f"stanford_subfield_group_differences_figure_{tag_suffix}.png"
            stats_tsv_path = STATS_DIR / f"stanford_subfield_group_differences_stats_{tag_suffix}.tsv"
            long_tsv_path = STATS_DIR / f"stanford_subfield_subject_level_long_{tag_suffix}.tsv"

            print(f"\n=======================================================", flush=True)
            print(f"=== Starting Analysis: {ATLAS_VERSIONS[mode]['title_label']} ({spec.tag}) ===", flush=True)
            print(f"=======================================================", flush=True)

            can_use_target = (
                generation_type == TARGET_GENERATION_MODE
                and not args.force_recompute
                and stats_tsv_path.exists()
                and long_tsv_path.exists()
            )

            if can_use_target:
                print(f"Loading existing exports for {spec.tag} direct target rendering...", flush=True)
                stats_results, subject_rows = load_exported_render_inputs(mode, spec=spec)
                whole_summary = compute_whole_hippocampus_summary(subject_rows, cohorts=cohorts)
                print(f"Rendering target figure to {fig_path}...", flush=True)
                render_group_differences_figure(
                    stats_results,
                    whole_summary,
                    mode,
                    fig_path,
                    spec=spec,
                    cohorts=cohorts,
                )
                if spec.tag == "2mm_36p" and mode in (TARGET_ATLAS_MODE, "v2_cohort_specific_atlas"):
                    shutil.copy2(fig_path, OUTPUT_DIR / "stanford_subfield_group_differences_figure.png")
                print(f"=== Completed target ({spec.tag}) in {time.time()-t0:.2f}s ===", flush=True)
                continue

            # Step 1: Load data
            subject_rows, cohort_subject_data = load_all_subject_data(mode, spec=spec)

            # Step 2: Compute dynamic whole-hippocampus summary for Panel B
            whole_summary = compute_whole_hippocampus_summary(subject_rows, cohorts=cohorts)

            # Step 3: Compute statistical contrasts
            print("Computing statistical contrasts (t-test, Mann-Whitney U, 5k permutation, FDR)...", flush=True)
            stats_results = compute_statistical_contrasts(cohort_subject_data, mode, cohorts=cohorts, spec=spec)

            # Step 4: Export TSVs
            export_analysis_tsvs(subject_rows, stats_results, file_tag, spec=spec, atlas_mode=mode)

            # Step 5: Render figure
            print(f"Rendering multi-panel comparison figure to {fig_path}...", flush=True)
            render_group_differences_figure(
                stats_results,
                whole_summary,
                mode,
                fig_path,
                spec=spec,
                cohorts=cohorts,
            )

            # Mirror default files for 2mm_36p (cohort-specific standard)
            if spec.tag == "2mm_36p" and mode in (TARGET_ATLAS_MODE, "v2_cohort_specific_atlas"):
                shutil.copy2(fig_path, OUTPUT_DIR / "stanford_subfield_group_differences_figure.png")

            print(f"=== Completed {mode} ({spec.tag}) in {time.time()-t0:.2f}s ===", flush=True)

    print(f"[RUNNER] Executing Stanford subfield differences across both 36p grid cells...", flush=True)
    with ThreadPoolExecutor(max_workers=len(grid_specs)) as executor:
        futures = [executor.submit(process_spec, s) for s in grid_specs]
        for f in futures:
            f.result()


if __name__ == "__main__":
    main()

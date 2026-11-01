#!/usr/bin/env python3
"""
Render Iowa 3T + Stanford Hippocampal Template Matching Dominant Subregions.

Formal renderer for the dual-dataset dominant-subregion suite. Branch mode
renders the two canonical 2mm/36P figures (standard, ctxreg). Each figure has
two dataset blocks side by side: left = Iowa 3T (ref HC, N=67), right =
Stanford (ref NTHC, N=51; always standard-branch TM, 7 groups). Rows 0-3 pair
Iowa cohorts with Stanford cohorts; rows 4-6 are Stanford-only.

Both stat columns per block are area % of hippocampus (subregion area as %
of the subject's own hemisphere total midthickness area):
  - col 1: age+sex-adjusted OLS (y ~ 1 + group + age + female), uncorrected
  - col 2: two-sided Welch (scipy.stats.ttest_ind, equal_var=False),
    uncorrected (no FDR). This column replaces the former vertex-count
    column; bars/dots in both columns show raw area % values.
Stars: + 0.05<=p<0.10 (trend), * p<0.05, ** p<0.01, *** p<0.001.
Plot: grouped bars (mean +/- SEM) + individual dots + bracket stars.

Grid mode (--grids) keeps the legacy Iowa-only count+area Welch panels and
is unchanged by the dual-dataset extension.
Each branch figure displays 4 Iowa cohorts in the left block:
1. Healthy Controls (HC, N=67)
2. Depression Patients (Patient, N=26, sub-3003 excluded)
3. Pure MDD Subgroup (Pure MDD, N=7)
4. Comorbid MDD Subgroup (Comorbid MDD, N=19)
and 7 Stanford cohorts in the right block:
NTHC (51), TEHC (40), NTS (56), NTS Any MDD (43, nested), NTS Pure MDD (11,
nested), TIS (47), TIS PTSD (35, nested).

Each cohort block contains:
- Left metadata card
- Dominant Subregion Atlas (Anterior/VAN, Middle/SMN, Posterior/DMN)
- Dominance Heatmap (% Agreement among subjects) with 'magma' colormap [33.3%, 80%]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import warnings

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import stats

# Repository paths
REPO_ROOT = Path(__file__).resolve().parents[3]
CORE_DIR = REPO_ROOT / "scripts" / "core_workflow"
VIS_LIB_DIR = REPO_ROOT / "scripts" / "visualize" / "lib"
for p in (CORE_DIR, VIS_LIB_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


from lib.surface_utils import make_network_label_gifti, render_locked_grid

from lib.workbench_scene_batch import (
    save_shape_gii, build_scene_root, capture_scene_root as _capture_scene_root,
)

# Canonical Paths
DEFAULT_36P_ROOT = (
    REPO_ROOT
    / "outputs_iowa_jj_tms"
    / "work"
    / "hcp151_fixed_template_matching"
    / "subjects"
)
DEFAULT_22P_ROOT = (
    REPO_ROOT
    / "outputs_iowa_jj_tms"
    / "work"
    / "hcp151_fixed_template_matching_22p"
    / "subjects"
)
DEFAULT_SCENE = REPO_ROOT / "config" / "wb_locked_native_view_lateral_medial.scene"
DEFAULT_SPEC = (
    REPO_ROOT
    / "outputs_iowa_jj_tms"
    / "inputs"
    / "hippunfold"
    / "dense_corobl_batch"
    / "sub-1001"
    / "hippunfold"
    / "sub-1001"
    / "surf"
    / "sub-1001_den-512_label-hipp_surfaces.spec"
)
DEFAULT_SURF_L = (
    REPO_ROOT
    / "outputs_iowa_jj_tms"
    / "inputs"
    / "kong2022"
    / "templates"
    / "hipp_group_average"
    / "group_average_surfaces"
    / "group-average_hemi-L_space-corobl_den-512_label-hipp_midthickness.surf.gii"
)
DEFAULT_SURF_R = (
    REPO_ROOT
    / "outputs_iowa_jj_tms"
    / "inputs"
    / "kong2022"
    / "templates"
    / "hipp_group_average"
    / "group_average_surfaces"
    / "group-average_hemi-R_space-corobl_den-512_label-hipp_midthickness.surf.gii"
)

# Colors and Semantics
SUBREGION_COLORS = {
    "Anterior/VAN": (220, 92, 255, 255),  # #dc5cff
    "Middle/SMN": (99, 150, 194, 255),  # #6396c2
    "Posterior/DMN": (216, 92, 107, 255),  # #d85c6b
}
SUBREGION_HEX = {
    1: "#dc5cff",
    2: "#6396c2",
    3: "#d85c6b",
}
SUBREGION_NAMES = {
    1: "Anterior / VAN",
    2: "Middle / SMN",
    3: "Posterior / DMN",
}
NETWORK_NAMES = ["Anterior/VAN", "Middle/SMN", "Posterior/DMN"]

TEMPLATE_MATCHING_LABEL_SEMANTICS = {
    1: "Anterior/VAN",
    2: "Middle/SMN",
    3: "Posterior/DMN",
}
SUBREGION_DISPLAY_NAMES = {
    1: "Anterior / VAN",
    2: "Middle / SMN",
    3: "Posterior / DMN",
}

# ---------------- TTEST extension (copy-only) ----------------
WB_COMMAND = "/home/yojiao/FASTANS/workbench/bin_linux64/wb_command"
HIPPUNFOLD_BATCH_ROOT = (
    REPO_ROOT
    / "outputs_iowa_jj_tms"
    / "inputs"
    / "hippunfold"
    / "dense_corobl_batch"
)
TTEST_STAR_LEVELS = ((0.001, "***"), (0.01, "**"), (0.05, "*"), (0.10, "+"))
TTEST_METRICS = ("count", "area_pct")
TTEST_CONTRASTS = ("iowa_pat", "iowa_pure_mdd", "iowa_comorbid_mdd")
TTEST_REF = "iowa_hc"

# ---------------- Dual-dataset branch extension ----------------
# Branch t-test figures (2mm/36P standard + ctxreg) show two dataset blocks
# side by side: left = Iowa 3T (ref HC), right = Stanford (ref NTHC).
# Both stat columns are area % of hippocampus: the former vertex-count
# column is replaced by the age+sex-adjusted OLS column; the unadjusted
# Welch column is kept. Bars/dots always show raw area % values.
DUAL_STAT_COLS = ("area_pct_adj", "area_pct_welch")
STAN_REF = "stan_nthc"
STAN_CONTRASTS = (
    "stan_tehc",
    "stan_nts",
    "stan_nts_any_mdd",
    "stan_nts_pure_mdd",
    "stan_tis",
    "stan_tis_ptsd",
)
STANFORD_TM_ROOT = (
    REPO_ROOT
    / "outputs_stanford"
    / "work"
    / "hcp151_fixed_template_matching"
    / "subjects"
)
STANFORD_HIPPUNFOLD_BATCH_ROOT = (
    REPO_ROOT
    / "outputs_stanford"
    / "inputs"
    / "hippunfold"
    / "dense_corobl_batch"
)
STANFORD_MANIFEST_DIR = (
    REPO_ROOT
    / "outputs_stanford"
    / "inputs"
    / "kong2022"
    / "manifests"
)
STANFORD_CC_DATABASE = REPO_ROOT / "data" / "CC_RC_Master_Database_10.23.18_Data.csv"
IOWA_SHARE_WORKBOOK = (
    REPO_ROOT / "outputs_iowa_jj_tms" / "TMSfMRI_DataSharing.xlsx"
)
STANFORD_SPEC = (
    REPO_ROOT
    / "outputs_stanford"
    / "inputs"
    / "hippunfold"
    / "dense_corobl_batch"
    / "sub-1001"
    / "hippunfold"
    / "sub-1001"
    / "surf"
    / "sub-1001_den-512_label-hipp_surfaces.spec"
)
STANFORD_SURF_L = (
    REPO_ROOT
    / "outputs_stanford"
    / "inputs"
    / "kong2022"
    / "templates"
    / "hipp_group_average"
    / "group_average_surfaces"
    / "group-average_hemi-L_space-corobl_den-512_label-hipp_midthickness.surf.gii"
)
STANFORD_SURF_R = (
    REPO_ROOT
    / "outputs_stanford"
    / "inputs"
    / "kong2022"
    / "templates"
    / "hipp_group_average"
    / "group_average_surfaces"
    / "group-average_hemi-R_space-corobl_den-512_label-hipp_midthickness.surf.gii"
)

STANFORD_COHORTS_DEF = [
    {
        "key": "stan_nthc",
        "name": "Non-Trauma Healthy Controls",
        "badge": "NTHC",
        "badge_color": "#7c2d12",
        "bg_color": "#fff7ed",
        "desc": "Stanford reference (standard-branch TM)",
        "roster_file": "stanford_nthc_1xxx_qc_pass_t1only_51.txt",
        "subjects": [],
    },
    {
        "key": "stan_tehc",
        "name": "Trauma-Exposed Healthy Controls",
        "badge": "TEHC",
        "badge_color": "#9a3412",
        "bg_color": "#fff7ed",
        "desc": "Trauma-exposed controls",
        "roster_file": "stanford_tehc_2xxx_qc_pass_t1only_40.txt",
        "subjects": [],
    },
    {
        "key": "stan_nts",
        "name": "Non-Trauma Symptomatic",
        "badge": "NTS",
        "badge_color": "#0f766e",
        "bg_color": "#f0fdfa",
        "desc": "Non-trauma symptomatic",
        "roster_file": "stanford_nts_3xxx_qc_pass_t1only_56.txt",
        "subjects": [],
    },
    {
        "key": "stan_nts_any_mdd",
        "name": "NTS Any MDD",
        "badge": "NTS Any MDD",
        "badge_color": "#115e59",
        "bg_color": "#f0fdfa",
        "desc": "Non-trauma MDD (all, nested in NTS)",
        "roster_file": "stanford_nts_any_mdd_3xxx_qc_pass_t1only_43.txt",
        "subjects": [],
    },
    {
        "key": "stan_nts_pure_mdd",
        "name": "NTS Pure MDD",
        "badge": "NTS Pure MDD",
        "badge_color": "#854d0e",
        "bg_color": "#fffbeb",
        "desc": "Non-trauma pure MDD (nested in NTS)",
        "roster_file": "stanford_nts_pure_mdd_3xxx_qc_pass_t1only_11.txt",
        "subjects": [],
    },
    {
        "key": "stan_tis",
        "name": "Trauma-Exposed Symptomatic",
        "badge": "TIS",
        "badge_color": "#1d4ed8",
        "bg_color": "#eff6ff",
        "desc": "Trauma-exposed symptomatic",
        "roster_file": "stanford_tis_4xxx_qc_pass_t1only_47.txt",
        "subjects": [],
    },
    {
        "key": "stan_tis_ptsd",
        "name": "TIS PTSD",
        "badge": "TIS PTSD",
        "badge_color": "#6d28d9",
        "bg_color": "#f5f3ff",
        "desc": "TIS with current PTSD (nested in TIS)",
        "roster_file": "stanford_tis_ptsd_4xxx_current_ptsd_qc_pass_t1only_35.txt",
        "subjects": [],
    },
]
STANFORD_EXPECTED_N = {
    "stan_nthc": 51,
    "stan_tehc": 40,
    "stan_nts": 56,
    "stan_nts_any_mdd": 43,
    "stan_nts_pure_mdd": 11,
    "stan_tis": 47,
    "stan_tis_ptsd": 35,
}


def load_stanford_roster(fname: str) -> list[str]:
    lines = (STANFORD_MANIFEST_DIR / fname).read_text(encoding="utf-8").splitlines()
    return [f"sub-{l.strip()}" for l in lines if l.strip() and not l.startswith("#")]


def stanford_cohorts() -> list[dict[str, Any]]:
    cohorts = []
    for cohort in STANFORD_COHORTS_DEF:
        subjects = load_stanford_roster(cohort["roster_file"])
        expected = STANFORD_EXPECTED_N[cohort["key"]]
        if len(subjects) != expected or len(set(subjects)) != expected:
            raise ValueError(
                f"Stanford roster count/duplicate mismatch: {cohort['roster_file']} "
                f"(got {len(subjects)}, expected {expected})"
            )
        cohorts.append({**cohort, "subjects": subjects})
    return cohorts


def stars_for_p(p: float) -> str:
    try:
        p = float(p)
    except (TypeError, ValueError):
        return "n.s."
    if not np.isfinite(p):
        return "n.s."
    for alpha, star in TTEST_STAR_LEVELS:
        if p < alpha:
            return star
    return "n.s."


def _normalize_iowa_subject(value: object) -> str:
    text = str(value).strip()
    if text.upper().startswith("JJ_TMS_"):
        text = text[len("JJ_TMS_") :]
    text = text.removeprefix("sub-")
    if not text.isdigit():
        raise ValueError(f"Unrecognized Iowa subject ID format: {value!r}")
    return f"sub-{text}"


def load_iowa_age_sex(repo_root: Path = REPO_ROOT) -> dict[str, tuple[float, float]]:
    """Load Iowa age + female indicator keyed by sub-xxxx. Fail closed."""
    import pandas as pd

    workbook = Path(repo_root).resolve() / "outputs_iowa_jj_tms" / "TMSfMRI_DataSharing.xlsx"
    if not workbook.is_file():
        raise FileNotFoundError(f"Missing Iowa workbook: {workbook}")
    raw = pd.read_excel(workbook, sheet_name="data")
    demo: dict[str, tuple[float, float]] = {}
    for _, row in raw.iterrows():
        sub = _normalize_iowa_subject(row["record_subject_id"])
        age = float(row["pt_age"])
        sex = int(row["pt_sex"])
        if not np.isfinite(age):
            raise ValueError(f"Missing Iowa age: {sub}")
        if sex not in (1, 2):
            raise ValueError(f"Unsupported Iowa sex coding for {sub}: {sex!r}")
        if sub in demo:
            raise ValueError(f"Duplicate Iowa workbook subject: {sub}")
        demo[sub] = (age, 1.0 if sex == 2 else 0.0)
    return demo


def load_stanford_age_sex(
    repo_root: Path = REPO_ROOT,
) -> dict[str, tuple[float, float]]:
    """Load Stanford age + female indicator keyed by sub-xxxx. Fail closed.

    Sex coding: female = 1.0 iff Gender == 2, else 0.0. The single Gender == 3
    record (TIS/TIS PTSD overlap) is coded as non-female and documented in
    figure/CSV provenance rather than silently dropped.
    """
    import pandas as pd

    db = Path(repo_root).resolve() / "data" / "CC_RC_Master_Database_10.23.18_Data.csv"
    if not db.is_file():
        raise FileNotFoundError(f"Missing Stanford database: {db}")
    raw = pd.read_csv(db, low_memory=False)
    demo: dict[str, tuple[float, float]] = {}
    for _, row in raw.iterrows():
        sub = f"sub-{int(row['CC_Post_Intake_ID']):04d}"
        age = float(row["Intake_Age"])
        gender = int(row["Gender"])
        if not np.isfinite(age):
            raise ValueError(f"Missing Stanford age: {sub}")
        if gender not in (1, 2, 3):
            raise ValueError(f"Unsupported Stanford gender coding for {sub}: {gender!r}")
        if sub in demo:
            raise ValueError(f"Duplicate Stanford database subject: {sub}")
        demo[sub] = (age, 1.0 if gender == 2 else 0.0)
    return demo


def ols_group_test(
    case_vals: np.ndarray,
    ctrl_vals: np.ndarray,
    case_cov: np.ndarray,
    ctrl_cov: np.ndarray,
) -> dict[str, float]:
    """Two-sided OLS group test: y ~ 1 + group + age + female (numpy/scipy only).

    Covariates are pooled-mean centered (leaves the group coefficient
    unchanged, improves conditioning). Returns the group-coefficient
    t-statistic and two-sided p-value. Fail closed on rank deficiency.
    """
    y = np.concatenate(
        [np.asarray(ctrl_vals, dtype=np.float64), np.asarray(case_vals, dtype=np.float64)]
    )
    g = np.concatenate(
        [np.zeros(len(ctrl_vals)), np.ones(len(case_vals))]
    ).astype(np.float64)
    c = np.concatenate(
        [np.asarray(ctrl_cov, dtype=np.float64), np.asarray(case_cov, dtype=np.float64)]
    )
    if c.ndim != 2 or c.shape[1] != 2:
        raise ValueError(f"Expected (n, 2) covariates, got {c.shape}")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(c))):
        raise ValueError("Non-finite values in OLS inputs")
    c_centered = c - np.mean(c, axis=0, keepdims=True)
    x = np.column_stack([np.ones(len(y)), g, c_centered])
    n, p = x.shape
    dof = n - p
    if dof < 1:
        raise ValueError(f"Not enough degrees of freedom for OLS: n={n}, p={p}")
    beta, residuals, rank, sv = np.linalg.lstsq(x, y, rcond=None)
    if rank < p or float(np.min(sv)) <= 1e-10:
        raise ValueError("Rank-deficient OLS design matrix")
    resid = y - x @ beta
    s2 = float(resid @ resid / dof)
    if not np.isfinite(s2) or s2 <= 0:
        raise ValueError("Non-positive OLS residual variance")
    xtx_inv = np.linalg.inv(x.T @ x)
    se = float(np.sqrt(xtx_inv[1, 1] * s2))
    if not np.isfinite(se) or se <= 0:
        raise ValueError("Non-finite OLS standard error")
    t_stat = float(beta[1] / se)
    p_val = float(2.0 * stats.t.sf(abs(t_stat), dof))
    if not np.isfinite(t_stat) or not np.isfinite(p_val):
        raise ValueError("Non-finite OLS group test result")
    return {"t": t_stat, "p": p_val}


def area_pct_dual_tests(
    per_cohort: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
    demo: dict[str, tuple[float, float]],
    ref_key: str,
    contrasts: tuple[str, ...],
) -> dict[str, dict[str, dict[int, dict[str, float]]]]:
    """Welch + age/sex-adjusted OLS tests on area_pct per contrast x hemi x k.

    Result entries carry both tests; stars use TTEST_STAR_LEVELS for each.
    Missing demographics fail closed (no silent complete-case drop except the
    explicit error below naming the subject).
    """
    if ref_key not in per_cohort:
        raise ValueError(f"Missing reference cohort: {ref_key}")
    results: dict[str, dict[str, dict[int, dict[str, float]]]] = {}
    for contrast in contrasts:
        if contrast not in per_cohort:
            raise ValueError(f"Missing contrast cohort: {contrast}")
        results[contrast] = {}
        for hemi in ("L", "R"):
            results[contrast][hemi] = {}
            ref_subs = list(per_cohort[ref_key])
            cur_subs = list(per_cohort[contrast])
            for sub in ref_subs + cur_subs:
                if sub not in demo:
                    raise ValueError(f"Missing demographics for {sub}")
            ref_cov = np.array([demo[s] for s in ref_subs], dtype=np.float64)
            cur_cov = np.array([demo[s] for s in cur_subs], dtype=np.float64)
            ref = np.asarray(
                [per_cohort[ref_key][s]["area_pct"][hemi] for s in ref_subs],
                dtype=np.float64,
            )
            cur = np.asarray(
                [per_cohort[contrast][s]["area_pct"][hemi] for s in cur_subs],
                dtype=np.float64,
            )
            for k in (1, 2, 3):
                a, b = cur[:, k - 1], ref[:, k - 1]
                if len(a) < 2 or len(b) < 2:
                    raise ValueError(
                        f"Need N>=2 for tests: {contrast} {hemi} k={k} ({len(a)}, {len(b)})"
                    )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    res = stats.ttest_ind(a, b, equal_var=False)
                w_t, w_p = float(res.statistic), float(res.pvalue)
                if not np.isfinite(w_t) or not np.isfinite(w_p):
                    raise ValueError(
                        f"Non-finite Welch result: {contrast} area_pct {hemi} k={k}"
                    )
                adj = ols_group_test(a, b, cur_cov, ref_cov)
                sa = float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
                sb = float(np.std(b, ddof=1)) if len(b) > 1 else 0.0
                denom = float(np.sqrt((sa * sa + sb * sb) / 2.0))
                diff = float(np.mean(a) - np.mean(b))
                cohens_d = diff / denom if denom > 0 else 0.0
                results[contrast][hemi][k] = {
                    "welch_t": w_t,
                    "welch_p": w_p,
                    "welch_stars": stars_for_p(w_p),
                    "adj_t": adj["t"],
                    "adj_p": adj["p"],
                    "adj_stars": stars_for_p(adj["p"]),
                    "case_n": float(len(a)),
                    "ctrl_n": float(len(b)),
                    "case_mean": float(np.mean(a)),
                    "ctrl_mean": float(np.mean(b)),
                    "case_sd": sa,
                    "ctrl_sd": sb,
                    "diff": diff,
                    "d": cohens_d,
                }
    return results


def native_midthickness_path(
    subject: str,
    hemi: str,
    hippunfold_batch_root: Path = HIPPUNFOLD_BATCH_ROOT,
) -> Path:
    if hemi not in ("L", "R"):
        raise ValueError(f"Invalid hemi: {hemi}")
    return (
        hippunfold_batch_root
        / subject
        / "hippunfold"
        / subject
        / "surf"
        / f"{subject}_hemi-{hemi}_space-T1w_den-512_label-hipp_midthickness.surf.gii"
    )


def numpy_vertex_areas(surface_path: Path) -> np.ndarray:
    img = nib.load(str(surface_path))
    coords = np.asarray(img.agg_data("pointset"), dtype=np.float64)
    faces = np.asarray(img.agg_data("triangle"), dtype=np.int64)
    if coords.shape[0] != 512:
        raise ValueError(f"Expected 512 vertices in {surface_path}, got {coords.shape}")
    tri = coords[faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )
    weights = np.zeros(coords.shape[0], dtype=np.float64)
    for corner in range(3):
        np.add.at(weights, faces[:, corner], areas / 3.0)
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError(f"Invalid numpy vertex areas: {surface_path}")
    return weights


def wb_vertex_areas(
    surface_path: Path, cache_dir: Path, cache_namespace: str = ""
) -> np.ndarray:
    """Authoritative vertex areas via wb_command, cross-checked against numpy.

    Mirrors scripts/visualize/cortex_network_area/prepare_iowa_surfaces.py.
    cache_namespace isolates datasets sharing subject IDs (Iowa/Stanford).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"{surface_path.parent.parent.parent.name}_{surface_path.name}".replace(
        "/", "_"
    )
    if cache_namespace:
        key = f"{cache_namespace}_{key}"
    cached = cache_dir / f"{key}.npy"
    if cached.is_file():
        arr = np.load(str(cached))
        if arr.shape != (512,):
            raise ValueError(f"Stale area cache shape: {cached}")
        return arr
    wb = Path(WB_COMMAND)
    if not wb.is_file():
        raise FileNotFoundError(f"Missing wb_command for native area: {wb}")
    if not surface_path.is_file():
        raise FileNotFoundError(f"Missing native midthickness surface: {surface_path}")
    scratch = REPO_ROOT / "tmp"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="native_vertex_areas_", dir=scratch) as temporary:
        metric_path = Path(temporary) / "areas.shape.gii"
        subprocess.run(
            [str(wb), "-surface-vertex-areas", str(surface_path), str(metric_path)],
            check=True,
        )
        expected = np.asarray(
            nib.load(str(metric_path)).darrays[0].data, dtype=np.float64
        ).reshape(-1)
    if expected.shape != (512,):
        raise ValueError(f"wb_command returned shape {expected.shape}: {surface_path}")
    actual = numpy_vertex_areas(surface_path)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    np.save(str(cached), expected)
    return expected


def compute_subject_counts_areas(
    smoothing: str,
    subjects: list[str],
    subjects_root: Path,
    cache_dir: Path,
    signal_branch: str | None = None,
    hippunfold_batch_root: Path = HIPPUNFOLD_BATCH_ROOT,
    cache_namespace: str = "",
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Per-subject, per-hemisphere vertex counts and area percentages for k=1..3.

    Counts come from matching labels.npy; areas weight the same labels by
    native midthickness vertex areas (wb_command, numpy cross-checked) and are
    then expressed as a percentage of the subject's OWN hemisphere total
    midthickness area (sum over all 512 vertices, including any label-0
    vertices), so between-subject whole-hippocampus size scaling is removed.
    Hemispheres are kept separate (no bilateral summation).
    """
    out: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for s in subjects:
        out[s] = {"count": {}, "area_pct": {}}
        for hemi in ("L", "R"):
            if signal_branch is None:
                f_lbl = (
                    subjects_root
                    / s
                    / "matching"
                    / smoothing
                    / f"hemi-{hemi}"
                    / "labels.npy"
                )
            else:
                manifest_path = (
                    subjects_root
                    / s
                    / "matching"
                    / smoothing
                    / f"hemi-{hemi}"
                    / "combo_manifest.json"
                )
                if not manifest_path.is_file():
                    raise FileNotFoundError(
                        f"Missing matching manifest: {manifest_path}"
                    )
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                f_lbl = _declared_output_path(manifest_path, manifest, "labels")
            if not Path(f_lbl).is_file():
                raise FileNotFoundError(
                    f"Missing labels for {s} {smoothing} hemi-{hemi}: {f_lbl}"
                )
            labels = np.asarray(np.load(str(f_lbl)), dtype=np.int32).reshape(-1)
            if labels.shape != (512,):
                raise ValueError(f"Expected 512 labels for {f_lbl}, got {labels.shape}")
            if not set(np.unique(labels)).issubset({0, 1, 2, 3}):
                raise ValueError(
                    f"Unexpected label values in {f_lbl}: {np.unique(labels)}"
                )
            w = wb_vertex_areas(
                native_midthickness_path(s, hemi, hippunfold_batch_root),
                cache_dir,
                cache_namespace,
            )
            hemi_total = float(np.sum(w))
            if not np.isfinite(hemi_total) or hemi_total <= 0:
                raise ValueError(
                    f"Invalid hemisphere total area for {s} hemi-{hemi}: {hemi_total}"
                )
            counts = np.zeros(3, dtype=np.float64)
            areas = np.zeros(3, dtype=np.float64)
            for k in (1, 2, 3):
                mask = labels == k
                counts[k - 1] = float(np.sum(mask))
                areas[k - 1] = float(np.sum(w[mask]))
            out[s]["count"][hemi] = counts
            out[s]["area_pct"][hemi] = 100.0 * areas / hemi_total
    return out


def welch_vs_hc(
    per_cohort: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
) -> dict[str, dict[str, dict[str, dict[int, dict[str, float]]]]]:
    """Two-sided Welch t-test per contrast x metric x hemisphere x subregion. No FDR.

    Effect size is Welch-consistent Cohen's d: ``d = (mean_case - mean_ctrl) /
    sqrt((sa**2 + sb**2) / 2)`` (unweighted pooled SD, matching the unequal-
    variance test; sign follows case-minus-ctrl).
    """
    if TTEST_REF not in per_cohort:
        raise ValueError(f"Missing HC reference: {TTEST_REF}")
    results: dict[str, dict[str, dict[str, dict[int, dict[str, float]]]]] = {}
    for contrast in TTEST_CONTRASTS:
        if contrast not in per_cohort:
            raise ValueError(f"Missing contrast cohort: {contrast}")
        results[contrast] = {}
        for metric in TTEST_METRICS:
            results[contrast][metric] = {}
            for hemi in ("L", "R"):
                results[contrast][metric][hemi] = {}
                ref = np.asarray(
                    [
                        per_cohort[TTEST_REF][s][metric][hemi]
                        for s in per_cohort[TTEST_REF]
                    ],
                    dtype=np.float64,
                )
                cur = np.asarray(
                    [
                        per_cohort[contrast][s][metric][hemi]
                        for s in per_cohort[contrast]
                    ],
                    dtype=np.float64,
                )
                for k in (1, 2, 3):
                    a, b = cur[:, k - 1], ref[:, k - 1]
                    if len(a) < 2 or len(b) < 2:
                        raise ValueError(
                            f"Need N>=2 for Welch: {contrast} {hemi} k={k} ({len(a)}, {len(b)})"
                        )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        res = stats.ttest_ind(a, b, equal_var=False)
                    t_stat, p_val = float(res.statistic), float(res.pvalue)
                    if not np.isfinite(t_stat) or not np.isfinite(p_val):
                        raise ValueError(
                            f"Non-finite Welch result: {contrast} {metric} {hemi} k={k}"
                        )
                    sa = float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
                    sb = float(np.std(b, ddof=1)) if len(b) > 1 else 0.0
                    denom = float(np.sqrt((sa * sa + sb * sb) / 2.0))
                    diff = float(np.mean(a) - np.mean(b))
                    cohens_d = diff / denom if denom > 0 else 0.0
                    results[contrast][metric][hemi][k] = {
                        "t": t_stat,
                        "p": p_val,
                        "stars": stars_for_p(p_val),
                        "case_n": float(len(a)),
                        "ctrl_n": float(len(b)),
                        "case_mean": float(np.mean(a)),
                        "ctrl_mean": float(np.mean(b)),
                        "case_sd": sa,
                        "ctrl_sd": sb,
                        "diff": diff,
                        "d": cohens_d,
                    }
    return results


def _lighten_hex(hex_color: str, amount: float = 0.45) -> str:
    """Mix a hex color toward white for softer bar fills (keeps hue identity)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    r = int(round(r + (255 - r) * amount))
    g = int(round(g + (255 - g) * amount))
    b = int(round(b + (255 - b) * amount))
    return f"#{r:02x}{g:02x}{b:02x}"


TTEST_HC_BAR = "#93a5b8"
TTEST_SHORT_LABELS = ("Ant/L", "Mid/L", "Post/L", "Ant/R", "Mid/R", "Post/R")


def render_ttest_panel(
    contrast_key: str,
    contrast_badge: str,
    contrast_color: str,
    metric: str,
    per_cohort: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
    ttest: dict[str, dict[str, dict[str, dict[int, dict[str, float]]]]],
    out_path: Path,
    subregion_names: dict[int, str] | None = None,
) -> Path:
    """Grouped bars (mean +/- SEM) + dots + bracket stars, L/R hippocampus in one panel.

    Six clusters (3 subregions x 2 hemispheres), two bars each: HC vs cohort.
    Legend maps bar color to group explicitly.
    """
    del subregion_names
    ref_n = len(per_cohort[TTEST_REF])
    cur_n = len(per_cohort[contrast_key])
    ylabel = (
        "vertices (per hemisphere)"
        if metric == "count"
        else "area % of hippocampus (per hemisphere)"
    )
    title = (
        "Vertex count vs HC — Left / Right hippocampus"
        if metric == "count"
        else "Area % of hippocampus vs HC — Left / Right"
    )
    cohort_fill = _lighten_hex(contrast_color, 0.35)

    order = [("L", 1), ("L", 2), ("L", 3), ("R", 1), ("R", 2), ("R", 3)]
    fig, ax = plt.subplots(figsize=(6.0, 3.9), dpi=100)
    x = np.arange(len(order))
    width = 0.36
    rng = np.random.default_rng(42)
    for i, (hemi, k) in enumerate(order):
        a = np.asarray(
            [
                per_cohort[contrast_key][s][metric][hemi][k - 1]
                for s in per_cohort[contrast_key]
            ],
            dtype=np.float64,
        )
        b = np.asarray(
            [
                per_cohort[TTEST_REF][s][metric][hemi][k - 1]
                for s in per_cohort[TTEST_REF]
            ],
            dtype=np.float64,
        )
        for vals, xc, color in (
            (b, x[i] - width / 2, TTEST_HC_BAR),
            (a, x[i] + width / 2, cohort_fill),
        ):
            mean = float(np.mean(vals))
            sem = (
                float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                if len(vals) > 1
                else 0.0
            )
            ax.bar(
                xc,
                mean,
                width=width,
                color=color,
                edgecolor="white",
                linewidth=1.0,
                zorder=2,
            )
            ax.errorbar(
                [xc],
                [mean],
                yerr=[sem],
                color="#334155",
                capsize=3,
                linewidth=1.2,
                zorder=4,
            )
            jx = rng.normal(loc=xc, scale=width * 0.09, size=len(vals))
            ax.scatter(
                jx, vals, s=14, color="#1e293b", alpha=0.5, linewidths=0, zorder=5
            )
        st = ttest[contrast_key][metric][hemi][k]
        top = max(float(np.max(a)), float(np.max(b)))
        sem_a = float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0
        sem_b = float(np.std(b, ddof=1) / np.sqrt(len(b))) if len(b) > 1 else 0.0
        y_top = top + max(sem_a, sem_b) * 1.2
        span = max(
            top * 0.02,
            (float(max(np.max(a), np.max(b))) - float(min(np.min(a), np.min(b))) + 1e-9)
            * 0.03,
        )
        x1, x2 = x[i] - width / 2, x[i] + width / 2
        ax.plot(
            [x1, x1, x2, x2],
            [y_top, y_top + span, y_top + span, y_top],
            color="black",
            linewidth=1.0,
        )
        label = st["stars"] if st["stars"] != "n.s." else "n.s."
        ax.text(
            x[i],
            y_top + span * 1.15,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels(list(TTEST_SHORT_LABELS), fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(
        f"{title}\n{contrast_badge} (N={cur_n}) vs HC (N={ref_n}) — Welch, uncorrected",
        fontsize=9,
        fontweight="bold",
    )
    import matplotlib.patches as _mpatches

    ax.legend(
        handles=[
            _mpatches.Patch(
                facecolor=TTEST_HC_BAR, edgecolor="white", label=f"HC (N={ref_n})"
            ),
            _mpatches.Patch(
                facecolor=cohort_fill,
                edgecolor="white",
                label=f"{contrast_badge} (N={cur_n})",
            ),
        ],
        fontsize=8,
        loc="upper right",
        framealpha=0.95,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def render_dual_areapct_panel(
    *,
    dataset_label: str,
    ref_key: str,
    ref_badge: str,
    contrast_key: str,
    contrast_badge: str,
    contrast_color: str,
    per_cohort: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
    tests: dict[str, dict[str, dict[int, dict[str, float]]]],
    mode: str,
    out_path: Path,
) -> Path:
    """Area-% panel for the dual-dataset branch figures (raw bars/dots).

    mode == "welch": unadjusted two-sided Welch stars.
    mode == "adj": age+sex-adjusted OLS stars (bars/dots stay raw so the two
    columns remain visually comparable; the test source is in the title).
    """
    if mode not in ("welch", "adj"):
        raise ValueError(f"Unsupported dual panel mode: {mode}")
    if contrast_key not in per_cohort or ref_key not in per_cohort:
        raise ValueError(f"Missing cohorts for panel: {contrast_key} vs {ref_key}")
    ref_n = len(per_cohort[ref_key])
    cur_n = len(per_cohort[contrast_key])
    ylabel = "area % of hippocampus (per hemisphere)"
    test_name = "Welch, uncorrected" if mode == "welch" else "age+sex adjusted (OLS)"
    title = f"Area % of hippocampus vs {ref_badge} — Left / Right"
    cohort_fill = _lighten_hex(contrast_color, 0.35)

    order = [("L", 1), ("L", 2), ("L", 3), ("R", 1), ("R", 2), ("R", 3)]
    fig, ax = plt.subplots(figsize=(6.0, 3.9), dpi=100)
    x = np.arange(len(order))
    width = 0.36
    rng = np.random.default_rng(42)
    for i, (hemi, k) in enumerate(order):
        a = np.asarray(
            [
                per_cohort[contrast_key][s]["area_pct"][hemi][k - 1]
                for s in per_cohort[contrast_key]
            ],
            dtype=np.float64,
        )
        b = np.asarray(
            [per_cohort[ref_key][s]["area_pct"][hemi][k - 1] for s in per_cohort[ref_key]],
            dtype=np.float64,
        )
        for vals, xc, color in (
            (b, x[i] - width / 2, TTEST_HC_BAR),
            (a, x[i] + width / 2, cohort_fill),
        ):
            mean = float(np.mean(vals))
            sem = (
                float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                if len(vals) > 1
                else 0.0
            )
            ax.bar(
                xc,
                mean,
                width=width,
                color=color,
                edgecolor="white",
                linewidth=1.0,
                zorder=2,
            )
            ax.errorbar(
                [xc],
                [mean],
                yerr=[sem],
                color="#334155",
                capsize=3,
                linewidth=1.2,
                zorder=4,
            )
            jx = rng.normal(loc=xc, scale=width * 0.09, size=len(vals))
            ax.scatter(
                jx, vals, s=14, color="#1e293b", alpha=0.5, linewidths=0, zorder=5
            )
        st = tests[contrast_key][hemi][k]
        star_key = "welch_stars" if mode == "welch" else "adj_stars"
        top = max(float(np.max(a)), float(np.max(b)))
        sem_a = float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0
        sem_b = float(np.std(b, ddof=1) / np.sqrt(len(b))) if len(b) > 1 else 0.0
        y_top = top + max(sem_a, sem_b) * 1.2
        span = max(
            top * 0.02,
            (float(max(np.max(a), np.max(b))) - float(min(np.min(a), np.min(b))) + 1e-9)
            * 0.03,
        )
        x1, x2 = x[i] - width / 2, x[i] + width / 2
        ax.plot(
            [x1, x1, x2, x2],
            [y_top, y_top + span, y_top + span, y_top],
            color="black",
            linewidth=1.0,
        )
        label = st[star_key] if st[star_key] != "n.s." else "n.s."
        ax.text(
            x[i],
            y_top + span * 1.15,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels(list(TTEST_SHORT_LABELS), fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(
        f"{dataset_label}: {title}\n{contrast_badge} (N={cur_n}) vs {ref_badge} (N={ref_n}) — {test_name}",
        fontsize=9,
        fontweight="bold",
    )
    import matplotlib.patches as _mpatches

    # Legend below the axes: the upper-right corner overlaps the Post/R
    # bracket stars when significance is high.
    ax.legend(
        handles=[
            _mpatches.Patch(
                facecolor=TTEST_HC_BAR, edgecolor="white", label=f"{ref_badge} (N={ref_n})"
            ),
            _mpatches.Patch(
                facecolor=cohort_fill,
                edgecolor="white",
                label=f"{contrast_badge} (N={cur_n})",
            ),
        ],
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=2,
        framealpha=0.95,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_dual_contrasts_csv(
    iowa_tests: dict[str, dict[str, dict[int, dict[str, float]]]],
    stan_tests: dict[str, dict[str, dict[int, dict[str, float]]]],
    csv_path: Path,
    *,
    grid_tag: str,
    signal_branch: str | None = None,
) -> None:
    rows = []
    for dataset, tests, ref in (
        ("iowa", iowa_tests, TTEST_REF),
        ("stanford", stan_tests, STAN_REF),
    ):
        for contrast, hemi_dict in tests.items():
            for hemi in ("L", "R"):
                for k in (1, 2, 3):
                    st = hemi_dict[hemi][k]
                    rows.append(
                        {
                            "dataset": dataset,
                            "grid": grid_tag,
                            "signal_branch_iowa": signal_branch or "",
                            "signal_branch_stanford": "standard",
                            "contrast": f"{contrast}_vs_{ref}",
                            "metric": "area_pct",
                            "hemisphere": hemi,
                            "subregion_id": k,
                            "subregion": SUBREGION_NAMES[k],
                            "case_n": int(st["case_n"]),
                            "ctrl_n": int(st["ctrl_n"]),
                            "case_mean": f"{st['case_mean']:.4f}",
                            "ctrl_mean": f"{st['ctrl_mean']:.4f}",
                            "case_sd": f"{st['case_sd']:.4f}",
                            "ctrl_sd": f"{st['ctrl_sd']:.4f}",
                            "diff_case_minus_ctrl": f"{st['diff']:.4f}",
                            "welch_t": f"{st['welch_t']:.4f}",
                            "cohens_d_raw": f"{st['d']:.4f}",
                            "p_welch_uncorrected": f"{st['welch_p']:.6g}",
                            "stars_welch": st["welch_stars"],
                            "adj_t_ols_age_sex": f"{st['adj_t']:.4f}",
                            "p_adj_ols_age_sex": f"{st['adj_p']:.6g}",
                            "stars_adj": st["adj_stars"],
                            "covariates": "age+female",
                            "sex_coding": "female=1 iff recorded female; Stanford Gender==3 coded 0 (documented, N unchanged)",
                            "fdr": "none",
                        }
                    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    import csv as _csv

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved dual contrasts CSV: {csv_path}")


def write_ttest_contrasts_csv(
    ttest: dict[str, dict[str, dict[str, dict[int, dict[str, float]]]]],
    csv_path: Path,
    *,
    grid_tag: str,
    signal_branch: str | None = None,
) -> None:
    rows = []
    for contrast in TTEST_CONTRASTS:
        for metric in TTEST_METRICS:
            for hemi in ("L", "R"):
                for k in (1, 2, 3):
                    st = ttest[contrast][metric][hemi][k]
                    rows.append(
                        {
                            "grid": grid_tag,
                            "signal_branch": signal_branch or "",
                            "contrast": f"{contrast}_vs_{TTEST_REF}",
                            "metric": metric,
                            "hemisphere": hemi,
                            "subregion_id": k,
                            "subregion": SUBREGION_NAMES[k],
                            "case_n": int(st["case_n"]),
                            "ctrl_n": int(st["ctrl_n"]),
                            "case_mean": f"{st['case_mean']:.4f}",
                            "ctrl_mean": f"{st['ctrl_mean']:.4f}",
                            "case_sd": f"{st['case_sd']:.4f}",
                            "ctrl_sd": f"{st['ctrl_sd']:.4f}",
                            "diff_case_minus_ctrl": f"{st['diff']:.4f}",
                            "welch_t": f"{st['t']:.4f}",
                            "cohens_d": f"{st['d']:.4f}",
                            "p_uncorrected": f"{st['p']:.6g}",
                            "stars": st["stars"],
                            "fdr": "none",
                        }
                    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    import csv as _csv

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved t-test contrasts CSV: {csv_path}")


BRANCH_CLUSTER_RGBA = {
    1: (220, 92, 255, 255),
    2: (99, 150, 194, 255),
    3: (216, 92, 107, 255),
}


def branch_network_colors(
    names: dict[int, str],
) -> dict[str, tuple[int, int, int, int]]:
    return {names[idx]: BRANCH_CLUSTER_RGBA[idx] for idx in (1, 2, 3)}


def branch_matching_roots(data_root: Path) -> dict[str, Path]:
    root = Path(data_root).resolve()
    base = root / "outputs_iowa_jj_tms" / "work" / "hcp151_fixed_template_matching"
    return {
        "standard": base / "subjects",
        "ctxreg": base / "pfm_ctxreg" / "subjects",
    }


def branch_surface_paths(data_root: Path) -> dict[str, Path]:
    inputs = (
        Path(data_root).resolve() / "outputs_iowa_jj_tms" / "inputs"
    )
    return {
        "spec": inputs
        / "hippunfold"
        / "dense_corobl_batch"
        / "sub-1001"
        / "hippunfold"
        / "sub-1001"
        / "surf"
        / "sub-1001_den-512_label-hipp_surfaces.spec",
        "surf_l": inputs
        / "kong2022"
        / "templates"
        / "hipp_group_average"
        / "group_average_surfaces"
        / "group-average_hemi-L_space-corobl_den-512_label-hipp_midthickness.surf.gii",
        "surf_r": inputs
        / "kong2022"
        / "templates"
        / "hipp_group_average"
        / "group_average_surfaces"
        / "group-average_hemi-R_space-corobl_den-512_label-hipp_midthickness.surf.gii",
    }


def load_manifest_roster(fname: str, repo_root: Path = REPO_ROOT) -> list[str]:
    lines = (
        (Path(repo_root) / "manifests" / fname).read_text(encoding="utf-8").splitlines()
    )
    return [f"sub-{l.strip()}" for l in lines if l.strip() and not l.startswith("#")]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_branch_subregion_names(
    signal_branch: str,
    data_root: Path = REPO_ROOT,
    cohorts: list[dict[str, Any]] | None = None,
    subjects_root: Path | None = None,
) -> dict[int, str]:
    if signal_branch not in {"standard", "ctxreg"}:
        raise ValueError(f"Unsupported signal branch: {signal_branch}")
    roots = branch_matching_roots(data_root)
    root = Path(subjects_root).resolve() if subjects_root is not None else roots[signal_branch]
    expected_semantics = dict(TEMPLATE_MATCHING_LABEL_SEMANTICS)
    cohort_defs = COHORTS_DEF if cohorts is None else cohorts
    subjects = sorted(
        {subject for cohort in cohort_defs for subject in cohort["subjects"]}
    )
    observed: dict[int, str] | None = None
    for subject in subjects:
        for hemi in ("L", "R"):
            manifest_path = (
                root
                / subject
                / "matching"
                / "2mm"
                / f"hemi-{hemi}"
                / "combo_manifest.json"
            )
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"Missing template-matching manifest: {manifest_path}"
                )
            obj = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                obj.get("smoothing") != "2mm"
                or obj.get("hemi") != hemi
                or obj.get("status") != "ok"
            ):
                raise ValueError(
                    f"Template-matching manifest metadata mismatch: {manifest_path}"
                )
            manifest_branch = obj.get("signal_branch")
            if signal_branch == "ctxreg" and manifest_branch != "ctxreg":
                raise ValueError(f"CtxReg manifest branch mismatch: {manifest_path}")
            if signal_branch == "standard" and manifest_branch not in (
                None,
                "standard",
            ):
                raise ValueError(f"Standard manifest branch mismatch: {manifest_path}")
            raw_semantics = obj.get("label_semantics")
            semantics = (
                {int(key): str(value) for key, value in raw_semantics.items()}
                if isinstance(raw_semantics, dict)
                else None
            )
            if semantics != expected_semantics:
                raise ValueError(
                    f"Unexpected template-matching label semantics: {manifest_path}"
                )
            if observed is None:
                observed = semantics
            elif semantics != observed:
                raise ValueError(
                    f"Disagreeing template-matching label semantics: {manifest_path}"
                )
    if observed != expected_semantics:
        raise ValueError(
            f"No valid template-matching semantics found for {signal_branch}"
        )
    return dict(SUBREGION_DISPLAY_NAMES)


COHORTS_DEF = [
    {
        "key": "iowa_hc",
        "name": "Healthy Controls",
        "badge": "HC",
        "badge_color": "#0b3c5d",
        "bg_color": "#f8fafc",
        "desc": "Baseline Healthy Volunteers",
        "roster_file": "jj_tms_hc_67_subjects.txt",
        "subjects": load_manifest_roster("jj_tms_hc_67_subjects.txt"),
    },
    {
        "key": "iowa_pat",
        "name": "Depression Patients",
        "badge": "Patient",
        "badge_color": "#831843",
        "bg_color": "#fff1f2",
        "desc": "Clinical MDD (sub-3003 excluded)",
        "roster_file": "jj_tms_patient_26_subjects.txt",
        "subjects": load_manifest_roster("jj_tms_patient_26_subjects.txt"),
    },
    {
        "key": "iowa_pure_mdd",
        "name": "Pure MDD Subgroup",
        "badge": "Pure MDD",
        "badge_color": "#854d0e",
        "bg_color": "#fefce8",
        "desc": "MDD without Clinical Anxiety",
        "roster_file": "jj_tms_pure_mdd_7_subjects.txt",
        "subjects": load_manifest_roster("jj_tms_pure_mdd_7_subjects.txt"),
    },
    {
        "key": "iowa_comorbid_mdd",
        "name": "Comorbid MDD Subgroup",
        "badge": "Comorbid MDD",
        "badge_color": "#14532d",
        "bg_color": "#f0fdf4",
        "desc": "MDD with MINI-confirmed GAD",
        "roster_file": "jj_tms_comorbid_mdd_19_subjects.txt",
        "subjects": load_manifest_roster("jj_tms_comorbid_mdd_19_subjects.txt"),
    },
]


def cohorts_for_repo(repo_root: Path = REPO_ROOT) -> list[dict[str, Any]]:
    root = Path(repo_root).resolve()
    if root == REPO_ROOT.resolve():
        return COHORTS_DEF
    return [
        {**cohort, "subjects": load_manifest_roster(cohort["roster_file"], root)}
        for cohort in COHORTS_DEF
    ]


def build_tight_4view_composite(native_im: Image.Image) -> Image.Image:
    """Build a tight 4-surface composite ordered: [Left Front, Right Front, Left Back, Right Back]."""
    arr = np.array(native_im)

    def trim_b(img: Image.Image) -> Image.Image:
        a = np.array(img)
        corners = np.array([a[0, 0, :3], a[0, -1, :3], a[-1, 0, :3], a[-1, -1, :3]])
        is_white_bg = float(np.mean(corners)) > 127.0
        if is_white_bg:
            fg = np.any(a[..., :3] < 245, axis=2)
        else:
            fg = a[..., :3].sum(axis=2) > 15
        if not np.any(fg):
            return img
        rmin, rmax = np.where(np.any(fg, axis=1))[0][[0, -1]]
        cmin, cmax = np.where(np.any(fg, axis=0))[0][[0, -1]]
        return img.crop((cmin, rmin, cmax + 1, rmax + 1))

    # Top: Front (Lateral), Bottom: Back (Medial)
    v_l_front = trim_b(Image.fromarray(arr[0:512, 0:800]))
    v_r_front = trim_b(Image.fromarray(arr[0:512, 800:1600]))
    v_l_back = trim_b(Image.fromarray(arr[512:1024, 0:800]))
    v_r_back = trim_b(Image.fromarray(arr[512:1024, 800:1600]))

    target_h = max(v_l_front.height, v_r_front.height, v_l_back.height, v_r_back.height)

    # 4 views ordered: [Left Front, Right Front, Left Back, Right Back]
    views = [v_l_front, v_r_front, v_l_back, v_r_back]
    scaled_views = []
    for v in views:
        sc = target_h / float(v.height)
        sw = max(1, int(round(v.width * sc)))
        scaled_views.append(v.resize((sw, target_h), Image.Resampling.LANCZOS))

    gap_inner = 3
    gap_pair = 10
    pad = 1

    total_w = (
        pad * 2
        + scaled_views[0].width
        + gap_inner
        + scaled_views[1].width
        + gap_pair
        + scaled_views[2].width
        + gap_inner
        + scaled_views[3].width
    )
    total_h = pad * 2 + target_h

    canvas = Image.new("RGBA", (total_w, total_h), (255, 255, 255, 255))
    x = pad
    canvas.paste(
        scaled_views[0],
        (x, pad),
        scaled_views[0] if scaled_views[0].mode == "RGBA" else None,
    )
    x += scaled_views[0].width + gap_inner
    canvas.paste(
        scaled_views[1],
        (x, pad),
        scaled_views[1] if scaled_views[1].mode == "RGBA" else None,
    )
    x += scaled_views[1].width + gap_pair
    canvas.paste(
        scaled_views[2],
        (x, pad),
        scaled_views[2] if scaled_views[2].mode == "RGBA" else None,
    )
    x += scaled_views[2].width + gap_inner
    canvas.paste(
        scaled_views[3],
        (x, pad),
        scaled_views[3] if scaled_views[3].mode == "RGBA" else None,
    )
    return canvas


@dataclass
class CohortRowResult:
    cohort_key: str
    cohort_name: str
    cohort_badge: str
    badge_color: str
    bg_color: str
    desc: str
    n_subjects: int
    dominant_atlas_png: Path
    dominance_heatmap_png: Path
    dominant_counts: dict[int, int]
    dominant_percentages: dict[int, float]
    mean_dominance_pct: float
    median_dominance_pct: float
    min_dominance_pct: float
    max_dominance_pct: float
    subregion_names: dict[int, str] | None = None


def _declared_output_path(
    manifest_path: Path, manifest: dict[str, Any], key: str
) -> Path:
    record = manifest.get("outputs", {}).get(key)
    if not isinstance(record, dict):
        raise ValueError(f"Missing output receipt {key!r}: {manifest_path}")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise ValueError(
            f"Output receipt {key!r} is not an absolute path: {manifest_path}"
        )
    path = Path(raw_path)
    if not path.is_file():
        raise FileNotFoundError(f"Declared output {key!r} is missing: {path}")
    expected_size = record.get("size_bytes")
    if expected_size is not None and path.stat().st_size != int(expected_size):
        raise ValueError(f"Declared output size mismatch for {key!r}: {path}")
    expected_sha = record.get("sha256")
    if expected_sha and _sha256_file(path) != expected_sha:
        raise ValueError(f"Declared output checksum mismatch for {key!r}: {path}")
    return path


def compute_cohort_data(
    smoothing: str,
    subjects: list[str],
    subjects_root: Path,
    signal_branch: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Compute dominant label and dominance percentage for left and right hemispheres."""
    dominant_labels = {}
    dominance_pcts = {}

    all_hemi_labels = []
    all_hemi_pcts = []

    for hemi in ("L", "R"):
        sub_labels = []
        sub_corrs = []
        for s in subjects:
            if signal_branch is None:
                f_lbl = (
                    subjects_root
                    / s
                    / "matching"
                    / smoothing
                    / f"hemi-{hemi}"
                    / "labels.npy"
                )
                f_cor = (
                    subjects_root
                    / s
                    / "matching"
                    / smoothing
                    / f"hemi-{hemi}"
                    / "template_correlations.npy"
                )
            else:
                manifest_path = (
                    subjects_root
                    / s
                    / "matching"
                    / smoothing
                    / f"hemi-{hemi}"
                    / "combo_manifest.json"
                )
                if not manifest_path.is_file():
                    raise FileNotFoundError(
                        f"Missing matching manifest for {s} {smoothing} hemi-{hemi}: {manifest_path}"
                    )
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if (
                    manifest.get("smoothing") != smoothing
                    or manifest.get("hemi") != hemi
                    or manifest.get("status") != "ok"
                ):
                    raise ValueError(
                        f"Template-matching manifest metadata mismatch: {manifest_path}"
                    )
                manifest_branch = manifest.get("signal_branch")
                if signal_branch == "ctxreg" and manifest_branch != "ctxreg":
                    raise ValueError(
                        f"CtxReg manifest branch mismatch: {manifest_path}"
                    )
                if signal_branch == "standard" and manifest_branch not in (
                    None,
                    "standard",
                ):
                    raise ValueError(
                        f"Standard manifest branch mismatch: {manifest_path}"
                    )
                raw_semantics = manifest.get("label_semantics")
                semantics = (
                    {int(key): str(value) for key, value in raw_semantics.items()}
                    if isinstance(raw_semantics, dict)
                    else None
                )
                if semantics != TEMPLATE_MATCHING_LABEL_SEMANTICS:
                    raise ValueError(
                        f"Unexpected template-matching label semantics: {manifest_path}"
                    )
                f_lbl = _declared_output_path(manifest_path, manifest, "labels")
                f_cor = _declared_output_path(manifest_path, manifest, "scores")
            if not f_lbl.is_file() or not f_cor.is_file():
                raise FileNotFoundError(
                    f"Missing matching result for {s} {smoothing} hemi-{hemi}: {f_lbl}"
                )
            labels = np.asarray(np.load(f_lbl), dtype=np.int32)
            corrs = np.asarray(np.load(f_cor), dtype=np.float32)
            if labels.shape != (512,):
                raise ValueError(
                    f"Expected 512 den-512 labels for {f_lbl}, got {labels.shape}"
                )
            if corrs.shape != (512, 3):
                raise ValueError(
                    f"Expected (512, 3) template correlations for {f_cor}, got {corrs.shape}"
                )
            if not np.isfinite(corrs).any():
                raise ValueError(
                    f"Template correlations contain no finite values: {f_cor}"
                )
            sub_labels.append(labels)
            sub_corrs.append(corrs)

        arr_labels = np.asarray(sub_labels, dtype=np.int32)  # (N, 512)
        arr_corrs = np.asarray(sub_corrs, dtype=np.float32)  # (N, 512, 3)

        c1 = np.sum(arr_labels == 1, axis=0)
        c2 = np.sum(arr_labels == 2, axis=0)
        c3 = np.sum(arr_labels == 3, axis=0)
        counts = np.stack([c1, c2, c3], axis=-1)  # (512, 3)

        mean_corrs = np.mean(arr_corrs, axis=0)  # (512, 3)
        # Deterministic tie-breaking using mean template correlation
        combined_score = counts.astype(np.float64) + 1e-4 * mean_corrs.astype(
            np.float64
        )

        winner = np.argmax(combined_score, axis=-1) + 1  # 1, 2, 3
        max_counts = np.max(counts, axis=-1)
        pct = (max_counts / float(len(subjects))) * 100.0  # % agreement

        dominant_labels[hemi] = winner.astype(np.int32)
        dominance_pcts[hemi] = pct.astype(np.float32)

        all_hemi_labels.append(winner)
        all_hemi_pcts.append(pct)

    combined_labels = np.concatenate(all_hemi_labels)
    combined_pcts = np.concatenate(all_hemi_pcts)

    total_vertices = len(combined_labels)
    dominant_counts = {k: int(np.sum(combined_labels == k)) for k in (1, 2, 3)}
    dominant_percentages = {
        k: (dominant_counts[k] / total_vertices) * 100.0 for k in (1, 2, 3)
    }

    stats_summary = {
        "dominant_counts": dominant_counts,
        "dominant_percentages": dominant_percentages,
        "mean_dominance_pct": float(np.mean(combined_pcts)),
        "median_dominance_pct": float(np.median(combined_pcts)),
        "min_dominance_pct": float(np.min(combined_pcts)),
        "max_dominance_pct": float(np.max(combined_pcts)),
    }

    return dominant_labels, dominance_pcts, stats_summary


def apply_heatmap_metric_palette(
    metric_path: Path,
    palette_name: str = "magma",
    vmin: float = 33.33,
    vmax: float = 80.0,
) -> None:
    """Apply Workbench metric palette with explicit user min/max range."""
    cmd = [
        "wb_command",
        "-metric-palette",
        str(metric_path),
        "MODE_USER_SCALE",
        "-palette-name",
        str(palette_name),
        "-disp-pos",
        "true",
        "-disp-neg",
        "false",
        "-disp-zero",
        "true",
        "-interpolate",
        "true",
        "-normalization",
        "NORMALIZATION_ALL_MAP_DATA",
        "-pos-user",
        str(vmin),
        str(vmax),
    ]
    subprocess.run(cmd, check=True)


def render_cohort_panels(
    cohort_key: str,
    smoothing: str,
    variant: str,
    dominant_labels: dict[str, np.ndarray],
    dominance_pcts: dict[str, np.ndarray],
    panels_dir: Path,
    scratch_dir: Path,
    surf_l: Path = DEFAULT_SURF_L,
    surf_r: Path = DEFAULT_SURF_R,
    spec_path: Path = DEFAULT_SPEC,
    scene_path: Path = DEFAULT_SCENE,
    vmin: float = 33.33,
    vmax: float = 80.0,
    palette_name: str = "magma",
    force_rerender_heatmaps: bool = False,
    force_rerender_panels: bool = False,
    signal_branch: str | None = None,
    subregion_names: dict[int, str] | None = None,
) -> tuple[Path, Path]:
    """Render dominant atlas panel and dominance heatmap panel using Workbench."""
    cache_tag = f"{variant}_{signal_branch}" if signal_branch else variant
    display_names = subregion_names or SUBREGION_NAMES
    network_names = (
        [display_names[i] for i in (1, 2, 3)] if signal_branch else NETWORK_NAMES
    )
    network_colors = (
        branch_network_colors(display_names) if signal_branch else SUBREGION_COLORS
    )
    panels_dir.mkdir(parents=True, exist_ok=True)
    atlas_cached = (
        panels_dir
        / f"wb_{cohort_key}_{smoothing}_{cache_tag}_dominant_atlas_tight4view.png"
    )
    heatmap_cached = (
        panels_dir
        / f"wb_{cohort_key}_{smoothing}_{cache_tag}_dominance_heatmap_tight4view.png"
    )

    # 1. Render Dominant Subregion Atlas
    if atlas_cached.is_file() and not force_rerender_panels:
        atlas_comp_path = atlas_cached
    else:
        lbl_l_path = (
            scratch_dir / f"lbl_{cohort_key}_{smoothing}_{cache_tag}_hemi-L.label.gii"
        )
        lbl_r_path = (
            scratch_dir / f"lbl_{cohort_key}_{smoothing}_{cache_tag}_hemi-R.label.gii"
        )
        nib.save(
            make_network_label_gifti(
                labels=dominant_labels["L"],
                network_names=network_names,
                network_colors=network_colors,
            ),
            str(lbl_l_path),
        )
        nib.save(
            make_network_label_gifti(
                labels=dominant_labels["R"],
                network_names=network_names,
                network_colors=network_colors,
            ),
            str(lbl_r_path),
        )

        render_res = render_locked_grid(
            subject="1001",
            scene=scene_path,
            outdir=scratch_dir / f"atlas_{cohort_key}_{smoothing}_{cache_tag}",
            name=f"dominant_atlas_{cohort_key}_{smoothing}_{cache_tag}",
            title=f"Dominant Atlas | {cohort_key} {smoothing} {cache_tag}",
            left_labels=lbl_l_path,
            right_labels=lbl_r_path,
            left_surface=surf_l,
            right_surface=surf_r,
            spec_path=spec_path,
        )
        native_png = Path(render_res["native_scene_png"])
        atlas_comp = build_tight_4view_composite(Image.open(native_png).convert("RGBA"))
        atlas_comp.save(atlas_cached)
        atlas_comp_path = atlas_cached

    # 2. Render Dominance Heatmap
    if heatmap_cached.is_file() and not force_rerender_heatmaps:
        heatmap_comp_path = heatmap_cached
    else:
        shp_l_path = (
            scratch_dir / f"shape_{cohort_key}_{smoothing}_{cache_tag}_hemi-L.shape.gii"
        )
        shp_r_path = (
            scratch_dir / f"shape_{cohort_key}_{smoothing}_{cache_tag}_hemi-R.shape.gii"
        )
        save_shape_gii(dominance_pcts["L"], shp_l_path)
        save_shape_gii(dominance_pcts["R"], shp_r_path)

        apply_heatmap_metric_palette(shp_l_path, palette_name, vmin=vmin, vmax=vmax)
        apply_heatmap_metric_palette(shp_r_path, palette_name, vmin=vmin, vmax=vmax)

        metric_scene_root = build_scene_root(
            scene_path=scene_path,
            template_subject="1001",
            left_surface=surf_l,
            right_surface=surf_r,
            spec_path=spec_path,
            left_shape=shp_l_path,
            right_shape=shp_r_path,
            vmin=vmin,
            vmax=vmax,
            palette_name=palette_name,
        )

        metric_native_png = (
            scratch_dir / f"metric_{cohort_key}_{smoothing}_{cache_tag}_native.png"
        )
        _capture_scene_root(
            root=metric_scene_root,
            out_png=metric_native_png,
            width=1600,
            height=1024,
            renderer="OSMesa",
        )
        heatmap_comp = build_tight_4view_composite(
            Image.open(metric_native_png).convert("RGBA")
        )
        heatmap_comp.save(heatmap_cached)
        heatmap_comp_path = heatmap_cached

    return atlas_comp_path, heatmap_comp_path


def create_colorbar_image(
    width: int = 240, height: int = 16, vmin: float = 33.33, vmax: float = 80.0
) -> Image.Image:
    """Create horizontal colorbar image with magma colormap."""
    fig, ax = plt.subplots(figsize=(width / 100.0, height / 100.0), dpi=100)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cb = matplotlib.colorbar.ColorbarBase(
        ax, cmap=cm.magma, norm=norm, orientation="horizontal"
    )
    ax.axis("off")
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return Image.fromarray(rgba)


def compose_grid_figure(
    smoothing: str,
    variant: str,
    cohort_results: list[CohortRowResult],
    output_png: Path,
    signal_branch: str | None = None,
    subregion_names: dict[int, str] | None = None,
) -> None:
    """Compose single grid figure presenting all 4 Iowa cohorts in stacked rows."""
    total_w = 1940
    row_h = 575
    margin_x = 35
    margin_top = 185
    margin_bottom = 35
    gap_y = 20

    total_h = (
        margin_top
        + len(cohort_results) * row_h
        + (len(cohort_results) - 1) * gap_y
        + margin_bottom
    )

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Fonts
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 30)
        subtitle_font = ImageFont.truetype("DejaVuSans.ttf", 18)
        panel_title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 17)
        view_font = ImageFont.truetype("DejaVuSans.ttf", 13)
        stat_header_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
        stat_bold_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
        legend_font = ImageFont.truetype("DejaVuSans.ttf", 15)
        legend_bold = ImageFont.truetype("DejaVuSans-Bold.ttf", 15)
        badge_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
        n_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except OSError:
        title_font = subtitle_font = panel_title_font = view_font = stat_header_font = (
            stat_bold_font
        ) = legend_font = legend_bold = badge_font = n_font = ImageFont.load_default()

    # 1. Main Figure Title & Subtitle
    title_text = (
        "Iowa 3T Hippocampal Functional Atlas: Dominant Subregions & Agreement Heatmaps"
    )
    draw.text((margin_x, 22), title_text, fill=(20, 30, 45), font=title_font)

    display_names = subregion_names or SUBREGION_NAMES
    if signal_branch:
        branch_desc = (
            "standard cortical signal"
            if signal_branch == "standard"
            else "ctxreg cortical-residual signal"
        )
        sub_text = (
            f"{smoothing.upper()} Surface Smoothing  |  36P  |  {signal_branch.upper()} ({branch_desc})  "
            f"—  Iowa 3T Cohorts (N=93 Unique Subjects)"
        )
    else:
        variant_desc = (
            "36-Parameter (36P) Confound Regression"
            if variant == "36p"
            else "22-Parameter (22P) Confound Regression"
        )
        sub_text = (
            f"{smoothing.upper()} Surface Smoothing  |  {variant.upper()} Confound Regression ({variant_desc})  "
            f"—  Iowa 3T Cohorts (N=93 Total Subjects)"
        )
    draw.text((margin_x, 66), sub_text, fill=(71, 85, 105), font=subtitle_font)

    # 2. Global Legend Header Box
    legend_box_y = 105
    legend_box_h = 56
    box_w = total_w - margin_x * 2
    draw.rounded_rectangle(
        [(margin_x, legend_box_y), (margin_x + box_w, legend_box_y + legend_box_h)],
        radius=8,
        fill="#f8fafc",
        outline="#e2e8f0",
        width=1,
    )

    cur_x = margin_x + 18
    draw.text(
        (cur_x, legend_box_y + 18), "Subregions:", fill=(30, 41, 59), font=legend_bold
    )
    cur_x += int(draw.textlength("Subregions:", font=legend_bold)) + 16

    # Subregion Swatches: Anterior / Middle / Posterior
    legend_subregions = [
        (display_names[1], "#dc5cff"),
        (display_names[2], "#6396c2"),
        (display_names[3], "#d85c6b"),
    ]
    for name, hex_color in legend_subregions:
        draw.rounded_rectangle(
            [(cur_x, legend_box_y + 17), (cur_x + 20, legend_box_y + 37)],
            radius=3,
            fill=hex_color,
            outline="#475569",
            width=1,
        )
        cur_x += 26
        draw.text((cur_x, legend_box_y + 18), name, fill=(30, 41, 59), font=legend_font)
        cur_x += int(draw.textlength(name, font=legend_font)) + 20

    # Separator
    draw.line(
        [(cur_x, legend_box_y + 10), (cur_x, legend_box_y + legend_box_h - 10)],
        fill="#cbd5e1",
        width=1,
    )
    cur_x += 20

    # Dominance Agreement Colorbar (NO OVERLAP)
    cbar_title = "Dominance Agreement:"
    draw.text(
        (cur_x, legend_box_y + 18), cbar_title, fill=(30, 41, 59), font=legend_bold
    )
    cur_x += int(draw.textlength(cbar_title, font=legend_bold)) + 20

    cbar_w, cbar_h = 200, 16
    cbar_img = create_colorbar_image(width=cbar_w, height=cbar_h, vmin=33.33, vmax=80.0)
    canvas.paste(cbar_img, (cur_x, legend_box_y + 13))

    # Ticks under colorbar (precisely placed)
    tick_y = legend_box_y + 33
    t1 = "33.3%"
    t2 = "56.7%"
    t3 = "≥80%"
    draw.text((cur_x - 6, tick_y), t1, fill=(71, 85, 105), font=view_font)
    t2_w = draw.textlength(t2, font=view_font)
    draw.text(
        (cur_x + cbar_w // 2 - int(t2_w // 2), tick_y),
        t2,
        fill=(71, 85, 105),
        font=view_font,
    )
    t3_w = draw.textlength(t3, font=view_font)
    draw.text(
        (cur_x + cbar_w - int(t3_w) + 4, tick_y), t3, fill=(71, 85, 105), font=view_font
    )
    cur_x += cbar_w + 30

    # Separator
    draw.line(
        [(cur_x, legend_box_y + 10), (cur_x, legend_box_y + legend_box_h - 10)],
        fill="#cbd5e1",
        width=1,
    )
    cur_x += 20

    # Anatomical orientation hint
    draw.text(
        (cur_x, legend_box_y + 18),
        "Views (each panel): Left Lateral  |  Right Lateral  |  Left Medial  |  Right Medial",
        fill=(100, 116, 139),
        font=legend_font,
    )

    # 3. Render Stacked Cohort Rows
    meta_w = 210
    panel_gap = 25
    available_w = total_w - 2 * margin_x - meta_w - 2 * panel_gap
    target_pw = available_w // 2

    for r_idx, cr in enumerate(cohort_results):
        row_y = margin_top + r_idx * (row_h + gap_y)

        # Card container background
        draw.rounded_rectangle(
            [(margin_x, row_y), (total_w - margin_x, row_y + row_h)],
            radius=10,
            fill="#ffffff",
            outline="#cbd5e1",
            width=2,
        )

        # Left Metadata Card Column
        draw.rounded_rectangle(
            [(margin_x + 2, row_y + 2), (margin_x + meta_w, row_y + row_h - 2)],
            radius=9,
            fill=cr.bg_color,
            outline="#e2e8f0",
            width=1,
        )

        # Left card content: clean, centered, NO redundant descriptions
        meta_cx = margin_x + meta_w // 2
        badge_text = cr.cohort_badge
        n_text = f"N = {cr.n_subjects}"
        badge_w = int(draw.textlength(badge_text, font=badge_font)) + 36
        badge_h = 44
        badge_y = row_y + row_h // 2 - 32

        draw.rounded_rectangle(
            [
                (meta_cx - badge_w // 2, badge_y),
                (meta_cx + badge_w // 2, badge_y + badge_h),
            ],
            radius=8,
            fill=cr.badge_color,
        )
        draw.text(
            (
                meta_cx - int(draw.textlength(badge_text, font=badge_font) // 2),
                badge_y + 10,
            ),
            badge_text,
            fill="#ffffff",
            font=badge_font,
        )
        draw.text(
            (meta_cx - int(draw.textlength(n_text, font=n_font) // 2), badge_y + 56),
            n_text,
            fill="#1e293b",
            font=n_font,
        )

        # Vertical divider between meta card and panels
        draw.line(
            [(margin_x + meta_w, row_y + 10), (margin_x + meta_w, row_y + row_h - 10)],
            fill="#e2e8f0",
            width=1,
        )

        # Panel 1: Dominant Subregion Atlas
        p1_x = margin_x + meta_w + panel_gap
        draw.text(
            (p1_x + 4, row_y + 14),
            "Dominant Subregion Atlas",
            fill=(30, 41, 59),
            font=panel_title_font,
        )

        im_atlas = Image.open(cr.dominant_atlas_png).convert("RGBA")
        scale_ratio = target_pw / float(im_atlas.width)
        target_ph = int(round(im_atlas.height * scale_ratio))
        im_atlas_scaled = im_atlas.resize(
            (target_pw, target_ph), Image.Resampling.LANCZOS
        )

        surf_y = row_y + 42
        canvas.paste(im_atlas_scaled, (p1_x, surf_y), im_atlas_scaled)

        # Views hint under Atlas
        view_labels = ["LH Lateral", "RH Lateral", "LH Medial", "RH Medial"]
        v_step = target_pw / 4.0
        views_y = surf_y + target_ph + 4
        for v_idx, v_label in enumerate(view_labels):
            v_x = int(p1_x + v_idx * v_step + v_step * 0.22)
            draw.text((v_x, views_y), v_label, fill=(148, 163, 184), font=view_font)

        # Stats box for Atlas (Enlarged)
        stats_box_y = views_y + 20
        stats_box_h = 60
        draw.rounded_rectangle(
            [(p1_x, stats_box_y), (p1_x + target_pw, stats_box_y + stats_box_h)],
            radius=8,
            fill="#f8fafc",
            outline="#cbd5e1",
            width=1,
        )
        cnt1, pct1 = cr.dominant_counts[1], cr.dominant_percentages[1]
        cnt2, pct2 = cr.dominant_counts[2], cr.dominant_percentages[2]
        cnt3, pct3 = cr.dominant_counts[3], cr.dominant_percentages[3]
        label1 = display_names[1] if signal_branch else "Anterior/VAN"
        label2 = display_names[2] if signal_branch else "Middle/SMN"
        label3 = display_names[3] if signal_branch else "Posterior/DMN"

        s1_text = (
            f"{label1}: {pct1:.1f}% ({cnt1} vtx)   |   "
            f"{label2}: {pct2:.1f}% ({cnt2} vtx)   |   "
            f"{label3}: {pct3:.1f}% ({cnt3} vtx)"
        )
        draw.text(
            (p1_x + 16, stats_box_y + 10),
            "Subregion Composition across 1024 Bilateral Vertices:",
            fill="#475569",
            font=stat_header_font,
        )
        draw.text(
            (p1_x + 16, stats_box_y + 33), s1_text, fill="#0f172a", font=stat_bold_font
        )

        # Panel 2: Percentage Dominance Heatmap
        p2_x = p1_x + target_pw + panel_gap
        draw.text(
            (p2_x + 4, row_y + 14),
            "Percentage Dominance Heatmap",
            fill=(30, 41, 59),
            font=panel_title_font,
        )

        im_heatmap = Image.open(cr.dominance_heatmap_png).convert("RGBA")
        im_heatmap_scaled = im_heatmap.resize(
            (target_pw, target_ph), Image.Resampling.LANCZOS
        )
        canvas.paste(im_heatmap_scaled, (p2_x, surf_y), im_heatmap_scaled)

        # Views hint under Heatmap
        for v_idx, v_label in enumerate(view_labels):
            v_x = int(p2_x + v_idx * v_step + v_step * 0.22)
            draw.text((v_x, views_y), v_label, fill=(148, 163, 184), font=view_font)

        # Stats box for Heatmap (Enlarged)
        draw.rounded_rectangle(
            [(p2_x, stats_box_y), (p2_x + target_pw, stats_box_y + stats_box_h)],
            radius=8,
            fill="#f8fafc",
            outline="#cbd5e1",
            width=1,
        )
        s2_title = f"Subject Dominance Agreement Statistics ({cr.cohort_badge}, N={cr.n_subjects}):"
        s2_text = (
            f"Mean: {cr.mean_dominance_pct:.1f}%   |   "
            f"Median: {cr.median_dominance_pct:.1f}%   |   "
            f"Range: [{cr.min_dominance_pct:.1f}%, {cr.max_dominance_pct:.1f}%]"
        )
        draw.text(
            (p2_x + 16, stats_box_y + 10),
            s2_title,
            fill="#475569",
            font=stat_header_font,
        )
        draw.text(
            (p2_x + 16, stats_box_y + 33), s2_text, fill="#0f172a", font=stat_bold_font
        )

    # Save output
    output_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_png, "PNG", dpi=(300, 300))
    print(f"Successfully saved grid figure: {output_png} ({total_w}x{total_h})")


def compose_grid_figure_ttest(
    smoothing: str,
    variant: str,
    cohort_results: list[CohortRowResult],
    output_png: Path,
    ttest_panels: dict[str, dict[str, Path]],
    ttest: dict[str, dict[str, dict[str, dict[int, dict[str, float]]]]],
    signal_branch: str | None = None,
    subregion_names: dict[int, str] | None = None,
) -> None:
    """4-column grid: meta | atlas | heatmap | count-ttest | area-ttest.

    Columns 1-3 (title, legend, meta card, atlas, heatmap, stats boxes) are
    verbatim copies of compose_grid_figure in the formal renderer; only the
    two t-test columns are appended. The HC row leaves both t-test columns
    completely blank (no titles, panels, boxes, or borders).
    """
    # Base 1940 layout (identical to formal renderer) + 2 appended t-test columns.
    ttest_w = 600
    total_w = 1940 + 2 * (ttest_w + 25)
    row_h = 640
    margin_x = 35
    margin_top = 185
    margin_bottom = 35
    gap_y = 20

    total_h = (
        margin_top
        + len(cohort_results) * row_h
        + (len(cohort_results) - 1) * gap_y
        + margin_bottom
    )

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Fonts
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 30)
        subtitle_font = ImageFont.truetype("DejaVuSans.ttf", 18)
        panel_title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 17)
        view_font = ImageFont.truetype("DejaVuSans.ttf", 13)
        stat_header_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
        stat_bold_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
        legend_font = ImageFont.truetype("DejaVuSans.ttf", 15)
        legend_bold = ImageFont.truetype("DejaVuSans-Bold.ttf", 15)
        badge_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
        n_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
        tiny_font = ImageFont.truetype("DejaVuSans.ttf", 11)
    except OSError:
        title_font = subtitle_font = panel_title_font = view_font = stat_header_font = (
            stat_bold_font
        ) = legend_font = legend_bold = badge_font = n_font = tiny_font = (
            ImageFont.load_default()
        )

    # 1. Main Figure Title & Subtitle
    title_text = (
        "Iowa 3T Hippocampal Functional Atlas: Dominant Subregions & Agreement Heatmaps"
    )
    draw.text((margin_x, 22), title_text, fill=(20, 30, 45), font=title_font)

    display_names = subregion_names or SUBREGION_NAMES
    if signal_branch:
        branch_desc = (
            "standard cortical signal"
            if signal_branch == "standard"
            else "ctxreg cortical-residual signal"
        )
        sub_text = (
            f"{smoothing.upper()} Surface Smoothing  |  36P  |  {signal_branch.upper()} ({branch_desc})  "
            f"—  Iowa 3T Cohorts (N=93 Unique Subjects)"
        )
    else:
        variant_desc = (
            "36-Parameter (36P) Confound Regression"
            if variant == "36p"
            else "22-Parameter (22P) Confound Regression"
        )
        sub_text = (
            f"{smoothing.upper()} Surface Smoothing  |  {variant.upper()} Confound Regression ({variant_desc})  "
            f"—  Iowa 3T Cohorts (N=93 Total Subjects)"
        )
    draw.text((margin_x, 66), sub_text, fill=(71, 85, 105), font=subtitle_font)

    # 2. Global Legend Header Box
    legend_box_y = 105
    legend_box_h = 56
    box_w = total_w - margin_x * 2
    draw.rounded_rectangle(
        [(margin_x, legend_box_y), (margin_x + box_w, legend_box_y + legend_box_h)],
        radius=8,
        fill="#f8fafc",
        outline="#e2e8f0",
        width=1,
    )

    cur_x = margin_x + 18
    draw.text(
        (cur_x, legend_box_y + 18), "Subregions:", fill=(30, 41, 59), font=legend_bold
    )
    cur_x += int(draw.textlength("Subregions:", font=legend_bold)) + 16

    # Subregion Swatches: Anterior / Middle / Posterior
    legend_subregions = [
        (display_names[1], "#dc5cff"),
        (display_names[2], "#6396c2"),
        (display_names[3], "#d85c6b"),
    ]
    for name, hex_color in legend_subregions:
        draw.rounded_rectangle(
            [(cur_x, legend_box_y + 17), (cur_x + 20, legend_box_y + 37)],
            radius=3,
            fill=hex_color,
            outline="#475569",
            width=1,
        )
        cur_x += 26
        draw.text((cur_x, legend_box_y + 18), name, fill=(30, 41, 59), font=legend_font)
        cur_x += int(draw.textlength(name, font=legend_font)) + 20

    # Separator
    draw.line(
        [(cur_x, legend_box_y + 10), (cur_x, legend_box_y + legend_box_h - 10)],
        fill="#cbd5e1",
        width=1,
    )
    cur_x += 20

    # Dominance Agreement Colorbar (NO OVERLAP)
    cbar_title = "Dominance Agreement:"
    draw.text(
        (cur_x, legend_box_y + 18), cbar_title, fill=(30, 41, 59), font=legend_bold
    )
    cur_x += int(draw.textlength(cbar_title, font=legend_bold)) + 20

    cbar_w, cbar_h = 200, 16
    cbar_img = create_colorbar_image(width=cbar_w, height=cbar_h, vmin=33.33, vmax=80.0)
    canvas.paste(cbar_img, (cur_x, legend_box_y + 13))

    # Ticks under colorbar (precisely placed)
    tick_y = legend_box_y + 33
    t1 = "33.3%"
    t2 = "56.7%"
    t3 = "≥80%"
    draw.text((cur_x - 6, tick_y), t1, fill=(71, 85, 105), font=view_font)
    t2_w = draw.textlength(t2, font=view_font)
    draw.text(
        (cur_x + cbar_w // 2 - int(t2_w // 2), tick_y),
        t2,
        fill=(71, 85, 105),
        font=view_font,
    )
    t3_w = draw.textlength(t3, font=view_font)
    draw.text(
        (cur_x + cbar_w - int(t3_w) + 4, tick_y), t3, fill=(71, 85, 105), font=view_font
    )
    cur_x += cbar_w + 30

    # Separator
    draw.line(
        [(cur_x, legend_box_y + 10), (cur_x, legend_box_y + legend_box_h - 10)],
        fill="#cbd5e1",
        width=1,
    )
    cur_x += 20

    # Anatomical orientation hint
    draw.text(
        (cur_x, legend_box_y + 18),
        "Views (each panel): Left Lateral  |  Right Lateral  |  Left Medial  |  Right Medial",
        fill=(100, 116, 139),
        font=legend_font,
    )

    # 3. Render Stacked Cohort Rows
    meta_w = 210
    panel_gap = 25
    # Identical to formal renderer: panel width derived from the base 1940 grid.
    target_pw = (1940 - 2 * margin_x - meta_w - 2 * panel_gap) // 2

    for r_idx, cr in enumerate(cohort_results):
        row_y = margin_top + r_idx * (row_h + gap_y)

        # Card container background
        draw.rounded_rectangle(
            [(margin_x, row_y), (total_w - margin_x, row_y + row_h)],
            radius=10,
            fill="#ffffff",
            outline="#cbd5e1",
            width=2,
        )

        # Left Metadata Card Column
        draw.rounded_rectangle(
            [(margin_x + 2, row_y + 2), (margin_x + meta_w, row_y + row_h - 2)],
            radius=9,
            fill=cr.bg_color,
            outline="#e2e8f0",
            width=1,
        )

        # Left card content: clean, centered, NO redundant descriptions
        meta_cx = margin_x + meta_w // 2
        badge_text = cr.cohort_badge
        n_text = f"N = {cr.n_subjects}"
        badge_w = int(draw.textlength(badge_text, font=badge_font)) + 36
        badge_h = 44
        badge_y = row_y + row_h // 2 - 32

        draw.rounded_rectangle(
            [
                (meta_cx - badge_w // 2, badge_y),
                (meta_cx + badge_w // 2, badge_y + badge_h),
            ],
            radius=8,
            fill=cr.badge_color,
        )
        draw.text(
            (
                meta_cx - int(draw.textlength(badge_text, font=badge_font) // 2),
                badge_y + 10,
            ),
            badge_text,
            fill="#ffffff",
            font=badge_font,
        )
        draw.text(
            (meta_cx - int(draw.textlength(n_text, font=n_font) // 2), badge_y + 56),
            n_text,
            fill="#1e293b",
            font=n_font,
        )

        # Vertical divider between meta card and panels
        draw.line(
            [(margin_x + meta_w, row_y + 10), (margin_x + meta_w, row_y + row_h - 10)],
            fill="#e2e8f0",
            width=1,
        )

        # Panel 1: Dominant Subregion Atlas
        p1_x = margin_x + meta_w + panel_gap
        draw.text(
            (p1_x + 4, row_y + 14),
            "Dominant Subregion Atlas",
            fill=(30, 41, 59),
            font=panel_title_font,
        )

        im_atlas = Image.open(cr.dominant_atlas_png).convert("RGBA")
        scale_ratio = target_pw / float(im_atlas.width)
        target_ph = int(round(im_atlas.height * scale_ratio))
        im_atlas_scaled = im_atlas.resize(
            (target_pw, target_ph), Image.Resampling.LANCZOS
        )

        surf_y = row_y + 42
        canvas.paste(im_atlas_scaled, (p1_x, surf_y), im_atlas_scaled)

        # Views hint under Atlas
        view_labels = ["LH Lateral", "RH Lateral", "LH Medial", "RH Medial"]
        v_step = target_pw / 4.0
        views_y = surf_y + target_ph + 4
        for v_idx, v_label in enumerate(view_labels):
            v_x = int(p1_x + v_idx * v_step + v_step * 0.22)
            draw.text((v_x, views_y), v_label, fill=(148, 163, 184), font=view_font)

        # Stats box for Atlas (Enlarged)
        stats_box_y = views_y + 20
        stats_box_h = 60
        draw.rounded_rectangle(
            [(p1_x, stats_box_y), (p1_x + target_pw, stats_box_y + stats_box_h)],
            radius=8,
            fill="#f8fafc",
            outline="#cbd5e1",
            width=1,
        )
        cnt1, pct1 = cr.dominant_counts[1], cr.dominant_percentages[1]
        cnt2, pct2 = cr.dominant_counts[2], cr.dominant_percentages[2]
        cnt3, pct3 = cr.dominant_counts[3], cr.dominant_percentages[3]
        label1 = display_names[1] if signal_branch else "Anterior/VAN"
        label2 = display_names[2] if signal_branch else "Middle/SMN"
        label3 = display_names[3] if signal_branch else "Posterior/DMN"

        s1_text = (
            f"{label1}: {pct1:.1f}% ({cnt1} vtx)   |   "
            f"{label2}: {pct2:.1f}% ({cnt2} vtx)   |   "
            f"{label3}: {pct3:.1f}% ({cnt3} vtx)"
        )
        draw.text(
            (p1_x + 16, stats_box_y + 10),
            "Subregion Composition across 1024 Bilateral Vertices:",
            fill="#475569",
            font=stat_header_font,
        )
        draw.text(
            (p1_x + 16, stats_box_y + 33), s1_text, fill="#0f172a", font=stat_bold_font
        )

        # Panel 2: Percentage Dominance Heatmap
        p2_x = p1_x + target_pw + panel_gap
        draw.text(
            (p2_x + 4, row_y + 14),
            "Percentage Dominance Heatmap",
            fill=(30, 41, 59),
            font=panel_title_font,
        )

        im_heatmap = Image.open(cr.dominance_heatmap_png).convert("RGBA")
        im_heatmap_scaled = im_heatmap.resize(
            (target_pw, target_ph), Image.Resampling.LANCZOS
        )
        canvas.paste(im_heatmap_scaled, (p2_x, surf_y), im_heatmap_scaled)

        # Views hint under Heatmap
        for v_idx, v_label in enumerate(view_labels):
            v_x = int(p2_x + v_idx * v_step + v_step * 0.22)
            draw.text((v_x, views_y), v_label, fill=(148, 163, 184), font=view_font)

        # Stats box for Heatmap (Enlarged)
        draw.rounded_rectangle(
            [(p2_x, stats_box_y), (p2_x + target_pw, stats_box_y + stats_box_h)],
            radius=8,
            fill="#f8fafc",
            outline="#cbd5e1",
            width=1,
        )
        s2_title = f"Subject Dominance Agreement Statistics ({cr.cohort_badge}, N={cr.n_subjects}):"
        s2_text = (
            f"Mean: {cr.mean_dominance_pct:.1f}%   |   "
            f"Median: {cr.median_dominance_pct:.1f}%   |   "
            f"Range: [{cr.min_dominance_pct:.1f}%, {cr.max_dominance_pct:.1f}%]"
        )
        draw.text(
            (p2_x + 16, stats_box_y + 10),
            s2_title,
            fill="#475569",
            font=stat_header_font,
        )
        draw.text(
            (p2_x + 16, stats_box_y + 33), s2_text, fill="#0f172a", font=stat_bold_font
        )

        # Panels 3-4: t-tests vs HC (left/right hippocampus). HC row: fully blank.
        if cr.cohort_key == TTEST_REF:
            continue
        panels = ttest_panels.get(cr.cohort_key, {})
        for c_idx, metric in enumerate(TTEST_METRICS):
            px = p2_x + target_pw + panel_gap + c_idx * (ttest_w + panel_gap)
            title = (
                "Vertex Count vs HC (Left / Right Hippocampus)"
                if metric == "count"
                else "Area % of Hippocampus vs HC (Left / Right)"
            )
            draw.text(
                (px + 4, row_y + 14), title, fill=(30, 41, 59), font=panel_title_font
            )
            png = panels.get(metric)
            if png is None or not Path(png).is_file():
                raise FileNotFoundError(
                    f"Missing t-test panel: {cr.cohort_key} {metric}"
                )
            im_t = Image.open(png).convert("RGBA")
            th = int(round(im_t.height * (ttest_w / float(im_t.width))))
            im_t_scaled = im_t.resize((ttest_w, th), Image.Resampling.LANCZOS)
            canvas.paste(im_t_scaled, (px, surf_y), im_t_scaled)
            ty = surf_y + th + 4
            tbox_h = 100
            draw.rounded_rectangle(
                [(px, ty), (px + ttest_w, ty + tbox_h)],
                radius=8,
                fill="#f8fafc",
                outline="#cbd5e1",
                width=1,
            )
            draw.text(
                (px + 16, ty + 8),
                "Welch two-sided vs HC, uncorrected (no FDR; + 0.05<=p<0.10):",
                fill="#475569",
                font=stat_header_font,
            )
            for li, k in enumerate((1, 2, 3)):
                parts = []
                for hemi in ("L", "R"):
                    st = ttest[cr.cohort_key][metric][hemi][k]
                    parts.append(
                        f"{hemi} p={st['p']:.3g} {st['stars']} (t={st['t']:.2f}, d={st['d']:.2f})"
                    )
                line = f"{display_names[k]}:  " + "   |   ".join(parts)
                draw.text(
                    (px + 16, ty + 30 + li * 22), line, fill="#0f172a", font=tiny_font
                )

    # Save output
    output_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_png, "PNG", dpi=(300, 300))
    print(f"Successfully saved t-test grid figure: {output_png} ({total_w}x{total_h})")


def write_comprehensive_csv(
    all_results: dict[str, list[CohortRowResult]], csv_path: Path
) -> None:
    """Write comprehensive metrics across all 4 grid combinations and 4 cohorts."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for grid_tag, cohort_list in all_results.items():
        smoothing, variant = grid_tag.split("_")
        for cr in cohort_list:
            rows.append(
                {
                    "grid": grid_tag,
                    "smoothing": smoothing,
                    "variant": variant,
                    "cohort_key": cr.cohort_key,
                    "cohort_name": cr.cohort_name,
                    "n_subjects": cr.n_subjects,
                    "van_ant_vtx_count": cr.dominant_counts[1],
                    "van_ant_vtx_pct": f"{cr.dominant_percentages[1]:.2f}",
                    "smn_mid_vtx_count": cr.dominant_counts[2],
                    "smn_mid_vtx_pct": f"{cr.dominant_percentages[2]:.2f}",
                    "dmn_post_vtx_count": cr.dominant_counts[3],
                    "dmn_post_vtx_pct": f"{cr.dominant_percentages[3]:.2f}",
                    "mean_dominance_agreement_pct": f"{cr.mean_dominance_pct:.2f}",
                    "median_dominance_agreement_pct": f"{cr.median_dominance_pct:.2f}",
                    "min_dominance_agreement_pct": f"{cr.min_dominance_pct:.2f}",
                    "max_dominance_agreement_pct": f"{cr.max_dominance_pct:.2f}",
                }
            )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved comprehensive stats CSV: {csv_path}")


def write_manifest_json(
    all_results: dict[str, list[CohortRowResult]],
    rendered_figures: dict[str, Path],
    manifest_path: Path,
) -> None:
    """Write run manifest and provenance metadata."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    def file_sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "script": str(Path(__file__).resolve()),
        "cohorts": [
            {
                "key": c["key"],
                "name": c["name"],
                "badge": c["badge"],
                "n_subjects": len(c["subjects"]),
                "roster_file": c["roster_file"],
            }
            for c in COHORTS_DEF
        ],
        "grid_combinations": ["2mm_36p", "2mm_22p", "4mm_36p", "4mm_22p"],
        "master_figures": {
            grid: {
                "path": str(fig_path.resolve()),
                "sha256": file_sha256(fig_path),
                "size_bytes": fig_path.stat().st_size,
            }
            for grid, fig_path in rendered_figures.items()
        },
        "condition_metrics": {
            grid: [
                {
                    "cohort_key": cr.cohort_key,
                    "cohort_name": cr.cohort_name,
                    "n_subjects": cr.n_subjects,
                    "dominant_counts": cr.dominant_counts,
                    "dominant_percentages": cr.dominant_percentages,
                    "mean_dominance_pct": cr.mean_dominance_pct,
                    "median_dominance_pct": cr.median_dominance_pct,
                    "min_dominance_pct": cr.min_dominance_pct,
                    "max_dominance_pct": cr.max_dominance_pct,
                    "atlas_png": str(cr.dominant_atlas_png),
                    "heatmap_png": str(cr.dominance_heatmap_png),
                }
                for cr in cohort_list
            ]
            for grid, cohort_list in all_results.items()
        },
    }

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved run manifest: {manifest_path}")


DUAL_SHORT_LABELS = ("Ant/L", "Mid/L", "Post/L", "Ant/R", "Mid/R", "Post/R")
DUAL_SUBREGION_SHORT = {1: "Ant", 2: "Mid", 3: "Post"}


def _dual_fonts() -> dict[str, Any]:
    try:
        return {
            "title": ImageFont.truetype("DejaVuSans-Bold.ttf", 30),
            "subtitle": ImageFont.truetype("DejaVuSans.ttf", 17),
            "panel_title": ImageFont.truetype("DejaVuSans-Bold.ttf", 15),
            "view": ImageFont.truetype("DejaVuSans.ttf", 12),
            "stat_header": ImageFont.truetype("DejaVuSans-Bold.ttf", 13),
            "stat_bold": ImageFont.truetype("DejaVuSans-Bold.ttf", 13),
            "legend": ImageFont.truetype("DejaVuSans.ttf", 14),
            "legend_bold": ImageFont.truetype("DejaVuSans-Bold.ttf", 14),
            "badge": ImageFont.truetype("DejaVuSans-Bold.ttf", 19),
            "n": ImageFont.truetype("DejaVuSans-Bold.ttf", 17),
            "tiny": ImageFont.truetype("DejaVuSans.ttf", 10),
        }
    except OSError:
        fallback = ImageFont.load_default()
        return {k: fallback for k in ("title", "subtitle", "panel_title", "view", "stat_header", "stat_bold", "legend", "legend_bold", "badge", "n", "tiny")}


def _draw_dual_meta_card(
    draw: ImageDraw.ImageDraw,
    fonts: dict[str, Any],
    x0: int,
    row_y: int,
    row_h: int,
    meta_w: int,
    cr: CohortRowResult | None,
) -> None:
    draw.rounded_rectangle(
        [(x0 + 2, row_y + 2), (x0 + meta_w, row_y + row_h - 2)],
        radius=9,
        fill=cr.bg_color if cr is not None else "#f1f5f9",
        outline="#e2e8f0",
        width=1,
    )
    meta_cx = x0 + meta_w // 2
    if cr is None:
        dash = "—"
        draw.text(
            (meta_cx - int(draw.textlength(dash, font=fonts["badge"]) // 2), row_y + row_h // 2 - 16),
            dash,
            fill="#94a3b8",
            font=fonts["badge"],
        )
        return
    badge_text = cr.cohort_badge
    n_text = f"N = {cr.n_subjects}"
    badge_w = int(draw.textlength(badge_text, font=fonts["badge"])) + 36
    badge_h = 44
    badge_y = row_y + row_h // 2 - 32
    draw.rounded_rectangle(
        [(meta_cx - badge_w // 2, badge_y), (meta_cx + badge_w // 2, badge_y + badge_h)],
        radius=8,
        fill=cr.badge_color,
    )
    draw.text(
        (meta_cx - int(draw.textlength(badge_text, font=fonts["badge"]) // 2), badge_y + 10),
        badge_text,
        fill="#ffffff",
        font=fonts["badge"],
    )
    draw.text(
        (meta_cx - int(draw.textlength(n_text, font=fonts["n"]) // 2), badge_y + 56),
        n_text,
        fill="#1e293b",
        font=fonts["n"],
    )


def _draw_dual_surface_block(
    draw: ImageDraw.ImageDraw,
    canvas: Image.Image,
    fonts: dict[str, Any],
    x0: int,
    row_y: int,
    target_pw: int,
    target_ph: int,
    cr: CohortRowResult,
    block_label: str,
    display_names: dict[int, str],
) -> None:
    view_labels = ["LH Lateral", "RH Lateral", "LH Medial", "RH Medial"]
    v_step = target_pw / 4.0
    surf_y = row_y + 42
    draw.text((x0 + 4, row_y + 14), f"Dominant Subregion Atlas ({block_label})", fill=(30, 41, 59), font=fonts["panel_title"])
    im_atlas = Image.open(cr.dominant_atlas_png).convert("RGBA")
    im_atlas_scaled = im_atlas.resize((target_pw, target_ph), Image.Resampling.LANCZOS)
    canvas.paste(im_atlas_scaled, (x0, surf_y), im_atlas_scaled)
    views_y = surf_y + target_ph + 4
    for v_idx, v_label in enumerate(view_labels):
        v_x = int(x0 + v_idx * v_step + v_step * 0.22)
        draw.text((v_x, views_y), v_label, fill=(148, 163, 184), font=fonts["view"])
    stats_box_y = views_y + 20
    stats_box_h = 60
    draw.rounded_rectangle(
        [(x0, stats_box_y), (x0 + target_pw, stats_box_y + stats_box_h)],
        radius=8, fill="#f8fafc", outline="#cbd5e1", width=1,
    )
    cnt1, pct1 = cr.dominant_counts[1], cr.dominant_percentages[1]
    cnt2, pct2 = cr.dominant_counts[2], cr.dominant_percentages[2]
    cnt3, pct3 = cr.dominant_counts[3], cr.dominant_percentages[3]
    s1_text = (
        f"{display_names[1]}: {pct1:.1f}% ({cnt1}) | "
        f"{display_names[2]}: {pct2:.1f}% ({cnt2}) | "
        f"{display_names[3]}: {pct3:.1f}% ({cnt3})"
    )
    draw.text((x0 + 12, stats_box_y + 8), "Composition (1024 bilateral vertices):", fill="#475569", font=fonts["stat_header"])
    draw.text((x0 + 12, stats_box_y + 32), s1_text, fill="#0f172a", font=fonts["stat_bold"])

    p2_x = x0 + target_pw + 20
    draw.text((p2_x + 4, row_y + 14), f"Percentage Dominance Heatmap ({block_label})", fill=(30, 41, 59), font=fonts["panel_title"])
    im_heatmap = Image.open(cr.dominance_heatmap_png).convert("RGBA")
    im_heatmap_scaled = im_heatmap.resize((target_pw, target_ph), Image.Resampling.LANCZOS)
    canvas.paste(im_heatmap_scaled, (p2_x, surf_y), im_heatmap_scaled)
    for v_idx, v_label in enumerate(view_labels):
        v_x = int(p2_x + v_idx * v_step + v_step * 0.22)
        draw.text((v_x, views_y), v_label, fill=(148, 163, 184), font=fonts["view"])
    draw.rounded_rectangle(
        [(p2_x, stats_box_y), (p2_x + target_pw, stats_box_y + stats_box_h)],
        radius=8, fill="#f8fafc", outline="#cbd5e1", width=1,
    )
    s2_title = f"Agreement ({cr.cohort_badge}, N={cr.n_subjects}):"
    s2_text = (
        f"Mean: {cr.mean_dominance_pct:.1f}% | Median: {cr.median_dominance_pct:.1f}% | "
        f"Range: [{cr.min_dominance_pct:.1f}%, {cr.max_dominance_pct:.1f}%]"
    )
    draw.text((p2_x + 12, stats_box_y + 8), s2_title, fill="#475569", font=fonts["stat_header"])
    draw.text((p2_x + 12, stats_box_y + 32), s2_text, fill="#0f172a", font=fonts["stat_bold"])


def _draw_dual_stat_panel(
    draw: ImageDraw.ImageDraw,
    canvas: Image.Image,
    fonts: dict[str, Any],
    px: int,
    row_y: int,
    surf_y: int,
    ttest_w: int,
    cr: CohortRowResult,
    panel_png: Path,
    tests: dict[str, dict[str, dict[int, dict[str, float]]]],
    mode: str,
    ref_badge: str,
    display_names: dict[int, str],
) -> None:
    title = (
        f"Area % vs {ref_badge} — age+sex adjusted (OLS)"
        if mode == "adj"
        else f"Area % vs {ref_badge} — Welch, uncorrected"
    )
    draw.text((px + 4, row_y + 14), title, fill=(30, 41, 59), font=fonts["panel_title"])
    if not panel_png.is_file():
        raise FileNotFoundError(f"Missing dual stat panel: {cr.cohort_key} {mode}")
    im_t = Image.open(panel_png).convert("RGBA")
    th = int(round(im_t.height * (ttest_w / float(im_t.width))))
    im_t_scaled = im_t.resize((ttest_w, th), Image.Resampling.LANCZOS)
    canvas.paste(im_t_scaled, (px, surf_y), im_t_scaled)
    ty = surf_y + th + 4
    tbox_h = 100
    draw.rounded_rectangle(
        [(px, ty), (px + ttest_w, ty + tbox_h)],
        radius=8, fill="#f8fafc", outline="#cbd5e1", width=1,
    )
    header = (
        "OLS group+age+sex, uncorrected (no FDR; + 0.05<=p<0.10):"
        if mode == "adj"
        else "Welch two-sided, uncorrected (no FDR; + 0.05<=p<0.10):"
    )
    draw.text((px + 12, ty + 8), header, fill="#475569", font=fonts["stat_header"])
    for li, k in enumerate((1, 2, 3)):
        parts = []
        for hemi in ("L", "R"):
            st = tests[cr.cohort_key][hemi][k]
            if mode == "adj":
                parts.append(f"{hemi} p={st['adj_p']:.3g} {st['adj_stars']} (t={st['adj_t']:.2f})")
            else:
                parts.append(f"{hemi} p={st['welch_p']:.3g} {st['welch_stars']} (t={st['welch_t']:.2f}, d={st['d']:.2f})")
        line = f"{DUAL_SUBREGION_SHORT[k]}:  " + " | ".join(parts)
        draw.text((px + 12, ty + 30 + li * 22), line, fill="#0f172a", font=fonts["tiny"])


def compose_dual_dataset_figure(
    *,
    output_png: Path,
    iowa_results: list[CohortRowResult],
    stan_results: list[CohortRowResult],
    iowa_panels: dict[str, dict[str, Path]],
    stan_panels: dict[str, dict[str, Path]],
    iowa_tests: dict[str, dict[str, dict[int, dict[str, float]]]],
    stan_tests: dict[str, dict[str, dict[int, dict[str, float]]]],
    signal_branch: str,
    subregion_names: dict[int, str] | None = None,
) -> None:
    """Dual-dataset grid: left block Iowa (4 rows), right block Stanford (7 rows).

    Rows 0-3 pair Iowa[i] with Stanford[i]; rows 4-6 are Stanford-only (Iowa
    side blank). Each side shows atlas + heatmap + adjusted area % + Welch
    area % (vertex count removed). Reference rows (HC / NTHC) leave both stat
    columns blank.
    """
    if len(iowa_results) != 4:
        raise ValueError(f"Expected 4 Iowa rows, got {len(iowa_results)}")
    if len(stan_results) != 7:
        raise ValueError(f"Expected 7 Stanford rows, got {len(stan_results)}")
    n_rows = 7
    meta_w = 200
    panel_gap = 20
    mid_gap = 60
    target_pw = 700
    ttest_w = 520
    block_w = meta_w + panel_gap + target_pw + panel_gap + target_pw + panel_gap + ttest_w + panel_gap + ttest_w
    margin_x = 35
    margin_top = 235
    margin_bottom = 35
    gap_y = 20
    row_h = 640
    total_w = margin_x + block_w + mid_gap + block_w + margin_x
    total_h = margin_top + n_rows * row_h + (n_rows - 1) * gap_y + margin_bottom

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    fonts = _dual_fonts()
    display_names = subregion_names or SUBREGION_NAMES

    branch_desc = "standard cortical signal" if signal_branch == "standard" else "ctxreg cortical-residual signal"
    draw.text((margin_x, 22), "Iowa 3T + Stanford Hippocampal Functional Atlas: Dominant Subregions & Agreement Heatmaps", fill=(20, 30, 45), font=fonts["title"])
    draw.text((margin_x, 64), f"2MM Surface Smoothing  |  36P  |  {signal_branch.upper()} ({branch_desc})  —  Left block: Iowa 3T (N=93 unique)  |  Right block: Stanford (N=194 unique)", fill=(71, 85, 105), font=fonts["subtitle"])
    draw.text((margin_x, 92), "Stanford block uses standard-branch TM in both figures.  Adj = OLS group + age + sex on area % (bars/dots show raw area %);  Welch uncorrected (no FDR).", fill=(71, 85, 105), font=fonts["subtitle"])

    legend_box_y = 128
    legend_box_h = 72
    box_w = total_w - margin_x * 2
    draw.rounded_rectangle(
        [(margin_x, legend_box_y), (margin_x + box_w, legend_box_y + legend_box_h)],
        radius=8, fill="#f8fafc", outline="#e2e8f0", width=1,
    )
    cur_x = margin_x + 18
    draw.text((cur_x, legend_box_y + 24), "Subregions:", fill=(30, 41, 59), font=fonts["legend_bold"])
    cur_x += int(draw.textlength("Subregions:", font=fonts["legend_bold"])) + 16
    for name, hex_color in [(display_names[1], "#dc5cff"), (display_names[2], "#6396c2"), (display_names[3], "#d85c6b")]:
        draw.rounded_rectangle([(cur_x, legend_box_y + 23), (cur_x + 20, legend_box_y + 43)], radius=3, fill=hex_color, outline="#475569", width=1)
        cur_x += 26
        draw.text((cur_x, legend_box_y + 24), name, fill=(30, 41, 59), font=fonts["legend"])
        cur_x += int(draw.textlength(name, font=fonts["legend"])) + 20
    draw.line([(cur_x, legend_box_y + 10), (cur_x, legend_box_y + legend_box_h - 10)], fill="#cbd5e1", width=1)
    cur_x += 20
    draw.text((cur_x, legend_box_y + 24), "Dominance Agreement:", fill=(30, 41, 59), font=fonts["legend_bold"])
    cur_x += int(draw.textlength("Dominance Agreement:", font=fonts["legend_bold"])) + 20
    cbar_w, cbar_h = 200, 16
    cbar_img = create_colorbar_image(width=cbar_w, height=cbar_h, vmin=33.33, vmax=80.0)
    canvas.paste(cbar_img, (cur_x, legend_box_y + 19))
    draw.text((cur_x - 6, legend_box_y + 41), "33.3%", fill=(71, 85, 105), font=fonts["view"])
    draw.text((cur_x + cbar_w - 30, legend_box_y + 41), "≥80%", fill=(71, 85, 105), font=fonts["view"])
    cur_x += cbar_w + 30
    draw.line([(cur_x, legend_box_y + 10), (cur_x, legend_box_y + legend_box_h - 10)], fill="#cbd5e1", width=1)
    cur_x += 20
    draw.text((cur_x, legend_box_y + 12), "Left block ref: HC (Iowa)   |   Right block ref: NTHC (Stanford)", fill=(30, 41, 59), font=fonts["legend"])
    draw.text((cur_x, legend_box_y + 38), "Stat cols per block: [age+sex adjusted area %]  [Welch uncorrected area %]", fill=(100, 116, 139), font=fonts["legend"])

    # Atlas composite target height (shared by both blocks)
    sample = Image.open(iowa_results[0].dominant_atlas_png).convert("RGBA")
    target_ph = int(round(sample.height * (target_pw / float(sample.width))))
    surf_y_off = 42

    for r_idx in range(n_rows):
        row_y = margin_top + r_idx * (row_h + gap_y)
        draw.rounded_rectangle(
            [(margin_x, row_y), (total_w - margin_x, row_y + row_h)],
            radius=10, fill="#ffffff", outline="#cbd5e1", width=2,
        )
        mid_x = margin_x + block_w + mid_gap // 2
        draw.line([(mid_x, row_y + 10), (mid_x, row_y + row_h - 10)], fill="#cbd5e1", width=2)
        left_cr = iowa_results[r_idx] if r_idx < len(iowa_results) else None
        right_cr = stan_results[r_idx]
        for block_x0, cr, block_label, ref_key, ref_badge, panels, tests in (
            (margin_x, left_cr, "Iowa", TTEST_REF, "HC", iowa_panels, iowa_tests),
            (margin_x + block_w + mid_gap, right_cr, "Stanford", STAN_REF, "NTHC", stan_panels, stan_tests),
        ):
            _draw_dual_meta_card(draw, fonts, block_x0, row_y, row_h, meta_w, cr)
            draw.line([(block_x0 + meta_w, row_y + 10), (block_x0 + meta_w, row_y + row_h - 10)], fill="#e2e8f0", width=1)
            if cr is None:
                continue
            surf_x0 = block_x0 + meta_w + panel_gap
            _draw_dual_surface_block(draw, canvas, fonts, surf_x0, row_y, target_pw, target_ph, cr, block_label, display_names)
            if cr.cohort_key == ref_key:
                continue
            stat_x0 = surf_x0 + target_pw + panel_gap + target_pw + panel_gap
            for c_idx, mode in enumerate(("adj", "welch")):
                px = stat_x0 + c_idx * (ttest_w + panel_gap)
                _draw_dual_stat_panel(
                    draw, canvas, fonts, px, row_y, row_y + surf_y_off, ttest_w,
                    cr, panels[cr.cohort_key][mode], tests, mode, ref_badge, display_names,
                )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_png, "PNG", dpi=(300, 300))
    print(f"Successfully saved dual-dataset grid figure: {output_png} ({total_w}x{total_h})")


def render_branch_outputs_ttest(
    selected_branches: tuple[str, ...],
    *,
    out_dir: Path,
    repo_root: Path,
    data_root: Path,
    force_heatmaps: bool,
    force_rerender_panels: bool,
) -> None:
    """Branch t-test figures (formal): 2mm/36P standard/ctxreg, dual dataset.

    Left block: Iowa 3T (ref HC). Right block: Stanford (ref NTHC, always
    standard-branch TM). Both stat columns are area % of hippocampus:
    age+sex-adjusted OLS replaces the former vertex-count column; the
    unadjusted Welch column is kept. These are the canonical
    standard/ctxreg products and carry no _ttest suffix.
    """
    repo_root = Path(repo_root).resolve()
    data_root = Path(data_root).resolve()
    out_dir = Path(out_dir).resolve()
    cohorts = cohorts_for_repo(repo_root)
    stan_cohort_list = stanford_cohorts()
    all_matching_roots = branch_matching_roots(data_root)
    matching_roots = {
        branch: all_matching_roots[branch] for branch in selected_branches
    }
    surface_paths = branch_surface_paths(data_root)
    stan_surface_paths = {
        "spec": STANFORD_SPEC,
        "surf_l": STANFORD_SURF_L,
        "surf_r": STANFORD_SURF_R,
    }
    scene_path = repo_root / "config" / "wb_locked_native_view_lateral_medial.scene"
    required_inputs = {"scene": scene_path, **surface_paths, **{f"stan_{k}": v for k, v in stan_surface_paths.items()}}
    for label, path in required_inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing branch-render input {label}: {path}")

    iowa_demo = load_iowa_age_sex(repo_root)
    stan_demo = load_stanford_age_sex(repo_root)
    for cohort in cohorts:
        for sub in cohort["subjects"]:
            if sub not in iowa_demo:
                raise ValueError(f"Missing Iowa demographics for {sub}")
    for cohort in stan_cohort_list:
        for sub in cohort["subjects"]:
            if sub not in stan_demo:
                raise ValueError(f"Missing Stanford demographics for {sub}")
    stan_names = load_branch_subregion_names(
        "standard",
        data_root=data_root,
        cohorts=stan_cohort_list,
        subjects_root=STANFORD_TM_ROOT,
    )

    def _render_rows(
        cohort_list: list[dict[str, Any]],
        sub_root: Path,
        data_branch: str,
        surfaces: dict[str, Path],
        scratch_dir: Path,
        panels_dir: Path,
        names: dict[int, str],
    ) -> list[CohortRowResult]:
        rows: list[CohortRowResult] = []
        for cohort in cohort_list:
            cohort_key = cohort["key"]
            print(
                f"  Rendering cohort {cohort_key} "
                f"({cohort['name']}, N={len(cohort['subjects'])})..."
            )
            dominant_labels, dominance_pcts, stats = compute_cohort_data(
                smoothing="2mm",
                subjects=cohort["subjects"],
                subjects_root=sub_root,
                signal_branch=data_branch,
            )
            atlas_png, heatmap_png = render_cohort_panels(
                cohort_key=cohort_key,
                smoothing="2mm",
                variant="36p",
                dominant_labels=dominant_labels,
                dominance_pcts=dominance_pcts,
                panels_dir=panels_dir,
                scratch_dir=scratch_dir,
                surf_l=surfaces["surf_l"],
                surf_r=surfaces["surf_r"],
                spec_path=surfaces["spec"],
                scene_path=scene_path,
                force_rerender_heatmaps=force_heatmaps,
                force_rerender_panels=force_rerender_panels,
                signal_branch=data_branch,
                subregion_names=names,
            )
            rows.append(
                CohortRowResult(
                    cohort_key=cohort_key,
                    cohort_name=cohort["name"],
                    cohort_badge=cohort["badge"],
                    badge_color=cohort["badge_color"],
                    bg_color=cohort["bg_color"],
                    desc=cohort["desc"],
                    n_subjects=len(cohort["subjects"]),
                    dominant_atlas_png=atlas_png,
                    dominance_heatmap_png=heatmap_png,
                    dominant_counts=stats["dominant_counts"],
                    dominant_percentages=stats["dominant_percentages"],
                    mean_dominance_pct=stats["mean_dominance_pct"],
                    median_dominance_pct=stats["median_dominance_pct"],
                    min_dominance_pct=stats["min_dominance_pct"],
                    max_dominance_pct=stats["max_dominance_pct"],
                    subregion_names=dict(names),
                )
            )
        return rows

    out_dir.mkdir(parents=True, exist_ok=True)
    panels_dir = out_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    ttest_cache = out_dir / "ttest_cache" / "vertex_areas"
    ttest_fig_dir = out_dir / "panels_ttest"
    rendered_figures: dict[str, Path] = {}
    contrast_csvs: dict[str, Path] = {}

    with tempfile.TemporaryDirectory(prefix="hipp_dominant_subregions_ttest_") as tmp:
        scratch_dir = Path(tmp)
        for signal_branch in selected_branches:
            sub_root = matching_roots[signal_branch]
            subregion_names = load_branch_subregion_names(
                signal_branch,
                data_root=data_root,
                cohorts=cohorts,
            )
            print("\n==================================================")
            print(f"Processing dual-dataset branch: 2mm 36P {signal_branch}")
            print(f"Iowa matching root: {sub_root}")
            print(f"Stanford matching root: {STANFORD_TM_ROOT} (standard)")
            print("==================================================")

            iowa_results = _render_rows(
                cohorts, sub_root, signal_branch, surface_paths,
                scratch_dir, panels_dir, subregion_names,
            )
            stan_results = _render_rows(
                stan_cohort_list, STANFORD_TM_ROOT, "standard", stan_surface_paths,
                scratch_dir, panels_dir, stan_names,
            )

            iowa_per: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]] = {}
            for cohort in cohorts:
                iowa_per[cohort["key"]] = compute_subject_counts_areas(
                    smoothing="2mm",
                    subjects=cohort["subjects"],
                    subjects_root=sub_root,
                    cache_dir=ttest_cache,
                    signal_branch=signal_branch,
                    cache_namespace="",
                )
            stan_per: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]] = {}
            for cohort in stan_cohort_list:
                stan_per[cohort["key"]] = compute_subject_counts_areas(
                    smoothing="2mm",
                    subjects=cohort["subjects"],
                    subjects_root=STANFORD_TM_ROOT,
                    cache_dir=ttest_cache,
                    signal_branch="standard",
                    hippunfold_batch_root=STANFORD_HIPPUNFOLD_BATCH_ROOT,
                    cache_namespace="stanford",
                )
            iowa_tests = area_pct_dual_tests(iowa_per, iowa_demo, TTEST_REF, TTEST_CONTRASTS)
            stan_tests = area_pct_dual_tests(stan_per, stan_demo, STAN_REF, STAN_CONTRASTS)

            iowa_panels: dict[str, dict[str, Path]] = {}
            for cr in iowa_results:
                if cr.cohort_key == TTEST_REF:
                    iowa_panels[cr.cohort_key] = {}
                    continue
                iowa_panels[cr.cohort_key] = {}
                for mode in ("adj", "welch"):
                    panel_path = (
                        ttest_fig_dir
                        / f"dual_2mm_36p_{signal_branch}_iowa_{cr.cohort_key}_{mode}.png"
                    )
                    render_dual_areapct_panel(
                        dataset_label="Iowa",
                        ref_key=TTEST_REF,
                        ref_badge="HC",
                        contrast_key=cr.cohort_key,
                        contrast_badge=cr.cohort_badge,
                        contrast_color=cr.badge_color,
                        per_cohort=iowa_per,
                        tests=iowa_tests,
                        mode=mode,
                        out_path=panel_path,
                    )
                    iowa_panels[cr.cohort_key][mode] = panel_path
            stan_panels: dict[str, dict[str, Path]] = {}
            for cr in stan_results:
                if cr.cohort_key == STAN_REF:
                    stan_panels[cr.cohort_key] = {}
                    continue
                stan_panels[cr.cohort_key] = {}
                for mode in ("adj", "welch"):
                    panel_path = (
                        ttest_fig_dir
                        / f"dual_2mm_36p_{signal_branch}_stan_{cr.cohort_key}_{mode}.png"
                    )
                    render_dual_areapct_panel(
                        dataset_label="Stanford",
                        ref_key=STAN_REF,
                        ref_badge="NTHC",
                        contrast_key=cr.cohort_key,
                        contrast_badge=cr.cohort_badge,
                        contrast_color=cr.badge_color,
                        per_cohort=stan_per,
                        tests=stan_tests,
                        mode=mode,
                        out_path=panel_path,
                    )
                    stan_panels[cr.cohort_key][mode] = panel_path
            contrast_csv = (
                out_dir
                / f"template_matching_dominant_subregions_2mm_36p_{signal_branch}_contrasts.csv"
            )
            write_dual_contrasts_csv(
                iowa_tests,
                stan_tests,
                contrast_csv,
                grid_tag="2mm_36p",
                signal_branch=signal_branch,
            )
            contrast_csvs[signal_branch] = contrast_csv

            figure_path = (
                out_dir
                / f"template_matching_dominant_subregions_2mm_36p_{signal_branch}.png"
            )
            compose_dual_dataset_figure(
                output_png=figure_path,
                iowa_results=iowa_results,
                stan_results=stan_results,
                iowa_panels=iowa_panels,
                stan_panels=stan_panels,
                iowa_tests=iowa_tests,
                stan_tests=stan_tests,
                signal_branch=signal_branch,
                subregion_names=subregion_names,
            )
            rendered_figures[signal_branch] = figure_path

    branches_tag = "__".join(selected_branches)
    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "script": str(Path(__file__).resolve()),
        "mode": "branch-dual-dataset",
        "test": "area_pct = subregion area as % of subject's own hemisphere total midthickness area. Welch: two-sided, uncorrected (no FDR). Adj: OLS y ~ 1 + group + age + female, uncorrected (no FDR); bars/dots show raw area %. Iowa ref HC (N=67); Stanford ref NTHC (N=51), standard-branch TM in both figures. Stanford sex: female=1 iff Gender==2 (single Gender==3 record coded 0, N unchanged). cohens_d = diff/sqrt((sa^2+sb^2)/2) on raw values.",
        "stars": "+ 0.05<=p<0.10 (trend), * p<0.05, ** p<0.01, *** p<0.001",
        "signal_branches": sorted(rendered_figures),
        "master_figures": {
            branch: str(path.resolve()) for branch, path in rendered_figures.items()
        },
        "contrast_csvs": {
            branch: str(path.resolve()) for branch, path in contrast_csvs.items()
        },
    }
    manifest_path = (
        out_dir
        / f"template_matching_dominant_subregions_2mm_36p_{branches_tag}_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved branch t-test manifest: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render Iowa 3T Hippocampal Template Matching Dominant Subregions & Dominance Heatmaps (4 Grids)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "scripts"
        / "visualize"
        / "template_matching_dominant_subregions",
        help="Directory to save output figures and stats.",
    )
    parser.add_argument(
        "--grids",
        nargs="+",
        default=None,
        choices=["2mm_36p", "2mm_22p", "4mm_36p", "4mm_22p", "all"],
        help="Grid combinations to render.",
    )
    parser.add_argument(
        "--signal-branch",
        choices=("all", "standard", "ctxreg"),
        default=None,
        help="Render explicit 2mm/36P signal branches instead of the legacy grid mode.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Formal repository root for manifests, scenes, and code.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT,
        help="Explicit data root containing outputs_iowa_jj_tms.",
    )
    parser.add_argument(
        "--clean-cache",
        action="store_true",
        help="Force re-rendering of all intermediate panel composites.",
    )
    parser.add_argument(
        "--force-heatmaps",
        action="store_true",
        help="Force re-rendering of dominance heatmap panels with updated contrast settings.",
    )
    args = parser.parse_args()

    if args.signal_branch is not None:
        if args.grids is not None:
            parser.error("--signal-branch cannot be combined with --grids")
        selected_branches = (
            ("standard", "ctxreg")
            if args.signal_branch == "all"
            else (args.signal_branch,)
        )
        render_branch_outputs_ttest(
            selected_branches=selected_branches,
            out_dir=args.output_dir,
            repo_root=args.repo_root,
            data_root=args.data_root,
            force_heatmaps=args.clean_cache or args.force_heatmaps,
            force_rerender_panels=args.clean_cache,
        )
        print(f"\nCompleted {len(selected_branches)} t-test branch figure(s).")
        return

    grids = args.grids or ["2mm_36p", "2mm_22p", "4mm_36p", "4mm_22p"]
    selected_grids = (
        ["2mm_36p", "2mm_22p", "4mm_36p", "4mm_22p"] if "all" in grids else grids
    )

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    panels_dir = out_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    force_heatmaps = args.clean_cache or args.force_heatmaps

    grid_roots = {
        "36p": DEFAULT_36P_ROOT,
        "22p": DEFAULT_22P_ROOT,
    }

    all_results: dict[str, list[CohortRowResult]] = {}
    rendered_figures: dict[str, Path] = {}

    with tempfile.TemporaryDirectory(prefix="hipp_dominant_subregions_") as tmp:
        scratch_dir = Path(tmp)

        for grid_tag in selected_grids:
            smoothing, variant = grid_tag.split("_")
            sub_root = grid_roots[variant]
            print("\n==================================================")
            print(f"Processing Grid Combination: {grid_tag} ({smoothing}, {variant})")
            print("==================================================")

            cohort_results = []
            for c in COHORTS_DEF:
                ckey = c["key"]
                print(
                    f"  Rendering cohort {ckey} ({c['name']}, N={len(c['subjects'])})..."
                )

                dom_labels, dom_pcts, stats = compute_cohort_data(
                    smoothing=smoothing,
                    subjects=c["subjects"],
                    subjects_root=sub_root,
                )

                atlas_png, heatmap_png = render_cohort_panels(
                    cohort_key=ckey,
                    smoothing=smoothing,
                    variant=variant,
                    dominant_labels=dom_labels,
                    dominance_pcts=dom_pcts,
                    panels_dir=panels_dir,
                    scratch_dir=scratch_dir,
                    force_rerender_heatmaps=force_heatmaps,
                )

                cohort_results.append(
                    CohortRowResult(
                        cohort_key=ckey,
                        cohort_name=c["name"],
                        cohort_badge=c["badge"],
                        badge_color=c["badge_color"],
                        bg_color=c["bg_color"],
                        desc=c["desc"],
                        n_subjects=len(c["subjects"]),
                        dominant_atlas_png=atlas_png,
                        dominance_heatmap_png=heatmap_png,
                        dominant_counts=stats["dominant_counts"],
                        dominant_percentages=stats["dominant_percentages"],
                        mean_dominance_pct=stats["mean_dominance_pct"],
                        median_dominance_pct=stats["median_dominance_pct"],
                        min_dominance_pct=stats["min_dominance_pct"],
                        max_dominance_pct=stats["max_dominance_pct"],
                    )
                )

            all_results[grid_tag] = cohort_results

            # ---- TTEST extension: subject-level hemi counts/areas + Welch vs HC ----
            ttest_cache = out_dir / "ttest_cache" / "vertex_areas"
            ttest_fig_dir = out_dir / "panels_ttest"
            per_cohort: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]] = {}
            for c in COHORTS_DEF:
                per_cohort[c["key"]] = compute_subject_counts_areas(
                    smoothing=smoothing,
                    subjects=c["subjects"],
                    subjects_root=sub_root,
                    cache_dir=ttest_cache,
                )
            ttest_res = welch_vs_hc(per_cohort)
            ttest_panels: dict[str, dict[str, Path]] = {}
            for cr in cohort_results:
                if cr.cohort_key == TTEST_REF:
                    ttest_panels[cr.cohort_key] = {}
                    continue
                ttest_panels[cr.cohort_key] = {}
                for metric in TTEST_METRICS:
                    panel_path = (
                        ttest_fig_dir / f"ttest_{grid_tag}_{cr.cohort_key}_{metric}.png"
                    )
                    render_ttest_panel(
                        contrast_key=cr.cohort_key,
                        contrast_badge=cr.cohort_badge,
                        contrast_color=cr.badge_color,
                        metric=metric,
                        per_cohort=per_cohort,
                        ttest=ttest_res,
                        out_path=panel_path,
                    )
                    ttest_panels[cr.cohort_key][metric] = panel_path
            write_ttest_contrasts_csv(
                ttest_res,
                out_dir
                / f"template_matching_dominant_subregions_{grid_tag}_ttest_contrasts.csv",
                grid_tag=grid_tag,
            )

            # Compose TTEST Grid Figure (4 data columns + meta)
            grid_png = (
                out_dir / f"template_matching_dominant_subregions_{grid_tag}_ttest.png"
            )
            print(f"  Composing t-test figure for {grid_tag} -> {grid_png.name}...")
            compose_grid_figure_ttest(
                smoothing=smoothing,
                variant=variant,
                cohort_results=cohort_results,
                output_png=grid_png,
                ttest_panels=ttest_panels,
                ttest=ttest_res,
            )
            rendered_figures[grid_tag] = grid_png

        # Write Summary CSV (ttest-suffixed; never overwrite canonical files)
        csv_path = out_dir / "template_matching_dominant_subregions_ttest_stats.csv"
        write_comprehensive_csv(all_results=all_results, csv_path=csv_path)

        # Write Run Manifest (ttest-suffixed; never overwrite canonical files)
        manifest_path = (
            out_dir / "template_matching_dominant_subregions_ttest_manifest.json"
        )
        write_manifest_json(
            all_results=all_results,
            rendered_figures=rendered_figures,
            manifest_path=manifest_path,
        )

    print("\nAll 4 grid figures and statistics completed successfully!")


if __name__ == "__main__":
    main()

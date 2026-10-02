"""HC (67) vs patient (26) group-difference screen, age/sex-adjusted.

Scope (user-confirmed): all 197 declared clinical fields.
Method: rank-based ANCOVA analogue consistent with the primary family.
  - rank(y) and rank(age); design0 = [1, age_rank, sex].
  - residualize rank(y) and group dummy (HC=0, patient=1) on design0.
  - observed partial rank rho = corr(resid_y, resid_g).
  - Freedman-Lane style null: fit reduced rank(y) ~ design0, permute
    residuals, y* = fitted + permuted resid, recompute partial rho.
    Two-sided p = (1 + extreme) / (draws + 1), draws=10000.
  - Unadjusted Mann-Whitney U p and medians reported as context only;
    selection uses the adjusted p.

Eligibility mirrors the primary pipeline (fail closed, no substitutes):
  - nih_* fields: only nih_timing_eligible subjects enter the model.
  - all other fields: complete cases for that field.
  - require n_hc_valid >= 5, n_pat_valid >= 5, total >= 20,
    y variance > tol in the analysis sample; else not_estimable.
  - MADRS recorded totals are screened (they exist as fields) but flagged
    blocked_score_mismatch per pipeline policy; selection downstream
    should exclude them from inference.

Reads only manifest-declared results/*.csv.
Writes clinical_associations_group_screen.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT_NAME = "clinical_associations_group_screen.csv"
PERMUTATIONS = 10000
SEED = 20260917
TOL = 1e-12


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranked = np.empty(values.size, dtype=float)
    starts = np.r_[0, np.flatnonzero(values[order][1:] != values[order][:-1]) + 1]
    stops = np.r_[starts[1:], values.size]
    for start, stop in zip(starts, stops):
        ranked[order[start:stop]] = (start + 1 + stop) / 2.0
    return ranked


def _residualize(values: np.ndarray, design: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coef, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
    fitted = design @ coef
    return values - fitted, fitted


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    left = left - left.mean()
    right = right - right.mean()
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    scale = max(np.linalg.norm(left), np.linalg.norm(right), 1.0)
    if denom <= TOL * scale:
        return float("nan")
    return float(np.dot(left, right) / denom)


def screen_field(y: np.ndarray, group: np.ndarray, age: np.ndarray,
                 sex: np.ndarray, seed: int,
                 draws: int = PERMUTATIONS) -> dict:
    """Adjusted group-difference test for one field (complete cases in)."""
    n = y.size
    if n < 20 or np.ptp(y) <= TOL:
        return {"rho_adj": float("nan"), "p_adj": float("nan"),
                "status": "constant_or_small_n",
                "reason": f"n={n} or y constant"}
    uniq_g, counts = np.unique(group, return_counts=True)
    if set(np.unique(group)) - {0.0, 1.0}:
        return {"rho_adj": float("nan"), "p_adj": float("nan"),
                "status": "bad_group_coding", "reason": "groups must be HC=0/patient=1"}
    cnt = dict(zip(uniq_g.tolist(), counts.tolist()))
    if cnt.get(0.0, 0) < 5 or cnt.get(1.0, 0) < 5:
        return {"rho_adj": float("nan"), "p_adj": float("nan"),
                "status": "insufficient_per_group",
                "reason": f"hc={cnt.get(0.0,0)}, pat={cnt.get(1.0,0)}"}
    yr, ar = _rank(y), _rank(age)
    design0 = np.column_stack([np.ones(n), ar, sex])
    if np.linalg.matrix_rank(design0) < design0.shape[1]:
        return {"rho_adj": float("nan"), "p_adj": float("nan"),
                "status": "rank_deficient", "reason": "nuisance design rank-deficient"}
    resid_y, fitted_y = _residualize(yr, design0)
    resid_g, _ = _residualize(group, design0)
    rho = _corr(resid_y, resid_g)
    if not np.isfinite(rho):
        return {"rho_adj": rho, "p_adj": float("nan"),
                "status": "no_residual_variance",
                "reason": "no residual variance after age/sex"}
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(draws):
        perm = rng.permutation(n)
        y_star = fitted_y + resid_y[perm]
        # y_star is on the rank scale; residualize directly (no rerank:
        # fitted + permuted resid already preserves the reduced model).
        r_star, _ = _residualize(y_star, design0)
        rho_star = _corr(r_star, resid_g)
        if np.isfinite(rho_star) and abs(rho_star) >= abs(rho) - TOL:
            extreme += 1
    p = float((1 + extreme) / (draws + 1))
    return {"rho_adj": rho, "p_adj": p, "status": "ok",
            "reason": f"{draws} Freedman-Lane permutations"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--draws", type=int, default=PERMUTATIONS)
    args = ap.parse_args()

    subj_path = RESULTS / "clinical_associations_subjects.csv"
    scales_path = RESULTS / "clinical_associations_scales.csv"
    for p in (subj_path, scales_path):
        if not p.is_file():
            raise FileNotFoundError(f"missing required input: {p}")
    subjects = pd.read_csv(subj_path)
    scales = pd.read_csv(scales_path)
    if set(subjects["group"].unique()) - {0, 1}:
        raise ValueError("subjects.group must use HC=0 and patient=1")

    out_path = RESULTS / OUT_NAME
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"exists, pass --overwrite to replace: {out_path}")

    rows = []
    for idx, srow in scales.iterrows():
        field = str(srow["field"])
        eligible = subjects["nih_timing_eligible"].to_numpy(bool) \
            if field.startswith("nih_") else np.ones(len(subjects), bool)
        frame = subjects.loc[eligible, ["group", "age", "sex", field]].copy()
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
        frame = frame.dropna(subset=[field, "group", "age", "sex"])
        y = frame[field].to_numpy(float)
        g = frame["group"].to_numpy(float)
        age = frame["age"].to_numpy(float)
        sex = frame["sex"].to_numpy(float)
        n_hc = int((g == 0).sum())
        n_pat = int((g == 1).sum())
        res = screen_field(y, g, age, sex, seed=SEED + int(idx),
                           draws=int(args.draws))
        # Unadjusted context only.
        try:
            mwu_p = float(mannwhitneyu(y[g == 0], y[g == 1],
                                       alternative="two-sided").pvalue) \
                if n_hc >= 1 and n_pat >= 1 and np.ptp(y) > TOL else float("nan")
        except ValueError:
            mwu_p = float("nan")
        rows.append({
            "field": field,
            "scale": str(srow["scale"]),
            "level": str(srow["level"]),
            "label": str(srow["label"]),
            "score_status": str(srow.get("score_status", "")),
            "nih_restricted": bool(field.startswith("nih_")),
            "n_hc_valid": n_hc,
            "n_pat_valid": n_pat,
            "n_total_valid": n_hc + n_pat,
            "median_hc": float(np.median(y[g == 0])) if n_hc else float("nan"),
            "median_pat": float(np.median(y[g == 1])) if n_pat else float("nan"),
            "rho_group_adj": res["rho_adj"],
            "p_group_adj": res["p_adj"],
            "p_mwu_unadj": mwu_p,
            "status": res["status"],
            "reason": res["reason"],
        })
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    ok = out[out.status == "ok"]
    sig = ok[ok.p_group_adj < 0.05].copy()
    # MADRS recorded totals are quarantined from inference by pipeline policy.
    sig_infer = sig[~((sig.scale == "MADRS-SR") & (sig.level == "total"))]
    print(f"screen: {len(out)} fields, {len(ok)} estimable, "
          f"{len(sig)} raw p<0.05, {len(sig_infer)} inferable "
          f"(MADRS totals quarantined: {len(sig) - len(sig_infer)}).")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

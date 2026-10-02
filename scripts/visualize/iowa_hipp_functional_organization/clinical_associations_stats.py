"""Bounded rank-based clinical association statistics.

The public ``analyze`` function implements the pre-specified Iowa analysis:
partial Spearman association, Freedman--Lane permutation inference, stratified
subject bootstrap confidence intervals, leave-one-out sensitivity, and an
optional incremental R-squared for an area-controlled model.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


_TOL = 1e-12
_RESULT_KEYS = (
    "n", "rho", "ci_low", "ci_high", "p", "status", "reason",
    "loo_min", "loo_max", "loo_sign_flip", "delta_r2", "n_boot_valid",
)


def _rank(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks, including average ranks for ties."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranked = np.empty(values.size, dtype=float)
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    stops = np.r_[starts[1:], values.size]
    for start, stop in zip(starts, stops):
        ranked[order[start:stop]] = (start + 1 + stop) / 2.0
    return ranked


def _base_result(n: int) -> dict:
    return {
        "n": int(n), "rho": np.nan, "ci_low": np.nan, "ci_high": np.nan,
        "p": np.nan, "status": "not_estimable", "reason": "",
        "loo_min": np.nan, "loo_max": np.nan, "loo_sign_flip": False,
        "delta_r2": np.nan, "n_boot_valid": 0,
    }


def _design(age_rank: np.ndarray, sex: np.ndarray, groups: np.ndarray,
            control_rank: np.ndarray | None, adjusted: bool) -> np.ndarray:
    n = age_rank.size
    columns = [np.ones(n)]
    if adjusted:
        columns.extend((age_rank, sex))
        unique_groups = np.unique(groups)
        if unique_groups.size > 1:
            # The approved design has one binary patient indicator.
            columns.append((groups == unique_groups[-1]).astype(float))
        if control_rank is not None:
            columns.append(control_rank)
    return np.column_stack(columns)


def _residualize(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    return values - design @ coefficients


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = left - np.mean(left)
    right = right - np.mean(right)
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    scale = max(np.linalg.norm(left), np.linalg.norm(right), 1.0)
    if denom <= _TOL * scale:
        return np.nan
    return float(np.dot(left, right) / denom)


def _fit_rank_model(x: np.ndarray, y: np.ndarray, age: np.ndarray,
                    sex: np.ndarray, groups: np.ndarray,
                    control: np.ndarray | None, adjusted: bool) -> dict:
    xr, yr, ar = _rank(x), _rank(y), _rank(age)
    cr = None if control is None else _rank(control)
    design = _design(ar, sex, groups, cr, adjusted)
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return {"valid": False, "reason": "rank-deficient nuisance design"}
    residual_x = _residualize(xr, design)
    residual_y = _residualize(yr, design)
    rho = _correlation(residual_x, residual_y)
    if not np.isfinite(rho):
        return {"valid": False, "reason": "no residual variance", "rho": rho}

    ss_total = float(np.sum((yr - np.mean(yr)) ** 2))
    delta_r2 = np.nan
    if ss_total > _TOL:
        reduced_error = float(np.sum(residual_y ** 2))
        full = np.column_stack((design, xr))
        if np.linalg.matrix_rank(full) == full.shape[1]:
            full_error = float(np.sum(_residualize(yr, full) ** 2))
            delta_r2 = max(0.0, (reduced_error - full_error) / ss_total)
    return {
        "valid": True, "rho": rho, "design": design, "xr": xr, "yr": yr,
        "residual_x": residual_x, "residual_y": residual_y,
        "fitted_y": yr - residual_y, "delta_r2": delta_r2,
    }


def _permutation_p(fit: dict, groups: np.ndarray, adjusted: bool,
                   draws: int, rng: np.random.Generator) -> float:
    if draws <= 0:
        return np.nan
    n = groups.size
    permutations = np.empty((draws, n), dtype=int)
    base = np.arange(n)
    if adjusted and np.unique(groups).size > 1:
        strata = [np.flatnonzero(groups == group) for group in np.unique(groups)]
        for row in permutations:
            row[:] = base
            for indices in strata:
                row[indices] = rng.permutation(indices)
    else:
        for row in permutations:
            row[:] = rng.permutation(base)

    outcomes = fit["fitted_y"][None, :] + fit["residual_y"][permutations]
    design = fit["design"]
    coefficients = np.linalg.lstsq(design, outcomes.T, rcond=None)[0]
    residuals = outcomes - (design @ coefficients).T
    fixed = fit["residual_x"] - np.mean(fit["residual_x"])
    residuals -= np.mean(residuals, axis=1, keepdims=True)
    denominators = np.linalg.norm(fixed) * np.linalg.norm(residuals, axis=1)
    correlations = residuals @ fixed / denominators
    extreme = np.count_nonzero(np.abs(correlations) >= abs(fit["rho"]) - _TOL)
    return float((1 + extreme) / (draws + 1))


def _stratified_indices(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pieces = []
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        pieces.append(rng.choice(indices, size=indices.size, replace=True))
    return np.concatenate(pieces)


def _bootstrap_rhos(x: np.ndarray, y: np.ndarray, age: np.ndarray,
                    sex: np.ndarray, groups: np.ndarray,
                    control: np.ndarray | None, adjusted: bool, draws: int,
                    rng: np.random.Generator, batch_size: int = 256) -> np.ndarray:
    """Rerank and refit stratified subject bootstraps in bounded batches."""
    if draws <= 0:
        return np.empty(0, dtype=float)
    strata = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    group_levels = np.unique(groups)
    collected = []
    for start in range(0, draws, batch_size):
        count = min(batch_size, draws - start)
        sampled = np.concatenate([
            rng.choice(indices, size=(count, indices.size), replace=True)
            for indices in strata
        ], axis=1)
        xr = rankdata(x[sampled], axis=1, method="average")
        yr = rankdata(y[sampled], axis=1, method="average")
        ar = rankdata(age[sampled], axis=1, method="average")
        columns = [np.ones_like(xr)]
        if adjusted:
            columns.extend((ar, sex[sampled]))
            sampled_groups = groups[sampled]
            if group_levels.size > 1:
                columns.append((sampled_groups == group_levels[-1]).astype(float))
            if control is not None:
                columns.append(rankdata(control[sampled], axis=1, method="average"))
        design = np.stack(columns, axis=2)
        valid_design = np.linalg.matrix_rank(design) == design.shape[2]
        if not np.any(valid_design):
            continue
        dv = design[valid_design]
        xrv = xr[valid_design]
        yrv = yr[valid_design]
        projection = np.linalg.pinv(dv)
        residual_x = xrv - np.squeeze(dv @ (projection @ xrv[..., None]), axis=2)
        residual_y = yrv - np.squeeze(dv @ (projection @ yrv[..., None]), axis=2)
        residual_x -= residual_x.mean(axis=1, keepdims=True)
        residual_y -= residual_y.mean(axis=1, keepdims=True)
        norm_x = np.linalg.norm(residual_x, axis=1)
        norm_y = np.linalg.norm(residual_y, axis=1)
        denominator = norm_x * norm_y
        scale = np.maximum.reduce((norm_x, norm_y, np.ones(norm_x.size)))
        valid_variance = denominator > _TOL * scale
        numerator = np.sum(residual_x * residual_y, axis=1)
        collected.extend((numerator[valid_variance]
                          / denominator[valid_variance]).tolist())
    return np.asarray(collected, dtype=float)


def _refit_rho(indices: np.ndarray, x: np.ndarray, y: np.ndarray,
               age: np.ndarray, sex: np.ndarray, groups: np.ndarray,
               control: np.ndarray | None, adjusted: bool) -> float:
    sampled_control = None if control is None else control[indices]
    fitted = _fit_rank_model(x[indices], y[indices], age[indices], sex[indices],
                             groups[indices], sampled_control, adjusted)
    return float(fitted["rho"]) if fitted.get("valid") else np.nan


def analyze(x, y, age, sex, groups, *, adjusted=True, control=None,
            permutations=10000, bootstraps=2000, seed=20260917, min_n=20,
            loo=True) -> dict:
    """Analyze one imaging metric / clinical-score association.

    Missing and nonfinite observations are removed jointly for this model.
    Resampling reranks all continuous variables within each resample.
    """
    named = {"x": x, "y": y, "age": age, "sex": sex, "groups": groups}
    arrays = {}
    for name, values in named.items():
        array = np.asarray(values, dtype=float)
        if array.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        arrays[name] = array
    lengths = {array.size for array in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("x, y, age, sex, and groups must have equal lengths")
    control_array = None
    if control is not None:
        control_array = np.asarray(control, dtype=float)
        if control_array.ndim != 1 or control_array.size != arrays["x"].size:
            raise ValueError("control must be one-dimensional and match x")

    complete = np.ones(arrays["x"].size, dtype=bool)
    for array in arrays.values():
        complete &= np.isfinite(array)
    if control_array is not None:
        complete &= np.isfinite(control_array)
        control_array = control_array[complete]
    x_array, y_array, age_array, sex_array, group_array = (
        arrays[name][complete] for name in ("x", "y", "age", "sex", "groups")
    )
    result = _base_result(x_array.size)
    if np.unique(sex_array).size > 2:
        raise ValueError("sex must be a binary numeric variable")
    if not np.all(np.isin(np.unique(group_array), [0.0, 1.0])):
        raise ValueError("groups must use HC=0 and Patient=1")
    if x_array.size < 2 or np.ptp(x_array) <= _TOL or np.ptp(y_array) <= _TOL:
        result.update(status="constant_input", reason="x or y is constant")
        return result

    if x_array.size < int(min_n):
        result.update(status="insufficient_n",
                      reason=f"n={x_array.size} is below min_n={int(min_n)}")
        return result

    fit = _fit_rank_model(x_array, y_array, age_array, sex_array, group_array,
                          control_array, bool(adjusted))
    if not fit.get("valid"):
        status = ("rank_deficient" if "rank-deficient" in fit["reason"]
                  else "no_residual_variance")
        result.update(rho=fit.get("rho", np.nan), status=status,
                      reason=fit["reason"])
        return result
    result.update(rho=fit["rho"], delta_r2=fit["delta_r2"])
    rng = np.random.default_rng(seed)
    result["p"] = _permutation_p(fit, group_array, bool(adjusted),
                                  int(permutations), rng)

    valid_boot = _bootstrap_rhos(
        x_array, y_array, age_array, sex_array, group_array, control_array,
        bool(adjusted), int(bootstraps), rng,
    )
    result["n_boot_valid"] = len(valid_boot)
    required = int(np.ceil(0.8 * int(bootstraps)))
    if bootstraps > 0 and len(valid_boot) >= required:
        result["ci_low"], result["ci_high"] = (
            float(value) for value in np.percentile(valid_boot, [2.5, 97.5])
        )

    if loo:
        loo_values = []
        for omitted in range(x_array.size):
            indices = np.r_[0:omitted, omitted + 1:x_array.size]
            value = _refit_rho(indices, x_array, y_array, age_array, sex_array,
                               group_array, control_array, bool(adjusted))
            if np.isfinite(value):
                loo_values.append(value)
        if loo_values:
            result["loo_min"] = float(np.min(loo_values))
            result["loo_max"] = float(np.max(loo_values))
            result["loo_sign_flip"] = bool(any(
                value * fit["rho"] < 0 for value in loo_values
            ))

    if bootstraps > 0 and len(valid_boot) < required:
        result.update(status="bootstrap_insufficient",
                      reason=(f"only {len(valid_boot)}/{int(bootstraps)} bootstrap "
                              "replicates were valid; at least 80% required"))
    else:
        reason = ("bootstrap not requested" if int(bootstraps) == 0 else
                  f"{len(valid_boot)}/{int(bootstraps)} bootstrap replicates valid")
        result.update(status="ok", reason=reason)
    return {key: result[key] for key in _RESULT_KEYS}


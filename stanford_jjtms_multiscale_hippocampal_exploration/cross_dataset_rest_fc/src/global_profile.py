"""Auxiliary omnibus distances for complete subject-level FC profiles.

This module is deliberately separate from the primary localisation engine.
For every fixed Task 2 branch it builds one row per normalized subject, keeps
all network cells for that subject together, and tests the exact Task 3
contrast with a Freedman--Lane residual permutation.  The resulting profile
tests are descriptive/auxiliary: no profile p-value is used by, or exposed to,
the primary localisation implementation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

try:  # Support both direct src imports and package imports.
    from .contrasts import (
        JJTMS_CELLS,
        JJTMS_ESTIMANDS,
        STANFORD_CELLS,
        STANFORD_ESTIMANDS,
    )
    from .localisation import (
        HEMISPHERE_MODES,
        REGIONS,
        _normalise_contrasts,
        _prepare_tensor,
        _task2_network_metadata,
        _validate_planned_components,
    )
except ImportError:  # pragma: no cover - exercised by focused tests
    from contrasts import JJTMS_CELLS, JJTMS_ESTIMANDS, STANFORD_CELLS, STANFORD_ESTIMANDS
    from localisation import (
        HEMISPHERE_MODES,
        REGIONS,
        _normalise_contrasts,
        _prepare_tensor,
        _task2_network_metadata,
        _validate_planned_components,
    )


OUTPUT_COLUMNS = [
    "dataset",
    "estimand",
    "method",
    "smoothing_mm",
    "hemisphere_mode",
    "network_resolution",
    "fc_estimator",
    "profile_scope",
    "region",
    "n_total",
    "n_complete",
    "effective_dimension",
    "statistic",
    "p_perm",
    "n_perm",
    "seed",
    "derived_seed",
    "failure_reason",
]

_BRANCH_COLUMNS = (
    "dataset",
    "method",
    "smoothing_mm",
    "hemisphere_mode",
    "resolution",
    "fc_estimator",
)
_PROFILE_SCOPES = (
    ("subregion_profile", "anterior"),
    ("subregion_profile", "middle"),
    ("subregion_profile", "posterior"),
    ("whole_hippocampus_profile", "all"),
)
_DATASET_ORDER = {"jjtms": 0, "stanford": 1}
_MODE_ORDER = {value: index for index, value in enumerate(HEMISPHERE_MODES)}
_REGION_ORDER = {value: index for index, value in enumerate(REGIONS)}


def _integer(value: Any, *, field: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{field} must be an integer")
    return int(value)


def _subject_key(value: Any) -> tuple[Any, ...]:
    text = str(value)
    suffix = text[4:] if text.casefold().startswith("sub-") else text
    if suffix.isdigit():
        return (0, int(suffix), text)
    return (1, suffix.casefold(), text)


def _scope_key(
    *,
    dataset: str,
    estimand: str,
    method: str,
    smoothing_mm: int,
    hemisphere_mode: str,
    resolution: int,
    fc_estimator: str,
    profile_scope: str,
    region: str,
) -> tuple[Any, ...]:
    return (
        str(dataset),
        str(estimand),
        str(method),
        int(smoothing_mm),
        str(hemisphere_mode),
        int(resolution),
        str(fc_estimator),
        str(profile_scope),
        str(region),
    )


def _derive_seed(seed: int, scope_key: tuple[Any, ...]) -> int:
    """Derive a branch-local uint64 seed without Python's process hash."""

    payload = json.dumps(
        {"seed": int(seed), "scope": list(scope_key)},
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big")


def _null_basis(contrast: np.ndarray) -> np.ndarray:
    """Return a deterministic basis for the exact contrast null space."""

    vector = np.asarray(contrast, dtype=float).reshape(-1)
    if vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("contrast coefficients must be finite and non-empty")
    pivot_candidates = np.flatnonzero(np.abs(vector) > np.finfo(float).eps)
    if pivot_candidates.size == 0:
        raise ValueError("contrast is the zero vector and is not estimable")
    pivot = int(pivot_candidates[0])
    basis = []
    for column in range(vector.size):
        if column == pivot:
            continue
        candidate = np.zeros(vector.size, dtype=float)
        candidate[column] = 1.0
        candidate[pivot] = -vector[column] / vector[pivot]
        basis.append(candidate)
    return np.column_stack(basis)


def _cell_design(labels: np.ndarray, cells: tuple[str, ...]) -> np.ndarray:
    design = np.zeros((labels.size, len(cells)), dtype=float)
    for index, cell in enumerate(cells):
        design[:, index] = labels == cell
    return design


def _fit_full_profile(
    values: np.ndarray,
    labels: np.ndarray,
    cells: tuple[str, ...],
    contrast: np.ndarray,
) -> tuple[dict[str, Any] | None, str]:
    """Fit a full cell-means model and return its profile distance."""

    if values.ndim != 2 or values.shape[0] != labels.size:
        return None, "profile matrix has invalid shape"
    if values.shape[0] == 0 or not np.isfinite(values).all():
        return None, "profile contains no finite complete subjects"
    design = _cell_design(labels, cells)
    rank = int(np.linalg.matrix_rank(design))
    if rank != len(cells):
        missing = [cell for cell in cells if not np.any(labels == cell)]
        if missing:
            return None, "missing required cell(s): " + ", ".join(missing)
        return None, "full cell-means design is not full rank"
    df_resid = int(values.shape[0] - rank)
    if df_resid <= 0:
        return None, "unestimable residual degrees of freedom"
    beta, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
    residual = values - design @ beta
    sse = np.sum(residual**2, axis=0)
    residual_sd = np.sqrt(sse / float(df_resid))
    if (
        not np.isfinite(residual_sd).all()
        or np.any(residual_sd <= 0.0)
        or not np.isfinite(sse).all()
    ):
        return None, "invalid full-model residual SD"
    effect = np.asarray(contrast, dtype=float) @ beta
    standardized = effect / residual_sd
    statistic = float(np.dot(standardized, standardized))
    if not np.isfinite(effect).all() or not np.isfinite(standardized).all():
        return None, "nonfinite contrast profile effect"
    if not np.isfinite(statistic):
        return None, "nonfinite profile-distance statistic"
    return {
        "statistic": statistic,
        "effect": effect,
        "residual_sd": residual_sd,
        "design": design,
        "beta": beta,
        "residual": residual,
        "df_resid": df_resid,
    }, ""


def _fit_reduced(
    values: np.ndarray,
    design: np.ndarray,
    contrast: np.ndarray,
) -> tuple[np.ndarray | None, str]:
    """Fit the reduced model whose coefficient space satisfies c beta == 0."""

    basis = _null_basis(contrast)
    reduced_design = design @ basis
    expected_rank = basis.shape[1]
    rank = int(np.linalg.matrix_rank(reduced_design))
    if rank != expected_rank:
        return None, "reduced design is rank deficient"
    if int(np.linalg.matrix_rank(design)) != design.shape[1]:
        return None, "full design is rank deficient"
    fitted = reduced_design @ np.linalg.lstsq(reduced_design, values, rcond=None)[0]
    if not np.isfinite(fitted).all():
        return None, "nonfinite reduced-model fit"
    return fitted, ""


def _failure_reasons(
    branch: pd.DataFrame,
    required: tuple[tuple[str, str], ...],
    subjects: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, str]:
    """Pivot complete subjects and retain deterministic audit reasons."""

    component_maps: dict[tuple[str, str], dict[str, pd.Series]] = {}
    for region, network in required:
        component = branch.loc[
            branch["region"].eq(region) & branch["network"].eq(network)
        ]
        component_maps[(region, network)] = {
            str(subject): row
            for subject, row in component.set_index("subject", drop=False).iterrows()
        }

    complete_rows: list[np.ndarray] = []
    complete_labels: list[str] = []
    reasons: list[str] = []
    for subject in subjects:
        subject_values: list[float] = []
        subject_reasons: list[str] = []
        subject_group: str | None = None
        for region, network in required:
            row = component_maps[(region, network)].get(subject)
            if row is None:
                subject_reasons.append(
                    f"missing component {region}/{network} for {subject}"
                )
                continue
            if subject_group is None:
                subject_group = str(row["group"])
            status = str(row["status"])
            value = float(row["fisher_z"]) if pd.notna(row["fisher_z"]) else np.nan
            if status != "present":
                reason = str(row["failure_reason"]).strip() or f"status={status}"
                subject_reasons.append(f"{subject}: {reason}")
            elif not np.isfinite(value):
                subject_reasons.append(f"nonfinite Fisher-z for {subject} at {region}/{network}")
            else:
                subject_values.append(value)
        if subject_reasons:
            reasons.extend(subject_reasons)
            continue
        if len(subject_values) != len(required):
            reasons.append(f"incomplete profile for {subject}")
            continue
        complete_rows.append(np.asarray(subject_values, dtype=float))
        if subject_group is None:
            reasons.append(f"missing group for {subject}")
            complete_rows.pop()
            continue
        complete_labels.append(subject_group)

    unique_reasons: list[str] = []
    for reason in reasons:
        if reason not in unique_reasons:
            unique_reasons.append(reason)
    return (
        np.asarray(complete_rows, dtype=float)
        if complete_rows
        else np.empty((0, len(required)), dtype=float),
        np.asarray(complete_labels, dtype=object),
        "; ".join(unique_reasons),
    )


def _profile_components(
    network_order: tuple[str, ...],
    profile_scope: str,
    region: str,
) -> tuple[tuple[str, str], ...]:
    if profile_scope == "subregion_profile":
        return tuple((region, network) for network in network_order)
    return tuple(
        (component_region, network)
        for component_region in REGIONS
        for network in network_order
    )


def _run_scope(
    *,
    branch: pd.DataFrame,
    branch_values: Mapping[str, Any],
    dataset: str,
    estimand: str,
    contrast: np.ndarray,
    cells: tuple[str, ...],
    network_order: tuple[str, ...],
    profile_scope: str,
    region: str,
    seed: int,
    n_perm: int,
) -> dict[str, Any]:
    resolution = int(branch_values["resolution"])
    required = _profile_components(network_order, profile_scope, region)
    subjects = tuple(sorted(branch["subject"].unique(), key=_subject_key))
    values, labels, missing_reason = _failure_reasons(branch, required, subjects)
    dimension = len(required)
    scope_key = _scope_key(
        dataset=dataset,
        estimand=estimand,
        method=str(branch_values["method"]),
        smoothing_mm=int(branch_values["smoothing_mm"]),
        hemisphere_mode=str(branch_values["hemisphere_mode"]),
        resolution=resolution,
        fc_estimator=str(branch_values["fc_estimator"]),
        profile_scope=profile_scope,
        region=region,
    )
    derived_seed = _derive_seed(seed, scope_key)
    output: dict[str, Any] = {
        "dataset": dataset,
        "estimand": estimand,
        "method": str(branch_values["method"]),
        "smoothing_mm": int(branch_values["smoothing_mm"]),
        "hemisphere_mode": str(branch_values["hemisphere_mode"]),
        "network_resolution": resolution,
        "fc_estimator": str(branch_values["fc_estimator"]),
        "profile_scope": profile_scope,
        "region": region,
        "n_total": int(len(subjects)),
        "n_complete": int(values.shape[0]),
        "effective_dimension": int(dimension),
        "statistic": np.nan,
        "p_perm": np.nan,
        "n_perm": int(n_perm),
        "seed": int(seed),
        "derived_seed": int(derived_seed),
        "failure_reason": str(missing_reason or ""),
    }
    if values.shape[0] == 0:
        output["failure_reason"] = output["failure_reason"] or "no complete subject profiles"
        return output

    full, full_failure = _fit_full_profile(values, labels, cells, contrast)
    if full is None:
        output["failure_reason"] = "; ".join(
            reason for reason in (output["failure_reason"], full_failure) if reason
        )
        return output
    output["statistic"] = float(full["statistic"])
    reduced_fit, reduced_failure = _fit_reduced(values, full["design"], contrast)
    if reduced_fit is None:
        output["statistic"] = np.nan
        output["failure_reason"] = "; ".join(
            reason for reason in (output["failure_reason"], reduced_failure) if reason
        )
        return output
    if n_perm == 0:
        output["p_perm"] = 1.0
        return output
    reduced_residual = values - reduced_fit
    rng = np.random.default_rng(derived_seed)
    count = 0
    permutation_failure = ""
    for _ in range(n_perm):
        permutation = rng.permutation(values.shape[0])
        permuted_values = reduced_fit + reduced_residual[permutation, :]
        permuted_fit, failure = _fit_full_profile(
            permuted_values, labels, cells, contrast
        )
        if permuted_fit is None:
            permutation_failure = failure
            continue
        if float(permuted_fit["statistic"]) >= float(full["statistic"]):
            count += 1
    if permutation_failure:
        output["failure_reason"] = "; ".join(
            reason
            for reason in (
                output["failure_reason"],
                f"invalid permutation fit: {permutation_failure}",
            )
            if reason
        )
        output["statistic"] = np.nan
        output["p_perm"] = np.nan
        return output
    output["p_perm"] = float((1.0 + count) / (n_perm + 1.0))
    return output


def _sort_result(result: pd.DataFrame) -> pd.DataFrame:
    if result.empty:
        return result.reindex(columns=OUTPUT_COLUMNS)
    result = result.copy()
    result["_dataset_order"] = result["dataset"].map(_DATASET_ORDER).fillna(99)
    result["_estimand_order"] = result["estimand"].map(
        {
            **{value: index for index, value in enumerate(JJTMS_ESTIMANDS)},
            **{value: index for index, value in enumerate(STANFORD_ESTIMANDS)},
        }
    ).fillna(99)
    result["_mode_order"] = result["hemisphere_mode"].map(_MODE_ORDER).fillna(99)
    result["_profile_order"] = result["profile_scope"].map(
        {"subregion_profile": 0, "whole_hippocampus_profile": 1}
    ).fillna(99)
    result["_region_order"] = result["region"].map(_REGION_ORDER).fillna(99)
    result["_resolution_order"] = result["network_resolution"].map({17: 0, 8: 1}).fillna(99)
    result = result.sort_values(
        [
            "_dataset_order",
            "_estimand_order",
            "method",
            "smoothing_mm",
            "_mode_order",
            "_resolution_order",
            "fc_estimator",
            "_profile_order",
            "_region_order",
        ],
        kind="stable",
    )
    return result.drop(
        columns=[
            "_dataset_order",
            "_estimand_order",
            "_mode_order",
            "_profile_order",
            "_region_order",
            "_resolution_order",
        ]
    ).reindex(columns=OUTPUT_COLUMNS).reset_index(drop=True)


def run_global_tests(
    tensor: pd.DataFrame,
    contrasts: pd.DataFrame,
    seed: int,
    n_perm: int,
) -> pd.DataFrame:
    """Run auxiliary subregion and whole-hippocampus profile tests.

    The statistic is the squared Euclidean norm of the exact Task 3 contrast
    effect after component-wise scaling by each component's full-model
    residual SD.  It is an auxiliary omnibus profile-distance statistic, not
    a confirmatory multiplicity-corrected endpoint.
    """

    seed = _integer(seed, field="seed")
    n_perm = _integer(n_perm, field="n_perm")
    if n_perm < 0:
        raise ValueError("n_perm must be non-negative")
    frame = _prepare_tensor(tensor)
    branches = _validate_planned_components(frame)
    metadata = _task2_network_metadata(frame)
    contrast_blocks = _normalise_contrasts(contrasts, set(frame["dataset"]))
    rows: list[dict[str, Any]] = []

    branch_items = sorted(
        branches.items(),
        key=lambda item: (
            _DATASET_ORDER.get(str(item[0][0]), 99),
            str(item[0][1]),
            int(item[0][2]),
            _MODE_ORDER.get(str(item[0][3]), 99),
            int(item[0][4]),
            str(item[0][5]),
        ),
    )
    for branch_key, branch in branch_items:
        branch_values = dict(zip(_BRANCH_COLUMNS, branch_key))
        dataset = str(branch_values["dataset"])
        resolution = int(branch_values["resolution"])
        method = str(branch_values["method"])
        frozen = metadata.get((method, resolution))
        if frozen is None:
            raise ValueError(
                f"missing Task 2 network metadata for method={method!r}, "
                f"resolution={resolution}"
            )
        network_order = tuple(sorted(frozen, key=lambda network: int(frozen[network])))
        cells = tuple(STANFORD_CELLS if dataset == "stanford" else JJTMS_CELLS)
        block = contrast_blocks[dataset]
        for estimand in block.index:
            contrast = block.loc[estimand].to_numpy(dtype=float)
            if contrast.size != len(cells):
                raise ValueError(
                    f"{dataset} contrast {estimand!r} has invalid cell dimension"
                )
            for profile_scope, region in _PROFILE_SCOPES:
                rows.append(
                    _run_scope(
                        branch=branch,
                        branch_values=branch_values,
                        dataset=dataset,
                        estimand=str(estimand),
                        contrast=contrast,
                        cells=cells,
                        network_order=network_order,
                        profile_scope=profile_scope,
                        region=region,
                        seed=seed,
                        n_perm=n_perm,
                    )
                )

    result = _sort_result(pd.DataFrame(rows, columns=OUTPUT_COLUMNS))
    result.attrs["model"] = (
        "full cell-means model with component-wise residual-SD-scaled "
        "squared Euclidean profile distance"
    )
    result.attrs["permutation"] = (
        "Freedman-Lane subject-row residual permutation under the exact "
        "contrast null; plus-one upper-tail p"
    )
    result.attrs["auxiliary"] = True
    result.attrs["multiplicity"] = "no confirmatory multiplicity correction"
    return result


__all__ = ["run_global_tests"]

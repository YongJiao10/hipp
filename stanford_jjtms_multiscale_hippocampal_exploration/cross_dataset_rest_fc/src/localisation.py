"""Primary region-by-network rest-FC localisation models.

The public entry point consumes the long-form hemisphere views produced by
``features.build_hemisphere_views`` and one of the frozen contrast matrices
from ``contrasts``.  It fits an unadjusted cell-means model independently for
every planned region/network component and returns HC3/t inference on the raw
scale.  Multiplicity correction is deliberately deferred until the final
analysis-object set is frozen.  Profile/global tests are intentionally outside
this module and cannot gate localisation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

try:  # Support both ``import localisation`` from src and package imports.
    from .contrasts import (
        JJTMS_CELLS,
        JJTMS_ESTIMANDS,
        STANFORD_CELLS,
        STANFORD_ESTIMANDS,
        jjtms_contrast_matrix,
        stanford_contrast_matrix,
    )
except ImportError:  # pragma: no cover - exercised by focused tests
    from contrasts import (
        JJTMS_CELLS,
        JJTMS_ESTIMANDS,
        STANFORD_CELLS,
        STANFORD_ESTIMANDS,
        jjtms_contrast_matrix,
        stanford_contrast_matrix,
    )


REGIONS = ("anterior", "body", "tail")
HEMISPHERE_MODES = ("bilateral", "L", "R", "L-R")
DATASET_ORDER = ("jjtms", "stanford")
OUTPUT_COLUMNS = [
    "dataset",
    "estimand",
    "resolution",
    "method",
    "smoothing_mm",
    "hemisphere_mode",
    "region",
    "network",
    "fc_estimator",
    "n",
    "coefficient",
    "hc3_se",
    "ci_low",
    "ci_high",
    "standardized_effect",
    "p_raw",
    "q_bh",
    "correction_status",
    "failure_reason",
]

_BRANCH_COLUMNS = [
    "dataset",
    "method",
    "smoothing_mm",
    "hemisphere_mode",
    "resolution",
    "fc_estimator",
]
_COMPONENT_COLUMNS = _BRANCH_COLUMNS + ["region", "network"]
_TASK2_STATUS_VOCABULARY = frozenset(
    {"present", "unavailable", "missing", "blocked", "failed"}
)
_SUCCESS_STATUSES = frozenset({"present"})
_FAILURE_STATUSES = frozenset(_TASK2_STATUS_VOCABULARY - _SUCCESS_STATUSES)


def _is_missing(value: Any) -> bool:
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _normalise_subject(value: Any) -> str:
    """Canonicalise numeric ``sub-*`` IDs while retaining synthetic IDs."""

    if _is_missing(value):
        raise ValueError("subject identifier is missing")
    text = str(value).strip()
    if not text:
        raise ValueError("subject identifier is empty")
    lowered = text.casefold()
    if lowered.startswith("sub-"):
        suffix = lowered[4:]
        if not suffix:
            raise ValueError(f"invalid subject identifier: {value!r}")
        if suffix.isdigit():
            return f"sub-{int(suffix):04d}"
        return f"sub-{suffix}"
    if text.isdigit():
        return f"sub-{int(text):04d}"
    return lowered


def _normalise_dataset(value: Any) -> str:
    if _is_missing(value):
        raise ValueError("dataset is missing")
    text = str(value).strip().casefold().replace("-", "_")
    if text == "jj_tms":
        text = "jjtms"
    if text not in {"stanford", "jjtms"}:
        raise ValueError(f"unknown dataset label: {value!r}")
    return text


def _normalise_group(value: Any, dataset: str) -> str:
    if _is_missing(value):
        raise ValueError("group is missing; labels must not be inferred")
    text = str(value).strip()
    expected = STANFORD_CELLS if dataset == "stanford" else JJTMS_CELLS
    if dataset == "jjtms":
        text = text.casefold()
    if text not in expected:
        raise ValueError(f"unknown {dataset} group label: {value!r}")
    return text
def _normalise_nonempty(value: Any, *, field: str, casefold: bool = False) -> str:
    if _is_missing(value):
        raise ValueError(f"{field} label is missing")
    text = str(value).strip()
    if casefold:
        text = text.casefold()
    if not text:
        raise ValueError(f"{field} label is empty")
    return text




def _normalise_flag(value: Any, *, field: str) -> int | None:
    if _is_missing(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer, float, np.floating)):
        numeric = float(value)
        if np.isfinite(numeric) and numeric in (0.0, 1.0):
            return int(numeric)
        raise ValueError(f"{field} must be binary when supplied: {value!r}")
    text = str(value).strip().casefold()
    aliases = {
        "0": 0,
        "false": 0,
        "no": 0,
        "nontrauma": 0,
        "non-trauma": 0,
        "healthy": 0,
        "1": 1,
        "true": 1,
        "yes": 1,
        "trauma": 1,
        "symptomatic": 1,
        "patient": 1,
    }
    if text not in aliases:
        raise ValueError(f"{field} must be binary when supplied: {value!r}")
    return aliases[text]


def _check_stanford_group_fields(frame: pd.DataFrame) -> None:
    expected = {
        "NTHC": (0, 0),
        "TEHC": (1, 0),
        "NTS": (0, 1),
        "TIS": (1, 1),
    }
    for index, row in frame.iterrows():
        group = str(row["group"])
        trauma = _normalise_flag(row["trauma"], field="trauma") if "trauma" in frame else None
        symptomatic = (
            _normalise_flag(row["symptomatic"], field="symptomatic")
            if "symptomatic" in frame
            else None
        )
        expected_trauma, expected_symptomatic = expected[group]
        if trauma is not None and trauma != expected_trauma:
            raise ValueError(
                f"trauma/group mismatch for {row['subject']!r}: {trauma} != {expected_trauma}"
            )
        if symptomatic is not None and symptomatic != expected_symptomatic:
            raise ValueError(
                "symptomatic/group mismatch for "
                f"{row['subject']!r}: {symptomatic} != {expected_symptomatic}"
            )


def _normalise_resolution(frame: pd.DataFrame) -> pd.Series:
    has_resolution = "resolution" in frame.columns
    has_network_resolution = "network_resolution" in frame.columns
    if not has_resolution and not has_network_resolution:
        raise ValueError("tensor needs resolution or network_resolution")
    source = frame["resolution"] if has_resolution else frame["network_resolution"]
    values = pd.to_numeric(source, errors="coerce")
    if values.isna().any() or (~np.isfinite(values.to_numpy(dtype=float))).any():
        raise ValueError("tensor resolution must be finite integers")
    numeric = values.to_numpy(dtype=float)
    if np.any(numeric != np.floor(numeric)):
        raise ValueError("tensor resolution must be finite integers")
    result = pd.Series(numeric.astype(int), index=frame.index)
    if not set(result).issubset({8, 17}):
        raise ValueError(f"unknown network resolution(s): {sorted(set(result) - {8, 17})}")
    if has_resolution and has_network_resolution:
        other = pd.to_numeric(frame["network_resolution"], errors="coerce")
        if other.isna().any() or not np.array_equal(result.to_numpy(), other.to_numpy(dtype=float)):
            raise ValueError("resolution and network_resolution disagree")
    return result


def _prepare_tensor(tensor: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(tensor, pd.DataFrame):
        raise TypeError("tensor must be a pandas DataFrame")
    if tensor.empty:
        raise ValueError("tensor is empty")
    required = {
        "dataset",
        "subject",
        "group",
        "method",
        "smoothing_mm",
        "region",
        "network",
        "fc_estimator",
        "fisher_z",
        "status",
    }
    missing = sorted(required - set(tensor.columns))
    if missing:
        raise ValueError(f"tensor is missing columns: {missing}")
    frame = tensor.copy(deep=True)
    # ``network_ordinals`` is the frozen Task 2 registry attached to the
    # DataFrame.  Preserve it explicitly through this normalization copy.
    frame.attrs = dict(tensor.attrs)
    frame["resolution"] = _normalise_resolution(frame)
    has_hemisphere = "hemisphere" in frame.columns
    has_mode = "hemisphere_mode" in frame.columns
    if not has_mode and not has_hemisphere:
        raise ValueError("tensor needs Task 2 hemisphere_mode views")
    if has_mode:
        modes = frame["hemisphere_mode"].map(
            lambda value: _normalise_nonempty(value, field="hemisphere_mode")
        )
    else:
        modes = frame["hemisphere"].map(
            lambda value: _normalise_nonempty(value, field="hemisphere")
        )
    if has_hemisphere and has_mode:
        hemispheres = frame["hemisphere"].map(
            lambda value: _normalise_nonempty(value, field="hemisphere")
        )
        if hemispheres.ne(modes).any():
            raise ValueError(
                "hemisphere and hemisphere_mode disagree; Task 2 collapsed views "
                "must retain the same L, R, bilateral, or L-R representation"
            )
    if not set(modes).issubset(set(HEMISPHERE_MODES)):
        raise ValueError(f"unknown hemisphere_mode label(s): {sorted(set(modes) - set(HEMISPHERE_MODES))}")
    frame["hemisphere_mode"] = modes

    frame["dataset"] = frame["dataset"].map(_normalise_dataset)
    frame["subject"] = frame["subject"].map(_normalise_subject)
    frame["group"] = [
        _normalise_group(value, dataset)
        for value, dataset in zip(frame["group"], frame["dataset"])
    ]
    frame["method"] = frame["method"].map(
        lambda value: _normalise_nonempty(value, field="method")
    )
    smoothing = pd.to_numeric(frame["smoothing_mm"], errors="coerce")
    if smoothing.isna().any() or (~np.isfinite(smoothing.to_numpy(dtype=float))).any():
        raise ValueError("tensor smoothing_mm must be finite integers")
    smoothing_values = smoothing.to_numpy(dtype=float)
    if np.any(smoothing_values != np.floor(smoothing_values)):
        raise ValueError("tensor smoothing_mm must be finite integers")
    frame["smoothing_mm"] = smoothing_values.astype(int)
    frame["region"] = frame["region"].map(
        lambda value: "tail" if str(value).strip().casefold() == "posterior" else str(value).strip().casefold()
    )
    if not set(frame["region"]).issubset(set(REGIONS)):
        raise ValueError(f"unknown region label(s): {sorted(set(frame['region']) - set(REGIONS))}")
    frame["network"] = frame["network"].map(
        lambda value: _normalise_nonempty(value, field="network")
    )
    frame["fc_estimator"] = frame["fc_estimator"].map(lambda value: str(value).strip().casefold())
    if not frame["fc_estimator"].eq("fisher_z").all():
        raise ValueError("localisation requires the registered fisher_z estimator")
    frame["fisher_z"] = pd.to_numeric(frame["fisher_z"], errors="coerce")
    frame["status"] = frame["status"].map(
        lambda value: _normalise_nonempty(value, field="status", casefold=True)
    )
    unknown_statuses = set(frame["status"]) - set(_TASK2_STATUS_VOCABULARY)
    if unknown_statuses:
        raise ValueError(
            "unknown Task 2 status label(s): " + ", ".join(sorted(unknown_statuses))
        )
    if "failure_reason" not in frame.columns:
        frame["failure_reason"] = ""
    frame["failure_reason"] = frame["failure_reason"].fillna("").map(str)
    finite = np.isfinite(frame["fisher_z"].to_numpy(dtype=float))
    frame.loc[~finite & frame["failure_reason"].eq(""), "failure_reason"] = "nonfinite Fisher-z value"
    if "trauma" not in frame.columns:
        frame["trauma"] = pd.NA
    if "symptomatic" not in frame.columns:
        frame["symptomatic"] = pd.NA
    stanford = frame["dataset"].eq("stanford")
    if stanford.any():
        _check_stanford_group_fields(frame.loc[stanford])

    group_counts = frame.groupby(
        ["dataset", "subject"], dropna=False, sort=False
    )["group"].nunique(dropna=False)
    if (group_counts > 1).any():
        conflict = group_counts[group_counts > 1].index[0]
        raise ValueError(
            "conflicting group assignment for normalized subject "
            f"{conflict[0]!r}/{conflict[1]!r}"
        )

    duplicate_key = [
        "dataset",
        "subject",
        "method",
        "smoothing_mm",
        "hemisphere_mode",
        "region",
        "resolution",
        "network",
        "fc_estimator",
    ]
    if frame.duplicated(duplicate_key, keep=False).any():
        raise ValueError("duplicate subject rows after normalization for a fixed component")
    return frame


def _task2_network_metadata(frame: pd.DataFrame) -> dict[tuple[str, int], dict[str, int]]:
    """Validate and index the frozen Task 2 method×resolution registry."""

    raw = frame.attrs.get("network_ordinals")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("tensor is missing Task 2 network_ordinals metadata")
    metadata: dict[tuple[str, int], dict[str, int]] = {}
    for raw_key, raw_ordinal in raw.items():
        if not isinstance(raw_key, tuple) or len(raw_key) != 3:
            raise ValueError(
                "Task 2 network_ordinals metadata keys must be "
                "(method, resolution, network) tuples"
            )
        method_value, resolution_value, network_value = raw_key
        method = _normalise_nonempty(method_value, field="metadata method")
        network = _normalise_nonempty(network_value, field="metadata network")
        if isinstance(resolution_value, (bool, np.bool_)):
            raise ValueError("Task 2 network metadata resolution must be an integer")
        if not isinstance(resolution_value, (int, np.integer)):
            raise ValueError("Task 2 network metadata resolution must be an integer")
        resolution = int(resolution_value)
        if resolution not in {8, 17}:
            raise ValueError(f"unknown Task 2 network metadata resolution: {resolution}")
        if isinstance(raw_ordinal, (bool, np.bool_)) or not isinstance(
            raw_ordinal, (int, np.integer)
        ):
            raise ValueError("Task 2 network metadata ordinals must be integers")
        ordinal = int(raw_ordinal)
        bucket = metadata.setdefault((method, resolution), {})
        if network in bucket:
            raise ValueError(
                f"duplicate Task 2 network metadata label {network!r} "
                f"for method={method!r}, resolution={resolution}"
            )
        if ordinal in bucket.values():
            raise ValueError(
                f"duplicate Task 2 network metadata ordinal {ordinal} "
                f"for method={method!r}, resolution={resolution}"
            )
        bucket[network] = ordinal

    for (method, resolution), labels in metadata.items():
        expected_ordinals = set(range(resolution))
        if len(labels) != resolution or set(labels.values()) != expected_ordinals:
            raise ValueError(
                "Task 2 network metadata must contain exactly "
                f"{resolution} labels with ordinals 0..{resolution - 1} "
                f"for method={method!r}"
            )
    return metadata


def _validate_planned_components(frame: pd.DataFrame) -> dict[tuple[Any, ...], pd.DataFrame]:
    metadata = _task2_network_metadata(frame)
    branches: dict[tuple[Any, ...], pd.DataFrame] = {}
    for branch_key, branch in frame.groupby(_BRANCH_COLUMNS, dropna=False, sort=False):
        if not isinstance(branch_key, tuple):
            branch_key = (branch_key,)
        if set(branch["region"]) != set(REGIONS):
            raise ValueError(
                "missing planned region component keys for "
                f"{branch_key}: expected {REGIONS}"
            )
        networks_by_region = {
            region: set(branch.loc[branch["region"].eq(region), "network"])
            for region in REGIONS
        }
        network_sets = list(networks_by_region.values())
        if any(network_set != network_sets[0] for network_set in network_sets[1:]):
            raise ValueError(f"missing planned network component keys for {branch_key}")
        networks = network_sets[0]
        if not networks:
            raise ValueError(f"empty planned network family for {branch_key}")
        expected_network_count = int(branch_key[4])
        method = str(branch_key[1])
        expected_metadata = metadata.get((method, expected_network_count))
        if expected_metadata is None:
            raise ValueError(
                "missing Task 2 network metadata for "
                f"method={method!r}, resolution={expected_network_count}"
            )
        expected_networks = set(expected_metadata)
        if networks != expected_networks:
            raise ValueError(
                "network labels do not match Task 2 frozen metadata for "
                f"method={method!r}, resolution={expected_network_count}: "
                f"expected {sorted(expected_networks)}, observed {sorted(networks)}"
            )
        if len(networks) != expected_network_count:
            raise ValueError(
                "planned network family is incomplete for "
                f"{branch_key}: expected {expected_network_count} networks, "
                f"observed {len(networks)}"
            )
        expected = {
            (region, network)
            for region in REGIONS
            for network in networks
        }
        observed = set(zip(branch["region"], branch["network"]))
        # Multiple subject rows are expected within each component; only the
        # planned component grid itself must be unique and complete.
        component_keys = branch[["region", "network"]].drop_duplicates()
        if observed != expected or len(component_keys) != len(expected):
            raise ValueError(f"duplicate/missing planned component keys for {branch_key}")
        # Each row is a subject-level component; keep the branch frame for the
        # deterministic component traversal below.
        branches[tuple(branch_key)] = branch.copy()
    return branches


def _expected_contrast_frame(dataset: str) -> pd.DataFrame:
    return stanford_contrast_matrix() if dataset == "stanford" else jjtms_contrast_matrix()


def _validate_contrast_block(block: pd.DataFrame, dataset: str) -> pd.DataFrame:
    expected = _expected_contrast_frame(dataset)
    cells = tuple(STANFORD_CELLS if dataset == "stanford" else JJTMS_CELLS)
    estimands = tuple(STANFORD_ESTIMANDS if dataset == "stanford" else JJTMS_ESTIMANDS)
    required = {"estimand", *cells}
    missing = sorted(required - set(block.columns))
    if missing:
        raise ValueError(f"{dataset} contrast matrix is missing columns: {missing}")
    if block["estimand"].duplicated().any() or set(block["estimand"]) != set(estimands):
        raise ValueError(f"{dataset} contrast matrix must contain exactly {estimands}")
    block = block.set_index("estimand").loc[list(estimands), list(cells)].copy()
    for column in cells:
        block[column] = pd.to_numeric(block[column], errors="coerce")
    if not np.isfinite(block.to_numpy(dtype=float)).all():
        raise ValueError(f"{dataset} contrast coefficients must be finite")
    expected_values = expected.set_index("estimand").loc[list(estimands), list(cells)].to_numpy(dtype=float)
    if not np.array_equal(block.to_numpy(dtype=float), expected_values):
        raise ValueError(f"{dataset} contrast matrix does not match the frozen Task 3 coefficients")
    return block


def _normalise_contrasts(
    contrasts: pd.DataFrame, datasets: set[str]
) -> dict[str, pd.DataFrame]:
    if not isinstance(contrasts, pd.DataFrame):
        raise TypeError("contrasts must be a pandas DataFrame")
    if contrasts.empty:
        raise ValueError("contrast matrix is empty")
    blocks: dict[str, pd.DataFrame] = {}
    if "dataset" in contrasts.columns:
        source = contrasts.copy(deep=True)
        source["dataset"] = source["dataset"].map(_normalise_dataset)
        unknown = set(source["dataset"]) - datasets
        if unknown:
            raise ValueError(f"contrast matrix has datasets absent from tensor: {sorted(unknown)}")
        for dataset in sorted(set(source["dataset"])):
            blocks[dataset] = _validate_contrast_block(
                source.loc[source["dataset"].eq(dataset)].drop(columns=["dataset"]), dataset
            )
    else:
        possible: list[str] = []
        for dataset in datasets:
            cells = set(STANFORD_CELLS if dataset == "stanford" else JJTMS_CELLS)
            if cells.issubset(set(contrasts.columns)):
                possible.append(dataset)
        if len(possible) == 1:
            dataset = possible[0]
            blocks[dataset] = _validate_contrast_block(contrasts, dataset)
        elif set(possible) == datasets and "estimand" in contrasts.columns:
            # A deterministic union of the two Task 3 matrices is accepted
            # when rows are still identifiable by their frozen estimand IDs.
            for dataset in sorted(datasets):
                estimands = set(
                    STANFORD_ESTIMANDS if dataset == "stanford" else JJTMS_ESTIMANDS
                )
                cells = list(STANFORD_CELLS if dataset == "stanford" else JJTMS_CELLS)
                subset = contrasts.loc[contrasts["estimand"].isin(estimands), ["estimand", *cells]]
                blocks[dataset] = _validate_contrast_block(subset, dataset)
        else:
            raise ValueError(
                "contrast matrix without dataset column must identify the tensor dataset(s)"
            )
    missing = datasets - set(blocks)
    if missing:
        raise ValueError(f"contrast matrix lacks blocks for tensor datasets: {sorted(missing)}")
    return blocks


def _fit_cell_means(
    component: pd.DataFrame,
    cells: tuple[str, ...],
    contrast: np.ndarray,
) -> tuple[dict[str, float] | None, str]:
    values_by_cell: dict[str, np.ndarray] = {}
    for cell in cells:
        selected = component.loc[
            component["group"].eq(cell)
            & component["status"].isin(_SUCCESS_STATUSES)
            & np.isfinite(component["fisher_z"].to_numpy(dtype=float))
        ]
        values = selected["fisher_z"].to_numpy(dtype=float)
        values_by_cell[cell] = values
    missing = [cell for cell, values in values_by_cell.items() if values.size == 0]
    if missing:
        return None, f"missing required cell(s): {', '.join(missing)}"
    if any(values.size < 2 for values in values_by_cell.values()):
        return None, "HC3 requires at least two independent subjects per cell"

    y = np.concatenate([values_by_cell[cell] for cell in cells])
    n = int(y.size)
    p = len(cells)
    x = np.zeros((n, p), dtype=float)
    offset = 0
    for index, cell in enumerate(cells):
        count = values_by_cell[cell].size
        x[offset : offset + count, index] = 1.0
        offset += count
    rank = int(np.linalg.matrix_rank(x))
    df_resid = n - rank
    if rank < p or df_resid <= 0:
        return None, "unestimable residual degrees of freedom"
    xtx = x.T @ x
    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        return None, "cell-means design is singular"
    beta = xtx_inv @ x.T @ y
    residual = y - x @ beta
    sse = float(residual @ residual)
    if not np.isfinite(sse) or sse <= 0.0:
        return None, "invalid residual variance for standardized effect"
    residual_sd = float(np.sqrt(sse / df_resid))
    if not np.isfinite(residual_sd) or residual_sd <= 0.0:
        return None, "invalid residual variance for standardized effect"
    leverage = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
    denominator = (1.0 - leverage) ** 2
    if not np.isfinite(denominator).all() or (denominator <= np.finfo(float).eps).any():
        return None, "HC3 leverage is undefined"
    meat = x.T @ ((residual**2 / denominator)[:, None] * x)
    covariance = xtx_inv @ meat @ xtx_inv
    variance = float(contrast @ covariance @ contrast)
    if not np.isfinite(variance) or variance <= 0.0:
        return None, "HC3 contrast variance is invalid"
    hc3_se = float(np.sqrt(variance))
    coefficient = float(contrast @ beta)
    critical = float(stats.t.ppf(0.975, df_resid))
    t_value = coefficient / hc3_se
    p_value = float(2.0 * stats.t.sf(abs(t_value), df_resid))
    ci_low = float(coefficient - critical * hc3_se)
    ci_high = float(coefficient + critical * hc3_se)
    standardised = float(coefficient / residual_sd)
    values = {
        "n": n,
        "coefficient": coefficient,
        "hc3_se": hc3_se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "standardized_effect": standardised,
        "p_raw": p_value,
    }
    if not all(np.isfinite(value) for value in values.values()):
        return None, "nonfinite model result"
    return values, ""


def _component_failure_reason(component: pd.DataFrame) -> str:
    reasons = []
    for status, reason in zip(component["status"], component["failure_reason"]):
        if str(status) not in _SUCCESS_STATUSES:
            text = str(reason).strip() or f"status={status}"
            if text not in reasons:
                reasons.append(text)
    return "; ".join(reasons)


def _network_sort_key(value: Any) -> tuple[Any, ...]:
    text = str(value)
    pieces = []
    for part in text.split("-"):
        if part.isdigit():
            pieces.append((0, int(part)))
        else:
            pieces.append((1, part.casefold()))
    return tuple(pieces)


def _sort_result(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.reindex(columns=OUTPUT_COLUMNS)
    dataset_order = {value: index for index, value in enumerate(DATASET_ORDER)}
    estimand_order = {
        **{value: index for index, value in enumerate(JJTMS_ESTIMANDS)},
        **{value: index for index, value in enumerate(STANFORD_ESTIMANDS)},
    }
    mode_order = {value: index for index, value in enumerate(HEMISPHERE_MODES)}
    region_order = {value: index for index, value in enumerate(REGIONS)}
    result = frame.copy()
    result["_dataset_order"] = result["dataset"].map(dataset_order).fillna(99)
    result["_estimand_order"] = result["estimand"].map(estimand_order).fillna(99)
    result["_mode_order"] = result["hemisphere_mode"].map(mode_order).fillna(99)
    result["_region_order"] = result["region"].map(region_order).fillna(99)
    result["_resolution_order"] = result["resolution"].map({17: 0, 8: 1}).fillna(99)
    result["_network_order"] = result["network"].map(_network_sort_key)
    result = result.sort_values(
        [
            "_dataset_order",
            "_estimand_order",
            "method",
            "smoothing_mm",
            "_mode_order",
            "_resolution_order",
            "_region_order",
            "_network_order",
            "fc_estimator",
        ],
        kind="stable",
    )
    return result.drop(
        columns=[
            "_dataset_order",
            "_estimand_order",
            "_mode_order",
            "_region_order",
            "_resolution_order",
            "_network_order",
        ]
    ).reindex(columns=OUTPUT_COLUMNS).reset_index(drop=True)




def fit_local_effects(tensor: pd.DataFrame, contrasts: pd.DataFrame) -> pd.DataFrame:
    """Fit every planned region/network localisation component.

    ``tensor`` must contain subject-level Task 2 hemisphere views.  Failed or
    unavailable rows are retained as failure results.  Raw p-values and
    confidence intervals are returned without multiplicity correction because
    the final analysis-object set has not yet been frozen.
    """

    frame = _prepare_tensor(tensor)
    branches = _validate_planned_components(frame)
    contrast_blocks = _normalise_contrasts(contrasts, set(frame["dataset"]))
    rows: list[dict[str, Any]] = []

    for branch_key, branch in branches.items():
        branch_values = dict(zip(_BRANCH_COLUMNS, branch_key))
        dataset = str(branch_values["dataset"])
        contrast_block = contrast_blocks[dataset]
        network_order = sorted(set(branch["network"]), key=_network_sort_key)
        expected_component_keys = {
            (region, network) for region in REGIONS for network in network_order
        }
        observed_component_keys = set(zip(branch["region"], branch["network"]))
        if expected_component_keys != observed_component_keys:
            raise AssertionError(f"planned component keys do not close for {branch_key}")
        components = {
            (region, network): group.copy()
            for (region, network), group in branch.groupby(["region", "network"], sort=False)
        }
        for estimand in contrast_block.index:
            contrast = contrast_block.loc[estimand].to_numpy(dtype=float)
            cells = tuple(STANFORD_CELLS if dataset == "stanford" else JJTMS_CELLS)
            for region in REGIONS:
                for network in network_order:
                    component = components[(region, network)]
                    source_failure = _component_failure_reason(component)
                    n = int(
                        np.sum(
                            component["status"].isin(_SUCCESS_STATUSES).to_numpy(dtype=bool)
                            & np.isfinite(component["fisher_z"].to_numpy(dtype=float))
                        )
                    )
                    fit, model_failure = _fit_cell_means(component, cells, contrast)
                    failure_reason = (
                        ""
                        if fit is not None
                        else "; ".join(reason for reason in (source_failure, model_failure) if reason)
                    )
                    output: dict[str, Any] = {
                        **branch_values,
                        "estimand": str(estimand),
                        "region": region,
                        "network": network,
                        "n": n,
                        "coefficient": np.nan,
                        "hc3_se": np.nan,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "standardized_effect": np.nan,
                        "p_raw": np.nan,
                        "q_bh": np.nan,
                        "correction_status": "deferred_until_analysis_set_freeze",
                        "failure_reason": str(failure_reason or ""),
                    }
                    if fit is not None:
                        output.update(fit)
                    rows.append(output)

    result = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    result.attrs["model"] = "unadjusted cell-means OLS with HC3 sandwich SE and finite-df t inference"
    result.attrs["standardized_effect"] = "coefficient / sqrt(SSE / residual_df)"
    result.attrs["multiplicity_correction"] = "deferred"
    result.attrs["analysis_object_set"] = "not_frozen"
    result.attrs["fdr"] = "not_applied; final analysis-object set is not frozen"
    return _sort_result(result)


__all__ = ["fit_local_effects"]

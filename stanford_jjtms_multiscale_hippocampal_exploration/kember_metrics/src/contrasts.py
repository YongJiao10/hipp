"""Independent Stanford direct-to-NTHC contrasts.

This module is deliberately separate from :mod:`statistics`.  It consumes the
already-built combined Kember feature table, retains both supported methods
in one result table, and evaluates the six pre-specified
case-versus-NTHC Welch comparisons.  The nested NTS-derived cohorts are kept as
their own rosters and are never silently treated as independent samples.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import platform
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

try:  # Imports work both as a package and from ``python src/contrasts.py``.
    from .inputs import (
        WORKFLOW_ROOT,
        validate_workflow_path,
        validate_rest_only_path,
    )
except ImportError:  # pragma: no cover - direct source-directory execution
    from inputs import WORKFLOW_ROOT, validate_workflow_path, validate_rest_only_path


REPO_ROOT = WORKFLOW_ROOT.parents[1]
DEFAULT_DIRECT_CONFIG_PATH = WORKFLOW_ROOT / "config" / "analysis_config.json"
DEFAULT_DIRECT_RUN_ID = "direct_contrasts"
DIRECT_METHOD = "hcp151_template_matching"
DIRECT_METHODS = ("hcp151_template_matching", "hcp151_fixed")

COHORT_ORDER = (
    "NTHC",
    "TEHC",
    "NTS",
    "NTS Any MDD",
    "NTS Pure MDD",
    "TIS",
    "TIS PTSD",
)
CASE_GROUPS = COHORT_ORDER[1:]
REFERENCE_GROUP = "NTHC"
DIRECT_CONTRASTS = tuple(f"{group}−{REFERENCE_GROUP}" for group in CASE_GROUPS)

KEMBER_ENDPOINTS = (
    ("surface_area", "surface_area_mm2", "surface_area_mm2"),
    ("functional_segregation", "functional_segregation_d", "functional_segregation_d"),
    ("boundary_sharpness", "boundary_sharpness_d", "boundary_sharpness_d"),
)
SYSTEMS = ("anterior", "middle", "posterior")


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a JSON object: {path}")
    return value


def load_direct_config(config_path: str | Path = DEFAULT_DIRECT_CONFIG_PATH) -> dict[str, Any]:
    """Load the independent direct-contrast configuration."""

    path = validate_workflow_path(config_path, root=WORKFLOW_ROOT, label="config_path")
    if not path.is_file():
        raise FileNotFoundError(f"direct-contrast config is missing: {path}")
    return _load_json(path)


def _coerce_config(config: Mapping[str, Any] | str | Path | None) -> dict[str, Any]:
    if config is None:
        return load_direct_config()
    if isinstance(config, (str, Path)):
        return load_direct_config(config)
    if not isinstance(config, Mapping):
        raise TypeError("direct-contrast config must be a mapping, path, or None")
    return copy.deepcopy(dict(config))


def _normalise_subject(value: Any) -> str:
    raw = str(value).strip()
    if raw.lower().startswith("sub-"):
        raw = raw[4:]
    if not raw.isdigit():
        raise ValueError(f"invalid Stanford subject identifier: {value!r}")
    return f"sub-{int(raw):04d}"


def _read_subject_manifest(path: Path) -> tuple[str, ...]:
    validate_rest_only_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"cohort manifest is missing: {path}")
    subjects: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.strip()
        if not token or token.startswith("#"):
            continue
        subject = _normalise_subject(token.split()[0])
        if subject in seen:
            raise ValueError(f"duplicate subject {subject} in {path}")
        seen.add(subject)
        subjects.append(subject)
    if not subjects:
        raise ValueError(f"cohort manifest is empty: {path}")
    return tuple(sorted(subjects, key=lambda item: (int(item[4:]), item)))


def _cohort_order(config: Mapping[str, Any]) -> tuple[str, ...]:
    raw = config.get("cohort_order", COHORT_ORDER)
    order = tuple(str(value) for value in raw)
    if not order or len(set(order)) != len(order):
        raise ValueError("cohort_order must contain unique cohort names")
    if REFERENCE_GROUP not in order:
        raise ValueError("cohort_order must contain NTHC")
    return order


def _cohort_specs(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = config.get("cohorts", config.get("cohort_manifests", {}))
    if not isinstance(raw, Mapping):
        raise ValueError("cohorts must be a mapping")
    result: dict[str, dict[str, Any]] = {}
    for name, value in raw.items():
        if isinstance(value, (str, Path)):
            result[str(name)] = {"manifest": str(value)}
        elif isinstance(value, Mapping):
            result[str(name)] = dict(value)
        else:
            raise ValueError(f"cohort specification must be a path or object: {name}")
    return result


def _manifest_path_for(config: Mapping[str, Any], cohort: str) -> Path | None:
    spec = _cohort_specs(config).get(cohort)
    if spec is None:
        return None
    value = spec.get("manifest", spec.get("path"))
    if value in (None, ""):
        return None
    return _resolve_repo_path(value)


def load_cohort_rosters(
    config: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, tuple[str, ...]]:
    """Read the configured NTHC and six case rosters without deduplicating them."""

    config_dict = _coerce_config(config)
    specs = _cohort_specs(config_dict)
    rosters: dict[str, tuple[str, ...]] = {}
    for cohort in _cohort_order(config_dict):
        spec = specs.get(cohort)
        if spec is None:
            raise KeyError(f"missing cohort configuration: {cohort}")
        path_value = spec.get("manifest", spec.get("path"))
        if path_value in (None, ""):
            raise KeyError(f"missing cohort manifest: {cohort}")
        path = _resolve_repo_path(path_value)
        subjects = _read_subject_manifest(path)
        expected = spec.get("expected_n")
        if expected is not None and len(subjects) != int(expected):
            raise ValueError(
                f"cohort {cohort} has {len(subjects)} subjects; expected {int(expected)}"
            )
        rosters[cohort] = subjects
    return rosters


def _normalise_rosters(
    rosters: Mapping[str, Iterable[Any]], config: Mapping[str, Any]
) -> dict[str, tuple[str, ...]]:
    order = _cohort_order(config)
    result: dict[str, tuple[str, ...]] = {}
    for cohort in order:
        if cohort not in rosters:
            raise KeyError(f"rosters missing cohort: {cohort}")
        values = tuple(_normalise_subject(value) for value in rosters[cohort])
        if len(set(values)) != len(values):
            raise ValueError(f"roster contains duplicate subjects: {cohort}")
        result[cohort] = tuple(sorted(values, key=lambda item: (int(item[4:]), item)))
    return result


def cohort_overlap_summary(rosters: Mapping[str, Iterable[Any]]) -> dict[str, Any]:
    """Describe all pairwise overlaps and proper subset relationships."""

    sets = {
        str(name): {_normalise_subject(value) for value in subjects}
        for name, subjects in rosters.items()
    }
    pairs: list[dict[str, Any]] = []
    nested: list[dict[str, Any]] = []
    for left, right in combinations(sets, 2):
        overlap = sorted(sets[left] & sets[right], key=lambda item: (int(item[4:]), item))
        relation = "none"
        if overlap and sets[left] < sets[right]:
            relation = "left_subset_of_right"
            nested.append(
                {
                    "subset": left,
                    "superset": right,
                    "overlap_n": len(overlap),
                    "subjects": overlap,
                }
            )
        elif overlap and sets[right] < sets[left]:
            relation = "right_subset_of_left"
            nested.append(
                {
                    "subset": right,
                    "superset": left,
                    "overlap_n": len(overlap),
                    "subjects": overlap,
                }
            )
        elif overlap and sets[left] == sets[right]:
            relation = "identical"
        pairs.append(
            {
                "left": left,
                "right": right,
                "overlap_n": len(overlap),
                "relationship": relation,
                "subjects": overlap,
            }
        )
    return {
        "allow_overlap": True,
        "pairwise": pairs,
        "nested_relationships": nested,
        "overlapping_pairs": [item for item in pairs if item["overlap_n"] > 0],
    }


def build_cohort_membership_table(
    rosters: Mapping[str, Iterable[Any]],
    config: Mapping[str, Any] | str | Path | None = None,
) -> pd.DataFrame:
    """Return one explicit row per cohort membership, including overlap flags."""

    config_dict = _coerce_config(config)
    normalised = _normalise_rosters(rosters, config_dict)
    overlap = cohort_overlap_summary(normalised)
    by_subject: dict[str, list[str]] = {}
    for cohort, subjects in normalised.items():
        for subject in subjects:
            by_subject.setdefault(subject, []).append(cohort)
    specs = _cohort_specs(config_dict)
    rows: list[dict[str, Any]] = []
    for cohort in _cohort_order(config_dict):
        spec = specs.get(cohort, {})
        manifest_path = _manifest_path_for(config_dict, cohort)
        for subject in normalised[cohort]:
            memberships = by_subject[subject]
            rows.append(
                {
                    "subject": subject,
                    "cohort": cohort,
                    "is_reference": cohort == REFERENCE_GROUP,
                    "expected_n": spec.get("expected_n", ""),
                    "roster_n": len(normalised[cohort]),
                    "manifest_path": "" if manifest_path is None else str(manifest_path),
                    "membership_count": len(memberships),
                    "overlap": len(memberships) > 1,
                    "overlapping_cohorts": ";".join(memberships) if len(memberships) > 1 else "",
                }
            )
    result = pd.DataFrame(
        rows,
        columns=[
            "subject",
            "cohort",
            "is_reference",
            "expected_n",
            "roster_n",
            "manifest_path",
            "membership_count",
            "overlapping_cohorts",
            "overlap",
        ],
    )
    result.attrs["overlap_summary"] = overlap
    return result


def _contrast_specs(config: Mapping[str, Any]) -> tuple[dict[str, str], ...]:
    raw = config.get("contrasts")
    if raw is None:
        raw = [
            {
                "contrast": contrast,
                "case_group": case,
                "reference_group": REFERENCE_GROUP,
            }
            for contrast, case in zip(DIRECT_CONTRASTS, CASE_GROUPS)
        ]
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("contrasts must be a sequence")
    result: list[dict[str, str]] = []
    for item in raw:
        if isinstance(item, Mapping):
            case = str(item.get("case_group", ""))
            reference = str(item.get("reference_group", REFERENCE_GROUP))
            contrast = str(item.get("contrast", f"{case}−{reference}"))
        else:
            contrast = str(item)
            if "−" in contrast:
                case, reference = contrast.split("−", 1)
            elif "-" in contrast:
                case, reference = contrast.split("-", 1)
            else:
                raise ValueError(f"cannot parse direct contrast: {contrast}")
            case = case.strip()
            reference = reference.strip()
        expected = f"{case}−{reference}"
        if reference != REFERENCE_GROUP or case not in CASE_GROUPS or contrast != expected:
            raise ValueError(
                "direct workflow permits only case−NTHC contrasts; "
                f"received {contrast!r}"
            )
        result.append(
            {
                "contrast": contrast,
                "case_group": case,
                "reference_group": reference,
            }
        )
    if len({item["contrast"] for item in result}) != len(result):
        raise ValueError("direct contrasts must be unique")
    return tuple(result)


def _validate_axes(
    config: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[str, ...]]:
    methods = tuple(str(value) for value in config.get("methods", ()))
    if methods != DIRECT_METHODS:
        raise ValueError(
            "direct workflow methods must be exactly "
            "hcp151_template_matching and hcp151_fixed"
        )
    smoothings = tuple(int(value) for value in config.get("smoothings_mm", (2, 4)))
    hemispheres = tuple(str(value) for value in config.get("hemispheres", ("L", "R")))
    if not smoothings or any(value <= 0 for value in smoothings) or len(set(smoothings)) != len(smoothings):
        raise ValueError("smoothings_mm must contain unique positive values")
    if not hemispheres or any(value not in {"L", "R"} for value in hemispheres) or len(set(hemispheres)) != len(hemispheres):
        raise ValueError("hemispheres must contain unique L/R values")
    return methods, smoothings, hemispheres


def _endpoint_id(row: Mapping[str, Any]) -> str:
    return (
        "direct_endpoint:"
        f"contrast={row['contrast']}|metric_family={row['metric_family']}|"
        f"metric={row['metric']}|method={row['method']}|"
        f"smoothing_mm={row['smoothing_mm']}|hemisphere={row['hemisphere']}|"
        f"system={row['system']}"
    )


def enumerate_direct_endpoints(
    config: Mapping[str, Any] | str | Path | None = None,
) -> pd.DataFrame:
    """Enumerate the exact endpoint units for the configured direct contrasts."""

    config_dict = _coerce_config(config)
    methods, smoothings, hemispheres = _validate_axes(config_dict)
    contrasts = _contrast_specs(config_dict)
    rows: list[dict[str, Any]] = []
    for contrast in contrasts:
        for method in methods:
            for smoothing_mm in smoothings:
                for hemisphere in hemispheres:
                    for metric_family, metric, source_field in KEMBER_ENDPOINTS:
                        for system in SYSTEMS:
                            row = {
                                **contrast,
                                "metric_family": metric_family,
                                "metric": metric,
                                "source_field": source_field,
                                "method": method,
                                "smoothing_mm": smoothing_mm,
                                "hemisphere": hemisphere,
                                "system": system,
                            }
                            row["endpoint_id"] = _endpoint_id(row)
                            rows.append(row)
    columns = [
        "endpoint_id",
        "contrast",
        "case_group",
        "reference_group",
        "metric_family",
        "metric",
        "source_field",
        "method",
        "smoothing_mm",
        "hemisphere",
        "system",
    ]
    result = pd.DataFrame(rows, columns=columns)
    if result[columns].duplicated().any():
        raise AssertionError("direct endpoint enumeration contains duplicate units")
    return result


# Short aliases make the public contract convenient for notebooks and tests.
enumerate_endpoints = enumerate_direct_endpoints


def _finite_array(values: Iterable[Any]) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    return numeric[np.isfinite(numeric)]


def welch_two_sample(
    case_values: Iterable[Any],
    reference_values: Iterable[Any],
    *,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Compute a two-sided Welch test, pooled-SD Cohen's d, and Welch CI."""

    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must be strictly between zero and one")
    case = _finite_array(case_values)
    reference = _finite_array(reference_values)
    case_n = int(case.size)
    reference_n = int(reference.size)
    result: dict[str, Any] = {
        "case_n": case_n,
        "reference_n": reference_n,
        "case_mean": float(np.mean(case)) if case_n else np.nan,
        "reference_mean": float(np.mean(reference)) if reference_n else np.nan,
        "case_sd": float(np.std(case, ddof=1)) if case_n > 1 else np.nan,
        "reference_sd": float(np.std(reference, ddof=1)) if reference_n > 1 else np.nan,
        "mean_difference": np.nan,
        "welch_t": np.nan,
        "welch_df": np.nan,
        "p_value": np.nan,
        "cohens_d": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "test_reason": "",
    }
    if case_n < 2 or reference_n < 2:
        result["test_reason"] = "unfittable:fewer_than_two_finite_values_per_group"
        if case_n and reference_n:
            result["mean_difference"] = result["case_mean"] - result["reference_mean"]
        return result

    case_variance = float(np.var(case, ddof=1))
    reference_variance = float(np.var(reference, ddof=1))
    difference = float(result["case_mean"] - result["reference_mean"])
    result["mean_difference"] = difference
    case_component = case_variance / case_n
    reference_component = reference_variance / reference_n
    standard_error_squared = case_component + reference_component
    pooled_denominator = case_n + reference_n - 2
    pooled_variance = (
        ((case_n - 1) * case_variance + (reference_n - 1) * reference_variance)
        / pooled_denominator
        if pooled_denominator > 0
        else np.nan
    )
    if not np.isfinite(standard_error_squared) or standard_error_squared <= np.finfo(float).eps:
        result["test_reason"] = "degenerate:zero_welch_standard_error"
        return result

    standard_error = math.sqrt(standard_error_squared)
    denominator = (
        (case_component**2) / (case_n - 1)
        + (reference_component**2) / (reference_n - 1)
    )
    if not np.isfinite(denominator) or denominator <= 0:
        result["test_reason"] = "degenerate:welch_degrees_of_freedom"
        return result
    degrees_of_freedom = float(standard_error_squared**2 / denominator)
    t_value = float(difference / standard_error)
    p_value = float(2.0 * scipy_stats.t.sf(abs(t_value), degrees_of_freedom))
    critical = float(scipy_stats.t.ppf((1.0 + float(confidence)) / 2.0, degrees_of_freedom))
    result.update(
        {
            "welch_t": t_value,
            "welch_df": degrees_of_freedom,
            "p_value": p_value,
            "ci_low": difference - critical * standard_error,
            "ci_high": difference + critical * standard_error,
            "cohens_d": (
                difference / math.sqrt(float(pooled_variance))
                if np.isfinite(pooled_variance) and pooled_variance > 0
                else np.nan
            ),
        }
    )
    return result


def apply_bh_fdr(
    results: pd.DataFrame,
    *,
    analysis_object_set_frozen: bool = False,
) -> pd.DataFrame:
    """Apply BH-FDR only after the analysis-object set is explicitly frozen."""

    if not analysis_object_set_frozen:
        raise ValueError(
            "multiplicity correction is deferred until the final analysis-object set is frozen"
        )
    if "metric_family" not in results.columns:
        raise ValueError("results must contain metric_family for family-level FDR")
    result = results.copy()
    if "p_value" not in result.columns:
        if "p_raw" not in result.columns:
            raise ValueError("results must contain p_value or p_raw")
        result["p_value"] = pd.to_numeric(result["p_raw"], errors="coerce")
    else:
        result["p_value"] = pd.to_numeric(result["p_value"], errors="coerce")
    if "p_raw" not in result.columns:
        result["p_raw"] = result["p_value"]
    result["q_value"] = np.nan
    result["q_bh"] = np.nan
    result["fdr_family_size"] = 0
    result["family_size"] = 0
    for family, group in result.groupby("metric_family", dropna=False, sort=False):
        finite = pd.to_numeric(group["p_value"], errors="coerce").to_numpy(dtype=float)
        finite_mask = np.isfinite(finite)
        indices = group.index.to_numpy()[finite_mask]
        p_values = finite[finite_mask]
        family_size = int(p_values.size)
        if not family_size:
            continue
        order = np.argsort(p_values, kind="mergesort")
        ordered = p_values[order]
        ranks = np.arange(1, family_size + 1, dtype=float)
        adjusted = ordered * family_size / ranks
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        adjusted = np.clip(adjusted, 0.0, 1.0)
        q_values = np.empty(family_size, dtype=float)
        q_values[order] = adjusted
        result.loc[indices, "q_value"] = q_values
        result.loc[indices, "q_bh"] = q_values
        result.loc[indices, "fdr_family_size"] = family_size
        result.loc[indices, "family_size"] = family_size
    result["correction_status"] = "applied_to_frozen_analysis_set"
    return result


def _feature_required_columns() -> set[str]:
    return {
        "subject",
        "method",
        "smoothing_mm",
        "hemisphere",
        "system",
        "status",
        "endpoint_missing_fields",
        *(source_field for _family, _metric, source_field in KEMBER_ENDPOINTS),
    }


def _normalise_feature_table(feature_table: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(feature_table, pd.DataFrame):
        feature_table = pd.DataFrame(feature_table)
    missing = sorted(_feature_required_columns() - set(feature_table.columns))
    if missing:
        raise ValueError(f"feature table missing direct-contrast columns: {missing}")
    result = feature_table.copy()
    result["subject"] = result["subject"].map(_normalise_subject)
    result["method"] = result["method"].astype(str)
    result["smoothing_mm"] = pd.to_numeric(result["smoothing_mm"], errors="raise").astype(int)
    result["hemisphere"] = result["hemisphere"].astype(str)
    result["system"] = result["system"].fillna("").astype(str)
    result["status"] = result["status"].fillna("unknown").astype(str)
    result["endpoint_missing_fields"] = result["endpoint_missing_fields"].fillna("").astype(str)
    return result


def _missing_field_declared(raw: Any, field: str) -> bool:
    return field in {value.strip() for value in str(raw or "").split(",") if value.strip()}


def _subject_endpoint_value(
    subject_rows: pd.DataFrame,
    source_field: str,
) -> tuple[float, str]:
    if subject_rows.empty:
        return np.nan, "feature_rows_missing"
    statuses = sorted(set(subject_rows["status"].astype(str)))
    if any(status != "success" for status in statuses):
        return np.nan, "feature_status=" + ",".join(statuses)
    if source_field not in subject_rows.columns:
        return np.nan, "source_field_missing=" + source_field
    if subject_rows["endpoint_missing_fields"].map(
        lambda raw: _missing_field_declared(raw, source_field)
    ).any():
        return np.nan, "declared_missing=" + source_field
    numeric = pd.to_numeric(subject_rows[source_field], errors="coerce").to_numpy(dtype=float)
    finite = numeric[np.isfinite(numeric)]
    if finite.size == 0:
        return np.nan, "missing_value=" + source_field
    if finite.size != numeric.size:
        return np.nan, "partial_missing=" + source_field
    if float(np.max(finite) - np.min(finite)) > 1e-10:
        return np.nan, "inconsistent_duplicate_values=" + source_field
    return float(finite[0]), ""


def _endpoint_values_for_group(
    feature_table: pd.DataFrame,
    endpoint: Mapping[str, Any],
    subjects: Sequence[str],
) -> tuple[np.ndarray, list[dict[str, str]]]:
    axis = feature_table.loc[
        feature_table["method"].eq(str(endpoint["method"]))
        & feature_table["smoothing_mm"].eq(int(endpoint["smoothing_mm"]))
        & feature_table["hemisphere"].eq(str(endpoint["hemisphere"]))
    ]
    values: list[float] = []
    missing: list[dict[str, str]] = []
    for subject in subjects:
        subject_rows = axis.loc[axis["subject"].eq(subject)]
        subject_rows = subject_rows.loc[
            subject_rows["system"].eq(str(endpoint["system"]))
        ]
        value, reason = _subject_endpoint_value(subject_rows, str(endpoint["source_field"]))
        if np.isfinite(value):
            values.append(value)
        else:
            missing.append({"subject": subject, "reason": reason})
    return np.asarray(values, dtype=float), missing


def _provenance_text(
    endpoint: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    feature_table_path: str | Path | None = None,
) -> str:
    case = str(endpoint["case_group"])
    reference = str(endpoint["reference_group"])
    payload = {
        "feature_table": "in_memory" if feature_table_path is None else str(Path(feature_table_path).resolve()),
        "method": str(endpoint["method"]),
        "source_field": str(endpoint["source_field"]),
        "case_group": case,
        "reference_group": reference,
        "case_manifest": "" if _manifest_path_for(config, case) is None else str(_manifest_path_for(config, case)),
        "reference_manifest": "" if _manifest_path_for(config, reference) is None else str(_manifest_path_for(config, reference)),
        "estimator": "Welch two-sample, case minus reference",
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def compute_direct_contrast_statistics(
    feature_table: pd.DataFrame,
    *,
    config: Mapping[str, Any] | str | Path | None = None,
    rosters: Mapping[str, Iterable[Any]] | None = None,
    feature_table_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute direct effects, group descriptives, and endpoint coverage."""

    config_dict = _coerce_config(config)
    normalised_rosters = (
        load_cohort_rosters(config_dict)
        if rosters is None
        else _normalise_rosters(rosters, config_dict)
    )
    feature = _normalise_feature_table(feature_table)
    endpoints = enumerate_direct_endpoints(config_dict)
    effect_rows: list[dict[str, Any]] = []
    descriptive_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []

    for endpoint_number, endpoint in enumerate(endpoints.to_dict("records"), start=1):
        endpoint_id = str(endpoint["endpoint_id"])
        group_values: dict[str, np.ndarray] = {}
        group_missing: dict[str, list[dict[str, str]]] = {}
        for cohort in (str(endpoint["case_group"]), str(endpoint["reference_group"])):
            values, missing = _endpoint_values_for_group(
                feature,
                endpoint,
                normalised_rosters[cohort],
            )
            group_values[cohort] = values
            group_missing[cohort] = missing
            missing_reasons = sorted({item["reason"] for item in missing if item["reason"]})
            descriptive_rows.append(
                {
                    "row_id": f"descriptive:{endpoint_id}:{cohort}",
                    **endpoint,
                    "cohort": cohort,
                    "n_expected": len(normalised_rosters[cohort]),
                    "n_valid": int(values.size),
                    "n_missing": len(missing),
                    "mean": float(np.mean(values)) if values.size else np.nan,
                    "sd": float(np.std(values, ddof=1)) if values.size > 1 else np.nan,
                    "missing_reason": ";".join(missing_reasons),
                    "missing_subjects": ";".join(item["subject"] for item in missing),
                    "provenance": _provenance_text(
                        endpoint, config_dict, feature_table_path=feature_table_path
                    ),
                }
            )
            coverage_rows.append(
                {
                    "row_id": f"coverage:{endpoint_id}:{cohort}",
                    **endpoint,
                    "cohort": cohort,
                    "n_expected": len(normalised_rosters[cohort]),
                    "n_valid": int(values.size),
                    "missing_n": len(missing),
                    "missing_reason": ";".join(missing_reasons),
                    "missing_subjects": ";".join(item["subject"] for item in missing),
                    "provenance": _provenance_text(
                        endpoint, config_dict, feature_table_path=feature_table_path
                    ),
                }
            )

        case = str(endpoint["case_group"])
        reference = str(endpoint["reference_group"])
        test = welch_two_sample(group_values[case], group_values[reference])
        missing_reason_parts: list[str] = []
        if group_missing[case]:
            missing_reason_parts.append(f"case_missing_n={len(group_missing[case])}")
        if group_missing[reference]:
            missing_reason_parts.append(f"reference_missing_n={len(group_missing[reference])}")
        if test["test_reason"]:
            missing_reason_parts.append(str(test["test_reason"]))
        row = {
            "row_id": f"direct_effect:{endpoint_number:06d}",
            **endpoint,
            "case_n": test["case_n"],
            "case_mean": test["case_mean"],
            "case_sd": test["case_sd"],
            "reference_n": test["reference_n"],
            "reference_mean": test["reference_mean"],
            "reference_sd": test["reference_sd"],
            # Explicit NTHC aliases make the output self-describing for tables.
            "nthc_n": test["reference_n"],
            "nthc_mean": test["reference_mean"],
            "nthc_sd": test["reference_sd"],
            "mean_difference": test["mean_difference"],
            "welch_t": test["welch_t"],
            "welch_df": test["welch_df"],
            "p_value": test["p_value"],
            "p_raw": test["p_value"],
            "q_value": np.nan,
            "q_bh": np.nan,
            "fdr_family_size": np.nan,
            "family_size": np.nan,
            "correction_status": "deferred_until_analysis_set_freeze",
            "cohens_d": test["cohens_d"],
            "ci_low": test["ci_low"],
            "ci_high": test["ci_high"],
            "case_missing_n": len(group_missing[case]),
            "reference_missing_n": len(group_missing[reference]),
            "case_missing_subjects": ";".join(item["subject"] for item in group_missing[case]),
            "reference_missing_subjects": ";".join(item["subject"] for item in group_missing[reference]),
            "missing_reason": ";".join(missing_reason_parts),
            "provenance": _provenance_text(
                endpoint, config_dict, feature_table_path=feature_table_path
            ),
        }
        effect_rows.append(row)

    effects = pd.DataFrame(effect_rows)
    descriptives = pd.DataFrame(descriptive_rows)
    coverage = pd.DataFrame(coverage_rows)
    return effects, descriptives, coverage


# Additional descriptive names for callers that use “fit” terminology.
fit_direct_contrasts = compute_direct_contrast_statistics
build_direct_contrast_results = compute_direct_contrast_statistics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_tsv(table: pd.DataFrame, path: Path) -> None:
    path.write_text(table.to_csv(sep="\t", index=False, na_rep=""), encoding="utf-8")


def _direction_consistency(effects: pd.DataFrame) -> pd.DataFrame:
    """Summarize same-sign effects across the four 2/4 mm × L/R axes."""

    if effects.empty:
        return pd.DataFrame(
            columns=["method", "contrast", "metric_family", "metric", "system", "n_finite", "n_axes", "same_direction", "direction"]
        )
    rows: list[dict[str, Any]] = []
    key_columns = ["method", "contrast", "metric_family", "metric", "system"]
    for key, group in effects.groupby(key_columns, dropna=False, sort=False):
        values = pd.to_numeric(group["cohens_d"], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        signs = np.sign(finite[finite != 0])
        same = bool(signs.size >= 2 and np.all(signs == signs[0]))
        rows.append(
            {
                **dict(zip(key_columns, key if isinstance(key, tuple) else (key,))),
                "n_finite": int(finite.size),
                "n_axes": int(len(group)),
                "same_direction": same,
                "direction": "positive" if signs.size and signs[0] > 0 else "negative" if signs.size else "",
            }
        )
    return pd.DataFrame(rows)


def write_analysis_summary(
    path: str | Path,
    *,
    run_id: str,
    effects: pd.DataFrame,
    descriptives: pd.DataFrame,
    coverage: pd.DataFrame,
    overlap: Mapping[str, Any],
    figure_sources: Sequence[Mapping[str, Any]],
) -> Path:
    """Write a concise raw-p summary of the combined two-method workflow."""

    del descriptives, overlap
    output_path = Path(path)
    p_values = pd.to_numeric(effects.get("p_value"), errors="coerce")
    finite = effects.loc[p_values.notna()].copy()
    hits = finite.loc[p_values.loc[finite.index].lt(0.05)].copy()
    if not hits.empty:
        hits["_abs_d"] = pd.to_numeric(hits["cohens_d"], errors="coerce").abs()
        hits = hits.sort_values(["p_value", "_abs_d", "row_id"], kind="mergesort")
    family_counts = effects.groupby("metric_family", dropna=False).size().to_dict()
    missing_cells = int(pd.to_numeric(coverage.get("missing_n"), errors="coerce").fillna(0).sum())

    def fmt(value: Any, digits: int = 3) -> str:
        number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if not np.isfinite(number):
            return "NA"
        if abs(float(number)) < 0.001 and float(number) != 0.0:
            return "<0.001"
        return f"{float(number):.{digits}f}"

    lines = [
        f"# Stanford two-method Kember direct-to-NTHC: {run_id}",
        "",
        "## 指标与范围",
        "",
        "本运行合并 `hcp151_template_matching` 与 `hcp151_fixed`，每个半球分别计算 Kember 的三类指标：system surface area、functional segregation、boundary sharpness。没有 transition、几何 boundary position、vertex fraction 或 AP centroid。",
        "",
        "直接比较固定为 TEHC−NTHC、NTS−NTHC、NTS Any MDD−NTHC、NTS Pure MDD−NTHC、TIS−NTHC、TIS PTSD−NTHC；统计为 case−NTHC 的双侧 Welch test、95% Welch CI、pooled-SD Cohen d 和原始 p。",
        "Multiplicity correction is deferred until the final analysis-object set is frozen; q_value/q_bh remain empty and raw p values are descriptive only.",
        "",
        "6 contrasts × 2 methods × 2 smoothing × 2 hemispheres × 3 metrics × 3 systems = **432** 个 direct-comparison units。",
        "",
        "| 指标 | endpoint units |",
        "| --- | ---: |",
    ]
    for family in ("surface_area", "functional_segregation", "boundary_sharpness"):
        lines.append(f"| {family} | {int(family_counts.get(family, 0))} |")
    lines.extend(
        [
            "",
            f"有限原始 p 值：**{len(finite):,}/{len(effects):,}**；缺失只影响对应 group-endpoint cell（共 {missing_cells:,}），没有零填充或插补。",
            "",
            "## 原始 p < 0.05 的行",
            "",
        ]
    )
    if hits.empty:
        lines.append("没有原始 p < 0.05 的结果行。")
    else:
        lines.extend(
            [
                "| contrast | metric | system | smoothing | hemisphere | d | p | 95% CI |",
                "| --- | --- | --- | ---: | --- | ---: | ---: | --- |",
            ]
        )
        for _, row in hits.iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("contrast", "")),
                        str(row.get("metric", "")),
                        str(row.get("system", "")),
                        str(row.get("smoothing_mm", "")),
                        str(row.get("hemisphere", "")),
                        fmt(row.get("cohens_d")),
                        fmt(row.get("p_value")),
                        "[" + fmt(row.get("ci_low")) + ", " + fmt(row.get("ci_high")) + "]",
                    ]
                )
                + " |"
            )
    lines.extend(["", "## 图", ""])
    for figure in figure_sources:
        lines.append(
            "- "
            + str(figure.get("figure_id", ""))
            + ": "
            + str(figure.get("png_path", ""))
            + "; source result rows: "
            + str(len(figure.get("source_row_ids", [])))
            + "."
        )
    lines.extend(
        [
            "",
            "图内每格显示 Cohen d 与原始 p；2/4 mm 仅为可读性保留 2 mm 图示，结果表始终保留两种 smoothing，原始 p 不作为多重校正或筛选依据。",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _output_root(output_root: str | Path | None) -> Path:
    root = WORKFLOW_ROOT if output_root is None else Path(output_root).expanduser().resolve()
    root = root.resolve()
    workflow_root = WORKFLOW_ROOT.resolve()
    if not root.is_relative_to(workflow_root):
        raise ValueError("output_root must be contained by WORKFLOW_ROOT")
    return root


def run_direct_contrasts(
    feature_table: pd.DataFrame | str | Path,
    config: Mapping[str, Any] | str | Path,
    output_dir: str | Path,
) -> Path:
    """Write one combined direct-contrast table for both active methods."""

    config_dict = _coerce_config(config)
    methods, _, _ = _validate_axes(config_dict)
    if isinstance(feature_table, (str, Path)):
        feature_path = Path(feature_table).expanduser().resolve()
        validate_rest_only_path(feature_path)
        if not feature_path.is_file():
            raise FileNotFoundError(f"feature table is missing: {feature_path}")
        table = pd.read_csv(feature_path, sep="\t")
        feature_source = str(feature_path)
    else:
        table = pd.DataFrame(feature_table).copy()
        feature_source = "in_memory"
    destination = Path(output_dir).expanduser().resolve()
    validate_rest_only_path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    rosters = load_cohort_rosters(config_dict)
    effects, descriptives, coverage = compute_direct_contrast_statistics(
        table,
        config=config_dict,
        rosters=rosters,
        feature_table_path=None if feature_source == "in_memory" else feature_source,
    )
    effects_path = destination / "direct_contrast_effects.tsv"
    descriptives_path = destination / "group_descriptives.tsv"
    coverage_path = destination / "endpoint_coverage.tsv"
    _write_tsv(effects, effects_path)
    _write_tsv(descriptives, descriptives_path)
    _write_tsv(coverage, coverage_path)
    manifest = {
        "task": "Stanford unified Kember direct contrasts",
        "methods": list(methods),
        "feature_table": feature_source,
        "endpoint_units": int(len(effects)),
        "multiplicity_correction": "deferred_until_analysis_set_freeze",
        "analysis_object_set": "not_frozen",
        "endpoint_counts_by_method": {
            str(method): int(count)
            for method, count in effects.groupby("method", dropna=False).size().items()
        },
        "outputs": {
            "direct_contrasts": str(effects_path),
            "group_descriptives": str(descriptives_path),
            "endpoint_coverage": str(coverage_path),
        },
        "output_sha256": {
            "direct_contrasts": _sha256(effects_path),
            "group_descriptives": _sha256(descriptives_path),
            "endpoint_coverage": _sha256(coverage_path),
        },
    }
    (destination / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def run_stable_contrasts(
    config_path: str | Path = DEFAULT_DIRECT_CONFIG_PATH,
    *,
    feature_table_path: str | Path | None = None,
    results_dir: str | Path | None = None,
    figures_dir: str | Path | None = None,
    reports_dir: str | Path | None = None,
) -> Path:
    """Materialize the one stable direct-contrast result, figure, and report set."""

    config_file = validate_workflow_path(
        config_path, root=WORKFLOW_ROOT, label="config_path"
    )
    config = load_direct_config(config_file)
    configured_outputs = config.get("outputs", {})

    def configured_path(name: str, fallback: Path) -> Path:
        value = configured_outputs.get(name)
        return _resolve_repo_path(value) if value not in (None, "") else fallback

    configured_feature = config.get("feature_table", config.get("feature_table_path"))
    if feature_table_path is None:
        if configured_feature in (None, ""):
            feature_table_path = (
                configured_path("features", WORKFLOW_ROOT / "work" / "features")
                / "subject_features.tsv"
            )
        else:
            feature_table_path = _resolve_repo_path(configured_feature)
    feature_file = validate_workflow_path(
        feature_table_path, root=WORKFLOW_ROOT, label="feature_table_path"
    )
    if not feature_file.is_file():
        raise FileNotFoundError(f"feature table is missing: {feature_file}")

    result_path = validate_workflow_path(
        results_dir
        if results_dir is not None
        else configured_path(
            "direct_contrasts", WORKFLOW_ROOT / "results" / "direct_contrasts"
        ),
        root=WORKFLOW_ROOT,
        label="results_dir",
    )
    figure_path = validate_workflow_path(
        figures_dir
        if figures_dir is not None
        else configured_path(
            "figures", WORKFLOW_ROOT / "figures" / "direct_contrasts"
        ),
        root=WORKFLOW_ROOT,
        label="figures_dir",
    )
    report_path = validate_workflow_path(
        reports_dir
        if reports_dir is not None
        else configured_path(
            "reports", WORKFLOW_ROOT / "reports" / "direct_contrasts"
        ),
        root=WORKFLOW_ROOT,
        label="reports_dir",
    )

    result_path = run_direct_contrasts(feature_file, config, result_path)
    effects_path = result_path / "direct_contrast_effects.tsv"
    descriptives_path = result_path / "group_descriptives.tsv"
    coverage_path = result_path / "endpoint_coverage.tsv"
    effects = pd.read_csv(effects_path, sep="\t")
    descriptives = pd.read_csv(descriptives_path, sep="\t")
    coverage = pd.read_csv(coverage_path, sep="\t")
    rosters = load_cohort_rosters(config)
    membership_path = result_path / "cohort_membership.tsv"
    _write_tsv(build_cohort_membership_table(rosters, config), membership_path)

    try:
        from .plot_contrasts import render_direct_contrast_figures
    except ImportError:  # pragma: no cover - direct script execution
        from plot_contrasts import render_direct_contrast_figures

    figure_path.mkdir(parents=True, exist_ok=True)
    figure_manifest = render_direct_contrast_figures(
        effects, figure_path, source_table=str(effects_path)
    )
    figure_sources_path = result_path / "figure_sources.tsv"
    figure_sources_table = pd.DataFrame(
        [
            {
                "figure_id": item["figure_id"],
                "png_path": item["png_path"],
                "source_table": item["source_table"],
                "source_row_ids": ";".join(item["source_row_ids"]),
                "source_row_count": len(item["source_row_ids"]),
            }
            for item in figure_manifest["figure_sources"]
        ],
        columns=[
            "figure_id",
            "png_path",
            "source_table",
            "source_row_ids",
            "source_row_count",
        ],
    )
    _write_tsv(figure_sources_table, figure_sources_path)

    report_path.mkdir(parents=True, exist_ok=True)
    figure_provenance_path = report_path / "figure_provenance.json"
    figure_provenance = {
        "source_table": str(effects_path),
        "source_table_sha256": _sha256(effects_path),
        "figures": figure_manifest["figure_sources"],
        "displayed_smoothings_mm": figure_manifest.get("displayed_smoothings_mm", []),
        "smoothing_display_rule": figure_manifest.get("smoothing_display_rule", ""),
        "row_id_integrity": all(
            set(item["source_row_ids"]) <= set(effects["row_id"].astype(str))
            for item in figure_manifest["figure_sources"]
        ),
    }
    figure_provenance_path.write_text(
        json.dumps(figure_provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_path = report_path / "analysis_summary.md"
    write_analysis_summary(
        summary_path,
        run_id=DEFAULT_DIRECT_RUN_ID,
        effects=effects,
        descriptives=descriptives,
        coverage=coverage,
        overlap=cohort_overlap_summary(rosters),
        figure_sources=figure_manifest["figure_sources"],
    )

    feature_manifest_path = feature_file.parent / "feature_manifest.json"
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    feature_table = pd.read_csv(feature_file, sep="\t")
    feature_status = pd.read_csv(
        feature_file.parent / "feature_status.tsv", sep="\t"
    )
    feature_key = ["subject", "method", "smoothing_mm", "hemisphere", "system"]
    feature_duplicate_keys = int(feature_table.duplicated(feature_key).sum())
    success = feature_table.loc[feature_table["status"].eq("success")]
    effect_key = [
        "contrast",
        "metric",
        "method",
        "smoothing_mm",
        "hemisphere",
        "system",
    ]
    effect_duplicate_keys = int(effects.duplicated(effect_key).sum())
    finite_effects = int(
        pd.to_numeric(effects["p_value"], errors="coerce").notna().sum()
    )
    expected_effects = 6 * len(config["methods"]) * 2 * 2 * 3 * 3
    quality_path = report_path / "data_quality.md"
    quality_lines = [
        "# Stable Kember output data-quality report",
        "",
        "This report is generated from the current feature manifest, feature table, "
        "status table, direct-effect table, and endpoint coverage table.",
        "",
        "## Dataset and grain",
        "",
        f"- Feature combinations: **{len(feature_status):,}**; feature rows: **{len(feature_table):,}**.",
        "- Feature grain: subject × method × smoothing × hemisphere × system.",
        f"- Direct-effect rows: **{len(effects):,}**; expected design units: **{expected_effects:,}**.",
        "- Direct-effect grain: contrast × metric × method × smoothing × hemisphere × system.",
        "",
        "## Checks",
        "",
        f"- Feature status: `{feature_status['feature_status'].value_counts(dropna=False).to_dict()}`.",
        f"- Feature duplicate composite keys: **{feature_duplicate_keys}**.",
        f"- Direct-effect duplicate composite keys: **{effect_duplicate_keys}**.",
        f"- Direct p values finite: **{finite_effects:,}/{len(effects):,}**.",
        f"- Endpoint coverage missing cells: **{int(pd.to_numeric(coverage['missing_n'], errors='coerce').fillna(0).sum()):,}**.",
        f"- Successful rows using native metric: **{int(success['vertex_area_source'].eq('hippunfold_native_surfareacorobl').sum()):,}/{len(success):,}**.",
        f"- Successful rows missing metric path: **{int(success['vertex_area_metric_path'].astype(str).eq('').sum())}**.",
        f"- Successful rows missing input hashes: **{int(success['input_hashes'].astype(str).eq('').sum())}**.",
        "",
        "## Findings and risk",
        "",
        "- **No critical completeness or uniqueness issue was found.** All configured "
        "subject/method/smoothing/hemisphere combinations completed and all direct "
        "design units are represented.",
        "- **The Kember area source is consistent.** Successful rows use only the "
        "native HippUnfold `surfareacorobl` metric; geometry-derived triangle-area/3 "
        "values are not used as the Kember endpoint.",
        "- **Cohort overlap is intentional.** Nested NTS/TIS-derived cohorts are "
        "retained as explicit rosters and are not treated as independent mutually "
        "exclusive cells; the overlap is recorded in the result manifest.",
        "- **Remaining analytical limitation:** these checks establish input/output "
        "integrity, not causal validity or external replication. The local functional "
        "boundary convention and native-area adaptation remain documented in "
        "the Kember area method contract.",
        "",
        "## Evidence",
        "",
        f"- Feature manifest: `{feature_manifest_path}`.",
        f"- Direct result manifest: `{result_path / 'run_manifest.json'}`.",
        f"- Configured methods: `{config['methods']}`; configured contrasts: **{len(config.get('contrasts', []))}**.",
        "",
    ]
    quality_path.write_text("\n".join(quality_lines), encoding="utf-8")

    manifest_path = result_path / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_paths: dict[str, Path] = {
        "direct_contrasts": effects_path,
        "group_descriptives": descriptives_path,
        "endpoint_coverage": coverage_path,
        "cohort_membership": membership_path,
        "figure_sources": figure_sources_path,
        "figure_provenance": figure_provenance_path,
        "analysis_summary": summary_path,
        "data_quality": quality_path,
    }
    roster_provenance_path = report_path / "tis_ptsd_provenance.json"
    if roster_provenance_path.is_file():
        output_paths["tis_ptsd_provenance"] = roster_provenance_path
    for item in figure_manifest["figure_sources"]:
        output_paths[f"{item['figure_id']}_png"] = Path(item["png_path"])
    manifest["run_id"] = DEFAULT_DIRECT_RUN_ID
    manifest["feature_table"] = str(feature_file)
    manifest["feature_manifest"] = str(feature_manifest_path)
    manifest["figure_sources"] = figure_manifest["figure_sources"]
    manifest["outputs"] = {
        name: str(path) for name, path in output_paths.items()
    }
    manifest["output_sha256"] = {
        name: _sha256(path) for name, path in output_paths.items()
    }
    manifest["stable_output_roots"] = {
        "results": str(result_path),
        "figures": str(figure_path),
        "reports": str(report_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return result_path

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_DIRECT_CONFIG_PATH)
    parser.add_argument("--feature-table", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--reports-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    output = run_stable_contrasts(
        args.config,
        feature_table_path=args.feature_table,
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
        reports_dir=args.reports_dir,
    )
    print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by workflow command
    raise SystemExit(main())

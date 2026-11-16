"""Deterministic Task 6 reporting for the cross-dataset rest-FC analysis.

This is the sole Task 6 reporting implementation. It consumes only the reviewed
Task 1--5 interfaces and does not import experimental reporting fragments.
All outputs are resting-state Fisher-z FC and are written under the current
semantic run id so corrected artifacts can replace the previous contents.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

try:  # Support direct ``src`` imports and package imports.
    from . import contrasts, features, global_profile, inventory, localisation
    from .plotting import plot_all_figures
except ImportError:  # pragma: no cover - exercised by focused tests
    import contrasts  # type: ignore
    import features  # type: ignore
    import global_profile  # type: ignore
    import inventory  # type: ignore
    import localisation  # type: ignore
    from plotting import plot_all_figures  # type: ignore


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WORKFLOW_ROOT.parent
REPO_ROOT = PROJECT_ROOT.parent
try:  # Keep the unified planner importable from package and direct src use.
    from .branch_plan import Branch, build_branch_plan
    from .branch_plan import run_priority_batches as _run_priority_batches
except ImportError:  # pragma: no cover - exercised by direct src imports
    from branch_plan import Branch, build_branch_plan  # type: ignore
    from branch_plan import run_priority_batches as _run_priority_batches  # type: ignore

_ATLAS_SRC = PROJECT_ROOT / "external_group_atlas_analysis" / "src"
if _ATLAS_SRC.is_dir() and str(_ATLAS_SRC) not in sys.path:
    sys.path.insert(0, str(_ATLAS_SRC))
try:  # The current config decides whether the bridge is exercised.
    import atlas_compare  # type: ignore
except ImportError:  # pragma: no cover - fixture-only reporting imports
    atlas_compare = None  # type: ignore[assignment]

DEFAULT_CONFIG_PATH = WORKFLOW_ROOT / "config" / "analysis_config.json"
DEFAULT_RUN_ID = "primary_screen"
DEFAULT_SEED = 20260824
PORTABLE_REPORT_CACHE_ROOT = Path(
    "/home/yojiao/.codex/plugins/cache/openai-curated-remote/data-analytics"
)
DEFAULT_N_PERM = 499
REPORT_TITLE = "JJTMS + Stanford resting-state hippocampal FC localisation"
AVAILABLE_METHODS = ("hcp_fixed", "hcp_template_matching")
GROUP_ATLAS_METHOD = "group_atlas_gradient_spectral"
REGISTERED_METHODS = AVAILABLE_METHODS + (GROUP_ATLAS_METHOD,)
NETWORK_RESOLUTIONS = (8, 17)
HEMISPHERE_MODES = ("bilateral", "L", "R", "L-R")
REGIONS = ("anterior", "middle", "posterior")
STANFORD_ALIGNED_ESTIMANDS = ("symptom_marginal", "NTS-NTHC", "TIS-TEHC")


def _run_timestamp() -> str:
    """Return the UTC start timestamp used by one materialized run."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _portable_report_builder() -> Path:
    """Return the installed portable-report builder without pinning a cache version."""

    candidates = sorted(
        path
        for path in PORTABLE_REPORT_CACHE_ROOT.glob(
            "*/skills/build-report/scripts/deliver_portable_artifact.mjs"
        )
        if path.is_file()
    )
    if not candidates:
        raise FileNotFoundError(
            "portable report builder is unavailable under "
            f"{PORTABLE_REPORT_CACHE_ROOT}"
        )
    return candidates[-1].resolve()


def _is_finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _stable_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(float(value)):
            return "NA"
        return format(float(value), ".12g")
    return str(value)


def _json_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, np.ndarray)) else False:
        return None
    return value


def _json_rows(frame: pd.DataFrame, *, limit: int | None = None) -> list[dict[str, Any]]:
    source = frame if limit is None else frame.head(int(limit))
    rows: list[dict[str, Any]] = []
    for record in source.to_dict("records"):
        rows.append({str(key): _json_scalar(value) for key, value in record.items()})
    return rows


def _sort_for_output(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
    result = frame.copy(deep=True)
    if result.empty:
        return result.reset_index(drop=True)
    keys = [str(column) for column in (columns or result.columns) if column in result.columns]
    if keys:
        # Stable string shadows keep mixed numeric/object columns deterministic
        # without changing the reader-facing column values.
        shadow: list[str] = []
        for index, key in enumerate(keys):
            name = f"__sort_{index}"
            result[name] = result[key].map(_stable_value)
            shadow.append(name)
        result = result.sort_values(shadow, kind="stable").drop(columns=shadow)
    return result.reset_index(drop=True)


def add_source_row_ids(
    frame: pd.DataFrame,
    prefix: str,
    key_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Attach deterministic, collision-resistant source row identifiers."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    result = frame.copy(deep=True)
    if "source_row_id" in result.columns:
        result = result.drop(columns=["source_row_id"])
    keys = list(key_columns or result.columns)
    keys = [key for key in keys if key in result.columns]
    if not keys:
        keys = list(result.columns)
    identifiers: list[str] = []
    for record in result[keys].to_dict("records"):
        payload = "|".join(f"{key}={_stable_value(record.get(key))}" for key in keys)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
        identifiers.append(f"{prefix}:{digest}")
    result.insert(0, "source_row_id", identifiers)
    return result


def write_table(
    frame: pd.DataFrame,
    path: Path,
    *,
    source_row_id_prefix: str | None = None,
    key_columns: Sequence[str] | None = None,
) -> Path:
    """Write a deterministic CSV/CSV.GZ table, replacing the current file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy(deep=True)
    if source_row_id_prefix and "source_row_id" not in output.columns:
        output = add_source_row_ids(output, source_row_id_prefix, key_columns)
    csv_text = output.to_csv(
        index=False,
        lineterminator="\n",
        na_rep="",
        float_format="%.12g",
    )
    if path.name.endswith(".gz"):
        with path.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
                compressed.write(csv_text.encode("utf-8"))
    else:
        path.write_text(csv_text, encoding="utf-8", newline="")
    return path


def _write_json(payload: Mapping[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _finalize_builder_receipt(
    receipt: dict[str, Any],
    *,
    controller_attempt: int,
    prior_reader_timeouts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return a JSON-safe receipt without retaining mutable self-references."""
    finalized = dict(receipt)
    finalized["controller_attempt"] = int(controller_attempt)
    if prior_reader_timeouts:
        finalized["prior_reader_timeouts"] = [dict(item) for item in prior_reader_timeouts]
    return finalized


def _repo_relative(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"source/output path escapes repository root: {path}") from exc
    if ".." in relative.parts:
        raise ValueError(f"source path contains traversal: {path}")
    return relative.as_posix()


def _configured_path(value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return inventory.validate_rest_only_path(candidate, (REPO_ROOT,))


def build_subject_manifest(rosters: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in ("jjtms", "stanford"):
        roster = rosters.get(dataset, pd.DataFrame()).copy(deep=True)
        for record in roster.to_dict("records"):
            rows.append(
                {
                    "dataset": dataset,
                    "subject": record.get("subject"),
                    "group": record.get("group"),
                    "four_group": record.get("four_group", ""),
                    "trauma": record.get("trauma"),
                    "symptomatic": record.get("symptomatic"),
                    "source_manifest": record.get("source_manifest", ""),
                    "source_manifest_primary": record.get("source_manifest_primary", ""),
                    "source_manifest_secondary": record.get("source_manifest_secondary", ""),
                    "status": "present",
                    "failure_reason": "",
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        result = pd.DataFrame(
            columns=[
                "dataset",
                "subject",
                "group",
                "four_group",
                "trauma",
                "symptomatic",
                "source_manifest",
                "source_manifest_primary",
                "source_manifest_secondary",
                "status",
                "failure_reason",
            ]
        )
    result = _sort_for_output(result, ["dataset", "group", "subject"])
    return add_source_row_ids(result, "manifest", ["dataset", "group", "subject"])


def build_group_counts(
    rosters: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    datasets = config.get("datasets", {})
    for dataset in ("jjtms", "stanford"):
        roster = rosters.get(dataset, pd.DataFrame())
        expected = datasets.get(dataset, {}).get("expected_group_counts", {}) if isinstance(datasets, Mapping) else {}
        groups = tuple(expected) if expected else tuple(sorted(set(roster.get("group", []))))
        for group in groups:
            observed = int((roster.get("group", pd.Series(dtype=object)) == group).sum())
            target = int(expected.get(group, observed)) if isinstance(expected, Mapping) else observed
            rows.append(
                {
                    "dataset": dataset,
                    "group": group,
                    "n_expected": target,
                    "n_observed": observed,
                    "n": observed,
                    "status": "ready" if observed == target else "mismatch",
                    "failure_reason": "" if observed == target else f"expected {target}, observed {observed}",
                }
            )
    result = _sort_for_output(pd.DataFrame(rows), ["dataset", "group"])
    return add_source_row_ids(result, "group-count", ["dataset", "group"])


def _group_statistics(values: np.ndarray) -> tuple[int, float, float, float, float, str]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        return 0, np.nan, np.nan, np.nan, np.nan, "no present finite FC rows"
    mean = float(np.mean(values))
    if n < 2:
        return n, mean, np.nan, np.nan, np.nan, "fewer than two subjects for SD/95% CI"
    sd = float(np.std(values, ddof=1))
    critical = float(stats.t.ppf(0.975, n - 1))
    half = critical * sd / np.sqrt(n)
    return n, mean, sd, mean - half, mean + half, ""


def build_stanford_four_group_descriptives(views: pd.DataFrame) -> pd.DataFrame:
    """Return n/mean/SD/95% CI for every Stanford component and group."""

    if not isinstance(views, pd.DataFrame) or views.empty:
        return add_source_row_ids(pd.DataFrame(), "stanford-desc")
    source = views.loc[views["dataset"].eq("stanford")].copy()
    branch_columns = ["method", "smoothing_mm", "hemisphere_mode", "network_resolution", "region", "network"]
    rows: list[dict[str, Any]] = []
    for key, branch in source.groupby(branch_columns, sort=False, dropna=False):
        branch_values = dict(zip(branch_columns, key))
        for group in contrasts.STANFORD_CELLS:
            selected = branch.loc[branch["group"].eq(group)]
            valid = selected.loc[
                selected["status"].eq("present")
                & pd.to_numeric(selected["fisher_z"], errors="coerce").notna()
            ]
            values = pd.to_numeric(valid["fisher_z"], errors="coerce").to_numpy(dtype=float)
            n, mean, sd, ci_low, ci_high, reason = _group_statistics(values)
            if not reason and len(valid) != len(selected):
                failures = [str(value) for value in selected.loc[~selected.index.isin(valid.index), "failure_reason"].dropna().unique() if str(value)]
                reason = "; ".join(failures)
            rows.append(
                {
                    **branch_values,
                    "dataset": "stanford",
                    "group": group,
                    "n": n,
                    "mean": mean,
                    "sd": sd,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "units": "Fisher-z FC",
                    "status": "present" if n else "unavailable",
                    "failure_reason": reason,
                }
            )
    result = _sort_for_output(rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows), ["method", "smoothing_mm", "hemisphere_mode", "network_resolution", "region", "network", "group"])
    return add_source_row_ids(result, "stanford-desc", ["dataset", *branch_columns, "group"])


def build_direction_consistency(primary: pd.DataFrame) -> pd.DataFrame:
    """Compare only aligned JJTMS and Stanford rest-FC estimands.

    Method, smoothing, hemisphere, resolution, region, and network remain
    explicit columns.  These are correlated specifications, not independent
    replication rows.
    """

    if not isinstance(primary, pd.DataFrame):
        raise TypeError("primary must be a DataFrame")
    source = primary.copy(deep=True)
    if "source_row_id" not in source.columns:
        source = add_source_row_ids(source, "primary")
    keys = ["method", "smoothing_mm", "hemisphere_mode", "resolution", "region", "network"]
    jjtms = source.loc[
        source["dataset"].eq("jjtms") & source["estimand"].eq("patient-healthy")
    ].copy()
    stanford = source.loc[
        source["dataset"].eq("stanford") & source["estimand"].isin(STANFORD_ALIGNED_ESTIMANDS)
    ].copy()
    left_columns = keys + ["coefficient", "n", "source_row_id"]
    right_columns = keys + ["estimand", "coefficient", "n", "source_row_id"]
    if jjtms.empty or stanford.empty:
        columns = [
            "comparison",
            "jjtms_estimand",
            "stanford_estimand",
            *keys,
            "jjtms_coefficient",
            "stanford_coefficient",
            "direction_relation",
            "direction_consistent",
            "specification_note",
            "source_row_id_jjtms",
            "source_row_id_stanford",
        ]
        return pd.DataFrame(columns=columns)
    merged = jjtms[left_columns].merge(
        stanford[right_columns],
        on=keys,
        how="inner",
        suffixes=("_jjtms", "_stanford"),
    )
    rows: list[dict[str, Any]] = []
    for record in merged.to_dict("records"):
        left = record.get("coefficient_jjtms")
        right = record.get("coefficient_stanford")
        if _is_finite(left) and _is_finite(right) and float(left) != 0 and float(right) != 0:
            same = bool(np.sign(float(left)) == np.sign(float(right)))
            relation = "same" if same else "opposite"
            consistent: bool | None = same
        else:
            relation = "unavailable"
            consistent = None
        stanford_estimand = record.get("estimand")
        rows.append(
            {
                "comparison": f"JJTMS patient-healthy vs Stanford {stanford_estimand}",
                "jjtms_estimand": "patient-healthy",
                "stanford_estimand": stanford_estimand,
                **{key: record.get(key) for key in keys},
                "jjtms_coefficient": left,
                "stanford_coefficient": right,
                "direction_relation": relation,
                "direction_consistent": consistent,
                "specification_note": "Method/smoothing/hemisphere/resolution rows are correlated specifications, not independent replications.",
                "source_row_id_jjtms": record.get("source_row_id_jjtms"),
                "source_row_id_stanford": record.get("source_row_id_stanford"),
            }
        )
    result = _sort_for_output(pd.DataFrame(rows), ["stanford_estimand", *keys])
    return add_source_row_ids(result, "direction", ["comparison", *keys])


_CANDIDATE_COLUMNS = [
    "candidate_id",
    "status",
    "dataset",
    "estimand",
    "region",
    "hemisphere_mode",
    "method",
    "smoothing_mm",
    "resolution",
    "network_a",
    "network_b",
    "effect_a",
    "effect_b",
    "direction_a",
    "direction_b",
    "source_row_id_a",
    "source_row_id_b",
    "cross_spec_sign_consistency_a",
    "cross_spec_sign_consistency_b",
    "selection_rule",
    "future_validation_target",
]


def build_candidate_registry(primary: pd.DataFrame, *, cap: int = 5) -> pd.DataFrame:
    """Select at most ``cap`` untested network-pair hypotheses.

    The rule uses only absolute source component effects and sign stability
    across fixed/template and 2/4 mm specifications.  It never reads p/q
    values and never emits a discovery-sample pair test.
    """

    if not isinstance(primary, pd.DataFrame):
        raise TypeError("primary must be a DataFrame")
    source = primary.copy(deep=True)
    if "source_row_id" not in source.columns:
        source = add_source_row_ids(source, "primary")
    finite = source.loc[
        source["fc_estimator"].eq("fisher_z")
        & pd.to_numeric(source["coefficient"], errors="coerce").notna()
    ].copy()
    if finite.empty:
        result = pd.DataFrame(columns=_CANDIDATE_COLUMNS)
        result.attrs["empty_reason"] = "no finite absolute network-wise source rows"
        return result
    component_keys = ["dataset", "estimand", "resolution", "hemisphere_mode", "region", "network"]
    stability: dict[tuple[Any, ...], float] = {}
    for key, group in finite.groupby(component_keys, sort=False, dropna=False):
        signs = np.sign(pd.to_numeric(group["coefficient"], errors="coerce").to_numpy(dtype=float))
        signs = signs[signs != 0]
        if signs.size < 2:
            stability[tuple(key)] = 0.0
        else:
            stability[tuple(key)] = float(max(np.mean(signs > 0), np.mean(signs < 0)))
    reference = finite.loc[
        finite["method"].eq("hcp_fixed")
        & finite["smoothing_mm"].eq(2)
        & finite["hemisphere_mode"].eq("bilateral")
    ].copy()
    candidates: list[dict[str, Any]] = []
    group_keys = ["dataset", "estimand", "resolution", "hemisphere_mode", "region"]
    for key, group in reference.groupby(group_keys, sort=False, dropna=False):
        rows = group.sort_values(["network", "source_row_id"], kind="stable").to_dict("records")
        for index, first in enumerate(rows):
            first_key = tuple(first.get(column) for column in component_keys)
            first_stability = stability.get(first_key, 0.0)
            if first_stability < 0.75 or not _is_finite(first.get("coefficient")) or float(first["coefficient"]) == 0:
                continue
            for second in rows[index + 1 :]:
                second_key = tuple(second.get(column) for column in component_keys)
                second_stability = stability.get(second_key, 0.0)
                if second_stability < 0.75 or not _is_finite(second.get("coefficient")) or float(second["coefficient"]) == 0:
                    continue
                if np.sign(float(first["coefficient"])) == np.sign(float(second["coefficient"])):
                    continue
                pair = [(first, first_stability), (second, second_stability)]
                pair.sort(
                    key=lambda item: (
                        -abs(float(item[0]["coefficient"])),
                        str(item[0].get("network")),
                    )
                )
                (ordered_first, ordered_first_stability), (
                    ordered_second,
                    ordered_second_stability,
                ) = pair
                payload = "|".join(
                    _stable_value(ordered_first.get(column))
                    for column in ("dataset", "estimand", "resolution", "hemisphere_mode", "region", "network")
                ) + "|" + _stable_value(ordered_second.get("network"))
                candidate_id = "candidate:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
                candidates.append(
                    {
                        "candidate_id": candidate_id,
                        "status": "untested_candidate",
                        "dataset": ordered_first.get("dataset"),
                        "estimand": ordered_first.get("estimand"),
                        "region": ordered_first.get("region"),
                        "hemisphere_mode": ordered_first.get("hemisphere_mode"),
                        "method": ordered_first.get("method"),
                        "smoothing_mm": ordered_first.get("smoothing_mm"),
                        "resolution": ordered_first.get("resolution"),
                        "network_a": ordered_first.get("network"),
                        "network_b": ordered_second.get("network"),
                        "effect_a": float(ordered_first.get("coefficient")),
                        "effect_b": float(ordered_second.get("coefficient")),
                        "direction_a": "positive" if float(ordered_first.get("coefficient")) > 0 else "negative",
                        "direction_b": "positive" if float(ordered_second.get("coefficient")) > 0 else "negative",
                        "source_row_id_a": ordered_first.get("source_row_id"),
                        "source_row_id_b": ordered_second.get("source_row_id"),
                        "cross_spec_sign_consistency_a": ordered_first_stability,
                        "cross_spec_sign_consistency_b": ordered_second_stability,
                        "selection_rule": "Absolute network-wise source rows only; reference hcp_fixed/2mm/bilateral; opposite directions; each component sign agrees in at least 75% of available fixed/template × 2/4mm specifications; cap 5; no pair p/q test.",
                        "future_validation_target": "Pre-registered held-out or external cohort with independent FC preprocessing and target-engagement validation.",
                    }
                )
    candidates.sort(
        key=lambda row: (
            -min(abs(float(row["effect_a"])), abs(float(row["effect_b"]))),
            -abs(float(row["effect_a"])) - abs(float(row["effect_b"])),
            str(row["dataset"]),
            str(row["estimand"]),
            str(row["region"]),
            str(row["network_a"]),
            str(row["network_b"]),
        )
    )
    unique_candidates: list[dict[str, Any]] = []
    seen_candidate_ids: set[str] = set()
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        if candidate_id in seen_candidate_ids:
            continue
        seen_candidate_ids.add(candidate_id)
        unique_candidates.append(candidate)
    result = pd.DataFrame(unique_candidates[: int(cap)], columns=_CANDIDATE_COLUMNS)
    if result.empty:
        result.attrs["empty_reason"] = "no opposite-direction component pair met the 75% cross-spec sign-consistency rule"
    return result.reset_index(drop=True)



def _snapshot_frame(frame: pd.DataFrame, *, limit: int = 800) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    return _sort_for_output(frame.copy(deep=True)).head(int(limit)).reset_index(drop=True)

def build_specification_consistency(primary: pd.DataFrame, *, top_n: int = 12) -> pd.DataFrame:
    """Return top reference components with their fixed/template 2/4mm rows."""

    source = primary.copy(deep=True)
    if "source_row_id" not in source.columns:
        source = add_source_row_ids(source, "primary")
    source = source.loc[
        source["fc_estimator"].eq("fisher_z")
        & source["hemisphere_mode"].eq("bilateral")
        & source["method"].isin(AVAILABLE_METHODS)
        & pd.to_numeric(source["coefficient"], errors="coerce").notna()
    ].copy()
    columns = ["component_id", "dataset", "estimand", "resolution", "region", "network", "method", "smoothing_mm", "hemisphere_mode", "coefficient", "ci_low", "ci_high", "p_raw", "correction_status", "direction", "source_row_id", "specification_note"]
    if source.empty:
        return pd.DataFrame(columns=columns)
    source["_abs"] = source["coefficient"].abs()
    component_keys = ["dataset", "estimand", "resolution", "region", "network"]
    top = (
        source.loc[source["method"].eq("hcp_fixed") & source["smoothing_mm"].eq(2)]
        .sort_values(["_abs", *component_keys], ascending=[False, True, True, True, True, True], kind="stable")
        .drop_duplicates(component_keys)
        .head(int(top_n))[component_keys]
    )
    result = source.merge(top, on=component_keys, how="inner")
    result["component_id"] = result[component_keys].astype(str).agg("|".join, axis=1)
    result["direction"] = np.where(result["coefficient"] > 0, "positive", "negative")
    result["specification_note"] = "Fixed/template and 2/4mm are correlated specifications, not independent replications."
    return _sort_for_output(result[columns], ["dataset", "estimand", "resolution", "component_id", "method", "smoothing_mm"])


def build_negative_findings(
    *,
    tensors: Mapping[str, pd.DataFrame],
    views: Mapping[str, pd.DataFrame],
    primary: pd.DataFrame,
    auxiliary: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Summarize failures, audited unavailable branches, and pending work."""

    del views
    rows: list[dict[str, Any]] = []
    for dataset in ("jjtms", "stanford"):
        tensor = tensors.get(dataset, pd.DataFrame())
        if tensor.empty:
            continue
        for (method, status), group in tensor.groupby(["method", "status"], dropna=False, sort=True):
            if str(status) == "present":
                continue
            reasons = [str(value) for value in group.get("failure_reason", pd.Series(dtype=object)).dropna().unique() if str(value)]
            rows.append({
                "finding_type": "coverage_failure_or_unavailable",
                "dataset": dataset,
                "method": method,
                "status": status,
                "n_rows": int(len(group)),
                "reason": "; ".join(reasons[:3]) or f"status={status}",
                "interpretation": "Retained as an audited negative/unavailable branch; no zero imputation.",
            })
    finite_primary = pd.to_numeric(primary.get("coefficient", pd.Series(dtype=float)), errors="coerce")
    failed_primary = primary.loc[~np.isfinite(finite_primary.to_numpy(dtype=float))] if not primary.empty else primary
    rows.append({
        "finding_type": "primary_nonfinite_rows",
        "dataset": "both",
        "method": "all",
        "status": "retained_failure_rows",
        "n_rows": int(len(failed_primary)),
        "reason": "missing required group/component or unavailable input; q/p/effect not assigned",
        "interpretation": "The component grid remains closed despite failures; multiplicity correction is deferred until the analysis-object set is frozen.",
    })
    auxiliary_values = pd.to_numeric(auxiliary.get("p_perm", pd.Series(dtype=float)), errors="coerce")
    if auxiliary.empty or not np.isfinite(auxiliary_values.to_numpy(dtype=float)).any():
        rows.append({
            "finding_type": "auxiliary_profile_unavailable",
            "dataset": "both",
            "method": "all",
            "status": "non_gating_unavailable",
            "n_rows": int(len(auxiliary)),
            "reason": "No complete subject profiles yielded finite permutation p-values.",
            "interpretation": "Auxiliary profiles never gate primary localisation.",
        })
    atlas = config.get("group_atlas", {}) if isinstance(config, Mapping) else {}
    rows.append({
        "finding_type": "group_atlas_geometry",
        "dataset": "stanford",
        "method": GROUP_ATLAS_METHOD,
        "status": "registered_unified" if isinstance(atlas, Mapping) and atlas else "not_configured",
        "n_rows": 0,
        "reason": "Group-atlas geometry is executed by the unified priority plan and reported separately from subject-level FC rows.",
        "interpretation": "HC-only group-map comparisons are descriptive geometry diagnostics; they do not enter subject-level FC inference.",
    })
    result = _sort_for_output(pd.DataFrame(rows), ["finding_type", "dataset", "method", "status"])
    return add_source_row_ids(result, "negative", ["finding_type", "dataset", "method", "status"])


def _source_objects(run_id: str) -> list[dict[str, Any]]:
    result_root = f"stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/results/{run_id}"
    figure_root = f"stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/figures/{run_id}"
    entries = [
        ("subject_manifest", "Frozen subject manifest", f"{result_root}/subject_manifest.csv.gz"),
        ("group_counts", "Frozen group counts", f"{result_root}/group_counts.csv.gz"),
        ("fc_tensor", "Full subject-level Fisher-z FC tensor", f"{result_root}/fc_tensor.csv.gz"),
        ("hemisphere_views", "Bilateral, L, R, and L-R views", f"{result_root}/hemisphere_views.csv.gz"),
        ("primary_localisation", "Full primary localisation table", f"{result_root}/primary_localisation.csv.gz"),
        ("auxiliary_profiles", "Auxiliary non-gating permutation profiles", f"{result_root}/auxiliary_profiles.csv.gz"),
        ("stanford_descriptives", "Stanford four-group descriptives", f"{result_root}/stanford_four_group_descriptives.csv.gz"),
        ("direction_consistency", "Aligned cross-dataset direction table", f"{result_root}/direction_consistency.csv.gz"),
        ("candidate_registry", "Small untested candidate registry", f"{result_root}/candidate_network_contrasts.csv.gz"),
        ("negative_findings", "Negative and unavailable findings", f"{result_root}/negative_findings.csv.gz"),
        ("specification_consistency", "Correlated specification consistency table", f"{result_root}/specification_consistency.csv.gz"),
        ("branch_manifest", "Unified priority branch execution ledger", f"{result_root}/branch_manifest.json"),
        ("group_atlas_inventory", "HC-referenced group-atlas inventory", f"{result_root}/group_atlas/atlas_inventory.csv"),
        ("group_atlas_comparisons", "HC-only group-atlas comparisons", f"{result_root}/group_atlas/matched_group_pair_comparison.csv"),
        ("group_atlas_contrasts", "HC-only group-atlas descriptive contrasts", f"{result_root}/group_atlas/descriptive_contrasts.csv"),
        ("figure_primary_effect_ci", "Primary effect CI figure", f"{figure_root}/primary_effect_ci.png"),
        ("figure_heatmaps", "Primary localisation heatmaps", f"{figure_root}/primary_localisation_heatmaps.png"),
        ("figure_profiles", "Stanford four-group profiles", f"{figure_root}/stanford_four_group_profiles.png"),
        ("figure_consistency", "Specification consistency figure", f"{figure_root}/specification_consistency.png"),
        ("figure_auxiliary", "Auxiliary profile overview figure", f"{figure_root}/auxiliary_profile_overview.png"),
        ("analysis_config", "Frozen analysis configuration", "stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/config/analysis_config.json"),
        ("source_inventory", "Reviewed inventory interface", "stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/src/inventory.py"),
        ("source_features", "Reviewed FC feature interface", "stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/src/features.py"),
        ("source_localisation", "Reviewed localisation interface", "stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/src/localisation.py"),
        ("source_global_profile", "Reviewed auxiliary profile interface", "stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/src/global_profile.py"),
        ("source_contrasts", "Frozen contrast interface", "stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/src/contrasts.py"),
        ("source_branch_plan", "Unified priority branch planner", "stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/src/branch_plan.py"),
        ("source_atlas_compare", "Unified group-atlas geometry implementation", "stanford_jjtms_multiscale_hippocampal_exploration/external_group_atlas_analysis/src/atlas_compare.py"),
        ("source_reporting", "Task 6 reporting implementation", "stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/src/reporting.py"),
        ("source_plotting", "Task 6 static plotting implementation", "stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/src/plotting.py"),
    ]
    table_ids = {"subject_manifest", "group_counts", "fc_tensor", "hemisphere_views", "primary_localisation", "auxiliary_profiles", "stanford_descriptives", "direction_consistency", "candidate_registry", "negative_findings", "specification_consistency"}
    sources = []
    for source_id, label, path in entries:
        source = {"id": source_id, "label": label, "path": path}
        if source_id in table_ids:
            source["query"] = {
                "sql": f"SELECT * FROM read_csv_auto(\'{path}\')",
                "description": "Reviewed deterministic CSV.GZ snapshot rows used by the bounded report view.",
                "engine": "duckdb",
                "language": "sql",
                "tables_used": [path],
                "filters": ["Frozen Task 6 run snapshot; bounded rows are selected only for report display."],
            }
        sources.append(source)
    return sources


def _top_primary_snapshot(primary: pd.DataFrame, limit: int = 48) -> pd.DataFrame:
    frame = primary.copy(deep=True)
    if frame.empty:
        return frame
    frame["coefficient"] = pd.to_numeric(frame["coefficient"], errors="coerce")
    frame["_abs"] = frame["coefficient"].abs()
    frame = frame.loc[np.isfinite(frame["coefficient"].to_numpy(dtype=float))].copy()
    frame = frame.sort_values(["_abs", "dataset", "estimand", "region", "network"], ascending=[False, True, True, True, True], kind="stable").head(limit).copy()
    frame["plot_index"] = np.arange(len(frame), dtype=float)
    frame["plot_label"] = frame["dataset"].str.upper() + " · " + frame["estimand"] + " · " + frame["region"] + "/" + frame["network"]
    return frame.drop(columns=["_abs"])


def _heatmap_snapshot(primary: pd.DataFrame, limit: int = 1600) -> pd.DataFrame:
    frame = primary.loc[
        primary["method"].eq("hcp_fixed")
        & primary["smoothing_mm"].eq(2)
        & primary["hemisphere_mode"].isin(("bilateral", "L-R"))
        & primary["estimand"].isin(("patient-healthy", "symptom_marginal"))
    ].copy()
    if frame.empty:
        return frame
    all_networks = (*features.DEFAULT_NETWORK_ORDER_8, *features.DEFAULT_NETWORK_ORDER_17)
    network_order = {network: index for index, network in enumerate(all_networks)}
    frame["x_index"] = frame["network"].map(network_order).fillna(0).astype(float)
    frame["y_index"] = frame["region"].map({region: index for index, region in enumerate(REGIONS)}).astype(float)
    frame["branch"] = frame["method"] + " · " + frame["hemisphere_mode"]
    columns = [
        "source_row_id", "dataset", "estimand", "resolution", "method",
        "smoothing_mm", "hemisphere_mode", "region", "network", "n",
        "coefficient", "p_raw", "correction_status", "x_index", "y_index", "branch",
    ]
    frame = frame.loc[:, [column for column in columns if column in frame.columns]]
    return _sort_for_output(
        frame, ["dataset", "resolution", "method", "hemisphere_mode", "region", "network"]
    ).head(limit)


def _profile_snapshot(descriptives: pd.DataFrame, limit: int = 400) -> pd.DataFrame:
    required = {"method", "smoothing_mm", "hemisphere_mode", "dataset", "network_resolution", "region", "network", "group"}
    if not isinstance(descriptives, pd.DataFrame) or not required.issubset(descriptives.columns):
        return pd.DataFrame()
    frame = descriptives.loc[
        descriptives["method"].eq("hcp_fixed")
        & descriptives["smoothing_mm"].eq(2)
        & descriptives["hemisphere_mode"].eq("bilateral")
        & descriptives["dataset"].eq("stanford")
    ].copy()
    if frame.empty:
        return frame
    order8 = {network: index for index, network in enumerate(features.DEFAULT_NETWORK_ORDER_8)}
    order17 = {network: index for index, network in enumerate(features.DEFAULT_NETWORK_ORDER_17)}
    frame["x_index"] = [
        float((0 if int(resolution) == 8 else 100) + (REGIONS.index(str(region)) if str(region) in REGIONS else 0) * 20 + (order8 if int(resolution) == 8 else order17).get(str(network), 0))
        for resolution, region, network in zip(
            frame["network_resolution"], frame["region"], frame["network"]
        )
    ]
    frame["y_index"] = frame["group"].map({group: index for index, group in enumerate(contrasts.STANFORD_CELLS)}).astype(float)
    columns = [
        "method", "smoothing_mm", "hemisphere_mode", "network_resolution",
        "region", "network", "dataset", "group", "n", "mean", "ci_low",
        "ci_high", "x_index", "y_index",
    ]
    frame = frame.loc[:, [column for column in columns if column in frame.columns]]
    return _sort_for_output(frame, ["network_resolution", "group", "region", "network"]).head(limit)


def _auxiliary_snapshot(auxiliary: pd.DataFrame, limit: int = 300) -> pd.DataFrame:
    required = {"method", "smoothing_mm", "hemisphere_mode", "network_resolution", "dataset", "estimand", "profile_scope", "region"}
    if not isinstance(auxiliary, pd.DataFrame) or not required.issubset(auxiliary.columns):
        return pd.DataFrame()
    frame = auxiliary.loc[
        auxiliary["method"].eq("hcp_fixed")
        & auxiliary["smoothing_mm"].eq(2)
        & auxiliary["hemisphere_mode"].eq("bilateral")
        & auxiliary["network_resolution"].eq(8)
    ].copy()
    if frame.empty:
        return frame
    frame["plot_index"] = np.arange(len(frame), dtype=float)
    frame["label"] = frame["dataset"] + " · " + frame["estimand"] + " · " + frame["profile_scope"] + " · " + frame["region"]
    failure = frame.get("failure_reason", pd.Series("", index=frame.index)).fillna("").astype(str).str.strip()
    frame["profile_status"] = np.where(failure.eq(""), "available", "failed")
    columns = [
        "source_row_id", "dataset", "estimand", "method", "smoothing_mm",
        "hemisphere_mode", "network_resolution", "fc_estimator", "profile_scope",
        "region", "n_total", "n_complete", "effective_dimension", "statistic",
        "p_perm", "n_perm", "derived_seed", "profile_status", "plot_index", "label",
    ]
    frame = frame.loc[:, [column for column in columns if column in frame.columns]]
    return _sort_for_output(frame, ["dataset", "estimand", "profile_scope", "region"]).head(limit)


def _direction_snapshot(direction: pd.DataFrame, limit: int = 300) -> pd.DataFrame:
    columns = [
        "comparison",
        "method",
        "hemisphere_mode",
        "resolution",
        "region",
        "network",
        "jjtms_coefficient",
        "stanford_coefficient",
        "direction_relation",
    ]
    if not isinstance(direction, pd.DataFrame) or not set(columns).issubset(direction.columns):
        return pd.DataFrame(columns=columns)
    return _snapshot_frame(direction.loc[:, columns], limit=limit)


def _consistency_snapshot(consistency: pd.DataFrame, limit: int = 48) -> pd.DataFrame:
    required = {
        "component_id", "dataset", "estimand", "resolution", "region", "network",
        "method", "smoothing_mm", "hemisphere_mode", "coefficient", "direction",
    }
    if not isinstance(consistency, pd.DataFrame) or not required.issubset(consistency.columns):
        return pd.DataFrame()
    frame = consistency.copy(deep=True)
    spec_order = {
        ("hcp_fixed", 2): 0.0,
        ("hcp_fixed", 4): 1.0,
        ("hcp_template_matching", 2): 2.0,
        ("hcp_template_matching", 4): 3.0,
    }
    frame["spec_index"] = [
        spec_order.get((str(method), int(smoothing)), np.nan)
        for method, smoothing in zip(frame["method"], frame["smoothing_mm"])
    ]
    frame = frame.loc[np.isfinite(frame["spec_index"].to_numpy(dtype=float))].copy()
    frame["component_label"] = (
        frame["dataset"].str.upper()
        + " · " + frame["estimand"]
        + " · " + frame["region"]
        + "/" + frame["network"]
    )
    columns = [
        "component_id", "component_label", "dataset", "estimand", "resolution",
        "region", "network", "method", "smoothing_mm", "hemisphere_mode",
        "coefficient", "ci_low", "ci_high", "p_raw", "correction_status", "direction", "spec_index",
    ]
    frame = frame.loc[:, [column for column in columns if column in frame.columns]]
    return _sort_for_output(frame, ["component_id", "spec_index"]).head(limit)


def _table_spec(table_id: str, title: str, dataset: str, source_id: str, frame: pd.DataFrame) -> dict[str, Any]:
    columns: list[dict[str, Any]] = []
    for field in frame.columns:
        if field == "source_row_id":
            continue
        values = frame[field].dropna() if field in frame else pd.Series(dtype=object)
        numeric = bool(pd.to_numeric(values, errors="coerce").notna().all()) if not values.empty else False
        columns.append({"field": str(field), "label": str(field).replace("_", " "), "type": "number" if numeric else "text"})
    if not columns:
        columns = [{"field": "status", "label": "status", "type": "text"}]
    default_field = columns[0]["field"]
    return {
        "id": table_id,
        "title": title,
        "subtitle": "Stable source rows; full table remains in the current supporting artifact.",
        "dataset": dataset,
        "sourceId": source_id,
        "defaultSort": {"field": default_field, "direction": "asc"},
        "columns": columns,
    }


def build_artifact(
    bundle: Mapping[str, Any],
    *,
    run_id: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build the canonical bounded portable report payload."""

    timestamp = generated_at or _run_timestamp()

    primary = bundle.get("primary_localisation", pd.DataFrame())
    descriptives = bundle.get("stanford_four_group_descriptives", bundle.get("descriptives", pd.DataFrame()))
    auxiliary = bundle.get("auxiliary_profiles", pd.DataFrame())
    direction = bundle.get("direction_consistency", pd.DataFrame())
    candidates = bundle.get("candidate_registry", pd.DataFrame())
    negatives = bundle.get("negative_findings", pd.DataFrame())
    group_counts = bundle.get("group_counts", pd.DataFrame())
    consistency = bundle.get("specification_consistency", build_specification_consistency(primary))
    snapshots = {
        "summary_counts": _snapshot_frame(group_counts, limit=100),
        "primary_effect_ci": _top_primary_snapshot(primary, limit=48),
        "primary_localisation_heatmaps": _heatmap_snapshot(primary, limit=1600),
        "stanford_four_group_profiles": _profile_snapshot(descriptives, limit=400),
        "specification_consistency": _consistency_snapshot(consistency, limit=48),
        "auxiliary_profile_overview": _auxiliary_snapshot(auxiliary, limit=300),
        "direction_consistency": _direction_snapshot(direction, limit=300),
        "candidate_registry": _snapshot_frame(candidates, limit=5),
        "negative_findings": _snapshot_frame(negatives, limit=100),
    }
    manifest_sources = _source_objects(run_id)
    source_ids = {source["id"] for source in manifest_sources}
    charts = [
        {
            "id": "primary_effect_ci",
            "title": "Primary network-wise effects",
            "subtitle": "Bounded full-table rank preview; values are signed Fisher-z FC coefficients; multiplicity correction is deferred.",
            "type": "scatter",
            "dataset": "primary_effect_ci",
            "sourceId": "primary_localisation",
            "encodings": {
                "x": {"field": "coefficient", "type": "quantitative", "label": "Coefficient"},
                "y": {"field": "plot_index", "type": "quantitative", "label": "Rank"},
                "color": {"field": "dataset", "type": "nominal", "label": "Dataset"},
                "label": {"field": "plot_label", "type": "nominal", "label": "Component"},
                "tooltip": [{"field": "ci_low", "label": "CI low"}, {"field": "ci_high", "label": "CI high"}, {"field": "p_raw", "label": "raw p"}, {"field": "correction_status", "label": "correction status"}],
            },
        },
        {
            "id": "primary_localisation_heatmaps",
            "title": "Primary localisation heatmap source rows",
            "subtitle": "Bounded hcp_fixed reference at 2 mm, bilateral and L-R, for 8/17 network resolutions; full tables and static figures retain template matching.",
            "type": "scatter",
            "dataset": "primary_localisation_heatmaps",
            "sourceId": "primary_localisation",
            "encodings": {
                "x": {"field": "x_index", "type": "quantitative", "label": "Network index"},
                "y": {"field": "y_index", "type": "quantitative", "label": "Region index"},
                "color": {"field": "coefficient", "type": "quantitative", "label": "Coefficient"},
                "tooltip": [{"field": "network", "label": "Network"}, {"field": "region", "label": "Region"}, {"field": "p_raw", "label": "raw p"}, {"field": "correction_status", "label": "correction status"}],
            },
        },
        {
            "id": "stanford_four_group_profiles",
            "title": "Stanford four-group mean profiles",
            "subtitle": "Reference-branch descriptive means; common Fisher-z scale across 8/17 network views.",
            "type": "scatter",
            "dataset": "stanford_four_group_profiles",
            "sourceId": "stanford_descriptives",
            "encodings": {
                "x": {"field": "x_index", "type": "quantitative", "label": "Region/network index"},
                "y": {"field": "y_index", "type": "quantitative", "label": "Group index"},
                "color": {"field": "mean", "type": "quantitative", "label": "Mean Fisher-z FC"},
                "tooltip": [{"field": "group", "label": "Group"}, {"field": "network", "label": "Network"}, {"field": "n", "label": "n"}],
            },
        },
        {
            "id": "specification_consistency",
            "title": "Specification consistency",
            "subtitle": "Bounded point preview across fixed/template and 2/4 mm correlated specifications; the supporting static figure retains component trajectories.",
            "type": "scatter",
            "dataset": "specification_consistency",
            "sourceId": "specification_consistency",
            "encodings": {
                "x": {"field": "spec_index", "type": "quantitative", "label": "Specification index"},
                "y": {"field": "coefficient", "type": "quantitative", "label": "Coefficient"},
                "color": {"field": "dataset", "type": "nominal", "label": "Dataset"},
                "tooltip": [
                    {"field": "component_label", "label": "Component"},
                    {"field": "method", "label": "Method"},
                    {"field": "smoothing_mm", "label": "Smoothing"},
                    {"field": "direction", "label": "Direction"},
                ],
            },
        },
        {
            "id": "auxiliary_profile_overview",
            "title": "Auxiliary profile permutation overview",
            "subtitle": "Exploratory, non-gating profile p-values from 499 permutations.",
            "type": "scatter",
            "dataset": "auxiliary_profile_overview",
            "sourceId": "auxiliary_profiles",
            "encodings": {
                "x": {"field": "p_perm", "type": "quantitative", "label": "Permutation p"},
                "y": {"field": "plot_index", "type": "quantitative", "label": "Profile index"},
                "color": {"field": "dataset", "type": "nominal", "label": "Dataset"},
                "label": {"field": "label", "type": "nominal", "label": "Profile"},
                "tooltip": [{"field": "n_complete", "label": "Complete n"}, {"field": "n_perm", "label": "Permutations"}],
            },
        },
    ]
    charts = [
        chart
        for chart in charts
        if chart["id"] == "primary_effect_ci" and chart["sourceId"] in source_ids
    ]
    tables: list[dict[str, Any]] = []
    portable_snapshots = {"primary_effect_ci": snapshots["primary_effect_ci"]}
    portable_body = (
        f"# {REPORT_TITLE}\n\n"
        f"Current technical report · run `{run_id}`. Primary localisation is shown "
        "below. Stanford group-atlas geometry is executed by the same unified priority plan and kept in its own analysis-family tables. Full methods, limitations, negative findings, "
        "tables, and the five standard report figures are in the companion `technical_report.md`; the focused core figure is maintained alongside them."
    )
    blocks = [
        {"id": "title", "type": "markdown", "layout": "full", "body": portable_body},
        {"id": "primary_chart", "type": "chart", "layout": "full", "chartId": "primary_effect_ci"},
    ]
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": REPORT_TITLE,
            "description": "Technical, snapshot-first report for frozen cross-dataset resting-state Fisher-z FC localisation.",
            "generatedAt": timestamp,
            "sources": manifest_sources,
            "cards": [],
            "charts": charts,
            "tables": tables,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": timestamp,
            "status": "ready",
            "datasets": {
                key: _json_rows(value) for key, value in portable_snapshots.items()
            },
            "accessIssues": [],
        },
        "sources": manifest_sources,
    }
    return artifact


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _file_entry(path: Path, *, role: str, rows: int | None = None, schema: list[str] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": _relative(path),
        "role": role,
        "size": int(path.stat().st_size),
        "sha256": _sha256(path),
    }
    if rows is not None:
        result["row_count"] = int(rows)
    if schema is not None:
        result["schema"] = [str(value) for value in schema]
    return result


def _collect_run_outputs(
    results_dir: Path,
    figures_dir: Path,
    reports_dir: Path,
    *,
    html_generated: bool,
    builder_receipt_written: bool,
) -> list[Path]:
    """Collect current run files without inheriting optional stale artifacts."""

    excluded: set[Path] = set()
    if not html_generated:
        excluded.add((reports_dir / "report.html").resolve())
    if not builder_receipt_written:
        excluded.add((reports_dir / "builder_receipt.json").resolve())
    return sorted(
        (
            path
            for root in (results_dir, figures_dir, reports_dir)
            for path in root.rglob("*")
            if path.is_file() and path.suffix.casefold() != ".pdf" and path.resolve() not in excluded
        ),
        key=lambda path: path.as_posix(),
    )


def _render_pattern(pattern: str, *, root: Path, subject: str, smoothing: int, hemisphere: str) -> Path:
    values = {
        "subject": subject,
        "subject_id": subject.removeprefix("sub-"),
        "smoothing": int(smoothing),
        "smoothing_mm": int(smoothing),
        "smoothing_label": f"{int(smoothing)}mm",
        "hemisphere": hemisphere,
        "hemi": hemisphere,
    }
    rendered = str(pattern).format(**values)
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


def _configured_input_files(config: Mapping[str, Any], rosters: Mapping[str, pd.DataFrame]) -> list[tuple[Path, str]]:
    """Resolve only explicit paths that the reviewed feature adapter can open."""

    files: dict[str, tuple[Path, str]] = {}
    config_path = DEFAULT_CONFIG_PATH
    if config_path.is_file():
        files[_relative(config_path)] = (config_path, "configuration")
    datasets = config.get("datasets", {})
    for dataset in ("jjtms", "stanford"):
        dataset_config = datasets.get(dataset, {}) if isinstance(datasets, Mapping) else {}
        roster = rosters.get(dataset, pd.DataFrame())
        for value in (dataset_config.get("rosters", {}) if isinstance(dataset_config, Mapping) else {}).values():
            candidate = Path(str(value))
            if not candidate.is_absolute():
                candidate = REPO_ROOT / candidate
            candidate = candidate.resolve()
            if candidate.is_file():
                files[_relative(candidate)] = (candidate, "subject_manifest")
        specs = dataset_config.get("parcellations", ()) if isinstance(dataset_config, Mapping) else ()
        fc_inputs = dataset_config.get("fc_inputs", {}) if isinstance(dataset_config, Mapping) else {}
        if not isinstance(specs, (list, tuple)):
            specs = tuple(specs.values()) if isinstance(specs, Mapping) else ()
        for raw_spec in specs:
            if not isinstance(raw_spec, Mapping):
                continue
            method = str(raw_spec.get("method", ""))
            availability = str(raw_spec.get("availability", "available")).lower()
            if availability not in {"", "available"}:
                continue
            root_value = raw_spec.get("fc_root", fc_inputs.get("root"))
            if root_value is None:
                continue
            root = Path(str(root_value))
            if not root.is_absolute():
                root = REPO_ROOT / root
            root = root.resolve()
            fc_pattern = raw_spec.get("fisher_z_pattern", fc_inputs.get("fisher_z_pattern"))
            provenance_pattern = raw_spec.get("provenance_pattern", fc_inputs.get("provenance_pattern"))
            if not fc_pattern or not provenance_pattern:
                continue
            for subject in sorted(set(str(value) for value in roster.get("subject", []))):
                for smoothing in (2, 4):
                    for hemisphere in ("L", "R"):
                        candidate_paths: list[tuple[Path, str]] = [
                            (_render_pattern(str(fc_pattern), root=root, subject=subject, smoothing=smoothing, hemisphere=hemisphere), "fc_array"),
                            (_render_pattern(str(provenance_pattern), root=root, subject=subject, smoothing=smoothing, hemisphere=hemisphere), "fc_provenance"),
                        ]
                        for sidecar in raw_spec.get("provenance_sidecar_patterns", ()):
                            if not isinstance(sidecar, Mapping):
                                continue
                            side_root = Path(str(sidecar.get("root", root)))
                            if not side_root.is_absolute():
                                side_root = REPO_ROOT / side_root
                            side_pattern = sidecar.get("pattern")
                            if side_pattern:
                                candidate_paths.append((_render_pattern(str(side_pattern), root=side_root.resolve(), subject=subject, smoothing=smoothing, hemisphere=hemisphere), "provenance_sidecar"))
                        for key, label in (("labels_pattern", "region_labels"), ("valid_mask_pattern", "valid_mask"), ("region_weight_pattern", "region_weights")):
                            pattern = raw_spec.get(key)
                            if pattern:
                                side_root = Path(str(raw_spec.get(f"{key.removesuffix('_pattern')}_root", root)))
                                if not side_root.is_absolute():
                                    side_root = REPO_ROOT / side_root
                                candidate_paths.append((_render_pattern(str(pattern), root=side_root.resolve(), subject=subject, smoothing=smoothing, hemisphere=hemisphere), label))
                        for candidate, label in candidate_paths:
                            if candidate.is_file() and candidate.is_relative_to(REPO_ROOT.resolve()):
                                files[_relative(candidate)] = (candidate, label)
            registry = raw_spec.get("registry_path")
            if registry:
                candidate = Path(str(registry))
                if not candidate.is_absolute():
                    candidate = REPO_ROOT / candidate
                candidate = candidate.resolve()
                if candidate.is_file():
                    files[_relative(candidate)] = (candidate, "parcel_registry")
            for key in ("labels_root", "valid_mask_root"):
                value = raw_spec.get(key)
                if value:
                    candidate = Path(str(value))
                    if not candidate.is_absolute():
                        candidate = REPO_ROOT / candidate
                    candidate = candidate.resolve()
                    if candidate.is_file():
                        files[_relative(candidate)] = (candidate, "feature_sidecar_root")
    return [files[key] for key in sorted(files)]




def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return path


def _write_tables(
    tables: Mapping[str, pd.DataFrame],
    results_dir: Path,
) -> tuple[list[Path], dict[str, dict[str, Any]]]:
    paths: list[Path] = []
    meta: dict[str, dict[str, Any]] = {}
    for name, frame in tables.items():
        path = results_dir / f"{name}.csv.gz"
        output = frame.copy(deep=True)
        if "source_row_id" not in output.columns:
            output = add_source_row_ids(output, name)
        write_table(output, path)
        paths.append(path)
        meta[name] = {"rows": int(len(output)), "schema": [str(value) for value in output.columns]}
    return paths, meta


def _technical_report(
    run_id: str,
    tables: Mapping[str, pd.DataFrame],
    figures: Mapping[str, Mapping[str, str]],
    candidates: pd.DataFrame,
    *,
    generated_at: str | None = None,
    seed: int = DEFAULT_SEED,
    n_perm: int = DEFAULT_N_PERM,
) -> str:
    timestamp = generated_at or _run_timestamp()
    primary = tables["primary_localisation"]
    finite = pd.to_numeric(primary.get("coefficient", pd.Series(dtype=float)), errors="coerce")
    finite_n = int(np.isfinite(finite.to_numpy(dtype=float)).sum())
    candidate_line = "No candidate network-pair registry entry met the predeclared opposite-direction/75% cross-spec rule." if candidates.empty else f"The registry contains {len(candidates)} untested candidate network contrasts; it includes no discovery-sample pair p/q tests."
    figure_lines = "\n".join(f"- `{name}`: `{Path(info['png']).name}` (PNG)" for name, info in figures.items())
    return f"""# {REPORT_TITLE}

Run `{run_id}` · run started {timestamp} · Fisher-z FC only.

## Technical summary

The full primary table contains {finite_n} finite region-by-network effects across JJTMS and Stanford. Multiplicity correction is deferred until the final analysis-object set is frozen, so this report makes no q-value or threshold claim. These results localise group associations in resting-state FC; they do not establish causality, treatment normalisation, or a TMS target.

The available fixed/template branches are compared across 2/4 mm and bilateral/L/R/L-R views. The Stanford K=3 group-atlas geometry family is executed by the same strict priority plan, uses the current HC atlas as reference, and emits only non-HC-versus-HC descriptive map comparisons. Its geometry rows remain separate from subject-level FC rows so the data types are not mixed.

## Key findings with visual evidence

The primary effect/CI figure is a deterministic top-absolute preview selected from the full primary table, with JJTMS patient-healthy and all Stanford estimand panels represented when finite rows are available. Points show signed coefficients with 95% confidence intervals; all signed panels retain a zero reference. The displayed effects are raw and unadjusted pending analysis-object freeze.

The localisation heatmaps show anterior/middle/posterior × network structure for both 8- and 17-network resolutions on fixed/template 2 mm bilateral and L-R reference branches. The Stanford profile heatmaps are descriptive four-group means on a common scale. The specification panel is a sensitivity comparison, not an independent replication. The auxiliary profile panel is exploratory/non-gating (seed={seed}, n_perm={n_perm}).

{figure_lines}

{candidate_line}

## Scope, data, and metric definitions

The frozen rosters are JJTMS healthy=67/patient=27 and Stanford NTHC=51, TEHC=40, NTS=56, TIS=47. FC is absolute Fisher-z vertex-to-network connectivity aggregated to anterior, middle, and posterior and to canonical family-8/family-17 networks. Bilateral is the recorded region-weighted mean of valid L/R values; L-R is the within-subject Fisher-z difference.

Cross-workflow topology metrics are outside this cross-dataset rest-FC scope.

## Methodology and permutation specification

Primary localisation uses unadjusted cell-means models, HC3 standard errors, finite-residual-df two-sided t inference, and 95% confidence intervals. Multiplicity correction is not applied at this stage; the final analysis-object set is not frozen. JJTMS uses patient-healthy; Stanford uses symptom_marginal, trauma_marginal, interaction, NTS-NTHC, TIS-TEHC, TEHC-NTHC, and TIS-NTS.

Auxiliary subregion and whole-hippocampus profile distances use complete subject profiles and Freedman-Lane row permutations with plus-one p-values, seed={seed} and n_perm={n_perm}. They never gate primary localisation. No exhaustive network-pair tests are performed.

## Limitations, uncertainty, and negative findings

Missing/failed rows remain in the full tables with statuses and reasons; they are not zero-filled. Method, smoothing, hemisphere, and resolution variants are correlated specifications and cannot be counted as independent replications. The unified branch ledger enforces 8-network/2-mm first, then the middle tier, then 17-network/4-mm plus group-atlas 4-mm work. The cross-sectional resting-state association does not imply a causal TMS mechanism.

## Recommended next steps

- Pre-register reliability, missingness, and covariate sensitivity checks for any finite component that appears promising.
- Validate any candidate network contrast in an independent held-out/external cohort without discovery-sample pair testing.
- Pair surviving anterior/middle/posterior effects with parcel-localisation and target-engagement evidence before funding hypotheses.

## Further questions

Which components survive independent preprocessing and subject-level reliability checks? How stable are the HC-referenced group-atlas geometry summaries across specifications? Are candidate effects preserved in held-out data and related to target engagement?

Supporting full tables: `results/{run_id}/`; report payload: `reports/{run_id}/artifact.json`.
"""


def _chart_map(run_id: str, figures: Mapping[str, Mapping[str, str]]) -> str:
    rows = [
        "# Chart map",
        "",
        "Static figures are supporting evidence; the current reader-facing materials are `technical_report.md` and `artifact.json`. A portable HTML is optional when the report builder is available.",
        "",
        "| Figure | Question / form | Source table | Selection or caveat |",
        "| --- | --- | --- | --- |",
        "| primary_effect_ci | Full-table top absolute components; faceted dot + 95% CI | primary_localisation.csv.gz | top 4 per dataset × estimand; raw effects, correction deferred |",
        "| primary_localisation_heatmaps | Signed region × network localisation | primary_localisation.csv.gz | fixed/template, 2mm, bilateral/L-R references; specifications not replications |",
        "| stanford_four_group_profiles | NTHC/TEHC/NTS/TIS means | stanford_four_group_descriptives.csv.gz | common scale, descriptive |",
        "| specification_consistency | Direction/effect across fixed/template and 2/4mm | specification_consistency.csv.gz | correlated sensitivity specifications |",
        "| auxiliary_profile_overview | Profile permutation overview | auxiliary_profiles.csv.gz | exploratory/non-gating, n_perm=499 |",
        "| group_atlas_geometry | HC-only K=3 group-map geometry comparison | results/{run_id}/group_atlas/ | descriptive; no subject-level FC inference |",
        "",
        f"Run: `{run_id}`",
    ]
    return "\n".join(rows)


def _source_notes(run_id: str, input_files: list[tuple[Path, str]], tables: Mapping[str, pd.DataFrame]) -> str:
    return "\n".join([
        "# Source notes",
        "",
        f"- Run: `{run_id}`; source scope: resting-state Fisher-z FC plus descriptive Stanford group-atlas geometry.",
        "- Exact input files are listed with SHA-256 in `results/%s/run_manifest.json`; configured missing paths were not opened." % run_id,
        "- Full tensors/views/localisation/profile/descriptive tables are current supporting artifacts; `technical_report.md` is the reader-facing fallback when HTML is not built.",
        "- The Stanford K=3 group-atlas geometry family is executed in the same priority-gated run; its HC-only descriptive tables are kept separate from subject-level FC inference.",
        "- Candidate registry is capped at five and contains no pairwise discovery-sample p/q test.",
        "- Input file count observed before writing outputs: %d; primary table rows: %d." % (len(input_files), len(tables["primary_localisation"])),
    ])


def _group_atlas_input_root(config: Mapping[str, Any]) -> Path | None:
    atlas_config = config.get("group_atlas")
    if not isinstance(atlas_config, Mapping):
        return None
    raw_root = atlas_config.get("input_root")
    if raw_root is None:
        raise ValueError("group_atlas.input_root is required for the unified run")
    root = Path(str(raw_root)).expanduser()
    if not root.is_absolute():
        root = REPO_ROOT / root
    return root.resolve()


def _branch_manifest(
    plan: Sequence[Branch],
    statuses: Mapping[str, str],
    records: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    branch_rows: list[dict[str, Any]] = []
    record_by_id = {str(record["branch_id"]): dict(record) for record in records}
    for branch in plan:
        row = {
            "branch_id": branch.branch_id,
            "analysis_family": branch.analysis_family,
            "method": branch.method,
            "dataset": branch.dataset,
            "network_resolution": branch.network_resolution,
            "smoothing_mm": branch.smoothing_mm,
            "hemisphere_modes": list(branch.hemisphere_modes),
            "group": branch.group,
            "assignment_variant": branch.assignment_variant,
            "priority_tier": branch.priority_tier,
            "status": statuses.get(branch.branch_id, "pending"),
        }
        execution = record_by_id.get(branch.branch_id)
        if execution and "error" in execution:
            row["error"] = execution["error"]
        branch_rows.append(row)
    tiers: list[dict[str, Any]] = []
    for tier in sorted({branch.priority_tier for branch in plan}):
        tier_rows = [row for row in branch_rows if row["priority_tier"] == tier]
        tiers.append(
            {
                "priority_tier": tier,
                "branch_count": len(tier_rows),
                "completed": sum(row["status"] == "complete" for row in tier_rows),
                "failed": sum(row["status"] == "failed" for row in tier_rows),
                "pending": sum(row["status"] == "pending" for row in tier_rows),
            }
        )
    policy = config.get("unified_branch_policy", {})
    return {
        "schema_version": 1,
        "barrier": policy.get("barrier", "strict")
        if isinstance(policy, Mapping)
        else "strict",
        "branch_count": len(branch_rows),
        "tiers": tiers,
        "branches": branch_rows,
    }


def run_priority_batches(
    config: Mapping[str, Any],
    rosters: Mapping[str, pd.DataFrame],
    plan: Iterable[Branch] | None = None,
    execute_branch: Callable[[Branch], str | None] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, str], dict[str, Any]]:
    """Materialize subject FC and group-atlas branches behind one strict barrier."""

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    branch_plan = tuple(plan or build_branch_plan(config))
    tensor_parts: dict[str, list[pd.DataFrame]] = {}
    execution_records: list[dict[str, Any]] = []

    def execute(branch: Branch) -> str | None:
        if execute_branch is not None:
            return execute_branch(branch)
        if branch.analysis_family == "subject_fc":
            if branch.dataset not in rosters:
                raise KeyError(f"missing roster for branch dataset {branch.dataset!r}")
            tensor = features.build_fc_tensor(
                branch.dataset,
                rosters[branch.dataset],
                config,
                methods=(branch.method,),
                smoothings_mm=(branch.smoothing_mm,),
                network_resolutions=(branch.network_resolution,),
            )
            tensor_parts.setdefault(branch.dataset, []).append(tensor)
            return "complete"
        if branch.analysis_family == "group_atlas_geometry":
            if atlas_compare is None:
                raise ImportError("group-atlas comparison module is unavailable")
            input_root = _group_atlas_input_root(config)
            if input_root is None:
                raise ValueError("group-atlas branch is missing its input root")
            atlas_config = config["group_atlas"]
            expected_n_vertices = int(atlas_config.get("expected_n_vertices", 512))
            cells = atlas_compare.discover_atlases(
                input_root,
                expected_n_vertices=expected_n_vertices,
                groups=(branch.group,),
                variants=(branch.assignment_variant,),
                smoothings_mm=(branch.smoothing_mm,),
                hemispheres=branch.hemisphere_modes,
            )
            if len(cells) != 1:
                raise RuntimeError(
                    f"expected one atlas cell for {branch.branch_id}, got {len(cells)}"
                )
            cell = cells[0]
            atlas_compare.compute_atlas_region_area(
                cell.vertices,
                cell.faces,
                cell.labels,
            )
            return "complete"
        raise ValueError(f"unknown analysis family: {branch.analysis_family!r}")

    records, statuses = _run_priority_batches(branch_plan, execute)
    execution_records.extend(records)
    tensors: dict[str, pd.DataFrame] = {}
    views: dict[str, pd.DataFrame] = {}
    for dataset, parts in tensor_parts.items():
        tensor = pd.concat(parts, ignore_index=True)
        tensor.attrs["network_ordinals"] = {
            key: value
            for part in parts
            for key, value in part.attrs.get("network_ordinals", {}).items()
        }
        tensors[dataset] = _sort_for_output(
            tensor,
            [
                "dataset",
                "subject",
                "method",
                "smoothing_mm",
                "hemisphere",
                "region",
                "network_resolution",
                "network",
            ],
        )
        tensors[dataset].attrs["network_ordinals"] = tensor.attrs["network_ordinals"]
        views[dataset] = features.build_hemisphere_views(tensors[dataset])
    manifest = _branch_manifest(
        branch_plan,
        statuses,
        execution_records,
        config=config,
    )
    return tensors, views, statuses, manifest


def run_analysis(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    run_id: str = DEFAULT_RUN_ID,
    seed: int = DEFAULT_SEED,
    n_perm: int = DEFAULT_N_PERM,
    results_root: Path | None = None,
    figures_root: Path | None = None,
    reports_root: Path | None = None,
    run_command: str | None = None,
    build_html: bool = True,
) -> dict[str, Any]:
    """Run the complete real-data Task 6 materialization once.

    The current semantic run path is stable. Rerunning it replaces the current files so
    the tables, figures, reports, and manifest remain synchronized.
    """

    if isinstance(seed, bool) or int(seed) != seed:
        raise ValueError("seed must be an integer")
    if isinstance(n_perm, bool) or int(n_perm) != n_perm or int(n_perm) < 0:
        raise ValueError("n_perm must be a non-negative integer")
    run_started_at = _run_timestamp()
    config_path = Path(config_path).resolve()
    config = inventory.load_config(config_path)
    result_root = Path(results_root or (WORKFLOW_ROOT / "results"))
    figure_root = Path(figures_root or (WORKFLOW_ROOT / "figures"))
    report_root = Path(reports_root or (WORKFLOW_ROOT / "reports"))
    effective_run_id = str(run_id)
    results_dir = result_root / effective_run_id
    figures_dir = figure_root / effective_run_id
    reports_dir = report_root / effective_run_id
    for directory in (results_dir, figures_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rosters = inventory.build_rosters(config)
    input_files = _configured_input_files(config, rosters)
    coverage = inventory.inspect_fc_inputs(config, rosters)
    plan = build_branch_plan(config)
    tensors, views, branch_statuses, branch_manifest = run_priority_batches(
        config,
        rosters,
        plan,
    )
    for dataset in tuple(tensors):
        tensors[dataset] = add_source_row_ids(
            tensors[dataset],
            f"tensor-{dataset}",
            [
                "dataset",
                "subject",
                "method",
                "smoothing_mm",
                "hemisphere",
                "region",
                "network_resolution",
                "network",
            ],
        )
        views[dataset] = add_source_row_ids(
            views[dataset],
            f"view-{dataset}",
            [
                "dataset",
                "subject",
                "method",
                "smoothing_mm",
                "hemisphere_mode",
                "region",
                "network_resolution",
                "network",
            ],
        )
    if set(tensors) != {"jjtms", "stanford"}:
        raise RuntimeError(
            "unified branch plan did not produce both subject-level FC datasets"
        )
    group_atlas_result: dict[str, Path] | None = None
    atlas_input_root = _group_atlas_input_root(config)
    if atlas_input_root is not None:
        if atlas_compare is None:
            raise ImportError("group-atlas comparison module is unavailable")
        group_atlas_result = atlas_compare.run_analysis(
            input_root=atlas_input_root,
            run_id=effective_run_id,
            results_dir=results_dir / "group_atlas",
            figure_dir=figures_dir / "group_atlas",
            report_dir=reports_dir / "group_atlas",
        )
    branch_manifest_path = results_dir / "branch_manifest.json"
    _write_json(branch_manifest, branch_manifest_path)
    tensor_all = pd.concat(list(tensors.values()), ignore_index=True)
    view_all = pd.concat(list(views.values()), ignore_index=True)
    tensor_all = _sort_for_output(tensor_all, ["dataset", "subject", "method", "smoothing_mm", "hemisphere", "region", "network_resolution", "network"])
    view_all = _sort_for_output(view_all, ["dataset", "subject", "method", "smoothing_mm", "hemisphere_mode", "region", "network_resolution", "network"])
    tensor_all.attrs["network_ordinals"] = {**tensors["jjtms"].attrs.get("network_ordinals", {}), **tensors["stanford"].attrs.get("network_ordinals", {})}
    view_all.attrs["network_ordinals"] = tensor_all.attrs["network_ordinals"]
    jjtms_contrasts = contrasts.jjtms_contrast_matrix().assign(dataset="jjtms")
    stanford_contrasts = contrasts.stanford_contrast_matrix().assign(dataset="stanford")
    contrast_table = pd.concat([jjtms_contrasts, stanford_contrasts], ignore_index=True)
    primary = localisation.fit_local_effects(view_all, contrast_table)
    primary = add_source_row_ids(primary, "primary", ["dataset", "estimand", "resolution", "method", "smoothing_mm", "hemisphere_mode", "region", "network"])
    auxiliary = global_profile.run_global_tests(view_all, contrast_table, seed=int(seed), n_perm=int(n_perm))
    auxiliary = add_source_row_ids(auxiliary, "auxiliary", ["dataset", "estimand", "method", "smoothing_mm", "hemisphere_mode", "network_resolution", "profile_scope", "region"])
    descriptives = build_stanford_four_group_descriptives(view_all)
    direction = build_direction_consistency(primary)
    candidates = build_candidate_registry(primary)
    consistency = build_specification_consistency(primary)
    negatives = build_negative_findings(tensors=tensors, views=views, primary=primary, auxiliary=auxiliary, config=config)
    subject_manifest = build_subject_manifest(rosters)
    group_counts = build_group_counts(rosters, config)
    coverage = add_source_row_ids(coverage, "coverage")
    tables = {
        "subject_manifest": subject_manifest,
        "group_counts": group_counts,
        "input_coverage": coverage,
        "fc_tensor": tensor_all,
        "hemisphere_views": view_all,
        "primary_localisation": primary,
        "auxiliary_profiles": auxiliary,
        "stanford_four_group_descriptives": descriptives,
        "direction_consistency": direction,
        "candidate_network_contrasts": candidates,
        "negative_findings": negatives,
        "specification_consistency": consistency,
    }
    table_paths, table_meta = _write_tables(tables, results_dir)
    figure_info = plot_all_figures(primary, descriptives, auxiliary, figures_dir, run_id=effective_run_id, n_perm=int(n_perm))
    artifact = build_artifact(
        {**tables, "descriptives": descriptives},
        run_id=effective_run_id,
        generated_at=run_started_at,
    )
    artifact_path = reports_dir / "artifact.json"
    _write_json(artifact, artifact_path)
    technical_path = reports_dir / "technical_report.md"
    _write_text(
        technical_path,
        _technical_report(
            effective_run_id,
            tables,
            figure_info,
            candidates,
            generated_at=run_started_at,
            seed=int(seed),
            n_perm=int(n_perm),
        ),
    )
    _write_text(reports_dir / "chart_map.md", _chart_map(effective_run_id, figure_info))
    _write_text(reports_dir / "source_notes.md", _source_notes(effective_run_id, input_files, tables))
    report_html_path = reports_dir / "report.html"
    report_html: Path | None = None
    builder_receipt_path = reports_dir / "builder_receipt.json"
    builder_receipt_written = False
    html_generated = False
    if build_html:
        try:
            builder = _portable_report_builder()
        except FileNotFoundError as exc:
            builder_receipt = {
                "ok": False,
                "stage": "skipped",
                "code": "portable_report_builder_unavailable",
                "reason": "portable report builder is not installed",
                "error": str(exc),
            }
        else:
            command = [
                "node", str(builder), "--input", str(artifact_path), "--output", str(report_html_path),
                "--ready-timeout-ms", "10000", "--timeout-ms", "20000",
            ]
            prior_reader_timeouts: list[dict[str, Any]] = []
            for builder_attempt in (1, 2):
                completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
                receipt_text = completed.stdout.strip() or completed.stderr.strip()
                try:
                    builder_receipt = json.loads(receipt_text) if receipt_text else {"ok": False, "error": "builder returned no receipt"}
                except json.JSONDecodeError:
                    builder_receipt = {"ok": False, "error": receipt_text}
                if completed.returncode == 0 or builder_receipt.get("code") != "reader_timeout":
                    break
                prior_reader_timeouts.append(dict(builder_receipt))
            builder_receipt = _finalize_builder_receipt(
                builder_receipt,
                controller_attempt=builder_attempt,
                prior_reader_timeouts=prior_reader_timeouts,
            )
            _write_json(builder_receipt, builder_receipt_path)
            builder_receipt_written = True
            if completed.returncode != 0:
                raise RuntimeError(f"portable report builder failed: {builder_receipt}")
            if not report_html_path.is_file():
                raise RuntimeError("portable report builder reported success but wrote no HTML")
            report_html = report_html_path
            html_generated = True
    else:
        builder_receipt = {"ok": False, "stage": "skipped", "reason": "build_html=False"}
    if not builder_receipt_written:
        _write_json(builder_receipt, builder_receipt_path)
        builder_receipt_written = True
    all_outputs = _collect_run_outputs(
        results_dir,
        figures_dir,
        reports_dir,
        html_generated=html_generated,
        builder_receipt_written=builder_receipt_written,
    )
    manifest_path = results_dir / "run_manifest.json"
    output_entries: list[dict[str, Any]] = []
    meta_by_path = {str(results_dir / f"{name}.csv.gz"): details for name, details in table_meta.items()}
    for path in sorted(all_outputs, key=lambda item: _relative(item)):
        if path.resolve() == manifest_path.resolve():
            continue
        relative = str(path)
        details = meta_by_path.get(relative, {})
        output_entries.append(_file_entry(path, role="output", rows=details.get("rows"), schema=details.get("schema")))
    input_entries: list[dict[str, Any]] = []
    for path, role in input_files:
        if path.is_file():
            input_entries.append(_file_entry(path, role=role))
    source_code = [
        WORKFLOW_ROOT / "src" / "reporting.py",
        WORKFLOW_ROOT / "src" / "run_primary.py",
        WORKFLOW_ROOT / "src" / "plotting.py",
        WORKFLOW_ROOT / "src" / "inventory.py",
        WORKFLOW_ROOT / "src" / "features.py",
        WORKFLOW_ROOT / "src" / "branch_plan.py",
        WORKFLOW_ROOT / "src" / "localisation.py",
        WORKFLOW_ROOT / "src" / "global_profile.py",
        WORKFLOW_ROOT / "src" / "contrasts.py",
        PROJECT_ROOT / "external_group_atlas_analysis" / "src" / "atlas_compare.py",
    ]
    for path in source_code:
        if path.is_file() and all(entry["path"] != _relative(path) for entry in input_entries):
            input_entries.append(_file_entry(path, role="source_code"))
    report_paths: dict[str, str] = {
        "artifact": _relative(artifact_path),
        "technical_markdown": _relative(technical_path),
        "group_atlas_report": _relative(reports_dir / "group_atlas"),
        "builder_receipt": _relative(builder_receipt_path),
    }
    if report_html is not None:
        report_paths["html"] = _relative(report_html)
    manifest = {
        "schema_version": 1,
        "run_id": effective_run_id,
        "base_run_id": str(run_id),
        "created_at": run_started_at,
        "git_sha": _git_sha(),
        "python_executable": str(sys.executable),
        "python_version": platform.python_version(),
        "run_command": run_command or f"py314 src/run_primary.py --run-id {effective_run_id} --seed {int(seed)} --n-perm {int(n_perm)}",
        "seed": int(seed),
        "n_perm": int(n_perm),
        "scope": "rest-fMRI Fisher-z FC plus descriptive Stanford group-atlas geometry; beta/TMS-response/CON/native/archive inputs prohibited",
        "analysis_families": [
            "subject_fc",
            "group_atlas_geometry",
        ],
        "branch_manifest": _relative(branch_manifest_path),
        "branch_policy": branch_manifest,
        "branch_statuses": branch_statuses,
        "group_atlas_outputs": {
            key: _relative(path)
            for key, path in (group_atlas_result or {}).items()
            if isinstance(path, Path) and path.is_file()
        },
        "input_files": sorted(input_entries, key=lambda item: item["path"]),
        "output_files": output_entries,
        "output_closure": {"manifest_excluded": True, "all_other_run_files_hashed": True},
        "row_counts": {name: details["rows"] for name, details in table_meta.items()},
        "figure_names": sorted(figure_info),
        "report_paths": report_paths,
    }
    _write_json(manifest, manifest_path)
    return {
        "run_id": effective_run_id,
        "results_dir": results_dir,
        "figures_dir": figures_dir,
        "reports_dir": reports_dir,
        "tables": tables,
        "table_paths": table_paths,
        "figures": figure_info,
        "artifact_path": artifact_path,
        "report_html": report_html,
        "builder_receipt": builder_receipt,
        "manifest_path": manifest_path,
        "manifest": manifest,
    }



def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run frozen cross-dataset rest-FC Task 6 analysis/reporting")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-perm", type=int, default=DEFAULT_N_PERM)
    parser.add_argument("--no-html", action="store_true")
    args = parser.parse_args(argv)
    result = run_analysis(
        config_path=args.config,
        run_id=args.run_id,
        seed=args.seed,
        n_perm=args.n_perm,
        build_html=not args.no_html,
        run_command=" ".join(sys.argv),
    )
    print(json.dumps({"run_id": result["run_id"], "manifest": str(result["manifest_path"]), "report": str(result["report_html"]) if result["report_html"] is not None else None}, sort_keys=True))
    return 0




if __name__ == "__main__":
    raise SystemExit(main())

__all__ = [
    "add_source_row_ids", "write_table", "build_subject_manifest", "build_group_counts",
    "build_stanford_four_group_descriptives", "build_direction_consistency",
    "build_candidate_registry", "build_specification_consistency",
    "build_negative_findings", "build_artifact", "run_priority_batches",
    "run_analysis", "main",
]

"""Statistical analysis and tabular exports for cortical network area.

The primary outcome is network area as a percentage of each subject's valid
cortical midthickness area.  Absolute area in mm2 is carried beside it as a
secondary outcome.
"""

from __future__ import annotations

# Permit both ``python -m ...`` and direct execution of this entry module.
if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path
    _root = _Path(__file__).resolve().parents[3]
    if str(_root) not in _sys.path:
        _sys.path.insert(0, str(_root))
    __package__ = "scripts.visualize.cortex_network_area"

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class Comparison:
    """A fixed case-minus-control cohort comparison."""

    key: str
    case_cohort: str
    control_cohort: str
    patient_comparison: bool


COMPARISONS: tuple[Comparison, ...] = (
    Comparison("patient_vs_iowa_hc", "Patient", "Iowa HC", True),
    Comparison("patient_vs_hcp_hc", "Patient", "HCP HC", True),
    Comparison("pure_mdd_vs_iowa_hc", "Pure MDD", "Iowa HC", True),
    Comparison("pure_mdd_vs_hcp_hc", "Pure MDD", "HCP HC", True),
    Comparison("comorbid_mdd_vs_iowa_hc", "Comorbid MDD", "Iowa HC", True),
    Comparison("comorbid_mdd_vs_hcp_hc", "Comorbid MDD", "HCP HC", True),
    Comparison("iowa_hc_vs_hcp_hc", "Iowa HC", "HCP HC", False),
)


def _finite_values(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def compute_welch_summary(
    case_values: Iterable[float], control_values: Iterable[float]
) -> dict[str, float | int]:
    """Return two-sided Welch statistics with case-minus-control direction."""

    case = _finite_values(case_values)
    control = _finite_values(control_values)
    n_case = int(case.size)
    n_control = int(control.size)
    mean_case = float(np.mean(case)) if n_case else float("nan")
    mean_control = float(np.mean(control)) if n_control else float("nan")
    sd_case = float(np.std(case, ddof=1)) if n_case >= 2 else float("nan")
    sd_control = float(np.std(control, ddof=1)) if n_control >= 2 else float("nan")
    difference = mean_case - mean_control
    relative = (
        100.0 * difference / mean_control
        if np.isfinite(mean_control) and mean_control != 0.0
        else float("nan")
    )

    ci_low = ci_high = statistic = degrees_freedom = p_value = float("nan")
    cohens_d = float("nan")
    if n_case >= 2 and n_control >= 2:
        case_var = sd_case**2
        control_var = sd_control**2
        standard_error_sq = case_var / n_case + control_var / n_control
        if standard_error_sq > 0.0:
            standard_error = float(np.sqrt(standard_error_sq))
            denominator = (
                (case_var / n_case) ** 2 / (n_case - 1)
                + (control_var / n_control) ** 2 / (n_control - 1)
            )
            degrees_freedom = float(standard_error_sq**2 / denominator)
            critical = float(stats.t.ppf(0.975, degrees_freedom))
            ci_low = difference - critical * standard_error
            ci_high = difference + critical * standard_error
            result = stats.ttest_ind(case, control, equal_var=False, nan_policy="raise")
            statistic = float(result.statistic)
            p_value = float(result.pvalue)
        elif difference == 0.0:
            ci_low = ci_high = difference
            statistic = 0.0
            p_value = 1.0

        pooled_variance = (
            (n_case - 1) * case_var + (n_control - 1) * control_var
        ) / (n_case + n_control - 2)
        if pooled_variance > 0.0:
            cohens_d = float(difference / np.sqrt(pooled_variance))
        elif difference == 0.0:
            cohens_d = 0.0

    return {
        "n_case": n_case,
        "n_control": n_control,
        "mean_case": mean_case,
        "sd_case": sd_case,
        "mean_control": mean_control,
        "sd_control": sd_control,
        "difference_pp": difference,
        "relative_difference_percent": relative,
        "ci95_low_pp": ci_low,
        "ci95_high_pp": ci_high,
        "welch_t": statistic,
        "welch_df": degrees_freedom,
        "cohens_d": cohens_d,
        "p_value": p_value,
    }


_STRATA = (
    "variant",
    "method",
    "network_classification",
    "taxonomy",
    "network",
)


def _secondary_area_summary(
    case_values: Iterable[float], control_values: Iterable[float]
) -> dict[str, float]:
    case = _finite_values(case_values)
    control = _finite_values(control_values)
    return {
        "mean_case_mm2": float(np.mean(case)) if case.size else float("nan"),
        "sd_case_mm2": float(np.std(case, ddof=1)) if case.size >= 2 else float("nan"),
        "mean_control_mm2": float(np.mean(control)) if control.size else float("nan"),
        "sd_control_mm2": (
            float(np.std(control, ddof=1)) if control.size >= 2 else float("nan")
        ),
        "difference_mm2": (
            float(np.mean(case) - np.mean(control))
            if case.size and control.size
            else float("nan")
        ),
    }


def build_statistics_table(
    subject_metrics: pd.DataFrame,
    *,
    comparisons: Sequence[Comparison] = COMPARISONS,
) -> pd.DataFrame:
    """Compute all fixed comparisons for every network/variant stratum."""

    required = {
        "subject",
        "cohort",
        "variant",
        "method",
        "network_classification",
        "taxonomy",
        "network",
        "component_kind",
        "area_percent",
        "area_mm2",
    }
    missing = sorted(required.difference(subject_metrics.columns))
    if missing:
        raise ValueError(f"subject metrics missing required columns: {', '.join(missing)}")

    network_rows = subject_metrics.loc[
        subject_metrics["component_kind"].eq("network")
    ].copy()
    rows: list[dict[str, object]] = []
    for stratum_values, stratum in network_rows.groupby(list(_STRATA), sort=False):
        common = dict(zip(_STRATA, stratum_values, strict=True))
        for comparison in comparisons:
            case = stratum.loc[stratum["cohort"].eq(comparison.case_cohort)]
            control = stratum.loc[stratum["cohort"].eq(comparison.control_cohort)]
            summary = compute_welch_summary(case["area_percent"], control["area_percent"])
            rows.append(
                {
                    **common,
                    "comparison": comparison.key,
                    "case_cohort": comparison.case_cohort,
                    "control_cohort": comparison.control_cohort,
                    "patient_comparison": comparison.patient_comparison,
                    **summary,
                    **_secondary_area_summary(case["area_mm2"], control["area_mm2"]),
                }
            )

    statistics = pd.DataFrame(rows)
    if statistics.empty:
        return statistics
    return apply_grouped_bh_fdr(statistics)


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    count = p_values.size
    order = np.argsort(p_values, kind="stable")
    ranked = p_values[order]
    adjusted = ranked * count / np.arange(1, count + 1, dtype=np.float64)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    result = np.empty(count, dtype=np.float64)
    result[order] = adjusted
    return result


def apply_grouped_bh_fdr(statistics: pd.DataFrame) -> pd.DataFrame:
    """Apply BH within method x network-classification families.

    Variants, networks, and all seven comparisons remain in the same family,
    matching the planned auxiliary correction.  These q values do not drive
    the patient-result figure filter.
    """

    required = {"method", "network_classification", "p_value"}
    missing = sorted(required.difference(statistics.columns))
    if missing:
        raise ValueError(f"statistics missing required columns: {', '.join(missing)}")
    corrected = statistics.copy()
    corrected["q_value_bh"] = np.nan
    corrected["fdr_family_size"] = 0
    for _, indices in corrected.groupby(
        ["method", "network_classification"], sort=False, dropna=False
    ).groups.items():
        locations = list(indices)
        p_values = pd.to_numeric(corrected.loc[locations, "p_value"], errors="coerce")
        finite = np.isfinite(p_values.to_numpy(dtype=np.float64))
        valid_locations = list(np.asarray(locations, dtype=object)[finite])
        family_size = len(valid_locations)
        corrected.loc[locations, "fdr_family_size"] = family_size
        if valid_locations:
            corrected.loc[valid_locations, "q_value_bh"] = _bh_adjust(
                corrected.loc[valid_locations, "p_value"].to_numpy(dtype=np.float64)
            )
    return corrected


def write_analysis_outputs(
    output_dir: Path,
    *,
    subject_metrics: pd.DataFrame,
    statistics: pd.DataFrame,
    input_audit: pd.DataFrame,
    run_manifest: Mapping[str, object],
) -> dict[str, Path]:
    """Write the four required analysis artifacts and return their paths."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "subject_metrics": output_dir / "subject_network_area_metrics.csv",
        "statistics": output_dir / "network_area_statistics.csv",
        "input_audit": output_dir / "input_audit.csv",
        "run_manifest": output_dir / "run_manifest.json",
    }
    subject_metrics.to_csv(paths["subject_metrics"], index=False)
    statistics.to_csv(paths["statistics"], index=False)
    input_audit.to_csv(paths["input_audit"], index=False)
    paths["run_manifest"].write_text(
        json.dumps(dict(run_manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths


class InputCompatibilityError(RuntimeError):
    """Raised when a record audited as ready fails during metric extraction."""

    def __init__(self, audit: pd.DataFrame, failures: Sequence[Mapping[str, object]]) -> None:
        self.audit = audit
        self.failures = tuple(dict(item) for item in failures)
        count = len(self.failures)
        first = self.failures[0].get("error", "unknown error") if self.failures else "unknown error"
        super().__init__(f"{count} input record(s) became incompatible during metric extraction: {first}")


SUBJECT_METRIC_COLUMNS: tuple[str, ...] = (
    "subject",
    "cohort",
    "dataset",
    "variant",
    "method",
    "network_classification",
    "taxonomy",
    "network",
    "component_kind",
    "area_mm2",
    "area_percent",
    "total_cortical_area_mm2",
    "noise_area_mm2",
    "unassigned_area_mm2",
    "anatomical_mask_source",
    "label_source",
    "mapping_source",
    "surface_L",
    "surface_R",
    "functional_manifest",
)

STATISTICS_COLUMNS: tuple[str, ...] = (
    *_STRATA,
    "comparison",
    "case_cohort",
    "control_cohort",
    "patient_comparison",
    "n_case",
    "n_control",
    "mean_case",
    "sd_case",
    "mean_control",
    "sd_control",
    "difference_pp",
    "relative_difference_percent",
    "ci95_low_pp",
    "ci95_high_pp",
    "welch_t",
    "welch_df",
    "cohens_d",
    "p_value",
    "mean_case_mm2",
    "sd_case_mm2",
    "mean_control_mm2",
    "sd_control_mm2",
    "difference_mm2",
    "q_value_bh",
    "fdr_family_size",
)


def expand_variants(variant: str) -> tuple[str, ...]:
    """Expand the CLI variant selector without inventing resolution branches."""

    if variant in {"all", "36p"}:
        return ("36p",)
    raise ValueError("variant must be one of: 36p, all")


def run_analysis(
    *,
    repo_root: Path,
    output_dir: Path,
    variant: str,
    check_readiness: bool = False,
    layout: object | None = None,
    records: Sequence[object] | None = None,
    iowa_surface_manifest: Path | None = None,
) -> dict[str, Path]:
    """Audit inputs and, unless readiness-only, calculate and compare metrics.

    records is an explicit injection point used by controlled checks and
    site-specific launchers. Normal CLI execution derives records from the
    formal rosters and the declared input layout.
    """

    from .core import audit_subject_records

    selected_variants = expand_variants(variant)
    if records is None:
        records = build_subject_records(
            repo_root=Path(repo_root), variants=selected_variants, layout=layout
        )
    record_list = list(records)
    if iowa_surface_manifest is not None:
        from .prepare_iowa_surfaces import apply_manifest
        record_list = apply_manifest(record_list, iowa_surface_manifest)
    input_audit = pd.DataFrame(audit_subject_records(record_list))
    status_counts = (
        {str(key): int(value) for key, value in input_audit["status"].value_counts().items()}
        if "status" in input_audit
        else {}
    )

    def build_manifest(
        counts: Mapping[str, int], actual_n: Mapping[str, object]
    ) -> dict[str, object]:
        return {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "repo_root": str(Path(repo_root).resolve()),
            "variant_request": variant,
            "variants": list(selected_variants),
            "check_readiness": bool(check_readiness),
            "audit_status_counts": dict(counts),
            "actual_n": dict(actual_n),
            "comparisons": [
                {
                    "key": item.key,
                    "case_cohort": item.case_cohort,
                    "control_cohort": item.control_cohort,
                    "difference_direction": "case_minus_control",
                }
                for item in COMPARISONS
            ],
            "primary_test": "two-sided Welch t-test on cortical area percentage",
            "preprocessing_inputs": {
                "HCP HC": {
                    "source": "ICA-FIX",
                    "variant_policy": "36p",
                },
                "Iowa": {
                    "36p": "explicit 36P branch",
                },
            },
            "patient_figure_filter": "none; all networks, uncorrected significance symbols",
            "iowa_surface_manifest": str(iowa_surface_manifest) if iowa_surface_manifest else None,
            "fdr": (
                "Benjamini-Hochberg within method x network_classification, "
                "across variants, networks, and all seven comparisons; auxiliary only"
            ),
            "interpretation_notes": [
                "Iowa HC versus HCP HC is a cross-dataset baseline comparison, not an equivalence test.",
                "Patient overlaps Pure MDD and Comorbid MDD; subgroup results are not independent replications.",
            ],
        }

    def write_failure(audit: pd.DataFrame, failures: Sequence[Mapping[str, object]]) -> dict[str, Path]:
        counts = {
            str(key): int(value)
            for key, value in audit["status"].value_counts().items()
        }
        manifest = build_manifest(counts, {})
        manifest["analysis_status"] = "failed_input_compatibility"
        manifest["input_errors"] = [dict(item) for item in failures]
        return write_analysis_outputs(
            Path(output_dir),
            subject_metrics=pd.DataFrame(columns=SUBJECT_METRIC_COLUMNS),
            statistics=pd.DataFrame(columns=STATISTICS_COLUMNS),
            input_audit=audit,
            run_manifest=manifest,
        )

    initial_failures = input_audit.loc[
        input_audit["status"].ne("ready")
    ] if "status" in input_audit else pd.DataFrame()
    if not check_readiness and not initial_failures.empty:
        paths = write_failure(input_audit, initial_failures.to_dict(orient="records"))
        raise RuntimeError(
            f"Input audit found {len(initial_failures)} missing or incompatible record(s); "
            f"audit written to {paths['input_audit']}"
        )

    subject_metrics = pd.DataFrame(columns=SUBJECT_METRIC_COLUMNS)
    statistics = pd.DataFrame(columns=STATISTICS_COLUMNS)
    if not check_readiness:
        try:
            subject_metrics, input_audit = calculate_all_subject_metrics(
                repo_root=Path(repo_root), records=record_list, input_audit=input_audit
            )
        except InputCompatibilityError as exc:
            paths = write_failure(exc.audit, exc.failures)
            raise RuntimeError(
                f"{exc}; audit written to {paths['input_audit']}"
            ) from exc
        statistics = build_statistics_table(subject_metrics)
        if statistics.empty:
            statistics = pd.DataFrame(columns=STATISTICS_COLUMNS)
        else:
            statistics = statistics.reindex(columns=STATISTICS_COLUMNS)
        status_counts = {
            str(key): int(value)
            for key, value in input_audit["status"].value_counts().items()
        }

    manifest = build_manifest(status_counts, summarize_actual_n(subject_metrics))
    return write_analysis_outputs(
        Path(output_dir),
        subject_metrics=subject_metrics,
        statistics=statistics,
        input_audit=input_audit,
        run_manifest=manifest,
    )


def summarize_actual_n(subject_metrics: pd.DataFrame) -> dict[str, object]:
    """Return nested actual sample sizes from the valid-input intersection."""

    if subject_metrics.empty:
        return {}
    required = {"variant", "method", "taxonomy", "cohort", "subject"}
    missing = sorted(required.difference(subject_metrics.columns))
    if missing:
        raise ValueError(f"subject metrics missing actual-n columns: {', '.join(missing)}")
    counts = (
        subject_metrics.groupby(
            ["variant", "method", "taxonomy", "cohort"], sort=False
        )["subject"]
        .nunique()
        .reset_index(name="n")
    )
    result: dict[str, object] = {}
    for row in counts.itertuples(index=False):
        variant_bucket = result.setdefault(str(row.variant), {})
        method_bucket = variant_bucket.setdefault(str(row.method), {})
        taxonomy_bucket = method_bucket.setdefault(str(row.taxonomy), {})
        taxonomy_bucket[str(row.cohort)] = int(row.n)
    return result


def build_subject_records(*, repo_root: Path, variants: Sequence[str], layout: object | None = None) -> list[object]:
    """Expand the five formal rosters into explicit method/variant records."""

    from .core import load_formal_rosters, resolve_subject_record

    rosters = load_formal_rosters(Path(repo_root))
    records: list[object] = []
    for variant in variants:
        for cohort, roster in rosters.items():
            for atlas in ("kong2022", "lynch2024"):
                for subject in roster.subjects:
                    records.append(
                        resolve_subject_record(
                            subject,
                            cohort,
                            variant,
                            atlas,
                            layout,
                            repo_root=Path(repo_root),
                        )
                    )
    return records



def calculate_all_subject_metrics(
    *,
    repo_root: Path,
    records: Sequence[object],
    input_audit: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate ready records and fail closed on record-level incompatibility."""

    from .core import calculate_subject_record

    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    audit = input_audit.copy()
    if "status" not in audit.columns:
        audit["status"] = "missing"
    for record in records:
        selector = (
            (audit["subject"].eq(record.subject))
            & (audit["cohort"].eq(record.cohort))
            & (audit["variant"].eq(record.variant))
            & (audit["atlas"].eq(record.atlas))
        )
        matches = audit.index[selector].tolist()
        if not matches or audit.loc[matches[0], "status"] != "ready":
            continue
        try:
            metrics = calculate_subject_record(record, repo_root=Path(repo_root))
            rows.extend(_subject_metric_rows(record, metrics))
        except Exception as exc:
            error = str(exc)
            failure = {
                "subject": record.subject,
                "cohort": record.cohort,
                "variant": record.variant,
                "atlas": record.atlas,
                "error": error,
            }
            failures.append(failure)
            for index in matches:
                audit.loc[index, "status"] = "incompatible"
                audit.loc[index, "error"] = error
    if failures:
        raise InputCompatibilityError(audit, failures)
    result = pd.DataFrame(rows, columns=SUBJECT_METRIC_COLUMNS)
    return result, audit


def build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze cortical network surface area percentages")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "stats")
    parser.add_argument("--variant", choices=("36p", "all"), default="36p")
    parser.add_argument("--iowa-surface-manifest", type=Path, help="Explicit validated complete-roster surface manifest")
    parser.add_argument("--check-readiness", action="store_true", help="Only audit inputs and write readiness tables")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    paths = run_analysis(
        repo_root=args.repo_root,
        output_dir=args.output_dir,
        variant=args.variant,
        check_readiness=args.check_readiness,
        iowa_surface_manifest=args.iowa_surface_manifest,
    )
    for name, path in paths.items():
        print(f"{name}\t{path}")
    return 0



def flatten_subject_metrics(
    record: object,
    metrics_by_taxonomy: Mapping[str, object],
    taxonomies: Mapping[str, object],
) -> list[dict[str, object]]:
    """Flatten all network, Noise, and Unassigned components for one subject."""

    rows: list[dict[str, object]] = []
    for taxonomy_key, metrics in metrics_by_taxonomy.items():
        taxonomy = taxonomies.get(taxonomy_key)
        if taxonomy is None:
            raise ValueError(f"Missing taxonomy definition for {taxonomy_key}")
        network_order = tuple(getattr(taxonomy, "network_order", metrics.network_area_mm2.keys()))
        method = str(getattr(taxonomy, "method", "Kong2022" if taxonomy_key.startswith("kong_") else "Lynch2024"))
        classification = str(getattr(taxonomy, "classification", "canonical7"))
        mapping_source = str(getattr(taxonomy, "source", "config/cross_atlas_network_merge.json"))
        base = {
            "subject": record.subject,
            "cohort": record.cohort,
            "dataset": record.dataset,
            "variant": record.variant,
            "method": method,
            "network_classification": classification,
            "taxonomy": taxonomy_key,
            "total_cortical_area_mm2": float(metrics.total_cortical_area_mm2),
            "noise_area_mm2": float(metrics.noise_area_mm2),
            "unassigned_area_mm2": float(metrics.unassigned_area_mm2),
            "anatomical_mask_source": (
                "CIFTI BrainModel cortical vertices"
                if str(record.label_path).endswith(".dlabel.nii")
                else "explicit full fsLR hemisphere vertex numbering"
            ),
            "label_source": str(record.label_path),
            "mapping_source": mapping_source,
            "surface_L": str(record.surface_paths["L"]),
            "surface_R": str(record.surface_paths["R"]),
            "functional_manifest": str(record.functional_manifest),
        }
        for network in network_order:
            rows.append({
                **base,
                "network": network,
                "component_kind": "network",
                "area_mm2": float(metrics.network_area_mm2[network]),
                "area_percent": float(metrics.area_percent[network]),
            })
        total = float(metrics.total_cortical_area_mm2)
        for component, area in (("noise", metrics.noise_area_mm2), ("unassigned", metrics.unassigned_area_mm2)):
            rows.append({
                **base,
                "network": component.title(),
                "component_kind": component,
                "area_mm2": float(area),
                "area_percent": 100.0 * float(area) / total if total else float("nan"),
            })
    return rows


def _subject_metric_rows(record: object, metrics_by_taxonomy: Mapping[str, object]) -> list[dict[str, object]]:
    from .core import load_taxonomies
    repo_root = Path(__file__).resolve().parents[3]
    return flatten_subject_metrics(record, metrics_by_taxonomy, load_taxonomies(repo_root))


if __name__ == "__main__":
    raise SystemExit(main())

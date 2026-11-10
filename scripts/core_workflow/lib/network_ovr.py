"""Descriptive OVR annotation from subjectwise cluster-mean-timeseries ROI FC."""
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from . import kong2022_fc_contract as kong_contract
from .kong2022_label_utils import FAMILIES_17_ORDER, canonical_families_7_order
from .io_utils import write_tsv
from .workflow_utils import write_json

OVR_VARIANTS = ("ovr_logistic_canonical7", "ovr_logistic_family17")
NETWORK_TS_OVR_VARIANT = "ovr_logistic_canonical7_network_ts"
ALL_OVR_VARIANTS = (*OVR_VARIANTS, NETWORK_TS_OVR_VARIANT)
OVR_ASSIGNMENT_METRIC = "largest_positive_raw_beta"
INTERPRETATION = (
    "Descriptive conditional association; observations from the same subject are dependent. "
    "This is not out-of-sample accuracy, an inferential test, or a causal estimate. "
    "Future uncertainty estimation must resample subjects together. Raw odds ratios are per "
    "one-unit Fisher-z change with other network features held fixed; standardized odds ratios "
    "are per one SD over these exact subject-by-cluster observations. This changes reporting "
    "units and does not refit a standardized L2 model. Magnitudes are not directly comparable "
    "with Kember's different features, standardization domain, and model inputs."
)


def assignment_rows(beta, cluster_ids, network_names) -> list[dict]:
    values = np.asarray(beta, dtype=np.float64)
    if values.shape != (len(cluster_ids), len(network_names)) or not np.isfinite(values).all():
        raise ValueError("OVR coefficients must be finite and class/network aligned")
    rows = []
    for cid, coefficients in zip(cluster_ids, values, strict=True):
        winner = int(np.argmax(coefficients))
        assigned = coefficients[winner] > 0
        rows.append({
            "cluster_id": int(cid), "status": "dominant" if assigned else "unassigned",
            "top1_network": network_names[winner] if assigned else "unassigned",
            "top1_beta": float(coefficients[winner]),
            "unassigned_reason": None if assigned else "no_strictly_positive_raw_coefficient",
            "assignment_metric": OVR_ASSIGNMENT_METRIC,
            "beta_by_network": dict(zip(network_names, coefficients.tolist(), strict=True)),
        })
    return rows


def fit_network_ovr(roi_profiles, subject_ids, cluster_ids, roi_registry, *, taxonomy: str) -> dict:
    if taxonomy not in {"canonical7", "family17"}:
        raise ValueError(f"Unsupported OVR taxonomy: {taxonomy}")
    data = np.asarray(roi_profiles, dtype=np.float64)
    subjects, clusters = list(map(str, subject_ids)), list(map(int, cluster_ids))
    if data.shape != (len(subjects), len(clusters), 400):
        raise ValueError(f"OVR requires subjects x clusters x 400 ROI profiles, got {data.shape}")
    if not subjects or len(set(subjects)) != len(subjects):
        raise ValueError("OVR requires nonempty unique subject IDs")
    if len(clusters) < 2 or len(set(clusters)) != len(clusters) or min(clusters) < 1:
        raise ValueError("OVR requires at least two unique positive cluster IDs")
    kong_contract.validate_kong2022_roi_registry(roi_registry)
    expected = kong_contract.authoritative_kong2022_roi_registry()
    for index, (row, authoritative) in enumerate(zip(roi_registry, expected, strict=True)):
        for field in ("family17", "canonical7"):
            if row.get(field) != authoritative.get(field) or not row.get(field):
                raise ValueError(f"OVR ROI taxonomy mismatch: row={index} field={field}")
    names = list(canonical_families_7_order() if taxonomy == "canonical7" else FAMILIES_17_ORDER)
    network = np.full((*data.shape[:2], len(names)), np.nan, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for index, name in enumerate(names):
            columns = [i for i, row in enumerate(roi_registry) if row[taxonomy] == name]
            if columns:
                network[:, :, index] = np.nanmean(data[:, :, columns], axis=2)
    missing = np.argwhere(~np.isfinite(network))
    if missing.size:
        detail = "; ".join(f"subject={subjects[s]} cluster={clusters[c]} network={names[n]}"
                           for s, c, n in missing[:5])
        raise ValueError(f"Missing required OVR network feature: {detail}")
    x = network.reshape(-1, len(names))
    y = np.tile(clusters, len(subjects))
    model = OneVsRestClassifier(LogisticRegression(
        solver="liblinear", C=1., tol=1e-4, max_iter=1000, random_state=20260420,
    ), n_jobs=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            model.fit(x, y)
        except ConvergenceWarning as exc:
            raise RuntimeError(f"OVR {taxonomy} failed to converge") from exc
    classes = list(map(int, model.classes_))
    if set(classes) != set(clusters):
        raise RuntimeError(f"OVR estimator classes differ: {classes} vs {clusters}")
    if len(classes) == 2:
        coefficient = model.estimators_[0].coef_[0]
        intercept = float(model.estimators_[0].intercept_[0])
        beta_by_class = {classes[0]: -coefficient, classes[1]: coefficient}
        intercept_by_class = {classes[0]: -intercept, classes[1]: intercept}
    else:
        beta_by_class = {c: estimator.coef_[0] for c, estimator in zip(classes, model.estimators_, strict=True)}
        intercept_by_class = {c: float(estimator.intercept_[0]) for c, estimator in zip(classes, model.estimators_, strict=True)}
    beta = np.stack([beta_by_class[c] for c in clusters])
    sd = np.std(x, axis=0, ddof=0)
    standardized = beta * sd
    result = {
        "taxonomy": taxonomy, "variant": f"ovr_logistic_{taxonomy}",
        "subject_ids": subjects, "cluster_ids": clusters, "network_names": names,
        "estimator_classes": classes, "network_profiles": network,
        "beta": beta, "intercept": np.asarray([intercept_by_class[c] for c in clusters]),
        "odds_ratio": np.exp(beta), "feature_sd": sd,
        "standardized_beta": standardized, "standardized_odds_ratio": np.exp(standardized),
        "assignment_rows": assignment_rows(beta, clusters, names),
        "feature_sd_ddof": 0, "feature_sd_population": "exact_flattened_subject_by_cluster_fitting_observations",
        "n_observations": len(x), "interpretation": INTERPRETATION,
        "input_fc_definition": "subject_cluster_mean_hippocampal_timeseries_to_cortical_roi_fisher_z",
        "network_aggregation": "equal_ROI_nanmean_within_authoritative_network",
        "solver": "liblinear", "C": 1., "tol": 1e-4, "max_iter": 1000,
        "random_state": 20260420, "regularization": "default_L2",
        "predictor_scaling": "none", "class_weight": None,
        "n_iter": [int(np.max(estimator.n_iter_)) for estimator in model.estimators_],
    }
    return result


def fit_network_ovr_from_network_profiles(
    network_profiles,
    subject_ids,
    cluster_ids,
    network_names,
    *,
    taxonomy: str = "canonical7",
    feature_source: str = "network_ts",
    prefix: str | None = None,
) -> dict:
    """Fit descriptive OVR from direct subjectwise cluster-to-network Fisher-z features."""
    data = np.asarray(network_profiles, dtype=np.float64)
    subjects, clusters = list(map(str, subject_ids)), list(map(int, cluster_ids))
    names = list(map(str, network_names))
    if data.ndim != 3 or data.shape != (len(subjects), len(clusters), len(names)):
        raise ValueError(
            f"OVR network features require shape ({len(subjects)}, {len(clusters)}, {len(names)}), got {data.shape}"
        )
    if not subjects or len(set(subjects)) != len(subjects):
        raise ValueError("OVR requires nonempty unique subject IDs")
    if len(clusters) < 2 or len(set(clusters)) != len(clusters) or min(clusters) < 1:
        raise ValueError("OVR requires at least two unique positive cluster IDs")
    missing = np.argwhere(~np.isfinite(data))
    if missing.size:
        detail = "; ".join(
            f"subject={subjects[s]} cluster={clusters[c]} network={names[n]}"
            for s, c, n in missing[:5]
        )
        raise ValueError(f"Missing required OVR network feature: {detail}")
    x = data.reshape(-1, len(names))
    y = np.tile(clusters, len(subjects))
    model = OneVsRestClassifier(
        LogisticRegression(
            solver="liblinear", C=1.0, tol=1e-4, max_iter=1000, random_state=20260420
        ),
        n_jobs=1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            model.fit(x, y)
        except ConvergenceWarning as exc:
            raise RuntimeError(f"OVR {taxonomy} ({feature_source}) failed to converge") from exc
    classes = list(map(int, model.classes_))
    if set(classes) != set(clusters):
        raise RuntimeError(f"OVR estimator classes differ: {classes} vs {clusters}")
    if len(classes) == 2:
        coefficient = model.estimators_[0].coef_[0]
        intercept = float(model.estimators_[0].intercept_[0])
        beta_by_class = {classes[0]: -coefficient, classes[1]: coefficient}
        intercept_by_class = {classes[0]: -intercept, classes[1]: intercept}
    else:
        beta_by_class = {c: estimator.coef_[0] for c, estimator in zip(classes, model.estimators_, strict=True)}
        intercept_by_class = {c: float(estimator.intercept_[0]) for c, estimator in zip(classes, model.estimators_, strict=True)}
    beta = np.stack([beta_by_class[c] for c in clusters])
    sd = np.std(x, axis=0, ddof=0)
    standardized = beta * sd
    out_prefix = prefix or f"network_ovr_{taxonomy}_{feature_source}"
    result = {
        "taxonomy": taxonomy,
        "variant": f"ovr_logistic_{taxonomy}_{feature_source}",
        "feature_source": feature_source,
        "prefix": out_prefix,
        "subject_ids": subjects,
        "cluster_ids": clusters,
        "network_names": names,
        "estimator_classes": classes,
        "network_profiles": data,
        "beta": beta,
        "intercept": np.asarray([intercept_by_class[c] for c in clusters]),
        "odds_ratio": np.exp(beta),
        "feature_sd": sd,
        "standardized_beta": standardized,
        "standardized_odds_ratio": np.exp(standardized),
        "assignment_rows": assignment_rows(beta, clusters, names),
        "feature_sd_ddof": 0,
        "feature_sd_population": "exact_flattened_subject_by_cluster_fitting_observations",
        "n_observations": len(x),
        "interpretation": INTERPRETATION,
        "input_fc_definition": "subject_cluster_mean_hippocampal_timeseries_to_cortical_network_timeseries_fisher_z",
        "network_aggregation": "weighted_by_roi_n_grayordinates_used",
        "solver": "liblinear",
        "C": 1.0,
        "tol": 1e-4,
        "max_iter": 1000,
        "random_state": 20260420,
        "regularization": "default_L2",
        "predictor_scaling": "none",
        "class_weight": None,
        "n_iter": [int(np.max(estimator.n_iter_)) for estimator in model.estimators_],
    }
    return result


def plot_network_ovr(result: dict):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")
    entries = (
        ("beta", "Raw beta (per 1 Fisher-z unit)", 0.),
        ("standardized_beta", "Beta per 1 SD", 0.),
        ("odds_ratio", "OR per 1 Fisher-z unit", 1.),
        ("standardized_odds_ratio", "OR per 1 SD", 1.),
    )
    count = len(result["cluster_ids"])
    x = np.arange(len(result["network_names"]))
    width = .8 / count
    for ax, (key, unit, reference) in zip(axes.flat, entries, strict=True):
        ax.axhline(reference, color="black", linewidth=.8, linestyle="--")
        for index, cid in enumerate(result["cluster_ids"]):
            ax.bar(x + (index - (count - 1) / 2) * width, result[key][index], width,
                   label=f"Cluster {cid}")
        ax.set_xticks(x, result["network_names"], rotation=60, ha="right")
        ax.set_ylabel(unit)
        ax.set_title(f"{result['taxonomy']} | {unit}")
        ax.legend(fontsize=8)
    fig.suptitle("Subjectwise OVR conditional associations | per-SD values change reporting units only")
    return fig


def export_network_ovr(result: dict, out_dir: Path, *, prefix: str | None = None) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = prefix or result.get("prefix", f"network_ovr_{result['taxonomy']}")
    paths = {}
    arrays = ("network_profiles", "beta", "intercept", "odds_ratio", "feature_sd",
              "standardized_beta", "standardized_odds_ratio")
    for key in arrays:
        paths[key] = out_dir / f"{out_prefix}_{key}.npy"
        np.save(paths[key], result[key])
    paths["input_table"] = out_dir / f"{out_prefix}_inputs.tsv"
    rows = [{"subject_id": sid, "cluster_id": cid,
             **dict(zip(result["network_names"], result["network_profiles"][s, c].tolist(), strict=True))}
            for s, sid in enumerate(result["subject_ids"]) for c, cid in enumerate(result["cluster_ids"])]
    write_tsv(paths["input_table"], rows, ["subject_id", "cluster_id", *result["network_names"]])
    paths["coefficient_figure"] = out_dir / f"{out_prefix}_coefficients.png"
    fig = plot_network_ovr(result)
    fig.savefig(paths["coefficient_figure"], dpi=160)
    import matplotlib.pyplot as plt
    plt.close(fig)
    paths["metadata"] = out_dir / f"{out_prefix}.json"
    metadata = {key: value for key, value in result.items() if key not in arrays}
    metadata["artifacts"] = {key: str(path.resolve()) for key, path in paths.items()}
    write_json(paths["metadata"], metadata)
    return paths

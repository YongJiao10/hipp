#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import adjusted_rand_score


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_WORKFLOW_DIR = REPO_ROOT / "scripts" / "core_workflow"
if str(CORE_WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_WORKFLOW_DIR))

from lib.output_layout import (  # noqa: E402
    DEFAULT_KONG2022_TEMPLATE_ROOT,
    resolve_crossfit_template_matching_overview_root,
    resolve_crossfit_template_matching_root,
)
from lib.surface_utils import long_axis_order  # noqa: E402


DEFAULT_N_FOLDS = 5
DEFAULT_K = 3
DEFAULT_SMOOTHINGS = ("2mm", "4mm")
DEFAULT_HEMIS = ("L", "R")
DEFAULT_FIGURE_NAME = "hcp_routea_crossfit_k3_robustness_overview.png"
SUMMARY_FIELDS = (
    "smoothing",
    "hemi",
    "n_folds",
    "mean_pairwise_macro_dice",
    "min_pairwise_macro_dice",
    "mean_pairwise_ari",
    "min_pairwise_ari",
    "mean_pairwise_prototype_correlation",
    "min_pairwise_prototype_correlation",
    "mean_vertex_agreement",
    "min_vertex_agreement",
)


def _validate_fold_vertex_arrays(
    labels_by_fold: np.ndarray,
    support_by_fold: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels_by_fold, dtype=np.int32)
    support = np.asarray(support_by_fold, dtype=bool)
    if labels.ndim != 2:
        raise ValueError(f"labels_by_fold must be 2D, got shape={labels.shape}")
    if support.shape != labels.shape:
        raise ValueError(
            "labels/support shape mismatch: "
            f"labels={labels.shape} support={support.shape}"
        )
    if labels.shape[0] < 1:
        raise ValueError("labels_by_fold must contain at least one fold")
    return labels, support


def pairwise_label_metrics(
    labels_by_fold: np.ndarray,
    support_by_fold: np.ndarray,
    *,
    k: int = 3,
) -> dict[str, np.ndarray]:
    labels, support = _validate_fold_vertex_arrays(labels_by_fold, support_by_fold)
    if int(k) < 1:
        raise ValueError("k must be positive")
    invalid = labels[(labels < 0) | (labels > int(k))]
    if invalid.size:
        raise ValueError(f"labels must be in [0, {int(k)}]")

    n_folds = labels.shape[0]
    macro_dice = np.empty((n_folds, n_folds), dtype=np.float64)
    ari = np.empty((n_folds, n_folds), dtype=np.float64)
    for first in range(n_folds):
        for second in range(first, n_folds):
            comparison_support = support[first] & support[second]
            if not np.any(comparison_support):
                raise ValueError(
                    "Fold pair has empty comparison support: "
                    f"folds=({first}, {second})"
                )
            first_labels = labels[first, comparison_support]
            second_labels = labels[second, comparison_support]
            dice_by_label: list[float] = []
            for label in range(1, int(k) + 1):
                first_members = first_labels == label
                second_members = second_labels == label
                denominator = int(first_members.sum() + second_members.sum())
                dice_by_label.append(
                    1.0
                    if denominator == 0
                    else float(2 * np.sum(first_members & second_members) / denominator)
                )
            this_dice = float(np.mean(dice_by_label))
            this_ari = float(adjusted_rand_score(first_labels, second_labels))
            macro_dice[first, second] = macro_dice[second, first] = this_dice
            ari[first, second] = ari[second, first] = this_ari
    return {
        "macro_dice": macro_dice,
        "ari": ari,
        "adjusted_rand_index": ari,
    }


def _finite_pearson(first: np.ndarray, second: np.ndarray) -> float:
    finite = np.isfinite(first) & np.isfinite(second)
    if int(np.sum(finite)) < 2:
        return float("nan")
    first_finite = np.asarray(first[finite], dtype=np.float64)
    second_finite = np.asarray(second[finite], dtype=np.float64)
    if np.ptp(first_finite) == 0.0 or np.ptp(second_finite) == 0.0:
        return float("nan")
    return float(np.corrcoef(first_finite, second_finite)[0, 1])


def pairwise_prototype_correlations(
    profiles_by_fold: np.ndarray,
) -> np.ndarray:
    profiles = np.asarray(profiles_by_fold, dtype=np.float64)
    if profiles.ndim != 3:
        raise ValueError(
            "profiles_by_fold must have shape folds x prototypes x features, "
            f"got {profiles.shape}"
        )
    n_folds, n_prototypes, _n_features = profiles.shape
    if n_folds < 1 or n_prototypes < 1:
        raise ValueError("profiles_by_fold must contain folds and prototypes")
    correlations = np.empty((n_folds, n_folds, n_prototypes), dtype=np.float64)
    for first in range(n_folds):
        for second in range(first, n_folds):
            for prototype in range(n_prototypes):
                value = (
                    1.0
                    if first == second
                    else _finite_pearson(
                        profiles[first, prototype],
                        profiles[second, prototype],
                    )
                )
                correlations[first, second, prototype] = value
                correlations[second, first, prototype] = value
    return correlations


def vertex_agreement(labels_by_fold: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels_by_fold, dtype=np.int32)
    if labels.ndim != 2:
        raise ValueError(f"labels_by_fold must be 2D, got shape={labels.shape}")
    if labels.shape[0] < 1:
        raise ValueError("labels_by_fold must contain at least one fold")
    agreement = np.empty(labels.shape[1], dtype=np.float64)
    for vertex in range(labels.shape[1]):
        _values, counts = np.unique(labels[:, vertex], return_counts=True)
        agreement[vertex] = float(np.max(counts) / labels.shape[0])
    return agreement


def _canonical_surface_path(template_surfaces_dir: Path, hemi: str) -> Path:
    return (
        Path(template_surfaces_dir)
        / f"group-average_hemi-{hemi}_space-corobl_den-512_"
        "label-hipp_midthickness.surf.gii"
    )


def _fold_template_dir(
    crossfit_root: Path,
    fold_index: int,
    smoothing: str,
    hemi: str,
) -> Path:
    return (
        Path(crossfit_root)
        / "templates"
        / f"fold-{int(fold_index) + 1:02d}"
        / str(smoothing)
        / f"hemi_{hemi}"
    )


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _load_combination(
    *,
    crossfit_root: Path,
    template_surfaces_dir: Path,
    smoothing: str,
    hemi: str,
    n_folds: int,
) -> dict[str, Any]:
    canonical_surface = _canonical_surface_path(template_surfaces_dir, hemi).resolve()
    if not canonical_surface.exists():
        raise FileNotFoundError(f"Missing canonical template surface: {canonical_surface}")

    labels: list[np.ndarray] = []
    supports: list[np.ndarray] = []
    profiles: list[np.ndarray] = []
    source_folds: list[dict[str, Any]] = []
    for fold_index in range(int(n_folds)):
        template_dir = _fold_template_dir(
            crossfit_root,
            fold_index,
            smoothing,
            hemi,
        )
        source_paths = {
            "manifest": (template_dir / "template_manifest.json").resolve(),
            "labels": (template_dir / "group_labels_k3.npy").resolve(),
            "support_mask": (template_dir / "group_support_mask.npy").resolve(),
            "template_profiles": (template_dir / "template_profiles.npy").resolve(),
        }
        missing = [str(path) for path in source_paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing fold {fold_index + 1} robustness inputs: {missing}"
            )
        manifest = _load_json(source_paths["manifest"])
        expected = {
            "status": "ok",
            "fold_index": fold_index,
            "smoothing": str(smoothing),
            "hemi": str(hemi),
        }
        mismatches = [
            f"{key}: expected={value!r} actual={manifest.get(key)!r}"
            for key, value in expected.items()
            if manifest.get(key) != value
        ]
        density_evidence = manifest.get("density_evidence")
        consumed_surface = (
            density_evidence.get("consumed_asset")
            if isinstance(density_evidence, dict)
            else None
        )
        if consumed_surface is None:
            mismatches.append("density_evidence.consumed_asset: missing")
        elif Path(str(consumed_surface)).resolve() != canonical_surface:
            mismatches.append(
                "density_evidence.consumed_asset: "
                f"expected={str(canonical_surface)!r} actual={consumed_surface!r}"
            )
        if mismatches:
            raise ValueError(
                f"Fold {fold_index + 1} template manifest mismatch: "
                + "; ".join(mismatches)
            )

        fold_labels = np.load(source_paths["labels"]).astype(np.int32).reshape(-1)
        fold_support = np.load(source_paths["support_mask"]).astype(bool).reshape(-1)
        fold_profiles = np.load(source_paths["template_profiles"]).astype(np.float64)
        if fold_support.shape != fold_labels.shape:
            raise ValueError(
                f"Fold {fold_index + 1} label/support mismatch: "
                f"labels={fold_labels.shape} support={fold_support.shape}"
            )
        if fold_profiles.ndim != 2 or fold_profiles.shape[0] != DEFAULT_K:
            raise ValueError(
                f"Fold {fold_index + 1} template profiles must have "
                f"{DEFAULT_K} AP-ordered rows, got {fold_profiles.shape}"
            )
        labels.append(fold_labels)
        supports.append(fold_support)
        profiles.append(fold_profiles)
        source_folds.append(
            {
                "fold_index": fold_index,
                "fold_id": f"fold-{fold_index + 1:02d}",
                **{key: str(path) for key, path in source_paths.items()},
                "density_evidence": density_evidence,
            }
        )

    try:
        labels_by_fold = np.stack(labels, axis=0)
        support_by_fold = np.stack(supports, axis=0)
        profiles_by_fold = np.stack(profiles, axis=0)
    except ValueError as exc:
        raise ValueError(
            f"Fold input shapes disagree for {smoothing}/hemi_{hemi}"
        ) from exc
    order = np.asarray(long_axis_order(canonical_surface), dtype=np.int32).reshape(-1)
    if order.size != labels_by_fold.shape[1] or not np.array_equal(
        np.sort(order), np.arange(order.size)
    ):
        raise ValueError(
            f"Invalid long-axis order for {canonical_surface}: "
            f"order={order.size} vertices={labels_by_fold.shape[1]}"
        )
    return {
        "smoothing": str(smoothing),
        "hemi": str(hemi),
        "canonical_surface": str(canonical_surface),
        "labels_by_fold": labels_by_fold,
        "support_by_fold": support_by_fold,
        "profiles_by_fold": profiles_by_fold,
        "long_axis_order": order,
        "source_folds": source_folds,
    }


def _unique_pair_values(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"Expected square pairwise matrix, got {values.shape}")
    unique = values[np.triu_indices(values.shape[0], k=1)]
    finite = unique[np.isfinite(unique)]
    if finite.size == 0:
        raise ValueError("Pairwise metric has no finite unique-fold values")
    return finite


def _finite_summary(values: np.ndarray, *, name: str) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError(f"{name} has no finite values")
    return float(np.mean(finite)), float(np.min(finite))


def _analyse_combination(combination: dict[str, Any]) -> dict[str, Any]:
    label_metrics = pairwise_label_metrics(
        combination["labels_by_fold"],
        combination["support_by_fold"],
        k=DEFAULT_K,
    )
    prototype_correlations = pairwise_prototype_correlations(
        combination["profiles_by_fold"]
    )
    agreement = vertex_agreement(combination["labels_by_fold"])

    dice_values = _unique_pair_values(label_metrics["macro_dice"])
    ari_values = _unique_pair_values(label_metrics["adjusted_rand_index"])
    pair_indices = np.triu_indices(prototype_correlations.shape[0], k=1)
    prototype_values = prototype_correlations[pair_indices[0], pair_indices[1], :]
    mean_dice, min_dice = _finite_summary(dice_values, name="macro-Dice")
    mean_ari, min_ari = _finite_summary(ari_values, name="ARI")
    mean_prototype, min_prototype = _finite_summary(
        prototype_values,
        name="prototype correlation",
    )
    mean_agreement, min_agreement = _finite_summary(
        agreement,
        name="vertex agreement",
    )
    summary = {
        "smoothing": combination["smoothing"],
        "hemi": combination["hemi"],
        "n_folds": int(combination["labels_by_fold"].shape[0]),
        "mean_pairwise_macro_dice": mean_dice,
        "min_pairwise_macro_dice": min_dice,
        "mean_pairwise_ari": mean_ari,
        "min_pairwise_ari": min_ari,
        "mean_pairwise_prototype_correlation": mean_prototype,
        "min_pairwise_prototype_correlation": min_prototype,
        "mean_vertex_agreement": mean_agreement,
        "min_vertex_agreement": min_agreement,
    }
    return {
        **combination,
        "label_metrics": label_metrics,
        "prototype_correlations": prototype_correlations,
        "vertex_agreement": agreement,
        "summary": summary,
    }


def _draw_heatmap(
    axis: Any,
    values: np.ndarray,
    title: str,
    *,
    vmin: float = 0.0,
) -> None:
    image = axis.imshow(values, vmin=vmin, vmax=1.0, cmap="viridis")
    fold_labels = [str(index + 1) for index in range(values.shape[0])]
    axis.set_xticks(range(values.shape[0]), fold_labels, fontsize=7)
    axis.set_yticks(range(values.shape[0]), fold_labels, fontsize=7)
    axis.set_xlabel("Fold", fontsize=8)
    axis.set_ylabel("Fold", fontsize=8)
    axis.set_title(title, fontsize=10)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                f"{values[row, column]:.2f}",
                ha="center",
                va="center",
                color="white" if values[row, column] < 0.65 else "black",
                fontsize=6,
            )
    axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


def _render_figure(analyses: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    n_rows = len(analyses)
    figure, axes = plt.subplots(
        n_rows,
        5,
        figsize=(19, max(3.5, 3.6 * n_rows)),
        squeeze=False,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.5, 1.0, 1.0, 1.0, 1.7]},
    )
    label_cmap = ListedColormap(["#9e9e9e", "#dc504b", "#ebaf46", "#4673be"])
    label_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], label_cmap.N)
    for row_index, analysis in enumerate(analyses):
        row_axes = axes[row_index]
        order = analysis["long_axis_order"]
        ordered_labels = analysis["labels_by_fold"][:, order]
        row_axes[0].imshow(
            ordered_labels,
            aspect="auto",
            interpolation="nearest",
            cmap=label_cmap,
            norm=label_norm,
        )
        row_axes[0].set_yticks(
            range(ordered_labels.shape[0]),
            [f"Fold {index + 1}" for index in range(ordered_labels.shape[0])],
            fontsize=7,
        )
        row_axes[0].set_xticks([])
        row_axes[0].set_xlabel("Anterior → posterior", fontsize=8)
        row_axes[0].set_title(
            f"{analysis['smoothing']} hemi-{analysis['hemi']} labels",
            fontsize=10,
        )
        _draw_heatmap(row_axes[1], analysis["label_metrics"]["macro_dice"], "Macro-Dice")
        _draw_heatmap(
            row_axes[2],
            analysis["label_metrics"]["ari"],
            "ARI",
            vmin=-1.0,
        )
        mean_prototype_matrix = np.nanmean(
            analysis["prototype_correlations"],
            axis=2,
        )
        _draw_heatmap(
            row_axes[3],
            mean_prototype_matrix,
            "Prototype r",
            vmin=-1.0,
        )
        agreement = analysis["vertex_agreement"][order]
        row_axes[4].plot(np.arange(agreement.size), agreement, color="#304ffe", linewidth=1.2)
        row_axes[4].axhline(1.0, color="#777777", linewidth=0.7, linestyle="--")
        row_axes[4].set_ylim(0.0, 1.05)
        row_axes[4].set_xlim(0, max(1, agreement.size - 1))
        row_axes[4].set_xlabel("Anterior → posterior", fontsize=8)
        row_axes[4].set_ylabel("Modal fraction", fontsize=8)
        row_axes[4].set_title("Vertex agreement", fontsize=10)
        row_axes[4].tick_params(labelsize=7)
    figure.suptitle("HCP RouteA five-fold K=3 template robustness", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_summary_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(SUMMARY_FIELDS),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _json_matrix(values: np.ndarray) -> list[Any]:
    array = np.asarray(values, dtype=np.float64)
    return json.loads(json.dumps(array.tolist(), allow_nan=False))


def render_robustness_overview(
    *,
    crossfit_root: Path,
    template_surfaces_dir: Path,
    output_dir: Path,
    smoothings: Sequence[str] = DEFAULT_SMOOTHINGS,
    hemis: Sequence[str] = DEFAULT_HEMIS,
    n_folds: int = DEFAULT_N_FOLDS,
) -> dict[str, Path]:
    if int(n_folds) < 2:
        raise ValueError("n_folds must be at least 2")
    requested = [(str(smoothing), str(hemi)) for smoothing in smoothings for hemi in hemis]
    if not requested:
        raise ValueError("At least one smoothing/hemisphere combination is required")

    analyses = [
        _analyse_combination(
            _load_combination(
                crossfit_root=Path(crossfit_root),
                template_surfaces_dir=Path(template_surfaces_dir),
                smoothing=smoothing,
                hemi=hemi,
                n_folds=int(n_folds),
            )
        )
        for smoothing, hemi in requested
    ]
    output_root = Path(output_dir)
    png_path = output_root / DEFAULT_FIGURE_NAME
    summary_path = output_root / "robustness_summary.tsv"
    manifest_path = output_root / "robustness_manifest.json"
    _render_figure(analyses, png_path)
    _write_summary_tsv(summary_path, [analysis["summary"] for analysis in analyses])

    combinations: list[dict[str, Any]] = []
    for analysis in analyses:
        combinations.append(
            {
                "smoothing": analysis["smoothing"],
                "hemi": analysis["hemi"],
                "canonical_surface": analysis["canonical_surface"],
                "long_axis_order_source": analysis["canonical_surface"],
                "source_folds": analysis["source_folds"],
                "summary": analysis["summary"],
                "pairwise_macro_dice": _json_matrix(
                    analysis["label_metrics"]["macro_dice"]
                ),
                "pairwise_adjusted_rand_index": _json_matrix(
                    analysis["label_metrics"]["adjusted_rand_index"]
                ),
                "pairwise_prototype_correlations": _json_matrix(
                    analysis["prototype_correlations"]
                ),
                "vertex_agreement": _json_matrix(analysis["vertex_agreement"]),
            }
        )
    payload = {
        "status": "ok",
        "workflow": "hcp_routea_crossfit_k3_robustness_overview",
        "n_folds": int(n_folds),
        "crossfit_root": str(Path(crossfit_root).resolve()),
        "template_surfaces_dir": str(Path(template_surfaces_dir).resolve()),
        "combinations": combinations,
        "outputs": {
            "png": str(png_path.resolve()),
            "summary_tsv": str(summary_path.resolve()),
            "manifest": str(manifest_path.resolve()),
        },
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "png": png_path.resolve(),
        "summary_tsv": summary_path.resolve(),
        "manifest": manifest_path.resolve(),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render quantitative five-fold cross-fit template robustness."
    )
    parser.add_argument(
        "--crossfit-root",
        type=Path,
        default=resolve_crossfit_template_matching_root(),
    )
    parser.add_argument(
        "--template-surfaces-dir",
        type=Path,
        default=DEFAULT_KONG2022_TEMPLATE_ROOT / "group_average_surfaces",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=resolve_crossfit_template_matching_overview_root(),
    )
    parser.add_argument("--smoothings", nargs="+", default=list(DEFAULT_SMOOTHINGS))
    parser.add_argument("--hemis", nargs="+", choices=list(DEFAULT_HEMIS), default=list(DEFAULT_HEMIS))
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    outputs = render_robustness_overview(
        crossfit_root=args.crossfit_root,
        template_surfaces_dir=args.template_surfaces_dir,
        output_dir=args.output_dir,
        smoothings=args.smoothings,
        hemis=args.hemis,
        n_folds=int(args.n_folds),
    )
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

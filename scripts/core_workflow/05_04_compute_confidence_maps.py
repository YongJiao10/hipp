#!/usr/bin/env python3
from __future__ import annotations

"""
Compute vertex-level silhouette confidence maps for group-level gradient
parcellations in the G1/G2/G3 feature space used by clustering.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lib.gradient_clustering import GRADIENT_METHODS, is_gradient_method
from lib.output_layout import DEFAULT_ROUTEA_GROUP_ATLAS_ROOT
from lib.traceability import StageSpec, run_stage
from lib.workflow_utils import write_json
from lib.confidence_map import (
    compute_silhouette_map,
    silhouette_to_full_surface,
    silhouette_summary,
)


# Silhouette distance that matches each method's clustering geometry: Ward and GMM
# partition Euclidean feature space; the spectral methods use cosine affinity.
EUCLIDEAN_METHODS = frozenset({"gradient-ward", "gradient-gmm"})


def silhouette_metric_for_method(method: str) -> str:
    return "euclidean" if method in EUCLIDEAN_METHODS else "cosine"


@dataclass(frozen=True)
class ConfidenceSourceSpec:
    name: str
    label: str
    features_path: Path
    metric: str
    output_stem: str


def confidence_source_spec(
    *,
    run_dir: Path,
    metric: str,
) -> ConfidenceSourceSpec:
    return ConfidenceSourceSpec(
        name="gradient_feature",
        label="Gradient-feature silhouette",
        features_path=Path(run_dir) / "group_gradient_features_active.npy",
        metric=str(metric),
        output_stem="confidence_silhouette_gradient_feature",
    )


def compute_one_source(
    *,
    spec: ConfidenceSourceSpec,
    run_dir: Path,
    labels_active: np.ndarray,
    support_mask: np.ndarray,
) -> dict[str, object]:
    if not spec.features_path.exists():
        raise FileNotFoundError(
            f"Missing {spec.name} features file: {spec.features_path}"
        )

    features = np.load(spec.features_path).astype(np.float32)
    n_support = int(np.sum(support_mask))
    if len(features) != n_support:
        raise ValueError(
            f"{spec.name} feature shape {features.shape} does not match "
            f"active mask vertices {n_support}."
        )

    silhouette_active = compute_silhouette_map(
        features=features,
        labels=labels_active,
        metric=spec.metric,
    )

    silhouette_full = silhouette_to_full_surface(
        silhouette_active=silhouette_active,
        support_mask=support_mask,
        fill_value=np.nan,
    )
    stats = silhouette_summary(
        silhouette_active=silhouette_active,
        labels_active=labels_active,
    )

    out_npy = run_dir / f"{spec.output_stem}.npy"
    out_json = run_dir / f"{spec.output_stem}_summary.json"
    out_manifest = run_dir / f"{spec.output_stem}_manifest.json"
    np.save(out_npy, silhouette_full)
    write_json(out_json, stats)

    manifest = {
        "confidence_source": spec.name,
        "confidence_label": spec.label,
        "metric": spec.metric,
        "features_file": str(spec.features_path),
        "output_file": str(out_npy.name),
        "summary_file": str(out_json.name),
        "global_mean_silhouette": stats["global_mean"],
        "global_min_silhouette": stats["global_min"],
        "global_max_silhouette": stats["global_max"],
        "global_frac_negative": stats["global_frac_negative"],
        "n_active_vertices": int(n_support),
    }
    write_json(out_manifest, manifest)

    print(f"  {spec.name}: mean={stats['global_mean']:.4f} min={stats['global_min']:.4f} max={stats['global_max']:.4f}")
    return manifest


def main(args: argparse.Namespace) -> int:
    out_root = Path(args.out_root).resolve()
    comp_str = args.gradient_components.strip()
    comp_dir_name = "g" + comp_str.replace(",", "g")
    processed_count = 0
    failed: list[dict[str, str]] = []

    def record_failure(combo: str, exc: Exception) -> None:
        print(f"Failed confidence combo {combo}: {type(exc).__name__}: {exc}")
        failed.append(
            {"combo": combo, "error_type": type(exc).__name__, "error": str(exc)}
        )

    for method in args.methods:
        if not is_gradient_method(method):
            raise ValueError(f"Confidence maps require a declared gradient method: {method}")

        for smoothing in args.smoothings:
            for thresh in args.coverage_thresholds:
                thresh_str = f"cov_{thresh:.2f}".replace(".", "p")
                for hemi in args.hemis:
                    hemi_upper = hemi.upper()
                    combo = f"{method}|{smoothing}|{thresh_str}|{comp_dir_name}|hemi_{hemi_upper}"
                    run_dir = (
                        out_root
                        / method
                        / "kong2022"
                        / smoothing
                        / thresh_str
                        / comp_dir_name
                        / f"hemi_{hemi_upper}"
                    )
                    if not run_dir.exists():
                        record_failure(combo, FileNotFoundError(f"Missing run dir: {run_dir}"))
                        continue

                    labels_path = run_dir / "group_labels_final.npy"
                    mask_path = run_dir / "group_support_mask.npy"
                    if not labels_path.exists():
                        record_failure(combo, FileNotFoundError(f"Missing final labels file: {labels_path}"))
                        continue
                    if not mask_path.exists():
                        record_failure(combo, FileNotFoundError(f"Missing support mask file: {mask_path}"))
                        continue

                    print(f"Processing combo: {method} | {smoothing} | {thresh_str} | {comp_dir_name} | hemi_{hemi_upper}")
                    try:
                        labels = np.load(labels_path).astype(np.int32)
                        support_mask = np.load(mask_path).astype(bool)
                        labels_active = labels[support_mask]

                        spec = confidence_source_spec(
                            run_dir=run_dir,
                            metric=args.metric or silhouette_metric_for_method(method),
                        )
                        compute_one_source(
                            spec=spec,
                            run_dir=run_dir,
                            labels_active=labels_active,
                            support_mask=support_mask,
                        )
                    except Exception as exc:
                        import traceback
                        traceback.print_exc()
                        record_failure(combo, exc)
                        continue
                    processed_count += 1

    print(f"Completed processing {processed_count} confidence maps.")
    if failed:
        print(f"Completed with {len(failed)} failed combo(s); refusing silent partial output:")
        for item in failed:
            print(f"  {item['combo']} {item['error_type']}: {item['error']}")
        return 1
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute vertex-level silhouette confidence maps.")
    parser.add_argument("--out-root", default=str(DEFAULT_ROUTEA_GROUP_ATLAS_ROOT))
    parser.add_argument("--methods", nargs="+", choices=sorted(GRADIENT_METHODS), default=["gradient-spectral"])
    parser.add_argument("--smoothings", nargs="+", default=["2mm", "4mm"])
    parser.add_argument("--hemis", nargs="+", default=["L", "R"])
    parser.add_argument("--coverage-thresholds", nargs="+", type=float, default=[0.80])
    parser.add_argument(
        "--gradient-components",
        default="1,2,3",
        help="Comma-separated gradient components to process (e.g. 1,2,3).",
    )
    parser.add_argument(
        "--metric",
        default=None,
        help="Silhouette distance metric; defaults to euclidean for gradient-ward and gradient-gmm, cosine otherwise.",
    )
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    raise SystemExit(
        run_stage(
            StageSpec(
                step_id="05_04_compute_confidence_maps",
                purpose="Compute vertex-level silhouette confidence maps for group-level functional parcellations.",
                stage_main=lambda: main(parser.parse_args()),
                upstream_defaults=("05_02_run_group_atlas",),
            )
        )
    )

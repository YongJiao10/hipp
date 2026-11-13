#!/usr/bin/env python3
from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = (REPO_ROOT / "scripts" / "core_workflow").resolve()
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from lib.confidence_map import publication_display_data, publication_display_upper


def _load_numbered_module(script_name: str, alias: str):
    module_path = CORE_DIR / script_name
    spec = importlib.util.spec_from_file_location(alias, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


compute_mod = _load_numbered_module("05_04_compute_confidence_maps.py", "core_05_04_compute_confidence_maps")


def test_publication_display_clips_negatives_and_uses_positive_auto_upper() -> None:
    values = np.array([-0.7, -0.1, 0.0, 0.13, 0.52, 0.73, np.nan], dtype=np.float32)

    display = publication_display_data(values, vmax=None)

    assert np.isnan(display[-1])
    assert display[:2].tolist() == [0.0, 0.0]
    assert display[3] == values[3]
    assert np.isclose(publication_display_upper(values, vmax=None), 0.73)


def test_confidence_source_returns_gradient_feature() -> None:
    spec = compute_mod.confidence_source_spec(
        run_dir=Path("/work/group/gradient-spectral/kong2022/2mm/cov_0p80/g1g2g3/hemi_L"),
        metric="cosine",
    )

    assert spec.name == "gradient_feature"
    assert spec.features_path.name == "group_gradient_features_active.npy"
    assert spec.metric == "cosine"


def test_silhouette_metric_follows_the_clustering_geometry() -> None:
    assert compute_mod.silhouette_metric_for_method("gradient-ward") == "euclidean"
    assert compute_mod.silhouette_metric_for_method("gradient-gmm") == "euclidean"
    assert compute_mod.silhouette_metric_for_method("gradient-spectral") == "cosine"
    assert compute_mod.build_arg_parser().parse_args([]).metric is None


def main() -> int:
    test_publication_display_clips_negatives_and_uses_positive_auto_upper()
    test_confidence_source_returns_gradient_feature()
    test_silhouette_metric_follows_the_clustering_geometry()
    print("ok: confidence map checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

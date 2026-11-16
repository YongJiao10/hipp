"""Stable-output contracts for the rebuilt Stanford workflow."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pandas as pd


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = WORKFLOW_ROOT / "config" / "analysis_config.json"


def test_stable_contrast_entrypoint_exists() -> None:
    module = importlib.import_module(
        "stanford_jjtms_multiscale_hippocampal_exploration.kember_metrics.src.contrasts"
    )
    assert callable(module.run_stable_contrasts)


def test_stable_outputs_close_over_feature_manifest() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    output_dirs = {
        key: Path(value)
        for key, value in config["outputs"].items()
    }
    for path in output_dirs.values():
        assert path.is_dir(), path

    feature_manifest_path = output_dirs["features"] / "feature_manifest.json"
    result_manifest_path = output_dirs["direct_contrasts"] / "run_manifest.json"
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    result_manifest = json.loads(result_manifest_path.read_text(encoding="utf-8"))
    feature_table_path = Path(feature_manifest["outputs"]["subject_features"])
    assert feature_table_path.is_file()
    assert Path(result_manifest["feature_table"]).resolve() == feature_table_path.resolve()

    feature_table = pd.read_csv(feature_table_path, sep="\t")
    effects_path = Path(result_manifest["outputs"]["direct_contrasts"])
    effects = pd.read_csv(effects_path, sep="\t")
    assert {"method", "smoothing_mm", "hemisphere", "system", "metric"} <= set(effects)
    assert set(effects["method"].dropna()) <= {
        "hcp_template_matching",
        "hcp_fixed",
    }
    successful = feature_table.loc[feature_table["status"].eq("success")]
    if not successful.empty:
        assert successful["vertex_area_source"].eq(
            "hippunfold_native_surfareacorobl"
        ).all()
        assert successful["vertex_area_metric_path"].astype(str).ne("").all()
        assert successful["input_hashes"].astype(str).ne("").all()

    assert not any(
        token in path.name
        for root in output_dirs.values()
        for path in root.rglob("*")
        if path.is_file()
        for token in ("_v1", "_v2", "_attempt")
    )

"""End-to-end writer contracts for the combined Kember feature manifest."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


def test_feature_builder_source_never_reconstructs_kember_vertex_area() -> None:
    source = (SRC / "feature_build.py").read_text(encoding="utf-8")

    assert "compute_vertex_area" not in source


def test_run_feature_build_writes_one_two_method_table_and_native_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("feature_build")
    rows = pd.DataFrame(
        [
            {
                **{column: "" for column in module.KEMBER_FEATURE_COLUMNS},
                "subject": "sub-1001",
                "method": method,
                "status": "success",
            }
            for method in module.METHODS
        ]
    ).reindex(columns=module.KEMBER_FEATURE_COLUMNS)
    statuses = pd.DataFrame(
        [
            {
                **{column: "" for column in module.KEMBER_STATUS_COLUMNS},
                "subject": "sub-1001",
                "method": method,
                "feature_status": "success",
            }
            for method in module.METHODS
        ]
    ).reindex(columns=module.KEMBER_STATUS_COLUMNS)
    bundle = {
        "native_area_metric": {"path": "native.shape.gii", "sha256": "a" * 64},
        "consumed_native_surface": {"consumed_asset": "native.surf.gii"},
        "native_labels": {"path": "native.label.gii", "sha256": "b" * 64},
        "label_transfer": {"method": "ADAP_BARY_AREA"},
        "density_evidence": {"n_vertices": 4},
        "source_functional_inputs": {
            "timeseries_path": "source.npy",
            "valid_mask": {"path": "valid.npy", "sha256": "c" * 64},
            "source_vertex_count": 512,
        },
        "input_hashes": {"native_area_metric": "a" * 64},
    }
    monkeypatch.setattr(
        module,
        "_build_feature_outputs",
        lambda *args, **kwargs: (rows, statuses, [bundle]),
    )
    output_dir = tmp_path / "features"

    result = module.run_feature_build(
        {"methods": list(module.METHODS), "smoothings_mm": [2], "hemispheres": ["L"]},
        output_dir,
    )

    table = pd.read_csv(result / "subject_features.tsv", sep="\t", keep_default_na=False)
    manifest = json.loads((result / "feature_manifest.json").read_text(encoding="utf-8"))
    assert set(table["method"]) == set(module.METHODS)
    assert manifest["methods"] == list(module.METHODS)
    assert manifest["area_source"] == "hippunfold_native_surfareacorobl"
    assert manifest["source_functional_density"] == 512
    assert manifest["native_provenance"] == [bundle]
    assert manifest["native_provenance"][0]["source_functional_inputs"][
        "source_vertex_count"
    ] == 512

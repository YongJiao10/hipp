"""Readiness checks must fail closed on missing comparison products."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LIB_DIR = REPO_ROOT / "scripts" / "visualize" / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

import json

import multidataset_cohorts as cohorts_mod
from multidataset_cohorts import (
    _comparison_lynch_input_errors,
    _comparison_overview_input_errors,
)


def test_comparison_overview_readiness_reports_each_missing_declared_asset(tmp_path):
    missing_root = Path("/workspace/missing_comparison_overviews")
    entry = {
        "overview_grad": missing_root / "gradient.png",
        "overview_grad_normalized_top20_count": missing_root / "gradient_normalized.png",
        "overview_wta": missing_root / "wta.png",
    }

    assert _comparison_overview_input_errors(entry, method="kong2022_7net") == [
        f"overview_grad:{entry['overview_grad']}",
        f"overview_grad_normalized_top20_count:{entry['overview_grad_normalized_top20_count']}",
        f"overview_wta:{entry['overview_wta']}",
    ]


def test_comparison_overview_readiness_rejects_ephemeral_asset(tmp_path):
    entry = {
        "overview_grad": Path("/workspace/.tmp_render/gradient.png"),
        "overview_wta": Path(__file__).resolve(),
    }

    errors = _comparison_overview_input_errors(entry, method="lynch2024_7net")

    assert errors == [f"overview_grad_temporary:{entry['overview_grad']}"]


def test_lynch_pfm_readiness_rejects_temporary_dtseries(tmp_path, monkeypatch):
    fake_root = tmp_path / "repo"
    roster = fake_root / "manifests/jj_tms_hc_67_subjects.txt"
    roster.parent.mkdir(parents=True)
    roster.write_text("1001\n", encoding="utf-8")
    manifest_dir = (
        fake_root
        / "outputs_iowa_jj_tms/work/cortex_pfm/lynch2024_iowa3t"
        / "sub-1001/lynch2024"
    )
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "fastans_pfm_manifest.json").write_text(
        json.dumps(
            {
                "status": "done",
                "inputs": {
                    "dtseries": "/workspace/.tmp_lynch/merged.dtseries.nii",
                    "left_surface": "/workspace/L.surf.gii",
                    "right_surface": "/workspace/R.surf.gii",
                    "priors": "/workspace/priors.pickle",
                    "labels": "/workspace/labels.txt",
                    "cortex_template": "/workspace/CORTEX.dscalar.nii",
                },
                "outputs": {
                    "component_timeseries": "/workspace/components.npy",
                    "component_tsv": "/workspace/components.tsv",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cohorts_mod, "REPO_ROOT", fake_root)

    errors = _comparison_lynch_input_errors("iowa_hc")

    assert any("pfm_temporary_dtseries_subjects:1001" in error for error in errors)


def test_lynch_pfm_readiness_accepts_complete_in_memory_run_inputs(
    tmp_path, monkeypatch
):
    fake_root = tmp_path / "repo"
    roster = fake_root / "manifests/jj_tms_hc_67_subjects.txt"
    roster.parent.mkdir(parents=True)
    roster.write_text("1001\n", encoding="utf-8")
    manifest_dir = (
        fake_root
        / "outputs_iowa_jj_tms/work/cortex_pfm/lynch2024_iowa3t"
        / "sub-1001/lynch2024"
    )
    manifest_dir.mkdir(parents=True)
    input_paths = {
        key: tmp_path / f"{key}.asset"
        for key in (
            "left_surface",
            "right_surface",
            "priors",
            "labels",
            "cortex_template",
        )
    }
    run_paths = [tmp_path / "run01.dtseries.nii", tmp_path / "run03.dtseries.nii"]
    output_paths = {
        "component_timeseries": tmp_path / "components.npy",
        "component_tsv": tmp_path / "components.tsv",
    }
    for path in [*input_paths.values(), *run_paths, *output_paths.values()]:
        path.touch()
    (manifest_dir / "fastans_pfm_manifest.json").write_text(
        json.dumps(
            {
                "status": "done",
                "inputs": {
                    **{key: str(path) for key, path in input_paths.items()},
                    "dtseries": None,
                    "dtseries_inputs": [str(path) for path in run_paths],
                    "dtseries_input_mode": "in_memory_run_concat",
                },
                "outputs": {
                    key: str(path) for key, path in output_paths.items()
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cohorts_mod, "REPO_ROOT", fake_root)
    monkeypatch.setattr(cohorts_mod, "_is_ephemeral_provenance_path", lambda value: False)

    assert _comparison_lynch_input_errors("iowa_hc") == []

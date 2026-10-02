from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import json
import numpy as np
import pytest


CORE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "core_workflow"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))
MODULE_PATH = CORE_DIR / "03_03_build_lynch2024_reference.py"
spec = importlib.util.spec_from_file_location(
    "lynch_reference_run_metadata", MODULE_PATH
)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_run_spec_preserves_recorded_ids_and_lengths(tmp_path: Path) -> None:
    metadata = tmp_path / "subject_manifest.json"
    metadata.write_text(
        '{"header_evidence": {"runs": [{"run_id": 1, "run_length": 229}, '
        '{"run_id": 2, "run_length": 214}, {"run_id": 3, "run_length": 212}], '
        '"run_lengths": [229, 214, 212], "concat_timepoints": 655}}'
    )
    assert module.load_run_spec(metadata) == ((1, 2, 3), (229, 214, 212))


def test_common_run_manifest_records_metadata_boundary_source(tmp_path: Path) -> None:
    segments_root = tmp_path / "shared"
    module._write_common_run_manifests(
        subject="3002",
        shared_root=segments_root,
        n_timepoints=655,
        run_lengths=[229, 214, 212],
        run_ids=(1, 2, 3),
        run_boundary_source="explicit_run_metadata_template",
    )
    summary = (
        segments_root
        / "sub-3002"
        / "reference"
        / "common"
        / "runs"
        / "run-1"
        / "reference_summary.json"
    ).read_text()
    assert '"run_boundary_source": "explicit_run_metadata_template"' in summary


def test_run_spec_rejects_missing_explicit_lengths(tmp_path: Path) -> None:
    metadata = tmp_path / "subject_manifest.json"
    metadata.write_text('{"header_evidence": {"runs": [{"run_id": 1}]}}')
    try:
        module.load_run_spec(metadata)
    except ValueError as exc:
        assert "explicit" in str(exc) or "length" in str(exc)
    else:
        raise AssertionError("metadata without explicit run lengths must fail")


@pytest.mark.parametrize(
    "payload",
    [
        {"run_lengths": [3, 4]},
        {
            "run_ids": [1, 3],
            "run_lengths": [3, 4],
            "runs": [{"run_id": 1, "run_length": 3}, {"run_id": 2, "run_length": 4}],
        },
        {"run_ids": [1, 2], "run_lengths": [3.5, 4]},
    ],
)
def test_run_spec_rejects_inferred_or_contradictory_values(tmp_path, payload):
    path = tmp_path / "runs.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        module.load_run_spec(path)


@pytest.mark.parametrize("lengths", [[2, 3], [2, 4]])
def test_invalid_metadata_does_not_write_reference_products(
    tmp_path, monkeypatch, lengths
):
    np.save(tmp_path / "components.npy", np.ones((2, 5), dtype=np.float32))
    metadata = tmp_path / "runs.json"
    payload = {"run_lengths": lengths}
    if sum(lengths) != 5:
        payload["run_ids"] = [1, 3]
    metadata.write_text(json.dumps(payload))

    def unexpected_write(**kwargs):
        pytest.fail("reference writer was called before validating run metadata")

    monkeypatch.setattr(module, "build_lynch_network_references", unexpected_write)
    with pytest.raises(ValueError):
        module.process_subject(
            subject="3002",
            shared_root=tmp_path / "shared",
            cortex_pfm_root=tmp_path,
            component_timeseries_template=str(tmp_path / "components.npy"),
            component_tsv_template=str(tmp_path / "components.tsv"),
            merge_config=tmp_path / "merge.json",
            run_lengths=None,
            run_metadata_template=str(metadata),
        )
    assert not (tmp_path / "shared").exists()


def test_parse_run_segments_rejects_missing_explicit_boundaries() -> None:
    with pytest.raises(ValueError, match="requires explicit"):
        module.parse_run_segments(None, 12, run_ids=[1, 2])


def test_process_subject_requires_selected_run_metadata_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np.save(tmp_path / "components.npy", np.ones((2, 5), dtype=np.float32))

    def unexpected_write(**kwargs: object) -> None:
        pytest.fail("reference writer was called without selected run metadata")

    monkeypatch.setattr(module, "build_lynch_network_references", unexpected_write)
    with pytest.raises(ValueError, match="run metadata"):
        module.process_subject(
            subject="3002",
            shared_root=tmp_path / "shared",
            cortex_pfm_root=tmp_path,
            component_timeseries_template=str(tmp_path / "components.npy"),
            component_tsv_template=str(tmp_path / "components.tsv"),
            merge_config=tmp_path / "merge.json",
            run_lengths=None,
            run_metadata_template=None,
        )
    assert not (tmp_path / "shared").exists()

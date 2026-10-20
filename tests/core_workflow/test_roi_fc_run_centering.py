"""Stage 04_01 pools physical runs without interpreting shared offsets as FC."""
from __future__ import annotations

import csv
import importlib
import json

import numpy as np
import pytest


materializer = importlib.import_module("scripts.core_workflow.04_01_materialize_kong2022_roi_fc")


# Each physical run has zero-mean signal; the second retains larger hippocampal
# variance. A common baseline step reverses the uncentered pooled correlation.
HIPP_SIGNAL = np.array([-3, -1, 1, 3, -6, -2, 2, 6], dtype=np.float32)
CORTEX_SIGNAL = np.array([3, 1, -2, -2, 1, 3, -1, -3], dtype=np.float32)
STEP = np.array([0, 0, 0, 0, 100, 100, 100, 100], dtype=np.float32)
POOLED_FC = -50 / np.sqrt(3800)
RUN_FC = (-18 / np.sqrt(360), -0.8)


def _write_kong400_inputs(root, cortex_reference_factory):
    surface_root = root / "surfaces"
    reference_root = root / "references"
    output_root = root / "fc_outputs"
    surface_dir = surface_root / "sub-01/surface/2mm"
    surface_dir.mkdir(parents=True)
    hipp = np.repeat((HIPP_SIGNAL + STEP)[None, :], 4, axis=0)
    hipp[2, 2] = np.nan
    hipp[3] = STEP
    np.save(surface_dir / "sub-01_hemi-L_timeseries.npy", hipp)
    gate_dir = surface_root / "sub-01/surface/tsnr_gate"
    gate_dir.mkdir()
    np.save(gate_dir / "sub-01_hemi-L_valid_mask.npy", [True, False, True, True])

    ref_dir = reference_root / "sub-01/reference/kong2022"
    ref_dir.mkdir(parents=True)
    registry = materializer.authoritative_kong2022_roi_registry()
    assert len(registry) == 400
    with (ref_dir / "cortex_roi_parcels.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["parcel_id", "parcel_name", "family17", "canonical8", "canonical7"])
        for row in registry:
            writer.writerow([row["roi_id"], row["roi_name"], row["family17"], row["canonical8"], row["canonical7"]])
    cortex = np.repeat((CORTEX_SIGNAL + STEP)[None, :], 400, axis=0)
    cortex[399] = np.nan
    np.save(ref_dir / "cortex_roi_parcel_timeseries.npy", cortex)
    cortex_reference_factory(reference_root, dataset="hcp", run_ids=(1, 2), run_lengths=(4, 4))
    summary = ref_dir / "reference_summary.json"
    gates = [ref_dir.parent / "common/tsnr_gate" / f"cortex_{side}_tsnr_gate_summary.json"
             for side in ("left", "right")]
    return surface_root, reference_root, output_root, summary, gates


def _materialize(surface_root, reference_root, output_root):
    materializer.materialize_fc(
        shared_root=output_root, surface_root=surface_root, reference_root=reference_root,
        subject="01", smoothing="2mm", hemi="L", dataset="hcp",
    )
    return output_root / "sub-01/fc/kong2022/2mm/hemi_L"


def test_kong400_materialization_pools_centered_runs_and_retains_missing_signal(tmp_path, cortex_reference_factory):
    surface_root, reference_root, output_root, _, _ = _write_kong400_inputs(tmp_path, cortex_reference_factory)
    assert np.corrcoef(HIPP_SIGNAL + STEP, CORTEX_SIGNAL + STEP)[0, 1] > 0.99
    fc_dir = _materialize(surface_root, reference_root, output_root)
    pooled = np.load(fc_dir / "hipp_vertex_to_network_fc.npy")
    assert pooled.shape == (4, 400)
    np.testing.assert_allclose(pooled[0, :399], POOLED_FC, atol=2e-7)
    assert np.isnan(pooled[1:]).all(), "Masked, incomplete and offset-only rows stay undefined"
    assert np.isnan(pooled[:, 399]).all(), "An empty ROI keeps its fixed Kong400 column"
    np.testing.assert_allclose(np.load(fc_dir / "hipp_vertex_to_roi_fc.npy"), pooled, equal_nan=True)
    active = np.load(fc_dir / "hipp_vertex_to_network_fc_active.npy")
    assert active.shape == (3, 400)
    np.testing.assert_allclose(active, pooled[[0, 2, 3]], equal_nan=True)

    canonical7 = np.load(fc_dir / "hipp_vertex_to_canonical7_fc.npy")
    assert canonical7.shape == (4, 7)
    np.testing.assert_allclose(canonical7[0], POOLED_FC, atol=2e-7)
    assert np.isnan(canonical7[1:]).all()
    for run_id, expected in enumerate(RUN_FC, start=1):
        run_dir = fc_dir / "runs" / f"run-{run_id}"
        run_roi = np.load(run_dir / "hipp_vertex_to_network_fc_active.npy")
        np.testing.assert_allclose(run_roi[0, :399], expected, atol=2e-7)
        assert np.isnan(run_roi[:, 399]).all()
        assert np.isnan(run_roi[2]).all(), "A constant physical-run row remains undefined"
        np.testing.assert_allclose(
            np.load(run_dir / "hipp_vertex_to_canonical7_fc_active.npy")[0], expected, atol=2e-7
        )


@pytest.mark.parametrize("policy_source", ["reference", "left_gate", "right_gate"])
def test_kong400_reference_policy_mismatch_fails_before_output(tmp_path, policy_source, cortex_reference_factory):
    surface_root, reference_root, output_root, summary, gates = _write_kong400_inputs(tmp_path, cortex_reference_factory)
    path = {"reference": summary, "left_gate": gates[0], "right_gate": gates[1]}[policy_source]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["dataset"] = "stanford"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=str(path)):
        _materialize(surface_root, reference_root, output_root)
    assert not output_root.exists()


def test_kong400_requires_run_ids_and_lengths_from_exact_atlas_summary(tmp_path, cortex_reference_factory):
    surface_root, reference_root, output_root, summary, _ = _write_kong400_inputs(tmp_path, cortex_reference_factory)
    policy = json.loads(summary.read_text(encoding="utf-8"))
    policy.pop("run_ids")
    policy.pop("run_lengths")
    summary.write_text(json.dumps(policy), encoding="utf-8")
    # A complete common-run table cannot substitute for the selected atlas summary.
    for run_id in (1, 2):
        common = summary.parent.parent / "common/runs" / f"run-{run_id}"
        common.mkdir(parents=True)
        (common / "reference_summary.json").write_text(json.dumps({
            "run_id": run_id, "run_start": (run_id - 1) * 4,
            "run_stop": run_id * 4, "run_length": 4,
        }), encoding="utf-8")
    with pytest.raises(ValueError, match=str(summary)):
        _materialize(surface_root, reference_root, output_root)
    assert not output_root.exists()


def test_ctxdrop_array_materializer_removes_common_run_step_before_roi_fc(tmp_path):
    fc_dir = tmp_path / "fc"
    materializer.materialize_ctxdrop_fc_from_arrays(
        fc_dir=fc_dir, subject="01", smoothing="2mm", hemi="L",
        hipp_ts=np.repeat((HIPP_SIGNAL + STEP)[None, :], 2, axis=0),
        valid_mask=np.array([True, False]),
        cortex_ts=np.array([CORTEX_SIGNAL + STEP, HIPP_SIGNAL + STEP]),
        cortex_roi_ids=np.array([1, 1]),
        roi_rows=[{"parcel_id": 1, "parcel_name": "retained_roi"}, {"parcel_id": 2, "parcel_name": "empty_roi"}],
        neighbor_indices=[np.array([1]), np.array([1])],
        run_segments=[
            {"run_id": 1, "run_start": 0, "run_stop": 4, "run_length": 4},
            {"run_id": 2, "run_start": 4, "run_stop": 8, "run_length": 4},
        ],
        distance_mm=20.0, local_neighbor_dir=tmp_path / "neighbors",
    )
    pooled = np.load(fc_dir / "hipp_vertex_to_network_fc.npy")
    np.testing.assert_allclose(pooled[0, 0], POOLED_FC, atol=2e-7)
    assert np.isnan(pooled[1]).all()
    assert np.isnan(pooled[:, 1]).all()
    for run_id, expected in enumerate(RUN_FC, start=1):
        run_roi = np.load(fc_dir / "runs" / f"run-{run_id}" / "hipp_vertex_to_network_fc_active.npy")
        np.testing.assert_allclose(run_roi[0, 0], expected, atol=2e-7)
        assert np.isnan(run_roi[0, 1])
    np.testing.assert_array_equal(np.load(tmp_path / "neighbors/ctxdrop_roi_cortex_counts.npy"), [[1, 0], [1, 0]])

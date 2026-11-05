"""Intrinsic pooled FC uses the declared runs and preserves undefined cells."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np

intrinsic = importlib.import_module('04_02_materialize_intrinsic_fc')


def fixture(root: Path, run_ids: list[int]) -> tuple[Path, Path, Path]:
    output = root / 'output'
    surface = root / 'surface'
    reference = root / 'reference'
    values = []
    for index in range(len(run_ids)):
        offset = index * 100.0
        values.append(np.asarray([[offset, offset + 1, offset + 2],
                                  [offset + 2, offset + 1, offset],
                                  [offset, offset, offset]], np.float32))
    ts = np.concatenate(values, axis=1)
    ts_dir = surface / 'sub-01/surface/2mm'
    ts_dir.mkdir(parents=True)
    np.save(ts_dir / 'sub-01_hemi-L_timeseries.npy', ts)
    gate = surface / 'sub-01/surface/validity'
    gate.mkdir(parents=True)
    np.save(gate / 'sub-01_hemi-L_valid_mask.npy', np.ones(3, bool))
    ref = reference / 'sub-01/reference/kong2022'
    ref.mkdir(parents=True)
    (ref / 'reference_summary.json').write_text(json.dumps({'run_ids': run_ids, 'run_lengths': [3] * len(run_ids)}))
    for rid in run_ids:
        # These older per-run files must not choose the authoritative run roster.
        old = reference / f'sub-01/reference/common/runs/run-{rid}'
        old.mkdir(parents=True)
        (old / 'reference_summary.json').write_text(json.dumps({'run_length': 3}))
    return output, surface, reference


def compute(root: Path, run_ids: list[int]) -> tuple[np.ndarray, Path]:
    output, surface, reference = fixture(root, run_ids)
    intrinsic.materialize_subject(output, surface, '01', ['2mm'], ['L'], reference_root=reference)
    fc_dir = output / 'sub-01/fc/intrinsic/2mm/hemi_L'
    return np.load(fc_dir / 'hipp_vertex_to_vertex_fc_active.npy'), fc_dir


def test_intrinsic_pooled_correlation_removes_run_offset_sign_flip(tmp_path: Path) -> None:
    fc, _ = compute(tmp_path, [1, 2, 3, 4])
    np.testing.assert_allclose(fc[:2, :2], [[1., -1.], [-1., 1.]], atol=1e-6)


def test_intrinsic_stepwise_constant_row_stays_undefined_including_diagonal(tmp_path: Path) -> None:
    fc, fc_dir = compute(tmp_path, [1, 2, 3, 4])
    assert np.isnan(fc[2]).all()
    assert np.isnan(fc[:, 2]).all()
    run = np.load(fc_dir / 'runs/run-1/hipp_vertex_to_vertex_fc_active.npy')
    assert np.isnan(run[2]).all()
    assert np.isnan(run[:, 2]).all()


def test_intrinsic_uses_declared_nonconsecutive_runs(tmp_path: Path) -> None:
    fc, fc_dir = compute(tmp_path, [2, 7])
    np.testing.assert_allclose(fc[:2, :2], [[1., -1.], [-1., 1.]], atol=1e-6)
    for rid in (2, 7):
        run = np.load(fc_dir / f'runs/run-{rid}/hipp_vertex_to_vertex_fc_active.npy')
        np.testing.assert_allclose(run[:2, :2], [[1., -1.], [-1., 1.]], atol=1e-6)
        assert np.isnan(run[2]).all()
    assert not (fc_dir / 'runs/run-1').exists()
    manifest = json.loads((fc_dir / 'stage_manifest.json').read_text())
    assert (manifest['run_ids'], manifest['run_lengths']) == ([2, 7], [3, 3])

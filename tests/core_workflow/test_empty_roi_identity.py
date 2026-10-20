"""An empty Kong parcel keeps its fixed feature identity after screening."""
from __future__ import annotations

import csv
import importlib
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

shared = importlib.import_module('scripts.core_workflow.lib.extract_cortex_merged_network_timeseries')
labels = importlib.import_module('scripts.core_workflow.lib.kong2022_label_utils')


def test_empty_parcel_keeps_its_row_and_diagnostics() -> None:
    registry = [dict(roi_id=i, roi_name=f'roi{i}', hemi='L', family17='Aud', canonical8='Auditory',
                     canonical7='Somatomotor', subregion='', rank=i) for i in (1, 2, 3)]
    ts, rows, empty, stats = shared.aggregate_hemi_roi_timeseries(
        hemi='L', hemi_dt=np.asarray([[1., 2., 3.], [3., 4., 5.], [9., 8., 7.]]),
        aligned_parcel_ids=np.asarray([1, 1, 2]), valid_mask=np.asarray([True, True, False]),
        roi_registry=registry,
    )
    assert [row['parcel_id'] for row in rows] == [1, 2, 3]
    np.testing.assert_allclose(ts[0], [2., 3., 4.])
    assert np.isnan(np.asarray(ts)[1:]).all()
    assert [row['parcel_id'] for row in empty] == [2, 3]
    assert [row['n_grayordinates_total'] for row in rows] == [2, 1, 0]
    assert [row['n_grayordinates_used'] for row in rows] == [2, 0, 0]
    assert [row['excluded_by_tsnr_gate'] for row in rows] == [False, True, True]
    assert stats['n_grayordinates_used'] == 2


@pytest.mark.parametrize('run_lengths', [(4,), (2, 2), (2, 3)])
@pytest.mark.parametrize(('module_name', 'dataset'), [
    ('extract_cortex_kong2022_route_a_roi_timeseries', 'hcp'),
    ('extract_cortex_kong2022_routeb_roi_timeseries', 'hcp'),
    ('extract_cortex_kong2022_route_a_roi_timeseries', 'stanford'),
])
def test_reference_outputs_keep_400_rows_and_count_only_contributors(tmp_path: Path, monkeypatch, module_name: str, dataset: str, run_lengths) -> None:
    module = importlib.import_module('scripts.core_workflow.lib.' + module_name)
    bundle = labels.load_kong2022_bundle()
    registry = bundle['roi_registry']
    ids = np.asarray([row['roi_id'] for row in registry], dtype=np.int32)
    left = nib.cifti2.BrainModelAxis.from_surface(np.arange(200), 32492, name='CORTEX_LEFT')
    right = nib.cifti2.BrainModelAxis.from_surface(np.arange(200), 32492, name='CORTEX_RIGHT')
    cortex = left + right
    values = np.tile(np.asarray([10., 12., 14., 12.], np.float32)[:, None], (1, 400))
    values[:, 0] = 0
    dtseries = tmp_path / 'source.dtseries.nii'
    nib.save(nib.Cifti2Image(values, header=nib.Cifti2Header.from_axes([
        nib.cifti2.SeriesAxis(0, 1, 4), cortex])), dtseries)
    dlabel = tmp_path / 'atlas.dlabel.nii'
    nib.save(nib.Cifti2Image(ids[None, :], header=nib.Cifti2Header.from_axes([
        bundle['img'].header.get_axis(0), cortex])), dlabel)
    metadata = tmp_path / 'runs.json'
    run_ids = [1] if len(run_lengths) == 1 else [1, 3]
    metadata.write_text(json.dumps({'run_ids': run_ids, 'run_lengths': run_lengths}))
    root = tmp_path / 'shared'
    args = [module.__file__, '--subject', '01', '--dtseries', str(dtseries),
            '--run-metadata-template', str(metadata), '--shared-root', str(root),
            '--kong2022-dlabel', str(dlabel)]
    if module_name.endswith('route_a_roi_timeseries'):
        label_dir = tmp_path / 'labels/sub-01'
        label_dir.mkdir(parents=True)
        for hemi, subset in [('lh', ids[:200]), ('rh', ids[200:])]:
            full = np.zeros(32492, np.int32)
            full[:200] = subset
            np.savetxt(label_dir / f'{hemi}_labels.txt', full, fmt='%d')
        args += ['--route-a-root', str(label_dir.parent), '--dataset', dataset]
    if dataset == 'stanford':
        relative_template = os.path.relpath(tmp_path / 'baseline-{hemi}.npz', Path.cwd())
        args += ['--tsnr-input-template', relative_template]

        def baseline_gate(*, data, outdir, hemi, threshold, tsnr_input, task, dataset, **kwargs):
            letter = 'L' if hemi == 'left' else 'R'
            # Observe the real extractor boundary; baseline math is covered by
            # the dedicated tSNR tests rather than this path-selection test.
            assert tsnr_input == (tmp_path / f'baseline-{letter}.npz').resolve()
            valid = np.isfinite(data).all(axis=1) & (np.ptp(data, axis=1) > 0)
            outdir.mkdir(parents=True, exist_ok=True)
            mask = outdir / f'cortex_{hemi}_valid_mask.npy'
            np.save(mask, valid)
            (outdir / f'cortex_{hemi}_tsnr_gate_summary.json').write_text(json.dumps({
                'dataset': dataset, 'task': task, 'validity_policy': 'runwise_baseline_over_detrended_sd',
                'threshold': threshold, 'baseline_input': str(tsnr_input),
                'n_grayordinates_total': int(valid.size), 'n_grayordinates_valid': int(valid.sum()),
                'n_grayordinates_invalid': int((~valid).sum()),
            }))
            return mask

        monkeypatch.setattr(module, 'compute_cortex_tsnr_gate', baseline_gate)
    monkeypatch.setattr(sys, 'argv', args)
    if sum(run_lengths) != 4:
        with pytest.raises(ValueError, match="metadata|lengths|frames|timepoints"):
            module.main()
        assert not root.exists()
        return
    assert module.main() == 0
    ref = root / 'sub-01/reference/kong2022'
    data = np.load(ref / 'cortex_roi_parcel_timeseries.npy')
    assert data.shape == (400, 4)
    assert np.isnan(data[0]).all()
    np.testing.assert_array_equal(data[1], [10., 12., 14., 12.])
    with (ref / 'cortex_roi_parcels.tsv').open() as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    assert [int(row['parcel_id']) for row in rows] == ids.tolist()
    summary = json.loads((ref / 'reference_summary.json').read_text())
    assert summary['source_run_lengths'] == list(run_lengths)
    assert summary['run_ids'] == run_ids
    for rid, length in zip(run_ids, run_lengths, strict=True):
        segment = json.loads((root / f'sub-01/reference/common/runs/run-{rid}/reference_summary.json').read_text())
        assert segment['run_length'] == length
    assert (summary['n_parcels_used_after_tsnr'], summary['n_parcels_empty_after_tsnr']) == (399, 1)
    expected_policy = 'hcp_10000_over_std' if dataset == 'hcp' else 'runwise_baseline_over_detrended_sd'
    assert (summary['dataset'], summary['task'], summary['validity_policy']) == (dataset, 'rest', expected_policy)
    if dataset == 'stanford':
        for side, hemi in [('left', 'L'), ('right', 'R')]:
            gate = json.loads((ref.parent / f'common/tsnr_gate/cortex_{side}_tsnr_gate_summary.json').read_text())
            assert gate['baseline_input'] == str((tmp_path / f'baseline-{hemi}.npz').resolve())
    run = json.loads((root / 'sub-01/reference/common/runs/run-1/reference_summary.json').read_text())
    assert run['n_roi_parcels_used'] == 399
    for name, column, key in [('canonical7', 'canonical7', 'canonical7_network'),
                              ('family17', 'family17', 'family17_network')]:
        with (ref / f'cortex_{name}_networks.tsv').open() as handle:
            network_rows = list(csv.DictReader(handle, delimiter='\t'))
        affected = next(row for row in network_rows if row[key] == registry[0][column])
        assert int(affected['n_parcels_merged']) == sum(row[column] == registry[0][column] for row in registry) - 1


def test_task_reference_requires_manifest_declared_original_bold_before_loading(tmp_path, monkeypatch):
    module = importlib.import_module('scripts.core_workflow.lib.extract_cortex_kong2022_route_a_roi_timeseries')
    monkeypatch.setattr(sys, 'argv', ['extract', '--subject', '01', '--dataset', 'hcp',
        '--task', 'emotion', '--dtseries', str(tmp_path / 'concat.dtseries.nii'),
        '--run-metadata-template', str(tmp_path / 'runs.json'), '--shared-root', str(tmp_path / 'out')])
    with pytest.raises(ValueError, match='Task.*run-dtseries.*run-artifact-manifest'):
        module.main()
    assert not (tmp_path / 'out').exists()

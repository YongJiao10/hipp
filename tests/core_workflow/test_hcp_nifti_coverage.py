"""Hippocampal coverage follows the sampled NIfTI, not another anatomical label."""
from __future__ import annotations

import importlib
import json
import os

import nibabel as nib
import numpy as np
import pytest

prep = importlib.import_module('scripts.core_workflow.02_01_prepare_group_inputs')
sampling = importlib.import_module('lib.surface_sampling')


def _record_sampling_worker_pid(subject, output_dir):
    from pathlib import Path
    (Path(output_dir) / subject).write_text(str(os.getpid()))
    return {'subject': subject, 'ok': True}


@pytest.mark.parametrize('stage', ['raw', 'smooth'])
def test_workbench_stages_share_python_process_to_bound_concurrency(tmp_path, stage):
    prep.run_parallel(stage, ['01', '02'], 2, _record_sampling_worker_pid, str(tmp_path))
    assert {(tmp_path / subject).read_text() for subject in ('01', '02')} == {str(os.getpid())}


def test_bilateral_trilinear_sampling_reads_volume_once_and_preserves_signals(tmp_path, monkeypatch):
    bold = tmp_path / 'bold.nii.gz'
    x, y, z, t = np.indices((3, 3, 3, 4))
    nib.save(nib.Nifti1Image((100*x + 10*y + z + 1000*t).astype(np.float32), np.eye(4)), bold)
    points = {
        'L': np.asarray([[.2, .3, .4], [.7, .2, .2], [1, 1, 1], [0, 2, 0]], dtype=np.float32),
        'R': np.asarray([[1.4, 1.2, 1], [.3, 1.4, .6], [1.3, .5, .1], [1, 0, 2]], dtype=np.float32),
    }
    surfaces = {}
    for hemi in ('L', 'R'):
        full = np.zeros((512, 3), dtype=np.float32)
        full[:4] = points[hemi]
        points[hemi] = full
        path = tmp_path / f'{hemi}.surf.gii'
        nib.save(nib.gifti.GiftiImage(darrays=[
            nib.gifti.GiftiDataArray(full, intent='NIFTI_INTENT_POINTSET'),
            nib.gifti.GiftiDataArray(np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int32),
                                   intent='NIFTI_INTENT_TRIANGLE'),
        ]), path)
        surfaces[hemi] = path
    calls = []
    original = sampling.run_workbench

    def run_workbench(args, **kwargs):
        if args[0] == '-volume-to-surface-mapping':
            calls.append(args[1])
        return original(args, **kwargs)

    monkeypatch.setattr(sampling, 'run_workbench', run_workbench)
    records = sampling._sample_prepared_surfaces(
        bold=bold, native=surfaces, mapped=surfaces, subject='01', density='512',
        outdir=tmp_path / 'sampled',
        registration=sampling.SurfaceRegistration('ants', tmp_path / 'transform.h5'),
        mapping_method='trilinear', smooth_iters=0,
    )
    for record in records:
        hemi = record['hemi']
        expected = (points[hemi] @ np.asarray([100, 10, 1]))[:, None] + 1000*np.arange(4)
        actual = sampling._load_metric_gifti(record['metric_gii'], n_vertices=512, expected_timepoints=4)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-4)
        assert record['source_surface'] == str(surfaces[hemi])
        assert record['surface'] == str(surfaces[hemi])
    assert calls == [str(bold)]


@pytest.mark.parametrize('dataset', ['hcp', 'banda', 'pdc', 'stanford', 'iowa_jj_tms'])
def test_hcp_raw_coverage_uses_nifti_without_cifti_and_checks_each_run(tmp_path, monkeypatch, dataset):
    bold = tmp_path / 'bold.nii.gz'
    values = np.broadcast_to(np.arange(4, dtype=np.float32), (3, 3, 3, 4)).copy()
    values[1, 0, 0] = [0, 0, 1, 2]  # Concatenation varies, but the first run does not.
    values[0, 2, 0] = np.nan
    nib.save(nib.Nifti1Image(values, np.eye(4)), bold)
    metadata = tmp_path / 'runs.json'
    metadata.write_text(json.dumps({'subject': 'sub-01', 'run_ids': [1, 2], 'run_lengths': [2, 2],
        'runs': [{'run_id': 1, 'run_label': 'first', 'n_timepoints': 2},
                 {'run_id': 2, 'run_label': 'second', 'n_timepoints': 2}]}))
    surfaces = {}
    for hemi in ('L', 'R'):
        surface = tmp_path / f'{hemi}.surf.gii'
        points = np.zeros((512, 3), dtype=np.float32)
        points[:4] = [[0, 0, 0], [1, 0, 0], [0, 2, 0], [9, 0, 0]]
        nib.save(nib.gifti.GiftiImage(darrays=[
            nib.gifti.GiftiDataArray(points,
                                   intent='NIFTI_INTENT_POINTSET'),
            nib.gifti.GiftiDataArray(np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int32),
                                   intent='NIFTI_INTENT_TRIANGLE'),
        ]), surface)
        surfaces[hemi] = surface

    # Registration is supplied by the fixture; the actual Workbench NIfTI sampler runs.
    def sample(**kwargs):
        records = []
        for hemi, surface in surfaces.items():
            metric = kwargs['outdir'] / f'sub-01_hemi-{hemi}_space-T1w_den-512_label-hipp_bold.func.gii'
            sampled = sampling.sample_volume_to_surface(bold=kwargs['bold'], surface=surface, out_metric=metric)
            records.append({'hemi': hemi, 'surface': str(surface), 'source_surface': str(surface),
                            'metric_gii': str(metric), 'n_vertices': sampled.shape[0],
                            'n_timepoints': sampled.shape[1], 'space': 'T1w',
                            'sampling_space': 'MNI152NLin6Asym', 'mapping_method': 'trilinear'})
        return records

    monkeypatch.setattr(prep, 'sample_hipp_surface_timeseries', sample)
    monkeypatch.setattr(prep, 'dataset_registration', lambda *args, **kwargs: None)
    logs = tmp_path / 'logs'
    logs.mkdir()
    result = prep.run_raw_sampling('01', str(tmp_path / 'batch'), str(tmp_path / 'shared'),
        str(tmp_path), str(bold), str(logs), run_metadata_template=str(metadata), dataset=dataset)

    assert result['ok'], result
    raw = tmp_path / 'shared/sub-01/surface/raw'
    coverage = json.loads((raw / 'sub-01_coverage_summary.json').read_text())
    assert coverage['run_lengths'] == [2, 2]
    assert coverage['hemispheres']['L']['n_vertices_unsupported'] == 3
    assert coverage['hemispheres']['R']['n_vertices_valid'] == 509
    np.testing.assert_array_equal(np.load(raw / 'sub-01_hemi-L_nifti_support.npy'), [1, 1, 1, 0] + [1] * 508)

    # The production mask must match participant coverage without EV/baseline inputs.
    for hemi, source in surfaces.items():
        target = prep.surface_path(tmp_path / 'batch', '01', hemi)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    mask_result = prep.run_validity('01', str(tmp_path / 'batch'), str(tmp_path / 'shared'),
                                    dataset=dataset)
    assert mask_result['ok']
    mask = np.load(tmp_path / 'shared/sub-01/surface/validity/sub-01_hemi-L_valid_mask.npy')
    assert mask.tolist() == [True, False, False, False] + [True] * 508

"""Hippocampal coverage follows the sampled NIfTI, not another anatomical label."""
from __future__ import annotations

import importlib
import json

import nibabel as nib
import numpy as np

prep = importlib.import_module('scripts.core_workflow.02_01_prepare_group_inputs')
sampling = importlib.import_module('lib.surface_sampling')


def test_hcp_raw_coverage_uses_nifti_without_cifti_and_checks_each_run(tmp_path, monkeypatch):
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
    logs = tmp_path / 'logs'
    logs.mkdir()
    result = prep.run_raw_sampling('01', str(tmp_path / 'batch'), str(tmp_path / 'shared'),
        str(tmp_path), str(bold), str(logs), run_metadata_template=str(metadata))

    assert result['ok'], result
    raw = tmp_path / 'shared/sub-01/surface/raw'
    coverage = json.loads((raw / 'sub-01_coverage_summary.json').read_text())
    assert coverage['run_lengths'] == [2, 2]
    assert coverage['hemispheres']['L']['n_vertices_unsupported'] == 3
    assert coverage['hemispheres']['R']['n_vertices_valid'] == 509
    np.testing.assert_array_equal(np.load(raw / 'sub-01_hemi-L_nifti_support.npy'), [1, 1, 1, 0] + [1] * 508)

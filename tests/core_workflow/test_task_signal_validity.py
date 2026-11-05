"""Explicit validity-only QC preserves task variance and rejects unusable signals."""
from pathlib import Path
import json
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.core_workflow.lib.io_utils import compute_cortex_validity


def test_task_cortex_keeps_finite_signals(tmp_path):
    signal = 10000 + 2000 * np.sin(np.arange(352))
    data = np.tile(signal, (5, 1))
    data[1] = 0
    data[2, :176], data[2, 176:] = 10000, 12000
    data[3, 176:] = 0
    data[4, 100] = np.nan
    mask = compute_cortex_validity(
        data=data, outdir=tmp_path, hemi='left', dataset='hcp',
        task='emotion', run_lengths=[176, 176],
    )
    assert np.load(mask).tolist() == [True, True, True, True, False]
    summary = json.loads((tmp_path / 'cortex_left_validity_summary.json').read_text())
    assert summary['validity_policy'] == 'finite_cortex_signal'


def test_task_gate_requires_exact_run_lengths(tmp_path):
    with pytest.raises(ValueError, match='run lengths'):
        compute_cortex_validity(data=np.ones((2, 6)), outdir=tmp_path, hemi='left',
                                dataset='hcp', task='emotion', run_lengths=[3, 2])


def test_task_validity_invariant_to_scale_and_run_baselines():
    from scripts.core_workflow.lib.task_quality import task_signal_validity
    x = np.array([[1., 2., 3., 4., 5., 6.]])
    expected = task_signal_validity(x, [3, 3])
    changed = x * 10000
    changed[:, 3:] += 50000
    np.testing.assert_array_equal(task_signal_validity(changed, [3, 3]), expected)


def test_task_consumer_rejects_old_smoothing(tmp_path):
    from scripts.core_workflow.lib.task_quality import require_task_surface_validity
    surface = tmp_path / 'sub-1' / 'surface'
    surface.mkdir(parents=True)
    (surface / 'stage_manifest.json').write_text(json.dumps({'params': {'smoothing_roi_policy': 'legacy_policy'}}))
    with pytest.raises(ValueError, match='Rebuild'):
        require_task_surface_validity(tmp_path, '1', 'L', 'emotion')


@pytest.mark.parametrize("task,dataset", [("emotion", "hcp"), ("rest", "stanford"), ("rest", "iowa_jj_tms")])
def test_hippocampal_validity_respects_dataset_and_run_policy(
    tmp_path, task, dataset, emotion_run_paths,
):
    import importlib.util
    import nibabel as nib
    sys.path.insert(0, str(ROOT / 'scripts/core_workflow'))
    spec = importlib.util.spec_from_file_location('prepare_task_validity_test', ROOT / 'scripts/core_workflow/02_01_prepare_group_inputs.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = tmp_path / 'shared/sub-1/surface/raw'
    raw.mkdir(parents=True)
    (raw / 'surface_sampling_summary.json').write_text(json.dumps({
        'task': task, 'run_ids': [1, 2], 'run_lengths': [176, 176],
        'bold_inputs': [str(path) for path in emotion_run_paths],
    }))
    signal = 10000 + 2000 * np.sin(np.arange(352))
    data = np.tile(signal, (512, 1)).astype(np.float32)
    data[1] = 0
    data[2, :176] = 10000
    data[2, 176:] = 20000
    data[3, 200] = np.inf
    for hemi in ['L', 'R']:
        nib.save(module.make_func_gifti(data), raw / f'sub-1_hemi-{hemi}_space-T1w_den-512_label-hipp_bold.func.gii')
        surface = module.surface_path(tmp_path / 'batch', '1', hemi)
        surface.parent.mkdir(parents=True, exist_ok=True)
        surface.touch()
        np.save(raw / f'sub-1_hemi-{hemi}_nifti_support.npy', np.ones(512, dtype=np.float32))
    (raw / 'sub-1_coverage_summary.json').write_text(json.dumps({
        'subject': 'sub-1', 'coverage_policy': 'bold_nifti_runwise_trilinear_validity',
        'sampling_space': 'MNI152NLin6Asym', 'run_ids': [1, 2], 'run_lengths': [176, 176],
        'hemispheres': {hemi: {'nifti_support_path': str(raw / f'sub-1_hemi-{hemi}_nifti_support.npy')}
                        for hemi in ('L', 'R')},
    }))
    module.run_validity(
        '1', str(tmp_path / 'batch'), str(tmp_path / 'shared'), task, dataset,
    )
    gate = raw.parent / 'validity'
    mask = np.load(gate / 'sub-1_hemi-L_valid_mask.npy')
    assert mask.sum() == 509
    assert mask[:4].tolist() == [True, False, False, False]
    summary = json.loads((gate / 'sub-1_hemi-L_validity_summary.json').read_text())
    assert summary['validity_policy'] == 'finite_nonconstant_each_run'
    assert summary['coverage_policy'] == 'bold_nifti_runwise_trilinear_validity'


def test_task_consumer_accepts_rebuilt_metadata(tmp_path):
    from scripts.core_workflow.lib.task_quality import require_task_surface_validity
    surface = tmp_path / 'sub-1/surface'
    gate = surface / 'validity'
    gate.mkdir(parents=True)
    policy = {'task': 'emotion', 'validity_policy': 'finite_nonconstant_each_run'}
    (surface / 'stage_manifest.json').write_text(json.dumps({'params': policy}))
    (gate / 'sub-1_hemi-L_validity_summary.json').write_text(json.dumps(policy))
    require_task_surface_validity(tmp_path, '1', 'L', 'emotion')
    with pytest.raises(ValueError, match='Rebuild'):
        require_task_surface_validity(tmp_path, '1', 'L', 'wm')


@pytest.mark.parametrize('dataset', ['stanford', 'iowa_jj_tms'])
def test_denoised_rest_rejects_invalid_signal_without_amplitude_cutoff(tmp_path, dataset):
    data = np.array([[0, 1000, -1000, 0], [0, 0, 0, 0],
                     [1, np.nan, 2, 3], [10, 10, 20, 20]], dtype=float)
    mask = compute_cortex_validity(data=data, outdir=tmp_path, hemi='left',
                                   dataset=dataset, task='rest', run_lengths=[4])
    assert np.load(mask).tolist() == [True, True, False, True]
    summary = json.loads((tmp_path / 'cortex_left_validity_summary.json').read_text())
    assert summary['validity_policy'] == 'finite_cortex_signal'
    assert summary['dataset'] == dataset

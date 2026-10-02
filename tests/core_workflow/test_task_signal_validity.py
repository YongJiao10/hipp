"""Task QC must preserve task variance and reject unusable run signals."""
from pathlib import Path
import json
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.core_workflow.lib.io_utils import compute_cortex_tsnr_gate


def test_task_cortex_keeps_large_response_but_rejects_invalid_runs(tmp_path):
    data = np.array([
        [10000, 12000, 8000, 15000, 17000, 13000],
        [0, 0, 0, 0, 0, 0],
        [10000, 10000, 10000, 12000, 12000, 12000],
        [10000, 10001, 10002, 0, 0, 0],
        [10000, np.nan, 10002, 10000, 10001, 10002],
    ])
    mask = compute_cortex_tsnr_gate(
        data=data, outdir=tmp_path, hemi='left', threshold=25,
        task='emotion', run_lengths=[3, 3],
    )
    assert np.load(mask).tolist() == [True, False, False, False, False]
    summary = json.loads((tmp_path / 'cortex_left_tsnr_gate_summary.json').read_text())
    assert summary['threshold'] is None
    assert summary['tsnr_definition'] is None
    assert summary['validity_policy'] == 'finite_nonconstant_each_run'


def test_rest_cortex_retains_existing_threshold(tmp_path):
    data = np.array([[10000, 12000, 8000], [10000, 10001, 10002], [0, 0, 0]])
    mask = compute_cortex_tsnr_gate(data=data, outdir=tmp_path, hemi='left', threshold=25)
    assert np.load(mask).tolist() == [False, True, False]


def test_task_gate_requires_exact_run_lengths(tmp_path):
    with pytest.raises(ValueError, match='run lengths'):
        compute_cortex_tsnr_gate(
            data=np.ones((2, 6)), outdir=tmp_path, hemi='left', threshold=25,
            task='emotion', run_lengths=[3, 2],
        )


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
    (surface / 'stage_manifest.json').write_text(json.dumps({'params': {'smoothing_roi_policy': 'high_tsnr_vertices_only'}}))
    with pytest.raises(ValueError, match='Rebuild'):
        require_task_surface_validity(tmp_path, '1', 'L', 'emotion')


@pytest.mark.parametrize("task,dataset", [("emotion", "hcp"), ("rest", "stanford"), ("rest", "iowa_jj_tms")])
def test_hippocampal_validity_respects_dataset_and_run_policy(tmp_path, task, dataset):
    import importlib.util
    import nibabel as nib
    sys.path.insert(0, str(ROOT / 'scripts/core_workflow'))
    spec = importlib.util.spec_from_file_location('prepare_task_validity_test', ROOT / 'scripts/core_workflow/02_01_prepare_group_inputs.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = tmp_path / 'shared/sub-1/surface/raw'
    raw.mkdir(parents=True)
    (raw / 'surface_sampling_summary.json').write_text(json.dumps({'task': 'emotion', 'run_lengths': [176, 176]}))
    signal = 10000 + 2000 * np.sin(np.arange(352))
    data = np.tile(signal, (512, 1)).astype(np.float32)
    data[1] = 0
    data[2, :176] = 10000
    data[2, 176:] = 20000
    data[3, 200] = np.inf
    for hemi in ['L', 'R']:
        nib.save(module.make_func_gifti(data), raw / f'sub-1_hemi-{hemi}_space-corobl_den-512_label-hipp_bold.func.gii')
        surface = module.surface_path(tmp_path / 'batch', '1', hemi)
        surface.parent.mkdir(parents=True, exist_ok=True)
        surface.touch()
    module.run_tsnr_gate('1', str(tmp_path / 'batch'), str(tmp_path / 'shared'), task, dataset)
    gate = raw.parent / 'tsnr_gate'
    mask = np.load(gate / 'sub-1_hemi-L_valid_mask.npy')
    assert mask.sum() == (509 if task != "rest" else 510)
    assert mask[:4].tolist() == [True, False, task == "rest", False]
    assert not (gate / 'sub-1_hemi-L_tsnr.npy').exists()
    summary = json.loads((gate / 'sub-1_hemi-L_tsnr_gate_summary.json').read_text())
    assert summary['validity_policy'] == ('finite_nonconstant_each_run' if task != 'rest' else 'finite_nonconstant')
    assert summary['threshold'] is None


def test_task_consumer_accepts_rebuilt_metadata(tmp_path):
    from scripts.core_workflow.lib.task_quality import require_task_surface_validity
    surface = tmp_path / 'sub-1/surface'
    gate = surface / 'tsnr_gate'
    gate.mkdir(parents=True)
    policy = {'task': 'emotion', 'validity_policy': 'finite_nonconstant_each_run'}
    (surface / 'stage_manifest.json').write_text(json.dumps({'params': policy}))
    (gate / 'sub-1_hemi-L_tsnr_gate_summary.json').write_text(json.dumps(policy))
    require_task_surface_validity(tmp_path, '1', 'L', 'emotion')
    with pytest.raises(ValueError, match='Rebuild'):
        require_task_surface_validity(tmp_path, '1', 'L', 'wm')


@pytest.mark.parametrize('dataset', ['stanford', 'iowa_jj_tms'])
def test_denoised_rest_rejects_invalid_signal_without_amplitude_cutoff(tmp_path, dataset):
    data = np.array([[0, 1000, -1000, 0], [0, 0, 0, 0],
                     [1, np.nan, 2, 3], [10, 10, 20, 20]], dtype=float)
    mask = compute_cortex_tsnr_gate(data=data, outdir=tmp_path, hemi='left',
                                    threshold=25, dataset=dataset)
    assert np.load(mask).tolist() == [True, False, False, True]
    summary = json.loads((tmp_path / 'cortex_left_tsnr_gate_summary.json').read_text())
    assert summary['threshold'] is None
    assert summary['tsnr_definition'] is None
    assert summary['validity_policy'] == 'finite_nonconstant'
    assert summary['dataset'] == dataset

"""Parameter changes must reach real gates, rosters, and smoothed signals."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PREP = importlib.import_module('scripts.core_workflow.02_01_prepare_group_inputs')


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    batch, shared, route = tmp_path/'batch', tmp_path/'shared', tmp_path/'route'
    subject = '01'
    points = np.array([[x, y, 0] for y in range(32) for x in range(16)], dtype=np.float32)
    faces = []
    for y in range(31):
        for x in range(15):
            a = y*16+x
            faces.extend([[a, a+1, a+16], [a+1, a+17, a+16]])
    for hemi in ('L', 'R'):
        surface = PREP.surface_path(batch, subject, hemi)
        surface.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.gifti.GiftiImage(darrays=[
            nib.gifti.GiftiDataArray(points, intent='NIFTI_INTENT_POINTSET'),
            nib.gifti.GiftiDataArray(np.asarray(faces, dtype=np.int32), intent='NIFTI_INTENT_TRIANGLE'),
        ]), surface)
        dice = surface.parent.parent/'qc'/f'sub-{subject}_hemi-{hemi}_desc-unetf3d_dice.tsv'
        dice.parent.mkdir(exist_ok=True)
        dice.write_text('0.95\n')

    # Only the process scheduler is replaced; real production work functions run.
    monkeypatch.setattr(PREP, 'run_parallel', lambda _label, subjects, _jobs, function, *args:
                        [function(sub, *args) for sub in subjects])
    wave = np.array([-1, 1, -1, 1]*3, dtype=np.float32)
    motion = tmp_path / 'Movement_Regressors.txt'
    np.savetxt(motion, np.zeros((12, 12)))
    motion_manifest = tmp_path / 'motion.tsv'
    motion_manifest.write_text('subject\trun_label\tfilename\tlss_path\n'
                              f'01\trfMRI_REST1_7T_PA\tMovement_Regressors.txt\t{motion}\n')

    def nifti_support(weights):
        raw_dir = shared / f'sub-{subject}/surface/raw'
        records = {}
        for hemi in ('L', 'R'):
            path = raw_dir / f'sub-{subject}_hemi-{hemi}_nifti_support.npy'
            np.save(path, np.asarray(weights, dtype=np.float32))
            records[hemi] = {'nifti_support_path': str(path), 'n_vertices_total': 512,
                             'n_vertices_unsupported': int(np.count_nonzero(np.asarray(weights) == 0))}
        (raw_dir / 'surface_sampling_summary.json').write_text(json.dumps({'run_ids': [1], 'run_lengths': [12]}))
        (raw_dir / f'sub-{subject}_coverage_summary.json').write_text(json.dumps({
            'subject': f'sub-{subject}', 'coverage_policy': 'bold_nifti_runwise_trilinear_validity',
            'sampling_space': 'MNI152NLin6Asym', 'run_ids': [1], 'run_lengths': [12],
            'run_labels': ['rfMRI_REST1_7T_PA'], 'hemispheres': records,
            'density_evidence': {'density': '512', 'n_vertices': {'L': 512, 'R': 512}},
        }))

    def raw(amplitudes):
        data = 10000 + np.broadcast_to(np.asarray(amplitudes).reshape(-1, 1), (512, 1))*wave
        for hemi in ('L', 'R'):
            path = shared/f'sub-{subject}/surface/raw/sub-{subject}_hemi-{hemi}_space-T1w_den-512_label-hipp_bold.func.gii'
            path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(PREP.make_func_gifti(data), path)
        nifti_support(np.ones(512))
        return data

    def gate():
        assert PREP.run_validity(subject, str(batch), str(shared))['ok']

    def run(stage, *, dataset='hcp'):
        arguments = [
            'prepare', '--hippunfold-batch-root', str(batch), '--route-root', str(route),
            '--shared-root', str(shared), '--template-root', str(tmp_path/'template'),
            '--subjects', subject, '--jobs', '1', '--stage', stage,
            '--dataset', dataset,
            '--motion-manifest', str(motion_manifest),
        ]
        monkeypatch.setattr(sys, 'argv', arguments)
        assert PREP.main() == 0

    def state():
        directory = shared/f'sub-{subject}/surface/validity'
        return (json.loads((directory/f'sub-{subject}_hemi-L_validity_summary.json').read_text()),
                np.load(directory/f'sub-{subject}_hemi-L_valid_mask.npy'))

    return SimpleNamespace(batch=batch, shared=shared, route=route, subject=subject,
                           raw=raw, gate=gate, run=run, state=state, nifti_support=nifti_support)


def test_hcp_reuses_matching_validity_mask(workflow, capsys):
    workflow.raw(np.full(512, 1000))
    workflow.run('validity')
    summary, valid = workflow.state()
    assert valid.all()
    before = (workflow.shared/'sub-01/surface/validity/sub-01_hemi-L_valid_mask.npy').stat().st_mtime_ns
    capsys.readouterr()
    workflow.run('validity')
    assert 'validity todo=0' in capsys.readouterr().out
    assert (workflow.shared/'sub-01/surface/validity/sub-01_hemi-L_valid_mask.npy').stat().st_mtime_ns == before


def test_replacing_raw_signal_recomputes_gate(workflow):
    workflow.raw(np.full(512, 100))
    workflow.gate()
    assert workflow.state()[1].all()
    workflow.raw(np.zeros(512))
    workflow.run('validity')
    assert workflow.state()[1].sum() == 0


def test_replacing_nifti_support_recomputes_a_resumed_vertex_mask(workflow):
    workflow.raw(np.full(512, 100))
    workflow.gate()
    weights = np.ones(512)
    weights[0] = 0
    workflow.nifti_support(weights)
    workflow.run('validity')
    assert workflow.state()[1][:2].tolist() == [False, True]


def test_hcp_mask_rejects_a_consumed_series_with_stale_run_lengths(workflow):
    data = workflow.raw(np.full(512, 100))
    for hemi in ('L', 'R'):
        raw = workflow.shared/f'sub-01/surface/raw/sub-01_hemi-{hemi}_space-T1w_den-512_label-hipp_bold.func.gii'
        nib.save(PREP.make_func_gifti(data[:, :-1]), raw)
    with pytest.raises(ValueError, match='run|timepoint|frame'):
        workflow.gate()


def test_explicit_qc_stage_rebuilds_existing_roster_after_nifti_coverage_changes(workflow):
    workflow.raw(np.full(512, 1000))
    workflow.nifti_support(np.zeros(512))
    workflow.gate()
    # Zero passes are a valid QC result, recorded without running downstream work.
    workflow.run('qc')
    roster = workflow.route/'manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt'
    assert roster.read_text().strip() == ''
    workflow.nifti_support(np.ones(512))
    workflow.gate()
    workflow.run('qc')
    assert roster.read_text().split() == ['01']


def test_stage_separated_smoothing_updates_when_gate_roi_changes(workflow, capsys):
    amplitude = np.full(512, 100)
    amplitude[0] = 1000
    workflow.raw(amplitude)
    weights = np.ones(512)
    weights[0] = 0
    workflow.nifti_support(weights)
    workflow.gate()
    workflow.run('qc')
    workflow.run('smooth')
    signal = workflow.shared/'sub-01/surface/2mm/sub-01_hemi-L_timeseries.npy'
    before = np.load(signal)
    np.testing.assert_allclose(before[0], 0)
    workflow.nifti_support(np.ones(512))
    workflow.gate()
    workflow.run('qc')
    workflow.run('smooth')
    updated = np.load(signal)
    assert np.all(np.isfinite(updated))
    assert np.any(updated[0] != before[0])
    capsys.readouterr()
    workflow.run('smooth')
    assert 'todo=0' in capsys.readouterr().out


@pytest.mark.parametrize('amplitude', [100, 1000])
def test_hcp_gate_rejects_nonfinite_and_constant_vertex_series_without_sd_ceiling(workflow, amplitude):
    data = workflow.raw(np.full(512, amplitude))
    data[0, 4] = np.nan
    data[1, 4] = np.inf
    data[2] = np.nan
    data[3] = 10000
    for hemi in ('L', 'R'):
        raw = workflow.shared/f'sub-01/surface/raw/sub-01_hemi-{hemi}_space-T1w_den-512_label-hipp_bold.func.gii'
        nib.save(PREP.make_func_gifti(data), raw)
    workflow.gate()
    summary, mask = workflow.state()
    assert mask[:5].tolist() == [False, False, False, False, True]
    assert summary['n_vertices_valid'] == 508
    assert summary['n_vertices_invalid'] == 4
    directory = workflow.shared/'sub-01/surface/validity'
    roi = nib.load(directory/'sub-01_hemi-L_valid_roi.shape.gii').darrays[0].data
    np.testing.assert_array_equal(roi, mask.astype(np.float32))


@pytest.mark.parametrize('dataset', ['stanford', 'banda', 'pdc'])
def test_other_datasets_resume_from_nifti_inputs_without_baseline(workflow, dataset):
    workflow.raw(np.full(512, 100))
    PREP.run_validity('01', str(workflow.batch), str(workflow.shared),
                       dataset=dataset)
    assert workflow.state()[1].all()
    assert PREP.validity_done(workflow.shared, '01', dataset)
    workflow.raw(np.zeros(512))
    assert not PREP.validity_done(workflow.shared, '01', dataset)
    workflow.run('validity', dataset=dataset)
    summary, valid = workflow.state()
    assert not valid.any()
    assert PREP.validity_done(workflow.shared, '01', dataset)

"""Shared participant QC must accept explicit runs and separate cohort outputs."""
import csv
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from test_hcp_qc_policy import QC, prepare, result
import importlib

PREP = importlib.import_module('scripts.core_workflow.02_01_prepare_group_inputs')


def test_qc_run_inputs_require_complete_roster(tmp_path, monkeypatch):
    batch, shared, _, out, manifests = prepare(tmp_path, monkeypatch)
    inputs = tmp_path / 'qc_runs.tsv'
    inputs.write_text('subject\trun_id\trun_label\tbold_path\tn_timepoints\tmotion_path\tmotion_format\n'
                      f'100610\t1\trest\t{tmp_path}/bold.nii.gz\t2\t{tmp_path}/absent_motion.tsv\thcp_movement_degrees\n')
    monkeypatch.setattr(sys, 'argv', ['qc', '--qc-run-inputs', str(inputs),
        '--hippunfold-batch-root', str(batch), '--shared-root', str(shared),
        '--out-dir', str(out), '--manifest-dir', str(manifests)])
    with pytest.raises(SystemExit) as exc:
        QC.main()
    assert exc.value.code == 2
    assert not out.exists()


def test_qc_roster_includes_subject_without_declared_runs(tmp_path, monkeypatch):
    batch, shared, _, out, manifests = prepare(tmp_path, monkeypatch)
    inputs = tmp_path / 'qc_runs.tsv'
    inputs.write_text('subject\trun_id\trun_label\tbold_path\tn_timepoints\tmotion_path\tmotion_format\n'
                      f'100610\t1\trest\t{tmp_path}/bold.nii.gz\t2\t{tmp_path}/absent_motion.tsv\thcp_movement_degrees\n')
    roster = tmp_path / 'subjects.txt'
    roster.write_text('100610\nBANDA199\n')
    monkeypatch.setattr(sys, 'argv', ['qc', '--qc-run-inputs', str(inputs), '--subjects-manifest', str(roster),
        '--hippunfold-batch-root', str(batch), '--shared-root', str(shared),
        '--out-dir', str(out), '--manifest-dir', str(manifests)])
    assert QC.main() == 0
    payload = result(out)
    missing = next(row for row in payload['rows'] if row['subject'] == 'sub-BANDA199')
    assert missing['qc_status'] == 'unassessable'
    assert payload['status'] == 'incomplete_inputs'


def test_preparation_run_inputs_require_complete_roster(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['prepare', '--qc-run-inputs', str(tmp_path / 'runs.tsv'),
        '--stage', 'qc'])
    with pytest.raises(SystemExit) as exc:
        PREP.main()
    assert exc.value.code == 2


@pytest.mark.parametrize('dataset,task,labels,prefix', [
    ('hcp', 'emotion', ['tfMRI_EMOTION_LR', 'tfMRI_EMOTION_RL'], 'hcp_3t_emotion'),
    ('banda', 'rest', ['rfMRI_REST1_AP', 'rfMRI_REST1_PA'], 'banda'),
    ('pdc', 'rest', ['rfMRI_REST1_AP', 'rfMRI_REST1_PA'], 'pdc'),
    ('stanford', 'rest', ['rest1', 'rest2'], 'stanford'),
])
def test_explicit_runs_apply_shared_qc_and_name_the_selected_cohort(tmp_path, monkeypatch, dataset, task, labels, prefix):
    batch, shared, motion, out, manifests = prepare(tmp_path, monkeypatch)
    monkeypatch.setattr(QC, 'DEFAULT_SHARED_ROOT', tmp_path / 'hcp_rest/surface')
    monkeypatch.setattr(QC, 'DEFAULT_OUT_DIR', tmp_path / 'hcp_rest/qc')
    monkeypatch.setattr(QC, 'DEFAULT_MANIFEST_DIR', tmp_path / 'hcp_rest/manifests')
    old_rows = list(csv.DictReader(motion.open(), delimiter='\t'))
    rows = []
    bold_paths = []
    for run_id, (row, label) in enumerate(zip(old_rows, labels), 1):
        bold = tmp_path / f'run-{run_id}.nii.gz'
        nib.save(nib.Nifti1Image(np.zeros((2, 2, 2, 2), dtype=np.float32), np.eye(4)), bold)
        bold_paths.append(str(bold))
        motion_format = 'hcp_movement_degrees'
        if dataset == 'stanford':
            increment = (0.8, 0.1)[run_id - 1]
            Path(row['lss_path']).write_text(f'framewise_displacement\nn/a\n{increment}\n')
            motion_format = 'fmriprep_confounds'
        rows.append(dict(subject='100610', run_id=run_id, run_label=label,
                         bold_path=str(bold), n_timepoints=2,
                         motion_path=row['lss_path'], motion_format=motion_format))
    inputs = tmp_path / 'qc_runs.tsv'
    with inputs.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    coverage = shared / 'sub-100610/surface/raw/sub-100610_coverage_summary.json'
    data = json.loads(coverage.read_text())
    data.update(run_labels=labels, bold_inputs=bold_paths)
    coverage.write_text(json.dumps(data))
    monkeypatch.setattr(sys, 'argv', ['qc', '--dataset', dataset, '--task', task,
        '--qc-run-inputs', str(inputs), '--hippunfold-batch-root', str(batch),
        '--shared-root', str(shared), '--out-dir', str(out), '--manifest-dir', str(manifests),
        '--subjects', '100610'])
    assert QC.main() == 0
    payload = result(out)
    assert payload['rows'][0]['qc_status'] == 'pass'
    assert payload['rows'][0]['overall_mean_power_fd_mm'] == pytest.approx(0.225)
    assert payload['dataset'] == dataset and payload['task'] == task
    assert payload['motion_rotation_units'] == (None if dataset == 'stanford' else 'degrees')
    assert (manifests / f'{prefix}_hippocampus_fmri_qc_pass_t1only_latest.txt').read_text() == '100610\n'
    assert not (manifests / 'hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt').exists()


def test_required_reconstruction_version_rejects_incomplete_workflow_despite_high_dice(tmp_path, monkeypatch):
    batch, _, _, out, _ = prepare(tmp_path, monkeypatch)
    check = batch / 'sub-100610/hippunfold/output_check.json'
    check.write_text(json.dumps([{'subject': '100610', 'status': 'missing_required_outputs',
                                'run_contract': {'hippunfold_version': '2.3.0'}}]))
    monkeypatch.setattr(sys, 'argv', [*sys.argv, '--require-hippunfold-version', '2.3.0'])
    assert QC.main() == 0
    row = result(out)['rows'][0]
    assert row['qc_status'] == 'fail'
    assert row['exclude_reason'] == 'hippunfold_incomplete'


def test_fmriprep_motion_uses_original_fd_and_sets_run_first_frame_zero(tmp_path):
    path = tmp_path / 'confounds.tsv'
    path.write_text('framewise_displacement\ttrans_x\n' 'n/a\t0\n0.4\t1\n0.2\t2\n')
    mean, runs, missing = QC.read_motion([{'run_id': 1, 'run_label': 'rest', 'path': str(path),
                                        'motion_format': 'fmriprep_confounds', 'n_timepoints': 3}])
    assert not missing
    assert mean == pytest.approx(0.2)
    assert runs[0]['n_frames'] == 3


@pytest.mark.parametrize('dataset,task', [('banda', 'rest'), ('pdc', 'rest'), ('hcp', 'emotion')])
def test_other_cohorts_cannot_write_default_hcp_rest_reports(tmp_path, monkeypatch, dataset, task):
    batch, _, _, out, _ = prepare(tmp_path, monkeypatch)
    existing = out / 't1only_hippunfold_tsnr_qc.json'
    out.mkdir()
    existing.write_text('existing HCP report\n')
    monkeypatch.setattr(sys, 'argv', ['qc', '--dataset', dataset, '--task', task,
        '--qc-run-inputs', str(tmp_path / 'inputs.tsv'), '--hippunfold-batch-root', str(batch)])
    with pytest.raises(SystemExit) as error:
        QC.main()
    assert error.value.code == 2
    assert existing.read_text() == 'existing HCP report\n'


def test_full_preparation_reports_missing_source_without_scheduling_sampling(tmp_path, monkeypatch):
    batch, shared, motion, _, _ = prepare(tmp_path, monkeypatch, increments=(0.1,))
    motion_path = list(csv.DictReader(motion.open(), delimiter='\t'))[0]['lss_path']
    inputs = tmp_path / 'inputs.tsv'
    inputs.write_text('subject\trun_id\trun_label\tbold_path\tn_timepoints\tmotion_path\tmotion_format\n'
                      f'100610\t1\trfMRI_REST1_AP\t\t2\t{motion_path}\thcp_movement_degrees\n')
    route = tmp_path / 'banda'
    def scheduler(label, subjects, *args):
        assert not subjects, f'{label} scheduled a subject without a declared BOLD source'
    monkeypatch.setattr(PREP, 'run_parallel', scheduler)
    monkeypatch.setattr(sys, 'argv', ['prepare', '--dataset', 'banda', '--stage', 'all',
        '--qc-run-inputs', str(inputs), '--hippunfold-batch-root', str(batch),
        '--route-root', str(route), '--shared-root', str(shared),
        '--template-root', str(tmp_path / 'templates'), '--subjects', '100610'])
    assert PREP.main() == 0
    payload = result(route / 'qc')
    assert payload['n_subjects_unassessable'] == 1
    assert payload['rows'][0]['unassessable_reason'].startswith('missing_bold_nifti_declaration')
    assert not (tmp_path / 'templates/group_average_surfaces').exists()


def test_declared_bold_missing_reports_the_exact_source_path(tmp_path, monkeypatch):
    batch, shared, motion, _, _ = prepare(tmp_path, monkeypatch, increments=(0.1,))
    movement = list(csv.DictReader(motion.open(), delimiter='\t'))[0]['lss_path']
    missing = tmp_path / 'missing_bold.nii.gz'
    row, _ = QC.assess_subject('100610', batch, shared,
        [{'run_id': 1, 'run_label': 'rest', 'path': movement, 'bold_path': str(missing), 'n_timepoints': 2}],
        fd_threshold=0.5, dice_threshold=0.7, max_unsupported_vertices=20)
    assert row['qc_status'] == 'unassessable'
    assert row['unassessable_reason'] == f'missing_bold_nifti:{missing}'


@pytest.mark.parametrize('reason', ['motion', 'incomplete_anatomy'])
def test_preparation_skips_failed_prerequisites_before_raw_sampling(tmp_path, monkeypatch, reason):
    batch, shared, motion, _, _ = prepare(tmp_path, monkeypatch,
        increments=(2.0,) if reason == 'motion' else (0.1,))
    movement = list(csv.DictReader(motion.open(), delimiter='\t'))[0]['lss_path']
    bold = tmp_path / 'bold.nii.gz'
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2, 2), dtype=np.float32), np.eye(4)), bold)
    inputs = tmp_path / 'inputs.tsv'
    inputs.write_text('subject\trun_id\trun_label\tbold_path\tn_timepoints\tmotion_path\tmotion_format\n'
        f'100610\t1\trest\t{bold}\t2\t{movement}\thcp_movement_degrees\n')
    (batch / 'sub-100610/hippunfold/output_check.json').write_text(json.dumps([
        {'subject': '100610', 'status': 'missing_required_outputs',
         'run_contract': {'hippunfold_version': '2.3.0'}}]))
    def scheduler(label, subjects, *args):
        assert not subjects, f'{label} scheduled {reason} failure'
    monkeypatch.setattr(PREP, 'run_parallel', scheduler)
    monkeypatch.setattr(sys, 'argv', ['prepare', '--dataset', 'banda', '--stage', 'raw',
        '--qc-run-inputs', str(inputs), '--require-hippunfold-version', '2.3.0',
        '--hippunfold-batch-root', str(batch), '--shared-root', str(shared),
        '--route-root', str(tmp_path / 'banda'), '--subjects', '100610'])
    assert PREP.main() == 0


def test_censored_run_uses_original_fd_and_retained_signal_lengths(tmp_path, monkeypatch):
    batch, shared, _, _, _ = prepare(tmp_path, monkeypatch, increments=(0.1,))
    motion = tmp_path / 'motion.tsv'
    motion.write_text('framewise_displacement\n0\n0.6\n0.1\n')
    mask = tmp_path / 'outliers.tsv'
    mask.write_text('framewise_displacement\n0\n1\n0\n')
    runs = [{'run_id': 1, 'run_label': 'ses-1_task-rest', 'path': str(motion),
             'motion_format': 'fmriprep_confounds', 'n_timepoints': 2,
             'temporal_mask_path': str(mask)}]
    mean, measured, missing = QC.read_motion(runs)
    assert not missing and mean == pytest.approx(0.7 / 3)
    assert measured[0]['n_frames'] == 3
    coverage = shared / 'sub-100610/surface/raw/sub-100610_coverage_summary.json'
    payload = json.loads(coverage.read_text())
    payload['run_labels'] = ['ses-1_task-rest']
    coverage.write_text(json.dumps(payload))
    unsupported, _ = QC.read_coverage(coverage, '100610', measured)
    assert unsupported == {'L': 20, 'R': 20}


@pytest.mark.parametrize('session', [None, 'ses-1'])
def test_nih_registration_uses_the_consumed_anatomical_source(tmp_path, monkeypatch, session):
    import lib.surface_coordinates as coordinates
    monkeypatch.setattr(coordinates, 'REPO_ROOT', tmp_path / 'repo')
    anatomy = tmp_path / '3_Data_Working/Depression/NIH_rTMS/derivatives/restfmri/sub-D01'
    if session:
        anatomy = anatomy / session
    anatomy = anatomy / 'anat'
    anatomy.mkdir(parents=True)
    prefix = 'sub-D01' + (f'_{session}' if session else '')
    source = anatomy / f'{prefix}_desc-preproc_T1w.nii.gz'
    source.touch()
    link = coordinates.REPO_ROOT / 'outputs_nih_rtms/inputs/hippunfold/bids_input/sub-D01/anat/sub-D01_T1w.nii.gz'
    link.parent.mkdir(parents=True)
    link.symlink_to(source)
    registration = coordinates.dataset_registration('nih_rtms', 'D01', root=tmp_path)
    assert registration.method == 'ants'
    assert registration.image_transform == anatomy / f'{prefix}_from-MNI152NLin6Asym_to-T1w_mode-image_xfm.h5'

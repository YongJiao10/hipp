"""Physical-space regressions for the real corobl/MNI sampling defect."""
from __future__ import annotations

import importlib
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sampling = importlib.import_module('lib.surface_sampling')
prep = importlib.import_module('scripts.core_workflow.02_01_prepare_group_inputs')


def test_corobl_cannot_be_used_for_physical_volume_sampling(tmp_path):
    with pytest.raises(ValueError, match='T1w'):
        sampling.sample_hipp_surface_timeseries(
            bold=tmp_path/'bold.nii.gz', hippunfold_dir=tmp_path,
            subject='01', density='512', outdir=tmp_path/'out', space='corobl',
        )


def test_image_pullback_field_is_inverted_for_native_to_mni_points(tmp_path):
    coordinates = importlib.import_module('lib.surface_coordinates')
    grid = np.moveaxis(np.indices((20, 20, 20)), 0, -1).astype(float)
    displacement = grid * np.array([.1, -.05, .08]) + [1, 2, -1]
    path = tmp_path/'mni_to_native_world.nii.gz'
    nib.save(nib.Nifti1Image(displacement, np.eye(4)), path)
    target = np.array([[5., 7., 9.], [12., 10., 6.]])
    native = target * [1.1, .95, 1.08] + [1, 2, -1]
    actual = coordinates.invert_world_displacement(native, path)
    np.testing.assert_allclose(actual, target, atol=.001)


def test_inverse_field_cannot_silently_sample_outside_its_grid(tmp_path):
    coordinates = importlib.import_module('lib.surface_coordinates')
    path = tmp_path/'small_field.nii.gz'
    nib.save(nib.Nifti1Image(np.zeros((3, 3, 3, 3)), np.eye(4)), path)
    with pytest.raises(ValueError, match='grid'):
        coordinates.invert_world_displacement(np.array([[99., 99., 99.]]), path)


def test_physical_smoothing_surface_is_native_t1w():
    path = prep.surface_path(Path('/anatomy'), '01', 'L')
    assert '_space-T1w_' in path.name


def test_resume_rejects_old_corobl_raw_signals(tmp_path):
    import json
    raw = tmp_path/'sub-01/surface/raw'
    raw.mkdir(parents=True)
    for hemi in ('L', 'R'):
        (raw/f'sub-01_hemi-{hemi}_space-corobl_den-512_label-hipp_bold.func.gii').touch()
    (raw/'surface_sampling_summary.json').write_text(json.dumps([
        {'hemi': hemi, 'space': 'corobl'} for hemi in ('L', 'R')
    ]))
    with pytest.raises(ValueError, match='sampling|coordinate|corobl'):
        prep.raw_done(tmp_path, '01')


def test_ants_points_convert_ras_to_lps_and_back(tmp_path, monkeypatch):
    coordinates = importlib.import_module('lib.surface_coordinates')
    monkeypatch.setenv('CORE_WORKFLOW_ANTS_POINTS', sys.executable)
    transform = tmp_path/'from-MNI152NLin6Asym_to-T1w_mode-image_xfm.h5'

    def apply_points(command, **kwargs):
        assert command[-1] == str(transform)
        source = Path(command[command.index('-i')+1])
        target = Path(command[command.index('-o')+1])
        values = np.atleast_2d(np.loadtxt(source, delimiter=',', skiprows=1))
        np.testing.assert_allclose(values, [[-10., -20., 30.]])
        np.savetxt(target, values + [3, -2, 1], delimiter=',', header='x,y,z', comments='')

    monkeypatch.setattr(coordinates.subprocess, 'run', apply_points)
    actual = coordinates.ants_native_to_mni(np.array([[10., 20., 30.]]), transform, workdir=tmp_path)
    np.testing.assert_allclose(actual, [[7., 22., 31.]])


@pytest.mark.parametrize('module,function', [
    ('scripts.core_workflow.04_00_materialize_cortex_regressed_surface_timeseries', 'hipp_surface_path'),
    ('scripts.core_workflow.04_01_materialize_kong2022_roi_fc', 'hipp_surface_path'),
    ('scripts.core_workflow.06_08_render_hcp_subfield_composition', 'midthickness_surface_path'),
])
def test_distance_and_area_geometry_is_native_t1w(module, function):
    path = getattr(importlib.import_module(module), function)(Path('/anatomy'), '01', 'L')
    assert '_space-T1w_' in path.name


def test_stanford_native_beta_uses_t1w_surfaces():
    module = importlib.import_module('scripts.exploratory_beta.run_native_surface_beta')
    assert module.SURFACE_SPACES['stanford'] == 'T1w'


def test_native_beta_cannot_relabel_old_corobl_mapping_as_t1w(tmp_path):
    import json
    module = importlib.import_module('scripts.exploratory_beta.run_native_surface_beta')
    (tmp_path/'run_manifest.json').write_text(json.dumps({
        'dataset': 'stanford', 'native_surface_space': 'corobl',
    }))
    with pytest.raises(ValueError, match='T1w'):
        module.require_native_mapping_space(tmp_path, 'stanford')


def test_resume_cannot_use_old_qc_after_raw_coordinates_are_fixed(tmp_path):
    import json
    gate = tmp_path/'sub-01/surface/tsnr_gate'
    gate.mkdir(parents=True)
    for hemi in ('L', 'R'):
        (gate/f'sub-01_hemi-{hemi}_tsnr_gate_summary.json').write_text(json.dumps({'invalid_fraction': .1}))
    with pytest.raises(ValueError, match='sampling|coordinate|registered'):
        prep.tsnr_done(tmp_path, '01')


def test_resume_cannot_use_corobl_millimetre_smoothing(tmp_path):
    import json
    surface = tmp_path/'sub-01/surface'
    surface.mkdir(parents=True)
    (surface/'stage_manifest.json').write_text(json.dumps({'params': {'smoothings': ['2mm', '4mm']}}))
    for smoothing in ('2mm', '4mm'):
        (surface/smoothing).mkdir()
        for hemi in ('L', 'R'):
            np.save(surface/smoothing/f'sub-01_hemi-{hemi}_timeseries.npy', np.ones((3, 4)))
    with pytest.raises(ValueError, match='smoothing|T1w'):
        prep.smoothing_done(tmp_path, '01')


def test_mni_cortex_points_use_image_pullback_to_native_without_inversion(tmp_path):
    coordinates = importlib.import_module('lib.surface_coordinates')
    path = tmp_path/'mni_to_native_world.nii.gz'
    field = np.zeros((20, 20, 20, 3))
    field[...] = [2., -3., 1.]
    nib.save(nib.Nifti1Image(field, np.eye(4)), path)
    native = coordinates.apply_world_displacement(np.array([[7., 8., 9.]]), path)
    np.testing.assert_allclose(native, [[9., 5., 10.]])


@pytest.mark.parametrize('name', ['run_branch_downstream', 'run_runwise_downstream'])
def test_iowa_resume_rejects_qc_from_before_registration(tmp_path, name):
    import json
    from types import SimpleNamespace
    sys.path.insert(0, str(ROOT/'scripts/iowa7t'))
    module = importlib.import_module(name)
    gate = tmp_path/'tsnr_gate'
    gate.mkdir()
    for hemi in ('L', 'R'):
        (gate/f'sub-01_hemi-{hemi}_tsnr_gate_summary.json').write_text(json.dumps({
            'validity_policy': module.REST_POLICY,
        }))
    assert not module.qc_policy_done(SimpleNamespace(subject='01', surface_root=tmp_path))

"""Upgrade behavior at the CLI, resource and batch boundaries."""
from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.core_workflow.lib import hippunfold_run_contract as contract_module

ROOT = Path(__file__).resolve().parents[2]
PY314 = Path('/home/yojiao/.conda/envs/py314/bin/python')


@pytest.fixture
def launcher(monkeypatch):
    path = ROOT / 'scripts/core_workflow/01_01_run_hippunfold_t1w_local.py'
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location('hippunfold_220_launcher', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_gate(monkeypatch, tmp_path, version):
    executable = tmp_path / 'hippo2/bin/hippunfold'
    executable.parent.mkdir(parents=True)
    executable.write_text('')
    monkeypatch.setattr(contract_module, 'HIPPUNFOLD_EXECUTABLE', executable)
    monkeypatch.setattr(contract_module, 'HIPPO2_PREFIX', executable.parent.parent)

    def run(argv, **kwargs):
        if argv[-1] == '--version':
            stdout = version
        elif argv[-1] == '--help':
            stdout = '--output-density {512,2k} --inner_outer_reg_method {greedy,syn} --inner_outer_reg_smoothing'
        elif argv[0] == '/bin/bash':
            stdout = str(executable)
        else:
            stdout = 'CUDA available'
        return subprocess.CompletedProcess(argv, 0, stdout, '')

    monkeypatch.setattr(contract_module.subprocess, 'run', run)


def test_runtime_gate_accepts_the_upgraded_cli(monkeypatch, tmp_path):
    fake_gate(monkeypatch, tmp_path, '2.4.0')
    contract_module.validate_local_assets({})


@pytest.mark.parametrize('version', ['2.0.0', '2.2.0', '2.3.0'])
def test_runtime_gate_rejects_the_old_cli(monkeypatch, tmp_path, version):
    fake_gate(monkeypatch, tmp_path, version)
    with pytest.raises(RuntimeError, match=r'2\.4\.0'):
        contract_module.validate_local_assets({})


def test_public_dry_run_declares_the_new_registration_parameters():
    result = subprocess.run(
        [str(PY314), str(ROOT / 'scripts/core_workflow/01_01_run_hippunfold_t1w_local.py'),
         '102311', '--dataset', 'hcp', '--dry-run'],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    payload = json.loads(result.stdout)
    assert payload['hippunfold_version'] == '2.4.0'
    assert payload['inner_outer_reg_method'] == 'greedy'
    assert payload['inner_outer_reg_smoothing'] == 3.0
    argv = payload['hippunfold_argv']
    assert argv[argv.index('--inner_outer_reg_method') + 1] == 'greedy'
    assert argv[argv.index('--inner_outer_reg_smoothing') + 1] == '3.0'


def test_an_old_cleanup_marker_does_not_skip_the_upgrade(launcher, tmp_path):
    (tmp_path / 'core_workflow_cleanup_receipt.json').write_text('{}')
    contract = SimpleNamespace(subject='102311', output_dir=tmp_path)
    assert launcher._is_completed(contract) is False


def test_new_run_refuses_an_old_atlas_output_contract(launcher, tmp_path, monkeypatch):
    manifest = tmp_path / '_batch_logs/output_integrity_manifest.json'
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({'expected_outputs': [
        'sub-{subject}/metric/sub-{subject}_atlas-multihist7_subfields.label.gii'
    ]}))
    contract = SimpleNamespace(subject='102311', output_dir=tmp_path,
                               preset=SimpleNamespace(output_root=tmp_path))
    monkeypatch.setattr(launcher.subprocess, 'run', lambda *a, **k: None)
    (tmp_path / 'output_check.json').write_text(json.dumps([{'subject': '102311'}]))
    contract.as_json = lambda: {}
    with pytest.raises(RuntimeError, match='multihist7sgi'):
        launcher._check_subject(contract)


def test_hcp_batch_rejects_a_budget_above_twelve_subjects(launcher, monkeypatch):
    monkeypatch.setattr(launcher, '_is_completed', lambda contract: False)
    monkeypatch.setattr(launcher, 'detect_system_name', lambda: 'idas')
    monkeypatch.setattr(launcher, 'prepare_runtime_directories', lambda env: None)
    def unexpected_gate(env):
        raise AssertionError('Oversized batch reached runtime validation')
    monkeypatch.setattr(launcher, 'validate_local_assets', unexpected_gate)
    with pytest.raises(ValueError, match='between 1 and 12'):
        launcher._run_batch([contract_module.resolve_run_contract('hcp', '102311')], jobs=13)


def checked_subject(tmp_path, outputs):
    output_dir = tmp_path / 'sub-102311/hippunfold'
    output_dir.mkdir(parents=True)
    manifest = tmp_path / '_batch_logs/output_integrity_manifest.json'
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({'expected_outputs': outputs}))
    run_contract = {'subject': '102311', 'hippunfold_version': '2.3.0'}
    report = [{'subject': '102311', 'status': 'complete_all_outputs_present',
               'missing_count': 0, 'run_contract': run_contract}]
    (output_dir / 'output_check.json').write_text(json.dumps(report))
    return SimpleNamespace(subject='102311', output_dir=output_dir,
                           preset=SimpleNamespace(output_root=tmp_path),
                           as_json=lambda: run_contract)


def test_restart_requires_the_checked_full_outputs_to_remain_present(launcher, tmp_path):
    contract = checked_subject(tmp_path, ['sub-{subject}/surf/example.surf.gii'])
    surface = contract.output_dir / 'sub-102311/surf/example.surf.gii'
    surface.parent.mkdir(parents=True)
    surface.write_text('surface')
    assert launcher._is_completed(contract) is True
    surface.rename(surface.with_suffix('.interrupted'))
    assert launcher._is_completed(contract) is False


def test_empty_output_contract_cannot_skip_a_subject(launcher, tmp_path):
    contract = checked_subject(tmp_path, [])
    with pytest.raises(RuntimeError, match='output_integrity_manifest.json'):
        launcher._is_completed(contract)


def test_interrupted_check_report_names_its_path(launcher, tmp_path):
    contract = checked_subject(tmp_path, ['sub-{subject}/surf/example.surf.gii'])
    (contract.output_dir / 'output_check.json').write_text('{')
    with pytest.raises(RuntimeError, match='output_check.json'):
        launcher._is_completed(contract)

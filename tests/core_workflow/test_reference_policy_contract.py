"""Reject a cached cortex reference built for a different declared QC policy."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest


quality = importlib.import_module('scripts.core_workflow.lib.task_quality')
routea = importlib.import_module('scripts.core_workflow.03_02_build_kong2022_routea_reference')
io = importlib.import_module('scripts.core_workflow.lib.io_utils')
jj = importlib.import_module('scripts.jj_tms.materialize_jj_tms_shared_inputs')



def test_reference_policy_accepts_matching_explicit_dataset(tmp_path: Path, cortex_reference_factory) -> None:
    ref = cortex_reference_factory(tmp_path, dataset='stanford')
    result = quality.require_cortex_reference_policy(ref, dataset='stanford', task='rest')
    assert result['validity_policy'] == 'finite_cortex_signal'


@pytest.mark.parametrize('mutation', ['dataset', 'policy', 'missing'])
def test_reference_policy_rejects_mismatched_right_gate(tmp_path: Path, mutation: str, cortex_reference_factory) -> None:
    ref = cortex_reference_factory(tmp_path, dataset='stanford')
    path = ref.parent / 'common/validity/cortex_right_validity_summary.json'
    gate = json.loads(path.read_text())
    if mutation == 'dataset':
        gate['dataset'] = 'iowa_jj_tms'
    elif mutation == 'policy':
        gate['validity_policy'] = 'hcp_10000_over_std'
    elif mutation == 'missing':
        del gate['task']
    path.write_text(json.dumps(gate))
    with pytest.raises(ValueError, match=r'Rebuild.*cortex_right_validity_summary.json'):
        quality.require_cortex_reference_policy(ref, dataset='stanford')


def test_reference_policy_rejects_old_hcp_reference_for_stanford(tmp_path: Path, cortex_reference_factory) -> None:
    ref = cortex_reference_factory(tmp_path, dataset='hcp')
    with pytest.raises(ValueError, match=r'Rebuild.*reference_summary.json'):
        quality.require_cortex_reference_policy(ref, dataset='stanford')


def test_reference_policy_rejects_missing_policy_metadata(tmp_path: Path, cortex_reference_factory) -> None:
    ref = cortex_reference_factory(tmp_path, dataset='hcp')
    path = ref / 'reference_summary.json'
    summary = json.loads(path.read_text())
    del summary['dataset']
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match=r'Rebuild.*reference_summary.json'):
        quality.require_cortex_reference_policy(ref, dataset='hcp')


def test_routea_does_not_reuse_interrupted_reference_with_old_stage(tmp_path: Path, cortex_reference_factory) -> None:
    ref = cortex_reference_factory(tmp_path, dataset='hcp')
    (ref / 'stage_manifest.json').write_text(json.dumps({
        'status': 'done', 'params': {'validity_policy': 'legacy_policy'},
    }))
    assert not routea.subject_is_done(tmp_path, '01', 'kong2022', dataset='hcp', expected_inputs={})


@pytest.mark.parametrize('changed', ['dataset', 'source'])
def test_routea_resume_rebuilds_when_requested_contract_changes(tmp_path: Path, changed: str, cortex_reference_factory) -> None:
    shared = tmp_path / 'shared'
    ref = cortex_reference_factory(shared, dataset='stanford')
    source = tmp_path / 'source.dtseries.nii'
    source.write_bytes(b'header-placeholder')
    metadata = tmp_path / 'runs.json'
    metadata.write_text(json.dumps({'run_ids': [2], 'run_lengths': [4]}))
    label_root = tmp_path / 'labels'
    expected = {
        'dtseries_input_mode': 'single_concat', 'dtseries_inputs': [str(source)],
        'run_artifact_manifest': None, 'run_metadata': str(metadata), 'run_ids': [2],
        'run_lengths': [4],
        'route_a_lh_labels': str(label_root / 'sub-01/lh_labels.txt'),
        'route_a_rh_labels': str(label_root / 'sub-01/rh_labels.txt'),
    }
    summary = json.loads((ref / 'reference_summary.json').read_text())
    summary.update(expected)
    (ref / 'reference_summary.json').write_text(json.dumps(summary))
    kwargs = dict(dataset='stanford', expected_inputs=expected)
    assert routea.subject_is_done(shared, '01', 'kong2022', **kwargs)
    if changed == 'dataset':
        kwargs['dataset'] = 'iowa_jj_tms'
    else:
        kwargs['expected_inputs'] = {**expected, 'run_ids': [3]}
    assert not routea.subject_is_done(shared, '01', 'kong2022', **kwargs)


def test_jj_adapter_reads_source_policy(tmp_path: Path) -> None:
    path = tmp_path / 'subject_manifest.json'
    source = {'subject': 'sub-01', 'workflow': 'jj_tms_hcp_fixed_template_matching',
              'parameters': {'task': 'rest', 'cortex_validity_policy': 'finite_cortex_signal'}}
    path.write_text(json.dumps(source))
    assert jj.read_cortex_reference_policy(path, subject='01') == {
        'dataset': 'iowa_jj_tms', 'task': 'rest', 'validity_policy': 'finite_cortex_signal'}
    source['parameters']['cortex_validity_policy'] = 'legacy_policy'
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match='subject_manifest.json'):
        jj.read_cortex_reference_policy(path, subject='01')


def test_manifest_resume_rebuilds_changed_run_partition_with_same_paths_and_ids(tmp_path: Path, monkeypatch, cortex_reference_factory) -> None:
    import hashlib
    import subprocess
    import nibabel as nib

    shared = tmp_path / 'shared'
    ref = cortex_reference_factory(shared, dataset='hcp')
    paths = [tmp_path / 'first.dtseries.nii', tmp_path / 'second.dtseries.nii']
    cortex = nib.cifti2.BrainModelAxis.from_surface(np.asarray([0]), 1, name='CORTEX_LEFT')
    for path, length in zip(paths, [4, 2], strict=True):
        nib.save(nib.Cifti2Image(np.arange(length, dtype=np.float32)[:, None],
            header=nib.Cifti2Header.from_axes([nib.cifti2.SeriesAxis(0, 1, length), cortex])), path)
    manifest = tmp_path / 'source.json'
    manifest.write_text(json.dumps({'subject': 'sub-01', 'runs': [
        {'run_id': rid, 'dtseries': {'path': str(path), 'size_bytes': path.stat().st_size,
                                   'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}
        for rid, path in zip([2, 7], paths, strict=True)
    ]}))
    label_root = tmp_path / 'labels'
    summary_path = ref / 'reference_summary.json'
    summary = json.loads(summary_path.read_text())
    summary.update({
        'dtseries_input_mode': 'in_memory_run_concat', 'dtseries_inputs': [str(path) for path in paths],
        'run_artifact_manifest': str(manifest), 'run_metadata': None, 'run_ids': [2, 7],
        'run_lengths': [3, 3],
        'route_a_lh_labels': str(label_root / 'sub-01/lh_labels.txt'),
        'route_a_rh_labels': str(label_root / 'sub-01/rh_labels.txt'),
        'kong2022_dlabel': str(routea.KONG2022_DLABEL_PATH),
        'cross_atlas_merge_json': str(routea.KONG2022_MERGE_CONFIG),
    })
    summary_path.write_text(json.dumps(summary))
    # The external extractor is unnecessary here: this test observes the batch
    # worker's real decision to rebuild rather than accept the stale partition.
    monkeypatch.setattr(routea, 'build_kong2022_reference', lambda **kwargs: 0)
    result = routea.run_subject(
        subject='01', shared_root=shared, route_a_root=label_root, data_root=tmp_path,
        dtseries_template='unused', run_artifact_manifest_template=str(manifest),
        kong2022_dlabel=None, atlas_slug='kong2022', overwrite=False, dataset='hcp',
    )
    assert result == ('01', True, 'ok')

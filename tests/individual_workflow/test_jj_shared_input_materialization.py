"""The JJ bridge must preserve actual acquisition runs and missing ROI columns."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts/core_workflow'))
ADAPTER = importlib.import_module('scripts.jj_tms.materialize_jj_tms_shared_inputs')
from lib.io_utils import load_run_segment_metadata
from lib.kong2022_label_utils import load_kong2022_bundle
from lib.tsnr import rest_tsnr_input


def save(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, values)


def metric(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(
        np.asarray(values, dtype=np.float32), intent='NIFTI_INTENT_TIME_SERIES')]), path)


@pytest.fixture
def source(tmp_path, monkeypatch):
    for name, folder in [('JJ_MATCH_ROOT', 'matching'), ('SHARED_SURFACE_ROOT', 'surface'),
                         ('SHARED_ROUTEA_ROOT', 'routea')]:
        monkeypatch.setattr(ADAPTER, name, tmp_path/folder)

    def build(ids=(1, 2, 3), *, missing_roi=False):
        sub = ADAPTER.JJ_MATCH_ROOT/'sub-1001'
        wave = np.array([-1, 1, -1, 1], dtype=np.float32)
        signs = [1, -1, 1][:len(ids)]
        hipp = np.tile(np.tile(wave, len(ids)), (512, 1))
        roi = np.tile(np.concatenate([sign*wave for sign in signs]), (400, 1))
        counts = np.ones(400, dtype=np.int32)
        if missing_roi:
            roi[0] = np.nan
            counts[0] = 0
        baseline_inputs = {}

        def baseline(domain, hemi, n_vertices, invalid_index=None):
            path = sub / 'baseline' / f'{domain}-{hemi}.npz'
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = dict(dataset='iowa_jj_tms', subject='1001', hemi=hemi, n_runs=len(ids))
            if domain == 'cortex':
                payload['vertex_ids'] = np.arange(n_vertices)
            for index in range(len(ids)):
                samples = np.tile(100 + wave, (n_vertices, 1))
                if invalid_index is not None:
                    samples[invalid_index] = 0
                payload[f'run_{index}'] = samples
                payload[f'frames_{index}'] = np.arange(4)
            np.savez(path, **payload)
            baseline_inputs[f'tsnr_{domain}_{hemi}'] = {'path': str(path.resolve())}
            return rest_tsnr_input(
                path, dataset='iowa_jj_tms', subject='1001', hemi=hemi,
                n_vertices=n_vertices, n_frames=4 * len(ids), threshold=25.0,
                vertex_ids=np.arange(n_vertices) if domain == 'cortex' else None,
            )

        for hemi in ('L', 'R'):
            scores, valid = baseline('hipp', hemi, 512, invalid_index=511)
            save(sub/f'surface/tsnr_gate/hemi-{hemi}_valid_mask.npy', valid)
            save(sub/f'surface/tsnr_gate/hemi-{hemi}_tsnr.npy', np.min(scores, axis=0))
            metric(sub/f'surface/tsnr_gate/hemi-{hemi}_valid_roi.shape.gii', valid)
            save(sub/f'surface/raw/hemi-{hemi}_timeseries.npy', hipp)
            metric(sub/f'surface/raw/sub-1001_hemi-{hemi}_space-T1w_den-512_label-hipp_bold.func.gii', hipp)
            for smooth in ('2mm', '4mm'):
                save(sub/f'surface/{smooth}/hemi-{hemi}_timeseries.npy', hipp)
                metric(sub/f'surface/{smooth}/sub-1001_hemi-{hemi}_space-T1w_den-512_label-hipp_bold_{smooth}.func.gii', hipp)
                # Legacy pooled cache deliberately differs from true run FCs.
                pooled = np.full((512, 400), np.mean(signs), dtype=np.float32)
                pooled[-1] = np.nan
                if missing_roi:
                    pooled[:, 0] = np.nan
                save(sub/f'matching/{smooth}/hemi-{hemi}/vertex_to_roi_pearson_fc.npy', pooled)
            cortex_scores, cortex_valid = baseline(
                'cortex', hemi, 200, invalid_index=0 if missing_roi and hemi == 'L' else None,
            )
            save(sub/f'cortex/hemi-{hemi}_valid_mask.npy', cortex_valid)
            save(sub/f'cortex/hemi-{hemi}_tsnr.npy', np.min(cortex_scores, axis=0))
        save(sub/'cortex/kong2022_roi_timeseries.npy', roi)
        save(sub/'cortex/kong2022_roi_vertex_counts.npy', counts)
        evidence = {'subject': '1001', 'run_lengths': [4]*len(ids),
                    'total_timepoints': 4*len(ids), 'concat_timepoints': 4*len(ids),
                    'runs': [{'run_id': rid, 'volume_timepoints': 4,
                              'dtseries_timepoints': 4, 'grayordinates': 91282} for rid in ids]}
        metadata = sub/'subject_manifest.json'
        metadata.write_text(json.dumps({
            'subject': 'sub-1001', 'workflow': 'jj_tms_hcp151_fixed_template_matching',
            'parameters': {'task': 'rest', 'tsnr_threshold': 25.0,
                           'tsnr_definition': 'runwise_baseline_over_detrended_sd',
                           'validity_policy': 'runwise_baseline_over_detrended_sd'},
            'inputs': baseline_inputs, 'header_evidence': evidence,
        }))
        return sub, metadata, signs
    return build


def test_registry_comes_from_shipped_authoritative_atlas():
    rows = ADAPTER.load_kong2022_reference_table()
    expected = load_kong2022_bundle()['roi_registry']
    assert [r['roi_id'] for r in rows] == [r['roi_id'] for r in expected]
    assert [r['roi_name'] for r in rows] == [r['roi_name'] for r in expected]


@pytest.mark.parametrize('ids', [(1, 2, 3), (2, 3)])
def test_bridge_computes_actual_run_fc_and_preserves_roi_missingness(source, ids):
    _, metadata, signs = source(ids, missing_roi=True)
    ADAPTER.materialize_subject('1001', load_kong2022_bundle()['roi_registry'])
    for smooth in ('2mm', '4mm'):
        for hemi in ('L', 'R'):
            directory = ADAPTER.SHARED_ROUTEA_ROOT/f'sub-1001/fc/kong2022/{smooth}/hemi_{hemi}'
            for filename in ('hipp_vertex_to_network_fc.npy', 'hipp_vertex_to_roi_fc.npy'):
                pooled = np.load(directory/filename)
                np.testing.assert_allclose(pooled[:-1, 1:], np.mean(signs), atol=2e-6)
                assert np.isnan(pooled[-1]).all()
                assert np.isnan(pooled[:, 0]).all()
            for rid, sign in zip(ids, signs):
                run = directory/f'runs/run-{rid}'
                for family in ('network', 'roi'):
                    actual = np.load(run/f'hipp_vertex_to_{family}_fc_active.npy')
                    np.testing.assert_allclose(actual[:, 1:], sign, atol=2e-6)
                    assert np.isnan(actual[:, 0]).all()
                np.testing.assert_allclose(np.load(run/'hipp_vertex_to_canonical7_fc_active.npy'), sign, atol=2e-6)
            for rid in {1, 2, 3, 4}-set(ids):
                assert not (directory/f'runs/run-{rid}').exists()
            summary = json.loads((directory/'fc_summary.json').read_text())
            assert summary['run_ids'] == list(ids)
            assert summary['run_lengths'] == [4]*len(ids)
            assert summary['run_labels'] == [f'run-{rid}' for rid in ids]
            assert Path(summary['run_metadata_path']) == metadata.resolve()


@pytest.mark.parametrize('problem', ['missing', 'modalities', 'lengths', 'total'])
def test_bad_source_run_metadata_fails_before_writing_fc(source, problem):
    _, metadata, _ = source()
    evidence = json.loads(metadata.read_text())['header_evidence']
    if problem == 'missing':
        # Move the agent-owned fixture rather than delete any file.
        metadata.rename(metadata.with_suffix('.unselected.json'))
    else:
        if problem == 'modalities':
            evidence['runs'][1]['dtseries_timepoints'] = 5
        elif problem == 'lengths':
            evidence['run_lengths'][1] = 5
        else:
            evidence['concat_timepoints'] = 13
        metadata.write_text(json.dumps({'header_evidence': evidence}))
    with pytest.raises((ValueError, FileNotFoundError), match='metadata|timepoints|disagree|length'):
        ADAPTER.materialize_subject('1001', load_kong2022_bundle()['roi_registry'])
    assert not (ADAPTER.SHARED_ROUTEA_ROOT/'sub-1001/fc').exists()


def test_actual_jj_header_evidence_is_understood(source):
    _, metadata, _ = source((2, 3))
    assert load_run_segment_metadata(metadata) == ((2, 3), (4, 4))

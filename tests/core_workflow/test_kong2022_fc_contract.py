from __future__ import annotations
import importlib
import json
import sys
from pathlib import Path
import numpy as np
import pytest
CORE = Path(__file__).resolve().parents[2] / 'scripts/core_workflow'
sys.path.insert(0, str(CORE))
materializer = importlib.import_module('04_01_materialize_kong2022_roi_fc')
gradients = importlib.import_module('05_01_compute_hippomaps_gradients')
labels = importlib.import_module('lib.kong2022_label_utils')

def fixture(tmp_path, *, columns=400, permute=False):
    ref = tmp_path / 'sub-01/reference/kong2022'
    ref.mkdir(parents=True)
    rows = labels.load_kong2022_bundle()['roi_registry']
    if permute:
        rows[0], rows[1] = rows[1], rows[0]
    (ref / 'cortex_roi_parcels.tsv').write_text('parcel_id\tparcel_name\n' + ''.join(f"{r['roi_id']}\t{r['roi_name']}\n" for r in rows))
    np.save(ref / 'cortex_roi_parcel_timeseries.npy', np.ones((400, 8)))
    fc_dir = tmp_path / 'sub-01/fc/kong2022/2mm/hemi_L'
    fc_dir.mkdir(parents=True)
    np.save(fc_dir / 'hipp_vertex_to_network_fc.npy', np.ones((3, columns)))
    definition = 'pearson_vertex_to_kong2022_roi_timeseries'
    (fc_dir / 'stage_manifest.json').write_text(json.dumps({'stage':'shared_fc','params':{'fc_definition':definition},'inputs':[str(ref / 'cortex_roi_parcel_timeseries.npy')]}))
    (fc_dir / 'fc_summary.json').write_text(json.dumps({'fc_definition':definition, 'n_roi_targets_used':columns,'fc_shape':[3,columns],'n_timepoints':8}))
    return fc_dir

def test_gradient_rejects_seven_features_for_any_kong_route(tmp_path):
    fixture(tmp_path, columns=7)
    with pytest.raises(ValueError, match='400'):
        gradients.load_subject_fc(tmp_path, '01', 'kong2022', '2mm', 'L', np.ones(3, bool))

def test_gradient_accepts_validated_legacy_400(tmp_path):
    fixture(tmp_path)
    result, _ = gradients.load_subject_fc(tmp_path, '01', 'kong2022', '2mm', 'L', np.ones(3, bool), require_canonical7=True)
    assert result.shape == (3, 400)

def test_gradient_rejects_wrong_roi_order(tmp_path):
    fixture(tmp_path, permute=True)
    with pytest.raises(ValueError, match='ROI order'):
        gradients.load_subject_fc(tmp_path, '01', 'kong2022', '2mm', 'L', np.ones(3, bool))

def test_materializer_targets_roi_timeseries():
    assert materializer.reference_fc_target_files('kong2022') == ('cortex_roi_parcels.tsv', 'cortex_roi_parcel_timeseries.npy', 'parcel_name')


def test_materializer_derives_canonical7_for_validated_historical_roi_table(tmp_path):
    # Break caught: directly aggregating an old 400-ROI TSV without canonical7.
    ref = tmp_path / 'sub-01/reference/kong2022'
    ref.mkdir(parents=True)
    rows = labels.load_kong2022_bundle()['roi_registry']
    tsv = ref / 'cortex_roi_parcels.tsv'
    tsv.write_text('parcel_id\tparcel_name\tfamily17\tcanonical8\n' + ''.join(
        f"{r['roi_id']}\t{r['roi_name']}\t{r['family17']}\t{r['canonical8']}\n"
        for r in rows
    ))
    original_tsv = tsv.read_bytes()
    roi = np.broadcast_to(np.arange(8, dtype=np.float32), (400, 8)).copy()
    np.save(ref / 'cortex_roi_parcel_timeseries.npy', roi)
    result = materializer.prepare_reference_profiles(
        ref_subj_root=ref.parent.parent, atlas_slug='kong2022',
        target_tsv_path=tsv, target_timeseries_path=ref / 'cortex_roi_parcel_timeseries.npy',
        roi_ts=roi, roi_names=[r['roi_name'] for r in rows],
    )
    np.testing.assert_array_equal(result[0], roi)
    np.testing.assert_array_equal(result[4], np.broadcast_to(np.arange(8), (7, 8)))
    assert result[7]['feature_count'] == 400
    assert all(r['canonical7'] == 'Somatomotor' for r in result[6] if r['family17'] == 'Aud')
    assert tsv.read_bytes() == original_tsv

# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportOptionalSubscript=false, reportPossiblyUnboundVariable=false
"""Inference must preserve subjects, missingness and per-map multiplicity."""
from pathlib import Path
import sys
import numpy as np
import pytest
from scipy import stats
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/visualize/lib'))
import cortex_group_contrasts as contrasts


def test_figure_cohorts_uses_explicit_kong_comparison_column():
    class Renderer:
        def get_multidataset_cohorts(self, grid):
            raise AssertionError("legacy multidataset resolver must not be called")

    grid = contrasts.renderer_module().GridSpec("2mm", "36p", signal_branch="standard")
    cohorts = contrasts.figure_cohorts(Renderer(), grid)
    assert len(cohorts) == 5
    assert {entry["dataset_key"] for entry in cohorts} == {"hcp", "iowa_jj_tms"}


def test_wta_profile_cache_path_separates_standard_and_ctxreg(monkeypatch, tmp_path):
    monkeypatch.setattr(contrasts, "FIGURE_DIR", tmp_path)
    cohort = {
        "id": "iowa_pat",
        "method": "kong2022_7net",
        "legacy_7net_namespace": True,
    }
    standard = SimpleNamespace(signal_branch="standard", variant="36p", smoothing="2mm")
    ctxreg = SimpleNamespace(signal_branch="ctxreg", variant="36p", smoothing="2mm")

    standard_path = contrasts.wta_profile_cache_path(cohort, standard)
    ctxreg_path = contrasts.wta_profile_cache_path(cohort, ctxreg)

    assert standard_path.name.endswith("wta_pseudocluster_bilat_2mm.npy")
    assert ctxreg_path.name.endswith("wta_pseudocluster_bilat_2mm_ctxreg.npy")
    assert standard_path != ctxreg_path


def test_extraction_fingerprint_includes_lineage_gate():
    assert any(Path(item['path']).name == 'group_atlas_lineage.py'
               for item in contrasts.extraction_code())


def test_contrast_fingerprint_rejects_archive_before_reading(tmp_path):
    path = tmp_path / 'link.json'
    path.symlink_to(tmp_path / '_archive/missing.json')
    with pytest.raises(ValueError, match='archive'):
        contrasts.fingerprint(path)


def test_welch_direction_and_raw_p_threshold():
    rng = np.random.default_rng(17)
    hc = rng.normal(size=(67, 2, 400))
    group = rng.normal(size=(26, 2, 400))
    group[:, 0, :20] += 2
    group[:, 0, 20:40] -= 2
    result = contrasts.compare_profiles(group, hc)
    expected = stats.ttest_ind(group, hc, axis=0, equal_var=False)
    np.testing.assert_allclose(result['t'], expected.statistic)
    np.testing.assert_allclose(result['p'], expected.pvalue)
    np.testing.assert_array_equal(result['significant'], np.isfinite(result['p']) & (result['p'] < 0.05))
    assert result['significant'][0, :40].all()
    assert (result['t'][0, :20] > 0).all()
    single = contrasts.compare_profiles(group[:, :1], hc[:, :1])
    np.testing.assert_array_equal(single['p'][0], result['p'][0])
    assert 'q' not in result


def test_raw_p_threshold_is_used_without_cross_roi_correction():
    base = np.arange(10, dtype=float)
    group = np.broadcast_to(base[:, None, None], (10, 1, 400)).copy()
    reference = group.copy()
    group[:, 0, 0] = group[:, 0, 0] + 3.0
    result = contrasts.compare_profiles(group, reference)
    assert result['p'][0, 0] < 0.05
    assert result['significant'][0, 0]
    assert int(result['significant'].sum()) == 1
    assert 'q' not in result


def test_missing_and_degenerate_rois_never_become_zero_measurements():
    rng = np.random.default_rng(3)
    hc = rng.normal(size=(10, 1, 400))
    group = rng.normal(size=(7, 1, 400))
    group[:, 0, 0] = np.nan
    group[1:, 0, 1] = np.nan
    hc[:, 0, 2] = group[:, 0, 2] = 1
    result = contrasts.compare_profiles(group, hc)
    assert result['n_group'][0, :3].tolist() == [0, 1, 7]
    assert np.isnan(result['t'][0, :3]).all()
    assert np.isnan(result['p'][0, :3]).all()
    assert not result['significant'][0, :3].any()
    assert 'q' not in result


def test_bilateral_alignment_uses_ids_and_preserves_missingness():
    left = np.array([[[1., np.nan]], [[3., 5.]]])
    right = np.array([[[7., 9.]], [[5., 7.]]])
    values = contrasts.bilateral_profiles(left, ['a', 'b'], right, ['b', 'a'])
    np.testing.assert_allclose(values, [[[3., np.nan]], [[5., 7.]]], equal_nan=True)
    with pytest.raises(ValueError, match='subject'):
        contrasts.bilateral_profiles(left, ['a', 'a'], right, ['b', 'a'])
    with pytest.raises(ValueError, match='subject'):
        contrasts.bilateral_profiles(left, ['a', 'b'], right, ['b', 'c'])


def test_requires_400_rois_and_sufficient_subjects():
    with pytest.raises(ValueError):
        contrasts.compare_profiles(np.ones((5, 3, 7)), np.ones((5, 3, 7)))
    with pytest.raises(ValueError):
        contrasts.compare_profiles(np.ones((1, 3, 400)), np.ones((5, 3, 400)))


def test_composite_keeps_brain_pixels_and_hc_has_no_map(tmp_path):
    from PIL import Image
    r = contrasts.renderer_module()
    fc = tmp_path / 'fc.png'
    t = tmp_path / 't.png'
    Image.new('RGB', (1000, 1408), 'red').save(fc)
    Image.new('RGB', (1000, 700), 'blue').save(t)
    output = r.stack_contrast_panel(fc, t, tmp_path / 'stack.png', 't: group - Iowa HC')
    with Image.open(output) as image:
        assert image.size == (1000, 2186)
        assert image.getpixel((500, 100))[:3] == (255, 0, 0)
        assert image.getpixel((500, 1600))[:3] == (0, 0, 255)
    hc = r.stack_contrast_panel(fc, None, tmp_path / 'hc.png', 'Reference group - no comparison')
    with Image.open(hc) as image:
        assert image.size == (1000, 2186)
        assert image.getpixel((500, 1800))[:3] == (255, 255, 255)


def test_t_colors_mask_nonsignificant_and_distinguish_fc():
    r = contrasts.renderer_module()
    colors = r.contrast_colors(np.array([-4., 0., 4., np.nan]), np.array([True, False, True, False]), 5.)
    np.testing.assert_array_equal(colors[[1, 3]], np.ones((2, 4)))
    assert not np.allclose(colors[0], r.strength_colors(np.array([-0.12]))[0])
    assert not np.allclose(colors[0], colors[2])


def test_t_display_scale_is_explicit_and_separate_from_statistical_peak():
    r = contrasts.renderer_module()
    assert r.T_DISPLAY_VMAX == 4.0
    assert r.contrast_display_vmax({'t_vmax': 17.0, 't_display_vmax': 4.0}) == 4.0
    with pytest.raises(ValueError, match='t_display_vmax'):
        r.contrast_display_vmax({'t_vmax': 17.0})


def test_fc_extraction_empty_regions_stay_missing_and_partial_frames_fail():
    ts = np.array([[0., 1., 2., 3.], [3., 2., 1., 0.]])
    roi = np.array([[0., 1., 2., 3.], [np.nan] * 4])
    masks = [np.array([True, False]), np.array([False, False])]
    result = contrasts.subject_label_fc(ts, roi, masks, np.ones(2, dtype=bool), 'fixture')
    assert result[0, 0] > 7
    assert np.isnan(result[1]).all()
    assert np.isnan(result[:, 1]).all()
    roi[0, 0] = np.nan
    with pytest.raises(ValueError, match='nonfinite'):
        contrasts.subject_label_fc(ts, roi, masks, np.ones(2, dtype=bool), 'fixture')
    with pytest.raises(ValueError, match='frames'):
        contrasts.subject_label_fc(ts, roi[:, :3], masks, np.ones(2, dtype=bool), 'fixture')


def test_subject_cache_requires_rebuild_when_source_changes(tmp_path):
    import json
    src = tmp_path / 'source.txt'
    src.write_text('first')
    path = tmp_path / 'subjects.npz'
    np.savez(path, values=np.ones((2, 7, 400)), left=np.ones((2, 7, 400)), right=np.ones((2, 7, 400)), subjects=['a','b'], panels=list('abcdefg'))
    entry = dict(schema=contrasts.SCHEMA, profile=contrasts.fingerprint(path),
                 subjects=['a','b'], panels=list('abcdefg'), sources=[contrasts.fingerprint(src)],
                 code=[])
    path.with_suffix('.manifest.json').write_text(json.dumps(entry))
    values, loaded = contrasts.read_subject_cache(path, ['a','b'], list('abcdefg'), [])
    assert values.shape == (2, 7, 400)
    src.write_text('changed')
    with pytest.raises(ValueError, match='rebuild-contrast-profiles'):
        contrasts.read_subject_cache(path, ['a','b'], list('abcdefg'), [])


def test_display_mean_mismatch_fails_closed():
    x = np.ones((3, 4, 400))
    expected = np.ones((4, 400))
    contrasts.validate_display_mean(x, expected, 'canonical_wta.npy')
    x[0, 0, 0] = 9
    with pytest.raises(ValueError, match='canonical_wta.npy'):
        contrasts.validate_display_mean(x, expected, 'canonical_wta.npy')


def test_hippo_panels_reject_auditory_and_wrong_sample_count(tmp_path):
    import nibabel as nib
    r = contrasts.renderer_module()
    path = tmp_path / 'labels.label.gii'
    from lib.gifti_labels import make_named_label_gifti
    nib.save(make_named_label_gifti(np.array([1, 1]), [(0, 'Null', (0,0,0,0)), (1, 'Auditory', (0,255,255,255))]), path)
    manifest = {'network_taxonomy': 'canonical7', 'n_subjects': 26, 'subjects': [f'sub-{i}' for i in range(26)]}
    with pytest.raises(ValueError, match='Auditory'):
        r.validate_hippo_panel_source(manifest, path, 26, 'canonical7')
    nib.save(make_named_label_gifti(np.array([1, 1]), [(0, 'Null', (0,0,0,0)), (1, 'Somatomotor', (0,255,255,255))]), path)
    r.validate_hippo_panel_source(manifest, path, 26, 'canonical7')
    with pytest.raises(ValueError, match='sample'):
        r.validate_hippo_panel_source(manifest, path, 27, 'canonical7')
    manifest['network_taxonomy'] = 'canonical8'
    with pytest.raises(ValueError, match='taxonomy'):
        r.validate_hippo_panel_source(manifest, path, 26, 'canonical7')


def test_original_source_selection_is_explicit_and_only_iowa36p():
    r = contrasts.renderer_module()
    g = r.GridSpec('2mm', '36p')
    cohort = {'dataset_key': 'iowa_jj_tms'}
    paths = contrasts.subject_input_paths(r, cohort, g, '3010', 'L', 'original-denoised')
    assert str(paths[1]).endswith('hcp_fixed_template_matching/subjects/sub-3010/cortex/kong2022_roi_timeseries.npy')
    current = contrasts.subject_input_paths(r, cohort, g, '3010', 'L', 'canonical')
    assert 'shared_inputs/pfm/' in str(current[1])
    with pytest.raises(ValueError):
        contrasts.subject_input_paths(r, cohort, g, '3010', 'L', 'unknown')


def test_original_registry_uses_accepted_hcp_400_root():
    registry = str(contrasts.ORIGINAL_REGISTRY)
    assert "work/group_atlas/pfm_ctxreg/" in registry
    assert "routea_ctxreg_full_k3_fisherz" not in registry


def test_original_source_hashes_reject_modified_data(tmp_path, monkeypatch):
    import json
    base = tmp_path / 'sub-3010'
    base.mkdir()
    ts, roi, registry, mask = [base / x for x in ('ts.npy', 'roi.npy', 'roi.tsv', 'mask.npy')]
    np.save(ts, np.ones((2, 6), dtype=np.float32))
    np.save(roi, np.ones((400, 6), dtype=np.float32))
    np.save(mask, np.ones(2, dtype=bool))
    registry.write_text('roi_id\troi_name\n1\tROI\n')
    manifest = dict(status='ok', subject='3010', roi_registry_fingerprint=contrasts.fingerprint(registry),
                    header_evidence={'concat_timepoints':6},
                    outputs={p.name:contrasts.fingerprint(p) for p in (ts,roi,mask)})
    (base/'subject_manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(contrasts,'ORIGINAL_ROOT',tmp_path)
    contrasts.validate_original_source([ts,roi,registry,mask],'3010')
    np.save(roi,np.zeros((400,6),dtype=np.float32))
    with pytest.raises(ValueError,match='hash mismatch'):
        contrasts.validate_original_source([ts,roi,registry,mask],'3010')


def test_recovery_subject_roster_does_not_allow_silent_exclusions():
    m = {'subjects':['sub-a','sub-b'], 'cluster_roi_profile_subjects_used':None, 'n_subjects':2}
    assert contrasts.validate_profile_subjects(m,2,True)==['a','b']
    with pytest.raises(ValueError):
        contrasts.validate_profile_subjects(m,2,False)
    m['cluster_roi_profile_subjects_used']=['sub-a']
    with pytest.raises(ValueError):
        contrasts.validate_profile_subjects(m,2,True)


def test_shared_scale_rejects_changed_other_grid_result(tmp_path, monkeypatch):
    import json
    r = contrasts.renderer_module()
    monkeypatch.setattr(contrasts, 'extraction_code', lambda: [])
    names = list(r.GRADIENT_SUBREGIONS)+list(r._wta_network_order(True))
    source = tmp_path/'source.txt';source.write_text('source')
    profile = tmp_path/'profile.npz';np.savez(profile,values=np.ones((2,7,400)))
    grids={}
    for tag in r.GRID_VARIANTS:
        path=tmp_path/f'{tag}.npz'
        result=contrasts.compare_profiles(np.random.default_rng(1).normal(size=(3,7,400)),np.random.default_rng(2).normal(size=(4,7,400)))
        np.savez(path,**result)
        entry=dict(panels=names,profile=contrasts.fingerprint(profile),sources=[contrasts.fingerprint(source)],comparison=contrasts.fingerprint(path))
        grids[tag]={'iowa_pat':entry,'iowa_hc':dict(entry,comparison=None)}
    receipt=dict(schema=contrasts.SCHEMA,version='legacy',colormap=contrasts.T_CMAP,t_vmax=5.,t_display_vmax=4.,code=[],grids=grids)
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(receipt))
    contrasts.load_contrasts(manifest,r.GridSpec('2mm','36p'),'legacy')
    (tmp_path/'4mm_36p.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Stale contrast source'):
        contrasts.load_contrasts(manifest,r.GridSpec('2mm','36p'),'legacy')


def test_hcp7t_and_iowa_hc_have_no_contrast_comparison(tmp_path, monkeypatch):
    import json
    r = contrasts.renderer_module()
    monkeypatch.setattr(contrasts, 'extraction_code', lambda: [])
    names = list(r.GRADIENT_SUBREGIONS) + list(r._wta_network_order(True))
    source = tmp_path / 'source.txt'
    source.write_text('source')
    profile = tmp_path / 'profile.npz'
    np.savez(profile, values=np.ones((2, 7, 400)))

    grids = {}
    for tag in r.GRID_VARIANTS:
        path = tmp_path / f'{tag}.npz'
        result = contrasts.compare_profiles(
            np.random.default_rng(1).normal(size=(3, 7, 400)),
            np.random.default_rng(2).normal(size=(4, 7, 400))
        )
        np.savez(path, **result)
        pat_entry = dict(panels=names, profile=contrasts.fingerprint(profile), sources=[contrasts.fingerprint(source)], comparison=contrasts.fingerprint(path))
        hc_entry = dict(panels=names, profile=contrasts.fingerprint(profile), sources=[contrasts.fingerprint(source)], comparison=None)
        hcp_entry = dict(panels=names, profile=contrasts.fingerprint(profile), sources=[contrasts.fingerprint(source)], comparison=None)
        grids[tag] = {'iowa_pat': pat_entry, 'iowa_hc': hc_entry, 'hcp_7t': hcp_entry}

    receipt = dict(schema=contrasts.SCHEMA, version='legacy', colormap=contrasts.T_CMAP, t_vmax=5., t_display_vmax=4., code=[], grids=grids)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(receipt))

    receipt_loaded, results = contrasts.load_contrasts(manifest, r.GridSpec('2mm', '36p'), 'legacy')
    assert results['hcp_7t'] is None
    assert results['iowa_hc'] is None
    assert results['iowa_pat'] is not None

    # Verify append_contrast_maps behaves correctly when result is None
    cdata = {'spec': {'id': 'hcp_7t'}, 'grad_cortex_pngs': {'Anterior': 'dummy_ant.png'}, 'wta_cortex_pngs': {}}
    out_cdata = r.append_contrast_maps(cdata, None, receipt, {}, {}, tmp_path, r.GridSpec('2mm', '36p'))
    assert out_cdata['has_contrast'] is False
    assert out_cdata['grad_cortex_pngs']['Anterior'] == 'dummy_ant.png'

    # If hcp_7t has a comparison attached, load_contrasts must fail closed
    grids['2mm_36p']['hcp_7t']['comparison'] = pat_entry['comparison']
    manifest.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='hcp_7t must not have a comparison'):
        contrasts.load_contrasts(manifest, r.GridSpec('2mm', '36p'), 'legacy')

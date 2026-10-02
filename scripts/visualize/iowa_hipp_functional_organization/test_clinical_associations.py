import importlib
import numpy as np
import pandas as pd


def test_no_multiple_testing_correction_keeps_raw_p_and_family_labels():
    m = importlib.import_module('scripts.visualize.iowa_hipp_functional_organization.clinical_associations')
    rows = pd.DataFrame([
        dict(branch='standard',population='all_adjusted',metric='area_percent',p=.01),
        dict(branch='standard',population='all_adjusted',metric='segregation',p=.04),
        dict(branch='standard',population='all_adjusted',metric='area_percent',p=np.nan),
    ])
    actual = m.annotate_families(rows)
    np.testing.assert_allclose(actual.p, [.01,.04,np.nan], equal_nan=True)
    assert 'q' not in actual.columns
    assert 'family_size' not in actual.columns
    assert actual.family.tolist() == ['primary:area_and_segregation'] * 3


def test_patient_subsets_and_nih_timing_mask_are_explicit():
    m = importlib.import_module('scripts.visualize.iowa_hipp_functional_organization.clinical_associations')
    subjects = pd.DataFrame({'group':[0,1,1],'nih_timing_eligible':[True,False,True]})
    assert m.eligible_mask(subjects, 'patient', 'nih_sad_tscore').tolist() == [False,False,True]
    assert m.eligible_mask(subjects, 'all_adjusted', 'phq_1').tolist() == [True,True,True]

def test_analyze_stage_replaces_inventory_used_by_later_render(monkeypatch):
    import json
    from pathlib import Path
    import tempfile
    m = importlib.import_module('scripts.visualize.iowa_hipp_functional_organization.clinical_associations')
    # A stale clinical value must be replaced by the exact new analysis input.
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent, prefix='.clinical-test-') as folder:
        root = Path(folder)
        output = root / 'results'
        output.mkdir()
        pd.DataFrame({'subject':['sub-1001'],'score':[99]}).to_csv(
            output/'clinical_associations_subjects.csv', index=False)
        subjects = pd.DataFrame({'subject':['sub-1001'],'score':[7]})
        other = pd.DataFrame({'field':['score']})
        monkeypatch.setattr(m, 'HERE', root)
        monkeypatch.setattr(m, 'runtime_gate', lambda workers: None)
        monkeypatch.setattr(m, 'load_inputs', lambda: (subjects,other,other,other,other,{'caveats':[]}))
        def compute(s, *args, **kwargs):
            saved = pd.read_csv(output/'clinical_associations_subjects.csv')
            assert saved.score.tolist() == [7]
            assert not json.loads((output/'clinical_associations_methods.json').read_text())['completed_statistics']
            return pd.DataFrame({'family':['primary:area_and_segregation'],'p':[.5]})
        monkeypatch.setattr(m, 'compute', compute)
        monkeypatch.setattr('sys.argv', ['clinical_associations','--stage','analyze','--overwrite','--workers','1'])
        m.main()
        assert pd.read_csv(output/'clinical_associations_subjects.csv').score.tolist() == [7]
        assert json.loads((output/'clinical_associations_methods.json').read_text())['completed_statistics']

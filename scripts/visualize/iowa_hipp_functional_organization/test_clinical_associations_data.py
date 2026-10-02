"""Clinical joins must not change the participant or scoring unit."""
import importlib
import numpy as np
import pandas as pd
import pytest


def module():
    return importlib.import_module('scripts.visualize.iowa_hipp_functional_organization.clinical_associations_data')


def test_duplicate_cohort_membership_does_not_duplicate_people():
    row = dict(subject='sub-3001', branch='standard', hemi='L', system='A',
               metric='segregation', value=1.2, atlas='tm_individual', cohort='patient')
    frame = pd.DataFrame([row, dict(row, cohort='comorbid_mdd')])
    result = module().deduplicate_metrics(frame)
    assert len(result) == 1
    assert result.iloc[0]['value'] == 1.2


def test_conflicting_duplicate_metrics_fail_closed():
    row = dict(subject='sub-3001', branch='standard', hemi='L', system='A',
               metric='segregation', value=1.2, atlas='tm_individual', cohort='patient')
    with pytest.raises(ValueError, match='conflicting'):
        module().deduplicate_metrics(pd.DataFrame([row, dict(row, value=2.0)]))


def test_missing_items_never_become_zero_in_total_audit():
    result = module().audit_sum(pd.DataFrame({'a':[1,1,np.nan], 'b':[2,np.nan,np.nan],
                                              'total':[3,1,0]}), 'total', ['a','b'])
    assert result['n_mismatch'] == 0
    assert result['n_incomplete_with_total'] == 2
    assert result['status'] == 'blocked_score_mismatch'


def test_reverse_scoring_and_divisor_are_applied_once():
    d = pd.DataFrame({'a':[1,4], 'b':[2,3], 'total':[3,2]})
    assert module().audit_sum(d, 'total', ['a','b'], reverse=['a'], divisor=2)['status'] == 'verified'


def test_dictionary_hierarchy_does_not_make_phq_function_question_a_symptom():
    d = pd.DataFrame([
        ['phq_total_score','phq9','Total','[phq_1] + [phq_2]'],
        ['phq_1','phq9','Interest','0, None | 1, Some'],
        ['phq_9','phq9','Function impact','0, None | 1, Some'],
        ['phq_suicide','phq9','Suicide','0, None | 1, Some'],
    ], columns=['field','form','label','formula'])
    rows = module().build_registry(d).set_index('field')
    assert rows.loc['phq_9','parent'] == ''
    assert rows.loc['phq_1','parent'] == 'phq_total_score'
    assert rows.loc['phq_suicide','parent'] != 'phq_total_score'


def test_unknown_record_id_cannot_be_guessed():
    assert module().normalize_subject('JJ_TMS_3001') == 'sub-3001'
    with pytest.raises(ValueError):
        module().normalize_subject('subject 3001')

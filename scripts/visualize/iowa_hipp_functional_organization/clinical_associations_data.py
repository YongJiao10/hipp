"""Exact Iowa clinical fields and subject-level TM inputs; no input fallbacks."""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
WORKBOOK = ROOT / 'outputs_iowa_jj_tms/TMSfMRI_DataSharing.xlsx'
ROSTERS = {'hc': 'jj_tms_hc_subjects.txt', 'patient': 'jj_tms_patient_subjects.txt',
           'pure_mdd': 'jj_tms_pure_mdd_subjects.txt',
           'comorbid_mdd': 'jj_tms_comorbid_mdd_subjects.txt'}
FORMS = {'phq9': ('PHQ', 'Depression'), 'bdiii': ('BDI-II', 'Depression'),
         'madrssr': ('MADRS-SR', 'Depression'), 'pre_tms_panassf': ('PANAS', 'Affect'),
         'gad7': ('GAD-7', 'Anxiety'), 'stai': ('STAI', 'Anxiety'),
         'pcl5': ('PCL-5', 'Trauma'), 'whodas': ('WHODAS', 'Function'),
         'dsm5_crosscutting_measures': ('DSM-5 Cross-cutting', 'Cross-cutting'),
         'nih_toolbox_emotion_domain': ('NIH Emotion', 'Emotion / social function')}
TOTALS = ['phq_total_score', 'phq_total_score_w_suicide', 'bdi_total',
          'bdi_total_w_suicide', 'madrs_total', 'madrs_total_w_suicide',
          'gad_total_score', 'whodas_total', 'pcl_total']
SUBSCALES = ['panas_positive', 'panas_negative', 'stai_s_total', 'stai_t_total']
ALIASES = {'phq_score': 'phq_total_score', 'bdi_score': 'bdi_total',
           'madrs_score': 'madrs_total', 'gad_score': 'gad_total_score',
           'whodas_score': 'whodas_total', 'stai_state_score': 'stai_s_total',
           'stai_trait_score': 'stai_t_total', 'pcl_score': 'pcl_total',
           'panas_pos_score': 'panas_positive', 'panas_neg_score': 'panas_negative'}
LABELS = {'phq_total_score': 'PHQ: 8 symptom sum (suicide excluded)',
          'phq_total_score_w_suicide': 'PHQ: 9 symptom sum (with suicide)',
          'bdi_total': 'BDI-II: 20 item sum (suicide excluded)',
          'bdi_total_w_suicide': 'BDI-II: 21 item sum (with suicide)',
          'madrs_total': 'MADRS-SR: recorded 8 item total — scoring unresolved',
          'madrs_total_w_suicide': 'MADRS-SR: recorded 9 item total — scoring unresolved',
          'panas_positive': 'Positive affect', 'panas_negative': 'Negative affect',
          'stai_s_total': 'State anxiety', 'stai_t_total': 'Trait anxiety'}


def active_path(path):
    path = Path(path)
    if any('archive' in part.lower() for part in path.parts + path.resolve().parts):
        raise ValueError(f'Archive access prohibited: {path}')
    if not path.is_file():
        raise FileNotFoundError(f'Required exact input: {path}')
    return path


def normalize_subject(value):
    match = re.fullmatch(r'(?:JJ_TMS_|sub-)(\d{4})', str(value))
    if not match:
        raise ValueError(f'Unrecognized subject ID format: {value!r}')
    return 'sub-' + match.group(1)


def load_rosters():
    result = {}
    for name, filename in ROSTERS.items():
        lines = active_path(ROOT / 'manifests' / filename).read_text().splitlines()
        ids = ['sub-' + s.strip().removeprefix('sub-') for s in lines if s.strip() and not s.startswith('#')]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f'Roster empty/duplicate mismatch: {filename}')
        result[name] = ids
    if set(result['hc']) & set(result['patient']):
        raise ValueError('HC / patient overlap')
    if (set(result['pure_mdd']) & set(result['comorbid_mdd']) or
            set(result['pure_mdd']) | set(result['comorbid_mdd']) != set(result['patient'])):
        raise ValueError('Patient subgroup partition mismatch')
    return result


def clean_label(value):
    return re.sub(r'\s+', ' ', html.unescape(re.sub('<[^>]*>', ' ', str(value)))).strip()


def build_registry(dictionary):
    """Only declared instrument scores/items; no inferred synthetic subscales."""
    d = dictionary.copy()
    d.columns = ['field', 'form', 'label', 'formula']
    d = d.fillna('')
    by_field = d.set_index('field')
    included = []
    for row in d.to_dict('records'):
        f, form = row['field'], row['form']
        if form not in FORMS:
            continue
        is_item = bool(re.fullmatch(r'(?:phq|bdi|madrs|gad|whodas|pcl|panas|dsm5crossad)_\d+|phq_suicide|stai_[st]\d+', f))
        is_nih = form == 'nih_toolbox_emotion_domain' and f.endswith('_tscore')
        if not (is_item or f in TOTALS or f in SUBSCALES or is_nih):
            continue
        level = 'item' if is_item else ('subscale' if f in SUBSCALES or is_nih else 'total')
        scale, domain = FORMS[form]
        parent = ''
        if level == 'item':
            # A field can belong to two score variants. The shorter score is the
            # displayed parent; all true memberships remain explicit below.
            candidates = [k for k in TOTALS + SUBSCALES if k in by_field.index
                          and f in re.findall(r'\[([^]]+)\]', by_field.loc[k, 'formula'])]
            parent = candidates[0] if candidates else ''
        else:
            candidates = []
        direction = 'higher symptom burden'
        if f.startswith('panas_'):
            positive = f == 'panas_positive' or f in {f'panas_{i}' for i in [1,3,5,9,10,12,14,16,17,19]}
            direction = 'higher positive affect' if positive else 'higher negative affect'
        if f.startswith('stai_') and level == 'item' and parent:
            reverse = re.findall(r'if\(\[([^]]+)\]=1, 4', by_field.loc[parent, 'formula'])
            if f in reverse:
                direction = 'higher positive state (reverse-scored in parent)'
        if f.startswith('nih_'):
            direction = 'higher construct as named (original T-score direction)'
        if f in {'phq_1','bdi_4','bdi_12','madrs_7','dsm5crossad_1','nih_posaffect_tscore','panas_positive'}:
            domain = 'Positive affect / anhedonia-related'
        elif f in {'phq_7','bdi_19','bdi_13','madrs_5','whodas_6','pcl_19','panas_17'}:
            domain = 'Subjective attention'
        elif f in {'dsm5crossad_15','whodas_3','pcl_8'}:
            domain = 'Subjective memory'
        elif f in {'phq_8','bdi_11','bdi_20','madrs_6'}:
            domain = 'Psychomotor / energy'
        elif f in {'dsm5crossad_16','bdi_2','bdi_5','bdi_7','bdi_8','madrs_8','gad_2','gad_3','stai_t9'}:
            domain = 'Worry / negative cognition (proxy)'
        included.append(dict(field=f, form=form, scale=scale, level=level,
                              label=LABELS.get(f, clean_label(row['label'])), parent=parent,
                              memberships='|'.join(candidates), domain=domain,
                              direction=direction, formula=row['formula'],
                              timing_status='baseline_visit1', score_status='verified',
                              hierarchy_order=len(included)))
    return pd.DataFrame(included)


def audit_sum(data, total, fields, *, reverse=(), divisor=1):
    x = data[fields].apply(pd.to_numeric, errors='raise').copy()
    for field in reverse:
        x[field] = 5 - x[field]
    expected = x.sum(axis=1, min_count=len(fields)) / divisor
    actual = pd.to_numeric(data[total], errors='raise')
    complete = expected.notna() & actual.notna()
    mismatch = complete & ~np.isclose(expected, actual, atol=1e-8, equal_nan=True)
    incomplete = expected.isna() & actual.notna()
    return dict(field=total, n_complete=int(complete.sum()), n_mismatch=int(mismatch.sum()),
                n_incomplete_with_total=int(incomplete.sum()),
                status='blocked_score_mismatch' if mismatch.any() or incomplete.any() else 'verified')


def deduplicate_metrics(frame):
    keys = ['subject','branch','hemi','system','metric']
    selected = frame.loc[(frame.atlas == 'tm_individual') & frame.system.isin(['A','B','P','unassigned']) &
                         frame.metric.isin(['area_fraction','segregation','margin'])].copy()
    records = []
    for key, group in selected.groupby(keys, sort=False):
        values = group.value.to_numpy(float)
        if not np.allclose(values, values[0], rtol=0, atol=1e-12, equal_nan=True):
            raise ValueError(f'conflicting duplicate metric {key}')
        records.append(group.iloc[0])
    result = pd.DataFrame(records).reset_index(drop=True)
    area = result.metric == 'area_fraction'
    result.loc[area, 'value'] *= 100
    result.loc[area, 'metric'] = 'area_percent'
    return result


def load_inputs():
    rosters = load_rosters()
    workbook = active_path(WORKBOOK)
    raw = pd.read_excel(workbook, sheet_name='data')
    dictionary = pd.read_excel(workbook, sheet_name='dictionary')
    raw['subject'] = raw.record_subject_id.map(normalize_subject)
    if raw.subject.duplicated().any():
        raise ValueError('Duplicate workbook subject IDs')
    ids = rosters['hc'] + rosters['patient']
    missing = set(ids) - set(raw.subject)
    if missing:
        raise ValueError(f'Missing clinical subjects: {sorted(missing)}')
    raw = raw.set_index('subject').loc[ids].reset_index()
    scales = build_registry(dictionary)
    for f in scales.field:
        if f not in raw:
            raise ValueError(f'Dictionary field absent from data: {f}')
        raw[f] = pd.to_numeric(raw[f], errors='raise')
    # Exact questionnaire totals are canonical, not visit summary aliases.
    audits = []
    for r in scales.to_dict('records'):
        if r['level'] == 'item' or '[' not in r['formula']:
            continue
        fields = list(dict.fromkeys(re.findall(r'\[([^]]+)\]', r['formula'])))
        reverse = list(dict.fromkeys(re.findall(r'if\(\[([^]]+)\]=1, 4', r['formula'])))
        audit = audit_sum(raw, r['field'], fields, reverse=reverse,
                          divisor=2 if r['form'] == 'madrssr' else 1)
        audits.append(dict(audit, kind='dictionary_formula', note='No recomputation or replacement'))
        if audit['status'] != 'verified':
            if r['form'] != 'madrssr':
                raise ValueError(f'Score formula mismatch: {audit}')
            # Explicit user decision: quarantine both MADRS totals; keep items.
            scales.loc[scales.field == r['field'], 'score_status'] = 'blocked_score_mismatch'
    for row in scales.loc[scales.level == 'item'].itertuples():
        codes = [float(v) for v in re.findall(r'(?:^|\|)\s*(-?\d+(?:\.\d+)?),', row.formula)]
        if row.form == 'madrssr':
            codes = [v/2 for v in codes]  # exported response labels, not raw REDCap codes
        if codes and not raw[row.field].dropna().isin(codes).all():
            raise ValueError(f'Unexpected item code: {row.field}; allowed {codes}')
    for alias, canonical in ALIASES.items():
        a, b = raw[alias], raw[canonical]
        paired = a.notna() & b.notna()
        audits.append(dict(field=alias, kind='excluded_visit_summary', n_complete=int(paired.sum()),
                           n_mismatch=int((a[paired] != b[paired]).sum()),
                           n_incomplete_with_total=0, status='excluded_alias',
                           note=f'Use exact questionnaire field {canonical}; summary may include suicide or transcription differences'))
    # Visit 1 explicitly contains rest_pfile_1..3 and the questionnaire summary.
    if raw.visit1_date.isna().any() or raw[['rest_pfile_1','rest_pfile_2']].isna().any().any():
        raise ValueError('Missing Visit 1 / resting scan timing evidence')
    nih_delta = (pd.to_datetime(raw.nih_date) - pd.to_datetime(raw.visit1_date)).dt.days
    nih_before_tms = pd.to_datetime(raw.nih_date) < pd.to_datetime(raw.visit2_date)
    nih_eligible = (nih_delta == 0) | (nih_delta.between(1,5) & nih_before_tms)
    scales.loc[scales.form == 'nih_toolbox_emotion_domain', 'timing_status'] = 'baseline_same_day_or_within_5_days_strictly_before_visit2'
    audits.append(dict(field='nih_date', kind='timing', n_complete=int(nih_eligible.sum()),
                       n_mismatch=int((~nih_eligible).sum()), n_incomplete_with_total=0,
                       status='verified' if nih_eligible.all() else 'partial_timing_exclusion',
                       note='Same day as Visit 1; otherwise within 5 days and strictly before Visit 2'))
    subjects = pd.DataFrame(dict(subject=raw.subject, group=raw.subject.isin(rosters['patient']).astype(int),
                                 subgroup=['hc' if s in rosters['hc'] else 'pure_mdd' if s in rosters['pure_mdd'] else 'comorbid_mdd' for s in raw.subject],
                                 age=pd.to_numeric(raw.pt_age, errors='raise'),
                                 sex=pd.to_numeric(raw.pt_sex, errors='raise')))
    if not subjects.sex.isin([1,2]).all() or subjects.age.isna().any():
        raise ValueError('Missing age or unsupported sex coding; do not silently recode')
    subjects['nih_timing_eligible'] = nih_eligible.to_numpy()
    subjects = pd.concat([subjects, raw[list(scales.field)].reset_index(drop=True)], axis=1)
    # Run manifest must declare the precise existing metrics file; no discovery.
    run = json.loads(active_path(HERE/'results/run_manifest.json').read_text())
    metric_path = active_path(HERE/'results/subject_metrics.csv')
    if 'metrics' not in run['completed_stages']:
        raise ValueError(f'Run manifest does not declare completed metrics: {metric_path}')
    from .inputs import verify_record
    for filename in ['subject_metrics.csv','analysis_config.json']:
        record = run['outputs'][filename]
        if record['path'] != str(HERE/'results'/filename):
            raise ValueError(f'Unexpected manifest-declared path for {filename}: {record["path"]}')
        verify_record(record)  # honor existing producer contract; no new hashes or records
    config = json.loads(active_path(HERE/'results/analysis_config.json').read_text())
    if config['scope'] != dict(density=512,n_roi=400,smoothing='2mm',taxonomy='canonical7',variant='36p'):
        raise ValueError(f'Wrong input scope: {config["scope"]}')
    frame = pd.read_csv(metric_path)
    imaging = deduplicate_metrics(frame)
    if set(imaging.subject) != set(ids):
        raise ValueError('Imaging/clinical roster mismatch')
    for (branch, hemi, system, metric), part in imaging.groupby(['branch','hemi','system','metric']):
        if len(part) != len(set(ids)):
            raise ValueError(f'Incomplete subject cells: {branch}/{hemi}/{system}/{metric}')
    expected = {(b,h,s,m) for b in ['standard','ctxreg'] for h in ['L','R'] for s in ['A','B','P']
                for m in ['area_percent','segregation','margin']}
    actual = set(imaging[['branch','hemi','system','metric']].itertuples(index=False,name=None))
    if not expected.issubset(actual):
        raise ValueError(f'Missing exact TM variants: {expected-actual}')
    coverage = []
    for r in scales.to_dict('records'):
        for group in ['hc','patient','pure_mdd','comorbid_mdd']:
            selected = subjects.group == 1 if group == 'patient' else subjects.subgroup == group
            values = subjects.loc[selected, r['field']].dropna()
            n = int(selected.sum())
            coverage.append(dict(field=r['field'], scale=r['scale'], label=r['label'], population=group,
                                 n_total=n, n_valid=len(values), n_unique=values.nunique(),
                                 n_eligible=int((selected & subjects[r['field']].notna() &
                                      (subjects.nih_timing_eligible if r['field'].startswith('nih_') else True)).sum()),
                                 min=values.min(), max=values.max(), zero_fraction=float((values == 0).mean()) if len(values) else np.nan,
                                 missing_fraction=1-len(values)/n))
    metadata = dict(sources=[str(workbook),str(metric_path),str(ROOT/'manifests')],
                    n_subjects=93, n_hc=67, n_patient=26,
                    nih_days_from_visit1=nih_delta.value_counts().to_dict(),
                    caveats=[
                        'MADRS recorded totals fail dictionary/item consistency checks: excluded from inference by explicit user choice; items retained in exported 0–3 units.',
                        'PHQ primary total excludes suicide (8 items); BDI primary total excludes suicide (20 items). Full variants are not substitutes.',
                        'Clinical fields are exact baseline questionnaire fields; visit summary aliases and post-TMS questionnaires are excluded.',
                        'NIH: 90/93 have confirmed baseline timing; 3 assessed on Visit 2 without pre/post timing are excluded only from NIH models. Original scores remain in coverage.',
                        'PCL results describe trauma-screened respondents only; unadministered items are missing, not zero.',
                        'Group-adjusted pooled association assumes a common within-group slope. Patient estimates and floor effects must be inspected.',
                        'No motion or medication adjustment; observational associations are not causal or validated clinical predictions.',
                        'Area percentages share a hemisphere denominator. Segregation and TM labels use the same resting data.',
                        'Positive affect, worry and subjective cognitive items are related constructs/proxies, not interchangeable validated anhedonia, rumination or objective cognition scales.',
                        'Exploratory families and ctxreg are sensitivity analyses, not independent replications; nominal 95% intervals are not simultaneous intervals.',
                        'At 10,000 permutations the smallest raw p is 1/10,001; raw p-values are Monte Carlo estimates and should be interpreted descriptively alongside effect sizes and confidence intervals.',
                    ])
    return subjects, scales, imaging, pd.DataFrame(coverage), pd.DataFrame(audits), metadata

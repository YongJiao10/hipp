"""Iowa baseline symptom associations with individual template matching.

Run as a module from the repository root. Existing outputs require --overwrite;
the exact upstream input contracts are resolved in clinical_associations_data.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import time

import numpy as np
import pandas as pd

from .clinical_associations_data import HERE, load_inputs
from .clinical_associations_stats import analyze

PREFIX = 'clinical_associations_'
OUTPUTS = ['statistics.csv','scales.csv','coverage.csv','score_checks.csv',
           'subjects.csv','imaging.csv','methods.json','summary.md']


def eligible_mask(subjects, population, field):
    if population not in {'all_adjusted','all_raw','hc','patient'}:
        raise ValueError(f'Unknown population: {population}')
    mask = np.ones(len(subjects), dtype=bool)
    if population in {'hc','patient'}:
        mask &= subjects.group.to_numpy() == (1 if population == 'patient' else 0)
    if field.startswith('nih_'):
        mask &= subjects.nih_timing_eligible.to_numpy(bool)
    return mask


def family_name(branch, population, metric):
    if metric.endswith('_given_area'):
        return f'{branch}:exploratory_incremental_all_and_patient'
    if branch == 'standard' and population == 'all_adjusted':
        return 'primary:area_and_segregation' if metric != 'margin' else 'exploratory:margin'
    return f'{branch}:{population}:sensitivity'


def annotate_families(frame):
    """Label planned analysis families without applying multiple-testing correction."""
    frame = frame.copy()
    frame['family'] = [family_name(r.branch,r.population,r.metric) for r in frame.itertuples()]
    return frame


def _run_field(task):
    index, scale, subjects, wide, permutations, bootstraps, seed = task
    output = []
    for branch in ['standard','ctxreg']:
        for population in ['all_adjusted','patient','hc','all_raw']:
            selection = eligible_mask(subjects, population, scale['field'])
            selected = subjects.loc[selection]
            metrics = ['area_percent','segregation','margin']
            if population in {'all_adjusted','patient'}:
                metrics += ['segregation_given_area','margin_given_area']
            for metric in metrics:
                base_metric = metric.removesuffix('_given_area')
                for hemi in ['L','R']:
                    for system in ['A','B','P']:
                        x = wide[(branch,hemi,system,base_metric)][selection]
                        y = selected[scale['field']].to_numpy(float)
                        control = wide[(branch,hemi,system,'area_percent')][selection] if metric.endswith('_given_area') else None
                        if scale['score_status'] != 'verified' or scale['timing_status'].startswith('blocked'):
                            valid = np.isfinite(x) & np.isfinite(y)
                            result = dict(n=int(valid.sum()),rho=np.nan,p=np.nan,ci_low=np.nan,ci_high=np.nan,
                                          status=scale['score_status'] if scale['score_status']!='verified' else scale['timing_status'],
                                          reason='Preserved source field; excluded from inference by explicit scoring/timing contract',
                                          loo_min=np.nan,loo_max=np.nan,loo_sign_flip=False,delta_r2=np.nan,n_boot_valid=0)
                        else:
                            # Model-specific seeds are stable across process scheduling.
                            result = analyze(x,y,selected.age.to_numpy(float),selected.sex.to_numpy(float),
                                             selected.group.to_numpy(float), adjusted=population!='all_raw',control=control,
                                             permutations=permutations,bootstraps=bootstraps,
                                             seed=seed + index*1000 + len(output),
                                             loo=population=='all_adjusted')
                        result.update({k:scale[k] for k in ['field','scale','level','label','parent','domain','direction']})
                        result.update(branch=branch,population=population,metric=metric,hemi=hemi,system=system)
                        output.append(result)
    return output


def compute(subjects, scales, imaging, *, permutations, bootstraps, seed, workers):
    core = imaging[imaging.system.isin(['A','B','P'])]
    wide = core.pivot(index='subject',columns=['branch','hemi','system','metric'],values='value').loc[subjects.subject]
    arrays = {key:wide[key].to_numpy(float) for key in wide.columns}
    tasks = [(j,r,subjects,arrays,permutations,bootstraps,seed) for j,r in enumerate(scales.to_dict('records'))]
    results = []
    started = time.monotonic()
    if workers == 1:
        for j,t in enumerate(tasks,1):
            results.extend(_run_field(t))
            print(f'Clinical fields {j}/{len(tasks)}; {time.monotonic()-started:.0f}s',flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for j, rows in enumerate(pool.map(_run_field,tasks,chunksize=1),1):
                results.extend(rows)
                print(f'Clinical fields {j}/{len(tasks)}; {time.monotonic()-started:.0f}s',flush=True)
    return annotate_families(pd.DataFrame(results))


def write_summary(results, scales, metadata, path):
    main = results[results.family == 'primary:area_and_segregation']
    nominal = main[main.p < .05].sort_values('p')
    lines = ['# Iowa clinical associations', '',
             '93 unique subjects: 67 HC + 26 patients (7 pure, 19 comorbid).',
             f'{len(scales)} declared clinical fields; {len(main)} planned primary cells; '
             f'{main.p.notna().sum()} estimable primary cells; {len(nominal)} primary cells with nominal p < 0.05. '
             'No multiple-testing correction was applied.', '',
             'Primary models: rank-based association adjusted for age, sex and HC/patient status. '
             'Area and segregation across all fields and six anatomical cells are reported with raw two-sided permutation p values.', '',
             '## Primary findings', '']
    if nominal.empty:
        lines.append('No primary association had nominal p < 0.05. These raw p values are descriptive and do not establish TM clinical value.')
    else:
        lines += ['| Field | Metric | Hemisphere/system | N | rho | 95% CI | raw p |',
                  '|---|---|---|---:|---:|---|---:|']
        for r in nominal.itertuples():
            lines.append(f'| {r.field} | {r.metric} | {r.hemi}-{r.system} | {r.n} | {r.rho:.3f} | '
                         f'[{r.ci_low:.3f}, {r.ci_high:.3f}] | {r.p:.5g} |')
    lines += ['', '## Analysis families (descriptive grouping only)', '',
              '| Family | Planned cells | Estimable | nominal p < .05 |', '|---|---:|---:|---:|']
    for family, group in results.groupby('family', sort=False):
        lines.append(f'| {family} | {len(group)} | {group.p.notna().sum()} | {(group.p < .05).sum()} |')
    lines += ['', '## Interpretation limits', ''] + ['- ' + s for s in metadata['caveats']]
    lines += ['', 'The complete field hierarchy and all raw p values '
              'are available in clinical_associations_statistics.csv.', '']
    path.write_text('\n'.join(lines))

def runtime_gate(workers):
    signals = [os.environ.get(k,'').lower() for k in ['SYSTEM_NAME','CLUSTER_NAME','SGE_CLUSTER_NAME']]
    signals.append(socket.gethostname().lower())
    systems = {s for s in ['idas','argon'] if any(s in value for value in signals)}
    if len(systems)!=1:
        raise ValueError('Ambiguous system; set SYSTEM_NAME explicitly')
    if not 1 <= workers <= 4:
        raise ValueError('This analysis permits 1–4 workers plus one parent (below IDAS cap 10)')
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
        if os.environ.get(key) != '1':
            raise ValueError(f'Set {key}=1 before starting analysis')



def write_inputs(output, subjects, scales, imaging, coverage, audits, metadata):
    """Persist precisely the reviewed in-memory inputs; never select substitutes."""
    for suffix, df in [('subjects.csv',subjects),('scales.csv',scales),('imaging.csv',imaging),
                       ('coverage.csv',coverage),('score_checks.csv',audits)]:
        df.to_csv(output/(PREFIX+suffix),index=False)
    (output/(PREFIX+'methods.json')).write_text(json.dumps(metadata,indent=2,ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['all','inventory','analyze'],default='all')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--permutations',type=int,default=10000)
    parser.add_argument('--bootstraps',type=int,default=2000)
    parser.add_argument('--seed',type=int,default=20260917)
    parser.add_argument('--overwrite',action='store_true')
    args = parser.parse_args()
    runtime_gate(args.workers)
    if args.permutations < 10000 or args.bootstraps < 2000:
        parser.error('Formal outputs require >=10000 permutations and >=2000 bootstraps; use unit tests for small checks')
    output = HERE/'results'
    if not output.is_dir():
        raise FileNotFoundError(f'Required existing output directory: {output}')
    stage_outputs = {'inventory':['scales.csv','coverage.csv','score_checks.csv','subjects.csv','imaging.csv','methods.json'],
                     'analyze':['statistics.csv','summary.md','scales.csv','coverage.csv','score_checks.csv','subjects.csv','imaging.csv','methods.json'],
                     'render':['overview.png']}
    targets = OUTPUTS if args.stage=='all' else stage_outputs[args.stage]
    conflicts = [str(output/(PREFIX+s)) for s in targets if (output/(PREFIX+s)).exists()]
    if conflicts and not args.overwrite:
        raise FileExistsError('Existing exact stage outputs require --overwrite: '+', '.join(conflicts))
    if args.stage in {'all','inventory','analyze'}:
        subjects,scales,imaging,coverage,audits,metadata = load_inputs()
        metadata.update(generated_utc=datetime.now(timezone.utc).isoformat(),
                        permutations=args.permutations,bootstraps=args.bootstraps,seed=args.seed,
                        min_n=20,ci='stratified percentile subject bootstrap',
                        p='two-sided Freedman-Lane; within-group permutations for adjusted pooled models',
                        multiple_testing='none; raw two-sided permutation p values only; no FDR or other multiplicity correction',
                        madrs_policy='quarantine_recorded_totals_user_confirmed',
                        completed_statistics=False)
    if args.stage in {'all','inventory','analyze'}:
        write_inputs(output,subjects,scales,imaging,coverage,audits,metadata)
        print(f'Inventory: {len(subjects)} unique subjects, {len(scales)} fields',flush=True)
    if args.stage in {'all','analyze'}:
        results = compute(subjects,scales,imaging,permutations=args.permutations,bootstraps=args.bootstraps,
                          seed=args.seed,workers=args.workers)
        results.to_csv(output/(PREFIX+'statistics.csv'),index=False)
        metadata['completed_statistics'] = True
        (output/(PREFIX+'methods.json')).write_text(json.dumps(metadata,indent=2,ensure_ascii=False))
        write_summary(results,scales,metadata,output/(PREFIX+'summary.md'))
    if args.stage == 'render':
        metadata = json.loads((output/(PREFIX+'methods.json')).read_text())
        if not metadata.get('completed_statistics'):
            raise ValueError('Statistics stage incomplete; cannot render inference')
        subjects,scales,imaging,coverage,results = [pd.read_csv(output/(PREFIX+s)) for s in
            ['subjects.csv','scales.csv','imaging.csv','coverage.csv','statistics.csv']]
        scales = scales.fillna('')
    if args.stage in {'all','render'}:
        from .clinical_associations_report import render_report
        files = render_report(results,scales,coverage,subjects,imaging,output,metadata)
        print('Rendered: '+', '.join(map(str,files)),flush=True)


if __name__ == '__main__':
    main()

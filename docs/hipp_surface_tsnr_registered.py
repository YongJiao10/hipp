"""Complete the surface comparison with tSNR at registered anatomical vertices.

Consumes the preceding coverage test and its explicit native-to-MNI geometry.
Production inputs stay read-only. No directory selection, archives or deletion.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import nibabel as nib
import numpy as np

import hipp_surface_qc as qc

sys.path.insert(0, str(qc.ROOT / 'scripts/core_workflow'))
from lib.tsnr import runwise_tsnr, task_tsnr


def load_geometry(group, sid):
    result = {}
    for h in qc.HEMIS:
        path = qc.SAMPLES / group / sid / f'hemi-{h}_geometry.npz'
        with np.load(qc.checked(path), allow_pickle=False) as f:
            result[h] = {k: f[k] for k in ('native', 'triangles', 'area_mm2', 'mni')}
    return result


def projected_runs(paths, geom):
    result = {h: [] for h in qc.HEMIS}
    for path in paths:
        im = nib.load(qc.checked(path))
        axis = im.header.matrix.get_index_map(1)
        affine = np.asarray(axis.volume.transformation_matrix_voxel_indices_ijk_to_xyz.matrix)
        for h, name in qc.HEMIS.items():
            models = [m for m in axis.brain_models if m.brain_structure == name]
            if len(models) != 1:
                raise ValueError(f'Expected one hippocampal model: {path}/{h}')
            m = models[0]
            weights, _ = qc.interpolation(geom[h]['mni'], np.asarray(m.voxel_indices_ijk), affine)
            raw = np.asarray(im.dataobj[:, m.index_offset:m.index_offset + m.index_count], dtype=float).T
            good = np.isfinite(raw).all(1) & (raw.std(1, ddof=1) > 0)
            y = weights @ np.where(good[:, None], raw, 0)
            scale = weights @ good.astype(float)
            np.divide(y, scale[:, None], out=y, where=scale[:, None] > 0)
            result[h].append(y)
    return result


def stanford_baseline(sid, geom):
    """Sample one explicit fMRIPrep baseline volume once for both variants."""
    out = qc.SAMPLES / 'Stanford' / sid
    points = np.vstack([geom[h]['mni'] for h in qc.HEMIS])
    triangles = np.vstack([geom[h]['triangles'] + i*512 for i, h in enumerate(qc.HEMIS)])
    mid = out / 'bilateral_space-MNI_den-512_midthickness.surf.gii'
    nib.save(nib.gifti.GiftiImage(darrays=[
        nib.gifti.GiftiDataArray(points.astype('float32'), intent='NIFTI_INTENT_POINTSET'),
        nib.gifti.GiftiDataArray(triangles.astype('int32'), intent='NIFTI_INTENT_TRIANGLE')]), mid)
    volume = qc.J / ('fmriprep_processed_CausalConnectome/rest_fmriprep/fmriprep/'
                     f'sub-{sid}/func/sub-{sid}_task-rest_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz')
    dest = out / 'bilateral_registered_preproc_bold.func.gii'
    qc.run([qc.WB, '-volume-to-surface-mapping', qc.checked(volume), mid, dest, '-trilinear'])
    y = np.column_stack([a.data for a in nib.load(qc.checked(dest)).darrays])
    if y.ndim != 2 or y.shape[0] != 1024:
        raise ValueError(f'Expected bilateral 1024-by-time samples: {dest}: {y.shape}')
    return {h: y[i*512:(i+1)*512] for i, h in enumerate(qc.HEMIS)}


def stanford_frames(sid, variant, n_source, n_analysis):
    branch = 'xcpd_zr/xcpd_vol_22P' if variant == '22' else 'xcp_d_36P_Head05'
    q = qc.J / f'fmriprep_processed_CausalConnectome/rest_fmriprep/{branch}/sub-{sid}/func'
    stem = f'sub-{sid}_task-rest'
    rows = qc.read_tsv(q / f'{stem}_space-MNI152NLin6Asym_desc-linc_qc.tsv')
    if len(rows) != 1:
        raise ValueError(f'Expected one LINC row: {q}')
    dummy = int(rows[0]['num_dummy_volumes'])
    mask = qc.read_tsv(q / f'{stem}_outliers.tsv')
    keep = np.array([i + dummy for i, row in enumerate(mask)
                     if float(row['framewise_displacement']) == 0], dtype=int)
    if len(mask) + dummy != n_source or len(keep) != n_analysis or len(keep) != int(rows[0]['num_retained_volumes']):
        raise ValueError(f'Stanford original/retained frame mismatch: {sid}/{variant}P')
    return keep


def evaluate(spec, sid, old, baseline=None, baseline_error=''):
    row = dict(old)
    row['T25_current_selector_reference'] = old['T25']
    row['Basic_current_selector_reference'] = old['Basic']
    row['T25_current_selector_source'] = old['T25_source']
    for h in qc.HEMIS:
        for bound in ('lower', 'upper'):
            key = f'T25_invalid_fraction_{bound}_{h}'
            if key in row:
                row[f'T25_current_selector_invalid_fraction_{bound}_{h}'] = row.pop(key)
    if spec['kind'] in {'banda', 'pdc'}:
        return row
    row.pop('T25_source')
    notes = [old['errors']] if old['errors'] else []
    anatomy = qc.conjunction(*(None if row.get(f'dice_{h}', '') == '' else
                              bool(float(row[f'dice_{h}']) >= .7) for h in qc.HEMIS))
    basic, t25 = qc.conjunction(anatomy, None), qc.conjunction(anatomy, None)
    try:
        geom = load_geometry(spec['group'], sid)
        paths = qc.paths_for(spec, sid)
        runs = projected_runs(paths, geom)
        valids = {h: np.logical_and.reduce([np.isfinite(y).all(1) & (y.std(1, ddof=1) > 0)
                                          for y in runs[h]]) for h in qc.HEMIS}
        basic = qc.conjunction(anatomy, all((~v).mean() < .20 for v in valids.values()))
        t25_states = []
        arrays = {}
        for h in qc.HEMIS:
            blocks = runs[h]
            valid = valids[h]
            if spec['kind'] == 'hcp':
                # The delivered HCP7T signal has no temporal mean baseline.
                # Preserve the declared 10000/SD proxy, at corrected vertices.
                if len(blocks) != 1 or blocks[0].shape[1] != 3600:
                    raise ValueError(f'HCP7T expected declared 3600-frame concatenation: {sid}')
                score = np.full((1, 512), np.nan)
                score[0, valid] = 10000 / blocks[0][valid].std(1, ddof=1)
                row['T25_source'] = 'registered_CIFTI_surface_HCP7T_legacy_10000_over_concat_SD_proxy'
            elif spec['kind'] == 'task':
                declaration = qc.read_json(qc.HCP / f'kong2022/manifests/task-{spec["task"]}/sub-{sid}.json')
                bolds = [r['bold']['path'] for r in declaration['runs']]
                score, _ = task_tsnr(np.concatenate(blocks, axis=1), [b.shape[1] for b in blocks],
                                     bolds, task=spec['task'], threshold=25.)
                row['T25_source'] = 'registered_CIFTI_surface_task_OLS_DCT128_residual_SD_ddof1'
            else:
                if baseline is None:
                    raise ValueError(f'Explicit registered Stanford baseline missing: {baseline_error}')
                keep = stanford_frames(sid, spec['variant'], baseline[h].shape[1], blocks[0].shape[1])
                score, _ = runwise_tsnr([baseline[h]], frames=[keep], threshold=25.)
                row['T25_source'] = 'registered_volume_surface_matched_preproc_baseline_over_detrended_SD'
            tmask = valid & np.all(np.isfinite(score) & (score >= 25), axis=0)
            t25_states.append(bool((~tmask).mean() < .20))
            row[f'V_invalid_fraction_{h}'] = float((~valid).mean())
            row[f'T25_invalid_fraction_{h}'] = float((~tmask).mean())
            row[f'median_run_tsnr_{h}'] = ','.join(f'{np.nanmedian(s):.6g}' for s in score)
            arrays[f'{h}_tsnr'] = score
            arrays[f'{h}_T25_mask'] = tmask
            arrays[f'{h}_valid_mask'] = valid
        t25 = qc.conjunction(basic, qc.conjunction(*t25_states))
        out = qc.SAMPLES / spec['name'].replace(' ', '_') / sid
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / 'registered_tsnr_maps.npz', **arrays)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        notes.append(str(exc))
        t25 = qc.conjunction(basic, None)
    c20 = qc.state(row['coverage_geometry_only'])
    nearest = (None if row.get('nearest_unsupported_mm2_L', '') == '' else
               all(float(row[f'nearest_unsupported_mm2_{h}']) < 20 for h in qc.HEMIS))
    coverage = qc.conjunction(basic, c20)
    row.update(Basic=qc.status(basic), T25=qc.status(t25), Coverage20=qc.status(coverage),
               T25_and_Coverage20=qc.status(qc.conjunction(t25, coverage)),
               Nearest20_diagnostic=qc.status(qc.conjunction(basic, nearest)),
               registered_tsnr_errors=' | '.join(notes))
    return row


def job(specs, sid, rows):
    baseline, error = None, ''
    if specs[0]['kind'] == 'stanford':
        try:
            baseline = stanford_baseline(sid, load_geometry('Stanford', sid))
        except (OSError, ValueError, RuntimeError) as exc:
            error = str(exc)
    return [evaluate(spec, sid, rows[(spec['name'], sid)], baseline, error) for spec in specs]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--workers', type=int, choices=(1, 2), default=2)
    args = p.parse_args()
    host = os.environ.get('SYSTEM_NAME') or os.uname().nodename
    if 'idas' not in host.lower() and 'argon' not in host.lower():
        raise ValueError(f'Explicit SYSTEM_NAME=idas or argon required: {host}')
    suffix = '_pilot' if args.pilot else ''
    specs, _ = qc.configurations()
    source_rows = qc.read_tsv(qc.ROOT / f'docs/hipp_surface_qc{suffix}_subjects.tsv')
    rows = {(r['dataset'], r['subject']): r for r in source_rows}
    jobs = defaultdict(list)
    for spec in specs:
        for sid in spec['ids']:
            if args.pilot and (spec['name'], sid) not in rows:
                continue
            jobs[(spec['group'], sid)].append(spec)
    result = []
    start = time.monotonic()
    print(f'REGISTERED START {len(jobs)} subject groups, {args.workers} workers, numerical/ITK threads=1', flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(job, ss, sid, rows): (group, sid) for (group, sid), ss in jobs.items()}
        for i, future in enumerate(as_completed(futures), 1):
            result.extend(future.result())
            if i % 20 == 0 or args.pilot:
                print(f'REGISTERED {i}/{len(jobs)} {futures[future]} elapsed={time.monotonic()-start:.1f}s', flush=True)
    order = {s['name']: i for i, s in enumerate(specs)}
    result.sort(key=lambda r: (order[r['dataset']], r['subject']))
    qc.write_tsv(qc.ROOT / f'docs/hipp_surface_qc_registered{suffix}_subjects.tsv', result,
                 ['dataset', 'subject', 'participant', *qc.SCHEMES, 'T25_current_selector_reference'])
    summary = []
    for spec in specs:
        selected = [r for r in result if r['dataset'] == spec['name']]
        summary += qc.summarize(selected, spec['name'], 'visit' if spec['kind'] == 'pdc' else 'person')
        if spec['kind'] == 'pdc':
            people = defaultdict(list)
            for r in selected:
                people[r['participant']].append(r)
            aggregated = []
            for visits in people.values():
                pr = {}
                for scheme in qc.SCHEMES:
                    states = [r[scheme] for r in visits]
                    pr[scheme] = 'pass' if 'pass' in states else 'unassessable' if 'unassessable' in states else 'fail'
                aggregated.append(pr)
            summary += qc.summarize(aggregated, 'PDC any passing visit', 'person')
    tasks = [{r['subject']: r for r in result if r['dataset'] == name} for name in ('HCP3T EMOTION', 'HCP3T WM')]
    both = [{s: qc.status(qc.conjunction(*(qc.state(t[sid][s]) for t in tasks))) for s in qc.SCHEMES} for sid in tasks[0]]
    summary += qc.summarize(both, 'HCP3T both tasks', 'person')
    qc.write_tsv(qc.ROOT / f'docs/hipp_surface_qc_registered{suffix}_summary.tsv', summary,
                 ['dataset', 'unit', 'scheme', 'n_total', 'n_pass', 'n_fail', 'n_unassessable',
                  'n_evaluable', 'pass_percent_total', 'pass_percent_evaluable'])
    for r in summary:
        print(json.dumps(r), flush=True)
    print(f'REGISTERED COMPLETE {len(result)} entries elapsed={time.monotonic()-start:.1f}s', flush=True)


if __name__ == '__main__':
    main()

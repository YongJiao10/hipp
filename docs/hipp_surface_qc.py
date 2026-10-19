"""Five-dataset hippocampal surface QC comparison; production inputs are read-only.

See hipp_surface_qc.md for definitions and coordinate conventions. Run with
explicit py314 and numerical/ITK threads=1. No archive access or deletion.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
J = Path('/home/yojiao/LSS/jianglab/3_Data_Working')
HCP = ROOT / 'outputs_hcp/inputs/hcp'
SF = ROOT / 'outputs_stanford/inputs/stanford'
SAMPLES = ROOT / 'docs/hipp_surface_qc_samples'
WB = Path('/home/yojiao/FASTANS/workbench/bin_linux64/wb_command')
ANTS = ROOT / '.cache/conda_envs/hippo2/bin/antsApplyTransformsToPoints'
HEMIS = {'L': 'CIFTI_STRUCTURE_HIPPOCAMPUS_LEFT',
         'R': 'CIFTI_STRUCTURE_HIPPOCAMPUS_RIGHT'}
SCHEMES = ('Basic', 'T25', 'Coverage20', 'T25_and_Coverage20', 'Nearest20_diagnostic')


def checked(path):
    path = Path(path)
    if any(p.lower() in {'archive', '_archive', 'arch'} for p in path.resolve().parts):
        raise ValueError(f'Archive access is not authorized: {path}')
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return path


def read_json(path):
    return json.loads(checked(path).read_text())


def read_tsv(path):
    with checked(path).open() as f:
        return list(csv.DictReader(f, delimiter='\t'))


def roster(path):
    ids = [s.strip().removeprefix('sub-') for s in checked(path).read_text().splitlines()
           if s.strip() and not s.startswith('#')]
    if len(ids) != len(set(ids)):
        raise ValueError(f'Duplicate roster IDs: {path}')
    return ids


def conjunction(*xs):
    return False if any(x is False for x in xs) else None if any(x is None for x in xs) else True


def status(x):
    return 'unassessable' if x is None else 'pass' if x else 'fail'


def state(s):
    return {'pass': True, 'fail': False, 'unassessable': None}[s]


def local(path):
    prefix = '/Shared/jianglab/'
    if not path.startswith(prefix):
        raise ValueError(f'Expected declared Argon Jianglab prefix: {path}')
    return Path('/home/yojiao/LSS/jianglab') / path[len(prefix):]


def run(args):
    result = subprocess.run([str(x) for x in args], capture_output=True, text=True,
                            env={**os.environ, 'OMP_NUM_THREADS': '1',
                                 'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS': '1'})
    if result.returncode:
        raise RuntimeError(f'Command failed ({result.returncode}): {args}\n{result.stderr[-2000:]}')


def surface(path):
    im = nib.load(checked(path))
    v = np.asarray(im.agg_data('pointset'), dtype=float)
    t = np.asarray(im.agg_data('triangle'), dtype=int)
    if v.shape != (512, 3) or t.ndim != 2 or t.shape[1] != 3 or not np.isfinite(v).all():
        raise ValueError(f'Expected folded den-512 mesh: {path}')
    a = np.linalg.norm(np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]]), axis=1) / 2
    va = np.bincount(t.ravel(), weights=np.repeat(a / 3, 3), minlength=len(v))
    return v, t, va


def inverse_points(native, world_warp):
    """Invert the provided MNI->native point field only at required vertices.

    Workbench converts FNIRT to NIfTI RAS displacement convention. Newton's
    method solves MNI + d(MNI) = native; a failed residual makes QC unknown.
    This avoids inventing an inverse image field or applying a forward image
    warp in the wrong direction. Both solved points and residuals are saved.
    """
    im = nib.load(checked(world_warp))
    data = np.asarray(im.dataobj, dtype=float)
    inv = np.linalg.inv(im.affine)

    def disp(x):
        ijk = nib.affines.apply_affine(inv, x)
        return np.column_stack([map_coordinates(data[..., j], ijk.T, order=1,
                                                mode='constant', cval=np.nan) for j in range(3)])

    x = native.copy()
    for _ in range(20):
        residual = x + disp(x) - native
        norms = np.linalg.norm(residual, axis=1)
        if not np.isfinite(norms).all():
            raise ValueError('Native-to-MNI inverse solve left the declared warp grid')
        if norms.max() < .001:
            return x, norms
        jac = np.stack([disp(x + np.eye(3)[j] * .5) - disp(x - np.eye(3)[j] * .5)
                        for j in range(3)], axis=-1) + np.eye(3)
        x -= np.linalg.solve(jac, residual[..., None])[..., 0]
    raise ValueError(f'Native-to-MNI inverse did not converge: max residual {norms.max()} mm')


def geometry(group, sid, source=None):
    """Exact native surface declarations, with explicit transformations to MNI."""
    dataset = 'hcp' if group == 'HCP' else 'stanford' if group == 'Stanford' else group.lower()
    base = ROOT / f'outputs_{dataset}/inputs/{dataset}'
    out = SAMPLES / group / sid
    out.mkdir(parents=True, exist_ok=True)
    paths = {h: base / f'hippunfold/dense_corobl_batch/sub-{sid}/hippunfold/sub-{sid}/surf/'
             f'sub-{sid}_hemi-{h}_space-T1w_den-512_label-hipp_midthickness.surf.gii' for h in HEMIS}
    meshes = {h: surface(p) for h, p in paths.items()}
    combined = np.vstack([meshes[h][0] for h in HEMIS])
    residual = np.zeros(len(combined))
    if group == 'HCP':
        src = J / f'HCP_7T_Hippocampus/subjects/{sid}/source'
        world = out / 'MNI_to_native_world.nii.gz'
        run([WB, '-convert-warpfield', '-from-fnirt', checked(src / 'acpc_dc2standard.nii.gz'),
             checked(src / 'T1w_acpc_dc_restore.nii.gz'), '-to-world', world])
        points, residual = inverse_points(combined, world)
        transform = str(src / 'acpc_dc2standard.nii.gz') + ';Workbench world conversion;pointwise inverse'
    elif group == 'Stanford':
        transform = J / ('fmriprep_processed_CausalConnectome/rest_fmriprep/fmriprep/'
                         f'sub-{sid}/anat/sub-{sid}_from-MNI152NLin6Asym_to-T1w_mode-image_xfm.h5')
        lps = combined.copy()
        lps[:, :2] *= -1
        src, dst = out / 'native_lps.csv', out / 'MNI_lps.csv'
        np.savetxt(src, lps, delimiter=',', header='x,y,z', comments='')
        run([ANTS, '-d', '3', '-i', src, '-o', dst, '-t', checked(transform)])
        rows = np.genfromtxt(dst, delimiter=',', names=True)
        points = np.column_stack([rows[k] for k in ('x', 'y', 'z')])
        points[:, :2] *= -1
    else:
        if source is None:
            raise ValueError(f'Explicit structural source required: {group}/{sid}')
        xf = source / 'MNINonLinear/xfms'
        mapped = []
        for h in HEMIS:
            dst = out / f'sub-{sid}_hemi-{h}_space-MNI_den-512_midthickness.surf.gii'
            run([WB, '-surface-apply-warpfield', paths[h], checked(xf / 'standard2acpc_dc.nii.gz'),
                 dst, '-fnirt', checked(xf / 'acpc_dc2standard.nii.gz')])
            mapped.append(surface(dst)[0])
        points = np.vstack(mapped)
        transform = str(xf / 'standard2acpc_dc.nii.gz') + ';FNIRT pair'
    if points.shape != combined.shape or not np.isfinite(points).all():
        raise ValueError(f'Invalid transformed points: {group}/{sid}')
    result = {}
    for i, h in enumerate(HEMIS):
        v, t, areas = meshes[h]
        p = points[i * 512:(i + 1) * 512]
        result[h] = dict(native=v, triangles=t, areas=areas, mni=p, path=str(paths[h]),
                         transform=str(transform), max_inverse_residual=float(residual.max()))
        np.savez_compressed(out / f'hemi-{h}_geometry.npz', native=v, triangles=t, area_mm2=areas,
                            mni=p, inverse_residual_mm=residual[i * 512:(i + 1) * 512])
    return result


def geometry_job(group, sid, source):
    try:
        return group, sid, geometry(group, sid, source), ''
    except (OSError, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        return group, sid, None, str(exc)


def interpolation(points, voxels, affine):
    """Trilinear vertex-by-modeled-voxel weights; unmodeled neighbors have zero weight."""
    p = nib.affines.apply_affine(np.linalg.inv(affine), points)
    lo = np.floor(p).astype(int)
    fraction = p - lo
    index = {tuple(v): i for i, v in enumerate(voxels)}
    rr, cc, ww = [], [], []
    for offset in np.ndindex(2, 2, 2):
        off = np.array(offset)
        weights = np.prod(np.where(off, fraction, 1 - fraction), axis=1)
        for r, (ijk, w) in enumerate(zip(lo + off, weights)):
            c = index.get(tuple(ijk))
            if c is not None and w > 0:
                rr.append(r); cc.append(c); ww.append(w)
    matrix = csr_matrix((ww, (rr, cc)), shape=(len(points), len(voxels)))
    nearest = np.array([index.get(tuple(v), -1) for v in np.floor(p + .5).astype(int)])
    return matrix, nearest


def functional_metrics(paths, geom, calculate_tsnr):
    models, masks, weights, nearest, signals = {}, {}, {}, {}, defaultdict(list)
    frames = []
    for path in paths:
        im = nib.load(checked(path))
        axis = im.header.matrix.get_index_map(1)
        frames.append(im.shape[0])
        affine = np.asarray(axis.volume.transformation_matrix_voxel_indices_ijk_to_xyz.matrix)
        for h, name in HEMIS.items():
            matches = [m for m in axis.brain_models if m.brain_structure == name]
            if len(matches) != 1 or matches[0].model_type != 'CIFTI_MODEL_TYPE_VOXELS':
                raise ValueError(f'Expected one hippocampal voxel model: {path}/{h}')
            model = matches[0]
            ijk = np.asarray(model.voxel_indices_ijk)
            if h in models:
                if not np.array_equal(models[h][0], ijk) or not np.array_equal(models[h][1], affine):
                    raise ValueError(f'Runwise hippocampal voxel/affine mismatch: {path}/{h}')
            else:
                models[h] = (ijk, affine)
                weights[h], nearest[h] = interpolation(geom[h]['mni'], ijk, affine)
                masks[h] = np.ones(len(ijk), dtype=bool)
            y = np.asarray(im.dataobj[:, model.index_offset:model.index_offset + model.index_count],
                           dtype=np.float64).T
            good = np.isfinite(y).all(1) & (y.std(1, ddof=1) > 0)
            masks[h] &= good
            if calculate_tsnr:
                projected = weights[h] @ np.where(good[:, None], y, 0)
                scale = weights[h] @ good.astype(float)
                np.divide(projected, scale[:, None], out=projected, where=scale[:, None] > 0)
                signals[h].append(projected)
    records, arrays = {}, {}
    for h in HEMIS:
        area = geom[h]['areas']
        total_weight = weights[h] @ masks[h].astype(float)
        supported = total_weight > 0
        nn = nearest[h]
        nearest_supported = np.zeros(512, dtype=bool)
        nearest_supported[nn >= 0] = masks[h][nn[nn >= 0]]
        records.update({f'area_mm2_{h}': float(area.sum()),
                        f'unsupported_mm2_{h}': float(area[~supported].sum()),
                        f'unsupported_vertices_{h}': int((~supported).sum()),
                        f'nearest_unsupported_mm2_{h}': float(area[~nearest_supported].sum()),
                        f'functional_voxels_{h}': len(masks[h]),
                        f'valid_functional_voxels_{h}': int(masks[h].sum())})
        arrays[f'{h}_GM_weight'] = total_weight
        arrays[f'{h}_supported'] = supported
        arrays[f'{h}_nearest_supported'] = nearest_supported
        if calculate_tsnr:
            values, valids = [], []
            for y in signals[h]:
                sd = y.std(1, ddof=1)
                valid = np.isfinite(y).all(1) & (sd > 0)
                value = np.full(512, np.nan)
                value[valid] = y[valid].mean(1) / sd[valid]
                values.append(value)
                valids.append(valid)
            values, valids = np.asarray(values), np.asarray(valids)
            t25 = np.all(np.isfinite(values) & (values >= 25), axis=0)
            valid = valids.all(0)
            records[f'T25_invalid_fraction_{h}'] = float((~t25).mean())
            records[f'V_invalid_fraction_{h}'] = float((~valid).mean())
            records[f'median_run_tsnr_{h}'] = ','.join(f'{np.nanmedian(v):.6g}' for v in values)
            arrays[f'{h}_tsnr'] = values
            arrays[f'{h}_T25_mask'] = t25
            arrays[f'{h}_valid_mask'] = valid
    records['frames'] = ','.join(map(str, frames))
    records['n_runs'] = len(paths)
    return records, arrays


def configurations():
    previous = {(r['dataset'], r['subject']): r for r in read_tsv(ROOT / 'docs/tsnr_qc_comparison_subjects.tsv')
                if r['dataset'] in {'HCP7T rest', 'HCP3T EMOTION', 'HCP3T WM', 'Stanford 22P', 'Stanford 36P'}}
    specs = []
    ids = roster(ROOT / 'manifests/hcp_7t_hippocampus_struct_complete_166.txt')
    specs.append(dict(name='HCP7T rest', group='HCP', ids=ids, kind='hcp'))
    for task in ('emotion', 'wm'):
        ids = roster(HCP / f'kong2022/shared_inputs/task-{task}/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_144.txt')
        specs.append(dict(name=f'HCP3T {task.upper()}', group='HCP', ids=ids, kind='task', task=task))
    for variant in ('22', '36'):
        specs.append(dict(name=f'Stanford {variant}P', group='Stanford',
                          ids=roster(SF / 'kong2022/manifests/stanford_all_subjects.txt'),
                          kind='stanford', variant=variant))
    for ds in ('banda', 'pdc'):
        base = ROOT / f'outputs_{ds}/inputs/{ds}'
        entries = read_json(base / f'kong2022/manifests/{ds}_cortex_inputs.json')['subjects']
        entries = {r['subject'].removeprefix('sub-'): r for r in entries}
        ids = roster(base / ('manifests/banda_all_subjects.txt' if ds == 'banda' else 'manifests/pdc_all_sessions.txt'))
        people = {sid: sid for sid in ids}
        sources = {sid: J / f'Depression/BANDA/imaging/{sid}_MR' for sid in ids}
        if ds == 'pdc':
            mapping = read_tsv(base / 'manifests/session_mapping.tsv')
            people = {r['bids_subject_id']: r['base_subject_id'] for r in mapping}
            sources = {r['bids_subject_id']: J / f'Depression/PDC/imaging/{r["original_session_dir"]}' for r in mapping}
        specs.append(dict(name=ds.upper(), group=ds.upper(), kind=ds, base=base,
                          ids=ids, people=people, sources=sources, entries=entries))
    return specs, previous


def paths_for(spec, sid):
    if spec['kind'] == 'hcp':
        return [ROOT / f'data/hippunfold_input/sub-{sid}/func/sub-{sid}_task-rest_run-concat.dtseries.nii']
    if spec['kind'] == 'task':
        data = read_json(HCP / f'kong2022/manifests/task-{spec["task"]}/sub-{sid}.json')
        return [Path(r['dtseries']['path']) for r in data['runs']]
    if spec['kind'] == 'stanford':
        return [J / ('fmriprep_processed_CausalConnectome/rest_fmriprep/'
                     f'DataOrg_PFM_{spec["variant"]}P/sub-{sid}/func/'
                     f'sub-{sid}_task-rest_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii')]
    if sid not in spec['entries']:
        raise ValueError(f'No independent rest runs declared: {spec["name"]}/{sid}')
    return [local(r['dtseries']) for r in spec['entries'][sid]['runs']]


def evaluate(spec, sid, geom_result, previous):
    row = dict(dataset=spec['name'], subject=sid, participant=spec.get('people', {}).get(sid, sid))
    notes = []
    basic, t25 = None, None
    new_signal = spec['kind'] in {'banda', 'pdc'}
    if not new_signal:
        old = previous[(spec['name'], sid)]
        basic, t25 = state(old['V']), state(old['T25'])
        row.update({k: old[k] for k in ('dice_L', 'dice_R', 'T25_source',
                    'T25_invalid_fraction_lower_L', 'T25_invalid_fraction_lower_R',
                    'T25_invalid_fraction_upper_L', 'T25_invalid_fraction_upper_R')})
    else:
        anatomy_states = []
        for h in HEMIS:
            p = spec['base'] / f'hippunfold/dense_corobl_batch/sub-{sid}/hippunfold/sub-{sid}/qc/sub-{sid}_hemi-{h}_desc-unetf3d_dice.tsv'
            try:
                dice = np.loadtxt(checked(p))
                if dice.size != 1 or not np.isfinite(dice).all():
                    raise ValueError(f'Expected one finite Dice score: {p}')
                row[f'dice_{h}'] = float(dice.item())
                anatomy_states.append(bool(dice.item() >= .7))
            except (OSError, ValueError) as exc:
                anatomy_states.append(None)
                notes.append(str(exc))
        basic = conjunction(*anatomy_states)
        row['T25_source'] = 'new_native_T1w_to_MNI_CIFTI_hipp_volume_trilinear_surface_mean_over_sample_SD'
    geom, error = geom_result
    c20, nearest20 = None, None
    if geom is None:
        notes.append(error)
        if new_signal:
            basic = conjunction(basic, None)
            t25 = conjunction(basic, None)
    else:
        row['geometry_available'] = 'yes'
        row['native_surface_L'] = geom['L']['path']
        row['native_surface_R'] = geom['R']['path']
        row['native_to_MNI'] = geom['L']['transform']
        row['max_inverse_residual_mm'] = geom['L']['max_inverse_residual']
        try:
            paths = paths_for(spec, sid)
            row['cifti_inputs'] = ';'.join(map(str, paths))
            metrics, arrays = functional_metrics(paths, geom, new_signal)
            row.update(metrics)
            row['unsupported_bilateral_mm2'] = sum(metrics[f'unsupported_mm2_{h}'] for h in HEMIS)
            c20 = all(metrics[f'unsupported_mm2_{h}'] < 20 for h in HEMIS)
            nearest20 = all(metrics[f'nearest_unsupported_mm2_{h}'] < 20 for h in HEMIS)
            out = SAMPLES / spec['name'].replace(' ', '_') / sid
            out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out / 'surface_qc_maps.npz', **arrays)
            if new_signal:
                basic = conjunction(basic, all(metrics[f'V_invalid_fraction_{h}'] < .20 for h in HEMIS))
                t25 = conjunction(basic, all(metrics[f'T25_invalid_fraction_{h}'] < .20 for h in HEMIS))
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            notes.append(str(exc))
            if new_signal:
                basic = conjunction(basic, None)
                t25 = conjunction(basic, None)
    coverage = conjunction(basic, c20)
    row.update(Basic=status(basic), T25=status(t25), Coverage20=status(coverage),
               T25_and_Coverage20=status(conjunction(t25, coverage)),
               Nearest20_diagnostic=status(conjunction(basic, nearest20)),
               coverage_geometry_only=status(c20), errors=' | '.join(notes))
    return row


def write_tsv(path, rows, first):
    columns = first + sorted(set().union(*(r.keys() for r in rows)) - set(first))
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, dataset, unit):
    result = []
    for scheme in SCHEMES:
        counts = Counter(r[scheme] for r in rows)
        n = len(rows)
        p, f, u = (counts[s] for s in ('pass', 'fail', 'unassessable'))
        if p + f + u != n:
            raise ValueError(f'Invalid count partition: {dataset}/{scheme}')
        result.append(dict(dataset=dataset, unit=unit, scheme=scheme, n_total=n, n_pass=p,
                           n_fail=f, n_unassessable=u, n_evaluable=n-u,
                           pass_percent_total=round(100*p/n, 3),
                           pass_percent_evaluable=round(100*p/(n-u), 3) if n > u else ''))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', action='store_true', help='one exact representative per branch')
    args = parser.parse_args()
    host = os.environ.get('SYSTEM_NAME') or os.uname().nodename
    if 'idas' not in host.lower() and 'argon' not in host.lower():
        raise ValueError(f'Explicit SYSTEM_NAME=idas or argon required: {host}')
    specs, previous = configurations()
    if args.pilot:
        selected = {'HCP': '100610', 'Stanford': '1017', 'BANDA': 'BANDA001', 'PDC': 'e0010s01'}
        for spec in specs:
            spec['ids'] = [selected[spec['group']]]
    start = time.monotonic()
    geom_jobs = {}
    for spec in specs:
        for sid in spec['ids']:
            geom_jobs[(spec['group'], sid)] = spec.get('sources', {}).get(sid)
    geoms = {}
    print(f'GEOMETRY {len(geom_jobs)} unique meshes; two workers; numerical/ITK threads=1; {host}', flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(geometry_job, group, sid, src) for (group, sid), src in geom_jobs.items()]
        for i, future in enumerate(as_completed(futures), 1):
            group, sid, geom, error = future.result()
            geoms[(group, sid)] = (geom, error)
            if i % 25 == 0 or error or args.pilot:
                print(f'GEOM {i}/{len(futures)} {group}/{sid} {error or "ok"} elapsed={time.monotonic()-start:.1f}s', flush=True)
    records = []
    for spec in specs:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(evaluate, spec, sid, geoms[(spec['group'], sid)], previous) for sid in spec['ids']]
            for i, future in enumerate(as_completed(futures), 1):
                row = future.result()
                records.append(row)
                if i % 25 == 0 or args.pilot:
                    print(f'SIGNAL {spec["name"]} {i}/{len(futures)} elapsed={time.monotonic()-start:.1f}s', flush=True)
    order = {s['name']: i for i, s in enumerate(specs)}
    records.sort(key=lambda r: (order[r['dataset']], r['subject']))
    suffix = '_pilot' if args.pilot else ''
    write_tsv(ROOT / f'docs/hipp_surface_qc{suffix}_subjects.tsv', records,
              ['dataset', 'subject', 'participant', *SCHEMES])
    summary = []
    for spec in specs:
        rows = [r for r in records if r['dataset'] == spec['name']]
        summary += summarize(rows, spec['name'], 'visit' if spec['kind'] == 'pdc' else 'person')
        if spec['kind'] == 'pdc':
            groups = defaultdict(list)
            for r in rows:
                groups[r['participant']].append(r)
            people = []
            for sid, visits in groups.items():
                pr = {'participant': sid}
                for scheme in SCHEMES:
                    states = [r[scheme] for r in visits]
                    pr[scheme] = ('pass' if 'pass' in states else 'unassessable'
                                  if 'unassessable' in states else 'fail')
                people.append(pr)
            summary += summarize(people, 'PDC any passing visit', 'person')
    emotion = {r['subject']: r for r in records if r['dataset'] == 'HCP3T EMOTION'}
    wm = {r['subject']: r for r in records if r['dataset'] == 'HCP3T WM'}
    if set(emotion) != set(wm):
        raise ValueError('Task comparison requires the same subject roster for EMOTION and WM')
    both = [{s: status(conjunction(state(emotion[sid][s]), state(wm[sid][s]))) for s in SCHEMES} for sid in emotion]
    summary += summarize(both, 'HCP3T both tasks', 'person')
    write_tsv(ROOT / f'docs/hipp_surface_qc{suffix}_summary.tsv', summary,
              ['dataset', 'unit', 'scheme', 'n_total', 'n_pass', 'n_fail', 'n_unassessable',
               'n_evaluable', 'pass_percent_total', 'pass_percent_evaluable'])
    for r in summary:
        print(json.dumps(r), flush=True)
    print(f'COMPLETE {len(records)} entries elapsed={time.monotonic()-start:.1f}s', flush=True)


if __name__ == '__main__':
    main()

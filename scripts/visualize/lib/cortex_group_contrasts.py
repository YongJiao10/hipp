"""Subject-level 400-ROI exports and independent Iowa-HC contrasts.

This is an explicit preparation stage, never a replacement for canonical mean
profiles. Missing inputs fail closed; empty physiological support stays missing.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
CORE_WORKFLOW_DIR = ROOT / 'scripts/core_workflow'
if str(CORE_WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_WORKFLOW_DIR))

from lib.output_layout import resolve_routea_ctxreg_group_atlas_root
from lib.group_atlas_lineage import active_path
from multidataset_cohorts import GRID_VARIANTS, get_comparison_visual_columns

FIGURE_DIR = ROOT / 'scripts/visualize/multidataset_hipp_cortex_gradient_wta'
SCHEMA = 'subject_roi_welch_hc_v4_uncorrected'
T_CMAP = 'PiYG_r'


def bilateral_profiles(left, left_ids, right, right_ids):
    if (len(set(left_ids)) != len(left_ids) or len(set(right_ids)) != len(right_ids)
            or set(left_ids) != set(right_ids)):
        raise ValueError('Bilateral subject identities must be unique and identical')
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or left.shape[0] != len(left_ids):
        raise ValueError('Bilateral subject profile shape mismatch')
    index = {s: i for i, s in enumerate(right_ids)}
    return (left + right[[index[s] for s in left_ids]]) / 2.0


def compare_profiles(group, reference):
    group, reference = np.asarray(group, dtype=float), np.asarray(reference, dtype=float)
    if (group.ndim != 3 or reference.ndim != 3 or group.shape[1:] != reference.shape[1:]
            or group.shape[-1] != 400 or min(group.shape[0], reference.shape[0]) < 2):
        raise ValueError('Expected independent subject x panel x 400 ROI arrays, each N >= 2')
    if np.isinf(group).any() or np.isinf(reference).any():
        raise ValueError('Infinite subject FC is invalid')
    ng, nr = np.isfinite(group).sum(axis=0), np.isfinite(reference).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        warnings.simplefilter('ignore', stats._axis_nan_policy.SmallSampleWarning)
        test = stats.ttest_ind(group, reference, axis=0, equal_var=False, nan_policy='omit')
        variance = np.nanvar(group, axis=0, ddof=1) / np.maximum(ng, 1)
        variance += np.nanvar(reference, axis=0, ddof=1) / np.maximum(nr, 1)
    valid = (ng >= 2) & (nr >= 2) & (variance > 0) & np.isfinite(test.statistic) & np.isfinite(test.pvalue)
    t = np.where(valid, test.statistic, np.nan)
    p = np.where(valid, test.pvalue, np.nan)
    significant = valid & (p < 0.05)
    return dict(t=t, p=p, n_group=ng, n_reference=nr,
                testable=valid, significant=significant)


def fingerprint(path):
    path = active_path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing canonical contrast input: {path}')
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'path': str(path), 'sha256': digest.hexdigest()}


def renderer_module():
    path = FIGURE_DIR / 'render_multidataset_hipp_cortex_gradient_wta.py'
    spec = importlib.util.spec_from_file_location('hipp_cortex_renderer', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _signal_branch_for_grid(cohort, grid):
    branch = str(getattr(grid, "signal_branch", "standard"))
    if branch not in ("standard", "ctxreg"):
        raise ValueError(f"Unknown signal branch {branch!r}")
    recorded = cohort.get("signal_branch")
    if recorded is not None and str(recorded) != branch:
        raise ValueError(
            "Cohort signal branch disagrees with requested grid: "
            f"cohort={cohort.get('id')!r} recorded={recorded!r} requested={branch!r}"
        )
    return branch


def wta_profile_cache_path(cohort, grid):
    """Return the exact branch-qualified WTA profile cache path."""
    branch = _signal_branch_for_grid(cohort, grid)
    prefix = f"{cohort['method']}_" if cohort.get("method") else ""
    atlas = "7net_" if cohort.get("legacy_7net_namespace", False) else ""
    suffix = "" if branch == "standard" else f"_{branch}"
    return (
        FIGURE_DIR
        / "panels"
        / f"{prefix}{cohort['id']}_{grid.variant}_{atlas}"
        f"wta_pseudocluster_bilat_{grid.smoothing}{suffix}.npy"
    )


def figure_cohorts(renderer, grid, version="canonical"):
    del version
    kong_column, _lynch_column = get_comparison_visual_columns(grid)
    # Cortex contrast panels use the Kong2022 hippocampal WTA/gradient products.
    # The separate Lynch2024 column is rendered by the comparison figures.
    cohorts = [dict(c, col=1) for c in kong_column]
    if len(cohorts) != 5 or sum(c['id'] == 'iowa_hc' for c in cohorts) != 1:
        raise ValueError('Expected exactly the five HCP/Iowa cohorts, with one Iowa HC reference')
    return cohorts


def _ids(manifest, key='subjects'):
    ids = [str(s).removeprefix('sub-') for s in (manifest.get(key) or [])]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f'Missing or duplicate subject IDs: {key}')
    return ids


def subject_label_fc(ts, roi, masks, valid, context):
    """Producer FC math with explicit missing-support and complete-frame gates."""
    core = str(ROOT / 'scripts/core_workflow')
    if core not in sys.path:
        sys.path.insert(0, core)
    from lib.fc_math import corrcoef_rows, fisher_z_transform_fc
    ts, roi, valid = np.asarray(ts), np.asarray(roi), np.asarray(valid, dtype=bool)
    if ts.ndim != 2 or roi.ndim != 2 or ts.shape[1] != roi.shape[1] or ts.shape[1] < 2:
        raise ValueError(f'FC frames must match: hippocampus={ts.shape}, cortex={roi.shape}: {context}')
    if valid.shape != (ts.shape[0],) or any(np.shape(m) != valid.shape for m in masks):
        raise ValueError(f'FC vertex masks do not match timeseries: {context}')
    roi_observed = np.isfinite(roi).all(axis=1)
    roi_empty = np.isnan(roi).all(axis=1)
    if np.any(~roi_observed & ~roi_empty):
        raise ValueError(f'Partially nonfinite cortex time series: {context}')
    cluster_ts = np.full((len(masks), ts.shape[1]), np.nan, dtype=np.float32)
    for i, mask in enumerate(masks):
        selected = np.asarray(mask, dtype=bool) & valid
        if selected.any():
            if not np.isfinite(ts[selected]).all():
                raise ValueError(f'nonfinite hippocampal time series inside valid support: {context}')
            cluster_ts[i] = np.mean(ts[selected], axis=0, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        values = fisher_z_transform_fc(corrcoef_rows(cluster_ts, roi))
        usable = np.isfinite(cluster_ts).all(axis=1) & (np.nanstd(cluster_ts, axis=1) > 0)
        usable_roi = roi_observed & (np.nanstd(roi, axis=1) > 0)
    values[~usable] = np.nan
    values[:, ~usable_roi] = np.nan
    return values


def extraction_code():
    return [fingerprint(p) for p in (
        Path(__file__), Path(__file__).with_name('multidataset_cohorts.py'),
        FIGURE_DIR / 'render_multidataset_hipp_cortex_gradient_wta.py',
        ROOT / 'scripts/core_workflow/lib/fc_math.py',
        ROOT / 'scripts/core_workflow/lib/cortex_template_profiles.py',
        ROOT / 'scripts/core_workflow/lib/cluster_roi_profile_contract.py',
        ROOT / 'scripts/core_workflow/lib/kong2022_label_utils.py',
        ROOT / 'scripts/core_workflow/lib/group_atlas_lineage.py',
    )]


def read_subject_cache(path, subjects, panels, code):
    """Reuse a named export only if its contents and every source still match."""
    path = Path(path)
    sidecar = path.with_suffix('.manifest.json')
    try:
        entry = json.loads(sidecar.read_text())
        if (entry['schema'] != SCHEMA or entry['subjects'] != subjects
                or entry['panels'] != panels or entry['code'] != code
                or fingerprint(path) != entry['profile']):
            raise ValueError('Subject export identity or code mismatch')
        for source in entry['sources']:
            if fingerprint(source['path']) != source:
                raise ValueError(f'Stale source {source["path"]}')
        with np.load(path) as data:
            if data['subjects'].tolist() != subjects or data['panels'].tolist() != panels:
                raise ValueError('Subject export ID or panel order mismatch')
            shape = (len(subjects), len(panels), 400)
            if any(data[key].shape != shape for key in ('values','left','right')):
                raise ValueError('Subject export shape mismatch')
            values = data['values']
            if not np.allclose(values, (data['left']+data['right'])/2, equal_nan=True):
                raise ValueError('Subject export bilateral mean mismatch')
        return values, entry
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f'Cannot reuse subject FC {path}: {exc}; explicitly run --rebuild-contrast-profiles after resolving input contracts') from exc


ORIGINAL_ROOT = ROOT / 'outputs_iowa_jj_tms/work/hcp_fixed_template_matching/subjects'
# Byte-identical surviving copy of the original producer's ROI registry; this
# explicit recovery does not select a source by filesystem existence.
ORIGINAL_REGISTRY = (
    resolve_routea_ctxreg_group_atlas_root('hcp')
    / 'gradient-spectral/kong2022/2mm/cov_0p80/g1g2g3/hemi_L/cluster_roi_registry.tsv'
)


def subject_input_paths(renderer, cohort, grid, subject, hemi, source_mode):
    if source_mode not in ('canonical', 'original-denoised'):
        raise ValueError(f'Unknown Iowa36P source mode: {source_mode}')
    if source_mode == 'original-denoised' and grid.variant == '36p' and cohort['dataset_key'] == 'iowa_jj_tms':
        base = ORIGINAL_ROOT / f'sub-{subject}'
        return [base/f'surface/{grid.smoothing}/hemi-{hemi}_timeseries.npy',
                base/'cortex/kong2022_roi_timeseries.npy', ORIGINAL_REGISTRY,
                base/f'surface/validity/hemi-{hemi}_valid_mask.npy']
    surface, reference, masks = renderer.resolve_cohort_inputs_roots(cohort, grid.variant)
    return [surface/f'sub-{subject}/surface/{grid.smoothing}/sub-{subject}_hemi-{hemi}_timeseries.npy',
            reference/f'sub-{subject}/reference/kong2022/cortex_roi_parcel_timeseries.npy',
            reference/f'sub-{subject}/reference/kong2022/cortex_roi_parcels.tsv',
            masks/f'sub-{subject}/surface/validity/sub-{subject}_hemi-{hemi}_valid_mask.npy']


def validate_original_source(paths, subject):
    manifest_path = ORIGINAL_ROOT/f'sub-{subject}/subject_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('status') != 'ok' or str(manifest.get('subject')).removeprefix('sub-') != subject:
        raise ValueError(f'Original source subject/status mismatch: {manifest_path}')
    registry_record = manifest['roi_registry_fingerprint']
    if fingerprint(paths[2])['sha256'] != registry_record['sha256']:
        raise ValueError(f'Original ROI row registry hash mismatch: {paths[2]}')
    recorded = {str(Path(v['path']).resolve()):v for v in manifest['outputs'].values()}
    for path in (paths[0],paths[1],paths[3]):
        actual = fingerprint(path)
        if actual['path'] not in recorded or recorded[actual['path']]['sha256'] != actual['sha256']:
            raise ValueError(f'Original subject output hash mismatch: {path}')
    if np.load(paths[1], mmap_mode='r').shape[1] != manifest['header_evidence']['concat_timepoints']:
        raise ValueError(f'Original denoised frame count mismatch: {paths[1]}')
    return fingerprint(manifest_path)


def rebuild_wta_mean(renderer, cohort, grid, values, provenance):
    """Approved recovery: publish mean of the same subject values used by t tests."""
    import shutil
    import datetime
    branch = _signal_branch_for_grid(cohort, grid)
    path = wta_profile_cache_path(cohort, grid)
    sidecar = path.with_suffix('.manifest.json')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backups = []
    for old in (path, sidecar):
        if old.is_file():
            backup = old.with_name(old.name+f'.backup-recovery-{stamp}')
            shutil.copy2(old, backup)
            backups.append(fingerprint(backup))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mean = np.nan_to_num(np.nanmean(values[:,3:], axis=0), nan=0.).astype(np.float32)
    np.save(path, mean)
    sources = {h: renderer._wta_source_run(
        Path(cohort['wta_subpath']), h,
        smoothing=grid.smoothing,
        is_7net=grid.is_7net,
        signal_branch=branch,
    ) for h in ('L','R')}
    manifest = renderer._wta_profile_cache_manifest(profile_path=path,cohort_id=cohort['id'],
        variant=grid.variant,smoothing=grid.smoothing,source_runs=sources,atlas='kong2022',
        network_order=renderer._wta_network_order(grid.is_7net),
        signal_branch=branch)
    manifest.update(recovery='original-denoised', subject_fc_sources=provenance,
                    bilateral_policy='within-subject both hemispheres required', backups=backups)
    sidecar.write_text(json.dumps(manifest,indent=2)+'\n')


def preflight_frames(renderer, grid_specs, version, source_mode="canonical"):
    """Audit every subject before writing exports or expensive recomputation."""
    errors = []
    for grid in grid_specs:
        for cohort in figure_cohorts(renderer, grid, version):
            manifest_path = Path(cohort['grad_subpath'])/'hemi_L/group_atlas_manifest.json'
            manifest = json.loads(manifest_path.read_text())
            surface, reference, masks = renderer.resolve_cohort_inputs_roots(cohort, grid.variant)
            for subject in _ids(manifest):
                for hemi in ('L','R'):
                    ts, roi, registry, valid = subject_input_paths(renderer,cohort,grid,subject,hemi,source_mode)
                    for path in (ts, roi, registry, valid):
                        if not path.is_file():
                            errors.append(f'Missing input: {path}')
                    if ts.is_file() and roi.is_file():
                        a, b = np.load(ts, mmap_mode='r'), np.load(roi, mmap_mode='r')
                        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
                            errors.append(f'{grid.tag} {cohort["id"]} sub-{subject} hemi-{hemi}: hippocampus {a.shape} at {ts}; cortex {b.shape} at {roi}')
    if errors:
        report = FIGURE_DIR/'contrast_preflight_errors.txt'
        report.write_text('\n'.join(errors)+'\n')
        raise ValueError(f'Canonical FC inputs violate temporal alignment ({len(errors)} cohort/hemi/grid errors). Repair the upstream matching inputs; no truncation or subject exclusion is allowed. Report: {report}\n{errors[0]}')


def validate_profile_subjects(manifest, expected_n, recovery):
    subjects = _ids(manifest)
    if len(subjects) != expected_n or manifest.get('n_subjects') != expected_n:
        raise ValueError('Gradient profile subject count mismatch')
    # Explicit original-source recovery reconstructs the inclusion record from
    # individually hash-verified manifests and checks the resulting group mean.
    if not (recovery and manifest.get('cluster_roi_profile_subjects_used') is None):
        if subjects != _ids(manifest, 'cluster_roi_profile_subjects_used'):
            raise ValueError('Canonical gradient subject inclusion mismatch')
    return subjects


def extract_hemisphere(renderer, cohort, grid, hemi, bundle, source_mode="canonical"):
    """Use producer FC math on exact final labels; keep empty rows as NaN."""
    from lib.cortex_template_profiles import _read_subject_roi_tsv
    grad_dir = Path(cohort['grad_subpath'])
    branch = _signal_branch_for_grid(cohort, grid)
    mean = renderer._load_gradient_roi_profile(
        grad_dir,
        hemi,
        bundle,
        cohort_spec=cohort,
        expected_signal_branch=branch,
    )
    manifest_path = grad_dir / f'hemi_{hemi}' / 'group_atlas_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    recovery = source_mode == 'original-denoised' and cohort['dataset_key'] == 'iowa_jj_tms'
    subjects = validate_profile_subjects(manifest,cohort['n_subjects'],recovery)
    sources = [fingerprint(manifest_path), fingerprint(grad_dir / f'hemi_{hemi}' / 'cluster_roi_profile.npy')]
    arrays = []
    for key, filename in [('group_labels_final', 'group_labels_final.npy'), ('group_support_mask', 'group_support_mask.npy')]:
        p = grad_dir / f'hemi_{hemi}' / filename
        if Path(manifest.get(key, '')).resolve() != p.resolve():
            raise ValueError(f'Unregistered {key}: {manifest_path}')
        sources.append(fingerprint(p))
        arrays.append(np.load(p))
    grad_labels, grad_support = arrays[0], arrays[1].astype(bool)
    if (grad_labels.ndim != 1 or grad_labels.shape != grad_support.shape
            or not set(np.unique(grad_labels)).issubset({0, 1, 2, 3})):
        raise ValueError(f'Invalid final gradient labels/support: {manifest_path}')
    wta = renderer._wta_source_run(
        Path(cohort['wta_subpath']),
        hemi,
        smoothing=grid.smoothing,
        is_7net=grid.is_7net,
        signal_branch=branch,
    )
    if _ids(wta['manifest']) != subjects:
        raise ValueError(f'Gradient/WTA subject order mismatch: {manifest_path}')
    for key in ('manifest_path', 'label_path', 'support_path', 'label_map_path'):
        sources.append(fingerprint(wta['snapshot'][key]))
    surface, reference, masks = renderer.resolve_cohort_inputs_roots(cohort, grid.variant)
    root_contracts = [('profile_surface_root', surface)]
    if not recovery:
        root_contracts.append(('reference_root', reference))
    for key, expected in root_contracts:
        if Path(manifest.get(key, '')).resolve() != expected.resolve():
            raise ValueError(f'FC extraction root mismatch for {key}: {manifest_path}')
    networks = renderer._wta_network_order(grid.is_7net)
    masks_group = [(grad_labels == k) & grad_support for k in (1, 2, 3)]
    masks_group += [(wta['labels'] == wta['network_label_keys'][n]) & wta['support'] for n in networks]
    registry = bundle['roi_registry']
    columns = {int(row['roi_id']): i for i, row in enumerate(registry)}
    stack = []
    for subject in subjects:
        paths = subject_input_paths(renderer,cohort,grid,subject,hemi,source_mode)
        sources.extend(fingerprint(p) for p in paths)
        ts, roi, valid = np.load(paths[0]), np.load(paths[1]), np.load(paths[3])
        if source_mode == 'original-denoised' and grid.variant == '36p' and cohort['dataset_key'] == 'iowa_jj_tms':
            sources.append(validate_original_source(paths,subject))
            with paths[2].open() as handle:
                rows = [dict(parcel_id=row['roi_id'], parcel_name=row['roi_name']) for row in csv.DictReader(handle,delimiter='\t')]
        else:
            rows = _read_subject_roi_tsv(paths[2])
        if (ts.ndim != 2 or roi.ndim != 2 or ts.shape[1] != roi.shape[1]
                or ts.shape[0] != grad_labels.size or valid.shape != grad_labels.shape
                or roi.shape[0] != len(rows) or len(rows) != 400):
            raise ValueError(f'Subject FC shape/400 ROI contract mismatch: {paths}')
        roi_ids = [int(row['parcel_id']) for row in rows]
        if len(set(roi_ids)) != 400 or set(roi_ids) != set(columns):
            raise ValueError(f'ROI identity mismatch: {paths[2]}')
        for row in rows:
            reg = registry[columns[int(row['parcel_id'])]]
            if row['parcel_name'] != reg['roi_name']:
                raise ValueError(f'ROI name mismatch: {paths[2]}')
        values = subject_label_fc(ts, roi, masks_group, valid, str(paths))
        ordered = np.full((7, 400), np.nan, dtype=np.float32)
        ordered[:, [columns[i] for i in roi_ids]] = values
        stack.append(ordered)
    stack = np.stack(stack)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        extracted_mean = np.nanmean(stack[:, :3], axis=0)
    if not np.allclose(extracted_mean, mean, rtol=1e-5, atol=2e-6, equal_nan=True):
        raise ValueError(f'Individual Gradient FC does not reproduce canonical mean: {manifest_path}; max delta={np.nanmax(np.abs(extracted_mean-mean))}')
    return stack, subjects, sources


def canonical_wta_mean(renderer, cohort, grid):
    branch = _signal_branch_for_grid(cohort, grid)
    path = wta_profile_cache_path(cohort, grid)
    sources = {h: renderer._wta_source_run(
        Path(cohort['wta_subpath']),
        h,
        smoothing=grid.smoothing,
        is_7net=grid.is_7net,
        signal_branch=branch,
    ) for h in ('L','R')}
    mean = renderer._load_wta_profile_cache(path, path.with_suffix('.manifest.json'),
        expected_cohort_id=cohort['id'], expected_variant=grid.variant,
        expected_smoothing=grid.smoothing, source_runs=sources, expected_atlas='kong2022',
        expected_network_order=renderer._wta_network_order(grid.is_7net),
        expected_signal_branch=branch)
    return mean, [fingerprint(path), fingerprint(path.with_suffix('.manifest.json'))]


def validate_display_mean(subject_values, expected, context):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mean = np.nanmean(subject_values, axis=0)
    # The upper WTA map represents an entirely absent network as neutral zero;
    # its lower t map remains untestable, never a set of zero observations.
    if not np.allclose(np.nan_to_num(mean, nan=0.), expected, rtol=1e-5, atol=2e-6):
        raise ValueError(f'Individual bilateral FC mean differs from displayed canonical FC: {context}; align upstream inputs and regenerate canonical means before exporting contrasts')


def prepare_contrasts(
    grid_specs,
    version='canonical',
    rebuild_profiles=False,
    source_mode='canonical',
    manifest_path=None,
):
    """Export the requested grids before choosing their shared t color range."""
    renderer = renderer_module()
    bundle = renderer.load_kong2022_bundle(dlabel_path=renderer.resolve_cortex_assets()['dlabel'])
    if rebuild_profiles:
        preflight_frames(renderer, grid_specs, version, source_mode)
    out = FIGURE_DIR / 'contrasts'
    out.mkdir(exist_ok=True)
    receipt = {'schema': SCHEMA, 'version': version, 'iowa36p_source': source_mode, 'test': 'two-sided Welch',
               'correction': 'none; uncorrected p<0.05 independently per ROI',
               'threshold': 'p<0.05', 'direction': 'group - Iowa HC',
               'bilateral': 'within-subject arithmetic mean, both hemispheres required',
               'grids': {}, 't_vmax': 1., 't_display_vmax': renderer.T_DISPLAY_VMAX, 'colormap': T_CMAP,
               'code': extraction_code(),
               'roi_registry': bundle['roi_registry']}
    peak = 1.
    for grid in grid_specs:
        cohorts = figure_cohorts(renderer, grid, version)
        profiles, entries = {}, {}
        for cohort in cohorts:
            cid = cohort['id']
            names = list(renderer.GRADIENT_SUBREGIONS) + list(renderer._wta_network_order(grid.is_7net))
            path = out / f'{version}_{grid.tag}_{cid}_subjects.npz'
            manifest_file = Path(cohort['grad_subpath']) / 'hemi_L/group_atlas_manifest.json'
            if manifest_file.is_file():
                manifest = json.loads(manifest_file.read_text())
                ids = _ids(manifest)
            else:
                sidecar = path.with_suffix('.manifest.json')
                if not sidecar.is_file():
                    raise FileNotFoundError(f'Missing cohort manifest: {manifest_file}')
                ids = json.loads(sidecar.read_text())['subjects']
            if rebuild_profiles:
                print(f'[FC export] {grid.tag} {cid}', flush=True)
                left, ids, lsources = extract_hemisphere(renderer, cohort, grid, 'L', bundle, source_mode)
                right, rids, rsources = extract_hemisphere(renderer, cohort, grid, 'R', bundle, source_mode)
                values = bilateral_profiles(left, ids, right, rids)
                if source_mode == 'original-denoised' and grid.variant == '36p' and cohort['dataset_key'] == 'iowa_jj_tms':
                    rebuild_wta_mean(renderer,cohort,grid,values,lsources+rsources)
                wta_mean, mean_sources = canonical_wta_mean(renderer, cohort, grid)
                validate_display_mean(values[:, 3:], wta_mean, mean_sources[0]['path'])
                branch = _signal_branch_for_grid(cohort, grid)
                grad_mean = (renderer._load_gradient_roi_profile(
                                 Path(cohort['grad_subpath']),
                                 'L',
                                 bundle,
                                 cohort_spec=cohort,
                                 expected_signal_branch=branch,
                             ) + renderer._load_gradient_roi_profile(
                                 Path(cohort['grad_subpath']),
                                 'R',
                                 bundle,
                                 cohort_spec=cohort,
                                 expected_signal_branch=branch,
                             )) / 2
                validate_display_mean(values[:, :3], grad_mean, str(cohort['grad_subpath']))
                # Keep hemisphere exports so the displayed group mean and valid N
                # can be audited without reading timeseries again.
                np.savez_compressed(path, values=values, left=left, right=right, subjects=np.array(ids), panels=np.array(names))
                entry = dict(schema=SCHEMA, iowa36p_source=source_mode, profile=fingerprint(path), subjects=ids, panels=names,
                             sources=lsources+rsources+mean_sources, code=receipt['code'])
                sidecar = path.with_suffix('.manifest.json')
                sidecar.write_text(json.dumps(entry, indent=2)+'\n')
            else:
                print(f'[FC cache] {grid.tag} {cid}', flush=True)
                values, entry = read_subject_cache(path, ids, names, receipt['code'])
                if entry.get('iowa36p_source') != source_mode:
                    raise ValueError(f'Subject FC source mode differs: {path}')
            profiles[cid] = values
            entries[cid] = entry
        hc = profiles['iowa_hc']
        for cid, values in profiles.items():
            if cid in ('iowa_hc', 'hcp_7t'):
                entries[cid]['comparison'] = None
                continue
            ids = set(entries[cid]['subjects'])
            cohort = next(c for c in cohorts if c['id'] == cid)
            if cohort['dataset_key'] == 'iowa_jj_tms' and ids & set(entries['iowa_hc']['subjects']):
                raise ValueError(f'Overlapping subjects with Iowa HC: {cid}')
            result = compare_profiles(values, hc)
            significant_t = np.abs(result['t'][result['significant']])
            if significant_t.size:
                peak = max(peak, float(significant_t.max()))
            path = out / f'{version}_{grid.tag}_{cid}_vs_iowa_hc.npz'
            np.savez_compressed(path, **result)
            entries[cid]['comparison'] = fingerprint(path)
            table = path.with_suffix('.tsv')
            with table.open('w') as handle:
                writer = csv.writer(handle, delimiter='\t')
                writer.writerow(['panel', 'roi_id', 'roi_name', 't', 'p', 'n_group', 'n_reference', 'testable', 'significant'])
                for j, name in enumerate(entries[cid]['panels']):
                    for i, roi in enumerate(bundle['roi_registry']):
                        writer.writerow([name, roi['roi_id'], roi['roi_name']] + [result[key][j, i] for key in ('t','p','n_group','n_reference','testable','significant')])
            print(f'[Welch/raw-p] {grid.tag} {cid}: {result["significant"].sum(axis=1).tolist()} significant ROI', flush=True)
        receipt['grids'][grid.tag] = entries
    receipt['t_vmax'] = float(math.ceil(peak))
    receipt['signal_branches'] = sorted({
        _signal_branch_for_grid({}, grid) for grid in grid_specs
    })
    if manifest_path is None:
        tags = sorted(str(grid.tag) for grid in grid_specs)
        if set(tags) == set(GRID_VARIANTS):
            path = out / f'{version}_all_grids.manifest.json'
        else:
            path = out / f"{version}_{'__'.join(tags)}.manifest.json"
    else:
        path = Path(manifest_path)
    path.write_text(json.dumps(receipt, indent=2)+'\n')
    print(f'[Contrasts ready] statistical t peak +/-{receipt["t_vmax"]}; display range +/-{receipt["t_display_vmax"]}: {path}', flush=True)
    return path


def load_contrasts(path, grid, version='canonical'):
    receipt = json.loads(Path(path).read_text())
    if receipt['schema'] != SCHEMA or receipt['version'] != version or receipt['colormap'] != T_CMAP:
        raise ValueError(f'Contrast manifest contract mismatch: {path}')
    grids = receipt.get('grids')
    if not isinstance(grids, dict) or grid.tag not in grids:
        raise ValueError(
            f'Contrast manifest does not contain requested grid/signal branch {grid.tag!r}: {path}'
        )
    requested_branch = str(getattr(grid, 'signal_branch', 'standard'))
    recorded_branches = receipt.get('signal_branches')
    if recorded_branches is None:
        if requested_branch != 'standard':
            raise ValueError(
                f'Contrast manifest lacks explicit signal_branches for requested {grid.tag!r}: {path}'
            )
    elif not isinstance(recorded_branches, list) or requested_branch not in {str(value) for value in recorded_branches}:
        raise ValueError(
            f'Contrast manifest signal branch does not match requested {grid.tag!r}: {path}'
        )
    checked = set()
    try:
        stat_vmax = float(receipt['t_vmax'])
        display_vmax = float(receipt['t_display_vmax'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'Contrast manifest must declare statistical and display t ranges: {path}') from exc
    if (not np.isfinite(stat_vmax) or stat_vmax < 1 or not np.isfinite(display_vmax)
            or display_vmax <= 0 or receipt['code'] != extraction_code()):
        raise ValueError(f'Contrast scale or extraction code mismatch: {path}')
    entries = grids[grid.tag]
    all_entries = [e for group in grids.values() for e in group.values()]
    all_sources = receipt['code'] + [e['profile'] for e in all_entries] + [e['comparison'] for e in all_entries if e['comparison'] is not None]
    for source in all_sources:
        if source['path'] not in checked:
            if fingerprint(source['path']) != source:
                raise ValueError(f'Stale contrast source: {source["path"]}; rerun suite preparation')
            checked.add(source['path'])
    for e in all_entries:
        for s in e.get('sources', []):
            sp = Path(s['path'])
            if sp.is_file() and str(sp) not in checked:
                if fingerprint(sp) != s:
                    raise ValueError(f'Stale contrast source: {sp}; rerun suite preparation')
                checked.add(str(sp))
    result = {}
    from lib.cluster_roi_profile_contract import KONG2022_CLUSTER_PROFILE_SUBREGIONS, KONG2022_WTA_TOP4_NETWORK_ORDER
    expected_panels = list(KONG2022_CLUSTER_PROFILE_SUBREGIONS) + list(KONG2022_WTA_TOP4_NETWORK_ORDER)
    for cid, entry in entries.items():
        if entry['panels'] != expected_panels:
            raise ValueError(f'Contrast panel order mismatch: {cid}')
        source = entry['comparison']
        if cid in ('iowa_hc', 'hcp_7t') and source is not None:
            raise ValueError(f'{cid} must not have a comparison')
        if source is None:
            if cid not in ('iowa_hc', 'hcp_7t'):
                raise ValueError(f'Missing comparison for {cid}')
            result[cid] = None
        else:
            if fingerprint(source['path']) != source:
                raise ValueError(f'Stale contrast result: {source["path"]}')
            with np.load(source['path']) as data:
                keys = ('t','p','n_group','n_reference','testable','significant')
                if set(data.files) != set(keys) or any(data[k].shape != (7,400) for k in keys):
                    raise ValueError(f'Contrast array contract mismatch: {source["path"]}')
                result[cid] = {key: data[key] for key in keys}
    return receipt, result

"""Render publication figures and a standalone explorer from hash-validated derivatives."""
from collections import defaultdict
import csv
import hashlib
import html
import json
import math
from pathlib import Path

NETWORKS = ('DMN', 'SMN', 'VAN', 'DAN', 'CON', 'LAN', 'VN')
BRANCHES = ('standard', 'ctxreg')
COMPARISONS = ('patient', 'pure_mdd', 'comorbid_mdd')
COHORTS = ('hc', *COMPARISONS)
HEMIS = ('L', 'R')
SYSTEMS = ('A', 'B', 'P')
ATLASES = ('tm_dominant', 'de_novo', 'wta')
CORTEX_BRANCH = 'shared_36p_cortex'
REQUIRED = ('statistics.csv', 'subject_metrics.csv', 'curves.csv', 'boundary_positions.csv', 'spatial_maps.json', 'roi_registry.json', 'roi_statistics.csv', 'analysis_config.json')
THEMES = ('identity', 'spatial', 'fingerprints', 'continuous')
STATIC_PDF = 'iowa_hipp_functional_organization_report.pdf'
FIXED_SCOPE = {'smoothing': '2mm', 'variant': '36p', 'taxonomy': 'canonical7', 'n_roi': 400, 'density': 512}
UNFOLD_AP_SPAN_MM = 38.75
IDENTITY_SUBJECT_METRICS = {
    ('tm_individual', 'smax'): ('whole', 'A', 'B', 'P'),
    ('tm_individual', 'margin'): ('whole', 'A', 'B', 'P'),
    ('hcp_fixed_territory', 'fixed_hcp_similarity'): ('A', 'B', 'P'),
    ('hcp_fixed_territory', 'fixed_hcp_advantage'): ('A', 'B', 'P'),
    ('tm_individual', 'segregation'): ('A', 'B', 'P'),
    ('tm_individual', 'boundary_sharpness'): ('A', 'B', 'P'),
}


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def p_symbol(value):
    p = number(value)
    if p is None:
        return ''
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else '+' if p < .1 else ''


def effect_label(row):
    g = number(row.get('g'))
    if g is None:
        return '×'
    return f'{g:.2f}{p_symbol(row.get("p"))}' + (' ↑' if g > 1.5 else ' ↓' if g < -1.5 else '')


STATISTIC_BASE_FIELDS = ('theme', 'branch', 'hemi', 'atlas', 'system', 'metric', 'target')


def _statistic_scope(rows, subject_rows):
    seen_main = set()
    seen_cortex = set()
    seen_effects = set()
    allowed_atlases = {
        'identity': {'tm_individual', 'hcp_fixed_territory'},
        'spatial': {'tm_individual'},
        'fingerprints': set(ATLASES),
        'continuous': {'tm_individual', 'individual_gradient'},
    }
    for row in rows:
        theme = _theme(row)
        branch = row.get('branch')
        if branch not in BRANCHES and branch != CORTEX_BRANCH:
            raise ValueError(f'statistics.csv fixed scope forbids branch {branch!r}')
        comparison = row.get('comparison')
        if comparison not in COMPARISONS:
            raise ValueError(f'statistics.csv comparison is outside the prespecified set: {row}')
        if row.get('case_cohort') != comparison or row.get('control_cohort') != 'hc':
            raise ValueError(f'statistics.csv case/control contract mismatch: {row}')
        effect_key = tuple(_theme(row) if field == 'theme' else row.get(field) for field in STATISTIC_BASE_FIELDS) + (comparison,)
        if effect_key in seen_effects:
            raise ValueError(f'duplicate statistics effect estimate: {effect_key}')
        seen_effects.add(effect_key)
        if branch in BRANCHES:
            if theme not in THEMES or row.get('hemi') not in HEMIS:
                raise ValueError(f'statistics.csv main-branch scope mismatch: {row}')
            if row.get('atlas') not in allowed_atlases[theme]:
                raise ValueError(f'statistics.csv atlas is invalid for {theme}: {row}')
            seen_main.add((theme, branch, row['hemi'], comparison))
        else:
            if (
                theme != 'spatial'
                or row.get('hemi') != 'bilateral_cortex'
                or row.get('atlas') != 'cortex_canonical7'
                or row.get('system') != 'cortex'
                or row.get('metric') != 'area_percent'
                or row.get('target') not in NETWORKS
            ):
                raise ValueError(f'statistics.csv cortex-only scope mismatch: {row}')
            key = (comparison, row['target'])
            if key in seen_cortex:
                raise ValueError(f'duplicate cortex-area effect estimate: {key}')
            seen_cortex.add(key)
    expected_main = {
        (theme, branch, hemi, comparison)
        for theme in THEMES
        for branch in BRANCHES
        for hemi in HEMIS
        for comparison in COMPARISONS
    }
    if seen_main != expected_main:
        raise ValueError(
            'statistics.csv missing prespecified branch/hemi/theme/comparison cells: '
            f'missing={sorted(expected_main - seen_main)}, extra={sorted(seen_main - expected_main)}'
        )
    expected_cortex = {(comparison, network) for comparison in COMPARISONS for network in NETWORKS}
    if seen_cortex != expected_cortex:
        raise ValueError(
            'statistics.csv cortex-area coverage mismatch: '
            f'missing={sorted(expected_cortex - seen_cortex)}, extra={sorted(seen_cortex - expected_cortex)}'
        )
    expected_bases = {
        tuple(_theme(row) if field == 'theme' else row.get(field) for field in STATISTIC_BASE_FIELDS)
        for row in subject_rows
    }
    expected_effects = {
        (*base, comparison)
        for base in expected_bases
        for comparison in COMPARISONS
    }
    if seen_effects != expected_effects:
        raise ValueError(
            'statistics.csv effect coverage mismatch: '
            f'missing={len(expected_effects - seen_effects)}, extra={len(seen_effects - expected_effects)}'
        )


def _identity_metric_scope(rows):
    expected = {
        (branch, hemi, atlas, system, metric, '')
        for branch in BRANCHES
        for hemi in HEMIS
        for (atlas, metric), systems in IDENTITY_SUBJECT_METRICS.items()
        for system in systems
    }
    seen = set()
    for row in rows:
        if _theme(row) != 'identity':
            continue
        key = (
            row.get('branch'), row.get('hemi'), row.get('atlas'), row.get('system'),
            row.get('metric'), row.get('target', ''),
        )
        if key not in expected:
            raise ValueError(f'identity metric coverage mismatch: {key}')
        seen.add(key)
    if seen != expected:
        raise ValueError(
            'identity metric coverage mismatch: '
            f'missing={sorted(expected - seen)}, extra={sorted(seen - expected)}'
        )


def _row_scope(rows, *, artifact):
    group_cohorts = {f'group_{cohort}' for cohort in COHORTS}
    for row in rows:
        theme = _theme(row)
        branch = row.get('branch')
        if artifact == 'curves.csv' and row.get('scope') == 'descriptive_group_atlas':
            if (
                theme != 'spatial'
                or branch not in BRANCHES
                or row.get('hemi') not in HEMIS
                or row.get('atlas') not in ATLASES
                or row.get('cohort') not in group_cohorts
            ):
                raise ValueError(f'{artifact} descriptive-group scope mismatch: {row}')
        elif branch in BRANCHES:
            if theme not in THEMES or row.get('hemi') not in HEMIS or row.get('cohort') not in COHORTS:
                raise ValueError(f'{artifact} main-branch scope mismatch: {row}')
        elif branch == CORTEX_BRANCH:
            if theme != 'spatial' or row.get('hemi') != 'bilateral_cortex' or row.get('cohort') not in COHORTS:
                raise ValueError(f'{artifact} cortex-only scope mismatch: {row}')
        else:
            raise ValueError(f'{artifact} fixed scope forbids branch {branch!r}')


def _map_scope(entries):
    import numpy as np

    expected_spatial = {
        (branch, atlas, cohort, hemi)
        for branch in BRANCHES
        for atlas in ATLASES
        for cohort in COHORTS
        for hemi in HEMIS
    }
    expected_continuous = {
        (branch, cohort, hemi)
        for branch in BRANCHES
        for cohort in COHORTS
        for hemi in HEMIS
    }
    seen_spatial = set()
    seen_continuous = set()
    for entry in entries:
        theme = entry.get('theme')
        branch = entry.get('branch')
        cohort = entry.get('cohort')
        hemi = entry.get('hemi')
        if branch not in BRANCHES or cohort not in COHORTS or hemi not in HEMIS:
            raise ValueError(f'spatial_maps.json fixed scope mismatch: {entry}')
        if theme == 'spatial':
            atlas = entry.get('atlas')
            if atlas not in ATLASES or entry.get('scope') != 'descriptive_group_atlas':
                raise ValueError(f'spatial_maps.json categorical atlas contract mismatch: {entry}')
            key = (branch, atlas, cohort, hemi)
            if key in seen_spatial:
                raise ValueError(f'duplicate descriptive group-map cell: {key}')
            _categorical_map_values(entry)
            seen_spatial.add(key)
        elif theme == 'continuous':
            if (
                entry.get('atlas') != 'individual_gradient'
                or entry.get('scope') != 'descriptive_group_mean'
            ):
                raise ValueError(f'spatial_maps.json gradient atlas contract mismatch: {entry}')
            key = (branch, cohort, hemi)
            if key in seen_continuous:
                raise ValueError(f'duplicate descriptive gradient-map cell: {key}')
            gradients = np.asarray(entry.get('gradients'), dtype=float)
            if gradients.shape != (512, 3):
                raise ValueError('spatial_maps.json gradient map needs 512 vertices by three components')
            seen_continuous.add(key)
        else:
            raise ValueError(f'spatial_maps.json has unsupported theme: {entry}')
    if seen_spatial != expected_spatial:
        raise ValueError(
            'spatial_maps.json descriptive group-map grid mismatch: '
            f'missing={sorted(expected_spatial - seen_spatial)}, extra={sorted(seen_spatial - expected_spatial)}'
        )
    if seen_continuous != expected_continuous:
        raise ValueError(
            'spatial_maps.json descriptive gradient-map grid mismatch: '
            f'missing={sorted(expected_continuous - seen_continuous)}, extra={sorted(seen_continuous - expected_continuous)}'
        )


def _roi_scope(rows, roi_ids):
    expected = {
        (branch, hemi, atlas, system, roi_id, comparison)
        for branch in BRANCHES
        for hemi in HEMIS
        for atlas in ATLASES
        for system in SYSTEMS
        for roi_id in roi_ids
        for comparison in COMPARISONS
    }
    seen = set()
    for row in rows:
        try:
            roi_id = int(row.get('roi_id', ''))
        except (TypeError, ValueError) as exc:
            raise ValueError(f'roi_statistics.csv has non-integer ROI ID: {row}') from exc
        if (
            _theme(row) != 'fingerprints'
            or row.get('branch') not in BRANCHES
            or row.get('hemi') not in HEMIS
            or row.get('atlas') not in ATLASES
            or row.get('system') not in SYSTEMS
            or row.get('metric') != 'fisher_z'
            or row.get('target_level') != 'roi'
            or roi_id not in roi_ids
            or row.get('comparison') not in COMPARISONS
            or row.get('case_cohort') != row.get('comparison')
            or row.get('control_cohort') != 'hc'
        ):
            raise ValueError(f'roi_statistics.csv fixed scope mismatch: {row}')
        key = (
            row['branch'], row['hemi'], row['atlas'], row['system'],
            roi_id, row['comparison'],
        )
        if key in seen:
            raise ValueError(f'duplicate ROI effect estimate: {key}')
        seen.add(key)
    if seen != expected:
        raise ValueError(
            'roi_statistics.csv ROI effect coverage mismatch: '
            f'missing={len(expected - seen)}, extra={len(seen - expected)}'
        )


def _boundary_scope(rows):
    pairs = {'A_B', 'A_P', 'B_P'}
    for row in rows:
        if (
            row.get('theme') != 'spatial'
            or row.get('atlas') != 'tm_individual'
            or row.get('scope') != 'individual_tm'
            or row.get('branch') not in BRANCHES
            or row.get('hemi') not in HEMIS
            or row.get('cohort') not in COHORTS
            or row.get('system_pair') not in pairs
            or not row.get('subject')
        ):
            raise ValueError(f'malformed boundary position row: {row}')
        position = number(row.get('position'))
        if position is None or not 0.0 <= position <= 1.0:
            raise ValueError(f'boundary position must use the declared [0, 1] AP axis: {row}')


def _fixed_scope(payload):
    config = payload['analysis_config.json']
    if config.get('scope') != FIXED_SCOPE:
        raise ValueError(f'analysis_config.json fixed scope mismatch: {config.get("scope")}')
    tests = config.get('tests', {})
    required_tests = {
        'primary': 'two-sided Welch',
        'covariates': 'none',
        'correction': 'none',
    }
    if {key: tests.get(key) for key in required_tests} != required_tests:
        raise ValueError('analysis_config.json statistical-contract mismatch')
    _row_scope(payload['subject_metrics.csv'], artifact='subject_metrics.csv')
    _statistic_scope(payload['statistics.csv'], payload['subject_metrics.csv'])
    _identity_metric_scope(payload['subject_metrics.csv'])
    _row_scope(payload['curves.csv'], artifact='curves.csv')
    _map_scope(payload['spatial_maps.json'])
    roi_ids = {int(row['roi_id']) for row in payload['roi_registry.json']}
    _roi_scope(payload['roi_statistics.csv'], roi_ids)
    _boundary_scope(payload['boundary_positions.csv'])


def _load(output_dir):
    root = Path(output_dir)
    manifest_path = root / 'run_manifest.json'
    if not manifest_path.is_file():
        raise ValueError(f'Missing render contract: {manifest_path}')
    manifest = json.loads(manifest_path.read_text())
    if 'statistics' not in manifest.get('completed_stages', []):
        raise ValueError(f'{manifest_path}: statistics stage is incomplete')
    payload = {}
    for name in REQUIRED:
        declaration = manifest.get('outputs', {}).get(name)
        path = root / name
        if not isinstance(declaration, dict) or not declaration.get('sha256') or not path.is_file():
            raise ValueError(f'Missing declared artifact/hash: {path}')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != declaration['sha256']:
            raise ValueError(f'{path}: SHA256 mismatch; rerun the deriving stage')
        payload[name] = list(csv.DictReader(raw.decode().splitlines())) if name.endswith('.csv') else json.loads(raw)
    registry = payload['roi_registry.json']
    if len(registry) != 400 or len({r['roi_id'] for r in registry}) != 400:
        raise ValueError('roi_registry.json must declare exactly 400 distinct cortical ROIs')
    _fixed_scope(payload)
    return payload

def _palette():
    config = Path(__file__).resolve().parents[3] / 'config' / 'hipp_network_style.json'
    styles = {entry['name']: entry['rgba'] for entry in json.loads(config.read_text()).values()}
    names = dict(DMN='Default', SMN='Somatomotor', VAN='VentralAttention', DAN='DorsalAttention', CON='Control', LAN='Language', VN='Visual')
    return {key: tuple(v / 255 for v in styles[name][:3]) for key, name in names.items()}


def _theme(row):
    value = str(row.get('theme', ''))
    return dict(zip(('1','2','3','4'), THEMES)).get(value, value)


def _unfolded_display_coordinates(axis, transverse, hemi):
    """Use millimetres and one reflected transverse orientation for paired maps."""
    import numpy as np

    axis = np.asarray(axis, dtype=float)
    transverse = np.asarray(transverse, dtype=float)
    if axis.ndim != 1 or transverse.shape != axis.shape:
        raise ValueError('unfolded display coordinates must be matching one-dimensional arrays')
    if not np.isfinite(axis).all() or not np.isfinite(transverse).all():
        raise ValueError('unfolded display coordinates must be finite')
    if np.any(axis < 0.0) or np.any(axis > 1.0):
        raise ValueError('unfolded longitudinal coordinates must use the declared [0, 1] axis')
    if hemi not in ('L', 'R'):
        raise ValueError(f'unknown unfolded-map hemisphere: {hemi!r}')
    sign = 1.0 if hemi == 'L' else -1.0
    if np.any(sign * transverse < 0.0):
        raise ValueError('unfolded transverse coordinates must retain one single signed hemisphere orientation')
    return axis * UNFOLD_AP_SPAN_MM, sign * transverse


def _format_row_label(k):
    if not isinstance(k, tuple):
        return str(k)
    if len(k) == 4:
        branch, hemi, system, target = k
        branch_map = {'standard': 'Standard', 'ctxreg': 'Ctx-Reg', 'shared_36p_cortex': 'Cortex'}
        hemi_map = {'L': 'LH', 'R': 'RH', 'bilateral_cortex': 'Bilateral'}
        system_map = {
            'A': 'Ant', 'B': 'Mid', 'P': 'Post', 'whole': 'Whole', 'cortex': 'Cortex',
            'unassigned': 'Unassigned',
            'rA': 'Ant (rA)', 'rB': 'Mid (rB)', 'rP': 'Post (rP)', 'rP_minus_rA': 'Post − Ant',
            'G1': 'G1 (Principal)', 'G2': 'G2 (Secondary)', 'G3': 'G3 (Tertiary)'
        }
        b_str = branch_map.get(str(branch), str(branch))
        h_str = hemi_map.get(str(hemi), str(hemi))
        s_str = system_map.get(str(system), str(system))
        if target:
            if b_str == 'Cortex' and s_str == 'Cortex':
                return f'Cortex · {h_str} · {target}'
            return f'{b_str} · {h_str} · {s_str} · {target}'
        return f'{b_str} · {h_str} · {s_str}'
    if len(k) == 2:
        h, c = k
        hemi_map = {'L': 'LH', 'R': 'RH'}
        cohort_map = {'patient': 'All Patients (n=26)', 'pure_mdd': 'Pure MDD (n=7)', 'comorbid_mdd': 'Comorbid MDD (n=19)', 'hc': 'HC (n=67)'}
        return f'{hemi_map.get(str(h), str(h))} · {cohort_map.get(str(c), str(c))}'
    return ' · '.join(str(x) for x in k if x)


def _format_col_label(k):
    if isinstance(k, tuple):
        system_map = {'A': 'Ant', 'B': 'Mid', 'P': 'Post'}
        parts = [system_map.get(str(x), str(x)) for x in k if x]
        return ' · '.join(parts)
    cohort_map = {
        'patient': 'All Patients\n(n=26)',
        'pure_mdd': 'Pure MDD\n(n=7)',
        'comorbid_mdd': 'Comorbid MDD\n(n=19)',
        'hc': 'HC (n=67)',
    }
    return cohort_map.get(str(k), str(k))


METRIC_INFO = {
    # Theme 1: Functional Identity
    'smax': (
        'Peak Template Similarity (Smax)',
        'Individual TM · vs. Healthy Controls',
        'Best match among Ant / Mid / Post systems;\nhigher = stronger functional identity'
    ),
    'margin': (
        'Separation Margin (Top1 − Top2)',
        'Individual TM · vs. Healthy Controls',
        'Difference between top two template matches;\nhigher = sharper identity (~0 = ambiguous)'
    ),
    'fixed_hcp_similarity': (
        'Fixed HCP Territory Similarity',
        'HCP151 Reference Territory · vs. Healthy Controls',
        'Correlation with the matched HCP template\ninside the same external HCP territory for every group'
    ),
    'fixed_hcp_advantage': (
        'Fixed HCP Territory Advantage',
        'HCP151 Reference Territory · vs. Healthy Controls',
        'Matched HCP template minus strongest competitor\ninside the same external HCP territory for every group'
    ),
    'segregation': (
        'System Segregation (Cohen’s d)',
        'Individual TM (Kember) · vs. HC',
        'Within- vs. between-system cohesion;\nlower = loss of modular segregation'
    ),
    'boundary_sharpness': (
        'Boundary Sharpness (4mm Geodesic d)',
        'Individual TM (Kember) · vs. HC',
        'Cross-border transition steepness;\nlower = blurred / diffuse boundary'
    ),
    # Theme 2: Spatial Organization
    'area_fraction': (
        'Area Fraction',
        'Individual TM · vs. Healthy Controls',
        'Proportion of surface vertices assigned to system;\nhigher = expanded territorial representation'
    ),
    'area_mm2': (
        'Surface Area (mm²)',
        'Individual TM · vs. Healthy Controls',
        'Absolute anatomical surface area in mm²;\nquantitative metric of physical subfield extent'
    ),
    'area_percent': (
        'Cortical Network Coverage (%)',
        'Bilateral Cortex (Canonical 7) · vs. HC',
        'Whole-cortex percentage occupied by network;\nreference baseline for neocortical representation'
    ),
    'centroid': (
        'Longitudinal Centroid Position',
        'Individual TM · vs. Healthy Controls',
        'Center of mass along AP axis [0=Ant, 1=Post];\nshift indicates territorial relocation'
    ),
    'vertex_count': (
        'Surface Vertex Count',
        'Individual TM · vs. Healthy Controls',
        'Total number of 512 vertices assigned to system;\nraw discrete count underlying area fraction'
    ),
    # Theme 4: Continuous Organization & Gradients
    'axis_correlation': (
        'Longitudinal Axis Alignment (r)',
        'Individual Gradients · vs. Healthy Controls',
        'Correlation of G1 with anatomical AP axis;\nhigher = gradient strictly follows AP axis'
    ),
    'reference_correlation': (
        'Group Template Concordance (r)',
        'Individual Gradients · vs. Healthy Controls',
        'Concordance with HC group gradient template;\nlower = abnormal topological organization'
    ),
    'robust_range': (
        'Dynamic Range (95th − 5th %ile)',
        'Individual Gradient / TM · vs. HC',
        'Spread between 95th and 5th percentiles;\nhigher = greater functional differentiation span'
    ),
    'slope': (
        'AP Profile Slope (Δr / mm)',
        'Individual TM · vs. Healthy Controls',
        'Longitudinal slope of template matching score;\nsteepness of anterior-to-posterior transition'
    ),
}


def _format_title(title):
    parts = [p.strip() for p in title.split('|')]
    metric_key = parts[0].strip()
    atlas_part = parts[1].strip() if len(parts) > 1 else ''

    if metric_key == 'robust_range':
        if atlas_part == 'tm_individual':
            return 'Template Correlation Dynamic Range\n[Individual TM · vs. Healthy Controls]\n(95th − 5th percentile span of continuous correlation;\nhigher = wider functional contrast)'
        if atlas_part == 'individual_gradient':
            return 'Gradient Dynamic Range (G1−G3)\n[Individual Gradients · vs. Healthy Controls]\n(95th − 5th percentile spread of gradient scores;\nhigher = greater functional differentiation span)'

    if metric_key in METRIC_INFO:
        m_name, m_context, m_desc = METRIC_INFO[metric_key]
        if atlas_part and atlas_part != 'tm_individual':
            atlas_name = {
                'individual_gradient': 'Individual Gradients',
                'cortex_canonical7': 'Bilateral Cortex',
                'tm_dominant': 'Group Dominant Atlas',
                'de_novo': 'De Novo Atlas',
                'wta': 'WTA Atlas'
            }.get(atlas_part, atlas_part)
            m_context = f'{atlas_name} · vs. HC'
        return f'{m_name}\n[{m_context}]\n({m_desc})'

    if 'network fingerprint g' in title:
        branch = parts[0].strip()
        atlas = parts[1].replace(': network fingerprint g', '').strip()
        b_name = {'standard': 'Standard Pipeline', 'ctxreg': 'Ctx-Reg Pipeline'}.get(branch, branch)
        a_name = {'tm_dominant': 'Group Dominant Atlas', 'de_novo': 'De Novo Atlas', 'wta': 'WTA Atlas'}.get(atlas, atlas)
        return f'{b_name} · {a_name}\n[Cortical Network FC (Fisher’s z g)]\n(Connections between Ant/Mid/Post & 7 canonical networks)'

    if 'group atlas composition' in title:
        branch = parts[0].strip()
        atlas = parts[1].strip()
        b_name = {'standard': 'Standard Pipeline', 'ctxreg': 'Ctx-Reg Pipeline'}.get(branch, branch)
        a_name = {'tm_dominant': 'Group Dominant Atlas', 'de_novo': 'De Novo Atlas', 'wta': 'WTA Atlas'}.get(atlas, atlas)
        return f'{b_name} · {a_name}\n[Descriptive Group Surface Composition]\n(Blue: Ant · Orange: Mid · Green: Post · White: Unsupported)'

    if 'individual gradients' in title:
        branch = parts[0].strip()
        b_name = {'standard': 'Standard Pipeline', 'ctxreg': 'Ctx-Reg Pipeline'}.get(branch, branch)
        return f'{b_name}\n[Descriptive Connectopic Gradients]\n(Unfolded coordinates · Shared 2nd–98th percentile scale)'

    title_replacements = {
        'tm_individual': 'Individual TM',
        'tm_dominant': 'Group Dominant Atlas',
        'de_novo': 'De Novo Atlas',
        'wta': 'WTA Atlas',
        'cortex_canonical7': 'Bilateral Cortex (Canonical 7)',
        'individual_gradient': 'Individual Gradients',
        'all comparisons': 'vs. Healthy Controls',
    }
    formatted = [title_replacements.get(p, p) for p in parts]
    if len(formatted) == 3:
        return f'{formatted[0]}\n[{formatted[1]} · {formatted[2]}]'
    if len(formatted) == 2:
        return f'{formatted[0]}\n[{formatted[1]}]'
    return '\n'.join(formatted)


def _matrix(ax, rows, rowkeys, colkeys, rowfn, colfn, title):
    import numpy as np
    from matplotlib import colormaps
    values = np.full((len(rowkeys), len(colkeys)), np.nan)
    lookup = {}
    for row in rows:
        key = (rowfn(row), colfn(row))
        if key in lookup:
            raise ValueError(f'Duplicate report matrix cell in {title}: {key}')
        lookup[key] = row
    for i, rkey in enumerate(rowkeys):
        for j, ckey in enumerate(colkeys):
            row = lookup.get((rkey, ckey), {})
            g = number(row.get('g'))
            if g is not None:
                values[i, j] = g
                text_color = 'white' if abs(g) >= 0.8 else '#111111'
            else:
                text_color = '#555555'
            label = effect_label(row)
            weight = 'bold' if any(s in label for s in ('*', '+')) else 'normal'
            ax.text(j, i, label, ha='center', va='center', fontsize=7.5, color=text_color, fontweight=weight)
    cmap = colormaps['RdBu_r'].copy()
    cmap.set_bad('#dedede')
    ax.imshow(values, cmap=cmap, vmin=-1.5, vmax=1.5, aspect='auto', interpolation='none')
    ax.set_xticks(range(len(colkeys)), [_format_col_label(k) for k in colkeys], rotation=0 if len(colkeys) <= 4 else 45, ha='center' if len(colkeys) <= 4 else 'right', fontsize=7.5)
    ax.set_yticks(range(len(rowkeys)), [_format_row_label(k) for k in rowkeys], fontsize=7.5)
    ax.set_title(_format_title(title), fontsize=8.5, fontweight='bold', pad=8, linespacing=1.2)
    ax.set_xticks(np.arange(-.5, len(colkeys), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(rowkeys), 1), minor=True)
    ax.grid(which='minor', color='#ffffff', linestyle='-', linewidth=0.6, alpha=0.6)
    ax.tick_params(which='minor', bottom=False, left=False)


def _effect_panels(rows, theme):
    panels = []
    if theme == 'fingerprints':
        for branch in (branch for branch in BRANCHES if any(r['branch'] == branch for r in rows)):
            for atlas in ATLASES:
                selected = [r for r in rows if r['branch'] == branch and r['atlas'] == atlas]
                rkeys = [(h,c) for h in ('L','R') for c in COMPARISONS]
                ckeys = [(s,n) for s in ('A','B','P') for n in NETWORKS]
                panels.append(('effects', selected, rkeys, ckeys, lambda r:(r['hemi'],r['comparison']), lambda r:(r['system'],r['target']), f'{branch} | {atlas}: network fingerprint g'))
    else:
        groups = defaultdict(list)
        for row in rows:
            groups[(row['metric'], row['atlas'])].append(row)
        for (metric, atlas), selected in sorted(groups.items()):
            key = lambda r: (r['branch'], r['hemi'], r['system'], r['target'])
            keys = sorted({key(r) for r in selected})
            # Limit vertical size of each panel without selecting by significance.
            for start in range(0, len(keys), 22):
                subset = keys[start:start+22]
                panels.append(('effects', [r for r in selected if key(r) in subset], subset, COMPARISONS, key, lambda r:r['comparison'], f'{metric} | {atlas} | all comparisons'))
    return panels


def _curve_legend_handles(theme):
    from matplotlib.lines import Line2D
    if theme == 'continuous':
        sys_handles = [
            Line2D([], [], color='#2b83ba', lw=2.2, label='Ant (rA)'),
            Line2D([], [], color='#e66101', lw=2.2, label='Mid (rB)'),
            Line2D([], [], color='#2ca25f', lw=2.2, label='Post (rP)'),
            Line2D([], [], color='#7b3294', lw=2.2, label='Post − Ant'),
        ]
    else:
        sys_handles = [
            Line2D([], [], color='#2b83ba', lw=2.2, label='Ant'),
            Line2D([], [], color='#e66101', lw=2.2, label='Mid'),
            Line2D([], [], color='#2ca25f', lw=2.2, label='Post'),
            Line2D([], [], color='#999999', lw=2.2, label='Unassigned'),
        ]
    coh_handles = [
        Line2D([], [], color='#333333', lw=2.2, linestyle='-', label='HC (n=67)'),
        Line2D([], [], color='#333333', lw=1.8, linestyle='--', label='All Patients (n=26)'),
        Line2D([], [], color='#333333', lw=1.8, linestyle=':', label='Pure MDD (n=7)'),
        Line2D([], [], color='#333333', lw=1.8, linestyle='-.', label='Comorbid (n=19)'),
    ]
    return sys_handles + coh_handles


def _draw_curves(ax, rows, title, show_legend=True, show_ylabel=True):
    import numpy as np
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        pos, val = number(row['position']), number(row['value'])
        if pos is not None and val is not None:
            groups[(row['cohort'], row['system'])][pos].append(val)

    system_colors = {
        'A': '#2b83ba',
        'rA': '#2b83ba',
        'B': '#e66101',
        'rB': '#e66101',
        'P': '#2ca25f',
        'rP': '#2ca25f',
        'rP_minus_rA': '#7b3294',
        'unassigned': '#999999',
    }
    cohort_styles = {
        'hc': ('-', 2.2, 0.95),
        'group_hc': ('-', 2.2, 0.95),
        'patient': ('--', 1.8, 0.9),
        'group_patient': ('--', 1.8, 0.9),
        'pure_mdd': (':', 1.8, 0.85),
        'group_pure_mdd': (':', 1.8, 0.85),
        'comorbid_mdd': ('-.', 1.8, 0.85),
        'group_comorbid_mdd': ('-.', 1.8, 0.85),
    }

    sys_order = ['A', 'rA', 'B', 'rB', 'P', 'rP', 'rP_minus_rA', 'unassigned']
    coh_order = ['hc', 'group_hc', 'patient', 'group_patient', 'pure_mdd', 'group_pure_mdd', 'comorbid_mdd', 'group_comorbid_mdd']

    def sort_key(item):
        cohort, system = item[0]
        s_idx = sys_order.index(system) if system in sys_order else 99
        c_idx = coh_order.index(cohort) if cohort in coh_order else 99
        return (s_idx, c_idx)

    for (cohort, system), bins in sorted(groups.items(), key=sort_key):
        x = sorted(bins)
        y = [np.mean(bins[p]) for p in x]
        linestyle, lw, alpha = cohort_styles.get(cohort, ('-', 1.2, 0.8))
        color = system_colors.get(system, '#333333')
        ax.plot(x, y, color=color, linestyle=linestyle, linewidth=lw, alpha=alpha)

    parts = [p.strip() for p in title.split('|')]
    if len(parts) >= 4:
        branch_map = {'standard': 'Standard', 'ctxreg': 'Ctx-Reg'}
        hemi_map = {'L': 'LH', 'R': 'RH'}
        atlas_map = {
            'tm_individual': 'Individual TM',
            'tm_dominant': 'Group Dominant',
            'de_novo': 'De Novo',
            'wta': 'WTA'
        }
        metric_map = {
            'mean': 'Continuous Template Correlation',
            'area_fraction': 'Subfield Area Fraction'
        }
        desc_map = {
            'mean': 'Mean r along AP axis [0=Ant, 1=Post]',
            'area_fraction': 'Proportion of surface vertices along AP axis'
        }
        b_str = branch_map.get(parts[0], parts[0])
        h_str = hemi_map.get(parts[1], parts[1])
        a_str = atlas_map.get(parts[2], parts[2])
        m_str = metric_map.get(parts[3], parts[3])
        d_str = desc_map.get(parts[3], 'Descriptive subject-mean curves')
        ax.set_title(f'{b_str} · {h_str} · {a_str}\n[{m_str}]\n({d_str})', fontsize=8, fontweight='bold', pad=5, linespacing=1.2)
    else:
        ax.set_title(title + '\nDescriptive subject-mean curves', fontsize=8, fontweight='bold', pad=5)

    ax.set_xlabel('Longitudinal Axis (Ant: 0.0 ──> Post: 1.0)', fontsize=7.5)
    if show_ylabel:
        ax.set_ylabel('Template Correlation (r)' if 'mean' in title else 'Area Fraction', fontsize=7.5)
    if 'area_fraction' in title:
        ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle='--', alpha=0.35)

    if show_legend:
        handles = _curve_legend_handles('continuous' if 'mean' in title else 'spatial')
        ax.legend(handles=handles, fontsize=6, ncol=4, framealpha=0.9, loc='upper right', columnspacing=0.8, handletextpad=0.4)


def _draw_boundaries(ax, rows, title, show_ylabel=True):
    """Show every individual boundary edge; no single transition is selected."""
    cohorts = ('hc', *COMPARISONS)
    pairs = ('A_B', 'A_P', 'B_P')
    pair_labels = {'A_B': 'Ant–Mid', 'A_P': 'Ant–Post', 'B_P': 'Mid–Post'}
    pair_colors = {'A_B': '#2b83ba', 'A_P': '#7b3294', 'B_P': '#d7191c'}
    cohort_labels = {
        'hc': 'HC (n=67)',
        'patient': 'All Patients (n=26)',
        'pure_mdd': 'Pure MDD (n=7)',
        'comorbid_mdd': 'Comorbid MDD (n=19)',
    }
    groups = defaultdict(list)
    for row in rows:
        value = number(row.get('position'))
        if value is not None:
            groups[(row.get('cohort'), row.get('system_pair'))].append(value)
    labels = []
    for index, (cohort, pair) in enumerate((cohort, pair) for cohort in cohorts for pair in pairs):
        values = groups[(cohort, pair)]
        if values:
            ax.scatter(values, [index] * len(values), s=16, alpha=0.40, color=pair_colors[pair], edgecolors='none', rasterized=True)
            import numpy as np
            med = np.median(values)
            ax.scatter([med], [index], s=26, marker='D', color='#111111', zorder=4)
        c_name = cohort_labels.get(cohort, cohort)
        p_name = pair_labels.get(pair, pair.replace('_', '–'))
        labels.append(f'{c_name} · {p_name}')

    for sep in (2.5, 5.5, 8.5):
        ax.axhline(sep, color='#dddddd', linestyle='-', lw=0.9)

    ax.set_xlim(0, 1)
    ax.set_yticks(range(len(labels)))
    if show_ylabel:
        ax.set_yticklabels(labels, fontsize=7)
    else:
        ax.set_yticklabels([])
        ax.tick_params(labelleft=False)
    ax.set_xlabel('Longitudinal AP Position (Ant: 0.0 ──> Post: 1.0; Diamond: Median)', fontsize=7.5)

    parts = [p.strip() for p in title.split('|')]
    branch_map = {'standard': 'Standard Pipeline', 'ctxreg': 'Ctx-Reg Pipeline'}
    hemi_map = {'L': 'Left Hippocampus (LH)', 'R': 'Right Hippocampus (RH)'}
    b_str = branch_map.get(parts[0], parts[0])
    h_str = hemi_map.get(parts[1], parts[1])
    ax.set_title(
        f'{b_str} · {h_str}\n[Individual Boundary Transition Midpoints (TM)]\n(All valid native-edge AP positions; multiple transitions retained)',
        fontsize=8, fontweight='bold', pad=6, linespacing=1.2
    )
    ax.grid(axis='x', linestyle='--', alpha=0.35)



def _categorical_map_values(entry):
    """Use NaN for unsupported vertices so invalid support is never rendered as label 0."""
    import numpy as np

    raw = np.asarray(entry.get('labels'), dtype=object)
    support = np.asarray(entry.get('support'))
    if raw.shape != (512,) or support.shape != (512,) or not np.isin(support, (0, 1)).all():
        raise ValueError('spatial_maps.json categorical map needs 512 labels and a binary 512-vertex support mask')
    mapping = {'A': 1, 'B': 2, 'P': 3}
    colors = np.full(512, np.nan)
    for index, value in enumerate(raw):
        mapped = mapping.get(str(value), number(value))
        if mapped is not None:
            colors[index] = mapped
    if not np.isin(colors[np.isfinite(colors)], (0, 1, 2, 3)).all():
        raise ValueError('spatial_maps.json categorical labels must be A/B/P or 0/1/2/3')
    colors[~support.astype(bool)] = np.nan
    return colors


def _draw_map(ax, entry, gradient=None):
    import numpy as np
    from matplotlib.colors import ListedColormap
    axis = np.asarray(entry['axis'], dtype=float)
    transverse = np.asarray(entry['transverse'], dtype=float)
    if len(axis) != 512 or len(transverse) != 512:
        raise ValueError('spatial_maps.json: consumed hippocampal map must have 512 vertices')
    axis, transverse = _unfolded_display_coordinates(axis, transverse, str(entry.get('hemi', '')))
    if gradient is None:
        colors = _categorical_map_values(entry)
        artist = ax.scatter(axis, transverse, c=colors, s=7, cmap=ListedColormap(['#969696','#4477AA','#EE6677','#228833']), vmin=0,vmax=3)
        suffix = 'A blue / B red / P green / white unsupported'
    else:
        values = np.asarray(entry['gradients'],dtype=float)[:,gradient]
        artist = ax.scatter(axis, transverse, c=values, s=7, cmap='viridis')
        ax.figure.colorbar(artist,ax=ax,shrink=.6); suffix=f'Gradient {gradient+1}'
    ax.set_title(' | '.join(str(entry[k]) for k in ('branch','hemi','atlas','cohort')) + '\nDescriptive group map · ' + suffix,fontsize=8)
    ax.set_xlabel('Anterior → posterior (mm)'); ax.set_ylabel('Transverse (mm; R reflected)')


def _draw_map_grid(ax, entries, title, gradient=None, show_cohorts=True):
    """Render all cohort-by-hemisphere group maps in one descriptive panel."""
    import numpy as np
    from matplotlib.colors import ListedColormap

    cohorts = ("hc", "patient", "pure_mdd", "comorbid_mdd")
    hemis = ("L", "R")
    expected = {(cohort, hemi) for cohort in cohorts for hemi in hemis}
    lookup = {}
    coordinate_pairs = []
    gradient_values = []
    for entry in entries:
        key = (str(entry.get("cohort", "")), str(entry.get("hemi", "")))
        if key not in expected or key in lookup:
            raise ValueError(f"Invalid or duplicate descriptive group-map cell: {key}")
        axis = np.asarray(entry.get("axis"), dtype=float)
        transverse = np.asarray(entry.get("transverse"), dtype=float)
        if axis.shape != (512,) or transverse.shape != (512,) or not np.isfinite(axis).all() or not np.isfinite(transverse).all():
            raise ValueError("spatial_maps.json: every group-map cell needs 512 finite coordinates")
        coordinate_pairs.append(_unfolded_display_coordinates(axis, transverse, key[1]))
        if gradient is not None:
            values = np.asarray(entry.get("gradients"), dtype=float)
            if values.shape != (512, 3):
                raise ValueError("spatial_maps.json: gradient map needs 512 vertices by three components")
            gradient_values.append(values[:, gradient])
        lookup[key] = entry
    if set(lookup) != expected:
        raise ValueError(
            "Incomplete descriptive group-map grid: "
            f"got {sorted(lookup)}, expected {sorted(expected)}"
        )

    all_axis = np.concatenate([pair[0] for pair in coordinate_pairs])
    all_transverse = np.concatenate([pair[1] for pair in coordinate_pairs])
    xlim = (float(np.min(all_axis)), float(np.max(all_axis)))
    ylim = (float(np.min(all_transverse)), float(np.max(all_transverse)))
    if xlim[0] == xlim[1] or ylim[0] == ylim[1]:
        raise ValueError("spatial_maps.json: descriptive group-map coordinates must vary")
    if gradient is not None:
        finite = np.concatenate(gradient_values)
        finite = finite[np.isfinite(finite)]
        if not len(finite):
            raise ValueError("spatial_maps.json: no finite gradient values for descriptive map")
        low, high = np.percentile(finite, [2, 98])
        if low == high:
            low, high = float(np.min(finite)), float(np.max(finite))
        if low == high:
            high = low + 1.0
        suffix = f"G{gradient + 1}; shared 2nd–98th percentile color range (viridis)"
    else:
        low = high = None
        suffix = "Ant: blue · Mid: orange · Post: green · gray: unassigned · white: unsupported · RH reflected"

    ax.set_axis_off()
    b_name = {'standard': 'Standard Pipeline', 'ctxreg': 'Ctx-Reg Pipeline'}.get(title.split('|')[0].strip(), title.split('|')[0].strip())
    if gradient is not None:
        g_titles = {0: 'Gradient 1 (Principal AP Axis)', 1: 'Gradient 2 (Secondary Subfield)', 2: 'Gradient 3 (Tertiary)'}
        g_desc = g_titles.get(gradient, f'Gradient {gradient+1}')
        panel_title = f'{b_name} · {g_desc}\n[Descriptive Connectopic Gradient Map (512 vertices)]\n(Unfolded surface coordinates in mm; shared 2nd–98th percentile scale)'
    else:
        atlas_raw = title.split('|')[1].strip() if len(title.split('|')) > 1 else ''
        a_name = {'tm_dominant': 'Group Dominant Atlas', 'de_novo': 'De Novo Atlas', 'wta': 'WTA Atlas'}.get(atlas_raw, atlas_raw)
        panel_title = f'{b_name} · {a_name}\n[Descriptive Group Subfield Composition (512 vertices)]\n(Ant: blue, Mid: orange, Post: green, Gray: unassigned, White: unsupported)'
    ax.set_title(panel_title, fontsize=8.5, fontweight='bold', pad=8, loc='left', linespacing=1.2)

    cohort_names = {"hc": "HC\n(n=67)", "patient": "All Patients\n(n=26)", "pure_mdd": "Pure MDD\n(n=7)", "comorbid_mdd": "Comorbid MDD\n(n=19)"}
    hemi_headers = {"L": "Left Hippocampus (LH)", "R": "Right Hippocampus (RH; reflected)"}
    for column, hemi in enumerate(hemis):
        ax.text(.30 + .46 * column, .89, hemi_headers[hemi], transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    for row, cohort in enumerate(cohorts):
        bottom = .06 + (len(cohorts) - row - 1) * .205
        if show_cohorts:
            ax.text(.01, bottom + .085, cohort_names[cohort], transform=ax.transAxes, ha="left", va="center", fontsize=6.5)
        for column, hemi in enumerate(hemis):
            inset = ax.inset_axes([.10 + .46 * column, bottom, .38, .175])
            entry = lookup[(cohort, hemi)]
            axis = np.asarray(entry["axis"], dtype=float)
            transverse = np.asarray(entry["transverse"], dtype=float)
            axis, transverse = _unfolded_display_coordinates(axis, transverse, hemi)
            if gradient is None:
                colors = _categorical_map_values(entry)
                inset.scatter(axis, transverse, c=colors, s=2.8, cmap=ListedColormap(["#969696", "#2b83ba", "#e66101", "#2ca25f"]), vmin=0, vmax=3)
            else:
                values = np.asarray(entry["gradients"], dtype=float)[:, gradient]
                inset.scatter(axis, transverse, c=values, s=2.8, cmap="viridis", vmin=low, vmax=high)
            inset.set_xlim(*xlim)
            inset.set_ylim(*ylim)
            inset.set_aspect("equal", adjustable="box")
            inset.set_xticks([])
            inset.set_yticks([])
    ax.text(.5, .012, suffix, transform=ax.transAxes, ha="center", va="bottom", fontsize=6.5)


def _map_grid_panels(theme, maps):
    """Group all descriptive maps without selecting cohorts or hemispheres."""
    entries = [entry for entry in maps if entry.get("theme") == theme]
    present_branches = [branch for branch in BRANCHES if any(entry.get("branch") == branch for entry in entries)]
    panels = []
    if theme == "spatial":
        for branch in present_branches:
            for atlas in ATLASES:
                selected = [entry for entry in entries if entry.get("branch") == branch and entry.get("atlas") == atlas]
                if not selected:
                    raise ValueError(f"Missing descriptive spatial group maps: {branch}/{atlas}")
                panels.append(("map_grid", selected, f"{branch} | {atlas} | group atlas composition", None))
    elif theme == "continuous":
        for branch in present_branches:
            selected = [entry for entry in entries if entry.get("branch") == branch and entry.get("atlas") == "individual_gradient"]
            if not selected:
                raise ValueError(f"Missing descriptive gradient group maps: {branch}")
            for gradient in range(3):
                panels.append(("map_grid", selected, f"{branch} | individual gradients", gradient))
    return panels
def _figure_identity(root, theme, data, effect_panels=None, groups=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as patches
    import numpy as np

    stats = data['statistics.csv']
    subjects = data['subject_metrics.csv']

    METRIC_CONFIGS = [
        ('smax', 'Peak Template Similarity (Smax)', 'Best match among Ant/Mid/Post; higher = stronger functional identity', 'Peak Similarity Smax (r)'),
        ('margin', 'Separation Margin (Top1 − Top2)', 'Difference between top two matches; higher = sharper differentiation', 'Separation Margin (Δr)'),
        ('fixed_hcp_similarity', 'Fixed HCP Territory Similarity', 'Matched HCP template correlation in the same external HCP territory for every group', 'HCP Template Similarity (r)'),
        ('fixed_hcp_advantage', 'Fixed HCP Territory Advantage (Target − Max Competitor)', 'Matched HCP template relative to its strongest competitor in the same HCP territory', 'HCP Template Advantage (Δr)'),
        ('segregation', 'System Segregation (Cohen’s d)', 'Within- vs. between-system cohesion (Kember); lower = loss of modularity', 'System Segregation (d)'),
        ('boundary_sharpness', 'Boundary Sharpness (4mm Geodesic d)', 'Local cross-border transition steepness (Kember); lower = diffuse border', 'Boundary Sharpness (d)'),
    ]

    COHORT_META = [
        ('hc', 'Healthy Controls (HC, n=67)', '#2b83ba'),
        ('patient', 'All Patients (n=26)', '#fdae61'),
        ('pure_mdd', 'Pure MDD (n=7, uncomplicated unipolar)', '#2ca25f'),
        ('comorbid_mdd', 'Comorbid MDD (n=19, with anxiety/PTSD/SUD)', '#d7191c'),
    ]

    def _render_branch(branch, out_path):
        fig = plt.figure(figsize=(26, 25))
        gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.46, wspace=0.25)
        
        subregions = [('A', 'Ant'), ('B', 'Mid'), ('P', 'Post')]
        hemis = [('L', 'LH'), ('R', 'RH')]
        bar_width = 0.17
        hemi_centers = {'L': [0.0, 1.05, 2.10], 'R': [3.60, 4.65, 5.70]}
        
        stat_lookup = {}
        for r in stats:
            if _theme(r) == 'identity' and r.get('branch') == branch:
                stat_lookup[(r.get('hemi'), r.get('system'), r.get('metric'), r.get('comparison'))] = r
                
        rng = np.random.default_rng(42)
        
        for idx, (m_key, m_title, m_desc, y_label) in enumerate(METRIC_CONFIGS):
            row_idx = idx // 2
            col_idx = idx % 2
            ax = fig.add_subplot(gs[row_idx, col_idx])
            
            x_ticks = []
            x_tick_labels = []
            
            subregion_bar_xs = {}
            subregion_local_top = {}
            all_pts_in_panel = []
            all_bars_in_panel = []
            
            for h_code, h_name in hemis:
                for s_code, s_name in subregions:
                    center_x = hemi_centers[h_code][subregions.index((s_code, s_name))]
                    x_ticks.append(center_x)
                    x_tick_labels.append(f'{s_name}')
                    
                    n_cohorts = len(COHORT_META)
                    offsets = (np.arange(n_cohorts) - (n_cohorts - 1) / 2) * bar_width
                    
                    bar_xs = {}
                    local_subregion_pts = []
                    local_bar_tops = []
                    
                    for c_idx, (cohort_id, cohort_label, color) in enumerate(COHORT_META):
                        bx = center_x + offsets[c_idx]
                        bar_xs[cohort_id] = bx
                        
                        raw_pts = [
                            float(r['value']) for r in subjects
                            if _theme(r) == 'identity'
                            and r.get('branch') == branch
                            and r.get('hemi') == h_code
                            and r.get('system') == s_code
                            and r.get('metric') == m_key
                            and r.get('cohort') == cohort_id
                            and r.get('value') != ''
                            and r.get('value') is not None
                        ]
                        pts = [v for v in raw_pts if np.isfinite(v)]
                        if pts:
                            m = np.mean(pts)
                            sem = np.std(pts, ddof=1) / np.sqrt(len(pts)) if len(pts) > 1 else 0
                            local_subregion_pts.extend(pts)
                            local_bar_tops.append(m + sem)
                            all_pts_in_panel.extend(pts)
                            all_bars_in_panel.append(m)
                            
                            # Bar
                            ax.bar(bx, m, width=bar_width * 0.90, color=color, alpha=0.35, edgecolor=color, linewidth=1.3, zorder=2)
                            # SEM
                            ax.errorbar(bx, m, yerr=sem, fmt='none', ecolor='#222222', elinewidth=1.5, capsize=3.5, capthick=1.3, zorder=3)
                            # Jitter points
                            jit = rng.uniform(-bar_width * 0.22, bar_width * 0.22, size=len(pts))
                            ax.scatter(bx + jit, pts, s=15, color=color, edgecolors='white', linewidths=0.4, alpha=0.75, zorder=4)
                    
                    subregion_bar_xs[(h_code, s_code)] = bar_xs
                    # Local top for bracket positioning: strictly above the maximum scatter point / bar in this subregion
                    if local_subregion_pts:
                        bar_max = max(local_bar_tops) if local_bar_tops else 0
                        subregion_local_top[(h_code, s_code)] = max(max(local_subregion_pts), bar_max)
                    else:
                        subregion_local_top[(h_code, s_code)] = 0.5
            
            # Calculate panel-wide characteristic scale for brackets
            p98_panel = np.percentile(all_pts_in_panel, 98) if all_pts_in_panel else 1.0
            p02_panel = np.percentile(all_pts_in_panel, 2) if all_pts_in_panel else 0.0
            char_span = max(p98_panel - p02_panel, 0.2)
            y_step = char_span * 0.11
            
            max_bracket_y = -np.inf
            
            # Draw significance brackets locally for each subregion (placed strictly above all scatter points)
            for h_code, h_name in hemis:
                for s_code, s_name in subregions:
                    bar_xs = subregion_bar_xs.get((h_code, s_code), {})
                    if 'hc' not in bar_xs:
                        continue
                    hc_x = bar_xs['hc']
                    comps = [
                        ('patient', bar_xs.get('patient')),
                        ('pure_mdd', bar_xs.get('pure_mdd')),
                        ('comorbid_mdd', bar_xs.get('comorbid_mdd')),
                    ]
                    sig_items = []
                    for comp_id, target_x in comps:
                        if target_x is None:
                            continue
                        r = stat_lookup.get((h_code, s_code, m_key, comp_id))
                        if r and r.get('significance') in ('+', '*', '**', '***'):
                            g_val = r.get('g')
                            if g_val is not None and g_val != '':
                                g = float(g_val)
                                sig_items.append((comp_id, target_x, g, r['significance']))
                                
                    if sig_items:
                        curr_y = subregion_local_top[(h_code, s_code)] + y_step * 0.70
                        for comp_id, target_x, g, star in sig_items:
                            x1, x2 = min(hc_x, target_x), max(hc_x, target_x)
                            # bracket line
                            ax.plot([x1, x1, x2, x2], [curr_y - y_step * 0.25, curr_y, curr_y, curr_y - y_step * 0.25], color='#444444', lw=1.1, zorder=5)
                            # text without +/-: red font encodes positive, blue font encodes negative
                            font_color = '#d7191c' if g > 0 else '#2166ac'
                            txt = f'{star} ({abs(g):.2f})'
                            ax.text((x1 + x2) / 2, curr_y + y_step * 0.14, txt, ha='center', va='bottom', fontsize=11.0, fontweight='bold', color=font_color, zorder=6)
                            curr_y += y_step * 1.55
                        max_bracket_y = max(max_bracket_y, curr_y + y_step * 0.80)
                        
            # Visual divider between LH and RH
            divider_x = (hemi_centers['L'][-1] + hemi_centers['R'][0]) / 2
            ax.axvline(divider_x, color='#999999', linestyle='--', linewidth=1.1, alpha=0.6, zorder=1)
            
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_tick_labels, fontsize=12.5, fontweight='bold')
            ax.set_xlim(-0.55, 6.25)
            ax.tick_params(axis='y', labelsize=12.0)
            
            # Adaptive y-limits: ensure bars have substantial, readable height while giving brackets breathing room
            all_max = max(all_pts_in_panel) if all_pts_in_panel else 1.0
            all_min = min(all_pts_in_panel) if all_pts_in_panel else 0.0
            overall_top = max(all_max, max_bracket_y) if np.isfinite(max_bracket_y) else all_max
            
            if m_key == 'fixed_hcp_advantage':
                limit = max(abs(all_min), abs(overall_top), 0.35) * 1.18
                ax.set_ylim(-limit, limit)
                ax.axhline(0, color='#666666', linestyle=':', linewidth=1.0, zorder=1)
            elif m_key == 'fixed_hcp_similarity':
                ax.set_ylim(min(-0.12, all_min * 1.1), max(overall_top * 1.16, 0.66))
                ax.axhline(0, color='#666666', linestyle=':', linewidth=0.8, zorder=1)
            elif m_key in ('smax', 'margin'):
                ax.set_ylim(0.0, max(overall_top * 1.15, 0.58 if m_key == 'smax' else 0.48))
            elif m_key == 'segregation':
                ax.set_ylim(0.0, max(overall_top * 1.15, 3.2))
            elif m_key == 'boundary_sharpness':
                ax.set_ylim(0.0, max(overall_top * 1.12, 5.0))
            else:
                ax.set_ylim(min(0.0, all_min * 1.1), overall_top * 1.15)
                
            y_top = ax.get_ylim()[1]
            y_bot = ax.get_ylim()[0]
            y_span = y_top - y_bot
            lh_mid = (hemi_centers['L'][0] + hemi_centers['L'][-1]) / 2
            rh_mid = (hemi_centers['R'][0] + hemi_centers['R'][-1]) / 2
            ax.text(lh_mid, y_top + y_span * 0.02, '◀ Left Hippocampus (LH)', ha='center', va='bottom', fontsize=12.5, fontweight='bold', color='#1b7837')
            ax.text(rh_mid, y_top + y_span * 0.02, 'Right Hippocampus (RH) ▶', ha='center', va='bottom', fontsize=12.5, fontweight='bold', color='#762a83')
            
            ax.set_ylabel(y_label, fontsize=12.5, fontweight='bold')
            ax.set_title(f'{m_title}\n{m_desc}', fontsize=13.0, fontweight='bold', pad=22, linespacing=1.2)
            ax.grid(axis='y', linestyle=':', alpha=0.45, zorder=0)

        # Full-width reference card beneath the six metric panels.
        ax_legend = fig.add_subplot(gs[3, :])
        ax_legend.set_axis_off()
        ax_legend.set_xlim(0, 100)
        ax_legend.set_ylim(0, 100)
        
        card_bg = patches.FancyBboxPatch((1.5, 1.5), 97, 97, boxstyle='round,pad=1.2', facecolor='#fafafa', edgecolor='#cccccc', lw=1.5)
        ax_legend.add_patch(card_bg)
        
        ax_legend.text(50, 94.5, 'Metrics & Statistical Reference Guide', ha='center', va='center', fontsize=13.5, fontweight='bold', color='#111111')
        ax_legend.plot([4, 96], [90.5, 90.5], color='#dddddd', lw=1.2)
        
        # Top Left: Cohorts
        ax_legend.text(4, 85.5, 'Participant Cohorts (Mean ± SEM & dots):', ha='left', va='center', fontsize=11.5, fontweight='bold', color='#222222')
        cohorts = [
            ('#2b83ba', 'Healthy Controls (HC, n=67)', 79.5),
            ('#fdae61', 'All Patients (n=26)', 74.0),
            ('#2ca25f', 'Pure MDD (n=7, uncomplicated unipolar)', 68.5),
            ('#d7191c', 'Comorbid MDD (n=19, with anxiety/PTSD)', 63.0),
        ]
        for col, label, y_pos in cohorts:
            swatch = patches.Rectangle((5, y_pos - 1.8), 3.0, 3.8, facecolor=col, edgecolor='#333333', lw=0.8, alpha=0.85)
            ax_legend.add_patch(swatch)
            ax_legend.text(9.5, y_pos, label, ha='left', va='center', fontsize=10.5, fontweight='semibold', color='#111111')
            
        # Top Right: Statistics
        ax_legend.text(54, 85.5, 'Welch’s t-test Contrasts vs. HC (Hedges’ g):', ha='left', va='center', fontsize=11.5, fontweight='bold', color='#222222')
        red_box = patches.FancyBboxPatch((54, 77.5), 14, 3.8, boxstyle='round,pad=0.3', facecolor='#fee8e8', edgecolor='#d7191c', lw=0.9)
        ax_legend.add_patch(red_box)
        ax_legend.text(61, 79.4, '0.42 / * 0.58', ha='center', va='center', fontsize=10.2, fontweight='bold', color='#d7191c')
        ax_legend.text(70, 79.4, 'RED = Positive (Patient > HC, elevated)', ha='left', va='center', fontsize=10.5, color='#222222')
        
        blue_box = patches.FancyBboxPatch((54, 72.0), 14, 3.8, boxstyle='round,pad=0.3', facecolor='#e8f2fa', edgecolor='#2166ac', lw=0.9)
        ax_legend.add_patch(blue_box)
        ax_legend.text(61, 73.9, '0.49 / *** 0.69', ha='center', va='center', fontsize=10.2, fontweight='bold', color='#2166ac')
        ax_legend.text(70, 73.9, 'BLUE = Negative (Patient < HC, reduced)', ha='left', va='center', fontsize=10.5, color='#222222')
        
        ax_legend.text(54, 67.0, 'Levels: + p < 0.10 (trend) · * p < 0.05 · ** p < 0.01 · *** p < 0.001', ha='left', va='center', fontsize=10.2, color='#444444')
        ax_legend.text(54, 62.5, 'Effect Size: Hedges’ g (unbiased; unannotated: p ≥ 0.10)', ha='left', va='center', fontsize=10.2, color='#555555')
        
        ax_legend.plot([4, 96], [58.5, 58.5], color='#dddddd', lw=1.2)
        
        # Bottom Section: six identity metrics.
        ax_legend.text(4, 54.0, 'Key Functional Organization Metrics Guide (Biological & Methodological Definitions):', ha='left', va='center', fontsize=11.5, fontweight='bold', color='#111111')
        
        metrics_guide = [
            ('• Peak Similarity (Smax):', 'Max correlation with canonical templates; higher = stronger functional identity.', 47.0),
            ('• Separation Margin (Δr):', 'Top1 − Top2 match difference; higher = sharper differentiation between competitor networks.', 40.0),
            ('• Fixed HCP Similarity:', 'Matched HCP template correlation in the same external HCP territory for every group.', 33.0),
            ('• Fixed HCP Advantage (Δr):', 'Matched HCP template minus the strongest competitor in the same external HCP territory.', 26.0),
            ('• System Segregation (d):', 'Within- vs. between-system cohesion (Kember et al.); lower = loss of functional modularity.', 19.0),
            ('• Boundary Sharpness (d):', 'Cross-border transition steepness over 4mm geodesic (Kember); lower = diffuse, blurry border.', 12.0),
        ]
        
        for m_name_label, desc, y_pos in metrics_guide:
            ax_legend.text(4, y_pos, m_name_label, ha='left', va='center', fontsize=10.5, fontweight='bold', color='#1b7837')
            ax_legend.text(28.5, y_pos, desc, ha='left', va='center', fontsize=10.2, color='#333333')

        branch_title = 'Standard Preprocessing Pipeline' if branch == 'standard' else 'Cortex-Regressed Preprocessing Pipeline (Ctx-Reg)'
        fig.suptitle(
            f'Hippocampal Functional Identity & Differentiation Across Longitudinal Subregions\n[{branch_title}]\n'
            'Grouped Bar & Scatter Distributions (Ant · Mid · Post) | Brackets: Welch contrasts vs. HC with Hedges’ g effect size',
            fontsize=16.5, fontweight='bold', y=0.993, linespacing=1.28
        )
        
        fig.subplots_adjust(top=0.932, bottom=0.038, left=0.055, right=0.975)
        fig.savefig(out_path, dpi=180, facecolor='white')
        plt.close(fig)
        return out_path

    p_std = _render_branch('standard', root / 'hipp_functional_identity_standard.png')
    p_ctx = _render_branch('ctxreg', root / 'hipp_functional_identity_ctxreg.png')
    return [p_std, p_ctx]


def _fingerprint_network_text_color(rgb):
    """Darken light canonical-network hues so x-tick text stays legible on white."""
    try:
        r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
    except (TypeError, ValueError, IndexError):
        return '#111111'
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    if lum > 0.60:
        f = 0.62
        r, g, b = r * f, g * f, b * f
    r = max(0.0, min(1.0, r))
    g = max(0.0, min(1.0, g))
    b = max(0.0, min(1.0, b))
    return '#%02x%02x%02x' % (round(r * 255), round(g * 255), round(b * 255))


def _matrix_fingerprint(ax, rows, rowkeys, colkeys, rowfn, colfn, title):
    import numpy as np
    from matplotlib import colormaps
    try:
        _net_palette = _palette()
    except (OSError, ValueError, KeyError):
        _net_palette = {}
    values = np.full((len(rowkeys), len(colkeys)), np.nan)
    lookup = {}
    for row in rows:
        key = (rowfn(row), colfn(row))
        if key in lookup:
            raise ValueError(f'Duplicate report matrix cell in {title}: {key}')
        lookup[key] = row
    for i, rkey in enumerate(rowkeys):
        for j, ckey in enumerate(colkeys):
            row = lookup.get((rkey, ckey), {})
            g = number(row.get('g'))
            if g is not None:
                values[i, j] = g
                text_color = 'white' if abs(g) >= 0.8 else '#111111'
            else:
                text_color = '#555555'
            label = effect_label(row)
            weight = 'bold' if any(s in label for s in ('*', '+')) else 'normal'
            ax.text(j, i, label, ha='center', va='center', fontsize=9.0, color=text_color, fontweight=weight)

    cmap = colormaps['RdBu_r'].copy()
    cmap.set_bad('#dedede')
    ax.imshow(values, cmap=cmap, vmin=-1.5, vmax=1.5, aspect='equal', interpolation='none')
    ax.set_aspect('equal', adjustable='box')

    # Subregion headers above columns (shortened so larger fonts still fit square cells)
    ax.text(3, -1.05, 'Anterior (Ant / rA)', ha='center', va='bottom', fontsize=11, fontweight='bold', color='#1f78b4')
    ax.text(10, -1.05, 'Middle (Mid / rB)', ha='center', va='bottom', fontsize=11, fontweight='bold', color='#d95f02')
    ax.text(17, -1.05, 'Posterior (Post / rP)', ha='center', va='bottom', fontsize=11, fontweight='bold', color='#1b9e77')

    # Dividing lines between systems and hemispheres
    ax.axvline(6.5, color='#111111', lw=1.8)
    ax.axvline(13.5, color='#111111', lw=1.8)
    ax.axhline(2.5, color='#111111', lw=1.8)

    # Minor grid between cells
    ax.set_xticks(np.arange(-.5, len(colkeys), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(rowkeys), 1), minor=True)
    ax.grid(which='minor', color='#ffffff', linestyle='-', linewidth=0.9, alpha=0.9)
    ax.tick_params(which='minor', bottom=False, left=False)

    # X-ticks (colored by canonical network) and Y-ticks
    ax.set_xticks(range(len(colkeys)))
    ax.set_xticklabels([k[1] for k in colkeys], rotation=55, ha='right', fontsize=10, fontweight='bold')
    for _lbl, _ckey in zip(ax.get_xticklabels(), colkeys):
        _net = _ckey[1] if isinstance(_ckey, tuple) and len(_ckey) > 1 else str(_ckey)
        _rgb = _net_palette.get(_net)
        if _rgb is not None:
            _lbl.set_color(_fingerprint_network_text_color(_rgb))
    ax.set_yticks(range(len(rowkeys)))
    ax.set_yticklabels([_format_row_label(k) for k in rowkeys], fontsize=10)

    parts = [p.strip() for p in title.split('|')]
    branch = parts[0].strip()
    atlas = parts[1].replace(': network fingerprint g', '').strip()
    b_name = {'standard': 'Standard', 'ctxreg': 'Ctx-Reg'}.get(branch, branch)
    a_name = {'tm_dominant': 'Group Dominant', 'de_novo': 'De Novo', 'wta': 'WTA'}.get(atlas, atlas)
    # Single-line panel title; pipeline/metric context lives in the suptitle.
    ax.set_title(f'{b_name} · {a_name}', fontsize=13, fontweight='bold', pad=30, linespacing=1.2)


def _figure_fingerprints(root, theme, data, panels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    fig = plt.figure(figsize=(36, 11.5))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.40)

    for idx, panel in enumerate(panels):
        row_idx = idx // 3
        col_idx = idx % 3
        ax = fig.add_subplot(gs[row_idx, col_idx])
        kind, *args = panel
        _matrix_fingerprint(ax, *args)

    fig.suptitle(
        'Iowa Hippocampal Functional Organization — Theme 3: Neocortical Network Fingerprints\n'
        'Patient − HC cortical-network FC (Fisher’s z); Hedges’ g; + .05 ≤ p < .1; * p < .05; ** p < .01; *** p < .001 (Two-sided Welch t-tests; exact g retained)\n'
        'Canonical Cortical Networks: DMN (Default Mode) · SMN (Somatomotor) · VAN (Ventral Attention) · DAN (Dorsal Attention) · CON (Control) · LAN (Language) · VN (Visual)\n'
        'Hippocampal Subregions: Ant = Anterior (VAN-coupled) · Mid = Middle (SMN-coupled) · Post = Posterior (DMN-coupled)',
        fontsize=15, fontweight='bold', y=0.975
    )

    colorax = fig.add_axes((0.30, 0.04, 0.40, 0.015))
    cb = fig.colorbar(ScalarMappable(norm=Normalize(-1.5, 1.5), cmap='RdBu_r'), cax=colorax, orientation='horizontal')
    cb.ax.set_title(
        'Colorbar: Hedges’ g effect size (Patient − HC cortical-network FC, Fisher’s z)',
        fontsize=13, fontweight='bold', pad=10
    )
    cb.set_label('Shared scale [-1.5, +1.5]; Blue: Patient < HC, Red: Patient > HC; ×: non-computable', fontsize=11)
    cb.ax.tick_params(labelsize=10)

    fig.subplots_adjust(top=0.885, bottom=0.135, left=0.07, right=0.98)
    path = root / 'hipp_network_fingerprints.png'
    fig.savefig(path, dpi=180, facecolor='white')
    plt.close(fig)
    return path


def _figure_continuous(root, theme, data, effect_panels, curve_panels, map_panels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    fig = plt.figure(figsize=(28, 32))
    gs_outer = gridspec.GridSpec(4, 1, figure=fig, height_ratios=[1.15, 0.95, 0.95, 0.95], hspace=0.38)
    gs_row0 = gs_outer[0].subgridspec(1, 5, wspace=0.48)
    gs_row1 = gs_outer[1].subgridspec(1, 4, wspace=0.32)
    gs_row2 = gs_outer[2].subgridspec(1, 3, wspace=0.30)
    gs_row3 = gs_outer[3].subgridspec(1, 3, wspace=0.30)

    # 1. Effect panels (5 panels)
    p_axis = next(p for p in effect_panels if 'axis_correlation' in p[6])
    p_ref = next(p for p in effect_panels if 'reference_correlation' in p[6])
    p_rr_grad = next(p for p in effect_panels if 'robust_range' in p[6] and 'individual_gradient' in p[6])
    p_rr_tm = next(p for p in effect_panels if 'robust_range' in p[6] and 'tm_individual' in p[6])
    p_slope = next(p for p in effect_panels if 'slope' in p[6])
    ordered_effects = [p_axis, p_ref, p_rr_grad, p_rr_tm, p_slope]

    for col_idx, panel in enumerate(ordered_effects):
        ax = fig.add_subplot(gs_row0[0, col_idx])
        kind, *args = panel
        _matrix(ax, *args)

    # 2. Curve panels (4 panels)
    c_std_L = next(p for p in curve_panels if 'standard' in p[2] and 'L' in p[2])
    c_std_R = next(p for p in curve_panels if 'standard' in p[2] and 'R' in p[2])
    c_ctx_L = next(p for p in curve_panels if 'ctxreg' in p[2] and 'L' in p[2])
    c_ctx_R = next(p for p in curve_panels if 'ctxreg' in p[2] and 'R' in p[2])
    ordered_curves = [c_std_L, c_std_R, c_ctx_L, c_ctx_R]

    for col_idx, panel in enumerate(ordered_curves):
        ax = fig.add_subplot(gs_row1[0, col_idx])
        kind, *args = panel
        _draw_curves(ax, *args, show_legend=(col_idx == 0), show_ylabel=(col_idx == 0))

    # 3. Standard Gradients (3 panels)
    m_std_g1 = next(p for p in map_panels if 'standard' in p[2] and p[3] == 0)
    m_std_g2 = next(p for p in map_panels if 'standard' in p[2] and p[3] == 1)
    m_std_g3 = next(p for p in map_panels if 'standard' in p[2] and p[3] == 2)
    for col_idx, panel in enumerate([m_std_g1, m_std_g2, m_std_g3]):
        ax = fig.add_subplot(gs_row2[0, col_idx])
        kind, *args = panel
        _draw_map_grid(ax, *args, show_cohorts=(col_idx == 0))

    # 4. Ctx-Reg Gradients (3 panels)
    m_ctx_g1 = next(p for p in map_panels if 'ctxreg' in p[2] and p[3] == 0)
    m_ctx_g2 = next(p for p in map_panels if 'ctxreg' in p[2] and p[3] == 1)
    m_ctx_g3 = next(p for p in map_panels if 'ctxreg' in p[2] and p[3] == 2)
    for col_idx, panel in enumerate([m_ctx_g1, m_ctx_g2, m_ctx_g3]):
        ax = fig.add_subplot(gs_row3[0, col_idx])
        kind, *args = panel
        _draw_map_grid(ax, *args, show_cohorts=(col_idx == 0))

    fig.suptitle(
        'Iowa Hippocampal Functional Organization — Theme 4: Continuous Topography & Gradients\n'
        'Patient − HC; Hedges’ g; + .05 ≤ p < .1; * p < .05; ** p < .01; *** p < .001 (Two-sided Welch t-tests; exact g retained)\n'
        'Row 1: Topological Effect Sizes (g) | Row 2: AP Longitudinal Correlation Profiles | Rows 3 & 4: Unfolded Connectopic Gradients (512 vertices)',
        fontsize=12, fontweight='bold', y=0.985
    )

    colorax = fig.add_axes((0.32, 0.015, 0.36, 0.007))
    cb = fig.colorbar(ScalarMappable(norm=Normalize(-1.5, 1.5), cmap='RdBu_r'), cax=colorax, orientation='horizontal')
    cb.set_label('Hedges’ g (Patient − HC; shared scale [-1.5, +1.5]; Blue: Patient < HC, Red: Patient > HC)', fontsize=9)

    fig.subplots_adjust(top=0.94, bottom=0.045, left=0.055, right=0.985)
    path = root / 'hipp_continuous_gradients.png'
    fig.savefig(path, dpi=180, facecolor='white')
    plt.close(fig)
    return path


def _figure_spatial(root, theme, data, effect_panels, curve_panels, boundary_panels, map_panels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    fig = plt.figure(figsize=(28, 44))
    gs_outer = gridspec.GridSpec(4, 1, figure=fig, height_ratios=[1.10, 2.50, 0.70, 1.30], hspace=0.35)
    gs_row0 = gs_outer[0].subgridspec(1, 5, wspace=0.48)
    gs_row1 = gs_outer[1].subgridspec(4, 4, hspace=0.38, wspace=0.28)
    gs_row2 = gs_outer[2].subgridspec(1, 4, wspace=0.22)
    gs_row3 = gs_outer[3].subgridspec(2, 3, hspace=0.35, wspace=0.28)

    # 1. Effect panels (5 panels)
    p_frac = next(p for p in effect_panels if 'area_fraction' in p[6])
    p_mm2 = next(p for p in effect_panels if 'area_mm2' in p[6])
    p_cent = next(p for p in effect_panels if 'centroid' in p[6])
    p_vert = next(p for p in effect_panels if 'vertex_count' in p[6])
    p_cortex = next(p for p in effect_panels if 'area_percent' in p[6])
    ordered_effects = [p_frac, p_mm2, p_cent, p_vert, p_cortex]

    for col_idx, panel in enumerate(ordered_effects):
        ax = fig.add_subplot(gs_row0[0, col_idx])
        kind, *args = panel
        _matrix(ax, *args)

    # 2. Curve panels: 4 conditions x 4 atlases = 16 panels
    conditions = [('standard', 'L'), ('standard', 'R'), ('ctxreg', 'L'), ('ctxreg', 'R')]
    atlases = ['tm_individual', 'tm_dominant', 'de_novo', 'wta']
    for row_idx, (branch, hemi) in enumerate(conditions):
        for col_idx, atlas in enumerate(atlases):
            ax = fig.add_subplot(gs_row1[row_idx, col_idx])
            p_curve = next(p for p in curve_panels if p[2].startswith(f'{branch} | {hemi} | {atlas}'))
            kind, *args = p_curve
            _draw_curves(ax, *args, show_legend=(row_idx == 0 and col_idx == 0), show_ylabel=(col_idx == 0))

    # 3. Boundary panels: 4 panels
    for col_idx, (branch, hemi) in enumerate(conditions):
        ax = fig.add_subplot(gs_row2[0, col_idx])
        p_bound = next(p for p in boundary_panels if p[2].startswith(f'{branch} | {hemi}'))
        kind, *args = p_bound
        _draw_boundaries(ax, *args, show_ylabel=(col_idx == 0))

    # 4. Map panels: 2 branches x 3 atlases = 6 panels
    map_atlases = ['tm_dominant', 'de_novo', 'wta']
    for row_idx, branch in enumerate(['standard', 'ctxreg']):
        for col_idx, atlas in enumerate(map_atlases):
            ax = fig.add_subplot(gs_row3[row_idx, col_idx])
            p_map = next(p for p in map_panels if p[2].startswith(f'{branch} | {atlas}'))
            kind, *args = p_map
            _draw_map_grid(ax, *args, show_cohorts=(col_idx == 0))

    fig.suptitle(
        'Iowa Hippocampal Functional Organization — Theme 2: Spatial Organization & Topography\n'
        'Patient − HC; Hedges’ g; + .05 ≤ p < .1; * p < .05; ** p < .01; *** p < .001 (Two-sided Welch t-tests; exact g retained)\n'
        'Tier 1: Morphometric Effect Sizes | Tier 2: AP Subfield Area Fraction Profiles (4x4) | Tier 3: Transition Midpoints | Tier 4: Descriptive Group Surface Composition',
        fontsize=12, fontweight='bold', y=0.988
    )

    colorax = fig.add_axes((0.32, 0.012, 0.36, 0.005))
    cb = fig.colorbar(ScalarMappable(norm=Normalize(-1.5, 1.5), cmap='RdBu_r'), cax=colorax, orientation='horizontal')
    cb.set_label('Hedges’ g (Patient − HC; shared scale [-1.5, +1.5]; Blue: Patient < HC, Red: Patient > HC)', fontsize=9)

    fig.subplots_adjust(top=0.955, bottom=0.035, left=0.055, right=0.985)
    path = root / 'hipp_spatial_organization.png'
    fig.savefig(path, dpi=180, facecolor='white')
    plt.close(fig)
    return path


def _figure(root, theme, data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    stats = [r for r in data['statistics.csv'] if _theme(r) == theme]
    panels = _effect_panels(stats, theme)
    if theme == 'identity':
        # Exactly six (metric, atlas) panels dispatch to the dedicated per-branch
        # bar+scatter renderer. Anything else fails closed here so the generic
        # grid can never again silently overwrite the identity figures.
        if len(panels) != 6:
            raise ValueError(
                'identity theme requires exactly six (metric, atlas) effect panels '
                f'for the dedicated bar+scatter renderer, got {len(panels)}'
            )
        return _figure_identity(root, theme, data, panels)
    if theme == 'fingerprints':
        if len(panels) == 6:
            return _figure_fingerprints(root, theme, data, panels)
    if theme in ('spatial','continuous'):
        curves = []
        groups = defaultdict(list)
        for row in data['curves.csv']:
            if _theme(row) == theme:
                groups[(row['branch'],row['hemi'],row['atlas'],row['metric'])].append(row)
        for key, rows in sorted(groups.items()):
            curves.append(('curves',rows,' | '.join(key)))
        boundaries = []
        if theme == 'spatial':
            boundary_rows = data['boundary_positions.csv']
            for branch in BRANCHES:
                for hemi in ('L', 'R'):
                    selected = [
                        row for row in boundary_rows
                        if row['branch'] == branch and row['hemi'] == hemi
                    ]
                    if selected:
                        boundaries.append(
                            ('boundaries', selected, f'{branch} | {hemi} | TM boundary edge positions')
                        )
        maps = _map_grid_panels(theme, data['spatial_maps.json'])
        if theme == 'continuous' and len(panels) == 5 and len(curves) == 4 and len(maps) == 6:
            return _figure_continuous(root, theme, data, panels, curves, maps)
        if theme == 'spatial' and len(panels) == 5 and len(curves) == 16 and len(boundaries) == 4 and len(maps) == 6:
            return _figure_spatial(root, theme, data, panels, curves, boundaries, maps)
        panels.extend(curves)
        panels.extend(boundaries)
        panels.extend(maps)
    if not panels:
        panels = [('empty',)]
    columns = 3 if theme == 'fingerprints' else 2
    nrows = math.ceil(len(panels)/columns)
    fig, axes = plt.subplots(nrows,columns,figsize=(24 if theme=='fingerprints' else 18, max(5,nrows*4.8)), squeeze=False)
    for ax,panel in zip(axes.flat,panels):
        kind,*args = panel
        if kind == 'effects': _matrix(ax,*args)
        elif kind == 'curves': _draw_curves(ax,*args)
        elif kind == 'boundaries': _draw_boundaries(ax,*args)
        elif kind == 'map_grid': _draw_map_grid(ax,*args)
        else: ax.text(.5,.5,'No derived rows declared for this theme',ha='center'); ax.set_axis_off()
    for ax in list(axes.flat)[len(panels):]: ax.set_axis_off()
    fig.suptitle(f'Iowa hippocampal functional organization — {theme}\nPatient − HC; Hedges g; × missing; + .05 ≤ p < .1; * p < .05; ** p < .01; *** p < .001\nUncorrected prespecified Welch tests; arrows mark color clipping, exact g retained. Kember is a separate identity metric.',fontsize=12)
    fig.tight_layout(rect=(0,.035,1,.965), h_pad=3,w_pad=3)
    colorax=fig.add_axes((.35,.009,.3,.008))
    fig.colorbar(ScalarMappable(norm=Normalize(-1.5,1.5),cmap='RdBu_r'),cax=colorax,orientation='horizontal',label='Hedges g (shared scale)')
    theme_filename_map = {
        'spatial': 'hipp_spatial_organization.png',
        'fingerprints': 'hipp_network_fingerprints.png',
        'continuous': 'hipp_continuous_gradients.png',
    }
    path = root / theme_filename_map[theme]
    fig.savefig(path, dpi=180, facecolor='white')
    plt.close(fig)
    return path


def _static_pdf(root, figures):
    """Create a paginated, static companion from the exact rendered theme PNGs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from PIL import Image

    expected = (
        root / "hipp_functional_identity_standard.png",
        root / "hipp_functional_identity_ctxreg.png",
        root / "hipp_spatial_organization.png",
        root / "hipp_network_fingerprints.png",
        root / "hipp_continuous_gradients.png",
    )
    ordered = tuple(Path(figure) for figure in figures)
    if ordered != expected:
        raise ValueError("static PDF requires the exact five newly rendered theme PNGs in theme order")
    if not all(path.is_file() and path.stat().st_size > 0 for path in ordered):
        raise ValueError("static PDF cannot use a missing or empty rendered theme PNG")

    path = root / STATIC_PDF
    portrait = (8.27, 11.69)
    landscape = (11.69, 8.27)
    with PdfPages(path) as pdf:
        metadata = pdf.infodict()
        metadata["Title"] = "Iowa hippocampal functional organization"
        metadata["Subject"] = "Static companion to the hash-validated Iowa report"
        metadata["Author"] = "Iowa hippocampal functional-organization workflow"

        cover = plt.figure(figsize=portrait)
        cover.text(.09, .91, "Iowa hippocampal functional organization", fontsize=20, fontweight="bold", va="top")
        cover.text(.09, .855, "Static PDF companion", fontsize=13, va="top")
        cover.text(
            .09,
            .78,
            "\n".join(
                (
                    "Scope",
                    "Kong2022 canonical7 | 36P | 2 mm smoothing | 400 cortical ROIs",
                    "Separate standard and ctxreg branches; left and right hippocampi",
                    "Patient (26), Pure MDD (7), and Comorbid MDD (19), each versus HC (67)",
                    "",
                    "Reading guide",
                    "Effects are patient minus HC Hedges g. Welch tests are two-sided, uncorrected,",
                    "and use no covariates. + marks .05 <= p < .10; *, **, and *** mark p < .05,",
                    "p < .01, and p < .001. A gray x marks a declared non-computable estimate.",
                    "",
                    "Contents",
                    "The following pages contain the five complete theme figures, split into standard",
                    "print pages when needed. Exact group means, SDs, N, differences, CIs, g, and p",
                    "remain available in statistics.csv and roi_statistics.csv alongside this PDF.",
                )
            ),
            fontsize=10.5,
            va="top",
            linespacing=1.45,
        )
        cover.text(.09, .08, "Generated only from the same manifest-hash-validated derived statistics as the PNG and HTML reports.", fontsize=8, va="bottom")
        pdf.savefig(cover, dpi=300)
        plt.close(cover)

        for source in ordered:
            theme_name = source.stem.replace('_', ' ').title()
            with Image.open(source) as image:
                width, height = image.size
                if width <= 0 or height <= 0:
                    raise ValueError(f"static PDF cannot use an invalid image: {source}")
                page_width, page_height = landscape if width > height else portrait
                left, right, bottom, heading = .42, .42, .40, .74
                usable_width = page_width - left - right
                usable_height = page_height - bottom - heading
                chunk_height = max(1, int(width * usable_height / usable_width))
                parts = math.ceil(height / chunk_height)
                for part in range(parts):
                    start = part * chunk_height
                    stop = min(height, start + chunk_height)
                    page = plt.figure(figsize=(page_width, page_height))
                    page.text(
                        .5,
                        1 - .28 / page_height,
                        f"{theme_name} ({part + 1}/{parts})",
                        ha="center",
                        va="top",
                        fontsize=11,
                        fontweight="bold",
                    )
                    axes = page.add_axes(
                        (left / page_width, bottom / page_height, usable_width / page_width, usable_height / page_height)
                    )
                    axes.imshow(image.crop((0, start, width, stop)), interpolation="none", aspect="auto")
                    axes.set_axis_off()
                    pdf.savefig(page, dpi=300)
                    plt.close(page)
    if not path.is_file() or not path.read_bytes().startswith(b"%PDF-"):
        raise ValueError(f"static PDF was not written correctly: {path}")
    return path

def _html(root, data, figures=None):
    rows = data['statistics.csv']
    for row in rows: row['display'] = effect_label(row)
    payload = json.dumps({'statistics':rows,'rois':data['roi_registry.json'],'roiStatistics':data['roi_statistics.csv'],'colors':_palette()}, ensure_ascii=False).replace('</','<\\/')
    content = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Iowa hippocampal functional organization</title>
<style>body{font:14px system-ui;margin:24px;color:#18202b}select,input{padding:6px;margin:8px}table{border-collapse:collapse;font-size:12px}td,th{padding:7px;border:1px solid #ddd}th{position:sticky;top:0;background:white}section{overflow:auto;max-height:650px;margin:20px 0}.cell{min-width:70px;text-align:center}.missing{background:#ddd}img{max-width:100%}label{white-space:nowrap}.legend{display:flex;gap:14px;flex-wrap:wrap}</style>
<h1>Iowa hippocampal functional organization</h1><p>Patient − HC; Hedges g uses the shared zero-centered range [−1.5, 1.5]. Arrows indicate clipped colors; exact effects remain displayed. × is missing, distinct from zero. + .05 ≤ p &lt; .1; * p &lt; .05; ** p &lt; .01; *** p &lt; .001. All prespecified comparisons are shown, without significance selection. P values are uncorrected Welch tests.</p>
<div id="filters"></div><div class="legend" id="networks"></div><p id="count"></p><section id="matrix"></section><h2>Exact estimates</h2><section id="estimates"></section><p>Every valid TM boundary edge position is retained in <a href="boundary_positions.csv">boundary_positions.csv</a>; no single boundary is inferred from a discontinuous map.</p>
<h2>All 400 cortical ROIs: localization and annotations</h2><p>When spatial coordinates are unavailable, localization uses hemisphere and anatomical labels from the authoritative ROI registry; it is not a reconstructed cortical surface. Search retains all matching ROIs.</p>
<label>ROI search <input id="roi-search" placeholder="ROI, network, hemisphere, anatomy"></label><label><input id="show-all-roi-effects" type="checkbox"> render all matching ROI effects</label>
<div id="localization"></div><section id="roi-table"></section>
<h3>ROI effect estimates</h3><p id="roi-effect-count"></p><section id="roi-effects"></section>

<h2>Static PDF</h2><p><a href="iowa_hipp_functional_organization_report.pdf">Open the static PDF companion</a> for direct reading or printing. It contains the four theme figures; exact estimates remain in the CSV files.</p><h2>Publication figures</h2>FIGURES
<script id="data" type="application/json">PAYLOAD</script><script>
const D=JSON.parse(document.getElementById('data').textContent);const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fields=['branch','hemi','atlas','metric'];const select={};
for(const field of fields){const l=document.createElement('label');l.textContent=field;const s=document.createElement('select');s.id='filter-'+field;for(const val of ['',...new Set(D.statistics.map(r=>r[field]))]){const o=document.createElement('option');o.value=val;o.textContent=val||'All';s.append(o)}l.append(s);document.getElementById('filters').append(l);select[field]=s;s.onchange=render}
const finite=x=>x!==''&&x!==null&&x!==undefined&&Number.isFinite(+x);const color=g=>{if(!finite(g))return '#ddd';const t=Math.min(Math.abs(+g)/1.5,1);const end=+g>=0?[178,24,43]:[33,102,172];return 'rgb('+end.map(v=>Math.round(255*(1-t)+v*t)).join(',')+')'};
const tip=r=>['mean_patient','sd_patient','n_patient','mean_hc','sd_hc','n_hc','difference','ci_low','ci_high','g','p','df','reason'].map(k=>k+': '+(r[k]??'')).join('\n');
function render(){const rows=D.statistics.filter(r=>fields.every(k=>!select[k].value||r[k]===select[k].value));document.getElementById('count').textContent=rows.length+' estimates';const groups=new Map;for(const r of rows){const key=['theme','branch','hemi','atlas','system','metric','target'].map(k=>r[k]).join(' / ');if(!groups.has(key))groups.set(key,{});groups.get(key)[r.comparison]=r}const comps=['patient','pure_mdd','comorbid_mdd'];let matrix='<table><tr><th>Theme / branch / hemisphere / atlas / system / metric / target</th>'+comps.map(c=>'<th>'+c+' − HC</th>').join('')+'</tr>';for(const [key,cells] of groups){matrix+='<tr><td>'+esc(key)+'</td>'+comps.map(c=>{const r=cells[c];return '<td class="cell '+(!r||!finite(r.g)?'missing':'')+'" style="background:'+color(r?.g)+'" title="'+esc(r?tip(r):'Missing declared comparison')+'">'+esc(r?r.display:'×')+'</td>'}).join('')+'</tr>'}document.getElementById('matrix').innerHTML=matrix+'</table>';const cols=['theme',...fields,'system','target','comparison','mean_patient','sd_patient','n_patient','mean_hc','sd_hc','n_hc','difference','ci_low','ci_high','g','p','df','reason'];document.getElementById('estimates').innerHTML=table(rows,cols)}
function table(rows,cols){return '<table><tr>'+cols.map(k=>'<th>'+esc(k)+'</th>').join('')+'</tr>'+rows.map(r=>'<tr title="'+esc(tip(r))+'">'+cols.map(k=>'<td>'+esc(typeof r[k]==='object'?JSON.stringify(r[k]):r[k])+'</td>').join('')+'</tr>').join('')+'</table>'}
function rois(){const q=document.getElementById('roi-search').value.toLowerCase();const rows=D.rois.filter(r=>JSON.stringify(r).toLowerCase().includes(q));const keys=[...new Set(D.rois.flatMap(r=>Object.keys(r)))];document.getElementById('roi-table').innerHTML=table(rows,keys);const coordinates=rows.filter(r=>Array.isArray(r.coordinates)&&r.coordinates.length===3);document.getElementById('localization').textContent=rows.length+' matching cortical ROIs. '+(coordinates.length?'Declared centroid coordinates shown below (x/y projection).':'Localization: hemisphere + anatomical labels.');if(coordinates.length){const xs=coordinates.map(r=>+r.coordinates[0]),ys=coordinates.map(r=>+r.coordinates[1]);const xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);document.getElementById('localization').innerHTML+='<svg viewBox="0 0 600 320" width="600" role="img" aria-label="Declared cortical ROI centroids">'+coordinates.map(r=>'<circle r="4" cx="'+(20+560*(r.coordinates[0]-xmin)/(xmax-xmin||1))+'" cy="'+(300-280*(r.coordinates[1]-ymin)/(ymax-ymin||1))+'" fill="'+(D.colors[r.network]?'rgb('+D.colors[r.network].map(v=>Math.round(v*255)).join(',')+')':'gray')+'"><title>'+esc(JSON.stringify(r))+'</title></circle>').join('')+'</svg>'}}
function roiEffects(){
const q=document.getElementById('roi-search').value.toLowerCase();const matchedRois=D.rois.filter(r=>JSON.stringify(r).toLowerCase().includes(q));const ids=new Set(matchedRois.map(r=>String(r.roi_id)));
const effects=D.roiStatistics.filter(r=>ids.has(String(r.roi_id))&&fields.every(k=>!select[k].value||r[k]===select[k].value));const showAll=document.getElementById('show-all-roi-effects').checked;const shown=showAll?effects:effects.slice(0,500);
const cols=['branch','hemi','atlas','system','roi_id','target','comparison','mean_patient','sd_patient','n_patient','mean_hc','sd_hc','n_hc','difference','ci_low','ci_high','g','p','df','reason'];
document.getElementById('roi-effect-count').textContent=effects.length+' matching ROI effect estimates.'+(showAll||effects.length===shown.length?'':' Showing first '+shown.length+'; select the checkbox to render all.');
document.getElementById('roi-effects').innerHTML=table(shown,cols);
}

for(const [n,c]of Object.entries(D.colors)){document.getElementById('networks').innerHTML+='<span style="border-left:12px solid rgb('+c.map(v=>Math.round(v*255)).join(',')+');padding-left:4px">'+n+'</span>'}document.getElementById('roi-search').oninput=rois;render();rois();
document.getElementById('roi-search').oninput=()=>{rois();roiEffects()};
document.getElementById('show-all-roi-effects').onchange=roiEffects;
for(const s of Object.values(select))s.addEventListener('change',roiEffects);
roiEffects();
</script></html>'''
    if figures is None:
        figure_list = [
            root / "hipp_functional_identity_standard.png",
            root / "hipp_functional_identity_ctxreg.png",
            root / "hipp_spatial_organization.png",
            root / "hipp_network_fingerprints.png",
            root / "hipp_continuous_gradients.png",
        ]
    else:
        figure_list = figures
    fig_tags = ''.join(f'<p><a href="{f.name}">{f.stem.replace("_", " ").title()}</a></p><img loading="lazy" src="{f.name}" alt="{f.stem} publication figure">' for f in figure_list)
    path = root / 'index.html'; path.write_text(content.replace('PAYLOAD',payload).replace('FIGURES',fig_tags))
    return path


def render(output_dir):
    """Validate every derived input before producing PNG, PDF, and offline HTML reports."""
    root = Path(output_dir)
    data = _load(root)
    figures = []
    for theme in THEMES:
        fig_res = _figure(root, theme, data)
        if isinstance(fig_res, list):
            figures.extend(fig_res)
        else:
            figures.append(fig_res)
    outputs = [*figures, _static_pdf(root, figures), _html(root, data, figures)]
    return outputs

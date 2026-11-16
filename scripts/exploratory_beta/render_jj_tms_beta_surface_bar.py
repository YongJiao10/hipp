#!/home/yojiao/.conda/envs/py314/bin/python
"""Selected JJ_TMS HCP raw beta surfaces and descriptive group bars.

PNG by default; PDF requires --pdf. Never fits a model or reads archives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'LP_NUM_THREADS'):
    os.environ[_key] = '1'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import nibabel as nib
import numpy as np
from PIL import Image
from scipy.stats import t

import run_native_surface_beta as beta
import compare_native_standard_surface_bad_rates as mapping

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/core_workflow'))
from lib import workbench_scene_batch as scene_tools
from lib.workbench import capture_scene
from lib.workbench_grid import split_native_hemi_panels

PATIENTS = dict(zip('3010 3011 3017 3018 3020 3023'.split(),
                    'DMN CON VAN DMN VAN CON'.split()))
CONTROLS = '1006 1007 1017 1020 1023 1035 1041 1044 1052 1054'.split()
SUBJECTS = [*CONTROLS, *PATIENTS]
NETWORKS = {**dict.fromkeys(CONTROLS, 'VAN'), **PATIENTS}
MDD_VAN_SUBJECTS = ('3017', '3020')
STEM = 'jj_tms_hcp_fixed_2mm_36p_beta_surface_bar'
RUN = ROOT / 'outputs_iowa_jj_tms/work/sptms_hipp_cluster_surface_beta_native/runs/jj_tms_native_surface_beta_20260820_78sub'
SCENE = ROOT / 'config/wb_locked_native_view_lateral_medial.scene'
CMAP = plt.get_cmap('RdBu_r')
SCATTER_SIZE = 180
SCATTER_TEXT_SCALE = 1.5
BAR_TITLE_FONT_SIZE = 22 * SCATTER_TEXT_SCALE
BAR_AXIS_LABEL_FONT_SIZE = 17 * SCATTER_TEXT_SCALE
BAR_TICK_FONT_SIZE = 15 * SCATTER_TEXT_SCALE
BAR_LEGEND_FONT_SIZE = 16 * SCATTER_TEXT_SCALE


def card_rows():
    return [CONTROLS[:5], CONTROLS[5:], list(PATIENTS)]


def card_title(subject):
    subject = str(subject)
    group = 'HC' if subject in CONTROLS else 'MDD'
    return f'sub-{subject}  ·  {group}  ·  {NETWORKS[subject]}'


def marker_for_subject(subject):
    return 'D' if str(subject) in MDD_VAN_SUBJECTS else 'o'


def combine_card_panels(card, gap=10):
    """Join the two cropped Workbench panels before placing a card."""
    if gap < 0:
        raise ValueError('card panel gap must be nonnegative')
    left = card['L'].convert('RGB')
    right = card['R'].convert('RGB')
    height = max(left.height, right.height)
    combined = Image.new('RGB', (left.width + gap + right.width, height), 'black')
    combined.paste(left, (0, (height - left.height) // 2))
    combined.paste(right, (left.width + gap, (height - right.height) // 2))
    return combined


def select_rows(rows):
    selected = [dict(r) for r in rows if r['subject'] in SUBJECTS and r['parcellation'] == 'hcp']
    expected = {(s, h, c) for s in SUBJECTS for h in ('L', 'R') for c in (1, 2, 3)}
    keys = [(r['subject'], r['hippocampal_hemi'], int(r['cluster'])) for r in selected]
    if len(keys) != 96 or set(keys) != expected:
        raise ValueError('Expected exactly 96 unique subject × L/R × HCP cluster cells')
    for r in selected:
        if not np.isfinite(float(r['beta_mean'])) or int(r.get('n_sites', 1)) != 1:
            raise ValueError('Expected finite, single-site subject beta')
        if r['roi_policy'] != 'nonzero':
            raise ValueError('Expected nonzero ROI policy')
        r['figure_group'] = 'HC' if r['subject'] in CONTROLS else 'Pure MDD'
        r['tms_site_network'] = NETWORKS[r['subject']]
        source_network = r.get('site_network')
        if source_network and {'Default': 'DMN', 'Control': 'CON', 'VentralAttention': 'VAN'}.get(source_network) != r['tms_site_network']:
            raise ValueError(f"Network mismatch: {r['subject']}")
    return selected


def validate_metric(values, bad, labels, areas, expected):
    if any(np.shape(a) != (512,) for a in (values, bad, labels, areas)):
        raise ValueError('All metric, mask, label and area arrays must have 512 vertices')
    if not np.all(np.isfinite(areas)) or np.any(areas <= 0):
        raise ValueError('Invalid surface areas')
    valid = np.isfinite(values) & np.isfinite(bad) & (bad < .5)
    means = []
    for c in (1, 2, 3):
        mask = valid & (labels == c)
        if not mask.any():
            raise ValueError(f'No valid vertices in cluster {c}')
        means.append(float(np.average(values[mask], weights=areas[mask])))
    if not np.allclose(means, expected, rtol=1e-5, atol=1e-6):
        raise ValueError(f'Raw metric means disagree with original beta_mean: {means} vs {expected}')
    return valid, means


def summarize(rows):
    result = []
    for hemi in ('L', 'R'):
        for c in (1, 2, 3):
            for group in ('HC', 'Pure MDD'):
                cells = [r for r in rows if r['hippocampal_hemi'] == hemi and int(r['cluster']) == c and r['figure_group'] == group]
                values = np.array([float(r['beta_mean']) for r in cells])
                mean = float(values.mean())
                half = float(t.ppf(.975, len(values)-1) * values.std(ddof=1) / np.sqrt(len(values)))
                result.append(dict(hemi=hemi, cluster=c, group=group, n=len(values), mean=mean,
                                   ci95_low=mean-half, ci95_high=mean+half,
                                   subjects=';'.join(r['subject'] for r in cells)))
    return result


def prepare_metrics(rows, sites, out):
    """Recover only missing selected hemisphere metrics, keeping source tables immutable."""
    metrics, audit = {}, []
    selected_sites = select_rows(sites)
    for s in SUBJECTS:
        for h in ('L', 'R'):
            cells = sorted([r for r in selected_sites if r['subject'] == s and r['hippocampal_hemi'] == h], key=lambda r: int(r['cluster']))
            r = cells[0]
            surfaces = mapping.native_surface_paths('jj_tms', s, h)
            path, bad_path = Path(r['native_metric_path']), Path(r['native_bad_vertices_path'])
            recovered = not path.is_file()
            if recovered:
                directory = out / 'raw_metrics' / f'sub-{s}'
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / f'hemi-{h}_con_beta_nonzero.func.gii'
                bad_path = directory / f'hemi-{h}_bad_vertices_nonzero.shape.gii'
                # A complete pair from a prior interrupted invocation can be validated and reused.
                if not (path.is_file() and bad_path.is_file()):
                    roi = directory / 'nonzero_con_roi.nii.gz'
                    if not roi.is_file():
                        mapping.save_roi(Path(r['native_con_path']), roi, 'nonzero')
                    subprocess.run([*mapping.wb_prefix(), '-volume-to-surface-mapping',
                        r['native_con_path'], str(surfaces['midthickness']), str(path),
                        '-ribbon-constrained', str(surfaces['inner']), str(surfaces['outer']),
                        '-volume-roi', str(roi), '-voxel-subdiv', '3',
                        '-bad-vertices-out', str(bad_path)], check=True)
            values = mapping.load_gifti_vector(path)
            bad = mapping.load_gifti_vector(bad_path)
            labels = beta.load_labels('jj_tms', s, h, 'hcp')
            areas = beta.load_surface_areas(surfaces['midthickness'])
            expected = [float(next(r['beta_mean'] for r in rows if r['subject'] == s and r['hippocampal_hemi'] == h and int(r['cluster']) == c)) for c in (1, 2, 3)]
            if not np.allclose([float(r['beta_mean']) for r in cells], expected, rtol=1e-10, atol=1e-10):
                raise ValueError('Site and subject tables disagree')
            valid, means = validate_metric(values, bad, labels, areas, expected)
            metrics[s, h] = np.where(valid, values, np.nan)
            audit.append(dict(subject=s, hemi=h, native_metric_path=str(path),
                original_metric_path=r['native_metric_path'], native_con_path=r['native_con_path'],
                bad_vertices_path=str(bad_path), recovered=recovered, vertices=512,
                masked_vertices=int((~valid).sum()), max_abs_mean_error=float(np.max(np.abs(np.array(means)-expected))),
                surfaces={k: str(v) for k,v in surfaces.items()}))
            print(f'Validated sub-{s} {h}: {valid.sum()}/512 vertices', flush=True)
    return metrics, audit


def vertex_colors(values, limit):
    """One display label per vertex; colors encode raw scalars, never cluster means."""
    table = nib.gifti.GiftiLabelTable()
    null = nib.gifti.GiftiLabel(key=0, red=.65, green=.65, blue=.65, alpha=1)
    null.label = 'Masked vertex'
    table.labels.append(null)
    keys = np.zeros(512, dtype=np.int32)
    for i, value in enumerate(values):
        if np.isfinite(value):
            keys[i] = i+1
            rgba = CMAP(Normalize(-limit, limit)(value))
            label = nib.gifti.GiftiLabel(key=i+1, red=rgba[0], green=rgba[1], blue=rgba[2], alpha=1)
            label.label = f'vertex-{i}: beta={value:.9g}'
            table.labels.append(label)
    return nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(keys, intent='NIFTI_INTENT_LABEL')], labeltable=table)


def display_geometry():
    base = ROOT / 'outputs_hcp/work/group_atlas/pfm/gradient-spectral/kong2022/2mm/cov_0p80/g1g2g3'
    manifests = [json.loads((base / f'hemi_{h}/group_atlas_manifest.json').read_text()) for h in ('L', 'R')]
    return {h: Path(m['template_left_or_right_surface']) for h,m in zip(('L','R'), manifests)}, Path(manifests[0]['template_spec_path'])


def render_card(subject, metrics, limit, out):
    directory = out / 'surface_cards' / f'sub-{subject}'
    directory.mkdir(parents=True, exist_ok=True)
    root = ET.parse(SCENE).getroot()
    scene_tools.absolutize_scene_paths(root, SCENE)
    refs = scene_tools.find_current_structural_label_refs(root)
    surface_refs = scene_tools.find_current_corobl_surface_refs(root)
    surfaces, spec = display_geometry()
    for hemi in ('L','R'):
        if beta.load_surface_areas(surfaces[hemi]).shape != (512,):
            raise ValueError('Display geometry must use corresponding den-512 vertices')
        label_path = directory / f'hemi-{hemi}_raw_beta_colors.label.gii'
        nib.save(vertex_colors(metrics[subject, hemi], limit), label_path)
        scene_tools.replace_label_refs(root, refs[hemi], str(label_path))
        scene_tools.replace_exact_ref(root, surface_refs[hemi], str(surfaces[hemi]))
    scene_tools.replace_exact_ref(root, scene_tools.find_current_spec_ref(root), str(spec))
    scene_out = directory / 'raw_beta.scene'
    ET.ElementTree(root).write(scene_out, encoding='UTF-8', xml_declaration=True)
    png = directory / 'raw_beta.png'
    capture_scene(scene_out, 1, png, 1600, 1024, 'OSMesa')
    return split_native_hemi_panels(Image.open(png).convert('RGB'))


def compose(rows, summaries, cards, limit, out, pdf=False):
    fig = plt.figure(figsize=(34, 30), facecolor='white')
    grid = fig.add_gridspec(5, 12, height_ratios=[1.45,1.45,1.45,.12,1.1], hspace=.36, wspace=.015)
    for row_index, subjects in enumerate(card_rows()):
        start_column = 1 if len(subjects) == 5 else 0
        for column_index, s in enumerate(subjects):
            slot = grid[row_index, start_column + 2 * column_index:start_column + 2 * column_index + 2]
            ax = fig.add_subplot(slot)
            ax.imshow(combine_card_panels(cards[s]), interpolation='nearest')
            ax.axis('off')
            ax.text(.25, 1.005, 'L', transform=ax.transAxes, ha='center', va='bottom', fontsize=18, weight='bold')
            ax.text(.75, 1.005, 'R', transform=ax.transAxes, ha='center', va='bottom', fontsize=18, weight='bold')
            box = slot.get_position(fig)
            fig.text((box.x0+box.x1)/2, box.y1+.010, card_title(s), ha='center', fontsize=18, weight='bold')
    cax = fig.add_subplot(grid[3,5:7])
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-limit,limit), cmap=CMAP), cax=cax, orientation='horizontal')
    colorbar.set_label('Native surface beta (shared scale)', fontsize=14, labelpad=3)
    colorbar.ax.tick_params(labelsize=13, pad=2)
    for j,h in enumerate(('L','R')):
        ax = fig.add_subplot(grid[4,j*6:(j+1)*6])
        for k,group in enumerate(('HC','Pure MDD')):
            color = ('#407caa','#cf7057')[k]
            for c in (1,2,3):
                summary = next(r for r in summaries if r['hemi']==h and r['cluster']==c and r['group']==group)
                x = c + (k-.5)*.34
                ax.bar(x, summary['mean'], width=.31, color=color, alpha=.55)
                ax.errorbar(x, summary['mean'], yerr=summary['ci95_high']-summary['mean'], color='#222222', capsize=5, linewidth=1.4)
                cells = sorted([r for r in rows if r['hippocampal_hemi']==h and int(r['cluster'])==c and r['figure_group']==group], key=lambda r:r['subject'])
                offsets = np.linspace(-.10,.10,len(cells))
                for marker in ('o', 'D'):
                    marked = [(offset, r) for offset, r in zip(offsets, cells) if marker_for_subject(r['subject']) == marker]
                    if marked:
                        ax.scatter([x + offset for offset, _r in marked], [float(r['beta_mean']) for _offset, r in marked], s=SCATTER_SIZE, marker=marker, color=color, edgecolor='#333333', linewidth=0.8, zorder=4)
        ax.axhline(0,color='#777777',linewidth=.7)
        ax.set_xticks([1,2,3], ['Anterior/VAN','Middle/SMN','Posterior/DMN'])
        ax.set_title(f'{h} hippocampus', weight='bold', fontsize=BAR_TITLE_FONT_SIZE, pad=16)
        ax.set_ylabel('Area-weighted beta_mean', fontsize=BAR_AXIS_LABEL_FONT_SIZE, labelpad=16)
        if h == 'R':
            ax.yaxis.set_label_position('right')
        ax.tick_params(axis='both', labelsize=BAR_TICK_FONT_SIZE, pad=11)
        ax.spines[['top','right']].set_visible(False)
    axes = fig.axes[-2:]
    lo = min(a.get_ylim()[0] for a in axes); hi = max(a.get_ylim()[1] for a in axes)
    for a in axes: a.set_ylim(lo,hi)
    fig.suptitle('Iowa 3T · HCP fixed · HC (n=10) and Pure MDD (n=6)', fontsize=32, y=.985)
    fig.text(.5,.955,'Raw native T1w den-512 beta',ha='center',fontsize=18)
    fig.legend(handles=[
        Patch(facecolor='#407caa', edgecolor='none', alpha=.55, label='HC'),
        Patch(facecolor='#cf7057', edgecolor='none', alpha=.55, label='Pure MDD'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='#cf7057', markeredgecolor='#333333', markersize=18, label='Other MDD'),
        Line2D([0], [0], marker='D', color='none', markerfacecolor='#cf7057', markeredgecolor='#333333', markersize=18, label='MDD VAN: sub-3017 + sub-3020'),
    ], loc='lower center', bbox_to_anchor=(.5,.06), ncol=4, frameon=False, fontsize=BAR_LEGEND_FONT_SIZE)
    fig.text(.5,.045,'Bars: group mean with 95% t CI · Each dot: one subject · L/R kept separate · sub-3015 excluded',ha='center',fontsize=15)
    fig.savefig(out / f'{STEM}.png',dpi=240,bbox_inches='tight')
    if pdf: fig.savefig(out / f'{STEM}.pdf',bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=RUN)
    parser.add_argument('--pdf',action='store_true',help='Also export PDF (off by default)')
    args = parser.parse_args()
    out = args.run_dir.resolve() / 'five_figures'
    out.mkdir(parents=True,exist_ok=True)
    rows = select_rows(beta.read_csv(args.run_dir / 'subject_network_beta.csv'))
    summaries = summarize(rows)
    metrics,audit = prepare_metrics(rows,beta.read_csv(args.run_dir / 'site_level_beta.csv'),out)
    limit = max(float(np.nanmax(np.abs(v))) for v in metrics.values()) or 1.
    cards = {}
    for s in SUBJECTS:
        cards[s] = render_card(s,metrics,limit,out)
        print(f'Rendered sub-{s}', flush=True)
    compose(rows,summaries,cards,limit,out,args.pdf)
    beta.write_csv(out / f'{STEM}_selected_beta.csv',rows)
    beta.write_csv(out / f'{STEM}_group_summary.csv',summaries)
    surfaces,spec = display_geometry()
    manifest = dict(subjects=SUBJECTS, groups={'HC':CONTROLS,'Pure MDD':list(PATIENTS)},
        card_layout={'rows': card_rows(), 'description': 'Two HC rows of five cards, then one Pure MDD row of six cards'},
        figure_size_inches=[34, 30],
        font_sizes={'suptitle': 32, 'subtitle': 18, 'card_title': 18, 'hemi_label': 18, 'bar_title': BAR_TITLE_FONT_SIZE, 'axis_label': BAR_AXIS_LABEL_FONT_SIZE, 'tick_label': BAR_TICK_FONT_SIZE, 'legend': BAR_LEGEND_FONT_SIZE, 'scatter_text_scale': SCATTER_TEXT_SCALE},
        figure_label='Iowa 3T',
        marker_encoding={'default': 'o', 'MDD VAN subjects': list(MDD_VAN_SUBJECTS), 'MDD VAN marker': 'D', 'size_points2': SCATTER_SIZE},
        colorbar_layout={'grid_columns': 12, 'occupied_columns': '5:7', 'relative_width': '2/12'},
        bar_layout={'legend_anchor_y': .06, 'right_panel_ylabel': 'outside_right', 'footer_rows': 1},
        tms_site_network=NETWORKS, excluded={'3015':'Explicit exclusion in requested 16-subject cohort'},
        n_records=len(rows), parcellation='hcp_fixed', measurement='beta_mean',
        ci='two-sided 95% Student t interval of independent subject means, df=n-1',
        source_tables={name:dict(path=str(args.run_dir.resolve()/name),sha256=hashlib.sha256((args.run_dir/name).read_bytes()).hexdigest()) for name in ('subject_network_beta.csv','site_level_beta.csv')},
        mapping=dict(method='ribbon-constrained',voxel_subdiv=3,roi='nonzero',surface_space='T1w',density_evidence='512 vertices in consumed native metrics and surfaces'),
        scale=[-limit,limit],colormap='RdBu_r', scene=str(SCENE),visual_preset='2mm_36p',
        display_surfaces={h:str(p) for h,p in surfaces.items()},display_spec=str(spec),
        display_encoding='One unique display label per valid vertex colored by its raw beta; no cluster-mean filling. Shared den-512 geometry, vertex correspondence; no smoothing or resampling.',
        pdf=args.pdf,metrics=audit,outputs=[f'{STEM}.png',f'{STEM}_selected_beta.csv',f'{STEM}_group_summary.csv']+([f'{STEM}.pdf'] if args.pdf else []))
    (out / f'{STEM}_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(out / f'{STEM}.png',flush=True)


if __name__ == '__main__':
    main()

"""Explicit canonical7/36P input inventory. No artifact discovery or fallback."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import nibabel as nib

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ROSTERS = {'hc':'jj_tms_hc_subjects.txt','patient':'jj_tms_patient_subjects.txt',
           'pure_mdd':'jj_tms_pure_mdd_subjects.txt','comorbid_mdd':'jj_tms_comorbid_mdd_subjects.txt'}
COHORTS={'hc':'iowa_hc','patient':'iowa_pat','pure_mdd':'iowa_pure_mdd','comorbid_mdd':'iowa_comorbid_mdd'}
NETWORKS={'Default':'DMN','Somatomotor':'SMN','VentralAttention':'VAN','DorsalAttention':'DAN','Control':'CON','Language':'LAN','Visual':'VN'}
HCP_TEMPLATE_SPECS={
    'standard':{'directory':'hcp_routea_full_k3_fisherz','template_id':'hcp_routea_full_k3_fisherz'},
    'ctxreg':{'directory':'hcp_routea_ctxreg','template_id':'hcp_routea_ctxreg'},
}
HCP_TEMPLATE_SEMANTICS={'1':'Anterior/VAN','2':'Middle/SMN','3':'Posterior/DMN'}

def active_path(path):
    p=Path(path)
    if any('archive' in v.lower() for v in p.parts) or any('archive' in v.lower() for v in p.resolve().parts):
        raise ValueError(f'archive access prohibited: {path}')
    return p.resolve()

def fingerprint(path):
    p=active_path(path)
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return {'path':str(p),'sha256':h.hexdigest(),'size_bytes':p.stat().st_size}

def verify_record(rec):
    got=fingerprint(rec['path'])
    if got['sha256']!=rec['sha256']: raise ValueError(f"hash mismatch: {rec['path']}")
    if 'size_bytes' in rec and got['size_bytes']!=rec['size_bytes']: raise ValueError(f"size mismatch: {rec['path']}")
    return Path(got['path'])

def read_json(path): return json.loads(active_path(path).read_text())
def read_tsv(path):
    with active_path(path).open() as f: return list(csv.DictReader(f,delimiter='\t'))

def validate_registry(rows):
    if [int(r['roi_id']) for r in rows]!=list(range(1,401)):
        raise ValueError('400 ROI order must be exactly 1..400')
    if len({r['roi_name'] for r in rows})!=400: raise ValueError('duplicate ROI names')
    if set(r['canonical7'] for r in rows)!=set(NETWORKS): raise ValueError('canonical7 network registry mismatch')
    if set(r['hemi'] for r in rows)!={'L','R'}: raise ValueError('bilateral cortex registry required')
    return [dict(r,network=NETWORKS[r['canonical7']]) for r in rows]

def validate_arrays(ts,roi,valid,context):
    if ts.ndim!=2 or roi.ndim!=2 or ts.shape[0]!=512 or roi.shape[0]!=400 or valid.shape!=(512,):
        raise ValueError(f'512 surface / 400 ROI shape mismatch: {context}')
    if ts.shape[1]!=roi.shape[1] or ts.shape[1]<3: raise ValueError(f'time alignment mismatch: {context}, {ts.shape}, {roi.shape}')
    if not np.isin(valid,[0,1]).all(): raise ValueError(f'invalid mask: {context}')

def surface(path):
    p=active_path(path)
    if '_den-512_' not in p.name: raise ValueError(f'actual den-512 asset required: {p}')
    img=nib.load(p); points=np.asarray(img.agg_data('pointset'),float); faces=np.asarray(img.agg_data('triangle'),int)
    if points.shape!=(512,3) or faces.ndim!=2 or faces.shape[1]!=3 or not np.isfinite(points).all():
        raise ValueError(f'actual den-512 pointset mismatch: {p}')
    if faces.min()<0 or faces.max()>=512: raise ValueError(f'surface topology mismatch: {p}')
    return points,faces

def vertex_areas(points,faces):
    tri=points[faces]
    areas=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
    out=np.zeros(len(points)); np.add.at(out,faces.ravel(),np.repeat(areas/3,3))
    return out

def unfold_axis(points):
    # HippUnfold den-512 unfolded grid has y=0 at anterior and y=-38.75 posterior.
    # Fixed global bounds, never re-estimated per subject or sorted by native PCA.
    if not np.isfinite(points).all() or np.any(points[:,1]<-38.7501) or np.any(points[:,1]>0.0001):
        raise ValueError('unfolded AP coordinate outside declared [-38.75,0] bounds')
    return -points[:,1]/38.75

def load_rosters():
    rosters={}
    for key,filename in ROSTERS.items():
        values=['sub-'+s.strip() for s in (ROOT/'manifests'/filename).read_text().splitlines() if s.strip() and not s.startswith('#')]
        if not values or len(values)!=len(set(values)):
            raise ValueError(f'roster empty/duplicates: {filename}')
        rosters[key]=values
    if set(rosters['hc']) & set(rosters['patient']): raise ValueError('HC/patient overlap')
    if set(rosters['pure_mdd']) & set(rosters['comorbid_mdd']) or set(rosters['pure_mdd'])|set(rosters['comorbid_mdd'])!=set(rosters['patient']):
        raise ValueError('patient subgroup partition mismatch')
    return rosters

def check_branch(m,branch,path,*,standard_schema=False):
    # Explicitly named original standard schema omitted signal_branch; only the
    # standard registry path in this inventory permits that documented schema.
    expected=None if standard_schema and branch=='standard' and 'signal_branch' not in m else branch
    if m.get('signal_branch')!=expected: raise ValueError(f'signal branch mismatch: {path}')

def make_inventory():
    from scripts.visualize.lib.multidataset_cohorts import GridSpec,get_comparison_cohorts
    rosters=load_rosters()
    inventory={'schema':'iowa_organization_v1','scope':{'smoothing':'2mm','variant':'36p','taxonomy':'canonical7','n_roi':400,'density':512},
               'rosters':rosters,'roster_files':{k:fingerprint(ROOT/'manifests'/v) for k,v in ROSTERS.items()},
               'subjects':[],'groups':[],'references':[],'hcp_templates':[], 'standard_manifest_schema':'original_standard_without_signal_branch'}
    registered=None
    # De-duplicate file hashing without changing selection; every request names an exact path.
    receipts={}
    hcp_template_records={}
    def rec(p):
        p=str(active_path(p))
        if p not in receipts: receipts[p]=fingerprint(p)
        return receipts[p]
    for branch in ['standard','ctxreg']:
        specs={c['id']:c for c in get_comparison_cohorts(GridSpec(signal_branch=branch))['kong2022_7net']}
        hcp_route={'standard':'routea','ctxreg':'routea_ctxreg'}[branch]
        for hemi in ['L','R']:
            template_spec=HCP_TEMPLATE_SPECS[branch]
            template_manifest=ROOT/'outputs_hcp/templates'/template_spec['directory']/'template_manifest.json'
            template_payload=read_json(template_manifest)
            if (template_payload.get('status')!='complete' or template_payload.get('template_id')!=template_spec['template_id']
                    or template_payload.get('n_subjects')!=len(template_payload.get('subjects') or []) or template_payload.get('n_rois')!=400
                    or template_payload.get('density')!=512 or template_payload.get('k')!=3
                    or template_payload.get('fc_space')!='subject-mean Fisher-z vertex-to-Kong2022-ROI FC'
                    or template_payload.get('cluster_label_semantics')!=HCP_TEMPLATE_SEMANTICS):
                raise ValueError(f'fixed HCP template contract: {template_manifest}')
            combinations=[item for item in template_payload.get('combinations',[]) if item.get('smoothing')=='2mm' and item.get('hemi')==hemi]
            if len(combinations)!=1:
                raise ValueError(f'fixed HCP template combination must be unique: {template_manifest} / {hemi}')
            outputs=combinations[0].get('outputs',{})
            if set(outputs)!={'labels','prototypes','support'}:
                raise ValueError(f'fixed HCP template outputs contract: {template_manifest} / {hemi}')
            template_entry=dict(branch=branch,hemi=hemi,template_id=template_spec['template_id'],manifest=rec(template_manifest),
                                labels=rec(outputs['labels']['path']),prototypes=rec(outputs['prototypes']['path']),support=rec(outputs['support']['path']))
            hcp_labels=np.load(template_entry['labels']['path'],allow_pickle=False)
            hcp_support=np.load(template_entry['support']['path'],allow_pickle=False)
            if (hcp_labels.shape!=(512,) or hcp_support.shape!=(512,) or not np.isin(hcp_labels,[0,1,2,3]).all()
                    or not np.isin(hcp_support,[0,1]).all() or not np.array_equal(hcp_labels!=0,hcp_support.astype(bool))):
                raise ValueError(f'fixed HCP labels/support contract: {template_manifest} / {hemi}')
            template_key=(branch,hemi)
            if template_key in hcp_template_records:
                raise ValueError(f'duplicate fixed HCP template: {template_key}')
            hcp_template_records[template_key]=template_entry
            inventory['hcp_templates'].append(template_entry)
            gp=ROOT/f'outputs_hcp/work/hippomaps_gradients/{hcp_route}/2mm/hemi_{hemi}/hippomaps_gradient_manifest.json'
            gm=read_json(gp)
            if gm['feature_count']!=400 or gm['n_components_computed']!=5 or gm['n_subjects']!=len(gm.get('subjects') or []) or gm['route']!=hcp_route or gm['hemi']!=hemi or gm['smoothing']!='2mm':
                raise ValueError(f'HCP external gradient reference contract: {gp}')
            inventory['references'].append(dict(branch=branch,hemi=hemi,manifest=rec(gp),gradients=rec(gm['outputs']['hippomaps_gradients_full']),support=rec(gm['outputs']['group_support_mask'])))
        for cohort,cid in COHORTS.items():
            c=specs[cid]
            for hemi in ['L','R']:
                gp=Path(c['grad_subpath'])/f'hemi_{hemi}/group_atlas_manifest.json'; gm=read_json(gp)
                check_branch(gm,branch,gp,standard_schema=True)
                if ['sub-'+str(s).removeprefix('sub-') for s in gm['subjects']]!=rosters[cohort]: raise ValueError(f'group roster/order mismatch: {gp}')
                if gm['method']!='gradient-spectral' or gm['atlas']!='kong2022' or gm['network_registry_column']!='canonical7': raise ValueError(f'group atlas variant mismatch: {gp}')
                rr=validate_registry(read_tsv(gm['cluster_roi_registry']))
                if registered is None: registered=rr
                if rr!=registered: raise ValueError(f'ROI registry differs: {gp}')
                wp=Path(c['wta_subpath'])/f'hemi_{hemi}/group_atlas_manifest.json'; wm=read_json(wp)
                check_branch(wm,branch,wp,standard_schema=True)
                if wm['default_assignment_variant']!='normalized_top20_count' or wm['network_taxonomy']!='canonical7' or wm['method']!='wta-top': raise ValueError(f'WTA variant mismatch: {wp}')
                if ['sub-'+str(s).removeprefix('sub-') for s in wm['subjects']]!=rosters[cohort]: raise ValueError(f'WTA roster mismatch: {wp}')
                inventory['groups'].append(dict(branch=branch,cohort=cohort,hemi=hemi,gradient_manifest=rec(gp),
                    labels=rec(gm['group_labels_final']),support=rec(gm['group_support_mask']),surface=rec(gm['template_left_or_right_surface']),
                    roi_profile=rec(gm['cluster_roi_profile']),registry=rec(gm['cluster_roi_registry']),
                    wta_manifest=rec(wp),wta_labels=rec(wm['group_functional_atlas_labels']),wta_support=rec(wm['group_support_mask']),wta_map=rec(wm['group_functional_atlas_label_map'])))
        c=specs['iowa_hc']
        for subject in rosters['hc']+rosters['patient']:
            for hemi in ['L','R']:
                base=ROOT/'outputs_iowa_jj_tms/work/hcp_fixed_template_matching'
                subject_root={'standard':base/'subjects','ctxreg':base/'routea_ctxreg/subjects'}[branch]
                mp=subject_root/subject/f'matching/2mm/hemi-{hemi}/combo_manifest.json'; m=read_json(mp)
                check_branch(m,branch,mp,standard_schema=True)
                if m['subject']!=subject or m['hemi']!=hemi or m['smoothing']!='2mm' or m['status']!='ok': raise ValueError(f'TM subject/hemisphere contract: {mp}')
                if m['label_semantics']!={'1':'Anterior/VAN','2':'Middle/SMN','3':'Posterior/DMN'}: raise ValueError(f'TM semantics: {mp}')
                row=dict(branch=branch,hemi=hemi,subject=subject,manifest=rec(mp))
                for k in ['scores','labels','margin','fisherz_fc']:
                    row[k]=rec(m['outputs'][k]['path'])
                    if row[k]['sha256']!=m['outputs'][k]['sha256']: raise ValueError(f'TM producer hash mismatch {k}: {mp}')
                for k in ['roi_registry','template_prototypes']:
                    row[k]=rec(m[k]['path'])
                    if row[k]['sha256']!=m[k]['sha256']: raise ValueError(f'TM producer hash mismatch {k}: {mp}')
                if row['template_prototypes']!=hcp_template_records[(branch,hemi)]['prototypes']:
                    raise ValueError(f'TM prototypes do not match fixed HCP template: {mp}')
                if validate_registry(read_tsv(row['roi_registry']['path']))!=registered: raise ValueError(f'TM registry mismatch: {mp}')
                native=active_path(m['density_evidence']['consumed_asset'])
                if m['density_evidence']['n_vertices']!=512: raise ValueError(f'density evidence mismatch: {mp}')
                unfolded=native.with_name(f'{subject}_hemi-{hemi}_space-unfold_den-512_label-hipp_midthickness.surf.gii')
                for k,p in [('native',native),('unfolded',unfolded),
                    ('ts',Path(c['surface_root'])/subject/f'surface/2mm/{subject}_hemi-{hemi}_timeseries.npy'),
                    ('roi',Path(c['reference_root'])/subject/'reference/kong2022/cortex_roi_parcel_timeseries.npy'),
                    ('roi_table',Path(c['reference_root'])/subject/'reference/kong2022/cortex_roi_parcels.tsv'),
                    ('valid',Path(c['mask_root'])/subject/f'surface/validity/{subject}_hemi-{hemi}_valid_mask.npy')]: row[k]=rec(p)
                points,faces=surface(native); up,uf=surface(unfolded)
                if not np.array_equal(faces,uf): raise ValueError(f'native/unfold topology mismatch: {mp}')
                axis=unfold_axis(up)
                if np.corrcoef(axis,points[:,1])[0,1]>=0: raise ValueError(f'anterior to posterior orientation mismatch: {mp}')
                ts,roi,valid=[np.load(row[k]['path'],mmap_mode='r',allow_pickle=False) for k in ['ts','roi','valid']]
                validate_arrays(ts,roi,valid,mp)
                sr=read_tsv(row['roi_table']['path'])
                if [(int(r['parcel_id']),r['parcel_name']) for r in sr]!=[(int(r['roi_id']),r['roi_name']) for r in registered]: raise ValueError(f'ROI time-series order mismatch: {mp}')
                scores=np.load(row['scores']['path']); labels=np.load(row['labels']['path']); fc=np.load(row['fisherz_fc']['path']); margin=np.load(row['margin']['path'])
                if scores.shape!=(512,3) or labels.shape!=(512,) or fc.shape!=(512,400) or not np.isin(labels,[0,1,2,3]).all(): raise ValueError(f'TM shape/labels: {mp}')
                ordered=np.sort(scores,axis=1)
                if not np.allclose(ordered[:,-1]-ordered[:,-2],margin,equal_nan=True,atol=1e-6): raise ValueError(f'existing margin mismatch: {mp}')
                row['n_timepoints']=ts.shape[1]
                inventory['subjects'].append(row)
        print(f'Inventoried {branch}: all roster subjects x 2 hemispheres',flush=True)
    inventory['registry']=registered
    inventory['cortex_statistics']=rec(HERE.parent/'cortex_network_area/stats/network_area_statistics.csv')
    inventory['cortex_subjects']=rec(HERE.parent/'cortex_network_area/stats/subject_network_area_metrics.csv')
    inventory['axis_contract']={'coordinate':'unfold y','anterior':0.0,'posterior':-38.75,'bins':20,'transform':'-y/38.75'}
    return inventory

def validate_inventory(inv):
    if inv['schema']!='iowa_organization_v1' or inv['scope']!={'smoothing':'2mm','variant':'36p','taxonomy':'canonical7','n_roi':400,'density':512}: raise ValueError('inventory scope mismatch')
    if inv['rosters']!=load_rosters(): raise ValueError('rosters changed since inventory')
    visited=set()
    def visit(obj):
        if isinstance(obj,dict):
            if 'path' in obj and 'sha256' in obj:
                key=(obj['path'],obj['sha256'])
                if key not in visited: verify_record(obj);visited.add(key)
            else:
                for value in obj.values(): visit(value)
        elif isinstance(obj,list):
            for value in obj: visit(value)
    visit(inv)
    hcp_templates=inv.get('hcp_templates')
    expected_templates={(branch,hemi) for branch in ('standard','ctxreg') for hemi in ('L','R')}
    if not isinstance(hcp_templates,list) or {(row.get('branch'),row.get('hemi')) for row in hcp_templates}!=expected_templates:
        raise ValueError('fixed HCP template inventory coverage mismatch')
    if len(inv['subjects'])!=372 or len({(r['branch'],r['hemi'],r['subject']) for r in inv['subjects']})!=372: raise ValueError('incomplete/duplicate subject inventory')
    return inv

"""Prepare the explicitly audited Iowa fsLR surface set; no existence fallback."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess

import nibabel as nib
import numpy as np

from scripts.visualize.cortex_network_area.core import load_formal_rosters, compute_vertex_areas

# These subjects lack only the fsLR 32k derivative in derivatives_surf.
# Their native anatomy and fsLR registration exist. Never use the older
# repo surfaces_32k cache: its provenance names the fsaverage sphere.
RESAMPLE_SUBJECTS = frozenset(['sub-1028', *[f'sub-{i}' for i in range(1055, 1069)], 'sub-3012', 'sub-3028'])
SOURCE_ROOT = Path('/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS/derivatives_surf')
SPHERE_ROOT = Path(__file__).resolve().parents[3] / 'external/FASTANS/resources/templates/standard_mesh_atlases'
WB = Path('/home/yojiao/FASTANS/workbench/bin_linux64/wb_command')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_output_paths(repo_root: Path, work_dir: Path, manifest_path: Path):
    base = repo_root.resolve() / 'outputs_iowa_jj_tms/inputs/iowa_jj_tms/cortex_surfaces'
    if not work_dir.resolve().is_relative_to(base) or not manifest_path.resolve().is_relative_to(work_dir.resolve()):
        raise ValueError(f'Derived surfaces and manifest must use persistent dataset inputs beneath {base}')


def prepare(repo_root: Path, work_dir: Path, manifest_path: Path):
    validate_output_paths(repo_root, work_dir, manifest_path)
    if manifest_path.exists():
        raise FileExistsError(f'Refusing to replace existing surface manifest: {manifest_path}')
    subjects = sorted({s for c, r in load_formal_rosters(repo_root).items() if c != 'HCP HC' for s in r.subjects})
    targets = {h: repo_root / 'external/FASTANS/resources/templates/standard_mesh_atlases' / f'{h}.sphere.32k_fs_LR.surf.gii' for h in ('L', 'R')}
    # Validate the complete write set before creating any output. No overwrite
    # or existence-selected reuse of partial surface products is permitted.
    outputs = [work_dir / s / f'{s}_hemi-{h}_space-fsLR_den-32k_midthickness.surf.gii'
               for s in sorted(RESAMPLE_SUBJECTS) for h in ('L', 'R')]
    outputs += [work_dir / f'{s}_L_vertex_area.shape.gii' for s in ('sub-1001', 'sub-3012')]
    for path in outputs:
        if path.exists():
            raise FileExistsError(f'Refusing to replace existing surface product: {path}')
    triangles = {h: nib.load(p).darrays[1].data for h,p in targets.items()}
    work_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for subject in subjects:
        for hemi in ('L', 'R'):
            stem = f'{subject}_hemi-{hemi}'
            anat = SOURCE_ROOT / subject / 'anat'
            native = anat / f'{stem}_midthickness.surf.gii'
            sphere = anat / f'{stem}_space-fsLR_desc-reg_sphere.surf.gii'
            filename = f'{stem}_space-fsLR_den-32k_midthickness.surf.gii'
            command = None
            if subject in RESAMPLE_SUBJECTS:
                surface = work_dir / subject / filename
                for source in (native, sphere, targets[hemi]):
                    if not source.is_file():
                        raise FileNotFoundError(source)
                a, b = nib.load(native), nib.load(sphere)
                if not np.array_equal(a.darrays[1].data, b.darrays[1].data):
                    raise ValueError(f'Native/sphere topology mismatch: {subject}/{hemi}')
                surface.parent.mkdir(parents=True, exist_ok=True)
                command = [str(WB), '-surface-resample', str(native), str(sphere), str(targets[hemi]), 'BARYCENTRIC', str(surface)]
                subprocess.run(command, check=True)
            else:
                surface = anat / filename
            image = nib.load(surface)
            if not np.array_equal(image.darrays[1].data, triangles[hemi]):
                raise ValueError(f'fsLR target topology mismatch: {surface}')
            if image.darrays[0].data.shape != (32492, 3):
                raise ValueError(f'fsLR vertex count mismatch: {surface}')
            provenance = dict(image.meta).get('Provenance', '')
            if sphere.name not in provenance:
                raise ValueError(f'Wrong registration sphere provenance: {surface}')
            entries.append(dict(subject=subject, hemisphere=hemi, surface=str(surface), sha256=digest(surface),
                                native_surface=str(native), registered_fsLR_sphere=str(sphere), target_sphere=str(targets[hemi]),
                                provenance=provenance, command=command, topology_verified=True))
        if subject in RESAMPLE_SUBJECTS:
            print(f'Prepared {subject}', flush=True)
    # Cross-check area formula against Workbench for one retained and one rebuilt subject.
    checks=[]
    for subject in ('sub-1001', 'sub-3012'):
        entry=next(e for e in entries if e['subject']==subject and e['hemisphere']=='L')
        surface=Path(entry['surface']); metric=work_dir/f'{subject}_L_vertex_area.shape.gii'
        subprocess.run([str(WB),'-surface-vertex-areas',str(surface),str(metric)],check=True)
        mesh=nib.load(surface); expected=nib.load(metric).darrays[0].data
        actual=compute_vertex_areas(mesh.darrays[0].data,mesh.darrays[1].data)
        np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-5)
        checks.append(dict(subject=subject,max_abs_error_mm2=float(np.max(np.abs(actual-expected)))))
    manifest_path.write_text(json.dumps(dict(profile='iowa_fsLR_complete_roster_v1', entries=entries, workbench_area_checks=checks),indent=2)+'\n')


def apply_manifest(records, path: Path):
    payload=json.loads(path.read_text())
    if payload['profile'] != 'iowa_fsLR_complete_roster_v1':
        raise ValueError(f'Unsupported surface profile: {path}')
    entries={(e['subject'],e['hemisphere']):e for e in payload['entries']}
    if len(entries)!=len(payload['entries']):
        raise ValueError(f'Duplicate surface manifest entries: {path}')
    for entry in entries.values():
        surface = Path(entry['surface']).resolve()
        if surface.parts[:2] == ('/', 'tmp') or str(surface).startswith('/tmp/'):
            raise ValueError(f'temporary surface path is not allowed in a production manifest: {surface}')
    needed={(r.subject,h) for r in records if r.cohort!='HCP HC' for h in ('L','R')}
    if needed!=set(entries):
        raise ValueError(f'Surface manifest does not cover exact Iowa roster: {path}')
    for e in entries.values():
        if digest(e['surface']) != e['sha256']:
            raise ValueError(f'Stale surface manifest: {e["surface"]}')
    return [replace(r,surface_paths={h:Path(entries[(r.subject,h)]['surface']) for h in ('L','R')}) if r.cohort!='HCP HC' else r for r in records]


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[3])
    parser.add_argument('--work-dir',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    args=parser.parse_args()
    prepare(args.repo_root.resolve(),args.work_dir.resolve(),args.manifest.resolve())

"""Explicit physical-run inputs shared by sampling and participant QC."""
from __future__ import annotations

import csv
import json
import math

import numpy as np
from pathlib import Path


MOTION_FORMATS = ('hcp_movement_degrees', 'fmriprep_confounds')


def cohort_prefix(dataset: str, task: str) -> str:
    if dataset == 'hcp':
        return 'hcp_7t' if task == 'rest' else f'hcp_3t_{task}'
    return dataset if task == 'rest' else f'{dataset}_{task}'


def pass_roster_name(dataset: str, task: str, count: int | str = 'latest') -> str:
    return f'{cohort_prefix(dataset, task)}_hippocampus_fmri_qc_pass_t1only_{count}.txt'


def load_qc_run_inputs(path: Path) -> dict[str, list[dict[str, object]]]:
    """Read exact paths and physical run identities; no search or hashing."""
    required = {'subject', 'run_id', 'run_label', 'bold_path', 'n_timepoints',
                'motion_path', 'motion_format'}
    subjects: dict[str, list[dict[str, object]]] = {}
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'QC run inputs require {sorted(required)}: {path}')
        for row in reader:
            subject = row['subject'].removeprefix('sub-')
            run_id, frames = int(row['run_id']), int(row['n_timepoints'])
            if not subject or not row['run_label'] or run_id < 1 or frames < 2:
                raise ValueError(f'Invalid physical run declaration in {path}: {row}')
            if row['motion_format'] not in MOTION_FORMATS:
                raise ValueError(f'Unsupported motion_format in {path}: {row["motion_format"]}')
            for key in ('bold_path', 'motion_path'):
                if key == 'bold_path' and not row[key]:
                    continue  # Missing source declaration remains explicitly unassessable.
                if not Path(row[key]).is_absolute():
                    raise ValueError(f'QC run inputs require an exact absolute {key}: {path}: {row[key]}')
            runs = subjects.setdefault(subject, [])
            if any(r['run_id'] == run_id or r['run_label'] == row['run_label'] for r in runs):
                raise ValueError(f'Duplicate physical run for sub-{subject} in {path}: {row}')
            runs.append({'run_id': run_id, 'run_label': row['run_label'],
                         'bold_path': row['bold_path'], 'n_timepoints': frames,
                         'path': row['motion_path'], 'motion_format': row['motion_format']})
            if row.get('temporal_mask_path'):
                mask = Path(row['temporal_mask_path'])
                if not mask.is_absolute():
                    raise ValueError(f'QC inputs require an absolute temporal_mask_path: {path}: {mask}')
                runs[-1]['temporal_mask_path'] = str(mask)
    for runs in subjects.values():
        runs.sort(key=lambda row: row['run_id'])
    return subjects


def read_hippunfold_completion(batch_root: Path, subject: str, required_version: str) -> tuple[bool | None, Path]:
    """Read recorded official completion, including a declared post-cleanup result."""
    path = batch_root / f'sub-{subject}/hippunfold/output_check.json'
    if not path.is_file():
        return None, path
    report = json.loads(path.read_text())
    if not isinstance(report, list) or len(report) != 1 or str(report[0].get('subject', '')).removeprefix('sub-') != subject:
        raise ValueError(f'Malformed HippUnfold completion report: {path}')
    completed = report[0]
    version = completed.get('run_contract', {}).get('hippunfold_version')
    if version != required_version:
        raise ValueError(f'HippUnfold version mismatch: {path}: expected={required_version}, recorded={version}')
    return completed.get('status') in {'complete_all_outputs_present', 'complete_retained_outputs_after_cleanup'}, path


def read_motion(runs: list[dict[str, object]]) -> tuple[float | None, list[dict[str, object]], list[str]]:
    if not runs:
        return None, [], ["missing_motion_declaration"]
    missing = [f"missing_motion:{run['path']}" for run in runs if not Path(str(run["path"])).is_file()]
    missing.extend(f"missing_temporal_mask:{run['temporal_mask_path']}" for run in runs
                   if run.get('temporal_mask_path') and not Path(str(run['temporal_mask_path'])).is_file())
    if missing:
        return None, [], missing
    measured = []
    total_fd = 0.0
    total_frames = 0
    for run in runs:
        path = Path(str(run["path"]))
        motion_format = run.get('motion_format', 'hcp_movement_degrees')
        if motion_format == 'fmriprep_confounds':
            with path.open(newline='', encoding='utf-8') as stream:
                reader = csv.DictReader(stream, delimiter='\t')
                if 'framewise_displacement' not in (reader.fieldnames or []):
                    raise ValueError(f'Missing framewise_displacement column: {path}')
                fd = [row['framewise_displacement'] for row in reader]
            if not fd:
                raise ValueError(f'Empty motion input: {path}')
            values = np.asarray([0.0] + [float(value) for value in fd[1:]])
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f'Invalid framewise_displacement: {path}')
            frames, fd_sum = len(values), float(values.sum())
        elif motion_format == 'hcp_movement_degrees':
            motion = np.loadtxt(path, ndmin=2)
            if motion.shape[0] < 1 or motion.shape[1] != 12 or not np.isfinite(motion).all():
                raise ValueError(f"Expected finite physical-run HCP motion with 12 columns: {path}, shape={motion.shape}")
            differences = np.abs(np.diff(motion[:, :6], axis=0))
            fd_sum = float(differences[:, :3].sum() + differences[:, 3:].sum() * (math.pi / 180) * 50)
            frames = int(motion.shape[0])
        else:
            raise ValueError(f'Unsupported motion format {motion_format!r}: {path}')
        if run.get('temporal_mask_path'):
            mask_path = Path(str(run['temporal_mask_path']))
            with mask_path.open(newline='', encoding='utf-8') as stream:
                reader = csv.DictReader(stream, delimiter='\t')
                if 'framewise_displacement' not in (reader.fieldnames or []):
                    raise ValueError(f'Missing framewise_displacement censor column: {mask_path}')
                mask = np.asarray([float(row['framewise_displacement']) for row in reader])
            if (len(mask) != frames or not np.isfinite(mask).all()
                    or not np.isin(mask, [0, 1]).all()
                    or int(np.count_nonzero(mask == 0)) != run['n_timepoints']):
                raise ValueError(f'Motion/censor/BOLD frame mismatch: {mask_path}: original={frames}, retained={run["n_timepoints"]}')
        elif 'n_timepoints' in run and frames != run['n_timepoints']:
            raise ValueError(f'Motion/BOLD frame mismatch: {path}: motion={frames}, declared={run["n_timepoints"]}')
        total_fd += fd_sum
        total_frames += frames  # First physical-run frame contributes zero FD.
        measured.append({**run, "n_frames": frames, "mean_power_fd_mm": fd_sum / frames})
    return total_fd / total_frames, measured, []

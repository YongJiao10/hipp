#!/usr/bin/env python3
"""Package explicit baseline-preserving surface samples and retained frame indices.

Samples must already use the analysis geometry and mapping method. This utility
never substitutes a denoised residual for missing baseline data.
"""
import argparse
import sys
from pathlib import Path

CORE_WORKFLOW_DIR = Path(__file__).resolve().parents[1]
if str(CORE_WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_WORKFLOW_DIR))

import nibabel as nib
import numpy as np
from lib.extract_cortex_merged_network_timeseries import extract_structure_data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['stanford', 'iowa_jj_tms', 'iowa7t'], required=True)
    p.add_argument('--subject', required=True)
    p.add_argument('--hemi', choices=['L', 'R'], required=True)
    p.add_argument('--domain', choices=['hipp', 'cortex'], required=True)
    p.add_argument('--samples', nargs='+', type=Path, required=True)
    p.add_argument('--retained-frames', nargs='+', type=Path, required=True,
                   help='One NPY integer original-frame-index vector per run, in sample order')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if len(a.samples) != len(a.retained_frames):
        p.error('Provide a retained-frame array for every original run')
    data = dict(dataset=a.dataset, subject=a.subject.removeprefix('sub-'), hemi=a.hemi,
                n_runs=len(a.samples), source_paths=np.array([str(s.resolve()) for s in a.samples]))
    vertices = None
    for i, (source, keep) in enumerate(zip(a.samples, a.retained_frames)):
        image = nib.load(str(source))
        if a.domain == 'cortex':
            ids, values = extract_structure_data(image.header.get_axis(1), np.asarray(image.dataobj).T,
                                                 'CORTEX_LEFT' if a.hemi == 'L' else 'CORTEX_RIGHT')
            if vertices is not None and not np.array_equal(vertices, ids):
                raise ValueError(f'Cortical run brain-model vertex order mismatch: {source}')
            vertices = ids
        else:
            if not isinstance(image, nib.gifti.GiftiImage):
                raise ValueError(f'Hippocampal samples must be baseline GIFTI metrics: {source}')
            values = np.column_stack([d.data for d in image.darrays])
            if values.ndim != 2 or values.shape[0] != 512:
                raise ValueError(f'Expected 512 baseline hippocampal vertices: {source}: {values.shape}')
        data[f'run_{i}'] = values
        data[f'frames_{i}'] = np.load(keep, allow_pickle=False)
    if vertices is not None:
        data['vertex_ids'] = vertices
    from lib.tsnr import runwise_tsnr
    runwise_tsnr([data[f'run_{i}'] for i in range(len(a.samples))],
                 frames=[data[f'frames_{i}'] for i in range(len(a.samples))], threshold=25)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **data)


if __name__ == '__main__':
    main()

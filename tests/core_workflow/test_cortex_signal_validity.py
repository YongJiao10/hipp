"""Cortical masks retain usable signals without a numerical quality cutoff."""
from pathlib import Path
import json
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/core_workflow'))
from lib import io_utils


@pytest.mark.parametrize('dataset,task', [('hcp', 'rest'), ('hcp', 'emotion'), ('stanford', 'rest')])
def test_cortex_mask_excludes_nonfinite_values_only(tmp_path, dataset, task):
    assert hasattr(io_utils, 'compute_cortex_validity'), 'Missing cortical signal-validity mask producer'
    values = np.array([
        [10000, 12000, 8000, 50000, 52000, 48000],
        [0, 0, 0, 0, 0, 0],
        [1, 2, 3, 4, 4, 4],
        [1, 2, 3, 4, np.nan, 6],
    ], dtype=float)
    mask = io_utils.compute_cortex_validity(
        data=values, outdir=tmp_path, hemi='left', dataset=dataset,
        task=task, run_lengths=[3, 3],
    )
    assert np.load(mask).tolist() == [True, True, True, False]
    summary = json.loads((tmp_path / 'cortex_left_validity_summary.json').read_text())
    assert summary['validity_policy'] == 'finite_cortex_signal'
    assert summary['n_grayordinates_valid'] == 3
    assert summary['run_lengths'] == [3, 3]

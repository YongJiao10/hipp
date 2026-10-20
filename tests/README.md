# Tests

Run from the repository root with the core-workflow interpreter. Keep numerical
threads at one and temporary products under repository `tmp/`:

```bash
export TMPDIR="$PWD/tmp"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
/home/yojiao/.conda/envs/py314/bin/python -m pytest tests -q --capture=sys --basetemp="tmp/pytest-$(date -u +%Y%m%dT%H%M%S)-$$"
```

The timestamp and process ID give each run a fresh `--basetemp` path; pytest
clears an existing base directory, so do not reuse one containing useful files.
Replace `tests` with a test module or directory to run a smaller set.
`--capture=sys` keeps Python output capture in memory, avoiding per-test NFS
temporary-file churn. Tests needing native-process output can request `capfd`.

## Organization

- Directory names follow the maintained workflow: `core_workflow`,
  `individual_workflow`, `iowa7t`, `exploratory_beta`, and `visualize`.
- Mathematical regressions use small arrays and independent expected values.
  These cover Pearson missingness, physical run offsets and variance weighting.
- Entry-point regressions use small on-disk inputs to check that actual
  producers and consumers apply the numerical rules, masks and run declarations.
  Keep these alongside the mathematical tests: they detect different failures.
- Use synthetic inputs and temporary outputs for rendering tests. Do not use
  historical cohort TSVs, PNGs or production result trees as test fixtures.

`core_workflow/conftest.py` supplies the core import path and shared factories
for reference metadata, baseline NPZ inputs, and HCP emotion-run design files.
Factories create only requested inputs; each test states its expected result.
Repository atlas metadata remains an explicit input where ROI identity matters.
Tests changing imported module globals must restore them with `monkeypatch`.

Coordinate regressions check space selection, transform direction and cache
rejection. Anatomical alignment still needs small-sample image inspection.

# Hipp Parcellation Step-by-Step Flow (Canonical)

This is the execution-ordered flow for active workflows.
Run from `scripts/core_workflow/` entrypoints.

## 1. Inputs and Cohort

1. `01_01_prepare_group_inputs.py`
2. `01_02_build_qc_manifest.py`

Recommended manifest output pattern:

```bash
python scripts/core_workflow/01_01_prepare_group_inputs.py \
  --workflow-branch kong2022_routeb \
  --step-manifest-out manifests/core_workflow/01_01_prepare_group_inputs.json
```

## 2. HippUnfold Runtime and Validation

1. `02_01_run_hippunfold_t1w_local.py`
2. `02_02_check_hippunfold_outputs.py`

## 3. Cortex References

- `03_01_build_kong2022_reference.py` for shared Kong2022 branch usage
- `03_02_build_kong2022_routea_reference.py` for RouteA-specific reference source

## 4. Hippocampal FC Materialization

1. `04_01_materialize_kong2022_roi_fc.py`
2. `04_02_materialize_intrinsic_fc.py`
3. `04_03_materialize_kong2019_fc.py` (Kong2019 branch only)

## 5. Group Atlas Construction

1. `05_01_run_group_atlas.py`
2. `05_02_run_spectral_fast_rerun.py` (optional urgent rerun)

Concurrency policy reminder: keep public batch launches at `<=10` jobs.

## 6. Output Materialization and Presentation

1. `06_03_materialize_group_labels.py`
2. `06_01_build_group_overview.py`
3. `06_04_build_cluster_cortex_panels.py`
4. `06_02_build_group_matrix.py`

## Branch Smoke Targets

- `kong2019`
- `kong2022_routea`
- `kong2022_routeb`

For branch flags, path roots, and exact artifact contracts, use:

- `core_workflow_step_registry.md`
- `scripts/core_workflow/step_registry.json`

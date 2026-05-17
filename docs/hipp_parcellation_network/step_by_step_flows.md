# Hipp Parcellation Step-by-Step Flow (Canonical)

This is the execution-ordered flow for active workflows.
Run from `scripts/core_workflow/` entrypoints.

## 1. Inputs and Cohort

1. `01_01_prepare_group_inputs.py`
2. `01_02_build_qc_manifest.py`

Recommended manifest output pattern:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/01_01_prepare_group_inputs.py \
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

Kong2022 group-gradient branch example:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/05_01_run_group_atlas.py \
  --methods gradient-spectral \
  --atlases kong2022 \
  --gradient-components 1,2,3
```

`--gradient-components` accepts `1`, `2`, `3`, `1,2`, and `1,2,3`; `1,2,3` is
the default. The branch computes diffusion gradients from group-averaged
Kong2022 vertex-to-ROI FC and then uses the selected gradient columns as the
spectral clustering feature matrix. This branch should be interpreted as a
full-continuous FC analysis: do not treat `top-percent` / `10perc` sparsified
and binarized FC as the default gradient input or as a maintained overview-mode
split.

Concurrency policy reminder: keep public batch launches at `<=10` jobs.

## 6. Output Materialization and Presentation

1. `06_03_materialize_group_labels.py`
2. `06_01_build_group_overview.py`
3. `06_04_build_cluster_cortex_panels.py`
4. `06_02_build_group_matrix.py`
5. `06_05_render_group_gradient_heatmaps.py` (optional RouteA `gradient-spectral` G1-G3 heatmap figure)

Maintained matrix note:

- `network-spectral` and `intrinsic-spectral` keep the `10perc/full` mode split.
- `gradient-spectral` exports route-level overviews from available `*_full`
  group roots only and does not use a `10perc` branch.
- Optional presentation artifact for RouteA full: a locked-scene 2x3 heatmap figure
  (`2mm`/`4mm` rows, `G1`/`G2`/`G3` columns) written to
  `present_group_t1only/kong2022_gradient-spectral_group_gradients_heatmap_routea_2mm4mm_g1g2g3.png`.
  The renderer uses a `plasma` palette and annotates each cell as
  `G# EV=xx.x%`, using the mean left/right share of that component's
  non-trivial diffusion eigenvalue among the exported non-trivial eigenvalues
  for that smoothing.

## Branch Smoke Targets

- `kong2019`
- `kong2022_routea`
- `kong2022_routeb`

For branch flags, path roots, and exact artifact contracts, use:

- `core_workflow_step_registry.md`
- `scripts/core_workflow/step_registry.json`

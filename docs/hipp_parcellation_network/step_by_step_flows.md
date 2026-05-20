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

1. `05_03_compute_hippomaps_gradients.py` for Kong2022 gradient artifacts
2. `05_01_run_group_atlas.py` for fixed-K group atlas outputs
3. `05_02_run_spectral_fast_rerun.py` (legacy/optional urgent rerun)

Kong2022 HippoMaps gradient artifact example:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/05_03_compute_hippomaps_gradients.py \
  --routes routea routeb \
  --smoothings 2mm 4mm \
  --hemis L R \
  --n-components 5 \
  --selected-components 1,2,3
```

Kong2022 group-gradient clustering example:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/05_01_run_group_atlas.py \
  --methods intrinsic-spectral network-spectral gradient-spectral wta-top \
  --atlases kong2022 \
  --gradient-root outputs_core/hcp7t_hippocampus/runs/<run_id>/gradients \
  --route-tag routea \
  --k-min 3 --k-max 3 --force-final-k 3
```

`05_03` keeps the Fisher-z transform before group averaging, applies column-wise
feature z-score across vertices, then calls
`GradientMaps(n_components=N, kernel="pearson")` on the z-scored group mean
Kong2022 vertex-to-ROI FC. It saves continuous full-surface gradients, full
lambdas, EV fractions, and group mean FC. `05_01` loads those artifacts for
`gradient-spectral` and selects `G1,G2,G3` by default. The legacy auto-K logic
remains in code behind `--auto-k`, but fixed final `K=3` is the maintained
default. Public partial-correlation spectral methods are retired.

Concurrency policy reminder: keep public batch launches at `<=10` jobs.

## 6. Output Materialization and Presentation

1. `06_03_materialize_group_labels.py`
2. `06_01_build_group_overview.py`
3. `06_04_build_cluster_cortex_panels.py`
4. `06_02_build_group_matrix.py`
5. `06_05_render_hippomaps_gradient_products.py`
6. `06_05_render_group_gradient_heatmaps.py` (legacy/optional locked Workbench heatmap)

Maintained output layout:

- Run root: `outputs_core/hcp7t_hippocampus/runs/<run_id>/`
- Smoke setup: `00_01_prepare_smoke_run.py` writes `inputs/subjects_smoke5.txt`,
  `manifests/smoke_command_manifest.json`, and `manifests/run_smoke_commands.sh`.
- All method overview PNGs: `present/overviews/` under the run root.
- Continuous HippoMaps gradient products: `present/gradients/`.
- Cortex/hippocampus panels: `present/cortex_panels/`.
- The maintained matrix uses fixed full-FC route roots only; no `10perc/full`
  presentation split and no `present_fixedk` directory are canonical.

HippoMaps continuous-gradient presentation uses official
`surfplot_canonical_foldunfold` folded/unfolded surfaces with `plasma`,
`unfoldAPrescale=False` for `den=512`, `tighten_cwindow=True`, row-shared
scaling, right color bars, EV plots from the full lambda vector, and top/bottom
25% endpoint FC contrast plots. For `den=512`, it treats `512` as HippoMaps v2
in the current Python process; no AP surfarea rescale is used.

## Branch Smoke Targets

- `kong2019`
- `kong2022_routea`
- `kong2022_routeb`

For branch flags, path roots, and exact artifact contracts, use:

- `core_workflow_step_registry.md`
- `scripts/core_workflow/step_registry.json`

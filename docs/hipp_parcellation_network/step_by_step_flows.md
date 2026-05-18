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
3. `05_03_compute_hippomaps_gradients.py` (optional continuous HippoMaps-style route)

Kong2022 group-gradient branch example:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/05_01_run_group_atlas.py \
  --methods gradient-spectral \
  --atlases kong2022 \
  --gradient-components 1,2,3
```

`--gradient-components` accepts `1`, `2`, `3`, `1,2`, and `1,2,3`; `1,2,3` is
the default. The branch computes diffusion gradients from group-averaged
Kong2022 vertex-to-ROI FC using BrainSpace (`GradientMaps`,
`approach="dm"`, `kernel="pearson"`, `n_components=5`, `sparsity=0.9`) and
then uses selected gradient columns as the spectral clustering feature matrix.
BrainSpace `sparsity=0.9` here means the smallest 90% of row values are
zeroed (top 10% retained). This branch should be interpreted as a
full-continuous FC analysis: do not treat `top-percent` / `10perc` sparsified
and binarized FC as the default gradient input or as a maintained overview-mode
split.

Concurrency policy reminder: keep public batch launches at `<=10` jobs.

HippoMaps-style continuous gradient example:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/05_03_compute_hippomaps_gradients.py \
  --routes routea routeb \
  --smoothings 2mm 4mm \
  --hemis L R
```

This route uses the same Fisher-z transform before group averaging, applies
column-wise feature z-score across vertices, then calls
`GradientMaps(n_components=N, kernel="pearson")` on the z-scored group mean
Kong2022 vertex-to-ROI FC. It saves continuous full-surface gradients, full
lambdas, EV fractions, and group mean FC. The default computes five components,
retains BrainSpace default `sparsity=0.9`, and renders G1-G3. It is upstream of
visualization only in this phase; `gradient-spectral` clustering behavior
remains in `05_01` unchanged.

## 6. Output Materialization and Presentation

1. `06_03_materialize_group_labels.py`
2. `06_01_build_group_overview.py`
3. `06_04_build_cluster_cortex_panels.py`
4. `06_02_build_group_matrix.py`
5. `06_05_render_hippomaps_gradient_products.py` (optional HippoMaps continuous gradient products)
6. `06_05_render_group_gradient_heatmaps.py` (legacy/optional RouteA `gradient-spectral` locked Workbench G1-G3 heatmap figure)

Maintained matrix note:

- `network-spectral` and `intrinsic-spectral` keep the `10perc/full` mode split.
- `gradient-spectral` exports route-level overviews from available `*_full`
  group roots only and does not use a `10perc` branch.
- Optional presentation artifact for RouteA full: a locked-scene 2x3 heatmap figure
  (`2mm`/`4mm` rows, `G1`/`G2`/`G3` columns) written to
  `present_group_t1only/kong2022_gradient-spectral_group_gradients_heatmap_routea_2mm4mm_g1g2g3.png`.
  The renderer uses Workbench as the single color engine (`palette_engine=workbench`)
  with default `palette=magma`; unsupported palette requests (for example `plasma`)
  fail directly and are not auto-mapped. Surface colors and bottom colorbars are
  both rendered by Workbench from identical metric palette ranges, and Pillow
  only performs composition/cropping/text layout. Cells are annotated as
  `G# EV=xx.x%`, using the mean left/right share of that component's eigenvalue
  among exported BrainSpace `G1-G5` eigenvalues for that smoothing.
- HippoMaps continuous-gradient presentation uses official
  `surfplot_canonical_foldunfold` folded/unfolded surfaces with `plasma`,
  `unfoldAPrescale=False` for `den=512`, `tighten_cwindow=True`, row-shared
  scaling, right color bars, EV plots from the full lambda vector, and
  top/bottom 25% endpoint FC contrast plots. For `den=512`, it treats `512` as
  HippoMaps v2 in the current Python process; no AP surfarea rescale is used.
  It reads `05_03` outputs and does not mutate clustering outputs. The default
  render is bilateral; pass `--hemis L` or `--hemis R` to write
  single-hemisphere products with hemisphere-tagged filenames.

## Branch Smoke Targets

- `kong2019`
- `kong2022_routea`
- `kong2022_routeb`

For branch flags, path roots, and exact artifact contracts, use:

- `core_workflow_step_registry.md`
- `scripts/core_workflow/step_registry.json`

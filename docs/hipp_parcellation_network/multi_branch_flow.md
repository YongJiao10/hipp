# Hipp Parcellation Multi-Branch Flow (Canonical)

This document defines branch semantics for the active core workflow surface.

## Branch Mainlines

- `kong2019`
- `kong2022_routea`
- `kong2022_routeb`

`kong2019` and `kong2022` are treated as parallel mainlines.
Within `kong2022`, RouteA and RouteB are parallel mainlines.

## Pipeline Order

Use numbered public entrypoints in `scripts/core_workflow/`:

- `01_xx` inputs and cohort manifests
- `02_xx` HippUnfold run and validation
- `03_xx` cortex reference generation
- `04_xx` hippocampal FC materialization
- `05_xx` group atlas construction
- `06_xx` label export and presentation outputs

## Branch Applicability

- Kong2019 branch:
  - shared input root uses Kong2019 FC banks (`04_03_materialize_kong2019_fc`)
  - uses `05_01_run_group_atlas` for group outputs
- Kong2022 RouteA branch:
  - may include RouteA reference build (`03_02_build_kong2022_routea_reference`)
  - uses `04_01`, `04_02`, `05_01`, optional `05_02`
  - may run `05_03` HippoMaps gradients followed by `gradient-spectral` in `05_01`
- Kong2022 RouteB branch:
  - uses shared Kong2022 reference build (`03_01`) and standard FC stages
  - uses `05_01`, optional `05_02`
  - may run `05_03` HippoMaps gradients followed by `gradient-spectral` in `05_01`

## Gradient-Spectral Branch

`gradient-spectral` is a Kong2022-only group atlas method. `05_03` first
computes HippoMaps/BrainSpace continuous gradients from the RouteA/RouteB
Kong2022 vertex-to-ROI FC caches: Fisher-z per subject, group mean, column-wise
feature z-score across vertices, then
`GradientMaps(n_components=N, kernel="pearson")`. The default computes G1-G5
and records lambdas/EV fractions.

`05_01_run_group_atlas.py` then consumes those saved artifacts from
`--gradient-root` for `gradient-spectral` and clusters selected gradient columns
(`G1,G2,G3` by default). Fixed final `K=3` is the maintained default for all
public presentation outputs. The legacy automatic K selector remains available
through `--auto-k` but is no longer the default policy. The partial-correlation
spectral branches are retired from public method choices.

Because the embedding is derived from group-averaged continuous FC profiles,
`gradient-spectral` is maintained as a full-FC route-level analysis. The
`top-percent` / `10perc` sparsified and binarized FC mode is not a maintained
presentation split.

## HippoMaps-Style Continuous Gradient Route

`05_03_compute_hippomaps_gradients.py` consumes the QC-passed subject manifest,
RouteA/RouteB Kong2022 vertex-to-ROI FC caches from `04_01`, surface support
masks, and group-average hippocampal surfaces. Outputs include group mean
Fisher-z FC, active and full-surface gradient arrays, selected `G1`-`G3` full
arrays with NaNs outside support, the full returned lambda vector, EV fractions,
and a manifest recording source FC paths, subject count, vertex count,
`external/hippomaps` import path, and HippoMaps commit.

`06_05_render_hippomaps_gradient_products.py` renders this continuous route
with official HippoMaps plotting code:
`hippomaps.plotting.surfplot_canonical_foldunfold(..., cmap="plasma",
unfoldAPrescale=False for den=512, tighten_cwindow=True, share="row",
color_bar="right")`. It also writes EV plots and Kong2022 ROI FC contrast plots
comparing vertices in the top 25% vs bottom 25% of each gradient.

Canonical run outputs live under
`outputs_core/hcp7t_hippocampus/runs/<run_id>/`: all maintained method overview
PNGs in `present/overviews/`, continuous gradient products in
`present/gradients/`, cortex panels in `present/cortex_panels/`, and manifests
under `manifests/`.

## Default Cohort Reminder

Default analysis cohort remains the QC-passed 151-subject set.
See `fmri_qc.md` for manifest paths and exclusion rationale.

## Canonical References

- Workflow overview: `core_workflow_overview.md`
- Step/IO matrix: `core_workflow_step_registry.md`
- Ordered run flow: `step_by_step_flows.md`
- Machine registry: `scripts/core_workflow/step_registry.json`

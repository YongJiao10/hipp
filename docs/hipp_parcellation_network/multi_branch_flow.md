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
  - uses `05_02_run_group_atlas` for group outputs
- Kong2022 RouteA branch:
  - may include RouteA reference build (`03_02_build_kong2022_routea_reference`)
  - uses `04_01`, `04_02`, `05_01`, then `05_02`; `05_03` is optional rerun orchestration
  - runs `05_01` HippoMaps gradients before `gradient-spectral` in `05_02`
- Kong2022 RouteB branch:
  - uses shared Kong2022 reference build (`03_01`) and standard FC stages
  - uses `05_01`, then `05_02`; `05_03` is optional rerun orchestration
  - runs `05_01` HippoMaps gradients before `gradient-spectral` in `05_02`

## Gradient-Spectral Branch

`gradient-spectral` is a Kong2022-only group atlas method. `05_01` first
computes HippoMaps/BrainSpace continuous gradients from the RouteA/RouteB
Kong2022 vertex-to-ROI FC caches: Fisher-z per subject, group mean, column-wise
feature z-score across vertices, then
`GradientMaps(n_components=N, kernel="pearson")`. The default computes raw G1-G5,
aligns component order and sign within each route to `4mm/hemi_L` when that combo
is present, writes only aligned gradient arrays, and records lambdas/EV fractions
in aligned component order.

`05_02_run_group_atlas.py` then consumes those aligned saved artifacts from
`--gradient-root` for `gradient-spectral` and clusters selected gradient columns
(`G1,G2,G3` by default). The `--gradient-components` argument accepts any ordered
non-empty subset of `{1,2,3}` (e.g. `1`, `2`, `3`, `1,2`, `1,3`, `2,3`).
Output paths encode the selected components as a `g{components}` subdirectory
under the coverage threshold (e.g. `gradient-spectral/kong2022/2mm/cov_0p80/g1/hemi_L/`)
so that separate experiments with different gradient component selections do not
overwrite each other. The maintained default evaluates the full `K=2..10` diagnostic curve for clustering methods while fixing final presentation labels at `K=3`. The legacy automatic K selector remains available through `--auto-k` but is no longer the default policy. The partial-correlation
spectral branches are retired from public method choices.

Because the embedding is derived from group-averaged continuous FC profiles,
`gradient-spectral` is maintained as a full-FC route-level analysis. The
`top-percent` / `10perc` sparsified and binarized FC mode is not a maintained
presentation split.

## Gradient-Spectral Confidence Maps

`05_04_compute_confidence_maps.py` computes per-vertex silhouette scores for
`gradient-spectral` parcellations in the G1/G2/G3 feature space — the same
space used by spectral clustering. The silhouette metric is `cosine` distance,
following Yeo et al. 2011. Outputs include full-surface silhouette arrays
(NaN outside support), per-cluster and global summary statistics, and a
manifest recording the source features and metric.

`06_07_render_confidence_maps.py` renders these silhouette maps on folded and
unfolded hippocampal surfaces via HippoMaps plotting. Raw silhouette values
remain signed; rendered figures use a Yeo/Buckner-style nonnegative display
window from 0 to the maximum positive silhouette. Output PNGs are bilateral
and live under `outputs_core/present/gradients/routea/{smoothing}/`.

## HippoMaps-Style Continuous Gradient Route

`05_01_compute_hippomaps_gradients.py` consumes the QC-passed subject manifest,
RouteA/RouteB Kong2022 vertex-to-ROI FC caches from `04_01`, surface support
masks, and group-average hippocampal surfaces. Outputs include group mean
Fisher-z FC, aligned active and full-surface gradient arrays, selected `G1`-`G3`
full arrays with NaNs outside support, the full lambda vector reordered to
aligned component order, EV fractions, and a manifest recording source FC paths,
subject count, vertex count, alignment provenance, `external/hippomaps` import
path, and HippoMaps commit.

`06_05_render_hippomaps_gradient_products.py` renders this continuous route
with official HippoMaps plotting code:
`hippomaps.plotting.surfplot_canonical_foldunfold(..., cmap="plasma",
unfoldAPrescale=False for den=512, tighten_cwindow=True, share="row",
color_bar="right")`. For `den=512`, the renderer treats `512` as HippoMaps v2
in the current Python process; no AP surfarea rescale is used. It also writes EV
plots and Kong2022 ROI FC contrast plots comparing vertices in the top 25% vs
bottom 25% of each aligned gradient. The default render is bilateral; `06_05`
renders each already-aligned hemisphere separately with its own colorbar and EV
labels before stitching the two sides. `--hemis L` or
`--hemis R` writes single-hemisphere gradient, EV, endpoint-contrast, and
manifest files with hemisphere tags in the filenames. This route is upstream and
continuous only. Alignment changes the gradient feature values consumed by
`gradient-spectral`, but does not change K selection, regularization, or label
assignment rules.

Canonical presentation outputs live under `outputs_core/present/`: maintained
method overview PNGs in `overviews/`, continuous gradient products in
`gradients/`, and cortex panels in `cortex_panels/`. Smoke tests stay isolated
under `outputs_core/smoke_runs/<run_id>/`.

Full-cohort group-atlas intermediates owned by the active core workflow live
under `outputs_core/work/group_atlas/`. The maintained RouteA full-FC fixed-K=3
root is `outputs_core/work/group_atlas/routea_full_k3_fisherz/`; all maintained
RouteA overview methods read from this root. `07_01_build_group_matrix.py` defaults to RouteA only, refreshes the WTA-top long-axis network-profile line products through `06_02_build_wta_long_axis_profiles.py`, and should include RouteB only after RouteB intermediates are generated. Kong2022 cluster-level functional atlas labels are preserved as two official
assignment variants: `fisherz` (`top1_fisher_z`) and `count`
(`topn_count_positive_roi`). Unsuffixed compatibility files point to `fisherz`,
while suffixed `_fisherz` and `_count` files and overview PNGs preserve both
results.

## Default Cohort Reminder

Default analysis cohort remains the QC-passed 151-subject set.
See `fmri_qc.md` for manifest paths and exclusion rationale.

## Canonical References

- Workflow overview: `core_workflow_overview.md`
- Step/IO matrix: `core_workflow_step_registry.md`
- Ordered run flow: `step_by_step_flows.md`
- Machine registry: `scripts/core_workflow/step_registry.json`

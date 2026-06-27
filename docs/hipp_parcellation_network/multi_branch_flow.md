# Hipp Parcellation Multi-Branch Flow (Canonical)

This document defines branch semantics for the active core workflow surface.

## Branch Mainlines

- `kong2019`
- `kong2022_routea`
- `kong2022_routea_17net`
- `kong2022_routea_ctxreg`
- `kong2022_routea_ctxdrop`
- `kong2022_routeb`

`kong2019` and `kong2022` are treated as parallel mainlines. Within `kong2022`, RouteA, the RouteA 17-net taxonomy variant, RouteA cortex-regression control, RouteA cortex-drop control, and RouteB are parallel mainlines. The cortex-control branches are isolated from uncontrolled RouteA outputs.

## Pipeline Order

Use numbered public entrypoints in `scripts/core_workflow/`:

- `01_xx` HippUnfold run and validation
- `02_xx` shared inputs and QC cohort manifests
- `03_xx` cortex reference generation
- `04_xx` hippocampal time-series control and FC materialization
- `05_xx` gradient and group atlas construction
- `06_xx` label export and presentation outputs
- `07_xx` presentation index

## Branch Applicability

- Kong2019 branch:
  - shared input root uses Kong2019 FC banks (`04_03_materialize_kong2019_fc`)
  - uses `05_02_run_group_atlas` for group outputs
- Kong2022 RouteA branch:
  - requires generated RouteA ArealMSHBM individualized cortex labels before `03_02`; see `core_workflow_artifact_lineage.md` for the external-backend artifact contract and evidence requirements
  - runs RouteA reference build (`03_02_build_kong2022_routea_reference`) from generated RouteA label files
  - uses `04_01`, `04_02`, `05_01`, then `05_02`; `05_03` is optional rerun orchestration
  - runs `05_01` HippoMaps gradients before `gradient-spectral` in `05_02`
- Kong2022 RouteA 17-net taxonomy branch:
  - reuses `shared_inputs/routea`, `surface`, RouteA 400-ROI FC caches, and RouteA HippoMaps gradients; it does not create a separate input tree or gradient tree
  - `03_02_build_kong2022_routea_reference.py` also writes compact `cortex_family17_networks.tsv` and `cortex_family17_network_timeseries.npy` summaries under each existing RouteA subject reference
  - runs `05_02_run_group_atlas.py` with `--route-tag routea_17net --network-taxonomy family17`, writing only taxonomy-dependent group outputs under `outputs_hcp/work/group_atlas/routea_17net_full_k3_fisherz`
- Kong2022 RouteA cortex-control branch:
  - starts from RouteA references and `surface` hippocampal time series
  - runs `04_00_materialize_cortex_regressed_surface_timeseries.py` to write controlled time series and diagnostics under `shared_inputs/routea_ctxreg_151` after run-wise local cortex regression
  - then runs `04_01`, `04_02`, `05_01`, and `05_02` with `--route-tag routea_ctxreg`; static RouteA references are read from `routea` and masks from `surface` through explicit root arguments rather than being copied
  - writes gradients under `outputs_hcp/work/hippomaps_gradients/routea_ctxreg` and group atlas outputs under `outputs_hcp/work/group_atlas/routea_ctxreg_full_k3_fisherz`
  - controls local cortical smear before FC materialization and does not redefine or overwrite uncontrolled RouteA outputs
- Kong2022 RouteA cortex-drop branch:
  - starts from RouteA references, `surface` hippocampal time series, subject cortex dtseries, generated RouteA individualized Kong2022 labels, and the same 20 mm local cortex-neighborhood definition used by `routea_ctxreg`
  - runs `04_01_materialize_kong2022_roi_fc.py --ctxdrop` to write isolated vertex-to-ROI FC caches under `shared_inputs/routea_ctxdrop_151`; hippocampal time series are not rewritten
  - for each hippocampal vertex and ROI, excludes valid cortex vertices within the 20 mm local neighborhood before computing that ROI mean time series and Pearson FC
  - maintained downstream outputs are `gradient-spectral` and `wta-top` only, with gradients under `outputs_hcp/work/hippomaps_gradients/routea_ctxdrop`, group atlas outputs under `outputs_hcp/work/group_atlas/routea_ctxdrop_full_k3_fisherz`, and shared local-neighbor hippocampal surface overlays rendered by `06_07_render_ctxdrop_neighbor_counts.py` under `outputs_hcp/present/local_cortex_neighbors/`
- Kong2022 RouteB branch:
  - uses shared Kong2022 reference build (`03_01`) and standard FC stages
  - uses `05_01`, then `05_02`; `05_03` is optional rerun orchestration
  - runs `05_01` HippoMaps gradients before `gradient-spectral` in `05_02`
  - writes maintained fixed-K=3 group atlas outputs under `outputs_hcp/work/group_atlas/routeb_full_k3_fisherz`

## Gradient-Spectral Branch

`gradient-spectral` is a Kong2022-only group atlas method. `05_01` first computes HippoMaps/BrainSpace continuous gradients from RouteA, RouteA cortex-control, RouteA cortex-drop, or RouteB Kong2022 vertex-to-ROI FC caches from `04_01`: Fisher-z per subject, group mean, column-wise feature z-score across vertices, then `GradientMaps(n_components=N, kernel="pearson", random_state=0)` by default. The default computes raw G1-G5 with a fixed diffusion initialization seed, aligns component order and sign within each route to `4mm/hemi_L` when that combo is present, writes only aligned gradient arrays, and records lambdas/EV fractions plus `gradient_random_state` in aligned component order.

`05_02_run_group_atlas.py` consumes those aligned saved artifacts from `--gradient-root` for `gradient-spectral` and clusters selected gradient columns (`G1,G2,G3` by default). Output paths encode selected components as a `g{components}` subdirectory under the coverage threshold so separate component experiments do not overwrite each other. The maintained default evaluates the full `K=2..10` diagnostic curve while fixing final presentation labels at `K=3`.

## Confidence And Presentation

`06_06_render_confidence_maps.py` renders silhouette maps under `outputs_hcp/present/gradients/{route_tag}/{smoothing}/`. `06_04_render_hippomaps_gradient_products.py` renders continuous gradients for `routea`, `routea_ctxreg`, `routea_ctxdrop`, and `routeb` from `outputs_hcp/work/hippomaps_gradients/{route}`. Continuous gradient PNGs are display-aligned across branches by default: `routea` is the sign reference, non-reference route/hemi/component maps are flipped only for rendering when their correlation to the matching `routea` component is negative, and per-component color ranges are shared across rendered routes within each smoothing level. The display decisions are recorded in the `06_04` manifests; the saved `05_01` arrays and downstream clustering inputs are not rewritten.

Canonical presentation outputs live under `outputs_hcp/present/`: maintained method overview PNGs in `overviews/`, continuous gradient products in `gradients/`, and cortex panels in `cortex_panels/`. For direct method-overview rendering, pass `--assignment-variant both` to `06_01_build_group_overview.py` when cluster-based outputs should include both `fisherz` and `count`, including `gradient-spectral`; WTA remains a single clusterless overview. The WTA-top overview uses a compact first column, an embedded fixed 20-bin transposed long-axis network profile in the second column, and the hippocampal surface render in the third column; standalone WTA long-axis profile code and PNG/TSV products are removed from the maintained workflow. Smoke tests stay isolated under `outputs_hcp/smoke_runs/<run_id>/`.

Full-cohort group-atlas intermediates owned by the active core workflow live under `outputs_hcp/work/group_atlas/`. The post-hoc `weighted_fisherz` comparison branch is generated by `05_05_build_weighted_fisherz_branch.py` under `outputs_hcp/work/group_atlas/routea_weighted_fisherz_full_k3/`; it preserves existing RouteA `fisherz` files, adds a `weighted_fisherz` assignment variant for `gradient-spectral`, and rewrites `wta-top` labels by reconstructing vertex-count weighted network time series and recomputing Fisher-z FC. The maintained RouteA full-FC fixed-K=3 root is `outputs_hcp/work/group_atlas/routea_full_k3_fisherz/`; the RouteA 17-net taxonomy root is `outputs_hcp/work/group_atlas/routea_17net_full_k3_fisherz/`; the RouteB fixed-K=3 root is `outputs_hcp/work/group_atlas/routeb_full_k3_fisherz/`; the RouteA cortex-regression root is `outputs_hcp/work/group_atlas/routea_ctxreg_full_k3_fisherz/`, and the RouteA cortex-drop root is `outputs_hcp/work/group_atlas/routea_ctxdrop_full_k3_fisherz/`. Kong2022 cluster-level functional atlas labels are preserved as two official assignment variants: `fisherz` (`top1_fisher_z`) and `count` (`topn_count_positive_roi`). WTA comparison outputs are four-way: canonical top20 count, weighted Fisher-z, canonical8 normalized top20 count, and family17 normalized top20 count. Normalized WTA means the top-20 positive ROI count divided by the finite available ROI count in each network; it is not Fisher-z averaging. The normalized HCP comparison roots are `outputs_hcp/work/group_atlas/routea_wta_normalized_top20_count_full_k3/` and `outputs_hcp/work/group_atlas/routea_17net_wta_normalized_top20_count_full_k3/`, rendered together under `outputs_hcp/present/overviews/wta_normalized/` as `top20_count.png` and `family17.png`. These comparison roots do not replace canonical `topn_count_positive_roi` WTA outputs.

## Default Cohort Reminder

Default analysis cohort remains the QC-passed 151-subject set. See `fmri_qc.md` for manifest paths and exclusion rationale.

## Canonical References

- Workflow overview: `core_workflow_overview.md`
- Step/IO matrix: `core_workflow_step_registry.md`
- Artifact lineage: `core_workflow_artifact_lineage.md`
- Ordered run flow: `step_by_step_flows.md`
- Machine registry: `scripts/core_workflow/step_registry.json`

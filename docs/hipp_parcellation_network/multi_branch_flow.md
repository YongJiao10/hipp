# Hipp Parcellation Multi-Branch Flow (Canonical)

This document defines branch semantics for the active core workflow surface.

## Branch Mainlines

- `kong2019`
- `kong2022_routea`
- `kong2022_routea_17net`
- `kong2022_routea_ctxreg`
- `kong2022_routea_ctxdrop`
- `kong2022_routeb`
- `lynch2024_20net`
- `lynch2024_7net`

`kong2019` and `kong2022` are treated as parallel mainlines. Within `kong2022`, unsuffixed RouteA is the canonical7 default; the RouteA 17-net taxonomy variant, RouteA cortex-regression control, RouteA cortex-drop control, and RouteB remain explicit parallel branches. The cortex-control branches are isolated from uncontrolled RouteA outputs. Lynch2024 branches are all-base-subject FASTANS/PFM comparison routes against Kong2022 RouteA with separate input and group-atlas roots; their NSI table is audit metadata, not a selection gate.

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
  - runs `05_02_run_group_atlas.py` with `--route-tag routea_17net --network-taxonomy family17 --methods gradient-spectral`, writing only taxonomy-dependent group outputs under `outputs_hcp/work/group_atlas/routea_17net_full_k3_fisherz`; WTA is not valid for family17
- Kong2022 RouteA canonical7 default:
  - is the existing RouteA numbered workflow, not a separate `_7net` pipeline or code folder
  - uses `03_02 --atlas-slug kong2022` to write direct `reference/kong2022/` artifacts from Route-A labels/dtseries
  - runs existing `04_01`, `05_01`, and `05_02` with unsuffixed `kong2022` / `routea`; FC, gradients, clustering, and K selection all record `network_taxonomy=canonical7`
  - keeps the direct 7-network FC cache separate from the 400-ROI cache used by ROI/WTA methods and renders through the existing overview entrypoints

Kong2022 WTA output uses the single direct ROI-registry path with
`normalized_top20_count`; canonical7 and canonical8 are taxonomy views of that
same assignment, while family17 remains gradient-only. Lynch2024 supports both
`gradient-spectral` and direct-network `wta-top`; Lynch WTA assigns supported
vertices by top-1 Fisher-z network value and uses the Fisher-z margin as
confidence, without a Kong ROI registry.
- Kong2022 RouteA cortex-control branch:
  - starts from RouteA references and `surface` hippocampal time series
  - runs `04_00_materialize_cortex_regressed_surface_timeseries.py` to write controlled time series and diagnostics under the formal `outputs_hcp/inputs/hcp/kong2022/shared_inputs/routea_ctxreg` root after run-wise local cortex regression
  - then runs `04_01`, `04_02`, `05_01`, and `05_02` with `--route-tag routea_ctxreg`; static RouteA references are read from `routea` and masks from `surface` through explicit root arguments rather than being copied
  - gradients are maintained under `outputs_hcp/work/hippomaps_gradients/routea_ctxreg` and group atlases under `outputs_hcp/work/group_atlas/routea_ctxreg`; run identifiers and validation evidence belong in manifests and logs, while legacy suffixed trees remain migration sources until the approved cleanup
  - controls local cortical smear before FC materialization and does not redefine or overwrite uncontrolled RouteA outputs
- Kong2022 RouteA cortex-drop branch:
  - starts from RouteA references, `surface` hippocampal time series, subject cortex dtseries, generated RouteA individualized Kong2022 labels, and the same 20 mm local cortex-neighborhood definition used by `routea_ctxreg`
  - runs `04_01_materialize_kong2022_roi_fc.py --ctxdrop` to write isolated vertex-to-ROI FC caches under the formal `outputs_hcp/inputs/hcp/kong2022/shared_inputs/routea_ctxdrop` root; hippocampal time series are not rewritten
  - for each hippocampal vertex and ROI, excludes valid cortex vertices within the 20 mm local neighborhood before computing that ROI mean time series and Pearson FC
  - maintained downstream outputs are `gradient-spectral` and `wta-top` only, with gradients under `outputs_hcp/work/hippomaps_gradients/routea_ctxdrop`, group atlases under `outputs_hcp/work/group_atlas/routea_ctxdrop`, and shared local-neighbor hippocampal surface overlays rendered by `06_07_render_ctxdrop_neighbor_counts.py` under `outputs_hcp/present/local_cortex_neighbors/`. Legacy suffixed trees are migration sources only and are not active resolver targets
- Lynch2024 FASTANS/PFM branches:
  - start from the full 151-subject base QC manifest via `02_03`; retain the NSI TSV/CSV only as an audit table, with no threshold or expected-count rule, and use external local FASTANS priors under `external/FASTANS/resources/PFM/priors/Lynch2024/`
  - run `03_04_run_lynch2024_fastans_pfm.py` to create official per-subject hard-label PFM dlabel/dscalar assets and hard-label component time series under `outputs_hcp/work/cortex_pfm/sub-<id>/lynch2024/`, then write current shared inputs under `outputs_hcp/inputs/hcp/lynch2024_fast_pfm_t1only/shared_inputs/pfm_hcp151/`
  - `lynch2024_20net` keeps exactly 20 unique Lynch parent networks after excluding `Noise`; `lynch2024_7net` uses the canonical merge from `config/cross_atlas_network_merge.json`
  - `gradient-spectral` uses one shared non-Noise Lynch20 component gradient bank with shifted-cosine surface affinity; `lynch2024_7net` does not keep a separate 8-column gradient bank
  - `05_02` passes `--profile-surface-root outputs_hcp/inputs/hcp/kong2022/shared_inputs/routea_ctxreg` for cluster-network profile annotation, matching Kong RouteA's cluster-mean hippocampal time-series correlation while leaving Lynch masks, FC caches, references, and gradient inputs under the Lynch shared root
  - Lynch K diagnostics use the same full-sample cluster-to-network FC redundancy gate as Kong2022; summaries record finite maximum profile similarity and redundant-pair counts
  - the maintained downstream outputs are `gradient-spectral` and direct-network `wta-top`, with current full-cohort group roots `outputs_hcp/work/group_atlas/lynch2024_20net_hcp151_full_k3_fisherz/` and `outputs_hcp/work/group_atlas/lynch2024_7net_hcp151_full_k3_fisherz/`; when `--methods` is omitted, both methods are generated for a Lynch atlas. Lynch WTA uses direct vertex-to-network Fisher-z profiles rather than the Kong ROI-registry normalized Top-20 contract.
- Kong2022 RouteB branch:
  - uses shared Kong2022 reference build (`03_01`) and standard FC stages
  - uses `05_01`, then `05_02`; `05_03` is optional rerun orchestration
  - runs `05_01` HippoMaps gradients before `gradient-spectral` in `05_02`
  - writes maintained fixed-K=3 group atlas outputs under `outputs_hcp/work/group_atlas/routeb_full_k3_fisherz`

## Gradient-Spectral Branch

`gradient-spectral` is maintained for Kong2022 ROI-registry routes and Lynch2024 component-network routes. `05_01` first computes HippoMaps/BrainSpace continuous gradients from vertex-to-ROI/component FC caches from `04_01`: Fisher-z per subject, group mean, column-wise feature z-score across vertices, then `GradientMaps(n_components=N, kernel="pearson", random_state=0)` by default. Kong400 keeps BrainSpace high-dimensional sparsity (`0.9`); low-dimensional Lynch component banks use full-graph affinity (`0.0`). The default computes raw G1-G5 with a fixed diffusion initialization seed, aligns component order and sign within each route to `4mm/hemi_L` when that combo is present, writes only aligned gradient arrays, and records lambdas/EV fractions plus `gradient_random_state` in aligned component order.

`05_02_run_group_atlas.py` consumes those aligned saved artifacts from `--gradient-root` for `gradient-spectral` and clusters selected gradient columns (`G1,G2,G3` by default). Output paths encode selected components as a `g{components}` subdirectory under the coverage threshold so separate component experiments do not overwrite each other. The maintained default evaluates K=2..8, records automatic K selection, and uses explicit default K=3 only for primary labels and overview display.

## Confidence And Presentation

`06_06_render_confidence_maps.py` renders silhouette maps under `outputs_hcp/present/gradients/{route_tag}/{smoothing}/`. `06_04_render_hippomaps_gradient_products.py` renders continuous gradients for `routea`, `routea_ctxreg`, `routea_ctxdrop`, and `routeb` from `outputs_hcp/work/hippomaps_gradients/{route}`. Continuous gradient PNGs are display-aligned across branches by default: `routea` is the sign reference, non-reference route/hemi/component maps are flipped only for rendering when their correlation to the matching `routea` component is negative, and per-component color ranges are shared across rendered routes within each smoothing level. The display decisions are recorded in the `06_04` manifests; the saved `05_01` arrays and downstream clustering inputs are not rewritten.

Canonical presentation outputs live under `outputs_hcp/present/`: maintained method overview PNGs in `overviews/`, continuous gradient products in `gradients/`, and cortex panels in `cortex_panels/`. For direct method-overview rendering, pass `--assignment-variant both` to `06_01_build_group_overview.py` when cluster-based outputs should include both `fisherz` and `normalized_top20_count`, including `gradient-spectral`; the only clusterless WTA overview is the Kong2022 ROI-registry `normalized_top20_count` product. The WTA-top overview uses a compact first column, an embedded fixed 20-bin transposed long-axis network profile in the second column, and the hippocampal surface render in the third column; standalone WTA long-axis profile code and PNG/TSV products are removed from the maintained workflow. Smoke tests stay isolated under `outputs_hcp/smoke_runs/<run_id>/`.

Full-cohort group-atlas intermediates owned by the active core workflow live under `outputs_hcp/work/group_atlas/`. The post-hoc `weighted_fisherz` comparison branch is generated by `05_05_build_weighted_fisherz_branch.py` under `outputs_hcp/work/group_atlas/routea_weighted_fisherz_full_k3/`; it preserves existing RouteA `fisherz` files and adds a `weighted_fisherz` assignment variant for `gradient-spectral` only. The maintained multi-dataset canonical7 resolver keeps the published comparison namespaces (`kong2022_7net_*` for HCP/Iowa and `routea_7net_*` for Stanford), but every active Kong gradient/WTA input resolves inside `gradient-spectral/kong2022` or `wta-top/kong2022`; taxonomy is read from the manifest. The former unsuffixed canonical8 gradient/WTA subtrees used by that resolver were moved under each dataset's `work/archive/8net_20260909/`; only an explicit `canonical8` request may address that historical namespace. The RouteA 17-net taxonomy root is `outputs_hcp/work/group_atlas/routea_17net_full_k3_fisherz/`; the RouteB fixed-K=3 root is `outputs_hcp/work/group_atlas/routeb_full_k3_fisherz/`; HCP cortex-control branches use the formal direct roots `outputs_hcp/work/group_atlas/routea_ctxreg/` and `outputs_hcp/work/group_atlas/routea_ctxdrop/`. Kong2022 cluster-level functional atlas labels retain `fisherz` (`top1_fisher_z`) and `normalized_top20_count` (`top20_count_normalized_by_network_roi_count`) for cluster-based methods; these are not alternative WTA definitions. The only WTA assignment is the Kong2022 ROI-registry `normalized_top20_count` product: top-20 positive ROI/component counts divided by finite available ROI/component counts in each network. Raw `count` is neither registered nor generated.

The formal gradient-spectral affinity is fixed to shifted-cosine. Group and
overview manifests preserve its name, definition, source group manifests, and
connected-graph diagnostics; overview filenames are stable and visible titles
identify shifted-cosine. Formal release-audit and refresh maintenance helpers were retired from this
checkout. Do not invoke or restore deleted maintenance paths. Use the active
numbered workflow, generated group manifests, and the multi-dataset suite
readiness check for current validation. Archive movement remains gated on an
exact inventory and explicit user confirmation.

## Default Cohort Reminder

Default Kong2022 analysis cohort remains the QC-passed 151-subject set. Lynch2024 comparison routes use the same 151-subject base QC manifest from `02_03`; NSI values are recorded for audit only. Historical 78-subject NSI-filtered outputs remain separate and are not current inputs. See `fmri_qc.md` for the base QC manifest paths and exclusion rationale.

## Canonical References

- Workflow overview: `core_workflow_overview.md`
- Step/IO matrix: `core_workflow_step_registry.md`
- Artifact lineage: `core_workflow_artifact_lineage.md`
- Ordered run flow: `step_by_step_flows.md`
- Machine registry: `scripts/core_workflow/step_registry.json`

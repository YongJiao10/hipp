# Hipp Parcellation Multi-Branch Flow (Canonical)

This document defines branch semantics for the active core workflow surface.

## Branch Mainlines

- `kong2022_pfm`
- `kong2022_pfm_17net`
- `kong2022_pfm_ctxreg`
- `kong2022_pfm_ctxreg_10mm`
- `kong2022_fixed`
- `kong2022_fixed_17net`
- `lynch2024_20net`
- `lynch2024_7net`

Within `kong2022`, unsuffixed `pfm` is the canonical7 individualized default; the PFM 17-net taxonomy variant (`pfm_17net`), PFM cortex-regression controls (`pfm_ctxreg` at 20 mm and `pfm_ctxreg_10mm` at 10 mm), Fixed standard template (`fixed`, canonical7), and Fixed 17-net (`fixed_17net`) remain explicit parallel branches. The cortex-control branches are isolated from uncontrolled PFM outputs and from each other. Lynch2024 branches are all-base-subject FASTANS/PFM comparison routes against Kong2022 PFM with separate input and group-atlas roots; their NSI table is audit metadata, not a selection gate.

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

- Kong2022 PFM (`pfm`) branch:
  - requires generated PFM ArealMSHBM individualized cortex labels before `03_02`; see `core_workflow_artifact_lineage.md` for the external-backend artifact contract and evidence requirements
  - runs PFM reference build (`03_02_build_kong2022_routea_reference`) from generated PFM label files
  - uses `04_01`, `04_02`, `05_01`, then `05_02`; `05_03` is optional rerun orchestration
  - runs `05_01` HippoMaps gradients before `gradient-spectral` in `05_02`
- Kong2022 17-net taxonomy branches (`pfm_17net` and `fixed_17net`):
  - reuse `shared_inputs/pfm` (or `shared_inputs/fixed`), `surface`, 400-ROI FC caches, and `pfm` (or `fixed`) HippoMaps gradients; they do not create a separate input tree or gradient tree
  - `03_02_build_kong2022_routea_reference.py` (and `03_01`) write compact `cortex_family17_networks.tsv` and `cortex_family17_network_timeseries.npy` summaries under each subject reference
  - run `05_02_run_group_atlas.py` with `--route-tag pfm_17net` (or `fixed_17net`) `--network-taxonomy family17`, the source profile and gradient method, and `--reuse-fixed-cluster-root` pointing to the matching canonical7 profile root. Spatial clusters and K diagnostics are shared; taxonomy-dependent outputs are written under `outputs_hcp/work/group_atlas/pfm_17net` (or `fixed_17net`)
  - Family17 `wta-top` is a clusterless direct run: `--methods wta-top --route-tag pfm_17net` (or `fixed_17net`) `--network-taxonomy family17` without `--reuse-fixed-cluster-root` or `--gradient-root`; it applies the same `normalized_top20_count` assignment to the `family17` column of the Kong2022 ROI registry and writes `<route>/wta-top/kong2022/{2mm,4mm}/cov_0p80/hemi_{L,R}/`.
- Kong2022 PFM canonical7 default:
  - is the existing PFM numbered workflow, not a separate `_7net` pipeline or code folder
  - uses `03_02 --atlas-slug kong2022` to write direct `reference/kong2022/` artifacts from PFM labels/dtseries
  - runs existing `04_01`, `05_01`, and `05_02` with unsuffixed `kong2022` / `pfm`; FC, gradients, clustering, and K selection all record `network_taxonomy=canonical7`
  - keeps the direct 7-network FC cache separate from the 400-ROI cache used by ROI/WTA methods and renders through the existing overview entrypoints

Kong2022 WTA output uses the single direct ROI-registry path with
`normalized_top20_count` for `canonical7` and `family17`. Lynch2024 supports both
`gradient-spectral` and direct-network `wta-top`; Lynch WTA assigns supported
vertices by top-1 Fisher-z network value and uses the Fisher-z margin as
confidence, without a Kong ROI registry.
- Kong2022 PFM cortex-control branches (`pfm_ctxreg`, signal branch `ctxreg`; `pfm_ctxreg_10mm`, signal branch `ctxreg_10mm`):
  - both regress, per run, the mean of valid cortex within a native T1w radius out of each hippocampal vertex series; `pfm_ctxreg` is defined at 20 mm and `pfm_ctxreg_10mm` at 10 mm (`lib/branch_config.CTXREG_ROUTE_DISTANCE_MM`)
  - 10 mm is the smallest radius at which essentially every hippocampal vertex has cortex in range (a handful do not) and matches the 6-10 mm range of earlier subcortical work; its regressor is mostly default-mode network (DMN), so the two radii bracket the same boundary-signal removal, stronger at 10 mm. Both are conservative sensitivity analyses
  - start from PFM references and `surface` hippocampal time series
  - run `04_00_materialize_cortex_regressed_surface_timeseries.py --route-tag {pfm_ctxreg,pfm_ctxreg_10mm}` to write controlled time series and diagnostics under the formal `outputs_hcp/inputs/kong2022/shared_inputs/<route>` root after run-wise local cortex regression; `--distance-mm` defaults to the route's radius and a different value is rejected
  - per-vertex diagnostics record neighbor counts, variance before/after, and `local_cortex_corr`, the pooled Pearson correlation between each hippocampal vertex and its local cortex mean after run-wise mean centering (the same pooling as FC); the summary declares this in `local_cortex_corr_definition`
  - then run `04_01`, `04_02`, `05_01`, and `05_02` with the route's controlled root and `--route-tag`; static PFM references are read from `pfm` and masks from `surface` through explicit root arguments rather than being copied
  - gradients are maintained under `outputs_hcp/work/hippomaps_gradients/<route>/topsim` and group atlases under `outputs_hcp/work/group_atlas/<route>/{topsim/gradient-ward,wta-top}`
  - control local cortical smear before FC materialization and do not redefine or overwrite uncontrolled PFM outputs
- Local cortex-control diagnostic (not a branch; the former `pfm_ctxdrop` branch is retired):
  - `06_07_render_local_cortex_control_diagnostics.py` compares standard Kong2022 PFM 400-ROI FC with ctxreg (per-run regression of the local cortex mean from each hippocampal series) and ctxdrop (per-vertex exclusion of local cortex from the 400 ROI averages) at 3, 6, 10 and 20 mm, and reports the normalized top-20 WTA winner change (the `05_02` `normalized_top20_count` rule), the canonical7 make-up of the local cortex, hippocampus-to-cortex correlation by distance ring and a boundary identity test
  - per-subject arrays go under `outputs_hcp/work/local_cortex_control/sub-<id>/` and the figure, summary TSV and manifest under `outputs_hcp/present/local_cortex_control/`; nothing downstream consumes them
  - The figure shows only the ctxreg diagnostics at 10 and 20 mm (regressor size and DMN make-up, hippocampus-regressor correlation, WTA change maps, distance-ring coupling, network FC change and WTA share); target-side exclusion and the boundary identity test are recorded in the summary TSV only.
  - in HCP 7T, target-side exclusion changes the winning network for only about 1-4% of hippocampal vertices because Kong2022 ROIs lose only a minority of their vertices, while ctxreg changes about a quarter to two fifths of vertices and that change is reproduced by regressing only cortex within 3 mm, where hippocampal and cortical signals are strongly shared; `pfm_ctxreg` and `pfm_ctxreg_10mm` are therefore read as conservative sensitivity analyses that removes boundary-shared signal, possibly including genuine hippocampal or subicular signal
- Lynch2024 FASTANS/PFM branches:
  - start from the full base QC manifest via `02_03`; retain the NSI TSV/CSV only as an audit table, with no threshold or expected-count rule, and use external local FASTANS priors under `external/FASTANS/resources/PFM/priors/Lynch2024/`
  - run `03_04_run_lynch2024_fastans_pfm.py` to create official per-subject hard-label PFM dlabel/dscalar assets and hard-label component time series under `outputs_hcp/work/cortex_pfm/sub-<id>/lynch2024/`, then write current shared inputs under `outputs_hcp/inputs/lynch2024/shared_inputs/pfm/`
  - `lynch2024_20net` keeps exactly 20 unique Lynch parent networks after excluding `Noise`; `lynch2024_7net` uses the canonical merge from `config/cross_atlas_network_merge.json`
  - `gradient-spectral` uses one shared non-Noise Lynch20 component gradient bank with shifted-cosine surface affinity; `lynch2024_7net` does not keep a separate gradient bank
  - `05_02` annotates cluster-network profiles from the uncontrolled `outputs_hcp/inputs/kong2022/shared_inputs/surface` hippocampal time series (the default `--surface-root`), the same cluster-mean time-series correlation used by Kong PFM; Lynch masks, FC caches, references, and gradient inputs stay under the Lynch shared root. CtxReg residuals are rejected as a Lynch profile root because they belong to a different signal branch
  - Lynch K diagnostics use the same full-sample cluster-to-network FC redundancy gate as Kong2022; summaries record finite maximum profile similarity and redundant-pair counts
  - the maintained downstream outputs are `gradient-spectral` and direct-network `wta-top`, with current full-cohort group roots `outputs_hcp/work/group_atlas/lynch2024_20net/` and `outputs_hcp/work/group_atlas/lynch2024_7net/`; when `--methods` is omitted, both methods are generated for a Lynch atlas. Lynch WTA uses direct vertex-to-network Fisher-z profiles rather than the Kong ROI-registry normalized Top-20 contract.
- Kong2022 Fixed (`fixed` / `fixed_17net`) branch:
  - uses shared Kong2022 reference build (`03_01`) and standard FC stages
  - uses `05_01`, then `05_02`; `05_03` is optional rerun orchestration
  - runs `05_01` HippoMaps gradients before `gradient-spectral` in `05_02`
  - writes maintained fixed-K=3 group atlas outputs under `outputs_hcp/work/group_atlas/fixed` (and `fixed_17net` for original 17 networks)

## Gradient-Spectral Branch

`gradient-spectral` is maintained for Kong2022 ROI-registry routes and Lynch2024 component-network routes. `05_01` first computes HippoMaps/BrainSpace continuous gradients from vertex-to-ROI/component FC caches from `04_01`: Fisher-z per subject, group mean, and column-wise feature z-score across vertices by default. It constructs nonnegative Pearson affinity explicitly and embeds it with the exact, seed-free diffusion map without a second sparsification pass. Kong400 defaults to feature sparsity `0.9`; low-dimensional Lynch component banks use full-graph affinity (`0.0`). FC column scaling and the sparsification stage can also be selected independently for comparisons. The default computes raw G1-G5, requires the `4mm/hemi_L` anchor for each route and preprocessing setting, writes only aligned gradient arrays, and records lambdas/EV fractions in aligned component order.

`05_02_run_group_atlas.py` consumes those aligned saved artifacts from `--gradient-root` for `gradient-spectral` and clusters selected gradient columns (`G1,G2,G3` by default). Output paths encode selected components as a `g{components}` subdirectory under the coverage threshold so separate component experiments do not overwrite each other. The maintained default evaluates K=2..8, records automatic K selection, and uses explicit default K=3 only for primary labels and overview display.

## Confidence And Presentation

`06_06_render_confidence_maps.py` renders silhouette maps under `outputs_hcp/present/gradients/{route_tag}/{smoothing}/`, or `outputs_hcp/present/gradients/{route_tag}/{profile}/{method}/{smoothing}/` with `--profile`. `05_04` measures the silhouette with Euclidean distance for `gradient-ward` and `gradient-gmm` and cosine distance for the spectral methods unless `--metric` is given. `06_04_render_hippomaps_gradient_products.py` renders continuous gradients for `pfm`, `pfm_ctxreg`, and `fixed` from `outputs_hcp/work/hippomaps_gradients/{route}`. Continuous gradient PNGs are display-aligned across branches by default: `pfm` is the sign reference, non-reference route/hemi/component maps are flipped only for rendering when their correlation to the matching `pfm` component is negative, and per-component color ranges are shared across rendered routes within each smoothing level. The display decisions are recorded in the `06_04` manifests; the saved `05_01` arrays and downstream clustering inputs are not rewritten.

Canonical presentation outputs live under `outputs_hcp/present/`: maintained method overview PNGs in `overviews/<route>/`, continuous gradient products in `gradients/`, and cortex panels in `cortex_panels/`. For direct method-overview rendering, pass the required `--smoothing 2mm` or `--smoothing 4mm` to `06_01_build_group_overview.py`. For Kong2022 gradient methods, `--assignment-variant both` produces the ROI Fisher-z/OVR, network-time-series Fisher-z/OVR, and normalized Top20 figures; for other Kong2022 cluster methods it selects the default `network_ts_fisherz` and `normalized_top20_count`. The only clusterless WTA overview is the Kong2022 ROI-registry `normalized_top20_count` product. The WTA-top overview uses a compact first column, an embedded fixed 20-bin transposed long-axis network profile in the second column, and the hippocampal functional-atlas panel (folded surface plus unfolded maps) in the third column; standalone WTA long-axis profile code and PNG/TSV products are removed from the maintained workflow. Run-owned smoke products and non-official execution records use `outputs_hcp/runs/<run_id>/`.

Full-cohort group-atlas intermediates owned by the active core workflow live under `outputs_<dataset>/work/group_atlas/`. HCP branches use `pfm`, `pfm_17net`, `pfm_ctxreg`, `pfm_ctxreg_10mm`, `fixed`, `fixed_17net`, `lynch2024_20net`, and `lynch2024_7net`. The network-time-series assignment (`network_ts_fisherz`) is computed by `05_02` inside each formal run directory; there is no separate weighted branch. Stanford and Iowa cohort products use `<route>/<cohort>/`, with an unsuffixed route for 36P. The maintained canonical7 visualizers resolve `gradient-spectral/kong2022` and `wta-top/kong2022` within those formal roots; the taxonomy is declared in the manifest. Seven-network views do not create a separate `routea_7net` or `kong2022_7net` input root. Family17 branches reuse their PFM or Fixed 400-ROI FC and gradient banks. Historical output trees remain archived and are not runtime substitutes for missing formal products. Kong2022 cluster-level functional atlas labels default to `network_ts_fisherz` (`top1_network_ts_fisher_z`) and retain ROI `fisherz` (`top1_fisher_z`) and `normalized_top20_count` (`top20_count_normalized_by_network_roi_count`) as comparisons for cluster-based methods. The only Kong2022 WTA assignment is the ROI-registry `normalized_top20_count` product: top-20 positive ROI counts divided by finite available ROI counts in each network. Raw `count` is neither registered nor generated.

The formal gradient-spectral affinity is fixed to shifted-cosine. Group and
overview manifests preserve its name, definition, source group manifests, and
connected-graph diagnostics; overview filenames are stable and visible titles
identify shifted-cosine. Formal release-audit and refresh maintenance helpers were retired from this
checkout. Do not invoke or restore deleted maintenance paths. Use the active
numbered workflow, generated group manifests, and the multi-dataset suite
readiness check for current validation. Archive movement remains gated on an
exact inventory and explicit user confirmation.

## Default Cohort Reminder

Default Kong2022 analysis cohort remains the current QC-passed set. Lynch2024 comparison routes use the same base QC manifest from `02_03`; NSI values are recorded for audit only. Historical NSI-filtered outputs remain separate and are not current inputs. See `fmri_qc.md` for the base QC manifest paths and exclusion rationale.

## Canonical References

- Workflow overview: `core_workflow_overview.md`
- Step/IO matrix: `core_workflow_step_registry.md`
- Artifact lineage: `core_workflow_artifact_lineage.md`
- Ordered run flow: `step_by_step_flows.md`
- Machine registry: `scripts/core_workflow/step_registry.json`

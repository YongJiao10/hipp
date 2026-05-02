# Kong2022 Route B Flow

## Summary

Route B supports two cortex reference sources:

- `group` (default): public atlas label file
  - `external/Schaefer2018_Kong2022/Schaefer2018_400Parcels_Kong2022_17Networks_order.dlabel.nii`
- `route_a` (optional): Route A individualized `lh/rh_labels.txt` per subject

No individualized cortex PFM is run in this route. Each subject still gets:

- 400-ROI cortex timeseries (after cortex tSNR gate)
- merged canonical-8 cortex timeseries derived from those ROI timeseries (kept for compatibility; not used by the three active assignment methods)

Atlas slug is fixed to `kong2022`.

For T1only, Route B must compute its own cortex tSNR gate and run slicing from the subject dtseries. Do not consume T2only Kong2019/PFM cortex reference outputs.

## Pipeline Steps

1. Build per-subject Kong2022 reference:
   - source `group`:
     - script: `scripts/cortex/extract_cortex_kong2022_roi_timeseries.py`
     - batch entrypoint: `scripts/cortex/run_cortex_kong2022_reference_batch.py`
   - source `route_a`:
     - script: `scripts/cortex/extract_cortex_kong2022_route_a_roi_timeseries.py`
     - batch entrypoint: `scripts/cortex/run_cortex_kong2022_route_a_reference_batch.py`
   - pipeline switch: `run_t1only_kong2022_route_b_pipeline.py --cortex-reference-source {group|route_a}`
   - output root: `outputs_migration/hipp_group_inputs_shared/sub-{sid}/reference/kong2022/`
2. Materialize hippocampal vertex-to-ROI FC:
   - script: `scripts/hipp_parcellation_network/materialize_kong2022_vertex_to_roi_fc.py`
   - requires `reference/common/runs/run-{1..4}/reference_summary.json` from Step 1
   - output root: `outputs_migration/hipp_group_inputs_shared/sub-{sid}/fc/kong2022/`
3. Run group atlas driver:
   - script: `scripts/hipp_parcellation_network/run_merged_canonical_group_atlas.py`
   - include `--atlases kong2022`
4. Render present overview:
   - script: `scripts/hipp_parcellation_network/build_present_group_overview_merged_canonical.py`
   - pipeline default methods: `group-intrinsic-spectral`, `group-network-spectral`
   - naming must include route/mode suffix:
     - `kong2022_{method}_group_overview_{route-tag}_{mode-tag}.png`
   - maintained export root is only: `present_group_t1only/`
5. One-shot 8-way matrix export (maintained default):
   - script: `scripts/hipp_parcellation_network/build_present_group_t1only_matrix.py`
   - fixed matrix: `2 methods x 2 routes x allon/alloff = 8`
   - strict root naming: must use `..._routea_allon/alloff` and `..._routeb_allon/alloff` (no fallback aliases)
   - fixed consolidated manifest: `present_group_t1only/group_overview_manifest_t1only_8way.json`
   - fixed consolidated README: `present_group_t1only/README.md`
6. Render standalone cluster-ROI cortex figures (independent from overview):
   - script: `scripts/hipp_parcellation_network/build_cluster_roi_cortex_independent.py`
   - single output root: `present_group_t1only/runtime/` (cluster figures, manifests, and quicklook index all co-located)
   - merged-hemisphere quicklook overviews are written directly under `present_group_t1only/`
   - cluster methods keep per-smoothing quicklooks: `cortex_<atlas>_<method>_<smoothing>_overview.png`

## Upstream Common Inputs

Route B now stores atlas-invariant cortex upstream products in `reference/common/`:

- cortex tSNR masks:
  - `reference/common/tsnr_gate/cortex_left_valid_mask.npy`
  - `reference/common/tsnr_gate/cortex_right_valid_mask.npy`
- run boundaries:
  - `reference/common/runs/run-{1..4}/reference_summary.json`

These products are atlas-invariant and are consumed by downstream FC stages.

Historical legacy outputs under `reference/kong2022/{tsnr_gate,runs}` are migrated into `reference/common/` and removed by the batch entrypoint.

Run storage policy for Route B is metadata slicing, not duplicate arrays:

- keep full-length arrays once:
  - `reference/kong2022/cortex_roi_parcel_timeseries.npy`
  - `reference/kong2022/cortex_canonical_network_timeseries.npy`
- keep run boundaries in:
  - `reference/common/runs/run-{1..4}/reference_summary.json`
- downstream scripts must read full arrays and slice `[run_start:run_stop]` at runtime.

## Group Outputs Specific to Kong2022

For each `(method, smoothing, coverage, hemi)` combo:

- `cluster_network_profile.npy` still reports merged canonical-8 cluster profiles (cluster method only)
- `cluster_roi_profile.npy` stores `K_final x 400` cluster-to-ROI Fisher-z profile
- `cluster_roi_top20.json` stores per-cluster:
  - strongest positive 20 ROI
  - strongest negative 20 ROI
  - each ROI annotated with `family17` and `canonical8`
- `cluster_roi_registry.tsv` stores the full 400-ROI registry

If a combo cannot produce a valid final `K`, the combo is soft-skipped: `group_atlas_manifest.json` is written with `status=skipped_*`, `k_selection_summary.tsv/json` is still emitted for diagnostics, and downstream renderers should skip that combo.

For spectral methods, `run_merged_canonical_group_atlas.py` supports `--k-selection-mode redundancy-aware`:

- candidate-`K` rejection gates:
  - redundant: `max_profile_similarity > 0.80` or `redundant_pair_count > 0`
  - nontrivial fail: existing min-size/connectedness/coherence checks fail
- final selection in redundancy-aware mode (three-tier over the survivor pool):
  1. 1-SE rule: keep survivors within 1 SE of the survivor-pool best mean ARI
  2. local-max preference: among within-1SE survivors, prefer ARI-curve local maxima
  3. parsimony tiebreak: smallest K when quality is genuinely tied

For `group-intrinsic-spectral`, cluster network assignment uses raw TopN positive ROI majority (`N=min(20, n_positive_roi)` from `mean_z > 0`), not canonical `8 x T` cortex timeseries. One-sided t-test + BH-FDR outputs are still computed for stats fields only. If no positive ROI exists, status is `unassigned` with reason `no_positive_roi`. If TopN ROI belong to only one network, `2nd` is recorded as `none` (not duplicated from `1st`).

`group-network-top` and `group-network-all` are legacy branches and are retired from maintained default runs.

## Independent Cluster-ROI Cortex Figure Outputs

All outputs are co-located under `present_group_t1only/runtime/` for each `(method, smoothing, coverage, hemi)` combo:

- one folder per cluster with:
  - `cluster_cortex_toproi_4view.png`
  - `cluster_roi_stats.tsv`
  - `cluster_roi_stats.json`
- combo-level `cluster_roi_cortex_combo_manifest.json`
- top-level `cluster_roi_cortex_runtime_manifest.json`
- `combo_manifests/` per-combo index JSON
- top-level `cluster_roi_cortex_manifest.json` (quicklook index)

Top-level `present_group_t1only/` contains the maintained 8-way overview PNG set:

- `kong2022_group-intrinsic-spectral_group_overview_routea_allon.png`
- `kong2022_group-intrinsic-spectral_group_overview_routea_alloff.png`
- `kong2022_group-intrinsic-spectral_group_overview_routeb_allon.png`
- `kong2022_group-intrinsic-spectral_group_overview_routeb_alloff.png`
- `kong2022_group-network-spectral_group_overview_routea_allon.png`
- `kong2022_group-network-spectral_group_overview_routea_alloff.png`
- `kong2022_group-network-spectral_group_overview_routeb_allon.png`
- `kong2022_group-network-spectral_group_overview_routeb_alloff.png`

Historical `cortex_*` quicklook outputs are deprecated from the maintained top-level export set.

Statistical and selection policy:

- one-sided one-sample t-tests (vs 0) on subject-level cluster-to-ROI Fisher-z values
- BH-FDR correction per cluster and per sign (positive/negative), threshold `q < 0.05`
- intrinsic cluster mode: Sig-only selection, capped at top 20 significant ROI per sign
- clusterless mode: global top connections use TopN positive + TopN negative edges (total up to `2N`) with significance thresholding (`q < 0.05`)
- network hue follows canonical8 color mapping; sign is encoded by shade (negative uses lighter tone)
- cluster figure legend is network-color based (no per-ROI name legend)
- default method set for this stage is `group-intrinsic-spectral`, `group-network-spectral`
- rendering uses first scene view only (single 4-panel cortex layout), no multi-view expansion
- overview builder composes from runtime cluster panels and keeps those panel PNGs in runtime
- per-cluster figure includes network legend with sign-coding note and selected ROI counts
- merged overview panels annotate `n=total (+pos/-neg)` for each subplot

## Method Support

Active default methods for Kong2022:

- `group-intrinsic-spectral`
- `group-network-spectral`

Legacy (retired from default runs):

- `group-network-top`
- `group-network-all`

Output policy:

- Do not create `present_group_t1only_*` sibling directories.
- Keep only the single maintained export root `present_group_t1only/`.

## No-Speed Consistency Check (2026-05-02)

If routeA and routeB overviews appear identical, re-run all four route/mode combos with:

- `run_t1only_spectral_fast_rerun.py --no-redundancy-early-stop`
- route/mode isolated roots (`..._routea_allon`, `..._routea_alloff`, `..._routeb_allon`, `..._routeb_alloff`)
- then rebuild overviews only via `build_present_group_t1only_matrix.py`

Acceptance rule for the maintained 8-way set:

- files: `kong2022_{method}_group_overview_{routea|routeb}_{allon|alloff}.png`
- count: exactly 8
- each file should be byte-distinct (`sha256sum` all different); exact duplicates are treated as workflow regression signals

Partial methods are unsupported for Kong2022 in this route and are warning+skip in the group driver.

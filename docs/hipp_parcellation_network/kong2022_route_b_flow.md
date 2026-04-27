# Kong2022 Route B Flow

## Summary

Route B uses the public atlas label file directly:

- `external/Schaefer2018_Kong2022/Schaefer2018_400Parcels_Kong2022_17Networks_order.dlabel.nii`

No individualized cortex PFM is run in this route. Each subject gets:

- 400-ROI cortex timeseries (after cortex tSNR gate)
- merged canonical-8 cortex timeseries derived from those ROI timeseries (kept for compatibility; not used by the three active assignment methods)

Atlas slug is fixed to `kong2022`.

For T1only, Route B must compute its own cortex tSNR gate and run slicing from the subject dtseries. Do not consume T2only Kong2019/PFM cortex reference outputs.

## Pipeline Steps

1. Build per-subject Kong2022 reference:
   - script: `scripts/cortex/extract_cortex_kong2022_roi_timeseries.py`
   - batch entrypoint: `scripts/cortex/run_cortex_kong2022_reference_batch.py`
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
   - pipeline default methods: `group-intrinsic-spectral`, `group-network-top`, `group-network-all`
5. Render standalone cluster-ROI cortex figures (independent from overview):
   - script: `scripts/hipp_parcellation_network/build_cluster_roi_cortex_independent.py`
   - single output root: `present_group_t1only/runtime/` (cluster figures, manifests, and quicklook index all co-located)
   - merged-hemisphere quicklook overviews are written directly under `present_group_t1only/`
   - clusterless methods (`group-network-top`, `group-network-all`) write merged-smoothing quicklooks: `cortex_<atlas>_<method>_overview.png`
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

- method-agnostic preprocessing before spectral affinity:
  - spatial graph smoothing (`--spatial-smooth-iters`, `--spatial-smooth-alpha`)
  - optional PCA reconstruction (`--feature-denoise-mode pca`, `--feature-pca-var`, `--feature-pca-max-components`)
- candidate-`K` rejection gates:
  - unstable: `group_ari_mean < 0.85` or `run_aware_ari_mean < 0.90`
  - redundant: `max_profile_similarity > 0.90` or `redundant_pair_count > 0`
  - underfit: actionable `K+1` adds distinct profile gain `>= 0.10`
  - nontrivial fail: existing min-size/connectedness/coherence checks fail
- final selection in redundancy-aware mode: choose the smallest `K` that passes all gates.

For `group-intrinsic-spectral`, cluster network assignment uses raw TopN positive ROI majority (`N=min(20, n_positive_roi)` from `mean_z > 0`), not canonical `8 x T` cortex timeseries. One-sided t-test + BH-FDR outputs are still computed for stats fields only. If no positive ROI exists, status is `unassigned` with reason `no_positive_roi`. If TopN ROI belong to only one network, `2nd` is recorded as `none` (not duplicated from `1st`).

For `group-network-top` and `group-network-all`, assignment is vertex-level (clusterless):

- `group-network-top`: raw TopN positive ROI majority (`mean_z > 0`); significance remains stats-only
- `group-network-all`: network-wide ROI mean score assignment with no significance gate (always assign in support mask); significance remains stats-only

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

Top-level `present_group_t1only/` contains direct-access images:

- clusterless: `cortex_<atlas>_<method>_overview.png` merged-hemisphere overview collage
- cluster mode: `cortex_<atlas>_<method>_<smoothing>_overview.png` merged-hemisphere overview collage

Statistical and selection policy:

- one-sided one-sample t-tests (vs 0) on subject-level cluster-to-ROI Fisher-z values
- BH-FDR correction per cluster and per sign (positive/negative), threshold `q < 0.05`
- intrinsic cluster mode: Sig-only selection, capped at top 20 significant ROI per sign
- clusterless mode: global top connections use TopN positive + TopN negative edges (total up to `2N`) with significance thresholding (`q < 0.05`)
- network hue follows canonical8 color mapping; sign is encoded by shade (negative uses lighter tone)
- cluster figure legend is network-color based (no per-ROI name legend)
- default method set for this stage is `group-intrinsic-spectral`, `group-network-top`, `group-network-all`
- rendering uses first scene view only (single 4-panel cortex layout), no multi-view expansion
- overview builder composes from runtime cluster panels and keeps those panel PNGs in runtime
- per-cluster figure includes network legend with sign-coding note and selected ROI counts
- merged overview panels annotate `n=total (+pos/-neg)` for each subplot

## Method Support

Active default methods for Kong2022:

- `group-intrinsic-spectral`
- `group-network-top`
- `group-network-all`

Opt-in only:

- `group-network-spectral`

Partial methods are unsupported for Kong2022 in this route and are warning+skip in the group driver.

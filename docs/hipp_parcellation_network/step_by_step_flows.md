# Hipp Parcellation Step-by-Step Flows

## 1. Build Shared Inputs

1. `prepare` / subject surfaces and valid masks.
2. Kong2022 reference extraction:
   - `run_cortex_kong2022_reference_batch.py`
3. FC materialization:
   - `materialize_kong2022_vertex_to_roi_fc.py`
   - `materialize_intrinsic_vertex_to_vertex_fc.py`

## 2. Run Group Atlas Driver

Run:

```bash
python scripts/hipp_parcellation_network/run_merged_canonical_group_atlas.py \
  --atlases kong2022 \
  --methods group-intrinsic-spectral group-network-top group-network-all
```

Opt-in spectral run (cluster-based pair):

```bash
python scripts/hipp_parcellation_network/run_merged_canonical_group_atlas.py \
  --atlases kong2022 \
  --methods group-network-spectral group-intrinsic-spectral \
  --k-selection-mode redundancy-aware
```

Opt-in experimental ROI-sim spectral run (`kong2022` only):

```bash
# Frozen in this round: do not run.
# See docs/hipp_parcellation_network/group_roi_sim_spectral_frozen_plan.md
```

Canonical urgent T1only spectral rerun:

```bash
python scripts/hipp_parcellation_network/run_t1only_spectral_fast_rerun.py
```

This launcher fixes the default urgent runtime policy to:

- `group-intrinsic-spectral` first, then `group-network-spectral`
- rebuild overview immediately after each method
- parallelize across the four independent combos (`2mm/4mm x L/R`)
- keep each combo's `K` loop sequential with `--redundancy-early-stop`
- default subject FC preprocessing in the launcher is `--subject-fc-top-fraction 0.10 --subject-fc-binarize`
- launcher override knobs are available: `--subject-fc-top-fraction <0..1>` and `--no-subject-fc-binarize`

Spectral FC preprocessing defaults in this driver:

- `--subject-fc-top-fraction 0.10` (keep largest raw-value Top 10% per subject FC map)
- `--subject-fc-binarize` (default on; retained Top entries are set to `1`)
- optional disable: `--no-subject-fc-binarize`

Surface/topology rule for all cluster-based runs:

- Group clustering uses shared group-average template surfaces (template-first mode).
- The driver validates each manifest subject's surface topology against the shared template.
- Any topology mismatch causes hard failure with subject list.

### 2.1 `group-intrinsic-spectral`

1. Build support mask from coverage threshold.
2. Per subject, preprocess intrinsic FC map (`512 x 512`) before any group averaging:
   - keep largest raw-value Top `10%` finite entries, set other finite entries to `0`
   - by default, binarize retained Top entries to `1`
3. Run intrinsic spectral clustering and select `K`.
   - in `redundancy-aware` mode, reject `K` if:
     - redundant (`max_profile_similarity > 0.80` or `redundant_pair_count > 0`)
     - nontriviality checks fail
   - final K selection is applied after removing redundant/nontrivial-fail candidates
   - final K selection (three-tier over the remaining eligible pool):
     1. 1-SE rule: keep survivors within 1 SE of the survivor-pool best mean ARI
     2. local-max preference: among within-1SE survivors, prefer ARI-curve local maxima
     3. parsimony tiebreak: smallest K when quality is genuinely tied
   - if the reduced eligible pool is non-empty, one final `K` is always selected from that pool
   - urgent heuristic mode `--redundancy-early-stop`:
     - evaluate `K` in ascending order
     - stop at the first redundant `K` and skip all larger `K`
     - this is a speed shortcut for interim runs; do not use it for final reporting runs
4. Compute cluster-to-ROI Fisher-z profiles (400 ROI).
5. Per cluster: one-sided t + BH-FDR (`q<0.05`) over ROI.
6. Keep significance outputs for stats (`n_significant_*`, selected significant ROI list).
7. Select assignment TopN from raw positive ROI (`mean_z > 0`), `N=min(20, n_positive_roi)`.
8. Assign cluster network by TopN ROI count majority (raw TopN, not significance-filtered).
9. If no positive ROI exists, mark `unassigned` with reason `no_positive_roi`.
10. If TopN ROI are all in one network, set `2nd` to `none` (do not duplicate `1st`).
11. Write:
   - `group_labels_final.npy`
   - `cluster_assignment_status.json`
   - `group_functional_atlas_labels.npy`

### 2.2 `group-network-top`

1. Skip clustering.
2. Use vertex-to-ROI Fisher-z features directly.
3. Per vertex: one-sided t + BH-FDR (`q<0.05`) over ROI.
4. Keep significance outputs for stats (`n_significant_*`).
5. Select assignment TopN from raw positive ROI (`mean_z > 0`), `N=min(20, n_positive_roi)`.
6. Assign vertex network by TopN ROI count majority (raw TopN, not significance-filtered).
7. If no positive ROI exists, mark `unassigned` with reason `no_positive_roi`.
8. Write:
   - `vertex_assignment_status.json`
   - `group_functional_atlas_labels.npy`
   - `global_top_connections.json`

### 2.3 `group-network-all`

1. Skip clustering.
2. Use vertex-to-ROI Fisher-z features directly.
3. Per vertex: one-sided t + BH-FDR (`q<0.05`) over ROI.
4. Keep significance outputs for stats (`n_significant_*`).
5. Assign by max canonical-network mean score over all ROI in that network (always assign; no significance gate).
6. Write:
   - `vertex_assignment_status.json`
   - `group_functional_atlas_labels.npy`
   - `global_top_connections.json`

### 2.4 `group-network-spectral` (opt-in)

1. Build support mask from coverage threshold.
2. Per subject, preprocess network FC map (`512 x 400`) before any group averaging:
   - keep largest raw-value Top `10%` finite entries, set other finite entries to `0`
   - by default, binarize retained Top entries to `1`
3. Run spectral clustering on vertex-to-ROI Fisher-z features.
4. Use the same optional `redundancy-aware` `K` selection logic as intrinsic spectral.
   - the same optional `--redundancy-early-stop` heuristic applies here
5. For `kong2022`, assign cluster network by the same raw TopN positive ROI majority rule used by intrinsic spectral:
   - one-sided t + BH-FDR outputs are kept for stats only
   - assignment uses `N=min(20, n_positive_roi)` from raw positive ROI (`mean_z > 0`)
   - overview confidence and legend semantics use TopN count mode, not Fisher-z maxima
6. Write:
   - `group_labels_final.npy`
   - `cluster_assignment_status.json`
   - `group_functional_atlas_labels.npy`

### 2.5 `group-roi-sim-spectral` (opt-in, experimental)

1. Frozen in this round (`status=frozen`, `not implemented`).
2. Not part of default runs or urgent rerun bundles.
3. Design is preserved in:
   - `docs/hipp_parcellation_network/group_roi_sim_spectral_frozen_plan.md`

## 3. Build Overviews

1. `build_present_group_overview_merged_canonical.py`
   - intrinsic confidence uses TopN-count `1st/2nd`
   - TopN-count y-axis labels do not show per-row `N`
   - x-axis label uses dynamic TopN-cap (`Top{top_n_cap}` when available, otherwise `TopN`)
   - confidence `1st` bar color follows the `top1_network` canonical color
   - intrinsic heatmap annotation shows numeric magnitude only (no `+/-` sign)
   - `K` panel text now reports per-K reject reasons with tags:
     - `redundant`, `nontrivial_fail`
   - cluster methods now show a K-metric gate panel (replacing `cluster_indicator`):
     - `max_profile_similarity` (threshold `0.80`)
     - selected K (`K*`) vertical dashed markers for L/R
   - clusterless methods render without cluster artifacts
   - for `group-network-top` / `group-network-all`, left pane shows only `vertex assignment counts`
   - clusterless group overview no longer includes left-pane global-top text/metadata blocks
2. `build_cluster_roi_cortex_independent.py`
   - intrinsic: per-cluster cortex figures
   - clusterless: global-top figure uses cortex ROI + hippocampal surface vertex map (not long-axis strip) + right-side legend styled like intrinsic panels
   - this cortex overview remains the location for global-top visualization

## 4. Rule Reminder

For the three active methods in this flow, assignment logic does not consume cortex canonical `8 x T` timeseries arrays.

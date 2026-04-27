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

### 2.1 `group-intrinsic-spectral`

1. Build support mask from coverage threshold.
2. Run intrinsic spectral clustering and select `K`.
3. Compute cluster-to-ROI Fisher-z profiles (400 ROI).
4. Per cluster: one-sided t + BH-FDR (`q<0.05`) over ROI.
5. Keep significance outputs for stats (`n_significant_*`, selected significant ROI list).
6. Select assignment TopN from raw positive ROI (`mean_z > 0`), `N=min(20, n_positive_roi)`.
7. Assign cluster network by TopN ROI count majority (raw TopN, not significance-filtered).
8. If no positive ROI exists, mark `unassigned` with reason `no_positive_roi`.
9. If TopN ROI are all in one network, set `2nd` to `none` (do not duplicate `1st`).
10. Write:
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

## 3. Build Overviews

1. `build_present_group_overview_merged_canonical.py`
   - intrinsic confidence uses TopN-count `1st/2nd`
   - TopN-count y-axis labels do not show per-row `N`
   - x-axis label uses dynamic TopN-cap (`Top{top_n_cap}` when available, otherwise `TopN`)
   - confidence `1st` bar color follows the `top1_network` canonical color
   - intrinsic heatmap annotation shows numeric magnitude only (no `+/-` sign)
   - clusterless methods render without cluster artifacts
   - for `group-network-top` / `group-network-all`, left pane shows only `vertex assignment counts`
   - clusterless group overview no longer includes left-pane global-top text/metadata blocks
2. `build_cluster_roi_cortex_independent.py`
   - intrinsic: per-cluster cortex figures
   - clusterless: global-top figure uses cortex ROI + hippocampal surface vertex map (not long-axis strip) + right-side legend styled like intrinsic panels
   - this cortex overview remains the location for global-top visualization

## 4. Rule Reminder

For the three active methods in this flow, assignment logic does not consume cortex canonical `8 x T` timeseries arrays.

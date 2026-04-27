# Hipp Parcellation Multi-Branch Flow

## Active Branches (Kong2022 Route B)

- `group-intrinsic-spectral` (cluster-based)
- `group-network-top` (clusterless, vertex-level)
- `group-network-all` (clusterless, vertex-level)

`group-network-top` and `group-network-all` are `kong2022`-only in the current implementation.

Optional spectral branch (opt-in):

- `group-network-spectral` (cluster-based)

## Shared Upstream

1. Extract cortex Kong2022 ROI references:
   - `scripts/cortex/run_cortex_kong2022_reference_batch.py`
2. Materialize hippocampal FC caches:
   - ROI FC: `scripts/hipp_parcellation_network/materialize_kong2022_vertex_to_roi_fc.py`
   - intrinsic FC: `scripts/hipp_parcellation_network/materialize_intrinsic_vertex_to_vertex_fc.py`
3. Run group atlas:
   - `scripts/hipp_parcellation_network/run_merged_canonical_group_atlas.py`

## Branch-Specific Logic

### `group-intrinsic-spectral`

- Input feature for clustering: hippocampal intrinsic vertex-to-vertex Fisher-z FC.
- Select `K` by ARI-based constraints.
- Assign cluster network by raw TopN positive ROI majority:
  - one-sided t-test + BH-FDR (`q<0.05`) per cluster across 400 ROI is retained for stats fields only
  - assignment uses `N = min(20, n_positive_roi)` from raw positive ROI (`mean_z > 0`), not significance-filtered ROI
  - confidence rank `1st/2nd` is TopN ROI count ranking
  - `unassigned` is used only when no positive ROI exists (`unassigned_reason = no_positive_roi`)
  - if TopN ROI all belong to one network, `2nd` is `none` (never duplicate `1st`)
  - optional `--k-selection-mode redundancy-aware`:
    - stable gate: `group_ari_mean >= 0.85`, `run_aware_ari_mean >= 0.90`
    - redundant gate: `max_profile_similarity <= 0.90` and `redundant_pair_count == 0`
    - underfit gate: actionable `K+1` with `profile_gain_to_next >= 0.10` rejects current `K`
    - choose smallest `K` passing all gates

### `group-network-spectral` (opt-in)

- Input feature for clustering: hippocampal vertex-to-Kong2022-ROI Fisher-z FC (400 ROI).
- Uses the same optional redundancy-aware `K` selection and preprocessing knobs as intrinsic spectral.

### `group-network-top`

- No clustering (`K` not used).
- Vertex-level network assignment by raw TopN positive ROI majority:
  - one-sided t-test + BH-FDR (`q<0.05`) per vertex across 400 ROI is retained for stats fields only
  - assignment uses `N = min(20, n_positive_roi)` from raw positive ROI (`mean_z > 0`)
  - `unassigned` is used only when no positive ROI exists (`unassigned_reason = no_positive_roi`)

### `group-network-all`

- No clustering (`K` not used).
- Vertex-level network assignment by all-ROI network mean score:
  - per-vertex ROI significance via one-sided t-test + BH-FDR (`q<0.05`) is retained for stats fields only
  - assignment is always by highest canonical-network mean over all ROI in that network (no significance gate)

## Cortex / Hippomaps Figure Rule

- For clusterless branches, no cluster panel is generated.
- Use global top connections from `global_top_connections.json`:
  - TopN positive + TopN negative (max `2N`)
  - this selection remains significance-thresholded (`q<0.05`) and is separate from assignment logic
- Render clusterless global-top panel with:
  - cortex ROI panel
  - hippocampal vertex panel rendered on hippocampal surface maps (not long-axis strip)
  - right-side legend block aligned with intrinsic panel style (sign coding + network counts)

## Group Overview Rule (Clusterless)

- In `build_present_group_overview_merged_canonical.py` for `group-network-top` / `group-network-all`:
  - the left pane contains only `vertex assignment counts`
  - vertex confidence histogram, global-top text preview, and metadata text block are removed
  - global-top visualization remains in cortex overview outputs, not in the group overview text panel

## Overview Confidence Label Rule

- In confidence panels for TopN-count methods:
  - y-axis labels do not show per-row `N`
  - `2nd` network text is suppressed when it is `none`/`unassigned`
  - x-axis uses dynamic TopN-cap label (`Top{top_n_cap}` when available, otherwise `TopN`)
  - `1st` bars are colored by the corresponding `top1_network` canonical color
  - intrinsic heatmap cell annotation uses numeric magnitude only (no `+/-` sign)

## 8xT Timeseries Rule

For the three active branches above, assignment logic must not consume cortex canonical `8 x T` timeseries arrays.

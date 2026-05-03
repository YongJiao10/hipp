# Hipp Parcellation Multi-Branch Flow

## Active Branches (Kong2022 Route B)

- `group-intrinsic-spectral` (cluster-based)
- `group-network-top` (clusterless, vertex-level)
- `group-network-all` (clusterless, vertex-level)

`group-network-top` and `group-network-all` are `kong2022`-only in the current implementation.

Optional spectral branch (opt-in):

- `group-network-spectral` (cluster-based)
- `group-roi-sim-spectral` (design frozen in this round; not implemented/enabled)

## Kong2019 Branch Convention (Fast-PFM T1only)

Kong2019 does not use `routea/routeb` naming.

- branch root: `outputs_experiments/hippunfold_recommended_t1w_20260423/kong2019_fast_pfm_t1only/`
- shared inputs: `.../shared_inputs/pfm_151/`
- atlas run: `kong2019_8net` only
- mode split only: `allon` / `alloff`
- presentation names use `kong2019_*_group_overview_{allon|alloff}.png`

## Shared Upstream

1. Extract cortex Kong2022 ROI references:
   - `scripts/cortex/run_cortex_kong2022_reference_batch.py`
2. Materialize hippocampal FC caches:
   - ROI FC: `scripts/hipp_parcellation_network/materialize_kong2022_vertex_to_roi_fc.py`
   - intrinsic FC: `scripts/hipp_parcellation_network/materialize_intrinsic_vertex_to_vertex_fc.py`
3. Run group atlas:
   - `scripts/hipp_parcellation_network/run_merged_canonical_group_atlas.py`
   - Surface topology and long-axis order are derived from the shared group-average template surfaces.
   - Manifest subjects are validated for topology compatibility against that shared template.

Kong2019 fast-PFM T1only input relocation:

1. move (`mv`) archive shared tree:
   - `_archive/outputs_migration/hipp_group_inputs_shared`
   - `-> outputs_experiments/hippunfold_recommended_t1w_20260423/kong2019_fast_pfm_t1only/shared_inputs/pfm_151`
2. keep moved `sub-*/fc/kong2022/*` for redundancy-aware `K` selection support
3. regenerate `kong2019_8net` in the new root via:
   - `scripts/hipp_parcellation_network/materialize_kong7_kong8_pearson_partial_fc.py --atlases kong2019_8net`

## Branch-Specific Logic

### `group-intrinsic-spectral`

- Per-subject FC preprocessing is applied before group averaging (default):
  - keep largest raw-value Top `10%` entries in each subject `512 x 512` FC map, set other finite entries to `0`
  - default binarization sets retained Top entries to `1` (`--no-subject-fc-binarize` disables this)
  - fraction is configurable via `--subject-fc-top-fraction`
- Input feature for clustering: hippocampal intrinsic vertex-to-vertex Fisher-z FC.
- Select `K` by ARI-based constraints.
- Assign cluster network by raw TopN positive ROI majority:
  - one-sided t-test + BH-FDR (`q<0.05`) per cluster across 400 ROI is retained for stats fields only
  - assignment uses `N = min(20, n_positive_roi)` from raw positive ROI (`mean_z > 0`), not significance-filtered ROI
  - confidence rank `1st/2nd` is TopN ROI count ranking
  - `unassigned` is used only when no positive ROI exists (`unassigned_reason = no_positive_roi`)
  - if TopN ROI all belong to one network, `2nd` is `none` (never duplicate `1st`)
  - optional `--k-selection-mode redundancy-aware`:
    - first remove redundant `K` candidates (`max_profile_similarity <= 0.80` and `redundant_pair_count == 0`)
    - then run final K selection on the remaining nonredundant, nontrivial pool:
      1. 1-SE rule: keep survivors within 1 SE of the survivor-pool best mean ARI
      2. local-max preference: among within-1SE survivors, prefer ARI-curve local maxima
      3. parsimony tiebreak: smallest K wins when quality is genuinely tied
    - if any nonredundant, nontrivial candidate remains, the pipeline selects one final `K` from that reduced pool (it does not reject lower `K` in favor of later redundant candidates)
    - optional urgent speed heuristic: `--redundancy-early-stop`
      - evaluate `K` in ascending order
      - once the first redundant `K` is observed, skip all larger `K`
      - this disables within-combo `K` parallelism and is intended only for interim fast runs, not final paper results

### `group-network-spectral` (opt-in)

- Per-subject FC preprocessing is applied before group averaging (default):
  - keep largest raw-value Top `10%` entries in each subject `512 x 400` FC map, set other finite entries to `0`
  - default binarization sets retained Top entries to `1` (`--no-subject-fc-binarize` disables this)
  - fraction is configurable via `--subject-fc-top-fraction`
- Input feature for clustering: hippocampal vertex-to-Kong2022-ROI Fisher-z FC (400 ROI).
- For `kong2022`, cluster network assignment now follows the same TopN-count rule as intrinsic spectral:
  - one-sided t-test + BH-FDR (`q<0.05`) over ROI is retained for stats fields only
  - assignment uses raw TopN positive ROI majority (`N = min(20, n_positive_roi)`)
  - overview confidence labels, ambiguity text, and network naming use TopN count fields rather than canonical-network Fisher-z maxima
- Uses the same optional redundancy-aware `K` selection and preprocessing knobs as intrinsic spectral.

Kong2019 implementation note:

- for `atlas=kong2019_8net` with `--k-selection-mode redundancy-aware`, the driver currently reads `fc/kong2022/*` as ROI-FC bank for redundancy/profile checks
- therefore `fc/kong2022/*` must exist in the Kong2019 shared root

### `group-roi-sim-spectral` (opt-in, experimental)

- Frozen plan only in this round (`status=frozen`, `not implemented`).
- Refer to `docs/hipp_parcellation_network/group_roi_sim_spectral_frozen_plan.md`.
- Do not include this branch in default or urgent rerun method sets.

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

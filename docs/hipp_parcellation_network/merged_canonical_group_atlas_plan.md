# Merged-Canonical Group Atlas Reference

## Summary

This document is the active reference for hippocampal **group-level** analysis in this repository.

The workflow builds merged-canonical hippocampal group atlases for the active T1only 151-subject QC cohort and stops at the group-atlas stage. Individual inference and learned group priors are out of scope.

## Active Methods

- `group-intrinsic-spectral`
- `group-network-top`
- `group-network-all`

Opt-in only (not in default run):

- `group-network-spectral` (legacy spectral baseline; not in the default method set)

`group-network-top` and `group-network-all` are currently implemented for `kong2022` only.

Retired (no longer run; code retained for reference):

- ~~`group-network-partial-spectral`~~ — one-sided partial correlation features; retired, not running
- ~~`group-intrinsic-partial-spectral`~~ — intrinsic + partial; retired, not running

Active atlases:

- `kong2019`
- `kong2019_8net`
- `kong2022`

## Required Inputs

Canonical shared input root:

- `outputs_migration/hipp_group_inputs_shared/`

Required input classes:

- canonical-network FC caches (raw Pearson r in [-1, 1] on disk; unconditionally Fisher-z transformed at load)
- intrinsic FC caches (raw Pearson r, same treatment)
- per-subject valid masks
- shared run-boundary metadata (`sub-*/reference/common/runs/run-*/reference_summary.json`)
- canonical network order metadata

Note: one-sided partial-FC caches (`hipp_vertex_to_network_partial_fc*.npy`) remain on disk but are no longer consumed by active methods.

Default subject manifests:

- `outputs_experiments/hippunfold_recommended_t1w_20260423/kong2022_route_b/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_151.txt`
- `outputs_experiments/hippunfold_recommended_t1w_20260423/kong2022_route_b/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt`

## Canonical Network Merge

The active group analysis still depends on atlas-specific parent-network merges into a shared canonical network axis. For the active Kong atlases, the preserved merge rule is:

```text
Kong2019
  DefaultA + DefaultB + DefaultC        -> Default
  VisualA + VisualB + VisualC           -> Visual
  SomatomotorA + SomatomotorB           -> Somatomotor
  DorsalAttentionA + DorsalAttentionB   -> DorsalAttention
  VentralAttentionA + VentralAttentionB -> VentralAttention
  ControlA + ControlB + ControlC        -> Control
  Auditory                              -> Auditory
  TemporalParietal                      -> Default

kong2019_8net
  same base merge map, except `TemporalParietal` remains independent in the 8-network atlas

kong2022
  DefaultA + DefaultB + DefaultC        -> Default
  ContA + ContB + ContC                 -> Control
  SalVenAttnA + SalVenAttnB             -> VentralAttention
  DorsAttnA + DorsAttnB                 -> DorsalAttention
  SomMotA + SomMotB                     -> Somatomotor
  VisualA + VisualB + VisualC           -> Visual
  Aud                                   -> Auditory
  Language                              -> Language
```

The upstream merge file also defines other atlas mappings, but the active merged-canonical group workflow consumes the Kong2019 / kong2019_8net / kong2022 set.

## Core Rules

- analyze left and right hippocampus separately
- keep `N_ref` distinct from hippocampal cluster count `K`
- choose `K` data-driven on the group workflow, not from cortical network count
- `group-network-top` and `group-network-all` are clusterless (no `K` selection / no cluster labels)
- order final cluster IDs along hippocampal long axis (`head -> tail`)
- record exact subject list, coverage threshold, and non-triviality settings in manifests
- if a combo has no valid `K` (or no `K` survives constraints), write a `group_atlas_manifest.json` with `status=skipped_*` and continue other combos
- use direct ARI-space reporting for K-selection summaries and null references
- FC cache construction uses sample-standardized Pearson (`np.nanstd(..., ddof=1)` with `T-1` denominator)
- FC features are **unconditionally** Fisher-z transformed at load (raw r ∈ [-1, 1] on disk); no conditional fallback
- for the three active methods (`group-intrinsic-spectral`, `group-network-top`, `group-network-all`), assignment logic must not consume cortex `8 x T` canonical timeseries arrays
- cosine similarity is mapped to affinity via `max(cosine, 0)` — anti-correlations contribute zero weight, not a positive attraction
- after Hadamard fusion with mesh adjacency, `W` has `diag=0` (no self-loops); symmetrized via `W.maximum(W.T)` for float safety

## Output Contract

Canonical runtime output root:

- `outputs_migration/hipp_group_merged_canonical_atlas/`

Each run emits group-atlas products including:

- `k_selection_summary.tsv/json`
- `group_atlas_manifest.json`
- `group_functional_atlas_labels.npy`
- `group_functional_atlas_label_map.json`

Cluster-based (`group-intrinsic-spectral`) adds:

- `group_labels_final.npy`
- `cluster_network_profile.npy`
- `cluster_assignment_status.json` (TopN-count confidence keys: `top1_count` / `top2_count`)
- `cluster_roi_profile.npy` (`K_final x 400` Fisher-z mean)
- `cluster_roi_top20.json` (positive/negative Top20 ROI per cluster)

Clusterless (`group-network-top`, `group-network-all`) adds:

- `vertex_assignment_status.json`
- `vertex_roi_profile.npy` (+ `vertex_support_indices.npy`)
- `global_top_connections.json` (global TopN positive + TopN negative edges, total up to `2N`)
- `cluster_roi_registry.tsv` (kong2022 only; ROI metadata dictionary for index mapping)

For skipped combos (`status` starts with `skipped_`), clustering/final-label products are absent by design; only support masks, `k_selection_summary.tsv/json`, and skip metadata are emitted.

Presentation companion:

- `present_group/`
- built by `build_present_group_overview_merged_canonical.py`
- public figure set is restricted to the six active atlas+method overview PNGs

## Route A (Kong2022_ArealMSHBM) - Minimal Loop Implemented

Route A minimal loop is now implemented against official `Kong2022_ArealMSHBM` (not Kong2019 compatibility wrappers):

- subject driver: `scripts/cortex/run_cortex_mshbm_subject.py`
- MATLAB wrapper: `scripts/cortex/matlab/run_kong2022_mshbm_subject.m`
- batch smoke runner: `scripts/cortex/run_cortex_kong2022_mshbm_batch.py`
- subject figure renderer: `scripts/cortex/plot_cortex_kong2022_mshbm.py`

Current implemented scope:

- fs_LR_32k ArealMSHBM individualized parcellation (default `gMSHBM`, 400 ROIs)
- subject-specific `label.gii` and `dlabel.nii`
- per-subject overview PNG
- strict fail-fast preflight for MATLAB/CBIG dependency gaps
- outputs rooted at `outputs_experiments/cortex_kong2022_areal_mshbm/`

Still pending for full Route A integration:

- downstream `kong2022_areal_mshbm` reference extraction/FC/group-atlas wiring
- cohort-scale production runs beyond two-subject smoke

## Archive Boundary

The following are retired and archived under `_archive/hipp_parcellation_network/`:

- subject-level clustering pipeline (including `scripts/common/spectral_clustering.py`, which has no live caller)
- `group prior + fast-PFM`
- legacy flow docs
- MSHBM experiments
- `group-network-partial-spectral` and `group-intrinsic-partial-spectral` branches (code remains in `run_merged_canonical_group_atlas.py` but no runs are planned)

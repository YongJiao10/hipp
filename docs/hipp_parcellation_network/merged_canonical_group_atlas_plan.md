# Merged-Canonical Group Atlas Reference

## Summary

This document is the active reference for hippocampal **group-level** analysis in this repository.

The workflow builds merged-canonical hippocampal group atlases for the default 153-subject QC cohort and stops at the group-atlas stage. Individual inference and learned group priors are out of scope.

## Active Methods

- `group-network-spectral`
- `group-intrinsic-spectral`

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

Default subject manifest:

- `manifests/hcp_7t_hippocampus_fmri_qc_pass_153.txt`

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
- order final cluster IDs along hippocampal long axis (`head -> tail`)
- record exact subject list, coverage threshold, and non-triviality settings in manifests
- use direct ARI-space reporting for K-selection summaries and null references
- FC cache construction uses sample-standardized Pearson (`np.nanstd(..., ddof=1)` with `T-1` denominator)
- FC features are **unconditionally** Fisher-z transformed at load (raw r ∈ [-1, 1] on disk); no conditional fallback
- cosine similarity is mapped to affinity via `max(cosine, 0)` — anti-correlations contribute zero weight, not a positive attraction
- after Hadamard fusion with mesh adjacency, `W` has `diag=0` (no self-loops); symmetrized via `W.maximum(W.T)` for float safety

## Output Contract

Canonical runtime output root:

- `outputs_migration/hipp_group_merged_canonical_atlas/`

Each run emits group-atlas products including:

- `k_selection_summary.tsv/json`
- `group_labels_final.npy`
- `cluster_network_profile.npy`
- `cluster_assignment_status.json`
- `group_atlas_manifest.json`
- `cluster_roi_profile.npy` (kong2022 only; `K_final x 400` Fisher-z mean)
- `cluster_roi_top20.json` (kong2022 only; positive/negative Top20 ROI per cluster)
- `cluster_roi_registry.tsv` (kong2022 only; ROI metadata dictionary for index mapping)

Presentation companion:

- `present_group/`
- built by `build_present_group_overview_merged_canonical.py`
- public figure set is restricted to the six active atlas+method overview PNGs

## Route A (Kong2022 individualized PFM) - TODO

Route A is intentionally stubbed in this change set and is not runnable yet.

- asset pointer: `external/Schaefer2018_Kong2022/Params_Final.mat`
- code placeholder: `scripts/cortex/run_cortex_pfm_subject_kong2022.py`
- planned delta vs Kong2019 Route A:
  - prior source is `Params_Final.mat` (Kong2022)
  - individualized target is 400 ROI labels (not 17 parent networks)
  - expected outputs would be Kong2022 individualized label files plus a companion extractor
- blockers:
  - MATLAB/CBIG individualized pipeline is not wired in this repository
  - per-subject label-id consistency policy for 400 ROI outputs is still open
  - individualized Kong2022 ROI extractor implementation is pending

## Archive Boundary

The following are retired and archived under `_archive/hipp_parcellation_network/`:

- subject-level clustering pipeline (including `scripts/common/spectral_clustering.py`, which has no live caller)
- `group prior + fast-PFM`
- legacy flow docs
- MSHBM experiments
- `group-network-partial-spectral` and `group-intrinsic-partial-spectral` branches (code remains in `run_merged_canonical_group_atlas.py` but no runs are planned)

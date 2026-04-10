# Hippocampal Functional Parcellation: Network-First Multi-Branch Flow

## Purpose

This document defines the network-first variant implemented in this worktree.

It is intentionally separate from the parcel-informed line in the main workspace. In this worktree, the defining feature basis is:

```text
cortex canonical merged network timeseries
  -> direct hippocampal vertex-to-network FC
```

For step-by-step per-branch procedures, see [step_by_step_flows.md](/Users/jy/Documents/HippoMaps-network-first/docs/experiments/hipp_functional_parcellation/step_by_step_flows.md).
For execution notes, see [network_first_runbook.md](/Users/jy/Documents/HippoMaps-network-first/docs/experiments/hipp_functional_parcellation/network_first_runbook.md).
For the clean 166-subject HPC transfer package, see [network_first_hpc_bundle_handoff.md](/Users/jy/Documents/HippoMaps-network-first/docs/experiments/hipp_functional_parcellation/network_first_hpc_bundle_handoff.md).

The current network-first comparison matrix is:

```text
branches   network-gradient / network-prob-cluster / network-prob-cluster-nonneg / network-prob-soft / network-prob-soft-nonneg / network-wta
atlases    lynch2024 / hermosillo2024 / kong2019
subjects   100610 / 102311 / 102816
smoothing  2mm / 4mm
```

The dedicated HPC bundle intentionally narrows this matrix to:

```text
branches   network-gradient / network-prob-cluster-nonneg
atlases    lynch2024 / kong2019
subjects   hcp_7t_hippocampus_struct_complete_166
```

## Core Rules

1. Left and right hippocampi are modeled independently.
2. Every branch starts from direct `vertex-to-network FC` computed against cortex canonical merged network timeseries.
3. `network-gradient`, `network-prob-cluster`, `network-prob-cluster-nonneg`, `network-prob-soft`, and `network-prob-soft-nonneg` perform hippocampal clustering with final `K` chosen independently for each hemisphere from `2..10` using run-aware instability.
4. `network-wta` does not perform clustering; it directly outputs one label per predefined cortical network.
5. Smoothing is compared inside each overview; it is not promoted to a separate top-level comparison dimension.
6. Every `branch x atlas x subject` produces one overview image copied to `present_network/`.
7. Density semantics are strict: all consumed hippocampal assets must be `den-<density>` files matching the run argument.
8. Legacy assets without `den-` are invalid for analysis in this worktree and must trigger explicit failure.
9. Cross-directory fallback is disallowed: only `<hippunfold-dir>/sub-<id>/surf` is a valid structural surface source.

## Shared Upstream

All branches share the same upstream preprocessing:

1. Extract individualized cortical ROI-component timeseries from the chosen cortical atlas output.
2. Merge atlas-specific parent networks to canonical cross-atlas labels using [cross_atlas_network_merge.json](/Users/jy/Documents/HippoMaps-network-first/config/cross_atlas_network_merge.json).
3. Exclude `Noise`.
4. Resolve run-wise inputs for run-aware instability:
   if `run-1..4` `dtseries` and `bold` files are present, use them directly; otherwise split `run-concat` inputs into four equal runs before staging downstream artifacts.
5. Average ROI-component timeseries within each retained canonical network to create cortex canonical network timeseries.
6. Sample hippocampal resting-state timeseries on the left and right `corobl` surfaces.
7. Compute direct `vertex-to-network FC` for each smoothing condition.

In notation:

```text
cortex ROI components
  -> canonical network merge
  -> Noise exclusion
  -> cortex canonical network timeseries
  -> hippocampal surface timeseries
  -> direct vertex-to-network FC
  -> branch-specific network-first features
```

## Per-Atlas Merge Summary

Count change:

```text
Atlas            Raw Atlas Labels  Noise Excluded  Canonical Networks Retained
Kong2019                      17             no                            7
Hermosillo2024               14             no                            8
Lynch2024                    21            yes                            8
```

Canonical merged labels actually used by each atlas:

```text
Atlas            Canonical Labels Kept After Merge
Kong2019         Default / Visual / Somatomotor / DorsalAttention / VentralAttention / Control / Auditory
Hermosillo2024   Default / Visual / Somatomotor / DorsalAttention / VentralAttention / Control / Auditory / Limbic
Lynch2024        Default / Visual / Somatomotor / DorsalAttention / VentralAttention / Control / Auditory / Language
```

Shared cross-atlas functional networks:

```text
Default / Visual / Somatomotor / DorsalAttention / VentralAttention / Control / Auditory
```

Atlas-specific eighth network:

```text
Kong2019         none (merged into Default)
Hermosillo2024   Limbic
Lynch2024        Language
```

Intuitive per-atlas merge map:

```text
Kong2019
  DefaultA + DefaultB + DefaultC                      -> Default
  VisualA + VisualB + VisualC                         -> Visual
  SomatomotorA + SomatomotorB                         -> Somatomotor
  DorsalAttentionA + DorsalAttentionB                 -> DorsalAttention
  VentralAttentionA + VentralAttentionB               -> VentralAttention
  ControlA + ControlB + ControlC                      -> Control
  Auditory                                            -> Auditory
  TemporalParietal                                    -> Default

Hermosillo2024
  DMN + PMN + PON                                     -> Default
  Vis                                                 -> Visual
  SMd + SMl                                           -> Somatomotor
  DAN                                                 -> DorsalAttention
  VAN + Sal                                           -> VentralAttention
  FP + CO                                             -> Control
  Aud                                                 -> Auditory
  MTL + Tpole                                         -> Limbic

Lynch2024
  Default_Parietal + Default_Anterolateral
  + Default_Dorsolateral + Default_Retrosplenial
  + MedialParietal                                    -> Default
  Visual_Lateral + Visual_Dorsal/VentralStream
  + Visual_V1 + Visual_V5                             -> Visual
  Somatomotor_Hand + Somatomotor_Face
  + Somatomotor_Foot + SomatoCognitiveAction          -> Somatomotor
  DorsalAttention + Premotor/DorsalAttentionII        -> DorsalAttention
  Salience                                            -> VentralAttention
  Frontoparietal + CinguloOpercular/Action-mode       -> Control
  Auditory                                            -> Auditory
  Language                                            -> Language
  Noise                                               -> excluded
```

## Branch Definitions

### `network-gradient`

This branch preserves the gradient logic, but the embedding is computed on direct `vertex-to-network FC`.

For each hemisphere:

1. Treat each vertex as a network-FC feature vector.
2. Build a sparse vertex-by-vertex affinity graph from network profiles.
3. Run diffusion-map embedding.
4. Use the first `3` gradients as clustering features.
5. Run spatially constrained Ward clustering for `K=2..10`.
6. Select the smallest stable `K`.
7. Annotate final clusters by their dominant canonical network.

Interpretation:

- cortical feature granularity = canonical merged `network`
- hippocampal output type = clustered subregions
- network labels = post hoc annotations of those subregions

### `network-prob-cluster`

This branch is the direct network-probability clustering route.

For each hemisphere:

1. Convert direct `vertex-to-network FC` into probability vectors using:
   `Fisher z -> shift positive -> row normalize to sum=1`
2. Use those network-probability vectors as clustering features.
3. Run spatially constrained Ward clustering for `K=2..10`.
4. Select the smallest stable `K`.
5. Summarize each final cluster by its mean soft network probabilities.

Interpretation:

- cortical feature granularity = canonical merged `network`
- hippocampal output type = clustered subregions
- cluster summaries = mean soft network profiles

### `network-prob-cluster-nonneg`

This branch matches `network-prob-cluster`, except negative Fisher-z FC values are clipped to zero before row normalization.

For each hemisphere:

1. Convert direct `vertex-to-network FC` into probability vectors using:
   `Fisher z -> clip negative values to 0 -> row normalize to sum=1`
2. Use those network-probability vectors as clustering features.
3. Run spatially constrained Ward clustering for `K=2..10`.
4. Select the smallest stable `K`.
5. Summarize each final cluster by its mean soft network probabilities.

Interpretation:

- cortical feature granularity = canonical merged `network`
- hippocampal output type = clustered subregions
- cluster summaries = mean soft network profiles

### `network-prob-soft`

This branch is the strict soft-first network route.

For each hemisphere:

1. Convert direct `vertex-to-network FC` into network probability vectors.
2. Regularize those probabilities on the hippocampal surface using mesh adjacency plus a long-axis smoothing term.
3. Cluster vertices by similarity of the regularized probability profiles for `K=2..10`.
4. Select the smallest stable `K`.
5. Save the regularized soft probabilities as the main scientific result.
6. Derive optional regularized argmax labels only for auxiliary inspection.

Interpretation:

- cortical feature granularity = canonical merged `network`
- main scientific output type = regularized soft network probabilities
- hippocampal parcellation output type = clustered subregions from those soft profiles
- auxiliary output type = regularized argmax network labels

### `network-prob-soft-nonneg`

This branch matches `network-prob-soft`, except negative Fisher-z FC values are clipped to zero before probability normalization.

For each hemisphere:

1. Convert direct `vertex-to-network FC` into network probability vectors using:
   `Fisher z -> clip negative values to 0 -> row normalize to sum=1`
2. Regularize those probabilities on the hippocampal surface using mesh adjacency plus a long-axis smoothing term.
3. Cluster vertices by similarity of the regularized probability profiles for `K=2..10`.
4. Select the smallest stable `K`.
5. Save the regularized soft probabilities as the main scientific result.
6. Derive optional regularized argmax labels only for auxiliary inspection.

Interpretation:

- cortical feature granularity = canonical merged `network`
- main scientific output type = regularized soft network probabilities
- hippocampal parcellation output type = clustered subregions from those soft profiles
- auxiliary output type = regularized argmax network labels

### `network-wta`

This branch is the pure hard-assignment network route.

For each hemisphere:

1. Start from direct `vertex-to-network FC`.
2. For each hippocampal vertex, find the network with the highest FC score.
3. Assign that vertex to the winning network.
4. Save confidence as `max_score - second_max_score`.
5. Skip clustering and `K` selection.

Interpretation:

- cortical feature granularity = canonical merged `network`
- hippocampal output type = direct network labels
- this is the only branch that does not create new hippocampal subregions

## K Evaluation

The final comparison uses `K=2..10` for `network-gradient`, `network-prob-cluster`, `network-prob-cluster-nonneg`, `network-prob-soft`, and `network-prob-soft-nonneg`.

Each candidate `K` records:

- run-pair instability `I_mean = 1 - mean(ARI)`
- instability standard error
- run-pair mean `ARI`
- homogeneity
- silhouette
- Calinski-Harabasz
- Davies-Bouldin
- WCSS
- delta-WCSS
- minimum cluster-size fraction
- BSS/TSS
- connected-component count

Selection rule:

1. Compute run-aware instability from independent run pairs, with `ARI` as the chance-corrected agreement score.
   Run-wise inputs come either from explicit `run-1..4` files or from an equal four-way split of `run-concat` inputs staged by the workflow.
2. Restrict candidates to local minima of `I_mean(K)`.
3. Apply the `1-SE` rule relative to the best instability point.
4. Among surviving candidates, choose the smallest `K` that also passes the pre-registered `V_min` vertex-count threshold and single-component connectivity constraints.

`network-wta` is exempt because it has no clustering stage.

## Outputs

Active outputs are written to:

```text
outputs/hipp_functional_parcellation_network/<branch>/<atlas>/sub-<subject>/
present_network/sub-<subject>_<atlas>_<branch>_overview.png
```

Each result directory keeps:

```text
hipp_functional_parcellation_network_overview.png
k_selection_curves.png                      (absent for network-wta)
network_probability_heatmaps.png
final_selection_core.json
final_selection_summary.json
summary_manifest.json
```

Structural render input rule:

- network-first structural panels now read HippUnfold structural subfields from the subject-level hippocampal CIFTI `dlabel.nii`
- the workflow then separates that `dlabel.nii` into left/right hippocampal `label.gii` render inputs inside the run output tree before composing the locked overview

Overview heatmap semantics:

- the probability/score heatmap x-axis shows all retained merged networks for that atlas in canonical order
- the left and right hemisphere heatmaps are displayed as separate panels with widened spacing for readability

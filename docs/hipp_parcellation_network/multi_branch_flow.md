# Hippocampal Functional Parcellation: Network-First Multi-Branch Flow

## Purpose

This document defines the network-first variant implemented in this worktree.

It is intentionally separate from the parcel-informed line in the main workspace. In this worktree, the defining feature bases are:

```text
cortex canonical merged network timeseries
  -> direct hippocampal vertex-to-network FC
raw hippocampal vertex timeseries
  -> direct hippocampal vertex-to-vertex FC
```

For step-by-step per-branch procedures, see [step_by_step_flows.md](/Users/jy/Documents/HippoMaps-network-first/docs/hipp_parcellation_network/step_by_step_flows.md).
For execution notes, see [network_first_runbook.md](/Users/jy/Documents/HippoMaps-network-first/docs/hipp_parcellation_network/network_first_runbook.md).
For the clean 166-subject HPC transfer package, see [network_first_hpc_bundle_handoff.md](/Users/jy/Documents/HippoMaps-network-first/docs/hipp_parcellation_network/network_first_hpc_bundle_handoff.md).

The current network-first comparison matrix is:

```text
branches   network-gradient / network-prob-cluster / network-prob-cluster-nonneg / network-prob-soft / network-prob-soft-nonneg / network-wta / network-spectral / network-spectral-nonneg / intrinsic-spectral / intrinsic-spectral-nonneg
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
2. Every branch starts from shared direct feature stores computed against atlas-specific cortex canonical merged network timeseries: `vertex-to-network FC` for network branches and `vertex-to-vertex FC` for intrinsic branches.
3. `network-gradient`, `network-prob-cluster`, `network-prob-cluster-nonneg`, `network-prob-soft`, `network-prob-soft-nonneg`, `network-spectral`, `network-spectral-nonneg`, `intrinsic-spectral`, and `intrinsic-spectral-nonneg` perform hippocampal clustering with final `K` chosen independently for each hemisphere from `2..10` using run-aware instability.
4. `network-wta` does not perform clustering; it directly outputs one label per predefined cortical network.
5. Smoothing is compared inside each overview; it is not promoted to a separate top-level comparison dimension.
6. Every `branch x atlas x subject` produces one overview image copied to `present_network/`.
7. Density semantics are strict: all consumed hippocampal assets must be `den-<density>` files matching the run argument.
8. Legacy assets without `den-` are invalid for analysis in this worktree and must trigger explicit failure.
9. Cross-directory fallback is disallowed: only `<hippunfold-dir>/sub-<id>/surf` is a valid structural surface source.

## K Selection Mode Split (New vs Old)

For canonical batch runs, use fixed launchers:

- `scripts/hipp_parcellation_network/run_mainline.sh`
- `scripts/hipp_parcellation_network/run_experimental.sh`

Do not hand-write `--k-selection-mode`, `--run-split-mode`, `--out-root`, or `--present-dir` in canonical run commands.

- `mainline` (default, current production rule):
  - Select smallest `K` with `null_corrected_score >= best - 0.02` and `min_cluster_size_fraction >= 0.05`.
  - No local-minimum / non-triviality gate.
- `experimental` (future-testing rule):
  - `local-minimum + 1-SE + non-triviality` constraints.
  - Includes `V_min` and connectivity checks.

Reproducibility requirement:
- Every run record and summary must state `k_selection_mode`.

## Shared Upstream

All branches share the same upstream preprocessing:

1. Compute cortex `tSNR = 10000 / std(t)` on left and right cortical grayordinates from the pre-downstream `dtseries`, then hard-mask all cortex grayordinates with `tSNR < 25`.
2. Extract individualized cortical ROI-component timeseries from the chosen cortical atlas output using only the remaining high-tSNR grayordinates.
3. Merge atlas-specific parent networks to canonical cross-atlas labels using [cross_atlas_network_merge.json](/Users/jy/Documents/HippoMaps-network-first/config/cross_atlas_network_merge.json).
4. Exclude `Noise`.
5. Resolve run-wise inputs for run-aware instability:
   use explicit `run-1..4` `dtseries` files when present to derive run lengths; otherwise infer equal four-way run boundaries from `run-concat.dtseries` length.
   The workflow does not materialize run-level `dtseries` files.
   Run-wise hippocampal inputs are then derived only by splitting shared concat hippocampal surface timeseries with those run boundaries (no raw volume BOLD run-splitting).
6. Average ROI-component timeseries within each retained canonical network to create cortex canonical network timeseries.
7. Generate the hippocampal raw surface timeseries inside the shared pipeline store by sampling `run-concat_bold` onto the left and right `corobl` surfaces with `trilinear` mapping and `smooth_iters = 0`, then treat the resulting `.func.gii` files as the only valid raw source.
8. Compute hippocampal `tSNR = 10000 / std(t)` on those raw unsmoothed shared-pipeline `.func.gii` timeseries and hard-mask vertices with `tSNR < 25`.
9. Apply all Workbench `2mm / 4mm` smoothing only inside the high-tSNR hippocampal ROI and only after the tSNR gate; masked hippocampal vertices never participate in smoothing.
10. Compute direct `vertex-to-network FC` and `vertex-to-vertex FC` for each smoothing condition and hemisphere on high-tSNR vertices only, and store those matrices in the shared upstream store at the `subject x atlas x smoothing x hemisphere` level.

In notation:

```text
raw cortex dtseries
  -> cortex tSNR gate (threshold 25)
cortex ROI components
  -> canonical network merge
  -> Noise exclusion
  -> cortex canonical network timeseries
  -> raw hippocampal surface timeseries
  -> hippocampal tSNR gate (threshold 25)
  -> ROI-restricted 2mm / 4mm smoothing
  -> shared direct vertex-to-network / vertex-to-vertex FC
  -> branch-specific network-first features
```

Hard rules for this upstream:

- Cortical `tSNR < 25` grayordinates do not participate in any ROI average, parent-network average, or canonical-network average.
- Hippocampal raw input is strict: the analysis reads only the shared-pipeline raw `.func.gii` files generated from `run-concat_bold` with `trilinear` mapping and `smooth_iters = 0`.
- No hippocampal fallback source is allowed: no archived directory reuse, no `.npy` substitution, and no pre-smoothed raw replacement.
- Direct hippocampal `vertex-to-network FC` and `vertex-to-vertex FC` are shared upstream by `subject x atlas x smoothing x hemisphere`; branch routes must read those shared FC artifacts rather than recomputing them.
- Hippocampal `tSNR < 25` vertices do not participate in smoothing, FC estimation, clustering, or final label assignment.

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

1. Start from direct `vertex-to-network FC` cached as Pearson correlation.
2. Apply Fisher z-transform to each vertex network profile.
3. Treat each vertex as a Fisher-z network-FC feature vector.
4. Build a sparse vertex-by-vertex affinity graph from those transformed network profiles.
5. Run diffusion-map embedding.
6. Use the first `3` gradients as clustering features.
7. Run spatially constrained Ward clustering for `K=2..10`.
8. Select the smallest stable `K`.
9. Annotate final clusters by their dominant canonical network.

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

1. Start from direct `vertex-to-network FC` cached as Pearson correlation.
2. Apply Fisher z-transform to the network FC profile of each vertex.
3. For each hippocampal vertex, find the network with the highest Fisher-z FC score.
4. Assign that vertex to the winning network.
5. Save confidence as `max_score - second_max_score` in Fisher-z units.
6. Skip clustering and `K` selection.

Interpretation:

- cortical feature granularity = canonical merged `network`
- hippocampal output type = direct network labels
- this is the only branch that does not create new hippocampal subregions

### `network-spectral`

This branch implements spatially constrained spectral clustering (`scripts/common/spectral_clustering.py`).

For each hemisphere:

1. Start from direct `vertex-to-network FC` (Pearson correlation, shape `N x N_network`, where `N_network` is atlas-specific after canonical merging).
2. Apply Fisher z-transform to the network FC rows.
3. Build a functional affinity matrix from pairwise cosine similarity of Fisher-z FC rows, mapped to `[0, 1]`.
4. Build a binary spatial adjacency matrix from the HippUnfold hippocampal surface mesh triangles.
5. Fuse the two graphs by Hadamard (element-wise) product, retaining only spatially adjacent vertex pairs weighted by their functional similarity.
6. Run spectral clustering on the fused graph with a precomputed affinity matrix.
7. Run for `K=2..10`, select the smallest stable `K` via run-aware instability.
8. For each cluster, average hippocampal vertex timeseries across all vertices assigned to that cluster.
9. Compute `cluster-mean-timeseries -> canonical-network-timeseries` Pearson FC, apply Fisher z-transform, and annotate final clusters by dominant canonical network.

Interpretation:

- cortical feature granularity = canonical merged `network`
- hippocampal output type = clustered subregions
- spatial constraint = HippUnfold surface mesh adjacency
- clustering algorithm = spectral + KMeans (contrast with Ward used by other clustering branches)

### `network-spectral-nonneg`

This branch matches `network-spectral`, except negative Fisher-z `vertex-to-network FC` values are clipped to `0` before the standard spectral feature preprocessing.

For each hemisphere:

1. Start from direct `vertex-to-network FC` (Pearson correlation, shape `N x N_network`, where `N_network` is atlas-specific after canonical merging).
2. Apply Fisher z-transform to the network FC rows.
3. Clip negative transformed FC values to `0`.
4. Z-score those nonnegative Fisher-z features across vertices, matching the standard spectral branch thereafter.
5. Build a functional affinity matrix from pairwise cosine similarity of the clipped FC rows, mapped to `[0, 1]`.
6. Build a binary spatial adjacency matrix from the HippUnfold hippocampal surface mesh triangles.
7. Fuse the two graphs by Hadamard (element-wise) product, retaining only spatially adjacent vertex pairs weighted by their functional similarity.
8. Run spectral clustering on the fused graph with a precomputed affinity matrix.
9. Run for `K=2..10`, select the smallest stable `K` via run-aware instability.
10. For each cluster, average hippocampal vertex timeseries across all vertices assigned to that cluster.
11. Compute `cluster-mean-timeseries -> canonical-network-timeseries` Pearson FC, apply Fisher z-transform, clip negatives to `0`, and annotate final clusters by dominant canonical network.

Interpretation:

- cortical feature granularity = canonical merged `network`
- hippocampal output type = clustered subregions
- spatial constraint = HippUnfold surface mesh adjacency
- negative FC policy = clip to `0` before spectral feature standardization
- clustering algorithm = spectral + KMeans (contrast with Ward used by other clustering branches)

### `intrinsic-spectral`

This branch applies spatially constrained spectral clustering to intrinsic hippocampal coupling profiles.

For each hemisphere:

1. Start from direct `vertex-to-vertex FC` on high-tSNR hippocampal vertices (shape `N x N`).
2. Apply Fisher z-transform to the intrinsic FC matrix with clipped input bounds and set the diagonal to `0`.
3. Treat each vertex row of that Fisher-z matrix as a feature profile.
4. Build a functional affinity matrix from pairwise cosine similarity of those intrinsic profiles, mapped to `[0, 1]`.
5. Build a binary spatial adjacency matrix from the HippUnfold hippocampal surface mesh triangles.
6. Fuse the two graphs by Hadamard (element-wise) product, retaining only spatially adjacent vertex pairs weighted by intrinsic functional similarity.
7. Run spectral clustering on the fused graph with a precomputed affinity matrix.
8. Run for `K=2..10`, select the smallest stable `K` via run-aware instability.
9. After clustering, average hippocampal vertex timeseries within each cluster, compute `cluster-mean-timeseries -> canonical-network-timeseries` Pearson FC, apply Fisher z-transform, and summarize each cluster for post hoc network interpretation.

Interpretation:

- cortical feature granularity = none during clustering (intrinsic-only)
- hippocampal output type = clustered subregions
- spatial constraint = HippUnfold surface mesh adjacency
- clustering algorithm = spectral + KMeans
- network labeling policy = post hoc `cluster-to-network FC` annotation

### `intrinsic-spectral-nonneg`

This branch matches `intrinsic-spectral`, except negative Fisher-z intrinsic FC values are clipped to `0` before spectral feature standardization.

For each hemisphere:

1. Start from direct `vertex-to-vertex FC` on high-tSNR hippocampal vertices (shape `N x N`).
2. Apply Fisher z-transform with clipped input bounds and set diagonal to `0`.
3. Clip negative Fisher-z values to `0`.
4. Treat each vertex row of the clipped intrinsic matrix as a feature profile.
5. Build a functional affinity matrix from pairwise cosine similarity of those intrinsic profiles, mapped to `[0, 1]`.
6. Build a binary spatial adjacency matrix from the HippUnfold hippocampal surface mesh triangles.
7. Fuse the two graphs by Hadamard (element-wise) product.
8. Run spectral clustering on the fused graph with a precomputed affinity matrix.
9. Run for `K=2..10`, select the smallest stable `K` via run-aware instability.
10. Summarize each final cluster by averaging cluster timeseries, computing `cluster-mean-timeseries -> canonical-network-timeseries` Pearson FC, applying Fisher z-transform, and clipping negatives to `0` before post hoc network interpretation.

Interpretation:

- cortical feature granularity = none during clustering (intrinsic-only)
- hippocampal output type = clustered subregions
- spatial constraint = HippUnfold surface mesh adjacency
- negative FC policy = Fisher-z then clip to `0`
- clustering algorithm = spectral + KMeans
- network labeling policy = post hoc `cluster-to-network FC` annotation

## K Evaluation

The final comparison uses `K=2..10` for `network-gradient`, `network-prob-cluster`, `network-prob-cluster-nonneg`, `network-prob-soft`, `network-prob-soft-nonneg`, `network-spectral`, `network-spectral-nonneg`, `intrinsic-spectral`, and `intrinsic-spectral-nonneg`.

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
   Run-wise inputs come from explicit `run-1..4` `dtseries` files or four-way boundaries inferred from `run-concat.dtseries` length (without writing run-level `dtseries` files), and run-wise hippocampal features are generated by splitting shared concat surface timeseries with those run boundaries.
2. Restrict candidates to local minima of `I_mean(K)`.
3. Apply the `1-SE` rule relative to the best instability point.
4. Among surviving candidates, choose the smallest `K` that also passes the pre-registered `V_min` vertex-count threshold.

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
- for `network-spectral` and `intrinsic-spectral`, heatmap rows display raw cluster-level Fisher-z `vertex-to-network FC` profiles computed from cluster-mean timeseries (signed)
- for `network-spectral-nonneg` and `intrinsic-spectral-nonneg`, those cluster-level Fisher-z profiles are clipped to `0` before export
- spectral heatmap rows are raw cluster profiles, not probability-normalized rows
- the left and right hemisphere heatmaps are displayed as separate panels with widened spacing for readability

## Downstream Group Prior + Fast-PFM Mapping

An optional downstream workflow consumes spectral branch outputs to produce reusable group priors and individual soft functional maps for new subjects.

Script:

- `scripts/hipp_parcellation_network/run_group_prior_fastpfm.py`

Scope:

- input branches: `network-spectral`, `network-spectral-nonneg`, `intrinsic-spectral`, `intrinsic-spectral-nonneg`
- grouping unit: `branch x atlas x smoothing x hemi`
- group `K` rule: aggregate subject `instability_mean` by `K`, then `local-min + 1-SE + min_parcel_pass_rate`, choose smallest surviving `K`
- label alignment: Hungarian matching to the first subject reference
- group prior payload: `prior_matrix (K x N_vertex)` plus `cluster_network_probs (K x N_network)` and dominant network labels
- individual mapping: Fast-PFM-style temporal projection and correlation scoring against prior clusters, then row-min-shift normalization to `scores_prob (K x N_vertex)`

Downstream outputs:

```text
outputs_migration/hipp_group_prior_fastpfm/<branch>/<atlas>/<smoothing>/
  group_k_selection.tsv/json
  priors/group_prior_*_hemi-{L|R}.pickle
  individual_soft_maps/sub-*/sub-*_soft_functional_map.pickle
  template + subject workbench assets/renders
  group_prior_manifest.json
  individual_softmap_manifest.json
```

Operational guardrails:

- this workflow never reruns `run_subject.py`; it only consumes existing subject outputs plus `_shared` hippocampal surface timeseries
- missing required inputs fail fast with explicit absolute paths (no fallback)
- current locked-scene rendering mode is `layout=1x2` (ventral extraction from native scene capture)

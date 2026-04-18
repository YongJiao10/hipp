# Hippocampal Functional Parcellation: Network-First Step-by-Step Flows

This document records the ten active network-first routes in this worktree.

The defining rule of this variant is simple:

- first build cortex canonical merged `network` timeseries
- then compute shared direct hippocampal `vertex-to-network FC` (network branches) and `vertex-to-vertex FC` (intrinsic branches)
- never compute `vertex-to-parcel FC` as the scientific feature basis for these ten branches

It is the detailed companion to [multi_branch_flow.md](/Users/jy/Documents/HippoMaps-network-first/docs/hipp_parcellation_network/multi_branch_flow.md). The high-level branch definitions there and the step-by-step procedures here should always stay in sync.

For the narrowed HPC handoff profile, also see [network_first_hpc_bundle_handoff.md](/Users/jy/Documents/HippoMaps-network-first/docs/hipp_parcellation_network/network_first_hpc_bundle_handoff.md). That bundle keeps only `network-gradient` and `network-prob-cluster-nonneg` with `lynch2024` and `kong2019`, but it uses the same shared upstream logic documented here.

## Shared Upstream

All ten methods share the same upstream steps before branch-specific processing:

1. Compute cortex `tSNR = 10000 / std(t)` directly on left and right cortical grayordinates from the pre-downstream `dtseries`.
2. Hard-mask all cortical grayordinates with `tSNR < 25`.
3. Extract individualized cortical ROI-component timeseries from the chosen cortical atlas output using only the remaining high-tSNR cortical grayordinates.
4. Merge atlas-specific parent networks to canonical cross-atlas network labels using [cross_atlas_network_merge.json](/Users/jy/Documents/HippoMaps-network-first/config/cross_atlas_network_merge.json).
5. Exclude `Noise`.
6. Resolve run-wise inputs for run-pair instability:
   if explicit `run-1..4` `dtseries` files are available, use them to derive run lengths; otherwise infer equal four-way run boundaries from `run-concat.dtseries` length.
   Do not materialize run-level `dtseries` files.
   Run-wise hippocampal inputs are then generated only by splitting shared concat hippocampal surface timeseries with those run boundaries (no raw volume BOLD run-splitting).
7. Average ROI-component timeseries within each retained canonical network to obtain cortex `network` timeseries.
8. Generate left and right hippocampal raw surface timeseries inside the shared pipeline store by sampling `run-concat_bold` onto the `corobl` surfaces with `trilinear` mapping and `smooth_iters = 0`; these shared-pipeline `.func.gii` files are the only valid raw source.
9. Compute hippocampal `tSNR = 10000 / std(t)` on those raw unsmoothed shared-pipeline `.func.gii` timeseries and hard-mask all vertices with `tSNR < 25`.
10. Run `2mm` and `4mm` smoothing only after the hippocampal tSNR gate and only within the remaining high-tSNR hippocampal ROI so masked vertices never contribute to smoothed values.
11. Compute direct hippocampal `vertex-to-network FC` separately for each hemisphere and smoothing condition using only high-tSNR vertices, then cache those FC matrices in the shared upstream store at the `subject x atlas x smoothing x hemisphere` level.
12. Compute direct hippocampal `vertex-to-vertex FC` separately for each hemisphere and smoothing condition on the same high-tSNR vertices, and cache those intrinsic FC matrices in the same shared upstream store for intrinsic branch reuse.

## K Selection Modes (Explicit)

The implementation supports two explicit `K`-selection modes via `--k-selection-mode`:

- `mainline` (default): current production rule; choose the smallest `K` within `0.02` of best `null_corrected_score` and with `min_cluster_size_fraction >= 0.05`.
- `experimental`: future-testing rule; local-minimum + `1-SE` + minimum parcel-size constraint (`V_min` only; connectivity is diagnostic but not a gate).

Operational rule:
- For canonical batch runs, use fixed launchers:
  - `scripts/hipp_parcellation_network/run_mainline.sh`
  - `scripts/hipp_parcellation_network/run_experimental.sh`
- Do not hand-write `--k-selection-mode`, `--run-split-mode`, `--out-root`, or `--present-dir` in canonical run commands.
- Do not infer mode from branch name or previous runs.

Notation:

- `N_vertex` = number of hippocampal surface vertices in one hemisphere
- `N_network` = number of canonical merged functional networks retained for that atlas

## Network Merge

Branch-specific analysis starts only after cortex canonical network timeseries have been created.

So the effective branch input is:

```text
raw cortex dtseries
  -> cortex tSNR gate (threshold 25)
cortex ROI-component timeseries
  -> cross-atlas network merge
  -> Noise exclusion
  -> cortex canonical network timeseries
  -> raw hippocampal vertex timeseries
  -> hippocampal tSNR gate (threshold 25)
  -> ROI-restricted smoothing
   -> shared direct vertex-to-network FC (N_vertex x N_network)
   -> shared direct vertex-to-vertex FC (N_vertex x N_vertex)
  -> branch-specific network-first analysis
```

Operational consequences:

- cortical `tSNR < 25` grayordinates are removed before any ROI / network mean and never re-enter later steps.
- hippocampal raw input is strict: use only the shared-pipeline raw `.func.gii` generated from `run-concat_bold` with `trilinear` mapping and `smooth_iters = 0`.
- hippocampal `.npy` files, archived directories, and any other fallback source are disallowed for formal tSNR gating.
- hippocampal `tSNR < 25` vertices are excluded before any smoothing; smoothing is not allowed to propagate signal through masked vertices.
- direct hippocampal `vertex-to-network FC` and `vertex-to-vertex FC` are shared upstream artifacts keyed by `subject x atlas x smoothing x hemisphere`, not branch-local recomputation.
- hippocampal `tSNR < 25` vertices remain excluded through clustering and final rendering.

Per-atlas count change:

```text
Atlas            Raw Atlas Labels  Noise Excluded  Canonical Networks Retained
Kong2019                      17             no                            7
Hermosillo2024               14             no                            8
Lynch2024                    21            yes                            8
```

Per-atlas merge logic:

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

Shared canonical networks across all three atlases:

```text
Default / Visual / Somatomotor / DorsalAttention / VentralAttention / Control / Auditory
```

Atlas-specific eighth network:

```text
Kong2019         none (merged into Default)
Hermosillo2024   Limbic
Lynch2024        Language
```

## `network-gradient`

This is the network-space diffusion-gradient route.

### Step-by-step

1. Start from the hemisphere-specific direct `vertex-to-network FC` matrix.
2. Apply Fisher z-transform to that FC matrix with clipped correlation bounds.
3. Treat each hippocampal vertex as an `N_network`-dimensional Fisher-z network fingerprint.
4. Build a sparse vertex-by-vertex affinity graph from those network fingerprints.
5. Run diffusion-map embedding on that graph.
6. Keep the first `3` nontrivial diffusion gradients as clustering features.
7. Z-score those gradient features across vertices.
8. Run spatially constrained Ward clustering for each `K` in `2..10`.
9. Evaluate each `K` with run-pair instability, `ARI`, homogeneity, parcel-size, and connectivity metrics.
10. Choose the final `K` with the repository instability rule: local minima -> `1-SE` -> non-triviality constraints.
11. Save the final hippocampal subregion labels.
12. Annotate each final cluster by its dominant canonical network.

### Shapes

```text
cortex canonical network timeseries  N_network x N_time
vertex-to-network FC                 N_vertex x N_network
Fisher-z vertex-to-network FC        N_vertex x N_network
vertex-vertex affinity graph         N_vertex x N_vertex
diffusion gradients                  N_vertex x 5
clustering features                  N_vertex x 3
final subregion labels               N_vertex
cluster probability rows             K x N_network
```

### Interpretation

- cortical feature granularity = canonical merged `network`
- hippocampal output type = clustered subregions
- network labels = post hoc annotations

## `network-prob-cluster`

This is the network-probability clustering route.

### Step-by-step

1. Start from the hemisphere-specific direct `vertex-to-network FC` matrix.
2. Convert each vertex network profile into a probability vector using:
   `Fisher z -> shift positive -> row normalize to sum=1`
3. Use those vertex-wise network probability vectors as clustering features.
4. Z-score the probability features across vertices.
5. Run spatially constrained Ward clustering for each `K` in `2..10`.
6. Evaluate each `K` with run-pair instability, `ARI`, homogeneity, parcel-size, and connectivity metrics.
7. Choose the final `K` with the repository instability rule: local minima -> `1-SE` -> `V_min` vertex-count.
8. Save the final hippocampal subregion labels.
9. Summarize each final cluster by its mean soft network probabilities.

### Shapes

```text
cortex canonical network timeseries  N_network x N_time
vertex-to-network FC                 N_vertex x N_network
network probabilities                N_vertex x N_network
clustering features                  N_vertex x N_network
final subregion labels               N_vertex
cluster probability rows             K x N_network
```

### Interpretation

- cortical feature granularity = canonical merged `network`
- hippocampal output type = clustered subregions
- cluster summaries = mean soft network profiles

## `network-prob-cluster-nonneg`

This is the network-probability clustering route with negative Fisher-z FC values clipped to zero before row normalization.

### Step-by-step

1. Start from the hemisphere-specific direct `vertex-to-network FC` matrix.
2. Convert each vertex network profile into a probability vector using:
   `Fisher z -> clip negative values to 0 -> row normalize to sum=1`
3. Use those vertex-wise network probability vectors as clustering features.
4. Z-score the probability features across vertices.
5. Run spatially constrained Ward clustering for each `K` in `2..10`.
6. Evaluate each `K` with run-pair instability, `ARI`, homogeneity, parcel-size, and connectivity metrics.
7. Choose the final `K` with the repository instability rule: local minima -> `1-SE` -> `V_min` vertex-count.
8. Save the final hippocampal subregion labels.
9. Summarize each final cluster by its mean soft network probabilities.

### Shapes

```text
cortex canonical network timeseries  N_network x N_time
vertex-to-network FC                 N_vertex x N_network
network probabilities                N_vertex x N_network
clustering features                  N_vertex x N_network
final subregion labels               N_vertex
cluster probability rows             K x N_network
```

### Interpretation

- cortical feature granularity = canonical merged `network`
- hippocampal output type = clustered subregions
- cluster summaries = mean soft network profiles

## `network-prob-soft`

This is the strict soft-first network route.

### Step-by-step

1. Start from the hemisphere-specific direct `vertex-to-network FC` matrix.
2. Convert each vertex network profile into a probability vector using:
   `Fisher z -> shift positive -> row normalize to sum=1`
3. Estimate a long-axis ordering of hippocampal vertices from the surface geometry.
4. Regularize the vertex-wise probability vectors with:
   - mesh-adjacency smoothing on the hippocampal surface
   - long-axis smoothing so anterior-posterior organization is not erased
5. Re-normalize each vertex probability row after each regularization pass so rows still sum to `1`.
6. Use the regularized probability vectors as clustering features.
7. Z-score those regularized probability features across vertices.
8. Run spatially constrained Ward clustering for each `K` in `2..10`.
9. Evaluate each `K` with run-pair instability, `ARI`, homogeneity, parcel-size, and connectivity metrics.
10. Choose the final `K` with the repository instability rule: local minima -> `1-SE` -> `V_min` vertex-count.
11. Save the final hippocampal subregion labels.
12. Save the regularized soft probabilities as the main soft output.
13. Derive optional regularized `argmax` labels only for auxiliary inspection and summary statistics.

### Shapes

```text
cortex canonical network timeseries  N_network x N_time
vertex-to-network FC                 N_vertex x N_network
network probabilities                N_vertex x N_network
regularized probabilities            N_vertex x N_network
clustering features                  N_vertex x N_network
final subregion labels               N_vertex
cluster probability rows             K x N_network
argmax occupancy summary             N_network
```

### Interpretation

- cortical feature granularity = canonical merged `network`
- main scientific output type = regularized soft network probabilities
- hippocampal parcellation output type = clustered subregions from those regularized profiles
- auxiliary output type = regularized `argmax` labels and occupancy summaries

## `network-prob-soft-nonneg`

This is the strict soft-first network route with negative Fisher-z FC values clipped to zero before probability normalization.

### Step-by-step

1. Start from the hemisphere-specific direct `vertex-to-network FC` matrix.
2. Convert each vertex network profile into a probability vector using:
   `Fisher z -> clip negative values to 0 -> row normalize to sum=1`
3. Estimate a long-axis ordering of hippocampal vertices from the surface geometry.
4. Regularize the vertex-wise probability vectors with:
   - mesh-adjacency smoothing on the hippocampal surface
   - long-axis smoothing so anterior-posterior organization is not erased
5. Re-normalize each vertex probability row after each regularization pass so rows still sum to `1`.
6. Use the regularized probability vectors as clustering features.
7. Z-score those regularized probability features across vertices.
8. Run spatially constrained Ward clustering for each `K` in `2..10`.
9. Evaluate each `K` with run-pair instability, `ARI`, homogeneity, parcel-size, and connectivity metrics.
10. Choose the final `K` with the repository instability rule: local minima -> `1-SE` -> `V_min` vertex-count.
11. Save the final hippocampal subregion labels.
12. Save the regularized soft probabilities as the main soft output.
13. Derive optional regularized `argmax` labels only for auxiliary inspection and summary statistics.

### Shapes

```text
cortex canonical network timeseries  N_network x N_time
vertex-to-network FC                 N_vertex x N_network
network probabilities                N_vertex x N_network
regularized probabilities            N_vertex x N_network
clustering features                  N_vertex x N_network
final subregion labels               N_vertex
cluster probability rows             K x N_network
argmax occupancy summary             N_network
```

### Interpretation

- cortical feature granularity = canonical merged `network`
- main scientific output type = regularized soft network probabilities
- hippocampal parcellation output type = clustered subregions from those regularized profiles
- auxiliary output type = regularized `argmax` labels and occupancy summaries

## `network-spectral`

This branch applies spectral clustering to a graph that combines atlas-specific canonical network FC similarity with hippocampal surface mesh adjacency.

### Step-by-step

1. Start from the hemisphere-specific direct `vertex-to-network FC` matrix.
2. Apply Fisher z-transform to the FC matrix with clipped correlation bounds.
3. Treat the retained canonical merged networks for the chosen atlas as the feature axis, so the Fisher-z FC matrix has shape `N_vertex x N_network`.
4. Convert the vertex-wise Fisher-z FC rows into a dense functional affinity matrix using cosine similarity mapped to `[0, 1]`.
5. Build a binary spatial adjacency matrix from the HippUnfold hippocampal surface mesh triangles.
6. Fuse functional affinity and mesh adjacency by element-wise multiplication so only mesh-neighbor pairs retain non-zero functional weights.
7. Run spectral clustering with the fused graph as a precomputed affinity matrix.
8. Repeat clustering for each `K` in `2..10`.
9. Evaluate each `K` with run-pair instability, `ARI`, homogeneity, parcel-size, and connectivity metrics.
10. Choose the final `K` with the repository instability rule: local minima -> `1-SE` -> `V_min` vertex-count.
11. Save the final hippocampal subregion labels.
12. For each cluster, average hippocampal vertex timeseries across all vertices assigned to that cluster.
13. Compute `cluster-mean-timeseries -> canonical-network-timeseries` Pearson FC.
14. Apply Fisher z-transform to that cluster-level FC vector, then summarize each cluster by its dominant canonical network profile.

### Shapes

```text
cortex canonical network timeseries  N_network x N_time
vertex-to-network FC                 N_vertex x N_network
Fisher-z vertex-to-network FC        N_vertex x N_network
functional affinity                  N_vertex x N_vertex
surface mesh adjacency               N_vertex x N_vertex
fused spectral graph                 N_vertex x N_vertex
final subregion labels               N_vertex
cluster mean timeseries              K x N_time
cluster Fisher-z FC profile          K x N_network
cluster probability rows             K x N_network
```

### Interpretation

- cortical feature granularity = canonical merged `network`
- hippocampal output type = clustered subregions
- spatial constraint = HippUnfold surface mesh adjacency
- clustering algorithm = spectral clustering on fused functional-spatial affinity

## `network-spectral-nonneg`

This branch matches `network-spectral`, except negative Fisher-z `vertex-to-network FC` values are clipped to `0` before the spectral feature standardization and affinity construction steps.

### Step-by-step

1. Start from the hemisphere-specific direct `vertex-to-network FC` matrix.
2. Apply Fisher z-transform to the FC matrix with clipped correlation bounds.
3. Clip negative transformed FC values to `0`, retaining only non-negative network coupling per vertex.
4. Treat the retained canonical merged networks for the chosen atlas as the feature axis, so the FC matrix has shape `N_vertex x N_network`.
5. Z-score the clipped Fisher-z FC features across vertices, exactly as in `network-spectral`.
6. Convert the vertex-wise FC rows into a dense functional affinity matrix using cosine similarity mapped to `[0, 1]`.
7. Build a binary spatial adjacency matrix from the HippUnfold hippocampal surface mesh triangles.
8. Fuse functional affinity and mesh adjacency by element-wise multiplication so only mesh-neighbor pairs retain non-zero functional weights.
9. Run spectral clustering with the fused graph as a precomputed affinity matrix.
10. Repeat clustering for each `K` in `2..10`.
11. Evaluate each `K` with run-pair instability, `ARI`, homogeneity, parcel-size, and connectivity metrics.
12. Choose the final `K` with the repository instability rule: local minima -> `1-SE` -> `V_min` vertex-count.
13. Save the final hippocampal subregion labels.
14. For each cluster, average hippocampal vertex timeseries across all vertices assigned to that cluster.
15. Compute `cluster-mean-timeseries -> canonical-network-timeseries` Pearson FC.
16. Apply Fisher z-transform and clip negatives to `0`, then summarize each cluster by its dominant canonical network profile.

### Shapes

```text
cortex canonical network timeseries  N_network x N_time
vertex-to-network FC                 N_vertex x N_network
Fisher-z vertex-to-network FC        N_vertex x N_network
nonnegative vertex-to-network FC     N_vertex x N_network
functional affinity                  N_vertex x N_vertex
surface mesh adjacency               N_vertex x N_vertex
fused spectral graph                 N_vertex x N_vertex
final subregion labels               N_vertex
cluster mean timeseries              K x N_time
cluster Fisher-z FC profile          K x N_network
cluster probability rows             K x N_network
```

### Interpretation

- cortical feature granularity = canonical merged `network`
- hippocampal output type = clustered subregions
- spatial constraint = HippUnfold surface mesh adjacency
- negative FC policy = clip to `0` before spectral feature standardization
- clustering algorithm = spectral clustering on fused functional-spatial affinity

## `intrinsic-spectral`

This branch applies spectral clustering to intrinsic hippocampal coupling profiles, then uses canonical cortical networks only for post hoc cluster interpretation.
Rendered panel titles keep the explicit branch slug (`intrinsic-spectral`) so signed and nonnegative intrinsic outputs are visually distinguishable during review.

### Step-by-step

1. Start from the hemisphere-specific direct `vertex-to-vertex FC` matrix on high-tSNR vertices.
2. Apply Fisher z-transform to that FC matrix with clipped correlation bounds for numerical stability.
3. Set the matrix diagonal to `0` so self-correlation does not contribute as a feature.
4. Z-score the transformed intrinsic FC rows across vertices.
5. Build a dense functional affinity matrix from pairwise cosine similarity of those intrinsic rows, mapped to `[0, 1]`.
6. Build a binary spatial adjacency matrix from the HippUnfold hippocampal surface mesh triangles.
7. Fuse functional affinity and mesh adjacency by element-wise multiplication so only mesh-neighbor pairs retain non-zero functional weights.
8. Run spectral clustering with the fused graph as a precomputed affinity matrix.
9. Repeat clustering for each `K` in `2..10`.
10. Evaluate each `K` with run-pair instability, `ARI`, homogeneity, parcel-size, and connectivity metrics.
11. Choose the final `K` with the repository instability rule.
12. Save the final hippocampal subregion labels.
13. For each cluster, average hippocampal vertex timeseries across all vertices assigned to that cluster.
14. Compute `cluster-mean-timeseries -> canonical-network-timeseries` Pearson FC.
15. Apply Fisher z-transform to that cluster-level FC vector, then label each cluster by dominant network.

### Shapes

```text
cortex canonical network timeseries  N_network x N_time
vertex-to-vertex FC                 N_vertex x N_vertex
Fisher-z intrinsic FC               N_vertex x N_vertex
functional affinity                 N_vertex x N_vertex
surface mesh adjacency              N_vertex x N_vertex
fused spectral graph                N_vertex x N_vertex
final subregion labels              N_vertex
cluster mean timeseries             K x N_time
cluster Fisher-z FC profile         K x N_network
cluster probability rows            K x N_network
```

### Interpretation

- clustering feature basis = intrinsic hippocampal coupling only
- hippocampal output type = clustered subregions
- spatial constraint = HippUnfold surface mesh adjacency
- network summaries = post hoc `cluster-to-network FC` annotation only

## `intrinsic-spectral-nonneg`

This branch matches `intrinsic-spectral`, except negative Fisher-z intrinsic FC values are clipped to `0` before spectral feature standardization.
Rendered panel titles keep the explicit branch slug (`intrinsic-spectral-nonneg`) so nonnegative outputs are not conflated with the signed branch in exported figures.

### Step-by-step

1. Start from the hemisphere-specific direct `vertex-to-vertex FC` matrix on high-tSNR vertices.
2. Apply Fisher z-transform with clipped correlation bounds.
3. Set diagonal to `0`.
4. Clip negative transformed FC values to `0`.
5. Z-score the clipped intrinsic FC rows across vertices.
6. Build a dense functional affinity matrix from cosine similarity of those rows, mapped to `[0, 1]`.
7. Build hippocampal mesh adjacency from surface triangles.
8. Fuse functional affinity and mesh adjacency by element-wise multiplication.
9. Run spectral clustering for each `K` in `2..10`.
10. Evaluate each `K` with run-pair instability, `ARI`, homogeneity, parcel-size, and connectivity metrics.
11. Choose final `K` with the repository instability rule.
12. Save final labels.
13. For each cluster, average hippocampal vertex timeseries across all vertices assigned to that cluster.
14. Compute `cluster-mean-timeseries -> canonical-network-timeseries` Pearson FC.
15. Apply Fisher z-transform, clip negatives to `0`, and export post hoc `cluster-to-network FC` summaries.

### Shapes

```text
cortex canonical network timeseries  N_network x N_time
vertex-to-vertex FC                 N_vertex x N_vertex
Fisher-z intrinsic FC               N_vertex x N_vertex
nonnegative intrinsic FC            N_vertex x N_vertex
functional affinity                 N_vertex x N_vertex
surface mesh adjacency              N_vertex x N_vertex
fused spectral graph                N_vertex x N_vertex
final subregion labels              N_vertex
cluster mean timeseries             K x N_time
cluster Fisher-z FC profile         K x N_network
cluster probability rows            K x N_network
```

### Interpretation

- clustering feature basis = intrinsic hippocampal coupling only
- hippocampal output type = clustered subregions
- spatial constraint = HippUnfold surface mesh adjacency
- negative FC policy = Fisher-z then clip to `0`
- network summaries = post hoc `cluster-to-network FC` annotation only

## `network-wta`

This is the pure network winner-takes-all route.

### Step-by-step

1. Start from the hemisphere-specific direct `vertex-to-network FC` matrix.
2. Apply Fisher z-transform to the FC matrix with clipped correlation bounds.
3. For each hippocampal vertex, find the maximum transformed FC score across all retained networks.
4. Assign that vertex the label of the winning network.
5. Calculate a confidence metric as the difference between the maximum transformed FC score and the second highest transformed FC score.
6. Skip clustering.
7. Save the winner-takes-all labels, confidence values, raw Pearson FC summaries, and Fisher-z FC summaries.

### Shapes

```text
cortex canonical network timeseries  N_network x N_time
vertex-to-network FC                 N_vertex x N_network
Fisher-z vertex-to-network FC        N_vertex x N_network
final network labels                 N_vertex
confidence values                    N_vertex
mean network FC summary              N_network
network occupancy summary            N_network
```

### Interpretation

- cortical feature granularity = canonical merged `network`
- hippocampal output type = direct network labels
- this is the only branch that does not create new hippocampal subregions

## Stage Output Policy

- Pipeline runs in resumable stages: `reference`, `surface`, `compute`, `render`, `summary`
- Outputs are isolated to `outputs/hipp_functional_parcellation_network/`
- Final overview copies are isolated to `present_network/`
- Density input is authoritative: if run with `--hipp-density X`, every consumed hippocampal asset must be `den-X`.
- Files without `den-` are treated as legacy and must fail fast with regeneration guidance.
- Surface source fallback from `work/sub-*/surf` is disallowed; analysis reads only `<hippunfold-dir>/sub-<id>/surf`.
- Default retention keeps render-layer artifacts so legend/layout changes can rerender without recomputing feature or clustering stages
- Structural hippocampal labels for network-first renders are sourced from HippUnfold's subject-level structural `dlabel.nii`, then separated into left/right label files inside the run output tree for rendering
- In `network_probability_heatmaps.png`, the x-axis always shows all retained merged networks for that atlas in canonical order
- For `network-spectral` and `intrinsic-spectral`, heatmap values are cluster-level Fisher-z `vertex-to-network FC` profiles computed from `cluster mean timeseries` (signed)
- For `network-spectral-nonneg` and `intrinsic-spectral-nonneg`, the same cluster-level Fisher-z profiles are clipped to `0` before export
- Spectral heatmaps use these raw cluster profiles, not probability-normalized rows
- The left and right hemisphere heatmaps are rendered as separate panels with widened spacing for readability

## Downstream Group Prior + Fast-PFM (Spectral)

This optional downstream stage is designed for cross-subject prior construction and individualized remapping after spectral clustering outputs are already available.

Input contract:

1. Existing spectral branch subject outputs under `outputs_migration/hipp_functional_parcellation_network/<branch>/<atlas>/sub-<subject>/`.
2. Existing shared hippocampal timeseries and tSNR masks under `outputs_migration/hipp_functional_parcellation_network/_shared/sub-<subject>/surface/`.

Workflow steps per `branch x atlas x smoothing x hemi`:

1. Read each subject `per_k_summary.tsv`, aggregate `instability_mean` by `K`, compute group mean/SE and pass-rate.
2. Choose group `K` using `local minima -> 1-SE -> min_parcel_pass_rate`, then smallest surviving `K`.
3. Load each subject `cluster_labels_full.npy` at chosen `K`.
4. Align subject labels to the first-subject reference with Hungarian matching on overlapping labeled vertices.
5. Build `prior_matrix (K x N_vertex)` as the mean aligned one-hot map.
6. Read each subject `cluster_annotation.json` probability rows, reorder by the same mapping, average to `cluster_network_probs (K x N_network)`, and assign dominant network by `argmax`.

Individual inference steps:

1. Load subject hemisphere timeseries `X (N_vertex x T)` and valid mask.
2. Intersect subject valid mask with prior-supported mask.
3. Standardize time series in time domain, project with `prior_ts = prior_matrix @ X^T`.
4. Standardize `prior_ts`, compute correlation-like scores `scores_raw (K x N_vertex)`.
5. Convert to `scores_prob` using row-min-shift + per-vertex normalization.
6. Save `wta_labels` and `confidence_margin` as auxiliary outputs.

Output contract:

- group prior pickle:
  - required keys include `prior_matrix`, `k_final`, `cluster_network_probs`, `cluster_dominant_network`, `valid_vertex_mask`
- individual soft map pickle:
  - required keys include `scores_raw`, `scores_prob`, `wta_labels`, `confidence_margin`, `valid_vertex_mask`, `prior_pickle`
- manifests:
  - `group_k_selection.json`
  - `group_prior_manifest.json`
  - `individual_softmap_manifest.json`

Rendering:

- group template and individual WTA views are rendered by converting pickles to Workbench labels and reusing the locked scene batch renderer.
- current implementation uses `layout=1x2` for the locked native scene.

## Maintenance Note

If any network-first branch definition, feature construction rule, regularization rule, clustering rule, selection rule, output naming rule, or overview interpretation changes, this document and [multi_branch_flow.md](/Users/jy/Documents/HippoMaps-network-first/docs/hipp_parcellation_network/multi_branch_flow.md) must be updated in the same change.

# Hippocampal Functional Parcellation: Network-First Step-by-Step Flows

This document records the six active network-first routes in this worktree.

The defining rule of this variant is simple:

- first build cortex canonical merged `network` timeseries
- then compute direct hippocampal `vertex-to-network FC`
- never compute `vertex-to-parcel FC` as the scientific feature basis for these six branches

It is the detailed companion to [multi_branch_flow.md](/Users/jy/Documents/HippoMaps-network-first/docs/experiments/hipp_functional_parcellation/multi_branch_flow.md). The high-level branch definitions there and the step-by-step procedures here should always stay in sync.

## Shared Upstream

All six methods share the same upstream steps before branch-specific processing:

1. Extract individualized cortical ROI-component timeseries from the chosen cortical atlas output.
2. Merge atlas-specific parent networks to canonical cross-atlas network labels using [cross_atlas_network_merge.json](/Users/jy/Documents/HippoMaps-network-first/config/cross_atlas_network_merge.json).
3. Exclude `Noise`.
4. Average ROI-component timeseries within each retained canonical network to obtain cortex `network` timeseries.
5. Sample left and right hippocampal resting-state timeseries on the `corobl` surfaces.
6. Compute direct hippocampal `vertex-to-network FC` separately for each hemisphere and smoothing condition.

Notation:

- `N_vertex` = number of hippocampal surface vertices in one hemisphere
- `N_network` = number of canonical merged functional networks retained for that atlas

## Network Merge

Branch-specific analysis starts only after cortex canonical network timeseries have been created.

So the effective branch input is:

```text
cortex ROI-component timeseries
  -> cross-atlas network merge
  -> Noise exclusion
  -> cortex canonical network timeseries
  -> hippocampal vertex timeseries
  -> direct vertex-to-network FC (N_vertex x N_network)
  -> branch-specific network-first analysis
```

Per-atlas count change:

```text
Atlas            Raw Atlas Labels  Noise Excluded  Canonical Networks Retained
Kong2019                      17             no                            8
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
  TemporalParietal                                    -> TemporalParietal

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
Kong2019         TemporalParietal
Hermosillo2024   Limbic
Lynch2024        Language
```

## `network-gradient`

This is the network-space diffusion-gradient route.

### Step-by-step

1. Start from the hemisphere-specific direct `vertex-to-network FC` matrix.
2. Treat each hippocampal vertex as an `N_network`-dimensional network fingerprint.
3. Build a sparse vertex-by-vertex affinity graph from those network fingerprints.
4. Run diffusion-map embedding on that graph.
5. Keep the first `3` nontrivial diffusion gradients as clustering features.
6. Z-score those gradient features across vertices.
7. Run spatially constrained Ward clustering for each `K` in `3..8`.
8. Evaluate each `K` with split-half `ARI` and the other selection metrics.
9. Choose the final `K` using the repository selection rule.
10. Save the final hippocampal subregion labels.
11. Annotate each final cluster by its dominant canonical network.

### Shapes

```text
cortex canonical network timeseries  N_network x N_time
vertex-to-network FC                 N_vertex x N_network
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
5. Run spatially constrained Ward clustering for each `K` in `3..8`.
6. Evaluate each `K` with split-half `ARI` and the other selection metrics.
7. Choose the final `K` using the repository selection rule.
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
5. Run spatially constrained Ward clustering for each `K` in `3..8`.
6. Evaluate each `K` with split-half `ARI` and the other selection metrics.
7. Choose the final `K` using the repository selection rule.
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
8. Run spatially constrained Ward clustering for each `K` in `3..8`.
9. Evaluate each `K` with split-half `ARI` and the other selection metrics.
10. Choose the final `K` using the repository selection rule.
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
8. Run spatially constrained Ward clustering for each `K` in `3..8`.
9. Evaluate each `K` with split-half `ARI` and the other selection metrics.
10. Choose the final `K` using the repository selection rule.
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

## `network-wta`

This is the pure network winner-takes-all route.

### Step-by-step

1. Start from the hemisphere-specific direct `vertex-to-network FC` matrix.
2. For each hippocampal vertex, find the maximum FC score across all retained networks.
3. Assign that vertex the label of the winning network.
4. Calculate a confidence metric as the difference between the maximum FC score and the second highest FC score.
5. Skip clustering.
6. Save the winner-takes-all labels, confidence values, and network score summaries.

### Shapes

```text
cortex canonical network timeseries  N_network x N_time
vertex-to-network FC                 N_vertex x N_network
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
- Default retention keeps render-layer artifacts so legend/layout changes can rerender without recomputing feature or clustering stages
- In `network_probability_heatmaps.png`, the x-axis always shows all retained merged networks for that atlas in canonical order
- The left and right hemisphere heatmaps are rendered as separate panels with widened spacing for readability

## Maintenance Note

If any network-first branch definition, feature construction rule, regularization rule, clustering rule, selection rule, output naming rule, or overview interpretation changes, this document and [multi_branch_flow.md](/Users/jy/Documents/HippoMaps-network-first/docs/experiments/hipp_functional_parcellation/multi_branch_flow.md) must be updated in the same change.

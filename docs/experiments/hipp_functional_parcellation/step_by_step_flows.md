# Hippocampal Functional Parcellation: Step-by-Step Flows

This document records the four active post-`vertex-to-parcel FC` routes in explicit step-by-step form.

It is the detailed companion to [multi_branch_flow.md](/Users/jy/Documents/HippoMaps/docs/experiments/hipp_functional_parcellation/multi_branch_flow.md). The high-level branch definitions in that document and the step-by-step procedures in this document should always stay in sync.

## Shared Upstream

All four methods share the same upstream steps before branch-specific processing:

1. Extract individualized cortical ROI-component timeseries from the chosen cortical atlas output.
2. Sample left and right hippocampal resting-state timeseries on the `corobl` surfaces.
3. Compute a hippocampal `vertex-to-parcel FC` matrix separately for each hemisphere and smoothing condition.

Notation:

- `N_vertex` = number of hippocampal surface vertices in one hemisphere
- `N_parcel` = number of cortical parcels after filtering
- `N_network` = number of cortical parent networks after grouping

For the current single-subject example used in repository docs:

- `vertex-to-parcel FC` shape = `12761 x 404`

## `gradient`

This is the diffusion-gradient route.

### Step-by-step

1. Start from the hemisphere-specific `vertex-to-parcel FC` matrix.
2. Treat each hippocampal vertex as a `404`-dimensional cortical connectivity fingerprint.
3. Build a sparse vertex-by-vertex affinity graph from those FC fingerprints.
4. Run diffusion-map embedding on that graph.
5. Keep the first `3` nontrivial diffusion gradients as clustering features.
6. Z-score those gradient features across vertices.
7. Run spatially constrained Ward clustering for each `K` in `3..8`.
8. Evaluate each `K` with split-half `ARI` and the other selection metrics.
9. Choose the final `K` using the repository selection rule.
10. Save the final hippocampal subregion labels.
11. Annotate each final cluster by its dominant cortical parent network.

### Shapes

Typical shape flow for one hemisphere:

```text
vertex-to-parcel FC              N_vertex x N_parcel
vertex-vertex affinity graph     N_vertex x N_vertex
diffusion gradients              N_vertex x 5
clustering features              N_vertex x 3
final labels                     N_vertex
cluster probability rows         K x N_network
```

### Interpretation

- Main result = clustered hippocampal subregions discovered from FC geometry
- Network labels = post hoc annotations

## `prob-cluster`

This is the network-probability clustering route.

### Step-by-step

1. Start from the hemisphere-specific `vertex-to-parcel FC` matrix.
2. Group parcel FC values by cortical parent network.
3. Convert each vertex network profile into a probability vector using:
   `Fisher z -> shift positive -> row normalize to sum=1`
4. Use those vertex-wise network probability vectors as clustering features.
5. Z-score the probability features across vertices.
6. Run spatially constrained Ward clustering for each `K` in `3..8`.
7. Evaluate each `K` with split-half `ARI` and the other selection metrics.
8. Choose the final `K` using the repository selection rule.
9. Save the final hippocampal subregion labels.
10. Summarize each final cluster by its mean soft network probabilities.

### Shapes

Typical shape flow for one hemisphere:

```text
vertex-to-parcel FC              N_vertex x N_parcel
grouped network FC               N_vertex x N_network
network probabilities            N_vertex x N_network
clustering features              N_vertex x N_network
final labels                     N_vertex
cluster probability rows         K x N_network
```

### Interpretation

- Main result = clustered hippocampal subregions
- Cluster summaries = mean soft network profiles of those subregions

## `prob-soft`

This is the strict soft-first route requested for the current repository state.

### Step-by-step

1. Start from the hemisphere-specific `vertex-to-parcel FC` matrix.
2. Group parcel FC values by cortical parent network.
3. Convert each vertex network profile into a probability vector using:
   `Fisher z -> shift positive -> row normalize to sum=1`
4. Estimate a long-axis ordering of hippocampal vertices from the surface geometry.
5. Regularize the vertex-wise probability vectors with:
   - mesh-adjacency smoothing on the hippocampal surface
   - long-axis smoothing so anterior-posterior organization is not erased
6. Re-normalize each vertex probability row after each regularization pass so rows still sum to `1`.
7. Use the regularized probability vectors as clustering features.
8. Z-score those regularized probability features across vertices.
9. Run spatially constrained Ward clustering for each `K` in `3..8`.
10. Evaluate each `K` with split-half `ARI` and the other selection metrics.
11. Choose the final `K` using the repository selection rule.
12. Save the final hippocampal subregion labels.
13. Save the regularized soft probabilities as the main soft output.
14. Derive optional regularized `argmax` labels only for auxiliary inspection and summary statistics.

### Shapes

Typical shape flow for one hemisphere:

```text
vertex-to-parcel FC              N_vertex x N_parcel
grouped network FC               N_vertex x N_network
network probabilities            N_vertex x N_network
regularized probabilities        N_vertex x N_network
clustering features              N_vertex x N_network
final subregion labels           N_vertex
cluster probability rows         K x N_network
argmax occupancy summary         N_network
```

### Interpretation

- Main soft result = regularized vertex-wise network probabilities
- Main parcellation result = clustered hippocampal subregions from those regularized probability profiles
- Auxiliary result = regularized `argmax` labels and occupancy summaries
- Workbench render legend semantics = grouped by dominant network; all clusters under the same network use the same color and a shared legend entry
- Workbench render layout semantics = 2x2 ventral+dorsal views (top ventral, bottom dorsal) with white `L/R` column headers shared by structural and functional renders

## `wta`

This is the pure winner-takes-all hard assignment route.

### Step-by-step

1. Start from the hemisphere-specific `vertex-to-parcel FC` matrix.
2. Group parcel FC values by cortical parent network, yielding an average FC score per network.
3. For each hippocampal vertex, find the maximum average correlation across all networks.
4. Assign that vertex the label of the winning network (a discrete integer from 1 to `N_network`).
5. Calculate a "confidence" metric as the difference between the maximum correlation and the second highest correlation.
6. Skip clustering (`K` is implicitly equal to `N_network`).
7. Save the winner-takes-all labels and the confidence values.

### Shapes

Typical shape flow for one hemisphere:

```text
vertex-to-parcel FC              N_vertex x N_parcel
grouped network FC               N_vertex x N_network
final labels                     N_vertex
confidence values                N_vertex
```

### Interpretation

- Main parcellation result = discrete, non-overlapping hard assignment of vertices to predefined networks based solely on maximum correlation

## Stage Output Policy

- Pipeline runs in resumable stages: `reference`, `surface`, `compute`, `render`, `summary`
- Each stage writes `stage_manifest.json` to persist parameter signatures, input stamps, and produced artifacts
- Default retention keeps render-layer artifacts so legend/layout changes can rerender without recomputing feature or clustering stages

## Maintenance Note

If any branch definition, feature construction rule, regularization rule, clustering rule, selection rule, or overview interpretation changes, this document and [multi_branch_flow.md](/Users/jy/Documents/HippoMaps/docs/experiments/hipp_functional_parcellation/multi_branch_flow.md) must be updated in the same change.

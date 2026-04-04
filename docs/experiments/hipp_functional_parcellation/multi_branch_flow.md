# Hippocampal Functional Parcellation: Multi-Branch Comparison Flow

## Purpose

This document defines the current formal comparison workflow for hippocampal functional parcellation in this repository.

For step-by-step per-branch procedures, see [step_by_step_flows.md](/Users/jy/Documents/HippoMaps/docs/experiments/hipp_functional_parcellation/step_by_step_flows.md).

The goal is to start from the same subject-level `vertex-to-parcel FC` matrix and compare four downstream analysis branches:

- `gradient`: HippoMaps-style gradient discovery followed by spatial clustering
- `prob-cluster`: screenshot-style network probabilities followed by spatial clustering
- `prob-soft`: strict soft-first route: regularized vertex-wise network probabilities followed by hippocampal subregion clustering
- `wta`: winner-takes-all hard assignment of each vertex to its most correlated functional network

The current formal experiment matrix is:

```text
branches   gradient / prob-cluster / prob-soft / wta
atlases    lynch2024 / hermosillo2024
          (optional) kong2019
subjects   100610 / 102311 / 102816
smoothing  light(1-ring) / 4mm
```

## Core Rules

1. Left and right hippocampi are modeled independently.
2. Each `branch x atlas x subject x smoothing x hemi` gets its own features, clustering metrics, and final selection.
3. Final `K` is chosen independently for each hemisphere from `3..8`.
4. Smoothing is compared inside each overview; it is not promoted to a separate top-level comparison dimension.
5. Every `branch x atlas x subject` produces one overview image copied to `present/`.

## Shared Upstream

All branches share the same upstream preprocessing:

1. Extract individualized cortical ROI-component timeseries from the chosen cortical atlas output.
2. Sample hippocampal resting-state timeseries on the left and right `corobl` surfaces.
3. Build a hippocampal `vertex-to-parcel FC` matrix separately for each smoothing condition.

In notation:

```text
cortex ROI components
  -> parcel timeseries
  -> hippocampal surface timeseries
  -> vertex-to-parcel FC
  -> branch-specific features
```

## Branch Definitions

### `gradient`

This branch is the closest to the HippoMaps cartography logic.

For each hemisphere:

1. Treat each vertex as a parcel-FC feature vector.
2. Build a sparse vertex-by-vertex affinity graph.
3. Run diffusion-map embedding.
4. Use the first `3` gradients as clustering features.
5. Run spatially constrained Ward clustering for `K=3..8`.
6. Select the smallest stable `K`.
7. Annotate final clusters by their dominant cortical parent network.

### `prob-cluster`

This branch follows the screenshot logic, but still produces explicit parcels.

For each hemisphere:

1. Group parcel-FC values by cortical parent network.
2. Convert each vertex network profile into probabilities using:
   `Fisher z -> shift positive -> row normalize to sum=1`
3. Use those probability vectors as clustering features.
4. Run the same spatially constrained Ward clustering for `K=3..8`.
5. Select the smallest stable `K`.
6. Summarize each final cluster by its mean soft network probabilities.

### `prob-soft`

This branch now follows the strict soft-first route after `vertex-to-parcel FC`.

For each hemisphere:

1. Compute the same vertex-wise network probability vectors used by `prob-cluster`.
2. Regularize those probabilities on the hippocampal surface using mesh adjacency plus a long-axis smoothing term.
3. Cluster vertices by similarity of the regularized probability profiles for `K=3..8`.
4. Select the smallest stable `K`.
5. Save the regularized soft probabilities as the branch's main scientific result.
6. Derive optional regularized argmax labels only for auxiliary inspection.

Interpretation:

- main result = regularized soft vertex-wise network probabilities
- subregion result = clustered hippocampal subregions from those probability profiles
- auxiliary result = regularized argmax labels

### `wta`

This branch produces a simple winner-takes-all (Hard) functional assignment based directly on the raw FC matrix.

For each hemisphere:

1. Group parcel-FC values by cortical parent network, returning an average correlation score per network.
2. For each hippocampal vertex, find the network with the highest correlation score (the "winner").
3. Assign that vertex to the winning network (producing K=number of networks in the atlas).
4. No clustering or K-selection is performed; the output labels directly map to the predefined network list.
5. The model natively outputs confidence values based on `max_score - second_max_score`.

## K Evaluation

The final comparison uses `K=3..8` for every hemisphere independently.

Each candidate `K` records:

- odd/even split-half `ARI`
- silhouette
- Calinski-Harabasz
- Davies-Bouldin
- WCSS
- delta-WCSS
- minimum cluster-size fraction
- BSS/TSS
- connected-component count

Selection rule:

1. Find the best `ARI`.
2. Keep all `K` within `0.02` of that best `ARI`.
3. Among those, choose the smallest `K` with minimum cluster fraction `>= 0.05`.
4. If none pass that guardrail, fall back to the best-ARI solution.

## Outputs

Active outputs are written to:

```text
outputs/hipp_functional_parcellation/<branch>/<atlas>/sub-<subject>/
```

Each active result directory keeps summary files plus reusable stage artifacts:

```text
hipp_functional_parcellation_overview.png
k_selection_curves.png
cluster_probability_heatmaps.png
final_selection_core.json
final_selection_summary.json
summary_manifest.json
reference/stage_manifest.json
surface/stage_manifest.json
clustering/stage_manifest.json
renders/stage_manifest.json
summary/stage_manifest.json
```

By default, render-layer artifacts are retained so legend/layout updates can rerender without recomputing clustering.

## Overview Layout

Each `branch x atlas x subject` overview contains:

1. left and right metric panels at the top
2. branch-specific probability/profile summaries in the middle
3. bilateral render panels at the bottom (2x2: top ventral, bottom dorsal; left column L, right column R)

All three branches use `K=3..8` metric curves with `best K` annotations in the top row and multiple `K` render panels plus the final solution in the bottom row. `prob-soft` additionally keeps the regularized soft summaries in the middle row.

Workbench hippocampal render legends are network-level (not cluster-level): clusters that share the same dominant network share one color and one legend swatch.
Both structural and functional renders use the same ventral+dorsal 2x2 layout with white `L/R` column headers.

## Present Copy Rule

Every finished overview is copied to:

```text
present/sub-<subject>_<atlas>_<branch>_overview.png
```

The expected total is:

```text
4 branches x 2 atlases x 3 subjects = 24 overview images
```

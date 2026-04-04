# Hippocampal Functional Parcellation: Standard Single-Subject Flow

> Note
> This document describes the earlier gradient-centric implementation that clustered hippocampal vertices after a single combined left-right feature construction step. The current formal comparison workflow is documented in [multi_branch_flow.md](/Users/jy/Documents/HippoMaps/docs/experiments/hipp_functional_parcellation/multi_branch_flow.md), which defines the multi-branch `gradient / prob-cluster / prob-soft` experiment with per-hemisphere modeling and selection.

## Purpose

This document records the current standard workflow for single-subject hippocampal functional parcellation in this repository. It is written to be explicit about:

- what is treated as the fixed analysis backbone,
- how clustering is actually performed,
- where diffusion gradients enter the workflow,
- which parameter choices are arbitrary or pragmatic rather than theoretically unique.

This document reflects the current implementation centered on:

- individualized cortical reference features from `Lynch2024` ROI components,
- hippocampal surface timeseries sampled in `corobl` space at `2 mm` density,
- diffusion-map embedding of hippocampal vertex-to-cortex FC,
- spatially constrained clustering in gradient space,
- post hoc annotation of clusters by cortical parent networks.

## High-Level Logic

The workflow is intentionally split into two conceptual stages:

1. Discover continuous hippocampal functional organization from cortical connectivity fingerprints.
2. Convert that continuous organization into a compact set of spatially contiguous hippocampal functional parcels.

In practice, the pipeline is:

`individualized cortex parcels -> hippocampal vertex-to-parcel FC -> vertex similarity graph -> diffusion gradients -> spatially constrained clustering -> network annotation`

This means the final hippocampal parcels are **not** obtained by direct winner-take-all labeling from cortical networks. Instead, parcels are first discovered from the geometry of hippocampal connectivity patterns, and only then interpreted in network terms.

## Input Definition

For the current standard single-subject implementation:

- cortical feature basis: individualized `Lynch2024` ROI components,
- subject-level input timeseries: `dtseries` for cortex and hippocampal surface timeseries sampled from resting-state fMRI,
- hippocampal representation: left and right hippocampal surface vertices in `corobl` space,
- spatial density: `2 mm`.

The cortical feature basis is intentionally individualized rather than atlas-fixed. This is the main reason `Schaefer400` is not used as the principal feature space for this analysis.

## Step 1. Cortex Feature Extraction

The cortex is represented as individualized ROI-component parcels derived from `Lynch2024`.

Implementation:

- [extract_cortex_roi_component_timeseries.py](/Users/jy/Documents/HippoMaps/scripts/cortex/extract_cortex_roi_component_timeseries.py)

The pipeline:

1. Load left and right cortical ROI-component labels.
2. Extract parcel-level timeseries from the subject `dtseries`.
3. Merge left and right parcels into a single cortical feature space.
4. Exclude any ROI components derived from the `Noise` parent network.
5. Save both parcel-level timeseries and a metadata table mapping parcels to parent networks.

The output is a matrix of shape:

`n_cortex_parcels x n_timepoints`

For the current subject-level example, this feature space has `404` usable cortical parcels after `Noise` exclusion.

## Step 2. Hippocampal Surface Timeseries

The hippocampus is represented vertex-wise on the left and right midthickness surfaces.

The current standard comparison includes three smoothing conditions:

- `none`
- `1-ring`
- `4mm`

Meaning:

- `none`: no additional smoothing after surface sampling,
- `1-ring`: one local mesh-neighbor averaging step,
- `4mm`: explicit surface metric smoothing with `4 mm FWHM`.

Important note:

`1-ring` is a mesh-neighbor smoothing operation, not a literal `1 mm` or `2 mm FWHM` kernel. It is a lightweight local regularization step chosen as a pragmatic compromise between denoising and preserving hippocampal topology.

## Step 3. Vertex-to-Parcel FC Matrix

For each hippocampal vertex, the pipeline computes a correlation with every cortical parcel timeseries.

Implementation:

- [compute_fc_gradients.py](/Users/jy/Documents/HippoMaps/scripts/common/compute_fc_gradients.py)
- [run_subject.py](/Users/jy/Documents/HippoMaps/scripts/experiments/hipp_functional_parcellation/run_subject.py)

Mathematically:

- rows = hippocampal vertices,
- columns = cortical parcels,
- each row is a cortical connectivity fingerprint for one hippocampal vertex.

So the FC matrix is:

`n_hippocampal_vertices x n_cortex_parcels`

For the current subject-level run, this is:

`12761 x 404`

This matrix is the key object that links hippocampal organization to cortex-wide functional coupling.

## Step 4. Why Diffusion Gradients Are Used

The pipeline does not cluster the raw FC matrix directly.

Instead, it first asks:

"Which hippocampal vertices have similar cortex-wide FC fingerprints?"

This is important because hippocampal organization is expected to be partly continuous rather than purely discrete. A direct hard clustering on the raw high-dimensional FC matrix would be more sensitive to noise and would obscure dominant smooth axes of variation.

The gradient step proceeds as follows:

1. Treat each hippocampal vertex as a feature vector over cortical parcels.
2. Compute a vertex-by-vertex similarity matrix from those connectivity fingerprints.
3. Sparsify that matrix to retain only the strongest relationships.
4. Run diffusion-map embedding on the resulting graph.

Conceptually:

- `Gradient 1` captures the strongest continuous axis of hippocampal FC variation,
- `Gradient 2` captures the next strongest independent axis,
- `Gradient 3` captures the next one after that.

This creates a low-dimensional representation in which nearby points have similar cortical connectivity fingerprints.

## Step 5. Affinity Graph Construction

The vertex similarity graph is built from the FC matrix using the following sequence:

1. Normalize each vertex FC fingerprint to unit length.
2. Compute pairwise cosine similarity between hippocampal vertices.
3. Rescale similarity to `[0, 1]`.
4. Keep only a sparse neighborhood of strongest edges.
5. Symmetrize the graph and set the diagonal to `1`.

Implementation:

- [build_sparse_affinity](/Users/jy/Documents/HippoMaps/scripts/common/compute_fc_gradients.py)

The current sparsity setting is:

- `sparsity = 0.1`

This means the graph retains only a fraction of strongest neighbors per vertex, rather than operating on a dense fully connected similarity matrix.

## Step 6. Diffusion-Map Embedding

After graph construction, the pipeline computes a normalized graph operator and obtains its leading eigenvectors.

Implementation:

- [diffusion_map_embedding](/Users/jy/Documents/HippoMaps/scripts/common/compute_fc_gradients.py)

The implementation currently:

- computes up to `5` diffusion components,
- discards the trivial first eigenvector,
- saves the remaining gradients,
- uses only the first `3` gradients for clustering.

This mirrors the common logic in gradient analyses: calculate several components, then carry a smaller set of dominant gradients into downstream tasks.

## Step 7. Clustering in Gradient Space

Clustering is performed in the space of the first three diffusion gradients, not in the original FC matrix.

Implementation:

- [run_subject.py](/Users/jy/Documents/HippoMaps/scripts/experiments/hipp_functional_parcellation/run_subject.py)

The clustering features are:

- `Gradient 1`
- `Gradient 2`
- `Gradient 3`

These are column-z-scored before clustering.

The clustering method is:

- Ward agglomerative clustering,
- with hippocampal surface adjacency as a connectivity constraint.

Implementation:

- [cluster_embedding](/Users/jy/Documents/HippoMaps/scripts/experiments/hipp_functional_parcellation/run_subject.py)
- [build_surface_adjacency](/Users/jy/Documents/HippoMaps/scripts/experiments/hipp_functional_parcellation/run_subject.py)

Why this method is used:

- Ward clustering favors compact clusters in feature space.
- The mesh adjacency constraint prevents the algorithm from grouping spatially distant vertices into the same parcel simply because their features are similar.
- This helps produce contiguous hippocampal parcels rather than fragmented scattered assignments.

## Step 8. Why Left and Right Hippocampus Are Clustered Separately

The current implementation does not allow direct cross-hemisphere cluster growth.

Instead:

1. Left and right hippocampal vertices are split.
2. Total `K` is divided across hemispheres in proportion to vertex counts.
3. Each hemisphere is clustered separately under its own surface adjacency graph.
4. Labels are then concatenated and renumbered.

Implementation:

- [cluster_embedding_blockdiag](/Users/jy/Documents/HippoMaps/scripts/experiments/hipp_functional_parcellation/run_subject.py)

This is a pragmatic engineering choice designed to avoid pathological behavior that can happen when disconnected graphs are handed to standard agglomerative clustering.

## Step 9. K Evaluation

The pipeline evaluates:

- `K = 2..8`

For each `K`, it records:

- split-half `ARI` from odd/even timepoint partitions,
- silhouette score,
- Calinski-Harabasz score,
- Davies-Bouldin score,
- within-cluster sum of squares,
- delta WCSS,
- minimum cluster size fraction,
- BSS/TSS ratio,
- cluster connected-component count.

The role of these metrics is not identical:

- `ARI` is a stability metric,
- silhouette, Calinski-Harabasz, Davies-Bouldin, WCSS and BSS/TSS are compactness/separation metrics,
- minimum cluster size fraction is a fragmentation guardrail,
- connected-component count checks whether clusters remain spatially contiguous.

## Step 10. Final K Selection Rule

The current standard selection rule is intentionally conservative.

Only `K = 3, 4, 5, 6` are eligible for the final choice.

Selection rule:

1. Find the `K` in that shortlist with the highest odd/even `ARI`.
2. Define a tolerance window of `best_ARI - 0.02`.
3. Among all shortlist values inside that window, choose the **smallest** `K` whose minimum cluster fraction is at least `0.05`.
4. If none satisfy that criterion, use the `K` with the highest `ARI`.

Implementation:

- [select_final_k](/Users/jy/Documents/HippoMaps/scripts/experiments/hipp_functional_parcellation/run_subject.py)

This rule explicitly favors:

- reproducibility,
- compactness,
- avoidance of very small parcels.

It does **not** claim to identify a uniquely correct biological number of hippocampal subregions.

## Step 11. Network Annotation

After clustering, the resulting hippocampal parcels are interpreted in cortical-network terms.

This is done by:

1. grouping cortical parcels by parent `Lynch2024` network,
2. averaging parcel-level FC within each parent network,
3. computing each hippocampal cluster's mean network profile,
4. assigning a dominant network label based on the strongest average profile.

Implementation:

- [group_fc_by_network](/Users/jy/Documents/HippoMaps/scripts/experiments/hipp_functional_parcellation/run_subject.py)

This means network labels are used as **annotations**, not as the primary segmentation mechanism.

## Arbitrary or Pragmatic Choices

Several choices in the current workflow are reasonable but not uniquely mandated by theory.

```text
Choice                               Current Standard             Why It Was Chosen                                               Why It Is Still Arbitrary
-----------------------------------  ---------------------------  ----------------------------------------------------------------  --------------------------------------------------------------
Cortical feature basis               Lynch2024 ROI components     Individualized, finer than parent networks, avoids fixed atlas  Another individualized basis could be used
Exclude Noise network               Yes                          Keeps feature space biologically interpretable                    Could be relaxed for a different QC philosophy
Hippocampal smoothing set           none / 1-ring / 4mm          Balances denoising and sensitivity analysis                       No universal best smoothing level exists
Main smoothing for reporting        1-ring                       Mild regularization without strong blurring                       Another study could choose none or 4mm
Affinity sparsity                   0.1                          Matches a sparse graph logic and practical stability              Different sparsity values could change embeddings
Number of gradients computed        5                            Close to the HippoMaps-style dimensionality reduction logic       Could compute fewer or more
Number of gradients clustered       3                            Good compromise between resolution and overfitting                2 or 4 may be equally defensible
Clustering family                   Ward agglomerative           Produces compact clusters                                          Spectral or other constrained methods are possible
Spatial constraint                  Surface adjacency            Enforces contiguous parcels                                        Could be relaxed or regularized differently
Left/right clustering               Separate, then merged        Avoids disconnected-graph artifacts                                A custom joint disconnected-graph method could be used
K evaluation range                  2..8                         Covers coarse to moderately fine solutions                         Wider ranges are possible
Final K shortlist                   3..6                         Excludes trivial K=2 and very fine high-K solutions               Another study could keep K=2 or extend above 6
Final K rule                        ARI-first, smallest stable   Favors conservative reproducible solutions                         Elbow-first or CH/DB-first rules are also possible
Cluster annotation                  Parent-network dominance     Clear biological summary after clustering                          Could use soft profiles instead of dominant labels
Probability display                 Row top-3 union             Compact visualization that preserves row-wise top networks        A full 21-network display is also valid
```

## Relation to HippoMaps

This workflow is inspired by the HippoMaps logic of:

- surface-based hippocampal representation,
- functional connectivity contextualization,
- diffusion-based dimensionality reduction,
- emphasis on continuous hippocampal organization before hard discretization.

However, the present parcellation workflow adds an explicit downstream clustering stage, because the target here is a discrete but not overly fragmented hippocampal functional parcellation.

## Practical Interpretation

The current standard workflow should be interpreted as:

- a principled way to derive hippocampal parcels from cortical connectivity structure,
- a clustering-based discretization of a continuous functional organization,
- a reproducible analysis recipe rather than a claim of a single uniquely true atlas.

The most important methodological idea is:

first discover the low-dimensional geometry of hippocampal connectivity, then discretize it conservatively, and only afterward assign cortical network labels for interpretation.

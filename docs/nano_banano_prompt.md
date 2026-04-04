# Nano Banano Prompt: Nature-Style Workflow Figure

Create a clean, publication-style scientific workflow figure in a visual style inspired by *Nature Methods* or *Nature Neuroscience* graphical summaries.

## Overall Goal

Draw a multi-panel workflow figure that explains a single-subject hippocampal functional parcellation pipeline.

The figure should communicate:

- the computational sequence from cortex features to hippocampal parcels,
- why diffusion gradients are used before clustering,
- which outputs are continuous versus discrete,
- which parameter choices are fixed standards versus pragmatic/arbitrary choices.

## Visual Style

Use a restrained, high-end scientific style:

- white background,
- minimal clutter,
- thin vector lines,
- subtle color palette,
- modern sans-serif typography,
- consistent spacing and alignment,
- no cartoon brain icons,
- no glossy effects,
- no childish colors,
- no decorative 3D rendering.

The result should feel like a real figure from a high-impact methods paper.

## Figure Layout

Use a horizontal or slightly top-to-bottom reading flow, whichever is cleanest.

The figure should have **five main stages** connected by arrows.

### Stage 1. Individualized Cortical Feature Space

Show:

- left and right cortex represented as individualized `Lynch2024 ROI components`,
- note that `Noise` components are excluded,
- output as a parcel-by-time matrix.

Suggested label:

`Individualized cortical ROI components`

Sub-label:

`Lynch2024 ROI components, Noise excluded`

## Stage 2. Hippocampal Surface Timeseries

Show:

- left and right hippocampal surfaces in `corobl` space,
- vertex-wise resting-state timeseries sampled on the hippocampal surface,
- three smoothing variants as a compact side note: `none`, `1-ring`, `4 mm`.

Suggested label:

`Hippocampal surface timeseries`

Sub-label:

`Vertex-wise rs-fMRI sampled on folded hippocampal surfaces`

## Stage 3. Vertex-to-Parcel FC Matrix

Show:

- a matrix with rows = hippocampal vertices and columns = cortical parcels,
- annotate the current example shape as:
  `12761 hippocampal vertices x 404 cortical parcels`

Suggested label:

`Vertex-to-parcel FC fingerprints`

Sub-label:

`Each hippocampal vertex is represented by its cortical connectivity profile`

## Stage 4. Diffusion Embedding

This is a key conceptual panel.

Show:

- hippocampal vertices transformed into a vertex similarity graph,
- then into a low-dimensional gradient space,
- highlight `Gradient 1`, `Gradient 2`, `Gradient 3`,
- visually communicate that this is a **continuous organization** stage, not yet a hard parcellation.

Suggested label:

`Diffusion-map embedding`

Sub-label:

`Continuous axes of hippocampal FC organization`

Add a small note:

`Compute 5 components; use Gradients 1-3 for clustering`

## Stage 5. Spatially Constrained Clustering

Show:

- clustering in gradient space,
- left and right hippocampus clustered separately with mesh adjacency constraints,
- final compact hippocampal parcels on the surface,
- then post hoc network annotation.

Suggested label:

`Spatially constrained clustering`

Sub-label:

`Ward agglomerative clustering in Gradient 1-3 space`

Add a second sub-label:

`Left/right hippocampus clustered separately, then merged`

Then show a final annotation box:

`Post hoc network annotation`

Sub-label:

`Dominant Lynch2024 parent network per cluster`

## Evaluation Sidebar

Add a slim side panel or bottom strip labeled:

`Model selection`

Include:

- `Evaluate K = 2-8`
- `Final shortlist: K = 3-6`
- `Metrics: ARI, silhouette, Calinski-Harabasz, Davies-Bouldin, WCSS, delta WCSS, min cluster fraction, BSS/TSS`
- `Current standard: choose the smallest K within 0.02 of the best ARI, with min cluster fraction >= 0.05`

This should read like a compact methods note, not a large central panel.

## Arbitrary Choices Sidebar

Add a second slim note panel titled:

`Pragmatic choices`

List concise bullets:

- cortical basis = individualized Lynch ROI components
- smoothing comparison = none / 1-ring / 4 mm
- graph sparsity = 0.1
- compute 5 gradients, cluster on first 3
- Ward clustering with surface adjacency
- network labels used for interpretation, not for primary segmentation

This panel should visually signal that these are implementation choices, not universal truths.

## Desired Scientific Message

The figure should make the following message immediately obvious:

1. The hippocampus is first represented continuously in a connectivity-gradient space.
2. Discrete functional parcels are derived afterward by conservative clustering.
3. Cortical network labels are used to interpret clusters, not to define them from the outset.

## Composition Notes

- Keep the main pipeline visually dominant.
- Use the evaluation and pragmatic-choice panels as secondary side information.
- Avoid overcrowding.
- Prefer fewer words inside the main pipeline boxes and keep longer text in small side panels.
- Make sure arrows and grouping clearly separate:
  - data representation,
  - dimensionality reduction,
  - clustering,
  - annotation.

## Output Requirement

Produce a figure that could plausibly serve as:

- Figure 1 for a methods-oriented hippocampal parcellation manuscript,
- a lab meeting summary slide,
- or a schematic overview for supplementary methods.


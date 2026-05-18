# Core Workflow Overview

This document is the authoritative workflow surface for active hippocampal processing.

## Scope

- Active modality regime: T1w-only processing surface
- Public execution layer: `scripts/core_workflow/`
- Numbering convention: `NN_MM_<semantic_name>.py`
- Global order is encoded by `NN`, independent of branch names

Branch mainlines:

- `kong2019`
- `kong2022_routea`
- `kong2022_routeb`

Within `kong2022`, RouteA and RouteB are treated as equal mainlines.

## Pipeline Stages

- `01_xx`: manifests, cohort and shared input preparation
- `02_xx`: HippUnfold run and output validation
- `03_xx`: cortex reference generation
- `04_xx`: hippocampal FC materialization
- `05_xx`: merged-canonical group atlas construction
- `06_xx`: label materialization, overview rendering, presentation outputs

Group atlas method families include intrinsic FC, network/ROI FC, WTA, and
`gradient-spectral`. The `gradient-spectral` branch is Kong2022-only: it
averages vertex-to-ROI FC at group level, computes diffusion gradients from that
group profile using BrainSpace (`GradientMaps`, `approach="dm"`,
`kernel="pearson"`), and clusters selected gradient columns. Embedding computes
`G1-G5` (`n_components=5`) with BrainSpace `sparsity=0.9` (keep top 10% per row),
while selected clustering defaults remain `G1,G2,G3`
(`--gradient-components 1,2,3`); single-gradient and two-gradient combinations
are available through the same option. In maintained presentation outputs,
`gradient-spectral` is treated as a full-continuous FC route-level analysis
rather than a `10perc/full` mode-split branch.

The separate HippoMaps-style route is `05_03_compute_hippomaps_gradients.py`
followed by `06_05_render_hippomaps_gradient_products.py`. It keeps the same
upstream Fisher-z FC preprocessing but does not alter clustering. `05_03`
imports official package code from `external/hippomaps`, calls
`brainspace.gradient.GradientMaps(n_components=N, kernel="pearson")` on the
column-wise z-scored group mean Fisher-z vertex-to-Kong2022 ROI FC matrix, and
saves continuous full-surface gradients with NaNs outside the support mask, the
full lambda vector, EV fractions, and the group mean FC. The default computes
five components, matching the HippoMaps paper's maximum component count, while
presentation shows G1-G3. BrainSpace default sparsity is retained for this
within-method rsfMRI-style reduction. `06_05` uses
`hippomaps.plotting.surfplot_canonical_foldunfold` with `cmap="plasma"`,
`tighten_cwindow=True`, `share="row"`, and `color_bar="right"` for
folded/unfolded hippocampal gradient figures, plus EV and top-25% minus
bottom-25% gradient endpoint FC contrast plots. For `den=512`, `06_05` treats
`512` as HippoMaps v2 in the current Python process and disables
`unfoldAPrescale`; no AP surfarea rescale is used for 512-density figures.
Manifests record `external/hippomaps` import path, commit, and plotter support
settings.
The renderer defaults to bilateral output but accepts `--hemis L` or `--hemis R`
for single-hemisphere figures and writes hemisphere-specific output filenames.

`scripts/core_workflow/` is both:
- canonical public interface (`NN_MM_*`)
- active implementation home (`scripts/core_workflow/lib/*`)

Active numbered steps no longer shell out to the legacy hippocampal/cortex runtime directories.
The active runtime boundary allows external assets and the Workbench executable
`software/workbench/bin_linux64/wb_command`, but not external helper code scripts.

## Traceability Contract

Each entrypoint can emit a standardized step manifest via `--step-manifest-out` with:

- `step_id`
- `workflow_branch`
- `upstream_step_ids`
- `input_artifacts`
- `output_artifacts`
- `command`
- `git_commit`

Use these manifests to reconstruct exact lineage for each generated artifact set.

## References

- Step matrix: `docs/hipp_parcellation_network/core_workflow_step_registry.md`
- Machine-readable registry: `scripts/core_workflow/step_registry.json`
- Branch semantics: `docs/hipp_parcellation_network/multi_branch_flow.md`
- Ordered procedure: `docs/hipp_parcellation_network/step_by_step_flows.md`
- Archive index: `docs/hipp_parcellation_network/archive_index.md`

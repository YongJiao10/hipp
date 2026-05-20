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
`gradient-spectral`. The maintained presentation policy is fixed final `K=3`;
`05_01_run_group_atlas.py` defaults to `--k-min 3 --k-max 3
--force-final-k 3`. The legacy automatic K selector is still available through
`--auto-k` for diagnostics but is no longer the default display policy. Partial
correlation spectral branches are retired from public method choices.

`gradient-spectral` is Kong2022-only. It consumes HippoMaps/BrainSpace gradient
artifacts written by `05_03_compute_hippomaps_gradients.py` under
`--gradient-root`, then clusters selected columns. `05_03` computes G1-G5 from
the column-wise z-scored group mean Fisher-z vertex-to-Kong2022 ROI FC matrix
using `GradientMaps(n_components=N, kernel="pearson")`; `05_01` selects G1-G3
by default (`--gradient-components 1,2,3`).

`06_05_render_hippomaps_gradient_products.py` renders the same continuous
gradient artifacts with `hippomaps.plotting.surfplot_canonical_foldunfold`,
`cmap="plasma"`, `tighten_cwindow=True`, `share="row"`, and
`color_bar="right"`. For `den=512`, it treats `512` as HippoMaps v2 in the
current Python process and disables `unfoldAPrescale`; no AP surfarea rescale is
used for 512-density figures. The renderer defaults to bilateral output and
accepts `--hemis L` or `--hemis R` for single-hemisphere files.

Canonical run outputs live under
`outputs_core/hcp7t_hippocampus/runs/<run_id>/`. All maintained method overview
PNGs are written to `present/overviews/` under that run root. Supplementary
continuous-gradient products live in `present/gradients/`, cortex panels live in
`present/cortex_panels/`, and runtime/manifests remain under the same run root.
Directory names intentionally do not encode `t1only`, `present_fixedk`, or a
`10perc/full` presentation split.

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

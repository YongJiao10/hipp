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
- `kong2022_routea_17net`
- `kong2022_routeb`

Within `kong2022`, RouteA and RouteB are treated as equal mainlines; `kong2022_routea_17net` is a shallow RouteA taxonomy variant that reuses RouteA inputs and gradients.

## Pipeline Stages

- `01_xx`: manifests, cohort and shared input preparation
- `02_xx`: HippUnfold run and output validation
- `03_xx`: cortex reference generation
- `04_xx`: hippocampal FC materialization
- `05_xx`: merged-canonical group atlas construction
- `06_xx`: label materialization, overview rendering, presentation outputs

Group atlas method families include intrinsic FC, network/ROI FC, WTA, and
`gradient-spectral`. The maintained presentation policy is fixed final `K=3`;
`05_02_run_group_atlas.py` defaults to evaluating `K=2..10` while using `--force-final-k 3`. The legacy automatic K selector is still available through
`--auto-k` for diagnostics but is no longer the default display policy. Partial
correlation spectral branches are retired from public method choices.

`gradient-spectral` is Kong2022-only. It consumes aligned HippoMaps/BrainSpace
gradient artifacts written by `05_01_compute_hippomaps_gradients.py` under
`--gradient-root`, then clusters selected columns. `05_01` computes raw G1-G5
from the column-wise z-scored group mean Fisher-z vertex-to-Kong2022 ROI FC
matrix using `GradientMaps(n_components=N, kernel="pearson")`, aligns component
order and sign within each route to the `4mm/hemi_L` anchor when available, and
writes only the aligned arrays. `05_02` selects aligned G1-G3 by default
(`--gradient-components 1,2,3`).

`06_04_render_hippomaps_gradient_products.py` renders the same continuous
gradient artifacts with `hippomaps.plotting.surfplot_canonical_foldunfold`,
`cmap="plasma"`, `tighten_cwindow=True`, `share="row"`, and
`color_bar="right"`. For `den=512`, it treats `512` as HippoMaps v2 in the
current Python process and disables `unfoldAPrescale`; no AP surfarea rescale is
used for 512-density figures. The renderer defaults to bilateral output and accepts `--hemis L` or
`--hemis R` for single-hemisphere files. Although left and right hemispheres are
fit independently upstream, `05_01` writes aligned arrays before `06_04` renders
them. `06_04` then applies a display-only cross-route alignment by default:
`routea` is the reference, each selected route/hemi/component is sign-flipped for
rendering when its Pearson correlation to the matching `routea` component is
negative, and the saved `05_01` arrays remain unchanged. Continuous-gradient
figures also use shared per-smoothing, per-component color ranges across the
rendered routes so RouteA, cortex-regression, cortex-drop, and RouteB panels are
visually comparable for paper figures. Bilateral figures are still assembled
from separate L/R HippoMaps renders so each side keeps its own EV labels.
Manifests record the `external/hippomaps` import path, commit, plotter support
settings, upstream gradient alignment provenance, display flip decisions, and
shared color ranges.

Canonical presentation outputs live under `outputs_core/present/`. All maintained
method overview PNGs are written to `outputs_core/present/overviews/`.
Supplementary continuous-gradient products live in `outputs_core/present/gradients/`,
and cortex panels live in `outputs_core/present/cortex_panels/`. Smoke tests stay
isolated under `outputs_core/smoke_runs/<run_id>/`.
Directory names intentionally do not encode `t1only`, `present_fixedk`, or a
`10perc/full` presentation split.

`scripts/core_workflow/` is both:
- canonical public interface (`NN_MM_*`)
- active implementation home (`scripts/core_workflow/lib/*`)

RouteA reference construction has an explicit upstream label-generation dependency. The artifact-level contract for generated RouteA ArealMSHBM labels, local numbered-step products, presentation products, and required evidence is maintained in `docs/hipp_parcellation_network/core_workflow_artifact_lineage.md`.

The `kong2022_routea_17net` branch uses the original Kong2022 17 network families for cluster annotation and network profiles. It reuses `shared_inputs/routea_151`, `surface_151`, 400-ROI FC caches, and RouteA gradients, then writes only taxonomy-dependent group outputs under `outputs_core/work/group_atlas/routea_17net_full_k3_fisherz/`. Run `05_02_run_group_atlas.py` with `--route-tag routea_17net --network-taxonomy family17`; do not run `05_01` for a separate `routea_17net` gradient tree.

Network display colors are centralized in `config/hipp_network_style.json`. Group-atlas label materialization, hippocampal overviews, WTA long-axis profiles, and cortex panels load network colors from that single file; `config/cross_atlas_network_merge.json` is mapping-only and must not carry color overrides. Missing colors are treated as errors, not gray/blue fallback display colors. For the RouteA 17-net taxonomy, same-family A/B/C subnetworks use deliberately separated lightness and saturation levels so overview bars, profiles, and label-table colors remain distinguishable while preserving the parent family hue.

The `kong2022_routea_ctxreg` branch is an isolated RouteA derivative: `04_00_materialize_cortex_regressed_surface_timeseries.py` writes `shared_inputs/routea_ctxreg_151` with run-wise local cortex residualization before `04_01` and `04_02` materialize FC. It does not overwrite uncontrolled RouteA roots.

The `kong2022_routea_ctxdrop` branch is the matched RouteA local-cortex exclusion control. `04_01_materialize_kong2022_roi_fc.py --ctxdrop` keeps hippocampal time series unchanged, excludes valid cortex vertices within 20 mm separately for each hippocampal vertex before ROI averaging, and writes isolated FC caches plus smoothing-independent local cortex-neighbor diagnostics under `shared_inputs/routea_ctxdrop_151/sub-*/surface/local_cortex_neighbors/hemi_*/`. Its maintained downstream scope is `gradient-spectral` and `wta-top`, with group outputs in `outputs_core/work/group_atlas/routea_ctxdrop_full_k3_fisherz/`. The shared local-cortex-neighbor count surface overlay for ctxreg/ctxdrop is rendered by `06_07_render_ctxdrop_neighbor_counts.py` under `outputs_core/present/local_cortex_neighbors/`.

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
- Artifact lineage: `docs/hipp_parcellation_network/core_workflow_artifact_lineage.md`
- Machine-readable registry: `scripts/core_workflow/step_registry.json`
- Branch semantics: `docs/hipp_parcellation_network/multi_branch_flow.md`
- Ordered procedure: `docs/hipp_parcellation_network/step_by_step_flows.md`
- Archive index: `docs/hipp_parcellation_network/archive_index.md`

The WTA-top overview generated by `06_01_build_group_overview.py` embeds a fixed 20-bin long-axis network-composition profile between the compact vertex-assignment count panel and the hippocampal surface render. `06_02_build_wta_long_axis_profiles.py` and standalone `kong2022_wta-top_long_axis_network_profile_*` PNG/TSV products have been removed from the maintained workflow; `07_01_build_group_matrix.py` indexes the embedded overview only.

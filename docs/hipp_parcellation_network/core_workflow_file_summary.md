# Core Workflow Source File Index

This is a quick index of active source files under `scripts/core_workflow/`.
It is not an IO, artifact, or provenance contract. Use these canonical sources
for those details:

- Step and branch IO matrix: `core_workflow_step_registry.md`
- Artifact lineage and evidence requirements: `core_workflow_artifact_lineage.md`
- Machine-readable step registry: `scripts/core_workflow/step_registry.json`

## Entrypoints And Registry

| File | Purpose |
|---|---|
| `scripts/core_workflow/README.md` | Public workflow surface, runtime boundary, and implementation layout notes |
| `scripts/core_workflow/step_registry.json` | Machine-readable step IDs, branches, and declared IO |
| `tests/core_workflow/smoke_prepare_run.py` | Isolated smoke-run command-plan utility |

## Numbered Workflow Steps

| File | Purpose |
|---|---|
| `scripts/core_workflow/01_01_run_hippunfold_t1w_local.py` | Run local T1w HippUnfold workflow |
| `scripts/core_workflow/01_02_check_hippunfold_outputs.py` | Validate the full HippUnfold output contract, then receipt and trim `anat/`, `coords/`, `warps/`, and `cifti/` while retaining core-consumed surface, metric, and QC assets |
| `scripts/core_workflow/02_01_prepare_group_inputs.py` | Prepare shared surfaces, signal-validity metadata, and group templates |
| `scripts/core_workflow/02_02_build_qc_manifest.py` | Build QC-pass and excluded-subject manifests |
| `scripts/core_workflow/03_01_build_kong2022_reference.py` | Build shared Kong2022 cortex references |
| `scripts/core_workflow/03_02_build_kong2022_routea_reference.py` | Build RouteA cortex references and compact 17-net summaries |
| `scripts/core_workflow/04_00_materialize_cortex_regressed_surface_timeseries.py` | Build isolated RouteA cortex-regressed hippocampal time series |
| `scripts/core_workflow/04_01_materialize_kong2022_roi_fc.py` | Materialize Kong2022 vertex-to-ROI FC, including ctxdrop mode |
| `scripts/core_workflow/04_02_materialize_intrinsic_fc.py` | Materialize intrinsic hippocampal FC |
| `scripts/core_workflow/05_01_compute_hippomaps_gradients.py` | Compute aligned HippoMaps/BrainSpace gradient artifacts |
| `scripts/core_workflow/05_02_run_group_atlas.py` | Build group atlas outputs and K diagnostics |
| `scripts/core_workflow/05_03_run_spectral_fast_rerun.py` | Optional rerun orchestration after primary group-atlas construction |
| `scripts/core_workflow/05_04_compute_confidence_maps.py` | Compute vertex-level gradient-parcellation confidence maps (Euclidean silhouette for Ward/GMM, cosine for spectral); requested combos fail closed with a failure list and non-zero exit |
| `scripts/core_workflow/lib/formal_cohorts.py` | Shared exact-size, canonical-syntax, unique-subject contracts for formal HCP, Lynch, and Stanford cohorts |
| `scripts/core_workflow/06_01_build_group_overview.py` | Render per-method group overview figures |
| `scripts/core_workflow/06_02_materialize_group_labels.py` | Materialize functional-atlas label arrays |
| `scripts/core_workflow/06_03_build_cluster_cortex_panels.py` | Render cluster-based cortex and hippocampal significance panels |
| `scripts/core_workflow/06_10_render_cluster_roi_strength_panels.py` | Render gradient-spectral and normalized WTA full 400-ROI cortex templates |
| `scripts/core_workflow/06_04_render_hippomaps_gradient_products.py` | Render continuous-gradient products |
| `scripts/core_workflow/06_05_render_group_gradient_heatmaps.py` | Deprecated heatmap entrypoint that exits with guidance |
| `scripts/core_workflow/06_06_render_confidence_maps.py` | Render confidence maps |
| `scripts/core_workflow/06_07_render_ctxdrop_neighbor_counts.py` | Render ctxreg/ctxdrop local cortex-neighbor overlays |
| `scripts/core_workflow/06_08_render_hcp_subfield_composition.py` | Render HCP subfield composition plots |
| `scripts/core_workflow/06_09_plot_cortex_kong2022_mshbm.py` | Render subject-level Kong2022 MSHBM cortical parcellation brain maps |
| `scripts/core_workflow/07_01_build_group_matrix.py` | Build the maintained presentation matrix/index |

## Shared Library Modules

| File | Purpose |
|---|---|
| `scripts/core_workflow/lib/branch_config.py` | Branch names, aliases, and explicit standard/ctxreg signal-branch mapping |
| `scripts/core_workflow/lib/cortex_fc_tiles.py` | Cortical FC tile (four-view inflated cortex plus top-10 ROI bars) shared by the group overview and the multi-dataset figure |
| `scripts/core_workflow/lib/cortex_template_profiles.py` | Shared subjectwise WTA pseudo-cluster and 400-ROI template calculations |
| `scripts/core_workflow/lib/extract_cortex_kong2022_routeb_roi_timeseries.py` | RouteB Kong2022 cortex ROI time-series extraction from the public dlabel |
| `scripts/core_workflow/lib/extract_cortex_kong2022_route_a_roi_timeseries.py` | RouteA flat label loading and ROI time-series extraction |
| `scripts/core_workflow/lib/extract_cortex_merged_network_timeseries.py` | Network-level cortex time-series aggregation and shared RouteA/RouteB ROI aggregation |
| `scripts/core_workflow/lib/fc_math.py` | Shared FC numeric helpers |
| `scripts/core_workflow/lib/gifti_labels.py` | GIFTI label construction helpers |
| `scripts/core_workflow/lib/hipp_density_assets.py` | Hippocampal density asset resolution |
| `scripts/core_workflow/lib/hippomaps_adapter.py` | Adapter for external HippoMaps functionality |
| `scripts/core_workflow/lib/io_utils.py` | Shared filesystem and array IO helpers |
| `scripts/core_workflow/lib/kong2022_label_utils.py` | Kong2022 label and network mapping helpers |
| `scripts/core_workflow/lib/output_layout.py` | Standardized output path layout helpers |
| `scripts/core_workflow/lib/render_cortex_pfm_scene.py` | Cortex/PFM scene rendering helpers |
| `scripts/core_workflow/lib/surface_sampling.py` | Surface sampling helpers |
| `scripts/core_workflow/lib/surface_utils.py` | Surface geometry helpers |
| `scripts/core_workflow/lib/traceability.py` | Standardized step-manifest helpers |
| `scripts/core_workflow/lib/workbench.py` | Workbench command wrapper and thread controls |
| `scripts/core_workflow/lib/workbench_grid.py` | Workbench grid/batch helpers |
| `scripts/core_workflow/lib/workbench_scene_batch.py` | Workbench scene batch helpers |
| `scripts/core_workflow/lib/workflow_utils.py` | Runtime metadata, JSON, and general workflow utilities |

# Locked Workbench Rendering

This project's official hippocampal figure workflow is now fixed to a structural Workbench render plus a formal hippocampal gradient render.

For active core-workflow RouteA gradient presentation, use:

- `scripts/core_workflow/06_06_render_group_gradient_heatmaps.py`
- scene template: `config/wb_locked_native_view_lateral_medial.scene`
- active input root: `outputs_core/inputs/hcp7t_hippocampus/kong2022/hipp_group_merged_canonical_atlas_routea_full`

## Official Inputs

- Hippocampal overview renderer: `scripts/core_workflow/06_01_build_group_overview.py`
- Cortex panel renderer: `scripts/core_workflow/06_04_build_cluster_cortex_panels.py`
- Continuous HippoMaps gradient renderer: `scripts/core_workflow/06_05_render_hippomaps_gradient_products.py`
- Scene templates: `config/wb_locked_native_view_lateral_medial.scene` and `config/manual_wb_scenes/cortex_manual.scene`
- Workbench executable: `software/workbench/bin_linux64/wb_command` through `scripts/core_workflow/lib/workbench.py`

## Locked Visual Rules

- Use the saved Workbench native/folded camera in `config/wb_locked_native_view.scene`.
- Keep the left/right hippocampi in the mirrored symmetric layout produced by that scene.
- Do not place text labels on the surface itself.
- Place structural region names in a right-side legend panel.
- Use the enlarged legend style defined in `_archive/deprecated_2026-05-20/scripts_pre_core_workflow/workbench/compose_wb_with_side_legend.py`.

## Official Output Pattern

For each subject, the official output files are:

- `sub-<id>_structural.png`
- `sub-<id>_gradient.png`

These should be written under a stable output folder such as:

- `outputs/dense_corobl_batch/final_wb_locked/sub-<id>/`

## Archive Policy

Older exploratory rendering code and exploratory rendering outputs are archived and should not be treated as the official figure path.

- Legacy code archive: `archive/legacy_rendering_code/`
- Legacy result archive: `outputs/dense_corobl_batch/_archive/legacy_rendering_results/`

If the rendering needs to be rerun in the future, rerun the active `scripts/core_workflow/06_*` render entrypoints and write into `outputs_core/present/`.

## RouteA Group Gradient Heatmap Output

The maintained continuous gradient figure artifact is:

- `outputs_core/present/gradients/kong2022_gradient-spectral_group_gradients_heatmap_routea_2mm4mm_g1g2g3.png`
- manifest: `outputs_core/present/gradients/kong2022_gradient-spectral_group_gradients_heatmap_routea_2mm4mm_g1g2g3_manifest.json`
- runtime scalar/render assets: `outputs_core/present/gradients/group_gradient_heatmaps/runtime/routea_full/`

This output is generated from active `gradient-spectral` RouteA full outputs only
(`2mm` and `4mm`, `G1-G3`) and uses symmetric per-component min/max scaling across
both smoothings and both hemispheres. The renderer uses a single Workbench-native
palette source (`palette_engine=workbench`) with default `palette=magma`.
`plasma` is not a default and is not auto-mapped. Surface colors and bottom
colorbars are both rendered by Workbench from the same metric palette settings;
Pillow is used only for panel composition/cropping/text layout. Each cell is
labeled `G# EV=xx.x%`, where EV is the mean left/right share of that component's
eigenvalue among exported BrainSpace `G1-G5` eigenvalues for that smoothing.

# Locked Workbench Rendering

This project's official hippocampal figure workflow is now fixed to a structural Workbench render plus a formal hippocampal gradient render.

For active core-workflow RouteA gradient presentation, use:

- `scripts/core_workflow/06_04_render_hippomaps_gradient_products.py`
- scene template: `config/wb_locked_native_view_lateral_medial.scene`
- active input root: `outputs_hcp/inputs/hcp/kong2022/hipp_group_merged_canonical_atlas_routea_full`

## Official Inputs

- Hippocampal overview renderer: `scripts/core_workflow/06_01_build_group_overview.py`
- Cortex panel renderer: `scripts/core_workflow/06_03_build_cluster_cortex_panels.py`
- Continuous HippoMaps gradient renderer: `scripts/core_workflow/06_04_render_hippomaps_gradient_products.py`
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

If the rendering needs to be rerun in the future, rerun the active `scripts/core_workflow/06_*` render entrypoints and write into `outputs_hcp/present/`.

## Retired Legacy Heatmap Output

The legacy locked-scene heatmap artifact has been retired and removed.

Do not use `06_05_render_group_gradient_heatmaps.py` for active outputs.
Use `06_04_render_hippomaps_gradient_products.py` and consume:

- `outputs_hcp/present/gradients/routea/{2mm,4mm}/`
- `outputs_hcp/present/gradients/routeb/{2mm,4mm}/`

These are the maintained final gradient products (G1-G3 figure, EV plot, endpoint FC contrast, and manifest).

# Locked Workbench Rendering

This project's official hippocampal figure workflow is now fixed to a structural Workbench render plus a formal hippocampal gradient render.

For active core-workflow RouteA gradient presentation, use:

- `scripts/core_workflow/06_05_render_group_gradient_heatmaps.py`
- scene template: `config/wb_locked_native_view_lateral_medial.scene`
- active input root: `outputs_experiments/hippunfold_recommended_t1w_20260423/kong2022/hipp_group_merged_canonical_atlas_routea_full`

## Official Inputs

- Scene template: `config/wb_locked_native_view.scene`
- Batch renderer: `scripts/workbench/render_locked_wb_views.py`
- Workbench scene capture: `scripts/workbench/render_wb_scene_batch.py`
- Right-side legend compositor: `scripts/workbench/compose_wb_with_side_legend.py`

## Locked Visual Rules

- Use the saved Workbench native/folded camera in `config/wb_locked_native_view.scene`.
- Keep the left/right hippocampi in the mirrored symmetric layout produced by that scene.
- Do not place text labels on the surface itself.
- Place structural region names in a right-side legend panel.
- Use the enlarged legend style defined in `scripts/workbench/compose_wb_with_side_legend.py`.

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

If the rendering needs to be rerun in the future, rerun `scripts/workbench/render_locked_wb_views.py` rather than reviving older exploratory scripts.

## RouteA Group Gradient Heatmap Output

The maintained continuous gradient figure artifact is:

- `present_group_t1only/kong2022_gradient-spectral_group_gradients_heatmap_routea_2mm4mm_g1g2g3.png`
- manifest: `present_group_t1only/kong2022_gradient-spectral_group_gradients_heatmap_routea_2mm4mm_g1g2g3_manifest.json`
- runtime scalar/render assets: `present_group_t1only/group_gradient_heatmaps/runtime/routea_full/`

This output is generated from active `gradient-spectral` RouteA full outputs only
(`2mm` and `4mm`, `G1-G3`) and uses symmetric per-component min/max scaling across
both smoothings and both hemispheres. The heatmap renderer uses a `plasma`
palette and labels each smoothing/component cell as `G# EV=xx.x%`, where EV is
the mean left/right share of that component's non-trivial diffusion eigenvalue
among the exported non-trivial eigenvalues for that smoothing.

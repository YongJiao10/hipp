# Locked Workbench Rendering

This project's official hippocampal figure workflow is now fixed to a single Workbench-based rendering pipeline.

## Official Inputs

- Scene template: `config/wb_locked_native_view.scene`
- Batch renderer: `scripts/render_locked_wb_views.py`
- Workbench scene capture: `scripts/render_wb_scene_batch.py`
- WTA label asset preparation: `scripts/prepare_wta_workbench_assets.py`
- Right-side legend compositor: `scripts/compose_wb_with_side_legend.py`

## Locked Visual Rules

- Use the saved Workbench native/folded camera in `config/wb_locked_native_view.scene`.
- Keep the left/right hippocampi in the mirrored symmetric layout produced by that scene.
- Do not place text labels on the surface itself.
- Place region/network names in a right-side legend panel.
- Use the enlarged legend style defined in `scripts/compose_wb_with_side_legend.py`.

## Official Output Pattern

For each subject, the official output files are:

- `sub-<id>_structural.png`
- `sub-<id>_wta.png`

These should be written under a stable output folder such as:

- `outputs/dense_corobl_batch/final_wb_locked/sub-<id>/`

## Archive Policy

Older exploratory rendering code and exploratory rendering outputs are archived and should not be treated as the official figure path.

- Legacy code archive: `archive/legacy_rendering_code/`
- Legacy result archive: `outputs/dense_corobl_batch/_archive/legacy_rendering_results/`

If the rendering needs to be rerun in the future, rerun `scripts/render_locked_wb_views.py` rather than reviving older exploratory scripts.

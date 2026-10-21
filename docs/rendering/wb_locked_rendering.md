# Locked Workbench Rendering

This project's official hippocampal figure workflow is now fixed to a structural Workbench render plus a formal hippocampal gradient render.

For active core-workflow RouteA gradient presentation, use:

- `scripts/core_workflow/06_04_render_hippomaps_gradient_products.py`
- scene template: `config/wb_locked_native_view_lateral_medial.scene`
- active input root: `outputs_hcp/inputs/kong2022/hipp_group_merged_canonical_atlas_routea_full`

## Gradient Assignment Contract

Gradient surface panels must validate the exact assignment variant declared by
the hemisphere manifest through
scripts/core_workflow/lib/gradient_assignment_contract.py. The validator checks
the status JSON, functional-label NPY, label map, Workbench GIFTI, and, when
applicable, the declared Fisher-z profile as one product. It also checks that
the GIFTI label table uses the canonical network colors and that every cluster
winner agrees with the surface labels. A missing, ambiguous,
cross-directory, malformed, or disagreeing artifact fails closed; no
competing filename or older cache is selected implicitly.

Signal-branch path resolution is additive: the historical four-cell
2mm/4mm by 36P/22P registry remains unchanged, while ctxreg is available only
as an explicit canonical 2mm/36P branch. Every GridSpec has an explicit
signal_branch value, with standard as the legacy default. Standard paths are
never replaced with ctxreg paths because a branch directory happens to exist.

## Official Inputs

For the maintained multi-dataset canonical7 grid, the unqualified `gradient`
group-atlas result resolves to
`outputs_hcp/work/group_atlas/kong2022_7net_full_k3_fisherz/gradient-spectral/kong2022/`.
The former `routea_full_k3_fisherz/gradient-spectral/kong2022/` canonical8
subtree is archived under `wrong/outputs_hcp/work/archive/8net_20260909/`; it is not an
active default. Retired weighted-fisherz/relu-cosine products are archival
provenance, not active defaults.

Gradient overview manifests preserve `surface_affinity=shifted-cosine`, the
exact `affinity_description`, and `source_group_manifests`. Canonical PNG
filenames remain stable, while the visible title identifies shifted-cosine.
Formal release-audit and refresh maintenance helpers were retired from this
checkout. Presentation lineage is recorded in the generated manifests and
checked through the active visualization readiness commands; do not invoke or
restore deleted maintenance paths. Archive movement remains gated on an exact
inventory and explicit user confirmation.

- Hippocampal overview renderer: `scripts/core_workflow/06_01_build_group_overview.py`
- Cluster significance panel renderer: `scripts/core_workflow/06_03_build_cluster_cortex_panels.py`
- Full 400-ROI template renderer (gradient-spectral and normalized WTA): `scripts/core_workflow/06_10_render_cluster_roi_strength_panels.py`
- Continuous HippoMaps gradient renderer: `scripts/core_workflow/06_04_render_hippomaps_gradient_products.py`
- HCP 151 cluster/subfield composition renderer: `scripts/core_workflow/06_08_render_hcp_subfield_composition.py`; its default group root is the formal RouteA shifted-cosine branch. It outputs PNG, subject-long TSV, summary TSV, and manifest under `outputs_hcp/present/subfield_composition/`, computing area-weighted subject-level cluster/subfield proportions from HippUnfold `space-corobl_den-512_label-hipp_midthickness.surf.gii` surfaces.
- Scene templates: `config/wb_locked_native_view_lateral_medial.scene` and `config/manual_wb_scenes/cortex_manual.scene`
- Workbench executable: `software/workbench/bin_linux64/wb_command` through `scripts/core_workflow/lib/workbench.py`

## Locked Visual Rules

- Use the saved Workbench native/folded camera in `config/wb_locked_native_view.scene`.
- Keep the left/right hippocampi in the mirrored symmetric layout produced by that scene.
- Default to pure white background (`[255, 255, 255]`) and dark foreground (`[0, 0, 0]`) in both scene templates (`config/wb_locked_native_view_lateral_medial.scene` and `config/manual_wb_scenes/cortex_manual.scene`).
- Downstream compositing via `scripts/core_workflow/lib/workbench_grid.py` defaults to pure white canvas background (`(255, 255, 255, 255)`), black text, and adaptive white/black boundary trimming.
- Keep the `06_01_build_group_overview.py` cluster-confidence panel on the RouteA reference layout: use the top-1 Fisher-z range with the `1.22` headroom factor and the default lower-right `1st`/`2nd` legend; this rule is independent of the white Workbench background.
- Do not place text labels on the surface itself.
- Place structural region names in a right-side legend panel.
- Use the enlarged legend style defined in `_archive/deprecated_2026-05-20/scripts_pre_core_workflow/workbench/compose_wb_with_side_legend.py`.

## Official Output Pattern

For each subject, the official output files are:

- `sub-<id>_structural.png`
- `sub-<id>_gradient.png`

These should be written under a stable output folder such as:

- `outputs/dense_corobl_batch/final_wb_locked/sub-<id>/`

When a downstream panel needs an auditable subject-specific scene, call
render_scene_batch with no_template_scene=True and retain_scene=True. This
keeps the replaced scene at sub-<id>/sub-<id>_template.scene while preserving
the default ephemeral temporary-scene behavior when retain_scene is omitted.

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

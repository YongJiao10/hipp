# Hipp Parcellation Multi-Branch Flow (Canonical)

This document defines branch semantics for the active core workflow surface.

## Branch Mainlines

- `kong2019`
- `kong2022_routea`
- `kong2022_routeb`

`kong2019` and `kong2022` are treated as parallel mainlines.
Within `kong2022`, RouteA and RouteB are parallel mainlines.

## Pipeline Order

Use numbered public entrypoints in `scripts/core_workflow/`:

- `01_xx` inputs and cohort manifests
- `02_xx` HippUnfold run and validation
- `03_xx` cortex reference generation
- `04_xx` hippocampal FC materialization
- `05_xx` group atlas construction
- `06_xx` label export and presentation outputs

## Branch Applicability

- Kong2019 branch:
  - shared input root uses Kong2019 FC banks (`04_03_materialize_kong2019_fc`)
  - uses `05_01_run_group_atlas` for group outputs
- Kong2022 RouteA branch:
  - may include RouteA reference build (`03_02_build_kong2022_routea_reference`)
  - uses `04_01`, `04_02`, `05_01`, optional `05_02`
  - may run `gradient-spectral` in `05_01` from group-averaged Kong2022 vertex-to-ROI FC
- Kong2022 RouteB branch:
  - uses shared Kong2022 reference build (`03_01`) and standard FC stages
  - uses `05_01`, optional `05_02`
  - may run `gradient-spectral` in `05_01` from group-averaged Kong2022 vertex-to-ROI FC

## Gradient-Spectral Branch

`gradient-spectral` is a Kong2022-only group atlas method. It consumes the same
vertex-to-ROI FC caches as the Kong2022 network/WTA branches, averages those FC
profiles across the active cohort at group level, computes diffusion gradients,
and clusters selected gradient columns. The default selected set is `G1,G2,G3`;
valid alternatives are `G1`, `G2`, `G3`, and `G1,G2`. Because the embedding is
derived from group-averaged continuous FC profiles, this branch is maintained as
a full-FC route-level analysis. The `top-percent` / `10perc` sparsified and
binarized FC mode used for some discrete clustering comparisons is not a
maintained gradient branch.

For RouteA full outputs, optional presentation rendering may export the locked
2x3 continuous heatmap figure (`2mm`/`4mm` x `G1`/`G2`/`G3`) via
`06_05_render_group_gradient_heatmaps.py`, producing:
`present_group_t1only/kong2022_gradient-spectral_group_gradients_heatmap_routea_2mm4mm_g1g2g3.png`
with companion manifest JSON. Each heatmap cell is rendered with a `plasma`
palette and labeled `G# EV=xx.x%`; EV is the left/right-averaged share of that
component's non-trivial diffusion eigenvalue among the exported non-trivial
eigenvalues for the smoothing.

## Default Cohort Reminder

Default analysis cohort remains the QC-passed 151-subject set.
See `fmri_qc.md` for manifest paths and exclusion rationale.

## Canonical References

- Workflow overview: `core_workflow_overview.md`
- Step/IO matrix: `core_workflow_step_registry.md`
- Ordered run flow: `step_by_step_flows.md`
- Machine registry: `scripts/core_workflow/step_registry.json`

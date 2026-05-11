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
- Kong2022 RouteB branch:
  - uses shared Kong2022 reference build (`03_01`) and standard FC stages
  - uses `05_01`, optional `05_02`

## Default Cohort Reminder

Default analysis cohort remains the QC-passed 151-subject set.
See `fmri_qc.md` for manifest paths and exclusion rationale.

## Canonical References

- Workflow overview: `core_workflow_overview.md`
- Step/IO matrix: `core_workflow_step_registry.md`
- Ordered run flow: `step_by_step_flows.md`
- Machine registry: `scripts/core_workflow/step_registry.json`

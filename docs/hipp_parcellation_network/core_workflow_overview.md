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

Implementation scripts remain under `scripts/hipp_parcellation_network/` and `scripts/cortex/`.
`scripts/core_workflow/` is the canonical public interface.

## Traceability Contract

Each wrapper can emit a standardized step manifest via `--step-manifest-out` with:

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

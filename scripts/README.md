# Scripts Layout

This directory is organized by layer.

- `core_workflow/`: canonical public entrypoints (`NN_MM_<semantic_name>.py`)
- `hipp_parcellation_network/`: implementation layer for hippocampal analysis scripts
- `cortex/`: cortex reference derivation and extraction
- `common/`: shared numerical helpers
- `workbench/`: rendering and figure assembly helpers

The active workflow should be launched from `scripts/core_workflow/` rather than directly from implementation scripts.

Traceability contract for `core_workflow` wrappers:

- supports `--workflow-branch` with `kong2019`, `kong2022_routea`, `kong2022_routeb`
- supports repeatable `--upstream-step-id`, `--input-artifact`, `--output-artifact`
- supports `--step-manifest-out` emitting standardized JSON containing:
  - `step_id`
  - `workflow_branch`
  - `upstream_step_ids`
  - explicit input/output artifacts
  - command and git commit

Step inventory is in `scripts/core_workflow/step_registry.json`.
Legacy or retired surfaces are preserved under `_archive/`.

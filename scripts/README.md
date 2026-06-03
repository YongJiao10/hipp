# Scripts Layout

This directory is organized by layer.

- `core_workflow/`: canonical active workflow (`NN_MM_<semantic_name>.py`) and local implementation modules (`core_workflow/lib/`)
- `hipp_parcellation_network/`: legacy/auxiliary hippocampal scripts (active core entrypoints are archived)
- `cortex/`: legacy/auxiliary cortex scripts (active core entrypoints are archived)
- `common/`: shared numerical helpers
- `workbench/`: rendering and figure assembly helpers

The active workflow should be launched from `scripts/core_workflow/` and resolved within `scripts/core_workflow/lib/`.
Active numbered core steps must stay self-contained in that tree and must not call legacy helper code
from `common/`, `workbench/`, or the repo `wb_command` wrapper at runtime.

Traceability contract for `core_workflow` entrypoints:

- supports `--workflow-branch` with `kong2019`, `kong2022_routea`, `kong2022_routea_17net`, `kong2022_routea_ctxreg`, `kong2022_routea_ctxdrop`, `kong2022_routeb`
- supports repeatable `--upstream-step-id`, `--input-artifact`, `--output-artifact`
- supports `--step-manifest-out` emitting standardized JSON containing:
  - `step_id`
  - `workflow_branch`
  - `upstream_step_ids`
  - explicit input/output artifacts
  - command and git commit

Step inventory is in `scripts/core_workflow/step_registry.json`.
Legacy or retired surfaces are preserved under `_archive/`.

Interpreter contract:

- Use explicit Python path for repo tasks: `/home/yojiao/.conda/envs/py314/bin/python`
- Do not run core workflow checks with bare `python`
- Full runbook: `docs/environment/agent_runtime_contract.md`

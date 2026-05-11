# Archive Index

This index tracks assets moved out of the active workflow surface during core-workflow reorganization.

## 2026-05-11 Core Workflow Reorganization

Moved to archive to reduce active-surface ambiguity:

- `docs/hipp_parcellation_network/kong2022_route_b_flow.md`
  - reason: superseded by canonical `core_workflow_*` docs and numbered entrypoints
- `docs/hipp_parcellation_network/t1only_kong2022_route_b_runbook.md`
  - reason: superseded by canonical step registry and branch-neutral workflow overview
- `scripts/run_hippunfold_local_t2w_t1reg_isolated_cache.sh`
  - reason: retired T2-specific launcher, no longer part of active public workflow

Archive destinations:

- `_archive/hipp_parcellation_network/legacy_docs/core_reorg_2026-05-11/`
- `_archive/deprecated_2026-05-11/scripts/`

Notes:

- Historical paths are retained for traceability in archive; no active docs should depend on archived paths for execution.
- Active execution must start from `scripts/core_workflow/`.

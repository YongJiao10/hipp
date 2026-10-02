# Archive Index

This index tracks assets moved out of the active workflow surface during core-workflow reorganization.

## 2026-05-11 Core Workflow Reorganization

Moved to archive to reduce active-surface ambiguity:

- Retired branch-specific RouteB runbooks and old group-atlas planning notes are superseded by the canonical `core_workflow_*` docs and numbered entrypoints.
- Retired T2-specific launchers are superseded by the active T1w-only core workflow surface.

Archive destinations:

- `_archive/hipp_parcellation_network/legacy_docs/core_reorg_2026-05-11/`
- `_archive/deprecated_2026-05-11/scripts/`

Notes:

- Historical paths are retained for traceability in archive; no active docs should depend on archived paths for execution.
- Active execution must start from `scripts/core_workflow/`.

## 2026-08-18 Native-space beta exploration

The prior Stanford and JJ_TMS Standard/MNI beta entry points were retired from
the active source surface after the Native-space comparison. The current
replacement remains exploratory and lives under `scripts/exploratory_beta/`;
it is not part of the numbered formal workflow.

Current exploratory entry points and runbooks:

- `scripts/exploratory_beta/README.md`
- `docs/hipp_parcellation_network/stanford_tms_surface_beta.md`
- `docs/hipp_parcellation_network/jj_tms_sptms_cluster_beta.md`

Do not mix historical Standard/MNI beta tables with the Native-space
exploratory runs, and do not treat either branch as a frozen reportable result.

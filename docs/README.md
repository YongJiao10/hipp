# Docs Layout

Documentation is grouped by intent.

Keep one-off analysis scripts, notebooks, result tables, logs and generated
research images under the repository's `tmp/` workspace. The hippocampal QC
comparison uses `tmp/hippocampal_qc/scripts/` for code and
`tmp/hippocampal_qc/outputs/` for products; this directory retains its reports.

For the source-only GitHub scope and future pushes, see [GitHub synchronization](github_sync.md).

- `official/`: project-level context and cross-cutting summaries
- `hipp_parcellation_network/`: canonical hippocampal workflow docs and branch references
- `cortex/`: cortex derivation references
- `rendering/`: Workbench rendering conventions
- `structural/`: structural-only notes
- Visualization Suites: multi-dataset and 2x2 grid visualization architecture (Figures 1–8), see [`scripts/visualize/README.md`](../scripts/visualize/README.md)

Canonical active hippocampal workflow docs:

- `hipp_parcellation_network/core_workflow_overview.md`
- `hipp_parcellation_network/core_workflow_step_registry.md`
- `hipp_parcellation_network/core_workflow_artifact_lineage.md`
- `hipp_parcellation_network/multi_branch_flow.md`
- `hipp_parcellation_network/step_by_step_flows.md`
- `hipp_parcellation_network/fmri_qc.md`

Supporting active references:

- `hipp_parcellation_network/core_workflow_file_summary.md`: quick source-file
  index only; use the step registry and artifact lineage doc for IO/provenance
  contracts.
- `hipp_parcellation_network/kong2022_route_a_smoke.md`: RouteA smoke-test
  notes.
- `hipp_parcellation_network/archive_index.md`: historical-doc index.

The public script surface for this workflow is `scripts/core_workflow/`.
Historical docs and retired branch-specific runbooks are preserved under
`_archive/` and should not be treated as active workflow contracts.

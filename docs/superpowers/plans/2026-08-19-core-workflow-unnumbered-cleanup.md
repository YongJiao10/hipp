# Core Workflow Unnumbered-Code Cleanup Implementation Plan

> **Status (2026-09-10): Superseded.** The formal maintenance helpers listed in this historical plan were intentionally retired and deleted. Do not execute their historical paths; use the current numbered workflow, manifests, and readiness checks.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove non-step code from the protected `scripts/core_workflow/` and `scripts/individual_workflow/` numbered-entrypoint surfaces by deleting the broken unused JJ-TMS runner, numbering mature reusable individual workflows, and moving maintenance/JJ_TMS utilities to explicit tool or dataset-specific locations.

**Architecture:** Keep `scripts/core_workflow/` and `scripts/individual_workflow/` as protected mature workflow surfaces: numbered entrypoints plus their documented shared implementations only. Keep shared implementation modules in the existing `scripts/core_workflow/lib/` seam and backend launch assets in `sge_argon/`. Move formal group-operation utilities to a new `scripts/maintenance/` surface and JJ-TMS-specific utilities to a new `scripts/jj_tms/` surface. Update all path-sensitive imports, registry entries, tests, and active documentation together.

**Tech Stack:** Python 3.14 (`/home/yojiao/.conda/envs/py314/bin/python`), Git path renames, Markdown/JSON workflow contracts, pytest.

**Spec:** User requirement in the current conversation: every unnumbered code file must either be deleted if unused or moved to its correct non-numbered location; unnumbered code must not remain in the numbered `core_workflow` root.

## Global Constraints

- Do not add a new core pipeline step; numbering mature reusable individual
  workflow entrypoints is allowed and required by the protected-surface rule.
- Do not delete any path except the exact user-confirmed target listed in Task 1.
- Preserve the existing `core_workflow/lib/` and `core_workflow/sge_argon/` subdirectory roles.
- Use `/home/yojiao/.conda/envs/py314/bin/python` for repository Python checks; never bare `python`.
- Do not launch production workflow runs; verify only imports, syntax, contracts, and focused tests.
- Update documentation and registries in the same change as path changes.

---

### Task 1: Confirm and remove the unused broken runner

**Files:**
- Delete only after explicit user confirmation: `scripts/core_workflow/run_jj_tms_94_complete_pipeline.py`

**Evidence:**
- The file is untracked and has no repository references.
- py314 AST parsing fails at line 43 with `unterminated f-string literal`.
- The other JJ-TMS runner is syntactically valid and will be moved in Task 3.

- [x] Present the exact deletion target to the user and wait for explicit confirmation.
- [x] Verify the target resolves below the repository working directory.
- [x] Delete only the confirmed file.
- [x] Confirm the target is absent and no other deletion occurred.

### Task 2: Move formal group-operation utilities out of the workflow surface

**Files:**
- Move to `scripts/maintenance/`:
  - `<retired-formal-release-audit-helper>`
  - `migrate_hcp_shared_input_paths.py`
  - `retire_legacy_gradient_spectral_outputs.py`
  - `<retired-formal-refresh-runner>`
  - `run_hcp_routea_gradient_spectral_background.py`
  - `<retired-hcp-provenance-helper>`
- Modify each moved script's `CORE_WORKFLOW_DIR`, `REPO_ROOT`, and same-directory imports to resolve from the new depth.
- Add `scripts/maintenance/README.md` documenting that this directory contains non-step formal audit/migration/launch utilities and is not part of either protected workflow surface.

**Interfaces:**
- Preserve every CLI option and output contract.
- Preserve imports from `scripts/core_workflow/lib/` through the new parent path.
- Preserve the formal runner's same-directory import of the retained-provenance upgrader.

- [x] Apply path-only moves without changing workflow behavior.
- [x] Update unit-test module paths and any subprocess/script path assertions.
- [x] Update `step_registry.json`, source indexes, artifact-lineage tables, runbooks, handoff notes, and active docs.
- [x] Search for stale `scripts/core_workflow/<old-name>.py` references.

### Task 3: Move useful JJ-TMS utilities to a dataset-specific surface

**Files:**
- Move to `scripts/jj_tms/`:
  - `scripts/jj_tms/map_jj_tms_sites_to_kong2022.py`
  - `scripts/jj_tms/run_jj_tms_24_hippunfold_batch.py`
  - `scripts/jj_tms/jj_tms_hcp151_fixed_template_matching.py`
  - `scripts/jj_tms/render_jj_tms_hcp151_fixed_template_overviews.py`
  - `scripts/jj_tms/run_jj_tms_hcp151_fixed_template_overviews.sh`
- Do not place either file in `scripts/individual_workflow/`.
- Modify the moved mapping utility's core-library path setup to point explicitly at `scripts/core_workflow`.
- Add `scripts/jj_tms/README.md` stating that this is the dataset-specific surface for JJ_TMS orchestration and mapping, not a mature multi-dataset workflow surface.
- Update JJ-TMS documentation and focused tests to use the new paths.

Mature reusable individual entrypoints are numbered in place:
`03_00_run_hcp_crossfit_template_matching.py`,
`03_01_validate_hcp_crossfit_robustness.py`,
`04_01_render_hcp_crossfit_individual_overviews.py`,
`05_01_build_hcp151_fixed_template.py`, and the `01_*` RouteA individual
runner/launcher. The large RouteA implementation is kept in
`individual_workflow/lib/` rather than as an unnumbered root entrypoint.

**Interfaces:**
- Keep the mapping CLI and output schema unchanged.
- Keep the batch runner's fixed subject set, explicit py314 interpreter, and concurrency cap unchanged.
- Keep all core numbered steps invoked by these utilities at their existing numbered paths.

- [x] Apply the JJ_TMS path moves and mature individual-entrypoint numbering.
- [x] Update all code, tests, and docs that reference the old paths.
- [x] Verify the moved scripts import or parse without creating workflow outputs.

### Task 4: Verify the cleaned layout

**Files:**
- Modify only documentation/contract files required by the moves.

- [x] Assert that the protected `scripts/core_workflow/` and `scripts/individual_workflow/` surfaces contain no newly introduced dataset-specific runner or mapping code; the protected roots contain no unnumbered `.py` or `.sh` entrypoint.
- [x] Run AST parsing for all affected Python files with py314.
- [x] Run focused tests for the moved formal utilities and JJ-TMS mapping.
- [x] Run the repository's relevant documentation/traceability contract checks.
- [x] Review `git diff --check` and `git status --short`; report any pre-existing user modifications separately.

> Pending explicit confirmation: five unreferenced `.orig` backup files were
> identified separately for optional deletion. No deletion has been performed
> for those paths.

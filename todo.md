# TODO — HCP canonical paths and family-aware cross-fit handoff

Updated: 2026-07-11 UTC

## User intent and constraints

- Rename exactly these five directories under
  `outputs_hcp/inputs/hcp/kong2022/shared_inputs/`:
  `routea_151 -> routea`, `surface_151 -> surface`,
  `routeb_151 -> routeb`, `routea_ctxreg_151 -> routea_ctxreg`, and
  `routea_ctxdrop_151 -> routea_ctxdrop`.
- Do not keep old-name symlinks and do not delete/copy directory contents.
- Then run family-aware five-fold cross-fit and its robustness overview.
- Production must not use `--allow-subject-only-folds`.
- User explicitly requested **no Git commit during this task**. Do not commit
  unless a later user instruction overrides this.
- Do not access archive contents. Use explicit py314 for repository Python.
- Cross-fit remains sequential and raw HCP `Family_ID` stays memory-only.

Design and plan, both intentionally uncommitted:

- `docs/superpowers/specs/2026-07-11-remove-hcp-151-suffixes-and-run-crossfit-design.md`
- `docs/superpowers/plans/2026-07-11-remove-hcp-151-suffixes-and-run-crossfit.md`

## Completed dirty work

### HCP family input

`routea_crossfit_template_matching.py` now accepts generic
`subject,family_id` and HCP-native `Subject,Family_ID` headings
case-insensitively, accepts a UTF-8 BOM, and rejects duplicate semantic
columns. Tests and docs were updated.

Production input:

`data/HCP_YA_subjects_2026_04_09_16_01_42.csv`

Read-only aggregate validation (no identifiers printed): 1206 CSV rows;
151/151 QC subjects covered; 84 in-cohort families; maximum family size 3;
66 families contain multiple cohort members.

### Canonical active path contracts

Active code, tests, and docs were changed from the five approved suffixed
shared-input components to unsuffixed names. Cohort-specific template assets
such as `hipp_group_average_t1only_151` were intentionally not renamed.

Key changed areas include `output_layout.py`, core workflow defaults/route
maps, RouteA/RouteB/ctxreg/ctxdrop render/materialization scripts, workflow
READMEs, path/gradient/ctxreg tests, and active docs under
`docs/hipp_parcellation_network/`.

### Migration utility

Created, untracked:

- `scripts/core_workflow/migrate_hcp_shared_input_paths.py`
- `tests/core_workflow/test_migrate_hcp_shared_input_paths.py`

It preflights all five moves, supports read-only dry-run, uses `Path.rename`,
creates no symlink, uses atomic text writes, skips archive and relationship
paths, and writes a privacy-safe migration manifest. Tests cover collision,
dry-run, inode/content preservation, path boundaries, nested provenance,
`_archive`, and privacy.

Last successful verification checkpoints:

- Migration utility tests: 5 passed.
- Focused family/path/ctxreg/gradient regressions: 95 passed, 3 warnings.
- `py_compile` and `git diff --check` passed.

## Completed runtime execution

- The revised immutable-history policy was adopted: historical runtime
  manifests were not rewritten.
- All five canonical destination directories exist; all five `_151` source
  names are absent; no compatibility symlink exists.
- The temporary old-to-new runtime mapping manifest was deleted with explicit
  user authorization after migration.
- Family-aware preflight passed for 151 subjects, five folds, 20 template fits,
  604 mappings, zero overlap, and zero missing inputs.
- Production cross-fit completed with `status=ok`, 20 templates and 604
  individualized mappings.
- Robustness PNG, TSV, and JSON were generated for all four smoothing/hemi
  combinations.
- Independent verification recomputed all 2,416 subject output fingerprints
  without mismatch; no family crossed folds and no subject appeared in its own
  training template.
- Full test suite: 326 passed, 3 warnings; `py_compile` and
  `git diff --check` passed.

## Remaining scientific and documentation work

The mechanical workflow is complete. The remaining blocker is scientific:

- Right hemisphere folds 4–5 show Body-cluster collapse at both smoothing
  levels: only 4 and 2 vertices, versus roughly 193–211 vertices in folds 1–3.
- Right 2 mm minimum macro-Dice/ARI are approximately 0.491/0.345 and the
  corresponding Body prototype correlation reaches -0.143.
- Right 4 mm minimum macro-Dice/ARI are approximately 0.484/0.339 and the
  Body prototype correlation reaches 0.410.
- Left hemisphere robustness is high and does not show this collapse.

Before presenting the right individualized labels as reviewer-ready, diagnose
surface-spectral clustering/local-optimum behavior, fold-specific training
composition, and seed/minimum-cluster-size sensitivity. Preserve the current
outputs as QC evidence; do not silently replace them.

Active immutable-history documentation was corrected during acceptance. The
temporary runtime mapping manifest was removed with explicit user approval.
No commit is authorized yet.

## Dirty worktree inventory

Modified tracked files:

```text
docs/hipp_parcellation_network/core_workflow_artifact_lineage.md
docs/hipp_parcellation_network/core_workflow_overview.md
docs/hipp_parcellation_network/core_workflow_step_registry.md
docs/hipp_parcellation_network/crossfit_template_matching.md
docs/hipp_parcellation_network/fmri_qc.md
docs/hipp_parcellation_network/multi_branch_flow.md
docs/hipp_parcellation_network/step_by_step_flows.md
scripts/core_workflow/03_01_build_kong2022_reference.py
scripts/core_workflow/04_01_materialize_kong2022_roi_fc.py
scripts/core_workflow/04_02_materialize_intrinsic_fc.py
scripts/core_workflow/05_01_compute_hippomaps_gradients.py
scripts/core_workflow/05_02_run_group_atlas.py
scripts/core_workflow/05_05_build_weighted_fisherz_branch.py
scripts/core_workflow/06_07_render_ctxdrop_neighbor_counts.py
scripts/core_workflow/06_10_render_cluster_roi_strength_panels.py
scripts/core_workflow/README.md
scripts/core_workflow/lib/output_layout.py
scripts/individual_workflow/README.md
scripts/individual_workflow/routea_crossfit_template_matching.py
tests/core_workflow/smoke_prepare_run.py
tests/core_workflow/test_hippomaps_gradient_route.py
tests/core_workflow/test_path_templates.py
tests/core_workflow/test_routea_ctxreg_cortex_regression.py
tests/individual_workflow/test_routea_crossfit_template_matching.py
todo.md
```

Untracked files:

```text
docs/superpowers/plans/2026-07-11-remove-hcp-151-suffixes-and-run-crossfit.md
docs/superpowers/specs/2026-07-11-remove-hcp-151-suffixes-and-run-crossfit-design.md
scripts/core_workflow/migrate_hcp_shared_input_paths.py
tests/core_workflow/test_migrate_hcp_shared_input_paths.py
```

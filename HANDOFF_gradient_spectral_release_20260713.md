# Gradient-spectral release handoff (2026-07-13)

## Current state

- `main` contains the formal shifted-cosine implementation at commit `61060c3`
  plus streaming archive hashing fix `020153d`.
- The pre-existing root-level dirty deletions were preserved and are unrelated to
  this release. Do not restore, stage, or commit them without a new user request.
- Feature worktree remains at `.worktrees/shifted-cosine-crossfit`; it was not
  removed.

Implementation rationale and workflow contract are documented in:

- `docs/superpowers/specs/2026-07-11-shifted-cosine-crossfit-design.md`
- `docs/superpowers/plans/2026-07-11-shifted-cosine-crossfit.md`
- `scripts/core_workflow/README.md`
- `docs/hipp_parcellation_network/crossfit_template_matching.md`

## Completed production transition

1. Old relu-cosine gradient-spectral products were moved to
   `_archive/gradient_spectral_relu_cosine_20260713/` using the reviewed plan
   `reviewed_plan.json`.
   - plan status: `ok`
   - items moved: `243/243`
   - plan digest:
     `ab034e2f752291246eb08b3e49b1b492690f23450a47c4bdcd42733fffb519d6`
   - HCP RouteA, weighted-branch, and individual-adapter WTA spot checks remained
     in active outputs.
2. The validated shifted-cosine five-fold result was promoted from
   `outputs_hcp/reruns/shifted_cosine_crossfit_20260711/` to canonical
   `outputs_hcp/work/crossfit_template_matching/` and
   `outputs_hcp/present/overviews/crossfit_template_matching/`.
   - promotion status: `ok`
   - template manifests: `20`
   - subject-combo mappings: `604`
   - robustness rows: `4`
   - rewritten text files: `629`
   - the source rerun was preserved.

## Background release run

The strictly sequential `jobs=1` release was launched and must not be polled at
high frequency.

- run ID: `routea_gradient_spectral_release_20260713T065155Z`
- PID file:
  `outputs_hcp/work/group_atlas/logs/routea_gradient_spectral_release_20260713T065155Z.pid`
- status:
  `outputs_hcp/work/group_atlas/logs/routea_gradient_spectral_release_20260713T065155Z.status.json`
- stdout:
  `outputs_hcp/work/group_atlas/logs/routea_gradient_spectral_release_20260713T065155Z.stdout.log`
- stderr:
  `outputs_hcp/work/group_atlas/logs/routea_gradient_spectral_release_20260713T065155Z.stderr.log`

The worker runs, in order: RouteA canonical8 group, canonical8 overviews,
RouteA family17 group, family17 overviews. It produces only G1-G3
gradient-spectral, diagnostic K=2..8/final K=3, Fisher-z and count.

When the status becomes `ok`, validate with
`scripts/maintenance/run_hcp_routea_gradient_spectral_background.py`'s
`validate_release_outputs()` contract: 8 group manifests must record
`surface_affinity=shifted-cosine`, K range `[2,8]`, final K 3, a connected
affinity graph, both assignment variants, and four non-empty overview PNGs.

## Verification already completed

- Full repository suite before merge: `353 passed, 3 warnings`.
- Final archive selection tests after the safety refinements: `11 passed`.
- Independent read-only review: no blockers.
- Real cross-fit preflight: 20 template manifests, 604 mappings, four
  robustness rows; robustness gate, lineage, privacy scan, and finite metrics
  passed.

## Suggested skills

- `verification-before-completion` when the background status reaches a terminal
  state.
- `diagnosing-bugs` only if the status is `failed`; inspect the recorded failed
  stage and stderr before changing code.
- `requesting-code-review` before any follow-up behavioral change.


## Formal refresh addendum (2026-07-14)

- `main` now includes commit `ae00412` (`Fix formal release cohort and Lynch preflight`).
  It normalizes each pinned raw cohort manifest to canonical `sub-` identities,
  and uses Lynch branch masks with RouteA-ctxreg timeseries exclusively for
  Lynch cluster-profile annotation.
- Verification before launch: real 19-stage preflight passed; full suite: `441
  passed, 3 warnings`; changed modules compiled and `git diff --check` passed.
- The new formal worker is running sequentially and should not be polled at
  high frequency:
  - run ID: `formal_gradient_spectral_refresh_20260714`
  - PID: `719202`
  - status: `outputs_hcp/work/group_atlas/logs/formal_gradient_spectral_refresh_20260714.status.json`
  - stdout: `outputs_hcp/work/group_atlas/logs/formal_gradient_spectral_refresh_20260714.stdout.log`
  - stderr: `outputs_hcp/work/group_atlas/logs/formal_gradient_spectral_refresh_20260714.stderr.log`
  - audit: `outputs_hcp/work/group_atlas/logs/formal_gradient_spectral_refresh_20260714.audit.json`
- After a terminal `status: ok`, run the formal release audit and present an
  exact archive inventory for the superseded active Lynch path set. Do not move
  or remove any such paths without a new explicit user confirmation.


## Formal audit recovery (2026-07-14)

- The original sequential worker completed its first 18 stages but recorded a
  terminal audit failure caused only by presentation-manifest lineage handling.
  Do not reinterpret that historical status as failed FC, gradient, or group
  computation.
- Commit `561dfef` preserves each renderer-generated `rendered_title` and pins
  the audit to the nine formal manifest names, excluding historical generic
  presentation manifests from formal lineage.
- Only the seven overview stages (11 render commands) were rerun; no FC,
  gradient, or group-atlas calculation was repeated.
- Fresh independent audit evidence is in
  `outputs_hcp/work/group_atlas/logs/formal_gradient_spectral_refresh_20260714.audit.json`:
  `status: ok`, `group_combinations: 36`, `overview_items: 9`, `failures: []`.
  All nine formal items have a visible `shifted-cosine` title and four source
  group manifests.

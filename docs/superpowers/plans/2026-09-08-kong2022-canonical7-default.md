# Kong2022 canonical7 default namespace migration plan

## Goal

Make the existing unsuffixed Kong2022 RouteA workflow the canonical7 workflow.
Future default commands must use `kong2022`, `routea`, `reference/kong2022`,
`fc/kong2022`, and the existing unsuffixed RouteA group-atlas root; they must
not create or require `_7net` directory/route suffixes. Canonical8 must not be
selected by any default production path.

The explicit artifact filenames `cortex_canonical7_networks.tsv` and
`cortex_canonical7_network_timeseries.npy` remain unchanged. They identify the
seven-network contract inside the unsuffixed namespace and prevent ambiguity
with ROI-level outputs. Existing canonical8 mappings and historical metadata
remain readable only where required for provenance or non-Kong2022 historical
branches; they are not an implicit fallback for the new default.

## Architecture and invariants

- `kong2022` means direct canonical7 for the default RouteA and group-atlas
  paths.
- `routea` means the canonical7 RouteA branch; `routea_7net` and
  `kong2022_7net` are removed from active default choices and output routing.
- The canonical7 order is fixed: Default, Visual, Somatomotor,
  DorsalAttention, VentralAttention, Control, Language. Auditory is merged
  into Somatomotor by the existing explicit mapping.
- Default reference, FC, gradient, group-atlas, weighted-Fisher-z, and
  overview manifests record `network_taxonomy=canonical7`.
- Missing canonical7 artifacts fail closed. No path-existence fallback may
  silently select the old canonical8 output.
- Existing user changes in the dirty worktree are preserved; only this
  migration's files and tests are edited.

## Spec

User request in the current conversation; no separate specification file.

## Implementation tasks

### Task 1: Add failing contract tests first

Update/add focused tests for the unsuffixed default contract:

- `routea` resolves to the unsuffixed group-atlas and shared-input roots and
  is validated as canonical7.
- `kong2022` reference/FC targets are the direct seven-network targets under
  unsuffixed paths.
- default parser values for reference extraction, group-atlas, and weighted
  Fisher-z select canonical7.
- canonical7 metadata and order are required, while canonical8 fallback is
  rejected for the default branch.
- no active default code path emits `routea_7net` or `kong2022_7net`.

Run the focused tests with the repository interpreter and confirm they fail
for the old behavior before changing production code.

### Task 2: Migrate shared route and output contracts

Update `lib/route_tags.py` and `lib/output_layout.py` so the unsuffixed RouteA
tag/root is canonical7. Consolidate duplicate `_7net` routing and update
route/taxonomy validation, manifests, and artifact path helpers. Preserve
explicit non-Kong2022 taxonomies and historical read-only provenance fields.

### Task 3: Make reference extraction canonical7 by default

Update the Kong2022 reference builders and cortex ROI-timeseries extractor so
the default `kong2022` branch writes direct canonical7 outputs to
`reference/kong2022/`, records canonical7 metadata, and does not emit the old
canonical8 network file as the default product. Keep ROI parcel outputs and
explicit canonical7 filenames intact.

### Task 4: Migrate downstream FC, gradient, atlas, and overview defaults

Update FC materialization, Hippomaps gradients, group-atlas/K-selection,
weighted Fisher-z, group overview, matrix, and rendering entry points so their
unsuffixed defaults consume canonical7 and write unsuffixed RouteA outputs.
Remove implicit `_7net` special cases and make canonical7 provenance checks
apply to the default `kong2022`/`routea` branch.

### Task 5: Synchronize documentation and registries

Update core-workflow README, artifact lineage, step registry, multi-branch and
step-by-step flow docs, and any output manifests touched by the migration.
Document that canonical7 is now the default and that canonical8 is not an
active default branch. Do not rewrite historical plans that describe the
superseded parallel branch.

### Task 6: Validate and stage the 23P handoff

Run focused tests, compile checks, and the relevant full core-workflow test
suite with `/home/yojiao/.conda/envs/py314/bin/python`. Inspect diffs and
verify no default command/path emits `_7net` or selects canonical8.

After code validation, relocate the already-generated 23P canonical7 products
from the temporary `reference/kong2022_7net` namespace to
`reference/kong2022`, validate all subjects and matrix shapes, and present the
exact old-source target list before any cleanup of the temporary namespace.

## Verification commands

- `/home/yojiao/.conda/envs/py314/bin/python -m pytest ...` for focused and
  full core-workflow tests.
- `/home/yojiao/.conda/envs/py314/bin/python -m py_compile` on changed Python
  files.
- Read-only path/metadata checks for the 23P handoff, including 38 subjects,
  seven network rows, canonical7 order, and zero NaNs.
- `git diff --check` and a final `git diff` review limited to this migration.

# Selective ctxreg worktree integration

Source: `ctxreg_worktree`, branch `feat/ctxreg-visualize-local`, uncommitted
changes on `8cb3537`. Target: formal `main` checkout at the same commit.
Selection date: 2026-09-15. This is a source integration, not scientific
product migration or full CtxReg production acceptance.

## Included

- Kong `03_01`, `03_02`, both extraction modules, and `lib/io_utils.py`:
  reject the legacy Iowa persistent concat tree and retain run-level inputs.
- Lynch `03_04`: explicit ordered run inputs, in-memory concatenation,
  persistent input validation and recorded run provenance.
- Lynch `03_03`: explicit per-subject run metadata. Integration additionally
  rejects missing/conflicting IDs and noninteger lengths, validates frame
  coverage before reference writes, and records IDs/segments/metadata paths.
- Iowa7T branch downstream: explicit per-subject run metadata is required for
  reference/FC stages; non-four-way run IDs and boundaries are preserved in
  stage manifests and summaries, and coverage is checked before materializing
  outputs.
- `cortex_network_area/prepare_iowa_surfaces.py`: persistent surface output
  scope, refusal to overwrite existing targets, explicit repository root,
  and rejection of historical temporary surface entries.
- Read-only fixed-template overview preflight/receipt contract and its focused
  regression test. It validates the existing scene/source/roster contract but
  does not rebuild or overwrite formal overview products.
- Formal template-matching presentation extension: the maintained renderer
  preserves the established four-grid mode and adds explicit Standard/CtxReg
  2mm/36P branch rendering. The original formal four-grid implementation is
  preserved byte-for-byte as the explicitly named
  `render_template_matching_dominant_subregions_legacy.py`.
- User-authorized formal-directory regeneration of the two branch PNGs, the
  comparison CSV, and the branch manifest. These products were generated from
  formal inputs and compared against the worktree reference; no image was
  copied from the worktree.
- Focused regression tests and matching documentation for the included input,
  run-metadata, Iowa7T, cortex-network-area, and fixed-template contracts.

## Excluded and reasons

- `lib/output_layout.py`: chooses local/parent output roots by existence,
  violating the formal no-silent-fallback contract.
- Fixed-template builder changes: default validation derives the expected
  template ID from the manifest being validated, weakening identity checks.
- Coupled `04_00`, `04_01`, gradient, group-atlas, overview and branch routing
  changes: deferred as a unit. The worktree documents incomplete full
  production acceptance; selecting fragments would leave consumers and
  producers inconsistent. Main also has independent group-atlas edits.
- Worktree `05_02`/`06_01` adds signal-branch fields, but also enables
  experimental K selection and retains existence/`next(...)`/FC-mask
  fallback paths; it does not satisfy the exact-artifact contract and must
  remain coupled to a separately reviewed downstream redesign.
- `gifti_labels.py` imports the worktree-only `gradient_assignment_contract`
  and rewrites group-atlas NPY/GIFTI/manifest assets, so it cannot be
  promoted without that producer/consumer and output migration set.
- `workbench_scene_batch.py` only supports the deferred generated scene
  sidecars; the main checkout has no independent consumer for its new option.
- The broad visualization-suite/cohort-resolver redesign and unrelated
  renderer changes remain excluded. Only the narrowly scoped
  template-matching renderer extension above is included; the rest requires a
  separate contract review before promotion.
- One-time restoration/repair scripts, dated receipts, active roster snapshot,
  matching wrapper and tests tied to excluded functions: outside this minimal
  reusable integration.
- Generated PNG/NPZ/TSV/CSV/JSON products, diagnostic logs, runtime links and
  worktree-only setup: not source integration inputs.
- Source deletions, archived material and historical production claims: not
  imported. No archive contents were inspected.

Formal regression tests are retained; exploratory experiments are excluded.
Existing main edits and its staged archive relocation are outside this change.
No `outputs_*` production result trees are written by this integration. The
user-authorized template-matching branch products were regenerated in the
formal renderer's own `scripts/visualize/template_matching_dominant_subregions`
directory.
Read-only formal-product audit: the HCP `routea_ctxreg` overview manifest
has 5/5 items with existing PNGs and source group manifests; the Iowa
`routea_ctxreg/patient` overview manifest has 1/1 with both references
present. All audited overview items use `route_tag=routea_ctxreg`.
Kong/PFM `reference_root` paths are upstream reference dependencies, not
active branch roots; the controlled shared, gradient, and overview roots
remain `routea_ctxreg`.
Stanford fixed-template's existing manifest is complete 194/194, and all
declared PNGs, source manifests, and label GIFTIs exist, but its recorded
scene fingerprint differs from the current scene fingerprint; all 194 item
receipts therefore refer to the stale scene. This is a rebuild candidate,
not permission to overwrite now: the future write set is the overview
manifest, status table, 194 item JSON receipts, and 194 declared PNGs.
No such production writes have been authorized or performed.

## Verification

Run with `/home/yojiao/.conda/envs/py314/bin/python`, one test process,
`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and
`PYTHONDONTWRITEBYTECODE=1`. Use pytest with `-p no:cacheprovider` for the
input, run-metadata, Iowa7T, cortex network area, and fixed-template tests.
Tests use synthetic inputs or read-only contract fixtures; no full real-data
PFM, residualization, FC or figure production is claimed.

Result on 2026-09-15: all 101 selected tests passed (8.02 s), with no skips.
The added negative cases first reproduced five failures in the source
candidate; strict ID/length and pre-write validation resolved those failures.
The selected tracked patch passed its targeted `git diff --check`; the global
check reports one unrelated pre-existing trailing blank line in
`stanford_iowa3t_depression_anhedonia_scales.md`.

The formal template-matching renderer contract test passed 3/3. Its regenerated
Standard and CtxReg PNGs were byte- and pixel-identical to the worktree
reference images; the branch CSV was byte-identical and the manifests matched
after normalizing only checkout paths and timestamps.

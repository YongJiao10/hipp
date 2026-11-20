# Repository Rules

## Workspace and execution

- Resolve the checkout root from the working directory or `git rev-parse --show-toplevel`.
  Use `<repository>/tmp/` for all temporary repository work.
- Identify `idas` or `argon` from `SYSTEM_NAME`, `CLUSTER_NAME`,
  `SGE_CLUSTER_NAME` or hostname before parallel work or an SGE array. If detection
  is ambiguous, require an explicit `SYSTEM_NAME`.
- Use host-appropriate paths. Never put local mount paths or interpreters in
  Argon qsub commands, production manifests or production logs.
- Start Argon sessions from the checkout with `codex-argon`, `codex-argon resume`
  or `codex-argon agents`. Personal launcher details belong in
  `/Users/yojiao/.local/share/codex-argon.md`.
- The main checkout (`/home/yojiao/hpchome/hipp` on idas) stays on `main`, which
  holds the mainline work. Side work by other agents uses its own branch in a
  separate git worktree outside the main checkout (for example
  `/home/yojiao/hpchome/hipp-<agent>`); never switch, commit on, reset or rebase
  the main checkout from side work, and never rewrite another agent's branch.
  Check `git branch --show-current` before any git write.

## Filesystem authorization

- Obtain explicit permission before accessing or browsing archive contents.
- Production result-tree migration, restructuring, deletion, cleanup or
  out-of-workflow edits (hand-written or one-off scripts that rewrite formal
  products) require an exact user-approved path list. Keep those trees
  read-only until approval.
- Routine regeneration needs no path list: rerunning a maintained workflow
  step with its documented arguments to rewrite its own formal products at
  their existing paths, such as `--resume` reruns or re-rendering overviews
  and indexes after upstream changes. Report what was regenerated.
- `wrong/` contains legacy erroneous HippUnfold results from a completed
  one-time migration. Do not access, reuse, add to, modify, move or delete it.
- In planning mode, obtain confirmation before creating a directory; state its
  name, purpose and disk location.

## Documentation

- Keep `AGENTS.md` focused on durable operating rules. Maintain workflow steps,
  batch settings, output paths and naming tables in code, configuration and
  their workflow documentation.
- Write project documentation in English using current complete datasets and
  authoritative analysis rosters. Replace outdated text in place; keep batch
  and upload histories out of active instructions.
- Update documentation with behavior, path, cache, launcher, environment or output
  changes. Runtime-stack changes also require an environment-manifest update.
- Formal documentation must not depend on pre-existing one-off scripts, reports,
  previews or logs in `tmp/`. Maintain reusable tools in their project locations;
  temporary-workspace guidance and self-contained commands may use `tmp/`.

## Scientific branches and outputs

- Use the existing path interfaces in
  [output_layout.py](scripts/core_workflow/lib/output_layout.py) and the
  [workflow guide](scripts/core_workflow/README.md) for formal output locations.
- Preserve scientific branches, including 36P, method and atlas variants.
  Repairs replace approved formal products in place; do not introduce dated,
  recovered/regenerated roots or compatibility symlinks.
- File, directory, template, route, branch, constant and figure names must not
  contain subject or cohort counts. Record `n_subjects` in manifests and validate
  it against roster length. Dates and run IDs belong in logs/manifests.
- Establish density from consumed surface/CIFTI assets and recorded
  `density_evidence`. Report disagreements with arguments and treat the consumed
  assets as authoritative.

## Functional network names

- Use the shared display mapping in
  [network_colors.py](scripts/core_workflow/lib/network_colors.py) and the
  [cross-atlas mapping](config/cross_atlas_network_merge.json).
- Preserve source keys, native network identities and published terminology;
  display aliases must not change artifact selection. Identify the atlas and
  taxonomy in figures, and define full network names on first use in prose.

## Runtime

- Use `/home/yojiao/.conda/envs/py314/bin/python` for local workflow scripts,
  tests and documentation checks. First run `python -c "import sys; print(sys.executable)"`
  for context; rewrite bare-Python examples to the explicit interpreter at execution.
  If the wrong interpreter is used, stop and rerun correctly.
- Use the documented Argon runtime for SGE/MATLAB launchers. Preserve branches
  that do not need Python rather than adding a local interpreter dependency.
- Before execution, follow the [agent runtime contract](docs/environment/agent_runtime_contract.md)
  and [HippUnfold runtime manifest](docs/environment/hippo2_reproducibility_manifest.md)
  for effective CLI, model/CUDA and cache gates, launch settings and retained outputs.
- New HippUnfold runs use the official release pinned in
  `environment/hippo2_server.yml` with documented resource/runtime adapters.
  Do not substitute development `main` or apply local boundary, reference-plane
  or Workbench scientific fixes.
- HippUnfold 2.4.0 and 2.4.1 results are equivalent for this project. Existing
  2.4.0 results remain valid; do not reopen equivalence checks, flag results as
  stale or propose reruns solely because these versions differ. Preserve the
  actual version in historical records; new-execution gates are not retroactive.

## Exact artifact selection

- Resolve the exact manifest-declared filename and variant. Missing, stale,
  ambiguous, malformed or mismatched artifacts must stop with an actionable
  error naming the contract and path/metadata mismatch.
- Never substitute another file, older cache, label variant, recomputed
  approximation or taxonomy. Filesystem existence, priority, glob/directory
  order, `next(...)`, `glob()[0]` and unsuffixed aliases are not selection policies.
- Compatibility must be caller-selected, separately tested, validated and
  provenance-recorded. New fallback-like branches require documented
  justification and explicit user approval before implementation.

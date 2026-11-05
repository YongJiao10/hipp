# Dataset Output Layout

This document defines the maintained layout for repository-local
`outputs_hcp/`, `outputs_stanford/`, `outputs_iowa_jj_tms/`,
`outputs_iowa_estt/`, `outputs_banda/`, and `outputs_pdc/`.
The core workflow implements its paths in
[`output_layout.py`](../../scripts/core_workflow/lib/output_layout.py).
The external Iowa7T production layout has its own
[contract](../iowa7t_36p_hipp_outputs_layout.md).

## Directory structure

```text
outputs_<dataset>/
├── inputs/                         upstream assets consumed by analysis
├── work/
│   ├── pfm_nsi/<original_package>/  NSI quality-analysis packages
│   └── <existing_analysis>/         reusable computation products
├── present/                        formal presentation products
├── templates/<template_id>/        HCP functional transfer templates
└── runs/<run_id>/                  independent execution records/products
```

Create a directory when a workflow has concrete outputs for it. The tree
shows permitted roles; each dataset only needs the categories it uses.
Retain coherent existing paths and internal scientific package structures.
The first-level directory count follows actual product ownership.

## Scientific ownership

- `inputs/` contains upstream assets, including generated structural surfaces,
  cortical references, extracted signals/FC banks, and QC/cohort manifests.
  A generated asset can be an input to downstream analysis.
- `work/` contains retained computation products, including cortical
  parcellations, gradients, group and individual atlases, matching results,
  validation/statistical analyses, and NSI quality analyses.
- `present/` contains the formal presentation products defined by each
  workflow. A self-contained scientific package can retain its own figures,
  tables, and logs.
- Anatomical group-average surfaces use
  `inputs/kong2022/templates/hipp_group_average/`.
- HCP fixed functional transfer templates use
  `outputs_hcp/templates/<template_id>/`. They contain functional cluster
  labels, support, and cortical connectivity prototypes. The individual
  template builder, JJ_TMS matching workflows, and Iowa7T exporter share
  this contract.
- Kong2022 ArealMSHBM assets in
  `inputs/kong2022/cortex_references/pfm_areal_mshbm/` and Lynch2024 FASTANS
  assets in `work/cortex_pfm/` retain their different producers and
  downstream reference contracts.

Preserve separate preprocessing variants, cohorts, methods, taxonomies,
cortex controls, and scientific parameter branches. Formal HCP control routes remain `pfm_ctxreg`
and `pfm_ctxdrop`. Exact group-atlas roots are documented in
[Formal Group-Atlas Layout](active_group_atlas_inventory.md).

Existing valid cortical references retain their scientific contents.
Hippocampal coordinate corrections do not require cortical reference
regeneration because those references use cortical time series and labels.

## NSI quality-analysis packages

Use `work/pfm_nsi/<original_package_name>/` for PFM-NSI results. Keep subject
scores, group summaries, figures, logs, and original execution records
inside each package. Preserve existing package names, sparsity variants,
and acquisition variants, including JJ_TMS's separate single-run package.

The NSI CLI continues to accept an explicit `--outdir`. Consumers use the
exact declared input file at its current package location. Paths recorded
by the original execution remain historical evidence; directory relocation
preserves those original records and all scientific file contents.

## Execution records and non-official products

Use `runs/<run_id>/` for an execution that needs independent records or
non-official products, including smoke testing, dedicated validation, and
exploratory rendering. A run is one workflow execution; acquisition-run
segmentation remains with its scientific input.

Formal scientific results retain their existing stable locations. Packages
can keep their own logs and manifests, and their execution metadata need
not be extracted into `runs/`.

`tests/core_workflow/smoke_prepare_run.py` prepares a command plan and its
run-owned directories under `outputs_hcp/runs/<run_id>/`. It preserves its
existing run IDs, directory keys, and command-manifest schema. Stage 01
continues to use the canonical HCP HippUnfold batch root; the execution
layout applies to the products owned by the run.

## Archives and filesystem changes

Keep output archives under
`wrong/outputs_<dataset>/<original-relative-path>`, preserving names and
internal hierarchy. Retained repair backups belong there alongside their
original records.

Use exact user-approved source and destination lists for result-tree moves.
Retain scientific contents and move whole packages where possible. Any
separate container deletion requires an empty target and explicit approval
of that deletion path. Update affected active producers, consumers, and
documentation together. Resolve exact declared artifacts and report missing
inputs instead of selecting a different variant or historical product.

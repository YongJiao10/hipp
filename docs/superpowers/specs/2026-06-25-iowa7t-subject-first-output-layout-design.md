# Iowa7T Subject-First Output Layout Design

## Goal

Replace the existing stage-first Iowa7T hippocampal output tree with one fixed,
subject-first layout. The layout must support both single-echo and multi-echo
processing without duplicating subject structural products or creating new
dataset-level output roots.

## Canonical Roots

The storage is the same on both systems:

```text
Argon: /Shared/jianglab
Local: /home/yojiao/LSS/jianglab
```

The canonical Iowa7T source and output roots are:

```text
<jianglab>/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P/
<jianglab>/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P/hipp_outputs/
```

Repository-local `outputs_iowa/` is not a valid production output root.

## Fixed Layout

The tree below defines valid canonical locations. Directories are created only
when they receive real content; empty future-scoped directories such as
dataset-level `group/` must not be pre-created.

```text
hipp_outputs/
├── subjects/
│   ├── sub-04172026/
│   │   ├── hippunfold/
│   │   └── single_echo/
│   │       ├── cortex/
│   │       │   └── areal_mshbm/
│   │       ├── surface/
│   │       │   ├── raw/
│   │       │   ├── tsnr_gate/
│   │       │   ├── 2mm/
│   │       │   └── 4mm/
│   │       ├── fc/
│   │       │   └── kong2022/
│   │       ├── atlas/
│   │       │   ├── gradient/
│   │       │   └── wta/
│   │       ├── qc/
│   │       ├── figures/
│   │       │   ├── gradient/
│   │       │   └── wta/
│   │       └── provenance.json
│   └── sub-06112026/
│       ├── hippunfold/
│       ├── single_echo/
│       └── multi_echo/
└── runs/
    ├── logs/
    ├── summaries/
    ├── handoff/
    └── cache/
```

There is no `variants/` wrapper. Processing branches are direct, fixed siblings
under each subject: `single_echo/` and `multi_echo/`.

## Structural and Functional Ownership

`hippunfold/` is subject structural output and is shared by all functional
branches. It must be generated once per subject and must not be copied into
`single_echo/` or `multi_echo/`.

Everything dependent on the functional time series belongs to its processing
branch:

- individualized cortex labels;
- hippocampal sampled and smoothed time series;
- tSNR masks;
- cortex references;
- FC matrices;
- gradient and WTA atlases;
- branch QC;
- figures.

`provenance.json` records the exact input CIFTI/NIfTI paths, processing method,
software versions, timepoint count, CIFTI brain-model description, parameters,
and upstream structural path. Processing details do not create additional
dataset-level roots.

## Group and Runtime Ownership

`group/` contains only cross-subject products and is created only when a
cross-subject product exists. A one-subject run may populate these paths when it
creates an actual reusable group-scoped asset, but no empty `group/` subtree is
created for future intent alone.

`runs/` contains execution metadata and reproducible caches:

- logs and heartbeats;
- status summaries;
- handoff records;
- temporary overview adapters and other regenerable caches.

Runtime cache content is not a scientific result and must not appear under a
subject's `atlas/` or `figures/`.

## Legacy-to-Canonical Mapping

For `04172026`, the maintained migration is:

```text
inputs/hippunfold/dense_corobl_batch/sub-04172026/hippunfold/
  -> subjects/sub-04172026/hippunfold/

inputs/kong2022/shared_inputs/surface/sub-04172026/surface/
  -> subjects/sub-04172026/single_echo/surface/

inputs/kong2022/shared_inputs/routea/sub-04172026/reference/
  -> subjects/sub-04172026/single_echo/cortex/

inputs/kong2022/shared_inputs/routea/sub-04172026/fc/
  -> subjects/sub-04172026/single_echo/fc/

work/individual_atlas/sub-04172026/gradient/
  -> subjects/sub-04172026/single_echo/atlas/gradient/

work/individual_atlas/sub-04172026/wta/
  -> subjects/sub-04172026/single_echo/atlas/wta/

present/overviews/individual/gradient/
  -> subjects/sub-04172026/single_echo/figures/gradient/

present/overviews/individual/wta/
  -> subjects/sub-04172026/single_echo/figures/wta/

work/individual_atlas/_overview_core_adapter/
  -> runs/cache/overview_core_adapter/

work/individual_atlas/_runs/
  -> runs/

inputs/kong2022/templates/hipp_group_average_t1only_151/
  -> group/hippocampal_template/      # only if source has real content

inputs/kong2022/qc/
  -> group/qc/                        # only if source has real content

inputs/kong2022/manifests/
  -> group/manifests/                 # only if source has real content
```

The empty legacy `inputs/qc/` has no canonical destination because it has no
current or future responsibility.

## Redundancy Rules

A directory level is retained only when it represents a real set of current or
future siblings. The following legacy wrappers are not part of the canonical
layout:

```text
inputs/
inputs/qc/
inputs/hippunfold/
dense_corobl_batch/
inputs/kong2022/
shared_inputs/
work/
individual_atlas/
present/
overviews/
individual/
```

Scientific method names such as `gradient`, `wta`, `single_echo`, and
`multi_echo` remain because they are genuine sibling categories.

## Code Contract

One path-layout module is the only authority for canonical Iowa7T paths. All
launchers, workflow stages, renderers, monitors, Argon handoff scripts, tests,
and documentation must use it. Production commands must not require repeated
manual path overrides.

The contract must:

- select the local or Argon Jianglab prefix from the runtime environment;
- reject repository-local `outputs_iowa/` for production Iowa7T runs;
- validate subject IDs and the fixed branch names;
- return shared HippUnfold paths separately from branch-dependent paths;
- prevent accidental writes to legacy trees after migration.

## Migration Safety

Migration is manifest-driven:

1. Inventory every legacy file and symlink without accessing archive content.
2. Build an exact old-to-new mapping.
3. Detect collisions and unresolved references before moving anything.
4. Dry-run the transfer.
5. Copy or move into canonical locations, creating only parent directories for
   files or symlinks that are actually transferred.
6. Rewrite path-bearing manifests through maintained code.
7. Verify file counts, sizes, hashes for small critical files, CIFTI/GIFTI
   readability, HippUnfold completeness, and individual atlas manifests.
8. Switch all code and documentation to the canonical layout.
9. Present the exact obsolete path list for explicit user deletion approval.
10. Delete nothing until that approval is received.

The migration must preserve `04172026` outputs and establish the same layout for
`06112026` before new processing starts.

## Acceptance Criteria

- `04172026` remains complete after migration.
- `06112026` uses one shared HippUnfold result and direct `single_echo/` and
  `multi_echo/` branches.
- No maintained command writes to `outputs_iowa/` or the legacy stage-first
  Jianglab tree.
- No redundant wrapper directory remains in the canonical design.
- All path-dependent tests and documentation use the fixed contract.
- Obsolete legacy paths are removed only after explicit approval of the exact
  deletion list.

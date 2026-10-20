# Scripts Layout

This directory is organized by layer.

- `core_workflow/`: protected canonical multi-dataset workflow (`NN_MM_<semantic_name>.py`) and shared implementation modules (`core_workflow/lib/`); its root must contain numbered steps only
- `hipp_parcellation_network/`: legacy/auxiliary hippocampal scripts (active core entrypoints are archived)
- `cortex/`: legacy/auxiliary cortex scripts (active core entrypoints are archived)
- `common/`: shared numerical helpers
- `workbench/`: rendering and figure assembly helpers
- `individual_workflow/`: protected numbered mature multi-dataset individual
  workflows. HCP uses family-aware five-fold crossfit; non-HCP datasets use
  the immutable HCP151 template for direct matching. Dataset-specific adapters
  belong under `scripts/<dataset>/` (JJ_TMS: `scripts/jj_tms/`).
- `maintenance/`: formal audit, migration, promotion, and launch utilities;
  these are not numbered workflow steps
- `jj_tms/`: JJ_TMS-specific HippUnfold, HCP151 matching, overview, and target
  mapping utilities; this is not a multi-dataset workflow surface
- `exploratory_beta/`: non-numbered Stanford and JJ_TMS Native-space beta
  extraction, models, and figures; these remain exploratory and are not formal
  workflow steps

The active group workflow should be launched from `scripts/core_workflow/` and
resolved within `scripts/core_workflow/lib/`. Do not add unnumbered dataset or
maintenance entrypoints to `core_workflow/` or `individual_workflow/`.
Active numbered core steps must stay self-contained in that tree and must not call legacy helper code
from `common/`, `workbench/`, or the repo `wb_command` wrapper at runtime.

## Pipeline Stage Layers

The numbered `NN_MM` prefix reflects pipeline order, but scripts fall into two distinct layers:

| Layer | Steps | Role |
|-------|-------|------|
| **Core computation** | 01_xx – 05_xx | HippUnfold, surface sampling, tSNR/QC, cortex reference, FC materialization, gradients, group clustering — these produce the parcellation outputs and are required for every dataset. |
| **Presentation / rendering** | 06_xx, 07_xx | Group overview figures, matrix summaries — these read finished group-atlas intermediates and produce PNG visualizations. They are **not** core computation; they are post-hoc viewers that can be re-run independently against any valid group-atlas root. |

## Typical Usage (non-HCP datasets)

For a new dataset (Stanford, Iowa, …) that differs from the HCP file layout,
use path-template arguments instead of broad symlink trees:

```bash
PY314=/home/yojiao/.conda/envs/py314/bin/python

# Core computation:
01_01_run_hippunfold_t1w_local.py  <subject> --dataset <hcp|stanford|iowa_jj_tms> [--dry-run]
02_01_prepare_group_inputs.py      --bold-template "{bold_root}/{sub_subject}/func/…"
                                    --hippunfold-batch-root …
                                    --jobs 2
03_01_build_kong2022_reference.py  --dtseries-template "{data_root}/{sub_subject}/func/…"
04_01_materialize_kong2022_roi_fc.py  --dataset <dataset> --dtseries-template … --left-cortex-surface-template …
05_01_compute_hippomaps_gradients.py  --routes … --shared-inputs-root … --surface-root …
05_02_run_group_atlas.py           --shared-root … --gradient-root … --atlases kong2022

# Presentation (optional, re-runnable):
06_01_build_group_overview.py      --group-root … --atlases kong2022 --methods …
06_10_render_cluster_roi_strength_panels.py --group-root ... --route-tag routea
06_10_render_cluster_roi_strength_panels.py --method wta-top --wta-root outputs_hcp/work/group_atlas/routea_wta_normalized_top20_count_full_k3
07_01_build_group_matrix.py        --group-atlas-root … --routes … --methods …
```

Path-template helper: `scripts/core_workflow/lib/path_templates.py`.
Supported variables: `{data_root}`, `{bold_root}`, `{subject}`, `{sub_subject}`, `{hemi}`, `{hemi_lower}`.

Traceability contract for `core_workflow` entrypoints:

- supports `--workflow-branch` with `kong2019`, `kong2022_routea`, `kong2022_routea_17net`, `kong2022_routea_ctxreg`, `kong2022_routea_ctxdrop`, `kong2022_routeb`
- supports repeatable `--upstream-step-id`, `--input-artifact`, `--output-artifact`
- supports `--step-manifest-out` emitting standardized JSON containing:
  - `step_id`
  - `workflow_branch`
  - `upstream_step_ids`
  - explicit input/output artifacts
  - command and git commit

Step inventory is in `scripts/core_workflow/step_registry.json`.
Legacy or retired surfaces are preserved under `_archive/`.

Interpreter contract:

- Use explicit Python path for repo tasks: `/home/yojiao/.conda/envs/py314/bin/python`
- Do not run core workflow checks with bare `python`
- Full runbook: `docs/environment/agent_runtime_contract.md`

The HCP cross-fitted template-matching workflow deliberately uses full concat
FC rather than run-level splits. Its production family map is an external
`subject,family_id` input, and its labels/templates are isolated under
`outputs_hcp/work/crossfit_template_matching/`. Other datasets consume the
immutable HCP151 template through their own direct-matching adapters. Usage
and review outputs are documented in
`docs/hipp_parcellation_network/crossfit_template_matching.md`.

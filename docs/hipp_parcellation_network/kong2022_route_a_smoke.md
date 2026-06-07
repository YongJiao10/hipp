# Kong2022 Route A ArealMSHBM Smoke Runbook

## Scope
This runbook covers Route A minimal loop using **Kong2022_ArealMSHBM**:

- subject-level ArealMSHBM individualized cortex parcellation (default `gMSHBM`)
- subject-specific `label.gii` + `dlabel.nii`
- one overview PNG per subject

This runbook does not include downstream `reference`, `fc`, or `group-atlas`.

For the maintained core workflow, the Argon SGE scripts under
`scripts/core_workflow/sge_argon/` are tracked backend assets for the RouteA
ArealMSHBM label-generation algorithm. They are not local numbered entrypoints
and their contents should not be rewritten for local execution. Their handoff
contract to `03_02_build_kong2022_routea_reference.py` is the generated label
pair:

- Flat Path (required): `sub-{sid}/lh_labels.txt` and `sub-{sid}/rh_labels.txt`

Those files are required upstream artifacts for RouteA reference construction;
they are not assumed to exist naturally.

For the MATLAB-only smoke used on SGE, submit one job each time instead of
running MATLAB on the login node. Non-array `qsub` defaults to subject index
`1`, even when Argon exposes `SGE_TASK_ID=undefined`:

```bash
sh -c 'qsub scripts/core_workflow/sge_argon/matlab_kong2022_mshbm_subject_sge.sh'
```

This script defaults to subject index `1` (`100610`). To run one explicit
subject without the dual-queue launcher:

```bash
qsub -q PINC -v "SUBJECT_IDS_CSV=100610" \
  scripts/core_workflow/sge_argon/matlab_kong2022_mshbm_subject_sge.sh
```

To run `102311` from the default two-subject list, submit a second single job:

```bash
qsub -v SUBJECT_INDEX=2 scripts/core_workflow/sge_argon/matlab_kong2022_mshbm_subject_sge.sh
```

The dual-queue launcher in `scripts/core_workflow/sge_argon/` calls the
same-directory MATLAB job script above, splits the manifest into UI and PINC
arrays, and keeps the job script's default `mem_free=16G` / `smp 3` resource
request.

## Command
Two-subject smoke (`100610`, `102311`):

```bash
/Users/yojiao/miniconda3/envs/py313/bin/python \
  _archive/deprecated_2026-05-20/scripts_pre_core_workflow/cortex/run_cortex_kong2022_mshbm_batch.py \
  --subjects 100610 102311 \
  --jobs 2 \
  --model gMSHBM \
  --num-rois 400 \
  --w 30 --c 30 --beta 50
```

Preflight-only check:

```bash
/Users/yojiao/miniconda3/envs/py313/bin/python \
  _archive/deprecated_2026-05-20/scripts_pre_core_workflow/cortex/run_cortex_kong2022_mshbm_batch.py \
  --subjects 100610 102311 \
  --jobs 2 \
  --dry-run
```

## Output Layout
Per subject RouteA labels are written to:

- `outputs_hcp/inputs/hcp/kong2022/cortex_references/routea_areal_mshbm/sub-{sid}/`

Key RouteA files:

- `lh_labels.txt`
- `rh_labels.txt`
- `matlab_wrapper.log`

Batch logs:

- `outputs_hcp/inputs/hcp/kong2022/cortex_references/routea_areal_mshbm/_logs/sub-{sid}_kong2022_areal_mshbm.log`

## Failure Diagnostics
The driver is fail-fast and does not auto-download dependencies.

Common blockers:

- `matlab` not visible in current shell `PATH`
- incomplete local CBIG checkout (missing `stable_projects/brain_parcellation/Kong2022_ArealMSHBM`)
- missing CBIG dependencies (`CBIG_ComputeCorrelationProfile.m`, `CBIG_corr.m`,
  `CBIG_ComputeConnectedComponentsFromSurface.m`, `mtimesx`)
- on this SGE runtime, the connected-components helper uses MATLAB
  `graph/conncomp` instead of the `matlab_bgl` `components_mex` path to avoid
  relying on an old MEX binary during the EM connectedness check
- missing CBIG vendored Python dependency
  `external_packages/python/mapalign-master/mapalign/embed.py`; Kong2022
  diffusion embedding imports this through `py313`
- missing CBIG fsLR downsample template
  `data/templates/surface/fs_LR_32k_downsample_900/fslr_downsample_900mesh_parcellation.dlabel.nii`;
  profile generation reads it through `CBIG_ComputeCorrelationProfile`
- missing Kong2022 ArealMSHBM group prior (`Params_Final.mat`)

Route A execution is valid only when these dependencies pass preflight.

## Historical SGE Smoke Blocker

At the 2026-04-28 smoke checkpoint, the dependency and profile-generation
stages passed, but both smoke subjects were killed by SGE with exit status
`137` after `It is EM iteration.. 2..` / `Estep..` in
`CBIG_ArealMSHBM_gMSHBM_generate_individual_parcellation`.

Observed mitigation attempts that did not complete that smoke MATLAB label
stage:

- `#$ -tc 1` to serialize subjects
- `#$ -l mem_free=32G`
- replacing the local connected-components helper with MATLAB `graph/conncomp`

This historical smoke status does not change the maintained core-workflow
handoff contract: full RouteA reference construction requires generated
flat `sub-{sid}/lh_labels.txt` and `sub-{sid}/rh_labels.txt` files for
each subject before `03_02_build_kong2022_routea_reference.py` is run.

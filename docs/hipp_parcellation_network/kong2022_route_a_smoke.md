# Kong2022 Route A ArealMSHBM Smoke Runbook

## Scope
This runbook covers Route A minimal loop using **Kong2022_ArealMSHBM**:

- subject-level ArealMSHBM individualized cortex parcellation (default `gMSHBM`)
- subject-specific `label.gii` + `dlabel.nii`
- one overview PNG per subject

This runbook does not include downstream `reference`, `fc`, or `group-atlas`.

For the MATLAB-only smoke used on SGE, submit one job each time instead of
running MATLAB on the login node:

```bash
sh -c 'qsub tmp/matlab_kong2022_mshbm_subject_sge.sh'
```

This script defaults to subject index `1` (`100610`). To run `102311`, submit a
second single job:

```bash
qsub -v SUBJECT_INDEX=2 tmp/matlab_kong2022_mshbm_subject_sge.sh
```

The SGE launcher is configured with `-q UI-HM` and `mem_free=64G`.

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
Per subject outputs are written to:

- `outputs_experiments/cortex_kong2022_areal_mshbm/sub-{sid}/kong2022_areal_mshbm/`

Key files:

- `sub-{sid}_atlas-kong2022_areal_mshbm.dlabel.nii`
- `labels/sub-{sid}_atlas-kong2022_areal_mshbm.L.label.gii`
- `labels/sub-{sid}_atlas-kong2022_areal_mshbm.R.label.gii`
- `figures/sub-{sid}_kong2022_areal_mshbm_overview.png`
- `summary.json`
- `stage_manifest.json`
- `matlab_wrapper.log`

Batch logs:

- `outputs_experiments/cortex_kong2022_areal_mshbm/_logs/sub-{sid}_kong2022_areal_mshbm.log`

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

## Current SGE Blocker
As of 2026-04-28, the dependency and profile-generation stages pass, but both
smoke subjects are killed by SGE with exit status `137` after
`It is EM iteration.. 2..` / `Estep..` in
`CBIG_ArealMSHBM_gMSHBM_generate_individual_parcellation`.

Observed mitigation attempts that did not complete the MATLAB label stage:

- `#$ -tc 1` to serialize subjects
- `#$ -l mem_free=32G`
- replacing the local connected-components helper with MATLAB `graph/conncomp`

No `project_dir/route_a_outputs/lh_labels.txt` or `rh_labels.txt` files have
been generated yet.

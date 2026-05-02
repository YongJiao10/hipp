# Kong2022 Route A — ArealMSHBM Only

## Summary
Route A minimal loop in this repository targets **official Kong2022_ArealMSHBM** (not Kong2019 compatibility wrappers):

- fs_LR_32k subject-level individualized parcellation via `CBIG_ArealMSHBM_parcellation_single_subject`
- subject `label.gii` + `dlabel.nii` generation
- one overview figure per subject
- strict fail-fast preflight

The default runtime output root is:

- `outputs_experiments/cortex_kong2022_areal_mshbm/`

No new Route A outputs are written to `outputs_migration`.

## Implemented Scripts
- `scripts/cortex/run_cortex_mshbm_subject.py`
- `scripts/cortex/matlab/run_kong2022_mshbm_subject.m`
- `scripts/cortex/run_cortex_kong2022_mshbm_batch.py`
- `scripts/cortex/plot_cortex_kong2022_mshbm.py`

## Defaults
- `model=gMSHBM`
- `num_rois=400`
- `w=30`, `c=30`, `beta=50`
- `run_lengths=900 900 900 900`
- SGE MATLAB-only smoke uses single-job submission (`no array`), queue
  `UI-HM`, and `mem_free=64G`.
- The local connected-components helper uses MATLAB `graph/conncomp` instead
  of the old `matlab_bgl` MEX path while preserving the connectedness check.

## Preflight Requirements
- `matlab` executable available in current shell
- complete CBIG checkout containing:
  - `stable_projects/brain_parcellation/Kong2022_ArealMSHBM`
  - `CBIG_ArealMSHBM_parcellation_single_subject.m`
  - `CBIG_ArealMSHBM_generate_profiles.m`
  - `CBIG_ArealMSHBM_gMSHBM_generate_individual_parcellation.m`
- CBIG dependencies present:
  - `CBIG_ComputeCorrelationProfile.m`
  - `CBIG_corr.m`
  - `CBIG_ComputeConnectedComponentsFromSurface.m`
  - `mtimesx`
  - `external_packages/python/mapalign-master/mapalign/embed.py`
  - `data/templates/surface/fs_LR_32k_downsample_900/fslr_downsample_900mesh_parcellation.dlabel.nii`
- valid group prior `Params_Final.mat`
- `py313` conda environment import path is used by
  `CBIG_SPGrad_diffusion_embedding.sh` for diffusion embedding

## Current Workspace Status
The local `external/CBIG` tree has been supplemented with Kong2022_ArealMSHBM
dependencies, including the vendored Python `mapalign-master` package required
by the diffusion embedding step.

The remaining blocker is SGE exit `137` during the MATLAB EM stage after
`It is EM iteration.. 2..` / `Estep..`; no Route A label text outputs have been
created yet.

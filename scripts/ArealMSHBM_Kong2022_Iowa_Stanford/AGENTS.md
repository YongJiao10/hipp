# Argon-Specific Prompt

Use these instructions for Iowa/Stanford Kong2022 ArealMSHBM work on Argon.

- Scope is path and data adaptation only. Do not change Kong2022_ArealMSHBM
  algorithm logic, model parameters, session semantics, or MATLAB internals
  unless the user explicitly requests it.
- The source scripts to mirror are in
  `/old_Users/yojiao/hipp/scripts/core_workflow/sge_argon`, especially
  `submit_kong2022_route_a_dual_queues.sh` and
  `matlab_kong2022_mshbm_subject_sge.sh`.
- Supported dataset presets and subject functional inputs are only:
  - `DATASET=jj_tms`:
    `/Shared/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS/derivatives_surf/sub-*/func/*_task-rest_run-0*_space-fsLR_den-91k_bold.dtseries.nii`
  - `DATASET=stanford`:
    `/Shared/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/DataOrg_PFM_36P/sub-*/func/*_task-rest_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii`
- Multiple rest runs are passed to MATLAB as separate sessions in
  `params.lh_fMRI_list`, with one matching all-ones censor file per dtseries in
  `params.censor_list`. Stanford currently has one rest dtseries per subject.
  Do not concatenate or merge rest runs before MATLAB.
- Do not use subject `surf.gii` files for this adaptation. The original
  dtseries-based flow did not require them, so path/data adaptation must not add
  them as inputs, preflight checks, or provenance requirements.
- The repository AGENTS.md 10-concurrent-process hard cap is not an Argon SGE
  policy for this workflow. For qsub arrays, mirror the source Argon launcher:
  deterministic UI/PINC split with UI `-tc 5` and PINC `-tc 15`. Do not add a
  new concurrency variable solely because of the local repository rule.
- Login node work should stay lightweight: text edits, manifests, dry-run command
  construction, and small path inspections. MATLAB execution belongs on compute
  nodes via qsub.
- Before submitting a real smoke or full batch, show the exact qsub command or
  use `DRY_RUN=1`. Do not submit the full batch until the user approves it.

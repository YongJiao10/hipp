# Kong2022 ArealMSHBM Interface Rules

Use these instructions for all Kong2022 ArealMSHBM dataset adapters and Argon
launches in this directory.

- Scope is path and data adaptation only. Do not change Kong2022_ArealMSHBM
  algorithm logic, model parameters, session semantics, or MATLAB internals
  unless the user explicitly requests it.
- This directory is the canonical source for the standard interface. Do not
  add a new dataset launcher under `scripts/cortex`, `scripts/iowa7t`,
  `scripts/core_workflow/sge_argon`, or another dataset folder. Extend
  `submit_kong2022_areal_mshbm.sh` and its shared task script instead.
- Supported dataset presets and subject functional inputs are:
  - `DATASET=jj_tms`:
    `/Shared/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS/xcpd_surf/sub-*/func/*_task-rest_run-*_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii`
  - `DATASET=stanford STANFORD_VARIANT=36p`:
    `/Shared/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/DataOrg_PFM_36P/sub-*/func/*_task-rest_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii`
  - `DATASET=stanford STANFORD_VARIANT=22p`:
    `/Shared/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/DataOrg_PFM_22P/sub-*/func/*_task-rest_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii`
  - `DATASET=hcp7t`:
    `${REPO_ROOT}/data/hippunfold_input/sub-*/func/sub-*_task-rest_run-concat.dtseries.nii`, paired with the exact `outputs_hcp/inputs/qc/run_metadata/sub-*.json` record. Split each concatenated CIFTI into four MATLAB sessions using its recorded run lengths; write labels under `outputs_hcp/inputs/kong2022/cortex_references/pfm_areal_mshbm/sub-*/`.
- `DATASET=iowa7t`:
  use the existing `IOWA7T_BRANCH=multiband|multi_echo` and
  `IOWA7T_VARIANT=0p|22p|32p|36p|noxcpd` contracts in the launcher. Iowa7T outputs
  remain under `hipp_outputs/subjects/sub-<id>/<branch>/cortex/areal_mshbm`.
- `DATASET=estt`:
  `/Shared/jianglab/3_Data_Working/fmriprep_processed_esTT/DataOrg_PFM_36P/sub-*/func/sub-*_task-rest_run-concat.dtseries.nii`
  with the established ESTT subject output root
  `/Shared/jianglab/3_Data_Working/fmriprep_processed_esTT/DataOrg_PFM_36P/hipp_outputs/cortex_kong2022_areal_mshbm/sub-<id>`.
  `ESTT_VARIANT=22p` selects the corresponding `DataOrg_PFM_22P` input root
  and isolated output root
  `/Shared/jianglab/3_Data_Working/fmriprep_processed_esTT/DataOrg_PFM_22P/hipp_outputs/cortex_kong2022_areal_mshbm/sub-<id>`.
  `ESTT_VARIANT=23p` selects the corresponding `DataOrg_PFM_23P` input root
  and isolated output root
  `/Shared/jianglab/3_Data_Working/fmriprep_processed_esTT/DataOrg_PFM_23P/hipp_outputs/cortex_kong2022_areal_mshbm/sub-<id>`.
- Multiple JJ_TMS/Stanford rest runs are passed to MATLAB as separate sessions
  in `params.lh_fMRI_list`, with one matching all-ones censor file per dtseries
  in `params.censor_list`. Do not concatenate or merge those source runs.
- ESTT is a separate dataset contract: its one concatenated CIFTI is split into
  four equal-duration CIFTI sessions before MATLAB, matching the completed ESTT
  cohort. Read the timepoint count separately from each selected 22P/23P/36P
  CIFTI; do not reuse one variant's session lengths for another. Do not route ESTT through the
  Iowa7T branch or output tree.
- HCP7T uses the QC physical-run metadata as the authoritative four-session
  boundary contract. Validate subject identity, run IDs 1–4, frame counts, and
  total concatenated map count before splitting. Retain prepared split CIFTIs
  and censors under `${REPO_ROOT}/tmp/`; do not overwrite existing HCP7T label
  or provenance files through the standard launcher.
- Do not use subject `surf.gii` files for JJ_TMS, Stanford, or Iowa7T path/data
  adaptation. The original dtseries-based flows do not require them. ESTT is
  the explicit exception because its established output evidence and label
  materialization contract records the subject fsLR32k surfaces and sulc asset.
- Identify the active system before submitting: use `SYSTEM_NAME`,
  `CLUSTER_NAME`, `SGE_CLUSTER_NAME`, or the hostname. If it is not clearly
  `idas` or `argon`, stop and set `SYSTEM_NAME` explicitly.
- On IDAS, keep total concurrency at or below 10; the standard launcher
  defaults to UI `-tc 3` and PINC `-tc 7`.
- On Argon, use UI `-tc 5` and PINC `-tc 15` by default, for a 20-task cap.
  The launcher rejects UI/PINC overrides whose sum exceeds the detected cap.
  When the Stanford-to-ESTT watcher is deliberately started early while
  Stanford UI `-tc 5` remains active, constrain ESTT to at most 15 total tasks
  (for example UI `-tc 5` plus PINC `-tc 10`) so the combined Argon usage stays
  at or below 20.
  If a watcher restart follows a partial ESTT submission, attach its saved
  array IDs with `ESTT_EXISTING_JOB_IDS` rather than submitting duplicate
  arrays.
- Keep Stanford 22P and 36P runs isolated. `STANFORD_VARIANT=22p` selects the
  22P input root and defaults to the sibling output root
  `/Shared/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/ArealMSHBM_Kong2022_22P`;
  never pair a 22P input root with the unversioned 36P output root.
- Keep ESTT 22P, 23P, and 36P runs isolated. `ESTT_VARIANT=23p` selects the
  `DataOrg_PFM_23P` input and its matching `DataOrg_PFM_23P/hipp_outputs`
  output tree; never pair one ESTT variant's data with another variant's output
  tree. The launcher auto-detects the variant from a standard `DATA_ROOT` suffix
  and rejects a conflicting explicit variant.
- Login node work should stay lightweight: text edits, manifests, dry-run command
  construction, and small path inspections. MATLAB execution belongs on compute
  nodes via qsub.
- For Stanford 22P subjects with existing `lh_labels.txt` and `rh_labels.txt`
  but missing `provenance.json`, the dedicated repair task
  (`repair_kong2022_areal_mshbm_provenance_sge.sh`, archived 2026-09-14 under
  `_archive/deprecated_2026-09-14/ArealMSHBM_Kong2022/`) reconstructed the
  input manifest on an Argon compute node and validated label hashes before and
  after writing metadata; it never reran MATLAB or targeted the 36P roots.
  Retired after the missing-provenance backlog was cleared.
- Before submitting a real smoke or full batch, show the exact qsub command or
  use `DRY_RUN=1`. Do not submit the full batch until the user approves it.

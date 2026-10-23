# Kong2022 ArealMSHBM

This directory maintains dataset adaptation, SGE submission, MATLAB execution
and status reporting for Kong2022_ArealMSHBM. It preserves the scientific
algorithm and model. Use the standard submit/task interface for all dataset
presets; select the separate Iowa7T runwise interface only for one-run-per-call
analysis. Agent constraints are in [AGENTS.md](AGENTS.md).

Run Argon commands from the checkout root. Argon paths use `/Shared/jianglab`;
Python-backed adapters use `/Users/yojiao/miniconda3/envs/py313/bin/python`.
Local workflow Python paths do not apply to these launchers.

## Datasets

Select `DATASET=jj_tms`, `DATASET=stanford`, `DATASET=iowa7t`, `DATASET=estt`,
`DATASET=iowa_hcp500`, `DATASET=hcp7t`, `DATASET=pdc`, or `DATASET=banda`.
The CLI defaults to `jj_tms`; set the dataset explicitly in commands.

| Dataset | Default input root | Default output root | Rest dtseries pattern |
| --- | --- | --- | --- |
| `jj_tms` | `/Shared/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS/xcpd_surf` | `/Shared/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS/ArealMSHBM_Kong2022` | `func/*_task-rest_run-*_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii` |
| `stanford` + `STANFORD_VARIANT=36p` | `/Shared/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/DataOrg_PFM_36P` | `/Shared/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/ArealMSHBM_Kong2022` | `func/*_task-rest_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii` |
| `stanford` + `STANFORD_VARIANT=22p` | `/Shared/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/DataOrg_PFM_22P` | `/Shared/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/ArealMSHBM_Kong2022_22P` | `func/*_task-rest_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii` |
| `iowa7t` | `/Shared/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P` | `/Shared/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P/hipp_outputs` | selected by `IOWA7T_BRANCH` and `IOWA7T_VARIANT` |
| `estt` + `ESTT_VARIANT=36p` | `/Shared/jianglab/3_Data_Working/fmriprep_processed_esTT/DataOrg_PFM_36P` | `/Shared/jianglab/3_Data_Working/fmriprep_processed_esTT/DataOrg_PFM_36P/hipp_outputs/cortex_kong2022_areal_mshbm` | `func/sub-<id>_task-rest_run-concat.dtseries.nii` |
| `estt` + `ESTT_VARIANT=22p` | `/Shared/jianglab/3_Data_Working/fmriprep_processed_esTT/DataOrg_PFM_22P` | `/Shared/jianglab/3_Data_Working/fmriprep_processed_esTT/DataOrg_PFM_22P/hipp_outputs/cortex_kong2022_areal_mshbm` | `func/sub-<id>_task-rest_run-concat.dtseries.nii` |
| `estt` + `ESTT_VARIANT=23p` | `/Shared/jianglab/3_Data_Working/fmriprep_processed_esTT/DataOrg_PFM_23P` | `/Shared/jianglab/3_Data_Working/fmriprep_processed_esTT/DataOrg_PFM_23P/hipp_outputs/cortex_kong2022_areal_mshbm` | `func/sub-<id>_task-rest_run-concat.dtseries.nii` |
| `iowa_hcp500` | `/old_Users/yojiao/HCPpipeline500/work/qunex_iowa7t_hcppipeline500/sessions` | `${REPO_ROOT}/outputs_iowa_hcp500/inputs/kong2022/cortex_references/routea_areal_mshbm` | `<subject>/hcp/<subject>/MNINonLinear/Results/REST/REST_Atlas_MSMAll_hp0_clean_rclean.dtseries.nii` |
| `hcp7t` | `${REPO_ROOT}/data/hippunfold_input` | `${REPO_ROOT}/outputs_hcp/inputs/kong2022/cortex_references/pfm_areal_mshbm` | `func/sub-<id>_task-rest_run-concat.dtseries.nii`, split using `outputs_hcp/inputs/qc/run_metadata/sub-<id>.json` |
| `pdc` | `/Shared/jianglab/3_Data_Working/Depression/PDC/imaging` | `${REPO_ROOT}/outputs_pdc/inputs/kong2022/cortex_references/routea_areal_mshbm` | exact run filenames in `COHORT_MANIFEST`, mapped from `<base>_<visit>_MR/MNINonLinear/Results/` |
| `banda` | `/Shared/jianglab/3_Data_Working/Depression/BANDA/imaging` | `${REPO_ROOT}/outputs_banda/inputs/kong2022/cortex_references/routea_areal_mshbm` | exact run filenames in `COHORT_MANIFEST`, mapped from `BANDA<ID>_MR/MNINonLinear/Results/` |

`DATA_ROOT` and `OUT_ROOT` are explicit overrides. Stanford and ESTT
variant selectors must agree with recognized source-root suffixes and must
use isolated matching output roots. Keep ESTT separate from Iowa7T.

## Sessions and subject products

- JJ_TMS and Stanford pass each selected rest CIFTI as a separate MATLAB
  session, with one matching all-ones censor file. Source runs are not merged.
- HCP7T uses `HCP_RUN_METADATA_ROOT` (default
  `${REPO_ROOT}/outputs_hcp/inputs/qc/run_metadata`). Validate subject, IDs 1–4,
  run lengths and total frames, then split the supplied CIFTI into those four
  physical sessions. Existing label/provenance files are protected.
- ESTT uses `ESTT_VARIANT=22p|23p|36p`. Read each selected CIFTI's actual frame
  count and split it into four balanced sessions. Retain the subject CBIG work,
  flat labels, `labels/*.func.gii`, `labels/*.label.gii`, flat `.dlabel.nii`,
  `provenance.json`, `summary.json`, `stage_manifest.json` and `matlab_wrapper.log`.
  Completion requires the materialized assets; existing labels can resume
  materialization without repeating MATLAB.
- Iowa HCP500 and Iowa7T resolve their declared CIFTI names with Workbench on
  Argon, without adding a Python dependency. Iowa7T multi-echo input is rebuilt
  as a persistent cortex-only fsLR32k CIFTI before MATLAB.

Standard subject products are `${OUT_ROOT}/sub-<id>/{lh_labels.txt,rh_labels.txt,provenance.json}`.
Iowa7T uses `${OUT_ROOT}/subjects/sub-<id>/<branch>/cortex/areal_mshbm/`;
its derived multi-echo CIFTIs use `runs/derived_inputs/<run_id>/`.
Each hemisphere has 32,492 labels. Provenance records consumed runs and inputs.

### PDC and BANDA

These presets use HCP MSMAll clean CIFTIs from exact official
`metadata/datastructure_manifest.txt` declarations. They have no fMRIPrep
22P/36P selector. The complete cortex roster is 455 PDC visits (984 runs) and
202 BANDA participants (808 runs); BANDA199 is explicitly excluded because its
source metadata has no run-level resting CIFTI.

PDC visits remain distinct (`e0011_02_MR` becomes `sub-e0011s02`);
`BANDA001_MR` becomes `sub-BANDA001`. PDC has one to four available rest runs,
BANDA four. Each declared run is one MATLAB session; the separately released
concatenated `rfMRI_REST` is excluded to avoid duplicate frames.

Build complete input manifests on Argon; the builder refuses replacement:

```bash
/Users/yojiao/miniconda3/envs/py313/bin/python scripts/ArealMSHBM_Kong2022/prepare_kong2022_areal_mshbm_subject_inputs.py --action build-cohort --dataset pdc --data-root /Shared/jianglab/3_Data_Working/Depression/PDC/imaging --manifest outputs_pdc/inputs/kong2022/manifests/pdc_cortex_inputs.json
/Users/yojiao/miniconda3/envs/py313/bin/python scripts/ArealMSHBM_Kong2022/prepare_kong2022_areal_mshbm_subject_inputs.py --action build-cohort --dataset banda --data-root /Shared/jianglab/3_Data_Working/Depression/BANDA/imaging --manifest outputs_banda/inputs/kong2022/manifests/banda_cortex_inputs.json
```

`COHORT_MANIFEST` selects the exact contract. Subject selectors must name
identities declared there. Tasks validate identity, variant, file metadata,
frame counts and both cortical brain models. Missing/stale/conflicting inputs
fail before MATLAB. These presets require `KEEP_TMP=1 OVERWRITE=0` and retain
persistent subject `inputs/` plus a unique CBIG work directory.

### Iowa7T variants

`DATASET=iowa7t` requires `IOWA7T_BRANCH=multiband|multi_echo`.
`IOWA7T_VARIANT=0p|22p|32p|36p|noxcpd` selects the matching
`DataOrg_PFM_0P`, `DataOrg_PFM_22P`, `DataOrg_PFM_32P`,
`DataOrg_PFM_36P` or `DataOrg_PFM_NoXCPD` tree; 36P is the default.

| Acquisition | Exact subject filename below `sub-<id>/func/` |
|---|---|
| Multiband 0P/22P/32P/36P | `sub-<id>_task-rest_dir-concat_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii` |
| Multi-echo 0P/22P/32P/36P | `sub-<id>_task-rest_run-concat_space-fsLR_den-91k_desc-MEICA_XCPD<0P\|22P\|32P\|36P>_denoised_bold.dtseries.nii` |
| Multiband NoXCPD | `sub-<id>_task-rest_dir-concat_space-fsLR_den-91k_desc-NoXCPD_bold.dtseries.nii` |
| Multi-echo NoXCPD | `sub-<id>_task-rest_run-concat_space-fsLR_den-91k_desc-MEICA_NoXCPD_bold.dtseries.nii` |

See [Iowa7T output layout](../../docs/iowa7t_36p_hipp_outputs_layout.md)
for production structure and [Iowa7T utilities](../iowa7t/README.md) for downstream
sampling, baseline tSNR and FC.

## Submit and inspect

The standard entrypoint is `submit_kong2022_areal_mshbm.sh`; it validates Bash
syntax and writes queue-specific wrappers that execute
`scripts/ArealMSHBM_Kong2022/run_kong2022_areal_mshbm_subject_sge.sh`.
It splits subjects deterministically between UI/PINC (ratio 1:2), requesting
`-pe smp 4` per task.

| System | UI throttle | PINC throttle | Combined cap |
|---|---|---|---|
| IDAS | 3 | 7 | 10 |
| Argon | 5 | 15 | 20 |

Account for other active batches. Use `HOLD_JID` with both upstream array IDs
to sequence cohorts; dependencies wait for completion, not scientific success.
Inspect outputs before treating a dependent run as valid.

Preview an explicit subject selection:

```bash
SYSTEM_NAME=argon DATASET=stanford STANFORD_VARIANT=22p SUBJECT_IDS_CSV=1001 KEEP_TMP=1 DRY_RUN=1 scripts/ArealMSHBM_Kong2022/submit_kong2022_areal_mshbm.sh
```

Use the same selection with `TASK_DRY_RUN=1` for task preflight without MATLAB.
A real smoke requires current-session submission authorization; a full batch
also requires a successful smoke with both label files. Remove `DRY_RUN=1`
only for the authorized operation. `KEEP_TMP=1` prevents automatic workdir cleanup.

| Control | Meaning |
|---|---|
| `SUBJECT_IDS_FILE`, `SUBJECT_IDS_CSV` | Explicit subject roster |
| `SMOKE_N` | Restrict the selected roster to N subjects |
| `STANFORD_VARIANT=22p\|36p` | Matching Stanford input/output variant |
| `ESTT_VARIANT=22p\|23p\|36p` | Matching ESTT input/output variant |
| `IOWA7T_BRANCH`, `IOWA7T_VARIANT` | Iowa7T acquisition and denoising selection |
| `DATA_ROOT`, `OUT_ROOT`, `COHORT_MANIFEST`, `HCP_RUN_METADATA_ROOT` | Explicit input/output contracts |
| `SYSTEM_NAME`, `UI_TC`, `PINC_TC`, `HOLD_JID` | Cluster, throttle and dependencies |
| `RUN_ID` | Manifest/log identifier |
| `WB_COMMAND` | Workbench path; default `software/workbench/bin_linux64/wb_command` |
| `DRY_RUN=1`, `TASK_DRY_RUN=1` | Submission preview or task input preflight |
| `OVERWRITE=1` | Explicit recomputation where supported, after exact-path approval |
| `KEEP_TMP=1` | Retain temporary work; required for PDC/BANDA |

Iowa7T manifests and logs use `${OUT_ROOT}/runs/manifests/${RUN_ID}` and
`${OUT_ROOT}/runs/logs/${RUN_ID}`. Other presets use `_manifests` and `_sge_logs`.
Task logs identify the exact subject manifest as `SubjectIDsFile=...`.
Use the same dataset/variant selectors for status reporting:

```bash
DATASET=stanford STANFORD_VARIANT=22p scripts/ArealMSHBM_Kong2022/report_kong2022_areal_mshbm_status.sh <job_id> sub-1001
```

## Iowa7T runwise interface

`submit_kong2022_areal_mshbm_runwise.sh` selects one subject and 22P or 36P.
Four separate calls use `run-01=ME01`, `run-02=ME02`, `run-03=MB03`,
`run-04=MB04`. They are not paired acquisitions or multi-session MATLAB calls.

```bash
SYSTEM_NAME=argon IOWA7T_VARIANT=22p SUBJECT_ID=<subject> RUN_TAG=<explicit_tag> DRY_RUN=1 scripts/ArealMSHBM_Kong2022/submit_kong2022_areal_mshbm_runwise.sh
```

The default array has four tasks with `-tc 4`. Its root is
`${DATA_ROOT}/kong2022_runwise_${RUN_TAG}/`, outside `hipp_outputs/`, with
`run-XX/{lh_labels.txt,rh_labels.txt,provenance.json}`, manifests, logs and
`derived_inputs/`. ME inputs become cortex-only CIFTIs; MB inputs use their
run-level 91k CIFTIs. Each MATLAB call receives exactly one run-list entry and
one censor-list entry pointing to an all-ones per-frame censor file.
Select this exact run tag again for the [downstream adapter](../iowa7t/README.md#downstream-stages).

## Runtime and retained work

Workbench is passed through qsub and prepended to PATH. MATLAB runs with
`-singleCompThread`; OMP, MKL, OpenBLAS, NumExpr and vecLib limits are one,
with the documented oneMKL compatibility setting. MATLAB/mapalign use their
existing deterministic initializations; these settings do not imply byte
identity across runtimes or input versions.

ESTT retains its subject CBIG work; HCP7T keeps task work below repository
`tmp/`. `KEEP_TMP=1` is the default for HCP7T, Iowa HCP500, PDC and BANDA.
Other presets may clean temporary work by default, so pass `KEEP_TMP=1`
unless exact cleanup paths have been explicitly approved.

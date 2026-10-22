# Kong2022 ArealMSHBM

This is the single maintained interface for subject-level Kong2022_ArealMSHBM
parcellation on JJ_TMS, Stanford, Iowa7T, ESTT, Iowa HCP500, HCP7T, PDC, and BANDA. Dataset
selection, input adaptation, SGE submission, MATLAB execution, provenance, and
status reporting live in this directory. It does not modify the
Kong2022_ArealMSHBM algorithm logic.

## Datasets

Select the dataset with `DATASET=jj_tms`, `DATASET=stanford`, `DATASET=iowa7t`,
`DATASET=estt`, `DATASET=iowa_hcp500`, `DATASET=hcp7t`, `DATASET=pdc`, or `DATASET=banda`. ESTT and Iowa7T are separate dataset
branches with separate input/output roots; they are not aliases of one another.
If unset,
`DATASET=jj_tms` is used for backward compatibility.

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

JJ_TMS and Stanford tasks scan one `sub-*` directory for the dataset's rest
dtseries pattern. PDC/BANDA tasks consume only metadata-declared exact paths.
The resulting fsLR sessions are passed to
`CBIG_ArealMSHBM_parcellation_single_subject` in `params.lh_fMRI_list`, with a
matching all-ones censor file in `params.censor_list`. JJ_TMS and Stanford
source runs are not merged before MATLAB. ESTT is the explicit compatibility
exception: its single concatenated CIFTI is split into four sessions to match
the established ESTT cohort contract.
HCP7T reads the exact concatenated CIFTI and physical-run metadata, validates
the declared run order and frame counts, then splits the CIFTI into four
sessions using those recorded lengths before MATLAB. It writes individualized
labels under the formal HCP PFM cortex-reference root and records the source
CIFTI, run metadata, run ranges, and session files in provenance. Existing
HCP7T labels or provenance cause the launcher to fail closed.
The `iowa_hcp500` task resolves its fixed HCP REST path directly and obtains
the timepoint count with `wb_command -file-information`; it does not require a
Python runtime on Argon. The `iowa7t` task also resolves its fixed branch CIFTI
path directly and obtains the timepoint count with `wb_command -file-information`;
it does not require a Python runtime on Argon. Existing JJ_TMS and Stanford
presets continue to use the Python input preparer. ESTT uses the Argon
`hippo_nnunet` Python runtime for label-asset materialization.

### PDC and BANDA

These presets consume the released HCP `MSMAll_hp0_clean` CIFTIs directly;
they are not fMRIPrep 22P/36P branches. Both datasets use their official
`metadata/datastructure_manifest.txt` as the authoritative input source.
The complete cortex roster is 455 PDC
visits (984 runs) and 202 BANDA participants (808 runs). BANDA199 has no
run-level resting CIFTI in the source metadata and is explicitly recorded in
`excluded_subjects`; the structural HippUnfold roster is not a substitute for
the cortex roster.

PDC visits retain separate outputs: `e0011_02_MR` maps to `sub-e0011s02`,
with the original session, base subject and visit recorded in provenance.
`BANDA001_MR` maps to `sub-BANDA001`. Each declared `rfMRI_REST1_AP`,
`rfMRI_REST1_PA`, `rfMRI_REST2_AP`, or `rfMRI_REST2_PA` is one MATLAB session.
PDC visits have one to four available runs; BANDA participants have four.
The separately released concatenated `rfMRI_REST` is excluded to avoid
counting the same frames twice. No run splitting, task fMRI, extra denoising,
subject surface requirement, or Kong2022 model change is introduced.

Build the complete input contracts on Argon:

```bash
/Users/yojiao/miniconda3/envs/py313/bin/python scripts/ArealMSHBM_Kong2022/prepare_kong2022_areal_mshbm_subject_inputs.py --action build-cohort \
  --dataset pdc --data-root /Shared/jianglab/3_Data_Working/Depression/PDC/imaging \
  --manifest /old_Users/yojiao/hipp/outputs_pdc/inputs/kong2022/manifests/pdc_cortex_inputs.json
/Users/yojiao/miniconda3/envs/py313/bin/python scripts/ArealMSHBM_Kong2022/prepare_kong2022_areal_mshbm_subject_inputs.py --action build-cohort \
  --dataset banda --data-root /Shared/jianglab/3_Data_Working/Depression/BANDA/imaging \
  --manifest /old_Users/yojiao/hipp/outputs_banda/inputs/kong2022/manifests/banda_cortex_inputs.json
```

The builder refuses to replace an existing manifest. It checks every selected
file's identity against source metadata and records exact paths, on-disk sizes,
modification times, source metadata hashes and
identity mappings. Tasks fail on missing, duplicate, stale, misidentified or
wrong-variant declarations. CIFTI headers must contain both 32,492-vertex
cortical models; consumed asset evidence is recorded for every run.

Preview the shared launchers before requesting production approval:

```bash
SYSTEM_NAME=argon DATASET=pdc KEEP_TMP=1 DRY_RUN=1 \
  scripts/ArealMSHBM_Kong2022/submit_kong2022_areal_mshbm.sh
SYSTEM_NAME=argon DATASET=banda KEEP_TMP=1 DRY_RUN=1 \
  HOLD_JID=pdc_k22_ui,pdc_k22_pinc \
  scripts/ArealMSHBM_Kong2022/submit_kong2022_areal_mshbm.sh
```

After approval, submit PDC first and set BANDA's `HOLD_JID` to both returned
PDC array IDs. Each cohort uses UI `-tc 5` plus PINC `-tc 15`; holding BANDA
until both PDC arrays finish keeps combined concurrency at 20 or below.
Check the live queue before submission. SGE dependencies wait for completion,
not scientific success; inspect each cohort's label/provenance outputs.
`SMOKE_N=1` selects one declared subject, while `TASK_DRY_RUN=1` validates
headers and prepared run/censor lists without starting MATLAB.

`COHORT_MANIFEST` selects an exact alternative manifest explicitly. A
`SUBJECT_IDS_FILE` or `SUBJECT_IDS_CSV` may select only identities declared
there; malformed or unknown identities fail. Per-subject products are
`lh_labels.txt`, `rh_labels.txt`, `provenance.json` and persistent `inputs/`
under the selected formal output root. Each hemisphere has 32,492 labels;
provenance records parcel counts, run identity, frame counts, input fingerprints
and source/cohort manifest hashes. Label/provenance files already present cause
an error for inspection, rather than reuse of an unvalidated cache.
PDC/BANDA requires `OVERWRITE=0` and `KEEP_TMP=1`; every task creates a unique
temporary CBIG project and retains it without automatic deletion.
The Argon runtime is the existing `py313` interpreter, nibabel, NumPy,
Workbench, MATLAB R2021a and CBIG; the local core-workflow py314 environment
is not required on Argon.

### Stanford version isolation

Use `STANFORD_VARIANT=22p` for the Stanford 22P cohort. The launcher then
selects `DataOrg_PFM_22P` and, unless `OUT_ROOT` is explicitly supplied, writes
to the separate sibling root `ArealMSHBM_Kong2022_22P`. `STANFORD_VARIANT=36p`
retains the existing unversioned 36P contract. If `DATA_ROOT` ends in
`DataOrg_PFM_22P` or `DataOrg_PFM_36P`, the variant is auto-detected; an explicit
variant that conflicts with that suffix is rejected.

Do not reuse the 36P output root for a 22P run. The subject label pair remains
flat under the selected output root:

```text
${OUT_ROOT}/sub-<id>/lh_labels.txt
${OUT_ROOT}/sub-<id>/rh_labels.txt
```

### ESTT version isolation

Use `ESTT_VARIANT=22p` for the new ESTT 22P cohort. The launcher selects
`/Shared/jianglab/3_Data_Working/fmriprep_processed_esTT/DataOrg_PFM_22P` and,
unless `OUT_ROOT` is explicitly supplied, writes to:

```text
/Shared/jianglab/3_Data_Working/fmriprep_processed_esTT/DataOrg_PFM_22P/hipp_outputs/cortex_kong2022_areal_mshbm/sub-<id>/
```

`ESTT_VARIANT=36p` retains the existing 36P input/output root. If
`ESTT_VARIANT=23p` selects the matching
`/Shared/jianglab/3_Data_Working/fmriprep_processed_esTT/DataOrg_PFM_23P` input
and output root:

```text
/Shared/jianglab/3_Data_Working/fmriprep_processed_esTT/DataOrg_PFM_23P/hipp_outputs/cortex_kong2022_areal_mshbm/sub-<id>/
```

If `DATA_ROOT` ends in `DataOrg_PFM_22P`, `DataOrg_PFM_23P`, or
`DataOrg_PFM_36P`, the variant is
auto-detected; an explicit variant that conflicts with that suffix is rejected.
Never use a different variant's ESTT `hipp_outputs` tree.

ESTT reads `wb_command -file-information -only-number-of-maps` separately for
every subject's selected concatenated CIFTI, then splits that subject's actual
timepoints into four balanced sessions. It does not reuse 36P timepoint counts;
the per-subject lengths are recorded in the input manifest and provenance.

For `DATASET=iowa7t`, `IOWA7T_BRANCH` is required and `IOWA7T_VARIANT` selects
the fMRIPrep denoising tree. `IOWA7T_VARIANT=36p` is the default. If
`DATA_ROOT` ends in `DataOrg_PFM_0P`, `DataOrg_PFM_22P`, `DataOrg_PFM_32P`, or
`DataOrg_PFM_NoXCPD`, the launcher auto-selects `0p`, `22p`, `32p`, or
`noxcpd`; callers can also set it explicitly.

With `IOWA7T_VARIANT=36p`, the branch controls the exact subject CIFTI consumed
by the task:

- `IOWA7T_BRANCH=multiband`:
  `sub-<ID>/func/sub-<ID>_task-rest_dir-concat_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii`
- `IOWA7T_BRANCH=multi_echo`:
  source CIFTI
  `sub-<ID>/func/sub-<ID>_task-rest_run-concat_space-fsLR_den-91k_desc-MEICA_XCPD36P_denoised_bold.dtseries.nii`
is split into left/right cortex metrics and rebuilt as a persistent
cortex-only fsLR32k dtseries before MATLAB.

For `DATASET=estt` with `ESTT_VARIANT=22p`, `ESTT_VARIANT=23p`, or
`ESTT_VARIANT=36p`, the
launcher consumes the selected version's concatenated 32k ESTT CIFTI,
splits its timepoints into four equal-duration sessions, and passes those four
sessions to the existing MATLAB wrapper. The ESTT branch retains its CBIG work
directory under `sub-<id>/kong2022_areal_mshbm/` and writes the completed
subject contract directly under:

```text
${OUT_ROOT}/sub-<id>/
```

It produces the established flat `lh_labels.txt`/`rh_labels.txt` files plus
`labels/*.func.gii`, `labels/*.label.gii`, the flat subject `.dlabel.nii`,
`provenance.json`, `summary.json`, `stage_manifest.json`, and
`matlab_wrapper.log`. These outputs are intentionally not written below the
Iowa7T `subjects/<id>/<branch>/cortex/areal_mshbm` tree.
The launcher treats an ESTT subject as complete only when both label text files,
`provenance.json`, `summary.json`, `stage_manifest.json`, and the flat dlabel
are present; otherwise a non-overwrite rerun resumes the materialization step
without rerunning MATLAB when the labels already exist.

With `IOWA7T_VARIANT=noxcpd`, the task consumes the NoXCPD equivalents:

- `IOWA7T_BRANCH=multiband`:
  `sub-<ID>/func/sub-<ID>_task-rest_dir-concat_space-fsLR_den-91k_desc-NoXCPD_bold.dtseries.nii`
- `IOWA7T_BRANCH=multi_echo`:
  source CIFTI
  `sub-<ID>/func/sub-<ID>_task-rest_run-concat_space-fsLR_den-91k_desc-MEICA_NoXCPD_bold.dtseries.nii`
  is split into left/right cortex metrics and rebuilt as a persistent
  cortex-only fsLR32k dtseries before MATLAB.

With `IOWA7T_VARIANT=0p`, `IOWA7T_VARIANT=22p`, or
`IOWA7T_VARIANT=32p`, the multiband CIFTI uses the same `desc-denoised_bold`
filename as 36P, and the multi-echo branch consumes the variant-specific MEICA
CIFTI:

- `IOWA7T_VARIANT=0p`, `IOWA7T_BRANCH=multi_echo`:
  `sub-<ID>/func/sub-<ID>_task-rest_run-concat_space-fsLR_den-91k_desc-MEICA_XCPD0P_denoised_bold.dtseries.nii`
- `IOWA7T_VARIANT=32p`, `IOWA7T_BRANCH=multi_echo`:
  `sub-<ID>/func/sub-<ID>_task-rest_run-concat_space-fsLR_den-91k_desc-MEICA_XCPD32P_denoised_bold.dtseries.nii`
- `IOWA7T_VARIANT=22p`, `IOWA7T_BRANCH=multi_echo`:
  `sub-<ID>/func/sub-<ID>_task-rest_run-concat_space-fsLR_den-91k_desc-MEICA_XCPD22P_denoised_bold.dtseries.nii`

### Iowa7T runwise rerun

For the new Iowa7T source tree that contains four separately named runs, use
`submit_kong2022_areal_mshbm_runwise.sh`. This is a separate launcher from the
standard branch launcher above. It makes four independent Kong2022 calls; it
does not pair `ME` and `MB` runs and does not pass two runs to MATLAB together:

```text
run-01 = ME01
run-02 = ME02
run-03 = MB03
run-04 = MB04
```

For subject `06112026`, the default output root is intentionally a shallow
sibling of the production `hipp_outputs` tree:

```text
/Shared/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_22P/kong2022_runwise_06112026_20260904/
├── run-01/
├── run-02/
├── run-03/
├── run-04/
├── manifests/
├── logs/
└── derived_inputs/
```

The `run-XX/` directories directly contain that run's `lh_labels.txt`,
`rh_labels.txt`, and `provenance.json`; there is no additional
`subjects/sub-06112026/` layer. `ME01` and `ME02` are converted from 91k to
cortex-only fsLR32k before MATLAB. `MB03` and `MB04` use their corresponding
run-level 91k CIFTI directly. Every MATLAB call receives one CIFTI and one
matching one-line censor list. The `censor_list.txt` entry points to the
run-level `censor.txt`, which contains one all-ones entry per CIFTI frame;
the list file must not contain the frame values themselves. The task validates
that `run_list.txt` and `censor_list.txt` each contain exactly one entry before
starting MATLAB.

Run a no-submit validation first:

```bash
SYSTEM_NAME=argon IOWA7T_VARIANT=22p SUBJECT_ID=06112026 \
  RUN_TAG=06112026_20260904 DRY_RUN=1 \
  scripts/ArealMSHBM_Kong2022/submit_kong2022_areal_mshbm_runwise.sh
```

The launcher defaults to one UI array with four tasks (`-t 1-4`) and
`-tc 4`. It refuses to reuse an existing `OUT_ROOT` unless `OVERWRITE=1` is
explicitly set, and it never routes this rerun into `hipp_outputs`.

### Runwise downstream stages

To generate the surface time series, cortex parcel time series, and FC matrices
for all four runs under `kong2022_runwise_${RUN_TAG}/run-XX/`:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/iowa7t/run_runwise_downstream.py \
  --subject 06112026 \
  --variant all \
  --run all \
  --stage all
```

The script automatically detects the latest `kong2022_runwise_<subject>_*` tree if `--run-tag` is omitted, or an explicit `--run-tag` can be specified.

### Standard Iowa7T output layout (unchanged)


Per-subject outputs are written directly under:

```bash
${OUT_ROOT}/sub-<id>
```

For `DATASET=iowa7t`, `OUT_ROOT` is the selected tree's production
`hipp_outputs` root and per-branch outputs are written under:

```bash
${OUT_ROOT}/subjects/${subject}/${iowa7t_branch}/cortex/areal_mshbm
```

The Iowa7T `multi_echo` branch writes its derived cortex-only input under:

```bash
${OUT_ROOT}/runs/derived_inputs/${RUN_ID}/subjects/${subject}/multi_echo/cortex_only
```

Key RouteA outputs:

- `lh_labels.txt`
- `rh_labels.txt`
- `provenance.json` after successful non-dry-run MATLAB completion

JJ_TMS, Stanford and Iowa7T keep CBIG working files and prepared input lists
under `/tmp/<dataset>_kong2022_<job>_<task>_<subject>` during execution.
PDC/BANDA keeps its input lists and manifests under each formal subject's
`inputs/` directory and uses a unique temporary CBIG project with a random
suffix. Automatic temporary-directory cleanup occurs only when `KEEP_TMP=0`.
ESTT deliberately retains its subject-level work directory because that is
part of the existing ESTT reproducibility contract.
For `iowa_hcp500` and `hcp7t`, `KEEP_TMP=1` is the default. HCP7T task work
files are retained under `${REPO_ROOT}/tmp/`. PDC/BANDA requires `KEEP_TMP=1`.
Set `KEEP_TMP=0` for a supported dataset only when deletion has been explicitly
approved.

The task script prepends `${REPO_ROOT}/software/workbench/bin_linux64` to
`PATH` and passes `WB_COMMAND` through qsub so CBIG/MATLAB calls to
`wb_command` resolve on compute nodes. Override with `WB_COMMAND=/path/to/wb_command`
only when needed.

For repeatable numerical execution, the task fixes OMP, MKL, OpenBLAS,
NumExpr, and vecLib to one thread, disables dynamic OMP/MKL thread changes,
selects the oneMKL `COMPATIBLE` conditional-numerical-reproducibility branch,
and starts MATLAB with `-singleCompThread`. The exported task environment is
inherited by the CBIG diffusion helper and its `py313` Python process. These
controls define the maintained runtime configuration; they do not claim
bitwise equivalence across different MATLAB, SciPy, BLAS, CPU, code, or input
versions.

The Kong2022 gradient helper itself seeds MATLAB's `randperm` path with
`rng(scan_idx, 'twister')`, and the vendored mapalign diffusion solver uses a
fixed all-ones eigenvector start. The duplicate-output smoke described below
is still the acceptance check for the complete label pipeline. The watcher can
record a mismatch and continue with the requested batch when
`PROCEED_ON_MISMATCH=1` (the default); set it to `0` to hold submission.

### 30-minute test-to-production watcher (archived 2026-09-14)

> Retired: `watch_stanford_22p_kong2022.sh` now lives at
> `_archive/deprecated_2026-09-14/ArealMSHBM_Kong2022/watch_stanford_22p_kong2022.sh`.
> The Stanford 22P formal batch + ESTT 22P chain it orchestrated is complete;
> the text below is audit history only.

The watcher is an external login-node process. It checks the duplicate test
every 30 minutes, compares both hemispheres byte-for-byte, and then submits the
full Stanford 22P cohort with the Argon cap (`UI_TC=5`, `PINC_TC=15`). By
default, after all 197 Stanford outputs are complete, it submits the ESTT 22P
cohort with the same cap, using the separate `DataOrg_PFM_22P/hipp_outputs`
root. If Stanford UI tasks are still running but PINC capacity is idle, set
`START_ESTT_EARLY=1 ESTT_UI_TC=5 ESTT_PINC_TC=10`; this starts ESTT early with
15 tasks so the 5 Stanford UI tasks plus 15 ESTT tasks stay within Argon's
20-task cap. The watcher records `ESTT_SUBMITTED` before resuming Stanford
monitoring, so the later completion check cannot submit a duplicate ESTT
batch. It keeps the test and production output roots separate and retains task
temporary files (`KEEP_TMP=1`). It cannot send a message to a completed chat
session; its log, hashes, and submission records are written under the test
`run2/_watcher` directory.

For the current test job, start it once from the Argon login node as follows:

```bash
mkdir -p /Shared/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/ArealMSHBM_Kong2022_22P_randomness_test/run2/_watcher
nohup env SYSTEM_NAME=argon TEST_JOB_ID=<run2-job-id> \
  WATCH_INTERVAL_SECONDS=1800 PROCEED_ON_MISMATCH=1 \
  FORMAL_RUN_ID=stanford_kong2022_22p_full_20260903 \
  WATCH_ROOT=/Shared/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/ArealMSHBM_Kong2022_22P_randomness_test/run2/_watcher \
  scripts/ArealMSHBM_Kong2022/watch_stanford_22p_kong2022.sh \
  > /Shared/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/ArealMSHBM_Kong2022_22P_randomness_test/run2/_watcher/watcher.log 2>&1 < /dev/null &
```

The watcher first emits the exact production dry-run command to
`formal_dry_run.txt`, then submits the 197-subject arrays. Once Stanford is
complete it emits `estt_dry_run.txt` and submits the ESTT 22P arrays. Its
`watcher.state`, `randomness_result.txt`, `label_hashes.sha256`,
`formal_submission.txt`, and `estt_submission.txt` files are the durable audit
trail.

The legacy shared ESTT launcher
`/Shared/jianglab/3_Data_Working/fmriprep_processed_esTT/src/hipp/run_lss_kong2022_mshbm_array_task.sh`
now invokes `scripts/ArealMSHBM_Kong2022/run_cortex_mshbm_subject.py`. That
compatibility shim delegates to the maintained `DATASET=estt` branch above, so
old array submissions and current dataset-aware submissions use one
implementation. There is no `scripts/cortex` implementation.

## Submit

Dry-run JJ_TMS without submitting:

```bash
DRY_RUN=1 scripts/ArealMSHBM_Kong2022/submit_kong2022_areal_mshbm.sh
```

Dry-run Stanford without submitting:

```bash
DATASET=stanford DRY_RUN=1 scripts/ArealMSHBM_Kong2022/submit_kong2022_areal_mshbm.sh
```

Dry-run Stanford 22P with one explicit smoke subject:

```bash
DATASET=stanford STANFORD_VARIANT=22p SUBJECT_IDS_CSV=1001 SMOKE_N=1 \
  DRY_RUN=1 scripts/ArealMSHBM_Kong2022/submit_kong2022_areal_mshbm.sh
```

Submit qsub tasks that validate Python, CBIG, and subject dtseries inputs, then
stop before MATLAB:

```bash
DATASET=stanford TASK_DRY_RUN=1 scripts/ArealMSHBM_Kong2022/submit_kong2022_areal_mshbm.sh
```

To test label reproducibility before the 22P batch, run the same subject twice
with the same code/input and two separate output roots, then compare both label
files byte-for-byte:

```bash
DATASET=stanford STANFORD_VARIANT=22p SUBJECT_IDS_CSV=1001 SMOKE_N=1 \
  KEEP_TMP=1 OUT_ROOT=/Shared/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/ArealMSHBM_Kong2022_22P_randomness_test/run1 \
  RUN_ID=stanford_kong2022_22p_randomness_run1 \
  scripts/ArealMSHBM_Kong2022/submit_kong2022_areal_mshbm.sh
DATASET=stanford STANFORD_VARIANT=22p SUBJECT_IDS_CSV=1001 SMOKE_N=1 \
  KEEP_TMP=1 OUT_ROOT=/Shared/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/ArealMSHBM_Kong2022_22P_randomness_test/run2 \
  RUN_ID=stanford_kong2022_22p_randomness_run2 \
  scripts/ArealMSHBM_Kong2022/submit_kong2022_areal_mshbm.sh
cmp /Shared/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/ArealMSHBM_Kong2022_22P_randomness_test/run1/sub-1001/lh_labels.txt \
    /Shared/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/ArealMSHBM_Kong2022_22P_randomness_test/run2/sub-1001/lh_labels.txt
cmp /Shared/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/ArealMSHBM_Kong2022_22P_randomness_test/run1/sub-1001/rh_labels.txt \
    /Shared/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/ArealMSHBM_Kong2022_22P_randomness_test/run2/sub-1001/rh_labels.txt
```

### Repair missing Stanford 22P provenance (archived 2026-09-14)

> Retired: `repair_kong2022_areal_mshbm_provenance_sge.sh` and
> `repair_kong2022_areal_mshbm_provenance.py` now live under
> `_archive/deprecated_2026-09-14/ArealMSHBM_Kong2022/`. The text below is audit
> history for the missing-provenance backlog repair.

If an existing Stanford 22P subject has both validated label text files but no
`provenance.json`, use the dedicated repair task. It reconstructs the subject
input manifest on an Argon compute node, validates the existing 32,492-value
left/right labels, and writes provenance only; it does not rerun MATLAB or
overwrite labels. The repair script refuses 36P data/output roots and requires
an explicit subject list.

First run one repair smoke task (the original source task for `sub-1053` was
Stanford UI array `6453867`, task `10`):

```bash
qsub -N k22_prov_test -q PINC -t 1-1 -tc 1 -pe smp 1 -l mem_free=8G -l h_rt=01:00:00 -j y \
  -o /Shared/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/ArealMSHBM_Kong2022_22P/_sge_logs/provenance_repair_test.\$JOB_ID.\$TASK_ID.log \
  -v SYSTEM_NAME=argon,REPO_ROOT=/old_Users/yojiao/hipp,DATA_ROOT=/Shared/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/DataOrg_PFM_22P,OUT_ROOT=/Shared/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/ArealMSHBM_Kong2022_22P,SUBJECT_IDS_CSV=1053,SOURCE_QSUB_MAP=1053=6453867:10 \
  scripts/ArealMSHBM_Kong2022/repair_kong2022_areal_mshbm_provenance_sge.sh
```

After the smoke task succeeds, submit the remaining missing-provenance
subjects with a throttle that fits the active scheduler workload. The repair
tasks are metadata-only and must still obey the Argon 20-task cap; do not use
the normal 20-task setting if ESTT tasks are active.

Smoke one real subject:

```bash
DATASET=stanford SMOKE_N=1 scripts/ArealMSHBM_Kong2022/submit_kong2022_areal_mshbm.sh
```

Iowa HCP500 single-subject launcher dry-run, task preflight, and real smoke:

```bash
DATASET=iowa_hcp500 SUBJECT_IDS_CSV=04172026 DRY_RUN=1 \
  scripts/ArealMSHBM_Kong2022/submit_kong2022_areal_mshbm.sh
DATASET=iowa_hcp500 SUBJECT_IDS_CSV=04172026 TASK_DRY_RUN=1 \
  scripts/ArealMSHBM_Kong2022/submit_kong2022_areal_mshbm.sh
DATASET=iowa_hcp500 SUBJECT_IDS_CSV=04172026 \
  scripts/ArealMSHBM_Kong2022/submit_kong2022_areal_mshbm.sh
```

ESTT 36P dry-run for an explicit subject list:

```bash
DATASET=estt SUBJECT_IDS_CSV=477,559,622,714,717,721,741,771,789,809,841,913 \
  RUN_ID=estt_kong2022_new_20260823 DRY_RUN=1 \
  scripts/ArealMSHBM_Kong2022/submit_kong2022_areal_mshbm.sh
```

ESTT 22P dry-run for the complete currently available cohort:

```bash
DATASET=estt ESTT_VARIANT=22p SYSTEM_NAME=argon \
  RUN_ID=estt_kong2022_22p_full_20260903 DRY_RUN=1 \
  scripts/ArealMSHBM_Kong2022/submit_kong2022_areal_mshbm.sh
```

ESTT 23P dry-run for the complete 38-subject cohort:

```bash
DATASET=estt ESTT_VARIANT=23p SYSTEM_NAME=argon \
  RUN_ID=estt_kong2022_23p_full_20260908 DRY_RUN=1 \
  scripts/ArealMSHBM_Kong2022/submit_kong2022_areal_mshbm.sh
```

Iowa7T 36P branch dry-runs for the two requested production subjects:

```bash
DATASET=iowa7t IOWA7T_BRANCH=multiband SUBJECT_IDS_CSV=06112026,06122026 \
  RUN_ID=iowa7t_36p_single_06112026_06122026 DRY_RUN=1 \
  scripts/ArealMSHBM_Kong2022/submit_kong2022_areal_mshbm.sh
DATASET=iowa7t IOWA7T_BRANCH=multi_echo SUBJECT_IDS_CSV=06112026,06122026 \
  RUN_ID=iowa7t_36p_multi_06112026_06122026 DRY_RUN=1 \
  scripts/ArealMSHBM_Kong2022/submit_kong2022_areal_mshbm.sh
```

Iowa7T NoXCPD branch dry-runs for the two requested production subjects:

```bash
DATASET=iowa7t IOWA7T_VARIANT=noxcpd IOWA7T_BRANCH=multiband \
  SUBJECT_IDS_CSV=06112026,06122026 RUN_ID=iowa7t_noxcpd_single_06112026_06122026 \
  DRY_RUN=1 scripts/ArealMSHBM_Kong2022/submit_kong2022_areal_mshbm.sh
DATASET=iowa7t IOWA7T_VARIANT=noxcpd IOWA7T_BRANCH=multi_echo \
  SUBJECT_IDS_CSV=06112026,06122026 RUN_ID=iowa7t_noxcpd_multi_06112026_06122026 \
  DRY_RUN=1 scripts/ArealMSHBM_Kong2022/submit_kong2022_areal_mshbm.sh
```

Iowa7T 0P and 32P single-subject branch dry-runs:

```bash
DATASET=iowa7t IOWA7T_VARIANT=0p IOWA7T_BRANCH=multiband \
  SUBJECT_IDS_CSV=06122026 RUN_ID=iowa7t_0p_multiband_06122026 DRY_RUN=1 \
  scripts/ArealMSHBM_Kong2022/submit_kong2022_areal_mshbm.sh
DATASET=iowa7t IOWA7T_VARIANT=0p IOWA7T_BRANCH=multi_echo \
  SUBJECT_IDS_CSV=06122026 RUN_ID=iowa7t_0p_multi_echo_06122026 DRY_RUN=1 \
  scripts/ArealMSHBM_Kong2022/submit_kong2022_areal_mshbm.sh
DATASET=iowa7t IOWA7T_VARIANT=32p IOWA7T_BRANCH=multiband \
  SUBJECT_IDS_CSV=06122026 RUN_ID=iowa7t_32p_multiband_06122026 DRY_RUN=1 \
  scripts/ArealMSHBM_Kong2022/submit_kong2022_areal_mshbm.sh
DATASET=iowa7t IOWA7T_VARIANT=32p IOWA7T_BRANCH=multi_echo \
  SUBJECT_IDS_CSV=06122026 RUN_ID=iowa7t_32p_multi_echo_06122026 DRY_RUN=1 \
  scripts/ArealMSHBM_Kong2022/submit_kong2022_areal_mshbm.sh
```

Use `DRY_RUN=1` with any example above to print qsub commands only.

The default batch scans all subjects under `DATA_ROOT`. Before submission, the
launcher identifies the active system from `SYSTEM_NAME`, `CLUSTER_NAME`,
`SGE_CLUSTER_NAME`, or the hostname. It rejects unknown systems and rejects
task-script syntax errors or throttle overrides above the detected cap. It
runs `bash -n` on the canonical task script before creating or submitting the
arrays:

- deterministic UI/PINC split by subject index, ratio 1:2
- each qsub task requests `-pe smp 4`
- IDAS: `UI -tc 3` plus `PINC -tc 7`, capping total concurrency at 10
- Argon: `UI -tc 5` plus `PINC -tc 15`, capping total concurrency at 20

Runtime subject lists and scheduler logs are written under `${OUT_ROOT}/_manifests`
and `${OUT_ROOT}/_sge_logs` for legacy presets, including ESTT; for
`DATASET=iowa7t`, they are
written under `${OUT_ROOT}/runs/manifests/${RUN_ID}` and
`${OUT_ROOT}/runs/logs/${RUN_ID}`. The script directory should contain source files
only. The launcher writes a queue-specific qsub wrapper under the manifest root that
exports the selected subject manifest before executing the task script
`scripts/ArealMSHBM_Kong2022/run_kong2022_areal_mshbm_subject_sge.sh`. Task
logs show the consumed manifest as `SubjectIDsFile=...`; verify this path when
auditing a batch.

## Controls

- `DATASET=jj_tms|stanford|iowa7t|estt|iowa_hcp500|hcp7t`: dataset preset and default path selector.
- `IOWA7T_BRANCH=multiband|multi_echo`: required with `DATASET=iowa7t`.
- `DATA_ROOT`: override the selected dataset input root.
- `OUT_ROOT`: override the selected dataset output root.
- `HCP_RUN_METADATA_ROOT`: HCP7T physical-run metadata root; defaults to
  `${REPO_ROOT}/outputs_hcp/inputs/qc/run_metadata`.
- `STANFORD_VARIANT=22p|36p`: Stanford input/output version selector; it is
  auto-detected from a standard `DATA_ROOT` suffix and rejects known suffix
  conflicts.
- `ESTT_VARIANT=22p|23p|36p`: ESTT input/output version selector; it is
  auto-detected from a standard `DATA_ROOT` suffix and rejects known suffix
  conflicts.
- `RUN_ID`: run identifier for manifest/log roots; Iowa7T defaults to branch plus timestamp.
- `IOWA7T_VARIANT=0p|22p|32p|36p|noxcpd`: Iowa7T input naming/root selector;
  defaults to `36p` unless `DATA_ROOT` basename identifies another variant.
- `WB_COMMAND`: override Workbench command path; default is
  `${REPO_ROOT}/software/workbench/bin_linux64/wb_command`.
- `SMOKE_N=N`: submit only the first N selected subjects.
- `SUBJECT_IDS_CSV`: comma-separated subject IDs.
- `SUBJECT_IDS_FILE`: subject manifest, one ID per line.
- `SYSTEM_NAME=idas|argon`: override system detection when the scheduler
  environment does not identify the cluster unambiguously.
- `UI_TC=N`: UI array throttle; defaults to `3` on IDAS and `5` on Argon.
- `PINC_TC=N`: PINC array throttle; defaults to `7` on IDAS and `15` on Argon.
  The sum must stay within the detected system cap (10 on IDAS, 20 on Argon).
- Watcher-only controls: `START_ESTT_EARLY=1` submits ESTT before Stanford
  completion; `ESTT_UI_TC` and `ESTT_PINC_TC` set its separate throttles. Early
  ESTT requires their sum to be at most 15 because Stanford UI retains `-tc 5`.
  `ESTT_EXISTING_JOB_IDS="<ui-id> <pinc-id>"` is a recovery control for
  attaching the watcher to arrays that were already submitted before a watcher
  restart; it records the IDs without submitting a duplicate batch.
- `TASK_DRY_RUN=1`: task preflight only, no MATLAB.
- `DRY_RUN=1`: launcher dry-run only, no qsub submit.
- `OVERWRITE=1`: recompute existing outputs.
- `KEEP_TMP=1`: retain task temporary files; default for `iowa_hcp500` and
  `hcp7t`.

## Status

After a submitted smoke or batch job, summarize scheduler state, logs, and
expected label text files with:

```bash
DATASET=stanford scripts/ArealMSHBM_Kong2022/report_kong2022_areal_mshbm_status.sh <job_id> sub-1001
```

For Stanford 22P, include the version so status points to the isolated output
root:

```bash
DATASET=stanford STANFORD_VARIANT=22p \
  scripts/ArealMSHBM_Kong2022/report_kong2022_areal_mshbm_status.sh <job_id> sub-1001
```

For Iowa HCP500:

```bash
DATASET=iowa_hcp500 scripts/ArealMSHBM_Kong2022/report_kong2022_areal_mshbm_status.sh <job_id> sub-04172026
```

For HCP7T:

```bash
DATASET=hcp7t scripts/ArealMSHBM_Kong2022/report_kong2022_areal_mshbm_status.sh <job_id> sub-102816
```

For Iowa7T 36P:

```bash
DATASET=iowa7t IOWA7T_BRANCH=multiband \
  scripts/ArealMSHBM_Kong2022/report_kong2022_areal_mshbm_status.sh <job_id> sub-06112026
```

For Iowa7T NoXCPD:

```bash
DATASET=iowa7t IOWA7T_VARIANT=noxcpd IOWA7T_BRANCH=multiband \
  scripts/ArealMSHBM_Kong2022/report_kong2022_areal_mshbm_status.sh <job_id> sub-06112026
```

Only submit a full batch after a real smoke run succeeds and both label text
files are present for the smoke subject.

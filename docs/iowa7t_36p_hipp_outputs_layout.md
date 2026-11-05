# Iowa7T 36P Hippocampal Output Layout

Canonical production roots:

- 0P local: `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_0P/hipp_outputs`
- 0P Argon: `/Shared/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_0P/hipp_outputs`
- 32P local: `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_32P/hipp_outputs`
- 32P Argon: `/Shared/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_32P/hipp_outputs`
- 36P local: `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P/hipp_outputs`
- 36P Argon: `/Shared/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P/hipp_outputs`
- NoXCPD local: `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_NoXCPD/hipp_outputs`
- NoXCPD Argon: `/Shared/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_NoXCPD/hipp_outputs`

Rules:

- Iowa7T means the Jianglab fMRIPrep Iowa7T trees, not HCPpipeline500.
- `36p` remains the default workflow variant for backward compatibility.
- `0p`, `32p`, and `noxcpd` use parallel `DataOrg_PFM_*` trees and must not write into the 36P output root.
- `outputs_iowa_estt/` and `outputs_iowa_jj_tms/` in the repository are derived-output roots for ESTT (3T) and JJ_TMS (3T) respectively, not Iowa 7T production output roots.
- Subject structural HippUnfold output is shared at `subjects/sub-<ID>/hippunfold/` (non-36P variants symlink to the shared 36P HippUnfold output tree via relative symlink `../../../../DataOrg_PFM_36P/hipp_outputs/subjects/sub-<ID>/hippunfold`).
- Functional branches are direct siblings: `multiband/` and `multi_echo/`.
- There is no `variants/` wrapper.
- Do not pre-create empty future directories such as `group/`.
- Create directories only as parents for concrete files/symlinks/manifests being written.
- Migration skips filesystem and notebook checkpoint junk: `.DS_Store`, `._*`, and `.ipynb_checkpoints/`.
- Do not delete legacy paths until the exact deletion list is approved in the current session.
- HippUnfold runs must use the shared asset store hardcoded in the wrappers; do not create subject/run-specific `HIPPUNFOLD_CACHE_DIR` or `HIPPUNFOLD_RUNTIME_SOURCE_CACHE` paths.

Current canonical subject layout:

```text
hipp_outputs/
└── subjects/
    └── sub-<ID>/
        ├── hippunfold/
        ├── multiband/
        │   ├── cortex/
        │   │   └── areal_mshbm/
        │   │       ├── lh_labels.txt
        │   │       ├── rh_labels.txt
        │   │       └── provenance.json
        │   ├── surface/
        │   ├── fc/
        │   ├── qc/
        │   ├── figures/
        │   └── provenance.json
        └── multi_echo/
            ├── cortex/
            │   └── areal_mshbm/
            │       ├── lh_labels.txt
            │       ├── rh_labels.txt
            │       └── provenance.json
            ├── surface/
            ├── fc/
            ├── qc/
            ├── figures/
            └── provenance.json
```

Use `scripts.core_workflow.lib.output_layout` as the path authority:

```python
from scripts.core_workflow.lib.output_layout import (
    iowa7t_branch_layout,
    iowa7t_subject_layout,
)

subject = iowa7t_subject_layout("06112026")
single = iowa7t_branch_layout("06112026", "multiband")
multi = iowa7t_branch_layout("06112026", "multi_echo")
noxcpd = iowa7t_branch_layout("06112026", "multi_echo", variant="noxcpd")
```

## Explicit run metadata for branch downstream

`run_branch_downstream.py` requires `--run-metadata-template` for `all`,
`reference`, and `fc` stages. The selected per-subject JSON is the only source
of run IDs and boundaries; the launcher never infers runs from filenames or
assumes four equal runs. A minimal receipt is:

```json
{"run_ids": [1, 2, 3, 4], "run_lengths": [230, 228, 228, 230], "concat_timepoints": 916}
```

The template supports `{data_root}`, `{output_root}`, `{subject}`, and
`{sub_subject}`. The metadata file must exist before the command starts and its
run summaries must agree with the declared IDs and lengths. `surface` alone does
not need run metadata.

Migration helper:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/iowa7t/migrate_hipp_outputs_layout.py --subjects 04172026 --mode dry-run
/home/yojiao/.conda/envs/py314/bin/python scripts/iowa7t/migrate_hipp_outputs_layout.py --subjects 04172026 --mode copy
```

`dry-run` writes only a concrete migration manifest under `runs/manifests/`.
`copy` copies or recreates symlinks into canonical locations and leaves legacy
paths in place. Deletion is a separate user-approved operation.
Migration preserves structural, surface, cortex-reference, FC, and group
assets. Individual-clustering labels, figures, overview-adapter caches, and
their run metadata are excluded; the migration does not delete those files.

## 0P and 32P Variants

0P and 32P are first-class Iowa7T variants. Local wrappers accept
`--variant 0p` and `--variant 32p` and resolve defaults to
`DataOrg_PFM_0P/hipp_outputs` and `DataOrg_PFM_32P/hipp_outputs`.

0P/32P branch inputs are fixed:

- `multiband` consumes the same `desc-denoised` MNI volume and 91k dtseries
  filename pattern as 36P.
- `0p` `multi_echo` consumes `desc-MEICA_XCPD0P_denoised` MNI volume and 91k
  dtseries files.
- `32p` `multi_echo` consumes `desc-MEICA_XCPD32P_denoised` MNI volume and
  91k dtseries files.

Like NoXCPD, 0P and 32P use independent HippUnfold subject trees under their
selected `hipp_outputs/subjects/sub-<ID>/hippunfold/`. Use an isolated BIDS
staging root for anatomical inputs before running HippUnfold so SnakeBIDS does
not scan existing `hipp_outputs` as inputs.

Example 06122026 downstream and FC heatmap runs:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/iowa7t/run_branch_downstream.py --variant 0p --job 06122026:multiband --job 06122026:multi_echo --stage all --run-metadata-template '/absolute/path/to/run_metadata/{sub_subject}.json' --overwrite
/home/yojiao/.conda/envs/py314/bin/python scripts/iowa7t/run_branch_downstream.py --variant 32p --job 06122026:multiband --job 06122026:multi_echo --stage all --run-metadata-template '/absolute/path/to/run_metadata/{sub_subject}.json' --overwrite

/home/yojiao/.conda/envs/py314/bin/python scripts/iowa7t/generate_all_subject_plots.py --variant 0p --variant 32p --subject 06122026 --branch multiband --branch multi_echo
```

## NoXCPD Variant

NoXCPD is a first-class Iowa7T variant. Local wrappers accept
`--variant noxcpd` and resolve defaults to
`DataOrg_PFM_NoXCPD/hipp_outputs`; 36P remains the default when `--variant` is
omitted.

NoXCPD branch inputs are fixed:

- `multiband` consumes `desc-NoXCPD` MNI volume and 91k dtseries files.
- `multi_echo` consumes `desc-MEICA_NoXCPD` MNI volume and 91k dtseries files.

NoXCPD uses an independent HippUnfold subject tree by default:
`DataOrg_PFM_NoXCPD/hipp_outputs/subjects/sub-<ID>/hippunfold/`. Do not let
NoXCPD production wrappers fall back to the 36P subject tree. Override with
`--hippunfold-batch-root` only when intentionally testing an explicit alternate
subject tree.

The closed local HippUnfold stage deliberately has no NoXCPD preset. Do not
invoke `scripts/core_workflow/01_01_run_hippunfold_t1w_local.py` with custom
staging or output paths: its positional path interface was retired after it
allowed batches to diverge. NoXCPD production runs require a separately reviewed
dataset preset before they can use the canonical stage-01 entrypoint.

Example downstream rerun:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/iowa7t/run_branch_downstream.py   --variant noxcpd   --job 06112026:multiband   --job 06112026:multi_echo   --job 06122026:multiband   --job 06122026:multi_echo   --run-metadata-template '/absolute/path/to/run_metadata/{sub_subject}.json'
```

Each NoXCPD branch must produce its own surface, cortex-reference, and FC
outputs under the selected NoXCPD root. Individual-clustering atlas and
overview products are not branch completion requirements.

## Cortex ArealMSHBM

Iowa7T 36P cortex individualized labels use the Argon
`scripts/ArealMSHBM_Kong2022` launcher with
`DATASET=iowa7t` and an explicit `IOWA7T_BRANCH`.

Branch inputs are fixed:

- `multiband` consumes
  `sub-<ID>/func/sub-<ID>_task-rest_dir-concat_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii`
- `multi_echo` consumes
  source CIFTI
  `sub-<ID>/func/sub-<ID>_task-rest_run-concat_space-fsLR_den-91k_desc-MEICA_XCPD36P_denoised_bold.dtseries.nii`;
  the Argon launcher derives a persistent cortex-only fsLR32k CIFTI under
  `hipp_outputs/runs/derived_inputs/<run_id>/subjects/sub-<ID>/multi_echo/cortex_only/`
  and passes that derived file to MATLAB

Branch outputs are fixed:

- `subjects/sub-<ID>/multiband/cortex/areal_mshbm/`
- `subjects/sub-<ID>/multi_echo/cortex/areal_mshbm/`

Each completed branch writes `lh_labels.txt`, `rh_labels.txt`, and
`provenance.json` in that `areal_mshbm` directory. The provenance records the
input CIFTI path, output directory, qsub job/task IDs, MATLAB script, model
parameters, CIFTI timepoint count, and elapsed runtime.

For Argon submissions, qsub logs belong under:

```text
hipp_outputs/runs/logs/<run_id>/
```

Do not route these jobs to HCP500 paths, repo-local `outputs_iowa*` roots, or a
`variants/` wrapper.

## Four-run independent Kong2022 rerun

The runwise Iowa7T launcher is an isolated rerun contract for source trees that
contain `ME01`, `ME02`, `MB03`, and `MB04` subject directories. It runs each
source run independently; `ME` and `MB` inputs are not paired into two-session
MATLAB calls:

```text
run-01 = ME01
run-02 = ME02
run-03 = MB03
run-04 = MB04
```

For `sub-06112026` on Argon, the default 36P root is:

```text
/Shared/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P/kong2022_runwise_06112026_20260904/
├── run-01/
├── run-02/
├── run-03/
├── run-04/
├── manifests/
├── logs/
└── derived_inputs/
```

Each `run-XX/` directory directly contains the two label text files and its
provenance record. The root deliberately has no `subjects/` or
`sub-06112026/` wrapper and is separate from
`DataOrg_PFM_36P/hipp_outputs`.

Use the dedicated launcher and inspect its printed qsub command before any
submission:

```bash
SYSTEM_NAME=argon IOWA7T_VARIANT=36p SUBJECT_ID=06112026 \
  RUN_TAG=06112026_20260904 DRY_RUN=1 \
  scripts/ArealMSHBM_Kong2022/submit_kong2022_areal_mshbm_runwise.sh
```

### Runwise downstream stages (surface, cortex, FC)

After runwise Kong2022 labels are complete, run the downstream processor to
extract hippocampal surface timeseries, cortex ROI parcel timeseries, and
compute hippocampal-cortical functional connectivity matrices for each run
directly under its corresponding `run-XX/` directory:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/iowa7t/run_runwise_downstream.py \
  --subject 06112026 \
  --variant all \
  --run all \
  --stage all
```

The script automatically detects the latest `kong2022_runwise_<subject>_*` tree if `--run-tag` is omitted, or an explicit `--run-tag` (e.g. `--run-tag 06122026_20260906`) can be specified.

Output tree per run:
- `run-XX/cortex/`: `cortex_roi_parcel_timeseries.npy` (400 × T), `cortex_roi_parcels.tsv`, `cortex_canonical7_network_timeseries.npy` (7 × T), `cortex_family17_network_timeseries.npy` (17 × T)
- `run-XX/surface/`: `raw/`, `validity/`, `2mm/` and `4mm/` smoothed hippocampal timeseries (`sub-<subject>_hemi-{L,R}_timeseries.npy`)
- `run-XX/fc/`: `2mm/` and `4mm/` FC matrices (`hipp_vertex_to_network_fc.npy`, `hipp_vertex_to_network_fc_active.npy`, and `hippo512_to_kong400_corr.npy`)


## Individual Hippocampal Labels

Individual labels must come from matching to a declared group-trained
template. Never fit clustering to an individual hippocampus, including
through a group entrypoint supplied with a single-subject cohort. Group
clustering and training-cohort clustering within cross-validation remain
supported; held-out individuals receive labels only by template matching.

The branch downstream entrypoints documented here produce surface, cortex,
and FC products. They do not produce individual parcellation labels. The
retained HCP family-aware cross-validation and template-matching workflow is
documented in `docs/hipp_parcellation_network/crossfit_template_matching.md`;
non-HCP template-matching adapters belong under `scripts/<dataset>/`.

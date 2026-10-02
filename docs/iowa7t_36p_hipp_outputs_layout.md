# Iowa7T 36P Hippocampal Output Layout

Canonical production roots:

- 0P local: `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_0P/hipp_outputs`
- 0P Argon: `/Shared/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_0P/hipp_outputs`
- 22P local: `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_22P/hipp_outputs`
- 22P Argon: `/Shared/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_22P/hipp_outputs`
- 32P local: `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_32P/hipp_outputs`
- 32P Argon: `/Shared/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_32P/hipp_outputs`
- 36P local: `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P/hipp_outputs`
- 36P Argon: `/Shared/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P/hipp_outputs`
- NoXCPD local: `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_NoXCPD/hipp_outputs`
- NoXCPD Argon: `/Shared/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_NoXCPD/hipp_outputs`

Rules:

- Iowa7T means the Jianglab fMRIPrep Iowa7T trees, not HCPpipeline500.
- `36p` remains the default workflow variant for backward compatibility.
- `0p`, `22p`, `32p`, and `noxcpd` use parallel `DataOrg_PFM_*` trees and must not write into the 36P output root.
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
        │   ├── atlas/
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
            ├── atlas/
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

## 0P, 22P, and 32P Variants

0P, 22P, and 32P are first-class Iowa7T variants. Local wrappers accept
`--variant 0p`, `--variant 22p`, and `--variant 32p` and resolve defaults to
`DataOrg_PFM_0P/hipp_outputs`, `DataOrg_PFM_22P/hipp_outputs`, and `DataOrg_PFM_32P/hipp_outputs`.

0P/22P/32P branch inputs are fixed:

- `multiband` consumes the same `desc-denoised` MNI volume and 91k dtseries
  filename pattern as 36P.
- `0p` `multi_echo` consumes `desc-MEICA_XCPD0P_denoised` MNI volume and 91k
  dtseries files.
- `22p` `multi_echo` consumes `desc-MEICA_XCPD22P_denoised` MNI volume and 91k
  dtseries files.
- `32p` `multi_echo` consumes `desc-MEICA_XCPD32P_denoised` MNI volume and
  91k dtseries files.

Like NoXCPD, 0P, 22P, and 32P use independent HippUnfold subject trees under their
selected `hipp_outputs/subjects/sub-<ID>/hippunfold/`. Use an isolated BIDS
staging root for anatomical inputs before running HippUnfold so SnakeBIDS does
not scan existing `hipp_outputs` as inputs.

Example 06122026 downstream, overview, and FC heatmap reruns:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/iowa7t/run_branch_downstream.py --variant 0p --job 06122026:multiband --job 06122026:multi_echo --stage all --run-metadata-template '/absolute/path/to/run_metadata/{sub_subject}.json' --overwrite
/home/yojiao/.conda/envs/py314/bin/python scripts/iowa7t/run_branch_downstream.py --variant 32p --job 06122026:multiband --job 06122026:multi_echo --stage all --run-metadata-template '/absolute/path/to/run_metadata/{sub_subject}.json' --overwrite

/home/yojiao/.conda/envs/py314/bin/python scripts/iowa7t/run_branch_overviews.py --variant 0p --job 06122026:multiband --job 06122026:multi_echo --jobs 4 --overwrite
/home/yojiao/.conda/envs/py314/bin/python scripts/iowa7t/run_branch_overviews.py --variant 32p --job 06122026:multiband --job 06122026:multi_echo --jobs 4 --overwrite

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

NoXCPD is complete only when each corresponding branch has the same relative
overview PNG inventory as 36P under `figures/gradient/` and `figures/wta/`.
The images are expected to differ in content; completion means the same named
products exist in the NoXCPD tree.

Audit the completed overview inventory with:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/iowa7t/audit_branch_completion.py   --variant noxcpd   --compare-variant 36p   --job 06112026:multiband   --job 06112026:multi_echo   --job 06122026:multiband   --job 06122026:multi_echo
```

Regenerate overview figures with the same wrapper used for 36P, adding
`--variant noxcpd`. Run the default canonical7 and family17 when the 36P branch has
both figure families:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/iowa7t/run_branch_overviews.py   --variant noxcpd   --job 06112026:multiband   --job 06112026:multi_echo   --job 06122026:multiband   --job 06122026:multi_echo   --jobs 4   --overwrite

/home/yojiao/.conda/envs/py314/bin/python scripts/iowa7t/run_branch_overviews.py   --variant noxcpd   --network-column family17   --job 06112026:multiband   --job 06112026:multi_echo   --job 06122026:multiband   --job 06122026:multi_echo   --jobs 4   --overwrite
```

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

For `sub-06112026` on Argon, the default 22P root is:

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

Each `run-XX/` directory directly contains the two label text files and its
provenance record. The root deliberately has no `subjects/` or
`sub-06112026/` wrapper and is separate from
`DataOrg_PFM_22P/hipp_outputs`. The analogous 36P rerun uses the same layout
under `DataOrg_PFM_36P/kong2022_runwise_06112026_20260904/`.

Use the dedicated launcher and inspect its printed qsub command before any
submission:

```bash
SYSTEM_NAME=argon IOWA7T_VARIANT=22p SUBJECT_ID=06112026 \
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
- `run-XX/surface/`: `raw/`, `tsnr_gate/`, `2mm/` and `4mm/` smoothed hippocampal timeseries (`sub-<subject>_hemi-{L,R}_timeseries.npy`)
- `run-XX/fc/`: `2mm/` and `4mm/` FC matrices (`hipp_vertex_to_network_fc.npy`, `hipp_vertex_to_network_fc_active.npy`, and `hippo512_to_kong400_corr.npy`)


## Branch Overview Figures

Use the predefined overview wrapper for Iowa7T branch-level overview
regeneration:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/iowa7t/run_branch_overviews.py --job 04172026:multiband --overwrite
```

The wrapper adapts branch-scoped individual products to
`scripts/individual_workflow/02_build_individual_overviews.py` and the shared
core overview renderer. It renders the maintained four method panels:

- `gradient_spectral_k3`
- `gradient_count`
- `gradient_weighted_fisherz`
- `wta_normalized_top20_count`

Final subject-facing PNGs are copied back to the existing branch figure roots. The wrapper preserves copy errors instead of falling back to content-only copying, and it skips notebook checkpoint artifacts because they are transient, non-output files:

- `subjects/sub-<ID>/<branch>/figures/gradient/`
- `subjects/sub-<ID>/<branch>/figures/wta/`

The weighted Fisher-z gradient panel reuses existing weighted gradient atlas
products from `subjects/sub-<ID>/<branch>/atlas/gradient/` and synchronizes them
into the existing overview compatibility cache under `runs/cache/`; no weighted
WTA output is produced. Kong2022 weighted
profiles, status JSON, label maps, and rendered heatmap columns must use the
canonical7 network order `Default`, `Visual`, `Somatomotor`,
`DorsalAttention`, `VentralAttention`, `Control`, `Language`. Run
the audit before trusting or publishing regenerated weighted panels:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/iowa7t/audit_repair_weighted_network_order.py --output-root /home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P/hipp_outputs
```

Use `--repair` only when the audit reports mismatched derived products. The
repair rewrites weighted Fisher-z profiles/status/label products in canonical
network order and does not delete files. The overview adapter stores
method-isolated renderer inputs under `overview_core_adapter/<method>/` so
parallel rendering cannot mix `fisherz`, `normalized_top20_count`, and
`weighted_fisherz` profiles. Stale or missing manifest paths fail instead of falling back to
same-run-dir variant files. For `gradient_fisherz`, the rendered heatmap is the same
Kong2022 ROI-mean Fisher-z network profile used by confidence/status, not a
separate direct cluster-to-network timeseries profile. For `gradient_weighted_fisherz`,
the adapter recomputes network time series from ROI time series using
`n_grayordinates_used` weights, then correlates cluster time series with those
weighted network time series; it is not copied from `gradient_fisherz`. The
normalized WTA panel is the canonical8 top-20 positive ROI-count assignment,
normalized by the finite available ROI count in each network, not Fisher-z
averaging. It is written as
`wta_top_labels_normalized_top20_count.npy`.

For Kong2022 17-network gradient overviews, run the same wrapper with
`--network-column family17`. This uses an isolated cache named
`runs/cache/iowa7t_branch_individual_overviews_family17/` and publishes only
the three gradient PNGs with explicit family17 filenames, leaving the default
default canonical7 atlas and overview outputs untouched:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/iowa7t/run_branch_overviews.py \
  --network-column family17 \
  --job 04172026:multiband \
  --job 06112026:multiband \
  --job 06112026:multi_echo \
  --job 06122026:multiband \
  --job 06122026:multi_echo \
  --output-root /home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P/hipp_outputs \
  --jobs 4 \
  --overwrite
```

The family17 gradient files are written under
`subjects/sub-<ID>/<branch>/figures/gradient/` as
`sub-<ID>_gradient_family17_fisherz_overview.png`,
`sub-<ID>_gradient_family17_count_overview.png`, and
`sub-<ID>_gradient_family17_weighted_fisherz_overview.png`. For family17, the
weighted Fisher-z overview is recomputed from ROI time series and ROI
grayordinate weights for the 17-network taxonomy; it is not copied from
`gradient_fisherz` or from canonical8 weighted products. Identical surface
colors can still occur when the weighted and unweighted profiles choose the
same top-1 network, but the audit checks the underlying Fisher-z profile and
status values. The legacy `gradient_count` overview alias uses the normalized
Top-20 ROI profile and `normalized_top20_count` status rows. Audit all five
branch outputs before using them:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/iowa7t/audit_family17_gradient_overviews.py \
  --output-root /home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P/hipp_outputs
```

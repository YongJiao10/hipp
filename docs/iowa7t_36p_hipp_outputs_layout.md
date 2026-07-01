# Iowa7T 36P Hippocampal Output Layout

Canonical production roots:

- 36P local: `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P/hipp_outputs`
- 36P Argon: `/Shared/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P/hipp_outputs`
- NoXCPD local: `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_NoXCPD/hipp_outputs`
- NoXCPD Argon: `/Shared/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_NoXCPD/hipp_outputs`

Rules:

- Iowa7T means the Jianglab fMRIPrep Iowa7T trees, not HCPpipeline500.
- `36p` remains the default workflow variant for backward compatibility.
- `noxcpd` uses the parallel `DataOrg_PFM_NoXCPD` tree and must not write into the 36P output root.
- `outputs_iowa/` in the repository is not a production output root.
- Subject structural HippUnfold output is shared at `subjects/sub-<ID>/hippunfold/`.
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

Migration helper:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/iowa7t/migrate_hipp_outputs_layout.py --subjects 04172026 --mode dry-run
/home/yojiao/.conda/envs/py314/bin/python scripts/iowa7t/migrate_hipp_outputs_layout.py --subjects 04172026 --mode copy
```

`dry-run` writes only a concrete migration manifest under `runs/manifests/`.
`copy` copies or recreates symlinks into canonical locations and leaves legacy
paths in place. Deletion is a separate user-approved operation.

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

Run NoXCPD HippUnfold before downstream products. Use an isolated BIDS staging
root for the anatomical inputs so SnakeBIDS does not scan
`hipp_outputs/subjects/sub-*/hippunfold/` as additional T1w inputs during reruns:

```bash
mkdir -p /tmp/noxcpd_hippunfold_bids_staging/sub-06112026 /tmp/noxcpd_hippunfold_bids_staging/sub-06122026
ln -sfn /home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_NoXCPD/sub-06112026/anat /tmp/noxcpd_hippunfold_bids_staging/sub-06112026/anat
ln -sfn /home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_NoXCPD/sub-06122026/anat /tmp/noxcpd_hippunfold_bids_staging/sub-06122026/anat

HIPPUNFOLD_CACHE_ROOT=/home/yojiao/hpchome/hipp/runtime/hippunfold_workdirs/noxcpd_sub-06112026 \
  /home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/01_01_run_hippunfold_t1w_local.py \
  06112026 \
  /tmp/noxcpd_hippunfold_bids_staging \
  /home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_NoXCPD/hipp_outputs/subjects/sub-06112026/hippunfold

HIPPUNFOLD_CACHE_ROOT=/home/yojiao/hpchome/hipp/runtime/hippunfold_workdirs/noxcpd_sub-06122026 \
  /home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/01_01_run_hippunfold_t1w_local.py \
  06122026 \
  /tmp/noxcpd_hippunfold_bids_staging \
  /home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_NoXCPD/hipp_outputs/subjects/sub-06122026/hippunfold
```

Example downstream rerun:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/iowa7t/run_branch_downstream.py   --variant noxcpd   --job 06112026:multiband   --job 06112026:multi_echo   --job 06122026:multiband   --job 06122026:multi_echo
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
`--variant noxcpd`. Run both canonical8 and family17 when the 36P branch has
both figure families:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/iowa7t/run_branch_overviews.py   --variant noxcpd   --job 06112026:multiband   --job 06112026:multi_echo   --job 06122026:multiband   --job 06122026:multi_echo   --jobs 4   --overwrite

/home/yojiao/.conda/envs/py314/bin/python scripts/iowa7t/run_branch_overviews.py   --variant noxcpd   --network-column family17   --job 06112026:multiband   --job 06112026:multi_echo   --job 06122026:multiband   --job 06122026:multi_echo   --jobs 4   --overwrite
```

## Cortex ArealMSHBM

Iowa7T 36P cortex individualized labels use the Argon
`scripts/ArealMSHBM_Kong2022_Iowa_Stanford` launcher with
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

## Branch Overview Figures

Use the predefined overview wrapper for Iowa7T branch-level overview
regeneration:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/iowa7t/run_branch_overviews.py --job 04172026:multiband --overwrite
```

The wrapper adapts branch-scoped individual products to
`scripts/individual_workflow/02_build_individual_overviews.py` and the shared
core overview renderer. It renders the maintained seven method panels:

- `gradient_spectral_k3`
- `gradient_count`
- `gradient_weighted_fisherz`
- `wta_top`
- `wta_weighted_fisherz`
- `wta_normalized_top20_count`
- `wta_family17_normalized_top20_count`

Final subject-facing PNGs are copied back to the existing branch figure roots. The wrapper preserves copy errors instead of falling back to content-only copying, and it skips notebook checkpoint artifacts because they are transient, non-output files:

- `subjects/sub-<ID>/<branch>/figures/gradient/`
- `subjects/sub-<ID>/<branch>/figures/wta/`

The weighted Fisher-z panels reuse existing weighted atlas products from
`subjects/sub-<ID>/<branch>/atlas/{gradient,wta}/` and synchronize them into the
existing overview compatibility cache under `runs/cache/`; do not create
separate weighted output folders for these overviews. Kong2022 weighted
profiles, status JSON, label maps, and rendered heatmap columns must use the
canonical8 network order `Default`, `Visual`, `Somatomotor`,
`DorsalAttention`, `VentralAttention`, `Control`, `Auditory`, `Language`. Run
the audit before trusting or publishing regenerated weighted panels:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/iowa7t/audit_repair_weighted_network_order.py --output-root /home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P/hipp_outputs
```

Use `--repair` only when the audit reports mismatched derived products. The
repair rewrites weighted Fisher-z profiles/status/label products in canonical
network order and does not delete files. The overview adapter stores
method-isolated renderer inputs under `overview_core_adapter/<method>/` so
parallel rendering cannot mix `fisherz`, `count`, and `weighted_fisherz`
profiles. Stale or missing manifest paths fail instead of falling back to
same-run-dir variant files. For `gradient_fisherz`, the rendered heatmap is the same
Kong2022 ROI-mean Fisher-z network profile used by confidence/status, not a
separate direct cluster-to-network timeseries profile. For `gradient_weighted_fisherz`,
the adapter recomputes network time series from ROI time series using
`n_grayordinates_used` weights, then correlates cluster time series with those
weighted network time series; it is not copied from `gradient_fisherz`. The two normalized WTA panels are top-20 positive ROI count
assignments normalized by the finite available ROI count in each network, not
Fisher-z averaging. They are written as
`wta_top_labels_normalized_top20_count.npy` for canonical8 and
`wta_top_labels_family17_normalized_top20_count.npy` for family17.

For Kong2022 17-network gradient overviews, run the same wrapper with
`--network-column family17`. This uses an isolated cache named
`runs/cache/iowa7t_branch_individual_overviews_family17/` and publishes only
the three gradient PNGs with explicit family17 filenames, leaving the default
canonical8 atlas and overview outputs untouched:

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
status values. The count overview heatmap uses the same top-N
ROI count profile family as the count confidence panel. Audit all five branch
outputs before using them:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/iowa7t/audit_family17_gradient_overviews.py \
  --output-root /home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P/hipp_outputs
```


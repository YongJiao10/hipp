# Iowa7T 36P Hippocampal Output Layout

Canonical production root:

- Local: `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P/hipp_outputs`
- Argon: `/Shared/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P/hipp_outputs`

Rules:

- Iowa7T means the Jianglab 36P fMRIPrep tree, not HCPpipeline500.
- `outputs_iowa/` in the repository is not a production output root.
- Subject structural HippUnfold output is shared at `subjects/sub-<ID>/hippunfold/`.
- Functional branches are direct siblings: `single_echo/` and `multi_echo/`.
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
        ├── single_echo/
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
single = iowa7t_branch_layout("06112026", "single_echo")
multi = iowa7t_branch_layout("06112026", "multi_echo")
```

Migration helper:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/iowa7t/migrate_hipp_outputs_layout.py --subjects 04172026 --mode dry-run
/home/yojiao/.conda/envs/py314/bin/python scripts/iowa7t/migrate_hipp_outputs_layout.py --subjects 04172026 --mode copy
```

`dry-run` writes only a concrete migration manifest under `runs/manifests/`.
`copy` copies or recreates symlinks into canonical locations and leaves legacy
paths in place. Deletion is a separate user-approved operation.

## Cortex ArealMSHBM

Iowa7T 36P cortex individualized labels use the Argon
`scripts/ArealMSHBM_Kong2022_Iowa_Stanford` launcher with
`DATASET=iowa7t` and an explicit `IOWA7T_BRANCH`.

Branch inputs are fixed:

- `single_echo` consumes
  `sub-<ID>/func/sub-<ID>_task-rest_dir-concat_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii`
- `multi_echo` consumes
  source CIFTI
  `sub-<ID>/func/sub-<ID>_task-rest_run-concat_space-fsLR_den-91k_desc-MEICA_XCPD36P_denoised_bold.dtseries.nii`;
  the Argon launcher derives a persistent cortex-only fsLR32k CIFTI under
  `hipp_outputs/runs/derived_inputs/<run_id>/subjects/sub-<ID>/multi_echo/cortex_only/`
  and passes that derived file to MATLAB

Branch outputs are fixed:

- `subjects/sub-<ID>/single_echo/cortex/areal_mshbm/`
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

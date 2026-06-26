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
        │   ├── surface/
        │   ├── fc/
        │   ├── atlas/
        │   ├── qc/
        │   ├── figures/
        │   └── provenance.json
        └── multi_echo/
            ├── cortex/
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

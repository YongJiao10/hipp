# Iowa7T Branch Downstream Adapter Design

## Goal

Run post-ArealMSHBM downstream hippocampal/cortex comparison products for Iowa7T 36P subjects without creating legacy extra nesting such as `single_echo/sub-06112026/...`.

## Fixed inputs

- Local Jianglab root: `/home/yojiao/LSS/jianglab`
- Iowa7T 36P root: `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P`
- Output root: `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P/hipp_outputs`
- HippUnfold root per subject: `hipp_outputs/subjects/sub-<ID>/hippunfold/`
- Branch root per echo branch: `hipp_outputs/subjects/sub-<ID>/<single_echo|multi_echo>/`
- ArealMSHBM labels per branch: `<branch_root>/cortex/areal_mshbm/lh_labels.txt` and `rh_labels.txt`

## Fixed outputs

The adapter writes directly under each branch root:

- `<branch_root>/surface/raw/`
- `<branch_root>/surface/tsnr_gate/`
- `<branch_root>/surface/2mm/`
- `<branch_root>/surface/4mm/`
- `<branch_root>/cortex/common/`
- `<branch_root>/cortex/kong2022/`
- `<branch_root>/fc/kong2022/<2mm|4mm>/hemi_<L|R>/`
- `<branch_root>/atlas/gradient/<2mm|4mm>/<L|R>/`
- `<branch_root>/atlas/wta/<2mm|4mm>/<L|R>/`
- `<branch_root>/figures/gradient/`
- `<branch_root>/figures/wta/`

The overview wrapper may use non-empty compatibility caches under `hipp_outputs/runs/cache/iowa7t_branch_individual_overviews/sub-<ID>_<branch>/`, but final subject-facing outputs must be direct branch-scoped paths.
It must not create `variants/`, empty `group/`, or a nested `<branch_root>/sub-<ID>/` compatibility tree.
`cortex/common/` is allowed because it contains concrete shared cortex tSNR masks and run-boundary summaries consumed by FC.

## Branch input templates

For `single_echo`:

- BOLD volume: `sub-<ID>/func/sub-<ID>_task-rest_dir-concat_space-MNI152NLin6Asym_res-2_desc-denoised_bold.nii.gz`
- CIFTI dtseries: `sub-<ID>/func/sub-<ID>_task-rest_dir-concat_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii`

For `multi_echo`:

- BOLD volume: `sub-<ID>/func/sub-<ID>_task-rest_run-concat_space-MNI152NLin6Asym_res-2_desc-MEICA_XCPD36P_denoised_bold.nii.gz`
- CIFTI dtseries: `sub-<ID>/func/sub-<ID>_task-rest_run-concat_space-fsLR_den-32k_desc-MEICA_XCPD36P_denoised_bold.dtseries.nii`

## Implementation approach

Create a thin Iowa7T-specific adapter script, `scripts/iowa7t/run_branch_downstream.py`, that:

1. resolves subject/branch paths using the fixed layout above;
2. validates existing HippUnfold outputs, BOLD volume, dtseries, and ArealMSHBM label files before expensive work;
3. materializes surface sampling, hippocampal tSNR gate, 2 mm and 4 mm smoothed hippocampal timeseries directly under the branch root;
4. builds Kong2022 cortex reference products directly under `<branch_root>/cortex/kong2022/`;
5. materializes vertex-to-Kong2022 ROI FC directly under `<branch_root>/fc/kong2022/`;
6. preserves the 400-row Kong2022 ROI dimension for branch comparison; ROI labels with no vertices are retained as NaN rows/FC columns and flagged in TSV/summary metadata;
7. writes stage manifests only where a real stage has produced files.

The adapter reuses core workflow helper functions and math where possible, but it does not call wrappers whose public interface always appends `sub-<ID>` below their roots.

## Validation

Tests must assert the path contract before implementation:

- branch output roots are direct branch-scoped paths;
- input templates differ correctly between `single_echo` and `multi_echo`;
- `cortex/areal_mshbm` is treated as flat branch-local label input.

Runtime validation must verify the expected output files for all requested subject/branch jobs after the run.

# Bundle Agent Notes

## Goal

Run the 166-subject network-first pipeline on HPC for:

- `network-gradient`
- `network-prob-cluster-nonneg`
- `lynch2024`
- `kong2019`

## Recommended Entry Point

Use:

`scripts/run_network_first_166_bundle.py`

Do not manually improvise stage order unless debugging.

## Required Environment Adjustments

- Activate the intended Python environment before running bundle scripts.
- Ensure `hippunfold` is available on `PATH`.
- Ensure Workbench `wb_command` is available either at the default mac path or via `WB_COMMAND_BIN`.
- Ensure FASTANS code/resources are available under `external/FASTANS` or override with `--fastans-root`.
- Prefer bootstrapping with `setup_hpc_env.sh` before manual debugging.

## Authoritative Subject List

Use:

`manifests/hcp_7t_hippocampus_struct_complete_166.txt`

## Expected Outputs

Per `subject x atlas x branch`, expect:

- `final_selection_summary.json`
- `final_selection_core.json`
- `summary_manifest.json`
- `k_selection_curves.png`
- `hipp_functional_parcellation_network_overview.png`
- full `features/` and `clustering/` directories

## Debug Order

If something fails, check in this order:

1. input staging under `data/hippunfold_input`
2. `outputs/dense_corobl_batch`
3. `outputs/cortex_pfm`
4. `outputs/hipp_functional_parcellation_network`

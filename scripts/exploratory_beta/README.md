# Exploratory Native-Space Beta Analysis

Stanford and JJ_TMS native-space hippocampal beta analyses live here. Each
command is exploratory, is not a formal-workflow step, and its outputs are
not automatically suitable for reporting. Keep these utilities outside the
numbered individual workflow.

## Inputs and methods

Sample native con volumes on each subject's `space-T1w_den-512` geometry and
use native vertex areas. Functional labels come from registered-BOLD processing.
The selected mapping manifest must declare the same dataset and
`native_surface_space=T1w`; rebuild incompatible corobl mappings explicitly.

Use the dataset method guides for cohorts, models, inputs and commands:

- [Stanford native beta](../../docs/hipp_parcellation_network/stanford_tms_surface_beta.md)
- [JJ_TMS native beta](../../docs/hipp_parcellation_network/jj_tms_sptms_cluster_beta.md)

## Entrypoints

| Purpose | Scripts |
|---|---|
| Input staging and coverage comparison | `copy_native_con_inputs.py`, `compare_native_standard_surface_bad_rates.py` |
| Native L/R beta tables | `run_native_surface_beta.py` |
| Statistical models | `stanford_tms_surface_model.R`, `jj_tms_sptms_native_surface_model.R` |
| Diagnostic figures | `render_native_surface_beta_figures.py` |
| Dataset figure sets | `stanford_tms_five_figures.py`, `render_jj_tms_five_figures.py` |
| Selected 16-person HCP raw-beta figure | `render_jj_tms_beta_surface_bar.py` |

Run from the repository root with the explicit interpreter:

```bash
/home/yojiao/.conda/envs/py314/bin/python   scripts/exploratory_beta/run_native_surface_beta.py   --dataset jj_tms --run-id <run_id> --roi-policy nonzero
```

Figure scripts write PNG by default; request `--pdf` explicitly for PDF output.
The healthy-profile VAN/DMN/CON Welch comparisons are exploratory and
uncorrected; their raw results are exported with the figures.

## Selected JJ_TMS raw-beta figure

`render_jj_tms_beta_surface_bar.py` uses the selected run's existing
`five_figures/` outputs and original subject means. It requires 96 unique finite
HCP beta cells for 16 subjects, keeping L/R separate. Missing metrics are
recovered serially on native T1w den-512 ribbons and checked against original
area-weighted means; missing masks or mismatches stop rendering.

The figure uses corresponding den-512 vertices for Workbench display, no added
smoothing, a shared raw-beta scale, and original subject means with 95% Student
t intervals. It is descriptive and does not perform an HC-versus-Pure-MDD
inferential test. Use the script's `--help` for run/output selection.

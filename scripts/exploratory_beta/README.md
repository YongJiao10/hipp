# Exploratory Native-Space Beta Analysis

This directory contains the current Stanford and JJ_TMS Native-space
hippocampal surface-beta exploration. It is deliberately outside
`scripts/individual_workflow/`. Each command is exploratory and is not a
numbered or frozen formal-workflow step, and its outputs are not automatically
suitable for reporting.

The command interface, model formulas, input tables, and output layouts remain
unchanged from the local migration snapshot. Only the source location and
analysis-status documentation changed.

## Entry points

- `copy_native_con_inputs.py`: stage the Native con inputs for the selected
  exploratory cohort.
- `compare_native_standard_surface_bad_rates.py`: create Native surface metrics
  and compare bad-vertex coverage with the immutable historical reference.
- `run_native_surface_beta.py`: extract Stanford or JJ_TMS Native L/R beta
  tables.
- `stanford_tms_surface_model.R`: fit the Stanford exploratory models.
- `jj_tms_sptms_native_surface_model.R`: fit the JJ_TMS exploratory models.
- `render_native_surface_beta_figures.py`: render the compact Native L/R
  diagnostic figures.
- `stanford_tms_five_figures.py`: render the Stanford five-figure set.
- `render_jj_tms_five_figures.py`: adapt the Stanford renderer for JJ_TMS.

Run Python commands with the explicit repository interpreter. For example:

```bash
/opt/miniconda3/envs/py314/bin/python \
  scripts/exploratory_beta/run_native_surface_beta.py \
  --dataset jj_tms \
  --run-id <run_id> \
  --roi-policy nonzero
```

The current method notes and concrete commands are in:

- `docs/hipp_parcellation_network/stanford_tms_surface_beta.md`
- `docs/hipp_parcellation_network/jj_tms_sptms_cluster_beta.md`

Historical Standard/MNI, bilateral-volume, whole-hippocampus, laterality, and
other retired beta branches are not part of this active exploratory source
surface.

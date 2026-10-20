# Exploratory Native-Space Beta Analysis

This directory contains the current Stanford and JJ_TMS Native-space
hippocampal surface-beta exploration. It is deliberately outside
`scripts/individual_workflow/`. Each command is exploratory and is not a
numbered or frozen formal-workflow step, and its outputs are not automatically
suitable for reporting.

Both Stanford and JJ_TMS Native con volumes use the individual's
`space-T1w_den-512` surfaces for volume sampling and physical vertex areas.
Functional parcellation labels must come from the registered-BOLD workflow.
The selected mapping's existing `run_manifest.json` must declare the same
dataset and `native_surface_space=T1w`; old corobl mappings must be rebuilt
before beta extraction.

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
- `render_jj_tms_beta_surface_bar.py`: 16-person HCP151 raw-beta surface/bar
  figure; PNG by default, PDF only with explicit `--pdf`.

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

The healthy profile figure in the five-figure renderer includes an exploratory,
uncorrected Welch comparison of VAN against DMN and FPCN at each
parcellation × hippocampal hemisphere × cluster. A significant VAN-higher
comparison is drawn with a red bracket and `*`/`**`/`***`; a significant
VAN-lower comparison is blue. Non-significant comparisons have no bracket or
label. The raw results are written to
`healthy_van_vs_fpcn_dmn_ttests.csv`. The five renderers write PNG by default;
pass `--pdf` when PDF files are explicitly needed.

JJ_TMS five-figure titles use `Iowa 3T` as the visible dataset label; the
machine-readable dataset identifier remains `jj_tms`.

## Selected JJ_TMS raw-beta figure

```bash
/home/yojiao/.conda/envs/py314/bin/python \
  scripts/exploratory_beta/render_jj_tms_beta_surface_bar.py
```

Default run: `jj_tms_native_surface_beta_20260820_78sub`. All outputs go to its
existing `five_figures/`, with stem
`jj_tms_hcp151_fixed_2mm_36p_beta_surface_bar_16sub` (PNG, selected beta CSV,
group summary CSV, figure manifest; PDF only with `--pdf`).

Exactly 96 unique finite HCP151 `beta_mean` cells are required, keeping L/R
separate. Metric provenance comes from `site_level_beta.csv`. Missing selected
hemisphere metrics are recovered serially under `five_figures/raw_metrics/`
with native T1w den-512 ribbons, nonzero ROI and voxel subdivision 3. Complete
recovered pairs may be reused after interruption; all metrics are checked
against original area-weighted means (rtol 1e-5, atol 1e-6). Missing bad masks
or mean mismatches stop rendering. No archive access or model fitting occurs.

The existing locked Workbench scene and HCP 2mm_36p display geometry are reused
through corresponding den-512 vertices. The composite surface cards are laid
out in three rows: HC 5 + 5, then Pure MDD 6. Bilateral panels are joined into
each card before composition so the hippocampi use the available card width.
The figure canvas and labels use deliberately readable sizes, and the colorbar
stays compact below the cards. Each vertex receives its own raw-beta color on a
shared symmetric RdBu_r scale; display labels encode vertices, not subregions.
Invalid vertices are gray. No extra smoothing or 2 mm resampling is applied.
Persistent scenes and images remain in `surface_cards/` for review.
Bottom panels use original subject means, large points, group means and
two-sided 95% Student t intervals (df=n-1). Their title, axis labels, tick
labels and legend use a deliberately oversized 1.5× text scale for review
readability. This figure is descriptive only and does not perform or annotate
an HC-versus-Pure-MDD inferential comparison. The two MDD VAN
subjects (sub-3017 and sub-3020) share one large diamond marker; the other four
MDD subjects share large circular markers. The enlarged legend is placed on its
own row below the x-axis tick labels, with the explanatory footer below the
legend. The right-panel y-axis title is placed on the outside right
edge so it cannot intrude into the left panel.
Visible labels intentionally omit mask-state and display-preset details; those
technical provenance fields remain in the figure manifest.

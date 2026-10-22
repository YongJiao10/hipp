# Formal Group-Atlas Layout

The active layout is defined by `scripts/core_workflow/lib/output_layout.py`
and `route_tags.py`. A listed path describes the formal destination; its name
alone does not establish that a scientific product is present or valid.

## HCP

Root: `outputs_hcp/work/group_atlas/`.

- Kong2022 canonical7: `pfm/`, `fixed/`, `pfm_ctxreg/`, `pfm_ctxdrop/`.
- Kong2022 family17: `pfm_17net/`, `fixed_17net/`.
- Post-hoc weighted assignment comparison: `pfm_weighted/`.
- Lynch2024 comparison: `lynch2024_20net/`, `lynch2024_7net/`.

HCP has no native 22P branch. Family17 products reuse the corresponding PFM or
Fixed input and gradient banks. Assignment variants such as `fisherz` and
`normalized_top20_count` remain explicit files and manifest entries inside
these roots.

## Iowa JJ_TMS and Stanford

Roots: `outputs_iowa_jj_tms/work/group_atlas/` and
`outputs_stanford/work/group_atlas/`.

Canonical7 cohort products use `pfm/<cohort>/` for 36P and
`pfm_22p/<cohort>/` for 22P. Iowa cohorts are `hc`, `patient`, `pure_mdd`, and
`comorbid_mdd`. Stanford cohorts are `nthc`, `tehc`, `nts`, `nts_any_mdd`,
`nts_pure_mdd`, `tis`, and `tis_ptsd`. A signal control retains its explicit
route identity, such as `pfm_ctxreg/<cohort>/`. Scientific variants and cohorts
must remain separate.

## Retained validation and cortex-PFM workspaces

Stanford's saved canonical8 validation workspaces are
`outputs_stanford/work/validation_gradient_cluster_stats/` and
`outputs_stanford/work/validation_gradient_cluster_stats_weighted/`. They retain
the Auditory-versus-VentralAttention analysis contract and its separate weighted
comparison. Their group-input definitions remain distinct from the current
canonical7 group-atlas roots; relocating the workspaces does not retarget or
rerun that analysis.

Iowa JJ_TMS's separate 70-subject single-run PFM package is retained under
`outputs_iowa_jj_tms/work/cortex_pfm/pfm_nsi_v0p1p0_sparse0p25_single_run01/`.
Its subjects, model settings, logs, and acquisition records remain together.
It is a distinct scientific run and is not merged into the main PFM package.

## Inputs and readiness

Kong reference inputs use `inputs/kong2022/shared_inputs/pfm/` or `fixed/`;
the supported 22P input branch uses `inputs/kong2022_22p/shared_inputs/pfm/`.
Lynch references use `inputs/lynch2024/shared_inputs/pfm/` for the HCP base
cohort, with explicit `pfm_nsi043/` roots for retained NSI-filtered comparisons.
Iowa 22P Lynch references use `inputs/lynch2024_22p/shared_inputs/pfm_nsi043/`.

Reference path cleanup preserves the original cortex reference files and QC
provenance. Hippocampal coordinate corrections do not require cortex reference
regeneration: the references use cortical CIFTI time series and cortical labels.
Existing consumers validate the declared subject roster, taxonomy, and artifact
lineage under their documented analysis contract.
Archived FC, gradients, group atlases, and presentation products remain
archived until an exact restoration or rebuild scope is approved. Missing
formal products must fail closed rather than select a historical tree.

The numbered overview and matrix steps and multi-dataset visualizers consume
these formal roots. Use their existing readiness checks to assess actual
products. Archive mappings are declared explicitly in
`config/legacy_wrong_path_mapping.json`; archive inspection and filesystem
changes remain subject to the repository's explicit approval rules.

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

Family17 products reuse the corresponding PFM or
Fixed input and gradient banks. Assignment variants such as `fisherz` and
`normalized_top20_count` remain explicit files and manifest entries inside
these roots.

## Iowa JJ_TMS and Stanford

Roots: `outputs_iowa_jj_tms/work/group_atlas/` and
`outputs_stanford/work/group_atlas/`.

Canonical7 cohort products use `pfm/<cohort>/` for 36P. Iowa cohorts are `hc`, `patient`, `pure_mdd`, and
`comorbid_mdd`. Stanford cohorts are `nthc`, `tehc`, `nts`, `nts_any_mdd`,
`nts_pure_mdd`, `tis`, and `tis_ptsd`. A signal control retains its explicit
route identity, such as `pfm_ctxreg/<cohort>/`. Scientific variants and cohorts
must remain separate.

## Retained validation and NSI workspaces

Stanford's saved canonical8 validation workspaces are
`outputs_stanford/work/validation_gradient_cluster_stats/` and
`outputs_stanford/work/validation_gradient_cluster_stats_weighted/`. They retain
the Auditory-versus-VentralAttention analysis contract and its separate weighted
comparison. Their group-input definitions remain distinct from the current
canonical7 group-atlas roots; relocating the workspaces does not retarget or
rerun that analysis.

NSI quality-analysis packages belong under
`outputs_<dataset>/work/pfm_nsi/<original_package_name>/`. Their subject-level
scores, group summaries, figures, and original execution records remain together.
Iowa JJ_TMS's separate 70-subject single-run NSI package is retained under
`outputs_iowa_jj_tms/work/pfm_nsi/pfm_nsi_v0p1p0_sparse0p25_single_run01/`,
separate from the main NSI package. Parameter and acquisition variants remain
distinct. See [Output Layout](output_layout.md) for the full directory contract.

## Inputs and readiness

Kong reference inputs use `inputs/kong2022/shared_inputs/pfm/` or `fixed/`.
Lynch references use `inputs/lynch2024/shared_inputs/pfm/` for the HCP base
cohort, with explicit `pfm_nsi043/` roots for retained NSI-filtered comparisons.

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

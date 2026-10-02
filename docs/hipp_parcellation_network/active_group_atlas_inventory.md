# Active Group-Atlas Inventory

Read-only inventory baseline captured on 2026-09-12 on IDAS. This note covers
active output roots only; no archive content was opened. The HCP control roots
were subsequently rebuilt under their approved formal names, and the six
approved dated/recovered migration roots were removed after validation.

## HCP

Root: `outputs_hcp/work/group_atlas/`.

- Retain as active science branches: `kong2022_7net_full_k3_fisherz`,
  `routea_17net_full_k3_fisherz`,
  `routea_7net_weighted_fisherz_full_k3`,
  `routeb_full_k3_fisherz` (when present), and the maintained
  `lynch2024_20net_hcp151_full_k3_fisherz` /
  `lynch2024_7net_hcp151_full_k3_fisherz` roots. Their manifests identify
  the expected Kong canonical7, family17, weighted, or Lynch taxonomy and
  their consumers are the numbered overview/matrix steps plus the
  multidataset visualizers.
- Formal HCP control roots `outputs_hcp/work/group_atlas/routea_ctxreg/` and
  `outputs_hcp/work/group_atlas/routea_ctxdrop/` are the only active control
  entrances. Each has four gradient combinations and four rebuilt WTA
  combinations (2 mm/4 mm × L/R). The WTA manifests use canonical7, have
  assigned support on every combination, and feed the three validated
  overview variants under the matching formal present root.
- The active consumers are `06_01_build_group_overview.py`,
  `06_03_build_cluster_cortex_panels.py`, `06_06_render_confidence_maps.py`,
  `07_01_build_group_matrix.py`, and the multidataset cortex-gradient/WTA
  visualizers. The formal resolver now selects the direct route roots, not
  the dated source trees.

Recommendation: retain the valid science branches and the two formal control
roots. The dated/recovered control trees listed in the approved cleanup record
were removed after their valid inputs were migrated or numerically verified.
Do not infer that a control product is valid from its directory name.

## Iowa JJ_TMS

Root: `outputs_iowa_jj_tms/work/group_atlas/` (24 direct roots).

- Kong canonical7 roots:
  `kong2022_7net_{hc,patient,pure_mdd,comorbid_mdd}{,_22p}_full_k3_fisherz`.
- Lynch canonical7 WTA roots:
  `lynch2024_7net_iowa3t_{hc,patient,pure_mdd,comorbid_mdd}{,_22p}_full_k3_fisherz_wta`.
- Weighted RouteA roots:
  `routea_7net_{hc,pure_mdd,comorbid_mdd}{,_22p}_weighted_fisherz_full_k3`.
- Patient RouteA roots:
  `routea_patient{,_22p}_full_k3_fisherz`.

The read-only WTA audit found valid assigned support in the named canonical7
roots. The `_22p` roots are scientific branches and must remain separate from
the unversioned 36P roots. Consumers are the numbered overview/matrix steps,
weighted-comparison workflows, and the multidataset visualizers.

Recommendation: retain all named roots. This round proposes no Iowa rename or
cleanup; any root lacking a future manifest/consumer proof should be marked
待核实 in a separate change rather than selected by name.

## Stanford

Root: `outputs_stanford/work/group_atlas/` (29 direct roots).

- Seven maintained cohorts (`nthc`, `tehc`, `nts`, `nts_any_mdd`,
  `nts_pure_mdd`, `tis`, `tis_ptsd`) each have `routea_7net_<cohort>` full
  and `_22p` roots, plus the corresponding
  `weighted_fisherz_full_k3` roots. These are real cohort and 22P/36P
  scientific branches, not old/new aliases.
- `routea_full_k3_fisherz` is present with zero manifests and has no proven
  active consumer in this inventory; status is 待核实.

The named cohort roots contain Kong canonical7 gradient/WTA manifests and the
read-only WTA audit found assigned support. Consumers are the numbered
overview/matrix steps and weighted comparison/visualization workflows.

Recommendation: retain the named cohort and weighted roots. Do not remove or
rename the empty `routea_full_k3_fisherz` root without a separate consumer
decision and an exact approved cleanup list.

## Evidence boundary

Manifest status, taxonomy, support-domain status, and direct consumer
references were checked read-only. All sixteen HCP formal group-atlas
manifests and six control overview images were then content-checked; the
formal control WTA products are no longer blocked by the former unassigned
support-domain failure.

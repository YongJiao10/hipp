# Stanford cohort group-atlas paths

The Stanford CausalConnectome workflow uses the complete QC-passed roster for
36P: 194 subjects. The
canonical seven overview cohorts are NTHC, TEHC, NTS, NTS Any MDD, NTS Pure
MDD, TIS, and TIS PTSD. Diagnostic subsets overlap their parent cohorts.

| Cohort | Path component | 36P N |
| --- | --- | ---: |
| Non-Trauma Healthy Control | `nthc` | 51 |
| Trauma-Exposed Healthy Control | `tehc` | 40 |
| Non-Trauma Symptomatic | `nts` | 56 |
| NTS Any MDD | `nts_any_mdd` | 43 |
| NTS Pure MDD | `nts_pure_mdd` | 11 |
| Trauma-Induced Symptomatic | `tis` | 47 |
| TIS PTSD | `tis_ptsd` | 35 |

Use each cohort's explicit manifest under
`outputs_stanford/inputs/kong2022/manifests/`. Cohort selection
must use that declared roster. The NTS Pure MDD roster remains the established
11-person input; this path cleanup does not rederive diagnoses or membership.
TIS PTSD is a subset of TIS and is not an additional factorial cell.

For 36P, the formal roots relative to `outputs_stanford/` are:

| Product | Root |
| --- | --- |
| Retained cortical references | `inputs/kong2022/shared_inputs/pfm/` |
| Hippocampal surface inputs and masks | `inputs/kong2022/shared_inputs/surface/` |
| Group-average surfaces | `inputs/kong2022/templates/hipp_group_average/group_average_surfaces/` |
| Gradients | `work/hippomaps_gradients/pfm/<cohort>/` |
| Base group atlases | `work/group_atlas/pfm/<cohort>/` |
| Weighted Fisher-z group atlases | `work/group_atlas/pfm_weighted/<cohort>/` |
| Base overviews | `present/overviews/pfm/<cohort>/` |
| Weighted overviews | `present/overviews/pfm_weighted/<cohort>/` |

Method, atlas, smoothing, coverage,
gradient dimensions, and hemisphere remain below the stage roots. Fisher-z
and normalized Top-20 count assignments remain explicit filenames/metadata.
Missing formal products must be reported by their exact paths.

The maintained stages are `05_01_compute_hippomaps_gradients.py`,
`05_02_run_group_atlas.py`, `05_05_build_weighted_fisherz_branch.py`, and
`06_01_build_group_overview.py`. Pass Stanford's selected reference, surface,
template, and cohort roots explicitly. Cortical reference data and their
scientific/QC provenance are preserved during location-only cleanup.

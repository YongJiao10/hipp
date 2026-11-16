# JJ_TMS Utilities

This directory is the dataset-specific surface for JJ_TMS orchestration and
mapping. It contains the JJ_TMS HippUnfold batch runner, shared inputs and template
materializer (`materialize_jj_tms_shared_inputs.py`), target-to-Kong2022
mapping, HCP direct-template matcher, and fixed-template overview renderer.

JJ_TMS code does not belong in the protected multi-dataset numbered surfaces
`scripts/core_workflow/` or `scripts/individual_workflow/`. The reusable
HCP template builder remains numbered at
`scripts/individual_workflow/03_01_build_hcp_fixed_template.py`.

See:

- `docs/hipp_parcellation_network/jj_tms_hcp_fixed_template_matching.md`
- `docs/hipp_parcellation_network/jj_tms_target_network_mapping.md`

## Shared-input materializer

`materialize_jj_tms_shared_inputs.py` adapts the direct matcher's surface and
reference layout. The ordered run IDs and lengths come from each source
`subject_manifest.json` (`header_evidence.runs`, including matching volume and
CIFTI timepoint counts). Missing or inconsistent run declarations stop before
FC output is written. The 400-ROI registry comes from the shipped authoritative
Kong2022 atlas rather than a prior HCP group-result directory.

Pooled and per-run FC use the core `04_01` implementation and the declared time
slices. Pooled FC removes each run's mean while preserving run variances; stored
baseline-bearing series remain unchanged. Reference policy fields and both cortex
gate records are copied from the explicitly selected matcher source and checked
as `iowa_jj_tms`; missing or conflicting source policy stops materialization.
Only actual run IDs are emitted; pooled FC is never copied into run
files. Outputs include explicit 400-ROI and canonical7 FC files and their run
metadata. Empty or constant reference rows remain NaN, rather than zero
correlation. This adapter does not choose a new hippocampal QC policy.

## JJ_TMS CtxReg fixed-template matcher

`scripts/jj_tms/jj_tms_ctxreg_fixed_template_matching.py` is a separate
CtxReg adapter. It requires the exact template identity
`hcp_routea_ctxreg`, exact 2mm CtxReg FC caches under
`shared_inputs/pfm_ctxreg/`, the standard validity masks under
`shared_inputs/surface/`, and the consumed HippUnfold
`space-T1w_den-512` surfaces. It does not substitute the standard
`pfm` FC branch or select an alternate file by existence order.

For run-level volume sampling, the native T1w mesh is transformed using
`derivatives_surf/sub-<id>/anat/sub-<id>_from-MNI152NLin6Asym_to-T1w_mode-image_xfm.h5`
and ANTs point mapping. The mapped meshes are saved in each subject's
`surface/sampling_surfaces/`. Native T1w geometry defines smoothing; metrics
use `space-T1w` filenames. If `04_00` or `04_01 --ctxdrop` is used with JJ_TMS
native cortical meshes, explicitly pass `--cortex-coordinate-space T1w`.

The default roster is the concatenation of the two existing caller-selected
manifests `manifests/jj_tms_hc_subjects.txt` and
`manifests/jj_tms_patient_subjects.txt`. No redundant
union manifest is created. The adapter supports `--mode preflight|run|validate`
and caps `--jobs` at 10. A run preflights every selected subject before
creating any subject output directory. Existing subject outputs are reused only
with `--resume` and only after current template/input fingerprints, exact
paths, shapes, dtypes, labels, scores, margins, GIFTIs, and manifests validate;
incompatible or malformed state fails closed. There is no `--overwrite`.

The CtxReg template builder must be called with the explicit identity:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/individual_workflow/03_01_build_hcp_fixed_template.py \
  --mode preflight \
  --template-id hcp_routea_ctxreg \
  --group-root outputs_hcp/work/group_atlas/pfm_ctxreg \
  --gradient-root outputs_hcp/work/hippomaps_gradients/pfm_ctxreg \
  --out-root outputs_hcp/templates/hcp_routea_ctxreg
```

The builder verifies the source manifests themselves identify
`pfm_ctxreg`; changing only the output directory or template label cannot
turn standard RouteA artifacts into a CtxReg template. Rebuilding an existing
formal template or running the full-roster matcher is a separate,
explicitly reviewed product operation.

The JJ_TMS clinical/processing source is:

`/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS/subject_infomation.xlsx`

Its `Sheet1` columns `C:E` contain the background `MDD`, `PTSD`, and
`GAD` labels; column `K` contains QC status. Select `QC == "Yes"` for the
fixed imaging roster. Do not use `TMSfMRI_DataSharing.xlsx` to
replace this roster or to infer whether a patient has usable fMRI data. The
fixed Iowa3T two-group definitions and IDs are documented in the HCP
matching document linked above.


## Iowa3T diagnostic subgroup core-workflow outputs

The fixed subgroup labels are `pure MDD` (`n=7`) and `comorbid MDD` (`n=19`);
`JJ_TMS_3003` (past-only MDD/current GAD) is excluded. Their manifests are
`manifests/jj_tms_pure_mdd_subjects.txt` and
`manifests/jj_tms_comorbid_mdd_subjects.txt`.

The default unversioned subgroup branch is 36P. Its subgroup FC caches are
isolated under
`outputs_iowa_jj_tms/inputs/kong2022/shared_inputs/pfm/{pure_mdd,comorbid_mdd}/`,
with downstream roots under `outputs_iowa_jj_tms/{work,present}/` using the
same unversioned subgroup names.

## JJ_TMS canonical7 branch

The JJ_TMS canonical7 output is the default branch of the existing numbered
core workflow. The unsuffixed `kong2022`/`pfm` roots are canonical7; the
canonical7 reference, FC, gradient, and group-atlas products are computed
directly from the same raw Route-A labels/dtseries as canonical8; canonical7
must not consume canonical8 FC, gradients, clustering, or K-selection outputs.
Pass `--atlas-slug kong2022` to `03_02`, `--atlas kong2022` to `04_01`,
`--routes pfm` to `05_01`, and
`--atlases kong2022 --network-taxonomy canonical7 --route-tag pfm` to
`05_02`. The existing renderers then consume the resulting direct branch.

The fixed Iowa Patient cohort excludes sub-3003. The
active Patient branch uses `pfm/patient`; Lynch2024 7net uses the
corresponding `lynch2024_7net_iowa3t_patient*` roots. There is no parallel
27-person Patient branch.

## Signal validity

The fixed-template matcher retains finite, nonconstant hippocampal series and
finite cortical signals. It uses the declared physical-run lengths for the
cortical time series. Existing
products made under another vertex-selection policy require a rebuild.

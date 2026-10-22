# JJ_TMS Utilities

This directory is the dataset-specific surface for JJ_TMS orchestration and
mapping. It contains the JJ_TMS HippUnfold batch runner, shared inputs and template
materializer (`materialize_jj_tms_shared_inputs.py`), target-to-Kong2022
mapping, HCP151 direct-template matcher, and fixed-template overview renderer.

JJ_TMS code does not belong in the protected multi-dataset numbered surfaces
`scripts/core_workflow/` or `scripts/individual_workflow/`. The reusable
HCP151 template builder remains numbered at
`scripts/individual_workflow/03_01_build_hcp151_fixed_template.py`.

See:

- `docs/hipp_parcellation_network/jj_tms_hcp151_fixed_template_matching.md`
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
`hcp151_routea_ctxreg`, exact 2mm CtxReg FC caches under
`shared_inputs/pfm_ctxreg/`, the standard validity masks under
`shared_inputs/surface/`, and the consumed HippUnfold
`space-T1w_den-512` surfaces. It does not substitute the standard
`routea` FC branch or select an alternate file by existence order.

For run-level volume sampling, the native T1w mesh is transformed using
`derivatives_surf/sub-<id>/anat/sub-<id>_from-MNI152NLin6Asym_to-T1w_mode-image_xfm.h5`
and ANTs point mapping. The mapped meshes are saved in each subject's
`surface/sampling_surfaces/`. Native T1w geometry defines smoothing; metrics
use `space-T1w` filenames. If `04_00` or `04_01 --ctxdrop` is used with JJ_TMS
native cortical meshes, explicitly pass `--cortex-coordinate-space T1w`.

The default roster is the concatenation of the two existing caller-selected
manifests `manifests/jj_tms_hc_67_subjects.txt` and
`manifests/jj_tms_patient_26_subjects.txt` (67 + 26 = 93). No redundant
union manifest is created. The adapter supports `--mode preflight|run|validate`
and caps `--jobs` at 10. A run preflights every selected subject before
creating any subject output directory. Existing subject outputs are reused only
with `--resume` and only after current template/input fingerprints, exact
paths, shapes, dtypes, labels, scores, margins, GIFTIs, and manifests validate;
incompatible or malformed state fails closed. There is no `--overwrite`.

The CtxReg template builder must be called with the explicit identity:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/individual_workflow/03_01_build_hcp151_fixed_template.py \
  --mode preflight \
  --template-id hcp151_routea_ctxreg \
  --group-root outputs_hcp/work/group_atlas/pfm_ctxreg \
  --gradient-root outputs_hcp/work/hippomaps_gradients/pfm_ctxreg \
  --out-root outputs_hcp/templates/hcp151_routea_ctxreg
```

The builder verifies the source manifests themselves identify
`routea_ctxreg`; changing only the output directory or template label cannot
turn standard RouteA artifacts into a CtxReg template. Rebuilding an existing
formal template or running the full 93-subject matcher is a separate,
explicitly reviewed product operation.

The JJ_TMS clinical/processing source is:

`/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS/subject_infomation.xlsx`

Its `Sheet1` columns `C:E` contain the background `MDD`, `PTSD`, and
`GAD` labels; column `K` contains QC status. Select `QC == "Yes"` for the
fixed 94-subject imaging roster. Do not use `TMSfMRI_DataSharing.xlsx` to
replace this roster or to infer whether a patient has usable fMRI data. The
fixed Iowa3T two-group definitions and IDs are documented in the HCP151
matching document linked above.


## Iowa3T diagnostic subgroup core-workflow outputs

The fixed subgroup labels are `pure MDD` (`n=7`) and `comorbid MDD` (`n=19`);
`JJ_TMS_3003` (past-only MDD/current GAD) is excluded. Their manifests are
`manifests/jj_tms_pure_mdd_7_subjects.txt` and
`manifests/jj_tms_comorbid_mdd_19_subjects.txt`.

The default unversioned subgroup branch is 36P. Its subgroup FC caches are
isolated under
`outputs_iowa_jj_tms/inputs/kong2022/shared_inputs/pfm/{pure_mdd,comorbid_mdd}/`,
with downstream roots under `outputs_iowa_jj_tms/{work,present}/` using the
same unversioned subgroup names.

The completed 22P subgroup rerun reads versioned inputs from
`outputs_iowa_jj_tms/inputs/kong2022_22p/shared_inputs/{pfm,surface}/`.
Its FC caches use `inputs/kong2022_22p/shared_inputs/pfm/{pure_mdd,comorbid_mdd}/`.
Gradients and base group atlases use `pfm_22p/{pure_mdd,comorbid_mdd}/`
under their respective work roots; weighted group atlases use
`pfm_weighted_22p/{pure_mdd,comorbid_mdd}/`. Presentation roots use
`outputs_iowa_jj_tms/present/overviews/pfm_22p/{pure_mdd,comorbid_mdd}/`.
Each presentation directory contains four PNGs:
Fisher-z gradient-spectral, normalized Top-20 gradient-spectral, normalized
Top-20 WTA, and weighted-Fisher-z gradient-spectral.


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

The fixed Iowa Patient cohort is 26 subjects with sub-3003 excluded. The
active Patient branch uses the standard names
`pfm/patient` and
`pfm_22p/patient`; Lynch2024 7net uses the
corresponding `lynch2024_7net_iowa3t_patient*` roots. There is no parallel
27-person Patient branch.
The 22P inputs use `kong2022_22p/shared_inputs/`; downstream roots use
`pfm_22p/<cohort>/`. Unversioned inputs and `pfm/<cohort>/` are the 36P branch.

## JJ-TMS 22P core-workflow contract

The 22P cortex branch uses 22P XCP-D dtseries and matching 22P individualized
cortex labels. The hippocampal branch must use full, unsmoothed, denoised 22P
volumes for the same runs. The structural HippUnfold `T1w den-512` geometry
is independent of 22P/36P because it is derived from structural images, but the
sampled hippocampal BOLD time series are variant-specific and live under the
dedicated root:

`outputs_iowa_jj_tms/inputs/kong2022_22p/shared_inputs/surface/`

The available JJ-TMS 22P sources are:

- functional CIFTI root:
  `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS/xcpd_surf_22P`
- individualized cortex labels:
  `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS/ArealMSHBM_Kong2022_22P`
- full denoised 22P volume root:
  `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS/xcpd_vol_22P/restfmri`

The full denoised 22P volume runs are now available across all 94 subjects under
`xcpd_vol_22P/restfmri/sub-<id>/func/sub-<id>_task-rest_run-*_space-MNI152NLin6Asym_res-2_desc-denoised_bold.nii.gz`.
Do not substitute the generic `xcpd/restfmri` tree, which is a 36P derivative;
do not use raw fMRIPrep preprocessed volumes; and do not reconstruct a sparse
volume from CIFTI subcortical grayordinates. Pass 22P volumes through
`--run-bold-glob` or `--volume-root`, pair them with the 22P CIFTI roster through
`--run-dtseries-glob` or `--dtseries-root`, and require matching numeric run IDs
and timepoint counts. For a subject such as sub-1027, run-02/run-03 remain run
IDs 2/3. In core-workflow hippocampal surface extraction, surface and cortex run
blocks are concatenated only in memory; no concat Volume or CIFTI is written. Pass
the versioned 22P surface/reference roots explicitly to downstream commands.

The subject-level invalid-vertex threshold remains the standard 0.20. Do not
apply a sparse-CIFTI-specific threshold override.

Keep versioned outputs isolated. The 22P inputs use
`kong2022_22p/shared_inputs/`; gradients, group atlases, and presentations use
`pfm_22p/{hc,patient,pure_mdd,comorbid_mdd}/` under their stage roots.
The corresponding 36P roots use `pfm/{hc,patient,pure_mdd,comorbid_mdd}/`.
The Patient cohort contains the fixed 26 subjects.

For the complete HC/patient 22P overview rerun, use these isolated roots:

- gradients: `outputs_iowa_jj_tms/work/hippomaps_gradients/pfm_22p/{hc,patient}/`
- base group atlases: `outputs_iowa_jj_tms/work/group_atlas/pfm_22p/{hc,patient}/`
- weighted Fisher-z comparison: `outputs_iowa_jj_tms/work/group_atlas/pfm_weighted_22p/{hc,patient}/`
- overviews: `outputs_iowa_jj_tms/present/overviews/pfm_22p/{hc,patient}/`

Each cohort presentation directory contains four maintained products:
gradient-spectral `fisherz`, gradient-spectral `normalized_top20_count`,
gradient-spectral `weighted_fisherz` (the ROI-size-normalized/weighted Fisher-z
comparison), and WTA `normalized_top20_count`. Both products described as
"count" therefore use normalized Top-20 count. Raw `count` is obsolete and
must not be generated. Run the weighted branch with the versioned 22P
surface/reference roots above.

## Baseline tSNR inputs

The fixed-template matcher now requires `--tsnr-input-template` with
`{subject}`, `{hemi}` and `{domain}` (`hipp`/`cortex`). It computes per-run
baseline mean / detrended SD on retained frames (threshold 25), intersected
with finite/nonconstant analysis data. The old XCP `10000/SD` gate and the
interim validity-only rule are replaced. Baseline samples must use the consumed
mapping (the matcher and core workflow both use trilinear interpolation),
geometry and cortex vertex order. Both samplers call the shared
`lib.surface_sampling.sample_volume_to_surface` implementation; per-run
publication and matching outputs retain the JJ_TMS file layout.
See the core-workflow README's dataset-specific tSNR section for the NPZ input
format. Existing products are not automatically rebuilt by this code change.

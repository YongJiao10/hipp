# HCP tSNR: primary-source evidence

This note distinguishes published QC definitions from the project's inclusion
criteria. The current complete-cohort task procedure is documented in the
[task workflow](../hcp_3t_task_fmri_hipp_memory_emotion_plan.md).

## Task residual tSNR

Barch et al. (2013) calculate tSNR per run after 4 mm FWHM smoothing. The
denominator is the square root of FEAT `sigmasquareds`, representing residual
standard deviation after whitening and model fitting. Maps are averaged across
runs and participants to show signal loss. This definition does not validate a
hippocampal vertex cutoff of 25 or an all-runs-pass inclusion rule.
[Methods and QC results](https://pmc.ncbi.nlm.nih.gov/articles/PMC4011498/)

## Cortical surface mapping

Glasser et al. (2013) exclude ribbon voxels whose local coefficient of variation
exceeds the mean within a Gaussian neighborhood (sigma 5 mm) by 0.5 standard
deviations. This relative criterion accounts for spatial variation in coil
sensitivity and EPI signal loss. It describes voxel selection for cortical
mapping, rather than HippUnfold vertex or participant exclusion.
[fMRISurface methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC3720813/)

## Rapid functional QC

Marcus et al. (2013) describe `FUNCTIONAL_QC` after DICOM conversion, using
motion correction, brain extraction, intensity normalization, and voxelwise
mean/SD. Table 1 reports whole-brain mean tSNR across task and rest runs. A
descriptive distribution at that processing stage does not establish a
hippocampal acceptance threshold.
[FUNCTIONAL_QC and Table 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC3845379/)

## Processing version

The HCP 2025 release notes describe SEBASED bias correction, changes to motion
regression, 3T task multi-run FIX, and Reclean/temporal ICA for 3T and 7T fMRI.
They explicitly distinguish these products from S1200 (2017). QC descriptions
must identify the consumed processing version and whether the denominator is
time-series SD or model-residual SD.
[HCP release updates](https://wiki.humanconnectome.org/docs/HCP%20Data%20Release%20Updates%20Known%20Issues%20and%20Planned%20fixes.html)

The maintainer's explanation of
`rfMRI_REST_hp2000_clean_rclean_tclean.nii.gz` confirms improved bias correction
and spatial/temporal ICA cleaning in both volume and CIFTI products.
[HCP-Users explanation](https://groups.google.com/a/humanconnectome.org/g/hcp-users/c/X44pAWZ0gys)

These sources do not validate the combination of tSNR >=25 in every run and
participant exclusion at 20% invalid hippocampal vertices as an official HCP
standard. The project's current task workflow records residual tSNR without a
numerical cutoff; its runwise signal-validity, Dice, and coverage checks are
stated separately in the task guide.

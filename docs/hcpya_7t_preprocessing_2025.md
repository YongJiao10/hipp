# HCP-YA 2025 7T resting-state fMRI preprocessing

## Scope

This note refers to the HCP-Young Adult 2025 Release (released 11 August 2025),
not the older 2017 S1200 release. The official release page says that the 2025
release updates previously released 3T and 7T data and should not be mixed with
the previous S1200 release:

- https://icb.humanconnectome.org/study/hcp-young-adult/document/hcp-young-adult-2025-release
- https://humanconnectome.org/storage/app/media/documentation/HCP-YA2025/HCP-YA_2025_Release_Appendix_III.pdf

## What the recommended 7T rfMRI product contains

The official Appendix III describes the 7T recommended resting-state package as
per-run and concatenated CIFTI plus NIfTI products. It identifies the processing
as HCP Functional Preprocessing consisting of:

1. `GenericfMRIVolumeProcessingPipeline`;
2. `GenericfMRISurfaceProcessingPipeline`;
3. `hcp_fix_multi_run`;
4. spatial-ICA component reclassification;
5. `MSMAllPipeline`; and
6. `tICAPipeline`.

The HCP volume pipeline includes the HCP spatial preprocessing operations such
as gradient/susceptibility distortion correction, motion correction, EPI to
structural registration, atlas resampling, and intensity normalization. The
exact options are recorded by the HCP pipeline scripts; the generic volume
pipeline documents spin-echo TOPUP, motion correction, and SEBASED/legacy bias
correction options:

- https://github.com/Washington-University/HCPpipelines/blob/master/fMRIVolume/GenericfMRIVolumeProcessingPipeline.sh
- https://github.com/Washington-University/HCPpipelines/blob/master/fMRISurface/GenericfMRISurfaceProcessingPipeline.sh

For 7T, the recommended package has both 2.0 mm and 1.6 mm products. The
Appendix states that 2.0 mm is recommended for comparison with 3T, while 1.6 mm
is explicitly marked in the filename.

## Interpreting the filename

For a file such as
`rfMRI_REST_7T_hp2000_clean_rclean_tclean.nii.gz` or its matching CIFTI:

| Component | Meaning | XCP-D? |
|---|---|---|
| `hp2000` | HCP high-pass operation using 2000 seconds as the full-width parameter | No |
| `clean` | spatial ICA + FIX cleaned output | No |
| `rclean` | Reclean output, an improvement/reclassification stage for spatial ICA cleanup | No |
| `tclean` | temporal ICA-cleaned output | No |
| `Atlas_MSMAll` | atlas-space CIFTI after HCP surface alignment; it is not an XCP-D label | No |

The HCP ICAFIX script documents `clean` as sICA+FIX cleaning, permits
`clean_rclean`, and documents the default `motion-regression` option as false.
The HCP-YA 2025 release notes additionally state that movement-regressor
regression was removed as a step of fMRI data cleaning, while Reclean and
Temporal ICA outputs were added for all 3T and 7T fMRI data:

- https://github.com/Washington-University/HCPpipelines/blob/master/ICAFIX/README.md
- https://github.com/Washington-University/HCPpipelines/blob/master/ICAFIX/hcp_fix_multi_run
- https://icb.humanconnectome.org/study/hcp-young-adult/document/hcp-young-adult-2025-release

This does not mean that motion correction was omitted. Motion estimates and
`Movement_Regressors.txt` are still part of the HCP outputs; they are not proof
that those regressors were removed from the released BOLD time series.

## Relation to the current workspace

The local manifest points to the HCP-YA 2025 source files named
`rfMRI_REST_7T_hp2000_clean_rclean_tclean` for both the CIFTI and NIfTI products:

- `manifests/100610_minimal_manifest_runwise.json`

The current hippocampal GIFTI provenance for subject 878877 shows that the
`func_uncompressed.nii` used for HippUnfold sampling was derived from the HCP
2025 `rfMRI_REST_7T_hp2000_clean_rclean_tclean.nii.gz` volume together with the
HCP variance-normalization map. It is therefore a misleading filename if read
as “raw/unprocessed volume”; it is not the unprocessed acquisition:

- `outputs_hcp/inputs/qc/vertex_qc_t1only/shared_inputs_166/sub-878877/surface/raw/sub-878877_hemi-L_space-corobl_den-512_label-hipp_bold.func.gii`

## Analysis implication

HCP-YA 2025 native data use HCP processing rather than XCP-D. For a native HCP
analysis, use the matching HCP CIFTI for cortex and the matching processed NIfTI
for volume-to-HippUnfold sampling.

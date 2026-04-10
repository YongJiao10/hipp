# No-Separate-Volume Findings

## Scope

This note records the April 8, 2026 investigation into whether the current hippocampal functional parcellation
workflow can be run:

1. without reading a separate hippocampal volume BOLD file
2. without reusing hippocampal surface timeseries that were themselves derived from that separate volume file
3. while still producing the same style overview summary figure

## Bottom Line

```text
Question                                                                 Answer
-----------------------------------------------------------------------  ---------------------------------------------------------------------------
Can the current workflow be run with strict "no separate volume data"?   No
Can the current workflow be run from `dtseries` only and still be valid? Not without changing the hippocampal input model
Should we continue pursuing "completely no volume data" here?            No
```

## Findings

```text
Topic                                           Finding
----------------------------------------------  --------------------------------------------------------------------------
`dtseries` cortex representation                Surface grayordinates
`dtseries` hippocampus representation           Volume grayordinates, not hippocampal surface
Verification source                            Direct inspection of the actual `sub-100610 ... dtseries.nii`
Current workflow hippocampal input requirement  Hippocampal surface timeseries (`.func.gii` / `.npy`)
Existing hippocampal surface timeseries         Derived from separate 4D volume BOLD via `volume-to-surface-mapping`
Implication                                     Existing archived hippocampal `.func.gii` cannot satisfy the new constraint
```

## Evidence

### 1. `dtseries` does include hippocampus, but as volume grayordinates

The checked file was:

[sub-100610_task-rest_run-concat.dtseries.nii](/Users/jy/Documents/HippoMaps/data/hippunfold_input/sub-100610/func/sub-100610_task-rest_run-concat.dtseries.nii)

Validated facts from direct inspection:

```text
Check                                       Result
------------------------------------------  -------------------------------------------------------------
Maps to surface                             Yes
Maps to volume                              Yes
Cortex entries                              Listed as surface vertices
Hippocampus entries                         Listed as voxels
Left hippocampus count                      764 voxels
Right hippocampus count                     795 voxels
Conclusion                                  Hippocampus is stored as subcortical volume grayordinates
```

This means `dtseries` can replace the cortex input source directly, but does not by itself provide hippocampal
surface fMRI.

### 2. The existing hippocampal surface timeseries are not acceptable under the new rule

The current hippocampal surface files live under:

[surface](/Users/jy/Documents/HippoMaps/outputs/dense_corobl_batch/_archived_volume_functional/sub-100610/post_dense_corobl/surface)

They were produced by the post-HippUnfold volume-to-surface step:

[run_post_hippunfold_pipeline.py](/Users/jy/Documents/HippoMaps-network-first/scripts/run_post_hippunfold_pipeline.py#L145)
[sample_hipp_surface_timeseries.py](/Users/jy/Documents/HippoMaps-network-first/scripts/common/sample_hipp_surface_timeseries.py#L78)

Those scripts explicitly use a separate BOLD volume input:

```text
Input                                        Role
-------------------------------------------  ---------------------------------------------------------
`sub-<id>_task-rest_run-concat_bold.nii.gz`  Separate 4D volume BOLD source
`-volume-to-surface-mapping`                 Generates hippocampal surface `.func.gii` / `.npy`
```

So even if the current workflow no longer reads that volume file at runtime, reusing these archived surface outputs
would still violate the "no separate volume file" requirement.

### 3. A direct `dtseries hippocampus -> HippUnfold corobl surface` replacement is not ready

An isolated worktree test showed that directly separating hippocampal signal from CIFTI and then mapping it onto the
HippUnfold `corobl` surface does not currently provide a validated replacement path.

The observed blocker was geometric:

```text
Attempt                                             Result
--------------------------------------------------  ----------------------------------------------------------------------
Separate hippocampus from CIFTI                     Succeeds
Directly map separated hippocampal volume to surf   Invalid for current workflow geometry
Reason                                              CIFTI hippocampal volume model is not the same geometric object as the old dense BOLD volume path
```

So the current workflow cannot simply swap:

`separate volume BOLD -> hippocampal surface`

for:

`CIFTI hippocampal volume grayordinates -> hippocampal surface`

and expect the same overview outputs to remain valid.

## Interpretation

The main issue is not that the `dtseries` hippocampus is "wrong". The issue is that the current workflow expects a
specific hippocampal surface-timeseries input, while the available inputs split like this:

```text
Input source                             Usable now?  Why
---------------------------------------  -----------  ----------------------------------------------------------------------
Cortex signal from `dtseries`            Yes          Already surface grayordinates
Hippocampus signal from `dtseries`       No           Volume grayordinates, not surface timeseries
Archived hippocampal surface `.func.gii` No           Derived from separate volume BOLD, violates the new rule
```

That leaves no legal hippocampal input source for the current overview-producing workflow under the stricter
"no separate volume data" requirement.

## Conclusion

```text
Decision                                                      Status
------------------------------------------------------------  ---------
Keep pursuing "completely no separate volume data" here       Stop
Treat archived hippocampal `.func.gii` as acceptable here     No
Claim current workflow can satisfy the stricter requirement   No
```

The current project should not continue trying to validate a "completely no volume data" variant of the existing
workflow in this branch history. Any future attempt would first need a new, explicitly validated hippocampal input
model, not just a source swap.

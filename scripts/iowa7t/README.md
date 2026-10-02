# Iowa7T downstream tSNR

Both `run_branch_downstream.py` and `run_runwise_downstream.py` require explicit
baseline QC inputs through `--tsnr-input-template`. Supported placeholders:
`{subject}`, `{hemi}`, `{domain}` (hipp/cortex), `{variant}`, `{branch}` and `{run}`
(runwise adapter only). Inputs use the NPZ format documented in
`../core_workflow/README.md`. Preserve separate MB/ME baseline inputs and
variant-specific retained frames.

The gate computes per-run original mean / linearly detrended sample SD on
retained frames, requiring all included runs to reach 25 and the analysis
series to be finite and nonconstant. HCP's constant 10000 is removed. Missing
baselines cause an error. Use `--overwrite` for an approved complete surface,
cortex and dependent-FC rebuild; changing code alone does not change products.
Threshold 25 remains an operational comparison value, not a validated
cross-acquisition cutoff.

Existing gate and FC summaries must identify the new tSNR policy before cache
reuse. A changed gate also invalidates existing smoothing during the surface
stage. FC refuses old surface/cortex gates and recomputes old FC summaries.
For changing baseline samples under the same policy, explicitly use
`--overwrite` on the approved output tree.

# Iowa7T utilities

## HCP7T spatial subregion atlas

`export_hcp_subregion_atlas.py` exports the fixed 2mm HCP151 anterior/body/posterior
labels for matching HippUnfold 512-vertex surfaces, with MATLAB and Python
examples. The colleague-facing guide is
[HCP7T subregion atlas](../../docs/hipp_parcellation_network/hcp7t_subregion_atlas.md).
The requested delivery root is
`/home/yojiao/LSS/jianglab/3_Data_Working/HCP7T_subregion_atlas/`.

Stage to a new directory with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /home/yojiao/.conda/envs/py314/bin/python scripts/iowa7t/export_hcp_subregion_atlas.py --out-root .cache/NEW_SPATIAL_ATLAS_DIRECTORY
```

The exporter uses the formal fixed-template validator and manifest-declared
label files, plus the exact standard unfolded surface declared by each
hemisphere's 2mm group template spec. Both label grids must share its
topology. It does not change `HCP_templates` or regenerate scientific
results. The destination must not already exist.

## Downstream tSNR

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

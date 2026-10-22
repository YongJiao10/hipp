# Iowa7T LSS HCP templates

The portable template root is:
`/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/HCP_templates`.
Directory spelling is `HCP_templates`.

The package recipes use the complete HCP151 dataset and its formal inputs.

## Current sources

Use the explicit 151-subject QC roster:
`outputs_hcp/inputs/qc/vertex_qc_t1only/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_151.txt`.

- Gradient prototypes: `outputs_hcp/templates/hcp151_routea_full_k3_fisherz/`.
- Retained 400-ROI group WTA FC and support:
  `outputs_hcp/work/group_atlas/pfm/wta-top/kong2022/`.
- Original hippocampal time series/masks:
  `outputs_hcp/inputs/kong2022/shared_inputs/surface/`.
- Cortical ROI time series and subject ROI registries:
  `outputs_hcp/inputs/kong2022/shared_inputs/pfm/`.

## Template recipes

All packages have rows `2mm_L`, `2mm_R`, `4mm_L`, `4mm_R` and columns Kong2022
ROI IDs 1 through 400. Every source manifest must match the exact HCP151
roster. The consumed surfaces have 512 vertices per hemisphere.

- `gradient_spectral`: Anterior, Body, Posterior. Copy the current formal
  fixed-template prototypes: finite mean group Fisher-z vertex-to-ROI FC
  within supported, AP-ordered gradient labels.
- `wta_normalized/7net`: direct canonical7 Fisher-z WTA, with selected rows
  Default, VentralAttention, Somatomotor. Construct canonical7 network mean
  time series from the subject's equally weighted ROI means using the
  maintained Kong2022 merge mapping. Average subject vertex-to-network
  Fisher-z FC; assign each supported vertex across all seven networks by
  its largest finite value. Only then extract the three named networks.
- `wta_normalized/8net`: canonical8 normalized-top20 WTA, selected rows
  Default, VentralAttention, Auditory, Somatomotor.
- `wta_normalized/17net`: family17 normalized-top20 WTA, selected rows
  DefaultA, SalVenAttnA, DefaultC, SomMotB.

The last two recipes use the maintained normalized-top20 function on the
retained 400-ROI group Fisher-z bank, with the explicitly selected registry
column. It selects up to 20 positive ROI values and normalizes each network's
vote by its finite available ROI count. The current canonical7 normalized
WTA labels are not direct-Fisher-z labels and are not inputs to the 7net
package recipe.

For each WTA pseudo-cluster, average its supported, individually valid
hippocampal vertex time series within each subject, correlate that mean with
each cortical ROI time series, apply Fisher-z, then average over all 151
subjects. Keep unobserved rows/ROIs as NaN. Do not replace this operation by
averaging vertex-to-ROI correlations. The legacy `wta_normalized/7net` package
name stays in place; its metadata must declare `fisherz` / `top1_fisher_z`.

Stage with the explicit py314 interpreter and one native thread. Publishing
replaces only an approved exact file list in the existing LSS root; it does
not recompute Iowa7T subject matching or FC products.

Choose a new directory under `.cache/` and replace `NEW_OUTPUT_DIRECTORY`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /home/yojiao/.conda/envs/py314/bin/python scripts/iowa7t/export_hcp_templates.py --out-root .cache/NEW_OUTPUT_DIRECTORY
```

The exporter reuses the maintained
`build_normalized_topn_count_vertex_rows`, `build_fisherz_vertex_assignment_rows`,
`aggregate_kong2022_network_timeseries`, and `subject_label_fc` calculations.

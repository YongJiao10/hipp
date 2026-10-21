# Stanford 2 mm template-matched cluster FC

`04_02_compute_subject_template_cluster_fc.py` produces the subject-level FC
needed after HCP151 template matching. It uses each subject's own
`matching/2mm/hemi-{L|R}/labels.npy` and `valid_mask.npy`, averages the
subject's 2 mm hippocampal surface time series within labels 1, 2, and 3, and
then correlates those three time series with the same subject's 400 Kong2022
ROI time series. It does not average an existing `512 x 400` vertex-FC matrix.

The default source roots are the current materialized Stanford HCP151 inputs:

```text
outputs_stanford/work/hcp151_fixed_template_matching/subjects/
outputs_stanford/inputs/kong2022/shared_inputs/surface/
outputs_stanford/inputs/kong2022/shared_inputs/routea/
```

The surface arrays are `512 x T` and the ROI arrays are `400 x T`. The script
requires the exact 2 mm paths, validates all subjects before writing, and
fails closed for missing inputs, malformed labels, missing clusters, or
partial output trees.

Run the full 194-subject job serially after the dry-run passes:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/individual_workflow/04_02_compute_subject_template_cluster_fc.py --dry-run
/home/yojiao/.conda/envs/py314/bin/python scripts/individual_workflow/04_02_compute_subject_template_cluster_fc.py --resume
```

`--resume` reuses only a complete subject whose current input fingerprints and
local/external output hashes match its manifests. Existing partial or
conflicting files are never overwritten.

If an upstream JSON provenance manifest changes after the calculation while the
numeric inputs and outputs remain valid, refresh only the recorded fingerprints
with:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/individual_workflow/04_02_compute_subject_template_cluster_fc.py --refresh-provenance
```

This mode validates all existing matrices and their copies first, then updates
JSON provenance and records the refresh reason; it never recomputes or replaces
an FC array.

For each hemisphere, the repository-local output is written beside the
template-matching labels:

```text
outputs_stanford/work/hcp151_fixed_template_matching/subjects/sub-<id>/matching/2mm/hemi-<L|R>/
  cluster_mean_timeseries_fc_pearson.npy
  cluster_mean_timeseries_fc_fisherz.npy
  cluster_mean_timeseries_fc_manifest.json
```

The colleague-facing copies are written to:

```text
/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/template_matched_profiles/sub-<id>/
  sub-<id>_2mm_hemi-<L|R>_labels.npy
  sub-<id>_2mm_hemi-<L|R>_cluster_mean_timeseries_fc_pearson.npy
  sub-<id>_2mm_hemi-<L|R>_cluster_mean_timeseries_fc_fisherz.npy
  sub-<id>_2mm_cluster_mean_timeseries_fc_manifest.json
```

Every matrix has shape `(3, 400)`. Rows are `1=Anterior/VAN`, `2=Body/SMN`,
and `3=Posterior/DMN`; columns are Kong2022 ROI IDs 1 through 400 in the
subject registry order. The Fisher-z matrix is the template-matching analysis
scale, and the Pearson matrix is retained with the same cluster definition.
The destination root also receives `README.md` and
`template_matched_profiles_manifest.json`.

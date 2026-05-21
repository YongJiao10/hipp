# fMRI QC

The active default hippocampal group-analysis cohort is the T1only 151-subject QC-passed set.
Vertex-level QC is pre-Kong behavior and uses a branch-agnostic root:

- `outputs_core/inputs/hcp7t_hippocampus/qc/vertex_qc_t1only/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_151.txt`
- `outputs_core/inputs/hcp7t_hippocampus/qc/vertex_qc_t1only/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt`

Legacy `manifests/hcp_7t_hippocampus_fmri_qc_pass_153.txt` is T2only-era and is not the active default for T1only workflows.

Audit/provenance artifacts for current on-disk assets:

- `outputs_core/inputs/hcp7t_hippocampus/qc/vertex_qc_t1only/qc/hcp7t_t1only_vertex_qc_166_audit.tsv`
- `outputs_core/inputs/hcp7t_hippocampus/qc/vertex_qc_t1only/qc/hcp7t_t1only_vertex_qc_166_audit.json`
- `outputs_core/inputs/hcp7t_hippocampus/qc/vertex_qc_t1only/manifests/hcp_7t_hippocampus_fmri_qc_excluded_t1only_15.txt`
- `outputs_core/inputs/hcp7t_hippocampus/qc/vertex_qc_t1only/manifests/hcp_7t_hippocampus_fmri_qc_excluded_t1only_latest.txt`

Path plan (canonical):

- Root: `outputs_core/inputs/hcp7t_hippocampus/qc/vertex_qc_t1only/`
- Pass manifest: `manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_151.txt`
- Excluded manifest: `manifests/hcp_7t_hippocampus_fmri_qc_excluded_t1only_15.txt`
- 166-audit table: `qc/hcp7t_t1only_vertex_qc_166_audit.tsv`
- 166-audit metadata: `qc/hcp7t_t1only_vertex_qc_166_audit.json`

Kong2022 consumes this vertex-level QC cohort for downstream atlas/model steps; it is not the ownership root for QC artifacts.

QC criteria:

1. HippUnfold Dice `>= 0.70` for both hemispheres
2. fMRI null-vertex rate `< 20%` for both hemispheres

tSNR convention used for the null-vertex QC gate:

- The HCP minimal preprocessing pipeline intensity-normalizes fMRI data to a global 4D mean of `10000` (described by HCP as "like in FEAT").
- FSL FEAT describes intensity normalization as scaling each 4D volume so the global mean becomes a preset constant.
- For this branch's hp2000-filtered hippocampal surface data, the implemented proxy is `tSNR = 10000 / std_t`, with `ddof=1`; vertices with `sd == 0` are written as `NaN` and counted as invalid.
- This is a HCP-style tSNR proxy, not the generic textbook `mean_t / std_t` definition.
- If you need to describe it in a paper, say that the QC gate uses an HCP-normalized hippocampal tSNR proxy with threshold `25`, rather than implying a newly estimated mean signal.

T1only uses the same thresholds, but the consumed HippUnfold surfaces and Dice TSVs come from:

- `outputs_core/inputs/hcp7t_hippocampus/hippunfold/dense_corobl_batch/`

Excluded subjects:

- `sub-102816`
- `sub-109123`
- `sub-155938`
- `sub-158136`
- `sub-164131`
- `sub-173334`
- `sub-182739`
- `sub-198653`
- `sub-550439`
- `sub-680957`
- `sub-783462`
- `sub-878776`
- `sub-878877`
- `sub-898176`
- `sub-901442`

Current audited breakdown for 166 -> 151:

- `14` subjects fail the fMRI null-vertex threshold (`>= 20%` in at least one hemisphere)
- `1` subject fails Dice threshold (`sub-182739`, right Dice `0.69471951`)
- This split is reflected in `qc/hcp7t_t1only_vertex_qc_166_audit.tsv`

Active shared group-input root:

- `outputs_core/inputs/hcp7t_hippocampus/kong2022/shared_inputs/routeb_151/`

Example active raw surface path:

- `outputs_core/inputs/hcp7t_hippocampus/kong2022/shared_inputs/routeb_151/sub-<id>/surface/raw/sub-<id>_hemi-{L,R}_space-corobl_den-512_label-hipp_bold.func.gii`

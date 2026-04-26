# Kong2022 Route A — Official MSHBM Only

## Summary
Implement Route A using the official CBIG MSHBM MATLAB pipeline with Kong2022 400-ROI priors. Drop the FASTANS/PFM heuristic entirely. The new atlas slug is `kong2022_mshbm`; defaults are `w=100`, `c=50`; the current concat dtseries is split into four run/session dtseries and passed to MSHBM as four sessions.

## Key Changes
- Add a Route A MSHBM subject driver that:
  - validates `matlab`, `CBIG_CODE_DIR`, CBIG utilities, `CBIG_ComputeCorrelationProfile`, `mtimesx`, and fsLR surface helpers before running;
  - uses `external/Schaefer2018_Kong2022/Params_Final.mat` as `group_prior`;
  - generates fsLR32k `fs_LR_900` connectivity profiles via official CBIG profile generation;
  - calls `CBIG_MSHBM_generate_individual_parcellation(project_dir, 'fs_LR_32k', '4', '400', '1', '100', '50', 'test_set')`;
  - aborts if dependencies are missing; no heuristic fallback.

- Add a thin MATLAB wrapper instead of editing CBIG source:
  - sets paths for Kong2019_MSHBM `step1`, `step3`, and `lib`;
  - prepares `project_dir/priors/Params_Final.mat`;
  - writes `profile_list/test_set/sess1.txt` through `sess4.txt`;
  - calls official profile generation and official individual parcellation with `num_clusters=400`.

- Add CIFTI integration around the MATLAB output:
  - split each subject’s concat dtseries into four run-level fsLR32k dtseries using existing run boundaries;
  - convert MSHBM `lh_labels/rh_labels` to a 400-label CIFTI dlabel using the Kong2022 label list and ID/name registry;
  - output subject atlas files under `outputs_migration/cortex_mshbm/sub-{sid}/kong2022_mshbm/`;
  - write a summary JSON with paths, `w/c`, `num_sessions=4`, profile dimensions, observed label IDs, and dependency versions.

- Reuse Route B downstream by parameterization:
  - parameterize the Kong2022 ROI timeseries extractor with `--atlas-slug kong2022_mshbm` and `--kong2022-dlabel <subject dlabel>`;
  - parameterize the Kong2022 FC materializer with `--atlas-slug`;
  - extend the group atlas/presentation drivers to include `kong2022_mshbm` wherever the Kong2022 400-ROI family is supported;
  - keep strict 400-ID registry checks; do not allow name drift unless a later QC failure proves it necessary.

- Add a Route A pipeline driver:
  - stages: `prepare`, `kong2022_mshbm-individualize`, `kong2022_mshbm-reference`, `kong2022_mshbm-fc`, `intrinsic-fc`, `group-atlas`, `present`;
  - hard-cap concurrency at 10;
  - default full-cohort subject manifest remains `manifests/hcp_7t_hippocampus_fmri_qc_pass_153.txt`.

## Test Plan
- Preflight test:
  - verify `matlab` is callable;
  - verify `CBIG_CODE_DIR` resolves to the full CBIG tree;
  - verify required MATLAB functions and MEX dependencies are discoverable.
- One-subject smoke test on `100610`:
  - generate four run dtseries;
  - generate four profile `.mat` files with `profile_mat` shape `64984 x 1483`;
  - run MSHBM and verify `lh_labels/rh_labels` contain IDs in `1..400`;
  - convert to dlabel and verify 400-label map metadata.
- Three-subject pipeline smoke test on `100610 102311 104416`:
  - extract `reference/kong2022_mshbm` ROI/canonical8 timeseries;
  - materialize FC caches;
  - run one group atlas method/smoothing/hemi with `kong2022_mshbm`.
- Regression checks:
  - Route B `kong2022` outputs remain untouched;
  - `kong2022_mshbm` paths are separate;
  - docs and manifests state this is official MSHBM Route A, not FASTANS/PFM.

## Assumptions
- Use `w=100`, `c=50`.
- Treat the four HCP runs as four MSHBM sessions.
- Use slug `kong2022_mshbm`.
- No FASTANS/PFM fallback is implemented.
- If the local CBIG checkout lacks required utilities or MATLAB is unavailable, implementation stops at an explicit environment error rather than producing approximate outputs.

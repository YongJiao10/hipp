# Cross-dataset tSNR assessment — 2026-09-26

> **2026-09-26 用户最新决定：暂停所有新增筛选和由此触发的重跑，至少等 HCP 3T task 151 人数据完整后再判断。下文规则与人数是历史实现/敏感性结果，不是当前部署指令。已有task筛选结果未回滚，源码门槛未关闭。后续接手先读根目录 [TSNR_QC_HANDOFF.md](../TSNR_QC_HANDOFF.md)。**

This is a chronological investigation record. The initial audit was read-only; later sections describe code changes and the approved task rebuild. The earlier finite/nonconstant implementation was superseded by the real runwise tSNR code described in the final execution section below. Samples are illustrative, not cohort-wide estimates. No archive content was accessed.

## Current result

The implementation now calculates dataset-specific tSNR, while 25 remains an unvalidated operating cutoff. HCP 7T rest retains its HCP-specific intensity-normalized `10000 / SD` gate; this investigation did not establish that the HCP7T gate is wrong. Stanford, Iowa3T JJ_TMS, and Iowa7T rest use each run's baseline-preserving mean divided by linearly detrended sample SD, with every run required to reach 25. HCP 3T task uses each run's baseline mean divided by task-model residual sample SD, also requiring every run to reach 25. Each gate is intersected with finite/nonconstant signal validity. The HCP task products were rebuilt at 25; the threshold-sensitivity analysis below shows that this setting is not yet supported as a final scientific inclusion rule.

The HCP task products were rebuilt: final group size is 13 for EMOTION and 11 for WM, versus zero under the old gate plus the existing `<20% invalid` rule. Stanford/JJ_TMS and Iowa7T calculations below are impact comparisons; their production trees have not been rewritten. The corrected Stanford/JJ hippocampal roster counts are 180/194 (36P), 180/192 (22P), 82/94 (JJ36P), and 87/94 (JJ22P). Every audited Stanford/JJ cortex subject changes at least one vertex. Applying those masks requires updating each subject's ROI time series and FC, then dependent individual/group products. fMRIPrep, XCP-D, and HippUnfold need not be rerun for this mask change.

Iowa7T's available comparison found five multiband products that fail the existing 20% hippocampal-invalid rule and six multi-echo products that pass. Two multiband products lack the baseline runs declared by their dataset links and remain uncomputed. Existing rest outputs remain unchanged pending approval of the exact result paths to replace, as required by repository rules.

## Local implementation

- `scripts/core_workflow/02_01_prepare_group_inputs.py:364`: computes temporal SD with ddof=1 over the complete sampled series, then `10000 / SD >= 25`. This is equivalent to `0 < SD <= 400` for finite SD, not conventional vertexwise temporal mean / SD.
- `scripts/iowa7t/run_branch_downstream.py:281`: same formula and threshold. This is Iowa7T evidence, not proof of the Iowa3T launcher contract.
- `02_01_prepare_group_inputs.py:440`: mask feeds ROI smoothing.
- `scripts/core_workflow/02_02_build_qc_manifest.py:132`: both hemispheres must have invalid fraction strictly below 0.20 by default.
- `scripts/core_workflow/04_03_materialize_task_activation.py:110`: reads the mask before task GLM, setting invalid responses to NaN.

## Selected current files

For each root below, read `sub-<id>/surface/raw/sub-<id>_hemi-L_space-corobl_den-512_label-hipp_bold.func.gii`. Values computed using explicit py314, nibabel and numpy, one process, ddof=1. Passing counts independently apply the existing formula.

| Root | Subject | Frames | Median absolute temporal mean | Median SD | Passing / 512 |
| --- | --- | ---: | ---: | ---: | ---: |
| `outputs_stanford/inputs/stanford/kong2022/shared_inputs/surface` | 1017 | 231 | 0.2462 | 9.9651 | 512 |
| `outputs_stanford/inputs/stanford/kong2022_22p/shared_inputs/surface` | 1017 | 231 | 0.0947 | 11.3644 | 512 |
| `outputs_iowa_jj_tms/inputs/iowa_jj_tms/kong2022/shared_inputs/surface` | 1020 | 675 | 0.1794 | 17.7432 | 512 |
| `outputs_hcp/inputs/hcp/kong2022/shared_inputs/surface` | 115017 | 3600 | 0.00004 | 270.5630 | 510 |
| `outputs_hcp/inputs/hcp/kong2022/shared_inputs/task-emotion/surface` | 115017 | 352 | 11488.2341 | 580.2086 | 46 |

Existing gate summaries matched the counts for Stanford and HCP. The selected Iowa3T sample had no summary at the corresponding exact gate path: its count above is a hypothetical application of the formula, not confirmation of its actual historical gate.

Some existing Stanford36 and HCP rest sampling summaries contain legacy output paths. These were observed as metadata only; legacy paths were not followed. Therefore these samples do not constitute a complete input provenance audit.

Task sampling summary explicitly records two 176-frame EMOTION runs from `tfMRI_EMOTION_{LR,RL}_hp0_clean_rclean_tclean.nii.gz`, concatenated for sampling. In this left hemisphere:

- Median absolute difference between run means: 696.43.
- Run 1 median SD 439.90, 136 vertices satisfy SD <= 400.
- Run 2 median SD 416.18, 208 vertices satisfy SD <= 400.
- Removing each run's own mean before concatenation: median SD 427.20, 171 vertices satisfy the existing gate, versus 46 originally.
- Median vertexwise fraction of total variance attributable to between-run mean differences: 39.98%, using `1 - var(run-centered concatenation) / var(original concatenation)`.

This is a diagnostic calculation, not a proposed validated replacement gate. Run differences explain a substantial part of this sample's variance; remaining variance was not separated into neural/task and noise components.

## Interpretation and primary sources

1. HCP's [generic fMRI intensity normalization script](https://github.com/Washington-University/HCPpipelines/blob/master/fMRIVolume/scripts/IntensityNormalization.sh) scales global masked mean to 10000. This supports the HCP-specific `10000 / SD` convention for data that remain in that intensity scale; it does not establish that every hippocampal vertex has mean 10000. The evidence here does not require changing HCP 7T rest. Post-XCP-D centered/residual series from Stanford and JJ_TMS do not retain that same baseline convention, so carrying the 10000 numerator into those series is unsupported.
2. [HCP 2025 release updates](https://wiki.humanconnectome.org/docs/HCP%20Data%20Release%20Updates%20Known%20Issues%20and%20Planned%20fixes.html) describe multi-run FIX for 3T task, Reclean and temporal ICA for 3T/7T data. Clean/rclean/tclean should not be described as unprocessed task data. Project task GLM is a separate stage.
3. [HCP multi-run FIX source](https://github.com/Washington-University/HCPpipelines/blob/master/ICAFIX/hcp_fix_multi_run) specifies linear detrending for hp0. hp0 does not prove absence of all temporal processing.
4. [XCP-D workflow documentation](https://xcp-d.readthedocs.io/en/0.10.4/workflows.html) describes detrending/centering, filtering and nuisance regression. Centering alone does not change SD: `SD(x-c)=SD(x)`. Regression/filtering and intensity scaling can affect SD; low residual mean does not establish why amplitude differs across datasets.
5. [Nipype TSNR implementation](https://github.com/nipy/nipype/blob/master/nipype/algorithms/confounds.py) implements temporal mean / temporal SD. Replacing residual numerator with its near-zero mean is not an appropriate fix for centered residual series.

Uniform SD ceilings across unlike intensity scales and processing stages lack demonstrated cross-dataset validity here. High pass rates establish neither absence of downstream effects nor actual admission of poor-quality vertices. A plausible failure mode is that weak near-zero signals pass an upper-SD-only gate. Demonstrating real contamination requires independent quality evidence. No calibration evidence for the historical threshold 25 was established in this assessment.

A correction should define the intended quality property, use explicit dataset/run processing and intensity contracts, and separate acquisition quality from residual-noise assessment. Task run boundaries must be respected; task-associated variance should not automatically be treated as noise. Neither raising thresholds to achieve desired retention nor moving the unchanged formula after GLM is justified by this assessment alone.

## Historical follow-up: existing masks under the old formula

At this stage of the chronological investigation, the follow-up examined every subject directory under the exact current surface and routea roots listed in `docs/tsnr_existing_mask_scope_2026-09-26.ipynb`. The later sections record the code changes and task deployment. These populations are existing-product populations, not automatically final analysis cohorts.

### Confirmed scope

Iowa3T JJ_TMS is now directly confirmed: `scripts/jj_tms/jj_tms_hcp151_fixed_template_matching.py:453` implements `HCP_MODE / SD` with HCP_MODE=10000 and threshold=25, using XCP-D denoised volume/CIFTI inputs. `materialize_jj_tms_shared_inputs.py:104` copies its masks into the shared surface tree. Cortex also uses the formula, both in this JJ_TMS implementation and in `scripts/core_workflow/lib/io_utils.py:472`.

| Existing branch | Subjects with both hippocampal masks | Hippocampal excluded / total | Excluded by finite tSNR below 25 | Excluded by nonfinite tSNR |
| --- | ---: | ---: | ---: | ---: |
| Stanford 36P | 196 | 781 / 200704 (0.389%) | 0 | 781 |
| Stanford 22P | 192 | 518 / 196608 (0.263%) | 0 | 518 |
| Iowa3T JJ_TMS 36P | 94 | 111 / 96256 (0.115%) | 0 | 111 |
| Iowa3T JJ_TMS 22P | 94 | 21 / 96256 (0.0218%) | 0 | 21 |
| HCP 7T rest | 151 | 6186 / 154624 (4.001%) | 5554 | 632 |
| HCP 3T EMOTION | 25 | 19301 / 25600 (75.395%) | 19301 | 0 |
| HCP 3T WM | 25 | 19299 / 25600 (75.387%) | 19299 | 0 |

Every invalid hippocampal vertex in the four Stanford/JJ_TMS branches was checked against its current raw func.gii: all had exactly zero temporal SD, none had nonfinite raw samples, and all examined raw-derived masks matched stored masks. Thus removal of just the arbitrary upper SD ceiling, while retaining existing invalid/zero-variance exclusions, would change zero hippocampal mask entries in these populations. This does not establish equivalence to a new acquisition-quality mask.

Stanford36 QC has 199 rows: 194 pass all, four fail Dice (three also lack tSNR summaries), and sub-4117 fails the hippocampal coverage criterion with 26.953% left / 24.414% right invalid vertices. Its 263 invalid vertices are zero-variance, not finite high-SD exclusions. The other three audited Stanford/JJ_TMS branches have no existing hippocampal-mask subject at or above 20% invalid in either hemisphere. This is not a claim that all satisfy every other QC criterion.

For cortex routea, finite threshold exclusions were: Stanford36 1 out of 11,585,340 represented subject-grayordinates (195 subjects); Stanford22 7 out of 11,407,104 (192 subjects); JJ_TMS36 and JJ_TMS22 zero out of 5,584,728 each (94 subjects each). Other excluded cortical entries have nonfinite stored tSNR; their raw CIFTI values were not fully audited. HCP7 rest cortex had 8,071 finite threshold exclusions out of 8,971,212 entries. Small cortical counts alone do not prove small effects in particular ROIs.

The initial Iowa7T audit only measured exclusions under the old `10000/SD` gate; its apparent zero finite threshold exclusions is superseded by the runwise baseline-tSNR comparison below. The explicit NoXCPD subjects root was absent; no substitute was selected. Iowa ESTT's inspected local `outputs_iowa_estt/inputs/iowa` directory was empty, so no result-impact conclusion is available for ESTT.

### What is wrong versus what remains uncertain

- Applying the HCP arbitrary-unit ceiling as a valid cross-dataset tSNR test to XCP-D residual series is unsupported and needs correction. This does not show that XCP-D denoising itself was wrong.
- For HCP task, applying a total-SD ceiling before task modeling, across concatenated runs with different baselines, has a demonstrated substantial coverage effect. The numerical rejection fraction is not an estimate of the fraction that a validated replacement would restore.
- HCP7 rest has an actual nontrivial threshold effect. Its `10000 / SD` gate is supported by the HCP intensity-normalization convention for that branch; this audit does not establish that prior HCP7 exclusions were wrong or require changing them.
- Rest impact has two distinct parts: the direct effect of the arbitrary finite-SD threshold (quantified above), and failures to exclude genuinely low-quality but nonconstant series (not identifiable from old masks alone).
- Correlation itself is invariant to positive scalar rescaling. An intensity mismatch in this gate does not by itself prove that existing FC values are incorrectly scaled. Changes in masks, selected frames/runs, ROI membership, or quality contamination can change FC and later products.
- HCP-derived reference templates are used by other branches. If the HCP reference changes under corrected QC, downstream matching must also be assessed even where local XCP-D masks were unaffected.

### Corrective processing proposal (historical; implemented below)

1. Separate spatial coverage/registration/dropout and basic finite/nonconstant validity from acquisition tSNR and from post-denoising residual quality. Keep the 22P/36P and Iowa7T multiband/multi-echo scientific branches distinct.
2. For acquisition tSNR, use matched pre-denoising BOLD that retains baseline and calculate each run separately with the same sampling geometry as the analysis: temporal mean divided by detrended temporal SD. Document censoring/detrending stage. Do not estimate baseline from centered residual means or silently use 10000. If a matching input is unavailable, acquisition tSNR is unavailable until it is supplied.
3. For XCP-D rest, keep the intended denoised series for FC; assess FD, censoring, usable duration, residual artifacts and registration/coverage separately. A matched pre-regression mean divided by residual SD may be reported as an explicitly defined residual metric, with processing and intensity units matched; it is not interchangeable with acquisition tSNR.
4. For task, retain run boundaries, assess spatial coverage/motion first, fit the intended task and nuisance model per run, and inspect residual variance/model uncertainty and degrees of freedom. Do not use the unchanged rest total-SD gate before GLM or simply move it after GLM. A task GLM must not use already-zeroed excluded series to estimate the quality of the original vertex; start from the preserved raw samples when revising the mask.
5. Establish thresholds against quality evidence and repeat-run reliability, blinded to target scientific outcomes. No replacement numeric threshold has been validated by this audit. Do not tune thresholds to equalize cohort retention.
6. Compare existing and proposed masks, subjects/runs, and retained durations first. Recompute from the first changed dependency: hippocampal mask -> smoothing and FC; cortex mask -> ROI time series and FC; subsequent gradients, group atlas, template matching and presentation products depend on those changes. Structural HippUnfold need not be rerun solely because this gate changes.
7. Quantify scientific sensitivity at fixed cohort and then with revised inclusion: FC differences, aligned gradient correspondence, label agreement, and clinical/group effect estimates with uncertainty. Keep mask and cohort effects distinguishable. For changes to HCP reference templates, assess propagation into Iowa/Stanford matching.

[AFNI's official processing help](https://afni.nimh.nih.gov/pub/dist/doc/program_help/afni_proc.py.html) explicitly distinguishes post-motion-correction mean/detrended-SD from pre-regression mean/post-regression residual-SD. [FSL FEAT output documentation](https://fsl.fmrib.ox.ac.uk/fsl/docs/task_fmri/feat/feat_output.html) documents residual variance, estimated degrees of freedom, and contrast variance. [XCP-D workflow documentation](https://xcp-d.readthedocs.io/en/0.10.4/workflows.html) describes motion, DVARS, registration and censoring QC. These support the proposed separation of metrics; they do not validate a project-specific cutoff of 25.

The present audit resolves the existing-mask question. It does not yet quantify missed-quality exclusions or changes in final scientific conclusions: those require matched baseline/quality data and a justified replacement QC policy. Existing top-level Stanford and JJ_TMS source trees contain fMRIPrep/derivative and QC entry points, but their per-run completeness and matching have not yet been established. Production output repairs/rebuilds require an exact approved target list under repository policy.

## Task repair execution update

Following explicit approval of six task output directories, an interim
finite/nonconstant-only task rebuild was performed and then superseded by the
true runwise residual-tSNR rebuild described below. The interim 34816/34816
validity counts are historical and are not the current production masks. Rest
outputs were not rebuilt by this repair.

## Historical interim finite-validity implementation (superseded)

An interim implementation changed core preparation `02_01` and RouteA reference `03_02`/extractor to accept explicit
`--dataset stanford` or `--dataset iowa_jj_tms`. Both select finite, nonconstant
complete residual time series, without the HCP amplitude ceiling or task runwise
intersection. HCP remains the default dataset and retains its existing rest
formula; task behavior is retained. The dedicated JJ_TMS fixed-template entry
point now uses the residual-validity rule directly. Its current artifact layout
still writes legacy inverse-SD diagnostic arrays, but they no longer gate vertices;
parameters explicitly distinguish them from acquisition tSNR. No rest output
products were rewritten during this interim follow-up. Documentation was
updated with execution flags and cache behavior. The 95 tests reported at that
stage covered the interim rule; they do not verify the later true-tSNR code.

### Complete hippocampal comparison

All current raw surface series in the four audited branches were compared with
stored masks under that interim finite/nonconstant-only rule, including checks
for partial nonfinite data:
Stanford36 392 hemispheric masks, Stanford22 384, JJ_TMS36 188, JJ_TMS22 188.
Across all 1,152 masks there were **zero changed vertices** and zero vertices
with nonfinite raw samples. This interim result did not estimate the effect of
actual tSNR and is superseded by the full runwise tSNR comparison below.

### Interim cortex comparison under the old amplitude ceiling (superseded)

The finite stored tSNR exclusions identified in the prior audit all belong to
sub-1039. The exact declared CIFTIs and individual labels were read, and the
old ROI means were reproduced against stored ROI time series before evaluating
addition of those vertices. Only amplitude-ceiling removals were modeled; this
was not a full-cohort CIFTI scan for partial NaNs.

| Variant | Hemisphere | fsLR vertex IDs | Parcel | Existing -> proposed members | Old/new ROI time-series r | Maximum absolute hippocampal FC-r change, both hemispheres and 2/4mm |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 36P | R | 21307 | 276 | 166 -> 167 | 0.998149 | 0.037953 |
| 22P | L | 21236 | 9 | 105 -> 106 | 0.999766 | 0.011452 |
| 22P | R | 21289, 21307, 21335, 21360, 21370, 21371 | 207 | 126 -> 132 | 0.959298 | 0.116998 |

The 22P parcel-207 effect was material even under this ceiling-only comparison.
This limited comparison was superseded by true tSNR, which changes cortex masks
for every audited subject as quantified below. Sub-1039 is in the 36P NTHC/HC/nontrauma
and overall rosters; in the maintained seven-cohort 22P set it is in NTHC (and
the overall 192-person roster). Variant-specific labels differ; parcel 276 and
207 are not asserted to be the same parcel.

### Existing XCP-D QC is present and censoring is reflected downstream

Read exact branch-specific LINC QC reports from:

- Stanford36: `.../fmriprep_processed_CausalConnectome/rest_fmriprep/xcp_d_36P_Head05/sub-*/func/*_desc-linc_qc.tsv`.
- Stanford22: `.../fmriprep_processed_CausalConnectome/rest_fmriprep/xcpd_zr/xcpd_vol_22P/sub-*/func/*_desc-linc_qc.tsv`.
- JJ_TMS36: `.../fmriprep_processed_JJ_TMS/xcpd/restfmri/sub-*/func/*_desc-linc_qc.tsv`.
- JJ_TMS22: `.../fmriprep_processed_JJ_TMS/xcpd_vol_22P/restfmri/sub-*/func/*_desc-linc_qc.tsv`.

Here `...` is `/home/yojiao/LSS/jianglab/3_Data_Working`. All available hippocampal
subjects had QC reports: 196 and 192 Stanford runs; 281 and 280 JJ_TMS runs
across 94 subjects per variant. Retained frame counts agreed with available
Stanford reference summaries and with actual JJ_TMS ROI array dimensions.

| Branch | Retained frames per run min / median / max | Median / maximum censor fraction | Median / maximum post-censor mean FD (mm) |
| --- | --- | --- | --- |
| Stanford36 | 67 / 234 / 234 | 0 / 0.713675 | 0.090859 / 0.354884 |
| Stanford22 | 144 / 234 / 234 | 0 / 0.384615 | 0.089863 / 0.294678 |
| JJ_TMS36 | 142 / 230 / 230 | 0 / 0.382609 | 0.077511 / 0.274481 |
| JJ_TMS22 | 142 / 230 / 230 | 0 / 0.382609 | 0.077151 / 0.274481 |

The Stanford36 subject sub-1046 retains only 67 frames after censoring 167/234
(71.37%). Its consumed CIFTI has TR=2 seconds, hence 134 seconds retained; it is
included in the existing 194-person QC-pass roster. This merits a focused
usable-duration sensitivity analysis of groups containing it, rather than an
automatic exclusion or reprocessing every subject. It is absent from the audited
22P population. Stanford sub-4078 retains 144/234 frames; JJ_TMS sub-1030 and
sub-1003 contain the largest observed run censoring fractions. These observations
are not newly imposed exclusion thresholds.

The finite/nonconstant-only comparison identified sub-1039 cortex changes and
sub-1046 duration concerns. It does NOT establish the impact of genuine tSNR
QC, or limit the eventual correction to these subjects. See the true-tSNR
pilot below; the earlier inference of only local corrections is withdrawn.
Exact production replacement paths still require approval before execution;
existing rest products remain unchanged. Sources and executable read-only
analysis code are included in the existing mask-scope notebook's follow-up cells.


## True tSNR pilot and correction of scope claims

The user has no separate dataset thresholds and expected a fixed threshold.
Dataset-specific input selection does not require inventing a different numeric
threshold for every dataset. The existing value 25 is a common candidate to
assess, not a validated universal cutoff and not a target acceptance rate.

These calculations use unsmoothed 512-vertex hippocampal samples and the
branch's production mapping: Stanford and JJ_TMS 22P use trilinear Workbench
mapping; JJ_TMS 36P uses enclosing-voxel mapping. The initial JJ_TMS single
subject median-tSNR pilot used trilinear mapping and is descriptive only for
JJ_TMS 36P; the full-cohort table uses corrected per-branch sampling. Rest uses
fMRIPrep pre-XCP-D BOLD in MNI152NLin6Asym res-2 (234 frames for Stanford;
230 per sampled JJ_TMS run). No censoring or nonsteady-state removal was applied
in this pilot. Both the numerator and denominator come from that SAME sampled
series, not a pre-XCP mean divided by an unverified differently scaled residual.
The detrended estimate retains the original temporal mean in its numerator and
uses sample SD (ddof=1) after intercept and linear-trend removal.

Task estimates use the preserved raw hippocampal samples, split by declared run,
with each run's own baseline mean. The residual estimate uses the existing
EV + 12-motion + intercept + linear-trend design; no smoothing or prior QC mask
is used to estimate noise. Residual SD uses ddof=1, not a degrees-of-freedom
adjusted GLM variance estimator. Thus this is descriptive residual tSNR, not a
t-statistic. AFNI explicitly distinguishes detrended and model-residual tSNR:
[official afni_proc.py documentation](https://afni.nimh.nih.gov/pub/dist/doc/program_help/afni_proc.py.html).

The two definitions must remain explicit; identical numerical thresholds do not
establish identical rejection strength across them. The following remaining
pilot counts at 25 are sensitivity results, not newly applied production
exclusions.

| Dataset | Subject | Hemi | Run | tSNR definition | Median tSNR | Vertices ≥25 /512 |
|---|---|---|---|---|---:|---:|
| stanford | 1017 | L | single | linear_detrended | 92.33 | 512 |
| stanford | 1017 | R | single | linear_detrended | 66.47 | 512 |
| hcp_emotion | 115017 | L | tfMRI_EMOTION_LR | linear_detrended | 25.30 | 272 |
| hcp_emotion | 115017 | L | tfMRI_EMOTION_LR | task_model_residual | 26.09 | 303 |
| hcp_emotion | 115017 | L | tfMRI_EMOTION_RL | linear_detrended | 27.67 | 350 |
| hcp_emotion | 115017 | L | tfMRI_EMOTION_RL | task_model_residual | 28.64 | 371 |
| hcp_emotion | 115017 | R | tfMRI_EMOTION_LR | linear_detrended | 27.52 | 339 |
| hcp_emotion | 115017 | R | tfMRI_EMOTION_LR | task_model_residual | 28.60 | 357 |
| hcp_emotion | 115017 | R | tfMRI_EMOTION_RL | linear_detrended | 28.51 | 357 |
| hcp_emotion | 115017 | R | tfMRI_EMOTION_RL | task_model_residual | 29.51 | 383 |
| hcp_wm | 115017 | L | tfMRI_WM_LR | linear_detrended | 27.39 | 329 |
| hcp_wm | 115017 | L | tfMRI_WM_LR | task_model_residual | 27.87 | 344 |
| hcp_wm | 115017 | L | tfMRI_WM_RL | linear_detrended | 27.45 | 337 |
| hcp_wm | 115017 | L | tfMRI_WM_RL | task_model_residual | 27.90 | 351 |
| hcp_wm | 115017 | R | tfMRI_WM_LR | linear_detrended | 28.94 | 357 |
| hcp_wm | 115017 | R | tfMRI_WM_LR | task_model_residual | 29.54 | 368 |
| hcp_wm | 115017 | R | tfMRI_WM_RL | linear_detrended | 27.87 | 350 |
| hcp_wm | 115017 | R | tfMRI_WM_RL | task_model_residual | 28.40 | 364 |
| stanford | 1039 | L | single | linear_detrended | 47.62 | 450 |
| stanford | 1039 | R | single | linear_detrended | 40.44 | 422 |
| stanford | 1046 | L | single | linear_detrended | 25.43 | 263 |
| stanford | 1046 | R | single | linear_detrended | 21.93 | 208 |
| stanford | 4078 | L | single | linear_detrended | 38.19 | 399 |
| stanford | 4078 | R | single | linear_detrended | 28.26 | 306 |

Findings: true tSNR can exclude hippocampal vertices that the old residual
10000/SD gate and basic validity both retained. Consequently the previous
1,152-mask equality result cannot justify no downstream reruns under true tSNR.
Stanford sub-1039, sub-1046 and sub-4078 show this directly in the pilot. The
small, deliberately selected sample does not estimate cohort-wide impact.
Task model regression only modestly improves the pilot runwise tSNR; task
activation alone does not explain the original severe rejection.

Existing fMRIPrep and XCP-D products need not be recomputed just to calculate
these QC maps. Rerun decisions for smoothing, cortex references, FC and group
products depend on the finalized masks and retained runs/subjects. Production
rest outputs have not been replaced. At the time this chronological note was
written, true tSNR integration was outstanding; the implementation and impact
results below supersede that status.

## Actual tSNR implementation and task deployment

Implemented `scripts/core_workflow/lib/tsnr.py` and wired it into core hippocampal
preparation, RouteA cortex extraction, the JJ_TMS fixed-template matcher, and
both Iowa7T downstream adapters. HCP7T rest's existing formula is unchanged.
Rest requires explicit baseline samples and original retained-frame indices;
cortex additionally requires matching brain-model vertex order. Missing inputs
raise an error, without substituting basic validity. The final gate also checks
validity of the consumed analysis series. Threshold 25 is retained as the
operational setting; it has not been calibrated as a universal cutoff.

Task uses the original mean and residual sample SD separately per run, with the
existing EV + 12 motion + intercept + linear trend model. Every included run
must pass. The existing subject rule (both hemispheres <20% invalid, Dice >=0.7)
was retained. These choices change masks AND the final task cohort.

### Task: all 34 subjects per task, both hemispheres

| Measure | EMOTION | WM |
|---|---:|---:|
| Total hippocampal vertices | 34,816 | 34,816 |
| Original concatenated 10000/SD gate: passing | 8,741 (25.11%) | 8,630 (24.79%) |
| Interim validity-only gate: passing | 34,816 | 34,816 |
| Corrected per-run residual tSNR: passing | 25,174 (72.31%) | 25,463 (73.14%) |
| Recovered relative to original gate | 16,793 | 17,200 |
| Newly excluded relative to original gate | 360 | 367 |
| Subjects passing original gate + <20% invalid | 0/34 | 0/34 |
| Subjects passing corrected gate + <20% invalid | 13/34 | 11/34 |
| Cortex entries total | 2,020,008 | 2,020,008 |
| Cortex passing original gate | 1,462,857 | 1,461,497 |
| Cortex passing corrected gate | 1,962,980 | 1,961,308 |

Actual production task hippocampal QC, 2/4mm smoothing, cortex references,
2/4mm activation/background FC, group tables/maps and maintained 2mm figures
were rebuilt within the six previously approved task roots. The final group
cohorts are 13 (EMOTION) and 11 (WM), reduced from the interim validity-only
34-person groups. All 34 individuals have recomputed individual products.
The production renderer was fixed to select 2mm table rows explicitly; mixed
2/4mm tables previously caused duplicated-region errors. It still renders the
maintained 2mm figure; 4mm numerical products are present. Logs are under each
task shared-input root's `logs/tsnr_rebuild_20260926/`.

The 25/20% settings remain restrictive under corrected task tSNR. The code
correction does not promise that every subject passes. HCP7T rest templates
and all rest production trees have not been changed by this deployment.

### Rest pilot with actual retained frames

Stanford and JJ_TMS 22P use their existing trilinear sampling; JJ_TMS 36P uses
enclosing-voxel sampling. The earlier JJ_TMS pilot used trilinear for both and
is superseded for 36P by this corrected full-cohort calculation. Values below
combine run thresholds with current signal validity. Each hemisphere contains
512 vertices.

| Dataset / subject | Branch | L old -> new | R old -> new |
|---|---|---:|---:|
| Stanford 1017 | 22P and 36P | 512 -> 512 | 501 -> 501 |
| Stanford 1039 | 22P and 36P | 512 -> 465 | 512 -> 428 |
| Stanford 1046 | 36P | 512 -> 316 | 512 -> 253 |
| Stanford 4078 | 22P and 36P | 512 -> 434 | 512 -> 358 |
| JJ_TMS 1001 | 36P | 512 -> 501 | 512 -> 482 |
| JJ_TMS 1001 | 22P | 512 -> 509 | 512 -> 499 |
| Iowa7T 06112026 MB | 36P | 512 -> 319 | 512 -> 334 |
| Iowa7T 06112026 ME | 36P | 512 -> 501 | 512 -> 495 |

Iowa7T MB uses the XCP-declared fMRIPrep input in `derivatives`; ME uses the
XCP-declared `xcpd/tedana_input` baseline (post-MEICA, pre-XCP), preserving its
actual mean. These are different acquisition/processing stages. They are not
claimed to have identically calibrated tSNR scales. Retained frames: MB AP04
386, PA03 377; ME run01 399, run02 392. No Iowa7T production files were modified.

All 136 task hippocampal masks and the Stanford/JJ pilot masks were computed
through the actual `02_01` gate, and matched the direct numeric calculations.
All 136 task cortex masks were computed through `compute_cortex_tsnr_gate`.
Edited CLI modules loaded; syntax was parsed. No test suite was run.

### Full Stanford/JJ rest cohort impact

The audited candidate Stanford/JJ rest rule uses threshold 25 per run. For each run,
tSNR is the original retained-frame mean divided by the sample SD (ddof=1) after
removing a linear trend from those retained frames. Every included run must
pass, and the resulting mask is intersected with the existing finite,
nonconstant signal-validity mask. HCP7T rest remains unchanged.

The table compares this rule with the existing masks. Vertex counts sum both
hemispheres; the removal percentage is `(old valid - new valid) / old valid`.
The 20% column counts subjects with at least one hippocampal hemisphere at or
above 20% invalid vertices. These are mask-impact counts; they do not apply the
separate Dice eligibility check or define a final downstream cohort.

| Branch | Domain | Subjects with result | Subjects with changed mask | Old → new valid vertices | Old valid removed | New hippocampal 20% failures |
|---|---|---:|---:|---:|---:|---:|
| Stanford 36P | Hippocampus | 196 | 173 | 199,923 → 190,006 | 4.96% | 15 total (1 already failed before tSNR; 14 additional) |
| Stanford 36P | Cortex | 195 | 195 | 11,485,440 → 11,265,691 | 1.91% | — |
| Stanford 22P | Hippocampus | 192 | 169 | 196,090 → 187,038 | 4.62% | 12 |
| Stanford 22P | Cortex | 192 | 192 | 11,308,726 → 11,103,537 | 1.81% | — |
| JJ_TMS 36P | Hippocampus | 94 | 92 | 96,145 → 87,540 | 8.95% | 12 |
| JJ_TMS 36P | Cortex | 94 | 94 | 5,542,544 → 5,394,218 | 2.68% | — |
| JJ_TMS 22P | Hippocampus | 94 | 85 | 96,235 → 90,655 | 5.80% | 7 |
| JJ_TMS 22P | Cortex | 94 | 94 | 5,542,544 → 5,395,057 | 2.66% | — |

### Sampled FC effect from the cortex-mask change

For these seven subjects, the old ROI time series were first reproduced from
the stored masks. FC was then recomputed with the proposed cortex masks while
holding the saved hippocampal 2/4mm time series fixed. Thus the table isolates
the cortex-mask effect; it does not include the additional effect of changed
hippocampal smoothing. `Changed parcels` counts parcels whose mean changed.

| Dataset / subject | Changed parcels | Median absolute Δr | 95th percentile | Maximum absolute Δr |
|---|---:|---:|---:|---:|
| Stanford 36P / 1017 | 80 | 0.00417 | 0.04100 | 0.31958 |
| Stanford 36P / 1039 | 172 | 0.01061 | 0.11280 | 0.71744 |
| Stanford 36P / 1046 | 199 | 0.01564 | 0.19275 | 1.02615 |
| Stanford 36P / 4078 | 226 | 0.00842 | 0.10359 | 0.59235 |
| Stanford 22P / 1039 | 171 | 0.00844 | 0.09804 | 0.64844 |
| JJ_TMS 36P / 1001 | 68 | 0.00311 | 0.02600 | 0.12721 |
| JJ_TMS 22P / 1001 | 64 | 0.00267 | 0.02029 | 0.08566 |

These are targeted examples, not cohort-wide effect estimates. They show that
the cortex-mask changes can alter FC materially even when the removed cortex
fraction is small. Because every audited subject has at least one changed
cortex vertex, ROI time series and FC require recomputation for all such
subjects if the new masks are adopted.

The primary computation reports 2,300 branch/domain/hemisphere rows for 290
Stanford/JJ subject records and one error. Stanford sub-4117 had no existing
36P cortex reference summary, so the primary run has no cortex result for that
subject. Its error is retained in the notebook output. The separately computed
hippocampal supplement adds L 374 → 341 and R 387 → 361 valid vertices for
sub-4117. This subject already failed the prior hippocampal 20% rule and still
fails it under tSNR; it is included in the 36P hippocampal totals above, but not
the 195-subject cortex denominator.

Every subject with an available cortex reference has at least one changed
cortex-mask vertex in each branch. If this rule is applied to formal rest
products, those subjects need updated ROI and dependent FC products. This does
not require rerunning fMRIPrep, XCP-D, or HippUnfold. The formal rest output
trees have not been replaced; replacement requires the exact approved output
path list under repository policy.

### Proposed changes to existing formal subject lists

The branch-level 20% failures translate to these proposed list changes when
compared with the existing manifests. The Stanford 36P cortex result set has
195 subjects with references; the list comparison below starts from the 194
IDs in its named formal manifest. The result-set count is not a final analyzed
cohort size. Sub-4117 already failed the prior hippocampal 20% rule and is not
one of the newly failing IDs.

| Branch | Existing manifest | Existing → proposed passing IDs | Newly failing IDs under hippocampal 25/20 rule |
|---|---|---:|---|
| Stanford 36P | `outputs_stanford/inputs/stanford/kong2022/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_194.txt` | 194 → 180 | 1043, 1046, 1107, 2022, 2070, 2072, 3109, 3120, 4052, 4058, 4068, 4078, 4088, 4157 |
| Stanford 22P | `outputs_stanford/inputs/stanford/kong2022_22p/manifests/stanford_22p_qc_pass_t1only_192.txt` | 192 → 180 | 1043, 1107, 2022, 2070, 3109, 3120, 4052, 4058, 4068, 4078, 4088, 4157 |
| JJ_TMS 36P | `manifests/jj_tms_94_subjects.txt` | 94 → 82 | 1007, 1008, 1022, 1027, 1028, 1030, 1031, 1041, 3002, 3006, 3010, 3015 |
| JJ_TMS 22P | `manifests/jj_tms_94_subjects.txt` | 94 → 87 | 1022, 1027, 1028, 1030, 1031, 1041, 3010 |

These are proposed QC list deltas only. No manifest, formal rest output, or
final downstream cohort has been changed or inferred here.

### Corrected Stanford and JJ_TMS subgroup counts

These counts intersect the current cohort manifests with the hippocampal
20%-invalid failures above. They are proposed cohort sizes; formal manifests
and downstream results remain unchanged.

| Dataset / variant | NTHC | TEHC | NTS | NTS Any MDD | NTS Pure MDD | TIS | TIS PTSD | JJ pure MDD | JJ comorbid MDD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Stanford 36P | 48/51 | 37/40 | 54/56 | 41/43 | 11/11 | 41/47 | 31/35 | — | — |
| Stanford 22P | 48/50 | 37/39 | 54/56 | 41/43 | 11/11 | 41/47 | 31/35 | — | — |
| JJ_TMS 36P | — | — | — | — | — | — | — | 5/7 | 17/19 |
| JJ_TMS 22P | — | — | — | — | — | — | — | 6/7 | 19/19 |

### Iowa7T hippocampal comparison

The runwise baseline tSNR comparison used each exact DatasetLinks source and
its run-specific XCP-D outlier/censor data. Multiband uses its declared
pre-XCP fMRIPrep source; multi-echo uses `xcpd/tedana_input` (after tedana,
before XCP-D). Threshold 25 is applied per run, and every run must pass. All
existing masks and raw inputs were finite/nonconstant for the 22 computed
hemispheres; the differences below are due to the tSNR cutoff.

| Variant / subject / branch | Left valid / 512 | Right valid / 512 | 20% rule |
|---|---:|---:|---|
| 0P / 06122026 / multiband | 227 | 329 | Fail |
| 0P / 06122026 / multi-echo | 503 | 502 | Pass |
| 22P / 06112026 / multi-echo | 501 | 495 | Pass |
| 22P / 06122026 / multiband | 227 | 329 | Fail |
| 22P / 06122026 / multi-echo | 503 | 502 | Pass |
| 32P / 06122026 / multiband | 227 | 329 | Fail |
| 32P / 06122026 / multi-echo | 503 | 502 | Pass |
| 36P / 04172026 / multiband | 339 | 369 | Fail |
| 36P / 06112026 / multiband | 319 | 334 | Fail |
| 36P / 06112026 / multi-echo | 501 | 495 | Pass |
| 36P / 06122026 / multi-echo | 503 | 502 | Pass |

Two products could not be calculated because the exact baseline runs required by
their declared source roots are absent: 22P / 06112026 / multiband lacks AP04
and PA03 in `derivatives_synthstrip`; 36P / 06122026 / multiband lacks those
runs in `derivatives`. No alternate source was selected. The comparison covers
hippocampus only; Iowa7T cortex was not recalculated. No Iowa7T formal outputs
were rewritten.

### Formal rest roots awaiting approval

The four repository roots whose shared masks/references would change are:

- `outputs_stanford/inputs/stanford/kong2022/shared_inputs/`
- `outputs_stanford/inputs/stanford/kong2022_22p/shared_inputs/`
- `outputs_iowa_jj_tms/inputs/iowa_jj_tms/kong2022/shared_inputs/`
- `outputs_iowa_jj_tms/inputs/iowa_jj_tms/kong2022/shared_inputs_22p/`

Their downstream gradient, matching, group-atlas, and presentation roots are
not included in this list. No formal rest tree has been overwritten; the
repository policy requires the exact output-path list to be approved before
replacing or rebuilding those products.

## Post-review threshold sensitivity

An independent review found that the calculation changes were implemented
alongside an unvalidated hard cutoff of 25 and an all-runs intersection. To
measure the size of that choice, the saved per-run scores were re-evaluated at
10, 15, 20, 25, and 30. These are sensitivity values, not proposed replacements.
For rest, the existing finite/nonconstant validity of the analyzed series was
reapplied at each cutoff; at 25 the scan exactly reproduces the audited masks.
The all-runs-pass and subject-level `<20% invalid` rules were held fixed. Task
also kept the existing Dice requirement. No production output was changed by
this scan.

| Rest branch | Subjects in full score sweep | Cutoff | Valid vertices / total | Subjects with either hemisphere ≥20% invalid |
|---|---:|---:|---:|---:|
| Stanford 36P | 195* | 10 | 198,938 / 199,680 | 0 |
| Stanford 36P | 195* | 15 | 197,502 / 199,680 | 1 |
| Stanford 36P | 195* | 20 | 194,412 / 199,680 | 5 |
| Stanford 36P | 195* | 25 | 189,304 / 199,680 | 14 |
| Stanford 36P | 195* | 30 | 182,072 / 199,680 | 37 |
| Stanford 22P | 192 | 10 | 195,884 / 196,608 | 0 |
| Stanford 22P | 192 | 15 | 194,651 / 196,608 | 0 |
| Stanford 22P | 192 | 20 | 191,825 / 196,608 | 4 |
| Stanford 22P | 192 | 25 | 187,038 / 196,608 | 12 |
| Stanford 22P | 192 | 30 | 180,200 / 196,608 | 34 |
| JJ_TMS 36P | 94 | 10 | 95,409 / 96,256 | 0 |
| JJ_TMS 36P | 94 | 15 | 93,860 / 96,256 | 3 |
| JJ_TMS 36P | 94 | 20 | 91,234 / 96,256 | 6 |
| JJ_TMS 36P | 94 | 25 | 87,540 / 96,256 | 12 |
| JJ_TMS 36P | 94 | 30 | 82,881 / 96,256 | 28 |
| JJ_TMS 22P | 94 | 10 | 95,895 / 96,256 | 0 |
| JJ_TMS 22P | 94 | 15 | 94,969 / 96,256 | 0 |
| JJ_TMS 22P | 94 | 20 | 93,271 / 96,256 | 5 |
| JJ_TMS 22P | 94 | 25 | 90,655 / 96,256 | 7 |
| JJ_TMS 22P | 94 | 30 | 87,261 / 96,256 | 14 |

* Stanford 36P sub-4117 was calculated separately and is not part of this
threshold sweep. It already failed the historical 20% coverage rule; at 25 it
has 341/512 and 361/512 valid vertices. Thus the full 25-cutoff count remains
15 failing subjects when that separately calculated subject is included, as
reported above.

| HCP 3T task | Cutoff | Valid vertices / 34,816 | Pass tSNR + existing Dice + <20% invalid |
|---|---:|---:|---:|
| EMOTION | 10 | 34,646 | 34 / 34 |
| EMOTION | 15 | 33,673 | 34 / 34 |
| EMOTION | 20 | 31,005 | 27 / 34 |
| EMOTION | 25 | 25,174 | 13 / 34 |
| EMOTION | 30 | 16,620 | 0 / 34 |
| WM | 10 | 34,602 | 34 / 34 |
| WM | 15 | 33,681 | 34 / 34 |
| WM | 20 | 31,182 | 30 / 34 |
| WM | 25 | 25,463 | 11 / 34 |
| WM | 30 | 16,843 | 0 / 34 |

The all-runs intersection is a separate source of exclusions. The following
diagnostic counts the vertices that pass in every run versus at least one run,
before intersecting the rest analysis-series validity mask. The second number
is an upper bound for any run-specific-mask approach, not an adopted rule.

| Branch | Cutoff | Pass every run | Pass at least one run | Lost to the shared all-run intersection |
|---|---:|---:|---:|---:|
| Stanford 36P (one run) | 25 | 189,740 | 189,740 | 0 |
| Stanford 22P (one run) | 25 | 187,474 | 187,474 | 0 |
| JJ_TMS 36P | 20 | 91,328 | 95,556 | 4,228 |
| JJ_TMS 36P | 25 | 87,618 | 94,640 | 7,022 |
| JJ_TMS 22P | 20 | 93,289 | 95,956 | 2,667 |
| JJ_TMS 22P | 25 | 90,672 | 95,449 | 4,777 |
| EMOTION | 20 | 31,005 | 33,174 | 2,169 |
| EMOTION | 25 | 25,174 | 29,543 | 4,369 |
| WM | 20 | 31,182 | 33,185 | 2,003 |
| WM | 25 | 25,463 | 29,651 | 4,188 |

This confirms that 25 is a high-impact setting: lowering it to 20 changes the
task cohorts by +14 EMOTION and +19 WM subjects, and retains another 5,831 and
5,719 hippocampal vertices, respectively. Lowering it to 15 makes the subject
rule admit everyone, so that setting would provide almost no subject-level
screening. The steep change does not establish 20 as scientifically correct;
the audit contains no independent reliability or image-quality criterion that
validates either cutoff. It does establish that the task groups of 13 and 11
are conditional on an unvalidated cutoff and must not be presented as a
validated final QC cohort.

For Stanford/JJ_TMS, changing 25 to 20 would retain an additional 5,108, 4,787,
3,694, and 2,616 hippocampal vertices in the four branches above, respectively.
The old production masks remain in place. The proper next decision is to
validate the cutoff against independent acquisition-quality or run-reliability
evidence; until that decision, do not replace Stanford/JJ_TMS outputs on the
basis of the 25-cutoff comparison. If a different cutoff is adopted for the
already rebuilt task products, the task mask-dependent outputs must be rebuilt
at that adopted cutoff. A 20 cutoff changes at least one mask vertex for all
34 task subjects in both branches; adopting it therefore requires rebuilding
the mask-dependent individual outputs for all 34 subjects and the affected
group outputs. fMRIPrep, XCP-D, and HippUnfold are upstream of these masks and
do not need rerunning for that change.

The executable threshold sweep and its source paths are appended to the
companion notebook `docs/tsnr_existing_mask_scope_2026-09-26.ipynb`.

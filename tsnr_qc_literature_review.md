# tSNR and hippocampal fMRI quality control: primary-source review

Research date: 2026-10-18. This note records the primary-source review and original input inspection. The resulting executed comparison, including task residual-tSNR measurements and cross-dataset pass counts, is in [docs/tsnr_qc_comparison.md](docs/tsnr_qc_comparison.md). Project documentation and research notes are written in English.

Starting points were the existing [cross-dataset assessment](docs/tsnr_cross_dataset_assessment_2026-09-26.md), [HCP primary-source notes](docs/tsnr_hcp_primary_sources_2026-09-26.md), and parsed primary papers under `refs/`. Their historical population counts are not the current task denominator. Available PDF-derived prose was read from Markdown and structure/metadata from JSON; no archive content was accessed.

## Verified current project state

The authoritative task rosters contain **144 participants per task**, all within the 151-participant HCP 7T resting-state reference cohort. The task cohort is complete according to its current roster; 151 is not the task denominator. The two task rosters contain the same IDs. Their sources are:

- `outputs_hcp/inputs/hcp/kong2022/shared_inputs/task-emotion/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_144.txt`
- `outputs_hcp/inputs/hcp/kong2022/shared_inputs/task-wm/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_144.txt`
- Reference: `outputs_hcp/inputs/hcp/qc/vertex_qc_t1only/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_151.txt`

Read-only checks covered every task declaration, image header, required condition EV, movement-regressor array, RMS array, sampling summary, and existing hippocampal gate summary. No full BOLD signal audit or new screening was performed.

| Observed item | EMOTION | WM |
| --- | --- | --- |
| Participants / LR+RL runs | 144 / 288 | 144 / 288 |
| Declared BOLD / CIFTI files present | 288 / 288 | 288 / 288 |
| BOLD shape in every run | 91 × 109 × 91 × 176 | 91 × 109 × 91 × 405 |
| CIFTI shape in every run | 176 × 91,282 | 405 × 91,282 |
| Finite movement arrays of expected shape | 288, each 176 × 12 | 288, each 405 × 12 |
| Finite relative/absolute RMS arrays | 576, each 176 values | 576, each 405 values |
| Required finite three-column condition EV files | 576 | 2,304 |
| Sampling summaries | 144, trilinear, density 512, LR/RL order matches declarations | 144, same |
| Existing hemisphere QC summaries | 288 | 288 |
| Current numerical tSNR cutoff | None | None |
| Actual tSNR calculation recorded | **False in every summary** | **False in every summary** |
| Invalid vertices per hemisphere | 0–2 of 512 | 0–2 of 512 |
| Existing QC pass count / group roster | 144 / 144 | 144 / 144 |

The existing group rosters exactly match their authoritative task rosters. Group metadata currently permits a regional observation with at least 10% finite supported area and records zero coverage-excluded entries. Neither this setting nor the low null-vertex counts establishes absence of focal susceptibility dropout, correct registration, or reliable task effects.

The current `tsnr_gate` files are **signal-validity products**: finite, nonconstant signal is required in each run, with the existing Dice and hemisphere invalid-fraction rules applied separately. In [task_tsnr](scripts/core_workflow/lib/tsnr.py), `threshold=None` returns all-NaN score arrays and the validity mask. Thus a policy name containing `runwise_baseline_over_task_residual_sd` does not mean that residual tSNR was measured. If a tSNR diagnostic is later requested, calculate it separately from exclusion.

### Timing observations

All **576 consumed task CIFTIs** have a SeriesAxis step of **1.0 SECOND**. All **576 paired BOLD headers** record TR **0.72 seconds** with seconds as their temporal unit. The current activation summaries inspected for `sub-100610`, both tasks, also record 0.72 seconds, the constant used by [task_design.py](scripts/core_workflow/lib/task_design.py). This is a confirmed metadata disagreement; it does not demonstrate that the CIFTI signal was temporally resampled. The origin of the CIFTI time-axis metadata has not been established.

This matters because the official [HCP TaskfMRILevel1.sh](https://github.com/Washington-University/HCPpipelines/blob/master/TaskfMRIAnalysis/scripts/TaskfMRILevel1.sh) reads the consumed dense file's step interval when constructing a task analysis. Record and reconcile the acquisition/source timing before using these CIFTIs in a workflow that reads their headers. The present hippocampal GLM's explicit 0.72-second value is a separate fact; a metadata discrepancy alone does not establish that its betas need rebuilding.

Every EMOTION run has a final `fear` block extending beyond the 176 × 0.72 = 126.72-second image duration. For example, `sub-100610` LR lists onset 116.352 seconds and duration 18 seconds. The official [HCP task protocol](https://www.humanconnectome.org/hcp-protocols-ya-task-fmri) documents early termination before the final three trials of EMOTION's last block. Therefore the overrun is not, by itself, evidence for a global event offset or corrupt run. The current convolution clips events at the last frame. Confirm the delivered final-block duration against actual task timing before changing that model; no source-backed uniform subtraction of 8 or 10 seconds was found. WM condition events did not overrun the 405 × 0.72-second window in the complete inspection.

The checks above establish file availability, dimensions, finite tabular inputs, and these timing observations. They do not verify every signal value, image registration, behavioral response, motion burden, or scientific reliability.

## Recommendation for this project

**Do not adopt the tested mean-RMS0.5 rule (M) as the common main QC standard on the basis of Savage alone. Do not automatically restore the old tSNR cutoff of 25.** Expanded [primary-source review](docs/tsnr_task_rest_qc_justification.md) documents multiple authoritative frame-motion/data-sufficiency approaches and both descriptive and numerical-exclusion uses of tSNR. M remains a measured comparison, not a complete signal-quality screen. The [BANDA/PDC follow-up](docs/banda_pdc_qc.md) includes actual temporal motion peaks hidden by the all-pass mean gate. Use a definition-compatible published procedure rather than changing a cutoff to achieve a desired retention rate.

The complete 144-participant task cohorts have now been measured separately for LR and RL using the current model's residual SD. The existing masks remain unchanged. Savage et al. (2024) supplies a direct EMOTION mean-relative-RMS precedent, but not a sufficient basis to adopt that gate across task/rest. The expanded review also includes Tozzi et al. (Nature Medicine, 2024: contaminated-volume fraction plus image review) and Hack/Jubeir et al. (Nature Mental Health, 2025: FD-censored fraction plus tSNR>=50). Their definitions must be matched before transferring numerical thresholds to the hippocampal measurements.

This is one ordinary QC analysis, not a new QC-method validation project. The literature catalogue below supplies precedents and context, not additional required analyses.

## What the original sources actually did

### 1. HCP's original functional QC explicitly included tSNR

[Marcus et al., *Human Connectome Project informatics: quality control, database services, and data visualization*](https://pmc.ncbi.nlm.nih.gov/articles/PMC3845379/) was published online on 2013-05-24 and in the 2013-10-15 NeuroImage issue. Its rapid FUNCTIONAL_QC route applied MCFLIRT motion correction to the central frame, BET brain extraction, and one multiplicative scaling factor to a grand median of 10,000. Voxelwise tSNR was temporal mean divided by temporal standard deviation. Reports also included tSD, movement traces, DVARS, smoothness, and brain-mask medians. This rapid route differed from the full minimal-preprocessing pipeline.

In 20 unusually high-quality participants, WM/emotion activation Z statistics had correlations below 0.26 in magnitude with the evaluated QC measures. The authors explicitly cautioned that this sample did not establish effects at more extreme artifact levels. Their acquisition discussion also explains why small voxels and short TR yield lower tSNR than some conventional protocols. These results do not supply a portable hippocampal task tSNR cutoff.

[Glasser et al., *The minimal preprocessing pipelines for the Human Connectome Project*](https://pmc.ncbi.nlm.nih.gov/articles/PMC3720813/) (online 2013-05-11; issue 2013-10-15) supplies the relevant distinction between spatial corrections/registration and further processing. Accordingly, the stage and representation of the input must accompany every reported tSNR value.

### 2. HCP resting-state tSNR comparisons controlled the measurement stage

[Smith et al., *Resting-state fMRI in the Human Connectome Project*](https://pmc.ncbi.nlm.nih.gov/articles/PMC3720828/) (online 2013-05-20; issue 2013-10-15) computed protocol-comparison tSNR after motion correction and high-pass filtering, without spatial smoothing. The brain edge was eroded by three voxels before calculating a whole-brain median. This is a protocol evaluation, not a hippocampal subject-exclusion rule.

[Harms et al., *Extending the Human Connectome Project across ages*](https://pmc.ncbi.nlm.nih.gov/articles/PMC6484842/) (2018 online; 2019 journal issue) compared protocols after matching duration and applying the same minimal preprocessing and 2,000-second high-pass filter. It also provided separate phase-encoding maps. This supports matching preprocessing, duration, and acquisition context when interpreting differences, rather than directly comparing unmatched rest and task tSNR distributions.

### 3. The HCP task battery requires task-model evidence

[Barch et al., *Function in the human connectome: task-fMRI and individual differences in behavior*](https://pmc.ncbi.nlm.nih.gov/articles/PMC4011498/) (online 2013-05-16; issue 2013-10-15) models WM category crossed with memory load, and distinguishes broad functional mapping from isolating individual cognitive processes.

Crucially, its task tSNR maps used **each run's temporal mean divided by the square root of FEAT `sigmasquareds`**, after **4 mm FWHM smoothing**. That denominator estimates residual variance after whitening and model fitting. Maps were then averaged across runs and participants. The [locally parsed Methods](refs/hcp_task_fmri_hippocampus/Barch_2013_HCP_task_fMRI.md) explicitly document this definition. This is direct precedent for task-residual tSNR, but not for a hippocampal cutoff of 25, a minimum across runs, or subject exclusion at 20% invalid vertices.

Inference for this project: preserve the exact HCP task and contrast names. WM is not interchangeable with an episodic-memory localizer. Test temporal alignment, behavior, and the full run-level design before interpreting a hippocampal contrast. Weak or negative hippocampal effects remain possible scientific results after technical QC passes.

[Savage et al., *Dissecting task-based fMRI activity using normative modelling: an application to the Emotional Face Matching Task*](https://www.nature.com/articles/s42003-024-06573-z) (2024-07-20) gives a more recent, directly relevant task example. It excluded participants with **mean relative RMS > 0.5 mm**, added prefrontal coverage QC for UK Biobank, and harmonized smoothing at 5 mm FWHM. Its task model used faces/shapes regressors, HRF temporal derivatives, FILM prewhitening, a 200-second high-pass filter, and runwise estimates followed by within-participant combination where needed. It also evaluated held-out HCP retest data. The documented QC rule is a relative-RMS criterion, not tSNR. This supplies a concrete reference for reviewing EMOTION motion, not automatic validation of the same cutoff or smoothing for hippocampal subregional mapping.

### 4. A foundational hippocampal gradient paper used tSNR as a control

[Vos de Wael et al., *Anatomical and microstructural determinants of hippocampal subfield functional connectome embedding*](https://pmc.ncbi.nlm.nih.gov/articles/PMC6176604/) (online 2018-09-24; PNAS issue 2018-10-02) selected discovery and validation cohorts with four resting-state runs and acceptable image quality. It repeated gradient–T1w/T2w analyses controlling for local volume, CSF partial volume, and estimated rs-fMRI tSNR. It assessed within-person stability using two runs versus the other two and replication in an independent cohort.

This is direct precedent for checking whether a spatial biological association persists beyond local signal quality. It is not evidence for excluding task subjects below a particular tSNR. Detailed inclusion and preprocessing information is deferred to the supplementary appendix; this review did not retrieve and parse that appendix, so the exact supplementary tSNR input and any additional criteria are not asserted here.

### 5. The closest current hippocampal precision-mapping study used motion and geometry

[Kember et al., *The hippocampus becomes topographically and functionally specialized along the longitudinal axis with development*](https://www.nature.com/articles/s41467-026-74572-1) was first published on 2026-06-18; the [publisher's Crossmark record](https://crossmark.crossref.org/dialog-content?doi=10.1038%2Fs41467-026-74572-1&domain=pdf) lists a version of record from 2026-08-04. The current publisher Methods excludes participants with:

- Less than 10 minutes of clean resting-state data, defining low motion as FD ≤ 0.25 mm.
- HippUnfold surface reconstruction Dice below 0.7.
- At least **20 mm²** of hippocampal surface with improper fMRI interpolation.

The last quantity is an absolute surface area, not 20% of vertices. The Methods do not specify a numeric tSNR exclusion threshold. These are study-specific resting-state mapping criteria; copying the 10-minute criterion to a short task run would be inappropriate. If the project uses `invalid_fraction < 0.20`, describe it as a project rule, not as reproduction of the published 20 mm² rule. Surface area also depends on the actual subject mesh geometry.

### 6. HippoMaps supports checking mapping quality and reproducibility

[DeKraker et al., *HippoMaps: multiscale cartography of human hippocampal organization*](https://www.nature.com/articles/s41592-025-02783-3) was published on 2025-10-01. It evaluated cross-participant consistency and included independent 7T replication. Its task examples used GLMs with trial types, HRF terms, motion, and drift regressors. It warns that susceptibility dropout and anterior/posterior edge interpolation can affect maps. No universal task tSNR threshold is established in the accessed paper.

The separate [HippoMaps MRI-3T-rsfMRI tutorial](https://hippomaps.readthedocs.io/en/latest/tutorials/MRI-3T-rsfMRI.html) computes `mean(cdata_hipp)/std(cdata_hipp)` but displays group values around −0.000325 to 0.000367. These values warrant checking whether the input retained a meaningful baseline; the tutorial itself does not establish their cause. Inference: this example must not be copied as a calibrated tSNR exclusion rule. A tutorial's numerical output and a validated acquisition QC definition are different evidence.

### 7. General QC tools use several complementary measurements

The current [MRIQC functional IQM documentation](https://mriqc.readthedocs.io/en/latest/iqms/bold.html), accessed on the research date, defines tSNR as temporal mean/SD and summarizes the tSNR map by its median. It also documents DVARS, FD, ghost-to-signal ratio, outlier ratio, and nonsteady-state volumes. Its `fd_num`/`fd_perc` convention uses 0.20 mm for counting; this documentation does not make that count a universal subject-exclusion rule.

[Reynolds, Taylor and Glen, *Quality control practices in FMRI analysis*](https://pmc.ncbi.nlm.nih.gov/articles/PMC9922690/) (2023-01-30) demonstrates inspection of acquired data, alignment, temporal behavior, and statistical modeling. Its explicit example notes that low signal in a subcortical target may matter to one study while leaving a visual-cortex study usable. For this project, whole-brain QC must therefore be accompanied by hippocampal maps and inspection.

[Fair et al., *Correction of respiratory artifacts in MRI head motion estimates*](https://pmc.ncbi.nlm.nih.gov/articles/PMC7307712/) (online 2019-11-25; journal 2020-03) demonstrates respiratory contamination of multiband movement estimates, including HCP examples. It supports considering respiratory pseudo-motion before adopting a rigid FD gate. It does not justify silently applying an unvalidated filter to task BOLD data.

### 8. The newest direct alternative is FC quality, with a narrower intended use

[Lynch et al., *Objective quality assessment for precision functional MRI data*](https://pmc.ncbi.nlm.nih.gov/articles/PMC13288760/) appeared online in Neuron on 2026-06-22; the accessed author manuscript became available on 2026-06-24. Its published NSI is the **Network Similarity Index**: canonical network templates reconstruct each seed's full FC map using ridge regression; seedwise R² values are summarized by their median. It evaluates recoverable large-scale network structure in individual FC datasets, rather than relying only on scan duration or motion.

The authors explicitly show that registration failure can escape NSI and require upstream integrity inspection. Validation uses surface-registered CIFTI data; the paper does not establish a hippocampal task-activation exclusion threshold. Therefore NSI is a useful extension for resting-state or explicitly modeled residual-FC work, after geometry and preprocessing checks. It should not become an activation gate. Existing project `nsi` columns cannot be presumed to implement this published formula.

The authors' [PFM-NSI repository](https://github.com/cjl2007/pfm-nsi) documents the current Python/MATLAB implementations and the λ = 10, median-R² usability convention. Its calibrated usability projection is an FC-use model; it needs a declared input definition before project adoption.

### 9. Other existing project references answer different QC questions

[Leferink et al., 2024](https://academic.oup.com/cercor/article/34/1/bhad429/7439494), directly relevant to HCP WM scene/face preference, retained 157 of 181 participants after alignment/localizer checks. It visually reviewed alignment and used successful localization of category-responsive cortical regions as a selection criterion. That is a functional-localizer decision; importing a requirement for significant hippocampal activation would select on this project's target outcome. The [parsed Methods](refs/hcp_task_fmri_hippocampus/Leferink_2024_hippocampal_scenes_faces.md) distinguish these uses.

The existing [Kong et al., 2021 Methods](<refs/Kong et al. - 2021 - Individual-Specific Areal-Level Parcellations Improve Functional Connectivity Prediction of Behavior.md>) describe further nuisance regression/censoring after HCP ICA-FIX, reject resting runs with more than half their frames censored, and use out-of-sample task/rest data to evaluate parcellations. This supports recording usable data and independent evaluation; its resting-data rule does not establish a task tSNR threshold.

The [Ezama et al., 2021 preprocessing section](<refs/Ezama et al. - 2021 - Functional connectivity of the hippocampus and its subfields in resting-state networks.md>) uses HCP FIX-denoised, demeaned resting-state series with further CSF/white-matter regression. It provides another concrete reason to identify the baseline and processing stage before calculating mean/SD. Its accessed Methods do not supply a transferable numerical task tSNR gate.

## Minimal analysis supported by the review

1. **Keep the existing anatomical and signal-validity checks.** Current task masks and their Dice/coverage criteria remain the starting point. Kember's 20 mm² and the project's 20% criterion are different settings and should be described accurately.
2. **Keep vertexwise functional signal in the main comparison.** HCP task residual tSNR and matched-baseline rest tSNR are local surface measurements. Retain the existing T25 mask/coverage comparison, while preserving their different definitions and the current production threshold settings. A global head-motion pass does not identify usable hippocampal vertices.
3. **Compare a direct published hippocampal surface-coverage criterion.** Kember2026 (Nature Communications) excludes participants with >=20 mm² of improperly interpolated hippocampal surface due to insufficient functional gray-matter overlap. Test this surface-specific coverage criterion alongside vertexwise T25; its physical area is distinct from the project's 20% invalid-vertex rule. Mean relative RMS<=0.5 mm remains the supplementary motion candidate; per-run R20 is excluded from further consideration.

The executed comparison leaves production masks, rosters and results unchanged. The R20 plus existing anatomy/basic-validity results are supplementary checks, not final hippocampal surface signal-quality selections. BANDA/PDC volume-ROI measurements are explicitly separate from pending functional surface measurements. HCP7T motion is now complete for 166 participants: mean RMS0.5 passes 166/166 and R20 passes 134/166; the formal reference roster is unchanged.

## Measurement and literature justification

[Yeo et al., 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3174820/) explicitly used temporal mean/SD from motion-corrected data for exclusion: a subject was excluded if any run's whole-brain mean was below 100. [Buckner et al., 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3214121/) also lists SNR below 100 as an exclusion criterion, citing Yeo. These are direct precedents for tSNR-based QC, including screening, rather than just tSNR visualization. Their whole-brain criterion is different from the project's hippocampal vertex cutoff of 25.

For task-residual tSNR, Barch provides direct HCP task precedent. The available cleaned task inputs and existing OLS residual-SD calculation can provide a supplementary diagnostic when described accurately; they are not an exact reproduction of Barch's 4 mm-smoothed, whitened FEAT variance estimate. This distinction does not require rebuilding the model just to calculate the diagnostic.

Developing DVARS, NSI, ICC, contrast-variance exports, denoising comparisons, or a new screening-validation framework is outside the required QC work for this question.

## Timing observations and change scope

The CIFTI/BOLD time-axis mismatch recorded above matters to a consumer that uses the CIFTI header for timing. The current hippocampal GLM explicitly uses the matching BOLD TR of 0.72 seconds, so that observation alone is not a reason to block its analysis or rebuild its results. The EMOTION terminal event overrun must be interpreted in light of the official early-termination note; it does not justify shifting all events.

Apply changes only to the inputs/results actually affected. The task review does not require a rebuild of HCP 7T reference templates, Stanford/Iowa analyses, or upstream structural/preprocessing workflows. Formal output replacement remains subject to the repository's exact-path approval rule.

## Release boundary and evidence limits

The official [HCP-YA release update](https://wiki.humanconnectome.org/docs/HCP%20Data%20Release%20Updates%20Known%20Issues%20and%20Planned%20fixes.html) identifies the 2025-08-11 release, including multi-run FIX for 3T task data and Reclean/temporal-ICA updates. It explicitly says not to mix that release with the 2017 S1200 release. Declare the local release and processing variant; a completeness repair is not authorization to silently replace one task's preprocessing with a newer release.

The official [HCP QC issue documentation](https://wiki.humanconnectome.org/docs/HCP%20Subjects%20with%20Identified%20Quality%20Control%20Issues%20%28QC_Issue%20measure%20codes%20explained%29.html) also warns that missing issue flags do not guarantee acceptable data. Its task-data warnings about coil instabilities reflect the S1200 processing context and must be interpreted alongside the newer release documentation.

This was a targeted primary-source review, including current publisher/author sources through 2026-10-18, not an exhaustive systematic review. Some PMC direct requests returned browser challenges; indexed original text was used where available. No newly downloaded PDFs were needed. Supplementary materials not read are not treated as verified methods. No reviewed source establishes a universal hippocampal task tSNR threshold or validates the current project's complete task data merely from their presence on disk.

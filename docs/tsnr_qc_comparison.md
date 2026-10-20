# Existing-data QC comparison

Executed on 2026-10-18 on IDAS after discussion with pi's default model, `google-vertex/gemini-4-argon`. The comparison contains 1,054 participant-by-dataset-branch entries. Production masks, rosters, and analysis results were not changed.

**Interpretation:** these are measurements of existing sampled signals, including
HCP/Stanford corobl-to-MNI sampling now confirmed to be misregistered. They do
not quantify correctly localized hippocampal quality. Use the
[registered surface report](hipp_surface_qc.md) for the current five-dataset
comparison and the [motion report](qc_five_datasets.md) for completed HCP7T
motion. Numerical results below preserve the original comparison, not a
production recommendation. Iowa7T/JJ_TMS are outside the current scope.

There is direct precedent for tSNR-based exclusion, but no evidence here for a universal hippocampal cutoff of 25. [Yeo et al. (2011)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3174820/) excluded subjects if any run had whole-brain temporal mean/SD below 100. A directly relevant EMOTION study instead used mean relative RMS >0.5 mm for exclusion: [Savage et al., Communications Biology (2024)](https://www.nature.com/articles/s42003-024-06573-z). The recent hippocampal precision-mapping study by [Kember et al., Nature Communications (2026)](https://www.nature.com/articles/s41467-026-74572-1) used usable low-motion duration, reconstruction Dice, and interpolation coverage, without a numerical tSNR gate in its documented QC. These examples justify a conventional motion/anatomy/coverage approach; they do not establish the prevalence of any one practice across the literature.

## Fixed comparison definitions

Every scheme requires both hemispheres to have Dice >=0.70 and an invalid-vertex fraction **<0.20**. The 20% rule is the existing project rule. It is not Kember's **20 mm²** interpolation-area rule.

| Scheme | Additional functional definition |
| --- | --- |
| H: historical proxy | Finite concatenated series with sample SD >0 and <=400, equivalent to `10000/SD >=25`. This is a historical benchmark, not a common calibrated tSNR definition. |
| V: basic validity | Finite, nonconstant signal within each explicitly declared run; use the consumed concatenation where no acquisition partition is declared. |
| M: validity + motion | V plus the arithmetic mean of runwise mean relative RMS <=0.50 mm. Task uses `Movement_RelativeRMS.txt`; rest uses LINC `mean_relative_rms`. |
| C: usable-data sensitivity | M plus **each** declared run's existing censored/(retained+censored) <=0.50. No new frame censoring is introduced. |
| T25: numerical sensitivity | V intersected with runwise baseline/detrended tSNR >=25 for Stanford/JJ/Iowa rest, or current task-model residual tSNR >=25. HCP7T remains the separately labelled historical proxy because a matched baseline measurement is unavailable. T25 does not additionally impose M or C. |

M borrows Savage's participant relative-RMS threshold; aggregating the two task runs and extending it to WM/rest are explicit adaptations. This is **not a mean-FD threshold**. C borrows the >50%-censored run rejection from the [existing parsed Kong et al. (2021) Methods](../refs/Kong%20et%20al.%20-%202021%20-%20Individual-Specific%20Areal-Level%20Parcellations%20Improve%20Functional%20Connectivity%20Prediction%20of%20Behavior.md); requiring every included run to meet it is the stated participant-level adaptation. The task inputs retain all modeled frames, so their existing censor fraction is zero and C=M; this is not an independent demonstration of low framewise motion.

Task tSNR uses the existing model: condition regressors with the SPM HRF, 12 movement regressors, an intercept and 128-second DCT terms, OLS residual SD with `ddof=1`, and the run's signal mean as numerator. It is calculated separately for LR/RL, with their valid-vertex masks intersected. [Barch et al. (2013)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4011498/) supplies the HCP task-residual tSNR precedent, but used 4 mm smoothing and FEAT's whitened variance estimate. The present calculation is an explicitly described adaptation, not an exact FEAT reproduction or a published justification for 25.

## Main results

Cells show **confirmed passed / fixed total (percentage)**. When U>0, the percentage is confirmed retention from the fixed parent roster, not a complete QC pass-rate estimate. U denotes unassessable, not failed. The machine-readable summary also provides passed/evaluable separately.

| Dataset | Total | H: historical proxy | V: basic validity | M: + relative RMS | C: + censor fraction | T25: numerical sensitivity |
| --- | ---: | --- | --- | --- | --- | --- |
| HCP7T rest | 166 | 151/166 (90.96%) | 165/166 (99.40%) | 0 confirmed; U=165 | 0 confirmed; U=165 | 151/166 (90.96%); **proxy** |
| HCP3T EMOTION | 144 | 0/144 (0%) | 144/144 (100%) | 144/144 (100%) | 144/144 (100%) | 39/144 (27.08%) |
| HCP3T WM | 144 | 0/144 (0%) | 144/144 (100%) | 144/144 (100%) | 144/144 (100%) | 36/144 (25.00%) |
| Stanford 36P | 199 | 194/199 (97.49%) | 194/199 (97.49%) | 194/199 (97.49%) | 193/199 (96.98%) | 180/199 (90.45%) |
| Stanford 22P | 199 | 192/199 (96.48%); U=3 | 192/199 (96.48%); U=3 | 192/199 (96.48%); U=3 | 192/199 (96.48%); U=3 | 180/199 (90.45%); U=3 |
| JJ_TMS 36P | 94 | 94/94 (100%) | 94/94 (100%) | 94/94 (100%) | 94/94 (100%) | 82/94 (87.23%) |
| JJ_TMS 22P | 94 | 94/94 (100%) | 94/94 (100%) | 94/94 (100%) | 94/94 (100%) | 87/94 (92.55%) |

HCP7T M/C cannot supply a complete pass rate: 182739 already fails Dice, and the other 165 lack declared unpacked RMS/censor inputs. Do not read zero confirmed passes as 0% data quality. Its T25 column repeats H's proxy; it is not an independent matched-baseline tSNR result. V's extra 14 participants are not automatically approved for inclusion in the existing 151-person reference analysis.

Stanford's common parent roster has 199 participants. Four fail Dice (1055, 2001, 3046, 3107). In 36P, 4117 fails coverage; C additionally excludes 1046, whose source run retains 67 frames and censors 167/234 (71.37%). In 22P, 1046, 2072 and 4117 have no existing raw surface time series and remain unassessable; the 192 available products do not redefine the denominator. For that branch, passed/evaluable is 192/196 (97.96%) under H/V/M/C and 180/196 (91.84%) under T25.

Compared with M, T25 excludes another 105 EMOTION participants and 108 WM participants, versus 14 Stanford36P, 12 Stanford22P, 12 JJ36P and 7 JJ22P participants. This demonstrates different selection consequences, not a quality ranking between acquisitions. The historical proxy includes task-related variance and run differences in its concatenated SD; it is not the task-residual measurement.

## Task threshold sensitivity

The same freshly calculated residual-tSNR arrays were evaluated at all three cutoffs. Both hemispheres and both runs must meet the vertex/coverage rule. These are alternatives to M's additional motion criterion, not sequential exclusions after M (all task participants happen to pass M).

| Dataset | Total | T15 | T20 | T25 |
| --- | ---: | --- | --- | --- |
| HCP3T EMOTION | 144 | 144/144 (100%) | 125/144 (86.81%) | 39/144 (27.08%) |
| HCP3T WM | 144 | 144/144 (100%) | 128/144 (88.89%) | 36/144 (25.00%) |

The measured participant mean relative-RMS ranges are 0.0432–0.2417 mm for EMOTION and 0.0364–0.2165 mm for WM. No participant exceeds the predefined 0.5 mm rule. Choosing 15 simply because it retains everyone would be retention-driven tuning; the comparison does not recommend that substitution. It quantifies the threshold effect; it does not settle the local surface-quality criterion or justify replacing vertexwise QC with head motion.

## Existing Iowa7T products

These are small current subject-product populations, kept separate by preprocessing variant and acquisition. All percentages below use the fixed total shown. U is reported explicitly; a cell with U has an incomplete outcome. The four H/V/M/C columns coincide here, but remain distinct schemes.

| Dataset | Total | H | V | M | C | T25 |
| --- | ---: | --- | --- | --- | --- | --- |
| Iowa7T 0P multiband | 1 | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) | 0/1 (0%) |
| Iowa7T 0P multi_echo | 1 | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) |
| Iowa7T 22P multiband | 2 | 1/2 (50%) | 1/2 (50%) | 1/2 (50%) | 1/2 (50%) | 0/2 (0%) |
| Iowa7T 22P multi_echo | 2 | 1/2 (50%) | 1/2 (50%) | 1/2 (50%) | 1/2 (50%) | 1/2 (50%) |
| Iowa7T 32P multiband | 1 | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) | 0/1 (0%) |
| Iowa7T 32P multi_echo | 1 | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) |
| Iowa7T 36P multiband | 3 | 2/3 (66.67%) | 2/3 (66.67%) | 2/3 (66.67%) | 2/3 (66.67%) | 0 confirmed; 2 fail; U=1 |
| Iowa7T 36P multi_echo | 3 | 1/3 (33.33%); U=1 | 1/3 (33.33%); U=1 | 1/3 (33.33%); U=1 | 1/3 (33.33%); U=1 | 1/3 (33.33%); U=1 |

The populations are 06122026 for 0P/32P, 06112026+06122026 for 22P, and 04172026+06112026+06122026 for 36P. Shared anatomy for 06112026 has right Dice=0.6191, so it fails every scheme before further metrics are required. The 04172026 multi_echo product is absent. Under T25, 06122026 multiband retains L=227/R=329 of 512 vertices in 0P/22P/32P and fails coverage; 04172026 multiband retains L=339/R=369 and also fails. Only 36P/06122026 multiband has genuinely absent declared pre-XCP-D BOLD and remains unassessable. The evaluable 06122026 multi_echo products pass T25, but these tiny populations cannot calibrate a threshold.

Iowa multiband baseline sources are selected from each variant's `dataset_description.json` `DatasetLinks.preprocessed`: 0P/22P/32P explicitly name `/data/derivatives_synthstrip`, while 36P names `/data/derivatives`. The `/data` container prefix is explicitly mapped to the local Iowa7T root. Every Iowa baseline uses the sole shared-anatomy path declared by [the current layout contract](iowa7t_36p_hipp_outputs_layout.md), `DataOrg_PFM_36P/hipp_outputs/subjects/sub-ID/hippunfold/`. This also handles the documented layout migration for 04172026; its current surfaces agree with the existing consumed-asset size/mtime fields. No existence-based alternative is selected and no production migration is performed. Multi_echo uses the existing pre-XCP-D tedana-input baseline recipe.

## Inputs, reuse, and remaining availability

The calculation uses the 166-person HCP structural parent, both authoritative 144-person task rosters, the 199-person Stanford parent, and the comment-aware 94-person JJ_TMS roster. Exact paths and scientific variants are declared in [the script](../tmp/hippocampal_qc/scripts/tsnr_qc_comparison.py). The HCP reference151 and excluded15 explicitly select their respective existing raw roots by roster membership; dangling reference links in the excluded-set QC root are not used.

Basic validity was recomputed from current raw surface inputs, and available motion/censoring reports were read directly. Iowa7T T25 was calculated using the explicitly selected canonical anatomy and matched baseline frames. Stanford/JJ T25 reuses the saved numerical outputs in [the existing notebook](../tmp/hippocampal_qc/scripts/tsnr_existing_mask_scope_2026-09-26.ipynb), cell19 plus Stanford4117 in cell21. Hemisphere, variant, declared sampling method, 512-vertex count and retained frame count were checked against current inputs. These reused rest scores are not described as newly recalculated baseline time series. A one-participant JJ36 mapping check gave 67 invalid left vertices with enclosing sampling, matching the saved value, versus 43 with trilinear sampling.

For JJ, some vertices are constant in one run but variable in the concatenation. The notebook's saved T25 masks already included whole-series validity. If their invalid count is B and current runwise validity adds D invalid vertices, the combined invalid count lies between `max(B, current_invalid_count)` and `min(512, B+D)`. For every JJ participant these bounds settle the <20% decision; none straddles the boundary. The participant table records lower/upper fractions, and leaves the exact T25 fraction blank when the overlap is unknown. No per-vertex mask was reconstructed from a scalar. This reuses the existing measurements without changing the QC rule or rereading hundreds of baseline volumes.

| Other existing dataset | Declared population | Availability for these hippocampal schemes |
| --- | --- | --- |
| BANDA | 203 participants; 202 with 808 independent runs | The current [registered surface test](hipp_surface_qc.md) measures den-512 vertex QC; the [native-CIFTI follow-up](banda_pdc_qc.md) is a separate volume-ROI comparison. |
| PDC | 218 participants, 455 sessions, 984 runs | The current [registered surface test](hipp_surface_qc.md) measures available bilateral den-512 surfaces; missing visits remain unassessable. |
| Iowa esTT | No hippocampal analysis roster established | The exact 36P hippocampal subject-product root is absent. The existing 25-person cortex NSI analysis is not hippocampal QC. |

Archives were not accessed and no upstream pipeline was launched to fill these gaps.

## Initial pi discussion and implementation

The initial two discussion rounds reached agreement on the five schemes. Argon initially proposed mean post-censor FD gates; Codex challenged the interpretation because Kember's 0.25 mm is a per-frame criterion and FD is not relative RMS. Argon withdrew those gates and accepted the relative-RMS comparison. Both agreed to fixed parent denominators, the <=50% censoring boundary, and explicit missing-input counts. During measured-results review, Argon accepted the JJ union-count calculation and the M recommendation, and identified the Iowa variant-specific baseline sources and canonical surface-layout issue. Those inputs were corrected and all 14 Iowa entries recomputed, resolving four T25 cases as failures rather than missing data. Retention differences are interpreted as selection sensitivity, not proof that a protocol or participant is poor quality.

In the fourth round, Codex challenged Argon's stronger claim that the retention step itself proved a lack of cross-protocol robustness. Argon explicitly withdrew that inference. The discussion confined its numerical conclusion to selection sensitivity, identified Dice/coverage as project rules, and distinguished Savage's relative-RMS precedent from a complete published protocol. The current decision at the top of this report incorporates the subsequently expanded primary-source evidence.

The comparison used two analysis workers with numerical libraries limited to one thread, then serial Iowa7T sampling. Individual outcomes and all summary counts were checked directly, including known failures with missing inputs, task threshold nesting, and representative recomputations. No hashes, manifests, production masks, or new screening infrastructure were introduced.

Outputs:

- [Participant decisions and measured criteria](../tmp/hippocampal_qc/outputs/tsnr_qc_comparison_subjects.tsv)
- [All scheme/dataset totals, passes, failures, unassessable counts and rates](../tmp/hippocampal_qc/outputs/tsnr_qc_comparison_summary.tsv)
- [Analysis script](../tmp/hippocampal_qc/scripts/tsnr_qc_comparison.py)
- [Agreed implementation plan](superpowers/plans/2026-10-18-tsnr-qc-comparison.md)
- [Primary-source review](../tsnr_qc_literature_review.md)

To recalculate from the exact declared current inputs:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 CORE_WORKFLOW_WB_THREADS=1 \
CORE_WORKFLOW_WB_COMMAND=/home/yojiao/FASTANS/workbench/bin_linux64/wb_command \
/home/yojiao/.conda/envs/py314/bin/python -u tmp/hippocampal_qc/scripts/tsnr_qc_comparison.py
```

Supplementary R20 motion counts and the current surface/volume distinction are in [the follow-up comparison](qc_five_datasets.md). These original H/V/M/C/T25 results remain comparisons; production gates and the HCP7T resting reference are unchanged.

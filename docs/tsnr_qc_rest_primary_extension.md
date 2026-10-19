# Resting-state fMRI QC: focused primary-source extension

Reviewed 18 October 2026. This is a targeted comparison of established and recent
primary studies, not a systematic prevalence estimate. It addresses the earlier
proposal to use participant-mean relative RMS displacement <=0.5 mm as the common
main functional QC rule. No production masks or participant lists were changed.

## Finding

The exact 0.5 mm mean-relative-RMS rule has published precedents, but the sources
below do not establish it as a sufficient or generally preferred main QC rule.
Repeated practice across these studies combines motion treatment at the frame
level with enough retained data and appropriate anatomical/sampling checks.
Numerical participant-level tSNR exclusion is absent from the stated QC criteria
of the principal examples below. This does not imply that signal quality or
dropout was ignored, or that a tSNR exclusion rule can never be appropriate.

All participants passing a motion threshold does not logically invalidate it.
It does mean that passing it provides no differentiation within this sample;
its sufficiency cannot be established from its retention rate. Mean displacement
can also conceal brief large movements. Neither tightening the cutoff until
somebody fails nor selecting a cutoff to maximize retention is justified here.

## Exact units and levels of decision

| Primary study | Motion rule and decision level | Data sufficiency / other QC | Numerical tSNR exclusion in these stated criteria? |
|---|---|---|---|
| [Marek et al., Nature 2022](https://www.nature.com/articles/s41586-022-04492-9), main ABCD analysis | Censor frames above respiratory-filtered FD 0.08 mm; a separate 0.3 mm mask is used during nuisance regression | Participant inclusion requires at least 600 low-motion frames, equivalent to 8 min; full-brain coverage | No |
| [Gordon et al., Neuron 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5576360/), Midnight Scan Club | Power-style frame FD >0.2 mm is censored; special motion filtering is documented for two participants | 28% +/-18% of frames removed; even the worst participant retains almost 2 h. Cortical sampling excludes locally excessive coefficient-of-variation voxels | No uniform participant tSNR cutoff |
| [Gratton et al., Neuron 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC5912345/), same underlying MSC cohort | Frame FD >0.2 mm is censored in rest and task connectivity; segments <4 contiguous frames also removed | At least 25 good volumes per run and 50 per session per task; MSC08 excluded for high motion and reported sleep | No |
| [Ciric et al., NeuroImage 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5483393/), confound-regression benchmark | Participant exclusion: mean relative RMS >0.2 mm OR >20 volumes with relative RMS >0.25 mm. Separately, tested scrubbing uses Power FD >0.2 mm | 84 participants excluded for gross motion before benchmarking; residual motion-FC associations and data loss explicitly evaluated | No |
| [Power et al., NeuroImage 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC3849338/), canonical motion methods | Iterative example censors frame FD >0.2 mm OR initial DVARS >20; also removes segments <5 frames | Remove runs with <50 remaining volumes and participants with <125 across runs; criteria are specific to these acquisitions | No |
| [Kember et al., Nature Communications 2026](https://www.nature.com/articles/s41467-026-74572-1), HCP-D hippocampal surfaces | Low-motion frames have FD <=0.25 mm; participant-level exclusion is based on insufficient clean time | At least 10 min low-motion data, HippUnfold Dice >=0.7, and improperly interpolated surface area <20 mm^2; 471/652 retained | No |

These are not six independent votes: Gordon and Gratton share a dataset and
methods lineage; Ciric and Power are methods studies rather than hippocampal
application studies. Marek's numerical motion rule above describes its main ABCD
analysis, not a newly verified uniform rule for all its HCP/UKB comparisons.
Gordon's cortical coefficient-of-variation mask is not a hippocampal tSNR25 gate.
Kember's 20 mm^2 criterion is surface area, not 20% of vertices.
Its HTML Methods do not specify the FD formula, rotation radius, differentiation
lag or respiratory motion filtering. The linked precision_mapping README
describes mapping already processed GIFTI time series, not FD calculation;
additional source-code file requests were unavailable through the browser.
Consequently, applying a newly calculated Power FD 0.25 mm / 10 min rule here
would be a defined sensitivity comparison inspired by Kember, not a verified
exact reproduction of that study's motion calculation.

## Evidence specifically for mean-relative-RMS 0.5 mm

[Ahrends et al., Human Brain Mapping 2025](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.70409)
uses a participant-mean relative RMS threshold of 0.5 mm for resting-state data;
no participants exceeded it. Its full denoising also includes trained FIX ICA
classification and 24 motion regressors. This is a second explicit precedent
beyond Savage 2024, but it does not show that this threshold alone is sufficient.
Conversely, Ciric's explicit mean-relative-RMS exclusion is 0.2 mm, demonstrating
that RMS-based participant exclusion exists without establishing a universal
0.5 mm value. This focused search cannot support a percentage of the field or a
claim that one exact cutoff is used by most authoritative studies.

## Why FD, RMS, and their averaging cannot be interchanged

Power FD sums absolute changes in six realignment parameters after expressing
rotations as displacement. Relative RMS uses a different geometric definition.
Ciric explicitly treats relative RMS >0.25 mm and Power FD >0.2 mm as different
frame criteria. A frame threshold, a mean over frames, a run average, and a
participant average are different rules even when they share the same number.
Marek's respiratory-filtered FD 0.08 mm cannot be applied to an unfiltered FD or
RMS column as if the measures were equivalent. Power also warns that rapid-TR
acquisitions change the behavior of motion metrics across acquisitions.

## Recent context, without adding a new workflow

[Lynch et al., Neuron 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13288760/)
introduces the Network Similarity Index to assess interpretable individual-level
FC structure and argues that scan duration and motion alone are incomplete
proxies. Its core frame censoring uses respiration-filtered FD at 0.3 mm with a
two-TR backward difference, again differing from a simple mean-RMS rule. The PMC
version is an accepted manuscript. This work reinforces the limit of the earlier
M-only recommendation; it is not a requirement to implement NSI for this project.

## Selected reuse for the five requested datasets

The completed R20 comparison requires <=20% frames with relative RMS>0.25 mm
in every declared run; it is excluded from further consideration. Mean relative
RMS<=0.5 mm is the retained supplementary motion candidate. Primary QC concerns hippocampal surface
vertices, addressed through local sampling coverage and vertexwise signal
measurements as described in [the current decision](tsnr_task_rest_qc_justification.md). It is stated explicitly in
[Hearne et al., Science Advances 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8279516/)
and reused by [Keane et al., NeuroImage: Clinical 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11225660/)
in analyses of rest and task. This is one published methodological lineage,
not a prevalence estimate. The [task/rest source extension](tsnr_qc_task_banda_primary_extension.md)
and [executed five-dataset comparison](qc_five_datasets.md) give the exact rule,
measured pass counts and alternative FD-motion comparison.

Existing anatomical/signal checks accompany R20. BANDA/PDC use explicit native
CIFTI hippocampal ROI measurements, distinguished from pending den-512 samples.
HCP7T now has all 664 unpacked runwise motion traces, declared by the existing
`manifests/hcp7t_motion_transfer_20261019.tsv`. Mean RMS0.5 passes 166/166;
R20 passes 134/166. No archive access or new denoising workflow was required.

## Audit of proposed volume-based hippocampal rules

Reviewed 19 October 2026 against primary full text. Neither of the following
claims supplies an additional published surface-vertex exclusion threshold.

**Hickling et al. (2024): retain anatomical QC, omit the attributed +/-3 SD
volume gate.** The exact article is *Automated protocols for delineating human
hippocampal subfields from 3 Tesla and 7 Tesla magnetic resonance imaging data*,
Hippocampus 34(6):302–308, DOI 10.1002/hipo.23606. It develops ASHS atlases
using 140 participants scanned at 3T and 40 scanned at 7T. Its six regions
include DG/CA4, CA2/3, CA1, subiculum, pre/parasubiculum, and uncus. Validation
compares automatic with manual segmentations using Dice, with visual inspection
of their agreement. The primary main text does not specify subfield-volume
outliers beyond +/-3 SD as automatic exclusions or a functional-extraction QC
step. The supporting file is a DOCX and was not inspected in this audit;
therefore the claimed threshold is unsupported by the checked text, rather
than verified as a rule from this paper.
[Primary article, atlas construction and validation](https://onlinelibrary.wiley.com/doi/full/10.1002/hipo.23606).

**Ezama et al. (2021): keep the existing basic invalid-time-series check, omit
the attribution of an over-regression exclusion protocol.** The exact article
is *Functional connectivity of the hippocampus and its subfields in
resting-state networks*, European Journal of Neuroscience 53(10):3378–3393,
DOI 10.1111/ejn.15213; it is not a NeuroImage paper. Sections 2.1, 2.3 and
2.4.1 describe HCP QC exclusions, FIX denoising followed by CSF/white-matter
regression, and visual checking of subfield segmentations. Section 2.4.4
averages connectivity-map Z values within subfields. These Methods do not
describe rejecting subfields with near-zero temporal variance, an
over-regression test, or a numerical variance-collapse cutoff. A zero-variance
or nonfinite time series is an appropriate basic validity check in our
surface workflow, already represented by Signal_V; no new near-zero cutoff is
justified by this citation.
[Primary article, Methods](https://onlinelibrary.wiley.com/doi/full/10.1111/ejn.15213);
[existing parsed primary text, preprocessing and segmentation](../refs/Ezama_2021_Hippocampus_Subfields_RSFC/Ezama_2021_Hippocampus_Subfields_RSFC.md#23--preprocessing).

For the present hippocampal surface focus, use the existing anatomical checks
and nonfinite/constant-vertex masking. Do not start a separate subfield-volume
screen or an additional variance-threshold experiment on the basis of these
two attributed rules. The next substantive comparison should assess local
functional sampling coverage and vertexwise signal quality.

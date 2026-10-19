# Which hippocampal QC options to test

Primary-source assessment, 19 October 2026. The target is functional signal at
hippocampal surface vertices across HCP7T rest, HCP3T task, Stanford, BANDA and
PDC. The user has excluded per-run R20 motion screening. Mean relative RMS0.5
remains a supplementary motion candidate, separate from the local QC below.

**Prioritize functional surface coverage and vertexwise signal/tSNR.** Retain
existing midthickness sampling, anatomical checks and finite/nonconstant signal
checks. Do not implement all six proposed sets of rules. Keep the existing tSNR
definitions and selector counts as a reference; the
[five-dataset surface test](hipp_surface_qc.md) also measures them at registered
anatomical vertices after identifying a corobl-to-MNI sampling mismatch.

## Decisions by paper

| Proposed source | Decision for this project | Verified scope |
|---|---|---|
| DeKraker2025 HippoMaps | Keep existing midthickness sampling; no new outlier/profile/density workflow | Several claimed mandatory fMRI rules are study choices, histology methods or interpretive cautions |
| Kember2026 | Highest-priority new surface-QC comparison | Participant exclusion at >=20 mm² improperly interpolated hippocampal surface due to inadequate overlap with functional gray matter |
| Vos de Wael2018 | Optional downstream robustness check | Corrected G2–T1w/T2w structure/function association for local columnar volume, CSF partial volume and rs-fMRI tSNR |
| Barch2013 | Reuse existing task vertexwise residual-tSNR comparison | General HCP task-QA measure at voxel/grayordinate level, not a hippocampus-specific threshold |
| Hickling2024 | No new automatic volume-outlier exclusion | Main text validates ASHS segmentations using Dice/visual comparison; proposed +/-3 SD volume rule not verified |
| Ezama2021 | Keep current invalid/constant-series checks; no new near-zero threshold | Main methods do not document the claimed variance-collapse exclusion; journal is European Journal of Neuroscience |

## 1. HippoMaps: distinguish modalities and choices

[DeKraker et al., Nature Methods 2025](https://www.nature.com/articles/s41592-025-02783-3)
is a multimodal resource. Its histology procedures are not fMRI preprocessing.
The parsed [Methods](../refs/DeKraker_2025_HippoMaps/DeKraker_2025_HippoMaps.md)
and [page metadata](../refs/DeKraker_2025_HippoMaps/DeKraker_2025_HippoMaps.json)
locate Histology/MRI on physical page 13, Surface mapping on page 14, and the
edge caution on page 8.

- Midthickness: the fMRI examples use hippocampal midthickness sampling. This is
  a useful anatomical choice that reduces boundary contamination risk; the paper
  does not promise elimination of partial volume. Existing HCP/Stanford sampling
  records already select midthickness surfaces and record 512 sampled vertices.
- Density: the paper samples fMRI at approximately 2-mm spacing, 419 vertices.
  This is not a prohibition on 0.5-mm/7,262-vertex surfaces. Kember2026 itself uses
  the denser surface with 2-mm FWHM smoothing. Keep the current consumed surfaces;
  do not infer their physical spacing solely from a filename or vertex count.
- The ten-vertex-radius local 4-SD outlier rule and two-vertex dilation occur in
  Histology, addressing staining and missing-tissue boundaries. Do not transfer
  this to fMRI time series, tSNR or gradients as if it were published fMRI QC.
- The 25-depth microstructural-profile shift, bounded by 25% of thickness, also
  belongs to histological profile alignment. It does not apply to the current
  single-surface BOLD analysis.
- The anterior/posterior edge discussion cautions against interpretation of
  extreme values and limited sampling. It does not prescribe a fixed buffer
  width or automatic pole mask. Inspect these regions in existing QC views;
  do not discard them solely because the paper gives an interpretive caution.

The [official rs-fMRI tutorial](https://hippomaps.readthedocs.io/en/latest/tutorials/MRI-3T-rsfMRI.html)
also computes vertexwise temporal mean/SD tSNR, without a 25 exclusion gate.
It supports the measurement, not a transferable numerical threshold or an
assumption that every example input retains a suitable signal baseline.

## 2. Kember: surface coverage is the best new comparison

[Kember et al., Nature Communications 2026](https://www.nature.com/articles/s41467-026-74572-1)
reports the >=20-mm² interpolation-area criterion in Quality control. Parsed
[prose](../refs/Kember_2026_Hippocampus_Longitudinal_Axis/Kember_2026_Hippocampus_Longitudinal_Axis.md)
and [JSON](../refs/Kember_2026_Hippocampus_Longitudinal_Axis/Kember_2026_Hippocampus_Longitudinal_Axis.json)
place mapping on physical page 17 and QC on page 18.

This directly addresses whether the reconstructed hippocampal surface has
functional-gray-matter support. It is distinct from numerical noise amplitude,
and complements local tSNR. Calculate support at actual sampled vertices and
physical area on the subject's folded surface, rather than assuming equal vertex
areas or converting the project's 20% invalid-vertex rule into 20 mm².

The available paper text does not specify whether 20 mm² is per hemisphere or
bilateral, nor provide a numerical vertex-level definition of improper
interpolation. The user's 'any single hemisphere' wording is therefore not a
verified quote from the paper. Report left/right unsupported areas separately;
state the support-mask and hemisphere convention explicitly when implementing
the comparison. Matching a selected convention is a project implementation
choice, not a claim that every detail was published. The
[surface test](hipp_surface_qc.md) defines unsupported vertices as zero actual
trilinear functional-GM contribution, uses the native folded mesh for area,
and fails when either hemisphere reaches 20 mm2. It also records a stricter
nearest-voxel diagnostic separately. Official mapping code traced in the source extension did
not resolve these details; its separate 25-mm² cluster cleanup is unrelated.

## 3. Vos de Wael: a sensitivity analysis, not a mandatory exclusion rule

[Vos de Wael et al., PNAS 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6176604/),
Results subsection 'The Second Principal Gradient Reflects Internal Hippocampal
Microstructure', re-evaluates the association between G2 and T1w/T2w after
correcting local columnar volume, CSF partial-volume effects and temporal SNR.
This is not a universal rule to remove those covariates before estimating every
long-axis gradient, and it is not a participant/vertex rejection threshold.
Local columnar volume is a morphology measure, not by definition atrophy.
Use this as an optional later robustness check if the scientific analysis
interprets gradient patterns or their structural associations. Do not add three
new exclusion criteria or automatically regress anatomy out of the primary FC.

## 4. Barch: valid task-QA definition, overstated hippocampal specificity

[Barch et al., NeuroImage 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC4011498/)
creates runwise volume and grayordinate tSNR maps using the temporal mean divided
by the square root of FEAT sigmasquareds after whitening/model fitting, on data
smoothed at 4-mm FWHM; maps are averaged over runs and people. The parsed
[paper](../refs/hcp_task_fmri_hippocampus/Barch_2013_HCP_task_fMRI.md) and
[JSON](../refs/hcp_task_fmri_hippocampus/Barch_2013_HCP_task_fMRI.json)
locate the definition on physical page 17.

This is general HCP task QA. The text does not establish a hippocampus-specific
uncus/dropout protocol or a numerical cutoff of 25. Residuals are not guaranteed
to be pure background noise; they may include unmodeled signal and artifacts.
Applying this idea to hippocampal vertices is a useful stated adaptation.
The project's existing task calculation uses OLS residual SD with ddof=1 and
128-second DCT terms, not FEAT's whitened variance estimate. Preserve that
label. Rest signal uses its declared baseline/rest definition, not an invented
task model. Keep the available tSNR25 masks as a numerical comparison rather
than claiming that the paper calibrates 25 for all five datasets.

## 5–6. Keep existing anatomical/basic-signal checks

[Hickling et al., Hippocampus 2024](https://onlinelibrary.wiley.com/doi/full/10.1002/hipo.23606),
*Automated protocols for delineating human hippocampal subfields from 3 Tesla and
7 Tesla magnetic resonance imaging data*, develops and validates ASHS atlases.
Its main text documents segmentation comparisons, not the proposed automatic
+/-3-SD volume gate or functional-vertex exclusion procedure. The supplementary
DOCX was not examined, so the precise conclusion is that this attribution has
not been verified. An unusual volume can prompt anatomical review if there is
an actual suspicious segmentation; it is not sufficient alone to classify a
real anatomical difference as an error.

[Ezama et al., European Journal of Neuroscience 2021](https://onlinelibrary.wiley.com/doi/full/10.1111/ejn.15213)
uses HCP QC, FIX, CSF/WM regression and visual segmentation checks. The paper's
methods do not specify the claimed over-regression/near-zero-variance exclusion
or numerical threshold. Keep the existing finite/nonconstant-series checks;
there is no need to introduce a new acquisition-dependent 'near zero' cutoff.
The full extraction is in [the volume-source audit](tsnr_qc_rest_primary_extension.md).

## Minimal comparison

1. Preserve the current midthickness surface inputs and anatomical/basic-signal
   checks. No remeshing, profile sampling, histology filtering or pole trimming.
2. Retain the existing tSNR25 selector as a reference and recompute at registered
   anatomical vertices for the main comparison. Task residual tSNR, matched
   baseline rest tSNR and the HCP7T nominal-10000 proxy keep their stated definitions.
3. Add one new analysis: a functional-support map and unsupported surface area,
   using the Kember-inspired coverage criterion with an explicit convention.
   Compare it with the existing tSNR selection, then report their overlap; this
   does not require developing another QC metric or adding every listed method.
4. Reserve the Vos de Wael-style confound check for later interpretation of
   gradients. It is not a prerequisite for the first local QC pass-count table.

Codex and pi Argon agree on this priority after the primary-source audit. This
assessment does not change masks, rosters, surfaces or formal scientific results.

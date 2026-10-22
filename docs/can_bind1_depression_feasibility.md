# CAN-BIND-1: feasibility for the hippocampus and depression project

Evidence checked: 22 October 2026. This note distinguishes the study protocol,
published analysis samples, and the publicly described controlled release. No
participant files were accessed, and no workflow or output tree was changed.

## Verified study design

CAN-BIND-1 is an adult, multisite, open-label treatment study. Patients receive
escitalopram (10–20 mg/day) for eight weeks. Week-8 responders (at least 50%
MADRS reduction) continue escitalopram; nonresponders receive adjunctive
aripiprazole (2–10 mg/day) for another eight weeks. Treatment assignment is not
randomized and there is no placebo arm. The protocol collects MRI at baseline,
week 2, and week 8; week 16 includes clinical and molecular follow-up, not a
scheduled fourth MRI. Healthy comparison participants complete the same
applicable assessments without treatment, including repeat MRI. These facts
support studies of early functional change and later outcome, but a group
comparison cannot isolate a medication-specific causal effect.
[Lam et al., 2016, study protocol](https://link.springer.com/article/10.1186/s12888-016-0785-x).

The acquisition protocol confirms native resting-state BOLD at **4 × 4 × 4 mm**,
TR 2 seconds, 300 volumes (approximately 10 minutes), with **1 mm isotropic T1w**
structural imaging. Sites used different 3 T scanners and coils. Protocol
harmonization, acquisition checks, manual image QC, and monthly phantom scans
were used; these measures do not make the spatial information equivalent to a
finer native acquisition.
[MacQueen et al., 2019, MRI protocols, Tables 2–3](https://pmc.ncbi.nlm.nih.gov/articles/PMC6606427/).

## Sample counts must follow the analysis

| Population or analysis | Verified count and interpretation |
| --- | --- |
| Foundational cohort | 211 MDD and 112 healthy participants across six clinical sites; these are study cohort counts, not a usable matched imaging sample. [OBI release announcement](https://braininstitute.ca/news-events/2023/new-data-sheds-light-on-depression-treatment-responses). |
| Baseline + week-2 rsfMRI with week-8 outcome | Harris et al. had 157 complete cases, excluded 13 for severe imaging artifacts or incidental findings, and analyzed **144 MDD** participants. No further cases exceeded their mean-FD threshold of 0.55 mm. [Harris et al., 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9421454/). |
| Three-session, longer functional recordings | A later study selected **68 MDD + 39 controls** with complete, high-quality rest **and two tasks** at baseline, week 2, and week 8. This directly confirms repeat control MRI, but its approximately 30 minutes/session include task data and are not 30 minutes of pure rest. This restricted sample is not the maximum available rest-only cohort. [van der Wijk et al., 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11163402/). |

In the 144-patient study, baseline-only and week-2-only whole-brain FC classifiers
did not outperform default models significantly; baseline-to-week-2 FC changes
performed better. This supports investigating early change, while giving no
guarantee that a new hippocampal feature will predict individual response.
[Harris et al., 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9421454/).

## Release status and remaining availability questions

The current public CONP entry lists **version 2.0**, updated **12 June 2024**,
and explicitly offers baseline plus **Phase 1 (weeks 2–8)** for request. Access
requires a Brain-CODE application and committee review; approved files are
downloaded from Brain-CODE, not CONP. The listing reports **309 subjects**,
whereas its study description reports 211 + 112 = 323. Public metadata do not
explain the difference or establish matched raw T1w/BOLD/outcome counts.
[Official CONP dataset listing](https://portal.conp.ca/dataset?id=projects/braincode_CAN-BIND_Biomarkers_for_Depression_Baseline_Data_Release).

The March 2023 OBI announcement describes Phase 1 as released and Phase 2
(weeks 10–16) as forthcoming. No affirmative current official public listing
of the Phase-2 release was verified in this review. This is **unconfirmed
availability**, not evidence that those outcomes do not exist. Published
16-week analyses alone would not establish that external users can currently
obtain those variables.
[OBI announcement](https://braininstitute.ca/news-events/2023/new-data-sheds-light-on-depression-treatment-responses).

Before fixing the analysis sample, the access request should establish the
participant-by-visit intersection of raw T1w, native BOLD, clinical outcomes,
site/scanner identifiers, and QC/motion information; whether usable fieldmaps
or other distortion-correction inputs are included; and whether week-16 MADRS,
augmentation assignment, dose, and adherence variables are now available.
These are practical unresolved input questions, not verified omissions.

## Methodological interpretation

The treatment schedule and repeat untreated controls make CAN-BIND-1 useful
for longitudinal clinical validation. Baseline-to-week-2 change can be related
to week-8 outcome without requiring Phase 2. Analyses involving week-16 outcome
must wait for confirmed access to those variables. Repeated scans taken during
treatment should retain their visit identity: combining them solely to obtain
more resting-state minutes would change the scientific question.

Because the intervention is open-label and response determines augmentation,
describe associations with response or treatment-period change. A direct
escitalopram-versus-augmentation contrast would mix treatment exposure with
prior response and clinical course. Site/scanner and retained scan duration
also need to be handled in the analysis, particularly when using small
hippocampal regions.

## Fit to the current hippocampal workflow

The active project uses T1w-based HippUnfold anatomy, hippocampal surface FC,
group gradients, and externally trained anterior/body/posterior templates.
Its `2mm` and `4mm` template branches denote **surface smoothing**, not native
BOLD acquisition resolution. The existing BANDA/PDC structural presets consume
HCP bias-corrected T1w products; their functional input contract cannot simply
be assumed for a new raw-volume dataset.
[Core workflow](hipp_parcellation_network/core_workflow_overview.md),
[template matching](hipp_parcellation_network/crossfit_template_matching.md),
[structural input presets](environment/hippo2_reproducibility_manifest.md).

BANDA's original acquisition uses 2 mm isotropic BOLD, TR 800 ms, and four
resting-state runs; its MRI is cross-sectional, with longitudinal clinical
assessments. PDC studies rapid interventions including ketamine, ECT, and total
sleep deprivation. CAN-BIND therefore adds a distinct adult SSRI longitudinal
setting rather than an interchangeable imaging acquisition or treatment arm.
[BANDA dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11297143/),
[PDC official protocols](https://www.humanconnectome.org/study/crhd-perturbation-treatment-resistant-depression-connectome-fast-acting-therapies/project-protocol/imaging-protocols-pdc).

The following are methodological recommendations, not demonstrated CAN-BIND
hippocampal results. A native 4 mm voxel has eight times the volume of a 2 mm
voxel. The hippocampal ribbon is only about 0.5–2 mm thick, so partial-volume
mixing and shared source voxels are serious concerns for fine functional
maps. Interpolation onto a den-512 anatomical surface does not create 512
independent functional measurements. Good anatomical reconstruction and
nonzero functional coverage do not establish signal specificity; local cortex
controls cannot recover information already mixed within a source voxel.
[HippUnfold original paper](https://elifesciences.org/articles/77945).

- **Recommended primary role:** a separate longitudinal clinical validation
  cohort, using independently fixed whole-hippocampus or coarse anterior/posterior
  regions and prespecified cortical network FC measures. Analyze cohorts
  separately before comparing effects across datasets.
- **Conditional secondary role:** mean FC within externally fixed K=3 regions
  or broad long-axis organization, after demonstrating that these summaries
  remain distinguishable at this acquisition scale. De novo individual boundary
  displacement, fine proximal-distal gradients, and CA/DG-specific functional
  claims should not be primary endpoints with these inputs.
- **Useful clinical question:** whether baseline-to-week-2 hippocampal FC change
  relates to week-8 MADRS after accounting for baseline severity, site, and
  motion. Week-8 FC change versus concurrent symptom change is an association,
  not pretreatment prediction. Keep every visit from a participant in the same
  validation fold when estimating predictive performance.
- **Practical feasibility work:** inspect native EPI-to-anatomy alignment,
  hippocampal coverage, tissue mixing, and retained duration in a small paired
  sample. Spatially degrade suitable high-resolution project data to a 4 mm
  grid and match the approximately 10-minute duration to assess which planned
  summaries survive; a 4 mm smoothing kernel alone is not this experiment.
  This simulation cannot reproduce all CAN-BIND scanner and distortion effects.
- **Structural complement:** the 1 mm T1w scans permit a separately QC'd
  anatomical analysis, but do not improve the native functional resolution or
  guarantee fine subfield morphometric accuracy.

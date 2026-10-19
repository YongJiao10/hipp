# tSNR QC: current handoff

The immediate priority is a small, coordinate-only functional comparison before
final QC selection. Do not launch a production rebuild. The confirmed sampling
defect affects hippocampal functional inputs and their consumers; it does not
invalidate raw data or anatomical reconstructions by itself.

The active evidence review is
[docs/tsnr_task_rest_qc_justification.md](docs/tsnr_task_rest_qc_justification.md).
The [six-paper assessment](docs/hipp_surface_qc_options.md) prioritizes functional
surface coverage plus vertexwise signal/tSNR, reusing existing midthickness
sampling. Histology filters, depth-profile alignment and unverified volume/
variance exclusion rules are outside the current fMRI comparison.
The primary QC target is hippocampal surface vertices: local functional
sampling coverage and vertexwise signal/tSNR. Participant mean relative
RMS<=0.5 mm is the retained supplementary motion candidate. Per-run R20
is excluded from further consideration at the user's request. The existing
Dice and finite/nonconstant checks are prerequisites rather than proof of
adequate local BOLD signal. The current
[five-dataset comparison](docs/qc_five_datasets.md) covers HCP7T rest,
HCP3T task, Stanford, BANDA and PDC. Iowa7T/JJ_TMS are outside scope.

The main local comparison is the
[registered five-dataset surface test](docs/hipp_surface_qc.md), with 1,510
participant/session-by-branch entries and 45 summary rows. It compares surface
tSNR25, functional-GM unsupported area >=20 mm2 and their intersection, using
native folded midthickness geometry explicitly transformed to the functional
volume space. BANDA/PDC are now true surface projections, distinct from the
earlier native-CIFTI volume-ROI measurements. The primary-source review is in
[tsnr_qc_literature_review.md](tsnr_qc_literature_review.md). The active task
workflow is [hcp_3t_task_fmri_hipp_memory_emotion_plan.md](hcp_3t_task_fmri_hipp_memory_emotion_plan.md).

## Current inputs and outputs

- HCP 3T EMOTION and WM each have an authoritative 144-participant roster,
  with LR and RL for every participant. All task participants belong to the
  HCP 7T resting-state 151-participant reference cohort.
- Source data are at
  `/home/yojiao/LSS/jianglab/3_Data_Working/HCP_tast_3T/`.
  Exact run declarations are under
  `outputs_hcp/inputs/hcp/kong2022/manifests/task-emotion/` and `task-wm/`.
- Current task surface sampling is trilinear on consumed den-512 surfaces.
  Both task QC tables and group rosters contain all 144 task participants.
- Every current task gate summary records `threshold: null`,
  `tsnr_computed: false`, and `run_aggregation: validity_only_no_tsnr`.
  These are finite/nonconstant signal masks, not measured residual-tSNR maps.
- Dice >= 0.70 and invalid-vertex fraction < 0.20 in both hemispheres remain
  the existing project checks. They do not establish acceptable motion,
  registration, focal dropout, or task-model reliability.

## What the current review establishes

The retained mean-relative-RMS0.5 motion candidate comes from Savage2024
(Communications Biology, Quality control). It uses whole-scan motion estimates,
not local hippocampal signal. HCP7T passes 166/166 (100.00%), including 151/151
of the existing reference cohort. R20 is no longer an active candidate; its
completed research measurements remain in the numerical comparison tables.
A report-only tSNR is descriptive QA and has no direct inclusion/mask effect;
numeric tSNR exclusion also has published examples. The surface input audit
identified direct sampling of corobl coordinates into MNI functional volumes.
A same-volume sub-100610 EMOTION LR check exactly reproduces the old raw cache
with the wrong coordinates; correcting only coordinates changes vertex time
series profoundly (median old/new correlation approximately zero). This affects
the functional data and dependent FC, activation, gradients and functional
parcellations where those caches are consumed. It is not only a QC-threshold
issue. The new research test has not repaired formal downstream products.
The registered comparison and its complete retention table are in the main
surface report. HCP's new comparison also changes functional representation to
a hippocampal CIFTI-GM projection: its cohort-wide changes cannot be attributed
to coordinates alone. HCP7T retains its labeled nominal-10000/SD proxy. The
Kember-inspired coverage definition is a project adaptation, not an exact
replication of unreported implementation details. Production gates remain
unchanged. Yeo2011/Buckner2011 provide numerical tSNR-exclusion precedents;
Barch2013 supplies the HCP task-residual tSNR precedent.

Kember's interpolation criterion is 20 mm², not 20% of vertices.
The CIFTI timing issue affects header-driven consumers; the current
hippocampal GLM explicitly uses the BOLD TR of 0.72 seconds.

The generic preparation CLI still defaults to a numerical tSNR cutoff of
25. A future task preparation intended to preserve current validity-only
behavior must explicitly pass `--tsnr-threshold none`; do not launch the
screened default as a way to collect diagnostics.

## Authorized scope and downstream changes

The comparison uses current complete parent rosters, declared functional runs,
explicit anatomical transforms and matched baseline/retained frames. Existing
selector counts are retained separately from the registered calculations. It
writes standalone research tables, vertex maps and documentation. It does not activate a
production gate, change a cohort or mask, access
archives, or rebuild/replace formal results. A proposed production repair
must identify the exact affected paths for the approval required by this
repository. New measurement alone does not require a downstream rebuild;
changed models, masks, or cohorts determine the affected dependencies.

Completed alternative-scheme results are reviewable research outputs; formal
production masks and analysis cohorts are unchanged. The HCP7T reference
remains its existing 151-person cohort. HCP task completeness does
not establish a cross-dataset cutoff or require rerunning fMRIPrep, XCP-D,
or HippUnfold. The dated cross-dataset assessment under `docs/` is an
investigation record, not the active task workflow or a deployment instruction.

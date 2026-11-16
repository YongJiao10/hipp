# HCP7T Rest Rerun Implementation Plan

**Goal:** Rebuild the authorized HCP7T functional products using the accepted
motion, anatomy and NIfTI sampling-coverage rules, then cross-validate and match
held-out individuals to group-trained templates.

**Inputs:** The complete structural-source roster is
`manifests/hcp_7t_hippocampus_struct_complete.txt`. Physical-run motion paths
come from `manifests/hcp7t_motion_transfer_20261019.tsv`. Existing concatenated
BOLD and CIFTI files are read without creating new concatenated images. Native
T1w surfaces use the declared HCP native-to-BOLD registration.

## Constraints

- The user authorized an independent ten-process allowance for this HCP run.
  Pin numerical-library threads to one and count coordinators and subprocesses.
- Apply overall frame-weighted Power FD <=0.50 mm, bilateral Dice >=0.70 and
  at most 20 unusable NIfTI-sampled vertices per consumed 512-vertex hemisphere.
  A vertex requires positive image-grid support and a finite, nonconstant
  sampled signal in every declared physical run; no CIFTI hippocampal label
  is an additional anatomical requirement.
- Calculate participation from the rules. Neither 151 nor the previously
  observed 148 is an expected-result requirement.
- Preserve reference arrays, masks and ROI tables. Correct only reference JSON
  metadata after checking existing measurements and physical-run frame counts;
  the user explicitly requested this instead of a compatibility read mode.
- Retain but do not rerun `pfm_ctxdrop`, intrinsic spectral, Lynch or NSI.
- Never cluster individual hippocampal data. Group and training-fold clustering
  remain supported; held-out individuals receive template-matched labels.
- Use only the exact output roots approved in this conversation; no research
  output deletion or archive access is part of this run.

## Execution

1. Commit the completed removal of individual-clustering code and instructions.
   Completed in `b715994`.
2. Implement and test HCP7T QC without the historical HCP-result restriction.
3. Integrate actual BOLD NIfTI grid support and runwise sampled-signal validity
   into registered sampling and the consumed vertex mask. Keep unusable
   vertices excluded from smoothing and FC. Use explicit physical-run metadata
   for concatenated NIfTI sources; CIFTI remains a separate cortical input.
4. Correct retained reference JSON metadata from existing evidence. Populate
   missing reference inputs only when their declared cortical labels exist;
   missing scientific inputs must stop the affected branch.
5. Fix session-split comparison participant selection to consume the explicit
   QC roster, rather than reference-directory membership.
6. Run focused tests, then start registered sampling in parallel. Evaluate QC
   for the complete source roster, smooth passing participants and rebuild
   group-average surfaces from that measured roster.
7. Materialize PFM and cortex-regressed FC; compute their group gradients,
   gradient-spectral/WTA atlas products, the PFM family17 atlas, and the
   weighted Fisher-z comparison. The legacy `fixed` Route B and its
   `fixed_17net` derivative are outside this rerun. Pass `--force-final-k 3`
   explicitly to every applicable `05_02` call: K=3 is the fixed primary
   output, while `--default-k 3` alone still allows candidate-selection
   constraints to skip a combo with no surviving K. Run `pfm_17net` with its
   documented family17 group-atlas route; do not pass fixed-cluster reuse.
8. Render the base Fisher-z gradient, normalized Top-20 gradient, normalized
   Top-20 WTA, and weighted Fisher-z gradient overview products. Run family-aware
   cross-validation, held-out template matching and their robustness figures.
9. Inspect active processes, progress and errors every twenty minutes after launch.
   Report actual counts and incomplete dependencies without changing the roster
   to fit existing output availability.

## Validation

Use focused existing tests plus behavior tests for motion aggregation,
NIfTI support and runwise signal validity, QC output ownership and explicit session-split roster
selection. No new hashes, receipts or verification framework are required.

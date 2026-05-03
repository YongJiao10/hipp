# group-roi-sim-spectral Frozen Plan

- `status`: `frozen`
- `implementation_state`: `not_implemented_in_this_round`
- `default_flow`: `disabled` (not included in default method sets or urgent rerun bundles)

## Scope

This document freezes the design intent for `group-roi-sim-spectral` so it can be resumed later without ambiguity. This round does **not** implement or enable it in production defaults.

## Method Definition (Frozen)

1. Input family:
   - Subject-level hippocampal `vertex x ROI` FC (`512 x 400`) for `kong2022`.
2. Subject preprocessing:
   - Reuse spectral subject-FC preprocessing knobs:
     - keep largest raw-value Top-`p` entries per vertex row
     - set others to `0`
     - optional binarization of retained entries
3. Similarity transform:
   - Convert each subject preprocessed `vertex x ROI` matrix to `vertex x vertex` centered-cosine similarity.
   - Enforce finite symmetric matrix with diagonal `0`.
4. Group clustering path:
   - Aggregate subject similarity matrices and run spectral clustering.
   - Reuse the same K-selection diagnostics and stability checks as intrinsic spectral.
5. Annotation path:
   - For `kong2022`, reuse the same TopN positive-ROI majority assignment semantics used by spectral cluster-based methods.

## Expected Outputs (When Implemented Later)

- `group_labels_final.npy`
- `cluster_assignment_status.json`
- `cluster_network_profile.npy` (or equivalent cluster summary profile)
- `group_functional_atlas_labels.npy`
- `group_atlas_manifest.json` with full provenance and K-selection audit
- template `.label.gii` exports for cluster and functional atlas labels

## Required Tests Before Unfreeze

1. Subject preprocessing parity with `group-network-spectral` knobs.
2. Similarity matrix invariants per subject:
   - symmetric
   - finite
   - diagonal equals `0`
3. K-selection and stability path parity with intrinsic spectral branch.
4. Kong2022 TopN assignment parity (cluster-level) with existing spectral methods.
5. End-to-end smoke on RouteA/RouteB isolated roots with manifest/status artifact checks.

## Non-Goals For This Round

- No method default changes.
- No rerun launcher default changes to include ROI-sim.
- No integration into 8-way default pipelines.

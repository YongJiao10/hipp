# Group Hippocampal Atlas Plan v3

## Scope

This document defines a concrete implementation plan for building **group-level hippocampal functional atlases** in this repository using the default **153-subject QC-passed cohort**.

The fixed Kong group-analysis workflow now runs **three parallel group strategies**:

- `group-network-spectral`
  - canonical-network-driven hippocampal clustering
  - reference features come from merged canonical cortical-network Pearson FC profiles
- `group-network-partial-spectral`
  - canonical-network-driven hippocampal clustering
  - reference features come from one-sided networks-only partial FC profiles
  - for target network `j`, residualize the hippocampal vertex timeseries by all other canonical-network timeseries and correlate that residual with the raw target-network timeseries
  - do not residualize the target network and do not control other hippocampal vertices
- `group-intrinsic-spectral`
  - intrinsic hippocampal clustering
  - primary features come from hippocampal intrinsic coupling structure
  - cortex is used only for post hoc canonical-network annotation

Both strategies are:

- **data-driven** in hippocampal cluster count `K_hipp`
- **group-atlas only**
- **without individual inference**
- evaluated with the **same K-selection protocol** and the **same output contract**

This document does **not** decide which strategy should be treated as the main scientific result. The implementation responsibility is to run both and report both.

---

## Requirements

### R1. Cohort definition

- Use the default analysis cohort of **153 subjects** from:
  - `manifests/hcp_7t_hippocampus_fmri_qc_pass_153.txt`
- Do not silently expand to 166 subjects.
- Every output manifest must record the exact subject list used.

### R2. Group atlas only

- This plan stops at the **group-level hippocampal atlas**.
- `individual inference` is explicitly out of scope.
- No learned group priors for later subject-level posterior estimation are assumed.

### R3. Separate reference dimension from atlas granularity

Two quantities must remain distinct:

1. `N_ref`
   - canonical reference-feature dimension
   - `7` for `kong2019`
   - `8` for `kong2019_8net`, where `TemporalParietal` is independent rather than merged into `Default`
2. `K_hipp`
   - hippocampal cluster count
   - unknown a priori
   - chosen data-driven

The cortical canonical-network count is **not** the hippocampal cluster count.

### R4. Final atlas definition

For each strategy and hemisphere, the final atlas is a **data-driven cluster atlas** with chosen `K_hipp`, not a fixed `7`- or `8`-label atlas.

The final product must provide:

- hippocampal hard cluster labels `z(v) in {1..K_hipp}` on supported vertices
- cluster ID numbering ordered along hippocampal long axis (`head -> tail`), not by cluster size
- cluster sizes and support statistics
- cluster-to-canonical-network association profiles
- dominant canonical-network annotation per cluster
- optional derived vertex-wise canonical-network view for visualization

The cluster atlas is primary. Network labeling is interpretive.

### R5. Canonical networks are allowed to be absent

- A canonical network may have no strong hippocampal representation.
- Multiple hippocampal clusters may map to the same canonical network.
- The implementation must never force one hippocampal cluster per canonical network.

### R6. Spatial domain

- Model left and right hippocampus separately.
- Work in the actual consumed HippUnfold surface space.
- Keep `den-512` evidence explicit in manifests.

### R7. Reuse existing shared caches

The implementation must reuse the existing shared caches under:

- `outputs_migration/hipp_functional_parcellation_network/_shared/sub-<subject>/fc/<atlas>/<smoothing>/hemi_<L|R>/...`
- runwise counterparts under the same shared tree
- valid masks under `surface/tsnr_gate/sub-<subject>_hemi-<L|R>_valid_mask.npy`

No new cortex sampling or hippocampal BOLD resampling should be introduced.

### R8. Existing spectral machinery is reusable

Current spectral code can be reused as:

- feature transforms
- affinity construction helpers
- spatial regularization machinery
- diagnostics

But current subject-level free-`K` branch outputs are not themselves the final group atlas.

### R9. Explicit K-selection mode

The group-atlas implementation must explicitly record:

- `k_selection_mode = experimental`

in all manifests and summaries.

Reason:

- this plan uses the repository's `local-minimum + 1-SE + non-triviality` route
- failing to record this would conflict with the repository's current K-selection mode split

### R10. HippUnfold vertex correspondence assumption

This plan assumes that all subjects are represented on the same HippUnfold `den-512` unfolded surface with anatomically corresponding vertex indices across subjects.

Therefore:

- group averaging
- subject-split stability
- Hungarian matching across split-wise group solutions

all rely on that cross-subject vertex correspondence.

The manifest must record explicit `den-512` evidence and measured vertex counts, following the repository's density-audit guardrail.

### R11. Fixed non-triviality threshold

The minimum parcel-size rule is fixed before formal runs as:

- `V_min_fraction = 0.03`
- `V_min = ceil(0.03 * N_support)` for each run combination

This threshold must be recorded in every manifest and must not be changed mid-analysis.

### R12. Canonical network ordering consistency

For each atlas:

- `cortex_canonical_networks.tsv` row order must be identical across all subjects in the analysis cohort
- one subject may be used as the template ordering only after all other subjects have been verified to match exactly
- if any subject differs, the group-atlas run must fail fast rather than silently borrowing a single-subject label order

---

## What Yeo2011 Contributes

The vendored Yeo reference code is present at:

- `external/CBIG/stable_projects/brain_parcellation/Yeo2011_fcMRI_clustering`

The relevant Yeo logic is:

1. compute subject-level FC profiles on a common surface
2. build a group representation
3. cluster at the group level
4. align labels as needed for interpretability and comparison

What should not be copied literally:

- the cortex-wide ROI-profile construction
- the assumption that the published `7` and `17` are the only valid `K`

For this hippocampal problem, `K_hipp` may be `2`, `3`, `5`, or another value. It must be chosen by stability logic, not inherited from cortical network counts.

---

## Two Parallel Strategies

## Strategy A: `group-network-spectral`

Scientific role:

- cluster the hippocampus using **canonical-network reference fingerprints**

Data object definition:

- subject-level high-tSNR hippocampal `vertex x canonical-merged-network` Fisher-z FC matrix, shape `V x N_ref`

Preferred input path:

- `_shared/.../fc/<atlas>/<smoothing>/hemi_<L|R>/hipp_vertex_to_network_fc.npy`

If the implementation discovers a different cache filename with the same data semantics, it may consume that path instead, but the actual consumed path must be recorded in the manifest.

Baseline feature definition:

1. load the subject `V x N_ref` FC matrix
2. apply Fisher-z transform with clipped input bounds if the cache is still in correlation space
3. z-score each network column across vertices
4. use the resulting matrix as `F_ref_s`

Affinity and clustering:

1. compute cosine similarity between vertex feature rows
2. map similarity to `[0, 1]`
3. apply Hadamard product with the HippUnfold mesh adjacency matrix
4. run spectral clustering on the precomputed affinity
5. cluster the spectral embedding with `KMeans(n_init = 50)`

Baseline group-representation rule:

- Yeo-style group-average feature clustering: compute subject features first, group-average those features, then cluster the group representation

Sensitivity option:

- embedding-alignment sensitivity: subject-level spectral embedding -> cross-subject alignment (for example Procrustes) -> group-average embedding -> clustering

The `Yeo-style` label here refers to the broad Yeo2011 logic of building subject FC profiles, forming a group-level representation, and clustering at group level. Do not introduce numbered Yeo method names for these implementation variants, because that numbering is not official Yeo2011 terminology.

## Strategy B: `group-network-partial-spectral`

Scientific role:

- cluster the hippocampus using canonical-network fingerprints after removing shared variance with the other canonical networks only

Data object definition:

- subject-level high-tSNR hippocampal `vertex x canonical-merged-network` one-sided partial FC matrix, shape `V x N_ref`

Preferred input path:

- `_shared/.../fc/<atlas>/<smoothing>/hemi_<L|R>/hipp_vertex_to_network_partial_fc.npy`

Partial FC definition:

1. for target network `j`, take the hippocampal vertex timeseries matrix `X` and the canonical-network timeseries matrix `Y`
2. form controls `Y_-j` from all canonical networks except `j`
3. regress each hippocampal vertex timeseries in `X` on `Y_-j`
4. correlate the resulting hippocampal residual timeseries with raw `Y_j`
5. repeat for every target network

The target network is not residualized. Other hippocampal vertices are not covariates. This is intentionally not a two-sided partial correlation.

Feature transform, affinity, clustering, support masks, nulls, run-aware diagnostics, and `K` selection are identical to `group-network-spectral`.

## Strategy C: `group-intrinsic-spectral`

Scientific role:

- cluster the hippocampus using **intrinsic hippocampal coupling structure**

Data object definition:

- subject-level high-tSNR hippocampal `vertex x vertex` Fisher-z intrinsic FC matrix, shape `V x V`, with input tanh-bound clipping and diagonal forced to `0`

Preferred input path:

- `_shared/.../fc/intrinsic/<smoothing>/hemi_<L|R>/hipp_vertex_to_vertex_fc_active.npy`

The implementation requires this atlas-agnostic intrinsic path. Missing intrinsic cache is a hard error (no atlas-specific fallback). Intrinsic FC remains expected to be atlas-invariant; any non-identical intrinsic objects for the same `subject x smoothing x hemi` are treated as cache bugs and must fail fast.

Baseline feature definition:

1. load the subject `V x V` intrinsic FC object
2. apply Fisher-z transform with clipped input bounds if needed
3. set diagonal to `0`
4. treat each row as a `V`-dimensional intrinsic profile

Affinity and clustering:

- use the same spectral-clustering and `KMeans(n_init = 50)` procedure as `group-network-spectral`

Atlas role:

- clustering is atlas-independent
- atlas affects **annotation only**

For this method, the manifest records a single `atlas` field. Its role is annotation-only for `group-intrinsic-spectral`.

To save compute, the implementation may cache intrinsic clustering outputs and reuse them across the two atlas values, but coverage threshold, smoothing, hemisphere, and `K` remain explicit run axes.

---

## Proposed Strategy

## Stage 0. Define one atlas run

A single run is defined by:

- `method` in `{group-network-spectral, group-network-partial-spectral, group-intrinsic-spectral}`
- `atlas` in `{kong2019, kong2019_8net}`
- `smoothing` in `{2mm, 4mm}`
- `hemi` in `{L, R}`
- subject set = default 153-subject manifest unless explicitly overridden

Each run produces one data-driven hippocampal group atlas for one hemisphere.

## Stage 1. Build the group support mask

Inputs:

- subject valid masks from `_shared/.../tsnr_gate/`

Outputs:

- `group_support_mask[v]`
- `subject_coverage_count[v]`
- `subject_coverage_fraction[v]`

Coverage policy:

- do not use strict all-subject intersection
- do not use raw union
- use a configurable valid-subject fraction threshold

Required sensitivity set:

- `coverage_fraction >= 0.70`
- `coverage_fraction >= 0.80`
- `coverage_fraction >= 0.90`

Recommended default export:

- `coverage_fraction >= 0.80`

Every manifest must record:

- chosen threshold
- supported vertex count
- excluded vertex count

## Stage 2. Candidate K range

Use the repository's established range:

- `K ∈ {2..10}`

Then apply non-triviality clipping with `V_min`:

- `K_max_effective <= floor(N_support / V_min)`

Every manifest must record:

- requested `K` range = `2..10`
- effective `K` range after `V_min` clipping
- the `V_min` value used

## Stage 3. Build group atlas candidates for each K

For each `method x atlas x smoothing x hemi x K`:

1. construct the method-defined group representation
2. cluster the group representation into `K` parcels
3. export parcel labels
4. compute parcel mean canonical-network profiles
5. compute cluster-to-network association matrix

The baseline group clustering algorithm is uniquely fixed as:

- spectral clustering on a precomputed affinity matrix, followed by `KMeans(n_init = 50)` in the spectral embedding space

Ward clustering is not part of the baseline implementation.

## Stage 4. Choose K_hipp using Yeo-like stability logic

### Principle

Choose `K_hipp` by reproducibility of the group atlas, not by resemblance to the number of canonical networks.

### Primary resampling design

Use:

- **subject-split stability**

Parameters:

- `B = 100` subject-split resamples
- the exact seed list for those resamples must be stored in the manifest

For each resample `b` and candidate `K`:

1. split the 153 subjects into two independent subject subsets `A_b` and `B_b`
2. within that run, use the current main coverage threshold `τ` for both split masks
3. build a split-specific support mask `A_mask(τ)` from subset `A_b` and `B_mask(τ)` from subset `B_b`
4. build a group representation separately from each subset on its own split-specific mask
5. cluster each subset separately using the same method and `K`
6. compute the overlap support mask `A_mask(τ) ∩ B_mask(τ)`
7. Hungarian-match the two clusterings on that overlap mask
8. compute `ARI` on that overlap mask
9. define instability `I(K,b) = 1 - ARI(K,b)`

### Secondary resampling diagnostic

Also report:

- run-aware instability using independent runwise group representations

But subject-split reproducibility is the primary selection basis.

### Null references

A two-tier null reference is required.

Weak null:

- random-label baseline with matched `K`

Strong null:

- `group-network-spectral` and `group-network-partial-spectral`: randomly permute vertices (rows) of the `V x N_ref` feature matrix
- `group-intrinsic-spectral`: apply the same random permutation `π` to both rows and columns of the `V x V` intrinsic FC matrix, i.e. `FC_null[i, j] = FC[π(i), π(j)]`

These strong nulls preserve the legal object type while breaking the mapping between functional identity and real anatomical vertex identity.

The plan does **not** use cortex-style spin nulls as baseline, because hippocampal surface geometry requires separate justification.

### Null usage

Phase-1 null usage is fixed as diagnostic-only.

- `B_null = 100` permutation/null draws
- for each candidate `K`, report `I_null_mean(K)` and an upper reference envelope such as `I_null_95pct(K)`
- show those null summaries on the instability curve and in the tabular outputs
- Phase-1 does **not** use null as a hard K-selection gate

So null is a reference line for interpretation, not a screening rule in the Phase-1 implementation.

### Aggregation

For each `K`:

- `I_mean(K) = mean_b I(K,b)`
- `I_se(K) = sd_b(I) / sqrt(B)`

### Selection rule

Use the repository's documented protocol with explicit mode recording:

- `k_selection_mode = experimental`

Selection steps:

1. retain local minima of `I_mean(K)`
2. apply `1-SE`
3. apply non-triviality constraints:
   - minimum parcel size `V_min`
   - spatial connectedness diagnostics
   - within-cluster homogeneity / profile coherence
4. if several `K` survive, choose the smallest surviving `K`

All stable local minima must still be reported.

## Stage 5. Annotation stage with explicit atlas dependence

For all methods, `cluster_network_profile` must be computed by the following single allowed subject-level procedure:

1. fix the final group labels `group_labels_final`
2. for each subject, use `group_labels_final` on that subject's hippocampal timeseries to compute subject-level cluster-mean hippocampal timeseries
3. for that same subject, compute the cluster-to-network association using the method's FC mode:
   - `group-network-spectral` and `group-intrinsic-spectral`: Pearson correlation between cluster-mean hippocampal timeseries and canonical-network timeseries
   - `group-network-partial-spectral`: one-sided networks-only partial FC, with the cluster-mean hippocampal timeseries residualized by other canonical-network timeseries and correlated with the raw target-network timeseries
4. apply Fisher-z transform to the resulting subject-level cluster-to-network correlation matrix
5. average those subject-level Fisher-z profile matrices across subjects within the current run combination
6. export the resulting signed group mean matrix as:
   - `cluster_network_profile[K*, N_ref]`
7. if a cluster has `cluster_network_profile <= 0` for every canonical network, label it `unassigned`
8. otherwise, define the dominant network as the maximum entry of `cluster_network_profile`
9. if the top-two signed Fisher-z values differ by less than `0.05`, keep the hard top-1 dominant-network label but flag `low_margin_lt_0p05` as a confidence diagnostic

This is the only valid Yeo-style adaptation for the annotation stage in this project. Any implementation that averages hippocampal timeseries or canonical-network timeseries across subjects before computing the correlation is disallowed. The raw signed cluster-to-network Fisher-z profile is the primary annotation object. A cluster with no positive coupling to any canonical network is treated as not belonging to any canonical network.

For `group-intrinsic-spectral`, the `atlas` field determines the post hoc annotation network set. The clustering input is expected to be atlas-invariant.

`cluster_assignment_status.json` must contain one record per cluster, with at least:

- `cluster_id`
- `status` in `{dominant, unassigned}`
- `top1_network`
- `top1_fisher_z`
- `top2_network`
- `top2_fisher_z`
- `margin`

## Stage 6. Export products

Per `method x atlas x smoothing x hemi`, write:

- `group_support_mask.npy`
- `subject_coverage_fraction.npy`
- `group_feature_summary.json`
- `k_selection_summary.tsv`
- `k_selection_summary.json`
- `group_labels_k<k>.npy` for retained candidate solutions if enabled
- `group_labels_final.npy`
- `cluster_network_profile.npy`
- `cluster_dominant_network.json`
- `cluster_assignment_status.json`
- `group_atlas_manifest.json`

Every manifest must include:

- `method`
- `atlas`
- `smoothing`
- `hemi`
- `k_selection_mode = experimental`
- coverage threshold
- `V_min_fraction`
- `V_min_effective`
- feature-family description
- actual consumed input paths
- clustering algorithm description
- affinity description
- candidate `K` range
- chosen `K*`
- all subject-split seeds
- measured density / vertex-count evidence

Render products:

- WTA parcel surface for `K*`
- parcel-by-network profile heatmap
- K-selection instability curve
- occupancy table
- coverage summary

Optional derived view:

- vertex-wise dominant-network map inherited from assigned parcels

That view must be labeled as derived.

---

## Validation

For each method, validation must include:

- subject-split stability of `K*`
- run-aware stability diagnostic
- parcel-size validity
- spatial coherence
- reproducibility of cluster-network profiles
- interpretation quality without forcing absent networks to appear

Cross-condition consistency outputs that must be exported:

- left-right hemisphere consistency
- cross-smoothing consistency (`2mm` vs `4mm`)
- cross-atlas consistency where comparable reference structure exists

These are diagnostic outputs only. They are not used for automatic method selection or arbitration.

Failure signs include:

- salt-and-pepper parcel maps
- strong seed dependence
- widespread tiny parcels near the `V_min` floor
- unstable `K` without clear local minima
- all clusters collapsing onto one network profile

---

## Runtime Constraint

Any scheduler, multiprocessing runner, or parallel executor used by this workflow must respect the repository hard cap:

- maximum concurrency = `10`

No batch submission or executor configuration may exceed that limit without explicit user permission.

### Expected Runtime Scale

A naive upper bound for this workflow is on the order of:

- `B = 100` resamples
- `K = 2..10`
- `3` methods
- `2` Kong atlas values
- `2` smoothing values
- `2` hemispheres
- `3` coverage thresholds

which implies a large number of spectral-clustering fits. Actual runtime should be lower than that naive upper bound when existing manifests are resumed and intrinsic caches are reused, but the expected walltime is still at the scale of many hours rather than minutes.

---

## Minimal Code Plan

Phase 1 should add one new downstream driver script:

- `scripts/hipp_parcellation_network/run_merged_canonical_group_atlas.py`

The Kong 8net / partial-FC cache preparation helper is:

- `scripts/hipp_parcellation_network/materialize_kong7_kong8_pearson_partial_fc.py`

This script should:

1. load the 153-subject manifest by default
2. iterate over the fixed methods:
   - `group-network-spectral`
   - `group-network-partial-spectral`
   - `group-intrinsic-spectral`
3. load the required shared caches for `kong2019` and `kong2019_8net`
4. build support masks for threshold set `{0.70, 0.80, 0.90}`
5. run group clustering for each candidate `K in {2..10}`
6. evaluate subject-split instability with `B = 100`
7. apply `experimental` K selection
8. annotate final parcels with canonical-network profiles
9. export manifests and renders

Optional helper split if needed:

- `scripts/common/group_network_spectral.py`
- `scripts/common/group_intrinsic_spectral.py`
- `scripts/common/group_k_selection.py`

The new workflow must remain separate from:

- `run_group_prior_fastpfm.py`
- `run_kong2019_mshbm.py`

because those solve different downstream problems.

---

## Doc Sync Deliverables

If implementation begins, the same change must update:

- `docs/hipp_parcellation_network/multi_branch_flow.md`
- `docs/hipp_parcellation_network/step_by_step_flows.md`
- `docs/hipp_parcellation_network/network_first_runbook.md`

If the runtime stack changes, also update:

- `docs/environment/hippo2_reproducibility_manifest.md`

This is part of the deliverable, not optional cleanup.

---

## Decision Summary

The implementation direction is:

- run three parallel Kong group methods
  - `group-network-spectral`
  - `group-network-partial-spectral`
  - `group-intrinsic-spectral`
- run two Kong atlas definitions
  - `kong2019` = 7net, `TemporalParietal` merged into `Default`
  - `kong2019_8net` = 8net, `TemporalParietal` independent
- keep canonical networks as reference / interpretation space, not forced atlas labels
- choose `K_hipp` data-driven for each method using the same explicit `experimental` stability protocol over `K ∈ {2..10}`
- export both sets of results without automatic method selection
- stop at group atlas generation

That is the implementation plan this repository should follow.

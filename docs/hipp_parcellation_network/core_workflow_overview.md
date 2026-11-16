# Core Workflow Overview

This document is the authoritative workflow surface for active hippocampal processing.

## Scope

- Active modality regime: T1w-only processing surface
- Public execution layer: `scripts/core_workflow/`
- Numbering convention: `NN_MM_<semantic_name>.py`
- Global order is encoded by `NN`, independent of branch names

Branch mainlines:

- `kong2022_pfm`
- `kong2022_pfm_17net`
- `kong2022_pfm_ctxreg`
- `kong2022_pfm_ctxdrop`
- `kong2022_fixed`
- `kong2022_fixed_17net`
- `lynch2024_20net`
- `lynch2024_7net`

Within `kong2022`, PFM (`pfm`, canonical7), PFM 17-net (`pfm_17net`), PFM cortex-regression (`pfm_ctxreg`), PFM cortex-drop (`pfm_ctxdrop`), Fixed standard template (`fixed`, canonical7), and Fixed 17-net (`fixed_17net`) are treated as parallel mainlines. The Lynch2024 branches are all-base-subject FASTANS/PFM comparison branches against Kong2022 PFM, with separate shared inputs and group roots. The NSI table is retained as audit metadata and is not a cohort-selection gate.

## Pipeline Stages

- `01_xx`: HippUnfold run and output validation
- `02_xx`: shared inputs, cohort manifests, and QC
- `03_xx`: cortex reference generation
- `04_xx`: hippocampal FC materialization
- `05_xx`: gradients, group atlas construction, and comparison products
- `06_xx`: label materialization, overview rendering, presentation outputs
- `07_xx`: group matrix indexing

Group atlas method families include intrinsic FC, network/ROI FC, WTA, and
gradient clustering. `05_02_run_group_atlas.py` evaluates K=2..8, records its automatic selection separately, and uses non-forced `--default-k 3` for primary labels and overviews. Partial
correlation spectral branches are retired from public method choices.

`gradient-spectral` consumes aligned HippoMaps/BrainSpace gradient artifacts
written by `05_01_compute_hippomaps_gradients.py` under `--gradient-root`, then
clusters selected columns. For Kong2022 ROI-registry routes and Lynch2024
component-network routes, `05_01` computes G1-G5 from group mean Fisher-z
vertex-to-ROI/component FC. FC column scaling (`none` or `col_zscore`) and
sparsification (`features`, `affinity`, or `none`) are independent explicit
choices. The production default remains column z-scoring followed by feature
sparsification. Pearson affinity is constructed explicitly, clipped
nonnegative and mean-symmetrized, retaining its kernel diagonal; embedding
uses an exact diffusion map (alpha 0.5, multi-scale eigenvalues): the
symmetric conjugate of the diffusion operator is decomposed completely with a
direct solver, so gradients have no start vector and no random seed. Gradient
manifests record `gradient_preprocessing.eigensolver = symmetric_eigh`; banks
without it must be regenerated. The producer
aligns component order and sign within each preprocessing setting/route to the
anchor hemisphere (`hemi_L`), which defaults to `4mm` for historical production parity
and supports explicit `--alignment-anchor-smoothing {2mm,4mm}`. `05_02` accepts the same
`--alignment-anchor-smoothing` flag to enforce anchor consistency during resume and
downstream artifact validation, and writes only the aligned arrays. `05_02`
selects aligned G1-G3 by default (`--gradient-components 1,2,3`) and applies
the separate declared clustering scaling (`none` or component-wise `unit_sd`).
The full effective preprocessing contract is required for gradient reuse,
including `--resume` and `--reuse-fixed-cluster-root`. Fixed-cluster reuse
requires matching explicit preprocessing declarations in the feature summary
and upstream gradient manifest, as well as matching clustering feature scaling,
active gradient features, and the exact declared RBF bandwidth when applicable.
Resuming a gradient result requires its declared feature summary
and upstream gradient manifest; missing or conflicting source declarations
stop with an explicit regeneration error. Runs skipped because no K was valid
or survived selection retain their skip status when their input signature is
unchanged; a run with no valid K does not require a gradient cache.
Subject-specific spatial-permutation nulls (`strong_null_ari`)
permute each participant's vertex FC and validity mask with the same permutation.

For explicit comparisons, `05_02` also accepts `gradient-ward`,
`gradient-gmm`, and `gradient-spectral-rbf`. Ward uses Euclidean features
and a connected binary mesh; GMM uses full-covariance mixtures without a
spatial constraint and reports disconnected clusters. RBF spectral clustering
weights unique mesh edges by gradient distance; `--rbf-scale` multiplies the
median positive edge distance. Its diagnostics include effective connectivity
at a relative edge-weight cutoff of `1e-5`. The established
`gradient-spectral` method retains shifted-cosine affinity. These methods use
the same explicitly selected gradient features, and the comparison does not
change the production default. `--k-selection-mode fixed-comparison` evaluates
the requested fixed K without selecting an automatic K; `--comparison-output`
requires this mode, an explicit dataset, and an output location under `tmp/`.

For Kong2022 subjectwise 400-ROI inputs, the explicitly selected assignment
variants `ovr_logistic_canonical7` and `ovr_logistic_family17` fit separate
one-versus-rest models to cluster-mean hippocampal time-series FC. They annotate
existing spatial clusters using the largest strictly positive raw coefficient;
a cluster without one is unassigned. The exports include input observations,
coefficients, intercepts, and raw and per-SD odds ratios. Per-SD values change
the reporting units of the same fitted model. These associations are
descriptive and are not directly comparable with Kember's different input and
standardization procedure. `--assignment-variant both` continues to select
Fisher-z and normalized Top-20 only; request each OVR variant explicitly.

`06_01_build_group_overview.py` requires an explicit `--smoothing` for each
figure. `--annotation-pair roi-fisherz` renders Fisher-z and the matching
canonical7 or family17 OVR rows using the same spatial clusters and declared K
diagnostics. The two-smoothing overview layout is retired. The group matrix
indexes the separate method/smoothing outputs and does not select old figures.
Rendering enforces the upstream-gradient and assignment contracts. Programmatic comparison callers
can supply `profile_vmax_by_variant` to `render_combo` for shared, separately
specified Fisher-z and raw-beta color limits. Rendering scratch files remain
under the repository temporary workspace. Comparison-only experiments do not
introduce numbered workflow steps.

`06_04_render_hippomaps_gradient_products.py` renders the same continuous
gradient artifacts with `hippomaps.plotting.surfplot_canonical_foldunfold`,
`cmap="plasma"`, `tighten_cwindow=True`, `share="row"`, and
`color_bar="right"`. For `den=512`, it treats `512` as HippoMaps v2 in the
current Python process and disables `unfoldAPrescale`; no AP surfarea rescale is
used for 512-density figures. The renderer defaults to bilateral output and accepts `--hemis L` or
`--hemis R` for single-hemisphere files. Although left and right hemispheres are
fit independently upstream, `05_01` writes aligned arrays before `06_04` renders
them. `06_04` then applies a display-only cross-route alignment by default:
`pfm` is the reference, each selected route/hemi/component is sign-flipped for
rendering when its Pearson correlation to the matching `pfm` component is
negative, and the saved `05_01` arrays remain unchanged. Continuous-gradient
figures also use shared per-smoothing, per-component color ranges across the
rendered routes so PFM, cortex-regression, cortex-drop, and Fixed panels are
visually comparable for paper figures. Bilateral figures are still assembled
from separate L/R HippoMaps renders so each side keeps its own EV labels.
Manifests record the `external/hippomaps` import path, commit, plotter support
settings, upstream gradient alignment provenance, display flip decisions, and
shared color ranges.

Canonical presentation outputs live under `outputs_hcp/present/`. All maintained
method overview PNGs are written to `outputs_hcp/present/overviews/<route>/`.
Supplementary continuous-gradient products live in `outputs_hcp/present/gradients/`,
and cortex panels live in `outputs_hcp/present/cortex_panels/`. Run-owned smoke
outputs, independent validation, and non-official execution records use
`outputs_hcp/runs/<run_id>/`. See [Output Layout](output_layout.md) for the
complete directory contract and the canonical Stage 01 asset boundary.
For the cross-dataset 7-network renderer, the producer must write the
long-axis-relabelled `(3, 400)` profile and register it in the group-atlas
manifest with contract tag `roi_ap`. Consumers fail closed when that
artifact or its ordering metadata is missing; raw cluster labels and
network-level profiles are not valid substitutes for ROI rows.

The same renderer treats normalized WTA projections as a separate
`wta_roi` four-row contract. Canonical-7 rows are named and ordered
`Default`, `VentralAttention`, `Somatomotor`, `Language`; canonical-8 rows are
`Default`, `VentralAttention`, `Auditory`, `Somatomotor`. WTA rows are created
from the explicit label-map network names, not from raw numeric label order.
Each bilateral `(4, 400)` cache must carry a sidecar recording the exact atlas,
network order, the profile's own SHA-256, L/R source manifests, labels,
support masks, label maps, and SHA-256 fingerprints. Consumers fail closed on a missing/stale sidecar,
subject-order mismatch, or shape/order mismatch; an empty network is retained
as its named row rather than allowing later networks to shift.


The complete gradient-spectral and normalized-WTA cluster-to-cortex template renderer is `06_10_render_cluster_roi_strength_panels.py`. Gradient mode consumes saved `cluster_roi_profile.npy` Fisher-z matrices with shape `K x 400`; WTA mode consumes the existing vertex-level `normalized_top20_count` labels and treats the four most frequent networks as named pseudo-clusters. Both modes paint all Kong2022 parcels using a global symmetric `RdBu_r` scale across 2mm/4mm and L/R, write rows 2mm-L, 2mm-R, 4mm-L, and 4mm-R, and show bilateral L/R cortex in every panel. Gradient K=3 columns are validated as `Anterior`, `Middle`, and `Posterior`; WTA mode reads canonical7 WTA labels only; family17 WTA products are rendered by `06_01` overviews, not by this template view. This full-template strength view is separate from the significance-filtered cluster panels generated by `06_03_build_cluster_cortex_panels.py`.
Directory names intentionally do not encode `t1only`, `present_fixedk`, or a
`10perc/full` presentation split.

Retained Kong cortex references use `inputs/kong2022/shared_inputs/pfm/` or
`fixed/`. These
references are derived from cortical CIFTI time series and cortical labels,
independently of hippocampal sampling coordinates. Correcting hippocampal
coordinates affects hippocampal sampling and products derived from it, rather
than invalidating the cortex references. A reference path migration preserves
the existing arrays, network summaries, and QC provenance without regeneration
or scientific relabelling.

Cortex label sources use
`inputs/kong2022/cortex_references/pfm_areal_mshbm/`. Group-average surfaces
use `inputs/kong2022/templates/hipp_group_average/group_average_surfaces/`
within the corresponding dataset root.

`scripts/core_workflow/` is both:
- canonical public interface (`NN_MM_*`)
- active implementation home (`scripts/core_workflow/lib/*`)

Individual hippocampal parcellation uses group-trained template matching in
`scripts/individual_workflow/`. HCP uses family-aware cross-validation:
clustering fits the training cohort's group data, and held-out subjects are
matched to that template. Never cluster an individual's hippocampal data or
pass a single-subject cohort through a group-clustering entrypoint.
Computation outputs live under
`outputs_hcp/work/crossfit_template_matching/`, and presentation products under
`outputs_hcp/present/overviews/crossfit_template_matching/`. Non-HCP matching
adapters live under `scripts/<dataset>/`. See
`docs/hipp_parcellation_network/crossfit_template_matching.md` and
`scripts/individual_workflow/README.md`.

RouteA reference construction has an explicit upstream label-generation dependency. The artifact-level contract for generated RouteA ArealMSHBM labels, local numbered-step products, presentation products, and required evidence is maintained in `docs/hipp_parcellation_network/core_workflow_artifact_lineage.md`.

The `kong2022_pfm_17net` and `kong2022_fixed_17net` branches use the original Kong2022 17 network families for cluster annotation and network profiles. They reuse `shared_inputs/pfm` (or `shared_inputs/fixed`), `surface`, 400-ROI FC caches, and `pfm` (or `fixed`) gradients, then write taxonomy-dependent group outputs under `outputs_hcp/work/group_atlas/pfm_17net/` (or `outputs_hcp/work/group_atlas/fixed_17net/`). To match an existing canonical7 result exactly, run `05_02_run_group_atlas.py` with `--route-tag pfm_17net` (or `fixed_17net`), `--network-taxonomy family17`, the same profile and gradient method, and `--reuse-fixed-cluster-root` pointing to the corresponding canonical7 profile root. The source and target share gradients, spatial labels, and K diagnostics; network profiles and assignments are calculated for family17. This includes TopSim/Ward and TopFC/shifted-cosine spectral clustering. Do not run `05_01` for a separate `*_17net` gradient tree. Family17 `wta-top` is a clusterless direct run: `--methods wta-top --route-tag pfm_17net` (or `fixed_17net`) `--network-taxonomy family17` without `--reuse-fixed-cluster-root` or `--gradient-root`; it applies the same `normalized_top20_count` assignment to the `family17` column of the Kong2022 ROI registry and writes `<route>/wta-top/kong2022/{2mm,4mm}/cov_0p80/hemi_{L,R}/`.

The `lynch2024_20net` and `lynch2024_7net` branches use every subject in the authoritative HCP QC-pass manifest from `02_03_build_lynch2024_nsi_manifest.py`. The NSI table is written to `hcp_7t_lynch2024_nsi_audit_latest.tsv` for provenance only; it does not filter subjects or require an expected count. External priors live under `external/FASTANS/resources/PFM/priors/Lynch2024/`; `03_04_run_lynch2024_fastans_pfm.py` generates official per-subject hard-label PFM dlabel/dscalar assets and hard-label component time series under `outputs_hcp/work/cortex_pfm/sub-<id>/lynch2024/`; current shared inputs live under `outputs_hcp/inputs/lynch2024/shared_inputs/pfm/`. `03_03_build_lynch2024_reference.py` writes an exact ordered Lynch20 reference containing 20 unique networks after excluding `Noise`, plus canonical7 merged references from `config/cross_atlas_network_merge.json`. The maintained downstream methods are `gradient-spectral` and direct-network `wta-top`. It uses the non-Noise Lynch20 component bank; BrainSpace receives the low-dimensional Lynch feature matrix as a full graph (`brainspace_sparsity=0.0`), and `lynch2024_7net` reuses that gradient bank while applying the merged canonical7 taxonomy downstream. In `05_02`, Lynch gradient runs use shifted-cosine surface affinity for spectral clustering and the same full-sample cluster-to-network FC redundancy diagnostic as Kong2022. They use `--profile-surface-root outputs_hcp/inputs/kong2022/shared_inputs/pfm_ctxreg` so cluster profiles are computed with the same cluster-mean hippocampal time-series correlation used by Kong PFM; this does not change the Lynch PFM FC, masks, references, or gradient inputs. The direct Lynch WTA path consumes the vertex-to-network FC profile and assigns each supported vertex to its strongest finite Fisher-z network with a Fisher-z margin. When methods are omitted, both `gradient-spectral` and direct-network `wta-top` are generated. Historical `pfm_nsi043` roots are retained only for old-result provenance and are not current branch defaults.

The formal `gradient-spectral` clustering affinity is `shifted-cosine`, defined
as `(1 + cosine_similarity) / 2` on surface-adjacent vertices. Group manifests
record `surface_affinity`, `affinity_description`, and connected-graph
diagnostics. Overview items record `surface_affinity`,
`affinity_description`, and `source_group_manifests`; filenames remain stable,
while the visible title identifies shifted-cosine.

Formal release-audit and refresh maintenance helpers were retired from this
checkout. Do not invoke or restore deleted maintenance paths. Current validation
uses the active numbered workflow, group manifests, and the multi-dataset suite
readiness check; provenance remains recorded in the generated manifests.

Network labels in overview heatmaps, assignment bars and WTA profiles use the canonical7, Kong2022 native17 or Lynch2024 native20 abbreviations in [AGENTS.md](../../AGENTS.md#functional-network-names); source keys and native-network identities remain unchanged. Assignment-bar legends sit to the right, aligned with the adjacent profile colorbars. Diagnostic plot boxes use a height-to-width ratio of 3:4. Network display colors are centralized in `config/hipp_network_style.json`. Group-atlas label materialization, hippocampal overviews, WTA long-axis profiles, and cortex panels load network colors from that single file; `config/cross_atlas_network_merge.json` is mapping-only and must not carry color overrides. Missing colors are treated as errors, not gray/blue fallback display colors. For the RouteA 17-net taxonomy, same-family A/B/C subnetworks use deliberately separated lightness and saturation levels so overview bars, profiles, and label-table colors remain distinguishable while preserving the parent family hue.

The `kong2022_pfm_ctxreg` branch is an isolated PFM derivative: `04_00_materialize_cortex_regressed_surface_timeseries.py` writes run-wise local-cortex-residualized time series before `04_01` and `04_02` materialize FC. It does not overwrite uncontrolled PFM roots. The HCP 400-ROI control is maintained under `outputs_hcp/inputs/kong2022/shared_inputs/pfm_ctxreg`, with gradients under `outputs_hcp/work/hippomaps_gradients/pfm_ctxreg/`, group atlases under `outputs_hcp/work/group_atlas/pfm_ctxreg/`, and the three-image overview under `outputs_hcp/present/overviews/pfm_ctxreg/`. Legacy suffixed trees are migration sources only until the approved cleanup and are not runtime inputs.

The `kong2022_pfm_ctxdrop` branch is the matched PFM local-cortex exclusion control. `04_01_materialize_kong2022_roi_fc.py --ctxdrop` keeps hippocampal time series unchanged, excludes valid cortex vertices within 20 mm separately for each hippocampal vertex before ROI averaging, and writes isolated FC caches plus smoothing-independent local cortex-neighbor diagnostics. Its maintained downstream scope is `gradient-spectral` and `wta-top`. The HCP 400-ROI control is maintained under `outputs_hcp/inputs/kong2022/shared_inputs/pfm_ctxdrop`, with gradients under `outputs_hcp/work/hippomaps_gradients/pfm_ctxdrop/`, group atlases under `outputs_hcp/work/group_atlas/pfm_ctxdrop/`, and the three-image overview under `outputs_hcp/present/overviews/pfm_ctxdrop/`. Legacy suffixed trees are migration sources only until the approved cleanup. The shared local-cortex-neighbor count surface overlay for ctxreg/ctxdrop is rendered by `06_07_render_ctxdrop_neighbor_counts.py` under `outputs_hcp/present/local_cortex_neighbors/`.

Active numbered steps no longer shell out to the legacy hippocampal/cortex runtime directories.
The active runtime boundary allows external assets and the Workbench executable
`software/workbench/bin_linux64/wb_command`, but not external helper code scripts.

## Traceability Contract

Each entrypoint can emit a standardized step manifest via `--step-manifest-out` with:

- `step_id`
- `workflow_branch`
- `upstream_step_ids`
- `input_artifacts`
- `output_artifacts`
- `command`
- `git_commit`

Use these manifests to reconstruct exact lineage for each generated artifact set.

## References

- Step matrix: `docs/hipp_parcellation_network/core_workflow_step_registry.md`
- Artifact lineage: `docs/hipp_parcellation_network/core_workflow_artifact_lineage.md`
- Machine-readable registry: `scripts/core_workflow/step_registry.json`
- Branch semantics: `docs/hipp_parcellation_network/multi_branch_flow.md`
- Ordered procedure: `docs/hipp_parcellation_network/step_by_step_flows.md`
- Archive index: `docs/hipp_parcellation_network/archive_index.md`

The WTA-top overview generated by `06_01_build_group_overview.py` embeds a fixed 20-bin long-axis network-composition profile between the compact vertex-assignment count panel and the hippocampal surface render; `07_01_build_group_matrix.py` indexes the embedded overview only.

## Formal HCP Hippocampal Gradient Parcellation Workflow (TopFC and TopSim Profiles)

The formal HCP PFM canonical7 gradient parcellation workflow is defined by `GRADIENT_PARCELLATION_FORMAL_WORKFLOW_PLAN.md` and provides reproducible, rigorously aligned parcellations and 20 formal overview figures.

### 1. Active Profiles and Roles
- **TopFC (`historical_baseline`)**: Features without column scaling (`--gradient-fc-scaling none`); row sparsification to top 10% entries (`sparsity=0.90`, `--gradient-sparsify-stage features`); shifted-cosine spectral clustering (`gradient-spectral`). Serves as the historical-method baseline.
- **TopSim (`primary`)**: Features without column scaling (`--gradient-fc-scaling none`); full Pearson correlation affinity row-sparsified to top 10% (`sparsity=0.90`, `--gradient-sparsify-stage affinity`); surface-connected Ward clustering (`gradient-ward`). Current primary formal method.
- **TopSim GMM (`comparison_display`)**: Same TopSim features and affinity; Gaussian Mixture Model (`gradient-gmm`) with full covariance (`n_init=10`, `max_iter=1000`, `tol=1e-3`, `reg_covar=1e-6`). Visual comparison display only.
- Former exploratory routes A and D are retired from active formal selection.

### 2. Geometry, Resampling, and Alignment Reference
- **Gradients**: G1–G3 extracted from 5 fitted components, coverage threshold `0.80`. The exact diffusion-map solver takes no seed, so the Ward partition is a deterministic function of the data. Both 2 mm and 4 mm use the 2 mm left-hemisphere anchor (`--alignment-anchor-smoothing 2mm`) as the alignment reference.
- **K Range**: Full evaluation of $K=2, 3, 4, 5, 6, 7, 8$ across all 12 settings (3 methods × 2 smoothings × 2 hemispheres = 84 setting/K records). Displayed/default result is $K=3$.
- **Resampling**: 20 subject-split repetitions (83/83 split, seeds `20260420 + 1 + 13*i`) and 20 null repetitions (row-permuted FC and valid mask, seeds `20260420 + 5000 + 17*i`; one generator per repetition draws every subject's permutation, so repetitions share none). The K-selection table stores the null mean, 5th and 95th percentiles; the overview ARI panel draws the 95th percentile. Split/null gradient halves are precomputed once per setting and reused across $K=2..8$. Failures are preserved as explicit `NaN` values with error reason (no zero-filling).
- **Diagnostics**: FC-coherence from raw FC feature bank; cluster redundancy diagnostic threshold is `0.90` for both maximum similarity and pair counts (while coverage remains `0.80`). Redundancy plots span y-limits `[0.50, 1.02]`. Clustering diagnostics describe the displayed K; automatic K remains separate.

### 3. Annotation Branches
Three side-by-side annotation pairs are generated for every clustering result:
- **Pair A (`roi-fisherz`)**: ROI-equal Fisher-z network profile + standard 400-ROI OVR logistic regression (`ovr_logistic_canonical7` or `ovr_logistic_family17` for family17 routes).
- **Pair B (`network-ts-fisherz`)**: Direct network-time-series Fisher-z profile (weighted by cortical parcel grayordinates) + matching time-series OVR (`ovr_logistic_canonical7_network_ts` or `ovr_logistic_family17_network_ts` for family17 routes).
- **Pair C (`top20`)**: Normalized Top20 positive count assignment + standard 400-ROI OVR.

### 4. Overview Presentation and Output Layout
Output tree:
```text
outputs_hcp/work/hippomaps_gradients/pfm/{topfc,topsim}/{2mm,4mm}/hemi_{L,R}/
outputs_hcp/work/group_atlas/pfm/{topfc,topsim}/{method}/kong2022/{2mm,4mm}/cov_0p80/g1g2g3/hemi_{L,R}/
outputs_hcp/present/overviews/pfm/{topfc,topsim}/{method}/{2mm,4mm}/
outputs_hcp/present/overviews/pfm/wta-top/{2mm,4mm}/
outputs_hcp/present/overviews/pfm_17net/wta-top/{2mm,4mm}/
```
- **18 Clustering Overviews** (3 methods × 2 smoothings × 3 annotation pairs): `roi-fisherz_ovr_overview.png`, `network-ts-fisherz_ovr_overview.png`, `top20_overview.png` (normalized Top20 only; no OVR model is fitted on Top20 counts, so this figure has no OVR rows). Rendered at 300 dpi with consistent grid alignment (column widths match within 1%, heatmap heights match within 1%) and horizontal positive-beta winner/runner-up OVR association bars. The right-hand column, titled “Cluster-to-cortex FC”, shows the cortical FC of the three clusters (C1 Anterior, C2 Middle, C3 Posterior): the left/right mean of `cluster_roi_profile.npy` on the inflated cortex, each with its ten most positive and ten most negative ROIs. The cortex colour scale is piecewise linear and unclipped: |z| <= 0.15 Fisher-z uses the central 80% of the colormap for contrast, and the outer 10% on each side runs from 0.15 to the figure's own minimum or maximum across the three clusters; the three tiles share one colour bar drawn on a linear Fisher-z axis from that minimum to that maximum, without separate minimum/maximum annotations. The bar charts scale their axes to each cluster and show values to two decimal places. Two-row figures stack the three tiles with the bars beside each map; the one-row Top20 figure places them side by side with the bars below.
- **2 WTA Overviews per route** (2 mm and 4 mm; canonical7 `pfm` and family17 `pfm_17net`): `wta_overview.png`, preserving the complete whole-hippocampus count panel, long-axis 20-bin curves, and native surface maps.

Overview titles list method, atlas, route, annotation and smoothing before the cohort and sample size. The Functional atlas column identifies Fisher-z and OVR rows without repeating those labels on each diagnostic panel.

Network-profile heatmap values use 7.5-point text for up to eight network
columns, 6-point text for denser profiles, and 4-point text for the 17-network
family17 profiles so the values fit within their cells.
The overview reuses hippocampal surface panels under
`tmp/overview_hippocampal_surfaces/` for unchanged exact surface and label
inputs. Legends and overview plots are still recomposed for each run.
The three Workbench cortex surfaces are cached separately under
`tmp/overview_cortex_surfaces/` and reused across annotation layouts only when a
SHA-256 digest of the bilateral FC profiles, cortex assets, colour/rendering
code and Workbench entrypoint matches the digest recorded beside each surface,
and the Python rendering-library versions are unchanged. Bar charts and the final overview are recomposed;
changed inputs or rendering dependencies trigger fresh cortex renders. Each
overview manifest item lists its cortex surfaces under `cortex_surface_cache`
with status `reused` or `rendered`. A directory manifest that still records a
retired overview layout stops the merge; archive those figures under
`wrong/outputs_<dataset>/` and remove their records before rerunning.
- **Group Matrix Index**: Built via `07_01_build_group_matrix.py`, presenting TopSim + Ward at $K=3$ as primary selection.

For Kong2022 cluster-based ROI-registry outputs, the official assignment variants are `fisherz` and `normalized_top20_count`; `fisherz` (`top1_fisher_z`) is the cluster-level default, while normalized Top-20 count is an explicit comparison using top-20 positive ROI/component counts divided by the finite available ROI/component count in each network. Raw `count` is no longer registered or generated by `core_workflow`. For Kong2022 ROI-registry WTA outputs, `normalized_top20_count` is the only assignment, and canonical7/canonical8 are taxonomy views of that same assignment. Lynch2024 WTA is a direct vertex-to-network Fisher-z assignment; its cluster-level `fisherz` annotation remains the gradient branch’s cluster annotation and is not the WTA definition. Manifests and overviews point to the branch-specific default assignment.

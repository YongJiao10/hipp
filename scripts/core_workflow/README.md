# Core Workflow

`scripts/core_workflow/` contains the public numbered entrypoints and their
implementation modules. Run commands from the repository root. The first number
in `NN_MM_<semantic_name>.py` identifies the pipeline layer.

## Documentation

| Need | Reference |
|---|---|
| Execution order and branch commands | [Step-by-step flows](../../docs/hipp_parcellation_network/step_by_step_flows.md) |
| Entrypoints and input/output inventory | [Step registry](../../docs/hipp_parcellation_network/core_workflow_step_registry.md), [machine-readable registry](step_registry.json) |
| Scientific methods and branch definitions | [Workflow overview](../../docs/hipp_parcellation_network/core_workflow_overview.md) |
| Required upstream artifacts | [Artifact lineage](../../docs/hipp_parcellation_network/core_workflow_artifact_lineage.md) |
| Participant selection and vertex validity | [fMRI QC](../../docs/hipp_parcellation_network/fmri_qc.md) |
| Surface registration and BOLD sampling | [Coordinate spaces](../../docs/hipp_parcellation_network/coordinate_spaces_and_functional_sampling.md) |
| Output directories | [Output layout](../../docs/hipp_parcellation_network/output_layout.md) |
| Python, Workbench and runtime preflight | [Agent runtime contract](../../docs/environment/agent_runtime_contract.md) |
| HippUnfold installation, parameters and batch launch | [hippo2 runtime](../../docs/environment/hippo2_reproducibility_manifest.md) |
| HCP EMOTION/WM cohorts, analysis and figures | [Task workflow](../../hcp_3t_task_fmri_hipp_memory_emotion_plan.md) |

## Runtime and CLI

Use `/home/yojiao/.conda/envs/py314/bin/python` for local workflow scripts and
tests. Direct HippUnfold execution uses the validated `hippo2` runtime described
in its manifest. Argon SGE/MATLAB launchers use their own documented runtime and
`/Shared/jianglab` data paths.

Inspect a stage's arguments before selecting exact inputs and outputs:

```bash
/home/yojiao/.conda/envs/py314/bin/python \
  scripts/core_workflow/02_01_prepare_group_inputs.py --help
```

Workbench calls use `lib/workbench.py`, with native thread limits set to one.
`02_01` caps subject-level Workbench stages at two concurrent subjects.

The `run_stage` wrappers (`05_01`, `05_04`, `06_04`, `06_06`, `06_09`)
accept `--workflow-branch`, repeated `--upstream-step-id`, `--input-artifact`
and `--output-artifact`, `--step-manifest-out`, and `--dry-run` before `--`.
Stage-specific arguments follow `--`; use `-- --help` for their help. Wrapper
`--dry-run` emits a manifest without executing the scientific stage:

```bash
/home/yojiao/.conda/envs/py314/bin/python \
  scripts/core_workflow/05_04_compute_confidence_maps.py \
  --workflow-branch kong2022_pfm \
  --upstream-step-id 05_02_run_group_atlas \
  --step-manifest-out tmp/confidence_stage.json --dry-run
```

Other entrypoints expose their own options and stage manifests.

## Pipeline

| Layer | Main entrypoints | Product |
|---|---|---|
| `01` | `01_01` run HippUnfold; `01_02` check outputs | Individual anatomy and den-512 surfaces |
| `02` | `02_01` prepare inputs; `02_02` participant QC; `02_03` Lynch NSI audit | Sampled signals, masks, templates and analysis rosters |
| `03` | `03_01` Fixed references; `03_02` PFM references; Lynch `03_04` then `03_03` | Cortical reference time series |
| `04` | `04_00` cortex regression; `04_01` ROI FC; `04_02` intrinsic FC | Branch-specific signals and FC |
| `05` | `05_01` gradients; `05_02` group atlas; `05_04` confidence | Gradients, labels and assignment products |
| `06` | Numbered renderers, `06_02` label materialization and the `06_07` local cortex-control diagnostic | Figures, diagnostics and GIFTI/CIFTI labels |
| `07` | `07_01` presentation matrix | Overview index |

Task analysis uses `04_03` for first-level activation, optional exploratory
`04_04` background FC, `05_06` group statistics, and `06_11` rendering.
`04_04` is not connected to the gradient pipeline. `06_05` is retired; use
`06_04` for continuous-gradient rendering. Pass `--profile topsim` or
`--profile topfc` with `--routes pfm` to render a formal gradient bank into
`present/gradients/pfm/<profile>/`; without `--profile` it renders the
unprofiled bank. `05_04` and `06_06` take the profile group root as
`--out-root`; `06_06 --profile` writes confidence figures to
`present/gradients/pfm/<profile>/<method>/<smoothing>/`.

### Stale presentation products

A figure is stale when the group-atlas or gradient manifest it was rendered
from is newer than the figure, or when that group atlas is older than the
gradient or parent atlas it was built from. `05_01` and `05_02` print a
reminder when a run leaves figures stale. List them with:

```bash
/home/yojiao/.conda/envs/py314/bin/python \
  scripts/core_workflow/07_01_build_group_matrix.py --check-sources
```

The command checks `present/overviews` and `present/gradients` by file
modification time and exits nonzero when any figure is stale. Re-rendering a
figure on a group atlas that predates the current gradient leaves it stale
until `05_02` is rerun. `06_02` exits nonzero when any manifest fails, and
accepts every assignment variant that `05_02` writes.

### HippUnfold

`01_01_run_hippunfold_t1w_local.py` accepts either a subject or a manifest,
with exactly one `--dataset hcp|stanford|iowa_jj_tms|banda|pdc` preset:

```bash
/home/yojiao/.conda/envs/py314/bin/python \
  scripts/core_workflow/01_01_run_hippunfold_t1w_local.py \
  1001 --dataset stanford --dry-run
```

The closed preset selects the official HippUnfold release declared in
`environment/hippo2_server.yml`, T1w, the `multihist7sgi` unfolded atlas,
density 512, corobl output, unfolded registration and GPU inference. HCP,
BANDA and PDC skip N4 on their bias-corrected inputs; Stanford and Iowa JJ-TMS
retain N4. The runtime/model/CUDA gate runs once per pending batch. Existing older output contracts name the previous atlas and must be replaced
from completed runs of the active release before checking upgraded products.
HCP defaults to eight subjects and accepts `--jobs` up to twelve; other datasets
use two, with BANDA serial. The approved HCP launch, CPU affinity, GPU slots and
resource patches are maintained in the [runtime manifest](../../docs/environment/hippo2_reproducibility_manifest.md#batch-resources-and-retained-outputs).

For the complete HCP reconstruction, manifest mode explicitly uses
`--keep-going` to attempt independent subjects after an individual failure.
The default remains stop-on-failure. A keep-going batch still returns a
nonzero final status if any subject fails and does not bypass official QC.

Stage 01 runs the complete official participant DAG and checks all 96 formal
products with `--fail-on-missing --no-cleanup`. `--notemp` retains anatomy,
coordinates, warps and CIFTI intermediates. Resume requires a current matching
`output_check.json` and every declared output; a cleanup receipt, a boundary
function test or a pair of meshes is insufficient.
Do not apply retired local scientific patches or use development `main` in
place of the selected release.
For `IncompleteFilesException`, wait for active workers to exit, activate
hippo2, and rerun the subject's exact `hippunfold_argv` from preset `--dry-run`
with `--rerun-incomplete`. Then resume the normal manifest command.

## Inputs and QC

Select exact artifact paths, ordered run IDs and physical-run lengths.
Missing, ambiguous or mismatched declarations stop processing; another file or
an older cache is never selected by existence order. Run concatenation stays
in memory.

- `02_01` samples BOLD NIfTI only. Run-level mode uses
  `--run-artifact-manifest-template`; HCP7T rest with an existing concatenated
  BOLD uses `--bold-template` plus `--run-metadata-template`. It does not require
  or read CIFTI declarations. Run lengths must match the BOLD frame count.
- `03_01` and `03_02` consume cortical CIFTIs. Run-level mode uses
  `--run-artifact-manifest-template`; a supplied single rest CIFTI requires
  `--run-metadata-template`. Task references require declared run-level CIFTIs
  and their original BOLD inputs. Iowa's `xcpd_surf_concat` tree is unsupported.
- PFM `03_02` requires individualized Kong2022 ArealMSHBM labels generated by
  the [Argon backend](../ArealMSHBM_Kong2022/README.md). It does not generate
  those labels. Fixed `03_01` uses the selected public atlas.
- Lynch `03_04` consumes the exact run-artifact manifest and local FASTANS
  priors; `03_03` requires explicit run metadata. `02_03` includes every base
  QC subject and records NSI as audit metadata without an NSI cutoff.

Sampling transforms native `space-T1w_den-512` meshes into the declared BOLD
space and saves mapped meshes in `surface/raw/sampling_surfaces/`. Raw metrics,
physical smoothing, cortical-neighbour distances and subject vertex areas use
native T1w geometry; corobl surfaces provide display correspondence. For cortex
controls, native cortical meshes require `--cortex-coordinate-space T1w`.
Each run is sampled once on a temporary disconnected bilateral mesh and split
back into the original left/right vertex order, avoiding repeated decompression.
Raw and smoothing jobs share coordinator threads; at most two subjects launch
single-threaded Workbench processes concurrently.

Run-level sampling saves each hemisphere's complete concatenated sequence as
`surface/raw/sub-<ID>_hemi-<L|R>_timeseries.npy` and a `*_bold.func.gii`.
`surface_sampling_summary.json` records physical run order and frame lengths;
keep the coverage arrays, vertex masks and native surfaces with these signals.
Fixed surface analyses can consume these saved inputs. The current raw-resume
and participant-QC commands still require the declared source BOLD files;
deleting volumes before changing that input contract produces missing-input
status. New geometry or voxel-level analyses require the volumes again.

The shared participant QC implements overall mean Power FD <=0.50 mm, bilateral Dice >=0.70,
and at most 10 unusable vertices per consumed den-512 hemisphere. Unusable means
no trilinear NIfTI grid support or a nonfinite/constant sampled signal in any
physical run. Coverage uses `*_nifti_support.npy` and
`sub-<subject>_coverage_summary.json` with policy
`bold_nifti_runwise_trilinear_validity`.
Participant QC and smoothing use the same vertex validity calculation.

The current `latest` QC roster defines the cohort; there is no fixed historical
subject-count requirement. `--qc-run-inputs` supplies the same explicit
physical-run BOLD/motion table to sampling and participant QC for every dataset
and task. Supply the complete roster with `--subjects` or `--subjects-manifest`
so participants without declared runs remain visible as unassessable. Columns
are `subject,run_id,run_label,bold_path,n_timepoints,motion_path,motion_format`
(tab separated). Motion formats are `hcp_movement_degrees` and `fmriprep_confounds`.
For censored XCP-D volumes, add `temporal_mask_path` for the exact outlier TSV:
motion uses original post-dummy frames, while `n_timepoints` and coverage use
the retained BOLD frames. NIH rTMS uses `--dataset nih_rtms` and separate
`outputs_nih_rtms/inputs/kong2022/ses-1/` and `ses-2/` roots.
Use `--require-hippunfold-version` with the version expected for the reconstruction being assessed.
Task/non-HCP preparation requires its own `--route-root`, `--shared-root`, and
an explicit `--template-root` for group-average preparation.

`02_01 --stage group-average` builds a self-contained group template in
`<template-root>/group_average_surfaces/`: the mean folded (corobl)
midthickness per hemisphere, the HippUnfold unfolded midthickness and subfield
labels, and `group-average_den-512_label-hipp_surfaces.spec`, which declares
all six files. Each subject's spec-declared unfolded surface and subfield labels
must be identical; the step stops and names any subject that differs. Steps
that need folded and unfolded geometry (`05_02`, `05_06`, `06_01`, `06_10`,
`01_01`–`01_04`) read both from this template only; the template spec must
declare the folded surface it sits beside. The stage manifest records the
subfield atlas and the number of subjects checked. Workbench renders of the
folded hippocampus still use the HippUnfold spec of `template_subject`.
Pass lists use the selected dataset/task prefix; HCP rest retains `hcp_7t`.
Missing declared BOLD inputs are reported as unassessable and are skipped by
sampling, rather than stopping other participants or selecting another modality.
Rebuild incompatible sampling, coverage and masks explicitly with `--no-resume`
before smoothing; changing source code does not replace production products.

### Cortical signal validity

Cortical reference generation retains vertices with finite signal
throughout the declared physical runs. The per-hemisphere masks and summaries are
written under `reference/common/validity/`. References and downstream FC must
be rebuilt when their recorded validity policy differs from this one.

## Branches and analysis defaults

Gradient producers and independent fit callers expose `--feature-scaling
{none,col_zscore}`, `--sparsify-stage {features,affinity,none}` and `--sparsity`.
The defaults preserve column z-scoring and BrainSpace feature sparsification
(0.9 for Kong400; 0.0 for banks with at most 50 features). Pearson affinity
is constructed once, clipped nonnegative, mean-symmetrized with its kernel
diagonal retained, then embedded with an exact diffusion map and no second
sparsification. The embedding decomposes the symmetric conjugate of the
diffusion operator with a direct solver, so it takes no random seed and
Ward partitions are a deterministic function of the data. Disconnected
diffusion support is rejected.

`gradient_preprocessing` records the effective settings; missing or mismatched
settings require explicit regeneration instead of cache reuse.

Clustering has a separate `--clustering-feature-scaling {none,unit_sd}` choice.
Unit-SD scaling fits component means and population SDs (`ddof=0`) on supported
training vertices after component selection/alignment. Each independent split
fits its own transform; held-out data use the training transform. Zero-variance
components fail. Native component weighting remains the default.

Functional K=3 cluster names are `Anterior`, `Middle`, `Posterior` in
anterior-to-posterior order (`Ant`, `Mid`, `Post` in compact figures).
The anatomical set is `Head`, `Body`, `Tail`; keep each set complete.
Source labels remain separate from data-derived network annotations.

| Workflow branch | Route | Reference/gradient contract |
|---|---|---|
| `kong2022_pfm` | `pfm` | Individualized Kong400 FC; canonical7 default |
| `kong2022_fixed` | `fixed` | Public-atlas Kong400 FC; canonical7 default |
| `kong2022_pfm_17net`, `kong2022_fixed_17net` | `pfm_17net`, `fixed_17net` | Reuse PFM/Fixed FC, gradients and the corresponding canonical7 spatial clusters; explicit family17; gradient methods plus clusterless `wta-top` |
| `kong2022_pfm_ctxreg` | `pfm_ctxreg` | Hippocampal signals with the 20 mm local cortex mean regressed out per run; upstream PFM references and surface masks |
| `kong2022_pfm_ctxreg_10mm` | `pfm_ctxreg_10mm` | Hippocampal signals with the 10 mm local cortex mean regressed out per run; upstream PFM references and surface masks |
| `lynch2024_20net`, `lynch2024_7net` | Matching atlas slug | Hard-label Lynch references; both reuse the Lynch20 gradient bank |

`pfm_ctxreg` (signal branch `ctxreg`) and `pfm_ctxreg_10mm` (signal branch
`ctxreg_10mm`) are the cortex-control signal branches; the former
`pfm_ctxdrop` branch is retired. Both regress, per run, the mean of valid
cortex within a native T1w radius out of each hippocampal vertex series in
`04_00`; the radius is fixed per route in
`lib/branch_config.CTXREG_ROUTE_DISTANCE_MM` (20 mm and 10 mm).
`04_00 --route-tag {pfm_ctxreg,pfm_ctxreg_10mm}` defaults to `pfm_ctxreg`;
`--distance-mm` defaults to the route's radius and rejects any other value, and
`--out-shared-root` defaults to the route's formal root. 10 mm is the smallest
radius at which essentially every hippocampal vertex has cortex in range and
matches the 6-10 mm range of earlier subcortical work; its regressor is mostly
default-mode network (DMN), so the two routes bracket the same boundary-signal
removal, stronger at 10 mm. Each route keeps `topsim` gradients and
`topsim/gradient-ward` plus `wta-top` group atlases and overviews under its own
roots. `06_07_render_local_cortex_control_diagnostics.py`
reports target-side local-cortex exclusion (ctxdrop) next to ctxreg at 3, 6,
10 and 20 mm as an FC/WTA-level diagnostic that feeds nothing downstream:
`--stage subjects` writes `outputs_hcp/work/local_cortex_control/sub-<id>/`,
`--stage render` requires every roster subject and writes
`outputs_hcp/present/local_cortex_control/`. In HCP 7T, exclusion changes the
winning network for only about 1-4% of hippocampal vertices, whereas ctxreg
changes about a quarter to two fifths, an effect reproduced by regressing only
cortex within 3 mm; both ctxreg routes are therefore conservative sensitivity
analyses that remove boundary-shared signal, possibly including genuine hippocampal or
subicular signal.

For family17 group overviews, pass `--route-tag pfm_17net` or
`--route-tag fixed_17net` to `06_01_build_group_overview.py`. The explicit
route selects the family17 assignment taxonomy for lineage validation;
the consumed gradient FC bank must have 400 ordered Kong2022 ROI columns.
Family17 overviews use the same canvas and column proportions as canonical7,
with vertical network labels. Signed
Fisher-z heatmap annotations preserve negative signs.
For the same per-smoothing layouts as canonical7, select `--smoothing 2mm` or
`4mm` and each `--annotation-pair` (`roi-fisherz`, `network-ts-fisherz`,
`top20`). Their OVR rows use family17 ROI or family17 network-time-series
features, respectively.

Family17 `wta-top` is a clusterless run over the `family17` registry column
with the same `normalized_top20_count` assignment as canonical7. It does not
use `--reuse-fixed-cluster-root`:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/05_02_run_group_atlas.py --shared-root outputs_hcp/inputs/kong2022/shared_inputs/pfm --surface-root outputs_hcp/inputs/kong2022/shared_inputs/surface --reference-root outputs_hcp/inputs/kong2022/shared_inputs/pfm --out-root outputs_hcp/work/group_atlas/pfm_17net --methods wta-top --atlases kong2022 --route-tag pfm_17net --network-taxonomy family17
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/06_01_build_group_overview.py --group-root outputs_hcp/work/group_atlas/pfm_17net --present-dir outputs_hcp/present/overviews/pfm_17net/wta-top/2mm --atlases kong2022 --methods wta-top --coverage-threshold 0.80 --route-tag pfm_17net --assignment-variant normalized_top20_count --smoothing 2mm
```

Repeat the overview with `4mm`. `06_01` rejects a Kong WTA source whose
manifest `network_taxonomy` differs from the route (`family17` for
`*_17net`, otherwise `canonical7`).

To produce family17 annotations for an existing TopSim/Ward result, use
`05_02_run_group_atlas.py --route-tag pfm_17net --network-taxonomy family17
--profile topsim --methods gradient-ward --reuse-fixed-cluster-root
outputs_hcp/work/group_atlas/pfm/topsim` with the same source inputs and cohort.
The analogous TopFC run selects `--profile topfc --methods gradient-spectral`
and the `pfm/topfc` source. The output records its canonical7 source and reuses
its spatial labels and K diagnostics.

For HCP cortex controls, the formal route roots are
`outputs_hcp/inputs/kong2022/shared_inputs/<route>/`,
`outputs_hcp/work/hippomaps_gradients/<route>/`,
`outputs_hcp/work/group_atlas/<route>/`, and
`outputs_hcp/present/overviews/<route>/`. Pass PFM references and surface masks
explicitly when a controlled signal root is separate.

- `gradient-spectral` consumes aligned `05_01` gradients, defaults to G1/G2/G3,
  and uses shifted-cosine surface affinity. `05_02` evaluates K=2..8 and renders
  primary K=3 while recording automatic K selection separately.
- `--gradient-components` (default `1,2,3`) selects the clustering gradients for
  every gradient method in `05_02` and the rendered input in `06_01`; each
  selection is a separate `cov_<tau>/<g-tag>/` run (`g1`, `g1g2`, `g1g2g3`).
  `06_01` renders exactly the selected tag and puts non-default selections in a
  `<g-tag>/` subdirectory of the overview destination. TopSim Ward also has
  `g1` and `g1g2` runs for `pfm`, `pfm_ctxreg` and `pfm_ctxreg_10mm` (ctxreg
  routes use their own `--shared-root`, `--out-root` and `--route-tag`):

  ```bash
  /home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/05_02_run_group_atlas.py --shared-root outputs_hcp/inputs/kong2022/shared_inputs/pfm --surface-root outputs_hcp/inputs/kong2022/shared_inputs/surface --reference-root outputs_hcp/inputs/kong2022/shared_inputs/pfm --out-root outputs_hcp/work/group_atlas/pfm/topsim --methods gradient-ward --profile topsim --atlases kong2022 --gradient-root outputs_hcp/work/hippomaps_gradients --route-tag pfm --network-taxonomy canonical7 --k-min 2 --k-max 8 --default-k 3 --gradient-components 1,2
  /home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/06_01_build_group_overview.py --group-root outputs_hcp/work/group_atlas/pfm/topsim --methods gradient-ward --atlases kong2022 --route-tag pfm --smoothing 2mm --gradient-components 1,2
  ```
- The head-to-tail axis is the anterior-posterior coordinate of the HippUnfold
  unfolded sheet stored in the group template (see Inputs and QC). The folded
  surface's principal axis is not used: it compresses the curled head, so a
  thin strip running head to tail can tie with the cluster that fills the head.
  `05_02` numbers final clusters head to tail (C1 nearest the head) by the mean
  unfolded coordinate of their vertices and records `cluster_order_axis:
  hippunfold_unfolded_anterior_coordinate` with `cluster_order_template_spec`.
  `06_01`, `06_10` and `05_06` reject a `cluster_order_head_to_tail` that
  differs from the order recomputed on that axis. Long-axis bins and profiles
  (`05_06` task bins, `06_01` WTA profiles, the WTA long-axis figures) and the
  individual cross-fit steps (`01_01`–`01_04`) use the same axis: equal-vertex
  bins follow the unfolded coordinate, and vertices of one unfolded row keep
  their folded-axis order.
- Kong2022 cluster assignments default to `network_ts_fisherz`; ROI `fisherz`
  and `normalized_top20_count` are explicit comparisons. `05_02` computes
  `network_ts_fisherz` for every Kong2022 cluster run, and `--resume`
  recomputes a run whose manifest records another default. Lynch cluster
  runs default to their direct-network `fisherz`. Kong ROI-registry `wta-top` uses only
  `normalized_top20_count`. Lynch direct-network WTA uses `fisherz`.
- Kong2022 retains all 400 ordered ROI rows, including NaN rows for empty
  parcels. Pooled FC removes each declared run mean at computation time and
  preserves within-run variance weighting. Undefined correlations remain NaN.
- Gradient clusters have three network-assignment branches: `fisherz`
  (correlate with each ROI, average z within network), `network_ts_fisherz`
  (average ROI time series into a grayordinate-weighted network series, then
  correlate) and `normalized_top20_count`. `wta-top` is the clusterless
  fourth branch. OVR models exist for `fisherz` and `network_ts_fisherz`
  inputs only; the Top20 overview has no OVR rows. `06_01 --assignment-variant
  both` renders all three gradient annotation pairs; `07_01` marks the TopSim
  Ward 2 mm `network-ts-fisherz` overview as the primary figure.
- Kong2022 gradient overviews of a three-cluster partition add a column of
  cluster-to-cortex FC tiles (C1 Anterior, C2 Middle, C3 Posterior). Each tile
  shows the left/right mean of the run's `cluster_roi_profile.npy` on the
  inflated cortex (`RdBu_r`, linear and zero-centred, saturating beyond the
  90th percentile of |Fisher-z| across the figure's three tiles) with its ten most positive and
  ten most negative ROIs. `lib/cortex_fc_tiles.py` draws the tile for the
  overview and for the multi-dataset hippocampus/cortex figure. Family17
  gradient overviews use the same 400-ROI cortical profiles and column;
  `wta-top` overviews have no such column.
- Renderers consume exact manifest-declared assignment and template assets.
  Select `--group-root`/`--out-root`, route and assignment variant explicitly
  for render-only runs. `07_01` defaults to `--routes pfm`; other routes and
  methods require their corresponding built intermediates.

### Dataset and preprocessing variants

For cohort gradients, pass the same `--cohort <name> --variant 36p` to
`05_01`, `05_02` and `06_04`. Gradient root arguments name the stage root above
the route; for example `pfm` plus `--cohort nthc` resolves to
`<gradient-stage-root>/pfm/nthc/<smoothing>/hemi_<hemi>/`.
Select matching upstream roots, subject rosters and group-atlas roots separately.
HCP omits `--cohort`. Iowa JJ_TMS subgroup routes use 36P.

The seven overview cohorts are NTHC, TEHC, NTS, NTS Any MDD, NTS Pure MDD, TIS,
and TIS PTSD. Select the complete roster for the intended analysis.

### Additional gradient methods

The shared `lib/gradient_clustering.py` dispatcher accepts `gradient-ward`,
`gradient-gmm`, and `gradient-spectral-rbf`, alongside the existing
`gradient-spectral` shifted-cosine method. Production defaults remain unchanged.
Ward uses Euclidean features and binary mesh connectivity and rejects disconnected
support. Full-covariance GMM has no spatial constraint and reports disconnected
clusters, convergence, and covariance conditioning without repairing labels.
RBF spectral uses unique undirected binary mesh edges and float64 weights with
sigma equal to `--rbf-scale` times the median positive edge distance. The primary
scale is 1; 0.5 and 2 are separate sensitivity settings. Disconnected positive
support or support after the declared relative-weight cutoff of 1e-5 fails the
candidate. Small clusters are reported, never rebalanced.

Stage `05_02` supports `--k-selection-mode fixed-comparison --k-min 3 --k-max 3`.
This exports only the requested K and its diagnostics, leaves automatic K unset,
and treats data and fitting errors as fatal. To write a dataset comparison to
repository `tmp/`, pass `--comparison-output --dataset outputs_hcp` and an
explicit output root with either K-selection mode; consumed source roots must
belong to that dataset. Formal output routing is unchanged and a comparison does
not replace it.

`--automatic-primary-k` exports the automatically chosen K as the primary
partition, in place of the `--default-k 3` display default. A combination in
which no K passes the gates is recorded as `skipped_no_k_survived_constraints`
and has no overview. `--k-final-rule` sets how the automatic K is taken from the
candidates that pass the minimum-size, connectedness, coherence and redundancy
gates. `smallest-stable` (default) takes the smallest split-half ARI local
maximum within one standard error of the best. `largest-admissible` takes the
largest passing K; split-half ARI is then reported as a diagnostic and does not
decide K. The manifest records the rule as `k_final_rule`.

`--undersized-default-k-fallback <K>` applies to a `--default-k` primary
partition. When the default-K partition has a parcel below the minimum parcel
size, the declared fallback K is exported in its place, provided that partition
meets the size gate; otherwise the run stops. The manifest records
`primary_k_source: default_k_undersized_fallback` and the replaced K as
`undersized_default_k`. The participant-graph comparison uses `--default-k 3
--undersized-default-k-fallback 2` with `--redundancy-threshold 0.80`.

### Mean-similarity Ward

`mean-similarity-ward` is a Kong2022-only gradient method outside the
production defaults and requires `--profile topsim`. Each participant's
feature is the Pearson similarity between the Fisher-z 400-ROI fingerprints of
that participant's valid vertices. The group affinity averages each vertex
pair over the participants who observe both vertices, sets the diagonal to 1,
keeps the top 10% of each row, clips negatives and mean-symmetrizes before the
exact diffusion map. Gradients are aligned to the method's own 2 mm-left
anchor and clustered with the mesh-constrained Ward of `gradient-ward`, so
`--gradient-components`, the K scan, split-half ARI, nulls, redundancy gate and
annotation follow the gradient cluster contract. Full-cohort gradients are
computed in `05_02`; manifests record `gradient_recomputed: true` and no
`05_01` source. `--resume` is not supported for this method.

`subject-gradient-ward` uses the same participant similarity and TopSim
affinity, but embeds each participant separately (exact diffusion map, G1–G5).
Each participant's gradients are rotated onto the `mean-similarity-ward`
template of the same training subjects with orthogonal Procrustes (reflections
allowed); generalized Procrustes then iterates with the vertexwise mean over
observing participants, re-rotated onto the template so G1, G2 and G3 keep the
template's identity. The selected components of that mean are clustered with
mesh-constrained Ward. A participant whose affinity is disconnected stops the
fit. Manifests record the Procrustes iterations, convergence and the
per-component correlation with the template.

`subject-gradient-hbp` keeps every participant's Procrustes-aligned selected
components instead of averaging them, divided by one common scale (the RMS of
the Procrustes mean). HierarchBayesParcel (`--hbp-root`, revision recorded)
fits a vertex-specific independent arrangement with an isotropic Gaussian
mixture emission (unnormalized centres, 10 initializations). The group parcels
are mesh-constrained Ward on the per-vertex group prior probabilities, so each
is connected; the posterior argmax gives the individual maps. Fits are
serialized so the seed determines each fit. A mean-field Potts arrangement was
tested and not adopted: with a learned group prior it produced confident,
geometry-driven partitions on vertex-permuted data. Manifests record the prior
entropy, the fragmentation of the prior argmax, its agreement with the connected
map, and individual-map fragmentation and agreement.

`multigraph-gradient-spectral` keeps the `multigraph-spectral` participant
graphs and shared-partition discretization, but replaces each participant's
leading eigenvectors with a constant column followed by the eigenvalue-scaled
G1..G(K−1) of the exact diffusion map. It has no mesh constraint.

The method comparison figure
(`scripts/visualize/method_comparison_summary/render_method_comparison_summary.py`)
places gradient Ward, mean-similarity Ward, subject-gradient Ward and
subject-gradient HBP (G1, G1+G2, G1–G3) and the three participant-graph methods
in rows, and Standard, CtxReg 10 mm and CtxReg 20 mm
side by side within each smoothing. Each card shows split-half ARI, the
redundancy gate, and the network-TS Fisher-z and network-TS OVR dominant bars
and unfolded/folded hippocampus maps of the exported partition. A green frame around the
hippocampus marks an annotation row in which both hemispheres export K=3, pass the redundancy gate
at K=3, assign three different networks to the three clusters and assign the
same network to each anterior-to-posterior cluster in both hemispheres. One tall
frame spans both rows when the Fisher-z and OVR annotations both qualify. The renderer
stops unless every run shares the route, smoothing, coverage, canonical7
taxonomy, 0.80 redundancy threshold, roster, K scan and resampling settings:

```bash
HIPP_DATA_CHECKOUT=/home/yojiao/hpchome/hipp LP_NUM_THREADS=1 VTK_SMP_MAX_THREADS=1 xvfb-run -a \
  /home/yojiao/.conda/envs/py314/bin/python \
  scripts/visualize/method_comparison_summary/render_method_comparison_summary.py \
  --group-atlas-root <outputs_hcp/work/group_atlas: gradient rows at <route>/topsim/<method>, graph rows at <route>/<method>> \
  --out <output directory>
```

It writes the combined figure, gradient-only and graph-only versions as
300 dpi 256-colour palette PNGs (`--dpi` changes the resolution), and
`method_comparison_summary_manifest.json` with every consumed manifest and
label checksum.

`--rows <key>,<key>,...` (for example the G1–G3 row of every method) instead
draws only those rows, in that order, as `method_comparison_summary_by_tier.png`
with `method_comparison_summary_by_tier_manifest.json`. G1–G3 is the default
component set and is left out of its row labels. Rows are grouped by how much
they average across participants before the partition is formed: individual
model (the partition is estimated from every participant's own data; a cohort
average may serve only as an intermediate alignment reference, as in HBP),
subject estimates averaged (per-participant similarity, graph or gradients
averaged, then clustered), and averaged FC.

### Participant-graph spectral methods

`multigraph-spectral` and `mean-graph-spectral` (`lib/multigraph_spectral.py`)
are Kong2022-only methods outside the production defaults. Each participant's
subject feature is the Pearson similarity between the Fisher-z 400-ROI
fingerprints of that participant's valid vertices. The affinity keeps positive
weights among each vertex's top 10% neighbours and uses no mesh constraint.
`multigraph-spectral` (Shen et al., 2013) embeds every participant's graph
separately and fits one shared partition with one rotation per participant.
`mean-graph-spectral` averages the participant affinities edgewise and applies
the same discretization once. A disconnected participant or mean graph stops
the run with the participant and component sizes.

Both methods use the standard K scan (K = 2..8, at most eight spectral
coordinates). The lowest-loss solution of ten seeded starts is exported for
each K as fitted, so an unsupported K appears as an empty or undersized parcel
and fails the minimum-size gate. Split-half and null repetitions in which a
half assigns all shared vertices to one parcel are counted as skipped. The
coherence gate uses the cosine of each vertex's K spectral coordinates to its
parcel centroid within each fitted graph. Annotation, OVR and manifests follow
the gradient cluster contract; `06_01` renders the same annotation-pair
overviews, and its cluster-to-cortex column shows every exported parcel (L/R
mean when both hemispheres export the same K, otherwise per hemisphere).

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/05_02_run_group_atlas.py --comparison-output --dataset outputs_hcp --shared-root outputs_hcp/inputs/kong2022/shared_inputs/pfm --surface-root outputs_hcp/inputs/kong2022/shared_inputs/surface --reference-root outputs_hcp/inputs/kong2022/shared_inputs/pfm --out-root tmp/<comparison>/group_atlas/pfm --methods multigraph-spectral mean-graph-spectral --atlases kong2022 --route-tag pfm --network-taxonomy canonical7 --k-min 2 --k-max 8 --instability-resamples 20 --null-resamples 20 --default-k 3 --undersized-default-k-fallback 2 --k-final-rule largest-admissible --redundancy-threshold 0.80 --assignment-variant both network_ts_fisherz ovr_logistic_canonical7 ovr_logistic_canonical7_network_ts
xvfb-run -a /home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/06_01_build_group_overview.py --group-root tmp/<comparison>/group_atlas/pfm --present-dir tmp/<comparison>/overviews/pfm/multigraph-spectral/2mm --methods multigraph-spectral --atlases kong2022 --route-tag pfm --smoothing 2mm --annotation-pair network-ts-fisherz
```

Explicit `--assignment-variant ovr_logistic_canonical7 ovr_logistic_family17`
adds separate descriptive OVR models to the existing Fisher-z and normalized
Top-20 exports. Each model uses the same subject-by-cluster Kong400 profiles
computed from cluster-mean hippocampal time series. It aggregates equal ROI
Fisher-z means within each authoritative network, without imputing missing
network values. The fixed liblinear L2 model uses C=1, tol=1e-4 and max_iter=1000,
without extra feature scaling or class weighting; its primal solver is
deterministic and takes no seed.
`both` continues to mean Fisher-z plus normalized Top-20; OVR requires explicit
selection and does not change cluster boundaries. The largest strictly positive
raw coefficient determines each OVR label; all other cases are unassigned.

Each OVR export includes the exact subject-by-cluster input table, class/network
order, beta, intercept, raw OR, feature SD, per-SD beta and OR, and a coefficient
figure. SD uses ddof=0 over the exact flattened fitting observations. Raw OR is
per one Fisher-z unit; per-SD OR changes reporting units, not model fitting.
Coefficient plots use zero and OR plots use one as the reference. These are
descriptive conditional associations with dependent observations within subjects,
not accuracy estimates, inferential tests, causal effects, or a numerical
reproduction of Kember's different inputs and standardization.

OVR group overviews summarize the six fitted left/right clusters; the blank
hemisphere separator in the coefficient heatmap is excluded from bar summaries.


### HCP affinity averaging comparison

The standalone comparison fits the Standard, CtxReg 10 mm and CtxReg 20 mm
routes at 2 mm and 4 mm for both hemispheres. Both arms use the HCP QC roster,
Kong2022 400-ROI FC, 0.80 vertex coverage, TopSim with 0.90 affinity sparsity,
the 2 mm left alignment anchor, and surface-connected Ward with G1–G3 at
K=3. The mean-similarity arm computes Pearson across jointly observed ROIs
within each subject and gives equal weight to subjects with a defined
correlation for that vertex pair. TopSim sparsification follows aggregation
in both arms. The mean-FC arm averages Fisher-z FC before computing Pearson.

The comparison exports cluster label GIFTIs, aligned gradients, descriptive
Kong2022 canonical7 network labels, and five paired subject half-splits.
The four annotation variants are ROI Fisher-z, grayordinate-weighted network
time-series Fisher-z, and OVR from each of these two feature sets. The weighted
variant combines cortical ROI time series using their valid grayordinate
counts before correlating with hippocampal cluster time series. The ROI variant
correlates each ROI separately and then averages Fisher-z within networks.
Each OVR model uses the largest strictly positive raw coefficient for labeling.
Each half is fitted independently, then its maps score the other half.
Each held-out score subtracts the mean Pearson similarity of neighboring
vertices across parcel boundaries from that of neighboring vertices within
parcels. The summary averages each subject's paired scores across the splits
before pooling subjects, and reports split-half adjusted Rand index separately.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
/home/yojiao/.conda/envs/py314/bin/python scripts/hcp_ctxreg_affinity_comparison/compare.py \
  --out tmp/ctxreg-mean-subject-similarity \
  --ctxreg-10mm-fc-root outputs_hcp/inputs/kong2022/shared_inputs/pfm_ctxreg_10mm
LP_NUM_THREADS=1 VTK_SMP_MAX_THREADS=1 xvfb-run -a \
  /home/yojiao/.conda/envs/py314/bin/python scripts/hcp_ctxreg_affinity_comparison/compare.py \
  --out tmp/ctxreg-mean-subject-similarity --render-only
```

The renderer reads the exported label GIFTIs and the recorded template
surfaces. It uses the standard BrainSpace hippocampal grid and the HCP CtxReg
2×2 composite. The top row shows dorsal views and the bottom row shows ventral
views, with the left and right hemispheres side by side. It writes
`comparison.tsv` and `ctxreg_affinity_averaging.png` in the selected experiment
directory, alongside four variant-suffixed PNGs and `network_labels.tsv`.
Each network panel names the assigned networks in C1 to C3 order for each
hemisphere. To annotate an existing experiment, run `compare.py --out <dir>
--annotate-only --mask-root <shared surface root>` before rendering. Pass
`--kong2022-dlabel <path>` when running from a worktree without the atlas asset.
Pass the absolute experiment path to `--out` to render from a separate worktree. The 10 mm FC root must contain the complete roster. Generate
missing FC inputs with Stage `04_01` from the 10 mm residuals,
the Standard PFM cortical references and shared hippocampal validity masks.

## Implementation boundary

Numbered scripts contain orchestration and stage logic; shared implementation
belongs in [lib/](lib/). Use the existing I/O, FC, surface-coordinate, template,
reference and rendering modules. Dataset-specific utilities belong in their
own script directories. Keep the numbered root free of unnumbered launch,
audit and migration helpers.

Formal inputs, intermediate analyses, presentation products and execution
records belong under each dataset's `inputs/`, `work/`, `present/` and `runs/`
roots as defined in the output layout. Production-tree replacement, migration,
and archive access require the repository's explicit path authorization.

# Core Workflow Entrypoints and Implementation

This directory is the canonical active workflow surface.
Both CLI entrypoints and stage implementation modules live under `scripts/core_workflow/`.

Formal reference input contracts:

- `03_01` and `03_02` (including their extraction modules) reject Iowa's
  legacy `xcpd_surf_concat` input tree. Select canonical run CIFTIs with
  `--run-artifact-manifest-template`; the manifest supplies ordered paths,
  explicit run IDs, file sizes, and SHA256 receipts. Concatenation stays in
  memory. A single CIFTI requires an explicit `--run-metadata-template`.
- `02_01_prepare_group_inputs.py` uses the same artifact manifest template for
  exact BOLD/dtseries pairing and validates both modalities before creating
  surface outputs.
  Sampling starts from native `space-T1w_den-512` meshes, transforms their
  vertices into the declared MNI152NLin6Asym BOLD space, and saves those meshes
  under `surface/raw/sampling_surfaces/`. The core workflow, Iowa7T and
  JJ_TMS use trilinear BOLD interpolation; JJ_TMS and the core workflow share
  `lib.surface_sampling.sample_volume_to_surface`. Raw and smoothed metrics use
  `space-T1w` filenames because their vertices belong to the native mesh.
  Millimetre smoothing, cortical-neighbour distances and subject vertex areas
  use native T1w geometry. Corobl meshes remain available for display.
  For `04_00` and `04_01 --ctxdrop`, the HCP default cortical meshes are MNI
  and are transformed back to native T1w before distance queries; pass
  `--cortex-coordinate-space T1w` when supplying native fMRIPrep cortical meshes.
  Select the registration with `--dataset hcp|stanford|banda|pdc|iowa_jj_tms|iowa7t`;
  `--registration-root` explicitly overrides the system's Jianglab prefix.
  HCP uses its declared FNIRT warp/source image; BANDA/PDC use the declared
  FNIRT pair; fMRIPrep datasets use the MNI-to-T1w image composite for ANTs
  point mapping. An old corobl raw summary causes resume to stop with a rebuild
  instruction. See `docs/hipp_parcellation_network/step_by_step_flows.md` for
  the coordinate contract; QC cutoffs are selected independently.
- `03_04_run_lynch2024_fastans_pfm.py --run-artifact-manifest-template`
  selects one exact per-subject manifest, preserves its declared order, and
  records input paths, run IDs, and receipts. All consumed production inputs
  reject temporary directory components and the legacy Iowa concat tree.
- `03_03_build_lynch2024_reference.py --run-metadata-template` selects a
  per-subject JSON containing explicit run IDs and lengths, optionally under
  `header_evidence`. Missing IDs, contradictory records, noninteger lengths,
  and frame-total mismatches fail before reference products are written.
  It uses the shared `lib.io_utils.load_run_segment_metadata` parser.
  Run IDs, segments and the metadata source are recorded in output manifests.
  The metadata option is required; there is no equal-four-way fallback.
- Direct CIFTI extractor run mode requires both `--run-artifact-manifest`
  and matching run inputs; `--run-ids`, when supplied, must match the manifest.
  Single-file rest input requires `--run-metadata-template`: its run lengths
  must sum to the CIFTI header's total frames, and its declared partition is
  retained for QC and run slicing. Task references require run-level CIFTIs
  and their manifest-declared original BOLD inputs.

Shared implementation boundaries:

- `03_01` and `03_02` call their extraction functions in the same Python
  process through `ReferenceRequest`. Each worker returns its own success or
  error; RouteA source-aware resume and explicit artifact selection remain in
  the batch stage. Both extractors use
  `lib.kong2022_reference.materialize_reference_timeseries` for ordered
  400-parcel export and unweighted canonical7/family17 means. RouteA keeps
  dataset/task-aware QC and individualized labels; RouteB keeps its public
  dlabel and HCP rest tSNR25 policy.
  Both `fixed` (standard group atlas) entrypoints default to `shared_inputs/fixed`, separately from
  `pfm` (individualized functional mapping). The callable `fixed` extractor rejects unsupported dataset, task,
  threshold and atlas selections before reading inputs.
- `04_00` and `04_01` share native hippocampal surface paths, coordinate loading
  and valid-cortex neighbour queries in `lib.surface_coordinates`. Coordinate
  registration and the independent coordinate plausibility check remain
  explicit; this extraction does not change the physical space or 20 mm query.
- `05_01` and `05_02` use `lib.branch_config.resolve_signal_branch_for_route`
  for the standard/ctxreg contract. Other atlas routes keep their own identity.
  `05_01`, `05_02`, and `06_04` select cohort gradient banks explicitly with
  `--cohort <name> --variant 36p|22p`. Their `--out-root` / `--gradient-root`
  arguments name the gradient stage root, above the route directory. For
  example, `--cohort nthc --variant 22p` with route `pfm` writes and reads
  `<gradient-stage-root>/pfm_22p/nthc/<smoothing>/hemi_<hemi>/`.
  Omit `--cohort` for HCP and use the default `--variant 36p`. Cortex atlas
  and signal route names stay `pfm`, `fixed`, or their explicit controls;
  the variant parameter supplies the `_22p` directory suffix. Select the
  matching input, surface, subject-manifest, and group-atlas output paths
  explicitly; these parameters do not choose upstream inputs.
  Gradient manifests record both selectors. `05_02` includes them in its
  existing resume key, so selecting a different cohort or variant prevents
  reuse of results from the previously selected gradient bank.
- `06_03`, `06_09` and `06_10` call
  `lib.render_cortex_pfm_scene.render_cortex_pfm_scene` directly. Workbench
  remains an external process; scene paths, view rotations and image sizes
  are unchanged. `05_03` and `07_01` retain process-level job orchestration.
  `06_03` and `06_10` read cortex assets only from
  `<data-root>/sub-<cortex-template-subject>/anat/`, using the exact filenames
  `sub-<id>_hemi-{L,R}_space-fsLR_den-32k_desc-MSMAll_inflated.surf.gii` and
  `sub-<id>_space-fsLR_den-32k_desc-MSMAll_sulc.dscalar.nii`. Missing inputs
  report their expected paths. Each invocation splits both sulc metrics from
  that declared dscalar into its existing render runtime's `cortex_assets/`
  directory, replacing the derived pair rather than reusing a global cache.
  `--scene` and the Kong2022 dlabel remain explicit inputs; `06_10` exposes
  `--dlabel`, while `06_03` uses the maintained Kong2022 dlabel.
- `06_04`, `06_06` and `06_07` share HippoMaps loading, screenshot return-type
  handling and bilateral image assembly in `lib.hippomaps_adapter`; each stage
  supplies its own data, colour ranges and annotations.
- `06_05` is a retired entrypoint that reports its replacement (`06_04`) and
  exits with code 2. Live hippocampal scene utilities belong to
  `lib.workbench_scene_batch`; visualization consumers import that library
  directly. `06_02` remains the independently runnable label materializer.

See [the selective integration record](../../docs/ctxreg_selective_integration.md)
for scope and deliberately excluded worktree changes.

- Naming scheme: `NN_MM_<semantic_name>.py`
- Numbering reflects global pipeline order, not branch identity.
- Active branches: `kong2022_pfm`, `kong2022_pfm_17net`, `kong2022_pfm_ctxreg`, `kong2022_pfm_ctxdrop`, `kong2022_fixed`, `kong2022_fixed_17net`, `lynch2024_20net`, `lynch2024_7net`

`05_02_run_group_atlas.py` accepts `--signal-branch standard|ctxreg` for
PFM controls. Lynch and Fixed routes remain independent and do not inherit
the CtxReg signal contract.

The root is a protected numbered workflow surface: do not add unnumbered
dataset-specific, audit, migration, promotion, or launch code here. Shared
implementation belongs in `lib/`; this checkout ships no formal maintenance
entrypoints; JJ_TMS-specific utilities belong in `scripts/jj_tms/`.

## Stanford 22P input paths

For the Stanford Causal-Connectome 22P input stage, use only the following
Jianglab paths. The `surf` input is the cortex time-series CIFTI, not a
structural surface:

- `surf` CIFTI:
  `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/DataOrg_PFM_22P/sub-*/func/*_task-rest_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii`
- `vol` denoised BOLD:
  `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/DataOrg_PFM_22P/sub-*/func/*_task-rest_space-MNI152NLin6Asym_res-2_desc-denoised_bold.nii.gz`
- Kong2022 labels:
  `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/ArealMSHBM_Kong2022_22P/sub-*/{lh_labels.txt,rh_labels.txt}`

The authoritative 22P root is
`/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/DataOrg_PFM_22P/`.
Do not use `DataOrg_PFM_36P` or the unversioned `ArealMSHBM_Kong2022`
directory for a 22P run. On Argon, use the equivalent `/Shared/jianglab`
prefix.

The canonical seven 22P overview cohorts are: NTHC, TEHC, NTS, NTS Any MDD,
NTS Pure MDD, TIS, and TIS PTSD.

## Locked HippUnfold stage-01 contract

Stage `01_01_run_hippunfold_t1w_local.py` has one closed launch contract. Run it
with the explicit py314 interpreter and select exactly one dataset preset:

```bash
/home/yojiao/.conda/envs/py314/bin/python \
  scripts/core_workflow/01_01_run_hippunfold_t1w_local.py \
  1001 --dataset stanford
```

The presets are fixed and are not positional input/output arguments:

- `hcp`: `data/hippunfold_input` → `outputs_hcp/inputs/hippunfold/dense_corobl_batch/`, T1w `.nii.gz`.
- `stanford`: `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/DataOrg_PFM_36P` → `outputs_stanford/inputs/hippunfold/dense_corobl_batch/`, T1w `.nii.gz`.
- `iowa_jj_tms`: `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS/raw` → `outputs_iowa_jj_tms/inputs/hippunfold/dense_corobl_batch/`, T1w `.nii`.
- `banda`: `/home/yojiao/LSS/jianglab/3_Data_Working/Depression/BANDA/bids_input` → `outputs_banda/inputs/hippunfold/dense_corobl_batch/`, T1w `.nii.gz`.
- `pdc`: `/home/yojiao/LSS/jianglab/3_Data_Working/Depression/PDC/bids_input` → `outputs_pdc/inputs/hippunfold/dense_corobl_batch/`, T1w `.nii.gz`.

Every preset fixes HippUnfold 2.0 to modality `T1w`, density `512`, output
space `corobl`, autotopology `hipp dentate`, unfolded registration enabled,
GPU inference, one core, and the canonical shared local model/cache. No
`HIPPUNFOLD_*` override or `--omit-from`/free-form Snakemake argument is accepted.
Use `--dry-run` to inspect the resolved JSON contract without creating outputs. A
direct one-subject run gates the local `hippo2` executable in the same fixed
environment used for processing, requiring exact executable resolution, CLI
version `2.0.0`, and successful T1w/512 help.

### Input preprocessing

The BANDA/PDC presets set `skip_preproc=True` and automatically pass
`--skip-preproc`: their staged T1w inputs link to HCP-processed
`imaging/<scan>/T1w/T1w_acpc_dc_restore.nii.gz` products, which already have
bias-field correction. The established HCP, Stanford, and JJ_TMS presets retain
`skip_preproc=False`. The dry-run JSON includes the selected value and command.

For future datasets, choose this setting from the documented upstream
bias-field correction status of the consumed input, rather than field strength
or voxel size. In the T1w workflow, `--skip-preproc` bypasses N4; template
registration, cropping/resampling, segmentation, and unfolding still run.
See the [official preprocessing documentation](https://hippunfold.khanlab.ca/en/main/pipeline/pipeline.html).

The same `01_01_run_hippunfold_t1w_local.py` entry point accepts either
`<subject> --dataset <dataset>` or
`--dataset <dataset> --manifest <subject-list.txt> --jobs <N>`.
Single-subject `--dry-run` emits one JSON contract; manifest `--dry-run` emits
an ordered JSON array without running the environment gate or processing.
The BANDA/PDC batch scripts under `outputs_<dataset>/work/` use manifest mode.
The batch gates the runtime once before its
pending subjects; each direct one-subject invocation still performs its own
gate, and a new batch checks again. PDC runs up to three HippUnfold subjects in
parallel; BANDA remains serial. Output checking and automatic cleanup run one
subject at a time because they share the batch integrity manifest. After
successful output validation, the checker removes `anat/`, `coords/`, `warps/`,
and `cifti/`, retaining `surf/`, `metric/`, and `qc/`.

After an interrupted run, Snakemake may report `IncompleteFilesException`.
Recover the named subject with its validated preset command plus the explicit
Snakemake argument `--rerun-incomplete`, then run `01_02` for that subject.
Resume the normal manifest command after existing workers have exited; validated
completion receipts let the batch skip finished subjects. The recovery argument
is a one-off operator action and is not added to the standard launch preset.
Only subjects with an existing `core_workflow_cleanup_receipt.json` are skipped.
Outputs without that receipt resume through HippUnfold and the complete output
check; a pair of surface files alone does not establish completion.
The batch invokes the checker with `--fail-on-missing`. A processing or checking
failure stops dispatching new subjects; already running subjects finish and
receive their serial output check before the batch reports the failure.
Runtime preparation, execution helpers, and system detection are maintained in
`lib/hippunfold_run_contract.py`; batch scheduling stays in the stage-01 entry
point.

### Shared resource resolution

HippUnfold checks the exact shared cache for the selected registration template,
surface atlas, and nnUNet model before registering their download rules. Complete
local resources are external DAG inputs and cause no HTTP requests, including
metadata `HEAD` requests. An already unpacked nnUNet v1 model is reusable without
its original archive. Atlas files already materialized in `atlases/` are reused;
otherwise the existing `atlases_dl/` extraction supplies the required copies.
When a resource is absent, its original upstream download rule and declared URL
are used. Existing incomplete template, extraction, or model directories fail
with the exact missing filenames so they can be repaired explicitly.

The installed Snakemake source cache refreshes local workflow code when its
modification time changes. This keeps the shared runtime source cache from
continuing to execute rules from before a package patch. Stage 01 checks that the
resource adapter is installed before launching. The runtime patches and package
reinstallation commands are recorded in
[the environment manifest](../../docs/environment/hippo2_reproducibility_manifest.md#local-resource-resolution).

The shell adapter `lib/hippunfold_shell/run_hippunfold_local.sh` is a
compatibility entrypoint only; it accepts exactly `<dataset> <subject>` and
delegates to this entrypoint.
The output checker requires fixed L/R midthickness anchors in `space-corobl`,
`space-T1w`, and `space-unfold` (all `den-512`), plus the surface spec and both
Dice QC files before a subject can be marked complete or cleaned up.

The `run_stage` entrypoints (`05_01`, `05_04`, `06_04`, `06_06` and `06_09`)
accept the wrapper options below before `--`. Stage-specific options follow
`--`; use `-- --help` for the stage's own argument help. `--dry-run` emits the
selected wrapper manifest without executing the scientific stage. Other steps
write their own stage manifests and expose their own CLI options.

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/05_04_compute_confidence_maps.py \
  --workflow-branch kong2022_pfm \
  --upstream-step-id 05_02_run_group_atlas \
  --step-manifest-out tmp/confidence_stage.json \
  --dry-run
```

PFM reference construction has a required algorithmic upstream artifact: subject-level Kong2022 ArealMSHBM labels generated by the Argon backend before `03_02_build_kong2022_routea_reference.py`. The artifact-level contract, including the flat subject label paths and required evidence, is maintained in `docs/hipp_parcellation_network/core_workflow_artifact_lineage.md`.

For the PFM cortex-control branch, run `04_00_materialize_cortex_regressed_surface_timeseries.py` first. It writes only controlled hippocampal time series and diagnostics under the formal root `outputs_hcp/inputs/kong2022/shared_inputs/pfm_ctxreg/`; PFM references and `surface` masks remain in their upstream roots and are passed to downstream steps via `--reference-root` and `--mask-root`. `04_01` and `04_02` should write FC into the controlled root while reading references from `pfm` and masks from `surface`; uncontrolled PFM roots are not overwritten.

For the PFM cortex-drop branch, run `04_01_materialize_kong2022_roi_fc.py --dataset hcp --ctxdrop` with `--shared-root outputs_hcp/inputs/kong2022/shared_inputs/pfm_ctxdrop`, `--surface-root outputs_hcp/inputs/kong2022/shared_inputs/surface`, `--reference-root outputs_hcp/inputs/kong2022/shared_inputs/pfm`, and `--mask-root outputs_hcp/inputs/kong2022/shared_inputs/surface`. This branch keeps hippocampal time series unchanged, excludes valid cortex vertices within 20 mm before ROI averaging, stores smoothing-independent local cortex-neighbor diagnostics under `sub-*/surface/local_cortex_neighbors/hemi_*/`, and is maintained for `gradient-spectral` and `wta-top` outputs. Use `06_07_render_ctxdrop_neighbor_counts.py --source-branch pfm_ctxdrop` to render the group-average local neighbor-count hippocampal surface overlay under `outputs_hcp/present/local_cortex_neighbors/`; the same renderer supports `--source-branch pfm_ctxreg` from the ctxreg diagnostics.

For the 17-net taxonomy branches (`pfm_17net` and `fixed_17net`), do not create a new input root or gradient root. Reuse `outputs_hcp/inputs/kong2022/shared_inputs/pfm` (or `fixed`) and run `05_02_run_group_atlas.py` with `--route-tag pfm_17net` (or `fixed_17net`) `--network-taxonomy family17 --methods gradient-spectral --out-root outputs_hcp/work/group_atlas/pfm_17net` (or `fixed_17net`). `05_02` reads gradients from `outputs_hcp/work/hippomaps_gradients/pfm` (or `fixed`) for `gradient-spectral`, so `05_01` should not be run as `pfm_17net` or `fixed_17net`. Because these branches intentionally reuse their upstream 400-ROI input roots, `--route-tag auto` refuses `--network-taxonomy family17`; pass `--route-tag pfm_17net` or `fixed_17net` explicitly. WTA is restricted to the canonical7 views and cannot run with family17.

The unsuffixed Kong2022 `pfm` branch is the canonical7 default. `03_02 --dataset hcp --atlas-slug
kong2022` writes direct `reference/kong2022/` canonical7 products, `04_01` materializes the ordered 400-ROI FC bank under the historical
`hipp_vertex_to_network_fc.npy` filename and writes explicit canonical7 summary
files for network-only diagnostics; `05_01` computes the `pfm` gradient bank
from the 400-ROI columns, and `05_02` performs clustering and K selection in the `pfm`
root. In `05_02`, canonical7 network-profile time series are always derived
from the ordered 400-ROI reference; any coexisting direct canonical7 summary is
not selected by filesystem presence. Taxonomy-only fixed-cluster reuse also
requires matching source/target gradient arrays and 400-ROI FC provenance.
`04_01` validates that ordered 400-ROI registry first, then derives canonical7
labels explicitly from its family17 values; legacy TSVs that omit the derived
canonical7 column remain valid, while direct seven-network files are never used
as a fallback or replacement for the 400-ROI FC bank.
HCP has no native 22P input branch.

`03_02`, the direct RouteA extractor, and `04_01` require explicit `--dataset`
selection (`hcp`, `stanford`, `iowa_jj_tms`, `iowa7t`, `banda`, or `pdc`).
Reference resume compares the selected source paths, run IDs/lengths, baseline
inputs, dataset, task, and cutoff. A changed selection is recomputed;
`--overwrite` also forces recomputation. Before consuming a Kong2022 reference,
`04_01` requires matching dataset/task/validity policy/cutoff in its summary and
both cortex gate summaries. Historical null-policy references must be rebuilt
with the intended dataset and inputs; their presence does not establish validity.

Empty cortical parcels retain their ordered rows with all-NaN time series and
zero contributing vertices, preserving the fixed 400-ROI bank. Missing or constant
series have NaN FC and do not count as zero-valued observations in group means.
Weighted cluster profiles intersect group support with each subject's valid mask
and exclude incomplete/constant rows; entirely missing network profiles remain
unassigned.

Stored baseline-bearing surface and cortex time series are unchanged. Pooled FC
in `04_01` (including ctxdrop), `04_02`, cluster annotation, weighted profiles, and
cluster-cortex panels removes each declared run's mean at computation time. It
preserves within-run variance weighting, rather than z-scoring runs or averaging
run-wise Fisher-z correlations. Per-run FC still uses that run's original series.
Consumers require explicit ordered run IDs and lengths from the exact reference
`reference_summary.json` (or a caller-selected run metadata file where supported);
missing/mismatched boundaries fail before FC writes. This numerical contract does
not deploy a new participant QC threshold.

Template assets are a producer invariant, not a presentation fallback. The
existing `05_02` branch and any `05_05` copied-branch path derive
`template_left_or_right_surface`, `template_subject`, and `template_spec_path`
from the target dataset and preprocessing variant before writing a manifest.
The shared `lib/template_assets.py` module contains that reusable canonical
contract. `06_01` and `06_10` only validate the written paths and fail clearly
when a manifest contains a foreign absolute path; they do not repair stale
paths from another dataset.

For the Lynch2024 FASTANS/PFM comparison branches, first run `02_03_build_lynch2024_nsi_manifest.py` against the QC-pass manifest and an NSI TSV/CSV with `subject,nsi`. `02_03` includes every base subject and writes the NSI values to an audit table; it has no NSI threshold or expected-count gate. External FASTANS priors are local assets under `external/FASTANS/resources/PFM/priors/Lynch2024/`; `03_04_run_lynch2024_fastans_pfm.py` generates official per-subject hard-label PFM dlabel/dscalar assets and hard-label component time series under `outputs_hcp/work/cortex_pfm/sub-<id>/lynch2024/`; current shared inputs live under `outputs_hcp/inputs/lynch2024/shared_inputs/pfm/`. `03_03_build_lynch2024_reference.py` consumes those hard-label component assets and writes both `reference/lynch2024_20net/` for original Lynch parent networks excluding `Noise` and `reference/lynch2024_7net/` for canonical merged networks from `config/cross_atlas_network_merge.json`. `04_01_materialize_kong2022_roi_fc.py --atlas lynch2024_20net` reads `cortex_roi_parcel_timeseries.npy` / `cortex_roi_parcels.tsv`; `--atlas lynch2024_7net` reads `cortex_canonical7_network_timeseries.npy` / `cortex_canonical7_networks.tsv`, so canonical7 FC caches are 7-column outputs rather than duplicated 20-component FC. The maintained Lynch downstream methods are `gradient-spectral` and direct-network `wta-top`, with K=2..8 evaluation and a non-forced default K=3 for gradient primary labels; `05_01` computes one non-Noise Lynch20 component gradient bank with BrainSpace full-graph affinity (`brainspace_sparsity=0.0`), and the `lynch2024_7net` route reuses that bank for `gradient-spectral` while applying canonical7 references downstream. `05_02` applies shifted-cosine surface affinity to gradient clusters and direct top-1 Fisher-z network assignments to Lynch WTA outputs. When `--methods` is omitted for a Lynch atlas, the producer and overview default to both `gradient-spectral` and `wta-top`.

For Lynch7 taxonomy-only refreshes, pass `--reuse-fixed-cluster-root` to the existing `05_02_run_group_atlas.py` output root that contains the fixed labels. The runner validates the source support mask and K, reuses the existing gradient/clustering/K-selection products, and recomputes only the Lynch7 reference-based profile and assignment products.

For the Kong2022 `gradient-spectral` branch, run
`05_01_compute_hippomaps_gradients.py` first. `05_01` writes aligned
HippoMaps/BrainSpace gradient artifacts: raw G1-G5 are matched and sign-flipped
within each route to the `4mm/hemi_L` anchor when available, and only aligned
arrays are saved. `05_02_run_group_atlas.py` loads those aligned artifacts from
`--gradient-root` and clusters the selected columns. For the cortex-control
routes, pass the controlled FC root through `--shared-inputs-root` and the
unmodified PFM cortex references through the explicit
`--reference-inputs-root outputs_hcp/inputs/kong2022/shared_inputs/pfm`;
this keeps the 400-ROI FC source and reference taxonomy separate. The default
gradient set is
`G1,G2,G3`; valid overrides are `--gradient-components 1`, `2`, `3`, `1,2`, and
`1,2,3`. The maintained default evaluates `K=2..8` and renders the explicit primary `--default-k 3`; automatic K selection remains recorded in the K summary rather than being overwritten.
`05_01` records the resolved `signal_branch` (`standard` or `ctxreg`) in every
gradient manifest, and `05_02 --signal-branch` validates both that field and
the manifest route before consuming the gradient array. For CtxReg, the
default hippocampal profile root is the controlled shared-input root, while
the default cortex reference root is the explicitly mapped RouteA sibling;
an explicit `--reference-root` remains available when that mapping is not the
intended input. Lynch and RouteB gradient runs do not inherit this CtxReg
contract.

The formal gradient-spectral surface affinity is fixed to `shifted-cosine`,
`(1 + cosine_similarity) / 2` on surface-adjacent vertices. Group and cross-fit
workflows record this definition and graph diagnostics in their manifests.

The former formal-release audit, refresh, migration, promotion, and background
launch helpers under `scripts/maintenance/` were retired from this checkout.
Do not invoke or restore those deleted paths. Current validation uses active
numbered workflow entrypoints, generated manifests, and the multi-dataset suite
readiness check.

HippoMaps-style continuous gradients are computed by
`05_01_compute_hippomaps_gradients.py` and rendered by
`06_04_render_hippomaps_gradient_products.py`. This route imports official
package code from `external/hippomaps`, keeps Fisher-z before group averaging,
applies column-wise feature z-score before `GradientMaps(kernel="pearson")`,
uses BrainSpace sparsity `0.0` for low-dimensional feature banks (`n_features <= 50`) and `0.9` for high-dimensional banks such as Kong400, then aligns
component order and sign before writing the maintained gradient arrays. For Lynch2024 this means direct full-graph gradient fitting on the non-Noise hard-label component bank, not a separate 8-column gradient bank. For `den=512`, the renderer treats `512` as HippoMaps v2 in the current
Python process and disables `unfoldAPrescale`; no AP surfarea rescale is used
for 512-density figures. The renderer defaults to bilateral products and renders
each already-aligned hemisphere with its own colorbar and EV labels before
stitching the two sides. Use `--hemis L` or `--hemis R` for single-hemisphere
output files. Endpoint FC contrast and `gradient-spectral` clustering consume
the aligned gradient orientation.


Each dataset stores inputs directly under `outputs_<dataset>/inputs/`, for
example `outputs_hcp/inputs/kong2022/` and
`outputs_stanford/inputs/hippunfold/`. The dataset name appears only in the
outer output root. Scientific variants remain explicit below `inputs/`,
including Stanford `kong2022` and `kong2022_22p`. Retired output archives
belong under `wrong/outputs_<dataset>/<original-relative-path>`; moving an
archive preserves its name and internal hierarchy. Active workflows use
the canonical dataset roots rather than archive locations.

Canonical presentation outputs live under `outputs_hcp/present/`. Method
overview PNGs use `outputs_hcp/present/overviews/<route>/`. Supplementary
continuous-gradient products live in `outputs_hcp/present/gradients/`, and
cortex panels live in `outputs_hcp/present/cortex_panels/`. Directory names
intentionally omit `t1only`, `present_fixedk`, and `10perc`; fixed K=3 is the
default presentation policy. Use `tests/core_workflow/smoke_prepare_run.py` only for isolated
5-subject tests under `outputs_hcp/smoke_runs/<run_id>/`, and use
`outputs_hcp/reruns/<run_id>/` for non-official render-only records. Subject-specific cortical parcellation map rendering for Kong2022 ArealMSHBM is performed by `06_09_plot_cortex_kong2022_mshbm.py`, storing GIfTI and CIFTI assets under `outputs_hcp/inputs/kong2022/cortex_references/pfm_areal_mshbm/sub-<subject>/assets/` and composed 2x2 overview PNGs under `outputs_hcp/inputs/kong2022/cortex_references/pfm_areal_mshbm/sub-<subject>/figures/`.

The saved gradient-spectral cluster-to-cortex template can be rendered with `06_10_render_cluster_roi_strength_panels.py`. It reads `gradient-spectral/kong2022/{2mm,4mm}/cov_0p80/g1g2g3/hemi_{L,R}/cluster_roi_profile.npy`, keeps Fisher-z values, paints all 400 parcels with a shared blue-white-red scale, and writes `cortex_kong2022_gradient-spectral_2mm4mm_template_strength_overview.png` plus a manifest under `outputs_hcp/present/cortex_panels/`. The four rows are 2mm-L, 2mm-R, 4mm-L, and 4mm-R for the source hippocampus; every panel colors bilateral L/R cortex, and K=3 columns use validated Anterior/Body/Posterior names.

The same entrypoint renders the canonical7 normalized WTA pseudo-cluster template by default with `--method wta-top --wta-root outputs_hcp/work/group_atlas/pfm`. It consumes the existing vertex-level `normalized_top20_count` WTA labels, selects the four most frequent canonical7 networks across the four 2mm/4mm L/R runs, recomputes subjectwise mean hippocampal-network time series to 400 cortical ROI Fisher-z templates, and writes the single 4x4 overview `cortex_kong2022_wta-normalized_7net_2mm4mm_template_strength_overview.png`.

Official full-cohort group-atlas intermediates generated by `05_02` live under
`outputs_hcp/work/group_atlas/`. The maintained PFM 7-net root is
`outputs_hcp/work/group_atlas/pfm/`. The PFM 17-net taxonomy root is `outputs_hcp/work/group_atlas/pfm_17net/`. Fixed standard-atlas branches use `outputs_hcp/work/group_atlas/fixed/` and `outputs_hcp/work/group_atlas/fixed_17net/`. HCP 400-ROI cortex-control matrix and overview consumers use the formal direct roots `outputs_hcp/work/group_atlas/pfm_ctxreg/` and `outputs_hcp/work/group_atlas/pfm_ctxdrop/`; dated, recovered, or regenerated trees are migration sources only and are never active resolver targets. Current Lynch2024 comparison roots are `outputs_hcp/work/group_atlas/lynch2024_20net/` and `outputs_hcp/work/group_atlas/lynch2024_7net/`. For Kong2022
`intrinsic-spectral`, `network-spectral`, and `gradient-spectral`, the workflow
keeps two official cluster-to-network assignment variants: `fisherz`
(`top1_fisher_z` from the mean network profile) and `normalized_top20_count`
(`top20_count_normalized_by_network_roi_count`: top-20 positive ROI counts
divided by the finite available ROI count in each network). The normalized
Top-20 count variant is an explicit cluster-level comparison, not the default.
The cluster-level default is `fisherz` (`top1_fisher_z`). Raw `count` is no
longer registered or generated by `core_workflow`; existing historical
raw-count artifacts are not removed by this contract change. Cluster-level
outputs do not keep unsuffixed default duplicates. `group_atlas_manifest.json`
records `default_assignment_variant` and its top-level default fields point
directly to the selected suffixed variant files, normally `_fisherz`.
`06_02_materialize_group_labels.py` consumes only the exact in-run paths
declared for every assignment variant (including its status, NPY, label-map,
and GIFTI files); missing cluster labels, missing declarations, outside-run
paths, and cross-artifact disagreement fail closed. It does not invent a
variant filename or maintain an unsuffixed alias.

For `gradient-spectral`, `06_01_build_group_overview.py` validates every
declared assignment variant against its exact status, NPY, label-map, GIFTI,
and profile assets before rendering; it does not repair those inputs. Each
render also writes a `.png.sources.json` sidecar containing the overview hash,
source manifests, and exact assignment-asset fingerprints.

For an existing run whose top-level pointers were generated under the retired
normalized default, use the explicit metadata-only command
`scripts/core_workflow/lib/repair_cluster_level_assignment_default.py` with one
exact `--run-dir` per target. It validates the manifest-declared Fisher-z
artifacts before writing and does not recompute, delete, or overwrite the
`normalized_top20_count` variant.

For clusterless `wta-top` Kong2022 ROI-registry atlases, the only label
assignment is `normalized_top20_count`: top-20 positive ROI/component counts
divided by the finite available ROI/component count in each network. The same
assignment is used for canonical8 and canonical7 taxonomy views. No other WTA
assignment is registered or generated by the active workflow.

Render-only reruns may point `06_01_build_group_overview.py`,
`06_03_build_cluster_cortex_panels.py`, and `06_06_render_confidence_maps.py`
at validated group-atlas intermediates with `--group-root`/`--out-root`. When
`--route-tag` is omitted or set to `auto`, these renderers infer the route from
the group-atlas directory name using exact route names (`pfm`, `pfm_17net`,
`pfm_ctxreg`, `pfm_ctxdrop`, `fixed`, `fixed_17net`, `lynch2024_20net`, or
`lynch2024_7net`). If an explicit route tag disagrees with a canonical
`group_atlas/<route>` directory, the script aborts instead of writing
cross-route outputs. `07_01_build_group_matrix.py` defaults to `--routes pfm`
and exports PNGs below `--present-dir/<route>/` while keeping its matrix manifest and README at `--present-dir`. Direct `06_01` calls derive the matching `present/overviews/<route>[/<cohort>]` destination from a canonical group-atlas root unless an exact `--present-dir` is supplied. `07_01` reads branch group-atlas outputs from the explicit route roots under `outputs_hcp/work/group_atlas/`; this includes `pfm`, `pfm_17net`, `pfm_ctxreg`, `pfm_ctxdrop`, `fixed`, or `fixed_17net` as requested. The `pfm_17net`/`fixed_17net` (`family17`) branches are gradient-only and are excluded from WTA matrix jobs. Lynch20/Lynch7 direct WTA is not routed through the Kong ROI-registry WTA matrix contract; it uses direct network Fisher-z assignments. It renders both
`fisherz` and `normalized_top20_count` overviews for cluster-based methods when both variants are registered, and one normalized WTA overview for the Kong ROI-registry atlas. Direct `06_01_build_group_overview.py` runs can pass `--assignment-variant both`; it resolves to the registered cluster variants and the single Kong WTA normalized Top-20 assignment. Before rendering, producer steps write dataset/variant-canonical surface and HippUnfold spec paths into each manifest. The renderer uses the shared `scripts/core_workflow/lib/template_assets.py` contract only to validate those paths; it does not repair obsolete absolute mounts or search foreign fallbacks. A stale or foreign path fails clearly, and bounded panel messages prevent a long exception path from overlapping neighboring panels.
pass `--routes pfm pfm_17net pfm_ctxreg pfm_ctxdrop fixed fixed_17net` only after the requested non-default route intermediates have
been generated. For partial exports (for example only `gradient-spectral` and
`wta-top`), use `--methods` to restrict matrix rendering to already-built
methods. The
public method names and intermediate child directories are `intrinsic-spectral`,
`network-spectral`, `gradient-spectral`, and `wta-top`; legacy `group-*` method
directories are not part of the active contract. For cortex panels,
`--shared-root` must be a merged per-subject cache containing both `surface/` and
`reference/` trees, such as `.../fixedk_work/shared_routea_merged`, not only
the reference or surface subtree.

Execution environment (hard requirement):

- Do not run core workflow tasks with bare `python`.
- Use `/home/yojiao/.conda/envs/py314/bin/python` for core workflow scripts/tests.
- Workbench calls go through `scripts/core_workflow/lib/workbench.py`, which
  defaults `CORE_WORKFLOW_WB_THREADS=1` and sets the common native thread
  variables (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
  `NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`,
  `ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS`) before every `wb_command` call.
- `02_01_prepare_group_inputs.py` hard-caps subject-level Workbench stages at
  two concurrent subjects even when `--jobs` is higher. This keeps Python
  workers plus Workbench subprocesses below the repository concurrency limit.
- See `docs/environment/agent_runtime_contract.md` for preflight checks.

Entrypoint arguments:

Kong2022 reference batch entrypoints (`03_01` and `03_02`) accept
`--data-root`; `03_01`, `03_02`, and `05_02` also accept
`--kong2022-dlabel`. The wrappers pass explicit per-subject dtseries paths and
explicit atlas assets to extraction/annotation modules, so smoke runs can reuse
validated trees outside the current worktree without creating symlinks. `05_01`
and `05_02` accept `--hippomaps-root` for the same reason.

Path template arguments for non-HCP datasets:

- `02_01_prepare_group_inputs.py` accepts `--bold-template` (default:
  `{bold_root}/{subject}/func_uncompressed.nii`).
- For run-level surface sampling, `02_01_prepare_group_inputs.py` accepts
  `--run-artifact-manifest-template`. Each selected manifest run must contain
  exact absolute BOLD and dtseries paths with matching SHA256/size receipts;
  the two modalities must have matching timepoint counts. The volume is
  sampled onto the HippUnfold geometry. Surface arrays are concatenated in
  memory, and no volume or CIFTI concat artifact is written.
  CIFTI subcortical grayordinates are not a substitute for a full denoised
  volume.
  Subject-level QC uses the standard `02_02` invalid-vertex threshold of 0.20.
- `03_01_build_kong2022_reference.py`,
  `03_02_build_kong2022_routea_reference.py`,
  `04_00_materialize_cortex_regressed_surface_timeseries.py`, and
  `04_01_materialize_kong2022_roi_fc.py` accept `--dtseries-template`
  (default: `{data_root}/{sub_subject}/func/{sub_subject}_task-rest_run-concat.dtseries.nii`).
- Templates support `{data_root}`, `{bold_root}`, `{subject}`,
  `{sub_subject}`, `{hemi}`, and `{hemi_lower}` where relevant.
- Defaults preserve existing HCP behavior.
- `03_01_build_kong2022_reference.py` and
  `03_02_build_kong2022_routea_reference.py` also accept
  `--run-artifact-manifest-template`. Use this option when run-level CIFTIs
  are available; the selected manifest is validated before extraction, and its
  declared path order and run IDs are passed through unchanged. No
  `run-concat.dtseries.nii` or other temporary concat artifact is created.
  Per-run Workbench GIFTIs used for hippocampal surface sampling live only in
  temporary directories, and their ephemeral paths are omitted from the
  persisted surface summary. Pass
  `--no-resume` to `02_01` when existing surface outputs and summaries must be
  regenerated.
  `--dtseries-template` remains available for datasets that already provide a
  pre-concatenated input, but that mode also requires the selected
  `--run-metadata-template` with explicit run IDs and lengths.
- Stanford, Iowa, Iowa_HCP500, and future non-HCP datasets should prefer templates over broad
  symlink trees. Use the shared helper
  `scripts/core_workflow/lib/path_templates.py` to add template support to
  additional entrypoints when needed.
- `iowa_hcp500` represents the Iowa 7T dataset preprocessed via the HCP pipeline (source data under `/home/yojiao/hpchome/HCPpipeline500`). It maps outputs to the independent `/home/yojiao/hpchome/hipp/outputs_iowa_hcp500/` directory.

- `--workflow-branch`: one of `kong2022_pfm`, `kong2022_pfm_17net`, `kong2022_pfm_ctxreg`, `kong2022_pfm_ctxdrop`, `kong2022_fixed`, `kong2022_fixed_17net`, `lynch2024_20net`, `lynch2024_7net`
- `--upstream-step-id` (repeatable)
- `--input-artifact` (repeatable)
- `--output-artifact` (repeatable)
- `--step-manifest-out`
- `--dry-run`

Unknown arguments are forwarded to the local stage implementation module.

Runtime boundary:

- allowed external assets/executables:
  - configs/data/generated outputs
  - `software/workbench/bin_linux64/wb_command`
- forbidden for active steps:
  - Python/Shell helper scripts outside `scripts/core_workflow/`
  - direct runtime dependency on legacy helper trees (`common/*`, `workbench/*`) or the repo `wb_command` wrapper

The step inventory, branch coverage matrix, and artifact lineage are defined in:

- `scripts/core_workflow/step_registry.json`
- `docs/hipp_parcellation_network/core_workflow_artifact_lineage.md`
- `scripts/core_workflow/lib/`

Implementation structure contract:

- Numbered step scripts (`NN_MM_*.py`) should focus on flow orchestration and stage-specific domain logic.
- Shared I/O helpers belong in `scripts/core_workflow/lib/io_utils.py`.
- Shared FC numeric helpers belong in `scripts/core_workflow/lib/fc_math.py`.
- Shared GIFTI label builders belong in `scripts/core_workflow/lib/gifti_labels.py`.
- Runtime metadata helpers (timestamps, JSON writer, git/file metadata) remain in `scripts/core_workflow/lib/workflow_utils.py`.
- Shared hippocampus-to-cortex template calculations remain in `scripts/core_workflow/lib/cortex_template_profiles.py`.
- New utility-style helpers should be added to `lib/` modules rather than redefined in numbered step files.

The WTA-top overview embeds a fixed 20-bin long-axis network profile as its middle column, with percent-of-bin vertices on the horizontal axis and long-axis bins on the vertical axis. The WTA overview and task-activation profiles both use `lib.surface_utils.long_axis_order` followed by `lib.surface_utils.equal_vertex_long_axis_bins`: consecutive PCA-ordered vertices are split into 20 bins, with any extra vertices assigned to earlier bins. At density 512, bins 1–12 contain 26 vertices each and bins 13–20 contain 25 each. Both displays number bins 1–20. These are equal-vertex groups, not equal distances along an anatomical centerline.

For cluster-based `normalized_top20_count` overviews, `06_01_build_group_overview.py`
reconstructs the displayed normalized network profile from each cluster's
saved `normalized_count_by_network` status mapping. It does not reuse the
Fisher-z `cluster_network_profile.npy` for normalized-count validation or
plotting.

Post-hoc PFM weighted comparison outputs can be generated with `05_05_build_weighted_fisherz_branch.py`. This writes `outputs_hcp/work/group_atlas/pfm_weighted/` and `outputs_hcp/present/overviews/weighted_fisherz/`, preserving the original PFM `fisherz` branch while adding `weighted_fisherz` gradient assignment labels from vertex-count weighted network time series recomputed as Fisher-z FC. Each copied gradient manifest must carry the formal `shifted-cosine` provenance. WTA is not part of this comparison branch. Render the comparison branch with `06_01_build_group_overview.py --assignment-variant weighted_fisherz --mode-tag weighted_fisherz --methods gradient-spectral`. Gradient overview manifest items record the four exact source group manifests and the shifted-cosine definition, and their visible titles include `shifted-cosine` without changing canonical PNG filenames.

The weighted branch accepts canonical roots under `outputs_hcp`,
`outputs_stanford`, and `outputs_iowa_jj_tms`. Pass the intended source,
surface, and reference roots explicitly for each dataset/version.

Use `05_05_build_weighted_fisherz_branch.py --methods gradient-spectral`; the
weighted branch accepts gradient-spectral only.

The weighted builder requires each copied assignment variant to declare its
exact `template_functional_atlas_label`, and requires the source
`cluster_network_profile.npy` before replacing it with the weighted profile.
That source is preserved as
`cluster_network_profile_roi_equal_fisherz_baseline.npy` and is recorded in
the `fisherz` variant metadata; a missing label or baseline fails closed.

For Stanford subgroup reruns, keep the weighted outputs in separate branch roots and pass Stanford shared-input/template roots explicitly so manifests do not fall back to HCP assets. HC uses `outputs_stanford/work/group_atlas/routea_hc_weighted_fisherz_full_k3/`; Symp uses `outputs_stanford/work/group_atlas/routea_symp_weighted_fisherz_full_k3/`. Both should pass `--surface-root outputs_stanford/inputs/kong2022/shared_inputs/surface`, `--reference-root outputs_stanford/inputs/kong2022/shared_inputs/routea`, and `--template-surface-dir outputs_stanford/inputs/kong2022/templates/hipp_group_average_t1only_151/group_average_surfaces`, and `--template-spec outputs_stanford/inputs/hippunfold/dense_corobl_batch/sub-1001/hippunfold/sub-1001/surf/sub-1001_den-512_label-hipp_surfaces.spec`. The Stanford validation script supports `--analysis-version weighted_fisherz` to compare HCP weighted, Stanford HC weighted, and Stanford Symp weighted branches and to recompute subject-level endpoints from weighted network time series rather than equal ROI Fisher-z averaging. All ROI-registry WTA runs use normalized top20-count labels; no weighted or historical WTA assignment is produced.

The cortex-control materializer (`04_00`) and ROI-FC materializer (`04_01`)
accept an explicit `--run-metadata-template` for recorded run IDs and lengths;
run boundaries are validated for order, contiguity, and total timepoints, with
no inferred four-way split. `04_00` additionally supports an explicit
`--dtseries-run-template` and concatenates those exact run CIFTIs in memory;
the selected metadata is required for that mode. Cortex validity masks are
required by default for cortex regression and ctxdrop, and invalid hippocampal
vertices remain unchanged and are recorded in regression diagnostics.

Default-root contract for HCP cortex controls: omitting `--shared-root` from
`04_01_materialize_kong2022_roi_fc.py --dataset hcp --ctxdrop` selects the formal
`outputs_hcp/inputs/kong2022/shared_inputs/pfm_ctxdrop/` FC root and
reads references from the authoritative `pfm` root. `05_01`, `05_02`,
`06_01`, `06_07`, and `07_01` likewise resolve only the formal
`pfm_ctxreg`/`pfm_ctxdrop` roots. Dates and run identifiers belong in
manifests and logs, not in active result paths; explicit roots still take
precedence.

HCP 3T task branches (`emotion`, `wm`, `relational`) reuse the same anatomy
and RouteA labels without rerunning ArealMSHBM. `02_01` and `03_02` accept
`--task` (default `rest`, which preserves legacy behavior) plus
`--task-manifest-root`; task mode consumes prebuilt per-subject manifests
from `lib/task_manifests.py` (`task-<task>/sub-<subject>.json` with SHA256/size
receipts for paired BOLD volumes and dtseries), validates every run length
against the task frame contract (176/405/232), and records `task` in
summaries. New numbered stages `04_03` (first-level OLS activation with
SPM-canonical HRF-convolved FSL EVs plus 12-column movement nuisance;
maintained contrasts `faces_gt_shapes`, `twobk_gt_zerobk`, `rel_gt_match`,
`places_gt_others`, `load_x_places`; run identity/order/lengths from the
manifest, per-run contrast files, invalid-signal vertices NaNed before the fit
with coverage recorded; the cached surface series is only trusted when the
`surface/raw/surface_sampling_summary.json` receipt matches the manifest run
ids, lengths, and bold sources entry-by-entry) and `04_04` (EXPLORATORY
background FC from bilateral design regression with NaN-propagating
correlations, verifying both surface and cortex receipts; not wired into
`05_01`) consume those task roots. Each run's first-level design contains
task EVs, 12 motion regressors, an intercept, and the nonconstant DCT-II basis
from SPM's high-pass construction with a fixed 128-second cutoff period.
No separate linear trend is added. Joint OLS with these DCT nuisance columns
is equivalent to applying the same high-pass projection to the data and all
model columns before OLS; the solver does not perform temporal prewhitening.
At TR=0.72 s, emotion, WM, and relational runs use 1, 4, and 2 DCT columns,
respectively. `04_03`/`04_04 --tr` controls both HRF sampling and the DCT count;
their existing output summaries record `drift_model`, `high_pass_period_s`,
and `tr`. The drift construction follows
[SPM's filter](https://github.com/spm/spm12/blob/main/spm_filter.m) and
[DCT basis](https://github.com/spm/spm12/blob/main/spm_dctmtx.m).
To regenerate task results with this model, rerun `04_03` for activation,
`04_04` for exploratory background FC when used, and `05_06`/`06_11` for
downstream group activation and presentation products. Anatomical processing
and surface sampling do not need to be rerun for this drift change.

Task hippocampal and cortical masks require finite, nonconstant data in
**every declared run**. With a numerical tSNR threshold, they also use the
runwise residual tSNR gate (`runwise_baseline_over_task_residual_sd`);
`--tsnr-threshold none` applies signal validity only. There is no concatenated
total-SD ceiling and no `10000/std >= 25` rejection for task data. Motion,
registration and model-fit QC remain separate. A constant run remains invalid
even when a baseline jump makes the concatenated series variable. The existing Dice and
maximum invalid-vertex-fraction checks still apply.

The existing `tsnr_gate/` mask and summary paths are retained for downstream
path compatibility. Task summaries record the selected threshold, the residual tSNR
definition, policy and task. Task mode writes per-run `*_run_tsnr.npy` and
minimum-over-runs `*_tsnr.npy`. `02_01` recomputes requested task
gate/QC/smoothing stages even with resume enabled; `03_02` likewise rebuilds
task references. `04_03`/`04_04` reject old smoothing policy metadata, and
`04_04` also rejects old cortex reference policy metadata.

After approval to replace the exact production paths, rebuild in this order:
`02_01 --task <task> --stage tsnr`, then `--stage qc`, then `--stage smooth`;
`03_02 --task <task>`; `04_03` and `04_04`; `05_06` and `06_11`.
Pass explicit task surface/reference roots, the existing task manifest root,
and the selected subject roster throughout. Set `02_01 --route-root` to the
task root and `--shared-root` to its surface directory. `02_01` rejects task
or non-HCP preparation that uses either HCP rest default before creating
directories or running any stage. For `--stage all` or `--stage group-average`,
it also requires a separate `--template-root`; routine task prep can select
only `raw`, `tsnr`, `qc`, and `smooth` to reuse the existing formal anatomy.
Before writing either a QC report or a pass list to a formal HCP rest
destination, `02_02` requires the final QC-passed subject IDs to match the
exact HCP151 roster below. Protected destinations include the Kong2022
default `qc`/`manifests` directories and the branch-neutral
`qc/vertex_qc_t1only/qc` and `qc/vertex_qc_t1only/manifests` directories.
Default Kong2022 QC destinations also require the default rest surface
inputs. Each task uses its complete roster and explicitly selects its own
`--out-dir` and `--manifest-dir`. Its `latest` list contains the complete
QC-passed task cohort; the HCP rest cohort authority is the explicit
151-subject file below.

The formal average at `kong2022/templates/hipp_group_average_t1only_151`
accepts only the exact roster in
`outputs_hcp/inputs/qc/vertex_qc_t1only/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_151.txt`,
even if an existing average has missing or incorrect metadata. Other analyses
use their own explicit `--template-root`. Raw surface samples
can be reused. Do not feed old masked/smoothed arrays into a new gate. Rest
branches retain their existing signal-validity formula.

Task group analysis (`05_06`) requires the complete analysis roster through
`--subjects-manifest` or `--subjects`. The default contrast set contains all
maintained contrasts for the selected task. Missing required contrasts or
condition betas for any selected subject cause an error.
See [the complete task workflow](../../hcp_3t_task_fmri_hipp_memory_emotion_plan.md)
for the current EMOTION and WM rosters, paths, methods, and commands.

Task group stages `05_06` (group mean/t/d/beta maps plus subject and group
cluster tables across hcp151_atlas/template_matching zones
with native-area weighting, BH-FDR correction; unified
beta/contrast roster per combo; explicit rosters fail closed on missing files
instead of silently shrinking; subject-zone entries join statistics only above
`--min-valid-area-fraction` of zone tissue area (default 0.10), gated-out
entries reported in `coverage_exclusions.tsv`) and `06_11`
(GIFTI overlays plus profile/bars PNGs with FDR markers
and black paired-subregion brackets with range-scaled end caps; light bar
fills with dark dots and single red error bars; significance brackets use hierarchical
span sorting, global axis interval stacking, and generous dynamic headroom so overlapping
subregion comparisons never collide and asterisk markers never overflow the top frame;
profile curves use physiological decompositions
— category contrasts compare target category mean vs mean of remaining 3 categories,
interactions compare target category load effect Δ (2bk-0bk) vs remaining categories'
load effect Δ (2bk-0bk), and load contrasts compare 2-back mean vs 0-back mean —
so their difference algebraically reconstructs the contrast exactly on complete support;
for WM, renders both standalone cognitive load figure `task_wm_load.png` and
a modular 2x4 card layout `task_wm_categories.png` grouping Faces, Places,
Tools, and Body across specialization and interaction (with profile and cluster Y-axes
unified row-wise for direct horizontal visual comparison); a non-finite
required condition propagates NaN without reweighting. Cluster bars show each
region's effective n. Count labels sit above the plotting area;
the figure-wide N is explicitly the cohort size.
Missing `group_p_fdr`
fails closed instead of plotting raw p as FDR). `05_06` requires the complete
analysis roster through `--subjects` or `--subjects-manifest`; `06_11` uses
the resulting group products. Outputs use the stable
`work/task_activation/task-<task>/group/` and
`present/task_activation/task-<task>/` paths. The group manifest records the
analysis roster and consumed inputs.

For task bin profiles, `05_06` first averages finite vertices within each
subject's bin, then averages those subject values for all three profile
curves. It tests each contrast's subject-level bin means against zero with a
two-sided one-sample t-test; vertices are not independent observations.
`group_bin_summary.tsv` records bin 1–20, effective subject n, mean, t, p,
and BH-FDR q. A single correction family includes all selected 2mm task
contrasts, hemispheres, and bins in the group build (40 tests for the full
Emotion build; 520 for WM). Condition-beta rows supply colored curves and
are excluded from this correction family. Cluster-bar and paired-region
correction families remain separate.
`06_11` reads the exact bin table declared in `group_build_manifest.json`
and draws a narrow grayscale significance ribbon along the bottom of each
profile panel. White denotes q ≥ .05 or an untested bin; progressively darker
gray denotes q < .05, .01, .001, and .0001. Adjacent bins at the same level
share a continuous segment, with one shared symbol legend per figure
(`ns`, `+`, `*`, `**`, `***`) and no markers on the curves. The legend uses
the same thresholds as bar and bracket symbols.
Missing or duplicate bin rows fail closed; bins with insufficient subjects
are unmarked.
After updating this statistic, rerun `05_06` with the same complete roster
and then `06_11`; first-level GLMs do not need to be rerun. Display-only changes
require rerunning `06_11` alone.

Pearson FC in `lib/fc_math.py` is defined only for complete finite,
nonconstant source and target rows with at least two frames. Undefined pairs
remain NaN in pooled and run-level FC, including empty cortical ROIs; downstream
finite-entry operations exclude those missing pairs. This is a numerical
validity requirement and does not select a hippocampal tSNR cutoff.

### Dataset-specific tSNR QC

The [accepted fMRI QC policy](../../docs/hipp_parcellation_network/fmri_qc.md)
is overall mean Power FD <=0.30 mm, then bilateral HippUnfold Dice >=0.70,
then <=20 unsupported functional-GM vertices in each den-512 hemisphere.
Multiple runs use a frame-count-weighted overall FD without an individual-run
veto. Scope is HCP7T rest, HCP3T EMOTION/WM, Stanford 36P, BANDA and PDC visits.
Numerical tSNR exclusion and an additional mean-RMS gate are not part of this
policy. The three-criterion policy and its vertex-mask integration have not
yet been deployed in the core entry points. The operations below describe
existing tSNR machinery and inherited defaults, not the accepted QC selection.

The implementation is in `lib/tsnr.py`, called by hippocampal preparation
`02_01`, cortex extraction (`03_02` / `lib/io_utils.py`), and Iowa adapters.
HCP rest retains the inherited `10000 / SD` metric and requires a finite,
nonconstant complete analysis series. The numerical cutoff remains configurable;
the default 25 is an inherited comparison setting, not the proposed primary
hippocampal exclusion rule.

HCP task computes the original mean divided by task-model residual sample SD
(ddof=1), separately for every declared run, before smoothing or masking. The
model is the same EV + 12 motion + intercept + 128-second DCT drift design
as the task analysis, using the canonical HCP TR=0.72 s. If task tSNR is
enabled with a numerical threshold, rerun the task tSNR gate and its dependent
masking/reference steps when regenerating results with this model.
The current task `--tsnr-threshold none` mode applies finite/nonconstant run
validity and omits tSNR score computation. A future coverage implementation must keep any valid tSNR diagnostics
separate from its exclusion decision. With a
numerical threshold, run means are never pooled before estimating noise, and the gate writes
per-run scores and their minimum, plus masks.

Stanford, JJ_TMS and Iowa7T rest use baseline-preserving pre-XCP samples in the
same geometry/mapping as their analysis. On the explicitly retained original
frames, the numerator is temporal mean and the denominator is sample SD after
intercept and linear trend removal. The trend uses original frame positions.
The final mask intersects all run thresholds and validity of the consumed
analysis series. Centered XCP residual means and the constant 10000 are not used.

The inherited threshold 25 is an operational comparison setting, not an
empirically validated universal threshold for these different metrics. The
existing subject coverage criterion is a separate parameter. Updating code does
not itself replace existing research products or establish a new accepted cohort.

Rest entry points require `--tsnr-input-template` with `{subject}` and `{hemi}`.
The explicit input is an NPZ containing `dataset`, `subject` (without `sub-`),
`hemi`, `n_runs`, `run_0`, `frames_0`, etc. Runs are vertex-by-original-frame
baseline series; frames are sorted, unique original indices retained after dummy
removal and censoring. Cortex inputs additionally carry `vertex_ids`, which must
match the consumed CIFTI brain-model order. Use
`lib/prepare_tsnr_baseline_inputs.py --help` to package explicitly selected baseline
GIFTIs (hipp) or CIFTIs (cortex) and retained-index NPY files. The helper does not
perform registration or fabricate missing baseline inputs.

`02_01` takes hippocampal inputs; `03_02` takes cortex inputs. JJ_TMS's dedicated
matcher also uses `{domain}` (`hipp`/`cortex`). Iowa7T templates additionally
support `{variant}`, `{branch}` and `{run}`. Missing inputs raise an actionable
error. Core `02_01` resume reuses a rest gate only when the requested dataset,
task, cutoff, selected baseline path, and recorded input/output file metadata
match. Changed raw or baseline inputs trigger gate recomputation. Explicit `qc`
and `all` stages rebuild the subject pass list using the requested Dice threshold.
Smoothing consumes the current gate and is recomputed when its inputs or gate
cutoffs change; a separate `--stage smooth` does not require repeating the gate's
cutoff on the command line. Iowa7T replacement still requires `--overwrite`.
Preserve separate 22P/36P retained-frame
selections and MB/ME acquisition inputs. Existing output replacement follows
repository path-approval rules.

See [fMRI QC](../../docs/hipp_parcellation_network/fmri_qc.md) for the accepted
policy, evaluated retention and implementation status, and
[the task workflow](../../hcp_3t_task_fmri_hipp_memory_emotion_plan.md) for the
complete EMOTION and WM input cohorts.

The task renderer `06_11` explicitly selects 2mm rows when a group table
contains both 2mm and 4mm, matching its maintained 2mm figure contract.
4mm numerical group products remain available under the group root.

## Archived comparison path lookup

`config/legacy_wrong_path_mapping.json` declares archived comparison directories
and exact renamed overview files. `resolve_legacy_wrong_path()` first matches
an explicit artifact entry, then a declared directory prefix for descendants
whose relative names are unchanged. Renamed artifacts require their own entries;
filenames are never inferred from filesystem contents. Missing declarations
raise an error. The table describes names, not a disk-existence check.

Retained active cortex labels and group-average templates are excluded from this
archive table. Renaming those assets does not create an archived baseline.
Archive browsing and any asset migration require separate explicit authorization;
unit tests exercise the mapping against a synthetic repository under `tmp/`.

The maintained CtxReg subfield and confidence comparison renderers use the shared
`multidataset_cohorts.resolve_comparison_paths()` registry for Kong2022 and Lynch
inputs, group roots, and overview filenames. Both use the formal `pfm_ctxreg`
namespace and never select the old route directory based on its existence.

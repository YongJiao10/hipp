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
| `04` | `04_00` cortex regression; `04_01` ROI FC/ctxdrop; `04_02` intrinsic FC | Branch-specific signals and FC |
| `05` | `05_01` gradients; `05_02` group atlas; `05_04` confidence; `05_05` weighted comparison | Gradients, labels and assignment products |
| `06` | Numbered renderers and `06_02` label materialization | Figures and GIFTI/CIFTI labels |
| `07` | `07_01` presentation matrix | Overview index |

Task analysis uses `04_03` for first-level activation, optional exploratory
`04_04` background FC, `05_06` group statistics, and `06_11` rendering.
`04_04` is not connected to the gradient pipeline. `06_05` is retired; use
`06_04` for continuous-gradient rendering.

### HippUnfold

`01_01_run_hippunfold_t1w_local.py` accepts either a subject or a manifest,
with exactly one `--dataset hcp|stanford|iowa_jj_tms|banda|pdc` preset:

```bash
/home/yojiao/.conda/envs/py314/bin/python \
  scripts/core_workflow/01_01_run_hippunfold_t1w_local.py \
  1001 --dataset stanford --dry-run
```

The closed preset uses HippUnfold 2.2.0, T1w, density 512, corobl output,
unfolded registration and GPU inference. BANDA/PDC skip N4 on their already
bias-corrected inputs. The runtime/model/CUDA gate runs once per pending batch.
HCP defaults to eight subjects and accepts `--jobs` up to twelve; other datasets
use two, with BANDA serial. The approved HCP launch, CPU affinity, GPU slots and
resource patches are maintained in the [runtime manifest](../../docs/environment/hippo2_reproducibility_manifest.md#batch-resources-and-retained-outputs).

Stage 01 checks with `--fail-on-missing --no-cleanup`, retaining full structural
outputs. Resume requires a current matching `output_check.json` and every
declared output; a cleanup receipt or a pair of meshes is insufficient.
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

HCP7T rest implements overall mean Power FD <=0.30 mm, bilateral Dice >=0.70,
and at most 20 unusable vertices per consumed den-512 hemisphere. Unusable means
no trilinear NIfTI grid support or a nonfinite/constant sampled signal in any
physical run. Coverage uses `*_nifti_support.npy` and
`sub-<subject>_coverage_summary.json` with policy
`bold_nifti_runwise_trilinear_validity`. Use `--tsnr-threshold none`; this gate
does not apply a numerical tSNR cutoff or CIFTI hippocampal-membership mask.
Participant QC and smoothing use the same vertex validity calculation.

The current `latest` QC roster defines the cohort; there is no fixed historical
subject-count requirement. Participant `qc` and `all` stages currently reject
other datasets/tasks before writing. Their `raw`, `tsnr`, `smooth` and
`group-average` stages remain separately available with explicit branch roots.
Task/non-HCP preparation requires its own `--route-root`, `--shared-root`, and
an explicit `--template-root` for group-average preparation.
Rebuild incompatible sampling, coverage and masks explicitly with `--no-resume`
before smoothing; changing source code does not replace production products.

### Existing rest tSNR baseline inputs

Non-HCP rest adapters retain a separate baseline-based tSNR interface.
`--tsnr-input-template` selects an NPZ with `dataset`, `subject` (without
`sub-`), `hemi`, `n_runs`, and pairs `run_0`/`frames_0`, etc. Each run is a
vertex-by-original-frame baseline array; frames are sorted, unique original
indices retained after dummy removal and censoring. Cortex inputs also declare
`vertex_ids` matching the consumed CIFTI order. Package explicitly selected
baseline GIFTIs/CIFTIs with `lib/prepare_tsnr_baseline_inputs.py --help`.

The rest score is the original temporal mean divided by sample SD after
intercept/linear-trend removal on retained frames, evaluated per run.
The mask intersects run thresholds and validity of the analysis signal.
Threshold 25 is an inherited operational comparison setting. The accepted
participant policy is defined separately in [fMRI QC](../../docs/hipp_parcellation_network/fmri_qc.md).
Templates use `{subject}` and `{hemi}`; dataset adapters add `{domain}`,
`{variant}`, `{branch}` or `{run}` where documented. Keep 22P/36P retained frames
and MB/ME inputs distinct.

## Branches and analysis defaults

| Workflow branch | Route | Reference/gradient contract |
|---|---|---|
| `kong2022_pfm` | `pfm` | Individualized Kong400 FC; canonical7 default |
| `kong2022_fixed` | `fixed` | Public-atlas Kong400 FC; canonical7 default |
| `kong2022_pfm_17net`, `kong2022_fixed_17net` | `pfm_17net`, `fixed_17net` | Reuse PFM/Fixed FC and gradient banks; explicit family17, gradient-only |
| `kong2022_pfm_ctxreg` | `pfm_ctxreg` | Cortex-regressed hippocampal signals; upstream PFM references and surface masks |
| `kong2022_pfm_ctxdrop` | `pfm_ctxdrop` | Exclude valid cortex within 20 mm before ROI averaging; gradient-spectral/WTA |
| `lynch2024_20net`, `lynch2024_7net` | Matching atlas slug | Hard-label Lynch references; both reuse the Lynch20 gradient bank |

For HCP cortex controls, the formal route roots are
`outputs_hcp/inputs/kong2022/shared_inputs/<route>/`,
`outputs_hcp/work/hippomaps_gradients/<route>/`,
`outputs_hcp/work/group_atlas/<route>/`, and
`outputs_hcp/present/overviews/<route>/`. Pass PFM references and surface masks
explicitly when a controlled signal root is separate.

- `gradient-spectral` consumes aligned `05_01` gradients, defaults to G1/G2/G3,
  and uses shifted-cosine surface affinity. `05_02` evaluates K=2..8 and renders
  primary K=3 while recording automatic K selection separately.
- Kong2022 cluster assignments default to `fisherz`; `normalized_top20_count`
  is an explicit comparison. Kong ROI-registry `wta-top` uses only
  `normalized_top20_count`. Lynch direct-network WTA uses `fisherz`.
- Kong2022 retains all 400 ordered ROI rows, including NaN rows for empty
  parcels. Pooled FC removes each declared run mean at computation time and
  preserves within-run variance weighting. Undefined correlations remain NaN.
- `05_05` builds a separate `weighted_fisherz` gradient-spectral comparison.
  Select source, reference, surface and template roots for the intended dataset.
- Renderers consume exact manifest-declared assignment and template assets.
  Select `--group-root`/`--out-root`, route and assignment variant explicitly
  for render-only runs. `07_01` defaults to `--routes pfm`; other routes and
  methods require their corresponding built intermediates.

### Dataset and preprocessing variants

For cohort gradients, pass the same `--cohort <name> --variant 36p|22p` to
`05_01`, `05_02` and `06_04`. Gradient root arguments name the stage root above
the route; for example `pfm` plus `--cohort nthc --variant 22p` resolves to
`<gradient-stage-root>/pfm_22p/nthc/<smoothing>/hemi_<hemi>/`.
Select matching upstream roots, subject rosters and group-atlas roots separately.
HCP omits `--cohort`, uses `36p`, and has no native 22P input branch.
Iowa JJ_TMS unversioned subgroup routes are 36P; 22P routes append `_22p`.

Stanford Causal-Connectome 22P uses the following local roots under
`/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/`:

- CIFTI: `DataOrg_PFM_22P/sub-*/func/*_task-rest_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii`.
- BOLD: `DataOrg_PFM_22P/sub-*/func/*_task-rest_space-MNI152NLin6Asym_res-2_desc-denoised_bold.nii.gz`.
- Labels: `ArealMSHBM_Kong2022_22P/sub-*/{lh_labels.txt,rh_labels.txt}`.

Here `surf` means cortical time-series CIFTI. Use `/Shared/jianglab` on Argon.
The seven overview cohorts are NTHC, TEHC, NTS, NTS Any MDD, NTS Pure MDD, TIS,
and TIS PTSD. Select the complete roster for the intended analysis.

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

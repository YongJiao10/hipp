# Iowa HCP500 Full RouteA Individual Workflow Design

## Goal

Add `iowa_hcp500` as a maintained dataset preset that consumes completed
QuNex/HCP Pipelines 5.0.0 sessions from
`/home/yojiao/hpchome/HCPpipeline500`, runs the repository's complete local
RouteA preparation workflow, and produces isolated subject-level
gradient-spectral K=3 and WTA results under `outputs_iowa_hcp500/`.

The first end-to-end smoke subject is `04172026`. The design must also discover
and support future sessions once they satisfy the same completed-session input
contract.

## Scope

The maintained path is:

```text
completed HCPpipeline500 session
  -> HippUnfold
  -> hippocampal surface sampling, tSNR QC, and smoothing
  -> Argon Kong2022 ArealMSHBM RouteA labels
  -> RouteA cortical reference
  -> hippocampal vertex-to-Kong2022 ROI FC
  -> individual gradient-spectral K=3 and WTA
  -> individual overview products and run status
```

This work does not change the Kong2022 ArealMSHBM algorithm, model parameters,
gradient algorithm, WTA algorithm, network taxonomy, or final K.

## Dataset and Output Contract

The dataset key is `iowa_hcp500`. Its repository-scoped roots are:

```text
outputs_iowa_hcp500/
  inputs/iowa_hcp500/
    hippunfold/dense_corobl_batch/
    kong2022/
      manifests/
      shared_inputs/
        surface/
        routea/
      cortex_references/routea_areal_mshbm/
  work/individual_atlas/
  present/overviews/individual/
  smoke_runs/
  reruns/
```

All dataset-aware entrypoints must resolve these roots from
`--dataset iowa_hcp500`. Existing HCP, Iowa, and Stanford defaults remain
unchanged.

Large HCPpipeline500 products are consumed in place through explicit path
templates. The workflow must not copy the HCP output tree or create a broad
compatibility symlink tree.

## Completed-Session Discovery

Discovery scans only the immediate session directories under:

```text
/home/yojiao/hpchome/HCPpipeline500/work/qunex_iowa7t_hcppipeline500/sessions/
```

A session `<id>` is eligible only when all required inputs exist and are
non-empty under:

```text
sessions/<id>/hcp/<id>/
```

Required inputs are:

```text
T1w/T1w_acpc_dc_restore.nii.gz
T1w/T2w_acpc_dc_restore.nii.gz
MNINonLinear/Results/REST/REST_Atlas_MSMAll_hp0_clean_rclean.dtseries.nii
```

Discovery writes:

- a machine-readable TSV containing each session and its input status;
- a smoke manifest containing only `04172026`, when eligible;
- a full manifest containing all eligible sessions in deterministic order.

Incomplete sessions remain visible in the status TSV with explicit reasons but
are excluded from runnable manifests.

## Local Workflow Stages

### 1. Preflight and manifests

A dataset-specific orchestration entrypoint resolves all paths, runs discovery,
and prints the exact stage commands. Dry-run is the default for orchestration
that would launch substantial work. Each downstream stage consumes an explicit
subject manifest rather than rescanning independently.

### 2. HippUnfold

HippUnfold input preparation uses each session's ACPC-restored T1w and T2w
volumes. Direct runs must pass the repository `hippo2` gate:

1. activate `/opt/miniconda3/envs/hippo2`;
2. verify the resolved `hippunfold` executable;
3. verify CLI version `2.0.0`;
4. verify native `--output_density 512` support.

The consumed `space-corobl_den-512` surfaces and recorded `density_evidence`
are authoritative for density. Command arguments alone are not accepted as
density proof.

### 3. Surface preparation and QC

The core surface workflow consumes the completed HCP REST dtseries via an
explicit path template. It writes raw sampled hippocampal timeseries, tSNR
masks, 2 mm and 4 mm smoothed timeseries, QC summaries, and a QC-pass subject
manifest under the `iowa_hcp500` output tree.

The workflow uses existing sampling and QC algorithms without dataset-specific
branches in computation code.

### 4. Argon RouteA labels

The Argon ArealMSHBM wrappers gain an `iowa_hcp500` preset with HCP-pipeline
session discovery and the completed REST dtseries template. Each eligible
subject supplies its final REST dtseries directly to MATLAB as one session,
with a matching all-ones censor vector.

The local agent does not submit qsub jobs. It prepares and documents:

1. launcher dry-run;
2. compute-node task preflight;
3. one-subject real smoke for `04172026`;
4. full eligible-session submission, enabled only after smoke verification.

The user manually synchronizes the Argon scripts and manifest, executes these
commands, checks scheduler/log status, and copies the resulting
`lh_labels.txt` and `rh_labels.txt` files back to:

```text
outputs_iowa_hcp500/inputs/iowa_hcp500/kong2022/
  cortex_references/routea_areal_mshbm/sub-<id>/
```

Before local downstream work, label validation checks both hemispheres,
expected fsLR 32k vector length, integer-like values, and non-empty label
content.

### 5. RouteA reference and FC

The existing RouteA reference builder consumes:

- the dataset's final REST dtseries;
- the returned flat RouteA label files;
- the dataset-scoped shared surface inputs.

The existing ROI FC materializer then writes the per-subject, per-smoothing,
per-hemisphere `hipp_vertex_to_network_fc.npy` inputs required by the
individual workflow.

### 6. Individual products

The individual workflow gains a dataset-aware preset. With
`--dataset iowa_hcp500`, computation, monitoring, adapter intermediates, and
overview output roots all resolve inside `outputs_iowa_hcp500/`.

The maintained methods remain:

- subject-specific Fisher-z ROI FC gradients G1-G3 followed by
  gradient-spectral clustering at fixed K=3;
- subject-specific top-positive-ROI WTA network voting.

All combo manifests record the dataset key and resolved input/output paths.

## Orchestration and Resume Behavior

A thin dataset launcher may sequence or print existing core entrypoints, but it
must not duplicate scientific computation. It passes explicit dataset roots,
templates, and manifests to each stage.

Every stage supports preflight and resume semantics:

- completed valid outputs are skipped;
- missing required inputs stop the stage;
- failed stages do not create success manifests;
- downstream stages do not run when their upstream contract is incomplete.

The smoke workflow is fixed to `04172026`. Full execution uses the current
eligible-session manifest only after smoke verification.

## Error Handling

Preflight distinguishes at least:

- incomplete HCPpipeline500 session;
- missing T1w;
- missing T2w;
- missing final REST dtseries;
- missing or invalid HippUnfold outputs;
- failed surface/QC contract;
- missing or invalid RouteA labels;
- missing RouteA reference;
- missing ROI FC;
- failed individual combo or overview.

Errors include the exact subject and path. Dataset roots must never silently
fall back to HCP defaults when `iowa_hcp500` is selected.

## Concurrency

Local concurrency remains at or below the repository hard cap of 10 total
jobs/tasks/processes. Existing stricter Workbench and individual-workflow
limits remain in force.

Argon queue throttles follow the established ArealMSHBM wrapper policy. The
local implementation only constructs and tests commands; the user controls
actual submission.

## Testing Strategy

Implementation follows test-first development. Tests cover:

- `iowa_hcp500` output-layout resolution and isolation;
- completed-session discovery and deterministic manifests;
- rejection reasons for incomplete sessions;
- HCPpipeline500 structural and REST path templates;
- dataset-aware defaults for individual compute, monitor, and overview tools;
- no fallback to HCP roots;
- Argon `iowa_hcp500` preset, subject selection, and dry-run command creation;
- shell syntax and CLI help for changed entrypoints;
- smoke preflight for `04172026` without launching recomputation.

Fresh verification must include the relevant Python test suite, shell syntax
checks, CLI help checks, `04172026` discovery/preflight, and Argon launcher
dry-run.

## Documentation

The same change updates:

- core workflow README and dataset/path documentation;
- individual workflow README and maintained individual-flow document;
- multi-branch flow and step-by-step flow documents;
- environment/runtime documentation when execution commands change;
- Argon ArealMSHBM README and status instructions;
- a dedicated `iowa_hcp500` runbook containing local commands, synchronization
  points, manual Argon commands, expected outputs, and recovery steps.

All command examples use explicit interpreter paths.

## Acceptance Criteria

The implementation is accepted when:

1. `04172026` is discovered as eligible from the completed HCPpipeline500
   layout.
2. Dataset-aware commands resolve exclusively under
   `outputs_iowa_hcp500/`.
3. The local smoke can progress stage-by-stage without copying the HCP output
   tree.
4. Argon dry-run and task preflight commands are generated for
   `DATASET=iowa_hcp500`.
5. Returned RouteA labels pass the local validation contract.
6. RouteA reference and ROI FC inputs are produced for the smoke subject.
7. Subject-level gradient-spectral K=3 and WTA outputs and overviews are
   produced for `04172026`.
8. The full manifest automatically includes future completed sessions and
   excludes incomplete sessions with explicit reasons.
9. Tests and documented verification commands pass with no archive access,
   deletion, or unauthorized Argon submission.

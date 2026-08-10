# JJ_TMS HippUnfold full-corobl runbook

This is the only supported local procedure for the fixed 70-subject JJ_TMS
HippUnfold run. It is intentionally prescriptive: do not reconstruct the
cohort, targets, model handling, or acceptance criteria from existing outputs.

## Validated design

The standard wrapper copies the installed HippUnfold workflow and resources
into retained per-subject runtime scratch, produces a local-model-only overlay,
binds `get_model_tar()` explicitly to the canonical model tar, retains
`unpack_nnunet_model`, and records `local_model_overlay.json`. Production runs
the complete `participant/all` DAG with that validated Snakefile. A unique
fresh-output preflight proves the exact native-mesh, corobl-resampling, and
surface-spec branches before any worker can start. Existing outputs never
define the roster, expected files, target graph, or completion status.

## Fixed contract

- Subject roster: `manifests/jj_tms_hippunfold_subjects.txt`, exactly 70 unique
  IDs. Never derive the roster from raw-data counts, derivative counts, or
  existing output directories.
- Input root:
  `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS/raw`
- Output root:
  `outputs_iowa_jj_tms/inputs/iowa_jj_tms/hippunfold/dense_corobl_batch`
- Standard entry: `scripts/core_workflow/01_01_run_hippunfold_t1w_local.py`
- Model:
  `/home/yojiao/hpchome/hipp/runtime/hippunfold_asset_store_shared/hippunfold_assets/model/trained_model.3d_fullres.Task101_hcp1200_T1w.nnUNetTrainerV2.model_best.tar`
- Runtime: validated `hippo2` HippUnfold CLI 2.0.0 with native density 512.
- Output settings: `output_spaces: corobl`, HippUnfold density 512 for the
  hippocampus and dentate, both hemispheres. HippUnfold's density names refer
  to the hippocampus mesh; dentate has one quarter as many vertices (128).
- Concurrency: exactly 4 subject processes.
- Completion target: the complete default `participant/all` target through the standard entry.

## Mandatory preflight

Run repository Python commands only with the explicit py314 interpreter.

```bash
/home/yojiao/.conda/envs/py314/bin/python -c 'import sys; print(sys.executable)'
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/run_jj_tms_hippunfold_batch.py --preflight-only
```

The second command must report `status: pass`, `subject_count: 70`, no missing
T1w inputs, the canonical model tar, and a nonempty canonical unpacked model
directory. This prevents four workers from racing to perform first extraction.

Next run the representative DAG preflight:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/preflight_jj_tms_hippunfold_dag.py --subject 1001
```

This command creates a unique probe output under `/tmp`, retains it, and never
executes jobs. A stale probe output is rejected instead of being accepted as a
no-op. Do not proceed unless its JSON reports all of the following:

```text
status: pass
rule_counts.gen_native_mesh: 4
rule_counts.resample_native_surf_to_atlas_density: 12
rule_counts.create_spec_file: 4
```

The wrapper itself runs the full `hippo2` gate before the DAG is constructed:
the executable must be the repository-local `hippunfold`, its CLI version must
equal `2.0.0`, and help for `--output_density 512` must succeed.

## Production command

Only after both preflights pass, launch with:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/run_jj_tms_hippunfold_batch.py --concurrency 4
```

The production launcher repeats the fresh representative DAG preflight internally
and records its JSON in both run manifests. A failed, malformed, or incomplete DAG
returns before any subject executor is created; the manual preflight cannot be
bypassed by invoking the production command directly. Any `--subjects-file`
argument must equal the normalized IDs and order of the
canonical fixed roster.

The launcher calls the standard core entry for every manifest subject, keeps
existing output files, allows Snakemake to compute missing full-DAG
dependencies, limits concurrency to four, waits for all subject processes, and
writes run logs/manifests below:

```text
outputs_iowa_jj_tms/inputs/iowa_jj_tms/hippunfold/dense_corobl_batch/
  _batch_logs/jj_tms_hippunfold_full_corobl/<run_id>/
```

Do not report completion from process exit alone. The launcher returns nonzero
unless strict validation succeeds for all 70 subjects.

## Independent acceptance

```bash
RUN_STARTED_AT=...  # use started_at from the selected run_manifest.json
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/check_jj_tms_hippunfold.py --run-started-at "$RUN_STARTED_AT"
```

Pass the selected run's numeric `started_at` value so historical Snakemake
metadata cannot invalidate an otherwise clean current run.
`status: complete` is valid only at 70/70. Each subject must have a successful
current `all` log without failure markers; all 12 L/R hippocampus/dentate
corobl-den-512 inner, outer, and midthickness surfaces must be nonempty and
parseable. Their consumed meshes must contain 512 vertices for `hipp` and 128
for `dentate`; the `den-512` filename/spec label refers to the hippocampus
density. All seven density-512 spec files
must reference corobl-den-512 assets. `output_spaces` must be exactly `corobl`,
and the recorded overlay Snakefile/manifest must pass canonical model, rule,
and SHA-256 verification. Downstream consumers must use
`space-corobl_den-512`, never `space-unfold_den-512`. When downstream artifacts
are part of the same handoff, pass their explicit root with `--downstream-root`.

## Stop conditions

Stop without starting production when any of these occurs:

| Evidence | Meaning |
|---|---|
| Local-model overlay validation fails | The canonical production workflow is not active. |
| Native/corobl/spec rule counts differ from 4/12/4 in a fresh probe | The complete target graph was not generated. |
| `Nothing to be done` is the only DAG evidence | The probe reused existing output and is invalid. |
| The roster is not exactly the fixed 70-ID manifest | Cohort scope is ambiguous. |
| The overlay helper rejects an upstream rule layout | HippUnfold changed; inspect and update the overlay/test contract before running. |

Do not delete any smoke, runtime, cache, or output directory while following
this runbook unless the user confirms the exact deletion target list in the
current session.

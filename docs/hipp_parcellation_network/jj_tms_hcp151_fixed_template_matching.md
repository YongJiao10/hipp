# JJ_TMS × HCP151 Fixed-Template Matching

## Scope

This workflow builds one immutable HCP151 RouteA template and applies it to the
fixed 70-subject JJ_TMS roster. It uses only XCP-D `task-rest` denoised runs.
Runs are ordered by numeric run ID: `sub-1016` has the expected two runs and
the other 69 subjects have three. The HCP template is shared by reference and
is not copied into JJ_TMS outputs.

All repository Python commands use:

```bash
/home/yojiao/.conda/envs/py314/bin/python
```

The implementation has no new dependency or runtime-stack requirement.

## HCP151 Template Contract

Build the template from the formal HCP RouteA group labels/support and the
matching full-cohort group mean Fisher-z FC bank. Do not substitute
`cluster_roi_profile.npy`; that artifact has a different aggregation
definition.

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/individual_workflow/build_hcp151_fixed_template.py \
  --mode preflight \
  --group-root /home/yojiao/hpchome/hipp/outputs_hcp/work/group_atlas/routea_full_k3_fisherz \
  --gradient-root /home/yojiao/hpchome/hipp/outputs_hcp/work/hippomaps_gradients/routea \
  --out-root /home/yojiao/hpchome/hipp/outputs_hcp/templates/hcp151_routea_full_k3_fisherz

/home/yojiao/.conda/envs/py314/bin/python scripts/individual_workflow/build_hcp151_fixed_template.py \
  --mode build \
  --group-root /home/yojiao/hpchome/hipp/outputs_hcp/work/group_atlas/routea_full_k3_fisherz \
  --gradient-root /home/yojiao/hpchome/hipp/outputs_hcp/work/hippomaps_gradients/routea \
  --out-root /home/yojiao/hpchome/hipp/outputs_hcp/templates/hcp151_routea_full_k3_fisherz

/home/yojiao/.conda/envs/py314/bin/python scripts/individual_workflow/build_hcp151_fixed_template.py \
  --mode validate \
  --out-root /home/yojiao/hpchome/hipp/outputs_hcp/templates/hcp151_routea_full_k3_fisherz
```

The builder requires the same 151 unique normalized subject IDs across all four
combinations, consumed `den-512` surface evidence whose GIFTI pointset is
actually counted as 512 vertices, ordered AP labels `1..3`, a fixed ordered
Kong2022 registry `1..400`, and `float32[512,400]`
`group_mean_fisherz_fc_full.npy`. The flat output directory contains four sets
of:

```text
hcp151_<2mm|4mm>_hemi-<L|R>_labels.npy       int32[512]
hcp151_<2mm|4mm>_hemi-<L|R>_support.npy      uint8[512]
hcp151_<2mm|4mm>_hemi-<L|R>_prototypes.npy   float32[3,400]
```

`template_manifest.json` records the 151-subject, den-512, K=3, 400-ROI
contract plus SHA-256 fingerprints for every source and output. Prototype row
`k` is the finite feature-wise mean of
`group_mean_fisherz_fc_full[group_support_mask & (group_labels_final == k)]`.

## JJ_TMS Method

For each subject and numeric rest run, Workbench samples the MNI XCP-D volume
onto the subject's actually consumed HippUnfold corobl `den-512` midthickness
surface. Each run-level GIFTI is retained. The workflow concatenates sampled
surface arrays along time and does not create a duplicate concatenated NIfTI.
The volume grid/affine, volume-versus-run-dtseries time points, and summed
run-versus-91k-concat time points must agree before computation.

The subject's 32k ArealMSHBM labels must have exactly 32,492 entries per
hemisphere and use fixed global Kong2022 IDs (`1..200` left, `201..400`
right, with `0` allowed). Cortex vertices use the same `10000/std >= 25` gate
as the RouteA reference construction, then fixed registry order `1..400` is
used to extract ROI time series. Missing ROIs remain all-NaN columns.

Hippocampal vertices use `10000/std >= 25`, followed by 2 mm and 4 mm FWHM
surface smoothing within the valid-vertex ROI. Each valid vertex is correlated
with the 400 ROI time series, Fisher-z transformed, and matched to the
corresponding HCP prototype using Pearson correlation over jointly finite ROI
features. Outputs use:

- `1 = Anterior/VAN`
- `2 = Body/SMN`
- `3 = Posterior/DMN`

Each combination retains Pearson FC, Fisher-z FC, three prototype
correlations, top1-minus-top2 margin, integer labels, label GIFTI, a combo
manifest, source/output SHA-256 fingerprints, and consumed-surface density
evidence.

## CLI and Output Contract

The adapter CLI is fixed to `--mode preflight|run|validate`. It supports
`--subjects`, the fixed roster manifest, `--run-id`, `--jobs` (default 2,
maximum 10), `--resume`, and explicit path overrides. It intentionally has no
`--overwrite` option. Workbench, BLAS, NumExpr, and ITK nested-thread
environment variables are pinned to one before numerical-library import, so
`--jobs` is also the total worker/thread concurrency bound.

Canonical outputs are:

```text
outputs_iowa_jj_tms/work/hcp151_fixed_template_matching/
  subjects/sub-<id>/
  runs/<run_id>/
```

Publishing is exclusive. New run and validation manifests carry
`artifact_contract_schema_version=2`; new complete run items must contain both
the exact subject-manifest fingerprint and all four combo-manifest fingerprints.
Missing schema/anchors are not inferred as legacy: only the two explicitly
published pre-schema run IDs (`jj_tms_hcp151_smoke_20260809` and
`jj_tms_hcp151_full_20260809`) receive legacy compatibility, with the same
fresh dependency closure. `--resume` reuses a completed subject only when its
independently constructed canonical output set, four unique smoothing/hemi
combinations, paths, fingerprints, shapes, dtypes, parameters, and per-valid-
vertex score/margin contracts all still match. A completed run manifest is
immutable and returned without rewrite after its roster, paths, template, ROI
registry, and parameters validate. New run items anchor the exact subject
manifest and all four combo-manifest fingerprints; legacy run manifests are
accepted only after the current strict dependency closure passes. Every
functional GIFTI is validated as float32 [512,T], ROI-shape GIFTI as float32
[512], and label GIFTI as int32 [512], independently of self-declared
metadata. An interrupted/failed run partial is copied
verbatim to the next versioned run-attempt history before the active partial is
reused; failed runs keep a `status=failed` partial and never publish a final.

A failed subject may restart only when the active partial has exact inputs and
parameters, `retained_outputs` is empty, and the directory otherwise contains
only prior `subject_manifest.attempt-NNN.failed.json` histories. An empty
`in_progress` partial is first retained verbatim as an
`subject_manifest.attempt-NNN.interrupted.json` history. A complete `ok` subject
partial is independently validated and exclusively promoted; it is never
trusted from its self-declared status. The interrupted history is copied while
the active partial remains available, and prior failed or interrupted histories
are accepted. Extra files, retained intermediates,
fingerprint drift, or parameter drift fail closed. Complete run partials are
promoted only after every subject manifest and four-combination dependency has
been revalidated. Validation writes `validation_items.partial.tsv` and
`validation_manifest.partial.json`, then exclusively publishes each final; a
complete validation partial with a final table is promoted only after the
table, run, subject, and combination closure is revalidated. `--resume`
fingerprint-reuses completed validation and reruns the same closure checks,
including the canonical smoke gate. Fresh validation rows must exactly match
the recorded anchored items; malformed/truncated JSON partials are retained as
versioned interrupted history before recomputation. Interrupted tables/manifests
are moved to versioned attempt history without deletion only after resume and
final-coexistence checks; `resume=false` never changes retained state. The smoke
gate always requires dependency roots/parameters and runs its full closure.

## Fixed Launch Sequence

Smoke preflight:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/individual_workflow/jj_tms_hcp151_fixed_template_matching.py \
  --mode preflight --subjects 1001 1002 1003
```

Smoke run and validation:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/individual_workflow/jj_tms_hcp151_fixed_template_matching.py \
  --mode run --subjects 1001 1002 1003 --jobs 2 \
  --run-id jj_tms_hcp151_smoke_20260809

/home/yojiao/.conda/envs/py314/bin/python scripts/individual_workflow/jj_tms_hcp151_fixed_template_matching.py \
  --mode validate --subjects 1001 1002 1003 \
  --run-id jj_tms_hcp151_smoke_20260809
```

The full-run code enforces this hard gate before its own preflight: the fixed
`jj_tms_hcp151_smoke_20260809` validation must be `status=complete`, contain
1001/1002/1003 and exactly 12 unique combinations with all labels represented,
and retain matching template, ROI-registry, run-manifest, and validation-table
fingerprints. Only then launch the fixed full roster:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/individual_workflow/jj_tms_hcp151_fixed_template_matching.py \
  --mode run --jobs 2 --resume \
  --subjects-manifest manifests/jj_tms_hcp151_fixed_template_subjects.txt \
  --run-id jj_tms_hcp151_full_20260809

/home/yojiao/.conda/envs/py314/bin/python scripts/individual_workflow/jj_tms_hcp151_fixed_template_matching.py \
  --mode validate \
  --subjects-manifest manifests/jj_tms_hcp151_fixed_template_subjects.txt \
  --run-id jj_tms_hcp151_full_20260809
```

Validation rechecks every input/output fingerprint and array contract, requires
all valid vertices to have nonzero labels, three finite in-range prototype
correlations and a finite in-range margin, and all three labels to occur in each
combination. It verifies the consumed GIFTI pointset is actually 512 vertices
and writes label counts, margin summaries, and label SHA-256 values. A full run
is complete only when the manifest contains the exact fixed roster, 70/70 subjects, and 280/280
validated combinations. Any failure remains `failed` rather than being
reported as complete.

## Fixed-template Static Subject Overviews

After a complete fixed-roster validation, render one presentation PNG per
subject with the shared locked Workbench layout:

```bash
scripts/individual_workflow/run_jj_tms_hcp151_fixed_template_overviews.sh \
  --jobs 2
```

The launcher uses
`/home/yojiao/.conda/envs/py314/bin/python` and the repository Workbench
executable. It reads the completed mapping GIFTIs and consumed
`den-512` surfaces, never rewrites those inputs, and validates their subject
and combination manifest fingerprints before rendering. The display layer
retains the established HCP style: canonical network colors
`1=Anterior/VAN`, `2=Body/SMN`, `3=Posterior/DMN`, left/right hemispheres, and
the fixed two-row order `2mm` above `4mm`.

The canonical output is:

```text
outputs_iowa_jj_tms/present/overviews/hcp151_fixed_template_matching/
  sub-<id>_hcp151_fixed_template_matching_overview.png
  fixed_template_overview_status.tsv
  fixed_template_overview_manifest.json
```

The renderer validates the complete run/validation manifest when present (or
the 70 completed subject manifests otherwise), records source, scene, visual
contract, and PNG fingerprints, and caps renderer workers at five so the
worker plus one Workbench child per worker stays within ten processes.
Validation-backed rendering is production-only full scope: exactly 70 subjects,
280 unique TSV/items, complete run roots/parameters, and matcher dependency
closure are required. Subsets require --allow-subset and cannot use a
production validation manifest.
It refuses to overwrite an existing PNG or provenance file. `--resume` only
reuses an item after revalidating every source fingerprint and output PNG; a
complete manifest/status pair is byte-stable on reuse. Each subject also gets
an atomic item provenance record beside its PNG, and an in-progress aggregate
partial is published before workers start so orphan PNGs can be recovered only
after strict item validation. Failed aggregate
provenance is retained as numbered `.attempt-NNN.failed.*` history before a
resume retry, without deletion.

When a completed validation manifest is available, the renderer invokes the
matcher's full subject validator against its dependency roots. Validation
items must be the exact unique subject x smoothing x hemisphere closure and
their label SHA must match the matcher output. Aggregate status and manifest
files use partial staging and atomic publication, and completed reuse checks
the status TSV fingerprint. A subset requires explicit --allow-subset and is
reported with scope=subset rather than production ok.

# JJ_TMS × HCP Fixed-Template Matching

## Scope

This workflow builds one immutable HCP RouteA template and applies it to the
fixed JJ_TMS roster in `manifests/jj_tms_subjects.txt`. It uses only XCP-D `task-rest` denoised runs.
Runs are ordered by numeric run ID: `sub-1016` has the expected two runs and
the other subjects have three. The HCP template is shared by reference and
is not copied into JJ_TMS outputs.

## Clinical diagnosis and Iowa3T cohort provenance

The source of truth for JJ_TMS background diagnosis, scan/data availability,
processing status, and QC is the external workbook:

```text
/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS/subject_infomation.xlsx
```

Use worksheet `Sheet1`. Subject IDs are in column `A`; background labels are
in columns `C:E` (`MDD`, `PTSD`, `GAD`); data-upload fields are `G:H`;
data-processing fields are `I:J`; and `QC` is column `K`. The
imaging roster is the rows with `QC == "Yes"`. The analysis rosters in
`manifests/jj_tms_*_subjects.txt` are frozen snapshots of that rule, so a
workbook update never changes an analysis silently. Roster sizes are results of
the rules below, never inputs; `derive_jj_tms_groups.py` (below) reports any
difference between the workbook and the frozen manifests.

`outputs_iowa_jj_tms/TMSfMRI_DataSharing.xlsx` is a separate target-coordinate
and data-sharing workbook used by the stimulation-site mapping workflow. It is
not the source for the imaging roster or the Iowa3T diagnostic
grouping, and its patient list must not replace the roster above.

### Iowa3T group derivation

Groups are derived from workbook columns `A`, `C:E` and `K` as follows. The
scripted form is `scripts/jj_tms/derive_jj_tms_groups.py`.

| Cell | Rule |
| --- | --- |
| Roster | `QC == "Yes"`; subject IDs `JJ_TMS_1xxx` are healthy controls (`hc`), `JJ_TMS_3xxx` are patients |
| Current MDD | `MDD` cell is `No` (absent) or `Yes (...)` whose list contains `Current`; a list without `Current` is past-only MDD |
| PTSD | `PTSD == "Yes (Current)"` |
| GAD | MINI-confirmed only for `GAD == "Yes (Current)"`; `No (Current diagnosis, but no on MINI)` is clinical anxiety without confirmed GAD and does not count |

| Group | Rule |
| --- | --- |
| `patient` | QC-passed patient with current MDD |
| `pure_mdd` | `patient` with no PTSD and no confirmed GAD (current MDD only, or clinical anxiety without MINI-confirmed GAD) |
| `comorbid_mdd` | `patient` with PTSD and/or confirmed GAD (MDD+GAD, MDD+PTSD+GAD, MDD+PTSD) |
| excluded | QC-passed patient without current MDD (for example past-only MDD with current GAD, `JJ_TMS_3003`); assigned to no group |

`pure_mdd` and `comorbid_mdd` partition `patient`. Cell values outside the
documented set stop the script with the row and value named, so a new category
in the workbook forces an explicit rule decision.

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/jj_tms/derive_jj_tms_groups.py \
  --workbook /home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS/subject_infomation.xlsx
```

`--mode check` (default) prints each group's size, lists excluded subjects and
the per-group differences from the frozen manifests, and exits 1 on any
difference. `--mode write --out-dir <new directory>` writes the derived rosters
without overwriting existing files. Adopting a changed roster means replacing the
frozen manifests deliberately and re-running the dependent products. Rows with
an empty `QC` cell (for example `JJ_TMS_3021`, dropped during the Visit 1 scan)
are not in any roster.

The Iowa3T subgroup products use the 36P branch.

The 36P presentation directories are:

```text
outputs_iowa_jj_tms/present/overviews/pfm/pure_mdd/
outputs_iowa_jj_tms/present/overviews/pfm/comorbid_mdd/
```

All repository Python commands use:

```bash
/home/yojiao/.conda/envs/py314/bin/python
```

The implementation has no new dependency or runtime-stack requirement.

## HCP Template Contract

Build the template from the formal HCP RouteA group labels/support and the
matching full-cohort group mean Fisher-z FC bank. Do not substitute
`cluster_roi_profile.npy`; that artifact has a different aggregation
definition.

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/individual_workflow/03_01_build_hcp_fixed_template.py \
  --mode preflight \
  --group-root /home/yojiao/hpchome/hipp/outputs_hcp/work/group_atlas/pfm \
  --gradient-root /home/yojiao/hpchome/hipp/outputs_hcp/work/hippomaps_gradients/pfm \
  --out-root /home/yojiao/hpchome/hipp/outputs_hcp/templates/hcp_routea_full_k3_fisherz

/home/yojiao/.conda/envs/py314/bin/python scripts/individual_workflow/03_01_build_hcp_fixed_template.py \
  --mode build \
  --group-root /home/yojiao/hpchome/hipp/outputs_hcp/work/group_atlas/pfm \
  --gradient-root /home/yojiao/hpchome/hipp/outputs_hcp/work/hippomaps_gradients/pfm \
  --out-root /home/yojiao/hpchome/hipp/outputs_hcp/templates/hcp_routea_full_k3_fisherz

/home/yojiao/.conda/envs/py314/bin/python scripts/individual_workflow/03_01_build_hcp_fixed_template.py \
  --mode validate \
  --out-root /home/yojiao/hpchome/hipp/outputs_hcp/templates/hcp_routea_full_k3_fisherz
```

The builder requires the same unique normalized subject IDs across all four
combinations, consumed `den-512` surface evidence whose GIFTI pointset is
actually counted as 512 vertices, ordered AP labels `1..3`, a fixed ordered
Kong2022 registry `1..400`, and `float32[512,400]`
`group_mean_fisherz_fc_full.npy`. The flat output directory contains four sets
of:

```text
hcp_<2mm|4mm>_hemi-<L|R>_labels.npy       int32[512]
hcp_<2mm|4mm>_hemi-<L|R>_support.npy      uint8[512]
hcp_<2mm|4mm>_hemi-<L|R>_prototypes.npy   float32[3,400]
```

`template_manifest.json` records the subject roster, den-512, K=3, 400-ROI
contract plus SHA-256 fingerprints for every source and output. Prototype row
`k` is the finite feature-wise mean of
`group_mean_fisherz_fc_full[group_support_mask & (group_labels_final == k)]`.

## JJ_TMS CtxReg Template Contract

CtxReg matching is maintained by the separate
`scripts/jj_tms/jj_tms_ctxreg_fixed_template_matching.py` adapter. It requires
`template_manifest.json` to declare exactly
`template_id=hcp_routea_ctxreg`, complete 2mm L/R prototypes, the fixed
400-ROI registry, and source manifests whose gradient route is
`pfm_ctxreg` and whose group resume provenance has
`route_tag=pfm_ctxreg`. A standard RouteA template, a renamed output
directory, or a manifest with stale source fingerprints is rejected.

The adapter consumes exactly:

```text
outputs_iowa_jj_tms/inputs/kong2022/shared_inputs/pfm_ctxreg/
  sub-<id>/fc/kong2022/2mm/hemi_{L,R}/hipp_vertex_to_roi_fc.npy
outputs_iowa_jj_tms/inputs/kong2022/shared_inputs/surface/
  sub-<id>/surface/validity/sub-<id>_hemi-{L,R}_valid_mask.npy
outputs_iowa_jj_tms/inputs/hippunfold/dense_corobl_batch/
  sub-<id>/hippunfold/sub-<id>/surf/
  sub-<id>_hemi-{L,R}_space-corobl_den-512_label-hipp_midthickness.surf.gii
```

The default subject source is the existing HC manifest followed by the
Patient manifest. The matcher performs a complete preflight for all
selected subjects before creating any `subjects/sub-<id>/` output directory.
Its canonical output root is:

```text
outputs_iowa_jj_tms/work/hcp_fixed_template_matching/pfm_ctxreg/
  subjects/sub-<id>/matching/2mm/hemi-{L,R}/
  runs/<run_id>/
```

Each combination records the exact FC, mask, consumed surface, template
prototype, output-array/GIFTI fingerprints, valid-vertex summary, and density
evidence. `--resume` is strict: an existing subject manifest is reused only
after current inputs and every declared output independently validate; malformed
or incompatible state is an error and is never silently ignored or overwritten.
The adapter has no `--overwrite` mode.

The CtxReg builder is selected explicitly:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/individual_workflow/03_01_build_hcp_fixed_template.py \
  --mode preflight \
  --template-id hcp_routea_ctxreg \
  --group-root outputs_hcp/work/group_atlas/pfm_ctxreg \
  --gradient-root outputs_hcp/work/hippomaps_gradients/pfm_ctxreg \
  --out-root outputs_hcp/templates/hcp_routea_ctxreg
```

This section documents the contract only; it does not authorize rebuilding or
overwriting an existing formal template or matching output tree.

## Iowa Functional-Organization Branch Identity

The dedicated Iowa functional-organization report uses two fixed source
identities: `standard → pfm` and `ctxreg → pfm_ctxreg`. Standard is the
explicit original RouteA branch used by its producer schema; it is not selected
by filesystem fallback, and its original matching manifests deliberately omit
`signal_branch`. CtxReg matching must explicitly declare its CtxReg contract. No other
cortex-control branch is a substitute for the standard branch in this report.

## JJ_TMS Method

For each subject and numeric rest run, Workbench samples the MNI XCP-D volume
onto the subject's actually consumed HippUnfold corobl `den-512` midthickness
surface. Each run-level GIFTI is retained. The workflow concatenates sampled
surface arrays along time and does not create a duplicate concatenated NIfTI.
The volume grid/affine, volume-versus-run-dtseries time points, and summed
run-versus-91k-concat time points must agree before computation. Use the declared
physical run IDs for each subject. Cortex ROI time series concatenation is performed directly in memory when
`--concat-dtseries-root none` is specified.

The subject's 32k ArealMSHBM labels must have exactly 32,492 entries per
hemisphere and use fixed global Kong2022 IDs (`1..200` left, `201..400`
right, with `0` allowed). Cortex vertices with finite signal are retained,
then fixed registry order `1..400` is
used to extract ROI time series. Missing ROIs remain all-NaN columns.

Hippocampal vertices require finite, nonconstant concatenated signals,
followed by 2 mm and 4 mm FWHM
surface smoothing within the valid-vertex ROI. Each valid vertex is correlated
with the 400 ROI time series, Fisher-z transformed, and matched to the
corresponding HCP prototype using Pearson correlation over jointly finite ROI
features. Outputs use:

- `1 = Anterior/VAN`
- `2 = Middle/SMN`
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
outputs_iowa_jj_tms/work/hcp_fixed_template_matching/
  subjects/sub-<id>/
  runs/<run_id>/
```

Publishing is exclusive. New run and validation manifests carry
`artifact_contract_schema_version=2`; new complete run items must contain both
the exact subject-manifest fingerprint and all four combo-manifest fingerprints.
Missing schema/anchors are not inferred as legacy: only the two explicitly
published pre-schema run IDs (`jj_tms_hcp_smoke_20260809` and
`jj_tms_hcp_full_20260809`) receive legacy compatibility, with the same
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
/home/yojiao/.conda/envs/py314/bin/python scripts/jj_tms/jj_tms_hcp_fixed_template_matching.py \
  --mode preflight --subjects 1001 1002 1003
```

Smoke run and validation:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/jj_tms/jj_tms_hcp_fixed_template_matching.py \
  --mode run --subjects 1001 1002 1003 --jobs 2 \
  --run-id jj_tms_hcp_smoke_20260809

/home/yojiao/.conda/envs/py314/bin/python scripts/jj_tms/jj_tms_hcp_fixed_template_matching.py \
  --mode validate --subjects 1001 1002 1003 \
  --run-id jj_tms_hcp_smoke_20260809
```

The full-run code enforces this hard gate before its own preflight: the fixed
`jj_tms_hcp_smoke_20260809` validation must be `status=complete`, contain
1001/1002/1003 and exactly 12 unique combinations with all labels represented,
and retain matching template, ROI-registry, run-manifest, and validation-table
fingerprints. Only then launch the fixed full roster:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/jj_tms/jj_tms_hcp_fixed_template_matching.py \
  --mode run --jobs 2 --resume \
  --subjects-manifest manifests/jj_tms_hcp_fixed_template_subjects.txt \
  --run-id jj_tms_hcp_full_20260809

/home/yojiao/.conda/envs/py314/bin/python scripts/jj_tms/jj_tms_hcp_fixed_template_matching.py \
  --mode validate \
  --subjects-manifest manifests/jj_tms_hcp_fixed_template_subjects.txt \
  --run-id jj_tms_hcp_full_20260809
```

Validation rechecks every input/output fingerprint and array contract, requires
all valid vertices to have nonzero labels, three finite in-range prototype
correlations and a finite in-range margin, and all three labels to occur in each
combination. It verifies the consumed GIFTI pointset is actually 512 vertices
and writes label counts, margin summaries, and label SHA-256 values. A full run
is complete only when the manifest contains the exact fixed roster, every roster subject complete, and all
validated combinations. Any failure remains `failed` rather than being
reported as complete.

## Dominant-Subregion Branch Figures (Iowa 3T + Stanford, dual dataset)

The formal branch renderer
(`scripts/visualize/template_matching_dominant_subregions/render_template_matching_dominant_subregions.py`,
`--signal-branch standard|ctxreg`) produces the two canonical 2mm/36P
figures
`template_matching_dominant_subregions_2mm_36p_standard.png` and
`template_matching_dominant_subregions_2mm_36p_ctxreg.png`.
Each figure has two dataset blocks side by side: left = Iowa 3T (ref HC,
N=67), right = Stanford (ref NTHC, N=51; always standard-branch TM, 7
groups: NTHC/TEHC/NTS/NTS Any MDD/NTS Pure MDD/TIS/TIS PTSD). Rows 0-3 pair
Iowa cohorts with Stanford cohorts; rows 4-6 are Stanford-only.

Both stat columns per block are area % of hippocampus (subregion area as %
of the subject's own hemisphere total midthickness area). The former
vertex-count column is replaced by the age+sex-adjusted column:

- Adj: OLS `y ~ 1 + group + age + female`, uncorrected (no FDR).
- Welch: two-sided Welch, uncorrected (no FDR).

Bars/dots in both columns show raw area % values. Stars:
`+` 0.05<=p<0.10, `*` p<0.05, `**` p<0.01, `***` p<0.001.

Covariate sources: Iowa age/sex from
`outputs_iowa_jj_tms/TMSfMRI_DataSharing.xlsx` (`pt_age`, `pt_sex`);
Stanford age/sex from `data/CC_RC_Master_Database_10.23.18_Data.csv`
(`Intake_Age`, `Gender`; female = 1 iff `Gender == 2`; the single
`Gender == 3` record is coded 0 with N unchanged, documented in figure and
CSV provenance). Missing demographics fail closed. Per-contrast CSVs
(`..._contrasts.csv`) carry both tests plus covariate/sex-coding provenance.

Run on main with:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/visualize/template_matching_dominant_subregions/render_template_matching_dominant_subregions.py \
  --signal-branch all
```

Stanford atlas and heatmap panels use BrainSpace with Stanford group-average
surfaces; their panels are regenerated from the selected inputs.

## Fixed-template Static Subject Overviews

After a complete fixed-roster validation, render one presentation PNG per
subject with the shared BrainSpace hippocampal layout:

```bash
scripts/jj_tms/run_jj_tms_hcp_fixed_template_overviews.sh \
  --jobs 2
```

To redraw the complete declared roster in place with the current renderer, use
`--resume --refresh-visuals`; the refresh updates the subject PNGs and their
existing item and aggregate records.

The launcher uses
`/home/yojiao/.conda/envs/py314/bin/python` environment and the existing BrainSpace/VTK stack. It reads the completed mapping GIFTIs and consumed
`den-512` surfaces, never rewrites those inputs, and validates their subject
and combination manifest fingerprints before rendering. The display layer
retains the established HCP style: canonical network colors
`1=Anterior/VAN`, `2=Middle/SMN`, `3=Posterior/DMN`, left/right hemispheres, and
the fixed two-row order `2mm` above `4mm`. The top banner title is
fixed-template specific (`HCP fixed-template Fisher-z network matching |
sub-<id>`, passed explicitly to the shared stacked-row composer); earlier
PNGs rendered before this fix still show the inherited `HCP RouteA cross-fit
...` banner and are refreshed on the next production re-render.

The canonical output is:

```text
outputs_iowa_jj_tms/present/overviews/hcp_fixed_template_matching/
  sub-<id>_hcp_fixed_template_matching_overview.png
  fixed_template_overview_status.tsv
  fixed_template_overview_manifest.json
```

The renderer validates the complete run/validation manifest when present (or
the 94 completed subject manifests otherwise), records source, scene, visual
contract, and PNG fingerprints, and caps renderer workers at five so the
worker plus one Workbench child per worker stays within ten processes.
Validation-backed rendering is production-only full scope: the complete fixed roster,
unique TSV/items for every roster subject, complete run roots/parameters, and matcher dependency
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

## Iowa Functional-Organization Statistics and Visualization

The independent Iowa analysis and presentation entry point is
[`scripts/visualize/iowa_hipp_functional_organization/README.md`](../../scripts/visualize/iowa_hipp_functional_organization/README.md).
It uses the already validated Kong2022 canonical7, 36P, 2 mm standard and
ctxreg products with explicit manifest selection, never directory-order
fallback. Its fixed Patient, Pure-MDD, and Comorbid-MDD versus
Iowa-HC Welch comparisons are separate from this fixed-template matching
workflow; HCP supplies only templates and gradient references.

For this analysis, **de novo 指每个队列使用自身组平均 FC 所得梯度独立聚类的组
atlas；不是个体 TM，不是 group dominant，也不是逐个体重新聚类。** It is a
cohort-level descriptive atlas and is not an individual TM label or a
participant-level reclustering. The module records the actual `den-512` input
surface evidence, preserves invalid vertices as missing, and stores its
statistics, four PNG theme figures, and offline HTML report only beneath its
own `results/` directory.

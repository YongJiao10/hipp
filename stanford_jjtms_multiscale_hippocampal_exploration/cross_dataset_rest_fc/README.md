# Cross-dataset hippocampal rest-FC inventory

This directory is the fail-closed input boundary for the corrected JJTMS +
Stanford analysis. It covers resting-state Fisher-z functional connectivity
only. The Kember metrics remain a separate endpoint workflow, but the
group-atlas geometry family is registered in the same unified branch plan and
run manifest. Atlas geometry rows are not substituted for subject-level FC
rows.

## Frozen inputs

`config/analysis_config.json` is the only input registry. It names the
repository-local Stanford Kong2022 rest-FC root and QC manifests, and the
explicit JJTMS rest-FC/template-matching root with its per-combination
provenance manifests. The only registered FC estimator is exactly
`fisher_z`; a Pearson or other estimator is returned as `status=blocked` and
is never reported as a usable input. The frozen group counts are:

The subject-level FC registry contains `hcp151_fixed` and
`hcp151_template_matching` for both datasets. The unified analysis registry
also contains `group_atlas_gradient_spectral`, which consumes current
group-average K=3 atlas maps as a separate geometry family. Both FC methods
have explicit FC,
provenance, parcel-registry, and region-label/mask sidecar paths. The current primary_screen tables may retain audit-only placeholder rows from a superseded configuration;
those rows are audit-only and are excluded from the scientific branch inventory.
Each frozen parcel registry declares `fc_column_index_column` and
`fc_column_index_base`, so rows are aligned to the zero-based FC-column order
before aggregation.

| Dataset | Group | n |
| --- | --- | ---: |
| Stanford | NTHC | 51 |
| Stanford | TEHC | 40 |
| Stanford | NTS | 56 |
| Stanford | TIS | 47 |
| JJTMS | healthy | 67 |
| JJTMS | patient | 27 |

Stanford cells are formed only by intersection of the named healthy,
non-trauma, symptomatic, and trauma manifests. JJTMS labels come only from
the named healthy and patient manifests and are checked against the named
94-person all-subject manifest. Duplicate labels, overlapping groups, count
drift, and subjects absent from the all-subject manifest fail closed.

The Stanford group-atlas geometry family is executed by the same
`primary_screen` priority plan. Its rows are written under
`results/primary_screen/group_atlas/`; they are not injected into the
subject-level FC tensor and do not change subject-level FC input coverage.

## Unified branch priority

The active branch registry is the Cartesian product of the two subject-level
FC methods, both datasets, 8/17 networks, 2/4 mm, and the current group-atlas
HC plus seven cohort maps, plain/weighted assignment, 2/4 mm, and L/R. The
scheduler uses a strict barrier:

| Tier | Subject-level FC | Group-atlas geometry |
| --- | --- | --- |
| P0 | 8-network, 2 mm | 2 mm cells |
| P1 | 8-network, 4 mm; 17-network, 2 mm | none |
| P2 | 17-network, 4 mm | 4 mm cells |

Every branch in a higher tier must finish before any lower-tier branch starts.
A failure stops the run and leaves lower tiers blocked. The retired
individual-gradient branch is not in this registry.

## Path and coverage contracts

`src/inventory.py` exposes:

```text
validate_rest_only_path(path, allowed_roots) -> Path
build_rosters(config) -> dict[str, pandas.DataFrame]
inspect_fc_inputs(config, rosters) -> pandas.DataFrame
```

Every configured path is resolved and checked against the explicit allow-list
before `is_file()` or a file read. No glob, recursive walk, or inferred input
path is used. The gate permits legal `JJTMS` and `JJ_TMS` dataset components;
it rejects explicit beta, TMS-response/contrast, con-image, native-mapping,
archive, and traversal components. Missing FC or provenance files are returned
as `status=missing`; a ready dataset without a configured provenance pattern is
returned as `status=blocked`, and a configured-but-missing provenance file is
explicitly listed in `missing_components`. Coverage rows carry
`path_opened=false` because this task proves only explicit path presence; later
feature extraction owns array/header reads. Coverage also records the raw
`fc_estimator` and the `registered_fc_estimator` contract for auditability.

Run the focused tests with the repository interpreter:

```bash
/home/yojiao/.conda/envs/py314/bin/python -c "import sys; print(sys.executable)"
/home/yojiao/.conda/envs/py314/bin/python -m pytest \
  stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/tests/test_inventory.py -q
```

The roster tables are fresh snapshots and carry `attrs['immutable'] = True`;
downstream tasks must treat them as read-only. Each roster preserves
`source_manifest_primary`, `source_manifest_secondary`, `clinical_manifest`,
and `trauma_manifest`; Stanford intersection rows therefore retain both the
clinical and trauma/non-trauma source paths in deterministic order. The FC
inventory retains both 2/4 mm smoothing and L/R rows; bilateral and L−R views
are constructed by `src/features.py`.

## Subject-level tensor contract

`src/features.py` exposes:

```text
build_fc_tensor(dataset, roster, config) -> pandas.DataFrame
build_hemisphere_views(tensor) -> pandas.DataFrame
run_config_smoke(config_or_path, subjects_per_dataset=1) -> pandas.DataFrame
```

Each dataset configuration must add an explicit `parcellations` registry (or
`features.parcellations`). Every entry names a `method`, a region-label path
pattern (unless the FC has exactly three region rows), and one frozen parcel
registry containing `family17` and `canonical8` columns. The registry must
contain exactly 17 and 8 unique network labels; the latter is derived from the
same parcel rows and never inferred from network-name text. Optional
`region_weight_pattern` and `valid_mask_pattern` sidecars provide area or
valid-vertex weights; without a weight sidecar, the adapter uses the count of
valid vertices. Paths are rendered only from the configured FC root and are
validated by the same rest-only gate before reading.

Every usable FC combination must have explicit provenance evidence for
resting-state FC and Fisher-z. Configured sidecars are cross-checked for
subject, hemisphere, smoothing, dataset, method, and estimator identity;
missing or incompatible provenance fails closed before a value row is present.
Without an adapter, one provenance record must independently contain all three
semantic fields: acquisition/task (`task`/`modality` aliases normalized exactly
to rest), FC/connectivity role (`data_role`/`fc_definition`/`fc` aliases), and
Fisher-z scale (`fc_estimator`/`measure`/`fc_transform` aliases). Partial
records are never combined. The shipped legacy manifests use explicit
`provenance_semantics` adapters: each descriptor names `primary` or a numbered
`sidecar:N`, an exact field path, and exact normalized allowed values; the
Stanford `dtseries` adapter extracts every task entity and requires at least
one entity with every extracted entity exactly `rest` (repeated `rest` entities
are allowed). Mixed, unknown, or non-rest task entities fail closed. All
recognized aliases and records are still checked for contradictory or
non-Fisher values.
The bounded smoke is read-only and can be run directly from the shipped config:

```bash
/home/yojiao/.conda/envs/py314/bin/python -c \
  "import sys; sys.path.insert(0, 'stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/src'); import features; print(features.run_config_smoke('stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/config/analysis_config.json').to_string(index=False))"
```

It reads one deterministic subject per dataset by default and materializes no
output files.

The long-form tensor has the columns
`dataset, subject, group, trauma, symptomatic, method, smoothing_mm,
hemisphere, region, network_resolution, network, fc_estimator, fisher_z,
region_weight, valid_vertex_count, network_coverage, status,
failure_reason`. Regions are exactly `anterior`, `body`, and display `tail`
(machine `posterior`). Values remain absolute Fisher-z FC; incompatible
shapes, provenance mismatches, unknown labels, and missing sidecars remain
visible as failed/missing rows and are never zero-filled. The configured
valid-vertex mask is honored before aggregation. A vertex with any nonfinite
parcel FC is excluded from every network component for that subject/side so
all profile components use the same finite vertex set; `valid_vertex_count`
and `region_weight` use that retained set. `network_coverage` is calculated
against the configured pre-finiteness region mask, so subject-specific vertex
loss remains quantitative rather than silently disappearing. A region with no
finite valid vertices remains an explicit missing row.

`build_hemisphere_views` adds `hemisphere_mode` rows for raw `L`, raw `R`,
area/valid-vertex weighted `bilateral`, and paired within-subject `L-R`.
Both bilateral and `L-R` require valid values from both sides; bilateral uses
the recorded `region_weight`, while `L-R` is Fisher-z(L) − Fisher-z(R).
Neither view subtracts group means or fabricates a missing side.
The Task 2 row-level status vocabulary is exactly `present`, `unavailable`,
`missing`, `blocked`, or `failed`; the view table also retains the frozen
`tensor.attrs["network_ordinals"]` method×resolution registry. Localisation
validates those exact labels and contiguous ordinals (0…7 or 0…16) rather
than inferring a network family from display text.

## Frozen group estimands

`src/contrasts.py` exposes the coefficient-only interfaces
`stanford_contrast_matrix()` and `jjtms_contrast_matrix()`. Each returns a fresh
`pandas.DataFrame` with an explicit `estimand` column followed by the ordered
cell columns. Coefficients are numeric, and the returned snapshot is marked
`attrs[immutable] = True`; callers must not treat mutation of a snapshot as a
change to the frozen definitions.

The Stanford cell order is `NTHC, TEHC, NTS, TIS`. The rows, in deterministic
order, are `symptom_marginal`, `trauma_marginal`, `interaction`, `NTS-NTHC`,
`TIS-TEHC`, `TEHC-NTHC`, and `TIS-NTS`. The first two rows are the exact
equal-weight marginal effects, `0.5*(NTS+TIS)-0.5*(NTHC+TEHC)` and
`0.5*(TEHC+TIS)-0.5*(NTHC+NTS)`, respectively; they are not reference-cell
simple effects. The interaction is `(TIS-TEHC)-(NTS-NTHC)`. JJTMS has cell
order `healthy, patient` and one row, `patient-healthy`.

These functions only define rest-FC/Fisher-z coefficient matrices. They do not
read data or perform group statistics, p-value/FDR correction, profile
summaries, or figures.

## Primary region×network localisation

`src/localisation.py` exposes:

```text
fit_local_effects(tensor, contrasts) -> pandas.DataFrame
```

The input is the subject-level hemisphere-view table from
`build_hemisphere_views`; `L-R` is therefore consumed as the already paired
within-subject Fisher-z(L) − Fisher-z(R) value. The output schema is:

```text
dataset, estimand, resolution, method, smoothing_mm, hemisphere_mode,
region, network, fc_estimator, n, coefficient, hc3_se, ci_low, ci_high,
standardized_effect, p_raw, q_bh, correction_status, failure_reason
```

Each component is an unadjusted cell-means OLS fit on subject-level absolute
Fisher-z values. Standard errors use HC3, confidence intervals and two-sided
p-values use the finite residual t distribution, and `standardized_effect` is
`coefficient / sqrt(SSE / residual_df)` on the common model-residual scale.
Invalid residual degrees of freedom, HC3 variance, or residual variance remain
explicit failure rows; missing values are never zero-filled. Stanford uses all
four frozen cells for each of its seven estimands, while JJTMS uses healthy and
patient for `patient-healthy`. Frozen `group` labels are authoritative and
are cross-checked against supplied Stanford trauma/symptomatic fields.

Multiplicity correction is deliberately deferred until the final analysis-object
set is frozen. `q_bh` is retained as a compatibility column but is null for all
rows, and `correction_status` records `deferred_until_analysis_set_freeze`.
There is no provisional family denominator, q-value threshold, or q-based
selection in this run. Localisation is primary and is never gated by a
profile/global p-value.


## Auxiliary profile summaries

src/global_profile.py exposes:

    run_global_tests(tensor, contrasts, seed, n_perm) -> pandas.DataFrame

This is an auxiliary, non-gating summary. It consumes the same Task 2 hemisphere views and exact Task 3 contrasts as localisation, and emits one subregion_profile row for each of anterior, body, and tail, plus one whole_hippocampus_profile row (region=all) for every fixed dataset, estimand, method, smoothing, hemisphere mode, network resolution, and fisher_z estimator branch. The output records n_total, n_complete, effective_dimension, the squared Euclidean profile-distance statistic, plus-one p_perm, n_perm, seed, derived_seed, and an audited failure_reason.

Each complete subject contributes one fixed-order vector: 8 or 17 network components for a subregion, and all 3x8 or 3x17 region-network components for the whole hippocampus. Incomplete subjects are excluded as whole profiles and are never zero-imputed or permuted network-cell by network-cell. The statistic standardizes each exact contrast effect by its full cell-means model residual SD before summing squared components. Stanford marginal/interaction tests use Freedman-Lane residual permutations under the deterministic exact-contrast reduced model; JJTMS uses the intercept-null reduced model. A stable cryptographic branch/scope seed makes results reproducible and branch-order invariant. These profile p-values have no confirmatory multiplicity correction and cannot suppress or condition primary localisation.

## End-to-end primary run and reporting

`src/run_primary.py` is the entrypoint for the corrected first-two-point analysis. It materializes the frozen subject manifest, group counts, full Fisher-z FC tensor, L/R/bilateral/L-R views, primary region-by-network localisation, auxiliary profiles, Stanford four-group descriptives, cross-dataset direction consistency, the capped untested candidate registry, negative/unavailable rows, and correlated specification consistency. The current output target is stable: rerunning with the same `--run-id primary_screen` replaces the current tables, figures, reports, and manifest.

Run with the repository interpreter:

```bash
/home/yojiao/.conda/envs/py314/bin/python \
  stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/src/run_primary.py \
  --run-id primary_screen --seed 20260903 --n-perm 499
```

Each successful run writes the latest outputs under `results/primary_screen/`, `figures/primary_screen/`, and `reports/primary_screen/`. `results/primary_screen/run_manifest.json` records input/output SHA-256 values, table schemas/counts, the exact command, the actual UTC run-start `created_at`, seed, permutation count, and the complete P0/P1/P2 branch ledger. `created_at` is captured when the run starts; it is never a historical snapshot constant. The five standard report figures are supporting evidence; the focused core figure is maintained alongside them. The canonical report payload and complete scientific narrative, methods, limitations, negative findings, and figure links remain in `reports/primary_screen/`.


## Current unified primary_screen run

- 80 个统一 branch 全部完成：P0=36、P1=8、P2=36；严格按 P0→P1→P2 barrier 执行。
- group-atlas 产出 64-cell inventory、192 行 region-area、56 行 HC-only matched comparison、64 行 specification stability。
- 7 个 group-wise pair 全部是非 HC 对 HC；没有非 HC all-pairs，也没有 individual-gradient branch。
- 活动路径只展示最近一次统一 P0/P1/P2 run；`branch_manifest.json` 中的 `priority_tier` 是分层依据，归档快照不属于活动结果。
- 当前静态 PNG、technical Markdown、tables 和 SHA-256 manifest 是权威结果；`builder_receipt.json` 记录可选 HTML builder 不在环境中。

## Effect-first broad-signal visualization

`src/build_broad_signal_review.py` creates the current `broad_signal_review` derivative directly from the current `primary_screen` tables. It does not refit any statistical model and includes only the two current scientific parcellation specifications. Rerunning it replaces the current derivative outputs.

The level-effect landscape requires 12/12 coefficient signs to agree across method × smoothing × bilateral/L/R and absolute median standardized effect at least 0.35. The cross-dataset panel compares JJTMS patient-minus-healthy with the Stanford symptom marginal at 8-network resolution and requires same-direction 12/12 stability in both datasets. The laterality panel requires 4/4 L−R sign agreement across method × smoothing and absolute median standardized effect at least 0.35. No provisional q value is used for annotation or selection. A separate secondary layer retains the two lower-effect rows with 12/12 sign agreement for visual review, so method-sensitive leads remain visible without being promoted to robust anchors. These correlated specifications are stability summaries, not independent replications, and every displayed row remains exploratory.

Outputs are written under `results/broad_signal_review/`, `figures/broad_signal_review/`, and `reports/broad_signal_review/`. The two main figures are `effect_first_signal_landscape` and `broader_story_map`; the latter combines Stanford four-group shapes, cross-dataset convergence, and stable laterality candidates.

Current tables may retain audit-only unavailable placeholder rows from superseded configurations; they are excluded from the scientific branch inventory. The Stanford HC-referenced group-atlas geometry family is analysed by the same `primary_screen` run and is not injected into the subject-level FC tensor. `snapshot.accessIssues` is reserved for missing required report data. The official builder uses 10-second reader and 20-second whole-check budgets, with one retry only for its observed transient `reader_timeout`; all other verification failures stop the run.

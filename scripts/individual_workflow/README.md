# Group-Template Matching for Individual Hippocampal Parcellation

HCP uses family-aware cross-validation. Non-HCP datasets match a declared group
template through adapters under `scripts/<dataset>/`, including
[JJ_TMS](../jj_tms/README.md). Never cluster an individual's hippocampal data
or pass a single-subject cohort through a group-clustering entrypoint.
Held-out subjects match their FC to training-only group-template profiles.

The root contains numbered entrypoints; shared implementation belongs in `lib/`.
Dataset adapters and unnumbered utilities belong outside this numbered surface.
Native-space beta extraction, modeling and figures remain exploratory under
[scripts/exploratory_beta/](../exploratory_beta/README.md).

| Stage | Entrypoints | Purpose |
|---|---|---|
| `01` | `01_01`–`01_04` | HCP cross-fit matching, robustness and cohort/run comparisons |
| `02` | `02_01`–`02_02` | Individual surface overviews and subject-template cluster FC |
| `03` | `03_01` | Reusable HCP151 fixed-template construction |

Use `/home/yojiao/.conda/envs/py314/bin/python` from the repository root.
Stage `--help` describes its input and output selectors.

## HCP151 Template Builder for Non-HCP Datasets

`03_01_build_hcp151_fixed_template.py` builds the immutable HCP151 template
from matching `outputs_hcp/work/group_atlas/pfm/` and
`outputs_hcp/work/hippomaps_gradients/pfm/` products for 151 subjects.
The default identity is `hcp151_routea_full_k3_fisherz`.
`--template-id hcp151_routea_ctxreg` requires source manifests identifying
`pfm_ctxreg`; changing a directory or label cannot change template identity.

Dataset adapters consume the validated template and native
`space-T1w_den-512` subject meshes. Corobl label-GIFTI display filenames may
retain corresponding vertex numbering without resampling the BOLD volume.
The [JJ_TMS matching guide](../../docs/hipp_parcellation_network/jj_tms_hcp151_fixed_template_matching.md)
defines its complete matching and validation contract.

For Stanford, `02_02_compute_subject_template_cluster_fc.py` averages each
subject's raw hippocampal time series within their own 2mm labels before
computing the `3 x 400` Pearson/Fisher-z matrices. It does not average the
vertex-FC matrix. See [subject cluster FC](../../docs/hipp_parcellation_network/stanford_template_matched_cluster_fc.md).

## HCP crossfit

`01_01_run_hcp_crossfit_template_matching.py` learns group templates and
AP-ordered profiles in five family-grouped training folds, then matches
held-out subjects using full concatenated FC. Production requires an external
`subject,family_id` map; `--allow-subject-only-folds` is an explicit recorded
waiver. Raw family identifiers remain memory-only. Clustering uses the core
shifted-cosine affinity; resume validates current inputs and saved artifacts.

```bash
/home/yojiao/.conda/envs/py314/bin/python   scripts/individual_workflow/01_01_run_hcp_crossfit_template_matching.py   --family-map data/HCP_YA_subjects_2026_04_09_16_01_42.csv --dry-run
```

Use `--resume` for an authorized run. `01_02` validates robustness; `01_03`
compares disjoint cohorts; `01_04` compares runs and split halves.
Computation lives under `outputs_hcp/work/crossfit_template_matching/`;
figures and robustness tables use
`outputs_hcp/present/overviews/crossfit_template_matching/`.
See [crossfit methods and outputs](../../docs/hipp_parcellation_network/crossfit_template_matching.md).

## Individual overviews

`02_01_render_hcp_crossfit_individual_overviews.py` renders fixed 2mm/4mm
rows with left/right hippocampi. Source labels remain AP-ordered; the display
network legend derives from all fold/smoothing/hemisphere template profiles
and requires a stable unique winner under the canonical7 taxonomy.

Plain `--resume` reuses only validated current source/scene/display products.
Use `--resume --refresh-visuals` for an explicitly requested display replacement.
The renderer defaults to four workers and caps `--jobs` at ten. Outputs,
status and the network-label mapping are under
`present/overviews/crossfit_template_matching/individual/`.

## Family17 gradient views

The overview adapter's `--network-column family17` applies to gradient
`fisherz`, `weighted_fisherz` and the retained `gradient_count` compatibility
alias for `normalized_top20_count`. Family17 filenames carry a distinct
`gradient_family17_` prefix. Missing assignment declarations fail closed.
Weighted profiles use grayordinate-count-weighted ROI time series, declared
runwise centering and intersected validity; undefined profiles remain unassigned.

Individual WTA is separate: it uses the canonical8 ROI column and only
`normalized_top20_count`. Family17 computation requires `--gradient-only`;
it does not create family17 WTA labels.

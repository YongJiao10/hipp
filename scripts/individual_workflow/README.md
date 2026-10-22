# Group-Template Matching for Individual Hippocampal Parcellation

This directory contains reusable template-matching workflows. HCP uses
family-aware cross-validation; non-HCP datasets use a declared group template
through adapters under `scripts/<dataset>/` (JJ_TMS: `scripts/jj_tms/`).
Clustering fits group data, including a cross-validation training cohort.
Never cluster an individual's hippocampal data or pass a single-subject cohort
through a group-clustering entrypoint. Held-out subjects receive labels by
matching their FC to training-only group-template profiles.

The root contains numbered entrypoints only. Shared individual implementation
code belongs in `lib/`; maintenance and migration utilities belong in
`scripts/maintenance/`.

Entry points use consecutive stage and step numbers:

| Stage | Entry points | Purpose |
| --- | --- | --- |
| `01` | `01_01`–`01_04` | HCP cross-fit matching, robustness, and cohort/run comparisons |
| `02` | `02_01`–`02_02` | Individual surface overviews and subject-template cluster FC |
| `03` | `03_01` | Reusable HCP151 fixed-template construction |

Native-space beta extraction, modeling, and figures remain exploratory and
live in `scripts/exploratory_beta/`. Do not add beta entry points to this
formal individual-workflow surface until that analysis is explicitly promoted.

## HCP151 Template Builder for Non-HCP Datasets

`03_01_build_hcp151_fixed_template.py` builds an immutable HCP151 template
from `outputs_hcp/work/group_atlas/pfm/` and the matching
`work/hippomaps_gradients/pfm/` artifacts for 151 subjects. Its default identity is
`hcp151_routea_full_k3_fisherz`; the explicit
`--template-id hcp151_routea_ctxreg` identity is allowed only when all source
manifests prove `pfm_ctxreg`. The template identity is recorded in the
manifest and cannot be changed by renaming an output directory. It is a
reusable template-building step, not a JJ_TMS adapter. Non-HCP datasets
consume a validated template through dataset-specific direct-matching code
under `scripts/<dataset>/`.

JJ_TMS matching, target mapping, and fixed-template overview rendering live in
`scripts/jj_tms/`; their complete launch and validation contract is maintained
in `docs/hipp_parcellation_network/jj_tms_hcp151_fixed_template_matching.md`.

For Stanford's existing HCP151 template-matching labels, the subject-specific
cluster-mean 400-ROI FC stage is
`02_02_compute_subject_template_cluster_fc.py`. It is restricted to `2mm` and
uses each subject's own labels to average the raw hippocampal surface time
series before calculating the `3 x 400` Pearson and Fisher-z matrices. It never
averages the existing `512 x 400` vertex-FC matrix. The full input, output,
provenance, and colleague-copy contract is documented in
`docs/hipp_parcellation_network/stanford_template_matched_cluster_fc.md`.

## Family-Aware Cross-Fitted Labels

For reviewer-defensible individualized K=3 labels without run-level splitting,
use `01_01_run_hcp_crossfit_template_matching.py`. Five family-grouped folds learn
training-only group templates and AP-ordered FC profiles; each held-out subject
is then matched using their full concat FC. HCP production requires an external
`subject,family_id` family map. The explicit `--allow-subject-only-folds`
waiver is recorded and is not the preferred HCP definition.
Template clustering shares the core workflow's fixed `shifted-cosine`
surface affinity. No affinity compatibility switch is exposed; the fixed
definition is resume-locked and saved with graph-connectivity diagnostics.
The maintained source `data/HCP_YA_subjects_2026_04_09_16_01_42.csv` can be
passed directly; its native `Subject` and `Family_ID` headings are accepted
without creating a derived restricted-data file.
Raw family identifiers remain memory-only and are not persisted in workflow
outputs. Resume uses atomic in-progress state plus SHA-256 fingerprints for
inputs and saved artifacts, and subject surfaces are vertex-count validated.

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/individual_workflow/01_01_run_hcp_crossfit_template_matching.py --family-map /home/yojiao/hpchome/hipp/data/HCP_YA_subjects_2026_04_09_16_01_42.csv --dry-run
/home/yojiao/.conda/envs/py314/bin/python scripts/individual_workflow/01_01_run_hcp_crossfit_template_matching.py --family-map /home/yojiao/hpchome/hipp/data/HCP_YA_subjects_2026_04_09_16_01_42.csv --resume
/home/yojiao/.conda/envs/py314/bin/python scripts/individual_workflow/01_02_validate_hcp_crossfit_robustness.py
/home/yojiao/.conda/envs/py314/bin/python scripts/individual_workflow/01_03_run_and_render_hcp_crossfit_disjoint_folds.py
/home/yojiao/.conda/envs/py314/bin/python scripts/individual_workflow/01_04_render_hcp_runs_comparison.py
```

Fixed-template overview inputs declare native `space-T1w_den-512`
subject midthickness surfaces, matching the repaired sampling/smoothing
producer. Label GIFTIs may retain corobl display filenames because vertex
numbering and topology are shared across spaces; rendering a metric does
not resample the BOLD volume.

The computation lives under
`outputs_hcp/work/crossfit_template_matching/`; the quantitative five-fold
overview, `robustness_summary.tsv`, completely disjoint independent-cohort
Workbench 3D comparison
(`hcp_routea_crossfit_k3_disjoint_folds_workbench_comparison.png`,
`disjoint_folds_robustness_summary.tsv`), 4-run split-half / test-retest
2×2 comparison (`hcp_routea_crossfit_k3_four_runs_workbench_2x2_comparison.png`,
`four_runs_robustness_summary.tsv`), and session split-half × smoothing scale
2×2 comparison (`hcp_routea_crossfit_k3_splithalf_2x2_comparison.png`,
`splithalf_2x2_robustness_summary.tsv`) live under
`outputs_hcp/present/overviews/crossfit_template_matching/`. See
`docs/hipp_parcellation_network/crossfit_template_matching.md` for the
non-circularity argument, artifacts, and interpretation.


### Cross-Fit Individual Surface Overviews

Render a compact subject-level comparison of the cross-fitted labels. To
replace the prior AP-labelled PNGs with the Fisher-z network legend, run:

    /home/yojiao/.conda/envs/py314/bin/python scripts/individual_workflow/02_01_render_hcp_crossfit_individual_overviews.py --jobs 4 --resume --refresh-visuals

The immutable source codes remain AP-ordered (`1=Anterior`, `2=Body`,
`3=Posterior`) in cross-fit labels and source manifests. Only the display layer
changes: the renderer fingerprint-validates all five fold × two smoothing × L/R
`template_profiles.npy` artifacts, averages each code’s 400-ROI Fisher-z profile
within the canonical7 Kong2022 networks, and requires a stable unique winner. In
the current HCP run this gives `1=VAN`, `2=SMN`, and `3=DMN` in every one of the
20 templates. The legend keeps the short names but uses canonical Kong2022 style
keys: VAN/VentralAttention `(220,92,255,255)`, SMN/Somatomotor
`(99,150,194,255)`, and DMN/Default `(216,92,107,255)` from
`config/hipp_network_style.json`.

The command preserves the standard locked Workbench hippocampal rendering used
by core overviews. Each subject PNG has the fixed 2 mm row above the fixed 4 mm
row, with left/right hemisphere surfaces and the Fisher-z network legend; the
composite header wraps before the subject identifier when needed, so the full
`sub-…` title remains visible. Alternative subsets or row orders are rejected. It
validates the held-out combo
manifest, AP source-label semantics, label-GIFTI SHA-256 fingerprint, the exact
subject/hemi den-512 consumed surface, its recorded input fingerprint, its point
count, the locked Workbench scene SHA-256, all template-profile fingerprints,
and the display-contract fingerprint before rendering. A completed PNG is
atomically published. Plain `--resume` only reuses a PNG with matching source,
scene, image, and visual-label contract fingerprints; a legacy AP-labelled PNG
requires the explicit `--resume --refresh-visuals` replacement request. The
aggregate items list preserves the last validated subject records, while the
status TSV and `attempt_items` log the current invocation, so a failed retry
cannot erase valid provenance. Outputs are kept in the canonical
`outputs_hcp/present/overviews/crossfit_template_matching/individual/` subtree
with `individual_overview_status.tsv`, `individual_overview_manifest.json`, and
`network_label_mapping.json`. The default uses four workers and is capped at the
repository-wide limit of ten.


## Family17 Gradient Overview Rendering

The individual overview adapter accepts `--network-column family17` for
Kong2022 17-network gradient rendering. The option affects the gradient
`fisherz`, legacy `gradient_count` (normalized Top-20 count), and
`weighted_fisherz` overviews and writes PNG names with
a `gradient_family17_` prefix in the stem so canonical8 outputs are not
overwritten. The adapter records `network_column: family17` in
`group_feature_summary.json`; `fisherz` heatmaps use the ROI-mean Fisher-z
cluster-network profile, while `weighted_fisherz` recomputes network time
series from ROI time series using `n_grayordinates_used` weights before
cluster-to-network correlation. `fisherz` and `weighted_fisherz` heatmaps are
validated against their Fisher-z status rows. The legacy `gradient_count`
method name is retained as a compatibility alias, but its heatmap and status
rows use `normalized_top20_count` with the normalized Top-20 ROI profile.
Weighted time-series profiles use the exact reference run lengths to remove
within-run means before correlation, preserve run variances, and intersect the
subject validity mask with group support. Incomplete or constant rows remain
undefined; entirely missing network profiles are unassigned.
Stale or missing assignment-variant manifest paths fail instead of falling
back to same-run-dir files.

The individual WTA writer is separate from this gradient taxonomy option: it
always uses the canonical8 ROI column and writes only
`normalized_top20_count`. Passing `--network-column family17` cannot produce a
family17 WTA output; the computation runner must receive `--gradient-only` for
that taxonomy, and the overview adapter accepts only the three gradient
methods in that mode.

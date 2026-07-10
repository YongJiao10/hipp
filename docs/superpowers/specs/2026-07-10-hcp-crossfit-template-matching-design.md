# HCP Cross-Fitted Template Matching Design

## Goal

Generate one reviewer-defensible individualized hippocampal K=3 label map for
every QC-passed HCP 7T subject without run-level splitting and without allowing
any subject to contribute functional-connectivity data to the group template
used to label that subject.

## Scope

The maintained implementation is a RouteA/Kong2022 400-ROI individual workflow.
It uses the existing 151-subject QC manifest, vertex-validity masks, `2mm` and
`4mm` hippocampal vertex-to-ROI FC caches, and `space-corobl_den-512` template
and subject surfaces. It writes into new cross-fit work and presentation roots;
it never replaces the current subject-specific gradient/WTA products or the
full-cohort group atlas.

Run-level splitting, behavioral association testing, external-dataset transfer,
automatic K selection, and hierarchical Bayesian parcellation are outside this
implementation.

## Fold Contract

The default is five deterministic folds. A family mapping is an external TSV or
CSV with columns `subject` and `family_id`. Every family is assigned wholly to
one fold, and a greedy seeded assignment balances subject counts across folds.
The repository must not copy or commit restricted HCP family metadata.

An HCP production run without a family mapping fails. Subject-only folds are
available only through an explicit `--allow-subject-only-folds` waiver, and the
waiver is recorded in every manifest so those outputs cannot be mistaken for
the preferred family-aware analysis.

Each subject occurs in exactly one held-out fold. For every fold, manifests list
the training and held-out subjects and assert that the sets are disjoint. Fold
IDs and assignments are deterministic for the same manifest, family mapping,
fold count, and random seed.

## Fold Template

For each fold, smoothing, and hemisphere:

1. Use only training subjects to calculate the vertex coverage mask at the
   fixed `0.80` threshold.
2. Fisher-z transform each subject's vertex-by-400-ROI FC before calculating the
   training mean.
3. Fit the existing HippoMaps/BrainSpace Pearson-kernel gradients with the
   repository's fixed random state and retain G1-G3 for clustering.
4. Run the existing surface-adjacency-constrained spectral clustering at fixed
   K=3.
5. Order clusters head-to-tail using the anatomical long-axis coordinate, so
   labels 1, 2, and 3 mean Anterior, Body, and Posterior without reference to a
   full-cohort functional template.
6. Average the training group Fisher-z FC within each ordered cluster to create
   a `3 x 400` connectivity prototype matrix.

The common group-average surface is a structural coordinate asset rather than a
functional estimator. Its consumed path and filename-derived density evidence
are recorded, while all functional template quantities are cross-fitted.

## Held-Out Individual Matching

For each held-out subject, use the complete concatenated FC cache. Fisher-z
transform the FC and calculate a finite-feature Pearson correlation between
each valid hippocampal vertex and each of the three training prototypes. The
maximum correlation selects the label. Invalid vertices remain label 0.

No behavioral value, full-cohort template, or other held-out subject is used in
assignment. No topology cleanup is applied in the first maintained version;
this keeps the mapping rule transparent and avoids an unvalidated spatial
hyperparameter. The saved top1-minus-top2 correlation margin exposes ambiguous
vertices rather than silently smoothing them away.

Each subject/combo output contains:

- `individual_labels_k3.npy` and a named GIFTI label asset;
- `template_correlations.npy` with shape `n_vertices x 3`;
- `correlation_margin.npy`;
- a combo manifest containing the fold, training template, FC/mask/surface
  inputs, parameters, density evidence, label counts, and ambiguity summaries.

## Output Layout

```text
outputs_hcp/work/crossfit_template_matching/
  folds/fold_assignments.tsv
  folds/fold_manifest.json
  templates/fold-01/<2mm|4mm>/hemi_<L|R>/
  subjects/sub-<id>/<2mm|4mm>/hemi_<L|R>/
  crossfit_manifest.json

outputs_hcp/present/overviews/crossfit_template_matching/
  hcp_routea_crossfit_k3_robustness_overview.png
  robustness_summary.tsv
  robustness_manifest.json
```

## Robustness Overview

The five fold templates should be plotted because different held-out subjects
are labeled by different templates. A useful overview must quantify that fold
effect rather than merely display five attractive surfaces.

For each smoothing/hemisphere combo, the overview contains:

- five template-label strips ordered along the anatomical long axis;
- a pairwise macro-Dice matrix;
- a pairwise adjusted-Rand-index matrix;
- corresponding-cluster prototype correlations;
- vertex-wise modal agreement across folds.

The summary TSV records the mean and minimum pairwise Dice, ARI, and prototype
correlation plus mean and minimum vertex agreement. Fold template GIFTI files
remain available for optional Workbench surface inspection without making
Workbench rendering a prerequisite for the numerical robustness figure.

## Validation

Automated tests cover family containment, deterministic balanced assignment,
explicit subject-only waiver, train/test disjointness, Fisher-z template
profiles, finite-feature vertex-to-prototype correlations, AP label semantics,
null handling, output provenance, pairwise robustness metrics, and overview
generation from synthetic artifacts.

The implementation passes the existing group-gradient and individual-workflow
tests. A synthetic end-to-end smoke run exercises two hemispheres and at least
two folds without reading official full-cohort outputs. Full HCP computation is
not launched automatically; the CLI first supports a dry-run workload and
input audit.

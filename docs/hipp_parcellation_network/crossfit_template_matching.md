# HCP Cross-Fitted Template Matching

This workflow creates one individualized anterior/body/posterior hippocampal
label map per HCP subject, hemisphere, and smoothing level. Its inferential
unit is the subject (with family-aware grouping), not an fMRI run. Each subject
is labelled from their full concatenated FC using a group template learned
without that subject or any member of the same HCP family.

## Why this avoids circularity

Subjects are assigned deterministically to five folds. The production input is
an external TSV or CSV with the columns `subject,family_id`; all members of a
family stay in the same fold. For each held-out fold and each smoothing/hemi
combination, the workflow uses only the other four folds to estimate the group
support mask, group mean Fisher-z FC, gradients, fixed K=3 surface clusters,
and the three AP-ordered cluster FC profiles. A held-out vertex receives the
label of the training profile with the highest Pearson correlation.

The held-out subject's data therefore affect its own label assignment but not
the template used to define the candidate labels. This is cross-fitting at the
subject/family level. It is not run-level cross-validation and makes no claim
about independence between runs. Using the subject's full concat FC is
intentional: the target is a stable full-data individualized label, and the
same method can be applied to datasets that contain only one run.

The family map may contain restricted HCP metadata. It is an execution input,
not a repository artifact, and must not be committed. Production HCP results
require `--family-map`. The explicit `--allow-subject-only-folds` waiver is for
non-family datasets or interface checks; its use is recorded in the manifest
and is not the preferred HCP production definition.

For the maintained HCP run, pass
`data/HCP_YA_subjects_2026_04_09_16_01_42.csv` directly. The parser accepts the
HCP-native `Subject` and `Family_ID` headings case-insensitively as well as the
generic `subject,family_id` form, and reads only those two semantic columns.

Raw `family_id` values are used only in memory to keep relatives together.
They are not copied into `fold_assignments.tsv` or any JSON output; saved fold
assignments contain only subject and fold identifiers.

## Commands

Audit all expected inputs without writing workflow products:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/individual_workflow/routea_crossfit_template_matching.py --family-map /home/yojiao/hpchome/hipp/data/HCP_YA_subjects_2026_04_09_16_01_42.csv --dry-run
```

Run the computation, or resume an exactly matching partial/completed run:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/individual_workflow/routea_crossfit_template_matching.py --family-map /home/yojiao/hpchome/hipp/data/HCP_YA_subjects_2026_04_09_16_01_42.csv --resume
```

Resume validation locks the cohort, family/fold assignment, input and output
roots, smoothing/hemi combinations, numerical parameters, SHA-256 input
fingerprints, and saved-artifact fingerprints/shapes/dtypes. An atomic
`status=in_progress` run manifest is written before the first fold artifact.
Any nonempty output root requires `--resume`, and a nonempty root without that
manifest is rejected rather than guessed or overwritten.

After all five fold templates exist, render the quantitative robustness review:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/individual_workflow/03_build_crossfit_robustness_overview.py
```

All commands accept explicit root overrides. Run `--help` for the complete
interface. The default production combinations are 2 mm and 4 mm smoothing,
left and right hemispheres, fixed K=3, and five folds.

Fold-template gradient-spectral clustering uses the same fixed affinity as the
group-atlas workflow: `shifted-cosine`, defined as
`(1 + cosine_similarity) / 2` on surface-adjacent vertex pairs. There is no
runtime affinity selector or compatibility mode. The fixed definition and
graph-connectivity diagnostics are recorded in every template and run manifest
and are locked during resume.

## Outputs and provenance

Computation products are isolated under:

```text
outputs_hcp/work/crossfit_template_matching/
  crossfit_manifest.json
  folds/
    fold_assignments.tsv
    fold_manifest.json
  templates/fold-<01..05>/<2mm|4mm>/hemi_<L|R>/
    group_support_mask.npy
    group_labels_k3.npy
    group_labels_k3.label.gii
    template_profiles.npy
    template_manifest.json
  subjects/sub-<id>/<2mm|4mm>/hemi_<L|R>/
    individual_labels_k3.npy
    template_correlations.npy
    correlation_margin.npy
    sub-<id>_...label-crossfit_template_matching_k3_...label.gii
    combo_manifest.json
```

Template and subject manifests record the exact training subjects, input FC
and masks, consumed subject/template surfaces, density evidence, parameters,
SHA-256 integrity fingerprints, and output paths. Subject GIFTI point counts
must match the valid mask and individualized label length before the surface is
recorded as consumed density evidence. Consumed surface assets are
authoritative for density.

The robustness renderer writes:

```text
outputs_hcp/present/overviews/crossfit_template_matching/
  hcp_routea_crossfit_k3_robustness_overview.png
  robustness_summary.tsv
  robustness_manifest.json
```

Each smoothing/hemi row displays the five AP-ordered fold-label strips,
pairwise macro-Dice and adjusted Rand index heatmaps, corresponding-cluster
prototype correlations, and the long-axis vertex agreement trace. These
measure whether modest changes in the training cohort materially change the
learned template. They review template robustness; they are not held-out
prediction-accuracy estimates and do not replace downstream biological
validation.
Agreement excludes null labels and unsupported fold vertices; a vertex with no
supported nonzero fold label is recorded as missing rather than perfectly
stable. Before loading numeric arrays, the renderer verifies the producing
template manifest's surface/artifact SHA-256 fingerprints and native
shape/dtype contracts; a modified fold cannot silently yield a new overview.
The renderer also evaluates the registered production gate: every fold graph
must be connected, every cluster must contain at least 10% of supported
vertices, and all unique fold pairs must meet macro-Dice 0.85, ARI 0.70, and
prototype-correlation 0.50. It exits nonzero when the gate fails.

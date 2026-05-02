# Scripts Layout

This directory is organized by responsibility.

- `common/`
  Shared numerical utilities and reusable helpers used across pipelines.
- `cortex/`
  Cortex PFM derivation, ROI-component extraction, and cortex-specific rendering.
- `workbench/`
  Workbench scene capture, legend composition, and native-surface rendering helpers.
- `hipp_parcellation_network/`
  Active hippocampal **group-analysis** package.
  Public entry points:
  - `materialize_kong7_kong8_pearson_partial_fc.py`
  - `run_merged_canonical_group_atlas.py`
  - `build_present_group_overview_merged_canonical.py`
  - `build_present_group_t1only_matrix.py`
  - `run_t1only_spectral_fast_rerun.py`
  - `materialize_group_functional_atlas_labels.py`

Top-level `scripts/` still contains broader workflow helpers shared across the repository, such as HippUnfold wrappers and environment utilities.

Legacy subject-level parcellation entrypoints and retired group-prior scripts were moved to:

- `_archive/hipp_parcellation_network/legacy_subject_pipeline/`
- `_archive/hipp_parcellation_network/legacy_group_prior/`
- `_archive/hipp_parcellation_network/legacy_group_methods/`

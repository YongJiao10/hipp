# Scripts Layout

This directory is now organized by responsibility rather than keeping every entrypoint at the top level.

- `common/`
  Shared numerical utilities and reusable helpers used across pipelines.
- `cortex/`
  Cortex PFM derivation, ROI-component extraction, and cortex-specific rendering.
- `workbench/`
  Workbench scene capture, legend composition, and native-surface rendering helpers.
- `experiments/hipp_functional_parcellation/`
  Experimental comparison workflow for hippocampal functional parcellation branches.

Top-level `scripts/` still contains broad pipeline entrypoints and wrappers that are shared across multiple flows, such as `wb_command`, `run_hippomaps_pipeline.py`, and post-HippUnfold orchestration scripts.

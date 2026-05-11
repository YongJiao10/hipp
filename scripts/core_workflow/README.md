# Core Workflow Entrypoints

This directory is the canonical public script surface for active hippocampal workflow runs.

- Naming scheme: `NN_MM_<semantic_name>.py`
- Numbering reflects global pipeline order, not branch identity.
- Active branches: `kong2019`, `kong2022_routea`, `kong2022_routeb`

Each entrypoint wraps an implementation script and can emit a standardized traceability manifest:

```bash
python scripts/core_workflow/05_01_run_group_atlas.py \
  --workflow-branch kong2022_routeb \
  --upstream-step-id 04_01_materialize_kong2022_roi_fc \
  --upstream-step-id 04_02_materialize_intrinsic_fc \
  --input-artifact outputs_experiments/.../shared_inputs/routeb_151 \
  --output-artifact outputs_experiments/.../hipp_group_merged_canonical_atlas_routeb \
  --step-manifest-out manifests/core_workflow/05_01_run_group_atlas.json \
  -- --methods intrinsic-spectral network-spectral wta-top
```

Wrapper arguments:

- `--workflow-branch`: one of `kong2019`, `kong2022_routea`, `kong2022_routeb`
- `--upstream-step-id` (repeatable)
- `--input-artifact` (repeatable)
- `--output-artifact` (repeatable)
- `--step-manifest-out`
- `--dry-run`

Unknown arguments are forwarded to the wrapped implementation script.

The step inventory and branch coverage matrix are defined in:

- `scripts/core_workflow/step_registry.json`

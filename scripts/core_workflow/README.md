# Core Workflow Entrypoints and Implementation

This directory is the canonical active workflow surface.
Both CLI entrypoints and stage implementation modules live under `scripts/core_workflow/`.

- Naming scheme: `NN_MM_<semantic_name>.py`
- Numbering reflects global pipeline order, not branch identity.
- Active branches: `kong2019`, `kong2022_routea`, `kong2022_routeb`

Each entrypoint executes local implementation code and can emit a standardized traceability manifest:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/05_01_run_group_atlas.py \
  --workflow-branch kong2022_routeb \
  --upstream-step-id 04_01_materialize_kong2022_roi_fc \
  --upstream-step-id 04_02_materialize_intrinsic_fc \
  --input-artifact outputs_experiments/.../shared_inputs/routeb_151 \
  --output-artifact outputs_experiments/.../hipp_group_merged_canonical_atlas_routeb \
  --step-manifest-out manifests/core_workflow/05_01_run_group_atlas.json \
  -- --methods intrinsic-spectral network-spectral gradient-spectral wta-top
```

For the Kong2022 `gradient-spectral` branch, `05_01_run_group_atlas.py`
computes diffusion gradients from group-averaged vertex-to-ROI FC and clusters
the selected gradient columns. The default gradient set is `G1,G2,G3`; valid
overrides are `--gradient-components 1`, `2`, `3`, `1,2`, and `1,2,3`.

Execution environment (hard requirement):

- Do not run core workflow tasks with bare `python`.
- Use `/home/yojiao/.conda/envs/py314/bin/python` for core workflow scripts/tests.
- See `docs/environment/agent_runtime_contract.md` for preflight checks.

Entrypoint arguments:

- `--workflow-branch`: one of `kong2019`, `kong2022_routea`, `kong2022_routeb`
- `--upstream-step-id` (repeatable)
- `--input-artifact` (repeatable)
- `--output-artifact` (repeatable)
- `--step-manifest-out`
- `--dry-run`

Unknown arguments are forwarded to the local stage implementation module.

Runtime boundary:

- allowed external assets/executables:
  - configs/data/generated outputs
  - `software/workbench/bin_linux64/wb_command`
- forbidden for active steps:
  - Python/Shell helper scripts outside `scripts/core_workflow/`
  - direct runtime dependency on legacy helper trees (`common/*`, `workbench/*`) or the repo `wb_command` wrapper

The step inventory and branch coverage matrix are defined in:

- `scripts/core_workflow/step_registry.json`
- `scripts/core_workflow/lib/`

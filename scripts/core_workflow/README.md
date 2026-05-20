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
  -- --methods intrinsic-spectral network-spectral gradient-spectral wta-top \
  --gradient-root outputs_core/hcp7t_hippocampus/runs/<run_id>/gradients
```

For the Kong2022 `gradient-spectral` branch, run
`05_03_compute_hippomaps_gradients.py` first. `05_01_run_group_atlas.py` now
loads those HippoMaps/BrainSpace gradient artifacts from `--gradient-root` and
clusters the selected columns. The default gradient set is `G1,G2,G3`; valid
overrides are `--gradient-components 1`, `2`, `3`, `1,2`, and `1,2,3`. The
maintained default is fixed final `K=3` (`--k-min 3 --k-max 3
--force-final-k 3`); `--auto-k` remains available only as legacy diagnostics.

HippoMaps-style continuous gradients are computed by
`05_03_compute_hippomaps_gradients.py` and rendered by
`06_05_render_hippomaps_gradient_products.py`. This route imports official
package code from `external/hippomaps`, keeps Fisher-z before group averaging,
applies column-wise feature z-score before `GradientMaps(kernel="pearson")`, and
renders folded/unfolded gradients with `surfplot_canonical_foldunfold`. For
`den=512`, the renderer treats `512` as HippoMaps v2 in the current Python
process and disables `unfoldAPrescale`; no AP surfarea rescale is used for
512-density figures. The renderer defaults to bilateral products; use `--hemis
L` or `--hemis R` for single-hemisphere output files.

Canonical new outputs live under
`outputs_core/hcp7t_hippocampus/runs/<run_id>/`. Use
`00_01_prepare_smoke_run.py` to create a 5-subject smoke run manifest and
command plan. Method overview PNGs share one folder:
`outputs_core/hcp7t_hippocampus/runs/<run_id>/present/overviews/`. Supplementary
continuous-gradient products live in `present/gradients/`, and cortex panels live
in `present/cortex_panels/`. Directory names intentionally omit `t1only`,
`present_fixedk`, and `10perc`; fixed K=3 is the default presentation policy.

Execution environment (hard requirement):

- Do not run core workflow tasks with bare `python`.
- Use `/home/yojiao/.conda/envs/py314/bin/python` for core workflow scripts/tests.
- Workbench calls go through `scripts/core_workflow/lib/workbench.py`, which
  defaults `CORE_WORKFLOW_WB_THREADS=1` and sets the common native thread
  variables (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
  `NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`,
  `ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS`) before every `wb_command` call.
- `01_01_prepare_group_inputs.py` hard-caps subject-level Workbench stages at
  two concurrent subjects even when `--jobs` is higher. This keeps Python
  workers plus Workbench subprocesses below the repository concurrency limit.
- See `docs/environment/agent_runtime_contract.md` for preflight checks.

Entrypoint arguments:

Kong2022 reference batch entrypoints (`03_01` and `03_02`) accept
`--data-root`; `03_01`, `03_02`, and `05_01` also accept
`--kong2022-dlabel`. The wrappers pass explicit per-subject dtseries paths and
explicit atlas assets to extraction/annotation modules, so smoke runs can reuse
validated trees outside the current worktree without creating symlinks. `05_03`
and `05_01` accept `--hippomaps-root` for the same reason.

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

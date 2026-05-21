# Core Workflow Step Registry and IO Matrix

Canonical public entrypoints are under `scripts/core_workflow/`. The numeric prefixes are part of the contract: the first number is the hard pipeline layer, and a later layer must not be required by an earlier layer. The second number is reserved for parallel branches or branch-specific products within that layer. Smoke setup is a utility and is not part of the numbered main pipeline.

| Step ID | Purpose | Branches | Entrypoint | Primary Inputs | Primary Outputs |
|---|---|---|---|---|---|
| `01_01_run_hippunfold_t1w_local` | Run HippUnfold | `kong2019`, `kong2022_routea`, `kong2022_routeb` | `scripts/core_workflow/01_01_run_hippunfold_t1w_local.py` | BIDS structural/fMRI inputs and HippUnfold config | HippUnfold subject outputs |
| `01_02_check_hippunfold_outputs` | Validate HippUnfold outputs | `kong2019`, `kong2022_routea`, `kong2022_routeb` | `scripts/core_workflow/01_02_check_hippunfold_outputs.py` | HippUnfold outputs | Integrity reports / validation status |
| `02_01_prepare_group_inputs` | Prepare shared surfaces, tSNR metadata, and group templates | `kong2019`, `kong2022_routea`, `kong2022_routeb` | `scripts/core_workflow/02_01_prepare_group_inputs.py` | Validated HippUnfold outputs and fMRI volumes | Shared input roots, templates, prep manifest |
| `02_02_build_qc_manifest` | Build QC-pass cohort manifest | `kong2019`, `kong2022_routea`, `kong2022_routeb` | `scripts/core_workflow/02_02_build_qc_manifest.py` | HippUnfold Dice TSV and tSNR gate summaries | QC pass/excluded manifests and QC tables |
| `03_01_build_kong2022_reference` | Build shared Kong2022 cortex references | `kong2022_routea`, `kong2022_routeb` | `scripts/core_workflow/03_01_build_kong2022_reference.py` | QC subject list and cortex assets | Per-subject Kong2022 reference arrays |
| `03_02_build_kong2022_routea_reference` | Build RouteA-specific cortex references | `kong2022_routea` | `scripts/core_workflow/03_02_build_kong2022_routea_reference.py` | RouteA assets and QC subject list | RouteA-compatible reference arrays |
| `04_01_materialize_kong2022_roi_fc` | Materialize Kong2022 vertex-to-ROI FC | `kong2022_routea`, `kong2022_routeb` | `scripts/core_workflow/04_01_materialize_kong2022_roi_fc.py` | Smoothed hippocampal timeseries and cortex references | ROI FC caches |
| `04_02_materialize_intrinsic_fc` | Materialize intrinsic FC | `kong2019`, `kong2022_routea`, `kong2022_routeb` | `scripts/core_workflow/04_02_materialize_intrinsic_fc.py` | Smoothed hippocampal timeseries | Intrinsic FC caches |
| `04_03_materialize_kong2019_fc` | Materialize Kong2019 FC bank | `kong2019` | `scripts/core_workflow/04_03_materialize_kong2019_fc.py` | Kong2019 shared inputs and subject manifest | Kong2019/Kong2019_8net FC caches |
| `05_01_compute_hippomaps_gradients` | Compute HippoMaps-style continuous gradients | `kong2022_routea`, `kong2022_routeb` | `scripts/core_workflow/05_01_compute_hippomaps_gradients.py` | Kong2022 vertex-to-ROI FC caches, support masks, group-average surfaces | Full-surface gradients, lambdas, EV fractions, group mean Fisher-z FC, manifest |
| `05_02_run_group_atlas` | Build merged-canonical group atlas outputs with K=2..10 diagnostics and fixed final K=3 | `kong2019`, `kong2022_routea`, `kong2022_routeb` | `scripts/core_workflow/05_02_run_group_atlas.py` | FC caches, group templates, branch controls, optional `--gradient-root` from `05_01` | Group labels, K summaries, assignment status, functional atlas labels |
| `05_03_run_spectral_fast_rerun` | Optional rerun orchestration after primary group-atlas construction | `kong2022_routea`, `kong2022_routeb` | `scripts/core_workflow/05_03_run_spectral_fast_rerun.py` | Existing branch roots, manifests, templates | Refreshed spectral outputs and overview-ready group roots |
| `06_01_build_group_overview` | Build per-method group overview figures | `kong2019`, `kong2022_routea`, `kong2022_routeb` | `scripts/core_workflow/06_01_build_group_overview.py` | Group-atlas runtime artifacts from `05_02` | Method overview PNGs and summaries |
| `06_02_build_wta_long_axis_profiles` | Generate WTA-top long-axis network profile products | `kong2022_routea` | `scripts/core_workflow/06_02_build_wta_long_axis_profiles.py` | WTA-top functional labels and template long-axis surfaces from `05_02` | WTA profile PNG/TSV files and manifests |
| `06_03_materialize_group_labels` | Materialize functional-atlas label arrays | `kong2019`, `kong2022_routea`, `kong2022_routeb` | `scripts/core_workflow/06_03_materialize_group_labels.py` | Assignment status JSON | `group_functional_atlas_labels.npy` |
| `06_04_build_cluster_cortex_panels` | Build cortex/hippocampal summary panels | `kong2019`, `kong2022_routea`, `kong2022_routeb` | `scripts/core_workflow/06_04_build_cluster_cortex_panels.py` | Group labels, FC summaries, topology assets | Cluster and clusterless cortex-panel products |
| `06_05_render_hippomaps_gradient_products` | Render HippoMaps continuous-gradient products | `kong2022_routea`, `kong2022_routeb` | `scripts/core_workflow/06_05_render_hippomaps_gradient_products.py` | `05_01` gradient manifests and arrays | Folded/unfolded G1-G3 figures, EV plots, endpoint FC contrast plots, manifest |
| `06_06_render_group_gradient_heatmaps` | Optional legacy locked-scene group-gradient heatmap renderer | `kong2022_routea`, `kong2022_routeb` | `scripts/core_workflow/06_06_render_group_gradient_heatmaps.py` | Group-gradient products | Locked-scene heatmaps |
| `07_01_build_group_matrix` | Build maintained presentation index after layer 6 products | `kong2022_routea` by default | `scripts/core_workflow/07_01_build_group_matrix.py` | Method overview PNGs and WTA profile products | Matrix manifest, README, refreshed overview set |

## Utilities

| Utility ID | Purpose | Entrypoint | Output Root |
|---|---|---|---|
| `smoke_prepare_run` | Prepare an isolated smoke-run command plan; not a numbered pipeline step | `scripts/core_workflow/smoke_prepare_run.py` | `outputs_core/smoke_runs/<run_id>/` |

Machine-readable source of truth: `scripts/core_workflow/step_registry.json`.

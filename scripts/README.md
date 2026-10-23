# Scripts

Run commands from the repository root with the runtime for the selected system.
Use `/home/yojiao/.conda/envs/py314/bin/python` for local Python tasks;
Argon launchers use their documented cluster runtime. See the
[agent runtime contract](../docs/environment/agent_runtime_contract.md).

## Entry points

| Directory | Purpose | Guide |
|---|---|---|
| `core_workflow/` | Multi-dataset anatomy, registered BOLD sampling, QC, cortical references, FC, gradients and group atlases | [Core workflow](core_workflow/README.md) |
| `individual_workflow/` | HCP cross-fit matching and reusable fixed-template construction | [Individual workflows](individual_workflow/README.md) |
| `ArealMSHBM_Kong2022/` | Dataset adapters and Argon SGE/MATLAB cortex parcellation | [ArealMSHBM](ArealMSHBM_Kong2022/README.md) |
| `jj_tms/` | JJ_TMS preparation, direct template matching and target mapping | [JJ_TMS](jj_tms/README.md) |
| `iowa7t/` | Iowa7T acquisition/run adapters, template exports and rendering | [Iowa7T](iowa7t/README.md) |
| `nih_rtms/` | Argon fMRIPrep and XCP-D preprocessing | [NIH rTMS](nih_rtms/README.md) |
| `visualize/` | Multi-dataset figure suites and specialized renderers | [Visualization](visualize/README.md) |
| `exploratory_beta/` | Exploratory Stanford/JJ_TMS native-space beta extraction, models and figures | [Native beta analysis](exploratory_beta/README.md) |

The numbered roots `core_workflow/` and `individual_workflow/` contain public
`NN_MM_<semantic_name>` entrypoints; shared implementation belongs in their
`lib/` directories. Dataset-specific adapters belong in `scripts/<dataset>/`.
Keep unnumbered launch, audit and migration helpers out of the numbered roots.
Core steps execute through their maintained implementation tree.

## Workflow selection

Group processing follows anatomy (`01`), shared inputs/QC (`02`), cortical
references (`03`), FC or task activation (`04`), group analysis (`05`), then
presentation (`06`/`07`). Rendering can reuse valid saved group intermediates.
Select the dataset, physical-run metadata, preprocessing variant, analysis
roster and output roots explicitly; the core guide describes which stages and
QC contracts are implemented for each branch.

HCP individual labels use family-aware five-fold crossfit with an external
`subject,family_id` map. Non-HCP datasets use the declared immutable HCP151
fixed template through dataset adapters. Details are in the individual guide.
Native-space beta work remains exploratory; its entrypoint is
`scripts/exploratory_beta/run_native_surface_beta.py`.

Use the [core registry](core_workflow/step_registry.json) for group-step and
branch identities, the [artifact lineage](../docs/hipp_parcellation_network/core_workflow_artifact_lineage.md)
for required upstream products, and [output layout](../docs/hipp_parcellation_network/output_layout.md)
for formal result and execution-record roots.

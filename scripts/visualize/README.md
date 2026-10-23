# Hippocampal Visualization

This directory contains the multi-dataset figure suite and specialized renderers.
Run from the repository root with
`/home/yojiao/.conda/envs/py314/bin/python`. Rendering consumes validated analysis
products; it does not establish a new analysis roster or rebuild upstream FC.

## Multi-dataset suite

The maintained suite entrypoint is [run_multidataset_suite.py](run_multidataset_suite.py).
Its [cohort registry](lib/multidataset_cohorts.py) defines exact paths, rosters,
variants and comparison methods. Inspect readiness against current inputs:

```bash
/home/yojiao/.conda/envs/py314/bin/python \
  scripts/visualize/run_multidataset_suite.py --check-readiness
```

| Selector | Content |
|---|---|
| `--version legacy` (default) | HCP, Iowa3T and Stanford cohort suite |
| `--version hcp_jjtms_7net` | HCP/JJ_TMS Kong2022 and Lynch2024 canonical7 comparison |
| `--atlas 7net` (default), `kong2022` | Equivalent canonical7 suite aliases |
| `--signal-branch standard` (default) | Four standard cells: 2mm/4mm × 36P/22P |
| `--signal-branch ctxreg` | Explicit CtxReg 2mm/36P cell |
| `--signal-branch all` | Four standard cells plus CtxReg 2mm/36P, with branch-qualified names |

HCP uses ICA-FIX inputs and has no native 22P/36P distinction. Those grid
selectors distinguish Iowa3T/Stanford denoising; HCP's baseline changes only
with smoothing. The standard suite uses four concurrent grid workers; explicit
signal-branch modes select the cells described above. Account for child
rendering processes under the repository and module concurrency limits.

```bash
# Preview the selected commands after input preflight.
/home/yojiao/.conda/envs/py314/bin/python \
  scripts/visualize/run_multidataset_suite.py --version hcp_jjtms_7net --dry-run

# Render one figure family across the standard grid.
/home/yojiao/.conda/envs/py314/bin/python \
  scripts/visualize/run_multidataset_suite.py --version hcp_jjtms_7net \
  --figures cluster_confidence_stacked

# Explicit Standard/CtxReg selection.
/home/yojiao/.conda/envs/py314/bin/python \
  scripts/visualize/run_multidataset_suite.py --version hcp_jjtms_7net \
  --figures subfield_composition_2col --signal-branch all
```

Omit `--figures` or pass `--figures all` for all participating families.
Use the suite for these renderers; their standalone rendering CLIs are disabled.

| `--figures` key | Module guide |
|---|---|
| `cluster_confidence_stacked` | [Positive cluster profiles](multidataset_cluster_confidence_stacked/README.md) |
| `gradient_subregion_top20_stacked_2col` | [Gradient Top-20 composition](multidataset_gradient_subregion_top20_stacked_2col/README.md) |
| `hipp_cortex_gradient_wta` | [Hippocampal/cortical gradient and WTA](multidataset_hipp_cortex_gradient_wta/README.md) |
| `subfield_composition_2col` | [Subfield composition](multidataset_subfield_composition_2mm_2col/README.md) |
| `workbench_hippocampus_overlay` | [Workbench overlays](multidataset_workbench_hippocampus_overlay/README.md) |
| `wta_normalized_longaxis_stacked_2col` | [WTA long-axis profiles](multidataset_wta_normalized_longaxis_stacked_2col/README.md) |

## Input and output contracts

Group roots use the formal `outputs_<dataset>/work/group_atlas/` layout:
HCP `pfm/`; Iowa/Stanford `pfm/<cohort>/` for 36P and
`pfm_22p/<cohort>/` for 22P. Cortex controls use `pfm_ctxreg` and
`pfm_ctxdrop`. The registry selects exact Lynch and comparison products
separately; presentation tags do not select another scientific input bank.
See [output layout](../../docs/hipp_parcellation_network/output_layout.md).

- Kong2022 gradients and vertex profiles consume the ordered 400-ROI bank;
  canonical7 is an assignment/display taxonomy. AUD merges into SMN for
  canonical7. Cluster assignments distinguish Fisher-z and normalized Top-20;
  Kong ROI-registry WTA uses only `normalized_top20_count`. Lynch direct-network
  WTA uses Fisher-z. Read exact manifest-declared assets for each variant.
- Preflight checks missing, temporary, malformed and mismatched source paths
  before rendering. It does not substitute another cohort, an older overview
  or a different atlas. Current `--check-readiness` output determines availability;
  this guide does not maintain a frozen "all outputs ready" status.
- Saved contrast profiles require the selected source contract. Use
  `--rebuild-contrast-profiles` for an explicit approved rebuild.
  `--iowa36p-source original-denoised` is a caller-selected Iowa36P recovery
  mode; the default is `canonical`.
- Existing surface-panel cache contracts record the selected source manifests,
  labels, geometry, scene/style, grid and output. Only validated matching panels
  can be reused. Missing canonical inputs fail before cache lookup.
- Figure PNGs, statistics, manifests and `panels/` caches use each module's
  declared output paths. Preserve preprocessing, cohort, atlas, smoothing and
  signal-branch identity when selecting products or reusing caches.

Methods, statistical definitions, figure layouts and detailed outputs belong
in each module guide. Scientific branches and immutable reference templates
remain distinct from current participant-QC results.

## Specialized renderers

These have their own CLI and are separate from the six-family suite:

| Module | Guide |
|---|---|
| HCP Standard/CtxDrop/CtxReg comparison | [Cortex proximity controls](hcp_cortex_control_gradient_wta/README.md) |
| Iowa3T dominant subregions and template-matching comparisons | [Template matching](template_matching_dominant_subregions/README.md) |
| Standalone cortical network surface-area analysis | [Cortical network area](cortex_network_area/README.md) |
| Standard cortical atlas views | [Kong/Lynch cortical views](kong2022_standard_lateral/README.md) |
| Negative cluster-confidence analysis | [Negative profiles](multidataset_cluster_confidence_negative_stacked/README.md) |
| Workflow diagrams | [Flowchart](flowchart/README.md) |
| Iowa functional-organization figures | [Iowa organization](iowa_hipp_functional_organization/README.md) |

Use a module's documented `--help` and explicit inputs. Keep specialized
products outside unrelated suite runs.

## Presentation synchronization

[update_cross_dataset_presentation.py](update_cross_dataset_presentation.py)
updates presentations from existing figures. Inspect its `--help` and preview:

```bash
/home/yojiao/.conda/envs/py314/bin/python \
  scripts/visualize/update_cross_dataset_presentation.py \
  --version hcp_jjtms_7net --dry-run
```

Select the intended version and optional `--page` before an authorized update.
The synchronizer validates section identities and preserves image aspect ratios.
Figure readiness and slide counts come from the selected current products and
synchronization preview.

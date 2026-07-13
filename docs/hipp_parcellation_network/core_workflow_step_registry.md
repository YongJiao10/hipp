# Core Workflow Step Registry and IO Matrix

Canonical public entrypoints are under `scripts/core_workflow/`. The numeric prefixes are part of the contract: the first number is the hard pipeline layer, and a later layer must not be required by an earlier layer. Smoke setup is a utility and is not part of the numbered main pipeline.

| Step ID | Purpose | Branches | Entrypoint | Primary Inputs | Primary Outputs |
|---|---|---|---|---|---|
| `01_01_run_hippunfold_t1w_local` | Run HippUnfold | `kong2019`, `kong2022_routea`, `kong2022_routea_17net`, `kong2022_routea_ctxreg`, `kong2022_routea_ctxdrop`, `kong2022_routeb` | `scripts/core_workflow/01_01_run_hippunfold_t1w_local.py` | BIDS structural/fMRI inputs and HippUnfold config | HippUnfold subject outputs |
| `01_02_check_hippunfold_outputs` | Validate HippUnfold outputs | `kong2019`, `kong2022_routea`, `kong2022_routea_17net`, `kong2022_routea_ctxreg`, `kong2022_routea_ctxdrop`, `kong2022_routeb` | `scripts/core_workflow/01_02_check_hippunfold_outputs.py` | HippUnfold outputs | Integrity reports / validation status |
| `02_01_prepare_group_inputs` | Prepare shared surfaces, tSNR metadata, and group templates | `kong2019`, `kong2022_routea`, `kong2022_routea_17net`, `kong2022_routea_ctxreg`, `kong2022_routea_ctxdrop`, `kong2022_routeb` | `scripts/core_workflow/02_01_prepare_group_inputs.py` | Validated HippUnfold outputs and fMRI volumes | Shared input roots, templates, prep manifest |
| `02_02_build_qc_manifest` | Build QC-pass cohort manifest | `kong2019`, `kong2022_routea`, `kong2022_routea_17net`, `kong2022_routea_ctxreg`, `kong2022_routea_ctxdrop`, `kong2022_routeb` | `scripts/core_workflow/02_02_build_qc_manifest.py` | HippUnfold Dice TSV and tSNR gate summaries | QC pass/excluded manifests and QC tables |
| `02_03_build_lynch2024_nsi_manifest` | Build NSI-filtered Lynch2024 cohort manifest | `lynch2024_20net`, `lynch2024_8net` | `scripts/core_workflow/02_03_build_lynch2024_nsi_manifest.py` | Base QC subject manifest and NSI TSV/CSV with `subject,nsi` | Pass manifest, excluded TSV, and run manifest under `outputs_hcp/inputs/hcp/lynch2024_fast_pfm_t1only/manifests/` |
| `03_01_build_kong2022_reference` | Build shared Kong2022 cortex references | `kong2022_routea`, `kong2022_routeb` | `scripts/core_workflow/03_01_build_kong2022_reference.py` | QC subject list and cortex assets | Per-subject Kong2022 reference arrays |
| `03_02_build_kong2022_routea_reference` | Build RouteA-specific cortex references plus compact 17-net summaries | `kong2022_routea`, `kong2022_routea_17net` | `scripts/core_workflow/03_02_build_kong2022_routea_reference.py` | QC subject list, concat dtseries, and generated flat RouteA ArealMSHBM label outputs (`sub-{sid}/lh_labels.txt`, `sub-{sid}/rh_labels.txt`) | RouteA-compatible reference arrays |
| `03_04_run_lynch2024_fastans_pfm` | Generate official FASTANS Lynch2024 hard-label PFM assets | `lynch2024_20net`, `lynch2024_8net` | `scripts/core_workflow/03_04_run_lynch2024_fastans_pfm.py` | NSI pass subject list, FASTANS Lynch2024 priors, cortex dtseries/time-series inputs, and Workbench | Per-subject hard-label PFM dlabel/dscalar, label vector, hard-label component time series, component TSV, and manifest under `outputs_hcp/work/cortex_pfm/sub-<id>/lynch2024/` |
| `03_03_build_lynch2024_reference` | Build Lynch2024 hard-label PFM component references plus exact 20-network and canonical8 summaries | `lynch2024_20net`, `lynch2024_8net` | `scripts/core_workflow/03_03_build_lynch2024_reference.py` | NSI pass subject list and per-subject hard-label cortex PFM component assets under `outputs_hcp/work/cortex_pfm/sub-<id>/lynch2024/` | Shared inputs under `outputs_hcp/inputs/hcp/lynch2024_fast_pfm_t1only/shared_inputs/pfm_nsi043_78/sub-*/reference/{lynch2024_20net,lynch2024_8net}/` |
| `04_00_materialize_cortex_regressed_surface_timeseries` | Materialize local cortex-regressed hippocampal time series | `kong2022_routea_ctxreg` | `scripts/core_workflow/04_00_materialize_cortex_regressed_surface_timeseries.py` | RouteA references, `surface`, subject cortex dtseries/surfaces | `shared_inputs/routea_ctxreg` controlled surfaces and diagnostics |
| `04_01_materialize_kong2022_roi_fc` | Materialize ROI/component vertex-to-atlas FC; Lynch8 uses canonical8 reference timeseries | `kong2022_routea`, `kong2022_routea_17net`, `kong2022_routea_ctxreg`, `kong2022_routea_ctxdrop`, `kong2022_routeb`, `lynch2024_20net`, `lynch2024_8net` | `scripts/core_workflow/04_01_materialize_kong2022_roi_fc.py` | Smoothed hippocampal time series and cortex references | ROI/component FC caches with exact Lynch20 dimensionality and taxonomy evidence where applicable |
| `04_02_materialize_intrinsic_fc` | Materialize intrinsic FC | `kong2019`, `kong2022_routea`, `kong2022_routea_ctxreg`, `kong2022_routeb` | `scripts/core_workflow/04_02_materialize_intrinsic_fc.py` | Smoothed hippocampal time series | Intrinsic FC caches |
| `04_03_materialize_kong2019_fc` | Materialize Kong2019 FC bank | `kong2019` | `scripts/core_workflow/04_03_materialize_kong2019_fc.py` | Kong2019 shared inputs and subject manifest | Kong2019/Kong2019_8net FC caches |
| `05_01_compute_hippomaps_gradients` | Compute HippoMaps-style continuous gradients; Lynch2024 computes only the non-Noise Lynch20 component bank and `lynch2024_8net` reuses it downstream | `kong2022_routea`, `kong2022_routea_17net`, `kong2022_routea_ctxreg`, `kong2022_routea_ctxdrop`, `kong2022_routeb`, `lynch2024_20net` | `scripts/core_workflow/05_01_compute_hippomaps_gradients.py` | Vertex-to-ROI/component FC caches, support masks, group-average surfaces | Aligned full-surface gradients, aligned lambdas/EV fractions, group mean Fisher-z FC, alignment manifest |
| `05_02_run_group_atlas` | Build group atlas outputs with K=2..8 diagnostics, fixed final K=3, and shifted-cosine surface affinity; Lynch2024 scope is `gradient-spectral` and normalized `wta-top` | `kong2019`, `kong2022_routea`, `kong2022_routea_17net`, `kong2022_routea_ctxreg`, `kong2022_routea_ctxdrop`, `kong2022_routeb`, `lynch2024_20net`, `lynch2024_8net` | `scripts/core_workflow/05_02_run_group_atlas.py` | FC caches, group templates, branch controls, optional `--gradient-root` from `05_01`, optional `--profile-surface-root` for cluster-network profile time-series annotation | Group labels, K summaries, assignment status, functional atlas labels, and affinity/connected-graph provenance |
| `05_03_run_spectral_fast_rerun` | Optional rerun orchestration after primary group-atlas construction | `kong2022_routea`, `kong2022_routeb` | `scripts/core_workflow/05_03_run_spectral_fast_rerun.py` | Existing branch roots, manifests, templates | Refreshed spectral outputs and overview-ready group roots |
| `06_01_build_group_overview` | Build per-method group overview figures with stable filenames, visible shifted-cosine titles, and source affinity lineage | `kong2019`, `kong2022_routea`, `kong2022_routea_17net`, `kong2022_routea_ctxreg`, `kong2022_routea_ctxdrop`, `kong2022_routeb`, `lynch2024_20net`, `lynch2024_8net` | `scripts/core_workflow/06_01_build_group_overview.py` | Group-atlas runtime artifacts from `05_02` | Method overview PNGs and summaries with `source_group_manifests`; gradient items also carry affinity name and description |
| `06_02_materialize_group_labels` | Materialize functional-atlas label arrays | `kong2019`, `kong2022_routea`, `kong2022_routea_17net`, `kong2022_routeb` | `scripts/core_workflow/06_02_materialize_group_labels.py` | Assignment variant status JSON | Suffixed default/variant label arrays recorded in `group_atlas_manifest.json` |
| `06_03_build_cluster_cortex_panels` | Build cluster-based cortex/hippocampal significance panels | `kong2019`, `kong2022_routea`, `kong2022_routea_17net`, `kong2022_routea_ctxreg`, `kong2022_routea_ctxdrop`, `kong2022_routeb` | `scripts/core_workflow/06_03_build_cluster_cortex_panels.py` | Group labels, FC summaries, topology assets | Cluster-based cortex-panel products |
| `06_04_render_hippomaps_gradient_products` | Render HippoMaps continuous-gradient products with RouteA-referenced display sign alignment and shared color ranges | `kong2022_routea`, `kong2022_routea_17net`, `kong2022_routea_ctxreg`, `kong2022_routea_ctxdrop`, `kong2022_routeb` | `scripts/core_workflow/06_04_render_hippomaps_gradient_products.py` | `05_01` gradient manifests and arrays | Cross-branch comparable folded/unfolded G1-G3 figures, EV plots, endpoint FC contrast plots, manifest |
| `06_05_render_group_gradient_heatmaps` | Deprecated retired legacy locked-scene heatmap entrypoint; preserved only to emit deprecation guidance. | `kong2022_routea`, `kong2022_routeb` | `scripts/core_workflow/06_05_render_group_gradient_heatmaps.py` | none (exits deprecated) | none |
| `06_06_render_confidence_maps` | Render gradient-spectral confidence maps | `kong2022_routea`, `kong2022_routea_ctxreg` | `scripts/core_workflow/06_06_render_confidence_maps.py` | `05_04` confidence arrays | Folded/unfolded confidence PNGs |
| `06_07_render_ctxdrop_neighbor_counts` | Render RouteA local cortex-neighbor count summaries | `kong2022_routea_ctxreg`, `kong2022_routea_ctxdrop` | `scripts/core_workflow/06_07_render_ctxdrop_neighbor_counts.py` | Per-subject `surface/local_cortex_neighbors/hemi_*/neighbor_counts.npy` from ctxreg or ctxdrop | `outputs_hcp/present/local_cortex_neighbors/` group mean/sd/n arrays, TSV, hippocampal surface-overlay PNG, manifest |
| `06_08_render_hcp_subfield_composition` | Render HCP 151-subject weighted_fisherz gradient cluster subfield composition bars | `kong2022_routea` | `scripts/core_workflow/06_08_render_hcp_subfield_composition.py` | `05_05` weighted group-atlas manifests and subfield maps | HCP subfield composition manifest and plots |
| `06_09_plot_cortex_kong2022_mshbm` | Render subject-level Kong2022 MSHBM cortical brain maps | `kong2022_routea` | `scripts/core_workflow/06_09_plot_cortex_kong2022_mshbm.py` | Individual flat label files `lh_labels.txt` and `rh_labels.txt` | GIfTI and CIFTI label files, and composed 2x2 overview PNGs |
| `06_10_render_cluster_roi_strength_panels` | Render full 400-ROI gradient-spectral and normalized WTA cortex templates with shared diverging colorbars | `kong2022_routea`, `kong2022_routea_17net`, `kong2022_routea_ctxreg`, `kong2022_routea_ctxdrop`, `kong2022_routeb` | `scripts/core_workflow/06_10_render_cluster_roi_strength_panels.py` | `cluster_roi_profile.npy` or normalized WTA vertex labels, subject timeseries/references, Kong2022 ROI registry, Workbench cortex assets | Four-row gradient overview or separate canonical8/family17 WTA 4x4 overview PNGs and manifests |
| `07_01_build_group_matrix` | Build maintained presentation index after layer 6 products | `kong2022_routea` by default; `kong2022_routea_17net`, `kong2022_routea_ctxreg`, `kong2022_routea_ctxdrop`, and `kong2022_routeb` optional | `scripts/core_workflow/07_01_build_group_matrix.py` | Canonical group-atlas roots and method overview products | Matrix manifest, README, refreshed overview set |

## Utilities

| Utility ID | Purpose | Entrypoint | Output Root |
|---|---|---|---|
| `smoke_prepare_run` | Prepare an isolated smoke-run command plan; not a numbered pipeline step | `tests/core_workflow/smoke_prepare_run.py` | `outputs_hcp/smoke_runs/<run_id>/` |
| `upgrade_retained_hcp_formal_provenance` | Validate and atomically enrich only the eight retained HCP RouteA formal manifests; never recompute or rewrite binary assets | `scripts/core_workflow/upgrade_retained_hcp_formal_provenance.py` | In-place JSON provenance fields after all-manifest validation |
| `audit_formal_gradient_spectral_release` | Read-only exact nine-group release audit | `scripts/core_workflow/audit_formal_gradient_spectral_release.py` | JSON on stdout and optional `--output-json` path |
| `run_formal_gradient_spectral_refresh` | Fixed `jobs=1`, preflighted, sequential 19-stage formal refresh launcher | `scripts/core_workflow/run_formal_gradient_spectral_refresh.py` | Status, PID, stdout, stderr, and audit JSON under `outputs_hcp/work/group_atlas/logs/` |

The two formal-release utilities keep route applicability in `branches` and
encode their exact shared release scope separately in the machine registry's
`formal_groups` field. The nine ordered `dataset / group_root / atlas`
identities are:

- `outputs_hcp / routea_full_k3_fisherz / kong2022`
- `outputs_hcp / routea_17net_full_k3_fisherz / kong2022`
- `outputs_hcp / routea_weighted_fisherz_full_k3 / kong2022`
- `outputs_hcp / lynch2024_20net_nsi043_full_k3_normalized_wta / lynch2024_20net`
- `outputs_hcp / lynch2024_8net_nsi043_full_k3_normalized_wta / lynch2024_8net`
- `outputs_stanford / routea_hc_full_k3_fisherz / kong2022`
- `outputs_stanford / routea_symp_full_k3_fisherz / kong2022`
- `outputs_stanford / routea_hc_weighted_fisherz_full_k3 / kong2022`
- `outputs_stanford / routea_symp_weighted_fisherz_full_k3 / kong2022`

Machine-readable source of truth: `scripts/core_workflow/step_registry.json`.

The runner, audit, and both registry entries expose these same exact identities.
Formal cohort preflight accepts only unique canonical IDs at the exact sizes
Lynch 78, Stanford HC 91, and Stanford Symp 103. Formal group manifests bind
source provenance to the declared HCP or Stanford root; the audit verifies
source existence/root containment and presentation PNG/title integrity.
The HCP retained-manifest upgrader validates the canonical 151-subject cohort,
shifted-cosine/K/graph contract, exact RouteA roots, assignment assets, and
artifact fingerprints before any atomic JSON replacement. It runs as the first
command inside the existing `hcp_weighted_branch` stage, preserving the approved
19 top-level stage names while preceding weighted copying and all later HCP
overview/audit consumers.
The commit phase stages all eight JSON updates and rolls every manifest back to
its exact original bytes if replacement 2 through 8 fails, leaving no temporary
or backup files. Launcher preflight reuses the same pure prepare/validation
phase before detaching. Formal identity derivation reads actual producer and
otherwise-unrepresented overview commands without filtering through the
expected constant, retaining duplicate producers and unexpected groups so the
registry equality test detects drift.
Artifact-level provenance and required evidence are maintained in
`docs/hipp_parcellation_network/core_workflow_artifact_lineage.md`.

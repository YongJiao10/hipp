# Unified Kember metrics workflow

`KEMBER_METHOD_NATIVE_AREA_WORKFLOW_READY`

This package is the sole active Stanford Kember workflow; HCP 7T is an explicitly separate descriptive reference branch. Its Stanford subject-level methods are
`hcp_template_matching` and `hcp_fixed`; no subject-level clustering is
performed here.
禁止个体聚类。

绝对面积是 Kember 对齐的主指标：它是 HippUnfold native
`surfareacorobl` per-vertex surface-area metric 在 native system labels 内的
mm² 总和。相对面积是辅助指标：
`surface_area_fraction = surface_area_mm2 / total_valid_surface_area_mm2`。

For each method × subject × 2/4 mm smoothing × L/R hemisphere combination:

1. `inputs.py` resolves the consumed 512-source labels, valid mask, rest-fMRI
   time series, source surface, native surface, unfolded registration surfaces,
   and native `surfareacorobl` metric.
2. `native_label_alignment.align_labels_to_native` transfers source labels to
   the native surface with Workbench `ADAP_BARY_AREA`.
3. `surface_area.compute_system_surface_area` sums the supplied native metric
   within native labels. It is the only Kember area aggregation function.
4. `feature_build.compute_kember_metrics` computes functional segregation and
   boundary sharpness from the separate 512-source time-series path and joins
   them to the native area records.
5. `contrasts.run_stable_contrasts` materializes one direct-contrast result,
   one figure set, and one report set from the feature manifest.

The feature manifest records the native metric, consumed native surface, native
labels, transfer provenance, density evidence, vertex counts, and input hashes
together. Geometry-derived triangle-area/3 reconstruction is not a Kember
endpoint. `surface_geometry.py` only loads consumed mesh geometry and builds
adjacency for the functional boundary calculation.

Stable outputs are deliberately unversioned and are overwritten together:

- `work/features/`
- `results/direct_contrasts/`
- `figures/direct_contrasts/`
- `reports/direct_contrasts/`

Example feature command:

~~~bash
/home/yojiao/.conda/envs/py314/bin/python \
  stanford_jjtms_multiscale_hippocampal_exploration/kember_metrics/src/feature_build.py \
  --config stanford_jjtms_multiscale_hippocampal_exploration/kember_metrics/config/analysis_config.json \
  --output-dir stanford_jjtms_multiscale_hippocampal_exploration/kember_metrics/work/features \
  --workbench-command /home/yojiao/FASTANS/workbench/bin_linux64/wb_command
~~~

Example stable contrast command:

~~~bash
/home/yojiao/.conda/envs/py314/bin/python \
  stanford_jjtms_multiscale_hippocampal_exploration/kember_metrics/src/contrasts.py \
  --config stanford_jjtms_multiscale_hippocampal_exploration/kember_metrics/config/analysis_config.json \
  --feature-table stanford_jjtms_multiscale_hippocampal_exploration/kember_metrics/work/features/subject_features.tsv \
  --results-dir stanford_jjtms_multiscale_hippocampal_exploration/kember_metrics/results/direct_contrasts \
  --figures-dir stanford_jjtms_multiscale_hippocampal_exploration/kember_metrics/figures/direct_contrasts \
  --reports-dir stanford_jjtms_multiscale_hippocampal_exploration/kember_metrics/reports/direct_contrasts
~~~

The historical raw-area PNG and PDF artifacts have been removed. The provenance
record remains under `figures/retained_exploratory/`; that historical material
uses the triangle-area field and is not part of the active Kember endpoint. Do
not combine it with `results/direct_contrasts/`.

### Current comprehensive area summary (PNG only)

The active source is `src/plot_area_2mm_comprehensive_summary.py`. It reads
the current native Kember `surface_area_mm2` table, expands all seven configured
rosters including `TIS PTSD` (n=35), and writes the within-hemisphere system
proportion view by default to
`figures/direct_contrasts/area_2mm_comprehensive_summary_proportion.png`. Use
`--analysis-mode covariate` for total-area-adjusted residualized surface area;
`--analysis-mode proportion` is the explicit equivalent of the default. Both
modes retain the same figure layout and effect-size scale.

The figure displays the seven Stanford cohorts as the formal direct-to-NTHC result view and appends `HCP 7T` as a separate reference cohort. The HCP reference uses the maintained family-aware five-fold held-out cross-fit template-matching labels (`hcp_crossfit_template_matching`), not the full HCP template that supplied those labels. Each held-out 512-source label map is transferred with `ADAP_BARY_AREA` and summed against native HippUnfold `surfareacorobl` values. Its six HCP 7T−NTHC effects are descriptive cross-dataset references and are excluded from any current correction family.
The forest panel keeps category y-axis labels on the left hemisphere panel only; the repeated right-panel labels are intentionally hidden so they cannot intrude into the neighboring plot. The GridSpec uses compact row spacing and a reserved left margin for the long category labels. The exported typography is deliberately enlarged across titles, axes, ticks, value labels, significance labels, legend; the right forest panel retains no duplicate ticks. The footer is intentionally omitted to reduce dead space; the area-effect forest uses light-gray horizontal separators between groups instead of an unexplained gray x-grid.

The HCP source and cache paths are declared under `reference_hcp7t` in `config/analysis_config.json`; the generated cache, effect table, and run manifest live under `work/reference_hcp7t_crossfit_area/`. The entry point still writes one PNG only and does not generate or overwrite PDFs.

Run it with:

~~~bash
/home/yojiao/.conda/envs/py314/bin/python \
  stanford_jjtms_multiscale_hippocampal_exploration/kember_metrics/src/plot_area_2mm_comprehensive_summary.py
~~~

This entry point intentionally writes PNG only; it has no PDF output path.

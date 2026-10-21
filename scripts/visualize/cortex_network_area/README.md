# Cortical network surface area

This standalone module computes each network's cortical midthickness surface area as a percentage of the subject's valid cortical area. It keeps the absolute area in mm², Noise, and Unassigned components in the subject table. Kong2022 native 17-network and canonical7 outputs, and Lynch2024 native 20-network and canonical7 outputs, are kept as separate taxonomies.

Use the repository's explicit Python environment. Surface preparation requires
persistent dataset inputs:

```bash
/home/yojiao/.conda/envs/py314/bin/python -m scripts.visualize.cortex_network_area.prepare_iowa_surfaces --repo-root . --work-dir outputs_iowa_jj_tms/inputs/cortex_surfaces/fsLR32k --manifest outputs_iowa_jj_tms/inputs/cortex_surfaces/fsLR32k/surface_manifest.json
```

Consumers must explicitly pass that persistent manifest via
`--iowa-surface-manifest`. Preparation refuses existing manifest or surface
targets; consumers reject `/tmp` entries and validate exact roster and hashes.
This code integration does not authorize rebuilding existing surface products.

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/visualize/cortex_network_area/run_area_analysis.py --variant all --check-readiness --iowa-surface-manifest outputs_iowa_jj_tms/inputs/cortex_surfaces/fsLR32k/surface_manifest.json
/home/yojiao/.conda/envs/py314/bin/python scripts/visualize/cortex_network_area/run_area_analysis.py --variant 22p --iowa-surface-manifest outputs_iowa_jj_tms/inputs/cortex_surfaces/fsLR32k/surface_manifest.json
/home/yojiao/.conda/envs/py314/bin/python scripts/visualize/cortex_network_area/plot_area_figures.py --stats scripts/visualize/cortex_network_area/stats/network_area_statistics.csv
```

`--variant` accepts `22p`, `36p`, or `all`. HCP records resolve to the same ICA-FIX surface and label inputs for both variant labels. Iowa records resolve the explicit `ArealMSHBM_Kong2022_22P`/Lynch 22P roots or the unversioned 36P roots; no branch is substituted when an input is missing. `--check-readiness` writes `input_audit.csv` and an empty metrics/statistics table without calculating areas. A full run fails closed on any missing or incompatible roster input, including extraction failures: it writes the updated audit with empty metric/statistic tables and exits nonzero instead of reporting partial statistics.

The analysis writes `subject_network_area_metrics.csv`, `network_area_statistics.csv`, `input_audit.csv`, and `run_manifest.json` under `stats/`. The independent plotting entry point reads the statistics table and writes PNG forest figures under `figures/`; all comparisons retain every network, using uncorrected two-sided Welch significance symbols (`*` p<0.05, `**` p<0.01, `***` p<0.001; p≥0.05 is blank). BH-FDR q values are recorded as auxiliary values and do not determine the displayed significance symbols.

The fixed rosters are HCP HC 151, Iowa HC 67, Patient 26 (sub-3003 excluded), Pure MDD 7, and Comorbid MDD 19. The six patient-group contrasts plus Iowa HC–HCP HC are represented explicitly. Patient subgroups overlap, and the HC–HC comparison is a cross-dataset baseline comparison rather than an equivalence test.

## Figure layout and independent redraw

Redrawing reads only `stats/network_area_statistics.csv`; it does not rerun analysis or alter statistics. Outputs are PNG only. The unreadable stacked patient overview is retired and is no longer generated. Four individual plates show every network:

- `figures/cortex_network_area_patient_kong_canonical7_forest.png`
- `figures/cortex_network_area_patient_lynch_canonical7_forest.png`
- `figures/cortex_network_area_patient_kong_native17_forest.png`
- `figures/cortex_network_area_patient_lynch_native20_forest.png`

Each patient plate has Iowa HC/HCP HC control rows and Patient/Pure MDD/Comorbid MDD columns. Within each network row, filled circles denote 22P and open diamonds denote 36P, matching the HC baseline figure. Estimates and significance symbols still describe patient-minus-control contrasts within each preprocessing variant; they are not a statistical test of 22P versus 36P. Networks follow the fixed atlas order; all 7/17/20 networks and both control comparisons remain visible, including blank non-significant rows. Network positions and symmetric x-axis limits are shared within a taxonomy. Significance symbols occupy a separate right-hand column aligned with their estimates; p≥0.05 is left blank and exact p values remain in the statistics CSV. Sample sizes appear in panel headers. Colors come directly from `config/hipp_network_style.json`; missing colors raise an error. Shortened network labels affect display only.

`figures/cortex_network_area_hc_baseline_forest.png` pairs 22P (filled circle) and 36P (open diamond) within each network in a four-panel plate, retaining all HC–HC results. Point positions and 95% intervals come directly from the full statistics table.

## Complete Iowa surface input profile

The first run incorrectly treated absent fsLR derivatives as unavailable subjects: Iowa HC 52/67, Patient 24/26, Comorbid MDD 17/19. Native midthickness anatomy and fsLR registration exist for all 17 omitted individuals (HC sub-1028 and sub-1055–1068; patients sub-3012 and sub-3028). This was an input-preparation omission, not a cohort exclusion.

The explicit `iowa_fsLR_complete_roster_v1` profile retains the existing verified derivatives for the other subjects and generates those 17 derivatives using Workbench BARYCENTRIC resampling from each individual's native midthickness via `space-fsLR_desc-reg_sphere.surf.gii` onto the repository FASTANS standard 32k fsLR sphere. It never selects files by existence order. The older `outputs_iowa_jj_tms/inputs/surfaces_32k` cache is not used: its inspected provenance names `desc-reg_sphere` rather than the required fsLR-registered sphere.

```bash
/home/yojiao/.conda/envs/py314/bin/python -m scripts.visualize.cortex_network_area.prepare_iowa_surfaces --repo-root . --work-dir outputs_iowa_jj_tms/inputs/cortex_surfaces/fsLR32k --manifest outputs_iowa_jj_tms/inputs/cortex_surfaces/fsLR32k/surface_manifest.json
/home/yojiao/.conda/envs/py314/bin/python scripts/visualize/cortex_network_area/run_area_analysis.py --variant all --iowa-surface-manifest outputs_iowa_jj_tms/inputs/cortex_surfaces/fsLR32k/surface_manifest.json
/home/yojiao/.conda/envs/py314/bin/python scripts/visualize/cortex_network_area/plot_area_figures.py --stats scripts/visualize/cortex_network_area/stats/network_area_statistics.csv
```

The manifest records all 186 Iowa hemisphere paths, hashes, generating provenance, and target-topology validation. It is explicitly selected by `--iowa-surface-manifest` and recorded in the run manifest. No preprocessing branch is substituted. Analysis rejects missing or changed files; persistent surfaces do not depend on `/tmp` retention. Existing products require separately approved repair rather than an automatic rerun. Workbench vertex-area checks cover a retained and a regenerated surface. Runtime remains py314 plus the existing `/home/yojiao/FASTANS/workbench/bin_linux64/wb_command`; no environment change is required.

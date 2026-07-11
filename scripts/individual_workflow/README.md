# Individual RouteA Gradient K3 and WTA Workflow

This directory runs subject-level RouteA hippocampal products for the 151-subject HCP 7T cohort.

It reuses `scripts/core_workflow` helpers but stays outside the group-level workflow.

## Output Layout

Intermediate and per-combo computation products live under:

```bash
outputs_hcp/work/individual_atlas/
  sub-<id>/
    gradient/
      <2mm|4mm>/<L|R>/
    wta/
      <2mm|4mm>/<L|R>/
  _runs/
```

Per-subject presentation surface overviews live under the canonical present tree:

```bash
outputs_hcp/present/overviews/individual/
  gradient/sub-<id>_gradient_overview.png
  wta/sub-<id>_wta_overview.png
```

Gradient overview adapter rows cover K=2..10 for the K-selection and redundancy diagnostic panels, but final individual gradient labels remain fixed at K=3.

Run and handoff metadata for the computation stage live under:

```bash
outputs_hcp/work/individual_atlas/_runs/
```

## Run Computation

Use the repository Python contract:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/individual_workflow/routea_individual_gradients_wta.py --jobs 4 --resume
```

Dataset-specific runs can override `--subjects-manifest`, `--shared-root`,
`--surface-root`, `--hippunfold-batch-root`, and `--out-root` without changing
the workflow implementation.

Kong2022 network products use the canonical8 network order `Default`, `Visual`, `Somatomotor`, `DorsalAttention`, `VentralAttention`, `Control`, `Auditory`, `Language`; do not sort network names alphabetically when adding assignment variants or overview adapters.

Background launcher:

```bash
bash scripts/individual_workflow/run_routea_individual_gradients_wta_background.sh
```

The launcher uses `setsid nohup`, fixes native thread counts at 1, and starts 4 Python workers, staying below the repository concurrency cap.

## Monitor Computation

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/individual_workflow/01_monitor_individual_run.py
```

The monitor refreshes:

```bash
outputs_hcp/work/individual_atlas/_runs/summaries/combo_status.tsv
```

Each run also writes:

```bash
outputs_hcp/work/individual_atlas/_runs/logs/<run_id>/heartbeat.json
outputs_hcp/work/individual_atlas/_runs/handoff/<run_id>_handoff.md
```

## Build Overviews

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/individual_workflow/02_build_individual_overviews.py --jobs 4 --resume
```

The overview builder reads completed label arrays from `outputs_hcp/work/individual_atlas/` and writes quicklook PNGs plus `overview_status.tsv` and `overview_manifest.json` under `outputs_hcp/present/overviews/individual/`. For `gradient_fisherz`, the cluster-network heatmap and confidence bars are both derived from the same Kong2022 ROI-mean Fisher-z network profile; the direct cluster-to-network timeseries profile is not used for that rendered variant.

## Method Notes

- Route: RouteA only, from `shared_inputs/routea`.
- Subject count: 151 by default.
- Combos: `2mm`, `4mm`, `L`, `R`, for 604 total subject-combos.
- Gradient-spectral uses each subject's own valid mask and `space-corobl_den-512` HippUnfold hippocampal surface.
- Gradient-spectral final K is fixed at 3.
- WTA is single-subject top-positive-ROI network voting. It is not a group one-sample statistical inference result; manifests record `wta_statistical_inference=false`.

## Family-Aware Cross-Fitted Labels

For reviewer-defensible individualized K=3 labels without run-level splitting,
use `routea_crossfit_template_matching.py`. Five family-grouped folds learn
training-only group templates and AP-ordered FC profiles; each held-out subject
is then matched using their full concat FC. HCP production requires an external
`subject,family_id` family map. The explicit `--allow-subject-only-folds`
waiver is recorded and is not the preferred HCP definition.
The maintained source `data/HCP_YA_subjects_2026_04_09_16_01_42.csv` can be
passed directly; its native `Subject` and `Family_ID` headings are accepted
without creating a derived restricted-data file.
Raw family identifiers remain memory-only and are not persisted in workflow
outputs. Resume uses atomic in-progress state plus SHA-256 fingerprints for
inputs and saved artifacts, and subject surfaces are vertex-count validated.

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/individual_workflow/routea_crossfit_template_matching.py --family-map /home/yojiao/hpchome/hipp/data/HCP_YA_subjects_2026_04_09_16_01_42.csv --dry-run
/home/yojiao/.conda/envs/py314/bin/python scripts/individual_workflow/routea_crossfit_template_matching.py --family-map /home/yojiao/hpchome/hipp/data/HCP_YA_subjects_2026_04_09_16_01_42.csv --resume
/home/yojiao/.conda/envs/py314/bin/python scripts/individual_workflow/03_build_crossfit_robustness_overview.py
```

The computation lives under
`outputs_hcp/work/crossfit_template_matching/`; the quantitative five-fold
overview and `robustness_summary.tsv` live under
`outputs_hcp/present/overviews/crossfit_template_matching/`. See
`docs/hipp_parcellation_network/crossfit_template_matching.md` for the
non-circularity argument, artifacts, and interpretation.


## Family17 Gradient Overview Rendering

The individual overview adapter accepts `--network-column family17` for
Kong2022 17-network gradient rendering. The option affects the gradient
`fisherz`, `count`, and `weighted_fisherz` overviews and writes PNG names with
a `gradient_family17_` prefix in the stem so canonical8 outputs are not
overwritten. The adapter records `network_column: family17` in
`group_feature_summary.json`; `fisherz` heatmaps use the ROI-mean Fisher-z
cluster-network profile, while `weighted_fisherz` recomputes network time
series from ROI time series using `n_grayordinates_used` weights before
cluster-to-network correlation. `fisherz` and `weighted_fisherz` heatmaps are
validated against their Fisher-z status rows, and `count` heatmaps use the
same top-N ROI count profile used by the count status rows. Stale or missing
assignment-variant manifest paths fail instead of falling back to same-run-dir
files.

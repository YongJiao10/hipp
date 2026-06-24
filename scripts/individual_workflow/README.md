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

The overview builder reads completed label arrays from `outputs_hcp/work/individual_atlas/` and writes quicklook PNGs plus `overview_status.tsv` and `overview_manifest.json` under `outputs_hcp/present/overviews/individual/`.

## Method Notes

- Route: RouteA only, from `shared_inputs/routea`.
- Subject count: 151 by default.
- Combos: `2mm`, `4mm`, `L`, `R`, for 604 total subject-combos.
- Gradient-spectral uses each subject's own valid mask and `space-corobl_den-512` HippUnfold hippocampal surface.
- Gradient-spectral final K is fixed at 3.
- WTA is single-subject top-positive-ROI network voting. It is not a group one-sample statistical inference result; manifests record `wta_statistical_inference=false`.

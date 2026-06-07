# RouteA Individual Gradient-Spectral K3 and WTA Workflow

This workflow generates subject-level hippocampal products for the 151 HCP 7T subjects.

Public entrypoints live under:

```bash
scripts/individual_workflow/
```

Computation outputs live under:

```bash
outputs_hcp/work/individual_atlas/
```

Presentation surface overviews live under:

```bash
outputs_hcp/present/overviews/individual/
  gradient/sub-<id>_gradient_overview.png
  wta/sub-<id>_wta_overview.png
```

## Scope

- Route: `routea`
- Inputs: `outputs_hcp/inputs/hcp/kong2022/shared_inputs/routea_151`
- Subject manifest: `outputs_hcp/inputs/hcp/kong2022/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt`
- Smoothings: `2mm`, `4mm`
- Hemispheres: `L`, `R`
- Total default workload: 151 subjects x 2 smoothings x 2 hemispheres = 604 subject-combos

## Methods

`gradient_spectral_k3` computes each subject independently. It fits G1-G3 from that subject's Fisher-z vertex-to-Kong2022 ROI FC matrix, then runs spectral clustering with fixed `K=3` using the subject's own valid mask and `space-corobl_den-512` HippUnfold hippocampal surface adjacency.

`wta_top` is a single-subject vertex-level top-positive-ROI network vote. It is not a group one-sample statistical inference result; combo manifests record `wta_statistical_inference=false`.

## Runtime

Use the repository Python contract:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/individual_workflow/routea_individual_gradients_wta.py --jobs 4 --resume
```

The maintained background launcher is:

```bash
bash scripts/individual_workflow/run_routea_individual_gradients_wta_background.sh
```

The launcher uses `setsid nohup`, fixes native numerical thread counts to 1, and starts 4 Python workers, keeping the run below the repository concurrency cap.

## Presentation Overviews

After computation completes, build one overview PNG per subject per task with:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/individual_workflow/02_build_individual_overviews.py --jobs 4 --resume
```

The builder reads `cluster_labels_k3.npy` from `outputs_hcp/work/individual_atlas/gradient_spectral_k3/` and `wta_top_labels.npy` from `outputs_hcp/work/individual_atlas/wta_top/`. It writes `gradient/` and `wta/` PNGs, plus `overview_status.tsv` and `overview_manifest.json`, under `outputs_hcp/present/overviews/individual/`.

## Monitoring and Handoff

Monitor the latest run with:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/individual_workflow/01_monitor_individual_run.py
```

Run state is recorded in:

```bash
outputs_hcp/work/individual_atlas/gradient_spectral_k3/_runs/logs/<run_id>/heartbeat.json
outputs_hcp/work/individual_atlas/gradient_spectral_k3/_runs/summaries/combo_status.tsv
outputs_hcp/work/individual_atlas/gradient_spectral_k3/_runs/handoff/<run_id>_handoff.md
```

Use the handoff file for later agent continuation or failure triage.

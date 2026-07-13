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

The subject-first computation layout is:

```text
outputs_hcp/work/individual_atlas/sub-<id>/
  gradient/<2mm|4mm>/<L|R>/
  wta/<2mm|4mm>/<L|R>/
outputs_hcp/work/individual_atlas/_runs/
```

Presentation surface overviews live under:

```bash
outputs_hcp/present/overviews/individual/
  gradient/sub-<id>_gradient_overview.png
  wta/sub-<id>_wta_overview.png
```

## Scope

- Route: `routea`
- Inputs: `outputs_hcp/inputs/hcp/kong2022/shared_inputs/routea`
- Subject manifest: `outputs_hcp/inputs/hcp/kong2022/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt`
- Smoothings: `2mm`, `4mm`
- Hemispheres: `L`, `R`
- Total default workload: 151 subjects x 2 smoothings x 2 hemispheres = 604 subject-combos

## Methods

`gradient_spectral_k3` computes each subject independently. It fits G1-G3 from that subject's Fisher-z vertex-to-Kong2022 ROI FC matrix, then runs spectral clustering with fixed `K=3` using the subject's own valid mask and `space-corobl_den-512` HippUnfold hippocampal surface adjacency.

`wta_top` is a single-subject vertex-level top-positive-ROI network vote. It is not a group one-sample statistical inference result; combo manifests record `wta_statistical_inference=false`.

Kong2022 network labels are always emitted in canonical order, not alphabetical order: `Default`, `Visual`, `Somatomotor`, `DorsalAttention`, `VentralAttention`, `Control`, `Auditory`, `Language`. This order must match profile columns, assignment status rows, label maps, and rendered confidence/heatmap panels.

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

The builder reads `cluster_labels_k3.npy` and `wta_top_labels.npy` from each subject's `gradient/` and `wta/` directories under `outputs_hcp/work/individual_atlas/sub-<id>/`. For gradient overviews, the adapter writes K=2..8 diagnostic rows for the core renderer so the K-selection and redundancy panels show the maintained curve, while final labels remain fixed at K=3. The `gradient_fisherz` heatmap and confidence bars both use the same Kong2022 ROI-mean Fisher-z network profile. It writes both gradient assignment variants under `gradient/` using filename suffixes (`*_gradient_fisherz_overview.png` and `*_gradient_count_overview.png`) and the WTA overview under `wta/`, plus `overview_status.tsv` and `overview_manifest.json`, under `outputs_hcp/present/overviews/individual/`. Use `--methods count` or `--methods gradient_count` to backfill only the count overview for selected subjects.

## Monitoring and Handoff

Monitor the latest run with:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/individual_workflow/01_monitor_individual_run.py
```

Run state is recorded in:

```bash
outputs_hcp/work/individual_atlas/_runs/logs/<run_id>/heartbeat.json
outputs_hcp/work/individual_atlas/_runs/summaries/combo_status.tsv
outputs_hcp/work/individual_atlas/_runs/handoff/<run_id>_handoff.md
```

Use the handoff file for later agent continuation or failure triage.


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

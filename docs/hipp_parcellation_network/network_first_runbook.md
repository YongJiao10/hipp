# Network-First Runbook

## Branches

- `network-gradient`
- `network-prob-cluster`
- `network-prob-cluster-nonneg`
- `network-prob-soft`
- `network-prob-soft-nonneg`
- `network-wta`
- `network-spectral`

## Output Roots

This worktree writes new results only to the roots below.

```text
outputs/hipp_functional_parcellation_network/
present_network/
```

## Input Source Policy

All inputs must be present under `data/` and all outputs are written to `outputs_migration/`. There is no upstream fallback; missing inputs will cause an explicit error.

## K Selection Mode

Always set `--k-selection-mode` explicitly to avoid ambiguity:

- `mainline`: current production rule (best-ARI-within-0.02 and min cluster fraction guard).
- `experimental`: newer test rule (local-minimum + 1-SE + non-triviality).

## Network Definition

This worktree uses the true direct network-first path:

```text
cortex ROI-component timeseries
  -> canonical network merge
  -> Noise exclusion
  -> cortex canonical network timeseries
  -> direct hippocampal vertex-to-network FC
```

The atlas-to-canonical merge is defined in [cross_atlas_network_merge.json](/Users/jy/Documents/HippoMaps-network-first/config/cross_atlas_network_merge.json).

## Expected Final Figure Count

```text
7 branches x 3 atlases x 3 subjects = 63 overview images
(network-wta and network-spectral each count as 1 branch)
```

## Smoke Run

```bash
source /opt/miniconda3/bin/activate py314
python scripts/experiments/hipp_functional_parcellation_network/run_batch.py \
  --branches network-gradient network-prob-cluster network-prob-cluster-nonneg network-prob-soft network-prob-soft-nonneg network-wta network-spectral \
  --atlases lynch2024 \
  --subjects 100610 \
  --k-selection-mode mainline \
  --resume-mode force \
  --retain-level render \
  --cleanup-level none \
  --clear-present
```

## Full Batch Run

```bash
source /opt/miniconda3/bin/activate py314
python scripts/experiments/hipp_functional_parcellation_network/run_batch.py \
  --branches network-gradient network-prob-cluster network-prob-cluster-nonneg network-prob-soft network-prob-soft-nonneg network-wta network-spectral \
  --atlases lynch2024 hermosillo2024 kong2019 \
  --subjects 100610 102311 102816 \
  --k-selection-mode mainline \
  --resume-mode resume \
  --retain-level render \
  --cleanup-level none \
  --clear-present
```

# Network-First Runbook

## Branches

- `network-gradient`
- `network-prob-cluster`
- `network-prob-cluster-nonneg`
- `network-prob-soft`
- `network-prob-soft-nonneg`
- `network-wta`

## Output Roots

This worktree writes new results only to the roots below.

```text
outputs/hipp_functional_parcellation_network/
present_network/
```

## Input Source Policy

If local inputs are missing in this worktree, the network-first scripts automatically read upstream inputs from `/Users/jy/Documents/HippoMaps` as a read-only source while keeping all new outputs isolated here.

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
6 branches x 3 atlases x 3 subjects = 54 overview images
```

## Smoke Run

```bash
source /opt/miniconda3/bin/activate py314
python scripts/experiments/hipp_functional_parcellation_network/run_batch.py \
  --branches network-gradient network-prob-cluster network-prob-cluster-nonneg network-prob-soft network-prob-soft-nonneg network-wta \
  --atlases lynch2024 \
  --subjects 100610 \
  --resume-mode force \
  --retain-level render \
  --cleanup-level none \
  --clear-present
```

## Full Batch Run

```bash
source /opt/miniconda3/bin/activate py314
python scripts/experiments/hipp_functional_parcellation_network/run_batch.py \
  --branches network-gradient network-prob-cluster network-prob-cluster-nonneg network-prob-soft network-prob-soft-nonneg network-wta \
  --atlases lynch2024 hermosillo2024 kong2019 \
  --subjects 100610 102311 102816 \
  --resume-mode resume \
  --retain-level render \
  --cleanup-level none \
  --clear-present
```

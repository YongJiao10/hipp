# Network-First Runbook

## Branches

- `network-gradient`
- `network-prob-cluster`
- `network-prob-cluster-nonneg`
- `network-prob-soft`
- `network-prob-soft-nonneg`
- `network-wta`
- `network-spectral`
- `network-spectral-nonneg`
- `intrinsic-spectral`
- `intrinsic-spectral-nonneg`

## Output Roots

This worktree writes new results only to the roots below.

```text
mainline output root      outputs_migration/hipp_functional_parcellation_network/
mainline present root     present_network_migration/
experimental output root  outputs_experimental/hipp_functional_parcellation_network/
experimental present root present_network_experimental/
```

## Input Source Policy

All inputs must be present under `data/`.

- Mainline runs write to `outputs_migration/` + `present_network_migration/`.
- Experimental runs write to `outputs_experimental/` + `present_network_experimental/`.

There is no upstream fallback; missing inputs will cause an explicit error.

## Canonical Launchers (Mandatory)

Use fixed launcher scripts instead of hand-writing `run_batch.py` mode/output arguments:

- `scripts/hipp_parcellation_network/run_mainline.sh`
- `scripts/hipp_parcellation_network/run_experimental.sh`

Both launchers lock:

- `--k-selection-mode`
- `--run-split-mode=runwise`
- `--out-root`
- `--present-dir`

If any of these flags are passed manually, launchers fail fast to prevent accidental mode/output mix-ups.

Mode definitions:

- `mainline`: current production rule (best-ARI-within-0.02 and min cluster fraction guard).
- `experimental`: newer test rule (local-minimum + 1-SE + non-triviality).

## Network Definition

This worktree uses a shared network-first upstream with two branch feature families:

```text
cortex ROI-component timeseries
  -> canonical network merge
  -> Noise exclusion
  -> cortex canonical network timeseries
  -> direct hippocampal vertex-to-network FC (shared cache, Pearson r for network-* branches)
  -> direct hippocampal vertex-to-vertex FC (shared cache, Pearson r for intrinsic-* branches)
  -> branch-specific Fisher z transform before downstream modeling
```

The atlas-to-canonical merge is defined in [cross_atlas_network_merge.json](/Users/jy/Documents/HippoMaps-network-first/config/cross_atlas_network_merge.json).

## Expected Final Figure Count

```text
10 branches x 3 atlases x 3 subjects = 90 overview images
(network-wta, network-spectral, network-spectral-nonneg, intrinsic-spectral, and intrinsic-spectral-nonneg each count as 1 branch)
```

## Smoke Run

```bash
scripts/hipp_parcellation_network/run_mainline.sh \
  --branches network-gradient network-prob-cluster network-prob-cluster-nonneg network-prob-soft network-prob-soft-nonneg network-wta network-spectral network-spectral-nonneg intrinsic-spectral intrinsic-spectral-nonneg \
  --atlases lynch2024 \
  --subjects 100610 \
  --resume-mode force \
  --retain-level render \
  --cleanup-level none \
  --clear-present
```

## Full Batch Run

```bash
scripts/hipp_parcellation_network/run_mainline.sh \
  --branches network-gradient network-prob-cluster network-prob-cluster-nonneg network-prob-soft network-prob-soft-nonneg network-wta network-spectral network-spectral-nonneg intrinsic-spectral intrinsic-spectral-nonneg \
  --atlases lynch2024 hermosillo2024 kong2019 \
  --subjects 100610 102311 102816 \
  --resume-mode resume \
  --retain-level render \
  --cleanup-level none \
  --clear-present
```

## Full Batch Run (Experimental)

```bash
scripts/hipp_parcellation_network/run_experimental.sh \
  --branches network-gradient network-prob-cluster network-prob-cluster-nonneg network-prob-soft network-prob-soft-nonneg network-wta network-spectral network-spectral-nonneg intrinsic-spectral intrinsic-spectral-nonneg \
  --atlases lynch2024 hermosillo2024 kong2019 \
  --subjects 100610 102311 102816 \
  --resume-mode resume \
  --retain-level render \
  --cleanup-level none \
  --clear-present
```

## Single-Subject Note

- `scripts/hipp_parcellation_network/run_subject.py` now auto-runs `summarize_outputs.py` by default, so direct single-subject runs also emit:
  - `hipp_functional_parcellation_network_overview.png`
  - `summary_manifest.json`
- Use `--skip-summary` only when you intentionally want to defer summary generation.

## Group Prior + Fast-PFM Downstream

This downstream pipeline consumes existing spectral branch subject outputs and shared `_shared` hippocampal timeseries caches. It does not rerun `run_subject.py`.

Script:

- `scripts/hipp_parcellation_network/run_group_prior_fastpfm.py`

Current implementation scope:

- Input branches supported: `network-spectral`, `network-spectral-nonneg`, `intrinsic-spectral`, `intrinsic-spectral-nonneg`
- Group `K` rule: `mean-instability-1se`
- Group `K` unit: each `branch x atlas x smoothing x hemi`
- Prior form: `K`-cluster dominant group prior (`prior_matrix: K x N_vertex`)
- Individual inference: Fast-PFM-style soft map (`scores_prob: K x N_vertex`)
- Render mode: locked native scene, `--layout 1x2` (ventral panel extraction)

Group `K` rule details:

1. Aggregate subject `instability_mean` by `K`.
2. Keep local minima only.
3. Keep `K` values within `1-SE` of the best instability point.
4. Keep `K` values passing `min_parcel_pass_rate` (two-decimal comparison for small-`N` stability, so `2/3` counts as `0.67`).
5. Choose the smallest surviving `K`.

Canonical run command:

```bash
/opt/miniconda3/envs/py314/bin/python scripts/hipp_parcellation_network/run_group_prior_fastpfm.py \
  --branches network-spectral network-spectral-nonneg intrinsic-spectral intrinsic-spectral-nonneg \
  --atlases lynch2024 kong2019 \
  --subjects 100610 102311 102816 \
  --smoothings 2mm 4mm \
  --group-k-rule mean-instability-1se \
  --min-parcel-pass-rate 0.67 \
  --views ventral \
  --layout 1x2
```

If intrinsic spectral subject outputs are not present locally, run only available branches:

```bash
/opt/miniconda3/envs/py314/bin/python scripts/hipp_parcellation_network/run_group_prior_fastpfm.py \
  --branches network-spectral network-spectral-nonneg \
  --atlases lynch2024 kong2019 \
  --subjects 100610 102311 102816 \
  --smoothings 2mm 4mm \
  --views ventral \
  --layout 1x2
```

Downstream output root:

```text
outputs_migration/hipp_group_prior_fastpfm/<branch>/<atlas>/<smoothing>/
  group_k_selection.tsv
  group_k_selection.json
  priors/group_prior_<branch>_<atlas>_<smoothing>_hemi-{L|R}.pickle
  template/workbench_assets/*
  template/renders/*
  group_prior_manifest.json
  individual_soft_maps/sub-<subject>/*
  individual_softmap_manifest.json
```

Failure policy:

- No silent fallback.
- Missing required inputs fail fast during preflight with full absolute missing-path list.
- Missing `K`-specific assets fail with explicit file path.

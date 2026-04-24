# Scripts Layout

This directory is now organized by responsibility rather than keeping every entrypoint at the top level.

- `common/`
  Shared numerical utilities and reusable helpers used across pipelines.
- `cortex/`
  Cortex PFM derivation, ROI-component extraction, and cortex-specific rendering.
- `workbench/`
  Workbench scene capture, legend composition, and native-surface rendering helpers.
- `hipp_parcellation_network/`
  Main hippocampal functional parcellation pipeline (network-first). Entry points: `run_batch.py`, `run_subject.py`, `summarize_outputs.py`.

Top-level `scripts/` still contains broad pipeline entrypoints and wrappers that are shared across multiple flows, such as `wb_command`, `activate_hippo2.sh`, `run_hippomaps_pipeline.py`, and post-HippUnfold orchestration scripts. `run_hippunfold_local.sh` defaults to GPU-backed nnUNet inference and prefers `mamba` as the Snakemake conda frontend when `--use-conda` is enabled.

## Background Rerun Helper

- `run_remaining_hippunfold_background.sh`
  - Finds subjects under `data/hippunfold_input/sub-*` that are not yet completed (based on strict output integrity, not just log tail).
  - Starts a detached manager process (default `4`-parallel) and retries each remaining subject with `--rerun-incomplete` until success or retry limit.
  - Uses `check_hippunfold_output_integrity.py` to require the full `rule all` output set (83 final outputs per subject in current pipeline) before marking a subject complete.
  - Writes manager logs and per-subject logs to:
    - `outputs_migration/dense_corobl_batch/_batch_logs/manager_remaining_<timestamp>_conc4.log`
    - `outputs_migration/dense_corobl_batch/_batch_logs/sub-<id>_remaining_<timestamp>.log`
  - For a brand-new custom `--batch-root`, pre-create `sub-*/hippunfold` directories first (one per subject), otherwise some runs may fail early before Snakebids can create nested output paths.
  - For parallel reruns, do not share one disposable runtime cache root across workers. Use per-subject cache roots (or otherwise isolate cache roots) to avoid cache cleanup races.
  - If using a custom `--batch-root`, seed `--manifest-cache` (or copy `output_integrity_manifest.json`) before first dry-run. The integrity checker cannot auto-build a manifest from an empty fresh root.
- `run_hippunfold_local_t2w_t1reg_isolated_cache.sh`
  - Wrapper for recommended mode (`T2w + --t1-reg-template`) with safe parallel behavior.
  - Uses per-subject disposable runtime workdir (`HIPPUNFOLD_CACHE_ROOT`) to avoid cross-worker cleanup races.
  - Uses persistent shared asset store for heavy reusable downloads (templates + nnUNet model):
    - `runtime/hippunfold_asset_store_shared/hippunfold_assets`
    - `runtime/hippunfold_asset_store_shared/runtime_source_assets`
  - Note: HippUnfold's environment variable name is still `HIPPUNFOLD_CACHE_DIR`, but in this wrapper it points to the shared asset store path (not disposable per-run workdir).
- `run_hippunfold_local_t1w_isolated_cache.sh`
  - Wrapper for `T1w-only` mode with the same safe parallel behavior.
  - Uses per-subject disposable runtime workdir (`HIPPUNFOLD_CACHE_ROOT`) and the same persistent shared asset store.
  - Sets `HIPPUNFOLD_MODALITY=T1w` and does not enable `--t1-reg-template`.

Example:

```bash
scripts/run_remaining_hippunfold_background.sh --concurrency 4
```

Recommended isolated rerun example:

```bash
scripts/run_remaining_hippunfold_background.sh \
  --concurrency 4 \
  --batch-root outputs_experiments/hippunfold_recommended_t2w_t1reg_20260422/dense_corobl_batch \
  --runner scripts/run_hippunfold_local_t2w_t1reg_isolated_cache.sh

T1-only isolated rerun example:

```bash
scripts/run_remaining_hippunfold_background.sh \
  --concurrency 4 \
  --batch-root outputs_experiments/hippunfold_recommended_t1w_20260423/dense_corobl_batch \
  --runner scripts/run_hippunfold_local_t1w_isolated_cache.sh
```
```

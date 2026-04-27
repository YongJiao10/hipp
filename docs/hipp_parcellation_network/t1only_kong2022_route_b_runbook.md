# T1only Kong2022 Route B Runbook

## Scope

This runbook is the canonical path for the T1only HippUnfold branch:

- HippUnfold batch root: `outputs_experiments/hippunfold_recommended_t1w_20260423/dense_corobl_batch/`
- Route root: `outputs_experiments/hippunfold_recommended_t1w_20260423/kong2022_route_b/`
- Shared T1only group inputs: `outputs_experiments/hippunfold_recommended_t1w_20260423/kong2022_route_b/hipp_group_inputs_shared/`
- Group atlas outputs: `outputs_experiments/hippunfold_recommended_t1w_20260423/kong2022_route_b/hipp_group_merged_canonical_atlas/`
- Presentation overview outputs: `present_group_t1only/`
- Cluster-ROI cortex runtime outputs (full/deep): `present_group_t1only/runtime/`
- Cluster-ROI cortex viewer outputs (shallow/quicklook): `present_group_t1only/cluster_roi_cortex/`

The old `manifests/hcp_7t_hippocampus_fmri_qc_pass_153.txt` is not valid for this branch. T1only must derive a new pass manifest from its own HippUnfold Dice QC and its own hippocampal surface tSNR QC.

## Non-Reuse Rule

Kong2022 in this run uses Route B, the public 400-ROI dlabel:

- `external/Schaefer2018_Kong2022/Schaefer2018_400Parcels_Kong2022_17Networks_order.dlabel.nii`

Do not reuse T2only cortex PFM, Kong2019 reference outputs, or old `reference/kong2019` run summaries. Route B computes its own cortex tSNR gate directly from each subject's concat dtseries and extracts Kong2022 ROI/canonical-8 timeseries from the public dlabel.

## Command

Always launch with `setsid` so the pipeline runs in its own session (PID = PGID = SID).
This enables clean process-group kill that cascades to all workers and `wb_command` children:

```bash
setsid /home/yojiao/.conda/envs/py314/bin/python \
  scripts/hipp_parcellation_network/run_t1only_kong2022_route_b_pipeline.py \
  --jobs 4 \
  --resume >> /tmp/pipeline_restart.log 2>&1 &
echo "PID=$!"
```

To stop cleanly (kills orchestrator + workers + all wb_command subprocesses in one shot):

```bash
kill -- -<PID>   # PID == PGID when launched with setsid
```

**Never** `kill <PID>` alone — that orphans the ProcessPoolExecutor workers and their
`wb_command` children, which continue consuming CPU until manually hunted down.

The repository-wide concurrency cap is 10. Keep `--jobs` at or below 10.
**Recommended: `--jobs 4`** — see wb_command thread note below.

Useful restarts:

```bash
setsid /home/yojiao/.conda/envs/py314/bin/python \
  scripts/hipp_parcellation_network/run_t1only_kong2022_route_b_pipeline.py \
  --from-step kong2022-reference --jobs 4 --resume >> /tmp/pipeline_restart.log 2>&1 &

setsid /home/yojiao/.conda/envs/py314/bin/python \
  scripts/hipp_parcellation_network/run_t1only_kong2022_route_b_pipeline.py \
  --from-step group-atlas --jobs 4 --resume >> /tmp/pipeline_restart.log 2>&1 &

setsid /home/yojiao/.conda/envs/py314/bin/python \
  scripts/hipp_parcellation_network/run_t1only_kong2022_route_b_pipeline.py \
  --from-step cluster-roi-cortex --jobs 4 --resume >> /tmp/pipeline_restart.log 2>&1 &
```

## wb_command Thread Warning

`wb_command -metric-smoothing` defaults to **one thread per CPU core** via OpenMP.
On this host that means **128 threads per wb_command process**.
With `--jobs 4`, that is 4 × 128 = 512 threads, saturating the shared machine and
causing ~10× slowdown compared to capped threading (measured: 80 min/batch → 5 min/batch).

**The fix is already applied in `prepare_t1only_group_inputs.py`**: `run_smoothing()` passes
`OMP_NUM_THREADS=2` in the subprocess environment. Do not remove this.

If you ever call `wb_command -metric-smoothing` directly or from a new script, always prefix:

```bash
OMP_NUM_THREADS=2 wb_command -metric-smoothing ...
```

## Runtime Counting Rule

When monitoring `wb_command` concurrency, count only processes with `nlwp > 1`:

```bash
ps -eo pid,pgid,pcpu,nlwp,comm | grep wb_command | awk '$3 > 0'
```

- `nlwp=128` with high `%cpu` = active compute worker  
- `nlwp=1` with `%cpu=0` = launcher shell, do not count

Total expected CPU with `--jobs 4` and `OMP_NUM_THREADS=2`: ~4 × 200% = ~800%.

## Stage Contract

1. `prepare`
   - Checks Dice files under T1only HippUnfold `qc/`.
   - Samples `rsfmri_volume/<sid>/func_uncompressed.nii` onto T1only `space-corobl_den-512` hippocampal surfaces.
   - Builds hippocampal tSNR gates with `tSNR=10000/std`, threshold `25`.
   - Writes `manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_<N>.txt` and `..._latest.txt`.
   - Builds T1only group-average midthickness surfaces from pass subjects only.
   - Generates `surface/2mm` and `surface/4mm` func.gii plus `*_timeseries.npy` via
     `wb_command -metric-smoothing` with `OMP_NUM_THREADS=2` (hardcoded in `run_smoothing()`).
   - All sub-stages (raw/tSNR/QC/smooth/group-average) are resume-safe: re-running with
     `--resume` skips already-completed subjects and skips QC manifest generation if the
     manifest file already exists.

2. `kong2022-reference`
   - Extracts cortex ROI timeseries from the Kong2022 dlabel.
   - Computes cortex tSNR masks within `reference/common/tsnr_gate/` (atlas-invariant upstream).
   - Uses explicit/equal 4-run slicing from the concat dtseries; the current HCP 7T concat input is expected to be `3600 = 4 x 900`.
   - Stores full-length atlas arrays once under `reference/kong2022/` (`cortex_roi_parcel_timeseries.npy`, `cortex_canonical_network_timeseries.npy`) and stores shared run boundaries in `reference/common/runs/run-*/reference_summary.json` instead of duplicating run timeseries arrays.
   - Legacy `reference/kong2022/{tsnr_gate,runs}` content is migrated to `reference/common/` and removed.

3. `kong2022-fc`
   - Materializes hippocampal vertex-to-Kong2022-400ROI Pearson FC for `2mm` and `4mm`, both hemispheres.

4. `intrinsic-fc`
   - Materializes atlas-invariant hippocampal vertex-to-vertex intrinsic FC for `2mm` and `4mm`, both hemispheres.

5. `group-atlas`
   - Runs three methods for atlas `kong2022` by default: `group-intrinsic-spectral`, `group-network-top`, `group-network-all`.
   - `group-network-spectral` remains opt-in only.
   - Uses coverage threshold `0.80` and the T1only group-average surfaces.
   - The 8 method/smoothing/hemi combos are now dispatched in parallel by `run_merged_canonical_group_atlas.py --jobs`.
   - Pipeline forwards `--jobs` to group-atlas but caps combo parallelism at 4 (`group_atlas_jobs=4`).
   - If a combo has no valid `K` (or no `K` survives constraints), that combo is marked skipped in `group_atlas_manifest.json` (`status=skipped_*`) and the remaining combos continue.
   - Per-combo logs are written to `hipp_group_merged_canonical_atlas/_combo_logs/`.
   - Spectral methods support `--k-selection-mode redundancy-aware` with global preprocessing:
     - `--feature-denoise-mode {none,pca}`
     - `--feature-pca-var`
     - `--feature-pca-max-components`
     - `--spatial-smooth-iters`
     - `--spatial-smooth-alpha`
   - Redundancy-aware gates:
     - stability: `group_ari_mean >= 0.85` and `run_aware_ari_mean >= 0.90`
     - non-redundancy: `max_profile_similarity <= 0.90` and `redundant_pair_count == 0`
     - underfit gain: reject `K` when actionable `K+1` adds profile gain `>= 0.10`
   - Overview `K` diagnostics now expose reason tags per rejected `K`: `underfit`, `redundant`, `unstable`, `nontrivial_fail`.

### Redundancy-Aware Preset Sweep (Spectral Only)

Use one global preset for all methods/smoothings/hemis; do not tune per-combo. Ordered preset grid:

1. `baseline`: `none`, `iters=0`
2. `pca95`: `pca_var=0.95`, `iters=0`
3. `pca95_smooth2`: `pca_var=0.95`, `iters=2`, `alpha=0.35`
4. `pca90_smooth2`: `pca_var=0.90`, `iters=2`, `alpha=0.35`
5. `pca90_smooth4`: `pca_var=0.90`, `iters=4`, `alpha=0.35`
6. `pca85_smooth2_alpha50`: `pca_var=0.85`, `iters=2`, `alpha=0.50`

Target combos for this sweep:

- methods: `group-network-spectral`, `group-intrinsic-spectral`
- atlas: `kong2022`
- smoothings: `2mm`, `4mm`
- hemis: `L`, `R`
- coverage: `0.80`

Success criterion: all 8 combos choose `K=3` while passing the fixed gates above. If none pass, stop and report the failed preset table (do not relax thresholds ad hoc).

Reference command:

```bash
/home/yojiao/.conda/envs/py314/bin/python \
  scripts/hipp_parcellation_network/run_redundancy_aware_preset_sweep.py \
  --jobs 4
```

6. `present`
   - Builds overview figures in `present_group_t1only/`.
   - Default present methods in this pipeline are `group-intrinsic-spectral`, `group-network-top`, `group-network-all`.
   - For clusterless methods (`group-network-top`, `group-network-all`), the group overview left pane now contains only `vertex assignment counts`.

7. `cluster-roi-cortex`
   - Runs independent cortex/hippomaps figure generation. All outputs go to a single root: `present_group_t1only/runtime/`
   - Merged-hemisphere overview PNGs are written directly under `present_group_t1only/` for quick access.
   - Clusterless methods (`group-network-top`, `group-network-all`) merge `2mm+4mm` into one file per method: `cortex_<atlas>_<method>_overview.png`.
   - Cluster methods keep per-smoothing files: `cortex_<atlas>_<method>_<smoothing>_overview.png`.
   - Kept separate from overview canvas (no layout merge into `*_group_overview.png`).
   - Uses one-sided one-sample t-tests with BH-FDR (`q < 0.05`) on ROI edges.
   - Intrinsic cluster mode: `Sig-only`, capped at top 20 significant ROI per sign (positive/negative) per cluster.
   - Clusterless modes (`group-network-top`, `group-network-all`): no cluster concept; render global TopN positive + TopN negative connections (`2N` total max) with cortex ROI + hippocampal surface vertex map (not long-axis strip), plus right-side legend styled to match intrinsic panels.
   - `runtime/` stores per-cluster figures, combo manifests, and the quicklook index manifest.
   - Overview collage composes panels from cluster figures in `runtime/` (no auto-delete).
   - Default method set for this stage is now `group-intrinsic-spectral`, `group-network-top`, `group-network-all` (network-spectral remains opt-in).
   - Rendering uses only the first scene view (single 4-panel cortex layout), not multi-view expansion.
   - Each cluster figure includes network-color legend (not ROI-name legend) and explicit sign-coding note (`+` saturated network color, `-` light tint of same network color).
   - Merged overview panels annotate per-panel ROI counts as `n=total (+pos/-neg)`.

## Assignment Rule Update

- `group-intrinsic-spectral`: cluster network assignment default is raw TopN positive ROI majority (`N=min(20, n_positive_roi)` from `mean_z > 0`); one-sided t + BH-FDR outputs are retained for stats only; `unassigned` is used only for `no_positive_roi`.
- `group-network-top`: vertex-level raw TopN positive ROI majority (`mean_z > 0`), with significance retained for stats only; `unassigned` is used only for `no_positive_roi`.
- `group-network-all`: vertex-level network-wide ROI mean score assignment without significance gating (always assign in support mask); significance outputs are stats-only.
- The three active methods above do not consume cortex `8 x T` canonical timeseries for assignment.
   - Brain surface PNGs are cached: if the wb_view render already exists, only the matplotlib composite is regenerated.

## Traceability

Every stage writes a `stage_manifest.json` or top-level manifest containing:

- script path and command
- parameters
- input file metadata
- output file metadata
- timestamp
- git commit when available

The top-level pipeline manifest is:

- `outputs_experiments/hippunfold_recommended_t1w_20260423/kong2022_route_b/pipeline_manifest.json`

If an expected product exists without the corresponding manifest, treat it as non-authoritative and rerun that stage.

# T1only Kong2022 Route B Runbook

## Scope

This runbook is the canonical path for the T1only HippUnfold branch:

- HippUnfold batch root: `outputs_experiments/hippunfold_recommended_t1w_20260423/dense_corobl_batch/`
- Route root: `outputs_experiments/hippunfold_recommended_t1w_20260423/kong2022_route_b/`
- Shared T1only group inputs: `outputs_experiments/hippunfold_recommended_t1w_20260423/kong2022_route_b/hipp_group_inputs_shared/`
- Group atlas outputs: `outputs_experiments/hippunfold_recommended_t1w_20260423/kong2022_route_b/hipp_group_merged_canonical_atlas/`
- Presentation overview outputs: `present_group_t1only/`
- Single maintained presentation policy: `present_group_t1only/` only, 8 overview PNGs only

The old `manifests/hcp_7t_hippocampus_fmri_qc_pass_153.txt` is not valid for this branch. T1only must derive a new pass manifest from its own HippUnfold Dice QC and its own hippocampal surface tSNR QC.

## Non-Reuse Rule

Kong2022 in this run uses Route B, the public 400-ROI dlabel:

- `external/Schaefer2018_Kong2022/Schaefer2018_400Parcels_Kong2022_17Networks_order.dlabel.nii`

Do not reuse T2only cortex PFM, Kong2019 reference outputs, or old `reference/kong2019` run summaries. Route B computes its own cortex tSNR gate directly from each subject's concat dtseries and extracts Kong2022 ROI/canonical-8 timeseries from the public dlabel.
When needed, Route B can also switch the cortex reference source to Route A individualized labels via `--cortex-reference-source route_a` (see below) without changing downstream Route B stages.

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

`kong2022-reference` source selection:

- default (existing behavior): `--cortex-reference-source group`
- Route A individualized cortex labels: `--cortex-reference-source route_a --route-a-root outputs_experiments/cortex_kong2022_areal_mshbm`

Route A source example (minimal change, no Route B workflow rebuild):

```bash
setsid /home/yojiao/.conda/envs/py314/bin/python \
  scripts/hipp_parcellation_network/run_t1only_kong2022_route_b_pipeline.py \
  --cortex-reference-source route_a \
  --route-a-root /home/yojiao/hpchome/hipp/outputs_experiments/cortex_kong2022_areal_mshbm \
  --from-step kong2022-reference \
  --jobs 4 \
  --resume >> /tmp/pipeline_restart.log 2>&1 &
```

Optional: add `--cortex-reference-overwrite` if you intentionally want to replace existing non-Route-A `reference/kong2022` outputs.

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

## Canonical Spectral Fast Rerun

For urgent T1only spectral reruns, the canonical default entrypoint is now:

```bash
/home/yojiao/.conda/envs/py314/bin/python \
  scripts/hipp_parcellation_network/run_t1only_spectral_fast_rerun.py
```

This launcher fixes the runtime policy so future agents do not need ad hoc instructions:

- method order is fixed: `group-intrinsic-spectral` first, then `group-network-spectral`
- after each method finishes, it immediately rebuilds that method's overview PNG
- each spectral method runs `2mm/4mm x L/R = 4` combos in parallel
- each combo runs with `--jobs 1` internally, so `K` is evaluated sequentially within that combo
- each child process is forced to single-threaded BLAS/OpenMP:
  - `OMP_NUM_THREADS=1`
  - `OPENBLAS_NUM_THREADS=1`
  - `MKL_NUM_THREADS=1`
  - `NUMEXPR_NUM_THREADS=1`
- `redundancy-aware` + `--redundancy-early-stop` are enabled by default
- to disable speed mode and force full K evaluation, pass:
  - `--no-redundancy-early-stop`
- subject FC preprocessing defaults to:
  - `--subject-fc-top-fraction 0.10`
  - `--subject-fc-binarize`
- fast launcher exposes the same override knobs as group-atlas:
  - `--subject-fc-top-fraction <0..1>`
  - `--no-subject-fc-binarize`
- total concurrency stays at `4`, below the repository hard cap of `10`
- output directory is always `present_group_t1only/` (no `present_group_t1only_*` sibling dirs)
- overview PNG naming is route/mode explicit:
  - `kong2022_group-intrinsic-spectral_group_overview_<routea|routeb>_<allon|alloff>.png`
  - `kong2022_group-network-spectral_group_overview_<routea|routeb>_<allon|alloff>.png`
- per-run manifests from fast rerun are method-scoped files (not the consolidated 8-way manifest)

## Canonical 8-Way Export (Single Command)

Use this as the only maintained overview export entrypoint:

```bash
/home/yojiao/.conda/envs/py314/bin/python \
  scripts/hipp_parcellation_network/build_present_group_t1only_matrix.py
```

Behavior:

- fixed matrix: `2 methods x 2 routes x allon/alloff = 8` overviews
- source group roots are fixed to:
  - `hipp_group_merged_canonical_atlas_routea_allon`
  - `hipp_group_merged_canonical_atlas_routea_alloff`
  - `hipp_group_merged_canonical_atlas_routeb_allon`
  - `hipp_group_merged_canonical_atlas_routeb_alloff`
- single consolidated manifest:
  - `present_group_t1only/group_overview_manifest_t1only_8way.json`
- single consolidated README:
  - `present_group_t1only/README.md`
- strict precheck is preserved: missing clustering outputs causes hard failure

This fast rerun launcher is the default for interim spectral refreshes. Do not use it as the final paper run command.

## 2026-05-02 No-Speed Full Rerun + 8-Way Refresh

When routeA/routeB outputs look suspiciously similar, use the full no-speed rerun sequence below.

Route A (allon):

```bash
/home/yojiao/.conda/envs/py314/bin/python \
  scripts/hipp_parcellation_network/run_t1only_spectral_fast_rerun.py \
  --shared-root outputs_experiments/hippunfold_recommended_t1w_20260423/kong2022_route_b/rerun_20260502_routea_routeb_full/shared_routea \
  --group-root outputs_experiments/hippunfold_recommended_t1w_20260423/kong2022_route_b/rerun_20260502_routea_routeb_nospeed_full_v2/hipp_group_merged_canonical_atlas \
  --subjects-manifest outputs_experiments/hippunfold_recommended_t1w_20260423/kong2022_route_b/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_151.txt \
  --group-average-surfaces-dir outputs_experiments/hippunfold_recommended_t1w_20260423/kong2022_route_b/hipp_group_merged_canonical_atlas/group_average_surfaces \
  --present-dir outputs_experiments/hippunfold_recommended_t1w_20260423/kong2022_route_b/rerun_20260502_routea_routeb_nospeed_full_v2/present_intermediate \
  --route-source routea \
  --subject-fc-binarize \
  --no-redundancy-early-stop \
  --combo-jobs 4
```

Route A (alloff): same as above but replace `--subject-fc-binarize` with `--no-subject-fc-binarize`.

Route B (allon): same command pattern with:

- `--shared-root .../shared_routeb`
- `--route-source routeb`
- `--subject-fc-binarize`

Route B (alloff): same as routeB-allon but use `--no-subject-fc-binarize`.

Refresh canonical 8 PNGs:

```bash
/home/yojiao/.conda/envs/py314/bin/python \
  scripts/hipp_parcellation_network/build_present_group_t1only_matrix.py \
  --route-root outputs_experiments/hippunfold_recommended_t1w_20260423/kong2022_route_b/rerun_20260502_routea_routeb_nospeed_full_v2 \
  --present-dir present_group_t1only
```

Post-run verification:

```bash
ls -1 present_group_t1only | sort
cd present_group_t1only && sha256sum kong2022_*_group_overview_*.png | sort
```

Expected:

- exactly 8 overview PNG files + one consolidated manifest + README
- all 8 PNG hashes should differ (any exact duplicate is a pipeline red flag)

## Run Record (2026-05-02 UTC)

Full no-speed rerun completed for all four route/mode combinations:

- `logs/spectral_fast_rerun_routea_allon_20260502T061751Z`
- `logs/spectral_fast_rerun_routea_alloff_20260502T063703Z`
- `logs/spectral_fast_rerun_routeb_allon_20260502T065622Z`
- `logs/spectral_fast_rerun_routeb_alloff_20260502T072212Z`

Each log set contains 8 combo logs and all combos ended with `status: ok`.

Refreshed 8-way output directory:

- `present_group_t1only/`
- `present_group_t1only/group_overview_manifest_t1only_8way.json`

PNG hash snapshot (`sha256sum kong2022_*_group_overview_*.png | sort`):

```text
3f040dac72787c4287762670e19f9dd18db0735526f5fd75a9d3b64201e0aacc  kong2022_group-intrinsic-spectral_group_overview_routea_allon.png
90326c311abdb7c599303ec5d92a189041f9411c5d2ecbe7b37ead68dc188731  kong2022_group-network-spectral_group_overview_routeb_alloff.png
9cff223249f2a74002e3fd9efaca2861167040b839c5f0b098c5e24aaac26d1d  kong2022_group-intrinsic-spectral_group_overview_routea_alloff.png
9e73da1bbd89149c35ec2f32ff4b00d74ccf97723d59b991f024d2699b6a4216  kong2022_group-network-spectral_group_overview_routeb_allon.png
a84d408af719a233624f9d4df5cf0c4cc113dfa6fae50aac2d610ef2452ba3b5  kong2022_group-network-spectral_group_overview_routea_allon.png
aaf25df1950508392895248b75adf5c34b4fd010692b1bdf3256aebd20575283  kong2022_group-intrinsic-spectral_group_overview_routeb_allon.png
c691b4d0159fc35898222cb9bd6b405408435b6988e911b258ef0b2ef0072f38  kong2022_group-network-spectral_group_overview_routea_alloff.png
d430f56e75a181ac66d01d6b2ef60543708bce9a95e350dea305cd00a5af3485  kong2022_group-intrinsic-spectral_group_overview_routeb_alloff.png
```

Result check:

- all 8 PNG files are byte-distinct
- no exact duplicate overview image remained in the maintained 8-way set

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
   - Extracts cortex ROI timeseries from one of two sources:
     - `group` (default): Kong2022 public dlabel (`run_cortex_kong2022_reference_batch.py`)
     - `route_a`: individualized Route A `lh/rh_labels.txt` (`run_cortex_kong2022_route_a_reference_batch.py`)
   - Computes cortex tSNR masks within `reference/common/tsnr_gate/` (atlas-invariant upstream).
   - Uses explicit/equal 4-run slicing from the concat dtseries; the current HCP 7T concat input is expected to be `3600 = 4 x 900`.
   - Stores full-length atlas arrays once under `reference/kong2022/` (`cortex_roi_parcel_timeseries.npy`, `cortex_canonical_network_timeseries.npy`) and stores shared run boundaries in `reference/common/runs/run-*/reference_summary.json` instead of duplicating run timeseries arrays.
   - Legacy `reference/kong2022/{tsnr_gate,runs}` content is migrated to `reference/common/` and removed.

3. `kong2022-fc`
   - Materializes hippocampal vertex-to-Kong2022-400ROI Pearson FC for `2mm` and `4mm`, both hemispheres.

4. `intrinsic-fc`
   - Materializes atlas-invariant hippocampal vertex-to-vertex intrinsic FC for `2mm` and `4mm`, both hemispheres.

5. `group-atlas`
   - Runs two spectral methods for atlas `kong2022` by default: `group-intrinsic-spectral`, `group-network-spectral`.
   - `group-network-top` / `group-network-all` are legacy and retired from default runs.
   - Uses coverage threshold `0.80` and the T1only group-average surfaces.
   - `run_merged_canonical_group_atlas.py --jobs` now parallelizes candidate-`K` evaluation within each combo, capped at `min(--jobs, #K, 10)`.
   - Pipeline forwards `--jobs` to group-atlas and caps K-eval parallelism at 4 by default (`group_atlas_jobs=4`).
   - If a combo has no valid `K` (or no `K` survives constraints), that combo is marked skipped in `group_atlas_manifest.json` (`status=skipped_*`) and the remaining combos continue.
   - `--resume` now validates `resume_input_signature` per combo (upstream FC/reference inputs + masks + key K-selection controls). If signature mismatches, or the manifest is legacy without signature, that combo is recomputed instead of skipped.
   - Per-combo logs are written to `hipp_group_merged_canonical_atlas/_combo_logs/`.
   - Spectral methods support `--k-selection-mode redundancy-aware`.
   - The script suppresses scikit-learn's `"Graph is not fully connected"` warning so large spectral runs do not spam logs.
   - Redundancy-aware gates:
     - non-redundancy: `max_profile_similarity <= 0.80` and `redundant_pair_count == 0`
   - Final K selection (three-tier over the survivor pool that passes all gates):
     1. 1-SE rule: keep survivors within 1 SE of the survivor-pool best mean ARI
     2. local-max preference: among within-1SE survivors, prefer ARI-curve local maxima
     3. parsimony tiebreak: smallest K when quality is genuinely tied
   - Urgent speed heuristic `--redundancy-early-stop`:
     - evaluate `K` in ascending order
     - once the first redundant `K` is hit, skip all larger `K`
     - this is intentionally approximate and should be reserved for fast interim reruns, not final paper runs
   - For T1only spectral reruns, do not parallelize over `K`; the canonical fast path is to parallelize the four independent combos (`2mm/4mm x L/R`) while keeping each combo's `K` loop sequential.
   - Overview `K` diagnostics expose reason tags per rejected `K`: `redundant`, `nontrivial_fail`.

6. `present`
   - Builds overview figures in `present_group_t1only/` only.
   - Default present methods in this pipeline are `group-intrinsic-spectral`, `group-network-spectral`.
   - Maintained default is the 8-way matrix export (single command above), not ad hoc per-route present dirs.
   - For cluster methods, the previous `cluster_indicator` panel is replaced by a K-metric gate panel:
     - `max_profile_similarity` vs K with threshold `0.80`
     - selected K (`K*`) marked by vertical dashed lines
   - `group-network-top` / `group-network-all` are not part of maintained default present output.
   - Do not create new `present_group_t1only_*` sibling output directories.

7. `cluster-roi-cortex`
   - Runs independent cortex/hippomaps figure generation. All outputs go to a single root: `present_group_t1only/runtime/`
   - Merged-hemisphere overview PNGs are written directly under `present_group_t1only/` for quick access.
   - Cluster methods keep per-smoothing files: `cortex_<atlas>_<method>_<smoothing>_overview.png`.
   - Kept separate from overview canvas (no layout merge into `*_group_overview.png`).
   - Uses one-sided one-sample t-tests with BH-FDR (`q < 0.05`) on ROI edges.
   - Intrinsic cluster mode: `Sig-only`, capped at top 20 significant ROI per sign (positive/negative) per cluster.
   - Maintained defaults are spectral modes only (`group-intrinsic-spectral`, `group-network-spectral`).
   - `runtime/` stores per-cluster figures, combo manifests, and the quicklook index manifest.
   - Overview collage composes panels from cluster figures in `runtime/` (no auto-delete).
   - Default method set for this stage is `group-intrinsic-spectral`, `group-network-spectral`.
   - Rendering uses only the first scene view (single 4-panel cortex layout), not multi-view expansion.
   - Each cluster figure includes network-color legend (not ROI-name legend) and explicit sign-coding note (`+` saturated network color, `-` light tint of same network color).
   - Merged overview panels annotate per-panel ROI counts as `n=total (+pos/-neg)`.
   - Cortex template asset resolution supports public/shared fallback:
     - first tries legacy `outputs_migration/cortex_pfm/sub-<template>/assets/*_sulc.func.gii`
     - then tries shared files under `external/` (including FASTANS template metrics)
     - if hemisphere `sulc.func.gii` is still missing but `*_sulc.dscalar.nii` exists, it auto-generates left/right `sulc.func.gii` via `wb_command -cifti-separate` into `tmp/cortex_template_cache/sub-<template>/`.

## Assignment Rule Update

- `group-intrinsic-spectral`: cluster network assignment default is raw TopN positive ROI majority (`N=min(20, n_positive_roi)` from `mean_z > 0`); one-sided t + BH-FDR outputs are retained for stats only; `unassigned` is used only for `no_positive_roi`.
- `group-network-spectral`: cluster network assignment remains Kong2022 ROI-profile based for cluster annotation.
- Maintained default methods do not depend on legacy `group-network-top` / `group-network-all` branches.
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

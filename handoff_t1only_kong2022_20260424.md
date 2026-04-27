# T1only + Kong2022 Route B Handoff (Updated 2026-04-26)

## Scope
Full T1only flow from HippUnfold-based QC prep through Kong2022 Route B group atlas and overview figures.

## Repository Context
- Repo: `/home/yojiao/hpchome/hipp`
- Runbook: `docs/hipp_parcellation_network/t1only_kong2022_route_b_runbook.md`
- Output root: `outputs_experiments/hippunfold_recommended_t1w_20260423/kong2022_route_b/`

---

## Pipeline Status (2026-04-26 ~23:00 CDT)

| Stage | Status |
|-------|--------|
| prepare (raw / tSNR / QC / smooth / group-avg) | ✅ Complete |
| kong2022-reference | ✅ Complete — 151/151 |
| kong2022-fc | ✅ Complete |
| intrinsic-fc | ✅ Complete — 604 combos |
| **group-atlas** | ⏳ In progress — combo 1/8, currently at k=9 |
| present | Waiting |

**QC result**: 151 subjects pass (T1only Dice + tSNR gate).
**Group-average surfaces**: generated at `hipp_group_merged_canonical_atlas/group_average_surfaces/`.

### Active Process
- Pipeline orchestrator: PID `4052505` (setsid, PGID=PID)
- Watchdog: PID `4054273` (`/tmp/pipeline_watchdog.sh`)
- Logs: `outputs_experiments/.../kong2022_route_b/logs/pipeline_20260426T032109Z/`

---

## Critical Runtime Rules (Hard-Won Lessons)

### 1. Always launch with `setsid`
```bash
setsid /home/yojiao/.conda/envs/py314/bin/python \
  scripts/hipp_parcellation_network/run_t1only_kong2022_route_b_pipeline.py \
  --jobs 4 --resume >> /tmp/pipeline_restart.log 2>&1 &
```
To kill cleanly (cascades to all workers + wb_command children):
```bash
kill -- -<PID>   # PID == PGID when launched with setsid
```
**Never** `kill <PID>` alone — orphans ProcessPoolExecutor workers and wb_command children.

### 2. wb_command OMP_NUM_THREADS=2 — already fixed in code
`wb_command -metric-smoothing` defaults to 128 threads/process on this host.
With `--jobs 4` that was 512 threads → ~10× slowdown (80 min/batch → 5 min/batch).
Fix applied in `prepare_t1only_group_inputs.py:run_smoothing()`: passes `OMP_NUM_THREADS=2` in subprocess env.
**Do not remove this.**

### 3. --jobs 4 recommended for smooth stage
Higher values cause NFS I/O contention + wb_command thread explosion.

### 4. prepare is now fully resume-safe
All sub-stages skip already-completed work on `--resume`:
- raw: checks `surface_sampling_summary.json`
- tSNR: checks `tsnr_gate_summary.json` (fixed 2026-04-26)
- QC manifest: skips if `hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt` exists (fixed 2026-04-26)
- smooth: checks 4 timeseries.npy files
- group-average: always reruns (fast, ~seconds)

---

## Completed: group-atlas Combo Parallelism

group-atlas runs these 4 independent combos (default):
- 1 method: `group-intrinsic-spectral` (`group-network-spectral` is opt-in, not run by default)
- 2 smoothings: `2mm`, `4mm`
- 2 hemis: `L`, `R`

`run_merged_canonical_group_atlas.py` now supports `--jobs` and dispatches combo tasks in
parallel via `ProcessPoolExecutor` (each combo runs in an isolated subprocess with `--jobs 1`).

For this route, pipeline now forwards group-atlas combo parallelism and caps it at **4**:
- `run_t1only_kong2022_route_b_pipeline.py` → `group_atlas_jobs = min(jobs, 4)`
- `group-atlas` stage includes `--jobs <group_atlas_jobs>`

Per-combo logs: `hipp_group_merged_canonical_atlas/_combo_logs/*.log`

---

## Monitoring Commands

```bash
# Watchdog log
tail -5 /tmp/pipeline_watchdog.log

# Pipeline stdout
tail -20 /tmp/pipeline_restart.log

# group-atlas progress (current combo + k)
LATEST=$(ls -dt outputs_experiments/hippunfold_recommended_t1w_20260423/kong2022_route_b/logs/pipeline_* | head -1)
grep -E '^\[combo\]|\[k\] |\[skip\]' "$LATEST/group-atlas.log" | tail -10

# Check all stage logs present (= stages completed)
ls $LATEST/

# CPU sanity check
ps -eo pid,pgid,pcpu,nlwp,comm | grep wb_command | awk '$3>0'
ps aux --sort=-%cpu | head -8
```

## Restart Commands

```bash
# Full restart from prepare
setsid /home/yojiao/.conda/envs/py314/bin/python \
  scripts/hipp_parcellation_network/run_t1only_kong2022_route_b_pipeline.py \
  --from-step prepare --jobs 4 --resume >> /tmp/pipeline_restart.log 2>&1 &

# From group-atlas only
setsid /home/yojiao/.conda/envs/py314/bin/python \
  scripts/hipp_parcellation_network/run_t1only_kong2022_route_b_pipeline.py \
  --from-step group-atlas --jobs 4 --resume >> /tmp/pipeline_restart.log 2>&1 &
```

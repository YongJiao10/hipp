# T1only + Kong2022 Route B Handoff (Updated 2026-04-25 UTC)

## Scope
This handoff records the live state of the T1only full flow from HippUnfold-based QC prep through downstream Kong2022 group analysis.

## Repository / Branch Context
- Repo: `/home/yojiao/hpchome/hipp`
- Target branch intent: `T1only`
- Handoff file policy: keep handoff in repo root

## What Changed Since Previous Handoff
- Fixed run-time concurrency capping so one-off approved runs can exceed 10 when explicitly requested.
- Added env override `HIPP_MAX_CONCURRENCY` in both launchers.
- Updated runbook with override usage.

Modified files:
- `scripts/hipp_parcellation_network/run_t1only_kong2022_route_b_pipeline.py`
- `scripts/hipp_parcellation_network/prepare_t1only_group_inputs.py`
- `docs/hipp_parcellation_network/t1only_kong2022_route_b_runbook.md`

## Concurrency Behavior (Important)
Default cap remains 10.
One-off override now supported with env var:

```bash
HIPP_MAX_CONCURRENCY=12 \
/home/yojiao/.conda/envs/py314/bin/python scripts/hipp_parcellation_network/run_t1only_kong2022_route_b_pipeline.py --jobs 12 --resume
```

Implementation details:
- `run_t1only_kong2022_route_b_pipeline.py`: `jobs=min(requested, resolve_max_concurrency())`
- `prepare_t1only_group_inputs.py`: same resolver + removed internal hard cap in `run_parallel()` so effective workers follow `jobs`.

## Live Runtime Status (Latest Check)
Timestamp: `2026-04-25 03:04:50 UTC`

Active process:
- PID `3926063`
- Command:
  - `/home/yojiao/.conda/envs/py314/bin/python scripts/hipp_parcellation_network/prepare_t1only_group_inputs.py ... --jobs 12 --resume`

Active stage:
- `prepare` -> `raw sampling`

Effective current parallelism:
- `sample_hipp_surface_timeseries.py`: 12 active workers
- `wb_command -volume-to-surface-mapping`: 12 active workers

## Progress Snapshot
From artifacts/logs:
- `raw_done` (via `surface_sampling_summary.json`): `91`
- `raw_full_lr`: `91`
- `raw_partial`: `3`
- Latest completed raw subject timestamp: `2026-04-25 02:43:16 UTC` (`sub-246133`)

Current active prepare log:
- `outputs_experiments/hippunfold_recommended_t1w_20260423/kong2022_route_b/logs/pipeline_20260425T024709Z/prepare.log`
- This resumed prepare instance started with: `raw sampling todo=74`
- No FAIL lines yet in this new `pipeline_20260425T024709Z` log.

## Known Runtime Pattern
- Throughput is bursty/long-tail during raw mapping.
- Short windows can show no new completions even while 12 `wb_command` are active.

## Next Agent Action Plan
1. Keep current `prepare` process running (do not launch duplicate prepare/pipeline in parallel).
2. Monitor until `prepare` completes all substeps (`raw -> tsnr -> qc -> smooth -> group-average`).
3. After `prepare` done, run orchestrator from next stage:

```bash
HIPP_MAX_CONCURRENCY=12 \
/home/yojiao/.conda/envs/py314/bin/python scripts/hipp_parcellation_network/run_t1only_kong2022_route_b_pipeline.py \
  --from-step kong2022-reference \
  --jobs 12 \
  --resume
```

4. Continue to final outputs:
- `group_present_group_t1only/`

5. At QC completion, explicitly report two counts:
- Dice QC pass count
- tSNR QC pass count (final T1only pass cohort)

## Monitoring Commands
```bash
# raw completion count
find outputs_experiments/hippunfold_recommended_t1w_20260423/kong2022_route_b/hipp_group_inputs_shared \
  -path '*/surface/raw/surface_sampling_summary.json' -type f | wc -l

# active prepare PID
pgrep -af 'prepare_t1only_group_inputs.py'

# active worker counts
ps -ef | rg 'sample_hipp_surface_timeseries.py' | wc -l
ps -ef | rg 'exe_linux64/wb_command -volume-to-surface-mapping' | wc -l
ps -ef | rg '/bin/bash .*wb_command -volume-to-surface-mapping' | wc -l

# latest prepare log tail
tail -n 80 outputs_experiments/hippunfold_recommended_t1w_20260423/kong2022_route_b/logs/pipeline_20260425T024709Z/prepare.log
```

## Process Count Semantics
- Real compute concurrency: count only `.../exe_linux64/wb_command -volume-to-surface-mapping`.
- Wrapper processes: `/bin/bash ... wb_command ...` are launcher shells, not additional compute slots.
- `sample_hipp_surface_timeseries.py` is Python worker count; it may differ briefly from compute count during spawn/teardown.

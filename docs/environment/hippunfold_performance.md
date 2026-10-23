# HippUnfold shared-host performance measurement

Measured on 2026-10-23 using the current official v2.2.0 scientific runtime.
The complete structural roster remains `manifests/hcp_7t_hippocampus_struct_complete_166.txt`.
The user requested acceleration while preserving capacity for other tasks.
The user approved the eight-subject profile below. The scheduling changes are
installed and the full-roster launcher runs with the approved resource limits.

## Available resources and baseline

IDAS exposes 128 logical CPUs from two EPYC 9334 processors, but the container
CPU quota is 32 CPUs (`cpu.max = 3200000 100000`). The memory limit is
512,000,000,000 bytes (476.8 GiB), and the L40S provides 46,068 MiB of GPU memory.
Most current cgroup memory usage is reclaimable file cache; resident worker
memory is reported separately below.

The complete 102311 test took 44m35s. Its eight Greedy inner/outer registrations
took 974s altogether, including approximately 225s per dentate registration and
15s per hippocampal registration. Its two nnUNet inference rules took 42s
altogether. This workload primarily benefits from CPU parallelism.

## CPU measurement

Each worker replayed the same actual right-dentate inner-surface registration:
24,801,658 voxels, official `50x50x50` iterations and
`sqrt(3.0)vox 0.707vox` smoothing. Inputs are retained 102311 masks; each worker
writes its own warp under `tmp/hippunfold_perf_20261023/`. One production subject
continued in the background. Its load is included in cgroup CPU averages but
not in the per-worker CPU counters.

| Processes | Threads/process | Wave time, s | Registration tasks/hour | Mean cgroup CPU cores | Sum of worker peak RSS, GiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 1 | 227.1 | 63.4 | 5.38 | 11.6 |
| 8 | 1 | 239.1 | 120.5 | 8.89 | 23.2 |
| 12 | 1 | 230.1 | 187.7 | 13.20 | 34.7 |
| 6 | 2 | 126.1 | 171.4 | 12.39 | 17.4 |
| 3 | 4 | 97.2 | 111.1 | 8.31 | 8.7 |

These are registration-task throughputs, not completed-subject rates. Each
configuration is one wave of identical real tasks; this is not a repeated
statistical benchmark. Sum of per-worker peak RSS is an upper bound rather than
a simultaneous memory sample. A smaller hippocampal replay independently
favored more single-thread workers. The selected one-thread warps from the
4-, 8- and 12-worker waves have identical voxel arrays and affines.

## GPU measurement

Four independent nnUNet commands processed the same real preprocessed crop per
configuration, keeping all five Task101 model folds, `model_best`, trainer,
patch dimensions, mixed precision, disabled TTA and one preprocessing/save
worker each. The normal process startup/model-loading cost is included.

| Simultaneous inference processes | Four predictions, s | Predictions/hour | Peak GPU memory, MiB | Mean GPU utilization |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 62.13 | 231.8 | 2,265 | 14.2% |
| 2 | 32.72 | 440.1 | 4,527 | 25.3% |
| 4 | 19.07 | 755.2 | 9,051 | 33.8% |

Two processes deliver 1.90 times the one-process inference throughput, using
4.42 GiB of GPU memory and leaving approximately 40.6 GiB relative to the
observed peak. These measurements do not reserve GPU capacity from other jobs.
Four processes are faster but the user prefers additional headroom.

Repeated predictions are not bit-identical even with one process. Across all
pairs of four repetitions, label disagreements range from 0–8 voxels at one
process, 0–10 at two processes and 0–4 at four processes. The input has
4,194,304 voxels and the reference has 146,760 foreground voxels; the largest
observed disagreement is below 0.01% of the foreground count. This is measured
repeatability, not an anatomical-accuracy validation. Existing structural QC,
including native-unfold correspondence checks, remains necessary.

## Production profile

Use eight subjects, one CPU core per subject and two GPU inference slots.
Restrict the complete process tree to 16 CPUs and use niceness 10:

```bash
taskset --cpu-list 0-7,32-39 nice -n 10 /home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/01_01_run_hippunfold_t1w_local.py --dataset hcp --manifest manifests/hcp_7t_hippocampus_struct_complete_166.txt --jobs 8
```

The listed CPUs are 16 distinct physical cores on the current IDAS host. The
affinity bounds this workflow to half the 32-CPU quota, including helpers.
It does not reserve those CPUs exclusively. One-core scientific execution,
model validation once per batch, complete cached-resource reuse, full output
retention and completion contracts remain unchanged. Other dataset presets
retain their current limits.

The isolated proposal and exact source/doc changes are under
`tmp/hippunfold_perf_20261023/proposed/` and `proposed_scheduling.patch`.
Three new behavior tests failed against the original configuration as expected.
The isolated proposed modules pass all 58 relevant tests, including actual
two-slot contention/release. The complete 418-job DAG also builds successfully
from the isolated proposed source with HTTP requests blocked; it executes no
scientific jobs and does not replace installed code.
The small lock helper uses `-S` before exec to avoid importing Torch twice;
actual inference retains its normal Python startup hook.

The user explicitly approved this profile and its process-count exception:
helpers can bring the eight-subject process count to approximately 29, while
CPU affinity limits CPU execution to 16 cores. No 32-worker full-quota benchmark
was run. The user cancelled an additional pi/argon review for this scheduling
change; the performance proposal was not sent to that service.

The installed source passes all 58 relevant tests. The maintained local patch
also recreates the installed four-file resource/runtime adaptation from clean
official v2.2.0 source. Install runtime patches only after active HippUnfold
subprocesses exit: Snakemake reloads rule files during execution.

Two subjects encountered a missing scheduling environment variable during the
runtime transition. Both then passed dry runs with the new environment, without
requiring metadata cleanup or a recovery flag. Their retained preprocessing
outputs are reused by the normal full-roster restart. The old coordinator has
exited. The new coordinator executes the canonical 166-person manifest, skips
completed subjects through the existing completion contract and checks new
outputs serially with full retention.

An eight-subject end-to-end wave has not yet been measured. Its actual
subject/hour rate and remaining cohort time must be established after profile
launch; stage measurements must not be presented as whole-workflow
speedups. No scientific-method repair, functional downstream run or commit is
part of this performance change.

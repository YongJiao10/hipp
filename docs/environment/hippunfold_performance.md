# HippUnfold shared-host performance measurement

The measurements below used the official v2.2.0 scientific runtime before
the current v2.3.0 reconstruction. They inform resource allocation; the
[runtime manifest](hippo2_reproducibility_manifest.md) defines the current
scientific runtime and retained-output requirements.
The complete structural roster remains `manifests/hcp_7t_hippocampus_struct_complete_166.txt`.
The user requested acceleration while preserving capacity for other tasks.
HCP manifest runs default to eight subjects and accept an explicit `--jobs`
override up to twelve. The user approved twelve for the current full cohort. Its
approved profile uses `--jobs 12` for that workload.

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
`sqrt(3.0)vox 0.707vox` smoothing. Each worker used 102311 masks and wrote a
separate temporary warp. The measurements below are retained independently of
the disposable benchmark workspace. One production subject
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

For the approved HCP profile, use twelve subjects, one CPU core per subject and two GPU
inference slots. Future batches can select their concurrency with `--jobs`.
Restrict the complete process tree to 16 CPUs and use niceness 10:

```bash
taskset --cpu-list 0-7,32-39 nice -n 10 /home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/01_01_run_hippunfold_t1w_local.py --dataset hcp --manifest manifests/hcp_7t_hippocampus_struct_complete_166.txt --jobs 12
```

The listed CPUs are 16 distinct physical cores on the current IDAS host. The
affinity bounds this workflow to half the 32-CPU quota, including helpers.
It does not reserve those CPUs exclusively. One-core scientific execution,
model validation once per batch, complete cached-resource reuse, full output
retention and completion contracts remain unchanged. Other dataset presets
retain their current limits.

The eight-subject default passed 58 relevant tests and its full 418-job cached
DAG built with HTTP requests blocked. The twelve-subject update makes the
existing `--jobs` option adjustable up to twelve for HCP while keeping its
default at eight. The one-core HippUnfold arguments, two-slot GPU inference
lock, model gate, cache reuse and full-output checks remain unchanged.

The first eight-subject full-run wave completed eight people in approximately
50 minutes. These measurements do not establish the whole-subject throughput
of the current v2.3.0 reconstruction.
The CPU stage benchmark at twelve workers measured 187.7 registration tasks per
hour, but that is not a full-subject throughput estimate. The profile has more
than 40 total worker/helper processes; user approval covers this HCP-specific
process-count exception. Its affinity limits execution to 16 CPUs. The GPU
label-repeat differences described above remain below 0.01% of foreground
voxels; this profile still uses only two inference slots.

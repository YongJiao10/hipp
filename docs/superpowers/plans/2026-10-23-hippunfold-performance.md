# HippUnfold Performance Implementation Plan

> **For agentic workers:** Use `executing-plans` for direct execution. No commits. The user approved eight subjects, a 16-CPU affinity limit, niceness 10 and two GPU inference slots. Additional pi/argon review was explicitly cancelled for this scheduling change.

**Goal:** Measure a balanced HCP7T concurrency on the available 32-CPU quota and L40S, leaving capacity for other tasks, and prepare the corresponding production settings.

**Architecture:** Replay the actual expensive Greedy registration and nnUNet inference on retained completed-subject inputs, writing only to the project temporary workspace. Compare throughput, latency and resource use before changing the maintained launcher; scientific methods, model, atlas and density remain fixed.

**Tech Stack:** hippo2 native tools/PyTorch; py314 measurement coordinator; Linux cgroup counters and NVIDIA GPU counters.

**Spec:** User request on 2026-10-23: measure useful CPU/GPU parallelism, disregarding the previous concurrency caps. The preceding structural-only scope, full-output retention and no-commit instruction remain active.

## Global Constraints

- IDAS: 128 visible logical CPUs, effective CPU quota 32; cgroup memory limit 512,000,000,000 bytes; one L40S with 46,068 MiB GPU memory.
- Pause only the old coordinator's dispatch; active subjects continue. Run the normal serial output checker after they finish, then replace the coordinator with the approved eight-subject launch.
- Keep active scientific jobs running and retain completed results. The user approved the concrete measured profile and its helper-process exception.
- Use `tmp/hippunfold_perf_20261023/` for benchmark outputs. Do not delete or overwrite existing scientific products or read archives.
- Preserve official scientific parameters and once-per-batch cache/model validation. Test GPU concurrency independently of CPU concurrency.
- Keep user-facing production inspections twenty minutes apart; automated benchmark sampling may be frequent without repeated model log inspection.

## Review Focus

- Host CPU/RAM totals overstate resources available through the container quota.
- GPU process concurrency can reduce per-case performance or exhaust GPU memory.
- A faster individual registration need not improve whole-machine throughput.
- Existing completion checks must continue to accept scientifically identical completed subjects.
- Restarted production must not collide with benchmark work or repeat completed scientific steps.

## Tasks

- [x] Establish the actual resource limits and completed-run timing baseline. Greedy registration consumes 974 s of the 44m35s run; the expensive dentate registrations take about 225 s each, whereas hippocampal registrations take about 15 s. Two nnUNet calls consume 42 s in total. Pause old coordinator PID 1562766 while its subjects finish.
- [x] Run the expensive dentate registration: 4, 8 and 12 one-thread jobs, then six two-thread and three four-thread jobs. Throughputs are 63.4, 120.5, 187.7, 171.4 and 111.1 registration tasks/hour. Selected one-thread warp arrays are identical across parallelism levels. The smaller hippocampal replay also favors more single-thread processes.
- [x] Measure real five-fold nnUNet inference at one, two and four simultaneous processes, with four independent repetitions per configuration. Wall times: 62.13, 32.72 and 19.07 s. GPU peaks: 2265, 4527 and 9051 MiB. Two slots are the proposed balance for other GPU tasks.
- [x] Prepare and install the approved eight-subject/one-core/two-GPU-slot scheduling patch with matching docs. Three new tests failed on the original implementation as expected; all 58 relevant tests pass against the installed profile. The isolated full 418-job cached DAG builds with HTTP blocked. CPU affinity and niceness preserve capacity for other tasks; scientific completion contracts are unchanged. GPU serial/concurrent repeats show small label variability (at most 8/10/4 voxels at 1/2/4 processes).
- [x] Obtain explicit approval for the measured profile and its process-count exception. The user cancelled additional pi/argon review; no new performance payload was sent externally.
- [x] Retire the old dispatcher after both active subprocesses exit, then execute the full-roster eight-subject launcher. The two interrupted inference entries passed dry runs with the new environment and resume from retained preprocessing outputs.
- [ ] Measure actual cohort throughput from subsequent completed waves.

The scheduling profile is installed and its 58 relevant tests pass. The old
coordinator has exited. The canonical full-roster launcher now uses eight
subjects, 16-CPU affinity, niceness 10 and two GPU slots. Existing completed
outputs are retained. No commits or deletions were made.

# HippUnfold Performance Implementation Plan

> **For agentic workers:** Use `executing-plans` for direct execution. No commits. The HCP manifest default remains eight; the user approved `--jobs 12` for the current run. Keep the 16-CPU affinity, niceness 10 and two GPU inference slots.

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
- [x] Measure actual CPU registration and GPU inference concurrency. One-thread Greedy registration throughput rose to 187.7 tasks/hour at twelve workers; selected outputs were identical across the 4-, 8- and 12-worker waves. Two GPU slots delivered 1.90 times single-slot throughput in the measured repetitions.
- [x] Run the initial HCP roster at eight subjects; eight completed subjects passed the full output check in approximately 50 minutes.
- [x] Keep the HCP manifest default at eight and make `--jobs` configurable up to twelve, retaining one CPU core per subject, the 16-CPU affinity, niceness 10 and two GPU slots.
- [x] Obtain focused pi/argon review of the twelve-subject scheduling delta; the reviewer found no blockers.
- [x] Switch after the currently running eight-subject wave exits and is checked; the new full-roster coordinator starts twelve subjects.
- [ ] Measure actual throughput at twelve-subject concurrency; do not infer it from the registration-stage benchmark.

The twelve-subject run is active after a clean transition. The prior eight
active subjects passed their serial output checks; 28 of 166 roster entries are
now complete and twelve are running. No commits or deletions were made.

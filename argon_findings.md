# Argon Findings (Kong2022 Route A / MATLAB Kill Investigation)

This file records a historical investigation. Current submission commands,
runtime requirements and retained-work policy are maintained in the
[Kong2022 ArealMSHBM guide](scripts/ArealMSHBM_Kong2022/README.md).

## Scope
Date range: 2026-04-28
Environment: Argon SGE queues (`UI`, `PINC`, `UI-HM`, `all.q`)
Goal: find why Route A jobs are killed and whether MATLAB startup mode is the cause.

## High-Level Conclusions
- The failure is not a simple `matlab -batch` bug.
- Real Route A jobs are repeatedly killed during the main MATLAB stage on both `UI` and `PINC` (`exit_status 137`), even when using different subjects and different queues.
- Queue/node conditions strongly affect outcomes; same nominal memory target can pass or fail at different times.
- `UI-HM` was frequently unavailable (`AVAIL=0`), so no-memory submission did not bypass queue wait.

## Core Evidence
### 1) Real Route A jobs killed in MATLAB stage
- UI no-memory run:
  - `5888152` (`realUI_nomem_s1`): `exit_status 137`, `maxvmem 4.783GB`
  - `5888154` (`realUI_nomem_s2`): `exit_status 137`, `maxvmem 5.302GB`
- PINC no-memory run:
  - `5888157` (`realPINC_nomem_s1`): `exit_status 137`, `maxvmem 4.648GB`
  - `5888161` (`realPINC_nomem_s2`): `exit_status 137`, `maxvmem 4.456GB`
- Log signature for all above:
  - `line 117: ... Killed matlab -batch "${batch_expr}" ...`

### 2) Different-subject split still killed
- UI subject 100610 (`5888322`) and PINC subject 102311 (`5888324`) were run concurrently (no output-path collision across subject).
- Both still ended with `Killed` in MATLAB stage.

### 3) Memory probe vs real workload mismatch is real
- Synthetic memory probes (`memory_probe.py`) and real Kong2022 workload behave differently.
- Synthetic probes can pass much higher targets in some windows, while real workflow can still be killed around 4.5-5.5GB observed maxvmem.
- This indicates process shape/phase behavior and node pressure matter, not only requested memory numbers.

## Queue Behavior Findings
### UI-HM
- Multiple checks showed `UI-HM` had `AVAIL=0` while tests were attempted.
- Submitting without memory resources still remained `qw`; no bypass was observed.

### all.q
- Early no-memory probe on `all.q` (`5888024`) was killed (`exit_status 137`) at low observed maxvmem (~3GB), showing queue-specific volatility.

## MATLAB Mode Isolation (MATLAB-only probes)
### Completed probe results
- UI:
  - `mp_ui_b0` `5888556`: success
  - `mp_ui_r0` `5888558`: success
  - `mp_ui_b6` `5888561`: failed (`exit_status 9`, log cut during allocation)
  - `mp_ui_r6` `5888565`: failed (`exit_status 9`, log cut during allocation)
  - `mp_ui_nd6` `5888700` (`batch_nodisplay`, 6G): success
  - `mp_ui_nj4` `5888705` (`r_nodesktop_nojvm`, 4G): success
  - `mp_ui_nj6` `5888708` (`r_nodesktop_nojvm`, 6G): success
- PINC:
  - `mp_pinc_b0` `5888568`: success
  - `mp_pinc_r0` `5888572`: success
  - `mp_pinc_b6` `5888576`: success
  - `mp_pinc_r6` `5888580`: success

Interpretation:
- `-batch` itself is not a universal failure trigger.
- `-r -nodisplay -nodesktop` is not universally stable either.
- Outcome depends on queue/node conditions and runtime context.
- `-nojvm` reduces JVM usage (`usejava('jvm')=0`) and can improve stability in some UI probe runs, but this is not yet proven on full Route A end-to-end.

## Investigation workflow
The probes varied MATLAB startup mode, isolated subject outputs and recorded
runtime memory limits. Their controls describe this historical investigation;
use the maintained guide above for current launch options.

## Controlled A/B Results (real Route A)
Completed outcomes:
- UI:
  - `ab3_ui_batch` `5889001`: `exit_status 137 (Killed)`, `maxvmem 5.251GB`
  - `ab3_ui_nojvm` `5889008`: `exit_status 1`, `maxvmem 3.483GB`
- PINC:
  - `ab3_pinc_batch` `5889016`: `exit_status 137 (Killed)`, `maxvmem 5.727GB`
  - `ab3_pinc_nojvm` `5889025`: `exit_status 1`, `maxvmem 3.546GB`

noJVM failure signature:
- Java-dependent gifti/cifti stage fails with:
  - `Unable to find or import 'com.mathworks.mlwidgets.io.InterruptibleStreamCopier'`
  - stack includes `@gifti/private/zstream.m`

## Practical Constraints Identified
- Real jobs writing to the same subject/output tree can introduce extra instability and should be avoided during diagnosis.
- For A/B validation, always isolate output trees (suffix) and force `REPO_ROOT`.
- Queue availability (`UI-HM` especially) must be checked before interpreting queue-specific conclusions.

## Recommended Next Diagnostic Step
- Let `ab3_ui_batch`/`ab3_pinc_batch` finish (or fail), then automatically run their held `noJVM` counterparts.
- Compare 4 jobs directly on:
  - `exit_status`
  - `maxvmem`
  - failure stage in `matlab_wrapper.log`
- This gives the cleanest same-subject same-queue mode A/B evidence.

## Fast UI-Only Update (2026-04-29 01:23-01:30 CDT)
Submitted four UI-only jobs for fastest signal split:
- Real Route A:
  - `fast_ui_real_batch` `5890621` (running at snapshot time)
  - `fast_ui_real_nd` `5890623` (running at snapshot time)
- MATLAB-only probes (mode=`batch`):
  - `fast_ui_probe6` `5890624`: `exit_status 9`, `maxvmem 3.568GB`
  - `fast_ui_probe10` `5890626`: `exit_status 9`, `maxvmem 5.771GB`

Probe log evidence (`fast_ui_probe6.o5890624`, `fast_ui_probe10.o5890626`):
- Both show `Command terminated by signal 9`.
- Both are killed around process RSS/HWM near ~4.8GB in `/proc` printouts.

Interpretation:
- On UI, plain `matlab -batch` alone can be SIGKILLed near ~4.8GB RSS even without CBIG Route A workload.
- This strongly suggests a hard runtime limit/kill policy effect beyond simple script logic.

## Leaf cgroup Hard Evidence (2026-04-29, UI)
Two UI jobs with a dedicated probe that reads `/proc/self/cgroup` and leaf `memory.limit_in_bytes`:

1) `ui_memcg12` (`5891868`, queue `UI`, no `mem_free` request)
- `/proc/self/cgroup` shows memory controller path:
  - `7:memory:/job.slice/job-5891868.1.slice`
- Leaf limit:
  - `memory.limit_in_bytes=4946132992` (~4.61 GiB, ~4.95 GB)
- Near kill:
  - `memory.usage_in_bytes=4946100224`
  - `memory.max_usage_in_bytes=4946132992`
  - `memory.failcnt=109001`
  - then process line shows `Killed`
- Accounting:
  - `qacct -j 5891868` => `exit_status 137 (Killed)`, `maxvmem 5.006GB`

2) `ui_memcg32req` (`5891913`, queue `UI`, with `-l mem_free=32G`)
- `/proc/self/cgroup` shows memory controller at root:
  - `7:memory:/`
- cgroup path resolved to:
  - `/sys/fs/cgroup/memory`
- Reported limit:
  - `memory.limit_in_bytes=9223372036854771712` (effectively unlimited)
- Probe reached full 8GB target and exited normally:
  - `qacct -j 5891913` => `exit_status 0`

Interpretation:
- The observed kill behavior is consistent with whether the job lands in a constrained leaf cgroup (`/job.slice/job-<id>.1.slice` with ~4.95GB cap) versus unconstrained root cgroup (`/`).
- This explains why some jobs die around 5GB while others can allocate much more on the same queue.

## Production Recovery Update (2026-04-30)

### Scheduler Policy Clarification Received
- Cluster admin clarified:
  - `mem_free` is a placement filter, not an exclusive memory reservation.
  - Effective memory entitlement is tied to requested slots (`-pe smp <slots>`), queue-dependent.
- We switched production job template to:
  - `-pe smp 3`
  - `-l mem_free=16G`

### Validation Before Full Relaunch
Short validation jobs all succeeded (`exit_status 0`), including:
- `5909127` (`test_ui_s1_smp3`)
- `5909128` (`test_ui_s2_smp3`)
- `5909130` (`test_pinc_s1_smp3`)
- `5909131` (`test_pinc_s2_smp3`)

### Root Cause of "Many Jobs Died Immediately" (Not OOM)
First production relaunch:
- UI: `5909669`
- PINC: `5909670`

Observed behavior:
- Many tasks exited quickly with `exit_status 2`, very low `maxvmem` (tens of MB), not OOM profile.
- Logs showed repeated `ERROR: invalid SGE_TASK_ID=...`.

Confirmed root cause:
- Submission used `qsub -v SUBJECT_IDS_CSV=...` where the value itself was comma-separated.
- SGE `-v` also uses commas as separators, so subject list was truncated/split.
- Array task index then exceeded parsed list length, causing scripted `exit 2`.

### Submission/Script Fixes Applied
The relaunch supplied line-delimited subject files instead of passing a
comma-separated list through SGE `-v`, and selected the C locale for
submission. Current subject-selection controls are in the maintained guide.

Relaunch after fix:
- UI: `5909706` (`1-51`, `-tc 5`)
- PINC: `5909707` (`1-100`, `-tc 10`)
- New logs showed correct subject mapping and no `invalid SGE_TASK_ID` pattern.

### Disk Quota Incident and Mitigation
Argon quota automation triggered:
- Warning over 90%.
- Jobs held above threshold; running jobs may be deleted at 95%.
- Requirement to resume scheduling: bring usage below 80%, then `qrls -u $USER`.

Measured storage pattern:
- Label outputs are small (per subject roughly ~0.24 MB total for `lh_labels.txt`, `rh_labels.txt`, `route_a_outputs.mat`).
- Major growth came from intermediates:
  - `runs/*.dtseries.nii` (~1.26 GB per subject)
  - `project_dir/gradients` (hundreds of MB per subject)
  - `project_dir/profiles` (tens of MB per subject)

Emergency cleanup actions:
- Removed intermediate trees under Route A output dirs:
  - `runs/`
  - `censor_inputs/`
  - `project_dir/gradients/`
  - `project_dir/profiles/`
- Removed older test output directories (`ab3/fast/older prod variants`) where safe.

Quota recovery outcome:
- Home usage reduced to ~79%.
- Hold release executed successfully:
  - `qrls -u yojiao`
- Queued tasks resumed dispatch (`qstat` back to `r/t/qw` progression).

The historical recovery used cleanup to address the quota incident. Current
retention and deletion rules are defined by the maintained guide and repository
policy.

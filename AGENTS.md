# Repository Rules

## Concurrency Limit — HARD CAP

- **Maximum concurrent jobs/tasks/processes: 10.**
- This applies to ALL agents (Codex, Claude, or any automated runner) operating in this repository.
- Do NOT launch more than 10 parallel subprocesses, worker threads, agent tasks, or pipeline jobs at any one time, regardless of what the task or tool suggests as a default.
- Before launching a batch of parallel work, count the intended concurrency. If it exceeds 10, reduce it.
- This rule exists because a 24-way parallel run previously crashed the host system. There is no override without explicit user permission granted in that session.

## Archive Access

- You MUST obtain explicit user permission before accessing or browsing any archive content.

## Flow Doc Maintenance

- The hippocampal functional parcellation flow documents must be kept in sync with the implementation.
- If any `gradient`, `prob-cluster`, or `prob-soft` workflow changes, update these Markdown files in the same change:
  - [multi_branch_flow.md](docs/hipp_parcellation_network/multi_branch_flow.md)
  - [step_by_step_flows.md](docs/hipp_parcellation_network/step_by_step_flows.md)
- This includes updates to:
  - post-`vertex-to-parcel FC` feature definitions
  - probability transforms
  - regularization rules
  - clustering rules
  - `K` selection logic
  - output interpretation
  - overview figure semantics
- A code change that alters those workflows is not complete until the corresponding flow Markdown files are updated.

## Doc Sync Rule

- Any code change that affects behavior, paths, caches, launchers, environment setup, or workflow outputs must be accompanied by a matching documentation update in the same change.
- This includes edits to:
  - launcher scripts
  - wrapper scripts
  - environment manifests
  - runbooks
  - flow docs
  - reproducibility manifests
- Do not leave code and docs out of sync for future agents to infer intent from stale text.
- If a change touches the runtime stack, update the environment manifest as part of the same change.

## Default Analysis Cohort

- **The default analysis set is 151 subjects** for the active T1only branch, not the full 166.
- Subject list (pinned count): `outputs_experiments/hippunfold_recommended_t1w_20260423/vertex_qc_t1only/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_151.txt` (bare IDs, one per line)
- Subject list (rolling pointer): `outputs_experiments/hippunfold_recommended_t1w_20260423/vertex_qc_t1only/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt`
- QC criteria (both must pass):
  1. HippUnfold Dice ≥ 0.70 for both hemispheres (`*_desc-unetf3d_dice.tsv`)
  2. fMRI null vertex % < 20% for both hemispheres (tSNR < 25 or sd == 0)
- Current audited split (166 -> 151):
  1. `14` excluded by fMRI null-vertex threshold
  2. `1` excluded by Dice threshold (`sub-182739`, right Dice `< 0.70`)
- Excluded subjects: sub-102816, sub-109123, sub-155938, sub-158136, sub-164131, sub-173334, sub-182739, sub-198653, sub-550439, sub-680957, sub-783462, sub-878776, sub-878877, sub-898176, sub-901442
- Full QC details: `docs/hipp_parcellation_network/fmri_qc.md`
- When iterating over subjects for any analysis, load from the manifest — **do not assume all 166 are usable**.

## HippUnfold Density Audit Guardrail

- For any question about hippocampal "density/resolution" (for example `2mm`, `0p5mm`, `512`, `8k`, "how many vertices"), agents MUST separate:
  - command/config argument level (`--output-density`, config defaults)
  - actually consumed asset level (the concrete surface files used by downstream scripts/scenes)
- Agents MUST NOT infer density from parameter names alone. They must verify the real consumed files first.
- Required evidence before answering:
  - concrete file paths used by the target pipeline stage (for example the exact `*.surf.gii` path pattern in code)
  - measured vertex counts from those files (read GIFTI and report `POINTSET` vertex count)
- If command argument and consumed files disagree, agents must explicitly report the mismatch and treat consumed-file evidence as authoritative for "what was actually used".
- For legacy HippoMaps outputs in practice, do not assume `space-corobl_label-hipp_midthickness.surf.gii` implies `den-2mm`; verify by vertex count.

## HippUnfold Environment Gate

- For direct `hippunfold` runs in this repository, the canonical local runtime environment is `hippo2`.
- Do not trust `conda list` or the environment name alone as proof of HippUnfold compatibility.
- Required validation sequence before a direct HippUnfold run:
  1. `source /opt/miniconda3/bin/activate hippo2`
  2. `which hippunfold` must resolve to `/opt/miniconda3/envs/hippo2/bin/hippunfold`
  3. `hippunfold --version` must return `2.0.0`
  4. `hippunfold --modality T2w --output_density 512 --help` must succeed and show `512`
- Do not use `importlib.metadata.version("hippunfold")` as authoritative version proof. In this workspace it can still report `1.5.2rc2` even when the CLI is the validated `2.0.0` build.
- If the gate fails, repair in place with:

```bash
CONDA_SAFETY_CHECKS=disabled conda install -y -n hippo2 -c khanlab -c conda-forge -c bioconda khanlab::hippunfold=2.0.0=py_0 --force-reinstall
```

- Do not treat `bioconda::hippunfold` as sufficient evidence of native `512` support.
- Avoid forbidden workflow version labels for this project: `1.5.2`, `1.5.2-pre.2`, `1.5.2rc2`.

## Agent Runtime Environment Contract (Hard)

- Do not use bare `python` for repository tasks.
- Always use an explicit interpreter path based on task type:
  - Core workflow scripts/tests/doc-contract checks: `/home/yojiao/.conda/envs/py314/bin/python`
  - Direct HippUnfold CLI runs: `hippo2` gate above (`/opt/miniconda3/envs/hippo2/bin/hippunfold` after activation)
- Before running any Python-based repo task, run and verify:
  1. `/home/yojiao/.conda/envs/py314/bin/python -c "import sys,nibabel; print(sys.executable); print(nibabel.__version__)"`
  2. `python -c "import sys; print(sys.executable)"` (informational only; do not use this interpreter for repo tasks unless it exactly matches py314)
- If command examples in docs use `python`, agents must rewrite them to the explicit interpreter path at execution time.
- If an agent accidentally runs with the wrong interpreter, stop and rerun with the correct one; do not continue mixed-environment execution.

Reference runbook: `docs/environment/agent_runtime_contract.md`

## Planning Mode Folder Creation Guardrail

- In planning mode (`plan mode`), if any new folder/directory needs to be created, the agent MUST explicitly ask the user for confirmation and clarify:
  1. The folder name.
  2. The folder contents/purpose.
  3. The directory path (disk location) where the folder will be created.
- Do not create any directory in planning mode without this explicit user confirmation.

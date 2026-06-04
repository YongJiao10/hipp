# Repository Rules

## Concurrency Limit — HARD CAP

- **Maximum concurrent jobs/tasks/processes: 10.**
- This applies to ALL agents (Codex, Claude, or any automated runner) operating in this repository.
- Do NOT launch more than 10 parallel subprocesses, worker threads, agent tasks, or pipeline jobs at any one time, regardless of what the task or tool suggests as a default.
- Before launching a batch of parallel work, count the intended concurrency. If it exceeds 10, reduce it.

## Archive Access

- You MUST obtain explicit user permission before accessing or browsing any archive content.

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

## HippUnfold Density Guardrail

- For hippocampal density/resolution questions, do not infer actual density from command arguments alone.
- Prefer the consumed surface/CIFTI asset path and recorded `density_evidence` when available.
- If parameter density and consumed asset evidence disagree, report the mismatch and treat consumed assets as authoritative.

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
- Before running any Python-based repo task, run `python -c "import sys; print(sys.executable)"` for informational context only; do not use this interpreter for repo tasks unless it exactly matches py314.
- If command examples in docs use `python`, agents must rewrite them to the explicit interpreter path at execution time.
- If an agent accidentally runs with the wrong interpreter, stop and rerun with the correct one; do not continue mixed-environment execution.

Reference runbook: `docs/environment/agent_runtime_contract.md`

## Planning Mode Folder Creation Guardrail

- In planning mode (`plan mode`), if any new folder/directory needs to be created, the agent MUST explicitly ask the user for confirmation and clarify:
  1. The folder name.
  2. The folder contents/purpose.
  3. The directory path (disk location) where the folder will be created.
- Do not create any directory in planning mode without this explicit user confirmation.

## Deletion Safeguard

- Agents MUST NOT perform any file or directory deletion operations (e.g. `rm`, `rmdir`, `shutil.rmtree`, etc. in commands or scripts) without explicit confirmation from the user in the current session.
- Before performing any deletion, the agent must ask the user for confirmation and explain the target path and the reason.
- 禁止在未获得用户明确授权的情况下，执行任何文件或目录的删除操作（包括 `rm`, `rmdir`, `shutil.rmtree` 等）。在进行任何删除前，必须向用户确认目标路径与删除原因。

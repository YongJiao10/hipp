# Agent Runtime Contract

This repository uses a strict interpreter contract so new agents can run once and run correctly.

## Hard Rules

- Never run repository Python tasks with bare `python`.
- Use explicit interpreter paths for every command.
- Do not mix interpreters in a single task run.

## Interpreter Map

Use exactly one of these based on task type.

1. Core workflow scripts, contract tests, compile checks, documentation validation:
`/home/yojiao/.conda/envs/py314/bin/python`

2. Direct HippUnfold CLI execution:
- activate `hippo2` and pass the HippUnfold gate from `AGENTS.md`
- expected CLI path after activation: `/opt/miniconda3/envs/hippo2/bin/hippunfold`

## One-Shot Preflight (Run First)

```bash
/home/yojiao/.conda/envs/py314/bin/python -c "import sys,nibabel; print(sys.executable); print(nibabel.__version__)"
python -c "import sys; print(sys.executable)"
```

Interpretation:

- First line must print `/home/yojiao/.conda/envs/py314/bin/python` and a valid `nibabel` version.
- Second line is informational only. If it points elsewhere (for example `yazi-deps`), do not use it for repo tasks.

## Canonical Command Forms

Use these forms directly:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/hipp_parcellation_network/test_core_workflow_traceability_contract.py
/home/yojiao/.conda/envs/py314/bin/python -m py_compile scripts/core_workflow/*.py scripts/core_workflow/lib/*.py
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/01_01_prepare_group_inputs.py --help
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/06_04_build_cluster_cortex_panels.py --help
```

## Recovery Rule

If the wrong interpreter was used:

1. Stop immediately.
2. Re-run the command with the explicit correct interpreter.
3. Discard confidence in outputs from the wrong interpreter run.


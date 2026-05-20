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
/home/yojiao/.conda/envs/py314/bin/python -c "import sys,nibabel,brainspace; print(sys.executable); print(nibabel.__version__); print(brainspace.__version__)"
python -c "import sys; print(sys.executable)"
```

Interpretation:

- First line must print `/home/yojiao/.conda/envs/py314/bin/python`, a valid `nibabel` version, and a valid `brainspace` version (`0.2.1` expected for current `gradient-spectral` workflow).
- Second line is informational only. If it points elsewhere (for example `yazi-deps`), do not use it for repo tasks.

HippoMaps-style continuous gradients additionally require official package code
at `external/hippomaps` on `sys.path`. Current pinned source for this workspace:
`external/hippomaps`, commit `eebd4fe82e69f32e793ac691b8346e71780a757a`.
The `05_03`/`06_05` manifests record both the import path and commit. Do not
copy HippoMaps plotting code into this repository. For `den=512`, `06_05`
treats `512` as HippoMaps v2 in the current Python process and disables
`unfoldAPrescale`; no AP surfarea rescale is used for 512-density figures. The
py314 environment must also import HippoMaps' top-level package dependencies
used during official plotting import: `pygeodesic`, `adjustText`,
`eigenstrapping`, `wget`, `netneurotools`, `neuromaps`, `lapy`, `bctpy`, and
`parspin`. Install these without downgrading the existing py314 `numpy`,
`matplotlib`, `nilearn`,
`brainspace`, or `nibabel` stack.

## Workbench Concurrency

All core workflow Workbench calls must use
`scripts/core_workflow/lib/workbench.py`. The wrapper defaults
`CORE_WORKFLOW_WB_THREADS=1` and applies that value to the common native thread
environment variables before invoking `wb_command`, including
`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
`NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, and
`ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS`.

For `01_01_prepare_group_inputs.py`, subject-level Workbench stages are capped
at two concurrent subjects even if `--jobs` is higher. Do not bypass this cap
with direct Workbench subprocess calls.

## Canonical Command Forms

Use these forms directly:

```bash
/home/yojiao/.conda/envs/py314/bin/python tests/core_workflow/test_core_workflow_traceability_contract.py
/home/yojiao/.conda/envs/py314/bin/python -m py_compile scripts/core_workflow/*.py scripts/core_workflow/lib/*.py
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/01_01_prepare_group_inputs.py --help
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/06_04_build_cluster_cortex_panels.py --help
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/05_03_compute_hippomaps_gradients.py --help
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/06_05_render_hippomaps_gradient_products.py --help
```

## Recovery Rule

If the wrong interpreter was used:

1. Stop immediately.
2. Re-run the command with the explicit correct interpreter.
3. Discard confidence in outputs from the wrong interpreter run.

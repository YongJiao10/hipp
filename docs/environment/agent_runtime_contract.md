# Agent Runtime Contract

This repository uses a strict interpreter contract so new agents can run once and run correctly.

The selective ctxreg input integration uses the existing py314, nibabel,
NumPy and Workbench runtime. Hippocampal coordinate registration also uses
the existing SciPy interpolation/linear algebra and the installed
`antsApplyTransformsToPoints` executable at
`.cache/conda_envs/hippo2/bin/antsApplyTransformsToPoints`; set
`CORE_WORKFLOW_ANTS_POINTS` to select an explicit executable on another system.
ANTs receives native LPS points and the declared MNI-to-T1w image transform;
its output is converted back to RAS. HCP FNIRT conversion uses Workbench and
inverts the image pullback at the required points with a 0.001-mm residual
tolerance. Workbench remains single threaded, including registration calls.
No package installation or Python environment change is required.
Iowa derived cortex surfaces and their explicitly selected manifest belong in
`outputs_iowa_jj_tms/inputs/cortex_surfaces/fsLR32k/`, not `/tmp`.
Lynch PFM concatenates explicit run inputs in memory. Worktree asset links and
parent-directory output fallbacks are not part of the formal runtime contract.

## Hard Rules

- Never run repository Python tasks with bare `python`.
- Use explicit interpreter paths for every command.
- Do not mix interpreters in a single task run.

## Interpreter Map

Use exactly one of these based on task type.

1. Core workflow scripts, contract tests, compile checks, documentation validation:
`/home/yojiao/.conda/envs/py314/bin/python`

2. Direct HippUnfold CLI execution:
- activate `hippo2` with `source /opt/conda/bin/activate hippo2` and pass the HippUnfold gate from `AGENTS.md`
- expected CLI path after activation: `/home/yojiao/hpchome/hipp/.cache/conda_envs/hippo2/bin/hippunfold`
- the scientific source is the official v2.3.0 release, including the upstream
  boundary-terminal and unfolded-registration fixes; both CLI and package
  metadata must report `2.3.0`.
- only the documented cache/inference runtime adapters and CLI version-display
  correction are applied locally. Do not reapply a local scientific patch or
  substitute development `main`; follow the
  [runtime manifest](hippo2_reproducibility_manifest.md#scientific-parameters).
- the HCP166 rebuild uses the complete official participant DAG. Previously
  pruned anatomy, coordinates and transforms must regenerate when required;
  retained surfaces alone do not satisfy the full structural contract.
- retain all 96 formal outputs with `--notemp`; the standard checker must use
  `--fail-on-missing --no-cleanup`. A cleanup receipt, boundary-function test
  or partial surface chain cannot certify v2.3.0 completion.

3. Argon SGE Kong2022 launchers and their dataset adapters (including PDC/BANDA):
`/Users/yojiao/miniconda3/envs/py313/bin/python`. This is the existing Argon
runtime with nibabel and NumPy, alongside MATLAB R2021a, CBIG and Workbench.
PDC/BANDA reads metadata-declared HCP MSMAll clean CIFTIs under
`/Shared/jianglab/3_Data_Working/Depression/{PDC,BANDA}/imaging/` and validates
their cortical models as fsLR32k. No new environment or dependency is needed.
Use the shared `scripts/ArealMSHBM_Kong2022/` interface and its README for
manifest preparation and SGE execution; local py314 paths do not apply to
these Argon launchers.

## VS Code Workspace

Open the repository root as the workspace folder. The tracked
`.vscode/settings.json` sets `git.repositoryScanMaxDepth` to `0` so repository
discovery checks the workspace root. File watching excludes generated
`outputs_*`, `wrong`, and `.cache` trees. Refresh or reopen generated files when
inspecting updated results.

If Source Control remains at "Scanning folder for Git repositories", run
`Developer: Reload Window` from the Command Palette. The Git extension must
start again before Source Control can show changes. `git status --short`
reports source and documentation changes; generated outputs and archives are
excluded by `.gitignore`.

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
The `05_01`/`06_04` manifests record both the import path and commit. Do not
copy HippoMaps plotting code into this repository. For `den=512`, `06_04`
treats `512` as HippoMaps v2 in the current Python process and disables
`unfoldAPrescale`; no AP surfarea rescale is used for 512-density figures. The
py314 environment must also import HippoMaps' top-level package dependencies
used during official plotting import: `pygeodesic`, `adjustText`,
`eigenstrapping`, `wget`, `netneurotools`, `neuromaps`, `lapy`, `bctpy`, and
`parspin`. The py314 environment also includes `pytest 9.0.3` for core workflow tests. Install these without downgrading the existing py314 `numpy`,
`matplotlib`, `nilearn`,
`brainspace`, or `nibabel` stack.

## Workbench Concurrency

All core workflow Workbench calls must use
`scripts/core_workflow/lib/workbench.py`. Its default executable is
`software/workbench/bin_linux64/wb_command` (Workbench 2.2.1). Local HippUnfold
launches prepend the same directory to PATH. The wrapper defaults
`CORE_WORKFLOW_WB_THREADS=1` and applies that value to the common native thread
environment variables before invoking `wb_command`, including
`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
`NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, and
`ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS`.

For `02_01_prepare_group_inputs.py`, subject-level Workbench stages are capped
at two concurrent subjects even if `--jobs` is higher. Do not bypass this cap
with direct Workbench subprocess calls.

For `05_02_run_group_atlas.py`, native math/BLAS thread counts are now hard-capped
in script code at 1 before NumPy/SciPy imports. The script force-sets
`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
`NUMEXPR_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS` to `1` and prints a
`[thread-cap]` line at startup for auditability. Do not rely on ad-hoc shell
exports for this step.

## Canonical Command Forms

Use these forms directly:

```bash
/home/yojiao/.conda/envs/py314/bin/python tests/core_workflow/test_core_workflow_traceability_contract.py
/home/yojiao/.conda/envs/py314/bin/python -m py_compile scripts/core_workflow/*.py scripts/core_workflow/lib/*.py
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/02_01_prepare_group_inputs.py --help
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/06_03_build_cluster_cortex_panels.py --help
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/05_01_compute_hippomaps_gradients.py --help
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/06_04_render_hippomaps_gradient_products.py --help
```

## Recovery Rule

If the wrong interpreter was used:

1. Stop immediately.
2. Re-run the command with the explicit correct interpreter.
3. Discard confidence in outputs from the wrong interpreter run.

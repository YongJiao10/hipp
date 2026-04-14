# `hippo2` Reproducibility Manifest

This document records the exact environment stack that was used to run HippUnfold successfully in this workspace, so a future agent can recreate the setup without rediscovering the same pitfalls.

Snapshot date: `2026-04-13`

## What Is Recorded

This manifest captures the environment in layers:

1. Host OS and architecture
2. Conda environment `hippo2`
3. Python packages visible inside `hippo2`
4. External command-line binaries and wrappers
5. Runtime/cache layout used by the launcher
6. Known version mismatches and footguns

Raw snapshots are stored in:

- `manifests/hippo2/uname.txt`
- `manifests/hippo2/sw_vers.txt`
- `manifests/hippo2/conda-info.json`
- `manifests/hippo2/conda-explicit.txt`
- `manifests/hippo2/conda-list.txt`
- `manifests/hippo2/pip-freeze.txt`

## Host Layer

```text
Item              Value
----------------- -------------------------------------------------
OS                macOS
Machine           arm64
Shell             zsh
Python runtime    /opt/miniconda3/envs/hippo2/bin/python
Conda base        /opt/miniconda3
Workspace root    /Users/jy/Documents/HippoMaps-network-first
```

The launcher and wrappers assume this is an Apple Silicon Mac.

## Conda Layer

The active environment used for HippUnfold runs is `hippo2`.

```text
Package      Version      Build / Channel          Notes
-----------  -----------  -----------------------  ---------------------------
bash         5.2.37       h5b2bd6a_0               Installed into hippo2 so
                                                  Snakemake can use a Bash
                                                  that supports the launcher
                                                  shell features.
c3d          0.6.0        pyhd8ed1ab_0             Present in conda list.
hippunfold   2.0.0        py_0 / khanlab           CLI version used in runs.
nilearn      0.10.4       pyhd8ed1ab_0             Downgraded for compatibility.
nnunet       1.7.1        pypi_0                   Used by HippUnfold inference.
numpy        1.26.4       py311h7125741_0          Downstream numeric stack.
pandas       1.5.3        py311h4eec4a9_1          Critical fix for removed
                                                  `DataFrame.append`.
pybids       0.16.5       pyhd8ed1ab_0             BIDS path handling.
snakebids    0.15.0       pyhdfd78af_0             HippUnfold workflow glue.
snakemake    9.19.0       hdfd78af_0               Workflow engine.
torch        2.5.1        pip/conda mixed           Used by nnUNet.
torchvision  0.26.0        pip/conda mixed           Used by nnUNet.
```

The full package inventory is in `manifests/hippo2/conda-list.txt`.

The explicit conda lock snapshot is in `manifests/hippo2/conda-explicit.txt`.

The `pip freeze` snapshot is archival and audit-friendly, but it should not be treated as the only bootstrap source because it contains file URL references from the build environment.

## Python Layer

The environment was checked with `conda run -n hippo2 python`.

```text
Item                Observed value
------------------  ---------------------------------------------
Python              3.11.15
hippunfold module   /opt/miniconda3/envs/hippo2/lib/python3.11/
                    site-packages/hippunfold/__init__.py
nnUNet_predict      /opt/miniconda3/envs/hippo2/bin/nnUNet_predict
reg_aladin          /opt/miniconda3/envs/hippo2/bin/reg_aladin
bash                /opt/miniconda3/envs/hippo2/bin/bash
```

Important package versions observed from Python metadata:

```text
Package      Version
-----------  --------
hippunfold   1.5.2rc2
nnunet       1.7.1
nilearn      0.10.4
pandas       1.5.3
numpy        1.26.4
snakemake    9.19.0
snakebids    0.15.0
pybids       0.16.5
torch        2.5.1
torchvision  0.26.0
```

### Important HippUnfold Version Mismatch

This environment contains a deliberate-looking but confusing mismatch that must be documented:

```text
Source                       Observed version
--------------------------  -----------------
`conda list -n hippo2`       2.0.0
`hippunfold --version`       2.0.0
`importlib.metadata`         1.5.2rc2
`site-packages` dist-info    hippunfold-1.5.2rc2.dist-info
```

The successful runs in this workspace were driven by the `hippunfold --version == 2.0.0` CLI behavior.

Do not rely on `importlib.metadata.version("hippunfold")` alone when reproducing the environment. That lookup currently resolves to the stale `1.5.2rc2` dist-info record even though the CLI reports `2.0.0`.

## Binary Layer

The actual runnable commands used by the workflow are:

```text
Command            Location / Behavior
-----------------  ----------------------------------------------------------
hippunfold         /opt/miniconda3/envs/hippo2/bin/hippunfold
nnUNet_predict     /opt/miniconda3/envs/hippo2/bin/nnUNet_predict
reg_aladin         /opt/miniconda3/envs/hippo2/bin/reg_aladin
bash               /opt/miniconda3/envs/hippo2/bin/bash
wb_command         /Applications/wb_view.app/Contents/usr/bin/wb_command
LN2_LAYERS         resolved through `scripts/LN2_LAYERS` wrapper
```

There is no PATH-stable `wb_command` in this repo. Use the wrapper in `scripts/wb_command`.

### Workbench Wrapper

[`scripts/wb_command`](/Users/jy/Documents/HippoMaps-network-first/scripts/wb_command) wraps the macOS Workbench binary and handles three cases:

1. Fails fast if the app binary is missing
2. For `-volume-to-surface-mapping`, routes through the fallback helper
3. On Apple Silicon macOS, runs the app via `arch -x86_64`

### LAYNII Wrapper

[`scripts/LN2_LAYERS`](/Users/jy/Documents/HippoMaps-network-first/scripts/LN2_LAYERS) wraps the real `LN2_LAYERS` binary so `.nii.gz` rims are decompressed to `.nii` before execution and the expected outputs are recompressed afterward.

## Runtime Layer

The launcher [`scripts/run_hippunfold_local.sh`](/Users/jy/Documents/HippoMaps-network-first/scripts/run_hippunfold_local.sh) is part of the environment definition.

It sets the following runtime layers before starting HippUnfold:

```text
Setting / Path                                 Purpose
--------------------------------------------  -------------------------------------------
`runtime/hippunfold_cache`                    Central cache root for this repo
`runtime_source_cache`                        Snakemake runtime source cache
`snakemake_conda`                             Optional local conda prefix for rule envs
`conda_pkgs`                                  Local conda package cache
`.runtime_py/sitecustomize.py`                 Multiprocessing + torch sharing tweaks
`PATH` precedence                             `scripts/` first, external binaries second
`XDG_CACHE_HOME`                               Redirects cache writes out of `~/Library`
`HIPPUNFOLD_CACHE_DIR`                        HippUnfold cache root
`CONDA_SUBDIR=osx-64`                         Needed for some x86_64-only dependencies
`nnUNet_n_proc_DA=1`                           Prevents runaway multiprocessing
`OMP/MKL/OPENBLAS/VECLIB=1`                  Keeps numeric libs single-threaded
`KMP_DUPLICATE_LIB_OK=TRUE`                   Avoids OpenMP duplicate-runtime crashes
`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`          Torch compatibility guard
```

The launcher also inserts a generated `sitecustomize.py` that forces:

```python
mp.set_start_method("fork")
torch.multiprocessing.set_sharing_strategy("file_system")
```

That file is created under `.runtime_py/` inside the repo root.

## HippUnfold Inputs and Outputs

The local launcher was run with:

```text
subject                     100610 / 102311 / 102816
input_dir                   data/hippunfold_input
output_dir                  outputs_migration/dense_corobl_batch/sub-<id>/hippunfold
workflow-profile            none
sdm                         env-modules
output-density              512
output-spaces               corobl
autotop_labels              hipp dentate
```

This manifest records the environment only. The actual subject outputs live under:

- `outputs_migration/dense_corobl_batch/sub-100610/hippunfold`
- `outputs_migration/dense_corobl_batch/sub-102311/hippunfold`
- `outputs_migration/dense_corobl_batch/sub-102816/hippunfold`

## Known Footguns

```text
Issue                                              Why it matters
-------------------------------------------------  ---------------------------------------------
`pandas` >= 2.x                                   Breaks `DataFrame.append` in a helper script
Stale `hippunfold-1.5.2rc2.dist-info`             Confuses Python metadata checks
`wb_command` on arm64 without `arch -x86_64`      Can misbehave or crash on this Mac
`~/Library/Caches` writes                         Must be redirected for this workspace
Hidden `.cache` growth                           Replaced with `runtime/hippunfold_cache`
`LN2_LAYERS` on `.nii.gz` rims                    Needs wrapper decompression/recompression
`nnUNet` multiprocessing on macOS                Needed a fork/sharing workaround here
```

## Rebuild Recipe

From scratch, a future agent should:

1. Create or activate the `hippo2` environment.
2. Install the conda snapshot from `manifests/hippo2/conda-explicit.txt`.
3. Verify `conda list -n hippo2` reports `hippunfold 2.0.0`.
4. Verify `hippunfold --version` reports `2.0.0`.
5. Ignore `importlib.metadata.version("hippunfold")` if it still reports `1.5.2rc2`.
6. Make sure `scripts/run_hippunfold_local.sh` is the only launcher used for local runs.
7. Make sure the repo `scripts/` directory stays first on `PATH` so `wb_command` and `LN2_LAYERS` wrappers win.
8. Keep caches under `runtime/hippunfold_cache`, not in hidden `~/.cache` or `~/Library/Caches`.
9. Use `data/hippunfold_input` as the input root and `outputs_migration/dense_corobl_batch/sub-<id>/hippunfold` as the output root.

## Minimal Verification Commands

```bash
conda activate hippo2
hippunfold --version
conda list -n hippo2 | rg '^hippunfold\s'
conda run -n hippo2 python -c 'import importlib.metadata as m; print(m.version("hippunfold"))'
conda run -n hippo2 bash -lc 'command -v reg_aladin; command -v nnUNet_predict; command -v bash'
```

For Workbench, use:

```bash
scripts/wb_command -help
```

On this Mac, that wrapper will internally invoke:

```bash
arch -x86_64 /Applications/wb_view.app/Contents/usr/bin/wb_command
```

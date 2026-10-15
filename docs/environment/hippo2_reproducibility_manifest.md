# `hippo2` Reproducibility Manifest

This document records the exact environment stack that was used to run HippUnfold successfully in this workspace, so a future agent can recreate the setup without rediscovering the same pitfalls.

Snapshot date: `2026-04-13`

Related hard runtime rules for agents are in:

- `docs/environment/agent_runtime_contract.md`
- `AGENTS.md` (`Agent Runtime Environment Contract (Hard)`)

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

Bootstrap sources for new servers:

- `environment/hippo2_server.yml`
- `_archive/deprecated_2026-05-20/scripts_pre_core_workflow/bootstrap_hippo2_server.sh`

## Host Layer

```text
Item              Value
----------------- -------------------------------------------------
OS                Linux
Machine           x86_64
Shell             bash/zsh
Python runtime    /home/yojiao/hpchome/hipp/.cache/conda_envs/hippo2/bin/python
Conda base        /opt/conda
Workspace root    /home/yojiao/hpchome/hipp
```

The launcher and wrappers assume this local Linux host uses `/opt/conda` to activate the repo-local `hippo2` environment.

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
mamba        2.3.2        hae34dd5_2 / conda-forge  Snakemake conda frontend
                                                  when `--use-conda` is enabled.
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

The supported bootstrap spec for new servers is `environment/hippo2_server.yml`.
Most entries in that file are pinned to exact versions recovered from
`manifests/hippo2/conda-list.txt` and `manifests/hippo2/pip-freeze.txt`.
The only intentionally unpinned entries are `greedyreg` and `laynii`, because
the captured manifest does not record an exact build string for those external
entry-point packages.
It is intentionally narrower than the lock snapshot so it is easier to read and
maintain, while the explicit snapshot remains the bit-for-bit historical record.

The `pip freeze` snapshot is archival and audit-friendly, but it should not be treated as the only bootstrap source because it contains file URL references from the build environment.

## Python Layer

The environment should be checked with `conda run -p /home/yojiao/hpchome/hipp/.cache/conda_envs/hippo2 python` so the repo-local prefix is unambiguous.

```text
Item                Observed value
------------------  ---------------------------------------------
Python              3.11.15
hippunfold module   /home/yojiao/hpchome/hipp/.cache/conda_envs/hippo2/lib/python3.11/
                    site-packages/hippunfold/__init__.py
nnUNet_predict      /home/yojiao/hpchome/hipp/.cache/conda_envs/hippo2/bin/nnUNet_predict
reg_aladin          /home/yojiao/hpchome/hipp/.cache/conda_envs/hippo2/bin/reg_aladin
bash                /home/yojiao/hpchome/hipp/.cache/conda_envs/hippo2/bin/bash
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
`conda list -p /home/yojiao/hpchome/hipp/.cache/conda_envs/hippo2`       2.0.0
`hippunfold --version`       2.0.0
`importlib.metadata`         1.5.2rc2
`site-packages` dist-info    hippunfold-1.5.2rc2.dist-info
```

The successful runs in this workspace were driven by the `hippunfold --version == 2.0.0` CLI behavior.

A local runtime hotfix is also required in this workspace: `hippunfold/workflow/scripts/gen_volume_tsv.py` must use `pd.concat(...)` instead of the removed `DataFrame.append()` API when running under the current pandas build. The active `hippo2` runtime used for direct local runs also needs `seaborn` importable because `plot_subj_subfields.py` executes in the main environment when `--use-conda` is disabled.

Do not rely on `importlib.metadata.version("hippunfold")` alone when reproducing the environment. That lookup currently resolves to the stale `1.5.2rc2` dist-info record even though the CLI reports `2.0.0`.

## Binary Layer

The actual runnable commands used by the workflow are:

```text
Command            Location / Behavior
-----------------  ----------------------------------------------------------
hippunfold         /home/yojiao/hpchome/hipp/.cache/conda_envs/hippo2/bin/hippunfold
nnUNet_predict     /home/yojiao/hpchome/hipp/.cache/conda_envs/hippo2/bin/nnUNet_predict
reg_aladin          /home/yojiao/hpchome/hipp/.cache/conda_envs/hippo2/bin/reg_aladin
bash               /home/yojiao/hpchome/hipp/.cache/conda_envs/hippo2/bin/bash
wb_command         /home/yojiao/hpchome/hipp/software/workbench/bin_linux64/wb_command
LN2_LAYERS         resolved through `_archive/deprecated_2026-05-20/scripts_pre_core_workflow/LN2_LAYERS` wrapper
```

Core workflow code resolves Workbench directly from
`software/workbench/bin_linux64/wb_command` via `scripts/core_workflow/lib/workbench.py`.
Legacy scripts may still call the wrapper in `_archive/deprecated_2026-05-20/scripts_pre_core_workflow/wb_command`.

### HippUnfold External Binary Bundle

`_archive/deprecated_2026-05-20/scripts_pre_core_workflow/run_hippunfold_local.sh` prepends `HIPPUNFOLD_EXTERNAL_BIN_DIR` to `PATH`
before calling `hippunfold`. The default on this Mac is:

```text
/Applications/ITK-SNAP.app/Contents/bin
```

That bundle is the expected source of the external registration helpers used by
the HippUnfold workflow, especially `greedy` and `c3d_affine_tool`.

On a server, make the equivalent directory available somewhere stable and set:

```bash
export HIPPUNFOLD_EXTERNAL_BIN_DIR=/path/to/your/bin-dir
```

Then verify the bundle with:

```bash
command -v c3d
command -v greedy
command -v c3d_affine_tool
```

If `c3d` is missing, install the conda-forge `c3d` package into `hippo2` or an
equivalent environment. If `greedy` or `c3d_affine_tool` is missing, the
launcher will not have the external helper bundle it expects.

### Greedy Binary Source

The HippUnfold workflow expects the medical-image registration binary `greedy`,
not the unrelated GeoPandas package named `greedy`.

On this workspace, the launcher gets `greedy` from the ITK-SNAP bundle listed
above. On a server, the shortest supported install path is usually the conda
package:

```bash
conda install -p /home/yojiao/hpchome/hipp/.cache/conda_envs/hippo2 -c khanlab greedyreg
```

That package is the one that provides the `greedy` executable used by the
registration helpers. After installation, verify:

```bash
conda run -p /home/yojiao/hpchome/hipp/.cache/conda_envs/hippo2 bash -lc 'command -v greedy; greedy -help | head -n 1'
```

If you already have an ITK-SNAP bundle or another vendor-supplied binary
directory, you can keep using it by pointing `HIPPUNFOLD_EXTERNAL_BIN_DIR` at
that directory instead.

### Workbench Command Resolution

Active `scripts/core_workflow` steps resolve and execute:

- `software/workbench/bin_linux64/wb_command`

Legacy helpers may still use [`_archive/deprecated_2026-05-20/scripts_pre_core_workflow/wb_command`](/home/yojiao/hpchome/hipp/_archive/deprecated_2026-05-20/scripts_pre_core_workflow/wb_command), which is a thin wrapper around the same binary:

1. Fails fast if the app binary is missing
2. Otherwise execs `software/workbench/bin_linux64/wb_command` directly

### LAYNII Wrapper

[`_archive/deprecated_2026-05-20/scripts_pre_core_workflow/LN2_LAYERS`](/Users/jy/Documents/HippoMaps-network-first/_archive/deprecated_2026-05-20/scripts_pre_core_workflow/LN2_LAYERS) wraps the real `LN2_LAYERS` binary so `.nii.gz` rims are decompressed to `.nii` before execution and the expected outputs are recompressed afterward.

### LAYNII Binary Source

`LN2_LAYERS` is part of the [LAYNII](https://github.com/layerfMRI/LAYNII) tool
suite. On this workspace it may come either from:

1. A conda installation of `laynii`
2. A manually unpacked LAYNII binary directory

For a server, the shortest supported install path is usually:

```bash
conda install -p /home/yojiao/hpchome/hipp/.cache/conda_envs/hippo2 -c conda-forge laynii
```

After installation, verify the executable is visible:

```bash
conda run -p /home/yojiao/hpchome/hipp/.cache/conda_envs/hippo2 bash -lc 'command -v LN2_LAYERS; LN2_LAYERS -help | head -n 1'
```

If the binary lives outside `hippo2`, point the launcher at it explicitly:

```bash
export HIPPUNFOLD_LN2_LAYERS_BIN=/path/to/LAYNII/LN2_LAYERS
```

The wrapper in `_archive/deprecated_2026-05-20/scripts_pre_core_workflow/LN2_LAYERS` exits with code `127` when neither the
environment variable nor `PATH` yields a real binary.

## Runtime Layer

The active launcher `scripts/core_workflow/01_01_run_hippunfold_t1w_local.py` and its closed contract module are part of the environment definition.

They set the following runtime layers before starting HippUnfold:

### Kong2022 Route A MATLAB Runtime

The SGE-only Kong2022_ArealMSHBM smoke does not use `hippo2` for its diffusion
embedding helper. `external/CBIG/utilities/matlab/speedup_gradients/CBIG_SPGrad_diffusion_embedding.sh`
activates `/Users/yojiao/miniconda3` environment `py313`, then imports CBIG's
vendored Python package from:

```text
external/CBIG/external_packages/python/mapalign-master/mapalign/embed.py
```

The outer subject task constrains OMP, MKL, OpenBLAS, NumExpr, and vecLib to
one thread, disables dynamic OMP/MKL thread changes, selects the oneMKL
`COMPATIBLE` CNR branch, and starts MATLAB with `-singleCompThread`. The
exported task environment is inherited by the diffusion helper and its
`py313` Python process. This is the maintained same-runtime reproducibility
configuration; it is not a cross-version, cross-backend, or cross-CPU bitwise
guarantee.

The outer Argon dataset adapter in
`scripts/ArealMSHBM_Kong2022/` is separate from the CBIG MATLAB
diffusion helper above. For `DATASET=iowa7t`, the adapter does not require a
Python runtime: it uses fixed branch CIFTI paths plus
`wb_command -file-information` for timepoint counts, writes branch-scoped
labels under the production `hipp_outputs` tree, writes provenance with shell
JSON emission after MATLAB succeeds, and writes qsub logs under
`hipp_outputs/runs/logs/<run_id>/`.

If that directory is missing, MATLAB reaches `Compute diffusion embedding
matrix ...` and fails with `ModuleNotFoundError: No module named 'mapalign'`.
After embedding, profile generation also requires the CBIG fsLR downsample
template:

```text
external/CBIG/data/templates/surface/fs_LR_32k_downsample_900/fslr_downsample_900mesh_parcellation.dlabel.nii
```

The same profile-generation stage calls the CBIG MATLAB stats helper:

```text
external/CBIG/utilities/matlab/stats/CBIG_corr.m
```

The individual parcellation connectedness check calls:

```text
external/CBIG/utilities/matlab/parcellation/CBIG_ComputeConnectedComponentsFromSurface.m
```

In this workspace that helper is implemented with MATLAB `graph/conncomp`
instead of the `matlab_bgl` `components_mex` call to avoid relying on the old
MEX path during the EM connectedness check.

The SGE smoke launcher requests `mem_free=32G` and uses `-tc 1` for the
two-subject array so only one MATLAB EM job runs at a time.
Even with this serialized launcher, the current smoke still reaches SGE exit
`137` after the second EM `Estep..`; this is the current runtime blocker for
Route A MATLAB label generation.

```text
Setting / Path                                 Purpose
--------------------------------------------  -------------------------------------------
`HIPPUNFOLD_CACHE_ROOT`                      Locked per-dataset/subject runtime scratch derived by the contract; caller overrides are scrubbed
`HIPPUNFOLD_CLEANUP_RUNTIME`                 Not accepted by the stage-01 contract; runtime scratch is retained for reproducibility
`runtime/hippunfold_asset_store_shared`       Hardcoded shared asset store for reusable template/model downloads
`runtime/hippunfold_asset_store_shared/hippunfold_assets`       Canonical `HIPPUNFOLD_CACHE_DIR`; the contract writes this value
`hippunfold_assets/model/trained_model.3d_fullres.Task101_hcp1200_T1w.nnUNetTrainerV2.model_best.tar`  Canonical local T1w nnUNet model; required before a run
`runtime/hippunfold_asset_store_shared/runtime_source_assets`   Canonical `HIPPUNFOLD_RUNTIME_SOURCE_CACHE`; the contract writes this value
`snakemake_conda`                             Not used by the locked stage-01 contract
`conda_pkgs`                                  Not used by the locked stage-01 contract
`PIP_CACHE_DIR`                              Derived per-subject wheel cache under the locked runtime root
`TMPDIR`                                      Derived per-subject temporary scratch under the locked runtime root
`.runtime_py/sitecustomize.py`                 Multiprocessing + torch sharing tweaks
`PATH` precedence                             `scripts/` first, then the validated `hippo2/bin`
`XDG_CACHE_HOME`                               Derived runtime scratch under the locked runtime root
`HIPPUNFOLD_USE_GPU`                          Not an environment override; `--use_gpu` is fixed in the argv contract
`LD_LIBRARY_PATH`                             Prepend `hippo2/lib` so the env libstdc++ overrides the system copy
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

HippUnfold model/template assets are intentionally not run-specific. The closed
contract sets `HIPPUNFOLD_CACHE_DIR` and `HIPPUNFOLD_RUNTIME_SOURCE_CACHE` to the
shared asset store under `/home/yojiao/hpchome/hipp/runtime/hippunfold_asset_store_shared`
and scrubs every caller-provided `HIPPUNFOLD_*` override before launching.
`HIPPUNFOLD_CACHE_ROOT` is derived from the dataset and subject, so two batches
cannot silently choose different runtime roots or model locations. Runtime
scratch is retained; no cleanup override is accepted by stage 01.

The active public T1w launcher `scripts/core_workflow/01_01_run_hippunfold_t1w_local.py`
accepts only `<subject> --dataset {hcp,stanford,iowa_jj_tms,banda,pdc}` plus `--dry-run`.
The closed dataset preset fixes the input root, output root, T1w extension filter,
T1w modality, `corobl` output space, density `512`, autotopology labels, GPU
inference, one core, the N4 preprocessing choice, and the canonical shared local cache/model paths. All
`HIPPUNFOLD_*` caller overrides are scrubbed; free-form Snakemake arguments are
not accepted. The local model is consumed through the canonical cache output, so
the workflow is never pruned with `--omit-from download_nnunet_model`. Before a real run, the launcher repeats the hippo2 gate: exact executable resolution, CLI version `2.0.0`, and T1w/512 help support.
JJ_TMS is the only preset using `extension=.nii`; the other presets use
`extension=.nii.gz`.

The BANDA/PDC presets consume HCP bias-field-corrected
`T1w_acpc_dc_restore.nii.gz` products and set `skip_preproc=True`, automatically
adding `--skip-preproc`. Established HCP, Stanford, and JJ_TMS launch contracts
retain `False`; the setting defaults to `False` in `DatasetPreset`. The public
dry-run JSON exposes the selected value and command without reading inputs.
The setting follows documented upstream bias-field correction of the consumed
input, rather than scanner field strength or voxel size. See
[stage-01 input preprocessing](../../scripts/core_workflow/README.md#input-preprocessing).

The BANDA/PDC batch drivers under `outputs_<dataset>/work/run_hippunfold_batch.sh`
use the checker's default automatic cleanup after each stage-01 run.

The output checker `scripts/core_workflow/01_02_check_hippunfold_outputs.py`
accepts legacy success markers, HippUnfold 2.x `N of N steps (100%) done`
markers, and no-op logs. When a 2.x log has no `localrule all` input list, it
derives the expected manifest from the successful subject output tree.

After dataset-appropriate full-DAG acceptance exists, the generic checker uses
its cached full-output contract for a one-time complete subject validation and
then removes `sub-<id>/{anat,coords,warps,cifti}` from the subject payload.
Before accepting a real HippUnfold manifest it additionally requires fixed L/R
`space-corobl`, `space-T1w`, and `space-unfold` `den-512` midthickness surfaces,
the `den-512` surface spec, and both unetf3d Dice QC files. This prevents a
template-only space-unfold pair from being mistaken for a complete run.
It retains `surf/`, `metric/`, and `qc/` for the maintained core workflow and
writes `core_workflow_cleanup_receipt.json` beside the payload. New receipts
are bound to the full contract SHA-256. Historical schema-1 `cleanup_complete`
receipts with an older fingerprint are reconciled only when their approved
directory list and removal evidence are complete, no pruned directory remains,
and every current retained expected output is present. Missing retained anchors
or non-approved outputs still fail validation.
`--no-cleanup` performs validation without pruning. The removed directories
cannot be used for HippUnfold resume, volumetric structural analysis, or CIFTI
derivative reuse and must be regenerated from the original inputs if needed.
`--contract-source-cache` may seed a missing dataset-local cache from a
validated compatible full-output contract; the local cache records the source
path and contract SHA-256 before any subject is checked or cleaned.

For persistent conda installs outside the launcher, `~/.condarc` is set to keep
`pkgs_dirs` and `envs_dirs` under the repo-local `.cache/` tree so package
downloads and future env creation stop consuming `/home/yojiao`.

For interactive shells, use the canonical gate in `AGENTS.md`: activate
`hippo2`, verify the executable resolves to the repo-local 2.0.0 build, and
verify the 512-density help output before a direct run.

GPU inference, modality, density, output space, workflow profile, scheduler
data-management mode, core count, N4 preprocessing choice, and T1w extension filter are fixed in the
stage-01 argv contract. They are not controlled by `HIPPUNFOLD_USE_GPU`,
`HIPPUNFOLD_USE_CONDA`, or any other caller environment variable.

## HippUnfold Inputs and Outputs

The local launcher was run with:

```text
subject                     100610 / 102311 / 102816
input_dir                   data/hippunfold_input
output_dir                  outputs_hcp/inputs/hcp/hippunfold/dense_corobl_batch/sub-<id>/hippunfold
workflow-profile            none
sdm                         env-modules
output-density              512
output-spaces               corobl
autotop_labels              hipp dentate
```

This manifest records the environment only. The actual subject outputs live under:

- `outputs_hcp/inputs/hcp/hippunfold/dense_corobl_batch/sub-100610/hippunfold`
- `outputs_hcp/inputs/hcp/hippunfold/dense_corobl_batch/sub-102311/hippunfold`
- `outputs_hcp/inputs/hcp/hippunfold/dense_corobl_batch/sub-102816/hippunfold`

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
`/home/yojiao` quota                              Keep `conda`/`pip` caches in repo-local `.cache`
Parallel HippUnfold with shared disposable cache  Can fail due to cache-cleanup races across workers
```

## Rebuild Recipe

From scratch, a future agent should:

1. Create or activate the repo-local `hippo2` prefix at `/home/yojiao/hpchome/hipp/.cache/conda_envs/hippo2`.
2. Install the conda snapshot from `manifests/hippo2/conda-explicit.txt`.
3. Verify `conda list -p /home/yojiao/hpchome/hipp/.cache/conda_envs/hippo2` reports `hippunfold 2.0.0`.
4. Verify `hippunfold --version` reports `2.0.0`.
5. Ignore `importlib.metadata.version("hippunfold")` if it still reports `1.5.2rc2`.
6. Build or update the server environment from `environment/hippo2_server.yml` before first run.
7. Use the active stage-01 launcher only through its closed preset interface:
   `/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/01_01_run_hippunfold_t1w_local.py <subject> --dataset <hcp|stanford|iowa_jj_tms>`.
8. Do not pass positional input/output paths, model paths, `HIPPUNFOLD_*` overrides, or free-form Snakemake arguments.
9. Make sure the validated `hippo2/bin/hippunfold` executable and canonical local model are resolvable before a real run.
10. Make sure `c3d`, `greedy`, `LN2_LAYERS`, and `N4BiasFieldCorrection` resolve from the conda env or explicit binary paths.
11. Runtime scratch is derived under `runtime/hippunfold_workdirs/<dataset>/sub-<id>/` and retained; do not share or clean it between subjects.
12. HCP, Stanford, and JJ-TMS input/output roots and extension filters are defined only in `scripts/core_workflow/lib/hippunfold_run_contract.py`.

## Activation Path Guardrail

Current canonical Conda paths for the local host:

- `/opt/conda/...` as the canonical activation root on the local host
- `/home/yojiao/hpchome/hipp/.cache/conda_envs/hippo2/...` as the canonical `hippo2` environment prefix

Canonical local activation:

```bash
source /opt/conda/bin/activate hippo2
which hippunfold
```

`which hippunfold` must print:

```text
/home/yojiao/hpchome/hipp/.cache/conda_envs/hippo2/bin/hippunfold
```

## Minimal Verification Commands

```bash
source /opt/conda/bin/activate hippo2
which hippunfold
hippunfold --version
hippunfold --modality T2w --output_density 512 --help | grep -F '512'
conda list -p /home/yojiao/hpchome/hipp/.cache/conda_envs/hippo2 | rg '^hippunfold\s'
conda list -p /home/yojiao/hpchome/hipp/.cache/conda_envs/hippo2 | rg '^c3d\s'
conda run -p /home/yojiao/hpchome/hipp/.cache/conda_envs/hippo2 python -c 'import importlib.metadata as m; print(m.version("hippunfold"))'
conda run -p /home/yojiao/hpchome/hipp/.cache/conda_envs/hippo2 bash -lc 'command -v c3d; command -v reg_aladin; command -v nnUNet_predict; command -v bash'
```

For Workbench, use:

```bash
_archive/deprecated_2026-05-20/scripts_pre_core_workflow/wb_command -help
```

On this Mac, that wrapper will internally invoke:

```bash
_archive/deprecated_2026-05-20/scripts_pre_core_workflow/wb_command -help
```

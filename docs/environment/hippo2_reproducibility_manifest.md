# hippo2 runtime and reproducibility

The stage-01 launcher and installed hippo2 runtime select HippUnfold 2.4.0 and
`multihist7sgi`. Output validation uses the complete contract from a successful
run of the selected release.

Runtime revision: 2026-11-01. The installed scientific workflow is the official
HippUnfold v2.4.0 release. Local changes are limited to cache/inference runtime
adapters and CLI version display. No local boundary, reference-plane or
Workbench scientific correction is part of this runtime.
The environment manager remains Conda; the complete upstream Pixi stack is
not installed.

HCP runs use the complete official participant DAG and the output contract of
the selected release. The [result assessment](hippunfold_result_impact.md)
records observations from the earlier v2.3.0 reconstruction.

## Runtime stack

| Component | Version or location |
| --- | --- |
| Environment | `/home/yojiao/hpchome/hipp/.cache/conda_envs/hippo2` |
| Activation | `/opt/conda/bin/activate hippo2` |
| Python | 3.13.13; supported by v2.4.0's `>=3.12,<3.15` requirement |
| HippUnfold | 2.4.0, official release source plus documented resource/runtime adapters |
| Snakemake / Snakebids | 9.27.1 / 0.15.0 |
| nnUNet / PyTorch | 1.7.1 / 2.14.1+cu126, CUDA inference |
| NumPy / SciPy / pandas | 2.5.4 / 1.18.1 / 3.1.0 |
| PyVista / VTK | 0.49.1 / 9.7.0 |
| nibabel / nilearn / pybids | 5.4.2 / 0.13.1 / 0.21.0 |
| ANTs / Greedy / NiftyReg / LAYNII | 2.6.5 / 0.0.20250712 / 1.5.69.6 / 2.7.0 |
| SQLite / libsqlite | 3.54.1, matching h9ffa6c4_0 / h13e7031_0 builds |
| Convert3D / ITK | 1.4.2 (Khanlab h109cef4_0) / 5.4.7 |
| Workbench | Official 2.2.1, `<repository>/software/workbench/bin_linux64/wb_command` |

`environment/hippo2_server.yml` declares the target stack and official v2.4.0 source.
The installed effective CLI and package metadata report 2.4.0. Verify the loaded
workflow and executable; the environment name and Conda package listing alone
do not establish the algorithm version. Python scientific packages remain at
the pip layer over the native Conda runtime.

The PyTorch wheel uses CUDA 12.6 on driver 550.163.01 and an L40S; CUDA 13
wheels require a newer driver. Snakemake requires packaging below 26. Greedy
uses the Conda-forge heeecbdd_7 build; its generic CLI label is 1.3.0-alpha.
Convert3D uses the Khanlab h109cef4_0 build linked against ITK 5.4.7. SQLite and
libsqlite remain pinned together. The local stack retains Snakebids 0.15,
PyBIDS 0.21 and unified nnUNet 1.7.1 inference rather than replacing the entire
dependency environment during this upgrade.

## Launch and environment gate

Use explicit py314 for repository orchestration:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/01_01_run_hippunfold_t1w_local.py 102311 --dataset hcp --dry-run
taskset --cpu-list 0-7,32-39 nice -n 10 /home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/01_01_run_hippunfold_t1w_local.py --dataset hcp --manifest manifests/hcp_7t_hippocampus_struct_complete_166.txt --jobs 12 --keep-going
```

The first command prints the resolved contract; it does not execute the
scientific workflow. Run the batch command only after the runtime gate and
the approved result-root preparation are complete.

Direct HippUnfold execution requires this gate, in order:

```bash
source /opt/conda/bin/activate hippo2
which hippunfold
hippunfold --version
hippunfold --modality T1w --output_density 512 --help
```

The executable must resolve to the canonical hippo2 prefix, the version must
match the release in the environment specification, and T1w/512 help must
include the inner/outer registration options.
Stage 01 additionally validates the resource adapter and CUDA. A pending batch
performs this gate once; a new batch checks again. CLI probes have a
300-second limit. An unavailable CUDA device stops dispatch rather than
selecting CPU inference.

The closed presets retain T1w, density 512, corobl output, hippocampus and
dentate, unfolded registration and GPU inference. HCP, BANDA and PDC use
`--skip-preproc` for bias-corrected inputs; Stanford and JJ-TMS retain N4.
Caller scientific arguments and HIPPUNFOLD environment overrides are rejected.

## Scientific parameters

The prepared upgrade selects the official [v2.4.0 release](https://github.com/khanlab/hippunfold/releases/tag/v2.4.0).
The adult T1w model remains Task101_hcp1200_T1w, nnUNetTrainerV2, model_best,
with the declared CITI168 template and multihist7sgi atlas. Establish density from
consumed surface paths and shapes rather than arguments alone.

The official release replaces the faulty terminal-contiguity operation with
four contiguous alternating boundary arcs, and creates/samples the unfolded
registration reference on the configured midthickness plane. It also uses
geometric image centres for registration initialization and adds source/sink
and surface QC. These are upstream scientific implementations, not the retired
local whole-component and exact-slice patches. See [issue580](https://github.com/khanlab/hippunfold/issues/580),
[issue581](https://github.com/khanlab/hippunfold/issues/581),
[PR582](https://github.com/khanlab/hippunfold/pull/582) and
[PR583](https://github.com/khanlab/hippunfold/pull/583).

Inner/outer registration retains the official greedy defaults:
`-n 50x50x50 -s 1.7320508075688772vox 0.707vox`, with method `greedy` and
update-field smoothing variance 3.0. Optional SyN remains an explicit upstream
method, not a replacement selected by missing outputs.

Use the release tag rather than development main. Version 2.4.0 pairs its
updated surface processing with the regenerated multihist7sgi atlas. Do not
install a local scientific patch or local Workbench binary to make a failed
subject appear complete.

## Complete HCP166 reconstruction

The formal root is
`outputs_hcp/inputs/hippunfold/dense_corobl_batch/`, with each subject's complete
participant run under `sub-<id>/hippunfold/`. The source roster is
`manifests/hcp_7t_hippocampus_struct_complete_166.txt`.

The official all-target DAG generates the required volumes, surfaces,
coordinates, CIFTI, transforms, spec files and QC. Do not force old surface
timestamps to suppress dependencies scheduled by the selected release.
Completion requires its complete output contract, matching run metadata and
official QC results. Old output checks and cleanup receipts cannot be relabelled
as completion of a newer release.

## Local resource resolution

Shared assets remain in `runtime/hippunfold_asset_store_shared/`:
`hippunfold_assets/` is the canonical model/template/atlas cache and
`runtime_source_assets/` is the Snakemake source cache. Complete selected assets
must register no remote dependency and issue no HTTP requests. Missing or
incomplete caches fail with their exact missing paths. A cached model never
justifies pruning participant rules with `--omit-from`.

The maintained adapters are:

- `scripts/core_workflow/lib/hippunfold_local_assets.patch`: cached resources,
  GPU inference slots, bounded nnUNet workers and CLI version display.
- `scripts/core_workflow/lib/snakemake_local_source_cache.patch`: refresh
  changed local source files while retaining remote-source behavior.

The prepared 2.4.0 adapter changes `workflow/rules/download.smk`,
`workflow/rules/pred.smk` and `workflow/lib/local_assets.py`. Its `run.py` and
`config/snakebids.yml` changes make CLI version display read installed package
metadata rather than the stale upstream static value. No local scientific
correction is installed or reapplied.

## Batch resources and retained outputs

The approved HCP profile defaults to eight subjects, accepts up to twelve,
uses one CPU core per subject and two GPU inference slots, and binds CPUs to
`0-7,32-39` with niceness 10. The profile is the HCP-only process-count
exception; other tasks retain their limits. Other datasets allow two subjects,
with BANDA one, and one GPU slot.

GPU slots use abstract Unix sockets `hippunfold-gpu0-<uid>` and
`hippunfold-gpu0-<uid>-1`; inference waits for a free slot. The lock helper
starts Python with `-S` before exec. nnUNet preprocessing/save workers and
OMP, MKL, OpenBLAS, NumExpr, vecLib, ITK and VTK threads remain one. Per-subject
runtime directories are `runtime/hippunfold_workdirs/<dataset>/sub-<id>/`;
shared resource caches are separate. PATH places repository wrappers and the
project Workbench before hippo2/bin; LD_LIBRARY_PATH prepends hippo2/lib.
The existing fork/Torch-sharing startup hook remains in `.runtime_py/`.

Use `--notemp` to retain generated intermediates. Standard full-output checks
use `--fail-on-missing --no-cleanup`.
Retain runtime scratch and failed outputs for diagnosis; do not run
cleanup-shadow or delete runtime directories.
The selected atlas must use its own complete official output contract and an
exact matching run contract. The standalone checker can inspect older results;
the stage-01 launcher rejects an old-atlas contract when checking a new run.
A retained-only cleanup receipt is not sufficient. The normal launcher stops
new dispatch after a failure; the HCP166 reconstruction explicitly selects
manifest-only `--keep-going` to attempt the remaining subjects. That option
preserves error-level QC and returns a nonzero final status if any subject
fails; it does not turn failures into completed results.

## Rebuild recipe

Create the canonical environment from `environment/hippo2_server.yml` with
flexible channel priority for the selected Conda-forge Greedy and Khanlab
Convert3D builds. Install the official Workbench 2.2.1 distribution under
`software/workbench/`; no local resampling fix is selected.

Install the clean official release before applying adapters so package metadata
records the release version. Stop batch dispatch and wait for all HippUnfold
subprocesses to exit before replacing installed workflow files. For a fresh
source checkout, with the `cached-http` storage plugin installed:

```bash
git clone --single-branch --branch v2.4.0 https://github.com/khanlab/hippunfold.git tmp/hippunfold_v240_fresh_source
.cache/conda_envs/hippo2/bin/pip install --no-deps tmp/hippunfold_v240_fresh_source
patch --batch --forward --dry-run -p1 -d .cache/conda_envs/hippo2/lib/python3.13/site-packages/hippunfold < scripts/core_workflow/lib/hippunfold_local_assets.patch
patch --batch --forward --no-backup-if-mismatch -p1 -d .cache/conda_envs/hippo2/lib/python3.13/site-packages/hippunfold < scripts/core_workflow/lib/hippunfold_local_assets.patch
```

Prepare the exact `multihist7sgi` atlas resources before dispatch.

After a Snakemake reinstall, dry-run/apply its separate maintained source-cache
patch. Activate hippo2, pass the declared-version/T1w/512 gate and use the closed stage-01
launcher. Complete cached resources should need no network request. Keep build
material under the repository `tmp/`; follow the exact-path authorization rule
for deletions or production replacement.

## Separate Argon MATLAB environment

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
`hippunfold_assets/model/trained_model.3d_fullres.Task101_hcp1200_T1w.nnUNetTrainerV2.model_best.tar`  Canonical T1w nnUNet archive; reuse the unpacked model or archive when present, otherwise use the declared upstream download
`runtime/hippunfold_asset_store_shared/runtime_source_assets`   Canonical `HIPPUNFOLD_RUNTIME_SOURCE_CACHE`; the contract writes this value
`snakemake_conda`                             Not used by the locked stage-01 contract
`conda_pkgs`                                  Not used by the locked stage-01 contract
`PIP_CACHE_DIR`                              Derived per-subject wheel cache under the locked runtime root
`TMPDIR`                                      Derived per-subject temporary scratch under the locked runtime root
`.runtime_py/sitecustomize.py`                 Multiprocessing + torch sharing tweaks
`PATH` precedence                             `scripts/`, project Workbench, then the validated `hippo2/bin`
`XDG_CACHE_HOME`                               Derived runtime scratch under the locked runtime root
`HIPPUNFOLD_USE_GPU`                          Not an environment override; `--use_gpu` is fixed in the argv contract
`HIPPUNFOLD_GPU_SLOTS`                        Locked to two for HCP and one for other presets
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

# hippo2 runtime and reproducibility

Runtime revision: 2026-10-22. The local structural workflow uses official
HippUnfold v2.2.0 scientific source in the refreshed hippo2 dependency stack.
The local resource/inference patch is required. The environment manager remains
Conda; adopting the upstream Pixi dependency stack is outside this configuration.

## Runtime stack

| Component | Active version or location |
| --- | --- |
| Environment | `/home/yojiao/hpchome/hipp/.cache/conda_envs/hippo2` |
| Activation | `/opt/conda/bin/activate hippo2` |
| Python | 3.13.13 |
| HippUnfold | 2.2.0, official tagged source plus the documented local patch |
| Snakemake / Snakebids | 9.27.1 / 0.15.0 |
| nnUNet / PyTorch | 1.7.1 / 2.14.1+cu126, CUDA inference |
| NumPy / SciPy / pandas | 2.5.4 / 1.18.1 / 3.1.0 |
| PyVista / VTK | 0.49.1 / 9.7.0 |
| nibabel / nilearn / pybids | 5.4.2 / 0.13.1 / 0.21.0 |
| ANTs / Greedy / NiftyReg / LAYNII | 2.6.5 / 0.0.20250712 / 1.5.69.6 / 2.7.0 |
| SQLite / libsqlite | 3.54.1, matching h9ffa6c4_0 / h13e7031_0 builds |
| Convert3D / ITK | 1.4.2 (Khanlab h109cef4_0) / 5.4.7 |
| Workbench | `/home/yojiao/FASTANS/workbench/bin_linux64/wb_command`, on PATH |

`environment/hippo2_server.yml` declares the current package stack and installs
HippUnfold from the official v2.2.0 tag. The old Conda HippUnfold package was removed through the package manager
before reinstalling the official v2.2.0 wheel; its obsolete PyVista constraint
no longer controls native-library updates. Python scientific packages are
installed at the pip layer over the native Conda runtime. The effective CLI must report 2.2.0; verify its
executable and loaded workflow rather than treating Conda metadata as the
algorithm version. The Python package metadata now reports 2.2.0 as well.

The PyTorch wheel uses CUDA 12.6 on driver 550.163.01 and an L40S.
Snakemake 9.27.1 requires packaging below 26; nilearn 0.13.1 satisfies the
HippUnfold range and that shared dependency. Eido 0.2.6 supports pandas 3.
Greedy uses the Conda-forge heeecbdd_7 build; its CLI retains the generic
1.3.0-alpha label but reports an August 26, 2026 build. Convert3D uses the
explicit Khanlab h109cef4_0 build linked against ITK 5.4.7. The upstream
Prefix.dev libitk547 build channel returned HTTP 403 on this host; the declared
Khanlab artifact is selected explicitly instead. Reapply the local source-cache
patch after updating Snakemake.

SQLite and libsqlite are pinned together. The matched library restores the
SQLite API required by Python, PyBIDS and Snakemake; ncurses 6.6 and readline
8.3 satisfy the selected SQLite CLI build.

The local stack retains Snakebids 0.15/PyBIDS 0.21 and unified nnUNet 1.7.1
inference. Upstream declares Snakebids 0.14 and a separate legacy nnUNet
environment. This repository validates its concrete local stack against the
tagged scientific source before launching production.

The Conda portion of the environment declaration passes a fresh-prefix dry run.
The complete 102311 structural test finishes all 418 rules with 84 required
outputs present and no output pruning. Both consumed T1w hippocampal surfaces
have 512 finite vertices. Template-agreement Dice is 0.76279 (L) and 0.79944 (R).
A complete atlas scan finds 14 right targets with candidate physical positions
more than 1 mm apart (maximum separation 13.46 mm); no left target meets that
criterion. The existing output check records a native-unfold correspondence
conflict. The conflict precedes unfolded registration; candidate separation is
not anatomical error against ground truth. Structural execution completion does
not approve this hemisphere for functional downstream analysis.

## Launch and environment gate

Use the explicit py314 interpreter for the repository launcher:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/01_01_run_hippunfold_t1w_local.py 102311 --dataset hcp
taskset --cpu-list 0-7,32-39 nice -n 10 /home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/01_01_run_hippunfold_t1w_local.py --dataset hcp --manifest manifests/hcp_7t_hippocampus_struct_complete_166.txt --jobs 8
```

The closed presets select HCP, Stanford, JJ-TMS, BANDA or PDC input/output roots,
T1w filtering, N4 preprocessing choice, 512 density, corobl output, hippocampus
and dentate topology, unfolded registration and GPU inference. BANDA/PDC retain
`--skip-preproc` for their already bias-corrected inputs. HCP/Stanford/JJ-TMS
retain N4. Free-form scientific arguments and caller HIPPUNFOLD overrides are
rejected. `--dry-run` prints the resolved JSON contract without processing.

The gate validates the resource adapter, CUDA availability, canonical executable,
CLI 2.2.0 and the T1w/512 help with inner/outer registration options. A batch
with pending subjects runs this full gate once and reuses it only within that
batch process. A new batch checks again. CLI probes have a 300-second limit.
An unavailable CUDA device stops the run; it does not select CPU inference.

## Scientific parameters

The adult T1w model remains Task101_hcp1200_T1w, nnUNetTrainerV2, model_best.
Reference templates and the multihist7 atlas remain the declared resources.
Actual consumed den-512 surface paths establish the output density.

Inner/outer registration uses the official v2.2.0 greedy defaults:
`-n 50x50x50 -s 1.7320508075688772vox 0.707vox`. The launch contract explicitly
selects method `greedy` and update-field smoothing variance 3.0. The source
retains the optional SyN implementation and method-specific transform paths.
The official PyVista point-ID compatibility fix and pandas-compatible volume
statistics implementation are included. Development changes after the v2.2.0
tag are outside this runtime.

## Local resource resolution

Shared models/templates reside in `runtime/hippunfold_asset_store_shared/`:
`hippunfold_assets/` is the canonical resource cache, and
`runtime_source_assets/` is the Snakemake runtime source cache.

Complete selected templates, atlas surfaces/registration metrics/subfield
labels and unpacked model folds are validated against their declared filenames.
Complete cached assets register no remote dependencies and issue no HTTP
requests. Absent resources retain their original declared download URLs.
Incomplete resource directories fail with exact missing paths. Reusing a model
never prunes the subject processing DAG with `--omit-from`.

The maintained source patches are:

- `scripts/core_workflow/lib/hippunfold_local_assets.patch`: cached-resource
  resolution, CLI version display, host-local GPU inference slot and bounded nnUNet workers.
- `scripts/core_workflow/lib/snakemake_local_source_cache.patch`: refresh changed
  local source files by modification time while retaining remote-source behavior.

The HippUnfold patch changes four explicit files: config/snakebids.yml,
workflow/rules/download.smk, workflow/rules/nnunet.smk and
workflow/lib/local_assets.py. It applies to clean official v2.2.0 source.
Other scientific processing files remain official v2.2.0 code.

## Batch resources and retained outputs

The shared-host HCP profile runs eight subjects, one CPU core per subject,
and two GPU inference slots. Its launch uses CPU affinity `0-7,32-39` (16 physical
cores on the current IDAS host) and niceness 10. The container has a 32-CPU quota;
this profile leaves at least half that CPU capacity available to other tasks.
Other datasets retain two subjects (BANDA one) and one GPU slot. This specifically
approved HCP profile supersedes the earlier two-subject/ten-task budget for this
workload; other repository task limits remain unchanged.

GPU slots use the abstract Unix sockets `hippunfold-gpu0-<uid>` and
`hippunfold-gpu0-<uid>-1`. Contending inference waits for a free declared slot;
process exit releases it automatically. The small lock helper starts Python
with `-S`, avoiding the Torch startup hook before exec; the actual inference
still loads the normal startup hook. nnUNet preprocessing and save workers
remain one each. OMP, MKL, OpenBLAS, NumExpr, vecLib, ITK and VTK limits remain
one. The existing fork/Torch-sharing hook remains in `.runtime_py/`.
Per-subject runtime directories remain under
`runtime/hippunfold_workdirs/<dataset>/sub-<id>/`; shared asset caches remain
separate from these subject-specific directories. PATH places repository
wrappers before hippo2/bin; LD_LIBRARY_PATH prepends hippo2/lib. The Torch
checkpoint compatibility setting remains enabled.

Output checks run serially with `--fail-on-missing --no-cleanup`, using the
existing full-output contract. All structural payloads remain available,
including anat, coords, warps and cifti. The per-subject output_check.json
records the check and its exact run contract. Resume skips a subject only when
this report matches the current contract and every declared full output remains
present. Older cleanup receipts do not establish v2.2.0 completion. Processing
or checking failures stop new dispatch; already running subjects finish and
receive their checks.

## Rebuild recipe

For a fresh environment, create the canonical prefix from the declaration and
provide the external Workbench binary on PATH:

```bash
CONDA_CHANNEL_PRIORITY=flexible /opt/conda/bin/conda env create --prefix /home/yojiao/hpchome/hipp/.cache/conda_envs/hippo2 --file environment/hippo2_server.yml
```

Use flexible channel priority for the explicitly selected Conda-forge Greedy
and Khanlab Convert3D artifacts. For an existing hippo2 upgrade, install the clean official v2.2.0 package and
the declared refreshed scientific/runtime dependencies. Use the CUDA 12.6
PyTorch wheel on this driver-550 host; CUDA 13 wheels require a newer driver.
Build/install the clean source before patching it so the package version remains
2.2.0.

Apply the maintained HippUnfold patch to a clean source copy, then copy the four
explicit patched files into the installed package. This also replaces any
unowned adapter retained by pip during reinstallation. Pause batch dispatch and
wait until all active HippUnfold subprocesses exit before replacing installed
files; Snakemake reloads rules during execution:

```bash
git clone --single-branch --branch v2.2.0 https://github.com/khanlab/hippunfold.git tmp/hippunfold_220_source
.cache/conda_envs/hippo2/bin/pip install --no-deps tmp/hippunfold_220_source
patch --batch --forward --dry-run -p1 -d tmp/hippunfold_220_source/hippunfold < scripts/core_workflow/lib/hippunfold_local_assets.patch
patch --batch --forward -p1 -d tmp/hippunfold_220_source/hippunfold < scripts/core_workflow/lib/hippunfold_local_assets.patch
hippunfold_site=/home/yojiao/hpchome/hipp/.cache/conda_envs/hippo2/lib/python3.13/site-packages/hippunfold
cp tmp/hippunfold_220_source/hippunfold/config/snakebids.yml "$hippunfold_site/config/snakebids.yml"
cp tmp/hippunfold_220_source/hippunfold/workflow/rules/download.smk "$hippunfold_site/workflow/rules/download.smk"
cp tmp/hippunfold_220_source/hippunfold/workflow/rules/nnunet.smk "$hippunfold_site/workflow/rules/nnunet.smk"
cp tmp/hippunfold_220_source/hippunfold/workflow/lib/local_assets.py "$hippunfold_site/workflow/lib/local_assets.py"
```

After reinstalling Snakemake, dry-run and apply its separate local-source-cache
patch to the installed snakemake directory. Activate hippo2, verify `which
hippunfold`, run the 2.2.0/T1w/512 gate, and use the closed stage-01 launcher.
For tests and temporary build material use the repository tmp/ workspace.
Do not delete structural outputs or runtime directories as an upgrade step.

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
`PATH` precedence                             `scripts/` first, then the validated `hippo2/bin`
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

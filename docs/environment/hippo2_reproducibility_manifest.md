# hippo2 runtime and reproducibility

Runtime revision: 2026-10-25. The scientific baseline is official HippUnfold
v2.3.0 in the existing hippo2 dependency stack. Local changes are limited to
cache/inference runtime adapters and CLI version display. No local boundary,
reference-plane or Workbench scientific correction is part of this baseline.
The environment manager remains Conda; the complete upstream Pixi stack is
not installed.

The complete official participant DAG has been attempted for all HCP166
subjects in the formal result tree, with 12 concurrent subjects and two GPU
inference slots. Exactly 163 subjects satisfy the current 96-product contract
and official error-level QC. Subjects 233326 and 249947 failed surface QC;
942658 failed right-dentate boundary-loop extraction. Generated intermediates
remain retained. Current scope and review status are recorded in the
[rebuild assessment](hippunfold_result_impact.md) and
[execution workspace](../../tmp/hcp7t_official_v230_20261025/README.md).

## Runtime stack

| Component | Version or location |
| --- | --- |
| Environment | `/home/yojiao/hpchome/hipp/.cache/conda_envs/hippo2` |
| Activation | `/opt/conda/bin/activate hippo2` |
| Python | 3.13.13; supported by v2.3.0's `>=3.12,<3.15` requirement |
| HippUnfold | 2.3.0, official release source plus documented resource/runtime adapters |
| Snakemake / Snakebids | 9.27.1 / 0.15.0 |
| nnUNet / PyTorch | 1.7.1 / 2.14.1+cu126, CUDA inference |
| NumPy / SciPy / pandas | 2.5.4 / 1.18.1 / 3.1.0 |
| PyVista / VTK | 0.49.1 / 9.7.0 |
| nibabel / nilearn / pybids | 5.4.2 / 0.13.1 / 0.21.0 |
| ANTs / Greedy / NiftyReg / LAYNII | 2.6.5 / 0.0.20250712 / 1.5.69.6 / 2.7.0 |
| SQLite / libsqlite | 3.54.1, matching h9ffa6c4_0 / h13e7031_0 builds |
| Convert3D / ITK | 1.4.2 (Khanlab h109cef4_0) / 5.4.7 |
| Workbench | Official 2.2.1, `<repository>/software/workbench/bin_linux64/wb_command` |

`environment/hippo2_server.yml` declares the stack and official v2.3.0 source.
The effective CLI and package metadata must both report 2.3.0. Verify the loaded
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
be 2.3.0, and T1w/512 help must include the inner/outer registration options.
Stage 01 additionally validates the resource adapter and CUDA. A pending batch
performs this gate once; a new batch checks again. CLI probes have a
300-second limit. An unavailable CUDA device stops dispatch rather than
selecting CPU inference.

The closed presets retain T1w, density 512, corobl output, hippocampus and
dentate, unfolded registration and GPU inference. BANDA/PDC keep
`--skip-preproc` for already bias-corrected inputs; HCP/Stanford/JJ-TMS keep N4.
Caller scientific arguments and HIPPUNFOLD environment overrides are rejected.

## Scientific parameters

Use the official [v2.3.0 release](https://github.com/khanlab/hippunfold/releases/tag/v2.3.0).
The adult T1w model remains Task101_hcp1200_T1w, nnUNetTrainerV2, model_best,
with the declared CITI168 template and multihist7 atlas. Establish density from
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

Use the release tag rather than development main. The maintainer states that
new surface filtering on main requires a newly generated unfolded atlas; it
is outside this runtime. See the [maintainer's release guidance](https://github.com/khanlab/hippunfold/issues/580#issuecomment-6398866471).
Do not install a local scientific patch, main-branch mesh repair or a local
Workbench binary to make a failed subject appear complete.

## Complete HCP166 reconstruction

The formal root is
`outputs_hcp/inputs/hippunfold/dense_corobl_batch/`, with each subject's complete
participant run under `sub-<id>/hippunfold/`. The source roster is
`manifests/hcp_7t_hippocampus_struct_complete_166.txt`.

Unchanged native physical meshes and boundary-distance metrics were retained,
but the postprocessed tissue volumes, nnUNet segmentations, GM masks,
preprocessed T1w volumes and transforms required for complete formal outputs
had been pruned. Reusing the surface inputs alone cannot produce a complete
v2.3.0 result set. The official all-target DAG must regenerate missing upstream
prerequisites and all affected downstream products, including volumetric
subfields, CIFTI, sampled coordinates, spec files and QC. Do not force old
surface timestamps to suppress dependencies scheduled by the official DAG.

The pi Argon model completed the step-3 dependency/reuse review; this was a
model review, not an Argon-cluster computation. Its substantive conclusion is
that the full official DAG is required. See the [review](../../tmp/hcp7t_official_v230_20261025/step3_argon_review.md).
The actual full-DAG execution has now attempted all 166 subjects; 163 completed
and three failed as described above. The
[step-5 pi Argon review](../../tmp/hcp7t_official_v230_20261025/step5_argon_review.md)
independently checked the actual newly generated outputs and confirmed the
completion counts, failure causes, coordinate frames and midthickness inverse
statistics. See the [result assessment](hippunfold_result_impact.md) for
the distinction between boundary-band collapse, local folding and workflow
failure.

The formal output contract contains 96 products per subject: the existing
84 structural products plus four source/sink/surface QC HTML/verification
files and eight hemisphere/structure QC metric JSONs. This full contract,
matching v2.3.0 run metadata, actual files and official QC results are required
for completion; old output checks and cleanup receipts cannot be relabelled.

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

The HippUnfold adapter changes `workflow/rules/download.smk`,
`workflow/rules/nnunet.smk` and `workflow/lib/local_assets.py`. Its fourth file,
`config/snakebids.yml`, only corrects CLI version display to 2.3.0. No local
scientific correction is installed or reapplied.

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

This reconstruction performs no 01_02 cleanup. `--notemp` keeps anat, coords,
warps, CIFTI and other generated intermediates. Every standard output check
must pass `--fail-on-missing --no-cleanup`. Retain runtime scratch and failed
outputs for diagnosis; do not run cleanup-shadow or delete runtime directories.
The output checker may mark a subject complete only when all 96 declared
formal products remain present and the exact current run contract matches.
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

Build/install the clean official v2.3.0 source before applying adapters so the
package metadata remains exactly 2.3.0. Stop batch dispatch and wait for all
HippUnfold subprocesses to exit before replacing installed workflow files.
For a fresh source checkout:

```bash
git clone --single-branch --branch v2.3.0 https://github.com/khanlab/hippunfold.git tmp/hippunfold_v230_source
.cache/conda_envs/hippo2/bin/pip install --no-deps tmp/hippunfold_v230_source
patch --batch --forward --dry-run -p1 -d tmp/hippunfold_v230_source/hippunfold < scripts/core_workflow/lib/hippunfold_local_assets.patch
patch --batch --forward -p1 -d tmp/hippunfold_v230_source/hippunfold < scripts/core_workflow/lib/hippunfold_local_assets.patch
hippunfold_site=/home/yojiao/hpchome/hipp/.cache/conda_envs/hippo2/lib/python3.13/site-packages/hippunfold
cp tmp/hippunfold_v230_source/hippunfold/config/snakebids.yml "$hippunfold_site/config/snakebids.yml"
cp tmp/hippunfold_v230_source/hippunfold/workflow/rules/download.smk "$hippunfold_site/workflow/rules/download.smk"
cp tmp/hippunfold_v230_source/hippunfold/workflow/rules/nnunet.smk "$hippunfold_site/workflow/rules/nnunet.smk"
cp tmp/hippunfold_v230_source/hippunfold/workflow/lib/local_assets.py "$hippunfold_site/workflow/lib/local_assets.py"
```

After a Snakemake reinstall, dry-run/apply its separate maintained source-cache
patch. Activate hippo2, pass the 2.3.0/T1w/512 gate and use the closed stage-01
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

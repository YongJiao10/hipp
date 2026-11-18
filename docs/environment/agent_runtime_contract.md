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
NIH rTMS hippocampal QC uses this same runtime and the fMRIPrep MNI-to-T1w
image transform paired with the exact preprocessed T1w consumed by HippUnfold.
Its input symlink identifies either the longitudinal `sub-<ID>/anat/` export
or the single-visit `sub-<ID>/ses-<N>/anat/` export below
`Depression/NIH_rTMS/derivatives/restfmri/`. Its explicit XCP-D
outlier TSV binds original motion frames to the retained NIfTI series; no
new dependency or HippUnfold runtime adapter is introduced.
Iowa derived cortex surfaces and their explicitly selected manifest belong in
`outputs_iowa_jj_tms/inputs/cortex_surfaces/fsLR32k/`, not `/tmp`.
Lynch PFM concatenates explicit run inputs in memory. Worktree asset links and
parent-directory output fallbacks are not part of the formal runtime contract.

## Hippocampal figure runtime

Hippocampal label panels use the existing py314 BrainSpace 0.2.1 and VTK 9.6.1
stack. Run headless figure commands with `xvfb-run -a` when no display is
available; `LP_NUM_THREADS=1` and `VTK_SMP_MAX_THREADS=1` limit rendering
threads. Workbench remains required for cortical scene rendering and
the documented image-processing and registration stages.

## Gradient comparison runtime

Gradient parcellation uses the local py314 numerical stack: NumPy 2.5.4,
SciPy 1.16.3, scikit-learn 1.9.1, BrainSpace 0.2.1 and numba 0.68.1 (required
by NumPy 2.5). BrainSpace supplies the Pearson affinity; the diffusion-map
eigendecomposition is the repository's exact solver in `lib/hippomaps_adapter.py`. Keep SciPy at 1.16.x: `06_04` imports HippoMaps plotting through
`netneurotools`, whose releases up to 0.3.0 import a private SciPy function
removed in SciPy 1.17. Stages `05_01` and `05_02` write the installed versions and the
full git commit to their manifests as `runtime_versions` and `git_commit`.
OVR omits the deprecated explicit `penalty` argument and retains default
L2. Run the HCP comparison on IDAS with one heavy fit at a time and
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1`;
keep combined active work at or below 10. Exploratory results belong under the
repository `tmp/` workspace. Formal output trees remain
read-only during this experiment. Tests in an isolated checkout must declare
the original shipped Kong400 dlabel explicitly; do not install another atlas
or substitute a different ROI registry.

## Subject-aware parcellation comparison runtime

The HCP experiment in `scripts/hcp_ctxreg_affinity_comparison/` uses the
existing py314 stack. The Bayesian step has been smoke-tested on CPU with
PyTorch 2.9.1+cu128 and the authors' HierarchBayesParcel checkout at
`d4031910ebd16e9013d756eef4d057ba7106d64c` (tag `v.1.0.1`). Pass the checkout
with `--bayes-source`. The runner imports the pinned source directly. Both
the numerical libraries and PyTorch CPU pools run with one thread. Use
`--device cuda` on IDAS to move the pinned vMF solver to the L40S with IEEE
float32 matmul throughout. The old upstream tensor factories require the
legacy CUDA default tensor type in this dedicated process. Keep the device
fixed across the reported comparisons.

Run from the experimental worktree. Supply the main checkout and the previous
Avg FC/Avg Sim run explicitly. Set `--routes pfm --smoothing 2mm` for the
development condition, and use a fresh `--out` under the worktree's `tmp/`.

```bash
git clone --branch v.1.0.1 --depth 1 https://github.com/DiedrichsenLab/HierarchBayesParcel.git \
  tmp/subject-aware-parcellation/HierarchBayesParcel

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  /home/yojiao/.conda/envs/py314/bin/python \
  scripts/hcp_ctxreg_affinity_comparison/compare_subject_models.py \
  --input-root /home/yojiao/hpchome/hipp \
  --reference-run /home/yojiao/hpchome/hipp/tmp/ctxreg-mean-subject-similarity \
  --bayes-source tmp/subject-aware-parcellation/HierarchBayesParcel \
  --out tmp/subject-aware-parcellation/standard-pilot \
  --routes pfm --smoothing 2mm
```

Pass the fitted output and the same route selection to
`scripts/hcp_ctxreg_affinity_comparison/annotate_subject_models.py` using
`--out` and `--routes`. The default annotates both new methods. Use
`--render-only` under the headless figure runtime to assemble the saved
label GIFTIs into the standard hippocampal panels. Per-condition
`fit_failures.json` records failures to satisfy the prespecified fit for
a fold. Paired scores use folds where both methods in that contrast produced
a usable map. OVR requires one observed profile for each participant and parcel.
The experiment records any ineligible maps in `network_annotation_unavailable.json`
and still computes both Fisher-z annotations where the parcel has observations.

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
- the scientific source is the official release pinned in
  `environment/hippo2_server.yml`; both CLI and package metadata must report
  the pinned release version.
- only the documented cache/inference runtime adapters and CLI version-display
  correction are applied locally. Do not reapply a local scientific patch or
  substitute development `main`; follow the
  [runtime manifest](hippo2_reproducibility_manifest.md#scientific-parameters).
- the HCP rebuild uses the complete official participant DAG. Previously
  pruned anatomy, coordinates and transforms must regenerate when required;
  retained surfaces alone do not satisfy the full structural contract.
- retain the selected release's formal outputs with `--notemp`; the standard
  checker must use `--fail-on-missing --no-cleanup` and the formal output
  contract. A cleanup receipt, boundary-function test or partial surface chain
  cannot certify completion.

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
at two concurrent subjects even if `--jobs` is higher. Raw sampling and
smoothing use coordinator threads to launch Workbench, without additional
Python worker processes. Workbench itself remains single threaded. Run the
current BANDA/PDC continuation with raw `--jobs 2` and mask `--jobs 1`, counting
the batch controller and coordinator threads within the total cap of 10.
That CPU-only continuation starts with an empty `LD_LIBRARY_PATH`; the
Workbench launcher supplies its bundled libraries. This avoids the inherited
CUDA/TensorRT paths and the launcher's extra `echo | grep` subprocess pipeline,
leaving room for status inspection under the cap. No package versions change.
Do not bypass the subject cap with direct Workbench subprocess calls.
The surface sampler joins the disconnected registered hemispheres for one
volume-to-surface invocation per physical run, then saves separate hemisphere
signals. This avoids reading and decompressing the same BOLD twice without
changing the interpolation method, coordinates or output vertex order.

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

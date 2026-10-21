# Repository Rules

## Temporary Workspace

- `<repository>/tmp/` is the only temporary workspace for this project.

## Concurrency Limit — Identify the System First

- Before launching parallel work or an SGE array, identify the active system
  from `SYSTEM_NAME`, `CLUSTER_NAME`, `SGE_CLUSTER_NAME`, or the hostname.
  Accept only `idas` or `argon`; if detection is ambiguous, stop and set
  `SYSTEM_NAME` explicitly.
- On **IDAS**, the hard cap is **10 total concurrent jobs/tasks/processes**.
- On **Argon**, the Kong2022 ArealMSHBM standard is UI `-tc 5` plus PINC
  `-tc 20`, for a **20-task total cap**. Do not apply the IDAS 10-task cap to
  an Argon array.
- These limits apply to all agents (Codex, Claude, or any automated runner)
  operating in this repository. Count intended concurrency before launching a
  batch; user-provided overrides must remain within the detected system cap.

## Archive Access

- You MUST obtain explicit user permission before accessing or browsing any archive content.
- Output archives belong under `wrong/outputs_<dataset>/<original-relative-path>`.
  Preserve their names and internal hierarchy; keep new archives outside active
  `outputs_*` trees. Existing migrations still require an exact approved path list.

## Doc Sync Rule

- Write project documentation in English, using the current complete datasets
  and their authoritative analysis rosters. Keep instructions focused on current
  inputs, methods, outputs, and usage.
- Do not retain incremental-ingestion rosters, batch histories, or instructions
  for partial uploads in active workflows or documentation.
- Use subtractive updates (减法更新): delete or replace incorrect/outdated statements in
  place. Do not retain them and append another statement explaining that they
  are wrong; keep one current, correct instruction instead of accumulating
  corrections.
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
## HCP Cortex-Control Naming Contract

- The only formal HCP cortex-control branch names are `routea_ctxdrop` and
  `routea_ctxreg`. Active resolvers, launchers, consumers, manifests, and
  documentation must use these names without `recovered`, `regenerated`, dates,
  run IDs, or `_full_k3_fisherz` suffixes.
- The formal HCP roots, relative to `outputs_hcp/`, are:
  `inputs/kong2022/shared_inputs/<route>/`,
  `work/hippomaps_gradients/<route>/`,
  `work/group_atlas/<route>/`, and
  `present/overviews/<route>/`. Method, atlas, smoothing, coverage, and
  hemisphere structure remains inside those roots; assignment variants such as
  `fisherz` and `normalized_top20_count` remain explicit files/metadata.
- Dates and run identifiers belong in logs and manifests only. Do not create a
  new dated/recovered/regenerated formal result tree, compatibility symlink, or
  existence-order fallback. A repair replaces the formal product in place.
- Real scientific branches, including 22P versus 36P and method or atlas
  variants, remain distinct and must not be merged as old/new versions.
- Any migration, rebuild, or cleanup of existing result trees requires an
  exact user-approved path list for that operation. Until approval, keep the
  output trees read-only and report unresolved or legacy paths explicitly.


- For hippocampal density/resolution questions, do not infer actual density from command arguments alone.
- Prefer the consumed surface/CIFTI asset path and recorded `density_evidence` when available.
- If parameter density and consumed asset evidence disagree, report the mismatch and treat consumed assets as authoritative.

## HippUnfold Environment Gate

- For direct `hippunfold` runs in this repository, the canonical local runtime environment is `hippo2`.
- Do not trust `conda list` or the environment name alone as proof of HippUnfold compatibility.
- Required validation sequence before a direct HippUnfold run on the local host:
  1. `source /opt/conda/bin/activate hippo2`
  2. `which hippunfold` must resolve to `/home/yojiao/hpchome/hipp/.cache/conda_envs/hippo2/bin/hippunfold`
  3. `hippunfold --version` must return `2.0.0`
  4. `hippunfold --modality T1w --output_density 512 --help` must succeed and show `512`
- Do not use `importlib.metadata.version("hippunfold")` as authoritative version proof. In this workspace it can still report `1.5.2rc2` even when the CLI is the validated `2.0.0` build.
- If the gate fails, repair in place with:

```bash
CONDA_SAFETY_CHECKS=disabled conda install -y -p /home/yojiao/hpchome/hipp/.cache/conda_envs/hippo2 -c khanlab -c conda-forge -c bioconda khanlab::hippunfold=2.0.0=py_0 --force-reinstall
```

- The local runtime also requires the resource patches documented in
  `docs/environment/hippo2_reproducibility_manifest.md#local-resource-resolution`.
  Reapply the HippUnfold patch after reinstalling HippUnfold and the local-source
  cache patch after reinstalling Snakemake. Complete cached resources must not
  register remote dependencies or issue HTTP requests; absent resources use the
  original declared download URL. Existing incomplete caches report the exact
  missing files. Stage 01 checks that the resource adapter is installed.
- Do not treat `bioconda::hippunfold` as sufficient evidence of native `512` support.
- Avoid forbidden workflow version labels for this project: `1.5.2`, `1.5.2-pre.2`, `1.5.2rc2`.

## Agent Runtime Environment Contract (Hard)

- Do not use bare `python` for repository tasks.
- Always use an explicit interpreter path based on task type:
  - Core workflow scripts/tests/doc-contract checks: `/home/yojiao/.conda/envs/py314/bin/python`
  - Direct HippUnfold CLI runs: `hippo2` gate above (`/home/yojiao/hpchome/hipp/.cache/conda_envs/hippo2/bin/hippunfold` after `/opt/conda` activation)
- Before running any Python-based repo task, run `python -c "import sys; print(sys.executable)"` for informational context only; do not use this interpreter for repo tasks unless it exactly matches py314.
- If command examples in docs use `python`, agents must rewrite them to the explicit interpreter path at execution time.
- If an agent accidentally runs with the wrong interpreter, stop and rerun with the correct one; do not continue mixed-environment execution.

Reference runbook: `docs/environment/agent_runtime_contract.md`

## Local vs Argon Environment Boundary (Hard)

This repository is used from two different execution contexts. Do not mix their
paths or Python environments.

## Stanford Causal-Connectome 22P core_workflow inputs

The authoritative Stanford 22P source is the `DataOrg_PFM_22P` tree below.
These are the three paths used by the core workflow input stage:

- Local 22P CIFTI surface time series (for extracting cortex time series):
  `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/DataOrg_PFM_22P/sub-*/func/*_task-rest_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii`
- Local 22P denoised BOLD volume time series:
  `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/DataOrg_PFM_22P/sub-*/func/*_task-rest_space-MNI152NLin6Asym_res-2_desc-denoised_bold.nii.gz`
- Local 22P Kong2022 labels:
  `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/ArealMSHBM_Kong2022_22P/sub-*/lh_labels.txt`
  and
  `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/ArealMSHBM_Kong2022_22P/sub-*/rh_labels.txt`

On Argon, replace the local Jianglab prefix with `/Shared/jianglab`:
`/Shared/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/DataOrg_PFM_22P/`
and
`/Shared/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/ArealMSHBM_Kong2022_22P/`.

For this dataset, `surf` means the cortex-extraction `fsLR den-91k`
`dtseries.nii` file, not a structural `.surf.gii` file. Do not substitute
`DataOrg_PFM_36P` or the unversioned `ArealMSHBM_Kong2022` tree when a 22P
input is required.

The canonical seven 22P overview cohorts are: NTHC, TEHC, NTS, NTS Any MDD,
NTS Pure MDD, TIS, and TIS PTSD.

## Iowa 3T JJ_TMS subgroup density naming

For Iowa 3T JJ_TMS subgroup products, an unversioned path is the default 36P
branch: `routea_pure_mdd` and `routea_comorbid_mdd`. The corresponding 22P
branch must append _22p everywhere: `routea_pure_mdd_22p` and
`routea_comorbid_mdd_22p`, including FC, gradient, group-atlas, weighted, and
presentation roots. Never describe an unversioned Iowa subgroup result as 22P.

### Local Repository / Core Workflow Context

- Use this context for local code edits, tests, doc-contract checks, local
  workflow scripts, and direct local HippUnfold runs.
- Repository root in this context is `/old_Users/yojiao/hipp`.
- Jianglab data may be visible through the local mount prefix:
  `/home/yojiao/LSS/jianglab`.
- For local core workflow Python tasks, use:
  `/home/yojiao/.conda/envs/py314/bin/python`
- For direct local HippUnfold CLI runs, use the `hippo2` gate above; do not use
  py314 for HippUnfold CLI execution.

### Argon SGE / MATLAB Launcher Context

- Use this context for qsub jobs, MATLAB jobs, and scripts intended to execute
  on Argon compute nodes.
- Argon Jianglab data paths must use the `/Shared/jianglab` prefix.
- Do not put the local mount prefix `/home/yojiao/LSS/jianglab` in Argon qsub
  commands, qsub wrappers, production manifests, or production logs.
- The local path `/home/yojiao/LSS/jianglab/...` is only the local equivalent of
  `/Shared/jianglab/...`; it is not the Argon command path.
- Do not apply the local py314 rule to Argon SGE MATLAB launchers.
- Existing Argon Kong2022/ArealMSHBM launchers default to:
  `/Users/yojiao/miniconda3/envs/py313/bin/python`
- If an Argon launcher already avoids Python for a dataset branch, keep it that
  way instead of introducing a local Python dependency.
- For Iowa7T 36P fMRIPrep production paths on Argon, the project root is:
  `/Shared/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P`
- The Iowa7T 36P production output root on Argon is:
  `/Shared/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P/hipp_outputs`
- qsub logs for Iowa7T 36P production runs belong under:
  `hipp_outputs/runs/logs/<run_id>/`

## Planning Mode Folder Creation Guardrail

- In planning mode (`plan mode`), if any new folder/directory needs to be created, the agent MUST explicitly ask the user for confirmation and clarify:
  1. The folder name.
  2. The folder contents/purpose.
  3. The directory path (disk location) where the folder will be created.
- Do not create any directory in planning mode without this explicit user confirmation.

## No Silent Fallbacks — HARD RULE

## Exact Artifact Selection — HARD RULE

- Code MUST disable preference, priority, existence-order, glob-order, and first-available selectors for competing artifacts; resolve the exact manifest-declared filename.
- Never use if-exists choice, next(...), glob()[0], directory iteration order, or an unsuffixed file as a substitute; missing, duplicate, or mismatched declarations MUST fail closed with path and variant.
- Compatibility is allowed only in an explicit caller-selected mode with validation and provenance; existence and lexical order are never selection policies.

- Rigorous analysis, workflow, and visualization code MUST NOT silently fall back
  from a canonical artifact to another file, an older cache, a different label
  variant, a recomputed approximation, or a different taxonomy merely because
  the canonical input is missing.
- Missing, stale, ambiguous, malformed, or mismatched artifacts MUST fail closed
  with an actionable error that names the missing contract and the exact path or
  metadata mismatch. Do not catch the error and continue with substitute data.
- If a compatibility or migration path is genuinely required, it MUST be an
  explicit, named command or configuration selected by the caller, separately
  validated by tests, and provenance-recorded in its outputs. An implicit
  `if missing: use other file` branch is prohibited.
- When multiple artifact variants are supported, selection MUST be explicit and
  contract-validated; filesystem existence order MUST NEVER choose the result.
- Every new fallback-like branch proposed for existing code requires a documented
  justification and explicit user approval before implementation.

## Deletion Safeguard

- Agents MUST NOT perform any file or directory deletion operations (e.g. `rm`, `rmdir`, `shutil.rmtree`, etc. in commands or scripts) without explicit confirmation from the user in the current session.

- Before performing any deletion, the agent must ask the user for confirmation and explain the target path and the reason.

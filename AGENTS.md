# Repository Rules

## Workspace and execution context

- Determine the checkout root from the current working directory or
  `git rev-parse --show-toplevel`. The local checkout is
  `/home/yojiao/hpchome/hipp`; Argon uses `/old_Users/yojiao/hipp`.
- `<repository>/tmp/` is the only temporary workspace for repository tasks.
- Local `/home/yojiao/LSS/jianglab` and Argon `/Shared/jianglab` access the same
  Jianglab storage. Use the prefix for the execution host; never put the local
  prefix in Argon qsub commands, production manifests or production logs.
- For Argon sessions, start from `/old_Users/yojiao/hipp` and use
  `codex-argon`, `codex-argon resume`, or `codex-argon agents`. This personal
  entrypoint connects to `argon-login-6.hpc` and preserves directory/arguments.
  Its executable is `/Users/yojiao/.local/bin/codex-argon`; its runbook is
  `/Users/yojiao/.local/share/codex-argon.md`. Existing sessions remain on the
  node where they started.

## Concurrency

Identify `idas` or `argon` from `SYSTEM_NAME`, `CLUSTER_NAME`,
`SGE_CLUSTER_NAME` or hostname before parallel work or an SGE array. If detection
is ambiguous, stop until `SYSTEM_NAME` explicitly identifies the system. Count
all intended jobs/tasks/processes, including child processes and other agents.

| Scope | IDAS | Argon |
|---|---|---|
| General repository cap | 16 total | 20 total |
| Kong2022 ArealMSHBM launcher | 10 total: UI `-tc 3`, PINC `-tc 7` | 20 total: UI `-tc 5`, PINC `-tc 15` |

Keep combined active batches within the applicable cap. The approved HCP
HippUnfold profile is a process-count exception for that workload only; use its
CPU affinity, GPU slots and launch command from the
[runtime manifest](docs/environment/hippo2_reproducibility_manifest.md#batch-resources-and-retained-outputs).
Other workflows retain their own limits.

## Filesystem authorization

- Do not access or browse archive contents without explicit user permission.
- Before any file/directory deletion, present the exact target paths and reason,
  verify that they resolve inside the current working directory, and wait for
  explicit confirmation in the current session. Paths outside that tree require
  explicit approval for those paths as well.
- Production result-tree migration, rebuild, replacement or cleanup requires an
  exact user-approved path list. Until approval, keep those trees read-only.
- Output archives belong under
  `wrong/outputs_<dataset>/<original-relative-path>`, outside active output
  trees. Preserve names and internal hierarchy.
- In planning mode, obtain explicit confirmation before creating a directory;
  state its name, purpose/contents and disk location.

## Documentation and scientific branches

- Write project documentation in English using current complete datasets and
  authoritative analysis rosters. Keep active instructions about current
  inputs, methods, outputs and usage; remove outdated text in place rather than
  appending corrections or retaining batch/upload histories.
- Formal documentation must not depend on pre-existing one-off scripts,
  reports, previews or logs in `tmp/`. Maintain tools needed for ongoing use in
  their appropriate project location; task-specific notes may record temporary
  artifacts. Generic temporary-workspace instructions and self-contained
  commands that create their own temporary files are appropriate.
- Behavior, path, cache, launcher, environment or output changes require matching
  documentation in the same change. Runtime-stack changes also require an
  environment-manifest update.
- Preserve real scientific branches, including 36P, method and atlas variants.
  Dates and run IDs belong in logs/manifests, not formal result-root names.
- HCP cortex-control routes are exactly `pfm_ctxreg` and `pfm_ctxdrop` below
  `outputs_hcp/inputs/kong2022/shared_inputs/`, `work/hippomaps_gradients/`,
  `work/group_atlas/` and `present/overviews/`. Keep method, atlas, smoothing,
  coverage and hemisphere structure within the route. Assignment variants such
  as `fisherz` and `normalized_top20_count` remain explicit files/metadata.
  Repairs replace approved formal products in place; do not create dated,
  recovered/regenerated roots or compatibility symlinks.
- Iowa JJ_TMS unversioned subgroup paths are 36P (`pfm/pure_mdd`,
  `pfm/comorbid_mdd`); retain those names for FC, gradients, group atlases,
  weighted and presentation products.
- Establish density from consumed surface/CIFTI paths and recorded
  `density_evidence`, not arguments alone. Report disagreements and treat
  consumed assets as authoritative.

## Functional network names

Use the following names and abbreviations for project-authored text, figures and
display fields. Define the full name on first use in prose. The machine-readable
keys remain as shown; display abbreviations do not replace input or output keys.

| Key | Full name | Display abbreviation |
|---|---|---|
| `Default` | Default Mode Network | DMN |
| `Visual` | Visual Network | VN |
| `Somatomotor` | Somatomotor Network | SMN |
| `DorsalAttention` | Dorsal Attention Network | DAN |
| `VentralAttention` | Ventral Attention Network | VAN |
| `Control` | Control Network | CON |
| `Language` | Language Network | LAN |
| `Auditory` | Auditory Network | AUD |
| `Limbic` | Limbic Network | LIM |
| `CinguloOpercular` | Cingulo-Opercular Network | CO |

CON means Control in this project; use CO for Cingulo-Opercular. For native
17- and 20-network views, use the following source-key-to-display mappings.
Keep the atlas and taxonomy identified in figure titles or captions. Preserve
source keys in data, metadata and artifact selection; these aliases change only
presentation. A cross-atlas merge is an analysis mapping, not a claim that its
source networks are identical; use `config/cross_atlas_network_merge.json` for
the defined canonical7 mapping.

Kong2022 native 17 networks:

| Source key | Display abbreviation |
|---|---|
| `DefaultA`, `DefaultB`, `DefaultC` | DMN-A, DMN-B, DMN-C |
| `ContA`, `ContB`, `ContC` | CON-A, CON-B, CON-C |
| `SalVenAttnA`, `SalVenAttnB` | VAN-A, VAN-B |
| `DorsAttnA`, `DorsAttnB` | DAN-A, DAN-B |
| `SomMotA`, `SomMotB` | SMN-A, SMN-B |
| `VisualA`, `VisualB`, `VisualC` | VN-A, VN-B, VN-C |
| `Aud`, `Language` | AUD, LAN |

The original `SalVenAttn` key denotes salience/ventral attention; VAN-A/B are
project display aliases and do not erase that source terminology.

Lynch2024 native 20 networks (Noise is a separate non-network label):

| Source key | Display abbreviation |
|---|---|
| `Default_Parietal` | DMN-Par |
| `Default_Anterolateral` | DMN-AntLat |
| `Default_Dorsolateral` | DMN-DorsLat |
| `Default_Retrosplenial` | DMN-Retro |
| `Visual_Lateral` | VN-Lat |
| `Visual_Dorsal/VentralStream` | VN-DV |
| `Visual_V5`, `Visual_V1` | VN-V5, VN-V1 |
| `Frontoparietal` | FPN |
| `DorsalAttention` | DAN |
| `Premotor/DorsalAttentionII` | PM/DAN-II |
| `Language` | LAN |
| `Salience` | SAL |
| `CinguloOpercular/Action-mode` | CO/AM |
| `MedialParietal` | MP |
| `Somatomotor_Hand`, `Somatomotor_Face`, `Somatomotor_Foot` | SMN-Hand, SMN-Face, SMN-Foot |
| `Auditory` | AUD |
| `SomatoCognitiveAction` | SCAN |

Preserve atlas-native names and published terminology when referring to the
source; do not rename existing artifacts or change selection contracts solely
to conform to display terminology. FPN and CO/AM remain distinct native Lynch
networks even though both map to canonical7 Control; SAL and VAN are likewise
not interchangeable native labels.

## Runtime

- Use `/home/yojiao/.conda/envs/py314/bin/python` for local workflow scripts,
  tests and documentation checks. Before a Python task, run
  `python -c "import sys; print(sys.executable)"` for informational context only;
  execute repository tasks with the explicit interpreter. Rewrite bare-Python
  documentation examples at execution time. If the wrong interpreter was used,
  stop and rerun correctly; do not mix environments.
- Argon SGE/MATLAB launchers use their documented cluster runtime, normally
  `/Users/yojiao/miniconda3/envs/py313/bin/python`. Preserve branches that do not
  need Python rather than adding a local interpreter dependency.
- Direct local HippUnfold runs require this gate in order:
  1. `source /opt/conda/bin/activate hippo2`
  2. `which hippunfold` resolves to
     `/home/yojiao/hpchome/hipp/.cache/conda_envs/hippo2/bin/hippunfold`
  3. `hippunfold --version` matches the release pinned in `environment/hippo2_server.yml`
  4. `hippunfold --modality T1w --output_density 512 --help` succeeds and shows `512`
- Validate the effective CLI and scientific source; environment names,
  `conda list` and `bioconda::hippunfold` alone are insufficient. Use official
  release source pinned in `environment/hippo2_server.yml` plus the documented
  local resource/runtime adapters. Use `multihist7sgi` for the current release.
  Do not apply local boundary, reference-plane or Workbench scientific fixes,
  and do not substitute development `main`. Do not use
  workflow labels `1.5.2`, `1.5.2-pre.2` or `1.5.2rc2`.
- Stage 01 gates the runtime/model/CUDA once per pending batch, with 300-second
  CLI probes, and checks the selected release's formal products with `--no-cleanup` while
  `--notemp` retains anatomical, coordinate, warp and CIFTI intermediates.
  Old cleanup receipts and local-candidate results do not establish completion.
  HCP defaults to eight subjects and accepts up to
  twelve under its approved profile; other datasets use two (BANDA one).
- Use the [agent runtime contract](docs/environment/agent_runtime_contract.md)
  for preflight and [hippo2 manifest](docs/environment/hippo2_reproducibility_manifest.md)
  for rebuilding, resource patches and launch parameters. Complete cached
  resources must issue no HTTP requests; incomplete caches report exact missing
  files. Reapply the documented patches after package reinstallations.
- Iowa7T 36P Argon production root is
  `/Shared/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P`;
  outputs use `hipp_outputs/` and qsub logs `hipp_outputs/runs/logs/<run_id>/`.

## Exact artifact selection

Resolve the exact manifest-declared filename and variant. Missing, stale,
ambiguous, malformed or mismatched artifacts must stop with an actionable
error naming the contract and path/metadata mismatch. Do not substitute another
file, older cache, label variant, recomputed approximation or taxonomy.

Preference, priority, first-available, filesystem-existence and glob/directory
order are not selection policies. Do not use `next(...)`, `glob()[0]`, an
unsuffixed alias or `if missing: use other file` to select competing artifacts.
Any compatibility mode must be explicitly caller-selected, separately tested,
validated and provenance-recorded. New fallback-like branches require a
documented justification and explicit user approval before implementation.

# Repository Rules

## Archive Access

- You MUST obtain explicit user permission before accessing or browsing any archive content.

## Flow Doc Maintenance

- The hippocampal functional parcellation flow documents must be kept in sync with the implementation.
- If any `gradient`, `prob-cluster`, or `prob-soft` workflow changes, update these Markdown files in the same change:
  - [multi_branch_flow.md](docs/hipp_parcellation_network/multi_branch_flow.md)
  - [step_by_step_flows.md](docs/hipp_parcellation_network/step_by_step_flows.md)
- This includes updates to:
  - post-`vertex-to-parcel FC` feature definitions
  - probability transforms
  - regularization rules
  - clustering rules
  - `K` selection logic
  - output interpretation
  - overview figure semantics
- A code change that alters those workflows is not complete until the corresponding flow Markdown files are updated.

## Doc Sync Rule

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

## HippUnfold Density Audit Guardrail

- For any question about hippocampal "density/resolution" (for example `2mm`, `0p5mm`, `512`, `8k`, "how many vertices"), agents MUST separate:
  - command/config argument level (`--output-density`, config defaults)
  - actually consumed asset level (the concrete surface files used by downstream scripts/scenes)
- Agents MUST NOT infer density from parameter names alone. They must verify the real consumed files first.
- Required evidence before answering:
  - concrete file paths used by the target pipeline stage (for example the exact `*.surf.gii` path pattern in code)
  - measured vertex counts from those files (read GIFTI and report `POINTSET` vertex count)
- If command argument and consumed files disagree, agents must explicitly report the mismatch and treat consumed-file evidence as authoritative for "what was actually used".
- For legacy HippoMaps outputs in practice, do not assume `space-corobl_label-hipp_midthickness.surf.gii` implies `den-2mm`; verify by vertex count.

## HippUnfold Environment Gate

- For direct `hippunfold` runs in this repository, the canonical local runtime environment is `hippo2`.
- Do not trust `conda list` or the environment name alone as proof of HippUnfold compatibility.
- Required validation sequence before a direct HippUnfold run:
  1. `source /opt/miniconda3/bin/activate hippo2`
  2. `which hippunfold` must resolve to `/opt/miniconda3/envs/hippo2/bin/hippunfold`
  3. `hippunfold --version` must return `2.0.0`
  4. `hippunfold --modality T2w --output_density 512 --help` must succeed and show `512`
- Do not use `importlib.metadata.version("hippunfold")` as authoritative version proof. In this workspace it can still report `1.5.2rc2` even when the CLI is the validated `2.0.0` build.
- If the gate fails, repair in place with:

```bash
CONDA_SAFETY_CHECKS=disabled conda install -y -n hippo2 -c khanlab -c conda-forge -c bioconda khanlab::hippunfold=2.0.0=py_0 --force-reinstall
```

- Do not treat `bioconda::hippunfold` as sufficient evidence of native `512` support.
- Avoid forbidden workflow version labels for this project: `1.5.2`, `1.5.2-pre.2`, `1.5.2rc2`.

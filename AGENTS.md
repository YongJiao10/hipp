# Repository Rules

## Flow Doc Maintenance

- The hippocampal functional parcellation flow documents must be kept in sync with the implementation.
- If any `gradient`, `prob-cluster`, or `prob-soft` workflow changes, update these Markdown files in the same change:
  - [multi_branch_flow.md](/Users/jy/Documents/HippoMaps/docs/experiments/hipp_functional_parcellation/multi_branch_flow.md)
  - [step_by_step_flows.md](/Users/jy/Documents/HippoMaps/docs/experiments/hipp_functional_parcellation/step_by_step_flows.md)
- This includes updates to:
  - post-`vertex-to-parcel FC` feature definitions
  - probability transforms
  - regularization rules
  - clustering rules
  - `K` selection logic
  - output interpretation
  - overview figure semantics
- A code change that alters those workflows is not complete until the corresponding flow Markdown files are updated.

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

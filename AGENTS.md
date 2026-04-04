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

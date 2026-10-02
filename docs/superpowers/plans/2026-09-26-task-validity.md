# Task signal validity implementation plan

Goal: remove the rest intensity ceiling from explicit task branches while retaining per-run signal validity.

Scope: task hippocampal preparation, cortex reference extraction, and their existing cache/consumer checks. Rest calculations remain as implemented. No production result rebuild in this change.

- Add a shared finite/nonconstant-per-run mask, rejecting malformed run boundaries.
- Use it when `--task` names a supported task in hippocampus and cortex preparation. Task summaries identify this as basic signal validity, with no tSNR threshold.
- Recompute task QC and smoothing rather than skipping old products. Task GLM consumers reject old smoothing/reference metadata.
- Test large task responses, between-run baseline differences, missing/constant signals, run-length mismatches, old-cache rejection and unchanged rest threshold behavior.
- Document the rebuild sequence and exact existing output roots before asking for the repository-required output replacement approval.

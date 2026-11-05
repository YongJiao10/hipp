# Task signal validity implementation plan

Goal: remove the rest intensity ceiling from explicit task branches while retaining per-run signal validity.

Scope: task hippocampal preparation, cortex reference extraction, and their existing cache/consumer checks.

- Add a shared finite/nonconstant-per-run mask, rejecting malformed run boundaries.
- Use runwise finite/nonconstant checks for hippocampal signals. Cortex ROI input excludes nonfinite signals.
- Recompute task QC and smoothing rather than skipping old products. Task GLM consumers reject old smoothing/reference metadata.
- Test large task responses, between-run baseline differences, missing/constant signals, run-length mismatches, and old-cache rejection.
- Document the rebuild sequence and exact existing output roots before asking for the repository-required output replacement approval.

# HippUnfold paired version comparison plan

> **For agentic workers:** Execute this authorized experiment inline, retaining experiment code, logs and results under `tmp/hippunfold_version_comparison/`. The user's request to compare authorizes execution; no additional plan approval is required.

**Goal:** Measure whether HippUnfold v2.2.0 materially changes existing HCP v2.0.0 den-512 surfaces, thickness, and downstream functional results. The user ended further reconstruction after three completed comparisons provided sufficient evidence; the final pilot includes 102311, 156334 and 878776.

**Architecture:** Keep formal outputs and the installed `hippo2` runtime unchanged. Obtain the exact v2.2.0 source in the experiment directory, reuse explicitly declared complete template/model resources, and run subjects serially with identical anatomical inputs and processing choices. Report a source-version comparison in the existing dependency environment separately from any fully isolated dependency-stack comparison.

**Tech stack:** Existing `hippo2` for HippUnfold execution; explicit py314 interpreter for selection and numerical analysis; existing core-workflow registration, sampling and gradient functions.

**Spec:** `docs/environment/hippunfold_2_2_0_result_impact.md` and the user's request to perform the proposed comparison.

## Constraints

- IDAS hostname identified; at most ten concurrent processes/tasks, with one subject workflow and one rule at a time. Rules may use up to four CPU workers; other native math threads remain constrained to one.
- All experiment files and temporary caches live below repository `tmp/hippunfold_version_comparison/`.
- No archive browsing, deletions, changes to canonical outputs, or replacement of the installed runtime.
- Select exact input/variant filenames; do not substitute competing artifacts.
- Preserve T1w input, Task101 model, density 512, preprocessing, unfolded registration and downstream BOLD/registration choices.
- Use existing direct CLI gate for the baseline; identify the comparison source and effective CLI version explicitly.

## Tasks

- [x] Select five completed HCP subjects spanning existing QC quality, including the previously inspected representative subject and a low-coverage subject; record exact paths and reasons.
- [x] Prepare exact v2.2.0 source and check CLI/dependency compatibility without changing `hippo2`; dry-run one subject to expose missing resources before inference.
- [x] Run three subjects into the experiment directory and analyze their complete outputs. After the user judged the differences sufficient, stop the batch and the fourth subject's experiment process tree; retain its partial files and leave the fifth unstarted. A graceful-stop request did not promptly leave the grouped execution, so the remaining experiment processes were terminated explicitly.
- [x] Compare thickness, inner/outer surfaces, final T1w den-512 midthickness geometry and subfield labels. Use exact corresponding vertex indices and report millimetre displacement and correlation/error summaries.
- [x] Reuse existing registration/sampling and functional-analysis functions with original BOLD, cortical inputs and settings; compare FC and sign/alignment-corrected gradients, and available equivalent parcel assignments for the three completed subjects.
- [x] Write an English experiment report, figures and tables, and update the version-impact note with measured results and limitations. Report any runtime failure as a failure, without substituting another version or analysis variant. Final products: `tmp/hippunfold_version_comparison/RESULTS.md`, `geometry.tsv`, `functional.tsv`, `analysis/group/functional.tsv`, and `comparison_summary.{png,pdf}`.

## Verification focus

The actual consumed GIFTI vertex count and declared source paths establish density. A successful workflow log alone does not establish complete outputs. Paired comparisons must use the same subject, hemisphere, sampling space, BOLD runs and connectivity variant. Dependency compatibility failures must be resolved explicitly or reported, rather than hidden by switching scientific inputs. Existing source tests are used only where changed reusable behavior requires them; no new validation framework is part of this experiment.

# HippUnfold 2.2.0 upgrade and HCP7T structural rebuild

## Accepted scope

Use official v2.2.0 scientific source in the existing hippo2 environment, retaining the repository's local resource, GPU and batch optimizations. The user subsequently authorized refreshing PyTorch and the aging scientific/runtime dependencies. Select supported stable versions and a CUDA 12 build compatible with the installed driver; validate the refreshed stack before production. Do not migrate to Pixi. Work in the current checkout and do not commit, including through the pi reviewer. Preserve unrelated working changes.

Archive the five local datasets' v2.0.0 HippUnfold outputs and hippocampus-dependent products, preserving relative paths under `wrong/outputs_<dataset>/`. Retain independent cortical labels, cortical time series, references and cortical analyses, including `reference/` children of mixed shared-input directories. Keep original inputs and reusable model/template resources. The user selected moving the two existing conflicting crossfit archives into `wrong/outputs_hcp/previous_archive/`, retaining their work/presentation hierarchy. Obtain confirmation of the expanded exact move list before moving.

After upgrade, testing and agreement with pi's default `google-vertex/gemini-4-argon` reviewer, rebuild the authoritative 166-person HCP7T structural source roster only. Finish with structural QC; functional downstream processing is outside this execution. Keep full structural outputs using `--no-cleanup`. Initial production concurrency is two subjects with one GPU inference slot, one core per subject and one preprocessing/save worker each; total active process/task concurrency must remain at most ten on IDAS. Report long-running production progress every twenty minutes.

The user confirmed continuing the complete structural batch after reviewing the execution evidence. Retain the official v2.2.0 scientific algorithms during this batch. Check native-to-atlas correspondence conflicts consistently across the completed cohort, determine the affected scope, and validate any scientific repair separately before functional downstream use. Preserve all intermediate outputs so an effective repair can rerun the affected stages without repeating unchanged segmentation.

## Implementation

- [x] Prepare and obtain confirmation of the exact archive moves; perform them without overwriting existing archives.
- [x] Read the official scientific-source changes, build the clean tagged package, and port the local resource/runtime patches.
- [x] Upgrade the launch contract, non-deleting output checks and bounded batch dispatch; synchronize environment and usage documentation.
- [x] Run focused behavior tests, real cached-resource DAG checks, and a GPU structural run of subject 102311 with geometry/QC inspection.
- [x] Discuss actual changes and test evidence with pi/argon until all blocking findings are resolved.
- [ ] Launch and monitor the complete HCP7T structural cohort, reporting execution completion separately from structural quality.

## Scientific and runtime requirements

Use the official greedy inner/outer default: 50x50x50 iterations with smoothing sqrt(3.0)vox and 0.707vox. Preserve the selected adult T1w model and 512-density atlas assets. Include the official method-dependent warp paths, optional SyN implementation and PyVista point-ID compatibility repair. Correct the tagged source's stale CLI version display. Preserve once-per-batch environment validation, fully cached operation without HTTP, unpacked-model reuse, incomplete-cache errors and the Snakemake local-source refresh patch. CLI probes have a 300-second limit; a timeout is reported rather than automatically retried.

Tests must exercise batch validation reuse, correct version/parameter handling, full-DAG cached resources, retained full outputs, restart completion and failure dispatch. Inspect the known 102311 right-hemisphere correspondence issue as a quality finding, separately from the thickness-method correction. No new hashes, verification framework or compatibility fallback is part of this work.

## Execution notes

Execution started on IDAS. Existing unrelated changes are limited to the NIH rTMS README and launchers. The approved conversation and this plan replace skill defaults that would create a worktree, commits, coordination ledgers or cleanup operations.

All 329 user-approved moves are complete, including both prior crossfit relocations. Independent cortical references, their canonical7 variant and physical-run metadata remain in place. The clean official wheel reports 2.2.0; only the local resource/inference-control/display patch is applied. Direct comparison confirms that all 61 unpatched installed scientific/configuration files match official v2.2.0. The complete cached HCP DAG passes with HTTP blocked and emits the official new greedy parameters. Host-local abstract Unix socket GPU exclusion passes a real two-process contention/release test and works on the local NFS-backed project.

The refreshed stack includes PyTorch 2.14.1+cu126, ANTs 2.6.5, Greedy package 0.0.20250712 (August 26, 2026 build), Convert3D 1.4.2, ITK 5.4.7 and VTK 9.7.0. Native updates use explicitly selected Conda-forge/Khanlab artifacts; the old Conda HippUnfold package and its obsolete PyVista constraint are removed. SQLite/libsqlite 3.54.1, readline 8.3 and ncurses 6.6 are installed consistently. The final dependency check, SQLAlchemy query, exact-environment CUDA matrix/3D convolution, VTK geometry operation, canonical CLI/resource gate and 55 focused tests pass. The fresh environment declaration dry run succeeds. All 166 declared T1w inputs exist.

The full 102311 test completed in 44m35s: 418/418 rules, 84/84 required outputs, no pruning, and successful retained-output resume. Template Dice is L 0.76279/R 0.79944; both consumed T1w hippocampal meshes have 512 finite vertices. The complete retained payload is 1.18 GiB; 166 similar payloads estimate 196 GiB against 379 GiB available.

A full scan of this test's native-to-atlas mapping finds 14 right atlas targets with candidate physical positions more than 1 mm apart, reaching 13.46 mm; the nine earlier version-displacement outliers were a subset. No left atlas target meets this conflict criterion. Recomputing the official Laplace–Beltrami solve reproduces the stored coordinates to 3e-8 after official rank uniformization; conflicting mappings also exist before that uniformization. Restricting boundary labeling to topological boundary vertices in an isolated experiment does not remove the right conflict. These checks localize the issue to native unfolding before registration and do not establish a single-line repair. The existing output check records this concrete correspondence conflict, not an unspecified manual-review request. Structural reconstruction and QC can proceed; functional sampling of this hemisphere requires resolving its native mapping. No diagnostic reconstruction replaces a formal output.

The requested pi reviewer used `google-vertex/gemini-4-argon`, read 32 source/test/document files, found no blocking implementation issues and agreed to the structural-only launch. A follow-up discussion confirmed the difference between the 14 final-map conflict targets and the 12 raw pre-uniformization targets. The review and agreement are in `tmp/hippunfold_220_upgrade/pi_argon_review.log` and `pi_argon_agreement.log`.

The full-roster coordinator launched on IDAS at 2026-10-23 02:55 UTC, PID 1562766, with two concurrent subjects and one GPU inference slot. Its log is `outputs_hcp/inputs/hippunfold/dense_corobl_batch/_batch_logs/hippunfold_220_structural_20261023T025501Z.log`. At the first 20-minute inspection (03:15 UTC), 102311 was skipped as complete and 100610/102816 were both active; the coordinator log reported no failure. GPU utilization was zero at that instant, so this observation does not establish that inference had run yet. Cohort execution and structural QC remain in progress; functional downstream processing has not started. No commits were made.

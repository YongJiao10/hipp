# Kong2022 400-ROI Recovery Implementation Plan

> **For agentic workers:** Use executing-plans and subagent-driven-development for bounded implementation and review. Track tasks below.

**Goal:** Restore hippocampal vertex × 400 ROI FC → continuous gradients → spatial clusters → canonical7 assignments and rebuild affected products including six HCP control overviews.

**Architecture:** Keep the historical FC filenames with strict ordered ROI evidence. Taxonomy belongs downstream of gradients and clustering. Explicit fixed-cluster reuse validates provenance; never discover replacement caches by existence.

**Tech Stack:** Existing Python core workflow and visualization suite; explicit py314 interpreter.

**Spec:** User-supplied Kong2022 recovery plan in this session (2026-09-10).

## Global Constraints

- IDAS host verified as idas-research-yojiao; max 10 concurrent processes, numerical threads 1.
- Do not access archives. No HippUnfold rerun or raw-input/preprocessing changes.
- Batch-one 14 specified targets authorized by user plan; inspect and record exact targets/checksums before deleting. Batch two requires separate exact-list approval.
- Kong400 features, gradients, masks, clusters and K must be taxonomy invariant. Top20 means at most 20 ROIs normalized by network ROI counts.
- HCP retains 151 ICA-FIX subjects, den-512 asset evidence, coverage .80, G1–G3, K3, shifted cosine, 2mm/4mm L/R; no HCP22P branch.
- Patient 36P and22P each fixed26; no cross-preprocessing sources.
- Documentation changes accompany behavioral changes. Fail closed on malformed/missing provenance.

## Tasks and verification

- [x] Audit: inspect exact14 targets and active HCP/Stanford/Iowa lineage; save absolute/resolved paths, type, metadata, checksums and dependency evidence before any deletion.
- [x] FC and gradient repair: scripts/core_workflow/04_01_materialize_kong2022_roi_fc.py and 05_01_compute_hippomaps_gradients.py. First add failing tests for rejection of7 columns and shuffled400 registry; accept validated400 including legacy network_fc filename. Validate full/run/diagnostic inputs. Keep network summaries separate. Update workflow docs.
- [x] Fixed-cluster reuse: scripts/core_workflow/05_02_run_group_atlas.py. Add failing tests for Kong reuse and rejection of cohort/surface/mask/gradient/K mismatch; preserve original K diagnostic evidence and labels, recompute canonical7 profile assignments only.
- [x] Paths: shared cohort resolver, hardcoded plotting entries, manifests and docs use active gradient-spectral/kong2022 with explicit taxonomy; reject stale lineage.
- [x] Delete authorized batch one only after audit and revalidation; generate exact batch-two dependency list for approval. The eight stale confidence sidecars were deleted after the user confirmed the exact list; the corresponding PNG panels were retained and recorded in the batch-two deletion receipt.
- [x] Restore Patient26 36P/22P from explicitly verified400 sources; validate/retain other correct sources and rebuild any proven bad assignment.
- [x] Rebuild HCP ctxdrop and ctxreg400 FC, gradients, clustering, assignments, WTA and six overviews using existing run controls. The accepted outputs are isolated under `outputs_hcp/inputs/hcp/kong2022/shared_inputs/kong400_regeneration_20260911/`, `outputs_hcp/work/hippomaps_gradients/kong400_regeneration_20260911/`, `outputs_hcp/work/group_atlas/*_kong400_regenerated_20260911_full_k3_fisherz/`, and the two dated overview roots. The ctxdrop run repaired two zero-byte run-wise files before final atlas generation.
- [x] Recompute affected visualization/statistics via suite grids; record actual input/output hashes and source metadata. The comparison suite reports 10/10 for all four grids; the six control overview PNGs, both controls' HippoMaps gradient/EV/endpoint products, confidence PNGs, and local-neighbor overlays were refreshed from the accepted dated roots.
- [x] Run focused core, control-array, taxonomy-invariance, documentation, resolver, and plot-lineage checks; final self-audit and diff check completed. The focused core suite passes (72 passed, 1 warning), the visualization suite passes (83 passed, 1 deselected), all four comparison grids report 10/10 readiness, and active scans find no references to the user-retired maintenance utilities.

## Execution ledger

- Initial checkout clean on main. Branch fix/kong400-recovery-20260910 selected in current production-path workspace; worktree preference asked asynchronously, default announced after opportunity to respond.
- Sandbox bwrap initialization fails; reviewed escalation used for tool execution.
- Confirmed code defect: materializer selects seven network timeseries; gradient RouteA explicitly requires7. Existing fixed-cluster reuse limited to Lynch and fabricates some K diagnostic defaults; must strengthen for Kong.
- Audit and implementation completed. The authorized 14 batch-one targets were snapshot, revalidated inside the repository, and removed; the exact receipt is `docs/kong400_recovery_audit/batch1_deletion_receipt.json`.
- Patient 36P/22P, HCP ctxdrop/ctxreg, and the 32 Iowa Lynch WTA matrix manifests were rebuilt or explicitly reused. HCP control overviews and all 12 comparison-suite PNGs were regenerated from validated sources. The HCP controls' final independent audits report 151 subjects, 604 full/run FC summaries per control, 400-ROI feature columns, four gradient manifests per control, and three overview items per control.
- Batch-two audit resolved 16 stale sidecars by rerendering active products. Eight legacy confidence PNGs were later individually confirmed by the user and deleted; the historical batch-two receipt still records their prior retention and is not rewritten.
- 2026-09-11: Formal canonical7 repair and acceptance executed per `KONG400_CANONICAL7_REPAIR_ACCEPTANCE_PLAN.md`. 84 group-atlas self-references repaired with explicit `historical_source` blocks; canonical8 production entries retired with negative tests; 172 contract tests passing. Final acceptance status recorded in `KONG400_CANONICAL7_ACCEPTANCE.md`; the HCP ctxreg/ctxdrop control regeneration is now independently accepted in `docs/kong400_recovery_audit/2026-09-11-review-followup.md`.
- 2026-09-11: The default HCP group-matrix and route-tag output resolvers were corrected to the dated regenerated control roots; the old recovered names remain compatibility aliases only and no longer select historical candidate outputs.

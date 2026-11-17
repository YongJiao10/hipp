# Explicit Kong fixed-cluster reuse

`05_02_run_group_atlas.py --reuse-fixed-cluster-root PATH` explicitly reuses a spatial clustering for another network taxonomy. `PATH` must identify the exact fixed run, its selected gradient-method `kong2022` atlas directory, or a group-atlas profile root. The resolver constructs one deterministic `smoothing/cov_<threshold>/g1g2g3/hemi_<hemi>` path and never searches sibling directories. Family17 routes accept a canonical7 source with the same gradient method, including TopSim/Ward and TopFC/gradient-spectral; their network profiles and labels are computed in the 17-network taxonomy.

For Kong, reuse validates the source manifest, ordered subjects, smoothing, hemisphere, surface evidence, support mask, K range, and the referenced gradient manifest's 400-ROI `group_fc_shape` against the actual declared FC array. Older gradient producers do not record the redundant `feature_count`; when present it must also agree. This does not select substitute inputs. The source `k_selection_summary.json` must contain the selected K row; every diagnostic value is preserved. Missing or mismatched evidence fails closed. Gradients, spatial labels, support masks, and K are reused; cluster profiles, Fisher-z and normalized_top20_count assignments in the target taxonomy (canonical7 or family17), and surface labels are recomputed downstream from the current ordered 400-ROI references.

The shared `group_atlas_lineage.py` gate is called by fixed-cluster reuse,
weighted production, overview readiness/rendering, comparison readiness, and
cortex gradient-profile loading. It requires complete local scientific artifacts,
checks selected gradients against the declared upstream, validates both parent
declaration fields, and rejects cycles and conflicting or unknown provenance.
Archive paths (including symlinks) are rejected before reads. Historical reuse
needs a source identity, a historical-only role, and explicit non-recomputation
flags; it never opens the historical reference. Patient 36P's four in-place
reuse manifests use `canonical7_in_place_fixed_cluster_reuse`.

Subject-contrast cache fingerprints include the lineage validator and reject
archive sources before hashing. After a validator/consumer code change or
source-manifest repair, explicitly rebuild subject contrasts with the existing
recorded source mode; do not replace hashes to bless an old cache. The closeout
uses `prepare_contrasts(..., rebuild_profiles=True,
source_mode='original-denoised')`, matching the prior contrast receipt. It
checks numerical arrays against the pre-rebuild values; this does not rerun
upstream FC, gradients, clustering, or HippUnfold. Existing recovery-mode WTA
means are republished by that preparation command with its standard backups.

Presentation manifests follow the same boundary: active `kong2022_7net`
overview products record `atlas: kong2022`, `network_taxonomy: canonical7`,
and `presentation_namespace: kong2022_7net`, while their source paths end in
`gradient-spectral/kong2022` or `wta-top/kong2022`. Historical four-network
Patient pseudo-cluster sidecars remain excluded from the active comparison suite.
The accepted HCP
ctxreg control is independently regenerated from the dated 400-ROI roots
recorded in
`docs/kong400_recovery_audit/2026-09-11-review-followup.md`; its group
atlases prove a `(3,400)` profile and a 400-ROI gradient input. The ctxdrop
control named in those dated audits is retired; its target-side exclusion is
reported by `06_07_render_local_cortex_control_diagnostics.py`. The older
recovered trees were removed on 2026-09-12 and are not selected by active
manifests. The visualization audit in
`docs/kong400_recovery_audit/visualization_7net_audit.json` records comparison
PNGs and their source/statistics contracts. The dated closeout audit's `controls` section
checks the six HCP ctxdrop/ctxreg overview panels and sources that existed at
that time. Earlier audit files are annotated
as historical/superseded rather than having their old facts silently rewritten.

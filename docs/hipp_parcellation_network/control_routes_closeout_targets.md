# HCP Cortex-Control Presentation Targets

The formal cortex-control routes are `pfm_ctxreg` and `pfm_ctxdrop`.
Their input, gradient, group-atlas, and presentation roots use those exact
names. Historical presentation records do not establish current readiness.

For each route, the maintained overview destination is
`outputs_hcp/present/overviews/<route>/`, containing:

- `kong2022_gradient-spectral_group_overview_g1g2g3_<route>_fisherz.png`
- `kong2022_gradient-spectral_group_overview_g1g2g3_<route>_normalized_top20_count.png`
- `kong2022_wta-top_group_overview_<route>_normalized_top20_count.png`
- `group_overview_manifest.json`
- `README.md`

Ctxdrop original-FC annotation must remain explicit in the gradient overview
semantics. A group-atlas stage consuming saved gradients records consumption;
it must not claim that the group-atlas stage computed those gradients or
rewrite genuine upstream computation history as reuse.

Render only from the exact manifest-declared formal group products after
validating their subject roster, taxonomy, QC, support, and lineage. Archived
products are not runtime substitutes for missing formal outputs. Numerical
recomputation, archive restoration, and persistent result changes require an
exact approved path list. Prior approvals for historical closeout work do not
authorize changes to other current result trees.

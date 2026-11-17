# HCP Cortex-Control Presentation Targets

The formal cortex-control route is `pfm_ctxreg`. Its input, gradient,
group-atlas, and presentation roots use that exact name. The former
`pfm_ctxdrop` route is retired; target-side local-cortex exclusion is reported
by the `06_07_render_local_cortex_control_diagnostics.py` FC/WTA diagnostic
under `outputs_hcp/present/local_cortex_control/`, which is not a route and
has no overview products. Historical presentation records do not establish
current readiness.

The maintained overview destination is
`outputs_hcp/present/overviews/<route>/`, containing:

- `kong2022_gradient-spectral_group_overview_g1g2g3_<route>_fisherz.png`
- `kong2022_gradient-spectral_group_overview_g1g2g3_<route>_normalized_top20_count.png`
- `kong2022_wta-top_group_overview_<route>_normalized_top20_count.png`
- `group_overview_manifest.json`
- `README.md`

A group-atlas stage consuming saved gradients records consumption;
it must not claim that the group-atlas stage computed those gradients or
rewrite genuine upstream computation history as reuse.

Render only from the exact manifest-declared formal group products after
validating their subject roster, taxonomy, QC, support, and lineage. Archived
products are not runtime substitutes for missing formal outputs. Numerical
recomputation, archive restoration, and persistent result changes require an
exact approved path list. Prior approvals for historical closeout work do not
authorize changes to other current result trees.

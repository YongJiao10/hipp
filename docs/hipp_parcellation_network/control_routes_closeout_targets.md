# HCP Cortex-Control Presentation Targets

The formal cortex-control routes are `pfm_ctxreg` (20 mm local-cortex
regression) and `pfm_ctxreg_10mm` (10 mm). Each route's input, gradient,
group-atlas, and presentation roots use that exact name. The former
`pfm_ctxdrop` route is retired; target-side local-cortex exclusion is reported
by the `06_07_render_local_cortex_control_diagnostics.py` FC/WTA diagnostic
under `outputs_hcp/present/local_cortex_control/`, which is not a route and
has no overview products. Historical presentation records do not establish
current readiness.

The maintained overview destinations are
`outputs_hcp/present/overviews/<route>/topsim/gradient-ward/<sm>/` and
`outputs_hcp/present/overviews/<route>/wta-top/<sm>/` for `<sm>` in `2mm` and
`4mm`, rendered from the `topsim/gradient-ward` and `wta-top` group atlases:

- `topsim/gradient-ward/<sm>/`: `roi-fisherz_ovr_overview.png`,
  `network-ts-fisherz_ovr_overview.png`, `top20_overview.png`
- `wta-top/<sm>/`: `wta_overview.png`
- in each directory: `group_overview_manifest.json` and `README.md`

A group-atlas stage consuming saved gradients records consumption;
it must not claim that the group-atlas stage computed those gradients or
rewrite genuine upstream computation history as reuse.

Render only from the exact manifest-declared formal group products after
validating their subject roster, taxonomy, QC, support, and lineage. Archived
products are not runtime substitutes for missing formal outputs. Numerical
recomputation, archive restoration, and persistent result changes require an
exact approved path list. Prior approvals for historical closeout work do not
authorize changes to other current result trees.

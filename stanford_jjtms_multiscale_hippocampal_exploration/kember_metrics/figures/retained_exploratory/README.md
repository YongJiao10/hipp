# Retained exploratory area figure

The retained raw PNG and historical PDF copies have been removed. This directory
keeps the provenance record only; the historical figure used the triangle-area
field and is not a current Kember result.

## Provenance and status

- Original location: `fast_rest/figures/area_2mm_standalone/`.
- Original generator: `fast_rest/src/plot_area_2mm_summary.py`.
- Source checkpoint: Git commit `7dc801199aca34a8527cdf5b26af62bd91186dd1`.
- The figure is **retained exploratory context**, not a current Kember result.
- Its historical area field used `consumed_midthickness_triangle_area`.

The active Kember area endpoint now uses the Kember-compatible HippUnfold native
`surfareacorobl` per-vertex metric. Its current result-driven figure is
`../direct_contrasts/kember_effects_2mm.png`, and its
tables and manifest live under `../../results/direct_contrasts/`.

Do not combine the numerical values in this retained figure with the active
Kember tables. The exact file hashes and source/deletion history are recorded
in `figure_provenance.json`.

## Current replacement generator

The current runnable implementation is `../../src/plot_area_2mm_comprehensive_summary.py`.
It uses the native Kember area table and the seven configured rosters, including
`TIS PTSD` (n=35), and writes the proportion view by default to
`../direct_contrasts/area_2mm_comprehensive_summary_proportion.png`; use
`--analysis-mode covariate` for the total-area-adjusted view.

The historical PDF artifact was removed from this directory; the current
generator writes PNG only and does not create or overwrite PDF files.

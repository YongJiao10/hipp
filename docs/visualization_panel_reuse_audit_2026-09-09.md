# Multi-dataset panel reuse audit

Scope: the six maintained multi-dataset figure types in
`scripts/visualize/run_multidataset_suite.py`, comparison and legacy branches.
Archive content was not inspected. This audit does not recompute upstream FC
or gradient estimates and does not certify every scientific analysis upstream.

## Findings and changes

1. Gradient subregion Top-20: the previously diagnosed upper-row-only crop is
   fixed; the selected smoothing determines which overview row is consumed.
   The legacy four-grid refresh also includes this figure.
2. Cluster confidence, Workbench hippocampus overlay, and subfield composition:
   old thumbnail hits tested only PNG existence/size. Labels and surfaces could
   change while old images were reused with fresh statistics. These three
   loaders now validate canonical sources before cache lookup and require
   `.sources.json` containing SHA-256 fingerprints for input manifests, labels,
   surfaces, scene/spec, rendering code/style, grid selection, and output PNG.
   Unverified derived PNGs are regenerated from the same canonical sources.
3. Subfield comparison: Lynch declares only the Fisher-z assignment variant.
   The loader previously always requested normalized-top20, a mismatch hidden
   by early old-cache returns. It now selects the explicitly registered method
   variant: Kong normalized-top20, Lynch Fisher-z. No cross-method substitute
   is introduced.
4. The WTA long-axis and hippocampus/cortex projection renderers already have
   source/output fingerprint cache validation and smoothing/variant/method
   cache identity. Their code and existing regression coverage were inspected.
   Complete current comparison output certification is not claimed: WTA source
   manifests were being materialized by another active task during this audit.

## Evidence

- 176 gradient manifests across both branches matched their requested smoothing.
- Iowa/Stanford 22p and 36p resolved to distinct actual gradient directories;
  HCP intentionally shares its preprocessing source across these variants.
- Before/after RGBA pixel hashes were recorded for existing comparison panel
  caches. Workbench overlay: 18 of 40 matching cache paths changed after
  source-based regeneration. Subfield composition: 20 of 40 changed. These are
  image-content differences, not estimates of the number of mislabeled vertices.
- Confidence comparison panel names had changed from historical namespace
  names, so an exact-path before/after count is not reported for that figure.
- 68 focused tests passed, covering changed source files, corrupted outputs,
  missing manifests behind an existing PNG, method selection, smoothing crops,
  existing WTA/cortex cache contracts, and removal of standalone rendering.

## Refresh contract

All refreshes use the suite with `--grid all`; no standalone CLI is restored.
Native subfield rendering uses two cohort workers, leaving room for Workbench
child processes within the IDAS concurrency limit.

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/visualize/run_multidataset_suite.py --version hcp_jjtms_7net --grid all --figures cluster_confidence_stacked workbench_hippocampus_overlay subfield_composition_2col
/home/yojiao/.conda/envs/py314/bin/python scripts/visualize/run_multidataset_suite.py --grid all --atlas 7net --figures cluster_confidence_stacked workbench_hippocampus_overlay subfield_composition_2col gradient_subregion_top20_stacked_2col
```

## Final verification

- Comparison: three affected figure types × four grids = 12 master PNGs,
  all suite receipts SUCCESS.
- Legacy: those three types plus gradient Top-20 × four grids = 16 master
  PNGs, all suite receipts SUCCESS.
- All 28 final PNGs passed image-file verification.
- All 264 published panel provenance records (88 per cache-affected figure
  type) matched freshly recomputed source and output hashes; zero stale records.
- Subfield comparison was refreshed across all four grids again after the final
  worker-limit adjustment, so its provenance refers to the final renderer code.
- WTA preflight warnings concern separate upstream WTA manifests; the gradient
  sources required by the refreshed figures were present and consumed directly.

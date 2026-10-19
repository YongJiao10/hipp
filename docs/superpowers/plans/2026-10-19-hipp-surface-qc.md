# Hippocampal surface QC comparison implementation plan

> **For agentic workers:** Use executing-plans to run this small analysis inline.

**Goal:** Compare current surface tSNR25 selection, functional-GM surface
coverage and their participant-level intersection across the five current datasets.

**Architecture:** A standalone research script reads authoritative rosters and
declared inputs. Native folded midthickness meshes provide physical vertex areas;
existing anatomical transforms place the same vertices in the CIFTI volume grid.
Production results, masks and rosters remain read-only.

**Tech stack:** Explicit py314, nibabel, NumPy/SciPy, Workbench and existing ANTs.

**Spec:** `docs/hipp_surface_qc_options.md` and the user's five-dataset test request.

## Constraints

- IDAS: at most 10 concurrent processes/tasks; analysis uses two workers and
  single-thread numerical libraries. No archive access, deletions or new hashes.
- English project documentation; unknown inputs are distinct from failed QC.
- HCP3T EMOTION/WM and Stanford 22P/36P remain explicitly separated.
- PDC visits and people have separate denominators; any-passing-visit person
  counts describe data availability, not a longitudinal inclusion policy.
- The 20-mm2 test fails when either hemisphere has unsupported area >=20 mm2.
  This hemisphere convention and vertex support definition are project choices;
  Kember2026 does not report enough detail for an exact replication.
- Primary coverage asks whether the actual trilinear sampling neighborhood has
  any finite, nonconstant functional hippocampal GM support (positive total
  interpolation weight). A nearest-voxel GM test is recorded separately as a
  stricter diagnostic: a nearest voxel outside GM does not imply zero GM
  contribution from the other interpolation neighbors. Native triangle areas
  are divided equally among their three vertices. Record L/R and total area.
- Retain current HCP/Stanford tSNR25 participant decisions as a reference; do
  not reconstruct vertex masks from scalar summaries. The input audit found
  direct corobl-to-MNI sampling without a registration transform, so the main
  comparison additionally recomputes tSNR at registered anatomical vertices.
  BANDA/PDC need a new explicitly declared CIFTI-volume-to-surface signal
  projection, not their older volume-ROI counts. See the report for the
  measured-baseline definitions and HCP7T's retained nominal-10000 proxy.
- Joint selection requires both participant decisions. No extra cutoff.

## Tasks

- [x] Confirm one representative surface/transform/functional-grid combination
  per dataset. Stop an affected comparison if the declared coordinate mapping
  is missing; never substitute unregistered coordinates.
- [x] Add `docs/hipp_surface_qc.py`, `docs/hipp_surface_tsnr_registered.py` and matching documentation.
  Calculate coverage, BANDA/PDC surface tSNR, missing reasons and joint decisions.
- [x] Run all five parent rosters, retaining per-person/visit results and summary
  TSVs under `docs/`; retain research projection outputs under the same analysis.
- [x] Check physical units/topology, representative projection alignment and
  denominator arithmetic; discrepancies change the affected results, not just
  a validation label. Report pass/fail/unknown counts and interpret retention.

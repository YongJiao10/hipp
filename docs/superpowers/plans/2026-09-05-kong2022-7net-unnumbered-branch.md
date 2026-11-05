# Kong2022 canonical7 source-branch implementation plan

## Decision

Kong2022 7net is a taxonomy branch of the existing Kong2022 RouteA workflow,
not a new pipeline step. The implementation must not add a code folder, an
unnumbered runner, or a new numbered registration.

The branch contract is explicit: canonical7 and canonical8 are parallel
branches. Canonical7 references, 7-column FC, RouteA gradients, spatial
clustering, K diagnostics, K selection, and final group labels are computed
from canonical7 inputs and are not reused from canonical8. Aud is mapped into
Somatomotor in the fixed canonical7 order.

## Source layout

- Existing `03_02_build_kong2022_routea_reference.py` accepts
  `--atlas-slug kong2022_7net` and writes the direct `reference/kong2022_7net/`
  branch.
- Existing `04_01_materialize_kong2022_roi_fc.py --atlas kong2022_7net`
  produces a separate seven-column FC cache.
- Existing `05_01_compute_hippomaps_gradients.py --routes routea_7net`
  computes the independent RouteA gradient bank.
- Existing `05_02_run_group_atlas.py --atlases kong2022_7net` performs the
  independent group-atlas and K-selection run; its assignment-only reuse
  entry point is retired for canonical7.
- Existing `06_01_build_group_overview.py` renders the branch from the
  existing group-atlas artifacts.

A separate `routea_7net` output namespace protects canonical8 outputs from
overwrite, while active manifests record `atlas=kong2022_7net` and
`network_taxonomy=canonical7`.

## Cohorts and variants

The 36P release contains HCP, seven Stanford cohorts, and four JJ_TMS cohorts:
12 groups and four overview PNGs per group. The unversioned roots use 36P.

## Verification

- Focused canonical7 tests cover Aud mapping, direct FC/gradient/group-atlas
  routing, canonical7 order metadata, template provenance, and the absence of
  an unnumbered runner/folder.
- The numbering contract requires the existing numbered script list and no
  canonical7 utility registration.
- Python checks use
  `/home/yojiao/.conda/envs/py314/bin/python`.
- Existing generated outputs are preserved; no cleanup is part of this plan.
- The four obsolete numbered files listed in the implementation discussion
  remain pending explicit user confirmation before deletion.

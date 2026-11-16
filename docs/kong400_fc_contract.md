# Kong2022 400-ROI feature contract

Kong2022 gradient and WTA products consume the ordered 400-parcel profile from each subject's `reference/kong2022/cortex_roi_parcels.tsv` and `cortex_roi_parcel_timeseries.npy`. The historical `hipp_vertex_to_network_fc.npy` and `_active.npy` filenames remain for launcher compatibility, but their validated Kong contents are 400 ROI columns. The FC loader checks the array shape, the exact parcel ID/name order from the shipped Kong dlabel registry, the reference time-series row count, and any declared summary shape; a seven-column array or shuffled registry fails closed.

Canonical7 network time series and FC are summaries only. The materializer writes them to `hipp_vertex_to_canonical7_fc.npy` and `_active.npy`. They are available to network-spectral diagnostics and do not enter the Kong gradient bank. Gradient Fisher-z and normalized_top20_count annotations reuse the same spatial clusters; Fisher-z averages the 400 ROI profile by the selected taxonomy, while normalized_top20_count takes at most 20 ROI columns and divides counts by the total ROI count in each taxonomy network.

The `05_01` manifest records `feature_family: kong2022_400roi_fc`, `feature_count: 400`, `group_fc_shape`, and the ordered registry evidence. `05_02` rejects missing or stale gradient output paths and rejects Kong manifests without the 400-column evidence. Control routes `routea_ctxreg` and `routea_ctxdrop` use the same contract after their route-specific time-series construction.

## Cortex controls: preserve the standard annotation statistic

Production `05_02` annotation uses `profile_fc_mode=cluster_mean_timeseries`
for RouteA, ctxreg, and ctxdrop alike. For each subject, average supported,
valid hippocampal time series within each cluster, correlate that mean with
each of the 400 ROI reference time series, transform to Fisher-z, and then
average across subjects. Network profiles average the resulting ROI columns.
Do not switch to averaging vertex-wise Fisher-z FC merely because a route
is a cortex control. The production call, resume signature, and output
manifest must all record the same cluster-mean statistic.

Stage boundaries remain explicit: ctxreg removes the local cortical regressor
in `04_00`; its `--profile-surface-root` must point to those residual time
series. ctxdrop excludes local cortical vertices before ROI averaging in
`04_01`, affecting the FC features used by `05_01` and clustering.
Its subsequent cluster annotation uses the standard ROI reference and
hippocampal time series; this annotation is not a second, cluster-level
local-cortex exclusion analysis. Do not invent a union-of-spheres or another
cluster-level exclusion policy as part of annotation.

The erroneous control profiles produced by averaging vertex-wise Fisher-z FC
are not comparable to standard annotations and must be rebuilt before
presentation. Correcting this annotation does not require changing the
upstream gradient algorithm or surface affinity. The 2026-09-13 approved
deletion inventory is
`pending_ctxprofile_deletion_20260913.txt`: all 120 listed files were deleted.
The user subsequently approved removing the complete control result trees.
Both `routea_ctxreg/` and `routea_ctxdrop/` were deleted under each of these
six parents, relative to `outputs_hcp/` (12 directories including all contents):

- `work/hippomaps_gradients/`
- `work/group_atlas/`
- `present/overviews/`
- `present/gradients/`
- `present/cortex_panels/`
- `present/local_cortex_neighbors/`

Before the corrected rebuild, all 12 directories were verified absent. This
superseded the initial retention of gradients, cluster labels, K diagnostics,
and WTA results. The control input caches under `inputs/`, standard RouteA,
and archives were not deleted.
The corrected production rebuild was completed on 2026-09-13 for both
`routea_ctxreg` and `routea_ctxdrop`: `05_01` HippoMaps gradients, `05_02`
gradient-spectral and WTA-top group products, and the maintained overview,
gradient-product, and local-neighbor presentations. All four smoothing /
hemisphere combinations per method have `status: ok`, `n_subjects` equal to the roster length, and finite main
profiles. The control branches use the standard `cluster_mean_timeseries`
annotation statistic recorded in their manifests.

## Overview display semantics for cortex-control routes

The `06_01` overview renderer keeps the control stages separate from their
explanatory annotation. For `routea_ctxdrop`, gradient-spectral clustering and
vertex WTA retain the per-hippocampal-vertex local-cortex exclusion used by the
control route. The two gradient overview products do not introduce a new
cluster-level exclusion statistic: their cluster network panels are the
standard `cluster_mean_timeseries` annotation using the original surface
reference recorded by the four source manifests. Their title and manifest
therefore say `Original-FC annotation of cortex-controlled clusters`.

For the `normalized_top20_count` gradient product, the displayed counts are
counts derived from that original-FC ranking; they are not a second FC estimate
computed after taking a cluster-wide union of local cortical neighborhoods.
The WTA overview is not labeled as a cluster original-FC annotation. The
renderer validates the complete 2mm/4mm × L/R source-manifest grid, route tag,
profile mode, profile surface root, and reference-root consistency before
writing these labels; missing or mixed provenance fails closed. No FC values,
cluster labels, K selection, or control algorithm are changed by this display
metadata.

## 2026-09-13 closeout execution record

The single IDAS execution agent took over the pre-existing PID 1652039
(`04_00_materialize_cortex_regressed_surface_timeseries.py --jobs 8`) and did
not start a duplicate job. It exited after all 604 expected ctxreg units
(roster subjects × 2 smoothings × 2 hemispheres) had `status: done` stage
manifests, `status: ok` summaries, and controlled time-series outputs. No
ctxdrop/ctxreg `04_01`, `05_01`, or `05_02` numerical recomputation was started
in this closeout.

The only presentation computation was a sequential redraw of the six formal
overviews: two gradient assignment variants plus WTA for each of
`routea_ctxdrop` and `routea_ctxreg`. Both overview manifests now contain three
items and the README tables identify the ctxdrop gradient annotation. A
pre-render baseline of 64 group scientific files (labels, profiles, K/ARI
summaries, and assignment status files) was compared after rendering; all 64
SHA-256 hashes were unchanged. The 16 group manifests remain `status: ok`,
`n_subjects` equal to the roster length; all active consumed paths exist; gradient profiles are finite 3×7 with
400-ROI feature summaries, three components, and shifted-cosine affinity; WTA
status payloads are non-empty and valid.

The focused verification command passed with `54 passed, 3 warnings`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/yojiao/.conda/envs/py314/bin/python -m pytest -q \
tests/core_workflow/test_group_overview_affinity_provenance.py \
tests/core_workflow/test_kong400_subject_completeness.py \
tests/core_workflow/test_routea_ctxreg_cortex_regression.py
```

After separate explicit user approval, the producer-path provenance issue was
resolved by a metadata-only migration of the exact 2,416 JSON files in
[the approved inventory and verification receipt](ctxcontrol_metadata_migration_targets_20260913.md).
The 9,664 relocated fields comprise ctxdrop stage outputs (3,020), ctxdrop
neighbor diagnostics (604), ctxreg stage inputs (604), ctxreg stage outputs
(4,228), and ctxreg FC summary paths (1,208). All canonical counterparts exist.
Each file preserves original paths and the original metadata SHA-256 in an
explicit `path_migration` receipt; producer timestamps and computation
parameters are unchanged. Dated references now occur only in those historical
receipts, not the migrated active fields. This records path relocation, not
new FC computation or proof of historical/current input byte equality.
The migration performed no numerical writes, deletions, Git staging or commits.

Run core checks with:

Active group-atlas consumers also enforce the fail-closed lineage contract
in [kong400_fixed_cluster_reuse.md](kong400_fixed_cluster_reuse.md): actual
400-column upstream FC, selected-gradient equality, complete local artifacts,
and explicit non-cyclic provenance. Historical source references are descriptive
only and are never runtime inputs. Active regression coverage is maintained in
`tests/core_workflow/test_kong400_fixed_cluster_reuse.py` and
`tests/core_workflow/test_group_atlas_lineage.py`.

Focused FC tests:

```bash
/home/yojiao/.conda/envs/py314/bin/python -m pytest -q tests/core_workflow/test_kong2022_fc_contract.py tests/core_workflow/test_kong2022_7net.py tests/core_workflow/test_hippomaps_gradient_route.py
```

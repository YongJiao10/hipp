# HippoMaps Pipeline Summary

Within this repository area, the active hippocampal analysis surface is now **group analysis only**.

Active workflow:

- reuse shared per-subject group inputs from `outputs_migration/hipp_group_inputs_shared/`
- materialize canonical-network FC inputs for Kong atlases (`group-network-spectral`, `group-intrinsic-spectral` only)
- build merged-canonical hippocampal group atlases
- export the public four-figure Kong overview set into `present_group/`

Retired: partial-FC branches (`group-network-partial-spectral`, `group-intrinsic-partial-spectral`) are no longer run; code is retained but inactive.

Active references:

- `docs/hipp_parcellation_network/network_first_runbook.md`
- `docs/hipp_parcellation_network/merged_canonical_group_atlas_plan.md`

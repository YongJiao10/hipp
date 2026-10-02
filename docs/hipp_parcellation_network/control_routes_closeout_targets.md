# HCP control routes: presentation closeout and remaining provenance scope

Status: presentation closeout completed after user approval; subsequent producer-path metadata migration was explicitly approved and verified against [the exact inventory](../ctxcontrol_metadata_migration_targets_20260913.md). Paths below are relative to `/home/yojiao/hpchome/hipp/`. No Git commit is authorized or performed.

Closeout verification: 54 focused regression tests passed with 3 expected numerical warnings; all 16 formal control group manifests passed status/N151/active-path checks. All six formal overviews were redrawn and visually inspected. The two ctxdrop gradient items visibly and structurally identify `Original-FC annotation of cortex-controlled clusters`; normalized counts additionally identify their original-FC ranking source. Numerical group arrays and upstream producer manifests were not rewritten. Task-created rendering scratch was cleaned; no persistent alternate output tree was created. This was a bounded closeout, not a repeat of the prior full numerical audit.

Presentation scope: make ctxdrop original-FC annotation semantics explicit without changing upstream computation history; retain the existing heatmap/legend presentation fixes; redraw the six formal overviews in place and refresh their presentation metadata. No numerical arrays, WTA scientific outputs, directory names, archives, or other datasets are changed. The later separately approved metadata migration is recorded in the linked inventory; it preserves historical provenance and involves no deletion or recomputation. Related source, regression tests, and workflow documentation are updated together.

## Existing output files permitted to change only after approval

The approved formal presentation products are the following six PNGs and the
two route-level manifests/READMEs:

```text
outputs_hcp/present/overviews/routea_ctxdrop/kong2022_gradient-spectral_group_overview_g1g2g3_routea_ctxdrop_fisherz.png
outputs_hcp/present/overviews/routea_ctxdrop/kong2022_gradient-spectral_group_overview_g1g2g3_routea_ctxdrop_normalized_top20_count.png
outputs_hcp/present/overviews/routea_ctxdrop/kong2022_wta-top_group_overview_routea_ctxdrop_normalized_top20_count.png
outputs_hcp/present/overviews/routea_ctxdrop/group_overview_manifest.json
outputs_hcp/present/overviews/routea_ctxdrop/README.md
outputs_hcp/present/overviews/routea_ctxreg/kong2022_gradient-spectral_group_overview_g1g2g3_routea_ctxreg_fisherz.png
outputs_hcp/present/overviews/routea_ctxreg/kong2022_gradient-spectral_group_overview_g1g2g3_routea_ctxreg_normalized_top20_count.png
outputs_hcp/present/overviews/routea_ctxreg/kong2022_wta-top_group_overview_routea_ctxreg_normalized_top20_count.png
outputs_hcp/present/overviews/routea_ctxreg/group_overview_manifest.json
outputs_hcp/present/overviews/routea_ctxreg/README.md
```


Both routes are included because a group-atlas stage consuming upstream gradients must distinguish that consumption from computation performed by the group-atlas stage. Do not replace genuine upstream regeneration history with a reuse claim.

Verification is limited to focused regression tests, lineage validation, and visual inspection of the six redrawn overviews. Do not repeat the completed full subject-FC or gradient numerical audit. If rendering requires additional persistent output writes, stop and obtain approval for those exact paths rather than expanding this list implicitly.

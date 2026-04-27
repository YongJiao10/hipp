# Group Analysis Runbook

## Active Scope

This directory now supports **group analysis only**.

Active workflow:

- cortex reference extraction for `kong2022` (Route B: public 400-ROI dlabel)
- FC materialization for `kong2019` / `kong2019_8net` / `kong2022`
- merged-canonical group atlas construction
- present-group overview generation
- optional backfill of group functional atlas labels

For the T1only HippUnfold branch, do not use the default 153-subject T2only-era manifest or the default shared root. Use the dedicated runbook:

- `docs/hipp_parcellation_network/t1only_kong2022_route_b_runbook.md`

The canonical network axis is still atlas-specific, and the merge map is preserved in [merged_canonical_group_atlas_plan.md](/home/yojiao/hpchome/hipp/docs/hipp_parcellation_network/merged_canonical_group_atlas_plan.md). The important cases are:

- `Kong2019`: `TemporalParietal -> Default`, `Noise` is not retained
- `kong2019_8net`: same canonical merge logic, but `TemporalParietal` stays independent in the 8-network layout where required by the upstream mapping file
- `kong2022`: 400 ROI labels (`Schaefer2018_400Parcels_Kong2022_17Networks_order`) are merged to canonical-8 via
  `Default* -> Default`, `Cont* -> Control`, `SalVenAttn* -> VentralAttention`, `DorsAttn* -> DorsalAttention`,
  `SomMot* -> Somatomotor`, `Visual* -> Visual`, `Aud -> Auditory`, `Language -> Language`
- `Hermosillo2024`: `MTL + Tpole -> Limbic`
- `Lynch2024`: `Noise` is excluded, `Language` is retained

Retired workflows are archived and are no longer public entrypoints:

- subject-level hippocampal clustering (`scripts/common/spectral_clustering.py` has no live caller and is retained as reference only)
- batch subject orchestration
- `group prior + fast-PFM`
- MSHBM experiments
- `group-network-partial-spectral` and `group-intrinsic-partial-spectral` methods (code preserved, not running)

## Canonical Input Roots

```text
shared group input root   outputs_migration/hipp_group_inputs_shared/
active group output root  outputs_migration/hipp_group_merged_canonical_atlas/
active present root       present_group/
legacy code/docs/results  _archive/hipp_parcellation_network/
```

The active scripts fail fast if required inputs are missing. There is no fallback to archived subject outputs.

For Route B (`kong2022`), atlas-invariant cortex upstream artifacts are rooted at `sub-*/reference/common/`:

- `reference/common/tsnr_gate/` for cortex valid masks
- `reference/common/runs/` for run boundary metadata

Atlas-specific cortex reference arrays remain under `sub-*/reference/kong2022/`.

## Default Cohort

Use the default QC-passed 153-subject cohort:

- `manifests/hcp_7t_hippocampus_fmri_qc_pass_153.txt`

Do not silently expand to 166 subjects.

This default applies only to the legacy/default group input root documented below. T1only derives a new pass manifest after T1only HippUnfold Dice QC and T1only hippocampal tSNR QC.

## Active Entry Points

- `scripts/cortex/run_cortex_kong2022_reference_batch.py`
- `scripts/hipp_parcellation_network/materialize_kong7_kong8_pearson_partial_fc.py`
- `scripts/hipp_parcellation_network/materialize_kong2022_vertex_to_roi_fc.py`
- `scripts/hipp_parcellation_network/run_merged_canonical_group_atlas.py`
- `scripts/hipp_parcellation_network/build_present_group_overview_merged_canonical.py`
- `scripts/hipp_parcellation_network/materialize_group_functional_atlas_labels.py`

## Canonical Command Sequence

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/cortex/run_cortex_kong2022_reference_batch.py \
  --shared-root outputs_migration/hipp_group_inputs_shared \
  --jobs 6 \
  --resume

/home/yojiao/.conda/envs/py314/bin/python scripts/hipp_parcellation_network/materialize_kong7_kong8_pearson_partial_fc.py \
  --shared-root outputs_migration/hipp_group_inputs_shared \
  --jobs 6

/home/yojiao/.conda/envs/py314/bin/python scripts/hipp_parcellation_network/materialize_kong2022_vertex_to_roi_fc.py \
  --shared-root outputs_migration/hipp_group_inputs_shared \
  --jobs 6

/home/yojiao/.conda/envs/py314/bin/python scripts/hipp_parcellation_network/run_merged_canonical_group_atlas.py \
  --shared-root outputs_migration/hipp_group_inputs_shared \
  --resume \
  --coverage-thresholds 0.80 \
  --atlases kong2022 \
  --methods group-intrinsic-spectral group-network-top group-network-all \
  --out-root outputs_migration/hipp_group_merged_canonical_atlas

/home/yojiao/.conda/envs/py314/bin/python scripts/hipp_parcellation_network/build_present_group_overview_merged_canonical.py \
  --group-root outputs_migration/hipp_group_merged_canonical_atlas \
  --present-dir present_group \
  --atlases kong2022 \
  --methods group-intrinsic-spectral group-network-top group-network-all \
  --coverage-threshold 0.80

/home/yojiao/.conda/envs/py314/bin/python scripts/hipp_parcellation_network/materialize_group_functional_atlas_labels.py \
  --out-root outputs_migration/hipp_group_merged_canonical_atlas
```

## Active Outputs

```text
outputs_migration/hipp_group_merged_canonical_atlas/
  <method>/<atlas>/<smoothing>/<coverage>/hemi_<L|R>/...
  group_average_surfaces/

present_group/
  kong2022_group-intrinsic-spectral_group_overview.png
  kong2022_group-network-top_group_overview.png
  kong2022_group-network-all_group_overview.png
  group_overview_manifest.json
  README.md
```

## Interpretation Rules

- merged-canonical group outputs use cluster IDs ordered by hippocampal long-axis position (`head -> tail`), not by cluster size
- authoritative runtime products are under `outputs_migration/hipp_group_merged_canonical_atlas/`
- `present_group/` is a presentation/export surface, not the canonical runtime input source
- K-selection panels use direct subject-split `group_ari_mean`; null references are reported in ARI space
- spectral methods (`group-network-spectral`, `group-intrinsic-spectral`) can run `--k-selection-mode redundancy-aware`
- redundancy-aware `K` selection uses:
  - profile redundancy gate: `max_profile_similarity <= 0.90`, `redundant_pair_count == 0`
  - final K selection (three-tier over the survivor pool that passed all gates):
    1. 1-SE rule: keep survivors within 1 SE of the survivor-pool best mean ARI
    2. local-max preference: among within-1SE survivors, prefer ARI-curve local maxima
    3. parsimony tiebreak: smallest K when quality is genuinely tied
- overview `K` diagnostics show per-K reject reasons: `redundant`, `nontrivial_fail`
- if a combo cannot select a valid `K`, the driver writes `group_atlas_manifest.json` with `status=skipped_*` and continues remaining combos; `--resume` treats that manifest as completed
- `group-network-top` / `group-network-all` are clusterless, kong2022-only, and do not emit `group_labels_final.npy`
- the three active methods (`group-intrinsic-spectral`, `group-network-top`, `group-network-all`) do not consume cortex `8 x T` timeseries for network assignment

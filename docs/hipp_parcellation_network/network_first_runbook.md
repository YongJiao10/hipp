# Group Analysis Runbook

## Active Scope

This directory now supports **group analysis only**.

Active workflow:

- canonical-network materialization for `kong2019` / `kong2019_8net`
- merged-canonical group atlas construction
- present-group overview generation
- optional backfill of group functional atlas labels

The canonical network axis is still atlas-specific, and the merge map is preserved in [merged_canonical_group_atlas_plan.md](/home/yojiao/hpchome/hipp/docs/hipp_parcellation_network/merged_canonical_group_atlas_plan.md). The important cases are:

- `Kong2019`: `TemporalParietal -> Default`, `Noise` is not retained
- `kong2019_8net`: same canonical merge logic, but `TemporalParietal` stays independent in the 8-network layout where required by the upstream mapping file
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

## Default Cohort

Use the default QC-passed 153-subject cohort:

- `manifests/hcp_7t_hippocampus_fmri_qc_pass_153.txt`

Do not silently expand to 166 subjects.

## Active Entry Points

- `scripts/hipp_parcellation_network/materialize_kong7_kong8_pearson_partial_fc.py`
- `scripts/hipp_parcellation_network/run_merged_canonical_group_atlas.py`
- `scripts/hipp_parcellation_network/build_present_group_overview_merged_canonical.py`
- `scripts/hipp_parcellation_network/materialize_group_functional_atlas_labels.py`

## Canonical Command Sequence

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/hipp_parcellation_network/materialize_kong7_kong8_pearson_partial_fc.py \
  --shared-root outputs_migration/hipp_group_inputs_shared \
  --jobs 6

/home/yojiao/.conda/envs/py314/bin/python scripts/hipp_parcellation_network/run_merged_canonical_group_atlas.py \
  --shared-root outputs_migration/hipp_group_inputs_shared \
  --resume \
  --coverage-thresholds 0.80 \
  --atlases kong2019 kong2019_8net \
  --methods group-network-spectral group-intrinsic-spectral \
  --out-root outputs_migration/hipp_group_merged_canonical_atlas

/home/yojiao/.conda/envs/py314/bin/python scripts/hipp_parcellation_network/build_present_group_overview_merged_canonical.py \
  --group-root outputs_migration/hipp_group_merged_canonical_atlas \
  --present-dir present_group \
  --atlases kong2019 kong2019_8net \
  --methods group-network-spectral group-intrinsic-spectral \
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
  kong2019_group-network-spectral_group_overview.png
  kong2019_group-intrinsic-spectral_group_overview.png
  kong2019_8net_group-network-spectral_group_overview.png
  kong2019_8net_group-intrinsic-spectral_group_overview.png
  group_overview_manifest.json
  README.md
```

## Interpretation Rules

- merged-canonical group outputs use cluster IDs ordered by hippocampal long-axis position (`head -> tail`), not by cluster size
- authoritative runtime products are under `outputs_migration/hipp_group_merged_canonical_atlas/`
- `present_group/` is a presentation/export surface, not the canonical runtime input source
- K-selection panels use direct subject-split `group_ari_mean`; null references are reported in ARI space

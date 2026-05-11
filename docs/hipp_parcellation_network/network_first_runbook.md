# Group Analysis Runbook

## Active Scope

This directory now supports **group analysis only**.

Active workflow:

- cortex reference extraction for `kong2022` (Route B: public 400-ROI dlabel)
- FC materialization for `kong2019` / `kong2019_8net` / `kong2022`
- merged-canonical group atlas construction
- present-group overview generation
- optional backfill of group functional atlas labels

For the T1only HippUnfold branch, do not use the old 153-subject T2only-era manifest or the default shared root. Use the dedicated runbook:

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

## 2026-05-03 Kong2019 Fast-PFM T1only Update

This update adds a dedicated Kong2019 T1only root without `routea/routeb` naming:

- root: `outputs_experiments/hippunfold_recommended_t1w_20260423/kong2019_fast_pfm_t1only/`
- shared inputs: `outputs_experiments/hippunfold_recommended_t1w_20260423/kong2019_fast_pfm_t1only/shared_inputs/pfm_151/`
- cohort manifest: `outputs_experiments/hippunfold_recommended_t1w_20260423/kong2019_fast_pfm_t1only/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_151.txt`

Input migration policy for this branch:

- use `mv` from `_archive/outputs_migration/hipp_group_inputs_shared` into `shared_inputs/pfm_151`
- do not move active `kong2022/shared_inputs/*` trees
- hippocampal `surface/*` and all `fc/kong2019*` under the old tree are not reused as final outputs
- recompute `kong2019_8net` products under the new root from T1only hippocampal timeseries

Redundancy-aware note:

- `run_merged_canonical_group_atlas.py --k-selection-mode redundancy-aware` currently reads `fc/kong2022/*` as the ROI-profile bank for redundancy checks
- keep `fc/kong2022/*` available under `shared_inputs/pfm_151/sub-*/fc/kong2022/`

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

Use the active default QC-passed T1only 151-subject cohort:

- `outputs_experiments/hippunfold_recommended_t1w_20260423/kong2022/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_151.txt`
- `outputs_experiments/hippunfold_recommended_t1w_20260423/kong2022/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt`

Do not silently expand to 166 subjects.

## Active Entry Points

- `scripts/cortex/run_cortex_kong2022_reference_batch.py`
- `scripts/hipp_parcellation_network/materialize_kong7_kong8_pearson_partial_fc.py`
- `scripts/hipp_parcellation_network/materialize_kong2022_vertex_to_roi_fc.py`
- `scripts/hipp_parcellation_network/run_merged_canonical_group_atlas.py`
- `scripts/hipp_parcellation_network/build_present_group_overview_merged_canonical.py`
- `scripts/hipp_parcellation_network/plot_wta_top_long_axis_network_profiles.py`
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
  --methods intrinsic-spectral network-spectral wta-top \
  --jobs 6 \
  --out-root outputs_migration/hipp_group_merged_canonical_atlas

/home/yojiao/.conda/envs/py314/bin/python scripts/hipp_parcellation_network/build_present_group_overview_merged_canonical.py \
  --group-root outputs_migration/hipp_group_merged_canonical_atlas \
  --present-dir present_group \
  --atlases kong2022 \
  --methods intrinsic-spectral network-spectral wta-top \
  --coverage-threshold 0.80

/home/yojiao/.conda/envs/py314/bin/python scripts/hipp_parcellation_network/materialize_group_functional_atlas_labels.py \
  --out-root outputs_migration/hipp_group_merged_canonical_atlas
```

Kong2019 T1only (speed mode on, no route tag):

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/hipp_parcellation_network/materialize_kong7_kong8_pearson_partial_fc.py \
  --shared-root outputs_experiments/hippunfold_recommended_t1w_20260423/kong2019_fast_pfm_t1only/shared_inputs/pfm_151 \
  --subjects-manifest outputs_experiments/hippunfold_recommended_t1w_20260423/kong2019_fast_pfm_t1only/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_151.txt \
  --atlases kong2019_8net \
  --smoothings 2mm 4mm \
  --hemis L R \
  --jobs 4

/home/yojiao/.conda/envs/py314/bin/python scripts/hipp_parcellation_network/run_merged_canonical_group_atlas.py \
  --shared-root outputs_experiments/hippunfold_recommended_t1w_20260423/kong2019_fast_pfm_t1only/shared_inputs/pfm_151 \
  --surface-root outputs_experiments/hippunfold_recommended_t1w_20260423/kong2019_fast_pfm_t1only/shared_inputs/pfm_151 \
  --out-root outputs_experiments/hippunfold_recommended_t1w_20260423/kong2019_fast_pfm_t1only/hipp_group_merged_canonical_atlas_kong2019_allon \
  --subjects-manifest outputs_experiments/hippunfold_recommended_t1w_20260423/kong2019_fast_pfm_t1only/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_151.txt \
  --atlases kong2019_8net \
  --methods intrinsic-spectral network-spectral \
  --smoothings 2mm 4mm \
  --hemis L R \
  --coverage-thresholds 0.80 \
  --k-selection-mode redundancy-aware \
  --redundancy-early-stop \
  --jobs 4 \
  --resume

/home/yojiao/.conda/envs/py314/bin/python scripts/hipp_parcellation_network/run_merged_canonical_group_atlas.py \
  --shared-root outputs_experiments/hippunfold_recommended_t1w_20260423/kong2019_fast_pfm_t1only/shared_inputs/pfm_151 \
  --surface-root outputs_experiments/hippunfold_recommended_t1w_20260423/kong2019_fast_pfm_t1only/shared_inputs/pfm_151 \
  --out-root outputs_experiments/hippunfold_recommended_t1w_20260423/kong2019_fast_pfm_t1only/hipp_group_merged_canonical_atlas_kong2019_alloff \
  --subjects-manifest outputs_experiments/hippunfold_recommended_t1w_20260423/kong2019_fast_pfm_t1only/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_151.txt \
  --atlases kong2019_8net \
  --methods intrinsic-spectral network-spectral \
  --smoothings 2mm 4mm \
  --hemis L R \
  --coverage-thresholds 0.80 \
  --k-selection-mode redundancy-aware \
  --redundancy-early-stop \
  --jobs 4 \
  --no-subject-fc-binarize \
  --resume

/home/yojiao/.conda/envs/py314/bin/python scripts/hipp_parcellation_network/build_present_group_overview_merged_canonical.py \
  --group-root outputs_experiments/hippunfold_recommended_t1w_20260423/kong2019_fast_pfm_t1only/hipp_group_merged_canonical_atlas_kong2019_allon \
  --present-dir present_group_t1only \
  --methods intrinsic-spectral network-spectral \
  --atlases kong2019_8net \
  --coverage-threshold 0.80 \
  --mode-tag allon \
  --skip-readme \
  --skip-manifest

/home/yojiao/.conda/envs/py314/bin/python scripts/hipp_parcellation_network/build_present_group_overview_merged_canonical.py \
  --group-root outputs_experiments/hippunfold_recommended_t1w_20260423/kong2019_fast_pfm_t1only/hipp_group_merged_canonical_atlas_kong2019_alloff \
  --present-dir present_group_t1only \
  --methods intrinsic-spectral network-spectral \
  --atlases kong2019_8net \
  --coverage-threshold 0.80 \
  --mode-tag alloff \
  --skip-readme \
  --skip-manifest

/home/yojiao/.conda/envs/py314/bin/python scripts/hipp_parcellation_network/plot_wta_top_long_axis_network_profiles.py \
  --group-root outputs_experiments/hippunfold_recommended_t1w_20260423/kong2022/hipp_group_merged_canonical_atlas_routea_10perc \
  --present-dir present_group_t1only \
  --atlas kong2022 \
  --route-tag routea \
  --smoothings 2mm 4mm \
  --n-bins 25
```

## Active Outputs

```text
outputs_migration/hipp_group_merged_canonical_atlas/
  <method>/<atlas>/<smoothing>/<coverage>/hemi_<L|R>/...

present_group/
  kong2022_intrinsic-spectral_group_overview.png
  kong2022_network-spectral_group_overview.png
  kong2022_wta-top_group_overview.png
  group_overview_manifest.json
  README.md

present_group_t1only/
  kong2019_intrinsic-spectral_group_overview_allon.png
  kong2019_intrinsic-spectral_group_overview_alloff.png
  kong2019_network-spectral_group_overview_allon.png
  kong2019_network-spectral_group_overview_alloff.png
  kong2022_wta-top_long_axis_network_profile_routea_2mm_bins25.png
  kong2022_wta-top_long_axis_network_profile_routea_4mm_bins25.png
  kong2022_wta-top_long_axis_network_profile_routea_2mm_bins25.tsv
  kong2022_wta-top_long_axis_network_profile_routea_4mm_bins25.tsv
  kong2022_wta-top_long_axis_network_profile_routea_bins25_manifest.json
```

## Interpretation Rules

- merged-canonical group outputs use cluster IDs ordered by hippocampal long-axis position (`head -> tail`), not by cluster size
- authoritative runtime products are under `outputs_migration/hipp_group_merged_canonical_atlas/`
- `present_group/` is a presentation/export surface, not the canonical runtime input source
- K-selection panels use direct subject-split `group_ari_mean`; null references are reported in ARI space
- spectral methods (`network-spectral`, `intrinsic-spectral`) can run `--k-selection-mode redundancy-aware`
- subject surface topology/order now follows template-first mode using shared group-average surfaces (recommended shared path: `outputs_experiments/hippunfold_recommended_t1w_20260423/kong2022/templates/hipp_group_average_t1only_151/group_average_surfaces/`)
- `group-roi-sim-spectral` is frozen in this round (not implemented in active defaults); see `docs/hipp_parcellation_network/group_roi_sim_spectral_frozen_plan.md`
  - input path: per-subject `vertex x ROI` FC preprocessing (same knobs as `network-spectral`)
  - then transform to per-subject `vertex x vertex` centered-cosine similarity before clustering
- `run_merged_canonical_group_atlas.py --jobs N` now parallelizes candidate-`K` evaluation within each combo, capped at `min(N, #K, 10)`
- the script suppresses scikit-learn's `"Graph is not fully connected"` spectral-embedding warning to keep logs readable during large `K` sweeps
- redundancy-aware `K` selection uses:
  - profile redundancy gate: `max_profile_similarity <= 0.80`, `redundant_pair_count == 0`
  - final K selection (three-tier over the survivor pool that passed all gates):
    1. 1-SE rule: keep survivors within 1 SE of the survivor-pool best mean ARI
    2. local-max preference: among within-1SE survivors, prefer ARI-curve local maxima
    3. parsimony tiebreak: smallest K when quality is genuinely tied
- overview `K` diagnostics show per-K reject reasons: `redundant`, `nontrivial_fail`
- if a combo cannot select a valid `K`, the driver writes `group_atlas_manifest.json` with `status=skipped_*` and continues remaining combos
- `--resume` now skips a combo only when the stored `resume_input_signature` matches current upstream inputs and K-selection controls; legacy/no-signature manifests or upstream-changed inputs are automatically invalidated and recomputed
- maintained defaults are: `intrinsic-spectral`, `network-spectral`, `wta-top`
- `cluster-roi-cortex` rendering now supports shared/public cortex template fallback from `external/` and auto-generates missing hemisphere `sulc.func.gii` from `*_sulc.dscalar.nii` when needed

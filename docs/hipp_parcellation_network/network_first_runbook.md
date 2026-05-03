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
  --methods group-intrinsic-spectral group-network-spectral \
  --jobs 6 \
  --out-root outputs_migration/hipp_group_merged_canonical_atlas

/home/yojiao/.conda/envs/py314/bin/python scripts/hipp_parcellation_network/build_present_group_overview_merged_canonical.py \
  --group-root outputs_migration/hipp_group_merged_canonical_atlas \
  --present-dir present_group \
  --atlases kong2022 \
  --methods group-intrinsic-spectral group-network-spectral \
  --coverage-threshold 0.80

/home/yojiao/.conda/envs/py314/bin/python scripts/hipp_parcellation_network/materialize_group_functional_atlas_labels.py \
  --out-root outputs_migration/hipp_group_merged_canonical_atlas
```

## Active Outputs

```text
outputs_migration/hipp_group_merged_canonical_atlas/
  <method>/<atlas>/<smoothing>/<coverage>/hemi_<L|R>/...

present_group/
  kong2022_group-intrinsic-spectral_group_overview.png
  kong2022_group-network-spectral_group_overview.png
  group_overview_manifest.json
  README.md
```

## Interpretation Rules

- merged-canonical group outputs use cluster IDs ordered by hippocampal long-axis position (`head -> tail`), not by cluster size
- authoritative runtime products are under `outputs_migration/hipp_group_merged_canonical_atlas/`
- `present_group/` is a presentation/export surface, not the canonical runtime input source
- K-selection panels use direct subject-split `group_ari_mean`; null references are reported in ARI space
- spectral methods (`group-network-spectral`, `group-intrinsic-spectral`) can run `--k-selection-mode redundancy-aware`
- subject surface topology/order now follows template-first mode using shared group-average surfaces (recommended shared path: `outputs_experiments/hippunfold_recommended_t1w_20260423/kong2022/templates/hipp_group_average_t1only_151/group_average_surfaces/`)
- `group-roi-sim-spectral` is frozen in this round (not implemented in active defaults); see `docs/hipp_parcellation_network/group_roi_sim_spectral_frozen_plan.md`
  - input path: per-subject `vertex x ROI` FC preprocessing (same knobs as `group-network-spectral`)
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
- maintained defaults are spectral-only: `group-intrinsic-spectral`, `group-network-spectral`
- `group-network-top` / `group-network-all` are legacy branches and are retired from default runs
- `cluster-roi-cortex` rendering now supports shared/public cortex template fallback from `external/` and auto-generates missing hemisphere `sulc.func.gii` from `*_sulc.dscalar.nii` when needed

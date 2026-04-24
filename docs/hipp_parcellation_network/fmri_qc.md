# fMRI QC — HCP 7T Hippocampal Surface Sampling

## Summary

- **Total subjects with hippunfold output:** 166
- **QC-passed (default analysis set):** 153
- **QC-excluded:** 13

Subject list: `manifests/hcp_7t_hippocampus_fmri_qc_pass_153.txt`

---

## QC Criteria

Two sequential checks; a subject must pass **both** to be included.

### Step 1 — HippUnfold segmentation quality (Dice)

- Metric: `*_desc-unetf3d_dice.tsv` under `outputs_migration/dense_corobl_batch/<sub>/hippunfold/<sub>/qc/`
- Threshold: Dice ≥ 0.70 for **both** hemispheres
- Result: **0 / 166 excluded** (all passed; min Dice = 0.703 R hemi)

### Step 2 — fMRI surface signal quality (tSNR null %)

- Metric: percentage of hippocampal surface vertices with tSNR < 25, or with temporal std == 0 (constant signal)
- tSNR formula: `10000 / σ_t` (HCP mode-normalised, hp2000-filtered convention)
- Surface: `outputs_migration/hipp_functional_parcellation_network/_shared/<sub>/surface/raw/<sub>_hemi-{L,R}_space-corobl_den-512_label-hipp_bold.func.gii`
- Threshold: **either hemisphere > 20%** → exclude
- Result: **13 / 166 excluded**

#### Excluded subjects

| Subject | L null% | R null% |
|---------|---------|---------|
| sub-680957 | 56.1 | 13.1 |
| sub-898176 | 14.5 | 50.6 |
| sub-901442 | 33.2 | 42.2 |
| sub-173334 | 18.0 | 36.5 |
| sub-198653 |  8.0 | 29.9 |
| sub-155938 | 12.3 | 28.9 |
| sub-783462 |  8.2 | 27.0 |
| sub-109123 | 11.3 | 26.8 |
| sub-102816 | 26.2 |  3.1 |
| sub-164131 |  6.8 | 24.6 |
| sub-158136 |  7.4 | 24.6 |
| sub-146129 |  3.9 | 22.3 |
| sub-550439 |  9.0 | 21.1 |

#### Null % distribution across all 166 subjects (QC step 2)

| | Left | Right |
|-|------|-------|
| median | 2.15% | 4.88% |
| mean   | 4.03% | 7.03% |
| max    | 56.05% | 50.59% |
| min    | 0.00% | 0.00% |

Distribution plot: `outputs_migration/null_pct_distribution.png`
Script: `scripts/plot_null_pct_distribution.py`

---

## Default Analysis Convention

All downstream analyses (parcellation, FC, gradients) use the **153-subject QC-passed set** by default.

Scripts and pipelines should load subjects from:
```
manifests/hcp_7t_hippocampus_fmri_qc_pass_153.txt
```

To iterate in shell:
```bash
mapfile -t SUBJECTS < manifests/hcp_7t_hippocampus_fmri_qc_pass_153.txt
# Each entry is a bare ID (e.g. 100610); prepend sub- as needed:
for id in "${SUBJECTS[@]}"; do
    sub="sub-${id}"
    echo "$sub"
done
```

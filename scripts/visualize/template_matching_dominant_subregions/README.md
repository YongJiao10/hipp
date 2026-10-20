# Iowa 3T Template Matching: Dominant Subregions & Dominance Heatmaps

The formal renderer is maintained independently in
`render_template_matching_dominant_subregions.py`. The older four-grid
implementation remains in `render_template_matching_dominant_subregions_legacy.py`.
Both import scalar export, folded-scene assembly and scene capture from
`scripts/core_workflow/lib/workbench_scene_batch.py`; neither loads the retired
`06_05` renderer. Neither renderer imports or executes code from `ctxreg_worktree`.

Publication-ready visualization suite displaying the group-level hippocampal functional subregion atlases derived from individual template matching alongside percentage dominance agreement heatmaps across the repository's canonical $2 \times 2$ grid architecture ($2\text{mm} / 4\text{mm}$ hippocampal surface smoothing $\times$ $36\text{P} / 22\text{P}$ fMRIPrep/xcp-d confound regression).

Each grid figure presents all **4 Iowa 3T cohorts** in stacked comparative rows:

1. **Healthy Controls (`HC`)**: $N=67$ (`sub-1001` to `sub-1068`, omitting `sub-1043`)
2. **Depression Patients (`Patient`)**: $N=26$ (`sub-3001` to `sub-3027`, `sub-3003` excluded)
3. **Pure MDD Subgroup (`Pure MDD`)**: $N=7$ (unipolar MDD without comorbid psychiatric disorders)
4. **Comorbid MDD Subgroup (`Comorbid MDD`)**: $N=19$ (MDD comorbid with anxiety/PTSD/substance)

---

## 1. Subregion Nomenclature & Color Coding

Each vertex $v \in \{0, \dots, 511\}$ on the bilateral midthickness hippocampal surfaces (1024 total bilateral vertices) is assigned to one of three functional subregions via template matching:

- **Anterior / VAN**: Ventral Attention Network, `#dc5cff` (purple)
- **Middle / SMN**: Somatomotor Network, `#6396c2` (blue)
- **Posterior / DMN**: Default Mode Network, `#d85c6b` (coral/red)

### Metrics Computed per Cohort & Grid Cell

1. **Dominant Subregion Atlas**: Vertex-wise majority winner:
   $$\arg\max_{k \in \{1,2,3\}} N_k(v)$$
   with deterministic tie-breaking based on mean template correlation across subjects.
2. **Percentage Dominance Heatmap**: Vertex-wise subject agreement for the winning label:
   $$\text{Agreement}(v) = \frac{\max_{k \in \{1,2,3\}} N_k(v)}{N} \times 100\%$$
   Rendered with the continuous `magma` colormap scaled across $[33.3\%, 80.0\%]$ (chance level to high empirical consensus, with $\ge 80\%$ highlighted in bright yellow/white for maximum contrast).

---

## 2. Figure Layout & Architecture

Each master figure is rendered at $1940 \times 2400\text{ px}$ (300 DPI) with a vertical stacked-row layout:

```
+-----------------------------------------------------------------------------------------+
| MASTER TITLE: Iowa 3T Template Matching — Dominant Subregions & Dominance Heatmap        |
| SUBTITLE: Grid Cell Condition | Resolution & Denoising Description                      |
| [Subregion Legend: Anterior/VAN, Middle/SMN, Posterior/DMN]   [Agreement Colorbar: 33.3-80%] |
+-----------------------------------------------------------------------------------------+
| ROW 1: Healthy Controls (N=67)                                                          |
| [Metadata Card]  |  [ Dominant Atlas: 4 Views ]  |  [ Dominance Heatmap: 4 Views ]      |
|                  |  VAN: xx% | SMN: xx% | DMN: xx%|  Mean: xx% | Range: xx% - xx%       |
+-----------------------------------------------------------------------------------------+
| ROW 2: Depression Patients (N=26)                                                       |
| [Metadata Card]  |  [ Dominant Atlas: 4 Views ]  |  [ Dominance Heatmap: 4 Views ]      |
|                  |  VAN: xx% | SMN: xx% | DMN: xx%|  Mean: xx% | Range: xx% - xx%       |
+-----------------------------------------------------------------------------------------+
| ROW 3: Pure MDD Subgroup (N=7)                                                          |
| [Metadata Card]  |  [ Dominant Atlas: 4 Views ]  |  [ Dominance Heatmap: 4 Views ]      |
|                  |  VAN: xx% | SMN: xx% | DMN: xx%|  Mean: xx% | Range: xx% - xx%       |
+-----------------------------------------------------------------------------------------+
| ROW 4: Comorbid MDD Subgroup (N=19)                                                     |
| [Metadata Card]  |  [ Dominant Atlas: 4 Views ]  |  [ Dominance Heatmap: 4 Views ]      |
|                  |  VAN: xx% | SMN: xx% | DMN: xx%|  Mean: xx% | Range: xx% - xx%       |
+-----------------------------------------------------------------------------------------+
```

### Component Breakdown per Row

- **Left Metadata Card ($w=280\text{ px}$)**: Cohort badge, sample size ($N$), bilateral vertex count (1024 vertices), and description.
- **Center: Dominant Atlas ($w=790\text{ px}$)**:
  - 4-view tight composite: LH Lateral, RH Lateral, LH Medial, RH Medial.
  - Subregion composition box below panels: vertex counts and percentages for Anterior/VAN, Middle/SMN, and Posterior/DMN.
- **Right: Dominance Heatmap ($w=790\text{ px}$)**:
  - 4-view tight composite: LH Lateral, RH Lateral, LH Medial, RH Medial.
  - Dominance statistics box below panels: Mean agreement, Median agreement, and Range [Min% – Max%].

---

## 3. Scientific Summary & Statistics

All metrics across the 16 cohort $\times$ grid combinations are recorded in `template_matching_dominant_subregions_stats.csv`:

| Grid Cell | Cohort | $N$ | Anterior/VAN (%) | Middle/SMN (%) | Posterior/DMN (%) | Mean Agreement (%) | Max Agreement (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **`2mm_36p`** | Healthy Controls | 67 | 43.7% (447 vtx) | 26.0% (266 vtx) | 30.4% (311 vtx) | 55.1% | 83.6% |
| | Depression Patients | 26 | 36.5% (374 vtx) | 25.4% (260 vtx) | 38.1% (390 vtx) | 56.6% | 92.3% |
| | **Pure MDD Subgroup** | 7 | **29.0%** (297 vtx) | 22.3% (228 vtx) | **48.7%** (499 vtx) | **63.7%** | **100.0%** |
| | Comorbid MDD Subgroup | 19 | 37.6% (385 vtx) | 29.1% (298 vtx) | 33.3% (341 vtx) | 56.5% | 94.7% |
| **`2mm_22p`** | Healthy Controls | 67 | 43.4% (444 vtx) | 30.2% (309 vtx) | 26.5% (271 vtx) | 51.8% | 76.1% |
| | Depression Patients | 26 | 35.5% (364 vtx) | 32.4% (332 vtx) | 32.0% (328 vtx) | 56.8% | 88.5% |
| | **Pure MDD Subgroup** | 7 | 26.7% (273 vtx) | 35.6% (365 vtx) | **37.7%** (386 vtx) | **62.3%** | **100.0%** |
| | Comorbid MDD Subgroup | 19 | 37.2% (381 vtx) | 29.6% (303 vtx) | 33.2% (340 vtx) | 57.2% | 94.7% |
| **`4mm_36p`** | Healthy Controls | 67 | 43.2% (442 vtx) | 27.8% (285 vtx) | 29.0% (297 vtx) | 58.3% | 89.6% |
| | Depression Patients | 26 | 35.2% (361 vtx) | 29.3% (300 vtx) | 35.5% (363 vtx) | 59.8% | 96.2% |
| | **Pure MDD Subgroup** | 7 | 26.5% (271 vtx) | 26.4% (270 vtx) | **47.2%** (483 vtx) | **66.8%** | **100.0%** |
| | Comorbid MDD Subgroup | 19 | 34.9% (357 vtx) | 32.6% (334 vtx) | 32.5% (333 vtx) | 59.5% | 94.7% |
| **`4mm_22p`** | Healthy Controls | 67 | 42.8% (438 vtx) | 32.9% (337 vtx) | 24.3% (249 vtx) | 54.5% | 80.6% |
| | Depression Patients | 26 | 34.1% (349 vtx) | 34.7% (355 vtx) | 31.2% (320 vtx) | 60.2% | 96.2% |
| | **Pure MDD Subgroup** | 7 | 24.8% (254 vtx) | 40.4% (414 vtx) | 34.8% (356 vtx) | **65.5%** | **100.0%** |
| | Comorbid MDD Subgroup | 19 | 37.2% (381 vtx) | 32.2% (330 vtx) | 30.6% (313 vtx) | 60.3% | **100.0%** |

### Key Scientific Findings

1. **DMN Expansion in Pure MDD**: In unipolar Pure MDD patients ($N=7$), the Posterior / DMN subregion expands dramatically to **48.7%** (nearly half of the entire hippocampus under 2mm 36P), compared to only **30.4%** in HC. Dominance agreement in the posterior subregion reaches **100% consensus** across all 7 patients.
2. **Anterior / VAN Contraction**: In Pure MDD, Anterior / VAN contracts from **43.7%** in HC down to **29.0%** in Pure MDD (2mm 36P).
3. **Comorbid MDD Preservation**: Comorbid MDD patients ($N=19$) exhibit subregion proportions intermediate between HC and Pure MDD (VAN: ~37%, SMN: ~29%, DMN: ~33%), indicating distinct network engagement between pure and comorbid depression phenotypes.
4. **Smoothing Effect**: 4mm smoothing increases average dominance agreement by ~3–4% across all cohorts by reducing high-frequency vertex-level noise without altering the overall macroscopic boundaries.

---

## 4. Input Sources & Provenance

- **36P Template Matching Results**:
  `outputs_iowa_jj_tms/work/hcp151_fixed_template_matching/subjects/sub-{id}/matching/{smoothing}/hemi-{hemi}/labels.npy`
- **Explicit 2mm/36P signal branches**:
  - Standard: `outputs_iowa_jj_tms/work/hcp151_fixed_template_matching/subjects/...`
  - CtxReg: `outputs_iowa_jj_tms/work/hcp151_fixed_template_matching/routea_ctxreg/subjects/...`
  Both branches require the exact per-hemisphere `combo_manifest.json` contract;
  the renderer resolves the manifest-declared `labels` and `scores` artifacts
  and validates the fixed-template label semantics before rendering.
- **22P Template Matching Results**:
  `outputs_iowa_jj_tms/work/hcp151_fixed_template_matching_22p/subjects/sub-{id}/matching/{smoothing}/hemi-{hemi}/labels.npy`
- **Cohort Manifests**:
  - `manifests/jj_tms_hc_67_subjects.txt`
  - `manifests/jj_tms_patient_26_subjects.txt`
  - `manifests/jj_tms_pure_mdd_7_subjects.txt`
  - `manifests/jj_tms_comorbid_mdd_19_subjects.txt`
- **Group Template Surfaces**:
  `outputs_iowa_jj_tms/inputs/iowa_jj_tms/kong2022/templates/hipp_group_average_t1only_151/group_average_surfaces/group-average_hemi-{hemi}_space-corobl_den-512_label-hipp_midthickness.surf.gii`
- **Surface Spec File**:
  `outputs_iowa_jj_tms/inputs/iowa_jj_tms/hippunfold/dense_corobl_batch/sub-1001/hippunfold/sub-1001/surf/sub-1001_den-512_label-hipp_surfaces.spec`
- **Workbench Locked Scene**:
  `config/wb_locked_native_view_lateral_medial.scene`

---

## 5. Execution & Reproduction

Run the renderer with the repository's explicit Python interpreter:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/visualize/template_matching_dominant_subregions/render_template_matching_dominant_subregions.py
```

With no branch flag, the maintained entry point preserves the established
four-grid behavior. To render the explicit Standard and CtxReg 2mm/36P
comparison from the formal checkout:

```bash
/home/yojiao/.conda/envs/py314/bin/python \
  scripts/visualize/template_matching_dominant_subregions/render_template_matching_dominant_subregions.py \
  --signal-branch all \
  --repo-root /home/yojiao/hpchome/hipp \
  --data-root /home/yojiao/hpchome/hipp
```

### Outputs Generated

- **4 Canonical Master Figures (300 DPI, $1940 \times 2580\text{ px}$)**:
  - `template_matching_dominant_subregions_2mm_36p.png`
  - `template_matching_dominant_subregions_2mm_22p.png`
  - `template_matching_dominant_subregions_4mm_36p.png`
  - `template_matching_dominant_subregions_4mm_22p.png`
- **Summary Metrics**: `template_matching_dominant_subregions_stats.csv`
- **Provenance Manifest**: `template_matching_dominant_subregions_manifest.json` (records run timestamp, input paths, output SHA-256 hashes, and per-cohort metrics)
- **Workbench Panel Cache**: `panels/` (contains 32 intermediate 4-view composite PNGs for fast recomposition)

The explicit branch mode (which appends the t-test columns described in §6)
additionally writes the canonical branch products (no `_ttest` suffix):

- `template_matching_dominant_subregions_2mm_36p_standard.png`
- `template_matching_dominant_subregions_2mm_36p_ctxreg.png`
- `template_matching_dominant_subregions_2mm_36p_{standard,ctxreg}_contrasts.csv`
  (36 Welch tests each: 3 contrasts × 3 subregions × count/area_pct × L/R)
- `template_matching_dominant_subregions_2mm_36p_standard__ctxreg_manifest.json`
- Panel cache: `panels_ttest/`; area cache: `ttest_cache/vertex_areas/`

The branch figures are regenerated from the formal inputs rather than copied
from another checkout. Their manifest records the selected branch roots,
roster checksums, matcher label contract, panel paths, and output checksums.

---

## 6. T-Test Branch Mode (formal, no suffix)

The formal renderer appends two columns per cohort row; columns 1–3 (titles,
legend, meta card, atlas, heatmap, stats boxes) are identical to the plain
grid figure:

- **Col 4 — Vertex-count t-test vs HC**: per-subject, per-hemisphere vertex
  counts (Left/Right hippocampus kept separate, no bilateral summation).
- **Col 5 — Area-percentage t-test vs HC**: per-subject, per-hemisphere
  subregion area expressed as **% of that subject's own hemisphere total
  midthickness area** (sum over all 512 vertices, including the rare label-0
  vertices: ≤31 vertices / 6% in one HC hemisphere, ≤5 vertices elsewhere).
  `labels.npy` weighted by native `space-T1w_den-512 midthickness` vertex
  areas (`wb_command -surface-vertex-areas`, cross-checked against the
  `cortex_network_area` numpy formula; missing surfaces fail closed). Temporary
  area metrics are created under the repository `tmp/` workspace. Corobl
  surfaces remain display assets and are not used for physical areas.
  Normalizing by whole-hippocampus size removes between-subject head-size /
  segmentation scaling, so this column tests *relative redistribution*
  between subregions, while the count column keeps the absolute scale.

Each panel shows 6 clusters (Ant/Mid/Post × L/R) with HC vs cohort bars
(mean ± SEM + individual dots), an explicit color legend (HC slate vs cohort
tint), and bracket stars. Statistics: two-sided Welch
`ttest_ind(equal_var=False)`, **uncorrected (no FDR)**,
`+ 0.05≤p<0.10 (trend), * p<0.05, ** p<0.01, *** p<0.001`. Effect size shown
after each t value is Welch-consistent Cohen's `d = (mean_case − mean_ctrl) /
√((sa² + sb²)/2)` (unweighted pooled SD, sign follows case-minus-ctrl; also
recorded as `cohens_d` in the contrasts CSVs). The HC row leaves
both columns completely blank (no panels, titles, or boxes).

```bash
/home/yojiao/.conda/envs/py314/bin/python \
  scripts/visualize/template_matching_dominant_subregions/render_template_matching_dominant_subregions.py \
  --signal-branch all \
  --repo-root /home/yojiao/hpchome/hipp \
  --data-root /home/yojiao/hpchome/hipp
```

Grid mode without `--signal-branch` keeps the established four-grid behavior
and writes `_ttest`-suffixed outputs, so it never overwrites the canonical
tracked grid products.

2mm_36p standard finding (uncorrected, exploratory; Pure MDD N=7): Pure MDD
Posterior/DMN expansion is now nominally significant or trend in **both**
metrics × **both** hemispheres — count `L p=0.046 *` / `R p=0.071 +`,
area_pct `L p=0.031 *` / `R p=0.062 +`. Note the area_pct normalization moved
the left-hemisphere area contrast across the significance boundary (absolute
area was `L p=0.052 n.s.`), consistent with whole-hippocampus size variance
being removed. All ctxreg contrasts are `n.s.`

Interpretation caveat: the three area_pct values within a hemisphere sum to
≈100%, so the three within-hemisphere tests are negatively coupled — a
"DMN expansion" mathematically entails Ant/Mid contraction. Report it as a
*relative redistribution*, complementary to the absolute count column.

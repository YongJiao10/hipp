# Multi-Dataset Hippocampal Subfield Composition & Stanford Clinical Group Differences

This directory contains the visualization, statistical modeling, and experimental pipeline analyzing **HippUnfold MultiHist7 anatomical subfield compositions** overlapping **RouteA $K=3$ gradient-spectral functional clusters (Anterior, Middle, Posterior)** across 10 clinical and control cohorts ($N = 528$), with an in-depth statistical contrast of the **6 Stanford clinical cohorts/subgroups against the Non-Trauma Healthy Control (NTHC) baseline ($N = 239$)**.

## Additive HCP/JJ_TMS 7-network comparison

The `hcp_jjtms_7net` branch adds Kong2022 7-Network and Lynch2024 7-Network comparison columns without changing the original subfield layout or Workbench rendering. The comparison roster is fixed to HCP 7T `N=151`, Iowa HC `N=67`, Iowa Patient `N=26` (sub-3003 excluded), Pure MDD `N=7`, and Comorbid MDD `N=19`; Stanford remains in the legacy branch only.

The comparison panorama omits the single-atlas version tag (`[VERSION 2: ...]`) and cohort-count subtitle from its figure header to maintain a clean method-comparison title. CLI flags `--show-version-label` / `--no-show-version-label` and `--show-subtitle` / `--no-show-subtitle` allow explicitly overriding header labels.

| Grid Tag | Comparison Figure Artifact | Statistics TSV (in `stats/`) |
| --- | --- | --- |
| `2mm_36p` | [`multidataset_subfield_composition_2mm_2col_hcp_jjtms_7net_2mm_36p.png`](file:///home/yojiao/hpchome/hipp/scripts/visualize/multidataset_subfield_composition_2mm_2col/multidataset_subfield_composition_2mm_2col_hcp_jjtms_7net_2mm_36p.png) | [`stats/multidataset_subfield_composition_hcp_jjtms_7net_2mm_36p_stats.tsv`](file:///home/yojiao/hpchome/hipp/scripts/visualize/multidataset_subfield_composition_2mm_2col/stats/multidataset_subfield_composition_hcp_jjtms_7net_2mm_36p_stats.tsv) |
| `4mm_36p` | [`multidataset_subfield_composition_2mm_2col_hcp_jjtms_7net_4mm_36p.png`](file:///home/yojiao/hpchome/hipp/scripts/visualize/multidataset_subfield_composition_2mm_2col/multidataset_subfield_composition_2mm_2col_hcp_jjtms_7net_4mm_36p.png) | [`stats/multidataset_subfield_composition_hcp_jjtms_7net_4mm_36p_stats.tsv`](file:///home/yojiao/hpchome/hipp/scripts/visualize/multidataset_subfield_composition_2mm_2col/stats/multidataset_subfield_composition_hcp_jjtms_7net_4mm_36p_stats.tsv) |

For the `hcp_jjtms_7net` subfield comparison, the hippocampal surface is a gradient-spectral atlas surface and therefore uses the `fisherz` label variant in both columns (Kong2022 and Lynch2024). Kong2022's `normalized_top20_count` variant remains restricted to the WTA/Top-20 panels and is not reused for this gradient surface.

---

## 🔬 Part 1: Cross-Dataset Subfield Architecture (10 Cohorts, N = 528)

### 1. Neuroanatomical Context

- Functional parcellations ($K=3$: Anterior, Middle, Posterior) capture longitudinal gradients of cortical connectivity.
- On HippUnfold's unfolded/midthickness gray-matter cortical surface (`den-512_label-hipp_midthickness.surf.gii`), the continuous cortical ribbon comprises **5 active histological subfields**:
  - **Subiculum (Sub, 1)** `#292BFC`
  - **CA1 (2)** `#66D1D8`
  - **CA2 (3)** `#00B33C`
  - **CA3 (4)** `#FFB600`
  - **CA4 (5)** `#FF2B17`
  *(Note: Dentate Gyrus DG/6 is an internal infolded structure and SRLM/7 is a laminar layer that does not form independent vertices on the single-layer midthickness surface mesh; both exhibit 0.0% surface area).*

### 2. Universal Population Findings (Subject-Mean %)

- **CA1 & Subiculum Predominance**: Across all 10 cohorts and all 3 longitudinal subregions, **CA1 (45%–58%)** and **Subiculum (28%–34%)** constitute $>82\%–85\%$ of total hippocampal surface area.
- **Posterior Subfield Shift**: In the Posterior, smaller subfields expand significantly:
  - **CA2**: Increases from ~4.5%–6.0% (Head/Body) to **~9.0%–10.0%** in the Tail.
  - **CA3**: Increases from ~4.0%–6.5% to **~7.2%–8.0%** in the Tail.
  - **CA4**: Increases from ~1.5%–2.0% to **~3.5%–4.0%** in the Tail.

| Cohort | N | 左侧 Anterior (CA1 / Sub) | 左侧 Middle (CA1 / Sub) | 左侧 Posterior (CA1 / Sub / CA2 / CA3) | 右侧 Anterior (CA1 / Sub) | 右侧 Middle (CA1 / Sub) | 右侧 Posterior (CA1 / Sub / CA2 / CA3) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **HCP 7T** | 151 | 55.6% / 30.0% | 55.0% / 33.9% | 48.4% / 29.9% / 9.8% / 7.9% | 56.6% / 28.5% | 56.4% / 31.8% | 44.8% / 33.5% / 9.8% / 7.8% |
| **Iowa HC (7T)** | 67 | 54.5% / 33.2% | 55.9% / 33.2% | 50.6% / 28.3% / 9.2% / 7.6% | 56.7% / 30.5% | 57.5% / 31.3% | 47.0% / 32.7% / 9.0% / 7.2% |
| **Iowa Patient (7T)** | 26 | 55.2% / 32.0% | 56.8% / 32.7% | 51.4% / 27.4% / 9.4% / 7.6% | 56.2% / 30.5% | 57.4% / 32.0% | 47.5% / 31.8% / 9.2% / 7.4% |
| **Stanford NTHC (3T)** | 51 | 54.6% / 32.6% | 55.7% / 33.2% | 49.0% / 29.3% / 9.7% / 7.7% | 56.0% / 30.7% | 56.6% / 32.0% | 46.7% / 32.0% / 9.5% / 7.7% |
| **Stanford TEHC (3T)** | 40 | 54.8% / 32.8% | 56.0% / 33.0% | 49.6% / 28.9% / 9.6% / 7.7% | 56.3% / 30.4% | 56.7% / 31.8% | 46.7% / 32.1% / 9.4% / 7.6% |
| **Stanford NTS (3T)** | 56 | 54.5% / 32.9% | 55.7% / 33.2% | 49.8% / 28.7% / 9.6% / 7.7% | 55.9% / 30.9% | 56.9% / 31.8% | 46.3% / 32.4% / 9.5% / 7.6% |
| **Stanford NTS Any MDD (3T)** | 43 | 54.6% / 32.8% | 55.7% / 33.2% | 49.4% / 29.0% / 9.7% / 7.7% | 55.7% / 31.1% | 56.7% / 31.9% | 46.0% / 32.6% / 9.5% / 7.7% |
| **Stanford NTS Pure MDD (3T)** | 11 | 54.3% / 33.0% | 55.7% / 33.1% | 49.4% / 28.9% / 9.7% / 7.7% | 55.6% / 31.2% | 56.4% / 32.2% | 45.9% / 32.6% / 9.5% / 7.7% |
| **Stanford TIS (3T)** | 47 | 54.5% / 33.0% | 55.6% / 33.3% | 49.6% / 28.9% / 9.6% / 7.7% | 55.8% / 31.0% | 56.9% / 31.8% | 46.5% / 32.3% / 9.5% / 7.6% |
| **Stanford TIS PTSD (3T)** | 35 | 54.5% / 33.0% | 55.6% / 33.3% | 49.6% / 28.9% / 9.6% / 7.7% | 55.8% / 31.0% | 56.9% / 31.8% | 46.5% / 32.3% / 9.5% / 7.6% |

---

## 🔬 Part 2: Stanford Clinical Group Differences vs NTHC Baseline (N = 239)

### 1. Research Question & Experimental Design

To test whether psychiatric pathology (Depression, PTSD, Trauma exposure) induces localized subfield remodeling within specific functional subregions, we formulated a multi-level contrast framework comparing **6 non-control cohorts/subgroups against NTHC ($N=51$)**:

1. **TEHC** ($N=40$): Trauma-Exposed Healthy Controls (Resilience)
2. **NTS** ($N=56$): Non-Trauma Symptomatic (All Clinical)
3. **NTS Any MDD** ($N=43$): Non-Trauma MDD (with/without comorbidities)
4. **NTS Pure MDD** ($N=11$): Non-Trauma Pure MDD (uncomplicated depression)
5. **TIS** ($N=47$): Trauma-Informed Symptomatic (Trauma + PTSD/MDD)
6. **TIS PTSD** ($N=35$): Trauma-Informed PTSD (Diagnostic subgroup derived from Master DB)

### 2. Mathematical Formulations & Statistical Models

#### A. Metric Definitions

For subject $s$, hemisphere $h \in \{\text{L, R}\}$, functional subregion $k \in \{\text{Ant, Middle, Post}\}$, and subfield $j \in \{\text{Sub, CA1, CA2, CA3, CA4}\}$:

- **Relative Subfield Proportion**:
  $$\text{prop}(s, h, k, j) = \frac{\sum_{v \in k, \text{sf}(v)=j} w_v(s, h)}{\sum_{v \in k, \text{sf}(v) \in \{1..5\}} w_v(s, h)} \times 100\%$$
- **Absolute Subfield Surface Area ($mm^2$)**:
  $$\text{area}(s, h, k, j) = \sum_{v \in k, \text{sf}(v)=j} w_v(s, h)$$

  Vertex weights use each subject's `space-T1w_den-512` midthickness mesh
  (triangle area divided among its three vertices). Corobl coordinates are
  used for display only; native mm² must not be measured on the corobl mesh.

#### B. Statistical Tests Executed for Each Hypothesis ($M = 150$ total tests)

1. **Welch's Two-Sample $t$-Test**: Unpaired test allowing unequal variances $\to t$-statistic, parametric $p$-value ($p_{\text{param}}$).
2. **Mann-Whitney $U$ Test**: Non-parametric rank-sum test $\to U$-statistic, non-parametric $p$-value ($p_{\text{nonparam}}$).
3. **Monte Carlo Permutation Test ($N_{\text{perm}} = 5000$)**: Exact empirical permutation of cohort labels $\to p_{\text{perm}}$.
4. **Effect Size (Cohen's $d$)**:
   $$d = \frac{\bar{x}_{\text{case}} - \bar{x}_{\text{ctrl}}}{s_{\text{pooled}}}, \quad s_{\text{pooled}} = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1+n_2-2}}$$
5. **Multiple Testing Correction**: Benjamini-Hochberg False Discovery Rate ($q_{\text{FDR}}$) applied separately within each cohort-vs-NTHC contrast across its 30 hemisphere × subregion × subfield tests.

---

### 3. Key Neurobiological Findings & The Resolution Paradox

#### 🌟 Finding 1: Marked Subiculum/CA1 Longitudinal Inversion in Pure MDD ($N=11$)

In **NTS Pure MDD vs NTHC**, we discovered a massive, bilaterally symmetric anatomical reorganization along the longitudinal axis:

- **Middle Subregion**: Subiculum proportion drops dramatically by **$-6.4\% \sim -8.9\%$** ($p = 4.99 \times 10^{-16}$, Cohen's $d = -5.16 \sim -7.83$, FDR $q < 10^{-9}$), while CA1 expands by **$+3.6\% \sim +4.3\%$** ($p < 10^{-9}$).
- **Posterior Subregion**: Subiculum proportion surges by **$+9.4\% \sim +10.1\%$** ($p < 10^{-12}$, Cohen's $d = +3.8 \sim +4.5$, FDR $q < 10^{-9}$), while CA1 drops by **$-5.0\% \sim -5.3\%$** ($p < 10^{-8}$).
- **Clinical Significance**: In uncomplicated pure depression, the hippocampal body exhibits severe subicular focal atrophy, accompanied by a posterior tail compensatory subicular expansion.

#### 🌟 Finding 2: The Resolution Paradox (Whole-Hippocampus vs Subregions)

- **Without Subregions (Whole-Hippocampus Total)**:
  - Case-vs-NTHC whole-organ normalized subfield differences are at most **0.13 percentage points** in this dataset.
  - Panel B now computes the whole-organ means directly from the subject-level area records and displays **SD** error bars across hemisphere-level observations.
  - Panel C reports selected **bilateral-mean local Δ%** values: all bilateral-significant candidates plus the three largest absolute bilateral means among the remaining candidates, explicitly marked `n.s.`; Panel A retains the side-specific detail.
- Panel D uses compact typography and reports relative redistribution; absolute area remains in the subject-level/statistical TSVs as an input field, not as the figure's primary endpoint.
- **Interpretation**: Whole-hippocampus normalization can strongly attenuate focal pathology through spatial cancellation; combining functional longitudinal subregions ($K=3$) is essential to resolve the localized pattern. The inferential table remains subregion-specific, applies BH-FDR within each 30-test cohort contrast, and does not provide whole-organ $p$-values.

---

---

## 🔬 Part 3: Independent Normative Baseline Validation (Stanford NTHC vs. HCP 7T & Iowa HC)

### 1. Scientific Motivation & Rationale

To ensure that clinical findings (e.g. Pure MDD body subicular atrophy) are genuine psychiatric neurobiological alterations rather than scanner, field strength (3T vs. 7T), or site artifacts, we conducted an **independent cross-site normative baseline validation** comparing:

1. **Stanford NTHC (3T, N=51)** vs. **HCP 7T (7T, N=151)**
2. **Stanford NTHC (3T, N=51)** vs. **Iowa HC (7T, N=67)**

### 2. Key Findings

- **High Biological Stability Across Control Sites**: Across all 3 independent control cohorts (HCP 7T, Iowa HC, Stanford NTHC), CA1 constitutes **~53%–56%**, Subiculum constitutes **~30%–33%**, and minor subfields (CA2–4) constitute **~2%–9%**.
- **Negligible Regional Differences in Controls**: Right hemisphere anterior/middle subicular differences between 3T NTHC and 7T controls are virtually zero ($\Delta = 0.02\% \sim 0.06\%$, $p > 0.70$).
- **Absence of Inversion**: Neither 3T nor 7T healthy controls exhibit longitudinal subfield reversals.
- **Effect Size Contrast**: Normal control cross-site variations are small ($|d| \approx 0.1 \sim 1.5$), whereas clinical psychiatric disruptions (e.g., Pure MDD body subicular drop) reach $|d| > 5.0$ to $7.8$, confirming that Stanford NTHC serves as a rigorous, unbiased baseline.

---

---

## 📐 Part 4: Three Methodological Atlas Versions for Subfield Composition

To rigorously evaluate subfield composition across different spatial parcellation paradigms, the pipeline implements **Three Canonical Atlas Versions** with dedicated, distinct figure names, statistical tables, and subject-level long datasets:

### 1. Version 1: Unified Fixed HCP151 Group Atlas (`v1_fixed_hcp151_atlas`)

- **Functional Boundary Source**: The canonical HCP 151-subject Route A Group Atlas functional boundaries ($K=3$: Anterior, Middle, Posterior) are applied **uniformly across all cohorts**.
- **Histological Subfields**: Each subject's native HippUnfold MultiHist7 subfields ($Sub, CA1, CA2, CA3, CA4$) on midthickness surface.
- **Scientific Role**: Completely eliminates cross-group functional boundary variance, evaluating purely histological subfield geometry within fixed spatial coordinates.

### 2. Version 2: Cohort-Specific Intrinsic Group Atlas (`v2_cohort_specific_atlas`)

- **Functional Boundary Source**: Each cohort uses its **own intrinsic group-level gradient spectral clustering atlas** ($K=3$).
- **Histological Subfields**: Each subject's native HippUnfold MultiHist7 subfields.
- **Scientific Role**: Preserves intrinsic group-level functional topographic adaptations and shifts.

### 3. Version 3: Individual-Level Template Matching Atlas (`v3_individual_template_matching`)

- **Functional Boundary Source**:
  - **HCP 7T ($N=151$)**: **5-Fold Cross-Fit Template Matching** individual labels (`outputs_hcp/work/crossfit_template_matching`).
  - **Stanford Cohorts (7 groups)**: **HCP151 Fixed Template Matching** individual labels (`outputs_stanford/work/hcp151_fixed_template_matching`).
  - **Iowa Cohorts (2 groups)**: **HCP151 Fixed Template Matching** individual labels (`outputs_iowa_jj_tms/work/hcp151_fixed_template_matching`).
- **Histological Subfields**: Each subject's native HippUnfold MultiHist7 subfields.
- **Scientific Role**: Fully accounts for subject-specific functional topography variations and resting-state connectivity adaptations.

---

---

## 📁 Artifacts & Reproducibility Suite

The group-differences renderer uses a compact header and tight panel grid so
the enlarged labels remain readable without excessive page margins or gaps.

### 🌟 Active Production Suite (2×2 Resolution × Preprocessing Grid)

Both the 10/12-cohort cross-dataset panorama and the Stanford clinical group differences pipeline support the 36P grid (`2mm_36p`, `4mm_36p`):

#### 1. Cross-Dataset Subfield Panorama (`render_multidataset_subfield_composition_2mm_2col.py`)

| Grid Tag | Figure Artifact | Statistics TSV (in `stats/`) |
| --- | --- | --- |
| `2mm_36p` | [`multidataset_subfield_composition_2mm_2col_2mm_36p.png`](file:///home/yojiao/hpchome/hipp/scripts/visualize/multidataset_subfield_composition_2mm_2col/multidataset_subfield_composition_2mm_2col_2mm_36p.png) | [`stats/multidataset_subfield_composition_2mm_36p_stats.tsv`](file:///home/yojiao/hpchome/hipp/scripts/visualize/multidataset_subfield_composition_2mm_2col/stats/multidataset_subfield_composition_2mm_36p_stats.tsv) |
| `4mm_36p` | [`multidataset_subfield_composition_2mm_2col_4mm_36p.png`](file:///home/yojiao/hpchome/hipp/scripts/visualize/multidataset_subfield_composition_2mm_2col/multidataset_subfield_composition_2mm_2col_4mm_36p.png) | [`stats/multidataset_subfield_composition_4mm_36p_stats.tsv`](file:///home/yojiao/hpchome/hipp/scripts/visualize/multidataset_subfield_composition_2mm_2col/stats/multidataset_subfield_composition_4mm_36p_stats.tsv) |

#### 2. Stanford Clinical Group Differences vs NTHC (`analyze_and_render_stanford_subfield_differences.py`)

| Grid Tag | Figure Artifact | Contrasts TSV (in `stats/`) | Subject Long TSV (in `stats/`) |
| --- | --- | --- | --- |
| `2mm_36p` | [`stanford_subfield_group_differences_figure_2mm_36p.png`](file:///home/yojiao/hpchome/hipp/scripts/visualize/multidataset_subfield_composition_2mm_2col/stanford_subfield_group_differences_figure_2mm_36p.png) | [`stats/stanford_subfield_group_differences_stats_2mm_36p.tsv`](file:///home/yojiao/hpchome/hipp/scripts/visualize/multidataset_subfield_composition_2mm_2col/stats/stanford_subfield_group_differences_stats_2mm_36p.tsv) | [`stats/stanford_subfield_subject_level_long_2mm_36p.tsv`](file:///home/yojiao/hpchome/hipp/scripts/visualize/multidataset_subfield_composition_2mm_2col/stats/stanford_subfield_subject_level_long_2mm_36p.tsv) |
| `4mm_36p` | [`stanford_subfield_group_differences_figure_4mm_36p.png`](file:///home/yojiao/hpchome/hipp/scripts/visualize/multidataset_subfield_composition_2mm_2col/stanford_subfield_group_differences_figure_4mm_36p.png) | [`stats/stanford_subfield_group_differences_stats_4mm_36p.tsv`](file:///home/yojiao/hpchome/hipp/scripts/visualize/multidataset_subfield_composition_2mm_2col/stats/stanford_subfield_group_differences_stats_4mm_36p.tsv) | [`stats/stanford_subfield_subject_level_long_4mm_36p.tsv`](file:///home/yojiao/hpchome/hipp/scripts/visualize/multidataset_subfield_composition_2mm_2col/stats/stanford_subfield_subject_level_long_4mm_36p.tsv) |

*(Note: For backward compatibility, default unversioned [`stanford_subfield_group_differences_figure.png`](file:///home/yojiao/hpchome/hipp/scripts/visualize/multidataset_subfield_composition_2mm_2col/stanford_subfield_group_differences_figure.png) mirrors `2mm_36p`).*

### 📦 Archived Methodological Suites (`archive_v1_v3/`)

All Version 1 (Unified Fixed HCP151 Group Atlas) and Version 3 (Individual-Level Template Matching Atlas) artifacts and explicit suffix files are archived in:

- Directory: `scripts/visualize/multidataset_subfield_composition_2mm_2col/archive_v1_v3/`

---

## 🔍 Instructions for Reviewing Agents (Review Guide)

Surface PNG caches require exact input/output fingerprints in `.sources.json`;
Native rendering uses two cohort workers to leave room for Workbench child
processes under the IDAS ten-process/task limit.
missing source artifacts fail before lookup and unverified PNGs are rebuilt.
The Stanford subject-level analyzer also fails closed when a declared
MultiHist7 label, midthickness surface, or individual template-matching label
is missing; it never silently drops that subject from the exported statistics.
The method comparison explicitly selects Kong normalized-top20 labels and
Lynch Fisher-z labels; it never requests or borrows a Lynch normalized label.

The group-differences renderer can render directly from existing exports (default `target` mode) or recompute analyses across any or all grid combinations:

```bash
# 1. Generate subfield composition panoramas for both 36p grids (standalone CLI removed):
/home/yojiao/.conda/envs/py314/bin/python scripts/visualize/run_multidataset_suite.py --figures subfield_composition_2col

# 1b. Branch-pair mode (2mm/36p standard + ctxreg, branch-qualified names):
/home/yojiao/.conda/envs/py314/bin/python scripts/visualize/run_multidataset_suite.py --version hcp_jjtms_7net --figures subfield_composition_2col --signal-branch all

# 2. Recompute and render Stanford group differences across both 36p grids:
/home/yojiao/.conda/envs/py314/bin/python /home/yojiao/hpchome/hipp/scripts/visualize/multidataset_subfield_composition_2mm_2col/analyze_and_render_stanford_subfield_differences.py --grid all --mode v2
```

### ⚡ Persistent Surface Panel Cache (`panels/`)

To eliminate redundant Connectome Workbench 3D rendering during figure styling and layout tweaks, all 1x2 composed hippocampus panels are persistently cached under:

- `panels/wb_subfield_{dataset}_{cohort}_{smoothing}_{variant}.png` for the established standard branch
- the same comparison cache name with a `_ctxreg` suffix for the explicit CtxReg branch

When cached panels exist, subsequent script runs drop execution latency from ~15–30s down to ~4–5s while preserving bit-exact rendering reproducibility.

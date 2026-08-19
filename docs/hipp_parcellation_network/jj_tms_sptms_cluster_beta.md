# JJ_TMS Native-space hippocampal surface beta

旧的 Standard/MNI-space beta 结果已归档。当前探索分支使用 native Float32 con
直接映射到同一被试的 HippUnfold 海马表面；它尚未晋升为编号正式流程或冻结结果。

## 分析定义

- 网络：`Control`、`Default`、`VentralAttention`，显示为 FPCN、DMN、VAN；Language 不进入本次三网络主分析。
- cohort：59 个通过 `visit2_data_eligibility == 1` 和 site/network 输入检查的被试。
- 海马：L/R 分开，永不生成 bilateral 行。
- parcellation：individual template matching 和 HCP151 fixed atlas。
- 测量：只执行 Native `volume → surface`；不执行 `surface → volume`。
- con：Native-space Float32 `baseline_spTMS con_0001`，不 GrayZ、不重采样。
- 表面：HippUnfold den-512 `space-T1w` inner/midthickness/outer。
- beta：L/R 海马 cluster 1/2/3 的面积加权均值。

## 输入与转换结果

Native con：

```text
outputs_iowa_jj_tms/inputs/iowa_jj_tms/spm_glm_output/
summary_NativeSpace_Float32/baseline_spTMS/
```

已完成的 Native volume-to-surface 结果：

```text
outputs_native_surface_comparison/
  jj_tms_native_standard_bad_compare_20260818/jj_tms/
```

该目录的 `native_jobs.csv` 有 59 行，mapping errors=0。Native 背景包含精确 0，
因此转换阶段保存 `finite` 和 `nonzero` 两种 ROI；当前探索提取使用 `nonzero`。

## 探索性 Native beta 提取

```bash
/opt/miniconda3/envs/py314/bin/python \
  scripts/exploratory_beta/run_native_surface_beta.py \
  --dataset jj_tms \
  --run-id jj_tms_native_surface_beta_20260818 \
  --roi-policy nonzero
```

输出：

```text
outputs_iowa_jj_tms/work/sptms_hipp_cluster_surface_beta_native/runs/
  jj_tms_native_surface_beta_20260818/
```

关键表为 `cohort_flow.csv`、`site_level_beta.csv`、`subject_network_beta.csv`、
`extraction_summary.json` 和 `run_manifest.json`。当前探索 beta 表只包含
`hippocampal_hemi = L/R`。

## 模型

```bash
Rscript scripts/exploratory_beta/jj_tms_sptms_native_surface_model.R \
  outputs_iowa_jj_tms/work/sptms_hipp_cluster_surface_beta_native/runs/jj_tms_native_surface_beta_20260818
```

template matching 和 HCP151 分别拟合 primary subject-network 与 site-level
sensitivity。模型为：

```text
beta_mean ~ site_network * cluster * hippocampal_hemi
          + target_hemi * cluster
          + target_hemi * hippocampal_hemi
random = ~1 | subject / hippocampal_hemi
```

三组 network pairwise contrasts 在每个 parcellation × source branch 内 Holm 校正。

## PNG 刷新

```bash
/opt/miniconda3/envs/py314/bin/python \
  scripts/exploratory_beta/render_jj_tms_five_figures.py \
  --run-dir outputs_iowa_jj_tms/work/sptms_hipp_cluster_surface_beta_native/runs/jj_tms_native_surface_beta_20260818
```

输出 `five_figures/` 中与 Stanford 相同 renderer 生成的五组 PNG/PDF：两组
network profile、within-network pairwise、FC-beta heatmap 和 HCP151 FC-beta
scatter。heatmap 的左列固定为左海马，右列固定为右海马；target hemisphere
先在 subject 内等权平均，避免重复加权。

## 旧结果归档

```text
outputs_iowa_jj_tms/archive/20260818_standard_space_beta_replaced_by_native/
```

该目录保存旧 Standard-space surface/volume/whole-hipp/laterality 结果、旧入口
代码和归档说明。归档结果不再用于当前探索模型比较；当前 Native 结果也不得在
未完成正式晋升前表述为冻结或可直接报告的正式结果。

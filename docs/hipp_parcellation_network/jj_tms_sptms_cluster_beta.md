# JJ_TMS Native-space hippocampal surface beta


Newly generated figures use CON for the Control Network; existing figures generated before this convention may show FPCN.
旧的 Standard/MNI-space beta 结果以及旧的 59 人 `all`/`healthy_1xxx` 运行已归档至 `wrong/outputs_iowa_jj_tms/archive/`。
当前分析基于总样本 94 人中合格的 78 名被试（55 名健康被试 `healthy`，23 名患者被试 `patient`），使用 Native Float32 con 直接映射到同一被试的 HippUnfold 海马表面，并在 Healthy 与 Patient 双组下分别进行三网络海马激活与功能连接相关分析。

## 分析定义

- 网络：`Control`、`Default`、`VentralAttention`，显示为 CON、DMN、VAN；Language（2 人）不进入本次三网络主分析。
- 被试（78 人合格）：
  - `healthy`（健康组，ID 1xxx）：**55 人**（Control 9, Default 34, VentralAttention 10, Language 2）
  - `patient`（患者组，ID 3xxx）：**23 人**（Control 12, Default 6, VentralAttention 5）
- 海马：L/R 分开，永不生成 bilateral 行。
- parcellation：individual template matching 和 HCP151 fixed atlas。
- 测量：只执行 Native `volume → surface`；不执行 `surface → volume`。
- con：Native-space Float32 `baseline_spTMS con_0001`，不 GrayZ、不重采样。
- 表面：HippUnfold den-512 `space-T1w` inner/midthickness/outer。
- beta：L/R 海马 cluster 1/2/3 的面积加权均值。

## 输入与转换结果

Native con：

```text
outputs_iowa_jj_tms/inputs/spm_glm_output/
summary_NativeSpace_Float32/baseline_spTMS/
```

已完成的 Native volume-to-surface 结果：

```text
outputs_native_surface_comparison/
  jj_tms_native_surface_beta_20260820_78sub/jj_tms/
```

该目录的 `native_jobs.csv` 有 78 行，mapping errors=0。Native 背景包含精确 0，
因此转换阶段保存 `finite` 和 `nonzero` 两种 ROI；当前探索提取使用 `nonzero`。

## 探索性 Native beta 提取

```bash
/home/yojiao/.conda/envs/py314/bin/python \
  scripts/exploratory_beta/run_native_surface_beta.py \
  --dataset jj_tms \
  --run-id jj_tms_native_surface_beta_20260820_78sub \
  --mapping-root outputs_native_surface_comparison/jj_tms_native_surface_beta_20260820_78sub/jj_tms \
  --out-root outputs_iowa_jj_tms/work/sptms_hipp_cluster_surface_beta_native \
  --roi-policy nonzero
```

输出：

```text
outputs_iowa_jj_tms/work/sptms_hipp_cluster_surface_beta_native/runs/
  jj_tms_native_surface_beta_20260820_78sub/
```

关键表为 `cohort_flow.csv`、`site_level_beta.csv`、`subject_network_beta.csv`、
`extraction_summary.json` 和 `run_manifest.json`。

## 统计模型

```bash
/home/yojiao/.conda/envs/py314/bin/python \
  scripts/exploratory_beta/fit_native_surface_mixed_models.py \
  --run-dir outputs_iowa_jj_tms/work/sptms_hipp_cluster_surface_beta_native/runs/jj_tms_native_surface_beta_20260820_78sub
```

`healthy` 与 `patient` 分组独立拟合，template matching 和 HCP151 分别拟合 primary subject-network 与 site-level sensitivity。
对于多目标侧组（patient），模型调整 target hemisphere；对于单目标侧组（healthy，全 L 侧），模型拟合标准海马侧向交互。三组 network pairwise contrasts 在每个 group × parcellation × source branch 内执行 Holm 校正。

## 五张图（Five Figures）生成

```bash
/home/yojiao/.conda/envs/py314/bin/python \
  scripts/exploratory_beta/render_jj_tms_five_figures.py \
  --run-dir outputs_iowa_jj_tms/work/sptms_hipp_cluster_surface_beta_native/runs/jj_tms_native_surface_beta_20260820_78sub
```

输出 `five_figures/` 中与 Stanford 相同 renderer 生成的五张 PNG；默认不生成 PDF，显式加入
`--pdf` 才会额外生成 PDF：
图中可见的数据集名称统一显示为 `Iowa 3T`；manifest 和输入表中的内部数据集标识仍为
`jj_tms`。
1. `three_network_healthy_profile_boxscatter.png`（三网络 HC 海马 Beta Profile；原始 55 名 HC 中，三网络图实际绘制 CON 9、DMN 34、VAN 10，Language 2 人排除）
2. `three_network_patient_profile_boxscatter.png`（患者组三网络海马 Beta Profile）
3. `surface_template_matching_hcp151_within_network_pairwise_8x4.png`（两组 × 两套模板 × 左右海马簇间配对检验）
4. `fc_beta_network_by_cluster_pearson_heatmap_4x2.png`（4×2 布局：Healthy/Patient × Template/HCP151 行，左/右海马列 FC–Beta 相关热图）
5. `fc_beta_hcp151_network_by_cluster_scatter_4x9.png`（4×9 布局：HCP151 下 FC–Beta 散点拟合图）

HC profile 图在每个 parcellation × 海马侧 × cluster 的三网络组内，以 VAN 为核心分别与
DMN 和 CON 做未校正 Welch 独立样本 t 检验（不做多重比较校正）。VAN 显著高于比较网络时，
在对应两根箱线/散点组上方画红色括号并标 `*`、`**`、`***`；VAN 显著低于比较网络时使用蓝色。
不显著时不画线、不标记。完整结果写入
`healthy_van_vs_fpcn_dmn_ttests.csv`，共 24 个比较（2 parcellations × 2 hemispheres × 3 clusters × 2 comparators）。

## 16 人 HCP151 fixed beta 大图

```bash
/home/yojiao/.conda/envs/py314/bin/python \
  scripts/exploratory_beta/render_jj_tms_beta_surface_bar.py
```

默认只输出 PNG；仅显式传入 `--pdf` 时额外输出 PDF。所有产物放在现有 run
的 `five_figures/`，前缀为 `jj_tms_hcp151_fixed_2mm_36p_beta_surface_bar_16sub`。

- HC（10 人）：1006、1007、1017、1020、1023、1035、1041、1044、1052、1054；
  TMS site network 均为 VAN。
- Pure MDD（6 人）：3010=DMN、3011=CON、3017=VAN、3018=DMN、3020=VAN、3023=CON。
- 3015 按本图指定范围排除，不进入任何 surface、bar 或 scatter。

使用原 `subject_network_beta.csv` 的 HCP151 fixed `beta_mean`，要求恰好
96 条 finite 记录。通过 `site_level_beta.csv` 的 `native_metric_path` 追溯
逐顶点数据；仅恢复缺失的选中侧 metric（native T1w den-512、ribbon-constrained、
nonzero ROI、voxel-subdiv=3），保存至 `five_figures/raw_metrics/`。不重跑整批
beta 或模型。所有 metric 的面积加权 subregion 均值必须与原表一致
（rtol=1e-5、atol=1e-6）；缺失 bad mask、输入或均值不一致均停止出图。

上半部为 16 个 subject cards，改为三行排列：前两行各 5 个 HC，最后一行 6 个
Pure MDD，减少列间空白并放大海马图。L/R cropped panels 先合并为 bilateral card，
再放入布局，避免单个海马被列间空白压小。整体画布和标签使用可读字号，散点统一
放大；sub-3017 与 sub-3020（两个 MDD VAN）共同使用菱形，另外 4 个 MDD 共同使用
圆形，不区分两个 VAN 被试彼此之间的身份。每个有效顶点按自身 raw beta 着色，共享对称 RdBu_r 色阶；
bad/nonfinite 顶点为灰色。Workbench display label 每个顶点单独编码，不用分区均值
填色。沿用 locked 1×2 视角及 HCP 2mm_36p 显示表面，通过 den-512 对应顶点显示；
实际测量仍是 native T1w den-512，未额外平滑或 2 mm 重采样。colorbar 缩窄置于
cards 下方。显示表面、native 表面、映射证据、卡片行布局、字号和来源哈希写入
figure manifest。

下半部 L/R 分开显示 Anterior/VAN、Middle/SMN、Posterior/DMN，HC 与 Pure MDD
并列柱，每点一名被试，误差线为组均值双侧 95% t CI（df=n-1）。该图只做描述性展示，
不再对 HC 与 Pure MDD 做统计检验，也不画两组比较括号或 `ns`。散点图区域的标题、
坐标轴、刻度和图例使用约 1.5 倍字号，以便审阅。另输出
`_selected_beta.csv`、`_group_summary.csv`、`_manifest.json`，并将放大的图例单独放在
 x 轴刻度下方，底部说明再单独占一行；
右侧面板的 y 轴标题放在右图外侧，避免侵入左侧面板；同时保留
`surface_cards/` 中的 scenes 和原始渲染图供审计。图面不显示 mask 状态或
display preset 等技术细节，这些信息只保留在 figure manifest。

## 旧结果归档

```text
wrong/outputs_iowa_jj_tms/archive/20260820_59sub_beta_and_legacy_runs/
```

该目录保存旧版 59 人的 Standard-space 与 Native-space 运行记录、旧分组模型与历史图表。

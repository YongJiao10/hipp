# Stanford/JJ_TMS 多尺度海马探索实验

> **当前统一海马分支与 Kember 规范：** 边界见 [docs/analysis_map.md](docs/analysis_map.md)，
> Kember 细节见 [docs/kember_area_method.md](docs/kember_area_method.md)，重建记录见 [docs/restructure_record.md](docs/restructure_record.md)。subject-level FC
> 的方法严格为 `hcp_template_matching` 与 `hcp_fixed`，统一海马分支另登记
> `group_atlas_gradient_spectral` 几何 family。functional segregation
> 和 boundary sharpness 使用 512-source BOLD、labels 与 source-512 midthickness geometry；
> system surface area 单独使用 native HippUnfold `surfareacorobl` 逐顶点 mm² metric。
> 六个 contrast 的 method-aware 设计共 432 个 endpoint rows。

Task 1 freezes the written method decision and workflow boundaries in
[docs/kember_area_method.md](docs/kember_area_method.md),
[docs/analysis_map.md](docs/analysis_map.md), and
[docs/restructure_record.md](docs/restructure_record.md). The sole area source
is `area_source = hippunfold_native_surfareacorobl`; absolute native mm² is the
Kember-aligned primary endpoint and surface-area fraction is secondary context.


本目录是一个与现有正式流程隔离的研究设计与后续探索实现空间。现有
`outputs_stanford/`、`outputs_iowa_jj_tms/`、`outputs_hcp/` 和 `refs/` 只作为只读
输入；新代码、缓存、统计结果和图表必须留在本目录内。这里不使用 worktree、不读取任何 archive 内容；当前结果可直接覆盖写回，输入路径仍保持只读。
默认不做 HDR。

## 当前 Rest-FC 分析权威

最初第 1、2 项的正确执行范围见 `rest_fc_primary_analysis_spec.md`。两点都以 anterior/middle/posterior
对应的 17-network/8-network Fisher-z FC profile 为结局：第 1 项比较 JJTMS
patient-vs-healthy 及 Stanford symptomatic-vs-healthy；第 2 项在 Stanford 四组中分解
marginal symptomatic、marginal trauma、interaction 和四个预设 simple contrasts。
逐network定位是第1点的核心结果；anterior/middle/posterior及whole-hippocampus向量检验仅作
辅助总览，不作为定位门。保留bilateral、left、right和配对L−R偏侧化；不批量遍历
network-pair contrasts，只在定位后登记少量post-selection候选供后续验证。

`cross_dataset_rest_fc/` 是当前 Rest-FC 主工作流，负责跨数据集网络定位和 Stanford 四组
Fisher-z FC 分解，并通过同一 branch ledger 调度 group-atlas geometry；atlas 几何行与
subject-level FC 行保留不同的 `analysis_family`，不混入 FC 推断。

统一 Kember workflow 位于 `kember_metrics/`。`feature_build.py` 在一个
subject-level feature table 中写入 `hcp_template_matching` 与 `hcp_fixed`
两种 method；`contrasts.py` 以 method 为显式轴，在一个 direct-contrast table 中完成
六个 case−NTHC Welch 比较。每个 hemisphere 的三个 system 各有三项 Kember 文章
指标：system surface area、functional segregation、boundary sharpness。

它不含 transition、几何 boundary position、vertex fraction 或 AP centroid。
surface-area fraction 仅保留为归一化上下文，不作为第四个 direct endpoint。固定比较
覆盖六个 contrast、两种 method、2/4 mm、L/R、三类 metric 和三个 system，共
432 个 endpoint units。结果表保留 method 列。

其中 `TIS PTSD` 是现有 47 人 TIS QC roster 与 `data/CC_RC_Master_Database_10.23.18_Data.csv` 中 `Current_PTSD_Diagnosis == 1` 的交集，按 `CC_Post_Intake_ID` 连接得到 35 人。清单为 `outputs_stanford/inputs/kong2022/manifests/stanford_tis_ptsd_4xxx_current_ptsd_qc_pass_t1only_35.txt`，稳定报告中的 `tis_ptsd_provenance.json` 记录连接规则和输入哈希。该组仅用于 nested direct workflow，不改变正式 NTHC/TEHC/NTS/TIS 四组 factorial 设计。

稳定目标位于 `kember_metrics/work/features/`、
`kember_metrics/results/direct_contrasts/`、
`kember_metrics/figures/direct_contrasts/` 和
`kember_metrics/reports/direct_contrasts/`；四个目录共同描述一次当前分析，发现问题时直接覆盖修正。

## 当前三工作流

这项研究只有以下三个 active workflows；它们共享经过审计的输入，但 estimand 和输出不
互换：

1. `cross_dataset_rest_fc`：回答 cortex ↔ hippocampal-system Fisher-z FC 的定位、
   JJTMS 组差异和 Stanford `trauma × symptomatic` factorial 问题。
2. `kember_metrics`：以 `hcp_template_matching` 与 `hcp_fixed` 两种方法计算
   subject-level system surface area、functional segregation 和 boundary sharpness。
   功能指标保留 512-source geometry；只有面积消费 native `surfareacorobl`。
3. `group_atlas_gradient_spectral`（代码位于 `external_group_atlas_analysis`）：描述
   HC 加七个 parent/nested cohort atlas 的 group-level geometry、label agreement 和
   boundary geometry；它与 Rest-FC 共用 branch ledger，但不向 Kember subject-level
   feature table 或 FC subject rows 回填标签或面积。

Kember 没有第三种 subject-level method，也没有所谓“边界面积”端点。其六个 contrast
在 2 methods × 2 smoothing × 2 hemispheres × 3 metrics × 3 systems 上产生 432 个
method-aware endpoint rows。

corrected 第 1/2 项的可执行实现位于 `cross_dataset_rest_fc/`。逐 network
localisation 是主结果；profile permutation 是辅助、非门控结果。portable builder 可用时，HTML 只呈现紧凑的
主定位视图以满足官方 portable verifier，完整方法、限制、全表和五张静态图保存在同一
run 的 `technical_report.md`、`results/` 与 `figures/` 中。

Portable HTML is optional: when the report builder is unavailable, the run still
completes and records the reason in `builder_receipt.json`; the technical
Markdown report, static figures, tables, and SHA-256 manifest remain authoritative.

## 2026-08-24 权威 Rest-FC 运行与核心结果

当前权威统计输出是
`cross_dataset_rest_fc/*/primary_screen/`。其统一 branch ledger 可写成：
`JJTMS/Stanford × 8/17net × bilateral/L/R/L-R × anterior/middle/posterior × 2/4mm ×
hcp_fixed/template_matching × network-wise/profile`。
Stanford 的四个 parent factorial map、三个 nested clinical cohort map 与 HC reference
map 也登记在同一 ledger 中；它们的 geometry 表不回填 subject-level FC 数值，
但当前 run manifest 统一记录这些 atlas 分支。

- 样本闭合为 JJTMS 67 healthy + 27 patient，Stanford NTHC 51 + TEHC 40 +
  NTS 56 + TIS 47。fixed/template 的所有主模型分别使用完整 n=94 与 n=194；
  没有 zero imputation。
- 最清楚的结果是 Stanford `tail–Language` FC：`TIS-TEHC` 在全部 32 个
  fixed/template × 2/4mm × bilateral/L/R/L-R × 8/17net 规格中方向均为正；
  最强规格为 template matching、2mm、left、8net，b=0.0332，95% CI
  [0.0185, 0.0480]。left 最强，但 L-R 仍需在最终分析对象集合冻结并确定矫正范围后
  再判断，当前只能作为方向稳定的候选。
- Stanford symptom marginal 的 `tail–Language` 同样 32/32 方向为正；四组均值显示 TIS 接近 0，而
  NTHC、NTS、TEHC 更负，尤其 TIS 与 TEHC 分离。正式 interaction 的 tail-Language
  方向也为 32/32 正；这支持一个可能的 symptom × trauma 形状，但在矫正范围冻结前
  不作显著性结论。
- Stanford `TIS-NTS`、`NTS-NTHC` 和 `TEHC-NTHC` 的定位仍应按效应量、
  方向稳定性和方法一致性解释；anterior VisualB 只在 template matching 中出现，
  是方法特异的次级线索。
- JJTMS 的 `tail–Auditory` 在 fixed、2mm、left、8net 下为负
  (b=-0.0142)，但 template matching 不支持，Stanford 对齐分支也未复制，故只能
  作为脆弱探索信号。

## Effect-first 广义 promising 复核

`cross_dataset_rest_fc/*/broad_signal_review/` 是基于当前 primary_screen 表格的
可视化派生分析；它没有重拟合模型，发现问题时直接覆盖当前派生输出。该复核不使用
临时 q 值，优先展示效应量、方向稳定性、四组形状、跨数据集同向性和
L−R 偏侧化。主图包括 22 个方向稳定的 level-effect 候选、6 个 JJTMS/Stanford
同向 network-region 组合和 12 个偏侧化候选。其中 20 个来自预设的效应量层，另保留
2 个效应较小但方向稳定的 secondary leads。相关的 12 个 level 规格或 4 个
L−R 规格用于稳健性摘要，不能被当作独立重复。

除 posterior/tail–Language 外，当前值得后续验证的候选还包括：Stanford body–
Somatomotor 的 crossover-like interaction；JJTMS patient−healthy 与 Stanford
symptom marginal 同向的 body–Default 增强；TEHC 相对 NTHC 的 body–Somatomotor
增强与 body–SalVenAttnB 降低；以及 tail–VisualC、body–Language 的 L−R 模式。
这些是 discovery candidates，不是通过完整多重比较后的确认性发现。

## Stanford HC-referenced group atlas：统一运行中的几何 family

统一运行中的 group-atlas geometry family 覆盖七个 Stanford cohort：四个 parent factorial groups
（NTHC、TEHC、NTS、TIS）和三个 nested clinical cohorts（NTS Any MDD、NTS Pure
MDD、TIS PTSD）。四个 parent 仍是正式的 trauma × symptom factorial 设计；三个
nested cohort 是可重叠的临床子集，不是新的互斥 factorial cells。

HC 加七个 cohort × plain/weighted × 2/4mm × L/R = 64 个 K=3 atlas cells，均由 manifest
门验证后纳入 `cross_dataset_rest_fc/*/primary_screen/` 的统一 branch ledger。该
geometry family 计算 anterior/middle/posterior 面积与比例、每个非 HC 组对 HC 的 7 个 matched
pair 的 ARI/ordered Dice、1–2/2–3 边界 Dice 及 mesh-geodesic 边界距离；不生成非 HC
组之间的全配对，也没有个体层不确定性、p/q、split-half 或 ICC。

面积字段使用 `atlas_region_area_*` 命名空间，表示 triangle-area/3 的 atlas geometry
汇总；Kember 的唯一面积端点仍是 `kember_metrics` 中 native HippUnfold
`surfareacorobl` 的逐顶点求和。两者不互换。

严禁让每个 cohort 用自己训练的 atlas 提取 FC 后再直接比较组别；那会把组别信息写入
特征定义，造成循环推断。外部 atlas 只用于解释 FC 改变是否伴随 atlas 边界/面积
重组，不重新定义主 FC 分析。

该 family 的 `descriptive_contrasts.csv` 只保留七个非 HC 组对 HC 的 descriptive-only
summaries；Stanford subject-level FC 的 factorial interaction 仍由 FC 模型报告，不在
group-map 表中重复。统一输出位于 `cross_dataset_rest_fc/results/primary_screen/group_atlas/`。

## 文档

- `study_spec.md`：科学问题、假设层级、端点、统计模型、验证与论文路线。
- `docs/restructure_record.md`：结构重建、方法决策、输出闭合与清理记录。
- `literature_and_kember_method_notes.md`：Kember 方法及相关原始文献证据。

## 已确认的本地证据

- Stanford 静息态 QC 样本为 194 人：NTHC 51、TEHC 40、NTS 56、TIS 47。
- 现有 Stanford HC（NTHC+TEHC，n=91）与 symptomatic（NTS+TIS，n=103）结果提示
  `Auditory − VentralAttention` 相对连接在 anterior 和 middle 均向 symptomatic
  方向增加，并在 2/4 mm、普通和 weighted-Fisher-z 分支方向一致；它尚未回答
  `trauma × symptomatic` 交互。
- Stanford HCP 个体 template-matching 运行清单显示 194/194 complete。

## 执行边界

在开始实现前冻结 `analysis_spec.yaml`、被试/表型 manifest 的 SHA-256、主端点和对比。
发现层可以广，但每个结果必须标记为 `primary`、`secondary` 或 `discovery`。高水平论文
的判断门是效应稳定性、独立或外部验证和可解释机制，不是显著结果数量。

## 当前 Kember method 规格（2026-09-01）

[当前规范](docs/kember_area_method.md) 只保留两个 Kember methods：
`hcp_template_matching` 与 `hcp_fixed`。两者的 functional segregation 和
boundary sharpness 都在 consumed 512-source time series、labels 与 midthickness
geometry 上计算；标签经 `ADAP_BARY_AREA` 对齐到 native 网格后，system surface area
才使用 native HippUnfold `surfareacorobl` 求和。六个 contrast 的完整 method-aware
结果为 432 rows。Group-specific clustering 不是 Kember 的第三个 subject-level 方法；
它作为 `group_atlas_gradient_spectral` geometry family 纳入 Rest-FC 的统一优先级调度。

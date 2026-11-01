# Stanford/JJ_TMS 海马—皮层网络、四组交互与拓扑特化研究规格

> **状态说明（2026-08-23）：** 最初第1、2项的当前rest-only执行定义已由
> `rest_fc_primary_analysis_spec.md` 收紧：两点均以3-region × 17/8-network Fisher-z
> FC profile为结局，并必须包含JJTMS。Kember 拓扑内容只作为第 3 项扩展参考，
> 不得用来把面积或边界结果替代第 2 项的 Stanford FC 四组分析。

> **工作流规格（2026-09-02）：** 当前权威边界见
> [docs/kember_area_method.md](docs/kember_area_method.md) 与
> [docs/analysis_map.md](docs/analysis_map.md)。`cross_dataset_rest_fc` 是统一
> `primary_screen` 编排器：subject-level FC 分支与 Stanford group-atlas geometry
> family 共用 branch manifest 和严格优先级 barrier。`external_group_atlas_analysis`
> 保留为实现/报告兼容入口，不再是独立实验流程；Kember 严格只有
> `hcp151_template_matching` 与 `hcp151_fixed` 两种方法；**禁止个体聚类**。

## 1. 核心研究架构

### 1.1 第 1 项和第 2 项的区别

“cortical—海马 subregion network 组间差异”定义**分析的因变量**：每名被试、每侧
海马、每个 anterior/middle/posterior system 与每个个体化皮层网络之间的 Fisher-z FC
profile。

“Stanford 四组海马差异和 symptom × trauma interaction”定义**解释变量和对比结构**：

| Stanford 组 | Trauma `T` | Symptomatic `S` | 当前 QC n |
|---|---:|---:|---:|
| NTHC | 0 | 0 | 51 |
| TEHC | 1 | 0 | 40 |
| NTS | 0 | 1 | 56 |
| TIS | 1 | 1 | 47 |

因此第 1 项定义跨数据集 FC 表型，第 2 项用 Stanford 四组设计对同一 FC profile
分解 symptom、trauma、interaction 和四个 simple contrasts。面积、边界与 Kember-style
指标属于第 3 项拓扑扩展；其 group-atlas geometry 分支与 FC 行保持数据类型分离，
但由同一 branch ledger 调度。不能把 collapsed healthy-vs-patient
检验当成创伤效应，也不能由四组任意 pairwise p 值反推交互。

二元因子主模型为：

```text
Y = beta0 + betaT*T + betaS*S + betaTS*(T*S) + covariates + error
```

核心可解释对比固定为：

```text
non-trauma symptom effect = NTS - NTHC
trauma symptom effect     = TIS - TEHC
healthy trauma effect     = TEHC - NTHC
symptomatic trauma effect = TIS - NTS
interaction               = (TIS - TEHC) - (NTS - NTHC)
```

若四组使用同一把连续 symptom 量表且缺失与地板效应通过门控，再拟合
`Y ~ trauma * z_symptom_score`。若健康组量表结构性缺失、量程不同或几乎全为零，只报告
二元 `T × S` 模型及患者内 severity association，不能把它包装成全样本连续交互。

### 1.2 三个 active workflows 不是“三次复制”

| Workflow | 实际定义 | 研究角色 | 主要限制 |
|---|---|---|---|
| `cross_dataset_rest_fc` | shared-label subject-level Fisher-z FC + unified group-atlas branch ledger | 跨数据集定位与 Stanford factorial inference；统一编排 geometry family | FC 行不回答面积，geometry 行不混入 FC 数值推断 |
| `kember_metrics` | `hcp151_template_matching` 与 `hcp151_fixed` 的 subject-level Kember features | area、segregation、sharpness | 功能指标用 source-512；面积用 native metric |
| `external_group_atlas_analysis` | unified group-atlas geometry family 的实现/报告兼容表面 | HC-referenced group-level geometry 与 agreement | 不独立调度，不回填 Kember subject-level inference |

统计声明遵循下列规则：

- Kember 的活跃方法严格只有 `hcp151_template_matching` 与 `hcp151_fixed`；没有第三种
  subject-level clustering method。
- functional segregation 与 boundary sharpness 使用 consumed 512-source BOLD、labels
  与 source-512 midthickness geometry；system surface area 使用 native HippUnfold
  `surfareacorobl` 与对齐后的 native labels。
- 六个 case−NTHC contrast × 2 methods × 2 smoothing × 2 hemispheres × 3 metrics ×
  3 systems = 432 个 method-aware endpoint rows。
- 2 mm 与 4 mm 是相关规格，不是独立重复实验，也不以多个 p<.05 计票。
- consumed surface/CIFTI 与 density_evidence 是密度真值；命令行参数不是密度证据。


## 2. 当前证据与未回答问题

### 2.1 Stanford 静息态已观察到的信号

现有分析把 NTHC+TEHC 合并为 HC（n=91），把 NTS+TIS 合并为 symptomatic（n=103）。
使用固定 HCP labels 提取 subject-level FC 后，普通 Fisher-z 分支的 symptomatic-minus-HC
`Auditory − VentralAttention` 为：

| smoothing | system | Hedges g | permutation p | 解释 |
|---|---|---:|---:|---|
| 2 mm | anterior/head | 0.393 | 0.0061 | Auditory 较高且 VAN 较低的相对位移 |
| 2 mm | body | 0.586 | <0.0001 | 同方向、效应更强 |
| 4 mm | anterior/head | 0.394 | 0.0077 | 与 2 mm 一致 |
| 4 mm | body | 0.596 | <0.0001 | 与 2 mm 一致 |

weighted-Fisher-z 分支方向一致，head g≈0.38、body g≈0.56。当前结果支持的是一个
**相对 profile shift**；Auditory 和 VAN 的单独分量经相应多重校正后并不能被自动称为各自
显著。group-atlas 的 anterior dominant-network identity 有变化，但跨半球/加权方式并非
完全稳定；body 则主要是连续 profile shift，不是 dominant network 变类。

上述结果来自同一个 Stanford 样本，已经是 discovery evidence。后续四组分解能说明该
信号由症状、创伤还是交互驱动，但不能把同一样本中的显著重复称为独立 replication。

## 3. Analysis Aims 与假设层级

### Aim 1：跨数据集定位 hippocampal region × cortical network FC

每个 subject × hemisphere × anterior/middle/posterior × 17/8-network 的 Fisher-z FC
是共同结局。JJTMS 估计 patient−healthy；Stanford 估计 symptomatic−healthy marginal
effect。逐 network localization 是核心发现层，按固定
dataset × estimand × parcellation × resolution × hemisphere mode × smoothing ×
FC-estimator 规格保留三个 region × 全部 network 的 raw p 和效应量；最终
analysis-object set 冻结前不定义或应用 BH-FDR。

anterior/middle/posterior 各自的网络向量 global test 和 3×network whole-hippocampus profile
只作辅助总览，不能门控逐 network 结果。任何 network-pair contrast 都必须先由绝对
network 结果定位后再登记为下一阶段候选，不能在发现样本中批量遍历。

### Aim 2：用 Stanford 四组分解同一 Fisher-z FC profile

同一 subject-level FC tensor 用四 cell model 估计：

1. marginal symptomatic effect；
2. marginal trauma effect；
3. symptom × trauma interaction；
4. NTS−NTHC、TIS−TEHC、TEHC−NTHC、TIS−NTS 四个 simple contrasts。

bilateral、L、R 和 paired L−R 全部保留；L−R 检验组间偏侧化差异。四组
omnibus/profile 仍是辅助描述，不是逐 network localization 的统计门。

### Aim 3：独立的 Kember-style 拓扑扩展

本项目的面积方法已冻结为 `area_source = hippunfold_native_surfareacorobl`，具体
字段、证据和三个工作流边界见 [docs/kember_area_method.md](docs/kember_area_method.md)
与 [docs/analysis_map.md](docs/analysis_map.md)。Kember 对齐的主端点是 native
HippUnfold `surfareacorobl` 按 system label 求和得到的绝对 `surface_area_mm2`；
`surface_area_fraction = surface_area_mm2 / total_valid_surface_area_mm2` 仅作
secondary context。geometry-derived triangle-area/3 reconstruction 只允许作为 geometric-only QC，
`transition_area` 不是 Kember endpoint。

Kember 的三个量必须分开：

1. **Surface area**：每个 system 内 midthickness vertex area 的和（mm²）及其比例；
2. **Segregation**：system 内外相关分布的 Cohen’s d；
3. **Boundary sharpness**：边界局部内外相关分布的 Cohen’s d，不是边界面积。

本项目可另外描述 boundary position/length、AP centroid 和 template similarity margin，
但必须标为扩展指标。当前 `kember_metrics` 只使用 `hcp151_template_matching` 与
`hcp151_fixed`；`group_atlas_gradient_spectral` 的实现仍位于
`external_group_atlas_analysis`，但由 `cross_dataset_rest_fc` 的统一 branch ledger

这一扩展先做 promising-first 描述和基础组间模型；只有出现清晰信号后，再投入
split-half、ICC、2/4/6 mm 邻域或空间 null 等严谨化工作。

## 4. 数据、特征与 QC 合约

### 4.1 冻结队列

Stanford 主静息态队列使用现有 194-person QC roster，并把四个组 manifest 的并集、互斥性
和 ID→T/S 映射写入新的 hashed subject table。`stanford_all_subjects.txt` 有 199 行而主
template-matching roster 为 194；多出的 5 名只可在固定 HCP151 sensitivity 中出现，并须
在 flow diagram 明示，不能让不同方法的 n 静默变化。

JJTMS rest-FC 使用冻结的 94-person template-matching roster（67 healthy、27 patient）。
外部方向比较必须按 endpoint 给出实际 n，不使用“94 人数据”作为笼统分母。

### 4.2 表型最小字典

被试表必须至少含：

```text
subject, trauma_binary, symptomatic_binary, four_group,
age, sex, acquisition_site, mean_fd, usable_minutes
```

连续 symptom 分析额外要求：

```text
symptom_scale_name, symptom_total, assessment_interval_days
```

可选敏感性变量包括 medication、diagnosis subtype、comorbidity 和 intracranial volume。
表型来源、单位、编码、缺失原因和日期必须进入 data dictionary。影像 outcome 不做插补；
核心协变量 complete-case 与多重插补敏感性并列，且不把组别后果变量随意作为主模型协变量。

### 4.3 fMRI 与 surface QC

- 只消费已有正式 rest preprocessing 和 provenance manifest；不得混合 GSR/non-GSR、不同
  space 或不同 run concatenation contract。
- 主分析要求至少 10 分钟 low-motion data；同时把 usable minutes 连续调整，并做 ≥15 分钟
  sensitivity。
- 记录 mean FD、DVARS/可用则记录、censored frames、surface invalid area、每侧实际 vertex
  count 和 `density_evidence`。
- 检查组间 QC 缺失率；若 T/S 预测 exclusion，用 inverse-probability weighting 作为
  sensitivity，不把其替代主分析。
- cortex-near-hippocampus signal leakage 以 20 mm local-cortex exclusion (`ctxdrop`) 做敏感性。

### 4.4 网络 FC 定义

- 个体 cortical atlas：Kong2022 400 ROI，固定合并为 8 network family。
- 主 FC：先对每个 hippocampal system 和 cortical network 求面积/有效顶点加权 mean time
  series，再 Pearson r、clip、Fisher z；这对应当前 weighted-Fisher-z 逻辑。
- 敏感性 FC：vertex×ROI Fisher-z 先平均的现有普通分支。
- 每个相对 contrast 同时保存两个绝对分量；禁止只报告差值而不展示组成。
- ROI/network 缺失不以零填充；输出 finite-feature 数和 network coverage。

## 5. 统计策略

### 5.1 逐 network 主定位

JJTMS 使用 healthy/patient cell-means model；Stanford 使用四组 cell-means model 与冻结
contrast matrix。每行报告 mean difference、standardized effect、HC3 95% CI 和 raw p；
q_bh 在 analysis-object set 冻结前保持为空。缺失 network 不以零填充。

最终 analysis-object set 冻结后，才在预先记录的 dataset × estimand × parcellation ×
8/17-network resolution × hemisphere mode × smoothing × FC-estimator 规格中确定
correction family。左右、2/4 mm 与 fixed/template-matching 标签来源是相关规格，不计为独立重复；
group-atlas geometry 行不进入该 FC family，但其分支仍受统一 priority barrier 管理。

### 5.2 Profile 辅助检验

anterior/middle/posterior 的 8/17 维向量与 whole-hippocampus 的 3×8/3×17 拼接向量使用
subject-level permutation；一个被试的相关分量整体置换。profile p 值不作 confirmatory
multiplicity claim，也不决定是否报告逐 network localization。

### 5.3 Stanford 四组与协变量

Stanford 同一四 cell model 输出两项 marginal effect、interaction 和四项 simple contrasts。
当前权威运行是 unadjusted promising-first screen。age、sex、site、motion 与 usable minutes
只在发现 promising 信号后进入敏感性模型；它们不阻止基础结果查看。

连续 symptom × trauma 仅在四组共享同一量表、量程与 common support 时拟合；否则分开
报告二元 factorial model 与患者内 severity association。

### 5.4 Kember 拓扑扩展

面积比例按 compositional outcome 处理，同时报告绝对 mm² 与 total hippocampal area。
`hcp151_template_matching` 与 `hcp151_fixed` 的结果按 method 分开解释。group atlas
是登记在 `cross_dataset_rest_fc` 统一 branch ledger 下的 geometry family；
`external_group_atlas_analysis` 只保留实现/报告兼容入口。它只提供群体描述，没有
个体层 p/q，因此不能替代 subject-level Kember inference。

该 atlas 注册表包含显式 HC reference、四个 parent factorial groups（NTHC、TEHC、NTS、
TIS）和三个 nested clinical cohorts（NTS Any MDD、NTS Pure MDD、TIS PTSD），合计
8 × 2 assignment variants × 2 smoothing levels × 2 hemispheres = 64 个 atlas cells。
group-wise 只输出每个非 HC 组对 HC 的 7 个 matched pairs，不输出非 HC all-pairs。
nested cohorts 可与 parent groups 重叠，不新增互斥的 FC factorial cell。

若拓扑结果 promising，再增加 bootstrap、协变量、split-half/ICC、邻域敏感性和空间 null；
这些验证不是首轮结果查看的启动门。

## 6. 预设图表与可发表结果包

### Figure 1：队列、统一工作流与证据层级

四个 parent factorial group 与三个 nested cohort 的 flow diagram、实际 n、`cross_dataset_rest_fc`、`kember_metrics`、
`cross_dataset_rest_fc` 内 HC-referenced group-atlas geometry family 的角色、Kember 两方法与分析门。

### Figure 2：四组 cortex—hippocampus network profile

每组 3×8 effect heatmap、P1/P1-components forest plot、T/S/T×S contrast plot；左右主图合并，
hemisphere 置于分面/补充材料。

### Figure 3：统一运行中的 HC-referenced group-atlas 描述

HC 加七个 cohort 的 anterior/middle/posterior probability maps、group-level area composition、
agreement 和 boundary geometry；明确标记为统一运行中的 descriptive geometry family，
不作 subject-level Kember 推断。

### Figure 4：Kember-style specialization

surface area、segregation、boundary sharpness 的定义示意、reliability、parent 与 nested direct effects，
明确标记 den-512 adaptation。

### Figure 5：跨方法与跨数据验证

fixed/template-matching effect concordance、2/4 mm、weighted/unweighted、
leave-one-site-out 和 JJTMS patient-vs-healthy 方向比较。


所有图对应 machine-readable table、effect/CI、n、analysis tier、parcellation、smoothing、
source fingerprint 和 exact contrast。

## 7. 论文与 funding 决策树

### 路线 A：症状相关 network-profile reconfiguration

成立条件：P1 的症状效应稳定，四组分解显示 NTS 与 TIS 均有相似方向，T×S 较弱；body
且 fixed/template-matching 方向一致。这是 transdiagnostic symptom organization story，不应误称为
trauma-specific。

### 路线 B：trauma-specific hippocampal topographic specialization

成立条件：T×S 在 Kember-style area/segregation 或可靠 boundary metric 上稳定，TIS−TEHC 明显且 NTS−NTHC 不同，
并能排除 anatomy/motion/site。它更适合创伤机制论文与后续独立验证。


### 停止/降级规则

- 只有 group-specific atlas 有差异、外部模板无差异：降级为 atlas construction artifact。
- 结果只在一种分区或任意阈值中出现：降级为 method-specific exploratory result。
- 效应由单一 site、motion、scan length 或少数被试驱动：不进入主论文 claim。

## 8. 产物与可重复性要求

未来所有文件位于本实验目录：

```text
config/       frozen specs, schemas, contrast registry
inputs/       hashes and de-identified phenotype/subject manifests only
src/          experiment-specific code
tests/        unit, contract, and synthetic-statistics tests
work/         rebuildable feature caches
results/      immutable run-id statistical tables/manifests
figures/      figure run-id products
reports/      claims table, methods, and manuscript-ready summaries
logs/         bounded execution logs
```

不复制大型原始数据。每次 run manifest 记录 Git commit、配置 hash、输入 hash、解释器、随机
种子、并发数（≤10）、density evidence、成功/排除数和输出 hash。任何影响行为、路径、缓存、
环境或工作流输出的代码变更，都在同一 change 更新本目录 README/方法文档；若引入新依赖，
同步更新 repository environment manifest。默认不引入新依赖，使用
`/home/yojiao/.conda/envs/py314/bin/python` 的 NumPy/Pandas/SciPy/statsmodels/nibabel/pytest。

## 9. 当前执行顺序

1. 以 `cross_dataset_rest_fc/*/primary_screen/` 作为第 1、2 项的
   权威 unadjusted promising-first FC 运行，修正时直接覆盖现行表并同步清单。
2. 先围绕 tail–Language 等已定位信号补 coverage、年龄/性别/运动/usable-minutes
   敏感性与影响点检查。
3. Stanford HC-referenced group atlas 由 `cross_dataset_rest_fc` 的统一 branch ledger 调度：
   2 mm atlas cells 属 P0，4 mm atlas cells 属 P2；其 geometry 表与 subject-level FC
   数值行分开保存。严格先完成全部高优先级分支，再进入低优先级分支。
4. 若继续 Kember 项，只采用 `hcp151_template_matching` 与 `hcp151_fixed` 两种方法，
   保持 source-512 functional geometry 与 native `surfareacorobl` area 分离，并对六个
   contrast 输出 432 个 method-aware endpoint rows；只有结果 promising 后才启动更重的
   可靠性验证。
5. 新的验证数据集、estimand、region、hemisphere、resolution 和 method 必须在运行前
   在运行前明确冻结，不新增本地版本号；修正时直接更新对应 current 输出。

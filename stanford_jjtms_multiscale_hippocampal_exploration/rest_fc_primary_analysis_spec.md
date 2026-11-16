# JJTMS + Stanford 海马—皮层 Rest-FC 基础分析规格

## Authority

本文件是后续执行最初第 1、2 项分析的当前科学范围。它只涉及静息态
Fisher-z functional connectivity，不涉及 beta、TMS response、con image、
Native stimulation mapping 或任何 archive 内容。

本文件规范 `cross_dataset_rest_fc/` 的 Rest-FC 主工作流及其统一的
group-atlas geometry 分支。Kember `kember_metrics/` 仍是独立的 Kember
subject-level endpoint workflow；group atlas 与 FC 共用 branch ledger 和
priority barrier，但不向 subject-level FC 结果回填面积、边界或
group-specific 标签。
本文件不再维护已经退出的旧分支及其结果副本；当前有效路径和输出以各自
README、manifest 与 [docs/analysis_map.md](docs/analysis_map.md) 为准。

## 1. 两个问题的正确关系

### 1.1 第 1 项：跨数据集海马亚区—皮层网络 profile

分析表型是每名被试每侧海马 anterior、middle、posterior 三个 region 各自对应的
皮层网络 Fisher-z FC 向量：

现有机器字段中的 `posterior` 与显示名称 `tail` 是同一个region，不形成额外分支。

- 17-network resolution：每侧 `3 × 17`；
- 8-network resolution：每侧 `3 × 8`；
- 若主分析双侧面积加权平均，则每名被试分别为 51 维和 24 维；
- 左右分开结果作为明确标记的 hemisphere-specific 分析，不与双侧结果计作独立重复。

必须分析：

- JJTMS：patient vs healthy；
- Stanford：symptomatic/patient vs healthy 的 marginal effect。

17-network 与 8-network 都必须生成并保留；它们是不同分辨率的同一表型，不作“三次投票”。
在最终 analysis-object set 冻结前不定义或应用 multiplicity family，也不得以任一
分辨率的 raw p 值作为成功规则。

### 1.2 第 2 项：Stanford 四组对同一 FC profile 的因子分解

第 2 项仍以同一 `3 regions × 17/8 networks` Fisher-z FC profile 为结局，不以
面积或边界为主要结局。

| Stanford cell | Trauma | Symptomatic | n |
|---|---:|---:|---:|
| NTHC | 0 | 0 | 51 |
| TEHC | 1 | 0 | 40 |
| NTS | 0 | 1 | 56 |
| TIS | 1 | 1 | 47 |

四类报告固定为：

1. marginal symptomatic effect：
   `0.5 × [(NTS−NTHC) + (TIS−TEHC)]`；
2. marginal trauma effect：
   `0.5 × [(TEHC−NTHC) + (TIS−NTS)]`；
3. interaction：
   `(TIS−TEHC) − (NTS−NTHC)`；
4. 四个预设 simple contrasts：
   `NTS−NTHC`、`TIS−TEHC`、`TEHC−NTHC`、`TIS−NTS`。

`TIS−NTHC` 和 `TEHC−NTS` 同时改变两个因素，只作描述性补充，不作为主要机制
contrast。四组 omnibus/global profile test 是辅助总览，不是第五种生物学效应，也不作为
逐network定位的统计门。

## 2. FC 表型构建

### 2.1 输入与单位

- 只读取已有正式 rest-fMRI FC/time-series 资产及其 provenance/QC manifest；
- 不读取名称或路径包含 beta、TMS-response、con、native stimulation mapping 的资产；
  数据集合法名称 `JJTMS`/`JJ_TMS` 本身不属于禁用标记，路径验证必须检查完整禁止组件，
  不能因字符串 `tms` 出现在数据集名中而拒绝rest-only输入；
- 每个输出值必须明确为 Fisher-z FC；
- 不以零填补缺失 ROI/network；保存有效 cortical parcels、有效海马 vertices 和缺失原因。

### 2.2 网络向量

对每个 subject × parcellation × smoothing × hemisphere × hippocampal region：

1. 保留到 cortical parcel 的绝对 Fisher-z FC；
2. 依据冻结 registry 聚合为17-network向量；
3. 从同一 parcel-level值按冻结映射聚合为8-network向量；
4. 保存完整绝对网络分量；
5. 不遍历任意 network-pair contrasts。先完成逐network定位，再把方向相反、
   生物学可解释且跨规格可复核的少量network pair登记为候选contrast；这些候选属于
   post-selection exploratory hypothesis，不能在同一发现样本中包装成预设验证。

### 2.3 海马分区

- `hcp_fixed`：固定三region基线；
- `hcp_template_matching`：个体化三region分析。

两种方法输出同构的FC张量并分别标记。它们是相关表征，不是独立复制。

Stanford 的四个 parent factorial groups（NTHC、TEHC、NTS、TIS）与三个 nested clinical
cohorts（NTS Any MDD、NTS Pure MDD、TIS PTSD）以及当前 HC reference map 组成
group-atlas geometry family；这些 group maps 与 subject-level FC 共用同一个严格
priority barrier，2 mm atlas cells 与 8-network/2 mm FC 同属 P0，必须共同完成后
才能进入低优先级层。该 family 由
`external_group_atlas_analysis/` 实现，primary run 写入
`results/primary_screen/group_atlas/`。group-wise 只比较每个非 HC 组与 HC，
不生成非 HC 组之间的全配对。它不进入 subject-level FC 数值推断，避免把
group-trained atlas 定义循环带回 FC。
不能让每个 cohort 使用自己训练的 atlas 提取 FC 后直接做组间推断，因为 atlas 定义已
包含组别信息。外部分支当前没有个体层不确定性、p/q、split-half 或 ICC，只能用于解释
拓扑是否伴随主 FC 信号变化。

## 3. 分支登记与统计顺序

所有 subject-level FC 结果必须由以下键唯一标识：

```text
dataset × group_estimand × parcellation × network_resolution(8/17)
× hemisphere_mode(bilateral/L/R/L-R) × region(anterior/middle/posterior)
× smoothing_mm(2/4) × fc_estimator × network
```

统一 branch ledger 另为 group-atlas geometry 记录：

analysis_family × group(HC plus seven cohorts) × assignment_variant(plain/weighted)
× smoothing_mm(2/4) × hemisphere(L/R) × K=3

优先级是严格 barrier：

| Tier | Subject-level FC | Group-atlas geometry |
|---|---|---|
| P0 | 8-network + 2 mm | 2 mm |
| P1 | 8-network + 4 mm；17-network + 2 mm | 无 |
| P2 | 17-network + 4 mm | 4 mm |

只有高优先级全部分支完成后，才允许启动低优先级分支；任一高优先级
分支失败时，运行停止，低优先级保持 blocked。当前 registry 不包含
individual-gradient 分支。

### 3.1 Primary：逐network localisation

逐network结果是第1点最重要、最可解释的发现层，不受profile global test是否显著的门控：

- JJTMS报告patient−healthy mean difference、standardized effect、95% CI和raw p；
  `q_bh` 在 analysis-object set 冻结前保持为空；
- Stanford用同一个四cell模型的contrast matrix输出两项marginal effect、interaction和
  四个simple effects；
- 每个固定的dataset × estimand × parcellation × resolution × hemisphere_mode ×
  smoothing × fc-estimator规格内保留完整 region × network 分量；multiplicity correction
- bilateral、L、R和L−R全部保留。L−R由同一被试同一region/network的
  `Fisher-z(L)−Fisher-z(R)`构成，检验组间偏侧化差异；
- 左右、2/4 mm和两种parcellation不计作独立replication；
- 所有绝对network分量完整保留。

### 3.2 Secondary：profile summaries

顺便计算但不作为定位门：

- anterior、middle、posterior各自的8/17维profile global comparison；
- 拼接三个region的`3×8`/`3×17` whole-hippocampus global overview；
- subject-level permutation保持一个被试的全部相关分量一起置换；
- global表报告距离统计量、permutation p、组别n和有效维数。

### 3.3 Candidate network contrasts

完成逐network定位后，只生成精简候选清单，不自动遍历或检验network pairs。候选必须记录：

- 两个绝对network effect及其方向；
- 候选来自哪个dataset/region/hemisphere/parcellation/resolution；
- 选择规则和source row IDs；
- 拟在另一个数据集、held-out分析或下一阶段验证的目标estimand。

primary_screen 的候选表目前只写了通用的 external/held-out 目标，没有冻结精确验证 estimand/
dataset/branch，因此这五条只能视为 hypothesis-generation registry，不能称为
preregistered validation。真正进入下一阶段前必须在运行前明确冻结目标；修正现行结果时直接覆盖对应 current 输出并同步清单。

本阶段为unadjusted promising-first screen。age、sex、site、motion和usable minutes
在出现promising结果后进入敏感性模型，不阻止基础screen启动。

## 4. 必需产物

1. JJTMS与Stanford各自的subject manifest和组别计数；
2. 两个数据集的17-network与8-network FC tensors；
3. region×network localisation完整表和L−R偏侧化结果；
4. subregion/whole-hippocampus profile global辅助表；
5. Stanford marginal、interaction和四个simple contrasts完整表；
6. 每组 `3 × network` profile heatmap及effect/CI图；
7. 跨数据集方向一致性表，只比较可对齐的patient/symptomatic estimand；
8. 基于定位结果的精简candidate-contrast registry，不在发现样本批量检验；
9. source/config hash、缺失原因、完整结果与negative findings。
10. 统一 `branch_manifest.json`，记录每个 subject-FC/group-atlas 分支的
    analysis family、method、轴值、priority tier 和状态；
11. group-atlas 64-cell inventory、HC-only matched comparison 和 descriptive
    contrast tables。

## 5. 当前状态与执行门

| 要求 | 当前状态 |
|---|---|
| Stanford 17/8-network完整profile与逐network定位 | 已完成：primary_screen |
| Stanford marginal symptom/trauma、interaction和四个simple contrasts | 已完成：primary_screen |
| JJTMS rest-FC patient vs healthy与17/8-network profile | 已完成：primary_screen |
| bilateral/L/R/L−R与2/4 mm、fixed/template规格 | 已完成：primary_screen |
| JJTMS与Stanford方向对齐 | 已完成探索性汇总 |
| Stanford group-atlas geometry（兼容输出） | 兼容路径保留；权威执行已纳入 primary_screen |
| Stanford HC-referenced group atlas 描述 | 纳入统一 primary_screen：64 cells，7 个 HC-only group-wise pairs |

权威 FC 运行是
`cross_dataset_rest_fc/*/primary_screen/`。该运行使用稳定 current 目录，发现错误时直接覆盖对应结果并同步清单；
同一运行中的 group-atlas geometry 表解释拓扑，但保持与 subject-level FC
推断分离；当前 group-wise 只保留非 HC 对 HC。

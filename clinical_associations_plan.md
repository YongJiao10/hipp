# Iowa 3T 海马 template matching × 临床量表关联分析计划

## 1. 研究目标

评估以下两类海马功能组织指标是否与临床量表存在关联，并判断 template matching 分区是否提供超出全海马汇总值的解释信息：

1. **Template-matching 分区指标**：左、右海马的 anterior、middle、posterior 三个分区，重点分析各分区的面积比例（area percentage）。
2. **功能组织 aggregation 指标**：`iowa_hipp_functional_organization` 产生的 segregation/margin 类指标，包括标准分支（standard）和 ctxreg 敏感性分支（ctxreg）。

核心问题是：关联是否具有分区特异性、左右侧差异、前中后梯度，以及在控制面积比例后是否仍保留功能组织信息。

## 2. 分析单位与数据结构

- 分析单位：受试者。
- 海马结构：左/右半球分别分析。
- 分区：A（anterior）、B（middle）、P（posterior）。
- 指标组合：每个量表字段 × 影像指标 × 半球 × 分区。
- 临床字段按显式层级保留：
  - `total`：总分；
  - `subscale`：分量表；
  - `item`：原始条目。
- 每个 item 通过字典中的 `parent` 连接到对应 subscale/total，不根据名称猜测层级。

量表、条目、分量表和总分不互相替代；总分不显著时仍保留并分析其条目与分量表。

## 3. 人群策略

### 3.1 主分析：全样本协变量调整

- 纳入 HC 与患者，共 93 人（HC 67，患者 26）。
- 使用年龄、性别和诊断组别作为协变量，得到 `all_adjusted` 结果。
- 该分析用于回答“海马功能组织与临床量表是否存在跨样本连续关联”。

### 3.2 次要分析：患者内分析

- 只纳入患者（N=26），得到 `patient` 结果。
- 主要调整年龄、性别；不把 HC 混入患者症状关联模型。
- 用于判断关联是否主要由 HC–患者组间差异驱动。

### 3.3 小亚组

- Pure MDD（N=7）和 Comorbid MDD（N=19）只作为覆盖、方向和可视化层面的探索性分层。
- Pure MDD 样本过小，不作为稳定的独立推断单元，也不把组间差异解释为确证性结果。
- HC 不从整个项目中删除：它保留在全样本调整模型中，但患者内模型单独报告。

### 3.4 组间差异筛选（新增，2026-09-19确认口径）

- 对全部 197 个声明字段做 HC（67）vs 患者（26）组间差异检验，调整年龄、性别。
- 方法：rank-ANCOVA 类比主分析 —— rank(y)/rank(age) 对 `[1, age_rank, sex]` 残差化后求偏秩相关，
  10,000 次 Freedman-Lane 置换得双侧 p；另报告未调整 Mann-Whitney U p 与中位数作上下文，筛选只用调整后 p。
- 资格与主流程一致（fail closed，无替代）：NIH 字段仅 `nih_timing_eligible` 样本；其余按字段完整样本；
  要求 HC/患者每组有效数 ≥5 且总量 ≥20，且 y 有方差，否则记 `not_estimable`。
- 自杀/自伤条目（phq_suicide、bdi_9、dsm5crossad_11 及其合计）在 HC 侧有效数为 0，
  组间检验不可估 —— 按 fail-closed 原则排除而非视为阴性，在图注与 CSV 中明确列出。
- MADRS 记录总量按既有政策 quarantined，不进入推断。
- 本轮结果：191 可估中 187 调整后 p<0.05；可推断 186；筛不掉的 8 个 primary-hit 字段为
  bdi_9、dsm5crossad_11、phq_suicide、phq_total_score_w_suicide、
  bdi_total_w_suicide、madrs_total_w_suicide（HC n=0，不可估）＋ dsm5crossad_4/13/21、panas_12（n.s.）。
- 分组并排热图（`clinical_associations_grouped.png`）**按组内口径自选列**：HC-only 或
  patient-only 任一组有 ≥1 名义显著格即入选（standard 分支，年龄性别调整），共 133 字段、
  其中总量表 18 个加粗；行同为 18 个 primary 解剖单元；列序为量表块 → 层级 → 组内最小 p；
  数据源为已有 `standard:hc:sensitivity` / `standard:patient:sensitivity`，不重算；
  自杀/自伤列 HC 侧无数据（灰格），患者侧（n≦24）正常显示。筛选明细见
  `results/clinical_associations_group_screen.csv`，分组值见 `results/clinical_associations_grouped.csv`。

## 4. 临床量表处理

- 纳入字典中声明的所有 total、subscale 和 item。
- 分析前执行字段去重、范围检查、缺失率、唯一值数和组别覆盖审计。
- NIH 相关量表按照量表的时间点资格规则处理；不合格时间点不进入对应模型。
- MADRS：由于总分与原始条目/字典存在不一致，**暂缓 MADRS 总分相关推断**；保留问题说明和覆盖率审计，MADRS 原始条目照常分析。

## 5. 统计模型

### 5.1 主要效应

- 使用 Spearman 秩相关的协变量调整形式（partial rank association）。
- 输出：`rho`、有效样本量 N、名义 95% bootstrap CI、双侧置换 p 值、缺失/不可估计原因。
- 对每个分区分别分析，避免把 anterior/middle/posterior 预先平均而掩盖空间异质性。

### 5.2 面积条件分析

除直接分析外，对 segregation/margin 指标提供面积控制版本：

- `segregation_given_area`
- `margin_given_area`

这用于回答：功能组织 aggregation 是否与临床量表相关，而不是仅仅反映某个分区面积更大或更小。

### 5.3 不确定性与稳定性

- 置换检验：10,000 次，输出原始双侧 permutation p 值。
- Bootstrap：2,000 次，输出 nominal 95% CI。
- Leave-one-out（LOO）：报告相关系数范围和符号翻转情况。
- 增量解释量：报告控制协变量以及控制面积后的 `delta R²`。
- 对常数输入、有效 N 不足、单组样本或 score mismatch 明确标记为不可估计/受限。

### 5.4 多重比较政策

**不做 FDR，也不做其他多重比较校正。**

- 报告原始双侧置换 p 值，不生成 q 值。
- `p < .05` 仅标记为“名义显著/探索性信号”，不能解释为项目级确证结果。
- 由于检验数量较多，重点结合效应量、置信区间、LOO、bootstrap 稳定性、standard/ctxreg 一致性和解剖分区模式解读。
- 10,000 次置换的最小可达到原始 p 为 `1/10,001`；p 值是 Monte Carlo 估计，不应脱离效应大小和不确定性单独解读。

## 6. Template matching 价值的判定框架

优先寻找以下模式，而不是只看一个最小 p 值：

1. **区域特异性**：关联集中在 anterior、middle 或 posterior，而非三个分区同方向同幅度。
2. **侧化**：左、右海马的效应大小或方向不同。
3. **前后梯度**：anterior→middle→posterior 呈有序变化。
4. **面积独立性**：原始 area percentage 关联之外，segregation/margin 在控制 area 后仍有关系。
5. **分支稳健性**：standard 与 ctxreg 方向一致，或差异本身揭示模型敏感性。
6. **层级定位**：item 或 subscale 的关联比 total 更局部、更清晰，能指出具体症状维度。
7. **跨人群解释**：全样本关联在患者内仍保留，或明确显示主要由 HC–患者组别差异驱动。

只有当效应量、置信区间和稳定性诊断支持上述结构时，才把结果称为 template matching 的潜在价值；单个名义 p 值不作为充分证据。

## 7. 可视化设计

报告需要体现“量表层级—影像分区—统计结果”的层级关系：

- **层级树**：按 scale 展开 total → subscale → item，显示 parent 关系、方向和时间资格。
- **关联热图**：行按量表层级排列，列按 metric × hemisphere × anterior/middle/posterior 排列；颜色为 rho，圆点标出原始 p<.05。
- **明细面板**：点击任意单元格显示 N、CI、原始 p、LOO、delta R²、模型状态和原因。
- **年龄–面积散点图**（新增）：`clinical_associations_age_area_scatter.py` →
  `results/clinical_associations_age_area_scatter.png`，4 行（HC/患者/pure/comorbid）× 6 列
  （左右 × Ant/Mid/Post），原始散点 + 组内 sex 调整的偏秩 rho/p/n；标题两行式
  （ρ 两位小数、p 括号内三位有效数字，显著整行标红）；  Pure MDD（n=7）照画但脚注警示不稳；`--unadjusted` 输出纯 Spearman 对照版
  (`..._unadjusted.png`，不调任何变量)。
- **覆盖诊断图**：显示每个字段在 HC、patient、Pure MDD、Comorbid MDD 中的有效数、资格数、缺失率和取值范围。
- **分组并排热图**：`clinical_associations_grouped.png`，左 HC-only、右 patient-only，
  与原图同列同序（103 字段），星号为各自组内名义 p<0.05（无 FDR）；数据源为已有
  `standard:hc:sensitivity` / `standard:patient:sensitivity`，不重算；筛选明细见
  `results/clinical_associations_group_screen.csv`，分组值见   `results/clinical_associations_grouped.csv`。
- **summary PNG**：提供全量表字段的 rho 热图及临床字段覆盖图，便于论文/汇报引用。

## 8. 主要交付物

- `results/clinical_associations_inputs_subjects.csv`：受试者与协变量输入。
- `results/clinical_associations_inputs_scales.csv`：量表字段、层级与审计信息。
- `results/clinical_associations_inputs_imaging.csv`：subject-level template matching 与功能组织指标。
- `results/clinical_associations_statistics.csv`：全部模型结果，使用原始 p 值，不含 q 值。
- `results/clinical_associations_report.html`：离线交互式报告。
- `results/clinical_associations_summary.md`：方法、覆盖、状态和主要原始 p 值摘要。
- `results/clinical_associations_methods.json`：输入、参数、资格规则和可复核元数据。
- `results/clinical_associations_overview.png`、`clinical_associations_coverage.png`：静态概览图。

## 9. 结果解释边界

- 这是探索性、横断面关联分析，不推断因果。
- 没有 FDR 或其他多重校正，名义显著结果需要保守表述。
- Pure MDD N=7 的结果只能作为方向性线索。
- MADRS 总分在字段一致性修复前不做相关推断。
- 量表条目之间可能存在相关性，不能把每个 item 的独立名义 p 值理解为相互独立的证据。
- 最终优先报告具有分区定位、侧化、面积独立性和稳定性支持的模式。

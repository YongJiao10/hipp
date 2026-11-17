# Iowa 海马功能组织：统计分析与四主题可视化

本目录是 Iowa 3T JJ_TMS 海马功能组织分析的独立、可复现入口。它只
写入本目录下的 `results/`，不会覆盖既有正式分析产品、组 atlas 或上游
输入。分析代码、输入清单、缓存、统计表、PNG、静态 PDF 和离线 HTML 均在这里。

## 固定范围

- 输入固定为 Kong2022 canonical7、36P、2 mm 平滑；两个信号分支
  `standard` 与 `ctxreg` 分开分析。
- 对照为 Iowa HC（67 人）；病例比较固定为 Patient（26 人）、Pure MDD
  （7 人）及 Comorbid MDD（19 人）各自相对 HC。Pure MDD 和 Comorbid MDD
  是 Patient 的重叠子集，三个比较不是独立重复验证。
- HCP 只提供三模板和外部梯度参考，不进入病例–HC 统计。
- 左、右海马分别分析；皮层按双侧 canonical7 网络汇总。模板匹配和梯度
  始终使用原始 400-ROI 特征，不缩减为七维输入。
- 主检验是无协变量、双侧 Welch t 检验，不做 FDR。结果保存病例与 HC 的
  均值、标准差、有效样本量、病例减 HC 的均值差及 95% CI、Hedges' g 和
  精确 p 值。标记为 `+`（0.05 ≤ p < 0.10）、`*`（p < 0.05）、`**`
  （p < 0.01）和 `***`（p < 0.001）。

`standard` 和 `ctxreg` 分别与 HC 对比。两个原始 margin 的差异不被解释为
回归效果，也不以显著性是否消失判断两个分支不同。路径选择是固定合同：
`standard → routea`、`ctxreg → routea_ctxreg`。其他皮层控制分支都不是 standard
的替代物（原 ctxdrop 分支已退役）；本入口从不因为文件存在情况在这些分支间
切换。

## Atlas 与指标定义

个体 TM 是每位受试者的模板匹配标签。TM dominant 是在每个队列内，按个体
TM 标签面积加权汇总形成的组标签；每个顶点只计入 `valid=true`、A/B/P 标签和
三模板分数有限的来源，若没有可用来源则明确保留为 0，绝不以无效顶点填补。
**de novo 指每个队列使用自身组平均 FC 所得梯度独立聚类的组 atlas；不是个体
TM，不是 group dominant，也不是逐个体重新聚类。** WTA 保留原有七网络分配，
只提取 VAN/SMN/DMN 形成 A/B/P pseudo-clusters；其余网络不重新分配，也不将
三者重新归一化。描述性组图也保留每个 atlas 自己的支持域：TM dominant
使用有资格的个体 TM 来源，de novo 使用其 `support`，WTA 使用其独立的
`wta_support`。不支持的顶点不参与面积/曲线并在图上留空，绝不显示成灰色
未分配顶点。

分析包括：

1. 模板身份与分化：signed 三模板相关、`Smax`、`margin`、固定 HCP
   A/B/P 空间 territory 内的 matched-template similarity 与 advantage，以及 TM 的
   Kember segregation 与 4 mm 测地邻域 boundary sharpness。固定 HCP territory
   在每个 branch × hemisphere 内相同；它不使用任何 Iowa 组 atlas 来定义取样区域。
2. 空间重排：A/B/P/未分配面积、20 个固定前→后展开坐标区间、面积加权
   重心、所有有效 A/B/P 网格边的前后轴中点（保留多重或不连续转换，不压成唯一
   边界）及描述性三 atlas 组图；皮层面积只读取一次 canonical7/36P 结果。上游
   表中的 `Noise` 和 `Unassigned` 背景组件由其明确的 `component_kind` 识别并
   排除，汇总只保留七个 canonical7 网络。
3. 连接指纹：每个受试者和侧别从区域平均海马时间序列到全部 400 个 ROI 的
   Fisher-z 连接，再按双侧七网络汇总。完整 ROI 效应和未校正 p 保存在表格，
   网络结果仅作为同一数据的汇总定位。
4. 连续组织：`rA`、`rB`、`rP` 和 `rP−rA` 的长轴曲线、斜率与 P95−P5 范围；
   个体 400-ROI FC 的五维 BrainSpace 梯度经不缩放、未中心化的正交
   Procrustes 对齐到匹配的 HCP 参考，再汇总参考空间的 G1–G3。

无效顶点不会被写成零或未分配标签。有效面积与整侧面积同时记录；空系统、
不可计算相关和不足的边界邻域保留缺失原因。数值细节见
[numerical_notes.md](numerical_notes.md)。

## 输入契约与核验

`inputs.py` 从明确声明的精确文件路径构建 `input_manifest.json` 和
`input_audit.csv`，并验证 roster、内容哈希、信号分支、标签语义、400-ROI
顺序、时间维度、左右侧隔离，以及实际消耗的 HippUnfold `den-512` 表面。
它不按文件存在顺序、glob 顺序或旧缓存选择替代输入；任何缺失、重复或不匹配
都会失败并指出合同和路径。原始 standard producer 的无 `signal_branch` schema
是一个明确、仅限标准 RouteA 路径的合同：字段必须缺失，不能以空值或另一分支替代；
ctxreg 则必须明确声明 `signal_branch=ctxreg`。上游 `valid=false` 顶点始终作为
缺失保留；其 template label 或分数从不用于个体指标或组 TM 汇总，也不会导致自动
排除受试者。不会访问 archive。

个体缓存仅是可追溯的写入产物，分析从不读取它来替代输入。固定 HCP identity
territory、HCP 梯度参考和 Iowa 组 atlas 都由 manifest 中的精确记录选择。固定
territory 的 labels/support 与每个受试者已使用的 HCP prototypes 必须逐一匹配；组
atlas 不参与该 identity 指标。组 atlas 的展开显示几何使用明确
声明的 `sub-1001` `den-512` anchor，且所有组图均标为描述性，不进行把共享
组标签顶点当作个体样本的检验。

## 运行

在 IDAS 上以最多四个分析工作进程运行；主进程加四个工作进程不超过项目的
十进程上限。三个数值库变量必须为 `1`：

```bash
/home/yojiao/.conda/envs/py314/bin/python -c "import sys; print(sys.executable)"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/yojiao/.conda/envs/py314/bin/python \
  scripts/visualize/iowa_hipp_functional_organization/analysis.py \
  --stage all --workers 4
```

阶段可单独运行：`inventory` 写入并核验输入清单，`analyze` 从已验证清单计算
指标和统计，`identity` 只从带哈希的派生统计文件重绘 Standard/Ctx-Reg 两张 identity
PNG，`render` 重绘完整报告。默认输出路径是
`results/`。已有同阶段输出时程序会失败关闭；只有显式传入 `--overwrite` 才允许
替换该阶段文件（包括已有的个体 provenance cache），且该选项不会启用任何输入
回退或替代选择。

## 产物

`results/` 包含：

- `subject_metrics.csv`、`curves.csv`、`boundary_positions.csv`、
  `statistics.csv` 和 `roi_statistics.csv`；边界表保存每一条有效 TM 边界边，
  而不是每系统一个强制的边界。ROI 表保存每个 ROI 的完整效应和未校正 p。
- `spatial_maps.json`、`roi_registry.json`、`analysis_config.json`、
  `run_manifest.json` 以及输入审计文件和按受试者/侧别的缓存。报告在重绘前会对
  `analysis_config.json` 的固定范围、三项比较、两侧/两分支主题覆盖、三 atlas
  组图网格及全部 400 ROI 效应覆盖进行哈希和语义双重核验。
- `theme_1_identity.png`、`theme_2_spatial.png`、
  `theme_3_fingerprints.png`、`theme_4_continuous.png`、
  `hipp_functional_boundaries_standard.png`（由 `render_functional_boundaries.py` 生成的出版级专图，涵盖 Section 1 四组 3D/2D 表面边界轮廓、Section 2 方法一基于累积面积截断点的一维纵向功能领地色带条与位移 ΔAP、Section 3 方法二局部概率曲线亚网格精确数学交点与位移 ΔAP）、
  `iowa_hipp_functional_organization_report.pdf` 和 `index.html`。

四张总图采用相同 A/B/P、网络颜色、左右侧顺序和以零为中心的 Hedges' g 色标。
展开图的前后轴以 0–38.75 mm 显示，右侧横向坐标只为共同显示方向而镜像；这不改变
原始表面数据、标签或统计。静态 PDF 是便于直接打开和打印的分页总览，包含封面说明
与四张完整主题图；精确结果仍读取同目录 CSV。HTML 可筛选信号分支、侧别、atlas 和
指标，悬停/表格显示组均值、差值 CI、g、p 及 N，并提供 400 ROI 的可检索效应表和
皮层定位。渲染输入、静态布局和缺失值语义见 [report_notes.md](report_notes.md)。

## 临床量表关联（clinical_associations）

独立入口读取本目录已有个体 TM 指标，分析 **2 mm / 36P / standard 主分支**及
ctxreg 敏感性分支；不重跑上游，不修改既有组统计。已有生产者的指标文件与
analysis_config 声明及其既定完整性合同须通过；没有文件选择回退。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/yojiao/.conda/envs/py314/bin/python -m \
  scripts.visualize.iowa_hipp_functional_organization.clinical_associations \
  --stage all --workers 4
```

从仓库根目录运行。支持 inventory / analyze / render；analyze 总是重新读取并同步保存
其实际使用的临床与影像数据，render 只消费本入口明确命名的输出。默认 10,000 次
置换、2,000 次 bootstrap、种子 20260917；正式输出禁止低于这些次数。默认遇到该阶段
已有输出会停止，显式 --overwrite 只替换该阶段列明的 clinical_associations_ 文件。
不自动创建输出目录。IDAS 使用 4 个单线程 worker 加主进程。

### 样本、计分与时间

- 固定 67 HC + 26 Patient，7 Pure / 19 Comorbid 是患者的子组属性，不重复计数。
  同一受试者被 Patient 与子组重复记录的指标必须一致，然后合并为一个观察。
- 197 个问卷评分／条目精确来自 TMSfMRI_DataSharing.xlsx 的 data / dictionary。
  不纳入 MINI 跳题、自由文本、Visit 2 或 post-TMS 条目。NIH 使用 T-score，不将其
  raw/theta/SE 当额外独立症状；已有 PANAS、STAI 分量表保留，不自创临床子量表。
- PHQ 主总分实际为 8 个症状之和；phq_9 是功能影响题，phq_suicide 才是自杀题。
  BDI 主总分排除自杀项。含自杀总分各自保留，不与短版互换。
- MADRS 条目已经导出为 0–3、间隔 0.5，字典却按未转换编码描述求和后除以二；
  按导出条目求和也有不一致。**用户确认：两个记录总分不进入任何推断，
  原始值／覆盖与错误说明保留，MADRS 条目继续分析。** 不重算替换总分。
- 非 MADRS 计分异常会停止；STAI 反向题与问卷计算公式核验后才使用。条目缺失不填零。
  Visit 1 摘录总分可能含自杀项或转录差异，明确排除，使用问卷精确字段。
- Visit 1 明确记录静息扫描。NIH 88 人与 Visit 1 同日、2 人在其后 1–5 天且早于
  Visit 2；另外 3 人在 Visit 2 当天、干预前后不明，只从 NIH 模型排除。
  因此 NIH 基线最多 N=90。PCL 仅 48 人施测，不把未施测当零分。

### 估计与多重检验

每人每侧 A/B/P 的面积百分比（分母为同侧完整原生海马表面积）、segregation、
margin 分别关联每个评分。组 atlas 的占比与 agreement 不作为个体影像指标。
全样本调整 HC/Patient、年龄、性别；另给 Patient 内部、HC 内部及全样本原始相关。
Pure/Comorbid 只在覆盖数据中区分，不运行 N=7 的独立发现检验。

连续变量使用平均秩（包括年龄和面积控制），性别与诊断用分类编码；
相关是影像与评分各自对协变量回归后的秩残差相关。
Freedman–Lane 残差置换重新投影协变量；全样本调整模型的置换限制在诊断组内。
Bootstrap 在诊断组内按人重采样，每次重排秩、重拟合；有效重采样不足 80% 时不提供 CI。
每个模型按实际完整数据计 N；N<20、恒定变量或不可识别设计没有推断结果。
全样本调整模型另外给逐人剔除范围／符号变化。CI 为逐检验的名义 95% 区间。

standard 的全样本调整 **area + segregation × 6 位置 × 全部 197 字段**
作为预先声明的主要结果层；margin、ctxreg、患者内模型及面积控制增量均作为敏感性／探索性层。
所有预定单元均报告原始双侧置换 p 值，不进行 FDR 或其他多重校正；family 标签只用于结果分组，不用于校正，也不因总分 p 值不显著而隐藏条目。
 增量 R²是秩量表结局的样本内解释度，不是交叉验证预测性能。

### 精简版（valuable：命中字段横向热图，total 加粗一眼可辨）

`clinical_associations_valuable.py` 只取主要层
（standard / all_adjusted / area_percent + segregation，原始双侧置换 p，
不做 FDR），保留至少有一个 nominal p<0.05 格子的字段（103 个：15 个 total/subscale
混排在各自量表块内、加粗显示，88 个条目常规字体），去掉 94 个零结果字段；
整表无一命中则 GAD-7 整量表被去掉。只输出静态图与表：

```bash
/home/yojiao/.conda/envs/py314/bin/python \
  scripts/visualize/iowa_hipp_functional_organization/clinical_associations_valuable.py \
  --overwrite
```

产物（`results/`，`clinical_associations_valuable_` 前缀）：`valuable.csv`
（176 个 nominal 格，按 p 排序；含 CI、LOO、N——这些是热图上没有的增量信息）、
`valuable_dropped.csv`（94 个被去掉的零结果字段清单）、`valuable.png`（横向分组热图：
103 列按量表分块、块内 total/subscale 优先再按最小 p 排序，total 加粗、item 常规，
45° 大字标签；行为 18 个主要格，分 Area % / Segregation / Margin 三块；
`*` raw p<0.05、`**` raw p<0.01，非显著同字段格淡显作背景）。
Top-20 森林图、report.html、report.pdf、coverage.png、overview.png 已删除：
top20 是热图的子集、零增量；html/pdf/overview 为零增量大文件；
coverage 缺失结构信息保留在 coverage.csv 中。
行为测试见 `test_clinical_associations_valuable.py`。

### 输出与解释

已有 results/ 内的 clinical_associations_ 前缀文件：

- overview.png：总分和已有子量表的静态总览（total/subscale 全字段，含零结果）。
- statistics.csv：全部 37,824 个预定模型格子、状态、N、rho、CI、原始双侧 p、family、LOO 和增量解释度。
- scales.csv / coverage.csv / score_checks.csv：真实层级、计分方向、有效人数与计分／时间问题。
- subjects.csv / imaging.csv：该次分析实际使用的个体输入；属于本地敏感数据，不自动发布。
- methods.json / summary.md：参数、限制、原始 p 值说明与各 family 的描述性概览。

面积具有组成依赖；左右侧显著性不同不证明侧化。segregation 和分区使用相同静息数据。
未调整头动、药物等混杂，不作因果解释；ctxreg 不算独立样本复现。
积极情感不等于专门快感缺失评分，担忧／自责不等于正式反刍量表，主观认知不等于客观认知。
原始 p 值只作描述性、探索性证据；结合效应量、置信区间、LOO 稳定性与覆盖率解读，不把最小 p 值写成 TM 的临床效用。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/yojiao/.conda/envs/py314/bin/python -m pytest -q \
  scripts/visualize/iowa_hipp_functional_organization/test_clinical_associations*.py
```

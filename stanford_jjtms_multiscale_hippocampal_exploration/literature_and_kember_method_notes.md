# Stanford/JJTMS 多尺度海马分析：文献依据与 Kember 方法拆解

## 范围与证据等级

本文只记录可核验的一手论文、期刊页面、作者发布材料和 HippUnfold/Connectome Workbench 官方方法资料；`事实` 与 `建议/推断` 分开陈述。本次专项核对以 Kember 论文的最终期刊页面（2026-06-18 发表，2026-08-04 version of record）为准；本地解析 Markdown/JSON 只是该论文的工作副本，不替代期刊页面。论文全文见 [Kember et al., Nature Communications](https://www.nature.com/articles/s41467-026-74572-1)，作者在论文中链接的代码/数据入口为 [JonahKember/precision_mapping](https://github.com/JonahKember/precision_mapping) 和 [Zenodo 10.5281/zenodo.20086215](https://doi.org/10.5281/zenodo.20086215)。本次没有访问仓库中的任何 archive/归档目录或归档内容。

## 2026-09-01 Kember system-area 专项审计（本文件当前唯一面积结论）

本次结构重建将该证据收束为唯一书面决定：
`area_source = hippunfold_native_surfareacorobl`。后续代码必须使用
native HippUnfold `surfareacorobl` 逐顶点 mm² 值按 native system label 求和；
绝对面积是 Kember-aligned primary endpoint，fraction 仅是 secondary context。
字段合约和三工作流边界分别见 [docs/kember_area_method.md](docs/kember_area_method.md)
与 [docs/analysis_map.md](docs/analysis_map.md)。

### 结论先行

1. **Kember 的面积不是边界面积、过渡带面积、体积或顶点数。** 它定义为：某个功能 system 的标签所覆盖的 HippUnfold 顶点，其逐顶点 surface-area 值之和，单位为 `mm²`。system 本身是由海马—新皮层连接 profile 的 group template matching 得到的 anterior/middle/posterior 功能系统。
2. **原文主结果使用绝对面积。** 论文直接报告 posterior system 的 `mm²` 面积随年龄变化（例如年龄组间的 `650 ± 200` 到 `394 ± 125 mm²`），因此不能用面积比例替代原文主端点。
3. **原文存在相对面积/比例的归一化语句，但没有把它定义成唯一主端点。** 方法段说明在涉及“海马一部分”的计算时，用该部分逐顶点面积和除以相关范围内的全部逐顶点面积和。论文没有说明颅内容积、全脑面积、z-score 或 voxel count normalization；当前项目应把比例作为 secondary/context，而把绝对 `mm²` 作为 primary。
4. **统一后的 `kember_metrics` 是唯一的 Kember 面积实现。** 它直接消费 native HippUnfold per-vertex 面积 metric，再按 native 标签求和；这与论文“sum ... surface area ... output from HippUnfold”的输入语义最接近。the retired geometry-only mesh reconstruction 是合理的几何重算，但不是已被论文证明等价的原始输入路径。
5. **当前统一规则应是：** Kember 面积使用与 subject-specific native/folded midthickness 对齐的 HippUnfold 顶点面积 metric；保存每半球绝对面积，比例只作辅助；保留 mesh 重算仅用于独立 QC/数值一致性检查，不再把 legacy `transition_area_*` 或几何边界量称为 Kember 面积。

### 一手证据链

| 要回答的问题 | 一手证据 | 对当前项目的含义 |
| --- | --- | --- |
| 面积如何定义 | [Kember et al., Nature Communications](https://www.nature.com/articles/s41467-026-74572-1) Methods 的 “Quantifying topographic and functional specialization”：将 system 内每个 vertex 的 HippUnfold surface-area 相加 | 面积估计量是 `Σ A_v`，其中 `v` 是被 system label 选中的顶点 |
| 采用什么表面 | [Kember 论文](https://www.nature.com/articles/s41467-026-74572-1) Methods 说明 HippUnfold 从 `0.8 mm` T1w/T2w 构建 hippocampal archicortical ribbon 的 midthickness surface；BOLD 也投影到该 midthickness surface；图 3 在每名受试者的 midthickness surface 上叠加 system | 生物学 surface-area 应来自受试者自身的 folded/native midthickness 几何/metric，不是 unfolded 模板矩形面积，也不是 2/4-mm BOLD 网格面积 |
| 顶点面积的上游含义 | [HippUnfold 官方输出文档](https://hippunfold.khanlab.ca/en/latest/outputs/output_files.html) 说明 surface-area metric 基于 surface geometry，并在 native/T1w 空间提供；[HippUnfold 原始方法论文](https://elifesciences.org/articles/77945) 说明 morphometry 在 folded native surfaces 上用 Connectome Workbench 计算 | 上游 HippUnfold 面积 metric 是优先输入；文件名 `surfareacorobl` 是本仓库的具体资产约定，论文正文没有直接写出该文件名 |
| 几何重算的算法对应物 | [HCP/Connectome Workbench 官方 `-surface-vertex-areas` 文档](https://www.humanconnectome.org/software/workbench-command/-surface-vertex-areas)：每个顶点获得其参与的每个三角形面积的三分之一，单位为 `mm²` | the retired geometry-only mesh reconstruction 的 cross-product + `triangle_area/3` 与该 Workbench 顶点面积约定一致，但只有在表面、拓扑、空间和上游版本完全相同的前提下才可认为数值等价 |
| 是否有 normalization | Kember 论文同时报告绝对 `mm²`，并在方法中说明对“部分海马”的计算可除以相关范围的全部顶点面积和 | 当前项目应同时保留绝对面积和显式定义的 fraction；不能把 fraction 单独冒充 Kember 的原文主端点 |
| 代码/数据可复现入口 | 期刊页面的 Data/Code availability 链接到作者 GitHub、Zenodo 和 Code Ocean | 用这些一手入口核验作者发布实现；本报告不把本仓库旧版 source lock 或任何 archive 内容当作新的外部证据 |

### 面积估计量、输入和归一化的严格拆分

| 层级 | Kember 原文能确认的内容 | 本仓库应冻结的表述 |
| --- | --- | --- |
| 标签 | group-average hippocampal–neocortical connectivity → gradients → `k=3` GMM → anterior/middle/posterior templates；每个 subject 的 vertex 以与 template 最大相关者赋值 | 标签是功能 system label，不是 FreeSurfer anatomical subfield label，也不是“边界区域”标签 |
| 几何 | HippUnfold 的 subject-specific archicortical ribbon，midthickness surface；论文报告 native/unfolded 两种对应表面和约 `0.5 mm` vertex spacing | 面积使用 folded/native 的 subject-specific surface-area metric；unfolded 表面用于拓扑对应/显示，不用其模板几何直接替代 native 面积 |
| 顶点面积 | 论文明确说取 HippUnfold 输出的每顶点 surface-area 并求和，但没有在正文写出 metric 文件名或具体 Workbench 命令 | `surfareacorobl.shape.gii` 是当前仓库的 asset-backed 实现选择；其“对应论文输入”是高可信方法映射，不应写成论文正文已明确给出的文件名 |
| 绝对端点 | `system_area_mm2 = Σ_{v∈system} A_v` | primary：保留每半球每 system 的绝对 `mm²` |
| 相对端点 | 方法段说明在涉及海马一部分时按全部相关顶点面积和做除法；具体“相关范围”、无效顶点处理和双侧聚合在正文中没有充分展开 | secondary：`system_area_mm2 / total_valid_native_surface_area_mm2`，分母和 valid/unassigned 处理必须写入 manifest |
| 平滑 | Kember 的 2-mm BOLD smoothing 影响功能信号，不改变由结构 surface metric 得出的面积 | 2/4-mm functional branches 共用同一结构面积 metric，不为每种 BOLD smoothing 重新生成面积 |
| 非面积指标 | segregation 和 boundary sharpness 是 BOLD correlation 的 Cohen’s *d*；boundary sharpness 在边界邻域内算功能效应量 | `boundary_area`、`transition_area`、`boundary_length` 均不是 Kember 原文面积端点，必须单独命名为项目扩展 |

### 两种实现分别对应什么

#### the retired geometry-only mesh reconstruction

- `load_midthickness()` 读取 GIFTI 的 point coordinates 和 triangle faces。
- `compute_vertex_area()` 对每个三角形用 3D cross-product 求几何面积，并把该面积的三分之一加到三个顶点；这正是 Connectome Workbench 官方顶点面积命令所描述的离散分配规则。
- 迁移前的 the retired geometry feature implementation 从 512 surface geometry 重算
  面积；该实现现已退出活跃路径，仅作为历史几何/QC 对照。
- the retired topology feature implementation 还使用同一类 geometry weights 计算旧 topography 的 `area_mm2`、`area_fraction` 和 `transition_area_margin_*`。这些字段的标签/过渡定义属于项目自定义 topology 探索，不能因为使用了几何面积就改称 Kember system area。

该实现只有在以下条件同时满足时才可作为数值等价的独立复核：使用与上游 metric 完全相同的 folded/native midthickness、坐标单位为毫米、三角拓扑和顶点索引完全一致、没有额外重采样/裁剪，并且逐顶点结果与 HippUnfold metric 通过数值比较。当前代码没有以 `surfareacorobl` 作为输入，也没有在 Kember feature build 中强制这项逐顶点等价性检查。

#### `kember_metrics/src/surface_area.py`

- `compute_system_surface_area()` 接收外部提供的 `vertex_area_mm2`，不自行猜测或重算面积；按 system labels 求和，并保留完整面积、valid native 面积、已标记面积和未分配面积。
- `input_contract.py` 要求实际的 native midthickness、相应的 unfolded/native 对齐资产以及 native `surfareacorobl.shape.gii`；runner 检查 metric 长度与 consumed native surface 顶点数一致。
- 固定 512 标签先通过 `ADAP_BARY_AREA` 传到 native 网格；该面积加权只用于 label transfer，最终 Kember 面积仍是 native per-vertex metric 的求和。这解决了本项目“标签来源网格”和“面积测量网格”不同的问题。
- 当前 fraction 分母为 `total_valid_surface_area_mm2`，其中 valid 但 label 为 0 的顶点仍在分母；左/右双侧输出目前使用算术平均。这些是透明记录的本地适配约定，其中 fraction 分母细节和双侧聚合并未在 Kember 论文正文中完整说明，不能倒写成原文明确规定。

因此，`kember_metrics` 是直接使用 HippUnfold native 顶点面积输入的唯一活跃
Kember 实现；the retired geometry-only mesh reconstruction 只保留几何/QC 角色。

### 当前项目的统一选择

**选择 `kember_metrics` 作为唯一 canonical Kember system-area 路径。**理由是：

1. 它直接落实论文的“sum HippUnfold output”措辞，避免把本仓库自行重算的 geometry 数值误称为原文输出。
2. 它保留 native folded surface 的真实个体表面积，且通过顶点数、surface fingerprint 和输入哈希检查标签与 metric 是否同一索引空间。
3. 它能处理当前项目的 512-label → native-area 组合；the retired pre-rebuild branch 当前 branch 则停留在 corobl/512 geometry 上，容易让标签空间、metric 空间和面积来源混在一起。
4. 它使 fixed/template-matching 两种活跃方法共享同一面积测量定义；差异只来自标签来源，不来自面积算法。

后续代码重建时的最小规范应是：

- Kember 主表只从 native HippUnfold per-vertex area metric 读取 `A_v`；不能在另一个目录再次定义一套 Kember `area_mm2`。
- 每半球的绝对 `mm²` 是主端点；fraction、双侧汇总和未分配面积作为带公式和分母说明的派生字段。
- the retired geometry-only mesh reconstruction 若保留，应改为 geometry QC/通用 surface utility 的明确角色；不得继续作为 Kember 主结果的隐式面积供应器。
- 在实际 pilot 上逐顶点比较 `compute_vertex_area(native_surface)` 与 `surfareacorobl`，记录最大/平均绝对差和相对差；一旦不一致，以上游 HippUnfold metric 为权威，不以 mesh 重算覆盖它。

Task 1 冻结了文献/输入契约；Task 2 已在 `kember_metrics` 中落实该代码路径。本次未重跑或
重写 legacy 结果树。

### 不确定性与不能过度声称的部分

- Kember 最终论文明确了“对 HippUnfold 输出的逐顶点面积求和”，但没有在正文中写出 `surfareacorobl` 这一文件名，也没有公开说明作者当时是否直接调用 Workbench 生成 metric，还是读取 HippUnfold 已生成的 metric。故本仓库使用 `surfareacorobl` 是有官方 HippUnfold 输出语义支持的实现选择，但不是论文正文逐字给出的文件名。
- 论文方法的 normalization 语句没有完全规定 invalid vertex、label-0 vertex、单半球与双半球合并时的处理；本仓库 `total_valid_surface_area_mm2` 分母和左右算术平均必须标为 local adaptation，并在 manifest 中逐项记录。
- 论文显示并分别分析左右海马，但最终双侧字段的具体聚合在期刊方法文字中不够明确；正式复现应优先报告 L/R 原始值，双侧值作为明确写出公式的派生量。
- 论文的 `boundary sharpness` 不是几何面积，也不能从 system surface-area 变化反推出 boundary-area 变化；若项目要研究边界长度/过渡带面积，必须作为新增 exploratory estimand 另行命名、定义和验证。

## 最重要的术语纠正

**事实：Kember 没有测量“海马 subregion 边界面积”或“transition-zone area”。** 论文测了三个不同量：

1. **system surface area（系统表面积）**：属于 anterior/middle/posterior 某一标签的 HippUnfold 顶点面积之和；需要比例时，再除以全海马顶点面积之和。
2. **functional segregation（功能分离度）**：全海马每个顶点与目标 system 平均 BOLD 的相关分布中，system 内与 system 外的 Cohen's *d*。
3. **boundary sharpness（边界锐利度）**：只在标签边界附近计算上述内外相关分布差异；是一个 BOLD 功能对比效应量，不是几何边界的长度、面积或移动范围。

因此，最贴近 Kember 的复现应称为“**system 表面积 + 功能分离度 + 边界锐利度**”。若 Stanford/JJTMS 要分析“边界面积/过渡带面积”，那是合理但**新增的探索性指标**，不能表述为复现 Kember。

## Kember 2026：到底做了什么

### 输入、模板和个体映射（事实）

- 主样本为 HCP-D：N=471，5–21 岁，4 个采集站点；结构像 0.8 mm T1w/T2w，静息态 fMRI 2.0 mm，受试者平均保留 `21.78 ± 3.67` 分钟低运动数据。
- HippUnfold 从 T1w/T2w 构建海马 mid-thickness surface；文中报告 native/unfolded surface 上约 0.5 mm vertex spacing。预处理 BOLD 以 trilinear interpolation 投影到表面，再以 2 mm FWHM 平滑。
- 每个半球分别处理。每名受试者生成 `[7,262 hippocampal vertices × 32,492 neocortical vertices]` 的 BOLD 相关矩阵，再形成 group-average 矩阵。
- 对 group-average 海马-皮层连接进行 BrainSpace diffusion-map embedding（`normalized_angle` affinity），取前 100 个 gradients；作者说明实际加入约 5 个以上 gradients 已几乎不改变结果。
- 以 GMM `k=3` 对 gradient loadings 聚类，并依解剖位置人工命名为 posterior/body/anterior。每个 cluster 的平均皮层连接向量成为一个 template。
- 在每名受试者内，将每个海马顶点的皮层连接 profile 与三个 templates 相关；winner-take-all 分配给最大相关的 template。
- QC 排除标准：低运动数据 `<10 min`（低运动定义为 FD `≤0.25 mm`）、HippUnfold 重建 Dice `<0.7`、或未正确投影的海马表面积 `≥20 mm²`。
- 作者明确指出：三分区在每名受试者均可识别；`k≥4` 并非如此。作者认为更细个体分区可能需要更高空间分辨率（如 ≥7T）或更长扫描（如 ≥1 h）。posterior template 最独特，而 anterior/middle 更相似，这也限制了对“是哪一侧在发生分化”的唯一解释。

### 三个指标的精确定义（事实）

| 指标 | 计算输入 | 个体级输出 | 它不是什么 |
|---|---|---|---|
| system surface area | HippUnfold 每顶点面积 + 个体 template-matching 标签 | 每 system 的 mm²；或占全海马面积的比例 | 不是 boundary area |
| segregation | 每个海马顶点与目标 system 平均 BOLD 的 Pearson 相关 | system 内相关与 system 外相关的 pooled-SD 标准化均值差（Cohen's *d*） | 不是跨 system FC，也不是皮层-海马 FC |
| boundary sharpness | 上述顶点相关图 + 目标 system 的标签边界 | 对每个 boundary vertex，取其 `±4.0 mm` 邻域，计算边界内外相关的 Cohen's *d*，再跨 boundary vertices 平均 | 不是边界长度、表面积、位移或概率过渡带 |

正文/方法对左右半球的映射明确是分开进行，但本地解析文字对主结果最终是左右求和还是平均没有给出足够明确的操作说明；图 4 示意只展示左侧。**实现前必须以作者发布代码核定 laterality aggregation 和 `±4 mm` 的距离实现方式**，不能自行假定为欧氏、geodesic 或 unfolded-coordinate 距离。

### Stanford 2026-09-01 two-method 实现记录

- 活跃实现位于 `kember_metrics/src/`，同时消费 `hcp_template_matching` 与
  `hcp_fixed`。功能指标保留 512-source BOLD/labels；面积在标签经
  `ADAP_BARY_AREA` 对齐后消费 native `surfareacorobl`。
- 每个 hemisphere、每个 anterior/middle/posterior system 输出三项：`surface_area_mm2`、`functional_segregation_d`、`boundary_sharpness_d`。`surface_area_fraction` 保存在 feature table 作为归一化上下文，不另算第四个 direct-comparison endpoint。
- 本实现把文章所述的 ±4.0 mm 落实为 consumed source-512 midthickness 3D Euclidean radius 4.0 mm；这是本项目冻结的功能指标操作定义，记录为 `source_512_midthickness_euclidean_radius_mm`，不把它表述为已核验的作者源码细节。Native geometry/metric 只用于 system surface area。
- Stanford direct run 对 TEHC、NTS、NTS Any MDD、NTS Pure MDD、TIS、TIS PTSD 六个 case−NTHC contrast 做双侧 Welch test；2 methods × 2/4 mm × L/R × 3 metrics × 3 systems 合计 432 个 method-aware endpoint rows。文字与图使用原始 p、Welch CI 和 Cohen d；结果表保留 method 与完整两种 smoothing。
- 若所有对应 2/4-mm pair 的 d 方向及 p<0.05 结论一致，图只显示 2 mm；否则分别显示 2-mm 和 4-mm 图。

### 原文的统计与“null”（事实）

- surface area、segregation、boundary sharpness 与年龄的主检验主要是 Pearson *r*；system 间平均差用配对 *t* 检验；三指标共同预测年龄时用线性回归。
- 连接-行为的预设小家族使用 BH-FDR：12 tests = 3 个行为量 × 2 个网络 × 2 个 posterior contrasts，FDR `q=.05`。
- mediation 使用 bias-corrected nonparametric bootstrap。
- 稳健性包括 leave-one-site-out、控制 sex 和低运动数据量，以及独立 HBN 测试集。HBN 复现了表面积-年龄关系，但 **posterior segregation 和 boundary-sharpness 的年龄效应没有复现**；作者将其可能归因于更少数据（约 13 vs 22 min）、更低空间分辨率（2.5 vs 2.0 mm）或 movie/rest 混合。
- **没有报告**专门针对 surface area、segregation 或 boundary sharpness 的空间旋转 null、label permutation null、边界随机化 null 或 vertex-wise cluster null。这里的 Cohen's *d* 是每名受试者的描述性 feature，随后在受试者层面推断。
- 原文不是临床组间研究，没有提供“四组差异”或 symptom × trauma interaction 的现成统计方案。

### 不能直接从 Kember 推出的结论

- 不能把 system surface area 的改变说成“边界面积改变”；表面积变化可以由一个或多个边界移动、局部小岛标签或总海马面积变化造成。
- 不能把 boundary sharpness 解释为组织学边界或因果回路精化；它是同一 fMRI 信号上的相关分布效应量。
- 不能把儿少发育相关性直接外推为成人患者/创伤的疾病机制。

## 相关原始研究提供了什么依据

### 皮层网络 ↔ 海马长轴

- Barnett 等用数据驱动静息态分析区分 medial-temporal、posterior-medial、anterior-temporal 和 medial-prefrontal 网络；anterior hippocampus 相对更连 MP/AT，posterior hippocampus 相对更连 medial-temporal 网络的背侧 medial-parietal 部分。这支持把“皮层网络 × 海马位置”当作交互结构，而不是只比较全海马平均 FC。[PLOS Biology 2021, DOI 10.1371/journal.pbio.3001275](https://doi.org/10.1371/journal.pbio.3001275)
- Zheng 等在高密度个体数据中报告 parallel hippocampal-parietal circuits，并强调 group-average 与 individual-specific 映射、winner-take-all 选择会影响 posterior hippocampus 的结论。[PNAS 2021, DOI 10.1073/pnas.2101743118](https://doi.org/10.1073/pnas.2101743118)
- Angeli 等在两个小型 precision-fMRI 样本中得到 network-anchored 双分离：anterior-DN 区与 scene construction/remembering 更相关，posterior-SAL/PMN 区对 oddball/任务转换更敏感；作者同时明确其网络约束分区会弱化 posterior 内部复杂性。[PNAS 2025, DOI 10.1073/pnas.2422083122](https://doi.org/10.1073/pnas.2422083122)
- 连续梯度证据提醒我们不要把三标签当作唯一真实边界：Nordin 等在两个成人样本中得到至少两个重叠的海马-皮层连接维度，且年龄较大与 gradient transitions 更不清晰有关。[eLife 2025, DOI 10.7554/eLife.97658.3](https://doi.org/10.7554/eLife.97658.3) Xie 等也用 connectopic mapping 研究发育、皮层 hierarchy 与记忆。[PNAS 2024, DOI 10.1073/pnas.2403015121](https://doi.org/10.1073/pnas.2403015121)

### 个体化 template matching 的支持与边界

- MIDB Precision Brain Atlas 证明个体网络 topography 具有显著个体差异；其 template matching（TM）以独立模板组生成 network connectivity templates，再将受试者 grayordinate 的全脑连接 profile 分配给最大相似模板。用 split-half NMI 比较时，TM 的 within-person 一致性高于 between-person null，并可在约 10 min 低运动数据上工作。[Nature Neuroscience 2024, DOI 10.1038/s41593-024-01596-5](https://doi.org/10.1038/s41593-024-01596-5)
- 这不保证海马边界在 Stanford 数据中同样可靠：皮层 mapping 的 SNR、海马小体积/partial-volume、fMRI 分辨率和可用时长不同；Kember 的 HBN 结果已经显示 sharpness 对较短/较低分辨率数据不稳。
- TM 是监督式、template-constrained 方法；它能提高标签可比性，却可能错过患者特有的新型 topology。建议同时保存每个 template 的连续 similarity，而不只保存 winner-take-all 标签。

### 创伤/PTSD 与 anterior–posterior 假设：证据并不一致

- Lazarov 等比较 PTSD（n=48）与 trauma-exposed healthy controls（n=34）：全海马无组差，但分开 anterior/posterior 后发现 PTSD 中前后连接分化丢失，以及 posterior hippocampus–precuneus 连接差异。[Journal of Psychiatric Research 2017, DOI 10.1016/j.jpsychires.2017.06.003](https://doi.org/10.1016/j.jpsychires.2017.06.003)
- Chaposhloo 等在 31 名老年男性 Vietnam-veteran PTSD 与 29 名 combat-exposed controls 中报告 anterior hippocampus 与 affective regions 的 hyperconnectivity、anterior hippocampus–PCC/precuneus 与 CAPS 严重度负相关，以及左 anterior hippocampus 更 hub-like；样本的人口学与创伤类型限制外推。[NeuroImage: Clinical 2023, DOI 10.1016/j.nicl.2023.103417](https://doi.org/10.1016/j.nicl.2023.103417)
- Malivoire 等的小样本（PTSD n=11；trauma-exposed controls n=13）反而报告 PTSD 的 left posterior hippocampus–bilateral PCC FC 更高，并观察到若干未来症状相关性。[BMC Psychiatry 2018, DOI 10.1186/s12888-018-1716-9](https://doi.org/10.1186/s12888-018-1716-9)
- 一项 84 名 trauma-exposed adults 的动态 co-activation 研究显示：intrusive-memory emotional intensity 与 aHPC–DMN pattern 有关，而“here-and-now” reliving 与 pHPC–visual cortex pattern persistence 有关；传统症状总分并未显示同样关联。这支持把创伤表型分解为 affective/contextual/sensory-perceptual 维度，而非只用一个总分。[原始研究全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11408261/)

**结论：**“trauma 导致 anterior hippocampus 异常”可以作为预注册方向性假设之一，但不能作为既定事实。更稳妥的主假设是 **trauma/PTSD 改变 anterior–posterior differentiation 及其与 DMN/PMN/salience/visual 网络的耦合**；anterior-specific、posterior-specific 与去分化三种模式应作为可区分的竞争假设。

## Stanford/JJTMS 的可实施适配

### 1. 统一 primary workflow 与方法角色

| Workflow | 当前角色 | 关键边界 |
|---|---|---|
| `cross_dataset_rest_fc` | network × hippocampal-system Fisher-z FC 与 Stanford factorial inference；统一调度 group-atlas geometry | FC 行不解释为面积，geometry 行不混入 FC 数值推断 |
| `kember_metrics` | `hcp_template_matching` 与 `hcp_fixed` 的 subject-level area、segregation、sharpness | 功能指标使用 source-512；面积使用 native `surfareacorobl` |
| `external_group_atlas_analysis` | unified group-atlas geometry family 的实现/报告兼容入口 | 不独立调度，不作为第三个 Kember method，不回填 subject-level inference |

`cross_dataset_rest_fc` 是权威统一编排器，subject-FC 分支和 group-atlas geometry
family 共用 branch manifest、优先级顺序和失败 barrier。Kember 的 template 本身来自
group-average connectivity gradients → GMM → subject-level matching，但当前 package 的
方法仍严格只有 `hcp_template_matching` 与 `hcp_fixed`。Group-specific clustering
只生成统一运行中的 geometry family，不进行个体聚类。

**Template 规则（建议/推断）：**所有 Stanford 四组必须共用同一 template。首选与结局独立、年龄/采集相容的外部模板；否则在 Stanford 内做按组平衡的 training folds，并只在 held-out subjects 计算结局，重复 cross-fitting。绝不能各组各自建 template 后比较“面积”，因为标签定义已经随组改变。

### 2. Kember 指标的严格复现层

每半球、每 system 输出：`area_mm2`、`area_fraction`、`segregation_d`、`boundary_sharpness_d`、与解剖 subfield 的 overlap、有效边界顶点数、未投影面积、低运动时长和 mapping-confidence。主检验前先做：

1. split-half reliability（ICC/NMI/标签 Dice；连续 similarity 也要评估）；
2. motion-matched 或 equal-duration sensitivity；
3. 映射在 half A、sharpness 在 half B，再反向平均。这个 cross-fit 是对 Kember 的保守增强，因为同一 noisy BOLD 数据既用于选边界又用于评价边界可能产生选择偏差；独立选择/检验是避免 circular analysis 的通用要求。[Kriegeskorte et al. 2009, DOI 10.1038/nn.2303](https://doi.org/10.1038/nn.2303)
4. 若 Stanford 数据的低运动时长/空间分辨率明显弱于 Kember 主样本，则 boundary sharpness 降级为 exploratory，不能作为 primary endpoint。

### 3. 真正的“边界/过渡带”扩展层（明确标为新增）

建议同时输出以下互补量，但必须先冻结定义：

- **boundary position**：沿 HippUnfold 内在 anterior–posterior coordinate 的边界中位位置与离散度；直接回答边界前移/后移。
- **boundary length**：跨标签 mesh edges 的长度和；回答边界是否更曲折/碎片化。
- **transition-zone area**：不要用固定标签的几何 `±4 mm` 带简单充当新发现。更有功能意义的定义是 template top-1 与 top-2 similarity margin 小于预设阈值（或 label entropy 高于阈值）的顶点面积和；阈值必须在独立训练数据固定，并报告多阈值敏感性。
- **soft transition slope**：沿 AP 轴对 `similarity_posterior − similarity_body/anterior` 拟合平滑曲线，报告零交点与局部斜率。它比硬标签面积更少受 winner-take-all 抖动影响。

null 建议：在保持每名受试者 label 面积/空间自相关的前提下随机化边界位置或使用沿 AP 坐标的 circular shift，建立 boundary length/transition area 的 subject-specific null；组间推断仍在 subject level。简单随机打乱顶点会破坏空间自相关，不是合适 null。

### 4. 当前第 1、2 项与 Kember 扩展的关系

- **问题 1（跨数据集网络定位）**：对 JJTMS patient−healthy 与 Stanford symptomatic−healthy，
  比较每个 hippocampal region × cortical network 的 Fisher-z FC。
- **问题 2（Stanford 四组分解）**：对同一 Fisher-z FC profile 估计 marginal symptom、
  marginal trauma、interaction 和四个预设 simple contrasts。
- **Kember 扩展（独立第 3 项）**：system area、segregation、sharpness、边界位置等海马内部
  拓扑量；不能替代问题 2 的 FC interaction。

问题 1 与问题 2共享结局、区别在设计和 estimand；Kember 扩展才回答海马内部空间/功能组织
是否不同。探索数据中的关联或 mediation 只能称关联性分解，不能称机制因果。

### 5. 四组与 symptom × trauma interaction 的统计防护

1. 若四组来自两个二元因素，优先参数化为 `diagnosis + trauma + diagnosis×trauma`，直接检验 interaction；随后报告四组估计均值、95% CI 和预设 contrasts。不能用“一组显著、另一组不显著”代替组间差异检验。[Gelman & Stern 2006, DOI 10.1198/000313006X152649](https://doi.org/10.1198/000313006X152649)
2. symptom × trauma 若两者连续，保留连续尺度并中心化；绘制真实观测支持区间内的 conditional slopes，同时检查非线性和 common support。线性交互默认效应随 moderator 匀速变化，可能产生模型伪象；需用 bins/spline sensitivity 检查。[Hainmueller et al. 2019, DOI 10.1017/pan.2018.46](https://doi.org/10.1017/pan.2018.46)
3. 若 symptom 在 controls 中结构性缺失/几乎为零，不要在四组中强行拟合同一个 symptom slope；将病例内 symptom 模型与完整四组 factorial 模型分开。若 trauma 只是 group-defining binary，而非连续量，也不要将其写成剂量反应。
4. 调整 age、sex、motion、clean minutes、site/batch、medication 等预设 nuisance 时，为避免交互被 covariate-dependent effects 混杂，应加入 `covariate×diagnosis` 与 `covariate×trauma` 项；这是交互研究中特别容易漏掉的控制。[Keller 2014, DOI 10.1016/j.biopsych.2013.09.006](https://doi.org/10.1016/j.biopsych.2013.09.006)
5. repeated outcomes（hemisphere/system/network）使用 subject random intercept，并在数据允许时加入相应 random slopes；先做 omnibus interaction，再解释 planned contrasts。
6. 对预设 ROI/metric family 用 BH-FDR；对密集 vertex/edge 探索用 max-statistic permutation/FWE。含 nuisance 的 GLM permutation 应使用保持 exchangeability 的方法（如 Freedman–Lane），不是直接随意 shuffle labels。[Winkler et al. 2014, DOI 10.1016/j.neuroimage.2014.01.060](https://doi.org/10.1016/j.neuroimage.2014.01.060)
7. 面积比例是 composition（三个比例之和为 1），一个 system 缩小会机械性迫使另一个增大。应同时报告绝对 mm²、全海马总面积，并用 joint multivariate/compositional test 或一个预先指定的 posterior-vs-anterior contrast，避免把三个单独检验当独立发现。
8. interaction 通常功效低；在看到结果前用实际组大小、相关结构和可靠性做 simulation-based power/precision analysis。报告标准化效应与 CI，不以 nominal *p* 值排序“最佳发现”。

## 最低可发表方法包与停止规则

一个可信的首轮包应至少包含：

- 统一 BOLD/结构 QC、每组 QC 流失图、motion/clean-time 分布；
- 统一 `cross_dataset_rest_fc` branch manifest 下的 subject-level FC 与 group-atlas geometry
  输出；两类表保持数据类型分离，`kember_metrics` 仍是独立的 Kember endpoint package；
- Kember 仅预登记 `hcp_template_matching` 与 `hcp_fixed` 两种 method；
- 固定模板或 cross-fit 模板；
- fixed/template coverage、template similarity margin 与必要的可靠性；
- 问题 1 的 network × subregion omnibus model；
- Kember 的 source-512 functional geometry、native `surfareacorobl` area 与六个 contrast/432 method-aware endpoint rows；
- HC 加七个 cohort 的 group-atlas geometry 描述；group-wise 只做 7 个非 HC-versus-HC
  对比，不生成非 HC all-pairs；
- 四组 factorial effect 与单独、支持区间充分的 symptom × trauma model；
- 清楚定义的 test families 与 FDR/FWE；
- leave-one-subgroup/site/batch-out、equal-motion/equal-duration、medication sensitivity；
- 把 discovery、robustness 与 prospective validation 分开。

若个体标签 split-half reliability 低、某组 mapping failure 明显更高、interaction 在 moderator 范围没有共同支持，或结果只在一个任意阈值/一种 atlas 中出现，则停止机制性表述，降级为探索结果。只有跨分区定义、QC 敏感性和独立验证均保持方向与可解释效应量，才适合进入高水平论文主结论。

## 2026-09-01 三工作流规格更正

当前由 `cross_dataset_rest_fc` 统一编排 subject-level FC 与 group-atlas geometry
family；`external_group_atlas_analysis` 是兼容实现/报告入口，不是独立实验 workflow。
Kember 严格只有
`hcp_template_matching` 与 `hcp_fixed` 两种方法：functional segregation 和
boundary sharpness 保留 source-512 time series/labels/geometry，system surface area
使用经 `ADAP_BARY_AREA` 对齐的 native labels 与 native HippUnfold `surfareacorobl`。
六个 contrast 产生 432 个 method-aware endpoint rows。Group-specific atlas 仅为
统一 geometry family 的 group-level 描述产物，不是 Kember 的 subject-level 第三方法；
当前是 HC 加七个 cohort、64 个 cells，并只输出每个非 HC 组对 HC 的比较。
完整规范见 [docs/kember_area_method.md](docs/kember_area_method.md) 和 [docs/analysis_map.md](docs/analysis_map.md)。

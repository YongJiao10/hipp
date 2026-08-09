# HIPP 项目英文工作报告 Slides：重构制作计划

**日期：** 2026-07-17
**状态：** 仅完成制作计划；本轮不创建 PPTX、不改动分析代码或数据。
**面向对象：** PI 与实验室成员；同时作为新成员接手项目时的快速工作手册。
**幻灯片语言：** 英文。本文档保持中文，便于后续逐轮讨论和修订。

## 1. 本次重构后的定位

这套材料不是研究故事型 talk，也不是面向外部听众的宣传性 presentation。它应当是一份简洁的英文项目工作报告，回答四个实用问题：

1. 这个项目的分析流程由哪些功能模块组成，模块各自做什么；
2. 面对一个具体任务，应从哪一部分开始、会得到什么类型的结果；
3. 当前有哪些数据，哪些已经完成、哪些尚未启动；
4. 接下来最值得推进的科学问题、数据扩展和临床转化工作是什么。

主 deck 的顺序必须严格为：流程与代码结构 → 数据资源 → 当前进展与产品 → 下一步计划 → 接手索引。数据和流程不在同一张图中混讲。

正式英文页中不使用只有代码维护者能理解的内部简称。文件夹路径可以作为小号实现标签出现，但每次都配人类可读的功能名称。不要展示代码截图、命令行、manifest、provenance 或工程审计细节。

## 2. 已核查的项目结构与事实基础

以下结论来自当前活动工作区的文档、脚本目录、已生成产品和 refs；未浏览 _archive 中的任何内容。

### 2.1 当前工作流结构

活动分析入口主要位于：

- scripts/core_workflow：从 HippUnfold 表面、质量控制、皮层参照、海马-皮层功能连接，到群体地图与标准图件的主流程；
- scripts/individual_workflow：个体地图、留出式模板匹配、个体结果汇总；
- scripts/ArealMSHBM_Kong2022_Iowa_Stanford：Stanford 与 Iowa/ESTT 可复用的皮层参照相关流程；
- scripts/iowa7t：临床分支的下游图件、概览和质量检查辅助脚本；
- outputs_hcp、outputs_stanford、outputs_iowa_estt、outputs_iowa_jj_tms：相应分支的派生产物目录。outputs_iowa_estt 对应 ESTT 临床癫痫分支，outputs_iowa_jj_tms 对应 JJ_TMS 分支。

核心 workflow 的逻辑阶段为：

1. Hippocampal surface reconstruction and quality control
2. Group input preparation and cohort-level quality checks
3. Cortical reference construction
4. Hippocampus-to-cortex functional connectivity mapping
5. Group topology, gradients, and regional maps
6. Labels, summaries, and scientific figures
7. Presentation-ready overview products

这些阶段应该被转译为听众能理解的功能，而不是用脚本编号堆砌。

### 2.2 数据状态

- HCP 7T：当前主分析与模板建立数据集。活动质量控制队列为 151 名参与者，来自初始 166 名；已完成群体地图、个体分析和留出式模板稳定性评估。
- Stanford：已存在健康对照组和症状组的群体产物，文档记录的样本量分别为 91 和 103。它适合在主 deck 中作为外部队列工作分支，但不能过度表述为完整的跨队列验证，除非确认分析问题、匹配规则与比较结果。
- ESTT：数据位于 LSS。仓库中的 outputs_iowa / iowa7t 是其临床工作分支；正式分析尚未跑，属于近期计划。主 deck 中应使用 ESTT 或 clinical epilepsy cohort，不使用会让听众误解的数据目录名。
- Iowa 7T 癫痫病例：可作为 ESTT 临床起始病例说明，但主 deck 不显示可识别的受试者编号 0612。使用 current epilepsy case 或 initial clinical case，且仅在研究批准和去标识化范围内展示。
- HCP 3T：作为后续可推广的独立扫描条件，不应写成已经开始或已经具备结果。
- iEEG：未来治疗植入后可与影像地图衔接；本轮只写为有明确前置条件的长期计划，不能暗示临床可用性或既有数据。

### 2.3 当前 HCP 7T 关键结果

留出式模板稳定性评估覆盖 151 名参与者、5 个独立划分、左右海马及两种表面平滑水平，共 604 个参与者-半球-平滑组合。每个划分中，待评估参与者及其已知亲属均不参与为该参与者建立的模板。

四种设置的平均指标为：

| Surface setting | Mean region overlap | Mean partition agreement | Mean template-profile correlation |
|---|---:|---:|---:|
| 2 mm, left | 0.973 | 0.915 | 0.997 |
| 2 mm, right | 0.975 | 0.922 | 0.996 |
| 4 mm, left | 0.976 | 0.925 | 0.997 |
| 4 mm, right | 0.983 | 0.948 | 0.997 |

正式 slide 中需把指标写全：region overlap (Dice), partition agreement (adjusted Rand index), template-profile correlation。不要把内部统计文件名、gate 或 route 标签搬上去。

### 2.4 参考文献可导出的下一步方向

优先参考的来源及其可转化价值：

- Kember et al., 2026：沿海马长轴的个体化功能专化、发展变化与记忆关联；是个体地图加行为关联分析的最直接科学框架。
- Kong et al., 2019 和 2021：个体化皮层功能组织、可重复性、与行为的关联；可为皮层参照和独立验证设计提供方法学借鉴。
- DeKraker et al., 2025, HippoMaps：将折叠和展开海马空间与多模态 MRI、组织学和 iEEG 对齐；是未来临床电极定位和跨模态解释的关键框架。
- Lynch et al., 2026：precision mapping 的数据质量量化；适合用于外部数据转移前的质量分层和敏感性分析。
- Ezama et al., 2021：HCP 7T 中沿前后轴的网络偏好与海马亚区解释；用于定位解释和结果讨论。
- Lv et al., 2026：低维、动态功能模式与个体特质关联；适合列为远期动态功能连接扩展，而非当前主线。

## 3. 主 deck 总体规格

- 建议主 deck：17 页；另附 5–6 页 appendix。17 页适合覆盖大量工作内容，同时使主线仍可快速查阅。
- 画布：16:9，纯白背景。
- 主色：charcoal #202020；辅助灰 #6E6E6E 与 #D9D9D9；唯一强调色为 muted navy #1F4E79。
- 图像和科学图保留其原始色标；不要再加装饰性色块、渐变、图标阵列或大面积深色标题条。
- 字号：标题 30–34 pt；正文 15–17 pt；图注 11–12 pt。四周保留约 0.45 inch 空白。
- 每页只包含一个工作问题和一个可复用结论。主 deck 中表格不超过 6–7 行，详细矩阵移入 appendix。
- 所有 figure 均加一句英文 takeaway；不需要 speaker-note 式长段解释。
- 患者资料必须去标识化；不放受试者号、原始临床影像、植入计划或未获批准的临床细节。

## 4. 代码如何高效、美观地呈现在 Slides

不直接展示实际代码。使用三种互补图式即可让接手者快速定位：

1. 分层代码地图：用 4 个白底细边框模块展示目录层级，每个模块只有 “human-readable role + folder label + input/output”。例如 Core analysis workflow (scripts/core_workflow)。
2. 功能流水线：用 6–7 个顺序框展示数据如何经过重建、参照、连接、群体地图、个体地图和图件。框内写功能，不写脚本编号。
3. 任务到入口矩阵：用 4 行表回答 “我想做什么 → 从哪里开始 → 会得到什么”。这比展示几十个文件名更有用。

路径标签仅作为小号灰字帮助维护者定位；人类可读的功能短语应当是视觉主语。核心脚本模块的详细文件映射放在 appendix，而不放在主 deck。

## 5. 主 deck：逐页制作计划

### Slide 1 — Cover

**English title:** Hippocampal Functional Mapping Project
**English subtitle:** Workflow, Data Resources, Current Status, and Next Steps

**目的：** 明确这是内部项目工作报告和接手手册，不写故事性问题或宣传性 slogan。

**版式：** 左对齐标题；右下角仅放简洁的 hippocampal unfolded-surface 图或群体地图缩略图。白底，不要封面照片拼贴。

**页脚：** Lab / presenter / date。

### Slide 2 — Project Scope and Slide Guide

**English title:** Project Scope and Slide Guide

**On-slide English copy:**

- Workflow and code structure
- Data resources and their roles
- Completed analyses and available products
- Next analyses, data expansion, and clinical translation

**目的：** 提前说明流程在前、数据在后；帮助 PI 和实验室成员按需跳转。

**版式：** 四段等宽的细线导航条，编号 1–4。避免“research question”叙事图。

### Slide 3 — Core Analysis Workflow

**English title:** Core Analysis Workflow

**On-slide English copy:**

1. Hippocampal surface reconstruction and quality control
2. Cortical functional reference
3. Hippocampus-to-cortex connectivity profiles
4. Group topology and regional maps
5. Participant-specific maps
6. Quality-controlled figures and summaries

**目的：** 给出全项目最重要的功能流。此页不出现任何数据集名字，以免和数据部分混合。

**版式：** 自左向右的 6 步水平流程；在每个步骤下用一行小字写 input 或 output，例如 Surface coordinates / Cortical reference / Connectivity profile / Group map / Individual map / Figure set。

**建议图件：** 现有 HCP unfolding surface、皮层表面和海马地图各取一个裁剪缩略图，穿插在对应步骤，而不是新增装饰图。

### Slide 4 — Core Code Structure and Responsibilities

**English title:** Core Code Structure and Responsibilities

**On-slide English copy:**

| Project area | Main responsibility | Typical output |
|---|---|---|
| Core analysis workflow | Build surfaces, connectivity maps, group topology, and standard figures | Group maps and figure panels |
| Participant-specific workflow | Create individual maps and assess template stability | Individual labels and stability summaries |
| Cortical reference workflows | Create or reuse participant-specific cortical functional references | Cortical reference profiles |
| ESTT clinical workflow | Prepare clinical data products and clinical-case visual summaries | Clinical branch products |

**Implementation labels in small gray text:**

- scripts/core_workflow
- scripts/individual_workflow
- scripts/ArealMSHBM_Kong2022_Iowa_Stanford
- scripts/iowa7t and outputs_iowa

**目的：** 展示核心代码结构和每部分功能，但不展示源代码、命令、函数名或工程追踪文件。

**版式：** 4 条横向模块，每条左边是英文功能名，中间是一句职责，右边是输出。目录名只放在模块底部灰字。

### Slide 5 — Group-Level Mapping: What the Core Workflow Produces

**English title:** Group-Level Mapping: What the Core Workflow Produces

**On-slide English copy:**

Quality-controlled participants
→ hippocampal surfaces
→ cortical reference profiles
→ vertex-wise hippocampus-to-cortex connectivity
→ group connectivity gradients and three-region maps
→ standardized scientific figures

**Side panel:**

- Use for: template building, cohort summaries, external-cohort comparison
- Main products: gradients, regional labels, cortical connectivity panels, subfield summaries

**目的：** 把群体流程讲清楚，建立后面 HCP 7T 模板、Stanford 和 ESTT 的共同语言。

**版式：** 5 个细框加箭头；右侧为简短的 use/output 面板。只放一组真实的 group map + cortex panel，不塞完整图版。

### Slide 6 — Participant-Specific Mapping and Held-Out Template Stability

**English title:** Participant-Specific Mapping and Held-Out Template Stability

**On-slide English copy:**

- A group template is learned without the participant being evaluated.
- Known relatives are kept in the same validation split.
- The held-out participant is labeled by similarity to the independently learned template.
- Stability was evaluated across 151 participants, five independent splits, both hippocampi, and two smoothing levels.

**Bottom line:** 604 held-out participant–hemisphere–smoothing analyses.

**目的：** 将已完成的留出式评估用易懂语言讲清楚，避免 cross-fit、family-aware 等内部代称。

**版式：** 左侧三步小流程：Independent template → held-out participant → similarity-based regional map；右侧放现有稳定性 overview 的裁剪图。

### Slide 7 — How to Use the Project

**English title:** How to Use the Project

| If the goal is to… | Start with… | Main result |
|---|---|---|
| Build or update a cohort template | Core analysis workflow | Group topology and figure set |
| Generate participant maps | Participant-specific workflow | Individual regional maps and summaries |
| Compare an external cohort | Cortical reference and cohort workflow | Matched cohort products for comparison |
| Prepare a clinical case | ESTT clinical workflow | Quality-checked clinical visualization package |

**目的：** 这是 handbook 最重要的入口页。每个接手者都应能据此知道从哪里开始，而无需先读完整代码库。

**版式：** 三列表格；加一句小号提示：Read the workflow guide before changing an analysis branch. 不显示命令行。

### Slide 8 — Data Resources and Their Roles

**English title:** Data Resources and Their Roles

| Data resource | Role in the project | Current status |
|---|---|---|
| HCP 7T | Primary template-building and individual-mapping data set | Active and analysed |
| Stanford | External cohort workflow and group-level comparison branch | Group products available |
| ESTT clinical epilepsy cohort | Clinical translation branch; data are maintained on LSS | Workflow prepared; formal analysis not started |
| HCP 3T | Future generalization across routine-resolution acquisition | Planned |
| Future iEEG | Multimodal clinical and physiological validation after treatment implantation | Planned |

**目的：** 到这一页才介绍数据。清楚区分现有、可用产物和未来计划。

**版式：** 5 行简洁表格。可在 ESTT 行添加一枚无文字的淡灰 clinical-data 标记，但不要展示病例信息。

### Slide 9 — HCP 7T: Primary Template-Building Data Set

**English title:** HCP 7T: Primary Template-Building Data Set

**On-slide English copy:**

- Primary data set for group template construction and participant-specific mapping
- Initial cohort: 166 participants
- Active quality-controlled cohort: 151 participants
- High-resolution hippocampal surface representation
- Current analyses: group topology, participant-specific maps, and held-out template stability

**目的：** 讲清 HCP 7T 的中心地位以及已完成的核心范围。

**版式：** 左侧 166 → 151 的简洁 cohort flow；右侧放一张高质量 HCP group map 或 unfolded hippocampal surface。不要在此页列 QC 阈值和工程日志。

### Slide 10 — Stanford and ESTT: External and Clinical Data Branches

**English title:** Stanford and ESTT: External and Clinical Data Branches

**Left column — Stanford**

- External cohort branch with group products available
- Healthy-control group: n = 91
- Symptom group: n = 103
- Near-term use: matched group summaries and transferability assessment

**Right column — ESTT clinical epilepsy cohort**

- Data are stored and managed on LSS
- Workflow branch is prepared but formal analysis has not started
- Near-term use: establish a reproducible clinical-case workflow
- Long-term use: image-guided comparison with therapeutic iEEG coverage

**目的：** Stanford 和 ESTT 均放在数据 section，但必须清楚区别“已有群体产物”与“尚未正式运行”。

**版式：** 对称双栏。Stanford 使用已生成 overview 缩略图；ESTT 只用抽象的 hippocampus-to-electrode 示意图或空白流程占位，绝不展示患者身份。

### Slide 11 — Completed Work and Current Status

**English title:** Completed Work and Current Status

| Work package | Status | Evidence available |
|---|---|---|
| HCP 7T group mapping | Complete | Group topology, gradients, regional maps |
| HCP 7T participant-specific mapping | Complete | Individual maps and overview products |
| Held-out template stability assessment | Complete | Stability summary across four analysis settings |
| Stanford group workflow | Complete | Healthy-control and symptom-group products |
| ESTT clinical analysis | Not started | Data and workflow branch identified |
| HCP 3T and iEEG extension | Planned | Scientific and technical design pending |

**目的：** PI 能在一页看到哪些是真正完成、哪些是产品就绪、哪些只是计划。避免用“green/yellow/red dashboard”做花哨状态管理。

**版式：** 6 行表格，Complete / Not started / Planned 用文字和深浅灰区分，而不是颜色独自表达。

### Slide 12 — HCP 7T Template Stability Results

**English title:** HCP 7T Template Stability Results

**Lead sentence:**

Independent templates produced highly consistent participant-specific regional maps.

| Surface setting | Region overlap (Dice) | Partition agreement (adjusted Rand index) | Template-profile correlation |
|---|---:|---:|---:|
| 2 mm, left | 0.973 | 0.915 | 0.997 |
| 2 mm, right | 0.975 | 0.922 | 0.996 |
| 4 mm, left | 0.976 | 0.925 | 0.997 |
| 4 mm, right | 0.983 | 0.948 | 0.997 |

**Bottom note:** All four settings met the predefined stability criterion.

**目的：** 这是当前最具定量证据的一页。英文用完整指标名称，使 PI 和新成员不需要猜 ARI 或 Dice 的意思。

**版式：** 左侧为紧凑指标表；右侧为一个 held-out map / template similarity overview。图和表等高，留足空白。

### Slide 13 — Available Output Products

**English title:** Available Output Products

**On-slide English copy:**

- Group connectivity gradients and three-region maps
- Participant-specific regional maps
- Hippocampal–cortical connectivity panels
- Local cortical-context panels
- Subfield composition summaries
- Template-stability overviews

**目的：** 让接手者知道现成可以使用和检查的结果类型，而非看到目录后无从下手。

**版式：** 2 × 3 网格，直接使用 6 张真实、裁剪一致的现有产品；每图一行英文标签。所有小图保持白底并统一边框。

### Slide 14 — Next Analysis: Individual Topology and Behaviour

**English title:** Next Analysis: Individual Topology and Behaviour

**On-slide English copy:**

1. Define participant-level measures of hippocampal longitudinal-axis topology.
2. Select a small, pre-specified set of memory and visuospatial outcomes.
3. Test associations with appropriate demographic and imaging-quality covariates.
4. Confirm stability in independent data partitions and, when feasible, an external cohort.

**Scientific basis:** Individual specialization and memory-related posterior hippocampal connectivity motivate this direction.

**References on slide:** Kember et al., 2026; Kong et al., 2019, 2021.

**目的：** 把下一篇最直接的研究主线写成可执行、可检验的计划，而不是泛泛地说“做行为分析”。

**版式：** 左侧四步研究计划；右侧用 Kember 的概念示意或自有 AP map + behavior scatterplot placeholder。引用用小号 author-year，不占用正文空间。

### Slide 15 — Next Data Expansion: HCP 3T, ESTT, and iEEG

**English title:** Next Data Expansion: HCP 3T, ESTT, and iEEG

| Branch | Immediate work | Intended output |
|---|---|---|
| HCP 3T | Harmonize preprocessing, surface representation, and quality criteria | Generalization across acquisition resolution |
| ESTT | Run the clinical workflow on approved, de-identified cases | Reproducible clinical-case maps and QC summaries |
| iEEG | Link post-implant electrode locations to hippocampal coordinates after clinical availability | Multimodal validation of functional organization |

**Required safeguards:**

- Use de-identified clinical materials only.
- Separate research outputs from treatment decisions.
- Do not interpret an imaging map as a clinical recommendation without approved clinical validation.

**目的：** 把临床与多模态扩展写得具体、合规、不过度承诺。

**版式：** 上半页三行计划表；下半页淡灰边框 safeguard note。无需任何手术或患者照片。

### Slide 16 — Reference-Guided Research Directions

**English title:** Reference-Guided Research Directions

| Direction | Why it is promising | Key reference |
|---|---|---|
| Individual longitudinal-axis specialization and behaviour | Direct bridge from stable individual maps to memory-related function | Kember et al., 2026 |
| Cross-cohort generalization and data-quality sensitivity | Makes transfer claims interpretable rather than purely descriptive | Lynch et al., 2026; Kong et al., 2019 |
| Anatomically grounded multimodal mapping | Connects unfolded coordinates, subfields, MRI, and future iEEG | DeKraker et al., 2025 |
| Network and subfield interpretation | Provides an HCP 7T biological context for regional findings | Ezama et al., 2021 |
| Dynamic functional organization | Longer-term expansion beyond static connectivity maps | Lv et al., 2026 |

**目的：** 将 refs 中真正能产生独立研究线的方向收敛为五条，便于 PI 排优先级。不要声称任何一条必然高水平发表。

**版式：** 五行表格，每行一句可检验的意义，不放完整 bibliographic reference；完整版放 appendix。

### Slide 17 — Project Handover: Where to Find What

**English title:** Project Handover: Where to Find What

| Need | Read first | Project area |
|---|---|---|
| Understand the end-to-end workflow | Core workflow guide | scripts/core_workflow |
| Generate or inspect individual maps | Individual workflow guide | scripts/individual_workflow |
| Locate HCP 7T results | Output overview and presentation products | outputs_hcp |
| Locate Stanford group products | Stanford output overview | outputs_stanford |
| Start an approved ESTT analysis | Clinical workflow guide and LSS data access procedure | scripts/iowa7t, outputs_iowa, LSS |

**Closing line:** The next decision is to prioritize behaviour analysis, clinical workflow execution, or cross-acquisition generalization.

**目的：** 最后一页作为实用 handover 入口，不做带情绪的结束语。

**版式：** 紧凑三列表格。LSS 路径只写为 LSS data access procedure，不在主 deck 放绝对路径或临床目录细节。

## 6. Appendix 计划

### Appendix A1 — Detailed Code-to-Function Map

仅给接手者使用。按下列结构列出路径、功能、输入和主要输出：

- Core analysis workflow：surface reconstruction; cohort QC; cortical reference; connectivity; group topology; figures
- Participant-specific workflow：individual maps; independently learned template matching; summary overviews
- Cortical reference workflows：external cohort alignment and cortical reference construction
- ESTT clinical workflow：approved clinical data preparation and visualization support

不列每个 Python 文件名，除非后续讨论明确需要将此页升级为维护者级别索引。

### Appendix A2 — Detailed Group Analysis Flow

将 Slide 5 的流程扩成两行：

- Input and quality-control row
- Connectivity, topology, and figure-production row

每步最多一行功能描述与一个产物标签。可用真实 group outputs 作为小插图。

### Appendix A3 — Detailed Participant-Specific Analysis Flow

详细说明：

- independently learned template
- held-out participant map
- similarity-based regional assignment
- stability metrics
- individual visual summaries

这页可保留 “relatives remain in the same validation split” 的说明，但仍不使用内部英文简称。

### Appendix A4 — HCP 7T Cohort and Stability Details

包含：

- 166 initial participants to 151 active quality-controlled participants
- 5 independent validation splits
- left and right hippocampi
- 2 mm and 4 mm surface smoothing
- 四行完整稳定性指标表

不要放脚本日志、审计文件、内部 gate 名称或原始数据路径。

### Appendix A5 — Data Location and Clinical Handling

面向内部使用，内容：

- HCP 7T、Stanford、ESTT 的数据拥有者与受控访问说明
- ESTT 数据在 LSS；outputs_iowa 是 ESTT 对应的项目产物分支
- 临床病例、植入资料和 iEEG 的去标识化及访问审批要求
- 不把路径、病例号、账号或敏感元数据放在可外发版本

### Appendix A6 — Terminology and Reference List

提供两部分：

- slide wording dictionary
- 完整参考文献条目

推荐的正式 slide 用语：

| Avoid in slide text | Use instead |
|---|---|
| family-aware | relatives were kept in the same validation split |
| cross-fit | held-out template stability assessment |
| RouteA | participant-specific cortical reference |
| RouteB | shared cortical reference |
| ctxreg | local cortical-signal control |
| ctxdrop | local-neighbour exclusion control |
| gradient-spectral | connectivity-gradient clustering |
| WTA | network assignment from strongest positive connectivity |
| provenance manifest / audit | quality-controlled standardized outputs |

## 7. 图件与版式的具体取材方案

优先复用现有输出，不为了填版而生成装饰图：

| Slide | Preferred source | Crop/use |
|---|---|---|
| 1 | HCP unfolded hippocampal group map | 右下角单图，低饱和、无大标题覆盖 |
| 3 | surface / cortex / connectivity / group / individual 现有产品 | 6 个统一小缩略图 |
| 5 | group atlas 和 cortex panel | 1 张主图加 1 张侧图 |
| 6 | crossfit_template_matching overview | 只取能解释独立模板比较的区域 |
| 9 | HCP group map | 右侧大图 |
| 10 | Stanford overview | 左栏小图；ESTT 用概念图或空位 |
| 12 | robustness summary + held-out map | 指标表和图各半页 |
| 13 | existing present overviews | 2 × 3 产品网格 |
| 14 | existing AP group map | 与未来行为模型的占位图组合 |
| 15 | simple unfolded-coordinate-to-electrode schematic | 仅概念示意，不出现患者信息 |

若当前结果图中带有内部代码名称、过多文件路径、时间戳或不一致配色，制作 PPT 前先输出裁剪版。不要在 slide 内覆盖复杂遮挡框。

## 8. 事实表述与风险控制

1. HCP 7T 的稳定性结果可以陈述为 “highly consistent participant-specific regional maps”；它证明的是当前模板程序下的留出稳定性，不自动证明行为效度、临床效度或跨队列泛化。
2. Stanford 页面只能表述 group products available，除非后续核实有完整且预注册式的跨队列比较结果。
3. ESTT 页面必须清楚写 formal analysis has not started。数据在 LSS、项目中已有工作流分支，不等于已经得到临床结果。
4. HCP 3T 和 iEEG 必须使用 planned / future work。iEEG 只在植入、定位、研究审批和去标识化条件满足后进入分析。
5. 不在主 deck 中公开病例号 0612、绝对数据路径、内部集群路径、队列密钥或可识别临床信息。
6. 每张使用临床资料的 slide 在制作阶段需要由项目负责人确认其内部展示范围。

## 9. 实际制作前的素材清单

在用户确认这一框架后，再开始制作 PPTX，并先收集：

- HCP group topology / gradient 的最终展示版本；
- held-out template stability overview；
- 2–3 张代表性 individual maps；
- Stanford healthy-control 和 symptom-group overview；
- HCP 7T cohort flow 的最终数字与质量控制用语；
- 行为量表候选项及分析优先级；
- ESTT 可内部展示的、去标识化素材范围；
- iEEG 图能否仅使用概念示意，或已有获准的定位图；
- PI 希望将主 deck 保持为 17 页，还是将 appendix 中的 code map 前移为主 deck。

## 10. 建议的下一轮讨论顺序

1. 确认 17 页主 deck 是否合适，或希望压缩到 12–14 页；
2. 确认 Slide 14 的行为变量优先级和统计范围；
3. 确认 Stanford 应强调 external comparison 还是仅作为已建工作分支；
4. 确认 ESTT / Iowa 7T / future iEEG 在内部报告中的可展示边界；
5. 确认每张 slide 的现有图件来源和是否需要重新导出裁剪版；
6. 确认后才创建纯白学术风格的英文 PPTX。

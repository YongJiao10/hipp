# HIPP 项目英文技术手册 Slides：V2 重构计划

**日期：** 2026-07-18  
**状态：** 已完成代码、数据位置、原始图件与现有 deck 根因审计；本文件是下一版 deck 的唯一制作规格。  
**受众：** PI、lab members，以及未来接手该项目的维护者。  
**幻灯片语言：** 英文；本计划使用中文。  
**目标成品：** `hipp_project_technical_handbook_2026-07-18.pptx`。旧版工作报告保留，不覆盖、不删除。

## 1. 本次重构的结论

上一版将项目做成了概览型工作报告，而用户需要的是能让新成员真正使用大型代码库的技术手册。V2 不再以“项目故事”或成果展示为主线，而是按实际执行顺序说明：

1. `scripts/core_workflow/` 的主体目录和编号阶段；
2. 每个编号阶段的入口脚本、可选功能分支、关键输入、关键输出和适用任务；
3. `gradient-spectral`、`wta-top`、`ctxreg`、`ctxdrop`、Kong2022 ArealMSHBM、Lynch2024 FASTANS/PFM 的具体算法逻辑与运行入口；
4. 哪些脚本负责把计算结果变成可检查图件；
5. 所有数据的位置、当前状态和交接时应从哪里开始；
6. 已完成的 cross-fitted individual map 工作，以及接下来的行为、外部队列、ESTT 与 iEEG 计划。

正式英文页可使用真实的代码/分支标识，因为这是维护手册；但每个标识首次出现时必须并列完整的可读名称。例如：

| Code label | Formal English wording |
|---|---|
| `gradient-spectral` | Gradient-Spectral Clustering |
| `wta-top` | Winner-Take-All Network Assignment |
| `ctxreg` | Local Cortical-Signal Regression Control |
| `ctxdrop` | Local-Cortex Exclusion Control |
| `routea` | RouteA Individualized Cortical Reference |
| `routeb` | RouteB Shared Cortical Reference |

不讲 manifest、audit、provenance 的工程细节；只讲一个研究者/接手者需要运行、检查和解释的内容。

## 2. V1 问题的根因与 V2 约束

### 2.1 已确认的根因

1. V1 的生成脚本使用裁切函数从原始 PNG 中切取区域；因此会截掉原图边缘、色条、标题或内容。
2. V1 把同一张 `routea_ctxreg` 连续梯度图作为“通用海马图”反复放在封面、流程、结果、计划页；这不是该图实际能证明的内容。
3. 图件选择是按“视觉可填充性”而不是“算法—图源—结论”一一映射，导致图与页面主题不匹配。
4. 第 7 页后使用大量概括句，遗漏了项目交接最关键的脚本入口、参数分支、输出目录和算法细节。

### 2.2 不可妥协的制作约束

- 绝不裁切科学图；只按完整宽高比等比例缩放（contain），不使用 cover/crop。
- 同一张科学图在整套 deck 中最多使用一次。
- 每张科学图在制作前登记：slide ID、主题、精确源路径、图实际表示的内容、图注、是否完整显示。
- 每一张含图 slide 都必须能回答：“这张图是由哪个脚本产生的，它证明什么，又不证明什么？”
- 不使用随机配图、通用海马装饰图、未核查的概念图或重复缩略图。
- 不为填空间而降低信息密度。无图页面必须用可读的脚本树、输入/输出矩阵、算法步骤或最小命令卡承载信息。
- 所有临床材料保持去标识化；不展示 Iowa 病例号 0612、原始病人影像、植入计划或治疗建议。

## 3. 经核查的工作流与数据事实

### 3.1 核心工作流

活动公共入口是 `scripts/core_workflow/`，以编号定义顺序：

| 阶段 | 作用 | 主要编号脚本 |
|---|---|---|
| 01 | HippUnfold 与输出验证 | `01_01`, `01_02` |
| 02 | 群体输入、QC、Lynch NSI 队列 | `02_01`, `02_02`, `02_03` |
| 03 | 皮层参照 | `03_01`–`03_04` |
| 04 | 海马—皮层 FC 与局部皮层控制 | `04_00`–`04_03` |
| 05 | 连续梯度、群体 atlas、置信度、加权比较 | `05_01`–`05_05` |
| 06 | labels 与科学图件渲染 | `06_01`–`06_10` |
| 07 | 最终 group overview 索引 | `07_01` |

活跃主分支：

`kong2019`, `kong2022_routea`, `kong2022_routea_17net`, `kong2022_routea_ctxreg`, `kong2022_routea_ctxdrop`, `kong2022_routeb`, `lynch2024_20net`, `lynch2024_7net`。

### 3.2 关键算法事实

- `ctxreg`：对每个海马顶点，在每个 fMRI run 内，以 20 mm 内有效皮层顶点的平均信号作为自变量，拟合 `hippocampal signal ~ local cortical mean + intercept`，输出残差时间序列；不覆盖原始 RouteA。
- `ctxdrop`：不改变海马时间序列。对每个海马顶点和每个皮层 ROI，在 ROI 平均前排除 20 mm 内的有效皮层顶点，再计算 vertex-to-ROI FC；输出邻居数诊断。
- `gradient-spectral`：`05_01` 对群体平均 Fisher-z FC 特征做列 z-score，再以 BrainSpace `GradientMaps(kernel="pearson")` 计算并对齐 G1–G5；`05_02` 默认取已对齐的 G1–G3，在海马表面相邻顶点上以 shifted-cosine affinity `(1 + cosine similarity) / 2` 进行谱聚类。K=2–8 为诊断候选，K=3 是当前主输出。
- `wta-top`：每个海马顶点从正 FC 最强的 cortical ROI/component 得到网络投票；ROI-registry 分支的当前默认定义是 `normalized_top20_count`，即每个网络的 top-20 positive-ROI counts 除以该网络有限可用 ROI/component 的数量。
- Kong2022 RouteA：必须先由 Argon 的 ArealMSHBM backend 产生个体化皮层标签，再运行 `03_02` 建立 RouteA reference；RouteA 17-network 只是网络 taxonomy 变体，复用 RouteA FC/gradient。
- Lynch2024：`02_03` 用 NSI >= 0.43 建立 78 人队列；`03_04` 以 FASTANS Lynch priors、argmax hard labels、Workbench cleanup/dilation 和 component time series 生成 PFM；`03_03` 建 20-network reference 与 canonical7 merge。其维护下游为 `gradient-spectral` 与普通网络级 `wta-top`；Lynch WTA 使用直接 vertex-to-network Fisher-z argmax，不使用 ROI registry 或 `normalized_top20_count`。

### 3.3 数据访问位置与状态

| Data source | Local access location | Project role / status |
|---|---|---|
| HCP 7T fMRI | `/home/yojiao/LSS/jianglab/3_Data_Working/HCP_7T_Hippocampus/rsfmri_volume` | 主模板、群体 atlas、individual/cross-fitted maps；active |
| HCP 7T structural working input | `data/hippunfold_input/` | HippUnfold 的仓库工作输入 |
| Stanford | `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/DataOrg_PFM_36P` | 已有 HC 与 symptom group products |
| ESTT | `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_esTT/DataOrg_PFM_36P` | 计划中的临床癫痫队列；`outputs_iowa_estt/` 为其派生产物目录名 |
| JJ_TMS | `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS` | TMS 3T 队列；`outputs_iowa_jj_tms/` 为其派生产物目录名 |
| Iowa 7T clinical branch | `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P` | 独立临床工作分支；病例信息不放入 deck |
| HCP 3T time-series workspace | `/home/yojiao/hpchome/hcp_3t_timeseries`；HPC: `/Users/yojiao/hcp_3t_timeseries` | 数据工作区可访问，尚未接入 HIPP core workflow |
| Therapeutic iEEG | 尚无本项目已登记的分析数据根目录 | 未来在植入、定位、研究批准与去标识化完成后接入 |

## 4. V2 主 deck 架构（26 页）

V2 先完整讲流程与代码，再讲数据、进展和计划。页数故意高于常规报告，因为其承担 handbook 功能；每页应可以单独被以后接手者查询。

### Section A — Workflow map and entry points

| # | English title | 必须承载的技术信息 | 版式 |
|---:|---|---|---|
| 1 | HIPP Project Technical Handbook | 副标题：Core workflow, analysis branches, data locations, and current status；无装饰图 | 极简白底封面 |
| 2 | How to Use This Handbook | “Follow stages 01→07; choose a branch; run the numbered entry point; inspect the stated output.”；列出四个导航 section | 高密度导航条 |
| 3 | `scripts/core_workflow/`: Main Directory Structure | 完整显示顶层 numbered entrypoints 01–07；只展示 `README.md`、`01_*`…`07_*`，不展开 `lib/` | 左侧真实目录树；右侧 stage-to-purpose 表 |
| 4 | Active Analysis Branches and What They Change | 八条 branch 的 reference source、time-series control、maintained methods、典型使用场景；明确 RouteA/RouteB、17-net、ctxreg、ctxdrop、Lynch20/8 的差异 | 8 行矩阵 |

### Section B — Stage-by-stage core workflow

| # | English title | 入口与内容 | 版式 |
|---:|---|---|---|
| 5 | Stage 01 — HippUnfold Reconstruction and Validation | `01_01_run_hippunfold_t1w_local.py`：T1w HippUnfold；`01_02_check_hippunfold_outputs.py`：检查表面和输出。输入 `data/hippunfold_input/`，输出 `outputs_hcp/inputs/hcp/hippunfold/dense_corobl_batch/`。给出最小 run pattern 与“do not proceed before validation”。 | 左侧 input→output；右侧两张 script cards |
| 6 | Stage 02 — Shared Inputs, QC, and the Lynch NSI Cohort | `02_01`: surface/tSNR/template；`02_02`: 166→151 QC cohort；`02_03`: NSI >=0.43、n=78 Lynch cohort。把标准输入与输出路径放在表中。 | 三张 script cards + cohort flow |
| 7 | Stage 03A — Kong2022 Cortical References and ArealMSHBM | `03_01` shared reference；Argon ArealMSHBM label generation 是 RouteA prerequisite；`03_02` builds RouteA reference and 17-net summaries；`06_09` renders MSHBM cortical maps。写清“先运行哪一个、在哪里运行、输出在哪”。 | 一个 numbered run chain + exact output tree；不放 HCP subject ID 图 |
| 8 | Stage 03B — Lynch2024 FASTANS Precision Functional Mapping | `03_04`：priors → prior time courses → z-score/weights → argmax hard labels → Workbench cleanup/dilation → component time series；`03_03`: Lynch20 + canonical7 references。必须列出 priors 和 output roots，并给出两条实际可执行的命令卡。 | 公式/流程图 + short command cards + output tree |
| 9 | Stage 04 — Functional-Connectivity Products | `04_01`: Kong2022 and Lynch vertex-to-ROI/component FC；`04_02`: intrinsic hippocampal FC；`04_03`: Kong2019 FC；输出矩阵说明哪条 branch 能用哪类 FC。 | FC product matrix |
| 10 | Stage 04A — Local Cortical-Signal Regression Control (`ctxreg`) | 真实 OLS 逻辑、20 mm 邻居、按 run 回归、输出 residual hippocampal time series；明确 `04_00` 必须先于 controlled `04_01`/`04_02`，且不覆盖 RouteA。给出最小命令模式。 | 准确矢量算法图 + I/O cards |
| 11 | Stage 04B — Local-Cortex Exclusion Control (`ctxdrop`) | 海马时间序列不变；每个海马 vertex 在 ROI mean 前排除 20 mm 内 cortex vertices；`04_01 --ctxdrop`；仅维护 `gradient-spectral` 和 `wta-top`。给出诊断输出和 run card。 | 准确矢量算法图 + I/O cards |
| 12 | Stage 05A — Continuous Hippocampal Gradients | `05_01_compute_hippomaps_gradients.py` 的 inputs、Fisher-z group mean、feature z-score、BrainSpace Pearson gradient G1–G5、alignment、默认 G1–G3；给出 RouteA 示例命令。 | 左侧算法链；右侧完整原图 |
| 13 | Stage 05B — How to Run Gradient-Spectral Clustering | 明确顺序：先 `05_01`，再 `05_02 --methods gradient-spectral --gradient-components 1,2,3`；shifted-cosine surface affinity；K=2–8 diagnostics、K=3 primary；inputs/outputs/“use when”。 | 两条完整可读 command cards + compact algorithm steps |
| 14 | Stage 05C — How Winner-Take-All (`wta-top`) Is Calculated | top positive ROI/component votes → network counts → `normalized_top20_count` formula → vertex labels；与 gradient-spectral 的差别；`05_02 --methods wta-top` 和 `06_01` rendering。必须注明现有 `top20_count` figure 是该计算家族的已有 overview，不把它错误称作连续 gradient 或 cluster result。 | 公式 + run card |
| 15 | Stage 05D — Controlled, Taxonomy, and Comparison Branches | RouteA 17-net reuse；RouteB parallel reference；ctxreg/ctxdrop downstream scope；`05_04` confidence maps；`05_05` weighted Fisher-z comparison；`05_03` only optional fast rerun。 | branch-to-step matrix |
| 16 | Stage 06 — From Computation to Usable Figures | 显示 `06_01`–`06_10` 和各自的实际 visual output（overview, labels, cortex panels, gradients, confidence, ctxdrop, subfield, MSHBM, 400-ROI template）。`06_05` 已弃用，明确不作为主步骤。 | 10-row renderer index |
| 17 | Stage 06 Example — Gradient-Spectral Group Overview | `06_01` 输出的完整 RouteA gradient-spectral overview；右侧说明 K diagnostics, profiles, ambiguity, left/right 2/4mm surfaces；不能宣称该图表示 continuous gradients。 | 全图完整显示 + source/meaning panel |
| 18 | Stage 06 Example — Winner-Take-All Group Overview | 完整 `top20_count` WTA overview；说明 vertex assignment counts、20-bin long-axis profiles、surface labels；与第 14 页 WTA definition 一一对应。 | 全图完整显示 + source/meaning panel |
| 19 | Stage 06 Example — `ctxreg`/`ctxdrop` Geometry Diagnostic | 完整 local-cortex-neighbor count map；说明它是 20 mm 邻居数量诊断，服务于两个 local-cortex controls，不是 functional atlas。列出 `06_07` 命令入口。 | 全图完整显示 + meaning panel |
| 20 | Stage 06 Example — Hippocampus-to-Cortex Template Strengths | 完整 `06_10` gradient-spectral 400-ROI template-strength figure；解释 anterior/body/posterior group labels怎样回投至 400 cortical ROIs；WTA 仅使用 `normalized_top20_count`，canonical8/canonical7 只是同一 assignment 的 taxonomy views，family17 仅用于 gradient。 | 全图完整显示 + renderer I/O |
| 21 | Stage 07 — Final Overview Index and Where Outputs Live | `07_01_build_group_matrix.py`；`outputs_hcp/work/` vs `outputs_hcp/present/`；如何从完成 branch 找到 overview、labels、gradients、cortex panels。不要介绍 manifest。 | output location map + task lookup table |

### Section C — Individual workflow, data, and progress

| # | English title | 必须承载的技术信息 | 版式 |
|---:|---|---|---|
| 22 | Individual Mapping and Cross-Fitted Template Matching | `scripts/individual_workflow/` 与 core workflow 的关系；`individual_workflow/lib/routea_individual_gradients_wta.py`、`01_01_run_hcp_crossfit_template_matching.py`；五 folds、family kept together、K=3 AP labels、604 subject×hemisphere×smoothing combinations；给实际运行命令。 | workflow card + exact run cards |
| 23 | Completed Example — Five-Fold Template Robustness | 完整 crossfit robustness overview；说明 Macro-Dice、ARI、prototype correlation、vertex agreement；保持它仅为模板 robustness 而非 behavior/clinical validity。 | 全图完整显示 + concise interpretation |
| 24 | Data Locations and Dataset-Specific Entry Points | 表中列 HCP 7T、Stanford、ESTT、Iowa7T、HCP3T、future iEEG：local path、current state、dataset-specific entrypoint/branch。数据独立成节，不能和流程图混放。 | 6-row location matrix |
| 25 | Current Status and Immediate Handover Priorities | Complete：HCP group, gradient-spectral/WTA products, individual/crossfit; Available：Stanford products, Lynch branch; Pending：HCP3T integration, ESTT run, clinical iEEG. 每行有“next concrete action”。 | high-density status table |
| 26 | Planned Analyses and Publication-Oriented Directions | 1) Kember-guided individual longitudinal-axis topology + behavior; 2) Stanford/HCP3T transferability; 3) ESTT reproducible clinical branch; 4) future iEEG coordinate alignment. 每项写 data, analysis, validation, deliverable；不编造预测结果。 | 4-row plan matrix |

## 5. 需要呈现的最小命令卡

Command cards 是这套技术手册必要部分，但每页只放能启动对应任务的最小、真实命令；完整参数细节保留在现有 workflow docs。

### 5.1 Gradient-Spectral

```bash
/home/yojiao/.conda/envs/py314/bin/python \
  scripts/core_workflow/05_01_compute_hippomaps_gradients.py \
  --routes routea --smoothings 2mm 4mm --hemis L R \
  --n-components 5 --selected-components 1,2,3

/home/yojiao/.conda/envs/py314/bin/python \
  scripts/core_workflow/05_02_run_group_atlas.py \
  --route-tag routea --methods gradient-spectral --atlases kong2022 \
  --gradient-components 1,2,3 --k-min 2 --k-max 8 --default-k 3
```

Slide 13 必须在命令旁明确标注：在正式 run 中，`--shared-root`、`--reference-root`、`--surface-root`、`--out-root` 需与选择的 branch 对应；ctxreg/ctxdrop/Lynch 的完整 branch path 参见该页底部的 branch table，不通过错误的默认 RouteB root 偷跑。

### 5.2 Winner-Take-All

```bash
/home/yojiao/.conda/envs/py314/bin/python \
  scripts/core_workflow/05_02_run_group_atlas.py \
  --route-tag routea --methods wta-top --atlases kong2022 \
  --k-min 2 --k-max 8 --default-k 3

/home/yojiao/.conda/envs/py314/bin/python \
  scripts/core_workflow/06_01_build_group_overview.py \
  --group-root <route-specific-group-root> --methods wta-top \
  --assignment-variant normalized_top20_count
```

### 5.3 Lynch FASTANS/PFM

```bash
/home/yojiao/.conda/envs/py314/bin/python \
  scripts/core_workflow/03_04_run_lynch2024_fastans_pfm.py \
  --subjects-manifest outputs_hcp/inputs/hcp/lynch2024_fast_pfm_t1only/manifests/hcp_7t_lynch2024_nsi043_pass_78.txt \
  --wb-command /home/yojiao/FASTANS/workbench/bin_linux64/wb_command --jobs 6

/home/yojiao/.conda/envs/py314/bin/python \
  scripts/core_workflow/03_03_build_lynch2024_reference.py \
  --shared-root outputs_hcp/inputs/hcp/lynch2024_fast_pfm_t1only/shared_inputs/pfm_nsi043_78 \
  --subjects-manifest outputs_hcp/inputs/hcp/lynch2024_fast_pfm_t1only/manifests/hcp_7t_lynch2024_nsi043_pass_78.txt \
  --cortex-pfm-root outputs_hcp/work/cortex_pfm --jobs 8
```

### 5.4 Individual cross-fitted maps

```bash
/home/yojiao/.conda/envs/py314/bin/python \
  scripts/individual_workflow/01_01_run_hcp_crossfit_template_matching.py \
  --family-map data/HCP_YA_subjects_2026_04_09_16_01_42.csv --resume
```

## 6. Figure registry specification

制作开始前，先生成一份独立的 figure registry。下列内容是 V2 的固定初始注册表；任何新增图必须在写入 PPTX 前登记。

| Slide | Exact source | 图的真实含义 | Use rule |
|---:|---|---|---|
| 12 | `outputs_hcp/present/gradients/routea_ctxreg/4mm/hippomaps_gradients_routea_ctxreg_4mm_bilateral_g1g2g3.png` | RouteA ctxreg 4 mm continuous G1–G3, bilateral folded/unfolded surfaces | 完整显示一次；只用于 continuous gradients |
| 17 | `outputs_hcp/present/overviews/kong2022_gradient-spectral_group_overview_g1g2g3_routea_fisherz.png` | RouteA Kong2022 gradient-spectral overview with K diagnostics/profiles/surfaces | 完整显示一次；只用于 gradient-spectral |
| 18 | `outputs_hcp/present/overviews/wta_normalized/top20_count.png` | Existing RouteA top-20 positive-ROI count WTA overview | 完整显示一次；不把它称为 continuous gradient 或 cluster figure |
| 19 | `outputs_hcp/present/local_cortex_neighbors/kong2022_routea_local_cortex_neighbors_mean20mm.png` | Mean local cortical neighbor count within 20 mm | 完整显示一次；只用于 ctxreg/ctxdrop geometry diagnostic |
| 20 | `outputs_hcp/present/cortex_panels/cortex_kong2022_gradient-spectral_2mm4mm_template_strength_overview.png` | Gradient-spectral K=3 hippocampus-to-400-ROI Fisher-z template strengths | 完整显示一次；只用于 06_10 template renderer |
| 23 | `outputs_hcp/present/overviews/crossfit_template_matching/hcp_routea_crossfit_k3_robustness_overview.png` | Five-fold template robustness across smoothing/hemisphere settings | 完整显示一次；只用于 crossfit robustness |

刻意不使用 V1 的重复“通用海马图”策略。Kong2022 ArealMSHBM 与 Lynch PFM 页面先用精确算法流程、命令和 output tree；若后续需要加入 MSHBM/PFM 图，必须先得到去标识化/适合内部展示的独立全图，并在 registry 中登记。

## 7. 版式规范

- 16:9、纯白背景、黑/深灰正文、深蓝仅用于链接流程或当前选中 stage。
- 标题 28–32 pt；正文 15–18 pt；路径/命令 10–12 pt monospaced；图注 10–11 pt。
- 正文密度优先于空白装饰。每页至少包含一种可执行信息：entrypoint、branch choice、I/O、output location、算法步骤或检查点。
- 每个 stage page 的固定信息层级：**Purpose → Entry point(s) → Branch choice → Inputs → Outputs → Use when**。
- Full-figure pages使用 65–75% 页面宽度的无裁切图；旁侧只有三个文本字段：Produced by、What it shows、Do not infer。
- 不加无关照片、渐变色块、装饰 icon、封面海马 PNG 或重复 legend。

## 8. 制作与验收步骤

1. 以本计划建立独立 figure registry；检查精确路径存在、每一源图只登记一次、所有图按 contain 完整显示。
2. 创建新的 PPTX 生成脚本和新 deck 文件，不修改或删除 V1 文件/素材。
3. 先生成 HTML/static preview，逐页人工检查：
   - 科学图是否完整、可读、无边缘截断；
   - figure caption 是否准确描述图源；
   - command cards 是否可读且语义正确；
   - 从第 7 页之后是否仍保持同等技术密度。
4. 生成 PPTX 后渲染每一页，再进行第二次人工视觉审查。
5. 运行 deck 的 figure-use audit：每个 `.png` source 的使用次数必须为 1；若为 0 或 >1 均失败。
6. 最后交付时同时给出 deck、生成脚本、figure registry 和 rendered review PDF/PNG 索引。

## 9. 需要在下一版 deck 中避免的表述

- 不把 `gradient-spectral` 写成含糊的 “group topology”。
- 不把 `wta-top` 写成未解释的缩写或与 clustering 混为一谈。
- 不把 `ctxreg`/`ctxdrop` 当作仅仅不同名字；必须说明 residualization vs exclusion 的计算差异。
- 不把 `outputs_iowa_estt` 或 `outputs_iowa_jj_tms` 统称为模糊的 Iowa data set；应区别为 ESTT derived-output branch 与 JJ_TMS derived-output branch。
- 不把 HCP 3T 写成已经在 HIPP 流程中完成；应写为 data workspace available, pipeline integration pending。
- 不把 cross-fitted map robustness 写成 behavior validity、clinical validation 或 cross-cohort generalization。

## 10. 后续讨论可修改的项目

1. 是否要把 26 页主 deck 再拆成 main deck + compact technical appendix；
2. Stanford 与 Iowa7T/ESTT 的具体 launcher 页要放在 main deck 还是 appendix；
3. 是否在内部版本中允许加入一个去标识化的 MSHBM/PFM visual example；
4. Kember 2026 行为关联页的具体行为变量和统计设计；
5. HCP 3T 导入 HIPP core workflow 前，需要确认的 input layout 与 harmonization protocol。

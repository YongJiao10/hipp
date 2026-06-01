# Core Workflow 文件速览

这份文档只做简版汇总：每个文件分别写清楚“输入什么、做什么、输出什么”。  
范围只覆盖 `scripts/core_workflow/` 下的活跃源文件，不包含 `.ipynb_checkpoints/`。

## 入口与注册

| 文件 | 输入 | 做什么 | 输出 |
|---|---|---|---|
| `scripts/core_workflow/README.md` | 无 | 说明 core_workflow 的入口约定、运行边界和目录规范 | 工作流说明文档 |
| `scripts/core_workflow/step_registry.json` | 各编号步骤的元数据定义 | 记录步骤 ID、分支、输入和输出的机器可读注册表 | 供脚本和文档读取的步骤注册信息 |
| `tests/core_workflow/smoke_prepare_run.py` | 烟雾测试参数、分支信息、目标运行配置 | 生成独立的 smoke run 计划 | `outputs_core/smoke_runs/<run_id>/` 下的运行计划 |

## 主流程脚本

| 文件 | 输入 | 做什么 | 输出 |
|---|---|---|---|
| `scripts/core_workflow/01_01_run_hippunfold_t1w_local.py` | BIDS 结构像和 fMRI 输入、HippUnfold 配置 | 逐被试运行 HippUnfold | 被试级 HippUnfold 结果 |
| `scripts/core_workflow/01_02_check_hippunfold_outputs.py` | HippUnfold 输出 | 检查输出是否完整、是否满足规则 | 完整性报告和校验状态 |
| `scripts/core_workflow/02_01_prepare_group_inputs.py` | 已校验的 HippUnfold 输出、fMRI 体数据 | 准备共享 surface、tSNR 元数据和 group template | 共享输入根目录、模板、准备清单 |
| `scripts/core_workflow/02_02_build_qc_manifest.py` | HippUnfold Dice 表和 tSNR gate 结果 | 汇总 QC 通过和剔除名单 | QC 通过/剔除清单和 QC 表 |
| `scripts/core_workflow/03_01_build_kong2022_reference.py` | QC 通过的被试列表、cortex 资产 | 生成 Kong2022 共用 cortex reference | 每个被试的 Kong2022 reference 数组 |
| `scripts/core_workflow/03_02_build_kong2022_routea_reference.py` | RouteA 资产、QC 通过的被试列表 | 生成 RouteA 专用 cortex reference | RouteA 兼容的 reference 数组 |
| `scripts/core_workflow/04_00_materialize_cortex_regressed_surface_timeseries.py` | RouteA references, `surface_151`, subject cortex dtseries/surfaces | Builds the isolated RouteA cortex-control time-series root by run-wise local cortex regression | `shared_inputs/routea_ctxreg_151` controlled time series and diagnostics |
| `scripts/core_workflow/04_01_materialize_kong2022_roi_fc.py --ctxdrop` | RouteA references, `surface_151`, subject cortex dtseries/surfaces, RouteA individualized labels | Builds the isolated RouteA cortex-drop FC root by excluding local cortex vertices before ROI averaging | `shared_inputs/routea_ctxdrop_151` FC caches and ctxdrop diagnostics |
| `scripts/core_workflow/04_01_materialize_kong2022_roi_fc.py` | 平滑后的 hippocampal timeseries、cortex reference | 物化 Kong2022 的 vertex-to-ROI FC | ROI FC 缓存 |
| `scripts/core_workflow/04_02_materialize_intrinsic_fc.py` | 平滑后的 hippocampal timeseries | 物化 intrinsic FC | intrinsic FC 缓存 |
| `scripts/core_workflow/04_03_materialize_kong2019_fc.py` | Kong2019 共享输入、被试清单 | 物化 Kong2019 FC bank | Kong2019 / Kong2019_8net FC 缓存 |
| `scripts/core_workflow/05_01_compute_hippomaps_gradients.py` | Kong2022 FC 缓存、support mask、group average surface | 计算 HippoMaps 风格连续梯度，默认固定 BrainSpace `random_state=0`，并统一 component/sign alignment | aligned G1-G5 梯度、aligned lambda/EV fraction、group mean Fisher-z FC、alignment manifest |
| `scripts/core_workflow/05_02_run_group_atlas.py` | FC 缓存、group template、分支控制、可选 gradient 根目录 | 构建 merged-canonical group atlas，并做 K=2..10 诊断，最终固定 K=3 | group labels、K 汇总、assignment 状态、functional atlas labels |
| `scripts/core_workflow/05_03_run_spectral_fast_rerun.py` | 现有分支根目录、manifest、template | 在主 group atlas 之后做可选重跑编排 | 刷新的 spectral 输出和可继续用于展示的 group root |
| `scripts/core_workflow/06_01_build_group_overview.py` | `05_02` 的 group-atlas 运行产物 | 生成每种方法的 overview 图 | method overview PNG 和摘要 |
| `scripts/core_workflow/06_02_materialize_group_labels.py` | assignment status JSON | 物化 functional atlas label 数组 | `group_functional_atlas_labels.npy` |
| `scripts/core_workflow/06_03_build_cluster_cortex_panels.py` | group labels、FC 摘要、topology 资产 | 生成 cortex / hippocampus 汇总面板 | cluster 和 clusterless cortex panel 产品 |
| `scripts/core_workflow/06_04_render_hippomaps_gradient_products.py` | `05_01` 产生的梯度 manifest 和数组 | 渲染 HippoMaps 连续梯度产品，并默认以 `routea` 做跨分支展示符号对齐和共享色标 | 跨分支可比的 G1-G3 折叠/展开图、EV 图、端点 FC 对比图、manifest |
| `scripts/core_workflow/06_05_render_group_gradient_heatmaps.py` | group-gradient 产品 | 已废弃（retired）；仅输出弃用提示并退出 | 无（deprecated） |
| `scripts/core_workflow/06_06_render_confidence_maps.py` | `05_04` confidence arrays | 渲染 gradient-spectral silhouette confidence maps | folded/unfolded confidence PNGs |
| `scripts/core_workflow/06_07_render_ctxdrop_neighbor_counts.py` | `routea_ctxdrop_151` or `routea_ctxreg_151` `surface/local_cortex_neighbors/hemi_*/neighbor_counts.npy` | Renders group-average 20 mm local cortex-neighbor counts shared by ctxreg and ctxdrop | `outputs_core/present/local_cortex_neighbors/` arrays, TSV, hippocampal surface-overlay PNG, manifest |
| `scripts/core_workflow/07_01_build_group_matrix.py` | method overview PNG | 整理最终展示索引 | matrix manifest、README、刷新后的 overview 集合 |

## 共享库

| 文件 | 输入 | 做什么 | 输出 |
|---|---|---|---|
| `scripts/core_workflow/lib/branch_config.py` | 分支名、分支相关参数 | 解析分支配置 | 分支配置对象或字典 |
| `scripts/core_workflow/lib/extract_cortex_kong2022_roi_timeseries.py` | subject surface、time series、Kong2022 ROI 标签 | 提取 cortex ROI timeseries | ROI timeseries 缓存 |
| `scripts/core_workflow/lib/extract_cortex_kong2022_route_a_roi_timeseries.py` | RouteA 相关 surface、time series、ROI 标签 | 提取 RouteA 的 cortex ROI timeseries | RouteA ROI timeseries 缓存 |
| `scripts/core_workflow/lib/extract_cortex_merged_network_timeseries.py` | ROI / cortex timeseries、网络映射 | 汇总到 merged network timeseries | network timeseries 缓存 |
| `scripts/core_workflow/lib/fc_math.py` | FC 矩阵、数值处理参数 | 做 FC 归一化、汇总和相关数值运算 | 处理后的 FC 数学结果 |
| `scripts/core_workflow/lib/gifti_labels.py` | 标签定义 | 构造或写出 GIFTI label 表 | GIFTI label 文件或对象 |
| `scripts/core_workflow/lib/hipp_density_assets.py` | 目标 density、asset 根目录 | 解析可用的 hippocampal density 资源 | density 对应的资产路径和元数据 |
| `scripts/core_workflow/lib/hippomaps_adapter.py` | FC / 梯度数据、HippoMaps 参数 | 适配外部 HippoMaps 代码调用 | 梯度相关产物或适配结果 |
| `scripts/core_workflow/lib/io_utils.py` | 路径、文件内容、读写参数 | 封装通用 I/O 操作 | 读写结果和路径处理结果 |
| `scripts/core_workflow/lib/kong2022_label_utils.py` | Kong2022 label / network 定义 | 处理 Kong2022 标签映射 | 标签映射或注释结果 |
| `scripts/core_workflow/lib/output_layout.py` | workflow branch、step、产物信息 | 统一输出目录和文件布局 | 标准输出路径 |
| `scripts/core_workflow/lib/render_cortex_pfm_scene.py` | cortex PFM scene 参数 | 渲染 cortex / PFM scene | 面板图或 scene capture |
| `scripts/core_workflow/lib/surface_sampling.py` | surface、采样密度、采样参数 | 在 surface 上做采样 | 采样后的顶点或权重数据 |
| `scripts/core_workflow/lib/surface_utils.py` | surface 几何和辅助参数 | 做 surface 相关几何处理 | surface 变换结果或几何辅助量 |
| `scripts/core_workflow/lib/traceability.py` | step 元数据、输入输出 artifact、git 信息 | 生成标准 traceability manifest | 追踪清单 JSON |
| `scripts/core_workflow/lib/workbench.py` | `wb_command` 参数、线程和环境设置 | 统一封装 Workbench 调用 | Workbench 执行结果和日志 |
| `scripts/core_workflow/lib/workbench_grid.py` | grid / scene / 批处理参数 | 组织 Workbench 网格式批量执行 | 批量渲染或批量命令结果 |
| `scripts/core_workflow/lib/workbench_scene_batch.py` | scene batch 定义 | 批量渲染 Workbench scenes | scene 批处理产物 |
| `scripts/core_workflow/lib/workflow_utils.py` | 时间戳、git / file 元数据、通用参数 | 提供 workflow 通用辅助函数 | 通用元数据和 JSON 写出结果 |

## 说明

- 这里的“输入 / 做什么 / 输出”是为了快速浏览，不替代源码和 step registry。
- 如果你后面要，我也可以把这份内容再压成一版更短的“文件索引表”。

# Progress Log

## 2026-03-25
- 初始化会话，确认任务为项目级实现。
- 检查仓库现状：目录为空项目，仅有参考文献。
- 读取相关 skills：brainstorming, planning-with-files。
- 建立 `docs/`, `scripts/`, `config/`, `data/`, `manifests/`, `outputs/`, `logs/`。
- 编写项目文档、配置文件与核心脚本骨架。
- 成功连接远端 Mac，定位 `/Volumes/Elements/HCP-YA-2025`。
- 发现 `100610` 同时具备结构包与 7T resting-state volume。
- 下载官方 Schaefer400 7-network atlas。
- 完成 `100610` 的最小数据拷贝与 manifest 生成。
- 验证本机 `wb_command` 可用，但 `docker`/`hippunfold` 缺失。
- 用主控脚本确认当前本地输入为 `volume`，并按规则在方法切换前停止。
- 用户批准切换到 volume-based 分支，并同意继续安装最新可用工具链。
- 创建 `conda hippo` 环境；该项为历史阶段记录，当前支线执行口径已统一为 `hippunfold 2.0.0`（`hippo2`）。
- 安装 `HippoMaps 0.1.17`，发现顶层导入会因 Qt/VTK 崩溃，改为绕过顶层导入的实现路线。
- 发现本机 `wb_command` 同样因 Qt/processor 问题不可作为稳定依赖。
- 新增单一分析输入 staging、volume Schaefer 参考提取、WTA、surface-to-volume、seed FC 等脚本。
- 已成功提取 `400` 个 Schaefer parcel 时序与 `7` 个网络参考时序。
- HippUnfold 真正执行已启动并逐步排除缓存、wget、网络下载等环境问题。
- 从 `/Users/jy/Downloads/workbench_plotting_macbook` 找到关键线索，确认本机 `wb_command` 需通过 `arch -x86_64` 调用。
- 新增 `sample_hipp_surface_timeseries.py` 与 `combine_hemi_labels_to_bold.py`，用于 HippUnfold 后半程。
- 手动缓存 HippUnfold 所需的 nnUNet 模型、CITI168/upenn 模板与 multihist7 atlas，绕过在线下载阻塞。
- 为 HippUnfold 增加 `--use-conda` 路线，并将规则环境切到 `osx-64` 兼容模式以解决 `convert3d` 缺包问题。
- 将官方 `workbench` 规则环境补丁为最小可运行版本，直接复用本机经 `arch -x86_64` 调用的 `wb_command`。
- 已确认修补后的 Workbench 规则真实跑通，成功生成 `sub-100610_space-T2w_den-2mm_label-hipp_atlas-multihist7_subfields.dlabel.nii`。
- 后处理入口脚本已改为自动检测 folded surface 空间，并优先匹配本次实际产出的 `space-T2w`，避免后续因默认 `T1w` 命名而断开。
- 当前 HippUnfold 主流程已推进到 `run_inference`，正在对 `sub-100610` 右侧海马 corobl T2w 输入执行 nnUNet 推理。
- 已定位本地 macOS 下 `nnUNet_predict` 失败原因为 multiprocessing `spawn` 与 `nnunet.utilities.nd_softmax` 中 lambda 的 PicklingError。
- 新增 `scripts/patch_nnunet_compat.py`，对本地 Snakemake `nnUNet` 规则环境自动打补丁。
- 更新 `scripts/run_hippunfold_local.sh`：Darwin/macOS 下自动设置 `NNUNET_DISABLE_MULTIPROCESSING=1`，并应用本地兼容补丁；为未来 HPC/Linux 保留默认并行路径。
- 兼容补丁生效后，`run_inference` 已重新进入右侧海马 nnUNet 推理，当前不再出现原始 PicklingError，推理进程保持高 CPU 运行中。
- 检查 `/Users/jy/Downloads/hcp_seed_fc_pro.py` 后确认其“分块/低内存/HPC”思路可复用，但其接口仍写死为四个结构 seed，不适合当前功能 label 流程。
- 新增 `scripts/compute_seed_fc_hpc.py`，保留 HPC 分块相关计算方式，同时改为直接接受离散 `seed-labels` 图并按 `label 1..N` 自动输出多张 seed FC 图。
- 新增 `config/hipp_network_style.json`，统一 7-network 名称与颜色。
- 新增 `scripts/prepare_wta_workbench_assets.py`，可将左右海马 WTA `.npy` 转为 Workbench 标准 `label.gii` 与 `dlabel.nii`。
- 新增 `scripts/render_wta_unfolded_map.py`，直接渲染 unfolded 海马功能网络图，并在图中标注网络名称与总体比例。
- 已确认本机 `wb_command` 适合作为标准 label 资产与 scene/screenshot 管线，最终带文字注释的发布图则更适合由 Python/Matplotlib 直接输出。

## 2026-03-26
- 复盘 `102311` 失败链路，确认 nnUNet 5-fold 推理已完成，真正失败发生在 HippUnfold `postproc_boundary_vertices`。
- 验证右侧海马 4 个 `sdt.shape.gii` 全零，而对应 `sdt.nii.gz` 体数据并非全零，说明问题出在 surface-volume 映射而不是上游 source/sink mask 缺失。
- 定位 `space-corobl` 海马曲面相对 corobl 体仅在 x 方向整体偏移约 `30.6 mm`；按该偏移回推后，顶点重新全部落回体内，距离场采样恢复正常。
- 新增 `scripts/wb_volume_to_surface_fallback.py` 并扩展 `scripts/wb_command`：对 `-volume-to-surface-mapping` 的全零输出自动执行负-x 校正采样 fallback。
- 将 `102311` 右侧 4 个坏掉的 `sdt.shape.gii` 与一个依赖它的 unfolded surface 移到 `outputs/102311/hippunfold/_repair_backup_20260326/`，触发修复后重跑。
- 重新执行 `scripts/run_hippunfold_local.sh 102311 ...`，fallback 成功修复右侧与左侧 `shape.gii` 全零映射，`postproc_boundary_vertices` 两侧均通过。
- `102311` 的 HippUnfold 已于 `2026-03-26 10:39` 正常完成 `145/145` steps，关键 T2w 2mm surface、multihist7 subfield label、cropT2w volume summary 与 QC 图均已重新生成。
- 新增 `scripts/render_structural_unfolded_map.py`，可直接用 HippUnfold 的结构 subfield label 与 unfolded surface 输出结构 unfolded PNG。
- 用户更正需求后，新增 `scripts/render_native_surface_label_map.py`，基于 folded/native surface 的 PCA 投影统一渲染结构分区图与 WTA 功能分区图。
- 更新 `scripts/run_post_hippunfold_pipeline.py`：默认输出已从 unfolded 切换为 `sub-*_hipp_structural_native.png` 与 `sub-*_hipp_wta_native.png`，并把 WTA native 图的生成顺序前移到 volume 回投之前。
- 已为 `102311` 生成正式 folded/native 图：`outputs/102311/post_hippunfold_native/sub-102311_hipp_structural_native.png` 与 `.../sub-102311_hipp_wta_native.png`。
- 追查到 post pipeline 后续 `surface_labels_to_volume.py` 仍受 `hippomaps.utils.surface_to_volume` 的旧路径/空间假设限制：它要求 `sub-*/coords` 下存在 `space-T2w` Laplace coords，而当前 HippUnfold 实际 coords 位于 `work/sub-*/coords` 且空间名为 `corobl`，因此 volume 回投尚未完全恢复。

## 2026-03-27
- 用户最终确定 native/folded 发布图应锁定为 `Workbench scene + mirrored native surfaces + right-side enlarged legend`，不再在海马表面直接贴文字。
- 将用户手调确认的 Workbench 视角固化为 `config/wb_locked_native_view.scene`。
- 新增 `scripts/render_wb_scene_batch.py`，可直接从固定 Workbench scene 模板批量替换 subject 与 label 资源并离屏截图。
- 扩展 `scripts/prepare_wta_workbench_assets.py` 以支持 `space-corobl` folded/native surface，直接把 WTA `.npy` 转成 Workbench `label.gii/dlabel.nii`。
- 新增 `scripts/compose_wb_with_side_legend.py`，将 structural/WTA 分区名称与比例统一放到右侧大号图例面板。
- 新增正式入口 `scripts/render_locked_wb_views.py`，统一执行 `scene capture -> WTA assets -> structural/WTA legend composition`。
- 生成正式锁定版发布图：三位被试各两张，共六张，最终风格为 `wb_biglegend`。
- 新增文档 `docs/wb_locked_rendering.md`，明确官方绘图流程、固定视觉规则与 archive policy。

## 2026-04-01
- 用户明确要求：远端 archive 里已有 surface 原始数据，当前 workflow 从现在开始切回并锁定为 `surface-first / CIFTI-first`。
- 按仓库约定重新解析 `MPM619.local`，当前可用 IPv4 为 `192.168.0.183`，并改用 `ssh yojiao@192.168.0.183` 直连远端 Mac。
- 确认 `/Volumes/Elements/HCP-YA-2025/Resting State fMRI 7T Preprocessed Recommended archive` 中以 `*_Rest7TRecommended.zip` 保存 7T 功能 archive。
- 检查 `100610_Rest7TRecommended.zip` 与 `102311_Rest7TRecommended.zip` 后确认：archive 内确实包含聚合 `rfMRI_REST_7T_Atlas_1.6mm_MSMAll_hp2000_clean_rclean_tclean.dtseries.nii` 与逐 run `dtseries`。
- 复盘代码现状后确认：`run_hippomaps_pipeline.py` 只做了 `dtseries` 检测，但 `copy_hcp_minimal.py`、`stage_hippunfold_inputs.py`、`run_post_hippunfold_pipeline.py` 仍然是 volume-first，需要补齐 surface/CIFTI 主链。
- 已为 `100610`、`102311`、`102816` 三位被试补齐本地聚合 `dtseries`，三者在 `check_inputs` 下均被识别为 `functional_mode = cifti`。
- 新增 `scripts/formalize_structural_only.py`，用于验证本地 `dtseries`、归档旧 volume 功能结果，并只重建正式 structural 图。
- 新增 `docs/structural_only_formalization.md`，明确当前正式交付只保留结构 label 与 structural PNG。
- 已将旧 `post_dense_corobl` 与 `final_wb_locked/sub-*` 功能结果整体移到 `outputs/dense_corobl_batch/_archived_volume_functional/sub-*/`。
- 已重建新的 structural-only 正式输出：`outputs/dense_corobl_batch/final_structural_only/final/sub-100610/sub-100610_structural.png`、`sub-102311_structural.png`、`sub-102816_structural.png`。

## 2026-04-08
- 复核 `sub-100610_task-rest_run-concat.dtseries.nii` 后确认：CIFTI 中 cortex 为 surface grayordinates，而 hippocampus 为 volume grayordinates，不可直接当作 hippocampal surface fMRI。
- 将“完全不用单独 volume 数据”调查结论写入 `docs/experiments/hipp_functional_parcellation/2026-04-08_no_separate_volume_findings.md`，并删除失败实验分支/worktree `codex/surface-first-smoke`。
- 用户决定停止继续追逐 strict no-volume 路线，主线切回当前合法输入模型下的 `k_selection` 测试。
- 扩展 `scripts/copy_hcp_minimal.py` 与 `scripts/stage_hippunfold_inputs.py`，支持拉取并整理 `REST1..REST4` 的 per-run `dtseries + bold` 输入。
- 将 `scripts/experiments/hipp_functional_parcellation_network/run_subject.py` 改为 run-aware instability 版：
  - `K=2..10`
  - `run-pair` split
  - 输出 `run_metadata.json / per_k_summary.tsv / final_selection_log.json`
  - 最终 `K` 规则改为 `local minima -> 1-SE -> V_min/connectivity`
- `run_subject.py` 新增显式绝对阈值 `--v-min-count`，运行记录新增 `V_min_mode`，并在 `per_k_summary.tsv` 中保留最小 parcel 顶点数与阈值顶点数。
- `run_subject.py` 进一步改为“优先用显式 `run-1..4` 输入，否则从 `run-concat` 自动等分拆出 4 个 run”：
  - `dtseries` 缺失时可从 `run-concat.dtseries.nii` staging 出 4 个 `(900, 91282)` 的 run-wise `dtseries`
  - `bold` 缺失时可从 `run-concat_bold.nii.gz` staging 出 4 个 `(113, 136, 113, 900)` 的 run-wise `bold`
  - 共享 staging 目录位于 `_shared/sub-<subject>/runwise_inputs/`
- 更新 `scripts/experiments/hipp_functional_parcellation_network/summarize_outputs.py` 与 flow docs，使 overview 和文档语义同步到新 K 选择逻辑。
- 从远端 archive 补齐 `100610` 的 4 个 run 到本地 `data/hippunfold_input/sub-100610/func/`，并用其成功跑通 `100610 + lynch2024 + network-gradient` 本地 smoke。
- smoke 首次在 `V_min = 5%`、随后 `2.5%` 下均因最小 parcel 约束失败；最终以 exploratory `v_min_fraction = 0.01` 跑通完整闭环。
- 该 smoke 产生并确认过同款 overview：`hipp_functional_parcellation_network_overview.png`，以及 `k_selection_curves.png` 和 `network_probability_heatmaps.png`。
- smoke 结果摘要：
  - `2mm/L -> K=7`
  - `2mm/R -> K=4`
  - `4mm/L -> K=10`
  - `4mm/R -> K=6`
- 已将同一套 smoke 复制到 `100610 + lynch2024 + network-prob-cluster-nonneg`，并确认：
  - `v_min_count = 63` 会在 `2mm/L` 失败，因为局部极小值的最小 parcel 只有 `36` 顶点
  - exploratory `v_min_count = 36` 时可以跑通完整闭环并生成同款 overview
  - 对应结果为 `2mm/L -> K=7`, `2mm/R -> K=2`, `4mm/L -> K=6`, `4mm/R -> K=2`
- 已确认 `/Volumes/yojiao/HCP_7T_Hippocampus` 中的 `166` 是旧项目 `struct_complete`/`ashs_subjects` 子集，并将其固化为仓库 manifest：
  - `manifests/hcp_7t_hippocampus_struct_complete_166.txt`
  - `manifests/hcp_7t_hippocampus_struct_complete_166.json`
- 已按仓库规则删除 `/tmp/hipp_k_selection_smoke`、`/tmp/hipp_k_selection_smoke_prob_cluster` 以及对应的 Matplotlib 缓存目录，仅保留源码改动和结论记录。

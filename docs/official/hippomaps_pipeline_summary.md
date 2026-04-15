# HippoMaps 全流程汇总

## 1. 总览

这份文档总结当前项目的**正式流程**：如何从 `T1w + T2w + resting-state fMRI` 启动，经过 `HippUnfold（内部含 nnUNet） + hippocampal surface fMRI sampling + vertex-to-parcel FC + diffusion map gradients + 正式渲染`，最终得到海马的结构分区与连续功能组织图。

另有一条 `exploratory cortex ROI parcel` 支线，用于把 cortex PFM network 分割进一步拆成 surface connected-component parcels，并在 cortex 图上叠加 parcel 边界。该支线说明见：

- `docs/cortex_roi_parcels.md`

本文档只描述当前已经锁定的正式口径：

1. 结构链路以 `HippUnfold` 为核心。
2. 功能链路当前默认以 archive 中可用的 `HCP 7T CIFTI/dtseries` 作为皮层参考输入。
3. 正式分析结果以 **surface structural label + functional gradients** 为主。
4. 正式展示图片以锁定的 `Workbench native/folded` 渲染流程为主。
5. 历史上的探索性绘图、临时兼容脚本、失败兜底分支不作为主流程，只在最后单独说明。

从输入到输出的高层主链如下：

1. 从远端 HCP 数据中提取单被试最小必需文件。
2. 整理为 `HippUnfold` 可直接消费的 `sub-*/anat`、`sub-*/func` 输入树。
3. 用 `HippUnfold` 执行海马结构分割、表面重建、subfield 标注。
4. 从全脑体空间 rs-fMRI 中提取皮层参考 parcel 时序。
5. 将 rs-fMRI 映射到海马 folded/native surface，得到海马顶点时序。
6. 用海马顶点时序与皮层 parcel 时序计算 `vertex-to-parcel FC`。
7. 对海马 FC 模式做 `diffusion map embedding`，得到连续功能梯度。
8. 输出正式 structural 图和 `Gradient 1` 图。

## 2. 起始输入与数据整理

### 2.1 原始输入是什么

当前正式流程从 HCP 7T 单被试的最小输入出发，核心输入包含 4 类文件：

1. `T1w_acpc_dc_restore.nii.gz`（`0.7 mm isotropic`）
2. `T2w_acpc_dc_restore.nii.gz`（`0.7 mm isotropic`）
3. `rfMRI_REST_7T_hp2000_clean_rclean_tclean.nii.gz`（`1.6 mm isotropic`）
4. `rfMRI_REST_7T_brain_mask.nii.gz`（`1.6 mm isotropic`）

它们分别代表：

1. 结构像 T1w（`0.7 mm`），供结构定位与配准参考。
2. 结构像 T2w（`0.7 mm`），当前 `HippUnfold` 正式运行时直接作为主要模态输入。
3. 单被试 resting-state BOLD 4D 体数据（`1.6 mm`）。
4. 对应 BOLD 的脑掩膜（`1.6 mm`）。

这些文件的原始格式都是 `NIfTI-1 gzip`，即 `.nii.gz`。

当前正式流程中最关键的分辨率/密度如下：

```text
数据/空间                   分辨率或密度
T1w                         0.7 mm isotropic
T2w                         0.7 mm isotropic
rs-fMRI BOLD                1.6 mm isotropic
brain mask                  1.6 mm isotropic
Schaefer volume atlas       2.0 mm isotropic
海马 unfolded surface       den-2mm
海马 native/folded surface  分析密度固定为 2mm，展示为 folded/corobl 空间
```

### 2.2 Step 1: `copy_hcp_minimal.py`

这一步从远端 HCP 目录中抽取最小必需文件到本地工作区。

- 输入示例
  - 远端结构 zip 内成员：`100610/T1w/T1w_acpc_dc_restore.nii.gz`
  - 远端结构 zip 内成员：`100610/T1w/T2w_acpc_dc_restore.nii.gz`
  - 远端功能体数据：`100610/rfMRI_REST_7T_hp2000_clean_rclean_tclean.nii.gz`
  - 远端功能 mask：`100610/rfMRI_REST_7T_brain_mask.nii.gz`
- 输入内容
  - 已预处理的 HCP 结构像与 resting-state volume fMRI。
- 输入格式
  - `.nii.gz`
  - 结构像来自 zip 包内部成员路径，功能像为远端独立文件。
- 方法
  - 从远端盘最小化复制所需文件，不复制无关被试或冗余衍生物。
- 输出示例
  - `data/input_local/sub-100610/sub-100610_T1w_acpc_dc_restore.nii.gz`
  - `data/input_local/sub-100610/sub-100610_T2w_acpc_dc_restore.nii.gz`
  - `data/input_local/sub-100610/sub-100610_rfMRI_REST_7T_hp2000_clean_rclean_tclean.nii.gz`
  - `data/input_local/sub-100610/sub-100610_rfMRI_REST_7T_brain_mask.nii.gz`
  - `manifests/100610_minimal_manifest.json`
- 输出内容
  - 本地下游分析所需的最小结构/功能输入，以及一份 manifest。
- 输出格式
  - `.nii.gz`
  - `.json`
- 性质
  - `本地工程实现`

### 2.3 Step 2: `stage_hippunfold_inputs.py`

这一步把最小输入整理成 `HippUnfold` 能直接读取的输入目录结构。

- 输入示例
  - `data/input_local/sub-100610/sub-100610_T1w_acpc_dc_restore.nii.gz`
  - `data/input_local/sub-100610/sub-100610_T2w_acpc_dc_restore.nii.gz`
  - `data/input_local/sub-100610/sub-100610_rfMRI_REST_7T_hp2000_clean_rclean_tclean.nii.gz`
  - `data/input_local/sub-100610/sub-100610_rfMRI_REST_7T_brain_mask.nii.gz`
- 输入内容
  - 本地最小化结构像和功能像。
- 输入格式
  - `.nii.gz`
- 方法
  - 重命名并整理为 `data/hippunfold_input/sub-<id>/anat|func/...` 的 BIDS-like 树。
- 输出示例
  - `data/hippunfold_input/dataset_description.json`
  - `data/hippunfold_input/sub-100610/anat/sub-100610_T1w.nii.gz`
  - `data/hippunfold_input/sub-100610/anat/sub-100610_T2w.nii.gz`
  - `data/hippunfold_input/sub-100610/func/sub-100610_task-rest_run-concat_bold.nii.gz`
  - `data/hippunfold_input/sub-100610/func/sub-100610_task-rest_run-concat_desc-brain_mask.nii.gz`
- 输出内容
  - 供 `HippUnfold` 和后续功能链路共同使用的标准化输入树。
- 输出格式
  - `.nii.gz`
  - `.json`
- 性质
  - `本地工程实现`

### 2.4 主流程步骤总表

```text
Step   脚本/方法                              输入                                              输出                                                   格式                                 性质
1      copy_hcp_minimal.py                   远端 HCP T1w/T2w(0.7mm)/BOLD(1.6mm)/mask         本地最小化结构像、功能像、mask、manifest              .nii.gz, .json                       实现
2      stage_hippunfold_inputs.py            本地最小化 T1w/T2w(0.7mm)/BOLD(1.6mm)/mask       HippUnfold 可消费的 sub-*/anat, sub-*/func 树         .nii.gz, .json                       实现
3      HippUnfold CLI（内部含 nnUNet）       staged T1w/T2w(0.7mm)                             海马 surface(2mm density)、subfield labels、QC         .surf.gii, .label.gii, .tsv, 目录    论文+实现
4      extract_schaefer_volume_reference.py  BOLD(1.6mm) + mask(1.6mm) + Schaefer(2mm)        1.6mm BOLD 空间 parcel 时序与重采样 atlas             .npy, .json, .tsv, .nii.gz           实现
5      scripts/common/sample_hipp_surface_timeseries.py     BOLD(1.6mm) + hippocampal surfaces(2mm density)  海马顶点 BOLD 时序                                    .func.gii, .npy, .json               论文+实现
6      scripts/common/compute_fc_gradients.py               海马顶点时序(2mm density) + parcel 时序          vertex-to-parcel FC、diffusion-map gradients          .npy, .json                          论文对齐实现
7      重要兼容: wb_volume_to_surface_fallback.py  volume-to-surface mapping 结果          修复全零 surface 映射输出                             .shape.gii 等                        关键兼容
8      render_locked_wb_views.py             structural labels + gradient PNG                  正式 structural / gradient 图                         .png                                 实现
```

## 3. HippUnfold 结构链路

### 3.1 Step 3: `run_hippunfold_local.sh`

这是正式结构链路的入口脚本。它激活 `hippo` 环境并调用 `hippunfold` CLI。

`run_hippunfold_local.sh` 还会把外部二进制目录插到 `PATH` 前面；默认值是
`/Applications/ITK-SNAP.app/Contents/bin`。这一步是为了让 `greedy` 和
`c3d_affine_tool` 在 HippUnfold 运行时可见。

另外，`c3d` 本身不是这个外部 bundle 的一部分，而是 `hippo2` 里的
`conda-forge::c3d` 包。服务器侧如果换了安装布局，要同时满足这两类依赖：

1. `hippo2` 里能找到 `c3d`
2. `HIPPUNFOLD_EXTERNAL_BIN_DIR` 指向包含 `greedy` 和 `c3d_affine_tool` 的目录

- 输入示例
  - `data/hippunfold_input/sub-102311/anat/sub-102311_T1w.nii.gz`（`0.7 mm`）
  - `data/hippunfold_input/sub-102311/anat/sub-102311_T2w.nii.gz`（`0.7 mm`）
- 输入内容
  - 单被试的 T1w / T2w 结构像（`0.7 mm`）。
- 输入格式
  - `.nii.gz`
- 方法
  - 通过 `hippunfold <input> <output> participant --modality T2w --output-density 2mm ...` 调用 `HippUnfold`。
  - 当前正式结构主模态是 `T2w`。
  - `T1w` 仍作为结构输入的一部分保留在输入树中，便于需要时参与配准或派生步骤。
- 输出示例
  - `outputs/<subject>/hippunfold/sub-<id>/surf/sub-<id>_hemi-L_space-corobl_label-hipp_midthickness.surf.gii`
  - `outputs/<subject>/hippunfold/sub-<id>/surf/sub-<id>_hemi-R_space-corobl_label-hipp_midthickness.surf.gii`
  - `outputs/<subject>/hippunfold/sub-<id>/surf/sub-<id>_hemi-L_space-corobl_label-hipp_atlas-multihist7_subfields.label.gii`
  - `outputs/<subject>/hippunfold/sub-<id>/surf/sub-<id>_hemi-R_space-corobl_label-hipp_atlas-multihist7_subfields.label.gii`
  - `outputs/<subject>/hippunfold/sub-<id>/qc/...`
  - `outputs/<subject>/hippunfold/work/sub-<id>/...`
- 输出内容
  - 左右海马 folded/native surface
  - 结构 subfield surface labels
  - 质控文件
  - 中间坐标、配准、工作目录产物
- 输出格式
  - `.surf.gii`
  - `.label.gii`
  - `.shape.gii`
  - `.tsv`
  - 若干中间目录
- 性质
  - `论文/方法学核心步骤` + `本地工程实现`

这一步包含一个核心分辨率/表示转换：

- 输入是 `0.7 mm` 结构体数据
- `HippUnfold` 当前正式输出 density 固定为 `2 mm`
- 因此这里完成了“高分辨率结构体数据 -> 海马 2mm surface density 表示”的转换

在这一步里，`nnUNet` 并不是一个独立的下游步骤，而是 `HippUnfold` 内部的结构分割前端：

1. `run_hippunfold_local.sh` 调起 `hippunfold`
2. `hippunfold` 内部调用 `nnUNet` 进行海马分割推理
3. 分割结果继续进入 `HippUnfold` 的 surface 重建、边界处理和 subfield 标注

所以对 workflow 来说，更合理的理解是：

- `HippUnfold` 是这整个结构步骤
- `nnUNet` 是 `HippUnfold` 内部的一部分
- 它不单独构成一个功能步骤

### 3.2 HippUnfold 阶段的主要输出

结构链路最重要的正式输出有两类：

1. **surface 几何**
   - 例如 `sub-102311_hemi-L_space-corobl_label-hipp_midthickness.surf.gii`
   - 内容是海马表面的顶点与三角面
   - 格式是 `GIFTI surface`
2. **结构分区标签**
   - 例如 `sub-102311_hemi-L_space-corobl_label-hipp_atlas-multihist7_subfields.label.gii`
   - 内容是每个海马 surface vertex 的结构分区标签
   - 格式是 `GIFTI label`

结构 subfield label 到这一步已经是标准的 **surface label**，后续正式渲染直接可以使用它。

## 4. 功能参考网络链路

当前项目的正式皮层参考输入默认是 7T `dtseries`，因此正式参考时序路线恢复为 surface/CIFTI reference。

### Step 4: `extract_schaefer_volume_reference.py`

- 输入示例
  - `sub-102311_task-rest_run-concat_bold.nii.gz`（`1.6 mm`）
  - `sub-102311_task-rest_run-concat_desc-brain_mask.nii.gz`（`1.6 mm`）
  - `data/atlas/schaefer400/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz`（`2.0 mm`）
  - `data/atlas/schaefer400/Schaefer2018_400Parcels_7Networks_order.dlabel.nii`
- 输入内容
  - 单被试 4D resting-state BOLD
  - BOLD mask
  - Schaefer400 atlas 的 volume 与 label 定义
- 输入格式
  - `.nii.gz`
  - `.dlabel.nii`
- 方法
  - 将 Schaefer volume atlas 重采样到 BOLD 空间。
  - 提取 parcel-level voxel timeseries。
  - 进一步聚合成 7-network mean timeseries。
- 输出示例
  - `Schaefer400_2mm_in_bold_space.nii.gz`
  - `schaefer400_parcel_timeseries.npy`
  - `schaefer400_parcels.tsv`
  - `schaefer7_network_timeseries.npy`
  - `schaefer7_networks.json`
- 输出内容
  - 海马 vertex-to-parcel FC 所需的 `Schaefer400` parcel 时序。
- 输出格式
  - `.nii.gz`
  - `.npy`
  - `.tsv`
  - `.json`
- 性质
  - `本地工程实现`

这一步发生了明确的分辨率转换：

- Schaefer volume atlas 原始分辨率是 `2.0 mm`
- 目标 BOLD 空间分辨率是 `1.6 mm`
- 所以 atlas 会先从 `2.0 mm` 重采样到 `1.6 mm BOLD space`

这一步现在直接对齐论文偏好的 surface/CIFTI 路线，不再把 volume reference 当作默认主链。

## 5. 海马 surface 功能映射与梯度

### Step 5: `scripts/common/sample_hipp_surface_timeseries.py`

这一步把体空间 BOLD 信号采样到海马 folded/native surface 上。

- 输入示例
  - `sub-102311_task-rest_run-concat_bold.nii.gz`（`1.6 mm`）
  - `sub-102311_hemi-L_space-corobl_label-hipp_midthickness.surf.gii`（分析 density `2 mm`）
  - `sub-102311_hemi-R_space-corobl_label-hipp_midthickness.surf.gii`（分析 density `2 mm`）
- 输入内容
  - 4D BOLD 体数据
  - 左右海马 native/folded surface
- 输入格式
  - `.nii.gz`
  - `.surf.gii`
- 方法
  - 调用 Workbench `-volume-to-surface-mapping`
  - 将 BOLD 体信号映射到海马 surface 顶点
  - 生成可视化/兼容用的 `.func.gii`
  - 同时保存数值计算更方便的 `.npy`
- 输出示例
  - `sub-102311_hemi-L_space-corobl_den-2mm_label-hipp_bold.func.gii`
  - `sub-102311_hemi-L_space-corobl_den-2mm_label-hipp_bold.npy`
  - `surface_sampling_summary.json`
- 输出内容
  - 每个海马顶点的 BOLD 时间序列
- 输出格式
  - `.func.gii`
  - `.npy`
  - `.json`
- 性质
  - `论文/方法学核心步骤` + `本地工程实现`

这一步的表示转换是：

- 输入信号在体空间中是 `1.6 mm` voxel data
- 输出变成海马 `2 mm surface density` 上的 vertex timeseries

这里还要明确区分“方法学要求”和“当前实现命令”：

- 论文层面要求的是：**把 rs-fMRI 采样到海马 surface 上**
- 但这并不等于论文在工程上把 `Workbench -volume-to-surface-mapping` 规定成唯一命令
- 同时，这也不是我们脱离官方流程临时拼出来的额外步骤；在 `HippUnfold/HippoMaps` 的官方工程资源里，本来就使用了 Workbench 来完成这类 `volume -> surface` 映射
- 当前项目之所以显式采用这个命令，是因为手头的功能输入是 **体空间 volume rs-fMRI NIfTI**
- 在这种数据条件下，需要显式完成 `volume -> surface` 映射，所以这里沿用了官方工程中已有的 Workbench 实现路径

因此：

- “海马 surface 功能采样” 是论文核心步骤
- “具体通过 Workbench 的 `-volume-to-surface-mapping` 完成” 是当前 volume 路线下对官方工程实现的直接复用

### Step 6: 重要兼容步骤：`wb_volume_to_surface_fallback.py`

这一步不是论文中的标准方法步骤，但在当前工程流程里非常重要，因为它把 `volume-to-surface` 的兼容策略从“事后补救”改成了“映射前 QC + 条件纠偏”。

问题出在 `Workbench -volume-to-surface-mapping`：

1. 某些被试上，体数据本身不是空的
2. 但映射到 surface 后，中间输出可能变成全零
3. Workbench 依然返回成功码
4. 后续 boundary / label 传播步骤才在更后面崩溃

这类问题的根因是某些 surface 和 volume 在 world space 上错位，导致采样点全部落在目标体外，结果变成“全零 surface 映射”。

当前项目的兼容方案是：

1. 通过 `arch -x86_64 "/Applications/wb_view.app/Contents/usr/bin/wb_command"` 执行真正的 Workbench 命令
2. 在正式执行 `-volume-to-surface-mapping` 前，先做统一几何 QC
3. QC 会检查顶点入体比例、直接采样非零比例，以及采样值是否近似常数
4. 如果命中已知的负向 `x` 错位模式，就先做负向 `x` 纠偏，再执行正式映射
5. 如果 QC 异常但不符合已知纠偏模式，就直接报错停止，不再依赖映射后 fallback 覆盖结果

所以这一步应被理解为：

- 不是论文方法本身
- 而是对原始 Workbench 映射缺陷的关键工程兼容
- 没有它，某些被试会在后续步骤不可恢复地失败

### Step 7: `scripts/common/compute_fc_gradients.py`

这一步是当前正式主流程的功能核心步骤。

- 输入示例
  - `sub-102311_hemi-L_space-corobl_den-2mm_label-hipp_bold.npy`
  - `schaefer400_parcel_timeseries.npy`
- 输入内容
  - 海马顶点时序
  - 皮层 parcel 时序
- 输入格式
  - `.npy`
- 方法
  - 对海马和 parcel 时序做标准化
  - 计算 `vertex x parcel` 的 FC 模式
  - 对 FC 模式构图并做 `diffusion map embedding`
  - 输出连续功能梯度
- 输出示例
  - `hipp_vertex_to_parcel_fc.npy`
  - `hipp_fc_gradients.npy`
  - `hipp_fc_gradient1.npy`
- 输出内容
  - 每个海马 vertex 到 parcel 的 FC 模式
  - 海马连续功能梯度
- 输出格式
  - `.npy`
- 性质
  - `论文对齐实现` + `本地工程实现`

到这一步，正式功能结果已经产生，它本身就是连续标量结果，不再需要转成离散 label 资产。

## 6. label 的最终形态

这是整个流程里最容易混淆的问题之一，结论先写在前面：

1. **主流程先得到的是结构 surface label 与功能梯度。**
2. **volume label 不属于当前正式主流程。**

### 6.1 结构 label 是什么

结构 subfield label 在 `HippUnfold` 阶段就已经产生：

- 典型文件
  - `sub-102311_hemi-L_space-corobl_label-hipp_atlas-multihist7_subfields.label.gii`
- 含义
  - 每个海马 surface vertex 的结构亚区编号
- 形态
  - `surface label`
- 格式
  - `.label.gii`

### 6.2 正式功能结果是什么

当前正式主流程的功能结果是连续梯度，而不是离散 label。

1. 原始 FC 结果
   - `hipp_vertex_to_parcel_fc.npy`
2. 梯度结果
   - `hipp_fc_gradients.npy`
   - `hipp_fc_gradient1.npy`

因此：

- 原始计算结果是 `.npy`
- 正式主结果也是 `.npy` 连续标量
- 正式展示图是 `Gradient 1` PNG

### 6.4 Hippomaps 在哪里实际用到

workflow 里并不是只用了 `HippUnfold`。

更准确地说：

1. `HippUnfold`
   - 负责结构分割、surface 重建、subfield 标注
2. `Hippomaps`
   - 在当前正式主流程里主要作为方法学背景与资源来源，而不是主执行入口

### 6.5 最终 label 结论

```text
问题                         结论
结构分区 label 是什么         原生就是 surface label，格式通常为 .label.gii
功能结果最先是什么            vertex-to-parcel FC 与梯度数组
正式功能主结果是什么          连续梯度 .npy
最终分析主表示是什么           structural surface label + functional gradients
volume label 有没有            不属于当前正式主流程
```

## 7. 正式出图流程

这里需要把“分析链路”和“展示链路”分开看：

1. 分析链路负责得到结构 label 和功能梯度。
2. 展示链路负责把这些 label 渲染成正式图片。

### 7.1 分析阶段的预览图

[scripts/run_post_hippunfold_pipeline.py](../../scripts/run_post_hippunfold_pipeline.py) 现在会直接输出正式功能预览图：

- `sub-<id>_hipp_fc_gradient1_native.png`

### 7.2 正式锁定的 Workbench 出图链

正式出图固定走以下资产与脚本：

1. `config/wb_locked_native_view.scene`
2. `scripts/workbench/render_locked_wb_views.py`
3. `scripts/workbench/render_wb_scene_batch.py`
4. `scripts/workbench/compose_wb_with_side_legend.py`

这条链的核心思想是：

1. 用 Workbench scene 抓取结构图。
2. 结构图直接使用 `HippUnfold` 的结构 `.label.gii`。
3. 功能图直接使用 post pipeline 输出的 `Gradient 1` PNG。
4. 图上不直接贴结构 label 名字，而是通过后续合成生成图例面板。

### 7.3 Step 10: `render_locked_wb_views.py`

- 输入示例
  - `config/wb_locked_native_view.scene`
  - 结构 `.label.gii`
  - `sub-<id>_hipp_fc_gradient1_native.png`
  - `outputs/dense_corobl_batch/sub-<id>/hippunfold/...`
- 输入内容
  - 锁定视角模板
  - 结构标签
  - 功能梯度图
- 输入格式
  - `.scene`
  - `.label.gii`
  - `.png`
- 方法
  - 先调用 `render_wb_scene_batch.py` 用锁定 scene 抓取 structural 图
  - 再调用 `compose_wb_with_side_legend.py` 合成结构右侧图例
  - 功能侧直接使用 `Gradient 1` 图作为正式输出
- 输出示例
  - 中间图：`sub-102311_wb_structural_native.png`
  - 合成图：`sub-102311_wb_structural_biglegend.png`
  - 正式归档图：`outputs/dense_corobl_batch/final_wb_locked/sub-102311/sub-102311_structural.png`
  - 正式归档图：`outputs/dense_corobl_batch/final_wb_locked/sub-102311/sub-102311_gradient.png`
- 输出内容
  - 单被试正式 structural 图
  - 单被试正式 gradient 图
- 输出格式
  - `.png`
- 性质
  - `本地工程实现`

### 7.4 正式最终产物与格式总表

```text
类别         示例文件名                                                                  内容                                   主/辅   格式
结构 surface  sub-102311_hemi-L_space-corobl_label-hipp_midthickness.surf.gii           海马 folded/native surface 几何         主     .surf.gii
结构 label    sub-102311_hemi-L_space-corobl_label-hipp_atlas-multihist7_subfields.label.gii  海马结构亚区 surface label       主     .label.gii
功能时序      sub-102311_hemi-L_space-corobl_den-2mm_label-hipp_bold.npy                海马 vertex BOLD 时序                  主     .npy
FC 结果       hipp_vertex_to_parcel_fc.npy                                               海马 vertex 到 parcel 的 FC            主     .npy
梯度结果      hipp_fc_gradients.npy                                                      海马连续功能梯度                       主     .npy
主梯度图      sub-102311_hipp_fc_gradient1_native.png                                    单被试 Gradient 1 图                   主     .png
正式结构图    outputs/dense_corobl_batch/final_wb_locked/sub-102311/sub-102311_structural.png  正式 structural 图               主     .png
正式梯度图    outputs/dense_corobl_batch/final_wb_locked/sub-102311/sub-102311_gradient.png    正式 Gradient 1 图               主     .png
```

## 8. 论文步骤 vs 本地兼容兜底

### 8.1 哪些属于论文/方法学核心步骤

下面这些属于方法学核心思想，和论文主线是一致的：

1. 先基于结构像建立个体海马几何、surface 与 subfield。
2. 将 resting-state fMRI 采样到海马 surface。
3. 用海马 vertex 时序与皮层 parcel 时序计算相关。
4. 对 FC 模式做连续梯度分解。

也就是说，以下脚本承载的核心计算最接近论文步骤：

1. `HippUnfold` 结构分割与 surface/subfield 生成
2. `scripts/common/sample_hipp_surface_timeseries.py`
3. `scripts/common/compute_fc_gradients.py`

### 8.2 哪些属于本地工程实现或兼容兜底

下面这些主要是工程适配、平台兼容、数据形态兼容或展示流程锁定，并不是论文核心创新本身：

1. `copy_hcp_minimal.py`
2. `stage_hippunfold_inputs.py`
3. `extract_schaefer_volume_reference.py`
4. `render_locked_wb_views.py`
5. `render_wb_scene_batch.py`
6. `compose_wb_with_side_legend.py`

兼容/兜底部分主要包括：

1. `wb_volume_to_surface_fallback.py`
   - 对 `volume-to-surface-mapping` 做映射前几何 QC，并在命中已知负向 `x` 错位时先纠偏再映射。
2. 锁定版 Workbench scene 与右侧 legend 大字号版式
   - 属于展示流程工程规范，不属于方法学核心步骤。

### 8.3 集中对照表

```text
模块/步骤                                归类
HippUnfold（内部含 nnUNet）结构链         论文核心步骤 + 工程实现
海马 surface 功能采样                    论文核心步骤 + 工程实现
海马到皮层参考网络/parcel 的 FC          论文核心步骤 + 本地工程实现
diffusion map 功能梯度                   论文核心步骤 + 本地工程实现
最小化数据复制与 staging                 本地工程实现
surface/CIFTI Schaefer reference         本地工程实现
Workbench 锁定渲染与 legend              本地工程实现
volume-to-surface 映射前 QC 与条件纠偏   关键兼容修复
```

## 9. 关键问题直接回答

### `nnUNet` 在哪里用？

`nnUNet` 用在 `HippUnfold` 内部的**海马结构分割/推理阶段**。它是 `HippUnfold` 结构步骤内部的一部分，不单独作为一个后续步骤存在。它**不参与** FC 梯度计算。

### 最终得到的 `label` 是 surface 还是 volume？

主流程先得到的是 **surface structural label + functional gradients**。

1. 结构分区原生就是 `.label.gii`
2. 功能结果最先是 `.npy`
3. 功能正式主结果保持为连续梯度，不再强制转成 volume label

所以主分析表示是 **surface structural labels + FC gradients**。

### 正式输出是什么？

正式输出分两层：

1. **分析层**
   - 结构 `.label.gii`
   - FC `.npy`
   - gradients `.npy`
2. **展示层**
   - `Workbench` 生成的 `sub-<id>_structural.png`
   - 正式 `sub-<id>_gradient.png`

### 结构标签和功能梯度有什么区别？

1. 结构标签描述海马内部解剖亚区，来自 `HippUnfold` 结构链路。
2. 功能梯度描述海马 vertex-to-parcel FC 的连续组织，来自 rs-fMRI 功能链路。

### 为什么会同时出现 surface 和 volume 两种文件？

因为方法学主空间是海马 surface，而正式主流程现在停留在连续梯度表示。

## 10. 一句话版本

当前 HippoMaps 正式流程的核心是：

先用 `HippUnfold`（其内部包含 `nnUNet`）从 `T1w/T2w` 得到海马 folded/native surface 与结构 subfield surface label；再把体空间 rs-fMRI 通过官方工程里已有的 Workbench `volume -> surface` 路径采样到海马 surface，并在映射前做几何 QC 与条件纠偏；然后计算海马到 `Schaefer400` parcels 的 FC，并做 `diffusion map embedding` 得到连续功能梯度；最后输出正式结构图和 `Gradient 1` 图。

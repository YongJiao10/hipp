# HCP 7T 单被试 HippoMaps 项目文档

## 1. 目标

本项目面向单个 HCP 7T 被试，尽量贴近 DeKraker et al. 2025 的 HippoMaps/HippUnfold 思路，完成以下结果：

1. 海马 `WTA 7-network` 功能分区 label
2. 回投到单被试 `native BOLD space` 的海马 voxel label 图
3. 每个海马功能分区作为 seed 的全脑 voxel-wise FC 图

## 2. 与论文流程的对应关系

- **直接沿用**
  - 先以结构像建立个体海马几何与 unfolded/folded surface
  - 再把 rsfMRI 数据采样到海马 surface
  - 以海马顶点时序与皮层参考网络时序做 FC
  - 用 `winner-takes-all (WTA)` 生成每个海马 vertex 的离散网络归属
  - 需要时将 surface label 回投到体积空间

- **结合 HCP 7T 数据的本地适配**
  - 海马 surface density 固定为 `2mm`
  - 当前远端实际发现的是 `7T resting-state volume NIfTI`，尚未发现 `7T CIFTI/dtseries`
  - 因此皮层参考时序存在两条路线：
    - 论文偏好：`CIFTI/dtseries + Schaefer400 surface atlas`
    - 当前被试若仅有 volume：需切换为 `volume-based neocortical reference`
  - 用户已批准切换到 `volume-based neocortical reference`

## 3. 单被试最小输入清单

当前自动发现候选被试：`100610`

最小拷贝文件如下：

```text
Target   Path Type        Remote Source                                                                                         Notes
100610   structural T1w   /Volumes/Elements/HCP-YA-2025/Structural Preprocessed Recommended for 3T and 7T/100610_StructuralRecommended.zip::100610/T1w/T1w_acpc_dc_restore.nii.gz   HippUnfold 输入
100610   structural T2w   /Volumes/Elements/HCP-YA-2025/Structural Preprocessed Recommended for 3T and 7T/100610_StructuralRecommended.zip::100610/T1w/T2w_acpc_dc_restore.nii.gz   HippUnfold 输入
100610   rsfMRI concat    /Volumes/Elements/HCP-YA-2025/Resting State fMRI 7T Preprocessed Recommended/100610/rfMRI_REST_7T_hp2000_clean_rclean_tclean.nii.gz                       功能时序
100610   brain mask       /Volumes/Elements/HCP-YA-2025/Resting State fMRI 7T Preprocessed Recommended/100610/rfMRI_REST_7T_brain_mask.nii.gz                                     BOLD 掩膜
```

说明：
- 结构像不整包复制，而是只从远端 zip 中按成员路径提取 `T1w/T2w`。
- 当前没有复制多余被试、多余 run、扩展 QC 截图或无关衍生文件。

## 4. 坐标空间与变换链

目标变换链如下：

```text
T1w/T2w native
  -> HippUnfold 输入空间
  -> HippUnfold 输出 folded / unfolded hippocampal surfaces
  -> 将海马 surface 映射到 native BOLD / func space
  -> 从 concat rsfMRI 采样海马顶点时序
  -> 计算 WTA surface labels
  -> surface labels 回投到 native BOLD volume
  -> 每个海马功能分区提 seed -> whole-brain voxel-wise FC
```

## 5. 输入组织约定

- 项目只保留一套分析输入目录：`data/hippunfold_input`
- 该目录采用 **HippUnfold 可直接消费的 `sub-*/anat`, `sub-*/func` 组织形式**
- 这里的输入是 **HCP 已预处理结构像/功能像**，不是“未预处理原始 BIDS 数据”
- 为避免误解，项目后续不再把这套目录称为 `bids` 或 `raw`

当前单被试输入目录：

```text
data/hippunfold_input/
  dataset_description.json
  sub-100610/
    anat/
      sub-100610_T1w.nii.gz
      sub-100610_T2w.nii.gz
    func/
      sub-100610_task-rest_run-concat_bold.nii.gz
      sub-100610_task-rest_run-concat_desc-brain_mask.nii.gz
```

## 6. 依赖与版本锁定

- 分析口径：**当前机器实际可运行的 HippUnfold CLI (`1.5.2-pre.2`) + volume-based HippoMaps 实现**
- 海马 surface density：**`2mm`**
- 本地项目环境名：**`hippo`**
- 皮层 atlas 来源：**ThomasYeoLab/CBIG 官方 Schaefer400 7-network**

当前环境核查结果：

```text
Item         Status                Notes
remote ssh   OK                    已成功连接 192.168.0.113
remote data  OK                    已发现 HCP-YA-2025 与被试 100610
wb_command   OK                    /Applications/wb_view.app/Contents/usr/bin/wb_command，需通过 arch -x86_64 调用
docker       Missing               本机无 docker
hippunfold   Installed             hippo 环境中实际可运行 CLI 为 1.5.2-pre.2
conda pkg    Mismatch observed     conda 包索引显示 2.0.0，但实际安装 CLI 版本为 1.5.2-pre.2
7T CIFTI     Missing so far        当前仅发现 rsfMRI volume NIfTI
```

平台兼容说明：
- 当前本地 **macOS** 仅用于流程打通与脚本验证，不作为正式批量运行平台。
- 正式运行目标为 **HPC SGE / Linux**。
- 因此实现上区分两类执行语义：
  - `macOS local test`：对 `nnUNet` 自动启用单进程、低内存兼容路径，避免 `multiprocessing spawn` 引发的 PicklingError 与额外内存压力。
  - `HPC/Linux`：正式版以 **GPU** 为前提，保留默认并行语义，不使用本地测试机的兼容降级策略，也不单独维护 CPU 正式运行口径。

## 7. 必须停下来汇报的风险点

出现以下任一情况时必须先向用户汇报，不得擅自决定：

1. 远端外接盘路径变化或数据命名异常
2. `hippunfold` 实际可装版本与论文兼容基线不一致
3. 本机缺失 `docker` 或其他关键运行时，导致官方推荐流程无法直接执行
4. `wb_command`、`ANTs`、`hippomaps` 安装失败或版本不兼容
5. HippUnfold 输出空间与 BOLD 空间之间缺少稳定变换链
6. 磁盘空间不足或最小拷贝清单无法满足下游步骤

## 8. 当前实现状态

已完成：
- 本地项目骨架目录
- 远端自动发现逻辑
- 最小拷贝脚本
- Schaefer400 官方下载脚本
- 环境检查脚本
- pipeline 主控脚本骨架
- `volume-based` 分支已被用户批准
- volume-based Schaefer 参考时序提取
- 海马 surface 时序采样脚本
- 左右海马 label 合并并回采样到 BOLD 脚本

待继续实施：
- 完成 HippUnfold 模型下载并正式运行
- 基于 HippUnfold 输出执行海马 surface 采样、WTA、回投与 seed FC

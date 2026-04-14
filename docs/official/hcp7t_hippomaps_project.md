# HCP 7T 单被试 HippoMaps 项目文档

## 1. 目标

本项目面向单个 HCP 7T 被试，尽量贴近 DeKraker et al. 2025 的 HippoMaps/HippUnfold 思路，完成以下结果：

1. 海马 `vertex-to-parcel FC` 与连续功能梯度
2. 海马 `FC gradients` 连续功能图
3. 正式 structural / gradient 图

## 2. 与论文流程的对应关系

- **直接沿用**
  - 先以结构像建立个体海马几何与 unfolded/folded surface
  - 再把 rsfMRI 数据采样到海马 surface
  - 以海马顶点时序与皮层参考网络时序做 FC
  - 论文中重点展示的是 FC 模式及其连续组织/梯度

- **当前项目的派生实现**
  - 旧的 `WTA` 离散网络归属不再作为正式主流程
  - 正式主流程改成 `FC -> diffusion map gradients`

- **结合 HCP 7T 数据的本地适配**
  - 海马 surface density 固定为 `2mm`
  - 当前远端 archive 已确认存在 `7T CIFTI/dtseries`
  - 因此皮层参考时序默认走：`CIFTI/dtseries + Schaefer400 surface atlas`
  - 体空间 BOLD 仍保留，用于个体化海马 `volume -> surface` 采样

## 3. 单被试最小输入清单

当前自动发现候选被试：`100610`

最小拷贝文件如下：

```text
Target   Path Type        Remote Source                                                                                         Notes
100610   structural T1w   /Volumes/Elements/HCP-YA-2025/Structural Preprocessed Recommended for 3T and 7T/100610_StructuralRecommended.zip::100610/T1w/T1w_acpc_dc_restore.nii.gz   HippUnfold 输入
100610   structural T2w   /Volumes/Elements/HCP-YA-2025/Structural Preprocessed Recommended for 3T and 7T/100610_StructuralRecommended.zip::100610/T1w/T2w_acpc_dc_restore.nii.gz   HippUnfold 输入
100610   rsfMRI dtseries  /Volumes/Elements/HCP-YA-2025/Resting State fMRI 7T Preprocessed Recommended archive/100610_Rest7TRecommended.zip::100610/MNINonLinear/Results/rfMRI_REST_7T/rfMRI_REST_7T_Atlas_MSMAll_hp2000_clean_rclean_tclean.dtseries.nii  皮层 surface 参考时序
100610   rsfMRI concat    /Volumes/Elements/HCP-YA-2025/Resting State fMRI 7T Preprocessed Recommended archive/100610_Rest7TRecommended.zip::100610/MNINonLinear/Results/rfMRI_REST_7T/rfMRI_REST_7T_hp2000_clean_rclean_tclean.nii.gz                 海马采样源 volume
100610   brain mask       /Volumes/Elements/HCP-YA-2025/Resting State fMRI 7T Preprocessed Recommended archive/100610_Rest7TRecommended.zip::100610/MNINonLinear/Results/rfMRI_REST_7T/rfMRI_REST_7T_brain_mask.nii.gz                           BOLD 掩膜
```

说明：
- 结构像不整包复制，而是只从远端 zip 中按成员路径提取 `T1w/T2w`。
- 功能侧默认从 archive zip 中提取聚合 `dtseries`，并同时提取聚合 volume 与 mask 供海马 surface 采样使用。
- 当前没有复制多余被试、多余 run、扩展 QC 截图或无关衍生文件。

## 4. 坐标空间与变换链

目标变换链如下：

```text
T1w/T2w native
  -> HippUnfold 输入空间
  -> HippUnfold 输出 folded / unfolded hippocampal surfaces
  -> 将海马 surface 映射到 native BOLD / func space
  -> 从 concat rsfMRI 采样海马顶点时序
  -> 计算 vertex-to-parcel FC
  -> diffusion map gradients
  -> 输出正式 structural / gradient 图
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
      sub-100610_task-rest_run-concat.dtseries.nii
      sub-100610_task-rest_run-concat_bold.nii.gz
      sub-100610_task-rest_run-concat_desc-brain_mask.nii.gz
```

## 6. 依赖与版本锁定

- 分析口径：**HippUnfold 版本以 CLI 实测门禁为准；当前本机已验证 `hippo2` 环境可用 `2.0.0` 且支持 `--output_density 512`**
- 海马 surface density：**`2mm`**
- 本地项目环境名：**`py314`（通用 Python） + `hippo2`（HippUnfold CLI 专用）**
- 皮层 atlas 来源：**ThomasYeoLab/CBIG 官方 Schaefer400 7-network**
- HippUnfold 安装来源：**必须包含 `khanlab` 渠道并置于前位**
- `hippo2` 修复命令：`CONDA_SAFETY_CHECKS=disabled conda install -y -n hippo2 -c khanlab -c conda-forge -c bioconda khanlab::hippunfold=2.0.0=py_0 --force-reinstall`
- 说明：此前误判源于求解器落到 `bioconda::hippunfold-2.0.0-pyh7e72e81_0`，其 CLI/metadata 实体仍是 `1.5.2` 口径；不能只看 `conda list`。

当前环境核查结果：

```text
Item         Status                Notes
remote ssh   OK                    已成功连接 192.168.0.183
remote data  OK                    已发现 HCP-YA-2025 与被试 100610
wb_command   OK                    /Applications/wb_view.app/Contents/usr/bin/wb_command，需通过 arch -x86_64 调用
docker       Missing               本机无 docker
hippunfold   OK                    hippo2 环境 `--version = 2.0.0`（khanlab::hippunfold=2.0.0=py_0）
den-512 test Pass                  `--output_density 512 --help` 退出码 0，CLI 显示 `{native,512,2k,8k,18k}`
7T CIFTI     Available             archive zip 内已确认聚合与逐 run dtseries
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
- archive surface/CIFTI 数据已重新确认
- surface-first Schaefer 参考时序提取
- 海马 surface 时序采样脚本
- 梯度计算与正式 gradient 出图脚本

待继续实施：
- 完成 HippUnfold 模型下载并正式运行
- 基于 HippUnfold 输出执行海马 surface 采样、surface-based FC 梯度与正式 gradient 出图
